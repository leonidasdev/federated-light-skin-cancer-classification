# =============================================================================
# Tests for Federated Client Module
# =============================================================================
"""Tests for src.federated.client.SkinCancerClient."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.federated.client import SkinCancerClient
from src.models.dscatnet import create_dscatnet, get_model_parameters


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tiny_model():
    """Create a tiny DSCATNet for fast tests."""
    return create_dscatnet(num_classes=7, variant="tiny", pretrained=False, img_size=32)


@pytest.fixture
def synthetic_loaders():
    """Create synthetic train/val DataLoaders."""
    # 16 samples, 3 channels, 32x32 images, 7 classes
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 7, (16,))
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return train_loader, val_loader


@pytest.fixture
def client(tiny_model, synthetic_loaders):
    """Create a SkinCancerClient with synthetic data on CPU."""
    train_loader, val_loader = synthetic_loaders
    return SkinCancerClient(
        client_id=0,
        model=tiny_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        local_epochs=1,
        learning_rate=1e-3,
        use_amp=False,
    )


# =============================================================================
# Tests
# =============================================================================


class TestSkinCancerClientInit:
    """Tests for SkinCancerClient initialization."""

    def test_init_defaults(self, tiny_model, synthetic_loaders):
        train_loader, val_loader = synthetic_loaders
        c = SkinCancerClient(
            client_id=0,
            model=tiny_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            use_amp=False,
        )
        assert c.client_id == 0
        assert c.local_epochs == 1
        assert c.learning_rate == 1e-3
        assert c.scaler is None  # AMP off on CPU
        assert c.scheduler is None  # default scheduler_type="none"
        assert len(c.history["train_loss"]) == 0

    def test_init_with_class_weights(self, tiny_model, synthetic_loaders):
        train_loader, val_loader = synthetic_loaders
        weights = torch.ones(7)
        c = SkinCancerClient(
            client_id=1,
            model=tiny_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            class_weights=weights,
            use_amp=False,
        )
        assert c.criterion.weight is not None

    def test_init_cosine_scheduler(self, tiny_model, synthetic_loaders):
        train_loader, val_loader = synthetic_loaders
        c = SkinCancerClient(
            client_id=2,
            model=create_dscatnet(num_classes=7, variant="tiny", pretrained=False, img_size=32),
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            scheduler_type="cosine",
            scheduler_t_max=50,
            use_amp=False,
        )
        assert c.scheduler is not None


class TestSkinCancerClientParameters:
    """Tests for parameter get/set."""

    def test_get_parameters(self, client):
        params = client.get_parameters({})
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_set_parameters_roundtrip(self, client):
        original = client.get_parameters({})
        # Zero out
        zeros = [np.zeros_like(p) for p in original]
        client.set_parameters(zeros)
        after = client.get_parameters({})
        for p in after:
            np.testing.assert_array_equal(p, np.zeros_like(p))


class TestSkinCancerClientFit:
    """Tests for the fit() method."""

    def test_fit_returns_correct_shape(self, client):
        params = client.get_parameters({})
        updated, num_examples, metrics = client.fit(params, {"local_epochs": 1, "current_round": 1})
        assert isinstance(updated, list)
        assert len(updated) == len(params)
        assert num_examples == 16  # synthetic dataset size
        assert "train_loss" in metrics
        assert "train_accuracy" in metrics
        assert "client_id" in metrics
        assert metrics["client_id"] == 0

    def test_fit_updates_history(self, client):
        params = client.get_parameters({})
        client.fit(params, {})
        assert len(client.history["train_loss"]) == 1
        assert len(client.history["train_acc"]) == 1

    def test_fit_with_cosine_scheduler(self, tiny_model, synthetic_loaders):
        train_loader, val_loader = synthetic_loaders
        c = SkinCancerClient(
            client_id=0,
            model=create_dscatnet(num_classes=7, variant="tiny", pretrained=False, img_size=32),
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            scheduler_type="cosine",
            scheduler_t_max=10,
            use_amp=False,
        )
        params = c.get_parameters({})
        c.fit(params, {"current_round": 1})
        # Scheduler should have stepped — LR should be slightly reduced
        lr = c.optimizer.param_groups[0]["lr"]
        assert lr < 1e-3 or lr == pytest.approx(1e-3, abs=1e-6)

    def test_fit_multiple_rounds(self, client):
        params = client.get_parameters({})
        for r in range(3):
            params, _, _ = client.fit(params, {"current_round": r + 1})
        assert len(client.history["train_loss"]) == 3


class TestSkinCancerClientEvaluate:
    """Tests for the evaluate() method."""

    def test_evaluate_returns_correct_types(self, client):
        params = client.get_parameters({})
        loss, num_examples, metrics = client.evaluate(params, {})
        assert isinstance(loss, float)
        assert num_examples == 16
        assert "accuracy" in metrics
        assert "loss" in metrics
        assert "num_samples" in metrics
        assert "client_id" in metrics

    def test_evaluate_updates_history(self, client):
        params = client.get_parameters({})
        client.evaluate(params, {})
        assert len(client.history["val_loss"]) == 1
        assert len(client.history["val_acc"]) == 1

    def test_evaluate_per_class_metrics(self, client):
        params = client.get_parameters({})
        _, _, metrics = client.evaluate(params, {})
        # Should have at least one class_X_accuracy key
        class_keys = [k for k in metrics if k.startswith("class_")]
        assert len(class_keys) > 0


class TestSkinCancerClientTrainEpoch:
    """Tests for the _train_epoch() method."""

    def test_train_epoch_returns_loss_and_accuracy(self, client):
        loss, acc = client._train_epoch(epochs=1)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0.0 <= acc <= 1.0

    def test_train_epoch_multiple(self, client):
        loss, acc = client._train_epoch(epochs=2)
        assert isinstance(loss, float)
        assert isinstance(acc, float)


class TestSkinCancerClientEvaluateInternal:
    """Tests for the _evaluate() method."""

    def test_evaluate_internal_returns_tuple(self, client):
        loss, acc, metrics = client._evaluate()
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "loss" in metrics
        assert "num_samples" in metrics


class TestClientGetHistory:
    """Tests for get_history()."""

    def test_get_history_returns_dict(self, client):
        h = client.get_history()
        assert isinstance(h, dict)
        assert "train_loss" in h
        assert "val_loss" in h

    def test_get_history_after_training(self, client):
        params = client.get_parameters({})
        client.fit(params, {})
        client.evaluate(params, {})
        h = client.get_history()
        assert len(h["train_loss"]) == 1
        assert len(h["val_loss"]) == 1
