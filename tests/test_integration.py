# =============================================================================
# Integration Tests — Full Training Loops with Synthetic Data
# =============================================================================
"""
Integration tests for centralized and federated training loops.

These tests use a tiny DSCATNet variant with small synthetic datasets
to exercise the full training pipeline without requiring real data or
significant GPU resources. Marked @slow since they still take a few seconds.
"""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
from src.federated.simulation import ClientData, FLSimulator, SimulationConfig
from src.models.dscatnet import create_dscatnet


# =============================================================================
# Helpers
# =============================================================================

IMG_SIZE = 32
NUM_CLASSES = 3


def _make_tiny_model(device: str = "cpu") -> torch.nn.Module:
    """Create a tiny DSCATNet for fast tests."""
    return create_dscatnet(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        variant="tiny",
        pretrained=False,
        fine_patch_size=8,
        coarse_patch_size=16,
    )


def _make_synthetic_loader(n: int = 24, batch_size: int = 8, *, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader with random images and labels."""
    images = torch.randn(n, 3, IMG_SIZE, IMG_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# Centralized Integration Tests
# =============================================================================


@pytest.mark.slow
class TestCentralizedIntegration:
    """End-to-end centralized training with synthetic data."""

    def test_train_two_epochs(self, tmp_path):
        """Train for 2 epochs: loss should decrease or metrics should be recorded."""
        config = CentralizedConfig(
            model_variant="tiny",
            num_classes=NUM_CLASSES,
            pretrained=False,
            num_epochs=2,
            batch_size=8,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            image_size=IMG_SIZE,
            output_dir=str(tmp_path),
            experiment_name="integration_cent",
            early_stopping_patience=100,
            use_class_weights=False,
            num_workers=0,
            device="cpu",
            use_amp=False,
        )

        trainer = CentralizedTrainer(config)

        # Manually inject synthetic data loaders (bypass setup_data)
        trainer.train_loader = _make_synthetic_loader(24, batch_size=8)
        trainer.val_loader = _make_synthetic_loader(16, batch_size=8, shuffle=False)
        trainer.class_weights = None

        # Run training (this exercises train_epoch, evaluate, checkpointing)
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, 3):
            train_loss, train_acc = trainer.train_epoch(optimizer, criterion)
            val_loss, val_acc, _per_class = trainer.evaluate()

            trainer.history["epochs"].append(epoch)
            trainer.history["train_loss"].append(train_loss)
            trainer.history["train_accuracy"].append(train_acc)
            trainer.history["val_loss"].append(val_loss)
            trainer.history["val_accuracy"].append(val_acc)
            trainer.history["learning_rate"].append(config.learning_rate)

        # Verify history was recorded
        assert len(trainer.history["epochs"]) == 2
        assert all(isinstance(v, float) for v in trainer.history["train_loss"])
        assert all(0 <= v <= 1 for v in trainer.history["train_accuracy"])

    def test_checkpoint_save_load_resume(self, tmp_path):
        """Save checkpoint, load it in a new trainer, verify state is restored."""
        config = CentralizedConfig(
            model_variant="tiny",
            num_classes=NUM_CLASSES,
            pretrained=False,
            num_epochs=5,
            batch_size=8,
            image_size=IMG_SIZE,
            output_dir=str(tmp_path),
            experiment_name="ckpt_integration",
            num_workers=0,
            device="cpu",
            use_amp=False,
        )

        trainer = CentralizedTrainer(config)
        trainer.best_val_accuracy = 0.75
        trainer.best_epoch = 3
        trainer.history["epochs"] = [1, 2, 3]
        trainer.history["train_loss"] = [1.0, 0.8, 0.6]

        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)

        path = trainer.save_checkpoint(
            epoch=3, optimizer=optimizer, scheduler=None,
            metrics={"val_accuracy": 0.75}, is_best=True,
        )

        # Load into a fresh trainer
        trainer2 = CentralizedTrainer(config)
        optimizer2 = torch.optim.Adam(trainer2.model.parameters(), lr=1e-3)
        resumed_epoch = trainer2.load_checkpoint(path, optimizer2)

        assert resumed_epoch == 3
        assert trainer2.best_val_accuracy == 0.75
        assert trainer2.best_epoch == 3
        assert len(trainer2.history["epochs"]) == 3

    def test_full_run_writes_results(self, tmp_path):
        """CentralizedTrainer.run() should write results.json and config.json."""
        config = CentralizedConfig(
            model_variant="tiny",
            num_classes=NUM_CLASSES,
            pretrained=False,
            num_epochs=2,
            batch_size=8,
            gradient_accumulation_steps=1,
            image_size=IMG_SIZE,
            output_dir=str(tmp_path),
            experiment_name="full_run_test",
            early_stopping_patience=100,
            use_class_weights=False,
            num_workers=0,
            device="cpu",
            use_amp=False,
        )

        trainer = CentralizedTrainer(config)

        # Inject synthetic loaders
        trainer.train_loader = _make_synthetic_loader(24)
        trainer.val_loader = _make_synthetic_loader(16, shuffle=False)
        trainer.class_weights = None

        # Monkey-patch setup_data to skip dataset loading
        trainer.setup_data = lambda: None  # type: ignore[assignment]

        results = trainer.run()

        assert "best_val_accuracy" in results
        assert "total_time_seconds" in results
        assert "environment" in results
        assert results["environment"]["pytorch_version"] == torch.__version__

        results_file = tmp_path / "full_run_test" / "results.json"
        assert results_file.exists()
        saved = json.loads(results_file.read_text())
        assert "environment" in saved

        config_file = tmp_path / "full_run_test" / "config.json"
        assert config_file.exists()

    def test_test_split_evaluation(self, tmp_path):
        """When test_split > 0, results should include test_metrics."""
        config = CentralizedConfig(
            model_variant="tiny",
            num_classes=NUM_CLASSES,
            pretrained=False,
            num_epochs=2,
            batch_size=8,
            image_size=IMG_SIZE,
            test_split=0.15,
            output_dir=str(tmp_path),
            experiment_name="test_split_run",
            early_stopping_patience=100,
            use_class_weights=False,
            num_workers=0,
            device="cpu",
            use_amp=False,
        )

        trainer = CentralizedTrainer(config)

        # Inject synthetic loaders including test
        trainer.train_loader = _make_synthetic_loader(24)
        trainer.val_loader = _make_synthetic_loader(16, shuffle=False)
        trainer.test_loader = _make_synthetic_loader(16, shuffle=False)
        trainer.class_weights = None
        trainer.setup_data = lambda: None  # type: ignore[assignment]

        results = trainer.run()

        assert "test_metrics" in results
        tm = results["test_metrics"]
        assert "accuracy" in tm
        assert "balanced_accuracy" in tm
        assert "f1_macro" in tm
        assert 0 <= tm["accuracy"] <= 1


# =============================================================================
# Federated Integration Tests
# =============================================================================


@pytest.mark.slow
class TestFederatedIntegration:
    """End-to-end FL simulation with synthetic data."""

    def _build_simulator(self, tmp_path, name: str = "integration_fl") -> FLSimulator:
        """Create an FLSimulator with synthetic client data."""
        config = SimulationConfig(
            model_variant="tiny",
            num_classes=NUM_CLASSES,
            pretrained=False,
            num_clients=2,
            num_rounds=2,
            local_epochs=1,
            batch_size=8,
            gradient_accumulation_steps=1,
            image_size=IMG_SIZE,
            output_dir=str(tmp_path),
            experiment_name=name,
            early_stopping_patience=100,
            use_class_weights=False,
            num_workers=0,
            device="cpu",
        )

        simulator = FLSimulator(config)

        # Inject synthetic client data
        for cid in range(2):
            simulator.client_data[cid] = ClientData(
                client_id=cid,
                train_loader=_make_synthetic_loader(24),
                val_loader=_make_synthetic_loader(16, shuffle=False),
                num_train_samples=24,
                num_val_samples=16,
                class_distribution={0: 8, 1: 8, 2: 8},
                dataset_name=f"synthetic_{cid}",
            )

        return simulator

    def test_single_round(self, tmp_path):
        """Run a single FL round: should return metrics dict."""
        simulator = self._build_simulator(tmp_path)

        from src.models.dscatnet import get_model_parameters

        metrics = simulator.run_round(round_num=1)

        assert "train_loss" in metrics
        assert "val_accuracy" in metrics
        assert "communication_cost_mb" in metrics
        assert metrics["communication_cost_mb"] > 0
        assert metrics["clients_participated"] == 2

    def test_two_rounds(self, tmp_path):
        """Run 2 FL rounds: history should track both."""
        simulator = self._build_simulator(tmp_path)

        from src.models.dscatnet import get_model_parameters

        for rnd in range(1, 3):
            metrics = simulator.run_round(rnd)
            simulator.history["rounds"].append(rnd)
            simulator.history["train_loss"].append(metrics["train_loss"])
            simulator.history["train_accuracy"].append(metrics["train_accuracy"])
            simulator.history["val_loss"].append(metrics["val_loss"])
            simulator.history["val_accuracy"].append(metrics["val_accuracy"])
            simulator.history["communication_cost"].append(metrics["communication_cost_mb"])

        assert len(simulator.history["rounds"]) == 2
        assert all(c > 0 for c in simulator.history["communication_cost"])

    def test_full_run_writes_results(self, tmp_path):
        """FLSimulator.run() should write results.json with environment info."""
        config = SimulationConfig(
            model_variant="tiny",
            num_classes=NUM_CLASSES,
            pretrained=False,
            num_clients=2,
            num_rounds=2,
            local_epochs=1,
            batch_size=8,
            gradient_accumulation_steps=1,
            image_size=IMG_SIZE,
            output_dir=str(tmp_path),
            experiment_name="fl_full_run",
            early_stopping_patience=100,
            use_class_weights=False,
            num_workers=0,
            device="cpu",
        )

        simulator = FLSimulator(config)

        # Inject synthetic client data and skip setup_clients
        for cid in range(2):
            simulator.client_data[cid] = ClientData(
                client_id=cid,
                train_loader=_make_synthetic_loader(24),
                val_loader=_make_synthetic_loader(16, shuffle=False),
                num_train_samples=24,
                num_val_samples=16,
                class_distribution={0: 8, 1: 8, 2: 8},
                dataset_name=f"synthetic_{cid}",
            )

        simulator.setup_clients = lambda: None  # type: ignore[assignment]

        results = simulator.run()

        assert "best_val_accuracy" in results
        assert "total_communication_mb" in results
        assert results["total_communication_mb"] > 0
        assert "environment" in results
        assert results["environment"]["pytorch_version"] == torch.__version__

        results_file = tmp_path / "fl_full_run" / "results.json"
        assert results_file.exists()
        saved = json.loads(results_file.read_text())
        assert "environment" in saved

    def test_checkpoint_save_load(self, tmp_path):
        """Save and load FL checkpoint, verify state is restored."""
        simulator = self._build_simulator(tmp_path, name="fl_ckpt_test")
        simulator.best_val_accuracy = 0.65
        simulator.best_round = 1
        simulator.history["rounds"] = [1]
        simulator.history["train_loss"] = [0.9]

        metrics = {"val_accuracy": 0.65, "train_loss": 0.9}
        simulator.save_checkpoint(round_num=1, metrics=metrics)

        ckpt_path = tmp_path / "fl_ckpt_test" / "checkpoints" / "checkpoint_round_1.pt"
        assert ckpt_path.exists()

        # Load into fresh simulator
        simulator2 = self._build_simulator(tmp_path, name="fl_ckpt_test")
        resumed_round = simulator2.load_checkpoint(str(ckpt_path))

        assert resumed_round == 1
        assert simulator2.best_val_accuracy == 0.65
        assert simulator2.best_round == 1


# =============================================================================
# Environment Info Tests
# =============================================================================


class TestEnvironmentInfo:
    """Tests for collect_environment_info utility."""

    def test_basic_fields(self):
        """Environment info should contain python and pytorch versions."""
        from src.utils.helpers import collect_environment_info

        info = collect_environment_info()

        assert "python_version" in info
        assert "pytorch_version" in info
        assert "platform" in info
        assert "cuda_available" in info
        assert isinstance(info["cuda_available"], bool)

    def test_pytorch_version_matches(self):
        """Reported pytorch version should match actual."""
        from src.utils.helpers import collect_environment_info

        info = collect_environment_info()
        assert info["pytorch_version"] == torch.__version__
