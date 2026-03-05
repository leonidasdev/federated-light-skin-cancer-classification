# =============================================================================
# Tests for Federated Strategy and Server Modules
# =============================================================================
"""Tests for src.federated.strategy and src.federated.server."""

import numpy as np
import pytest
import torch

from src.federated.strategy import DSCATNetFedAvg, create_fedavg_strategy
from src.federated.server import FederatedServer


# =============================================================================
# DSCATNetFedAvg Strategy Tests
# =============================================================================


class TestDSCATNetFedAvg:
    """Tests for the custom FedAvg strategy."""

    def test_init_defaults(self):
        """Default initialization should set expected attributes."""
        strategy = DSCATNetFedAvg()
        assert strategy.save_path is None
        assert strategy.save_every == 10
        assert strategy.early_stopping_patience == 20
        assert strategy.min_delta == 0.001
        assert strategy.total_rounds == 100
        assert strategy.current_round == 0
        assert strategy.best_accuracy == 0.0
        assert strategy.patience_counter == 0
        assert not strategy.should_stop

    def test_init_custom_values(self, tmp_path):
        """Custom values should be stored."""
        strategy = DSCATNetFedAvg(
            save_path=str(tmp_path / "ckpts"),
            save_every=5,
            early_stopping_patience=10,
            min_delta=0.01,
            total_rounds=50,
        )
        assert strategy.save_path == tmp_path / "ckpts"
        assert strategy.save_every == 5
        assert strategy.early_stopping_patience == 10
        assert strategy.total_rounds == 50

    def test_init_creates_save_directory(self, tmp_path):
        """If save_path is given, the directory should be created."""
        save_dir = tmp_path / "deep" / "nested" / "ckpts"
        DSCATNetFedAvg(save_path=str(save_dir))
        assert save_dir.exists()

    def test_metrics_history_structure(self):
        """metrics_history should have the expected keys."""
        strategy = DSCATNetFedAvg()
        expected_keys = {
            "round", "train_loss", "train_accuracy",
            "val_loss", "val_accuracy", "num_clients", "client_metrics",
        }
        assert set(strategy.metrics_history.keys()) == expected_keys

    def test_get_history_returns_dict(self):
        """get_history() should return the metrics_history dict."""
        strategy = DSCATNetFedAvg()
        assert strategy.get_history() is strategy.metrics_history

    def test_save_checkpoint_noop_without_save_path(self):
        """_save_checkpoint should do nothing when save_path is None."""
        strategy = DSCATNetFedAvg(save_path=None)
        # Should not raise
        strategy._save_checkpoint(None, 1)

    def test_save_checkpoint_with_parameters(self, tmp_path):
        """_save_checkpoint should write npz files when given parameters."""
        from flwr.common import ndarrays_to_parameters

        strategy = DSCATNetFedAvg(save_path=str(tmp_path))
        params = ndarrays_to_parameters([np.ones((3, 3)), np.zeros((3,))])
        strategy._save_checkpoint(params, 5)

        assert (tmp_path / "model_round_5.npz").exists()
        assert (tmp_path / "metrics_round_5.npz").exists()

    def test_save_checkpoint_with_suffix(self, tmp_path):
        """Suffix should be appended to the filename."""
        from flwr.common import ndarrays_to_parameters

        strategy = DSCATNetFedAvg(save_path=str(tmp_path))
        params = ndarrays_to_parameters([np.ones((2, 2))])
        strategy._save_checkpoint(params, 1, suffix="best")

        assert (tmp_path / "model_round_1_best.npz").exists()


# =============================================================================
# create_fedavg_strategy Factory Tests
# =============================================================================


class TestCreateFedAvgStrategy:
    """Tests for the create_fedavg_strategy factory function."""

    def test_returns_dscatnet_fedavg(self):
        """Factory should return a DSCATNetFedAvg instance."""
        initial = [np.ones((4, 3)), np.zeros((3,))]
        strategy = create_fedavg_strategy(initial_parameters=initial)
        assert isinstance(strategy, DSCATNetFedAvg)

    def test_custom_client_counts(self):
        """Custom min_fit_clients should be respected."""
        initial = [np.ones((4, 3))]
        strategy = create_fedavg_strategy(
            initial_parameters=initial,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )
        assert strategy.min_fit_clients == 2

    def test_custom_fractions(self):
        """Participation fractions should be forwarded."""
        initial = [np.ones((4, 3))]
        strategy = create_fedavg_strategy(
            initial_parameters=initial,
            fraction_fit=0.5,
            fraction_evaluate=0.5,
        )
        assert strategy.fraction_fit == 0.5

    def test_save_path_forwarded(self, tmp_path):
        """save_path should be forwarded to the strategy."""
        initial = [np.ones((4, 3))]
        strategy = create_fedavg_strategy(
            initial_parameters=initial,
            save_path=str(tmp_path / "fl_ckpts"),
        )
        assert strategy.save_path == tmp_path / "fl_ckpts"


# =============================================================================
# FederatedServer Tests
# =============================================================================


class TestFederatedServer:
    """Tests for the FederatedServer wrapper."""

    @pytest.fixture
    def tiny_model(self):
        """Create a tiny DSCATNet for server tests."""
        from src.models.dscatnet import create_dscatnet
        return create_dscatnet(num_classes=7, variant="tiny")

    def test_init(self, tiny_model, tmp_path):
        """Server should initialize with model and config."""
        server = FederatedServer(
            model=tiny_model,
            num_rounds=5,
            checkpoint_dir=str(tmp_path),
        )
        assert server.num_rounds == 5

    def test_get_summary(self, tiny_model, tmp_path):
        """get_summary should return a dict with expected keys."""
        server = FederatedServer(
            model=tiny_model,
            num_rounds=5,
            checkpoint_dir=str(tmp_path),
        )
        summary = server.get_summary()
        assert isinstance(summary, dict)
        assert "total_rounds" in summary

    def test_save_and_load_checkpoint(self, tiny_model, tmp_path):
        """save_checkpoint + load_checkpoint should roundtrip."""
        server = FederatedServer(
            model=tiny_model,
            num_rounds=5,
            checkpoint_dir=str(tmp_path),
        )
        server.save_checkpoint(round_num=3)

        # Find saved file
        checkpoints = list(tmp_path.glob("*.pt"))
        assert len(checkpoints) >= 1

        loaded_round = server.load_checkpoint(str(checkpoints[0]))
        assert loaded_round == 3
