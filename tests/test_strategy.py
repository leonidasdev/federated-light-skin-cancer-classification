# =============================================================================
# Tests for Federated Strategy and Server Modules
# =============================================================================
"""Tests for src.federated.strategy and src.federated.server."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from src.federated.strategy import DSCATNetFedAvg, create_fedavg_strategy
from src.federated.server import FederatedServer, create_server


# =============================================================================
# Helpers
# =============================================================================


def _make_client_proxy(cid: str = "0") -> ClientProxy:
    proxy = MagicMock(spec=ClientProxy)
    proxy.cid = cid
    return proxy


def _make_fit_results(
    n_clients: int = 2,
    num_examples: int = 100,
) -> list[tuple[ClientProxy, FitRes]]:
    """Create mock FitRes results for aggregate_fit."""
    results = []
    for i in range(n_clients):
        proxy = _make_client_proxy(str(i))
        params = ndarrays_to_parameters([np.random.randn(4, 3).astype(np.float32)])
        fit_res = FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=params,
            num_examples=num_examples,
            metrics={
                "client_id": i,
                "train_loss": 0.5 - i * 0.1,
                "train_accuracy": 0.7 + i * 0.1,
                "round": 1,
                "learning_rate": 1e-3,
            },
        )
        results.append((proxy, fit_res))
    return results


def _make_eval_results(
    n_clients: int = 2,
    num_examples: int = 100,
    accuracy: float = 0.8,
) -> list[tuple[ClientProxy, EvaluateRes]]:
    """Create mock EvaluateRes results for aggregate_evaluate."""
    results = []
    for i in range(n_clients):
        proxy = _make_client_proxy(str(i))
        eval_res = EvaluateRes(
            status=Status(code=Code.OK, message="ok"),
            loss=0.3 + i * 0.05,
            num_examples=num_examples,
            metrics={
                "client_id": i,
                "accuracy": accuracy + i * 0.05,
                "num_samples": num_examples,
            },
        )
        results.append((proxy, eval_res))
    return results


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
        strategy = DSCATNetFedAvg(save_path=str(tmp_path))
        params = ndarrays_to_parameters([np.ones((3, 3)), np.zeros((3,))])
        strategy._save_checkpoint(params, 5)

        assert (tmp_path / "model_round_5.npz").exists()
        assert (tmp_path / "metrics_round_5.npz").exists()

    def test_save_checkpoint_with_suffix(self, tmp_path):
        """Suffix should be appended to the filename."""
        strategy = DSCATNetFedAvg(save_path=str(tmp_path))
        params = ndarrays_to_parameters([np.ones((2, 2))])
        strategy._save_checkpoint(params, 1, suffix="best")

        assert (tmp_path / "model_round_1_best.npz").exists()

    def test_aggregate_fit_computes_weighted_avg(self, tmp_path):
        """aggregate_fit should compute weighted average metrics."""
        strategy = DSCATNetFedAvg(
            save_path=str(tmp_path),
            save_every=100,  # don't trigger checkpoint
            initial_parameters=ndarrays_to_parameters(
                [np.zeros((4, 3), dtype=np.float32)]
            ),
        )

        results = _make_fit_results(n_clients=2, num_examples=100)
        agg_params, metrics = strategy.aggregate_fit(1, results, [])

        assert agg_params is not None
        assert "avg_train_loss" in metrics
        assert "avg_train_accuracy" in metrics
        assert metrics["num_clients_trained"] == 2
        assert metrics["num_failures"] == 0
        # History updated
        assert len(strategy.metrics_history["round"]) == 1
        assert len(strategy.metrics_history["train_loss"]) == 1

    def test_aggregate_fit_empty_results(self):
        """aggregate_fit with empty results should return None."""
        strategy = DSCATNetFedAvg()
        params, metrics = strategy.aggregate_fit(1, [], [])
        assert params is None
        assert metrics == {}

    def test_aggregate_fit_saves_checkpoint_on_interval(self, tmp_path):
        """aggregate_fit should save checkpoint when round % save_every == 0."""
        strategy = DSCATNetFedAvg(
            save_path=str(tmp_path),
            save_every=1,  # save every round
            initial_parameters=ndarrays_to_parameters(
                [np.zeros((4, 3), dtype=np.float32)]
            ),
        )
        results = _make_fit_results(n_clients=1)
        strategy.aggregate_fit(1, results, [])
        assert (tmp_path / "model_round_1.npz").exists()

    def test_aggregate_evaluate_computes_accuracy(self):
        """aggregate_evaluate should compute weighted average accuracy."""
        strategy = DSCATNetFedAvg(
            initial_parameters=ndarrays_to_parameters(
                [np.zeros((4, 3), dtype=np.float32)]
            ),
        )
        results = _make_eval_results(n_clients=2, accuracy=0.8)
        loss, metrics = strategy.aggregate_evaluate(1, results, [])

        assert loss is not None
        assert "avg_val_accuracy" in metrics
        assert "best_accuracy" in metrics
        assert len(strategy.metrics_history["val_accuracy"]) == 1

    def test_aggregate_evaluate_empty_results(self):
        """aggregate_evaluate with empty results returns None."""
        strategy = DSCATNetFedAvg()
        loss, metrics = strategy.aggregate_evaluate(1, [], [])
        assert loss is None
        assert metrics == {}

    def test_early_stopping_triggered(self):
        """When accuracy doesn't improve, patience should count up and trigger."""
        strategy = DSCATNetFedAvg(
            early_stopping_patience=2,
            min_delta=0.01,
            initial_parameters=ndarrays_to_parameters(
                [np.zeros((4, 3), dtype=np.float32)]
            ),
        )
        # Round 1: accuracy 0.8 → best
        strategy.aggregate_evaluate(1, _make_eval_results(accuracy=0.8), [])
        assert strategy.best_accuracy > 0
        assert strategy.patience_counter == 0

        # Round 2: accuracy 0.8 → no improvement
        strategy.aggregate_evaluate(2, _make_eval_results(accuracy=0.8), [])
        assert strategy.patience_counter == 1

        # Round 3: still no improvement → should_stop
        strategy.aggregate_evaluate(3, _make_eval_results(accuracy=0.8), [])
        assert strategy.should_stop

    def test_early_stopping_reset_on_improvement(self):
        """Patience counter should reset when accuracy improves."""
        strategy = DSCATNetFedAvg(
            early_stopping_patience=5,
            min_delta=0.001,
            initial_parameters=ndarrays_to_parameters(
                [np.zeros((4, 3), dtype=np.float32)]
            ),
        )
        strategy.aggregate_evaluate(1, _make_eval_results(accuracy=0.7), [])
        strategy.aggregate_evaluate(2, _make_eval_results(accuracy=0.7), [])
        assert strategy.patience_counter == 1

        # Improvement
        strategy.aggregate_evaluate(3, _make_eval_results(accuracy=0.95), [])
        assert strategy.patience_counter == 0

    def test_configure_fit_adds_round_info(self):
        """configure_fit should add current_round and total_rounds to config."""
        strategy = DSCATNetFedAvg(
            total_rounds=50,
            initial_parameters=ndarrays_to_parameters(
                [np.zeros((4, 3), dtype=np.float32)]
            ),
            min_fit_clients=1,
            min_available_clients=1,
        )
        # Create a mock client manager that returns enough clients
        client_manager = MagicMock()
        proxy = _make_client_proxy("0")
        client_manager.num_available.return_value = 1
        client_manager.sample.return_value = [proxy]

        params = ndarrays_to_parameters([np.zeros((4, 3), dtype=np.float32)])
        config = strategy.configure_fit(1, params, client_manager)

        assert strategy.current_round == 1
        assert config is not None
        assert len(config) == 1
        _, fit_ins = config[0]
        assert fit_ins.config["current_round"] == 1
        assert fit_ins.config["total_rounds"] == 50

    def test_early_stopping_saves_best_checkpoint(self, tmp_path):
        """When accuracy improves, best model checkpoint should be saved."""
        initial_params = ndarrays_to_parameters(
            [np.zeros((4, 3), dtype=np.float32)]
        )
        strategy = DSCATNetFedAvg(
            save_path=str(tmp_path),
            early_stopping_patience=10,
            min_delta=0.001,
            initial_parameters=initial_params,
        )
        # Simulate that the server has assigned parameters to strategy
        strategy.parameters = initial_params
        results = _make_eval_results(accuracy=0.9)
        strategy.aggregate_evaluate(1, results, [])
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

    def test_default_evaluate_metrics_aggregation(self):
        """Default aggregation fn should compute weighted average for numeric metrics."""
        initial = [np.ones((4, 3))]
        strategy = create_fedavg_strategy(initial_parameters=initial)
        # The evaluate_metrics_aggregation_fn is set
        assert strategy.evaluate_metrics_aggregation_fn is not None

    def test_default_aggregation_fn_computes_weighted_avg(self):
        """Default aggregation fn should produce correct weighted avg."""
        initial = [np.ones((4, 3))]
        strategy = create_fedavg_strategy(initial_parameters=initial)
        fn = strategy.evaluate_metrics_aggregation_fn
        # Each tuple is (num_examples, metrics_dict)
        metrics = [
            (100, {"accuracy": 0.8, "loss": 0.3, "client_id": 0}),
            (200, {"accuracy": 0.9, "loss": 0.2, "client_id": 1}),
        ]
        result = fn(metrics)
        # client_id should be skipped
        assert "client_id" not in result
        # Weighted avg: (0.8*100 + 0.9*200)/300 = 260/300
        assert abs(result["accuracy"] - 260 / 300) < 1e-6

    def test_default_aggregation_fn_empty_metrics(self):
        """Default aggregation fn should return empty dict for empty input."""
        initial = [np.ones((4, 3))]
        strategy = create_fedavg_strategy(initial_parameters=initial)
        fn = strategy.evaluate_metrics_aggregation_fn
        assert fn([]) == {}

    def test_custom_evaluate_metrics_aggregation(self):
        """Custom aggregation fn should be used when provided."""
        def custom_fn(metrics):
            return {"custom": 1.0}

        initial = [np.ones((4, 3))]
        strategy = create_fedavg_strategy(
            initial_parameters=initial,
            evaluate_metrics_aggregation_fn=custom_fn,
        )
        assert strategy.evaluate_metrics_aggregation_fn is custom_fn


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

    def test_init_creates_dirs(self, tiny_model, tmp_path):
        """Server should create checkpoint and log directories."""
        ckpt_dir = tmp_path / "ckpts"
        log_dir = tmp_path / "logs"
        server = FederatedServer(
            model=tiny_model,
            num_rounds=5,
            checkpoint_dir=str(ckpt_dir),
            log_dir=str(log_dir),
        )
        assert ckpt_dir.exists()
        assert log_dir.exists()

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

    def test_configure_returns_strategy(self, tiny_model, tmp_path):
        """configure() should return a DSCATNetFedAvg strategy."""
        server = FederatedServer(
            model=tiny_model,
            num_rounds=5,
            checkpoint_dir=str(tmp_path),
        )
        strategy = server.configure(min_clients=2, fraction_fit=0.5)
        assert isinstance(strategy, DSCATNetFedAvg)

    def test_aggregate_metrics_weighted(self, tiny_model, tmp_path):
        """_aggregate_metrics should compute weighted average."""
        server = FederatedServer(
            model=tiny_model,
            num_rounds=5,
            checkpoint_dir=str(tmp_path),
        )
        metrics = [
            (100, {"accuracy": 0.8, "loss": 0.3}),
            (200, {"accuracy": 0.9, "loss": 0.2}),
        ]
        result = server._aggregate_metrics(metrics)
        # Weighted avg: (0.8*100 + 0.9*200) / 300 = 260/300 ≈ 0.8667
        assert abs(result["accuracy"] - (80 + 180) / 300) < 1e-6
        assert "loss" in result

    def test_aggregate_metrics_empty(self, tiny_model, tmp_path):
        """_aggregate_metrics with empty list returns empty dict."""
        server = FederatedServer(
            model=tiny_model,
            num_rounds=5,
            checkpoint_dir=str(tmp_path),
        )
        assert server._aggregate_metrics([]) == {}

    def test_aggregate_metrics_skips_client_id(self, tiny_model, tmp_path):
        """_aggregate_metrics should skip 'client_id' key."""
        server = FederatedServer(
            model=tiny_model,
            num_rounds=5,
            checkpoint_dir=str(tmp_path),
        )
        metrics = [(100, {"accuracy": 0.8, "client_id": 0})]
        result = server._aggregate_metrics(metrics)
        assert "client_id" not in result
        assert "accuracy" in result


# =============================================================================
# create_server Tests
# =============================================================================


class TestCreateServer:
    def test_returns_config_and_strategy(self):
        from src.models.dscatnet import create_dscatnet
        model = create_dscatnet(num_classes=7, variant="tiny")
        config, strategy = create_server(model, num_rounds=5, min_fit_clients=2,
                                          min_evaluate_clients=2, min_available_clients=2)
        assert config.num_rounds == 5
        assert isinstance(strategy, DSCATNetFedAvg)

    def test_custom_strategy_used(self):
        from src.models.dscatnet import create_dscatnet
        model = create_dscatnet(num_classes=7, variant="tiny")
        custom = DSCATNetFedAvg(total_rounds=99)
        _, strategy = create_server(model, num_rounds=5, strategy=custom)
        assert strategy is custom
