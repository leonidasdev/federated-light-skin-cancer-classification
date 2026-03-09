# =============================================================================
# Tests for Federated Learning Simulation
# =============================================================================
"""
Tests for FL Simulation.

Tests the federated learning simulation infrastructure.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest
import numpy as np

# =============================================================================
# Test Classes
# =============================================================================


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.federated.simulation import SimulationConfig

        config = SimulationConfig()

        assert config.num_clients == 4
        assert config.num_rounds == 100
        assert config.local_epochs == 1
        assert config.batch_size == 4
        assert config.noniid_type == "natural"
        assert config.use_class_weights is False  # Paper: unweighted cross-entropy

    def test_config_to_dict(self):
        """Test config serialization."""
        from src.federated.simulation import SimulationConfig

        config = SimulationConfig(
            num_rounds=10,
            experiment_name="test_exp"
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["num_rounds"] == 10
        assert config_dict["experiment_name"] == "test_exp"

    def test_config_from_dict(self):
        """Test config deserialization."""
        from src.federated.simulation import SimulationConfig

        config_dict = {
            "num_rounds": 25,
            "num_clients": 3,
            "local_epochs": 2,
        }

        config = SimulationConfig.from_dict(config_dict)

        assert config.num_rounds == 25
        assert config.num_clients == 3
        assert config.local_epochs == 2


class TestFLSimulator:
    """Tests for FLSimulator."""

    def test_simulator_init(self, tmp_path):
        """Test simulator initialization."""
        from src.federated.simulation import SimulationConfig, FLSimulator

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="test_sim",
            pretrained=False,  # Faster for testing
        )

        simulator = FLSimulator(config)

        assert simulator.config.experiment_name == "test_sim"
        assert simulator.global_model is not None
        assert len(simulator.history["rounds"]) == 0
        assert (tmp_path / "test_sim").exists()

    def test_output_directories_created(self, tmp_path):
        """Test that output directories are properly created."""
        from src.federated.simulation import SimulationConfig, FLSimulator

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="dir_test",
            pretrained=False,
        )

        # Creating the simulator should create output directories
        FLSimulator(config)

        assert (tmp_path / "dir_test").exists()
        assert (tmp_path / "dir_test" / "checkpoints").exists()

    def test_aggregate_parameters(self, tmp_path):
        """Test FedAvg parameter aggregation."""
        from src.federated.simulation import SimulationConfig, FLSimulator

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="agg_test",
            pretrained=False,
        )

        simulator = FLSimulator(config)

        # Create mock parameters
        params1 = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
        params2 = [np.array([2.0, 4.0, 6.0]), np.array([8.0, 10.0])]

        # Equal weights
        results = [
            (params1, 50),  # 50 samples
            (params2, 50),  # 50 samples
        ]

        aggregated = simulator.aggregate_parameters(results)

        # With equal samples, should be simple average
        np.testing.assert_array_almost_equal(
            aggregated[0],
            np.array([1.5, 3.0, 4.5])
        )
        np.testing.assert_array_almost_equal(
            aggregated[1],
            np.array([6.0, 7.5])
        )

    def test_aggregate_parameters_weighted(self, tmp_path):
        """Test weighted FedAvg aggregation."""
        from src.federated.simulation import SimulationConfig, FLSimulator

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="weighted_test",
            pretrained=False,
        )

        simulator = FLSimulator(config)

        # Client 1 has 3x more samples
        params1 = [np.array([1.0, 1.0])]
        params2 = [np.array([4.0, 4.0])]

        results = [
            (params1, 75),  # 75% of total
            (params2, 25),  # 25% of total
        ]

        aggregated = simulator.aggregate_parameters(results)

        # Weighted average: 0.75 * [1,1] + 0.25 * [4,4] = [1.75, 1.75]
        np.testing.assert_array_almost_equal(
            aggregated[0],
            np.array([1.75, 1.75])
        )


class TestClientSelection:
    """Tests for client selection and parallelism features."""

    def test_client_selection_full_participation(self, tmp_path):
        """Test that full participation (1.0) selects all clients."""
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="selection_test",
            pretrained=False,
            client_selection_fraction=1.0,
        )

        simulator = FLSimulator(config)

        # Mock client data
        dummy_data = torch.randn(8, 3, 32, 32)
        dummy_labels = torch.randint(0, 7, (8,))
        dummy_loader = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=2)

        for i in range(4):
            simulator.client_data[i] = ClientData(
                client_id=i,
                train_loader=dummy_loader,
                val_loader=dummy_loader,
                num_train_samples=100,
                num_val_samples=20,
                class_distribution={0: 50, 1: 50},
                dataset_name=f"client_{i}",
            )

        selected = simulator._select_clients(round_num=1)
        assert len(selected) == 4

    def test_client_selection_partial(self, tmp_path):
        """Test that partial participation selects correct number of clients."""
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="partial_test",
            pretrained=False,
            client_selection_fraction=0.5,
            min_fit_clients=1,
        )

        simulator = FLSimulator(config)

        # Mock client data
        dummy_data = torch.randn(8, 3, 32, 32)
        dummy_labels = torch.randint(0, 7, (8,))
        dummy_loader = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=2)

        for i in range(4):
            simulator.client_data[i] = ClientData(
                client_id=i,
                train_loader=dummy_loader,
                val_loader=dummy_loader,
                num_train_samples=100,
                num_val_samples=20,
                class_distribution={0: 50, 1: 50},
                dataset_name=f"client_{i}",
            )

        selected = simulator._select_clients(round_num=1)
        # 50% of 4 clients = 2
        assert len(selected) == 2

    def test_client_selection_reproducibility(self, tmp_path):
        """Test that same round number produces same selection."""
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="repro_test",
            pretrained=False,
            client_selection_fraction=0.5,
            min_fit_clients=1,
        )

        simulator = FLSimulator(config)

        # Mock client data
        dummy_data = torch.randn(8, 3, 32, 32)
        dummy_labels = torch.randint(0, 7, (8,))
        dummy_loader = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=2)

        for i in range(4):
            simulator.client_data[i] = ClientData(
                client_id=i,
                train_loader=dummy_loader,
                val_loader=dummy_loader,
                num_train_samples=100,
                num_val_samples=20,
                class_distribution={0: 50, 1: 50},
                dataset_name=f"client_{i}",
            )

        selected1 = simulator._select_clients(round_num=5)
        selected2 = simulator._select_clients(round_num=5)

        assert selected1 == selected2

    def test_parallel_workers_auto_detection(self, tmp_path):
        """Test auto-detection of parallel workers."""
        import os
        from src.federated.simulation import SimulationConfig, FLSimulator

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="parallel_test",
            pretrained=False,
            parallel_clients=0,  # Auto-detect
        )

        simulator = FLSimulator(config)
        workers = simulator._get_parallel_workers()

        # Should be between 1 and min(cpu_count, 4)
        expected_max = min(os.cpu_count() or 1, 4)
        assert 1 <= workers <= expected_max

    def test_client_selection_single_client(self, tmp_path):
        """Test client selection with only 1 client available."""
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="single_client_test",
            pretrained=False,
            client_selection_fraction=0.5,
            min_fit_clients=1,
        )

        simulator = FLSimulator(config)

        # Mock single client
        dummy_data = torch.randn(8, 3, 32, 32)
        dummy_labels = torch.randint(0, 7, (8,))
        dummy_loader = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=2)

        simulator.client_data[0] = ClientData(
            client_id=0,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            num_train_samples=100,
            num_val_samples=20,
            class_distribution={0: 50, 1: 50},
            dataset_name="single_client",
        )

        selected = simulator._select_clients(round_num=1)
        # With only 1 client, should select 1 (min_fit_clients)
        assert len(selected) == 1
        assert 0 in selected

    def test_client_selection_respects_min_fit_clients(self, tmp_path):
        """Test that client selection respects min_fit_clients."""
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="min_fit_test",
            pretrained=False,
            client_selection_fraction=0.1,  # Would select 0.4 clients
            min_fit_clients=2,  # But minimum is 2
        )

        simulator = FLSimulator(config)

        # Mock 4 clients
        dummy_data = torch.randn(8, 3, 32, 32)
        dummy_labels = torch.randint(0, 7, (8,))
        dummy_loader = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=2)

        for i in range(4):
            simulator.client_data[i] = ClientData(
                client_id=i,
                train_loader=dummy_loader,
                val_loader=dummy_loader,
                num_train_samples=100,
                num_val_samples=20,
                class_distribution={0: 50, 1: 50},
                dataset_name=f"client_{i}",
            )

        selected = simulator._select_clients(round_num=1)
        # 10% of 4 = 0.4, but min_fit_clients=2
        assert len(selected) >= 2

    def test_train_val_split_config(self):
        """Test that train_val_split config is applied."""
        from src.federated.simulation import SimulationConfig

        config = SimulationConfig(train_val_split=0.9)
        assert config.train_val_split == 0.9

        # Test serialization round-trip
        config_dict = config.to_dict()
        config_restored = SimulationConfig.from_dict(config_dict)
        assert config_restored.train_val_split == 0.9


class TestClientData:
    """Tests for ClientData dataclass."""

    def test_client_data_creation(self):
        """Test ClientData dataclass."""
        from src.federated.simulation import ClientData
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        # Small dummy DataLoader for testing
        dummy_data = torch.randn(8, 3, 32, 32)
        dummy_labels = torch.randint(0, 2, (8,))
        dummy_loader = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=2)

        client = ClientData(
            client_id=0,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            num_train_samples=100,
            num_val_samples=20,
            class_distribution={0: 50, 1: 50},
            dataset_name="test_dataset",
        )

        assert client.client_id == 0
        assert client.num_train_samples == 100
        assert client.dataset_name == "test_dataset"
        assert sum(client.class_distribution.values()) == 100


class TestClassWeights:
    """Tests for class weight computation in FL simulator."""

    def test_compute_class_weights(self, tmp_path):
        """Test that class weights are computed correctly from client distributions."""
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="cw_test",
            pretrained=False,
            use_class_weights=True,
            num_classes=3,
        )

        simulator = FLSimulator(config)

        # Mock client data with imbalanced distributions
        dummy_data = torch.randn(4, 3, 32, 32)
        dummy_labels = torch.randint(0, 3, (4,))
        dummy_loader = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=2)

        # Client 0: mostly class 0
        simulator.client_data[0] = ClientData(
            client_id=0, train_loader=dummy_loader, val_loader=dummy_loader,
            num_train_samples=100, num_val_samples=20,
            class_distribution={0: 80, 1: 10, 2: 10},
            dataset_name="client_0",
        )
        # Client 1: mostly class 1
        simulator.client_data[1] = ClientData(
            client_id=1, train_loader=dummy_loader, val_loader=dummy_loader,
            num_train_samples=100, num_val_samples=20,
            class_distribution={0: 10, 1: 80, 2: 10},
            dataset_name="client_1",
        )

        simulator._compute_class_weights()

        assert simulator.class_weights is not None
        assert simulator.class_weights.shape == (3,)
        # Total: 200 samples, class 0: 90, class 1: 90, class 2: 20
        # weight_0 = 200 / (3 * 90), weight_2 = 200 / (3 * 20) — class 2 should have highest weight
        assert simulator.class_weights[2] > simulator.class_weights[0]
        assert simulator.class_weights[2] > simulator.class_weights[1]

    def test_class_weights_disabled(self, tmp_path):
        """Test that class weights are None when disabled."""
        from src.federated.simulation import SimulationConfig, FLSimulator

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="cw_disabled_test",
            pretrained=False,
            use_class_weights=False,
        )

        simulator = FLSimulator(config)
        assert simulator.class_weights is None

    def test_class_weights_config_roundtrip(self):
        """Test use_class_weights serialization round-trip."""
        from src.federated.simulation import SimulationConfig

        config = SimulationConfig(use_class_weights=False)
        config_dict = config.to_dict()
        restored = SimulationConfig.from_dict(config_dict)
        assert restored.use_class_weights is False


class TestCommunicationCost:
    """Tests for communication cost calculations."""

    def test_model_size_calculation(self, tmp_path):
        """Test model size calculation for communication cost."""
        from src.federated.simulation import SimulationConfig, FLSimulator
        from src.models.dscatnet import get_model_parameters

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="comm_test",
            pretrained=False,
            model_variant="tiny",  # Smaller for testing
        )

        simulator = FLSimulator(config)
        params = get_model_parameters(simulator.global_model)

        # Calculate size
        total_bytes = sum(p.nbytes for p in params)

        # Should be positive and reasonable
        assert total_bytes > 0
        # Tiny model should be < 100MB
        assert total_bytes < 100 * 1024 * 1024


class TestTrainAndEvaluateClient:
    """Tests for train_client and evaluate_client methods."""

    @pytest.fixture
    def simulator_with_client(self, tmp_path):
        """Create a simulator with one mock client using tiny model."""
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="train_eval_test",
            pretrained=False,
            model_variant="tiny",
            image_size=32,
            local_epochs=1,
            batch_size=4,
        )

        simulator = FLSimulator(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        train_loader = DataLoader(TensorDataset(images, labels), batch_size=4, drop_last=True)
        val_loader = DataLoader(TensorDataset(images, labels), batch_size=4)

        simulator.client_data[0] = ClientData(
            client_id=0,
            train_loader=train_loader,
            val_loader=val_loader,
            num_train_samples=8,
            num_val_samples=8,
            class_distribution={0: 4, 1: 4},
            dataset_name="test_ds",
        )

        return simulator

    def test_train_client(self, simulator_with_client):
        from src.models.dscatnet import get_model_parameters
        params = get_model_parameters(simulator_with_client.global_model)
        updated, n_samples, metrics = simulator_with_client.train_client(0, params)
        assert isinstance(updated, list)
        assert n_samples == 8
        assert "train_loss" in metrics
        assert "train_accuracy" in metrics

    def test_train_client_invalid_id(self, simulator_with_client):
        from src.models.dscatnet import get_model_parameters
        params = get_model_parameters(simulator_with_client.global_model)
        with pytest.raises(ValueError, match="Client 99 not found"):
            simulator_with_client.train_client(99, params)

    def test_evaluate_client(self, simulator_with_client):
        from src.models.dscatnet import get_model_parameters
        params = get_model_parameters(simulator_with_client.global_model)
        loss, n_samples, metrics = simulator_with_client.evaluate_client(0, params)
        assert isinstance(loss, float)
        assert n_samples == 8
        assert "val_accuracy" in metrics

    def test_evaluate_client_invalid_id(self, simulator_with_client):
        from src.models.dscatnet import get_model_parameters
        params = get_model_parameters(simulator_with_client.global_model)
        with pytest.raises(ValueError, match="Client 99 not found"):
            simulator_with_client.evaluate_client(99, params)

    def test_train_with_adamw(self, tmp_path):
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from src.models.dscatnet import get_model_parameters
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="adamw_test",
            pretrained=False,
            model_variant="tiny",
            image_size=32,
            optimizer_type="adamw",
        )
        simulator = FLSimulator(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, drop_last=True)
        simulator.client_data[0] = ClientData(
            client_id=0, train_loader=loader, val_loader=loader,
            num_train_samples=8, num_val_samples=8,
            class_distribution={0: 4, 1: 4}, dataset_name="test",
        )

        params = get_model_parameters(simulator.global_model)
        _, _, metrics = simulator.train_client(0, params)
        assert "train_loss" in metrics

    def test_train_with_gradient_accumulation(self, tmp_path):
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from src.models.dscatnet import get_model_parameters
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="accum_test",
            pretrained=False,
            model_variant="tiny",
            image_size=32,
            gradient_accumulation_steps=2,
        )
        simulator = FLSimulator(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, drop_last=True)
        simulator.client_data[0] = ClientData(
            client_id=0, train_loader=loader, val_loader=loader,
            num_train_samples=8, num_val_samples=8,
            class_distribution={0: 4, 1: 4}, dataset_name="test",
        )

        params = get_model_parameters(simulator.global_model)
        _, _, metrics = simulator.train_client(0, params)
        assert "train_loss" in metrics


class TestSimulatorCheckpoints:
    """Tests for save_checkpoint, load_checkpoint, and save_best_model."""

    @pytest.fixture
    def simulator(self, tmp_path):
        from src.federated.simulation import SimulationConfig, FLSimulator
        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="ckpt_test",
            pretrained=False,
            model_variant="tiny",
        )
        return FLSimulator(config)

    def test_save_checkpoint(self, simulator):
        metrics = {"train_loss": 0.5, "val_accuracy": 0.8}
        simulator.save_checkpoint(round_num=3, metrics=metrics)
        assert (simulator.checkpoint_dir / "checkpoint_round_3.pt").exists()

    def test_save_and_load_checkpoint(self, simulator):
        simulator.best_val_accuracy = 0.9
        simulator.best_round = 5
        simulator.history["rounds"].append(1)
        simulator.save_checkpoint(round_num=1, metrics={"val_accuracy": 0.9})

        # Load into fresh simulator
        from src.federated.simulation import SimulationConfig, FLSimulator
        sim2 = FLSimulator(SimulationConfig(
            output_dir=str(simulator.output_dir.parent),
            experiment_name="ckpt_test2",
            pretrained=False,
            model_variant="tiny",
        ))
        ckpt_path = str(simulator.checkpoint_dir / "checkpoint_round_1.pt")
        resumed = sim2.load_checkpoint(ckpt_path)
        assert resumed == 1
        assert sim2.best_val_accuracy == 0.9

    def test_save_best_model(self, simulator):
        simulator.best_val_accuracy = 0.95
        simulator.save_best_model(round_num=10)
        assert (simulator.checkpoint_dir / "best_checkpoint.pt").exists()
        assert (simulator.checkpoint_dir / "best_model.pt").exists()


class TestRunRound:
    """Tests for run_round method."""

    def test_run_round_returns_metrics(self, tmp_path):
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from torch.utils.data import TensorDataset, DataLoader
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="round_test",
            pretrained=False,
            model_variant="tiny",
            image_size=32,
            batch_size=4,
        )
        simulator = FLSimulator(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, drop_last=True)

        simulator.client_data[0] = ClientData(
            client_id=0, train_loader=loader, val_loader=loader,
            num_train_samples=8, num_val_samples=8,
            class_distribution={0: 4, 1: 4}, dataset_name="test_ds",
        )

        metrics = simulator.run_round(round_num=1)
        assert "train_loss" in metrics
        assert "train_accuracy" in metrics
        assert "val_loss" in metrics
        assert "val_accuracy" in metrics
        assert "communication_cost_mb" in metrics
        assert "clients_participated" in metrics
        assert metrics["clients_participated"] == 1


class TestDirichletSubset:
    """Tests for DirichletSubset."""

    def test_len_and_getitem(self):
        from src.federated.simulation import DirichletSubset
        from torch.utils.data import TensorDataset
        import torch

        ds = TensorDataset(torch.randn(10, 3, 32, 32), torch.arange(10))
        combined = [(ds, i) for i in range(10)]
        subset = DirichletSubset(combined, [0, 2, 4])

        assert len(subset) == 3
        img, _label = subset[0]
        assert img.shape == (3, 32, 32)

    def test_different_indices(self):
        """DirichletSubset with different indices returns different items."""
        from src.federated.simulation import DirichletSubset
        from torch.utils.data import TensorDataset
        import torch

        ds = TensorDataset(torch.arange(10).float().unsqueeze(1), torch.arange(10))
        combined = [(ds, i) for i in range(10)]
        subset = DirichletSubset(combined, [1, 3, 5])
        _, label0 = subset[0]
        _, label1 = subset[1]
        assert label0 == 1
        assert label1 == 3


class TestGetTransforms:
    """Tests for FLSimulator._get_transforms."""

    def test_returns_pair(self, tmp_path):
        from src.federated.simulation import SimulationConfig, FLSimulator

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="tfm_test",
            pretrained=False,
        )
        simulator = FLSimulator(config)
        train_tfm, val_tfm = simulator._get_transforms()
        assert train_tfm is not None
        assert val_tfm is not None


class TestSetupClients:
    """Tests for setup_clients dispatching."""

    def test_setup_clients_unknown_noniid_type(self, tmp_path):
        """Unknown noniid_type should fall back to natural."""
        from src.federated.simulation import SimulationConfig, FLSimulator
        from unittest.mock import patch as mock_patch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="unknown_noniid_test",
            pretrained=False,
            noniid_type="unknown_type",
            use_class_weights=False,
        )
        simulator = FLSimulator(config)

        with mock_patch.object(simulator, 'setup_natural_noniid') as mock_natural:
            simulator.setup_clients()
            mock_natural.assert_called_once()


class TestRunSimulation:
    """Tests for the run() method and run_fl_simulation convenience function."""

    def test_run_with_mocked_clients(self, tmp_path):
        """Test the run() method with pre-populated client data."""
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from torch.utils.data import TensorDataset, DataLoader
        from unittest.mock import patch as mock_patch
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="run_test",
            pretrained=False,
            model_variant="tiny",
            image_size=32,
            batch_size=4,
            num_rounds=2,
            num_clients=1,
            checkpoint_interval=1,
            early_stopping_patience=100,
        )
        simulator = FLSimulator(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, drop_last=True)

        simulator.client_data[0] = ClientData(
            client_id=0, train_loader=loader, val_loader=loader,
            num_train_samples=8, num_val_samples=8,
            class_distribution={0: 4, 1: 4}, dataset_name="test_ds",
        )

        # Patch setup_clients to not load real data (we already set client_data)
        with mock_patch.object(simulator, 'setup_clients'):
            results = simulator.run()

        assert "history" in results
        assert "best_val_accuracy" in results
        assert len(results["history"]["rounds"]) == 2
        assert results["total_communication_mb"] > 0
        # Config saved
        assert (simulator.output_dir / "config.json").exists()
        # Results saved
        assert (simulator.output_dir / "results.json").exists()

    def test_run_early_stopping(self, tmp_path):
        """Test that early stopping works in run() method."""
        from src.federated.simulation import SimulationConfig, FLSimulator, ClientData
        from torch.utils.data import TensorDataset, DataLoader
        from unittest.mock import patch as mock_patch
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="early_stop_test",
            pretrained=False,
            model_variant="tiny",
            image_size=32,
            batch_size=4,
            num_rounds=100,
            num_clients=1,
            early_stopping_patience=2,
        )
        simulator = FLSimulator(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, drop_last=True)

        simulator.client_data[0] = ClientData(
            client_id=0, train_loader=loader, val_loader=loader,
            num_train_samples=8, num_val_samples=8,
            class_distribution={0: 4, 1: 4}, dataset_name="test_ds",
        )

        with mock_patch.object(simulator, 'setup_clients'):
            results = simulator.run()

        # Should stop early (well before 100 rounds)
        assert len(results["history"]["rounds"]) < 100

    def test_run_no_clients_raises(self, tmp_path):
        """Test that run() raises when no clients available."""
        from src.federated.simulation import SimulationConfig, FLSimulator
        from unittest.mock import patch as mock_patch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="no_clients_test",
            pretrained=False,
            model_variant="tiny",
        )
        simulator = FLSimulator(config)

        # setup_clients does nothing, leaving client_data empty
        with mock_patch.object(simulator, 'setup_clients'), pytest.raises(RuntimeError, match="No clients available"):
            simulator.run()

    def test_run_fl_simulation_convenience(self, tmp_path):
        """Test run_fl_simulation convenience function with mocked run."""
        from src.federated.simulation import run_fl_simulation, SimulationConfig
        from unittest.mock import patch as mock_patch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="convenience_test",
            pretrained=False,
            model_variant="tiny",
            image_size=32,
            num_rounds=1,
        )

        # Mock FLSimulator.run to avoid needing real data
        with mock_patch("src.federated.simulation.FLSimulator.run", return_value={"done": True}):
            result = run_fl_simulation(config)
        assert result == {"done": True}

    def test_load_checkpoint_partial_state(self, tmp_path):
        """Test load_checkpoint with partial checkpoint (missing some keys)."""
        from src.federated.simulation import SimulationConfig, FLSimulator
        import torch

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="partial_ckpt",
            pretrained=False,
            model_variant="tiny",
        )
        simulator = FLSimulator(config)

        # Save a minimal checkpoint without some optional keys
        checkpoint = {
            "round": 3,
            "model_state_dict": simulator.global_model.state_dict(),
            "metrics": {"val_accuracy": 0.75},
        }
        ckpt_path = tmp_path / "partial.pt"
        torch.save(checkpoint, ckpt_path)

        resumed = simulator.load_checkpoint(str(ckpt_path))
        assert resumed == 3
        assert simulator.best_val_accuracy == 0.75


# Integration tests that require actual data
@pytest.mark.integration
@pytest.mark.slow
class TestFLSimulatorIntegration:
    """Integration tests for FL simulation (require datasets). Run with: pytest -m slow"""

    @pytest.fixture(autouse=True)
    def check_datasets(self):
        """Skip if no datasets available."""
        from pathlib import Path
        data_root = Path(__file__).parent.parent / "data"
        ham_csv = data_root / "HAM10000" / "HAM10000_metadata.csv"
        if not ham_csv.exists():
            pytest.skip("HAM10000 dataset not available")

    def test_full_simulation_run(self, tmp_path):
        """Test complete simulation with actual data."""
        from src.federated.simulation import SimulationConfig, FLSimulator

        config = SimulationConfig(
            output_dir=str(tmp_path),
            experiment_name="full_test",
            data_root="./data",
            num_rounds=2,
            local_epochs=1,
            pretrained=False,
        )

        simulator = FLSimulator(config)
        results = simulator.run()

        assert "history" in results
        assert "best_val_accuracy" in results
        assert results["best_val_accuracy"] >= 0
