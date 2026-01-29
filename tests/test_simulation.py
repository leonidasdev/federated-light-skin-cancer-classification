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
        assert config.num_rounds == 50
        assert config.local_epochs == 1
        assert config.batch_size == 8  # Updated to match reduced batch size for memory efficiency
        assert config.noniid_type == "natural"
    
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
        
        simulator = FLSimulator(config)
        
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


class TestCommunicationCost:
    """Tests for communication cost calculations."""
    
    def test_model_size_calculation(self, tmp_path):
        """Test model size calculation for communication cost."""
        import torch
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


# Skip integration tests by default (require datasets)
@pytest.mark.skip(reason="Requires actual datasets")
class TestFLSimulatorIntegration:
    """Integration tests for FL simulation (require datasets)."""
    
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
