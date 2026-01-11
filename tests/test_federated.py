"""
Federated Learning Tests (Flower)
==================================

Unit tests for Flower-based federated learning components.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.federated import (
    LMSViTFlowerClient,
    create_client_fn,
    create_strategy,
    weighted_average,
)


class DummyModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestWeightedAverage:
    """Tests for weighted average aggregation function."""
    
    def test_weighted_average_basic(self):
        """Test weighted average with simple metrics."""
        metrics = [
            (100, {"accuracy": 0.8}),
            (100, {"accuracy": 0.9}),
        ]
        
        result = weighted_average(metrics)
        
        assert "accuracy" in result
        assert abs(result["accuracy"] - 0.85) < 1e-5
    
    def test_weighted_average_unequal_weights(self):
        """Test weighted average with unequal sample sizes."""
        metrics = [
            (100, {"accuracy": 0.8}),  # Weight 100
            (300, {"accuracy": 0.9}),  # Weight 300
        ]
        
        result = weighted_average(metrics)
        
        # Expected: (100*0.8 + 300*0.9) / 400 = (80 + 270) / 400 = 0.875
        assert abs(result["accuracy"] - 0.875) < 1e-5
    
    def test_weighted_average_empty(self):
        """Test weighted average with empty metrics."""
        result = weighted_average([])
        assert result == {}
    
    def test_weighted_average_multiple_metrics(self):
        """Test weighted average with multiple metrics."""
        metrics = [
            (100, {"accuracy": 0.8, "loss": 0.5}),
            (100, {"accuracy": 0.9, "loss": 0.3}),
        ]
        
        result = weighted_average(metrics)
        
        assert abs(result["accuracy"] - 0.85) < 1e-5
        assert abs(result["loss"] - 0.4) < 1e-5


class TestFlowerClient:
    """Tests for Flower client implementation."""
    
    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing."""
        return DummyModel()
    
    @pytest.fixture
    def dummy_dataloader(self):
        """Create a dummy dataloader for testing."""
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=8)
    
    def test_flower_client_init(self, dummy_model, dummy_dataloader):
        """Test FlowerClient initialization."""
        client = LMSViTFlowerClient(
            model=dummy_model,
            trainloader=dummy_dataloader,
            valloader=dummy_dataloader,
            num_classes=2,
            device="cpu",
        )
        
        assert client.model is dummy_model
        assert client.num_classes == 2
        assert client.device == "cpu"
    
    def test_flower_client_get_parameters(self, dummy_model, dummy_dataloader):
        """Test getting parameters from client."""
        client = LMSViTFlowerClient(
            model=dummy_model,
            trainloader=dummy_dataloader,
            valloader=dummy_dataloader,
            num_classes=2,
            device="cpu",
        )
        
        params = client.get_parameters(config={})
        
        # Should return list of numpy arrays
        assert isinstance(params, list)
        assert len(params) > 0
        # Check shapes match model parameters
        model_params = [p.detach().cpu().numpy() for p in dummy_model.parameters()]
        for p1, p2 in zip(params, model_params):
            assert p1.shape == p2.shape
    
    def test_flower_client_set_parameters(self, dummy_model, dummy_dataloader):
        """Test setting parameters on client."""
        client = LMSViTFlowerClient(
            model=dummy_model,
            trainloader=dummy_dataloader,
            valloader=dummy_dataloader,
            num_classes=2,
            device="cpu",
        )
        
        # Get original parameters
        original_params = client.get_parameters(config={})
        
        # Create new parameters (zeros)
        new_params = [p * 0 for p in original_params]
        
        # Set new parameters
        client.set_parameters(new_params)
        
        # Verify parameters were updated
        updated_params = client.get_parameters(config={})
        for p in updated_params:
            assert (p == 0).all()


class TestCreateStrategy:
    """Tests for strategy creation function."""
    
    def test_create_fedavg_strategy(self):
        """Test creating FedAvg strategy."""
        strategy = create_strategy("fedavg")
        
        assert strategy is not None
        assert strategy.__class__.__name__ == "FedAvg"
    
    def test_create_fedprox_strategy(self):
        """Test creating FedProx strategy."""
        strategy = create_strategy("fedprox", proximal_mu=0.1)
        
        assert strategy is not None
        assert strategy.__class__.__name__ == "FedProx"
    
    def test_create_fedadam_strategy(self):
        """Test creating FedAdam strategy."""
        strategy = create_strategy("fedadam")
        
        assert strategy is not None
        assert strategy.__class__.__name__ == "FedAdam"
    
    def test_create_unknown_strategy_raises(self):
        """Test that unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("unknown_strategy")


class TestCreateClientFn:
    """Tests for client function factory."""
    
    def test_create_client_fn_returns_callable(self):
        """Test that create_client_fn returns a callable."""
        model_fn = DummyModel
        
        # Create dummy client dataloaders
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        
        client_dataloaders = {
            "0": (loader, loader),
            "1": (loader, loader),
        }
        
        client_fn = create_client_fn(
            model_fn=model_fn,
            client_dataloaders=client_dataloaders,
            num_classes=2,
            device="cpu",
        )
        
        assert callable(client_fn)
    
    def test_create_client_fn_creates_client(self):
        """Test that client_fn creates valid client."""
        model_fn = DummyModel
        
        # Create dummy client dataloaders
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        
        client_dataloaders = {
            "0": (loader, loader),
            "1": (loader, loader),
        }
        
        client_fn = create_client_fn(
            model_fn=model_fn,
            client_dataloaders=client_dataloaders,
            num_classes=2,
            device="cpu",
        )
        
        # Create client for client ID "0"
        client = client_fn("0")
        
        assert isinstance(client, LMSViTFlowerClient)
