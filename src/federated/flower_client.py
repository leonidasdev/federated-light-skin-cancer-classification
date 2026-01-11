"""
Flower Client Implementation
============================

Flower (flwr) client for federated learning with LMS-ViT.
Implements NumPyClient interface for simulation and distributed training.

This module provides:
- LMSViTFlowerClient: NumPy-based client for Flower framework
- client_fn: Factory function for creating clients in simulation
"""

from collections import OrderedDict
from typing import Dict, List, Tuple, Callable, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import flwr as fl
from flwr.common import NDArrays, Scalar

from ..models import LMSViT, lmsvit_tiny, lmsvit_small, lmsvit_base
from ..data import get_train_transforms, get_val_transforms
from ..utils.logging import get_logger
from ..utils.metrics import MetricsCalculator


class LMSViTFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient implementation for LMS-ViT federated learning.
    
    This client:
    - Loads a local dataset partition
    - Creates DataLoaders for train/val
    - Initializes an LMS-ViT model
    - Loads received global parameters into the model
    - Trains locally for E epochs
    - Evaluates on local validation data
    - Returns model parameters as NumPy arrays, plus metrics
    
    Args:
        client_id: Unique identifier for this client
        train_dataset: Training dataset partition for this client
        val_dataset: Validation dataset partition for this client (optional)
        model_name: LMS-ViT variant ('tiny', 'small', 'base')
        num_classes: Number of output classes
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        local_epochs: Number of local training epochs per round
    """
    
    def __init__(
        self,
        client_id: int,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        model_name: str = "small",
        num_classes: int = 7,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        local_epochs: int = 1,
    ):
        super().__init__()
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if torch.cuda.is_available() else "cpu"
        self.local_epochs = local_epochs
        
        self.logger = get_logger(f"FlowerClient-{client_id}")
        self.metrics_calculator = MetricsCalculator(num_classes=num_classes)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # For simulation, use 0 to avoid multiprocessing issues
            pin_memory=True if self.device == "cuda" else False,
        )
        
        self.val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device == "cuda" else False,
            )
        
        # Initialize model
        self.model = self._create_model(model_name, num_classes)
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info(
            f"Client {client_id} initialized with {len(train_dataset)} training samples"
        )
    
    def _create_model(self, model_name: str, num_classes: int) -> nn.Module:
        """Create LMS-ViT model based on variant name."""
        model_factories = {
            "tiny": lmsvit_tiny,
            "small": lmsvit_small,
            "base": lmsvit_base,
        }
        
        if model_name not in model_factories:
            raise ValueError(
                f"Unknown model variant: {model_name}. "
                f"Choose from: {list(model_factories.keys())}"
            )
        
        return model_factories[model_name](num_classes=num_classes)
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Get current model parameters as NumPy arrays.
        
        Args:
            config: Configuration dictionary from server
            
        Returns:
            List of NumPy arrays representing model parameters
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model parameters from NumPy arrays.
        
        Args:
            parameters: List of NumPy arrays from server
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
            
        Returns:
            Tuple of:
            - Updated model parameters as NumPy arrays
            - Number of training samples
            - Dictionary of training metrics
        """
        # Load global parameters
        self.set_parameters(parameters)
        
        # Get training config (can be overridden by server)
        local_epochs = config.get("local_epochs", self.local_epochs)
        learning_rate = config.get("learning_rate", self.learning_rate)
        
        # Create optimizer (recreate each round to handle potential LR changes)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(int(local_epochs)):
            epoch_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
            
            total_loss += epoch_loss
        
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        metrics = {
            "client_id": self.client_id,
            "train_loss": float(avg_loss),
            "train_samples": len(self.train_dataset),
            "local_epochs": int(local_epochs),
        }
        
        self.logger.info(
            f"Client {self.client_id} - Round complete: "
            f"loss={avg_loss:.4f}, samples={len(self.train_dataset)}"
        )
        
        return self.get_parameters(config={}), len(self.train_dataset), metrics
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Global model parameters from server
            config: Evaluation configuration from server
            
        Returns:
            Tuple of:
            - Loss value
            - Number of evaluation samples
            - Dictionary of evaluation metrics
        """
        # Load global parameters
        self.set_parameters(parameters)
        
        # Use validation loader if available, otherwise use training loader
        eval_loader = self.val_loader if self.val_loader is not None else self.train_loader
        num_samples = len(self.val_dataset) if self.val_dataset else len(self.train_dataset)
        
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        metrics = self.metrics_calculator.calculate(
            y_pred=np.array(all_preds),
            y_true=np.array(all_labels),
        )
        
        # Add client info
        metrics["client_id"] = self.client_id
        metrics["eval_samples"] = num_samples
        
        self.logger.info(
            f"Client {self.client_id} - Evaluation: "
            f"loss={avg_loss:.4f}, accuracy={metrics['accuracy']:.4f}"
        )
        
        return float(avg_loss), num_samples, metrics


def create_client_fn(
    client_datasets: Dict[int, Tuple[Dataset, Optional[Dataset]]],
    model_name: str = "small",
    num_classes: int = 7,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    local_epochs: int = 1,
) -> Callable[[str], LMSViTFlowerClient]:
    """
    Create a client factory function for Flower simulation.
    
    This function returns a callable that creates LMSViTFlowerClient
    instances for each client ID in the simulation.
    
    Args:
        client_datasets: Dictionary mapping client IDs to (train_dataset, val_dataset) tuples
        model_name: LMS-ViT variant
        num_classes: Number of output classes
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on
        local_epochs: Number of local epochs per round
        
    Returns:
        Client factory function that takes a client ID string and returns a client
    """
    def client_fn(cid: str) -> LMSViTFlowerClient:
        """Create a Flower client for the given client ID."""
        client_id = int(cid)
        train_dataset, val_dataset = client_datasets[client_id]
        
        return LMSViTFlowerClient(
            client_id=client_id,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_name=model_name,
            num_classes=num_classes,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            local_epochs=local_epochs,
        )
    
    return client_fn


# TODO: Future extensions
# - Add FedProx support with proximal term in fit()
# - Add differential privacy support
# - Add secure aggregation hooks
# - Add personalization layers that stay local
# - Add support for heterogeneous models across clients
