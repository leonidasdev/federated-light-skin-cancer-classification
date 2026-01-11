"""
Federated Learning Client
=========================

Client implementation for federated learning.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from copy import deepcopy

from ..utils.logging import get_logger


class FederatedClient:
    """
    Federated Learning Client.
    
    Performs local training on private data and communicates with the server.
    
    Args:
        client_id: Unique client identifier
        model: Local model architecture (same as global)
        train_loader: Training data loader for this client
        optimizer_class: Optimizer class to use
        optimizer_kwargs: Optimizer keyword arguments
        criterion: Loss function
        device: Device to train on
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer_class: type = torch.optim.SGD,
        optimizer_kwargs: Optional[Dict] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
    ):
        self.client_id = client_id
        self.model = deepcopy(model).to(device)
        self.train_loader = train_loader
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 0.01}
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        
        self.logger = get_logger(f"Client-{client_id}")
        self.num_samples = len(train_loader.dataset)
        
        self._init_optimizer()
    
    def _init_optimizer(self) -> None:
        """Initialize optimizer with current model parameters."""
        self.optimizer = self.optimizer_class(
            self.model.parameters(),
            **self.optimizer_kwargs
        )
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get current model parameters.
        
        Returns:
            Model state dictionary
        """
        return deepcopy(self.model.state_dict())
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Set model parameters (from server).
        
        Args:
            parameters: Model state dictionary
        """
        self.model.load_state_dict(parameters)
        self._init_optimizer()
    
    def train(self, epochs: int = 1) -> Tuple[Dict[str, float], int]:
        """
        Perform local training.
        
        Args:
            epochs: Number of local epochs
            
        Returns:
            Tuple of (metrics dict, number of samples)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            total_loss += epoch_loss / len(self.train_loader)
        
        metrics = {
            'loss': total_loss / epochs,
            'accuracy': correct / total,
        }
        
        self.logger.debug(
            f"Local training complete - Loss: {metrics['loss']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )
        
        return metrics, self.num_samples
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate local model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(test_loader),
            'accuracy': correct / total,
        }
