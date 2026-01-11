"""
Base Trainer
============

Abstract base class for training pipelines.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable, List
from torch.utils.data import DataLoader
from pathlib import Path

from ..utils.logging import get_logger
from ..utils.metrics import MetricsCalculator


class Trainer(ABC):
    """
    Abstract base trainer class for both centralized and federated training.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on ('cuda' or 'cpu')
        logger: Logger instance
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        callbacks: Optional[List[Callable]] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callbacks = callbacks or []
        
        self.logger = get_logger(self.__class__.__name__)
        self.metrics_calculator = MetricsCalculator()
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
    
    @abstractmethod
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train
            
        Returns:
            Dictionary of metric histories
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics.get('accuracy', 0))
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics.get('accuracy', 0))
            
            # Execute callbacks
            for callback in self.callbacks:
                callback(self, train_metrics, val_metrics if val_loader else None)
            
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Train Acc: {train_metrics.get('accuracy', 0):.4f}"
            )
        
        return history
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.logger.info(f"Checkpoint loaded from {path}")
