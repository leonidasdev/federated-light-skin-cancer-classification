"""
Centralized Trainer
===================

Standard centralized training pipeline for baseline experiments.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm

from .trainer import Trainer


class CentralizedTrainer(Trainer):
    """
    Centralized training pipeline for baseline experiments.
    
    This serves as a baseline to compare against federated learning approaches.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on ('cuda' or 'cpu')
        callbacks: List of callback functions
        gradient_clip: Max gradient norm for clipping (None to disable)
        mixed_precision: Whether to use automatic mixed precision
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        callbacks: Optional[List[Callable]] = None,
        gradient_clip: Optional[float] = None,
        mixed_precision: bool = False,
    ):
        super().__init__(model, optimizer, criterion, device, callbacks)
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision
        
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary containing loss and accuracy
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            self.global_step += 1
            
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total,
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total,
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing loss, accuracy, and other metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = self.metrics_calculator.calculate(all_preds, all_labels)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
