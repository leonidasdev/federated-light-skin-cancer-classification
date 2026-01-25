"""
Flower FL Client for Skin Cancer Classification.

Each client represents a hospital/institution with its own dermoscopy dataset:
- Client 1: HAM10000
- Client 2: ISIC 2018
- Client 3: ISIC 2019
- Client 4: ISIC 2020
"""

from typing import Dict, List, Tuple, Sized, cast, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from flwr.client import Client as FLClient
from flwr.common import NDArrays, Scalar
from tqdm import tqdm

from ..models.dscatnet import DSCATNet, get_model_parameters, set_model_parameters

# ---------------------------------------------------------------------------
# AMP Compatibility: Use `torch.amp.autocast` if available (PyTorch >=2.0),
# otherwise fall back to the deprecated `torch.cuda.amp.autocast`.
# ---------------------------------------------------------------------------
try:
    _HAS_TORCH_AMP_AUTOCAST = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
except Exception:
    _HAS_TORCH_AMP_AUTOCAST = False


def _autocast():
    """Return appropriate autocast context manager based on PyTorch version."""
    if _HAS_TORCH_AMP_AUTOCAST:
        return torch.amp.autocast("cuda")  # type: ignore[attr-defined]
    return torch.cuda.amp.autocast()


class SkinCancerClient(NumPyClient):
    """
    Flower client for skin cancer classification with DSCATNet.
    
    This client handles local training and evaluation for a single
    dermoscopy dataset in the federated learning setup.
    
    Args:
        client_id: Unique identifier for this client (1-4)
        model: DSCATNet model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run training on
        local_epochs: Number of local training epochs per round
        learning_rate: Learning rate for optimizer
        class_weights: Optional class weights for imbalanced data
        use_amp: Enable Automatic Mixed Precision (AMP) for faster training
    """
    
    def __init__(
        self,
        client_id: int,
        model: DSCATNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 1,
        learning_rate: float = 1e-3,
        class_weights: Optional[torch.Tensor] = None,
        use_amp: bool = True
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        
        # Move model to device
        self.model.to(self.device)
        
        # AMP (Automatic Mixed Precision) for faster training
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Loss function with optional class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Scheduler for learning rate decay
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Will be adjusted based on FL rounds
            eta_min=1e-6
        )
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model parameters as numpy arrays."""
        return get_model_parameters(self.model)
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from numpy arrays."""
        set_model_parameters(self.model, parameters)
        
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train model on local dataset.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
            
        Returns:
            Tuple of (updated parameters, num_examples, metrics)
        """
        # Update model with global parameters
        self.set_parameters(parameters)
        
        # Get config values
        epochs = int(config.get("local_epochs", self.local_epochs))
        current_round = int(config.get("current_round", 0))
        
        # Train locally
        train_loss, train_acc = self._train_epoch(epochs)
        
        # Record history
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        
        # Step scheduler
        self.scheduler.step()
        
        # Prepare metrics
        metrics = {
            "client_id": self.client_id,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "round": current_round,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        
        num_examples = len(cast(Sized, self.train_loader.dataset))
        return self.get_parameters(config), num_examples, metrics
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate model on local validation set.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Update model with parameters
        self.set_parameters(parameters)
        
        # Evaluate
        loss, accuracy, metrics = self._evaluate()
        
        # Record history
        self.history['val_loss'].append(loss)
        self.history['val_acc'].append(accuracy)
        
        # Add client ID to metrics
        metrics["client_id"] = self.client_id
        
        num_val = len(cast(Sized, self.val_loader.dataset))
        return loss, num_val, metrics
    
    def _train_epoch(self, epochs: int = 1) -> Tuple[float, float]:
        """
        Train for specified number of epochs.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Progress bar for training batches
            pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Client {self.client_id} Epoch {epoch+1}/{epochs}",
                leave=False,
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
            
            for batch_idx, (images, labels) in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass with AMP support
                self.optimizer.zero_grad()
                
                if self.use_amp and self.scaler is not None:
                    with _autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    # Backward pass with AMP
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    # Backward pass
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{epoch_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.0*correct/total:.2f}%'
                })
                
            total_loss += epoch_loss
        
        avg_loss = total_loss / (len(self.train_loader) * epochs)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _evaluate(self) -> Tuple[float, float, Dict[str, Scalar]]:
        """
        Evaluate model on validation set.
        
        Returns:
            Tuple of (loss, accuracy, detailed_metrics)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class statistics
        class_correct = {}
        class_total = {}
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store predictions for detailed metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Per-class accuracy
                for label, pred in zip(labels, predicted):
                    label = label.item()
                    if label not in class_correct:
                        class_correct[label] = 0
                        class_total[label] = 0
                    class_total[label] += 1
                    if label == pred.item():
                        class_correct[label] += 1
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        # Compute detailed metrics
        metrics = {
            "accuracy": accuracy,
            "loss": avg_loss,
            "num_samples": total
        }
        
        # Add per-class accuracy
        for cls in class_total:
            metrics[f"class_{cls}_accuracy"] = (
                100.0 * class_correct[cls] / class_total[cls]
                if class_total[cls] > 0 else 0.0
            )
        
        return avg_loss, accuracy, metrics
    
    def get_history(self) -> Dict[str, List[float]]:
        """Return training history."""
        return self.history


def create_client(
    client_id: int,
    model: DSCATNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    **kwargs
) -> SkinCancerClient:
    """
    Factory function to create a Flower client.
    
    Args:
        client_id: Client identifier (1-4 for each dataset)
        model: DSCATNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Computation device
        **kwargs: Additional arguments (local_epochs, learning_rate)
        
    Returns:
        Configured SkinCancerClient
    """
    return SkinCancerClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        **kwargs
    )


def client_fn(client_id: str, model_config: dict, data_config: dict) -> FLClient:
    """
    Client function for Flower's simulation mode.
    
    This function is called by Flower to create client instances.
    
    Args:
        client_id: String client identifier
        model_config: Model configuration dictionary
        data_config: Data configuration dictionary
        
    Returns:
        Flower Client instance
    """
    # This will be implemented when we set up the full simulation
    # For now, return a placeholder
    raise NotImplementedError(
        "client_fn should be implemented with actual data loaders"
    )
