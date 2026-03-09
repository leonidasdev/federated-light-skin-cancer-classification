# =============================================================================
# Flower FL Client for Skin Cancer Classification
# =============================================================================
"""
Flower FL Client for Skin Cancer Classification.

Each client represents a hospital or institution with its own dermoscopy
dataset, following the natural non-IID partition where each site
contributes data from a different imaging source.

Training protocol per client follows Yadav et al. (PLOS ONE 2024):
Adam optimizer, LR=0.001, standard cross-entropy loss.
Gradient accumulation enables effective batch sizes larger than
what GPU memory allows.
"""

# =============================================================================
# Imports
# =============================================================================

from typing import cast
from collections.abc import Sized

import torch
from torch import nn
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

from ..models.dscatnet import DSCATNet, get_model_parameters, set_model_parameters
from ..utils.helpers import autocast, create_grad_scaler


# =============================================================================
# FL Client Implementation
# =============================================================================


class SkinCancerClient(NumPyClient):
    """
    Flower client for skin cancer classification with DSCATNet.

    This client handles local training and evaluation for a single
    dermoscopy dataset in the federated learning setup.

    Args:
        client_id: Unique identifier for this client (0-indexed, one per dataset).
        model: DSCATNet model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to run training on.
        local_epochs: Number of local training epochs per round.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer (paper-aligned default: 0.0).
        class_weights: Optional class weights for imbalanced data.
        use_amp: Enable Automatic Mixed Precision (AMP) for faster training.
        scheduler_type: LR scheduler type ("none" or "cosine").
        scheduler_t_max: T_max for CosineAnnealingLR scheduler (typically num_rounds).
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
        weight_decay: float = 0.0,
        class_weights: torch.Tensor | None = None,
        use_amp: bool = True,
        scheduler_type: str = "none",
        scheduler_t_max: int = 100,
        max_grad_norm: float | None = None,
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.scheduler_t_max = scheduler_t_max
        self.max_grad_norm = max_grad_norm

        # Move model to device
        self.model.to(self.device)

        # AMP (Automatic Mixed Precision) for faster training
        self.use_amp = use_amp and device.type == "cuda"
        if self.use_amp:
            self.scaler = create_grad_scaler()
        else:
            self.scaler = None

        # Loss function with optional class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer (paper-aligned: Adam with weight_decay=0.0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler for learning rate decay (paper-aligned: "none")
        self.scheduler_type = scheduler_type
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.scheduler_t_max,
                eta_min=1e-6,
            )
        else:
            self.scheduler = None

        # Training history
        self.history: dict[str, list[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return current model parameters as numpy arrays."""
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from numpy arrays."""
        set_model_parameters(self.model, parameters)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
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

        # Step scheduler (if configured)
        if self.scheduler is not None:
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
        config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
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

    def _train_epoch(self, epochs: int = 1) -> tuple[float, float]:
        """
        Train for specified number of epochs.

        Args:
            epochs: Number of local training epochs.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _epoch in range(epochs):
            epoch_loss = 0.0

            for _batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass with AMP support
                self.optimizer.zero_grad()

                if self.use_amp and self.scaler is not None:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    # Backward pass with AMP
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    # Backward pass
                    loss.backward()
                    # Gradient clipping for stability
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()

                # Statistics
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            total_loss += epoch_loss

        avg_loss = total_loss / (len(self.train_loader) * epochs)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _evaluate(self) -> tuple[float, float, dict[str, Scalar]]:
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
        accuracy = correct / total if total > 0 else 0.0

        # Compute detailed metrics
        metrics = {
            "accuracy": accuracy,
            "loss": avg_loss,
            "num_samples": total
        }

        # Add per-class accuracy
        for cls, cls_count in class_total.items():
            metrics[f"class_{cls}_accuracy"] = (
                class_correct[cls] / cls_count
                if cls_count > 0 else 0.0
            )

        return avg_loss, accuracy, metrics

    def get_history(self) -> dict[str, list[float]]:
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
        client_id: Client identifier (0-indexed, one per dataset)
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
