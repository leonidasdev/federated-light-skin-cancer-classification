"""
Centralized Training Baseline.

Provides centralized (non-federated) training for comparison with FL approaches.
This serves as the upper-bound baseline for model performance.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from ..models.dscatnet import create_dscatnet
from ..data.datasets import (
    HAM10000Dataset,
    ISIC2018Dataset,
    ISIC2019Dataset,
    ISIC2020Dataset,
)
from ..data.preprocessing import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


@dataclass
class CentralizedConfig:
    """Configuration for centralized training."""
    
    # Model configuration
    model_variant: str = "small"
    num_classes: int = 7
    pretrained: bool = True
    
    # Training configuration
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # Scheduler configuration
    scheduler_type: str = "cosine"  # cosine, plateau
    min_lr: float = 1e-6
    
    # Data configuration
    data_root: str = "./data"
    image_size: int = 224
    augmentation_level: str = "medium"
    use_dermoscopy_norm: bool = False
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Experiment configuration
    experiment_name: str = "centralized_baseline"
    output_dir: str = "./outputs"
    checkpoint_interval: int = 10
    early_stopping_patience: int = 15
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CentralizedConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class CentralizedTrainer:
    """
    Centralized Training for DSCATNet.
    
    Trains on combined data from all datasets as a baseline for comparison
    with federated learning approaches.
    """
    
    def __init__(self, config: CentralizedConfig):
        """
        Initialize the centralized trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = create_dscatnet(
            variant=config.model_variant,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
        ).to(self.device)
        
        # Training history
        self.history = {
            "epochs": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }
        
        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Data loaders (to be setup)
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        
        logger.info(f"Initialized CentralizedTrainer: {config.experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_data(self) -> None:
        """Setup combined dataset from all sources."""
        logger.info("Setting up combined dataset for centralized training")
        
        train_transform = get_train_transforms(
            img_size=self.config.image_size,
            augmentation_level=self.config.augmentation_level,
            use_dermoscopy_norm=self.config.use_dermoscopy_norm,
        )
        val_transform = get_val_transforms(
            img_size=self.config.image_size,
            use_dermoscopy_norm=self.config.use_dermoscopy_norm,
        )
        
        # Load all datasets
        datasets_train = []
        datasets_val = []
        
        dataset_classes = [
            (HAM10000Dataset, "ham10000"),
            (ISIC2018Dataset, "isic2018"),
            (ISIC2019Dataset, "isic2019"),
            (ISIC2020Dataset, "isic2020"),
        ]
        
        for dataset_cls, name in dataset_classes:
            data_path = Path(self.config.data_root) / name
            try:
                train_ds = dataset_cls(root=str(data_path), split="train", transform=train_transform)
                val_ds = dataset_cls(root=str(data_path), split="val", transform=val_transform)
                datasets_train.append(train_ds)
                datasets_val.append(val_ds)
                logger.info(f"Loaded {name}: {len(train_ds)} train, {len(val_ds)} val")
            except FileNotFoundError:
                logger.warning(f"Dataset {name} not found at {data_path}")
        
        if not datasets_train:
            raise RuntimeError("No datasets found. Please check data paths.")
        
        # Combine datasets
        combined_train = ConcatDataset(datasets_train)
        combined_val = ConcatDataset(datasets_val)
        
        # Create data loaders
        self.train_loader = DataLoader(
            combined_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            combined_val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        
        logger.info(f"Combined dataset: {len(combined_train)} train, {len(combined_val)} val")
    
    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        if self.train_loader is None:
            raise RuntimeError("train_loader is not initialized. Call setup_data() before training.")

        loader = self.train_loader

        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:
                total_batches = len(loader)
                logger.debug(f"Batch {batch_idx}/{total_batches}: loss={loss.item():.4f}")

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, Dict[str, float]]:
        """
        Evaluate model on validation set.
        
        Returns:
            Tuple of (loss, accuracy, per-class metrics).
        """
        self.model.eval()

        if self.val_loader is None:
            raise RuntimeError("val_loader is not initialized. Call setup_data() before evaluation.")

        loader = self.val_loader

        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        # Per-class tracking
        class_correct = {}
        class_total = {}

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                label = int(label)
                class_total.setdefault(label, 0)
                class_correct.setdefault(label, 0)
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        per_class = {
            f"class_{k}_acc": (class_correct[k] / class_total[k]) if class_total[k] > 0 else 0.0
            for k in class_total.keys()
        }

        return avg_loss, accuracy, per_class
    
    def save_checkpoint(self, epoch: int, optimizer, scheduler, metrics: Dict) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": self.config.to_dict(),
        }
        
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")
    
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete centralized training.
        
        Returns:
            Dictionary with training history and results.
        """
        logger.info("Starting centralized training")

        # Setup data
        self.setup_data()

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Setup scheduler
        if self.config.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.min_lr,
            )
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr,
            )

        criterion = nn.CrossEntropyLoss()

        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Training loop
        start_time = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)

            # Evaluate
            val_loss, val_acc, per_class = self.evaluate()

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Update scheduler (call with metric only for ReduceLROnPlateau)
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

            epoch_time = time.time() - epoch_start

            # Update history
            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["learning_rate"].append(current_lr)

            logger.info(
                f"Epoch {epoch}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Checkpointing
            metrics = {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
            }

            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch, optimizer, scheduler, metrics)

            # Best model tracking
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = float(val_acc)
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                # Save best model
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(self.model.state_dict(), best_path)
                logger.info(f"Saved best model (epoch {epoch}, acc={self.best_val_accuracy:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info("Early stopping triggered")
                break

        total_time = time.time() - start_time

        # Save final results
        results = {
            "history": self.history,
            "best_val_accuracy": float(self.best_val_accuracy),
            "best_epoch": int(self.best_epoch),
            "total_time_seconds": total_time,
        }

        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Training complete. Best accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch}")
        logger.info(f"Total time: {total_time/60:.2f} minutes")

        return results


def run_centralized_training(config: Optional[CentralizedConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to run centralized training.
    
    Args:
        config: Training configuration. If None, uses defaults.
        
    Returns:
        Training results.
    """
    if config is None:
        config = CentralizedConfig()
    
    trainer = CentralizedTrainer(config)
    return trainer.run()
