# =============================================================================
# Centralized Training Baseline
# =============================================================================
"""
Centralized Training Baseline.

Provides centralized (non-federated) training for comparison with FL approaches.
This serves as the upper-bound baseline for model performance.
"""

# =============================================================================
# Imports
# =============================================================================

import time
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
# Use torch.amp GradScaler when available; we'll import legacy scaler only as fallback where needed
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from ..models.dscatnet import create_dscatnet
from ..data.datasets import (
    DatasetSubset,
)
from ..data.preprocessing import get_train_transforms, get_val_transforms

# =============================================================================
# AMP Compatibility
# =============================================================================
# Use `torch.amp.autocast` if available (PyTorch >=2.0),
# otherwise fall back to the deprecated `torch.cuda.amp.autocast`.

try:
    _HAS_TORCH_AMP_AUTOCAST = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
except Exception:
    _HAS_TORCH_AMP_AUTOCAST = False


def _autocast():
    """Return appropriate autocast context manager based on PyTorch version."""
    if _HAS_TORCH_AMP_AUTOCAST:
        return torch.amp.autocast("cuda")  # type: ignore[attr-defined]
    return torch.cuda.amp.autocast()


logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CentralizedConfig:
    """Configuration for centralized training."""

    # Model configuration
    model_variant: str = "small"
    num_classes: int = 7
    pretrained: bool = True

    # Training configuration
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_epochs: int = 0
    optimizer_type: str = "adam"  # adam, adamw
    gradient_accumulation_steps: int = 1  # Accumulate gradients for effective larger batch

    # Scheduler configuration
    scheduler_type: str = "none"  # none, cosine, plateau
    min_lr: float = 1e-6

    # Data configuration
    data_root: str = "./data"
    image_size: int = 224
    augmentation_level: str = "medium"
    use_dermoscopy_norm: bool = False
    val_split: float = 0.15
    test_split: float = 0.15

    # Classification mode: 'multiclass' (7), 'multiclass_8' (8), or 'binary' (2)
    classification_mode: str = "multiclass"
    filter_unknown: bool = True
    use_class_weights: bool = True  # Use class weights in loss for imbalance

    # Dataset selection: list of datasets to use, or None/empty for all
    # Valid options: "HAM10000", "ISIC2018", "ISIC2019", "ISIC2020", "PAD-UFES-20"
    datasets: Optional[List[str]] = None

    # Resume training from checkpoint
    resume_from: Optional[str] = None

    # Experiment configuration
    experiment_name: str = "centralized_baseline"
    output_dir: str = "./outputs"
    checkpoint_interval: int = 10
    early_stopping_patience: int = 15

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # Mixed precision (AMP) for faster training on compatible GPUs
    use_amp: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CentralizedConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Centralized Trainer
# =============================================================================


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

        # AMP (Automatic Mixed Precision) for faster training
        self.use_amp = config.use_amp and self.device.type == "cuda"
        if self.use_amp:
            # Avoid direct attribute access to torch.amp.GradScaler (not exported in some stubs)
            amp_mod = getattr(torch, "amp", None)
            scaler_cls = None
            if amp_mod is not None:
                scaler_cls = getattr(amp_mod, "GradScaler", None)

            if scaler_cls is not None:
                try:
                    self.scaler = scaler_cls(device_type="cuda")
                except TypeError:
                    # Some versions may expect different signature; fall back to default construction
                    self.scaler = scaler_cls()
            else:
                # Fallback to legacy scaler for older PyTorch
                from torch.cuda.amp import GradScaler as _GradScaler
                self.scaler = _GradScaler()
        else:
            self.scaler = None

        logger.info(f"Initialized CentralizedTrainer: {config.experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"AMP enabled: {self.use_amp}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _get_transforms(self) -> Tuple[Any, Any]:
        """
        Get train and validation transforms based on config.

        Returns:
            Tuple of (train_transform, val_transform)
        """
        train_transform = get_train_transforms(
            img_size=self.config.image_size,
            augmentation_level=self.config.augmentation_level,
            use_dermoscopy_norm=self.config.use_dermoscopy_norm,
        )
        val_transform = get_val_transforms(
            img_size=self.config.image_size,
            use_dermoscopy_norm=self.config.use_dermoscopy_norm,
        )
        return train_transform, val_transform

    def setup_data(self) -> None:
        """Setup combined dataset from all sources using the DatasetRegistry."""
        from ..data.datasets import (
            DATASET_REGISTRY, get_dataset_paths, normalize_dataset_name, get_available_datasets
        )

        logger.info("Setting up combined dataset for centralized training")

        train_transform, val_transform = self._get_transforms()

        # Load all datasets and split into train/val using indices so transforms
        # can be different for train and val (use DatasetSubset).
        datasets_train = []
        datasets_val = []

        # Determine which datasets to load
        if self.config.datasets:
            dataset_names = [normalize_dataset_name(d) for d in self.config.datasets]
            # Validate all names
            valid_names = get_available_datasets()
            invalid = [n for n in dataset_names if n not in valid_names]
            if invalid:
                raise ValueError(
                    f"Unknown datasets: {invalid}. Valid options: {valid_names}"
                )
            logger.info(f"Using selected datasets: {dataset_names}")
        else:
            dataset_names = get_available_datasets()
            logger.info("Using all available datasets")

        for name in dataset_names:
            config = DATASET_REGISTRY[name]
            csv_path, dataset_root = get_dataset_paths(name, self.config.data_root)

            if csv_path is None or not csv_path.exists():
                logger.warning(f"Dataset {name}: CSV not found")
                continue

            if dataset_root is None or not dataset_root.exists():
                logger.warning(f"Dataset {name}: Image directory not found")
                continue

            try:
                # Instantiate full dataset (with train transforms for now)
                full_dataset = config.dataset_class(
                    root_dir=str(dataset_root),
                    csv_path=str(csv_path),
                    transform=train_transform,
                    classification_mode=self.config.classification_mode,
                    filter_unknown=self.config.filter_unknown
                )
            except Exception as e:
                logger.warning(f"Failed loading dataset {name}: {e}")
                continue

            n = len(full_dataset)
            if n == 0:
                logger.warning(f"Dataset {name} contains 0 samples")
                continue

            # Compute split sizes
            val_n = int(n * self.config.val_split)
            train_n = n - val_n

            # Reproducible random permutation
            gen = torch.Generator()
            gen.manual_seed(42)
            indices = torch.randperm(n, generator=gen).tolist()

            train_indices = indices[:train_n]
            val_indices = indices[train_n:]

            train_ds = DatasetSubset(full_dataset, train_indices, train_transform)
            val_ds = DatasetSubset(full_dataset, val_indices, val_transform)

            datasets_train.append(train_ds)
            datasets_val.append(val_ds)
            logger.info(f"Loaded {name}: {len(train_ds)} train, {len(val_ds)} val")

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
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            combined_val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        # Compute class weights if needed
        if self.config.use_class_weights:
            self._compute_class_weights(combined_train)
        else:
            self.class_weights = None

        logger.info(f"Combined dataset: {len(combined_train)} train, {len(combined_val)} val")

    def _compute_class_weights(self, dataset: ConcatDataset) -> None:
        """Compute class weights for handling class imbalance."""
        # Count labels across all sub-datasets
        all_labels: List[int] = []
        for sub_ds in dataset.datasets:
            if isinstance(sub_ds, DatasetSubset):
                for idx in sub_ds.indices:
                    all_labels.append(sub_ds.dataset.labels[idx])
            elif hasattr(sub_ds, 'labels'):
                labels = getattr(sub_ds, 'labels')
                all_labels.extend(labels)

        label_counts = Counter(all_labels)
        total = sum(label_counts.values())
        num_classes = self.config.num_classes

        # Compute inverse frequency weights
        weights = torch.zeros(num_classes)
        for cls, count in label_counts.items():
            if 0 <= cls < num_classes:
                weights[cls] = total / (num_classes * count)

        self.class_weights = weights.to(self.device)
        logger.info(f"Class weights: {dict(enumerate(weights.tolist()))}")

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """
        Train for one epoch with optional gradient accumulation.

        Supports gradient accumulation to simulate larger batch sizes
        when GPU memory is limited. Effective batch size =
        batch_size * gradient_accumulation_steps.

        Returns:
            Tuple of (average loss, accuracy).

        Raises:
            RuntimeError: If train_loader is not initialized (setup_data() not called).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        if self.train_loader is None:
            raise RuntimeError("train_loader is not initialized. Call setup_data() before training.")

        loader = self.train_loader
        accum_steps = self.config.gradient_accumulation_steps

        # Progress bar for batches
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc="Training",
            leave=False,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        optimizer.zero_grad()

        for batch_idx, (images, labels) in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Use AMP for faster training on compatible GPUs
            if self.use_amp and self.scaler is not None:
                with _autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    # Scale loss by accumulation steps for correct gradient magnitude
                    loss = loss / accum_steps
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                    # Gradient clipping with AMP
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                # Scale loss by accumulation steps for correct gradient magnitude
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            # Track unscaled loss for logging
            total_loss += loss.item() * accum_steps if (self.use_amp and self.scaler) else loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar with current metrics
            current_loss = total_loss / (batch_idx + 1)
            current_acc = correct / total if total > 0 else 0
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, Dict[str, float]]:
        """
        Evaluate model on validation set.

        Returns:
            Tuple of (loss, accuracy, per-class metrics).

        Raises:
            RuntimeError: If val_loader is not initialized (setup_data() not called).
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

        # Progress bar for validation
        pbar = tqdm(
            loader,
            desc="Validating",
            leave=False,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )

        for images, labels in pbar:
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

    def save_checkpoint(self, epoch: int, optimizer, scheduler, metrics: Dict, is_best: bool = False) -> str:
        """Save training checkpoint with all state needed for resumption.

        Args:
            epoch: Current epoch number
            optimizer: Optimizer state
            scheduler: Scheduler state (can be None)
            metrics: Current metrics dict
            is_best: If True, save as best_checkpoint.pt

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "metrics": metrics,
            "config": self.config.to_dict(),
            "history": self.history,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

        if is_best:
            path = self.checkpoint_dir / "best_checkpoint.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)

        # Also save model-only file for easy loading
        if is_best:
            model_only_path = self.checkpoint_dir / "best_model.pt"
            torch.save(self.model.state_dict(), model_only_path)

        logger.debug(f"Saved checkpoint: {path}")
        return str(path)

    def load_checkpoint(self, checkpoint_path: str, optimizer=None, scheduler=None) -> int:
        """Load checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optimizer to restore state to (optional)
            scheduler: Scheduler to restore state to (optional)

        Returns:
            Epoch number to resume from
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore training state
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        if "best_val_accuracy" in checkpoint:
            self.best_val_accuracy = checkpoint["best_val_accuracy"]
        if "best_epoch" in checkpoint:
            self.best_epoch = checkpoint["best_epoch"]
        if "epochs_without_improvement" in checkpoint:
            self.epochs_without_improvement = checkpoint["epochs_without_improvement"]

        epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resumed from epoch {epoch}, best accuracy: {self.best_val_accuracy:.4f}")
        return epoch


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
        if self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            # Default: Adam (matches DSCATNet paper)
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        # Setup scheduler
        scheduler = None
        if self.config.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr,
            )
        # scheduler_type == "none": no scheduler

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # Log effective batch size
        effective_bs = self.config.batch_size * self.config.gradient_accumulation_steps
        logger.info(
            f"Optimizer: {self.config.optimizer_type}, LR: {self.config.learning_rate}, "
            f"Weight Decay: {self.config.weight_decay}"
        )
        logger.info(
            f"Batch size: {self.config.batch_size}, Gradient accumulation: "
            f"{self.config.gradient_accumulation_steps}, Effective batch size: {effective_bs}"
        )
        logger.info(f"Scheduler: {self.config.scheduler_type}, AMP: {self.use_amp}")

        # Resume from checkpoint if specified
        start_epoch = 1
        if self.config.resume_from:
            resume_path = Path(self.config.resume_from)
            if resume_path.exists():
                start_epoch = self.load_checkpoint(str(resume_path), optimizer, scheduler) + 1
                logger.info(f"Resuming training from epoch {start_epoch}")
            else:
                logger.warning(f"Checkpoint not found at {resume_path}, starting from scratch")

        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Training loop
        start_time = time.time()

        # Epoch progress bar
        epoch_pbar = tqdm(
            range(start_epoch, self.config.num_epochs + 1),
            desc="Epochs",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )

        for epoch in epoch_pbar:
            epoch_start = time.time()

            epoch_pbar.set_description(f"Epoch {epoch}/{self.config.num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)

            # Evaluate
            val_loss, val_acc, per_class = self.evaluate()

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Update scheduler (call with metric only for ReduceLROnPlateau)
            if scheduler is not None:
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

            # Update epoch progress bar with metrics
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_acc': f'{val_acc:.4f}',
                'best': f'{self.best_val_accuracy:.4f}'
            })

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
                # Save best checkpoint (full state for resumption)
                self.save_checkpoint(epoch, optimizer, scheduler, metrics, is_best=True)
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
