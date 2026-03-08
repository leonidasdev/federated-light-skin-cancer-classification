# =============================================================================
# Federated Learning Simulation Module
# =============================================================================
"""
Federated Learning Simulation Module.

This module provides the complete FL simulation infrastructure for running
federated experiments with DSCATNet on dermoscopy datasets.
"""

# =============================================================================
# Imports
# =============================================================================

import os
import sys
import time
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from flwr.common import Scalar

from ..models.dscatnet import create_dscatnet, get_model_parameters, set_model_parameters
from ..data.datasets import (
    DATASET_REGISTRY,
    DatasetSubset,
    get_available_datasets,
    get_dataset_paths,
    normalize_dataset_name,
)
from ..data.preprocessing import get_transform_pair
from ..data.splits import create_noniid_split, deterministic_train_val_split
from ..utils.helpers import compute_class_weights

logger = logging.getLogger(__name__)

# =============================================================================
# Helper Classes
# =============================================================================


class DirichletSubset(torch.utils.data.Dataset):
    """
    Dataset wrapper for Dirichlet split subsets.

    This class wraps combined dataset references and provides proper
    indexing for samples assigned to a specific client.
    """

    def __init__(
        self,
        combined_images: List[Tuple[Any, int]],
        indices: List[int],
        transform: Optional[Any] = None
    ):
        """
        Initialize DirichletSubset.

        Args:
            combined_images: List of (dataset, original_idx) tuples
            indices: Indices into combined_images for this subset
            transform: Optional transform to apply to images
        """
        self.combined_images = combined_images
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample at index."""
        combined_idx = self.indices[idx]
        dataset, original_idx = self.combined_images[combined_idx]

        # Get the original sample
        image, label = dataset[original_idx]

        # Apply transform if different from dataset's transform
        if self.transform is not None and hasattr(dataset, 'transform'):
            # Re-load the raw image and apply our transform
            # This handles the case where we need val transforms
            if hasattr(dataset, 'img_paths'):
                img_path = dataset.img_paths[original_idx]
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)

        return image, label


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SimulationConfig:
    """Configuration for FL simulation."""

    # Model configuration
    model_variant: str = "small"
    num_classes: int = 7
    pretrained: bool = True

    # FL configuration
    num_clients: int = 4
    num_rounds: int = 50
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2

    # Client selection: fraction of clients to sample each round (0.0-1.0)
    # 1.0 = all clients participate, 0.5 = 50% of clients randomly selected
    client_selection_fraction: float = 1.0

    # Training configuration
    local_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer_type: str = "adam"  # adam, adamw
    gradient_accumulation_steps: int = 1  # Accumulate gradients for effective larger batch

    # Data configuration
    data_root: str = "./data"
    image_size: int = 224
    augmentation_level: str = "medium"
    use_dermoscopy_norm: bool = False
    train_val_split: float = 0.8  # Fraction of data for training (rest for validation)

    # Non-IID configuration
    noniid_type: str = "natural"  # natural, dirichlet, label_skew, quantity_skew
    dirichlet_alpha: float = 0.5

    # Dataset selection: list of datasets to use, or None/empty for all
    # Valid options: "HAM10000", "ISIC2018", "ISIC2019", "ISIC2020", "PAD-UFES-20"
    # For natural non-IID, each selected dataset becomes one client
    datasets: Optional[List[str]] = None

    # Resume training from checkpoint
    resume_from: Optional[str] = None

    # Experiment configuration
    experiment_name: str = "fl_experiment"
    output_dir: str = "./outputs"
    checkpoint_interval: int = 5
    early_stopping_patience: int = 10

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2

    # Class weights for handling class imbalance in loss function
    use_class_weights: bool = True

    # Parallelism configuration
    # Number of clients to train in parallel (CPU only, for GPU use 1)
    # Set to 0 for auto-detection based on CPU count
    parallel_clients: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class ClientData:
    """Data container for a single FL client.

    Attributes:
        client_id: Integer identifier for this client.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_train_samples: Number of training samples.
        num_val_samples: Number of validation samples.
        class_distribution: Mapping of class index to sample count.
        dataset_name: Name of the source dataset.
    """

    client_id: int
    train_loader: DataLoader
    val_loader: DataLoader
    num_train_samples: int
    num_val_samples: int
    class_distribution: Dict[int, int]
    dataset_name: str


# =============================================================================
# FL Simulator
# =============================================================================


class FLSimulator:
    """
    Federated Learning Simulator for DSCATNet.

    Orchestrates the complete FL training process including client setup,
    data distribution, training rounds, and evaluation.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the FL simulator.

        Args:
            config: Simulation configuration.
        """
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        self.global_model = create_dscatnet(
            variant=config.model_variant,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
        ).to(self.device)

        # Client data
        self.client_data: Dict[int, ClientData] = {}

        # Class weights for handling imbalanced data (computed after setup)
        self.class_weights: Optional[torch.Tensor] = None

        # Training history
        self.history = {
            "rounds": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "communication_cost": [],
        }

        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoints directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_round = 0
        self.rounds_without_improvement = 0

        logger.info(f"Initialized FLSimulator with config: {config.experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

    def _get_transforms(self) -> Tuple[Any, Any]:
        """Get train and validation transforms based on config."""
        return get_transform_pair(
            img_size=self.config.image_size,
            augmentation_level=self.config.augmentation_level,
            use_dermoscopy_norm=self.config.use_dermoscopy_norm,
        )

    def _resolve_datasets(
        self, transform: Any
    ) -> List[Tuple[str, Any]]:
        """Resolve and load datasets from the registry.

        Returns a list of (dataset_name, full_dataset) tuples for every
        dataset that was successfully loaded.
        """
        if self.config.datasets:
            dataset_names = [normalize_dataset_name(d) for d in self.config.datasets]
            valid_names = get_available_datasets()
            invalid = [n for n in dataset_names if n not in valid_names]
            if invalid:
                raise ValueError(
                    f"Unknown datasets: {invalid}. Valid options: {valid_names}"
                )
        else:
            dataset_names = get_available_datasets()

        loaded: List[Tuple[str, Any]] = []
        for dataset_name in dataset_names:
            config = DATASET_REGISTRY[dataset_name]
            csv_path, dataset_root = get_dataset_paths(dataset_name, self.config.data_root)

            if csv_path is None or not csv_path.exists():
                logger.warning(f"Dataset {dataset_name}: CSV not found")
                continue
            if dataset_root is None or not dataset_root.exists():
                logger.warning(f"Dataset {dataset_name}: Image directory not found")
                continue

            try:
                full_dataset = config.dataset_class(
                    root_dir=str(dataset_root),
                    csv_path=str(csv_path),
                    transform=transform,
                )
            except Exception as e:
                logger.warning(f"Failed loading dataset {dataset_name}: {e}")
                continue

            if len(full_dataset) == 0:
                logger.warning(f"Dataset {dataset_name} contains 0 samples")
                continue

            loaded.append((dataset_name, full_dataset))
            logger.info(f"Loaded {dataset_name}: {len(full_dataset)} samples")

        return loaded

    def setup_natural_noniid(self) -> None:
        """
        Setup natural non-IID: each client gets a different dataset.

        Uses the DATASET_REGISTRY for centralized path resolution.

        By default:
        - Client 0: HAM10000
        - Client 1: ISIC 2018
        - Client 2: ISIC 2019
        - Client 3: ISIC 2020

        If config.datasets is specified, only those datasets are used.
        """
        logger.info("Setting up natural non-IID distribution (each client = different dataset)")

        train_transform, val_transform = self._get_transforms()
        loaded_datasets = self._resolve_datasets(train_transform)

        for client_id, (dataset_name, full_dataset) in enumerate(loaded_datasets):
            if client_id >= self.config.num_clients:
                break

            train_indices, val_indices = deterministic_train_val_split(
                len(full_dataset), val_split=1.0 - self.config.train_val_split
            )

            train_dataset = DatasetSubset(full_dataset, train_indices, train_transform)
            val_dataset = DatasetSubset(full_dataset, val_indices, val_transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=(self.device.type == "cuda"),
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=(self.device.type == "cuda"),
            )

            # Calculate class distribution
            class_dist = {}
            for _, label in train_dataset:
                class_dist[label] = class_dist.get(label, 0) + 1

            self.client_data[client_id] = ClientData(
                client_id=client_id,
                train_loader=train_loader,
                val_loader=val_loader,
                num_train_samples=len(train_dataset),
                num_val_samples=len(val_dataset),
                class_distribution=class_dist,
                dataset_name=dataset_name,
            )

            logger.info(f"Client {client_id} ({dataset_name}): {len(train_dataset)} train, {len(val_dataset)} val samples")

    def setup_clients(self) -> None:
        """Setup client data based on configuration."""
        if self.config.noniid_type == "natural":
            self.setup_natural_noniid()
        elif self.config.noniid_type in ["dirichlet", "label_skew", "quantity_skew"]:
            # For synthetic non-IID, load and combine requested datasets, then split
            self.setup_dirichlet_noniid()
        else:
            logger.warning(f"Unknown noniid_type: {self.config.noniid_type}, using natural non-IID")
            self.setup_natural_noniid()

        # Compute class weights after client setup
        if self.config.use_class_weights:
            self._compute_class_weights()
        else:
            self.class_weights = None

    def _compute_class_weights(self) -> None:
        """Compute global class weights from all clients' class distributions.

        Uses inverse frequency weighting: weight_c = N_total / (C * N_c)
        where N_total is the total number of training samples, C is the
        number of classes, and N_c is the number of samples in class c.
        """
        global_counts: Counter = Counter()
        for client in self.client_data.values():
            for cls, count in client.class_distribution.items():
                global_counts[int(cls)] += count

        self.class_weights = compute_class_weights(
            dict(global_counts), self.config.num_classes
        ).to(self.device)
        logger.info(f"Class weights: {dict(enumerate(self.class_weights.tolist()))}")
    def setup_dirichlet_noniid(self) -> None:
        """
        Setup Dirichlet non-IID: split dataset(s) across clients using Dirichlet distribution.

        Uses the DATASET_REGISTRY for centralized path resolution.

        This creates heterogeneous label distributions across clients.
        Lower alpha = more heterogeneous (more non-IID)
        Higher alpha = more homogeneous (closer to IID)
        """
        logger.info(f"Setting up Dirichlet non-IID with alpha={self.config.dirichlet_alpha}")

        train_transform, val_transform = self._get_transforms()
        loaded_datasets = self._resolve_datasets(train_transform)

        # Combine all loaded datasets
        combined_images = []
        combined_labels = []

        for _dataset_name, full_dataset in loaded_datasets:
            for i in range(len(full_dataset)):
                combined_images.append((full_dataset, i))
                combined_labels.append(full_dataset.labels[i])

        if not combined_labels:
            raise RuntimeError("No data loaded. Please check dataset paths.")

        total_samples = len(combined_labels)
        logger.info(f"Total samples for Dirichlet split: {total_samples}")

        # Create Dirichlet split
        client_indices = create_noniid_split(
            labels=combined_labels,
            num_clients=self.config.num_clients,
            alpha=self.config.dirichlet_alpha,
        )

        # Create client data loaders
        for client_id, indices in client_indices.items():
            if len(indices) == 0:
                logger.warning(f"Client {client_id} has no samples, skipping")
                continue

            # Split into train/val using configurable ratio
            # Seed per-client for reproducibility on checkpoint resume
            rng = np.random.RandomState(42 + client_id)
            rng.shuffle(indices)
            split_idx = int(len(indices) * self.config.train_val_split)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            # Create wrapper datasets
            train_dataset = DirichletSubset(combined_images, train_indices, train_transform)
            val_dataset = DirichletSubset(combined_images, val_indices, val_transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=(self.device.type == "cuda"),
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=(self.device.type == "cuda"),
            )

            # Class distribution for this client
            class_dist = {}
            for idx in train_indices:
                label = combined_labels[idx]
                class_dist[int(label)] = class_dist.get(int(label), 0) + 1

            self.client_data[client_id] = ClientData(
                client_id=client_id,
                train_loader=train_loader,
                val_loader=val_loader,
                num_train_samples=len(train_indices),
                num_val_samples=len(val_indices),
                class_distribution=class_dist,
                dataset_name=f"dirichlet_client_{client_id}",
            )

            logger.info(f"Client {client_id}: {len(train_indices)} train, {len(val_indices)} val samples")
            logger.info(f"  Class distribution: {class_dist}")

    def train_client(
        self,
        client_id: int,
        model_parameters: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train a single client for local epochs.

        Args:
            client_id: Client identifier.
            model_parameters: Current global model parameters.

        Returns:
            Tuple of (updated parameters, num samples, metrics dict).

        Raises:
            ValueError: If client_id is not found in client_data.
        """
        if client_id not in self.client_data:
            raise ValueError(f"Client {client_id} not found")

        client = self.client_data[client_id]

        # Create local model and load parameters
        local_model = create_dscatnet(
            variant=self.config.model_variant,
            num_classes=self.config.num_classes,
            pretrained=False,
        ).to(self.device)
        set_model_parameters(local_model, model_parameters)

        # Setup optimizer
        if self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                local_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            # Default: Adam (matches DSCATNet paper)
            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        # Use class weights if available for handling class imbalance
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        accum_steps = self.config.gradient_accumulation_steps

        # Training
        local_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _epoch in range(self.config.local_epochs):
            optimizer.zero_grad()
            for batch_idx, (images, labels) in enumerate(client.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = local_model(images)
                loss = criterion(outputs, labels)
                # Scale loss by accumulation steps for correct gradient magnitude
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(client.train_loader):
                    # Gradient clipping for training stability
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calculate metrics
        avg_loss = total_loss / (len(client.train_loader) * self.config.local_epochs)
        accuracy = correct / total if total > 0 else 0.0

        metrics = {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "dataset": client.dataset_name,
        }

        return get_model_parameters(local_model), client.num_train_samples, metrics

    def evaluate_client(
        self,
        client_id: int,
        model_parameters: List[np.ndarray],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate model on a single client's validation data.

        Args:
            client_id: Client identifier.
            model_parameters: Model parameters to evaluate.

        Returns:
            Tuple of (loss, num samples, metrics dict).

        Raises:
            ValueError: If client_id is not found in client_data.
        """
        if client_id not in self.client_data:
            raise ValueError(f"Client {client_id} not found")

        client = self.client_data[client_id]

        # Create model and load parameters
        model = create_dscatnet(
            variant=self.config.model_variant,
            num_classes=self.config.num_classes,
            pretrained=False,
        ).to(self.device)
        set_model_parameters(model, model_parameters)

        model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in client.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        metrics = {
            "val_accuracy": accuracy,
            "dataset": client.dataset_name,
        }

        return avg_loss, client.num_val_samples, metrics

    def aggregate_parameters(
        self,
        results: List[Tuple[List[np.ndarray], int]],
    ) -> List[np.ndarray]:
        """
        Aggregate parameters using FedAvg.

        Args:
            results: List of (parameters, num_samples) tuples.

        Returns:
            Aggregated parameters.
        """
        total_samples = sum(num_samples for _, num_samples in results)

        # Initialize with zeros - results[0][0] is the list of params from first client
        first_client_params = results[0][0]
        aggregated = [np.zeros_like(param) for param in first_client_params]

        for params, num_samples in results:
            weight = num_samples / total_samples
            for i, param in enumerate(params):
                aggregated[i] += param * weight

        return aggregated

    def _select_clients(self, round_num: int) -> List[int]:
        """
        Select clients for this round based on client_selection_fraction.

        Args:
            round_num: Current round number (used as seed for reproducibility).

        Returns:
            List of selected client IDs.
        """
        all_client_ids = list(self.client_data.keys())
        n_clients = len(all_client_ids)

        # Full participation
        if self.config.client_selection_fraction >= 1.0:
            return all_client_ids

        # Calculate number of clients to select
        n_selected = max(
            self.config.min_fit_clients,
            int(n_clients * self.config.client_selection_fraction)
        )
        n_selected = min(n_selected, n_clients)

        # Use round number as seed for reproducibility
        rng = np.random.RandomState(round_num + 42)
        selected = rng.choice(all_client_ids, size=n_selected, replace=False).tolist()

        logger.info(f"Round {round_num}: Selected {len(selected)}/{n_clients} clients: {selected}")
        return selected

    def _get_parallel_workers(self) -> int:
        """
        Determine the number of parallel workers for client training.

        Returns:
            Number of workers to use (1 for sequential, >1 for parallel).
        """
        if self.config.parallel_clients == 0:
            # Auto-detect: use CPU count, but cap at 4 to avoid memory issues
            return min(os.cpu_count() or 1, 4)
        return self.config.parallel_clients

    def run_round(self, round_num: int, pbar: Optional[tqdm] = None) -> Dict[str, float]:
        """
        Run a single FL round with optional parallel client training.

        Executes one complete round of federated learning: client selection,
        local training, and model aggregation. Supports partial client
        participation via client_selection_fraction and parallel training
        via parallel_clients configuration.

        Args:
            round_num: Current round number (1-indexed).
            pbar: Optional tqdm progress bar to update with status.

        Returns:
            Dictionary containing aggregated metrics:
                - train_loss (float): Weighted average training loss across clients
                - train_accuracy (float): Weighted average training accuracy
                - val_loss (float): Validation loss on aggregated model
                - val_accuracy (float): Validation accuracy on aggregated model
                - communication_cost_mb (float): MB of data transferred this round
        """
        start_time = time.time()

        # Get current global parameters
        global_params = get_model_parameters(self.global_model)

        # Select clients for this round
        selected_clients = self._select_clients(round_num)
        n_workers = self._get_parallel_workers()

        # Client training
        fit_results = []
        client_train_metrics = []

        if n_workers > 1 and len(selected_clients) > 1 and self.device.type == "cpu":
            # Parallel training (CPU only - GPU parallelism needs special handling)
            if pbar:
                pbar.set_postfix_str(f"Training {len(selected_clients)} clients (parallel)...")

            # Note: We use ThreadPoolExecutor because model training is CPU-bound but
            # PyTorch operations release the GIL. For true multiprocessing, we'd need
            # to serialize models which adds overhead.
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.train_client, cid, global_params): cid
                    for cid in selected_clients
                }
                for future in as_completed(futures):
                    client_id = futures[future]
                    try:
                        params, num_samples, metrics = future.result()
                        fit_results.append((params, num_samples))
                        client_train_metrics.append(metrics)
                        logger.debug(
                            f"Client {client_id}: loss={metrics['train_loss']:.4f}, "
                            f"acc={metrics['train_accuracy']:.4f}"
                        )
                    except Exception as e:
                        logger.error(f"Client {client_id} training failed: {e}")
        else:
            # Sequential training (default for GPU or single worker)
            for client_id in selected_clients:
                client = self.client_data[client_id]
                if pbar:
                    pbar.set_postfix_str(f"Training {client.dataset_name}...")
                params, num_samples, metrics = self.train_client(client_id, global_params)
                fit_results.append((params, num_samples))
                client_train_metrics.append(metrics)
                logger.debug(
                    f"Client {client_id}: loss={metrics['train_loss']:.4f}, "
                    f"acc={metrics['train_accuracy']:.4f}"
                )

        if not fit_results:
            raise RuntimeError("No clients completed training")

        # Aggregate parameters
        if pbar:
            pbar.set_postfix_str("Aggregating...")
        aggregated_params = self.aggregate_parameters(fit_results)
        set_model_parameters(self.global_model, aggregated_params)

        # Client evaluation (always on all clients for consistent metrics)
        eval_results = []
        client_val_metrics = []

        for client_id in self.client_data.keys():
            loss, num_samples, metrics = self.evaluate_client(client_id, aggregated_params)
            eval_results.append((loss, num_samples, metrics))
            client_val_metrics.append(metrics)

        # Aggregate metrics (sample-weighted for FedAvg consistency)
        total_train_samples = sum(r[1] for r in fit_results)
        total_val_samples = sum(r[1] for r in eval_results)

        avg_train_loss = sum(
            m["train_loss"] * r[1] for m, r in zip(client_train_metrics, fit_results)
        ) / total_train_samples
        avg_train_acc = sum(
            m["train_accuracy"] * r[1] for m, r in zip(client_train_metrics, fit_results)
        ) / total_train_samples
        avg_val_loss = sum(r[0] * r[1] for r in eval_results) / total_val_samples
        avg_val_acc = sum(
            m["val_accuracy"] * r[1] for m, r in zip(client_val_metrics, eval_results)
        ) / total_val_samples

        round_time = time.time() - start_time

        # Calculate communication cost (model size * 2 * num_selected_clients)
        model_size_bytes = sum(p.nbytes for p in global_params)
        comm_cost = model_size_bytes * 2 * len(selected_clients)

        metrics = {
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_acc,
            "round_time": round_time,
            "communication_cost_mb": comm_cost / (1024 * 1024),
            "clients_participated": len(selected_clients),
        }

        return metrics

    def save_checkpoint(self, round_num: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint with full state for resumption.

        Args:
            round_num: Current FL round number.
            metrics: Metrics dict for this round.
        """
        checkpoint = {
            "round": round_num,
            "model_state_dict": self.global_model.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
            "history": self.history,
            "best_val_accuracy": self.best_val_accuracy,
            "best_round": self.best_round,
            "rounds_without_improvement": self.rounds_without_improvement,
        }

        path = self.checkpoint_dir / f"checkpoint_round_{round_num}.pt"
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

    def save_best_model(self, round_num: int) -> None:
        """Save the best model so far as both a full checkpoint and model-only file.

        Args:
            round_num: Round number at which the best accuracy was achieved.
        """
        # Full checkpoint for resumption
        best_checkpoint = {
            "round": round_num,
            "model_state_dict": self.global_model.state_dict(),
            "metrics": {"val_accuracy": self.best_val_accuracy},
            "config": self.config.to_dict(),
            "history": self.history,
            "best_val_accuracy": self.best_val_accuracy,
            "best_round": self.best_round,
            "rounds_without_improvement": self.rounds_without_improvement,
        }
        checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"
        torch.save(best_checkpoint, checkpoint_path)

        # Model-only file for easy inference loading
        model_only_path = self.checkpoint_dir / "best_model.pt"
        torch.save(self.global_model.state_dict(), model_only_path)

        logger.info(f"Saved best model (round {round_num}, acc={self.best_val_accuracy:.4f})")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Round number to resume from.
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model weights
        self.global_model.load_state_dict(checkpoint["model_state_dict"])

        # Restore training state
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        if "best_val_accuracy" in checkpoint:
            self.best_val_accuracy = checkpoint["best_val_accuracy"]
        elif "metrics" in checkpoint and "val_accuracy" in checkpoint["metrics"]:
            self.best_val_accuracy = checkpoint["metrics"]["val_accuracy"]
        if "best_round" in checkpoint:
            self.best_round = checkpoint["best_round"]
        if "rounds_without_improvement" in checkpoint:
            self.rounds_without_improvement = checkpoint["rounds_without_improvement"]

        round_num = checkpoint.get("round", 0)
        logger.info(f"Resumed from round {round_num}, best accuracy: {self.best_val_accuracy:.4f}")
        return round_num


    def run(self) -> Dict[str, Any]:
        """
        Run the complete FL simulation.

        Executes the federated learning training loop for the configured
        number of rounds, handling client selection, local training, and
        model aggregation. Supports checkpoint resumption and early stopping.

        Returns:
            Dictionary containing training results with keys:
                - history (dict): Round-by-round metrics containing:
                    - rounds (list[int]): Round numbers
                    - train_loss (list[float]): Training loss per round
                    - train_accuracy (list[float]): Training accuracy per round
                    - val_loss (list[float]): Validation loss per round
                    - val_accuracy (list[float]): Validation accuracy per round
                    - communication_cost (list[float]): MB transferred per round
                - best_val_accuracy (float): Best validation accuracy achieved
                - best_round (int): Round number with best accuracy
                - total_time_seconds (float): Total training duration
                - total_communication_mb (float): Cumulative data transferred
                - config (dict): Configuration used for the simulation

        Raises:
            RuntimeError: If no clients are available for training.

        Example:
            >>> config = SimulationConfig(num_rounds=10, num_clients=3)
            >>> simulator = FLSimulator(config)
            >>> results = simulator.run()
            >>> print(f"Best accuracy: {results['best_val_accuracy']:.4f}")
            Best accuracy: 0.8523
        """
        logger.info("Starting FL simulation")
        logger.info(f"Configuration: {self.config.num_rounds} rounds, {len(self.client_data) if self.client_data else self.config.num_clients} clients")

        # Setup clients
        self.setup_clients()

        if not self.client_data:
            raise RuntimeError("No clients available. Please check dataset paths.")

        # Resume from checkpoint if specified
        start_round = 1
        if self.config.resume_from:
            resume_path = Path(self.config.resume_from)
            if resume_path.exists():
                start_round = self.load_checkpoint(str(resume_path)) + 1
                logger.info(f"Resuming FL training from round {start_round}")
            else:
                logger.warning(f"Checkpoint not found at {resume_path}, starting from scratch")

        # Print client info
        print(f"\n{'='*60}")
        print(f"FL Simulation: {self.config.num_rounds} rounds, {len(self.client_data)} clients")
        if start_round > 1:
            print(f"Resuming from round {start_round}")
        print(f"{'='*60}")
        for cid, cdata in self.client_data.items():
            print(f"  Client {cid}: {cdata.dataset_name} ({cdata.num_train_samples} train, {cdata.num_val_samples} val)")
        print(f"{'='*60}\n")

        # Save initial config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Training loop with progress bar
        start_time = time.time()

        # Round-level progress bar routed to stdout for proper \r handling
        total_rounds = self.config.num_rounds - start_round + 1
        pbar = tqdm(
            total=total_rounds,
            desc=f"Round {start_round}/{self.config.num_rounds}",
            unit="round",
            file=sys.stdout,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
        )

        with logging_redirect_tqdm():
            for round_num in range(start_round, self.config.num_rounds + 1):
                pbar.set_description(f"Round {round_num}/{self.config.num_rounds}")
                metrics = self.run_round(round_num, pbar)

                # Update history
                self.history["rounds"].append(round_num)
                self.history["train_loss"].append(metrics["train_loss"])
                self.history["train_accuracy"].append(metrics["train_accuracy"])
                self.history["val_loss"].append(metrics["val_loss"])
                self.history["val_accuracy"].append(metrics["val_accuracy"])
                self.history["communication_cost"].append(metrics["communication_cost_mb"])

                # Check for improvement
                if metrics["val_accuracy"] > self.best_val_accuracy:
                    self.best_val_accuracy = metrics["val_accuracy"]
                    self.best_round = round_num
                    self.rounds_without_improvement = 0
                    self.save_best_model(round_num)
                else:
                    self.rounds_without_improvement += 1

                # Save checkpoint
                if round_num % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(round_num, metrics)

                # Update progress bar AFTER best tracking so 'best' is current
                pbar.set_postfix({
                    'loss': f'{metrics["train_loss"]:.4f}',
                    'val': f'{metrics["val_accuracy"]:.4f}',
                    'best': f'{self.best_val_accuracy:.4f}'
                })
                pbar.update(1)

                logger.info(
                    f"Round {round_num}/{self.config.num_rounds} | "
                    f"Train Loss: {metrics['train_loss']:.4f}, Train Acc: {metrics['train_accuracy']:.4f} | "
                    f"Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_accuracy']:.4f} | "
                    f"Clients: {metrics['clients_participated']}, Time: {metrics['round_time']:.1f}s"
                )

                # Early stopping
                if self.rounds_without_improvement >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

        pbar.close()
        total_time = time.time() - start_time

        # Final results
        results = {
            "history": self.history,
            "best_val_accuracy": self.best_val_accuracy,
            "best_round": self.best_round,
            "total_time_seconds": total_time,
            "total_communication_mb": sum(self.history["communication_cost"]),
            "config": self.config.to_dict(),
        }

        # Save results
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Simulation complete. Best accuracy: {self.best_val_accuracy:.4f} at round {self.best_round}")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Results saved to: {results_path}")

        return results


def run_fl_simulation(config: Optional[SimulationConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to run FL simulation.

    Args:
        config: Simulation configuration. If None, uses defaults.

    Returns:
        Simulation results.
    """
    if config is None:
        config = SimulationConfig()

    simulator = FLSimulator(config)
    return simulator.run()
