"""
Federated Learning Simulation Module.

This module provides the complete FL simulation infrastructure for running
federated experiments with DSCATNet on dermoscopy datasets.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Sized, cast
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from flwr.common import Scalar

from ..models.dscatnet import create_dscatnet, get_model_parameters, set_model_parameters
from ..data.datasets import (
    HAM10000Dataset,
    ISIC2018Dataset,
    ISIC2019Dataset,
    ISIC2020Dataset,
)
from ..data.preprocessing import get_train_transforms, get_val_transforms
from ..data.splits import create_noniid_split

logger = logging.getLogger(__name__)


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
    
    # Training configuration
    local_epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Data configuration
    data_root: str = "./data"
    image_size: int = 224
    augmentation_level: str = "medium"
    use_dermoscopy_norm: bool = False
    
    # Non-IID configuration
    noniid_type: str = "natural"  # natural, dirichlet, label_skew, quantity_skew
    dirichlet_alpha: float = 0.5
    
    # Experiment configuration
    experiment_name: str = "fl_experiment"
    output_dir: str = "./outputs"
    checkpoint_interval: int = 5
    early_stopping_patience: int = 10
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class ClientData:
    """Data container for a single FL client."""
    
    client_id: int
    train_loader: DataLoader
    val_loader: DataLoader
    num_train_samples: int
    num_val_samples: int
    class_distribution: Dict[int, int]
    dataset_name: str


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
        
        # Training history
        self.history = {
            "rounds": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "client_metrics": [],
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
    
    def setup_natural_noniid(self) -> None:
        """
        Setup natural non-IID: each client gets a different dataset.
        
        Client 0: HAM10000
        Client 1: ISIC 2018
        Client 2: ISIC 2019
        Client 3: ISIC 2020
        """
        logger.info("Setting up natural non-IID distribution (each client = different dataset)")
        
        train_transform = get_train_transforms(
            img_size=self.config.image_size,
            augmentation_level=self.config.augmentation_level,
            use_dermoscopy_norm=self.config.use_dermoscopy_norm,
        )
        val_transform = get_val_transforms(
            img_size=self.config.image_size,
            use_dermoscopy_norm=self.config.use_dermoscopy_norm,
        )
        
        dataset_classes = [
            (HAM10000Dataset, "HAM10000"),
            (ISIC2018Dataset, "ISIC2018"),
            (ISIC2019Dataset, "ISIC2019"),
            (ISIC2020Dataset, "ISIC2020"),
        ]
        
        for client_id, (dataset_cls, dataset_name) in enumerate(dataset_classes):
            if client_id >= self.config.num_clients:
                break
            
            data_path = Path(self.config.data_root) / dataset_name.lower()
            
            # Create train and validation datasets
            try:
                train_dataset = dataset_cls(
                    root=str(data_path),
                    split="train",
                    transform=train_transform,
                )
                val_dataset = dataset_cls(
                    root=str(data_path),
                    split="val",
                    transform=val_transform,
                )
            except FileNotFoundError:
                logger.warning(f"Dataset {dataset_name} not found at {data_path}, creating dummy data")
                continue
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
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
    
    def setup_synthetic_noniid(self, combined_dataset: torch.utils.data.Dataset) -> None:
        """
        Setup synthetic non-IID using Dirichlet distribution.
        
        Args:
            combined_dataset: Combined dataset from all sources.
        """
        logger.info(f"Setting up Dirichlet non-IID with alpha={self.config.dirichlet_alpha}")
        
        # Determine dataset length in a type-safe way for Pylance
        try:
            ds_len = len(cast(Sized, combined_dataset))
        except TypeError:
            # Fallback: iterate to count (may be expensive for large datasets)
            ds_len = sum(1 for _ in combined_dataset)

        # Get all labels
        labels = np.array([combined_dataset[i][1] for i in range(ds_len)])
        
        # Convert labels to a plain Python list of ints for the split function
        labels_list: List[int] = [int(x) for x in labels]

        # Create non-IID split
        client_indices = create_noniid_split(
            labels=labels_list,
            num_clients=self.config.num_clients,
            alpha=self.config.dirichlet_alpha,
        )
        
        train_transform = get_train_transforms(
            img_size=self.config.image_size,
            augmentation_level=self.config.augmentation_level,
            use_dermoscopy_norm=self.config.use_dermoscopy_norm,
        )
        val_transform = get_val_transforms(
            img_size=self.config.image_size,
            use_dermoscopy_norm=self.config.use_dermoscopy_norm,
        )
        
        # `create_noniid_split` returns a dict; enumerate over its values
        # so `indices` is a list of indices (not the dict key int).
        for client_id, indices in enumerate(client_indices.values()):
            # Split into train/val
            np.random.shuffle(indices)
            split_idx = int(len(indices) * 0.8)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            train_subset = Subset(combined_dataset, train_indices)
            val_subset = Subset(combined_dataset, val_indices)
            
            # Create loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
            
            # Class distribution
            class_dist = {}
            for idx in train_indices:
                label = labels[idx]
                class_dist[int(label)] = class_dist.get(int(label), 0) + 1
            
            self.client_data[client_id] = ClientData(
                client_id=client_id,
                train_loader=train_loader,
                val_loader=val_loader,
                num_train_samples=len(train_indices),
                num_val_samples=len(val_indices),
                class_distribution=class_dist,
                dataset_name="combined",
            )
            
            logger.info(f"Client {client_id}: {len(train_indices)} train, {len(val_indices)} val samples")
    
    def setup_clients(self) -> None:
        """Setup client data based on configuration."""
        if self.config.noniid_type == "natural":
            self.setup_natural_noniid()
        else:
            # For synthetic non-IID, would need to combine datasets first
            logger.warning("Synthetic non-IID requires combined dataset - using natural non-IID")
            self.setup_natural_noniid()
    
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
        optimizer = torch.optim.AdamW(
            local_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training
        local_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, (images, labels) in enumerate(client.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
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
        
        # Initialize with zeros
        aggregated = [np.zeros_like(param) for param, _ in results[0][0]]
        
        for params, num_samples in results:
            weight = num_samples / total_samples
            for i, param in enumerate(params):
                aggregated[i] += param * weight
        
        return aggregated
    
    def run_round(self, round_num: int) -> Dict[str, float]:
        """
        Run a single FL round.
        
        Args:
            round_num: Current round number.
            
        Returns:
            Dictionary of aggregated metrics.
        """
        logger.info(f"Starting round {round_num}")
        start_time = time.time()
        
        # Get current global parameters
        global_params = get_model_parameters(self.global_model)
        
        # Client training
        fit_results = []
        client_train_metrics = []
        
        for client_id in self.client_data.keys():
            params, num_samples, metrics = self.train_client(client_id, global_params)
            fit_results.append((params, num_samples))
            client_train_metrics.append(metrics)
            logger.debug(f"Client {client_id}: loss={metrics['train_loss']:.4f}, acc={metrics['train_accuracy']:.4f}")
        
        # Aggregate parameters
        aggregated_params = self.aggregate_parameters(fit_results)
        set_model_parameters(self.global_model, aggregated_params)
        
        # Client evaluation
        eval_results = []
        client_val_metrics = []
        
        for client_id in self.client_data.keys():
            loss, num_samples, metrics = self.evaluate_client(client_id, aggregated_params)
            eval_results.append((loss, num_samples, metrics))
            client_val_metrics.append(metrics)
        
        # Aggregate metrics
        total_train_samples = sum(r[1] for r in fit_results)
        total_val_samples = sum(r[1] for r in eval_results)
        
        avg_train_loss = np.mean([m["train_loss"] for m in client_train_metrics])
        avg_train_acc = np.mean([m["train_accuracy"] for m in client_train_metrics])
        avg_val_loss = sum(r[0] * r[1] for r in eval_results) / total_val_samples
        avg_val_acc = np.mean([m["val_accuracy"] for m in client_val_metrics])
        
        round_time = time.time() - start_time
        
        # Calculate communication cost (model size * 2 * num_clients for upload/download)
        model_size_bytes = sum(p.nbytes for p in global_params)
        comm_cost = model_size_bytes * 2 * len(self.client_data)
        
        metrics = {
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_acc,
            "round_time": round_time,
            "communication_cost_mb": comm_cost / (1024 * 1024),
        }
        
        logger.info(
            f"Round {round_num}: train_loss={avg_train_loss:.4f}, train_acc={avg_train_acc:.4f}, "
            f"val_loss={avg_val_loss:.4f}, val_acc={avg_val_acc:.4f}, time={round_time:.2f}s"
        )
        
        return metrics
    
    def save_checkpoint(self, round_num: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "round": round_num,
            "model_state_dict": self.global_model.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
        }
        
        path = self.checkpoint_dir / f"checkpoint_round_{round_num}.pt"
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")
    
    def save_best_model(self, round_num: int) -> None:
        """Save the best model."""
        path = self.checkpoint_dir / "best_model.pt"
        torch.save({
            "round": round_num,
            "model_state_dict": self.global_model.state_dict(),
            "val_accuracy": self.best_val_accuracy,
            "config": self.config.to_dict(),
        }, path)
        logger.info(f"Saved best model (round {round_num}, acc={self.best_val_accuracy:.4f})")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete FL simulation.
        
        Returns:
            Dictionary containing training history and final results.
        """
        logger.info("Starting FL simulation")
        logger.info(f"Configuration: {self.config.num_rounds} rounds, {self.config.num_clients} clients")
        
        # Setup clients
        self.setup_clients()
        
        if not self.client_data:
            raise RuntimeError("No clients available. Please check dataset paths.")
        
        # Save initial config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Training loop
        start_time = time.time()
        
        for round_num in range(1, self.config.num_rounds + 1):
            metrics = self.run_round(round_num)
            
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
            
            # Early stopping
            if self.rounds_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at round {round_num}")
                break
        
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
