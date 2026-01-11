#!/usr/bin/env python
"""
Train Federated LMS-ViT with Flower
===================================

Script for running federated learning experiments using Flower framework.
Supports local simulation mode for development and testing.

Usage:
    python scripts/train_federated_flower.py --config configs/experiments/federated_ham10000_flower.yaml
    
    # Quick test with defaults
    python scripts/train_federated_flower.py --num-clients 4 --num-rounds 5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import Dataset, Subset, random_split

from src.models import lmsvit_tiny, lmsvit_small, lmsvit_base
from src.data import (
    HAM10000Dataset, 
    ISIC2018Dataset, 
    ISIC2019Dataset, 
    ISIC2020Dataset,
    get_train_transforms, 
    get_val_transforms,
)
from src.data.federated import (
    FederatedDatasetPartitioner,
    iid_partition,
    dirichlet_partition,
)
from src.federated.flower_client import create_client_fn
from src.federated.flower_server import (
    run_flower_simulation,
    create_strategy,
    get_initial_parameters,
    print_history,
)
from src.utils import set_seed, setup_logging, get_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_config() -> dict:
    """Get default configuration for quick testing."""
    return {
        "experiment": {
            "name": "federated_flower_test",
            "description": "Federated learning with Flower - test run",
        },
        "model": {
            "name": "lmsvit_small",
            "num_classes": 7,
        },
        "data": {
            "dataset": "ham10000",
            "data_dir": "./data",
            "img_size": 224,
            "batch_size": 32,
        },
        "federated": {
            "num_clients": 4,
            "num_rounds": 10,
            "local_epochs": 1,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "strategy": "fedavg",
            "partition": {
                "strategy": "iid",
                "alpha": 0.5,  # For dirichlet
            },
        },
        "training": {
            "learning_rate": 0.0001,
        },
        "hardware": {
            "device": "cuda",
            "seed": 42,
        },
    }


def create_dataset(
    dataset_name: str,
    data_dir: str,
    img_size: int,
    split: str = "train",
) -> Dataset:
    """
    Create a dataset instance.
    
    Args:
        dataset_name: Name of dataset ('ham10000', 'isic2018', etc.)
        data_dir: Root directory for data
        img_size: Image size for transforms
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        Dataset instance
    """
    transform = get_train_transforms(img_size) if split == "train" else get_val_transforms(img_size)
    
    datasets = {
        "ham10000": HAM10000Dataset,
        "isic2018": ISIC2018Dataset,
        "isic2019": ISIC2019Dataset,
        "isic2020": ISIC2020Dataset,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_class = datasets[dataset_name]
    
    return dataset_class(
        root=data_dir,
        split=split,
        transform=transform,
    )


def prepare_client_datasets(
    full_train_dataset: Dataset,
    num_clients: int,
    partition_strategy: str = "iid",
    alpha: float = 0.5,
    val_split: float = 0.1,
    seed: int = 42,
) -> Dict[int, Tuple[Dataset, Optional[Dataset]]]:
    """
    Prepare datasets for each client.
    
    Args:
        full_train_dataset: Full training dataset to partition
        num_clients: Number of federated clients
        partition_strategy: How to partition ('iid', 'dirichlet', 'pathological')
        alpha: Dirichlet concentration parameter
        val_split: Fraction of client data to use for validation
        seed: Random seed
        
    Returns:
        Dictionary mapping client_id to (train_dataset, val_dataset) tuples
    """
    # Create partitioner
    partitioner = FederatedDatasetPartitioner(
        dataset=full_train_dataset,
        num_clients=num_clients,
        partition_strategy=partition_strategy,
        seed=seed,
    )
    
    # Partition data
    partitions = partitioner.partition(alpha=alpha)
    
    # Print statistics
    partitioner.print_statistics()
    
    # Split each client's data into train/val
    client_datasets = {}
    
    for client_id, client_subset in partitions.items():
        num_samples = len(client_subset)
        num_val = max(1, int(num_samples * val_split))
        num_train = num_samples - num_val
        
        # Use random_split to create train/val for this client
        generator = torch.Generator().manual_seed(seed + client_id)
        train_subset, val_subset = random_split(
            client_subset, 
            [num_train, num_val],
            generator=generator,
        )
        
        client_datasets[client_id] = (train_subset, val_subset)
    
    return client_datasets


def main(args):
    """Main entry point for federated training."""
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.num_clients:
        config["federated"]["num_clients"] = args.num_clients
    if args.num_rounds:
        config["federated"]["num_rounds"] = args.num_rounds
    if args.local_epochs:
        config["federated"]["local_epochs"] = args.local_epochs
    if args.strategy:
        config["federated"]["strategy"] = args.strategy
    if args.partition:
        config["federated"]["partition"]["strategy"] = args.partition
    
    # Setup
    seed = config.get("hardware", {}).get("seed", 42)
    set_seed(seed)
    setup_logging(log_dir="./logs/federated", log_to_file=True)
    logger = get_logger("FederatedFlower")
    
    # Device
    device = config.get("hardware", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Extract config values
    num_clients = config["federated"]["num_clients"]
    num_rounds = config["federated"]["num_rounds"]
    local_epochs = config["federated"]["local_epochs"]
    strategy_name = config["federated"]["strategy"]
    partition_strategy = config["federated"]["partition"]["strategy"]
    alpha = config["federated"]["partition"].get("alpha", 0.5)
    
    model_name = config["model"]["name"].replace("lmsvit_", "")
    num_classes = config["model"]["num_classes"]
    
    dataset_name = config["data"]["dataset"]
    data_dir = config["data"]["data_dir"]
    img_size = config["data"]["img_size"]
    batch_size = config["data"]["batch_size"]
    
    learning_rate = config["training"]["learning_rate"]
    
    logger.info("=" * 60)
    logger.info("FEDERATED LEARNING WITH FLOWER")
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.get('experiment', {}).get('name', 'unnamed')}")
    logger.info(f"Model: LMS-ViT {model_name} ({num_classes} classes)")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Clients: {num_clients}")
    logger.info(f"Rounds: {num_rounds}")
    logger.info(f"Local epochs: {local_epochs}")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Partition: {partition_strategy}" + (f" (alpha={alpha})" if partition_strategy == "dirichlet" else ""))
    logger.info(f"Device: {device}")
    logger.info("=" * 60)
    
    # Create full training dataset
    logger.info("Loading dataset...")
    try:
        full_train_dataset = create_dataset(
            dataset_name=dataset_name,
            data_dir=data_dir,
            img_size=img_size,
            split="train",
        )
        logger.info(f"Dataset loaded: {len(full_train_dataset)} samples")
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error(f"Please download {dataset_name} dataset to {data_dir}")
        return
    
    # Prepare client datasets
    logger.info("Partitioning data across clients...")
    client_datasets = prepare_client_datasets(
        full_train_dataset=full_train_dataset,
        num_clients=num_clients,
        partition_strategy=partition_strategy,
        alpha=alpha,
        val_split=0.1,
        seed=seed,
    )
    
    # Create client factory function
    client_fn = create_client_fn(
        client_datasets=client_datasets,
        model_name=model_name,
        num_classes=num_classes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        local_epochs=local_epochs,
    )
    
    # Create strategy
    initial_parameters = get_initial_parameters(model_name, num_classes)
    
    strategy_kwargs = {}
    if strategy_name == "fedprox":
        strategy_kwargs["proximal_mu"] = config["federated"].get("fedprox_mu", 0.01)
    
    strategy = create_strategy(
        strategy_name=strategy_name,
        num_clients=num_clients,
        fraction_fit=config["federated"].get("fraction_fit", 1.0),
        fraction_evaluate=config["federated"].get("fraction_evaluate", 1.0),
        min_fit_clients=min(2, num_clients),
        min_evaluate_clients=min(2, num_clients),
        min_available_clients=min(2, num_clients),
        initial_parameters=initial_parameters,
        **strategy_kwargs,
    )
    
    # Configure client resources
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if device == "cuda" and torch.cuda.is_available():
        # Share GPU among clients (be conservative)
        client_resources["num_gpus"] = min(0.5, 1.0 / num_clients)
    
    # Run simulation
    logger.info("Starting Flower simulation...")
    
    try:
        history = run_flower_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            num_rounds=num_rounds,
            strategy=strategy,
            model_name=model_name,
            num_classes=num_classes,
            client_resources=client_resources,
            ray_init_args={
                "include_dashboard": False,
                "num_cpus": num_clients + 1,  # +1 for server
            },
        )
        
        # Print results
        print_history(history)
        
        # Save history
        history_path = Path("./logs/federated") / f"history_{config.get('experiment', {}).get('name', 'unnamed')}.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_path, "w") as f:
            f.write(f"Experiment: {config.get('experiment', {}).get('name', 'unnamed')}\n")
            f.write(f"Clients: {num_clients}, Rounds: {num_rounds}\n")
            f.write(f"Strategy: {strategy_name}, Partition: {partition_strategy}\n\n")
            
            if history.losses_distributed:
                f.write("Distributed Losses:\n")
                for round_num, loss in history.losses_distributed:
                    f.write(f"  Round {round_num}: {loss:.4f}\n")
            
            if history.metrics_distributed:
                f.write("\nDistributed Metrics:\n")
                for metric_name, values in history.metrics_distributed.items():
                    f.write(f"\n  {metric_name}:\n")
                    for round_num, value in values:
                        f.write(f"    Round {round_num}: {value:.4f}\n")
        
        logger.info(f"History saved to {history_path}")
        logger.info("Federated training completed successfully!")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LMS-ViT with Federated Learning using Flower"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help="Number of federated clients (overrides config)",
    )
    
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Number of federated rounds (overrides config)",
    )
    
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=None,
        help="Local epochs per round (overrides config)",
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["fedavg", "fedprox", "fedadam", "fedadagrad"],
        default=None,
        help="Aggregation strategy (overrides config)",
    )
    
    parser.add_argument(
        "--partition",
        type=str,
        choices=["iid", "dirichlet", "pathological"],
        default=None,
        help="Data partition strategy (overrides config)",
    )
    
    args = parser.parse_args()
    main(args)
