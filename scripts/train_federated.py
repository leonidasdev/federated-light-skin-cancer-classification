#!/usr/bin/env python
"""
Train Federated Model
=====================

Script for running federated learning experiments.

Usage:
    python scripts/train_federated.py --config configs/experiments/federated_ham10000_iid.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.models import LMSViT
from src.data import (
    HAM10000Dataset,
    get_train_transforms,
    get_val_transforms,
    FederatedDatasetPartitioner,
)
from src.federated import FederatedServer, FederatedClient, FedAvg, FedProx
from src.utils import set_seed, setup_logging, get_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_clients(
    model: torch.nn.Module,
    partitioned_data: dict,
    config: dict,
    device: str,
) -> list:
    """Create federated learning clients."""
    clients = []
    
    for client_id, data_subset in partitioned_data.items():
        client_loader = torch.utils.data.DataLoader(
            data_subset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
        )
        
        client = FederatedClient(
            client_id=client_id,
            model=model,
            train_loader=client_loader,
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={'lr': config['training']['learning_rate']},
            device=device,
        )
        clients.append(client)
    
    return clients


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    set_seed(config['hardware']['seed'])
    setup_logging(
        log_dir=config['logging']['log_dir'],
        log_to_file=True,
    )
    logger = get_logger("FederatedTraining")
    
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    logger.info(f"Starting federated training on {device}")
    logger.info(f"Configuration: {config['experiment']['name']}")
    logger.info(f"Number of clients: {config['federated']['num_clients']}")
    logger.info(f"Partition strategy: {config['federated']['partition']['strategy']}")
    
    # Create dataset
    train_transform = get_train_transforms(config['data']['img_size'])
    val_transform = get_val_transforms(config['data']['img_size'])
    
    full_train_dataset = HAM10000Dataset(
        root=config['data']['data_dir'],
        split='train',
        transform=train_transform,
    )
    
    test_dataset = HAM10000Dataset(
        root=config['data']['data_dir'],
        split='test',
        transform=val_transform,
    )
    
    # Partition data for clients
    partitioner = FederatedDatasetPartitioner(
        dataset=full_train_dataset,
        num_clients=config['federated']['num_clients'],
        partition_strategy=config['federated']['partition']['strategy'],
        seed=config['hardware']['seed'],
    )
    
    alpha = config['federated']['partition'].get('alpha', 0.5)
    partitioned_data = partitioner.partition(alpha=alpha)
    
    # Create global model
    global_model = LMSViT(
        img_size=config['model']['img_size'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    
    # Create clients
    clients = create_clients(global_model, partitioned_data, config, device)
    
    # Create test loader for server evaluation
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
    )
    
    # Select aggregation strategy
    strategy_name = config['federated']['strategy']
    if strategy_name == 'fedavg':
        aggregation_fn = FedAvg().aggregate
    elif strategy_name == 'fedprox':
        aggregation_fn = FedProx(mu=config['federated']['fedprox']['mu']).aggregate
    else:
        aggregation_fn = None
    
    # Create server
    server = FederatedServer(
        model=global_model,
        num_clients=config['federated']['num_clients'],
        fraction_fit=config['federated']['fraction_fit'],
        min_fit_clients=config['federated']['min_fit_clients'],
        aggregation_fn=aggregation_fn,
        device=device,
    )
    
    # Training loop
    criterion = torch.nn.CrossEntropyLoss()
    best_accuracy = 0.0
    
    for round_num in range(config['federated']['rounds']):
        # Run federated round
        round_metrics = server.run_round(
            clients=clients,
            epochs_per_round=config['federated']['epochs_per_round'],
        )
        
        # Evaluate global model
        eval_metrics = server.evaluate(test_loader, criterion)
        
        logger.info(
            f"Round {round_num + 1}/{config['federated']['rounds']} - "
            f"Loss: {eval_metrics['loss']:.4f} - "
            f"Accuracy: {eval_metrics['accuracy']:.4f}"
        )
        
        # Save best model
        if eval_metrics['accuracy'] > best_accuracy:
            best_accuracy = eval_metrics['accuracy']
            save_path = Path(config['logging']['save_dir']) / 'best_global_model.pth'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(global_model.state_dict(), save_path)
            logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
    
    logger.info("Federated training complete!")
    logger.info(f"Best test accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated training script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/federated_ham10000_iid.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    main(args)
