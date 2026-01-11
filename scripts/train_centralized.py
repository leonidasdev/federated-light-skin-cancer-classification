#!/usr/bin/env python
"""
Train Centralized Model
=======================

Script for running centralized (non-federated) training experiments.

Usage:
    python scripts/train_centralized.py --config configs/experiments/centralized_ham10000.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.models import LMSViT
from src.data import HAM10000Dataset, get_train_transforms, get_val_transforms
from src.training import CentralizedTrainer, EarlyStopping, ModelCheckpoint
from src.utils import set_seed, setup_logging, get_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    set_seed(config['hardware']['seed'])
    setup_logging(
        log_dir=config['logging']['log_dir'],
        log_to_file=True,
    )
    logger = get_logger("CentralizedTraining")
    
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    logger.info(f"Starting centralized training on {device}")
    logger.info(f"Configuration: {config['experiment']['name']}")
    
    # Create datasets
    train_transform = get_train_transforms(config['data']['img_size'])
    val_transform = get_val_transforms(config['data']['img_size'])
    
    train_dataset = HAM10000Dataset(
        root=config['data']['data_dir'],
        split='train',
        transform=train_transform,
    )
    
    val_dataset = HAM10000Dataset(
        root=config['data']['data_dir'],
        split='val',
        transform=val_transform,
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )
    
    # Create model
    model = LMSViT(
        img_size=config['model']['img_size'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Create loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
        ),
        ModelCheckpoint(
            filepath=str(Path(config['logging']['save_dir']) / 'best_model.pth'),
            monitor='accuracy',
            mode='max',
        ),
    ]
    
    # Create trainer
    trainer = CentralizedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=callbacks,
        gradient_clip=config['training']['gradient_clip'],
        mixed_precision=config['training']['mixed_precision'],
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
    )
    
    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {max(history['val_acc']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized training script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/centralized_ham10000.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    main(args)
