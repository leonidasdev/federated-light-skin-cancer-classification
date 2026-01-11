#!/usr/bin/env python
"""
Evaluate Model
==============

Script for evaluating trained models on test data.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import json

from src.models import LMSViT
from src.data import HAM10000Dataset, get_val_transforms
from src.utils import MetricsCalculator, plot_confusion_matrix, get_logger, setup_logging


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    setup_logging()
    logger = get_logger("Evaluation")
    
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = LMSViT(
        img_size=config['model']['img_size'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    
    # Load test data
    transform = get_val_transforms(config['data']['img_size'])
    test_dataset = HAM10000Dataset(
        root=config['data']['data_dir'],
        split='test',
        transform=transform,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
    )
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    metrics_calc = MetricsCalculator(
        num_classes=config['model']['num_classes'],
        class_names=HAM10000Dataset.CLASS_NAMES,
    )
    
    metrics = metrics_calc.calculate(all_preds, all_labels, all_probs)
    
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Precision (macro): {metrics['precision']:.4f}")
    logger.info(f"Recall (macro): {metrics['recall']:.4f}")
    logger.info(f"F1 Score (macro): {metrics['f1']:.4f}")
    if 'auc' in metrics:
        logger.info(f"AUC (macro): {metrics['auc']:.4f}")
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info(metrics_calc.get_classification_report(all_preds, all_labels))
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({k: v for k, v in metrics.items() if not isinstance(v, list)}, f, indent=2)
    
    # Save confusion matrix plot
    cm = metrics_calc.get_confusion_matrix(all_preds, all_labels)
    plot_confusion_matrix(
        cm,
        class_names=HAM10000Dataset.CLASS_NAMES,
        save_path=str(output_dir / 'confusion_matrix.png'),
        title='Confusion Matrix - Test Set',
    )
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    main(args)
