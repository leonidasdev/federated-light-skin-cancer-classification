#!/usr/bin/env python
"""
Main Experiment Runner.

This script provides the entry point for running experiments:
- Centralized training (baseline)
- Federated learning simulation
- Comparison experiments

Usage:
    python run_experiment.py --mode [centralized|federated|comparison] [options]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import yaml
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def setup_file_logging(output_dir: Path) -> None:
    """Add file handler to root logger."""
    log_file = output_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_centralized(args: argparse.Namespace) -> Dict[str, Any]:
    """Run centralized training experiment."""
    from src.training.centralized import CentralizedConfig, CentralizedTrainer
    
    # Load config if provided
    if args.config:
        config_dict = load_config(args.config)
        config = CentralizedConfig.from_dict(config_dict.get("centralized", {}))
    else:
        config = CentralizedConfig()
    
    # Override with command line args
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.data_root:
        config.data_root = args.data_root
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        config.experiment_name = f"centralized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup output directory and logging
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("CENTRALIZED TRAINING EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    
    # Run training
    trainer = CentralizedTrainer(config)
    results = trainer.run()
    
    return results


def run_federated(args: argparse.Namespace) -> Dict[str, Any]:
    """Run federated learning experiment."""
    from src.federated.simulation import SimulationConfig, FLSimulator
    
    # Load config if provided
    if args.config:
        config_dict = load_config(args.config)
        config = SimulationConfig.from_dict(config_dict.get("federated", {}))
    else:
        config = SimulationConfig()
    
    # Override with command line args
    if args.rounds:
        config.num_rounds = args.rounds
    if args.clients:
        config.num_clients = args.clients
    if args.local_epochs:
        config.local_epochs = args.local_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.data_root:
        config.data_root = args.data_root
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.noniid_type:
        config.noniid_type = args.noniid_type
    if args.dirichlet_alpha:
        config.dirichlet_alpha = args.dirichlet_alpha
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        config.experiment_name = f"federated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup output directory and logging
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("FEDERATED LEARNING EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Non-IID type: {config.noniid_type}")
    
    # Run simulation
    simulator = FLSimulator(config)
    results = simulator.run()
    
    return results


def run_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    """Run both centralized and federated experiments for comparison."""
    from src.evaluation.metrics import compare_results, print_comparison
    from src.evaluation.visualization import (
        plot_training_curves,
        plot_fl_vs_centralized,
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create comparison output directory
    comparison_dir = Path(args.output_dir or "./outputs") / f"comparison_{timestamp}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(comparison_dir)
    
    logger.info("=" * 60)
    logger.info("COMPARISON EXPERIMENT")
    logger.info("=" * 60)
    
    # Run centralized
    logger.info("\n--- Running Centralized Baseline ---")
    args.experiment_name = f"centralized_{timestamp}"
    centralized_results = run_centralized(args)
    
    # Run federated
    logger.info("\n--- Running Federated Learning ---")
    args.experiment_name = f"federated_{timestamp}"
    federated_results = run_federated(args)
    
    # Compare and visualize
    logger.info("\n--- Generating Comparison ---")
    
    # Plot comparison
    if centralized_results.get("history") and federated_results.get("history"):
        plot_fl_vs_centralized(
            federated_results["history"],
            centralized_results["history"],
            metric="val_accuracy",
            save_path=comparison_dir / "comparison_accuracy.png",
            title="Federated vs Centralized: Validation Accuracy",
        )
        
        plot_fl_vs_centralized(
            federated_results["history"],
            centralized_results["history"],
            metric="val_loss",
            save_path=comparison_dir / "comparison_loss.png",
            title="Federated vs Centralized: Validation Loss",
        )
    
    # Summary comparison
    comparison_summary: Dict[str, Any] = {
        "centralized": {
            "best_accuracy": centralized_results.get("best_val_accuracy"),
            "best_epoch": centralized_results.get("best_epoch"),
            "total_time": centralized_results.get("total_time_seconds"),
        },
        "federated": {
            "best_accuracy": federated_results.get("best_val_accuracy"),
            "best_round": federated_results.get("best_round"),
            "total_time": federated_results.get("total_time_seconds"),
            "communication_cost_mb": federated_results.get("total_communication_mb"),
        },
    }
    
    # Calculate accuracy gap
    cent_acc = comparison_summary["centralized"]["best_accuracy"] or 0
    fed_acc = comparison_summary["federated"]["best_accuracy"] or 0
    comparison_summary["accuracy_gap"] = cent_acc - fed_acc
    comparison_summary["accuracy_gap_pct"] = (comparison_summary["accuracy_gap"] / cent_acc * 100) if cent_acc else 0
    
    # Save comparison
    with open(comparison_dir / "comparison_summary.json", "w") as f:
        json.dump(comparison_summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Centralized Best Accuracy: {cent_acc:.4f}")
    logger.info(f"Federated Best Accuracy:   {fed_acc:.4f}")
    logger.info(f"Accuracy Gap:              {comparison_summary['accuracy_gap']:.4f} ({comparison_summary['accuracy_gap_pct']:.2f}%)")
    logger.info("=" * 60)
    
    return comparison_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run DSCATNet Federated Learning Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run centralized baseline
    python run_experiment.py --mode centralized --epochs 100

    # Run federated learning with natural non-IID
    python run_experiment.py --mode federated --rounds 50 --noniid-type natural

    # Run comparison experiment
    python run_experiment.py --mode comparison --config configs/experiment_config.yaml
        """,
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["centralized", "federated", "comparison"],
        required=True,
        help="Experiment mode to run",
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    
    # Common arguments
    parser.add_argument("--data-root", type=str, help="Root directory for datasets")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--experiment-name", type=str, help="Name for this experiment")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    
    # Centralized arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs (centralized)")
    
    # Federated arguments
    parser.add_argument("--rounds", type=int, help="Number of FL rounds")
    parser.add_argument("--clients", type=int, help="Number of clients")
    parser.add_argument("--local-epochs", type=int, help="Local epochs per round")
    parser.add_argument(
        "--noniid-type",
        type=str,
        choices=["natural", "dirichlet", "label_skew", "quantity_skew"],
        help="Type of non-IID distribution",
    )
    parser.add_argument("--dirichlet-alpha", type=float, help="Dirichlet alpha parameter")
    
    # Parse args
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("DSCATNet Federated Learning Experiment")
    print(f"Mode: {args.mode.upper()}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60 + "\n")
    
    # Run experiment
    if args.mode == "centralized":
        results = run_centralized(args)
    elif args.mode == "federated":
        results = run_federated(args)
    elif args.mode == "comparison":
        results = run_comparison(args)
    
    print("\nExperiment completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
