#!/usr/bin/env python
"""
Main Experiment Runner for DSCATNet Federated Learning.

This script provides the unified entry point for running all experiments:
- Centralized training (baseline upper-bound)
- Federated learning simulation with various non-IID distributions
- Comparison experiments between centralized and federated approaches

Usage Examples:
    # Run federated learning with config file
    python run_experiment.py --mode federated --config configs/dscatnet_federated_benchmark.yaml
    
    # Run centralized baseline
    python run_experiment.py --mode centralized --config configs/dscatnet_centralized_original.yaml
    
    # Override config settings via CLI
    python run_experiment.py --mode federated --config configs/dscatnet_federated_benchmark.yaml --rounds 10
    
    # Run comparison experiment
    python run_experiment.py --mode comparison --config configs/experiment_config.yaml

Author: Leonardo Chen
Date: 2024
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# =============================================================================
# Third-Party Imports
# =============================================================================
import yaml
import torch

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def setup_file_logging(output_dir: Path) -> None:
    """
    Add file handler to root logger for experiment logging.
    
    Args:
        output_dir: Directory where experiment.log will be created.
    """
    log_file = output_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Dictionary containing parsed configuration.
        
    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_centralized(args: argparse.Namespace) -> Dict[str, Any]:
    """Run centralized training experiment."""
    from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
    
    # Load config if provided
    if args.config:
        config_dict = load_config(args.config)
        cent_config = config_dict.get("centralized", {})
        
        # Flatten nested config structure for CentralizedConfig
        flat_config = {}
        
        # Direct mappings
        for key in ["data_root", "output_dir", "datasets"]:
            if key in cent_config:
                flat_config[key] = cent_config[key]
        
        # Experiment section
        if "experiment" in cent_config:
            exp = cent_config["experiment"]
            if "name" in exp:
                flat_config["experiment_name"] = exp["name"]
        
        # Model section
        if "model" in cent_config:
            model = cent_config["model"]
            if "image_size" in model:
                flat_config["image_size"] = model["image_size"]
            if "variant" in model:
                flat_config["model_variant"] = model["variant"]
            if "num_classes" in model:
                flat_config["num_classes"] = model["num_classes"]
        
        # Training section
        if "training" in cent_config:
            train = cent_config["training"]
            if "batch_size" in train:
                flat_config["batch_size"] = train["batch_size"]
            if "lr" in train:
                flat_config["learning_rate"] = train["lr"]
            if "epochs" in train:
                flat_config["num_epochs"] = train["epochs"]
            if "weight_decay" in train:
                flat_config["weight_decay"] = train["weight_decay"]
            if "warmup_epochs" in train:
                flat_config["warmup_epochs"] = train["warmup_epochs"]
            if "scheduler" in train:
                flat_config["scheduler_type"] = train["scheduler"]
            if "min_lr" in train:
                flat_config["min_lr"] = train["min_lr"]
        
        # Splits section
        if "splits" in cent_config:
            splits = cent_config["splits"]
            if "val_split" in splits:
                flat_config["val_split"] = splits["val_split"]
            if "test_split" in splits:
                flat_config["test_split"] = splits["test_split"]
        
        # Augmentation section
        if "augmentation" in cent_config:
            aug = cent_config["augmentation"]
            if "level" in aug:
                flat_config["augmentation_level"] = aug["level"]
            if "use_dermoscopy_norm" in aug:
                flat_config["use_dermoscopy_norm"] = aug["use_dermoscopy_norm"]
        
        # Evaluation section
        if "evaluation" in cent_config:
            evl = cent_config["evaluation"]
            if "early_stopping_patience" in evl:
                flat_config["early_stopping_patience"] = evl["early_stopping_patience"]
            if "use_class_weights" in evl:
                flat_config["use_class_weights"] = evl["use_class_weights"]
        
        config = CentralizedConfig.from_dict(flat_config)
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
    if args.datasets:
        config.datasets = args.datasets
    if args.resume:
        config.resume_from = args.resume
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
        fed_config = config_dict.get("federated", {})
        
        # Flatten nested config structure for SimulationConfig
        flat_config = {}
        
        # Direct mappings
        for key in ["data_root", "output_dir", "datasets"]:
            if key in fed_config:
                flat_config[key] = fed_config[key]
        
        # Experiment section
        if "experiment" in fed_config:
            exp = fed_config["experiment"]
            if "name" in exp:
                flat_config["experiment_name"] = exp["name"]
        
        # Model section
        if "model" in fed_config:
            model = fed_config["model"]
            if "image_size" in model:
                flat_config["image_size"] = model["image_size"]
            if "variant" in model:
                flat_config["model_variant"] = model["variant"]
        
        # Training section
        if "training" in fed_config:
            train = fed_config["training"]
            if "batch_size" in train:
                flat_config["batch_size"] = train["batch_size"]
            if "lr" in train:
                flat_config["learning_rate"] = train["lr"]
            if "local_epochs" in train:
                flat_config["local_epochs"] = train["local_epochs"]
            if "num_rounds" in train:
                flat_config["num_rounds"] = train["num_rounds"]
            if "rounds" in train:
                flat_config["num_rounds"] = train["rounds"]
        
        # Federation section
        if "federation" in fed_config:
            fed = fed_config["federation"]
            if "num_clients" in fed:
                flat_config["num_clients"] = fed["num_clients"]
            if "num_rounds" in fed:
                flat_config["num_rounds"] = fed["num_rounds"]
            if "noniid_type" in fed:
                flat_config["noniid_type"] = fed["noniid_type"]
            if "dirichlet_alpha" in fed:
                flat_config["dirichlet_alpha"] = fed["dirichlet_alpha"]
            if "participation" in fed:
                flat_config["fraction_fit"] = fed["participation"]
                flat_config["fraction_evaluate"] = fed["participation"]
        
        # Augmentation section
        if "augmentation" in fed_config:
            aug = fed_config["augmentation"]
            if "level" in aug:
                flat_config["augmentation_level"] = aug["level"]
            if "use_dermoscopy_norm" in aug:
                flat_config["use_dermoscopy_norm"] = aug["use_dermoscopy_norm"]
        
        # Evaluation section
        if "evaluation" in fed_config:
            evl = fed_config["evaluation"]
            if "checkpoint_interval" in evl:
                flat_config["checkpoint_interval"] = evl["checkpoint_interval"]
            if "early_stopping_patience" in evl:
                flat_config["early_stopping_patience"] = evl["early_stopping_patience"]
        
        config = SimulationConfig.from_dict(flat_config)
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
    if args.datasets:
        config.datasets = args.datasets
        # Auto-adjust num_clients to match selected datasets for natural non-IID
        if config.noniid_type == "natural":
            config.num_clients = len(args.datasets)
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

    # Run centralized with specific dataset only
    python run_experiment.py --mode centralized --epochs 50 --datasets HAM10000
    
    # Run centralized with multiple specific datasets
    python run_experiment.py --mode centralized --epochs 50 --datasets HAM10000 ISIC2019
    
    # Resume training from checkpoint
    python run_experiment.py --mode centralized --epochs 100 --resume outputs/exp/checkpoints/best_checkpoint.pt

    # Run federated learning with natural non-IID
    python run_experiment.py --mode federated --rounds 50 --noniid-type natural
    
    # Run federated learning with specific datasets only
    python run_experiment.py --mode federated --rounds 30 --datasets HAM10000 ISIC2019

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
    parser.add_argument(
        "--datasets", 
        type=str, 
        nargs="+",
        choices=["HAM10000", "ISIC2018", "ISIC2019", "ISIC2020", "PAD-UFES-20"],
        help="Specific dataset(s) to use (default: all). For FL natural non-IID, each dataset = one client"
    )
    
    # Centralized arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs (centralized)")
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
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
