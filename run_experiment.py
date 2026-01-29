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


def run_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """Evaluate a trained model checkpoint using the DatasetRegistry."""
    from src.models.dscatnet import create_dscatnet
    from src.evaluation.metrics import ModelEvaluator
    from src.data.datasets import (
        DATASET_REGISTRY, get_dataset_paths, normalize_dataset_name,
        get_available_datasets, DatasetSubset
    )
    from src.data.preprocessing import get_val_transforms
    from torch.utils.data import DataLoader, ConcatDataset
    
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for evaluation mode")
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Get config from checkpoint or use defaults
    saved_config = checkpoint.get("config", {})
    model_variant = args.model_variant or saved_config.get("model_variant", "small")
    num_classes = args.num_classes or saved_config.get("num_classes", 7)
    image_size = args.image_size or saved_config.get("image_size", 224)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_dscatnet(
        variant=model_variant,
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume checkpoint is just state dict
        model.load_state_dict(checkpoint)
    
    logger.info(f"Model loaded: {model_variant}, {num_classes} classes")
    
    # Setup data using registry
    data_root = Path(args.data_root or saved_config.get("data_root", "./data"))
    datasets_to_use = args.datasets or saved_config.get("datasets")
    
    val_transform = get_val_transforms(img_size=image_size)
    
    # Determine which datasets to use
    if datasets_to_use:
        dataset_names = [normalize_dataset_name(d) for d in datasets_to_use]
    else:
        dataset_names = get_available_datasets()
    
    # Load datasets using registry
    test_datasets = []
    for name in dataset_names:
        if name not in DATASET_REGISTRY:
            logger.warning(f"Unknown dataset: {name}, skipping")
            continue
            
        config = DATASET_REGISTRY[name]
        csv_path, dataset_root = get_dataset_paths(name, data_root)
        
        if csv_path is None or not csv_path.exists():
            logger.warning(f"Dataset {name}: CSV not found, skipping")
            continue
        
        if dataset_root is None or not dataset_root.exists():
            logger.warning(f"Dataset {name}: Image directory not found, skipping")
            continue
        
        try:
            dataset = config.dataset_class(
                root_dir=str(dataset_root),
                csv_path=str(csv_path),
                transform=val_transform,
            )
            # Use last 20% as test set (same split logic as training)
            n = len(dataset)
            gen = torch.Generator()
            gen.manual_seed(42)
            indices = torch.randperm(n, generator=gen).tolist()
            test_indices = indices[int(n * 0.8):]
            test_ds = DatasetSubset(dataset, test_indices, val_transform)
            test_datasets.append(test_ds)
            logger.info(f"Loaded {name}: {len(test_ds)} test samples")
        except Exception as e:
            logger.warning(f"Failed loading {name}: {e}")
    
    if not test_datasets:
        raise RuntimeError("No datasets found for evaluation")
    
    combined_test = ConcatDataset(test_datasets)
    test_loader = DataLoader(
        combined_test,
        batch_size=args.batch_size or 32,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    
    logger.info(f"Total test samples: {len(combined_test)}")
    
    # Evaluate
    evaluator = ModelEvaluator(model, device, num_classes=num_classes)
    results = evaluator.evaluate(test_loader)
    
    # Print report
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:          {results.accuracy:.4f}")
    print(f"Balanced Accuracy: {results.balanced_accuracy:.4f}")
    print(f"Precision (macro): {results.precision_macro:.4f}")
    print(f"Recall (macro):    {results.recall_macro:.4f}")
    print(f"F1 (macro):        {results.f1_macro:.4f}")
    print(f"F1 (weighted):     {results.f1_weighted:.4f}")
    if results.auc_macro:
        print(f"AUC-ROC (macro):   {results.auc_macro:.4f}")
    print("=" * 60)
    
    print("\nPer-Class Metrics:")
    for class_name, metrics in results.per_class_metrics.items():
        print(f"  {class_name}: acc={metrics['accuracy']:.3f}, "
              f"prec={metrics['precision']:.3f}, rec={metrics['recall']:.3f}, "
              f"support={metrics['support']}")
    
    # Save results if output dir specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_dict = results.to_dict()
        results_dict["checkpoint"] = str(checkpoint_path)
        results_dict["datasets"] = [name for _, name in dataset_classes]
        
        results_file = output_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return results.to_dict()


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
            if "checkpoint_interval" in evl:
                flat_config["checkpoint_interval"] = evl["checkpoint_interval"]
        
        config = CentralizedConfig.from_dict(flat_config)
    else:
        config = CentralizedConfig()
    
    # Override with command line args (all hyperparameters)
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
    
    # Additional hyperparameter overrides
    if args.model_variant:
        config.model_variant = args.model_variant
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.warmup_epochs is not None:
        config.warmup_epochs = args.warmup_epochs
    if args.scheduler:
        config.scheduler_type = args.scheduler
    if args.early_stopping is not None:
        config.early_stopping_patience = args.early_stopping
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = args.checkpoint_interval
    if args.image_size is not None:
        config.image_size = args.image_size
    if args.num_classes is not None:
        config.num_classes = args.num_classes
    if args.augmentation:
        config.augmentation_level = args.augmentation
    if args.val_split is not None:
        config.val_split = args.val_split
    if args.no_amp:
        config.use_amp = False
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    
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
    
    # If resuming, try to load config from checkpoint first
    checkpoint_config = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"Loading config from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            checkpoint_config = checkpoint.get("config", {})
            if checkpoint_config:
                logger.info(f"Restored config from checkpoint: noniid_type={checkpoint_config.get('noniid_type')}, "
                          f"datasets={checkpoint_config.get('datasets')}")
    
    # Load config: priority is CLI args > YAML config > checkpoint config > defaults
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
    elif checkpoint_config:
        # No YAML config provided but resuming from checkpoint - use checkpoint's config
        logger.info("Using config from checkpoint as base (no YAML config provided)")
        config = SimulationConfig.from_dict(checkpoint_config)
    else:
        config = SimulationConfig()
    
    # Override with command line args (all hyperparameters)
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
    
    # Additional hyperparameter overrides
    if args.model_variant:
        config.model_variant = args.model_variant
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.early_stopping is not None:
        config.early_stopping_patience = args.early_stopping
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = args.checkpoint_interval
    if args.image_size is not None:
        config.image_size = args.image_size
    if args.num_classes is not None:
        config.num_classes = args.num_classes
    if args.augmentation:
        config.augmentation_level = args.augmentation
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.participation is not None:
        config.fraction_fit = args.participation
        config.fraction_evaluate = args.participation
    if args.client_selection is not None:
        config.client_selection_fraction = args.client_selection
    if args.parallel_clients is not None:
        config.parallel_clients = args.parallel_clients
    if args.train_val_split is not None:
        config.train_val_split = args.train_val_split
    
    # Resume from checkpoint
    if args.resume:
        config.resume_from = args.resume
    
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

    # Run centralized with specific dataset and model variant
    python run_experiment.py --mode centralized --epochs 50 --datasets HAM10000 --model-variant small
    
    # Resume centralized training from checkpoint
    python run_experiment.py --mode centralized --resume outputs/exp/checkpoints/best_checkpoint.pt

    # Run federated learning with natural non-IID
    python run_experiment.py --mode federated --rounds 50 --noniid-type natural
    
    # Run federated with Dirichlet split and custom alpha
    python run_experiment.py --mode federated --rounds 30 --noniid-type dirichlet --dirichlet-alpha 0.3
    
    # Resume federated training from checkpoint
    python run_experiment.py --mode federated --resume outputs/federated_xxx/checkpoints/checkpoint_round_10.pt

    # Evaluate a trained model checkpoint
    python run_experiment.py --mode evaluate --checkpoint outputs/exp/checkpoints/best_model.pt --datasets HAM10000

    # Run comparison experiment
    python run_experiment.py --mode comparison --config configs/experiment_config.yaml
        """,
    )
    
    # ==========================================================================
    # Mode Selection
    # ==========================================================================
    parser.add_argument(
        "--mode",
        type=str,
        choices=["centralized", "federated", "comparison", "evaluate"],
        required=True,
        help="Experiment mode: centralized, federated, comparison, or evaluate",
    )
    
    # ==========================================================================
    # Config File
    # ==========================================================================
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (overrides can still be applied via CLI)",
    )
    
    # ==========================================================================
    # Common Arguments (shared by all modes)
    # ==========================================================================
    common_group = parser.add_argument_group("Common Options")
    common_group.add_argument("--data-root", type=str, help="Root directory for datasets (default: ./data)")
    common_group.add_argument("--output-dir", type=str, help="Output directory for results (default: ./outputs)")
    common_group.add_argument("--experiment-name", type=str, help="Name for this experiment")
    common_group.add_argument("--batch-size", type=int, help="Batch size for training/evaluation")
    common_group.add_argument("--lr", type=float, help="Learning rate")
    common_group.add_argument(
        "--datasets", 
        type=str, 
        nargs="+",
        choices=["HAM10000", "ISIC2018", "ISIC2019", "ISIC2020", "PAD-UFES-20"],
        help="Specific dataset(s) to use. For FL natural non-IID, each dataset = one client"
    )
    
    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-variant",
        type=str,
        choices=["tiny", "small", "base"],
        help="DSCATNet variant: tiny (~5M params), small (~15M), base (~20M)"
    )
    model_group.add_argument("--num-classes", type=int, help="Number of output classes (default: 7)")
    model_group.add_argument("--image-size", type=int, help="Input image size (default: 224)")
    
    # ==========================================================================
    # Training Hyperparameters
    # ==========================================================================
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--weight-decay", type=float, help="Weight decay for optimizer (default: 0.01)")
    train_group.add_argument(
        "--augmentation",
        type=str,
        choices=["none", "light", "medium", "heavy"],
        help="Data augmentation level"
    )
    train_group.add_argument("--early-stopping", type=int, help="Early stopping patience (epochs/rounds without improvement)")
    train_group.add_argument("--checkpoint-interval", type=int, help="Save checkpoint every N epochs/rounds")
    train_group.add_argument("--num-workers", type=int, help="Number of data loader workers")
    
    # ==========================================================================
    # Centralized-Specific Arguments
    # ==========================================================================
    cent_group = parser.add_argument_group("Centralized Training Options")
    cent_group.add_argument("--epochs", type=int, help="Number of training epochs")
    cent_group.add_argument("--warmup-epochs", type=int, help="Number of warmup epochs for LR scheduler")
    cent_group.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "plateau"],
        help="Learning rate scheduler type"
    )
    cent_group.add_argument("--val-split", type=float, help="Validation split ratio (default: 0.15)")
    cent_group.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision (AMP)")
    
    # ==========================================================================
    # Federated-Specific Arguments
    # ==========================================================================
    fed_group = parser.add_argument_group("Federated Learning Options")
    fed_group.add_argument("--rounds", type=int, help="Number of FL communication rounds")
    fed_group.add_argument("--clients", type=int, help="Number of FL clients")
    fed_group.add_argument("--local-epochs", type=int, help="Local training epochs per FL round")
    fed_group.add_argument(
        "--noniid-type",
        type=str,
        choices=["natural", "dirichlet", "label_skew", "quantity_skew"],
        help="Non-IID distribution type for FL"
    )
    fed_group.add_argument("--dirichlet-alpha", type=float, help="Dirichlet alpha (lower = more non-IID)")
    fed_group.add_argument("--participation", type=float, help="Client participation rate per round (0.0-1.0)")
    fed_group.add_argument(
        "--client-selection",
        type=float,
        help="Fraction of clients to select each round (0.0-1.0, default: 1.0 = all)"
    )
    fed_group.add_argument(
        "--parallel-clients",
        type=int,
        help="Number of clients to train in parallel (CPU only, 0=auto, 1=sequential)"
    )
    fed_group.add_argument(
        "--train-val-split",
        type=float,
        help="Train/val split ratio (default: 0.8 = 80%% train, 20%% val)"
    )
    
    # ==========================================================================
    # Resume / Checkpoint Arguments
    # ==========================================================================
    resume_group = parser.add_argument_group("Checkpoint & Resume Options")
    resume_group.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint file to resume training from (centralized or federated)"
    )
    resume_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint/model file for evaluation (--mode evaluate)"
    )
    
    # Parse args
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("DSCATNet Federated Learning Experiment")
    print(f"Mode: {args.mode.upper()}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")
    
    # Run experiment based on mode
    if args.mode == "centralized":
        results = run_centralized(args)
    elif args.mode == "federated":
        results = run_federated(args)
    elif args.mode == "comparison":
        results = run_comparison(args)
    elif args.mode == "evaluate":
        results = run_evaluate(args)
    
    print("\nExperiment completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
