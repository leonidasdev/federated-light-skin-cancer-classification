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
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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


def load_config(config_path: str) -> dict[str, Any]:
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
    with open(config_path) as f:
        return yaml.safe_load(f)


# Section-to-field mappings shared by centralized and federated config flattening.
# Each entry: (yaml_section_name, [(yaml_key, config_field_name), ...])
_COMMON_SECTION_MAPPINGS = [
    (None, [  # Top-level keys
        ("data_root", "data_root"),
        ("output_dir", "output_dir"),
        ("datasets", "datasets"),
    ]),
    ("experiment", [
        ("name", "experiment_name"),
    ]),
    ("model", [
        ("image_size", "image_size"),
        ("variant", "model_variant"),
    ]),
    ("augmentation", [
        ("level", "augmentation_level"),
        ("use_dermoscopy_norm", "use_dermoscopy_norm"),
    ]),
    ("evaluation", [
        ("early_stopping_patience", "early_stopping_patience"),
        ("use_class_weights", "use_class_weights"),
        ("checkpoint_interval", "checkpoint_interval"),
        ("max_grad_norm", "max_grad_norm"),
    ]),
]


def _flatten_config(
    config_dict: dict[str, Any],
    extra_mappings: list,
) -> dict[str, Any]:
    """Flatten a nested YAML config dict into a flat dict for dataclass construction.

    Args:
        config_dict: The nested YAML config dict (e.g. centralized or federated section).
        extra_mappings: Additional section mappings beyond the common ones.

    Returns:
        Flat dict suitable for passing to ``Config.from_dict()``.
    """
    flat: dict[str, Any] = {}
    all_mappings = _COMMON_SECTION_MAPPINGS + extra_mappings

    for section_name, field_pairs in all_mappings:
        source = config_dict if section_name is None else config_dict.get(section_name, {})
        for yaml_key, config_key in field_pairs:
            if yaml_key in source:
                flat[config_key] = source[yaml_key]

    return flat


def _apply_cli_overrides(config: Any, args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to a config object (centralized or federated).

    Handles all common CLI flags shared between modes. Mode-specific overrides
    are applied by the caller after this function returns.

    Args:
        config: A CentralizedConfig or SimulationConfig instance.
        args: Parsed CLI arguments.
    """
    # Common overrides shared by both modes
    _CLI_COMMON = [
        ("batch_size", "batch_size"),
        ("lr", "learning_rate"),
        ("data_root", "data_root"),
        ("output_dir", "output_dir"),
        ("datasets", "datasets"),
        ("model_variant", "model_variant"),
        ("image_size", "image_size"),
        ("num_classes", "num_classes"),
        ("augmentation", "augmentation_level"),
        ("num_workers", "num_workers"),
        ("resume", "resume_from"),
    ]

    for arg_name, config_field in _CLI_COMMON:
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, config_field, val)

    # weight_decay and other flags that use ``is not None`` checks
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.early_stopping is not None:
        config.early_stopping_patience = args.early_stopping
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = args.checkpoint_interval
    if args.no_amp and hasattr(config, "use_amp"):
        config.use_amp = False


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate a trained model checkpoint using the DATASET_REGISTRY."""
    from src.models.dscatnet import create_dscatnet
    from src.evaluation.metrics import ModelEvaluator
    from src.data.datasets import (
        DATASET_REGISTRY, get_dataset_paths, normalize_dataset_name,
        get_available_datasets, DatasetSubset
    )
    from src.data.preprocessing import get_val_transforms
    from src.data.splits import deterministic_train_val_split
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
            logger.warning(f"Unknown dataset: {name}, skipping. Valid datasets: {', '.join(get_available_datasets())}")
            continue

        config = DATASET_REGISTRY[name]
        csv_path, dataset_root = get_dataset_paths(name, data_root)

        if csv_path is None or not csv_path.exists():
            logger.warning(
                f"Dataset {name}: CSV file not found at {csv_path}. "
                "Run 'python run_download.py --verify' to check setup."
            )
            continue

        if dataset_root is None or not dataset_root.exists():
            logger.warning(
                f"Dataset {name}: Image directory not found at {dataset_root}. "
                "Run 'python run_download.py --instructions' for setup help."
            )
            continue

        try:
            dataset = config.dataset_class(
                root_dir=str(dataset_root),
                csv_path=str(csv_path),
                transform=val_transform,
            )
            # Use last 20% as test set (same split logic as training)
            _, test_indices = deterministic_train_val_split(len(dataset), val_split=0.2)
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
        num_workers=min(4, os.cpu_count() or 1),
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

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_dict = results.to_dict()
        results_dict["checkpoint"] = str(checkpoint_path)
        results_dict["datasets"] = dataset_names

        results_file = output_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return results.to_dict()


def run_centralized(args: argparse.Namespace) -> dict[str, Any]:
    """Run centralized training experiment."""
    from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
    from src.utils.helpers import set_seed

    # Ensure reproducibility (seeds random, numpy, torch, cuDNN)
    set_seed(42)

    # Centralized-specific config section mappings
    _CENTRALIZED_SECTIONS = [
        ("model", [
            ("num_classes", "num_classes"),
            ("pretrained", "pretrained"),
        ]),
        ("training", [
            ("batch_size", "batch_size"),
            ("lr", "learning_rate"),
            ("epochs", "num_epochs"),
            ("weight_decay", "weight_decay"),
            ("warmup_epochs", "warmup_epochs"),
            ("scheduler", "scheduler_type"),
            ("min_lr", "min_lr"),
            ("optimizer", "optimizer_type"),
            ("gradient_accumulation_steps", "gradient_accumulation_steps"),
            ("use_amp", "use_amp"),
        ]),
        ("splits", [
            ("val_split", "val_split"),
            ("test_split", "test_split"),
        ]),
    ]

    # Load config if provided
    if args.config:
        config_dict = load_config(args.config)
        cent_config = config_dict.get("centralized", {})
        flat_config = _flatten_config(cent_config, _CENTRALIZED_SECTIONS)
        config = CentralizedConfig.from_dict(flat_config)
    else:
        config = CentralizedConfig()

    # Apply common CLI overrides
    _apply_cli_overrides(config, args)

    # Centralized-specific CLI overrides
    if args.epochs:
        config.num_epochs = args.epochs
    if args.warmup_epochs is not None:
        config.warmup_epochs = args.warmup_epochs
    if args.scheduler:
        config.scheduler_type = args.scheduler
    if args.val_split is not None:
        config.val_split = args.val_split

    # Experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    elif not args.config:
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


def run_federated(args: argparse.Namespace) -> dict[str, Any]:
    """Run federated learning experiment."""
    from src.federated.simulation import SimulationConfig, FLSimulator
    from src.utils.helpers import set_seed

    # Ensure reproducibility (seeds random, numpy, torch, cuDNN)
    set_seed(42)

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

    # Federated-specific config section mappings
    _FEDERATED_SECTIONS = [
        ("model", [
            ("num_classes", "num_classes"),
            ("pretrained", "pretrained"),
        ]),
        ("training", [
            ("batch_size", "batch_size"),
            ("lr", "learning_rate"),
            ("weight_decay", "weight_decay"),
            ("optimizer", "optimizer_type"),
            ("gradient_accumulation_steps", "gradient_accumulation_steps"),
            ("local_epochs", "local_epochs"),
            ("num_rounds", "num_rounds"),
            ("rounds", "num_rounds"),
            ("train_val_split", "train_val_split"),
        ]),
        ("federation", [
            ("num_clients", "num_clients"),
            ("num_rounds", "num_rounds"),
            ("noniid_type", "noniid_type"),
            ("dirichlet_alpha", "dirichlet_alpha"),
        ]),
    ]

    # Load config: priority is CLI args > YAML config > checkpoint config > defaults
    if args.config:
        config_dict = load_config(args.config)
        fed_config = config_dict.get("federated", {})
        flat_config = _flatten_config(fed_config, _FEDERATED_SECTIONS)

        # Special handling: federation.participation maps to two fields
        fed_section = fed_config.get("federation", {})
        if "participation" in fed_section:
            flat_config["fraction_fit"] = fed_section["participation"]
            flat_config["fraction_evaluate"] = fed_section["participation"]

        config = SimulationConfig.from_dict(flat_config)
    elif checkpoint_config:
        # No YAML config provided but resuming from checkpoint - use checkpoint's config
        logger.info("Using config from checkpoint as base (no YAML config provided)")
        config = SimulationConfig.from_dict(checkpoint_config)
    else:
        config = SimulationConfig()

    # Apply common CLI overrides
    _apply_cli_overrides(config, args)

    # Federated-specific CLI overrides
    if args.rounds:
        config.num_rounds = args.rounds
    if args.clients:
        config.num_clients = args.clients
    if args.local_epochs:
        config.local_epochs = args.local_epochs
    if args.noniid_type:
        config.noniid_type = args.noniid_type
    if args.dirichlet_alpha:
        config.dirichlet_alpha = args.dirichlet_alpha
    if args.datasets:
        # Auto-adjust num_clients to match selected datasets for natural non-IID
        if config.noniid_type == "natural":
            config.num_clients = len(args.datasets)
    if args.participation is not None:
        config.fraction_fit = args.participation
        config.fraction_evaluate = args.participation
    if args.client_selection is not None:
        config.client_selection_fraction = args.client_selection
    if args.parallel_clients is not None:
        config.parallel_clients = args.parallel_clients
    if args.train_val_split is not None:
        config.train_val_split = args.train_val_split

    # Experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    elif not args.config:
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


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Run both centralized and federated experiments for comparison."""
    from src.evaluation.visualization import (
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
    comparison_summary: dict[str, Any] = {
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
    gap = comparison_summary['accuracy_gap']
    gap_pct = comparison_summary['accuracy_gap_pct']
    logger.info(f"Accuracy Gap:              {gap:.4f} ({gap_pct:.2f}%)")
    logger.info("=" * 60)

    return comparison_summary


def main():
    # Version info
    __version__ = "1.0.0"

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

    # =============================================================================
    # Mode Selection
    # =============================================================================
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["centralized", "federated", "comparison", "evaluate"],
        required=True,
        help="Experiment mode: centralized, federated, comparison, or evaluate",
    )

    # =============================================================================
    # Config File
    # =============================================================================
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (overrides can still be applied via CLI)",
    )

    # =============================================================================
    # Common Arguments (shared by all modes)
    # =============================================================================
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
    common_group.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets with their details and exit"
    )
    common_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show effective configuration and exit without running experiment"
    )
    common_group.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate config file and check for errors, then exit"
    )

    # =============================================================================
    # Model Configuration
    # =============================================================================
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-variant",
        type=str,
        choices=["tiny", "small", "paper", "base"],
        help="DSCATNet variant: tiny (~5M), small (~29.4M), paper (~29.4M, 12 heads), base (~39M)"
    )
    model_group.add_argument("--num-classes", type=int, help="Number of output classes (default: 7)")
    model_group.add_argument("--image-size", type=int, help="Input image size (default: 224)")

    # =============================================================================
    # Training Hyperparameters
    # =============================================================================
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--weight-decay", type=float, help="Weight decay for optimizer (default: 0.0)")
    train_group.add_argument(
        "--augmentation",
        type=str,
        choices=["none", "light", "medium", "heavy"],
        help="Data augmentation level"
    )
    train_group.add_argument(
        "--early-stopping", type=int,
        help="Early stopping patience (epochs/rounds without improvement)"
    )
    train_group.add_argument("--checkpoint-interval", type=int, help="Save checkpoint every N epochs/rounds")
    train_group.add_argument("--num-workers", type=int, help="Number of data loader workers")

    # =============================================================================
    # Centralized-Specific Arguments
    # =============================================================================
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

    # =============================================================================
    # Federated-Specific Arguments
    # =============================================================================
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
        help="Fraction of clients to select each round (0.0-1.0, e.g., 0.75 = select 75%% of clients randomly)"
    )
    fed_group.add_argument(
        "--parallel-clients",
        type=int,
        help="Number of clients to train in parallel (CPU only, 0=auto, 1=sequential, e.g., 4 for quad-core)"
    )
    fed_group.add_argument(
        "--train-val-split",
        type=float,
        help="Validation set fraction (e.g., 0.15 = 15%% for validation, 85%% for training)"
    )

    # =============================================================================
    # Resume / Checkpoint Arguments
    # =============================================================================
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

    # Handle --list-datasets flag
    if args.list_datasets:
        from src.data.datasets import DATASET_REGISTRY
        print("\n" + "=" * 60)
        print("Available Datasets")
        print("=" * 60)
        for name, config in DATASET_REGISTRY.items():
            print(f"\n[*] {name}")
            print(f"    CSV File:    {config.csv_filename}")
            print(f"    Image Subdir: {config.image_subdir or '(root)'}")
            print(f"    Class:       {config.dataset_class.__name__}")
        print("\n" + "=" * 60)
        return 0

    # Handle --validate-config flag
    if args.validate_config:
        if not args.config:
            print("Error: --validate-config requires --config <file>")
            return 1

        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            return 1

        try:
            config = load_config(args.config)
            print("\n" + "=" * 60)
            print("Config Validation: PASSED")
            print("=" * 60)
            print(f"File: {args.config}")
            print(f"Top-level keys: {list(config.keys())}")

            # Check for common issues
            warnings = []
            if "centralized" in config and "federated" in config:
                warnings.append("Both 'centralized' and 'federated' sections present")
            if "model" in config:
                model_cfg = config["model"]
                if "variant" in model_cfg and model_cfg["variant"] not in ["tiny", "small", "base"]:
                    warnings.append(f"Unknown model variant: {model_cfg['variant']}")

            if warnings:
                print("\nWarnings:")
                for w in warnings:
                    print(f"  ⚠ {w}")
            else:
                print("\n✓ No warnings found")

            print("=" * 60)
            return 0
        except yaml.YAMLError as e:
            print(f"Error: Invalid YAML syntax in {args.config}")
            print(f"Details: {e}")
            return 1

    # Handle --dry-run flag
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Configuration Preview")
        print("=" * 60)
        print(f"Mode: {args.mode}")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

        if args.config:
            config = load_config(args.config)
            print(f"\nConfig file: {args.config}")
            print("\nEffective configuration:")
            for key, value in config.items():
                if isinstance(value, dict):
                    print(f"\n[{key}]")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")

        print("\nCLI overrides:")
        cli_overrides = {}
        if args.epochs:
            cli_overrides["epochs"] = args.epochs
        if args.rounds:
            cli_overrides["rounds"] = args.rounds
        if args.lr:
            cli_overrides["lr"] = args.lr
        if args.datasets:
            cli_overrides["datasets"] = args.datasets
        if args.model_variant:
            cli_overrides["model_variant"] = args.model_variant

        if cli_overrides:
            for k, v in cli_overrides.items():
                print(f"  --{k.replace('_', '-')}: {v}")
        else:
            print("  (none)")

        print("\n" + "=" * 60)
        print("Use without --dry-run to execute experiment")
        return 0

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
        run_centralized(args)
    elif args.mode == "federated":
        run_federated(args)
    elif args.mode == "comparison":
        run_comparison(args)
    elif args.mode == "evaluate":
        run_evaluate(args)

    print("\nExperiment completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
