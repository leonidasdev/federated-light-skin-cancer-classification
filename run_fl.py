#!/usr/bin/env python
"""
Federated Learning Quick Start Script.

A simplified script to quickly run FL experiments with sensible defaults.
For more control, use run_experiment.py with custom configs.

Usage:
    python run_fl.py              # Run with defaults
    python run_fl.py --quick      # Quick test run
    python run_fl.py --full       # Full experiment
    python run_fl.py --datasets HAM10000 ISIC2019  # Specific datasets
"""

import argparse
import sys
import torch


def main():
    parser = argparse.ArgumentParser(description="Quick Start FL Experiment")

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run (5 rounds, 2 epochs)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full experiment (50 rounds)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Data directory"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["HAM10000", "ISIC2018", "ISIC2019", "ISIC2020", "PAD-UFES-20"],
        help="Specific dataset(s) to use (default: all available)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint file to resume training from"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full configuration on start"
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from src.federated.simulation import SimulationConfig, FLSimulator

    # Build base config kwargs
    config_kwargs = {
        "data_root": args.data_root,
    }

    # Add datasets if specified
    if args.datasets:
        config_kwargs["datasets"] = args.datasets

    # Configure based on mode
    if args.quick:
        config = SimulationConfig(
            num_rounds=5,
            local_epochs=1,
            batch_size=8,
            checkpoint_interval=5,
            early_stopping_patience=3,
            experiment_name="fl_quick_test",
            **config_kwargs,
        )
        print("Quick test mode: 5 rounds, 1 local epoch")
    elif args.full:
        config = SimulationConfig(
            num_rounds=50,
            local_epochs=3,
            batch_size=16,
            checkpoint_interval=5,
            early_stopping_patience=10,
            experiment_name="fl_full_experiment",
            **config_kwargs,
        )
        print("Full experiment mode: 50 rounds, 3 local epochs")
    else:
        config = SimulationConfig(
            num_rounds=20,
            local_epochs=2,
            batch_size=16,
            checkpoint_interval=5,
            early_stopping_patience=7,
            experiment_name="fl_default",
            **config_kwargs,
        )
        print("Default mode: 20 rounds, 2 local epochs")

    # Print device info
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        device += f" ({torch.cuda.get_device_name(0)})"
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    if args.datasets:
        print(f"Datasets: {', '.join(args.datasets)}")
    print()

    # Print full config if verbose
    if args.verbose:
        print("=" * 50)
        print("Configuration:")
        print("=" * 50)
        for key, value in vars(config).items():
            print(f"  {key}: {value}")
        print("=" * 50)
        print()

    # Run simulation (with optional resume)
    simulator = FLSimulator(config)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        results = simulator.run(resume_from=args.resume)
    else:
        results = simulator.run()

    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Best Accuracy:     {results['best_val_accuracy']:.4f}")
    print(f"Best Round:        {results['best_round']}")
    print(f"Total Time:        {results['total_time_seconds']/60:.1f} minutes")
    print(f"Communication:     {results['total_communication_mb']:.1f} MB")
    print(f"Results saved to:  {config.output_dir}/{config.experiment_name}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
