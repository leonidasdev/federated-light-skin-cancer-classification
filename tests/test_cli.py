# =============================================================================
# Test Module: CLI Argument Parsing
# =============================================================================
"""
Tests for command-line interface argument parsing and validation.

This module provides unit tests for:
- argparse configuration validity
- CLI argument validation
- Default value handling
- Mutually exclusive argument groups

Author: Leonardo Chen
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_run_experiment_parser():
    """Create a mock parser matching run_experiment.py structure."""
    parser = argparse.ArgumentParser(
        description="Run DSCATNet Federated Learning Experiments"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["centralized", "federated", "comparison", "evaluate"],
        required=True,
        help="Experiment mode",
    )
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--rounds", type=int, help="Number of FL rounds")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["HAM10000", "ISIC2018", "ISIC2019", "ISIC2020", "PAD-UFES-20"],
        help="Datasets to use",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=["tiny", "small", "base"],
        help="Model variant",
    )
    parser.add_argument("--participation", type=float, help="Client participation rate")
    parser.add_argument("--dirichlet-alpha", type=float, help="Dirichlet alpha")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")

    return parser


@pytest.fixture
def mock_fl_parser():
    """Create a mock parser matching run_fl.py structure."""
    parser = argparse.ArgumentParser(description="Quick Start FL Experiment")

    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--dataset", type=str, default="HAM10000", help="Dataset name")

    return parser


# =============================================================================
# Mode Validation Tests
# =============================================================================


class TestModeValidation:
    """Test mode argument validation."""

    def test_valid_centralized_mode(self, mock_run_experiment_parser):
        """Test that centralized mode is accepted."""
        args = mock_run_experiment_parser.parse_args(["--mode", "centralized"])
        assert args.mode == "centralized"

    def test_valid_federated_mode(self, mock_run_experiment_parser):
        """Test that federated mode is accepted."""
        args = mock_run_experiment_parser.parse_args(["--mode", "federated"])
        assert args.mode == "federated"

    def test_valid_evaluate_mode(self, mock_run_experiment_parser):
        """Test that evaluate mode is accepted."""
        args = mock_run_experiment_parser.parse_args(["--mode", "evaluate"])
        assert args.mode == "evaluate"

    def test_invalid_mode_raises(self, mock_run_experiment_parser):
        """Test that invalid mode raises error."""
        with pytest.raises(SystemExit):
            mock_run_experiment_parser.parse_args(["--mode", "invalid_mode"])

    def test_missing_mode_raises(self, mock_run_experiment_parser):
        """Test that missing required --mode raises error."""
        with pytest.raises(SystemExit):
            mock_run_experiment_parser.parse_args([])


# =============================================================================
# Dataset Argument Tests
# =============================================================================


class TestDatasetArguments:
    """Test dataset argument handling."""

    def test_single_dataset(self, mock_run_experiment_parser):
        """Test single dataset specification."""
        args = mock_run_experiment_parser.parse_args(
            ["--mode", "centralized", "--datasets", "HAM10000"]
        )
        assert args.datasets == ["HAM10000"]

    def test_multiple_datasets(self, mock_run_experiment_parser):
        """Test multiple datasets specification."""
        args = mock_run_experiment_parser.parse_args(
            ["--mode", "federated", "--datasets", "HAM10000", "ISIC2019"]
        )
        assert args.datasets == ["HAM10000", "ISIC2019"]

    def test_invalid_dataset_raises(self, mock_run_experiment_parser):
        """Test that invalid dataset name raises error."""
        with pytest.raises(SystemExit):
            mock_run_experiment_parser.parse_args(
                ["--mode", "centralized", "--datasets", "InvalidDataset"]
            )

    def test_all_datasets(self, mock_run_experiment_parser):
        """Test all valid datasets can be specified."""
        all_datasets = ["HAM10000", "ISIC2018", "ISIC2019", "ISIC2020", "PAD-UFES-20"]
        args = mock_run_experiment_parser.parse_args(
            ["--mode", "federated", "--datasets"] + all_datasets
        )
        assert args.datasets == all_datasets


# =============================================================================
# Model Variant Tests
# =============================================================================


class TestModelVariantArguments:
    """Test model variant argument handling."""

    def test_tiny_variant(self, mock_run_experiment_parser):
        """Test tiny model variant."""
        args = mock_run_experiment_parser.parse_args(
            ["--mode", "centralized", "--model-variant", "tiny"]
        )
        assert args.model_variant == "tiny"

    def test_small_variant(self, mock_run_experiment_parser):
        """Test small model variant."""
        args = mock_run_experiment_parser.parse_args(
            ["--mode", "centralized", "--model-variant", "small"]
        )
        assert args.model_variant == "small"

    def test_base_variant(self, mock_run_experiment_parser):
        """Test base model variant."""
        args = mock_run_experiment_parser.parse_args(
            ["--mode", "centralized", "--model-variant", "base"]
        )
        assert args.model_variant == "base"

    def test_invalid_variant_raises(self, mock_run_experiment_parser):
        """Test that invalid model variant raises error."""
        with pytest.raises(SystemExit):
            mock_run_experiment_parser.parse_args(
                ["--mode", "centralized", "--model-variant", "large"]
            )


# =============================================================================
# Numeric Argument Tests
# =============================================================================


class TestNumericArguments:
    """Test numeric argument parsing."""

    def test_epochs_integer(self, mock_run_experiment_parser):
        """Test epochs accepts integer."""
        args = mock_run_experiment_parser.parse_args(
            ["--mode", "centralized", "--epochs", "100"]
        )
        assert args.epochs == 100

    def test_rounds_integer(self, mock_run_experiment_parser):
        """Test rounds accepts integer."""
        args = mock_run_experiment_parser.parse_args(
            ["--mode", "federated", "--rounds", "50"]
        )
        assert args.rounds == 50

    def test_learning_rate_float(self, mock_run_experiment_parser):
        """Test learning rate accepts float."""
        args = mock_run_experiment_parser.parse_args(
            ["--mode", "centralized", "--lr", "0.001"]
        )
        assert args.lr == pytest.approx(0.001)

    def test_participation_rate(self, mock_run_experiment_parser):
        """Test participation rate accepts float."""
        args = mock_run_experiment_parser.parse_args(
            ["--mode", "federated", "--participation", "0.75"]
        )
        assert args.participation == pytest.approx(0.75)

    def test_dirichlet_alpha(self, mock_run_experiment_parser):
        """Test Dirichlet alpha accepts float."""
        args = mock_run_experiment_parser.parse_args(
            ["--mode", "federated", "--dirichlet-alpha", "0.3"]
        )
        assert args.dirichlet_alpha == pytest.approx(0.3)


# =============================================================================
# Default Value Tests
# =============================================================================


class TestDefaultValues:
    """Test default values for optional arguments."""

    def test_fl_parser_default_rounds(self, mock_fl_parser):
        """Test FL parser has default rounds."""
        args = mock_fl_parser.parse_args([])
        assert args.rounds == 10

    def test_fl_parser_default_clients(self, mock_fl_parser):
        """Test FL parser has default clients."""
        args = mock_fl_parser.parse_args([])
        assert args.clients == 5

    def test_fl_parser_default_dataset(self, mock_fl_parser):
        """Test FL parser has default dataset."""
        args = mock_fl_parser.parse_args([])
        assert args.dataset == "HAM10000"

    def test_optional_none_when_not_provided(self, mock_run_experiment_parser):
        """Test optional args are None when not provided."""
        args = mock_run_experiment_parser.parse_args(["--mode", "centralized"])
        assert args.config is None
        assert args.epochs is None
        assert args.lr is None
        assert args.resume is None


# =============================================================================
# Complex Argument Combination Tests
# =============================================================================


class TestArgumentCombinations:
    """Test complex argument combinations."""

    def test_centralized_with_all_common_args(self, mock_run_experiment_parser):
        """Test centralized mode with common arguments."""
        args = mock_run_experiment_parser.parse_args([
            "--mode", "centralized",
            "--epochs", "50",
            "--lr", "0.0001",
            "--datasets", "HAM10000",
            "--model-variant", "small",
        ])
        assert args.mode == "centralized"
        assert args.epochs == 50
        assert args.lr == pytest.approx(0.0001)
        assert args.datasets == ["HAM10000"]
        assert args.model_variant == "small"

    def test_federated_with_all_fl_args(self, mock_run_experiment_parser):
        """Test federated mode with FL-specific arguments."""
        args = mock_run_experiment_parser.parse_args([
            "--mode", "federated",
            "--rounds", "30",
            "--participation", "0.5",
            "--dirichlet-alpha", "0.1",
            "--datasets", "HAM10000", "ISIC2018",
        ])
        assert args.mode == "federated"
        assert args.rounds == 30
        assert args.participation == pytest.approx(0.5)
        assert args.dirichlet_alpha == pytest.approx(0.1)

    def test_config_with_override(self, mock_run_experiment_parser):
        """Test config file with CLI override."""
        args = mock_run_experiment_parser.parse_args([
            "--mode", "federated",
            "--config", "configs/fl_config.yaml",
            "--rounds", "20",  # Override config
        ])
        assert args.config == "configs/fl_config.yaml"
        assert args.rounds == 20

    def test_resume_argument(self, mock_run_experiment_parser):
        """Test resume checkpoint argument."""
        args = mock_run_experiment_parser.parse_args([
            "--mode", "centralized",
            "--resume", "outputs/exp/checkpoints/best_model.pt",
        ])
        assert args.resume == "outputs/exp/checkpoints/best_model.pt"
