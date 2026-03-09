# =============================================================================
# Tests for Centralized Training
# =============================================================================
"""
Tests for Centralized Training.

Tests the centralized training baseline implementation.
"""

# =============================================================================
# Imports
# =============================================================================

import _pickle
import pytest

# =============================================================================
# Test Classes
# =============================================================================


class TestCentralizedConfig:
    """Tests for CentralizedConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.centralized.centralized import CentralizedConfig

        config = CentralizedConfig()

        assert config.num_epochs == 200
        assert config.batch_size == 4
        assert config.learning_rate == 1e-3  # Paper-aligned (Yadav et al.)
        assert config.scheduler_type == "none"  # Paper-aligned: no LR scheduler

    def test_config_serialization(self):
        """Test config to/from dict."""
        from src.centralized.centralized import CentralizedConfig

        config = CentralizedConfig(
            num_epochs=50,
            batch_size=64,
            experiment_name="test_cent"
        )

        config_dict = config.to_dict()
        restored = CentralizedConfig.from_dict(config_dict)

        assert restored.num_epochs == 50
        assert restored.batch_size == 64
        assert restored.experiment_name == "test_cent"


class TestCentralizedTrainer:
    """Tests for CentralizedTrainer."""

    def test_trainer_init(self, tmp_path):
        """Test trainer initialization."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="test_trainer",
            pretrained=False,
        )

        trainer = CentralizedTrainer(config)

        assert trainer.model is not None
        assert trainer.best_val_accuracy == 0.0
        assert (tmp_path / "test_trainer").exists()

    def test_output_directories(self, tmp_path):
        """Test output directory creation."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="dir_test_cent",
            pretrained=False,
        )

        # Creating the trainer should create output directories
        CentralizedTrainer(config)

        assert (tmp_path / "dir_test_cent" / "checkpoints").exists()

    def test_history_structure(self, tmp_path):
        """Test history tracking structure."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="hist_test",
            pretrained=False,
        )

        trainer = CentralizedTrainer(config)

        assert "epochs" in trainer.history
        assert "train_loss" in trainer.history
        assert "train_accuracy" in trainer.history
        assert "val_loss" in trainer.history
        assert "val_accuracy" in trainer.history
        assert "learning_rate" in trainer.history

    def test_load_checkpoint_missing_file(self, tmp_path):
        """Test loading checkpoint from non-existent file raises FileNotFoundError."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="missing_ckpt_test",
            pretrained=False,
        )

        trainer = CentralizedTrainer(config)

        missing_path = str(tmp_path / "nonexistent_checkpoint.pt")

        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(missing_path)

    def test_load_checkpoint_corrupted_file(self, tmp_path):
        """Test loading checkpoint from corrupted file raises an error."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="corrupt_ckpt_test",
            pretrained=False,
        )

        trainer = CentralizedTrainer(config)

        # Create a corrupted checkpoint file
        corrupt_path = tmp_path / "corrupted_checkpoint.pt"
        with open(corrupt_path, 'w') as f:
            f.write("This is not a valid PyTorch checkpoint")

        with pytest.raises((RuntimeError, EOFError, OSError, _pickle.UnpicklingError)):
            trainer.load_checkpoint(str(corrupt_path))

    def test_save_and_load_checkpoint_roundtrip(self, tmp_path):
        """Test saving and loading checkpoint preserves state."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        import torch

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="roundtrip_test",
            pretrained=False,
        )

        trainer = CentralizedTrainer(config)

        # Set some state
        trainer.best_val_accuracy = 0.85
        trainer.history["epochs"].append(1)
        trainer.history["train_loss"].append(0.5)

        # Create optimizer and scheduler for save_checkpoint
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        metrics = {"val_accuracy": 0.85, "val_loss": 0.3}

        # Save checkpoint using the trainer's save method
        checkpoint_path = trainer.save_checkpoint(
            epoch=5,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            is_best=True
        )

        # Create new trainer and load
        trainer2 = CentralizedTrainer(config)
        epoch = trainer2.load_checkpoint(checkpoint_path, optimizer, scheduler)

        assert epoch == 5
        assert trainer2.best_val_accuracy == 0.85

    def test_train_epoch_without_setup_data_raises(self, tmp_path):
        """train_epoch should raise if data not set up."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        import torch

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="no_data",
            pretrained=False,
        )
        trainer = CentralizedTrainer(config)
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        with pytest.raises(RuntimeError, match="train_loader"):
            trainer.train_epoch(optimizer, criterion)

    def test_evaluate_without_setup_data_raises(self, tmp_path):
        """evaluate should raise if data not set up."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="no_data_eval",
            pretrained=False,
        )
        trainer = CentralizedTrainer(config)

        with pytest.raises(RuntimeError, match="val_loader"):
            trainer.evaluate()

    def test_train_epoch_with_synthetic_data(self, tmp_path):
        """train_epoch should run with synthetic data."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="synth_train",
            pretrained=False,
            image_size=32,
        )
        trainer = CentralizedTrainer(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        trainer.train_loader = DataLoader(TensorDataset(images, labels), batch_size=4)

        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        loss, acc = trainer.train_epoch(optimizer, criterion)
        assert isinstance(loss, float) and loss >= 0
        assert isinstance(acc, float) and 0.0 <= acc <= 1.0

    def test_evaluate_with_synthetic_data(self, tmp_path):
        """evaluate should run with synthetic data."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="synth_eval",
            pretrained=False,
            image_size=32,
        )
        trainer = CentralizedTrainer(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        trainer.val_loader = DataLoader(TensorDataset(images, labels), batch_size=4)

        loss, acc, per_class = trainer.evaluate()
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert isinstance(per_class, dict)

    def test_save_checkpoint_best_creates_model_file(self, tmp_path):
        """Saving best checkpoint should also create best_model.pt."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        import torch

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="best_ckpt",
            pretrained=False,
        )
        trainer = CentralizedTrainer(config)
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)

        trainer.save_checkpoint(
            epoch=1, optimizer=optimizer, scheduler=None,
            metrics={"val_loss": 0.5}, is_best=True,
        )
        assert (trainer.checkpoint_dir / "best_checkpoint.pt").exists()
        assert (trainer.checkpoint_dir / "best_model.pt").exists()

    def test_save_checkpoint_regular(self, tmp_path):
        """Regular checkpoint should use epoch-numbered filename."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        import torch

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="regular_ckpt",
            pretrained=False,
        )
        trainer = CentralizedTrainer(config)
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)

        path = trainer.save_checkpoint(
            epoch=10, optimizer=optimizer, scheduler=None,
            metrics={}, is_best=False,
        )
        assert "checkpoint_epoch_10" in path

    def test_config_from_dict_ignores_unknown_keys(self):
        """from_dict should silently ignore unknown keys."""
        from src.centralized.centralized import CentralizedConfig
        d = {"num_epochs": 5, "unknown_key": "ignored"}
        config = CentralizedConfig.from_dict(d)
        assert config.num_epochs == 5

    def test_train_epoch_gradient_accumulation(self, tmp_path):
        """train_epoch with gradient_accumulation_steps > 1 should work."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="grad_accum",
            pretrained=False,
            image_size=32,
            gradient_accumulation_steps=2,
        )
        trainer = CentralizedTrainer(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        trainer.train_loader = DataLoader(TensorDataset(images, labels), batch_size=4)

        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        loss, _acc = trainer.train_epoch(optimizer, criterion)
        assert isinstance(loss, float)

    def test_run_with_synthetic_data(self, tmp_path):
        """run() should complete with synthetic data and mocked setup_data."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        from unittest.mock import patch
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="run_test",
            pretrained=False,
            image_size=32,
            num_epochs=2,
            checkpoint_interval=1,
            early_stopping_patience=100,
        )
        trainer = CentralizedTrainer(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        trainer.train_loader = DataLoader(
            TensorDataset(images, labels), batch_size=4, drop_last=True
        )
        trainer.val_loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        trainer.class_weights = None

        with patch.object(trainer, "setup_data"):
            results = trainer.run()

        assert "history" in results
        assert "best_val_accuracy" in results
        assert len(results["history"]["epochs"]) == 2
        assert (trainer.output_dir / "config.json").exists()
        assert (trainer.output_dir / "results.json").exists()

    def test_run_with_cosine_scheduler(self, tmp_path):
        """run() with cosine scheduler should work."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        from unittest.mock import patch
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="cosine_test",
            pretrained=False,
            image_size=32,
            num_epochs=2,
            scheduler_type="cosine",
            early_stopping_patience=100,
        )
        trainer = CentralizedTrainer(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        trainer.train_loader = DataLoader(
            TensorDataset(images, labels), batch_size=4, drop_last=True
        )
        trainer.val_loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        trainer.class_weights = None

        with patch.object(trainer, "setup_data"):
            results = trainer.run()

        assert len(results["history"]["learning_rate"]) == 2
        # Cosine scheduler should have changing LR
        assert results["history"]["learning_rate"][-1] <= results["history"]["learning_rate"][0]

    def test_run_with_plateau_scheduler(self, tmp_path):
        """run() with plateau scheduler should work."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        from unittest.mock import patch
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="plateau_test",
            pretrained=False,
            image_size=32,
            num_epochs=2,
            scheduler_type="plateau",
            early_stopping_patience=100,
        )
        trainer = CentralizedTrainer(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        trainer.train_loader = DataLoader(
            TensorDataset(images, labels), batch_size=4, drop_last=True
        )
        trainer.val_loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        trainer.class_weights = None

        with patch.object(trainer, "setup_data"):
            results = trainer.run()

        assert len(results["history"]["epochs"]) == 2

    def test_run_with_adamw(self, tmp_path):
        """run() with adamw optimizer should work."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        from unittest.mock import patch
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="adamw_test",
            pretrained=False,
            image_size=32,
            num_epochs=1,
            optimizer_type="adamw",
            early_stopping_patience=100,
        )
        trainer = CentralizedTrainer(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        trainer.train_loader = DataLoader(
            TensorDataset(images, labels), batch_size=4, drop_last=True
        )
        trainer.val_loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        trainer.class_weights = None

        with patch.object(trainer, "setup_data"):
            results = trainer.run()

        assert "best_val_accuracy" in results

    def test_run_early_stopping(self, tmp_path):
        """run() should trigger early stopping when no improvement."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer
        from unittest.mock import patch
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="early_stop_test",
            pretrained=False,
            image_size=32,
            num_epochs=100,
            early_stopping_patience=2,
        )
        trainer = CentralizedTrainer(config)

        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        trainer.train_loader = DataLoader(
            TensorDataset(images, labels), batch_size=4, drop_last=True
        )
        trainer.val_loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        trainer.class_weights = None

        with patch.object(trainer, "setup_data"):
            results = trainer.run()

        # Should stop well before 100 epochs
        assert len(results["history"]["epochs"]) < 100

    def test_run_centralized_training_convenience(self, tmp_path):
        """run_centralized_training convenience function works."""
        from src.centralized.centralized import run_centralized_training, CentralizedConfig
        from unittest.mock import patch

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="convenience_test",
            pretrained=False,
            image_size=32,
            num_epochs=1,
        )

        with patch("src.centralized.centralized.CentralizedTrainer.run", return_value={"done": True}):
            result = run_centralized_training(config)
        assert result == {"done": True}

    def test_get_transforms(self, tmp_path):
        """_get_transforms should return a pair of transforms."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="tfm_test",
            pretrained=False,
        )
        trainer = CentralizedTrainer(config)
        train_tfm, val_tfm = trainer._get_transforms()
        assert train_tfm is not None
        assert val_tfm is not None


# Integration tests that require actual data
@pytest.mark.integration
@pytest.mark.slow
class TestCentralizedTrainerIntegration:
    """Integration tests (require datasets). Run with: pytest -m slow"""

    @pytest.fixture(autouse=True)
    def check_datasets(self):
        """Skip if no datasets available."""
        from pathlib import Path
        data_root = Path(__file__).parent.parent / "data"
        ham_csv = data_root / "HAM10000" / "HAM10000_metadata.csv"
        if not ham_csv.exists():
            pytest.skip("HAM10000 dataset not available")

    def test_full_training_run(self, tmp_path):
        """Test complete training with actual data."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="full_cent_test",
            data_root="./data",
            num_epochs=2,
            pretrained=False,
        )

        trainer = CentralizedTrainer(config)
        results = trainer.run()

        assert "history" in results
        assert "best_val_accuracy" in results
        assert len(results["history"]["epochs"]) > 0
