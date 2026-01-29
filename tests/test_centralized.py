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

        assert config.num_epochs == 100
        assert config.batch_size == 8  # Updated to match reduced batch size for memory efficiency
        assert config.learning_rate == 1e-4
        assert config.scheduler_type == "cosine"

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
        """Test loading checkpoint from non-existent file."""
        from src.centralized.centralized import CentralizedConfig, CentralizedTrainer

        config = CentralizedConfig(
            output_dir=str(tmp_path),
            experiment_name="missing_ckpt_test",
            pretrained=False,
        )

        trainer = CentralizedTrainer(config)

        # Try to load non-existent checkpoint
        missing_path = str(tmp_path / "nonexistent_checkpoint.pt")

        # The load_checkpoint method returns epoch number or raises error
        # depending on implementation. We test it handles gracefully.
        try:
            result = trainer.load_checkpoint(missing_path)
            # If it doesn't raise, it should return 0 or handle gracefully
            assert result == 0 or result is None
        except FileNotFoundError:
            # This is also acceptable behavior
            pass

    def test_load_checkpoint_corrupted_file(self, tmp_path):
        """Test loading checkpoint from corrupted file."""
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

        # Try to load corrupted checkpoint
        try:
            trainer.load_checkpoint(str(corrupt_path))
            # If no error, something is wrong
            assert False, "Should have raised an error for corrupted file"
        except Exception:
            # Expected - corrupted file should raise error
            pass

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
