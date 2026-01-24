"""
Tests for Centralized Training.

Tests the centralized training baseline implementation.
"""

import pytest


class TestCentralizedConfig:
    """Tests for CentralizedConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from src.centralized.centralized import CentralizedConfig
        
        config = CentralizedConfig()
        
        assert config.num_epochs == 100
        assert config.batch_size == 32
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
        
        trainer = CentralizedTrainer(config)
        
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


# Skip integration tests that require actual data
@pytest.mark.skip(reason="Requires actual datasets")
class TestCentralizedTrainerIntegration:
    """Integration tests (require datasets)."""
    
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
