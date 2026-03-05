# =============================================================================
# Tests for Checkpoint Utilities
# =============================================================================
"""Tests for src.utils.checkpoints — CheckpointManager, save/load helpers."""

import pytest
import torch

from src.utils.checkpoints import (
    CheckpointManager,
    load_model_for_inference,
    save_model_for_inference,
)


# =============================================================================
# Helpers
# =============================================================================


def _simple_model():
    """Create a tiny model for checkpoint tests."""
    return torch.nn.Linear(4, 2)


# =============================================================================
# Tests for CheckpointManager
# =============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager save/load/cleanup cycle."""

    def test_init_creates_directory(self, tmp_path):
        """Manager should create the checkpoint directory on init."""
        ckpt_dir = tmp_path / "new_checkpoints"
        mgr = CheckpointManager(ckpt_dir, max_checkpoints=3)
        assert ckpt_dir.exists()
        assert mgr.best_checkpoint is None

    def test_save_creates_file(self, tmp_path):
        """Saving a checkpoint should create a .pt file."""
        model = _simple_model()
        mgr = CheckpointManager(tmp_path, max_checkpoints=5)
        path = mgr.save(model, epoch=1)
        assert path.exists()
        assert path.name == "checkpoint_epoch_1.pt"

    def test_save_custom_filename(self, tmp_path):
        """Custom filename should be respected."""
        model = _simple_model()
        mgr = CheckpointManager(tmp_path)
        path = mgr.save(model, filename="my_model.pt")
        assert path.name == "my_model.pt"

    def test_save_best_creates_best_model_file(self, tmp_path):
        """is_best=True should also save best_model.pt."""
        model = _simple_model()
        mgr = CheckpointManager(tmp_path)
        mgr.save(model, epoch=5, is_best=True)
        assert (tmp_path / "best_model.pt").exists()
        assert mgr.best_checkpoint == tmp_path / "best_model.pt"

    def test_save_with_optimizer_and_scheduler(self, tmp_path):
        """Optimizer and scheduler states should be saved and loadable."""
        model = _simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        mgr = CheckpointManager(tmp_path)

        path = mgr.save(model, optimizer=optimizer, scheduler=scheduler, epoch=3)
        ckpt = torch.load(path, weights_only=False)

        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert ckpt["epoch"] == 3

    def test_save_metrics_persisted(self, tmp_path):
        """Metrics dict should be stored in the checkpoint."""
        model = _simple_model()
        mgr = CheckpointManager(tmp_path)
        metrics = {"accuracy": 0.95, "loss": 0.12}
        path = mgr.save(model, epoch=1, metrics=metrics)
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["metrics"]["accuracy"] == 0.95

    def test_cleanup_removes_old_checkpoints(self, tmp_path):
        """Checkpoints exceeding max_checkpoints should be cleaned up."""
        model = _simple_model()
        mgr = CheckpointManager(tmp_path, max_checkpoints=2)

        for i in range(5):
            mgr.save(model, epoch=i)

        # Only the 2 most recent regular checkpoints should remain
        remaining = list(tmp_path.glob("checkpoint_*.pt"))
        assert len(remaining) == 2

    def test_load_restores_model(self, tmp_path):
        """Loading a checkpoint should restore identical weights."""
        model = _simple_model()
        mgr = CheckpointManager(tmp_path)
        path = mgr.save(model, epoch=1)

        # Zero out weights
        original_weight = model.weight.data.clone()
        model.weight.data.zero_()
        assert not torch.equal(model.weight.data, original_weight)

        # Restore
        mgr.load(model, checkpoint_path=path)
        assert torch.equal(model.weight.data, original_weight)

    def test_load_best(self, tmp_path):
        """load_best=True should load the best_model.pt file."""
        model = _simple_model()
        mgr = CheckpointManager(tmp_path)
        mgr.save(model, epoch=5, is_best=True)

        model.weight.data.zero_()
        mgr.load(model, load_best=True)
        # Weights restored (not all zero)
        assert model.weight.data.abs().sum() > 0

    def test_load_latest_when_no_path(self, tmp_path):
        """load() without a path should pick the latest checkpoint."""
        model = _simple_model()
        mgr = CheckpointManager(tmp_path, max_checkpoints=10)
        mgr.save(model, epoch=1)
        mgr.save(model, epoch=2)
        mgr.save(model, epoch=3)

        ckpt = mgr.load(model)
        assert ckpt["epoch"] == 3

    def test_load_raises_when_empty(self, tmp_path):
        """load() should raise FileNotFoundError when no checkpoints exist."""
        model = _simple_model()
        mgr = CheckpointManager(tmp_path)
        with pytest.raises(FileNotFoundError):
            mgr.load(model)

    def test_load_with_optimizer(self, tmp_path):
        """Optimizer state should be restored on load."""
        model = _simple_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(tmp_path)
        path = mgr.save(model, optimizer=optimizer, epoch=1)

        new_optimizer = torch.optim.SGD(model.parameters(), lr=0.99)
        mgr.load(model, checkpoint_path=path, optimizer=new_optimizer)
        # LR should be restored to 0.01
        assert new_optimizer.param_groups[0]["lr"] == 0.01

    def test_get_checkpoints(self, tmp_path):
        """get_checkpoints() should list all checkpoint_*.pt files."""
        model = _simple_model()
        mgr = CheckpointManager(tmp_path, max_checkpoints=10)
        mgr.save(model, epoch=1)
        mgr.save(model, epoch=2)
        assert len(mgr.get_checkpoints()) == 2


# =============================================================================
# Tests for save/load for inference
# =============================================================================


class TestInferenceHelpers:
    """Tests for save_model_for_inference and load_model_for_inference."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saving then loading should reproduce identical predictions."""
        model = _simple_model()
        model.eval()
        x = torch.randn(1, 4)
        with torch.no_grad():
            original_output = model(x)

        save_path = tmp_path / "inference_model.pt"
        save_model_for_inference(model, save_path, config={"variant": "tiny"})

        loaded = _simple_model()
        loaded = load_model_for_inference(loaded, save_path, device=torch.device("cpu"))

        with torch.no_grad():
            loaded_output = loaded(x)

        assert torch.allclose(original_output, loaded_output)

    def test_save_includes_config(self, tmp_path):
        """Config dict should be persisted alongside weights."""
        model = _simple_model()
        save_path = tmp_path / "model.pt"
        save_model_for_inference(model, save_path, config={"num_classes": 7})

        data = torch.load(save_path, weights_only=False)
        assert data["config"]["num_classes"] == 7

    def test_loaded_model_is_in_eval_mode(self, tmp_path):
        """Loaded model should be in eval mode."""
        model = _simple_model()
        save_path = tmp_path / "model.pt"
        save_model_for_inference(model, save_path)

        loaded = _simple_model()
        loaded = load_model_for_inference(loaded, save_path, device=torch.device("cpu"))
        assert not loaded.training
