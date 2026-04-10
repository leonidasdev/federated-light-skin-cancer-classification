# =============================================================================
# Tests for Utility Helper Functions
# =============================================================================
"""Tests for src.utils.helpers — seed, device, autocast, counting, formatting."""

import random

import numpy as np
import pytest
import torch

from src.utils.helpers import (
    autocast,
    collect_environment_info,
    compute_class_weights,
    count_parameters,
    create_grad_scaler,
    format_size,
    format_time,
    get_device,
    set_seed,
)

# =============================================================================
# Tests for set_seed
# =============================================================================


class TestSetSeed:
    """Tests for the set_seed reproducibility helper."""

    def test_deterministic_random(self):
        """Random stdlib output should be reproducible after set_seed."""
        set_seed(123)
        a = random.random()
        set_seed(123)
        b = random.random()
        assert a == b

    def test_deterministic_numpy(self):
        """Numpy random output should be reproducible after set_seed."""
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_torch(self):
        """Torch random output should be reproducible after set_seed."""
        set_seed(7)
        a = torch.rand(5)
        set_seed(7)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        """Different seeds should produce different sequences."""
        set_seed(1)
        a = random.random()
        set_seed(2)
        b = random.random()
        assert a != b


# =============================================================================
# Tests for get_device
# =============================================================================


class TestGetDevice:
    """Tests for the get_device utility."""

    def test_explicit_cpu(self):
        """Passing 'cpu' should return a CPU device."""
        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_returns_device(self):
        """Calling with None should return a valid torch.device."""
        device = get_device(None)
        assert isinstance(device, torch.device)

    def test_auto_matches_cuda_availability(self):
        """Auto-detected device should match torch.cuda availability."""
        device = get_device()
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert device.type == expected


# =============================================================================
# Tests for autocast
# =============================================================================


class TestAutocast:
    """Tests for the AMP autocast context-manager factory."""

    def test_returns_context_manager(self):
        """autocast() should return a usable context manager."""
        ctx = autocast()
        # Must have __enter__ and __exit__ (context-manager protocol)
        assert hasattr(ctx, "__enter__")
        assert hasattr(ctx, "__exit__")


# =============================================================================
# Tests for count_parameters
# =============================================================================


class TestCountParameters:
    """Tests for the parameter-counting utility."""

    def test_small_linear(self):
        """Count params of a simple Linear layer (weight + bias)."""
        model = torch.nn.Linear(10, 5)
        # 10*5 + 5 = 55
        assert count_parameters(model) == 55

    def test_no_bias(self):
        """Without bias, count should equal weight elements only."""
        model = torch.nn.Linear(8, 4, bias=False)
        assert count_parameters(model) == 32

    def test_trainable_only_flag(self):
        """Frozen parameters should be excluded when trainable_only=True."""
        model = torch.nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        assert count_parameters(model, trainable_only=True) == 0
        assert count_parameters(model, trainable_only=False) == 55

    def test_sequential(self):
        """Count parameters of a multi-layer model."""
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3, bias=False),
            torch.nn.Linear(3, 2, bias=False),
        )
        # 4*3 + 3*2 = 18
        assert count_parameters(model) == 18


# =============================================================================
# Tests for format_time
# =============================================================================


class TestFormatTime:
    """Tests for the human-readable time formatter."""

    def test_seconds_only(self):
        assert format_time(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_time(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        assert format_time(3661) == "1h 1m 1s"

    def test_zero(self):
        assert format_time(0) == "0s"

    def test_exact_hour(self):
        assert format_time(3600) == "1h 0s"


# =============================================================================
# Tests for format_size
# =============================================================================


class TestFormatSize:
    """Tests for the human-readable byte-size formatter."""

    def test_bytes(self):
        assert format_size(500) == "500.00 B"

    def test_kilobytes(self):
        assert format_size(2048) == "2.00 KB"

    def test_megabytes(self):
        assert format_size(1_048_576) == "1.00 MB"

    def test_gigabytes(self):
        assert format_size(2 * 1024**3) == "2.00 GB"

    def test_zero(self):
        assert format_size(0) == "0.00 B"

    def test_terabytes(self):
        assert format_size(1024**4) == "1.00 TB"

    def test_petabytes(self):
        assert format_size(1024**5) == "1.00 PB"


# =============================================================================
# Tests for create_grad_scaler
# =============================================================================


class TestCreateGradScaler:
    """Tests for create_grad_scaler utility."""

    def test_returns_grad_scaler(self):
        scaler = create_grad_scaler()
        # Should return some kind of GradScaler object
        assert scaler is not None
        assert hasattr(scaler, "scale")
        assert hasattr(scaler, "step")
        assert hasattr(scaler, "update")


# =============================================================================
# Tests for compute_class_weights
# =============================================================================


class TestComputeClassWeights:
    """Tests for compute_class_weights utility."""

    def test_balanced_classes(self):
        counts = {0: 100, 1: 100, 2: 100}
        w = compute_class_weights(counts, num_classes=3)
        assert w.shape == (3,)
        torch.testing.assert_close(w, torch.ones(3))

    def test_imbalanced_classes(self):
        counts = {0: 10, 1: 90}
        w = compute_class_weights(counts, num_classes=2)
        assert w[0] > w[1]  # minority class gets higher weight

    def test_missing_class_gets_zero_weight(self):
        counts = {0: 50}
        w = compute_class_weights(counts, num_classes=3)
        assert w[1] == 0.0
        assert w[2] == 0.0

    def test_negative_class_index_ignored(self):
        counts = {-1: 20, 0: 80}
        w = compute_class_weights(counts, num_classes=2)
        assert w[0] > 0.0


# =============================================================================
# Tests for collect_environment_info
# =============================================================================


class TestCollectEnvironmentInfo:
    """Tests for collect_environment_info utility."""

    def test_returns_dict(self):
        info = collect_environment_info()
        assert isinstance(info, dict)

    def test_has_basic_keys(self):
        info = collect_environment_info()
        assert "python_version" in info
        assert "platform" in info
        assert "pytorch_version" in info
        assert "cuda_available" in info

    def test_has_library_versions(self):
        info = collect_environment_info()
        # timm and flower should be found since they are installed
        assert "timm_version" in info
        assert "flower_version" in info
