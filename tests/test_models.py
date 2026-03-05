# =============================================================================
# Tests for DSCATNet Model and Utilities
# =============================================================================
"""Tests for src.models — DSCATNet variants, factory, FL param helpers."""

import numpy as np
import pytest
import torch

from src.models.dscatnet import (
    DSCATNet,
    create_dscatnet,
    get_model_parameters,
    set_model_parameters,
)
from src.models.patch_embedding import DualScalePatchEmbedding


# =============================================================================
# DSCATNet Factory Tests
# =============================================================================


class TestCreateDSCATNet:
    """Tests for the create_dscatnet factory function."""

    def test_tiny_variant(self):
        """Tiny variant should create a model with embed_dim=192."""
        model = create_dscatnet(num_classes=7, variant="tiny")
        assert isinstance(model, DSCATNet)
        assert model.embed_dim == 192

    def test_small_variant(self):
        """Small variant should create a model with embed_dim=384, depth=6."""
        model = create_dscatnet(num_classes=7, variant="small")
        assert model.embed_dim == 384
        assert model.depth == 6

    def test_base_variant(self):
        """Base variant should create a model with embed_dim=384, depth=8."""
        model = create_dscatnet(num_classes=7, variant="base")
        assert model.embed_dim == 384
        assert model.depth == 8

    def test_invalid_variant_raises(self):
        """Unknown variant should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown variant"):
            create_dscatnet(variant="huge")

    def test_custom_num_classes(self):
        """Model should respect custom num_classes."""
        model = create_dscatnet(num_classes=3, variant="tiny")
        assert model.num_classes == 3

    def test_pretrained_flag_ignored_gracefully(self):
        """pretrained=True should be ignored without error."""
        model = create_dscatnet(variant="tiny", pretrained=True)
        assert isinstance(model, DSCATNet)

    def test_unknown_kwargs_ignored(self):
        """Extra kwargs not in DSCATNet.__init__ should be silently dropped."""
        model = create_dscatnet(variant="tiny", some_unknown_key=42)
        assert isinstance(model, DSCATNet)


# =============================================================================
# DSCATNet Model Tests
# =============================================================================


class TestDSCATNet:
    """Tests for the DSCATNet model itself."""

    @pytest.fixture
    def tiny_model(self):
        """Create a tiny DSCATNet for testing."""
        return create_dscatnet(num_classes=7, variant="tiny", img_size=224)

    def test_forward_shape(self, tiny_model):
        """Forward pass should return (B, num_classes) logits."""
        x = torch.randn(2, 3, 224, 224)
        out = tiny_model(x)
        assert out.shape == (2, 7)

    def test_forward_features_shape(self, tiny_model):
        """forward_features should return (B, embed_dim) tensor."""
        x = torch.randn(2, 3, 224, 224)
        features = tiny_model.forward_features(x)
        assert features.shape == (2, tiny_model.embed_dim)

    def test_get_num_parameters(self, tiny_model):
        """get_num_parameters should return a positive integer."""
        n = tiny_model.get_num_parameters()
        assert isinstance(n, int)
        assert n > 0

    def test_get_model_config(self, tiny_model):
        """get_model_config should return a dict with expected keys."""
        config = tiny_model.get_model_config()
        expected_keys = {"img_size", "num_classes", "embed_dim", "depth", "fusion_method", "num_parameters"}
        assert expected_keys == set(config.keys())
        assert config["num_classes"] == 7
        assert config["num_parameters"] == tiny_model.get_num_parameters()

    def test_fusion_method_concat(self):
        """Model with fusion_method='concat' should work."""
        model = DSCATNet(num_classes=7, embed_dim=192, depth=2, num_heads=3, fusion_method="concat")
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 7)

    def test_fusion_method_add(self):
        """Model with fusion_method='add' should work."""
        model = DSCATNet(num_classes=7, embed_dim=192, depth=2, num_heads=3, fusion_method="add")
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 7)


# =============================================================================
# FL Utility Functions Tests
# =============================================================================


class TestFLParamUtils:
    """Tests for get_model_parameters and set_model_parameters."""

    def test_get_model_parameters_returns_numpy(self):
        """get_model_parameters should return a list of numpy arrays."""
        model = torch.nn.Linear(4, 2)
        params = get_model_parameters(model)
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_get_model_parameters_count(self):
        """Number of arrays should match state_dict entries."""
        model = torch.nn.Linear(4, 2)  # weight + bias = 2 entries
        params = get_model_parameters(model)
        assert len(params) == 2

    def test_set_model_parameters_restores_weights(self):
        """set_model_parameters should restore identical weights."""
        model = torch.nn.Linear(4, 2)
        original_params = get_model_parameters(model)

        # Zero out
        model.weight.data.zero_()
        model.bias.data.zero_()

        # Restore
        set_model_parameters(model, original_params)

        restored = get_model_parameters(model)
        for orig, rest in zip(original_params, restored):
            np.testing.assert_array_almost_equal(orig, rest)

    def test_roundtrip_preserves_forward_output(self):
        """get then set should preserve model outputs."""
        model = torch.nn.Linear(4, 2)
        model.eval()

        x = torch.randn(1, 4)
        with torch.no_grad():
            original_out = model(x).clone()

        params = get_model_parameters(model)
        set_model_parameters(model, params)

        with torch.no_grad():
            restored_out = model(x)

        assert torch.allclose(original_out, restored_out, atol=1e-6)


# =============================================================================
# Patch Embedding Tests
# =============================================================================


class TestDualScalePatchEmbedding:
    """Tests for the DualScalePatchEmbedding module."""

    def test_get_num_patches(self):
        """get_num_patches should return (fine_patches, coarse_patches)."""
        emb = DualScalePatchEmbedding(img_size=224, embed_dim=192)
        fine, coarse = emb.get_num_patches()
        assert fine > 0
        assert coarse > 0
        assert fine > coarse  # Fine patches should be more numerous
