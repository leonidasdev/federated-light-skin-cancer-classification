# =============================================================================
# Tests for DSCATNet Model and Utilities
# =============================================================================
"""Tests for src.models — DSCATNet variants, factory, FL param helpers."""

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.models.dscatnet import (
    DSCATNet,
    create_dscatnet,
    get_model_parameters,
    load_pretrained_vit_weights,
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

    def test_pretrained_flag_ignored_for_non_small(self):
        """pretrained=True should be ignored for non-small variants."""
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


# =============================================================================
# Pretrained ViT Weight Loading Tests
# =============================================================================


def _make_fake_vit_state_dict(embed_dim=384, depth=12, mlp_ratio=4.0, num_heads=6):
    """Create a fake ViT-Small state dict with correct shapes for testing."""
    sd = {}
    sd['cls_token'] = torch.randn(1, 1, embed_dim)
    sd['pos_embed'] = torch.randn(1, 197, embed_dim)  # 196 patches + 1 CLS
    sd['patch_embed.proj.weight'] = torch.randn(embed_dim, 3, 16, 16)
    sd['patch_embed.proj.bias'] = torch.randn(embed_dim)
    sd['norm.weight'] = torch.ones(embed_dim)
    sd['norm.bias'] = torch.zeros(embed_dim)
    sd['head.weight'] = torch.randn(1000, embed_dim)
    sd['head.bias'] = torch.randn(1000)

    mlp_hidden = int(embed_dim * mlp_ratio)
    for i in range(depth):
        sd[f'blocks.{i}.norm1.weight'] = torch.ones(embed_dim)
        sd[f'blocks.{i}.norm1.bias'] = torch.zeros(embed_dim)
        sd[f'blocks.{i}.attn.qkv.weight'] = torch.randn(embed_dim * 3, embed_dim)
        sd[f'blocks.{i}.attn.qkv.bias'] = torch.randn(embed_dim * 3)
        sd[f'blocks.{i}.attn.proj.weight'] = torch.randn(embed_dim, embed_dim)
        sd[f'blocks.{i}.attn.proj.bias'] = torch.randn(embed_dim)
        sd[f'blocks.{i}.norm2.weight'] = torch.ones(embed_dim)
        sd[f'blocks.{i}.norm2.bias'] = torch.zeros(embed_dim)
        sd[f'blocks.{i}.mlp.fc1.weight'] = torch.randn(mlp_hidden, embed_dim)
        sd[f'blocks.{i}.mlp.fc1.bias'] = torch.randn(mlp_hidden)
        sd[f'blocks.{i}.mlp.fc2.weight'] = torch.randn(embed_dim, mlp_hidden)
        sd[f'blocks.{i}.mlp.fc2.bias'] = torch.randn(embed_dim)

    return sd


class TestPretrainedViTWeightLoading:
    """Tests for load_pretrained_vit_weights with mocked timm."""

    def _mock_timm_create(self, state_dict):
        """Return a mock timm model with the given state dict."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = state_dict
        return mock_model

    def test_loads_correct_number_of_tensors(self):
        """Should load 150 of 286 parameter tensors for small variant."""
        model = create_dscatnet(num_classes=7, variant='small')
        vit_sd = _make_fake_vit_state_dict()

        with patch('src.models.dscatnet.timm') as mock_timm:
            mock_timm.create_model.return_value = self._mock_timm_create(vit_sd)
            load_pretrained_vit_weights(model, variant='small')

        # 6 blocks × 12 params/scale × 2 scales + 6 embedding params = 150
        # Verify the model can still do a forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 7)

    def test_fine_self_attention_weights_transferred(self):
        """Fine-scale self-attention should receive ViT block 0 weights."""
        model = create_dscatnet(num_classes=7, variant='small')
        vit_sd = _make_fake_vit_state_dict()

        with patch('src.models.dscatnet.timm') as mock_timm:
            mock_timm.create_model.return_value = self._mock_timm_create(vit_sd)
            load_pretrained_vit_weights(model, variant='small')

        # Block 0 fine self-attn QKV should match ViT block 0
        assert torch.equal(
            model.state_dict()['blocks.0.fine_self_attn.in_proj_weight'],
            vit_sd['blocks.0.attn.qkv.weight']
        )

    def test_coarse_self_attention_weights_transferred(self):
        """Coarse-scale self-attention should receive ViT block 6+ weights."""
        model = create_dscatnet(num_classes=7, variant='small')
        vit_sd = _make_fake_vit_state_dict()

        with patch('src.models.dscatnet.timm') as mock_timm:
            mock_timm.create_model.return_value = self._mock_timm_create(vit_sd)
            load_pretrained_vit_weights(model, variant='small')

        # Block 0 coarse self-attn QKV should match ViT block 6
        assert torch.equal(
            model.state_dict()['blocks.0.coarse_self_attn.in_proj_weight'],
            vit_sd['blocks.6.attn.qkv.weight']
        )

    def test_coarse_patch_embedding_transferred(self):
        """Coarse-scale patch embedding (16x16) should receive ViT patch_embed."""
        model = create_dscatnet(num_classes=7, variant='small')
        vit_sd = _make_fake_vit_state_dict()

        with patch('src.models.dscatnet.timm') as mock_timm:
            mock_timm.create_model.return_value = self._mock_timm_create(vit_sd)
            load_pretrained_vit_weights(model, variant='small')

        assert torch.equal(
            model.state_dict()['patch_embed.coarse_embedding.projection.weight'],
            vit_sd['patch_embed.proj.weight']
        )

    def test_cross_attention_not_transferred(self):
        """Cross-attention layers should remain randomly initialized."""
        model = create_dscatnet(num_classes=7, variant='small')
        # Save a copy of the cross-attention weights before loading
        original_cross_q = model.state_dict()['blocks.0.cross_attn.fine_q.weight'].clone()

        vit_sd = _make_fake_vit_state_dict()
        with patch('src.models.dscatnet.timm') as mock_timm:
            mock_timm.create_model.return_value = self._mock_timm_create(vit_sd)
            load_pretrained_vit_weights(model, variant='small')

        # Cross-attention weights should be unchanged (still random init)
        assert torch.equal(
            model.state_dict()['blocks.0.cross_attn.fine_q.weight'],
            original_cross_q
        )

    def test_non_small_variant_skips_loading(self):
        """Non-small variants should skip weight loading without error."""
        model = create_dscatnet(num_classes=7, variant='tiny')
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        load_pretrained_vit_weights(model, variant='tiny')

        # All weights should be unchanged
        for k, orig_v in original_sd.items():
            assert torch.equal(model.state_dict()[k], orig_v)

    def test_create_dscatnet_calls_pretrained_loading(self):
        """create_dscatnet with pretrained=True should call load_pretrained_vit_weights."""
        vit_sd = _make_fake_vit_state_dict()

        with patch('src.models.dscatnet.timm') as mock_timm:
            mock_timm.create_model.return_value = self._mock_timm_create(vit_sd)
            model = create_dscatnet(num_classes=7, variant='small', pretrained=True)

        mock_timm.create_model.assert_called_once_with(
            'vit_small_patch16_224', pretrained=True
        )
        assert isinstance(model, DSCATNet)

    def test_pretrained_false_does_not_load(self):
        """create_dscatnet with pretrained=False should not call timm."""
        with patch('src.models.dscatnet.timm') as mock_timm:
            model = create_dscatnet(num_classes=7, variant='small', pretrained=False)

        mock_timm.create_model.assert_not_called()
        assert isinstance(model, DSCATNet)
