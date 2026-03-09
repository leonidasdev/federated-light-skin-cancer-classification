# =============================================================================
# Tests for Configuration Schema Validation (Pydantic models)
# =============================================================================
"""
Tests for src.utils.config_schema — Pydantic model validators, detect_config_type,
validate_config, and validate_config_dict functions.
"""

import pytest
import yaml
from pydantic import ValidationError

from src.utils.config_schema import (
    ConfigType,
    DataConfig,
    ExperimentConfig,
    FederatedSettings,
    FLConfig,
    FLStrategy,
    FusionMethod,
    ModelConfig,
    ModelSettings,
    detect_config_type,
    validate_config,
    validate_config_dict,
)


# =============================================================================
# DataConfig Validator Tests
# =============================================================================


class TestDataConfigValidation:
    """Tests for DataConfig.validate_splits validator."""

    def test_valid_splits(self):
        """Normal splits should pass validation."""
        dc = DataConfig(val_split=0.15, test_split=0.15)
        assert dc.val_split == 0.15
        assert dc.test_split == 0.15

    def test_splits_sum_to_one_raises(self):
        """val_split + test_split == 1.0 should raise ValueError."""
        with pytest.raises(ValidationError, match="must be less than 1.0"):
            DataConfig(val_split=0.5, test_split=0.5)

    def test_splits_exceed_one_raises(self):
        """val_split + test_split > 1.0 should raise (Field constraint caps at 0.5 each)."""
        with pytest.raises(ValidationError):
            DataConfig(val_split=0.5, test_split=0.5)

    def test_zero_splits_valid(self):
        """Both splits = 0 means 100% train, should be valid."""
        dc = DataConfig(val_split=0.0, test_split=0.0)
        assert dc.val_split == 0.0


# =============================================================================
# FederatedSettings Validator Tests
# =============================================================================


class TestFederatedSettingsValidation:
    """Tests for FederatedSettings.validate_client_counts validator."""

    def test_valid_client_counts(self):
        """Valid client counts should pass."""
        fs = FederatedSettings(
            num_clients=4,
            min_fit_clients=2,
            min_evaluate_clients=2,
        )
        assert fs.num_clients == 4

    def test_min_fit_clients_exceeds_num_clients(self):
        """min_fit_clients > num_clients should raise ValueError."""
        with pytest.raises(ValidationError, match="min_fit_clients.*cannot exceed"):
            FederatedSettings(
                num_clients=2,
                min_fit_clients=5,
                min_evaluate_clients=1,
            )

    def test_min_evaluate_clients_exceeds_num_clients(self):
        """min_evaluate_clients > num_clients should raise ValueError."""
        with pytest.raises(ValidationError, match="min_evaluate_clients.*cannot exceed"):
            FederatedSettings(
                num_clients=2,
                min_fit_clients=1,
                min_evaluate_clients=5,
            )

    def test_both_min_clients_equal_num_clients(self):
        """min_fit_clients == min_evaluate_clients == num_clients is valid."""
        fs = FederatedSettings(
            num_clients=3,
            min_fit_clients=3,
            min_evaluate_clients=3,
        )
        assert fs.min_fit_clients == 3


# =============================================================================
# ModelSettings Validator Tests
# =============================================================================


class TestModelSettingsValidation:
    """Tests for ModelSettings.validate_patch_sizes validator."""

    def test_valid_model_settings(self):
        """Default settings should pass validation."""
        ms = ModelSettings()
        assert ms.img_size == 224

    def test_img_size_not_divisible_by_fine_patch(self):
        """img_size not divisible by fine_patch_size should raise."""
        with pytest.raises(ValidationError, match="divisible by fine_patch_size"):
            ModelSettings(img_size=100, fine_patch_size=8, coarse_patch_size=10)

    def test_img_size_not_divisible_by_coarse_patch(self):
        """img_size not divisible by coarse_patch_size should raise."""
        with pytest.raises(ValidationError, match="divisible by coarse_patch_size"):
            ModelSettings(img_size=224, fine_patch_size=8, coarse_patch_size=15)

    def test_embed_dim_not_divisible_by_num_heads(self):
        """embed_dim not divisible by num_heads should raise."""
        with pytest.raises(ValidationError, match="divisible by num_heads"):
            ModelSettings(embed_dim=384, num_heads=5)

    def test_valid_custom_patch_sizes(self):
        """Custom but valid patch sizes should pass."""
        ms = ModelSettings(img_size=128, fine_patch_size=8, coarse_patch_size=16)
        assert ms.img_size == 128


# =============================================================================
# detect_config_type Tests
# =============================================================================


class TestDetectConfigType:
    """Tests for detect_config_type function."""

    def test_detect_experiment(self):
        """Config with 'experiment' key should return EXPERIMENT."""
        config = {"experiment": {"name": "test"}}
        assert detect_config_type(config) == ConfigType.EXPERIMENT

    def test_detect_federated_experiments(self):
        """Config with 'federated_experiments' key should return EXPERIMENT."""
        config = {"federated_experiments": []}
        assert detect_config_type(config) == ConfigType.EXPERIMENT

    def test_detect_federated(self):
        """Config with federated.num_clients should return FEDERATED."""
        config = {"federated": {"num_clients": 4}}
        assert detect_config_type(config) == ConfigType.FEDERATED

    def test_detect_model(self):
        """Config with model.embed_dim should return MODEL."""
        config = {"model": {"embed_dim": 384}}
        assert detect_config_type(config) == ConfigType.MODEL

    def test_unknown_raises(self):
        """Config without matching keys should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            detect_config_type({"random_key": "value"})

    def test_empty_dict_raises(self):
        """Empty dict should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            detect_config_type({})


# =============================================================================
# validate_config Tests
# =============================================================================


class TestValidateConfig:
    """Tests for validate_config function (YAML file loading + validation)."""

    def test_validate_experiment_config(self, tmp_path):
        """Should load and validate an experiment config YAML."""
        cfg = {"experiment": {"name": "test", "seed": 42}}
        f = tmp_path / "exp.yaml"
        f.write_text(yaml.dump(cfg))
        result = validate_config(str(f), ConfigType.EXPERIMENT)
        assert isinstance(result, ExperimentConfig)
        assert result.experiment.name == "test"

    def test_validate_federated_config(self, tmp_path):
        """Should load and validate a federated config YAML."""
        cfg = {"federated": {"num_clients": 4}}
        f = tmp_path / "fl.yaml"
        f.write_text(yaml.dump(cfg))
        result = validate_config(str(f), ConfigType.FEDERATED)
        assert isinstance(result, FLConfig)
        assert result.federated.num_clients == 4

    def test_validate_model_config(self, tmp_path):
        """Should load and validate a model config YAML."""
        cfg = {"model": {"name": "DSCATNet", "embed_dim": 384, "num_heads": 6}}
        f = tmp_path / "model.yaml"
        f.write_text(yaml.dump(cfg))
        result = validate_config(str(f), ConfigType.MODEL)
        assert isinstance(result, ModelConfig)

    def test_auto_detect_type(self, tmp_path):
        """Should auto-detect config type if not specified."""
        cfg = {"experiment": {"name": "auto-detect"}}
        f = tmp_path / "auto.yaml"
        f.write_text(yaml.dump(cfg))
        result = validate_config(str(f))
        assert isinstance(result, ExperimentConfig)

    def test_file_not_found_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_config(str(tmp_path / "missing.yaml"))

    def test_empty_file_raises(self, tmp_path):
        """Should raise ValueError for empty YAML."""
        f = tmp_path / "empty.yaml"
        f.write_text("")
        with pytest.raises(ValueError, match="empty"):
            validate_config(str(f))

    def test_invalid_config_raises_validation_error(self, tmp_path):
        """Should raise ValidationError for invalid config."""
        cfg = {"federated": {"num_clients": -1}}
        f = tmp_path / "invalid.yaml"
        f.write_text(yaml.dump(cfg))
        with pytest.raises(ValidationError):
            validate_config(str(f), ConfigType.FEDERATED)


# =============================================================================
# validate_config_dict Tests
# =============================================================================


class TestValidateConfigDict:
    """Tests for validate_config_dict function."""

    def test_validate_experiment_dict(self):
        """Should validate an experiment config dict."""
        cfg = {"experiment": {"name": "test"}}
        result = validate_config_dict(cfg, ConfigType.EXPERIMENT)
        assert isinstance(result, ExperimentConfig)

    def test_validate_federated_dict(self):
        """Should validate a federated config dict."""
        cfg = {"federated": {"num_clients": 4}}
        result = validate_config_dict(cfg, ConfigType.FEDERATED)
        assert isinstance(result, FLConfig)

    def test_validate_model_dict(self):
        """Should validate a model config dict."""
        cfg = {"model": {"name": "DSCATNet", "embed_dim": 384, "num_heads": 6}}
        result = validate_config_dict(cfg, ConfigType.MODEL)
        assert isinstance(result, ModelConfig)

    def test_auto_detect_from_dict(self):
        """Should auto-detect config type from dict."""
        cfg = {"model": {"embed_dim": 384, "num_heads": 6}}
        result = validate_config_dict(cfg)
        assert isinstance(result, ModelConfig)

    def test_invalid_dict_raises(self):
        """Should raise ValidationError for invalid values."""
        cfg = {"model": {"embed_dim": 384, "num_heads": 5}}  # 384 % 5 != 0
        with pytest.raises(ValidationError):
            validate_config_dict(cfg, ConfigType.MODEL)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
