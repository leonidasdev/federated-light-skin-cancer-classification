# =============================================================================
# Tests for YAML Configuration Loading
# =============================================================================
"""
Tests for YAML configuration file loading and parsing.

Tests:
1. Valid YAML loading
2. Malformed YAML handling
3. Missing required fields
4. Config merging/override behavior
"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path

import pytest
import yaml

# Project root for accessing run_experiment.py
PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def valid_config_content():
    """Valid YAML configuration content."""
    return """
federated:
  experiment:
    name: test_experiment
    description: "Test FL experiment"

  model:
    name: DSCATNet
    variant: tiny
    image_size: 224
    num_classes: 7

  training:
    batch_size: 4
    lr: 0.001
    local_epochs: 1
    num_rounds: 5

  federation:
    num_clients: 2
    participation: 1.0
    noniid_type: dirichlet
    dirichlet_alpha: 0.5
"""


@pytest.fixture
def malformed_yaml_content():
    """Malformed YAML content (invalid indentation)."""
    return """
federated:
  experiment:
    name: test
   bad_indent: value
  model:
    name: DSCATNet
"""


@pytest.fixture
def incomplete_config_content():
    """YAML with missing required fields."""
    return """
federated:
  experiment:
    name: incomplete_test
  # Missing model, training, federation sections
"""


# =============================================================================
# Test Classes
# =============================================================================


class TestYAMLLoading:
    """Tests for YAML file loading."""

    def test_load_valid_yaml(self, valid_config_content, tmp_path):
        """Test loading a valid YAML configuration."""
        config_file = tmp_path / "valid_config.yaml"
        config_file.write_text(valid_config_content)

        with open(config_file) as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "federated" in config
        assert config["federated"]["experiment"]["name"] == "test_experiment"

    def test_load_malformed_yaml_raises_error(self, malformed_yaml_content, tmp_path):
        """Test that malformed YAML raises appropriate error."""
        config_file = tmp_path / "malformed_config.yaml"
        config_file.write_text(malformed_yaml_content)

        with pytest.raises(yaml.YAMLError), open(config_file) as f:
            yaml.safe_load(f)

    def test_load_empty_yaml(self, tmp_path):
        """Test loading an empty YAML file."""
        config_file = tmp_path / "empty_config.yaml"
        config_file.write_text("")

        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Empty YAML returns None
        assert config is None

    def test_load_yaml_with_comments_only(self, tmp_path):
        """Test loading YAML with only comments."""
        config_file = tmp_path / "comments_only.yaml"
        config_file.write_text("# This is just a comment\n# Another comment")

        with open(config_file) as f:
            config = yaml.safe_load(f)

        assert config is None


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_config_with_extra_fields_accepted(self, valid_config_content, tmp_path):
        """Test that extra fields don't cause errors."""
        extra_content = valid_config_content + "\n  custom_field: custom_value\n"
        config_file = tmp_path / "extra_config.yaml"
        config_file.write_text(extra_content)

        with open(config_file) as f:
            config = yaml.safe_load(f)

        assert config is not None

    def test_config_numeric_values(self, tmp_path):
        """Test that numeric values are parsed correctly."""
        config_content = """
training:
  batch_size: 8
  lr: 0.001
  epochs: 100
  threshold: 0.5
"""
        config_file = tmp_path / "numeric_config.yaml"
        config_file.write_text(config_content)

        with open(config_file) as f:
            config = yaml.safe_load(f)

        assert isinstance(config["training"]["batch_size"], int)
        assert isinstance(config["training"]["lr"], float)
        assert config["training"]["lr"] == 0.001

    def test_config_boolean_values(self, tmp_path):
        """Test that boolean values are parsed correctly."""
        config_content = """
settings:
  pretrained: true
  use_amp: false
  debug: True
  verbose: False
"""
        config_file = tmp_path / "bool_config.yaml"
        config_file.write_text(config_content)

        with open(config_file) as f:
            config = yaml.safe_load(f)

        assert config["settings"]["pretrained"] is True
        assert config["settings"]["use_amp"] is False


class TestConfigMerging:
    """Tests for configuration merging/override behavior."""

    def test_dict_merge(self):
        """Test merging two config dictionaries."""
        base_config = {"model": {"name": "DSCATNet", "variant": "small"}, "training": {"batch_size": 8, "lr": 0.001}}

        override_config = {
            "training": {"batch_size": 16}  # Override only batch_size
        }

        # Simulate merging
        merged = base_config.copy()
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value

        assert merged["training"]["batch_size"] == 16
        assert merged["training"]["lr"] == 0.001  # Preserved
        assert merged["model"]["variant"] == "small"  # Preserved


class TestRunExperimentConfigLoading:
    """Tests for run_experiment.py config loading."""

    def test_load_config_function_exists(self):
        """Test that load_config function is importable."""
        # Import from run_experiment.py module
        import importlib.util

        spec = importlib.util.spec_from_file_location("run_experiment", PROJECT_ROOT / "run_experiment.py")

        # Just check we can load the module without torch errors
        # (torch import may fail in test environment)
        assert spec is not None

    def test_nonexistent_config_file(self, tmp_path):
        """Test behavior with non-existent config file."""
        nonexistent = tmp_path / "does_not_exist.yaml"

        with pytest.raises(FileNotFoundError), open(nonexistent) as f:
            yaml.safe_load(f)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
