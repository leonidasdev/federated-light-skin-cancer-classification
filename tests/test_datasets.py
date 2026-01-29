# =============================================================================
# Tests for Dataset Registry and Utilities
# =============================================================================
"""
Tests for DatasetRegistry, dataset loading, and utility functions.

Tests:
1. DATASET_REGISTRY structure and content
2. normalize_dataset_name() with various input formats
3. get_dataset_paths() path resolution
4. get_available_datasets() listing
5. load_dataset() with mocked paths
"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.datasets import (
    DATASET_REGISTRY,
    normalize_dataset_name,
    get_dataset_paths,
    get_available_datasets,
    load_dataset,
)

# =============================================================================
# Tests for DATASET_REGISTRY
# =============================================================================


class TestDatasetRegistry:
    """Tests for DATASET_REGISTRY structure."""

    def test_registry_not_empty(self):
        """Registry should contain at least 5 datasets."""
        assert len(DATASET_REGISTRY) >= 5

    def test_registry_contains_expected_datasets(self):
        """Registry should contain all expected dataset names."""
        expected = ['HAM10000', 'ISIC2018', 'ISIC2019', 'ISIC2020', 'PAD-UFES-20']
        for name in expected:
            assert name in DATASET_REGISTRY, f"Missing {name} in registry"

    def test_registry_entry_has_required_fields(self):
        """Each registry entry should have required fields."""
        for name, config in DATASET_REGISTRY.items():
            assert hasattr(config, 'dataset_class'), f"{name} missing dataset_class"
            assert hasattr(config, 'csv_filename'), f"{name} missing csv_filename"
            # dataset_class should be callable
            assert callable(config.dataset_class), f"{name} dataset_class not callable"

    def test_registry_has_image_subdir_field(self):
        """Each registry entry should have image_subdir field (can be None)."""
        for name, config in DATASET_REGISTRY.items():
            assert hasattr(config, 'image_subdir'), f"{name} missing image_subdir"


# =============================================================================
# Tests for normalize_dataset_name
# =============================================================================


class TestNormalizeDatasetName:
    """Tests for normalize_dataset_name function."""

    def test_canonical_names_unchanged(self):
        """Canonical names should remain unchanged."""
        assert normalize_dataset_name("HAM10000") == "HAM10000"
        assert normalize_dataset_name("ISIC2018") == "ISIC2018"
        assert normalize_dataset_name("ISIC2019") == "ISIC2019"
        assert normalize_dataset_name("ISIC2020") == "ISIC2020"
        assert normalize_dataset_name("PAD-UFES-20") == "PAD-UFES-20"

    def test_lowercase_normalization(self):
        """Lowercase names should be normalized."""
        assert normalize_dataset_name("ham10000") == "HAM10000"
        assert normalize_dataset_name("isic2018") == "ISIC2018"
        assert normalize_dataset_name("pad-ufes-20") == "PAD-UFES-20"

    def test_mixed_case_normalization(self):
        """Mixed case names should be normalized."""
        assert normalize_dataset_name("Ham10000") == "HAM10000"
        assert normalize_dataset_name("Isic2018") == "ISIC2018"

    def test_underscore_normalization(self):
        """Names with underscores should be normalized."""
        assert normalize_dataset_name("PAD_UFES_20") == "PAD-UFES-20"
        assert normalize_dataset_name("pad_ufes_20") == "PAD-UFES-20"

    def test_no_separator_normalization(self):
        """Names without separators should be normalized."""
        assert normalize_dataset_name("PADUFES20") == "PAD-UFES-20"
        assert normalize_dataset_name("padufes20") == "PAD-UFES-20"

    def test_unknown_name_unchanged(self):
        """Unknown dataset names should be returned unchanged."""
        assert normalize_dataset_name("UnknownDataset") == "UnknownDataset"
        assert normalize_dataset_name("MY_CUSTOM_DATA") == "MY_CUSTOM_DATA"


# =============================================================================
# Tests for get_dataset_paths
# =============================================================================


class TestGetDatasetPaths:
    """Tests for get_dataset_paths function."""

    def test_unknown_dataset_returns_none(self):
        """Unknown dataset should return (None, None)."""
        csv_path, image_root = get_dataset_paths("UnknownDataset", "/data")
        assert csv_path is None
        assert image_root is None

    def test_returns_path_objects(self, tmp_path):
        """Should return Path objects for valid datasets."""
        # Create minimal directory structure
        dataset_dir = tmp_path / "HAM10000"
        dataset_dir.mkdir()
        csv_file = dataset_dir / "HAM10000_metadata.csv"
        csv_file.touch()

        csv_path, image_root = get_dataset_paths("HAM10000", tmp_path)

        assert isinstance(csv_path, Path)
        assert isinstance(image_root, Path)

    def test_normalized_name_works(self, tmp_path):
        """Should work with various name formats."""
        dataset_dir = tmp_path / "HAM10000"
        dataset_dir.mkdir()

        # All these should resolve to the same dataset
        for name in ["HAM10000", "ham10000", "Ham10000"]:
            csv_path, image_root = get_dataset_paths(name, tmp_path)
            assert image_root is not None


# =============================================================================
# Tests for get_available_datasets
# =============================================================================


class TestGetAvailableDatasets:
    """Tests for get_available_datasets function."""

    def test_returns_list(self):
        """Should return a list."""
        result = get_available_datasets()
        assert isinstance(result, list)

    def test_returns_at_least_5_datasets(self):
        """Should return at least 5 dataset names."""
        result = get_available_datasets()
        assert len(result) >= 5

    def test_contains_expected_names(self):
        """Should contain expected dataset names."""
        result = get_available_datasets()
        expected = ['HAM10000', 'ISIC2018', 'ISIC2019', 'ISIC2020', 'PAD-UFES-20']
        for name in expected:
            assert name in result


# =============================================================================
# Tests for load_dataset
# =============================================================================


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_unknown_dataset_returns_none(self, tmp_path):
        """Unknown dataset should return None."""
        result = load_dataset("UnknownDataset", tmp_path)
        assert result is None

    def test_missing_csv_returns_none(self, tmp_path):
        """Missing CSV file should return None."""
        # Create directory but no CSV
        dataset_dir = tmp_path / "HAM10000"
        dataset_dir.mkdir()

        result = load_dataset("HAM10000", tmp_path)
        assert result is None

    def test_accepts_transform_parameter(self, tmp_path):
        """Should accept transform parameter without error."""
        # Even if loading fails, the function signature should work
        mock_transform = MagicMock()
        result = load_dataset("HAM10000", tmp_path, transform=mock_transform)
        # Result is None because files don't exist, but no error
        assert result is None

    def test_accepts_classification_mode(self, tmp_path):
        """Should accept classification_mode parameter."""
        result = load_dataset("HAM10000", tmp_path, classification_mode='binary')
        assert result is None  # Files don't exist

    def test_accepts_filter_unknown(self, tmp_path):
        """Should accept filter_unknown parameter."""
        result = load_dataset("HAM10000", tmp_path, filter_unknown=False)
        assert result is None  # Files don't exist


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
