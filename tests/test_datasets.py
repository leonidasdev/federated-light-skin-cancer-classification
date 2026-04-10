# =============================================================================
# Tests for Dataset Registry and Utilities
# =============================================================================
"""
Tests for DATASET_REGISTRY, dataset loading, utility functions,
and actual dataset classes (HAM10000, ISIC2018, ISIC2019, ISIC2020, PAD-UFES-20).
"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.datasets import (
    CLASS_NAMES_7,
    CLASS_NAMES_8,
    CLASS_NAMES_BINARY,
    DATASET_REGISTRY,
    ISIC2019_CLASSES,
    UNIFIED_CLASSES_7,
    UNIFIED_CLASSES_BINARY,
    BaseDermoscopyDataset,
    DatasetSubset,
    HAM10000Dataset,
    ISIC2018Dataset,
    ISIC2019Dataset,
    ISIC2020Dataset,
    PADUFES20Dataset,
    get_available_datasets,
    get_dataset_paths,
    load_dataset,
    normalize_dataset_name,
)

# =============================================================================
# Helpers — create tiny images + CSVs for integration tests
# =============================================================================


def _create_tiny_image(path: Path, size: tuple[int, int] = (28, 28)) -> None:
    """Create a tiny RGB JPEG or PNG image at *path*."""
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _build_ham10000(root: Path, n: int = 4) -> Path:
    """Create a minimal HAM10000 directory with CSV + images."""
    ds = root / "HAM10000"
    ds.mkdir(parents=True, exist_ok=True)
    labels = ["akiec", "bcc", "bkl", "mel"]
    csv_path = ds / "HAM10000_metadata.csv"
    part1 = ds / "HAM10000_images_part_1"
    part1.mkdir(exist_ok=True)
    lines = ["image_id,dx"]
    for i in range(n):
        img_id = f"ISIC_{i:07d}"
        lines.append(f"{img_id},{labels[i % len(labels)]}")
        _create_tiny_image(part1 / f"{img_id}.jpg")
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return ds


def _build_isic2018(root: Path, n: int = 4) -> Path:
    """Create a minimal ISIC2018 directory with one-hot CSV + images."""
    ds = root / "ISIC2018"
    ds.mkdir(parents=True, exist_ok=True)
    img_dir = ds / "ISIC2018_Task3_Training_Input"
    img_dir.mkdir(exist_ok=True)
    cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    csv_path = ds / "ISIC2018_Task3_Training_GroundTruth.csv"
    lines = ["image," + ",".join(cols)]
    for i in range(n):
        img_id = f"ISIC_18_{i:04d}"
        one_hot = ["0.0"] * len(cols)
        one_hot[i % len(cols)] = "1.0"
        lines.append(f"{img_id}," + ",".join(one_hot))
        _create_tiny_image(img_dir / f"{img_id}.jpg")
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return ds


def _build_isic2019(root: Path, n: int = 4) -> Path:
    """Create a minimal ISIC2019 directory with one-hot CSV + images."""
    ds = root / "ISIC2019"
    ds.mkdir(parents=True, exist_ok=True)
    img_dir = ds / "ISIC_2019_Training_Input"
    img_dir.mkdir(exist_ok=True)
    cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    csv_path = ds / "ISIC_2019_Training_GroundTruth.csv"
    lines = ["image," + ",".join(cols)]
    for i in range(n):
        img_id = f"ISIC_19_{i:04d}"
        one_hot = ["0.0"] * len(cols)
        one_hot[i % len(cols)] = "1.0"
        lines.append(f"{img_id}," + ",".join(one_hot))
        _create_tiny_image(img_dir / f"{img_id}.jpg")
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return ds


def _build_isic2020(root: Path, n: int = 4) -> Path:
    """Create a minimal ISIC2020 directory with binary labels."""
    ds = root / "ISIC2020"
    ds.mkdir(parents=True, exist_ok=True)
    img_dir = ds / "ISIC_2020_Training_JPEG" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ds / "ISIC_2020_Training_GroundTruth.csv"
    diagnoses = ["nevus", "melanoma", "seborrheic keratosis", "nevus"]
    bm = ["benign", "malignant", "benign", "benign"]
    lines = ["image_name,target,diagnosis,benign_malignant"]
    for i in range(n):
        img_id = f"ISIC_20_{i:04d}"
        lines.append(f"{img_id},{i % 2},{diagnoses[i % len(diagnoses)]},{bm[i % len(bm)]}")
        _create_tiny_image(img_dir / f"{img_id}.jpg")
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return ds


def _build_padufes20(root: Path, n: int = 4) -> Path:
    """Create a minimal PAD-UFES-20 directory with CSV + images."""
    ds = root / "PAD-UFES-20"
    ds.mkdir(parents=True, exist_ok=True)
    part1 = ds / "imgs_part_1"
    part1.mkdir(exist_ok=True)
    csv_path = ds / "metadata.csv"
    labels = ["BCC", "MEL", "ACK", "NEV"]
    lines = ["img_id,diagnostic"]
    for i in range(n):
        fname = f"PAT_{i:04d}.png"
        lines.append(f"{fname},{labels[i % len(labels)]}")
        _create_tiny_image(part1 / fname)
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return ds


# =============================================================================
# Tests for DATASET_REGISTRY
# =============================================================================


class TestDatasetRegistryDict:
    """Tests for DATASET_REGISTRY structure."""

    def test_registry_not_empty(self):
        """Registry should contain at least 5 datasets."""
        assert len(DATASET_REGISTRY) >= 5

    def test_registry_contains_expected_datasets(self):
        """Registry should contain all expected dataset names."""
        expected = ["HAM10000", "ISIC2018", "ISIC2019", "ISIC2020", "PAD-UFES-20"]
        for name in expected:
            assert name in DATASET_REGISTRY, f"Missing {name} in registry"

    def test_registry_entry_has_required_fields(self):
        """Each registry entry should have required fields."""
        for name, config in DATASET_REGISTRY.items():
            assert hasattr(config, "dataset_class"), f"{name} missing dataset_class"
            assert hasattr(config, "csv_filename"), f"{name} missing csv_filename"
            # dataset_class should be callable
            assert callable(config.dataset_class), f"{name} dataset_class not callable"

    def test_registry_has_image_subdir_field(self):
        """Each registry entry should have image_subdir field (can be None)."""
        for name, config in DATASET_REGISTRY.items():
            assert hasattr(config, "image_subdir"), f"{name} missing image_subdir"


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
            _csv_path, image_root = get_dataset_paths(name, tmp_path)
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
        expected = ["HAM10000", "ISIC2018", "ISIC2019", "ISIC2020", "PAD-UFES-20"]
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
        result = load_dataset("HAM10000", tmp_path, classification_mode="binary")
        assert result is None  # Files don't exist

    def test_accepts_filter_unknown(self, tmp_path):
        """Should accept filter_unknown parameter."""
        result = load_dataset("HAM10000", tmp_path, filter_unknown=False)
        assert result is None  # Files don't exist

    def test_load_ham10000_with_real_files(self, tmp_path):
        """load_dataset should return HAM10000Dataset when files exist."""
        _build_ham10000(tmp_path, n=4)
        ds = load_dataset("HAM10000", tmp_path)
        assert ds is not None
        assert isinstance(ds, HAM10000Dataset)
        assert len(ds) == 4


# =============================================================================
# Tests for BaseDermoscopyDataset._map_label
# =============================================================================


class TestMapLabel:
    """Tests for label mapping in different classification modes."""

    def test_multiclass_maps_ham_labels(self, tmp_path):
        """Multiclass mode should map HAM10000 lowercase labels correctly."""
        ds_path = _build_ham10000(tmp_path, n=1)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
            classification_mode="multiclass",
        )
        assert ds._map_label("mel") == 4
        assert ds._map_label("nv") == 5
        assert ds._map_label("bcc") == 1

    def test_binary_maps_malignant(self, tmp_path):
        """Binary mode should map malignant labels to 1."""
        ds_path = _build_ham10000(tmp_path, n=1)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
            classification_mode="binary",
        )
        assert ds._map_label("mel") == 1
        assert ds._map_label("bcc") == 1
        assert ds._map_label("nv") == 0

    def test_multiclass_8_maps_scc(self, tmp_path):
        """Multiclass-8 mode should handle SCC from ISIC2019."""
        ds_path = _build_ham10000(tmp_path, n=1)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
            classification_mode="multiclass_8",
        )
        assert ds._map_label("SCC") == ISIC2019_CLASSES["SCC"]
        assert ds._map_label("UNK") == -1

    def test_unknown_label_returns_minus1(self, tmp_path):
        """Unknown labels should map to -1 in all modes."""
        ds_path = _build_ham10000(tmp_path, n=1)
        for mode in ("multiclass", "binary", "multiclass_8"):
            ds = HAM10000Dataset(
                root_dir=str(ds_path),
                csv_path=str(ds_path / "HAM10000_metadata.csv"),
                classification_mode=mode,
            )
            assert ds._map_label("nonsense_label") == -1


# =============================================================================
# Tests for HAM10000Dataset
# =============================================================================


class TestHAM10000Dataset:
    """Tests for HAM10000Dataset class."""

    def test_loads_correct_num_samples(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=6)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
        )
        assert len(ds) == 6

    def test_getitem_returns_tensor_and_int(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=2)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
        )
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.ndim == 3 and img.shape[0] == 3  # CHW
        assert isinstance(label, int)

    def test_class_distribution(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=4)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
        )
        dist = ds.get_class_distribution()
        assert isinstance(dist, dict)
        assert sum(dist.values()) == 4

    def test_class_weights(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=4)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
        )
        w = ds.get_class_weights()
        assert isinstance(w, torch.Tensor)
        assert w.shape == (7,)

    def test_binary_mode(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=4)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
            classification_mode="binary",
        )
        assert ds.num_classes == 2
        assert ds.class_names == CLASS_NAMES_BINARY
        for _, label in ds:
            assert label in (0, 1)

    def test_multiclass_8_mode(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=4)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
            classification_mode="multiclass_8",
        )
        assert ds.num_classes == 8
        assert ds.class_names == CLASS_NAMES_8

    def test_with_transform(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=2)
        transform = MagicMock(side_effect=lambda image: {"image": image})
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
            transform=transform,
        )
        _ = ds[0]
        transform.assert_called_once()

    def test_with_target_transform(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=2)
        target_transform = MagicMock(side_effect=lambda x: x + 100)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
            target_transform=target_transform,
        )
        _, label = ds[0]
        target_transform.assert_called_once()
        assert label >= 100


# =============================================================================
# Tests for ISIC2018Dataset
# =============================================================================


class TestISIC2018Dataset:
    """Tests for ISIC2018Dataset class."""

    def test_loads_one_hot_labels(self, tmp_path):
        ds_path = _build_isic2018(tmp_path, n=7)
        ds = ISIC2018Dataset(
            root_dir=str(ds_path / "ISIC2018_Task3_Training_Input"),
            csv_path=str(ds_path / "ISIC2018_Task3_Training_GroundTruth.csv"),
        )
        assert len(ds) == 7

    def test_getitem(self, tmp_path):
        ds_path = _build_isic2018(tmp_path, n=3)
        ds = ISIC2018Dataset(
            root_dir=str(ds_path / "ISIC2018_Task3_Training_Input"),
            csv_path=str(ds_path / "ISIC2018_Task3_Training_GroundTruth.csv"),
        )
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert 0 <= label <= 6


# =============================================================================
# Tests for ISIC2019Dataset
# =============================================================================


class TestISIC2019Dataset:
    """Tests for ISIC2019Dataset class."""

    def test_loads_with_8_class_mode(self, tmp_path):
        ds_path = _build_isic2019(tmp_path, n=9)
        ds = ISIC2019Dataset(
            root_dir=str(ds_path / "ISIC_2019_Training_Input"),
            csv_path=str(ds_path / "ISIC_2019_Training_GroundTruth.csv"),
            classification_mode="multiclass_8",
            filter_unknown=False,
        )
        assert ds.num_classes == 8
        assert len(ds) == 9

    def test_filters_unknown_by_default(self, tmp_path):
        ds_path = _build_isic2019(tmp_path, n=9)
        ds = ISIC2019Dataset(
            root_dir=str(ds_path / "ISIC_2019_Training_Input"),
            csv_path=str(ds_path / "ISIC_2019_Training_GroundTruth.csv"),
            classification_mode="multiclass_8",
            filter_unknown=True,
        )
        # UNK sample at index 8 should be filtered
        assert len(ds) == 8


# =============================================================================
# Tests for ISIC2020Dataset
# =============================================================================


class TestISIC2020Dataset:
    """Tests for ISIC2020Dataset class."""

    def test_binary_mode(self, tmp_path):
        ds_path = _build_isic2020(tmp_path, n=4)
        ds = ISIC2020Dataset(
            root_dir=str(ds_path / "ISIC_2020_Training_JPEG" / "train"),
            csv_path=str(ds_path / "ISIC_2020_Training_GroundTruth.csv"),
            classification_mode="binary",
        )
        assert ds.num_classes == 2
        for _, label in ds:
            assert label in (0, 1)

    def test_multiclass_mode_uses_diagnosis(self, tmp_path):
        ds_path = _build_isic2020(tmp_path, n=4)
        ds = ISIC2020Dataset(
            root_dir=str(ds_path / "ISIC_2020_Training_JPEG" / "train"),
            csv_path=str(ds_path / "ISIC_2020_Training_GroundTruth.csv"),
            classification_mode="multiclass",
        )
        assert ds.num_classes == 7
        assert len(ds) == 4


# =============================================================================
# Tests for PADUFES20Dataset
# =============================================================================


class TestPADUFES20Dataset:
    """Tests for PADUFES20Dataset class."""

    def test_loads_from_parts(self, tmp_path):
        ds_path = _build_padufes20(tmp_path, n=4)
        ds = PADUFES20Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "metadata.csv"),
        )
        assert len(ds) == 4

    def test_getitem_returns_correct_types(self, tmp_path):
        ds_path = _build_padufes20(tmp_path, n=2)
        ds = PADUFES20Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "metadata.csv"),
        )
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)

    def test_binary_mode(self, tmp_path):
        ds_path = _build_padufes20(tmp_path, n=4)
        ds = PADUFES20Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "metadata.csv"),
            classification_mode="binary",
        )
        assert ds.num_classes == 2


# =============================================================================
# Tests for DatasetSubset
# =============================================================================


class TestDatasetSubset:
    """Tests for DatasetSubset class."""

    def test_len(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=6)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
        )
        subset = DatasetSubset(ds, [0, 2, 4])
        assert len(subset) == 3

    def test_getitem(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=4)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
        )
        subset = DatasetSubset(ds, [1, 3])
        img, _label = subset[0]
        assert isinstance(img, torch.Tensor)
        assert img.ndim == 3

    def test_inherits_num_classes(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=2)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
        )
        subset = DatasetSubset(ds, [0, 1])
        assert subset.num_classes == ds.num_classes
        assert subset.class_names == ds.class_names

    def test_class_distribution(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=4)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
        )
        subset = DatasetSubset(ds, [0, 1])
        dist = subset.get_class_distribution()
        assert sum(dist.values()) == 2

    def test_with_separate_transform(self, tmp_path):
        ds_path = _build_ham10000(tmp_path, n=2)
        ds = HAM10000Dataset(
            root_dir=str(ds_path),
            csv_path=str(ds_path / "HAM10000_metadata.csv"),
        )
        transform = MagicMock(side_effect=lambda image: {"image": image})
        subset = DatasetSubset(ds, [0], transform=transform)
        _ = subset[0]
        transform.assert_called_once()


# =============================================================================
# Tests for load_dataset integration
# =============================================================================


class TestLoadDatasetIntegration:
    """Integration tests for load_dataset with real file structures."""

    def test_load_isic2018(self, tmp_path):
        _build_isic2018(tmp_path, n=3)
        ds = load_dataset("ISIC2018", tmp_path)
        assert ds is not None
        assert len(ds) == 3

    def test_load_padufes20(self, tmp_path):
        _build_padufes20(tmp_path, n=3)
        ds = load_dataset("PAD-UFES-20", tmp_path)
        assert ds is not None
        assert len(ds) == 3

    def test_load_with_binary_mode(self, tmp_path):
        _build_ham10000(tmp_path, n=4)
        ds = load_dataset("HAM10000", tmp_path, classification_mode="binary")
        assert ds is not None
        assert ds.num_classes == 2


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
