# =============================================================================
# Tests for Dataset Download Functions
# =============================================================================
"""
Tests for download functionality including:
1. DATASET_INFO structure validation
2. Directory structure creation
3. Kaggle availability check
4. Mendeley download function
5. ISIC API client
6. Unified download function
7. Verification functions
"""

# =============================================================================
# Imports
# =============================================================================

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Import download module directly to avoid torch dependency
# This works because download.py doesn't import torch
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "download",
    Path(__file__).parent.parent / "src" / "data" / "download.py"
)
download_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_module)

# Get exports from module
DATASET_INFO = download_module.DATASET_INFO
get_data_root = download_module.get_data_root
create_directory_structure = download_module.create_directory_structure
check_kaggle_available = download_module.check_kaggle_available
verify_dataset = download_module.verify_dataset
verify_all_datasets = download_module.verify_all_datasets
download_dataset = download_module.download_dataset
ISICArchiveClient = download_module.ISICArchiveClient


# =============================================================================
# Tests for DATASET_INFO Structure
# =============================================================================


class TestDatasetInfo:
    """Tests for DATASET_INFO dictionary."""

    def test_dataset_info_not_empty(self):
        """DATASET_INFO should contain datasets."""
        assert len(DATASET_INFO) >= 5

    def test_contains_all_expected_datasets(self):
        """Should contain all 5 expected datasets."""
        expected = ['HAM10000', 'ISIC2018', 'ISIC2019', 'ISIC2020', 'PAD-UFES-20']
        for name in expected:
            assert name in DATASET_INFO, f"Missing {name} in DATASET_INFO"

    def test_each_entry_has_required_fields(self):
        """Each entry should have required fields."""
        required_fields = ['description', 'source', 'download_url', 'classes',
                          'approx_images', 'expected_files', 'client_id']
        for name, info in DATASET_INFO.items():
            for field in required_fields:
                assert field in info, f"{name} missing field: {field}"

    def test_ham10000_has_kaggle_info(self):
        """HAM10000 should have Kaggle-specific info."""
        assert DATASET_INFO['HAM10000']['source'] == 'kaggle'
        assert 'kaggle_dataset' in DATASET_INFO['HAM10000']

    def test_padufes20_has_mendeley_info(self):
        """PAD-UFES-20 should have Mendeley-specific info."""
        assert DATASET_INFO['PAD-UFES-20']['source'] == 'mendeley'
        assert 'mendeley_url' in DATASET_INFO['PAD-UFES-20']

    def test_isic_datasets_have_isic_source(self):
        """ISIC datasets should have ISIC source."""
        for name in ['ISIC2018', 'ISIC2019', 'ISIC2020']:
            assert DATASET_INFO[name]['source'] == 'isic'
            assert 'isic_dataset' in DATASET_INFO[name]

    def test_client_ids_are_unique(self):
        """Each dataset should have a unique client_id."""
        client_ids = [info['client_id'] for info in DATASET_INFO.values()]
        assert len(client_ids) == len(set(client_ids))

    def test_approx_images_are_positive(self):
        """approx_images should be positive integers."""
        for name, info in DATASET_INFO.items():
            assert info['approx_images'] > 0
            assert isinstance(info['approx_images'], int)


# =============================================================================
# Tests for Directory Structure Creation
# =============================================================================


class TestDirectoryStructure:
    """Tests for create_directory_structure function."""

    def test_creates_all_dataset_directories(self, tmp_path):
        """Should create directories for all datasets."""
        paths = create_directory_structure(tmp_path)

        assert len(paths) == len(DATASET_INFO)
        for name in DATASET_INFO:
            assert name in paths
            assert paths[name].exists()

    def test_creates_raw_and_processed_dirs(self, tmp_path):
        """Should create raw and processed subdirectories."""
        create_directory_structure(tmp_path)

        assert (tmp_path / "raw").exists()
        assert (tmp_path / "processed").exists()

    def test_idempotent(self, tmp_path):
        """Calling twice should not cause errors."""
        create_directory_structure(tmp_path)
        create_directory_structure(tmp_path)  # Should not raise


# =============================================================================
# Tests for Kaggle Availability Check
# =============================================================================


class TestKaggleAvailability:
    """Tests for check_kaggle_available function."""

    @patch('shutil.which')
    def test_returns_false_when_kaggle_not_installed(self, mock_which):
        """Should return False if kaggle CLI not found."""
        mock_which.return_value = None
        assert check_kaggle_available() is False

    @patch('shutil.which')
    @patch('pathlib.Path.exists')
    def test_returns_false_when_no_credentials(self, mock_exists, mock_which):
        """Should return False if credentials file not found."""
        mock_which.return_value = "/usr/bin/kaggle"
        mock_exists.return_value = False
        assert check_kaggle_available() is False

    @patch('shutil.which')
    def test_returns_true_when_fully_configured(self, mock_which):
        """Should return True when kaggle is installed and configured."""
        mock_which.return_value = "/usr/bin/kaggle"

        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            assert check_kaggle_available() is True


# =============================================================================
# Tests for Verification Functions
# =============================================================================


class TestVerifyDataset:
    """Tests for verify_dataset function."""

    def test_unknown_dataset_returns_invalid(self, tmp_path):
        """Unknown dataset should return invalid result."""
        result = verify_dataset("UnknownDataset", tmp_path)
        assert result["valid"] is False
        assert "error" in result

    def test_empty_directory_returns_invalid(self, tmp_path):
        """Empty dataset directory should be invalid."""
        (tmp_path / "HAM10000").mkdir()
        result = verify_dataset("HAM10000", tmp_path)
        assert result["valid"] is False
        assert result["image_count"] == 0

    def test_includes_source_field(self, tmp_path):
        """Result should include source field."""
        (tmp_path / "HAM10000").mkdir()
        result = verify_dataset("HAM10000", tmp_path)
        assert "source" in result

    def test_ham10000_checks_correct_paths(self, tmp_path):
        """HAM10000 verification should check part_1 and part_2 folders."""
        dataset_dir = tmp_path / "HAM10000"
        dataset_dir.mkdir()

        # Create metadata
        (dataset_dir / "HAM10000_metadata.csv").touch()

        # Create image folders
        part1 = dataset_dir / "HAM10000_images_part_1"
        part1.mkdir()
        (part1 / "test1.jpg").touch()
        (part1 / "test2.jpg").touch()

        result = verify_dataset("HAM10000", tmp_path)
        assert result["image_count"] == 2
        assert result["csv_found"] is True

    def test_padufes20_checks_correct_paths(self, tmp_path):
        """PAD-UFES-20 verification should check imgs_part folders."""
        dataset_dir = tmp_path / "PAD-UFES-20"
        dataset_dir.mkdir()

        # Create metadata
        (dataset_dir / "metadata.csv").touch()

        # Create image folders
        for part in ["imgs_part_1", "imgs_part_2", "imgs_part_3"]:
            part_dir = dataset_dir / part
            part_dir.mkdir()
            (part_dir / "test.png").touch()

        result = verify_dataset("PAD-UFES-20", tmp_path)
        assert result["image_count"] == 3
        assert result["csv_found"] is True

    def test_completeness_calculation(self, tmp_path):
        """Completeness should be calculated correctly."""
        dataset_dir = tmp_path / "PAD-UFES-20"
        dataset_dir.mkdir()

        # Create some images (PAD-UFES-20 expects ~2298)
        imgs = dataset_dir / "imgs_part_1"
        imgs.mkdir()
        for i in range(100):
            (imgs / f"img_{i}.png").touch()

        result = verify_dataset("PAD-UFES-20", tmp_path)
        expected_completeness = (100 / 2298) * 100
        assert abs(result["completeness"] - expected_completeness) < 0.1


class TestVerifyAllDatasets:
    """Tests for verify_all_datasets function."""

    def test_returns_dict_for_all_datasets(self, tmp_path):
        """Should return results for all datasets."""
        results = verify_all_datasets(tmp_path)

        assert len(results) == len(DATASET_INFO)
        for name in DATASET_INFO:
            assert name in results


# =============================================================================
# Tests for Unified Download Function
# =============================================================================


class TestDownloadDataset:
    """Tests for download_dataset unified function."""

    def test_unknown_dataset_returns_false(self, tmp_path):
        """Unknown dataset should return False."""
        result = download_dataset("UnknownDataset", tmp_path)
        assert result is False

    @patch.object(download_module, 'check_kaggle_available')
    @patch.object(download_module, 'download_ham10000_kaggle')
    def test_ham10000_uses_kaggle_when_available(self, mock_download, mock_check, tmp_path):
        """HAM10000 should use Kaggle when available."""
        mock_check.return_value = True
        mock_download.return_value = True

        result = download_dataset("HAM10000", tmp_path)

        mock_download.assert_called_once()
        assert result is True

    @patch.object(download_module, 'check_kaggle_available')
    @patch.object(download_module, 'download_dataset_isic')
    def test_ham10000_falls_back_to_isic(self, mock_isic, mock_check, tmp_path):
        """HAM10000 should fall back to ISIC when Kaggle unavailable."""
        mock_check.return_value = False
        mock_isic.return_value = True

        result = download_dataset("HAM10000", tmp_path)

        mock_isic.assert_called_once()
        assert result is True

    @patch.object(download_module, 'download_padufes20_mendeley')
    def test_padufes20_uses_mendeley(self, mock_download, tmp_path):
        """PAD-UFES-20 should use Mendeley download."""
        mock_download.return_value = True

        result = download_dataset("PAD-UFES-20", tmp_path)

        mock_download.assert_called_once()
        assert result is True

    @patch.object(download_module, 'download_dataset_isic')
    def test_isic_datasets_use_isic(self, mock_download, tmp_path):
        """ISIC datasets should use ISIC Archive."""
        mock_download.return_value = True

        for name in ['ISIC2018', 'ISIC2019', 'ISIC2020']:
            mock_download.reset_mock()
            download_dataset(name, tmp_path)
            mock_download.assert_called_once()


# =============================================================================
# Tests for ISIC Archive Client
# =============================================================================


class TestISICArchiveClient:
    """Tests for ISICArchiveClient class."""

    def test_client_initialization(self):
        """Client should initialize with default parameters."""
        client = ISICArchiveClient()
        assert client.base_url == "https://api.isic-archive.com/api/v2"
        assert client.timeout == 30
        assert client.max_workers == 8

    def test_client_custom_parameters(self):
        """Client should accept custom parameters."""
        client = ISICArchiveClient(
            max_retries=10,
            timeout=60,
            max_workers=16
        )
        assert client.timeout == 60
        assert client.max_workers == 16

    @patch('requests.Session.get')
    def test_get_image_list_makes_request(self, mock_get):
        """get_image_list should make API request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = ISICArchiveClient()
        result = client.get_image_list(collection="HAM10000", limit=10)

        assert result == []
        mock_get.assert_called_once()


# =============================================================================
# Tests for get_data_root
# =============================================================================


class TestGetDataRoot:
    """Tests for get_data_root function."""

    def test_returns_path_object(self):
        """Should return a Path object."""
        result = get_data_root()
        assert isinstance(result, Path)

    def test_returns_data_directory(self):
        """Should return a path ending with 'data'."""
        result = get_data_root()
        assert result.name == "data"
