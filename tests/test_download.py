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
        for info in DATASET_INFO.values():
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

    @patch('requests.Session.get')
    def test_make_request_raises_on_error(self, mock_get):
        """_make_request should raise on HTTP errors."""
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("fail")

        client = ISICArchiveClient()
        with pytest.raises(req.exceptions.ConnectionError):
            client._make_request("/images/")

    @patch('requests.Session.get')
    def test_download_image_success(self, mock_get, tmp_path):
        """download_image should save file on success."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content.return_value = [b"fake image data"]
        mock_get.return_value = mock_response

        client = ISICArchiveClient()
        out = tmp_path / "img.jpg"
        result = client.download_image("ISIC_0000001", out)
        assert result is True
        assert out.exists()
        assert out.read_bytes() == b"fake image data"

    @patch('requests.Session.get')
    def test_download_image_failure(self, mock_get, tmp_path):
        """download_image should return False on network error."""
        mock_get.side_effect = Exception("network error")

        client = ISICArchiveClient()
        result = client.download_image("ISIC_0000001", tmp_path / "img.jpg")
        assert result is False

    @patch('requests.Session.get')
    def test_download_image_404_with_metadata_fallback(self, mock_get, tmp_path):
        """download_image should try metadata fallback on 404."""
        import requests as req

        # First call: 404 HTTPError
        resp_404 = MagicMock()
        resp_404.status_code = 404
        http_err = req.exceptions.HTTPError(response=resp_404)

        # Second call: metadata with file URL
        meta_resp = MagicMock()
        meta_resp.raise_for_status = MagicMock()
        meta_resp.json.return_value = {
            "files": {"full": {"url": "https://example.com/img.jpg"}}
        }

        # Third call: actual image download
        img_resp = MagicMock()
        img_resp.raise_for_status = MagicMock()
        img_resp.iter_content.return_value = [b"image data"]

        mock_get.side_effect = [http_err, meta_resp, img_resp]

        client = ISICArchiveClient()
        result = client.download_image("ISIC_0000001", tmp_path / "img.jpg")
        assert result is True

    def test_download_worker_skips_existing(self, tmp_path):
        """_download_worker should skip files that already exist."""
        out = tmp_path / "img.jpg"
        out.write_bytes(b"existing")

        client = ISICArchiveClient()
        _image_id, success = client._download_worker(("ISIC_0000001", out))
        assert success is True

    @patch('requests.Session.get')
    def test_download_images_parallel(self, mock_get, tmp_path):
        """download_images_parallel should return stats dict."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        client = ISICArchiveClient()
        images = [{"isic_id": f"ISIC_{i:07d}"} for i in range(3)]
        stats = client.download_images_parallel(images, tmp_path, max_workers=2)

        assert "total" in stats
        assert "success" in stats
        assert "failed" in stats
        assert stats["total"] == 3

    def test_save_metadata_csv(self, tmp_path):
        """save_metadata_csv should write a valid CSV."""
        client = ISICArchiveClient()
        images = [
            {
                "isic_id": "ISIC_0000001",
                "attribution": "test",
                "metadata": {"diagnosis": "melanoma", "clinical": {"sex": "male"}},
            },
            {
                "isic_id": "ISIC_0000002",
                "metadata": {"clinical": {"diagnosis": "nevus"}},
            },
        ]
        out = tmp_path / "meta.csv"
        client.save_metadata_csv(images, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "ISIC_0000001" in content
        assert "ISIC_0000002" in content

    def test_save_metadata_csv_empty(self, tmp_path):
        """save_metadata_csv with empty list should not create a file."""
        client = ISICArchiveClient()
        out = tmp_path / "empty.csv"
        client.save_metadata_csv([], out)
        assert not out.exists()


# =============================================================================
# Tests for download_dataset_isic
# =============================================================================


class TestDownloadDatasetISIC:
    """Tests for download_dataset_isic function."""

    def test_unknown_dataset_returns_false(self, tmp_path):
        result = download_module.download_dataset_isic("UnknownDataset", tmp_path)
        assert result is False

    @patch.object(download_module.ISICArchiveClient, 'get_all_images_for_collection')
    def test_returns_false_when_no_images(self, mock_get, tmp_path):
        mock_get.return_value = []
        result = download_module.download_dataset_isic("ISIC2018", tmp_path)
        assert result is False

    def test_already_complete_returns_true(self, tmp_path):
        """Should skip download when images already present."""
        ds_dir = tmp_path / "ISIC2018" / "images"
        ds_dir.mkdir(parents=True)
        # Create enough dummy images to pass 95% threshold
        approx = DATASET_INFO["ISIC2018"]["approx_images"]
        for i in range(int(approx * 0.96)):
            (ds_dir / f"img_{i}.jpg").touch()
        result = download_module.download_dataset_isic("ISIC2018", tmp_path)
        assert result is True


# =============================================================================
# Tests for download_all_datasets
# =============================================================================


class TestDownloadAllDatasets:
    """Tests for download_all_datasets function."""

    @patch.object(download_module, 'download_dataset')
    def test_downloads_all(self, mock_dl, tmp_path):
        mock_dl.return_value = True
        results = download_module.download_all_datasets(tmp_path)
        assert len(results) == len(DATASET_INFO)
        assert all(results.values())

    @patch.object(download_module, 'download_dataset')
    def test_downloads_subset(self, mock_dl, tmp_path):
        mock_dl.return_value = True
        results = download_module.download_all_datasets(tmp_path, datasets=["HAM10000"])
        assert len(results) == 1
        assert "HAM10000" in results


# =============================================================================
# Tests for print functions
# =============================================================================


class TestPrintFunctions:
    """Tests for print_verification_report and print_download_instructions."""

    def test_print_verification_report(self, capsys, tmp_path):
        results = verify_all_datasets(tmp_path)
        download_module.print_verification_report(results)
        captured = capsys.readouterr()
        assert "DATASET VERIFICATION REPORT" in captured.out

    def test_print_download_instructions(self, capsys):
        download_module.print_download_instructions()
        captured = capsys.readouterr()
        assert "DOWNLOAD" in captured.out
        assert "HAM10000" in captured.out


# =============================================================================
# Tests for DatasetSetupWizard
# =============================================================================


class TestDatasetSetupWizard:
    """Tests for DatasetSetupWizard class."""

    def test_init_with_path(self, tmp_path):
        wizard = download_module.DatasetSetupWizard(data_root=tmp_path)
        assert wizard.data_root == tmp_path

    def test_init_default_path(self):
        wizard = download_module.DatasetSetupWizard()
        assert wizard.data_root.name == "data"

    @patch.object(download_module, 'verify_all_datasets')
    @patch.object(download_module, 'print_verification_report')
    def test_run_all_valid(self, mock_print, mock_verify, tmp_path, capsys):
        mock_verify.return_value = {
            name: {"valid": True} for name in DATASET_INFO
        }
        wizard = download_module.DatasetSetupWizard(data_root=tmp_path)
        wizard.run()
        captured = capsys.readouterr()
        assert "ready" in captured.out.lower() or "proceed" in captured.out.lower()

    @patch.object(download_module, 'download_all_datasets')
    @patch.object(download_module, 'print_verification_report')
    @patch.object(download_module, 'verify_all_datasets')
    def test_run_auto_download(self, mock_verify, mock_print, mock_dl, tmp_path):
        mock_verify.return_value = {
            "HAM10000": {"valid": False},
            "ISIC2018": {"valid": True},
        }
        mock_dl.return_value = {"HAM10000": True}
        wizard = download_module.DatasetSetupWizard(data_root=tmp_path)
        wizard.run(auto_download=True)
        mock_dl.assert_called_once()


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
