# =============================================================================
# Tests for Data Verification Module
# =============================================================================
"""Tests for src.data.verify.DatasetVerifier."""

import pandas as pd
import pytest

from src.data.verify import DatasetVerifier


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def data_root(tmp_path):
    """Create a temporary data root directory."""
    return tmp_path / "data"


@pytest.fixture
def verifier(data_root):
    """Create a DatasetVerifier with temp directory."""
    data_root.mkdir()
    return DatasetVerifier(str(data_root))


# =============================================================================
# HAM10000 Tests
# =============================================================================


class TestVerifyHAM10000:
    def test_missing_csv(self, verifier):
        result = verifier.verify_ham10000()
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_valid_dataset(self, data_root):
        ds_path = data_root / "HAM10000"
        ds_path.mkdir(parents=True)
        img_dir = ds_path / "HAM10000_images_part_1"
        img_dir.mkdir()

        # Create metadata CSV
        df = pd.DataFrame({
            "image_id": ["img001", "img002", "img003"],
            "dx": ["mel", "nv", "bcc"],
        })
        df.to_csv(ds_path / "HAM10000_metadata.csv", index=False)

        # Create dummy images
        for img_id in ["img001", "img002", "img003"]:
            (img_dir / f"{img_id}.jpg").write_text("fake")

        verifier = DatasetVerifier(str(data_root))
        result = verifier.verify_ham10000()
        assert result["metadata_valid"] is True
        assert result["total_images"] == 3
        assert result["valid"] is True
        assert "mel" in result["class_distribution"]

    def test_missing_images(self, data_root):
        ds_path = data_root / "HAM10000"
        ds_path.mkdir(parents=True)

        df = pd.DataFrame({
            "image_id": ["img001", "img002"],
            "dx": ["mel", "nv"],
        })
        df.to_csv(ds_path / "HAM10000_metadata.csv", index=False)

        verifier = DatasetVerifier(str(data_root))
        result = verifier.verify_ham10000()
        assert result["total_images"] == 0
        assert result["valid"] is False


# =============================================================================
# ISIC2018 Tests
# =============================================================================


class TestVerifyISIC2018:
    def test_missing_csv(self, verifier):
        result = verifier.verify_isic2018()
        assert result["valid"] is False

    def test_valid_dataset(self, data_root):
        ds_path = data_root / "ISIC2018"
        ds_path.mkdir(parents=True)
        img_dir = ds_path / "ISIC2018_Task3_Training_Input"
        img_dir.mkdir()

        df = pd.DataFrame({
            "image": ["A", "B"],
            "MEL": [1, 0],
            "NV": [0, 1],
            "BCC": [0, 0],
        })
        df.to_csv(ds_path / "ISIC2018_Task3_Training_GroundTruth.csv", index=False)

        (img_dir / "A.jpg").write_text("fake")
        (img_dir / "B.jpg").write_text("fake")

        verifier = DatasetVerifier(str(data_root))
        result = verifier.verify_isic2018()
        assert result["metadata_valid"] is True
        assert result["total_images"] == 2
        assert result["valid"] is True
        assert result["class_distribution"]["MEL"] == 1


# =============================================================================
# ISIC2019 Tests
# =============================================================================


class TestVerifyISIC2019:
    def test_missing_csv(self, verifier):
        result = verifier.verify_isic2019()
        assert result["valid"] is False

    def test_valid_dataset(self, data_root):
        ds_path = data_root / "ISIC2019"
        ds_path.mkdir(parents=True)
        img_dir = ds_path / "ISIC_2019_Training_Input"
        img_dir.mkdir()

        df = pd.DataFrame({
            "image": ["A"],
            "MEL": [1], "NV": [0], "BCC": [0], "AK": [0],
            "BKL": [0], "DF": [0], "VASC": [0], "SCC": [0],
        })
        df.to_csv(ds_path / "ISIC_2019_Training_GroundTruth.csv", index=False)
        (img_dir / "A.jpg").write_text("fake")

        verifier = DatasetVerifier(str(data_root))
        result = verifier.verify_isic2019()
        assert result["valid"] is True


# =============================================================================
# ISIC2020 Tests
# =============================================================================


class TestVerifyISIC2020:
    def test_missing_csv(self, verifier):
        result = verifier.verify_isic2020()
        assert result["valid"] is False

    def test_valid_dataset(self, data_root):
        ds_path = data_root / "ISIC2020"
        ds_path.mkdir(parents=True)
        img_dir = ds_path / "train"
        img_dir.mkdir()

        df = pd.DataFrame({
            "image_name": ["X", "Y"],
            "target": [0, 1],
        })
        df.to_csv(ds_path / "train.csv", index=False)
        (img_dir / "X.jpg").write_text("fake")
        (img_dir / "Y.jpg").write_text("fake")

        verifier = DatasetVerifier(str(data_root))
        result = verifier.verify_isic2020()
        assert result["valid"] is True
        assert result["class_distribution"]["benign"] == 1
        assert result["class_distribution"]["malignant"] == 1


# =============================================================================
# verify_all / print_report / get_summary_stats Tests
# =============================================================================


class TestVerifyAll:
    def test_verify_all_runs(self, verifier):
        results = verifier.verify_all(verbose=False)
        assert "HAM10000" in results
        assert "ISIC2018" in results
        assert "ISIC2019" in results
        assert "ISIC2020" in results
        assert "PAD-UFES-20" in results

    def test_verify_all_with_report(self, verifier, capsys):
        verifier.verify_all(verbose=True)
        captured = capsys.readouterr()
        assert "DATASET VERIFICATION REPORT" in captured.out
        assert "SUMMARY" in captured.out


class TestGetSummaryStats:
    def test_summary_stats(self, verifier):
        verifier.verify_all(verbose=False)
        stats = verifier.get_summary_stats()
        assert stats["total_datasets"] == 5
        assert "all_valid" in stats
        assert "images_per_dataset" in stats

    def test_summary_stats_auto_verifies(self, verifier):
        stats = verifier.get_summary_stats()
        assert stats["total_datasets"] == 5
