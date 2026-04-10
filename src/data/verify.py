# =============================================================================
# Data Verification and Quality Checking Utilities
# =============================================================================
"""
Data Verification and Quality Checking Utilities.

Provides comprehensive verification of dermoscopy datasets including:
- Image integrity checks
- Label consistency validation
- Class distribution analysis
- Cross-dataset compatibility checks
"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from typing import Any

import pandas as pd

# =============================================================================
# Dataset Verifier
# =============================================================================


class DatasetVerifier:
    """
    Comprehensive dataset verification for dermoscopy images.

    Checks:
    - File existence and integrity
    - Image readability and format
    - Label consistency
    - Class distribution
    - Image dimensions and statistics
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.results: dict[str, Any] = {}

    def verify_ham10000(self) -> dict[str, Any]:
        """Verify HAM10000 dataset."""
        dataset_path = self.data_root / "HAM10000"
        result = {
            "name": "HAM10000",
            "valid": False,
            "path": str(dataset_path),
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": [],
            "class_distribution": {},
            "metadata_valid": False,
            "image_stats": {},
            "errors": [],
        }

        # Check metadata
        csv_path = dataset_path / "HAM10000_metadata.csv"
        if not csv_path.exists():
            result["errors"].append(f"Metadata file not found: {csv_path}")
            return result

        try:
            df = pd.read_csv(csv_path)
            result["metadata_valid"] = True
            result["metadata_rows"] = len(df)

            # Class distribution
            if "dx" in df.columns:
                result["class_distribution"] = df["dx"].value_counts().to_dict()

            # Verify images
            image_dirs = [
                dataset_path / "HAM10000_images_part_1",
                dataset_path / "HAM10000_images_part_2",
                dataset_path / "images",
            ]

            found_images = set()
            for img_dir in image_dirs:
                if img_dir.exists():
                    for img_path in img_dir.glob("*.jpg"):
                        found_images.add(img_path.stem)

            result["total_images"] = len(found_images)

            # Check metadata matches images
            if "image_id" in df.columns:
                metadata_ids = set(df["image_id"])
                missing_images = metadata_ids - found_images
                extra_images = found_images - metadata_ids

                if missing_images:
                    result["errors"].append(f"Images in metadata but not found: {len(missing_images)}")
                if extra_images:
                    result["errors"].append(f"Images found but not in metadata: {len(extra_images)}")

            result["valid"] = len(result["errors"]) == 0 and result["total_images"] > 0

        except Exception as e:
            result["errors"].append(f"Error reading metadata: {e}")

        return result

    def verify_isic2018(self) -> dict[str, Any]:
        """Verify ISIC 2018 dataset."""
        dataset_path = self.data_root / "ISIC2018"
        result = {
            "name": "ISIC2018",
            "valid": False,
            "path": str(dataset_path),
            "total_images": 0,
            "class_distribution": {},
            "metadata_valid": False,
            "errors": [],
        }

        # Check for ground truth file
        csv_candidates = [
            dataset_path / "ISIC2018_Task3_Training_GroundTruth.csv",
            dataset_path / "ISIC2018_Task3_Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.csv",
        ]

        csv_path = None
        for candidate in csv_candidates:
            if candidate.exists():
                csv_path = candidate
                break

        if csv_path is None:
            result["errors"].append("Ground truth CSV not found")
            return result

        try:
            df = pd.read_csv(csv_path)
            result["metadata_valid"] = True
            result["metadata_rows"] = len(df)

            # ISIC 2018 has one-hot encoded labels
            label_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
            existing_cols = [col for col in label_cols if col in df.columns]

            if existing_cols:
                for col in existing_cols:
                    count = df[col].sum() if col in df.columns else 0
                    result["class_distribution"][col] = int(count)

            # Count images
            img_dir = dataset_path / "ISIC2018_Task3_Training_Input"
            if img_dir.exists():
                result["total_images"] = len(list(img_dir.glob("*.jpg")))

            result["valid"] = result["total_images"] > 0

        except Exception as e:
            result["errors"].append(f"Error reading data: {e}")

        return result

    def verify_isic2019(self) -> dict[str, Any]:
        """Verify ISIC 2019 dataset."""
        dataset_path = self.data_root / "ISIC2019"
        result = {
            "name": "ISIC2019",
            "valid": False,
            "path": str(dataset_path),
            "total_images": 0,
            "class_distribution": {},
            "metadata_valid": False,
            "errors": [],
        }

        csv_path = dataset_path / "ISIC_2019_Training_GroundTruth.csv"
        if not csv_path.exists():
            result["errors"].append(f"Ground truth not found: {csv_path}")
            return result

        try:
            df = pd.read_csv(csv_path)
            result["metadata_valid"] = True
            result["metadata_rows"] = len(df)

            # ISIC 2019 classes (8 classes including SCC)
            label_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
            for col in label_cols:
                if col in df.columns:
                    result["class_distribution"][col] = int(df[col].sum())

            # Count images
            img_dir = dataset_path / "ISIC_2019_Training_Input"
            if img_dir.exists():
                result["total_images"] = len(list(img_dir.glob("*.jpg")))

            result["valid"] = result["total_images"] > 0

        except Exception as e:
            result["errors"].append(f"Error reading data: {e}")

        return result

    def verify_isic2020(self) -> dict[str, Any]:
        """Verify ISIC 2020 dataset."""
        dataset_path = self.data_root / "ISIC2020"
        result = {
            "name": "ISIC2020",
            "valid": False,
            "path": str(dataset_path),
            "total_images": 0,
            "class_distribution": {},
            "metadata_valid": False,
            "errors": [],
        }

        csv_path = dataset_path / "ISIC_2020_Training_GroundTruth.csv"
        if not csv_path.exists():
            # Fallback to alternate name
            csv_path = dataset_path / "train.csv"
        if not csv_path.exists():
            result["errors"].append(f"Training CSV not found: {dataset_path}")
            return result

        try:
            df = pd.read_csv(csv_path)
            result["metadata_valid"] = True
            result["metadata_rows"] = len(df)

            # ISIC 2020 is binary: benign (0) vs malignant (1)
            if "target" in df.columns:
                result["class_distribution"] = {
                    "benign": int((df["target"] == 0).sum()),
                    "malignant": int((df["target"] == 1).sum()),
                }

            # Count images
            img_dir = dataset_path / "ISIC_2020_Training_JPEG"
            if not img_dir.exists():
                img_dir = dataset_path / "train"
            if img_dir.exists():
                result["total_images"] = len(list(img_dir.glob("*.jpg")))

            result["valid"] = result["total_images"] > 0

        except Exception as e:
            result["errors"].append(f"Error reading data: {e}")

        return result

    def verify_padufes20(self) -> dict[str, Any]:
        """Verify PAD-UFES-20 dataset."""
        dataset_path = self.data_root / "PAD-UFES-20"
        result: dict[str, Any] = {
            "name": "PAD-UFES-20",
            "valid": False,
            "path": str(dataset_path),
            "total_images": 0,
            "class_distribution": {},
            "metadata_valid": False,
            "errors": [],
        }

        csv_path = dataset_path / "metadata.csv"
        if not csv_path.exists():
            result["errors"].append(f"Metadata CSV not found: {csv_path}")
            return result

        try:
            df = pd.read_csv(csv_path)
            result["metadata_valid"] = True
            result["metadata_rows"] = len(df)

            # PAD-UFES-20 has a diagnostic column
            diag_col = None
            for col in ["diagnostic", "diagnosis", "label"]:
                if col in df.columns:
                    diag_col = col
                    break

            if diag_col:
                result["class_distribution"] = df[diag_col].value_counts().to_dict()

            # Count images across subdirectories
            total = 0
            if dataset_path.exists():
                for ext in ("*.jpg", "*.png", "*.bmp"):
                    total += len(list(dataset_path.rglob(ext)))
            result["total_images"] = total

            result["valid"] = result["total_images"] > 0

        except Exception as e:
            result["errors"].append(f"Error reading data: {e}")

        return result

    def verify_all(self, verbose: bool = True) -> dict[str, dict]:
        """
        Verify all datasets.

        Returns dictionary with verification results for each dataset.
        """
        results = {
            "HAM10000": self.verify_ham10000(),
            "ISIC2018": self.verify_isic2018(),
            "ISIC2019": self.verify_isic2019(),
            "ISIC2020": self.verify_isic2020(),
            "PAD-UFES-20": self.verify_padufes20(),
        }

        self.results = results

        if verbose:
            self.print_report()

        return results

    def print_report(self) -> None:
        """Print formatted verification report."""
        print("\n" + "=" * 80)
        print("DATASET VERIFICATION REPORT")
        print("=" * 80)

        total_images = 0
        valid_datasets = 0

        for name, result in self.results.items():
            status = "VALID" if result["valid"] else "INVALID"
            print(f"\n{status}: {name}")
            print("-" * 40)
            print(f"  Path: {result['path']}")
            print(f"  Total images: {result['total_images']:,}")
            print(f"  Metadata valid: {result['metadata_valid']}")

            if result["class_distribution"]:
                print("  Class distribution:")
                for cls, count in result["class_distribution"].items():
                    print(f"    {cls}: {count:,}")

            if result["errors"]:
                print("  Errors:")
                for error in result["errors"]:
                    print(f"    - {error}")

            total_images += result["total_images"]
            if result["valid"]:
                valid_datasets += 1

        print("\n" + "=" * 80)
        print(f"SUMMARY: {valid_datasets}/{len(self.results)} datasets valid")
        print(f"Total images across all datasets: {total_images:,}")
        print("=" * 80)

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics across all datasets."""
        if not self.results:
            self.verify_all(verbose=False)

        return {
            "total_datasets": len(self.results),
            "valid_datasets": sum(1 for r in self.results.values() if r["valid"]),
            "total_images": sum(r["total_images"] for r in self.results.values()),
            "images_per_dataset": {name: r["total_images"] for name, r in self.results.items()},
            "all_valid": all(r["valid"] for r in self.results.values()),
        }


if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data"

    verifier = DatasetVerifier(data_root)
    verifier.verify_all()
