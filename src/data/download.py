"""
Dataset Download and Setup Scripts for Dermoscopy Datasets.

This module provides utilities to download and organize:
- HAM10000 (Client 1)
- ISIC 2018 (Client 2)
- ISIC 2019 (Client 3)
- ISIC 2020 (Client 4)

Note: Some datasets require manual download from Kaggle or ISIC Archive.
This script provides guidance and verifies the setup.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
import requests
from tqdm import tqdm
import hashlib


# Dataset information and expected structure
DATASET_INFO = {
    "HAM10000": {
        "description": "Human Against Machine with 10000 training images",
        "source": "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000",
        "alt_source": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T",
        "classes": 7,
        "approx_images": 10015,
        "expected_files": [
            "HAM10000_metadata.csv",
            "HAM10000_images_part_1/",
            "HAM10000_images_part_2/"
        ],
        "client_id": 1
    },
    "ISIC2018": {
        "description": "ISIC 2018 Challenge - Task 3: Lesion Diagnosis",
        "source": "https://challenge.isic-archive.com/data/#2018",
        "kaggle": "https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic",
        "classes": 7,
        "approx_images": 10015,
        "expected_files": [
            "ISIC2018_Task3_Training_GroundTruth.csv",
            "ISIC2018_Task3_Training_Input/"
        ],
        "client_id": 2
    },
    "ISIC2019": {
        "description": "ISIC 2019 Challenge - Dermoscopic Image Classification",
        "source": "https://challenge.isic-archive.com/data/#2019",
        "kaggle": "https://www.kaggle.com/datasets/andrewmvd/isic-2019",
        "classes": 8,
        "approx_images": 25331,
        "expected_files": [
            "ISIC_2019_Training_GroundTruth.csv",
            "ISIC_2019_Training_Input/"
        ],
        "client_id": 3
    },
    "ISIC2020": {
        "description": "ISIC 2020 Challenge - Melanoma Classification",
        "source": "https://challenge.isic-archive.com/data/#2020",
        "kaggle": "https://www.kaggle.com/c/siim-isic-melanoma-classification/data",
        "classes": 2,
        "approx_images": 33126,
        "expected_files": [
            "train.csv",
            "train/"
        ],
        "client_id": 4
    }
}


def get_data_root() -> Path:
    """Get the data root directory."""
    # Try to find project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists() or (parent / "src").exists():
            return parent / "data"
    # Fallback
    return Path("./data")


def create_directory_structure(data_root: Optional[Path] = None) -> Dict[str, Path]:
    """
    Create the expected directory structure for all datasets.
    
    Returns:
        Dictionary mapping dataset names to their paths
    """
    if data_root is None:
        data_root = get_data_root()
    
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    for dataset_name in DATASET_INFO:
        dataset_path = data_root / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        paths[dataset_name] = dataset_path
        
    # Also create raw and processed subdirectories
    (data_root / "raw").mkdir(exist_ok=True)
    (data_root / "processed").mkdir(exist_ok=True)
    
    print(f"Created directory structure at: {data_root}")
    return paths


def verify_dataset(dataset_name: str, data_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Verify that a dataset is properly set up.
    
    Args:
        dataset_name: Name of dataset (HAM10000, ISIC2018, etc.)
        data_root: Root data directory
        
    Returns:
        Dictionary with verification results
    """
    if data_root is None:
        data_root = get_data_root()
    
    data_root = Path(data_root)
    
    if dataset_name not in DATASET_INFO:
        return {"valid": False, "error": f"Unknown dataset: {dataset_name}"}
    
    info = DATASET_INFO[dataset_name]
    dataset_path = data_root / dataset_name
    
    result = {
        "valid": True,
        "dataset": dataset_name,
        "path": str(dataset_path),
        "expected_files": info["expected_files"],
        "found_files": [],
        "missing_files": [],
        "image_count": 0,
        "csv_found": False
    }
    
    # Check expected files
    for expected in info["expected_files"]:
        full_path = dataset_path / expected
        if full_path.exists():
            result["found_files"].append(expected)
            if expected.endswith(".csv"):
                result["csv_found"] = True
        else:
            result["missing_files"].append(expected)
            result["valid"] = False
    
    # Count images if directory exists
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        result["image_count"] += len(list(dataset_path.rglob(ext)))
    
    return result


def verify_all_datasets(data_root: Optional[Path] = None) -> Dict[str, Dict]:
    """Verify all datasets and return summary."""
    results = {}
    for dataset_name in DATASET_INFO:
        results[dataset_name] = verify_dataset(dataset_name, data_root)
    return results


def print_verification_report(results: Dict[str, Dict]) -> None:
    """Print a formatted verification report."""
    print("\n" + "=" * 70)
    print("DATASET VERIFICATION REPORT")
    print("=" * 70)
    
    all_valid = True
    for dataset_name, result in results.items():
        status = "✓" if result["valid"] else "✗"
        all_valid = all_valid and result["valid"]
        
        print(f"\n{status} {dataset_name}")
        print(f"  Path: {result['path']}")
        print(f"  Images found: {result['image_count']}")
        print(f"  CSV metadata: {'Found' if result['csv_found'] else 'Missing'}")
        
        if result["missing_files"]:
            print(f"  Missing files:")
            for f in result["missing_files"]:
                print(f"    - {f}")
    
    print("\n" + "=" * 70)
    if all_valid:
        print("All datasets are properly configured!")
    else:
        print("Some datasets are missing. See instructions below.")
    print("=" * 70)


def print_download_instructions() -> None:
    """Print instructions for downloading each dataset."""
    instructions = """
================================================================================
DATASET DOWNLOAD INSTRUCTIONS
================================================================================

Due to data usage agreements, most dermoscopy datasets must be downloaded
manually from their official sources or Kaggle.

--------------------------------------------------------------------------------
CLIENT 1: HAM10000
--------------------------------------------------------------------------------
Option A (Kaggle - Recommended):
  1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
  2. Download and extract to: data/HAM10000/
  
Option B (Harvard Dataverse):
  1. Go to: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
  2. Download HAM10000_images_part_1.zip, HAM10000_images_part_2.zip
  3. Download HAM10000_metadata.csv
  4. Extract to: data/HAM10000/

Expected structure:
  data/HAM10000/
  ├── HAM10000_metadata.csv
  ├── HAM10000_images_part_1/
  │   └── *.jpg
  └── HAM10000_images_part_2/
      └── *.jpg

--------------------------------------------------------------------------------
CLIENT 2: ISIC 2018
--------------------------------------------------------------------------------
Option A (ISIC Archive):
  1. Go to: https://challenge.isic-archive.com/data/#2018
  2. Download Task 3 Training Data
  3. Extract to: data/ISIC2018/

Option B (Kaggle):
  1. Search for "ISIC 2018 Task 3" on Kaggle
  2. Download and extract to: data/ISIC2018/

Expected structure:
  data/ISIC2018/
  ├── ISIC2018_Task3_Training_GroundTruth.csv
  └── ISIC2018_Task3_Training_Input/
      └── *.jpg

--------------------------------------------------------------------------------
CLIENT 3: ISIC 2019
--------------------------------------------------------------------------------
Option A (ISIC Archive):
  1. Go to: https://challenge.isic-archive.com/data/#2019
  2. Download Training Data and Ground Truth
  3. Extract to: data/ISIC2019/

Option B (Kaggle):
  1. Go to: https://www.kaggle.com/datasets/andrewmvd/isic-2019
  2. Download and extract to: data/ISIC2019/

Expected structure:
  data/ISIC2019/
  ├── ISIC_2019_Training_GroundTruth.csv
  └── ISIC_2019_Training_Input/
      └── *.jpg

--------------------------------------------------------------------------------
CLIENT 4: ISIC 2020
--------------------------------------------------------------------------------
Option A (Kaggle Competition - Recommended):
  1. Go to: https://www.kaggle.com/c/siim-isic-melanoma-classification/data
  2. Accept competition rules
  3. Download train.csv and jpeg/train/ folder
  4. Extract to: data/ISIC2020/

Expected structure:
  data/ISIC2020/
  ├── train.csv
  └── train/
      └── *.jpg

================================================================================
ALTERNATIVE: Using Kaggle API
================================================================================
If you have Kaggle API configured:

    pip install kaggle
    
    # HAM10000
    kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
    
    # ISIC 2019
    kaggle datasets download -d andrewmvd/isic-2019
    
    # ISIC 2020
    kaggle competitions download -c siim-isic-melanoma-classification

================================================================================
"""
    print(instructions)


def download_with_kaggle_api(
    dataset_slug: str,
    output_dir: Path,
    is_competition: bool = False
) -> bool:
    """
    Download dataset using Kaggle API.
    
    Args:
        dataset_slug: Kaggle dataset identifier
        output_dir: Directory to save files
        is_competition: Whether this is a competition dataset
        
    Returns:
        True if successful
    """
    try:
        import kaggle
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if is_competition:
            kaggle.api.competition_download_files(
                dataset_slug,
                path=str(output_dir),
                quiet=False
            )
        else:
            kaggle.api.dataset_download_files(
                dataset_slug,
                path=str(output_dir),
                unzip=True,
                quiet=False
            )
        
        return True
        
    except ImportError:
        print("Kaggle API not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        return False


def setup_ham10000(data_root: Optional[Path] = None) -> bool:
    """Setup HAM10000 dataset."""
    if data_root is None:
        data_root = get_data_root()
    
    dataset_path = Path(data_root) / "HAM10000"
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    print("\nAttempting to download HAM10000 via Kaggle API...")
    success = download_with_kaggle_api(
        "kmader/skin-cancer-mnist-ham10000",
        dataset_path
    )
    
    if not success:
        print("\nPlease download HAM10000 manually.")
        print("See instructions above for download links.")
    
    return verify_dataset("HAM10000", data_root)["valid"]


def organize_downloaded_files(data_root: Optional[Path] = None) -> None:
    """
    Organize downloaded files into expected structure.
    
    Some Kaggle downloads have different folder structures.
    This function reorganizes them.
    """
    if data_root is None:
        data_root = get_data_root()
    
    data_root = Path(data_root)
    
    # HAM10000 reorganization
    ham_path = data_root / "HAM10000"
    if ham_path.exists():
        # Check if images are in root instead of subdirectories
        root_images = list(ham_path.glob("ISIC_*.jpg"))
        if root_images and not (ham_path / "HAM10000_images_part_1").exists():
            print("Reorganizing HAM10000 images...")
            part1 = ham_path / "HAM10000_images_part_1"
            part1.mkdir(exist_ok=True)
            for img in tqdm(root_images, desc="Moving images"):
                shutil.move(str(img), str(part1 / img.name))
    
    # ISIC2020 reorganization
    isic2020_path = data_root / "ISIC2020"
    if isic2020_path.exists():
        # Check for nested jpeg folder from Kaggle
        jpeg_folder = isic2020_path / "jpeg" / "train"
        if jpeg_folder.exists() and not (isic2020_path / "train").exists():
            print("Reorganizing ISIC2020 images...")
            shutil.move(str(jpeg_folder), str(isic2020_path / "train"))


class DatasetSetupWizard:
    """Interactive wizard for setting up datasets."""
    
    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else get_data_root()
        
    def run(self) -> None:
        """Run the interactive setup wizard."""
        print("\n" + "=" * 70)
        print("DERMOSCOPY DATASET SETUP WIZARD")
        print("=" * 70)
        
        # Create directory structure
        print("\nStep 1: Creating directory structure...")
        create_directory_structure(self.data_root)
        
        # Verify current state
        print("\nStep 2: Checking existing datasets...")
        results = verify_all_datasets(self.data_root)
        print_verification_report(results)
        
        # Check which datasets are missing
        missing = [name for name, result in results.items() if not result["valid"]]
        
        if not missing:
            print("\nAll datasets are ready! You can proceed with training.")
            return
        
        # Print download instructions
        print(f"\nMissing datasets: {', '.join(missing)}")
        print_download_instructions()
        
        # Ask about Kaggle API
        print("\nWould you like to try automatic download via Kaggle API?")
        print("(Requires kaggle package and API credentials)")
        
    def quick_verify(self) -> bool:
        """Quick verification of all datasets."""
        results = verify_all_datasets(self.data_root)
        return all(r["valid"] for r in results.values())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Setup Utility")
    parser.add_argument("--data-root", type=str, default=None,
                       help="Root directory for datasets")
    parser.add_argument("--verify", action="store_true",
                       help="Verify existing datasets")
    parser.add_argument("--instructions", action="store_true",
                       help="Print download instructions")
    parser.add_argument("--setup", action="store_true",
                       help="Run interactive setup wizard")
    
    args = parser.parse_args()
    
    if args.verify:
        results = verify_all_datasets(args.data_root)
        print_verification_report(results)
    elif args.instructions:
        print_download_instructions()
    elif args.setup:
        wizard = DatasetSetupWizard(args.data_root)
        wizard.run()
    else:
        # Default: show status and instructions
        results = verify_all_datasets(args.data_root)
        print_verification_report(results)
        
        missing = [name for name, result in results.items() if not result["valid"]]
        if missing:
            print_download_instructions()
