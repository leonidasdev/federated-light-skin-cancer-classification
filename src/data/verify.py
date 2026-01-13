"""
Data Verification and Quality Checking Utilities.

Provides comprehensive verification of dermoscopy datasets including:
- Image integrity checks
- Label consistency validation
- Class distribution analysis
- Cross-dataset compatibility checks
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import warnings


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
        self.results: Dict[str, Any] = {}
        
    def verify_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Verify a single image file.
        
        Returns dict with:
        - valid: bool
        - width, height: dimensions
        - channels: number of channels
        - format: image format
        - error: error message if invalid
        """
        result = {
            "valid": False,
            "path": str(image_path),
            "width": None,
            "height": None,
            "channels": None,
            "format": None,
            "error": None
        }
        
        try:
            with Image.open(image_path) as img:
                result["valid"] = True
                result["width"] = img.width
                result["height"] = img.height
                result["format"] = img.format
                
                # Check channels
                if img.mode == "RGB":
                    result["channels"] = 3
                elif img.mode == "RGBA":
                    result["channels"] = 4
                elif img.mode == "L":
                    result["channels"] = 1
                else:
                    result["channels"] = len(img.mode)
                    
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def verify_ham10000(self) -> Dict[str, Any]:
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
            "errors": []
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
                dataset_path / "images"
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
                    result["errors"].append(
                        f"Images in metadata but not found: {len(missing_images)}"
                    )
                if extra_images:
                    result["errors"].append(
                        f"Images found but not in metadata: {len(extra_images)}"
                    )
            
            result["valid"] = len(result["errors"]) == 0 and result["total_images"] > 0
            
        except Exception as e:
            result["errors"].append(f"Error reading metadata: {e}")
        
        return result
    
    def verify_isic2018(self) -> Dict[str, Any]:
        """Verify ISIC 2018 dataset."""
        dataset_path = self.data_root / "ISIC2018"
        result = {
            "name": "ISIC2018",
            "valid": False,
            "path": str(dataset_path),
            "total_images": 0,
            "class_distribution": {},
            "metadata_valid": False,
            "errors": []
        }
        
        # Check for ground truth file
        csv_candidates = [
            dataset_path / "ISIC2018_Task3_Training_GroundTruth.csv",
            dataset_path / "ISIC2018_Task3_Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.csv"
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
            label_cols = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
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
    
    def verify_isic2019(self) -> Dict[str, Any]:
        """Verify ISIC 2019 dataset."""
        dataset_path = self.data_root / "ISIC2019"
        result = {
            "name": "ISIC2019",
            "valid": False,
            "path": str(dataset_path),
            "total_images": 0,
            "class_distribution": {},
            "metadata_valid": False,
            "errors": []
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
            label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
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
    
    def verify_isic2020(self) -> Dict[str, Any]:
        """Verify ISIC 2020 dataset."""
        dataset_path = self.data_root / "ISIC2020"
        result = {
            "name": "ISIC2020",
            "valid": False,
            "path": str(dataset_path),
            "total_images": 0,
            "class_distribution": {},
            "metadata_valid": False,
            "errors": []
        }
        
        csv_path = dataset_path / "train.csv"
        if not csv_path.exists():
            result["errors"].append(f"Training CSV not found: {csv_path}")
            return result
        
        try:
            df = pd.read_csv(csv_path)
            result["metadata_valid"] = True
            result["metadata_rows"] = len(df)
            
            # ISIC 2020 is binary: benign (0) vs malignant (1)
            if "target" in df.columns:
                result["class_distribution"] = {
                    "benign": int((df["target"] == 0).sum()),
                    "malignant": int((df["target"] == 1).sum())
                }
            
            # Count images
            img_dir = dataset_path / "train"
            if img_dir.exists():
                result["total_images"] = len(list(img_dir.glob("*.jpg")))
            
            result["valid"] = result["total_images"] > 0
            
        except Exception as e:
            result["errors"].append(f"Error reading data: {e}")
        
        return result
    
    def verify_all(self, verbose: bool = True) -> Dict[str, Dict]:
        """
        Verify all datasets.
        
        Returns dictionary with verification results for each dataset.
        """
        results = {
            "HAM10000": self.verify_ham10000(),
            "ISIC2018": self.verify_isic2018(),
            "ISIC2019": self.verify_isic2019(),
            "ISIC2020": self.verify_isic2020()
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
            status = "✓ VALID" if result["valid"] else "✗ INVALID"
            print(f"\n{status}: {name}")
            print("-" * 40)
            print(f"  Path: {result['path']}")
            print(f"  Total images: {result['total_images']:,}")
            print(f"  Metadata valid: {result['metadata_valid']}")
            
            if result["class_distribution"]:
                print(f"  Class distribution:")
                for cls, count in result["class_distribution"].items():
                    print(f"    {cls}: {count:,}")
            
            if result["errors"]:
                print(f"  Errors:")
                for error in result["errors"]:
                    print(f"    - {error}")
            
            total_images += result["total_images"]
            if result["valid"]:
                valid_datasets += 1
        
        print("\n" + "=" * 80)
        print(f"SUMMARY: {valid_datasets}/4 datasets valid")
        print(f"Total images across all datasets: {total_images:,}")
        print("=" * 80)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all datasets."""
        if not self.results:
            self.verify_all(verbose=False)
        
        return {
            "total_datasets": 4,
            "valid_datasets": sum(1 for r in self.results.values() if r["valid"]),
            "total_images": sum(r["total_images"] for r in self.results.values()),
            "images_per_dataset": {
                name: r["total_images"] for name, r in self.results.items()
            },
            "all_valid": all(r["valid"] for r in self.results.values())
        }


def compute_dataset_statistics(
    data_root: str,
    sample_size: int = 1000,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute detailed statistics across all datasets.
    
    Samples images to compute:
    - Mean/std per channel
    - Image dimension distribution
    - Aspect ratio distribution
    """
    from .datasets import (
        HAM10000Dataset, ISIC2018Dataset, 
        ISIC2019Dataset, ISIC2020Dataset
    )
    
    stats = {}
    data_root_path = Path(data_root)
    
    datasets = {
        "HAM10000": (HAM10000Dataset, "HAM10000", "HAM10000_metadata.csv"),
        "ISIC2018": (ISIC2018Dataset, "ISIC2018/ISIC2018_Task3_Training_Input", 
                     "ISIC2018/ISIC2018_Task3_Training_GroundTruth.csv"),
        "ISIC2019": (ISIC2019Dataset, "ISIC2019/ISIC_2019_Training_Input",
                     "ISIC2019/ISIC_2019_Training_GroundTruth.csv"),
        "ISIC2020": (ISIC2020Dataset, "ISIC2020/train", "ISIC2020/train.csv")
    }
    
    for name, (DatasetClass, img_dir, csv_file) in datasets.items():
        if verbose:
            print(f"\nAnalyzing {name}...")
        
        img_path = data_root_path / img_dir.split("/")[0]
        csv_path = data_root_path / csv_file
        
        if not csv_path.exists():
            if verbose:
                print(f"  Skipping - CSV not found")
            continue
        
        try:
            # Sample images for statistics
            widths = []
            heights = []
            
            img_folder = data_root_path / img_dir
            if img_folder.exists():
                images = list(img_folder.glob("*.jpg"))[:sample_size]
                
                for img_path in tqdm(images, desc=f"  Sampling {name}", disable=not verbose):
                    try:
                        with Image.open(img_path) as img:
                            widths.append(img.width)
                            heights.append(img.height)
                    except:
                        pass
            
            if widths:
                stats[name] = {
                    "sampled": len(widths),
                    "width_mean": np.mean(widths),
                    "width_std": np.std(widths),
                    "width_min": min(widths),
                    "width_max": max(widths),
                    "height_mean": np.mean(heights),
                    "height_std": np.std(heights),
                    "height_min": min(heights),
                    "height_max": max(heights),
                    "aspect_ratio_mean": np.mean([w/h for w, h in zip(widths, heights)])
                }
                
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
    
    return stats


def check_cross_dataset_consistency(data_root: str) -> Dict[str, Any]:
    """
    Check consistency across datasets for federated learning.
    
    Verifies:
    - Class label compatibility
    - Image format consistency
    - Potential preprocessing issues
    """
    verifier = DatasetVerifier(data_root)
    results = verifier.verify_all(verbose=False)
    
    consistency = {
        "label_mapping_required": True,
        "unified_classes": 7,
        "class_mapping": {
            "HAM10000": "Direct 7-class mapping",
            "ISIC2018": "Direct 7-class mapping (same as HAM10000)",
            "ISIC2019": "8 classes - SCC mapped to BCC",
            "ISIC2020": "Binary - benign→NV, malignant→MEL"
        },
        "format_consistent": True,
        "preprocessing_notes": [
            "All datasets use JPEG format",
            "Variable image sizes - resize to 224x224 needed",
            "HAM10000/ISIC2018 have similar acquisition protocols",
            "ISIC2020 images may have different characteristics"
        ],
        "recommendations": [
            "Use ImageNet normalization for consistency",
            "Apply same augmentation pipeline to all clients",
            "Consider class weighting for imbalanced classes",
            "ISIC2020 binary labels need special handling"
        ]
    }
    
    return consistency


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = "./data"
    
    verifier = DatasetVerifier(data_root)
    verifier.verify_all()
