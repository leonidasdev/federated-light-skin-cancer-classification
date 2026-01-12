"""
ISIC 2018 Challenge Dataset Loader
==================================

International Skin Imaging Collaboration 2018 Challenge Dataset.
Task 3: Lesion Diagnosis
https://challenge.isic-archive.com/landing/2018/

Dataset Statistics:
    - Training images: 10,015 (same as HAM10000)
    - Validation images: 193
    - Test images: 1,512 (labels not public)
    - Image size: Variable (mostly 600x450)
    - Classes: 7 skin lesion types

Classes:
    - MEL: Melanoma
    - NV: Melanocytic nevus
    - BCC: Basal cell carcinoma
    - AKIEC: Actinic keratosis / Bowen's disease
    - BKL: Benign keratosis
    - DF: Dermatofibroma
    - VASC: Vascular lesion
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image


class ISIC2018Dataset(Dataset):
    """
    ISIC 2018 Challenge Dataset for skin lesion classification (Task 3).
    
    Expected directory structure:
        root/
        ├── ISIC2018_Task3_Training_Input/
        │   └── *.jpg
        ├── ISIC2018_Task3_Training_GroundTruth/
        │   └── ISIC2018_Task3_Training_GroundTruth.csv
        ├── ISIC2018_Task3_Validation_Input/
        │   └── *.jpg
        └── ISIC2018_Task3_Validation_GroundTruth/
            └── ISIC2018_Task3_Validation_GroundTruth.csv
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ('train', 'val')
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to targets
    """
    
    # Class names - ISIC 2018 uses uppercase abbreviations
    CLASS_NAMES_ISIC = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
    NUM_CLASSES = 7
    
    # Mapping from ISIC uppercase to index
    ISIC_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES_ISIC)}
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.samples: List[Tuple[Path, int]] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """
        Load ISIC 2018 dataset.
        
        The dataset provides separate folders for training and validation.
        Ground truth is provided as one-hot encoded CSV files.
        """
        if self.split == "train":
            img_dir = self.root / "ISIC2018_Task3_Training_Input"
            gt_file = self.root / "ISIC2018_Task3_Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.csv"
        elif self.split in ["val", "test"]:
            img_dir = self.root / "ISIC2018_Task3_Validation_Input"
            gt_file = self.root / "ISIC2018_Task3_Validation_GroundTruth" / "ISIC2018_Task3_Validation_GroundTruth.csv"
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # Try alternative directory structures
        if not img_dir.exists():
            alt_dirs = [
                self.root / "Training_Input" if self.split == "train" else self.root / "Validation_Input",
                self.root / "train" / "images" if self.split == "train" else self.root / "val" / "images",
                self.root / "images",
            ]
            for alt in alt_dirs:
                if alt.exists():
                    img_dir = alt
                    break
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found at {img_dir}")
        
        # Find ground truth file
        if not gt_file.exists():
            csv_files = list(self.root.rglob("*.csv"))
            gt_candidates = [f for f in csv_files if "GroundTruth" in f.name or "groundtruth" in f.name.lower()]
            if gt_candidates:
                # Find the appropriate one for the split
                for candidate in gt_candidates:
                    if (self.split == "train" and "Training" in candidate.name) or \
                       (self.split != "train" and "Validation" in candidate.name):
                        gt_file = candidate
                        break
                if not gt_file.exists() and gt_candidates:
                    gt_file = gt_candidates[0]
        
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found")
        
        # Load ground truth (one-hot encoded)
        gt_df = pd.read_csv(gt_file)
        
        # Find image ID column
        id_col = gt_df.columns[0]  # Usually 'image' or similar
        
        # Build samples
        for _, row in gt_df.iterrows():
            image_id = row[id_col]
            
            # Find image
            img_path = img_dir / f"{image_id}.jpg"
            if not img_path.exists():
                img_path = img_dir / f"{image_id}.png"
            if not img_path.exists():
                continue
            
            # Get label from one-hot encoding
            label = self._get_label_from_row(row)
            if label is not None:
                self.samples.append((img_path, label))
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split '{self.split}'")
    
    def _get_label_from_row(self, row: pd.Series) -> Optional[int]:
        """Extract class label from one-hot encoded row."""
        for class_name in self.CLASS_NAMES_ISIC:
            if class_name in row.index and row[class_name] == 1.0:
                return self.ISIC_TO_IDX[class_name]
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        labels = [label for _, label in self.samples]
        counts = Counter(labels)
        total = len(labels)
        weights = torch.zeros(self.NUM_CLASSES)
        for cls_idx in range(self.NUM_CLASSES):
            count = counts.get(cls_idx, 1)
            weights[cls_idx] = total / (self.NUM_CLASSES * count)
        return weights
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in this split."""
        labels = [label for _, label in self.samples]
        counts = Counter(labels)
        return {self.IDX_TO_CLASS[idx]: counts.get(idx, 0) for idx in range(self.NUM_CLASSES)}
