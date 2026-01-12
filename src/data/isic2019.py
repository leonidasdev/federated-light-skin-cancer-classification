"""
ISIC 2019 Challenge Dataset Loader
==================================

International Skin Imaging Collaboration 2019 Challenge Dataset.
https://challenge.isic-archive.com/landing/2019/

Dataset Statistics:
    - Training images: 25,331
    - Test images: 8,238 (labels not public)
    - Image size: Variable
    - Classes: 8 skin lesion types (added SCC)

Classes:
    - MEL: Melanoma
    - NV: Melanocytic nevus
    - BCC: Basal cell carcinoma
    - AK: Actinic keratosis (renamed from AKIEC)
    - BKL: Benign keratosis
    - DF: Dermatofibroma
    - VASC: Vascular lesion
    - SCC: Squamous cell carcinoma (NEW in 2019)
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


class ISIC2019Dataset(Dataset):
    """
    ISIC 2019 Challenge Dataset for skin lesion classification.
    
    Expected directory structure:
        root/
        ├── ISIC_2019_Training_Input/
        │   └── *.jpg
        └── ISIC_2019_Training_GroundTruth.csv
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to targets
        train_ratio: Ratio for train split (default: 0.8)
        val_ratio: Ratio for val split (default: 0.1)
        seed: Random seed for splits
    """
    
    CLASS_NAMES_ISIC = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    CLASS_NAMES = ['mel', 'nv', 'bcc', 'ak', 'bkl', 'df', 'vasc', 'scc']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
    ISIC_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES_ISIC)}
    NUM_CLASSES = 8
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        
        self.samples: List[Tuple[Path, int]] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load dataset images and labels with train/val/test splits."""
        # Find image directory
        img_dir = self.root / "ISIC_2019_Training_Input"
        if not img_dir.exists():
            for alt in [self.root / "train" / "images", self.root / "images", self.root]:
                if alt.exists() and any(alt.glob("*.jpg")):
                    img_dir = alt
                    break
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found")
        
        # Find ground truth file
        gt_file = self.root / "ISIC_2019_Training_GroundTruth.csv"
        if not gt_file.exists():
            csv_files = list(self.root.rglob("*.csv"))
            for f in csv_files:
                if "GroundTruth" in f.name or "groundtruth" in f.name.lower():
                    gt_file = f
                    break
        
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found")
        
        # Load ground truth
        gt_df = pd.read_csv(gt_file)
        id_col = gt_df.columns[0]
        
        # Build all samples
        all_samples = []
        for _, row in gt_df.iterrows():
            image_id = row[id_col]
            img_path = img_dir / f"{image_id}.jpg"
            if not img_path.exists():
                continue
            
            label = self._get_label_from_row(row)
            if label is not None:
                all_samples.append((img_path, label))
        
        # Create stratified splits
        np.random.seed(self.seed)
        labels = [s[1] for s in all_samples]
        indices = np.arange(len(all_samples))
        
        train_idx, val_idx, test_idx = [], [], []
        for cls in range(self.NUM_CLASSES):
            cls_indices = indices[np.array(labels) == cls]
            np.random.shuffle(cls_indices)
            n = len(cls_indices)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            train_idx.extend(cls_indices[:n_train])
            val_idx.extend(cls_indices[n_train:n_train + n_val])
            test_idx.extend(cls_indices[n_train + n_val:])
        
        # Select samples for current split
        if self.split == "train":
            selected = train_idx
        elif self.split == "val":
            selected = val_idx
        elif self.split == "test":
            selected = test_idx
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        self.samples = [all_samples[i] for i in selected]
        
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
