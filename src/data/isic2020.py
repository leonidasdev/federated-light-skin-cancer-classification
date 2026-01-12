"""
ISIC 2020 Challenge Dataset Loader
==================================

International Skin Imaging Collaboration 2020 Grand Challenge.
https://challenge.isic-archive.com/landing/2020/

Dataset Statistics:
    - Training images: 33,126
    - Test images: 10,982 (labels not public)
    - Image size: Variable (mostly JPEG)
    - Classes: 2 (Binary classification)

Classes:
    - Benign: Non-melanoma lesions (majority)
    - Malignant: Melanoma lesions (minority ~2%)

Note: This dataset is highly imbalanced with melanoma being rare.
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


class ISIC2020Dataset(Dataset):
    """
    ISIC 2020 Challenge Dataset for melanoma classification.
    
    Binary classification task: Melanoma vs. Benign
    
    Expected directory structure:
        root/
        ├── train/
        │   └── *.jpg
        └── train.csv
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to targets
        train_ratio: Ratio for train split (default: 0.8)
        val_ratio: Ratio for val split (default: 0.1)
        seed: Random seed for splits
    """
    
    CLASS_NAMES = ['benign', 'malignant']
    CLASS_TO_IDX = {'benign': 0, 'malignant': 1}
    IDX_TO_CLASS = {0: 'benign', 1: 'malignant'}
    NUM_CLASSES = 2
    
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
        img_dir = None
        for candidate in [self.root / "train", self.root / "images", self.root]:
            if candidate.exists() and (any(candidate.glob("*.jpg")) or any(candidate.glob("*.jpeg"))):
                img_dir = candidate
                break
        
        if img_dir is None:
            raise FileNotFoundError("Image directory not found")
        
        # Find CSV file
        csv_file = None
        for candidate in [self.root / "train.csv", self.root / "metadata.csv"]:
            if candidate.exists():
                csv_file = candidate
                break
        
        if csv_file is None:
            csv_files = list(self.root.rglob("*.csv"))
            if csv_files:
                csv_file = csv_files[0]
        
        if csv_file is None or not csv_file.exists():
            raise FileNotFoundError("CSV metadata file not found")
        
        # Load metadata
        df = pd.read_csv(csv_file)
        
        # Find image ID and target columns
        id_col = None
        for col in ['image_name', 'image', 'image_id']:
            if col in df.columns:
                id_col = col
                break
        if id_col is None:
            id_col = df.columns[0]
        
        target_col = None
        for col in ['target', 'label', 'diagnosis']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError("Could not find target column in CSV")
        
        # Build all samples
        all_samples = []
        for _, row in df.iterrows():
            image_id = row[id_col]
            img_path = img_dir / f"{image_id}.jpg"
            if not img_path.exists():
                img_path = img_dir / f"{image_id}.jpeg"
            if not img_path.exists():
                continue
            
            label = int(row[target_col])
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
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for weighted random sampling."""
        class_weights = self.get_class_weights()
        return torch.tensor([class_weights[label].item() for _, label in self.samples])
