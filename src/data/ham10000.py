"""
HAM10000 Dataset Loader
=======================

Human Against Machine with 10000 training images dataset.
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Dataset Statistics:
    - Total images: 10,015
    - Image size: 600x450 (variable)
    - Classes: 7 skin lesion types

Classes:
    - akiec: Actinic keratoses and intraepithelial carcinoma (327 images)
    - bcc: Basal cell carcinoma (514 images)
    - bkl: Benign keratosis-like lesions (1099 images)
    - df: Dermatofibroma (115 images)
    - mel: Melanoma (1113 images)
    - nv: Melanocytic nevi (6705 images)
    - vasc: Vascular lesions (142 images)

Note: Dataset is highly imbalanced with 'nv' being the majority class.
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


class HAM10000Dataset(Dataset):
    """
    HAM10000 Dataset for skin lesion classification.
    
    Expected directory structure:
        root/
        ├── HAM10000_metadata.csv
        ├── HAM10000_images_part_1/
        │   └── *.jpg
        └── HAM10000_images_part_2/
            └── *.jpg
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to targets
        train_ratio: Ratio of data to use for training (default: 0.7)
        val_ratio: Ratio of data to use for validation (default: 0.15)
        seed: Random seed for reproducible splits
    """
    
    # Class names and their indices
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
    NUM_CLASSES = 7
    
    # Full class names for display
    CLASS_FULL_NAMES = {
        'akiec': 'Actinic Keratoses',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevi',
        'vasc': 'Vascular Lesions',
    }
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
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
        
        # Will be populated by _load_metadata
        self.samples: List[Tuple[Path, int]] = []
        self.metadata: Optional[pd.DataFrame] = None
        
        # Load and split dataset
        self._load_metadata()
    
    def _find_image_path(self, image_id: str) -> Optional[Path]:
        """
        Find the path to an image given its ID.
        
        Images may be in either HAM10000_images_part_1 or HAM10000_images_part_2.
        
        Args:
            image_id: Image ID without extension
            
        Returns:
            Path to image file or None if not found
        """
        possible_dirs = [
            self.root / "HAM10000_images_part_1",
            self.root / "HAM10000_images_part_2",
            self.root / "images",  # Alternative structure
            self.root,  # Images directly in root
        ]
        
        for img_dir in possible_dirs:
            img_path = img_dir / f"{image_id}.jpg"
            if img_path.exists():
                return img_path
        
        return None
    
    def _load_metadata(self) -> None:
        """
        Load and process dataset metadata.
        
        Creates train/val/test splits using stratified sampling
        to maintain class distribution across splits.
        """
        metadata_path = self.root / "HAM10000_metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Please download the HAM10000 dataset and ensure "
                "HAM10000_metadata.csv is in the root directory."
            )
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        
        # Validate required columns
        required_cols = ['image_id', 'dx']
        for col in required_cols:
            if col not in self.metadata.columns:
                raise ValueError(f"Required column '{col}' not found in metadata")
        
        # Create stratified splits based on diagnosis
        np.random.seed(self.seed)
        
        # Group by lesion_id to prevent data leakage (same lesion in train and test)
        if 'lesion_id' in self.metadata.columns:
            # Get unique lesions with their diagnosis
            lesion_groups = self.metadata.groupby('lesion_id').first().reset_index()
            unique_lesions = lesion_groups['lesion_id'].values
            diagnoses = lesion_groups['dx'].values
        else:
            # Fallback: treat each image as independent
            unique_lesions = self.metadata['image_id'].values
            diagnoses = self.metadata['dx'].values
        
        # Stratified split
        train_lesions, val_lesions, test_lesions = self._stratified_split(
            unique_lesions, diagnoses
        )
        
        # Select lesions for current split
        if self.split == "train":
            selected_lesions = set(train_lesions)
        elif self.split == "val":
            selected_lesions = set(val_lesions)
        elif self.split == "test":
            selected_lesions = set(test_lesions)
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'train', 'val', or 'test'")
        
        # Build samples list
        self.samples = []
        
        for _, row in self.metadata.iterrows():
            # Check if this sample belongs to current split
            id_col = 'lesion_id' if 'lesion_id' in self.metadata.columns else 'image_id'
            if row[id_col] not in selected_lesions:
                continue
            
            # Find image path
            img_path = self._find_image_path(row['image_id'])
            if img_path is None:
                continue  # Skip missing images
            
            # Get class label
            label = self.CLASS_TO_IDX[row['dx']]
            
            self.samples.append((img_path, label))
        
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split '{self.split}'. "
                "Please check that images exist in the expected directories."
            )
    
    def _stratified_split(
        self,
        ids: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train/val/test splits.
        
        Args:
            ids: Array of sample IDs
            labels: Array of labels corresponding to IDs
            
        Returns:
            Tuple of (train_ids, val_ids, test_ids)
        """
        train_ids, val_ids, test_ids = [], [], []
        
        # Split each class separately to maintain distribution
        for label in np.unique(labels):
            class_mask = labels == label
            class_ids = ids[class_mask]
            
            # Shuffle
            np.random.shuffle(class_ids)
            
            # Calculate split indices
            n = len(class_ids)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            # Split
            train_ids.extend(class_ids[:n_train])
            val_ids.extend(class_ids[n_train:n_train + n_val])
            test_ids.extend(class_ids[n_train + n_val:])
        
        return np.array(train_ids), np.array(val_ids), np.array(test_ids)
    
    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.
        
        Uses inverse frequency weighting: weight = total / (num_classes * count)
        
        Returns:
            Tensor of class weights of shape (num_classes,)
        """
        # Count samples per class
        labels = [label for _, label in self.samples]
        counts = Counter(labels)
        
        # Calculate weights
        total = len(labels)
        weights = torch.zeros(self.NUM_CLASSES)
        
        for cls_idx in range(self.NUM_CLASSES):
            count = counts.get(cls_idx, 1)  # Avoid division by zero
            weights[cls_idx] = total / (self.NUM_CLASSES * count)
        
        return weights
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in this split.
        
        Returns:
            Dictionary mapping class names to counts
        """
        labels = [label for _, label in self.samples]
        counts = Counter(labels)
        
        return {
            self.IDX_TO_CLASS[idx]: counts.get(idx, 0)
            for idx in range(self.NUM_CLASSES)
        }
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Get per-sample weights for weighted random sampling.
        
        Useful for creating a WeightedRandomSampler to balance batches.
        
        Returns:
            Tensor of sample weights of shape (num_samples,)
        """
        class_weights = self.get_class_weights()
        sample_weights = torch.tensor([
            class_weights[label].item()
            for _, label in self.samples
        ])
        return sample_weights
