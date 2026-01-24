"""
Dataset Classes for Dermoscopy Image Classification.

Implements dataset loaders for:
- Client 1: HAM10000
- Client 2: ISIC 2018
- Client 3: ISIC 2019
- Client 4: ISIC 2020

Each dataset is assigned to a different FL client to create
a realistic non-IID federated learning scenario.
"""

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import torch


# Class mappings for different datasets
HAM10000_CLASSES = {
    'akiec': 0,  # Actinic keratoses
    'bcc': 1,    # Basal cell carcinoma
    'bkl': 2,    # Benign keratosis
    'df': 3,     # Dermatofibroma
    'mel': 4,    # Melanoma
    'nv': 5,     # Melanocytic nevi
    'vasc': 6    # Vascular lesions
}

ISIC2018_CLASSES = HAM10000_CLASSES  # Same 7 classes

ISIC2019_CLASSES = {
    'AK': 0,     # Actinic keratosis
    'BCC': 1,    # Basal cell carcinoma
    'BKL': 2,    # Benign keratosis
    'DF': 3,     # Dermatofibroma
    'MEL': 4,    # Melanoma
    'NV': 5,     # Melanocytic nevus
    'SCC': 6,    # Squamous cell carcinoma (NEW)
    'VASC': 7,   # Vascular lesion
}

ISIC2020_CLASSES = {
    'benign': 0,
    'malignant': 1
}

# Unified class mapping (7 classes for compatibility)
UNIFIED_CLASSES = {
    'akiec': 0, 'AK': 0,
    'bcc': 1, 'BCC': 1,
    'bkl': 2, 'BKL': 2,
    'df': 3, 'DF': 3,
    'mel': 4, 'MEL': 4, 'malignant': 4,
    'nv': 5, 'NV': 5, 'benign': 5,
    'vasc': 6, 'VASC': 6,
    'SCC': 1,  # Map SCC to BCC (both carcinomas)
}

CLASS_NAMES = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevus',
    'Vascular Lesion'
]


class BaseDermoscopyDataset(Dataset):
    """
    Base class for dermoscopy datasets.
    
    Provides common functionality for loading and transforming
    dermoscopy images across different datasets.
    
    Args:
        root_dir: Root directory containing images
        csv_path: Path to metadata CSV file
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to labels
        use_unified_classes: Map to unified 7-class scheme
    """
    
    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_unified_classes: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.target_transform = target_transform
        self.use_unified_classes = use_unified_classes
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Build image list
        self.image_paths, self.labels = self._build_image_list()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load and preprocess metadata CSV."""
        raise NotImplementedError
        
    def _build_image_list(self) -> Tuple[List[str], List[int]]:
        """Build list of image paths and labels."""
        raise NotImplementedError
    
    def _map_label(self, label: str) -> int:
        """Map string label to integer class."""
        if self.use_unified_classes:
            return UNIFIED_CLASSES.get(label, -1)
        raise NotImplementedError
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        # Ensure return type is a torch.Tensor in CHW float format
        if not isinstance(image, torch.Tensor):
            # NumPy HWC -> Tensor CHW, scale to [0,1]
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
        else:
            # If tensor is HWC (last dim channels), convert to CHW
            if image.ndim == 3 and image.shape[-1] in (1, 3):
                image = image.permute(2, 0, 1).contiguous()
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes in the dataset."""
        from collections import Counter
        return dict(Counter(self.labels))
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced dataset handling."""
        dist = self.get_class_distribution()
        total = sum(dist.values())
        num_classes = len(set(self.labels))
        
        weights = torch.zeros(num_classes)
        for cls, count in dist.items():
            if cls >= 0 and cls < num_classes:
                weights[cls] = total / (num_classes * count)
        
        return weights


class HAM10000Dataset(BaseDermoscopyDataset):
    """
    HAM10000 Dataset (Human Against Machine with 10000 training images).
    
    7 diagnostic categories:
    - akiec: Actinic keratoses and intraepithelial carcinoma
    - bcc: Basal cell carcinoma
    - bkl: Benign keratosis-like lesions
    - df: Dermatofibroma
    - mel: Melanoma
    - nv: Melanocytic nevi
    - vasc: Vascular lesions
    
    Reference: Tschandl et al., 2018
    """
    
    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        return df
    
    def _build_image_list(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []
        
        for _, row in self.metadata.iterrows():
            # HAM10000 has images in multiple folders
            img_id = row['image_id']
            label_str = row['dx']
            
            # Try different possible paths
            for subdir in ['HAM10000_images_part_1', 'HAM10000_images_part_2', 'images']:
                img_path = self.root_dir / subdir / f"{img_id}.jpg"
                if img_path.exists():
                    break
            else:
                img_path = self.root_dir / f"{img_id}.jpg"
            
            if img_path.exists():
                image_paths.append(str(img_path))
                labels.append(self._map_label(label_str))
        
        return image_paths, labels
    
    def _map_label(self, label: str) -> int:
        if self.use_unified_classes:
            return UNIFIED_CLASSES.get(label, -1)
        return HAM10000_CLASSES.get(label, -1)


class ISIC2018Dataset(BaseDermoscopyDataset):
    """
    ISIC 2018 Challenge Dataset (Task 3: Lesion Diagnosis).
    
    Same 7 diagnostic categories as HAM10000.
    """
    
    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        return df
    
    def _build_image_list(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []
        
        # ISIC 2018 Task 3 ground truth format
        # Columns: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC (one-hot)
        label_cols = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        label_map = {
            'MEL': 'mel', 'NV': 'nv', 'BCC': 'bcc', 
            'AKIEC': 'akiec', 'BKL': 'bkl', 'DF': 'df', 'VASC': 'vasc'
        }
        
        for _, row in self.metadata.iterrows():
            img_id = row['image']
            img_path = self.root_dir / f"{img_id}.jpg"
            
            if img_path.exists():
                # Find which column is 1 (one-hot encoded)
                label_str = None
                for col in label_cols:
                    if col in row and row[col] == 1:
                        label_str = label_map[col]
                        break
                
                if label_str:
                    image_paths.append(str(img_path))
                    labels.append(self._map_label(label_str))
        
        return image_paths, labels


class ISIC2019Dataset(BaseDermoscopyDataset):
    """
    ISIC 2019 Challenge Dataset.
    
    8 diagnostic categories (adds SCC compared to ISIC 2018):
    - AK, BCC, BKL, DF, MEL, NV, SCC, VASC
    """
    
    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        return df
    
    def _build_image_list(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []
        
        # ISIC 2019 ground truth format
        label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
        
        for _, row in self.metadata.iterrows():
            img_id = row['image']
            img_path = self.root_dir / f"{img_id}.jpg"
            
            if img_path.exists():
                # Find which column is 1
                label_str = None
                for col in label_cols:
                    if col in row and row[col] == 1.0:
                        label_str = col
                        break
                
                if label_str:
                    image_paths.append(str(img_path))
                    labels.append(self._map_label(label_str))
        
        return image_paths, labels


class ISIC2020Dataset(BaseDermoscopyDataset):
    """
    ISIC 2020 Challenge Dataset.
    
    Binary classification: benign vs malignant
    For unified 7-class: benign -> nv, malignant -> mel
    """
    
    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        return df
    
    def _build_image_list(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []
        
        for _, row in self.metadata.iterrows():
            img_id = row['image_name']
            target = row['target']  # 0 = benign, 1 = malignant
            
            img_path = self.root_dir / f"{img_id}.jpg"
            
            if img_path.exists():
                label_str = 'malignant' if target == 1 else 'benign'
                image_paths.append(str(img_path))
                labels.append(self._map_label(label_str))
        
        return image_paths, labels


def get_client_dataloader(
    client_id: int,
    data_root: Union[str, Path],
    batch_size: int = 32,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation DataLoaders for a specific FL client.
    
    Args:
        client_id: Client identifier (1-4)
        data_root: Root directory for all datasets
        batch_size: Batch size for DataLoader
        train_transform: Transform for training data
        val_transform: Transform for validation data
        val_split: Fraction for validation split
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from .preprocessing import get_train_transforms, get_val_transforms
    from .splits import train_val_split
    
    data_root = Path(data_root)
    
    # Default transforms if not provided
    if train_transform is None:
        train_transform = get_train_transforms()
    if val_transform is None:
        val_transform = get_val_transforms()
    
    # Dataset paths based on client ID
    dataset_configs = {
        1: {
            'class': HAM10000Dataset,
            'root': data_root / 'HAM10000',
            'csv': data_root / 'HAM10000' / 'HAM10000_metadata.csv'
        },
        2: {
            'class': ISIC2018Dataset,
            'root': data_root / 'ISIC2018' / 'ISIC2018_Task3_Training_Input',
            'csv': data_root / 'ISIC2018' / 'ISIC2018_Task3_Training_GroundTruth.csv'
        },
        3: {
            'class': ISIC2019Dataset,
            'root': data_root / 'ISIC2019' / 'ISIC_2019_Training_Input',
            'csv': data_root / 'ISIC2019' / 'ISIC_2019_Training_GroundTruth.csv'
        },
        4: {
            'class': ISIC2020Dataset,
            # use the ISIC2020 folder as base; select specific image subdir later
            'root': data_root / 'ISIC2020',
            'csv': data_root / 'ISIC2020' / 'train.csv'
        }
    }
    
    if client_id not in dataset_configs:
        raise ValueError(f"Invalid client_id: {client_id}. Must be 1-4.")
    
    config = dataset_configs[client_id]

    # Accept alternative ISIC2020 ground-truth filename if `train.csv` is not present
    if client_id == 4:
        t1 = data_root / 'ISIC2020' / 'train.csv'
        t2 = data_root / 'ISIC2020' / 'ISIC_2020_Training_GroundTruth.csv'
        # prefer existing CSV file; fall back to the other candidate
        dataset_configs[4]['csv'] = t1 if t1.exists() else t2

        # Detect common image-folder names for ISIC2020
        possible_image_dirs = [
            data_root / 'ISIC2020' / 'train',
            data_root / 'ISIC2020' / 'ISIC_2020_Training_JPEG',
            data_root / 'ISIC2020'
        ]
        selected_root = next((p for p in possible_image_dirs if p.exists()), data_root / 'ISIC2020')
        dataset_configs[4]['root'] = selected_root
        config = dataset_configs[4]
    
    # Create full dataset with training transform initially
    full_dataset = config['class'](
        root_dir=str(config['root']),
        csv_path=str(config['csv']),
        transform=train_transform
    )
    
    # Split into train/val
    train_indices, val_indices = train_val_split(
        len(full_dataset),
        val_split=val_split,
        seed=seed
    )
    
    # Create train and val datasets
    train_dataset = DatasetSubset(full_dataset, train_indices, train_transform)
    val_dataset = DatasetSubset(full_dataset, val_indices, val_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


class DatasetSubset(Dataset):
    """Subset of a dataset with separate transform.

    Accepts a `BaseDermoscopyDataset` so attribute access (e.g. `image_paths`)
    is recognized by static type checkers like Pylance.
    """
    
    def __init__(self, dataset: "BaseDermoscopyDataset", indices: List[int], transform: Optional[Callable] = None):
        self.dataset: BaseDermoscopyDataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get original item
        real_idx = self.indices[idx]
        
        # Access the base dataset's image path and label directly
        img_path = self.dataset.image_paths[real_idx]
        label = self.dataset.labels[real_idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        # Ensure torch.Tensor CHW float
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
        else:
            if image.ndim == 3 and image.shape[-1] in (1, 3):
                image = image.permute(2, 0, 1).contiguous()

        return image, label
