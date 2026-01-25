"""
Dataset Classes for Dermoscopy Image Classification.

Implements dataset loaders for:
- Client 1: HAM10000 (7 classes)
- Client 2: ISIC 2018 (7 classes)
- Client 3: ISIC 2019 (8 classes + UNK)
- Client 4: ISIC 2020 (binary: benign/malignant, with diagnosis info)

Each dataset is assigned to a different FL client to create
a realistic non-IID federated learning scenario.

Classification Modes:
1. MULTICLASS (7 classes) - Unified across HAM10000/ISIC2018/2019
2. BINARY - Benign vs Malignant (used for ISIC2020 compatibility)
"""

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple, Union, Literal
from torch.utils.data import Dataset, DataLoader
import torch


# ============================================================================
# CLASS MAPPINGS FOR DIFFERENT DATASETS
# ============================================================================

# HAM10000: 7 diagnostic categories
HAM10000_CLASSES = {
    'akiec': 0,  # Actinic keratoses and intraepithelial carcinoma
    'bcc': 1,    # Basal cell carcinoma
    'bkl': 2,    # Benign keratosis-like lesions
    'df': 3,     # Dermatofibroma
    'mel': 4,    # Melanoma
    'nv': 5,     # Melanocytic nevi
    'vasc': 6    # Vascular lesions
}

# ISIC2018: Same 7 classes as HAM10000 (uses AKIEC not AK)
ISIC2018_CLASSES = {
    'MEL': 4,    # Melanoma
    'NV': 5,     # Melanocytic nevus
    'BCC': 1,    # Basal cell carcinoma
    'AKIEC': 0,  # Actinic keratosis (note: AKIEC not AK)
    'BKL': 2,    # Benign keratosis
    'DF': 3,     # Dermatofibroma
    'VASC': 6    # Vascular lesion
}

# ISIC2019: 8 classes + UNK (adds SCC, uses AK not AKIEC)
ISIC2019_CLASSES = {
    'MEL': 4,    # Melanoma
    'NV': 5,     # Melanocytic nevus
    'BCC': 1,    # Basal cell carcinoma
    'AK': 0,     # Actinic keratosis
    'BKL': 2,    # Benign keratosis
    'DF': 3,     # Dermatofibroma
    'VASC': 6,   # Vascular lesion
    'SCC': 7,    # Squamous cell carcinoma (NEW in 2019)
    'UNK': -1,   # Unknown (to be filtered or handled specially)
}

# ISIC2020: Binary classification with rich diagnosis metadata
# Primary: benign (0) vs malignant (1)
# Diagnosis field has: nevus, melanoma, seborrheic keratosis, etc.
ISIC2020_BINARY_CLASSES = {
    'benign': 0,
    'malignant': 1
}

# Mapping ISIC2020 diagnosis to unified 7-class (when not using binary mode)
ISIC2020_DIAGNOSIS_TO_UNIFIED = {
    'nevus': 5,                           # NV - Melanocytic nevi
    'melanoma': 4,                        # MEL - Melanoma
    'seborrheic keratosis': 2,            # BKL - Benign keratosis
    'lentigo NOS': 2,                     # BKL - Benign keratosis (lentigo)
    'lichenoid keratosis': 2,             # BKL - Benign keratosis
    'solar lentigo': 2,                   # BKL - Benign keratosis
    'cafe-au-lait macule': 5,             # NV - Benign pigmented lesion
    'atypical melanocytic proliferation': 4,  # MEL - Potential melanoma
    'unknown': -1,                        # Unknown - needs special handling
}

# PAD-UFES-20: 6 classes (Brazilian clinical images)
# Classes: BCC, SCC, ACK (Actinic Keratosis), SEK (Seborrheic Keratosis),
#          MEL (Melanoma), NEV (Nevus)
# Note: Uses 3-letter abbreviations different from ISIC
PADUFES20_CLASSES = {
    'BCC': 1,    # Basal cell carcinoma
    'SCC': 7,    # Squamous cell carcinoma (maps to class 7 in 8-class, 1 in 7-class)
    'ACK': 0,    # Actinic keratosis
    'SEK': 2,    # Seborrheic keratosis -> maps to BKL (benign keratosis)
    'MEL': 4,    # Melanoma
    'NEV': 5,    # Nevus
}

# ============================================================================
# UNIFIED CLASS MAPPINGS
# ============================================================================

# Unified 7-class mapping for multiclass classification
# Maps all dataset-specific labels to common indices
UNIFIED_CLASSES_7 = {
    # HAM10000 labels (lowercase)
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6,
    # ISIC2018 labels (uppercase, uses AKIEC)
    'AKIEC': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'VASC': 6,
    # ISIC2019 labels (uppercase, uses AK)
    'AK': 0,
    # ISIC2019 SCC -> mapped to BCC (both are carcinomas)
    'SCC': 1,
    # PAD-UFES-20 labels
    'ACK': 0,  # Actinic keratosis -> same as AK/AKIEC
    'SEK': 2,  # Seborrheic keratosis -> maps to BKL
    'NEV': 5,  # Nevus -> same as NV
    # Unknown handling
    'UNK': -1,
    'unknown': -1,
}

# Unified binary mapping (benign=0, malignant=1)
UNIFIED_CLASSES_BINARY = {
    # Malignant classes
    'mel': 1, 'MEL': 1, 'melanoma': 1, 'malignant': 1,
    'bcc': 1, 'BCC': 1,           # Basal cell carcinoma
    'akiec': 0, 'AKIEC': 0, 'AK': 0,  # Actinic keratosis (pre-cancerous, often benign)
    'SCC': 1,                      # Squamous cell carcinoma
    # Benign classes
    'nv': 0, 'NV': 0, 'nevus': 0, 'benign': 0,
    'bkl': 0, 'BKL': 0, 'seborrheic keratosis': 0,
    'df': 0, 'DF': 0,
    'vasc': 0, 'VASC': 0,
    'lentigo NOS': 0,
    'lichenoid keratosis': 0,
    'solar lentigo': 0,
    'cafe-au-lait macule': 0,
    'atypical melanocytic proliferation': 1,  # Potentially malignant
    # PAD-UFES-20 labels
    'ACK': 0,  # Actinic keratosis (pre-cancerous)
    'SEK': 0,  # Seborrheic keratosis (benign)
    'NEV': 0,  # Nevus (benign)
    # Unknown
    'UNK': -1, 'unknown': -1,
}

# Class names for 7-class
CLASS_NAMES_7 = [
    'Actinic Keratosis',      # 0
    'Basal Cell Carcinoma',   # 1
    'Benign Keratosis',       # 2
    'Dermatofibroma',         # 3
    'Melanoma',               # 4
    'Melanocytic Nevus',      # 5
    'Vascular Lesion'         # 6
]

# Class names for 8-class (includes SCC)
CLASS_NAMES_8 = CLASS_NAMES_7 + ['Squamous Cell Carcinoma']  # 7

# Class names for binary
CLASS_NAMES_BINARY = ['Benign', 'Malignant']

# Legacy aliases for backward compatibility
UNIFIED_CLASSES = UNIFIED_CLASSES_7
CLASS_NAMES = CLASS_NAMES_7

# Type for classification mode
ClassificationMode = Literal['multiclass', 'multiclass_8', 'binary']


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
        classification_mode: 'multiclass' (7), 'multiclass_8' (8), or 'binary' (2)
        filter_unknown: Whether to filter out unknown/UNK labels
        use_unified_classes: Legacy parameter (ignored, use classification_mode)
    """
    
    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        classification_mode: ClassificationMode = 'multiclass',
        filter_unknown: bool = True,
        use_unified_classes: bool = True  # Legacy, ignored
    ):
        self.root_dir = Path(root_dir)
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.target_transform = target_transform
        self.classification_mode = classification_mode
        self.filter_unknown = filter_unknown
        
        # Determine number of classes based on mode
        if classification_mode == 'binary':
            self.num_classes = 2
            self.class_names = CLASS_NAMES_BINARY
        elif classification_mode == 'multiclass_8':
            self.num_classes = 8
            self.class_names = CLASS_NAMES_8
        else:  # multiclass (default 7)
            self.num_classes = 7
            self.class_names = CLASS_NAMES_7
        
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
        """Map string label to integer class based on classification mode."""
        if self.classification_mode == 'binary':
            return UNIFIED_CLASSES_BINARY.get(label, -1)
        elif self.classification_mode == 'multiclass_8':
            # For 8-class, use ISIC2019 mapping with fallback
            return ISIC2019_CLASSES.get(label, UNIFIED_CLASSES_7.get(label, -1))
        else:  # multiclass (7)
            return UNIFIED_CLASSES_7.get(label, -1)
        
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
        
        weights = torch.zeros(self.num_classes)
        for cls, count in dist.items():
            if 0 <= cls < self.num_classes:
                weights[cls] = total / (self.num_classes * count)
        
        return weights
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for weighted sampling (handles class imbalance)."""
        class_weights = self.get_class_weights()
        sample_weights = torch.tensor([
            class_weights[label].item() if 0 <= label < self.num_classes else 0.0
            for label in self.labels
        ])
        return sample_weights


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
                mapped_label = self._map_label(label_str)
                # Filter unknown labels if requested
                if self.filter_unknown and mapped_label == -1:
                    continue
                image_paths.append(str(img_path))
                labels.append(mapped_label)
        
        return image_paths, labels


class ISIC2018Dataset(BaseDermoscopyDataset):
    """
    ISIC 2018 Challenge Dataset (Task 3: Lesion Diagnosis).
    
    7 diagnostic categories (same as HAM10000):
    - MEL, NV, BCC, AKIEC, BKL, DF, VASC
    
    Note: Uses AKIEC (not AK like ISIC2019).
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
        
        for _, row in self.metadata.iterrows():
            img_id = row['image']
            img_path = self.root_dir / f"{img_id}.jpg"
            
            if img_path.exists():
                # Find which column is 1 (one-hot encoded)
                label_str = None
                for col in label_cols:
                    if col in row and row[col] == 1.0:
                        label_str = col
                        break
                
                if label_str:
                    mapped_label = self._map_label(label_str)
                    if self.filter_unknown and mapped_label == -1:
                        continue
                    image_paths.append(str(img_path))
                    labels.append(mapped_label)
        
        return image_paths, labels


class ISIC2019Dataset(BaseDermoscopyDataset):
    """
    ISIC 2019 Challenge Dataset.
    
    9 categories (8 diagnostic + UNK):
    - MEL: Melanoma
    - NV: Melanocytic nevus  
    - BCC: Basal cell carcinoma
    - AK: Actinic keratosis (Note: AK not AKIEC like 2018)
    - BKL: Benign keratosis
    - DF: Dermatofibroma
    - VASC: Vascular lesion
    - SCC: Squamous cell carcinoma (NEW in 2019)
    - UNK: Unknown (none in training set, but supported)
    """
    
    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        return df
    
    def _build_image_list(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []
        
        # ISIC 2019 ground truth format (includes UNK)
        label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        
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
                    mapped_label = self._map_label(label_str)
                    # Filter unknown labels if requested
                    if self.filter_unknown and mapped_label == -1:
                        continue
                    image_paths.append(str(img_path))
                    labels.append(mapped_label)
        
        return image_paths, labels


class ISIC2020Dataset(BaseDermoscopyDataset):
    """
    ISIC 2020 Challenge Dataset.
    
    Binary classification: benign (0) vs malignant (1)
    
    The 'diagnosis' column contains specific diagnoses:
    - nevus, melanoma, seborrheic keratosis, lentigo NOS,
      lichenoid keratosis, solar lentigo, cafe-au-lait macule,
      atypical melanocytic proliferation, unknown
    
    In multiclass mode, we use the diagnosis field for richer labels.
    In binary mode, we use the target field directly.
    """
    
    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        return df
    
    def _build_image_list(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []
        
        for _, row in self.metadata.iterrows():
            img_id = row['image_name']
            
            img_path = self.root_dir / f"{img_id}.jpg"
            
            if img_path.exists():
                # Choose label based on classification mode
                if self.classification_mode == 'binary':
                    # Direct binary: use target column
                    label = int(row['target'])
                    image_paths.append(str(img_path))
                    labels.append(label)
                else:
                    # Multiclass: use diagnosis field for richer mapping
                    diagnosis = row.get('diagnosis', 'unknown')
                    if pd.isna(diagnosis):
                        diagnosis = 'unknown'
                    
                    # Use diagnosis-to-unified mapping for better granularity
                    mapped_label = ISIC2020_DIAGNOSIS_TO_UNIFIED.get(
                        diagnosis, 
                        self._map_label(row['benign_malignant'])
                    )
                    
                    # Filter unknown labels if requested
                    if self.filter_unknown and mapped_label == -1:
                        continue
                    
                    image_paths.append(str(img_path))
                    labels.append(mapped_label)
        
        return image_paths, labels


class PADUFES20Dataset(BaseDermoscopyDataset):
    """
    PAD-UFES-20 Dataset (Brazilian clinical skin lesion images).
    
    Collected from the Dermatological and Surgical Assistance Program (PAD)
    at the Federal University of Espírito Santo (UFES), Brazil.
    
    6 diagnostic categories:
    - BCC: Basal Cell Carcinoma
    - SCC: Squamous Cell Carcinoma (includes Bowen's disease)
    - ACK: Actinic Keratosis
    - SEK: Seborrheic Keratosis
    - MEL: Melanoma
    - NEV: Nevus
    
    Dataset characteristics:
    - 2,298 clinical images (smartphone-acquired, varying sizes)
    - 1,373 patients, 1,641 skin lesions
    - Images split across: imgs_part_1, imgs_part_2, imgs_part_3
    - ~58% biopsy-proven samples
    - Includes rich metadata (age, gender, Fitzpatrick type, etc.)
    
    Reference: Pacheco et al., 2020
    """
    
    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        return df
    
    def _build_image_list(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []
        
        for _, row in self.metadata.iterrows():
            img_id = row['img_id']  # e.g., PAT_1516_1765_530.png
            label_str = row['diagnostic']  # BCC, SCC, ACK, SEK, MEL, NEV
            
            # Images are split across 3 folders
            img_path = None
            for part_dir in ['imgs_part_1', 'imgs_part_2', 'imgs_part_3']:
                candidate = self.root_dir / part_dir / img_id
                if candidate.exists():
                    img_path = candidate
                    break
            
            # Also check root directory directly
            if img_path is None:
                candidate = self.root_dir / img_id
                if candidate.exists():
                    img_path = candidate
            
            if img_path is not None and img_path.exists():
                mapped_label = self._map_label(label_str)
                
                # Filter unknown labels if requested
                if self.filter_unknown and mapped_label == -1:
                    continue
                    
                image_paths.append(str(img_path))
                labels.append(mapped_label)
        
        return image_paths, labels


def get_client_dataloader(
    client_id: int,
    data_root: Union[str, Path],
    batch_size: int = 32,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    classification_mode: ClassificationMode = 'multiclass',
    filter_unknown: bool = True,
    use_weighted_sampling: bool = False
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
        classification_mode: 'multiclass' (7), 'multiclass_8' (8), or 'binary'
        filter_unknown: Whether to filter out unknown/UNK labels
        use_weighted_sampling: Use weighted sampler for class imbalance
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import WeightedRandomSampler
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
        },
        5: {
            'class': PADUFES20Dataset,
            'root': data_root / 'PAD-UFES-20',
            'csv': data_root / 'PAD-UFES-20' / 'metadata.csv'
        }
    }
    
    if client_id not in dataset_configs:
        raise ValueError(f"Invalid client_id: {client_id}. Must be 1-5.")
    
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
        transform=train_transform,
        classification_mode=classification_mode,
        filter_unknown=filter_unknown
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
    
    # Setup weighted sampling for class imbalance (especially for ISIC2020)
    train_sampler = None
    shuffle_train = True
    if use_weighted_sampling:
        # Get sample weights for the training subset
        all_sample_weights = full_dataset.get_sample_weights()
        train_sample_weights = all_sample_weights[train_indices]
        train_sampler = WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=len(train_sample_weights),
            replacement=True
        )
        shuffle_train = False  # Can't use shuffle with sampler
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
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
        # Expose num_classes and class_names from parent
        self.num_classes = dataset.num_classes
        self.class_names = dataset.class_names
        
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
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes in this subset."""
        from collections import Counter
        subset_labels = [self.dataset.labels[i] for i in self.indices]
        return dict(Counter(subset_labels))


def get_combined_dataset(
    data_root: Union[str, Path],
    datasets: List[str] = ['HAM10000', 'ISIC2018', 'ISIC2019', 'ISIC2020'],
    transform: Optional[Callable] = None,
    classification_mode: ClassificationMode = 'multiclass',
    filter_unknown: bool = True
) -> Tuple[Dataset, Dict[str, int]]:
    """
    Create a combined dataset from multiple sources.
    
    Args:
        data_root: Root directory for all datasets
        datasets: List of dataset names to combine
        transform: Transform to apply
        classification_mode: Classification mode for all datasets
        filter_unknown: Whether to filter unknown labels
        
    Returns:
        Tuple of (combined_dataset, dataset_sizes)
    """
    from torch.utils.data import ConcatDataset
    
    data_root = Path(data_root)
    
    dataset_configs = {
        'HAM10000': {
            'class': HAM10000Dataset,
            'root': data_root / 'HAM10000',
            'csv': data_root / 'HAM10000' / 'HAM10000_metadata.csv'
        },
        'ISIC2018': {
            'class': ISIC2018Dataset,
            'root': data_root / 'ISIC2018' / 'ISIC2018_Task3_Training_Input',
            'csv': data_root / 'ISIC2018' / 'ISIC2018_Task3_Training_GroundTruth.csv'
        },
        'ISIC2019': {
            'class': ISIC2019Dataset,
            'root': data_root / 'ISIC2019' / 'ISIC_2019_Training_Input',
            'csv': data_root / 'ISIC2019' / 'ISIC_2019_Training_GroundTruth.csv'
        },
        'ISIC2020': {
            'class': ISIC2020Dataset,
            'root': data_root / 'ISIC2020' / 'ISIC_2020_Training_JPEG',
            'csv': data_root / 'ISIC2020' / 'ISIC_2020_Training_GroundTruth.csv'
        },
        'PAD-UFES-20': {
            'class': PADUFES20Dataset,
            'root': data_root / 'PAD-UFES-20',
            'csv': data_root / 'PAD-UFES-20' / 'metadata.csv'
        }
    }
    
    loaded_datasets = []
    dataset_sizes = {}
    
    for name in datasets:
        if name not in dataset_configs:
            print(f"Warning: Unknown dataset {name}, skipping.")
            continue
            
        config = dataset_configs[name]
        
        # Check for alternative paths
        if name == 'ISIC2020':
            if not config['csv'].exists():
                alt_csv = data_root / 'ISIC2020' / 'train.csv'
                if alt_csv.exists():
                    config['csv'] = alt_csv
            if not config['root'].exists():
                alt_root = data_root / 'ISIC2020' / 'train'
                if alt_root.exists():
                    config['root'] = alt_root
        
        if not config['csv'].exists():
            print(f"Warning: CSV not found for {name} at {config['csv']}, skipping.")
            continue
            
        ds = config['class'](
            root_dir=str(config['root']),
            csv_path=str(config['csv']),
            transform=transform,
            classification_mode=classification_mode,
            filter_unknown=filter_unknown
        )
        
        dataset_sizes[name] = len(ds)
        loaded_datasets.append(ds)
        print(f"Loaded {name}: {len(ds)} images")
    
    combined = ConcatDataset(loaded_datasets)
    return combined, dataset_sizes


def print_dataset_statistics(
    data_root: Union[str, Path],
    classification_mode: ClassificationMode = 'multiclass'
):
    """Print statistics for all datasets."""
    data_root = Path(data_root)
    
    print(f"\n{'='*60}")
    print(f"Dataset Statistics (mode: {classification_mode})")
    print(f"{'='*60}\n")
    
    dataset_configs = {
        'HAM10000': (HAM10000Dataset, 'HAM10000', 'HAM10000_metadata.csv'),
        'ISIC2018': (ISIC2018Dataset, 'ISIC2018/ISIC2018_Task3_Training_Input', 
                     'ISIC2018/ISIC2018_Task3_Training_GroundTruth.csv'),
        'ISIC2019': (ISIC2019Dataset, 'ISIC2019/ISIC_2019_Training_Input',
                     'ISIC2019/ISIC_2019_Training_GroundTruth.csv'),
        'ISIC2020': (ISIC2020Dataset, 'ISIC2020/ISIC_2020_Training_JPEG',
                     'ISIC2020/ISIC_2020_Training_GroundTruth.csv'),
        'PAD-UFES-20': (PADUFES20Dataset, 'PAD-UFES-20', 'PAD-UFES-20/metadata.csv'),
    }
    
    class_names = (CLASS_NAMES_BINARY if classification_mode == 'binary' 
                   else CLASS_NAMES_8 if classification_mode == 'multiclass_8'
                   else CLASS_NAMES_7)
    
    total_samples = 0
    total_dist = {}
    
    for name, (cls, root_suffix, csv_suffix) in dataset_configs.items():
        root = data_root / root_suffix
        csv = data_root / csv_suffix
        
        # Handle ISIC2020 alternatives
        if name == 'ISIC2020':
            if not csv.exists():
                csv = data_root / 'ISIC2020' / 'train.csv'
            if not root.exists():
                root = data_root / 'ISIC2020' / 'train'
        
        if not csv.exists():
            print(f"{name}: CSV not found at {csv}")
            continue
            
        try:
            ds = cls(
                root_dir=str(root),
                csv_path=str(csv),
                classification_mode=classification_mode,
                filter_unknown=True
            )
            
            dist = ds.get_class_distribution()
            total_samples += len(ds)
            
            print(f"\n{name}:")
            print(f"  Total samples: {len(ds)}")
            print(f"  Class distribution:")
            
            for idx, count in sorted(dist.items()):
                if 0 <= idx < len(class_names):
                    pct = 100 * count / len(ds)
                    print(f"    {idx}: {class_names[idx]}: {count} ({pct:.1f}%)")
                    total_dist[idx] = total_dist.get(idx, 0) + count
                    
        except Exception as e:
            print(f"{name}: Error loading - {e}")
    
    print(f"\n{'='*60}")
    print(f"Combined Statistics (Total: {total_samples} samples)")
    print(f"{'='*60}")
    for idx, count in sorted(total_dist.items()):
        if 0 <= idx < len(class_names):
            pct = 100 * count / total_samples if total_samples > 0 else 0
            print(f"  {idx}: {class_names[idx]}: {count} ({pct:.1f}%)")
