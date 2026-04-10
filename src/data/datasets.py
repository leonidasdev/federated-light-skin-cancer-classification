# =============================================================================
# Dataset Classes for Dermoscopy Image Classification
# =============================================================================
"""
Dataset Classes for Dermoscopy Image Classification.

Implements dataset loaders for:
- HAM10000 (7 classes)
- ISIC 2018 (7 classes)
- ISIC 2019 (8 classes + UNK)
- ISIC 2020 (binary: benign/malignant, with diagnosis info)
- PAD-UFES-20 (6 clinical lesion types)

Each dataset can be assigned to a different FL client to create
a realistic non-IID federated learning scenario.

Classification Modes:
1. MULTICLASS (7 classes) - Unified across HAM10000/ISIC2018/2019
2. BINARY - Benign vs Malignant (used for ISIC2020 compatibility)
"""

# =============================================================================
# Imports
# =============================================================================

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# =============================================================================
# Class Mappings for Different Datasets
# =============================================================================

# ISIC2019: 8 classes + UNK (adds SCC, uses AK not AKIEC)
ISIC2019_CLASSES = {
    "MEL": 4,  # Melanoma
    "NV": 5,  # Melanocytic nevus
    "BCC": 1,  # Basal cell carcinoma
    "AK": 0,  # Actinic keratosis
    "BKL": 2,  # Benign keratosis
    "DF": 3,  # Dermatofibroma
    "VASC": 6,  # Vascular lesion
    "SCC": 7,  # Squamous cell carcinoma (added in ISIC 2019)
    "UNK": -1,  # Unknown (to be filtered or handled specially)
}

# Mapping ISIC2020 diagnosis to unified 7-class (when not using binary mode)
ISIC2020_DIAGNOSIS_TO_UNIFIED = {
    "nevus": 5,  # NV - Melanocytic nevi
    "melanoma": 4,  # MEL - Melanoma
    "seborrheic keratosis": 2,  # BKL - Benign keratosis
    "lentigo NOS": 2,  # BKL - Benign keratosis (lentigo)
    "lichenoid keratosis": 2,  # BKL - Benign keratosis
    "solar lentigo": 2,  # BKL - Benign keratosis
    "cafe-au-lait macule": 5,  # NV - Benign pigmented lesion
    "atypical melanocytic proliferation": 4,  # MEL - Potential melanoma
    "unknown": -1,  # Unknown - needs special handling
}

# =============================================================================
# UNIFIED CLASS MAPPINGS
# =============================================================================

# Unified 7-class mapping for multiclass classification
# Maps all dataset-specific labels to common indices
UNIFIED_CLASSES_7 = {
    # HAM10000 labels (lowercase)
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
    # ISIC2018 labels (uppercase, uses AKIEC)
    "AKIEC": 0,
    "BCC": 1,
    "BKL": 2,
    "DF": 3,
    "MEL": 4,
    "NV": 5,
    "VASC": 6,
    # ISIC2019 labels (uppercase, uses AK)
    "AK": 0,
    # ISIC2019 SCC -> mapped to BCC (both are carcinomas)
    "SCC": 1,
    # PAD-UFES-20 labels
    "ACK": 0,  # Actinic keratosis -> same as AK/AKIEC
    "SEK": 2,  # Seborrheic keratosis -> maps to BKL
    "NEV": 5,  # Nevus -> same as NV
    # Unknown handling
    "UNK": -1,
    "unknown": -1,
}

# Unified binary mapping (benign=0, malignant=1)
UNIFIED_CLASSES_BINARY = {
    # Malignant classes
    "mel": 1,
    "MEL": 1,
    "melanoma": 1,
    "malignant": 1,
    "bcc": 1,
    "BCC": 1,  # Basal cell carcinoma
    "akiec": 0,
    "AKIEC": 0,
    "AK": 0,  # Actinic keratosis (pre-cancerous, often benign)
    "SCC": 1,  # Squamous cell carcinoma
    # Benign classes
    "nv": 0,
    "NV": 0,
    "nevus": 0,
    "benign": 0,
    "bkl": 0,
    "BKL": 0,
    "seborrheic keratosis": 0,
    "df": 0,
    "DF": 0,
    "vasc": 0,
    "VASC": 0,
    "lentigo NOS": 0,
    "lichenoid keratosis": 0,
    "solar lentigo": 0,
    "cafe-au-lait macule": 0,
    "atypical melanocytic proliferation": 1,  # Potentially malignant
    # PAD-UFES-20 labels
    "ACK": 0,  # Actinic keratosis (pre-cancerous)
    "SEK": 0,  # Seborrheic keratosis (benign)
    "NEV": 0,  # Nevus (benign)
    # Unknown
    "UNK": -1,
    "unknown": -1,
}

# Class names for 7-class
CLASS_NAMES_7 = [
    "Actinic Keratosis",  # 0
    "Basal Cell Carcinoma",  # 1
    "Benign Keratosis",  # 2
    "Dermatofibroma",  # 3
    "Melanoma",  # 4
    "Melanocytic Nevus",  # 5
    "Vascular Lesion",  # 6
]

# Class names for 8-class (includes SCC)
CLASS_NAMES_8 = CLASS_NAMES_7 + ["Squamous Cell Carcinoma"]  # 7

# Class names for binary
CLASS_NAMES_BINARY = ["Benign", "Malignant"]

# Aliases for the default (7-class) constants
UNIFIED_CLASSES = UNIFIED_CLASSES_7
CLASS_NAMES = CLASS_NAMES_7

# Type for classification mode
ClassificationMode = Literal["multiclass", "multiclass_8", "binary"]


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
    """

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        classification_mode: ClassificationMode = "multiclass",
        filter_unknown: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.target_transform = target_transform
        self.classification_mode = classification_mode
        self.filter_unknown = filter_unknown

        # Determine number of classes based on mode
        if classification_mode == "binary":
            self.num_classes = 2
            self.class_names = CLASS_NAMES_BINARY
        elif classification_mode == "multiclass_8":
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

    def _build_image_list(self) -> tuple[list[str], list[int]]:
        """Build list of image paths and labels."""
        raise NotImplementedError

    def _map_label(self, label: str) -> int:
        """Map string label to integer class based on classification mode."""
        if self.classification_mode == "binary":
            return UNIFIED_CLASSES_BINARY.get(label, -1)
        if self.classification_mode == "multiclass_8":
            # For 8-class, use ISIC2019 mapping with fallback
            return ISIC2019_CLASSES.get(label, UNIFIED_CLASSES_7.get(label, -1))
        # multiclass (7)
        return UNIFIED_CLASSES_7.get(label, -1)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and return (image, label) for the given index."""
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Get label
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Ensure return type is a torch.Tensor in CHW float format
        if not isinstance(image, torch.Tensor):
            # NumPy HWC -> Tensor CHW, scale to [0,1]
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
        # If tensor is HWC (last dim channels), convert to CHW
        elif image.ndim == 3 and image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1).contiguous()

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_class_distribution(self) -> dict[int, int]:
        """Get distribution of classes in the dataset."""
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

    def _build_image_list(self) -> tuple[list[str], list[int]]:
        image_paths = []
        labels = []

        for _, row in self.metadata.iterrows():
            # HAM10000 has images in multiple folders
            img_id = row["image_id"]
            label_str = row["dx"]

            # Try different possible paths
            for subdir in ["HAM10000_images_part_1", "HAM10000_images_part_2", "images"]:
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

    def _build_image_list(self) -> tuple[list[str], list[int]]:
        image_paths = []
        labels = []

        # ISIC 2018 Task 3 ground truth format
        # Columns: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC (one-hot)
        label_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

        for _, row in self.metadata.iterrows():
            img_id = row["image"]
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
    - SCC: Squamous cell carcinoma (added in ISIC 2019)
    - UNK: Unknown (none in training set, but supported)
    """

    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        return df

    def _build_image_list(self) -> tuple[list[str], list[int]]:
        image_paths = []
        labels = []

        # ISIC 2019 ground truth format (includes UNK)
        label_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

        for _, row in self.metadata.iterrows():
            img_id = row["image"]
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

    def _build_image_list(self) -> tuple[list[str], list[int]]:
        image_paths = []
        labels = []

        for _, row in self.metadata.iterrows():
            img_id = row["image_name"]

            img_path = self.root_dir / f"{img_id}.jpg"

            if img_path.exists():
                # Choose label based on classification mode
                if self.classification_mode == "binary":
                    # Direct binary: use target column
                    label = int(row["target"])
                    image_paths.append(str(img_path))
                    labels.append(label)
                else:
                    # Multiclass: use diagnosis field for richer mapping
                    diagnosis = row.get("diagnosis", "unknown")
                    if pd.isna(diagnosis):
                        diagnosis = "unknown"

                    # Use diagnosis-to-unified mapping for better granularity
                    mapped_label = ISIC2020_DIAGNOSIS_TO_UNIFIED.get(
                        diagnosis, self._map_label(row["benign_malignant"])
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

    def _build_image_list(self) -> tuple[list[str], list[int]]:
        image_paths = []
        labels = []

        for _, row in self.metadata.iterrows():
            img_id = row["img_id"]  # e.g., PAT_1516_1765_530.png
            label_str = row["diagnostic"]  # BCC, SCC, ACK, SEK, MEL, NEV

            # Images are split across 3 folders
            img_path = None
            for part_dir in ["imgs_part_1", "imgs_part_2", "imgs_part_3"]:
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


class DatasetSubset(Dataset):
    """Subset of a dataset with separate transform.

    Accepts a `BaseDermoscopyDataset` so attribute access (e.g. `image_paths`)
    is recognized by static type checkers like Pylance.
    """

    def __init__(self, dataset: "BaseDermoscopyDataset", indices: list[int], transform: Callable | None = None):
        self.dataset: BaseDermoscopyDataset = dataset
        self.indices = indices
        self.transform = transform
        # Expose num_classes and class_names from parent
        self.num_classes = dataset.num_classes
        self.class_names = dataset.class_names

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and return (image, label) for the given index."""
        # Get original item
        real_idx = self.indices[idx]

        # Access the base dataset's image path and label directly
        img_path = self.dataset.image_paths[real_idx]
        label = self.dataset.labels[real_idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Ensure torch.Tensor CHW float
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
        elif image.ndim == 3 and image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1).contiguous()

        return image, label

    def get_class_distribution(self) -> dict[int, int]:
        """Get distribution of classes in this subset."""
        subset_labels = [self.dataset.labels[i] for i in self.indices]
        return dict(Counter(subset_labels))


# =============================================================================
# DATASET REGISTRY - Centralized configuration for all datasets
# =============================================================================


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    dataset_class: type
    csv_filename: str
    image_subdir: str | None = None  # None means images are in root
    alt_csv_filenames: list[str] | None = None  # Alternative CSV names
    alt_image_subdirs: list[str] | None = None  # Alternative image directories


# Forward references for dataset classes (they're defined above)
DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "HAM10000": DatasetConfig(
        dataset_class=HAM10000Dataset,
        csv_filename="HAM10000_metadata.csv",
        image_subdir=None,
    ),
    "ISIC2018": DatasetConfig(
        dataset_class=ISIC2018Dataset,
        csv_filename="ISIC2018_Task3_Training_GroundTruth.csv",
        image_subdir="ISIC2018_Task3_Training_Input",
    ),
    "ISIC2019": DatasetConfig(
        dataset_class=ISIC2019Dataset,
        csv_filename="ISIC_2019_Training_GroundTruth.csv",
        image_subdir="ISIC_2019_Training_Input",
    ),
    "ISIC2020": DatasetConfig(
        dataset_class=ISIC2020Dataset,
        csv_filename="ISIC_2020_Training_GroundTruth.csv",
        image_subdir="ISIC_2020_Training_JPEG/train",
        alt_csv_filenames=["train.csv"],
        alt_image_subdirs=["train", "ISIC_2020_Training_JPEG"],
    ),
    "PAD-UFES-20": DatasetConfig(
        dataset_class=PADUFES20Dataset,
        csv_filename="metadata.csv",
        image_subdir=None,
    ),
}


def normalize_dataset_name(name: str) -> str:
    """
    Normalize dataset name for comparison.

    Handles variations like 'PADUFES20', 'PAD-UFES-20', 'pad_ufes_20', etc.

    Args:
        name: Dataset name in any format

    Returns:
        Normalized name matching DATASET_REGISTRY keys
    """
    normalized = name.upper().replace("-", "").replace("_", "")

    # Map normalized forms back to canonical names
    name_mapping = {
        "HAM10000": "HAM10000",
        "ISIC2018": "ISIC2018",
        "ISIC2019": "ISIC2019",
        "ISIC2020": "ISIC2020",
        "PADUFES20": "PAD-UFES-20",
    }

    return name_mapping.get(normalized, name)


def get_dataset_paths(
    dataset_name: str,
    data_root: str | Path,
) -> tuple[Path | None, Path | None]:
    """
    Get the CSV path and image root for a dataset.

    Handles alternative paths for datasets with multiple possible locations.

    Args:
        dataset_name: Name of the dataset (will be normalized)
        data_root: Root directory containing all datasets

    Returns:
        Tuple of (csv_path, image_root) or (None, None) if not found
    """
    canonical_name = normalize_dataset_name(dataset_name)

    if canonical_name not in DATASET_REGISTRY:
        return None, None

    config = DATASET_REGISTRY[canonical_name]
    data_root = Path(data_root)
    dataset_dir = data_root / canonical_name

    # Find CSV file
    csv_path = dataset_dir / config.csv_filename
    if not csv_path.exists() and config.alt_csv_filenames:
        for alt_csv in config.alt_csv_filenames:
            alt_path = dataset_dir / alt_csv
            if alt_path.exists():
                csv_path = alt_path
                break

    # Find image directory
    if config.image_subdir:
        image_root = dataset_dir / config.image_subdir
        if not image_root.exists() and config.alt_image_subdirs:
            for alt_dir in config.alt_image_subdirs:
                alt_path = dataset_dir / alt_dir
                if alt_path.exists():
                    image_root = alt_path
                    break
    else:
        image_root = dataset_dir

    return csv_path, image_root


def load_dataset(
    dataset_name: str,
    data_root: str | Path,
    transform: Callable | None = None,
    classification_mode: ClassificationMode = "multiclass",
    filter_unknown: bool = True,
) -> BaseDermoscopyDataset | None:
    """
    Load a dataset by name using the registry.

    This is the unified way to load any supported dataset, handling
    path resolution and alternative locations automatically.

    Args:
        dataset_name: Name of the dataset (e.g., 'HAM10000', 'ISIC2018')
        data_root: Root directory containing all datasets
        transform: Optional transform to apply to images
        classification_mode: 'multiclass' (7), 'multiclass_8' (8), or 'binary'
        filter_unknown: Whether to filter out unknown labels

    Returns:
        Loaded dataset or None if loading fails

    Example:
        >>> dataset = load_dataset('HAM10000', './data', transform=my_transform)
        >>> if dataset:
        ...     print(f"Loaded {len(dataset)} samples")
    """
    canonical_name = normalize_dataset_name(dataset_name)

    if canonical_name not in DATASET_REGISTRY:
        logger.warning(f"Unknown dataset: {dataset_name}")
        return None

    config = DATASET_REGISTRY[canonical_name]
    csv_path, image_root = get_dataset_paths(canonical_name, data_root)

    if csv_path is None or not csv_path.exists():
        logger.warning(f"Dataset {canonical_name}: CSV not found")
        return None

    if image_root is None or not image_root.exists():
        logger.warning(f"Dataset {canonical_name}: Image directory not found at {image_root}")
        return None

    try:
        dataset = config.dataset_class(
            root_dir=str(image_root),
            csv_path=str(csv_path),
            transform=transform,
            classification_mode=classification_mode,
            filter_unknown=filter_unknown,
        )
        logger.info(f"Loaded {canonical_name}: {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.warning(f"Failed to load {canonical_name}: {e}")
        return None


def get_available_datasets() -> list[str]:
    """
    Get list of all available dataset names.

    Returns:
        List of canonical dataset names
    """
    return list(DATASET_REGISTRY.keys())
