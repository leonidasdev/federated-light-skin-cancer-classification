# =============================================================================
# Data Handling Utilities for Dermoscopy Datasets
# =============================================================================
"""
Data handling utilities for dermoscopy datasets.

This module provides:
- Dataset classes for HAM10000, ISIC2018, ISIC2019, ISIC2020, PAD-UFES-20
- Preprocessing and augmentation pipelines
- Data splitting utilities for IID and non-IID scenarios
- Dataset verification and download helpers
"""

# =============================================================================
# Dataset Imports
# =============================================================================

from .datasets import (
    CLASS_NAMES,
    DATASET_REGISTRY,
    UNIFIED_CLASSES,
    DatasetConfig,
    DatasetSubset,
    HAM10000Dataset,
    ISIC2018Dataset,
    ISIC2019Dataset,
    ISIC2020Dataset,
    PADUFES20Dataset,
    get_available_datasets,
    get_dataset_paths,
    load_dataset,
    normalize_dataset_name,
)

# =============================================================================
# Download & Verification Imports
# =============================================================================
from .download import (
    DATASET_INFO,
    DatasetSetupWizard,
    create_directory_structure,
    print_download_instructions,
    verify_all_datasets,
    verify_dataset,
)

# =============================================================================
# Preprocessing Imports
# =============================================================================
from .preprocessing import (
    DERMOSCOPY_MEAN,
    DERMOSCOPY_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_standardized_transforms,
    get_train_transforms,
    get_transform_pair,
    get_val_transforms,
)

# =============================================================================
# Splits Imports
# =============================================================================
from .splits import (
    create_iid_split,
    create_label_skew_split,
    create_noniid_split,
    create_quantity_skew_split,
    deterministic_train_val_split,
    deterministic_train_val_test_split,
    get_dataset_statistics,
    print_split_summary,
)
from .verify import DatasetVerifier

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Datasets
    "HAM10000Dataset",
    "ISIC2018Dataset",
    "ISIC2019Dataset",
    "ISIC2020Dataset",
    "PADUFES20Dataset",
    "CLASS_NAMES",
    "UNIFIED_CLASSES",
    # Dataset Registry
    "DATASET_REGISTRY",
    "DatasetConfig",
    "load_dataset",
    "get_dataset_paths",
    "get_available_datasets",
    "normalize_dataset_name",
    "DatasetSubset",
    # Preprocessing
    "get_train_transforms",
    "get_val_transforms",
    "get_standardized_transforms",
    "get_transform_pair",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DERMOSCOPY_MEAN",
    "DERMOSCOPY_STD",
    # Splits
    "deterministic_train_val_split",
    "deterministic_train_val_test_split",
    "create_iid_split",
    "create_noniid_split",
    "create_label_skew_split",
    "create_quantity_skew_split",
    "get_dataset_statistics",
    "print_split_summary",
    # Download & Verification
    "create_directory_structure",
    "verify_dataset",
    "verify_all_datasets",
    "print_download_instructions",
    "DatasetSetupWizard",
    "DatasetVerifier",
    "DATASET_INFO",
]
