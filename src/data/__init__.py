"""
Data handling utilities for dermoscopy datasets.
"""

from .datasets import (
    HAM10000Dataset,
    ISIC2018Dataset,
    ISIC2019Dataset,
    ISIC2020Dataset,
    get_client_dataloader,
    CLASS_NAMES,
    UNIFIED_CLASSES
)
from .preprocessing import (
    get_train_transforms,
    get_val_transforms,
    get_standardized_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DERMOSCOPY_MEAN,
    DERMOSCOPY_STD
)
from .splits import (
    train_val_split,
    create_iid_split,
    create_noniid_split,
    create_label_skew_split,
    create_quantity_skew_split,
    get_dataset_statistics,
    print_split_summary
)
from .download import (
    create_directory_structure,
    verify_dataset,
    verify_all_datasets,
    print_download_instructions,
    DatasetSetupWizard,
    DATASET_INFO
)
from .verify import DatasetVerifier

__all__ = [
    # Datasets
    "HAM10000Dataset",
    "ISIC2018Dataset", 
    "ISIC2019Dataset",
    "ISIC2020Dataset",
    "get_client_dataloader",
    "CLASS_NAMES",
    "UNIFIED_CLASSES",
    # Preprocessing
    "get_train_transforms",
    "get_val_transforms",
    "get_standardized_transforms",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DERMOSCOPY_MEAN",
    "DERMOSCOPY_STD",
    # Splits
    "train_val_split",
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
    "DATASET_INFO"
]
