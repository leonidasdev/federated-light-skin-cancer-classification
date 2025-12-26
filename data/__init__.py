"""
Módulo de gestión de datos.
"""

from .data_loader import (
    DatasetLoader,
    HAM10000Loader,
    ISIC2018Loader,
    ISIC2019Loader,
    ISIC2020Loader,
    PH2Loader,
    get_dataset_loader,
    load_node_data,
    split_data_iid,
    split_data_non_iid,
    load_external_validation_data,
    print_dataset_info
)

from .preprocessing import (
    ImagePreprocessor,
    DataAugmentor,
    split_dataset,
    compute_class_weights,
    apply_smote,
    normalize_images,
    resize_images,
    create_tf_dataset,
    validate_data_distribution,
    print_preprocessing_summary
)

__all__ = [
    # Data loaders
    'DatasetLoader',
    'HAM10000Loader',
    'ISIC2018Loader',
    'ISIC2019Loader',
    'ISIC2020Loader',
    'PH2Loader',
    'get_dataset_loader',
    'load_node_data',
    'split_data_iid',
    'split_data_non_iid',
    'load_external_validation_data',
    'print_dataset_info',
    
    # Preprocessing
    'ImagePreprocessor',
    'DataAugmentor',
    'split_dataset',
    'compute_class_weights',
    'apply_smote',
    'normalize_images',
    'resize_images',
    'create_tf_dataset',
    'validate_data_distribution',
    'print_preprocessing_summary'
]
