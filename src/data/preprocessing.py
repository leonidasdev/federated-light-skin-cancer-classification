# =============================================================================
# Standardized Preprocessing Pipeline for Dermoscopy Datasets
# =============================================================================
"""
Standardized Preprocessing Pipeline for Dermoscopy Datasets.

Ensures consistent preprocessing across all five datasets:
- HAM10000, ISIC 2018, ISIC 2019, ISIC 2020, PAD-UFES-20

Key preprocessing steps:
1. Resize to 224×224 (DSCATNet input size)
2. Normalization using ImageNet statistics (transfer learning compatibility)
3. Data augmentation for training (optional)
"""

# =============================================================================
# Imports
# =============================================================================

import logging

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Any, cast
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Dermoscopy-specific normalization (computed from ISIC datasets)
# Can be used as alternative to ImageNet stats
DERMOSCOPY_MEAN = (0.7635, 0.5461, 0.5705)
DERMOSCOPY_STD = (0.1409, 0.1520, 0.1693)

# =============================================================================
# Transform Functions
# =============================================================================


def get_train_transforms(
    img_size: int = 224, use_dermoscopy_norm: bool = False, augmentation_level: str = "medium"
) -> A.Compose:
    """
    Get training transforms with configurable data augmentation.

    Creates an Albumentations pipeline for dermoscopy image preprocessing
    with various levels of augmentation suitable for skin lesion classification.

    Args:
        img_size: Target image size (both height and width).
        use_dermoscopy_norm: If True, use dermoscopy-specific normalization
            statistics (computed from ISIC datasets). If False, use ImageNet
            statistics for compatibility with pretrained models.
        augmentation_level: Augmentation intensity. Options:
            - 'none': Only resize and normalize (for validation/test-like behavior)
            - 'light': Basic flips and small rotations
            - 'medium': Adds brightness/contrast, affine transforms, blur
            - 'heavy': Maximum augmentation with color jitter, dropout, etc.

    Returns:
        Albumentations Compose transform pipeline that outputs torch tensors.

    Raises:
        ValueError: If augmentation_level is not one of the valid options.

    Example:
        >>> train_transforms = get_train_transforms(img_size=224, augmentation_level='medium')
        >>> transformed = train_transforms(image=img_array)
        >>> tensor = transformed['image']  # Shape: (3, 224, 224)
    """
    mean = DERMOSCOPY_MEAN if use_dermoscopy_norm else IMAGENET_MEAN
    std = DERMOSCOPY_STD if use_dermoscopy_norm else IMAGENET_STD

    # Base transforms (always applied)
    base_transforms = [
        A.Resize(img_size, img_size),
    ]

    # Augmentation based on level
    if augmentation_level == "none":
        aug_transforms = []
    elif augmentation_level == "light":
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
        ]
    elif augmentation_level == "medium":
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Affine(
                translate_percent=0.1,
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.3,
            ),
        ]
    elif augmentation_level == "heavy":
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.Affine(
                translate_percent=0.15,
                scale=(0.85, 1.15),
                rotate=(-30, 30),
                p=0.6,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=0.05, p=1.0),
                    A.GridDistortion(distort_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                ],
                p=0.4,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(img_size // 16, img_size // 8),
                hole_width_range=(img_size // 16, img_size // 8),
                fill=0,
                p=0.3,
            ),
        ]
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")

    # Final normalization and tensor conversion
    final_transforms = [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    # use a Sequence annotation and cast when calling Compose to satisfy
    # different albumentations type signatures across environments
    transforms: Sequence[Any] = base_transforms + aug_transforms + final_transforms
    return A.Compose(cast(Any, transforms))


def get_val_transforms(img_size: int = 224, use_dermoscopy_norm: bool = False) -> A.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        img_size: Target image size
        use_dermoscopy_norm: Use dermoscopy-specific normalization

    Returns:
        Albumentations Compose transform
    """
    mean = DERMOSCOPY_MEAN if use_dermoscopy_norm else IMAGENET_MEAN
    std = DERMOSCOPY_STD if use_dermoscopy_norm else IMAGENET_STD

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_standardized_transforms(
    img_size: int = 224, is_training: bool = True, use_dermoscopy_norm: bool = False, augmentation_level: str = "medium"
) -> A.Compose:
    """
    Get standardized transforms for all datasets.

    This is the main function to use for consistent preprocessing
    across all FL clients.

    Args:
        img_size: Target image size
        is_training: Whether to apply training augmentations
        use_dermoscopy_norm: Use dermoscopy-specific normalization
        augmentation_level: Augmentation level for training

    Returns:
        Albumentations Compose transform
    """
    if is_training:
        return get_train_transforms(
            img_size=img_size, use_dermoscopy_norm=use_dermoscopy_norm, augmentation_level=augmentation_level
        )
    return get_val_transforms(img_size=img_size, use_dermoscopy_norm=use_dermoscopy_norm)


def get_transform_pair(
    img_size: int = 224,
    augmentation_level: str = "medium",
    use_dermoscopy_norm: bool = False,
) -> tuple[A.Compose, A.Compose]:
    """Return (train_transform, val_transform) for the given config.

    Convenience wrapper used by both CentralizedTrainer and FLSimulator
    to avoid duplicating the same call pattern.
    """
    train_transform = get_train_transforms(
        img_size=img_size,
        augmentation_level=augmentation_level,
        use_dermoscopy_norm=use_dermoscopy_norm,
    )
    val_transform = get_val_transforms(
        img_size=img_size,
        use_dermoscopy_norm=use_dermoscopy_norm,
    )
    return train_transform, val_transform
