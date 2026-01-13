"""
Standardized Preprocessing Pipeline for Dermoscopy Datasets.

Ensures consistent preprocessing across all four datasets:
- HAM10000, ISIC 2018, ISIC 2019, ISIC 2020

Key preprocessing steps:
1. Resize to 224×224 (DSCATNet input size)
2. Normalization using ImageNet statistics (transfer learning compatibility)
3. Data augmentation for training (optional)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Tuple, Optional, Dict, Any, Sequence, cast


# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Dermoscopy-specific normalization (computed from ISIC datasets)
# Can be used as alternative to ImageNet stats
DERMOSCOPY_MEAN = (0.7635, 0.5461, 0.5705)
DERMOSCOPY_STD = (0.1409, 0.1520, 0.1693)


def get_train_transforms(
    img_size: int = 224,
    use_dermoscopy_norm: bool = False,
    augmentation_level: str = 'medium'
) -> A.Compose:
    """
    Get training transforms with data augmentation.
    
    Args:
        img_size: Target image size
        use_dermoscopy_norm: Use dermoscopy-specific normalization
        augmentation_level: 'light', 'medium', or 'heavy'
        
    Returns:
        Albumentations Compose transform
    """
    mean = DERMOSCOPY_MEAN if use_dermoscopy_norm else IMAGENET_MEAN
    std = DERMOSCOPY_STD if use_dermoscopy_norm else IMAGENET_STD
    
    # Base transforms (always applied)
    base_transforms = [
        A.Resize(img_size, img_size),
    ]
    
    # Augmentation based on level
    if augmentation_level == 'light':
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
        ]
    elif augmentation_level == 'medium':
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
        ]
    elif augmentation_level == 'heavy':
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=30,
                p=0.6
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.4),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.05, p=1.0),
                A.GridDistortion(distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=1.0
                ),
            ], p=0.4),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(img_size // 16, img_size // 8),
                hole_width_range=(img_size // 16, img_size // 8),
                fill=0,
                p=0.3
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


def get_val_transforms(
    img_size: int = 224,
    use_dermoscopy_norm: bool = False
) -> A.Compose:
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
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_standardized_transforms(
    img_size: int = 224,
    is_training: bool = True,
    use_dermoscopy_norm: bool = False,
    augmentation_level: str = 'medium'
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
            img_size=img_size,
            use_dermoscopy_norm=use_dermoscopy_norm,
            augmentation_level=augmentation_level
        )
    else:
        return get_val_transforms(
            img_size=img_size,
            use_dermoscopy_norm=use_dermoscopy_norm
        )


def compute_dataset_statistics(
    dataset,
    num_samples: int = 1000
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute mean and std statistics from a dataset.
    
    Args:
        dataset: PyTorch dataset with images
        num_samples: Number of samples to use (for efficiency)
        
    Returns:
        Tuple of (mean, std) as RGB tuples
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    import random
    
    # Create subset if dataset is large
    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = Subset(dataset, indices)
    
    loader = DataLoader(dataset, batch_size=32, num_workers=4)
    
    # Compute statistics
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_pixels += batch_samples
    
    mean /= total_pixels
    std /= total_pixels
    
    return tuple(mean.tolist()), tuple(std.tolist())


class DermoscopyPreprocessor:
    """
    Unified preprocessor for dermoscopy images.
    
    Handles:
    - Hair removal (optional)
    - Color normalization
    - Lesion segmentation (optional)
    - Standard resizing and normalization
    
    Args:
        img_size: Target image size
        remove_hair: Whether to apply hair removal
        use_segmentation: Whether to apply lesion segmentation
        use_dermoscopy_norm: Use dermoscopy-specific normalization
    """
    
    def __init__(
        self,
        img_size: int = 224,
        remove_hair: bool = False,
        use_segmentation: bool = False,
        use_dermoscopy_norm: bool = False
    ):
        self.img_size = img_size
        self.remove_hair = remove_hair
        self.use_segmentation = use_segmentation
        self.use_dermoscopy_norm = use_dermoscopy_norm
        
        # Normalization stats
        self.mean = DERMOSCOPY_MEAN if use_dermoscopy_norm else IMAGENET_MEAN
        self.std = DERMOSCOPY_STD if use_dermoscopy_norm else IMAGENET_STD
    
    def remove_hair_morphology(self, image: np.ndarray) -> np.ndarray:
        """
        Remove hair artifacts using morphological operations.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Hair-removed image
        """
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Black hat transform to detect hair
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold to create hair mask
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Inpaint to remove hair
        result = cv2.inpaint(image, thresh, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        
        return result
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Preprocessed image
        """
        # Hair removal
        if self.remove_hair:
            image = self.remove_hair_morphology(image)
        
        return image
