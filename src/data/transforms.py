"""
Data Transforms
===============

Image transformation pipelines for training and evaluation.
"""

from typing import Tuple

from torchvision import transforms


def get_train_transforms(
    img_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
):
    """
    Get training transforms with data augmentation.
    
    Args:
        img_size: Target image size
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)
    
    Returns:
        Composed transforms for training
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    img_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
):
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        img_size: Target image size
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)
    
    Returns:
        Composed transforms for validation/testing
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_dermoscopy_transforms(
    img_size: int = 224,
):
    """
    Get dermoscopy-specific transforms with hair removal and artifact handling.
    
    Args:
        img_size: Target image size
    
    Returns:
        Composed transforms for dermoscopy images
    """
    # TODO: Implement dermoscopy-specific preprocessing
    # - Hair removal
    # - Vignette correction
    # - Color normalization
    raise NotImplementedError("Dermoscopy transforms pending")
