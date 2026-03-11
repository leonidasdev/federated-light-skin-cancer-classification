# =============================================================================
# Tests for Preprocessing Pipeline
# =============================================================================
"""
Tests for the preprocessing pipeline.

Validates that:
1. Transforms work correctly on all datasets
2. Output dimensions are consistent (224x224)
3. Normalization is applied correctly
4. DataLoaders work properly
"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    get_standardized_transforms,
)

# Project root for finding test data
PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# Test Functions
# =============================================================================


def test_transform_output_shape():
    """Test that transforms produce correct output shape."""
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (600, 450, 3), dtype=np.uint8)

    # Test training transform
    train_tf = get_train_transforms(img_size=224)
    result = train_tf(image=dummy_img)

    assert result["image"].shape == torch.Size([3, 224, 224]), f"Expected (3, 224, 224), got {result['image'].shape}"

    # Test validation transform
    val_tf = get_val_transforms(img_size=224)
    result = val_tf(image=dummy_img)

    assert result["image"].shape == torch.Size([3, 224, 224]), f"Expected (3, 224, 224), got {result['image'].shape}"


def test_normalization():
    """Test that normalization is applied correctly."""
    # Create solid color image
    solid_img = np.ones((224, 224, 3), dtype=np.uint8) * 128

    val_tf = get_val_transforms(img_size=224)
    result = val_tf(image=solid_img)

    # After normalization, values should be around 0 for middle gray
    tensor = result["image"]

    # Check that values are normalized (not in 0-255 range)
    assert tensor.min() < 1.0, "Values should be normalized"
    assert tensor.max() < 3.0, "Values should be normalized"


def test_augmentation_levels():
    """Test different augmentation levels."""
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    # Test all augmentation levels including 'none' (matches original DSCATNet paper)
    for level in ["none", "light", "medium", "heavy"]:
        tf = get_train_transforms(img_size=224, augmentation_level=level)
        result = tf(image=dummy_img)

        assert result["image"].shape == torch.Size([3, 224, 224]), f"Failed for augmentation level: {level}"


def test_dermoscopy_normalization():
    """Test dermoscopy-specific normalization."""
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # ImageNet normalization
    tf_imagenet = get_val_transforms(img_size=224, use_dermoscopy_norm=False)
    result_imagenet = tf_imagenet(image=dummy_img)

    # Dermoscopy normalization
    tf_derm = get_val_transforms(img_size=224, use_dermoscopy_norm=True)
    result_derm = tf_derm(image=dummy_img)

    # Results should be different
    assert not torch.allclose(result_imagenet["image"], result_derm["image"]), (
        "ImageNet and dermoscopy normalization should produce different results"
    )


def test_standardized_transforms():
    """Test the unified standardized transform function."""
    dummy_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)

    # Training mode
    tf_train = get_standardized_transforms(img_size=224, is_training=True, augmentation_level="medium")
    result_train = tf_train(image=dummy_img)

    # Validation mode
    tf_val = get_standardized_transforms(img_size=224, is_training=False)
    result_val = tf_val(image=dummy_img)

    assert result_train["image"].shape == result_val["image"].shape


def test_with_real_image():
    """Test transforms with a real dermoscopy image."""
    # Try to find a sample image from the datasets
    data_root = PROJECT_ROOT / "data"
    possible_paths = [
        data_root / "HAM10000" / "HAM10000_images_part_1",
        data_root / "ISIC2018" / "ISIC2018_Task3_Training_Input",
        data_root / "ISIC2019" / "ISIC_2019_Training_Input",
    ]

    sample_image_path = None
    for path in possible_paths:
        if path.exists():
            images = list(path.glob("*.jpg"))
            if images:
                sample_image_path = str(images[0])
                break

    if sample_image_path is None:
        pytest.skip("No sample image available - download a dataset first")

    img = Image.open(sample_image_path).convert("RGB")
    img_array = np.array(img)

    # Apply transforms
    train_tf = get_train_transforms(img_size=224, augmentation_level="medium")
    val_tf = get_val_transforms(img_size=224)

    train_result = train_tf(image=img_array)
    val_result = val_tf(image=img_array)

    # Verify dtype
    assert train_result["image"].dtype == torch.float32
    assert val_result["image"].dtype == torch.float32


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
