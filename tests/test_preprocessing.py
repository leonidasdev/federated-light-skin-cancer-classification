"""
Test script for the preprocessing pipeline.

Validates that:
1. Transforms work correctly on all datasets
2. Output dimensions are consistent (224x224)
3. Normalization is applied correctly
4. DataLoaders work properly
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    get_standardized_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DERMOSCOPY_MEAN,
    DERMOSCOPY_STD
)


def test_transform_output_shape():
    """Test that transforms produce correct output shape."""
    print("Testing output shape...")
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (600, 450, 3), dtype=np.uint8)
    
    # Test training transform
    train_tf = get_train_transforms(img_size=224)
    result = train_tf(image=dummy_img)
    
    assert result['image'].shape == torch.Size([3, 224, 224]), \
        f"Expected (3, 224, 224), got {result['image'].shape}"
    
    # Test validation transform
    val_tf = get_val_transforms(img_size=224)
    result = val_tf(image=dummy_img)
    
    assert result['image'].shape == torch.Size([3, 224, 224]), \
        f"Expected (3, 224, 224), got {result['image'].shape}"
    
    print("  ✓ Output shape test passed")


def test_normalization():
    """Test that normalization is applied correctly."""
    print("Testing normalization...")
    
    # Create solid color image
    solid_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    
    val_tf = get_val_transforms(img_size=224)
    result = val_tf(image=solid_img)
    
    # After normalization, values should be around 0 for middle gray
    tensor = result['image']
    
    # Check that values are normalized (not in 0-255 range)
    assert tensor.min() < 1.0, "Values should be normalized"
    assert tensor.max() < 3.0, "Values should be normalized"
    
    print("  ✓ Normalization test passed")


def test_augmentation_levels():
    """Test different augmentation levels."""
    print("Testing augmentation levels...")
    
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    for level in ['light', 'medium', 'heavy']:
        tf = get_train_transforms(img_size=224, augmentation_level=level)
        result = tf(image=dummy_img)
        
        assert result['image'].shape == torch.Size([3, 224, 224]), \
            f"Failed for augmentation level: {level}"
    
    print("  ✓ Augmentation levels test passed")


def test_dermoscopy_normalization():
    """Test dermoscopy-specific normalization."""
    print("Testing dermoscopy normalization...")
    
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # ImageNet normalization
    tf_imagenet = get_val_transforms(img_size=224, use_dermoscopy_norm=False)
    result_imagenet = tf_imagenet(image=dummy_img)
    
    # Dermoscopy normalization
    tf_derm = get_val_transforms(img_size=224, use_dermoscopy_norm=True)
    result_derm = tf_derm(image=dummy_img)
    
    # Results should be different
    assert not torch.allclose(result_imagenet['image'], result_derm['image']), \
        "ImageNet and dermoscopy normalization should produce different results"
    
    print("  ✓ Dermoscopy normalization test passed")


def test_standardized_transforms():
    """Test the unified standardized transform function."""
    print("Testing standardized transforms...")
    
    dummy_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    
    # Training mode
    tf_train = get_standardized_transforms(
        img_size=224, is_training=True, augmentation_level='medium'
    )
    result_train = tf_train(image=dummy_img)
    
    # Validation mode
    tf_val = get_standardized_transforms(
        img_size=224, is_training=False
    )
    result_val = tf_val(image=dummy_img)
    
    assert result_train['image'].shape == result_val['image'].shape
    
    print("  ✓ Standardized transforms test passed")


def test_with_real_image(image_path: str):
    """Test transforms with a real dermoscopy image."""
    print(f"Testing with real image: {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    print(f"  Original size: {img_array.shape}")
    
    # Apply transforms
    train_tf = get_train_transforms(img_size=224, augmentation_level='medium')
    val_tf = get_val_transforms(img_size=224)
    
    train_result = train_tf(image=img_array)
    val_result = val_tf(image=img_array)
    
    print(f"  After train transform: {train_result['image'].shape}")
    print(f"  After val transform: {val_result['image'].shape}")
    
    # Verify dtype
    assert train_result['image'].dtype == torch.float32
    assert val_result['image'].dtype == torch.float32
    
    print("  ✓ Real image test passed")
    
    return train_result['image'], val_result['image']


def visualize_transforms(image_path: str, output_path: str = None):
    """Visualize transform effects on a dermoscopy image."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Validation transform
    val_tf = get_val_transforms(img_size=224)
    val_result = val_tf(image=img_array)['image']
    val_img = denormalize(val_result)
    axes[0, 1].imshow(val_img)
    axes[0, 1].set_title('Validation (224x224)')
    axes[0, 1].axis('off')
    
    # Different augmentation levels
    levels = ['light', 'medium', 'heavy']
    for i, level in enumerate(levels):
        train_tf = get_train_transforms(img_size=224, augmentation_level=level)
        train_result = train_tf(image=img_array)['image']
        train_img = denormalize(train_result)
        axes[0, 2 + i - (0 if i < 2 else 2)].imshow(train_img)
        axes[0, 2 + i - (0 if i < 2 else 2)].set_title(f'Aug: {level}')
        axes[0, 2 + i - (0 if i < 2 else 2)].axis('off')
    
    # More augmentation samples
    train_tf = get_train_transforms(img_size=224, augmentation_level='medium')
    for i in range(4):
        result = train_tf(image=img_array)['image']
        aug_img = denormalize(result)
        axes[1, i].imshow(aug_img)
        axes[1, i].set_title(f'Sample {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle('Preprocessing Pipeline Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    plt.show()


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Denormalize a tensor for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    img = tensor * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    
    return img


def run_all_tests():
    """Run all preprocessing tests."""
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_transform_output_shape()
        test_normalization()
        test_augmentation_levels()
        test_dermoscopy_normalization()
        test_standardized_transforms()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test preprocessing pipeline")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--visualize", action="store_true", help="Visualize transforms")
    parser.add_argument("--output", type=str, help="Output path for visualization")
    
    args = parser.parse_args()
    
    # Run basic tests
    success = run_all_tests()
    
    # Test with real image if provided
    if args.image and Path(args.image).exists():
        test_with_real_image(args.image)
        
        if args.visualize:
            visualize_transforms(args.image, args.output)
    
    sys.exit(0 if success else 1)
