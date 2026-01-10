"""
Image preprocessing and augmentation for skin lesion classification.

Includes:
 - Resizing
 - Normalization
 - Data augmentation
 - Class balancing
 - Artifact removal (optional)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import cv2

from config.config import DATA_CONFIG


class ImagePreprocessor:
    """
    Image preprocessor for skin lesion images.
    """
    
    def __init__(self, target_size=(224, 224), normalize=True):
        """
        Initialize the preprocessor.

        Args:
            target_size (tuple): Target image size
            normalize (bool): Whether to normalize to [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image.

        Args:
            image (np.ndarray): Original image

        Returns:
            np.ndarray: Preprocessed image
        """
        # TODO: implement preprocessing
        # 1. Resize to target_size
        # 2. Normalize if requested
        # 3. Apply enhancements (optional)

        # Placeholder
        processed = image

        # Resize
        if image.shape[:2] != self.target_size:
            processed = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        if self.normalize and processed.max() > 1.0:
            processed = processed.astype(np.float32) / 255.0

        return processed
    
    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess a batch of images.

        Args:
            images (np.ndarray): Array of images

        Returns:
            np.ndarray: Preprocessed images
        """
        return np.array([self.preprocess_image(img) for img in images])
    
    def remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Remove common artifacts (hair, ruler marks, etc.).

        Args:
            image (np.ndarray): Original image

        Returns:
            np.ndarray: Image with reduced artifacts
        """
        # TODO: implement artifact removal
        # 1. Detect hair using morphology
        # 2. Inpainting to fill
        # 3. Remove border marks

        pass


class DataAugmentor:
    """
    Data augmentor for training.
    """
    
    def __init__(self, augmentation_config: dict = None):
        """
        Initialize the augmentor.

        Args:
            augmentation_config (dict): Augmentation configuration
        """
        if augmentation_config is None:
            augmentation_config = DATA_CONFIG['augmentation']
        
        self.config = augmentation_config
        self.augmentor = self._create_augmentor()
    
    def _create_augmentor(self) -> ImageDataGenerator:
        """
        Create the augmentation generator.

        Returns:
            ImageDataGenerator: Configured generator
        """
        return ImageDataGenerator(
            rotation_range=self.config['rotation_range'],
            width_shift_range=self.config['width_shift_range'],
            height_shift_range=self.config['height_shift_range'],
            shear_range=self.config['shear_range'],
            zoom_range=self.config['zoom_range'],
            horizontal_flip=self.config['horizontal_flip'],
            vertical_flip=self.config['vertical_flip'],
            fill_mode=self.config['fill_mode']
        )
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to a single image.

        Args:
            image (np.ndarray): Original image

        Returns:
            np.ndarray: Augmented image
        """
        # TODO: implement single-image augmentation
        pass
    
    def get_augmented_generator(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        """
        Crea un generador con augmentation para entrenamiento.
        
        Args:
            X (np.ndarray): Imágenes
            y (np.ndarray): Etiquetas
            batch_size (int): Tamaño del batch
        
        Returns:
            generator: Generador de datos augmentados
        """
        return self.augmentor.flow(X, y, batch_size=batch_size)


def split_dataset(X: np.ndarray, 
                  y: np.ndarray, 
                  train_ratio: float = 0.7, 
                  val_ratio: float = 0.15, 
                  test_ratio: float = 0.15,
                  stratify: bool = True,
                  random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Split dataset into train, validation and test sets.

    Args:
        X (np.ndarray): Images
        y (np.ndarray): Labels
        train_ratio (float): Train proportion
        val_ratio (float): Validation proportion
        test_ratio (float): Test proportion
        stratify (bool): Whether to stratify by class
        random_state (int): Random seed

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # TODO: implement stratified splitting
    # 1. Validate ratios sum to 1
    # 2. First split: train vs (val + test)
    # 3. Second split: val vs test
    # 4. Verify class distribution

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Placeholder
    return None, None, None, None, None, None


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute class weights to handle imbalance.

    Args:
        y (np.ndarray): Labels (one-hot or indices)

    Returns:
        dict: Mapping class -> weight
    """
    # TODO: implement class weight calculation
    # Use sklearn.utils.class_weight.compute_class_weight

    from sklearn.utils.class_weight import compute_class_weight

    # If one-hot, convert to indices
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y

    classes = np.unique(y_indices)
    weights = compute_class_weight('balanced', classes=classes, y=y_indices)

    return dict(zip(classes, weights))


def apply_smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE for oversampling minority classes.

    Args:
        X (np.ndarray): Images
        y (np.ndarray): Labels

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    # TODO: implement SMOTE for images
    # Note: SMOTE on images requires special care
    # Alternative: use selective data augmentation

    from imblearn.over_sampling import SMOTE

    # Flatten images for SMOTE
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)

    # Restore original shape
    X_resampled = X_resampled.reshape(-1, *X.shape[1:])

    return X_resampled, y_resampled


def normalize_images(images: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normaliza imágenes.
    
    Args:
        images (np.ndarray): Imágenes a normalizar
        method (str): Método ('minmax' para [0,1], 'standardize' para z-score)
    
    Returns:
        np.ndarray: Imágenes normalizadas
    """
    if method == 'minmax':
        # Normalize to [0, 1]
        return images.astype(np.float32) / 255.0 if images.max() > 1.0 else images

    elif method == 'standardize':
        # Standardize using ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (images - mean) / std

    else:
        raise ValueError(f"Method {method} not supported")


def resize_images(images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a collection of images.

    Args:
        images (np.ndarray): Array of images
        target_size (tuple): Target size (height, width)

    Returns:
        np.ndarray: Resized images
    """
    resized = []
    for img in images:
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        resized.append(resized_img)

    return np.array(resized)


def create_tf_dataset(X: np.ndarray, 
                      y: np.ndarray, 
                      batch_size: int = 32,
                      shuffle: bool = True,
                      augment: bool = False) -> tf.data.Dataset:
    """
    Create an optimized `tf.data.Dataset`.

    Args:
        X (np.ndarray): Images
        y (np.ndarray): Labels
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle
        augment (bool): Whether to apply augmentation

    Returns:
        tf.data.Dataset: Optimized dataset
    """
    # TODO: implement optimized tf.data pipeline
    # 1. Create base dataset
    # 2. Apply augmentation if needed
    # 3. Apply prefetch, cache

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ==================== FUNCIONES DE VALIDACIÓN ====================

def validate_data_distribution(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
    """
    Validate that class distribution is similar across splits.

    Args:
        y_train, y_val, y_test: Label arrays
    """
    # TODO: implement distribution validation
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION IN SPLITS")
    print("=" * 60)

    from config.config import CLASS_NAMES

    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        if y_split is not None:
            unique, counts = np.unique(np.argmax(y_split, axis=1) if len(y_split.shape) > 1 else y_split, return_counts=True)
            print(f"\n{split_name}:")
            for class_idx, count in zip(unique, counts):
                class_name = CLASS_NAMES.get(class_idx, f"Class {class_idx}")
                percentage = (count / len(y_split)) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")

    print("=" * 60 + "\n")


def print_preprocessing_summary(X_original: np.ndarray, X_processed: np.ndarray):
    """
    Print a preprocessing summary.

    Args:
        X_original: Original data
        X_processed: Processed data
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Original shape: {X_original.shape}")
    print(f"Processed shape: {X_processed.shape}")
    print(f"Original range: [{X_original.min():.2f}, {X_original.max():.2f}]")
    print(f"Processed range: [{X_processed.min():.2f}, {X_processed.max():.2f}]")
    print("=" * 60 + "\n")


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Testing preprocessing...")

    # Create dummy data
    dummy_images = np.random.randint(0, 255, size=(100, 300, 300, 3), dtype=np.uint8)
    dummy_labels = np.random.randint(0, 7, size=(100,))

    # Test preprocessor
    preprocessor = ImagePreprocessor()
    processed = preprocessor.preprocess_batch(dummy_images[:5])
    print(f"Processed images: {processed.shape}")

    # Test augmentor
    augmentor = DataAugmentor()
    print("Augmentor created with config:")
    print(augmentor.config)
