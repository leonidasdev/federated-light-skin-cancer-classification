"""
Preprocesamiento y augmentation de imágenes para clasificación de lesiones cutáneas.

Incluye:
- Redimensionamiento
- Normalización
- Data augmentation
- Balance de clases
- Remoción de artefactos (opcional)
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
    Preprocesador de imágenes para lesiones cutáneas.
    """
    
    def __init__(self, target_size=(224, 224), normalize=True):
        """
        Inicializa el preprocesador.
        
        Args:
            target_size (tuple): Tamaño objetivo de las imágenes
            normalize (bool): Si normalizar a [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen individual.
        
        Args:
            image (np.ndarray): Imagen original
        
        Returns:
            np.ndarray: Imagen preprocesada
        """
        # TODO: Implementar preprocesamiento
        # 1. Redimensionar a target_size
        # 2. Normalizar si es necesario
        # 3. Aplicar mejoras (opcional)
        
        # Placeholder
        processed = image
        
        # Redimensionar
        if image.shape[:2] != self.target_size:
            processed = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Normalizar a [0, 1]
        if self.normalize and processed.max() > 1.0:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocesa un batch de imágenes.
        
        Args:
            images (np.ndarray): Array de imágenes
        
        Returns:
            np.ndarray: Imágenes preprocesadas
        """
        return np.array([self.preprocess_image(img) for img in images])
    
    def remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Elimina artefactos comunes (pelo, marcas de regla, etc.).
        
        Args:
            image (np.ndarray): Imagen original
        
        Returns:
            np.ndarray: Imagen sin artefactos
        """
        # TODO: Implementar remoción de artefactos
        # 1. Detectar pelo usando morfología
        # 2. Inpainting para rellenar
        # 3. Eliminar marcas de borde
        
        pass


class DataAugmentor:
    """
    Augmentador de datos para entrenamiento.
    """
    
    def __init__(self, augmentation_config: dict = None):
        """
        Inicializa el augmentador.
        
        Args:
            augmentation_config (dict): Configuración de augmentation
        """
        if augmentation_config is None:
            augmentation_config = DATA_CONFIG['augmentation']
        
        self.config = augmentation_config
        self.augmentor = self._create_augmentor()
    
    def _create_augmentor(self) -> ImageDataGenerator:
        """
        Crea el generador de augmentation.
        
        Returns:
            ImageDataGenerator: Generador configurado
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
        Aplica augmentation a una imagen.
        
        Args:
            image (np.ndarray): Imagen original
        
        Returns:
            np.ndarray: Imagen augmentada
        """
        # TODO: Implementar augmentation individual
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
    Divide dataset en train, validation y test.
    
    Args:
        X (np.ndarray): Imágenes
        y (np.ndarray): Etiquetas
        train_ratio (float): Proporción de entrenamiento
        val_ratio (float): Proporción de validación
        test_ratio (float): Proporción de test
        stratify (bool): Si estratificar por clases
        random_state (int): Semilla aleatoria
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # TODO: Implementar división estratificada
    # 1. Validar que las proporciones sumen 1
    # 2. Primera división: train vs (val + test)
    # 3. Segunda división: val vs test
    # 4. Verificar distribución de clases
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Las proporciones deben sumar 1.0"
    
    # Placeholder
    return None, None, None, None, None, None


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Calcula pesos de clase para manejar desbalance.
    
    Args:
        y (np.ndarray): Etiquetas (one-hot o índices)
    
    Returns:
        dict: Diccionario de pesos por clase
    """
    # TODO: Implementar cálculo de class weights
    # Usar sklearn.utils.class_weight.compute_class_weight
    
    from sklearn.utils.class_weight import compute_class_weight
    
    # Si es one-hot, convertir a índices
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
    
    classes = np.unique(y_indices)
    weights = compute_class_weight('balanced', classes=classes, y=y_indices)
    
    return dict(zip(classes, weights))


def apply_smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica SMOTE para oversampling de clases minoritarias.
    
    Args:
        X (np.ndarray): Imágenes
        y (np.ndarray): Etiquetas
    
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    # TODO: Implementar SMOTE para imágenes
    # Nota: SMOTE en imágenes requiere cuidado especial
    # Alternativa: usar data augmentation selectivo
    
    from imblearn.over_sampling import SMOTE
    
    # Aplanar imágenes para SMOTE
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)
    
    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    
    # Recuperar forma original
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
        # Normalizar a [0, 1]
        return images.astype(np.float32) / 255.0 if images.max() > 1.0 else images
    
    elif method == 'standardize':
        # Estandarizar usando media y std de ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (images - mean) / std
    
    else:
        raise ValueError(f"Método {method} no soportado")


def resize_images(images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Redimensiona un conjunto de imágenes.
    
    Args:
        images (np.ndarray): Array de imágenes
        target_size (tuple): Tamaño objetivo (height, width)
    
    Returns:
        np.ndarray: Imágenes redimensionadas
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
    Crea un tf.data.Dataset optimizado.
    
    Args:
        X (np.ndarray): Imágenes
        y (np.ndarray): Etiquetas
        batch_size (int): Tamaño del batch
        shuffle (bool): Si mezclar datos
        augment (bool): Si aplicar augmentation
    
    Returns:
        tf.data.Dataset: Dataset optimizado
    """
    # TODO: Implementar pipeline tf.data optimizado
    # 1. Crear dataset base
    # 2. Aplicar augmentation si es necesario
    # 3. Aplicar prefetch, cache
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ==================== FUNCIONES DE VALIDACIÓN ====================

def validate_data_distribution(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
    """
    Valida que la distribución de clases sea similar en los splits.
    
    Args:
        y_train, y_val, y_test: Arrays de etiquetas
    """
    # TODO: Implementar validación de distribución
    print("\n" + "=" * 60)
    print("DISTRIBUCIÓN DE CLASES EN SPLITS")
    print("=" * 60)
    
    from config.config import CLASS_NAMES
    
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        if y_split is not None:
            unique, counts = np.unique(np.argmax(y_split, axis=1) if len(y_split.shape) > 1 else y_split, return_counts=True)
            print(f"\n{split_name}:")
            for class_idx, count in zip(unique, counts):
                class_name = CLASS_NAMES.get(class_idx, f"Clase {class_idx}")
                percentage = (count / len(y_split)) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print("=" * 60 + "\n")


def print_preprocessing_summary(X_original: np.ndarray, X_processed: np.ndarray):
    """
    Imprime resumen del preprocesamiento.
    
    Args:
        X_original: Datos originales
        X_processed: Datos procesados
    """
    print("\n" + "=" * 60)
    print("RESUMEN DE PREPROCESAMIENTO")
    print("=" * 60)
    print(f"Forma original: {X_original.shape}")
    print(f"Forma procesada: {X_processed.shape}")
    print(f"Rango original: [{X_original.min():.2f}, {X_original.max():.2f}]")
    print(f"Rango procesado: [{X_processed.min():.2f}, {X_processed.max():.2f}]")
    print("=" * 60 + "\n")


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Probando preprocesamiento...")
    
    # Crear datos dummy
    dummy_images = np.random.randint(0, 255, size=(100, 300, 300, 3), dtype=np.uint8)
    dummy_labels = np.random.randint(0, 7, size=(100,))
    
    # Probar preprocesador
    preprocessor = ImagePreprocessor()
    processed = preprocessor.preprocess_batch(dummy_images[:5])
    print(f"Imágenes procesadas: {processed.shape}")
    
    # Probar augmentador
    augmentor = DataAugmentor()
    print("Augmentador creado con configuración:")
    print(augmentor.config)
