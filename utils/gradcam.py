"""
Utilidades para interpretabilidad del modelo usando Grad-CAM.
Permite visualizar qué regiones de la imagen influyen en las predicciones.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple
from pathlib import Path

from config.config import METRICS_CONFIG, CLASS_NAMES_FULL


class GradCAM:
    """
    Implementación de Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Referencia: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"
    """
    
    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        """
        Inicializa Grad-CAM.
        
        Args:
            model (keras.Model): Modelo a interpretar
            layer_name (str): Nombre de la capa convolucional (última por defecto)
        """
        self.model = model
        
        # Si no se especifica, buscar última capa convolucional
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        
        self.layer_name = layer_name
        
        # Crear modelo para extraer activaciones y gradientes
        self.grad_model = self._create_grad_model()
    
    def _find_last_conv_layer(self) -> str:
        """Encuentra la última capa convolucional."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
        raise ValueError("No se encontró ninguna capa convolucional")
    
    def _create_grad_model(self) -> keras.Model:
        """Crea modelo para calcular gradientes."""
        return keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
    
    def compute_heatmap(self, 
                       image: np.ndarray,
                       class_idx: Optional[int] = None,
                       normalize: bool = True) -> np.ndarray:
        """
        Calcula el heatmap de Grad-CAM para una imagen.
        
        Args:
            image (np.ndarray): Imagen de entrada (sin batch dimension)
            class_idx (int): Índice de clase (None = clase predicha)
            normalize (bool): Si normalizar el heatmap
        
        Returns:
            np.ndarray: Heatmap de Grad-CAM
        """
        # Añadir batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Calcular gradientes
        with tf.GradientTape() as tape:
            # Forward pass
            conv_outputs, predictions = self.grad_model(image)
            
            # Si no se especifica clase, usar la predicha
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Output de la clase target
            class_channel = predictions[:, class_idx]
        
        # Calcular gradientes de la clase respecto a las activaciones
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling de gradientes
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Ponderar activaciones por gradientes
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Crear heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # ReLU (solo valores positivos)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalizar
        if normalize and heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        return heatmap
    
    def overlay_heatmap(self,
                       image: np.ndarray,
                       heatmap: np.ndarray,
                       alpha: float = 0.4,
                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Superpone el heatmap sobre la imagen original.
        
        Args:
            image (np.ndarray): Imagen original
            heatmap (np.ndarray): Heatmap de Grad-CAM
            alpha (float): Transparencia del heatmap
            colormap (int): Colormap de OpenCV
        
        Returns:
            np.ndarray: Imagen con heatmap superpuesto
        """
        # Redimensionar heatmap al tamaño de la imagen
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convertir heatmap a RGB con colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            colormap
        )
        
        # Convertir a float
        heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
        
        # Asegurar que la imagen esté en [0, 1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # Superponer
        overlayed = heatmap_colored * alpha + image * (1 - alpha)
        overlayed = np.clip(overlayed, 0, 1)
        
        return overlayed
    
    def visualize(self,
                 image: np.ndarray,
                 class_idx: Optional[int] = None,
                 save_path: Optional[str] = None,
                 show_plot: bool = True,
                 title: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualiza Grad-CAM para una imagen.
        
        Args:
            image (np.ndarray): Imagen original
            class_idx (int): Índice de clase (None = predicha)
            save_path (str): Ruta para guardar visualización
            show_plot (bool): Si mostrar el plot
            title (str): Título personalizado
        
        Returns:
            tuple: (heatmap, overlayed_image)
        """
        # Calcular heatmap
        heatmap = self.compute_heatmap(image, class_idx)
        
        # Superponer heatmap
        overlayed = self.overlay_heatmap(image, heatmap)
        
        # Obtener predicción
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        predictions = self.model.predict(image_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Visualizar
        if show_plot or save_path:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Imagen original
            axes[0].imshow(image if image.max() <= 1 else image / 255.0)
            axes[0].set_title('Imagen Original')
            axes[0].axis('off')
            
            # Heatmap
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(overlayed)
            axes[2].set_title('Superposición')
            axes[2].axis('off')
            
            # Título general
            if title is None:
                class_name = CLASS_NAMES_FULL.get(predicted_class, f"Clase {predicted_class}")
                title = f"Predicción: {class_name} ({confidence:.2%})\nCapa: {self.layer_name}"
            
            fig.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualización guardada en: {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        return heatmap, overlayed
    
    def visualize_multiple_classes(self,
                                   image: np.ndarray,
                                   class_indices: list,
                                   save_path: Optional[str] = None) -> dict:
        """
        Visualiza Grad-CAM para múltiples clases.
        
        Args:
            image (np.ndarray): Imagen original
            class_indices (list): Lista de índices de clases
            save_path (str): Ruta para guardar
        
        Returns:
            dict: Diccionario con heatmaps por clase
        """
        n_classes = len(class_indices)
        fig, axes = plt.subplots(2, n_classes + 1, figsize=(4 * (n_classes + 1), 8))
        
        # Imagen original (arriba izquierda)
        axes[0, 0].imshow(image if image.max() <= 1 else image / 255.0)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        heatmaps = {}
        
        # Grad-CAM para cada clase
        for i, class_idx in enumerate(class_indices):
            heatmap = self.compute_heatmap(image, class_idx)
            overlayed = self.overlay_heatmap(image, heatmap)
            
            class_name = CLASS_NAMES_FULL.get(class_idx, f"Clase {class_idx}")
            heatmaps[class_name] = heatmap
            
            # Heatmap
            axes[0, i + 1].imshow(heatmap, cmap='jet')
            axes[0, i + 1].set_title(f'{class_name}\n(Heatmap)')
            axes[0, i + 1].axis('off')
            
            # Overlay
            axes[1, i + 1].imshow(overlayed)
            axes[1, i + 1].set_title('Superposición')
            axes[1, i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualización guardada en: {save_path}")
        
        plt.show()
        
        return heatmaps


def apply_gradcam_to_batch(model: keras.Model,
                           images: np.ndarray,
                           layer_name: Optional[str] = None,
                           save_dir: Optional[str] = None) -> list:
    """
    Aplica Grad-CAM a un batch de imágenes.
    
    Args:
        model: Modelo
        images: Batch de imágenes
        layer_name: Capa convolucional
        save_dir: Directorio para guardar visualizaciones
    
    Returns:
        list: Lista de heatmaps
    """
    gradcam = GradCAM(model, layer_name)
    heatmaps = []
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    for i, image in enumerate(images):
        heatmap, _ = gradcam.visualize(
            image,
            save_path=str(save_path / f'gradcam_{i}.png') if save_dir else None,
            show_plot=False
        )
        heatmaps.append(heatmap)
    
    return heatmaps


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Probando Grad-CAM...")
    
    # Crear modelo dummy
    from models.cnn_model import create_cnn_model, compile_model
    
    model = create_cnn_model()
    model = compile_model(model)
    
    # Imagen dummy
    dummy_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Crear Grad-CAM
    gradcam = GradCAM(model)
    print(f"Usando capa: {gradcam.layer_name}")
    
    # Calcular heatmap
    heatmap = gradcam.compute_heatmap(dummy_image)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Visualizar (sin mostrar)
    heatmap, overlayed = gradcam.visualize(dummy_image, show_plot=False)
    print("✓ Grad-CAM funcionando correctamente")
