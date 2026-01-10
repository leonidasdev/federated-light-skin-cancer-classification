"""
Utilities for model interpretability using Grad-CAM.
Allows visualizing which image regions influence predictions.
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
    Implementation of Gradient-weighted Class Activation Mapping (Grad-CAM).

    Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"
    """
    
    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        """
        Initialize Grad-CAM.

        Args:
            model (keras.Model): Model to interpret
            layer_name (str): Name of convolutional layer (last by default)
        """
        self.model = model
        
        # Si no se especifica, buscar última capa convolucional
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        
        self.layer_name = layer_name
        
        # Crear modelo para extraer activaciones y gradientes
        self.grad_model = self._create_grad_model()
    
    def _find_last_conv_layer(self) -> str:
        """Find the last convolutional layer."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
        raise ValueError("No convolutional layer found")
    
    def _create_grad_model(self) -> keras.Model:
        """Create model to compute activations and gradients."""
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
        Compute the Grad-CAM heatmap for an image.

        Args:
            image (np.ndarray): Input image (no batch dimension)
            class_idx (int): Class index (None = predicted class)
            normalize (bool): Whether to normalize the heatmap

        Returns:
            np.ndarray: Grad-CAM heatmap
        """
        # Add batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            # Forward pass
            conv_outputs, predictions = self.grad_model(image)
            
            # If class not specified, use predicted class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Output de la clase target
            class_channel = predictions[:, class_idx]
        
        # Compute gradients of the class with respect to activations
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight activations by gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # ReLU (keep only positive values)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if normalize and heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        return heatmap
    
    def overlay_heatmap(self,
                       image: np.ndarray,
                       heatmap: np.ndarray,
                       alpha: float = 0.4,
                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay the heatmap on the original image.

        Args:
            image (np.ndarray): Original image
            heatmap (np.ndarray): Grad-CAM heatmap
            alpha (float): Heatmap transparency
            colormap (int): OpenCV colormap

        Returns:
            np.ndarray: Image with heatmap overlay
        """
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB with colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            colormap
        )
        
        # Convert to float
        heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
        
        # Ensure image is in [0, 1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # Overlay
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
        Visualize Grad-CAM for an image.

        Args:
            image (np.ndarray): Original image
            class_idx (int): Class index (None = predicted)
            save_path (str): Path to save visualization
            show_plot (bool): Whether to show the plot
            title (str): Custom title

        Returns:
            tuple: (heatmap, overlayed_image)
        """
        # Compute heatmap
        heatmap = self.compute_heatmap(image, class_idx)
        
        # Overlay heatmap
        overlayed = self.overlay_heatmap(image, heatmap)
        
        # Get prediction
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        predictions = self.model.predict(image_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Visualize
        if show_plot or save_path:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(image if image.max() <= 1 else image / 255.0)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Heatmap
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')

            # Overlay
            axes[2].imshow(overlayed)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            # Overall title
            if title is None:
                class_name = CLASS_NAMES_FULL.get(predicted_class, f"Class {predicted_class}")
                title = f"Prediction: {class_name} ({confidence:.2%})\nLayer: {self.layer_name}"

            fig.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to: {save_path}")

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
            print(f"Visualization saved to: {save_path}")

        plt.show()

        return heatmaps


def apply_gradcam_to_batch(model: keras.Model,
                           images: np.ndarray,
                           layer_name: Optional[str] = None,
                           save_dir: Optional[str] = None) -> list:
    """
    Apply Grad-CAM to a batch of images.

    Args:
        model: Model
        images: Batch of images
        layer_name: Convolutional layer
        save_dir: Directory to save visualizations

    Returns:
        list: List of heatmaps
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
    print("Testing Grad-CAM...")

    # Create dummy model
    from models.cnn_model import create_cnn_model, compile_model

    model = create_cnn_model()
    model = compile_model(model)

    # Dummy image
    dummy_image = np.random.rand(224, 224, 3).astype(np.float32)

    # Create Grad-CAM
    gradcam = GradCAM(model)
    print(f"Using layer: {gradcam.layer_name}")

    # Compute heatmap
    heatmap = gradcam.compute_heatmap(dummy_image)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # Visualize (without showing)
    heatmap, overlayed = gradcam.visualize(dummy_image, show_plot=False)
    print("✓ Grad-CAM working correctly")
