"""
Visualization Utilities
=======================

Plotting functions for results visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        title: Plot title
        normalize: Whether to normalize values
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss curves
    axes[0].plot(history.get('train_loss', []), label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history.get('train_acc', []), label='Train Acc')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_federated_rounds(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Federated Learning Progress",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot federated learning round metrics.
    
    Args:
        history: Dictionary with 'global_loss', 'global_accuracy'
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    rounds = list(range(1, len(history.get('global_loss', [])) + 1))
    
    # Loss
    axes[0].plot(rounds, history.get('global_loss', []), 'b-o', markersize=4)
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Global Loss')
    axes[0].set_title('Global Model Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(rounds, history.get('global_accuracy', []), 'g-o', markersize=4)
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Global Accuracy')
    axes[1].set_title('Global Model Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_data_distribution(
    client_distributions: Dict[int, Dict[int, int]],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Plot data distribution across federated clients.
    
    Args:
        client_distributions: Dict mapping client_id -> {class_id: count}
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # TODO: Implement data distribution visualization
    raise NotImplementedError("Data distribution plotting pending")
