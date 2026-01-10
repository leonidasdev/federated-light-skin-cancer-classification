"""
Evaluation metrics module for skin lesion classification.

Includes:
- Accuracy, Precision, Recall, F1
- Macro/micro AUC
- Sensitivity and specificity for melanoma
- Confusion matrix
- Per-class metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from config.config import CLASS_NAMES, CLASS_NAMES_FULL, METRICS_CONFIG


def calculate_metrics(y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      y_pred_proba: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate full set of classification metrics.

    Args:
        y_true (np.ndarray): True labels (indices or one-hot)
        y_pred (np.ndarray): Predictions (indices or one-hot)
        y_pred_proba (np.ndarray): Predicted probabilities (optional, for AUC)

    Returns:
        dict: Dictionary with computed metrics
    """
    # Convertir a índices si es one-hot
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_idx = np.argmax(y_true, axis=1)
    else:
        y_true_idx = y_true
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_idx = np.argmax(y_pred, axis=1)
    else:
        y_pred_idx = y_pred
    
    # Métricas básicas
    metrics = {
        'accuracy': accuracy_score(y_true_idx, y_pred_idx),
        'precision_macro': precision_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true_idx, y_pred_idx, average='micro', zero_division=0),
        'recall_macro': recall_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true_idx, y_pred_idx, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true_idx, y_pred_idx, average='micro', zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_true_idx, y_pred_idx)
    }
    
    # AUC (requiere probabilidades)
    if y_pred_proba is not None:
        try:
            # Convertir y_true a one-hot si no lo está
            if len(y_true.shape) == 1:
                from tensorflow.keras.utils import to_categorical
                num_classes = len(np.unique(y_true_idx))
                y_true_onehot = to_categorical(y_true_idx, num_classes)
            else:
                y_true_onehot = y_true
            
            metrics['auc_macro'] = roc_auc_score(y_true_onehot, y_pred_proba, average='macro', multi_class='ovr')
            metrics['auc_micro'] = roc_auc_score(y_true_onehot, y_pred_proba, average='micro', multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not compute AUC - {e}")
    
    # Métricas específicas para melanoma (si está configurado)
    if METRICS_CONFIG.get('binary_melanoma_metrics', False):
        melanoma_metrics = calculate_melanoma_metrics(y_true_idx, y_pred_idx)
        metrics.update(melanoma_metrics)
    
    return metrics


def calculate_melanoma_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate metrics specific to melanoma vs non-melanoma.

    Args:
        y_true (np.ndarray): True labels (indices)
        y_pred (np.ndarray): Predictions (indices)

    Returns:
        dict: Melanoma metrics
    """
    # Convertir a problema binario: melanoma (clase 0) vs resto
    melanoma_idx = METRICS_CONFIG.get('melanoma_class_index', 0)
    
    y_true_binary = (y_true == melanoma_idx).astype(int)
    y_pred_binary = (y_pred == melanoma_idx).astype(int)
    
    # Calcular métricas binarias
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall para melanoma
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision para melanoma
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        'melanoma_sensitivity': sensitivity,
        'melanoma_specificity': specificity,
        'melanoma_ppv': ppv,  # Positive Predictive Value
        'melanoma_npv': npv,  # Negative Predictive Value
        'melanoma_f1': f1_score(y_true_binary, y_pred_binary, zero_division=0)
    }


def calculate_per_class_metrics(y_true: np.ndarray, 
                                y_pred: np.ndarray) -> Dict[str, Dict]:
    """
    Calculate metrics for each class individually.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predictions

    Returns:
        dict: Per-class metrics
    """
    # Convertir a índices
    if len(y_true.shape) > 1:
        y_true_idx = np.argmax(y_true, axis=1)
    else:
        y_true_idx = y_true
    
    if len(y_pred.shape) > 1:
        y_pred_idx = np.argmax(y_pred, axis=1)
    else:
        y_pred_idx = y_pred
    
    # Calcular métricas por clase
    precision_per_class = precision_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
    recall_per_class = recall_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
    f1_per_class = f1_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
    
    # Organizar por clase
    per_class_metrics = {}
    for class_idx in range(len(precision_per_class)):
        class_name = CLASS_NAMES.get(class_idx, f"Class_{class_idx}")
        per_class_metrics[class_name] = {
            'precision': precision_per_class[class_idx],
            'recall': recall_per_class[class_idx],
            'f1': f1_per_class[class_idx]
        }
    
    return per_class_metrics


def compute_confusion_matrix(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             normalize: bool = False) -> np.ndarray:
    """
    Compute the confusion matrix.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predictions
        normalize (bool): Whether to normalize by row

    Returns:
        np.ndarray: Confusion matrix
    """
    # Convertir a índices
    if len(y_true.shape) > 1:
        y_true_idx = np.argmax(y_true, axis=1)
    else:
        y_true_idx = y_true
    
    if len(y_pred.shape) > 1:
        y_pred_idx = np.argmax(y_pred, axis=1)
    else:
        y_pred_idx = y_pred
    
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm


def plot_confusion_matrix(cm: np.ndarray, 
                         save_path: Optional[str] = None,
                         normalize: bool = False,
                         figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualize the confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix
        save_path (str): Path to save the figure (optional)
        normalize (bool): Whether to normalize
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Configurar formato
    fmt = '.2f' if normalize else 'd'
    
    # Crear heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=list(CLASS_NAMES.values()),
        yticklabels=list(CLASS_NAMES.values()),
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.ylabel('True Label')
    plt.xlabel('Prediction')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def print_classification_report(y_true: np.ndarray, 
                               y_pred: np.ndarray) -> str:
    """
    Generate classification report.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predictions

    Returns:
        str: Classification report
    """
    # Convertir a índices
    if len(y_true.shape) > 1:
        y_true_idx = np.argmax(y_true, axis=1)
    else:
        y_true_idx = y_true
    
    if len(y_pred.shape) > 1:
        y_pred_idx = np.argmax(y_pred, axis=1)
    else:
        y_pred_idx = y_pred
    
    # Generar reporte
    report = classification_report(
        y_true_idx,
        y_pred_idx,
        target_names=list(CLASS_NAMES_FULL.values()),
        digits=4
    )
    
    return report


def print_metrics_summary(metrics: Dict, title: str = "MÉTRICAS DE EVALUACIÓN"):
    """
    Print a summary of metrics.

    Args:
        metrics (dict): Dictionary of metrics
        title (str): Summary title
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    # General metrics
    print("\nGeneral Metrics:")
    general_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                      'auc_macro', 'cohen_kappa']
    for metric in general_metrics:
        if metric in metrics:
            print(f"  {metric.replace('_', ' ').title()}: {metrics[metric]:.4f}")
    
    # Melanoma metrics
    melanoma_metrics = {k: v for k, v in metrics.items() if 'melanoma' in k}
    if melanoma_metrics:
        print("\nMelanoma vs No-Melanoma Metrics:")
        for metric, value in melanoma_metrics.items():
            print(f"  {metric.replace('_', ' ').replace('melanoma ', '').title()}: {value:.4f}")
    
    print("=" * 60 + "\n")


# ==================== FUNCIONES DE UTILIDAD ====================

def calculate_roc_auc(y_true: np.ndarray, 
                     y_pred_proba: np.ndarray,
                     class_idx: Optional[int] = None) -> float:
    """
    Calculate ROC-AUC.

    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        class_idx (int): Specific class index (optional, for one-vs-rest)

    Returns:
        float: AUC score
    """
    # TODO: Implement per-class ROC-AUC calculation
    pass


def save_metrics_to_file(metrics: Dict, filepath: str):
    """
    Guarda métricas en archivo.
    
    Args:
        metrics (dict): Métricas a guardar
        filepath (str): Ruta del archivo
    """
    import json
    from pathlib import Path
    
    # Crear directorio si no existe
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar como JSON
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to: {filepath}")


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Testing metrics module...")
    
    # Crear datos de prueba
    from tensorflow.keras.utils import to_categorical
    
    np.random.seed(42)
    n_samples = 100
    n_classes = 7
    
    y_true = np.random.randint(0, n_classes, size=n_samples)
    y_pred = np.random.randint(0, n_classes, size=n_samples)
    
    y_true_onehot = to_categorical(y_true, n_classes)
    y_pred_proba = np.random.rand(n_samples, n_classes)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_onehot, y_pred_proba, y_pred_proba)
    print_metrics_summary(metrics, "METRICS TEST")
    
    # Matriz de confusión
    cm = compute_confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix:")
    print(cm)
    
    # Reporte de clasificación
    print("\n" + print_classification_report(y_true, y_pred))
