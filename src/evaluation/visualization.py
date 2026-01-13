"""
Visualization Utilities.

Plotting and visualization for training curves, confusion matrices,
and other analysis outputs.
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("matplotlib/seaborn not installed. Plotting functions disabled.")


def check_plotting_available() -> bool:
    """Check if plotting libraries are available."""
    if not HAS_PLOTTING:
        logger.warning("Plotting not available. Install matplotlib and seaborn.")
        return False
    return True


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Training Progress",
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'.
        save_path: Path to save the figure.
        title: Plot title.
    """
    if not check_plotting_available():
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    
    # Loss plot
    ax1 = axes[0]
    if "train_loss" in history:
        ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch/Round", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curves", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = axes[1]
    if "train_accuracy" in history:
        ax2.plot(epochs, history["train_accuracy"], "b-", label="Train Accuracy", linewidth=2)
    if "val_accuracy" in history:
        ax2.plot(epochs, history["val_accuracy"], "r-", label="Val Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch/Round", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy Curves", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved training curves to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array.
        class_names: List of class names.
        save_path: Path to save the figure.
        title: Plot title.
        normalize: Whether to normalize the matrix.
    """
    if not check_plotting_available():
        return
    
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_client_comparison(
    client_metrics: Dict[int, Dict[str, float]],
    metric_name: str = "accuracy",
    save_path: Optional[Path] = None,
    title: str = "Client Performance Comparison",
) -> None:
    """
    Plot bar chart comparing metrics across clients.
    
    Args:
        client_metrics: Dictionary mapping client_id to metrics dict.
        metric_name: Name of metric to compare.
        save_path: Path to save the figure.
        title: Plot title.
    """
    if not check_plotting_available():
        return
    
    clients = sorted(client_metrics.keys())
    values = [client_metrics[c].get(metric_name, 0) for c in clients]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cmap = plt.get_cmap("tab10")
    bars = ax.bar(
        [f"Client {c}" for c in clients],
        values,
        color=[cmap(i) for i in range(len(clients))],
        edgecolor="black",
        linewidth=1,
    )
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    ax.set_xlabel("Client", fontsize=12)
    ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved client comparison to {save_path}")
    
    plt.show()


def plot_noniid_distribution(
    client_distributions: Dict[int, Dict[int, int]],
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "Data Distribution Across Clients",
) -> None:
    """
    Plot stacked bar chart showing class distribution per client.
    
    Args:
        client_distributions: Dict mapping client_id to {class_idx: count}.
        class_names: List of class names.
        save_path: Path to save the figure.
        title: Plot title.
    """
    if not check_plotting_available():
        return
    
    num_clients = len(client_distributions)
    num_classes = len(class_names)
    
    # Prepare data
    clients = sorted(client_distributions.keys())
    data = np.zeros((num_clients, num_classes))
    
    for i, client_id in enumerate(clients):
        dist = client_distributions[client_id]
        for class_idx, count in dist.items():
            if class_idx < num_classes:
                data[i, class_idx] = count
    
    # Normalize to percentages
    totals = data.sum(axis=1, keepdims=True)
    data_pct = np.divide(data, totals, where=totals != 0) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(num_clients)
    width = 0.6
    bottom = np.zeros(num_clients)
    
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(num_classes)]
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        ax.bar(x, data_pct[:, i], width, bottom=bottom, label=class_name, color=color)
        bottom += data_pct[:, i]
    
    ax.set_xlabel("Client", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {c}" for c in clients])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved distribution plot to {save_path}")
    
    plt.show()


def plot_fl_vs_centralized(
    fl_history: Dict[str, List[float]],
    centralized_history: Dict[str, List[float]],
    metric: str = "val_accuracy",
    save_path: Optional[Path] = None,
    title: str = "Federated vs Centralized Learning",
) -> None:
    """
    Compare FL and centralized training curves.
    
    Args:
        fl_history: History from federated training.
        centralized_history: History from centralized training.
        metric: Metric to compare.
        save_path: Path to save the figure.
        title: Plot title.
    """
    if not check_plotting_available():
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if metric in fl_history:
        rounds = range(1, len(fl_history[metric]) + 1)
        ax.plot(rounds, fl_history[metric], "b-", label="Federated", linewidth=2, marker="o", markersize=3)
    
    if metric in centralized_history:
        epochs = range(1, len(centralized_history[metric]) + 1)
        ax.plot(epochs, centralized_history[metric], "r-", label="Centralized", linewidth=2, marker="s", markersize=3)
    
    ax.set_xlabel("Epoch / Round", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if "accuracy" in metric:
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {save_path}")
    
    plt.show()


def plot_communication_cost(
    rounds: List[int],
    cumulative_cost_mb: List[float],
    save_path: Optional[Path] = None,
    title: str = "Cumulative Communication Cost",
) -> None:
    """
    Plot communication cost over rounds.
    
    Args:
        rounds: List of round numbers.
        cumulative_cost_mb: Cumulative communication cost in MB.
        save_path: Path to save the figure.
        title: Plot title.
    """
    if not check_plotting_available():
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Calculate cumulative
    cumulative = np.cumsum(cumulative_cost_mb)
    
    ax.fill_between(rounds, 0, cumulative, alpha=0.3, color="blue")
    ax.plot(rounds, cumulative, "b-", linewidth=2)
    
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Cumulative Cost (MB)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Add annotation for final cost
    final_cost = cumulative[-1] if len(cumulative) > 0 else 0
    ax.annotate(
        f"Total: {final_cost:.1f} MB",
        xy=(rounds[-1], final_cost),
        xytext=(rounds[-1] * 0.8, final_cost * 1.1),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved communication cost plot to {save_path}")
    
    plt.show()
