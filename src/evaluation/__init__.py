# =============================================================================
# Evaluation Module
# =============================================================================
"""
Evaluation Module.

Provides metrics computation and visualization for model evaluation.
"""

# =============================================================================
# Metrics Imports
# =============================================================================

from .metrics import (
    EvaluationResults,
    ModelEvaluator,
    evaluate_model,
    compute_federated_metrics,
    compare_results,
    print_comparison,
)

# =============================================================================
# Visualization Imports
# =============================================================================

from .visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_client_comparison,
    plot_noniid_distribution,
    plot_fl_vs_centralized,
    plot_communication_cost,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Metrics
    "EvaluationResults",
    "ModelEvaluator",
    "evaluate_model",
    "compute_federated_metrics",
    "compare_results",
    "print_comparison",
    # Visualization
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_client_comparison",
    "plot_noniid_distribution",
    "plot_fl_vs_centralized",
    "plot_communication_cost",
]
