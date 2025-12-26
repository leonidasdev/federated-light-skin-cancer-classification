"""
Módulo de utilidades.
"""

from .metrics import (
    calculate_metrics,
    calculate_melanoma_metrics,
    calculate_per_class_metrics,
    compute_confusion_matrix,
    plot_confusion_matrix,
    print_classification_report,
    print_metrics_summary,
    save_metrics_to_file
)

from .logging_utils import (
    setup_logger,
    create_experiment_logger,
    log_system_info,
    log_experiment_config,
    log_training_progress,
    TensorBoardLogger,
    setup_federated_logging
)

from .security import (
    SecureChannel,
    SecureAggregation,
    DifferentialPrivacy,
    ClientAuthenticator,
    setup_secure_communication,
    hash_parameters
)

from .gradcam import (
    GradCAM,
    apply_gradcam_to_batch
)

__all__ = [
    # Metrics
    'calculate_metrics',
    'calculate_melanoma_metrics',
    'calculate_per_class_metrics',
    'compute_confusion_matrix',
    'plot_confusion_matrix',
    'print_classification_report',
    'print_metrics_summary',
    'save_metrics_to_file',
    
    # Logging
    'setup_logger',
    'create_experiment_logger',
    'log_system_info',
    'log_experiment_config',
    'log_training_progress',
    'TensorBoardLogger',
    'setup_federated_logging',
    
    # Security
    'SecureChannel',
    'SecureAggregation',
    'DifferentialPrivacy',
    'ClientAuthenticator',
    'setup_secure_communication',
    'hash_parameters',
    
    # Interpretability
    'GradCAM',
    'apply_gradcam_to_batch'
]
