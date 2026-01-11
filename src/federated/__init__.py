"""
Federated Learning Module
=========================

Flower-based federated learning for LMS-ViT skin cancer classification.

This module provides:
- LMSViTFlowerClient: NumPy-based client for Flower framework
- create_client_fn: Factory function for creating clients in simulation
- run_flower_server: Start a Flower gRPC server for distributed training
- run_flower_simulation: Run local simulation with Ray
- create_strategy: Create Flower aggregation strategies (FedAvg, FedProx, etc.)
"""

from .flower_client import LMSViTFlowerClient, create_client_fn
from .flower_server import (
    run_flower_server,
    run_flower_simulation,
    create_strategy,
    get_initial_parameters,
    print_history,
    weighted_average,
    fit_metrics_aggregation_fn,
    evaluate_metrics_aggregation_fn,
)

__all__ = [
    # Flower Client
    "LMSViTFlowerClient",
    "create_client_fn",
    # Flower Server
    "run_flower_server",
    "run_flower_simulation",
    "create_strategy",
    "get_initial_parameters",
    "print_history",
    # Metrics aggregation
    "weighted_average",
    "fit_metrics_aggregation_fn",
    "evaluate_metrics_aggregation_fn",
]
