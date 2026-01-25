"""
Federated Learning components using the Flower framework.

This module provides:
    - SkinCancerClient: Flower NumPyClient for local training
    - FLSimulator: Complete FL simulation infrastructure
    - SimulationConfig: Configuration dataclass for FL experiments
    - Server and strategy utilities for distributed deployment

Supported Non-IID Distributions:
    - natural: Each dataset becomes one client
    - dirichlet: Label distribution skew via Dirichlet sampling
    - label_skew: Artificial label imbalance
    - quantity_skew: Different sample counts per client
"""

from .client import SkinCancerClient, create_client
from .server import create_server, start_server
from .strategy import create_fedavg_strategy
from .simulation import SimulationConfig, FLSimulator, run_fl_simulation

__all__ = [
    # Client
    "SkinCancerClient",
    "create_client",
    # Server
    "create_server",
    "start_server",
    # Strategy
    "create_fedavg_strategy",
    # Simulation
    "SimulationConfig",
    "FLSimulator",
    "run_fl_simulation",
]
