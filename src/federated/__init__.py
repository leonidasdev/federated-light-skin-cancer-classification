"""
Federated Learning Module
=========================

Federated learning server and client implementations.
"""

from .server import FederatedServer
from .client import FederatedClient
from .aggregation import (
    federated_averaging,
    weighted_averaging,
    fedprox_aggregation,
)
from .strategies import (
    FedAvg,
    FedProx,
    FedNova,
    Scaffold,
)

__all__ = [
    "FederatedServer",
    "FederatedClient",
    "federated_averaging",
    "weighted_averaging",
    "fedprox_aggregation",
    "FedAvg",
    "FedProx",
    "FedNova",
    "Scaffold",
]
