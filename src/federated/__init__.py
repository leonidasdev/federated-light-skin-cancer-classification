"""
Federated Learning components using Flower framework.
"""

from .client import SkinCancerClient, create_client
from .server import create_server, start_server
from .strategy import create_fedavg_strategy

__all__ = [
    "SkinCancerClient",
    "create_client",
    "create_server",
    "start_server",
    "create_fedavg_strategy"
]
