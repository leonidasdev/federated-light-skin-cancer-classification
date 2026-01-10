"""
Federated client module.
"""

from .client import (
    FederatedClient,
    FederatedCallback,
    create_client,
    start_client
)

__all__ = [
    'FederatedClient',
    'FederatedCallback',
    'create_client',
    'start_client'
]
