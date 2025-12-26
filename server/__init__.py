"""
Módulo del servidor federado.
"""

from .server import (
    FederatedServer,
    CustomFedAvg,
    SecureAggregationStrategy,
    create_server
)

__all__ = [
    'FederatedServer',
    'CustomFedAvg',
    'SecureAggregationStrategy',
    'create_server'
]
