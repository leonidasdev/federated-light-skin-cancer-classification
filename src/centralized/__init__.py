"""
Centralized training package.

This package exposes the centralized training trainer and config used
for the non-federated (centralized) baseline experiments.
"""

from .centralized import (
    CentralizedConfig,
    CentralizedTrainer,
    run_centralized_training,
)

__all__ = [
    "CentralizedConfig",
    "CentralizedTrainer",
    "run_centralized_training",
]
