"""
Federated Learning for Skin Cancer Classification with DSCATNet.

This package provides the complete implementation of DSCATNet adapted
for federated learning using the Flower framework.
"""

__version__ = "0.1.0"
__author__ = "Leonardo Chen"

# Make key modules accessible at package level
from . import models
from . import federated
from . import data
from . import training
from . import evaluation
from . import utils
