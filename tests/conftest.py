# =============================================================================
# Pytest Configuration and Shared Fixtures
# =============================================================================
"""
Shared pytest fixtures and configuration for the test suite.

This module provides:
- Path setup (auto-loaded by pytest)
- Shared fixtures for common test resources
- Custom pytest markers (slow, integration, gpu)

Note:
    Fixtures defined here are automatically available to all tests.
    No need to import them explicitly.
"""

# =============================================================================
# Imports
# =============================================================================

import sys
from pathlib import Path

import pytest

# =============================================================================
# Path Configuration
# =============================================================================

# Add project root to path for all tests
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =============================================================================
# Pytest Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring datasets or external resources"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture
def project_root_path():
    """Get the project root path."""
    return project_root


@pytest.fixture
def data_root_path(project_root_path):
    """Get the data directory path."""
    return project_root_path / "data"


@pytest.fixture
def outputs_path(project_root_path):
    """Get the outputs directory path."""
    return project_root_path / "outputs"


@pytest.fixture
def configs_path(project_root_path):
    """Get the configs directory path."""
    return project_root_path / "configs"


@pytest.fixture
def mock_labels():
    """Generate mock labels for testing data splits."""
    import numpy as np
    
    def _generate(n_samples=10000, n_classes=7, imbalanced=True):
        if imbalanced:
            # Simulate imbalanced dermoscopy distribution
            weights = [0.05, 0.08, 0.12, 0.02, 0.10, 0.55, 0.08]
            labels = np.random.choice(n_classes, size=n_samples, p=weights)
        else:
            labels = np.random.randint(0, n_classes, size=n_samples)
        return labels.tolist()
    
    return _generate


@pytest.fixture
def sample_config():
    """Create a minimal sample configuration for testing."""
    return {
        "experiment": {
            "name": "test_experiment",
            "description": "Test configuration",
        },
        "model": {
            "name": "DSCATNet",
            "variant": "tiny",
            "image_size": 224,
            "num_classes": 7,
        },
        "training": {
            "batch_size": 4,
            "lr": 0.001,
            "local_epochs": 1,
            "num_rounds": 2,
        },
        "federation": {
            "num_clients": 2,
            "participation": 1.0,
        },
    }
