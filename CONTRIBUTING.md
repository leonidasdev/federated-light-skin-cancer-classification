# Contributing to Federated Skin Cancer Classification

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)

---

## Code of Conduct

Please be respectful and constructive in all interactions. This project follows standard open-source community guidelines.

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/federated-light-skin-cancer-classification.git
   cd federated-light-skin-cancer-classification
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### Prerequisites

- Python 3.10+
- pip or conda
- Git

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run tests
python run_tests.py --test

# Run linter
ruff check .
```

---

## Code Style

This project follows these conventions:

### Python Style

- **PEP 8** compliance (enforced by `ruff`)
- **Line length**: 120 characters maximum
- **Imports**: Organized by stdlib, third-party, local (ruff handles this)

### File Organization

Each Python file should follow this structure:

```python
# =============================================================================
# Module Title
# =============================================================================
"""
Module docstring describing purpose and contents.
"""

# =============================================================================
# Imports
# =============================================================================

import stdlib_module
import third_party_module
from local_module import function

# =============================================================================
# Constants
# =============================================================================

CONSTANT_VALUE = 42

# =============================================================================
# Classes
# =============================================================================

class MyClass:
    """Class docstring."""
    pass

# =============================================================================
# Functions
# =============================================================================

def my_function():
    """Function docstring."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int = 0) -> bool:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 0.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When param1 is empty.
    """
```

### Test Docstrings

Use imperative style for test docstrings:

```python
def test_something():
    """Test that something works correctly."""  # Imperative
    # NOT: "Tests that..." or "Testing..."
```

---

## Testing

### Running Tests

```bash
# Run all tests
python run_tests.py --test

# Run with verbose output
python run_tests.py --verbose

# Run with coverage report
python run_tests.py --coverage

# Run specific test file
pytest tests/test_simulation.py -v

# Run tests matching pattern
pytest -k "test_client" -v

# Skip slow tests
pytest -m "not slow" -v
```

### Testing Strategy

The project maintains a comprehensive test suite to ensure code quality, correctness, and reliability. Our testing philosophy includes:

- **Unit Tests**: Each module in `src/` should have a corresponding `test_*.py` file in `tests/` that tests its functions and classes in isolation.
- **Integration Tests**: We test how different components work together, such as the full training pipeline (`test_centralized.py::test_run_with_synthetic_data`) and end-to-end federated simulations (`test_integration.py`).
- **Fixtures**: Common test resources (like mock data, configurations, and paths) are defined as fixtures in `tests/conftest.py` to promote code reuse and simplify test setup.
- **Markers**: Tests are categorized using pytest markers (`@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.gpu`) to allow for selective test execution.
- **Synthetic Data**: Core logic (training, evaluation, etc.) is tested on small, in-memory synthetic datasets to ensure tests run quickly and do not require downloading the full datasets.

### Writing Tests

1. Place tests in `tests/` directory
2. Name test files `test_*.py`
3. Name test functions `test_*`
4. Use pytest fixtures from `conftest.py`
5. Add appropriate markers:
   - `@pytest.mark.slow` - Long-running tests
   - `@pytest.mark.integration` - Integration tests
   - `@pytest.mark.gpu` - GPU-required tests

### Test Structure

```python
class TestClassName:
    """Tests for ClassName functionality."""
    
    def test_feature_one(self, fixture_name):
        """Test that feature one works correctly."""
        # Arrange
        expected = "value"
        
        # Act
        result = function_under_test()
        
        # Assert
        assert result == expected
```

---

## CI/CD (Continuous Integration)

This project uses **GitHub Actions** for continuous integration. A CI pipeline is automatically triggered for every pull request and push to the `main` branch.

The CI workflow, defined in `.github/workflows/ci.yml`, performs the following checks:

1.  **Linting**: Runs `ruff check .` to enforce PEP 8 compliance and code style.
2.  **Testing**: Executes the full test suite using `pytest` (excluding tests marked as `gpu` or `integration`).

All checks must pass before a pull request can be merged. This ensures that the `main` branch always remains stable and that all contributions adhere to the project's quality standards.

---

## Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Run the full test suite**:
   ```bash
   python run_tests.py --test
   ruff check .
   ```
4. **Update CHANGELOG** (if applicable)
5. **Submit PR** with clear description

### PR Checklist

- [ ] Tests pass locally
- [ ] Linter passes (`ruff check .`)
- [ ] Documentation updated (if needed)
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

---

## Documentation

### Code Documentation

- All public functions need docstrings
- Complex logic should have inline comments
- Use type hints for function signatures

### Project Documentation

- Update `README.md` for user-facing changes
- Update `docs/CLAUDE.md` for AI agent context
- Update `docs/config-options-guide.md` for config changes
- Add new files to appropriate docs folder

---

## Project Structure Overview

```
federated-light-skin-cancer-classification/
├── src/                    # Main source code
│   ├── centralized/        # Centralized training
│   ├── data/               # Data loading and preprocessing
│   ├── evaluation/         # Metrics and visualization
│   ├── federated/          # FL client, server, simulation
│   ├── models/             # DSCATNet architecture
│   └── utils/              # Helpers, logging, checkpoints
├── tests/                  # Test files
├── configs/                # YAML configuration files
├── notebooks/              # Jupyter notebooks
├── docs/                   # Documentation
├── run_experiment.py       # Main experiment runner
├── run_download.py         # Dataset downloader
└── run_tests.py            # Test runner
```

---

## Questions?

If you have questions about contributing, please open an issue on GitHub.
