# Documentation Index

Welcome to the DSCATNet Federated Learning documentation!

## Quick Navigation

| Document | Description |
|----------|-------------|
| [Main README](../README.md) | Project overview, installation, and quick start guide |
| [Configuration Guide](config-options-guide.md) | Comprehensive YAML configuration reference |
| [Contributing Guide](../CONTRIBUTING.md) | Code style, testing, and PR process |
| [Architecture Overview](architecture.md) | System design and module documentation |
| [Benchmark Comparison](benchmark-comparison.md) | Federated vs centralized fairness audit |

## For AI Assistants

If you're an AI assistant (Claude, GPT, etc.), please read [CLAUDE.md](CLAUDE.md) in this directory for comprehensive context about this codebase.

## Getting Started

1. **Installation**: See [README.md](../README.md#installation)
2. **Dataset Setup**: See [README.md](../README.md#dataset-setup)
3. **Running Experiments**: See [README.md](../README.md#training-pipeline)

## Configuration

All experiments use YAML configuration files in `configs/`:

- **Full Reference**: [config-options-guide.md](config-options-guide.md)
- **Templates**: Available in `configs/templates/`
- **Validation**: `python src/utils/config_schema.py <config.yaml>`

## Development

- **Code Style**: [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Testing**: `pytest tests/ -v` or `python run_tests.py`
- **Architecture**: [architecture.md](architecture.md)

## Notebooks

Interactive Jupyter notebooks for exploration, evaluation, and statistical analysis:

| Notebook | Description |
|----------|-------------|
| [01_dataset_exploration](../notebooks/01_dataset_exploration.ipynb) | Dataset analysis, verification, and visualization |
| [02_model_evaluation](../notebooks/02_model_evaluation.ipynb) | Model evaluation, metrics export, and per-sample prediction data |
| [03_fl_vs_centralized_comparison](../notebooks/03_fl_vs_centralized_comparison.ipynb) | FL vs centralized comparison with exact statistical testing |

See [README.md - Notebooks](../README.md#notebooks) for detailed export schema and usage instructions.

## Project Structure

```
federated-light-skin-cancer-classification/
├── src/                    # Source code modules
│   ├── models/             # DSCATNet architecture
│   ├── federated/          # FL simulation components
│   ├── centralized/        # Baseline training
│   ├── data/               # Dataset classes and preprocessing
│   ├── evaluation/         # Metrics and visualization
│   └── utils/              # Helpers, checkpoints, config validation
├── configs/                # YAML configuration files
│   └── templates/          # Configuration templates
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit and integration tests (457 selected tests)
├── docs/                   # Documentation (you are here)
├── data/                   # Dataset files (not in git)
└── outputs/                # Experiment outputs (auto-generated)
```
