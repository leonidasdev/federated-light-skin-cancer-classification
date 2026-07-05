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

## Outputs

The project stores its generated artifacts under `outputs/`.

- [outputs/README.md](../outputs/README.md) explains the full output hierarchy.
- Training runs live under `dscatnet_*` folders.
- Evaluation exports live under `evaluation_dscatnet_*` folders.
- Multi-model comparisons and convergence analyses live under `evaluation_comparison_dscatnet_*` folders.
- The most useful research artifacts are `results_table.csv`, `all_datasets_comparison.csv`, `per_class_metrics.csv`, `confusion_matrix.csv`, `bootstrap_gap_table.csv`, `communication_efficiency.csv`, and the comparison PNG figures.

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

Note: `02_model_evaluation.ipynb` exports a timestamped JSON and `results_latest.json` containing per-sample fields used by Notebook 03 for paired statistical testing: `labels`, `predictions`, `sample_ids`, `sample_predictions`, plus aggregated `metrics` and `per_class_metrics`.

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
