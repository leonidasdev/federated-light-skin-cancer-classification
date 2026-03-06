# Project Information for AI Assistants

> **Purpose**: This document provides comprehensive context for AI models (Claude, GPT, etc.) to understand, modify, and extend this codebase effectively. Read this first before making changes.

## Related Documentation

| Document | Description |
|----------|-------------|
| [README.md](../README.md) | User-facing documentation, installation, usage |
| [config-options-guide.md](config-options-guide.md) | Complete YAML configuration reference |
| [architecture.md](architecture.md) | System architecture and key classes |
| [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md) | Federated vs centralized benchmark fairness audit |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Code style and contribution guidelines |

---

## Project Overview

**Name**: Federated Learning for Skin Cancer Classification with DSCATNet  
**Type**: Thesis Research Project  
**Domain**: Medical Image Classification, Federated Learning, Vision Transformers  
**Language**: Python 3.10+  
**Framework Stack**: PyTorch 2.7+, Flower 1.25+ (FL), scikit-learn, PIL/Pillow

### Core Objective

Evaluate whether a lightweight Vision Transformer (DSCATNet) can maintain classification accuracy when trained in a **federated learning** setting with **non-IID data** across simulated hospital clients, compared to centralized training baselines.

### Reference Papers

| Paper | Authors | Published | Key Contribution |
|-------|---------|-----------|------------------|
| DSCATNet | Yadav et al. | PLOS ONE, Dec 2024 | Dual-Scale Cross-Attention ViT for skin cancer |
| FL Evaluation | Khullar et al. | Scientific Reports, Jan 2025 | FL benchmark with EfficientNetV2S on ISIC 2019 |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTRY POINTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  run_experiment.py    Main CLI: centralized, federated, evaluate, compare   │
│  run_download.py      Dataset download and verification                     │
│  run_tests.py         Test suite runner                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONFIGURATION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  configs/*.yaml       YAML configs with nested structure                    │
│  CentralizedConfig    @dataclass in src/centralized/centralized.py          │
│  SimulationConfig     @dataclass in src/federated/simulation.py             │
│                                                                             │
│  Priority: CLI args > YAML config > dataclass defaults                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CORE MODULES                                   │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│   src/models/        │   src/federated/     │   src/centralized/            │
│   ├─ dscatnet.py     │   ├─ simulation.py   │   └─ centralized.py           │
│   ├─ cross_attention │   ├─ client.py       │       CentralizedTrainer      │
│   └─ patch_embedding │   ├─ server.py       │       - setup_data()          │
│                      │   └─ strategy.py     │       - train_epoch()         │
│   DSCATNet Model     │   FLSimulator        │       - evaluate()            │
│   ~29.4M params      │   FedAvg aggregation │       - save/load_checkpoint()│
│   (small variant)    │                      │                               │
└──────────────────────┴──────────────────────┴───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  src/data/datasets.py      HAM10000, ISIC2018/2019/2020, PADUFES20 classes  │
│  src/data/preprocessing.py  Transforms: get_train_transforms, get_val_...   │
│  src/data/splits.py         IID/Non-IID splitting (Dirichlet, label_skew)   │
│  src/data/download.py       ISIC API / Kaggle / Mendeley downloader         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION & UTILS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  src/evaluation/metrics.py       ModelEvaluator, EvaluationResults          │
│  src/evaluation/visualization.py  Plotting functions                        │
│  src/utils/checkpoints.py        CheckpointManager                          │
│  src/utils/helpers.py            Seed, device, formatting utilities         │
│  src/utils/logging_utils.py      Logging configuration                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Style & Conventions

### Python Style

- **Formatting**: PEP 8 compliant, 120-character line limit (configured in `pyproject.toml`)
- **Linter**: Ruff (rules: E, F, W, I)
- **Type Hints**: All function signatures include type hints
- **Docstrings**: Google-style docstrings for all public functions/classes
- **Imports**: Grouped (stdlib → third-party → local), absolute imports preferred

```python
# Example function signature style
def train_epoch(
    self,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        optimizer: PyTorch optimizer instance.
        criterion: Loss function module.

    Returns:
        Tuple of (average loss, accuracy).
    """
```

### Configuration Pattern

All configurable components use `@dataclass` with:
- Default values for all fields
- `to_dict()` method for JSON serialization
- `from_dict()` classmethod for deserialization

```python
@dataclass
class SimulationConfig:
    num_rounds: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    optimizer_type: str = "adam"           # adam, adamw
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1  # Effective BS = batch_size × steps

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
```

### YAML Config Structure

Configs use nested structure that gets flattened when loading:

```yaml
federated:
  experiment:
    name: experiment_name
  model:
    variant: small
    image_size: 224
  training:
    batch_size: 8
    lr: 0.001
    optimizer: adam
    weight_decay: 0.0
    gradient_accumulation_steps: 4
    scheduler: none
    use_amp: false
  federation:
    num_clients: 4  # Adjust based on number of datasets used
    noniid_type: dirichlet
```

---

## Key Classes & Their Responsibilities

### `CentralizedTrainer` (src/centralized/centralized.py)

**Purpose**: Standard PyTorch training loop for baseline comparison

**Key Methods**:
- `setup_data()`: Loads datasets, creates DataLoaders
- `train_epoch()`: Single epoch training with gradient accumulation and optional AMP
- `evaluate()`: Validation with per-class metrics
- `save_checkpoint()` / `load_checkpoint()`: Full state persistence
- `run()`: Main training loop with early stopping

**Training Features**:
- Gradient accumulation (effective BS = `batch_size × gradient_accumulation_steps`)
- Configurable optimizer (`adam` or `adamw`)
- Configurable scheduler (`none`, `cosine`, `plateau`)
- Optional AMP (disabled by default for stability)

**Checkpoint Contents**: model_state_dict, optimizer_state_dict, scheduler_state_dict, scaler_state_dict, history, best_val_accuracy

### `FLSimulator` (src/federated/simulation.py)

**Purpose**: Orchestrates federated learning simulation

**Key Methods**:
- `setup_clients()`: Routes to natural or Dirichlet non-IID setup
- `setup_natural_noniid()`: Each dataset = one client
- `setup_dirichlet_noniid()`: Split combined data via Dirichlet distribution
- `train_client()`: Local training with gradient accumulation and clipping
- `evaluate_client()`: Evaluate on client's validation data
- `aggregate_parameters()`: FedAvg weighted averaging
- `run_round()`: Single FL round (train all → aggregate → evaluate)
- `load_checkpoint()`: Restore model weights and training state
- `run()`: Main FL loop with resume support

**Checkpoint Contents**: model_state_dict, round, config, history, best_val_accuracy, best_round, rounds_without_improvement, metrics

**Resume Behavior**: When resuming without `--config`, `run_experiment.py` loads config from checkpoint to preserve original settings (noniid_type, datasets, hyperparameters). CLI args can still override specific values.

### `ModelEvaluator` (src/evaluation/metrics.py)

**Purpose**: Comprehensive model evaluation

**Returns**: `EvaluationResults` dataclass with accuracy, balanced_accuracy, precision_macro, recall_macro, f1_macro, f1_weighted, auc_macro, confusion_matrix, per_class_metrics

### `create_dscatnet()` (src/models/dscatnet.py)

**Purpose**: Factory function for DSCATNet model

**Variants**:
- `tiny`: embed_dim=192, depth=4, heads=3 (~5M params)
- `small`: embed_dim=384, depth=6, heads=6 (~29.4M params) **[DEFAULT]**
- `base`: embed_dim=384, depth=8, heads=6 (~39M params)

---

## Paper-Aligned Hyperparameters

The training configuration is aligned with the DSCATNet paper (Yadav et al., PLOS ONE, 2024):

| Parameter | Paper Value | Config Value | Notes |
|-----------|-------------|--------------|-------|
| Optimizer | Adam | `optimizer: adam` | Not AdamW |
| Learning Rate | 0.001 | `lr: 0.001` | Fixed LR throughout training |
| Weight Decay | 0.0 | `weight_decay: 0.0` | Paper uses Adam without L2 penalty |
| Effective Batch Size | 32 | `batch_size: 8` × `gradient_accumulation_steps: 4` | Fits 4GB VRAM |
| LR Scheduler | None | `scheduler: none` | Paper uses fixed LR |
| Epochs | 200 | `epochs: 200` | Full training run |
| AMP | Not used | `use_amp: false` | Disabled for stability |
| Image Size | 224 | `image_size: 224` | Standard ViT input |
| Augmentation | None | `augmentation: none` (HAM10000) | Paper reports no augmentation |

---

## Dataset Classes & Label Mapping

### Unified 7-Class Schema

All datasets map to:

| Index | Abbreviation | Full Name |
|-------|--------------|-----------|
| 0 | AK/AKIEC | Actinic Keratosis |
| 1 | BCC | Basal Cell Carcinoma |
| 2 | BKL | Benign Keratosis |
| 3 | DF | Dermatofibroma |
| 4 | MEL | Melanoma |
| 5 | NV | Melanocytic Nevus |
| 6 | VASC | Vascular Lesion |

### Dataset Classes (src/data/datasets.py)

Each inherits from `torch.utils.data.Dataset`:
- `HAM10000Dataset`: 10,015 images, 7 classes
- `ISIC2018Dataset`: ~10,015 images, 7 classes
- `ISIC2019Dataset`: ~25,331 images, 8+UNK → filtered to 7
- `ISIC2020Dataset`: ~33,126 images, binary → mapped to MEL/NV
- `PADUFES20Dataset`: 2,298 images, 6 classes

### DatasetSubset (src/data/datasets.py)

Wrapper for applying different transforms to train/val splits:

```python
train_ds = DatasetSubset(full_dataset, train_indices, train_transform)
val_ds = DatasetSubset(full_dataset, val_indices, val_transform)
```

---

## Testing Strategy

### Test Structure

```
tests/
├── conftest.py            # Shared pytest fixtures
├── test_centralized.py    # CentralizedConfig, CentralizedTrainer
├── test_checkpoints.py    # Checkpoint saving/loading
├── test_cli.py            # CLI argument parsing and validation
├── test_config_loading.py # YAML config loading and schema validation
├── test_datasets.py       # Dataset registry and loading functions
├── test_download.py       # Download functionality tests
├── test_evaluation.py     # EvaluationResults, metrics computation
├── test_helpers.py        # Helpers (set_seed, get_device, etc.)
├── test_model_evaluator.py# ModelEvaluator integration tests
├── test_models.py         # DSCATNet model architecture tests
├── test_preprocessing.py  # Transforms, augmentation levels
├── test_simulation.py     # SimulationConfig, FLSimulator, FedAvg
├── test_splits.py         # IID/Non-IID splitting utilities
└── test_strategy.py       # DSCATNetFedAvg strategy tests
```

### Running Tests

```bash
# All tests (fast, uses mocks)
pytest tests/ -v

# With coverage
pytest --cov=src --cov-report=term-missing tests/

# Specific module
pytest tests/test_simulation.py -v

# Run slow integration tests
pytest -m slow tests/ -v
```

### Test Conventions

- Unit tests use mocked data (no real datasets required)
- Integration tests marked with `@pytest.mark.slow` (deselected by default)
- Fixtures in `conftest.py` for common setup
- Assert both return values and side effects (file creation, etc.)

---

## Common Modification Patterns

### Adding a New CLI Flag

1. Add argument to `argparse` in `run_experiment.py`:
```python
parser.add_argument("--new-flag", type=int, help="Description")
```

2. Add field to relevant `@dataclass` config:
```python
@dataclass
class SimulationConfig:
    new_flag: int = 10  # Default value
```

3. Add override logic in `run_centralized()` or `run_federated()`:
```python
if args.new_flag is not None:
    config.new_flag = args.new_flag
```

4. Add mapping in YAML config flattening if nested:
```python
if "new_flag" in train:
    flat_config["new_flag"] = train["new_flag"]
```

### Adding a New Dataset

1. Create class in `src/data/datasets.py` inheriting from `Dataset`
2. Implement `__init__`, `__len__`, `__getitem__`, `labels` property
3. Add label mapping to unified 7-class schema
4. Add to `all_dataset_classes` list in:
   - `CentralizedTrainer.setup_data()`
   - `FLSimulator.setup_natural_noniid()`
   - `FLSimulator.setup_dirichlet_noniid()`
   - `run_evaluate()` in `run_experiment.py`
5. Add to `--datasets` choices in argparse

### Modifying Checkpoint Contents

Both `CentralizedTrainer` and `FLSimulator` have:
- `save_checkpoint()`: Add new fields to the dict
- `load_checkpoint()`: Restore new fields with fallback defaults

```python
# In save_checkpoint:
checkpoint = {
    "existing_field": ...,
    "new_field": self.new_state,
}

# In load_checkpoint:
if "new_field" in checkpoint:
    self.new_state = checkpoint["new_field"]
```

---

## Output Structure

```
outputs/
└── {mode}_{timestamp}/
    ├── checkpoints/
    │   ├── best_model.pt           # Weights only
    │   ├── best_checkpoint.pt      # Full state (centralized)
    │   └── checkpoint_{epoch/round}_N.pt
    ├── config.json                 # Serialized config
    ├── results.json                # Final metrics
    ├── history.json                # Training curves data
    └── experiment.log              # Full logs
```

---

## Error Handling Patterns

### Dataset Loading

```python
try:
    dataset = DatasetClass(root_dir=..., csv_path=..., transform=...)
except Exception as e:
    logger.warning(f"Failed loading {name}: {e}")
    continue  # Skip, don't crash
```

### Checkpoint Loading

```python
if resume_path.exists():
    start_epoch = self.load_checkpoint(str(resume_path), optimizer, scheduler) + 1
else:
    logger.warning(f"Checkpoint not found at {resume_path}, starting from scratch")
```

### Graceful Degradation

- Missing datasets: Skip with warning, continue with available
- Missing optional config fields: Use dataclass defaults
- AMP not available: Fallback to FP32 training

---

## Performance Considerations

### Memory Management

- `batch_size=8` default (fits 4GB VRAM with small variant)
- Gradient accumulation (`gradient_accumulation_steps=4`) for effective BS=32
- `num_workers=4` for data loading (Windows: may need `num_workers=0`)
- AMP disabled by default for training stability

### Bottlenecks

1. **Data Loading**: Large datasets → use SSD
2. **FL Communication**: ~112MB model params per round (small variant, fp32)
3. **Evaluation**: Full dataset inference → batch processing

---

## Verification Checklist

Before committing changes, verify:

1. **Tests Pass**: `pytest tests/ -v`
2. **Linter Passes**: `ruff check .`
3. **No Import Errors**: `python -c "from src.federated.simulation import FLSimulator"`
4. **CLI Help Works**: `python run_experiment.py --help`
5. **Config Round-Trip**: Config → dict → Config preserves all values

---

## Quick Reference Commands

```bash
# Activate environment (Windows)
.\.venv\Scripts\Activate.ps1

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest --cov=src --cov-report=term-missing tests/

# Lint
ruff check .

# Centralized training
python run_experiment.py --mode centralized --config configs/dscatnet_centralized_original.yaml

# Federated training
python run_experiment.py --mode federated --config configs/dscatnet_federated_ham10000.yaml

# Resume training
python run_experiment.py --mode federated --resume outputs/fed_xxx/checkpoints/checkpoint_round_5.pt --rounds 20

# Evaluate checkpoint
python run_experiment.py --mode evaluate --checkpoint outputs/exp/checkpoints/best_model.pt

# Verify datasets
python run_download.py --verify
```

---

## Known Limitations

1. **No Multi-GPU**: Single GPU training only
2. **Limited Aggregation**: Only FedAvg implemented (no FedProx, SCAFFOLD, etc.)
3. **No Differential Privacy**: No DP-SGD or noise mechanisms

### Client Participation Options

Two different mechanisms for partial client participation:

| CLI Flag | Config Key | Used By | Description |
|----------|------------|---------|-------------|
| `--participation` | `participation` | Flower/YAML | Sets `fraction_fit` and `fraction_evaluate` |
| `--client-selection` | `client_selection_fraction` | FLSimulator | Random selection each round |

---

## Contact & Attribution

- **Author**: Leonardo Chen
- **Institution**: Universidad Politécnica de Madrid
- **Year**: 2026
- **Base Model**: DSCATNet (Dual-Scale Cross-Attention Vision Transformer)
- **FL Framework**: Flower (flwr)
