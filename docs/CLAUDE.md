# Project Information for AI Assistants

> **Purpose**: This document provides comprehensive context for AI models (Claude, GPT, etc.) to understand, modify, and extend this codebase effectively. Read this first before making changes.

## Related Documentation

| Document | Description |
|----------|-------------|
| [README.md](../README.md) | User-facing documentation, installation, usage |
| [config-options-guide.md](config-options-guide.md) | Complete YAML configuration reference |
| [architecture.md](architecture.md) | System architecture and key classes |
| [benchmark-comparison.md](benchmark-comparison.md) | Federated vs centralized benchmark fairness audit |
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

## Notebook Exports

`02_model_evaluation.ipynb` generates `results_latest.json` (and timestamped JSON files) that include per-sample arrays used for exact paired statistical testing in `03_fl_vs_centralized_comparison.ipynb`: the JSON contains `labels`, `predictions`, `sample_ids`, `sample_predictions`, `metrics`, and `per_class_metrics`. When running automated analyses, prefer the timestamped JSON for archival reproducibility and `results_latest.json` for quick iteration.

## Notebooks

- `notebooks/01_dataset_exploration.ipynb`: dataset verification, sample visualizations, class-distribution histograms, and heterogeneity diagnostics. Produces exploratory figures and dataset summaries under `outputs/evaluation_dataset_exploration/`.

- `notebooks/02_model_evaluation.ipynb`: model evaluation including per-class metrics, ROC curves, confusion matrices, and confidence analysis. Exports per-experiment `results_latest.json` and timestamped `results_*.json` files used for paired comparisons.

- `notebooks/03_fl_vs_centralized_comparison.ipynb`: comparison pipeline between centralized and federated experiments (IID vs non-IID), statistical testing (McNemar exact test, Bonferroni correction), paired bootstrap gap confidence intervals, and communication-cost analysis. Produces summary tables and plots under `outputs/evaluation_comparison_dscatnet_all_datasets/`.

## Experiment Modalities (IID vs Non-IID)

This repository explicitly supports both IID and non-IID federated experiment modes. Configuration conventions:

- `configs/dscatnet_federated_{dataset}_iid.yaml` — near-IID experiments (Dirichlet alpha set very high, e.g., 1000.0) to approximate uniform class distributions across clients.
- `configs/dscatnet_federated_{dataset}_non_iid.yaml` — non-IID experiments (Dirichlet alpha typically 0.1–0.5) to simulate realistic heterogeneity.

The notebooks reflect these modalities: `01` inspects dataset distributions, `02` evaluates single-run metrics, and `03` performs paired statistical comparisons across IID/non-IID experiment outputs.

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
│   (paper variant)    │                      │                               │
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

### Writing Principles for Comments, Docstrings, and Documentation

- **No time-dependent language**: Avoid words like "now", "previously", "recently", "new", "old", "legacy", "was missing", "had always". Code and documentation should read as-is, not as a changelog.
- **No cross-module alignment references**: Each module's comments and docstrings must stand on their own. Do not write "matches centralized trainer" in federated code or "aligned with FL" in centralized code. State *what* the code does, not *that it matches something else*.
- **State facts, not history**: Write "Uses inverse-frequency class weights" instead of "Now uses class weights (previously missing)".

### Python Style

- **Formatting**: PEP 8 compliant, 120-character line limit (configured in `pyproject.toml`)
- **Linter**: Ruff (rules: E, F, W, B, UP, SIM, PIE, C4, PERF, RUF, PLC, PLE)
- **Type Hints**: All function signatures use PEP 585/604 style (`list[int]`, `str | None`)
- **Docstrings**: Google-style docstrings for all public functions/classes
- **Imports**: Grouped (stdlib → third-party → local), absolute imports preferred

```python
# Example function signature style
def train_epoch(
    self,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
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
    num_rounds: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-3
    optimizer_type: str = "adam"           # adam, adamw
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 8  # Effective BS = batch_size × steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimulationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
```

### YAML Config Structure

Configs use nested structure that gets flattened when loading:

```yaml
federated:
  experiment:
    name: experiment_name
  model:
    variant: paper
    image_size: 224
  training:
    batch_size: 4
    lr: 0.001
    optimizer: adam
    weight_decay: 0.0
    gradient_accumulation_steps: 8
    scheduler: none
    use_amp: false
  federation:
    num_clients: 4  # Adjust based on number of datasets used
    data_partition_type: dirichlet
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

**Checkpoint Contents**: model_state_dict, optimizer_state_dict, scheduler_state_dict, scaler_state_dict, metrics, config, history, best_val_accuracy, best_epoch, epochs_without_improvement

### `FLSimulator` (src/federated/simulation.py)

**Purpose**: Orchestrates federated learning simulation

**Key Methods**:
- `setup_clients()`: Routes to appropriate partition strategy
- `setup_natural()`: Each client uses a different dataset
- `setup_iid()`: Pooled random split for uniform class distribution
- `setup_dirichlet()`: Dirichlet-sampled class heterogeneity
- `setup_label_skew()`: Each client sees only a subset of classes
- `setup_quantity_skew()`: Clients receive different amounts of data
- `train_client()`: Local training with gradient accumulation and clipping
- `evaluate_client()`: Evaluate on client's validation data
- `aggregate_parameters()`: FedAvg weighted averaging
- `run_round()`: Single FL round (train all → aggregate → evaluate)
- `load_checkpoint()`: Restore model weights and training state
- `run()`: Main FL loop with resume support

**Checkpoint Contents**: model_state_dict, round, config, history, best_val_accuracy, best_round, rounds_without_improvement, metrics

**Resume Behavior**: When resuming without `--config`, `run_experiment.py` loads config from checkpoint to preserve original settings (data_partition_type, datasets, hyperparameters). CLI args can still override specific values.

### `ModelEvaluator` (src/evaluation/metrics.py)

**Purpose**: Comprehensive model evaluation

**Returns**: `EvaluationResults` dataclass with accuracy, balanced_accuracy, precision_macro, recall_macro, f1_macro, f1_weighted, auc_macro, confusion_matrix, per_class_metrics

### `create_dscatnet()` (src/models/dscatnet.py)

**Purpose**: Factory function for DSCATNet model

**Variants**:
- `tiny`: embed_dim=192, depth=4, heads=3 (~5M params)
- `small`: embed_dim=384, depth=6, heads=6 (~29.4M params)
- `paper`: embed_dim=384, depth=6, heads=12 (~29.4M params) **[DEFAULT]** — Paper-faithful H=12 heads (Yadav et al.)
- `base`: embed_dim=384, depth=8, heads=6 (~39M params)

**Pretrained Weight Loading** (`pretrained: true` in config):
- Loads ViT-Small (ImageNet-21k) weights from `timm` into compatible layers via `load_pretrained_vit_weights()`
- Supported for `small` and `paper` variants (embed_dim=384 matches `vit_small_patch16_224`)
- Maps ViT blocks 0–5 → fine-scale self-attention + FFN, blocks 6–11 → coarse-scale self-attention + FFN
- Transfers coarse patch embedding (16×16), positional embedding, CLS token, final LayerNorm
- Cross-attention, fine-scale embeddings, fusion, and classifier remain randomly initialized
- Loads 150/286 parameter tensors (~52% of model weights)

---

## Paper-Aligned Hyperparameters

The training configuration is aligned with the DSCATNet paper (Yadav et al., PLOS ONE, 2024):

| Parameter | Paper Value | Config Value | Notes |
|-----------|-------------|--------------|-------|
| Optimizer | Adam | `optimizer: adam` | Not AdamW |
| Learning Rate | 0.001 | `lr: 0.001` | Fixed LR throughout training |
| Weight Decay | 0.0 | `weight_decay: 0.0` | Paper uses Adam without L2 penalty |
| Effective Batch Size | 32 | `batch_size: 4` × `gradient_accumulation_steps: 8` | Fits 4GB VRAM |
| LR Scheduler | None | `scheduler: none` | Paper uses fixed LR |
| Epochs | 200 | `epochs: 200` | Full training run |
| AMP | Not used | `use_amp: false` | Disabled for stability |
| Gradient Clipping | Not mentioned | `max_grad_norm: null` | Disabled (null). Configurable via config. |
| Class Weights | Standard CE | `use_class_weights: false` | Paper uses unweighted cross-entropy |
| Image Size | 224 | `image_size: 224` | Standard ViT input |
| Augmentation | None | `augmentation: none` (HAM10000) | Paper reports no augmentation |

### Federated Experiment Structure

Each dataset has two federated configs for IID vs non-IID comparison:

| Config | Distribution | `dirichlet_alpha` | Purpose |
|--------|-------------|-------------------|---------|
| `dscatnet_federated_{dataset}_non_iid.yaml` | Non-IID | 0.5 | Heterogeneous data across clients |
| `dscatnet_federated_{dataset}_iid.yaml` | IID | 1000.0 | Uniform baseline for comparison |

Both IID and non-IID configs use Dirichlet sampling (`data_partition_type: dirichlet`). A high alpha (1000.0) produces near-uniform class distributions across clients, approximating IID. A low alpha (0.5) produces heterogeneous distributions where each client may specialize in a subset of classes.

**Federated hyperparameters** (shared across IID and non-IID):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Clients | 4 | Simulated hospital sites |
| Rounds | 100 | Communication rounds |
| Local Epochs | 1 | Per-round local training |
| Aggregation | FedAvg | Weighted by sample count |
| Participation | 1.0 | All clients each round |
| Train/Val Split | 0.85 | Per-client split |
| Checkpoint Interval | 1 | Every round |
| Early Stopping | 100 | Rounds without improvement |

### GPU Constraints

Training runs on an RTX 3050 (4GB VRAM). The effective batch size of 32 is achieved through gradient accumulation (`batch_size: 4 × gradient_accumulation_steps: 8`). Checkpoints enable resuming interrupted experiments via `resume_from` in config or `--resume` on the CLI.

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
├── test_client.py         # SkinCancerClient (Flower NumPyClient)
├── test_config_loading.py # YAML config loading and schema validation
├── test_config_schema.py  # Config schema validation rules
├── test_datasets.py       # Dataset registry and loading functions
├── test_download.py       # Download functionality tests
├── test_evaluation.py     # EvaluationResults, metrics computation
├── test_helpers.py        # Helpers (set_seed, get_device, etc.)
├── test_integration.py    # End-to-end centralized + FL integration
├── test_logging_utils.py  # MetricsTracker, CSV logging, resume safety
├── test_model_evaluator.py# ModelEvaluator integration tests
├── test_models.py         # DSCATNet model architecture tests
├── test_preprocessing.py  # Transforms, augmentation levels
├── test_simulation.py     # SimulationConfig, FLSimulator, FedAvg
├── test_splits.py         # IID/Non-IID splitting utilities
├── test_strategy.py       # DSCATNetFedAvg strategy tests
├── test_verify.py         # DatasetVerifier tests
└── test_visualization.py  # Visualization/plotting tests
```

- **457 tests**, `@slow` tests deselected by default
- **≥80% line coverage** (`fail_under = 80` in pyproject.toml)

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

1. Create class in `src/data/datasets.py` inheriting from `BaseDermoscopyDataset`
2. Implement `_load_metadata()` and `_build_image_list()`
3. Add label mapping to `UNIFIED_CLASSES_7` (and `UNIFIED_CLASSES_BINARY` if needed)
4. Register in `DATASET_REGISTRY` with a `DatasetConfig` entry
5. Add canonical name to `normalize_dataset_name()` mapping

Both `CentralizedTrainer.setup_data()` and `FLSimulator` resolve datasets through `DATASET_REGISTRY`, so no changes are needed in training code.

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
└── {experiment_name}/
    ├── checkpoints/
    │   ├── best_model.pt           # Weights only
    │   ├── best_checkpoint.pt      # Full state (centralized)
    │   └── checkpoint_{epoch/round}_N.pt
    ├── config.json                 # Serialized config
    ├── results.json                # Final metrics + training history
    ├── metrics/                    # Real-time CSV metrics
    │   └── {name}_metrics.csv
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

- `batch_size=4` default (fits 4GB VRAM with paper variant)
- Gradient accumulation (`gradient_accumulation_steps=8`) for effective BS=32
- `num_workers=4` for data loading (Windows: may need `num_workers=0`)
- AMP disabled by default for training stability

### Bottlenecks

1. **Data Loading**: Large datasets → use SSD
2. **FL Communication**: ~112MB model params per round (paper variant, fp32)
3. **Evaluation**: Full dataset inference → batch processing

---

## Verification Checklist

Before committing changes, verify:

1. **Tests Pass**: `pytest tests/ -v`
2. **Linter Passes**: `ruff check .`
3. **Formatter Check Passes**: `ruff format --check src/ tests/ run_experiment.py run_download.py run_tests.py`
4. **No Import Errors**: `python -c "from src.federated.simulation import FLSimulator"`
5. **CLI Help Works**: `python run_experiment.py --help`
6. **Config Round-Trip**: Config → dict → Config preserves all values

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

# Formatter check (CI)
ruff format --check src/ tests/ run_experiment.py run_download.py run_tests.py

# Centralized training
python run_experiment.py --mode centralized --config configs/dscatnet_centralized_original.yaml

# Federated training
python run_experiment.py --mode federated --config configs/dscatnet_federated_ham10000_non_iid.yaml

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
4. **Pretrained Weights**: Available for `small` and `paper` variants (ViT-Small from `timm`, embed_dim=384). Other variants use random initialization.

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
