# Configuration Options Reference

This document provides a comprehensive reference for all configuration options used in the DSCATNet Federated Learning project.

## Table of Contents

- [Configuration Files Overview](#configuration-files-overview)
- [Schema Validation](#schema-validation)
- [Experiment Configuration](#experiment-configuration)
- [Federated Learning Configuration](#federated-learning-configuration)
- [Model Configuration](#model-configuration)
- [Templates](#templates)

---

## Configuration Files Overview

The project uses YAML configuration files located in the `configs/` directory:

| File | Purpose |
|------|---------|
| `experiment_config.yaml` | Master configuration for comparison experiments |
| `fl_config.yaml` | Default federated learning settings |
| `model_config.yaml` | DSCATNet architecture configuration |
| `dscatnet_federated_*.yaml` | Dataset-specific FL experiment configs |

## Schema Validation

All configuration files can be validated using the built-in schema validation:

```bash
# Validate a specific config
python src/utils/config_schema.py configs/experiment_config.yaml

# With verbose output
python src/utils/config_schema.py configs/model_config.yaml -v

# Specify config type explicitly
python src/utils/config_schema.py configs/fl_config.yaml --type federated
```

### Programmatic Validation

```python
from src.utils.config_schema import validate_config, ConfigType

# Auto-detect config type
config = validate_config("configs/experiment_config.yaml")

# Explicit config type
config = validate_config("configs/fl_config.yaml", ConfigType.FEDERATED)

# Access validated values
print(config.federated.num_rounds)
```

---

## Experiment Configuration

Used by `run_experiment.py` for running centralized vs federated comparisons.

### experiment

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | "DSCATNet-FL-SkinCancer" | Unique experiment identifier |
| `description` | string | "" | Experiment description |
| `seed` | int | 42 | Random seed for reproducibility |

### hardware

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `device` | string | "cuda" | cuda, cpu | Compute device |
| `num_workers` | int | 2 | 0-16 | DataLoader workers |
| `pin_memory` | bool | true | - | Pin memory for GPU transfer |
| `mixed_precision` | bool | true | - | Enable automatic mixed precision |

### data

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `root_dir` | string | "./data" | - | Path to data directory |
| `img_size` | int | 224 | 32-512 | Input image size |
| `val_split` | float | 0.15 | 0.0-0.5 | Validation split fraction |
| `test_split` | float | 0.15 | 0.0-0.5 | Test split fraction |
| `normalization` | string | "imagenet" | imagenet, dermoscopy | Normalization type |
| `augmentation_level` | string | "medium" | none, light, medium, heavy | Augmentation intensity |
| `classification_mode` | string | "multiclass" | multiclass, multiclass_8, binary | Classification type |
| `num_classes` | int | 7 | 2-10 | Number of output classes |
| `filter_unknown` | bool | true | - | Filter UNK labels |
| `use_weighted_sampling` | bool | false | - | Use WeightedRandomSampler |
| `use_class_weights` | bool | true | - | Use class weights in loss |

**Note:** `val_split + test_split` must be less than 1.0.

### centralized

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `epochs` | int | 200 | 1-1000 | Training epochs |
| `batch_size` | int | 8 | 1-256 | Physical batch size |
| `gradient_accumulation_steps` | int | 1 | 1-32 | Gradient accumulation steps (effective BS = batch_size × steps) |
| `optimizer` | string | "adam" | adam, adamw | Optimizer type |
| `learning_rate` | float | 0.001 | >0, ≤1.0 | Learning rate |
| `weight_decay` | float | 0.0 | 0-1.0 | L2 regularization |
| `scheduler` | string | "none" | none, cosine, plateau | LR scheduler |
| `use_amp` | bool | false | - | Enable automatic mixed precision |
| `early_stopping_patience` | int | 15 | 1-100 | Early stopping patience |
| `pooled_data` | bool | true | - | Combine all datasets |

### federated_experiments

List of federated experiment configurations:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | **required** | Experiment name |
| `description` | string | "" | Experiment description |
| `num_rounds` | int | 100 | FL training rounds |
| `local_epochs` | int | 3 | Local epochs per round |
| `batch_size` | int | 8 | Local batch size |
| `noniid_type` | string | "natural" | Data distribution type |
| `dirichlet_alpha` | float | null | Dirichlet concentration (if using dirichlet) |

**noniid_type options:**
- `natural`: Each client uses its own dataset
- `dirichlet`: Dirichlet distribution (requires `dirichlet_alpha`)
- `label_skew`: Each client gets limited classes
- `quantity_skew`: Clients have different data amounts

### metrics

| Option | Type | Description |
|--------|------|-------------|
| `classification` | list[string] | Classification metrics to compute |
| `federated` | list[string] | FL-specific metrics |
| `per_class` | list[string] | Per-class metrics |

### logging

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `log_dir` | string | "./logs" | Log directory |
| `checkpoint_dir` | string | "./checkpoints" | Checkpoint directory |
| `tensorboard` | bool | true | Enable TensorBoard |
| `wandb.enabled` | bool | false | Enable W&B |
| `wandb.project` | string | "dscatnet-fl" | W&B project name |
| `wandb.entity` | string | null | W&B entity |

### reproducibility

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `deterministic` | bool | true | Use deterministic algorithms |
| `benchmark` | bool | false | Use cuDNN benchmark mode |
| `seed` | int | 42 | Global random seed |

---

## Federated Learning Configuration

Used by `run_experiment.py --mode federated` for federated training.

### federated

#### Framework Settings

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `framework` | string | "flower" | flower | FL framework |
| `strategy` | string | "FedAvg" | FedAvg, FedProx, FedNova | Aggregation strategy |

#### Client Configuration

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `num_clients` | int | 4 | 1-100 | Number of clients |
| `clients` | list | null | - | Optional client definitions |

Each client in `clients` list:
| Option | Type | Description |
|--------|------|-------------|
| `id` | int | Client ID (≥1) |
| `dataset` | string | Dataset name |
| `description` | string | Client description |

#### Training Rounds

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `num_rounds` | int | 100 | 1-1000 | FL training rounds |
| `early_stopping_patience` | int | 20 | 1-100 | Early stopping patience |

#### Client Participation

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `participation` | float | 1.0 | >0, ≤1.0 | YAML config: fraction of clients participating |
| `client_selection_fraction` | float | 1.0 | >0, ≤1.0 | SimulationConfig: random client selection |
| `fraction_fit` | float | 1.0 | >0, ≤1.0 | Flower internal: fraction for training |
| `fraction_evaluate` | float | 1.0 | >0, ≤1.0 | Flower internal: fraction for evaluation |
| `min_fit_clients` | int | 4 | ≥1 | Min clients for training |
| `min_evaluate_clients` | int | 4 | ≥1 | Min clients for evaluation |
| `min_available_clients` | int | 4 | ≥1 | Min available clients |

**Participation vs Client Selection:**

- **`participation`** (YAML): Sets both `fraction_fit` and `fraction_evaluate` for Flower-based federation
- **`client_selection_fraction`** (SimulationConfig): Used by the custom FLSimulator for random client selection each round
- **`--participation`** (CLI): Maps to `fraction_fit`/`fraction_evaluate` (Flower's native parameters)
- **`--client-selection`** (CLI): Maps to `client_selection_fraction` (custom simulator)

Both achieve partial client participation but through different mechanisms. Use `participation` in YAML configs or `--participation` CLI when using Flower-based training. Use `client_selection_fraction` or `--client-selection` when using the custom `FLSimulator`.

**Note:** `min_fit_clients` and `min_evaluate_clients` cannot exceed `num_clients`.

#### Local Training

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `local_epochs` | int | 1 | 1-50 | Local epochs per round |
| `local_batch_size` | int | 8 | 1-256 | Local batch size |
| `gradient_accumulation_steps` | int | 1 | 1-32 | Gradient accumulation steps (effective BS = batch_size × steps) |
| `train_val_split` | float | 0.85 | 0.5-1.0 | Fraction of data for training (rest for validation) |

#### Optimizer Configuration

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `optimizer` | string | "adam" | adam, adamw | Optimizer type |
| `learning_rate` | float | 0.001 | >0, ≤1.0 | Learning rate |
| `weight_decay` | float | 0.0 | 0-1.0 | L2 regularization |

#### Learning Rate Schedule

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `lr_scheduler` | string | "cosine" | cosine, step, plateau, none | LR scheduler |
| `warmup_epochs` | int | 5 | 0-50 | Warmup epochs |
| `min_lr` | float | 0.000001 | 0-1.0 | Minimum learning rate |

#### Communication & Checkpointing

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `save_every_rounds` | int | 10 | 1-100 | Checkpoint frequency |
| `evaluate_every_rounds` | int | 1 | 1-100 | Evaluation frequency |
| `server_address` | string | "[::]:8080" | - | Server address |

### scenarios

Named data distribution scenarios:

| Scenario | Description |
|----------|-------------|
| `natural_noniid` | Each client uses its own dataset |
| `iid_pooled` | Uniform distribution (α=1000) |
| `moderate_noniid` | Dirichlet with α=0.5 |
| `extreme_noniid` | Dirichlet with α=0.1 |

### strategies

Aggregation strategy configurations:

| Strategy | Description | Extra Parameters |
|----------|-------------|------------------|
| `FedAvg` | Weighted averaging by sample count | - |
| `FedProx` | FedAvg with proximal term | `mu` (default: 0.01) |
| `FedNova` | Normalized averaging | - |

---

## Model Configuration

Used for DSCATNet architecture configuration.

### model

#### Input Configuration

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `name` | string | "DSCATNet" | - | Model name |
| `img_size` | int | 224 | 32-512 | Input image size |
| `in_channels` | int | 3 | 1-4 | Input channels |
| `num_classes` | int | 7 | 2-10 | Output classes |

#### Architecture Parameters

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `embed_dim` | int | 384 | 64-1024 | Embedding dimension |
| `depth` | int | 6 | 1-24 | Transformer blocks |
| `num_heads` | int | 6 | 1-16 | Attention heads |
| `mlp_ratio` | float | 4.0 | 1.0-8.0 | MLP expansion ratio |

**Constraints:**
- `embed_dim` must be divisible by `num_heads`
- `img_size` must be divisible by both patch sizes

#### Dual-Scale Patch Sizes

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `fine_patch_size` | int | 8 | 4-32 | Fine scale patches |
| `coarse_patch_size` | int | 16 | 8-64 | Coarse scale patches |

Patch count calculation:
- Fine scale: `(img_size / fine_patch_size)²` patches
- Coarse scale: `(img_size / coarse_patch_size)²` patches

For `img_size=224`:
- `fine_patch_size=8` → 784 patches (28×28)
- `coarse_patch_size=16` → 196 patches (14×14)

#### Regularization

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `drop_rate` | float | 0.1 | 0.0-0.5 | Dropout rate |
| `attn_drop_rate` | float | 0.0 | 0.0-0.5 | Attention dropout |

#### Feature Fusion

| Option | Type | Default | Options | Description |
|--------|------|---------|---------|-------------|
| `fusion_method` | string | "concat" | concat, add, attention | Fusion method |

### variants

Pre-defined model variants:

| Variant | embed_dim | depth | num_heads | mlp_ratio | ~Params |
|---------|-----------|-------|-----------|-----------|---------|
| `tiny` | 192 | 4 | 3 | 3.0 | ~5M |
| `small` | 384 | 6 | 6 | 4.0 | ~15M |
| `base` | 384 | 8 | 6 | 4.0 | ~20M |

---

## Templates

Configuration templates are available in `configs/templates/`:

| Template | Use Case |
|----------|----------|
| `experiment_template.yaml` | New comparison experiments |
| `federated_template.yaml` | FL-specific configurations |
| `model_template.yaml` | Model architecture tuning |

### Using Templates

1. Copy the template:
   ```bash
   cp configs/templates/experiment_template.yaml configs/my_experiment.yaml
   ```

2. Modify values as needed

3. Validate your configuration:
   ```bash
   python src/utils/config_schema.py configs/my_experiment.yaml
   ```

4. Run your experiment:
   ```bash
   python run_experiment.py --config configs/my_experiment.yaml
   ```

---

## Best Practices

1. **Always validate configs** before running experiments
2. **Use templates** as a starting point for new configurations
3. **Keep seeds consistent** across related experiments
4. **Document changes** in your config files with comments
5. **Use meaningful names** for experiments and scenarios

## Common Issues

### Validation Errors

1. **"val_split + test_split must be less than 1.0"**
   - Reduce `val_split` or `test_split` in data config

2. **"min_fit_clients cannot exceed num_clients"**
   - Ensure `min_fit_clients <= num_clients`

3. **"img_size must be divisible by patch_size"**
   - Choose patch sizes that evenly divide `img_size`

4. **"embed_dim must be divisible by num_heads"**
   - Adjust `embed_dim` or `num_heads` for divisibility
