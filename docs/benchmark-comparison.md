# Federated vs Centralized Benchmark Comparison

> **Purpose**: Documents all implementation differences between the centralized and federated training pipelines. Any difference listed here is either (a) inherent to the FL paradigm and intentional, or (b) a controlled variable kept consistent across both pipelines.

---

## Summary

The goal of this project is to compare **centralized** (pooled data) training against **federated** (distributed, non-IID) training using the same DSCATNet model. All controllable hyperparameters are kept consistent, and the inherent differences that arise from the FL paradigm are documented below.

---

## Consistent Parameters

These settings are identical between centralized and federated training:

| Parameter | Centralized | Federated | Notes |
|-----------|-------------|-----------|-------|
| **Model** | DSCATNet small (~29.4M) | DSCATNet small (~29.4M) | Same architecture, same weights init |
| **Optimizer** | Adam | Adam | Matches DSCATNet paper |
| **Learning Rate** | 0.001 | 0.001 | Fixed LR, no scheduler |
| **Weight Decay** | 0.0 | 0.0 | Matches DSCATNet paper |
| **Effective Batch Size** | 32 (4 × 8) | 32 (4 × 8) | Same gradient accumulation |
| **Gradient Clipping** | `max_norm=1.0` | `max_norm=1.0` | Same threshold |
| **Image Size** | 224 × 224 | 224 × 224 | Standard ViT input |
| **Augmentation** | None | None | Matches DSCATNet paper |
| **Class Weights** | Inverse frequency | Inverse frequency | Both use `weight_c = N / (C × N_c)` |
| **Eval Loss** | Unweighted CE | Unweighted CE | Fair eval without data-dependent bias |
| **Pretrained** | Yes (ImageNet) | Yes (ImageNet) | Same initialization |
| **Num Classes** | 7 | 7 | Unified label schema |
| **Pin Memory** | CUDA-aware | CUDA-aware | `pin_memory=(device == "cuda")` |

---

## Inherent FL Differences (By Design)

These differences are **inherent** to the federated learning paradigm and cannot be eliminated without breaking the FL setup. They represent the actual factors being studied:

### 1. Data Distribution

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| **Data pooling** | All data combined into one dataset | Data partitioned across clients |
| **IID assumption** | Data is shuffled IID | Non-IID via Dirichlet (α=0.5) or natural partitioning |
| **Label balance** | Global class balance preserved | Per-client class balance varies significantly |
| **Effective samples/step** | Full dataset per epoch | Client-local subset per round |

### 2. Optimizer State

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| **Persistence** | Optimizer state (momentum, adaptive LR) persists across all epochs | Fresh optimizer created per client per round — no momentum carry-over |
| **Impact** | Adam's adaptive state accumulates useful gradient statistics | Each client starts "cold" every round, losing accumulated gradient moments |

This is a well-known limitation of FedAvg and is inherent to the protocol. Solutions like FedOpt (server-side momentum) or SCAFFOLD exist but are outside the scope of this benchmark.

### 3. Model Aggregation

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| **Updates** | Direct gradient updates on a single model | Local training → FedAvg weighted averaging of parameters |
| **Communication** | N/A | ~112 MB per round per client (paper variant, fp32) |
| **Staleness** | None — always the latest model | Each client trains on a slightly stale model snapshot |

### 4. Training Granularity

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| **Unit** | Epoch (full pass over all data) | Round (1 local epoch per client, then aggregate) |
| **Per update** | Sees entire dataset | Each client sees only its local partition |
| **Early stopping** | Based on global val accuracy per epoch | Based on aggregated val accuracy per round |

---

## Controlled Differences (Implementation Choices)

These are implementation details that differ but have been assessed for benchmark impact:

### 5. Learning Rate Scheduler

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| **Support** | `none`, `cosine`, `plateau` | No scheduler (always fixed LR) |
| **Benchmark config** | `scheduler: none` | N/A (fixed LR) |
| **Impact** | **None** — both use fixed LR of 0.001 in benchmark configs |

The centralized config uses `scheduler: none` in all benchmark experiments, so this difference has no practical effect on results.

### 6. Automatic Mixed Precision (AMP)

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| **Support** | Configurable via `use_amp` | Not implemented |
| **Benchmark config** | `use_amp: false` | N/A |
| **Impact** | **None** — AMP is disabled in all benchmark configs |

AMP is disabled for training stability (avoiding NaN issues with the DSCATNet architecture), so this difference does not affect results.

### 7. DataLoader Configuration

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| **`drop_last`** | `True` (train only) | `False` (default) |
| **`num_workers`** | 4 (default) | 2 (default) |
| **Impact** | **Minimal** — `drop_last` may discard up to `batch_size - 1` samples; `num_workers` only affects speed, not results |

`drop_last=True` in centralized prevents incomplete batches from affecting gradient accumulation. In FL, batch sizes are small enough relative to client data that this has negligible impact. `num_workers` is a performance setting with no effect on training outcomes.

### 8. Validation Split Ratio

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| **Config key** | `val_split: 0.15` | `train_val_split: 0.85` |
| **Effective split** | 85% train / 15% val | 85% train / 15% val |
| **Impact** | **None** — same effective split when configured consistently |

The naming convention differs (`val_split` vs `train_val_split`), but both are set to yield an 85/15 split in the benchmark YAML configs.

### 9. Early Stopping Patience

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| **Default** | 15 epochs | 10 rounds |
| **Benchmark config** | 200 | 100 |
| **Impact** | **None** — patience values match total training duration, effectively disabling early stopping to match the paper. |

### 10. Evaluation Granularity

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| **Per-class metrics** | Yes (per-class accuracy computed during `evaluate()`) | No (only aggregate loss and accuracy) |
| **Detailed evaluation** | Available during training | Available via `--mode evaluate` post-training |
| **Impact** | **None on training** — per-class metrics are logged but don't affect optimization |

### 11. Checkpoint Contents

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| **Optimizer state** | Saved and restored | Not saved (fresh optimizer per round) |
| **Scheduler state** | Saved and restored | N/A (no scheduler) |
| **AMP scaler** | Saved and restored | N/A (no AMP) |
| **Impact** | **None on training dynamics** — FL doesn't need optimizer state due to paradigm design |

---

## Class Weight Implementation

Both pipelines use the same inverse-frequency class weighting formula:

```
weight_c = N_total / (C × N_c)
```

Where:
- `N_total` = total number of training samples
- `C` = number of classes  
- `N_c` = number of samples in class `c`

### Centralized
- Computed from the combined `ConcatDataset` of all training data
- Applied via `nn.CrossEntropyLoss(weight=self.class_weights)`

### Federated
- Computed globally from all clients' `class_distribution` dicts after `setup_clients()`
- Applied via `nn.CrossEntropyLoss(weight=self.class_weights)` in `train_client()`
- **Privacy note**: Sharing class distribution counts (not individual samples) is a common FL practice that does not compromise sample-level privacy

### Why This Matters

Class weighting is available as a configuration option (`use_class_weights: true`) for practical clinical deployments where class imbalance is severe. In paper-aligned benchmark experiments, standard (unweighted) CrossEntropyLoss is used, matching the original DSCATNet evaluation protocol.

---

## Configuration Reference

### Centralized Benchmark Config

```yaml
centralized:
  training:
    batch_size: 4
    gradient_accumulation_steps: 8
    optimizer: adam
    lr: 0.001
    weight_decay: 0.0
    scheduler: none
    use_amp: false
  augmentation:
    level: none
  evaluation:
    early_stopping_patience: 200
    use_class_weights: false
    max_grad_norm: null
  splits:
    val_split: 0.15
```

### Federated Benchmark Config

```yaml
federated:
  training:
    batch_size: 4
    gradient_accumulation_steps: 8
    optimizer: adam
    lr: 0.001
    weight_decay: 0.0
    local_epochs: 1
    train_val_split: 0.85
  federation:
    num_clients: 4
    noniid_type: dirichlet
    dirichlet_alpha: 0.5
  augmentation:
    level: none
  evaluation:
    early_stopping_patience: 100
    use_class_weights: false
    max_grad_norm: null
```

---

## Code Locations

| Component | Centralized | Federated |
|-----------|-------------|-----------|
| **Config** | `CentralizedConfig` in [centralized.py](../src/centralized/centralized.py) | `SimulationConfig` in [simulation.py](../src/federated/simulation.py) |
| **Training loop** | `CentralizedTrainer.train_epoch()` | `FLSimulator.train_client()` |
| **Class weights** | `CentralizedTrainer._compute_class_weights()` | `FLSimulator._compute_class_weights()` |
| **Evaluation** | `CentralizedTrainer.evaluate()` | `FLSimulator.evaluate_client()` |
| **Checkpoints** | `CentralizedTrainer.save_checkpoint()` | `FLSimulator.save_checkpoint()` |

---

## Verdict

After this audit, the only remaining differences between the two pipelines are **inherent to the federated learning paradigm** (data distribution, optimizer state reset, FedAvg aggregation) or have **no practical impact** on benchmark results (disabled scheduler, disabled AMP, minor DataLoader settings). The comparison is fair for evaluating the impact of federated learning on DSCATNet performance.
