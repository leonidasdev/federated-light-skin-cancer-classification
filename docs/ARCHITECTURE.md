# Architecture Overview

This document provides a technical overview of the DSCATNet Federated Learning system architecture.

## System Architecture

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

## Key Classes & Responsibilities

### CentralizedTrainer

**Location**: `src/centralized/centralized.py`  
**Purpose**: Standard PyTorch training loop for baseline comparison

**Key Methods**:
| Method | Description |
|--------|-------------|
| `setup_data()` | Loads datasets, creates DataLoaders |
| `train_epoch()` | Single epoch training with gradient accumulation and optional AMP |
| `evaluate()` | Validation with per-class metrics |
| `save_checkpoint()` | Full state persistence |
| `load_checkpoint()` | Restore training state |
| `run()` | Main training loop with early stopping |

**Checkpoint Contents**:
```python
{
    "epoch": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict": dict,  # AMP
    "metrics": dict,
    "config": dict,
    "history": dict,
    "best_val_accuracy": float,
    "best_epoch": int,
    "epochs_without_improvement": int,
}
```

### FLSimulator

**Location**: `src/federated/simulation.py`  
**Purpose**: Orchestrates federated learning simulation

**Key Methods**:
| Method | Description |
|--------|-------------|
| `setup_clients()` | Routes to natural or Dirichlet non-IID setup |
| `setup_natural_noniid()` | Each dataset = one client |
| `setup_dirichlet_noniid()` | Split combined data via Dirichlet distribution |
| `train_client()` | Local training with gradient accumulation and clipping |
| `aggregate_parameters()` | FedAvg weighted averaging |
| `run_round()` | Single FL round (train → aggregate → evaluate) |
| `run()` | Main FL loop with resume support |

**Checkpoint Contents**:
```python
{
    "round": int,
    "model_state_dict": dict,
    "metrics": dict,
    "config": dict,
    "history": dict,
    "best_val_accuracy": float,
    "best_round": int,
    "rounds_without_improvement": int,
}
```

### ModelEvaluator

**Location**: `src/evaluation/metrics.py`  
**Purpose**: Comprehensive model evaluation

**Returns**: `EvaluationResults` dataclass with:
- `accuracy`, `balanced_accuracy`
- `precision_macro`, `recall_macro`
- `f1_macro`, `f1_weighted`
- `auc_macro`
- `confusion_matrix`
- `per_class_metrics`

### create_dscatnet()

**Location**: `src/models/dscatnet.py`  
**Purpose**: Factory function for DSCATNet model

**Variants**:
| Variant | embed_dim | depth | heads | ~Parameters |
|---------|-----------|-------|-------|-------------|
| `tiny` | 192 | 4 | 3 | ~5M |
| `small` | 384 | 6 | 6 | ~29.4M (default) |
| `paper` | 384 | 6 | 12 | ~29.4M (paper-faithful) |
| `base` | 384 | 8 | 6 | ~39M |

**Pretrained Weight Loading** (`pretrained: true`):
When `pretrained=True` and `variant='small'`, calls `load_pretrained_vit_weights()` to transfer compatible ViT-Small (ImageNet) weights from `timm` into self-attention, FFN, and coarse-scale embedding layers. Cross-attention and classifier layers remain randomly initialized.

---

## Dataset Classes

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

### Dataset Classes

Each inherits from `torch.utils.data.Dataset`:

| Class | Images | Classes | Notes |
|-------|--------|---------|-------|
| `HAM10000Dataset` | 10,015 | 7 | Primary dataset |
| `ISIC2018Dataset` | ~10,015 | 7 | ISIC Challenge |
| `ISIC2019Dataset` | ~25,331 | 8+UNK → 7 | Filtered |
| `ISIC2020Dataset` | ~33,126 | 2 → 7 | Binary mapped |
| `PADUFES20Dataset` | 2,298 | 6 | Brazilian dataset |

---

## Configuration Pattern

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
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationConfig":
        return cls(**{k: v for k, v in d.items() 
                      if k in cls.__dataclass_fields__})
```

---

## Output Structure

```
outputs/
└── {mode}_{timestamp}/
    ├── checkpoints/
    │   ├── best_model.pt           # Weights only (for inference)
    │   ├── best_checkpoint.pt      # Full state (for resumption)
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
    logger.warning(f"Checkpoint not found, starting from scratch")
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
