# Federated Learning for Skin Cancer Classification with DSCATNet

[![CI](https://github.com/leonidasdev/federated-light-skin-cancer-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/leonidasdev/federated-light-skin-cancer-classification/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/pytorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
[![Flower 1.25+](https://img.shields.io/badge/flower-1.25+-green.svg)](https://flower.dev/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Research Contribution](#research-contribution)
3. [Project Structure](#project-structure)
4. [Model Architecture](#model-architecture)
5. [Installation](#installation)
6. [Dataset Setup](#dataset-setup)
7. [Configuration System](#configuration-system)
8. [Training Pipeline](#training-pipeline)
9. [Checkpoints & Resume Training](#checkpoints--resume-training)
10. [Model Evaluation](#model-evaluation)
11. [CLI Reference](#cli-reference)
12. [Experiment Outputs](#experiment-outputs)
13. [Notebooks](#notebooks)
14. [Testing](#testing)
15. [Troubleshooting](#troubleshooting)
16. [Documentation](#documentation)
17. [Citation](#citation)
18. [License](#license)

---

## Overview

This project evaluates the **Dual-Scale Cross-Attention Vision Transformer (DSCATNet)** in a **Federated Learning** setting for dermoscopic skin lesion classification.

**This is a thesis project** investigating whether lightweight Vision Transformers can maintain their classification accuracy under federated learning constraints, specifically with non-IID (non-Independent and Identically Distributed) data across multiple simulated hospitals/institutions.

### Key Features

- **DSCATNet Implementation**: Lightweight ViT with dual-scale cross-attention (~29.4M parameters, paper variant)
- **Federated Learning**: Flower-based FL simulation with FedAvg aggregation
- **Multiple Non-IID Modes**: Natural (dataset-based), Dirichlet, label skew, quantity skew
- **5 Dermoscopy Datasets**: HAM10000, ISIC 2018/2019/2020, PAD-UFES-20
- **Comprehensive Evaluation**: Accuracy, F1, AUC-ROC, confusion matrices, per-class metrics
- **Checkpoint Management**: Resume training, best model tracking, automatic cleanup

---

## Research Contribution

| Aspect | Description |
|--------|-------------|
| **Novel Evaluation** | First adaptation and evaluation of DSCATNet in federated learning |
| **Real-World Non-IID** | Each FL client holds a different dermoscopy dataset (natural heterogeneity) |
| **Comprehensive Comparison** | Centralized vs. IID-FL vs. Non-IID-FL performance analysis |
| **Lightweight Focus** | Benchmarking against literature on efficient FL models |

---

## Project Structure

```
federated-light-skin-cancer-classification/
│
├── configs/                          # YAML configuration files
│   ├── dscatnet_federated_ham10000_non_iid.yaml  # Main FL experiment config
│   ├── dscatnet_centralized_original.yaml        # Centralized baseline config
│   ├── dscatnet_federated_padufes20_non_iid.yaml # Alternative FL config
│   ├── fl_config.yaml                      # FL framework defaults
│   ├── model_config.yaml                   # DSCATNet architecture settings
│   └── experiment_config.yaml              # Comparison experiment settings
│
├── data/                             # Datasets (download required)
│   ├── HAM10000/
│   ├── ISIC2018/
│   ├── ISIC2019/
│   ├── ISIC2020/
│   └── PAD-UFES-20/
│
├── outputs/                          # Training outputs (auto-generated)
│   └── <experiment_name>/
│       ├── checkpoints/
│       │   ├── best_model.pt
│       │   ├── best_checkpoint.pt
│       │   └── checkpoint_{epoch/round}_N.pt
│       ├── config.json
│       ├── results.json
│       ├── metrics/
│       │   └── <experiment_name>_metrics.csv
│       └── experiment.log
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── models/                       # DSCATNet implementation
│   │   ├── dscatnet.py               # Main model class
│   │   ├── cross_attention.py        # Cross-scale attention module
│   │   └── patch_embedding.py        # Dual-scale patch embedding
│   ├── federated/                    # FL components
│   │   ├── client.py                 # Flower NumPyClient
│   │   ├── server.py                 # FL server utilities
│   │   ├── simulation.py             # FL simulator (FedAvg)
│   │   └── strategy.py               # Aggregation strategies
│   ├── centralized/                  # Baseline training
│   │   └── centralized.py            # Centralized trainer
│   ├── data/                         # Data handling
│   │   ├── datasets.py               # Dataset classes (HAM10000, ISIC, PAD-UFES-20)
│   │   ├── preprocessing.py          # Transforms & augmentation
│   │   ├── splits.py                 # IID/Non-IID splitting utilities
│   │   ├── download.py               # ISIC API downloader
│   │   └── verify.py                 # Dataset verification
│   ├── evaluation/                   # Evaluation utilities
│   │   ├── metrics.py                # Classification metrics
│   │   └── visualization.py          # Plotting functions
│   └── utils/                        # Helpers
│       ├── checkpoints.py            # Checkpoint management
│       ├── config_schema.py          # YAML config validation
│       ├── helpers.py                # Seed, device, formatting
│       └── logging_utils.py          # Logging configuration
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_dataset_exploration.ipynb  # Dataset analysis & verification
│   ├── 02_model_evaluation.ipynb     # Model evaluation & metrics
│   └── 03_fl_vs_centralized_comparison.ipynb  # FL vs centralized comparison
│
├── tests/                            # Unit tests
│   ├── conftest.py               # Shared fixtures and markers
│   ├── test_centralized.py           # Centralized training tests
│   ├── test_checkpoints.py           # Checkpoint save/load tests
│   ├── test_cli.py                   # CLI argument parsing tests
│   ├── test_client.py                # FL client tests
│   ├── test_config_loading.py        # Config loading/validation tests
│   ├── test_config_schema.py         # Config schema validation tests
│   ├── test_datasets.py              # Dataset registry tests
│   ├── test_download.py              # Download functionality tests
│   ├── test_evaluation.py            # Evaluation metrics tests
│   ├── test_helpers.py               # Helper utility tests
│   ├── test_integration.py           # End-to-end integration tests
│   ├── test_logging_utils.py         # Logging & metrics tracker tests
│   ├── test_model_evaluator.py       # Model evaluator tests
│   ├── test_models.py                # DSCATNet architecture tests
│   ├── test_preprocessing.py         # Preprocessing pipeline tests
│   ├── test_simulation.py            # FL simulation tests
│   ├── test_splits.py                # Data splitting tests
│   ├── test_strategy.py              # FedAvg strategy tests
│   ├── test_verify.py                # Dataset verification tests
│   └── test_visualization.py         # Visualization tests
│
├── docs/                             # Documentation
│   ├── architecture.md               # System architecture
│   ├── benchmark-comparison.md       # FL vs centralized fairness audit
│   ├── CLAUDE.md                     # AI assistant context
│   ├── config-options-guide.md       # Configuration reference
│   └── README.md                     # Documentation index
│
├── run_experiment.py                 # Main entry point
├── run_download.py                   # Dataset downloader
├── run_tests.py                      # Test runner
├── CONTRIBUTING.md                   # Contribution guidelines
├── requirements.txt                  # Python dependencies
├── pyproject.toml                    # Project configuration
└── README.md
```

---

## Model Architecture

### DSCATNet (Dual-Scale Cross-Attention Vision Transformer)

DSCATNet is a lightweight Vision Transformer designed specifically for dermoscopic image classification. It captures both fine-grained local features and global contextual information through dual-scale processing.

```
Input Image (224×224×3)
         │
         ▼
┌─────────────────────────────────┐
│   Dual-Scale Patch Embedding    │
│  ┌───────────┬───────────┐      │
│  │ Fine 8×8  │Coarse 16×16│     │
│  │784 patches│196 patches │     │
│  └───────────┴───────────┘      │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Cross-Scale Attention Blocks   │
│  (6 blocks, 12 heads, dim=384)  │
│  Fine ←→ Coarse attention       │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Feature Fusion (concat)     │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   CLS Token Extraction          │
│   + Classification Head         │
│   → 7 classes (softmax)         │
└─────────────────────────────────┘
```

### Model Variants

| Variant | Embed Dim | Depth | Heads | Parameters | Use Case |
|---------|-----------|-------|-------|------------|----------|
| `tiny`  | 192       | 4     | 3     | ~5M        | Resource-constrained FL clients |
| `small` | 384       | 6     | 6     | ~29.4M     | Balanced performance |
| `paper` | 384       | 6     | 12    | ~29.4M     | **Default** - paper-faithful (Yadav et al.) |
| `base`  | 384       | 8     | 6     | ~39M       | Maximum accuracy |

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/leonidasdev/federated-light-skin-cancer-classification.git
cd federated-light-skin-cancer-classification
```

### 2. Create Virtual Environment

```bash
# Create venv
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; import flwr; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Python   | 3.10+   | 3.10+       |
| RAM      | 8GB     | 16GB+       |
| GPU VRAM | 4GB     | 8GB+        |
| Disk     | 30GB    | 50GB+       |
| CUDA     | 11.8+   | 12.0+       |

---

## Dataset Setup

### Supported Datasets

| Dataset | Images | Classes | Source | FL Client |
|---------|--------|---------|--------|-----------|
| HAM10000 | 10,015 | 7 | Kaggle | Client 1 |
| ISIC 2018 | ~10,015 | 7 | ISIC Archive | Client 2 |
| ISIC 2019 | ~25,331 | 8+UNK | ISIC Archive | Client 3 |
| ISIC 2020 | ~33,126 | 2 (binary) | ISIC Archive | Client 4 |
| PAD-UFES-20 | 2,298 | 6 | Mendeley | Client 5 |

### Unified 7-Class Mapping

All datasets are mapped to a unified 7-class schema:

| Class | Abbreviation | Description |
|-------|--------------|-------------|
| 0 | AK/AKIEC | Actinic Keratosis |
| 1 | BCC | Basal Cell Carcinoma |
| 2 | BKL | Benign Keratosis |
| 3 | DF | Dermatofibroma |
| 4 | MEL | Melanoma |
| 5 | NV | Melanocytic Nevus |
| 6 | VASC | Vascular Lesion |

###  Recommended: Manual Download

**For significantly faster download speeds, we strongly recommend downloading datasets manually via your web browser** rather than using the API downloader. Browser downloads are typically 10-50x faster than API-based downloads.

#### Download Links

| Dataset | Download Link | Size |
|---------|---------------|------|
| **HAM10000** | [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | ~2.5GB |
| **ISIC 2018** | [ISIC Archive](https://challenge.isic-archive.com/data/#2018) | ~2.5GB |
| **ISIC 2019** | [ISIC Archive](https://challenge.isic-archive.com/data/#2019) | ~9GB |
| **ISIC 2020** | [ISIC Archive](https://challenge.isic-archive.com/data/#2020) | ~25GB |
| **PAD-UFES-20** | [Mendeley](https://data.mendeley.com/datasets/zr7vgbcyr2/1) | ~1.2GB |

#### Manual Setup Steps

1. **Download** each dataset from the links above
2. **Extract** the archives
3. **Organize** into the following structure:

```
data/
├── HAM10000/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   │   └── *.jpg
│   └── HAM10000_images_part_2/
│       └── *.jpg
│
├── ISIC2018/
│   ├── ISIC2018_Task3_Training_GroundTruth.csv
│   └── ISIC2018_Task3_Training_Input/
│       └── *.jpg
│
├── ISIC2019/
│   ├── ISIC_2019_Training_GroundTruth.csv
│   └── ISIC_2019_Training_Input/
│       └── *.jpg
│
├── ISIC2020/
│   ├── train.csv
│   └── train/
│       └── *.jpg
│
└── PAD-UFES-20/
    ├── metadata.csv
    ├── imgs_part_1/
    ├── imgs_part_2/
    └── imgs_part_3/
        └── *.png
```

4. **Verify** the installation:

```bash
python run_download.py --verify
```

#### Alternative: API Download (Slower)

If you prefer automated downloading:

```bash
# Download all datasets (may take several hours)
python run_download.py --download-all --workers 16

# Download specific dataset
python run_download.py --download ISIC2019
```

---

## Configuration System

All experiments are configured via **YAML files** in the `configs/` directory. This provides reproducibility and easy parameter tuning.

### Main Configuration Files

| File | Purpose |
|------|---------|
| `dscatnet_federated_ham10000_non_iid.yaml` | Primary FL experiment config (non-IID) |
| `dscatnet_centralized_original.yaml` | Centralized baseline config |
| `model_config.yaml` | DSCATNet architecture settings |
| `fl_config.yaml` | FL framework defaults |

### Configuration Structure

```yaml
# Example: dscatnet_federated_ham10000_non_iid.yaml

federated:
  experiment:
    name: dscatnet_federated_isic2019
    description: "FL benchmark on ISIC2019"

  # Data
  data_root: ./data
  output_dir: ./outputs
  datasets:
    - ISIC2019

  # Model
  model:
    variant: paper        # tiny, small, paper, base
    image_size: 224
    num_classes: 7

  # Training
  training:
    batch_size: 4
    lr: 0.001
    local_epochs: 1
    num_rounds: 25

  # Federation
  federation:
    num_clients: 4  # Adjust based on number of datasets used
    data_partition_type: dirichlet    # natural, dirichlet, label_skew, quantity_skew, iid
    dirichlet_alpha: 0.5      # Lower = more non-IID

  # Augmentation
  augmentation:
    level: medium             # light, medium, heavy
```

### Non-IID Distribution Types

| Type | Description | When to Use |
|------|-------------|-------------|
| `natural` | Each dataset = 1 client | Simulating real hospitals |
| `dirichlet` | Dirichlet-based label skew | Controlled heterogeneity studies |
| `label_skew` | Artificial label imbalance | Extreme non-IID testing |
| `quantity_skew` | Different sample counts | Unbalanced client scenarios |

---

## Training Pipeline

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. CONFIGURATION                                               │
│     └── Load YAML config → SimulationConfig/CentralizedConfig   │
│                                                                 │
│  2. DATA SETUP                                                  │
│     ├── Load datasets (HAM10000, ISIC, PAD-UFES-20)            │
│     ├── Apply transforms (resize, normalize, augment)          │
│     └── Create train/val splits (stratified)                   │
│                                                                 │
│  3. MODEL INITIALIZATION                                        │
│     └── Create DSCATNet(variant, num_classes, pretrained)      │
│                                                                 │
│  4. TRAINING LOOP                                               │
│     ├── Centralized: Standard epoch-based training             │
│     └── Federated:                                             │
│         ├── Distribute model to clients                        │
│         ├── Local training (local_epochs)                      │
│         ├── Aggregate weights (FedAvg)                         │
│         └── Repeat for num_rounds                              │
│                                                                 │
│  5. CHECKPOINTING                                               │
│     ├── Save best_model.pt (best val accuracy)                 │
│     └── Save periodic checkpoints                              │
│                                                                 │
│  6. EVALUATION                                                  │
│     └── Compute metrics on validation/test set                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Running Experiments

#### Federated Learning (Recommended)

```bash
# Using config file (recommended)
python run_experiment.py --mode federated --config configs/dscatnet_federated_ham10000_non_iid.yaml

# Override specific settings
python run_experiment.py --mode federated \
    --config configs/dscatnet_federated_ham10000_non_iid.yaml \
    --rounds 50 \
    --batch-size 16 \
    --model-variant paper
```

#### Centralized Training (Baseline)

```bash
# Using config file
python run_experiment.py --mode centralized --config configs/dscatnet_centralized_original.yaml

# With overrides
python run_experiment.py --mode centralized \
    --config configs/dscatnet_centralized_original.yaml \
    --epochs 50 \
    --augmentation medium
```

#### Comparison Experiment

```bash
python run_experiment.py --mode comparison --config configs/experiment_config.yaml
```

#### Standalone Model Evaluation

```bash
# Evaluate a trained checkpoint on specific datasets
python run_experiment.py --mode evaluate \
    --checkpoint outputs/federated_20260126_005720/checkpoints/best_model.pt \
    --datasets HAM10000 ISIC2019

# Save evaluation results to file
python run_experiment.py --mode evaluate \
    --checkpoint outputs/experiment/checkpoints/best_model.pt \
    --output-dir ./evaluation_results
```

---

## Checkpoints & Resume Training

### Checkpoint Structure

Checkpoints are saved in `outputs/<experiment_name>/checkpoints/`:

```
checkpoints/
├── best_model.pt             # Best model weights only (for inference)
├── best_checkpoint.pt        # Full checkpoint with training state (for resumption)
├── checkpoint_epoch_10.pt    # Periodic checkpoint (centralized)
├── checkpoint_round_5.pt     # Periodic checkpoint (federated)
└── checkpoint_round_10.pt
```

### Checkpoint Contents

**Centralized checkpoints** contain full training state for perfect resumption:

```python
{
    "epoch": 10,                          # Current epoch number
    "model_state_dict": {...},            # Model weights
    "optimizer_state_dict": {...},        # Optimizer state (momentum, etc.)
    "scheduler_state_dict": {...},        # LR scheduler position
    "scaler_state_dict": {...},           # AMP scaler state (if enabled)
    "metrics": {
        "val_accuracy": 0.85,
        "val_loss": 0.42,
        ...
    },
    "config": {...},                      # Training configuration
    "history": {...},                     # Full training history
    "best_val_accuracy": 0.85,
    "best_epoch": 10,
    "epochs_without_improvement": 0,
}
```

**Federated checkpoints** contain:

```python
{
    "round": 10,                          # Current round number
    "model_state_dict": {...},            # Global model weights
    "metrics": {...},                     # Round metrics
    "config": {...},                      # Simulation configuration
    "history": {...},                     # Full training history
    "best_val_accuracy": 0.78,
    "best_round": 8,
    "rounds_without_improvement": 2,
}
```

### Resume Training from Checkpoint

**Resume Centralized Training:**

```bash
# Resume from best checkpoint (continues training)
python run_experiment.py --mode centralized \
    --resume outputs/centralized_20260125_120000/checkpoints/best_checkpoint.pt \
    --epochs 150

# Resume with config file + checkpoint
python run_experiment.py --mode centralized \
    --config configs/dscatnet_centralized_original.yaml \
    --resume outputs/experiment/checkpoints/checkpoint_epoch_50.pt \
    --epochs 100
```

**Resume Federated Training:**

```bash
# Resume FL from round 25 checkpoint, continue to round 50
python run_experiment.py --mode federated \
    --resume outputs/federated_20260126_005720/checkpoints/checkpoint_round_25.pt \
    --rounds 50

# Resume with config + new experiment name
python run_experiment.py --mode federated \
    --config configs/dscatnet_federated_ham10000_non_iid.yaml \
    --resume outputs/federated_20260126_005720/checkpoints/checkpoint_round_10.pt \
    --rounds 30 \
    --experiment-name federated_continued
```

### Loading Checkpoints in Code

```python
import torch
from src.models.dscatnet import create_dscatnet

# Create model
model = create_dscatnet(variant="paper", num_classes=7)

# Load checkpoint
checkpoint = torch.load("outputs/experiment/checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Check training progress
print(f"Loaded from epoch/round: {checkpoint.get('epoch') or checkpoint.get('round')}")
print(f"Best accuracy: {checkpoint.get('best_val_accuracy', checkpoint.get('val_accuracy')):.4f}")
```

---

## Model Evaluation

### Evaluation Metrics

The evaluation system computes comprehensive metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Balanced Accuracy** | Mean per-class accuracy (handles imbalance) |
| **Precision (macro)** | Average precision across classes |
| **Recall (macro)** | Average recall across classes |
| **F1-Score (macro/weighted)** | Harmonic mean of precision & recall |
| **AUC-ROC** | Area under ROC curve (one-vs-rest) |
| **Confusion Matrix** | Per-class prediction breakdown |
| **Per-Class Metrics** | Sensitivity/specificity per class |

### Running Evaluation

#### Evaluate a Trained Model

```python
from src.models.dscatnet import create_dscatnet
from src.evaluation.metrics import ModelEvaluator
from src.data.datasets import ISIC2019Dataset
from src.data.preprocessing import get_val_transforms
from torch.utils.data import DataLoader
import torch

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = create_dscatnet(variant="paper", num_classes=7)
checkpoint = torch.load("outputs/experiment/checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

# Prepare test data
transform = get_val_transforms(img_size=224)
test_dataset = ISIC2019Dataset(
    root_dir="data/ISIC2019/ISIC_2019_Training_Input",
    csv_path="data/ISIC2019/ISIC_2019_Training_GroundTruth.csv",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate
evaluator = ModelEvaluator(model, device, num_classes=7)
results = evaluator.evaluate(test_loader)

# Print report
evaluator.print_report(results)

# Access specific metrics
print(f"Accuracy: {results.accuracy:.4f}")
print(f"F1 (macro): {results.f1_macro:.4f}")
print(f"AUC-ROC: {results.auc_macro:.4f}")
```

### Evaluation After Training

Evaluation is automatically performed at the end of each experiment. Results are saved in:

```
outputs/<experiment_name>/
├── results.json              # Final metrics + training history
├── config.json               # Experiment configuration
├── metrics/                  # Real-time CSV metrics
│   └── <name>_metrics.csv
└── checkpoints/
    └── best_model.pt         # Best model weights
```

### Metrics JSON Structure

```json
{
    "accuracy": 0.8542,
    "balanced_accuracy": 0.7891,
    "precision_macro": 0.8123,
    "recall_macro": 0.7891,
    "f1_macro": 0.7956,
    "f1_weighted": 0.8412,
    "auc_macro": 0.9234,
    "per_class_metrics": {
        "AK": {"accuracy": 0.82, "precision": 0.79, "recall": 0.75, "support": 312},
        "BCC": {"accuracy": 0.88, "precision": 0.85, "recall": 0.82, "support": 514}
    }
}
```

---

## CLI Reference

### `run_experiment.py` (Main Entry Point)

```bash
python run_experiment.py --mode <MODE> [OPTIONS]
```

#### Mode Selection

| Argument | Type | Description |
|----------|------|-------------|
| `--mode` | required | `centralized`, `federated`, `comparison`, or `evaluate` |
| `--config` | path | YAML configuration file (CLI args override config values) |

#### Common Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--data-root` | path | Root directory for datasets (default: `./data`) |
| `--output-dir` | path | Output directory (default: `./outputs`) |
| `--experiment-name` | string | Custom experiment name |
| `--batch-size` | int | Batch size for training/evaluation |
| `--lr` | float | Learning rate |
| `--datasets` | list | Specific datasets: `HAM10000 ISIC2018 ISIC2019 ISIC2020 PAD-UFES-20` |

#### Model Configuration

| Argument | Type | Description |
|----------|------|-------------|
| `--model-variant` | string | DSCATNet variant: `tiny` (~5M), `small` (~29.4M), `paper` (~29.4M, default), `base` (~39M) |
| `--num-classes` | int | Number of output classes (default: 7) |
| `--image-size` | int | Input image size (default: 224) |

#### Training Hyperparameters

| Argument | Type | Description |
|----------|------|-------------|
| `--weight-decay` | float | Weight decay for optimizer (default: 0.0) |
| `--augmentation` | string | Data augmentation level: `none`, `light`, `medium`, `heavy` |
| `--early-stopping` | int | Early stopping patience (epochs/rounds without improvement) |
| `--checkpoint-interval` | int | Save checkpoint every N epochs/rounds |
| `--num-workers` | int | Number of data loader workers |

#### Centralized-Specific Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--epochs` | int | Number of training epochs |
| `--warmup-epochs` | int | Number of warmup epochs for LR scheduler |
| `--scheduler` | string | LR scheduler type: `cosine`, `plateau` |
| `--val-split` | float | Validation split ratio (default: 0.15) |
| `--no-amp` | flag | Disable automatic mixed precision (AMP) |

#### Federated-Specific Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--rounds` | int | Number of FL communication rounds |
| `--clients` | int | Number of FL clients |
| `--local-epochs` | int | Local epochs per round |
| `--data-partition-type` | string | `natural`, `dirichlet`, `label_skew`, `quantity_skew`, `iid` |
| `--dirichlet-alpha` | float | Dirichlet alpha (lower = more non-IID) |
| `--participation` | float | Client participation rate per round (0.0-1.0) |

#### Checkpoint & Resume Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--resume` | path | Checkpoint path to resume training from (centralized or federated) |
| `--checkpoint` | path | Checkpoint path for evaluation mode (`--mode evaluate`) |

### `run_download.py` (Dataset Management)

```bash
python run_download.py [OPTIONS]
```

| Argument | Description |
|----------|-------------|
| `--verify` | Verify existing dataset installation |
| `--instructions` | Print manual download instructions |
| `--setup` | Interactive setup wizard |
| `--download <DATASET>` | Download specific dataset |
| `--download-all` | Download all datasets |
| `--workers N` | Parallel download workers (default: 8) |
| `--force` | Force re-download existing files |

---

## Experiment Outputs

### Output Directory Structure

```
outputs/
└── <experiment_name>/
    ├── checkpoints/
    │   ├── best_model.pt             # Best weights only (inference)
    │   ├── best_checkpoint.pt        # Full state (resumption)
    │   ├── checkpoint_epoch_10.pt    # Periodic (centralized)
    │   └── checkpoint_round_5.pt    # Periodic (federated)
    ├── config.json                   # Experiment configuration
    ├── results.json                  # Final metrics + training history
    ├── metrics/                      # Real-time CSV metrics
    │   └── <name>_metrics.csv
    └── experiment.log                # Full training log
```

### Training History (in results.json)

The `results.json` file written at experiment completion includes the full training history:

```json
{
    "best_val_accuracy": 0.85,
    "best_epoch": 42,
    "total_time_seconds": 3600.0,
    "history": {
        "epochs": [1, 2, 3],
        "train_loss": [2.1, 1.8, 1.5],
        "val_loss": [2.0, 1.7, 1.4],
        "val_accuracy": [0.35, 0.52, 0.61],
        "learning_rate": [0.001, 0.001, 0.0009]
    },
    "environment": {
        "python_version": "3.13.3",
        "pytorch_version": "2.7.0",
        "cuda_available": true
    }
}
```

---

## Notebooks

Interactive Jupyter notebooks for exploration, evaluation, and analysis are provided in the `notebooks/` directory.

| Notebook | Description |
|----------|-------------|
| [01_dataset_exploration.ipynb](notebooks/01_dataset_exploration.ipynb) | Dataset verification, class distribution analysis, image statistics, non-IID visualization, preprocessing pipeline testing, and sample visualization |
| [02_model_evaluation.ipynb](notebooks/02_model_evaluation.ipynb) | Comprehensive model evaluation including performance metrics, confusion matrices, per-class analysis, ROC curves, confidence distribution analysis, and artifact export |
| [03_fl_vs_centralized_comparison.ipynb](notebooks/03_fl_vs_centralized_comparison.ipynb) | Head-to-head comparison between centralized training and federated learning approaches with statistical significance testing |

### Notebook 02: Model Evaluation & Export

Notebook 02 evaluates trained models and exports comprehensive artifacts:

**Configuration**: Select the experiment and dataset in the configuration cell, then run all cells sequentially.

**Key Exports** (saved to `outputs/evaluation_<experiment>/` for each dataset):

| File | Description |
|------|-------------|
| `results_latest.json` | Current evaluation snapshot with metrics and per-sample predictions |
| `results_<timestamp>.json` | Timestamped archive of evaluation results |
| `metrics_summary.csv` | Summary metrics (accuracy, F1, AUC, etc.) |
| `per_class_metrics.csv` | Per-class performance breakdown |
| `confusion_matrix.csv` | Confusion matrix in tabular form |
| `kpi_dashboard.png`, `confusion_matrix.png`, `per_class_metrics.png`, `roc_curves.png`, `confidence_analysis.png` | Visualizations |

**Results JSON Structure**:

```json
{
    "evaluation_timestamp": "2026-05-07T23:06:35...",
    "dataset": "HAM10000",
    "model_variant": "centralized",
    "num_samples": 10015,
    "metrics": {
        "accuracy": 0.8542,
        "balanced_accuracy": 0.7891,
        "f1_macro": 0.7956,
        "auc_macro": 0.9234
    },
    "per_class_metrics": { ... },
    "per_class_auc": { ... },
    "confusion_matrix": [...],
    "confidence_stats": { ... },
    "labels": [5, 1, 3, ...],                    // Ground truth per sample
    "predictions": [5, 1, 3, ...],              // Model predictions per sample
    "sample_ids": ["path/to/img1.jpg", ...],   // Unique identifier per sample
    "sample_predictions": [                     // Detailed per-sample results
        {
            "sample_index": 0,
            "sample_id": "HAM10000_000000",
            "y_true": 5,
            "y_pred": 5,
            "correct": true,
            "confidence": 0.9876
        },
        ...
    ]
}
```

**Purpose of Per-Sample Data**: The `labels`, `predictions`, `sample_ids`, and `sample_predictions` enable exact paired statistical tests in Notebook 03 (e.g., McNemar test for centralized vs FL) without requiring recomputation.

### Running Notebooks

```bash
# Start Jupyter Lab
jupyter lab notebooks/

# Or start Jupyter Notebook
jupyter notebook notebooks/
```

> **Note**: Ensure the virtual environment is activated and datasets are downloaded before running notebooks. Notebook 02 requires `results` object from model evaluation; Notebook 03 requires evaluation artifacts from Notebook 02.

---

## Testing

The project includes comprehensive unit tests for all major components.

### Test Modules

| Module | Description |
|--------|-------------|
| `test_centralized.py` | Tests for centralized training configuration and trainer |
| `test_checkpoints.py` | Tests for checkpoint saving, loading, and management |
| `test_cli.py` | Tests for CLI argument parsing and validation |
| `test_client.py` | Tests for Flower FL client |
| `test_config_loading.py` | Tests for YAML config loading and schema validation |
| `test_config_schema.py` | Tests for configuration schema validation |
| `test_datasets.py` | Tests for dataset registry and loading functions |
| `test_download.py` | Tests for download functionality |
| `test_evaluation.py` | Tests for evaluation metrics and visualization functions |
| `test_helpers.py` | Tests for seed, device, formatting, and other utilities |
| `test_integration.py` | End-to-end integration tests (marked `@slow`) |
| `test_logging_utils.py` | Tests for MetricsTracker, CSV logging, and resume safety |
| `test_model_evaluator.py` | Tests for ModelEvaluator integration |
| `test_models.py` | Tests for DSCATNet model architecture |
| `test_preprocessing.py` | Tests for image transforms, augmentation levels, and normalization |
| `test_simulation.py` | Tests for FL simulation, FedAvg aggregation, and client management |
| `test_splits.py` | Tests for IID/Non-IID data splitting utilities |
| `test_strategy.py` | Tests for DSCATNetFedAvg custom strategy |
| `test_verify.py` | Tests for dataset verification utilities |
| `test_visualization.py` | Tests for plotting and visualization functions |

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run all tests with pytest (verbose)
pytest tests/ -v

# Run specific test module
pytest tests/test_simulation.py -v

# Run specific test class
pytest tests/test_simulation.py::TestFLSimulator -v

# Run with coverage report
pytest --cov=src tests/

# Run with coverage and HTML report
pytest --cov=src --cov-report=html tests/
```

### Test Results

Expected output:

```
======================== test session starts ========================
collected 467 items / 10 deselected / 457 selected

tests/test_centralized.py ........................              [  5%]
tests/test_checkpoints.py ..................                    [  9%]
tests/test_cli.py .......................                       [ 14%]
tests/test_client.py ............                               [ 17%]
tests/test_config_loading.py ..........                         [ 19%]
tests/test_config_schema.py ....................................[ 27%]
tests/test_datasets.py .....................                    [ 32%]
tests/test_download.py ......................                   [ 36%]
tests/test_evaluation.py .......                                [ 38%]
tests/test_helpers.py ......................                    [ 43%]
tests/test_logging_utils.py ...........................         [ 49%]
tests/test_model_evaluator.py .............                     [ 52%]
tests/test_models.py ..................                         [ 56%]
tests/test_preprocessing.py ......                              [ 57%]
tests/test_simulation.py .....................                  [ 62%]
tests/test_splits.py ........                                   [ 64%]
tests/test_strategy.py ...............                          [ 67%]
tests/test_verify.py ..........................                 [ 73%]
tests/test_visualization.py ........................            [ 78%]

================= 457 passed, 10 deselected in ~100s =================
```

Test coverage: **≥80%** across all source modules.

> **Note**: Integration tests are deselected by default (marked `@pytest.mark.slow`). Run them with `pytest -m slow tests/`.

---

## Troubleshooting

### CUDA Issues on Windows

```powershell
# Reinstall PyTorch with CUDA support
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

### Out of Memory (OOM)

1. **Reduce batch size** in config: `batch_size: 4`
2. **Reduce num_workers**: `num_workers: 2`
3. **Use smaller model variant**: `variant: tiny`

### Dataset Not Found

```bash
# Verify dataset structure
python run_download.py --verify

# Check expected paths
python run_download.py --instructions
```

---

## Documentation

Additional documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Documentation index and navigation |
| [docs/config-options-guide.md](docs/config-options-guide.md) | Complete configuration reference |
| [docs/architecture.md](docs/architecture.md) | System architecture and module documentation |
| [docs/benchmark-comparison.md](docs/benchmark-comparison.md) | Federated vs centralized benchmark fairness audit |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributing guidelines and code style |

For AI assistants (Claude, GPT, etc.), see [docs/CLAUDE.md](docs/CLAUDE.md) for comprehensive codebase context.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@thesis{chen2026dscatnet_fl,
    title={Federated Learning for Skin Cancer Classification using Lightweight Vision Transformers},
    author={Chen, Leonardo},
    year={2026},
    school={Universidad Politécnica de Madrid}
}
```

**DSCATNet Reference:**

```bibtex
@article{dscatnet2024,
    title={DSCATNet: Dual-Scale Cross-Attention Vision Transformer for Skin Cancer Classification},
    journal={PLOS ONE},
    year={2024}
}
```

---

## License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- DSCATNet authors for the original architecture
- Flower team for the FL framework
- ISIC Archive for the dermoscopy datasets
- Universidad Politécnica de Madrid
