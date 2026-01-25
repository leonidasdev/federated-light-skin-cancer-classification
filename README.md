# Federated Learning for Skin Cancer Classification with DSCATNet

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flower 1.13+](https://img.shields.io/badge/flower-1.13+-green.svg)](https://flower.dev/)
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
16. [Citation](#citation)
17. [License](#license)

---

## Overview

This project evaluates the **Dual-Scale Cross-Attention Vision Transformer (DSCATNet)** in a **Federated Learning** setting for dermoscopic skin lesion classification.

**This is a Master's thesis project** investigating whether lightweight Vision Transformers can maintain their classification accuracy under federated learning constraints, specifically with non-IID (non-Independent and Identically Distributed) data across multiple simulated hospitals/institutions.

### Key Features

- **DSCATNet Implementation**: Lightweight ViT with dual-scale cross-attention (~15M parameters)
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
│   ├── dscatnet_federated_ham10000.yaml    # Main FL experiment config
│   ├── dscatnet_centralized_original.yaml  # Centralized baseline config
│   ├── dscatnet_federated_padufes20.yaml  # Alternative FL config
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
│   ├── federated_YYYYMMDD_HHMMSS/
│   │   ├── checkpoints/
│   │   │   ├── best_model.pt
│   │   │   └── checkpoint_round_*.pt
│   │   ├── config.json
│   │   ├── history.json
│   │   └── experiment.log
│   └── centralized_YYYYMMDD_HHMMSS/
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
│   ├── training/                     # Baseline training
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
│       ├── helpers.py                # Seed, device, formatting
│       └── logging_utils.py          # Logging configuration
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_dataset_exploration.ipynb  # Dataset analysis & verification
│   ├── 02_model_evaluation.ipynb     # Model evaluation & metrics
│   └── 03_fl_vs_centralized_comparison.ipynb  # FL vs centralized comparison
│
├── tests/                            # Unit tests
│   ├── test_centralized.py           # Centralized training tests
│   ├── test_evaluation.py            # Evaluation metrics tests
│   ├── test_preprocessing.py         # Preprocessing pipeline tests
│   ├── test_simulation.py            # FL simulation tests
│   └── test_splits.py                # Data splitting tests
│
├── run_experiment.py                 # Main entry point
├── run_fl.py                         # Quick FL runner
├── run_download.py                   # Dataset downloader
├── run_tests.py                      # Test runner
├── requirements.txt                  # Python dependencies
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
│  (6 blocks, 6 heads, dim=384)   │
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
│   Global Average Pooling        │
│   + Classification Head         │
│   → 7 classes (softmax)         │
└─────────────────────────────────┘
```

### Model Variants

| Variant | Embed Dim | Depth | Heads | Parameters | Use Case |
|---------|-----------|-------|-------|------------|----------|
| `tiny`  | 192       | 4     | 3     | ~5M        | Resource-constrained FL clients |
| `small` | 384       | 6     | 6     | ~15M       | **Default** - balanced performance |
| `base`  | 384       | 8     | 6     | ~20M       | Maximum accuracy |

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
| Python   | 3.9+    | 3.10+       |
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
| `dscatnet_federated_ham10000.yaml` | Primary FL experiment config |
| `dscatnet_centralized_original.yaml` | Centralized baseline config |
| `model_config.yaml` | DSCATNet architecture settings |
| `fl_config.yaml` | FL framework defaults |

### Configuration Structure

```yaml
# Example: dscatnet_federated_ham10000.yaml

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
    variant: small        # tiny, small, base
    image_size: 224
    num_classes: 7

  # Training
  training:
    batch_size: 8
    lr: 0.001
    local_epochs: 1
    num_rounds: 25

  # Federation
  federation:
    num_clients: 4
    noniid_type: dirichlet    # natural, dirichlet, label_skew
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
python run_experiment.py --mode federated --config configs/dscatnet_federated_ham10000.yaml

# Override specific settings
python run_experiment.py --mode federated \
    --config configs/dscatnet_federated_ham10000.yaml \
    --rounds 50 \
    --batch-size 16
```

#### Centralized Training (Baseline)

```bash
# Using config file
python run_experiment.py --mode centralized --config configs/dscatnet_centralized_original.yaml

# With overrides
python run_experiment.py --mode centralized \
    --config configs/dscatnet_centralized_original.yaml \
    --epochs 50
```

#### Comparison Experiment

```bash
python run_experiment.py --mode comparison --config configs/experiment_config.yaml
```

---

## Checkpoints & Resume Training

### Checkpoint Structure

Checkpoints are saved in `outputs/<experiment_name>/checkpoints/`:

```
checkpoints/
├── best_model.pt           # Best model (highest val accuracy)
├── checkpoint_round_5.pt   # Periodic checkpoint
├── checkpoint_round_10.pt
└── checkpoint_round_15.pt
```

### Checkpoint Contents

Each `.pt` file contains:

```python
{
    "epoch": 10,                      # Round/epoch number
    "model_state_dict": {...},        # Model weights
    "optimizer_state_dict": {...},    # Optimizer state
    "scheduler_state_dict": {...},    # LR scheduler state
    "metrics": {
        "val_accuracy": 0.85,
        "val_loss": 0.42,
        ...
    }
}
```

### Resume Training from Checkpoint

```bash
# Resume centralized training from checkpoint
python run_experiment.py --mode centralized \
    --config configs/dscatnet_centralized_original.yaml \
    --resume outputs/centralized_20260125_120000/checkpoints/best_model.pt

# Resume with specific epoch count (continues from checkpoint)
python run_experiment.py --mode centralized \
    --resume outputs/experiment/checkpoints/checkpoint_epoch_50.pt \
    --epochs 100
```

### Loading Checkpoints in Code

```python
import torch
from src.models.dscatnet import create_dscatnet

# Create model
model = create_dscatnet(variant="small", num_classes=7)

# Load checkpoint
checkpoint = torch.load("outputs/experiment/checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Check training progress
print(f"Loaded from epoch: {checkpoint['epoch']}")
print(f"Best accuracy: {checkpoint['metrics']['val_accuracy']:.4f}")
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
model = create_dscatnet(variant="small", num_classes=7)
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
├── metrics.json              # Final evaluation metrics
├── history.json              # Training history (loss, accuracy per round)
├── confusion_matrix.png      # Confusion matrix visualization
└── training_curves.png       # Loss/accuracy curves
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

| Argument | Type | Description |
|----------|------|-------------|
| `--mode` | required | `centralized`, `federated`, or `comparison` |
| `--config` | path | YAML configuration file |
| `--data-root` | path | Root directory for datasets (default: `./data`) |
| `--output-dir` | path | Output directory (default: `./outputs`) |
| `--experiment-name` | string | Custom experiment name |
| `--batch-size` | int | Override batch size |
| `--lr` | float | Override learning rate |
| `--datasets` | list | Specific datasets: `HAM10000 ISIC2019 ...` |

**Centralized-specific:**

| Argument | Type | Description |
|----------|------|-------------|
| `--epochs` | int | Number of training epochs |
| `--resume` | path | Checkpoint path to resume from |

**Federated-specific:**

| Argument | Type | Description |
|----------|------|-------------|
| `--rounds` | int | Number of FL rounds |
| `--clients` | int | Number of clients |
| `--local-epochs` | int | Local epochs per round |
| `--noniid-type` | string | `natural`, `dirichlet`, `label_skew`, `quantity_skew` |
| `--dirichlet-alpha` | float | Dirichlet alpha (lower = more non-IID) |

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

### `run_fl.py` (Quick FL Runner)

```bash
python run_fl.py [OPTIONS]
```

| Argument | Description |
|----------|-------------|
| `--quick` | Quick test (5 rounds, small settings) |
| `--full` | Full experiment preset |
| `--data-root` | Data directory |

---

## Experiment Outputs

### Output Directory Structure

```
outputs/
└── federated_20260125_181449/
    ├── checkpoints/
    │   ├── best_model.pt
    │   ├── checkpoint_round_5.pt
    │   └── checkpoint_round_10.pt
    ├── config.json               # Experiment configuration
    ├── history.json              # Training history
    ├── metrics.json              # Final evaluation metrics
    ├── experiment.log            # Full training log
    └── plots/
        ├── training_curves.png
        ├── confusion_matrix.png
        └── per_class_accuracy.png
```

### History JSON (Training Progress)

```json
{
    "rounds": [1, 2, 3],
    "train_loss": [2.1, 1.8, 1.5],
    "val_loss": [2.0, 1.7, 1.4],
    "val_accuracy": [0.35, 0.52, 0.61],
    "learning_rate": [0.001, 0.001, 0.0009]
}
```

---

## Notebooks

Interactive Jupyter notebooks for exploration, evaluation, and analysis are provided in the `notebooks/` directory.

| Notebook | Description |
|----------|-------------|
| [01_dataset_exploration.ipynb](notebooks/01_dataset_exploration.ipynb) | Dataset verification, class distribution analysis, image statistics, non-IID visualization, preprocessing pipeline testing, and sample visualization |
| [02_model_evaluation.ipynb](notebooks/02_model_evaluation.ipynb) | Comprehensive model evaluation with performance metrics, confusion matrices, per-class analysis, ROC curves, and prediction confidence analysis |
| [03_fl_vs_centralized_comparison.ipynb](notebooks/03_fl_vs_centralized_comparison.ipynb) | Head-to-head comparison between centralized training (original DSCATNet paper) and federated learning approaches |

### Running Notebooks

```bash
# Start Jupyter Lab
jupyter lab notebooks/

# Or start Jupyter Notebook
jupyter notebook notebooks/
```

> **Note**: Ensure the virtual environment is activated and datasets are downloaded before running notebooks.

---

## Testing

The project includes comprehensive unit tests for all major components.

### Test Modules

| Module | Description |
|--------|-------------|
| `test_centralized.py` | Tests for centralized training configuration and trainer |
| `test_evaluation.py` | Tests for evaluation metrics and visualization functions |
| `test_preprocessing.py` | Tests for image transforms, augmentation levels, and normalization |
| `test_simulation.py` | Tests for FL simulation, FedAvg aggregation, and client management |
| `test_splits.py` | Tests for IID/Non-IID data splitting utilities |

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
collected 35 items

tests/test_centralized.py ....                                 [ 14%]
tests/test_evaluation.py ....                                  [ 28%]
tests/test_preprocessing.py ......                             [ 45%]
tests/test_simulation.py ........                              [ 68%]
tests/test_splits.py ........                                  [100%]

======================== 32 passed, 3 skipped =======================
```

> **Note**: Some integration tests are skipped by default as they require actual datasets to be present.

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
