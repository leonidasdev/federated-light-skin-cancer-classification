# Federated Learning for Skin Cancer Classification with DSCATNet

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flower 1.13+](https://img.shields.io/badge/flower-1.13+-green.svg)](https://flower.dev/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## 📖 Overview

This project evaluates the **Dual-Scale Cross-Attention Vision Transformer (DSCATNet)** in a **Federated Learning** setting using the **Flower** framework for dermoscopic skin lesion classification.

**This is a thesis project** investigating whether lightweight Vision Transformers can maintain their accuracy under federated learning constraints, specifically with non-IID data distributions across multiple hospitals/institutions.

## 🎯 Research Contribution

- **Novel evaluation**: First adaptation and evaluation of DSCATNet in federated learning
- **Real-world non-IID scenario**: Each FL client holds a different dermoscopy dataset
- **Comprehensive comparison**: Centralized vs. IID-FL vs. Non-IID-FL
- **Benchmarking**: Comparison with literature on lightweight FL models

## 🏗️ Project Structure

```
federated-light-skin-cancer-classification/
├── configs/                      # Configuration files
│   ├── model_config.yaml         # DSCATNet architecture settings
│   ├── fl_config.yaml            # Federated learning settings
│   └── experiment_config.yaml    # Experiment parameters
├── data/                         # Datasets (download required)
│   ├── HAM10000/
│   ├── ISIC2018/
│   ├── ISIC2019/
│   └── ISIC2020/
├── experiments/                  # Experiment outputs
│   ├── centralized/              # Baseline results
│   └── federated/                # FL results
├── notebooks/                    # Jupyter analysis notebooks
│   └── 01_dataset_exploration.ipynb
├── src/
│   ├── data/                     # Data loading & preprocessing
│   │   ├── datasets.py           # Dataset classes
│   │   ├── download.py           # ISIC Archive API downloader
│   │   ├── preprocessing.py      # Augmentation & normalization
│   │   ├── splits.py             # IID/Non-IID splits
│   │   └── verify.py             # Dataset verification
│   ├── evaluation/               # Metrics & visualization
│   │   ├── metrics.py            # Classification metrics
│   │   └── visualization.py      # Plots & figures
│   ├── federated/                # Flower FL components
│   │   ├── client.py             # FL client
│   │   ├── server.py             # FL server
│   │   ├── simulation.py         # FL simulation
│   │   └── strategy.py           # FedAvg strategy
│   ├── models/                   # DSCATNet implementation
│   │   ├── dscatnet.py           # Main model
│   │   ├── cross_attention.py    # Cross-attention module
│   │   └── patch_embedding.py    # Dual-scale embeddings
│   ├── training/                 # Training loops
│   │   └── centralized.py        # Centralized baseline
│   └── utils/                    # Utilities
├── tests/                        # Unit tests
├── run_download.py               # Dataset download runner
├── run_experiment.py             # Main experiment runner
├── run_fl.py                     # FL simulation runner
├── run_tests.py                  # Test runner
├── requirements.txt              # Dependencies
└── README.md
```

## 📊 Datasets (FL Clients)

Each client simulates a different hospital/institution with its own dermoscopy dataset:

| Client | Dataset | Classes | Images | Class Distribution |
|--------|---------|---------|--------|-------------------|
| 1 | HAM10000 | 7 | ~10,015 | Highly imbalanced (NV dominant) |
| 2 | ISIC 2018 | 7 | ~10,015 | Similar to HAM10000 |
| 3 | ISIC 2019 | 8 | ~25,331 | Includes SCC class |
| 4 | ISIC 2020 | 2 | ~33,126 | Binary (benign/malignant) |

**Classes (7 unified):**
- `AK` - Actinic Keratosis
- `BCC` - Basal Cell Carcinoma
- `BKL` - Benign Keratosis
- `DF` - Dermatofibroma
- `MEL` - Melanoma
- `NV` - Melanocytic Nevus
- `VASC` - Vascular Lesion

### 📥 Dataset Download

All datasets are downloaded from the **official ISIC Archive** (CC-BY-NC license).
No API key required.

**Option 1: Automatic Download via ISIC Archive API (Recommended)**
```bash
# Download all datasets (~78,000 images, may take several hours)
python run_download.py --download-all

# Download specific dataset
python run_download.py --download HAM10000
python run_download.py --download ISIC2018
python run_download.py --download ISIC2019
python run_download.py --download ISIC2020

# Adjust parallel workers for faster download (default: 8)
python run_download.py --download-all --workers 16
```

**Option 2: Interactive Setup Wizard**
```bash
python run_download.py --setup
```

**Option 3: Manual Download**
```bash
# Print detailed instructions
python run_download.py --instructions
```

**Verify Installation:**
```bash
python run_download.py --verify
```
python -m src.data.download --verify
```

**Expected Directory Structure:**
```
data/
├── HAM10000/
│   ├── metadata.csv
│   └── images/
│       └── *.jpg
├── ISIC2018/
│   ├── metadata.csv
│   └── images/
│       └── *.jpg
├── ISIC2019/
│   ├── metadata.csv
│   └── images/
│       └── *.jpg
└── ISIC2020/
    ├── metadata.csv
    └── images/
        └── *.jpg
```

## 🧠 DSCATNet Architecture

| Component | Description |
|-----------|-------------|
| **Dual-Scale Patch Embedding** | 8×8 (fine) and 16×16 (coarse) patches |
| **Cross-Attention** | Information exchange between scales |
| **Transformer Encoder** | 6 blocks, 6 heads, 384 embed dim |
| **Classification Head** | Global average pooling + Softmax |

**Model Parameters:** ~5.8M (lightweight compared to ViT-Base: 86M)

## ⚙️ Federated Learning Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Framework | Flower | FL simulation framework |
| Strategy | FedAvg | Federated Averaging |
| Clients | 4 | One per dataset |
| Rounds | 50-100 | Communication rounds |
| Local Epochs | 1-5 | Training per round |
| Batch Size | 16-32 | Mini-batch size |
| Optimizer | Adam | lr=1e-3, weight_decay=1e-4 |
| Image Size | 224×224 | Input resolution |

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/leonidasdev/federated-light-skin-cancer-classification.git
cd federated-light-skin-cancer-classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import flwr; print('Installation successful!')"
```

**Requirements:**
- Python 3.9+
- CUDA 11.8+ (for GPU training)
- ~16GB RAM
- ~50GB disk space (for datasets)

## 📈 Usage

### 1. Dataset Exploration
```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

### 2. Centralized Training (Baseline)
```bash
python run_experiment.py --mode centralized --epochs 100 --batch-size 32
```

### 3. Federated Learning Simulation
```bash
# Natural Non-IID (each client = different dataset)
python run_experiment.py --mode federated --scenario natural_noniid --rounds 100

# IID Baseline (pooled data, uniform distribution)
python run_experiment.py --mode federated --scenario iid_baseline --rounds 100

# Synthetic Non-IID (Dirichlet α=0.5)
python run_experiment.py --mode federated --scenario moderate_noniid --rounds 100
```

### 4. Full Comparison Experiment
```bash
python run_experiment.py --mode comparison --config configs/experiment_config.yaml
```

## 📊 Evaluation Metrics

### Classification Metrics
- Accuracy, Balanced Accuracy
- Precision, Recall, F1-Score (macro/weighted)
- AUC-ROC (one-vs-rest)
- Confusion Matrix
- Per-class sensitivity/specificity

### Federated Learning Metrics
- Convergence per round
- Communication cost (bytes transmitted)
- IID vs Non-IID performance gap
- Client drift
- Training time per round

## 📚 Literature Comparison

| Paper | Model | Setting | Accuracy |
|-------|-------|---------|----------|
| DSCATNet (PLOS ONE 2024) | DSCATNet | Centralized | ~97-98% |
| CNN vs ViT Benchmark (Elsevier 2026) | Various | Centralized | ViTs > CNNs |
| Lightweight FL (Sci. Reports 2025) | EfficientNetV2S | Federated | ~90% |
| **This Work** | DSCATNet | Federated | TBD |

## 📁 Experiment Outputs

Results are saved in `experiments/`:
```
experiments/
├── centralized/
│   └── centralized_YYYYMMDD_HHMMSS/
│       ├── checkpoints/
│       ├── config.json
│       ├── history.json
│       ├── metrics.json
│       └── plots/
└── federated/
    └── natural_noniid_YYYYMMDD_HHMMSS/
        ├── checkpoints/
        ├── client_metrics/
        ├── global_metrics.json
        └── plots/
```

## 🧪 Testing

```bash
# Run all tests
python run_tests.py

# Run specific test
pytest tests/test_simulation.py -v
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@thesis{brando2026dscatnet_fl,
  title={Federated Learning for Skin Cancer Classification using Lightweight Vision Transformers},
  author={Brando, Leonidas},
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

## 📄 License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- DSCATNet authors for the original architecture
- Flower team for the FL framework
- ISIC Archive for the dermoscopy datasets
- Universidad Politécnica de Madrid
