# Federated Learning for Skin Cancer Classification with DSCATNet

## Overview
This project adapts the **Dual-Scale Cross-Attention Vision Transformer (DSCATNet)** to a **Federated Learning** environment using the **Flower** framework for dermoscopic skin lesion classification.

## Research Contribution
- First evaluation of DSCATNet in a federated learning setting
- Non-IID data distribution across 4 dermoscopy datasets
- Comparison of centralized vs. federated training paradigms

## Project Structure
```
federated-light-skin-cancer-classification/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dscatnet.py          # DSCATNet architecture
│   │   ├── patch_embedding.py    # Dual-scale patch embeddings
│   │   └── cross_attention.py    # Cross-attention mechanism
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── client.py             # Flower FL client
│   │   ├── server.py             # Flower FL server
│   │   └── strategy.py           # FedAvg strategy
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py           # Dataset loaders
│   │   ├── preprocessing.py      # Standardized preprocessing
│   │   └── splits.py             # IID/Non-IID splits
│   ├── training/
│   │   ├── __init__.py
│   │   ├── centralized.py        # Centralized training
│   │   └── federated.py          # Federated training loop
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py            # Evaluation metrics
│       └── visualization.py      # Plots and diagrams
├── configs/
│   ├── model_config.yaml
│   ├── fl_config.yaml
│   └── experiment_config.yaml
├── experiments/
│   ├── centralized/
│   └── federated/
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
└── README.md
```

## Datasets (FL Clients)
| Client | Dataset    | Classes | Images  |
|--------|------------|---------|---------|
| 1      | HAM10000   | 7       | ~10,015 |
| 2      | ISIC 2018  | 7       | ~10,015 |
| 3      | ISIC 2019  | 8       | ~25,331 |
| 4      | ISIC 2020  | 2       | ~33,126 |

## DSCATNet Architecture
- **Dual-scale patch embeddings**: 8×8 and 16×16 patches
- **Cross-attention**: Information exchange between scales
- **Lightweight transformer encoder**
- **Global average pooling + Softmax classifier**

## Federated Learning Configuration
- Framework: Flower
- Strategy: FedAvg
- Clients: 4
- Rounds: 50-100
- Local epochs: 1-5
- Batch size: 16-32
- Optimizer: Adam (lr=1e-3)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Centralized training
python -m src.training.centralized

# Federated training
python -m src.training.federated
```

## License
Apache 2.0 License
