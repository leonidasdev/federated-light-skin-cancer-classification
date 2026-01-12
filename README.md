# Federated Light Skin Cancer Classification

A research project evaluating the **LMS-ViT** (Lightweight Multi-Scale Vision Transformer, 2025) model under federated learning conditions for skin cancer classification.

## Project Overview

This project investigates the performance of LMS-ViT on skin lesion classification tasks using federated learning, comparing it against centralized training baselines. The federated setup is designed to run in Docker containers orchestrated by Kubernetes on Azure.

### Key Features

- 🔬 **LMS-ViT Model**: Lightweight Multi-Scale Vision Transformer optimized for medical imaging
- 🏥 **Multiple Datasets**: Support for HAM10000, ISIC 2018, ISIC 2019, and ISIC 2020
- 🔄 **Federated Learning**: Flower framework with FedAvg, FedProx, FedAdam, and FedAdagrad strategies
- 📊 **Non-IID Support**: IID, Dirichlet, and pathological data partitioning
- 🐳 **Container-Ready**: Docker and Kubernetes configurations for Azure deployment

## Project Structure

```
federated-light-skin-cancer-classification/
├── src/                          # Main source code package
│   ├── models/                   # LMS-ViT model implementation
│   │   ├── lms_vit.py           # Main model architecture
│   │   └── components.py        # Building blocks (attention, blocks)
│   ├── data/                     # Dataset loaders
│   │   ├── ham10000.py          # HAM10000 dataset
│   │   ├── isic2018.py          # ISIC 2018 dataset
│   │   ├── isic2019.py          # ISIC 2019 dataset
│   │   ├── isic2020.py          # ISIC 2020 dataset
│   │   ├── transforms.py        # Data augmentation
│   │   └── federated.py         # Data partitioning for FL
│   ├── training/                 # Training pipelines
│   │   ├── trainer.py           # Base trainer class
│   │   ├── centralized.py       # Centralized training
│   │   └── callbacks.py         # Training callbacks
│   ├── federated/                # Federated learning
│   │   ├── client.py            # Client implementation
│   │   └── server.py            # Server and simulation
│   └── utils/                    # Utilities
│       ├── logging.py           # Logging utilities
│       ├── metrics.py           # Evaluation metrics
│       ├── visualization.py     # Plotting functions
│       └── seed.py              # Reproducibility
├── configs/                      # Configuration files
│   ├── default.yaml             # Default hyperparameters
│   └── experiments/             # Experiment-specific configs
├── scripts/                      # Training and evaluation scripts
│   ├── train_centralized.py     # Centralized training
│   ├── train_federated.py       # Federated training
│   └── evaluate.py              # Model evaluation
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_federated_analysis.ipynb
├── tests/                        # Unit tests
├── docker/                       # Docker configurations
│   ├── Dockerfile.client        # FL client container
│   ├── Dockerfile.server        # FL server container
│   └── docker-compose.yaml      # Local development
├── k8s/                          # Kubernetes manifests
│   ├── deployments/             # Deployment configs
│   ├── namespace.yaml
│   ├── storage.yaml
│   └── configmap.yaml
├── data/                         # Dataset directory (git-ignored)
├── checkpoints/                  # Model checkpoints (git-ignored)
├── logs/                         # Training logs (git-ignored)
└── results/                      # Experiment results (git-ignored)
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- Docker (for containerized experiments)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/leonidasdev/federated-light-skin-cancer-classification.git
cd federated-light-skin-cancer-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download datasets (see [Data Preparation](#data-preparation))

## Usage

### Centralized Training (Baseline)

```bash
python scripts/train_centralized.py --config configs/experiments/centralized_ham10000.yaml
```

### Federated Learning (Flower Simulation)

```bash
# IID data distribution
python scripts/train_federated.py --config configs/experiments/federated_ham10000.yaml

# Non-IID data distribution (Dirichlet)
python scripts/train_federated.py --config configs/experiments/federated_ham10000.yaml \
  --partition-strategy dirichlet --dirichlet-alpha 0.5
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default.yaml
```

## Data Preparation

Download and organize the datasets:

1. **HAM10000**: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
2. **ISIC 2018**: [ISIC Archive](https://challenge.isic-archive.com/landing/2018/)
3. **ISIC 2019**: [ISIC Archive](https://challenge.isic-archive.com/landing/2019/)
4. **ISIC 2020**: [ISIC Archive](https://challenge.isic-archive.com/landing/2020/)

Place the data in the `data/` directory following this structure:
```
data/
├── ham10000/
│   ├── HAM10000_metadata.csv
│   └── images/
├── isic2018/
├── isic2019/
└── isic2020/
```

## Experiments

### Planned Experiments

1. **Centralized Baseline**: Standard training on pooled data
2. **FL with IID Data**: Federated learning with balanced data across clients
3. **FL with Non-IID Data**: Federated learning with heterogeneous data distributions
4. **Strategy Comparison**: FedAvg vs FedProx vs FedAdam vs FedAdagrad
5. **Communication Efficiency**: Analysis of rounds vs accuracy tradeoff

## Configuration

Hyperparameters are managed through YAML configuration files in `configs/`. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.embed_dim` | Transformer embedding dimension | 384 |
| `model.depth` | Number of transformer layers | 12 |
| `federated.num_clients` | Number of FL clients | 10 |
| `federated.rounds` | Number of FL rounds | 100 |
| `federated.partition.strategy` | Data partitioning (iid/dirichlet) | dirichlet |

## Docker Deployment

### Local Testing

```bash
cd docker
docker-compose up --build
```

### Kubernetes (Azure)

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/deployments/
```

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LMS-ViT architecture based on [paper reference]
- HAM10000 dataset by Tschandl et al.
- ISIC Challenge organizers

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{federated-lmsvit-2025,
  author = {Leonidas Brando},
  title = {Federated Learning for Skin Cancer Classification with LMS-ViT},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/leonidasdev/federated-light-skin-cancer-classification}
}
```
