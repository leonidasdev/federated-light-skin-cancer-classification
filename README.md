# Federated Learning for Skin Cancer Classification

Skin lesion classification system using Federated Learning with Flower and lightweight CNN.

## Project Structure

```
├── config/              # Global configuration
│   └── config.py        # System parameters
├── data/                # Data management
│   ├── data_loader.py   # Dataset loaders
│   └── preprocessing.py # Preprocessing and augmentation
├── models/              # Model architectures
│   └── cnn_model.py     # Lightweight CNN (Mamun et al. 2025)
├── server/              # Federated server
│   └── server.py        # Flower server logic
├── client/              # Federated client
│   └── client.py        # Local training
├── utils/               # Utilities
│   ├── metrics.py       # Evaluation metrics
│   ├── logging_utils.py # Logging system
│   ├── security.py      # Security and privacy
│   └── gradcam.py       # Grad-CAM interpretability
├── main_server.py       # Server entry point
├── main_client.py       # Client entry point
├── setup_project.py     # Project setup script
└── requirements.txt     # Dependencies
```

## Datasets

- **Primary**: HAM10000
- **Additional Nodes**: ISIC 2018, ISIC 2019, ISIC 2020
- **External Validation**: None configured

## Lightweight CNN Architecture

Inspired by Mamun et al. (2025):
- 3 convolutional blocks (32, 64, 128 filters)
- MaxPooling after each block
- Flatten → Dense(256, ReLU, Dropout) → Softmax(7 classes)

## Lesion Classes (7)

1. Melanoma (MEL)
2. Melanocytic Nevus (NV)
3. Basal Cell Carcinoma (BCC)
4. Actinic Keratosis (AKC)
5. Benign Keratosis (BKL)
6. Dermatofibroma (DF)
7. Vascular Lesion (VASC)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run project setup
python setup_project.py
```

## Usage

### Start federated server
```bash
python main_server.py --strategy FedAvg --rounds 50
```

### Start clients (in separate terminals)
```bash
# Client 0 - HAM10000
python main_client.py --node-id 0 --dataset HAM10000

# Client 1 - ISIC2018
python main_client.py --node-id 1 --dataset ISIC2018

# Client 2 - ISIC2020
python main_client.py --node-id 2 --dataset ISIC2020

# Client 3 - ISIC2019
python main_client.py --node-id 3 --dataset ISIC2019
```

### Monitor with TensorBoard
```bash
tensorboard --logdir=logs/tensorboard
```

## Features

- Federated Learning with Flower framework
- FedAvg / FedProx aggregation strategies
- IID and non-IID data distribution
- Robust data augmentation
- Specialized metrics (F1, AUC, sensitivity/specificity for melanoma)
- Grad-CAM interpretability
- Encrypted channels support
- Differential privacy ready

## Configuration

Key parameters can be adjusted in [config/config.py](config/config.py):

- **Model**: CNN architecture, input size, dropout
- **Training**: Learning rate, batch size, epochs
- **Federated**: Number of rounds, aggregation strategy, minimum clients
- **Data**: Augmentation, IID/non-IID distribution, class weights
- **Metrics**: Tracking, melanoma-specific metrics, Grad-CAM
- **Security**: Encryption, differential privacy, secure aggregation

## Project Status

This is a research project for federated learning applied to skin cancer classification using multiple dermoscopic image datasets.

## Author

Leonardo Chen - 2025
