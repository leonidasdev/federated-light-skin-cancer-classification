# Federated Learning para Clasificación de Cáncer de Piel

Sistema de clasificación de lesiones cutáneas usando Federated Learning con Flower y CNN ligera.

## Estructura del Proyecto

```
├── config/              # Configuración global
│   └── config.py        # Parámetros del sistema
├── data/                # Gestión de datos
│   ├── data_loader.py   # Carga de datasets
│   └── preprocessing.py # Preprocesamiento y augmentation
├── models/              # Arquitecturas de modelos
│   └── cnn_model.py     # CNN ligera (Mamun et al. 2025)
├── server/              # Servidor federado
│   └── server.py        # Lógica del servidor Flower
├── client/              # Cliente federado
│   └── client.py        # Entrenamiento local
├── utils/               # Utilidades
│   ├── metrics.py       # Métricas de evaluación
│   ├── logging_utils.py # Sistema de logging
│   └── security.py      # Seguridad y privacidad
├── main_server.py       # Punto de entrada del servidor
├── main_client.py       # Punto de entrada del cliente
└── requirements.txt     # Dependencias
```

## Datasets

- **Base**: HAM10000
- **Nodos adicionales**: ISIC 2018, ISIC 2020, (opcional) ISIC 2019
- **Validación externa**: PH2

## Arquitectura CNN Ligera

Inspirada en Mamun et al. (2025):
- 3 bloques convolucionales (32, 64, 128 filtros)
- MaxPooling después de cada bloque
- Flatten → Dense(256, ReLU, Dropout) → Softmax(7 clases)

## Clases de Lesiones (7)

1. Melanoma (MEL)
2. Melanocytic Nevus (NV)
3. Basal Cell Carcinoma (BCC)
4. Actinic Keratosis (AKC)
5. Benign Keratosis (BKL)
6. Dermatofibroma (DF)
7. Vascular Lesion (VASC)

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Iniciar servidor federado
```bash
python main_server.py
```

### Iniciar cliente
```bash
python main_client.py --node_id 0 --dataset HAM10000
```

## Características

- Federated Learning con Flower
- Agregación FedAvg / FedProx
- Estrategias IID y no-IID
- Data augmentation robusto
- Métricas especializadas (F1, AUC, sensibilidad/especificidad)
- Interpretabilidad con Grad-CAM
- Canales cifrados
- Preparado para privacidad diferencial

## Autor

Leonidas Brando - TFG 2025
