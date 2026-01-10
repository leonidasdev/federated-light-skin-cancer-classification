"""
Global configuration for the Federated Learning system
for skin cancer classification.
"""

# ==================== MODEL CONFIGURATION ====================
MODEL_CONFIG = {
    # Dimensiones de entrada
    'input_shape': (224, 224, 3),
    
    # Número de clases (lesiones cutáneas)
    'num_classes': 7,
    
    # Arquitectura CNN ligera (Mamun et al. 2025)
    'conv_blocks': [
        {'filters': 32, 'kernel_size': (3, 3), 'pool_size': (2, 2)},
        {'filters': 64, 'kernel_size': (3, 3), 'pool_size': (2, 2)},
        {'filters': 128, 'kernel_size': (3, 3), 'pool_size': (2, 2)}
    ],
    
    # Capa densa
    'dense_units': 256,
    'dropout_rate': 0.5,
    
    # Función de activación
    'activation': 'relu',
    'final_activation': 'softmax'
}

# ==================== TRAINING CONFIGURATION ====================
TRAINING_CONFIG = {
    # Hiperparámetros locales
    'local_epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.001, 
    
    # Optimizador
    'optimizer': 'adam',
    'optimizer_params': {
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-07
    },
    
    # Learning rate scheduler
    'use_lr_scheduler': True,
    'lr_scheduler_type': 'reduce_on_plateau',  # 'reduce_on_plateau' o 'exponential'
    'lr_scheduler_params': {
        'factor': 0.5,
        'patience': 3,
        'min_lr': 1e-6
    },
    
    # Función de pérdida
    'loss_function': 'categorical_crossentropy',  # o 'focal_loss' para desbalanceo
    'focal_loss_params': {
        'alpha': 0.25,
        'gamma': 2.0
    },
    
    # Early stopping
    'use_early_stopping': True,
    'early_stopping_patience': 15,  # Mayor paciencia para FL
    'early_stopping_min_delta': 0.0005,  # Delta más pequeño
    
    # Validation split para entrenamiento local
    'validation_split': 0.15
}

# ==================== FEDERATED SERVER CONFIGURATION ====================
FEDERATED_CONFIG = {
    # Configuración del servidor
    'server_address': '[::]:8080',
    'num_rounds': 50,  # Número de rondas federadas
    'min_fit_clients': 3,  # Mínimo de clientes para entrenamiento
    'min_evaluate_clients': 3,  # Mínimo de clientes para evaluación
    'min_available_clients': 4,  # Mínimo de clientes disponibles (4 nodos totales)
    'fraction_fit': 1.0,  # Fracción de clientes para fit
    'fraction_evaluate': 1.0,  # Fracción de clientes para evaluate
    
    # Estrategia de agregación
    'strategy': 'FedAvg',  # 'FedAvg' o 'FedProx'
    
    # Parámetros FedProx (para datos no-IID)
    'fedprox_mu': 0.01,  # Factor de regularización proximal
    
    # Configuración de timeout
    'timeout': 600,  # segundos (aumentado para 4 nodos)
    
    # Estrategia de selección de clientes
    'client_selection': 'all',  # 'all', 'random', 'weighted'
    
    # Pesos de agregación
    'weighted_aggregation': True,  # Ponderar por número de muestras
}

# ==================== DATA CONFIGURATION ====================
DATA_CONFIG = {
    # Rutas de datasets (ajustar según tu estructura)
    'data_root': './datasets',
    'ham10000_path': './datasets/HAM10000',
    'isic2018_path': './datasets/ISIC2018',
    'isic2019_path': './datasets/ISIC2019',
    'isic2020_path': './datasets/ISIC2020',
    
    
    # Preprocesamiento
    'image_size': (224, 224),
    'normalize': True,  # Normalizar a [0, 1]
    'standardize': False,  # Estandarizar con media/std ImageNet
    
    # Data augmentation
    'augmentation': {
        'rotation_range': 20,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'shear_range': 0.1,
        'zoom_range': 0.1,
        'horizontal_flip': True,
        'vertical_flip': True,
        'fill_mode': 'nearest'
    },
    
    # Distribución de datos
    'split_ratio': {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    },
    
    # Semilla para reproducibilidad
    'random_seed': 42,
    
    # Estrategia de distribución
    'distribution_strategy': 'iid',  # 'iid' o 'non_iid'
    'non_iid_alpha': 0.5,  # Parámetro de concentración Dirichlet (más bajo = más no-IID)
    
    # Balance de clases
    'apply_class_weights': True,
    'oversample_minority': False,  # Aplicar oversampling (SMOTE)
}

# ==================== NODES CONFIGURATION ====================
NODES_CONFIG = {
    'nodes': [
        {
            'node_id': 0,
            'name': 'HAM10000',
            'dataset': 'HAM10000',
            'is_primary': True
        },
        {
            'node_id': 1,
            'name': 'ISIC2018',
            'dataset': 'ISIC2018',
            'is_primary': False
        },
        {
            'node_id': 2,
            'name': 'ISIC2020',
            'dataset': 'ISIC2020',
            'is_primary': False
        },
        {
            'node_id': 3,
            'name': 'ISIC2019',
            'dataset': 'ISIC2019',
            'is_primary': False
        }
    ],
    
    # No external validation dataset configured
}

# ==================== METRICS CONFIGURATION ====================
METRICS_CONFIG = {
    # Métricas generales
    'track_metrics': ['accuracy', 'precision', 'recall', 'f1_macro', 'auc_macro', 'auc_micro'],
    
    # Métricas específicas para melanoma vs no-melanoma
    'binary_melanoma_metrics': True,
    'melanoma_class_index': 0,  # Índice de la clase melanoma
    
    # Matriz de confusión
    'save_confusion_matrix': True,
    
    # Interpretabilidad
    'use_gradcam': True,
    'gradcam_layer': 'last_conv',  # Última capa convolucional
    
    # Logging
    'log_interval': 1,  # Cada cuántas rondas se registran métricas
    'save_model_interval': 5,  # Cada cuántas rondas se guarda el modelo
}

# ==================== SECURITY CONFIGURATION ====================
SECURITY_CONFIG = {
    # Cifrado de comunicaciones
    'use_encryption': True,
    'encryption_protocol': 'TLS',
    'certificate_path': './certs/server.crt',
    'key_path': './certs/server.key',
    
    # Secure aggregation
    'use_secure_aggregation': False,  # Placeholder para implementación futura
    
    # Privacidad diferencial
    'use_differential_privacy': False,  # Placeholder para implementación futura
    'dp_epsilon': 1.0,
    'dp_delta': 1e-5,
    'dp_noise_multiplier': 0.1,
    
    # Validación de clientes
    'authenticate_clients': False,  # Placeholder para autenticación
    'allowed_client_ids': None  # Lista de IDs permitidos, None = todos
}

# ==================== LOGGING CONFIGURATION ====================
LOGGING_CONFIG = {
    'log_dir': './logs',
    'tensorboard_dir': './logs/tensorboard',
    'model_checkpoint_dir': './checkpoints',
    'results_dir': './results',
    
    # Nivel de logging
    'log_level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    
    # Formato
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    
    # Guardar logs
    'save_logs_to_file': True,
    'log_filename': 'federated_training.log'
}

# ==================== CLASS MAPPING ====================
CLASS_NAMES = {
    0: 'MEL',   # Melanoma
    1: 'NV',    # Melanocytic Nevus
    2: 'BCC',   # Basal Cell Carcinoma
    3: 'AKC',   # Actinic Keratosis
    4: 'BKL',   # Benign Keratosis
    5: 'DF',    # Dermatofibroma
    6: 'VASC'   # Vascular Lesion
}

CLASS_NAMES_FULL = {
    0: 'Melanoma',
    1: 'Melanocytic Nevus',
    2: 'Basal Cell Carcinoma',
    3: 'Actinic Keratosis',
    4: 'Benign Keratosis',
    5: 'Dermatofibroma',
    6: 'Vascular Lesion'
}

# ==================== UTILITIES ====================
def get_config(config_name):
    """
    Get a specific configuration by name.

    Args:
        config_name (str): Name of the configuration

    Returns:
        dict: Configuration dictionary
    """
    configs = {
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'federated': FEDERATED_CONFIG,
        'data': DATA_CONFIG,
        'nodes': NODES_CONFIG,
        'metrics': METRICS_CONFIG,
        'security': SECURITY_CONFIG,
        'logging': LOGGING_CONFIG
    }
    return configs.get(config_name, {})


def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"Model: Lightweight CNN (3 conv blocks)")
    print(f"Num classes: {MODEL_CONFIG['num_classes']}")
    print(f"Input shape: {MODEL_CONFIG['input_shape']}")
    print(f"FL strategy: {FEDERATED_CONFIG['strategy']}")
    print(f"Federated rounds: {FEDERATED_CONFIG['num_rounds']}")
    print(f"Active nodes: {len(NODES_CONFIG['nodes'])}")
    print(f"Distribution: {DATA_CONFIG['distribution_strategy']}")
    print(f"Security: {'Enabled' if SECURITY_CONFIG['use_encryption'] else 'Disabled'}")
    print("=" * 60)


if __name__ == '__main__':
    # Imprimir resumen de configuración
    print_config_summary()
