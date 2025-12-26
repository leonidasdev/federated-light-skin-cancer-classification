"""
Cliente federado para entrenamiento local con Flower.

Responsabilidades:
- Cargar datos locales del nodo
- Entrenar modelo localmente
- Enviar parámetros actualizados al servidor
- Evaluar modelo local
"""

import flwr as fl
import tensorflow as tf
from typing import Dict, Tuple, List
import numpy as np

from config.config import TRAINING_CONFIG, DATA_CONFIG, NODES_CONFIG
from models.cnn_model import create_cnn_model, compile_model
from data.data_loader import load_node_data
from data.preprocessing import ImagePreprocessor, DataAugmentor, create_tf_dataset
from utils.logging_utils import setup_logger
from utils.metrics import calculate_metrics


class FederatedClient(fl.client.NumPyClient):
    """
    Cliente para entrenamiento federado local.
    """
    
    def __init__(self, 
                 node_id: int,
                 dataset_name: str,
                 model: tf.keras.Model = None):
        """
        Inicializa el cliente federado.
        
        Args:
            node_id (int): ID del nodo
            dataset_name (str): Nombre del dataset asignado
            model (keras.Model): Modelo local (si None, se crea uno nuevo)
        """
        super().__init__()
        
        self.node_id = node_id
        self.dataset_name = dataset_name
        self.logger = setup_logger(f'Client_{node_id}')
        
        # Modelo local
        if model is None:
            self.model = create_cnn_model()
            self.model = compile_model(self.model)
        else:
            self.model = model
        
        # Datos
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        # Configuración de entrenamiento
        self.local_epochs = TRAINING_CONFIG['local_epochs']
        self.batch_size = TRAINING_CONFIG['batch_size']
        
        self.logger.info(f"Cliente {node_id} inicializado - Dataset: {dataset_name}")
    
    def load_data(self):
        """
        Carga los datos asignados a este nodo.
        """
        self.logger.info(f"Cargando datos para nodo {self.node_id}...")
        
        # TODO: Implementar carga real de datos
        # Por ahora, usar placeholder
        
        try:
            # Cargar datos del nodo
            self.X_train, self.y_train = load_node_data(
                node_id=self.node_id,
                nodes_config=NODES_CONFIG
            )
            
            # Dividir en train y validación local
            # TODO: Implementar división
            
            self.logger.info(f"Datos cargados: {len(self.X_train) if self.X_train is not None else 0} muestras de entrenamiento")
        
        except Exception as e:
            self.logger.error(f"Error al cargar datos: {e}", exc_info=True)
            raise
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Obtiene los parámetros actuales del modelo local.
        
        Args:
            config (dict): Configuración del servidor
        
        Returns:
            List[np.ndarray]: Pesos del modelo
        """
        return self.model.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Actualiza los parámetros del modelo local con los del servidor.
        
        Args:
            parameters (List[np.ndarray]): Nuevos pesos
        """
        self.model.set_weights(parameters)
        self.logger.debug("Parámetros del modelo actualizados desde el servidor")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Entrena el modelo localmente.
        
        Args:
            parameters (List[np.ndarray]): Parámetros globales del servidor
            config (dict): Configuración de entrenamiento
        
        Returns:
            tuple: (parámetros actualizados, número de muestras, métricas)
        """
        # Actualizar modelo con parámetros globales
        self.set_parameters(parameters)
        
        # Obtener configuración
        server_round = config.get('server_round', 0)
        local_epochs = config.get('local_epochs', self.local_epochs)
        batch_size = config.get('batch_size', self.batch_size)
        
        self.logger.info(f"Ronda {server_round}: Iniciando entrenamiento local ({local_epochs} epochs)")
        
        # TODO: Implementar entrenamiento real
        # Por ahora, placeholder
        
        try:
            # Cargar datos si no están cargados
            if self.X_train is None:
                self.load_data()
            
            # Entrenar modelo
            history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=local_epochs,
                batch_size=batch_size,
                validation_data=(self.X_val, self.y_val) if self.X_val is not None else None,
                verbose=1
            )
            
            # Obtener métricas
            metrics = {
                'loss': float(history.history['loss'][-1]),
                'accuracy': float(history.history['accuracy'][-1])
            }
            
            if 'val_loss' in history.history:
                metrics['val_loss'] = float(history.history['val_loss'][-1])
                metrics['val_accuracy'] = float(history.history['val_accuracy'][-1])
            
            self.logger.info(f"Entrenamiento completado - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
            
            # Retornar parámetros actualizados
            num_samples = len(self.X_train) if self.X_train is not None else 0
            
            return self.get_parameters(config), num_samples, metrics
        
        except Exception as e:
            self.logger.error(f"Error durante entrenamiento: {e}", exc_info=True)
            raise
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evalúa el modelo local.
        
        Args:
            parameters (List[np.ndarray]): Parámetros del servidor
            config (dict): Configuración de evaluación
        
        Returns:
            tuple: (loss, número de muestras, métricas)
        """
        # Actualizar modelo
        self.set_parameters(parameters)
        
        server_round = config.get('server_round', 0)
        self.logger.info(f"Ronda {server_round}: Evaluando modelo local")
        
        # TODO: Implementar evaluación real
        
        try:
            # Evaluar en datos de validación local
            if self.X_val is None:
                self.logger.warning("No hay datos de validación disponibles")
                return 0.0, 0, {}
            
            # Evaluar
            results = self.model.evaluate(
                self.X_val,
                self.y_val,
                batch_size=self.batch_size,
                verbose=0
            )
            
            # Extraer métricas
            loss = float(results[0])
            accuracy = float(results[1]) if len(results) > 1 else 0.0
            
            metrics = {
                'accuracy': accuracy
            }
            
            # Calcular métricas adicionales
            # TODO: Agregar F1, AUC, etc.
            
            num_samples = len(self.X_val)
            
            self.logger.info(f"Evaluación completada - Loss: {loss:.4f}, Acc: {accuracy:.4f}")
            
            return loss, num_samples, metrics
        
        except Exception as e:
            self.logger.error(f"Error durante evaluación: {e}", exc_info=True)
            raise


# ==================== FUNCIONES DE UTILIDAD ====================

def create_client(node_id: int, dataset_name: str) -> FederatedClient:
    """
    Factory para crear cliente federado.
    
    Args:
        node_id (int): ID del nodo
        dataset_name (str): Nombre del dataset
    
    Returns:
        FederatedClient: Cliente configurado
    """
    client = FederatedClient(node_id=node_id, dataset_name=dataset_name)
    client.load_data()
    return client


def start_client(node_id: int, 
                dataset_name: str,
                server_address: str = '[::]:8080') -> None:
    """
    Inicia un cliente y lo conecta al servidor.
    
    Args:
        node_id (int): ID del nodo
        dataset_name (str): Nombre del dataset
        server_address (str): Dirección del servidor
    """
    logger = setup_logger(f'ClientStarter_{node_id}')
    
    logger.info("=" * 60)
    logger.info(f"INICIANDO CLIENTE {node_id}")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Servidor: {server_address}")
    logger.info("=" * 60)
    
    try:
        # Crear cliente
        client = create_client(node_id=node_id, dataset_name=dataset_name)
        
        # Conectar al servidor
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        
        logger.info("Cliente finalizado correctamente")
    
    except Exception as e:
        logger.error(f"Error en cliente: {e}", exc_info=True)
        raise


# ==================== CALLBACKS PERSONALIZADOS ====================

class FederatedCallback(tf.keras.callbacks.Callback):
    """
    Callback personalizado para entrenamiento federado.
    """
    
    def __init__(self, node_id: int, logger=None):
        super().__init__()
        self.node_id = node_id
        self.logger = logger or setup_logger(f'Callback_{node_id}')
    
    def on_epoch_begin(self, epoch, logs=None):
        """Inicio de época."""
        self.logger.debug(f"Nodo {self.node_id} - Época {epoch + 1} iniciada")
    
    def on_epoch_end(self, epoch, logs=None):
        """Fin de época."""
        logs = logs or {}
        self.logger.info(
            f"Nodo {self.node_id} - Época {epoch + 1}: "
            f"Loss={logs.get('loss', 0):.4f}, "
            f"Acc={logs.get('accuracy', 0):.4f}"
        )
    
    def on_train_begin(self, logs=None):
        """Inicio de entrenamiento."""
        self.logger.info(f"Nodo {self.node_id} - Iniciando entrenamiento local")
    
    def on_train_end(self, logs=None):
        """Fin de entrenamiento."""
        self.logger.info(f"Nodo {self.node_id} - Entrenamiento local completado")


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Probando cliente federado...")
    
    # Crear cliente de prueba
    test_client = create_client(node_id=0, dataset_name='HAM10000')
    
    print(f"Cliente creado con {len(test_client.get_parameters({}))} arrays de parámetros")
