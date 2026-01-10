"""
Federated client for local training with Flower.

Responsibilities:
 - Load local node data
 - Train model locally
 - Send updated parameters to the server
 - Evaluate the local model
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
    Client for local federated training.
    """
    
    def __init__(self, 
                 node_id: int,
                 dataset_name: str,
                 model: tf.keras.Model = None):
        """
        Initialize the federated client.

        Args:
            node_id (int): Node ID
            dataset_name (str): Assigned dataset name
            model (keras.Model): Local model (if None, a new one is created)
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
        
        self.logger.info(f"Client {node_id} initialized - Dataset: {dataset_name}")
    
    def load_data(self):
        """
        Load data assigned to this node.
        """
        self.logger.info(f"Loading data for node {self.node_id}...")

        # TODO: implement real data loading
        # For now, use placeholder

        try:
            # Load node data
            self.X_train, self.y_train = load_node_data(
                node_id=self.node_id,
                nodes_config=NODES_CONFIG
            )

            # Split into local train/validation
            # TODO: implement splitting

            self.logger.info(f"Data loaded: {len(self.X_train) if self.X_train is not None else 0} training samples")

        except Exception as e:
            self.logger.error(f"Error loading data: {e}", exc_info=True)
            raise
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get current parameters of the local model.

        Args:
            config (dict): Server configuration

        Returns:
            List[np.ndarray]: Model weights
        """
        return self.model.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Update local model parameters with those from the server.

        Args:
            parameters (List[np.ndarray]): New weights
        """
        self.model.set_weights(parameters)
        self.logger.debug("Model parameters updated from server")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model locally.

        Args:
            parameters (List[np.ndarray]): Global server parameters
            config (dict): Training configuration

        Returns:
            tuple: (updated parameters, number of samples, metrics)
        """
        # Actualizar modelo con parámetros globales
        self.set_parameters(parameters)
        
        # Obtener configuración
        server_round = config.get('server_round', 0)
        local_epochs = config.get('local_epochs', self.local_epochs)
        batch_size = config.get('batch_size', self.batch_size)
        
        self.logger.info(f"Round {server_round}: Starting local training ({local_epochs} epochs)")
        
        # TODO: Implementar entrenamiento real
        # Por ahora, placeholder
        
        try:
            # Load data if not yet loaded
            if self.X_train is None:
                self.load_data()

            # Train model
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
            
            self.logger.info(f"Training completed - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
            
            # Retornar parámetros actualizados
            num_samples = len(self.X_train) if self.X_train is not None else 0
            
            return self.get_parameters(config), num_samples, metrics
        
        except Exception as e:
            self.logger.error(f"Error during training: {e}", exc_info=True)
            raise
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the local model.

        Args:
            parameters (List[np.ndarray]): Server parameters
            config (dict): Evaluation configuration

        Returns:
            tuple: (loss, number of samples, metrics)
        """
        # Actualizar modelo
        self.set_parameters(parameters)
        
        server_round = config.get('server_round', 0)
        self.logger.info(f"Round {server_round}: Evaluating local model")
        
        # TODO: Implementar evaluación real
        
        try:
            # Evaluar en datos de validación local
            if self.X_val is None:
                self.logger.warning("No validation data available")
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
            
            self.logger.info(f"Evaluation completed - Loss: {loss:.4f}, Acc: {accuracy:.4f}")
            
            return loss, num_samples, metrics
        
        except Exception as e:
            self.logger.error(f"Error durante evaluación: {e}", exc_info=True)
            raise


# ==================== UTILITY FUNCTIONS ====================

def create_client(node_id: int, dataset_name: str) -> FederatedClient:
    """
    Factory to create a federated client.

    Args:
        node_id (int): Node ID
        dataset_name (str): Dataset name

    Returns:
        FederatedClient: Configured client
    """
    client = FederatedClient(node_id=node_id, dataset_name=dataset_name)
    client.load_data()
    return client


def start_client(node_id: int, 
                dataset_name: str,
                server_address: str = '[::]:8080') -> None:
    """
    Start a client and connect it to the server.

    Args:
        node_id (int): Node ID
        dataset_name (str): Dataset name
        server_address (str): Server address
    """
    logger = setup_logger(f'ClientStarter_{node_id}')

    logger.info("=" * 60)
    logger.info(f"STARTING CLIENT {node_id}")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Server: {server_address}")
    logger.info("=" * 60)

    try:
        # Create client
        client = create_client(node_id=node_id, dataset_name=dataset_name)

        # Connect to server
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )

        logger.info("Client finished successfully")

    except Exception as e:
        logger.error(f"Error in client: {e}", exc_info=True)
        raise


# ==================== CUSTOM CALLBACKS ====================

class FederatedCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for federated training.
    """
    
    def __init__(self, node_id: int, logger=None):
        super().__init__()
        self.node_id = node_id
        self.logger = logger or setup_logger(f'Callback_{node_id}')
    
    def on_epoch_begin(self, epoch, logs=None):
        """Epoch begin."""
        self.logger.debug(f"Node {self.node_id} - Epoch {epoch + 1} started")
    
    def on_epoch_end(self, epoch, logs=None):
        """Epoch end."""
        logs = logs or {}
        self.logger.info(
            f"Node {self.node_id} - Epoch {epoch + 1}: "
            f"Loss={logs.get('loss', 0):.4f}, "
            f"Acc={logs.get('accuracy', 0):.4f}"
        )
    
    def on_train_begin(self, logs=None):
        """Train begin."""
        self.logger.info(f"Node {self.node_id} - Starting local training")
    
    def on_train_end(self, logs=None):
        """Train end."""
        self.logger.info(f"Node {self.node_id} - Local training completed")


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Testing federated client...")

    # Create test client
    test_client = create_client(node_id=0, dataset_name='HAM10000')

    print(f"Client created with {len(test_client.get_parameters({}))} parameter arrays")
