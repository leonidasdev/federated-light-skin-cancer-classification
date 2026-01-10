"""
Federated server for skin cancer classification using Flower.

Responsibilities:
- Initialize global model
- Manage federated training rounds
- Aggregate client parameters (FedAvg/FedProx)
- Evaluate global model
- Save checkpoints
"""

import flwr as fl
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

from config.config import FEDERATED_CONFIG, MODEL_CONFIG, LOGGING_CONFIG
from models.cnn_model import create_cnn_model, compile_model
from utils.logging_utils import setup_logger


class FederatedServer:
    """
    Server to coordinate federated training.
    """
    
    def __init__(self, 
                 strategy_name: str = 'FedAvg',
                 num_rounds: int = None,
                 server_address: str = None):
        """
        Initialize the federated server.

        Args:
            strategy_name (str): Aggregation strategy ('FedAvg' or 'FedProx')
            num_rounds (int): Number of federated rounds
            server_address (str): Server address
        """
        self.strategy_name = strategy_name or FEDERATED_CONFIG['strategy']
        self.num_rounds = num_rounds or FEDERATED_CONFIG['num_rounds']
        self.server_address = server_address or FEDERATED_CONFIG['server_address']
        
        self.logger = setup_logger('FederatedServer')
        self.global_model = None
        self.strategy = None
        
        self.logger.info(f"Server initialized - Strategy: {self.strategy_name}, Rounds: {self.num_rounds}")
    
    def initialize_global_model(self):
        """
        Initialize the global model.

        Returns:
            keras.Model: Initialized global model
        """
        self.logger.info("Initializing global model...")
        
        # Crear modelo
        self.global_model = create_cnn_model()
        self.global_model = compile_model(self.global_model)
        
        self.logger.info("Global model initialized successfully")
        
        return self.global_model
    
    def get_initial_parameters(self):
        """
        Obtiene los parámetros iniciales del modelo global.
        
        Returns:
            List[np.ndarray]: Lista de arrays con los pesos del modelo
        """
        if self.global_model is None:
            self.initialize_global_model()
        
        return self.global_model.get_weights()
    
    def create_strategy(self):
        """
        Create the aggregation strategy (FedAvg or FedProx).

        Returns:
            fl.server.strategy.Strategy: Configured strategy
        """
        self.logger.info(f"Creating strategy: {self.strategy_name}")

        # Initial parameters
        initial_parameters = fl.common.ndarrays_to_parameters(self.get_initial_parameters())

        # Common configuration
        strategy_config = {
            'fraction_fit': FEDERATED_CONFIG['fraction_fit'],
            'fraction_evaluate': FEDERATED_CONFIG['fraction_evaluate'],
            'min_fit_clients': FEDERATED_CONFIG['min_fit_clients'],
            'min_evaluate_clients': FEDERATED_CONFIG['min_evaluate_clients'],
            'min_available_clients': FEDERATED_CONFIG['min_available_clients'],
            'initial_parameters': initial_parameters,
            'evaluate_fn': self.get_evaluate_fn(),
            'on_fit_config_fn': self.fit_config,
            'on_evaluate_config_fn': self.evaluate_config
        }
        
        # Create strategy by type
        if self.strategy_name == 'FedAvg':
            self.strategy = fl.server.strategy.FedAvg(**strategy_config)
        
        elif self.strategy_name == 'FedProx':
            # FedProx adds proximal regularization for non-IID
            strategy_config['proximal_mu'] = FEDERATED_CONFIG.get('fedprox_mu', 0.01)
            self.strategy = fl.server.strategy.FedProx(**strategy_config)
        
        else:
            raise ValueError(f"Strategy {self.strategy_name} not supported")

        self.logger.info("Strategy created successfully")
        return self.strategy
    
    def fit_config(self, server_round: int) -> Dict:
        """
        Configuration sent to clients for training.

        Args:
            server_round (int): Current round

        Returns:
            dict: Configuration for clients
        """
        config = {
            'server_round': server_round,
            'local_epochs': FEDERATED_CONFIG.get('local_epochs', 5),
            'batch_size': FEDERATED_CONFIG.get('batch_size', 32)
        }
        
        self.logger.info(f"Round {server_round}/{self.num_rounds} - Sending config to clients")
        
        return config
    
    def evaluate_config(self, server_round: int) -> Dict:
        """
        Configuration sent to clients for evaluation.

        Args:
            server_round (int): Current round

        Returns:
            dict: Configuration for evaluation
        """
        return {
            'server_round': server_round
        }
    
    def get_evaluate_fn(self):
        """
        Evaluation function for the global model on the server.

        Returns:
            callable: Evaluation function
        """
        def evaluate(server_round: int, 
                    parameters: fl.common.NDArrays, 
                    config: Dict) -> Optional[Tuple[float, Dict]]:
            """
            Evaluate the global model.

            Args:
                server_round (int): Current round
                parameters: Model parameters
                config: Configuration

            Returns:
                tuple: (loss, metrics_dict) or None
            """
            # TODO: Implement centralized evaluation with a validation dataset
            # For now return None to use only federated evaluation

            self.logger.info(f"Centralized evaluation at round {server_round}")

            return None
        
        return evaluate
    
    def aggregate_fit(self, 
                     server_round: int,
                     results: List[Tuple],
                     failures: List[BaseException]) -> Tuple:
        """
        Aggregate training results from clients.

        Args:
            server_round (int): Current round
            results (list): Successful client results
            failures (list): Client failures

        Returns:
            tuple: Aggregated parameters and metrics
        """
        # TODO: Implement custom aggregation logic if needed
        # By default, the strategy handles aggregation

        self.logger.info(f"Round {server_round}: Aggregating {len(results)} clients, {len(failures)} failures")

        return results
    
    def save_global_model(self, save_path: str, round_num: int):
        """
        Save the global model.

        Args:
            save_path (str): Path to save
            round_num (int): Round number
        """
        # TODO: Implement checkpoint saving
        checkpoint_dir = Path(LOGGING_CONFIG['model_checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = checkpoint_dir / f"global_model_round_{round_num}.h5"
        self.global_model.save(str(model_path))
        
        self.logger.info(f"Global model saved: {model_path}")
    
    def start(self):
        """
        Start the federated server.
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING FEDERATED SERVER")
        self.logger.info("=" * 60)
        self.logger.info(f"Address: {self.server_address}")
        self.logger.info(f"Strategy: {self.strategy_name}")
        self.logger.info(f"Rounds: {self.num_rounds}")
        self.logger.info("=" * 60)
        
        # Inicializar modelo y estrategia
        self.initialize_global_model()
        strategy = self.create_strategy()
        
        # Iniciar servidor Flower
        try:
            fl.server.start_server(
                server_address=self.server_address,
                config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=strategy
            )
            
            self.logger.info("Server finished successfully")
        
        except Exception as e:
            self.logger.error(f"Server error: {e}", exc_info=True)
            raise


# ==================== ESTRATEGIAS PERSONALIZADAS ====================

class CustomFedAvg(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy with additional logging.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = setup_logger('CustomFedAvg')
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate parameters with logging."""
        self.logger.info(f"Round {server_round}: Aggregating {len(results)} updates")
        
        # Llamar a implementación base
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log aggregated metrics
        if aggregated_metrics:
            self.logger.info(f"Aggregated metrics: {aggregated_metrics}")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluations with logging."""
        self.logger.info(f"Round {server_round}: Evaluating {len(results)} clients")
        
        # Llamar a implementación base
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Log results
        self.logger.info(f"Aggregated loss: {aggregated_loss:.4f}")
        if aggregated_metrics:
            self.logger.info(f"Metrics: {aggregated_metrics}")
        
        return aggregated_loss, aggregated_metrics


class SecureAggregationStrategy(fl.server.strategy.FedAvg):
    """
    Strategy with secure aggregation (placeholder for future implementation).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = setup_logger('SecureAggregation')
        self.logger.warning("Secure Aggregation not fully implemented - using standard FedAvg")
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate parameters with secure aggregation.

        TODO: Implement real secure aggregation
        """
        # For now, use standard aggregation
        return super().aggregate_fit(server_round, results, failures)


# ==================== FUNCIONES DE UTILIDAD ====================

def create_server(strategy_name: str = None, **kwargs) -> FederatedServer:
    """
    Factory to create a federated server.

    Args:
        strategy_name (str): Name of the strategy
        **kwargs: Additional arguments for the server

    Returns:
        FederatedServer: Configured server
    """
    return FederatedServer(strategy_name=strategy_name, **kwargs)


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Initializing test server...")

    server = create_server()
    server.initialize_global_model()

    print("Global model created successfully")
    print(f"Initial parameters: {len(server.get_initial_parameters())} arrays")
