"""
Servidor federado para clasificación de cáncer de piel usando Flower.

Responsabilidades:
- Inicializar modelo global
- Gestionar rondas de entrenamiento federado
- Agregar parámetros de clientes (FedAvg/FedProx)
- Evaluar modelo global
- Guardar checkpoints
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
    Servidor para coordinar el entrenamiento federado.
    """
    
    def __init__(self, 
                 strategy_name: str = 'FedAvg',
                 num_rounds: int = None,
                 server_address: str = None):
        """
        Inicializa el servidor federado.
        
        Args:
            strategy_name (str): Estrategia de agregación ('FedAvg' o 'FedProx')
            num_rounds (int): Número de rondas federadas
            server_address (str): Dirección del servidor
        """
        self.strategy_name = strategy_name or FEDERATED_CONFIG['strategy']
        self.num_rounds = num_rounds or FEDERATED_CONFIG['num_rounds']
        self.server_address = server_address or FEDERATED_CONFIG['server_address']
        
        self.logger = setup_logger('FederatedServer')
        self.global_model = None
        self.strategy = None
        
        self.logger.info(f"Servidor inicializado - Estrategia: {self.strategy_name}, Rondas: {self.num_rounds}")
    
    def initialize_global_model(self):
        """
        Inicializa el modelo global.
        
        Returns:
            keras.Model: Modelo global inicializado
        """
        self.logger.info("Inicializando modelo global...")
        
        # Crear modelo
        self.global_model = create_cnn_model()
        self.global_model = compile_model(self.global_model)
        
        self.logger.info("Modelo global inicializado correctamente")
        
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
        Crea la estrategia de agregación (FedAvg o FedProx).
        
        Returns:
            fl.server.strategy.Strategy: Estrategia configurada
        """
        self.logger.info(f"Creando estrategia: {self.strategy_name}")
        
        # Parámetros iniciales
        initial_parameters = fl.common.ndarrays_to_parameters(self.get_initial_parameters())
        
        # Configuración común
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
        
        # Crear estrategia según tipo
        if self.strategy_name == 'FedAvg':
            self.strategy = fl.server.strategy.FedAvg(**strategy_config)
        
        elif self.strategy_name == 'FedProx':
            # FedProx añade regularización proximal para no-IID
            strategy_config['proximal_mu'] = FEDERATED_CONFIG.get('fedprox_mu', 0.01)
            self.strategy = fl.server.strategy.FedProx(**strategy_config)
        
        else:
            raise ValueError(f"Estrategia {self.strategy_name} no soportada")
        
        self.logger.info("Estrategia creada correctamente")
        return self.strategy
    
    def fit_config(self, server_round: int) -> Dict:
        """
        Configuración enviada a clientes para entrenamiento.
        
        Args:
            server_round (int): Ronda actual
        
        Returns:
            dict: Configuración para clientes
        """
        config = {
            'server_round': server_round,
            'local_epochs': FEDERATED_CONFIG.get('local_epochs', 5),
            'batch_size': FEDERATED_CONFIG.get('batch_size', 32)
        }
        
        self.logger.info(f"Ronda {server_round}/{self.num_rounds} - Enviando configuración a clientes")
        
        return config
    
    def evaluate_config(self, server_round: int) -> Dict:
        """
        Configuración enviada a clientes para evaluación.
        
        Args:
            server_round (int): Ronda actual
        
        Returns:
            dict: Configuración para evaluación
        """
        return {
            'server_round': server_round
        }
    
    def get_evaluate_fn(self):
        """
        Función de evaluación del modelo global en el servidor.
        
        Returns:
            callable: Función de evaluación
        """
        def evaluate(server_round: int, 
                    parameters: fl.common.NDArrays, 
                    config: Dict) -> Optional[Tuple[float, Dict]]:
            """
            Evalúa el modelo global.
            
            Args:
                server_round (int): Ronda actual
                parameters: Parámetros del modelo
                config: Configuración
            
            Returns:
                tuple: (loss, metrics_dict) o None
            """
            # TODO: Implementar evaluación centralizada con dataset de validación
            # Por ahora retornamos None para usar solo evaluación federada
            
            self.logger.info(f"Evaluación centralizada en ronda {server_round}")
            
            return None
        
        return evaluate
    
    def aggregate_fit(self, 
                     server_round: int,
                     results: List[Tuple],
                     failures: List[BaseException]) -> Tuple:
        """
        Agrega resultados del entrenamiento de clientes.
        
        Args:
            server_round (int): Ronda actual
            results (list): Resultados de clientes exitosos
            failures (list): Fallos de clientes
        
        Returns:
            tuple: Parámetros agregados y métricas
        """
        # TODO: Implementar lógica de agregación personalizada si es necesario
        # Por defecto, la estrategia maneja esto
        
        self.logger.info(f"Ronda {server_round}: Agregando {len(results)} clientes, {len(failures)} fallos")
        
        return results
    
    def save_global_model(self, save_path: str, round_num: int):
        """
        Guarda el modelo global.
        
        Args:
            save_path (str): Ruta donde guardar
            round_num (int): Número de ronda
        """
        # TODO: Implementar guardado de checkpoints
        checkpoint_dir = Path(LOGGING_CONFIG['model_checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = checkpoint_dir / f"global_model_round_{round_num}.h5"
        self.global_model.save(str(model_path))
        
        self.logger.info(f"Modelo global guardado: {model_path}")
    
    def start(self):
        """
        Inicia el servidor federado.
        """
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO SERVIDOR FEDERADO")
        self.logger.info("=" * 60)
        self.logger.info(f"Dirección: {self.server_address}")
        self.logger.info(f"Estrategia: {self.strategy_name}")
        self.logger.info(f"Rondas: {self.num_rounds}")
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
            
            self.logger.info("Servidor finalizado correctamente")
        
        except Exception as e:
            self.logger.error(f"Error en servidor: {e}", exc_info=True)
            raise


# ==================== ESTRATEGIAS PERSONALIZADAS ====================

class CustomFedAvg(fl.server.strategy.FedAvg):
    """
    Estrategia FedAvg personalizada con logging adicional.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = setup_logger('CustomFedAvg')
    
    def aggregate_fit(self, server_round, results, failures):
        """Agrega parámetros con logging."""
        self.logger.info(f"Ronda {server_round}: Agregando {len(results)} actualizaciones")
        
        # Llamar a implementación base
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log de métricas agregadas
        if aggregated_metrics:
            self.logger.info(f"Métricas agregadas: {aggregated_metrics}")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Agrega evaluaciones con logging."""
        self.logger.info(f"Ronda {server_round}: Evaluando {len(results)} clientes")
        
        # Llamar a implementación base
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Log de resultados
        self.logger.info(f"Loss agregado: {aggregated_loss:.4f}")
        if aggregated_metrics:
            self.logger.info(f"Métricas: {aggregated_metrics}")
        
        return aggregated_loss, aggregated_metrics


class SecureAggregationStrategy(fl.server.strategy.FedAvg):
    """
    Estrategia con agregación segura (placeholder para implementación futura).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = setup_logger('SecureAggregation')
        self.logger.warning("Secure Aggregation no implementada completamente - usando FedAvg estándar")
    
    def aggregate_fit(self, server_round, results, failures):
        """
        Agrega parámetros con agregación segura.
        
        TODO: Implementar secure aggregation real
        """
        # Por ahora, usar agregación estándar
        return super().aggregate_fit(server_round, results, failures)


# ==================== FUNCIONES DE UTILIDAD ====================

def create_server(strategy_name: str = None, **kwargs) -> FederatedServer:
    """
    Factory para crear servidor federado.
    
    Args:
        strategy_name (str): Nombre de la estrategia
        **kwargs: Argumentos adicionales para el servidor
    
    Returns:
        FederatedServer: Servidor configurado
    """
    return FederatedServer(strategy_name=strategy_name, **kwargs)


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Inicializando servidor de prueba...")
    
    server = create_server()
    server.initialize_global_model()
    
    print("Modelo global creado correctamente")
    print(f"Parámetros iniciales: {len(server.get_initial_parameters())} arrays")
