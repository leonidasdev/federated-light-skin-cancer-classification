"""
Flower FL Server for Skin Cancer Classification.

The server coordinates the federated learning process across all clients,
aggregating model updates and managing the training rounds.
"""

import flwr as fl
from flwr.server import ServerConfig
from flwr.server import start_server as fl_start_server
from flwr.server.strategy import Strategy
from flwr.server.history import History
from flwr.common import Parameters, Scalar
from typing import Dict, Optional, List, Tuple, Callable
import torch
import numpy as np
from pathlib import Path

from ..models.dscatnet import DSCATNet, get_model_parameters
from .strategy import create_fedavg_strategy


def create_server(
    model: DSCATNet,
    num_rounds: int = 50,
    min_fit_clients: int = 4,
    min_evaluate_clients: int = 4,
    min_available_clients: int = 4,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    strategy: Optional[Strategy] = None,
    save_path: Optional[str] = None
) -> Tuple[ServerConfig, Strategy]:
    """
    Create and configure the Flower FL server.
    
    Args:
        model: Initial DSCATNet model for parameter initialization
        num_rounds: Number of federated learning rounds
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum available clients to start
        fraction_fit: Fraction of clients for training (1.0 = all)
        fraction_evaluate: Fraction of clients for evaluation
        strategy: Optional custom strategy (defaults to FedAvg)
        save_path: Path to save checkpoints
        
    Returns:
        Tuple of (ServerConfig, Strategy)
    """
    # Get initial model parameters
    initial_parameters = get_model_parameters(model)
    
    # Create strategy if not provided
    if strategy is None:
        strategy = create_fedavg_strategy(
            initial_parameters=initial_parameters,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            save_path=save_path
        )
    
    # Server configuration
    config = ServerConfig(num_rounds=num_rounds)
    
    return config, strategy


def start_server(
    server_address: str = "[::]:8080",
    model: Optional[DSCATNet] = None,
    num_rounds: int = 50,
    strategy: Optional[Strategy] = None,
    **kwargs
) -> History:
    """
    Start the Flower FL server.
    
    Args:
        server_address: Address to run the server on
        model: DSCATNet model for initialization
        num_rounds: Number of FL rounds
        strategy: Custom strategy (optional)
        **kwargs: Additional arguments for create_server
        
    Returns:
        Flower History object with training results
    """
    if model is None:
        # Create default model for parameter initialization
        model = DSCATNet(num_classes=7)
    
    config, strategy = create_server(
        model=model,
        num_rounds=num_rounds,
        strategy=strategy,
        **kwargs
    )
    
    # Start server (uses Flower's start_server; deprecated in newer Flower)
    history = fl_start_server(
        server_address=server_address,
        config=config,
        strategy=strategy
    )
    
    return history


class FederatedServer:
    """
    High-level wrapper for managing FL server with DSCATNet.
    
    Provides convenient methods for:
    - Starting/stopping server
    - Saving/loading checkpoints
    - Tracking metrics
    - Visualization
    
    Args:
        model: DSCATNet model
        num_rounds: Number of FL rounds
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for TensorBoard logs
    """
    
    def __init__(
        self,
        model: DSCATNet,
        num_rounds: int = 50,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs"
    ):
        self.model = model
        self.num_rounds = num_rounds
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_round = 0
        self.history: Dict[str, List] = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'clients_trained': [],
            'clients_evaluated': []
        }
        
    def configure(
        self,
        min_clients: int = 4,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0
    ) -> Strategy:
        """Configure and return the FL strategy."""
        initial_params = get_model_parameters(self.model)
        
        strategy = create_fedavg_strategy(
            initial_parameters=initial_params,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            save_path=str(self.checkpoint_dir),
            evaluate_metrics_aggregation_fn=self._aggregate_metrics
        )
        
        return strategy
    
    def _aggregate_metrics(
        self,
        metrics: List[Tuple[int, Dict[str, Scalar]]]
    ) -> Dict[str, Scalar]:
        """Aggregate evaluation metrics from all clients."""
        if not metrics:
            return {}
        
        # Weighted average by number of samples
        total_samples = sum(num for num, _ in metrics)
        
        aggregated = {}
        metric_keys = metrics[0][1].keys()
        
        for key in metric_keys:
            if key == 'client_id':
                continue
            values = [m[key] for _, m in metrics if key in m]
            weights = [n for n, m in metrics if key in m]
            
            if values and all(isinstance(v, (int, float)) for v in values):
                # Ensure numeric types and explicit float conversion to satisfy type checkers
                numerator = sum([float(v) * float(w) for v, w in zip(values, weights)])
                total_weight = float(sum(weights)) if sum(weights) != 0 else 0.0
                aggregated[key] = (numerator / total_weight) if total_weight != 0.0 else 0.0
        
        return aggregated
    
    def save_checkpoint(self, round_num: int) -> str:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"model_round_{round_num}.pt"
        torch.save({
            'round': round_num,
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, checkpoint_path)
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return the round number."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['round']
    
    def get_summary(self) -> Dict:
        """Get summary of FL training."""
        return {
            'total_rounds': len(self.history['round']),
            'final_accuracy': self.history['accuracy'][-1] if self.history['accuracy'] else None,
            'best_accuracy': max(self.history['accuracy']) if self.history['accuracy'] else None,
            'final_loss': self.history['loss'][-1] if self.history['loss'] else None
        }
