"""
Federated Learning Strategy for Skin Cancer Classification.

Implements FedAvg and custom aggregation strategies for DSCATNet.
"""

from flwr.server.strategy import FedAvg
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Tuple, Optional, Callable, Union
import numpy as np
from pathlib import Path


class DSCATNetFedAvg(FedAvg):
    """
    Custom FedAvg strategy for DSCATNet with additional features:
    
    - Checkpoint saving
    - Detailed metrics logging
    - Client-specific weighting
    - Early stopping support
    
    Args:
        save_path: Directory to save model checkpoints
        save_every: Save checkpoint every N rounds
        early_stopping_patience: Rounds without improvement before stopping
        min_delta: Minimum improvement to reset patience
        **kwargs: Arguments passed to FedAvg
    """
    
    def __init__(
        self,
        save_path: Optional[str] = None,
        save_every: int = 10,
        early_stopping_patience: int = 20,
        min_delta: float = 0.001,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.save_path = Path(save_path) if save_path else None
        self.save_every = save_every
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        
        # Training tracking
        self.current_round = 0
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.should_stop = False
        
        # History for analysis
        self.metrics_history: Dict[str, List] = {
            'round': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'num_clients': [],
            'client_metrics': []
        }
        
        # Create save directory
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager
    ):
        """Configure fit with current round information."""
        self.current_round = server_round
        
        # Get base configuration
        config = super().configure_fit(server_round, parameters, client_manager)
        
        # Add round info to each client's config
        if config:
            updated_config = []
            for client, fit_ins in config:
                fit_ins.config["current_round"] = server_round
                fit_ins.config["total_rounds"] = 100  # Update based on actual
                updated_config.append((client, fit_ins))
            return updated_config
        
        return config
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results with custom logging.
        """
        if not results:
            return None, {}
        
        # Aggregate using parent class
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Collect client metrics
        client_metrics = []
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_samples = 0
        
        for client_proxy, fit_res in results:
            num_samples = fit_res.num_examples
            client_metric = fit_res.metrics
            
            client_metrics.append({
                'client_id': client_metric.get('client_id', 'unknown'),
                'train_loss': float(client_metric.get('train_loss', 0) or 0.0),
                'train_accuracy': float(client_metric.get('train_accuracy', 0) or 0.0),
                'num_samples': num_samples
            })
            # Safely coerce client-provided metrics to floats to avoid type issues
            try:
                client_loss_val = client_metric.get('train_loss', 0)
                client_loss = float(client_loss_val) if client_loss_val is not None else 0.0
            except (TypeError, ValueError):
                client_loss = 0.0

            try:
                client_acc_val = client_metric.get('train_accuracy', 0)
                client_acc = float(client_acc_val) if client_acc_val is not None else 0.0
            except (TypeError, ValueError):
                client_acc = 0.0

            total_train_loss += client_loss * num_samples
            total_train_acc += client_acc * num_samples
            total_samples += num_samples
        
        # Compute weighted averages
        avg_train_loss = total_train_loss / total_samples if total_samples > 0 else 0
        avg_train_acc = total_train_acc / total_samples if total_samples > 0 else 0
        
        # Update history
        self.metrics_history['round'].append(server_round)
        self.metrics_history['train_loss'].append(avg_train_loss)
        self.metrics_history['train_accuracy'].append(avg_train_acc)
        self.metrics_history['num_clients'].append(len(results))
        self.metrics_history['client_metrics'].append(client_metrics)
        
        # Add to aggregated metrics
        metrics['avg_train_loss'] = avg_train_loss
        metrics['avg_train_accuracy'] = avg_train_acc
        metrics['num_clients_trained'] = len(results)
        metrics['num_failures'] = len(failures)
        
        # Save checkpoint periodically
        if self.save_path and server_round % self.save_every == 0:
            self._save_checkpoint(aggregated_params, server_round)
        
        print(f"\n[Round {server_round}] Training - "
              f"Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.2f}%")
        
        return aggregated_params, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results with early stopping check.
        """
        if not results:
            return None, {}
        
        # Aggregate using parent class
        aggregated_loss, metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Compute weighted accuracy
        total_acc = 0.0
        total_samples = 0
        
        for client_proxy, eval_res in results:
            num_samples = eval_res.num_examples
            # Safely coerce reported accuracy to float to avoid mixing types
            try:
                acc_val = eval_res.metrics.get('accuracy', 0)
                acc = float(acc_val) if acc_val is not None else 0.0
            except (TypeError, ValueError):
                acc = 0.0

            total_acc += acc * num_samples
            total_samples += num_samples
        
        avg_accuracy = total_acc / total_samples if total_samples > 0 else 0
        
        # Update history
        self.metrics_history['val_loss'].append(aggregated_loss)
        self.metrics_history['val_accuracy'].append(avg_accuracy)
        
        # Early stopping check
        if avg_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = avg_accuracy
            self.patience_counter = 0
            
            # Save best model
            if self.save_path:
                # Some Flower versions store current parameters on the strategy;
                # use getattr to avoid static-analysis errors if the attribute
                # isn't present in the environment.
                params_to_save = getattr(self, "parameters", None)
                self._save_checkpoint(
                    params_to_save,
                    server_round,
                    suffix='best'
                )
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                self.should_stop = True
                print(f"\nEarly stopping triggered at round {server_round}")
        
        metrics['avg_val_accuracy'] = avg_accuracy
        metrics['best_accuracy'] = self.best_accuracy
        metrics['patience_counter'] = self.patience_counter
        
        print(f"[Round {server_round}] Evaluation - "
              f"Loss: {aggregated_loss:.4f}, Accuracy: {avg_accuracy:.2f}% "
              f"(Best: {self.best_accuracy:.2f}%)")
        
        return aggregated_loss, metrics
    
    def _save_checkpoint(
        self,
        parameters: Optional[Parameters],
        round_num: int,
        suffix: str = ''
    ) -> None:
        """Save model checkpoint."""
        if parameters is None or self.save_path is None:
            return
        
        # Convert parameters to numpy arrays
        ndarrays = parameters_to_ndarrays(parameters)
        
        # Create filename
        filename = f"model_round_{round_num}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".npz"
        
        # Save
        filepath = self.save_path / filename
        np.savez(filepath, *ndarrays)
        
        # Also save metrics history
        metrics_path = self.save_path / f"metrics_round_{round_num}.npz"
        np.savez(
            metrics_path,
            rounds=self.metrics_history['round'],
            train_loss=self.metrics_history['train_loss'],
            train_accuracy=self.metrics_history['train_accuracy'],
            val_loss=self.metrics_history['val_loss'],
            val_accuracy=self.metrics_history['val_accuracy']
        )
    
    def get_history(self) -> Dict[str, List]:
        """Return training history."""
        return self.metrics_history


def create_fedavg_strategy(
    initial_parameters: List[np.ndarray],
    min_fit_clients: int = 4,
    min_evaluate_clients: int = 4,
    min_available_clients: int = 4,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    save_path: Optional[str] = None,
    evaluate_metrics_aggregation_fn: Optional[Callable] = None
) -> DSCATNetFedAvg:
    """
    Factory function to create FedAvg strategy for DSCATNet.
    
    Args:
        initial_parameters: Initial model parameters as numpy arrays
        min_fit_clients: Minimum clients for training
        min_evaluate_clients: Minimum clients for evaluation
        min_available_clients: Minimum available clients
        fraction_fit: Fraction of clients for training
        fraction_evaluate: Fraction of clients for evaluation
        save_path: Path to save checkpoints
        evaluate_metrics_aggregation_fn: Custom metrics aggregation function
        
    Returns:
        Configured DSCATNetFedAvg strategy
    """
    # Convert to Flower Parameters
    initial_params = ndarrays_to_parameters(initial_parameters)
    
    # Default metrics aggregation if not provided
    if evaluate_metrics_aggregation_fn is None:
        def _default_evaluate_metrics_aggregation_fn(metrics):
            if not metrics:
                return {}

            total_samples = sum(n for n, _ in metrics)
            aggregated = {}

            for key in metrics[0][1].keys():
                if key == 'client_id':
                    continue
                values = [m[key] for _, m in metrics if key in m]
                weights = [n for n, m in metrics if key in m]

                if values and all(isinstance(v, (int, float)) for v in values):
                    aggregated[key] = sum(
                        v * w for v, w in zip(values, weights)
                    ) / sum(weights)

            return aggregated

        evaluate_metrics_aggregation_fn = _default_evaluate_metrics_aggregation_fn
    
    strategy = DSCATNetFedAvg(
        initial_parameters=initial_params,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        save_path=save_path
    )
    
    return strategy
