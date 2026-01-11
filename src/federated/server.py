"""
Federated Learning Server
=========================

Central server for coordinating federated learning across clients.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
from copy import deepcopy

from ..utils.logging import get_logger
from .aggregation import federated_averaging


class FederatedServer:
    """
    Federated Learning Server.
    
    Coordinates training across multiple clients, handles model aggregation,
    and manages the global model state.
    
    Args:
        model: Global model architecture
        num_clients: Total number of clients
        fraction_fit: Fraction of clients to sample each round
        min_fit_clients: Minimum number of clients for training
        aggregation_fn: Function to aggregate client updates
        device: Device to use for aggregation
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_clients: int,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        aggregation_fn: Optional[Callable] = None,
        device: str = "cpu",
    ):
        self.global_model = model.to(device)
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.aggregation_fn = aggregation_fn or federated_averaging
        self.device = device
        
        self.logger = get_logger("FederatedServer")
        self.current_round = 0
        self.history: Dict[str, List[float]] = {
            'global_loss': [],
            'global_accuracy': [],
        }
    
    def get_global_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get current global model parameters.
        
        Returns:
            Dictionary of model state dict
        """
        return deepcopy(self.global_model.state_dict())
    
    def select_clients(self, available_clients: List[int]) -> List[int]:
        """
        Select clients for the current round.
        
        Args:
            available_clients: List of available client IDs
            
        Returns:
            List of selected client IDs
        """
        num_clients = max(
            int(len(available_clients) * self.fraction_fit),
            self.min_fit_clients
        )
        num_clients = min(num_clients, len(available_clients))
        
        indices = torch.randperm(len(available_clients))[:num_clients]
        return [available_clients[i] for i in indices]
    
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> None:
        """
        Aggregate client model updates into global model.
        
        Args:
            client_updates: List of client model state dicts
            client_weights: Optional weights for weighted averaging
        """
        if not client_updates:
            self.logger.warning("No client updates received for aggregation")
            return
        
        aggregated_state = self.aggregation_fn(
            client_updates,
            weights=client_weights,
        )
        
        self.global_model.load_state_dict(aggregated_state)
        self.logger.info(f"Round {self.current_round}: Aggregated {len(client_updates)} client updates")
    
    def evaluate(
        self,
        test_loader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Evaluate global model on test data.
        
        Args:
            test_loader: Test data loader
            criterion: Loss function
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        metrics = {
            'loss': total_loss / len(test_loader),
            'accuracy': correct / total,
        }
        
        self.history['global_loss'].append(metrics['loss'])
        self.history['global_accuracy'].append(metrics['accuracy'])
        
        return metrics
    
    def run_round(
        self,
        clients: List["FederatedClient"],
        epochs_per_round: int = 1,
    ) -> Dict[str, float]:
        """
        Execute one federated learning round.
        
        Args:
            clients: List of client instances
            epochs_per_round: Local epochs per client
            
        Returns:
            Dictionary of round metrics
        """
        self.current_round += 1
        self.logger.info(f"Starting round {self.current_round}")
        
        # Select clients
        selected_indices = self.select_clients(list(range(len(clients))))
        selected_clients = [clients[i] for i in selected_indices]
        
        # Distribute global model
        global_params = self.get_global_model_parameters()
        
        # Client training
        client_updates = []
        client_weights = []
        
        for client in selected_clients:
            # Set client model to global parameters
            client.set_parameters(global_params)
            
            # Local training
            metrics, num_samples = client.train(epochs=epochs_per_round)
            
            # Collect updates
            client_updates.append(client.get_parameters())
            client_weights.append(num_samples)
        
        # Aggregate updates
        self.aggregate(client_updates, client_weights)
        
        return {'round': self.current_round, 'num_clients': len(selected_clients)}
