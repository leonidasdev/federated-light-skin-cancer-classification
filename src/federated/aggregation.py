"""
Aggregation Functions
=====================

Model aggregation strategies for federated learning.
"""

import torch
from typing import Dict, List, Optional
from copy import deepcopy


def federated_averaging(
    client_updates: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Federated Averaging (FedAvg) aggregation.
    
    Averages model parameters across clients, optionally weighted by
    the number of samples each client has.
    
    Args:
        client_updates: List of client model state dicts
        weights: Optional weights (e.g., number of samples per client)
        
    Returns:
        Aggregated model state dict
    """
    if not client_updates:
        raise ValueError("No client updates provided")
    
    # Default to equal weights
    if weights is None:
        weights = [1.0] * len(client_updates)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Initialize aggregated state
    aggregated = deepcopy(client_updates[0])
    
    for key in aggregated.keys():
        aggregated[key] = torch.zeros_like(aggregated[key], dtype=torch.float32)
        
        for client_state, weight in zip(client_updates, weights):
            aggregated[key] += weight * client_state[key].float()
        
        # Restore original dtype
        aggregated[key] = aggregated[key].to(client_updates[0][key].dtype)
    
    return aggregated


def weighted_averaging(
    client_updates: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """
    Weighted averaging with explicit weights.
    
    Args:
        client_updates: List of client model state dicts
        weights: Weights for each client (must sum to 1 or will be normalized)
        
    Returns:
        Aggregated model state dict
    """
    return federated_averaging(client_updates, weights)


def fedprox_aggregation(
    client_updates: List[Dict[str, torch.Tensor]],
    global_model: Dict[str, torch.Tensor],
    mu: float = 0.01,
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    FedProx aggregation with proximal term.
    
    Note: The proximal term is typically applied during client-side training,
    not during aggregation. This function performs weighted averaging
    similar to FedAvg.
    
    Args:
        client_updates: List of client model state dicts
        global_model: Previous global model state dict
        mu: Proximal term coefficient (used in client training)
        weights: Optional weights for averaging
        
    Returns:
        Aggregated model state dict
    """
    # FedProx aggregation is the same as FedAvg
    # The proximal term is applied during local training
    return federated_averaging(client_updates, weights)


def median_aggregation(
    client_updates: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise median aggregation (Byzantine-robust).
    
    Args:
        client_updates: List of client model state dicts
        
    Returns:
        Aggregated model state dict
    """
    if not client_updates:
        raise ValueError("No client updates provided")
    
    aggregated = deepcopy(client_updates[0])
    
    for key in aggregated.keys():
        stacked = torch.stack([u[key].float() for u in client_updates])
        aggregated[key] = torch.median(stacked, dim=0).values
        aggregated[key] = aggregated[key].to(client_updates[0][key].dtype)
    
    return aggregated


def trimmed_mean_aggregation(
    client_updates: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Trimmed mean aggregation (Byzantine-robust).
    
    Removes the top and bottom trim_ratio of values before averaging.
    
    Args:
        client_updates: List of client model state dicts
        trim_ratio: Fraction of values to trim from each end
        
    Returns:
        Aggregated model state dict
    """
    if not client_updates:
        raise ValueError("No client updates provided")
    
    num_clients = len(client_updates)
    trim_count = int(num_clients * trim_ratio)
    
    if 2 * trim_count >= num_clients:
        raise ValueError("trim_ratio too large for number of clients")
    
    aggregated = deepcopy(client_updates[0])
    
    for key in aggregated.keys():
        stacked = torch.stack([u[key].float() for u in client_updates])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        
        # Trim and average
        trimmed = sorted_vals[trim_count:num_clients - trim_count]
        aggregated[key] = torch.mean(trimmed, dim=0)
        aggregated[key] = aggregated[key].to(client_updates[0][key].dtype)
    
    return aggregated
