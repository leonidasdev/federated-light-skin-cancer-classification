"""
Flower Server Implementation
============================

Flower (flwr) server for federated learning with LMS-ViT.
Provides server configuration, strategy setup, and execution functions.

This module provides:
- run_flower_server: Start a Flower server for distributed training
- run_flower_simulation: Run a Flower simulation locally
- Custom strategy configurations
"""

from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar, FitRes, Parameters, ndarrays_to_parameters
from flwr.server.strategy import FedAvg, FedProx, FedAdagrad, FedAdam
from flwr.server.client_proxy import ClientProxy

from ..models import LMSViT, lmsvit_tiny, lmsvit_small, lmsvit_base
from ..utils.logging import get_logger, setup_logging


logger = get_logger("FlowerServer")


def get_initial_parameters(
    model_name: str = "small",
    num_classes: int = 7,
) -> Parameters:
    """
    Get initial model parameters for server-side initialization.
    
    Args:
        model_name: LMS-ViT variant ('tiny', 'small', 'base')
        num_classes: Number of output classes
        
    Returns:
        Flower Parameters object with initial weights
    """
    model_factories = {
        "tiny": lmsvit_tiny,
        "small": lmsvit_small,
        "base": lmsvit_base,
    }
    
    model = model_factories[model_name](num_classes=num_classes)
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    return ndarrays_to_parameters(ndarrays)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average.
    
    Args:
        metrics: List of (num_samples, metrics_dict) tuples from clients
        
    Returns:
        Aggregated metrics dictionary
    """
    if not metrics:
        return {}
    
    # Extract metric names from first client (excluding non-numeric)
    metric_names = [
        k for k, v in metrics[0][1].items() 
        if isinstance(v, (int, float)) and k not in ["client_id"]
    ]
    
    total_samples = sum(num_samples for num_samples, _ in metrics)
    
    aggregated = {}
    for name in metric_names:
        weighted_sum = sum(
            num_samples * m.get(name, 0) 
            for num_samples, m in metrics
        )
        aggregated[name] = weighted_sum / total_samples if total_samples > 0 else 0.0
    
    aggregated["total_samples"] = total_samples
    aggregated["num_clients"] = len(metrics)
    
    return aggregated


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training metrics from fit() calls."""
    aggregated = weighted_average(metrics)
    if aggregated:
        logger.info(
            f"Training aggregated - "
            f"loss: {aggregated.get('train_loss', 0):.4f}, "
            f"clients: {aggregated.get('num_clients', 0)}, "
            f"samples: {aggregated.get('total_samples', 0)}"
        )
    return aggregated


def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics from evaluate() calls."""
    aggregated = weighted_average(metrics)
    if aggregated:
        logger.info(
            f"Evaluation aggregated - "
            f"accuracy: {aggregated.get('accuracy', 0):.4f}, "
            f"f1: {aggregated.get('f1', 0):.4f}, "
            f"clients: {aggregated.get('num_clients', 0)}"
        )
    return aggregated


def create_strategy(
    strategy_name: str = "fedavg",
    num_clients: int = 4,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    initial_parameters: Optional[Parameters] = None,
    **strategy_kwargs,
) -> fl.server.strategy.Strategy:
    """
    Create a Flower aggregation strategy.
    
    Args:
        strategy_name: Strategy type ('fedavg', 'fedprox', 'fedadam', 'fedadagrad')
        num_clients: Total number of clients
        fraction_fit: Fraction of clients for training
        fraction_evaluate: Fraction of clients for evaluation
        min_fit_clients: Minimum clients for training
        min_evaluate_clients: Minimum clients for evaluation
        min_available_clients: Minimum available clients to start
        initial_parameters: Initial model parameters
        **strategy_kwargs: Additional strategy-specific arguments
        
    Returns:
        Configured Flower Strategy
    """
    common_args = {
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
        "initial_parameters": initial_parameters,
        "fit_metrics_aggregation_fn": fit_metrics_aggregation_fn,
        "evaluate_metrics_aggregation_fn": evaluate_metrics_aggregation_fn,
    }
    
    strategies = {
        "fedavg": FedAvg,
        "fedprox": FedProx,
        "fedadam": FedAdam,
        "fedadagrad": FedAdagrad,
    }
    
    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Choose from: {list(strategies.keys())}"
        )
    
    strategy_class = strategies[strategy_name]
    
    # Add strategy-specific arguments
    if strategy_name == "fedprox":
        common_args["proximal_mu"] = strategy_kwargs.get("proximal_mu", 0.01)
    elif strategy_name in ["fedadam", "fedadagrad"]:
        common_args["eta"] = strategy_kwargs.get("eta", 1e-1)
        common_args["eta_l"] = strategy_kwargs.get("eta_l", 1e-1)
        common_args["tau"] = strategy_kwargs.get("tau", 1e-9)
    
    logger.info(f"Creating {strategy_name.upper()} strategy with {num_clients} clients")
    
    return strategy_class(**common_args)


def run_flower_server(
    server_address: str = "[::]:8080",
    num_rounds: int = 10,
    strategy: Optional[fl.server.strategy.Strategy] = None,
    model_name: str = "small",
    num_classes: int = 7,
) -> fl.server.History:
    """
    Start a Flower server for distributed federated learning.
    
    This function starts a gRPC server that clients can connect to
    for federated training.
    
    Args:
        server_address: Server address (default: "[::]:8080")
        num_rounds: Number of federated rounds
        strategy: Flower strategy (default: FedAvg)
        model_name: LMS-ViT variant for initial parameters
        num_classes: Number of output classes
        
    Returns:
        Flower History object with training history
    """
    if strategy is None:
        initial_params = get_initial_parameters(model_name, num_classes)
        strategy = create_strategy(
            strategy_name="fedavg",
            initial_parameters=initial_params,
        )
    
    logger.info(f"Starting Flower server at {server_address}")
    logger.info(f"Running {num_rounds} rounds")
    
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    return history


def run_flower_simulation(
    client_fn: Callable[[str], fl.client.Client],
    num_clients: int = 4,
    num_rounds: int = 10,
    strategy: Optional[fl.server.strategy.Strategy] = None,
    model_name: str = "small",
    num_classes: int = 7,
    client_resources: Optional[Dict[str, float]] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
) -> fl.server.History:
    """
    Run a Flower simulation locally.
    
    This function simulates federated learning without requiring
    actual distributed clients. Useful for development and testing.
    
    Args:
        client_fn: Function that creates a client given a client ID
        num_clients: Number of simulated clients
        num_rounds: Number of federated rounds
        strategy: Flower strategy (default: FedAvg)
        model_name: LMS-ViT variant for initial parameters
        num_classes: Number of output classes
        client_resources: Resources per client (e.g., {"num_cpus": 1, "num_gpus": 0.25})
        ray_init_args: Arguments for Ray initialization
        
    Returns:
        Flower History object with training history
    """
    if strategy is None:
        initial_params = get_initial_parameters(model_name, num_classes)
        strategy = create_strategy(
            strategy_name="fedavg",
            num_clients=num_clients,
            min_fit_clients=min(2, num_clients),
            min_evaluate_clients=min(2, num_clients),
            min_available_clients=min(2, num_clients),
            initial_parameters=initial_params,
        )
    
    # Default client resources
    if client_resources is None:
        # Allocate modest resources per client
        client_resources = {
            "num_cpus": 1,
            "num_gpus": 0.0,  # Set > 0 if GPU available
        }
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            # Share GPU among clients
            client_resources["num_gpus"] = 1.0 / num_clients
    
    logger.info(f"Starting Flower simulation with {num_clients} clients")
    logger.info(f"Running {num_rounds} rounds")
    logger.info(f"Client resources: {client_resources}")
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args or {"include_dashboard": False},
    )
    
    return history


def print_history(history: fl.server.History) -> None:
    """
    Print training history in a formatted way.
    
    Args:
        history: Flower History object
    """
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING RESULTS")
    print("=" * 60)
    
    # Print distributed losses
    if history.losses_distributed:
        print("\nDistributed Losses per Round:")
        for round_num, loss in history.losses_distributed:
            print(f"  Round {round_num}: {loss:.4f}")
    
    # Print distributed metrics
    if history.metrics_distributed:
        print("\nDistributed Metrics per Round:")
        for metric_name, values in history.metrics_distributed.items():
            print(f"\n  {metric_name}:")
            for round_num, value in values:
                print(f"    Round {round_num}: {value:.4f}")
    
    # Print centralized metrics (if any)
    if history.metrics_centralized:
        print("\nCentralized Metrics per Round:")
        for metric_name, values in history.metrics_centralized.items():
            print(f"\n  {metric_name}:")
            for round_num, value in values:
                print(f"    Round {round_num}: {value:.4f}")
    
    print("\n" + "=" * 60)


# TODO: Future extensions
# - Add custom aggregation strategies (FedNova, SCAFFOLD)
# - Add server-side evaluation on held-out test set
# - Add model checkpointing per round
# - Add early stopping based on global metrics
# - Add TensorBoard/W&B logging integration
# - Add support for asynchronous federated learning
