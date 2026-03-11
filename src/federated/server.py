# =============================================================================
# Flower FL Server for Skin Cancer Classification
# =============================================================================
"""
Flower FL Server for Skin Cancer Classification.

The server coordinates the federated learning process across all clients,
aggregating model updates and managing the training rounds.
"""

# =============================================================================
# Imports
# =============================================================================

from flwr.server import ServerConfig
from flwr.server import start_server as fl_start_server
from flwr.server.strategy import Strategy
from flwr.server.history import History
from flwr.common import Scalar
from typing import Any
import torch
from pathlib import Path

from ..models.dscatnet import DSCATNet, get_model_parameters
from .strategy import create_fedavg_strategy

# =============================================================================
# Server Setup Functions
# =============================================================================


def create_server(
    model: DSCATNet,
    num_rounds: int = 50,
    min_fit_clients: int = 4,
    min_evaluate_clients: int = 4,
    min_available_clients: int = 4,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    strategy: Strategy | None = None,
    save_path: str | None = None,
) -> tuple[ServerConfig, Strategy]:
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
            save_path=save_path,
        )

    # Server configuration
    config = ServerConfig(num_rounds=num_rounds)

    return config, strategy


def start_server(
    server_address: str = "[::]:8080",
    model: DSCATNet | None = None,
    num_classes: int = 7,
    num_rounds: int = 50,
    strategy: Strategy | None = None,
    **kwargs,
) -> History:
    """
    Start the Flower FL server.

    Args:
        server_address: Address to run the server on
        model: DSCATNet model for initialization
        num_classes: Number of output classes (used when model is None)
        num_rounds: Number of FL rounds
        strategy: Custom strategy (optional)
        **kwargs: Additional arguments for create_server

    Returns:
        Flower History object with training results
    """
    if model is None:
        # Create default model for parameter initialization
        model = DSCATNet(num_classes=num_classes)

    config, strategy = create_server(model=model, num_rounds=num_rounds, strategy=strategy, **kwargs)

    # Start server
    history = fl_start_server(server_address=server_address, config=config, strategy=strategy)

    return history


# =============================================================================
# High-Level Server Wrapper
# =============================================================================


class FederatedServer:
    """
    High-level wrapper for managing FL server with DSCATNet.

    Provides convenient methods for:
    - Configuring FL strategy and aggregation
    - Saving/loading checkpoints
    - Tracking and aggregating metrics

    Args:
        model: DSCATNet model.
        num_rounds: Number of FL rounds.
        checkpoint_dir: Directory for saving checkpoints.
        log_dir: Directory for experiment logs.
    """

    def __init__(
        self, model: DSCATNet, num_rounds: int = 50, checkpoint_dir: str = "./checkpoints", log_dir: str = "./logs"
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
        self.history: dict[str, list[Any]] = {
            "round": [],
            "loss": [],
            "accuracy": [],
            "clients_trained": [],
            "clients_evaluated": [],
        }

    def configure(self, min_clients: int = 4, fraction_fit: float = 1.0, fraction_evaluate: float = 1.0) -> Strategy:
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
            evaluate_metrics_aggregation_fn=self._aggregate_metrics,
        )

        return strategy

    def _aggregate_metrics(self, metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
        """Aggregate evaluation metrics from all clients."""
        if not metrics:
            return {}

        aggregated = {}
        metric_keys = metrics[0][1].keys()

        for key in metric_keys:
            if key == "client_id":
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
        torch.save(
            {"round": round_num, "model_state_dict": self.model.state_dict(), "history": self.history}, checkpoint_path
        )
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return the round number.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            The round number from the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", self.history)
        return checkpoint["round"]

    def get_summary(self) -> dict[str, Any]:
        """Get summary of FL training.

        Returns:
            Dictionary with training summary including total rounds,
            final accuracy, best accuracy, and final loss.
        """
        return {
            "total_rounds": len(self.history["round"]),
            "final_accuracy": self.history["accuracy"][-1] if self.history["accuracy"] else None,
            "best_accuracy": max(self.history["accuracy"]) if self.history["accuracy"] else None,
            "final_loss": self.history["loss"][-1] if self.history["loss"] else None,
        }
