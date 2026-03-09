# =============================================================================
# Data Splitting Utilities for IID and Non-IID Scenarios
# =============================================================================
"""
Data Splitting Utilities for IID and Non-IID Scenarios.

Provides functions to create different data distributions
for federated learning experiments:
- IID: Each client gets uniform random sample
- Non-IID: Each client gets dataset from a specific source (natural non-IID)
- Synthetic Non-IID: Label distribution skew within each dataset
"""

# =============================================================================
# Imports
# =============================================================================

import logging

import numpy as np
import torch
from typing import Any
from collections import defaultdict

logger = logging.getLogger(__name__)

__all__ = [
    "create_iid_split",
    "create_label_skew_split",
    "create_noniid_split",
    "create_quantity_skew_split",
    "deterministic_train_val_split",
    "deterministic_train_val_test_split",
    "get_dataset_statistics",
    "print_split_summary",
]

# =============================================================================
# Basic Split Functions
# =============================================================================


def deterministic_train_val_split(
    total_size: int,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split indices into train and val sets using a torch Generator for reproducibility.

    This is the canonical split function used by both centralized and federated
    training pipelines to ensure identical splits across runs.

    Args:
        total_size: Total number of samples in the dataset.
        val_split: Fraction of samples reserved for validation.
        seed: Random seed for the torch Generator.

    Returns:
        Tuple of (train_indices, val_indices).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    indices = torch.randperm(total_size, generator=gen).tolist()

    val_n = int(total_size * val_split)
    train_n = total_size - val_n
    return indices[:train_n], indices[train_n:]


def deterministic_train_val_test_split(
    total_size: int,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split indices into train, val, and test sets.

    Uses the same Generator-based approach as deterministic_train_val_split
    for reproducibility. The test set is carved from the end, then val from
    the remaining indices, so that when test_split=0 the train/val split is
    identical to deterministic_train_val_split.

    Args:
        total_size: Total number of samples.
        val_split: Fraction reserved for validation.
        test_split: Fraction reserved for testing.
        seed: Random seed.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    indices = torch.randperm(total_size, generator=gen).tolist()

    test_n = int(total_size * test_split)
    remaining = total_size - test_n
    val_n = int(remaining * (val_split / (1.0 - test_split)))
    train_n = remaining - val_n

    return indices[:train_n], indices[train_n:train_n + val_n], indices[train_n + val_n:]


def create_iid_split(
    labels: list[int],
    num_clients: int = 4,
    seed: int = 42
) -> dict[int, list[int]]:
    """
    Create IID (Independent and Identically Distributed) split.

    Each client receives a uniformly random subset of the data,
    maintaining similar class distributions across clients.

    Args:
        labels: List of labels for all samples
        num_clients: Number of FL clients
        seed: Random seed

    Returns:
        Dictionary mapping client_id to list of sample indices
    """
    np.random.seed(seed)

    num_samples = len(labels)
    indices = np.random.permutation(num_samples)

    # Split indices evenly across clients
    splits = np.array_split(indices, num_clients)

    client_data = {
        i + 1: splits[i].tolist() for i in range(num_clients)
    }

    return client_data


def create_noniid_split(
    labels: list[int],
    num_clients: int = 4,
    alpha: float = 0.5,
    seed: int = 42
) -> dict[int, list[int]]:
    """
    Create Non-IID split using Dirichlet distribution.

    Lower alpha = more heterogeneous (more non-IID)
    Higher alpha = more homogeneous (closer to IID)

    Args:
        labels: List of labels for all samples
        num_clients: Number of FL clients
        alpha: Dirichlet concentration parameter
        seed: Random seed

    Returns:
        Dictionary mapping client_id to list of sample indices
    """
    np.random.seed(seed)

    np_labels = np.array(labels)
    num_classes = len(np.unique(np_labels[np_labels >= 0]))

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(np_labels):
        if label >= 0:
            class_indices[int(label)].append(idx)

    # Sample proportions from Dirichlet distribution
    client_data = {i + 1: [] for i in range(num_clients)}

    for class_id in range(num_classes):
        indices = np.array(class_indices[class_id])
        np.random.shuffle(indices)

        # Sample proportions for this class
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Normalize to handle rounding
        proportions = (proportions * len(indices)).astype(int)
        proportions[-1] = len(indices) - proportions[:-1].sum()

        # Distribute indices to clients
        start = 0
        for client_id in range(num_clients):
            end = start + proportions[client_id]
            client_data[client_id + 1].extend(indices[start:end].tolist())
            start = end

    # Shuffle each client's data
    for data in client_data.values():
        np.random.shuffle(data)

    return client_data


def create_label_skew_split(
    labels: list[int],
    num_clients: int = 4,
    num_classes_per_client: int = 3,
    seed: int = 42
) -> dict[int, list[int]]:
    """
    Create Non-IID split with label skew.

    Each client gets data from only a subset of classes.

    Args:
        labels: List of labels for all samples
        num_clients: Number of FL clients
        num_classes_per_client: Number of classes each client has
        seed: Random seed

    Returns:
        Dictionary mapping client_id to list of sample indices
    """
    np.random.seed(seed)

    np_labels = np.array(labels)
    unique_classes = np.unique(np_labels[np_labels >= 0])
    num_classes = len(unique_classes)

    # Assign classes to clients (with overlap possible)
    client_classes = {}
    for i in range(num_clients):
        # Rotate class assignment to ensure coverage
        start_class = (i * num_classes_per_client) % num_classes
        assigned_classes = [
            unique_classes[(start_class + j) % num_classes]
            for j in range(num_classes_per_client)
        ]
        client_classes[i + 1] = assigned_classes

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(np_labels):
        if label >= 0:
            class_indices[int(label)].append(idx)

    # Distribute to clients
    client_data = {i + 1: [] for i in range(num_clients)}

    for class_id in unique_classes:
        indices = class_indices[class_id]
        np.random.shuffle(indices)

        # Find clients that have this class
        clients_with_class = [
            c for c in range(1, num_clients + 1)
            if class_id in client_classes[c]
        ]

        if clients_with_class:
            # Split among clients with this class
            splits = np.array_split(indices, len(clients_with_class))
            for client_id, split in zip(clients_with_class, splits):
                client_data[client_id].extend(split.tolist())

    # Shuffle each client's data
    for data in client_data.values():
        np.random.shuffle(data)

    return client_data


def create_quantity_skew_split(
    labels: list[int],
    num_clients: int = 4,
    imbalance_factor: float = 0.5,
    seed: int = 42
) -> dict[int, list[int]]:
    """
    Create Non-IID split with quantity skew.

    Clients have different amounts of data.

    Args:
        labels: List of labels for all samples
        num_clients: Number of FL clients
        imbalance_factor: Controls imbalance (0 = balanced, 1 = very imbalanced)
        seed: Random seed

    Returns:
        Dictionary mapping client_id to list of sample indices
    """
    np.random.seed(seed)

    num_samples = len(labels)
    indices = np.random.permutation(num_samples).tolist()

    # Generate imbalanced proportions
    proportions = np.array([
        1 + imbalance_factor * (num_clients - 1 - i)
        for i in range(num_clients)
    ])
    proportions = proportions / proportions.sum()

    # Shuffle proportions to randomize which client gets more
    np.random.shuffle(proportions)

    # Distribute indices
    client_data = {}
    start = 0
    for i in range(num_clients):
        count = int(proportions[i] * num_samples)
        if i == num_clients - 1:
            count = num_samples - start  # Give remaining to last client
        client_data[i + 1] = indices[start:start + count]
        start += count

    return client_data


def get_dataset_statistics(
    client_data: dict[int, list[int]],
    labels: list[int]
) -> dict[str, Any]:
    """
    Compute statistics about the data distribution across clients.

    Args:
        client_data: Dictionary mapping client_id to indices
        labels: List of all labels

    Returns:
        Dictionary with distribution statistics
    """
    np_labels = np.array(labels)
    num_classes = len(np.unique(np_labels[np_labels >= 0]))

    stats = {
        'num_clients': len(client_data),
        'total_samples': sum(len(indices) for indices in client_data.values()),
        'samples_per_client': {},
        'class_distribution': {},
        'class_overlap': None,
        'earth_movers_distance': None
    }

    # Per-client statistics
    client_class_dist = {}
    for client_id, indices in client_data.items():
        client_labels = np_labels[indices]
        unique, counts = np.unique(client_labels[client_labels >= 0], return_counts=True)

        stats['samples_per_client'][client_id] = len(indices)

        # Class distribution for this client
        dist = {int(c): 0 for c in range(num_classes)}
        for c, count in zip(unique, counts):
            dist[int(c)] = int(count)
        client_class_dist[client_id] = dist

    stats['class_distribution'] = client_class_dist

    # Compute EMD-based heterogeneity metric
    global_dist = np.zeros(num_classes)
    for c in range(num_classes):
        global_dist[c] = sum(
            client_class_dist[cid].get(c, 0)
            for cid in client_data
        )
    global_dist = global_dist / global_dist.sum()

    emd_sum = 0
    for client_id in client_data:
        client_dist = np.array([
            client_class_dist[client_id].get(c, 0)
            for c in range(num_classes)
        ])
        if client_dist.sum() > 0:
            client_dist = client_dist / client_dist.sum()
            # Simple L1 distance as proxy for EMD
            emd_sum += np.abs(client_dist - global_dist).sum()

    stats['heterogeneity_score'] = emd_sum / len(client_data)

    return stats


def print_split_summary(
    client_data: dict[int, list[int]],
    labels: list[int],
    class_names: list[str] | None = None
) -> None:
    """Print a formatted summary of the data split."""
    stats = get_dataset_statistics(client_data, labels)

    print("=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    print(f"Number of clients: {stats['num_clients']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Heterogeneity score: {stats['heterogeneity_score']:.4f}")
    print()

    print("Samples per client:")
    for client_id, count in stats['samples_per_client'].items():
        print(f"  Client {client_id}: {count}")
    print()

    print("Class distribution per client:")
    num_classes = len(next(iter(stats['class_distribution'].values())))

    # Header
    header = "Client | " + " | ".join(
        class_names[i][:8] if class_names else f"Class {i}"
        for i in range(num_classes)
    )
    print(header)
    print("-" * len(header))

    # Rows
    for client_id, dist in stats['class_distribution'].items():
        row = f"  {client_id}   | " + " | ".join(
            f"{dist.get(i, 0):7d}" for i in range(num_classes)
        )
        print(row)

    print("=" * 60)
