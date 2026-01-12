"""
Federated Dataset Partitioner
=============================

Utilities for partitioning datasets across federated clients.
Supports IID and non-IID data distributions.

This module provides:
- FederatedDatasetPartitioner: Class-based partitioner with multiple strategies
- iid_partition: Simple function for IID partitioning
- dirichlet_partition: Non-IID partitioning using Dirichlet distribution
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, Subset
from collections import Counter

from ..utils.logging import get_logger


logger = get_logger("FederatedPartitioner")


def iid_partition(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> Dict[int, Subset]:
    """
    Partition dataset into IID (Independent and Identically Distributed) splits.
    
    Each client receives an equal-sized random subset of the data.
    
    Args:
        dataset: Full dataset to partition
        num_clients: Number of federated clients
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping client IDs (0 to num_clients-1) to dataset Subsets
    """
    np.random.seed(seed)
    
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    
    # Split indices as evenly as possible
    split_indices = np.array_split(indices, num_clients)
    
    partitions = {
        client_id: Subset(dataset, split_indices[client_id].tolist())
        for client_id in range(num_clients)
    }
    
    logger.info(
        f"IID partition: {num_samples} samples -> {num_clients} clients "
        f"(~{num_samples // num_clients} samples each)"
    )
    
    return partitions


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    min_samples_per_client: int = 10,
    seed: int = 42,
) -> Dict[int, Subset]:
    """
    Partition dataset using Dirichlet distribution for non-IID splits.
    
    Lower alpha values create more heterogeneous (non-IID) distributions.
    - alpha = 0.1: Very heterogeneous (clients have mostly one class)
    - alpha = 0.5: Moderately heterogeneous
    - alpha = 1.0: Mildly heterogeneous
    - alpha = 10.0: Nearly IID
    
    Args:
        dataset: Full dataset to partition
        num_clients: Number of federated clients
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        min_samples_per_client: Minimum samples each client must receive
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping client IDs to dataset Subsets
    """
    np.random.seed(seed)
    
    # Extract labels from dataset
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    labels = np.array(labels)
    
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    # Group indices by class
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
    
    # Initialize client indices
    client_indices = {i: [] for i in range(num_clients)}
    
    # Distribute each class using Dirichlet distribution
    for class_id, indices in class_indices.items():
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Adjust proportions to ensure minimum samples
        proportions = np.array([
            max(p, min_samples_per_client / len(indices)) 
            for p in proportions
        ])
        proportions = proportions / proportions.sum()
        
        # Allocate indices to clients
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)
        
        for client_id, client_class_indices in enumerate(split_indices):
            client_indices[client_id].extend(client_class_indices.tolist())
    
    # Create subsets
    partitions = {
        client_id: Subset(dataset, indices)
        for client_id, indices in client_indices.items()
    }
    
    # Log distribution statistics
    for client_id, subset in partitions.items():
        client_labels = [labels[i] for i in client_indices[client_id]]
        dist = Counter(client_labels)
        logger.debug(
            f"Client {client_id}: {len(subset)} samples, "
            f"distribution: {dict(dist)}"
        )
    
    logger.info(
        f"Dirichlet partition (alpha={alpha}): {num_samples} samples -> "
        f"{num_clients} clients"
    )
    
    return partitions


def pathological_partition(
    dataset: Dataset,
    num_clients: int,
    classes_per_client: int = 2,
    seed: int = 42,
) -> Dict[int, Subset]:
    """
    Create pathological non-IID partition where each client has few classes.
    
    This creates extreme heterogeneity by limiting each client to only
    a small subset of classes.
    
    Args:
        dataset: Full dataset to partition
        num_clients: Number of federated clients
        classes_per_client: Number of classes each client receives
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping client IDs to dataset Subsets
    """
    np.random.seed(seed)
    
    # Extract labels
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    labels = np.array(labels)
    
    num_classes = len(np.unique(labels))
    
    # Group indices by class
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    
    # Assign classes to clients (round-robin style)
    client_classes = {i: [] for i in range(num_clients)}
    all_classes = list(range(num_classes))
    
    for i, class_id in enumerate(all_classes * (classes_per_client * num_clients // num_classes + 1)):
        client_id = i % num_clients
        if len(client_classes[client_id]) < classes_per_client:
            client_classes[client_id].append(class_id)
    
    # Distribute samples
    client_indices = {i: [] for i in range(num_clients)}
    
    for client_id, classes in client_classes.items():
        for class_id in classes:
            # Get indices for this class and shuffle
            indices = class_indices[class_id].copy()
            np.random.shuffle(indices)
            
            # Take a portion for this client
            num_clients_with_class = sum(
                1 for c_classes in client_classes.values() if class_id in c_classes
            )
            portion_size = len(indices) // num_clients_with_class
            
            # Find how many clients already took from this class
            clients_before = sum(
                1 for cid in range(client_id) 
                if class_id in client_classes[cid]
            )
            
            start_idx = clients_before * portion_size
            end_idx = start_idx + portion_size
            
            client_indices[client_id].extend(indices[start_idx:end_idx])
    
    partitions = {
        client_id: Subset(dataset, indices)
        for client_id, indices in client_indices.items()
    }
    
    logger.info(
        f"Pathological partition ({classes_per_client} classes/client): "
        f"{len(dataset)} samples -> {num_clients} clients"
    )
    
    return partitions


class FederatedDatasetPartitioner:
    """
    Partitions a dataset across multiple federated learning clients.
    
    Supports various partitioning strategies:
    - IID (Independent and Identically Distributed)
    - Non-IID by labels (each client gets subset of classes)
    - Non-IID by Dirichlet distribution
    - Pathological non-IID (extreme heterogeneity)
    
    Args:
        dataset: Full dataset to partition
        num_clients: Number of federated clients
        partition_strategy: Partitioning strategy ('iid', 'dirichlet', 'pathological')
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        dataset: Dataset,
        num_clients: int,
        partition_strategy: str = "iid",
        seed: int = 42,
    ):
        self.dataset = dataset
        self.num_clients = num_clients
        self.partition_strategy = partition_strategy
        self.seed = seed
        
        self.client_indices: Dict[int, List[int]] = {}
        self._partitions: Optional[Dict[int, Subset]] = None
    
    def partition(
        self,
        alpha: float = 0.5,
        min_samples_per_client: int = 10,
        classes_per_client: int = 2,
    ) -> Dict[int, Subset]:
        """
        Partition the dataset according to the specified strategy.
        
        Args:
            alpha: Dirichlet concentration parameter (for 'dirichlet' strategy)
            min_samples_per_client: Minimum samples each client must receive
            classes_per_client: Classes per client (for 'pathological' strategy)
        
        Returns:
            Dictionary mapping client IDs to dataset subsets
        """
        if self.partition_strategy == "iid":
            self._partitions = iid_partition(
                self.dataset, 
                self.num_clients, 
                self.seed
            )
        elif self.partition_strategy == "dirichlet":
            self._partitions = dirichlet_partition(
                self.dataset,
                self.num_clients,
                alpha=alpha,
                min_samples_per_client=min_samples_per_client,
                seed=self.seed,
            )
        elif self.partition_strategy == "pathological":
            self._partitions = pathological_partition(
                self.dataset,
                self.num_clients,
                classes_per_client=classes_per_client,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition_strategy}")
        
        # Store indices for statistics
        for client_id, subset in self._partitions.items():
            self.client_indices[client_id] = subset.indices
        
        return self._partitions
    
    def get_statistics(self) -> Dict:
        """
        Get partition statistics for analysis.
        
        Returns:
            Dictionary with partition statistics
        """
        if self._partitions is None:
            raise RuntimeError("Call partition() first before getting statistics")
        
        # Extract labels
        labels = []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            labels.append(label)
        labels = np.array(labels)
        
        stats = {
            "num_clients": self.num_clients,
            "total_samples": len(self.dataset),
            "partition_strategy": self.partition_strategy,
            "samples_per_client": {},
            "class_distribution_per_client": {},
        }
        
        for client_id, indices in self.client_indices.items():
            client_labels = labels[indices]
            stats["samples_per_client"][client_id] = len(indices)
            stats["class_distribution_per_client"][client_id] = dict(Counter(client_labels.tolist()))
        
        # Calculate imbalance metrics
        samples_list = list(stats["samples_per_client"].values())
        stats["min_samples"] = min(samples_list)
        stats["max_samples"] = max(samples_list)
        stats["std_samples"] = float(np.std(samples_list))
        
        return stats
    
    def print_statistics(self) -> None:
        """Print partition statistics in a readable format."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("FEDERATED DATA PARTITION STATISTICS")
        print("=" * 60)
        print(f"Strategy: {stats['partition_strategy']}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Number of clients: {stats['num_clients']}")
        print(f"Samples range: [{stats['min_samples']}, {stats['max_samples']}]")
        print(f"Samples std: {stats['std_samples']:.2f}")
        print("\nPer-client distribution:")
        
        for client_id in range(self.num_clients):
            samples = stats["samples_per_client"][client_id]
            dist = stats["class_distribution_per_client"][client_id]
            print(f"  Client {client_id}: {samples} samples, classes: {dist}")
        
        print("=" * 60 + "\n")


# TODO: Future extensions
# - Add quantity skew partitioning (unequal sample sizes)
# - Add label noise injection for robustness testing
# - Add temporal/concept drift simulation
# - Add support for multi-dataset partitioning
# - Add visualization utilities for partition analysis

