"""
Skin lesion dataset loaders for Federated Learning.

Supported datasets:
 - HAM10000 (primary)
 - ISIC 2018
 - ISIC 2019
 - ISIC 2020

"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from config.config import DATA_CONFIG, CLASS_NAMES


class DatasetLoader:
    """
    Base class for skin lesion dataset loaders.
    """
    
    def __init__(self, dataset_name: str, dataset_path: str):
        """
        Initialize the dataset loader.

        Args:
            dataset_name (str): Dataset name ('HAM10000', 'ISIC2018', etc.)
            dataset_path (str): Path to the dataset directory
        """
        self.dataset_name = dataset_name
        self.dataset_path = Path(dataset_path)
        self.images = []
        self.labels = []
        self.metadata = None
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load the full dataset.

        Returns:
            tuple: (images, labels, metadata)
        """
        raise NotImplementedError("Subclases deben implementar load_data()")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Return class distribution for the dataset.

        Returns:
            dict: Mapping class -> count
        """
        # TODO: implement class counting
        pass
    
    def get_statistics(self) -> Dict:
        """
        Return dataset statistics.

        Returns:
            dict: Statistics (size, classes, distribution, etc.)
        """
        # TODO: implement statistics
        pass


class HAM10000Loader(DatasetLoader):
    """
    Loader for the HAM10000 dataset.

    Expected structure:
    HAM10000/
    ├── images/
    │   ├── ISIC_0024306.jpg
    │   └── ...
    └── HAM10000_metadata.csv
    """
    
    def __init__(self, dataset_path: str):
        super().__init__('HAM10000', dataset_path)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load HAM10000 data.

        Returns:
            tuple: (images, labels, metadata)
        """
        # TODO: implement HAM10000 loading
        # 1. Read metadata CSV
        # 2. Load images from images/ folder
        # 3. Map diagnoses to class indices
        # 4. Return numpy arrays

        print(f"Loading {self.dataset_name} from {self.dataset_path}...")

        # Placeholder
        return None, None, None


class ISIC2018Loader(DatasetLoader):
    """
    Loader for the ISIC 2018 dataset.

    Expected structure:
    ISIC2018/
    ├── ISIC2018_Task3_Training_Input/
    │   ├── ISIC_0000000.jpg
    │   └── ...
    └── ISIC2018_Task3_Training_GroundTruth.csv
    """
    
    def __init__(self, dataset_path: str):
        super().__init__('ISIC2018', dataset_path)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load ISIC 2018 data.

        Returns:
            tuple: (images, labels, metadata)
        """
        # TODO: implement ISIC2018 loading
        print(f"Loading {self.dataset_name} from {self.dataset_path}...")

        return None, None, None


class ISIC2019Loader(DatasetLoader):
    """
    Loader for the ISIC 2019 dataset.
    """
    
    def __init__(self, dataset_path: str):
        super().__init__('ISIC2019', dataset_path)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load ISIC 2019 data.

        Returns:
            tuple: (images, labels, metadata)
        """
        # TODO: implement ISIC2019 loading
        print(f"Loading {self.dataset_name} from {self.dataset_path}...")

        return None, None, None


class ISIC2020Loader(DatasetLoader):
    """
    Loader for the ISIC 2020 dataset.
    """
    
    def __init__(self, dataset_path: str):
        super().__init__('ISIC2020', dataset_path)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load ISIC 2020 data.

        Returns:
            tuple: (images, labels, metadata)
        """
        # TODO: implement ISIC2020 loading
        print(f"Loading {self.dataset_name} from {self.dataset_path}...")

        return None, None, None
# ==================== FUNCIONES DE UTILIDAD ====================

def get_dataset_loader(dataset_name: str, dataset_path: str) -> DatasetLoader:
    """
    Factory to obtain the appropriate loader for a dataset.

    Args:
        dataset_name (str): Dataset name
        dataset_path (str): Path to the dataset

    Returns:
        DatasetLoader: Specific loader for the dataset
    """
    loaders = {
        'HAM10000': HAM10000Loader,
        'ISIC2018': ISIC2018Loader,
        'ISIC2019': ISIC2019Loader,
        'ISIC2020': ISIC2020Loader,
    }
    
    loader_class = loaders.get(dataset_name)
    if loader_class is None:
        raise ValueError(f"Dataset {dataset_name} not supported. Options: {list(loaders.keys())}")
    
    return loader_class(dataset_path)


def load_node_data(node_id: int, nodes_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data for a specific FL node.

    Args:
        node_id (int): Node ID
        nodes_config (dict): Nodes configuration

    Returns:
        tuple: (X_train, y_train) for the node
    """
    # TODO: implement node-level data loading
    # 1. Read node configuration
    # 2. Load the corresponding dataset
    # 3. Apply distribution strategy (IID/non-IID)
    # 4. Return samples assigned to the node

    print(f"Loading data for node {node_id}...")

    return None, None


def split_data_iid(X: np.ndarray, y: np.ndarray, num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split data IID among clients.

    Args:
        X (np.ndarray): Images
        y (np.ndarray): Labels
        num_clients (int): Number of clients

    Returns:
        list: List of (X_client, y_client) tuples for each client
    """
    # TODO: implement IID splitting
    # 1. Shuffle data
    # 2. Split evenly
    # 3. Verify class distribution

    pass


def split_data_non_iid(X: np.ndarray, y: np.ndarray, num_clients: int, alpha: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split data non-IID using a Dirichlet distribution.

    Args:
        X (np.ndarray): Images
        y (np.ndarray): Labels
        num_clients (int): Number of clients
        alpha (float): Concentration parameter (lower = more heterogeneous)

    Returns:
        list: List of (X_client, y_client) tuples for each client
    """
    # TODO: implement Dirichlet non-IID splitting
    # 1. Sample proportions per class from Dirichlet
    # 2. Assign samples accordingly
    # 3. Ensure each client has data

    pass


def load_external_validation_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder for external validation dataset loader. External validation
    dataset support was removed from the project; this function returns
    (None, None) to indicate no external validation dataset is configured.

    Returns:
        tuple: (None, None)
    """
    print("No external validation dataset configured")
    return None, None


def print_dataset_info(dataset_name: str, X: np.ndarray, y: np.ndarray):
    """
    Print information about a dataset.

    Args:
        dataset_name (str): Dataset name
        X (np.ndarray): Images
        y (np.ndarray): Labels
    """
    print("\n" + "=" * 60)
    print(f"DATASET INFORMATION: {dataset_name}")
    print("=" * 60)

    if X is not None:
        print(f"Number of samples: {len(X)}")
        print(f"Image shape: {X[0].shape if len(X) > 0 else 'N/A'}")

    if y is not None:
        unique, counts = np.unique(y, return_counts=True)
        print(f"Number of classes: {len(unique)}")
        print("\nClass distribution:")
        for class_idx, count in zip(unique, counts):
            class_name = CLASS_NAMES.get(class_idx, f"Class {class_idx}")
            percentage = (count / len(y)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")

    print("=" * 60 + "\n")


# ==================== TESTING ====================

if __name__ == '__main__':
    # Test data loaders
    print("Testing data loaders...")

    # Example: load HAM10000
    try:
        loader = get_dataset_loader('HAM10000', DATA_CONFIG['ham10000_path'])
        X, y, metadata = loader.load_data()
        print_dataset_info('HAM10000', X, y)
    except Exception as e:
        print(f"Error loading HAM10000: {e}")
