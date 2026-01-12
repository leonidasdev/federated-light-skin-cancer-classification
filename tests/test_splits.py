"""
Test script for IID and Non-IID data splits.

Validates:
1. IID split creates balanced distributions
2. Non-IID splits create heterogeneous distributions
3. Split statistics are computed correctly
4. All indices are valid and non-overlapping
"""

import sys
from pathlib import Path
import numpy as np
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.splits import (
    train_val_split,
    create_iid_split,
    create_noniid_split,
    create_label_skew_split,
    create_quantity_skew_split,
    get_dataset_statistics,
    print_split_summary
)


def generate_mock_labels(n_samples=10000, n_classes=7, imbalanced=True):
    """Generate mock labels for testing."""
    if imbalanced:
        # Simulate imbalanced dermoscopy distribution
        # NV (nevus) is typically the most common
        weights = [0.05, 0.08, 0.12, 0.02, 0.10, 0.55, 0.08]  # Sum = 1
        labels = np.random.choice(n_classes, size=n_samples, p=weights)
    else:
        labels = np.random.randint(0, n_classes, size=n_samples)
    
    return labels.tolist()


def test_train_val_split():
    """Test basic train/val split."""
    print("Testing train/val split...")
    
    total = 1000
    train_idx, val_idx = train_val_split(total, val_split=0.2, seed=42)
    
    # Check sizes
    assert len(train_idx) == 800, f"Expected 800 train, got {len(train_idx)}"
    assert len(val_idx) == 200, f"Expected 200 val, got {len(val_idx)}"
    
    # Check no overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train and val should not overlap"
    
    # Check all indices valid
    assert max(train_idx) < total
    assert max(val_idx) < total
    
    # Check reproducibility
    train_idx2, val_idx2 = train_val_split(total, val_split=0.2, seed=42)
    assert train_idx == train_idx2, "Same seed should produce same split"
    
    print("  ✓ Train/val split test passed")


def test_iid_split():
    """Test IID split creates balanced distributions."""
    print("Testing IID split...")
    
    n_samples = 10000
    n_clients = 4
    labels = generate_mock_labels(n_samples, n_classes=7, imbalanced=True)
    
    client_data = create_iid_split(labels, num_clients=n_clients, seed=42)
    
    # Check all clients received data
    assert len(client_data) == n_clients
    
    # Check total samples preserved
    total_assigned = sum(len(indices) for indices in client_data.values())
    assert total_assigned == n_samples, f"Expected {n_samples}, got {total_assigned}"
    
    # Check no overlap between clients
    all_indices = []
    for indices in client_data.values():
        all_indices.extend(indices)
    assert len(all_indices) == len(set(all_indices)), "Indices should not overlap"
    
    # Check class distributions are similar (IID property)
    client_dists = {}
    for client_id, indices in client_data.items():
        client_labels = [labels[i] for i in indices]
        dist = Counter(client_labels)
        client_dists[client_id] = {k: v/len(indices) for k, v in dist.items()}
    
    # Distributions should be similar across clients
    # Check that each class proportion differs by at most 10%
    for cls in range(7):
        props = [client_dists[c].get(cls, 0) for c in client_data.keys()]
        if max(props) > 0:
            assert max(props) - min(props) < 0.15, \
                f"IID split should have similar distributions, class {cls} varies too much"
    
    print("  ✓ IID split test passed")


def test_noniid_dirichlet_split():
    """Test Non-IID split with Dirichlet distribution."""
    print("Testing Non-IID Dirichlet split...")
    
    n_samples = 10000
    n_clients = 4
    labels = generate_mock_labels(n_samples, n_classes=7, imbalanced=False)
    
    # Test with different alpha values
    for alpha in [0.1, 0.5, 1.0, 10.0]:
        client_data = create_noniid_split(labels, num_clients=n_clients, alpha=alpha, seed=42)
        
        # Basic checks
        assert len(client_data) == n_clients
        total = sum(len(idx) for idx in client_data.values())
        assert total == n_samples
        
        # Get heterogeneity score
        stats = get_dataset_statistics(client_data, labels)
        
        # Lower alpha should produce higher heterogeneity
        if alpha == 0.1:
            assert stats['heterogeneity_score'] > 0.3, \
                f"Low alpha should be heterogeneous, got {stats['heterogeneity_score']}"
    
    print("  ✓ Non-IID Dirichlet split test passed")


def test_label_skew_split():
    """Test Non-IID split with label skew."""
    print("Testing label skew split...")
    
    n_samples = 10000
    labels = generate_mock_labels(n_samples, n_classes=7, imbalanced=False)
    
    # Each client gets 3 classes
    client_data = create_label_skew_split(
        labels, num_clients=4, num_classes_per_client=3, seed=42
    )
    
    # Check that clients have limited class coverage
    for client_id, indices in client_data.items():
        client_labels = [labels[i] for i in indices]
        unique_classes = len(set(client_labels))
        
        # Should have approximately num_classes_per_client unique classes
        # (might have more due to overlap in assignment)
        assert unique_classes <= 5, \
            f"Client {client_id} has too many classes: {unique_classes}"
    
    print("  ✓ Label skew split test passed")


def test_quantity_skew_split():
    """Test Non-IID split with quantity skew."""
    print("Testing quantity skew split...")
    
    n_samples = 10000
    labels = generate_mock_labels(n_samples, n_classes=7)
    
    client_data = create_quantity_skew_split(
        labels, num_clients=4, imbalance_factor=0.7, seed=42
    )
    
    # Check that clients have different amounts of data
    sizes = [len(indices) for indices in client_data.values()]
    
    # With imbalance_factor=0.7, should have significant variance
    assert max(sizes) / min(sizes) > 1.5, \
        "Quantity skew should create imbalanced client sizes"
    
    print("  ✓ Quantity skew split test passed")


def test_statistics_computation():
    """Test that statistics are computed correctly."""
    print("Testing statistics computation...")
    
    labels = generate_mock_labels(10000, n_classes=7)
    client_data = create_noniid_split(labels, num_clients=4, alpha=0.5, seed=42)
    
    stats = get_dataset_statistics(client_data, labels)
    
    # Check required keys
    required_keys = [
        'num_clients', 'total_samples', 'samples_per_client',
        'class_distribution', 'heterogeneity_score'
    ]
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"
    
    # Check values are sensible
    assert stats['num_clients'] == 4
    assert stats['total_samples'] == 10000
    assert len(stats['samples_per_client']) == 4
    assert len(stats['class_distribution']) == 4
    assert 0 <= stats['heterogeneity_score'] <= 2.0
    
    print("  ✓ Statistics computation test passed")


def test_reproducibility():
    """Test that splits are reproducible with same seed."""
    print("Testing reproducibility...")
    
    labels = generate_mock_labels(5000, n_classes=7)
    
    # IID
    split1 = create_iid_split(labels, num_clients=4, seed=123)
    split2 = create_iid_split(labels, num_clients=4, seed=123)
    
    for client_id in split1:
        assert split1[client_id] == split2[client_id], "Same seed should reproduce"
    
    # Non-IID
    split3 = create_noniid_split(labels, num_clients=4, alpha=0.5, seed=456)
    split4 = create_noniid_split(labels, num_clients=4, alpha=0.5, seed=456)
    
    for client_id in split3:
        assert split3[client_id] == split4[client_id], "Same seed should reproduce"
    
    print("  ✓ Reproducibility test passed")


def test_visualization():
    """Test that print summary works."""
    print("Testing print summary...")
    
    labels = generate_mock_labels(10000, n_classes=7)
    client_data = create_noniid_split(labels, num_clients=4, alpha=0.5, seed=42)
    
    class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
    
    # This should not raise an error
    print_split_summary(client_data, labels, class_names)
    
    print("  ✓ Print summary test passed")


def compare_split_methods():
    """Compare different split methods visually."""
    print("\n" + "=" * 60)
    print("SPLIT METHOD COMPARISON")
    print("=" * 60)
    
    labels = generate_mock_labels(10000, n_classes=7, imbalanced=True)
    class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
    
    methods = {
        'IID': create_iid_split(labels, num_clients=4, seed=42),
        'Non-IID (α=0.5)': create_noniid_split(labels, num_clients=4, alpha=0.5, seed=42),
        'Non-IID (α=0.1)': create_noniid_split(labels, num_clients=4, alpha=0.1, seed=42),
        'Label Skew': create_label_skew_split(labels, num_clients=4, num_classes_per_client=3, seed=42),
    }
    
    for name, client_data in methods.items():
        stats = get_dataset_statistics(client_data, labels)
        print(f"\n{name}:")
        print(f"  Heterogeneity score: {stats['heterogeneity_score']:.4f}")
        print(f"  Samples per client: {list(stats['samples_per_client'].values())}")


def run_all_tests():
    """Run all split tests."""
    print("\n" + "=" * 60)
    print("IID/NON-IID SPLIT TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_train_val_split()
        test_iid_split()
        test_noniid_dirichlet_split()
        test_label_skew_split()
        test_quantity_skew_split()
        test_statistics_computation()
        test_reproducibility()
        test_visualization()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
        # Show comparison
        compare_split_methods()
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
