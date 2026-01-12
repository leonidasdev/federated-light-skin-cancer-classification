"""
Run all tests and setup verification for the project.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_tests():
    """Run all test modules."""
    print("=" * 70)
    print("FEDERATED LEARNING SKIN CANCER PROJECT - TEST SUITE")
    print("=" * 70)
    
    all_passed = True
    
    # Test preprocessing
    print("\n\n>>> Running preprocessing tests...")
    try:
        from tests.test_preprocessing import run_all_tests as test_preprocessing
        if not test_preprocessing():
            all_passed = False
    except Exception as e:
        print(f"Preprocessing tests failed: {e}")
        all_passed = False
    
    # Test splits
    print("\n\n>>> Running split tests...")
    try:
        from tests.test_splits import run_all_tests as test_splits
        if not test_splits():
            all_passed = False
    except Exception as e:
        print(f"Split tests failed: {e}")
        all_passed = False
    
    # Verify datasets
    print("\n\n>>> Verifying datasets...")
    try:
        from src.data.verify import DatasetVerifier
        verifier = DatasetVerifier(str(project_root / "data"))
        verifier.verify_all(verbose=True)
    except Exception as e:
        print(f"Dataset verification failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
    
    return all_passed


def setup_project():
    """Setup project directories and verify environment."""
    print("=" * 70)
    print("PROJECT SETUP")
    print("=" * 70)
    
    # Create directories
    dirs_to_create = [
        "data",
        "data/HAM10000",
        "data/ISIC2018",
        "data/ISIC2019",
        "data/ISIC2020",
        "data/raw",
        "data/processed",
        "experiments",
        "experiments/centralized",
        "experiments/federated",
        "checkpoints",
        "logs"
    ]
    
    for dir_name in dirs_to_create:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")
    
    print("\nProject structure ready!")
    
    # Print download instructions
    print("\n" + "-" * 70)
    print("NEXT STEPS:")
    print("-" * 70)
    print("""
1. Download the datasets:
   python -m src.data.download --instructions
   
2. Verify datasets are properly organized:
   python -m src.data.download --verify
   
3. Explore datasets with the Jupyter notebook:
   jupyter notebook notebooks/01_dataset_exploration.ipynb
   
4. Run tests to verify everything works:
   python run_tests.py
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Project setup and testing")
    parser.add_argument("--setup", action="store_true", help="Setup project structure")
    parser.add_argument("--test", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_project()
    elif args.test:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        # Default: do both
        setup_project()
        print("\n")
        run_tests()
