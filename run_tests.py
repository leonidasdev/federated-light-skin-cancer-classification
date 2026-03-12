# =============================================================================
# Test Runner and Project Setup
# =============================================================================
"""
Run all tests and setup verification for the project.

Usage:
    python run_tests.py              # Show help
    python run_tests.py --setup      # Setup project structure
    python run_tests.py --test       # Run tests only
    python run_tests.py --verbose    # Run tests with verbose output
    python run_tests.py --coverage   # Run tests with coverage report
"""

# =============================================================================
# Imports
# =============================================================================

import sys
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent


# =============================================================================
# Functions
# =============================================================================


def run_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """
    Run all test modules using pytest.

    Args:
        verbose: If True, show detailed test output.
        coverage: If True, generate code coverage report.

    Returns:
        True if all tests passed, False otherwise.
    """
    import pytest

    print("=" * 70)
    print("FEDERATED LEARNING SKIN CANCER PROJECT - TEST SUITE")
    print("=" * 70)

    args = [str(PROJECT_ROOT / "tests")]

    if verbose:
        args.append("-v")
    else:
        args.append("-q")

    if coverage:
        args.extend(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
            ]
        )

    exit_code = pytest.main(args)

    if coverage:
        print("\n" + "-" * 70)
        print("HTML coverage report generated: htmlcov/index.html")

    return exit_code == 0


def setup_project() -> None:
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
        "logs",
        "outputs",
    ]

    for dir_name in dirs_to_create:
        dir_path = PROJECT_ROOT / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")

    print("\nProject structure ready!")

    # Print download instructions
    print("\n" + "-" * 70)
    print("NEXT STEPS:")
    print("-" * 70)
    print("""
1. Download the datasets:
   python run_download.py --instructions

2. Verify datasets are properly organized:
   python run_download.py --verify

3. Explore datasets with the Jupyter notebook:
   jupyter notebook notebooks/01_dataset_exploration.ipynb

4. Run tests to verify everything works:
   pytest tests/ -v
""")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Project setup and testing")
    parser.add_argument("--setup", action="store_true", help="Setup project structure")
    parser.add_argument("--test", action="store_true", help="Run all tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose test output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage report")

    args = parser.parse_args()

    if args.setup:
        setup_project()
    elif args.test or args.coverage:
        success = run_tests(verbose=args.verbose, coverage=args.coverage)
        sys.exit(0 if success else 1)
    else:
        # Default: show help
        parser.print_help()
