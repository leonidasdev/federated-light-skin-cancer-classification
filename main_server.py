"""
Main entry point for the federated server.

Usage:
    python main_server.py [--strategy STRATEGY] [--rounds ROUNDS] [--address ADDRESS]

Example:
    python main_server.py --strategy FedAvg --rounds 50 --address [::]:8080
"""

import argparse
import sys
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import print_config_summary, FEDERATED_CONFIG
from server.server import create_server
from utils.logging_utils import setup_federated_logging
from utils.security import setup_secure_communication


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Federated Server for Skin Cancer Classification'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default=FEDERATED_CONFIG['strategy'],
        choices=['FedAvg', 'FedProx'],
        help='Federated aggregation strategy'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=FEDERATED_CONFIG['num_rounds'],
        help='Number of federated training rounds'
    )
    
    parser.add_argument(
        '--address',
        type=str,
        default=FEDERATED_CONFIG['server_address'],
        help='Server address (format: [host]:port)'
    )
    
    parser.add_argument(
        '--min-clients',
        type=int,
        default=FEDERATED_CONFIG['min_available_clients'],
        help='Minimum number of required clients'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='federated_skin_cancer',
        help='Experiment name (for logs)'
    )
    
    parser.add_argument(
        '--secure',
        action='store_true',
        help='Enable secure communications (TLS)'
    )
    
    return parser.parse_args()


def main():
    """Main server function."""

    # Parse arguments
    args = parse_args()

    # Setup logging
    logger, tb_logger = setup_federated_logging(args.experiment_name)

    # Print configuration
    print_config_summary()

    logger.info("=" * 60)
    logger.info("STARTING FEDERATED SERVER")
    logger.info("=" * 60)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Rounds: {args.rounds}")
    logger.info(f"Address: {args.address}")
    logger.info(f"Min clients: {args.min_clients}")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info("=" * 60)

    try:
        # Configure security if required
        if args.secure:
            logger.info("Configuring secure communications...")
            setup_secure_communication(args.address)

        # Create server
        logger.info("Creating federated server...")
        server = create_server(
            strategy_name=args.strategy,
            num_rounds=args.rounds,
            server_address=args.address
        )

        # Start server
        logger.info("Starting server - waiting for clients...")
        logger.info("Press Ctrl+C to stop the server")

        server.start()

        logger.info("Server finished successfully")

    except KeyboardInterrupt:
        logger.info("Server stopped by user")

    except Exception as e:
        logger.error(f"Critical error in server: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Close TensorBoard logger
        tb_logger.close()
        logger.info("Resources released successfully")


if __name__ == '__main__':
    main()
