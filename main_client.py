"""
Main entry point for the federated client.

Usage:
    python main_client.py --node-id NODE_ID --dataset DATASET [--server SERVER]

Example:
    python main_client.py --node-id 0 --dataset HAM10000 --server [::]:8080
"""

import argparse
import sys
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import FEDERATED_CONFIG, NODES_CONFIG
from client.client import start_client
from utils.logging_utils import setup_logger


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Federated Client for Skin Cancer Classification'
    )
    
    parser.add_argument(
        '--node-id',
        type=int,
        required=True,
        help='Client node ID'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['HAM10000', 'ISIC2018', 'ISIC2019', 'ISIC2020'],
        help='Dataset assigned to this node'
    )
    
    parser.add_argument(
        '--server',
        type=str,
        default=FEDERATED_CONFIG['server_address'],
        help='Server address (format: [host]:port)'
    )
    
    parser.add_argument(
        '--local-epochs',
        type=int,
        default=None,
        help='Number of local epochs per round (override config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training (override config)'
    )
    
    return parser.parse_args()


def main():
    """Main client function."""

    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logger(f'Client_{args.node_id}')

    logger.info("=" * 60)
    logger.info(f"STARTING CLIENT {args.node_id}")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Server: {args.server}")

    if args.local_epochs:
        logger.info(f"Local epochs: {args.local_epochs}")
    if args.batch_size:
        logger.info(f"Batch size: {args.batch_size}")

    logger.info("=" * 60)

    try:
        # Validate node_id in configuration
        node_configs = NODES_CONFIG['nodes']
        node_config = next((n for n in node_configs if n['node_id'] == args.node_id), None)

        if node_config is None:
            logger.warning(f"Node ID {args.node_id} not found in configuration, continuing...")
        else:
            logger.info(f"Node configuration found: {node_config['name']}")

        # Start client
        logger.info("Connecting to server...")
        logger.info("Press Ctrl+C to stop the client")

        start_client(
            node_id=args.node_id,
            dataset_name=args.dataset,
            server_address=args.server
        )

        logger.info("Client finished successfully")

    except KeyboardInterrupt:
        logger.info("Client stopped by user")

    except Exception as e:
        logger.error(f"Critical error in client: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
