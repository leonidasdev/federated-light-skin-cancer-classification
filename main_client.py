"""
Punto de entrada principal para el cliente federado.

Uso:
    python main_client.py --node-id NODE_ID --dataset DATASET [--server SERVER]

Ejemplo:
    python main_client.py --node-id 0 --dataset HAM10000 --server [::]:8080
"""

import argparse
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import FEDERATED_CONFIG, NODES_CONFIG
from client.client import start_client
from utils.logging_utils import setup_logger


def parse_args():
    """
    Parsea argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description='Cliente Federado para Clasificación de Cáncer de Piel'
    )
    
    parser.add_argument(
        '--node-id',
        type=int,
        required=True,
        help='ID del nodo cliente'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['HAM10000', 'ISIC2018', 'ISIC2019', 'ISIC2020'],
        help='Dataset asignado a este nodo'
    )
    
    parser.add_argument(
        '--server',
        type=str,
        default=FEDERATED_CONFIG['server_address'],
        help='Dirección del servidor (formato: [host]:port)'
    )
    
    parser.add_argument(
        '--local-epochs',
        type=int,
        default=None,
        help='Número de épocas locales por ronda (override config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Tamaño de batch para entrenamiento (override config)'
    )
    
    return parser.parse_args()


def main():
    """
    Función principal del cliente.
    """
    # Parsear argumentos
    args = parse_args()
    
    # Configurar logging
    logger = setup_logger(f'Client_{args.node_id}')
    
    logger.info("=" * 60)
    logger.info(f"INICIANDO CLIENTE {args.node_id}")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Servidor: {args.server}")
    
    if args.local_epochs:
        logger.info(f"Épocas locales: {args.local_epochs}")
    if args.batch_size:
        logger.info(f"Batch size: {args.batch_size}")
    
    logger.info("=" * 60)
    
    try:
        # Validar node_id en configuración
        node_configs = NODES_CONFIG['nodes']
        node_config = next((n for n in node_configs if n['node_id'] == args.node_id), None)
        
        if node_config is None:
            logger.warning(f"Node ID {args.node_id} no está en configuración, continuando...")
        else:
            logger.info(f"Configuración de nodo encontrada: {node_config['name']}")
        
        # Iniciar cliente
        logger.info("Conectando al servidor...")
        logger.info("Presiona Ctrl+C para detener el cliente")
        
        start_client(
            node_id=args.node_id,
            dataset_name=args.dataset,
            server_address=args.server
        )
        
        logger.info("Cliente finalizado exitosamente")
    
    except KeyboardInterrupt:
        logger.info("Cliente detenido por el usuario")
    
    except Exception as e:
        logger.error(f"Error crítico en cliente: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
