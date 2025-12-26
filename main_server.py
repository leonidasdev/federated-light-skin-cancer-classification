"""
Punto de entrada principal para el servidor federado.

Uso:
    python main_server.py [--strategy STRATEGY] [--rounds ROUNDS] [--address ADDRESS]

Ejemplo:
    python main_server.py --strategy FedAvg --rounds 50 --address [::]:8080
"""

import argparse
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import print_config_summary, FEDERATED_CONFIG
from server.server import create_server
from utils.logging_utils import setup_federated_logging
from utils.security import setup_secure_communication


def parse_args():
    """
    Parsea argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description='Servidor Federado para Clasificación de Cáncer de Piel'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default=FEDERATED_CONFIG['strategy'],
        choices=['FedAvg', 'FedProx'],
        help='Estrategia de agregación federada'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=FEDERATED_CONFIG['num_rounds'],
        help='Número de rondas de entrenamiento federado'
    )
    
    parser.add_argument(
        '--address',
        type=str,
        default=FEDERATED_CONFIG['server_address'],
        help='Dirección del servidor (formato: [host]:port)'
    )
    
    parser.add_argument(
        '--min-clients',
        type=int,
        default=FEDERATED_CONFIG['min_available_clients'],
        help='Número mínimo de clientes requeridos'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='federated_skin_cancer',
        help='Nombre del experimento (para logs)'
    )
    
    parser.add_argument(
        '--secure',
        action='store_true',
        help='Habilitar comunicaciones seguras (TLS)'
    )
    
    return parser.parse_args()


def main():
    """
    Función principal del servidor.
    """
    # Parsear argumentos
    args = parse_args()
    
    # Configurar logging
    logger, tb_logger = setup_federated_logging(args.experiment_name)
    
    # Imprimir configuración
    print_config_summary()
    
    logger.info("=" * 60)
    logger.info("INICIANDO SERVIDOR FEDERADO")
    logger.info("=" * 60)
    logger.info(f"Estrategia: {args.strategy}")
    logger.info(f"Rondas: {args.rounds}")
    logger.info(f"Dirección: {args.address}")
    logger.info(f"Clientes mínimos: {args.min_clients}")
    logger.info(f"Experimento: {args.experiment_name}")
    logger.info("=" * 60)
    
    try:
        # Configurar seguridad si es necesario
        if args.secure:
            logger.info("Configurando comunicaciones seguras...")
            setup_secure_communication(args.address)
        
        # Crear servidor
        logger.info("Creando servidor federado...")
        server = create_server(
            strategy_name=args.strategy,
            num_rounds=args.rounds,
            server_address=args.address
        )
        
        # Iniciar servidor
        logger.info("Iniciando servidor - esperando clientes...")
        logger.info("Presiona Ctrl+C para detener el servidor")
        
        server.start()
        
        logger.info("Servidor finalizado exitosamente")
    
    except KeyboardInterrupt:
        logger.info("Servidor detenido por el usuario")
    
    except Exception as e:
        logger.error(f"Error crítico en servidor: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Cerrar TensorBoard logger
        tb_logger.close()
        logger.info("Recursos liberados correctamente")


if __name__ == '__main__':
    main()
