"""
Sistema de logging para Federated Learning.

Proporciona logging unificado para servidor y clientes.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from config.config import LOGGING_CONFIG


def setup_logger(name: str, 
                log_file: Optional[str] = None,
                level: str = None) -> logging.Logger:
    """
    Configura un logger para el sistema.
    
    Args:
        name (str): Nombre del logger
        log_file (str): Archivo de log (opcional)
        level (str): Nivel de logging
    
    Returns:
        logging.Logger: Logger configurado
    """
    # Nivel de logging
    if level is None:
        level = LOGGING_CONFIG.get('log_level', 'INFO')
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Evitar duplicación de handlers
    if logger.handlers:
        return logger
    
    # Formato
    formatter = logging.Formatter(
        fmt=LOGGING_CONFIG.get('log_format'),
        datefmt=LOGGING_CONFIG.get('date_format')
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (si se especifica)
    if log_file or LOGGING_CONFIG.get('save_logs_to_file', False):
        if log_file is None:
            log_dir = Path(LOGGING_CONFIG['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / LOGGING_CONFIG['log_filename']
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_experiment_logger(experiment_name: str) -> logging.Logger:
    """
    Crea un logger específico para un experimento.
    
    Args:
        experiment_name (str): Nombre del experimento
    
    Returns:
        logging.Logger: Logger del experimento
    """
    # Crear directorio de logs del experimento
    log_dir = Path(LOGGING_CONFIG['log_dir']) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Archivo de log con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    return setup_logger(experiment_name, str(log_file))


def log_system_info(logger: logging.Logger):
    """
    Registra información del sistema.
    
    Args:
        logger: Logger a usar
    """
    import platform
    import tensorflow as tf
    
    logger.info("=" * 60)
    logger.info("INFORMACIÓN DEL SISTEMA")
    logger.info("=" * 60)
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"TensorFlow: {tf.__version__}")
    logger.info(f"Sistema Operativo: {platform.system()} {platform.release()}")
    logger.info(f"Procesador: {platform.processor()}")
    
    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPUs disponibles: {len(gpus)}")
        for gpu in gpus:
            logger.info(f"  - {gpu.name}")
    else:
        logger.info("No se detectaron GPUs - usando CPU")
    
    logger.info("=" * 60)


def log_experiment_config(logger: logging.Logger, config: dict):
    """
    Registra la configuración del experimento.
    
    Args:
        logger: Logger a usar
        config: Diccionario de configuración
    """
    logger.info("=" * 60)
    logger.info("CONFIGURACIÓN DEL EXPERIMENTO")
    logger.info("=" * 60)
    
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    logger.info("=" * 60)


def log_training_progress(logger: logging.Logger,
                         round_num: int,
                         metrics: dict):
    """
    Registra progreso del entrenamiento.
    
    Args:
        logger: Logger a usar
        round_num: Número de ronda
        metrics: Métricas de la ronda
    """
    logger.info(f"Ronda {round_num} completada:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")


class TensorBoardLogger:
    """
    Logger para TensorBoard.
    """
    
    def __init__(self, log_dir: str = None):
        """
        Inicializa el logger de TensorBoard.
        
        Args:
            log_dir (str): Directorio de logs
        """
        if log_dir is None:
            log_dir = LOGGING_CONFIG['tensorboard_dir']
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear subdirectorio con timestamp
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = self.log_dir / timestamp
        
        # File writers para diferentes secciones
        import tensorflow as tf
        self.train_writer = tf.summary.create_file_writer(str(self.run_dir / 'train'))
        self.val_writer = tf.summary.create_file_writer(str(self.run_dir / 'val'))
        self.test_writer = tf.summary.create_file_writer(str(self.run_dir / 'test'))
    
    def log_scalar(self, tag: str, value: float, step: int, mode: str = 'train'):
        """
        Registra un valor escalar.
        
        Args:
            tag (str): Nombre de la métrica
            value (float): Valor
            step (int): Paso/época
            mode (str): 'train', 'val' o 'test'
        """
        import tensorflow as tf
        
        writer = {
            'train': self.train_writer,
            'val': self.val_writer,
            'test': self.test_writer
        }.get(mode, self.train_writer)
        
        with writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            writer.flush()
    
    def log_metrics(self, metrics: dict, step: int, mode: str = 'train'):
        """
        Registra múltiples métricas.
        
        Args:
            metrics (dict): Diccionario de métricas
            step (int): Paso/época
            mode (str): 'train', 'val' o 'test'
        """
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.log_scalar(metric_name, metric_value, step, mode)
    
    def log_image(self, tag: str, image, step: int):
        """
        Registra una imagen.
        
        Args:
            tag (str): Nombre de la imagen
            image: Imagen (array numpy)
            step (int): Paso
        """
        import tensorflow as tf
        
        with self.train_writer.as_default():
            tf.summary.image(tag, image, step=step)
            self.train_writer.flush()
    
    def log_histogram(self, tag: str, values, step: int):
        """
        Registra un histograma.
        
        Args:
            tag (str): Nombre
            values: Valores (array)
            step (int): Paso
        """
        import tensorflow as tf
        
        with self.train_writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.train_writer.flush()
    
    def close(self):
        """Cierra los file writers."""
        self.train_writer.close()
        self.val_writer.close()
        self.test_writer.close()


def create_progress_bar(total: int, desc: str = "Progreso"):
    """
    Crea una barra de progreso.
    
    Args:
        total (int): Total de iteraciones
        desc (str): Descripción
    
    Returns:
        tqdm: Barra de progreso
    """
    from tqdm import tqdm
    return tqdm(total=total, desc=desc, unit='step')


# ==================== FUNCIONES DE UTILIDAD ====================

def get_log_file_path(name: str) -> Path:
    """
    Obtiene la ruta del archivo de log.
    
    Args:
        name (str): Nombre del log
    
    Returns:
        Path: Ruta del archivo
    """
    log_dir = Path(LOGGING_CONFIG['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return log_dir / f"{name}_{timestamp}.log"


def setup_federated_logging(experiment_name: str = "federated_experiment"):
    """
    Configura el sistema de logging completo para FL.
    
    Args:
        experiment_name (str): Nombre del experimento
    
    Returns:
        tuple: (logger, tensorboard_logger)
    """
    # Logger principal
    logger = create_experiment_logger(experiment_name)
    
    # Logger de TensorBoard
    tb_logger = TensorBoardLogger()
    
    # Registrar información del sistema
    log_system_info(logger)
    
    logger.info(f"Sistema de logging inicializado - Experimento: {experiment_name}")
    logger.info(f"TensorBoard: {tb_logger.run_dir}")
    
    return logger, tb_logger


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Probando sistema de logging...")
    
    # Crear logger de prueba
    logger = setup_logger('TestLogger')
    logger.info("Esto es un mensaje INFO")
    logger.debug("Esto es un mensaje DEBUG")
    logger.warning("Esto es un WARNING")
    
    # Probar TensorBoard logger
    tb_logger = TensorBoardLogger()
    tb_logger.log_scalar('test_metric', 0.95, step=1)
    tb_logger.log_metrics({'accuracy': 0.95, 'loss': 0.05}, step=1)
    tb_logger.close()
    
    print(f"\nLogs guardados en: {LOGGING_CONFIG['log_dir']}")
    print(f"TensorBoard: {tb_logger.run_dir}")
