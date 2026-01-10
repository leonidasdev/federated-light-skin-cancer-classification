"""
Logging system for Federated Learning.

Provides unified logging for server and clients.
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
    Configure a logger for the system.

    Args:
        name (str): Logger name
        log_file (str): Log file path (optional)
        level (str): Logging level

    Returns:
        logging.Logger: Configured logger
    """
    # Logging level
    if level is None:
        level = LOGGING_CONFIG.get('log_level', 'INFO')
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        fmt=LOGGING_CONFIG.get('log_format'),
        datefmt=LOGGING_CONFIG.get('date_format')
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
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
    Create an experiment-specific logger.

    Args:
        experiment_name (str): Experiment name

    Returns:
        logging.Logger: Experiment logger
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
    Log system information.

    Args:
        logger: Logger to use
    """
    import platform
    import tensorflow as tf
    
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"TensorFlow: {tf.__version__}")
    logger.info(f"Operating System: {platform.system()} {platform.release()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPUs available: {len(gpus)}")
        for gpu in gpus:
            logger.info(f"  - {gpu.name}")
    else:
        logger.info("No GPUs detected - using CPU")
    
    logger.info("=" * 60)


def log_experiment_config(logger: logging.Logger, config: dict):
    """
    Log the experiment configuration.

    Args:
        logger: Logger to use
        config: Configuration dictionary
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 60)
    
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    logger.info("=" * 60)


def log_training_progress(logger: logging.Logger,
                         round_num: int,
                         metrics: dict):
    """
    Log training progress.

    Args:
        logger: Logger to use
        round_num: Round number
        metrics: Round metrics
    """
    logger.info(f"Round {round_num} completed:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")


class TensorBoardLogger:
    """
    Logger for TensorBoard.
    """
    
    def __init__(self, log_dir: str = None):
        """
        Initialize the TensorBoard logger.

        Args:
            log_dir (str): Logs directory
        """
        if log_dir is None:
            log_dir = LOGGING_CONFIG['tensorboard_dir']
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = self.log_dir / timestamp
        
        # File writers for different sections
        import tensorflow as tf
        self.train_writer = tf.summary.create_file_writer(str(self.run_dir / 'train'))
        self.val_writer = tf.summary.create_file_writer(str(self.run_dir / 'val'))
        self.test_writer = tf.summary.create_file_writer(str(self.run_dir / 'test'))
    
    def log_scalar(self, tag: str, value: float, step: int, mode: str = 'train'):
        """
        Log a scalar value.

        Args:
            tag (str): Metric name
            value (float): Value
            step (int): Step/epoch
            mode (str): 'train', 'val' or 'test'
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
        Log multiple metrics.

        Args:
            metrics (dict): Metrics dictionary
            step (int): Step/epoch
            mode (str): 'train', 'val' or 'test'
        """
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.log_scalar(metric_name, metric_value, step, mode)
    
    def log_image(self, tag: str, image, step: int):
        """
        Log an image.

        Args:
            tag (str): Image name
            image: Image (numpy array)
            step (int): Step
        """
        import tensorflow as tf
        
        with self.train_writer.as_default():
            tf.summary.image(tag, image, step=step)
            self.train_writer.flush()
    
    def log_histogram(self, tag: str, values, step: int):
        """
        Log a histogram.

        Args:
            tag (str): Name
            values: Values (array)
            step (int): Step
        """
        import tensorflow as tf
        
        with self.train_writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.train_writer.flush()
    
    def close(self):
        """Close the file writers."""
        self.train_writer.close()
        self.val_writer.close()
        self.test_writer.close()


def create_progress_bar(total: int, desc: str = "Progreso"):
    """
    Create a progress bar.

    Args:
        total (int): Total iterations
        desc (str): Description

    Returns:
        tqdm: Progress bar
    """
    from tqdm import tqdm
    return tqdm(total=total, desc=desc, unit='step')


# ==================== FUNCIONES DE UTILIDAD ====================

def get_log_file_path(name: str) -> Path:
    """
    Get the log file path.

    Args:
        name (str): Log name

    Returns:
        Path: Path to the log file
    """
    log_dir = Path(LOGGING_CONFIG['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return log_dir / f"{name}_{timestamp}.log"


def setup_federated_logging(experiment_name: str = "federated_experiment"):
    """
    Setup the full logging system for FL.

    Args:
        experiment_name (str): Experiment name

    Returns:
        tuple: (logger, tensorboard_logger)
    """
    # Logger principal
    logger = create_experiment_logger(experiment_name)
    
    # Logger de TensorBoard
    tb_logger = TensorBoardLogger()
    
    # Log system information
    log_system_info(logger)

    logger.info(f"Logging system initialized - Experiment: {experiment_name}")
    logger.info(f"TensorBoard: {tb_logger.run_dir}")
    
    return logger, tb_logger


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Testing logging system...")

    # Create test logger
    logger = setup_logger('TestLogger')
    logger.info("This is an INFO message")
    logger.debug("This is a DEBUG message")
    logger.warning("This is a WARNING")

    # Test TensorBoard logger
    tb_logger = TensorBoardLogger()
    tb_logger.log_scalar('test_metric', 0.95, step=1)
    tb_logger.log_metrics({'accuracy': 0.95, 'loss': 0.05}, step=1)
    tb_logger.close()

    print(f"\nLogs saved to: {LOGGING_CONFIG['log_dir']}")
    print(f"TensorBoard: {tb_logger.run_dir}")
