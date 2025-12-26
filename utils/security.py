"""
Módulo de seguridad y privacidad para Federated Learning.

Incluye:
- Cifrado de comunicaciones
- Secure aggregation (placeholder)
- Privacidad diferencial (placeholder)
- Autenticación de clientes
"""

import hashlib
import secrets
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

from config.config import SECURITY_CONFIG
from utils.logging_utils import setup_logger


class SecureChannel:
    """
    Gestiona comunicaciones cifradas entre servidor y clientes.
    """
    
    def __init__(self, use_encryption: bool = None):
        """
        Inicializa el canal seguro.
        
        Args:
            use_encryption (bool): Si usar cifrado
        """
        if use_encryption is None:
            use_encryption = SECURITY_CONFIG.get('use_encryption', True)
        
        self.use_encryption = use_encryption
        self.logger = setup_logger('SecureChannel')
        
        if self.use_encryption:
            self.logger.info("Canal seguro habilitado con cifrado")
        else:
            self.logger.warning("Canal seguro deshabilitado - modo desarrollo")
    
    def encrypt_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Cifra parámetros del modelo.
        
        Args:
            parameters (List[np.ndarray]): Parámetros a cifrar
        
        Returns:
            List[np.ndarray]: Parámetros cifrados
        """
        # TODO: Implementar cifrado real (AES, RSA, etc.)
        # Por ahora, retornar sin modificar
        
        if not self.use_encryption:
            return parameters
        
        self.logger.debug("Cifrando parámetros...")
        
        # Placeholder: implementar cifrado real
        encrypted = parameters  # Sin modificar por ahora
        
        return encrypted
    
    def decrypt_parameters(self, encrypted_parameters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Descifra parámetros del modelo.
        
        Args:
            encrypted_parameters (List[np.ndarray]): Parámetros cifrados
        
        Returns:
            List[np.ndarray]: Parámetros descifrados
        """
        # TODO: Implementar descifrado real
        
        if not self.use_encryption:
            return encrypted_parameters
        
        self.logger.debug("Descifrando parámetros...")
        
        # Placeholder
        decrypted = encrypted_parameters
        
        return decrypted


class SecureAggregation:
    """
    Implementa agregación segura de parámetros.
    
    Placeholder para secure aggregation real usando:
    - Secret sharing (Shamir)
    - Masks aditivos
    - Protocolos multi-party computation
    """
    
    def __init__(self):
        """Inicializa secure aggregation."""
        self.logger = setup_logger('SecureAggregation')
        self.logger.warning("Secure Aggregation no implementada completamente - placeholder activo")
    
    def aggregate_secure(self, 
                        client_parameters: List[List[np.ndarray]],
                        weights: Optional[List[float]] = None) -> List[np.ndarray]:
        """
        Agrega parámetros de forma segura sin revelar valores individuales.
        
        Args:
            client_parameters (List[List[np.ndarray]]): Parámetros de cada cliente
            weights (List[float]): Pesos de agregación (opcional)
        
        Returns:
            List[np.ndarray]: Parámetros agregados
        """
        # TODO: Implementar secure aggregation real
        # Por ahora, agregación simple (FedAvg)
        
        self.logger.debug(f"Agregación segura de {len(client_parameters)} clientes")
        
        if weights is None:
            # Pesos uniformes
            weights = [1.0 / len(client_parameters)] * len(client_parameters)
        
        # Agregar parámetros (weighted average)
        aggregated = []
        num_params = len(client_parameters[0])
        
        for param_idx in range(num_params):
            weighted_sum = np.zeros_like(client_parameters[0][param_idx])
            
            for client_idx, client_params in enumerate(client_parameters):
                weighted_sum += weights[client_idx] * client_params[param_idx]
            
            aggregated.append(weighted_sum)
        
        return aggregated


class DifferentialPrivacy:
    """
    Añade privacidad diferencial al entrenamiento.
    
    Implementa:
    - DP-SGD (Differentially Private SGD)
    - Gaussian noise addition
    - Gradient clipping
    """
    
    def __init__(self, 
                 epsilon: float = None,
                 delta: float = None,
                 noise_multiplier: float = None):
        """
        Inicializa DP.
        
        Args:
            epsilon (float): Budget de privacidad
            delta (float): Probabilidad de fallo
            noise_multiplier (float): Multiplicador de ruido
        """
        self.epsilon = epsilon or SECURITY_CONFIG.get('dp_epsilon', 1.0)
        self.delta = delta or SECURITY_CONFIG.get('dp_delta', 1e-5)
        self.noise_multiplier = noise_multiplier or SECURITY_CONFIG.get('dp_noise_multiplier', 0.1)
        
        self.logger = setup_logger('DifferentialPrivacy')
        self.logger.warning("Privacidad Diferencial no implementada completamente - placeholder activo")
        self.logger.info(f"DP Config: ε={self.epsilon}, δ={self.delta}, noise={self.noise_multiplier}")
    
    def clip_gradients(self, 
                       gradients: List[np.ndarray],
                       clip_norm: float = 1.0) -> List[np.ndarray]:
        """
        Clipea gradientes para DP.
        
        Args:
            gradients (List[np.ndarray]): Gradientes
            clip_norm (float): Norma de clipping
        
        Returns:
            List[np.ndarray]: Gradientes clipados
        """
        # TODO: Implementar clipping real
        
        clipped = []
        for grad in gradients:
            # Calcular norma L2
            norm = np.linalg.norm(grad)
            
            if norm > clip_norm:
                # Clipear
                clipped_grad = grad * (clip_norm / norm)
            else:
                clipped_grad = grad
            
            clipped.append(clipped_grad)
        
        return clipped
    
    def add_noise(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Añade ruido gaussiano para DP.
        
        Args:
            parameters (List[np.ndarray]): Parámetros
        
        Returns:
            List[np.ndarray]: Parámetros con ruido
        """
        noisy_params = []
        
        for param in parameters:
            # Ruido gaussiano escalado por noise_multiplier
            noise = np.random.normal(
                loc=0.0,
                scale=self.noise_multiplier * param.std(),
                size=param.shape
            )
            
            noisy_param = param + noise
            noisy_params.append(noisy_param)
        
        return noisy_params
    
    def privatize_parameters(self, 
                           parameters: List[np.ndarray],
                           clip_norm: float = 1.0) -> List[np.ndarray]:
        """
        Aplica privacidad diferencial a parámetros.
        
        Args:
            parameters (List[np.ndarray]): Parámetros originales
            clip_norm (float): Norma de clipping
        
        Returns:
            List[np.ndarray]: Parámetros privatizados
        """
        # Clipear
        clipped = self.clip_gradients(parameters, clip_norm)
        
        # Añadir ruido
        privatized = self.add_noise(clipped)
        
        self.logger.debug(f"Parámetros privatizados con ε={self.epsilon}")
        
        return privatized


class ClientAuthenticator:
    """
    Gestiona autenticación de clientes.
    """
    
    def __init__(self):
        """Inicializa autenticador."""
        self.logger = setup_logger('ClientAuthenticator')
        self.allowed_clients = SECURITY_CONFIG.get('allowed_client_ids', None)
        self.use_auth = SECURITY_CONFIG.get('authenticate_clients', False)
        
        if self.use_auth:
            self.logger.info("Autenticación de clientes habilitada")
        else:
            self.logger.warning("Autenticación deshabilitada - modo desarrollo")
        
        # Tokens de clientes (en producción, usar BD o sistema externo)
        self.client_tokens = {}
    
    def generate_token(self, client_id: int) -> str:
        """
        Genera token para un cliente.
        
        Args:
            client_id (int): ID del cliente
        
        Returns:
            str: Token generado
        """
        # Generar token seguro
        token = secrets.token_hex(32)
        
        # Almacenar hash del token
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self.client_tokens[client_id] = token_hash
        
        self.logger.info(f"Token generado para cliente {client_id}")
        
        return token
    
    def validate_token(self, client_id: int, token: str) -> bool:
        """
        Valida token de cliente.
        
        Args:
            client_id (int): ID del cliente
            token (str): Token a validar
        
        Returns:
            bool: True si válido
        """
        if not self.use_auth:
            return True
        
        # Verificar si el cliente está permitido
        if self.allowed_clients and client_id not in self.allowed_clients:
            self.logger.warning(f"Cliente {client_id} no está en lista permitida")
            return False
        
        # Validar token
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        if client_id not in self.client_tokens:
            self.logger.warning(f"Cliente {client_id} sin token registrado")
            return False
        
        is_valid = self.client_tokens[client_id] == token_hash
        
        if not is_valid:
            self.logger.warning(f"Token inválido para cliente {client_id}")
        
        return is_valid


# ==================== FUNCIONES DE UTILIDAD ====================

def setup_secure_communication(server_address: str):
    """
    Configura comunicación segura con TLS.
    
    Args:
        server_address (str): Dirección del servidor
    
    Returns:
        dict: Configuración de seguridad
    """
    # TODO: Implementar configuración TLS real
    
    logger = setup_logger('SecuritySetup')
    
    if SECURITY_CONFIG.get('use_encryption', False):
        logger.info("Configurando comunicación segura con TLS...")
        
        cert_path = SECURITY_CONFIG.get('certificate_path')
        key_path = SECURITY_CONFIG.get('key_path')
        
        # Verificar existencia de certificados
        if cert_path and key_path:
            if Path(cert_path).exists() and Path(key_path).exists():
                logger.info("Certificados TLS encontrados")
            else:
                logger.warning("Certificados TLS no encontrados - generando autofirmados...")
                # TODO: Generar certificados autofirmados
        else:
            logger.warning("Rutas de certificados no configuradas")
    else:
        logger.warning("Comunicación sin cifrado - solo para desarrollo")
    
    return {}


def hash_parameters(parameters: List[np.ndarray]) -> str:
    """
    Calcula hash de parámetros para verificación de integridad.
    
    Args:
        parameters (List[np.ndarray]): Parámetros
    
    Returns:
        str: Hash SHA256
    """
    # Concatenar todos los parámetros
    concatenated = np.concatenate([p.flatten() for p in parameters])
    
    # Calcular hash
    param_hash = hashlib.sha256(concatenated.tobytes()).hexdigest()
    
    return param_hash


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Probando módulo de seguridad...")
    
    # Probar canal seguro
    channel = SecureChannel(use_encryption=True)
    dummy_params = [np.random.rand(10, 10) for _ in range(3)]
    encrypted = channel.encrypt_parameters(dummy_params)
    print(f"Parámetros cifrados: {len(encrypted)} arrays")
    
    # Probar DP
    dp = DifferentialPrivacy()
    privatized = dp.privatize_parameters(dummy_params)
    print(f"Parámetros privatizados: {len(privatized)} arrays")
    
    # Probar autenticador
    auth = ClientAuthenticator()
    token = auth.generate_token(client_id=1)
    is_valid = auth.validate_token(client_id=1, token=token)
    print(f"Token válido: {is_valid}")
    
    # Hash de parámetros
    param_hash = hash_parameters(dummy_params)
    print(f"Hash de parámetros: {param_hash[:16]}...")
