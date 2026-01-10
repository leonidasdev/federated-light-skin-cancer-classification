"""
Security and privacy module for Federated Learning.

Includes:
- Communication encryption
- Secure aggregation (placeholder)
- Differential privacy (placeholder)
- Client authentication
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
    Manage encrypted communications between server and clients.
    """
    
    def __init__(self, use_encryption: bool = None):
        """
        Initialize the secure channel.

        Args:
            use_encryption (bool): Whether to use encryption
        """
        if use_encryption is None:
            use_encryption = SECURITY_CONFIG.get('use_encryption', True)
        
        self.use_encryption = use_encryption
        self.logger = setup_logger('SecureChannel')
        
        if self.use_encryption:
            self.logger.info("Secure channel enabled with encryption")
        else:
            self.logger.warning("Secure channel disabled - development mode")
    
    def encrypt_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Encrypt model parameters.

        Args:
            parameters (List[np.ndarray]): Parameters to encrypt

        Returns:
            List[np.ndarray]: Encrypted parameters
        """
        # TODO: Implement real encryption (AES, RSA, etc.)
        # For now, return unmodified
        
        if not self.use_encryption:
            return parameters

        self.logger.debug("Encrypting parameters...")

        # Placeholder: implement real encryption
        encrypted = parameters  # Unmodified for now
        
        return encrypted
    
    def decrypt_parameters(self, encrypted_parameters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Decrypt model parameters.

        Args:
            encrypted_parameters (List[np.ndarray]): Encrypted parameters

        Returns:
            List[np.ndarray]: Decrypted parameters
        """
        # TODO: Implementar descifrado real
        
        if not self.use_encryption:
            return encrypted_parameters

        self.logger.debug("Decrypting parameters...")

        # Placeholder
        decrypted = encrypted_parameters
        
        return decrypted


class SecureAggregation:
    """
    Implement secure aggregation of parameters.

    Placeholder for real secure aggregation using:
    - Secret sharing (Shamir)
    - Additive masks
    - Multi-party computation protocols
    """
    
    def __init__(self):
        """Inicializa secure aggregation."""
        self.logger = setup_logger('SecureAggregation')
        self.logger.warning("Secure Aggregation not fully implemented - placeholder active")
    
    def aggregate_secure(self, 
                        client_parameters: List[List[np.ndarray]],
                        weights: Optional[List[float]] = None) -> List[np.ndarray]:
        """
        Securely aggregate parameters without revealing individual values.

        Args:
            client_parameters (List[List[np.ndarray]]): Parameters from each client
            weights (List[float]): Aggregation weights (optional)

        Returns:
            List[np.ndarray]: Aggregated parameters
        """
        # TODO: Implement real secure aggregation
        # For now, simple aggregation (FedAvg)

        self.logger.debug(f"Securely aggregating {len(client_parameters)} clients")
        
        if weights is None:
            # Uniform weights
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
    Add differential privacy to training.

    Implements:
    - DP-SGD (Differentially Private SGD)
    - Gaussian noise addition
    - Gradient clipping
    """
    
    def __init__(self, 
                 epsilon: float = None,
                 delta: float = None,
                 noise_multiplier: float = None):
        """
        Initialize DP.

        Args:
            epsilon (float): Privacy budget
            delta (float): Failure probability
            noise_multiplier (float): Noise multiplier
        """
        self.epsilon = epsilon or SECURITY_CONFIG.get('dp_epsilon', 1.0)
        self.delta = delta or SECURITY_CONFIG.get('dp_delta', 1e-5)
        self.noise_multiplier = noise_multiplier or SECURITY_CONFIG.get('dp_noise_multiplier', 0.1)
        
        self.logger = setup_logger('DifferentialPrivacy')
        self.logger.warning("Differential Privacy not fully implemented - placeholder active")
        self.logger.info(f"DP Config: ε={self.epsilon}, δ={self.delta}, noise={self.noise_multiplier}")
    
    def clip_gradients(self, 
                       gradients: List[np.ndarray],
                       clip_norm: float = 1.0) -> List[np.ndarray]:
        """
        Clip gradients for DP.

        Args:
            gradients (List[np.ndarray]): Gradients
            clip_norm (float): Clipping norm

        Returns:
            List[np.ndarray]: Clipped gradients
        """
        # TODO: Implementar clipping real
        
        clipped = []
        for grad in gradients:
            # Compute L2 norm
            norm = np.linalg.norm(grad)

            if norm > clip_norm:
                # Clip
                clipped_grad = grad * (clip_norm / norm)
            else:
                clipped_grad = grad
            
            clipped.append(clipped_grad)
        
        return clipped
    
    def add_noise(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add Gaussian noise for DP.

        Args:
            parameters (List[np.ndarray]): Parameters

        Returns:
            List[np.ndarray]: Parameters with noise
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
        Apply differential privacy to parameters.

        Args:
            parameters (List[np.ndarray]): Original parameters
            clip_norm (float): Clipping norm

        Returns:
            List[np.ndarray]: Privatized parameters
        """
        # Clip
        clipped = self.clip_gradients(parameters, clip_norm)

        # Add noise
        privatized = self.add_noise(clipped)

        self.logger.debug(f"Parameters privatized with ε={self.epsilon}")

        return privatized


class ClientAuthenticator:
    """
    Manage client authentication.
    """
    
    def __init__(self):
        """Initialize authenticator."""
        self.logger = setup_logger('ClientAuthenticator')
        self.allowed_clients = SECURITY_CONFIG.get('allowed_client_ids', None)
        self.use_auth = SECURITY_CONFIG.get('authenticate_clients', False)

        if self.use_auth:
            self.logger.info("Client authentication enabled")
        else:
            self.logger.warning("Authentication disabled - development mode")

        # Client tokens (in production, use DB or external system)
        self.client_tokens = {}
    
    def generate_token(self, client_id: int) -> str:
        """
        Generate a token for a client.

        Args:
            client_id (int): Client ID

        Returns:
            str: Generated token
        """
        # Generate secure token
        token = secrets.token_hex(32)

        # Store hash of token
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self.client_tokens[client_id] = token_hash

        self.logger.info(f"Token generated for client {client_id}")

        return token
    
    def validate_token(self, client_id: int, token: str) -> bool:
        """
        Validate a client's token.

        Args:
            client_id (int): Client ID
            token (str): Token to validate

        Returns:
            bool: True if valid
        """
        if not self.use_auth:
            return True

        # Check if client is allowed
        if self.allowed_clients and client_id not in self.allowed_clients:
            self.logger.warning(f"Client {client_id} not in allowed list")
            return False

        # Validate token
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        if client_id not in self.client_tokens:
            self.logger.warning(f"Client {client_id} has no registered token")
            return False

        is_valid = self.client_tokens[client_id] == token_hash

        if not is_valid:
            self.logger.warning(f"Invalid token for client {client_id}")

        return is_valid


# ==================== FUNCIONES DE UTILIDAD ====================

def setup_secure_communication(server_address: str):
    """
    Configure secure communication with TLS.

    Args:
        server_address (str): Server address

    Returns:
        dict: Security configuration
    """
    # TODO: Implement real TLS configuration

    logger = setup_logger('SecuritySetup')

    if SECURITY_CONFIG.get('use_encryption', False):
        logger.info("Configuring secure communication with TLS...")

        cert_path = SECURITY_CONFIG.get('certificate_path')
        key_path = SECURITY_CONFIG.get('key_path')

        # Check certificates existence
        if cert_path and key_path:
            if Path(cert_path).exists() and Path(key_path).exists():
                logger.info("TLS certificates found")
            else:
                logger.warning("TLS certificates not found - generating self-signed...")
                # TODO: Generate self-signed certificates
        else:
            logger.warning("Certificate paths not configured")
    else:
        logger.warning("Unencrypted communication - development only")

    return {}


def hash_parameters(parameters: List[np.ndarray]) -> str:
    """
    Compute a hash of parameters for integrity verification.

    Args:
        parameters (List[np.ndarray]): Parameters

    Returns:
        str: SHA256 hash
    """
    # Concatenate all parameters
    concatenated = np.concatenate([p.flatten() for p in parameters])

    # Compute hash
    param_hash = hashlib.sha256(concatenated.tobytes()).hexdigest()

    return param_hash


# ==================== TESTING ====================

if __name__ == '__main__':
    print("Testing security module...")

    # Test secure channel
    channel = SecureChannel(use_encryption=True)
    dummy_params = [np.random.rand(10, 10) for _ in range(3)]
    encrypted = channel.encrypt_parameters(dummy_params)
    print(f"Encrypted parameters: {len(encrypted)} arrays")

    # Test DP
    dp = DifferentialPrivacy()
    privatized = dp.privatize_parameters(dummy_params)
    print(f"Privatized parameters: {len(privatized)} arrays")

    # Test authenticator
    auth = ClientAuthenticator()
    token = auth.generate_token(client_id=1)
    is_valid = auth.validate_token(client_id=1, token=token)
    print(f"Token válido: {is_valid}")
    
    # Hash de parámetros
    param_hash = hash_parameters(dummy_params)
    print(f"Hash de parámetros: {param_hash[:16]}...")
