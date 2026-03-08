# =============================================================================
# Configuration Schema Validation
# =============================================================================
"""
Pydantic-based schema validation for YAML configuration files.

This module defines the schema for all configuration files used in the project:
- experiment_config.yaml: Master experiment configuration
- fl_config.yaml: Federated learning settings
- model_config.yaml: DSCATNet model architecture

Usage:
    from src.utils.config_schema import validate_config, ConfigType

    # Validate a specific config type
    config = validate_config("configs/fl_config.yaml", ConfigType.FEDERATED)

    # Auto-detect config type
    config = validate_config("configs/experiment_config.yaml")
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Enums for Valid Options
# =============================================================================


class ConfigType(str, Enum):
    """Types of configuration files."""

    EXPERIMENT = "experiment"
    FEDERATED = "federated"
    MODEL = "model"


class DeviceType(str, Enum):
    """Valid device types."""

    CUDA = "cuda"
    CPU = "cpu"


class NormalizationType(str, Enum):
    """Valid normalization types."""

    IMAGENET = "imagenet"
    DERMOSCOPY = "dermoscopy"


class AugmentationLevel(str, Enum):
    """Valid augmentation levels."""

    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class ClassificationMode(str, Enum):
    """Valid classification modes."""

    MULTICLASS = "multiclass"
    MULTICLASS_8 = "multiclass_8"
    BINARY = "binary"


class FLFramework(str, Enum):
    """Valid FL frameworks."""

    FLOWER = "flower"


class FLStrategy(str, Enum):
    """Valid FL strategies."""

    FEDAVG = "FedAvg"
    FEDPROX = "FedProx"
    FEDNOVA = "FedNova"


class OptimizerType(str, Enum):
    """Valid optimizer types."""

    ADAM = "Adam"
    ADAMW = "AdamW"
    SGD = "SGD"


class LRSchedulerType(str, Enum):
    """Valid learning rate scheduler types."""

    COSINE = "cosine"
    STEP = "step"
    PLATEAU = "plateau"
    NONE = "none"


class FusionMethod(str, Enum):
    """Valid feature fusion methods."""

    CONCAT = "concat"
    ADD = "add"
    ATTENTION = "attention"


class NonIIDType(str, Enum):
    """Valid non-IID distribution types."""

    NATURAL = "natural"
    DIRICHLET = "dirichlet"
    LABEL_SKEW = "label_skew"
    QUANTITY_SKEW = "quantity_skew"


# =============================================================================
# Sub-Models for Nested Configuration
# =============================================================================


class HardwareConfig(BaseModel):
    """Hardware configuration settings."""

    device: DeviceType = DeviceType.CUDA
    num_workers: int = Field(default=2, ge=0, le=16)
    pin_memory: bool = True
    mixed_precision: bool = True


class DataConfig(BaseModel):
    """Data configuration settings."""

    root_dir: str = "./data"
    img_size: int = Field(default=224, ge=32, le=512)
    val_split: float = Field(default=0.15, ge=0.0, le=0.5)
    test_split: float = Field(default=0.15, ge=0.0, le=0.5)
    normalization: NormalizationType = NormalizationType.IMAGENET
    augmentation_level: AugmentationLevel = AugmentationLevel.MEDIUM
    classification_mode: ClassificationMode = ClassificationMode.MULTICLASS
    num_classes: int = Field(default=7, ge=2, le=10)
    filter_unknown: bool = True
    use_weighted_sampling: bool = False
    use_class_weights: bool = True

    @model_validator(mode="after")
    def validate_splits(self) -> DataConfig:
        """Ensure total splits don't exceed 1.0."""
        if self.val_split + self.test_split >= 1.0:
            raise ValueError(
                f"val_split ({self.val_split}) + test_split ({self.test_split}) "
                "must be less than 1.0"
            )
        return self


class CentralizedConfig(BaseModel):
    """Centralized training configuration."""

    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=8, ge=1, le=256)
    learning_rate: float = Field(default=0.001, gt=0, le=1.0)
    weight_decay: float = Field(default=0.0001, ge=0, le=1.0)
    early_stopping_patience: int = Field(default=15, ge=1, le=100)
    pooled_data: bool = True


class FederatedExperiment(BaseModel):
    """Single federated experiment configuration."""

    name: str = Field(..., min_length=1)
    description: str = ""
    num_rounds: int = Field(default=100, ge=1, le=1000)
    local_epochs: int = Field(default=3, ge=1, le=50)
    batch_size: int = Field(default=8, ge=1, le=256)
    noniid_type: NonIIDType = NonIIDType.NATURAL
    dirichlet_alpha: float | None = Field(default=None, gt=0)


class MetricsConfig(BaseModel):
    """Metrics configuration."""

    classification: list[str] = Field(
        default=[
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
            "confusion_matrix",
        ]
    )
    federated: list[str] = Field(
        default=[
            "convergence_rounds",
            "communication_cost",
            "client_drift",
            "round_time",
        ]
    )
    per_class: list[str] = Field(default=["sensitivity", "specificity"])


class WandBConfig(BaseModel):
    """Weights & Biases configuration."""

    enabled: bool = False
    project: str = "dscatnet-fl"
    entity: str | None = None


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    tensorboard: bool = True
    wandb: WandBConfig = Field(default_factory=WandBConfig)


class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration."""

    deterministic: bool = True
    benchmark: bool = False
    seed: int = Field(default=42, ge=0)


class ExperimentInfo(BaseModel):
    """Experiment metadata."""

    name: str = Field(default="DSCATNet-FL-SkinCancer", min_length=1)
    description: str = ""
    seed: int = Field(default=42, ge=0)


# =============================================================================
# FL Config Sub-Models
# =============================================================================


class ClientConfig(BaseModel):
    """Individual client configuration."""

    id: int = Field(..., ge=1)
    dataset: str = Field(..., min_length=1)
    description: str = ""


class ScenarioConfig(BaseModel):
    """Non-IID scenario configuration."""

    description: str = ""
    noniid_type: NonIIDType = NonIIDType.NATURAL
    dirichlet_alpha: float | None = Field(default=None, gt=0)


class StrategyConfig(BaseModel):
    """FL strategy configuration."""

    name: FLStrategy
    description: str = ""
    mu: float | None = Field(default=None, ge=0)  # For FedProx


class FederatedSettings(BaseModel):
    """Core federated learning settings."""

    framework: FLFramework = FLFramework.FLOWER
    strategy: FLStrategy = FLStrategy.FEDAVG
    num_clients: int = Field(default=4, ge=1, le=100)
    clients: list[ClientConfig] | None = None
    num_rounds: int = Field(default=100, ge=1, le=1000)
    early_stopping_patience: int = Field(default=20, ge=1, le=100)
    fraction_fit: float = Field(default=1.0, gt=0, le=1.0)
    fraction_evaluate: float = Field(default=1.0, gt=0, le=1.0)
    min_fit_clients: int = Field(default=4, ge=1)
    min_evaluate_clients: int = Field(default=4, ge=1)
    min_available_clients: int = Field(default=4, ge=1)
    local_epochs: int = Field(default=3, ge=1, le=50)
    local_batch_size: int = Field(default=8, ge=1, le=256)
    train_val_split: float = Field(default=0.15, ge=0, le=0.5)
    optimizer: OptimizerType = OptimizerType.ADAM
    learning_rate: float = Field(default=0.001, gt=0, le=1.0)
    weight_decay: float = Field(default=0.0001, ge=0, le=1.0)
    lr_scheduler: LRSchedulerType = LRSchedulerType.COSINE
    warmup_epochs: int = Field(default=5, ge=0, le=50)
    min_lr: float = Field(default=0.000001, ge=0, le=1.0)
    save_every_rounds: int = Field(default=10, ge=1, le=100)
    evaluate_every_rounds: int = Field(default=1, ge=1, le=100)
    server_address: str = "[::]:8080"

    @model_validator(mode="after")
    def validate_client_counts(self) -> FederatedSettings:
        """Ensure client counts are consistent."""
        if self.min_fit_clients > self.num_clients:
            raise ValueError(
                f"min_fit_clients ({self.min_fit_clients}) cannot exceed "
                f"num_clients ({self.num_clients})"
            )
        if self.min_evaluate_clients > self.num_clients:
            raise ValueError(
                f"min_evaluate_clients ({self.min_evaluate_clients}) cannot exceed "
                f"num_clients ({self.num_clients})"
            )
        return self


# =============================================================================
# Model Config Sub-Models
# =============================================================================


class ModelSettings(BaseModel):
    """Core model architecture settings."""

    name: str = "DSCATNet"
    img_size: int = Field(default=224, ge=32, le=512)
    in_channels: int = Field(default=3, ge=1, le=4)
    num_classes: int = Field(default=7, ge=2, le=10)
    embed_dim: int = Field(default=384, ge=64, le=1024)
    depth: int = Field(default=6, ge=1, le=24)
    num_heads: int = Field(default=6, ge=1, le=16)
    mlp_ratio: float = Field(default=4.0, ge=1.0, le=8.0)
    fine_patch_size: int = Field(default=8, ge=4, le=32)
    coarse_patch_size: int = Field(default=16, ge=8, le=64)
    drop_rate: float = Field(default=0.1, ge=0.0, le=0.5)
    attn_drop_rate: float = Field(default=0.0, ge=0.0, le=0.5)
    fusion_method: FusionMethod = FusionMethod.CONCAT

    @model_validator(mode="after")
    def validate_patch_sizes(self) -> ModelSettings:
        """Ensure patch sizes divide image size evenly."""
        if self.img_size % self.fine_patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by "
                f"fine_patch_size ({self.fine_patch_size})"
            )
        if self.img_size % self.coarse_patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by "
                f"coarse_patch_size ({self.coarse_patch_size})"
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        return self


class ModelVariant(BaseModel):
    """Model variant configuration."""

    embed_dim: int = Field(ge=64, le=1024)
    depth: int = Field(ge=1, le=24)
    num_heads: int = Field(ge=1, le=16)
    mlp_ratio: float = Field(ge=1.0, le=8.0)
    approx_params: str | None = None


# =============================================================================
# Top-Level Configuration Models
# =============================================================================


class ExperimentConfig(BaseModel):
    """Complete experiment configuration schema."""

    experiment: ExperimentInfo = Field(default_factory=ExperimentInfo)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    centralized: CentralizedConfig = Field(default_factory=CentralizedConfig)
    federated_experiments: list[FederatedExperiment] = Field(default_factory=list)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)


class FLConfig(BaseModel):
    """Complete federated learning configuration schema."""

    federated: FederatedSettings
    scenarios: dict[str, ScenarioConfig] | None = None
    strategies: list[StrategyConfig] | None = None


class ModelConfig(BaseModel):
    """Complete model configuration schema."""

    model: ModelSettings
    variants: dict[str, ModelVariant] | None = None


# =============================================================================
# Validation Functions
# =============================================================================


def detect_config_type(config: dict[str, Any]) -> ConfigType:
    """
    Auto-detect the configuration type from its contents.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Detected ConfigType.

    Raises:
        ValueError: If config type cannot be determined.
    """
    if "experiment" in config or "federated_experiments" in config:
        return ConfigType.EXPERIMENT
    if "federated" in config and "num_clients" in config.get("federated", {}):
        return ConfigType.FEDERATED
    if "model" in config and "embed_dim" in config.get("model", {}):
        return ConfigType.MODEL
    raise ValueError(
        "Cannot auto-detect config type. Please specify ConfigType explicitly."
    )


def validate_config(
    config_path: str | Path,
    config_type: ConfigType | None = None,
) -> ExperimentConfig | FLConfig | ModelConfig:
    """
    Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.
        config_type: Type of configuration (auto-detected if not specified).

    Returns:
        Validated configuration model.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML is malformed.
        pydantic.ValidationError: If configuration is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    if config_type is None:
        config_type = detect_config_type(raw_config)

    if config_type == ConfigType.EXPERIMENT:
        return ExperimentConfig(**raw_config)
    if config_type == ConfigType.FEDERATED:
        return FLConfig(**raw_config)
    if config_type == ConfigType.MODEL:
        return ModelConfig(**raw_config)
    raise ValueError(f"Unknown config type: {config_type}")


def validate_config_dict(
    config: dict[str, Any],
    config_type: ConfigType | None = None,
) -> ExperimentConfig | FLConfig | ModelConfig:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration dictionary.
        config_type: Type of configuration (auto-detected if not specified).

    Returns:
        Validated configuration model.

    Raises:
        pydantic.ValidationError: If configuration is invalid.
    """
    if config_type is None:
        config_type = detect_config_type(config)

    if config_type == ConfigType.EXPERIMENT:
        return ExperimentConfig(**config)
    if config_type == ConfigType.FEDERATED:
        return FLConfig(**config)
    if config_type == ConfigType.MODEL:
        return ModelConfig(**config)
    raise ValueError(f"Unknown config type: {config_type}")


def get_default_config(config_type: ConfigType) -> dict[str, Any]:
    """
    Get default configuration for a given type.

    Args:
        config_type: Type of configuration.

    Returns:
        Dictionary with default configuration values.
    """
    if config_type == ConfigType.EXPERIMENT:
        return ExperimentConfig().model_dump()
    if config_type == ConfigType.FEDERATED:
        return FLConfig(
            federated=FederatedSettings()
        ).model_dump()
    if config_type == ConfigType.MODEL:
        return ModelConfig(
            model=ModelSettings()
        ).model_dump()
    raise ValueError(f"Unknown config type: {config_type}")


# =============================================================================
# CLI for Validation
# =============================================================================


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate configuration files")
    parser.add_argument("config_path", help="Path to YAML config file")
    parser.add_argument(
        "--type",
        choices=["experiment", "federated", "model"],
        help="Config type (auto-detected if not specified)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full config")

    args = parser.parse_args()

    config_type = ConfigType(args.type) if args.type else None

    try:
        validated = validate_config(args.config_path, config_type)
        print(f"✓ Configuration is valid: {args.config_path}")
        print(f"  Type: {type(validated).__name__}")

        if args.verbose:
            import json

            print("\nValidated configuration:")
            print(json.dumps(validated.model_dump(), indent=2))

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"✗ YAML Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Validation Error: {e}")
        sys.exit(1)
