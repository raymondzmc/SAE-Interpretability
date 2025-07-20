from models.saes.base import SAEConfig, SAEOutput, BaseSAE, SAELoss
from models.saes.relu_sae import ReluSAE
from models.saes.hc_sae import HardConcreteSAE, HardConcreteSAEConfig, HardConcreteSAEOutput
from models.saes.gated_sae import (
    GatedSAE, GatedSAEConfig, GatedSAEOutput,
    GatedHardConcreteSAE, GatedHardConcreteSAEConfig, GatedHardConcreteSAEOutput
)
from utils.enums import SAEType
from typing import Any


def create_sae_config(config_dict: dict[str, Any]) -> SAEConfig:
    """Factory function to create the appropriate SAE config based on sae_type.
    
    Args:
        config_dict: Dictionary containing SAE configuration parameters
        
    Returns:
        Appropriate SAEConfig subclass instance
        
    Raises:
        NotImplementedError: If sae_type is not supported
        ValueError: If sae_type is missing from config_dict
    """
    if "sae_type" not in config_dict:
        raise ValueError("sae_type must be specified in SAE config")
    
    sae_type = SAEType(config_dict["sae_type"])
    
    if sae_type == SAEType.HARD_CONCRETE:
        return HardConcreteSAEConfig.model_validate(config_dict)
    elif sae_type == SAEType.RELU:
        # For now, use base SAEConfig for ReLU SAEs
        return SAEConfig.model_validate(config_dict)
    elif sae_type == SAEType.GATED:
        return GatedSAEConfig.model_validate(config_dict)
    elif sae_type == SAEType.GATED_HARD_CONCRETE:
        return GatedHardConcreteSAEConfig.model_validate(config_dict)
    else:
        raise NotImplementedError(f"SAE type '{sae_type}' is not supported")


# Keep track of available SAE types for validation
AVAILABLE_SAE_TYPES = {SAEType.RELU, SAEType.HARD_CONCRETE}
IMPLEMENTED_SAE_TYPES = {SAEType.RELU, SAEType.HARD_CONCRETE}


__all__ = [
    "BaseSAE",
    "SAEConfig", 
    "SAELoss",
    "SAEOutput",
    "ReluSAE",
    "HardConcreteSAE", 
    "HardConcreteSAEConfig", 
    "HardConcreteSAEOutput",
    "GatedSAE",
    "GatedSAEConfig",
    "GatedSAEOutput",
    "create_sae_config",
    "AVAILABLE_SAE_TYPES",
    "IMPLEMENTED_SAE_TYPES",
]