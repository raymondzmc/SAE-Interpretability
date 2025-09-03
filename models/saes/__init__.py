from models.saes.base import SAEConfig, SAEOutput, BaseSAE, SAELoss
from models.saes.relu_sae import ReluSAE, ReLUSAEConfig
from models.saes.hc_sae import HardConcreteSAE, HardConcreteSAEConfig, HardConcreteSAEOutput
from models.saes.lagrangian_hc_sae import LagrangianHardConcreteSAE, LagrangianHardConcreteSAEConfig, LagrangianHardConcreteSAEOutput
from models.saes.gated_sae import (
    GatedSAE, GatedSAEConfig, GatedSAEOutput,
    GatedHardConcreteSAE, GatedHardConcreteSAEConfig, GatedHardConcreteSAEOutput
)
from models.saes.gumbel_topk_sae import GumbelTopKSAE, GumbelTopKSAEConfig, GumbelTopKSAEOutput
from models.saes.topk_sae import TopKSAE, TopKSAEConfig, TopKSAEOutput
from models.saes.vi_sae import VITopKSAE, VITopKSAEConfig, VITopKSAEOutput
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
    elif sae_type == SAEType.LAGRANGIAN_HARD_CONCRETE:
        return LagrangianHardConcreteSAEConfig.model_validate(config_dict)
    elif sae_type == SAEType.RELU:
        return ReLUSAEConfig.model_validate(config_dict)
    elif sae_type == SAEType.GATED:
        return GatedSAEConfig.model_validate(config_dict)
    elif sae_type == SAEType.GATED_HARD_CONCRETE:
        return GatedHardConcreteSAEConfig.model_validate(config_dict)
    elif sae_type == SAEType.TOPK:
        return TopKSAEConfig.model_validate(config_dict)
    elif sae_type == SAEType.GUMBEL_TOPK:
        return GumbelTopKSAEConfig.model_validate(config_dict)
    elif sae_type == SAEType.VI_TOPK:
        return VITopKSAEConfig.model_validate(config_dict)
    else:
        raise NotImplementedError(f"SAE type '{sae_type}' is not supported")


# Keep track of available SAE types for validation
AVAILABLE_SAE_TYPES = {SAEType.RELU, SAEType.HARD_CONCRETE, SAEType.LAGRANGIAN_HARD_CONCRETE, SAEType.GATED, SAEType.GATED_HARD_CONCRETE, SAEType.TOPK, SAEType.GUMBEL_TOPK, SAEType.VI_TOPK}
IMPLEMENTED_SAE_TYPES = {SAEType.RELU, SAEType.HARD_CONCRETE, SAEType.LAGRANGIAN_HARD_CONCRETE, SAEType.GATED, SAEType.GATED_HARD_CONCRETE, SAEType.TOPK, SAEType.GUMBEL_TOPK, SAEType.VI_TOPK}


__all__ = [
    "BaseSAE",
    "SAEConfig", 
    "SAELoss",
    "SAEOutput",
    "ReluSAE",
    "ReLUSAEConfig",
    "HardConcreteSAE", 
    "HardConcreteSAEConfig", 
    "HardConcreteSAEOutput",
    "LagrangianHardConcreteSAE",
    "LagrangianHardConcreteSAEConfig",
    "LagrangianHardConcreteSAEOutput",
    "GatedSAE",
    "GatedSAEConfig",
    "GatedSAEOutput",
    "GatedHardConcreteSAE",
    "GatedHardConcreteSAEConfig",
    "GatedHardConcreteSAEOutput",
    "TopKSAE",
    "TopKSAEConfig",
    "TopKSAEOutput",
    "GumbelTopKSAE",
    "GumbelTopKSAEConfig",
    "GumbelTopKSAEOutput",
    "VITopKSAE",
    "VITopKSAEConfig", 
    "VITopKSAEOutput",
    "create_sae_config",
    "AVAILABLE_SAE_TYPES",
    "IMPLEMENTED_SAE_TYPES",
]