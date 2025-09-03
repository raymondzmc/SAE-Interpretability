from enum import Enum


class SAEType(str, Enum):
    """Enum for different SAE types."""
    RELU = "relu"
    HARD_CONCRETE = "hard_concrete"
    LAGRANGIAN_HARD_CONCRETE = "lagrangean_hard_concrete"
    GATED = "gated"
    GATED_HARD_CONCRETE = "gated_hard_concrete"
    TOPK = "topk"
    GUMBEL_TOPK = "gumbel_topk"
    VI_TOPK = "vi_topk"
    # Add more SAE types as needed


class EncoderType(str, Enum):
    """Enum for different encoder types."""
    SCALE = "scale"
    SEPARATE = "separate"
    DECODER_TRANSPOSE = "decoder_transpose"
    NONE = "none"


class LossType(str, Enum):
    """Enum for different loss types (for future extensibility)."""
    MSE = "mse"
    L1 = "l1"
    L0 = "l0"
    # Add more loss types as needed
