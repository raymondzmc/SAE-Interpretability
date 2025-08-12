from enum import Enum


class SAEType(str, Enum):
    """Enum for different SAE types."""
    RELU = "relu"
    HARD_CONCRETE = "hard_concrete"
    GATED = "gated"
    GATED_HARD_CONCRETE = "gated_hard_concrete"
    TOPK = "topk"
    # Add more SAE types as needed
    

class LossType(str, Enum):
    """Enum for different loss types (for future extensibility)."""
    MSE = "mse"
    L1 = "l1"
    L0 = "l0"
    # Add more loss types as needed
