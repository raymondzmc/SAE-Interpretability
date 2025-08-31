import torch
import torch.nn.functional as F
from typing import Callable


def softplus0(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x) - F.softplus(torch.zeros((), device=x.device, dtype=x.dtype))


def get_activation(activation: str | None = None) -> Callable:
    ACTIVATION_MAP: dict[str, Callable] = {
        'relu': F.relu,
        'softplus': F.softplus,
        'softplus0': softplus0,
        'none': torch.nn.Identity(),
    }
    if activation is None:
        return torch.nn.Identity()
    else:
        return ACTIVATION_MAP[activation]
