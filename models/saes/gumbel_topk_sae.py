# prob_gated_sae.py
import torch
import torch.nn.functional as F
from typing import Any, Callable
from pydantic import Field, model_validator
from models.saes.base import SAEConfig, SAEOutput, SAELoss, BaseSAE
from utils.enums import SAEType

ACTIVATION_MAP: dict[str, Callable] = {
    "relu": F.relu,
    "softplus": F.softplus,
    "none": None,
}

class GumbelTopKSAEConfig(SAEConfig):
    sae_type: SAEType = Field(default=SAEType.GUMBEL_TOPK, description="Probabilistic Gated SAE")
    k: int = Field(..., description="Number of active features to keep per sample")
    gumbel_temperature: float = Field(1.0, description="Temperature for soft surrogate (grad) in Gumbel-TopK")
    magnitude_activation: str | None = Field("softplus")
    init_decoder_orthogonal: bool = Field(True)
    tied_encoder_init: bool = Field(True)
    decoder_bias: bool = Field(True)

    @model_validator(mode="before")
    @classmethod
    def _fix_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.GATED
        return values


class GumbelTopKSAEOutput(SAEOutput):
    z: torch.Tensor           # sampled hard/soft gate used for reconstruction (straight-through)
    z_soft: torch.Tensor      # soft surrogate used for gradients (logging/aux losses)
    p_open: torch.Tensor      # probability/confidence proxy
    logits: torch.Tensor      # gate logits (after optional centering/scaling)
    magnitude: torch.Tensor
    x_hat: torch.Tensor


def _sample_gumbel_topk(
    logits: torch.Tensor,
    K: int,
    temp: float,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gumbel-TopK sampling with ST estimator:
      - Sample hard one-hot K-hot mask z_hard by top-K of logits + Gumbel
      - Soft surrogate z_soft = K * softmax((logits + gumbel)/temp) for gradients
    Returns:
      z_st: straight-through K-hot mask (forward hard, backward soft)
      z_soft: soft K-summing mask (grad surrogate)
    """
    if training:
        eps = 1e-6
        g = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)  # Gumbel(0,1)
        y = logits + g
    else:
        y = logits

    topk = torch.topk(y, k=K, dim=-1)
    idx = topk.indices
    z_hard = torch.zeros_like(logits)
    z_hard.scatter_(-1, idx, 1.0)
    z_soft = K * F.softmax(y / temp, dim=-1)
    z_st = z_hard + z_soft - z_soft.detach()  # straight-through
    return z_st, z_soft


class GumbelTopKSAE(BaseSAE):
    """
    Probabilistic Gated SAE with *stochastic* gates:
      - gate_type="hard_concrete": independent HC gates (z in [0,1])
      - gate_type="gumbel_topk":  exact K-hot gates via Gumbel-TopK
    Purely input-dependent (gate encoder is bias=False).
    No α/μ dual variables; no dataset-level push to zero usage.
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        k: int,
        gumbel_temperature: float = 1.0,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        magnitude_activation: str | None = "softplus",
        decoder_bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_dict_components = n_dict_components

        self.k = int(k)
        self.gumbel_temp = float(gumbel_temperature)

        # Encoders/Decoder
        self.encoder_ln = torch.nn.LayerNorm(input_size)
        self.gate_encoder = torch.nn.Linear(input_size, n_dict_components, bias=True)
        self.magnitude_encoder = torch.nn.Linear(input_size, n_dict_components, bias=True)
        self.magnitude_activation = ACTIVATION_MAP.get((magnitude_activation or "none").lower())

        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=False)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(input_size)) if decoder_bias else None

        # Inits
        if init_decoder_orthogonal:
            self.decoder.weight.data = torch.nn.init.orthogonal_(self.decoder.weight.data.T).T
        if tied_encoder_init:
            self.magnitude_encoder.weight.data.copy_(self.decoder.weight.data.T)
        torch.nn.init.normal_(self.gate_encoder.weight, mean=0.0, std=0.02)

    @property
    def dict_elements(self):
        # Normalize dictionary columns
        return F.normalize(self.decoder.weight, dim=0)

    def _pre_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.decoder_bias is not None:
            x = x - self.decoder_bias
        return self.encoder_ln(x)

    def forward(self, x: torch.Tensor) -> GumbelTopKSAEOutput:
        """
        x: (B, D_in) or (B, T, D_in)
        Returns z (sampled), z_soft (surrogate), p_open (confidence), magnitude, x_hat
        """
        x = self._pre_x(x)
        logits = self.gate_encoder(x)

        z_st, z_soft = _sample_gumbel_topk(
            logits, K=self.k, temp=self.gumbel_temp, training=self.training
        )
        p_open = torch.sigmoid(logits / max(1e-6, self.gumbel_temp))

        # magnitude
        mag_pre = self.magnitude_encoder(x)
        magnitude = self.magnitude_activation(mag_pre) if self.magnitude_activation else mag_pre

        # codes and reconstruction (straight-through z)
        c = z_st * magnitude
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)

        return GumbelTopKSAEOutput(
            input=x, output=x_hat, c=c,
            z=z_st, z_soft=z_soft, p_open=p_open, logits=logits, magnitude=magnitude
        )

    def compute_loss(self, output: GumbelTopKSAEOutput) -> SAELoss:
        loss = F.mse_loss(output.output, output.input)
        return SAELoss(
            loss=loss,
            loss_dict={
                "mse_loss": loss.detach().clone(),
                "sum_z_mean": output.z.sum(dim=-1).mean().detach().clone(),
                "sum_z_soft_mean": output.z_soft.sum(dim=-1).mean().detach().clone(),
            },
        )

    @property
    def device(self):
        return next(self.parameters()).device
