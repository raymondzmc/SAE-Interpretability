import torch
import torch.nn.functional as F
from typing import Callable
from pydantic import Field
from models.saes.base import SAEConfig, SAEOutput, SAELoss, BaseSAE
from utils.enums import SAEType

ACTIVATION_MAP: dict[str, Callable] = {
    "relu": F.relu,
    "softplus": F.softplus,
    "none": lambda x: x,
}

class GumbelTopKSAEConfig(SAEConfig):
    sae_type: SAEType = Field(default=SAEType.GUMBEL_TOPK, description="Probabilistic Gated SAE")
    k: int = Field(..., description="Number of active features to keep per sample")
    gumbel_temperature: float = Field(1.0, description="Temperature for soft surrogate (grad) in Gumbel-TopK")
    magnitude_activation: str | None = Field("softplus")
    init_decoder_orthogonal: bool = Field(True)
    tied_encoder_init: bool = Field(True)
    decoder_bias: bool = Field(True)
    aux_k: int | None = Field(None, description="Auxiliary K for dead-feature loss (select top aux_k from the inactive set)")
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary reconstruction loss")


class GumbelTopKSAEOutput(SAEOutput):
    z: torch.Tensor
    z_soft: torch.Tensor
    magnitude: torch.Tensor


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
        aux_k: int | None = None,
        aux_coeff: float | None = None,
        gate_dropout: float | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_dict_components = n_dict_components

        self.k = int(k)
        self.gumbel_temp = float(gumbel_temperature)
        self.aux_k = aux_k
        self.aux_coeff = aux_coeff

        # Encoders/Decoder
        # self.encoder = torch.nn.Linear(input_size, n_dict_components, bias=False)
        # self.gate_encoder = torch.nn.Linear(input_size, n_dict_components, bias=False)
        # self.gate_dropout = torch.nn.Dropout(gate_dropout) if gate_dropout is not None else None
        # self.magnitude_encoder = torch.nn.Linear(input_size, n_dict_components, bias=False)
        self.r_mag = torch.nn.Parameter(torch.zeros(n_dict_components))
        self.magnitude_bias = torch.nn.Parameter(torch.zeros(n_dict_components))
        self.magnitude_activation = ACTIVATION_MAP.get((magnitude_activation or "none").lower())

        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=False)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(input_size)) if decoder_bias else None


        if init_decoder_orthogonal:
            self.decoder.weight.data = torch.nn.init.orthogonal_(self.decoder.weight.data.T).T
        if tied_encoder_init:
            self.magnitude_encoder.weight.data.copy_(self.decoder.weight.data.T)
            self.gate_encoder.weight.data.copy_(self.decoder.weight.data.T)

    @property
    def dict_elements(self):
        # Normalize dictionary columns
        return F.normalize(self.decoder.weight, dim=0)

    def forward(self, x: torch.Tensor) -> GumbelTopKSAEOutput:
        """
        x: (B, D_in) or (B, T, D_in)
        Returns z (sampled), z_soft (surrogate), p_open (confidence), magnitude, x_hat
        """
        x_centered = x - self.decoder_bias if self.decoder_bias is not None else x
        x_dir = x_centered / (x_centered.norm(dim=-1, keepdim=True) + 1e-8)
        encoder_out = F.linear(x_dir, self.dict_elements.t())
        z_st, z_soft = _sample_gumbel_topk(encoder_out, K=self.k, temp=self.gumbel_temp, training=self.training)
        
        magnitude = self.magnitude_activation(self.r_mag.exp() * encoder_out + self.magnitude_bias)
        # logits = self.gate_encoder(x_centered)
        # logits = logits - logits.mean(dim=-1, keepdim=True)

        # z_st, z_soft = _sample_gumbel_topk(
        #     logits, K=self.k, temp=self.gumbel_temp, training=self.training
        # )
        # mag_pre = self.magnitude_encoder(x_centered)
        # magnitude = self.magnitude_activation(mag_pre) if self.magnitude_activation else mag_pre

        c = z_st * magnitude
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)

        return GumbelTopKSAEOutput(
            input=x, output=x_hat, c=c,
            z=z_st, z_soft=z_soft, logits=None, magnitude=magnitude,
        )

    def compute_loss(self, output: GumbelTopKSAEOutput) -> SAELoss:
        loss = F.mse_loss(output.output, output.input)
        aux_loss = torch.zeros((), device=output.input.device)
        if self.training and self.aux_coeff and self.aux_k:
            e = output.input - output.output
            # pick from "inactive" set by zeroing current winners (use z_hard as in output.z)
            z_soft = output.z_soft
            inactive_scores = z_soft * (1.0 - output.z)  # prefer near-miss losers
            k_aux = max(0, min(self.aux_k, inactive_scores.size(-1) - self.k))
            if k_aux > 0:
                _, aux_idx = torch.topk(inactive_scores, k=k_aux, dim=-1)
                aux_mask = torch.zeros_like(inactive_scores).scatter_(-1, aux_idx, 1.0)
                aux_code = aux_mask * output.magnitude
                with torch.no_grad():
                    dec_w = F.normalize(self.decoder.weight.detach(), dim=0)
                e_hat = F.linear(aux_code, dec_w, bias=None)
                aux_loss = F.mse_loss(e_hat, e)
                loss += self.aux_coeff * aux_loss

        return SAELoss(
            loss=loss,
            loss_dict={
                "mse_loss": loss.detach().clone(),
                "aux_loss": aux_loss.detach().clone(),
            },
        )

    @property
    def device(self):
        return next(self.parameters()).device
