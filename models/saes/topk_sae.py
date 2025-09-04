# topk_sae.py

import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any
from pydantic import Field, ConfigDict, model_validator
from jaxtyping import Float

from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from utils.enums import SAEType


class TopKSAEConfig(SAEConfig):
    """
    Config for Top-K SAE.

    Notes (faithful to Gao et al., "Scaling and Evaluating Sparse Autoencoders"):
    - Enforce exact sparsity via a Top-K activation (no explicit L1 penalty).
    - Bias is a single learned vector used to center inputs before encoding and
      added back after decoding.
    - We support tied encoder initialization (encoder.weight = decoder.weight.T).
    - (Optional) Auxiliary loss to mitigate dead features by giving gradient signal
      to non-selected latents (implements a simple Aux-K).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    sae_type: SAEType = Field(default=SAEType.TOPK, description="Type of SAE (automatically set to topk)")
    k: int = Field(..., description="Number of active features to keep per sample")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")

    # Optional: dead-feature mitigation via auxiliary Top-K on the *inactive* set
    aux_k: int | None = Field(None, description="Auxiliary K for dead-feature loss (select top aux_k from the inactive set)")
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary reconstruction loss")

    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.TOPK
        return values


class TopKSAEOutput(SAEOutput):
    """
    TopK SAE output extending SAEOutput with useful intermediates for loss/analysis.
    """
    preacts: Float[torch.Tensor, "... c"]  # encoder linear outputs (after centering)
    mask: Float[torch.Tensor, "... c"]     # binary mask of selected Top-K indices


class TopKSAE(BaseSAE):
    """
    Top-K Sparse Autoencoder (PyTorch) faithful to Gao et al.:
      - Linear encoder/decoder (no bias on the linear layers)
      - Single learned decoder_bias used to center input and add back after decode
      - ReLU (optional) + Top-K selection per sample in the latent space
      - MSE reconstruction loss only (sparsity enforced by Top-K), with an optional Aux-K loss
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        k: int,
        sparsity_coeff: float | None = None,  # unused; kept for API parity
        mse_coeff: float | None = None,
        aux_k: int | None = None,
        aux_coeff: float | None = None,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
    ):
        """
        Args:
            input_size: Dimensionality of inputs (e.g., residual stream width).
            n_dict_components: Number of dictionary features (latent size).
            k: Number of active features to keep per sample (Top-K).
            sparsity_coeff: Unused for Top-K (present for interface compatibility).
            mse_coeff: Coefficient on MSE reconstruction loss (default 1.0).
            aux_k: If provided (>0), number of auxiliary features from the inactive set.
            aux_coeff: Coefficient on the auxiliary reconstruction loss (default 0.0 if aux_k is None).
            init_decoder_orthogonal: Initialize decoder weight columns to be orthonormal.
            tied_encoder_init: Initialize encoder.weight = decoder.weight.T.
        """
        super().__init__()
        assert k >= 0, "k must be non-negative"
        assert n_dict_components > 0 and input_size > 0

        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.k = int(k)
        assert self.k <= n_dict_components, "k must be less than or equal to n_dict_components"

        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 0.0  # not used, but kept for logs
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0

        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0

        # Bias used for input centering and added back on decode
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))

        # Linear maps (no bias)
        self.encoder = nn.Linear(input_size, n_dict_components, bias=False)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)

        # Initialize decoder, then (optionally) tie encoder init to decoder^T
        if init_decoder_orthogonal:
            self.decoder.weight.data = nn.init.orthogonal_(self.decoder.weight.data.T).T
        else:
            # Random unit-norm columns
            dec_w = torch.randn_like(self.decoder.weight)
            dec_w = F.normalize(dec_w, dim=0)
            self.decoder.weight.data.copy_(dec_w)

        if tied_encoder_init:
            self.encoder.weight.data.copy_(self.decoder.weight.data.T)
        
        p0 = max(1e-4, min(1 - 1e-4, self.k / self.n_dict_components))
        init_logit = math.log(p0) - math.log(1 - p0)
        self.gate_logit = nn.Parameter(torch.full((self.n_dict_components,), init_logit))
        self.gumbel_temp = 1.0
    
    def _st_gumbel_topk(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Straight-through Gumbel-TopK:
        forward: hard Top-K on z + gate_logit + gumbel
        backward: softmax((z + gate_logit) / tau) surrogate
        Returns:
        code: z masked by straight-through mask
        mask: hard mask used in forward (useful for analysis/metrics)
        """
        logits = self.gate_logit  # (C,)
        if self.training:
            # Sample per-sample, per-feature Gumbel noise
            u = torch.rand_like(z).clamp_(1e-6, 1 - 1e-6)
            g = -torch.log(-torch.log(u))
            scores = z + logits + self.gumbel_temp * g
        else:
            scores = z + logits  # deterministic at eval

        # Hard Top-K (exact K)
        topk_idx = torch.topk(scores, k=self.k, dim=-1).indices
        hard = torch.zeros_like(z).scatter(-1, topk_idx, 1.0)

        if self.training:
            # Soft surrogate for gradients (dense)
            soft = torch.softmax((z + logits) / self.gumbel_temp, dim=-1)
            mask = hard + soft - soft.detach()  # straight-through
        else:
            mask = hard

        code = z * mask
        return code, hard  # return hard for accounting

    def _apply_topk(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply (optional) ReLU then Top-K selection along the last dimension.
        Returns:
            code: sparse activations after masking to Top-K
            mask: binary mask (same shape as z) with ones at Top-K indices
        """
        # Compute Top-K per sample along last dim
        topk_idx = torch.topk(z, k=self.k, dim=-1)[1]
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk_idx, 1.0)
        code = z * mask
        return code, mask

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> TopKSAEOutput:
        """
        Forward pass (supports arbitrary leading batch dims; last dim == input_size).
        """
        # Center input
        x_centered = x - self.decoder_bias
        # Encoder preactivations
        preacts = self.encoder(x_centered)  # (..., n_dict_components)
        c, hard_mask = self._st_gumbel_topk(preacts)
        # Top-K sparsification
        # c, mask = self._apply_topk(preacts)
        # Decode using normalized dictionary elements + add bias back
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)
        return TopKSAEOutput(input=x, c=c, output=x_hat, logits=None, preacts=preacts, mask=hard_mask)
    
    def sample_hard_concrete(self, log_alpha: torch.Tensor, tau: float = 0.5,
                             limit_a: float = -0.1, limit_b: float = 1.1):
        # Maddison/Jang (Concrete) + Louizos et al. (Hard-Concrete)
        u = torch.rand_like(log_alpha).clamp_(1e-6, 1-1e-6)
        s = torch.sigmoid((log_alpha + torch.log(u) - torch.log(1 - u)) / tau)
        s_bar = s * (limit_b - limit_a) + limit_a
        z = s_bar.clamp(0.0, 1.0)  # gate in [0,1]
        return z

    def compute_loss(self, output: TopKSAEOutput) -> SAELoss:
        """
        Loss = mse_coeff * MSE + aux_coeff * AuxK (optional)

        - No explicit L1 sparsity term (sparsity enforced by Top-K).
        - AuxK: select top aux_k features from the *inactive* set (per-sample),
          reconstruct with a detached decoder to provide gradient to "dead" features
          without moving the decoder, then compute an auxiliary MSE to the input.
        """
        # Reconstruction loss
        mse_loss = F.mse_loss(output.output, output.input)
        total_loss = self.mse_coeff * mse_loss
        loss_dict: dict[str, torch.Tensor] = {"mse_loss": mse_loss.detach().clone()}

        p = torch.sigmoid(self.gate_logit)
        budget_loss = ((p.sum() - self.k) ** 2) / (self.n_dict_components + 1e-8)
        total_loss = total_loss + 0.005 * budget_loss
        loss_dict["budget_loss"] = budget_loss.detach().clone()

        # Optional auxiliary dead-feature loss
        if self.aux_k > 0 and self.aux_coeff > 0.0:
            z = output.preacts
            # Zero out the already-selected Top-K, then pick top aux_k from the remainder
            z_inactive = z * (1.0 - output.mask)
            # Handle edge cases (aux_k == 0 or >= latent dim)
            latent_dim = z_inactive.size(-1)
            aux_k = min(self.aux_k, max(0, latent_dim - self.k))
            if aux_k > 0:
                aux_idx = torch.topk(z_inactive, k=aux_k, dim=-1)[1]
                aux_mask = torch.zeros_like(z_inactive)
                aux_mask.scatter_(-1, aux_idx, 1.0)
                aux_code = z * aux_mask  # use actual (ReLUed) magnitudes for those indices

                # Reconstruct with DETACHED normalized decoder and bias
                with torch.no_grad():
                    dec_w_detached = F.normalize(self.decoder.weight.detach(), dim=0)
                    dec_b_detached = self.decoder_bias.detach()
                x_hat_aux = F.linear(aux_code, dec_w_detached, bias=dec_b_detached)

                aux_loss = F.mse_loss(x_hat_aux, output.input)
                total_loss = total_loss + self.aux_coeff * aux_loss
                loss_dict["aux_loss"] = aux_loss.detach().clone()
            else:
                # No room for auxiliary picks; report zero aux loss
                loss_dict["aux_loss"] = torch.zeros((), device=output.input.device)

        return SAELoss(loss=total_loss, loss_dict=loss_dict)

    @property
    def dict_elements(self) -> torch.Tensor:
        """
        Column-wise unit-norm decoder (dictionary) – normalized every forward.
        This mirrors common SAE practice and avoids degenerate scaling solutions.
        """
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
