# topk_sae.py
import torch
import torch.nn.functional as F
from typing import Any, Literal
from pydantic import Field, model_validator
from jaxtyping import Float
from utils.enums import SAEType
from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig


class HardConcreteTopKSAEConfig(SAEConfig):
    sae_type: SAEType = Field(default=SAEType.HARD_CONCRETE_TOPK, description="Type of SAE (automatically set to hard_concrete_topk)")
    k: int = Field(..., description="Number of active features to keep per sample")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")

    # Optional: dead-feature mitigation via auxiliary Top-K on the *inactive* set
    aux_k: int | None = Field(None, description="Auxiliary K for dead-feature loss (select top aux_k from the inactive set)")
    aux_coeff: float | None = Field(None, description="Coefficient for the auxiliary reconstruction loss")
    
    initial_beta: float = Field(5.0, description="Initial beta for hard concrete sampling")
    final_beta: float | None = Field(None, description="Final beta for hard concrete sampling")

    score_method: Literal["gate_only", "magnitude", "magnitude_detached"] = Field("gate_only", description="Method to compute the score for the Top-K selection")
    straight_through: bool = Field(False, description="Use straight-through Top-K")
    tau: float | None = Field(None, description="Temperature for straight-through Top-K")

    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.HARD_CONCRETE_TOPK
        return values


class HardConcreteTopKSAEOutput(SAEOutput):
    """
    TopK SAE output extending SAEOutput with useful intermediates for loss/analysis.
    """
    preacts: Float[torch.Tensor, "... c"]  # encoder linear outputs (after centering)
    mask: Float[torch.Tensor, "... c"]     # binary mask of selected Top-K indices
    scores: Float[torch.Tensor, "... c"]   # scores of the selected Top-K indices


class HardConcreteTopKSAE(BaseSAE):
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
        initial_beta: float = 5.0,
        final_beta: float | None = None,
        score_method: Literal["gate_only", "magnitude", "magnitude_detached"] = "gate_only",
        straight_through: bool = False,
        tau: float | None = None,
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
            initial_beta: Initial beta for hard concrete sampling.
            final_beta: Final beta for hard concrete sampling.
            score_method: Method to compute the score for the Top-K selection.
            straight_through: Use straight-through Top-K.
            tau: Temperature for straight-through Top-K.
        """
        super().__init__()
        assert k >= 0, "k must be non-negative"
        assert n_dict_components > 0 and input_size > 0

        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.k = int(k)
        assert self.k > 0 and self.k <= n_dict_components, "k must be greater than 0 and less than or equal to n_dict_components"

        # Loss coefficients
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 0.0  # not used, but kept for logs
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0

        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0

        # Bias used for input centering and added back on decode
        self.decoder_bias = torch.nn.Parameter(torch.zeros(input_size))

        # Linear maps (no bias)
        self.encoder = torch.nn.Linear(input_size, n_dict_components, bias=False)
        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=False)

        # Initialize decoder, then (optionally) tie encoder init to decoder^T
        if init_decoder_orthogonal:
            self.decoder.weight.data = torch.nn.init.orthogonal_(self.decoder.weight.data.T).T
        else:
            # Random unit-norm columns
            dec_w = torch.randn_like(self.decoder.weight)
            dec_w = F.normalize(dec_w, dim=0)
            self.decoder.weight.data.copy_(dec_w)

        if tied_encoder_init:
            self.encoder.weight.data.copy_(self.decoder.weight.data.T)

        self.gate_ln = torch.nn.LayerNorm(n_dict_components, elementwise_affine=False)
        self.gate_scale = torch.nn.Parameter(torch.randn(n_dict_components))
        self.gate_bias = torch.nn.Parameter(torch.ones(n_dict_components))
        
        self.register_buffer("train_progress", torch.tensor(0.0))
        self.register_buffer("beta", torch.tensor(initial_beta, dtype=torch.float32))
        self.final_beta = final_beta

        self.score_method = score_method
        self.straight_through = straight_through
        self.tau = 1.0 if straight_through and tau is None else tau

    def sample_hard_concrete(self, logits: torch.Tensor):
        u = torch.rand_like(logits).clamp_(1e-6, 1-1e-6)
        z = torch.sigmoid((logits + torch.log(u) - torch.log(1 - u)) / self.beta)
        return z

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> HardConcreteTopKSAEOutput:
        """
        Forward pass (supports arbitrary leading batch dims; last dim == input_size).
        """
        # Center input
        x_centered = x - self.decoder_bias
        preacts = self.encoder(x_centered)

        gate_logits = self.gate_scale * self.gate_ln(preacts) + self.gate_bias
        z = self.sample_hard_concrete(gate_logits)

        # Compute scores
        if self.score_method == "gate_only":
            scores = z
        elif self.score_method == "magnitude":
            scores = z + preacts.abs()
        elif self.score_method == "magnitude_detached":
            scores = z + preacts.detach().clone().abs()
        else:
            raise ValueError(f"Invalid score_method: {self.score_method}")

        # Select top-k indices
        topk_idx = torch.topk(scores, k=self.k, dim=-1)[1]
        mask = torch.zeros_like(preacts)
        mask.scatter_(-1, topk_idx, 1.0)

        # Add a straight-through soft mask
        if self.straight_through and self.training:
            soft = torch.softmax(scores / self.tau, dim=-1)
            soft_k = soft * (self.k / (soft.sum(dim=-1, keepdim=True) + 1e-8))
            soft_k = soft_k.clamp(max=1.0)
            mask += soft_k - soft_k.detach()

        c = preacts * mask

        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)
        return HardConcreteTopKSAEOutput(input=x, c=c, output=x_hat, preacts=preacts, mask=mask, scores=scores)


    def compute_loss(self, output: HardConcreteTopKSAEOutput) -> SAELoss:
        """
        Loss = mse_coeff * MSE + aux_coeff * AuxK (optional)

        - No explicit L1 sparsity term (sparsity enforced by Top-K).
        - AuxK: select top aux_k features from the *inactive* set (per-sample),
          reconstruct with a detached decoder to provide gradient to "dead" features
          without moving the decoder, then compute an auxiliary MSE to the input.
        """
        mse_loss = F.mse_loss(output.output, output.input)
        total_loss = self.mse_coeff * mse_loss
        loss_dict: dict[str, torch.Tensor] = {"mse_loss": mse_loss.detach().clone()}
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
