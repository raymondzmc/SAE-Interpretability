import math
import torch
import torch.nn.functional as F
from typing import Any
from pydantic import Field, model_validator
from models.saes.base import SAEConfig, SAEOutput, SAELoss, BaseSAE
from utils.enums import SAEType
from models.saes.activations import get_activation


class LagrangianHardConcreteSAEConfig(SAEConfig):
    sae_type: SAEType = Field(default=SAEType.LAGRANGIAN_HARD_CONCRETE, description="Type of SAE (automatically set to hard_concrete)")
    initial_beta: float = Field(0.5, description="Initial beta for Hard Concrete annealing")
    final_beta: float | None = Field(None, description="Final beta for Hard Concrete annealing")
    beta_annealing: bool = Field(False, description="Whether to anneal beta during training")
    hard_concrete_stretch_limits: tuple[float, float] = Field((-0.1, 1.1), description="Hard concrete stretch limits")
    tied_encoder_init: bool = Field(True, description="Whether to tie the encoder weights to the decoder weights")
    magnitude_activation: str | None = Field("softplus", description="Activation function for magnitude ('relu', 'softplus', 'gelu', etc.) or None")
    coefficient_threshold: float = Field(1e-3, description="Threshold for the coefficients during inference")
    initial_alpha: float = Field(0.0, description="Initial alpha for Lagrangian dual-ascent controller")
    alpha_lr: float = Field(1e-2, description="Learning rate for alpha")
    rho: float = Field(0.005, description="Target sparsity level for Lagrangian dual-ascent controller")
    
    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Ensure sae_type is set to hard_concrete."""
        if isinstance(values, dict):
            values["sae_type"] = SAEType.LAGRANGIAN_HARD_CONCRETE
        return values


class LagrangianHardConcreteSAEOutput(SAEOutput):
    """HardConcrete SAE output that extends SAEOutput with additional parameters."""
    z: torch.Tensor
    magnitude: torch.Tensor
    gate_logits: torch.Tensor
    p_open: torch.Tensor
    alpha: torch.Tensor


class LagrangianHardConcreteSAE(BaseSAE):
    """
    Hard Concrete Sparse AutoEncoder using Hard Concrete stochastic gates for coefficients (L0 Sparsity).
    Combines L0 gating with magnitude for reconstruction.
    
    Supports two gate types:
    - Input-dependent gates: Gates computed from input (position/content-specific for NLP)
    - Input-independent gates: Global parameter-based gates (same across all positions for NLP)
    
    For NLP models with sequence inputs, input-independent gates use the same gate values
    across all sequence positions, making them truly global and more interpretable.
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        initial_beta: float, # Initial temperature for Hard Concrete
        initial_alpha: float = 0.0, # Initial alpha for Lagrangian dual-ascent controller
        alpha_lr: float = 1e-2,
        mse_coeff: float | None = None,
        rho: float = 0.05,
        stretch_limits: tuple[float, float] = (-0.1, 1.1), # Stretch limits for Hard Concrete
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        magnitude_activation: str | None = None,
        coefficient_threshold: float = 1e-3,
    ):
        super().__init__()
        self.n_dict_components = n_dict_components
        self.input_size = input_size
        self.rho = rho
        self.mse_coeff = mse_coeff or 1.0
        self.alpha_lr = alpha_lr
        self.coefficient_threshold = coefficient_threshold

        self.encoder_layer_norm = torch.nn.LayerNorm(input_size, elementwise_affine=False)
        self.gate_encoder = torch.nn.Linear(input_size, n_dict_components, bias=False)
        self.magnitude_activation = get_activation(magnitude_activation or "none")
        self.r_mag = torch.nn.Parameter(torch.zeros(n_dict_components))
        self.mag_bias = torch.nn.Parameter(torch.zeros(n_dict_components))

        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=False)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(input_size))
        
        self.l, self.r = stretch_limits
        assert self.l < 0.0 and self.r > 1.0, "stretch_limits must satisfy l < 0 and r > 1 for L0 penalty calculation"
        self.register_buffer("beta", torch.tensor(initial_beta, dtype=torch.float32, device='cpu'))
        self.register_buffer("alpha", torch.tensor(initial_alpha, dtype=torch.float32, device='cpu'))

        if init_decoder_orthogonal:
            self.decoder.weight.data = torch.nn.init.orthogonal_(self.decoder.weight.data.T).T

        if tied_encoder_init:
            self.gate_encoder.weight.data.copy_(self.decoder.weight.data.T)
        
        if self.gate_encoder.bias is not None:
            self.gate_encoder.bias.data.fill_(math.log(self.rho / (1 - self.rho)))
        
        self.inference_mode = "topk"
        self.inference_topk = int(round(self.rho * self.n_dict_components))
        self.score_use_magnitude = False
    
    def hard_concrete(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from the Hard Concrete distribution using the reparameterization trick.
        Used for L0 regularization as described in https://arxiv.org/abs/1712.01312.
        """
        epsilon = 1e-6
        if self.training:
            u = torch.rand_like(logits)
            s = torch.sigmoid((torch.log(u + epsilon) - torch.log(1.0 - u + epsilon) + logits) / self.beta)
            s_stretched = s * (self.r - self.l) + self.l
            z = s_stretched.clamp(min=0.0, max=1.0)
            # z_hard = s_stretched.clamp(min=0.0, max=1.0)
            # z = z_hard + (s_stretched - z_hard.detach())
        else:
            s = torch.sigmoid(logits / self.beta)
            s_stretched = s * (self.r - self.l) + self.l
            z = s.clamp(min=0.0, max=1.0)
        return z

    @torch.no_grad()
    def _deterministic_gate(self, gate_logits: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
        """Return z_det in {0,1} using the chosen inference policy."""
        q = torch.sigmoid(gate_logits - self.beta.item() * math.log(-self.l / self.r))
        if self.inference_mode == "topk":
            score = q * magnitude.abs() if self.score_use_magnitude else q
            # topk per token
            idx = score.topk(k=self.inference_topk, dim=-1, largest=True).indices  # [B,T,K]
            z_det = torch.zeros_like(q)
            z_det.scatter_(-1, idx, 1.0)
            return z_det
        elif self.inference_mode == "q_threshold":
            tau = self.inference_q_threshold
            if tau is None:
                # You can set a default (e.g., 0.5) or calibrate offline to match ρ·D
                tau = 0.5
            return (q >= tau).to(q.dtype)
        else:  # "coeff_threshold"
            tau_c = self.inference_coeff_threshold
            score = q * magnitude.abs() if self.score_use_magnitude else magnitude.abs()
            return (score >= tau_c).to(q.dtype)

    def forward(self, x: torch.Tensor) -> LagrangianHardConcreteSAEOutput:
        x_centered = self.encoder_layer_norm(x - self.decoder_bias)
        gate_logits = self.gate_encoder(x_centered)
        gate_logits = gate_logits - gate_logits.mean(dim=-1, keepdim=True) 
        magnitude = self.magnitude_activation(F.linear(x_centered, self.dict_elements.t()))
        z = self.hard_concrete(gate_logits)
        # if self.training:
        #     z = self.hard_concrete(gate_logits)
        # else:
        #     z = self._deterministic_gate(gate_logits, magnitude)
        # if not self.training:
            # z = torch.where(z >= self.coefficient_threshold, z, 0.0)
        coefficients = z * magnitude
        x_hat = F.linear(coefficients, self.dict_elements, bias=self.decoder_bias)
        p_open = torch.sigmoid(gate_logits - self.beta * math.log(-self.l / self.r))
        return LagrangianHardConcreteSAEOutput(
            input=x,
            c=coefficients,
            output=x_hat,
            magnitude=magnitude,
            z=z,
            gate_logits=gate_logits,
            p_open=p_open,
            alpha=self.alpha,
        )

    def compute_loss(self, output: LagrangianHardConcreteSAEOutput) -> SAELoss:
        """Compute the loss for the HardConcreteSAE.
        
        Args:
            output: The output of the HardConcreteSAE.
        """
        q = output.p_open                      # [B,T,D], built from shifted logits
        B, T, D = q.shape

        # 1) per-token entropy bonus
        # p_bt = q / (q.sum(dim=-1, keepdim=True) + 1e-8)     # normalized per token
        # H_token = -(p_bt.clamp_min(1e-8) * p_bt.clamp_min(1e-8).log()).sum(dim=-1).mean()

        # 2) batch load-balance (MoE style)
        # usage = q.sum(dim=(0,1))                              # [D]
        # p_feat = usage / (usage.sum() + 1e-8)
        # lb_kl = (p_feat * (p_feat.clamp_min(1e-8).log() - math.log(1.0 / D))).sum()

        # 3) your K controller (lower-bound or band) on the *expected* K
        K_per_pos = (q.sum(dim=-1)/D).mean()                           # [B,T]
        rho_hat = q.sum(dim=-1).mean() / D
        g = self.rho - rho_hat                                 # >0 if too sparse
        lag = (self.alpha.detach() * g) + (0.05 * g**2)
        mse = F.mse_loss(output.output, output.input)
        loss = self.mse_coeff * mse + lag
        return SAELoss(
            loss=loss,
            loss_dict={
                "mse_loss": mse.detach().clone(),
                "sparsity_loss": lag.detach().clone(),
                # "lb_kl": lb_kl.detach().clone(),
                # "H_token": H_token.detach().clone(),
                "expected_K": rho_hat.detach().clone(),
            },
        )


        # # Lagrangian dual-ascent controller (Lagrangian multiplier)
        # c = output.p_open.mean() - self.rho
        # expected_K = output.p_open.sum(dim=-1).mean()
        # sparsity_loss = self.alpha.detach() * c.clamp_min(0.0) + (0.05 * c**2)

        # # MSE loss
        # mse_loss = F.mse_loss(output.output, output.input)

        # # Total loss
        # loss = sparsity_loss + self.mse_coeff * mse_loss
        # return SAELoss(
        #     loss=loss,
        #     loss_dict={
        #         "mse_loss": mse_loss.detach().clone(),
        #         "sparsity_loss": sparsity_loss.detach().clone(),
        #         "expected_K": expected_K.detach().clone(),
        #     },
        # )
    
    @torch.no_grad()
    def dual_ascent_update_alpha(self, rho_hat: torch.Tensor, inequality: bool = False) -> None:
        """
        Dual ascent on alpha using the current (or accumulated) batch:
            alpha <- alpha + alpha_lr * (mean_{b,t} p_open - rho)
        Set inequality=True to enforce (mean p_open <= rho) with alpha >= 0.
        """
        delta = rho_hat - self.rho
        self.alpha.add_(self.alpha_lr * delta)
        if inequality:
            self.alpha.clamp_(min=0.0)

    @property
    def dict_elements(self):
        """Dictionary elements are simply the normalized decoder weights."""
        # Normalize columns (dim=0) of the weight matrix
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
