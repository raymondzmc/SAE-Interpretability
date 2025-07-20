import torch
import torch.nn.functional as F
from torch import nn
from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from pydantic import ConfigDict, Field, model_validator
from typing import Any
import math
from utils.enums import SAEType


class GatedSAEConfig(SAEConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)
    aux_coeff: float | None = None


class GatedSAEOutput(SAEOutput):
    gates: torch.Tensor
    magnitudes: torch.Tensor
    mask: torch.Tensor


class GatedSAE(BaseSAE):
    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        sparsity_coeff: float | None = None,
        mse_coeff: float | None = None,
        aux_coeff: float | None = None,
    ):
        """
        Gated Sparse Autoencoder with tied encoder weights.
        input_dim: dimensionality of input x.
        hidden_dim: number of sparse features (dictionary size).
        """
        super().__init__()
        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0
        self.aux_coeff = aux_coeff if aux_coeff is not None else 1.0

        # Decoder bias for input centering
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))
        
        # Encoder (no bias, tied to decoder transpose)
        self.encoder = nn.Linear(input_size, n_dict_components, bias=False)
        
        # Magnitude network parameters
        self.r_mag = nn.Parameter(torch.zeros(n_dict_components))
        self.mag_bias = nn.Parameter(torch.zeros(n_dict_components))
        
        # Gating network parameters
        self.gate_bias = nn.Parameter(torch.zeros(n_dict_components))

        # Decoder (no bias, bias handled separately)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)
        
        # Initialize weights properly
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        nn.init.zeros_(self.decoder_bias)
        nn.init.zeros_(self.r_mag)
        nn.init.zeros_(self.gate_bias)
        nn.init.zeros_(self.mag_bias)

        # decoder weights are initialized to random unit vectors
        dec_weight = torch.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)
        # tie encoder weights to decoder transpose
        self.encoder.weight = nn.Parameter(dec_weight.clone().T)

    def forward(self, x: torch.Tensor):
        """
        Forward pass returns reconstruction and intermediate codes.
        """
        # Center input by subtracting decoder bias (treated as input mean)
        x_enc = self.encoder(x - self.decoder_bias)  # (batch, hidden_dim)

        # Gating network: simple bias addition (no scaling)
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).float()  # Heaviside step -> {0,1}

        # Magnitude network: exponential scaling + bias + ReLU
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = F.relu(pi_mag)

        # Combine gating and magnitude
        code = f_gate * f_mag
        
        # Decode
        recon = self.decoder(code) + self.decoder_bias  # (batch, input_dim)
        
        return GatedSAEOutput(
            input=x, 
            c=code, 
            output=recon, 
            logits=None, 
            gates=pi_gate,  # gating pre-activations
            magnitudes=f_mag,  # magnitude activations
            mask=f_gate  # binary gating mask
        )

    def compute_loss(self, output: GatedSAEOutput) -> SAELoss:
        """
        Compute the Gated SAE loss based on the reference implementation:
        L_recon + L_sparsity + L_aux
        """
        # L_recon: Reconstruction loss (MSE)
        L_recon = F.mse_loss(output.output, output.input)
        
        # L_sparsity: Sparsity loss using L1 norm on gate activations (ReLU of gate pre-activations)
        # In reference: f_gate = ReLU(pi_gate), then lp_norm(f_gate, p=1)
        f_gate = F.relu(output.gates)  # Gate activations (post-ReLU)
        L_sparsity = torch.norm(f_gate, p=1.0, dim=-1).mean()
        
        # L_aux: Auxiliary reconstruction loss using gate activations with detached decoder
        with torch.no_grad():
            # Detach decoder weights and bias to stop gradients
            dec_weight_detached = self.decoder.weight.detach()
            dec_bias_detached = self.decoder_bias.detach()
        
        # Reconstruct using gate activations: x_hat_gate = f_gate @ W_dec^T + b_dec
        x_hat_gate = f_gate @ dec_weight_detached.T + dec_bias_detached
        L_aux = F.mse_loss(x_hat_gate, output.input)
        
        # Total loss: L_recon + alpha * L_sparsity + L_aux
        # Note: Reference implementation doesn't use separate mse_coeff, typically mse_coeff=1.0
        total_loss = self.mse_coeff * L_recon + self.sparsity_coeff * L_sparsity + self.aux_coeff * L_aux
        
        loss_dict = {
            "mse_loss": L_recon.detach().clone(),
            "sparsity_loss": L_sparsity.detach().clone(), 
            "aux_loss": L_aux.detach().clone(),
        }
        
        return SAELoss(loss=total_loss, loss_dict=loss_dict)

    @property
    def device(self):
        return next(self.parameters()).device


def hard_concrete(
    logits: torch.Tensor,
    beta: float,
    l: float,
    r: float,
    training: bool = True,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Sample from the Hard Concrete distribution using the reparameterization trick.
    Used for L0 regularization as described in https://arxiv.org/abs/1712.01312.
    """
    if training:
        # Sample u ~ Uniform(0, 1)
        u = torch.rand_like(logits)
        # Transform to Concrete variable s ~ Concrete(logits, beta) = Sigmoid((log(u) - log(1-u) + logits) / beta)
        s = torch.sigmoid((torch.log(u + epsilon) - torch.log(1.0 - u + epsilon) + logits) / beta)
        # Stretch s to (l, r)
        s_stretched = s * (r - l) + l
        # Apply hard threshold (clamp to [0, 1]) -> z ~ HardConcrete(logits, beta)
        z = torch.clamp(s_stretched, min=0.0, max=1.0)
    else:
        # Evaluation mode: use deterministic output
        s = torch.sigmoid(logits)
        s_stretched = s * (r - l) + l
        z = torch.clamp(s_stretched, min=0.0, max=1.0)

    return z


class GatedHardConcreteSAEConfig(SAEConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sae_type: SAEType = Field(default=SAEType.GATED_HARD_CONCRETE, description="Type of SAE (automatically set to gated_hard_concrete)")
    aux_coeff: float | None = None
    initial_beta: float = Field(0.5, description="Initial beta for Hard Concrete annealing")
    final_beta: float | None = Field(None, description="Final beta for Hard Concrete annealing")
    beta_annealing: bool = Field(False, description="Whether to anneal beta during training")
    hard_concrete_stretch_limits: tuple[float, float] = Field((-0.1, 1.1), description="Hard concrete stretch limits")
    
    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Ensure sae_type is set to gated_hard_concrete."""
        if isinstance(values, dict):
            values["sae_type"] = SAEType.GATED_HARD_CONCRETE
        return values


class GatedHardConcreteSAEOutput(GatedSAEOutput):
    """GatedHardConcrete SAE output that extends GatedSAEOutput with Hard Concrete parameters."""
    beta: float
    l: float
    r: float
    gate_logits: torch.Tensor  # The logits used for Hard Concrete sampling


class GatedHardConcreteSAE(BaseSAE):
    """
    Gated SAE with Hard Concrete sampling for gates instead of binary thresholding.
    Combines the GatedSAE architecture (separate gating and magnitude networks) 
    with Hard Concrete distribution sampling for better gradient flow.
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        sparsity_coeff: float | None = None,
        mse_coeff: float | None = None,
        aux_coeff: float | None = None,
        initial_beta: float = 0.5,
        stretch_limits: tuple[float, float] = (-0.1, 1.1),
    ):
        """
        Gated Sparse Autoencoder with Hard Concrete sampling and tied encoder weights.
        
        Args:
            input_size: dimensionality of input x.
            n_dict_components: number of sparse features (dictionary size).
            sparsity_coeff: coefficient for L0 sparsity loss
            mse_coeff: coefficient for MSE reconstruction loss
            aux_coeff: coefficient for auxiliary loss
            initial_beta: initial temperature for Hard Concrete distribution
            stretch_limits: stretch limits (l, r) for Hard Concrete
        """
        super().__init__()
        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0
        self.aux_coeff = aux_coeff if aux_coeff is not None else 1.0

        # Hard Concrete parameters
        self.register_buffer("beta", torch.tensor(initial_beta))
        self.l, self.r = stretch_limits
        assert self.l < 0.0 and self.r > 1.0, "stretch_limits must satisfy l < 0 and r > 1 for L0 penalty calculation"

        # Decoder bias for input centering
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))
        
        # Encoder (no bias, tied to decoder transpose)
        self.encoder = nn.Linear(input_size, n_dict_components, bias=False)
        
        # Magnitude network parameters
        self.r_mag = nn.Parameter(torch.zeros(n_dict_components))
        self.mag_bias = nn.Parameter(torch.zeros(n_dict_components))
        
        # Gating network parameters (produces logits for Hard Concrete)
        self.gate_bias = nn.Parameter(torch.zeros(n_dict_components))

        # Decoder (no bias, bias handled separately)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)
        
        # Initialize weights properly
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Default method for initializing GatedHardConcreteSAE weights.
        """
        # biases are initialized to zero
        nn.init.zeros_(self.decoder_bias)
        nn.init.zeros_(self.r_mag)
        nn.init.zeros_(self.gate_bias)
        nn.init.zeros_(self.mag_bias)

        # decoder weights are initialized to random unit vectors
        dec_weight = torch.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)
        # tie encoder weights to decoder transpose
        self.encoder.weight = nn.Parameter(dec_weight.clone().T)

    def forward(self, x: torch.Tensor):
        """
        Forward pass with Hard Concrete sampling for gates.
        """
        # Center input by subtracting decoder bias (treated as input mean)
        x_enc = self.encoder(x - self.decoder_bias)  # (batch, hidden_dim)

        # Gating network: produce logits for Hard Concrete distribution
        gate_logits = x_enc + self.gate_bias
        
        # Sample gates from Hard Concrete distribution
        current_beta = self.beta.item()
        f_gate = hard_concrete(gate_logits, beta=current_beta, l=self.l, r=self.r, training=self.training)

        # Magnitude network: exponential scaling + bias + ReLU
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = F.relu(pi_mag)

        # Combine gating and magnitude
        code = f_gate * f_mag
        
        # Decode
        recon = self.decoder(code) + self.decoder_bias  # (batch, input_dim)
        
        return GatedHardConcreteSAEOutput(
            input=x, 
            c=code, 
            output=recon, 
            logits=None,  # Not used in this architecture
            gates=gate_logits,  # gate logits (pre-sampling)
            magnitudes=f_mag,  # magnitude activations
            mask=f_gate,  # sampled gates (continuous in [0,1])
            beta=current_beta,
            l=self.l,
            r=self.r,
            gate_logits=gate_logits
        )

    def calc_l0_loss(self, logits: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """Calculate L0 penalty using Hard Concrete distribution."""
        safe_l = self.l if abs(self.l) > epsilon else -epsilon
        safe_r = self.r if abs(self.r) > epsilon else epsilon

        # Ensure the argument to log is positive
        log_arg = -safe_l / safe_r
        if log_arg <= 0:
            print(f"Warning: Invalid term for log in L0 penalty: -l/r = {log_arg:.4f}. Returning 0 penalty.")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        log_ratio = math.log(log_arg)
        penalty_per_element = torch.sigmoid(logits - self.beta * log_ratio)
        return penalty_per_element.sum(dim=-1).mean() / self.input_size

    def compute_loss(self, output: GatedHardConcreteSAEOutput) -> SAELoss:
        """
        Compute the Gated Hard Concrete SAE loss:
        L_recon + L_sparsity (L0) + L_aux
        """
        # L_recon: Reconstruction loss (MSE)
        L_recon = F.mse_loss(output.output, output.input)
        
        # L_sparsity: L0 sparsity loss using Hard Concrete distribution
        L_sparsity = self.calc_l0_loss(output.gate_logits)
        
        # L_aux: Auxiliary reconstruction loss using sampled gates with detached decoder
        with torch.no_grad():
            # Detach decoder weights and bias to stop gradients
            dec_weight_detached = self.decoder.weight.detach()
            dec_bias_detached = self.decoder_bias.detach()
        
        # Reconstruct using sampled gates: x_hat_gate = f_gate @ W_dec^T + b_dec
        x_hat_gate = output.mask @ dec_weight_detached.T + dec_bias_detached
        L_aux = F.mse_loss(x_hat_gate, output.input)
        
        # Total loss: L_recon + alpha * L_sparsity + L_aux
        total_loss = self.mse_coeff * L_recon + self.sparsity_coeff * L_sparsity + self.aux_coeff * L_aux
        
        loss_dict = {
            "mse_loss": L_recon.detach().clone(),
            "sparsity_loss": L_sparsity.detach().clone(), 
            "aux_loss": L_aux.detach().clone(),
        }
        
        return SAELoss(loss=total_loss, loss_dict=loss_dict)

    @property
    def device(self):
        return next(self.parameters()).device
