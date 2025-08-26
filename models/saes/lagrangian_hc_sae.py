import math
import torch
import torch.nn.functional as F
from typing import Any, Callable
from pydantic import Field, model_validator
from models.saes.base import SAEConfig, SAEOutput, SAELoss, BaseSAE
from utils.enums import SAEType


ACTIVATION_MAP: dict[str, Callable] = {
    'relu': F.relu,
    'softplus': F.softplus,
    'none': None,
}


class LagrangianHardConcreteSAEConfig(SAEConfig):
    sae_type: SAEType = Field(default=SAEType.LAGRANGIAN_HARD_CONCRETE, description="Type of SAE (automatically set to hard_concrete)")
    initial_beta: float = Field(0.5, description="Initial beta for Hard Concrete annealing")
    final_beta: float | None = Field(None, description="Final beta for Hard Concrete annealing")
    beta_annealing: bool = Field(False, description="Whether to anneal beta during training")
    hard_concrete_stretch_limits: tuple[float, float] = Field((-0.1, 1.1), description="Hard concrete stretch limits")
    tied_encoder_init: bool = Field(True, description="Whether to tie the encoder weights to the decoder weights")
    magnitude_activation: str | None = Field("relu", description="Activation function for magnitude ('relu', 'softplus', 'gelu', etc.) or None")
    coefficient_threshold: float = Field(0.0, description="Threshold for the coefficients during inference")
    initial_alpha: float = Field(1.0, description="Initial alpha for Lagrangian dual-ascent controller")
    alpha_lr: float = Field(1e-2, description="Learning rate for alpha")
    rho: float = Field(0.05, description="Target sparsity level for Lagrangian dual-ascent controller")
    mu: float = Field(1.0, description="Regularization parameter for the sparsity loss")
    bias_l2_coeff: float = Field(1e-3, description="L2 regularization for the gate encoder bias")
    lb_coeff: float = Field(1e-3, description="Load balance regularization")
    
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
    beta: float
    l: float
    r: float
    magnitude: torch.Tensor
    gate_logits: torch.Tensor
    p_open: torch.Tensor
    alpha: torch.Tensor


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

    Produces samples in [0, 1] via a stretched, hard-thresholded sigmoid transformation
    of a log-uniform variable.

    Args:
        logits: Logits parameter (alpha) for the distribution. Shape: (*, num_features)
        beta: Temperature parameter. Controls the sharpness of the distribution.
        l: Lower bound of the stretch interval.
        r: Upper bound of the stretch interval.
        training: Whether in training mode (stochastic) or eval mode (deterministic).
        epsilon: Small constant for numerical stability.

    Returns:
        z: Sampled values (hard-thresholded in [0, 1]). Shape: (*, num_features)
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
        initial_alpha: float = 1.0,
        alpha_lr: float = 1e-2,
        mse_coeff: float | None = None,
        bias_l2_coeff: float = 1e-3, # L2 regularization for the gate encoder bias
        lb_coeff: float = 1e-3, # Load balance regularization
        rho: float = 0.05,
        mu: float = 1.0,
        stretch_limits: tuple[float, float] = (-0.1, 1.1), # Stretch limits for Hard Concrete
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        magnitude_activation: str | None = None,
        coefficient_threshold: float = 0.0,
    ):
        """Initialize the SAE with Hard Concrete gates.

        Args:
            input_size: Dimensionality of input data
            n_dict_components: Number of dictionary components (and Hard Concrete gates)
            initial_beta: Initial temperature for the Hard Concrete distribution. This will be annealed during training.
            initial_alpha: Initial alpha for Lagrangian dual-ascent controller
            alpha_lr: Learning rate for alpha
            mse_coeff: Coefficient for MSE loss term
            bias_l2_coeff: L2 regularization for the gate encoder bias
            lb_coeff: Load balance regularization
            rho: Target sparsity level for Lagrangian controller
            stretch_limits: Stretch limits (l, r) for Hard Concrete. Must have l < 0 and r > 1.
            init_decoder_orthogonal: Initialize the decoder weights to be orthonormal
            tied_encoder_init: Tie the encoder weights to the decoder weights
            magnitude_activation: Activation function name ('relu', 'softplus', 'gelu', etc.) or None for no activation
            coefficient_threshold: Threshold for the coefficients during inference
        """
        super().__init__()
        self.n_dict_components = n_dict_components
        self.input_size = input_size
        self.rho = rho
        self.mse_coeff = mse_coeff or 1.0
        self.bias_l2_coeff = bias_l2_coeff
        self.lb_coeff = lb_coeff
        self.alpha_lr = alpha_lr
        self.coefficient_threshold = coefficient_threshold

        self.encoder_layer_norm = torch.nn.LayerNorm(input_size)
        self.gate_encoder = torch.nn.Linear(input_size, n_dict_components, bias=True)
        self.magnitude_encoder = torch.nn.Linear(input_size, n_dict_components, bias=True)
        self.magnitude_activation = ACTIVATION_MAP.get((magnitude_activation or "none").lower())

        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=False)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(input_size))
        
        self.l, self.r = stretch_limits
        assert self.l < 0.0 and self.r > 1.0, "stretch_limits must satisfy l < 0 and r > 1 for L0 penalty calculation"
        self.register_buffer("log_neg_l_over_r", torch.tensor(math.log(-self.l / self.r), dtype=torch.float32, device='cpu'))
        self.register_buffer("beta", torch.tensor(initial_beta, dtype=torch.float32, device='cpu'))
        # self.register_buffer("alpha", torch.tensor(initial_alpha, dtype=torch.float32, device='cpu'))
        self.register_buffer("alpha", torch.full((n_dict_components,), initial_alpha, dtype=torch.float32, device='cpu'))
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32, device='cpu'))

        logit_rho = math.log(self.rho / (1 - self.rho))
        bias0 = logit_rho + float(self.beta) * float(self.log_neg_l_over_r)
        torch.nn.init.constant_(self.gate_encoder.bias, bias0)
        torch.nn.init.normal_(self.gate_encoder.weight, mean=0.0, std=0.02)

        if init_decoder_orthogonal:
            self.decoder.weight.data = torch.nn.init.orthogonal_(self.decoder.weight.data.T).T

        if tied_encoder_init:
            self.magnitude_encoder.weight.data.copy_(self.decoder.weight.data.T)

    def forward(self, x: torch.Tensor) -> LagrangianHardConcreteSAEOutput:
        """
        Forward pass through the SAE.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) or (batch_size, seq_len, input_size) for NLP
        
        For input-dependent gates:
            - Encoder outputs logits and pre-magnitude values
            - Gates are sampled from Hard Concrete using input-dependent logits
            
        For parameter-based gates (input-independent):
            - Encoder outputs only pre-magnitude values  
            - Gates are sampled from Hard Concrete using global learnable parameter logits
            - Same gate values are used across all positions for NLP models
            
        Returns:
            x_hat: Reconstructed input.
            c: Final coefficients (gate * magnitude).
            logits: The logits used for the gates (used for L0 penalty calculation).
            beta: The beta value used for sampling.
            l: The lower stretch limit used.
            r: The upper stretch limit used.
        """
        x_centered = self.encoder_layer_norm(x - self.decoder_bias) 

        # Get gate logits and magnitude from separate encoders
        gate_logits = self.gate_encoder(x_centered) 
        magnitude_pre = self.magnitude_encoder(x_centered)
        
        current_beta = self.beta.item() # Get current beta value from buffer
        z = hard_concrete(gate_logits, beta=current_beta, l=self.l, r=self.r, training=self.training) # Shape: same as magnitude
        
        # Apply threshold to z during evaluation
        if not self.training:
            z = torch.where(torch.abs(z) >= self.coefficient_threshold, z, 0.0)

        # Apply activation to magnitude if specified
        if self.magnitude_activation is not None:
            magnitude = self.magnitude_activation(magnitude_pre)
        else:
            magnitude = magnitude_pre

        # Combine gate and magnitude for final coefficients
        coefficients = z * magnitude

        # Reconstruct using the dictionary elements and final coefficients
        x_hat = F.linear(coefficients, self.dict_elements, bias=self.decoder_bias)

        p_open = torch.sigmoid(gate_logits - float(self.beta.item()) * float(self.log_neg_l_over_r))

        return LagrangianHardConcreteSAEOutput(input=x, c=coefficients, output=x_hat, logits=None, magnitude=magnitude, beta=current_beta, l=self.l, r=self.r, z=z, gate_logits=gate_logits, p_open=p_open, alpha=self.alpha)

    def compute_loss(self, output: LagrangianHardConcreteSAEOutput) -> SAELoss:
        """Compute the loss for the HardConcreteSAE.
        
        Args:
            output: The output of the HardConcreteSAE.
        """
        p = output.p_open
        m_d = p.mean(dim=(0, 1))
        c = m_d - self.rho
        sparsity_loss = (self.alpha.detach() * c).sum() + 0.5 * self.mu.item() * (c ** 2).mean()

        # rho_hat = output.p_open.mean()
        # sparsity_loss = self.alpha.detach() * (rho_hat - self.rho) + 0.5 * self.mu.item() * (rho_hat - self.rho)**2
        mse_loss = F.mse_loss(output.output, output.input)
        bias_l2 = self.bias_l2_coeff * (self.gate_encoder.bias.pow(2).sum())
        

        m = m_d / (m_d.sum() + 1e-8)
        load_balance_loss = self.lb_coeff * (m * (m.clamp_min(1e-12).log() + math.log(self.n_dict_components))).sum()
        loss = sparsity_loss + self.mse_coeff * mse_loss + bias_l2 + self.lb_coeff * load_balance_loss
        return SAELoss(
            loss=loss,
            loss_dict={
                "mse_loss": mse_loss.detach().clone(),
                "sparsity_loss": sparsity_loss.detach().clone(),
                "bias_l2": bias_l2.detach().clone(),
                "load_balance_loss": load_balance_loss.detach().clone(),
            },
        )

    @property
    def dict_elements(self):
        """Dictionary elements are simply the normalized decoder weights."""
        # Normalize columns (dim=0) of the weight matrix
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
