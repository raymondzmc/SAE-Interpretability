import math
import torch
import torch.nn.functional as F
from pydantic import Field, model_validator
from typing import Any, Callable
from models.saes.base import SAEConfig, SAEOutput, SAELoss, BaseSAE
from utils.enums import SAEType


ACTIVATION_MAP: dict[str, Callable] = {
    'relu': F.relu,
    'softplus': F.softplus,
    'none': None,
}


class HardConcreteSAEConfig(SAEConfig):
    sae_type: SAEType = Field(default=SAEType.HARD_CONCRETE, description="Type of SAE (automatically set to hard_concrete)")
    initial_beta: float = Field(0.5, description="Initial beta for Hard Concrete annealing")
    final_beta: float | None = Field(None, description="Final beta for Hard Concrete annealing")
    beta_annealing: bool = Field(False, description="Whether to anneal beta during training")
    hard_concrete_stretch_limits: tuple[float, float] = Field((-0.1, 1.1), description="Hard concrete stretch limits")
    tied_encoder_init: bool = Field(True, description="Whether to tie the encoder weights to the decoder weights")
    magnitude_activation: str | None = Field("relu", description="Activation function for magnitude ('relu', 'softplus', etc.) or None")
    coefficient_threshold: float = Field(0.0, description="Threshold for the coefficients during inference")
    
    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Ensure sae_type is set to hard_concrete."""
        if isinstance(values, dict):
            values["sae_type"] = SAEType.HARD_CONCRETE
        return values


class HardConcreteSAEOutput(SAEOutput):
    """HardConcrete SAE output that extends SAEOutput with additional parameters."""
    z: torch.Tensor
    beta: float
    l: float
    r: float
    magnitude: torch.Tensor
    gate_logits: torch.Tensor


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
        # Use the clamped stretched sigmoid mean approximation
        s = torch.sigmoid(logits / beta)
        s_stretched = s * (r - l) + l
        z = torch.clamp(s_stretched, min=0.0, max=1.0)

    return z


class HardConcreteSAE(BaseSAE):
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
        sparsity_coeff: float | None = None,
        mse_coeff: float | None = None,
        stretch_limits: tuple[float, float] = (-0.1, 1.1), # Stretch limits for Hard Concrete
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        magnitude_activation: str | None = "relu",
        coefficient_threshold: float = 0.0,
    ):
        """Initialize the SAE with Hard Concrete gates.

        Args:
            input_size: Dimensionality of input data
            n_dict_components: Number of dictionary components (and Hard Concrete gates)
            initial_beta: Initial temperature for the Hard Concrete distribution. This will be annealed during training.
            stretch_limits: Stretch limits (l, r) for Hard Concrete. Must have l < 0 and r > 1.
            init_decoder_orthogonal: Initialize the decoder weights to be orthonormal
            tied_encoder_init: Tie the encoder weights to the decoder weights
            magnitude_activation: Activation function for the magnitude
            coefficient_threshold: Threshold for the coefficients during inference
        """
        super().__init__()
        self.n_dict_components = n_dict_components
        self.input_size = input_size
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0
        self.magnitude_activation = ACTIVATION_MAP.get((magnitude_activation or "none").lower())
        self.coefficient_threshold = coefficient_threshold

        self.gate_encoder = torch.nn.Linear(input_size, n_dict_components, bias=True)
        self.magnitude_encoder = torch.nn.Linear(input_size, n_dict_components, bias=True)

        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=True)

        # Register beta as a buffer to allow updates during training without being a model parameter
        # Create on CPU first, will be moved to correct device by .to() call
        self.register_buffer("beta", torch.tensor(initial_beta, dtype=torch.float32, device='cpu'))
        self.l, self.r = stretch_limits
        assert self.l < 0.0 and self.r > 1.0, "stretch_limits must satisfy l < 0 and r > 1 for L0 penalty calculation"

        if init_decoder_orthogonal:
            self.decoder.weight.data = torch.nn.init.orthogonal_(self.decoder.weight.data.T).T

        if tied_encoder_init:
            self.encoder.weight.data.copy_(self.decoder.weight.data.T)

    def forward(self, x: torch.Tensor) -> HardConcreteSAEOutput:
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
        # Ensure input is on same device as module
        module_device = next(self.parameters()).device
        if x.device != module_device:
            x = x.to(module_device)
        
        x_centered = x - self.decoder_bias

        # Get encoder output
        gate_logits = self.gate_encoder(x_centered)
        magnitude = self.magnitude_encoder(x_centered)

        current_beta = self.beta.item() # Get current beta value from buffer
        z = hard_concrete(gate_logits, beta=current_beta, l=self.l, r=self.r, training=self.training) # Shape: same as magnitude
        
        # Apply threshold to z during evaluation
        if not self.training:
            z = torch.where(torch.abs(z) >= self.coefficient_threshold, z, 0.0)

        if self.magnitude_activation is not None:
            magnitude = self.magnitude_activation(magnitude)

        # Combine gate and magnitude for final coefficients
        coefficients = z * magnitude

        # Reconstruct using the dictionary elements and final coefficients
        x_hat = F.linear(coefficients, self.dict_elements, bias=self.decoder_bias)

        return HardConcreteSAEOutput(input=x, c=coefficients, output=x_hat, logits=None, magnitude=magnitude, beta=current_beta, l=self.l, r=self.r, z=z, gate_logits=gate_logits)

    def calc_l0_loss(self, gate_logits: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        safe_l = self.l if abs(self.l) > epsilon else -epsilon
        safe_r = self.r if abs(self.r) > epsilon else epsilon

        # Ensure the argument to log is positive
        log_arg = -safe_l / safe_r
        if log_arg <= 0:
            print(f"Warning: Invalid term for log in L0 penalty: -l/r = {log_arg:.4f}. Returning 0 penalty.")
            # Return a tensor with the correct device and dtype
            return torch.tensor(0.0, device=gate_logits.device, dtype=gate_logits.dtype)

        log_ratio = math.log(log_arg)
        penalty_per_element = torch.sigmoid(gate_logits - self.beta * log_ratio)
        return penalty_per_element.sum(dim=-1).mean() / self.input_size

    def compute_loss(self, output: HardConcreteSAEOutput) -> SAELoss:
        """Compute the loss for the HardConcreteSAE.
        
        Args:
            output: The output of the HardConcreteSAE.
        """
        sparsity_loss = self.calc_l0_loss(gate_logits=output.gate_logits)
        mse_loss = F.mse_loss(output.output, output.input)
        loss = self.sparsity_coeff * sparsity_loss + self.mse_coeff * mse_loss
        return SAELoss(
            loss=loss,
            loss_dict={
                "mse_loss": mse_loss.detach().clone(),
                "sparsity_loss": sparsity_loss.detach().clone(),
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
