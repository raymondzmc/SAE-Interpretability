import math
import torch
import torch.nn.functional as F
from pydantic import Field
from models.saes.base import SAEConfig, SAEOutput, SAELoss, BaseSAE
from utils.enums import SAEType
from models.saes.activations import get_activation


class HardConcreteSAEConfig(SAEConfig):
    sae_type: SAEType = Field(default=SAEType.HARD_CONCRETE, description="Type of SAE (automatically set to hard_concrete)")
    beta_annealing: bool = Field(False, description="Whether to anneal beta during training")
    initial_beta: float = Field(0.5, description="Initial beta for Hard Concrete annealing")
    final_beta: float | None = Field(None, description="Final beta for Hard Concrete annealing")
    hard_concrete_stretch_limits: tuple[float, float] = Field((-0.1, 1.1), description="Hard concrete stretch limits")
    magnitude_activation: str | None = Field("softplus0", description="Activation function for magnitude ('relu', 'softplus', etc.) or None")


class HardConcreteSAEOutput(SAEOutput):
    """HardConcrete SAE output that extends SAEOutput with additional parameters."""
    z: torch.Tensor
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


def cosine_ramp(progress: torch.Tensor, end: float) -> torch.Tensor:
    # progress in [0,1]; ramp in [0,end]
    p = (progress / end).clamp(0, 1)
    return 0.5 * (1 - torch.cos(torch.pi * p))


class HardConcreteSAE(BaseSAE):
    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        initial_beta: float, # Initial temperature for Hard Concrete
        sparsity_coeff: float | None = None,
        mse_coeff: float | None = None,
        stretch_limits: tuple[float, float] = (-0.1, 1.1), # Stretch limits for Hard Concrete
        init_decoder_orthogonal: bool = True,
        magnitude_activation: str | None = "softplus0",
    ):
        super().__init__()
        self.n_dict_components = n_dict_components
        self.input_size = input_size
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0
        self.encoder = torch.nn.Linear(input_size, n_dict_components, bias=False)
        self.layer_norm = torch.nn.LayerNorm(input_size, elementwise_affine=False)
        self.magnitude_encoder = torch.nn.Linear(input_size, n_dict_components, bias=True)
        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=False)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(input_size))
        self.register_buffer("beta", torch.tensor(initial_beta, dtype=torch.float32, device='cpu'))
        self.l, self.r = stretch_limits
        assert self.l < 0.0 and self.r > 1.0, "stretch_limits must satisfy l < 0 and r > 1 for L0 penalty calculation"

        if init_decoder_orthogonal:
            self.decoder.weight.data = torch.nn.init.orthogonal_(self.decoder.weight.data.T).T

        self.encoder.weight.data.copy_(self.decoder.weight.data.T)
        self.magnitude_encoder.weight.data.copy_(self.decoder.weight.data.T)


    def forward(self, x):
        x_centered = x - self.decoder_bias
        x_normalized = self.layer_norm(x_centered)
        gate_logits = self.encoder(x_normalized)
        magnitude = self.magnitude_encoder(x_normalized)
        z = hard_concrete(gate_logits, beta=self.beta, l=self.l, r=self.r, training=self.training)
        c = z * magnitude
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder.bias)
        
        return HardConcreteSAEOutput(
            input=x,
            c=c,
            output=x_hat,
            z=z,
            gate_logits=gate_logits,
            magnitude=magnitude,
        )
    
    def calc_l0_loss(self, logits: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        safe_l = self.l if abs(self.l) > epsilon else -epsilon
        safe_r = self.r if abs(self.r) > epsilon else epsilon

        # Ensure the argument to log is positive
        log_arg = -safe_l / safe_r
        if log_arg <= 0:
            print(f"Warning: Invalid term for log in L0 penalty: -l/r = {log_arg:.4f}. Returning 0 penalty.")
            # Return a tensor with the correct device and dtype
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        log_ratio = math.log(log_arg)
        penalty_per_element = torch.sigmoid(logits - self.beta * log_ratio)
        return penalty_per_element.sum(dim=-1).mean() / self.input_size

    def compute_loss(self, output: HardConcreteSAEOutput) -> SAELoss:
        sparsity_loss = self.calc_l0_loss(logits=output.logits)
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
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
