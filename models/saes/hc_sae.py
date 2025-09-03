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
    beta: float
    magnitude: torch.Tensor
    gate_logits: torch.Tensor


def kl_to_target(
    p_open: torch.Tensor,
    rho: float = 0.005,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL(q || rho) for Bernoulli gates, elementwise over p_open, reduced to a scalar.

    Args:
        p_open: Tensor of open probabilities q in [0,1], any shape.
                (For Hard-Concrete, use the expected-open surrogate; see helper below.)
        rho:    Target open probability in (0,1), typically K / n_dict.
        reduction: "mean" | "sum" | "none".
        eps:   Small epsilon for numerical stability.

    Returns:
        Scalar loss if reduction != "none"; else same shape as p_open.
    """
    # clamp both q and ρ away from {0,1} to avoid log(0)
    q = p_open.clamp(min=eps, max=1.0 - eps)
    rho_t = torch.as_tensor(rho, dtype=q.dtype, device=q.device).clamp(min=eps, max=1.0 - eps)

    # KL(q||ρ) = q log(q/ρ) + (1-q) log((1-q)/(1-ρ))
    kl = q * (torch.log(q) - torch.log(rho_t)) + (1.0 - q) * (torch.log(1.0 - q) - torch.log(1.0 - rho_t))

    if reduction == "none":
        return kl
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl.mean()


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
        self.encoder_layer_norm = torch.nn.LayerNorm(input_size, bias=False)
        self.n_dict_components = n_dict_components
        self.input_size = input_size
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0
        self.gate_encoder = torch.nn.Linear(input_size, n_dict_components, bias=False)
        self.magnitude_encoder = torch.nn.Linear(input_size, n_dict_components, bias=True)
        # self.r_mag = torch.nn.Parameter(torch.zeros(n_dict_components))
        # self.mag_bias = torch.nn.Parameter(torch.zeros(n_dict_components))
        self.magnitude_activation = get_activation(magnitude_activation)
        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=True)

        # Register beta as a buffer to allow updates during training without being a model parameter
        self.register_buffer("beta", torch.tensor(initial_beta, dtype=torch.float32, device='cpu'))
        self.register_buffer("train_progress", torch.tensor(0.0, dtype=torch.float32, device='cpu'))
        self.l, self.r = stretch_limits
        assert self.l < 0.0 and self.r > 1.0, "stretch_limits must satisfy l < 0 and r > 1 for L0 penalty calculation"

        if init_decoder_orthogonal:
            self.decoder.weight.data = torch.nn.init.orthogonal_(self.decoder.weight.data.T).T

        with torch.no_grad():
            self.gate_encoder.weight.data.copy_(self.decoder.weight.data.T)

    def hard_concrete(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from the Hard Concrete distribution using the reparameterization trick.
        Used for L0 regularization as described in https://arxiv.org/abs/1712.01312.
        """
        epsilon = 1e-6
        if self.training:
            u = torch.rand_like(logits)
            s = torch.sigmoid((torch.log(u + epsilon) - torch.log(1.0 - u + epsilon) + logits) / self.beta)
        else:
            s = torch.sigmoid(logits / self.beta)
        
        s_stretched = s * (self.r - self.l) + self.l
        z_hard = torch.clamp(s_stretched, min=0.0, max=1.0)
        z = z_hard + (s_stretched - z_hard).detach()  
        return z

    def forward(self, x: torch.Tensor) -> HardConcreteSAEOutput:
        x_centered = self.encoder_layer_norm(x - self.decoder.bias)
        gate_logits = self.gate_encoder(x_centered.detach())
        z = self.hard_concrete(gate_logits)
        magnitude = self.magnitude_activation(self.magnitude_encoder(x_centered))
        c = z * magnitude
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder.bias)
        return HardConcreteSAEOutput(input=x, c=c, output=x_hat, magnitude=magnitude, beta=self.beta, z=z, gate_logits=gate_logits)

    def compute_loss(self, output: HardConcreteSAEOutput) -> SAELoss:
        expected_open_prob = torch.sigmoid(output.gate_logits - self.beta * math.log(-self.l / self.r))
        # rho = 0.005   # hardcoded for now
        # bce_elem = F.binary_cross_entropy(expected_open_prob, torch.full_like(expected_open_prob, rho), reduction="none")
        # revkl_loss = bce_elem.sum(dim=-1).mean()

        # q_bar = expected_open_prob.mean(dim=(0,1))                        # (D,)
        # lifetime_loss = F.binary_cross_entropy(q_bar, torch.full_like(q_bar, rho))
        l0_loss = expected_open_prob.sum(dim=-1).mean() / self.input_size
        # l1_loss = (output.z * output.magnitude.abs()).mean()
        # kl_loss = kl_to_target(expected_open_prob, 0.005)
        expected_K = (expected_open_prob * self.n_dict_components).sum(dim=-1).mean()

        # sparsity_coeff = self.sparsity_coeff * cosine_ramp(self.train_progress, 0.1)
        # sparsity_loss = 1 * revkl_loss + 0.1 * lifetime_loss
        tau = 1e-3
        mse_loss = F.mse_loss(output.output, output.input)
        loss = self.sparsity_coeff * l0_loss + self.mse_coeff * mse_loss
        loss_dict = {
            "mse_loss": mse_loss.detach().clone(),
            "sparsity_loss": l0_loss.detach().clone(),
            # "sparsity_coeff": sparsity_coeff.detach().clone(),
            # "revkl_loss": revkl_loss.detach().clone(),
            # "lifetime_loss": lifetime_loss.detach().clone(),
            "expected_K": expected_K.detach().clone(),
        }
        return SAELoss(loss=loss, loss_dict=loss_dict)

    @property
    def dict_elements(self):
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
