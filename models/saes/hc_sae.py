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
    # z: torch.Tensor
    # beta: float
    # magnitude: torch.Tensor
    # gate_logits: torch.Tensor
    thresholds: torch.Tensor
    temperature: float
    soft_gates: torch.Tensor
    gates: torch.Tensor


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
        self.n_dict_components = n_dict_components
        self.input_size = input_size
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0
        self.encoder = torch.nn.Linear(input_size, n_dict_components, bias=False)
        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=False)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(input_size))
        self.threshold_mu = torch.nn.Parameter(torch.zeros(n_dict_components))
        self.threshold_log_var = torch.nn.Parameter(torch.ones(n_dict_components) * -2)
        self.register_buffer("beta", torch.tensor(1.0))
        self.temperature = 1.0

    def sample_thresholds(self, training=True):
        """Sample thresholds from learned distributions with temperature scaling."""
        if training:
            # Reparameterization trick
            std = torch.exp(0.5 * self.threshold_log_var)
            eps = torch.randn_like(std) * self.temperature
            thresholds = self.threshold_mu + eps * std
            
            # Ensure positive thresholds with softplus
            thresholds = F.softplus(thresholds)
        else:
            # Use mean during evaluation
            thresholds = F.softplus(self.threshold_mu)
        
        return thresholds

    def forward(self, x):
        x_centered = x - self.decoder_bias
        pre_acts = self.encoder(x_centered)
        
        # Sample thresholds from variational distribution
        thresholds = self.sample_thresholds(self.training)
        
        # Continuous relaxation during training, hard jump during inference
        if self.training:
            # Sigmoid-based soft threshold with temperature
            gate_logits = (pre_acts - thresholds) / self.temperature
            soft_gates = torch.sigmoid(gate_logits * 10)  # Sharper sigmoid
            
            # Straight-through estimator: hard forward, soft backward
            hard_gates = (pre_acts > thresholds).float()
            gates = hard_gates - soft_gates.detach() + soft_gates
            
            features = F.relu(pre_acts) * gates
        else:
            # Hard thresholding during inference
            soft_gates = (pre_acts > thresholds).float()
            features = F.relu(pre_acts) * soft_gates
        
        x_hat = self.decoder(features) + self.decoder_bias
        
        return HardConcreteSAEOutput(
            input=x,
            c=features,
            output=x_hat,
            gates=gates if self.training else (pre_acts > thresholds).float(),
            thresholds=thresholds,
            # gate_logits=gate_logits,
            soft_gates=soft_gates,
            temperature=self.temperature,
        )

    def compute_loss(self, output: HardConcreteSAEOutput) -> SAELoss:
        if self.training:
            # During training, use soft gates for differentiability
            # Expected L0 norm ≈ sum of gate probabilities
            l0_loss = output.soft_gates.sum(dim=-1).mean() / self.n_dict_components
        else:
            # During evaluation, use hard gates
            l0_loss = output.gates.sum(dim=-1).mean() / self.n_dict_components
        
        # expected_open_prob = torch.sigmoid(output.gate_logits - self.beta * math.log(-self.l / self.r))
        # rho = 0.005   # hardcoded for now
        # bce_elem = F.binary_cross_entropy(expected_open_prob, torch.full_like(expected_open_prob, rho), reduction="none")
        # revkl_loss = bce_elem.sum(dim=-1).mean()

        # q_bar = expected_open_prob.mean(dim=(0,1))                        # (D,)
        # lifetime_loss = F.binary_cross_entropy(q_bar, torch.full_like(q_bar, rho))
        # l0_loss = expected_open_prob.sum(dim=-1).mean() / self.input_size
        # # l1_loss = (output.z * output.magnitude.abs()).mean()
        # # kl_loss = kl_to_target(expected_open_prob, 0.005)
        # expected_K = (expected_open_prob * self.n_dict_components).sum(dim=-1).mean()

        # # sparsity_coeff = self.sparsity_coeff * cosine_ramp(self.train_progress, 0.1)
        # # sparsity_loss = 1 * revkl_loss + 0.1 * lifetime_loss
        # tau = 1e-3
        mse_loss = F.mse_loss(output.output, output.input)
        loss = self.sparsity_coeff * l0_loss + self.mse_coeff * mse_loss
        loss_dict = {
            "mse_loss": mse_loss.detach().clone(),
            "sparsity_loss": l0_loss.detach().clone(),
            # "sparsity_coeff": sparsity_coeff.detach().clone(),
            # "revkl_loss": revkl_loss.detach().clone(),
            # "lifetime_loss": lifetime_loss.detach().clone(),
            # "expected_K": expected_K.detach().clone(),
        }
        return SAELoss(loss=loss, loss_dict=loss_dict)

    @property
    def dict_elements(self):
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
