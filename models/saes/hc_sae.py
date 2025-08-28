import math
import torch
import torch.nn.functional as F
from pydantic import Field
from typing import Callable
from models.saes.base import SAEConfig, SAEOutput, SAELoss, BaseSAE
from utils.enums import SAEType


ACTIVATION_MAP: dict[str, Callable] = {
    'relu': F.relu,
    'softplus': F.softplus,
    'none': torch.nn.Identity(),
}


class HardConcreteSAEConfig(SAEConfig):
    sae_type: SAEType = Field(default=SAEType.HARD_CONCRETE, description="Type of SAE (automatically set to hard_concrete)")
    beta_annealing: bool = Field(False, description="Whether to anneal beta during training")
    initial_beta: float = Field(0.5, description="Initial beta for Hard Concrete annealing")
    final_beta: float | None = Field(None, description="Final beta for Hard Concrete annealing")
    hard_concrete_stretch_limits: tuple[float, float] = Field((-0.1, 1.1), description="Hard concrete stretch limits")
    magnitude_activation: str | None = Field("relu", description="Activation function for magnitude ('relu', 'softplus', etc.) or None")


class HardConcreteSAEOutput(SAEOutput):
    """HardConcrete SAE output that extends SAEOutput with additional parameters."""
    z: torch.Tensor
    beta: float
    magnitude: torch.Tensor
    gate_logits: torch.Tensor


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
        magnitude_activation: str | None = "relu",
    ):
        super().__init__()
        self.n_dict_components = n_dict_components
        self.input_size = input_size
        self.sparsity_coeff = sparsity_coeff if sparsity_coeff is not None else 1.0
        self.mse_coeff = mse_coeff if mse_coeff is not None else 1.0
        # self.encoder = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, n_dict_components // 10, bias=False),
        #     torch.nn.Softplus(),
        #     torch.nn.Linear(n_dict_components // 10, n_dict_components, bias=False),
        # )

        self.magnitude_activation = ACTIVATION_MAP.get((magnitude_activation or "none").lower())
        self.r_mag = torch.nn.Parameter(torch.zeros(n_dict_components))
        self.magnitude_bias = torch.nn.Parameter(torch.zeros(n_dict_components))

        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=True)

        # Register beta as a buffer to allow updates during training without being a model parameter
        self.register_buffer("beta", torch.tensor(initial_beta, dtype=torch.float32, device='cpu'))
        self.l, self.r = stretch_limits
        assert self.l < 0.0 and self.r > 1.0, "stretch_limits must satisfy l < 0 and r > 1 for L0 penalty calculation"

        if init_decoder_orthogonal:
            self.decoder.weight.data = torch.nn.init.orthogonal_(self.decoder.weight.data.T).T

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
        z = torch.clamp(s_stretched, min=0.0, max=1.0)
        return z

    def forward(self, x: torch.Tensor) -> HardConcreteSAEOutput:
        x_centered = x - self.decoder.bias
        logits = F.linear(x_centered, self.dict_elements.t())
        z = self.hard_concrete(logits)
        magnitude = self.magnitude_activation(self.r_mag.exp() * logits + self.magnitude_bias)
        c = z * magnitude
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder.bias)
        return HardConcreteSAEOutput(input=x, c=c, output=x_hat, magnitude=magnitude, beta=self.beta, z=z, gate_logits=logits)

    def compute_loss(self, output: HardConcreteSAEOutput) -> SAELoss:
        # log_ratio = math.log(-self.r / self.l)
        # sparsity_loss = torch.sigmoid(output.gate_logits - self.beta * log_ratio).mean()
        sparsity_loss = (output.z * output.magnitude.abs()).mean()
        mse_loss = F.mse_loss(output.output, output.input)
        loss = self.sparsity_coeff * sparsity_loss + self.mse_coeff * mse_loss
        loss_dict = {
            "mse_loss": mse_loss.detach().clone(),
            "sparsity_loss": sparsity_loss.detach().clone(),
        }
        return SAELoss(loss=loss, loss_dict=loss_dict)

    @property
    def dict_elements(self):
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
