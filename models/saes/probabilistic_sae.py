# prob_gated_sae.py
import math
import torch
import torch.nn.functional as F
from typing import Any, Callable, Literal
from pydantic import Field, model_validator
from models.saes.base import SAEConfig, SAEOutput, SAELoss, BaseSAE
from utils.enums import SAEType

ACTIVATION_MAP: dict[str, Callable] = {
    "relu": F.relu,
    "softplus": F.softplus,
    "none": None,
}

class ProbGatedSAEConfig(SAEConfig):
    sae_type: SAEType = Field(default=SAEType.GATED, description="Type automatically set to probabilistic gated SAE")

    # Gating / budget
    rho: float = Field(0.05, description="Per-token expected activation fraction; K = rho * D")
    gate_temperature: float = Field(2.0, description="Sigmoid temperature beta for gate sharpness")
    budget_mode: Literal["hard", "soft"] = Field("hard", description="Enforce per-token budget exactly ('hard') or by penalty ('soft')")
    budget_coeff: float = Field(1e-2, description="Only used if budget_mode='soft': penalty weight for (sum(p)-K)^2")
    per_token_center: bool = Field(True, description="Center gate logits per token before thresholding")
    per_token_scale: bool = Field(False, description="Scale gate logits to unit std per token (optional)")

    # Encoders / decoder
    init_decoder_orthogonal: bool = Field(True, description="Orthonormal init for dictionary columns")
    tied_encoder_init: bool = Field(True, description="Copy decoder^T to magnitude encoder weight at init")
    magnitude_activation: str | None = Field("softplus", description="Nonnegative magnitude ('relu', 'softplus', or None)")
    decoder_bias: bool = Field(True, description="Use a learned decoder bias to center inputs")

    # Regularization
    l1_code_coeff: float = Field(0.0, description="L1 on expected codes E|c| = mean(p*m)")
    load_balance_coeff: float = Field(0.0, description="Tiny weight for coverage: entropy or MSE to K/D")
    load_balance_mode: Literal["entropy", "mse"] = Field("entropy", description="Form of load balance regularizer")

    # Stochastic exploration (optional)
    sample_prob: float = Field(0.0, description="Prob of sampling z~Bernoulli(p) during train forward; else use expectation p")

    @model_validator(mode="before")
    @classmethod
    def _fix_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.GATED
        return values


class ProbGatedSAEOutput(SAEOutput):
    p: torch.Tensor                 # probabilities (B, T, D) or (B, D)
    tau: torch.Tensor | None        # per-token threshold (B, T, 1) when hard budget
    logits: torch.Tensor            # centered gate logits
    magnitude: torch.Tensor         # nonnegative magnitudes
    c: torch.Tensor                 # expected codes = p * magnitude
    x_hat: torch.Tensor             # reconstruction


def _bisection_tau_for_sigmoid_sum(
    logits: torch.Tensor, K: int, beta: float, max_iter: int = 20
) -> torch.Tensor:
    """
    Find tau per token such that sum_d sigmoid((logits - tau)/beta) ~= K.
    logits: (B, T, D) or (B, D)
    Returns tau: (B, T, 1) or (B, 1)
    """
    lo = logits.amin(dim=-1, keepdim=True) - 20.0
    hi = logits.amax(dim=-1, keepdim=True) + 20.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p = torch.sigmoid((logits - mid) / beta)
        s = p.sum(dim=-1, keepdim=True)  # (B,T,1) or (B,1)
        lo = torch.where(s > K, mid, lo)
        hi = torch.where(s <= K, mid, hi)
    return (lo + hi) / 2


class ProbGatedSAE(BaseSAE):
    """
    Probabilistic Gated SAE with per-token budgeted probabilities.
    - Biasless gate encoder => purely input-dependent gates
    - Per-token 'hard' budget (preferred) or 'soft' penalty
    - Optional tiny load-balancing across features
    - Reconstruct from expected codes: c = p * magnitude
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        rho: float = 0.05,
        gate_temperature: float = 2.0,
        budget_mode: str = "hard",
        budget_coeff: float = 1e-2,
        per_token_center: bool = True,
        per_token_scale: bool = False,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        magnitude_activation: str | None = "softplus",
        decoder_bias: bool = True,
        l1_code_coeff: float = 0.0,
        load_balance_coeff: float = 0.0,
        load_balance_mode: str = "entropy",
        sample_prob: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_dict_components = n_dict_components

        # Budget & gating
        self.rho = float(rho)
        self.K = max(1, int(round(self.rho * n_dict_components)))
        self.beta = float(gate_temperature)
        self.budget_mode = budget_mode
        self.budget_coeff = float(budget_coeff)
        self.per_token_center = per_token_center
        self.per_token_scale = per_token_scale
        self.sample_prob = float(sample_prob)

        # Encoders
        self.encoder_ln = torch.nn.LayerNorm(input_size)
        self.gate_encoder = torch.nn.Linear(input_size, n_dict_components, bias=False)  # biasless!
        self.magnitude_encoder = torch.nn.Linear(input_size, n_dict_components, bias=True)
        self.magnitude_activation = ACTIVATION_MAP.get((magnitude_activation or "none").lower())

        # Decoder
        self.decoder = torch.nn.Linear(n_dict_components, input_size, bias=False)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(input_size)) if decoder_bias else None

        # Regularization knobs
        self.l1_code_coeff = float(l1_code_coeff)
        self.load_balance_coeff = float(load_balance_coeff)
        self.load_balance_mode = load_balance_mode

        # Inits
        if init_decoder_orthogonal:
            self.decoder.weight.data = torch.nn.init.orthogonal_(self.decoder.weight.data.T).T
        if tied_encoder_init:
            self.magnitude_encoder.weight.data.copy_(self.decoder.weight.data.T)

        # Mild init for gates
        torch.nn.init.normal_(self.gate_encoder.weight, mean=0.0, std=0.02)

    @property
    def dict_elements(self):
        # Normalize dictionary columns
        return F.normalize(self.decoder.weight, dim=0)

    def _center_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.decoder_bias is not None:
            x = x - self.decoder_bias
        return self.encoder_ln(x)

    def forward(self, x: torch.Tensor) -> ProbGatedSAEOutput:
        """
        x: (B, D_in) or (B, T, D_in)
        returns probabilities p (confidence) and expected codes c = p * magnitude
        """
        x = self._center_inputs(x)
        logits = self.gate_encoder(x)  # (B, T, D) or (B, D), biasless => purely input-dependent

        # optional per-token centering / scaling of logits (stabilizes threshold search)
        if self.per_token_center:
            logits = logits - logits.mean(dim=-1, keepdim=True)
        if self.per_token_scale:
            logits = logits / (logits.std(dim=-1, keepdim=True) + 1e-6)

        # compute probabilities p with either a hard per-token budget or a soft budget penalty
        tau = None
        if self.budget_mode == "hard":
            tau = _bisection_tau_for_sigmoid_sum(logits, self.K, self.beta)  # (B, T, 1) or (B, 1)
            p = torch.sigmoid((logits - tau) / self.beta)
        else:
            # soft budget: no tau, just p = sigmoid(logits/beta) and we add a penalty in the loss
            p = torch.sigmoid(logits / self.beta)

        # magnitude (nonnegative recommended)
        mag_pre = self.magnitude_encoder(x)
        magnitude = self.magnitude_activation(mag_pre) if self.magnitude_activation else mag_pre

        # expected codes (default; optionally sample z~Bernoulli(p) during training)
        if self.training and self.sample_prob > 0.0:
            if torch.rand(()) < self.sample_prob:
                # Straight-through Bernoulli sampling (optional exploration)
                z = (torch.rand_like(p) < p).float()
                c = z * magnitude
            else:
                c = p * magnitude
        else:
            c = p * magnitude

        # reconstruction with normalized dictionary
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)

        return ProbGatedSAEOutput(
            input=x, output=x_hat, c=c, p=p, tau=tau, logits=logits, magnitude=magnitude
        )

    def compute_loss(self, output: ProbGatedSAEOutput) -> SAELoss:
        # Reconstruction
        mse = F.mse_loss(output.output, output.input)

        # Magnitude/code regularization (expected L1 on codes)
        l1 = (output.p * output.magnitude).mean() if self.l1_code_coeff > 0.0 else torch.tensor(0.0, device=output.output.device)

        # Soft budget penalty (only if budget_mode='soft')
        budget_penalty = torch.tensor(0.0, device=output.output.device)
        if self.budget_mode == "soft" and self.budget_coeff > 0.0:
            sum_p = output.p.sum(dim=-1)  # (B, T) or (B,)
            target = float(self.K)
            budget_penalty = ((sum_p - target) ** 2).mean()

        # Load balance across features (tiny)
        lb = torch.tensor(0.0, device=output.output.device)
        if self.load_balance_coeff > 0.0:
            m_d = output.p.mean(dim=tuple(range(output.p.dim()-1)))  # average over batch/time -> (D,)
            if self.load_balance_mode == "entropy":
                lb = -(m_d.clamp_min(1e-12) * (m_d.clamp_min(1e-12)).log()).sum() / m_d.numel()
            else:  # "mse"
                target = self.K / self.n_dict_components
                lb = ((m_d - target) ** 2).mean()

        loss = mse + self.l1_code_coeff * l1 + self.budget_coeff * budget_penalty + self.load_balance_coeff * lb

        return SAELoss(
            loss=loss,
            loss_dict={
                "mse_loss": mse.detach().clone(),
                "l1_codes": l1.detach().clone(),
                "budget_penalty": budget_penalty.detach().clone(),
                "load_balance": lb.detach().clone(),
                "sum_p_mean": output.p.sum(dim=-1).mean().detach().clone(),    # monitor budget realization
                "p_entropy": (-(output.p.clamp_min(1e-12) * output.p.clamp_min(1e-12).log()).mean()).detach().clone(),
            },
        )

    @property
    def device(self):
        return next(self.parameters()).device
