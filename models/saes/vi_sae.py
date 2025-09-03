# vi_topk_sae.py
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any
from pydantic import Field, ConfigDict, model_validator
from jaxtyping import Float

from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from utils.enums import SAEType


# ---------- utilities ----------
def _topk_st(scores: torch.Tensor, k: int, tau_st: float = 0.5):
    if k <= 0:
        hard = torch.zeros_like(scores); soft = torch.zeros_like(scores)
        st = hard + (soft - soft.detach()); return st, hard, soft
    if k >= scores.size(-1):
        hard = torch.ones_like(scores); soft = torch.ones_like(scores)
        st = hard + (soft - soft.detach()); return st, hard, soft

    topk = torch.topk(scores, k, dim=-1)
    hard = torch.zeros_like(scores)
    hard.scatter_(-1, topk.indices, 1.0)

    kth = topk.values[..., -1].unsqueeze(-1)
    soft = torch.sigmoid((scores - kth) / tau_st)
    st = hard + (soft - soft.detach())
    return st, hard, soft


def _sample_binary_concrete(eta: torch.Tensor, temp: float = 0.67, training: bool = True) -> torch.Tensor:
    """
    Binary-Concrete reparameterized sample in (0,1):
    s = sigmoid((eta + log(u) - log(1-u)) / temp)
    """
    if training:
        u = torch.rand_like(eta).clamp_(1e-6, 1 - 1e-6)
        s = torch.sigmoid((eta + torch.log(u) - torch.log1p(-u)) / max(temp, 1e-6))
        return s
    else:
        # Deterministic proxy (mean of binary-concrete ~ sigmoid(eta/temp))
        return torch.sigmoid(eta / max(temp, 1e-6))


def _kl_bern_bern(p: torch.Tensor, rho: float) -> torch.Tensor:
    """
    KL(Bern(p) || Bern(rho)) elementwise (stable clamps).
    """
    eps = 1e-6
    p = p.clamp(eps, 1 - eps)
    rho_t = torch.tensor(rho, device=p.device).clamp(eps, 1 - eps)
    return p * torch.log(p / rho_t) + (1 - p) * torch.log((1 - p) / (1 - rho_t))


# ---------- config & output ----------
class VITopKSAEConfig(SAEConfig):
    """
    Top-K SAE with variational Bernoulli gates.
    - Amortized Bernoulli q(z|x) with binary-Concrete sampling for gradients.
    - KL to Bernoulli prior of rate rho (default rho = K/C).
    - Dual ascent penalty enforces E[sum_i p_i(x)] ≈ K (optional).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    sae_type: SAEType = Field(default=SAEType.TOPK, description="Kept as TOPK for compatibility")
    k: int = Field(..., description="Number of active features per sample")
    tied_encoder_init: bool = Field(True, description="Initialize encoder = decoder.T")
    use_pre_relu: bool = Field(True, description="Apply ReLU before Top-K")

    # Gate/logit head & scoring
    vi_temp: float = Field(0.67, description="Binary-Concrete temperature")
    score_mix_lambda: float = Field(0.5, description="Blend between log-magnitude and probability evidence")
    st_tau: float = Field(0.5, description="Softness for straight-through Top-K")

    # Loss coeffs
    mse_coeff: float = Field(1.0, description="Reconstruction loss coefficient")
    kl_coeff: float = Field(1e-3, description="KL coefficient against prior Bernoulli(rho)")
    card_coeff: float = Field(0.05, description="Soft-cardinality calibration on soft mask")

    # Prior & dual ascent for expected-K
    prior_rate: float | None = Field(None, description="rho for prior; default K/C if None")
    dual_lr: float = Field(0.0, description="Dual ascent step size for lambda (0 disables)")
    dual_init: float = Field(0.0, description="Initial lambda for expected-K constraint")

    # Aux-K (optional)
    aux_k: int | None = Field(None, description="Auxiliary K from inactive set")
    aux_coeff: float | None = Field(None, description="Auxiliary reconstruction loss coeff")

    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.TOPK
        return values


class VITopKSAEOutput(SAEOutput):
    preacts: Float[torch.Tensor, "... c"]
    mask: Float[torch.Tensor, "... c"]
    soft_mask: Float[torch.Tensor, "... c"]
    p: Float[torch.Tensor, "... c"]             # Bernoulli probs q(z=1|x)
    eta: Float[torch.Tensor, "... c"]           # gate logits
    score: Float[torch.Tensor, "... c"]         # blended selection score


# ---------- model ----------
class VITopKSAE(BaseSAE):
    """
    Top-K SAE with variational Bernoulli gates and dual ascent K-constraint.
    Selection score (train): log(r) + log(z_tilde), where z_tilde ~ Binary-Concrete(eta, temp).
    Inference score: lambda*log(r) + (1-lambda)*log(p).
    """

    def __init__(
        self,
        input_size: int,
        n_dict_components: int,
        k: int,
        *,
        mse_coeff: float = 1.0,
        aux_k: int | None = None,
        aux_coeff: float | None = None,
        init_decoder_orthogonal: bool = True,
        tied_encoder_init: bool = True,
        use_pre_relu: bool = True,
        vi_temp: float = 0.67,
        score_mix_lambda: float = 0.5,
        st_tau: float = 0.5,
        kl_coeff: float = 1e-3,
        prior_rate: float | None = None,
        card_coeff: float = 0.05,
        dual_lr: float = 0.0,
        dual_init: float = 0.0,
    ):
        super().__init__()
        assert k >= 0 and n_dict_components > 0 and input_size > 0

        self.input_size = input_size
        self.n_dict_components = n_dict_components
        self.k = int(k)
        self.use_pre_relu = bool(use_pre_relu)
        self.register_buffer("train_progress", torch.tensor(0.0, dtype=torch.float32, device='cpu'))

        # Loss coeffs
        self.mse_coeff = float(mse_coeff)
        self.kl_coeff = float(kl_coeff)
        self.card_coeff = float(card_coeff)

        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0

        # Gating & scoring
        self.vi_temp = float(vi_temp)
        self.score_mix_lambda = float(score_mix_lambda)
        self.st_tau = float(st_tau)

        # Prior & dual
        self.prior_rate = float(prior_rate) if prior_rate is not None else (self.k / float(n_dict_components))
        self.dual_lr = float(dual_lr)
        self.register_buffer("lambda_dual", torch.tensor(float(dual_init)), persistent=True)

        # Bias and linear maps
        self.decoder_bias = nn.Parameter(torch.zeros(input_size))
        self.encoder = nn.Linear(input_size, n_dict_components, bias=False)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)

        if init_decoder_orthogonal:
            self.decoder.weight.data = nn.init.orthogonal_(self.decoder.weight.data.T).T
        else:
            dec_w = torch.randn_like(self.decoder.weight)
            dec_w = F.normalize(dec_w, dim=0)
            self.decoder.weight.data.copy_(dec_w)
        if tied_encoder_init:
            self.encoder.weight.data.copy_(self.decoder.weight.data.T)

        # Amortized Bernoulli head: eta = gamma_h * LN(r) + beta_h
        self.use_ln = True
        self.ln = nn.LayerNorm(n_dict_components, elementwise_affine=True)
        self.gate_gamma_h = nn.Parameter(torch.zeros(n_dict_components))
        self.gate_beta_h = nn.Parameter(torch.full((n_dict_components,), torch.logit(torch.tensor(self.prior_rate))))

        self.k_warmup_ratio = 0.4

    @property
    def dict_elements(self) -> torch.Tensor:
        return F.normalize(self.decoder.weight, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device

    def _gate_logits(self, r: torch.Tensor) -> torch.Tensor:
        h = self.ln(r) if self.use_ln else r
        return self.gate_gamma_h * h + self.gate_beta_h
    
    def _current_k(self):
        if self.train_progress >= self.k_warmup_ratio:
            return self.k
        t = min(1.0, self.train_progress.item() / self.k_warmup_ratio)
        # cosine interp from self.n_dict_components -> self.k
        alpha = 0.5 * (1 + torch.cos(torch.tensor(t * 3.1415926535, device=self.gate_beta_h.device)))
        return float(self.k + (self.n_dict_components - self.k) * alpha.item())

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> VITopKSAEOutput:
        x_centered = x - self.decoder_bias
        preacts = self.encoder(x_centered)
        r = F.relu(preacts) if self.use_pre_relu else preacts

        eta = self._gate_logits(r)
        p = torch.sigmoid(eta)

        # Binary-Concrete sample (train) or deterministic proxy (eval)
        z_tilde = _sample_binary_concrete(eta, temp=self.vi_temp, training=self.training)

        # Selection score
        eps = 1e-8
        if self.training:
            score = self.score_mix_lambda * torch.log(r + eps) + (1.0 - self.score_mix_lambda) * torch.log(z_tilde + eps)
        else:
            score = self.score_mix_lambda * torch.log(r + eps) + (1.0 - self.score_mix_lambda) * torch.log(p + eps)

        st_mask, hard_mask, soft_mask = _topk_st(score, self.k, tau_st=self.st_tau)

        # Exact Top-K codes (do not scale magnitudes by z; gate is used for selection)
        c = r * st_mask
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)

        return VITopKSAEOutput(
            input=x, c=c, output=x_hat, logits=None,
            preacts=preacts, mask=hard_mask, soft_mask=soft_mask, p=p, eta=eta, score=score
        )

    def compute_loss(self, output: VITopKSAEOutput) -> SAELoss:
        mse = F.mse_loss(output.output, output.input)
        total = self.mse_coeff * mse
        loss_dict: dict[str, torch.Tensor] = {"mse_loss": mse.detach().clone()}

        # KL(q(z|x) || Bernoulli(rho)) for probability calibration
        # if self.kl_coeff > 0.0:
        #     rho_t = self._current_prior()
        #     kl = _kl_bern_bern(output.p, rho_t).mean()
        #     total = total + self.kl_coeff * kl
        #     loss_dict["rho_t"] = rho_t
        #     loss_dict["kl_gate"] = kl.detach().clone()

        # Soft cardinality calibration on soft Top-K mask (stabilizes thresholding)
        if self.card_coeff > 0.0:
            soft_card = output.soft_mask.sum(dim=-1).mean()
            card_loss = (soft_card - self._current_k()) ** 2
            total = total + self.card_coeff * card_loss
            loss_dict["card_loss"] = card_loss.detach().clone()

        # Dual ascent penalty for expected-K: lambda * (sum_i p_i(x) - K)
        if self.dual_lr > 0.0:
            exp_card = output.p.sum(dim=-1).mean()
            lagr_term = self.lambda_dual * (exp_card - self.k)
            total = total + lagr_term
            loss_dict["lagrangian"] = lagr_term.detach().clone()
            loss_dict["exp_card"] = exp_card.detach().clone()

        # Optional Aux-K (same as your current Aux-K)
        if self.aux_k > 0 and self.aux_coeff > 0.0:
            z = F.relu(output.preacts) if self.use_pre_relu else output.preacts
            z_inactive = z * (1.0 - output.mask)
            latent_dim = z_inactive.size(-1)
            aux_k = min(self.aux_k, max(0, latent_dim - self.k))
            if aux_k > 0:
                aux_idx = torch.topk(z_inactive, k=aux_k, dim=-1)[1]
                aux_mask = torch.zeros_like(z_inactive)
                aux_mask.scatter_(-1, aux_idx, 1.0)
                aux_code = z * aux_mask
                with torch.no_grad():
                    dec_w_detached = F.normalize(self.decoder.weight.detach(), dim=0)
                    dec_b_detached = self.decoder_bias.detach()
                x_hat_aux = F.linear(aux_code, dec_w_detached, bias=dec_b_detached)
                aux_loss = F.mse_loss(x_hat_aux, output.input)
                total = total + self.aux_coeff * aux_loss
                loss_dict["aux_loss"] = aux_loss.detach().clone()
            else:
                loss_dict["aux_loss"] = torch.zeros((), device=output.input.device)

        return SAELoss(loss=total, loss_dict=loss_dict)

    @torch.no_grad()
    def update_dual(self, gap: torch.Tensor | float):
        """
        Dual ascent update for lambda:
            lambda <- max(0, lambda + dual_lr * gap), where gap = E[sum_i p_i(x)] - K
        Call this once per step with the *detached* minibatch gap (scalar).
        """
        if self.dual_lr <= 0.0:
            return
        if isinstance(gap, torch.Tensor):
            gap_val = float(gap.detach().cpu().item())
        else:
            gap_val = float(gap)
        self.lambda_dual.add_(self.dual_lr * gap_val).clamp_(min=0.0)
