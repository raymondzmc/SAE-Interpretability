# vi_topk_sae.py
import torch, torch.nn.functional as F
from torch import nn
from models.saes.base import BaseSAE, SAELoss, SAEOutput, SAEConfig
from typing import Tuple, Any
from pydantic import Field, ConfigDict, model_validator
from jaxtyping import Float
from utils.enums import SAEType

def _topk_st(scores: torch.Tensor, k: int, tau_st: float = 0.5):
    if k <= 0:
        hard = torch.zeros_like(scores); soft = torch.zeros_like(scores)
        return hard + (soft - soft.detach()), hard, soft
    if k >= scores.size(-1):
        hard = torch.ones_like(scores); soft = torch.ones_like(scores)
        return hard + (soft - soft.detach()), hard, soft
    topk = torch.topk(scores, k, dim=-1)
    hard = torch.zeros_like(scores)
    hard.scatter_(-1, topk.indices, 1.0)
    kth = topk.values[..., -1].unsqueeze(-1)
    soft = torch.sigmoid((scores - kth) / tau_st)
    st = hard + (soft - soft.detach())
    return st, hard, soft

class VIThresholdHead(nn.Module):
    """
    Amortized q(tau|x). Predicts (mu, log_sigma) from summary stats of the per-sample scores.
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        # We'll compute two stats per sample: mean and std over features (after LN)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, 2)  # outputs [mu, log_sigma]
        )
        self.register_buffer("eps", torch.tensor(1e-6))

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # s: (..., C). Collapse leading dims to batch
        B = s.numel() // s.size(-1)
        C = s.size(-1)
        x = s.reshape(B, C)
        # sample-wise stats
        m = x.mean(-1, keepdim=True)
        v = (x - m).pow(2).mean(-1, keepdim=True).clamp_min(self.eps)
        feat = torch.cat([m, v.sqrt()], dim=-1)  # (B, 2)
        mu, log_sigma = self.mlp(feat).chunk(2, dim=-1)  # (B,1) each
        # reshape back to broadcast over features
        mu = mu.view(*s.shape[:-1], 1)
        log_sigma = log_sigma.view(*s.shape[:-1], 1)
        return mu, log_sigma


class VITopKSAEConfig(SAEConfig):
    """
    Config for Variational Inference Top-K SAE.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    sae_type: SAEType = Field(default=SAEType.VI_TOPK, description="Type of SAE (automatically set to vi_topk)")
    k: int = Field(..., description="Number of active features to keep per sample")
    tied_encoder_init: bool = Field(True, description="Initialize encoder as decoder.T")

    # Variational inference parameters
    vi_temp: float = Field(0.67, description="Binary-Concrete temperature")
    score_mix_lambda: float = Field(0.5, description="Blend between log-magnitude and probability evidence")
    st_tau: float = Field(0.5, description="Softness for straight-through Top-K")

    # Loss coefficients
    kl_coeff: float = Field(1e-3, description="KL coefficient against prior")
    card_coeff: float = Field(0.05, description="Soft-cardinality calibration on soft mask")

    # Prior & dual ascent for expected-K
    prior_rate: float | None = Field(None, description="rho for prior; default K/C if None")
    dual_lr: float = Field(0.01, description="Dual ascent step size for lambda (0 disables)")
    dual_init: float = Field(0.0, description="Initial lambda for expected-K constraint")

    # Auxiliary K (optional)
    aux_k: int | None = Field(None, description="Auxiliary K from inactive set")
    aux_coeff: float | None = Field(None, description="Auxiliary reconstruction loss coeff")

    @model_validator(mode="before")
    @classmethod
    def set_sae_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            values["sae_type"] = SAEType.VI_TOPK
        return values


class VITopKSAEOutput(SAEOutput):
    """
    VI TopK SAE output extending SAEOutput with useful intermediates for loss/analysis.
    """
    preacts: Float[torch.Tensor, "... c"]  # encoder linear outputs (after centering)
    mask: Float[torch.Tensor, "... c"]     # binary mask of selected Top-K indices
    p: Float[torch.Tensor, "... c"]        # soft probabilities for VI inference
    tau_mu: Float[torch.Tensor, "... 1"]   # threshold mean
    tau_logsig: Float[torch.Tensor, "... 1"] # threshold log-sigma


class VITopKSAE(BaseSAE):
    """
    Variational Inference Top-K SAE with a variational scalar threshold tau per sample.
    Forward: hard Top-K on adjusted scores; Backward: soft threshold around tau.
    """
    def __init__(self, input_size: int, n_dict_components: int, k: int,
                 mse_coeff: float = 1.0, init_decoder_orthogonal: bool = True,
                 tied_encoder_init: bool = True,
                 st_tau: float = 0.5, vi_temp: float = 0.67,
                 kl_coeff: float = 1e-3, card_coeff: float = 5e-2,
                 score_mix_lambda: float = 0.5, prior_rate: float | None = None,
                 dual_lr: float = 0.01, dual_init: float = 0.0,
                 aux_k: int | None = None, aux_coeff: float | None = None):
        super().__init__()
        self.input_size, self.n_dict_components, self.k = input_size, n_dict_components, int(k)
        self.mse_coeff = float(mse_coeff)
        self.st_tau, self.vi_temp = float(st_tau), float(vi_temp)
        self.kl_coeff, self.card_coeff = float(kl_coeff), float(card_coeff)
        self.score_mix_lambda = float(score_mix_lambda)
        
        # Dual ascent parameters
        self.dual_lr = float(dual_lr)
        self.register_buffer("dual", torch.tensor(float(dual_init)))
        
        # Auxiliary parameters
        self.aux_k = int(aux_k) if aux_k is not None and aux_k > 0 else 0
        self.aux_coeff = (aux_coeff if aux_coeff is not None else 0.0) if self.aux_k > 0 else 0.0

        self.decoder_bias = nn.Parameter(torch.zeros(input_size))
        self.encoder = nn.Linear(input_size, n_dict_components, bias=False)
        self.decoder = nn.Linear(n_dict_components, input_size, bias=False)

        if init_decoder_orthogonal:
            self.decoder.weight.data = nn.init.orthogonal_(self.decoder.weight.data.T).T
        else:
            w = torch.randn_like(self.decoder.weight); w = F.normalize(w, dim=0)
            self.decoder.weight.data.copy_(w)
        if tied_encoder_init:
            self.encoder.weight.data.copy_(self.decoder.weight.data.T)

        # score normalization (keeps threshold identifiable)
        self.ln = nn.LayerNorm(n_dict_components, elementwise_affine=True)
        self.thresh_head = VIThresholdHead(n_dict_components)

        # prior over tau ~ N(mu0, sigma0^2)
        self.register_buffer("tau_mu0", torch.tensor(0.0))
        self.register_buffer("tau_logsig0", torch.tensor(0.0))  # sigma0 = exp(0)=1
        
        # prior rate for Bernoulli prior
        prior_rate = prior_rate if prior_rate is not None else k / n_dict_components
        self.register_buffer("prior_rate", torch.tensor(float(prior_rate)))

    @property
    def dict_elements(self): return F.normalize(self.decoder.weight, dim=0)
    @property
    def device(self): return next(self.parameters()).device

    def update_dual(self, gap: torch.Tensor):
        """Update dual variable for expected-K constraint"""
        if self.dual_lr > 0:
            self.dual.add_(self.dual_lr * gap.detach())
            self.dual.clamp_(min=0.0)  # Keep dual non-negative

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> VITopKSAEOutput:
        x_centered = x - self.decoder_bias
        pre = self.encoder(x_centered)
        # score = log magnitude after layernorm for scale invariance
        s = torch.log(self.ln(pre) + 1e-8)

        mu, log_sig = self.thresh_head(s.detach() if not self.training else s)  # amortized q(tau|x)
        if self.training:
            eps = torch.randn_like(mu)
            tau = mu + eps * log_sig.exp()
        else:
            tau = mu  # use mean at eval

        # soft mask around the threshold (broadcast tau over features)
        m_soft = torch.sigmoid((s - tau) / self.vi_temp)

        # IMPORTANT: use an adjusted score that *depends* on tau so it can
        # tilt the ranking across samples (tau is scalar -> shifts all equally,
        # but we still use it for soft grads; you can switch to groupwise τ below).
        adj = s  # scalar τ shift cancels in ranking; keep adj == s for Top-K
        st_mask, hard_mask, _ = _topk_st(adj, self.k, tau_st=self.st_tau)

        c = pre * st_mask
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)
        
        return VITopKSAEOutput(
            input=x, 
            c=c, 
            output=x_hat, 
            logits=None, 
            preacts=pre, 
            mask=hard_mask,
            p=m_soft, 
            tau_mu=mu, 
            tau_logsig=log_sig
        )

    def compute_loss(self, output: VITopKSAEOutput) -> SAELoss:
        mse = F.mse_loss(output.output, output.input)
        total = self.mse_coeff * mse
        logs = {"mse_loss": mse.detach()}

        # KL(q(tau|x) || p(tau)) with p=N(mu0, sig0^2)
        if self.kl_coeff > 0:
            mu, log_sig = output.tau_mu, output.tau_logsig
            mu0, log_sig0 = self.tau_mu0, self.tau_logsig0
            kl = (log_sig0 - log_sig + (log_sig.exp()**2 + (mu - mu0)**2) / (log_sig0.exp()**2) - 1) * 0.5
            kl = kl.mean()
            total = total + self.kl_coeff * kl
            logs["kl_loss"] = kl.detach()

        # Soft cardinality ~ K with dual ascent
        if self.card_coeff > 0:
            soft_card = output.p.sum(-1).mean()
            card_loss = (soft_card - self.k) ** 2 + self.dual * (soft_card - self.k)
            total = total + self.card_coeff * card_loss
            logs["card_loss"] = card_loss.detach()
            logs["dual"] = self.dual.detach()

        # Optional auxiliary dead-feature loss
        if self.aux_k > 0 and self.aux_coeff > 0.0:
            z = output.preacts
            z_inactive = z * (1.0 - output.mask)
            latent_dim = z_inactive.size(-1)
            aux_k = min(self.aux_k, max(0, latent_dim - self.k))
            if aux_k > 0:
                aux_idx = torch.topk(z_inactive, k=aux_k, dim=-1)[1]
                aux_mask = torch.zeros_like(z_inactive)
                aux_mask.scatter_(-1, aux_idx, 1.0)
                aux_code = z * aux_mask  # use actual (ReLUed) magnitudes for those indices

                # Reconstruct with DETACHED normalized decoder and bias
                with torch.no_grad():
                    dec_w_detached = F.normalize(self.decoder.weight.detach(), dim=0)
                    dec_b_detached = self.decoder_bias.detach()
                x_hat_aux = F.linear(aux_code, dec_w_detached, bias=dec_b_detached)

                aux_loss = F.mse_loss(x_hat_aux, output.input)
                total = total + self.aux_coeff * aux_loss
                logs["aux_loss"] = aux_loss.detach().clone()
            else:
                # No room for auxiliary picks; report zero aux loss
                logs["aux_loss"] = torch.zeros((), device=output.input.device)

        return SAELoss(loss=total, loss_dict=logs)
