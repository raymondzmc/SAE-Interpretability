# vi_threshold_topk_sae.py
import torch, torch.nn.functional as F
from torch import nn
from models.saes.base import BaseSAE, SAELoss
from typing import Tuple

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

class VIThresholdTopKSAE(BaseSAE):
    """
    Exact Top-K SAE with a variational scalar threshold tau per sample.
    Forward: hard Top-K on adjusted scores; Backward: soft threshold around tau.
    """
    def __init__(self, input_size: int, n_dict_components: int, k: int,
                 mse_coeff: float = 1.0, init_decoder_orthogonal: bool = True,
                 tied_encoder_init: bool = True, use_pre_relu: bool = True,
                 st_tau: float = 0.5, temp_soft: float = 0.5,
                 kl_tau_coeff: float = 1e-3, card_coeff: float = 5e-2,
                 lb_coeff: float = 1e-2):
        super().__init__()
        self.input_size, self.n_dict_components, self.k = input_size, n_dict_components, int(k)
        self.mse_coeff, self.use_pre_relu = float(mse_coeff), bool(use_pre_relu)
        self.st_tau, self.temp_soft = float(st_tau), float(temp_soft)
        self.kl_tau_coeff, self.card_coeff, self.lb_coeff = float(kl_tau_coeff), float(card_coeff), float(lb_coeff)

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

    @property
    def dict_elements(self): return F.normalize(self.decoder.weight, dim=0)
    @property
    def device(self): return next(self.parameters()).device

    def forward(self, x: torch.Tensor):
        x_centered = x - self.decoder_bias
        pre = self.encoder(x_centered)
        r = F.relu(pre) if self.use_pre_relu else pre
        # score = log magnitude after layernorm for scale invariance
        s = torch.log(self.ln(r) + 1e-8)

        mu, log_sig = self.thresh_head(s.detach() if not self.training else s)  # amortized q(tau|x)
        if self.training:
            eps = torch.randn_like(mu)
            tau = mu + eps * log_sig.exp()
        else:
            tau = mu  # use mean at eval

        # soft mask around the threshold (broadcast tau over features)
        m_soft = torch.sigmoid((s - tau) / self.temp_soft)

        # IMPORTANT: use an adjusted score that *depends* on tau so it can
        # tilt the ranking across samples (tau is scalar -> shifts all equally,
        # but we still use it for soft grads; you can switch to groupwise τ below).
        adj = s  # scalar τ shift cancels in ranking; keep adj == s for Top-K
        st_mask, hard_mask, _ = _topk_st(adj, self.k, tau_st=self.st_tau)

        c = r * st_mask
        x_hat = F.linear(c, self.dict_elements, bias=self.decoder_bias)
        return {"input": x, "output": x_hat, "preacts": pre, "mask": hard_mask,
                "soft_mask": m_soft, "tau_mu": mu, "tau_logsig": log_sig}

    def compute_loss(self, out):
        mse = F.mse_loss(out["output"], out["input"])
        total = self.mse_coeff * mse
        logs = {"mse_loss": mse.detach()}

        # KL(q(tau|x) || p(tau)) with p=N(mu0, sig0^2)
        if self.kl_tau_coeff > 0:
            mu, log_sig = out["tau_mu"], out["tau_logsig"]
            mu0, log_sig0 = self.tau_mu0, self.tau_logsig0
            kl = (log_sig0 - log_sig + (log_sig.exp()**2 + (mu - mu0)**2) / (log_sig0.exp()**2) - 1) * 0.5
            kl = kl.mean()
            total = total + self.kl_tau_coeff * kl
            logs["kl_tau"] = kl.detach()

        # Soft cardinality ~ K
        if self.card_coeff > 0:
            soft_card = out["soft_mask"].sum(-1).mean()
            card_loss = (soft_card - self.k) ** 2
            total = total + self.card_coeff * card_loss
            logs["card_loss"] = card_loss.detach()

        # Load-balancing across features: KL(q || uniform)
        if self.lb_coeff > 0:
            q = out["soft_mask"].mean(dim=tuple(range(out["soft_mask"].dim()-1)))  # average over batch dims -> (C,)
            q = q / (q.sum() + 1e-8)
            lb = (q * (q.clamp_min(1e-12).log() + torch.log(torch.tensor(q.numel(), device=q.device)))).sum()
            total = total + self.lb_coeff * lb
            logs["lb_loss"] = lb.detach()

        return SAELoss(loss=total, loss_dict=logs)
