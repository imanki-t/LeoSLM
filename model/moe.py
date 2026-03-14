import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import LeoConfig
from .norm   import RMSNorm


def swiglu(
    x:    torch.Tensor,
    gate: nn.Linear,
    up:   nn.Linear,
    down: nn.Linear,
) -> torch.Tensor:
    return down(F.silu(gate(x)) * up(x))


class ExpertFFN(nn.Module):
    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D = cfg.hidden_dim
        E = cfg.ffn_dim_expert
        self.gate = nn.Linear(D, E, bias=False)
        self.up   = nn.Linear(D, E, bias=False)
        self.down = nn.Linear(E, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x, self.gate, self.up, self.down)


class UWMRMoE(nn.Module):
    def __init__(self, cfg: LeoConfig):
        super().__init__()
        self.E         = cfg.moe_experts
        self.top_k     = cfg.moe_top_k
        self.sh_n      = cfg.moe_shared
        self.load_coef = cfg.moe_load_coeff

        self.experts   = nn.ModuleList([ExpertFFN(cfg) for _ in range(self.E)])
        self.shared    = nn.ModuleList([ExpertFFN(cfg) for _ in range(self.sh_n)])
        self.router    = nn.Linear(cfg.hidden_dim, self.E, bias=False)
        self.spec_bias = nn.Parameter(torch.zeros(self.E))
        self.gen_bias  = nn.Parameter(torch.zeros(self.E))
        self.out_norm  = RMSNorm(cfg.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        U: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        flat    = x.view(B * T, D)
        logits  = self.router(flat)

        if U is not None:
            u_flat  = U.reshape(B * T, 1).clamp(0.0, 1.0)
            logits  = logits + u_flat * self.spec_bias + (1.0 - u_flat) * self.gen_bias

        probs_all = logits.softmax(dim=-1)
        probs, idx = probs_all.topk(self.top_k, dim=-1)

        bal_loss = (
            self.E
            * (probs_all.mean(dim=0) ** 2).sum()
            * self.load_coef
        )

        out = torch.zeros_like(flat)
        for ki in range(self.top_k):
            for ei in range(self.E):
                weight   = ((idx[:, ki] == ei).float() * probs[:, ki]).unsqueeze(-1)
                out      = out + self.experts[ei](flat) * weight

        for sh in self.shared:
            out = out + sh(flat)

        return self.out_norm(out.view(B, T, D)), bal_loss
