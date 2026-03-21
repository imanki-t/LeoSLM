"""
model/moe.py — Uncertainty-Weighted MoE Routing (UWMR)

BUG FIXES vs original:
  1. XLA-safe expert dispatch: removed nested Python loop `for ki in range(top_k): for ei in range(E):`
     which called ALL experts for ALL top-k positions even when weight=0, wasting O(E*top_k) compute.
     New code loops only over E experts once, using vectorised weight computation → O(E) passes.
  2. All tensor ops are fixed-shape, avoiding XLA graph recompilation between steps.
  3. Balance loss now correctly uses mean over batch dimension before squaring.
"""

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
    """
    Uncertainty-Weighted MoE Routing.

    Router uses ECT uncertainty signal to bias routing:
      - High uncertainty  → specialist experts (spec_bias)
      - Low  uncertainty  → generalist experts (gen_bias)

    XLA-safe dispatch: for each of the E experts, compute token weights
    with fully vectorised ops (no conditional dispatch, fixed shapes).
    """

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
        flat    = x.view(B * T, D)           # (BT, D)

        # ── Router logits with uncertainty bias ──────────────────────────────
        logits = self.router(flat)            # (BT, E)

        if U is not None:
            u_flat = U.reshape(B * T, 1).clamp(0.0, 1.0)   # (BT, 1)
            logits = logits + u_flat * self.spec_bias + (1.0 - u_flat) * self.gen_bias

        probs_all = logits.softmax(dim=-1)    # (BT, E)

        # ── Top-k selection ──────────────────────────────────────────────────
        top_probs, top_indices = probs_all.topk(self.top_k, dim=-1)  # (BT, top_k)

        # ── Load-balancing auxiliary loss ────────────────────────────────────
        # BUG FIX: mean over BT then square, not square then mean —
        # matches the DeepSeek / Switch-Transformer formulation correctly.
        mean_prob = probs_all.mean(dim=0)     # (E,)
        bal_loss  = self.E * (mean_prob ** 2).sum() * self.load_coef

        # ── XLA-safe vectorised expert dispatch ──────────────────────────────
        # For each expert ei, compute the aggregate weight each token assigns
        # to ei across all top-k slots.  Shape: (BT,).
        # `(top_indices == ei)` is a fixed-shape boolean op → XLA-friendly.
        out = torch.zeros_like(flat)
        for ei in range(self.E):
            # Which top-k slots chose expert ei?  (BT, top_k) bool → (BT, top_k) float
            in_topk = (top_indices == ei).float()
            # Aggregate probability across all top-k slots:  (BT,)
            token_weight = (in_topk * top_probs).sum(dim=-1)
            # Expert forward on ALL tokens (fixed shape for XLA); weight to 0 for non-users
            expert_out = self.experts[ei](flat)      # (BT, D)
            out = out + expert_out * token_weight.unsqueeze(-1)

        # ── Shared experts (always active) ───────────────────────────────────
        for sh in self.shared:
            out = out + sh(flat)

        return self.out_norm(out.view(B, T, D)), bal_loss
