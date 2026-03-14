"""
model/moe.py — UWMR MoE (Uncertainty-Weighted MoE Routing)
============================================================
NOVEL — Aether, no prior art.

Standard MoE routes tokens to experts by learnable logits alone.
UWMR additionally biases routing by ECT uncertainty U:
  high-U tokens (uncertain) → specialist experts (spec_bias)
  low-U tokens  (confident) → generalist experts (gen_bias)

This means uncertain tokens automatically seek more focused processing,
while confident tokens efficiently use broad-coverage experts.

Load-balance loss ensures no expert is starved.
Shared experts (moe_shared) are always active regardless of routing,
providing a stable "backbone" of general capacity at every token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import LeoConfig


def swiglu(
    x:    torch.Tensor,
    gate: nn.Linear,
    up:   nn.Linear,
    down: nn.Linear,
) -> torch.Tensor:
    """SwiGLU activation: down(silu(gate(x)) * up(x))."""
    return down(F.silu(gate(x)) * up(x))


class ExpertFFN(nn.Module):
    """Single SwiGLU feed-forward expert."""

    def __init__(self, in_d: int, mid_d: int):
        super().__init__()
        self.gate = nn.Linear(in_d, mid_d, bias=False)
        self.up   = nn.Linear(in_d, mid_d, bias=False)
        self.down = nn.Linear(mid_d, in_d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x, self.gate, self.up, self.down)


class UWMRMoE(nn.Module):
    """
    UWMR-enhanced Mixture-of-Experts.

    Routing logits = router(x) + U * spec_bias + (1-U) * gen_bias

    Args:
        cfg : LeoConfig — reads moe_experts, moe_top_k, moe_shared,
              ffn_dim_expert, moe_load_coeff, uwmr_spec_scale, uwmr_gen_scale
    """

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D         = cfg.hidden_dim
        self.E    = cfg.moe_experts
        self.top_k = cfg.moe_top_k
        self.sh_n  = cfg.moe_shared

        self.experts    = nn.ModuleList([ExpertFFN(D, cfg.ffn_dim_expert) for _ in range(self.E)])
        self.shared     = nn.ModuleList([ExpertFFN(D, cfg.ffn_dim_expert) for _ in range(self.sh_n)])
        self.router     = nn.Linear(D, self.E, bias=False)

        # UWMR learnable routing biases (initialized near zero; shaped by training)
        self.spec_bias  = nn.Parameter(torch.zeros(self.E))
        self.gen_bias   = nn.Parameter(torch.zeros(self.E))
        self.load_coeff = cfg.moe_load_coeff

    def forward(
        self,
        x: torch.Tensor,
        U: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, T, D)
            U : (B, T)  — ECT uncertainty (None → standard routing)
        Returns:
            out      : (B, T, D)
            bal_loss : scalar load-balance loss
        """
        B, T, D = x.shape
        flat    = x.view(B * T, D)
        logits  = self.router(flat)                            # (BT, E)

        # UWMR: bias routing by uncertainty
        if U is not None:
            u_flat = U.view(B * T, 1).clamp(0, 1)
            logits = logits + u_flat * self.spec_bias + (1 - u_flat) * self.gen_bias

        probs, idx = torch.topk(logits.softmax(-1), self.top_k, dim=-1)

        # Load-balance loss: penalize concentration of tokens in few experts
        bal_loss = self.E * (logits.softmax(-1).mean(0) ** 2).sum() * self.load_coeff

        # Route tokens — fully static shapes for XLA.
        # Each expert runs on ALL tokens; its output is weighted by 0.0 for
        # non-routed tokens.  No boolean masking, no dynamic tensor sizes,
        # no host syncs.  XLA compiles this once and never recompiles.
        out = torch.zeros_like(flat)
        for ki in range(self.top_k):
            for ei in range(self.E):
                # weight (BT, 1): probs[:,ki] where routed to expert ei, else 0.0
                weight   = ((idx[:, ki] == ei).float() * probs[:, ki]).unsqueeze(-1)
                expert_o = self.experts[ei](flat)   # always (BT, D) — static shape
                out      = out + expert_o * weight

        # Shared experts always contribute (no gating)
        for sh in self.shared:
            out += sh(flat)

        return out.view(B, T, D), bal_loss
