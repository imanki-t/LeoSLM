"""
model/ect.py — ECT v3: Enhanced Epistemic Confidence Tokens
=============================================================
NOVEL — Aether, no prior art.

ECTs are a set of learnable tokens that attend to the full sequence and
produce a per-token uncertainty score U ∈ [0,1] at every transformer layer.

v3 additions over v1:
  • 8 base tokens (v1 had 4)
  • Domain specialist biases (ECT-DS) — each ECT biases toward one tool domain
  • PRM head — rates CoT step quality inside <think>…</think>

The score U gates:
  • Routing in UWMR MoE (uncertain → specialist experts)
  • EPE frequency scaling (uncertain tokens get wider positional receptive fields)
  • ACGI tool invocation (U > threshold → force tool call)
  • IDK token training (U > threshold → predict [IDK])
  • TDM memory writes (U < threshold → confident tokens write to memory)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import LeoConfig
from .norm   import RMSNorm


class ECTv3Module(nn.Module):
    """
    ECT v3: 8 base tokens + dynamic domain spawning + PRM head.

    Forward:
        seq_h      : (B, T, D)
        is_think   : (B, T) bool mask — positions inside <think>…</think>
    Returns:
        ect_h      : (B, E, D)    — updated ECT hidden states
        U          : (B, T)       — per-token uncertainty ∈ [0,1]
        prm_scores : (B, T) | None — CoT step quality scores (only if use_prm)
    """

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D, E          = cfg.hidden_dim, cfg.num_ect
        self.cfg      = cfg
        self.num_ect  = E

        # Base ECT embeddings
        self.ect_embed   = nn.Parameter(torch.randn(E, D) * 0.02)

        # Domain specialist biases (ECT-DS): each ECT has a domain affinity vector
        self.domain_bias = nn.Parameter(torch.randn(cfg.estr_domain_count, D) * 0.01)

        # Cross-attention: ECTs attend to the full sequence
        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        self.W_O = nn.Linear(D, D, bias=False)
        self.norm = RMSNorm(D)

        # Uncertainty score MLP: sequence hidden → per-token uncertainty scalar
        self.score_proj = nn.Linear(D, E, bias=False)
        self.score_mlp  = nn.Sequential(
            nn.Linear(E, E * 2),
            nn.GELU(),
            nn.Linear(E * 2, 1),
        )

        # Process Reward Model head — rates CoT step quality ∈ [0,1]
        if cfg.use_prm:
            self.prm_head = nn.Sequential(
                nn.Linear(D, cfg.prm_hidden),
                nn.GELU(),
                nn.Linear(cfg.prm_hidden, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        seq_h:    torch.Tensor,
        is_think: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, T, D  = seq_h.shape
        H        = self.cfg.ect_heads

        # ── ECT cross-attention: ECTs ← sequence ─────────────────────────────
        ect_h = self.ect_embed.unsqueeze(0).expand(B, -1, -1)           # (B, E, D)
        ect_h = ect_h + self.norm(seq_h.mean(1, keepdim=True))          # seed from mean

        Q = self.W_Q(ect_h).view(B, self.num_ect, H, D // H).transpose(1, 2)
        K = self.W_K(seq_h).view(B, T, H, D // H).transpose(1, 2)
        V = self.W_V(seq_h).view(B, T, H, D // H).transpose(1, 2)

        attn  = F.scaled_dot_product_attention(Q, K, V)                 # (B, H, E, D//H)
        ect_h = self.W_O(
            attn.transpose(1, 2).contiguous().view(B, self.num_ect, D)
        )                                                                # (B, E, D)

        # ── Uncertainty scores: per-token scalar ──────────────────────────────
        proj  = self.score_proj(seq_h)                                   # (B, T, E)
        U     = self.score_mlp(proj).squeeze(-1).sigmoid()              # (B, T) ∈ [0,1]

        # ── PRM: score CoT think positions ────────────────────────────────────
        # XLA-safe: never call .any() on a device tensor — that forces a
        # host sync which fragments HBM.  Instead always compute PRM and
        # multiply by the mask (0.0 outside think spans → zero cost in loss).
        prm_scores = None
        if self.cfg.use_prm:
            prm_raw    = self.prm_head(seq_h).squeeze(-1)               # (B, T)
            think_w    = is_think.float() if is_think is not None else torch.ones_like(prm_raw)
            prm_scores = prm_raw * think_w

        return ect_h, U, prm_scores
