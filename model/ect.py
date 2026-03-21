"""
model/ect.py — ECT v3 (Enhanced Epistemic Confidence Tokens)

BUG FIXES vs original:
  1. WRONG HEAD DIM in attention: W_Q/K/V projected `seq_h` from shape (B,T,D)
     to (B,T,D) then reshaped to (B,T,H,D//H). With H=ect_heads=4, each head
     has dim D//H = 2560//4 = 640. But W_Q produces dim=D=2560 total, so
     D//H = 640. This is architecturally valid (just large heads) but the
     RESHAPE was applied to full D=2560 output, which is correct. However:
     W_Q(ect_h) where ect_h is (B,E,D) was reshaped to (B,E,H,D//H) which
     equals (B,8,4,640). This is fine.
     
     ACTUAL BUG: `self.W_K = nn.Linear(D, D)` — K is applied to seq_h which
     has T tokens, not E=num_ect tokens. But Q has shape (B,E,H,D//H) while
     K has (B,T,H,D//H). These must be compatible for `scaled_dot_product_attention`.
     Q: (B, H, E, D//H)  
     K: (B, H, T, D//H)  ← this IS the cross-attention pattern (ECT → seq), correct.
     But original code used `attn.transpose(1,2).contiguous().view(B,num_ect,D)` 
     without `contiguous()` before the final view, which silently corrupts on some
     PyTorch versions when the tensor is non-contiguous.
     Fix: add `.contiguous()` before the final `.view()`.
     
  2. `score_mlp` input is E=num_ect features but `score_proj` projects from D→E
     per position. Output U has shape (B,T) from `squeeze(-1)` — correct.
     BUT: `score_proj(seq_h)` runs on (B,T,D) → (B,T,E). Then score_mlp on
     (B,T,E) → (B,T,1) → squeeze → (B,T). Fine.
  
  3. ECT embed expansion: `ect_h = self.ect_embed + self.norm(seq_h.mean(1,keepdim=True))`
     broadcasts correctly: (E,D) + (B,1,D) → (B,E,D). This is correct.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import LeoConfig
from .norm   import RMSNorm


class ECTv3Module(nn.Module):
    """
    Enhanced Epistemic Confidence Tokens v3.

    Maintains E learnable 'epistemic token' embeddings that cross-attend
    to the sequence to produce per-position uncertainty scores U ∈ (0,1).

    High U → model is uncertain at that position → triggers:
      - Diffusion denoising (hybrid generation)
      - IDK token routing
      - Broader positional encoding (EPE)
      - Expert specialisation (UWMR routing)
      - Tool invocation (ACGI)
    """

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D  = cfg.hidden_dim
        E  = cfg.num_ect
        H  = cfg.ect_heads      # number of attention heads in ECT cross-attention
        hd = D // H             # head dimension

        self.cfg     = cfg
        self.num_ect = E

        # Learnable ECT embeddings: (E, D)
        self.ect_embed   = nn.Parameter(torch.randn(E, D) * 0.02)
        # Domain bias for ESTR (ECT-Seeded Tool Routing)
        self.domain_bias = nn.Parameter(torch.randn(cfg.estr_domain_count, D) * 0.01)

        # Cross-attention: ECT queries, sequence as keys/values
        self.W_Q = nn.Linear(D, D, bias=False)   # (B, E, D) → (B, E, D)
        self.W_K = nn.Linear(D, D, bias=False)   # (B, T, D) → (B, T, D)
        self.W_V = nn.Linear(D, D, bias=False)   # (B, T, D) → (B, T, D)
        self.W_O = nn.Linear(D, D, bias=False)   # (B, E, D) → (B, E, D)
        self.norm = RMSNorm(D)

        # Per-position uncertainty head: seq_h → U
        self.score_proj = nn.Linear(D, E, bias=False)   # (B, T, D) → (B, T, E)
        self.score_mlp  = nn.Sequential(
            nn.Linear(E, E * 2),
            nn.GELU(),
            nn.Linear(E * 2, 1),
        )

        # Process Reward Model head (optional, activated inside <think> spans)
        if cfg.use_prm:
            self.prm_head = nn.Sequential(
                nn.Linear(D, cfg.prm_hidden),
                nn.GELU(),
                nn.Linear(cfg.prm_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.prm_head = None

        self._H  = H
        self._hd = hd

    def forward(
        self,
        seq_h:    torch.Tensor,                     # (B, T, D)
        is_think: Optional[torch.Tensor] = None,    # (B, T) bool/float mask
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = seq_h.shape
        H, hd   = self._H, self._hd

        # ── Initialise ECT tokens with global context ─────────────────────────
        # (E, D) + mean-pool of sequence → (B, E, D)
        ect_h = self.ect_embed.unsqueeze(0).expand(B, -1, -1)          # (B, E, D)
        ect_h = ect_h + self.norm(seq_h.mean(1, keepdim=True))         # (B, E, D)

        # ── Cross-attention: ECT queries, sequence key/value ──────────────────
        Q = self.W_Q(ect_h).view(B, -1, H, hd).transpose(1, 2)        # (B, H, E, hd)
        K = self.W_K(seq_h).view(B, T, H, hd).transpose(1, 2)         # (B, H, T, hd)
        V = self.W_V(seq_h).view(B, T, H, hd).transpose(1, 2)         # (B, H, T, hd)

        # Flash attention — no causal mask (ECT does bidirectional cross-attention)
        attn_out = F.scaled_dot_product_attention(Q, K, V)             # (B, H, E, hd)

        # BUG FIX: add .contiguous() before .view() to avoid silent corruption
        # when tensor layout is non-contiguous after transpose.
        ect_h = self.W_O(
            attn_out.transpose(1, 2).contiguous().view(B, -1, D)
        )                                                              # (B, E, D)

        # ── Per-position uncertainty score ────────────────────────────────────
        # seq_h → (B, T, E) → (B, T, 1) → squeeze → (B, T) → sigmoid
        proj = self.score_proj(seq_h)         # (B, T, E)
        U    = self.score_mlp(proj).squeeze(-1).sigmoid()   # (B, T)

        # ── PRM scores (only meaningful inside <think> spans) ─────────────────
        prm_scores = None
        if self.prm_head is not None:
            prm_raw    = self.prm_head(seq_h).squeeze(-1)             # (B, T)
            think_w    = is_think.float() if is_think is not None else torch.ones_like(prm_raw)
            prm_scores = prm_raw * think_w

        return ect_h, U, prm_scores
