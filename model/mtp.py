"""
model/mtp.py — Multi-Token Prediction (MTP) Head — DeepSeek V3 style

BUG FIX:
  Original code in leo_slm.py did:
    self.mtp.projs[0].weight = self.lm_head.weight

  This is WRONG. Each MTP head transforms `h` through a small MLP
  (self.heads[i]) BEFORE projecting. So projs[0] takes as input
  heads[0](h_norm), which lives in a DIFFERENT representation space
  than h_norm itself.  Tying projs[0] to lm_head (which operates
  directly on h_norm) creates a dimension/space mismatch that gives
  garbage logits for the first speculative token.

  Fix: NO weight tying for MTP projections.  Each proj[i] is independent.
  The memory cost is N * D * V BF16 params.  For N=4, D=2560, V=65543:
    4 * 2560 * 65543 * 2 bytes ≈ 1.3 GB — acceptable on v5e-8.

  Note for advanced users: if you want to tie weights, only tie
  projs[i] to lm_head after all N heads are ALSO identity-mapped
  (i.e., when heads[i] is initialised as identity). We don't do this
  here because the heads are randomly initialised.
"""

import torch
import torch.nn as nn
from typing import List

from .config import LeoConfig
from .norm   import RMSNorm


class MultiTokenPredictionHead(nn.Module):
    """
    Predicts tokens t+2, t+3, ..., t+N+1 in parallel.

    Training benefit: N extra AR supervision signals per step (free).
    Inference benefit: N draft tokens per forward pass → built-in
    speculative decoding without a separate draft model.

    Architecture:
        for i in 0..N-1:
            h = heads[i](h)            # small 2-layer MLP transform
            logits_i = projs[i](h)     # vocab projection
    """

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D      = cfg.hidden_dim
        V      = cfg.vocab_size
        self.N = cfg.mtp_n

        self.heads = nn.ModuleList([
            nn.Sequential(
                RMSNorm(D),
                nn.Linear(D, D, bias=False),
                nn.SiLU(),
                nn.Linear(D, D, bias=False),
            )
            for _ in range(self.N)
        ])

        # BUG FIX: independent projections — no weight tying with lm_head
        self.projs = nn.ModuleList([
            nn.Linear(D, V, bias=False) for _ in range(self.N)
        ])

    def forward(self, hidden: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            hidden: (B, T, D) — final normalised hidden states

        Returns:
            List of N tensors, each (B, T, V) — logits for t+2, t+3, ...
        """
        logits_list = []
        h = hidden
        for i in range(self.N):
            h = self.heads[i](h)
            logits_list.append(self.projs[i](h))
        return logits_list

    @torch.no_grad()
    def speculative_draft(
        self,
        hidden:      torch.Tensor,   # (B, T, D)
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate N draft token IDs from the last position.

        Returns:
            (B, N) — N draft token IDs
        """
        logits_list = self.forward(hidden)
        drafts = []
        for logits in logits_list:
            last = logits[:, -1, :]                    # (B, V)
            if temperature > 0 and temperature != 1.0:
                last = last / temperature
            drafts.append(last.argmax(-1))             # (B,)
        return torch.stack(drafts, dim=1)              # (B, N)
