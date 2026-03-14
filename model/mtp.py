"""
model/mtp.py — Multi-Token Prediction (DeepSeek V3 style)
===========================================================
Instead of predicting only the next token, MTP predicts the next N tokens
simultaneously using N lightweight single-layer heads.

Benefits:
  Training : N-1 extra AR loss terms per step (free auxiliary signal)
  Inference: N draft tokens per forward pass → built-in speculative decoding
             without a separate draft model

Projection weights are tied to the main lm_head embedding for parameter
efficiency (handled in LeoSLM.__init__).
"""

import torch
import torch.nn as nn
from typing import List

from .config import LeoConfig
from .norm   import RMSNorm


class MultiTokenPredictionHead(nn.Module):
    """
    MTP: N lightweight heads, each predicting one future position.

    Head i predicts the token at position t + i + 1.
    Heads share the same architecture but have independent weights.
    Projection weights are optionally tied to the main lm_head (set in LeoSLM).
    """

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D     = cfg.hidden_dim
        V     = cfg.vocab_size
        self.N = cfg.mtp_n

        # One lightweight block per future position: RMSNorm → Linear → SiLU → Linear
        self.heads = nn.ModuleList([
            nn.Sequential(
                RMSNorm(D),
                nn.Linear(D, D, bias=False),
                nn.SiLU(),
                nn.Linear(D, D, bias=False),
            )
            for _ in range(self.N)
        ])

        # Per-head projection to vocab (optionally tied to main lm_head in LeoSLM)
        self.projs = nn.ModuleList([
            nn.Linear(D, V, bias=False) for _ in range(self.N)
        ])

    def forward(self, hidden: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            hidden : (B, T, D) — final hidden states from the main model
        Returns:
            List of N logit tensors, each (B, T, V),
            representing predictions for positions t+1 … t+N
        """
        logits_list = []
        h = hidden
        for i in range(self.N):
            h = self.heads[i](h)
            logits_list.append(self.projs[i](h))
        return logits_list

    def speculative_draft(
        self,
        hidden:      torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Inference: draft N candidate token ids from the last position.
        Used for speculative decoding — no separate draft model required.

        Args:
            hidden      : (B, T, D)
            temperature : sampling temperature (0 = greedy)
        Returns:
            drafts      : (B, N) — N draft token ids
        """
        logits_list = self.forward(hidden)
        drafts = []
        for logits in logits_list:
            last = logits[:, -1, :]   # (B, V) — last position only
            if temperature > 0:
                last = last / temperature
            drafts.append(last.argmax(-1))
        return torch.stack(drafts, dim=1)   # (B, N)
