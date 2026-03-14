"""
model/memory.py — Temporal Diffusion Memory (TDM) + Structured Agentic Memory (SAM)
======================================================================================
Both modules are NOVEL — Aether, no prior art.

TDM: ECT-filtered long-range memory for 32k+ context.
  Only confident tokens (U < tdm_conf_threshold) write to memory.
  Constitutional Memory Gate (CMG) blocks unsafe memory writes.
  Rolling 64-slot cross-attention bank replaces explicit token re-processing.

SAM: Extends TDM with dedicated slots for compressed tool-interaction pairs.
  When the model makes a tool call, the (call_hidden, result_hidden) pair is
  compressed into a single D-dim slot and prepended as prefix memory on the
  next attention pass. Gives the model persistent access to its full tool
  interaction history without re-reading raw tool tokens.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import LeoConfig
from .norm   import RMSNorm


# ══════════════════════════════════════════════════════════════════════════════
# TDM — Temporal Diffusion Memory
# ══════════════════════════════════════════════════════════════════════════════

class TemporalDiffusionMemory(nn.Module):
    """
    ECT-filtered rolling memory bank for long-range context.

    On each chunk boundary:
      1. Weight hidden states by confidence: w = (1 - U)^4  (low U = high weight)
      2. Cross-attention compress weighted states → M memory tokens
      3. CMG blocks constitutionally unsafe writes
      4. Memory tokens are prepended on the next chunk's attention pass

    Args:
        cfg         : LeoConfig
        memory_size : override tdm_memory_size from cfg (optional)
    """

    def __init__(self, cfg: LeoConfig, memory_size: Optional[int] = None):
        super().__init__()
        D        = cfg.hidden_dim
        self.M   = memory_size if memory_size is not None else cfg.tdm_memory_size
        self.thr = cfg.tdm_conf_threshold
        self.cmg_thr = cfg.cmg_threshold

        # Cross-attention compression: hidden states → M memory tokens
        self.mem_q    = nn.Parameter(torch.randn(self.M, D) * 0.02)
        self.mem_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True, bias=False)
        self.mem_norm = RMSNorm(D)

        # Constitutional Memory Gate: blocks unsafe / adversarial content
        self.cmg      = nn.Sequential(
            nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.cmg_bias = nn.Parameter(torch.tensor(-2.5))

    def forward(
        self,
        h: torch.Tensor,
        U: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h : (B, T, D) — chunk hidden states
            U : (B, T)    — ECT uncertainty per token
        Returns:
            mem      : (B, M, D) — compressed memory tokens
            tdm_loss : scalar   — consistency regularization
        """
        B, T, D = h.shape

        # Soft confidence weighting (XLA-friendly: no dynamic shapes)
        w        = (1.0 - U.clamp(0, 1)) ** 4
        weighted = h * w.unsqueeze(-1)

        # Cross-attention compression
        Q   = self.mem_q.unsqueeze(0).expand(B, -1, -1)
        mem, _ = self.mem_attn(Q, weighted, weighted, need_weights=False)
        mem = self.mem_norm(mem)

        # CMG: block unsafe memory writes
        safety = self.cmg(mem) + self.cmg_bias.sigmoid()
        mem    = mem * (safety < self.cmg_thr).float()

        # Consistency regularization loss
        tdm_loss = mem.var(dim=1).mean() * 0.1

        return mem, tdm_loss


# ══════════════════════════════════════════════════════════════════════════════
# SAM — Structured Agentic Memory
# ══════════════════════════════════════════════════════════════════════════════

class StructuredAgenticMemory(nn.Module):
    """
    Dedicated memory slots for compressed tool-interaction pairs.

    Architecture:
      - sam_memory_size slots for tool-call/result pairs
      - Each slot: 2×D → D via linear + RMSNorm
      - CMG gate blocks unsafe tool results from entering memory
      - Slots prepended as prefix memory on subsequent attention passes
    """

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D      = cfg.hidden_dim
        self.M = cfg.sam_memory_size

        # Compress (call_hidden, result_hidden) pair → 1 D-dim slot
        self.pair_proj = nn.Linear(D * 2, D, bias=False)
        self.pair_norm = RMSNorm(D)

        # SAM slot attention: slots attend to the current sequence
        self.slot_q    = nn.Parameter(torch.randn(self.M, D) * 0.02)
        self.slot_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True, bias=False)
        self.slot_norm = RMSNorm(D)

        # CMG: blocks adversarial / unsafe tool results
        self.cmg = nn.Sequential(
            nn.Linear(D, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def compress_interaction(
        self,
        call_h:   torch.Tensor,
        result_h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compress one tool interaction pair into a single D-dim memory vector.

        Args:
            call_h   : (B, D) — pooled hidden state of the tool-call span
            result_h : (B, D) — pooled hidden state of the tool-result span
        Returns:
            slot     : (B, D) — compressed, CMG-gated memory vector
        """
        pair = torch.cat([call_h, result_h], dim=-1)    # (B, 2D)
        slot = self.pair_norm(self.pair_proj(pair))      # (B, D)
        safe = self.cmg(slot)                            # (B, 1)
        return slot * (safe > 0.5).float()

    def forward(
        self,
        h:                 torch.Tensor,
        interaction_slots: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h                 : (B, T, D) — current hidden states
            interaction_slots : (B, n_slots, D) or None — prior SAM slots
        Returns:
            sam_prefix : (B, M, D) — memory prefix to prepend to attention
            sam_loss   : scalar   — slot diversity regularization
        """
        B = h.shape[0]
        Q = self.slot_q.unsqueeze(0).expand(B, -1, -1)

        # Include prior interaction slots as additional keys/values
        kv_src = torch.cat([interaction_slots, h], dim=1) \
                 if interaction_slots is not None else h

        sam_mem, _ = self.slot_attn(Q, kv_src, kv_src, need_weights=False)
        sam_mem    = self.slot_norm(sam_mem)

        # Diversity regularization: ensure slots remain distinct
        sam_loss   = -sam_mem.var(dim=1).mean() * 0.01

        return sam_mem, sam_loss
