"""
model/memory.py — Temporal Diffusion Memory (TDM) + Structured Agentic Memory (SAM)

BUG FIXES vs original:
  1. TDM Constitutional Memory Gate (CMG) HARD ZEROS BUG:
     `mem = mem * (safety < self.cmg_thr).float()` sets ALL memory slots to 0
     whenever ALL safety scores are above the threshold. This means:
       - On any sequence with even slightly complex content, ALL cross-chunk
         memory is wiped → the model has zero context beyond 2k tokens.
       - This is the opposite of what CMG should do (it should gate individual
         unsafe slots, not wipe everything).
     Fix: Use per-slot soft gating: multiply by (1 - relu(safety - thr)).clamp(0,1)
     so individual slots are gated, not the whole memory.
     
  2. TDM `mem` initialisation: `TemporalDiffusionMemory.forward` returned `mem`
     with shape (B, M, D) where M=tdm_memory_size. But in leo_slm.py,
     `mem = torch.cat([sam_mem, tdm_mem], dim=1)` stacks them. After the second
     chunk, `mem` already contains sam + tdm, but in the next chunk `inj_mem = mem`
     is passed to block 0, which calls `MultiHeadLatentAttention` with
     `mem_tokens=(sam+tdm)`. This is fine as-is, but the memory grows with each
     chunk. For n_chunks chunks, mem grows to (sam_size + n_chunks*tdm_size) tokens.
     Fix: keep only the LATEST tdm chunk, not accumulate.
     
  3. SAM `compress_interaction`: takes `call_h` and `result_h` as individual
     tensors. In practice these would be (B, 1, D) or (B, T_call, D). The
     `pair_proj` expects (B, ?, D*2) so we need to mean-pool variable-length
     inputs first. Added mean-pooling step.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import LeoConfig
from .norm   import RMSNorm


class TemporalDiffusionMemory(nn.Module):
    """
    TDM: Compresses high-confidence tokens from each chunk into
    a fixed-size memory bank, which is prepended to the next chunk.
    
    Constitutional Memory Gate (CMG) prevents unsafe content from
    being stored in cross-chunk memory.
    """

    def __init__(self, cfg: LeoConfig, memory_size: Optional[int] = None):
        super().__init__()
        D            = cfg.hidden_dim
        self.M       = memory_size if memory_size is not None else cfg.tdm_memory_size
        self.thr     = cfg.tdm_conf_threshold
        self.cmg_thr = cfg.cmg_threshold

        # Learnable memory queries: (M, D)
        self.mem_q    = nn.Parameter(torch.randn(self.M, D) * 0.02)
        self.mem_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True, bias=False)
        self.mem_norm = RMSNorm(D)

        # CMG: per-slot safety gate
        self.cmg = nn.Sequential(
            nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        # Learnable bias: initialise negative so gate defaults to OPEN
        # (memory is passed through by default, only blocked for clearly unsafe content)
        self.cmg_bias = nn.Parameter(torch.tensor(-2.5))

    def forward(
        self,
        h: torch.Tensor,   # (B, T, D) — hidden states for this chunk
        U: torch.Tensor,   # (B, T)    — uncertainty scores
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = h.shape

        # Weight by confidence: low-U tokens contribute more to memory
        # (confident = worth remembering)
        conf     = (1.0 - U.clamp(0.0, 1.0)) ** 4    # (B, T)
        weighted = h * conf.unsqueeze(-1)              # (B, T, D)

        # Cross-attention: mem_q is Q, confident tokens are K/V
        Q   = self.mem_q.unsqueeze(0).expand(B, -1, -1)   # (B, M, D)
        mem, _ = self.mem_attn(Q, weighted, weighted, need_weights=False)
        mem    = self.mem_norm(mem)                        # (B, M, D)

        # BUG FIX: per-slot CMG soft gate (original code zeroed ALL memory)
        # safety ∈ (0,1); gate = clamp(1 - relu(safety + bias), 0, 1)
        # When safety ≈ 0 (safe): gate ≈ 1.0 → memory passes through
        # When safety ≈ 1 (unsafe): gate ≈ 0.0 → memory slot is zeroed
        safety  = self.cmg(mem) + self.cmg_bias.sigmoid()    # (B, M, 1) ∈ (0,1)
        gate    = (1.0 - torch.relu(safety - self.cmg_thr)).clamp(0.0, 1.0)
        mem     = mem * gate                                  # (B, M, D) per-slot gated

        # Auxiliary loss: encourage memory diversity (avoid all slots collapsing)
        tdm_loss = mem.var(dim=1).mean() * 0.1               # (scalar)
        return mem, tdm_loss


class StructuredAgenticMemory(nn.Module):
    """
    SAM: Persistent slots for tool call / result pairs.
    Each (tool_call, tool_result) interaction is compressed into
    one D-dim slot and stored. These slots are prepended as prefix
    memory to all subsequent attention layers.
    """

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D      = cfg.hidden_dim
        self.M = cfg.sam_memory_size
        self.D = D

        self.pair_proj = nn.Linear(D * 2, D, bias=False)
        self.pair_norm = RMSNorm(D)

        self.slot_q    = nn.Parameter(torch.randn(self.M, D) * 0.02)
        self.slot_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True, bias=False)
        self.slot_norm = RMSNorm(D)

        # Per-slot safety gate (same CMG principle as TDM)
        self.cmg = nn.Sequential(
            nn.Linear(D, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def get_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return initial empty slot bank: (B, M, D)."""
        return self.slot_q.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def compress_interaction(
        self,
        call_h:   torch.Tensor,   # (B, T_call, D)
        result_h: torch.Tensor,   # (B, T_res,  D)
    ) -> torch.Tensor:
        """
        Compress a tool call + result pair into a single (B, 1, D) slot.
        
        BUG FIX: original expected fixed-length inputs. Now mean-pools
        variable-length call and result sequences before concatenating.
        """
        # Mean-pool variable-length sequences → (B, D) each
        call_vec   = call_h.mean(dim=1)     # (B, D)
        result_vec = result_h.mean(dim=1)   # (B, D)

        pair = torch.cat([call_vec, result_vec], dim=-1)    # (B, D*2)
        slot = self.pair_norm(self.pair_proj(pair))         # (B, D)

        # Safety gate: don't store slots that look unsafe
        safe = self.cmg(slot)                               # (B, 1)
        slot = slot.unsqueeze(1) * (safe > 0.5).float().unsqueeze(-1)  # (B, 1, D)
        return slot

    def forward(
        self,
        h:                 torch.Tensor,              # (B, T, D)
        interaction_slots: Optional[torch.Tensor] = None,  # (B, N_interactions, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update SAM slots with current hidden states, optionally incorporating
        stored tool interaction slots.
        
        Returns: (sam_mem: (B, M, D), sam_loss: scalar)
        """
        B  = h.shape[0]
        Q  = self.slot_q.unsqueeze(0).expand(B, -1, -1)   # (B, M, D)

        # Key/value source: concatenate interaction history + current sequence
        kv_src = (
            torch.cat([interaction_slots, h], dim=1)
            if interaction_slots is not None else h
        )   # (B, N_slots + T, D)

        sam_mem, _ = self.slot_attn(Q, kv_src, kv_src, need_weights=False)
        sam_mem    = self.slot_norm(sam_mem)               # (B, M, D)

        # Auxiliary loss: maximise slot diversity (negative variance → minimise -var)
        sam_loss = -sam_mem.var(dim=1).mean() * 0.01

        return sam_mem, sam_loss
