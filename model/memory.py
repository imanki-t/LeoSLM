import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import LeoConfig
from .norm   import RMSNorm


class TemporalDiffusionMemory(nn.Module):

    def __init__(self, cfg: LeoConfig, memory_size: Optional[int] = None):
        super().__init__()
        D            = cfg.hidden_dim
        self.M       = memory_size if memory_size is not None else cfg.tdm_memory_size
        self.thr     = cfg.tdm_conf_threshold
        self.cmg_thr = cfg.cmg_threshold

        self.mem_q    = nn.Parameter(torch.randn(self.M, D) * 0.02)
        self.mem_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True, bias=False)
        self.mem_norm = RMSNorm(D)

        self.cmg      = nn.Sequential(
            nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.cmg_bias = nn.Parameter(torch.tensor(-2.5))

    def forward(
        self,
        h: torch.Tensor,
        U: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = h.shape

        w        = (1.0 - U.clamp(0, 1)) ** 4
        weighted = h * w.unsqueeze(-1)

        Q   = self.mem_q.unsqueeze(0).expand(B, -1, -1)
        mem, _ = self.mem_attn(Q, weighted, weighted, need_weights=False)
        mem    = self.mem_norm(mem)

        safety = self.cmg(mem) + self.cmg_bias.sigmoid()
        mem    = mem * (safety < self.cmg_thr).float()

        tdm_loss = mem.var(dim=1).mean() * 0.1
        return mem, tdm_loss


class StructuredAgenticMemory(nn.Module):

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

        self.cmg = nn.Sequential(
            nn.Linear(D, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def get_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.slot_q.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def compress_interaction(
        self,
        call_h:   torch.Tensor,
        result_h: torch.Tensor,
    ) -> torch.Tensor:
        pair = torch.cat([call_h, result_h], dim=-1)
        slot = self.pair_norm(self.pair_proj(pair))
        safe = self.cmg(slot)
        return slot * (safe > 0.5).float()

    def forward(
        self,
        h:                 torch.Tensor,
        interaction_slots: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B  = h.shape[0]
        Q  = self.slot_q.unsqueeze(0).expand(B, -1, -1)

        kv_src = (
            torch.cat([interaction_slots, h], dim=1)
            if interaction_slots is not None else h
        )

        sam_mem, _ = self.slot_attn(Q, kv_src, kv_src, need_weights=False)
        sam_mem    = self.slot_norm(sam_mem)
        sam_loss   = -sam_mem.var(dim=1).mean() * 0.01

        return sam_mem, sam_loss
