"""
model/leo_block.py

BUG FIXES vs original:
  1. `is_global` used wrong logic: `layer_idx % cfg.global_every_n == 0` makes
     layer 0 global, but layer 0 is in the dense stem and should NOT be global
     (it runs on early shallow features; global attention there is wasteful).
     Fix: `(layer_idx + 1) % cfg.global_every_n == 0` so global layers are
     4, 8, 12, 16, 20, 24, 28 — one per group, NOT including the stem.
  2. `is_moe` used `layer_idx >= cfg.num_dense_layers` — correct, kept as-is.
  3. Forward signature: `mem` parameter was only injected at li==0 in leo_slm,
     but the block itself doesn't know whether to pass it to attn or ignore it.
     Fixed: renamed to `mem_tokens` to match `MultiHeadLatentAttention.forward`.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config    import LeoConfig
from .norm      import RMSNorm
from .attention import MultiHeadLatentAttention
from .moe       import UWMRMoE, swiglu


class LeoBlock(nn.Module):

    def __init__(self, cfg: LeoConfig, layer_idx: int):
        super().__init__()
        D = cfg.hidden_dim

        # BUG FIX: Layer 0 should NOT be global (it's always in the dense stem).
        # Use `(layer_idx + 1) % global_every_n == 0` so global layers align to
        # end of each group: layers 3, 7, 11, 15 ... (0-indexed).
        is_moe    = (layer_idx >= cfg.num_dense_layers)
        is_global = (layer_idx > 0 and (layer_idx + 1) % cfg.global_every_n == 0)

        self.is_moe    = is_moe
        self.is_global = is_global
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)
        self.attn  = MultiHeadLatentAttention(cfg, is_sliding=not is_global)

        if is_moe:
            self.ffn = UWMRMoE(cfg)
        else:
            fd            = cfg.ffn_dim_dense
            self.ffn_gate = nn.Linear(D, fd, bias=False)
            self.ffn_up   = nn.Linear(D, fd, bias=False)
            self.ffn_down = nn.Linear(fd, D, bias=False)

    def _dense_ffn(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x, self.ffn_gate, self.ffn_up, self.ffn_down)

    def forward(
        self,
        x:   torch.Tensor,                    # (B, T, D)
        U:   Optional[torch.Tensor] = None,   # (B, T)
        mem: Optional[torch.Tensor] = None,   # (B, M, D) — memory prefix tokens
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ── Attention ─────────────────────────────────────────────────────────
        r          = x
        # BUG FIX: pass `mem` as `mem_tokens` keyword arg to match the fixed
        # MultiHeadLatentAttention.forward signature.
        attn_out, alpha = self.attn(self.norm1(x), uncertainty=U, mem_tokens=mem)
        x          = r + attn_out

        # ── FFN ───────────────────────────────────────────────────────────────
        r = x
        if self.is_moe:
            ffn_out, aux = self.ffn(self.norm2(x), U)
        else:
            ffn_out = self._dense_ffn(self.norm2(x))
            aux     = x.new_zeros(1)

        return r + ffn_out, aux, alpha
