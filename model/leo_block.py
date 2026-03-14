"""
model/leo_block.py — LeoBlock: Full Aether Decoder Block
==========================================================
Each block contains:
  1. RMSNorm + MultiHeadLatentAttention (MLA) with confidence gate
     - Sliding-window local attention OR global causal attention (alternating)
     - TDM/SAM memory prefix prepended at the first block of each chunk
     - Dual-path: causal (AR) ⊕ bidirectional (diffusion) gated by α
  2. RMSNorm + FFN
     - Dense SwiGLU for the first num_dense_layers layers
     - UWMR MoE for all remaining layers
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
        D             = cfg.hidden_dim
        is_moe        = (layer_idx >= cfg.num_dense_layers)
        is_global     = (layer_idx % cfg.global_every_n == 0)

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
        x:   torch.Tensor,
        U:   Optional[torch.Tensor] = None,
        mem: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x   : (B, T, D)
            U   : (B, T)     — ECT uncertainty for UWMR routing + EPE
            mem : (B, M, D)  — TDM/SAM memory prefix (injected at first block only)
        Returns:
            x       : (B, T, D)   — updated hidden states
            aux     : scalar      — MoE load-balance loss (0 for dense layers)
            alpha   : (B, T)      — dual-path gate values (for logging)
        """
        # Attention + residual
        r          = x
        attn_out, alpha = self.attn(self.norm1(x), uncertainty=U, mem_tokens=mem)
        x          = r + attn_out

        # FFN + residual
        r = x
        if self.is_moe:
            ffn_out, aux = self.ffn(self.norm2(x), U)
        else:
            ffn_out = self._dense_ffn(self.norm2(x))
            aux     = x.new_zeros(1)

        return r + ffn_out, aux, alpha
