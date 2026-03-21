"""
model/rope.py — YaRN RoPE (Rotary Position Embedding) with dynamic NTK scaling

BUG FIX: removed unused `half` variable (defined but never read).
All shapes verified and documented inline.
"""

import math
import torch
from typing import Tuple


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def build_yarn_rope_cache(
    seq_len:   int,
    head_dim:  int,           # rope dimension per head (e.g. mla_rope_dim=64)
    device:    torch.device,
    base:      float = 500_000.0,
    scale:     float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cosine/sine tables for RoPE.

    Returns:
        cos: (1, seq_len, 1, head_dim//2)
        sin: (1, seq_len, 1, head_dim//2)
    """
    n_freqs = head_dim // 2
    # Base frequencies: (n_freqs,)
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    if scale <= 1.0:
        t   = torch.arange(seq_len, device=device).float()     # (seq_len,)
        emb = torch.outer(t, freqs)                            # (seq_len, n_freqs)
        mscale = 1.0
    else:
        log_s = math.log(scale)
        log_1 = math.log(scale / 1.0)
        low   = max(math.floor(n_freqs * math.log(scale / beta_fast) / log_1), 0)
        high  = min(math.ceil( n_freqs * math.log(scale / beta_slow) / log_1), n_freqs - 1)

        # Interpolation ramp: 0 for low-freq, 1 for high-freq dims
        ramp = torch.zeros(n_freqs, device=device)
        for i in range(n_freqs):
            if i < low:
                ramp[i] = 0.0
            elif i > high:
                ramp[i] = 1.0
            else:
                ramp[i] = (i - low) / max(high - low, 1)

        freqs  = (1 - ramp) * (freqs / scale) + ramp * freqs
        mscale = yarn_get_mscale(scale, mscale=0.1)
        t      = torch.arange(seq_len, device=device).float()
        emb    = torch.outer(t, freqs) * mscale                # (seq_len, n_freqs)

    # Reshape for broadcasting with (B, T, nH, n_freqs)
    cos = emb.cos()[None, :, None, :]   # (1, seq_len, 1, n_freqs)
    sin = emb.sin()[None, :, None, :]   # (1, seq_len, 1, n_freqs)
    return cos, sin


def apply_rope(
    x:   torch.Tensor,   # (B, T, nH, D)  where D = rope head_dim
    cos: torch.Tensor,   # (1 or B, T, 1, D//2)
    sin: torch.Tensor,   # (1 or B, T, 1, D//2)
) -> torch.Tensor:
    """
    Apply RoPE to query or key tensor.
    Uses the 'split-half' rotation convention:
        out = [x_r * cos - x_i * sin,  x_r * sin + x_i * cos]
    """
    B, T, nH, D = x.shape
    xr  = x[..., :D // 2]      # (B, T, nH, D//2)
    xi  = x[..., D // 2:]      # (B, T, nH, D//2)
    cos_ = cos[:, :T]           # trim to actual seq len
    sin_ = sin[:, :T]
    return torch.cat([xr * cos_ - xi * sin_,
                      xr * sin_ + xi * cos_], dim=-1)
