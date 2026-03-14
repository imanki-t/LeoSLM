"""
model/rope.py — YaRN Rotary Positional Embeddings
===================================================
YaRN (Yet another RoPE extensioN) extends trained context 4× with ~90%
quality retention. Used by Qwen3, Llama 3.3, Mistral Large 2, DeepSeek V3.

Provides:
    yarn_get_mscale        — magnitude correction factor
    build_yarn_rope_cache  — (cos, sin) cache for given seq_len
    apply_rope             — apply (cos, sin) to Q or K tensor
"""

import math
import torch
from typing import Tuple


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """YaRN magnitude scaling factor. Identity at scale=1."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def build_yarn_rope_cache(
    seq_len:   int,
    head_dim:  int,
    device:    torch.device,
    base:      float = 500_000.0,
    scale:     float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (cos, sin) rotary cache with optional YaRN context extension.

    Args:
        base      : RoPE base frequency (500k = DeepSeek V3 / long-context style)
        scale     : 1.0 at train; set to 4.0 at inference for 4× context extension
        beta_fast : per-frequency ramp lower bound (default 32)
        beta_slow : per-frequency ramp upper bound (default 1)

    Returns:
        cos, sin  : each (1, seq_len, 1, head_dim//2)
    """
    half  = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    if scale <= 1.0:
        t      = torch.arange(seq_len, device=device).float()
        emb    = torch.outer(t, freqs)
        mscale = 1.0
    else:
        # YaRN: smooth ramp between linear interpolation (safe) and extrapolation
        log_s  = math.log(scale)
        log_1  = math.log(scale / 1.0)
        low    = max(math.floor(half * math.log(scale / beta_fast) / log_1), 0)
        high   = min(math.ceil(half  * math.log(scale / beta_slow) / log_1), half - 1)

        ramp   = torch.zeros(half, device=device)
        for i in range(half):
            if i < low:
                ramp[i] = 0.0
            elif i > high:
                ramp[i] = 1.0
            else:
                ramp[i] = (i - low) / max(high - low, 1)

        freqs  = (1 - ramp) * (freqs / scale) + ramp * freqs
        mscale = yarn_get_mscale(scale, mscale=0.1)
        t      = torch.arange(seq_len, device=device).float()
        emb    = torch.outer(t, freqs) * mscale

    cos = emb.cos()[None, :, None, :]   # (1, T, 1, D/2)
    sin = emb.sin()[None, :, None, :]
    return cos, sin


def apply_rope(
    x:   torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embeddings in-place style.

    Args:
        x   : (B, T, nH, D)
        cos : (1, T, 1, D/2)
        sin : (1, T, 1, D/2)
    Returns:
        rotated tensor same shape as x
    """
    B, T, nH, D = x.shape
    xr  = x[..., :D // 2]
    xi  = x[..., D // 2:]
    cos_ = cos[:, :T]
    sin_ = sin[:, :T]
    return torch.cat([xr * cos_ - xi * sin_,
                      xr * sin_ + xi * cos_], dim=-1)
