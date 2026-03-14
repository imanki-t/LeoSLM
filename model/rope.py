import math
import torch
from typing import Tuple


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
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
    half  = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    if scale <= 1.0:
        t      = torch.arange(seq_len, device=device).float()
        emb    = torch.outer(t, freqs)
        mscale = 1.0
    else:
        log_s  = math.log(scale)
        log_1  = math.log(scale / 1.0)
        low    = max(math.floor(half * math.log(scale / beta_fast) / log_1), 0)
        high   = min(math.ceil(half  * math.log(scale / beta_slow) / log_1), half - 1)

        ramp = torch.zeros(half, device=device)
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

    cos = emb.cos()[None, :, None, :]
    sin = emb.sin()[None, :, None, :]
    return cos, sin


def apply_rope(
    x:   torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    B, T, nH, D = x.shape
    xr   = x[..., :D // 2]
    xi   = x[..., D // 2:]
    cos_ = cos[:, :T]
    sin_ = sin[:, :T]
    return torch.cat([xr * cos_ - xi * sin_,
                      xr * sin_ + xi * cos_], dim=-1)
