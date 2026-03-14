"""
model/norm.py — RMSNorm
========================
Root Mean Square Layer Normalization.
Used by LLaMA, Qwen, Mistral, and LeoSLM instead of LayerNorm.
Cheaper (no mean subtraction) and XLA-friendly.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMS Layer Normalization — LLaMA / Qwen / Mistral style."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.scale
