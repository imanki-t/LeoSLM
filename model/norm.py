"""
model/norm.py — RMSNorm (Root Mean Square Layer Normalization)

No bugs found. Cast to float32 for numerical stability then back to input dtype.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for stability, return in original dtype
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.scale
