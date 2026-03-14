import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import LeoConfig
from .norm   import RMSNorm


class ECTv3Module(nn.Module):

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D, E             = cfg.hidden_dim, cfg.num_ect
        self.cfg         = cfg
        self.num_ect     = E

        self.ect_embed   = nn.Parameter(torch.randn(E, D) * 0.02)
        self.domain_bias = nn.Parameter(torch.randn(cfg.estr_domain_count, D) * 0.01)

        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        self.W_O = nn.Linear(D, D, bias=False)
        self.norm = RMSNorm(D)

        self.score_proj = nn.Linear(D, E, bias=False)
        self.score_mlp  = nn.Sequential(
            nn.Linear(E, E * 2),
            nn.GELU(),
            nn.Linear(E * 2, 1),
        )

        if cfg.use_prm:
            self.prm_head = nn.Sequential(
                nn.Linear(D, cfg.prm_hidden),
                nn.GELU(),
                nn.Linear(cfg.prm_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.prm_head = None

    def forward(
        self,
        seq_h:    torch.Tensor,
        is_think: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, T, D  = seq_h.shape
        H        = self.cfg.ect_heads

        ect_h = self.ect_embed.unsqueeze(0).expand(B, -1, -1)
        ect_h = ect_h + self.norm(seq_h.mean(1, keepdim=True))

        Q = self.W_Q(ect_h).view(B, self.num_ect, H, D // H).transpose(1, 2)
        K = self.W_K(seq_h).view(B, T, H, D // H).transpose(1, 2)
        V = self.W_V(seq_h).view(B, T, H, D // H).transpose(1, 2)

        attn  = F.scaled_dot_product_attention(Q, K, V)
        ect_h = self.W_O(
            attn.transpose(1, 2).contiguous().view(B, self.num_ect, D)
        )

        proj = self.score_proj(seq_h)
        U    = self.score_mlp(proj).squeeze(-1).sigmoid()

        prm_scores = None
        if self.prm_head is not None:
            prm_raw    = self.prm_head(seq_h).squeeze(-1)
            think_w    = is_think.float() if is_think is not None else torch.ones_like(prm_raw)
            prm_scores = prm_raw * think_w

        return ect_h, U, prm_scores
