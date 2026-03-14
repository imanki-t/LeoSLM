import torch
import torch.nn as nn
from typing import List

from .config import LeoConfig
from .norm   import RMSNorm


class MultiTokenPredictionHead(nn.Module):

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D      = cfg.hidden_dim
        V      = cfg.vocab_size
        self.N = cfg.mtp_n

        self.heads = nn.ModuleList([
            nn.Sequential(
                RMSNorm(D),
                nn.Linear(D, D, bias=False),
                nn.SiLU(),
                nn.Linear(D, D, bias=False),
            )
            for _ in range(self.N)
        ])

        self.projs = nn.ModuleList([
            nn.Linear(D, V, bias=False) for _ in range(self.N)
        ])

    def forward(self, hidden: torch.Tensor) -> List[torch.Tensor]:
        logits_list = []
        h = hidden
        for i in range(self.N):
            h = self.heads[i](h)
            logits_list.append(self.projs[i](h))
        return logits_list

    def speculative_draft(
        self,
        hidden:      torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        logits_list = self.forward(hidden)
        drafts = []
        for logits in logits_list:
            last = logits[:, -1, :]
            if temperature > 0:
                last = last / temperature
            drafts.append(last.argmax(-1))
        return torch.stack(drafts, dim=1)
