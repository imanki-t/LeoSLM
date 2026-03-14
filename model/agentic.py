import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .config import LeoConfig


class AgenticConfidenceGatedInvocation(nn.Module):

    TOOL_CLASSES = [
        "web_search", "code_exec", "mcp_generic", "custom_fn",
        "retrieval",  "calculator", "file_io",    "shell",
    ]

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D              = cfg.hidden_dim
        self.threshold = cfg.acgi_threshold
        self.n_tools   = len(self.TOOL_CLASSES)

        self.tool_embeds = nn.Embedding(self.n_tools, D)

        self.gate = nn.Sequential(
            nn.Linear(D + 1, 64), nn.GELU(),
            nn.Linear(64, 1),     nn.Sigmoid(),
        )

        self.router     = nn.Linear(D, self.n_tools)
        self.estr_align = nn.Linear(cfg.num_ect, self.n_tools, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        U:      torch.Tensor,
        ect_h:  torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, T, D = hidden.shape

        u_feat     = U.unsqueeze(-1)
        gate_score = self.gate(torch.cat([hidden, u_feat], dim=-1)).squeeze(-1)

        gate_mask = (gate_score > 0.5) & (U > self.threshold)

        tool_logits = self.router(hidden)

        ect_domain = ect_h.mean(dim=-1)
        estr_bias  = self.estr_align(ect_domain).unsqueeze(1).expand(-1, T, -1)

        return {
            "gate_score":  gate_score,
            "gate_mask":   gate_mask,
            "tool_logits": tool_logits + estr_bias,
            "estr_bias":   estr_bias,
        }

    def tool_call_loss(
        self,
        gate_score:  torch.Tensor,
        U:           torch.Tensor,
        tool_logits: torch.Tensor,
        tool_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_gate = (U > self.threshold).float()
        l_gate      = F.binary_cross_entropy(gate_score.clamp(1e-6, 1 - 1e-6), target_gate)
        l_tool      = gate_score.new_zeros(1)

        if tool_logits is not None and tool_labels is not None:
            flat_l = tool_logits.view(-1, self.n_tools)
            flat_t = tool_labels.view(-1)
            valid  = (flat_t >= 0).float()
            n      = valid.sum().clamp(min=1)
            ce     = F.cross_entropy(flat_l, flat_t.clamp(min=0), reduction="none")
            l_tool = (ce * valid).sum() / n

        return l_gate + 0.5 * l_tool
