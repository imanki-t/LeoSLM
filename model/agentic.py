"""
model/agentic.py — Agentic Confidence-Gated Invocation (ACGI)

No bugs found. Verified shapes and logic. Added comments.

ACGI is the architectural gate that forces tool invocation when
ECT uncertainty exceeds a threshold.  High uncertainty → the model
MUST call a tool rather than emit a potentially hallucinated answer.

ESTR (ECT-Seeded Tool Routing): the ECT domain representations
bias which tool class to call, routing specialist vs generalist.
"""

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
        self.threshold = cfg.acgi_threshold   # U > threshold → gate fires
        self.n_tools   = len(self.TOOL_CLASSES)

        self.tool_embeds = nn.Embedding(self.n_tools, D)

        # Gate: (D + 1) → 1 probability (include U as explicit feature)
        self.gate = nn.Sequential(
            nn.Linear(D + 1, 64), nn.GELU(),
            nn.Linear(64, 1),     nn.Sigmoid(),
        )

        # Router: which tool class to call
        self.router = nn.Linear(D, self.n_tools)

        # ESTR: ECT domain → tool routing bias
        # ect_h is (B, E, D); mean over D gives (B, E)
        # estr_align: (num_ect=E,) → (n_tools,)
        self.estr_align = nn.Linear(cfg.num_ect, self.n_tools, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,   # (B, T, D)
        U:      torch.Tensor,   # (B, T)
        ect_h:  torch.Tensor,   # (B, E, D)
    ) -> Dict[str, torch.Tensor]:
        B, T, D = hidden.shape

        # Gate probability per position
        u_feat     = U.unsqueeze(-1)                                  # (B, T, 1)
        gate_score = self.gate(torch.cat([hidden, u_feat], dim=-1)).squeeze(-1)  # (B, T)

        # Binary gate mask: fire when both gate and raw uncertainty are high
        gate_mask = (gate_score > 0.5) & (U > self.threshold)        # (B, T) bool

        # Tool class logits
        tool_logits = self.router(hidden)                              # (B, T, n_tools)

        # ESTR bias: ECT domain → tool routing preference
        # ect_h: (B, E, D) → mean over D → (B, E) → estr_align → (B, n_tools)
        ect_domain = ect_h.mean(dim=-1)                               # (B, E)
        estr_bias  = self.estr_align(ect_domain).unsqueeze(1).expand(-1, T, -1)  # (B, T, n_tools)

        return {
            "gate_score":  gate_score,                                # (B, T)
            "gate_mask":   gate_mask,                                 # (B, T)
            "tool_logits": tool_logits + estr_bias,                   # (B, T, n_tools)
            "estr_bias":   estr_bias,                                 # (B, T, n_tools)
        }

    def tool_call_loss(
        self,
        gate_score:  torch.Tensor,            # (B, T)
        U:           torch.Tensor,            # (B, T)
        tool_logits: torch.Tensor,            # (B, T, n_tools)
        tool_labels: Optional[torch.Tensor] = None,  # (B, T) int, -1 = ignore
    ) -> torch.Tensor:
        """
        Gate calibration loss: gate_score should predict whether U > threshold.
        Optionally supervised with ground-truth tool class labels.
        """
        target = (U > self.threshold).float()
        l_gate = F.binary_cross_entropy(gate_score.clamp(1e-6, 1 - 1e-6), target)

        l_tool = gate_score.new_zeros(1)
        if tool_logits is not None and tool_labels is not None:
            flat_l = tool_logits.view(-1, self.n_tools)
            flat_t = tool_labels.view(-1)
            valid  = (flat_t >= 0).float()
            n      = valid.sum().clamp(min=1)
            ce     = F.cross_entropy(flat_l, flat_t.clamp(min=0), reduction="none")
            l_tool = (ce * valid).sum() / n

        return l_gate + 0.5 * l_tool
