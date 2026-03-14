"""
model/agentic.py — ACGI: Agentic Confidence-Gated Invocation
==============================================================
NOVEL — Aether, no prior art.

Architecture-level agentic safety: when ECT uncertainty U > acgi_threshold,
the model is FORCED to emit a <|tool_call|> token rather than generating
an answer token directly. Tool use is an architectural property, not just
a learned behaviour — hallucination under uncertainty is blocked at source.

Gate logic:
    gate_score > 0.5  AND  U > acgi_threshold  →  emit <|tool_call|>
    otherwise                                   →  direct generation

ECT-Seeded Tool Routing (ESTR) biases the tool router toward domain-
specialist ECTs: one ECT per tool domain, so domain-relevant uncertainty
preferentially triggers the correct tool class.

Tool classes: web_search, code_exec, mcp_generic, custom_fn,
              retrieval, calculator, file_io, shell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .config import LeoConfig


class AgenticConfidenceGatedInvocation(nn.Module):
    """
    ACGI module.

    Forward:
        hidden     : (B, T, D)
        U          : (B, T)     — per-token ECT uncertainty
        ect_h      : (B, E, D)  — ECT hidden states (for ESTR routing bias)
    Returns dict:
        gate_score : (B, T)            — continuous gate probability
        gate_mask  : (B, T) bool       — True = emit tool_call_start here
        tool_logits: (B, T, n_tools)   — tool routing logits
        estr_bias  : (B, T, n_tools)   — ECT-seeded routing bias
    """

    TOOL_CLASSES = [
        "web_search", "code_exec", "mcp_generic", "custom_fn",
        "retrieval",  "calculator", "file_io",    "shell",
    ]

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D            = cfg.hidden_dim
        self.threshold = cfg.acgi_threshold
        self.n_tools   = len(self.TOOL_CLASSES)

        self.tool_embeds = nn.Embedding(self.n_tools, D)

        # Gate: (hidden ‖ uncertainty_scalar) → [0,1]
        self.gate = nn.Sequential(
            nn.Linear(D + 1, 64), nn.GELU(),
            nn.Linear(64, 1),     nn.Sigmoid(),
        )

        # Router: which tool class?
        self.router = nn.Linear(D, self.n_tools)

        # ESTR: learned alignment (num_ect → n_tools)
        self.estr_align = nn.Linear(cfg.num_ect, self.n_tools, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        U:      torch.Tensor,
        ect_h:  torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, T, D = hidden.shape

        # Gate input: concatenate hidden with per-token uncertainty scalar
        u_feat     = U.unsqueeze(-1)                           # (B, T, 1)
        gate_score = self.gate(torch.cat([hidden, u_feat], dim=-1)).squeeze(-1)  # (B, T)

        # Hard gate: tool call only when BOTH conditions hold
        gate_mask = (gate_score > 0.5) & (U > self.threshold)

        # Tool routing logits
        tool_logits = self.router(hidden)                      # (B, T, n_tools)

        # ESTR: ECT domain scores → broadcast routing bias
        ect_domain = ect_h.mean(dim=-1)                        # (B, E)
        estr_bias  = self.estr_align(ect_domain)               # (B, n_tools)
        estr_bias  = estr_bias.unsqueeze(1).expand(-1, T, -1)  # (B, T, n_tools)

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
        """
        ACGI training loss:
          1. Gate calibration BCE: gate_score should track (U > threshold).float()
          2. Tool routing CE (when tool_labels available from agentic SFT data)
        """
        target_gate = (U > self.threshold).float()
        l_gate      = F.binary_cross_entropy(gate_score, target_gate)
        l_tool      = gate_score.new_zeros(1)

        if tool_labels is not None:
            flat_l = tool_logits.view(-1, self.n_tools)
            flat_t = tool_labels.view(-1)
            valid  = flat_t >= 0
            if valid.any():
                l_tool = F.cross_entropy(flat_l[valid], flat_t[valid])

        return l_gate + 0.5 * l_tool
