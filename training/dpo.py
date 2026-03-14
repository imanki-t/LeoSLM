"""
training/dpo.py — Factuality-Aware DPO (Phase 5)
==================================================
Based on Lin et al. 2024 and NAACL 2025 factuality preference training.

Creates synthetic preference pairs from the model's OWN outputs:
    Chosen   : response with low ECT uncertainty (confident, likely factual)
    Rejected : response with high ECT uncertainty (uncertain, possibly hallucinated)

Using self-generated pairs avoids distilling new potentially wrong knowledge
into the model — the preference signal comes entirely from the model's
own calibrated uncertainty, not from external labels.

DPO loss:
    L_DPO = -log σ(β × (log p_chosen - log p_rejected))
    + loss_w_fact × relu(U_chosen - U_rejected)   [factuality bonus]

The factuality bonus penalises if the chosen response has HIGHER uncertainty
than the rejected one — which would mean we're training on inverted preferences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from model.config import LeoConfig


class FactualityDPO(nn.Module):
    """
    Factuality-aware Direct Preference Optimization for Phase 5.

    Args:
        cfg  : LeoConfig — reads loss_w_fact, pad_id
        beta : KL penalty coefficient (standard DPO value: 0.1)
    """

    def __init__(self, cfg: LeoConfig, beta: float = 0.1):
        super().__init__()
        self.cfg  = cfg
        self.beta = beta

    def _log_prob(
        self,
        logits: torch.Tensor,
        ids:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean log-probability of ids under logits.

        Args:
            logits : (B, T, V)
            ids    : (B, T)
        Returns:
            (B,) — mean log-prob per sequence, ignoring pad tokens
        """
        B, T, V = logits.shape
        l       = logits[:, :-1].contiguous().view(-1, V)
        t       = ids[:, 1:].contiguous().view(-1)
        m       = (t != self.cfg.pad_id)
        lp      = -F.cross_entropy(l, t, reduction="none")          # (B*(T-1),)
        lp_mat  = lp.view(B, T - 1)
        m_mat   = m.float().view(B, T - 1)
        return (lp_mat * m_mat).sum(-1) / m_mat.sum(-1).clamp(min=1)

    def forward(
        self,
        chosen_out:    Dict,
        rejected_out:  Dict,
        chosen_ids:    torch.Tensor,
        rejected_ids:  torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            chosen_out   : model output dict for the chosen (low-U) response
            rejected_out : model output dict for the rejected (high-U) response
            chosen_ids   : (B, T) token ids of the chosen response
            rejected_ids : (B, T) token ids of the rejected response
        Returns:
            total   : scalar loss
            metrics : dict with dpo, fact_bonus, margin
        """
        lp_chosen   = self._log_prob(chosen_out["ar_logits"],   chosen_ids)
        lp_rejected = self._log_prob(rejected_out["ar_logits"], rejected_ids)

        # DPO loss: maximise log σ(β × (log p_chosen - log p_rejected))
        margin    = lp_chosen - lp_rejected
        dpo_loss  = -F.logsigmoid(self.beta * margin).mean()

        # Factuality bonus: penalise if chosen has HIGHER uncertainty than rejected
        u_chosen   = chosen_out["uncertainty"].mean()
        u_rejected = rejected_out["uncertainty"].mean()
        fact_bonus = F.relu(u_chosen - u_rejected)

        total = dpo_loss + self.cfg.loss_w_fact * fact_bonus
        return total, {
            "dpo":        dpo_loss.item(),
            "fact_bonus": fact_bonus.item(),
            "margin":     margin.mean().item(),
        }
