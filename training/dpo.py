import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from model.config import LeoConfig


class FactualityDPO:
    def __init__(self, cfg: LeoConfig, beta: float = 0.1):
        self.cfg  = cfg
        self.beta = beta

    def _seq_log_prob(
        self,
        logits: torch.Tensor,
        ids:    torch.Tensor,
    ) -> torch.Tensor:
        B, T, V = logits.shape
        target  = ids[:, 1:].contiguous()
        lp = -F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, V),
            target.view(-1),
            reduction="none",
            ignore_index=self.cfg.pad_id,
        ).view(B, T - 1)
        valid = (target != self.cfg.pad_id).float()
        return (lp * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)

    def __call__(
        self,
        chosen_out:   Dict,
        rejected_out: Dict,
        chosen_ids:   torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        lp_c = self._seq_log_prob(chosen_out["ar_logits"],   chosen_ids)
        lp_r = self._seq_log_prob(rejected_out["ar_logits"], rejected_ids)

        U_c = chosen_out["uncertainty"].mean().detach()
        U_r = rejected_out["uncertainty"].mean().detach()
        ect_bonus = (U_r - U_c).clamp(0.0, 1.0) * 0.5

        dpo_loss = -F.logsigmoid(self.beta * (lp_c - lp_r) + ect_bonus).mean()

        ect_cons_c = chosen_out["uncertainty"].var(dim=-1).mean()
        ect_cons_r = rejected_out["uncertainty"].var(dim=-1).mean()
        cons_loss  = (ect_cons_c + ect_cons_r) * 0.01

        total = dpo_loss + cons_loss

        return total, {
            "dpo":         dpo_loss,
            "cons":        cons_loss,
            "total":       total,
            "lp_chosen":   lp_c.mean(),
            "lp_rejected": lp_r.mean(),
            "U_chosen":    U_c,
            "U_rejected":  U_r,
        }
