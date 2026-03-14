import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from model.config import LeoConfig


class LeoLoss(nn.Module):

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        self.cfg        = cfg
        self.lambda_mdm = 0.0

    def ar_loss(self, logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        B, T, V = logits.shape
        l = logits[:, :-1].contiguous().view(-1, V)
        t = ids[:, 1:].contiguous().view(-1)
        m = (t != self.cfg.pad_id).float()
        n = m.sum().clamp(min=1)
        return (F.cross_entropy(l, t, reduction="none") * m).sum() / n

    def mdm_loss(
        self,
        diff_logits: torch.Tensor,
        ids:         torch.Tensor,
        rate:        float = 0.15,
    ) -> torch.Tensor:
        B, T, V = diff_logits.shape
        mask_m  = torch.bernoulli(torch.full((B, T), rate, device=ids.device)).bool()
        target  = ids.clone()
        target[~mask_m] = self.cfg.pad_id
        l = diff_logits.view(-1, V)
        t = target.view(-1)
        v = (t != self.cfg.pad_id).float()
        n = v.sum().clamp(min=1)
        return (F.cross_entropy(l, t, reduction="none") * v).sum() / n

    def brier_loss(
        self,
        U:      torch.Tensor,
        logits: torch.Tensor,
        ids:    torch.Tensor,
    ) -> torch.Tensor:
        B, T, V = logits.shape
        l = logits[:, :-1].contiguous()
        t = ids[:, 1:].contiguous()
        u = U[:, :-1]
        with torch.no_grad():
            wrong = (l.argmax(-1) != t).float()
            valid = (t != self.cfg.pad_id).float()
        return ((u - wrong) ** 2 * valid).sum() / valid.sum().clamp(min=1)

    def idk_loss(
        self,
        logits: torch.Tensor,
        U:      torch.Tensor,
        ids:    torch.Tensor,
    ) -> torch.Tensor:
        B, T, V = logits.shape
        t      = ids[:, 1:].contiguous()
        u      = U[:, :-1]
        l      = logits[:, :-1]
        high_u = (u > self.cfg.uncertainty_thresh).float()
        idk_tgt = t * (1 - high_u.long()) + self.cfg.idk_id * high_u.long()
        v = high_u
        n = v.sum().clamp(min=1)
        ce = F.cross_entropy(l.view(-1, V), idk_tgt.view(-1), reduction="none")
        return (ce * v.view(-1)).sum() / n

    def ses_loss(self, model: nn.Module) -> torch.Tensor:
        spectra = []
        for block in model.blocks:
            if block.is_moe:
                for ex in block.ffn.experts:
                    W   = ex.gate.weight.float().reshape(-1)
                    fft = torch.fft.rfft(W).abs().pow(2)
                    spectra.append(fft / (fft.norm() + 1e-8))
                break

        if len(spectra) < 2:
            ref = spectra[0] if spectra else model.final_norm.scale
            return ref.new_zeros(1)

        loss = sum(
            F.cosine_similarity(spectra[i].unsqueeze(0), spectra[j].unsqueeze(0))
            for i in range(len(spectra))
            for j in range(i + 1, len(spectra))
        )
        n_pairs = max(1, len(spectra) * (len(spectra) - 1) // 2)
        return loss / n_pairs

    def mtp_loss(
        self,
        mtp_logits: Optional[List[torch.Tensor]],
        ids:        torch.Tensor,
    ) -> torch.Tensor:
        if mtp_logits is None:
            return ids.new_zeros(1, dtype=torch.float)
        total = ids.new_zeros(1, dtype=torch.float)
        B, T  = ids.shape
        for i, logits in enumerate(mtp_logits):
            offset = i + 2
            if offset >= T:
                break
            l = logits[:, :-offset].contiguous().view(-1, logits.size(-1))
            t = ids[:, offset:].contiguous().view(-1)
            m = (t != self.cfg.pad_id).float()
            n = m.sum().clamp(min=1)
            total = total + (F.cross_entropy(l, t, reduction="none") * m).sum() / n
        return total / max(len(mtp_logits), 1)

    def prm_loss(
        self,
        prm_scores: Optional[torch.Tensor],
        think_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prm_scores is None:
            return think_mask.new_zeros(1, dtype=torch.float)
        w    = think_mask.float()
        n    = w.sum().clamp(min=1)
        mean = (prm_scores * w).sum() / n
        var  = ((prm_scores - mean) ** 2 * w).sum() / n
        return var * 0.1

    def acgi_loss(
        self,
        gate_score:  torch.Tensor,
        U:           torch.Tensor,
        tool_logits: Optional[torch.Tensor] = None,
        tool_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = (U > self.cfg.acgi_threshold).float()
        l_gate = F.binary_cross_entropy(gate_score.clamp(1e-6, 1 - 1e-6), target)
        l_tool = gate_score.new_zeros(1)
        if tool_logits is not None and tool_labels is not None:
            flat_l = tool_logits.view(-1, tool_logits.size(-1))
            flat_t = tool_labels.view(-1)
            valid  = (flat_t >= 0).float()
            n      = valid.sum().clamp(min=1)
            ce     = F.cross_entropy(flat_l, flat_t.clamp(min=0), reduction="none")
            l_tool = (ce * valid).sum() / n
        return l_gate + 0.5 * l_tool

    def msra_loss(self, out: Dict, ids: torch.Tensor) -> torch.Tensor:
        tool_mask = out.get("tool_mask")
        U         = out.get("uncertainty")
        if tool_mask is None or U is None:
            ref = U if U is not None else ids
            return ref.new_zeros(1, dtype=torch.float)
        tool_u = (U * tool_mask.float()).sum() / tool_mask.float().sum().clamp(min=1)
        return tool_u * 0.1

    def set_lambda_mdm(self, v: float):
        self.lambda_mdm = v

    def forward(
        self,
        out:         Dict,
        ids:         torch.Tensor,
        model:       Optional[nn.Module] = None,
        phase:       int = 1,
        tool_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        cfg   = self.cfg
        l_ar  = self.ar_loss(out["ar_logits"], ids)
        l_ect = self.brier_loss(out["uncertainty"], out["ar_logits"], ids)
        l_moe = out["aux_loss"].mean()
        total = l_ar + cfg.loss_w_ect * l_ect + cfg.loss_w_moe * l_moe
        m     = {"l_ar": l_ar, "l_ect": l_ect, "l_moe": l_moe}

        if phase >= 2 and self.lambda_mdm > 0:
            l_mdm      = self.mdm_loss(out["diff_logits"], ids)
            total      = total + self.lambda_mdm * l_mdm
            m["l_mdm"] = l_mdm

        if phase >= 3:
            l_idk      = self.idk_loss(out["ar_logits"], out["uncertainty"], ids)
            total      = total + cfg.loss_w_idk * l_idk
            m["l_idk"] = l_idk

            if model is not None:
                l_ses      = self.ses_loss(model)
                total      = total + cfg.loss_w_ses * l_ses
                m["l_ses"] = l_ses

            if out.get("mtp_logits") is not None:
                l_mtp      = self.mtp_loss(out["mtp_logits"], ids)
                total      = total + cfg.loss_w_mtp * l_mtp
                m["l_mtp"] = l_mtp

        if phase >= 4 and out.get("prm_scores") is not None:
            l_prm      = self.prm_loss(out["prm_scores"], out["think_mask"])
            total      = total + cfg.loss_w_prm * l_prm
            m["l_prm"] = l_prm

        if phase >= 7:
            l_acgi      = self.acgi_loss(out["acgi_gate"], out["uncertainty"],
                                          out.get("tool_logits"), tool_labels)
            l_msra      = self.msra_loss(out, ids)
            total       = total + cfg.loss_w_acgi * l_acgi + cfg.loss_w_msra * l_msra
            m["l_acgi"] = l_acgi
            m["l_msra"] = l_msra

        m["total"] = total
        return total, m
