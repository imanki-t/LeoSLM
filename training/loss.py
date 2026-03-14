"""
training/loss.py — LeoLoss: Complete Loss Function for LeoSLM Aether
======================================================================
Combines 9 loss terms, one per anti-hallucination / alignment mechanism.
All weights are read from LeoConfig — no magic numbers in source code.

Loss schedule by training phase:
    Phase 1 (AR warmup)    : L_AR + w_ect·L_ECT + w_moe·L_MOE
    Phase 2 (Diffusion)    : + λ_mdm·L_MDM
    Phase 3+ (Joint/MoE)   : + w_idk·L_IDK + w_ses·L_SES + w_mtp·L_MTP
    Phase 4+ (SFT)         : + w_prm·L_PRM
    Phase 7+ (Agentic SFT) : + w_acgi·L_ACGI + w_msra·L_MSRA

Loss components:
    L_AR   : autoregressive cross-entropy          (base language modelling)
    L_MDM  : masked diffusion language model       (diffusion warmup)
    L_ECT  : ECT Brier score calibration           (anti-hallucination)
    L_IDK  : [IDK] token training                  (anti-hallucination)
    L_MOE  : MoE load-balance                      (routing efficiency)
    L_SES  : Spectral Expert Specialization        (novel Aether)
    L_TDM  : TDM memory consistency               (baked into aux_loss)
    L_MTP  : Multi-Token Prediction auxiliary      (speculative decoding)
    L_PRM  : Process Reward Model calibration      (think mode quality)
    L_ACGI : ACGI gate calibration                 (agentic safety)
    L_MSRA : Multi-Step Reward Attribution proxy   (agentic RL warmup)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from model.config import LeoConfig


class LeoLoss(nn.Module):

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        self.cfg        = cfg
        self.lambda_mdm = 0.0    # Set per-phase via set_lambda_mdm()

    # ── Core loss terms ───────────────────────────────────────────────────────

    def ar_loss(self, logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Standard next-token cross-entropy, ignoring pad positions.
        XLA-safe: weighted sum instead of boolean indexing (no dynamic shapes,
        no host sync guard clause)."""
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
        """
        Masked Diffusion Language Model loss (MDLM-style).
        Randomly masks 15% of tokens; diff_head must reconstruct originals.
        """
        B, T, V = diff_logits.shape
        mask_m  = torch.bernoulli(torch.full((B, T), rate, device=ids.device)).bool()
        target  = ids.clone()
        target[~mask_m] = self.cfg.pad_id
        l = diff_logits.view(-1, V)
        t = target.view(-1)
        # XLA-safe: weighted CE avoids dynamic boolean indexing + host sync guard
        v = (t != self.cfg.pad_id).float()
        n = v.sum().clamp(min=1)
        return (F.cross_entropy(l, t, reduction="none") * v).sum() / n

    def brier_loss(
        self,
        U:      torch.Tensor,
        logits: torch.Tensor,
        ids:    torch.Tensor,
    ) -> torch.Tensor:
        """
        ECT Brier score calibration: force U to equal per-token error probability.
        Correct predictions → U should be low; wrong predictions → U should be high.
        """
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
        """
        IDK token training: positions with high uncertainty should output [IDK].
        Builds target sequence where high-U next positions are replaced by idk_id.
        """
        B, T, V = logits.shape
        t      = ids[:, 1:].contiguous()
        u      = U[:, :-1]
        l      = logits[:, :-1]
        high_u = (u > self.cfg.uncertainty_thresh).float()
        # XLA-safe: build soft IDK target then weighted CE — no guards, no bool indexing.
        # Positions with high_u=0 get target=pad which the mask zeroes out below.
        idk_tgt = (t * (1 - high_u.long()) + self.cfg.idk_id * high_u.long())
        v = high_u  # weight is 1.0 for high-U positions, 0.0 elsewhere
        n = v.sum().clamp(min=1)
        ce = F.cross_entropy(l.view(-1, V), idk_tgt.view(-1), reduction="none")
        return (ce * v.view(-1)).sum() / n

    def ses_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Spectral Expert Specialization (SES) — novel Aether.
        Forces MoE experts to have orthogonal frequency spectra so each
        expert genuinely specialises in a different "kind" of computation.
        """
        spectra = []
        for block in model.blocks:
            if block.is_moe:
                for ex in block.ffn.experts:
                    W   = ex.gate.weight.float().reshape(-1)
                    fft = torch.fft.rfft(W).abs().pow(2)
                    spectra.append(fft / (fft.norm() + 1e-8))
                break   # Only compute on the first MoE block (efficiency)

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
        """
        Multi-Token Prediction auxiliary loss.
        Head i predicts token at position t + i + 2 (offset = i+2).
        """
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
            # XLA-safe: weighted CE — no dynamic boolean indexing
            m = (t != self.cfg.pad_id).float()
            n = m.sum().clamp(min=1)
            total = total + (F.cross_entropy(l, t, reduction="none") * m).sum() / n
        return total / max(len(mtp_logits), 1)

    def prm_loss(
        self,
        prm_scores: Optional[torch.Tensor],
        think_mask: torch.Tensor,
    ) -> torch.Tensor:
        """PRM calibration — XLA-safe: always compute, mask out non-think positions."""
        if prm_scores is None:
            return think_mask.new_zeros(1, dtype=torch.float)
        # Weighted variance over think tokens — no .sum()==0 host sync guard needed.
        w = think_mask.float()
        n = w.sum().clamp(min=1)
        mean  = (prm_scores * w).sum() / n
        var   = ((prm_scores - mean) ** 2 * w).sum() / n
        return var * 0.1

    def acgi_loss(
        self,
        gate_score:  torch.Tensor,
        U:           torch.Tensor,
        tool_logits: Optional[torch.Tensor] = None,
        tool_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ACGI gate calibration BCE + optional tool routing CE.
        XLA-safe: no .any() host sync; weighted CE for tool routing."""
        target = (U > self.cfg.acgi_threshold).float()
        l_gate = F.binary_cross_entropy(gate_score.clamp(1e-6, 1 - 1e-6), target)
        l_tool = gate_score.new_zeros(1)
        if tool_logits is not None and tool_labels is not None:
            flat_l = tool_logits.view(-1, tool_logits.size(-1))
            flat_t = tool_labels.view(-1)
            # XLA-safe: replace .any() + boolean index with soft weight
            valid  = (flat_t >= 0).float()
            n      = valid.sum().clamp(min=1)
            # Use ignore_index=-1 so CE ignores label=-1 positions natively
            l_tool = F.cross_entropy(flat_l, flat_t.clamp(min=0),
                                     reduction="none")
            l_tool = (l_tool * valid).sum() / n
        return l_gate + 0.5 * l_tool

    def msra_loss(
        self,
        out: Dict,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSRA proxy (Multi-Step Reward Attribution) — supervised pre-training version.
        Minimise ECT uncertainty inside tool-call spans: confident tool tokens
        mean the model knows what it's doing; uncertain tool tokens signal a
        likely hallucinated call.
        Full MSRA RL is implemented in AgenticGRPO (Phase 8).
        """
        tool_mask = out.get("tool_mask")
        U         = out.get("uncertainty")
        if tool_mask is None or U is None:
            ref = U if U is not None else ids
            return ref.new_zeros(1, dtype=torch.float)
        # XLA-safe: always compute, mask zeroes out non-tool positions — no .sum()==0 guard
        tool_u = (U * tool_mask.float()).sum() / tool_mask.float().sum().clamp(min=1)
        return tool_u * 0.1

    # ── Phase-dispatched forward ──────────────────────────────────────────────

    def set_lambda_mdm(self, v: float):
        """Set diffusion loss weight for current phase."""
        self.lambda_mdm = v

    def forward(
        self,
        out:         Dict,
        ids:         torch.Tensor,
        model:       Optional[nn.Module] = None,
        phase:       int = 1,
        tool_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss for the current training phase.

        Args:
            out         : model output dict from LeoSLM.forward()
            ids         : (B, T) input token ids
            model       : LeoSLM instance (needed for SES spectral loss)
            phase       : current training phase (1-8)
            tool_labels : (B, T) tool-class labels for ACGI routing (phase 7+)
        Returns:
            total   : scalar loss
            metrics : dict of per-component loss values for logging
        """
        cfg = self.cfg

        # ── Always active ──────────────────────────────────────────────────────
        l_ar  = self.ar_loss(out["ar_logits"], ids)
        l_ect = self.brier_loss(out["uncertainty"], out["ar_logits"], ids)
        l_moe = out["aux_loss"].mean()

        total = l_ar + cfg.loss_w_ect * l_ect + cfg.loss_w_moe * l_moe
        # NOTE: raw tensors — NO .item() here.  train.py materialises scalars
        # only at logging/checkpoint intervals to avoid per-step XLA host syncs.
        m     = {
            "l_ar":  l_ar,
            "l_ect": l_ect,
            "l_moe": l_moe,
        }

        # ── Phase 2+: diffusion loss ───────────────────────────────────────────
        if phase >= 2 and self.lambda_mdm > 0:
            l_mdm  = self.mdm_loss(out["diff_logits"], ids)
            total  = total + self.lambda_mdm * l_mdm
            m["l_mdm"] = l_mdm

        # ── Phase 3+: IDK + SES + MTP ─────────────────────────────────────────
        if phase >= 3:
            l_idk  = self.idk_loss(out["ar_logits"], out["uncertainty"], ids)
            total  = total + cfg.loss_w_idk * l_idk
            m["l_idk"] = l_idk

            if model is not None:
                l_ses  = self.ses_loss(model)
                total  = total + cfg.loss_w_ses * l_ses
                m["l_ses"] = l_ses

            if out.get("mtp_logits") is not None:
                l_mtp  = self.mtp_loss(out["mtp_logits"], ids)
                total  = total + cfg.loss_w_mtp * l_mtp
                m["l_mtp"] = l_mtp

        # ── Phase 4+: PRM ──────────────────────────────────────────────────────
        if phase >= 4 and out.get("prm_scores") is not None:
            l_prm  = self.prm_loss(out["prm_scores"], out["think_mask"])
            total  = total + cfg.loss_w_prm * l_prm
            m["l_prm"] = l_prm

        # ── Phase 7+: ACGI + MSRA ─────────────────────────────────────────────
        if phase >= 7:
            l_acgi = self.acgi_loss(
                out["acgi_gate"], out["uncertainty"],
                out.get("tool_logits"), tool_labels,
            )
            l_msra = self.msra_loss(out, ids)
            total  = total + cfg.loss_w_acgi * l_acgi + cfg.loss_w_msra * l_msra
            m["l_acgi"] = l_acgi
            m["l_msra"] = l_msra

        m["total"] = total
        return total, m
