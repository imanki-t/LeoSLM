"""
LeoSLM — MDLM Loss (Masked Diffusion Language Modeling)
=========================================================
Implements the diffusion training objective using SUBS parameterization
from the MDLM paper (Sahoo et al., NeurIPS 2024).

SUBS (Simplified Unified Bridge Sampler):
    - Model predicts original tokens x_0 from masked input x_t
    - Loss = CrossEntropy(pred_x0, true_x0) at MASKED positions only
    - Weighted by the derivative of the noise schedule: α'(t)
    - This weighting ensures uniform training signal across noise levels

Full training loss for LeoSLM:

    L = L_AR + λ_mdm * L_MDM + λ_ect * L_ECT

    L_AR  = CrossEntropy(ar_logits[:, :-1], tokens[:, 1:])     # next-token prediction
    L_MDM = weighted CE at masked positions (diffusion objective)
    L_ECT = Brier score loss on ECT uncertainty calibration
            (in training/calibration_loss.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict

from .noise_schedule import (
    mask_tokens,
    sample_timesteps,
    sample_timesteps_logit_normal,
    cosine_alpha,
)


# ---------------------------------------------------------------------------
# MDLM Loss
# ---------------------------------------------------------------------------

class MDLMLoss(nn.Module):
    """
    Masked Diffusion Language Model loss.

    Given a clean token sequence x_0:
        1. Sample t ~ Uniform or LogitNormal(0, 1)
        2. Corrupt x_0 → x_t by masking tokens with probability α(t)
        3. Run model forward on x_t → diff_logits
        4. Compute CrossEntropy at masked positions only
        5. Weight by schedule derivative for ELBO lower bound

    Args:
        mask_token_id   : id of [MASK] token
        pad_token_id    : id of [PAD] token
        label_smoothing : label smoothing for CE loss (0 = disabled)
        use_logit_normal: use logit-normal timestep sampling (often better)
        schedule_weight : weight CE loss by |α'(t)| (SUBS weighting)
    """

    def __init__(
        self,
        mask_token_id    : int,
        pad_token_id     : int   = 0,
        label_smoothing  : float = 0.0,
        use_logit_normal : bool  = True,
        schedule_weight  : bool  = True,
    ):
        super().__init__()
        self.mask_token_id    = mask_token_id
        self.pad_token_id     = pad_token_id
        self.label_smoothing  = label_smoothing
        self.use_logit_normal = use_logit_normal
        self.schedule_weight  = schedule_weight

    def _schedule_derivative(self, t: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Numerical derivative of cosine schedule: |dα/dt|.
        Used for SUBS importance weighting.
        Returns (B,) positive weights.
        """
        dt   = eps
        a_t  = cosine_alpha(t)
        a_tp = cosine_alpha((t + dt).clamp(max=1.0))
        deriv = (a_tp - a_t).abs() / dt
        return deriv.clamp(min=1e-6)   # prevent zero weights

    def forward(
        self,
        model_forward_fn,            # callable: (noisy_ids, noise_level) → dict with diff_logits
        clean_ids   : torch.Tensor,  # (B, T) — original clean token sequences
        device      : torch.device,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the MDLM loss for one training step.

        Args:
            model_forward_fn : function that takes (noisy_ids, t) and returns
                               a dict containing at least "diff_logits" (B, T, V)
            clean_ids        : clean token sequences (B, T)
            device           : torch device

        Returns:
            loss    : scalar diffusion loss
            metrics : dict with logging info (frac_masked, t_mean, etc.)
        """
        B, T = clean_ids.shape

        # ── Sample timesteps ────────────────────────────────────────────────
        if self.use_logit_normal:
            t = sample_timesteps_logit_normal(B, device)   # (B,)
        else:
            t = sample_timesteps(B, device)                # (B,)

        # ── Corrupt tokens ──────────────────────────────────────────────────
        noisy_ids, mask_bool = mask_tokens(
            input_ids    = clean_ids,
            t            = t,
            mask_token_id= self.mask_token_id,
            pad_token_id = self.pad_token_id,
        )
        # noisy_ids : (B, T) with some tokens replaced by [MASK]
        # mask_bool : (B, T) True at masked positions

        # ── Model forward pass ──────────────────────────────────────────────
        model_out   = model_forward_fn(noisy_ids, t)
        diff_logits = model_out["diff_logits"]   # (B, T, V)

        # ── Compute loss at masked positions only ────────────────────────────
        # Only compute loss where tokens were masked — elsewhere it's trivial
        if not mask_bool.any():
            # Degenerate case (t≈0, almost no masking) — return tiny loss
            loss = diff_logits.sum() * 0.0
            metrics = {"mdlm_loss": 0.0, "frac_masked": 0.0, "t_mean": t.mean().item()}
            return loss, metrics

        # Flatten for CE computation
        V = diff_logits.shape[-1]
        logits_flat  = diff_logits.view(B * T, V)       # (B*T, V)
        targets_flat = clean_ids.view(B * T)             # (B*T,)
        mask_flat    = mask_bool.view(B * T)             # (B*T,) bool

        # Select only masked positions
        logits_masked  = logits_flat[mask_flat]          # (num_masked, V)
        targets_masked = targets_flat[mask_flat]         # (num_masked,)

        # Cross-entropy at masked positions
        ce_loss = F.cross_entropy(
            logits_masked,
            targets_masked,
            label_smoothing = self.label_smoothing,
            reduction       = "none",                    # per-token, for weighting
        )                                                # (num_masked,)

        # ── SUBS schedule weighting ──────────────────────────────────────────
        if self.schedule_weight:
            # Get weight per sample, then expand to per-masked-token
            weights      = self._schedule_derivative(t)           # (B,)
            # Expand: each token in batch i gets weight weights[i]
            weights_exp  = weights.unsqueeze(1).expand(B, T)      # (B, T)
            weights_flat = weights_exp.reshape(B * T)[mask_flat]  # (num_masked,)
            ce_loss      = ce_loss * weights_flat                  # weighted per-token loss

        loss = ce_loss.mean()

        # ── Metrics ──────────────────────────────────────────────────────────
        with torch.no_grad():
            # Accuracy at masked positions
            preds_masked = logits_masked.argmax(dim=-1)
            acc = (preds_masked == targets_masked).float().mean().item()
            frac_masked = mask_bool.float().mean().item()

        metrics = {
            "mdlm_loss"   : loss.item(),
            "mdlm_acc"    : acc,
            "frac_masked" : frac_masked,
            "t_mean"      : t.mean().item(),
        }

        return loss, metrics


# ---------------------------------------------------------------------------
# AR Loss (standard causal LM loss)
# ---------------------------------------------------------------------------

class ARLoss(nn.Module):
    """
    Standard autoregressive language modeling loss.
    CrossEntropy on next-token prediction: predict token[t+1] given token[1..t].

    Args:
        pad_token_id    : ignored in loss computation
        label_smoothing : label smoothing (0 = disabled)
    """

    def __init__(self, pad_token_id: int = 0, label_smoothing: float = 0.0):
        super().__init__()
        self.pad_token_id    = pad_token_id
        self.label_smoothing = label_smoothing

    def forward(
        self,
        ar_logits: torch.Tensor,   # (B, T, V) — from model LM head
        input_ids: torch.Tensor,   # (B, T) — original token ids
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute AR language modeling loss.

        Labels: shift input_ids by 1 position.
            logits[:, :-1, :] predicts ids[:, 1:]

        Returns:
            loss    : scalar AR loss
            metrics : dict with logging info
        """
        B, T, V = ar_logits.shape

        # Shift: predict token[i+1] from position i
        logits_shifted = ar_logits[:, :-1, :].contiguous()    # (B, T-1, V)
        labels_shifted = input_ids[:, 1:].contiguous()        # (B, T-1)

        # Flatten
        loss = F.cross_entropy(
            logits_shifted.view(-1, V),
            labels_shifted.view(-1),
            ignore_index    = self.pad_token_id,
            label_smoothing = self.label_smoothing,
        )

        # Perplexity
        with torch.no_grad():
            ppl = math.exp(min(loss.item(), 20))   # cap to avoid overflow

        metrics = {
            "ar_loss": loss.item(),
            "ar_ppl" : ppl,
        }

        return loss, metrics


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class LeoLoss(nn.Module):
    """
    Combined training loss for LeoSLM:
        L = L_AR + λ_mdm * L_MDM

    λ_mdm is annealed during training:
        Phase 1: λ_mdm = 0     (AR only)
        Phase 2: λ_mdm → 0.5   (diffusion warmup)
        Phase 3: λ_mdm = 0.5   (joint training)

    ECT calibration loss (L_ECT) is handled separately in training/calibration_loss.py
    and added to this loss in train.py.
    """

    def __init__(
        self,
        mask_token_id : int,
        pad_token_id  : int   = 0,
        lambda_mdm    : float = 0.0,   # start at 0 for Phase 1
        label_smoothing: float= 0.0,
    ):
        super().__init__()
        self.lambda_mdm = lambda_mdm
        self.ar_loss    = ARLoss(pad_token_id=pad_token_id, label_smoothing=label_smoothing)
        self.mdlm_loss  = MDLMLoss(mask_token_id=mask_token_id, pad_token_id=pad_token_id)

    def set_lambda_mdm(self, lam: float):
        """Call this to change MDM weight during training phases."""
        self.lambda_mdm = lam

    def forward(
        self,
        model_out      : dict,           # output dict from LeoSLM.forward()
        input_ids      : torch.Tensor,   # (B, T)
        model_forward_fn = None,         # needed for MDLM (re-runs forward on noisy input)
        device         : torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, Dict]:

        all_metrics = {}

        # ── AR Loss ──────────────────────────────────────────────────────────
        ar_loss, ar_metrics = self.ar_loss(model_out["ar_logits"], input_ids)
        all_metrics.update(ar_metrics)

        total_loss = ar_loss

        # ── MDM Loss ─────────────────────────────────────────────────────────
        if self.lambda_mdm > 0 and model_forward_fn is not None:
            mdlm_loss, mdlm_metrics = self.mdlm_loss(model_forward_fn, input_ids, device)
            all_metrics.update(mdlm_metrics)
            total_loss = total_loss + self.lambda_mdm * mdlm_loss
        else:
            all_metrics["mdlm_loss"] = 0.0

        all_metrics["total_loss"] = total_loss.item()
        all_metrics["lambda_mdm"] = self.lambda_mdm

        return total_loss, all_metrics


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, V = 2, 32, 32768

    # Test AR Loss
    ar_logits = torch.randn(B, T, V)
    ids       = torch.randint(3, V, (B, T))

    ar_loss_fn = ARLoss(pad_token_id=0)
    loss, mets = ar_loss_fn(ar_logits, ids)
    print(f"AR loss: {loss.item():.4f}, PPL: {mets['ar_ppl']:.2f}")

    # Test MDLM Loss
    mask_id = 1
    mdlm_fn = MDLMLoss(mask_token_id=mask_id)

    def fake_model_fn(noisy_ids, t):
        return {"diff_logits": torch.randn(B, T, V)}

    mdlm_loss, mets = mdlm_fn(fake_model_fn, ids, torch.device("cpu"))
    print(f"MDLM loss: {mdlm_loss.item():.4f}")
    print(f"Metrics: {mets}")

    print("\n✅ mdlm_loss.py OK")
