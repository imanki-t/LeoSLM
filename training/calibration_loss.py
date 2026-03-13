"""
LeoSLM — ECT Calibration Loss
================================
Forces the Epistemic Confidence Tokens to produce WELL-CALIBRATED
uncertainty scores — not just high or low, but accurately reflecting
the probability of being wrong.

What calibration means:
    If a model says "I'm 30% uncertain", it should actually be wrong ~30% of the time.
    An uncalibrated model might always say 5% (overconfident) or 80% (underconfident).
    Hallucination is caused by OVERCONFIDENCE — high certainty despite being wrong.

We use the BRIER SCORE as the calibration loss:
    Brier = (U[i] - error_indicator[i])²

Where:
    U[i]              = ECT uncertainty score for token i ∈ [0, 1]
    error_indicator[i] = 1 if model's top-1 prediction is wrong, else 0

This forces ECTs to output U≈1 when wrong, U≈0 when right.

Additionally we compute Expected Calibration Error (ECE) for evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class ECTCalibrationLoss(nn.Module):
    """
    Brier score loss for ECT uncertainty calibration.

    Trains ECTs to predict per-token error probability:
        - U[i] → 1 when model prediction is wrong
        - U[i] → 0 when model prediction is correct

    This is the anti-hallucination objective:
        A model that knows when it's wrong cannot hallucinate confidently.

    Args:
        lambda_cal  : weight for this loss (default 0.1)
        stop_grad   : if True, stop gradient from flowing through error signal
                      (prevents the model from learning to be wrong on purpose)
    """

    def __init__(self, lambda_cal: float = 0.1, stop_grad: bool = True):
        super().__init__()
        self.lambda_cal = lambda_cal
        self.stop_grad  = stop_grad

    def forward(
        self,
        ar_logits   : torch.Tensor,   # (B, T, V) — AR head predictions
        input_ids   : torch.Tensor,   # (B, T)    — true token ids
        uncertainty : torch.Tensor,   # (B, T)    — ECT uncertainty ∈ [0,1]
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute ECT calibration loss.

        Args:
            ar_logits   : model AR logits from current forward pass
            input_ids   : true next tokens (shifted: input_ids[:, 1:])
            uncertainty : ECT uncertainty scores
            pad_token_id: positions to ignore

        Returns:
            loss    : scalar calibration loss
            metrics : dict with calibration metrics
        """
        B, T, V = ar_logits.shape

        # ── Compute per-token correctness ─────────────────────────────────────
        # Shift: logits[:, :-1] predicts ids[:, 1:]
        logits_shifted = ar_logits[:, :-1, :]         # (B, T-1, V)
        labels_shifted = input_ids[:, 1:].clone()      # (B, T-1)
        unc_shifted    = uncertainty[:, :-1]           # (B, T-1) — align with shifted

        # Top-1 prediction
        preds   = logits_shifted.argmax(dim=-1)        # (B, T-1)
        correct = (preds == labels_shifted)            # (B, T-1) bool

        # Ignore padding
        not_pad = (labels_shifted != pad_token_id)     # (B, T-1) bool
        valid   = not_pad

        if not valid.any():
            dummy = uncertainty.sum() * 0.0
            return dummy, {"ect_cal_loss": 0.0, "ect_cal_acc": 0.0}

        # Error indicator: 1 where model is WRONG, 0 where correct
        error_indicator = (~correct).float()           # (B, T-1), 1=wrong, 0=correct

        if self.stop_grad:
            # Don't let calibration loss teach the model to make more errors
            error_indicator = error_indicator.detach()

        # ── Brier Score: (U[i] - error[i])² at valid positions ───────────────
        brier = (unc_shifted - error_indicator) ** 2   # (B, T-1)
        brier = brier[valid]                           # select valid positions only

        loss = self.lambda_cal * brier.mean()

        # ── Metrics ──────────────────────────────────────────────────────────
        with torch.no_grad():
            # Expected Calibration Error (ECE) — binned
            ece = self._compute_ece(
                uncertainty = unc_shifted[valid],
                is_error    = error_indicator[valid].bool(),
            )
            # How often model is actually correct vs. uncertain
            ar_acc = correct[valid].float().mean().item()

            # Mean uncertainty at correct vs incorrect positions
            unc_correct   = unc_shifted[valid & correct].mean().item()  if (valid & correct).any()  else 0.0
            unc_incorrect = unc_shifted[valid & ~correct].mean().item() if (valid & ~correct).any() else 0.0

        metrics = {
            "ect_cal_loss"   : loss.item(),
            "ect_brier"      : brier.mean().item(),
            "ect_ece"        : ece,
            "ar_acc"         : ar_acc,
            "unc_correct"    : unc_correct,    # should be low (model confident when right)
            "unc_incorrect"  : unc_incorrect,  # should be high (model uncertain when wrong)
        }

        return loss, metrics

    def _compute_ece(
        self,
        uncertainty : torch.Tensor,  # (N,) predicted uncertainty scores
        is_error    : torch.Tensor,  # (N,) bool — True where model was wrong
        n_bins      : int = 10,
    ) -> float:
        """
        Expected Calibration Error: binned calibration metric.
        Perfect calibration: ECE = 0.
        Worst case: ECE = 1.
        """
        ece   = 0.0
        N     = uncertainty.shape[0]
        if N == 0:
            return 0.0

        bin_edges = torch.linspace(0, 1, n_bins + 1, device=uncertainty.device)

        for i in range(n_bins):
            lo, hi    = bin_edges[i], bin_edges[i + 1]
            in_bin    = (uncertainty >= lo) & (uncertainty < hi)
            n_in_bin  = in_bin.sum().item()

            if n_in_bin == 0:
                continue

            # Mean predicted uncertainty in bin
            bin_conf  = uncertainty[in_bin].mean().item()
            # Actual error rate in bin
            bin_err   = is_error[in_bin].float().mean().item()
            # Weighted abs diff
            ece      += (n_in_bin / N) * abs(bin_conf - bin_err)

        return ece


class IDKLoss(nn.Module):
    """
    "I Don't Know" token training loss.

    When ECT uncertainty is very high (> high_unc_threshold),
    penalize predicting any token other than [IDK].
    This trains the model to abstain rather than confabulate.

    Only applied during SFT phase (Phase 4), not pretraining.

    Args:
        idk_token_id       : id of [IDK] token in vocabulary
        high_unc_threshold : uncertainty above this → should predict [IDK]
        lambda_idk         : weight for this loss
    """

    def __init__(
        self,
        idk_token_id       : int,
        high_unc_threshold : float = 0.75,
        lambda_idk         : float = 0.05,
    ):
        super().__init__()
        self.idk_token_id        = idk_token_id
        self.high_unc_threshold  = high_unc_threshold
        self.lambda_idk          = lambda_idk

    def forward(
        self,
        ar_logits   : torch.Tensor,   # (B, T, V)
        uncertainty : torch.Tensor,   # (B, T) — ECT scores
    ) -> Tuple[torch.Tensor, Dict]:
        """
        At positions with very high uncertainty, push ar_logits toward [IDK] token.
        """
        B, T, V = ar_logits.shape

        # Positions where model should say IDK (very uncertain)
        should_idk = (uncertainty > self.high_unc_threshold)   # (B, T) bool

        if not should_idk.any():
            dummy = ar_logits.sum() * 0.0
            return dummy, {"idk_loss": 0.0, "idk_positions": 0}

        # Target: [IDK] token at uncertain positions
        idk_targets = torch.full(
            (should_idk.sum(),),
            self.idk_token_id,
            dtype=torch.long,
            device=ar_logits.device,
        )

        # Loss: push logits at uncertain positions toward [IDK]
        logits_idk = ar_logits[should_idk]    # (N_idk, V)
        loss = self.lambda_idk * F.cross_entropy(logits_idk, idk_targets)

        metrics = {
            "idk_loss"     : loss.item(),
            "idk_positions": should_idk.sum().item(),
        }
        return loss, metrics


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, V = 2, 32, 32768
    ar_logits   = torch.randn(B, T, V)
    input_ids   = torch.randint(3, V, (B, T))
    uncertainty = torch.rand(B, T)

    cal_loss = ECTCalibrationLoss(lambda_cal=0.1)
    loss, mets = cal_loss(ar_logits, input_ids, uncertainty)
    print(f"Calibration loss: {loss.item():.4f}")
    print(f"ECE: {mets['ect_ece']:.4f} (0 = perfect)")
    print(f"Unc @ correct: {mets['unc_correct']:.4f} (want low)")
    print(f"Unc @ incorrect: {mets['unc_incorrect']:.4f} (want high)")

    idk_loss = IDKLoss(idk_token_id=2, high_unc_threshold=0.75)
    loss_idk, mets_idk = idk_loss(ar_logits, uncertainty)
    print(f"\nIDK loss: {loss_idk.item():.4f}")
    print(f"IDK positions: {mets_idk['idk_positions']} / {B*T}")
    print("\n✅ calibration_loss.py OK")
