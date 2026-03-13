"""
LeoSLM — Confidence Gate
==========================
Merges causal (AR) and bidirectional (Diffusion) attention outputs
using a learned, uncertainty-aware gate α ∈ [0, 1].

    merged = α * bidir_out + (1 - α) * causal_out

When α → 1: bidirectional/diffusion dominates (uncertain, need full context)
When α → 0: causal/AR dominates (confident, fast generation)

The gate α is computed per-token from:
    1. A small linear projection of the ECT uncertainty scores
    2. A direct linear gate on the hidden states themselves
    3. These two signals are combined additively before sigmoid

This makes α differentiable and trainable end-to-end.

Design note:
    During Phase 1 training (AR warmup), the gate is FROZEN at α=0
    so gradients only flow through the causal path.
    We unfreeze it in Phase 2 when the diffusion path is introduced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConfidenceGate(nn.Module):
    """
    Learned gate that merges causal and bidirectional attention outputs.

    Gate computation:
        gate_input = concat(causal_out, bidir_out)  — or just hidden states
        α_hidden   = W_gate(gate_input)              — from hidden state signal
        α_ect      = W_ect(uncertainty)              — from ECT uncertainty
        α           = sigmoid((α_hidden + α_ect) / temperature)

    Args:
        hidden_dim  : model hidden dimension
        temperature : temperature for sigmoid (lower = sharper gate)
        init_bias   : initial bias for gate (negative → starts biased toward AR)
    """

    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 1.0,
        init_bias: float = -2.0,   # negative bias → α starts near 0 (AR dominant)
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.temperature = temperature

        # Gate from hidden states: concat of causal + bidir → scalar per token
        # We concatenate along last dim: (B, T, 2*D) → (B, T, 1)
        self.W_gate = nn.Linear(2 * hidden_dim, 1, bias=True)

        # Gate from ECT uncertainty: scalar uncertainty → gate contribution
        # Simple: scalar → scalar (per token)
        self.W_ect = nn.Linear(1, 1, bias=False)

        # Initialize gate to be AR-dominant at start of training
        nn.init.zeros_(self.W_gate.weight)
        nn.init.constant_(self.W_gate.bias, init_bias)
        nn.init.constant_(self.W_ect.weight, 1.0)

        # RMSNorm on merged output
        self.norm = nn.RMSNorm(hidden_dim)

        # Flag to freeze gate during Phase 1 training
        self._frozen = False

    def freeze(self):
        """Freeze gate: α is fixed at 0 (pure AR path). Call during Phase 1 training."""
        self._frozen = True
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """Unfreeze gate: α becomes learnable. Call at Phase 2 start."""
        self._frozen = False
        for p in self.parameters():
            p.requires_grad = True

    def forward(
        self,
        causal_out  : torch.Tensor,                # (B, T, D) — causal attention output
        bidir_out   : torch.Tensor,                # (B, T, D) — bidirectional attention output
        uncertainty : Optional[torch.Tensor] = None,  # (B, T) — ECT uncertainty scores
    ) -> torch.Tensor:
        """
        Merge causal and bidirectional outputs via the confidence gate.

        Args:
            causal_out  : output from causal (AR) attention path
            bidir_out   : output from bidirectional (Diffusion) attention path
            uncertainty : per-token uncertainty from ECT module, if available

        Returns:
            merged : (B, T, D) — gated combination of both paths
            alpha  : (B, T)    — gate values (for logging and analysis)
        """
        if self._frozen:
            # Phase 1: pure AR, no diffusion contribution
            # Return causal output directly, alpha = 0 everywhere
            alpha = torch.zeros(causal_out.shape[0], causal_out.shape[1],
                                device=causal_out.device)
            return causal_out, alpha

        # ── Compute gate α from hidden states ──────────────────────────────
        gate_input = torch.cat([causal_out, bidir_out], dim=-1)  # (B, T, 2D)
        alpha_hidden = self.W_gate(gate_input)                   # (B, T, 1)

        # ── Add ECT uncertainty signal if available ─────────────────────────
        if uncertainty is not None:
            # (B, T) → (B, T, 1)
            ect_signal  = uncertainty.unsqueeze(-1)
            alpha_ect   = self.W_ect(ect_signal)                 # (B, T, 1)
            alpha_logit = (alpha_hidden + alpha_ect) / self.temperature
        else:
            alpha_logit = alpha_hidden / self.temperature

        alpha = torch.sigmoid(alpha_logit).squeeze(-1)           # (B, T) ∈ [0, 1]

        # ── Merge paths ─────────────────────────────────────────────────────
        # α close to 1 → trust bidirectional (uncertain token, needs diffusion)
        # α close to 0 → trust causal (confident token, AR is fine)
        alpha_3d = alpha.unsqueeze(-1)                           # (B, T, 1) for broadcast
        merged   = alpha_3d * bidir_out + (1.0 - alpha_3d) * causal_out  # (B, T, D)
        merged   = self.norm(merged)

        return merged, alpha


class HardThresholdGate(nn.Module):
    """
    Inference-time hard gate (non-differentiable).
    Used during generation to EXPLICITLY select AR vs. Diffusion per token.

    If uncertainty[i] > threshold → use diffusion refinement for token i
    If uncertainty[i] <= threshold → use AR output directly

    This is NOT used during training (not differentiable).
    Only used in generate.py for the hybrid inference mode.

    Args:
        threshold : uncertainty cutoff (default 0.35, tunable)
    """

    def __init__(self, threshold: float = 0.35):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        uncertainty: torch.Tensor,   # (B, T) — ECT uncertainty scores
    ) -> torch.Tensor:
        """
        Returns a boolean mask: True = needs diffusion refinement.
        Shape: (B, T)
        """
        return uncertainty > self.threshold   # (B, T) bool tensor

    def get_uncertain_positions(
        self,
        uncertainty: torch.Tensor,    # (B, T)
        sequence_ids: torch.Tensor,   # (B, T) — token ids (to know what to mask)
    ):
        """
        Returns indices of positions that need diffusion refinement.
        Used in selective_sampler.py.

        Returns:
            flagged_mask : (B, T) bool — True at uncertain positions
            num_flagged  : int — number of positions to refine (for logging)
        """
        flagged_mask = self.forward(uncertainty)              # (B, T) bool
        num_flagged  = flagged_mask.sum().item()
        return flagged_mask, num_flagged


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, D = 2, 32, 512
    causal  = torch.randn(B, T, D)
    bidir   = torch.randn(B, T, D)
    unc     = torch.rand(B, T)             # fake uncertainty ∈ [0,1]

    gate = ConfidenceGate(hidden_dim=D, temperature=1.0)

    # ── Test normal (unfrozen) mode ──────────────────────────────────────────
    merged, alpha = gate(causal, bidir, uncertainty=unc)
    print(f"Causal shape  : {causal.shape}")
    print(f"Bidir shape   : {bidir.shape}")
    print(f"Merged shape  : {merged.shape}")                 # (2, 32, 512)
    print(f"Alpha shape   : {alpha.shape}")                  # (2, 32)
    print(f"Alpha range   : [{alpha.min():.3f}, {alpha.max():.3f}]")
    print(f"Param count   : {sum(p.numel() for p in gate.parameters()):,}")

    # ── Test frozen (Phase 1) mode ───────────────────────────────────────────
    gate.freeze()
    merged_frozen, alpha_frozen = gate(causal, bidir, uncertainty=unc)
    print(f"\n[Frozen] Alpha : {alpha_frozen.unique()}")      # should be all 0
    print(f"[Frozen] Output == Causal: {torch.allclose(merged_frozen, causal)}")  # True

    # ── Test hard threshold gate ──────────────────────────────────────────────
    hard_gate   = HardThresholdGate(threshold=0.35)
    flagged, n  = hard_gate.get_uncertain_positions(unc, torch.randint(0, 100, (B, T)))
    print(f"\nHard gate flagged: {n} / {B*T} positions ({n/(B*T)*100:.1f}%)")
    print("✅ confidence_gate.py OK")
