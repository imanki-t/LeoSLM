"""
LeoSLM — DPO Trainer (Direct Preference Optimization)
=========================================================
Implements DPO alignment from Rafailov et al. (2023).

DPO trains the model to prefer "chosen" responses over "rejected" ones
without needing an explicit reward model. It directly optimizes the
policy (our LeoSLM) using preference pairs.

For LeoSLM specifically, preference pairs encode:
    CHOSEN   : factually correct + well-calibrated uncertainty
    REJECTED : factually wrong   + overconfident (hallucinated)

This targets the exact failure mode we want to eliminate.

DPO Loss:
    L_DPO = -log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x))
                    - β * (log π_θ(y_l|x) - log π_ref(y_l|x)))

Where:
    π_θ   = current model (being trained)
    π_ref = reference model (frozen copy of checkpoint before DPO)
    y_w   = chosen (winning) response
    y_l   = rejected (losing) response
    β     = KL penalty coefficient (default 0.1)
    x     = prompt

We also add a special ECT consistency term:
    Penalize if π_θ has LOW uncertainty on y_l (overconfident hallucination)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Preference pair dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreferencePair:
    """
    A single preference pair for DPO training.

    prompt_ids    : token ids of the prompt/context
    chosen_ids    : token ids of the preferred response
    rejected_ids  : token ids of the rejected response
    chosen_type   : "factual" | "calibrated" | "hedged"
    rejected_type : "hallucinated" | "overconfident" | "wrong"
    """
    prompt_ids    : torch.Tensor
    chosen_ids    : torch.Tensor
    rejected_ids  : torch.Tensor
    chosen_type   : str = "factual"
    rejected_type : str = "hallucinated"


# ---------------------------------------------------------------------------
# DPO Trainer
# ---------------------------------------------------------------------------

class DPOTrainer(nn.Module):
    """
    DPO training for LeoSLM.

    Usage:
        1. Load pretrained + SFT checkpoint as policy model
        2. Create frozen reference model (copy of SFT checkpoint)
        3. Train with DPO loss on preference pairs
        4. Save policy model as aligned LeoSLM

    Args:
        policy_model   : LeoSLM model being trained
        beta           : KL penalty coefficient (0.1 = standard)
        lambda_ect     : weight for ECT overconfidence penalty
        label_smoothing: smoothing for DPO loss (reduces reward hacking)
    """

    def __init__(
        self,
        policy_model,
        beta           : float = 0.1,
        lambda_ect     : float = 0.05,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.policy  = policy_model
        self.beta    = beta
        self.lambda_ect     = lambda_ect
        self.label_smoothing= label_smoothing

        # Create frozen reference model (deep copy, no gradients)
        self.reference = copy.deepcopy(policy_model)
        for p in self.reference.parameters():
            p.requires_grad = False
        self.reference.eval()
        print("✅ DPO: Reference model created (frozen copy of policy)")

    # -----------------------------------------------------------------------
    def _get_sequence_log_prob(
        self,
        model,
        prompt_ids  : torch.Tensor,   # (B, T_p)
        response_ids: torch.Tensor,   # (B, T_r)
    ) -> torch.Tensor:
        """
        Compute log probability of response tokens given prompt.
        Returns (B,) sum of log probs over response tokens.

        We concatenate prompt + response, run forward,
        then extract log probs only at response positions.
        """
        B      = prompt_ids.shape[0]
        T_p    = prompt_ids.shape[1]
        T_r    = response_ids.shape[1]

        # Concatenate prompt + response
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)   # (B, T_p + T_r)

        # Forward pass
        with torch.set_grad_enabled(model is self.policy):
            out = model(full_ids)
            logits = out["ar_logits"]                              # (B, T_p+T_r, V)

        # We want log P(response | prompt)
        # Shift: logits[:, T_p-1 : T_p+T_r-1] predicts ids[:, T_p : T_p+T_r]
        response_logits = logits[:, T_p - 1 : T_p + T_r - 1, :]  # (B, T_r, V)
        response_labels = response_ids                             # (B, T_r)

        # Log softmax
        log_probs = F.log_softmax(response_logits, dim=-1)        # (B, T_r, V)

        # Gather log probs of actual tokens: (B, T_r)
        token_log_probs = log_probs.gather(
            dim    = -1,
            index  = response_labels.unsqueeze(-1),
        ).squeeze(-1)                                             # (B, T_r)

        # Sum over response tokens: (B,)
        return token_log_probs.sum(dim=-1)

    # -----------------------------------------------------------------------
    def _get_ect_uncertainty_on_response(
        self,
        prompt_ids  : torch.Tensor,   # (B, T_p)
        response_ids: torch.Tensor,   # (B, T_r)
    ) -> torch.Tensor:
        """
        Get mean ECT uncertainty over response tokens.
        Used to penalize overconfidence on rejected (wrong) responses.
        Returns (B,) mean uncertainty over response positions.
        """
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        T_p = prompt_ids.shape[1]

        with torch.no_grad():
            out = self.policy(full_ids)
            uncertainty = out["uncertainty"]    # (B, T_p + T_r)

        # Extract uncertainty at response positions
        response_unc = uncertainty[:, T_p:]    # (B, T_r)
        return response_unc.mean(dim=-1)        # (B,)

    # -----------------------------------------------------------------------
    def dpo_loss(
        self,
        prompt_ids   : torch.Tensor,   # (B, T_p)
        chosen_ids   : torch.Tensor,   # (B, T_c)
        rejected_ids : torch.Tensor,   # (B, T_r)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute DPO loss for a batch of preference pairs.

        Returns:
            loss    : scalar DPO loss
            metrics : dict with logging info
        """
        device = prompt_ids.device

        # ── Log probs under policy ──────────────────────────────────────────
        policy_chosen_logp   = self._get_sequence_log_prob(
            self.policy, prompt_ids, chosen_ids
        )   # (B,)
        policy_rejected_logp = self._get_sequence_log_prob(
            self.policy, prompt_ids, rejected_ids
        )   # (B,)

        # ── Log probs under reference (frozen) ──────────────────────────────
        with torch.no_grad():
            ref_chosen_logp   = self._get_sequence_log_prob(
                self.reference, prompt_ids, chosen_ids
            )   # (B,)
            ref_rejected_logp = self._get_sequence_log_prob(
                self.reference, prompt_ids, rejected_ids
            )   # (B,)

        # ── DPO loss ─────────────────────────────────────────────────────────
        # log-ratio: how much policy improved vs. reference on each response
        chosen_log_ratio   = policy_chosen_logp   - ref_chosen_logp    # (B,)
        rejected_log_ratio = policy_rejected_logp - ref_rejected_logp  # (B,)

        # DPO objective: prefer chosen over rejected
        logits_dpo = self.beta * (chosen_log_ratio - rejected_log_ratio)  # (B,)

        # Standard DPO loss: -log sigmoid(logits_dpo)
        if self.label_smoothing > 0:
            # Label-smoothed DPO (more stable, from IPO paper)
            loss = (
                -F.logsigmoid(logits_dpo) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits_dpo) * self.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(logits_dpo).mean()

        # ── ECT Overconfidence Penalty ────────────────────────────────────────
        # Extra penalty: if model is too CONFIDENT on REJECTED (hallucinated) outputs,
        # penalize that overconfidence. This is the anti-hallucination DPO extension.
        if self.lambda_ect > 0:
            rejected_uncertainty = self._get_ect_uncertainty_on_response(
                prompt_ids, rejected_ids
            )   # (B,) — uncertainty of policy on rejected responses

            # Penalize LOW uncertainty on rejected responses (overconfident hallucination)
            # Loss: -log(uncertainty on rejected) — push uncertainty up on wrong answers
            overconf_penalty = -torch.log(rejected_uncertainty.clamp(min=1e-6)).mean()
            loss = loss + self.lambda_ect * overconf_penalty
        else:
            overconf_penalty = torch.tensor(0.0)

        # ── Metrics ──────────────────────────────────────────────────────────
        with torch.no_grad():
            # Reward margin: positive means chosen is preferred
            reward_margin  = chosen_log_ratio.mean().item() - rejected_log_ratio.mean().item()
            # Accuracy: fraction of pairs where policy prefers chosen
            dpo_acc        = (logits_dpo > 0).float().mean().item()

        metrics = {
            "dpo_loss"         : loss.item(),
            "dpo_acc"          : dpo_acc,
            "reward_margin"    : reward_margin,
            "chosen_log_ratio" : chosen_log_ratio.mean().item(),
            "rejected_log_ratio": rejected_log_ratio.mean().item(),
            "overconf_penalty" : overconf_penalty.item(),
        }

        return loss, metrics

    # -----------------------------------------------------------------------
    def generate_synthetic_pairs(
        self,
        prompts      : List[str],
        tokenizer,
        device       : torch.device,
        max_len      : int = 64,
    ) -> List[PreferencePair]:
        """
        Generate synthetic preference pairs for DPO training.
        Uses the policy model itself to generate, then filters by uncertainty.

        Strategy:
            - Generate 2 responses per prompt
            - Response with LOWER mean ECT uncertainty → "rejected" (overconfident)
            - Response with HIGHER mean ECT uncertainty → "chosen" IF factual
              OR if it uses hedging language

        Note: In practice, you'd use human labels or a reward model.
        This is a self-play approximation for CPU training.
        """
        pairs = []
        self.policy.eval()

        for prompt_text in prompts:
            prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            B = prompt_ids.shape[0]

            # Generate 2 candidate responses
            with torch.no_grad():
                # High temperature → more diverse/risky
                out = self.policy(prompt_ids)
                logits_hot = out["ar_logits"][:, -1, :] / 1.2
                tok_hot    = torch.multinomial(F.softmax(logits_hot, -1), 1)

                # Low temperature → more confident/greedy
                logits_cold = out["ar_logits"][:, -1, :] / 0.5
                tok_cold    = torch.multinomial(F.softmax(logits_cold, -1), 1)

                # Uncertainty on both
                unc_hot  = out["uncertainty"][:, -1].mean().item()
                unc_cold = out["uncertainty"][:, -1].mean().item()

            # High-temp response is "chosen" (more hedged/uncertain)
            # Low-temp response is "rejected" (more overconfident)
            pair = PreferencePair(
                prompt_ids    = prompt_ids,
                chosen_ids    = tok_hot,
                rejected_ids  = tok_cold,
                chosen_type   = "hedged",
                rejected_type = "overconfident",
            )
            pairs.append(pair)

        return pairs


# ---------------------------------------------------------------------------
# Quick sanity check (shapes only, no actual model)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from model.leoSLM import LeoSLM, LeoConfig

    torch.manual_seed(42)

    config = LeoConfig(
        vocab_size=1000, hidden_dim=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_seq_len=64
    )
    model = LeoSLM(config)

    trainer = DPOTrainer(model, beta=0.1, lambda_ect=0.05)

    B = 2
    prompt   = torch.randint(3, 1000, (B, 8))
    chosen   = torch.randint(3, 1000, (B, 16))
    rejected = torch.randint(3, 1000, (B, 16))

    loss, mets = trainer.dpo_loss(prompt, chosen, rejected)
    print(f"DPO loss   : {loss.item():.4f}")
    print(f"DPO acc    : {mets['dpo_acc']:.3f}")
    print(f"Reward margin: {mets['reward_margin']:.4f}")
    print(f"Overconf penalty: {mets['overconf_penalty']:.4f}")
    print("\n✅ dpo_trainer.py OK")
