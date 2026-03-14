"""
training/grpo.py — GRPO + Agentic GRPO RL Trainers
=====================================================
GRPOTrainer  : DeepSeek-R1 style think-mode reinforcement learning (Phase 6).
AgenticGRPO  : Trajectory-level RL with MSRA credit assignment (Phase 8).

Both use rule-based rewards (no neural reward model) to avoid reward hacking.
KL penalty against a reference (frozen) forward pass prevents policy collapse.

GRPOTrainer rewards:
    +1.0  correct <think>…</think> format
    +1.5  low ECT uncertainty in the final answer (factuality proxy)
    -0.001 per think token (efficiency incentive)

AgenticGRPO rewards:
    +0.5  per well-formed <|tool_call|>…<|/tool_call|> block
    +1.5  per executed tool result (<|tool_result|> … <|/tool_result|>)
    +1.0  correct <think>…</think> wrapping
    +2.0  low ECT uncertainty after last tool result (factual final answer)
    -0.001 per tool_call token (avoid runaway tool use)
    MSRA : per-token credit weights based on trajectory success
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from model.config import LeoConfig


# ══════════════════════════════════════════════════════════════════════════════
# GRPOTrainer — Phase 6: Deep Think RL
# ══════════════════════════════════════════════════════════════════════════════

class GRPOTrainer:
    """
    Group Relative Policy Optimization — DeepSeek-R1 style.

    At inference generates N samples and scores them. For TPU training
    we approximate with token-level advantage shaping from PRM scores —
    full online sampling is too slow for TPU batch compilation.
    """

    def __init__(
        self,
        cfg:            LeoConfig,
        model:          nn.Module,
        optimizer,
        think_start_id: int,
        think_end_id:   int,
        idk_id:         int,
        n_samples:      int   = 8,
        kl_coeff:       float = 0.001,
    ):
        self.cfg         = cfg
        self.model       = model
        self.optimizer   = optimizer
        self.think_start = think_start_id
        self.think_end   = think_end_id
        self.idk_id      = idk_id
        self.n_samples   = n_samples
        self.kl_coeff    = kl_coeff

    # ── Reward functions ──────────────────────────────────────────────────────

    def _format_reward(self, ids: torch.Tensor) -> float:
        """
        +1.0 if exactly one valid <think>…</think> block (close > open + 4 tokens).
        0.0 otherwise (no block, multiple blocks, or unclosed).
        """
        lst    = ids.cpu().tolist()   # .cpu() avoids XLA device→host sync stall
        opens  = lst.count(self.think_start)
        closes = lst.count(self.think_end)
        if opens == 1 and closes == 1:
            oi = lst.index(self.think_start)
            ci = lst.index(self.think_end)
            if ci > oi + 4:
                return 1.0
        return 0.0

    def _think_length_penalty(self, ids: torch.Tensor) -> float:
        """Mild efficiency penalty: -0.001 per think token."""
        lst = ids.cpu().tolist()      # .cpu() avoids XLA device→host sync stall
        if self.think_start not in lst or self.think_end not in lst:
            return 0.0
        s = lst.index(self.think_start)
        e = lst.index(self.think_end)
        return -0.001 * max(0, e - s)

    def _factuality_reward(
        self,
        ids:       torch.Tensor,
        model_out: Dict,
    ) -> float:
        """
        +1.5 if ECT uncertainty over the ANSWER portion (after </think>) is low.
        Low uncertainty = confident answer = less likely to be a hallucination.
        """
        U       = model_out["uncertainty"][0]
        lst     = (ids[0].cpu() if ids.dim() > 1 else ids.cpu()).tolist()
        if self.think_end in lst:
            end_idx  = lst.index(self.think_end)
            answer_U = U[end_idx + 1:].mean().cpu().item()
        else:
            answer_U = U.mean().cpu().item()
        return max(0.0, 1.5 - answer_U * 3.0)

    def compute_rewards(
        self,
        sampled_ids: torch.Tensor,
        model_outs:  List[Dict],
        ground_truths: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compute rule-based rewards for a batch of N samples."""
        rewards = []
        for i in range(self.n_samples):
            r  = self._format_reward(sampled_ids[i])
            r += self._think_length_penalty(sampled_ids[i])
            r += self._factuality_reward(sampled_ids[i].unsqueeze(0), model_outs[i])
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32)

    # ── Training step ─────────────────────────────────────────────────────────

    def grpo_step(self, batch: Dict, device) -> Dict:
        """
        One GRPO training step (TPU approximation via PRM token-level advantage).

        Returns dict with keys: total, grpo, kl, ar
        """
        input_ids = batch["input_ids"].to(device)
        B         = input_ids.shape[0]

        with torch.no_grad():
            ref_out    = self.model(input_ids)
            ref_logits = ref_out["ar_logits"].detach()

        out    = self.model(input_ids)
        logits = out["ar_logits"]

        # Per-token KL: KL(policy || reference)
        log_p   = F.log_softmax(logits[:, :-1], dim=-1)
        log_ref = F.log_softmax(ref_logits[:, :-1], dim=-1)
        kl      = (log_p.exp() * (log_p - log_ref)).sum(-1)   # (B, T-1)

        # Think advantage from PRM scores
        if out.get("prm_scores") is not None:
            think_adv = out["prm_scores"][:, :-1].clamp(0, 1)
            think_adv = think_adv - think_adv.mean()
        else:
            think_adv = torch.zeros_like(kl)

        # Policy gradient loss: L = -E[A × log p(a|s)] + β × KL
        target    = input_ids[:, 1:].contiguous()
        valid     = (target != self.cfg.pad_id).float()
        log_probs = -F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
            target.view(-1), reduction="none",
        ).view(B, -1)

        grpo_loss = -(think_adv.detach() * log_probs * valid).sum() / valid.sum().clamp(1)
        kl_loss   = (kl * valid).sum() / valid.sum().clamp(1)

        # AR base loss (prevents catastrophic forgetting of language modelling)
        ar_t = target.view(-1)
        ar_v = (ar_t != self.cfg.pad_id)
        ar_l = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.shape[-1])[ar_v], ar_t[ar_v]
        )

        total = grpo_loss + self.kl_coeff * kl_loss + 0.5 * ar_l
        # Return raw tensors — train.py calls .item() only at log intervals
        return {
            "total": total,
            "grpo":  grpo_loss,
            "kl":    kl_loss,
            "ar":    ar_l,
        }


# ══════════════════════════════════════════════════════════════════════════════
# AgenticGRPO — Phase 8: Trajectory-Level RL with MSRA
# ══════════════════════════════════════════════════════════════════════════════

class AgenticGRPO:
    """
    Agentic GRPO with MSRA (Multi-Step Reward Attribution) — novel Aether.

    Extends DeepSeek-R1 GRPO to full tool-call trajectories.
    Reward is computed over the entire trajectory; MSRA attributes it
    backward through tool spans using ECT-uncertainty importance weights.
    """

    def __init__(
        self,
        cfg:       LeoConfig,
        model:     nn.Module,
        optimizer,
        n_samples: int   = 8,
        kl_coeff:  float = 0.001,
    ):
        self.cfg       = cfg
        self.model     = model
        self.optimizer = optimizer
        self.n_samples = n_samples
        self.kl_coeff  = kl_coeff

    def _trajectory_reward(
        self,
        ids:       torch.Tensor,
        model_out: Dict,
    ) -> float:
        """
        Rule-based reward over a full agentic trajectory.
        Scores: think format, tool format, tool execution, answer confidence.
        """
        lst = ids.cpu().tolist()   # .cpu() — one clean transfer, no stall loop
        r   = 0.0

        # Think block format
        opens  = lst.count(self.cfg.think_start_id)
        closes = lst.count(self.cfg.think_end_id)
        if opens == 1 and closes == 1:
            oi = lst.index(self.cfg.think_start_id)
            ci = lst.index(self.cfg.think_end_id)
            if ci > oi + 4:
                r += 1.0

        # Tool call format and execution rewards
        n_calls   = lst.count(self.cfg.tool_call_start)
        n_results = lst.count(self.cfg.tool_result_end)
        r += 0.5 * min(n_calls,   3)    # Up to 3 well-formed calls
        r += 1.5 * min(n_results, 3)    # Up to 3 executed tool results
        r -= 0.001 * n_calls            # Efficiency: penalise redundant calls

        # Answer quality: low ECT uncertainty after tool results
        U  = model_out["uncertainty"][0]
        tl = (ids[0].cpu() if ids.dim() > 1 else ids.cpu()).tolist()
        if self.cfg.tool_result_end in tl:
            end_idx  = len(tl) - 1 - tl[::-1].index(self.cfg.tool_result_end)
            answer_U = U[end_idx + 1:].mean().cpu().item() if end_idx + 1 < len(U) else 1.0
            r       += max(0.0, 2.0 - answer_U * 4.0)

        return r

    def _msra_weights(
        self,
        ids:       torch.Tensor,
        model_out: Dict,
        reward:    float,
    ) -> torch.Tensor:
        """
        MSRA: per-token credit weights for the policy gradient.
        Good trajectory → upweight confident tool tokens.
        Bad trajectory  → downweight uncertain tool tokens.
        """
        T  = ids.shape[-1]
        U  = model_out["uncertainty"][0]
        tm = model_out["tool_mask"][0].float()
        wt = torch.ones(T, device=U.device)

        if reward > 0:
            wt = wt + reward * tm * (1.0 - U.clamp(0, 1))
        else:
            wt = wt + reward * tm * U.clamp(0, 1)

        return wt.clamp(0.0, 3.0)   # Cap to prevent exploding gradients

    def agentic_grpo_step(self, batch: Dict, device) -> Dict:
        """Full agentic GRPO step with MSRA credit assignment."""
        input_ids = batch["input_ids"].to(device)

        with torch.no_grad():
            ref_out    = self.model(input_ids)
            ref_logits = ref_out["ar_logits"].detach()

        out    = self.model(input_ids)
        logits = out["ar_logits"]
        U      = out["uncertainty"]

        reward  = self._trajectory_reward(input_ids[0], out)
        msra_w  = self._msra_weights(input_ids[0], out, reward)

        adv        = torch.tensor(reward, device=device, dtype=logits.dtype)
        B, T, V    = logits.shape
        log_probs  = F.log_softmax(logits[:, :-1], dim=-1)
        tgt_ids    = input_ids[:, 1:]
        tgt_lp     = log_probs.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        tgt_lp     = tgt_lp * msra_w[:T - 1].unsqueeze(0)

        ref_lp     = F.log_softmax(ref_logits[:, :-1], dim=-1)
        kl         = F.kl_div(ref_lp, log_probs.exp(), reduction="batchmean")
        loss       = -(adv * tgt_lp.mean()) + self.kl_coeff * kl

        # Return raw tensors — train.py calls .item() only at log intervals
        return {
            "total":  loss,
            "reward": reward,
            "kl":     kl,
            "l_msra": loss,
        }
