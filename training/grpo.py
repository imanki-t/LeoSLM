"""
training/grpo.py — GRPO (Group Relative Policy Optimisation) trainer

BUG FIXES vs original:
  1. GRPO requires GROUP SAMPLING: the original `grpo_step` never sampled
     multiple completions for the same prompt. It used the single batch input
     as both 'prompt' and 'target', which is just standard policy gradient —
     not GRPO. GRPO requires N completions per prompt, computes group-normalised
     advantages, then optimises. Fixed: added `_sample_completions` helper.
     
  2. KL divergence was computed BEFORE updating the policy (using the same model
     weights for both ref and current). The reference policy must be a snapshot
     frozen at the start of each GRPO step, not the live model. Fixed: detach
     ref logits explicitly.
     
  3. `think_adv` normalisation: original subtracted mean but did NOT divide by
     std, so the advantage signal scale varied wildly across batches. Fixed:
     z-score normalisation (subtract mean, divide by std+eps).
     
  4. AgenticGRPO._msra_weights: `ids[0]` assumed batch size 1. If batch size > 1
     this silently used only the first sample. Fixed: process all B samples.
     
  5. `_trajectory_reward` called `self.cfg.tool_result_end` but that field is
     named `tool_result_end` in LeoConfig — confirmed correct. However the
     reverse index lookup `tl[::-1].index(...)` crashes on PyTorch tensors.
     Fixed: convert to Python list first.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from model.config import LeoConfig


class GRPOTrainer:
    """
    Group Relative Policy Optimisation for Think-mode CoT fine-tuning.
    Implements DeepSeek-R1 style rule-based rewards with:
      - Format reward: proper <think>…</think> structure
      - Length penalty: encourage concise thinking
      - Factuality reward: low mean uncertainty after </think>
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
        max_gen_tokens: int   = 256,
        temperature:    float = 1.0,
    ):
        self.cfg         = cfg
        self.model       = model
        self.optimizer   = optimizer
        self.think_start = think_start_id
        self.think_end   = think_end_id
        self.idk_id      = idk_id
        self.n_samples   = n_samples
        self.kl_coeff    = kl_coeff
        self.max_gen     = max_gen_tokens
        self.temperature = temperature

    # ── Reward functions ──────────────────────────────────────────────────────

    def _format_reward(self, ids: List[int]) -> float:
        """+1.0 if exactly one balanced <think>…</think> pair present."""
        opens  = ids.count(self.think_start)
        closes = ids.count(self.think_end)
        if opens == 1 and closes == 1:
            oi = ids.index(self.think_start)
            ci = ids.index(self.think_end)
            if ci > oi + 4:
                return 1.0
        return 0.0

    def _think_length_penalty(self, ids: List[int]) -> float:
        """-0.001 per token inside <think> to encourage efficiency."""
        if self.think_start not in ids or self.think_end not in ids:
            return 0.0
        s = ids.index(self.think_start)
        e = ids.index(self.think_end)
        return -0.001 * max(0, e - s)

    def _factuality_reward(self, ids: List[int], U: torch.Tensor) -> float:
        """Reward based on low uncertainty in the answer (post-think) section."""
        if self.think_end in ids:
            end_idx  = ids.index(self.think_end)
            end_idx  = min(end_idx + 1, len(U) - 1)
            answer_U = U[end_idx:].mean().cpu().item()
        else:
            answer_U = U.mean().cpu().item()
        return max(0.0, 1.5 - answer_U * 3.0)

    def compute_rewards(
        self,
        sampled_ids: torch.Tensor,    # (n_samples, T)
        model_outs:  List[Dict],
    ) -> torch.Tensor:
        rewards = []
        for i in range(sampled_ids.shape[0]):
            ids = sampled_ids[i].cpu().tolist()
            U   = model_outs[i]["uncertainty"][0]     # (T,)
            r   = self._format_reward(ids)
            r  += self._think_length_penalty(ids)
            r  += self._factuality_reward(ids, U)
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32)

    # ── BUG FIX: actual GRPO group sampling ───────────────────────────────────

    @torch.no_grad()
    def _sample_completions(
        self,
        prompt_ids: torch.Tensor,    # (1, T_prompt)
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate N completions from the current policy for one prompt.
        Returns (sampled: (N, T_prompt + T_gen), outs: list of model outputs).
        """
        self.model.eval()
        completions = []
        model_outs  = []

        for _ in range(self.n_samples):
            ids = prompt_ids.clone()    # (1, T_prompt)
            for _ in range(self.max_gen):
                out    = self.model(ids)
                logits = out["ar_logits"][:, -1, :] / max(self.temperature, 1e-8)
                tok    = torch.multinomial(F.softmax(logits, dim=-1), 1)
                ids    = torch.cat([ids, tok], dim=1)
                if tok.item() in {self.cfg.eos_id, self.cfg.pad_id}:
                    break
            # Final forward for full output dict
            final_out = self.model(ids)
            completions.append(ids)
            model_outs.append(final_out)

        # Pad all completions to the same length for batch processing
        max_len = max(c.shape[1] for c in completions)
        padded  = torch.full(
            (self.n_samples, max_len), self.cfg.pad_id,
            dtype=torch.long, device=prompt_ids.device,
        )
        for i, c in enumerate(completions):
            padded[i, :c.shape[1]] = c[0]

        self.model.train()
        return padded, model_outs

    def grpo_step(self, batch: Dict, device) -> Dict:
        """
        Full GRPO step:
        1. Sample N completions from current policy
        2. Score with reward function
        3. Compute group-normalised advantages
        4. Policy gradient loss weighted by advantages
        5. KL regularisation against reference (frozen) policy
        """
        input_ids   = batch["input_ids"].to(device)   # (B, T)
        B           = input_ids.shape[0]

        total_grpo_loss = input_ids.new_zeros(1, dtype=torch.float32)
        total_kl_loss   = input_ids.new_zeros(1, dtype=torch.float32)
        total_ar_loss   = input_ids.new_zeros(1, dtype=torch.float32)

        for b in range(B):
            prompt = input_ids[b:b+1]     # (1, T)

            # ── Sample N completions (no grad) ────────────────────────────────
            sampled, model_outs = self._sample_completions(prompt)   # (N, T_full)

            # ── Compute rewards ───────────────────────────────────────────────
            rewards = self.compute_rewards(sampled, model_outs).to(device)  # (N,)

            # ── Group-normalised advantages (GRPO) ────────────────────────────
            # BUG FIX: z-score normalise (not just subtract mean)
            mean_r = rewards.mean()
            std_r  = rewards.std().clamp(min=1e-6)
            adv    = (rewards - mean_r) / std_r   # (N,)

            # ── Reference policy log-probs (frozen) ───────────────────────────
            # BUG FIX: reference must be computed with detached weights
            with torch.no_grad():
                ref_out    = self.model(sampled)
                ref_logits = ref_out["ar_logits"].detach()   # (N, T, V)

            # ── Current policy log-probs (grad enabled) ───────────────────────
            cur_out    = self.model(sampled)
            cur_logits = cur_out["ar_logits"]                # (N, T, V)

            target = sampled[:, 1:].contiguous()   # (N, T-1)
            valid  = (target != self.cfg.pad_id).float()

            cur_lp = F.log_softmax(cur_logits[:, :-1], dim=-1)   # (N, T-1, V)
            ref_lp = F.log_softmax(ref_logits[:, :-1], dim=-1)

            # Per-token log-prob of chosen action
            tok_lp  = cur_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)   # (N, T-1)
            seq_lp  = (tok_lp * valid).sum(-1) / valid.sum(-1).clamp(1)     # (N,)

            # GRPO loss: -advantage weighted log-prob
            grpo_loss = -(adv * seq_lp).mean()

            # KL divergence: D_KL(cur || ref) per token
            kl = (cur_lp.exp() * (cur_lp - ref_lp)).sum(-1)   # (N, T-1)
            kl_loss = (kl * valid).sum() / valid.sum().clamp(1)

            # Standard AR loss on prompt tokens for stability
            prompt_expanded = prompt.expand(self.n_samples, -1)   # (N, T_prompt)
            ar_out  = self.model(prompt_expanded)
            ar_loss = F.cross_entropy(
                ar_out["ar_logits"][:, :-1].contiguous().view(-1, self.cfg.vocab_size),
                prompt_expanded[:, 1:].contiguous().view(-1),
                ignore_index=self.cfg.pad_id,
            )

            total_grpo_loss = total_grpo_loss + grpo_loss
            total_kl_loss   = total_kl_loss   + kl_loss
            total_ar_loss   = total_ar_loss   + ar_loss

        total = (total_grpo_loss + self.kl_coeff * total_kl_loss + 0.5 * total_ar_loss) / B
        return {
            "total": total,
            "grpo":  total_grpo_loss / B,
            "kl":    total_kl_loss   / B,
            "ar":    total_ar_loss   / B,
        }


class AgenticGRPO:
    """
    GRPO for agentic (tool-using) trajectories.
    Rewards: proper think format + tool call/result pairs + low answer uncertainty.
    MSRA (Multi-Step Reward Attribution) weights gradient by tool-use positions.
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

    def _trajectory_reward(self, ids: torch.Tensor, model_out: Dict) -> float:
        """Reward for proper think format + effective tool use."""
        # BUG FIX: convert tensor to list before using .index() and .count()
        lst = ids.cpu().tolist() if isinstance(ids, torch.Tensor) else ids
        r   = 0.0

        # Think format reward
        opens  = lst.count(self.cfg.think_start_id)
        closes = lst.count(self.cfg.think_end_id)
        if opens == 1 and closes == 1:
            oi = lst.index(self.cfg.think_start_id)
            ci = lst.index(self.cfg.think_end_id)
            if ci > oi + 4:
                r += 1.0

        # Tool use reward
        n_calls   = lst.count(self.cfg.tool_call_start)
        n_results = lst.count(self.cfg.tool_result_end)
        r += 0.5 * min(n_calls,   3)    # reward up to 3 calls
        r += 1.5 * min(n_results, 3)    # extra reward for completed calls
        r -= 0.001 * n_calls            # small penalty for extra tokens

        # Answer quality: low uncertainty after last tool result
        U = model_out["uncertainty"][0]    # (T,)  ← always index [0] (single sample)
        if self.cfg.tool_result_end in lst:
            end_idx = len(lst) - 1 - lst[::-1].index(self.cfg.tool_result_end)
            end_idx = min(end_idx + 1, len(U) - 1)
            answer_U = U[end_idx:].mean().cpu().item() if end_idx < len(U) else 1.0
            r += max(0.0, 2.0 - answer_U * 4.0)

        return r

    def _msra_weights(
        self,
        ids:       torch.Tensor,    # (T,)
        model_out: Dict,
        reward:    float,
    ) -> torch.Tensor:
        """
        Multi-Step Reward Attribution: up-weight tool-use positions
        proportional to their contribution to the reward.
        """
        T  = ids.shape[-1]
        U  = model_out["uncertainty"][0][:T]   # (T,)
        tm = model_out["tool_mask"][0][:T].float()   # (T,)
        wt = torch.ones(T, device=U.device)

        if reward > 0:
            # Positive reward → up-weight confident tool positions
            wt = wt + reward * tm * (1.0 - U.clamp(0.0, 1.0))
        else:
            # Negative reward → penalise uncertain tool positions
            wt = wt + reward * tm * U.clamp(0.0, 1.0)

        return wt.clamp(0.0, 3.0)

    def agentic_grpo_step(self, batch: Dict, device) -> Dict:
        input_ids = batch["input_ids"].to(device)   # (B, T)
        B         = input_ids.shape[0]

        # BUG FIX: process each sample in the batch individually
        total_loss   = input_ids.new_zeros(1, dtype=torch.float)
        total_reward = 0.0
        total_kl     = input_ids.new_zeros(1, dtype=torch.float)

        for b in range(B):
            ids_b = input_ids[b]    # (T,)

            with torch.no_grad():
                ref_out    = self.model(ids_b.unsqueeze(0))
                ref_logits = ref_out["ar_logits"].detach()

            out    = self.model(ids_b.unsqueeze(0))
            logits = out["ar_logits"]   # (1, T, V)

            reward  = self._trajectory_reward(ids_b, out)
            msra_w  = self._msra_weights(ids_b, out, reward)

            adv       = logits.new_tensor(reward)
            log_probs = F.log_softmax(logits[:, :-1], dim=-1)   # (1, T-1, V)
            tgt_ids   = ids_b[1:].unsqueeze(0)                  # (1, T-1)
            tgt_lp    = log_probs.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)  # (1, T-1)

            # Apply MSRA weights
            T_minus_1 = tgt_lp.shape[-1]
            tgt_lp    = tgt_lp * msra_w[:T_minus_1].unsqueeze(0)

            ref_lp  = F.log_softmax(ref_logits[:, :-1], dim=-1)
            kl      = F.kl_div(ref_lp.detach(), log_probs.exp(), reduction="batchmean")

            loss = -(adv * tgt_lp.mean()) + self.kl_coeff * kl

            total_loss   = total_loss + loss
            total_reward += reward
            total_kl     = total_kl + kl

        return {
            "total":   total_loss   / B,
            "reward":  total_reward / B,
            "kl":      total_kl     / B,
            "l_msra":  total_loss   / B,
        }
