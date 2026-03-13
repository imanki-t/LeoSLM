"""
LeoSLM — Selective Diffusion Sampler
======================================
Inference-time iterative unmasking, guided by ECT uncertainty scores.

Unlike standard diffusion that refines ALL tokens over T steps,
LeoSLM's selective sampler ONLY refines positions where ECT
uncertainty exceeds the threshold τ.

This makes hybrid inference:
    - Fast: only uncertain tokens go through diffusion (4–8× fewer steps)
    - Grounded: AR handles confident tokens, diffusion handles uncertain ones
    - Anti-hallucination: uncertain tokens are repeatedly refined

Algorithm (Hybrid Inference):
    1. AR decode full sequence (fast, greedy or sampling)
    2. Run ECT → get uncertainty U per token
    3. Flag uncertain positions: F = {i : U[i] > τ}
    4. For each diffusion step k = 1..K:
        a. Mask flagged positions in current sequence
        b. Run denoising head → predicted tokens
        c. For each flagged position, update with higher confidence predictions
        d. Recompute ECT uncertainty on updated sequence
        e. Remove newly confident positions from F
    5. Return final sequence

Also supports pure diffusion mode (no AR, full sequence).
And self-consistency voting for the most uncertain positions.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import math


class SelectiveDiffusionSampler:
    """
    ECT-guided selective diffusion sampler.

    Args:
        model               : LeoSLM model
        mask_token_id       : id of [MASK] token
        uncertainty_threshold: τ — above this, trigger diffusion (default 0.35)
        num_diffusion_steps : K — number of refinement iterations (default 10)
        temperature         : sampling temperature
        top_k               : top-k filtering (0 = disabled)
        top_p               : nucleus sampling (0 = disabled)
        consistency_votes   : N — for self-consistency voting on most uncertain tokens
        consistency_threshold: positions with U > this get voted on (default 0.7)
    """

    def __init__(
        self,
        model,
        mask_token_id         : int,
        uncertainty_threshold : float = 0.35,
        num_diffusion_steps   : int   = 10,
        temperature           : float = 1.0,
        top_k                 : int   = 50,
        top_p                 : float = 0.9,
        consistency_votes     : int   = 3,
        consistency_threshold : float = 0.70,
    ):
        self.model                  = model
        self.mask_token_id          = mask_token_id
        self.uncertainty_threshold  = uncertainty_threshold
        self.num_diffusion_steps    = num_diffusion_steps
        self.temperature            = temperature
        self.top_k                  = top_k
        self.top_p                  = top_p
        self.consistency_votes      = consistency_votes
        self.consistency_threshold  = consistency_threshold

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def ar_decode(
        self,
        prompt_ids  : torch.Tensor,   # (B, T_prompt) — prompt tokens
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        """
        Greedy/sampling autoregressive decoding.
        Returns (B, T_prompt + max_new_tokens) full sequence.
        """
        self.model.eval()
        B = prompt_ids.shape[0]
        device = prompt_ids.device

        generated = prompt_ids.clone()   # (B, T_prompt)

        for _ in range(max_new_tokens):
            out      = self.model(generated)
            logits   = out["ar_logits"][:, -1, :]    # (B, V) — last position
            logits   = self._filter_logits(logits)
            probs    = F.softmax(logits / self.temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)   # (B, 1)
            generated= torch.cat([generated, next_tok], dim=1)

        return generated

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def hybrid_generate(
        self,
        prompt_ids   : torch.Tensor,      # (B, T_prompt)
        max_new_tokens: int = 128,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full hybrid generation pipeline:
            1. AR decode → draft
            2. ECT uncertainty scan → flag uncertain tokens
            3. Selective diffusion refinement on flagged positions
            4. Self-consistency voting on highest-uncertainty positions

        Returns:
            output_ids : (B, T) — final token sequence
            info       : dict with uncertainty, flagged positions, per-step stats
        """
        self.model.eval()
        device = prompt_ids.device
        B      = prompt_ids.shape[0]

        # ── Step 1: AR draft ─────────────────────────────────────────────────
        draft = self.ar_decode(prompt_ids, max_new_tokens)   # (B, T)
        T     = draft.shape[1]
        prompt_len = prompt_ids.shape[1]

        # ── Step 2: Initial ECT scan ──────────────────────────────────────────
        out         = self.model(draft)
        uncertainty = out["uncertainty"]   # (B, T)

        # Only consider generated part (not prompt)
        uncertainty[:, :prompt_len] = 0.0

        # Flag uncertain positions
        flagged = uncertainty > self.uncertainty_threshold   # (B, T) bool
        num_flagged_initial = flagged.sum().item()

        step_stats = []

        # ── Step 3: Selective diffusion refinement ────────────────────────────
        current = draft.clone()

        for step in range(self.num_diffusion_steps):
            if not flagged.any():
                break   # all confident — stop early

            # Mask flagged positions
            masked = current.clone()
            masked[flagged] = self.mask_token_id

            # Compute noise level: decreasing over steps (annealed)
            t_val = 1.0 - (step + 1) / self.num_diffusion_steps
            t     = torch.full((B,), t_val, device=device)

            # Denoise with diffusion head
            out_d       = self.model(masked, noise_level=t)
            diff_logits = out_d["diff_logits"]   # (B, T, V)

            # Sample predicted tokens at flagged positions
            new_tokens = self._sample_from_logits(diff_logits)  # (B, T)

            # Update flagged positions
            current[flagged] = new_tokens[flagged]

            # Recompute uncertainty
            out_new     = self.model(current)
            uncertainty = out_new["uncertainty"]
            uncertainty[:, :prompt_len] = 0.0

            # Update flagged: remove newly confident positions
            flagged = uncertainty > self.uncertainty_threshold

            step_stats.append({
                "step"       : step,
                "num_flagged": flagged.sum().item(),
                "mean_unc"   : uncertainty[uncertainty > 0].mean().item() if flagged.any() else 0.0,
            })

        # ── Step 4: Self-consistency voting on most uncertain ──────────────────
        very_uncertain = uncertainty > self.consistency_threshold
        if very_uncertain.any() and self.consistency_votes > 1:
            current = self._self_consistency_vote(current, very_uncertain, device)

        info = {
            "initial_flagged" : num_flagged_initial,
            "final_flagged"   : flagged.sum().item(),
            "uncertainty"     : uncertainty,
            "step_stats"      : step_stats,
        }

        return current, info

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def pure_diffusion_generate(
        self,
        prompt_ids    : torch.Tensor,    # (B, T_prompt) — conditioning prefix
        total_len     : int   = 128,     # full sequence length to generate
    ) -> torch.Tensor:
        """
        Pure diffusion generation (no AR).
        Start from fully masked sequence, iteratively unmask.
        Conditioned on prompt_ids (prompt tokens are never masked).

        Returns (B, total_len) full sequence.
        """
        self.model.eval()
        B      = prompt_ids.shape[0]
        device = prompt_ids.device
        T_gen  = total_len - prompt_ids.shape[1]

        if T_gen <= 0:
            return prompt_ids

        # Start fully masked for generation part
        gen_part = torch.full((B, T_gen), self.mask_token_id, device=device)
        sequence = torch.cat([prompt_ids, gen_part], dim=1)   # (B, total_len)
        T        = sequence.shape[1]
        prompt_len = prompt_ids.shape[1]

        # Iterative unmasking from t=1.0 → t=0.0
        for step in range(self.num_diffusion_steps):
            t_val = 1.0 - (step + 1) / self.num_diffusion_steps
            t     = torch.full((B,), t_val, device=device)

            out         = self.model(sequence, noise_level=t)
            diff_logits = out["diff_logits"]   # (B, T, V)

            # Only update masked positions in generation part
            is_masked   = (sequence == self.mask_token_id)
            is_masked[:, :prompt_len] = False    # never touch prompt

            if not is_masked.any():
                break

            # Determine how many to unmask this step
            # Unmask a fraction based on step: more unmasking later in process
            unmask_fraction = (step + 1) / self.num_diffusion_steps
            n_to_unmask     = max(1, int(is_masked.sum().item() * unmask_fraction / self.num_diffusion_steps))

            # Score each masked position: use max-prob as confidence
            probs      = F.softmax(diff_logits / self.temperature, dim=-1)  # (B, T, V)
            max_prob   = probs.max(dim=-1).values                           # (B, T)

            # Pick top-n most confident masked positions to unmask
            mask_scores = max_prob.masked_fill(~is_masked, -1.0)
            _, top_idx  = mask_scores.view(B, -1).topk(min(n_to_unmask, is_masked.sum()), dim=-1)

            # Unmask those positions
            sampled = self._sample_from_logits(diff_logits)
            unmask_mask = torch.zeros_like(sequence, dtype=torch.bool)
            unmask_mask.scatter_(1, top_idx, True)
            unmask_mask &= is_masked

            sequence[unmask_mask] = sampled[unmask_mask]

        return sequence

    # -----------------------------------------------------------------------
    def _self_consistency_vote(
        self,
        current        : torch.Tensor,    # (B, T)
        uncertain_mask : torch.Tensor,    # (B, T) bool — positions to vote on
        device         : torch.device,
    ) -> torch.Tensor:
        """
        Run N diffusion samples at uncertain positions, take majority vote.
        Token-level Best-of-N.
        """
        B, T = current.shape
        V    = self.model.config.vocab_size
        votes = torch.zeros(B, T, V, device=device)

        for _ in range(self.consistency_votes):
            masked = current.clone()
            masked[uncertain_mask] = self.mask_token_id

            with torch.no_grad():
                out   = self.model(masked, noise_level=torch.zeros(B, device=device))
                probs = F.softmax(out["diff_logits"], dim=-1)   # (B, T, V)

            votes += probs   # accumulate probability votes

        # Majority vote: pick highest total probability
        voted_tokens = votes.argmax(dim=-1)   # (B, T)

        # Only update uncertain positions
        result = current.clone()
        result[uncertain_mask] = voted_tokens[uncertain_mask]
        return result

    # -----------------------------------------------------------------------
    def _sample_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample token ids from logit tensor.
        Applies temperature + top-k + top-p filtering.
        Input  : (B, T, V)
        Output : (B, T)
        """
        B, T, V = logits.shape
        logits = logits / self.temperature

        # Top-k
        if self.top_k > 0:
            top_vals, _ = logits.topk(self.top_k, dim=-1)
            threshold   = top_vals[..., -1].unsqueeze(-1)
            logits      = logits.masked_fill(logits < threshold, float("-inf"))

        # Top-p (nucleus)
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove   = cumprobs - F.softmax(sorted_logits, dim=-1) > self.top_p
            remove[..., 0] = False   # always keep top token
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

        probs   = F.softmax(logits, dim=-1)
        flat    = probs.view(B * T, V)
        sampled = torch.multinomial(flat, num_samples=1).squeeze(-1).view(B, T)
        return sampled

    # -----------------------------------------------------------------------
    def _filter_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-k and top-p to (B, V) logits (single step)."""
        if self.top_k > 0:
            top_vals, _ = logits.topk(self.top_k, dim=-1)
            threshold   = top_vals[:, -1].unsqueeze(-1)
            logits      = logits.masked_fill(logits < threshold, float("-inf"))

        if self.top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove   = cumprobs - F.softmax(sorted_logits, dim=-1) > self.top_p
            remove[:, 0] = False
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        return logits
