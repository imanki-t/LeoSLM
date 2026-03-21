"""
diffusion/mdm.py — Masked Diffusion Language Model (MDM) sampler

No crash bugs found. Verified tensor shapes throughout.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def mdm_noise(
    ids:     torch.Tensor,   # (B, T) — input token IDs
    mask_id: int,
    pad_id:  int,
    rate:    float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask `rate` fraction of non-padding tokens.

    Returns:
        noised:   (B, T) — ids with masked positions replaced by mask_id
        mask_pos: (B, T) bool — True where tokens were masked
    """
    B, T     = ids.shape
    mask_pos = torch.bernoulli(torch.full((B, T), rate, device=ids.device)).bool()
    mask_pos = mask_pos & (ids != pad_id)
    noised   = ids.clone()
    noised[mask_pos] = mask_id
    return noised, mask_pos


def mdm_denoise_step(
    model,
    noised_ids:  torch.Tensor,   # (B, T)
    uncertainty: torch.Tensor,   # (B, T)
    tau:         float,
    mask_id:     int,
    pad_id:      int,
) -> torch.Tensor:
    """One denoising step: re-predict high-uncertainty positions."""
    out     = model(noised_ids)
    new_ids = out["diff_logits"].argmax(-1)   # (B, T)

    # Only update tokens where U > tau
    mask_float = (uncertainty > tau).float()                 # (B, T)
    denoised   = (
        noised_ids * (1 - mask_float.long())
        + new_ids  * mask_float.long()
    )
    # Never touch padding tokens
    denoised[noised_ids == pad_id] = pad_id
    return denoised


class MaskedDiffusionSampler:
    """
    Iterative denoising sampler for the diffusion head.

    Two use modes:
      1. refine(ids): AR-generated ids → iterative uncertainty-guided refinement
      2. sample_from_scratch(shape): generate from all-MASK start (pure diffusion)
    """

    def __init__(
        self,
        model,
        mask_id:  int,
        pad_id:   int,
        tau:      float = 0.5,
        n_steps:  int   = 8,
    ):
        self.model   = model
        self.mask_id = mask_id
        self.pad_id  = pad_id
        self.tau     = tau
        self.n_steps = n_steps

    @torch.no_grad()
    def refine(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Iteratively refine uncertain positions.
        Stops early if no positions have U > tau.
        """
        refined = ids.clone()
        for _ in range(self.n_steps):
            out             = self.model(refined)
            uncertainty     = out["uncertainty"]
            still_uncertain = (uncertainty > self.tau).float()
            if still_uncertain.sum() == 0:
                break
            masked = (
                refined * (1 - still_uncertain.long())
                + self.mask_id * still_uncertain.long()
            )
            masked[refined == self.pad_id] = self.pad_id

            d_out   = self.model(masked)
            new_ids = d_out["diff_logits"].argmax(-1)

            refined = (
                refined * (1 - still_uncertain.long())
                + new_ids * still_uncertain.long()
            )
            refined[ids == self.pad_id] = self.pad_id

        return refined

    @torch.no_grad()
    def sample_from_scratch(
        self,
        shape:    Tuple[int, int],
        device:   torch.device,
        n_passes: int = 4,
    ) -> torch.Tensor:
        """
        Pure diffusion generation: start from all-MASK, iteratively unmask.
        Confidence-scheduled unmasking: reveal most-confident tokens first.
        """
        B, T  = shape
        ids   = torch.full(shape, self.mask_id, device=device, dtype=torch.long)

        for p in range(n_passes):
            frac      = (p + 1) / n_passes
            out       = self.model(ids)
            logits    = out["diff_logits"]    # (B, T, V)
            U         = out["uncertainty"]    # (B, T)

            # Reveal positions where U <= (1 - frac): progressively lower threshold
            threshold = 1.0 - frac
            confident = (U <= threshold).float()

            sampled = torch.multinomial(
                F.softmax(logits.view(-1, logits.shape[-1]), dim=-1), 1
            ).view(B, T)

            ids = (
                ids      * (1 - confident.long())
                + sampled * confident.long()
            )

        # Fill any remaining MASK tokens using argmax
        still_masked = (ids == self.mask_id)
        if still_masked.any():
            out  = self.model(ids)
            best = out["diff_logits"].argmax(-1)
            ids[still_masked] = best[still_masked]

        return ids

    def loss(
        self,
        diff_logits: torch.Tensor,   # (B, T, V)
        ids:         torch.Tensor,   # (B, T)
        pad_id:      int,
        rate:        float = 0.15,
    ) -> torch.Tensor:
        """MDM cross-entropy loss: predict only the masked positions."""
        B, T, V   = diff_logits.shape
        _, mask_pos = mdm_noise(ids, self.mask_id, pad_id, rate)
        target    = ids.clone()
        target[~mask_pos] = pad_id          # only supervise masked positions
        l = diff_logits.view(-1, V)
        t = target.view(-1)
        v = (t != pad_id).float()
        n = v.sum().clamp(min=1)
        return (F.cross_entropy(l, t, reduction="none") * v).sum() / n
