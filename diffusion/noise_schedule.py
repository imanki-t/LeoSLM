"""
LeoSLM — Noise Schedule
=========================
Cosine masking schedule for Masked Diffusion Language Modeling (MDLM).

Instead of Gaussian noise (which doesn't work on discrete tokens),
we use TOKEN MASKING as the corruption process:

    Forward process q(x_t | x_0):
        At time t ∈ [0, 1], each token is independently masked
        with probability α(t), where α(t) follows a cosine schedule.

    α(0) = 0   → no tokens masked (clean input)
    α(1) = 1   → all tokens masked (pure noise)

The cosine schedule ensures smooth interpolation and avoids
collapse at the boundaries (unlike linear schedule).

SUBS Parameterization (from MDLM paper):
    The model predicts x_0 (original tokens) from x_t (noisy tokens).
    At each step, unmasked tokens are kept, masked tokens are predicted.

References:
    - MDLM: Masked Diffusion Language Models (Sahoo et al., NeurIPS 2024)
    - SUBS: Simplified and Unified Bridge Sampler
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Cosine Schedule
# ---------------------------------------------------------------------------

def cosine_alpha(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """
    Cosine masking rate α(t) ∈ [0, 1].
    α(t) = 1 - cos²(π/2 * (t + s) / (1 + s))

    Higher α → more tokens masked.
    s is a small offset to prevent α(0) from being exactly 0 (numerical stability).

    Args:
        t : tensor of timesteps ∈ [0, 1], any shape
        s : small offset (default 0.008, from DDPM paper)

    Returns:
        alpha : masking rate, same shape as t, values in [0, 1]
    """
    numerator   = t + s
    denominator = 1.0 + s
    alpha = 1.0 - torch.cos(math.pi / 2.0 * numerator / denominator) ** 2
    return alpha.clamp(0.0, 1.0)


def linear_alpha(t: torch.Tensor) -> torch.Tensor:
    """Linear schedule (simpler, less smooth). Provided as alternative."""
    return t.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Masking Functions
# ---------------------------------------------------------------------------

def mask_tokens(
    input_ids    : torch.Tensor,      # (B, T) — original token ids
    t            : torch.Tensor,      # (B,)   — noise level per sample ∈ [0, 1]
    mask_token_id: int,
    pad_token_id : int   = 0,
    special_ids  : Optional[list] = None,   # token ids to never mask
    schedule     : str   = "cosine",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply masking corruption to input tokens for training the diffusion head.

    Each token is independently masked with probability α(t).
    Special tokens (pad, BOS, EOS) are never masked.

    Args:
        input_ids     : original clean token sequences  (B, T)
        t             : per-sample noise level ∈ [0,1]  (B,)
        mask_token_id : id of [MASK] token
        pad_token_id  : id of [PAD] token (never masked)
        special_ids   : list of additional token ids to protect from masking
        schedule      : "cosine" or "linear"

    Returns:
        masked_ids  : (B, T) — corrupted input (some tokens replaced with [MASK])
        mask_bool   : (B, T) — True where token was masked (loss computed here)
    """
    B, T = input_ids.shape
    device = input_ids.device

    # Compute masking probability per sample: α(t) → (B,)
    if schedule == "cosine":
        alpha = cosine_alpha(t)         # (B,)
    else:
        alpha = linear_alpha(t)         # (B,)

    # Expand to (B, T) for per-token sampling
    alpha_expanded = alpha.unsqueeze(1).expand(B, T)    # (B, T)

    # Sample mask: True = mask this token
    mask_bool = torch.bernoulli(alpha_expanded).bool()  # (B, T)

    # Never mask special tokens
    protect = (input_ids == pad_token_id)
    if special_ids:
        for sid in special_ids:
            protect = protect | (input_ids == sid)
    mask_bool = mask_bool & ~protect                    # (B, T)

    # Apply mask: replace masked positions with mask_token_id
    masked_ids = input_ids.clone()
    masked_ids[mask_bool] = mask_token_id

    return masked_ids, mask_bool


def unmask_tokens(
    masked_ids   : torch.Tensor,      # (B, T) — current noisy sequence
    pred_logits  : torch.Tensor,      # (B, T, V) — model predictions
    mask_token_id: int,
    temperature  : float = 1.0,
    top_k        : int   = 0,         # 0 = disabled
) -> torch.Tensor:
    """
    One step of the reverse diffusion process: unmask some tokens.
    Samples from predicted distribution at masked positions.
    Unmasked positions are kept as-is.

    Args:
        masked_ids    : current (noisy) token sequence (B, T)
        pred_logits   : model's predicted token distribution (B, T, V)
        mask_token_id : id of [MASK] token
        temperature   : sampling temperature
        top_k         : top-k filtering (0 = disabled)

    Returns:
        new_ids : (B, T) — sequence with some masks filled in
    """
    B, T = masked_ids.shape
    device = masked_ids.device

    # Only update [MASK] positions
    is_masked = (masked_ids == mask_token_id)   # (B, T)

    # Apply temperature
    logits = pred_logits / temperature          # (B, T, V)

    # Top-k filtering (optional)
    if top_k > 0:
        top_vals, _ = logits.topk(top_k, dim=-1)
        threshold   = top_vals[..., -1].unsqueeze(-1)
        logits      = logits.masked_fill(logits < threshold, float("-inf"))

    # Sample from predicted distribution
    probs    = F.softmax(logits, dim=-1)        # (B, T, V)
    flat     = probs.view(B * T, -1)
    sampled  = torch.multinomial(flat, num_samples=1).squeeze(-1)  # (B*T,)
    sampled  = sampled.view(B, T)               # (B, T)

    # Only update masked positions
    new_ids = masked_ids.clone()
    new_ids[is_masked] = sampled[is_masked]

    return new_ids


# ---------------------------------------------------------------------------
# Schedule inspection utils
# ---------------------------------------------------------------------------

def get_schedule_stats(num_steps: int = 20) -> dict:
    """Return schedule values at evenly spaced timesteps (for logging/plotting)."""
    t      = torch.linspace(0.0, 1.0, num_steps)
    alphas = cosine_alpha(t)
    return {
        "timesteps": t.tolist(),
        "mask_rate": alphas.tolist(),
    }


def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Sample random timesteps t ~ Uniform(0, 1) for training.
    Returns (B,) tensor of floats in [0, 1].
    """
    return torch.rand(batch_size, device=device)


def sample_timesteps_logit_normal(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Sample timesteps from logit-normal distribution (biased toward mid-noise).
    This often improves training stability vs. uniform sampling.
    Inspired by MDLM ablations.

    Returns (B,) tensor of floats in [0, 1].
    """
    u = torch.randn(batch_size, device=device)  # standard normal
    t = torch.sigmoid(u)                        # logit-normal → (0, 1)
    return t


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, V = 2, 16, 32768

    # Simulate clean token ids
    tokens    = torch.randint(3, V, (B, T))
    t_samples = sample_timesteps(B, torch.device("cpu"))

    print(f"Timesteps t: {t_samples}")
    alpha = cosine_alpha(t_samples)
    print(f"Mask rates α(t): {alpha}")

    # Apply masking
    masked, mask_bool = mask_tokens(
        input_ids    = tokens,
        t            = t_samples,
        mask_token_id= 1,
        pad_token_id = 0,
    )

    print(f"\nOriginal tokens[:, :8]: {tokens[0, :8].tolist()}")
    print(f"Masked   tokens[:, :8]: {masked[0, :8].tolist()}")
    print(f"Mask bool[:, :8]:       {mask_bool[0, :8].tolist()}")
    frac = mask_bool.float().mean()
    print(f"Fraction masked: {frac:.3f} (expected ~{alpha.mean():.3f})")

    # Test unmask step
    fake_logits = torch.randn(B, T, V)
    new_ids     = unmask_tokens(masked, fake_logits, mask_token_id=1)
    still_masked = (new_ids == 1).sum()
    print(f"\nAfter unmask step: {still_masked} masks remaining (was {mask_bool.sum()})")

    # Schedule stats
    stats = get_schedule_stats(10)
    print("\nSchedule at 10 steps:")
    for t_, a_ in zip(stats["timesteps"], stats["mask_rate"]):
        bar = "█" * int(a_ * 20)
        print(f"  t={t_:.2f}  α={a_:.3f}  {bar}")

    print("\n✅ noise_schedule.py OK")
