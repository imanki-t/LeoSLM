"""
LeoSLM — Constitutional AI Training
======================================
Implements Constitutional AI (CAI) principles as conditioning vectors
injected during the MDM (diffusion) training phase.

How it works:
    1. A set of 12 constitutional principles are embedded as text
    2. During training, for each batch we sample 1-2 principles
    3. The principle embedding is added to the noise-level embedding
       → the denoiser is conditioned on the principle
    4. If the model's prediction violates the principle,
       we add a KL-divergence penalty pushing it toward [IDK]

This is inspired by Anthropic's Constitutional AI (Bai et al., 2022)
but adapted for masked diffusion training rather than RLHF.

The 12 principles for LeoSLM:
    1. Be factually accurate — prefer uncertainty over confabulation
    2. Acknowledge when you don't know rather than guessing
    3. Don't generate harmful or dangerous content
    4. Be consistent — don't contradict yourself
    5. Be concise — don't pad with filler
    6. Attribute claims appropriately (hedge with "I think", "likely", etc.)
    7. Don't generate personally identifiable information
    8. Avoid stereotypes and biases
    9. Prefer simpler explanations over complex ones (Occam's razor)
    10. Signal uncertainty with language ("I believe", "probably", etc.)
    11. Don't pretend to have capabilities you lack
    12. Correct yourself if you detect an error in generation

During training, principles 1, 2, 6, 10 are most heavily weighted
(the anti-hallucination principles).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import random


# ---------------------------------------------------------------------------
# Constitutional Principles
# ---------------------------------------------------------------------------

PRINCIPLES = [
    # (principle_text, weight) — higher weight = sampled more often
    ("Be factually accurate. Prefer saying 'I don't know' over guessing.",          3.0),
    ("Acknowledge uncertainty. Hedge claims appropriately.",                         3.0),
    ("Do not generate harmful, dangerous, or offensive content.",                    2.0),
    ("Be internally consistent. Do not contradict earlier statements.",              1.5),
    ("Be concise. Remove filler and repetition.",                                   1.0),
    ("Attribute uncertain claims with hedges: 'I think', 'likely', 'probably'.",    2.5),
    ("Do not generate personally identifiable information about real people.",       1.5),
    ("Avoid stereotypes, biases, and unfair generalizations.",                      1.5),
    ("Prefer the simplest explanation that fits the evidence.",                     1.0),
    ("Signal low confidence explicitly in your language.",                          2.5),
    ("Do not pretend to have abilities or knowledge you lack.",                     2.0),
    ("If you notice an error in your output, correct it and flag it.",              1.5),
]

PRINCIPLE_TEXTS   = [p[0] for p in PRINCIPLES]
PRINCIPLE_WEIGHTS_RAW = [p[1] for p in PRINCIPLES]

# Normalize weights
total = sum(PRINCIPLE_WEIGHTS_RAW)
PRINCIPLE_WEIGHTS = [w / total for w in PRINCIPLE_WEIGHTS_RAW]


class ConstitutionalConditioner(nn.Module):
    """
    Embeds constitutional principles as conditioning vectors.

    During diffusion training:
        - A principle is randomly sampled each batch
        - Its embedding is added to the noise-level embedding
        - The model learns to denoise in a principle-consistent way

    Args:
        hidden_dim    : model hidden dimension
        num_principles: number of principles (default = len(PRINCIPLES))
        embed_dim     : dimension of principle embedding (default = hidden_dim)
    """

    def __init__(
        self,
        hidden_dim    : int,
        num_principles: int = len(PRINCIPLES),
        embed_dim     : Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim     = hidden_dim
        self.num_principles = num_principles
        embed_dim = embed_dim or hidden_dim

        # Learned principle embeddings
        self.principle_embeddings = nn.Embedding(num_principles, embed_dim)

        # Project to hidden_dim if different
        if embed_dim != hidden_dim:
            self.proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        else:
            self.proj = nn.Identity()

        # Principle weights (for sampling — not trained)
        self.register_buffer(
            "sampling_weights",
            torch.tensor(PRINCIPLE_WEIGHTS, dtype=torch.float32)
        )

        nn.init.normal_(self.principle_embeddings.weight, mean=0.0, std=0.02)

    def sample_principle(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample principle indices for a batch.
        Uses PRINCIPLE_WEIGHTS for weighted sampling.
        Returns (B,) long tensor of principle indices.
        """
        indices = torch.multinomial(
            self.sampling_weights.unsqueeze(0).expand(batch_size, -1),
            num_samples=1,
            replacement=True,
        ).squeeze(-1)   # (B,)
        return indices.to(device)

    def get_conditioning(
        self,
        principle_ids: torch.Tensor,   # (B,) principle indices
    ) -> torch.Tensor:
        """
        Get principle conditioning vector for a batch.
        Returns (B, hidden_dim) conditioning vector.
        """
        emb = self.principle_embeddings(principle_ids)   # (B, embed_dim)
        return self.proj(emb)                            # (B, hidden_dim)

    def forward(
        self,
        noise_embedding : torch.Tensor,  # (B, hidden_dim) — noise level embedding from model
        batch_size      : int,
        device          : torch.device,
        principle_ids   : Optional[torch.Tensor] = None,  # if None, sample randomly
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add constitutional conditioning to noise embedding.

        Args:
            noise_embedding : existing noise-level conditioning (B, D)
            batch_size      : batch size B
            device          : torch device
            principle_ids   : if provided, use these specific principles (for eval)

        Returns:
            conditioned_emb : (B, D) — noise + principle conditioning
            principle_ids   : (B,)  — which principles were sampled (for logging)
        """
        if principle_ids is None:
            principle_ids = self.sample_principle(batch_size, device)

        conditioning = self.get_conditioning(principle_ids)  # (B, D)

        # Add to noise embedding (simple additive conditioning)
        conditioned_emb = noise_embedding + conditioning

        return conditioned_emb, principle_ids


class ConstitutionalLoss(nn.Module):
    """
    KL divergence penalty for principle violations.

    For anti-hallucination principles (principles 0, 1, 5, 9),
    when the model predicts a very confident token at a high-uncertainty
    position, we add a KL penalty pushing the distribution toward [IDK].

    This directly trains the model to say "I don't know" when it should.

    Args:
        idk_token_id        : id of [IDK] token
        high_conf_threshold : logit confidence above which we penalize
        lambda_const        : weight for constitutional loss
    """

    def __init__(
        self,
        idk_token_id        : int,
        high_conf_threshold : float = 0.85,
        lambda_const        : float = 0.05,
    ):
        super().__init__()
        self.idk_token_id        = idk_token_id
        self.high_conf_threshold = high_conf_threshold
        self.lambda_const        = lambda_const

        # Anti-hallucination principle indices (see PRINCIPLES list)
        self.antihall_principles = {0, 1, 5, 9}

    def forward(
        self,
        diff_logits  : torch.Tensor,     # (B, T, V) — diffusion head logits
        uncertainty  : torch.Tensor,     # (B, T) — ECT scores
        principle_ids: torch.Tensor,     # (B,) — which principles were active
        vocab_size   : int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute constitutional constraint loss.

        Only active for anti-hallucination principles AND
        at positions where ECT signals high uncertainty.
        """
        B, T, V = diff_logits.shape
        device  = diff_logits.device

        # Which samples have anti-hallucination principle active?
        is_antihall = torch.tensor(
            [p.item() in self.antihall_principles for p in principle_ids],
            device=device,
        )   # (B,) bool

        if not is_antihall.any():
            dummy = diff_logits.sum() * 0.0
            return dummy, {"const_loss": 0.0}

        # Which positions are high-uncertainty (should prefer IDK)?
        high_unc = uncertainty > self.high_conf_threshold   # (B, T) bool

        # What is the model's confidence at these positions?
        probs       = F.softmax(diff_logits, dim=-1)        # (B, T, V)
        max_conf    = probs.max(dim=-1).values              # (B, T) max probability

        # Problematic: antihall principle active AND high uncertainty AND high confidence
        problematic = (
            is_antihall.unsqueeze(1) &   # (B, T)
            high_unc                 &   # (B, T)
            (max_conf > self.high_conf_threshold)   # (B, T)
        )

        if not problematic.any():
            dummy = diff_logits.sum() * 0.0
            return dummy, {"const_loss": 0.0, "n_violations": 0}

        # Target distribution: peaked on [IDK] token
        target_probs = torch.zeros(V, device=device)
        target_probs[self.idk_token_id] = 1.0
        target_probs = target_probs.unsqueeze(0)   # (1, V)

        # KL divergence at problematic positions
        logits_prob  = probs[problematic]          # (N_prob, V)
        kl           = F.kl_div(
            logits_prob.log().clamp(min=-100),
            target_probs.expand(logits_prob.shape[0], -1),
            reduction="batchmean",
        )

        loss = self.lambda_const * kl

        metrics = {
            "const_loss"  : loss.item(),
            "n_violations": problematic.sum().item(),
        }
        return loss, metrics


def get_principle_text(idx: int) -> str:
    """Get human-readable principle text by index."""
    return PRINCIPLE_TEXTS[idx]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B, D, V = 2, 512, 32768
    device = torch.device("cpu")

    conditioner = ConstitutionalConditioner(hidden_dim=D)

    noise_emb = torch.randn(B, D)
    cond_emb, p_ids = conditioner(noise_emb, B, device)

    print(f"Noise emb shape : {noise_emb.shape}")
    print(f"Conditioned emb : {cond_emb.shape}")
    print(f"Principles used : {p_ids.tolist()}")
    for i, pid in enumerate(p_ids.tolist()):
        print(f"  Sample {i}: [{pid}] {get_principle_text(pid)}")

    # Test constitutional loss
    const_loss_fn = ConstitutionalLoss(idk_token_id=2, lambda_const=0.05)
    diff_logits   = torch.randn(B, 32, V)
    uncertainty   = torch.rand(B, 32)
    loss, mets    = const_loss_fn(diff_logits, uncertainty, p_ids, V)
    print(f"\nConstitutional loss: {loss.item():.6f}")
    print(f"Violations: {mets['n_violations']}")
    print(f"\nAll principles:")
    for i, (text, weight) in enumerate(PRINCIPLES):
        print(f"  [{i:02d}] w={weight:.1f} {text}")
    print("\n✅ constitutional.py OK")
