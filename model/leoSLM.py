"""
LeoSLM — Full Model Assembly
==============================
Assembles the complete LeoSLM model:

    tokens → Embedding
           → 16 LeoBlocks (with ECTs flowing through)
           → ECT final aggregation → uncertainty map
           → LM Head  (AR logits)
           → Denoise Head (Diffusion logits)

Both heads share the embedding weight matrix (weight tying).
The uncertainty map from ECTs controls which head is trusted at inference.

Model size target: ~120M parameters
    - Embedding     :  32768 × 512      = 16.7M
    - 16 × LeoBlock :  ~6.2M each       = 99.2M
    - ECT module    :  ~0.5M
    - Heads (tied)  :  0 extra (tied to embedding)
    Total           :  ~116M ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field

from .dual_attention  import DualPathAttention
from .confidence_gate import ConfidenceGate, HardThresholdGate
from .ect             import EpistemicTokens
from .leo_block       import LeoBlock, compute_ffn_dim


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class LeoConfig:
    # Vocab & sequence
    vocab_size    : int   = 32768
    max_seq_len   : int   = 512
    pad_token_id  : int   = 0
    mask_token_id : int   = 1      # [MASK] token for diffusion
    idk_token_id  : int   = 2      # [IDK] token for uncertain outputs

    # Architecture
    hidden_dim    : int   = 512
    num_layers    : int   = 16
    num_heads     : int   = 8
    num_kv_heads  : int   = 2      # GQA: 4 queries share 1 KV head
    num_ect       : int   = 4
    ffn_dim       : Optional[int] = None    # auto-computed if None

    # Gate & uncertainty
    gate_temperature     : float = 1.0
    uncertainty_threshold: float = 0.35    # HardThresholdGate cutoff

    # Training
    dropout       : float = 0.0
    weight_tying  : bool  = True           # tie embedding ↔ lm_head weights

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = compute_ffn_dim(self.hidden_dim)


# ---------------------------------------------------------------------------
# LeoSLM
# ---------------------------------------------------------------------------

class LeoSLM(nn.Module):
    """
    Full LeoSLM model.

    Forward pass returns a dict with:
        - ar_logits     : (B, T, vocab_size)  — AR head logits
        - diff_logits   : (B, T, vocab_size)  — Diffusion head logits
        - uncertainty   : (B, T)              — ECT uncertainty scores [0,1]
        - alpha_list    : List[(B,T)]         — gate values per layer (logging)
        - ect_final     : (B, num_ect, D)     — final ECT states (for ECT loss)

    Args:
        config : LeoConfig dataclass
    """

    def __init__(self, config: LeoConfig):
        super().__init__()
        self.config = config

        # ── Token Embedding ──────────────────────────────────────────────────
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim,
                                       padding_idx=config.pad_token_id)

        # ── Stack of LeoBlocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            LeoBlock(
                hidden_dim   = config.hidden_dim,
                num_heads    = config.num_heads,
                num_kv_heads = config.num_kv_heads,
                num_ect      = config.num_ect,
                max_seq_len  = config.max_seq_len,
                ffn_dim      = config.ffn_dim,
                temperature  = config.gate_temperature,
                dropout      = config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # ── Final Layer Norm ─────────────────────────────────────────────────
        self.final_norm = nn.RMSNorm(config.hidden_dim)

        # ── ECT Module (final aggregation) ───────────────────────────────────
        self.ect_module = EpistemicTokens(
            hidden_dim = config.hidden_dim,
            num_ect    = config.num_ect,
        )

        # ── Output Heads ─────────────────────────────────────────────────────
        # AR head: standard language modeling head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Diffusion / denoising head: same shape, separate weights
        # (predicts original token from masked/noisy input)
        self.denoise_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # ── Hard Threshold Gate (inference only) ─────────────────────────────
        self.hard_gate = HardThresholdGate(threshold=config.uncertainty_threshold)

        # ── Weight Tying ─────────────────────────────────────────────────────
        if config.weight_tying:
            # lm_head and embedding share weights → saves 16.7M params
            self.lm_head.weight = self.embedding.weight

        # ── Initialize weights ───────────────────────────────────────────────
        self.apply(self._init_weights)

    # -----------------------------------------------------------------------
    def _init_weights(self, module: nn.Module):
        """
        LLaMA-style weight initialization:
        - Linear: normal(0, 0.02)
        - Embedding: normal(0, 0.02)
        - RMSNorm: ones
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    # -----------------------------------------------------------------------
    def freeze_gate_phase1(self):
        """Call this at start of training Phase 1 to fix α=0 (pure AR)."""
        for block in self.blocks:
            block.gate.freeze()
        print("🔒 Gates frozen — Phase 1 (pure AR training)")

    def unfreeze_gate_phase2(self):
        """Call this at Phase 2 start to allow bidirectional path."""
        for block in self.blocks:
            block.gate.unfreeze()
        print("🔓 Gates unfrozen — Phase 2 (joint AR + Diffusion training)")

    def freeze_backbone_phase2(self):
        """Freeze all blocks except denoise_head and ECT for Phase 2 diffusion warmup."""
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False
        for p in self.denoise_head.parameters():
            p.requires_grad = True
        for p in self.ect_module.parameters():
            p.requires_grad = True
        print("🧊 Backbone frozen — training only denoise head + ECT")

    def unfreeze_all(self):
        """Unfreeze everything for Phase 3 joint training."""
        for p in self.parameters():
            p.requires_grad = True
        print("🔥 All parameters unfrozen — Phase 3 joint training")

    # -----------------------------------------------------------------------
    def forward(
        self,
        input_ids  : torch.Tensor,                    # (B, T) — token ids
        noise_level: Optional[torch.Tensor] = None,   # (B,) — diffusion timestep t ∈ [0,1]
        labels     : Optional[torch.Tensor] = None,   # (B, T) — for teacher forcing
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through LeoSLM.

        Args:
            input_ids   : token ids, possibly with [MASK] tokens for diffusion input
            noise_level : scalar noise level t for diffusion conditioning (if None → AR mode)
            labels      : original unmasked tokens (used in diffusion loss)

        Returns dict:
            ar_logits   : (B, T, V) — logits from AR head
            diff_logits : (B, T, V) — logits from denoise head
            uncertainty : (B, T)   — ECT uncertainty per token
            alpha_list  : list of (B, T) gate values per layer
            ect_final   : (B, num_ect, D) — final ECT states
        """
        B, T = input_ids.shape
        device = input_ids.device

        # ── Token Embeddings ─────────────────────────────────────────────────
        x = self.embedding(input_ids)                             # (B, T, D)

        # ── Noise level conditioning ──────────────────────────────────────────
        # If noise_level provided, add sinusoidal noise embedding to sequence
        # This tells the model "how noisy" the input is for diffusion
        if noise_level is not None:
            noise_emb = self._noise_embedding(noise_level, x.device)  # (B, D)
            x = x + noise_emb.unsqueeze(1)                            # broadcast to (B, T, D)

        # ── Initialize ECT tokens ─────────────────────────────────────────────
        ect_hidden  = self.ect_module.get_ect_tokens(B, device)   # (B, num_ect, D)
        uncertainty = None   # not available until first ECT aggregation

        alpha_list  = []

        # ── Forward through all LeoBlocks ─────────────────────────────────────
        for i, block in enumerate(self.blocks):
            x, ect_hidden, alpha = block(
                x           = x,
                ect_hidden  = ect_hidden,
                uncertainty = uncertainty,
            )
            alpha_list.append(alpha)

            # Compute uncertainty at halfway point (block 8) and final block
            # Early computation allows later blocks to use it for gating
            if i == len(self.blocks) // 2 - 1 or i == len(self.blocks) - 1:
                uncertainty, ect_hidden = self.ect_module(
                    seq_hidden = x,
                    ect_hidden = ect_hidden,
                )

        # ── Final Norm ────────────────────────────────────────────────────────
        x = self.final_norm(x)                                    # (B, T, D)

        # ── Output Heads ─────────────────────────────────────────────────────
        ar_logits   = self.lm_head(x)                             # (B, T, V)
        diff_logits = self.denoise_head(x)                        # (B, T, V)

        return {
            "ar_logits"  : ar_logits,
            "diff_logits": diff_logits,
            "uncertainty": uncertainty,     # (B, T)
            "alpha_list" : alpha_list,      # List of (B, T)
            "ect_final"  : ect_hidden,      # (B, num_ect, D)
        }

    # -----------------------------------------------------------------------
    def _noise_embedding(
        self,
        noise_level: torch.Tensor,   # (B,) ∈ [0, 1]
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sinusoidal embedding for noise level t, similar to DDPM timestep embedding.
        Projects scalar t → vector of shape (B, hidden_dim).
        """
        D    = self.config.hidden_dim
        half = D // 2
        # Frequencies
        freqs = torch.exp(
            -torch.arange(half, device=device).float() * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )                                                         # (D//2,)
        # (B, 1) * (1, D//2) → (B, D//2)
        args  = noise_level.unsqueeze(1) * freqs.unsqueeze(0)
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)      # (B, D)
        return emb

    # -----------------------------------------------------------------------
    def get_inference_output(
        self,
        input_ids  : torch.Tensor,
        threshold  : Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference helper: returns merged logits based on per-token uncertainty.
        Tokens with uncertainty > threshold get diffusion logits,
        confident tokens get AR logits.

        Used in generate.py for hybrid mode.
        """
        if threshold is not None:
            self.hard_gate.threshold = threshold

        out = self.forward(input_ids)
        ar_logits   = out["ar_logits"]
        diff_logits = out["diff_logits"]
        uncertainty = out["uncertainty"]

        # Build binary mask: True = uncertain → use diffusion logits
        flagged, _ = self.hard_gate.get_uncertain_positions(uncertainty, input_ids)
        flagged_3d = flagged.unsqueeze(-1).expand_as(ar_logits)  # (B, T, V)

        # Merge: uncertain positions → diff, confident positions → AR
        merged_logits = torch.where(flagged_3d, diff_logits, ar_logits)

        return {
            "logits"    : merged_logits,
            "uncertainty": uncertainty,
            "flagged"   : flagged,
            **out,
        }

    # -----------------------------------------------------------------------
    def count_params(self) -> Dict[str, int]:
        """Return parameter counts by component."""
        def n(module): return sum(p.numel() for p in module.parameters())
        return {
            "embedding"     : n(self.embedding),
            "blocks"        : n(self.blocks),
            "ect_module"    : n(self.ect_module),
            "lm_head"       : 0 if self.config.weight_tying else n(self.lm_head),
            "denoise_head"  : n(self.denoise_head),
            "total"         : n(self),
        }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    config = LeoConfig()
    model  = LeoSLM(config)

    B, T   = 2, 64
    tokens = torch.randint(3, config.vocab_size, (B, T))   # avoid special tokens
    noise  = torch.rand(B)                                 # noise level ∈ [0,1]

    # AR forward pass (no noise)
    out_ar = model(tokens)
    print("── AR mode ──────────────────────────────────")
    print(f"ar_logits  : {out_ar['ar_logits'].shape}")     # (2, 64, 32768)
    print(f"diff_logits: {out_ar['diff_logits'].shape}")   # (2, 64, 32768)
    print(f"uncertainty: {out_ar['uncertainty'].shape}")   # (2, 64)
    print(f"alpha[0]   : {out_ar['alpha_list'][0].shape}") # (2, 64)
    print(f"ect_final  : {out_ar['ect_final'].shape}")     # (2, 4, 512)

    # Diffusion forward pass (with noise)
    out_d  = model(tokens, noise_level=noise)
    print("\n── Diffusion mode ───────────────────────────")
    print(f"diff_logits: {out_d['diff_logits'].shape}")

    # Parameter count
    params = model.count_params()
    print("\n── Parameter counts ─────────────────────────")
    for k, v in params.items():
        print(f"  {k:<18}: {v:>12,}")

    # Inference helper
    inf_out = model.get_inference_output(tokens)
    print(f"\nInference logits: {inf_out['logits'].shape}")
    unc = inf_out['uncertainty']
    print(f"Uncertainty range: [{unc.min():.3f}, {unc.max():.3f}]")
    print(f"Flagged for diffusion: {inf_out['flagged'].sum().item()} / {B*T} tokens")

    print("\n✅ leoSLM.py OK")
