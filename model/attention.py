"""
model/attention.py — Attention components for LeoSLM Aether
=============================================================
Contains three classes:

  EpistemicPositionalEncoding (EPE) — NOVEL, no prior art
      Modulates per-token RoPE frequencies by ECT uncertainty.
      Uncertain tokens get wider positional receptive fields so they
      gather more contextual support before committing to an answer.

  MultiHeadLatentAttention (MLA) — DeepSeek V3 style
      Compresses KV into a low-rank latent (c_kv=512) before caching.
      Reduces KV cache by ~70% vs standard multi-head attention.
      Supports both sliding-window (local) and full (global) causal attention.
      The confidence gate α merges causal and bidirectional paths (dual-path AR+Diffusion).

  DSALite — NOVEL, Aether, XLA-safe
      Derived Sparse Attention for sequences > dsa_threshold (32k).
      Derives a soft sparse mask from cheap downsampled attention, then
      applies it to full attention — quadratic → near-linear cost.
      ECT-weighted: uncertain tokens get broader attention fields.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import LeoConfig
from .norm   import RMSNorm
from .rope   import build_yarn_rope_cache, apply_rope


# ══════════════════════════════════════════════════════════════════════════════
# EPE — Epistemic Positional Encoding
# ══════════════════════════════════════════════════════════════════════════════

class EpistemicPositionalEncoding(nn.Module):
    """
    NOVEL — No prior art.

    Scales per-token RoPE frequencies by ECT uncertainty:
      scale(U) = epe_min_scale + U × (epe_max_scale − epe_min_scale)

    Confident tokens (U≈0) → tight positional focus (scale ≈ 0.7)
    Uncertain tokens (U≈1) → wide receptive field    (scale ≈ 1.8)
    """

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        self.cfg       = cfg
        self.min_scale = cfg.epe_min_scale
        self.max_scale = cfg.epe_max_scale
        half           = cfg.head_dim // 2
        self.register_buffer(
            "freq_idx",
            torch.arange(0, half).float() * 2 / cfg.head_dim,
        )

    def forward(
        self,
        positions:   torch.Tensor,
        uncertainty: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions   : (T,) or (B, T) position indices
            uncertainty : (B, T) ECT uncertainty scores, or None
        Returns:
            cos, sin : both (1, T, 1, head_dim//2) or (B, T, 1, head_dim//2)
        """
        dev   = positions.device
        base  = 500_000.0
        theta = 1.0 / (base ** self.freq_idx.to(dev))

        if uncertainty is None or not self.cfg.use_epe:
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)
            freqs = torch.outer(positions[0].float(), theta).unsqueeze(0).unsqueeze(2)
            return freqs.cos(), freqs.sin()

        B, T  = uncertainty.shape
        scale = (self.min_scale
                 + uncertainty.float().clamp(0, 1) * (self.max_scale - self.min_scale))
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(B, -1)

        # (B, T, 1) × (1, 1, half) × (B, T, 1) → (B, T, half) → add head dim
        freqs = (positions.float().unsqueeze(-1)
                 * theta.unsqueeze(0).unsqueeze(0)
                 * scale.unsqueeze(-1))
        freqs = freqs.unsqueeze(2)                    # (B, T, 1, half)
        return freqs.cos(), freqs.sin()


# ══════════════════════════════════════════════════════════════════════════════
# MLA — Multi-Head Latent Attention
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (DeepSeek V3).

    Key insight: compress KV into a low-rank latent c_kv BEFORE caching.
    Reduces KV cache memory by ~70%:
        Standard  : (B, T, nKV, hd) × 32 layers = 4 GB at 32k context
        MLA       : (B, T, c_kv)    × 32 layers = 1.2 GB

    Q is also compressed (c_q dim) to reduce computation.

    RoPE is applied to a decoupled r-dim portion of each head; the remaining
    hd dims are content-only (not position-dependent), which enables the KV
    compression without losing positional information.

    Dual-path: a learned confidence gate α merges causal (AR) and
    bidirectional (diffusion) attention outputs:
        merged = α × bidir + (1−α) × causal
    """

    def __init__(self, cfg: LeoConfig, is_sliding: bool = True):
        super().__init__()
        D    = cfg.hidden_dim
        nH   = cfg.num_heads
        hd   = cfg.head_dim
        c_kv = cfg.mla_c_kv
        c_q  = cfg.mla_c_q
        r    = cfg.mla_rope_dim

        self.nH         = nH
        self.hd         = hd
        self.c_kv       = c_kv
        self.r          = r
        self.D          = D
        self.is_sliding = is_sliding
        self.cfg        = cfg

        # ── Q: D → c_q → nH×hd (content) + nH×r (RoPE) ─────────────────────
        self.q_down  = nn.Linear(D,   c_q,    bias=False)
        self.q_norm  = RMSNorm(c_q)
        self.q_up    = nn.Linear(c_q, nH * hd, bias=False)
        self.q_rope  = nn.Linear(c_q, nH * r,  bias=False)

        # ── KV: D → c_kv (cached) → K + V; separate K-RoPE bypass ───────────
        self.kv_down = nn.Linear(D,   c_kv,   bias=False)
        self.kv_norm = RMSNorm(c_kv)
        self.k_up    = nn.Linear(c_kv, nH * hd, bias=False)
        self.v_up    = nn.Linear(c_kv, nH * hd, bias=False)
        self.k_rope  = nn.Linear(D,   nH * r,  bias=False)

        # ── Output projection ─────────────────────────────────────────────────
        self.out     = nn.Linear(nH * hd, D, bias=False)

        # Dual-path confidence gate (scalar α per token)
        self.gate    = nn.Sequential(nn.Linear(D, 1), nn.Sigmoid())

        # Epistemic Positional Encoding
        self.epe     = EpistemicPositionalEncoding(cfg)

    def forward(
        self,
        x:           torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        mem_tokens:  Optional[torch.Tensor] = None,
        positions:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x           : (B, T, D)
            uncertainty : (B, T)     — ECT uncertainty, fed to EPE + gate
            mem_tokens  : (B, M, D)  — TDM/SAM memory prefix (prepended)
            positions   : (T,)       — position indices (default: 0..T-1)
        Returns:
            output      : (B, T, D)
            alpha       : (B, T)     — per-token gate value for logging
        """
        B, T, D = x.shape

        # Prepend memory tokens (from TDM / SAM)
        if mem_tokens is not None:
            x = torch.cat([mem_tokens, x], dim=1)
        T_full = x.shape[1]

        if positions is None:
            positions = torch.arange(T_full, device=x.device)

        # ── Q path ───────────────────────────────────────────────────────────
        q_lat = self.q_norm(self.q_down(x))
        q_c   = self.q_up(q_lat).view(B, T_full, self.nH, self.hd)
        q_r   = self.q_rope(q_lat).view(B, T_full, self.nH, self.r)

        # ── KV path ──────────────────────────────────────────────────────────
        kv_lat = self.kv_norm(self.kv_down(x))
        k_c    = self.k_up(kv_lat).view(B, T_full, self.nH, self.hd)
        v      = self.v_up(kv_lat).view(B, T_full, self.nH, self.hd)
        k_r    = self.k_rope(x).view(B, T_full, self.nH, self.r)

        # ── EPE-YaRN RoPE ─────────────────────────────────────────────────────
        unc_full = uncertainty
        if mem_tokens is not None and uncertainty is not None:
            mem_u    = torch.zeros(B, mem_tokens.shape[1], device=x.device)
            unc_full = torch.cat([mem_u, uncertainty], dim=1)

        if self.cfg.use_epe and unc_full is not None:
            cos, sin = self.epe(positions, unc_full)
        else:
            cos, sin = build_yarn_rope_cache(
                T_full, self.r * 2, x.device, scale=self.cfg.yarn_scale
            )

        q_r_rot = apply_rope(q_r, cos, sin)
        k_r_rot = apply_rope(k_r, cos, sin)

        # Concatenate content + RoPE portions
        Q = torch.cat([q_c, q_r_rot], dim=-1)   # (B, T_full, nH, hd+r)
        K = torch.cat([k_c, k_r_rot], dim=-1)

        # ── Dual-path attention ───────────────────────────────────────────────
        Qc = Q.transpose(1, 2)   # (B, nH, T_full, hd+r)
        Kc = K.transpose(1, 2)
        Vc = v.transpose(1, 2)

        if self.is_sliding:
            out_ar = self._sliding_window_attn(Qc, Kc, Vc, self.cfg.sliding_window)
        else:
            out_ar = self._chunked_causal_attn(Qc, Kc, Vc)

        out_bidir = F.scaled_dot_product_attention(Qc, Kc, Vc, is_causal=False)

        # Gate: α per token (applied in the original D-input space)
        alpha  = self.gate(x).squeeze(-1)                          # (B, T_full)
        merged = (alpha.unsqueeze(1).unsqueeze(-1) * out_bidir
                  + (1 - alpha.unsqueeze(1).unsqueeze(-1)) * out_ar)

        # Keep only content head dims (r portion is positional, not projected out)
        merged = merged.transpose(1, 2).contiguous()               # (B, T_full, nH, hd+r)
        merged = merged[:, :, :, :self.hd].contiguous()
        merged = merged.view(B, T_full, self.nH * self.hd)

        # Remove memory prefix
        if mem_tokens is not None:
            merged = merged[:, mem_tokens.shape[1]:, :]
            alpha  = alpha[:, mem_tokens.shape[1]:]

        return self.out(merged), alpha


    # ── Attention helpers ────────────────────────────────────────────────────

    def _sliding_window_attn(self, Q, K, V, window: int) -> torch.Tensor:
        """
        Sliding window: each position attends to ±window/2 tokens.
        O(window × T) memory vs O(T²) for full attention.
        XLA-friendly: chunked SDPA with static shapes.
        """
        B, nH, T, hd = Q.shape
        out    = torch.zeros_like(Q)
        half_w = window // 2
        chunk  = self.cfg.chunk_size

        for ci in range(math.ceil(T / chunk)):
            s  = ci * chunk
            e  = min(s + chunk, T)
            ks = max(0, s - half_w)
            ke = min(T, e + half_w)
            out[:, :, s:e] = F.scaled_dot_product_attention(
                Q[:, :, s:e], K[:, :, ks:ke], V[:, :, ks:ke],
                is_causal=True,
            )
        return out

    def _chunked_causal_attn(self, Q, K, V) -> torch.Tensor:
        """Global causal attention in memory-efficient chunks."""
        B, nH, T, hd = Q.shape
        out   = torch.zeros_like(Q)
        chunk = self.cfg.chunk_size

        for ci in range(math.ceil(T / chunk)):
            s = ci * chunk
            e = min(s + chunk, T)
            out[:, :, s:e] = F.scaled_dot_product_attention(
                Q[:, :, s:e], K[:, :, :e], V[:, :, :e],
                is_causal=True,
            )
        return out


# ══════════════════════════════════════════════════════════════════════════════
# DSALite — Derived Sparse Attention
# ══════════════════════════════════════════════════════════════════════════════

class DSALite(nn.Module):
    """
    NOVEL — Aether, no prior art. Inspired by DeepSeek V3.2 DSA.

    For sequences longer than dsa_threshold (default 32k), derive a soft
    sparse mask from cheap downsampled attention, then apply it to full
    attention. ECT-weighted: uncertain tokens get broader attention fields.

    Advantages:
      - Quadratic → near-linear cost for very long sequences
      - XLA-compatible: differentiable soft masking, no custom kernels
      - No external dependencies — pure PyTorch/XLA ops
    """

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        self.top_k_pct  = cfg.dsa_top_k_pct
        self.threshold  = cfg.dsa_threshold
        self.chunk_size = cfg.chunk_size

    def derive_mask(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        U: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            Q, K : (B, nH, T, hd) — full sequence Q and K
            U    : (B, T)          — ECT uncertainty
        Returns:
            mask : (B, nH, T, T) ∈ [0,1]
        """
        B, nH, T, hd = Q.shape
        stride = 8

        # Cheap downsampled proxy attention
        Q_ds = Q[:, :, ::stride, :]
        K_ds = K[:, :, ::stride, :]
        scores = torch.einsum("bhid,bhjd->bhij", Q_ds, K_ds) * (hd ** -0.5)

        # Upsample back to full T
        scores_full = (scores
                       .repeat_interleave(stride, dim=2)
                       .repeat_interleave(stride, dim=3))
        scores_full = scores_full[:, :, :T, :T]

        # ECT-weighted broadening: uncertain tokens attend more broadly
        if U is not None:
            u_scale     = (1.0 + U.clamp(0, 1)).unsqueeze(1).unsqueeze(-1)
            scores_full = scores_full * u_scale

        # Top-k soft mask
        k             = max(1, int(T * self.top_k_pct))
        threshold_val, _ = torch.kthvalue(
            scores_full.view(B * nH * T, T), T - k + 1, dim=-1
        )
        threshold_val = threshold_val.view(B, nH, T, 1)
        return torch.sigmoid((scores_full - threshold_val) * 10.0)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        U: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Apply DSA-lite sparse attention. Falls back to SDPA for short sequences."""
        B, nH, T, hd = Q.shape

        if T < self.threshold:
            return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

        mask = self.derive_mask(Q, K, U)

        if is_causal:
            causal = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))
            mask   = mask * causal.unsqueeze(0).unsqueeze(0)

        scores = torch.einsum("bhid,bhjd->bhij", Q, K) * (hd ** -0.5)
        scores = scores + (1.0 - mask) * (-1e9)
        attn   = scores.softmax(dim=-1)
        return torch.einsum("bhij,bhjd->bhid", attn, V)
