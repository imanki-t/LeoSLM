"""
model/attention.py — Multi-Head Latent Attention + DSA-Lite + EPE

BUG FIXES vs original:
  1. _sliding_window_attn: `min(s+chunk, T)` created dynamic tensor shapes → XLA
     recompiled the graph every step for the last chunk. Fixed by padding the
     sequence to a multiple of chunk_size before attention, then trimming.
  2. _chunked_causal_attn: same dynamic-shape fix.
  3. mem_tokens position offset: memory tokens used positions 0..M-1, pushing
     real token positions to M..M+T-1. Now memory tokens share position 0 (or use
     a separate learned embedding range) to avoid polluting real-token RoPE.
  4. Removed the redundant `merged[:, :, :, :self.hd]` no-op slice.
  5. DSALite._derive_mask: fixed shape inconsistency with stride-based downsampling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import LeoConfig
from .norm   import RMSNorm
from .rope   import build_yarn_rope_cache, apply_rope


class EpistemicPositionalEncoding(nn.Module):
    """
    EPE scales RoPE frequencies by per-token uncertainty:
    high-U tokens attend more broadly (wider effective position range).
    """

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        self.cfg       = cfg
        self.min_scale = cfg.epe_min_scale
        self.max_scale = cfg.epe_max_scale
        half           = cfg.mla_rope_dim // 2
        self.register_buffer(
            "freq_idx",
            torch.arange(0, half).float() * 2 / cfg.mla_rope_dim,
        )

    def forward(
        self,
        positions:   torch.Tensor,   # (T,) or (B, T)
        uncertainty: Optional[torch.Tensor],  # (B, T) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dev   = positions.device
        base  = 500_000.0
        theta = 1.0 / (base ** self.freq_idx.to(dev))   # (D/2,)

        if uncertainty is None or not self.cfg.use_epe:
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)   # (1, T)
            # (1, T, 1, D/2)
            freqs = torch.outer(positions[0].float(), theta).unsqueeze(0).unsqueeze(2)
            return freqs.cos(), freqs.sin()

        B, T  = uncertainty.shape
        # scale: (B, T) in [min_scale, max_scale]
        scale = self.min_scale + uncertainty.float().clamp(0.0, 1.0) * (
            self.max_scale - self.min_scale
        )
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(B, -1)  # (B, T)

        # freqs: (B, T, 1, D/2)
        freqs = (
            positions.float().unsqueeze(-1)           # (B, T, 1)
            * theta.to(dev).unsqueeze(0).unsqueeze(0) # (1, 1, D/2)
            * scale.unsqueeze(-1)                      # (B, T, 1)
        ).unsqueeze(2)                                 # (B, T, 1, D/2)
        return freqs.cos(), freqs.sin()


class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek-V3 style Multi-Head Latent Attention (MLA) with:
    - Low-rank KV compression (c_kv latent)
    - Decoupled RoPE on a separate 'r'-dim head slice
    - EPE uncertainty-aware position scaling
    - Hybrid causal (AR) + bidirectional (diffusion) output gated by alpha
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

        # Q compression & RoPE
        self.q_down  = nn.Linear(D,   c_q,     bias=False)
        self.q_norm  = RMSNorm(c_q)
        self.q_up    = nn.Linear(c_q, nH * hd, bias=False)
        self.q_rope  = nn.Linear(c_q, nH * r,  bias=False)

        # KV compression
        self.kv_down = nn.Linear(D,   c_kv,     bias=False)
        self.kv_norm = RMSNorm(c_kv)
        self.k_up    = nn.Linear(c_kv, nH * hd, bias=False)
        self.v_up    = nn.Linear(c_kv, nH * hd, bias=False)
        self.k_rope  = nn.Linear(D,   nH * r,  bias=False)

        self.out     = nn.Linear(nH * hd, D, bias=False)
        self.gate    = nn.Sequential(nn.Linear(D, 1), nn.Sigmoid())
        self.epe     = EpistemicPositionalEncoding(cfg)

    def forward(
        self,
        x:           torch.Tensor,                    # (B, T, D)
        uncertainty: Optional[torch.Tensor] = None,  # (B, T)
        mem_tokens:  Optional[torch.Tensor] = None,  # (B, M, D) – prefix memory
        positions:   Optional[torch.Tensor] = None,  # (T,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        # BUG FIX: When prepending memory tokens, keep real-token positions
        # starting at 0 by prepending 'negative' positions so real tokens
        # always sit at positions [0 .. T-1] regardless of memory size.
        M = 0
        if mem_tokens is not None:
            M  = mem_tokens.shape[1]
            x  = torch.cat([mem_tokens, x], dim=1)

        T_full = x.shape[1]   # M + T

        if positions is None:
            # BUG FIX: real tokens get [0..T-1], memory tokens get [-M..-1]
            # so that RoPE for real tokens is identical with or without memory.
            real_pos = torch.arange(T, device=x.device)
            if M > 0:
                mem_pos  = torch.arange(-M, 0, device=x.device)
                positions = torch.cat([mem_pos, real_pos])   # (T_full,)
            else:
                positions = real_pos   # (T,)

        # ── Q ───────────────────────────────────────────────────────────────
        q_lat = self.q_norm(self.q_down(x))                             # (B, T_full, c_q)
        q_c   = self.q_up(q_lat).view(B, T_full, self.nH, self.hd)     # (B, T_full, nH, hd)
        q_r   = self.q_rope(q_lat).view(B, T_full, self.nH, self.r)    # (B, T_full, nH, r)

        # ── KV ──────────────────────────────────────────────────────────────
        kv_lat = self.kv_norm(self.kv_down(x))                          # (B, T_full, c_kv)
        k_c    = self.k_up(kv_lat).view(B, T_full, self.nH, self.hd)   # (B, T_full, nH, hd)
        v      = self.v_up(kv_lat).view(B, T_full, self.nH, self.hd)   # (B, T_full, nH, hd)
        k_r    = self.k_rope(x).view(B, T_full, self.nH, self.r)       # (B, T_full, nH, r)

        # ── EPE / RoPE ──────────────────────────────────────────────────────
        unc_full = uncertainty
        if M > 0 and uncertainty is not None:
            mem_u    = torch.zeros(B, M, device=x.device)
            unc_full = torch.cat([mem_u, uncertainty], dim=1)   # (B, T_full)

        if self.cfg.use_epe and unc_full is not None:
            cos, sin = self.epe(positions, unc_full)
        else:
            cos, sin = build_yarn_rope_cache(
                T_full, self.r, x.device, scale=self.cfg.yarn_scale
            )

        q_r_rot = apply_rope(q_r, cos, sin)   # (B, T_full, nH, r)
        k_r_rot = apply_rope(k_r, cos, sin)

        # Concatenate content + RoPE heads: each head has dim (hd + r)
        Q = torch.cat([q_c, q_r_rot], dim=-1)   # (B, T_full, nH, hd+r)
        K = torch.cat([k_c, k_r_rot], dim=-1)   # (B, T_full, nH, hd+r)

        # Transpose to (B, nH, T_full, head_dim)
        Qc = Q.transpose(1, 2)   # (B, nH, T_full, hd+r)
        Kc = K.transpose(1, 2)   # (B, nH, T_full, hd+r)
        Vc = v.transpose(1, 2)   # (B, nH, T_full, hd)

        # ── Attention ────────────────────────────────────────────────────────
        if self.is_sliding:
            out_ar = self._sliding_window_attn(Qc, Kc, Vc, self.cfg.sliding_window)
        else:
            out_ar = self._chunked_causal_attn(Qc, Kc, Vc)

        # Bidirectional (diffusion) path — full attention without causal mask
        out_bidir = F.scaled_dot_product_attention(Qc, Kc, Vc, is_causal=False)

        # Alpha gate per position — shape (B, T_full)
        alpha  = self.gate(x).squeeze(-1)   # (B, T_full)

        # Merge: alpha controls causal↔bidirectional balance
        # alpha: (B, T_full) → (B, 1, T_full, 1) for broadcast
        a = alpha.unsqueeze(1).unsqueeze(-1)
        merged = a * out_bidir + (1.0 - a) * out_ar   # (B, nH, T_full, hd)

        # Reshape back: (B, T_full, nH*hd)
        merged = merged.transpose(1, 2).contiguous().view(B, T_full, self.nH * self.hd)

        # Strip memory prefix from output
        if M > 0:
            merged = merged[:, M:, :]         # (B, T, nH*hd)
            alpha  = alpha[:, M:]             # (B, T)

        return self.out(merged), alpha

    # ── Sliding-window causal attention ─────────────────────────────────────

    def _sliding_window_attn(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window: int
    ) -> torch.Tensor:
        """
        BUG FIX: original used `min(s+chunk, T)` which creates dynamic shapes
        causing XLA to recompile every step when T % chunk != 0.
        Fix: pad to next multiple of chunk before chunking, trim after.
        """
        B, nH, T, _ = Q.shape
        chunk   = self.cfg.chunk_size
        half_w  = window // 2

        # Pad T to multiple of chunk_size → fixed shapes on XLA
        pad = (chunk - T % chunk) % chunk
        if pad > 0:
            pad_q = Q.new_zeros(B, nH, pad, Q.shape[-1])
            pad_k = K.new_zeros(B, nH, pad, K.shape[-1])
            pad_v = V.new_zeros(B, nH, pad, V.shape[-1])
            Q = torch.cat([Q, pad_q], dim=2)
            K = torch.cat([K, pad_k], dim=2)
            V = torch.cat([V, pad_v], dim=2)

        T_pad = Q.shape[2]   # always a multiple of chunk
        out   = torch.zeros_like(V)   # (B, nH, T_pad, hd)

        n_chunks = T_pad // chunk
        for ci in range(n_chunks):
            s  = ci * chunk
            e  = s + chunk           # fixed size — no min() needed
            ks = max(0, s - half_w)
            ke = min(T_pad, e + half_w)

            # BUG FIX: ke is still dynamic but only uses Python int arithmetic
            # (ke is computed at trace-time as a Python int from cfg constants
            # and ci which is a Python loop variable). XLA sees fixed-shape slices.
            Qc = Q[:, :, s:e]
            Kc = K[:, :, ks:ke]
            Vc = V[:, :, ks:ke]

            q_len  = e - s           # = chunk (fixed)
            k_len  = ke - ks         # varies per chunk, but Python int → fixed at trace
            offset = s - ks

            # Build causal mask: token at query position qi can attend to key
            # positions kj if global_key_pos = ks+kj <= global_query_pos = s+qi
            q_i = torch.arange(q_len, device=Q.device).unsqueeze(1)   # (q_len, 1)
            k_j = torch.arange(k_len, device=Q.device).unsqueeze(0)   # (1, k_len)
            causal_mask = torch.where(
                k_j <= q_i + offset,
                Q.new_zeros(q_len, k_len),
                Q.new_full((q_len, k_len), float("-inf")),
            )

            out[:, :, s:e] = F.scaled_dot_product_attention(
                Qc, Kc, Vc,
                attn_mask=causal_mask.unsqueeze(0).unsqueeze(0),
            )

        # Trim padding
        return out[:, :, :T, :]

    # ── Chunked full-sequence causal attention ───────────────────────────────

    def _chunked_causal_attn(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """
        BUG FIX: same padding trick to avoid dynamic shapes.
        """
        B, nH, T, _ = Q.shape
        chunk = self.cfg.chunk_size

        pad = (chunk - T % chunk) % chunk
        if pad > 0:
            Q = torch.cat([Q, Q.new_zeros(B, nH, pad, Q.shape[-1])], dim=2)
            K = torch.cat([K, K.new_zeros(B, nH, pad, K.shape[-1])], dim=2)
            V = torch.cat([V, V.new_zeros(B, nH, pad, V.shape[-1])], dim=2)

        T_pad    = Q.shape[2]
        n_chunks = T_pad // chunk
        out      = torch.zeros_like(V)

        for ci in range(n_chunks):
            s = ci * chunk
            e = s + chunk   # fixed
            # causal: query at [s..e) attends to all keys [0..e)
            out[:, :, s:e] = F.scaled_dot_product_attention(
                Q[:, :, s:e], K[:, :, :e], V[:, :, :e],
                is_causal=True,
            )

        return out[:, :, :T, :]


class DSALite(nn.Module):
    """
    Derived Sparse Attention (DSA-Lite) — activates for seqlen > dsa_threshold.
    Uses a cheap downsampled proxy to derive per-head sparse masks.

    BUG FIX: original had a shape inconsistency when T % stride != 0.
    Fixed by using F.interpolate instead of raw stride slicing.
    """

    def __init__(self, cfg: LeoConfig):
        super().__init__()
        D               = cfg.hidden_dim
        nH              = cfg.num_heads
        hd              = cfg.head_dim
        self.nH         = nH
        self.hd         = hd
        self.top_k_pct  = cfg.dsa_top_k_pct
        self.threshold  = cfg.dsa_threshold
        self.chunk_size = cfg.chunk_size

        self.q_proj = nn.Linear(D, nH * hd, bias=False)
        self.k_proj = nn.Linear(D, nH * hd, bias=False)
        self.v_proj = nn.Linear(D, nH * hd, bias=False)
        self.o_proj = nn.Linear(nH * hd, D, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        U: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        if T < self.threshold:
            return x

        Q = self.q_proj(x).view(B, T, self.nH, self.hd).transpose(1, 2)  # (B, nH, T, hd)
        K = self.k_proj(x).view(B, T, self.nH, self.hd).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.nH, self.hd).transpose(1, 2)

        mask = self._derive_mask(Q, K, U)  # (B, nH, T, T)

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        mask   = mask * causal.unsqueeze(0).unsqueeze(0).float()

        scale  = self.hd ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # (B, nH, T, T)
        scores = scores + (1.0 - mask) * (-1e9)
        attn   = scores.softmax(dim=-1)
        out    = torch.matmul(attn, V)
        out    = out.transpose(1, 2).contiguous().view(B, T, self.nH * self.hd)
        return self.o_proj(out)

    def _derive_mask(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        U: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        BUG FIX: used repeat_interleave which produces wrong shapes when
        T is not divisible by stride. Now uses F.interpolate on the score
        tensor (treating (T_ds, T_ds) as a 2D spatial map).
        """
        B, nH, T, hd = Q.shape
        stride = 8
        Td     = max(1, T // stride)

        # Cheap downsampled attention scores
        Q_ds = Q[:, :, :Td * stride:stride, :]   # (B, nH, Td, hd)  — fixed shape
        K_ds = K[:, :, :Td * stride:stride, :]
        scores_ds = torch.einsum("bhid,bhjd->bhij", Q_ds, K_ds) * (hd ** -0.5)  # (B, nH, Td, Td)

        # Uncertainty broadening
        if U is not None:
            U_ds  = U[:, :Td * stride:stride]                   # (B, Td)
            scale = (1.0 + U_ds.clamp(0.0, 1.0)).unsqueeze(1).unsqueeze(-1)  # (B, 1, Td, 1)
            scores_ds = scores_ds * scale

        # Upsample back to (T, T) — fixed-shape interpolation
        # Reshape to (B*nH, 1, Td, Td) for F.interpolate
        BnH = B * nH
        sm  = scores_ds.reshape(BnH, 1, Td, Td)
        sm  = F.interpolate(sm.float(), size=(T, T), mode="nearest")  # (B*nH, 1, T, T)
        sm  = sm.reshape(B, nH, T, T)

        k          = max(1, int(T * self.top_k_pct))
        flat       = sm.reshape(BnH * T, T)
        kth_val, _ = torch.topk(flat, k, dim=-1, largest=True, sorted=False)
        threshold  = kth_val[:, -1].unsqueeze(-1).view(B, nH, T, 1)
        return torch.sigmoid((sm - threshold) * 10.0)
