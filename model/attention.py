import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import LeoConfig
from .norm   import RMSNorm
from .rope   import build_yarn_rope_cache, apply_rope


class EpistemicPositionalEncoding(nn.Module):

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
        positions:   torch.Tensor,
        uncertainty: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dev   = positions.device
        base  = 500_000.0
        theta = 1.0 / (base ** self.freq_idx.to(dev))

        if uncertainty is None or not self.cfg.use_epe:
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)
            freqs = torch.outer(positions[0].float(), theta).unsqueeze(0).unsqueeze(2)
            return freqs.cos(), freqs.sin()

        B, T  = uncertainty.shape
        scale = self.min_scale + uncertainty.float().clamp(0, 1) * (self.max_scale - self.min_scale)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(B, -1)

        freqs = (
            positions.float().unsqueeze(-1)
            * theta.unsqueeze(0).unsqueeze(0)
            * scale.unsqueeze(-1)
        ).unsqueeze(2)
        return freqs.cos(), freqs.sin()


class MultiHeadLatentAttention(nn.Module):

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

        self.q_down  = nn.Linear(D,   c_q,     bias=False)
        self.q_norm  = RMSNorm(c_q)
        self.q_up    = nn.Linear(c_q, nH * hd, bias=False)
        self.q_rope  = nn.Linear(c_q, nH * r,  bias=False)

        self.kv_down = nn.Linear(D,   c_kv,    bias=False)
        self.kv_norm = RMSNorm(c_kv)
        self.k_up    = nn.Linear(c_kv, nH * hd, bias=False)
        self.v_up    = nn.Linear(c_kv, nH * hd, bias=False)
        self.k_rope  = nn.Linear(D,   nH * r,  bias=False)

        self.out     = nn.Linear(nH * hd, D, bias=False)
        self.gate    = nn.Sequential(nn.Linear(D, 1), nn.Sigmoid())
        self.epe     = EpistemicPositionalEncoding(cfg)

    def forward(
        self,
        x:           torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        mem_tokens:  Optional[torch.Tensor] = None,
        positions:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        if mem_tokens is not None:
            x = torch.cat([mem_tokens, x], dim=1)
        T_full = x.shape[1]

        if positions is None:
            positions = torch.arange(T_full, device=x.device)

        q_lat = self.q_norm(self.q_down(x))
        q_c   = self.q_up(q_lat).view(B, T_full, self.nH, self.hd)
        q_r   = self.q_rope(q_lat).view(B, T_full, self.nH, self.r)

        kv_lat = self.kv_norm(self.kv_down(x))
        k_c    = self.k_up(kv_lat).view(B, T_full, self.nH, self.hd)
        v      = self.v_up(kv_lat).view(B, T_full, self.nH, self.hd)
        k_r    = self.k_rope(x).view(B, T_full, self.nH, self.r)

        unc_full = uncertainty
        if mem_tokens is not None and uncertainty is not None:
            mem_u    = torch.zeros(B, mem_tokens.shape[1], device=x.device)
            unc_full = torch.cat([mem_u, uncertainty], dim=1)

        if self.cfg.use_epe and unc_full is not None:
            cos, sin = self.epe(positions, unc_full)
        else:
            cos, sin = build_yarn_rope_cache(
                T_full, self.r, x.device, scale=self.cfg.yarn_scale
            )

        q_r_rot = apply_rope(q_r, cos, sin)
        k_r_rot = apply_rope(k_r, cos, sin)

        Q = torch.cat([q_c, q_r_rot], dim=-1)
        K = torch.cat([k_c, k_r_rot], dim=-1)

        Qc = Q.transpose(1, 2)
        Kc = K.transpose(1, 2)
        Vc = v.transpose(1, 2)

        if self.is_sliding:
            out_ar = self._sliding_window_attn(Qc, Kc, Vc, self.cfg.sliding_window)
        else:
            out_ar = self._chunked_causal_attn(Qc, Kc, Vc)

        out_bidir = F.scaled_dot_product_attention(Qc, Kc, Vc, is_causal=False)

        alpha  = self.gate(x).squeeze(-1)
        merged = (
            alpha.unsqueeze(1).unsqueeze(-1) * out_bidir
            + (1 - alpha.unsqueeze(1).unsqueeze(-1)) * out_ar
        )

        merged = merged.transpose(1, 2).contiguous()
        merged = merged[:, :, :, :self.hd].contiguous()
        merged = merged.view(B, T_full, self.nH * self.hd)

        if mem_tokens is not None:
            merged = merged[:, mem_tokens.shape[1]:, :]
            alpha  = alpha[:, mem_tokens.shape[1]:]

        return self.out(merged), alpha

    def _sliding_window_attn(self, Q, K, V, window: int) -> torch.Tensor:
        B, nH, T, hd = Q.shape
        out    = torch.zeros_like(V)
        half_w = window // 2
        chunk  = self.cfg.chunk_size

        for ci in range(math.ceil(T / chunk)):
            s  = ci * chunk
            e  = min(s + chunk, T)
            ks = max(0, s - half_w)
            ke = min(T, e + half_w)

            Qc = Q[:, :, s:e]
            Kc = K[:, :, ks:ke]
            Vc = V[:, :, ks:ke]

            q_len  = e - s
            k_len  = ke - ks
            offset = s - ks

            q_i = torch.arange(q_len, device=Q.device).unsqueeze(1)
            k_j = torch.arange(k_len, device=Q.device).unsqueeze(0)
            causal_mask = torch.where(
                k_j <= q_i + offset,
                torch.zeros(q_len, k_len, device=Q.device),
                torch.full((q_len, k_len), float("-inf"), device=Q.device),
            )

            out[:, :, s:e] = F.scaled_dot_product_attention(
                Qc, Kc, Vc,
                attn_mask=causal_mask.unsqueeze(0).unsqueeze(0),
            )
        return out

    def _chunked_causal_attn(self, Q, K, V) -> torch.Tensor:
        B, nH, T, hd = Q.shape
        out   = torch.zeros_like(V)
        chunk = self.cfg.chunk_size

        for ci in range(math.ceil(T / chunk)):
            s = ci * chunk
            e = min(s + chunk, T)
            out[:, :, s:e] = F.scaled_dot_product_attention(
                Q[:, :, s:e], K[:, :, :e], V[:, :, :e],
                is_causal=True,
            )
        return out


class DSALite(nn.Module):

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

        Q = self.q_proj(x).view(B, T, self.nH, self.hd).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.nH, self.hd).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.nH, self.hd).transpose(1, 2)

        mask = self._derive_mask(Q, K, U)

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        mask   = mask * causal.unsqueeze(0).unsqueeze(0).float()

        scale  = self.hd ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
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
        B, nH, T, hd = Q.shape
        stride  = 8
        Q_ds    = Q[:, :, ::stride, :]
        K_ds    = K[:, :, ::stride, :]
        scores  = torch.einsum("bhid,bhjd->bhij", Q_ds, K_ds) * (hd ** -0.5)

        Td = scores.shape[-1]
        scores_full = scores.repeat_interleave(stride, dim=2).repeat_interleave(stride, dim=3)
        scores_full = scores_full[:, :, :T, :T]

        if U is not None:
            u_scale     = (1.0 + U.clamp(0, 1)).unsqueeze(1).unsqueeze(-1)
            scores_full = scores_full * u_scale

        k            = max(1, int(T * self.top_k_pct))
        flat         = scores_full.reshape(B * nH * T, T)
        kth_val, _   = torch.topk(flat, k, dim=-1, largest=True, sorted=False)
        threshold    = kth_val[:, -1].unsqueeze(-1).view(B, nH, T, 1)
        return torch.sigmoid((scores_full - threshold) * 10.0)
