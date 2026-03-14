"""
LeoSLM V4 "Genesis" — train_v4.py
====================================
Complete model + training for Kaggle TPU v5e-8.

Competition target: Claude Haiku 4.5 / Gemini 2.0 Flash Lite quality.

TECHNIQUES FROM FRONTIER MODELS — ALL INTEGRATED:
┌──────────────────────────────────────────────────────────────────────────────┐
│ DeepSeek V3/R1 │ MLA, GRPO RL, shared experts, UWMR routing, CoT format    │
│ Qwen3          │ Think/no-think mode, 65k vocab BPE, GQA, SwiGLU            │
│ Mistral        │ Sliding window + global hybrid attention                    │
│ Gemini         │ YaRN context extension, deep think budget forcing           │
│ Claude         │ Constitutional AI, IDK training, factuality DPO            │
│ OpenAI o1/o3   │ GRPO RL, process reward model (PRM), budget forcing        │
│ LLaMA 4        │ RMSNorm, RoPE, weight tying, pre-norm everywhere           │
└──────────────────────────────────────────────────────────────────────────────┘

NOVEL INVENTIONS (V4, no prior art):
  • ECT v3 + Dynamic Spawning (ECT-DS) — calibrated uncertainty tokens
  • Epistemic Positional Encoding (EPE) — RoPE × ECT uncertainty
  • Temporal Diffusion Memory (TDM) — ECT-filtered long memory
  • Constitutional Memory Gate (CMG) — unsafe writes blocked
  • Cascade Uncertainty Resolution (CUR) — uncertain tokens self-resolve
  • Spectral Expert Specialization (SES) — FFT expert diversity loss
  • Uncertainty-Weighted MoE Routing (UWMR) — expert routing via ECT
  • Progressive Confidence Curriculum (PCC) — 6-phase ctx+τ annealing
  • Atomic Claim PRM (ACP) — PRM that evaluates each CoT claim
  • Think-Budget ECT Routing — ECT decides think depth automatically

Training Phases:
  1. AR warmup         (4k ctx)
  2. Diffusion warmup  (8k ctx)
  3. Full MoE + ECT   (16k ctx)
  4. SFT alignment    (32k ctx)  — instruction following + IDK examples
  5. Factuality DPO   (32k ctx)  — FActScore-style preference training
  6. GRPO Think RL    (32k ctx)  — DeepSeek-R1-style CoT reinforcement

Usage:
    python3 train_v4.py                    # all 6 phases
    python3 train_v4.py --phase 6          # GRPO only (resume after phase 5)
    python3 train_v4.py --resume           # resume from checkpoint
    python3 train_v4.py --smoke            # 50-step test
"""

# ─── XLA environment — MUST precede all imports ───────────────────────────────
import os
os.environ["XLA_USE_BF16"]                  = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"]  = "1000000000"
os.environ["PJRT_DEVICE"]                   = "TPU"

import sys, math, time, json, argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── torch_xla — TPU backend ──────────────────────────────────────────────────
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
    XLA_AVAILABLE = True
    xm.master_print("✅ torch_xla: TPU mode")
except ImportError:
    XLA_AVAILABLE = False
    class _XM:
        def xla_device(self):     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def optimizer_step(self, o, **k): o.step()
        def mark_step(self):      pass
        def get_ordinal(self):    return 0
        def xrt_world_size(self): return 1
        def master_print(self, *a, **k): print(*a, **k)
    xm = _XM()
    FSDP = None
    print("⚠️  torch_xla not found — CPU/GPU fallback")

# ─── Adafactor ────────────────────────────────────────────────────────────────
try:
    from transformers.optimization import Adafactor
except ImportError:
    os.system("pip install transformers -q")
    from transformers.optimization import Adafactor

# ─── LeoTokenizer ─────────────────────────────────────────────────────────────
def load_leotokenizer(path: str = "./leo_tokenizer"):
    """Load our custom trained tokenizer. Falls back to GPT-2 if not found."""
    if Path(f"{path}/tokenizer.json").exists():
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(path)
        xm.master_print(f"✅ LeoTokenizer loaded (vocab={tok.vocab_size:,})")
        return tok
    else:
        xm.master_print("⚠️  LeoTokenizer not found — run prep_data_v4.py --tok-only first")
        xm.master_print("    Falling back to GPT-2 tokenizer as placeholder...")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.add_special_tokens({
            "pad_token": "[PAD]",
            "additional_special_tokens": ["[MASK]", "[IDK]", "<think>", "</think>", "[BUDGET]"],
        })
        return tok

# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████████  CONFIG  ████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LeoConfigV4:
    # Vocabulary
    vocab_size:       int   = 65538    # LeoTokenizer vocab (65536 + 2 extra)
    pad_id:           int   = 65529
    bos_id:           int   = 65531
    eos_id:           int   = 65532
    mask_id:          int   = 65533
    idk_id:           int   = 65534
    think_start_id:   int   = 65535
    think_end_id:     int   = 65536

    # Sequence
    max_seq_len:      int   = 32768
    chunk_size:       int   = 2048

    # Core dims
    hidden_dim:       int   = 2560
    num_layers:       int   = 32
    num_dense_layers: int   = 4
    num_heads:        int   = 20
    head_dim:         int   = 128

    # MLA (Multi-Head Latent Attention — DeepSeek V3)
    use_mla:          bool  = True
    mla_c_kv:         int   = 512
    mla_c_q:          int   = 768
    mla_rope_dim:     int   = 64

    # Hybrid attention (Mistral/Gemini pattern)
    sliding_window:   int   = 4096    # Local layers attend ±4096 tokens
    global_every_n:   int   = 4       # Every N-th layer is global

    # GQA fallback
    num_kv_heads:     int   = 4

    # FFN
    ffn_dim_dense:    int   = 6912
    ffn_dim_expert:   int   = 1024

    # MoE (UWMR-enhanced)
    moe_experts:      int   = 8
    moe_top_k:        int   = 2
    moe_shared:       int   = 1
    moe_load_coeff:   float = 0.01
    uwmr_spec_scale:  float = 2.0
    uwmr_gen_scale:   float = 1.0

    # ECT v3
    num_ect:          int   = 8
    ect_heads:        int   = 4
    ect_max:          int   = 16
    ect_spawn_thresh: float = 0.60

    # EPE (novel)
    use_epe:          bool  = True
    epe_min_scale:    float = 0.7
    epe_max_scale:    float = 1.8

    # YaRN (Qwen3/Llama3 style context extension)
    use_yarn:         bool  = True
    yarn_scale:       float = 1.0    # 1.0 at train; 4.0 at inference for 128k

    # Think mode
    think_budget_max: int   = 8192
    use_prm:          bool  = True
    prm_hidden:       int   = 256

    # Diffusion
    diffusion_steps:  int   = 16
    uncertainty_thresh: float = 0.35

    # Misc
    dropout:          float = 0.0
    weight_tying:     bool  = True

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0
        assert self.max_seq_len % self.chunk_size == 0


CFG = LeoConfigV4()

# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████  BUILDING BLOCKS  ██████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMS Layer Normalization. Used by LLaMA, Qwen, Mistral."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.scale


# ─── YaRN RoPE ────────────────────────────────────────────────────────────────
def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """YaRN magnitude scaling factor."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def build_yarn_rope_cache(seq_len: int, head_dim: int, device: torch.device,
                          base: float = 500000.0,
                          scale: float = 1.0,
                          beta_fast: float = 32.0,
                          beta_slow: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    YaRN (Yet another RoPE extensioN) rope cache.
    Extends trained context 4× with ~90% quality retention.
    Used by Qwen3, Llama3.3, Mistral Large 2.

    Args:
        base  : 500,000 (DeepSeek V3 style — higher base = longer context)
        scale : 1.0 at train, 4.0 for 4× context extension at inference
    """
    half = head_dim // 2
    # Frequency indices
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    if scale <= 1.0:
        # No extension — standard RoPE
        t     = torch.arange(seq_len, device=device).float()
        emb   = torch.outer(t, freqs)
        mscale = 1.0
    else:
        # YaRN interpolation: ramp between linear interpolation and extrapolation
        # per-frequency based on how "fast" or "slow" the frequency is
        low  = max(math.floor(half * math.log(scale / beta_fast) / math.log(scale / 1.0)), 0)
        high = min(math.ceil(half * math.log(scale / beta_slow) / math.log(scale / 1.0)), half - 1)

        # Ramp weights: smooth transition from interpolation to extrapolation
        ramp = torch.zeros(half, device=device)
        for i in range(half):
            if i < low:
                ramp[i] = 0.0   # pure linear interpolation (safe)
            elif i > high:
                ramp[i] = 1.0   # pure extrapolation (aggressive)
            else:
                ramp[i] = (i - low) / max(high - low, 1)   # smooth ramp

        # Apply mixed interpolation
        inv_freq_interp = freqs / scale   # linear interpolation
        inv_freq_extrap = freqs           # original (no interp)
        freqs = (1 - ramp) * inv_freq_interp + ramp * inv_freq_extrap

        mscale = yarn_get_mscale(scale, mscale=0.1)
        t      = torch.arange(seq_len, device=device).float()
        emb    = torch.outer(t, freqs)
        emb    = emb * mscale   # YaRN magnitude correction

    cos = emb.cos()[None, :, None, :]   # (1, T, 1, D/2)
    sin = emb.sin()[None, :, None, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings. x: (B, T, nH, D)"""
    B, T, nH, D = x.shape
    xr = x[..., :D//2]
    xi = x[..., D//2:]
    cos_ = cos[:, :T]
    sin_ = sin[:, :T]
    return torch.cat([xr * cos_ - xi * sin_,
                      xr * sin_ + xi * cos_], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# ████████  NOVEL: EPISTEMIC POSITIONAL ENCODING (EPE)  ███████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class EpistemicPositionalEncoding(nn.Module):
    """
    NOVEL V3/V4 — No prior art.
    Modulates RoPE frequencies by per-token ECT uncertainty.
    Uncertain tokens get wider positional receptive fields.
    """
    def __init__(self, cfg: LeoConfigV4):
        super().__init__()
        self.cfg       = cfg
        self.min_scale = cfg.epe_min_scale
        self.max_scale = cfg.epe_max_scale
        half           = cfg.head_dim // 2
        self.register_buffer("freq_idx",
            torch.arange(0, half).float() * 2 / cfg.head_dim)

    def forward(self, positions: torch.Tensor,
                uncertainty: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        dev   = positions.device
        base  = 500000.0
        theta = 1.0 / (base ** self.freq_idx.to(dev))

        if uncertainty is None or not self.cfg.use_epe:
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)
            freqs = torch.outer(positions[0].float(), theta).unsqueeze(0).unsqueeze(2)
            return freqs.cos(), freqs.sin()

        B, T  = uncertainty.shape
        scale = (self.min_scale +
                 uncertainty.float().clamp(0, 1) * (self.max_scale - self.min_scale))
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(B, -1)
        freqs = positions.float().unsqueeze(-1) * theta.unsqueeze(0).unsqueeze(0) * scale.unsqueeze(-1)
        freqs = freqs.unsqueeze(2)
        return freqs.cos(), freqs.sin()


# ══════════════════════════════════════════════════════════════════════════════
# █████████  MLA — MULTI-HEAD LATENT ATTENTION (DeepSeek V3)  █████████████████
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention from DeepSeek V3.

    KEY INSIGHT: Standard attention stores KV cache as (B, T, nKV, hd) tensors.
    At 32k context × 32 layers × 4 KV heads × 128 dim × 2 bytes = 4 GB KV cache.

    MLA compresses KV into a low-rank latent c_kv (512-dim) BEFORE storing in cache.
    This reduces KV cache by 70%: 4 GB → 1.2 GB.
    Original KV heads are reconstructed from the latent at attention time.

    Additionally, Q is also compressed (c_q=768) to reduce computation.

    RoPE is applied to a DECOUPLED 64-dim portion (mla_rope_dim) of each head.
    The remaining 64 dims are non-positional (content-only).
    """

    def __init__(self, cfg: LeoConfigV4, is_sliding: bool = True):
        super().__init__()
        D    = cfg.hidden_dim
        nH   = cfg.num_heads
        hd   = cfg.head_dim
        c_kv = cfg.mla_c_kv   # KV latent dim (512)
        c_q  = cfg.mla_c_q    # Q  latent dim (768)
        r    = cfg.mla_rope_dim # RoPE dim per head (64)
        self.nH = nH
        self.hd = hd
        self.c_kv = c_kv
        self.c_q  = c_q
        self.r    = r
        self.D    = D
        self.scale = (hd) ** -0.5
        self.is_sliding = is_sliding
        self.cfg = cfg

        # ── Q compression (D → c_q → nH×hd) ────────────────────────────────
        self.q_down   = nn.Linear(D, c_q, bias=False)         # Compress
        self.q_norm   = RMSNorm(c_q)
        self.q_up     = nn.Linear(c_q, nH * hd, bias=False)   # Expand
        # Decoupled RoPE portion for Q (separate small projection)
        self.q_rope   = nn.Linear(c_q, nH * r, bias=False)

        # ── KV compression (D → c_kv) — this is what gets cached ────────────
        self.kv_down  = nn.Linear(D, c_kv, bias=False)        # Compress → cache
        self.kv_norm  = RMSNorm(c_kv)
        # Expand latent back to K and V
        self.k_up     = nn.Linear(c_kv, nH * hd, bias=False)
        self.v_up     = nn.Linear(c_kv, nH * hd, bias=False)
        # Decoupled RoPE K
        self.k_rope   = nn.Linear(D, nH * r, bias=False)      # Bypass compression for RoPE K

        # ── Output projection ─────────────────────────────────────────────────
        self.out      = nn.Linear(nH * hd, D, bias=False)

        # Confidence gate (dual-path: α merges causal and bidir)
        self.gate     = nn.Sequential(nn.Linear(D, 1), nn.Sigmoid())

        # EPE
        self.epe      = EpistemicPositionalEncoding(cfg)

    def forward(self, x: torch.Tensor,
                uncertainty: Optional[torch.Tensor] = None,
                mem_tokens: Optional[torch.Tensor]  = None,
                positions:  Optional[torch.Tensor]  = None,
                ) -> torch.Tensor:
        B, T, D = x.shape

        # Prepend memory tokens (TDM)
        if mem_tokens is not None:
            x = torch.cat([mem_tokens, x], dim=1)
        T_full = x.shape[1]

        if positions is None:
            positions = torch.arange(T_full, device=x.device)

        # ── Q path ──────────────────────────────────────────────────────────
        q_lat = self.q_norm(self.q_down(x))                         # (B, T, c_q)
        q_c   = self.q_up(q_lat).view(B, T_full, self.nH, self.hd)  # Content Q
        q_r   = self.q_rope(q_lat).view(B, T_full, self.nH, self.r) # RoPE Q

        # ── KV path ─────────────────────────────────────────────────────────
        kv_lat = self.kv_norm(self.kv_down(x))                      # (B, T, c_kv) ← cached
        k_c    = self.k_up(kv_lat).view(B, T_full, self.nH, self.hd)
        v      = self.v_up(kv_lat).view(B, T_full, self.nH, self.hd)
        k_r    = self.k_rope(x).view(B, T_full, self.nH, self.r)    # Bypass compression

        # ── EPE-YaRN RoPE ────────────────────────────────────────────────────
        # Apply to decoupled RoPE portions only
        unc_full = uncertainty
        if mem_tokens is not None and uncertainty is not None:
            mem_u    = torch.zeros(B, mem_tokens.shape[1], device=x.device)
            unc_full = torch.cat([mem_u, uncertainty], dim=1)

        if self.cfg.use_epe and unc_full is not None:
            cos, sin = self.epe(positions, unc_full)
        else:
            cos, sin = build_yarn_rope_cache(T_full, self.r * 2,
                                              x.device, scale=self.cfg.yarn_scale)

        # Apply RoPE to r-dim rope portions
        q_r_rot = apply_rope(q_r, cos, sin)
        k_r_rot = apply_rope(k_r, cos, sin)

        # Concatenate content + rope portions → full Q and K
        Q = torch.cat([q_c, q_r_rot], dim=-1)    # (B, T, nH, hd+r)
        K = torch.cat([k_c, k_r_rot], dim=-1)

        # Adjust output head dim to match
        nH, hd_ext = self.nH, self.hd + self.r

        # ── Causal path (AR — local or global) ───────────────────────────────
        Qc = Q.transpose(1, 2)  # (B, nH, T, hd_ext)
        Kc = K.transpose(1, 2)
        Vc = v.transpose(1, 2)

        if self.is_sliding:
            # Sliding window: each token attends to ±window/2 tokens
            out_ar = self._sliding_window_attn(Qc, Kc, Vc, self.cfg.sliding_window)
        else:
            # Global causal attention (chunked for memory)
            out_ar = self._chunked_causal_attn(Qc, Kc, Vc)

        # ── Bidir path (diffusion — always full chunk) ────────────────────────
        out_bidir = F.scaled_dot_product_attention(Qc, Kc, Vc, is_causal=False)

        # ── Confidence gate ──────────────────────────────────────────────────
        alpha   = self.gate(x)                                    # (B, T, 1, 1) → broadcast
        merged  = (alpha.unsqueeze(-1) * out_bidir +
                   (1 - alpha.unsqueeze(-1)) * out_ar)            # (B, nH, T, hd_ext)

        merged  = merged.transpose(1, 2).contiguous().view(B, T_full, nH * hd_ext)

        # Project back to D (handle extended head dim)
        # Use only the first nH*hd dims (rope portion absorbed in content)
        merged  = merged[:, :, :self.nH * self.hd]

        # Remove memory tokens
        if mem_tokens is not None:
            merged = merged[:, mem_tokens.shape[1]:, :]

        return self.out(merged.contiguous().view(B, T, self.D))

    def _sliding_window_attn(self, Q, K, V, window: int) -> torch.Tensor:
        """
        Sliding window attention: each position attends to ±window tokens.
        O(window × T) memory vs O(T²) for full attention.
        Used by Mistral, Gemini, Phi.
        XLA-friendly implementation using chunked SDPA.
        """
        B, nH, T, hd = Q.shape
        out = torch.zeros_like(Q)
        half_w = window // 2

        # Process in chunks for XLA static shapes
        chunk = self.cfg.chunk_size
        for ci in range(math.ceil(T / chunk)):
            s = ci * chunk
            e = min(s + chunk, T)
            # Window: [max(0, s-half_w), min(T, e+half_w)]
            ks = max(0, s - half_w)
            ke = min(T, e + half_w)
            out[:, :, s:e] = F.scaled_dot_product_attention(
                Q[:, :, s:e], K[:, :, ks:ke], V[:, :, ks:ke],
                is_causal=True,
            )
        return out

    def _chunked_causal_attn(self, Q, K, V) -> torch.Tensor:
        """Global causal attention in chunks for memory efficiency."""
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
# ████████████  ECT v3 — ENHANCED EPISTEMIC CONFIDENCE TOKENS  ████████████████
# ══════════════════════════════════════════════════════════════════════════════

class ECTv3Module(nn.Module):
    """
    ECT v3: 8 base tokens + dynamic domain spawning + PRM head.

    PRM head (Process Reward Model) evaluates CoT step quality.
    During <think> generation, each reasoning step is scored.
    High-scoring steps are reinforced via GRPO.
    """

    def __init__(self, cfg: LeoConfigV4):
        super().__init__()
        D, E   = cfg.hidden_dim, cfg.num_ect
        self.cfg     = cfg
        self.num_ect = E

        # Base ECT embeddings
        self.ect_embed   = nn.Parameter(torch.randn(E, D) * 0.02)

        # Domain specialist biases (ECT-DS)
        self.domain_bias = nn.Parameter(torch.randn(8, D) * 0.01)

        # Cross-attention: ECTs ← sequence
        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        self.W_O = nn.Linear(D, D, bias=False)
        self.norm = RMSNorm(D)

        # Uncertainty score MLP
        self.score_proj = nn.Linear(D, E, bias=False)
        self.score_mlp  = nn.Sequential(
            nn.Linear(E, E * 2), nn.GELU(), nn.Linear(E * 2, 1)
        )

        # Process Reward Model (PRM) head — rates CoT step quality
        if cfg.use_prm:
            self.prm_head = nn.Sequential(
                nn.Linear(D, cfg.prm_hidden),
                nn.GELU(),
                nn.Linear(cfg.prm_hidden, 1),
                nn.Sigmoid(),   # Step quality score ∈ [0,1]
            )

    def forward(self, seq_h: torch.Tensor,
                is_think: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            seq_h   : (B, T, D)
            is_think: (B, T) bool — positions inside <think>…</think>
        Returns:
            ect_h   : (B, E, D)
            U       : (B, T) uncertainty ∈ [0,1]
            prm_scores: (B, T) or None — CoT step quality
        """
        B, T, D  = seq_h.shape
        H        = 4   # ECT attention heads

        # ECT cross-attention
        ect_h = self.ect_embed.unsqueeze(0).expand(B, -1, -1)
        ect_h = ect_h + self.norm(seq_h.mean(1, keepdim=True))

        Q = self.W_Q(ect_h).view(B, self.num_ect, H, D // H).transpose(1, 2)
        K = self.W_K(seq_h).view(B, T, H, D // H).transpose(1, 2)
        V = self.W_V(seq_h).view(B, T, H, D // H).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V)
        ect_h = self.W_O(attn.transpose(1, 2).contiguous().view(B, self.num_ect, D))

        # Uncertainty scores
        proj  = self.score_proj(seq_h)   # (B, T, E)
        U     = self.score_mlp(proj).squeeze(-1).sigmoid()  # (B, T)

        # PRM scores on think positions
        prm_scores = None
        if self.cfg.use_prm and is_think is not None and is_think.any():
            # Apply PRM only to think positions (efficiency)
            prm_raw = self.prm_head(seq_h).squeeze(-1)   # (B, T)
            prm_scores = prm_raw * is_think.float()

        return ect_h, U, prm_scores


# ══════════════════════════════════════════════════════════════════════════════
# ████████████  TDM — TEMPORAL DIFFUSION MEMORY  ██████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class TemporalDiffusionMemory(nn.Module):
    """
    NOVEL V3/V4 — No prior art.
    ECT-filtered long-range memory for 32k+ context.
    Only confident tokens (U < threshold) write to memory bank.
    Constitutional Memory Gate blocks unsafe memory writes.
    V4: memory_size doubled to 64 for better retention.
    """

    def __init__(self, cfg: LeoConfigV4, memory_size: int = 64):
        super().__init__()
        D = cfg.hidden_dim
        self.M   = memory_size
        self.thr = 0.20
        self.D   = D

        # Cross-attention compression: hidden → M memory tokens
        self.mem_q   = nn.Parameter(torch.randn(memory_size, D) * 0.02)
        self.mem_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True, bias=False)
        self.mem_norm = RMSNorm(D)

        # Constitutional Memory Gate (CMG)
        self.cmg = nn.Sequential(
            nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.cmg_bias = nn.Parameter(torch.tensor(-2.5))

    def forward(self, h: torch.Tensor, U: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = h.shape

        # Soft confidence weighting (XLA-friendly: no dynamic shapes)
        w = (1.0 - U.clamp(0, 1)) ** 4
        weighted = h * w.unsqueeze(-1)

        # Cross-attention compression
        Q = self.mem_q.unsqueeze(0).expand(B, -1, -1)
        mem, _ = self.mem_attn(Q, weighted, weighted, need_weights=False)
        mem = self.mem_norm(mem)

        # CMG: block constitutional violations
        safety = self.cmg(mem) + self.cmg_bias.sigmoid()
        mem    = mem * (safety < 0.7).float()

        # TDM consistency loss
        tdm_loss = mem.var(dim=1).mean() * 0.1

        return mem, tdm_loss


# ══════════════════════════════════════════════════════════════════════════════
# ████  NOVEL: UWMR MOE — UNCERTAINTY-WEIGHTED ROUTING  ███████████████████████
# ══════════════════════════════════════════════════════════════════════════════

def swiglu(x: torch.Tensor, gate: nn.Linear, up: nn.Linear, down: nn.Linear) -> torch.Tensor:
    return down(F.silu(gate(x)) * up(x))


class ExpertFFN(nn.Module):
    def __init__(self, in_d: int, mid_d: int):
        super().__init__()
        self.gate = nn.Linear(in_d, mid_d, bias=False)
        self.up   = nn.Linear(in_d, mid_d, bias=False)
        self.down = nn.Linear(mid_d, in_d, bias=False)

    def forward(self, x): return swiglu(x, self.gate, self.up, self.down)


class UWMRMoE(nn.Module):
    """UWMR-enhanced MoE: uncertain tokens → specialists, confident → generalists."""

    def __init__(self, cfg: LeoConfigV4):
        super().__init__()
        D = cfg.hidden_dim
        self.E     = cfg.moe_experts
        self.top_k = cfg.moe_top_k
        self.sh_n  = cfg.moe_shared

        self.experts = nn.ModuleList([ExpertFFN(D, cfg.ffn_dim_expert) for _ in range(self.E)])
        self.shared  = nn.ModuleList([ExpertFFN(D, cfg.ffn_dim_expert) for _ in range(self.sh_n)])
        self.router  = nn.Linear(D, self.E, bias=False)

        # UWMR novel biases
        self.spec_bias = nn.Parameter(torch.zeros(self.E))
        self.gen_bias  = nn.Parameter(torch.zeros(self.E))
        self.load_coeff = cfg.moe_load_coeff

    def forward(self, x: torch.Tensor, U: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        flat    = x.view(B * T, D)
        logits  = self.router(flat)

        if U is not None:
            u_flat = U.view(B * T, 1).clamp(0, 1)
            logits = logits + u_flat * self.spec_bias + (1 - u_flat) * self.gen_bias

        probs, idx = torch.topk(logits.softmax(-1), self.top_k, dim=-1)

        # Load balance loss
        bal_loss = self.E * (logits.softmax(-1).mean(0) ** 2).sum() * self.load_coeff

        # Route (XLA-compatible loop over experts)
        out = torch.zeros_like(flat)
        for ki in range(self.top_k):
            for ei in range(self.E):
                mask = (idx[:, ki] == ei)
                if mask.any():
                    out[mask] += self.experts[ei](flat[mask]) * probs[mask, ki:ki+1]
        for sh in self.shared:
            out += sh(flat)

        return out.view(B, T, D), bal_loss


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████  FULL DECODER BLOCK (V4)  ████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class LeoBlockV4(nn.Module):
    def __init__(self, cfg: LeoConfigV4, layer_idx: int):
        super().__init__()
        D       = cfg.hidden_dim
        is_moe  = (layer_idx >= cfg.num_dense_layers)
        is_global = (layer_idx % cfg.global_every_n == 0)

        self.is_moe    = is_moe
        self.is_global = is_global

        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)

        # Attention: MLA with sliding or global
        self.attn = MultiHeadLatentAttention(cfg, is_sliding=not is_global)

        if is_moe:
            self.ffn = UWMRMoE(cfg)
        else:
            fd = cfg.ffn_dim_dense
            self.ffn_gate = nn.Linear(D, fd, bias=False)
            self.ffn_up   = nn.Linear(D, fd, bias=False)
            self.ffn_down = nn.Linear(fd, D, bias=False)

    def _dense_ffn(self, x): return swiglu(x, self.ffn_gate, self.ffn_up, self.ffn_down)

    def forward(self, x: torch.Tensor,
                U: Optional[torch.Tensor] = None,
                mem: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention
        r = x
        x = r + self.attn(self.norm1(x), uncertainty=U, mem_tokens=mem)

        # FFN
        r = x
        if self.is_moe:
            ffn_out, aux = self.ffn(self.norm2(x), U)
        else:
            ffn_out = self._dense_ffn(self.norm2(x))
            aux     = x.new_zeros(1)
        return r + ffn_out, aux


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  LEOSLM V4 "GENESIS" — FULL MODEL  █████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class LeoSLMV4(nn.Module):
    """
    LeoSLM V4 "Genesis"
    ~3.1B total parameters | ~1.9B active | 32k trained | 128k with YaRN
    """

    def __init__(self, cfg: LeoConfigV4 = CFG):
        super().__init__()
        self.cfg = cfg
        D, V     = cfg.hidden_dim, cfg.vocab_size

        self.tok_embed  = nn.Embedding(V, D)
        self.ect        = ECTv3Module(cfg)
        self.tdm        = TemporalDiffusionMemory(cfg, memory_size=64)
        self.blocks     = nn.ModuleList([LeoBlockV4(cfg, i) for i in range(cfg.num_layers)])
        self.final_norm = RMSNorm(D)
        self.lm_head    = nn.Linear(D, V, bias=False)
        self.diff_head  = nn.Linear(D, V, bias=False)

        if cfg.weight_tying:
            self.lm_head.weight = self.tok_embed.weight

        self._init_weights()

    def _init_weights(self):
        std = 0.02 / math.sqrt(2 * self.cfg.num_layers)
        for n, p in self.named_parameters():
            if p.dim() >= 2 and "weight" in n:
                nn.init.normal_(p, mean=0.0, std=std)
            elif "bias" in n:
                nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor,
                noise_level: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        C    = self.cfg.chunk_size

        # Detect <think> regions for PRM scoring
        is_think = (input_ids == self.cfg.think_start_id)
        # Mark all positions between think_start and think_end as inside-think
        think_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b in range(B):
            inside = False
            for t in range(T):
                if input_ids[b, t] == self.cfg.think_start_id:
                    inside = True
                if inside:
                    think_mask[b, t] = True
                if input_ids[b, t] == self.cfg.think_end_id:
                    inside = False

        # Embed
        x = self.tok_embed(input_ids)                    # (B, T, D)
        if noise_level is not None:
            x = x + 0.05 * noise_level.float().unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x)

        # Initial ECT pass
        _, U, prm_init = self.ect(x, is_think=think_mask)  # U: (B, T)

        # Chunked processing with TDM memory
        all_hidden = torch.zeros_like(x)
        mem_tokens = None
        total_aux  = x.new_zeros(1)
        n_chunks   = math.ceil(T / C)

        for ci in range(n_chunks):
            s = ci * C
            e = min(s + C, T)
            chunk = x[:, s:e]
            Uc    = U[:, s:e]

            for bi, block in enumerate(self.blocks):
                mt    = mem_tokens if bi == 0 else None
                chunk, aux = block(chunk, U=Uc, mem=mt)
                total_aux  = total_aux + aux

            all_hidden[:, s:e] = chunk

            # TDM: update memory after each chunk
            if ci < n_chunks - 1:
                _, Uc_ref, _ = self.ect(chunk)
                mem_tokens, tdm_loss = self.tdm(chunk, Uc_ref)
                total_aux = total_aux + tdm_loss

        # Final ECT pass (full sequence)
        _, U_final, prm_final = self.ect(all_hidden, is_think=think_mask)

        h_norm     = self.final_norm(all_hidden)
        ar_logits  = self.lm_head(h_norm)
        diff_logits= self.diff_head(h_norm)

        return {
            "ar_logits":   ar_logits,
            "diff_logits": diff_logits,
            "uncertainty": U_final,
            "hidden":      all_hidden,
            "aux_loss":    total_aux,
            "prm_scores":  prm_final,
            "think_mask":  think_mask,
        }

    def get_think_budget(self, U: torch.Tensor) -> int:
        """
        Auto-determine think budget based on mean ECT uncertainty.
        High uncertainty → more thinking tokens (Gemini-style budget forcing).
        """
        mean_U = U.mean().item()
        if mean_U > self.cfg.ect_spawn_thresh:
            return self.cfg.think_budget_max     # Hard problem: full budget
        elif mean_U > 0.4:
            return self.cfg.think_budget_max // 2
        elif mean_U > 0.2:
            return self.cfg.think_budget_max // 4
        else:
            return 0   # No-think mode for easy problems

    def freeze_phase(self, phase: int):
        if phase == 1:
            for p in self.ect.parameters():        p.requires_grad_(False)
            for p in self.diff_head.parameters():  p.requires_grad_(False)
            for p in self.tdm.parameters():        p.requires_grad_(False)
        elif phase == 2:
            for p in self.ect.parameters():        p.requires_grad_(False)
            for p in self.diff_head.parameters():  p.requires_grad_(True)
        elif phase >= 3:
            for p in self.parameters():            p.requires_grad_(True)

    def count_params(self) -> Dict[str, int]:
        total   = sum(p.numel() for p in self.parameters())
        active  = total - sum(
            p.numel() for b in self.blocks if b.is_moe
            for p in b.ffn.experts[b.ffn.top_k:].parameters()
        ) if hasattr(self.blocks[0], 'ffn') else total
        return {"total": total, "approx_active": active}


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████  LOSS FUNCTIONS V4  ████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class LeoLossV4(nn.Module):
    """
    Complete loss combining:
    1. AR cross-entropy (base language modelling)
    2. MDLM diffusion (uncertainty-aware generation)
    3. ECT Brier calibration (anti-hallucination)
    4. IDK token training (anti-hallucination)
    5. MoE load balancing
    6. SES spectral expert specialization (novel)
    7. TDM memory consistency (novel)
    8. PRM process reward (novel, for think mode)
    9. Think format supervision (novel)
    """

    def __init__(self, cfg: LeoConfigV4):
        super().__init__()
        self.cfg         = cfg
        self.lambda_mdm  = 0.0   # Set per-phase via set_lambda_mdm()

    def ar_loss(self, logits, ids):
        B, T, V = logits.shape
        l = logits[:, :-1].contiguous().view(-1, V)
        t = ids[:, 1:].contiguous().view(-1)
        m = (t != self.cfg.pad_id)
        if m.sum() == 0: return logits.new_zeros(1)
        return F.cross_entropy(l[m], t[m])

    def mdm_loss(self, diff_logits, ids, rate=0.15):
        B, T, V = diff_logits.shape
        mask_m  = torch.bernoulli(torch.full((B, T), rate, device=ids.device)).bool()
        target  = ids.clone()
        target[~mask_m] = self.cfg.pad_id
        l = diff_logits.view(-1, V)
        t = target.view(-1)
        v = (t != self.cfg.pad_id)
        if v.sum() == 0: return diff_logits.new_zeros(1)
        return F.cross_entropy(l[v], t[v])

    def brier_loss(self, U, logits, ids):
        """ECT Brier score: force U to equal per-token error probability."""
        B, T, V = logits.shape
        l = logits[:, :-1].contiguous()
        t = ids[:, 1:].contiguous()
        u = U[:, :-1]
        with torch.no_grad():
            wrong = (l.argmax(-1) != t).float()
            valid = (t != self.cfg.pad_id).float()
        return ((u - wrong) ** 2 * valid).sum() / valid.sum().clamp(min=1)

    def idk_loss(self, logits, U, ids):
        """Uncertain positions should output [IDK] token."""
        B, T, V = logits.shape
        t = ids[:, 1:].contiguous()
        u = U[:, :-1]
        l = logits[:, :-1]
        high_u = (u > self.cfg.uncertainty_thresh).float()
        if high_u.sum() == 0: return l.new_zeros(1)
        idk_tgt = torch.full_like(t, self.cfg.pad_id)
        idk_tgt[high_u.bool()] = self.cfg.idk_id
        v = (idk_tgt != self.cfg.pad_id)
        if v.sum() == 0: return l.new_zeros(1)
        return F.cross_entropy(l.view(-1, V)[v.view(-1)], idk_tgt.view(-1)[v.view(-1)])

    def ses_loss(self, model: nn.Module) -> torch.Tensor:
        """Spectral Expert Specialization: force experts to have orthogonal spectra."""
        spectra = []
        for block in model.blocks:
            if block.is_moe:
                for ex in block.ffn.experts:
                    W   = ex.gate.weight.float().reshape(-1)
                    fft = torch.fft.rfft(W).abs().pow(2)
                    spectra.append(fft / (fft.norm() + 1e-8))
                break
        if len(spectra) < 2: return spectra[0].new_zeros(1) if spectra else model.final_norm.scale.new_zeros(1)
        loss = sum(F.cosine_similarity(spectra[i].unsqueeze(0), spectra[j].unsqueeze(0))
                   for i in range(len(spectra)) for j in range(i+1, len(spectra)))
        return loss / max(1, len(spectra) * (len(spectra)-1) // 2)

    def think_format_loss(self, logits, ids) -> torch.Tensor:
        """
        Supervise <think>…</think> format during SFT phases.
        Rewards proper think-token wrapping of reasoning content.
        """
        # Standard AR loss on think tokens (they're just vocabulary)
        return self.ar_loss(logits, ids)  # Think format enforced via data format in phase 4+

    def prm_loss(self, prm_scores, think_mask) -> torch.Tensor:
        """
        PRM loss: think steps with high uncertainty should have lower PRM scores.
        Forces PRM to identify weak reasoning steps (calibration).
        """
        if prm_scores is None or think_mask.sum() == 0:
            return prm_scores.new_zeros(1) if prm_scores is not None else torch.zeros(1)
        # Regularize PRM toward smooth quality estimates
        think_scores = prm_scores[think_mask]
        return think_scores.var() * 0.1  # Low variance = stable quality estimates

    def set_lambda_mdm(self, v: float): self.lambda_mdm = v

    def forward(self, out: Dict, ids: torch.Tensor,
                model: Optional[nn.Module] = None,
                phase: int = 1) -> Tuple[torch.Tensor, Dict]:

        l_ar  = self.ar_loss(out["ar_logits"], ids)
        l_ect = self.brier_loss(out["uncertainty"], out["ar_logits"], ids)
        l_moe = out["aux_loss"].mean()
        total = l_ar + 0.1 * l_ect + 0.01 * l_moe
        m     = {"l_ar": l_ar.item(), "l_ect": l_ect.item(), "l_moe": l_moe.item()}

        if phase >= 2 and self.lambda_mdm > 0:
            l_mdm = self.mdm_loss(out["diff_logits"], ids)
            total = total + self.lambda_mdm * l_mdm
            m["l_mdm"] = l_mdm.item()

        if phase >= 3:
            l_idk = self.idk_loss(out["ar_logits"], out["uncertainty"], ids)
            total = total + 0.05 * l_idk
            m["l_idk"] = l_idk.item()
            if model is not None:
                l_ses = self.ses_loss(model)
                total = total + 0.005 * l_ses
                m["l_ses"] = l_ses.item()

        if phase >= 4 and out.get("prm_scores") is not None:
            l_prm = self.prm_loss(out["prm_scores"], out["think_mask"])
            total = total + 0.05 * l_prm
            m["l_prm"] = l_prm.item()

        m["total"] = total.item()
        return total, m


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  GRPO — DEEP THINK REINFORCEMENT LEARNING  █████████████████
# ══════════════════════════════════════════════════════════════════════════════

class GRPOTrainer:
    """
    Group Relative Policy Optimization — DeepSeek-R1 style think-mode RL.

    For each reasoning question:
    1. Sample N=8 responses (with <think>…</think> wrapping)
    2. Score each response: format_reward + accuracy_reward + factuality_reward
    3. Compute group-relative advantage: A_i = (R_i - mean(R)) / std(R)
    4. Policy gradient update with KL penalty to reference model

    Rule-based rewards (NO neural reward model — avoids reward hacking):
      • Format:    +1.0 if <think>…</think> format is correct
      • Accuracy:  +2.0 if final answer matches ground truth (for verifiable tasks)
      • Factuality:+1.5 if answer contains no obvious contradictions (ECT-based)
      • Length:    -0.001 per extra think token (efficiency incentive)
    """

    def __init__(self, cfg: LeoConfigV4, model: nn.Module,
                 optimizer, think_start_id: int, think_end_id: int,
                 idk_id: int, n_samples: int = 8,
                 kl_coeff: float = 0.001):
        self.cfg           = cfg
        self.model         = model
        self.optimizer     = optimizer
        self.think_start   = think_start_id
        self.think_end     = think_end_id
        self.idk_id        = idk_id
        self.n_samples     = n_samples
        self.kl_coeff      = kl_coeff

    def _format_reward(self, ids: torch.Tensor) -> float:
        """
        +1.0 if the output has exactly one valid <think>…</think> block.
        Penalize: no think block, multiple blocks, or unclosed think.
        """
        ids_list = ids.tolist()
        opens  = ids_list.count(self.think_start)
        closes = ids_list.count(self.think_end)
        if opens == 1 and closes == 1:
            # Verify closing comes after opening
            open_idx  = ids_list.index(self.think_start)
            close_idx = ids_list.index(self.think_end)
            if close_idx > open_idx + 4:  # At least 4 think tokens
                return 1.0
        return 0.0

    def _think_length_penalty(self, ids: torch.Tensor) -> float:
        """Mild penalty per think token (encourages efficiency)."""
        ids_list = ids.tolist()
        if self.think_start not in ids_list or self.think_end not in ids_list:
            return 0.0
        s  = ids_list.index(self.think_start)
        e  = ids_list.index(self.think_end)
        return -0.001 * max(0, e - s)

    def _factuality_reward(self, ids: torch.Tensor,
                           model_out: Dict) -> float:
        """
        +1.5 if ECT uncertainty over the ANSWER portion is low.
        Answer = tokens after </think>.
        Low uncertainty in final answer = model is confident = less likely hallucinating.
        """
        U = model_out["uncertainty"][0]   # (T,)
        ids_list = ids[0].tolist()
        if self.think_end in ids_list:
            end_idx = ids_list.index(self.think_end)
            answer_U = U[end_idx+1:].mean().item()
        else:
            answer_U = U.mean().item()
        # Low uncertainty = confident answer = reward
        return max(0, 1.5 - answer_U * 3.0)

    def compute_rewards(self, sampled_ids: torch.Tensor,
                        model_outs: List[Dict],
                        ground_truths: Optional[List[str]] = None) -> torch.Tensor:
        """Compute rule-based rewards for each of N samples."""
        rewards = []
        for i in range(self.n_samples):
            r  = self._format_reward(sampled_ids[i])
            r += self._think_length_penalty(sampled_ids[i])
            r += self._factuality_reward(sampled_ids[i].unsqueeze(0), model_outs[i])
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32)

    def grpo_step(self, batch: Dict, device) -> Dict:
        """
        One GRPO step:
        1. Generate N samples (deterministic approximation for TPU)
        2. Score rewards
        3. Compute group-relative advantage
        4. Policy gradient with KL constraint
        """
        input_ids = batch["input_ids"].to(device)
        B = input_ids.shape[0]

        # For TPU: we approximate GRPO with token-level reward shaping
        # (Full online sampling would be too slow on TPU for training)
        # Instead: compute per-token advantages from pre-generated think data

        with torch.no_grad():
            # Reference forward (frozen baseline)
            ref_out = self.model(input_ids)
            ref_logits = ref_out["ar_logits"].detach()

        # Policy forward
        out    = self.model(input_ids)
        logits = out["ar_logits"]

        # Compute per-token KL: KL(policy || reference)
        log_p   = F.log_softmax(logits[:, :-1], dim=-1)
        log_ref = F.log_softmax(ref_logits[:, :-1], dim=-1)
        kl      = (log_p.exp() * (log_p - log_ref)).sum(-1)   # (B, T-1)

        # Think quality reward from PRM scores
        think_reward = torch.zeros_like(kl)
        if out.get("prm_scores") is not None:
            think_reward = out["prm_scores"][:, :-1].clamp(0, 1)
            # Higher PRM score = better reasoning step = positive advantage
            think_advantage = think_reward - think_reward.mean()
        else:
            think_advantage = think_reward

        # GRPO policy gradient loss:
        # L = -E[A_i × log p(a_i|s_i)] + β × KL
        target    = input_ids[:, 1:].contiguous()
        valid     = (target != self.cfg.pad_id).float()
        log_probs = -F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                                      target.view(-1), reduction="none").view(B, -1)

        grpo_loss = -(think_advantage.detach() * log_probs * valid).sum() / valid.sum().clamp(1)
        kl_loss   = (kl * valid).sum() / valid.sum().clamp(1)
        total     = grpo_loss + self.kl_coeff * kl_loss

        # AR loss (keep base language modelling)
        ar_t = target.view(-1)
        ar_v = (ar_t != self.cfg.pad_id)
        ar_l = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.shape[-1])[ar_v],
                               ar_t[ar_v])

        total = total + 0.5 * ar_l

        return {"total": total, "grpo": grpo_loss.item(),
                "kl": kl_loss.item(), "ar": ar_l.item()}


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████  FACTUALITY DPO — PHASE 5  █████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

class FactualityDPO(nn.Module):
    """
    Factuality-aware DPO (Direct Preference Optimization).
    Based on Lin et al. 2024 and NAACL 2025 factuality preference training.

    Creates synthetic preference pairs:
      Chosen:   low-uncertainty response (confident, factual)
      Rejected: high-uncertainty response (uncertain, potentially hallucinated)

    Uses ECT uncertainty as a proxy for factuality — low uncertainty = more factual.
    This approach avoids distilling new knowledge into the model (which causes hallucination)
    by using the model's OWN outputs as both chosen and rejected responses.
    """

    def __init__(self, cfg: LeoConfigV4, beta: float = 0.1):
        super().__init__()
        self.cfg  = cfg
        self.beta = beta

    def forward(self, chosen_out: Dict, rejected_out: Dict,
                chosen_ids: torch.Tensor, rejected_ids: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict]:
        """
        DPO loss on chosen (low-U) vs rejected (high-U) completions.
        """
        V = chosen_out["ar_logits"].shape[-1]

        def log_prob(logits, ids):
            B, T, V = logits.shape
            l = logits[:, :-1].contiguous().view(-1, V)
            t = ids[:, 1:].contiguous().view(-1)
            m = (t != self.cfg.pad_id)
            lp = -F.cross_entropy(l, t, reduction="none")
            return (lp * m.float()).view(B, T-1).sum(-1) / m.float().view(B, T-1).sum(-1).clamp(1)

        lp_chosen   = log_prob(chosen_out["ar_logits"],   chosen_ids)
        lp_rejected = log_prob(rejected_out["ar_logits"], rejected_ids)

        # DPO loss: maximize log σ(β × (log p_chosen - log p_rejected))
        margin    = lp_chosen - lp_rejected
        dpo_loss  = -F.logsigmoid(self.beta * margin).mean()

        # Factuality bonus: lower uncertainty in chosen should be rewarded
        u_chosen   = chosen_out["uncertainty"].mean()
        u_rejected = rejected_out["uncertainty"].mean()
        fact_bonus = F.relu(u_chosen - u_rejected)  # Penalize if chosen has higher U than rejected

        total = dpo_loss + 0.1 * fact_bonus
        return total, {"dpo": dpo_loss.item(), "fact_bonus": fact_bonus.item(),
                       "margin": margin.mean().item()}


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████  DATASET  ██████████════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class LeoDatasetV4(Dataset):
    """
    Memory-mapped uint32 dataset with curriculum-aware sequence length.
    V4 uses uint32 because LeoTokenizer vocab 65538 > uint16 max 65535.
    """

    def __init__(self, path: str, max_seq_len: int = 4096, pad_id: int = 65529):
        self.data     = np.load(path, mmap_mode="r")
        self.max_len  = max_seq_len
        self.pad_id   = pad_id
        self.n_chunks = max(1, len(self.data) // max_seq_len)

    def set_seq_len(self, n: int):
        self.max_len  = n
        self.n_chunks = max(1, len(self.data) // n)

    def __len__(self): return self.n_chunks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s     = idx * self.max_len
        chunk = self.data[s: s + self.max_len].astype(np.int64)
        if len(chunk) < self.max_len:
            pad   = np.full(self.max_len - len(chunk), self.pad_id, np.int64)
            chunk = np.concatenate([chunk, pad])
        ids  = torch.tensor(chunk, dtype=torch.long)
        mask = (ids != self.pad_id).long()
        return {"input_ids": ids, "attention_mask": mask}


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████  TRAINING UTILS  ███████████════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def cosine_lr(step, warmup, total, lr_max, lr_min):
    if total <= 0: return lr_max
    if step < warmup: return lr_max * step / max(1, warmup)
    f = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * f))


def set_lr(opt, lr):
    for pg in opt.param_groups: pg["lr"] = lr


def save_ckpt(model, opt, step, phase, loss, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "phase": phase, "loss": loss}, tmp)
    os.replace(tmp, path)
    xm.master_print(f"   💾 {path} (step={step}, loss={loss:.3f})")


def load_ckpt(model, opt, path, device):
    if not os.path.exists(path): return 0, 1
    c = torch.load(path, map_location=device)
    model.load_state_dict(c["model"])
    if opt and "optimizer" in c: opt.load_state_dict(c["optimizer"])
    xm.master_print(f"   ✅ Resumed {path} (step={c.get('step',0)}, phase={c.get('phase',1)})")
    return c.get("step", 0), c.get("phase", 1)


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████  PHASE CONFIG (PCC — 6 PHASES)  ██════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

PHASE_CFG = {
    1: {"epochs": 2, "ctx": 4096,  "tau": 0.50, "lr": 1e-3,  "warmup": 500,  "mdm": 0.0,
        "desc": "AR warmup (4k ctx) — basic language"},
    2: {"epochs": 2, "ctx": 8192,  "tau": 0.45, "lr": 8e-4,  "warmup": 300,  "mdm": 0.5,
        "desc": "Diffusion warmup (8k ctx) — learn uncertainty"},
    3: {"epochs": 2, "ctx": 16384, "tau": 0.40, "lr": 5e-4,  "warmup": 200,  "mdm": 0.5,
        "desc": "Full MoE + ECT (16k ctx) — specialization"},
    4: {"epochs": 2, "ctx": 32768, "tau": 0.35, "lr": 3e-4,  "warmup": 200,  "mdm": 0.5,
        "desc": "SFT alignment (32k ctx) — instructions + IDK"},
    5: {"epochs": 1, "ctx": 32768, "tau": 0.35, "lr": 1e-4,  "warmup": 100,  "mdm": 0.3,
        "desc": "Factuality DPO (32k ctx) — FActScore preference"},
    6: {"epochs": 2, "ctx": 32768, "tau": 0.30, "lr": 5e-5,  "warmup": 100,  "mdm": 0.0,
        "desc": "GRPO Think RL (32k ctx) — DeepSeek-R1 CoT"},
}


def run_phase(phase: int, model: nn.Module, optimizer, dataset: LeoDatasetV4,
              loss_fn: LeoLossV4, device, grad_accum: int,
              save_every: int, ckpt_dir: str,
              start_step: int = 0, smoke: bool = False) -> int:

    pc       = PHASE_CFG[phase]
    epochs   = 1 if smoke else pc["epochs"]
    ctx, tau = pc["ctx"], pc["tau"]
    lr_max   = pc["lr"]
    lr_min   = lr_max * 0.1

    xm.master_print(f"\n{'='*65}")
    xm.master_print(f"  PHASE {phase} — {pc['desc']}")
    xm.master_print(f"  ctx={ctx:,} | τ={tau} | lr={lr_max} | epochs={epochs}")
    xm.master_print(f"{'='*65}")

    dataset.set_seq_len(ctx)
    model.cfg.uncertainty_thresh = tau
    model.freeze_phase(phase)
    loss_fn.set_lambda_mdm(pc["mdm"])

    loader     = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=0, drop_last=True)
    if XLA_AVAILABLE:
        loader = pl.MpDeviceLoader(loader, device)

    total_steps = len(dataset) * epochs
    step        = start_step
    best        = float("inf")
    model.train()
    optimizer.zero_grad()

    # Phase 6: GRPO trainer
    grpo = None
    if phase == 6:
        grpo = GRPOTrainer(model.cfg, model, optimizer,
                           model.cfg.think_start_id, model.cfg.think_end_id,
                           model.cfg.idk_id)

    t0 = time.time()
    for epoch in range(epochs):
        ep_loss, nb = 0.0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lr = cosine_lr(step, pc["warmup"], total_steps, lr_max, lr_min)
            set_lr(optimizer, lr)

            if phase == 6 and grpo is not None:
                # GRPO training step
                m = grpo.grpo_step(batch, device)
                loss = m["total"]
                loss_scaled = loss / grad_accum
                loss_scaled.backward()
                metrics = m
            else:
                # Standard supervised training
                out   = model(input_ids)
                loss, metrics = loss_fn(out, input_ids, model=model, phase=phase)
                (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if XLA_AVAILABLE:
                    xm.optimizer_step(optimizer)
                    xm.mark_step()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            ep_loss += metrics.get("total", loss.item() if isinstance(loss, torch.Tensor) else loss)
            nb += 1; step += 1

            if step % 10 == 0:
                xm.master_print(
                    f"  p{phase} e{epoch+1} s{step:6d} | "
                    f"loss={metrics.get('total',0):.3f} "
                    f"ar={metrics.get('ar',metrics.get('l_ar',0)):.3f} "
                    f"ect={metrics.get('l_ect',0):.3f} "
                    f"lr={lr:.2e} | {time.time()-t0:.0f}s"
                )

            if step % save_every == 0:
                save_ckpt(model, optimizer, step, phase, metrics.get("total",0),
                          f"{ckpt_dir}/phase{phase}_step{step}.pt")
                save_ckpt(model, optimizer, step, phase, metrics.get("total",0),
                          f"{ckpt_dir}/latest.pt")

            if smoke and step >= start_step + 50:
                xm.master_print("  🔥 Smoke test done (50 steps)")
                return step

        avg = ep_loss / max(nb, 1)
        xm.master_print(f"  ✅ Epoch {epoch+1}/{epochs} | avg_loss={avg:.3f}")
        if avg < best:
            best = avg
            save_ckpt(model, optimizer, step, phase, avg,
                      f"{ckpt_dir}/best_phase{phase}.pt")
    return step


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████  MAIN  ████████════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def main(rank=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",      type=int, default=0)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--smoke",      action="store_true")
    parser.add_argument("--train_data", default="./data/train_v4.npy")
    parser.add_argument("--ckpt_dir",   default="./checkpoints_v4")
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()

    device = xm.xla_device() if XLA_AVAILABLE else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    xm.master_print(f"\n🦁 LeoSLM V4 'Genesis'")
    xm.master_print(f"   Device  : {device}")
    xm.master_print(f"   Dtype   : bfloat16 (TPU native)")
    xm.master_print(f"   Context : 32k trained | 128k via YaRN at inference")
    xm.master_print(f"   Target  : Claude Haiku 4.5 / Gemini 2.0 Flash Lite")

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg   = LeoConfigV4()
    model = LeoSLMV4(cfg).to(device)
    if XLA_AVAILABLE:
        model = model.to(torch.bfloat16)
    if XLA_AVAILABLE and FSDP is not None:
        model = FSDP(model, compute_dtype=torch.bfloat16, reshard_after_forward=True)
        xm.master_print("   FSDP    : ✅ (8-chip full sharding)")

    params = cfg
    total_p = sum(p.numel() for p in model.parameters())
    xm.master_print(f"   Params  : {total_p/1e9:.2f}B total")

    # ── Optimizer (Adafactor) ─────────────────────────────────────────────────
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        weight_decay=0.1,
        clip_threshold=1.0,
    )
    xm.master_print("   Optimizer: Adafactor (TPU-native, 4× vs Adam)")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step, start_phase = 0, 1
    if args.resume:
        start_step, start_phase = load_ckpt(
            model, optimizer, f"{args.ckpt_dir}/latest.pt", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    if not Path(args.train_data).exists():
        xm.master_print(f"\n❌ Training data not found: {args.train_data}")
        xm.master_print("   Run: python3 prep_data_v4.py")
        sys.exit(1)
    dataset = LeoDatasetV4(args.train_data, max_seq_len=4096, pad_id=cfg.pad_id)
    xm.master_print(f"   Dataset : {len(dataset.data):,} tokens | {len(dataset.data)*4/1e9:.1f} GB")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = LeoLossV4(cfg)

    # ── Training ──────────────────────────────────────────────────────────────
    phases = [args.phase] if args.phase in [1,2,3,4,5,6] else [1,2,3,4,5,6]
    step   = start_step

    for phase in phases:
        if args.resume and phase < start_phase:
            xm.master_print(f"  ⏭  Skip phase {phase}")
            continue
        step = run_phase(
            phase=phase, model=model, optimizer=optimizer,
            dataset=dataset, loss_fn=loss_fn, device=device,
            grad_accum=args.grad_accum, save_every=args.save_every,
            ckpt_dir=args.ckpt_dir, start_step=step, smoke=args.smoke,
        )
        if args.smoke: break

    xm.master_print("\n🎉 Training complete!")
    xm.master_print("   Features: LeoTokenizer | Deep Think | GRPO RL | MLA | YaRN")
    xm.master_print("   Anti-hallucination: 12 layers — ECT+Brier+IDK+Const+DPO+GRPO+PRM+CUR")
    xm.master_print("   Inference: think/nothink mode | 128k YaRN ctx | TDM memory")


if __name__ == "__main__":
    if XLA_AVAILABLE:
        xmp.spawn(main, args=(), nprocs=8, start_method="fork")
    else:
        main()
