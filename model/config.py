"""
model/config.py — LeoSLM "Aether" unified configuration
=========================================================
Single source of truth for every hyperparameter.
Imports from nowhere inside the project; everyone else imports from here.

Loss weights live here so they can be tuned without touching source code.
TDM / CMG thresholds live here for the same reason.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LeoConfig:
    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_size:        int   = 65543    # LeoTokenizer 65k BPE + 14 special tokens
    pad_id:            int   = 65529
    bos_id:            int   = 65531
    eos_id:            int   = 65532
    mask_id:           int   = 65533   # [MASK] — diffusion input
    idk_id:            int   = 65534   # [IDK]  — uncertain output
    think_start_id:    int   = 65535   # <think>
    think_end_id:      int   = 65536   # </think>
    tool_call_start:   int   = 65537   # <|tool_call|>
    tool_call_end:     int   = 65538   # <|/tool_call|>
    tool_result_start: int   = 65539   # <|tool_result|>
    tool_result_end:   int   = 65540   # <|/tool_result|>
    system_start:      int   = 65541   # <|system|>
    estr_domain_count: int   = 8       # ECT-Seeded Tool Routing domain slots

    # ── Sequence ──────────────────────────────────────────────────────────────
    max_seq_len:       int   = 32768
    chunk_size:        int   = 2048
    think_budget_max:  int   = 8192    # Max <think> tokens per call
    think_budget_min:  int   = 256     # Min <think> tokens for hard queries

    # ── Core dims ─────────────────────────────────────────────────────────────
    hidden_dim:        int   = 2560
    num_layers:        int   = 32
    num_dense_layers:  int   = 4       # First N layers use dense FFN; rest → MoE
    num_heads:         int   = 20
    head_dim:          int   = 128     # hidden_dim / num_heads == 128 (asserted)

    # ── MLA — Multi-Head Latent Attention (DeepSeek V3) ───────────────────────
    use_mla:           bool  = True
    mla_c_kv:          int   = 512     # KV latent dim — what goes in the KV cache
    mla_c_q:           int   = 768     # Q  latent dim
    mla_rope_dim:      int   = 64      # Decoupled RoPE portion per head

    # ── Hybrid attention (Mistral / Gemini pattern) ───────────────────────────
    sliding_window:    int   = 4096    # Local layers: each token attends ±4096
    global_every_n:    int   = 4       # Every N-th layer uses full causal attention

    # ── DSA-lite — Derived Sparse Attention (Aether novel, XLA-safe) ──────────
    use_dsa:           bool  = True
    dsa_threshold:     int   = 32768   # Activate sparse mask for seqlen > this
    dsa_top_k_pct:     float = 0.25    # Attend to top-25% positions per head

    # ── GQA fallback ──────────────────────────────────────────────────────────
    num_kv_heads:      int   = 4

    # ── FFN ───────────────────────────────────────────────────────────────────
    ffn_dim_dense:     int   = 6912    # Dense stem SwiGLU dim
    ffn_dim_expert:    int   = 1024    # Per-expert SwiGLU inner dim

    # ── MoE — UWMR-enhanced ───────────────────────────────────────────────────
    moe_experts:       int   = 8
    moe_top_k:         int   = 2       # Active experts per token
    moe_shared:        int   = 1       # Shared experts always active
    moe_load_coeff:    float = 0.01    # Load-balance loss coefficient
    uwmr_spec_scale:   float = 2.0     # UWMR bias for specialist routing
    uwmr_gen_scale:    float = 1.0     # UWMR bias for generalist routing

    # ── ECT v3 — Epistemic Confidence Tokens ─────────────────────────────────
    num_ect:           int   = 8
    ect_heads:         int   = 4       # Heads in ECT cross-attention
    ect_max:           int   = 16      # Max ECT tokens (dynamic spawning ceiling)
    ect_spawn_thresh:  float = 0.60    # U above this → spawn extra ECT + think budget

    # ── ACGI — Agentic Confidence-Gated Invocation ────────────────────────────
    acgi_threshold:    float = 0.72    # U above this → architecture-level tool gate

    # ── EPE — Epistemic Positional Encoding (novel) ───────────────────────────
    use_epe:           bool  = True
    epe_min_scale:     float = 0.7     # RoPE scale for confident tokens
    epe_max_scale:     float = 1.8     # RoPE scale for maximally uncertain tokens

    # ── YaRN context extension ────────────────────────────────────────────────
    use_yarn:          bool  = True
    yarn_scale:        float = 1.0     # 1.0 at train; 4.0 for 128k at inference

    # ── Think mode + PRM ──────────────────────────────────────────────────────
    use_prm:           bool  = True
    prm_hidden:        int   = 256
    use_ttip:          bool  = True    # Think-Tool Interleaving Protocol

    # ── MTP — Multi-Token Prediction (DeepSeek V3) ────────────────────────────
    use_mtp:           bool  = True
    mtp_n:             int   = 4       # Predict next N tokens simultaneously
    mtp_head_layers:   int   = 1       # Lightweight heads per future position

    # ── SAM — Structured Agentic Memory ───────────────────────────────────────
    use_sam:           bool  = True
    sam_memory_size:   int   = 32      # Agentic interaction slots

    # ── TDM — Temporal Diffusion Memory ───────────────────────────────────────
    tdm_memory_size:   int   = 64      # Rolling memory bank size
    tdm_conf_threshold: float = 0.20   # Tokens with U < this write to memory
    cmg_threshold:     float = 0.7     # Constitutional Memory Gate cutoff

    # ── Diffusion ─────────────────────────────────────────────────────────────
    diffusion_steps:   int   = 16
    uncertainty_thresh: float = 0.35   # Hard gate: uncertain → IDK / diffusion

    # ── Misc ──────────────────────────────────────────────────────────────────
    dropout:           float = 0.0
    weight_tying:      bool  = True

    # ── Loss weights (tunable without touching source code) ───────────────────
    loss_w_ect:        float = 0.10    # ECT Brier calibration
    loss_w_moe:        float = 0.01    # MoE load-balance
    loss_w_idk:        float = 0.05    # IDK token training
    loss_w_ses:        float = 0.005   # Spectral Expert Specialization
    loss_w_mtp:        float = 0.10    # Multi-Token Prediction auxiliary
    loss_w_prm:        float = 0.05    # Process Reward Model
    loss_w_acgi:       float = 0.10    # ACGI gate calibration
    loss_w_msra:       float = 0.05    # Multi-Step Reward Attribution proxy
    loss_w_fact:       float = 0.10    # FactualityDPO uncertainty bonus

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.max_seq_len % self.chunk_size == 0, (
            f"max_seq_len ({self.max_seq_len}) must be divisible by chunk_size ({self.chunk_size})"
        )
        assert self.head_dim == self.hidden_dim // self.num_heads, (
            f"head_dim ({self.head_dim}) must equal hidden_dim/num_heads "
            f"({self.hidden_dim}/{self.num_heads}={self.hidden_dim//self.num_heads})"
        )


# Module-level singleton — import `CFG` for a ready-to-use default config
CFG = LeoConfig()
