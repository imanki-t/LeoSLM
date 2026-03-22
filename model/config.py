"""
model/config.py  —  LeoSLM Aether  (~1B parameters)

Architecture changes from 3.1B original:
  hidden_dim:      2560  →  1792   (fits 14 heads × 128 head_dim exactly)
  num_layers:        32  →    24   (2 dense + 22 MoE)
  num_heads:         20  →    14
  num_dense_layers:   4  →     2
  mla_c_kv:         512  →   384
  mla_c_q:          768  →   512
  ffn_dim_dense:   6912  →  4864
  ffn_dim_expert:  1024  →   512
  tdm_memory_size:   64  →    32
  sam_memory_size:   32  →    16
  mtp_n:              4  →     3

Total params:  ~950M (~1B)
Checkpoint:    ~1.9 GB BF16
Vocab/IDs:     unchanged (65543, PAD=65529, EOS=65532, etc.)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LeoConfig:
    # ── Vocabulary + special tokens (unchanged) ───────────────────────────────
    vocab_size:        int   = 65543
    pad_id:            int   = 65529
    bos_id:            int   = 65531
    eos_id:            int   = 65532
    mask_id:           int   = 65533
    idk_id:            int   = 65534
    think_start_id:    int   = 65535
    think_end_id:      int   = 65536
    tool_call_start:   int   = 65537
    tool_call_end:     int   = 65538
    tool_result_start: int   = 65539
    tool_result_end:   int   = 65540
    system_start:      int   = 65541
    estr_domain_count: int   = 8

    # ── Sequence ──────────────────────────────────────────────────────────────
    max_seq_len:       int   = 32768
    chunk_size:        int   = 2048
    think_budget_max:  int   = 8192
    think_budget_min:  int   = 256

    # ── Core dimensions (REDUCED for ~1B) ────────────────────────────────────
    hidden_dim:        int   = 1792   # was 2560 — 14 heads × 128 = 1792 exactly
    num_layers:        int   = 24     # was 32
    num_dense_layers:  int   = 2      # was 4
    num_heads:         int   = 14     # was 20
    head_dim:          int   = 128    # unchanged — 1792/14 = 128 ✓

    # ── MLA (REDUCED) ────────────────────────────────────────────────────────
    use_mla:           bool  = True
    mla_c_kv:          int   = 384    # was 512
    mla_c_q:           int   = 512    # was 768
    mla_rope_dim:      int   = 64     # unchanged

    # ── Attention pattern ─────────────────────────────────────────────────────
    sliding_window:    int   = 4096   # unchanged
    global_every_n:    int   = 4      # unchanged (layers 3,7,11,... are global)
    use_dsa:           bool  = True
    dsa_threshold:     int   = 32768
    dsa_top_k_pct:     float = 0.25
    num_kv_heads:      int   = 4      # GQA fallback

    # ── FFN dimensions (REDUCED) ──────────────────────────────────────────────
    ffn_dim_dense:     int   = 4864   # was 6912
    ffn_dim_expert:    int   = 512    # was 1024

    # ── MoE ───────────────────────────────────────────────────────────────────
    moe_experts:       int   = 8      # unchanged
    moe_top_k:         int   = 2      # unchanged
    moe_shared:        int   = 1      # unchanged
    moe_load_coeff:    float = 0.01
    uwmr_spec_scale:   float = 2.0
    uwmr_gen_scale:    float = 1.0

    # ── ECT ───────────────────────────────────────────────────────────────────
    num_ect:           int   = 6      # was 8 (small saving)
    ect_heads:         int   = 4      # unchanged
    ect_max:           int   = 12     # was 16
    ect_spawn_thresh:  float = 0.60

    # ── ACGI ──────────────────────────────────────────────────────────────────
    acgi_threshold:    float = 0.72

    # ── EPE ───────────────────────────────────────────────────────────────────
    use_epe:           bool  = True
    epe_min_scale:     float = 0.7
    epe_max_scale:     float = 1.8

    # ── YaRN ──────────────────────────────────────────────────────────────────
    use_yarn:          bool  = True
    yarn_scale:        float = 1.0

    # ── PRM ───────────────────────────────────────────────────────────────────
    use_prm:           bool  = True
    prm_hidden:        int   = 128    # was 256 (small model → smaller PRM head)

    # ── TTIP ──────────────────────────────────────────────────────────────────
    use_ttip:          bool  = True

    # ── MTP (REDUCED) ────────────────────────────────────────────────────────
    use_mtp:           bool  = True
    mtp_n:             int   = 3      # was 4
    mtp_head_layers:   int   = 1

    # ── SAM (REDUCED) ────────────────────────────────────────────────────────
    use_sam:           bool  = True
    sam_memory_size:   int   = 16     # was 32

    # ── TDM (REDUCED) ────────────────────────────────────────────────────────
    use_tdm:           bool  = True
    tdm_memory_size:   int   = 32     # was 64
    tdm_conf_threshold: float = 0.20
    cmg_threshold:     float = 0.7

    # ── CUR ───────────────────────────────────────────────────────────────────
    use_cur:           bool  = True
    cur_max_passes:    int   = 3
    cur_threshold:     float = 0.05

    # ── Uncertainty ───────────────────────────────────────────────────────────
    uncertainty_thresh: float = 0.50

    # ── Gradient checkpointing ────────────────────────────────────────────────
    use_gradient_checkpointing: bool = False   # set True via --grad_ckpt flag

    # ── Loss weights ─────────────────────────────────────────────────────────
    loss_w_ect:  float = 0.10
    loss_w_moe:  float = 0.01
    loss_w_idk:  float = 0.10
    loss_w_ses:  float = 0.01
    loss_w_mtp:  float = 0.30
    loss_w_prm:  float = 0.05
    loss_w_acgi: float = 0.10
    loss_w_msra: float = 0.05


CFG = LeoConfig()
