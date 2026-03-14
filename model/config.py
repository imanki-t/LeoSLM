from dataclasses import dataclass
from typing import Optional


@dataclass
class LeoConfig:
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

    max_seq_len:       int   = 32768
    chunk_size:        int   = 2048
    think_budget_max:  int   = 8192
    think_budget_min:  int   = 256

    hidden_dim:        int   = 2560
    num_layers:        int   = 32
    num_dense_layers:  int   = 4
    num_heads:         int   = 20
    head_dim:          int   = 128

    use_mla:           bool  = True
    mla_c_kv:          int   = 512
    mla_c_q:           int   = 768
    mla_rope_dim:      int   = 64

    sliding_window:    int   = 4096
    global_every_n:    int   = 4

    use_dsa:           bool  = True
    dsa_threshold:     int   = 32768
    dsa_top_k_pct:     float = 0.25

    num_kv_heads:      int   = 4

    ffn_dim_dense:     int   = 6912
    ffn_dim_expert:    int   = 1024

    moe_experts:       int   = 8
    moe_top_k:         int   = 2
    moe_shared:        int   = 1
    moe_load_coeff:    float = 0.01
    uwmr_spec_scale:   float = 2.0
    uwmr_gen_scale:    float = 1.0

    num_ect:           int   = 8
    ect_heads:         int   = 4
    ect_max:           int   = 16
    ect_spawn_thresh:  float = 0.60

    acgi_threshold:    float = 0.72

    use_epe:           bool  = True
    epe_min_scale:     float = 0.7
    epe_max_scale:     float = 1.8

    use_yarn:          bool  = True
    yarn_scale:        float = 1.0

    use_prm:           bool  = True
    prm_hidden:        int   = 256
    use_ttip:          bool  = True

    use_mtp:           bool  = True
    mtp_n:             int   = 4
    mtp_head_layers:   int   = 1

    use_sam:           bool  = True
    sam_memory_size:   int   = 32

    use_tdm:           bool  = True
    tdm_memory_size:   int   = 64
    tdm_conf_threshold: float = 0.20
    cmg_threshold:     float = 0.7

    use_cur:           bool  = True
    cur_max_passes:    int   = 3
    cur_threshold:     float = 0.05

    uncertainty_thresh: float = 0.50

    loss_w_ect:  float = 0.10
    loss_w_moe:  float = 0.01
    loss_w_idk:  float = 0.10
    loss_w_ses:  float = 0.01
    loss_w_mtp:  float = 0.30
    loss_w_prm:  float = 0.05
    loss_w_acgi: float = 0.10
    loss_w_msra: float = 0.05


CFG = LeoConfig()
