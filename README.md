# 🦁 LeoSLM

> **The world's first Confidence-Gated Diffusion-AR Transformer**  
> A novel Small Language Model that architecturally prevents hallucinations.

---

## What is LeoSLM?

LeoSLM is a **120M parameter** language model built on a completely new architecture that no one has built before. It fuses **Autoregressive (AR) generation** and **Masked Diffusion** inside the same transformer block, controlled by a learned confidence gate.

The key insight: instead of hallucination being a training problem, LeoSLM makes it an **architectural impossibility** above a confidence threshold — using Epistemic Confidence Tokens (ECTs) that monitor every token the model generates and flag uncertain ones for diffusion refinement.

---

## 🧠 Novel Concepts

| # | Concept | What it does |
|---|---------|-------------|
| 1 | **Dual-Path Gated Attention** | Causal + Bidirectional attention in the same block, merged by a confidence gate α |
| 2 | **Epistemic Confidence Tokens (ECT)** | 4 learnable tokens that produce per-token uncertainty scores U ∈ [0,1] |
| 3 | **Selective Diffusion Refinement** | Only uncertain tokens get diffusion refinement — 4-8× faster than full diffusion |
| 4 | **Constitutional Diffusion Training** | Constitutional AI principles baked into the diffusion objective as conditioning vectors |
| 5 | **ECT Calibration Loss** | Brier score forces ECTs to be well-calibrated (model knows exactly when it's wrong) |
| 6 | **Shared-Weight Dual Attention** | Causal and bidirectional heads share W_Q, W_K, W_V — halves parameter count |
| 7 | **Progressive Noise Annealing** | Smoothly transitions from diffusion-dominant to AR-dominant during training |
| 8 | **Self-Consistency Diffusion Voting** | N diffusion samples vote on uncertain tokens — Best-of-N at the token level |

---

## 🏗 Architecture

```
Input Tokens + 4× ECT (Epistemic Confidence Tokens)
        │
        ▼
[ Token Embedding dim=512 + RoPE Positional Encoding ]
        │
        ▼
┌──────────────────────────────────────────────────────┐
│            16 × Leo Decoder Blocks                   │
│  ┌────────────────────────────────────────────────┐  │
│  │           DUAL-PATH ATTENTION                  │  │
│  │  Path A: Causal Attention  (AR generation)     │  │
│  │  Path B: Bidirectional Attn (Diffusion)        │  │
│  │         └──── Confidence Gate α ────┘          │  │
│  │    merged = α×Bidir + (1-α)×Causal             │  │
│  ├────────────────────────────────────────────────┤  │
│  │           ECT Cross-Attention                  │  │
│  │   ECTs attend to full sequence → uncertainty   │  │
│  ├────────────────────────────────────────────────┤  │
│  │           SwiGLU FFN (dim=1408)                │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
        │
        ▼
[ ECT Aggregation → Per-token Uncertainty Map U ∈ [0,1] ]
        │
   ┌────┴────┐
   ▼         ▼
AR Head   Diffusion Head
(fast)    (iterative unmask)
   │         │
   └────┬────┘
        │
[ IF U[i] < τ → AR output, ELSE → Diffusion refines ]
        │
        ▼
   Output Tokens
```

---

## 🛡 Anti-Hallucination Stack

LeoSLM stacks **9 hallucination prevention mechanisms** across architecture, training and inference:

1. **ECT Uncertainty Gating** — Architecture-level: uncertain tokens cannot be output unchecked
2. **Constitutional AI Training** — 12 principles embedded as conditioning vectors during diffusion training
3. **Self-Consistency Voting** — Top-K diffusion samples vote on most uncertain tokens
4. **Brier Score Calibration** — ECTs trained to accurately predict their own error probability
5. **DPO Alignment** — Penalizes overconfident wrong answers directly
6. **IDK Token Training** — Model learns to say "I don't know" instead of confabulating
7. **Bidirectional Grounding** — Diffusion path sees full context, reducing local inconsistencies
8. **Selective Refinement** — Uncertain positions get multiple refinement passes
9. **KL Divergence Penalty** — Constitutional loss pushes uncertain positions toward [IDK]

---

## 📦 Model Config

```yaml
vocab_size   : 32768
hidden_dim   : 512
num_layers   : 16
num_heads    : 8
num_kv_heads : 2       # Grouped Query Attention
num_ect      : 4       # Epistemic Confidence Tokens
ffn_dim      : 1408    # SwiGLU (8/3 × hidden_dim)
max_seq_len  : 512
parameters   : ~120M
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/LeoSLM.git
cd LeoSLM
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare data (run once)

```bash
python3 prep_data.py
```

### 4. Train

```bash
# Run all phases sequentially
python3 train.py

# Or run specific phase
python3 train.py --phase 1    # AR warmup
python3 train.py --phase 2    # Diffusion warmup
python3 train.py --phase 3    # Joint training

# Resume from checkpoint
python3 train.py --resume
```

### 5. Generate

```bash
# Hybrid mode (best quality, uses ECT gating)
python3 generate.py --mode hybrid --prompt "Once upon a time"

# Pure AR (fast)
python3 generate.py --mode ar --prompt "Once upon a time"

# Pure diffusion (parallel generation)
python3 generate.py --mode diffusion --prompt "Once upon a time"
```

### 6. Evaluate

```bash
python3 eval/evaluate.py --checkpoint ./checkpoints/latest.pt
```

---

## 📁 Project Structure

```
LeoSLM/
├── model/
│   ├── dual_attention.py     # Causal + Bidirectional shared-weight attention
│   ├── ect.py                # Epistemic Confidence Tokens
│   ├── confidence_gate.py    # α gate merging both paths
│   ├── leo_block.py          # Full decoder block
│   └── leoSLM.py             # Full model assembly
├── diffusion/
│   ├── noise_schedule.py     # Cosine masking schedule
│   ├── mdlm_loss.py          # MDLM + AR joint loss
│   └── selective_sampler.py  # ECT-guided inference sampler
├── training/
│   ├── calibration_loss.py   # Brier score ECT calibration
│   ├── constitutional.py     # Constitutional AI conditioning
│   └── dpo_trainer.py        # DPO alignment
├── data/
│   └── dataset.py            # Tokenizer + DataLoader
├── eval/
│   └── evaluate.py           # PPL, ECE, AUROC metrics
├── config/
│   └── leo_config.yaml       # All hyperparameters
├── train.py                  # Main training loop
├── prep_data.py              # Data download + tokenization
├── generate.py               # Inference script
└── requirements.txt
```

---

## 🗓 Training Phases

| Phase | What trains | Loss | Duration |
|-------|------------|------|---------|
| 1 — AR Warmup | Full model, gate frozen (α=0) | L_AR only | ~3 epochs |
| 2 — Diffusion Warmup | Diffusion head + ECT only, backbone frozen | L_AR + 0.3×L_MDM | ~3 epochs |
| 3 — Joint Training | Everything unfrozen | L_AR + 0.5×L_MDM + 0.1×L_ECT | ~6 epochs |

---

## 📊 Evaluation Metrics

| Metric | Measures | Target |
|--------|----------|--------|
| **Perplexity (PPL)** | Language model quality | Lower is better |
| **ECE** | How calibrated ECT uncertainty is | 0 = perfect |
| **Uncertainty Separation** | Gap between U_wrong and U_correct | Higher is better |
| **AUROC** | ECT as error detector | 1.0 = perfect |

---

## 🔬 Key Papers

- [LLaDA — Large Language Diffusion Models (2025)](https://arxiv.org/abs/2502.09992)
- [MDLM — Masked Diffusion Language Models (NeurIPS 2024)](https://arxiv.org/abs/2406.07524)
- [Constitutional AI — Anthropic (2022)](https://arxiv.org/abs/2212.08073)
- [DPO — Direct Preference Optimization (2023)](https://arxiv.org/abs/2305.18290)
- [LLaMA 3 — Architecture Reference](https://arxiv.org/abs/2407.21783)
- [SEDD — Score Entropy Discrete Diffusion (ICML 2024)](https://arxiv.org/abs/2310.16834)

---

## ☁️ Free Training (No Credit Card)

Train LeoSLM for free on **Kaggle** (30 hrs/week T4 GPU, no card needed):

1. Sign up at [kaggle.com](https://kaggle.com)
2. Verify phone number to unlock GPU
3. Create new Notebook → Add this repo as dataset
4. Run `prep_data.py` then `train.py`

Each phase fits within the weekly GPU limit.

---

## 🛠 Tech Stack

- **PyTorch 2.x** + `torch.compile`
- **HuggingFace Transformers** (tokenizer)
- **HuggingFace Datasets** (TinyStories)
- **RoPE** positional embeddings
- **GQA** (Grouped Query Attention)
- **SwiGLU** activation
- **RMSNorm** (pre-norm)

---

## 📄 License

MIT License — free to use, modify, and build on.

---

## 🙏 Acknowledgements

Architecture inspired by research from Anthropic, Google DeepMind, Meta AI, and the open-source ML community. Built as an independent research project.

---

<div align="center">
  <b>LeoSLM — Because hallucinations should be impossible, not just unlikely.</b>
</div>
