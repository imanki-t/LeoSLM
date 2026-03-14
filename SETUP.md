# LeoSLM "Aether" — Setup Guide
### Built by Unmuted

---

## Step 1 — What to DELETE from your existing repo

These files are dead. Nothing imports from them anymore. Delete them all.

```
# Stale v1 model files
model/leoSLM.py
model/dual_attention.py
model/ect.py
model/confidence_gate.py
model/leo_block.py

# Stale training files (superseded by training/loss.py)
training/calibration_loss.py
training/constitutional.py
training/dpo_trainer.py

# Stale diffusion folder (logic merged into model/ and generate.py)
diffusion/noise_schedule.py
diffusion/mdlm_loss.py
diffusion/selective_sampler.py
```

> **Do not delete** `config/leo_config.yaml` — it is now documentation.
> **Do not delete** `leo_rag.py`, `prep_data.py` — those are unchanged.

---

## Step 2 — Final folder structure (what you should have)

```
LeoSLM/
│
├── model/                         ← All architecture lives here
│   ├── __init__.py                ← Public API (import from here)
│   ├── config.py                  ← LeoConfig — single source of truth
│   ├── identity.py                ← LEO_IDENTITY + LEO_SYSTEM_PROMPT
│   ├── norm.py                    ← RMSNorm
│   ├── rope.py                    ← YaRN RoPE helpers
│   ├── ect.py                     ← ECTv3Module (Epistemic Confidence Tokens)
│   ├── memory.py                  ← TemporalDiffusionMemory + StructuredAgenticMemory
│   ├── attention.py               ← EPE + MultiHeadLatentAttention + DSALite
│   ├── moe.py                     ← UWMR MoE (ExpertFFN, UWMRMoE)
│   ├── mtp.py                     ← MultiTokenPredictionHead
│   ├── agentic.py                 ← ACGI (Agentic Confidence-Gated Invocation)
│   ├── leo_block.py               ← LeoBlock (Aether decoder block)
│   └── leo_slm.py                 ← LeoSLM — full model assembly
│
├── training/                      ← All loss/RL logic lives here
│   ├── __init__.py
│   ├── loss.py                    ← LeoLoss (all loss terms, no magic numbers)
│   ├── grpo.py                    ← GRPOTrainer + AgenticGRPO
│   └── dpo.py                     ← FactualityDPO
│
├── data/                          ← Dataset
│   ├── __init__.py
│   └── dataset.py                 ← LeoDataset (memory-mapped, curriculum-length)
│
├── eval/                          ← Evaluation
│   ├── __init__.py
│   └── evaluate.py                ← PPL + ECE + AUROC metrics
│
├── config/
│   └── leo_config.yaml            ← Documentation mirror (not loaded by code)
│
├── train.py                       ← Training orchestration ONLY (530 lines)
├── generate.py                    ← Inference script
├── prep_data.py                   ← Data download + tokenization (unchanged)
├── leo_rag.py                     ← RAG knowledge layer (unchanged)
└── requirements.txt
```

---

## Step 3 — What files to ADD (all from the refactor)

Add these files to your repo exactly as delivered. Each column shows the
file and where it goes.

### `model/` — 13 files total

| File | Add to |
|---|---|
| `config.py` | `model/config.py` |
| `identity.py` | `model/identity.py` |
| `norm.py` | `model/norm.py` |
| `rope.py` | `model/rope.py` |
| `ect.py` | `model/ect.py` |
| `memory.py` | `model/memory.py` |
| `attention.py` | `model/attention.py` |
| `moe.py` | `model/moe.py` |
| `mtp.py` | `model/mtp.py` |
| `agentic.py` | `model/agentic.py` |
| `leo_block.py` | `model/leo_block.py` |
| `leo_slm.py` | `model/leo_slm.py` |
| `__init__.py` | `model/__init__.py` |

### `training/` — 4 files total

| File | Add to |
|---|---|
| `loss.py` | `training/loss.py` |
| `grpo.py` | `training/grpo.py` |
| `dpo.py` | `training/dpo.py` |
| `__init__.py` | `training/__init__.py` |

### `data/` — 2 files total

| File | Add to |
|---|---|
| `dataset.py` | `data/dataset.py` |
| `__init__.py` | `data/__init__.py` |

### `eval/` — 2 files total

| File | Add to |
|---|---|
| `evaluate.py` | `eval/evaluate.py` |
| `__init__.py` | `eval/__init__.py` |

### Root — replace 3 files

| File | Action |
|---|---|
| `train.py` | **Replace** existing file entirely |
| `generate.py` | **Replace** existing file entirely |
| `requirements.txt` | **Replace** existing file entirely |

---

## Step 4 — Quick sanity check (run locally before Kaggle)

After adding all files, run this one-liner to confirm everything parses:

```bash
python3 -c "
import ast, pathlib
files = list(pathlib.Path('.').rglob('*.py'))
[ast.parse(f.read_text()) for f in files]
print(f'OK — {len(files)} files pass syntax check')
"
```

Expected output: `OK — 23 files pass syntax check`

---

## Step 5 — Running on Kaggle TPU v5e-8 (step by step)

### 5.1 — Create a Kaggle account and verify

1. Go to [kaggle.com](https://kaggle.com) and sign up
2. Go to **Settings → Phone Verification** and verify your phone number
   - This unlocks GPU/TPU access — you cannot skip this step
3. You get **30 hours of TPU v5e-8 per week** for free, no credit card needed

---

### 5.2 — Upload the repo to Kaggle

**Option A: GitHub (recommended)**

1. Push your repo to GitHub (public or private)
2. On Kaggle → **Datasets → New Dataset**
3. Click **Link GitHub Repository** → select your repo
4. This auto-syncs — every push updates the dataset

**Option B: Upload ZIP**

1. ZIP your entire `LeoSLM/` folder
2. On Kaggle → **Datasets → New Dataset → Upload**
3. Upload the ZIP — Kaggle extracts it automatically

---

### 5.3 — Create the Kaggle Notebook

1. Go to **Code → New Notebook**
2. Click **Settings** (right sidebar) → **Accelerator → TPU v5e-8**
3. Click **Settings → Internet → On** (needed for pip installs)
4. Click **Add Data** (right sidebar) → find your LeoSLM dataset → add it

Your dataset will appear at `/kaggle/input/leoslm/` inside the notebook.

---

### 5.4 — Cell 1: Environment setup

Paste this as the first cell and run it:

```python
import subprocess, sys, os

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install",
    "transformers>=4.40.0",
    "datasets>=2.18.0",
    "tokenizers>=0.19.0",
    "scikit-learn>=1.3.0",
    "-q"
], check=True)

# torch_xla is pre-installed on Kaggle TPU — verify it
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    print(f"✅ torch_xla: {torch_xla.__version__}")
    print(f"   TPU device: {xm.xla_device()}")
except ImportError:
    print("⚠️  torch_xla not found — check Accelerator setting is TPU v5e-8")

import torch
print(f"✅ PyTorch: {torch.__version__}")
```

---

### 5.5 — Cell 2: Copy repo to working directory

```python
import shutil, os

# Copy your dataset into the working directory
# (Kaggle datasets are read-only, so we work from /kaggle/working/)
SRC = "/kaggle/input/leoslm"   # adjust if your dataset has a different name
DST = "/kaggle/working/LeoSLM"

if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)

os.chdir(DST)
print(f"✅ Working directory: {os.getcwd()}")
print(f"   Files: {len(list(__import__('pathlib').Path('.').rglob('*.py')))} Python files")
```

---

### 5.6 — Cell 3: Prepare training data

```python
# Downloads FineWeb-Edu, FineMath, OpenWebMath and tokenises them.
# Run once — creates data/train.npy and data/val.npy (~8–15 minutes on TPU).
# On subsequent runs this cell is skipped if the files already exist.

import pathlib
if not pathlib.Path("./data/train.npy").exists():
    print("Preparing data...")
    import subprocess
    subprocess.run(["python3", "prep_data.py"], check=True)
else:
    import numpy as np
    tokens = np.load("./data/train.npy", mmap_mode="r")
    print(f"✅ Data ready: {len(tokens):,} tokens ({len(tokens)*4/1e9:.1f} GB)")
```

---

### 5.7 — Cell 4: Smoke test (50 steps, confirm everything works)

**Always run this before a full training run.** It catches config or import
errors in about 2 minutes.

```python
import subprocess
result = subprocess.run(
    ["python3", "train.py", "--smoke", "--grad_accum", "1"],
    capture_output=True, text=True
)
print(result.stdout[-3000:])   # Last 3000 chars
if result.returncode != 0:
    print("ERRORS:")
    print(result.stderr[-2000:])
else:
    print("✅ Smoke test passed — safe to run full training")
```

Expected output ends with:
```
  🔥 Smoke test complete (50 steps)
✅ Smoke test passed — safe to run full training
```

---

### 5.8 — Cell 5: Full training (all 8 phases)

```python
import subprocess

result = subprocess.run(
    [
        "python3", "train.py",
        "--grad_accum", "16",    # effective batch = 16 × 8 chips = 128
        "--save_every",  "200",
        "--ckpt_dir",    "./checkpoints",
        "--train_data",  "./data/train.npy",
    ],
    capture_output=False,   # live stdout in notebook
)
```

**Estimated time per phase on TPU v5e-8:**

| Phase | Context | Duration |
|---|---|---|
| 1 — AR warmup | 4k | ~3–4 hrs |
| 2 — Diffusion | 8k | ~3–4 hrs |
| 3 — MoE + ECT | 16k | ~4–5 hrs |
| 4 — SFT | 32k | ~4–5 hrs |
| 5 — DPO | 32k | ~2–3 hrs |
| 6 — GRPO | 32k | ~4–5 hrs |
| 7 — Agentic SFT | 32k | ~4–5 hrs |
| 8 — Agentic RL | 32k | ~4–5 hrs |

Total: ~28–36 hours across multiple sessions.

---

### 5.9 — Running specific phases (across multiple sessions)

Each Kaggle session is max 12 hours. Split the training across sessions:

**Session 1 (phases 1–3):**
```python
# Run phases 1, 2, 3 sequentially
for phase in [1, 2, 3]:
    subprocess.run([
        "python3", "train.py",
        "--phase", str(phase),
        "--grad_accum", "16",
        "--ckpt_dir", "./checkpoints",
    ])
```

**Session 2+ (resume from checkpoint):**
```python
subprocess.run([
    "python3", "train.py",
    "--resume",                    # picks up from checkpoints/latest.pt
    "--grad_accum", "16",
    "--ckpt_dir", "./checkpoints",
])
```

---

### 5.10 — Save checkpoints between sessions

Kaggle deletes `/kaggle/working/` when a session ends. Save your checkpoints
to a Kaggle Dataset so they persist:

```python
# Run at the end of every session BEFORE it times out
from kaggle_secrets import UserSecretsClient
import os

# Option A: Save to a Kaggle Dataset (create one named "leo-checkpoints" first)
os.system("kaggle datasets version -p ./checkpoints -m 'phase N complete'")

# Option B: Download directly (simpler)
# Click the "Output" tab in your notebook → download checkpoints/latest.pt
```

**Recommended**: create a Kaggle Dataset named `leo-checkpoints` and version
it after each phase. Then load it at the start of the next session:

```python
# At the top of your next session:
shutil.copytree("/kaggle/input/leo-checkpoints", "./checkpoints")
# Then run with --resume
```

---

### 5.11 — Generate / inference after training

```python
# Test generation after training is complete
subprocess.run([
    "python3", "generate.py",
    "--prompt", "Explain how transformers work",
    "--mode", "think",
    "--checkpoint", "./checkpoints/latest.pt",
    "--show_thinking",
])
```

---

### 5.12 — Evaluate

```python
subprocess.run([
    "python3", "eval/evaluate.py",
    "--checkpoint",  "./checkpoints/latest.pt",
    "--data_path",   "./data/val.npy",
    "--max_batches", "100",
    "--output",      "./eval_results.json",
])
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch_xla'`**
→ Make sure **Accelerator** in notebook settings is set to **TPU v5e-8**, not GPU or None.

**`No space left on device`**
→ Kaggle working dir is limited. Delete intermediate checkpoints:
```bash
ls -lh ./checkpoints/
rm ./checkpoints/phase*_step*.pt   # keep only latest.pt and best_phase*.pt
```

**`Training data not found`**
→ Run Cell 3 (prep_data.py) first. If it fails, check internet is ON in settings.

**`Missing keys in checkpoint`**
→ Normal when resuming after adding new modules. New parameters get random init.
The warning is safe to ignore as long as the count is small (<50 keys).

**`FSDP error on single GPU`**
→ FSDP only activates on TPU. On GPU/CPU it runs without sharding automatically.

**Out of memory on GPU**
→ Reduce `--grad_accum` to 4 or lower batch size in `data/dataset.py`.

---

## Key config knobs

Everything is in `model/config.py`. The most useful things to change:

```python
# To train a smaller model (faster iteration)
hidden_dim:    int = 1024   # was 2560
num_layers:    int = 16     # was 32
moe_experts:   int = 4      # was 8

# To adjust loss balance
loss_w_ect:    float = 0.05   # reduce ECT calibration weight
loss_w_mtp:    float = 0.20   # increase MTP auxiliary signal

# To extend context at inference (no retraining needed)
# Pass --yarn 4.0 to generate.py → 128k context via YaRN
yarn_scale:    float = 1.0    # 1.0 = 32k, 4.0 = 128k
```

---

*Leo Aether — built by Unmuted*
