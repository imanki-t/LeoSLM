# LeoSLM "Aether" — Kaggle Setup Guide
### github.com/imanki-t/LeoSLM · branch: Leo-v2

---

## ⚡ Quick Start (TL;DR)

```
NOTEBOOK 1 (GPU T4 — saves TPU hours):
  Cell 1 → clone repo + set tokens
  Cell 2 → install deps
  Cell 3 → prep full 2.1B token data (~5 hrs)  ← runs unattended, auto-saves
  Cell 4 → save data to Kaggle Dataset

NOTEBOOK 2 (TPU v5e-8 — actual training):
  Cell 1 → clone repo + set tokens
  Cell 2 → install deps
  Cell 3 → restore saved data (skip prep entirely)
  Cell 4 → smoke test
  Cell 5 → full training  ← auto-saves every 200 steps + on crash/timeout
  Cell 6 → restore in next session
```

---

## Step 1 — One-Time Setup

### 1.1 — Create Kaggle account & verify phone
1. Go to **https://kaggle.com** → Sign up (free)
2. Click your avatar → **Settings → Phone Verification**
3. Verify your number — **this unlocks TPU/GPU access, cannot be skipped**
4. You get **30 hrs TPU v5e-8 + 30 hrs GPU T4 per week** for free

### 1.2 — Get your HuggingFace token
1. Go to **https://huggingface.co/settings/tokens**
2. Click **New token** → name it `kaggle-leo` → **Read** permission → Copy it
3. Accept BigCode licence: **https://huggingface.co/datasets/bigcode/the-stack-smol**
   → Click "Access repository" → agree to terms

---

## Why Two Notebooks?

| | Notebook 1 (Data Prep) | Notebook 2 (Training) |
|---|---|---|
| Accelerator | **GPU T4** | **TPU v5e-8** |
| Why | prep_data is CPU-bound, wastes TPU hours | Training needs all 8 TPU chips |
| Time | ~5 hrs (unattended) | ~8 hrs/session |
| Hours used | GPU quota (separate!) | TPU quota |

This way your **30 TPU hours are 100% spent on training**, not data prep.

---

## NOTEBOOK 1 — Data Prep (GPU T4)

### Create notebook:
1. **https://kaggle.com/code** → New Notebook
2. Settings → **Accelerator: GPU T4 x1** → **Internet: On**

---

### N1 Cell 1 — Clone Repo + Set Tokens

> ✏️ Paste your actual HF token on line 3 before running.

```python
import subprocess, os

HF_TOKEN     = "hf_PASTE_YOUR_TOKEN_HERE"
GITHUB_TOKEN = ""    # leave "" if repo is public

os.environ["HF_TOKEN"] = HF_TOKEN
print(f"✅ HF_TOKEN set ({HF_TOKEN[:8]}...)")

BRANCH   = "Leo-v2"
DST      = "/kaggle/working/LeoSLM"
REPO_URL = "https://github.com/imanki-t/LeoSLM.git"

if GITHUB_TOKEN:
    REPO_URL = REPO_URL.replace("https://", f"https://{GITHUB_TOKEN}@")

if os.path.exists(DST):
    subprocess.run(["rm", "-rf", DST])

result = subprocess.run(
    ["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, DST],
    capture_output=True, text=True
)

if result.returncode == 0:
    os.chdir(DST)
    print(f"✅ Cloned LeoSLM ({BRANCH}) → {DST}")
else:
    print("❌ Clone failed:", result.stderr)
```

---

### N1 Cell 2 — Install Dependencies

```python
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install",
    "transformers>=4.40.0",
    "datasets>=2.18.0",
    "tokenizers>=0.19.0",
    "huggingface_hub>=0.22.0",
    "scikit-learn>=1.3.0",
    "-q"
], check=True)
print("✅ Dependencies installed")
```

---

### N1 Cell 3 — Run Full Data Prep  *(~5 hrs, fully unattended)*

> Start this and walk away. It auto-saves progress.

```python
import subprocess, os, pathlib, numpy as np

os.chdir("/kaggle/working/LeoSLM")
os.makedirs("data", exist_ok=True)

TRAIN = pathlib.Path("./data/train.npy")
VAL   = pathlib.Path("./data/val.npy")

if TRAIN.exists() and VAL.exists():
    train = np.load(str(TRAIN), mmap_mode="r")
    val   = np.load(str(VAL),   mmap_mode="r")
    print(f"✅ Data already exists — skipping prep")
    print(f"   Train: {len(train):,} tokens ({len(train)*4/1e9:.2f} GB)")
    print(f"   Val:   {len(val):,} tokens ({len(val)*4/1e6:.0f} MB)")
else:
    print("📦 Running full data prep — 2.1B tokens across 6 sources")
    print("   Sources: FineWeb-Edu, FineMath, The Stack, OpenWebMath, Dolma Books, Dolma Wiki")
    print("   Estimated time: ~5 hrs on GPU T4")
    print("   You can close this tab — kernel keeps running\n")

    result = subprocess.run(
        ["python3", "prep_data.py"],
        env={**os.environ}
    )

    if result.returncode == 0 and TRAIN.exists():
        train = np.load(str(TRAIN), mmap_mode="r")
        val   = np.load(str(VAL),   mmap_mode="r")
        print(f"\n✅ Data prep complete!")
        print(f"   Train: {len(train):,} tokens ({len(train)*4/1e9:.2f} GB)")
        print(f"   Val:   {len(val):,} tokens ({len(val)*4/1e6:.0f} MB)")
    else:
        print("❌ prep_data.py failed — scroll up to see error")
```

---

### N1 Cell 4 — Save Data as Kaggle Dataset

> Run this after Cell 3 finishes. Creates a persistent dataset you reuse every session.

```python
import subprocess, os, shutil
from pathlib import Path

os.chdir("/kaggle/working/LeoSLM")

SAVE_DIR = "/kaggle/working/leo_data_save"
os.makedirs(f"{SAVE_DIR}/data",          exist_ok=True)
os.makedirs(f"{SAVE_DIR}/leo_tokenizer", exist_ok=True)

# Save train/val data
for f in ["train.npy", "val.npy"]:
    src = Path(f"./data/{f}")
    if src.exists():
        shutil.copy(src, f"{SAVE_DIR}/data/{f}")
        print(f"✅ {f}  ({src.stat().st_size/1e9:.2f} GB)")

# Save tokenizer
if Path("./leo_tokenizer").exists():
    shutil.copytree("./leo_tokenizer", f"{SAVE_DIR}/leo_tokenizer", dirs_exist_ok=True)
    print("✅ leo_tokenizer saved")

# Write dataset metadata
with open(f"{SAVE_DIR}/dataset-metadata.json", "w") as meta:
    meta.write('{"title": "leo-training-data", "id": "YOUR_KAGGLE_USERNAME/leo-training-data", "licenses": [{"name": "CC0-1.0"}]}')

print(f"\n📦 Saved to {SAVE_DIR}")
print("\nNow go to: Output tab (right sidebar) → leo_data_save → Download")
print("Then: kaggle.com/datasets → New Dataset → Upload that folder")
print("Name it: leo-training-data")
print("Once uploaded, it will be at: /kaggle/input/leo-training-data/ in any notebook")
```

---

## NOTEBOOK 2 — Training (TPU v5e-8)

### Create notebook:
1. **https://kaggle.com/code** → New Notebook
2. Settings → **Accelerator: TPU v5e-8** → **Internet: On**
3. **Add Data** (right sidebar) → search `leo-training-data` → Add

> ✏️ Get your Kaggle API key first:
> **kaggle.com → Settings → API → Create New Token** → downloads `kaggle.json`
> Open that file — copy the `username` and `key` into Cell 1 below.

---

### N2 Cell 1 — Set All Tokens + Clone Repo

> Fill in the 4 values at the top. That is the only thing you ever touch. Everything else runs itself.

```python
import subprocess, os, json, shutil
from pathlib import Path

# ================================================================
# PASTE YOUR VALUES HERE
HF_TOKEN        = "hf_PASTE_YOUR_HF_TOKEN"
GITHUB_TOKEN    = ""            # only if repo is private, else leave ""
KAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY      = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# ================================================================

# HuggingFace token
os.environ["HF_TOKEN"] = HF_TOKEN
print(f"HF_TOKEN     ({HF_TOKEN[:8]}...)")

# Kaggle API credentials
os.makedirs("/root/.kaggle", exist_ok=True)
Path("/root/.kaggle/kaggle.json").write_text(
    json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY})
)
os.chmod("/root/.kaggle/kaggle.json", 0o600)
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
print(f"Kaggle API   ({KAGGLE_USERNAME})")

# Clone repo
BRANCH   = "Leo-v2"
DST      = "/kaggle/working/LeoSLM"
REPO_URL = "https://github.com/imanki-t/LeoSLM.git"
if GITHUB_TOKEN:
    REPO_URL = REPO_URL.replace("https://", f"https://{GITHUB_TOKEN}@")

if os.path.exists(DST):
    shutil.rmtree(DST)

result = subprocess.run(
    ["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, DST],
    capture_output=True, text=True
)
if result.returncode == 0:
    os.chdir(DST)
    print(f"Cloned LeoSLM ({BRANCH}) — {len(list(Path('.').rglob('*.py')))} py files")
else:
    print("Clone failed:", result.stderr)
    print("Check: Internet ON in Settings?")
```

---

### N2 Cell 2 — Install Dependencies + Verify TPU

```python
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install",
    "transformers>=4.40.0", "datasets>=2.18.0",
    "tokenizers>=0.19.0",   "huggingface_hub>=0.22.0",
    "scikit-learn>=1.3.0",  "-q"
], check=True)

try:
    import torch_xla.core.xla_model as xm
    devices = xm.get_xla_supported_devices()
    print(f"TPU ready — {xm.xla_device()} | {len(devices)}/8 chips")
except ImportError:
    print("torch_xla not found — set Accelerator to TPU v5e-8")

import torch
print(f"PyTorch {torch.__version__}")
```

---

### N2 Cell 3 — Restore Data + Checkpoints

```python
import shutil, os, numpy as np
from pathlib import Path

os.chdir("/kaggle/working/LeoSLM")
os.makedirs("data",          exist_ok=True)
os.makedirs("leo_tokenizer", exist_ok=True)

# Restore training data
DATA_SRC = "/kaggle/input/leo-training-data"
for f in ["train.npy", "val.npy"]:
    src, dst = Path(f"{DATA_SRC}/data/{f}"), Path(f"./data/{f}")
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"{f} restored ({src.stat().st_size/1e9:.2f} GB)")
    elif dst.exists():
        print(f"{f} already present")
    else:
        print(f"ERROR: {f} not found — add leo-training-data via Add Data")

if Path(f"{DATA_SRC}/leo_tokenizer").exists():
    shutil.copytree(f"{DATA_SRC}/leo_tokenizer", "./leo_tokenizer", dirs_exist_ok=True)
    print("Tokenizer restored")

# Restore checkpoints if resuming from previous session
CKPT_SRC = "/kaggle/input/leo-checkpoints"
if Path(CKPT_SRC).exists():
    shutil.copytree(CKPT_SRC, "./checkpoints", dirs_exist_ok=True)
    ckpts = list(Path("./checkpoints").glob("*.pt"))
    print(f"Checkpoints restored ({len(ckpts)} files) — will auto-resume")
else:
    print("No prior checkpoints — starting fresh (session 1 only)")

train = np.load("./data/train.npy", mmap_mode="r")
val   = np.load("./data/val.npy",   mmap_mode="r")
print(f"Train: {len(train):,} tokens ({len(train)*4/1e9:.2f} GB)")
print(f"Val:   {len(val):,} tokens ({len(val)*4/1e6:.0f} MB)")
```

---

### N2 Cell 4 — Smoke Test *(~2 min, auto-skipped on resume)*

```python
import subprocess, os
from pathlib import Path

os.chdir("/kaggle/working/LeoSLM")

if Path("./checkpoints/latest.pt").exists():
    print("Resuming from checkpoint — smoke test skipped")
else:
    result = subprocess.run(
        ["python3", "train.py", "--smoke", "--grad_accum", "1"],
        capture_output=True, text=True
    )
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("ERRORS:", result.stderr[-2000:])
        raise SystemExit("Fix errors before running Cell 5")
    print("Smoke test passed!")
```

---

### N2 Cell 5 — Full Training (close tab and walk away)

> Saves every 200 steps + every 30 min + on crash/timeout/exit.
> Pushes to your Kaggle Dataset automatically every time.
> Next session: just add `leo-checkpoints` via Add Data, run cells 1-5, done.

```python
import subprocess, os, shutil, signal, atexit, threading, time, json
from pathlib import Path

os.chdir("/kaggle/working/LeoSLM")

CKPT_DIR        = "/kaggle/working/LeoSLM/checkpoints"
SAVE_DIR        = "/kaggle/working/leo_checkpoint_save"
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
DATASET_NAME    = "leo-checkpoints"

# Create dataset metadata for first push
os.makedirs(SAVE_DIR, exist_ok=True)
meta = Path(f"{SAVE_DIR}/dataset-metadata.json")
if not meta.exists():
    meta.write_text(json.dumps({
        "title": DATASET_NAME,
        "id": f"{KAGGLE_USERNAME}/{DATASET_NAME}",
        "licenses": [{"name": "CC0-1.0"}]
    }))

def push_to_kaggle(reason):
    r = subprocess.run(
        ["kaggle", "datasets", "version", "-p", SAVE_DIR,
         "-m", f"auto: {reason}", "--dir-mode", "zip"],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print(f"   Pushed to kaggle.com/datasets/{KAGGLE_USERNAME}/{DATASET_NAME}")
    else:
        # First time — dataset does not exist yet, create it
        c = subprocess.run(
            ["kaggle", "datasets", "create", "-p", SAVE_DIR, "--dir-mode", "zip"],
            capture_output=True, text=True
        )
        if c.returncode == 0:
            print(f"   Dataset created at kaggle.com/datasets/{KAGGLE_USERNAME}/{DATASET_NAME}")
        else:
            print(f"   Kaggle push failed: {r.stderr[:150]}")
            print(f"   Checkpoints still safe in Output tab as backup")

def emergency_save(reason="unknown"):
    print(f"
  Saving ({reason})...")
    try:
        if Path(CKPT_DIR).exists():
            shutil.copytree(CKPT_DIR, f"{SAVE_DIR}/checkpoints", dirs_exist_ok=True)
            ckpts  = list(Path(f"{SAVE_DIR}/checkpoints").glob("*.pt"))
            latest = next((c for c in ckpts if c.name == "latest.pt"), None)
            size   = f"{latest.stat().st_size/1e6:.0f}MB" if latest else "?"
            print(f"   {len(ckpts)} checkpoints | latest.pt = {size}")
            push_to_kaggle(reason)
        else:
            print("   No checkpoints yet")
    except Exception as e:
        print(f"   Save error: {e}")

# Layer 1: on any crash/exit/OOM
atexit.register(emergency_save, reason="session_end_or_crash")

# Layer 2: SIGTERM (Kaggle timeout signal)
def _sig(sig, frame):
    emergency_save(reason="SIGTERM_timeout")
    exit(0)
signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT,  _sig)

# Layer 3: background thread every 30 min
def _bg():
    while True:
        time.sleep(30 * 60)
        emergency_save(reason="background_30min")
threading.Thread(target=_bg, daemon=True).start()

print("Auto-save armed:")
print("  Every 200 steps  -> latest.pt saved by train.py")
print("  Every 30 min     -> background thread pushes to Kaggle")
print("  Crash/timeout    -> atexit + SIGTERM pushes to Kaggle")
print(f"  Destination      -> kaggle.com/datasets/{KAGGLE_USERNAME}/{DATASET_NAME}")
print("
Training started. Close this tab whenever you want.
")

has_ckpt = Path(f"{CKPT_DIR}/latest.pt").exists()

subprocess.run([
    "python3", "train.py",
    *(["--resume"] if has_ckpt else []),
    "--grad_accum", "16",
    "--save_every",  "200",
    "--ckpt_dir",    CKPT_DIR,
    "--train_data",  "./data/train.npy",
], env={**os.environ})

emergency_save(reason="training_complete")
print("
All done! Final checkpoints pushed to Kaggle Dataset.")
```

---

### N2 Session 2, 3, 4... — Exactly the same steps

1. Go to your new notebook → **Add Data** → search `leo-checkpoints` → Add
2. Run **Cell 1** (paste tokens again — same values)
3. Run **Cell 2** (install deps)
4. Run **Cell 3** (restores data + checkpoints automatically)
5. Run **Cell 4** (auto-skipped since checkpoint exists)
6. Run **Cell 5** (auto-detects `latest.pt`, passes `--resume`, trains, pushes)

No manual download, no manual upload, no version buttons, nothing extra.


## Full Timeline Across Both Notebooks

| | Notebook | Accelerator | Time | Attended? |
|---|---|---|---|---|
| Data prep | Notebook 1 | GPU T4 | ~5 hrs | ❌ unattended |
| Save data | Notebook 1 | GPU T4 | ~5 min | ✅ once |
| Training Phase 1–2 | Notebook 2 | TPU v5e-8 | ~7 hrs | ❌ unattended |
| Training Phase 3–4 | Notebook 2 | TPU v5e-8 | ~9 hrs | ❌ unattended |
| Training Phase 5–6 | Notebook 2 | TPU v5e-8 | ~9 hrs | ❌ unattended |
| Training Phase 7–8 | Notebook 2 | TPU v5e-8 | ~9 hrs | ❌ unattended |

**Total GPU T4 used:** ~5 hrs (out of 30 free)
**Total TPU v5e-8 used:** ~28–36 hrs (across 4 sessions of 9 hrs)

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `git clone failed` | Internet must be **ON** in Settings |
| `Repository not found` | Repo is private — paste GitHub token in Cell 1 |
| `No module named 'torch_xla'` | Accelerator must be **TPU v5e-8** |
| `leo-training-data` not found | Add it via **Add Data** in notebook settings |
| `No space left on device` | `rm ./checkpoints/phase*_step*.pt` (keep `latest.pt` only) |
| `the-stack-smol` auth error | Accept licence at huggingface.co/datasets/bigcode/the-stack-smol |
| Checkpoints lost after session | Check Output tab — `leo_checkpoint_save` is always there |
| `Missing keys in checkpoint` | Safe if <50 keys — new params random init, continue |

---

*LeoSLM Aether — Built by Unmuted*
