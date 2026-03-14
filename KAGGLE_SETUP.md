# LeoSLM "Aether" — Kaggle Setup Guide
### github.com/imanki-t/LeoSLM · branch: Leo-v2

---

## ⚡ Quick Start (TL;DR)

```
ONE NOTEBOOK (TPU v5e-8 — does everything):
  Cell 1 → clone repo + set tokens
  Cell 2 → install deps + verify TPU
  Cell 3 → prep full 2.1B token data (~5 hrs, fully unattended)
            ↳ writes tokens in chunks directly to disk (no RAM spike)
            ↳ skipped automatically if data already exists
  Cell 4 → smoke test (skipped if resuming)
  Cell 5 → full training  ← auto-saves every 200 steps + on crash/timeout
  Session 2+: run cells 1–5, data+checkpoints auto-restored in seconds
```

> **Why single notebook?**
> Data prep (`prep_data.py`) is CPU-bound. The TPU v5e-8 node has **96 GB CPU RAM**
> and idle CPU cores while XLA isn't running — it handles prep perfectly.
> Keeping everything here means **30 TPU hours = 30 training hours**.
> No GPU quota consumed, no dataset upload dance, no two tabs to manage.

---

## Step 1 — One-Time Setup

### 1.1 — Create Kaggle account & verify phone
1. Go to **https://kaggle.com** → Sign up (free)
2. Click your avatar → **Settings → Phone Verification**
3. Verify your number — **this unlocks TPU access, cannot be skipped**
4. You get **30 hrs TPU v5e-8 per week** for free, no credit card needed

### 1.2 — Get your HuggingFace token
1. Go to **https://huggingface.co/settings/tokens**
2. Click **New token** → name it `kaggle-leo` → **Read** permission → Copy it
3. Accept BigCode licence: **https://huggingface.co/datasets/bigcode/the-stack-smol**
   → Click "Access repository" → agree to terms

### 1.3 — Get your Kaggle API key
1. **kaggle.com → Settings → API → Create New Token**
2. Downloads `kaggle.json` — open it, copy `username` and `key`
3. You'll paste them into Cell 1 below

---

## THE NOTEBOOK — TPU v5e-8

### Create notebook (one time ever):
1. **https://kaggle.com/code** → New Notebook
2. Settings → **Accelerator: TPU v5e-8** → **Internet: On**
3. Paste cells 1–5, fill in your 4 tokens in Cell 1, save

---

### Cell 1 — Set All Tokens + Clone Repo

> ✏️ Fill in the 4 values at the top. That is the **only** thing you ever touch.
> Everything else runs itself.

```python
import subprocess, os, json, shutil
from pathlib import Path

# ================================================================
# PASTE YOUR VALUES HERE — touch nothing else
HF_TOKEN        = "hf_PASTE_YOUR_HF_TOKEN"
GITHUB_TOKEN    = ""            # only if repo is private, else leave ""
KAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY      = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# ================================================================

os.environ["HF_TOKEN"] = HF_TOKEN
print(f"HF_TOKEN     ({HF_TOKEN[:8]}...)")

os.makedirs("/root/.kaggle", exist_ok=True)
Path("/root/.kaggle/kaggle.json").write_text(
    json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY})
)
os.chmod("/root/.kaggle/kaggle.json", 0o600)
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
print(f"Kaggle API   ({KAGGLE_USERNAME})")

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
    print(f"✅ Cloned LeoSLM ({BRANCH}) — {len(list(Path('.').rglob('*.py')))} py files")
else:
    print("❌ Clone failed:", result.stderr)
    print("   Check: Internet ON in Settings? Repo URL correct?")
```

---

### Cell 2 — Install Dependencies + Verify TPU

```python
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install",
    "transformers>=4.40.0", "datasets>=2.18.0",
    "tokenizers>=0.19.0",   "huggingface_hub>=0.22.0",
    "scikit-learn>=1.3.0",  "-q"
], check=True)
print("✅ Dependencies installed")

try:
    import torch_xla.core.xla_model as xm
    devices = xm.get_xla_supported_devices()
    print(f"✅ TPU ready — {xm.xla_device()} | {len(devices)}/8 chips")
except ImportError:
    print("❌ torch_xla not found — set Accelerator to TPU v5e-8 in Settings")

import torch
print(f"✅ PyTorch {torch.__version__}")
```

---

### Cell 3 — Auto-Restore + Data Prep  *(fully automatic every session)*

> **Every session:** automatically downloads checkpoints and training data
> from your Kaggle datasets using the API credentials you set in Cell 1.
> No "Add Data" clicking needed — ever.
>
> **Session 1:** checkpoints dataset doesn't exist yet → skips download → runs
> full data prep (~5 hrs) → auto-creates both datasets for future sessions.
>
> **Session 2+:** downloads checkpoints (~30s) + skips data prep entirely.

```python
import shutil, os, numpy as np, subprocess, json, zipfile
from pathlib import Path

os.chdir("/kaggle/working/LeoSLM")
os.makedirs("data",          exist_ok=True)
os.makedirs("leo_tokenizer", exist_ok=True)
os.makedirs("checkpoints",   exist_ok=True)

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
TRAIN = Path("./data/train.npy")
VAL   = Path("./data/val.npy")

def kaggle_dataset_exists(dataset_slug):
    """Returns True if the dataset exists on Kaggle (i.e. was created in a prior session)."""
    r = subprocess.run(
        ["kaggle", "datasets", "status", dataset_slug],
        capture_output=True, text=True
    )
    return r.returncode == 0

def download_and_unzip(dataset_slug, dest_dir):
    """Download a Kaggle dataset zip and unzip it into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    r = subprocess.run(
        ["kaggle", "datasets", "download", dataset_slug,
         "--path", dest_dir, "--unzip"],
        capture_output=True, text=True
    )
    return r.returncode == 0

# ── Auto-download checkpoints (session 2+) ────────────────────────────────
CKPT_SLUG = f"{KAGGLE_USERNAME}/leo-checkpoints"
if kaggle_dataset_exists(CKPT_SLUG):
    print("📦 Downloading checkpoints from Kaggle...")
    ok = download_and_unzip(CKPT_SLUG, "./checkpoints")
    if ok:
        ckpts = list(Path("./checkpoints").glob("*.pt"))
        print(f"✅ Checkpoints restored ({len(ckpts)} files) — will auto-resume")
    else:
        print("⚠️  Checkpoint download failed — starting fresh this session")
else:
    print("ℹ️  No checkpoints dataset yet — session 1, starting fresh")

# ── Auto-download training data (session 2+) ──────────────────────────────
DATA_SLUG = f"{KAGGLE_USERNAME}/leo-training-data"
if not TRAIN.exists() and kaggle_dataset_exists(DATA_SLUG):
    print("📦 Downloading training data from Kaggle (~8 GB, ~2 min)...")
    tmp = "/kaggle/working/leo_data_dl"
    ok  = download_and_unzip(DATA_SLUG, tmp)
    if ok:
        for f in ["train.npy", "val.npy"]:
            src = Path(f"{tmp}/data/{f}")
            if src.exists():
                shutil.copy(src, f"./data/{f}")
                print(f"   {f}  ({src.stat().st_size/1e9:.2f} GB)")
        tok_src = Path(f"{tmp}/leo_tokenizer")
        if tok_src.exists():
            shutil.copytree(str(tok_src), "./leo_tokenizer", dirs_exist_ok=True)
            print("   leo_tokenizer restored")
    else:
        print("⚠️  Data download failed — will re-run prep")

# ── Run prep if data still missing (session 1 only) ───────────────────────
if TRAIN.exists() and VAL.exists():
    train = np.load(str(TRAIN), mmap_mode="r")
    val   = np.load(str(VAL),   mmap_mode="r")
    print(f"\n✅ Data ready — skipping prep")
    print(f"   Train: {len(train):,} tokens ({len(train)*4/1e9:.2f} GB)")
    print(f"   Val:   {len(val):,} tokens   ({len(val)*4/1e6:.0f} MB)")
else:
    print("\n📦 Running full data prep — 2.1B tokens across 6 sources")
    print("   Tokens written in chunks to disk (no RAM bottleneck)")
    print("   Sources: FineWeb-Edu, FineMath, The Stack, OpenWebMath, Books, Wiki")
    print("   Estimated time: ~5 hrs (unattended — you can close this tab)\n")

    result = subprocess.run(["python3", "prep_data.py"], env={**os.environ})

    if result.returncode == 0 and TRAIN.exists():
        train = np.load(str(TRAIN), mmap_mode="r")
        val   = np.load(str(VAL),   mmap_mode="r")
        print(f"\n✅ Data prep complete!")
        print(f"   Train: {len(train):,} tokens ({len(train)*4/1e9:.2f} GB)")
        print(f"   Val:   {len(val):,} tokens   ({len(val)*4/1e6:.0f} MB)")

        # ── Push training data to Kaggle so future sessions auto-download it ──
        print("\n📦 Pushing training data to Kaggle Dataset...")
        SAVE_DIR = "/kaggle/working/leo_data_save"
        os.makedirs(f"{SAVE_DIR}/data",          exist_ok=True)
        os.makedirs(f"{SAVE_DIR}/leo_tokenizer", exist_ok=True)
        for f in ["train.npy", "val.npy"]:
            src = Path(f"./data/{f}")
            if src.exists():
                shutil.copy(src, f"{SAVE_DIR}/data/{f}")
        if Path("./leo_tokenizer").exists():
            shutil.copytree("./leo_tokenizer", f"{SAVE_DIR}/leo_tokenizer",
                            dirs_exist_ok=True)
        Path(f"{SAVE_DIR}/dataset-metadata.json").write_text(json.dumps({
            "title": "leo-training-data",
            "id": DATA_SLUG,
            "licenses": [{"name": "CC0-1.0"}]
        }))
        r = subprocess.run(
            ["kaggle", "datasets", "create", "-p", SAVE_DIR, "--dir-mode", "zip"],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            subprocess.run(
                ["kaggle", "datasets", "version", "-p", SAVE_DIR,
                 "-m", "data prep complete", "--dir-mode", "zip"],
                capture_output=True, text=True
            )
        print(f"✅ leo-training-data pushed — future sessions auto-download it")
    else:
        print("❌ prep_data.py failed — scroll up for error")
```

---

### Cell 4 — Smoke Test  *(~2 min, auto-skipped if resuming)*

```python
import subprocess, os
from pathlib import Path

os.chdir("/kaggle/working/LeoSLM")

if Path("./checkpoints/latest.pt").exists():
    print("✅ Resuming from checkpoint — smoke test skipped")
else:
    print("Running 50-step smoke test...")
    result = subprocess.run(
        ["python3", "train.py", "--smoke", "--grad_accum", "1"],
        capture_output=True, text=True
    )
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("ERRORS:", result.stderr[-2000:])
        raise SystemExit("Fix errors before running Cell 5")
    print("✅ Smoke test passed — safe to start training")
```

---

### Cell 5 — Full Training  *(close tab and walk away)*

> Saves every 200 steps + every 30 min + on crash/timeout/exit.
> Pushes to `leo-checkpoints` Kaggle Dataset automatically every time.
> Next session: add `leo-checkpoints` via **Add Data**, run cells 1–5, done.

```python
import subprocess, os, shutil, signal, atexit, threading, time, json
from pathlib import Path

os.chdir("/kaggle/working/LeoSLM")

CKPT_DIR        = "/kaggle/working/LeoSLM/checkpoints"
SAVE_DIR        = "/kaggle/working/leo_checkpoint_save"
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
DATASET_NAME    = "leo-checkpoints"

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
        print(f"   ✅ Pushed → kaggle.com/datasets/{KAGGLE_USERNAME}/{DATASET_NAME}")
    else:
        c = subprocess.run(
            ["kaggle", "datasets", "create", "-p", SAVE_DIR, "--dir-mode", "zip"],
            capture_output=True, text=True
        )
        if c.returncode == 0:
            print(f"   ✅ Dataset created → kaggle.com/datasets/{KAGGLE_USERNAME}/{DATASET_NAME}")
        else:
            print(f"   ⚠️  Kaggle push failed: {r.stderr[:150]}")
            print(f"   Checkpoints still safe in Output tab → leo_checkpoint_save")

def emergency_save(reason="unknown"):
    print(f"\n💾 Saving ({reason})...")
    try:
        if Path(CKPT_DIR).exists():
            shutil.copytree(CKPT_DIR, f"{SAVE_DIR}/checkpoints", dirs_exist_ok=True)
            ckpts  = list(Path(f"{SAVE_DIR}/checkpoints").glob("*.pt"))
            latest = next((c for c in ckpts if c.name == "latest.pt"), None)
            size   = f"{latest.stat().st_size/1e6:.0f}MB" if latest else "?"
            print(f"   {len(ckpts)} checkpoints | latest.pt = {size}")
            push_to_kaggle(reason)
        else:
            print("   No checkpoints yet — nothing to save")
    except Exception as e:
        print(f"   Save error: {e}")

# 3-layer auto-save
atexit.register(emergency_save, reason="session_end_or_crash")

def _sig(sig, frame):
    emergency_save(reason="SIGTERM_timeout")
    exit(0)
signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT,  _sig)

def _bg():
    while True:
        time.sleep(30 * 60)
        emergency_save(reason="background_30min")
threading.Thread(target=_bg, daemon=True).start()

print("✅ Auto-save armed:")
print("   Every 200 steps  → latest.pt saved by train.py itself")
print("   Every 30 min     → background thread pushes to Kaggle")
print("   Crash / timeout  → atexit + SIGTERM pushes to Kaggle")
print(f"   Destination      → kaggle.com/datasets/{KAGGLE_USERNAME}/{DATASET_NAME}")
print("\n🔥 Training started. Close this tab whenever you want.\n")

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
print("\n🎉 All done! Final checkpoints pushed to Kaggle Dataset.")
```

---

### Session 2, 3, 4... — Identical to Session 1

1. Open the **same notebook** (it's still there, nothing changed)
2. Click **Run All**
3. Walk away

That's it. Cell 3 automatically downloads your latest checkpoints and training
data from Kaggle using your API credentials. No "Add Data" clicking.
No Output tab. No manual uploads. Nothing.

---

## Full Timeline

| Task | Accelerator | Time | Attended? |
|---|---|---|---|
| Data prep (session 1 only) | TPU v5e-8 CPU cores | ~5 hrs | ❌ unattended |
| Auto-save data to Kaggle Dataset | TPU v5e-8 | ~5 min | ✅ auto (Cell 3) |
| Training Phase 1–2 | TPU v5e-8 (all 8 chips) | ~7 hrs | ❌ unattended |
| Training Phase 3–4 | TPU v5e-8 (all 8 chips) | ~9 hrs | ❌ unattended |
| Training Phase 5–6 | TPU v5e-8 (all 8 chips) | ~9 hrs | ❌ unattended |
| Training Phase 7–8 | TPU v5e-8 (all 8 chips) | ~9 hrs | ❌ unattended |

**Total TPU v5e-8 hours:** ~35–40 hrs across ~5 sessions (~30 free/week → ~2 weeks)
**GPU T4 hours used:** 0
**Your total effort:** fill in 4 tokens once → click Run All at the start of each session

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `git clone failed` | Internet must be **ON** in Settings |
| `Repository not found` | Repo is private — paste GitHub token in Cell 1 |
| `No module named 'torch_xla'` | Accelerator must be **TPU v5e-8**, not GPU or None |
| `leo-training-data` download fails | Check Internet is ON in Settings. Cell 3 will fall back to re-running prep |
| `No space left on device` | `rm ./checkpoints/phase*_step*.pt` (keep `latest.pt` only) |
| `the-stack-smol` auth error | Accept licence at huggingface.co/datasets/bigcode/the-stack-smol |
| Checkpoints lost after session | Output tab → `leo_checkpoint_save` always has them |
| `Missing keys in checkpoint` | Safe if <50 keys — new params random init, continue |
| OOM / kernel restart during prep | Already fixed: tokens written in chunks to binary file + memmap split |

---

*LeoSLM Aether — Built by Unmuted*
