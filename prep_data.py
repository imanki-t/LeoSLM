"""
prep_data.py  —  LeoSLM Aether data preparation
================================================

Data budget: 2B tokens total (max that fits in 19.5GB Kaggle storage)
  tokens_tmp.bin peak + train.npy = 15.76 GB → fits in 19.5 GB with headroom

Sources (FULL mode, 2B total):
  fineweb_edu:   1200M  (60%)  high-quality educational web
  finemath:       300M  (15%)  math reasoning
  the_stack:      260M  (13%)  code
  open_web_math:  120M  ( 6%)  web math
  dolma_books:     80M  ( 4%)  long-form prose
  dolma_wiki:      40M  ( 2%)  factual reference

Tokenizer: 65k vocab (65529 base BPE + 14 specials = 65543 total)
  WHY NOT 256k: at 1B params, 256k embedding = 459M params = 48% of model budget.
  65k embedding = 117M params = 12% — leaves room for actual computation.
"""

import os, sys, time, argparse, math, subprocess
import numpy as np
from pathlib import Path

_HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if _HF_TOKEN:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_HF_TOKEN, add_to_git_credential=False)
        print(f"HuggingFace login OK (token: {_HF_TOKEN[:8]}...)")
    except Exception as _e:
        print(f"HuggingFace login failed: {_e}")
        _HF_TOKEN = ""
else:
    print("HF_TOKEN not set — gated datasets will use fallback.")

parser = argparse.ArgumentParser()
parser.add_argument("--tok-only", action="store_true")
parser.add_argument("--fast",     action="store_true")
parser.add_argument("--medium",   action="store_true")
args = parser.parse_args()

os.makedirs("data",          exist_ok=True)
os.makedirs("leo_tokenizer", exist_ok=True)

print("Installing dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--user",
     "transformers", "datasets", "tokenizers", "-q"],
    check=False,
)
print("Done.\n")

# ══════════════════════════════════════════════════════════════════════════════
# 65k TOKENIZER SPEC
# ══════════════════════════════════════════════════════════════════════════════
# BPE trains on 65529 IDs (0..65528)
# add_special_tokens() appends 14 specials starting at ID 65529
# Final vocab = 65543 exactly — matches model/config.py
SPECIAL_TOKENS_ORDERED = [
    "[PAD]",           # 65529  cfg.pad_id
    "[UNK]",           # 65530
    "[BOS]",           # 65531  cfg.bos_id
    "[EOS]",           # 65532  cfg.eos_id
    "[MASK]",          # 65533  cfg.mask_id
    "[IDK]",           # 65534  cfg.idk_id
    "<think>",         # 65535  cfg.think_start_id
    "</think>",        # 65536  cfg.think_end_id
    "<|tool_call|>",   # 65537  cfg.tool_call_start
    "<|/tool_call|>",  # 65538  cfg.tool_call_end
    "<|tool_result|>", # 65539  cfg.tool_result_start
    "<|/tool_result|>",# 65540  cfg.tool_result_end
    "<|system|>",      # 65541  cfg.system_start
    "<|/system|>",     # 65542
]

BASE_VOCAB_SIZE  = 65543 - len(SPECIAL_TOKENS_ORDERED)   # = 65529
FINAL_VOCAB_SIZE = 65543

assert BASE_VOCAB_SIZE  == 65529
assert FINAL_VOCAB_SIZE == 65543

EXPECTED_IDS = {sp: BASE_VOCAB_SIZE + i for i, sp in enumerate(SPECIAL_TOKENS_ORDERED)}

print("=" * 65)
print("STEP 1 — Training LeoTokenizer (65k vocab)")
print("=" * 65)
print(f"  Base vocab   : {BASE_VOCAB_SIZE:,} BPE tokens (IDs 0..{BASE_VOCAB_SIZE-1})")
print(f"  Special toks : {len(SPECIAL_TOKENS_ORDERED)} (appended at IDs {BASE_VOCAB_SIZE}..{FINAL_VOCAB_SIZE-1})")
print(f"  Final vocab  : {FINAL_VOCAB_SIZE:,}  ← matches model/config.py LeoConfig")
print()
for sp, tok_id in EXPECTED_IDS.items():
    print(f"    {sp:<20} → {tok_id}")
print()

TOK_SAVE_PATH = "./leo_tokenizer"
TOK_SAMPLE    = "./data/tok_sample.txt"


def build_tokenizer_sample():
    print("  Building tokenizer training sample (~500M chars)...")
    from datasets import load_dataset
    TARGET_CHARS = 500_000_000
    written = 0

    with open(TOK_SAMPLE, "w", encoding="utf-8") as f:
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-edu",
                              name="sample-10BT", split="train", streaming=True)
            for item in ds:
                text = item.get("text", "")
                if text:
                    f.write(text + "\n")
                    written += len(text)
                if written >= TARGET_CHARS * 0.5:
                    break
            print(f"    FineWeb-Edu: {written:,} chars")
        except Exception as e:
            print(f"    FineWeb-Edu failed: {e}")

        math_written = 0
        try:
            ds = load_dataset("HuggingFaceTB/finemath", name="finemath-3plus",
                              split="train", streaming=True)
            for item in ds:
                text = item.get("text", "")
                if text:
                    f.write(text + "\n")
                    math_written += len(text)
                if math_written >= TARGET_CHARS * 0.25:
                    break
            written += math_written
            print(f"    FineMath:    {math_written:,} chars")
        except Exception as e:
            print(f"    FineMath failed: {e}")

        code_written = 0
        try:
            ds = load_dataset("bigcode/the-stack-smol", split="train",
                              streaming=True,
                              token=_HF_TOKEN if _HF_TOKEN else None)
            for item in ds:
                text = item.get("content", item.get("text", ""))
                if text:
                    f.write(text + "\n")
                    code_written += len(text)
                if code_written >= TARGET_CHARS * 0.25:
                    break
            written += code_written
            print(f"    The Stack:   {code_written:,} chars")
        except Exception as e:
            print(f"    Stack failed: {e}")

    total_mb = os.path.getsize(TOK_SAMPLE) / 1e6
    print(f"  Sample file: {total_mb:.0f} MB")


def train_leotokenizer():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from tokenizers.pre_tokenizers import ByteLevel, Sequence as PreSeq, Digits

    print(f"  Training BPE — target {BASE_VOCAB_SIZE:,} base tokens...")
    t0 = time.time()

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = PreSeq([
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size                = BASE_VOCAB_SIZE,   # 65529 — NO specials in trainer
        min_frequency             = 2,
        special_tokens            = [],                # EMPTY — appended after via HF
        initial_alphabet          = list(pre_tokenizers.ByteLevel.alphabet()),
        show_progress             = True,
        continuing_subword_prefix = "Ġ",
    )

    if os.path.exists(TOK_SAMPLE) and os.path.getsize(TOK_SAMPLE) > 1_000_000:
        tok.train(files=[TOK_SAMPLE], trainer=trainer)
    else:
        print("  tok_sample.txt missing — using fallback text...")
        dummy = "the quick brown fox jumps over the lazy dog " * 10000
        tok.train_from_iterator([dummy], trainer=trainer)

    actual_base = len(tok.get_vocab())
    print(f"  Base BPE vocab: {actual_base:,} (target={BASE_VOCAB_SIZE:,})")

    tok.save(f"{TOK_SAVE_PATH}/tokenizer.json")

    from transformers import PreTrainedTokenizerFast

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file = f"{TOK_SAVE_PATH}/tokenizer.json",
        bos_token      = "[BOS]",
        eos_token      = "[EOS]",
        unk_token      = "[UNK]",
        pad_token      = "[PAD]",
        mask_token     = "[MASK]",
    )
    # Add all remaining specials in one call — sequential IDs guaranteed
    hf_tok.add_special_tokens({
        "additional_special_tokens": [
            sp for sp in SPECIAL_TOKENS_ORDERED
            if sp not in ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
        ]
    })
    hf_tok.save_pretrained(TOK_SAVE_PATH)

    # ── Verify every ID matches config.py ─────────────────────────────────────
    final_vocab_size = len(hf_tok)
    print(f"\n  Verification:")
    print(f"  Final vocab: {final_vocab_size:,} (expected {FINAL_VOCAB_SIZE:,})")
    assert final_vocab_size == FINAL_VOCAB_SIZE, (
        f"VOCAB MISMATCH: got {final_vocab_size}, expected {FINAL_VOCAB_SIZE}"
    )

    all_ok = True
    for sp, expected_id in EXPECTED_IDS.items():
        actual_id = hf_tok.convert_tokens_to_ids(sp)
        ok = (actual_id == expected_id)
        print(f"    {'✓' if ok else '✗'} {sp:<20} → {actual_id} (expected {expected_id})")
        if not ok:
            all_ok = False

    assert all_ok, "Special token ID mismatch — check output above."

    elapsed = time.time() - t0
    print(f"\n  LeoTokenizer (65k) trained in {elapsed:.0f}s → {TOK_SAVE_PATH}/")

    test_cases = [
        ("Hello world",                                  "basic English"),
        ("2+2=4, so 1999+1=2000",                        "arithmetic"),
        ("def fibonacci(n):",                             "Python code"),
        ("<think> let me reason step by step </think>",  "think tokens"),
        ("[IDK] I am not sure",                          "IDK token"),
    ]
    print("\n  Round-trip checks:")
    for text, desc in test_cases:
        ids  = hf_tok.encode(text)
        back = hf_tok.decode(ids)
        ok   = back.strip() == text.strip()
        print(f"    {'OK' if ok else 'FAIL'}  {desc}: {len(ids)} tokens")

    return hf_tok


# Train or load tokenizer
if not Path(f"{TOK_SAVE_PATH}/tokenizer.json").exists() or args.tok_only:
    if not Path(TOK_SAMPLE).exists() or os.path.getsize(TOK_SAMPLE) < 1_000_000:
        build_tokenizer_sample()
    tok = train_leotokenizer()
else:
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(TOK_SAVE_PATH)
    if len(tok) != FINAL_VOCAB_SIZE:
        print(f"  Loaded tokenizer wrong size: {len(tok):,} != {FINAL_VOCAB_SIZE:,}")
        print("  Retraining...")
        if not Path(TOK_SAMPLE).exists() or os.path.getsize(TOK_SAMPLE) < 1_000_000:
            build_tokenizer_sample()
        tok = train_leotokenizer()
    else:
        print(f"  LeoTokenizer (65k) loaded (vocab={len(tok):,})")

if args.tok_only:
    print("\nTokenizer training complete!")
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Tokenise datasets
# ══════════════════════════════════════════════════════════════════════════════

from datasets import load_dataset

assert len(tok) == FINAL_VOCAB_SIZE, f"Vocab mismatch: {len(tok)} != {FINAL_VOCAB_SIZE}"
assert len(tok) <= 2**32, "Vocab too large for uint32"

if args.fast:
    MODE    = "FAST"
    sources = {
        "fineweb_edu":   60_000_000,
        "finemath":      15_000_000,
        "the_stack":     13_000_000,
        "open_web_math":  6_000_000,
        "dolma_books":    4_000_000,
        "dolma_wiki":     2_000_000,
    }
elif args.medium:
    MODE    = "MEDIUM"
    sources = {
        "fineweb_edu":  600_000_000,
        "finemath":     150_000_000,
        "the_stack":    130_000_000,
        "open_web_math": 60_000_000,
        "dolma_books":   40_000_000,
        "dolma_wiki":    20_000_000,
    }
else:
    # FULL — 2B tokens total
    # Disk budget: tmp_bin(8GB) + train.npy(7.76GB) peak = 15.76GB < 19.5GB ✓
    MODE    = "FULL"
    sources = {
        "fineweb_edu":   1_200_000_000,  # 60% — high-quality educational web
        "finemath":        300_000_000,  # 15% — math reasoning
        "the_stack":       260_000_000,  # 13% — code
        "open_web_math":   120_000_000,  #  6% — web math
        "dolma_books":      80_000_000,  #  4% — long-form prose
        "dolma_wiki":       40_000_000,  #  2% — factual reference
    }

total_target = sum(sources.values())
print("\n" + "=" * 65)
print(f"STEP 2 — Tokenising {total_target:,} tokens ({MODE} mode)")
print(f"  Vocab:   {len(tok):,} | uint32 storage: {total_target*4/1e9:.1f} GB")
print("=" * 65)
for name, budget in sources.items():
    print(f"  {name:<20}: {budget/1e6:>7.0f}M  ({budget/total_target*100:.0f}%)")
print()

BOS_ID = tok.convert_tokens_to_ids("[BOS]")
EOS_ID = tok.convert_tokens_to_ids("[EOS]")
print(f"  BOS={BOS_ID} (expect 65531) | EOS={EOS_ID} (expect 65532)")
assert BOS_ID == 65531, f"BOS_ID wrong: {BOS_ID}"
assert EOS_ID == 65532, f"EOS_ID wrong: {EOS_ID}"

TMP_BIN      = "data/tokens_tmp.bin"
CHUNK_TOKENS = 250_000


def stream_tokens_to_bin(ds, max_tokens, source_name, bin_f,
                         max_doc_length=8192,
                         text_fields=("text", "content", "code", "passage")):
    chunk   = []
    written = 0
    n_docs  = 0
    t_start = time.time()
    last_report = 0

    def _flush():
        nonlocal chunk, written
        if not chunk:
            return
        np.array(chunk, dtype=np.uint32).tofile(bin_f)
        written += len(chunk)
        chunk = []

    for item in ds:
        text = ""
        for field in text_fields:
            if field in item and item[field]:
                text = str(item[field])
                break
        if not text or len(text) < 10:
            continue

        ids = tok.encode(text, truncation=True, max_length=max_doc_length,
                         add_special_tokens=False)
        if not ids:
            continue

        full      = [BOS_ID] + ids + [EOS_ID]
        remaining = max_tokens - (written + len(chunk))
        if len(full) >= remaining:
            chunk.extend(full[:remaining])
            _flush()
            elapsed = time.time() - t_start
            print(f"   {source_name}: {written:>14,} tokens | {n_docs:>7,} docs | {elapsed:.0f}s")
            return written

        chunk.extend(full)
        n_docs += 1

        if len(chunk) >= CHUNK_TOKENS:
            _flush()

        total_so_far = written + len(chunk)
        if total_so_far - last_report >= 50_000:
            pct     = total_so_far / max_tokens * 100
            elapsed = time.time() - t_start
            tps     = max(total_so_far / elapsed, 1)
            eta     = (max_tokens - total_so_far) / tps
            print(f"   {source_name}: {total_so_far:>14,} / {max_tokens:,} "
                  f"({pct:5.1f}%) | {n_docs:>7,} docs "
                  f"| {tps/1000:.0f}k t/s | ETA {eta/60:.0f}m")
            last_report = total_so_far

    _flush()
    print(f"   {source_name}: only {written:,} / {max_tokens:,} available")
    return written


total_written = 0
_bin_f = open(TMP_BIN, "wb")

print("=" * 65)
print("SOURCE 1/6 — FineWeb-Edu")
print("=" * 65)
try:
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["fineweb_edu"], "fineweb-edu", _bin_f, 8192)
except Exception as e:
    print(f"   {e} — CC-MAIN fallback...")
    ds = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["fineweb_edu"], "fineweb-fallback", _bin_f, 8192)
print(f"   Running total: {total_written:,}\n")

print("=" * 65)
print("SOURCE 2/6 — FineMath")
print("=" * 65)
try:
    ds = load_dataset("HuggingFaceTB/finemath", name="finemath-3plus",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["finemath"], "finemath", _bin_f, 4096)
except Exception as e:
    print(f"   {e} — OpenWebMath fallback...")
    try:
        ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        total_written += stream_tokens_to_bin(ds, sources["finemath"], "owm-backup", _bin_f, 4096)
    except Exception as e2:
        print(f"   {e2} — proof-pile fallback...")
        ds = load_dataset("EleutherAI/proof-pile", split="train", streaming=True)
        total_written += stream_tokens_to_bin(ds, sources["finemath"], "proof-pile", _bin_f, 4096)
print(f"   Running total: {total_written:,}\n")

print("=" * 65)
print("SOURCE 3/6 — The Stack (code)")
print("=" * 65)
stack_written = 0
if _HF_TOKEN:
    try:
        ds = load_dataset("bigcode/the-stack-smol", split="train",
                          streaming=True, token=_HF_TOKEN)
        stack_written = stream_tokens_to_bin(ds, sources["the_stack"], "the-stack", _bin_f, 4096,
                                             text_fields=("content", "text"))
    except Exception as e:
        print(f"   the-stack-smol failed: {e}")

if not stack_written:
    try:
        ds = load_dataset("HuggingFaceTB/smollm-corpus", name="python-edu",
                          split="train", streaming=True)
        stack_written = stream_tokens_to_bin(ds, sources["the_stack"], "python-edu", _bin_f, 4096)
    except Exception as e:
        print(f"   python-edu failed: {e}")

if not stack_written:
    try:
        ds = load_dataset("codeparrot/github-code", streaming=True, split="train")
        stack_written = stream_tokens_to_bin(ds, sources["the_stack"], "codeparrot", _bin_f, 4096,
                                             text_fields=("code", "content", "text"))
    except Exception as e:
        print(f"   codeparrot failed: {e}")

if not stack_written:
    print("   All code sources failed — FineWeb-Edu padding...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    stack_written = stream_tokens_to_bin(ds, sources["the_stack"], "code-fallback", _bin_f, 8192)
total_written += stack_written
print(f"   Running total: {total_written:,}\n")

print("=" * 65)
print("SOURCE 4/6 — OpenWebMath")
print("=" * 65)
try:
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["open_web_math"], "open-web-math", _bin_f, 4096)
except Exception as e:
    print(f"   {e} — FineWeb-Edu extra...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["open_web_math"], "fineweb-extra", _bin_f, 8192)
print(f"   Running total: {total_written:,}\n")

print("=" * 65)
print("SOURCE 5/6 — Books")
print("=" * 65)
books_written = 0
for name, kwargs in [
    ("dolma-books",  {"path": "allenai/dolma",           "name": "books",  "split": "train", "streaming": True}),
    ("gutenberg",    {"path": "sedthh/gutenberg_english",                   "split": "train", "streaming": True}),
    ("pile-books",   {"path": "EleutherAI/pile",          "name": "all",    "split": "train", "streaming": True}),
]:
    try:
        ds = load_dataset(**kwargs)
        books_written = stream_tokens_to_bin(ds, sources["dolma_books"], name, _bin_f, 32768)
        break
    except Exception as e:
        print(f"   {name} failed: {e}")
if not books_written:
    print("   All book sources failed — FineWeb-Edu padding...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    books_written = stream_tokens_to_bin(ds, sources["dolma_books"], "books-fallback", _bin_f, 8192)
total_written += books_written
print(f"   Running total: {total_written:,}\n")

print("=" * 65)
print("SOURCE 6/6 — Wikipedia")
print("=" * 65)
wiki_written = 0
for name, kwargs in [
    ("dolma-wiki",   {"path": "allenai/dolma",        "name": "wiki",        "split": "train", "streaming": True}),
    ("wikipedia-en", {"path": "wikimedia/wikipedia",  "name": "20231101.en", "split": "train", "streaming": True}),
    ("fineweb-wiki", {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "streaming": True}),
]:
    try:
        ds = load_dataset(**kwargs)
        wiki_written = stream_tokens_to_bin(ds, sources["dolma_wiki"], name, _bin_f, 4096,
                                            text_fields=("text", "passage", "content"))
        break
    except Exception as e:
        print(f"   {name} failed: {e}")
total_written += wiki_written
print(f"   Running total: {total_written:,}\n")

_bin_f.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Save train/val split as uint32 .npy files
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(f"STEP 3 — Saving {total_written:,} tokens as uint32")
print(f"  Storage: {total_written*4/1e9:.2f} GB")
print("=" * 65)

mm = np.memmap(TMP_BIN, dtype=np.uint32, mode="r", shape=(total_written,))

split_idx  = int(total_written * 0.97)
n_train    = split_idx
n_val      = total_written - split_idx
SLICE_SIZE = 10_000_000

print(f"  Writing train.npy  ({n_train:,} tokens)...")
train_mm = np.lib.format.open_memmap(
    "data/train.npy", mode="w+", dtype=np.uint32, shape=(n_train,)
)
for start in range(0, n_train, SLICE_SIZE):
    end = min(start + SLICE_SIZE, n_train)
    train_mm[start:end] = mm[start:end]
    if (start // SLICE_SIZE) % 5 == 0:
        print(f"    train: {end:>12,} / {n_train:,}  ({end/n_train*100:.1f}%)")
del train_mm

print(f"  Writing val.npy  ({n_val:,} tokens)...")
val_mm = np.lib.format.open_memmap(
    "data/val.npy", mode="w+", dtype=np.uint32, shape=(n_val,)
)
for start in range(0, n_val, SLICE_SIZE):
    end = min(start + SLICE_SIZE, n_val)
    val_mm[start:end] = mm[split_idx + start : split_idx + end]
del val_mm
del mm

# ── Delete tokens_tmp.bin IMMEDIATELY — frees ~8GB ────────────────────────────
print(f"\n  Deleting {TMP_BIN} to free disk space...")
try:
    os.remove(TMP_BIN)
    print(f"  ✓ Deleted {TMP_BIN}  ({total_written*4/1e9:.1f} GB freed)")
except Exception as e:
    print(f"  Warning: could not delete {TMP_BIN}: {e}")

# Also delete tok_sample.txt (no longer needed after tokenizer is trained)
if Path("./data/tok_sample.txt").exists():
    size_mb = Path("./data/tok_sample.txt").stat().st_size / 1e6
    Path("./data/tok_sample.txt").unlink()
    print(f"  ✓ Deleted tok_sample.txt ({size_mb:.0f} MB freed)")

train_gb = os.path.getsize("data/train.npy") / 1e9
val_mb   = os.path.getsize("data/val.npy")   / 1e6

import shutil
_, _, free = shutil.disk_usage("/kaggle/working")

print(f"\nDONE!")
print(f"  Train:      {n_train:>12,} tokens | {train_gb:.2f} GB")
print(f"  Val:        {n_val:>12,} tokens | {val_mb:.0f} MB")
print(f"  Disk free:  {free/1e9:.1f} GB remaining")
print(f"  Tokenizer:  LeoTokenizer 65k → ./leo_tokenizer/")
print(f"\n  Special token IDs (matches model/config.py LeoConfig):")
from transformers import PreTrainedTokenizerFast
_tok = PreTrainedTokenizerFast.from_pretrained("./leo_tokenizer")
for sp in SPECIAL_TOKENS_ORDERED:
    print(f"    {sp:<20} = {_tok.convert_tokens_to_ids(sp)}")
print(f"\n  Token/param ratio: {total_written/1e9:.1f}B / 1.0B = {total_written/1e9:.1f}x")
print(f"  At 32k context: {n_train//32768:,} full-length windows")
print(f"\nLeo Aether data ready! Run: python3 train.py")
