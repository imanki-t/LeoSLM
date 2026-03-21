"""
prep_data.py — LeoSLM Aether: tokenizer training + dataset preparation

CRITICAL BUG FIXES vs original:
  BUG 1 — Special token IDs completely wrong:
    Original: BpeTrainer(vocab_size=65536, special_tokens=[...])
    → BPE assigns specials to IDs 0..12 (LOWEST ids)
    → But model/config.py expects them at IDs 65529..65541 (HIGHEST ids)
    → [PAD]=0 but cfg.pad_id=65529 → loss never ignores padding
    → [EOS]=3  but cfg.eos_id=65532 → generation NEVER stops
    → Every token in the stale dataset has wrong IDs → must regenerate

  FIX: Train BPE with vocab_size=65529 and NO special_tokens in trainer.
       Then call tokenizer.add_special_tokens() which APPENDS them at
       IDs 65529..65542 → final vocab_size = 65543 matching config exactly.

  BUG 2 — Vocab size mismatch:
    Original trained 65536 base + [BUDGET] added after = 65537.
    Model embedding expects 65543 → shape mismatch → crash on step 1.

  BUG 3 — [BUDGET] token not in BPE training data at all.
    Removed [BUDGET] (not referenced anywhere in model code).
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

# ── Special tokens — ORDER MATTERS: must match model/config.py exactly ────────
#
# model/config.py assigns:
#   pad_id=65529, unk_id=65530, bos_id=65531, eos_id=65532,
#   mask_id=65533, idk_id=65534, think_start=65535, think_end=65536,
#   tool_call_start=65537, tool_call_end=65538,
#   tool_result_start=65539, tool_result_end=65540,
#   system_start=65541
#   vocab_size=65543  (65529 base + 14 specials = 65543)
#
# So: base BPE vocab = 65529 tokens (IDs 0..65528)
#     add_special_tokens appends at 65529..65542 → total = 65543

SPECIAL_TOKENS_ORDERED = [
    "[PAD]",             # → 65529  (pad_id)
    "[UNK]",             # → 65530  (unk_id)
    "[BOS]",             # → 65531  (bos_id)
    "[EOS]",             # → 65532  (eos_id)
    "[MASK]",            # → 65533  (mask_id)
    "[IDK]",             # → 65534  (idk_id)
    "<think>",           # → 65535  (think_start_id)
    "</think>",          # → 65536  (think_end_id)
    "<|tool_call|>",     # → 65537  (tool_call_start)
    "<|/tool_call|>",    # → 65538  (tool_call_end)
    "<|tool_result|>",   # → 65539  (tool_result_start)
    "<|/tool_result|>",  # → 65540  (tool_result_end)
    "<|system|>",        # → 65541  (system_start)
    "<|/system|>",       # → 65542  (closing tag)
]  # 14 special tokens

BASE_VOCAB_SIZE  = 65529   # BPE trains exactly this many tokens (IDs 0..65528)
FINAL_VOCAB_SIZE = BASE_VOCAB_SIZE + len(SPECIAL_TOKENS_ORDERED)  # = 65543

assert FINAL_VOCAB_SIZE == 65543, f"Vocab mismatch: {FINAL_VOCAB_SIZE} != 65543"

print("=" * 65)
print("STEP 1 — Training LeoTokenizer")
print("=" * 65)
print(f"  Base vocab  : {BASE_VOCAB_SIZE:,} BPE tokens (IDs 0..{BASE_VOCAB_SIZE-1})")
print(f"  Special toks: {len(SPECIAL_TOKENS_ORDERED)} (appended at IDs {BASE_VOCAB_SIZE}..{FINAL_VOCAB_SIZE-1})")
print(f"  Final vocab : {FINAL_VOCAB_SIZE:,}  ← matches model/config.py exactly")
print()

TOK_SAVE_PATH = "./leo_tokenizer"
TOK_SAMPLE    = "./data/tok_sample.txt"


def build_tokenizer_sample():
    print("  Building tokenizer training sample (~500M chars)...")
    from datasets import load_dataset

    TARGET_CHARS = 500_000_000
    written = 0
    t0 = time.time()

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
    print(f"  Sample file: {total_mb:.0f} MB | {time.time()-t0:.0f}s")


def train_leotokenizer():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from tokenizers.pre_tokenizers import ByteLevel, Sequence as PreSeq, Digits

    print("  Training BPE tokenizer...")
    print(f"  Target: {BASE_VOCAB_SIZE:,} base tokens (specials added after)")
    t0 = time.time()

    tok = Tokenizer(models.BPE(unk_token=None))  # no unk in base BPE
    tok.pre_tokenizer = PreSeq([
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()

    # BUG FIX: NO special_tokens in BpeTrainer.
    # Special tokens must NOT be in the base BPE vocab — they are appended
    # after training at exact IDs 65529..65542 via add_special_tokens().
    # If you include them in BpeTrainer, they get IDs 0..13 which completely
    # breaks the model (cfg.pad_id=65529 would never match).
    trainer = trainers.BpeTrainer(
        vocab_size                = BASE_VOCAB_SIZE,   # 65529 — NO specials here
        min_frequency             = 2,
        special_tokens            = [],                # ← intentionally empty
        initial_alphabet          = list(pre_tokenizers.ByteLevel.alphabet()),
        show_progress             = True,
        continuing_subword_prefix = "Ġ",
    )

    if os.path.exists(TOK_SAMPLE) and os.path.getsize(TOK_SAMPLE) > 1_000_000:
        tok.train(files=[TOK_SAMPLE], trainer=trainer)
    else:
        print("  tok_sample.txt not found — using fallback text")
        dummy = "the quick brown fox jumps over the lazy dog " * 10000
        tok.train_from_iterator([dummy], trainer=trainer)

    actual_base_vocab = len(tok.get_vocab())
    print(f"  Base BPE vocab size: {actual_base_vocab:,} (target={BASE_VOCAB_SIZE:,})")

    tok.save(f"{TOK_SAVE_PATH}/tokenizer.json")

    # BUG FIX: use add_special_tokens (NOT special_tokens in PreTrainedTokenizerFast
    # constructor) so they are APPENDED at the END of the vocab.
    # This guarantees [PAD]=65529, [UNK]=65530, ... <|/system|>=65542.
    from transformers import PreTrainedTokenizerFast

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file = f"{TOK_SAVE_PATH}/tokenizer.json",
        # Do NOT specify special token names here yet — add them below
    )

    # add_special_tokens appends tokens that don't exist, giving them
    # IDs starting from current vocab_size (= BASE_VOCAB_SIZE = 65529).
    num_added = hf_tok.add_special_tokens({
        "pad_token":   "[PAD]",          # → 65529
        "unk_token":   "[UNK]",          # → 65530
        "bos_token":   "[BOS]",          # → 65531
        "eos_token":   "[EOS]",          # → 65532
        "mask_token":  "[MASK]",         # → 65533
        "additional_special_tokens": [
            "[IDK]",             # → 65534
            "<think>",           # → 65535
            "</think>",          # → 65536
            "<|tool_call|>",     # → 65537
            "<|/tool_call|>",    # → 65538
            "<|tool_result|>",   # → 65539
            "<|/tool_result|>",  # → 65540
            "<|system|>",        # → 65541
            "<|/system|>",       # → 65542
        ],
    })
    print(f"  Added {num_added} special tokens")

    final_vocab_size = len(hf_tok)
    print(f"  Final vocab size: {final_vocab_size:,} (expected {FINAL_VOCAB_SIZE:,})")
    assert final_vocab_size == FINAL_VOCAB_SIZE, (
        f"VOCAB SIZE MISMATCH: tokenizer={final_vocab_size} but model expects {FINAL_VOCAB_SIZE}.\n"
        f"Delete ./leo_tokenizer/ and re-run."
    )

    hf_tok.save_pretrained(TOK_SAVE_PATH)

    elapsed = time.time() - t0
    print(f"\n  LeoTokenizer trained in {elapsed:.0f}s → {TOK_SAVE_PATH}/")
    print(f"  Base vocab: {actual_base_vocab:,} | Special tokens: {num_added}")

    # ── Verify special token IDs match model/config.py ────────────────────────
    print("\n  Verifying special token IDs match model/config.py:")
    expected = {
        "[PAD]":          65529,
        "[UNK]":          65530,
        "[BOS]":          65531,
        "[EOS]":          65532,
        "[MASK]":         65533,
        "[IDK]":          65534,
        "<think>":        65535,
        "</think>":       65536,
        "<|tool_call|>":  65537,
        "<|/tool_call|>": 65538,
        "<|tool_result|>":65539,
        "<|/tool_result|>":65540,
        "<|system|>":     65541,
        "<|/system|>":    65542,
    }
    all_ok = True
    for token, expected_id in expected.items():
        actual_id = hf_tok.convert_tokens_to_ids(token)
        status = "✓" if actual_id == expected_id else "✗ MISMATCH"
        print(f"    {status}  {token:<22} → {actual_id}  (expected {expected_id})")
        if actual_id != expected_id:
            all_ok = False

    if not all_ok:
        raise ValueError(
            "Special token IDs don't match model/config.py!\n"
            "Delete ./leo_tokenizer/ and re-run prep_data.py"
        )

    # ── Round-trip sanity checks ──────────────────────────────────────────────
    test_cases = [
        ("Hello world",                                 "basic English"),
        ("2+2=4, so 1999+1=2000",                      "arithmetic (digit tokens)"),
        ("def fibonacci(n):",                           "Python code"),
        ("<think> let me reason step by step </think>", "think tokens"),
        ("[IDK] I am not sure",                         "IDK token"),
    ]
    print("\n  Round-trip sanity checks:")
    for text, desc in test_cases:
        ids  = hf_tok.encode(text)
        back = hf_tok.decode(ids)
        ok   = "✓" if back.strip() == text.strip() else "✗ MISMATCH"
        print(f"    {ok}  {desc}: {len(ids)} tokens")

    return hf_tok


# ── Train or load tokenizer ────────────────────────────────────────────────────
if not Path(f"{TOK_SAVE_PATH}/tokenizer.json").exists() or args.tok_only:
    if not Path(TOK_SAMPLE).exists() or os.path.getsize(TOK_SAMPLE) < 1_000_000:
        build_tokenizer_sample()
    tok = train_leotokenizer()
else:
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(TOK_SAVE_PATH)
    print(f"LeoTokenizer loaded (vocab={len(tok):,})")

    # Verify loaded tokenizer has correct vocab size
    if len(tok) != FINAL_VOCAB_SIZE:
        print(f"\n  WARNING: Loaded tokenizer has vocab={len(tok):,} "
              f"but model expects {FINAL_VOCAB_SIZE:,}")
        print(f"  Delete ./leo_tokenizer/ and re-run prep_data.py to fix this.")
        sys.exit(1)

if args.tok_only:
    print("\nTokenizer training complete!")
    print("Run full data prep with: python3 prep_data.py")
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Tokenise datasets
# ══════════════════════════════════════════════════════════════════════════════

from datasets import load_dataset

# Use len(tok) so it's always in sync (= 65543)
assert len(tok) <= 2**32, "Vocab too large for uint32"

if args.fast:
    MODE    = "FAST"
    sources = {
        "fineweb_edu":    80_000_000,
        "finemath":       40_000_000,
        "the_stack":      40_000_000,
        "open_web_math":  20_000_000,
        "dolma_books":    10_000_000,
        "dolma_wiki":     10_000_000,
    }
elif args.medium:
    MODE    = "MEDIUM"
    sources = {
        "fineweb_edu":   200_000_000,
        "finemath":      100_000_000,
        "the_stack":     100_000_000,
        "open_web_math":  60_000_000,
        "dolma_books":    20_000_000,
        "dolma_wiki":     20_000_000,
    }
else:
    MODE    = "FULL"
    sources = {
        "fineweb_edu":   700_000_000,
        "finemath":      300_000_000,
        "the_stack":     400_000_000,
        "open_web_math": 200_000_000,
        "dolma_books":   300_000_000,
        "dolma_wiki":    200_000_000,
    }

total_target = sum(sources.values())
print("\n" + "=" * 65)
print(f"STEP 2 — Tokenising {total_target:,} tokens ({MODE} mode)")
print(f"  Vocab: {len(tok):,} | Storage: uint32 ({total_target*4/1e9:.1f} GB)")
print("=" * 65)

BOS_ID = tok.convert_tokens_to_ids("[BOS]")
EOS_ID = tok.convert_tokens_to_ids("[EOS]")
print(f"  BOS_ID={BOS_ID} (expect 65531) | EOS_ID={EOS_ID} (expect 65532)")
assert BOS_ID == 65531, f"BOS_ID wrong: {BOS_ID}"
assert EOS_ID == 65532, f"EOS_ID wrong: {EOS_ID}"

TMP_BIN      = "data/tokens_tmp.bin"
CHUNK_TOKENS = 250_000


def stream_tokens_to_bin(ds, max_tokens, source_name, bin_f,
                         max_doc_length=8192,
                         text_fields=("text","content","code","passage")):
    chunk   = []
    written = 0
    n_docs  = 0
    t_start = time.time()
    last_report = 0

    def _flush():
        nonlocal chunk, written
        if not chunk: return
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
            pct = total_so_far / max_tokens * 100
            elapsed = time.time() - t_start
            tps = max(total_so_far / elapsed, 1)
            eta = (max_tokens - total_so_far) / tps
            print(f"   {source_name}: {total_so_far:>14,} / {max_tokens:,} "
                  f"({pct:5.1f}%) | {n_docs:>7,} docs | {tps/1000:.0f}k t/s | ETA {eta/60:.0f}m")
            last_report = total_so_far

    _flush()
    print(f"   {source_name}: only {written:,} / {max_tokens:,} available")
    return written


total_written = 0
_bin_f = open(TMP_BIN, "wb")

print("\n" + "=" * 65)
print("SOURCE 1/6 — FineWeb-Edu")
print("=" * 65)
try:
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["fineweb_edu"], "fineweb-edu", _bin_f, 8192)
except Exception as e:
    print(f"   Fallback: {e}")
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
    print(f"   Fallback: {e}")
    try:
        ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        total_written += stream_tokens_to_bin(ds, sources["finemath"], "owm-backup", _bin_f, 4096)
    except Exception as e2:
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
                                             text_fields=("content","text"))
    except Exception as e:
        print(f"   the-stack failed: {e}")
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
                                             text_fields=("code","content","text"))
    except Exception as e:
        print(f"   codeparrot failed: {e}")
if not stack_written:
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
    print(f"   Fallback: {e}")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["open_web_math"], "fineweb-extra", _bin_f, 8192)
print(f"   Running total: {total_written:,}\n")

print("=" * 65)
print("SOURCE 5/6 — Books")
print("=" * 65)
books_written = 0
for name, kwargs in [
    ("dolma-books",  dict(path="allenai/dolma",           name="books", split="train", streaming=True)),
    ("gutenberg",    dict(path="sedthh/gutenberg_english", split="train", streaming=True)),
    ("pile-books",   dict(path="EleutherAI/pile",         name="all",   split="train", streaming=True)),
]:
    try:
        ds = load_dataset(**kwargs)
        books_written = stream_tokens_to_bin(ds, sources["dolma_books"], name, _bin_f, 32768)
        break
    except Exception as e:
        print(f"   {name} failed: {e}")
if not books_written:
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
    ("dolma-wiki",   dict(path="allenai/dolma",       name="wiki",          split="train", streaming=True)),
    ("wikipedia-en", dict(path="wikimedia/wikipedia", name="20231101.en",   split="train", streaming=True)),
    ("fineweb-wiki", dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)),
]:
    try:
        ds = load_dataset(**kwargs)
        wiki_written = stream_tokens_to_bin(ds, sources["dolma_wiki"], name, _bin_f, 4096,
                                            text_fields=("text","passage","content"))
        break
    except Exception as e:
        print(f"   {name} failed: {e}")
total_written += wiki_written
print(f"   Running total: {total_written:,}\n")

_bin_f.close()

# ── Save train/val splits ──────────────────────────────────────────────────────
print("=" * 65)
print(f"SAVING {total_written:,} tokens as uint32 numpy arrays...")
print("=" * 65)

mm = np.memmap(TMP_BIN, dtype=np.uint32, mode="r", shape=(total_written,))

split_idx  = int(total_written * 0.97)
n_train    = split_idx
n_val      = total_written - split_idx
SLICE_SIZE = 10_000_000

print(f"  Writing train.npy ({n_train:,} tokens)...")
train_mm = np.lib.format.open_memmap(
    "data/train.npy", mode="w+", dtype=np.uint32, shape=(n_train,)
)
for start in range(0, n_train, SLICE_SIZE):
    end = min(start + SLICE_SIZE, n_train)
    train_mm[start:end] = mm[start:end]
    if (start // SLICE_SIZE) % 5 == 0:
        print(f"    train: {end:>12,} / {n_train:,}  ({end/n_train*100:.1f}%)")
del train_mm

print(f"  Writing val.npy ({n_val:,} tokens)...")
val_mm = np.lib.format.open_memmap(
    "data/val.npy", mode="w+", dtype=np.uint32, shape=(n_val,)
)
for start in range(0, n_val, SLICE_SIZE):
    end = min(start + SLICE_SIZE, n_val)
    val_mm[start:end] = mm[split_idx + start : split_idx + end]
del val_mm

del mm
os.remove(TMP_BIN)

train_gb = os.path.getsize("data/train.npy") / 1e9
val_mb   = os.path.getsize("data/val.npy")   / 1e6

print(f"\n{'='*65}")
print(f"  ✓ DONE")
print(f"  Train : {n_train:>16,} tokens | {train_gb:.2f} GB")
print(f"  Val   : {n_val:>16,} tokens | {val_mb:.0f} MB")
print(f"  Vocab : {len(tok):,} (IDs 0..{len(tok)-1})")
print(f"  Special tokens verified: [PAD]=65529 [EOS]=65532 <think>=65535")
print(f"{'='*65}")
print(f"\nNext: python3 train.py --smoke --phase 1")
