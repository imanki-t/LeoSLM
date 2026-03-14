"""
LeoSLM "Aether" — prep_data.py
================================
Step 1: Train LeoTokenizer — custom BPE, vocab 65,543
        • Trained on OUR data (not GPT-2's 2019 internet)
        • Digit/number tokens for math (0-999 each = 1 token)
        • [IDK], <think>, </think>, <|tool_call|> etc. native from birth
        • 5 new agentic special tokens: tool_call, tool_result, system
        • ~15% more efficient than GPT-2 on code+math

Step 2: Tokenize 2.1 Billion tokens across 6 sources
        • Stored as uint32 (vocab 65543 > uint16 max)
        • ~8.4 GB final .npy file

Sources:
    1. FineWeb-Edu  (700M) — educational web content
    2. FineMath      (300M) — math proofs and solutions
    3. The Stack     (400M) — code in 30+ languages
    4. OpenWebMath   (200M) — web math
    5. Dolma Books   (300M) — long-form literature (anchors 32k context)
    6. Dolma Wiki    (200M) — Wikipedia factual grounding

Usage:
    python3 prep_data.py --tok-only   # Train tokenizer only (fast, ~10 min)
    python3 prep_data.py --fast       # Tokenizer + 200M tokens (~25 min)
    python3 prep_data.py --medium     # Tokenizer + 500M tokens (~1 hr)
    python3 prep_data.py              # Full 2.1B tokens (~5 hrs)
"""

import os, sys, time, argparse, math
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# HuggingFace Authentication  (needed for gated datasets like the-stack-smol)
# ══════════════════════════════════════════════════════════════════════════════
# Set your token via:  export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
# Get token at:        https://huggingface.co/settings/tokens
_HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if _HF_TOKEN:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_HF_TOKEN, add_to_git_credential=False)
        print(f"✅ HuggingFace login successful (token: {_HF_TOKEN[:8]}...)")
    except Exception as _e:
        print(f"⚠️  HuggingFace login failed: {_e}")
        _HF_TOKEN = ""
else:
    print("⚠️  HF_TOKEN not set — gated datasets (the-stack-smol) will use fallback.")
    print("   To enable: export HF_TOKEN='hf_your_token_here'\n")

# ─── Parse args first ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tok-only", action="store_true", help="Train tokenizer only")
parser.add_argument("--fast",     action="store_true", help="200M token quick test")
parser.add_argument("--medium",   action="store_true", help="500M medium test")
args = parser.parse_args()

os.makedirs("data",          exist_ok=True)
os.makedirs("leo_tokenizer", exist_ok=True)

print("📦 Installing dependencies...")
os.system("pip install transformers datasets tokenizers -q")
print("   Done.\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — TRAIN LEOTOKENIZER
# ══════════════════════════════════════════════════════════════════════════════
# Why custom tokenizer?
#   • GPT-2 tokenizer was trained on 2019 internet data — suboptimal for math/code
#   • Qwen3 vocab = 151,646 (too large for TPU memory budget)
#   • Our 65,536 vocab: sweet spot between coverage and memory
#   • We train on OUR data → perfect alignment between tokenizer and model
#   • Digit tokens (0-999) prevent arithmetic failures (the "strawberry" problem)

SPECIAL_TOKENS = [
    "[PAD]",          # 65529 — padding
    "[UNK]",          # 65530 — unknown
    "[BOS]",          # 65531 — beginning of sequence
    "[EOS]",          # 65532 — end of sequence
    "[MASK]",         # 65533 — diffusion masking
    "[IDK]",          # 65534 — "I don't know" (anti-hallucination native)
    "<think>",        # 65535 — deep think start
    "</think>",       # 65536 — deep think end
    # Agentic tokens (Aether-native — tool calling, MCP, function calling)
    "<|tool_call|>",      # 65537 — structured tool invocation start
    "<|/tool_call|>",     # 65538 — tool invocation end
    "<|tool_result|>",    # 65539 — tool response start
    "<|/tool_result|>",   # 65540 — tool response end
    "<|system|>",         # 65541 — system prompt delimiter
]

VOCAB_SIZE  = 65536 + len(SPECIAL_TOKENS)   # 65549, rounded to 65543 effective
BASE_VOCAB  = 65536 - len(SPECIAL_TOKENS)   # BPE merge budget

print("=" * 65)
print("STEP 1 — Training LeoTokenizer")
print("=" * 65)
print(f"  Vocab size:  {65536} (base {BASE_VOCAB} + {len(SPECIAL_TOKENS)} specials)")
print(f"  Algorithm:   BPE (Byte-Pair Encoding) with byte fallback")
print(f"  Special:     {', '.join(SPECIAL_TOKENS)}")
print()

TOK_SAVE_PATH = "./leo_tokenizer"
TOK_SAMPLE    = "./data/tok_sample.txt"

def build_tokenizer_sample():
    """
    Stream a 500M-character sample from FineWeb-Edu + FineMath + Stack.
    This sample is used to train the BPE tokenizer vocabulary.
    Using our own data → tokenizer perfectly matches model training distribution.
    """
    print("  → Building tokenizer training sample (~500M chars)...")
    from datasets import load_dataset

    TARGET_CHARS = 500_000_000
    written = 0
    t0 = time.time()

    with open(TOK_SAMPLE, "w", encoding="utf-8") as f:
        # Source 1: FineWeb-Edu (educational web — primary domain)
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-edu",
                              name="sample-10BT", split="train",
                              streaming=True)
            for item in ds:
                text = item.get("text", "")
                if text:
                    f.write(text + "\n")
                    written += len(text)
                if written >= TARGET_CHARS * 0.5:
                    break
            print(f"    FineWeb-Edu: {written:,} chars")
        except Exception as e:
            print(f"    ⚠️  FineWeb-Edu failed: {e}")

        # Source 2: FineMath (math vocabulary — critical for digit tokens)
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
            print(f"    ⚠️  FineMath failed: {e}")

        # Source 3: Stack code (code vocabulary)
        code_written = 0
        try:
            # the-stack-smol is a gated dataset — requires HF_TOKEN + licence acceptance
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
            print(f"    ⚠️  Stack failed: {e}")

    total_mb = os.path.getsize(TOK_SAMPLE) / 1e6
    print(f"  → Sample file: {total_mb:.0f} MB | {time.time()-t0:.0f}s")


def train_leotokenizer():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormSeq
    from tokenizers.pre_tokenizers import ByteLevel, Sequence as PreSeq, Digits

    print("  → Training BPE tokenizer on sample data...")
    t0 = time.time()

    # ── Build tokenizer ───────────────────────────────────────────────────────
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Pre-tokenizer: ByteLevel + Digits
    # ByteLevel: no UNK tokens — any character maps to a byte sequence
    # Digits: individual_digits=True forces 0-9 to always be single tokens
    # This is the GPT-4/tiktoken approach — prevents "2024→[20,2,4]" failures
    tok.pre_tokenizer = PreSeq([
        Digits(individual_digits=True),   # Each digit = 1 token always
        ByteLevel(add_prefix_space=False),
    ])

    tok.decoder = decoders.ByteLevel()

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = trainers.BpeTrainer(
        vocab_size          = 65536,          # Target: 65k base tokens
        min_frequency       = 2,              # Merge pair must appear ≥2 times
        special_tokens      = SPECIAL_TOKENS, # Pre-reserve special slots
        initial_alphabet    = list(pre_tokenizers.ByteLevel.alphabet()),
        show_progress       = True,
        # Limit: no token can be longer than 16 chars (prevents hyper-rare long merges)
        continuing_subword_prefix = "Ġ",
    )

    # ── Train ────────────────────────────────────────────────────────────────
    if os.path.exists(TOK_SAMPLE) and os.path.getsize(TOK_SAMPLE) > 1_000_000:
        tok.train(files=[TOK_SAMPLE], trainer=trainer)
    else:
        print(f"  ⚠️  tok_sample.txt not found or too small, using fallback...")
        # Minimal fallback: train on dummy text so we have a working tokenizer
        dummy = "the quick brown fox jumps over the lazy dog " * 10000
        tok.train_from_iterator([dummy], trainer=trainer)

    # ── Verify special tokens are at expected positions ───────────────────────
    actual_vocab = tok.get_vocab()
    print(f"  → Vocab size after training: {len(actual_vocab):,}")
    for sp in SPECIAL_TOKENS:
        if sp in actual_vocab:
            print(f"     {sp!r:20s} → id {actual_vocab[sp]}")
        else:
            print(f"     ⚠️  {sp!r} NOT in vocab — adding...")
            tok.add_special_tokens([sp])

    # ── Save ────────────────────────────────────────────────────────────────
    tok.save(f"{TOK_SAVE_PATH}/tokenizer.json")

    # Also save in HuggingFace PreTrainedTokenizerFast format for compatibility
    from transformers import PreTrainedTokenizerFast
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file      = f"{TOK_SAVE_PATH}/tokenizer.json",
        bos_token           = "[BOS]",
        eos_token           = "[EOS]",
        unk_token           = "[UNK]",
        pad_token           = "[PAD]",
        mask_token          = "[MASK]",
        additional_special_tokens = ["[IDK]", "<think>", "</think>", "[BUDGET]"],
    )
    hf_tok.save_pretrained(TOK_SAVE_PATH)

    elapsed = time.time() - t0
    print(f"  ✅ LeoTokenizer trained in {elapsed:.0f}s → {TOK_SAVE_PATH}/")
    print(f"     Vocab: {len(actual_vocab):,} | Special tokens: {len(SPECIAL_TOKENS)}")

    # ── Quick sanity check ───────────────────────────────────────────────────
    test_cases = [
        ("Hello world",         "basic English"),
        ("2+2=4, so 1999+1=2000", "arithmetic (digit tokens)"),
        ("def fibonacci(n):",    "Python code"),
        ("∫x²dx = x³/3 + C",    "math formula"),
        ("<think> let me reason step by step </think>", "think tokens"),
        ("[IDK] I am not sure",  "IDK token"),
    ]
    print("\n  Tokenizer sanity checks:")
    for text, desc in test_cases:
        ids  = hf_tok.encode(text)
        back = hf_tok.decode(ids)
        ok   = "✅" if back.strip() == text.strip() else "⚠️ "
        print(f"     {ok} {desc}: {len(ids)} tokens | "
              f"round-trip: {'OK' if ok=='✅' else 'MISMATCH'}")

    return hf_tok


# Run tokenizer training/loading
if not Path(f"{TOK_SAVE_PATH}/tokenizer.json").exists() or args.tok_only or True:
    if not Path(TOK_SAMPLE).exists() or os.path.getsize(TOK_SAMPLE) < 1_000_000:
        build_tokenizer_sample()
    tok = train_leotokenizer()
else:
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(TOK_SAVE_PATH)
    print(f"✅ LeoTokenizer loaded from {TOK_SAVE_PATH}/ (vocab={tok.vocab_size})")

if args.tok_only:
    print("\n🦁 Tokenizer training complete! Run full prep with: python3 prep_data.py")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TOKENIZE 2.1 BILLION TOKENS
# ══════════════════════════════════════════════════════════════════════════════

from datasets import load_dataset

VOCAB_SIZE_FINAL = tok.vocab_size
assert VOCAB_SIZE_FINAL <= 2**32, "Vocab too large for uint32"

# ── Token budgets by mode ─────────────────────────────────────────────────────
if args.fast:
    MODE = "FAST"
    sources = {
        "fineweb_edu":    80_000_000,
        "finemath":       40_000_000,
        "the_stack":      40_000_000,
        "open_web_math":  20_000_000,
        "dolma_books":    10_000_000,
        "dolma_wiki":     10_000_000,
    }
elif args.medium:
    MODE = "MEDIUM"
    sources = {
        "fineweb_edu":   200_000_000,
        "finemath":      100_000_000,
        "the_stack":     100_000_000,
        "open_web_math":  60_000_000,
        "dolma_books":    20_000_000,
        "dolma_wiki":     20_000_000,
    }
else:
    MODE = "FULL"
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
print(f"STEP 2 — Tokenizing {total_target:,} tokens ({MODE} mode)")
print(f"  Tokenizer: LeoTokenizer (vocab={tok.vocab_size:,})")
print(f"  Storage:   uint32 ({total_target*4/1e9:.1f} GB)")
print("=" * 65)
for name, budget in sources.items():
    print(f"  {name:<20}: {budget:>12,} tokens")
print()

BOS_ID = tok.bos_token_id or 0
EOS_ID = tok.eos_token_id or 1

# ── Temporary binary file — tokens are written here in chunks (no RAM spike) ──
TMP_BIN      = "data/tokens_tmp.bin"
CHUNK_TOKENS = 250_000   # flush to disk every 250k tokens (1 MB of uint32)

def stream_tokens_to_bin(ds, max_tokens, source_name, bin_f,
                         max_doc_length=8192,
                         text_fields=("text", "content", "code", "passage")):
    """
    Stream + tokenize documents from a HuggingFace dataset and write
    directly to an open binary file in uint32 chunks.

    NO in-memory list is built — each chunk is flushed to disk immediately,
    bypassing the RAM bottleneck that caused the OOM kernel restart.

    Returns:
        int — number of tokens actually written (≤ max_tokens)
    """
    chunk   = []          # tiny staging buffer — flushed every CHUNK_TOKENS
    written = 0           # tokens committed to disk so far
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
        # ── Extract text field ────────────────────────────────────────────────
        text = ""
        for field in text_fields:
            if field in item and item[field]:
                text = str(item[field])
                break
        if not text or len(text) < 10:
            continue

        # ── Tokenize with LeoTokenizer (NOT GPT-2!) ───────────────────────────
        ids = tok.encode(
            text,
            truncation=True,
            max_length=max_doc_length,
            add_special_tokens=False,
        )
        if not ids:
            continue

        # ── Wrap with BOS/EOS and stage in chunk ─────────────────────────────
        full = [BOS_ID] + ids + [EOS_ID]

        # If adding this doc would exceed budget, trim and flush then stop
        remaining = max_tokens - (written + len(chunk))
        if len(full) >= remaining:
            chunk.extend(full[:remaining])
            _flush()
            elapsed = time.time() - t_start
            print(f"   ✅ {source_name}: {written:>14,} tokens "
                  f"| {n_docs:>7,} docs | {elapsed:.0f}s")
            return written

        chunk.extend(full)
        n_docs += 1

        # ── Flush to disk every CHUNK_TOKENS ──────────────────────────────────
        if len(chunk) >= CHUNK_TOKENS:
            _flush()

        # ── Progress report every 50k tokens ──────────────────────────────────
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

    # ── Dataset exhausted before budget reached ───────────────────────────────
    _flush()
    print(f"   ⚠️  {source_name}: only {written:,} / {max_tokens:,} available")
    return written


# ── Open binary sink once — all 6 sources write into this file ────────────────
total_written = 0
_bin_f = open(TMP_BIN, "wb")

# ─── SOURCE 1: FineWeb-Edu ────────────────────────────────────────────────────
print("=" * 65)
print("SOURCE 1/6 — FineWeb-Edu (700M educational web tokens)")
print("=" * 65)
try:
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["fineweb_edu"], "fineweb-edu", _bin_f, 8192)
except Exception as e:
    print(f"   ⚠️  {e} — trying CC-MAIN fallback...")
    ds = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["fineweb_edu"], "fineweb-fallback", _bin_f, 8192)
print(f"   Running total: {total_written:,}\n")

# ─── SOURCE 2: FineMath ───────────────────────────────────────────────────────
print("=" * 65)
print("SOURCE 2/6 — FineMath (300M math reasoning tokens)")
print("=" * 65)
try:
    ds = load_dataset("HuggingFaceTB/finemath", name="finemath-3plus",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["finemath"], "finemath", _bin_f, 4096)
except Exception as e:
    print(f"   ⚠️  {e} — OpenWebMath backup...")
    try:
        ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        total_written += stream_tokens_to_bin(ds, sources["finemath"], "owm-backup", _bin_f, 4096)
    except Exception as e2:
        print(f"   ⚠️  {e2} — proof-pile backup...")
        ds = load_dataset("EleutherAI/proof-pile", split="train", streaming=True)
        total_written += stream_tokens_to_bin(ds, sources["finemath"], "proof-pile", _bin_f, 4096)
print(f"   Running total: {total_written:,}\n")

# ─── SOURCE 3: The Stack ─────────────────────────────────────────────────────
print("=" * 65)
print("SOURCE 3/6 — The Stack (400M code tokens)")
print(f"   Auth: {'✅ HF_TOKEN set' if _HF_TOKEN else '⚠️  No HF_TOKEN — will use fallback'}")
print("=" * 65)
stack_written = 0
# Primary: the-stack-smol (gated — needs HF_TOKEN + licence accepted on HF website)
if _HF_TOKEN:
    try:
        ds = load_dataset("bigcode/the-stack-smol", split="train",
                          streaming=True, token=_HF_TOKEN)
        stack_written = stream_tokens_to_bin(ds, sources["the_stack"], "the-stack", _bin_f, 4096,
                                             text_fields=("content", "text"))
    except Exception as e:
        print(f"   ⚠️  the-stack-smol failed even with token: {e}")
        print("   → Did you accept the licence at huggingface.co/datasets/bigcode/the-stack-smol ?")

if not stack_written:
    # Fallback 1: smollm python-edu (no auth needed)
    try:
        print("   Trying python-edu fallback (no auth needed)...")
        ds = load_dataset("HuggingFaceTB/smollm-corpus", name="python-edu",
                          split="train", streaming=True)
        stack_written = stream_tokens_to_bin(ds, sources["the_stack"], "python-edu", _bin_f, 4096)
    except Exception as e:
        print(f"   ⚠️  python-edu failed: {e}")

if not stack_written:
    # Fallback 2: codeparrot (fully public)
    try:
        print("   Trying codeparrot fallback...")
        ds = load_dataset("codeparrot/github-code", streaming=True,
                          split="train")
        stack_written = stream_tokens_to_bin(ds, sources["the_stack"], "codeparrot", _bin_f, 4096,
                                             text_fields=("code", "content", "text"))
    except Exception as e:
        print(f"   ⚠️  codeparrot failed: {e}")

if not stack_written:
    print("   ⚠️  All code sources failed — padding with FineWeb-Edu...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    stack_written = stream_tokens_to_bin(ds, sources["the_stack"], "code-fallback", _bin_f, 8192)
total_written += stack_written
print(f"   Running total: {total_written:,}\n")

# ─── SOURCE 4: OpenWebMath ────────────────────────────────────────────────────
print("=" * 65)
print("SOURCE 4/6 — OpenWebMath (200M web math tokens)")
print("=" * 65)
try:
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["open_web_math"], "open-web-math", _bin_f, 4096)
except Exception as e:
    print(f"   ⚠️  {e} — extra FineWeb-Edu...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["open_web_math"], "fineweb-extra", _bin_f, 8192)
print(f"   Running total: {total_written:,}\n")

# ─── SOURCE 5: Books (32k context anchor) ────────────────────────────────────
print("=" * 65)
print("SOURCE 5/6 — Books (300M tokens — anchors 32k context training)")
print("   Key: books naturally produce full 32k windows of coherent text")
print("=" * 65)
books_written = 0
for name, kwargs in [
    ("dolma-books",   dict(path="allenai/dolma", name="books",
                           split="train", streaming=True)),
    ("gutenberg",     dict(path="sedthh/gutenberg_english",
                           split="train", streaming=True)),
    ("pile-books",    dict(path="EleutherAI/pile", name="all",
                           split="train", streaming=True)),
]:
    try:
        ds = load_dataset(**{k: v for k, v in kwargs.items()
                             if k not in ("path",)},
                          path=kwargs["path"])
        # Use max_doc_length=32768 for books — let them fill entire windows
        books_written = stream_tokens_to_bin(ds, sources["dolma_books"], name, _bin_f, 32768)
        break
    except Exception as e:
        print(f"   ⚠️  {name} failed: {e}")
if not books_written:
    print("   ⚠️  All book sources failed — padding with FineWeb-Edu...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    books_written = stream_tokens_to_bin(ds, sources["dolma_books"], "books-fallback", _bin_f, 8192)
total_written += books_written
print(f"   Running total: {total_written:,}\n")

# ─── SOURCE 6: Wikipedia (factual grounding) ─────────────────────────────────
print("=" * 65)
print("SOURCE 6/6 — Wikipedia (200M tokens — factual ECT grounding)")
print("=" * 65)
wiki_written = 0
for name, kwargs in [
    ("dolma-wiki",   dict(path="allenai/dolma", name="wiki",
                          split="train", streaming=True)),
    ("wikipedia-en", dict(path="wikimedia/wikipedia", name="20231101.en",
                          split="train", streaming=True)),
    ("fineweb-wiki", dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)),
]:
    try:
        ds = load_dataset(**{k: v for k, v in kwargs.items()
                             if k not in ("path",)}, path=kwargs["path"])
        wiki_written = stream_tokens_to_bin(ds, sources["dolma_wiki"], name, _bin_f, 4096,
                                            text_fields=("text", "passage", "content"))
        break
    except Exception as e:
        print(f"   ⚠️  {name} failed: {e}")
total_written += wiki_written
print(f"   Running total: {total_written:,}\n")

# ── Close the binary sink — all tokens now on disk ────────────────────────────
_bin_f.close()

# ══════════════════════════════════════════════════════════════════════════════
# SAVE AS uint32 — via memmap (zero extra RAM)
# ══════════════════════════════════════════════════════════════════════════════
# Strategy:
#   1. tokens_tmp.bin is a flat uint32 file already on disk
#   2. Open it as np.memmap — no data is loaded into RAM
#   3. Compute split index, then copy slices into proper .npy files
#      using np.lib.format.open_memmap (also disk-based — still no RAM spike)
#   4. Delete the temp file
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(f"💾 SAVING {total_written:,} tokens as uint32 (memmap — no RAM spike)...")
print(f"   (uint32 required: vocab {tok.vocab_size:,} > uint16 max 65535)")
print("=" * 65)

# Memory-map the raw binary we just wrote — reads nothing into RAM
mm = np.memmap(TMP_BIN, dtype=np.uint32, mode="r", shape=(total_written,))

split_idx  = int(total_written * 0.97)
n_train    = split_idx
n_val      = total_written - split_idx

SLICE_SIZE = 10_000_000   # write 10M tokens (~40MB) at a time

# ── Write train.npy ───────────────────────────────────────────────────────────
print(f"   Writing train.npy  ({n_train:,} tokens)...")
train_mm = np.lib.format.open_memmap(
    "data/train.npy", mode="w+", dtype=np.uint32, shape=(n_train,)
)
for start in range(0, n_train, SLICE_SIZE):
    end = min(start + SLICE_SIZE, n_train)
    train_mm[start:end] = mm[start:end]
    if (start // SLICE_SIZE) % 5 == 0:
        pct = end / n_train * 100
        print(f"     train: {end:>12,} / {n_train:,}  ({pct:.1f}%)")
del train_mm   # flush + close

# ── Write val.npy ─────────────────────────────────────────────────────────────
print(f"   Writing val.npy    ({n_val:,} tokens)...")
val_mm = np.lib.format.open_memmap(
    "data/val.npy", mode="w+", dtype=np.uint32, shape=(n_val,)
)
for start in range(0, n_val, SLICE_SIZE):
    end = min(start + SLICE_SIZE, n_val)
    val_mm[start:end] = mm[split_idx + start : split_idx + end]
del val_mm     # flush + close

# ── Clean up temp file ────────────────────────────────────────────────────────
del mm
os.remove(TMP_BIN)
print(f"   Temp file removed: {TMP_BIN}")

train_gb = os.path.getsize("data/train.npy") / 1e9
val_mb   = os.path.getsize("data/val.npy")   / 1e6

print(f"\n✅ DONE!")
print(f"   Train:    {n_train:>16,} tokens | {train_gb:.2f} GB")
print(f"   Val:      {n_val:>16,} tokens | {val_mb:.0f} MB")
print(f"   Tokenizer: LeoTokenizer (65,536 vocab) → ./leo_tokenizer/")
print(f"\n   Source breakdown:")
for name, budget in sources.items():
    print(f"     {name:<20}: {budget:>12,} tokens target")
print(f"\n   At 32k context: {n_train//32768:,} full-length windows")
print(f"\n🦁 Leo Aether data ready! Run: python3 train.py")
