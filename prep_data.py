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
    print("To enable: export HF_TOKEN='hf_your_token_here'\n")

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

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[MASK]",
    "[IDK]",
    "<think>",
    "</think>",
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_result|>",
    "<|/tool_result|>",
    "<|system|>",
]

VOCAB_SIZE = 65536 + len(SPECIAL_TOKENS)
BASE_VOCAB  = 65536 - len(SPECIAL_TOKENS)

print("=" * 65)
print("STEP 1 — Training LeoTokenizer")
print("=" * 65)
print(f"  Vocab size:  {65536} (base {BASE_VOCAB} + {len(SPECIAL_TOKENS)} specials)")
print(f"  Algorithm:   BPE with byte fallback")
print(f"  Special:     {', '.join(SPECIAL_TOKENS)}")
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
    from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormSeq
    from tokenizers.pre_tokenizers import ByteLevel, Sequence as PreSeq, Digits

    print("  Training BPE tokenizer on sample data...")
    t0 = time.time()

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = PreSeq([
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size                = 65536,
        min_frequency             = 2,
        special_tokens            = SPECIAL_TOKENS,
        initial_alphabet          = list(pre_tokenizers.ByteLevel.alphabet()),
        show_progress             = True,
        continuing_subword_prefix = "Ġ",
    )

    if os.path.exists(TOK_SAMPLE) and os.path.getsize(TOK_SAMPLE) > 1_000_000:
        tok.train(files=[TOK_SAMPLE], trainer=trainer)
    else:
        print("  tok_sample.txt not found or too small, using fallback...")
        dummy = "the quick brown fox jumps over the lazy dog " * 10000
        tok.train_from_iterator([dummy], trainer=trainer)

    actual_vocab = tok.get_vocab()
    print(f"  Vocab size after training: {len(actual_vocab):,}")
    for sp in SPECIAL_TOKENS:
        if sp in actual_vocab:
            print(f"     {sp!r:20s} -> id {actual_vocab[sp]}")
        else:
            print(f"     {sp!r} NOT in vocab — adding...")
            tok.add_special_tokens([sp])

    tok.save(f"{TOK_SAVE_PATH}/tokenizer.json")

    from transformers import PreTrainedTokenizerFast
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file            = f"{TOK_SAVE_PATH}/tokenizer.json",
        bos_token                 = "[BOS]",
        eos_token                 = "[EOS]",
        unk_token                 = "[UNK]",
        pad_token                 = "[PAD]",
        mask_token                = "[MASK]",
        additional_special_tokens = ["[IDK]", "<think>", "</think>", "[BUDGET]"],
    )
    hf_tok.save_pretrained(TOK_SAVE_PATH)

    elapsed = time.time() - t0
    print(f"  LeoTokenizer trained in {elapsed:.0f}s -> {TOK_SAVE_PATH}/")
    print(f"     Vocab: {len(actual_vocab):,} | Special tokens: {len(SPECIAL_TOKENS)}")

    test_cases = [
        ("Hello world",                                  "basic English"),
        ("2+2=4, so 1999+1=2000",                        "arithmetic (digit tokens)"),
        ("def fibonacci(n):",                             "Python code"),
        ("<think> let me reason step by step </think>",  "think tokens"),
        ("[IDK] I am not sure",                          "IDK token"),
    ]
    print("\n  Tokenizer sanity checks:")
    for text, desc in test_cases:
        ids  = hf_tok.encode(text)
        back = hf_tok.decode(ids)
        ok   = "OK" if back.strip() == text.strip() else "MISMATCH"
        print(f"     {ok}  {desc}: {len(ids)} tokens | round-trip: {ok}")

    return hf_tok


if not Path(f"{TOK_SAVE_PATH}/tokenizer.json").exists() or args.tok_only:
    if not Path(TOK_SAMPLE).exists() or os.path.getsize(TOK_SAMPLE) < 1_000_000:
        build_tokenizer_sample()
    tok = train_leotokenizer()
else:
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(TOK_SAVE_PATH)
    print(f"LeoTokenizer loaded from {TOK_SAVE_PATH}/ (vocab={tok.vocab_size})")

if args.tok_only:
    print("\nTokenizer training complete! Run full prep with: python3 prep_data.py")
    sys.exit(0)

from datasets import load_dataset

VOCAB_SIZE_FINAL = tok.vocab_size
assert VOCAB_SIZE_FINAL <= 2**32, "Vocab too large for uint32"

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

        ids = tok.encode(
            text,
            truncation=True,
            max_length=max_doc_length,
            add_special_tokens=False,
        )
        if not ids:
            continue

        full      = [BOS_ID] + ids + [EOS_ID]
        remaining = max_tokens - (written + len(chunk))
        if len(full) >= remaining:
            chunk.extend(full[:remaining])
            _flush()
            elapsed = time.time() - t_start
            print(f"   {source_name}: {written:>14,} tokens "
                  f"| {n_docs:>7,} docs | {elapsed:.0f}s")
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
print("SOURCE 1/6 — FineWeb-Edu (700M educational web tokens)")
print("=" * 65)
try:
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["fineweb_edu"], "fineweb-edu", _bin_f, 8192)
except Exception as e:
    print(f"   {e} — trying CC-MAIN fallback...")
    ds = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["fineweb_edu"], "fineweb-fallback", _bin_f, 8192)
print(f"   Running total: {total_written:,}\n")

print("=" * 65)
print("SOURCE 2/6 — FineMath (300M math reasoning tokens)")
print("=" * 65)
try:
    ds = load_dataset("HuggingFaceTB/finemath", name="finemath-3plus",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["finemath"], "finemath", _bin_f, 4096)
except Exception as e:
    print(f"   {e} — OpenWebMath backup...")
    try:
        ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        total_written += stream_tokens_to_bin(ds, sources["finemath"], "owm-backup", _bin_f, 4096)
    except Exception as e2:
        print(f"   {e2} — proof-pile backup...")
        ds = load_dataset("EleutherAI/proof-pile", split="train", streaming=True)
        total_written += stream_tokens_to_bin(ds, sources["finemath"], "proof-pile", _bin_f, 4096)
print(f"   Running total: {total_written:,}\n")

print("=" * 65)
print("SOURCE 3/6 — The Stack (400M code tokens)")
print(f"   Auth: {'HF_TOKEN set' if _HF_TOKEN else 'No HF_TOKEN — will use fallback'}")
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
        print("   Did you accept the licence at huggingface.co/datasets/bigcode/the-stack-smol ?")

if not stack_written:
    try:
        print("   Trying python-edu fallback...")
        ds = load_dataset("HuggingFaceTB/smollm-corpus", name="python-edu",
                          split="train", streaming=True)
        stack_written = stream_tokens_to_bin(ds, sources["the_stack"], "python-edu", _bin_f, 4096)
    except Exception as e:
        print(f"   python-edu failed: {e}")

if not stack_written:
    try:
        print("   Trying codeparrot fallback...")
        ds = load_dataset("codeparrot/github-code", streaming=True, split="train")
        stack_written = stream_tokens_to_bin(ds, sources["the_stack"], "codeparrot", _bin_f, 4096,
                                             text_fields=("code", "content", "text"))
    except Exception as e:
        print(f"   codeparrot failed: {e}")

if not stack_written:
    print("   All code sources failed — padding with FineWeb-Edu...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    stack_written = stream_tokens_to_bin(ds, sources["the_stack"], "code-fallback", _bin_f, 8192)
total_written += stack_written
print(f"   Running total: {total_written:,}\n")

print("=" * 65)
print("SOURCE 4/6 — OpenWebMath (200M web math tokens)")
print("=" * 65)
try:
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["open_web_math"], "open-web-math", _bin_f, 4096)
except Exception as e:
    print(f"   {e} — extra FineWeb-Edu...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    total_written += stream_tokens_to_bin(ds, sources["open_web_math"], "fineweb-extra", _bin_f, 8192)
print(f"   Running total: {total_written:,}\n")

print("=" * 65)
print("SOURCE 5/6 — Books (300M tokens)")
print("=" * 65)
books_written = 0
for name, kwargs in [
    ("dolma-books",  {"path": "allenai/dolma",           "name": "books",
                      "split": "train", "streaming": True}),
    ("gutenberg",    {"path": "sedthh/gutenberg_english",
                      "split": "train", "streaming": True}),
    ("pile-books",   {"path": "EleutherAI/pile",         "name": "all",
                      "split": "train", "streaming": True}),
]:
    try:
        ds = load_dataset(**kwargs)
        books_written = stream_tokens_to_bin(ds, sources["dolma_books"], name, _bin_f, 32768)
        break
    except Exception as e:
        print(f"   {name} failed: {e}")
if not books_written:
    print("   All book sources failed — padding with FineWeb-Edu...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    books_written = stream_tokens_to_bin(ds, sources["dolma_books"], "books-fallback", _bin_f, 8192)
total_written += books_written
print(f"   Running total: {total_written:,}\n")

print("=" * 65)
print("SOURCE 6/6 — Wikipedia (200M tokens)")
print("=" * 65)
wiki_written = 0
for name, kwargs in [
    ("dolma-wiki",   {"path": "allenai/dolma",        "name": "wiki",
                      "split": "train", "streaming": True}),
    ("wikipedia-en", {"path": "wikimedia/wikipedia",  "name": "20231101.en",
                      "split": "train", "streaming": True}),
    ("fineweb-wiki", {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT",
                      "split": "train", "streaming": True}),
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

print("=" * 65)
print(f"SAVING {total_written:,} tokens as uint32 (memmap — no RAM spike)...")
print(f"   (uint32 required: vocab {tok.vocab_size:,} > uint16 max 65535)")
print("=" * 65)

mm = np.memmap(TMP_BIN, dtype=np.uint32, mode="r", shape=(total_written,))

split_idx  = int(total_written * 0.97)
n_train    = split_idx
n_val      = total_written - split_idx
SLICE_SIZE = 10_000_000

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
del train_mm

print(f"   Writing val.npy    ({n_val:,} tokens)...")
val_mm = np.lib.format.open_memmap(
    "data/val.npy", mode="w+", dtype=np.uint32, shape=(n_val,)
)
for start in range(0, n_val, SLICE_SIZE):
    end = min(start + SLICE_SIZE, n_val)
    val_mm[start:end] = mm[split_idx + start : split_idx + end]
del val_mm

del mm
os.remove(TMP_BIN)
print(f"   Temp file removed: {TMP_BIN}")

train_gb = os.path.getsize("data/train.npy") / 1e9
val_mb   = os.path.getsize("data/val.npy")   / 1e6

print(f"\nDONE!")
print(f"   Train:    {n_train:>16,} tokens | {train_gb:.2f} GB")
print(f"   Val:      {n_val:>16,} tokens | {val_mb:.0f} MB")
print(f"   Tokenizer: LeoTokenizer (65,536 vocab) -> ./leo_tokenizer/")
print(f"\n   Source breakdown:")
for name, budget in sources.items():
    print(f"     {name:<20}: {budget:>12,} tokens target")
print(f"\n   At 32k context: {n_train//32768:,} full-length windows")
print(f"\nLeo Aether data ready! Run: python3 train.py")
