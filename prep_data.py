"""
LeoSLM — prep_data.py
======================
Downloads and tokenizes training data.

TWO dataset options:
    1. TinyStories  — 1GB, simple stories, fast, good for testing
    2. FineWeb-Edu  — streamed directly from HuggingFace, no storage needed,
                      much smarter model, real educational content

Usage:
    python3 prep_data.py                      # TinyStories (default)
    python3 prep_data.py --dataset fineweb    # FineWeb-Edu streamed
    python3 prep_data.py --max_tokens 50000000  # limit tokens (50M default for Kaggle)
"""

import numpy as np
import os
import argparse

os.makedirs("data", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    type=str, default="tinystories",
                    choices=["tinystories", "fineweb"],
                    help="Which dataset to use")
parser.add_argument("--max_tokens", type=int, default=50_000_000,
                    help="Max tokens to collect (default 50M, fits Kaggle storage)")
args = parser.parse_args()

print("Installing dependencies...")
os.system("pip3 install transformers datasets -q")

from datasets import load_dataset
from transformers import AutoTokenizer

print("\nLoading tokenizer (GPT-2)...")
tok = AutoTokenizer.from_pretrained("gpt2")
tok.add_special_tokens({
    "pad_token"               : "[PAD]",
    "additional_special_tokens": ["[MASK]", "[IDK]"]
})
print(f"   Vocab size: {len(tok)}")

if args.dataset == "tinystories":
    print("\nDownloading TinyStories (~1GB)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"   {len(ds):,} stories loaded")
    print("\nTokenizing...")
    all_tokens = []
    for i, item in enumerate(ds):
        ids = tok.encode(item["text"], truncation=False)
        all_tokens.extend(ids)
        all_tokens.append(tok.eos_token_id)
        if len(all_tokens) >= args.max_tokens:
            break
        if i % 20000 == 0:
            print(f"   {i:>6}/{len(ds)} | {len(all_tokens):>10,} tokens")

else:
    print("\nStreaming FineWeb-Edu (no download needed)...")
    print("   No storage used — fetches batches live from HuggingFace\n")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name      = "sample-10BT",
        split     = "train",
        streaming = True,
    )
    all_tokens = []
    i = 0
    for item in ds:
        ids = tok.encode(item["text"], truncation=False, max_length=1024)
        all_tokens.extend(ids)
        all_tokens.append(tok.eos_token_id)
        i += 1
        if len(all_tokens) >= args.max_tokens:
            print(f"   Reached {args.max_tokens:,} tokens at doc {i:,}")
            break
        if i % 5000 == 0:
            print(f"   {i:>6} docs | {len(all_tokens):>10,} / {args.max_tokens:,} tokens")

print(f"\nTotal tokens: {len(all_tokens):,}")
tokens = np.array(all_tokens, dtype=np.int32)
split  = int(len(tokens) * 0.95)

np.save("data/train.npy", tokens[:split])
np.save("data/val.npy",   tokens[split:])

print(f"   Train : {len(tokens[:split]):,} tokens -> data/train.npy")
print(f"   Val   : {len(tokens[split:]):,} tokens -> data/val.npy")
print(f"   Size  : ~{os.path.getsize('data/train.npy') / 1e6:.0f} MB on disk")
print("\nDone! Now run: python3 train.py --train_data data/train.npy")
