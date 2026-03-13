"""
LeoSLM — prep_data.py
======================
Run this ONCE before training to download and tokenize TinyStories.
Takes ~10-20 mins on Oracle A1.

Usage:
    python3 prep_data.py
"""

import numpy as np
import os

os.makedirs("data", exist_ok=True)

print("📦 Installing/checking dependencies...")
os.system("pip3 install transformers datasets -q")

from datasets import load_dataset
from transformers import AutoTokenizer

print("\n🔤 Loading tokenizer (GPT-2)...")
tok = AutoTokenizer.from_pretrained("gpt2")
tok.add_special_tokens({
    "pad_token": "[PAD]",
    "additional_special_tokens": ["[MASK]", "[IDK]"]
})
print(f"   Vocab size: {len(tok)}")

print("\n📥 Downloading TinyStories (~1GB, takes a few mins)...")
ds = load_dataset("roneneldan/TinyStories", split="train")
print(f"   Loaded {len(ds):,} stories")

print("\n🔄 Tokenizing (this is the slow part, ~15 mins on CPU)...")
all_tokens = []
for i, item in enumerate(ds):
    ids = tok.encode(item["text"], truncation=False)
    all_tokens.extend(ids)
    all_tokens.append(tok.eos_token_id)

    if i % 20000 == 0:
        print(f"   {i:>6}/{len(ds)} stories | {len(all_tokens):>10,} tokens")

print(f"\n✅ Total tokens: {len(all_tokens):,}")

tokens = np.array(all_tokens, dtype=np.int32)

split      = int(len(tokens) * 0.95)
train_toks = tokens[:split]
val_toks   = tokens[split:]

np.save("data/tinystories_train.npy", train_toks)
np.save("data/tinystories_val.npy",   val_toks)

print(f"   Train: {len(train_toks):,} tokens → data/tinystories_train.npy")
print(f"   Val  : {len(val_toks):,} tokens  → data/tinystories_val.npy")
print("\n🎉 Data ready! Now run: python3 train.py")
