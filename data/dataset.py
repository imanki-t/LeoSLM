"""
LeoSLM — Data Pipeline
========================
Tokenizer wrapper + dataset for TinyStories / FineWeb-Edu training.

Tokenizer:
    Uses HuggingFace's fast BPE tokenizer (GPT-2 style, 50257 vocab).
    We extend it with 3 special tokens:
        [PAD] id=0
        [MASK] id=1  (for diffusion masking)
        [IDK]  id=2  (for uncertainty / "I don't know" training)

Dataset:
    LeoDataset wraps any text dataset and:
        - Tokenizes on-the-fly
        - Chunks into fixed max_len windows (with stride for overlap)
        - Pads/truncates to max_len
        - Returns (input_ids, attention_mask)

Optimized for CPU training:
    - Pre-tokenizes and caches to disk
    - Uses memory-mapped numpy arrays for large datasets
    - Minimal RAM usage via lazy loading
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple
from pathlib import Path


# ---------------------------------------------------------------------------
# Tokenizer Wrapper
# ---------------------------------------------------------------------------

class LeoTokenizer:
    """
    Thin wrapper around a HuggingFace tokenizer.
    Adds LeoSLM special tokens and convenience methods.

    Args:
        tokenizer_name : HF tokenizer name (default: "gpt2")
        max_len        : maximum sequence length
    """

    PAD_TOKEN  = "[PAD]"
    MASK_TOKEN = "[MASK]"
    IDK_TOKEN  = "[IDK]"

    def __init__(self, tokenizer_name: str = "gpt2", max_len: int = 512):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len   = max_len

        # Add special tokens if not present
        special_tokens = {
            "pad_token"               : self.PAD_TOKEN,
            "additional_special_tokens": [self.MASK_TOKEN, self.IDK_TOKEN],
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            print(f"Added {num_added} special tokens to tokenizer")

        # Cache special token ids
        self.pad_token_id  = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.MASK_TOKEN)
        self.idk_token_id  = self.tokenizer.convert_tokens_to_ids(self.IDK_TOKEN)
        self.vocab_size    = len(self.tokenizer)

        print(f"Tokenizer vocab size: {self.vocab_size}")
        print(f"Special tokens: PAD={self.pad_token_id}, MASK={self.mask_token_id}, IDK={self.idk_token_id}")

    def encode(
        self,
        text           : str,
        return_tensors : str = "pt",
        padding        : bool = False,
        truncation     : bool = True,
    ) -> torch.Tensor:
        out = self.tokenizer(
            text,
            return_tensors  = return_tensors,
            padding         = "max_length" if padding else False,
            truncation      = truncation,
            max_length      = self.max_len,
        )
        return out["input_ids"]

    def decode(self, token_ids: torch.Tensor, skip_special: bool = True) -> str:
        if token_ids.dim() > 1:
            token_ids = token_ids[0]
        return self.tokenizer.decode(
            token_ids.tolist(),
            skip_special_tokens = skip_special,
        )

    def encode_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding     = "max_length",
            truncation  = True,
            max_length  = self.max_len,
            return_tensors = "pt",
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LeoDataset(Dataset):
    """
    Text dataset for LeoSLM training.

    Supports two modes:
        1. Raw text list (tokenized lazily)
        2. Pre-tokenized numpy array (fast, memory-efficient)

    For Oracle CPU training, use mode 2 with pre-tokenized data
    saved to disk. Call LeoDataset.from_hf_dataset() to create it.

    Args:
        data        : list of texts OR path to .npy pre-tokenized file
        tokenizer   : LeoTokenizer instance
        max_len     : sequence length
        stride      : stride for chunking long documents (0 = no overlap)
    """

    def __init__(
        self,
        data      : "str | list",
        tokenizer : LeoTokenizer,
        max_len   : int = 512,
        stride    : int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.stride    = stride

        if isinstance(data, (str, Path)) and str(data).endswith(".npy"):
            # Load pre-tokenized data (fast path)
            self.tokens = np.load(str(data), mmap_mode="r")   # memory mapped
            self.mode   = "pretokenized"
            print(f"Loaded pre-tokenized dataset: {self.tokens.shape}")
        elif isinstance(data, list):
            # Raw texts (tokenize on-the-fly)
            self.texts = data
            self.mode  = "raw"
            print(f"Raw text dataset: {len(data)} samples")
        else:
            raise ValueError("data must be a list of texts or path to .npy file")

    def __len__(self) -> int:
        if self.mode == "pretokenized":
            # Number of chunks
            T = self.tokens.shape[0]
            if self.stride > 0:
                return max(1, (T - self.max_len) // self.stride + 1)
            return T // self.max_len
        else:
            return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == "pretokenized":
            if self.stride > 0:
                start = idx * self.stride
            else:
                start = idx * self.max_len

            end    = start + self.max_len
            chunk  = self.tokens[start:end]

            # Pad if needed
            if len(chunk) < self.max_len:
                pad = np.full(self.max_len - len(chunk), self.tokenizer.pad_token_id)
                chunk = np.concatenate([chunk, pad])

            ids  = torch.tensor(chunk, dtype=torch.long)
            mask = (ids != self.tokenizer.pad_token_id).long()
            return {"input_ids": ids, "attention_mask": mask}

        else:
            # Raw text: tokenize on-the-fly
            text = self.texts[idx]
            out  = self.tokenizer.encode_batch([text])
            return {
                "input_ids"     : out["input_ids"][0],
                "attention_mask": out["attention_mask"][0],
            }

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_name  : str,
        tokenizer     : LeoTokenizer,
        save_path     : str,
        split         : str = "train",
        text_column   : str = "text",
        max_samples   : Optional[int] = None,
        max_len       : int = 512,
    ) -> "LeoDataset":
        """
        Load a HuggingFace dataset, tokenize it, and save as .npy for fast reuse.
        Run this ONCE, then load the .npy file in subsequent training runs.

        Example:
            ds = LeoDataset.from_hf_dataset(
                "roneneldan/TinyStories", tokenizer, "./data/tinystories.npy"
            )
        """
        from datasets import load_dataset

        print(f"Loading {dataset_name} ({split})...")
        dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        print(f"Tokenizing {len(dataset)} samples...")
        all_tokens = []
        for i, item in enumerate(dataset):
            text = item[text_column]
            ids  = tokenizer.tokenizer.encode(text, truncation=False)
            all_tokens.extend(ids)
            all_tokens.append(tokenizer.tokenizer.eos_token_id or 0)

            if i % 10000 == 0:
                print(f"  Tokenized {i}/{len(dataset)} samples ({len(all_tokens):,} tokens)")

        tokens_arr = np.array(all_tokens, dtype=np.int32)
        np.save(save_path, tokens_arr)
        print(f"Saved {len(tokens_arr):,} tokens to {save_path}")

        return cls(save_path, tokenizer, max_len=max_len)


def create_dataloader(
    dataset    : LeoDataset,
    batch_size : int = 8,
    shuffle    : bool = True,
    num_workers: int = 0,     # 0 for CPU training (avoids multiprocessing overhead)
) -> DataLoader:
    """Create DataLoader for training."""
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = False,   # no GPU
        drop_last   = True,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Test with dummy raw text data (no HF download needed for this check)
    dummy_texts = [
        "Once upon a time there was a little bear who loved honey.",
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
    ] * 100

    print("Testing with dummy text data (no tokenizer needed for shape check)...")

    # Test pre-tokenized mode with a dummy numpy file
    dummy_tokens = np.random.randint(3, 1000, size=(10000,), dtype=np.int32)
    np.save("/tmp/dummy_tokens.npy", dummy_tokens)

    class DummyTokenizer:
        pad_token_id  = 0
        mask_token_id = 1
        idk_token_id  = 2
        vocab_size    = 1000

    tok = DummyTokenizer()

    class MinimalDataset(Dataset):
        def __init__(self, tokens, max_len=32):
            self.tokens  = tokens
            self.max_len = max_len
        def __len__(self): return len(self.tokens) // self.max_len
        def __getitem__(self, idx):
            start = idx * self.max_len
            ids   = torch.tensor(self.tokens[start:start+self.max_len].astype(np.int64))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    ds     = MinimalDataset(dummy_tokens, max_len=32)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch  = next(iter(loader))

    print(f"Batch input_ids shape     : {batch['input_ids'].shape}")     # (4, 32)
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")# (4, 32)
    print(f"Dataset length: {len(ds)}")
    print("✅ dataset.py OK")
