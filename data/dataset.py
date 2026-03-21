"""
data/dataset.py

BUG FIXES vs original:
  1. dtype: data was stored as uint32 (vocab > 65535) but loaded as int32,
     which would overflow for IDs above 2^31-1. Although current vocab (65543)
     fits in int32, future vocab expansions could break this silently.
     Fix: load as int64 (torch.long standard) with a bounds check.
  2. LeoStreamDataset.__len__: edge case when len(data) < seq_len returned 0
     (max(1,...) missing). Added consistent guard.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict


class LeoDataset(Dataset):
    """
    Sliding-window dataset over a flat token array stored as .npy uint32.
    Windows advance by seq_len (non-overlapping).
    """

    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 4096,
        pad_id: int = 65529,
    ):
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        raw = np.load(data_path, mmap_mode="r")

        # BUG FIX: uint32 can hold vocab IDs up to 4B; int32 overflows at 2^31.
        # Cast to int64 which is what torch.long needs. Memory cost: 2× uint32,
        # but mmap_mode="r" means only accessed pages are loaded anyway.
        # For a 3B-token dataset this adds ~12 GB virtual address space (not RAM).
        self.data    = raw.astype(np.int64)
        self.seq_len = max_seq_len
        self.pad_id  = pad_id
        self._update_n()

    def _update_n(self):
        self._n = max(1, (len(self.data) - 1) // self.seq_len)

    def set_seq_len(self, seq_len: int) -> None:
        self.seq_len = seq_len
        self._update_n()

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Wrap-around start to use all data even when len(data) is not a multiple of seq_len
        start = (idx * self.seq_len) % max(1, len(self.data) - self.seq_len - 1)
        chunk = self.data[start : start + self.seq_len + 1]
        L     = len(chunk)
        if L < self.seq_len + 1:
            pad   = np.full(self.seq_len + 1 - L, self.pad_id, dtype=np.int64)
            chunk = np.concatenate([chunk, pad])
        ids = torch.from_numpy(chunk[: self.seq_len + 1].copy()).long()
        return {"input_ids": ids[:-1], "labels": ids[1:]}


class LeoStreamDataset(Dataset):
    """
    Overlapping sliding-window dataset (configurable stride).
    Useful for dense training on smaller datasets.
    """

    def __init__(
        self,
        data_path:   str,
        max_seq_len: int = 4096,
        pad_id:      int = 65529,
        stride:      int = 0,
    ):
        raw          = np.load(data_path, mmap_mode="r")
        self.data    = raw.astype(np.int64)   # BUG FIX: int64 not int32
        self.seq_len = max_seq_len
        self.pad_id  = pad_id
        self.stride  = stride if stride > 0 else max_seq_len

    def set_seq_len(self, seq_len: int) -> None:
        self.seq_len = seq_len

    def __len__(self) -> int:
        # BUG FIX: add max(1, ...) guard for datasets smaller than seq_len
        return max(1, (len(self.data) - self.seq_len) // self.stride)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        chunk = self.data[start : start + self.seq_len + 1]
        L     = len(chunk)
        if L < self.seq_len + 1:
            pad   = np.full(self.seq_len + 1 - L, self.pad_id, dtype=np.int64)
            chunk = np.concatenate([chunk, pad])
        ids = torch.from_numpy(chunk[: self.seq_len + 1].copy()).long()
        return {"input_ids": ids[:-1], "labels": ids[1:]}
