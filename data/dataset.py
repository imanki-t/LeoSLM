import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict


class LeoDataset(Dataset):
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
        self.data    = raw.astype(np.int32)
        self.seq_len = max_seq_len
        self.pad_id  = pad_id
        self._n      = max(1, (len(self.data) - 1) // max_seq_len)

    def set_seq_len(self, seq_len: int) -> None:
        self.seq_len = seq_len
        self._n      = max(1, (len(self.data) - 1) // seq_len)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = (idx * self.seq_len) % max(1, len(self.data) - self.seq_len - 1)
        chunk = self.data[start : start + self.seq_len + 1]
        L     = len(chunk)
        if L < self.seq_len + 1:
            pad = np.full(self.seq_len + 1 - L, self.pad_id, dtype=np.int32)
            chunk = np.concatenate([chunk, pad])
        ids = torch.from_numpy(chunk[: self.seq_len + 1].copy()).long()
        return {"input_ids": ids[:-1], "labels": ids[1:]}


class LeoStreamDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 4096,
        pad_id: int = 65529,
        stride: int = 0,
    ):
        self.data    = np.load(data_path, mmap_mode="r").astype(np.int32)
        self.seq_len = max_seq_len
        self.pad_id  = pad_id
        self.stride  = stride if stride > 0 else max_seq_len

    def set_seq_len(self, seq_len: int) -> None:
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(1, (len(self.data) - self.seq_len) // self.stride)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        chunk = self.data[start : start + self.seq_len + 1]
        L     = len(chunk)
        if L < self.seq_len + 1:
            pad   = np.full(self.seq_len + 1 - L, self.pad_id, dtype=np.int32)
            chunk = np.concatenate([chunk, pad])
        ids = torch.from_numpy(chunk[: self.seq_len + 1].copy()).long()
        return {"input_ids": ids[:-1], "labels": ids[1:]}
