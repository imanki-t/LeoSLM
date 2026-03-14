"""
data/dataset.py — LeoDataset
==============================
Memory-mapped uint32 dataset with curriculum-aware sequence length.

LeoTokenizer vocab size 65543 exceeds uint16 max (65535), so the
tokenised corpus is stored as uint32 (.npy files, 4 bytes/token).

The dataset supports Progressive Confidence Curriculum (PCC): each
training phase calls set_seq_len() to increase the context window
from 4k → 8k → 16k → 32k without reloading data from disk.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict


class LeoDataset(Dataset):
    """
    Memory-mapped, curriculum-length dataset.

    Args:
        path       : path to uint32 .npy token file
        max_seq_len: initial sequence length (updated per phase via set_seq_len)
        pad_id     : pad token id (default: LeoConfig.pad_id = 65529)
    """

    def __init__(
        self,
        path:        str,
        max_seq_len: int = 4096,
        pad_id:      int = 65529,
    ):
        self.data     = np.load(path, mmap_mode="r")
        self.max_len  = max_seq_len
        self.pad_id   = pad_id
        self.n_chunks = max(1, len(self.data) // max_seq_len)

    def set_seq_len(self, n: int):
        """Update sequence length for the next training phase (PCC)."""
        self.max_len  = n
        self.n_chunks = max(1, len(self.data) // n)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s     = idx * self.max_len
        chunk = self.data[s: s + self.max_len].astype(np.int64)

        # Pad if last chunk is shorter than max_len
        if len(chunk) < self.max_len:
            pad   = np.full(self.max_len - len(chunk), self.pad_id, dtype=np.int64)
            chunk = np.concatenate([chunk, pad])

        ids  = torch.tensor(chunk, dtype=torch.long)
        mask = (ids != self.pad_id).long()
        return {"input_ids": ids, "attention_mask": mask}
