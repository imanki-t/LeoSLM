"""
LeoSLM — Epistemic Confidence Tokens (ECT)
============================================
The core anti-hallucination innovation of LeoSLM.

4 learnable tokens are injected into every forward pass.
They attend to the FULL sequence bidirectionally and produce
a per-token uncertainty score U ∈ [0, 1] for every position.

How it works:
    1. ECTs are concatenated to input for the bidirectional path
    2. After all LeoBlocks, ECTs are extracted from the hidden states
    3. A small MLP aggregates 4 ECT vectors → 1 uncertainty score per token
    4. That score gates the confidence_gate (α) and drives the calibration loss

Design:
    - ECT embeddings: nn.Parameter (4, hidden_dim), trained end-to-end
    - Cross-attention: ECTs attend to sequence to gather evidence
    - Score MLP: 2-layer MLP → scalar per position
    - Output: U ∈ [0,1]^T (after sigmoid)

Why 4 tokens?
    Think of them as 4 "doubt neurons" that specialize:
    one for factual doubt, one for logical doubt, etc.
    (they learn their own specialization, not manually assigned)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class ECTCrossAttention(nn.Module):
    """
    Lightweight cross-attention: ECTs (queries) attend to sequence (keys/values).
    This lets each ECT token gather evidence from the whole sequence.

    Args:
        hidden_dim : model dimension
        num_heads  : number of attention heads (can be smaller than main model)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        # ECT tokens are queries, sequence tokens are keys/values
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        ect_hidden: torch.Tensor,    # (B, num_ect, hidden_dim)  — ECT hidden states
        seq_hidden: torch.Tensor,    # (B, T, hidden_dim)        — sequence hidden states
    ) -> torch.Tensor:
        """
        Returns updated ECT representations after attending to sequence.
        Output: (B, num_ect, hidden_dim)
        """
        B, E, D = ect_hidden.shape
        _, T, _ = seq_hidden.shape
        H = self.num_heads
        Dh = self.head_dim

        # Q from ECTs, K/V from sequence
        Q = self.W_Q(ect_hidden).view(B, E, H, Dh).transpose(1, 2)   # (B, H, E, Dh)
        K = self.W_K(seq_hidden).view(B, T, H, Dh).transpose(1, 2)   # (B, H, T, Dh)
        V = self.W_V(seq_hidden).view(B, T, H, Dh).transpose(1, 2)   # (B, H, T, Dh)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale    # (B, H, E, T)
        attn   = F.softmax(scores, dim=-1)                             # (B, H, E, T)
        out    = torch.matmul(attn, V)                                 # (B, H, E, Dh)

        out = out.transpose(1, 2).contiguous().view(B, E, D)          # (B, E, D)
        return self.W_O(out)


class EpistemicTokens(nn.Module):
    """
    Full ECT module.

    Responsibilities:
        1. Hold learnable ECT embedding parameters
        2. Perform cross-attention: ECTs attend to sequence hidden states
        3. Aggregate 4 ECT vectors per position → scalar uncertainty score
        4. Output U ∈ [0,1]^T for every sequence position

    The uncertainty score U[i] answers: "How unsure is the model about token i?"
        - U[i] close to 0 → model is confident → AR head trusted
        - U[i] close to 1 → model is uncertain → Diffusion refinement triggered

    Args:
        hidden_dim : model dimension (e.g. 512)
        num_ect    : number of epistemic tokens (default 4)
        num_heads  : heads for ECT cross-attention
    """

    def __init__(self, hidden_dim: int, num_ect: int = 4, num_heads: int = 4):
        super().__init__()
        self.num_ect    = num_ect
        self.hidden_dim = hidden_dim

        # Learnable ECT embeddings — initialized with small random values
        self.ect_embeddings = nn.Parameter(
            torch.randn(num_ect, hidden_dim) * 0.02
        )   # (num_ect, hidden_dim)

        # RMSNorm for ECT hidden states
        self.ect_norm = nn.RMSNorm(hidden_dim)

        # Cross-attention: ECTs → sequence
        self.cross_attn = ECTCrossAttention(hidden_dim, num_heads)

        # Post cross-attention norm
        self.post_attn_norm = nn.RMSNorm(hidden_dim)

        # Score MLP: projects (num_ect * hidden_dim) → T uncertainty scores
        # But we want per-POSITION scores, so we do: for each position i,
        # we attend ECTs → position i → get a score.
        # Simpler: project each ECT hidden state → scalar, then mean over ECTs.
        self.score_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1, bias=False),
        )

        # Position-wise uncertainty: ECT hidden states × sequence → uncertainty
        # We use another small cross-attention in reverse: seq attends to ECTs
        self.uncertainty_attn_Q = nn.Linear(hidden_dim, hidden_dim // 4, bias=False)
        self.uncertainty_attn_K = nn.Linear(hidden_dim, hidden_dim // 4, bias=False)
        self.uncertainty_attn_V = nn.Linear(hidden_dim, 1,               bias=False)

    def get_ect_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Expand ECT embeddings for a batch.
        Returns (B, num_ect, hidden_dim).
        """
        return self.ect_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        seq_hidden: torch.Tensor,    # (B, T, hidden_dim) — final layer sequence hidden states
        ect_hidden: torch.Tensor,    # (B, num_ect, hidden_dim) — ECT hidden states from last layer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token uncertainty scores.

        Args:
            seq_hidden : sequence hidden states from last LeoBlock  (B, T, D)
            ect_hidden : ECT hidden states extracted from last block (B, num_ect, D)

        Returns:
            uncertainty : (B, T) — uncertainty score per token, values in [0, 1]
            ect_updated : (B, num_ect, D) — updated ECT states for logging/loss
        """
        B, T, D = seq_hidden.shape

        # Normalize ECT hidden states
        ect_norm = self.ect_norm(ect_hidden)                           # (B, num_ect, D)

        # ECTs attend to sequence to gather evidence
        ect_updated = ect_norm + self.cross_attn(ect_norm, seq_hidden) # (B, num_ect, D)
        ect_updated = self.post_attn_norm(ect_updated)                 # (B, num_ect, D)

        # ── Compute per-position uncertainty ────────────────────────────────
        # For each sequence position i, score how much uncertainty ECTs signal
        # Mechanism: lightweight attention where seq positions query ECT states

        # Q from sequence: (B, T, D//4)
        Q_u = self.uncertainty_attn_Q(seq_hidden)
        # K from ECTs:     (B, num_ect, D//4)
        K_u = self.uncertainty_attn_K(ect_updated)
        # V from ECTs:     (B, num_ect, 1)
        V_u = self.uncertainty_attn_V(ect_updated)

        # Scores: (B, T, num_ect)
        scale  = math.sqrt(Q_u.shape[-1])
        scores = torch.matmul(Q_u, K_u.transpose(-2, -1)) / scale

        # Softmax over ECT dim: each position attends over the 4 ECTs
        attn_w = F.softmax(scores, dim=-1)                             # (B, T, num_ect)

        # Weighted sum of ECT scalar values: (B, T, 1)
        uncertainty_raw = torch.matmul(attn_w, V_u)                   # (B, T, 1)
        uncertainty     = torch.sigmoid(uncertainty_raw.squeeze(-1))   # (B, T) ∈ [0,1]

        return uncertainty, ect_updated


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, D = 2, 32, 512
    num_ect = 4

    ect_module = EpistemicTokens(hidden_dim=D, num_ect=num_ect)

    # Simulate final layer outputs
    seq_hidden = torch.randn(B, T, D)
    ect_hidden = torch.randn(B, num_ect, D)

    uncertainty, ect_updated = ect_module(seq_hidden, ect_hidden)

    print(f"Sequence hidden : {seq_hidden.shape}")
    print(f"ECT hidden      : {ect_hidden.shape}")
    print(f"Uncertainty     : {uncertainty.shape}")           # (2, 32)
    print(f"ECT updated     : {ect_updated.shape}")           # (2, 4, 512)
    print(f"Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")  # should be [0,1]
    print(f"Param count     : {sum(p.numel() for p in ect_module.parameters()):,}")

    # Test get_ect_tokens
    init_ect = ect_module.get_ect_tokens(batch_size=B, device=torch.device("cpu"))
    print(f"Init ECT tokens : {init_ect.shape}")              # (2, 4, 512)
    print("✅ ect.py OK")
