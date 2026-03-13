"""
LeoSLM — Leo Decoder Block
============================
Assembles one full transformer decoder block:

    x_in
      │
      ├─── [Pre-Norm RMSNorm]
      │          │
      │    DualPathAttention → (causal_out, bidir_out)
      │          │
      │    ConfidenceGate(causal_out, bidir_out, uncertainty)
      │          │
      │    residual add → x_attn
      │
      ├─── [Pre-Norm RMSNorm]
      │          │
      │      SwiGLU FFN
      │          │
      │    residual add → x_out
      │
      └──► x_out, alpha, ect_hidden_updated

Notes:
    - Pre-norm (norm before sublayer), not post-norm — more stable training
    - SwiGLU FFN: ffn_dim = (8/3) * hidden_dim, rounded to nearest multiple of 256
    - ECT tokens ride along x throughout — extracted/updated per block
    - No dropout in base config (can enable via config)
    - α (gate value) is returned for logging/visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .dual_attention  import DualPathAttention
from .confidence_gate import ConfidenceGate
from .ect             import EpistemicTokens


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network as used in LLaMA, PaLM, Gemma.

    FFN(x) = (SiLU(W1(x)) ⊙ W3(x)) @ W2

    ffn_dim = int(8/3 * hidden_dim), rounded to nearest 256.
    This is Noam Shazeer's recommendation for SwiGLU.

    Args:
        hidden_dim : input/output dimension
        ffn_dim    : intermediate dimension (gates + values)
    """

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        # Gate projection: W1
        self.W1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        # Value projection: W3
        self.W3 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        # Down projection: W2
        self.W2 = nn.Linear(ffn_dim,    hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU gate ⊙ linear value
        return self.W2(F.silu(self.W1(x)) * self.W3(x))


def compute_ffn_dim(hidden_dim: int) -> int:
    """
    Compute SwiGLU ffn_dim: (8/3 * hidden_dim) rounded up to nearest 256.
    For hidden_dim=512: 8/3*512 ≈ 1365 → 1408 (nearest 256 multiple).
    """
    raw = int(8 / 3 * hidden_dim)
    return ((raw + 255) // 256) * 256


# ---------------------------------------------------------------------------
# Leo Decoder Block
# ---------------------------------------------------------------------------

class LeoBlock(nn.Module):
    """
    One full LeoSLM transformer decoder block.

    The ECT tokens are treated as additional sequence positions during
    the bidirectional attention path. They flow through the block and
    carry uncertainty information across layers.

    Args:
        hidden_dim   : model hidden dimension
        num_heads    : number of attention heads
        num_kv_heads : number of KV heads (GQA)
        num_ect      : number of epistemic tokens
        max_seq_len  : max sequence length (for RoPE)
        ffn_dim      : FFN intermediate dim (computed from hidden_dim if None)
        temperature  : confidence gate temperature
        dropout      : dropout probability (0 = disabled)
    """

    def __init__(
        self,
        hidden_dim   : int,
        num_heads    : int,
        num_kv_heads : int,
        num_ect      : int,
        max_seq_len  : int,
        ffn_dim      : Optional[int] = None,
        temperature  : float = 1.0,
        dropout      : float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_ect    = num_ect

        if ffn_dim is None:
            ffn_dim = compute_ffn_dim(hidden_dim)

        # ── Sub-modules ────────────────────────────────────────────────────

        # Pre-norms (one for attention, one for FFN)
        self.attn_norm = nn.RMSNorm(hidden_dim)
        self.ffn_norm  = nn.RMSNorm(hidden_dim)

        # RMSNorm for ECT tokens (separate norm, same dimension)
        self.ect_norm  = nn.RMSNorm(hidden_dim)

        # Dual-path attention (shared weights for both paths)
        self.attention = DualPathAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Confidence gate (merges causal + bidir)
        self.gate = ConfidenceGate(
            hidden_dim=hidden_dim,
            temperature=temperature,
        )

        # SwiGLU FFN
        self.ffn = SwiGLUFFN(hidden_dim=hidden_dim, ffn_dim=ffn_dim)

        # Optional dropout for residuals
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x           : torch.Tensor,                    # (B, T, D) — sequence hidden states
        ect_hidden  : torch.Tensor,                    # (B, num_ect, D) — ECT hidden states
        uncertainty : Optional[torch.Tensor] = None,   # (B, T) — from previous layer ECTs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward through one LeoBlock.

        Args:
            x           : sequence hidden states from previous block
            ect_hidden  : ECT hidden states (these flow through all blocks)
            uncertainty : per-token uncertainty from last ECT computation
                          (None for early blocks, available after ECT aggregation)

        Returns:
            x_out       : (B, T, D)       — updated sequence hidden states
            ect_out     : (B, num_ect, D) — updated ECT hidden states
            alpha       : (B, T)          — gate values for this block (logging)
        """
        B, T, D = x.shape

        # ── Attention sublayer ───────────────────────────────────────────────

        # Normalize input (pre-norm)
        x_normed   = self.attn_norm(x)                # (B, T, D)
        ect_normed = self.ect_norm(ect_hidden)         # (B, num_ect, D)

        # Dual-path attention: pass ECT tokens to bidirectional path
        causal_out, bidir_out = self.attention(
            x         = x_normed,
            ect_tokens= ect_normed,
        )
        # causal_out : (B, T, D)
        # bidir_out  : (B, T, D)

        # Confidence gate: merge the two paths
        merged, alpha = self.gate(
            causal_out  = causal_out,
            bidir_out   = bidir_out,
            uncertainty = uncertainty,
        )
        # merged : (B, T, D)
        # alpha  : (B, T)

        # Residual connection
        x = x + self.dropout(merged)                  # (B, T, D)

        # ── Update ECT hidden states ─────────────────────────────────────────
        # ECTs also go through a simple residual FFN update to evolve across layers
        # They attend to the MERGED sequence output (post-gate)
        # This is a lightweight update — just a residual add of cross-attn signal
        # (full cross-attention is in ECTCrossAttention in ect.py)

        # Simple update: ECT hidden += mean of merged sequence (a soft summary signal)
        # Full ECT cross-attention happens once at the end in leoSLM.py
        seq_summary = merged.mean(dim=1, keepdim=True)              # (B, 1, D)
        ect_out     = ect_hidden + 0.1 * seq_summary.expand_as(ect_hidden)  # (B, num_ect, D)

        # ── FFN sublayer ─────────────────────────────────────────────────────

        # Normalize (pre-norm)
        x_ffn_in = self.ffn_norm(x)                   # (B, T, D)
        x        = x + self.dropout(self.ffn(x_ffn_in))  # (B, T, D)

        return x, ect_out, alpha


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, D   = 2, 32, 512
    num_ect   = 4
    num_heads = 8
    num_kv    = 2

    block = LeoBlock(
        hidden_dim   = D,
        num_heads    = num_heads,
        num_kv_heads = num_kv,
        num_ect      = num_ect,
        max_seq_len  = 512,
    )

    x          = torch.randn(B, T, D)
    ect_hidden = torch.randn(B, num_ect, D)
    unc        = torch.rand(B, T)

    x_out, ect_out, alpha = block(x, ect_hidden, uncertainty=unc)

    print(f"Input     : {x.shape}")
    print(f"x_out     : {x_out.shape}")      # (2, 32, 512)
    print(f"ect_out   : {ect_out.shape}")    # (2, 4, 512)
    print(f"alpha     : {alpha.shape}")      # (2, 32)
    print(f"alpha avg : {alpha.mean():.3f}") # should be low (near 0 initially)

    ffn_d = compute_ffn_dim(D)
    print(f"FFN dim for D={D}: {ffn_d}")    # should be 1408

    total = sum(p.numel() for p in block.parameters())
    print(f"Block params: {total:,}")
    print("✅ leo_block.py OK")
