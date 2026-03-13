"""
LeoSLM — Dual-Path Gated Attention
====================================
Single attention module with SHARED weights (W_Q, W_K, W_V, W_O)
that runs TWO paths simultaneously:
  - Path A: Causal (masked) attention  → AR generation
  - Path B: Bidirectional attention    → Diffusion grounding

The two outputs are later merged by the ConfidenceGate (confidence_gate.py).
Shared weights halve the param count vs. separate modules.

Key design choices:
  - GQA (Grouped Query Attention): num_kv_heads < num_heads → smaller KV cache
  - RoPE: rotary positional embeddings, applied to Q and K only
  - No bias on any projection (LLaMA3 style)
  - Flash-attention compatible shapes (B, H, T, D)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# RoPE (Rotary Positional Embedding)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for RoPE.
    Returns cos, sin each of shape (max_seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim//2,)
    positions = torch.arange(max_seq_len).float()                               # (T,)
    freqs = torch.outer(positions, theta)                                        # (T, head_dim//2)
    return freqs.cos(), freqs.sin()                                              # each (T, head_dim//2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to tensor x.
    x   : (B, num_heads, T, head_dim)
    cos : (T, head_dim // 2)
    sin : (T, head_dim // 2)
    """
    B, H, T, D = x.shape
    # split into two halves
    x1 = x[..., : D // 2]   # (B, H, T, D//2)
    x2 = x[..., D // 2 :]   # (B, H, T, D//2)

    # broadcast cos/sin to (1, 1, T, D//2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    # rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot


# ---------------------------------------------------------------------------
# Causal mask helper
# ---------------------------------------------------------------------------

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Upper-triangular mask filled with -inf for causal attention.
    Shape: (1, 1, T, T) — broadcastable over (B, H, T, T).
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, T, T)


# ---------------------------------------------------------------------------
# Core Dual-Path Attention
# ---------------------------------------------------------------------------

class DualPathAttention(nn.Module):
    """
    Dual-Path Attention block.

    Both causal and bidirectional paths share:
        W_Q, W_K, W_V, W_O

    The difference is ONLY the attention mask:
        - causal path  : upper-triangular -inf mask
        - bidir  path  : no mask (full attention)

    Args:
        hidden_dim     : model dimension (e.g. 512)
        num_heads      : number of query heads (e.g. 8)
        num_kv_heads   : number of key/value heads for GQA (e.g. 2)
        max_seq_len    : for RoPE precomputation (e.g. 512)
        dropout        : attention dropout (0 during inference)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.hidden_dim   = hidden_dim
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = hidden_dim // num_heads
        self.kv_head_dim  = hidden_dim // num_kv_heads
        self.groups       = num_heads // num_kv_heads   # GQA repeat factor
        self.scale        = math.sqrt(self.head_dim)
        self.dropout      = dropout

        # Shared projections — NO bias
        self.W_Q = nn.Linear(hidden_dim, num_heads    * self.head_dim,    bias=False)
        self.W_K = nn.Linear(hidden_dim, num_kv_heads * self.head_dim,    bias=False)
        self.W_V = nn.Linear(hidden_dim, num_kv_heads * self.head_dim,    bias=False)
        self.W_O = nn.Linear(num_heads  * self.head_dim, hidden_dim,      bias=False)

        # Precompute RoPE tables — register as buffer (moves with .to(device))
        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len + 16)
        self.register_buffer("rope_cos", cos)   # (max_seq_len, head_dim//2)
        self.register_buffer("rope_sin", sin)   # (max_seq_len, head_dim//2)

    # -----------------------------------------------------------------------
    def _project_qkv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project input to Q, K, V and reshape to (B, H, T, head_dim).
        K and V are expanded for GQA (repeated to match num_heads).
        """
        B, T, _ = x.shape

        # Q: (B, T, num_heads * head_dim) → (B, num_heads, T, head_dim)
        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: (B, T, num_kv_heads * head_dim) → (B, num_kv_heads, T, head_dim)
        K = self.W_K(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        Q = apply_rope(Q, self.rope_cos, self.rope_sin)
        K = apply_rope(K, self.rope_cos, self.rope_sin)

        # GQA: expand K and V from num_kv_heads → num_heads by repeating groups
        # (B, num_kv_heads, T, head_dim) → (B, num_heads, T, head_dim)
        K = K.repeat_interleave(self.groups, dim=1)
        V = V.repeat_interleave(self.groups, dim=1)

        return Q, K, V

    # -----------------------------------------------------------------------
    def _attend(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor],
        training: bool,
    ) -> torch.Tensor:
        """
        Core scaled dot-product attention.
        Q, K, V : (B, num_heads, T, head_dim)
        mask    : (1, 1, T, T) or None
        Returns : (B, T, hidden_dim)
        """
        B, H, T, D = Q.shape

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if mask is not None:
            scores = scores + mask                                   # add -inf to masked positions

        attn_weights = F.softmax(scores, dim=-1)                     # (B, H, T, T)

        if training and self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Weighted sum of values
        out = torch.matmul(attn_weights, V)                          # (B, H, T, D)

        # Reshape: (B, H, T, D) → (B, T, H*D)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)

        # Output projection
        return self.W_O(out)                                         # (B, T, hidden_dim)

    # -----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        ect_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass — returns BOTH causal and bidirectional outputs.
        The ConfidenceGate (external) decides how to merge them.

        Args:
            x          : (B, T, hidden_dim) — input sequence
            ect_tokens : (B, num_ect, hidden_dim) — epistemic tokens prepended
                         if provided, sequence becomes [ECT | x] for bidirectional path

        Returns:
            causal_out : (B, T, hidden_dim)  — AR path output
            bidir_out  : (B, T, hidden_dim)  — Diffusion path output
        """
        B, T, _ = x.shape

        # ── Bidirectional path ──────────────────────────────────────────────
        # Optionally prepend ECT tokens so they attend to full sequence
        if ect_tokens is not None:
            x_bidir = torch.cat([ect_tokens, x], dim=1)   # (B, T + num_ect, hidden_dim)
        else:
            x_bidir = x

        Q_b, K_b, V_b = self._project_qkv(x_bidir)
        bidir_full = self._attend(Q_b, K_b, V_b, mask=None, training=self.training)

        # If ECT was prepended, slice off ECT positions → keep only sequence positions
        if ect_tokens is not None:
            num_ect = ect_tokens.shape[1]
            bidir_out = bidir_full[:, num_ect:, :]         # (B, T, hidden_dim)
        else:
            bidir_out = bidir_full                         # (B, T, hidden_dim)

        # ── Causal path ─────────────────────────────────────────────────────
        causal_mask = build_causal_mask(T, x.device)       # (1, 1, T, T)
        Q_c, K_c, V_c = self._project_qkv(x)
        causal_out = self._attend(Q_c, K_c, V_c, mask=causal_mask, training=self.training)

        return causal_out, bidir_out


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, D = 2, 16, 512
    x = torch.randn(B, T, D)
    ect = torch.randn(B, 4, D)   # 4 epistemic tokens

    attn = DualPathAttention(
        hidden_dim=512,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=512,
    )

    causal_out, bidir_out = attn(x, ect_tokens=ect)

    print(f"Input shape      : {x.shape}")
    print(f"Causal out shape : {causal_out.shape}")   # expect (2, 16, 512)
    print(f"Bidir  out shape : {bidir_out.shape}")    # expect (2, 16, 512)
    print(f"Param count      : {sum(p.numel() for p in attn.parameters()):,}")
    print("✅ dual_attention.py OK")
