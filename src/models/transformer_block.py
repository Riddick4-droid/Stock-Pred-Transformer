"""
Single Transformer Block (Pre‑LayerNorm style)

Combines causal multi‑head attention and a position‑wise feed‑forward network,
both wrapped with residual connections and pre‑layer normalization.
This is the core building block of the decoder‑only architecture.
"""

import torch
import torch.nn as nn
from src.models.attention import MultiHeadCausalAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed‑forward inner dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x : (batch_size, seq_len, d_model)
        mask : optional causal mask passed to attention.
        Returns same shape.
        """
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x