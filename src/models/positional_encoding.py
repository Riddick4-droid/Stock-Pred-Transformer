"""
Learned Positional Embedding

Instead of fixed sinusoidal encodings, we learn a unique embedding vector
for each absolute position (up to a maximum sequence length).
This is the approach used in GPT‑2 and gives the model flexibility
to learn position‑dependent patterns in stock data.
"""

import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """
    Lookup table of learned positional embeddings.

    Parameters
    ----------
    max_seq_len : int
        Maximum number of days the model can handle.
    d_model : int
        Embedding dimension (must match model dimension).
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Return positional embeddings for a sequence of length `seq_len`.
        Parameters
        seq_len : int
            Actual sequence length (must be ≤ max_seq_len).
        device : torch.device
            Device to place the tensor on.
        Returns
        torch.Tensor of shape (1, seq_len, d_model).
        """
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        return self.pos_embed(positions)                                   # (1, seq_len, d_model)