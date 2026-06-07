
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Expands dimension from d_model -> d_ff -> d_model.

    Parameters
    ----------
    d_model : int
        Input / output dimension.
    d_ff : int
        Inner (hidden) dimension.
    dropout : float
        Dropout probability applied after the activation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(nn.functional.gelu(self.linear1(x))))