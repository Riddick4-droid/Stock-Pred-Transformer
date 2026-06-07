"""
Stock Transformer – Top‑Level Model

Assembles the input projection, learned positional embeddings, a stack of
causal Transformer blocks, and three output heads (price, direction, volatility).

This is the only model class the rest of the project imports.
It also supports LoRA injection via the inject_lora function from .lora.
"""

import torch
import torch.nn as nn
from src.models.positional_encoding import LearnedPositionalEmbedding
from src.models.transformer_block import TransformerBlock


class StockTransformer(nn.Module):
    """
    Decoder‑only Transformer for financial time series.
    Parameters
    num_features : int
        Number of input features (after engineering).
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer blocks.
    d_ff : int
        Feed‑forward inner dimension.
    max_seq_len : int
        Maximum sequence length (for positional embeddings).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,num_features: int,d_model: int = 512,n_heads: int = 16,n_layers: int = 8,d_ff: int = 2048,
        max_seq_len: int = 252,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features

        # Input projection (replaces token embedding)
        self.input_proj = nn.Linear(num_features, d_model)

        # Learned positional embeddings
        self.pos_embed = LearnedPositionalEmbedding(max_seq_len, d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm before output heads
        self.final_ln = nn.LayerNorm(d_model)

        #Output Heads
        #
        #  
        # Price head: mu and logvar for Gaussian NLL
        self.price_mu = nn.Linear(d_model, 1)
        self.price_logvar = nn.Linear(d_model, 1)

        # Direction head: 3 classes (Down, Flat, Up)
        self.direction_head = nn.Linear(d_model, 3)

        # Volatility head: real scalar (realised vol)
        self.volatility_head = nn.Linear(d_model, 1)

        # Weight initialisation
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        x : torch.Tensor
            (batch_size, seq_len, num_features) – already scaled features.
        Returns
        mu : torch.Tensor (batch_size,)
            Predicted next‑day log return.
        logvar : torch.Tensor (batch_size,)
            Predicted log variance of the log return.
        dir_logits : torch.Tensor (batch_size, 3)
            Logits for direction classification.
        vol_pred : torch.Tensor (batch_size,)
            Predicted next‑day realised volatility.
        """
        B, T, _ = x.shape

        # Linear projection to d_model
        h = self.input_proj(x)                                # (B, T, d_model)

        # Add learned positional embeddings
        h = h + self.pos_embed(T, x.device)                  # (B, T, d_model)

        # Causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()

        # Pass through transformer blocks
        for block in self.blocks:
            h = block(h, causal_mask)

        # Final layer norm
        h = self.final_ln(h)

        # Use only the last time step (most recent day) for predictions
        last_h = h[:, -1, :]   # (B, d_model)

        # Output heads
        mu = self.price_mu(last_h).squeeze(-1)           # (B,)
        logvar = self.price_logvar(last_h).squeeze(-1)   # (B,)
        dir_logits = self.direction_head(last_h)         # (B, 3)
        vol = self.volatility_head(last_h).squeeze(-1)   # (B,)

        return mu, logvar, dir_logits, vol
    
if __name__ == "__main__":
    from src.utils import get_model_info
    model = StockTransformer(num_features=36)
    model_info = get_model_info(model)
    for key,value in model_info.items():
        print(f"{key}:{value:_}")