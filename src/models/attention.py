#casual multi-head self-attention
#this implements the scaled-dot product attention with a causal mask so 
#each time step can only attend to itself and earlier steps-the core of the decoder-only architecture

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dropout:float=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:torch.Tensor,mask: torch.Tensor=None)->torch.Tensor:
        """
        x : (batch_size, seq_len, d_model)
        mask : optional boolean mask (True = allowed). If None, a causal
               lower‑triangular mask is created automatically.

        Returns (batch_size, seq_len, d_model)
        """
        B,T,C = x.shape
        q = self.W_q(x).view(B,T, self.n_heads, self.d_k).transpose(1,2) #(B,nH,T,dk)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (q@k.transpose(-2,-1)) / math.sqrt(self.d_k)

        if mask is None:
            mask = torch.tril(torch.ones(T,T, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.W_o(out)
