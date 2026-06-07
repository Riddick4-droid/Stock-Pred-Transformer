"""
Low‑Rank Adaptation (LoRA) for the Stock Transformer

Provides:
- LoRALayer: wrapper that adds a low‑rank update to a frozen linear layer.
- inject_lora: replaces attention projection matrices with LoRA‑wrapped versions.
"""

import math
import torch
import torch.nn as nn
from .attention import MultiHeadCausalAttention


class LoRALayer(nn.Module):
    """
    Wraps a linear layer to apply a low‑rank adaptation:
        output = original(x) + (alpha / r) * (x @ A @ B)
    The original weights are frozen; only A and B are trained.
    Parameters
    original : nn.Linear
        The linear layer to adapt.
    r : int
        Rank of the low‑rank decomposition.
    alpha : float
        Scaling factor (the update is multiplied by alpha / r).
    dropout : float
        Dropout applied to the input before the low‑rank path.
    """

    def __init__(self, original: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze original layer
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features

        # Low‑rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.lora_dropout = nn.Dropout(dropout)

        # Initialisation: A normal, B zero -> update starts at zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_out = self.original(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A) @ self.lora_B
        return frozen_out + lora_out * self.scaling


def inject_lora(
    model: nn.Module,
    target_modules: list = None,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05
) -> nn.Module:
    """
    Walk through the model and replace attention linear layers with LoRALayer.
    Parameters
    model : nn.Module
        The StockTransformer (or any model containing MultiHeadCausalAttention).
    target_modules : list of str, optional
        Which attribute names to wrap, e.g. ["W_q","W_k","W_v","W_o"].
    r, alpha, dropout : LoRA hyperparameters.
    Returns
    model : nn.Module
        The same model with LoRA injected (modified in‑place).
    """
    if target_modules is None:
        target_modules = ["W_q", "W_k", "W_v", "W_o"]

    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            attr = name.split('.')[-1]
            if attr in target_modules:
                parent_name = '.'.join(name.split('.')[:-1])
                parent = dict(model.named_modules()).get(parent_name)
                if parent is not None and isinstance(parent, MultiHeadCausalAttention):
                    replacements[name] = LoRALayer(module, r, alpha, dropout)

    for name, lora_layer in replacements.items():
        parent_name = '.'.join(name.split('.')[:-1])
        attr = name.split('.')[-1]
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, attr, lora_layer)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA injected (r={r}, alpha={alpha}). "
          f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model