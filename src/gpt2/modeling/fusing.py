# fusing.py

import torch
from torch import nn, Tensor

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean)**2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        return x_norm * self.g + self.b
