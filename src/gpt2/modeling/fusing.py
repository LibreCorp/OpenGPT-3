# fusing.py

import torch
from torch import nn, Tensor

class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # weights not created until first forward
        self.g = None
        self.b = None

    def forward(self, x: Tensor) -> Tensor:
        if self.g is None:
            D = x.size(-1)
            self.g = nn.Parameter(torch.ones(D, device=x.device))
            self.b = nn.Parameter(torch.zeros(D, device=x.device))
        mean = x.mean(dim=-1, keepdim=True)
        var  = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        return x_norm * self.g + self.b
