# conv1d.py

import torch
from torch import nn, Tensor

class Conv1D(nn.Module):
    def __init__(self, nx: int, nf: int, w_init_stdev: float = 0.02):
        super().__init__()
        self.nf = nf
        # immediately allocate exactly like Keras build:
        self.w = nn.Parameter(torch.empty(nx, nf).normal_(std=w_init_stdev))
        self.b = nn.Parameter(torch.zeros(nf))

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq, _ = x.size()
        x_flat = x.view(-1, x.size(-1))
        proj = x_flat @ self.w + self.b
        return proj.view(bsz, seq, self.nf)
