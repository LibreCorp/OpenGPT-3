# conv1d.py

import torch
from torch import nn, Tensor

class Conv1D(nn.Module):
    def __init__(self, nf: int, w_init_stdev: float = 0.02):
        super().__init__()
        self.nf = nf
        self.w = nn.Parameter(torch.randn(1,1) )  # placeholder
        self._w_init = {'std': w_init_stdev}

    def build(self, nx: int):
        # call once you know input dim
        self.w = nn.Parameter(torch.empty(nx, self.nf).normal_(std=self._w_init['std']))
        self.b = nn.Parameter(torch.zeros(self.nf))

    def forward(self, x: Tensor) -> Tensor:
        if not hasattr(self, 'b'):
            self.build(x.size(-1))
        bsz, seq, _ = x.size()
        x_flat = x.reshape(-1, x.size(-1))
        proj = x_flat @ self.w + self.b
        return proj.view(bsz, seq, self.nf)
