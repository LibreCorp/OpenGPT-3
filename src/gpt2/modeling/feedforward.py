# feedforward.py

import torch.nn as nn
from conv1d import Conv1D
from utils import gelu

class MLP(nn.Module):
    def __init__(self, n_embd: int, resid_pdrop: float):
        super().__init__()
        self.c_fc   = Conv1D(n_embd*4)
        self.c_proj = Conv1D(n_embd)
        self.pdrop  = resid_pdrop

    def forward(self, x, training: bool):
        h  = self.c_fc(x)
        h  = gelu(h)
        h2 = self.c_proj(h)
        if training and self.pdrop>0.0:
            h2 = nn.functional.dropout(h2, p=self.pdrop)
        return h2
