# utils.py

import torch
import numpy as np
from torch import Tensor

def shape_list(x: Tensor):
    return list(x.shape) if isinstance(x.shape, torch.Size) else list(x.size())

def stable_softmax(x: Tensor, axis: int = -1) -> Tensor:
    x = x - x.max(dim=axis, keepdim=True).values
    ex = x.exp()
    return ex / ex.sum(dim=axis, keepdim=True)

def gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * x.pow(3))))
