# split_merge.py

import torch
from torch import Tensor

def split_heads(x: Tensor, n_head: int) -> Tensor:
    b, seq, dim = x.size()
    head_dim = dim // n_head
    x = x.view(b, seq, n_head, head_dim)
    return x.permute(0,2,1,3)

def merge_heads(x: Tensor) -> Tensor:
    # x: (b, n_head, seq, head_dim)
    b, h, seq, hd = x.size()
    x = x.permute(0,2,1,3).contiguous()
    return x.view(b, seq, h*hd)
