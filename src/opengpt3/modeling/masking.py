# masking.py

import torch
from torch import Tensor

class PadMasking(torch.nn.Module):
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        shifted = torch.zeros(x.shape[:-1] + (1, offset), dtype=torch.bool, device=x.device)
        mask = torch.cat((shifted, is_pad), dim=-1)
        return mask.expand(x.shape + mask.shape[-1:])

class FutureMasking(torch.nn.Module):
    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        seq = x.size(-1)
        future = torch.ones(seq, seq+offset, dtype=torch.bool, device=x.device).triu(offset+1)
        mask = future.view((1,)* (x.ndim-1) + future.size())
        return mask.expand(x.shape + (future.size(-1),))

def causal_mask(q_len: int, k_len: int, device, dtype):
    m = torch.tril(torch.ones(q_len, k_len, dtype=dtype, device=device))
    return m.unsqueeze(0).unsqueeze(0)

def local_mask(q_len: int, k_len: int, window: int, device, dtype):
    idx_q = torch.arange(q_len, device=device)
    idx_k = torch.arange(k_len, device=device)
    diff = idx_k[None,:] - idx_q[:,None]
    m = (diff>=0) & (diff<=window)
    return m.to(dtype).unsqueeze(0).unsqueeze(0)
