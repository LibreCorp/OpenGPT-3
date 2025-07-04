# attention.py

import torch, math
from torch import nn, Tensor
from typing import Optional, Tuple
from split_merge import split_heads, merge_heads
from masking import causal_mask, local_mask
from utils import stable_softmax

Past = Tuple[Tensor, Tensor]

class MultiHeadAttn(nn.Module):
    def __init__(self, n_head: int, local_window: int, attn_pdrop: float):
        super().__init__()
        self.n_head = n_head
        self.local_window = local_window
        self.attn_pdrop = attn_pdrop

    def forward(self,
                q: Tensor, k: Tensor, v: Tensor,
                past: Optional[Past], layer_idx: int,
                training: bool) -> Tuple[Tensor, Past]:
        if past is not None:
            pk, pv = past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v)

        dk = q.size(-1)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(dk)
        seq_len = scores.size(-1)
        if layer_idx % 2 == 0:
            mask = causal_mask(seq_len, q.device, scores.dtype)
        else:
            mask = local_mask(seq_len, self.local_window, q.device, scores.dtype)
        mask = mask.expand(scores.size(0), self.n_head, seq_len, seq_len)

        scores = scores.masked_fill(mask==0, -1e9)
        weights = stable_softmax(scores, axis=-1)
        if training and self.attn_pdrop>0.0:
            weights = nn.functional.dropout(weights, p=self.attn_pdrop)
        context = weights @ v
        return context, present

class AttentionLayer(nn.Module):
    def __init__(self, n_head: int, dims: int, dropout: float, local_window: int, **kwargs):
        super().__init__(**kwargs)
        self.proj_q = nn.Linear(dims, dims)
        self.proj_k = nn.Linear(dims, dims)
        self.proj_v = nn.Linear(dims, dims)
        # exact local_window from hparams
        self.attn = MultiHeadAttn(n_head, local_window, attn_pdrop=dropout)
        self.out  = nn.Linear(dims, dims)

    def forward(self,
                x: Tensor, past: Optional[Past],
                mask: Optional[Tensor],
                layer_idx: int,
                training: bool) -> Tuple[Tensor, Past]:
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = split_heads(q, self.attn.n_head), split_heads(k, self.attn.n_head), split_heads(v, self.attn.n_head)
        a_out, present = self.attn(q, k, v, past, layer_idx, training)
        a_out = merge_heads(a_out)
        return self.out(a_out), present
