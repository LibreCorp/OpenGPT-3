# attention.py

import torch, math
from torch import nn, Tensor
from typing import Optional, Tuple
from .split_merge import split_heads, merge_heads
from .masking import causal_mask, local_mask
from .utils import softmax

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
        q_len, k_len = q.size(-2), k.size(-2)
        if layer_idx % 2 == 0:
            mask = causal_mask(q_len, k_len, q.device, scores.dtype)
        else:
            mask = local_mask(q_len, k_len, self.local_window, q.device, scores.dtype)

        scores = scores.masked_fill(mask==0, -1e9)
        weights = softmax(scores, axis=-1)
        if training and self.attn_pdrop>0.0:
            weights = nn.functional.dropout(weights, p=self.attn_pdrop)
        context = weights @ v
        return context, present

class AttentionLayer(nn.Module):
    def __init__(self, n_head: int, dims: int, dropout: float, local_window: int, **kwargs):
        super().__init__(**kwargs)
        # use Q/K/V computed in TransformerBlock via c_attn
        self.attn = MultiHeadAttn(n_head, local_window, attn_pdrop=dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        past: Optional[Past],
        layer_idx: int,
        training: bool,
    ) -> Tuple[Tensor, Past]:
        a_out, present = self.attn(q, k, v, past, layer_idx, training)
        a_out = merge_heads(a_out)
        return a_out, present
