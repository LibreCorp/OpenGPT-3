# attention.py

import torch, math
from torch import nn, Tensor
from typing import Optional, Tuple
from .split_merge import split_heads, merge_heads
from .masking import causal_mask, local_mask, strided_mask, combine_masks
from .utils import softmax

Past = Tuple[Tensor, Tensor]

class MultiHeadAttn(nn.Module):
    def __init__(self, n_head: int, local_window: int, attn_pdrop: float, stride: int = 0, pattern: str = "dense"):
        super().__init__()
        self.n_head = n_head
        self.local_window = local_window
        self.stride = stride
        self.pattern = pattern  # "dense" or "sparse"
        self.attn_pdrop = attn_pdrop
        self.dropout = nn.Dropout(attn_pdrop)

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
    def __init__(self, n_head: int, dims: int, dropout: float, local_window: int,
                 stride: int = 0, pattern: str = "dense"):
        super().__init__()
        # use Q/K/V computed in TransformerBlock via c_attn
        self.attn = MultiHeadAttn(
            n_head,
            local_window,
            attn_pdrop=dropout,
            stride=stride,
            pattern=pattern,
        )
        self.pattern = pattern
        self.local_window = local_window
        self.stride = stride

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

    def _build_mask(self, q_len: int, k_len: int, layer_idx: int, device, dtype):
        """
        GPT-3 uses alternating dense and sparse attention following Sparse Transformer (Child et al., 2019).
        We implement "sparse" as the union of a local banded mask and an optional strided mask.
        - Even layers: dense causal mask
        - Odd layers: sparse (local [+ strided if stride>0]) causal mask
        """
        if self.pattern == "dense":
            return causal_mask(q_len, k_len, device, dtype)
        # alternating by layer index
        dense_layer = (layer_idx % 2 == 0)
        if dense_layer:
            return causal_mask(q_len, k_len, device, dtype)
        # sparse layer
        m_local = local_mask(q_len, k_len, max(self.local_window, 0), device, dtype) if self.local_window > 0 else None
        m_stride = strided_mask(q_len, k_len, self.stride, device, dtype) if self.stride and self.stride > 1 else None
        m = combine_masks(m_local, m_stride)
        # fallback to causal if nothing set
        return m if m is not None else causal_mask(q_len, k_len, device, dtype)
