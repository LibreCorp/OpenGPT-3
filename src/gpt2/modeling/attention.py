import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

Past = Tuple[torch.Tensor, torch.Tensor]

class AttentionLayer(nn.Module):
    """
    Args:
      heads:         number of attention heads
      dims:          model dimensionality
      local_window:  window size for local attention on odd layers
      layer_idx:     layer index (0-based): even→causal, odd→local
      dropout:       attention-score dropout

    Inputs:
      q:    (..., query_len, dims)
      k:    (..., kv_len,    dims)
      v:    (..., kv_len,    dims)
      past: optional tuple of (k, v) with shapes (..., past_len, dims)

    Outputs:
      out:    (..., query_len, dims)
      present:(k_cat, v_cat)
    """
    def __init__(self, heads: int, dims: int, local_window: int, layer_idx: int, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.dims = dims
        self.local_window = local_window
        self.layer_idx = layer_idx

        self.attn = MultiHeadAttention(heads, dropout)
        self.proj_q = nn.Linear(dims, dims)
        self.proj_k = nn.Linear(dims, dims)
        self.proj_v = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                past: Optional[Past] = None
                ) -> Tuple[torch.Tensor, Past]:
        # project
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        # append past keys/values if present
        if past is not None:
            pk, pv = past
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)

        # build mask: True = mask out
        seq_k = k.size(-2)
        seq_q = q.size(-2)
        past_len = seq_k - seq_q
        device = q.device

        if (self.layer_idx % 2) == 0:
            # causal: disallow j > i_global
            # build global causal once, then slice
            global_mask = ~torch.tril(torch.ones(seq_k, seq_k, dtype=torch.bool, device=device))
        else:
            # local window: disallow positions j where i_global - j not in [0, window]
            idxs = torch.arange(seq_k, device=device)
            diff = idxs.unsqueeze(1) - idxs.unsqueeze(0)  # i_global - j_global
            allowed = (diff >= 0) & (diff <= self.local_window)
            global_mask = ~allowed

        # slice only rows for the current queries
        mask = global_mask[past_len:, :]               # shape (seq_q, seq_k)
        mask = mask.unsqueeze(0)                       # shape (1, seq_q, seq_k)
        # `MultiHeadAttention` will unsqueeze for heads and broadcast to batch

        # compute attention
        attn_out = self.attn(q, k, v, mask)            # (..., seq_q, dims)
        out = self.out_proj(attn_out)
        return out, (k, v)
