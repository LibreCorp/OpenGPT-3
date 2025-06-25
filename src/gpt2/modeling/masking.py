import torch
import torch.nn as nn

class PadMasking(nn.Module):
    """
    Produces a mask of shape (..., seq_len_q, seq_len_k)
    where positions corresponding to pad_idx in the input are masked out.

    Args:
      pad_idx: the token ID used for padding
    """
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (..., seq_len_k) token IDs
        # offset: how many past positions were cached before x
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        # create a “past” column of False (no masking) of width=offset
        pad_col = torch.zeros(x.size()[:-1] + (1, offset),
                              dtype=torch.bool, device=x.device)
        # concatenate so that every pad position masks out its entire column
        mask = torch.cat((pad_col, is_pad), dim=-1)  
        # broadcast over query dim
        return mask.expand(x.shape[:-1] + (x.size(-1) + offset,))


class FutureMasking(nn.Module):
    """
    Produces a causal‐style future mask of shape (..., seq_len_q, seq_len_k),
    blocking any query from attending to future keys.

    Args:
      (none)
    """
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (..., seq_len_q) just to get the seq_len_q
        seq_len_q = x.size(-1)
        seq_len_k = seq_len_q + offset

        # make a (seq_len_q, seq_len_k) matrix where True = mask out future
        # we want upper triangle above the main diagonal shifted by offset
        future = torch.ones((seq_len_q, seq_len_k),
                            dtype=torch.bool, device=x.device).triu(offset+1)

        # reshape to (..., seq_len_q, seq_len_k)
        mask = future.view((1,)*(x.ndim-1) + future.size())
        return mask.expand(x.shape[:-1] + future.size())
