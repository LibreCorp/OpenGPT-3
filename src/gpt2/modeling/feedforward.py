import math
import torch
import torch.nn as nn

# ---- GELU (approximation) ----

def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x.pow(3))))

# ---- GPT-3 MLP Block ----

class MLP(nn.Module):
    """
    Implements the GPT-3 feed-forward:
      - project up to 4× hidden size via a 1D “conv” (here nn.Linear)
      - GELU activation (approximate)
      - project back down via another 1D “conv”
      - dropout on the output

    Args:
      n_embd:      model hidden size
      resid_pdrop: dropout rate after the second projection
    """
    def __init__(self, n_embd: int, resid_pdrop: float):
        super().__init__()
        self.n_state = n_embd * 4
        self.c_fc   = nn.Linear(n_embd,    self.n_state)
        self.c_proj = nn.Linear(self.n_state, n_embd)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, n_embd) or (..., n_embd)
        h = self.c_fc(x)
        h = gelu(h)
        h = self.c_proj(h)
        return self.dropout(h)
