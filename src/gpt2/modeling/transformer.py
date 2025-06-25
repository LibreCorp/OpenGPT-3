import torch, numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from attention import AttentionLayer, Past
from conv1d import Conv1D
from feedforward import MLP
from fusing import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, hparams: dict, layer_idx: int):
        super().__init__()
        self.idx = layer_idx
        # compute initialization scale = 1/sqrt(n_layer)
        scale = 1.0 / np.sqrt(hparams['n_layer'])

        # lazy-build LayerNorm to match Keras behavior
        self.ln1 = LayerNorm()
        self.ln2 = LayerNorm()

        # self-attention projection (3*embd) as Conv1D
        self.c_attn = Conv1D(hparams['n_embd'], hparams['n_embd'] * 3)
        # scale its weights at init
        self.c_attn.w.data.mul_(scale)
        self.c_attn.b.data.mul_(scale)

        # propagate local_window from hparams
        self.attn = AttentionLayer(
            hparams['n_head'],
            hparams['n_embd'],
            hparams['attn_pdrop'],
            hparams['local_window']
        )

        # output projection as Conv1D
        self.c_proj = Conv1D(hparams['n_embd'], hparams['n_embd'])
        self.c_proj.w.data.mul_(scale)
        self.c_proj.b.data.mul_(scale)

        # feed-forward network
        self.mlp = MLP(hparams['n_embd'], hparams['resid_pdrop'])
        # scale MLP sublayer weights
        self.mlp.c_fc.w.data.mul_(scale)
        self.mlp.c_fc.b.data.mul_(scale)
        self.mlp.c_proj.w.data.mul_(scale)
        self.mlp.c_proj.b.data.mul_(scale)

        self.resid_pdrop = hparams['resid_pdrop']

    def forward(self, x: Tensor, past: Past, training: bool):
        # self-attention block
        ln1 = self.ln1(x)
        c = self.c_attn(ln1)
        q, k, v = c.split(ln1.size(-1), dim=-1)
        a, present = self.attn(q, k, v, past, self.idx, training)
        a = self.c_proj(a)
        if training and self.resid_pdrop > 0.0:
            a = F.dropout(a, p=self.resid_pdrop)
        x = x + a

        # feed-forward block
        ln2 = self.ln2(x)
        m = self.mlp(ln2, training)
        if training and self.resid_pdrop > 0.0:
            m = F.dropout(m, p=self.resid_pdrop)
        x = x + m
        return x, present

    @staticmethod
    def past_shape(hparams, batch_size=None, sequence=None):
        return [
            batch_size,
            hparams['n_layer'],
            2,
            hparams['n_head'],
            sequence,
            hparams['n_embd'] // hparams['n_head']
        ]

    @staticmethod
    def expand_tile(value, size):
        t = torch.as_tensor(value)
        ndims = t.dim()
        return t.unsqueeze(0).repeat(size, *([1] * ndims))

    @staticmethod
    def positions_for(tokens, past_length):
        batch_size = tokens.size(0)
        nsteps = tokens.size(1)
        seq = torch.arange(past_length, past_length + nsteps, device=tokens.device)
        return TransformerBlock.expand_tile(seq, batch_size)

class GPTModel(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.wte = nn.Parameter(torch.randn(hparams['n_vocab'], hparams['n_embd']) * 0.02)
        self.wpe = nn.Parameter(torch.randn(hparams['n_ctx'], hparams['n_embd']) * 0.01)
        self.drop = nn.Dropout(hparams['embd_pdrop'])
        self.blocks = nn.ModuleList([
            TransformerBlock(hparams, i)
            for i in range(hparams['n_layer'])
        ])
        # final normalization
        self.ln_f = LayerNorm()

    def forward(self, X: Tensor, past: list = None, training: bool = True):
        b, seq = X.size()
        device = X.device
        if past is None:
            pos = torch.arange(seq, device=device).unsqueeze(0).expand(b, -1)
        else:
            plen = past[0][0].size(2)
            pos = torch.arange(plen, plen + seq, device=device).unsqueeze(0).expand(b, -1)

        h = nn.functional.embedding(X, self.wte) + nn.functional.embedding(pos, self.wpe)
        h = self.drop(h)
        presents = []
        for i, block in enumerate(self.blocks):
            p = past[i] if past is not None else None
            h, pres = block(h, p, training)
            presents.append(pres)
        h = self.ln_f(h)
        logits = h.view(-1, h.size(-1)) @ self.wte.t()
        return (logits, presents) if not training else logits
