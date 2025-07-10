import os
import json
import torch, numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from .attention import AttentionLayer, Past
from .split_merge import split_heads, merge_heads
from safetensors.torch import save_file, load_file

class TransformerBlock(nn.Module):
    def __init__(self, hparams: dict, layer_idx: int):
        super().__init__()
        self.hparams = hparams
        self.idx = layer_idx
        # compute initialization scale = 1/sqrt(n_layer)
        scale = 1.0 / np.sqrt(hparams['n_layer'])

        # layer normalization (pre-attention and pre-FFN)
        self.ln1 = nn.LayerNorm(hparams['n_embd'], eps=hparams.get('layer_norm_epsilon', 1e-5))
        self.ln2 = nn.LayerNorm(hparams['n_embd'], eps=hparams.get('layer_norm_epsilon', 1e-5))

        # self-attention projection: Q, K, V in one linear layer
        self.c_attn = nn.Linear(hparams['n_embd'], hparams['n_embd'] * 3)
        # scale weights to match original init
        self.c_attn.weight.data.mul_(scale)
        self.c_attn.bias.data.mul_(scale)

        # propagate local_window from hparams
        self.attn = AttentionLayer(
            hparams['n_head'],
            hparams['n_embd'],
            hparams['attn_pdrop'],
            hparams['local_window']
        )

        self.c_proj = nn.Linear(hparams['n_embd'], hparams['n_embd'])
        self.c_proj.weight.data.mul_(scale)
        self.c_proj.bias.data.mul_(scale)

        # feed-forward network (expand-project MLP)
        self.mlp = nn.Sequential(
            nn.Linear(hparams['n_embd'], hparams['n_embd'] * hparams['rate']),
            nn.GELU(),
            nn.Linear(hparams['n_embd'] * hparams['rate'], hparams['n_embd']),
            nn.Dropout(hparams['resid_pdrop']),
        )

        self.resid_pdrop = hparams['resid_pdrop']

    def forward(self, x: Tensor, past: Past, training: bool):
        # self-attention block
        ln1 = self.ln1(x)
        c = self.c_attn(ln1)
        q, k, v = c.split(ln1.size(-1), dim=-1)
        # apply attention (expects q/k/v split across heads)
        a, present = self.attn(
            split_heads(q, self.hparams['n_head']),
            split_heads(k, self.hparams['n_head']),
            split_heads(v, self.hparams['n_head']),
            past, self.idx, training
        )
        a = self.c_proj(a)
        if training and self.resid_pdrop > 0.0:
            a = F.dropout(a, p=self.resid_pdrop)
        x = x + a

        # feed-forward block
        ln2 = self.ln2(x)
        m = self.mlp(ln2)
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
        # final layer normalization
        self.ln_f = nn.LayerNorm(hparams['n_embd'], eps=hparams.get('layer_norm_epsilon', 1e-5))

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

    def save_pretrained(self, save_directory: str):
        """
        Save model configuration and weights in safetensors format.
        """
        os.makedirs(save_directory, exist_ok=True)
        # save config (include model_type for HF AutoConfig recognition)
        config_path = os.path.join(save_directory, 'config.json')
        cfg = self.hparams.copy()
        cfg['model_type'] = 'gpt2'
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        # save weights safely
        weights = {k: v.cpu() for k, v in self.state_dict().items()}
        save_file(weights, os.path.join(save_directory, 'model.safetensors'))

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """
        Load model configuration and weights from safetensors format.
        """
        config_path = os.path.join(load_directory, 'config.json')
        with open(config_path, 'r') as f:
            hparams = json.load(f)
        model = cls(hparams)
        state_dict = load_file(os.path.join(load_directory, 'model.safetensors'))
        model.load_state_dict(state_dict)
        return model
