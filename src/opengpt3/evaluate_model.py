import argparse
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from opengpt3.modeling import Transformer
from opengpt3.evaluation import EvaluationSpec, EvaluateConfig, Evaluator
from typing import Dict, Optional


class opengpt3EvaluationSpec(EvaluationSpec):
    def __init__(
        self,
        dataset: str,
        tokenizer_path: str,
        seq_len: int,
        layers: int,
        heads: int,
        dims: int,
        rate: int,
        eval_split: str = 'validation',
        batch_eval: int = 16,
    ):
        self.dataset = dataset
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.dims = dims
        self.rate = rate
        self.eval_split = eval_split
        self.batch_eval = batch_eval

    def initialize(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.tokenizer_path, use_fast=True
        )
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def prepare_dataset(self):
        data_files = (
            {"train": self.dataset} if os.path.isfile(self.dataset) else None
        )
        raw_eval = load_dataset(
            "text" if data_files else self.dataset,
            data_files=data_files,
            split=self.eval_split,
            streaming=True,
        )

        def tokenize_fn(example):
            tok = self.tokenizer(
                example['text'], truncation=True, max_length=self.seq_len
            )
            ids = tok['input_ids']
            input_ids = ids[:-1]
            labels = ids[1:]
            pad_len = self.seq_len - len(ids)
            if pad_len > 0:
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                labels += [self.tokenizer.pad_token_id] * pad_len
            return {'input': torch.tensor(input_ids), 'output': torch.tensor(labels)}

        eval_ds = raw_eval.map(tokenize_fn)
        return DataLoader(eval_ds, batch_size=self.batch_eval)

    def construct_model(self, from_model: Optional[str] = None) -> nn.Module:
        if from_model:
            return Transformer.from_pretrained(from_model)
        hparams = {
            'n_vocab': self.tokenizer.vocab_size,
            'n_ctx': self.seq_len,
            'n_embd': self.dims,
            'n_layer': self.layers,
            'n_head': self.heads,
            'embd_pdrop': 0.0,
            'resid_pdrop': 0.0,
            'attn_pdrop': 0.0,
            'local_window': self.seq_len,
            'rate': self.rate,
            'layer_norm_epsilon': 1e-5,
        }
        return Transformer(hparams)

    def eval_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                       ) -> Dict[str, torch.Tensor]:
        logits, _ = model(data['input'], past=None, training=False)
        # Reshape logits to (batch_size, seq_len, vocab_size) and then transpose for CrossEntropyLoss
        logits = logits.view(data['input'].shape[0], data['input'].shape[1], -1).transpose(1, 2)
        loss = self.criterion(logits, data['output'])

        mask = (data['output'] != self.tokenizer.pad_token_id).float()
        loss = (loss * mask).sum() / mask.sum()
        perplexity = (loss.exp() * mask).sum() / mask.sum()

        return {'loss': loss, 'perplexity': perplexity}


def evaluate_opengpt3_model(args: argparse.Namespace):
    spec = opengpt3EvaluationSpec(
        dataset=args.dataset,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        layers=args.layers,
        heads=args.heads,
        dims=args.dims,
        rate=args.rate,
        eval_split=args.eval_split,
    )
    config = EvaluateConfig(
        batch_eval=args.batch_eval, total_steps=args.total_steps,
        use_gpu=args.use_gpu)

    print(Evaluator(spec, config).evaluate(from_model=args.model_path))


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('evaluate', help='evaluate GPT-2 model')

    parser.add_argument('--model_path', required=True,
                        help='trained GPT-2 model file path')

    group = parser.add_argument_group('Dataset and tokenizer')
    group.add_argument(
        '--dataset', required=True,
        help='HuggingFace dataset identifier or path to a local text file'
    )
    group.add_argument(
        '--eval_split', default='validation',
        help='split name to use for evaluation'
    )
    group.add_argument(
        '--tokenizer_path', required=True,
        help='pretrained tokenizer name or path for tokenization'
    )

    group = parser.add_argument_group('Model configurations')
    group.add_argument('--seq_len', default=64, type=int,
                       help='maximum sequence length')
    group.add_argument('--layers', default=12, type=int,
                       help='number of transformer layers')
    group.add_argument('--heads', default=16, type=int,
                       help='number of multi-heads in attention layer')
    group.add_argument('--dims', default=1024, type=int,
                       help='dimension of representation in each layer')
    group.add_argument('--rate', default=4, type=int,
                       help='increase rate of dimensionality in bottleneck')

    group = parser.add_argument_group('Evaluation options')
    group.add_argument('--batch_eval', default=64, type=int,
                       help='number of evaluation batch size')
    group.add_argument('--total_steps', default=-1, type=int,
                       help='number of total evaluation steps')
    group.add_argument('--use_gpu', action='store_true',
                       help='use gpu device in inferencing')

    parser.set_defaults(func=evaluate_opengpt3_model)
