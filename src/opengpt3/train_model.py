import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from .utils import fusing
from .modeling import Transformer
from .training import TrainConfig, TrainingSpec, Trainer
from typing import Tuple, Iterator, Dict


class opengpt3TrainingSpec(TrainingSpec):
    def __init__(
        self,
        dataset: str,
        tokenizer_path: str,
        seq_len: int,
        layers: int,
        heads: int,
        dims: int,
        rate: int,
        dropout: float,
        base_lr: float,
        wd_rate: float,
        total_steps: int,
        use_grad_ckpt: bool,
        train_split: str = 'train',
        eval_split: str = 'validation',
    ):
        self.dataset = dataset
        self.train_split = train_split
        self.eval_split = eval_split
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.dims = dims
        self.rate = rate
        self.dropout = dropout
        self.base_lr = base_lr
        self.wd_rate = wd_rate
        self.total_steps = total_steps
        self.use_grad_ckpt = use_grad_ckpt

    def initialize(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.tokenizer_path, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
                if getattr(self.tokenizer, 'eos_token_id', None) is not None
                else 0
            )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id, reduction='mean'
        )

    def prepare_datasets(self):
        data_files = (
            {"train": self.dataset}
            if os.path.isfile(self.dataset)
            else None
        )
        raw_train = load_dataset(
            "text" if data_files else self.dataset,
            data_files=data_files,
            split=self.train_split,
            streaming=True,
        )
        raw_eval = load_dataset(
            "text" if data_files else self.dataset,
            data_files=data_files,
            split=self.eval_split,
            streaming=True,
        )

        def tokenize_fn(example):
            tok = self.tokenizer(
                example["text"], truncation=True, max_length=self.seq_len
            )
            ids = tok["input_ids"]
            # for causal LM
            input_ids = ids[:-1]
            labels = ids[1:]
            pad_len = self.seq_len - len(ids)
            if pad_len > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [self.tokenizer.pad_token_id] * pad_len
            return {"input": input_ids, "output": labels}

        # apply tokenization and drop original text field so collate_fn only sees inputs/outputs
        train_ds = raw_train.map(tokenize_fn, remove_columns=["text"])
        eval_ds = raw_eval.map(tokenize_fn, remove_columns=["text"])
        return train_ds, eval_ds

    def construct_model(self) -> nn.Module:
        hparams = {
            'n_vocab': self.tokenizer.vocab_size,
            'n_ctx': self.seq_len,
            'n_embd': self.dims,
            'n_layer': self.layers,
            'n_head': self.heads,
            'embd_pdrop': self.dropout,
            'resid_pdrop': self.dropout,
            'attn_pdrop': self.dropout,
            'local_window': self.seq_len,
            'rate': self.rate,
            'layer_norm_epsilon': 1e-5,
        }
        return Transformer(hparams)

    def create_optimizer(self, params: Iterator[nn.Parameter]
                         ) -> Tuple[optim.Optimizer,
                                    optim.lr_scheduler._LRScheduler]:
        optimizer = fusing.Adam(
            params, lr=self.base_lr, weight_decay=self.wd_rate)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1 - step / self.total_steps)
        return optimizer, scheduler

    def train_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                        ) -> Dict[str, torch.Tensor]:
        logits = model(data['input'])
        # flatten logits and labels for cross-entropy: (batch*seq_len, vocab) vs (batch*seq_len)
        labels = data['output'].reshape(-1)
        loss = self.criterion(logits, labels)
        return {'loss': loss}

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


def train_opengpt3_model(args: argparse.Namespace):
    if getattr(args, 'device', None) is None:
        if torch.cuda.is_available():
            args.device = torch.device('cuda')
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            args.device = torch.device('mps')
        else:
            args.device = torch.device('cpu')
    else:
        args.device = torch.device(args.device)

    spec = opengpt3TrainingSpec(
        dataset=args.dataset,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        layers=args.layers,
        heads=args.heads,
        dims=args.dims,
        rate=args.rate,
        dropout=args.dropout,
        base_lr=args.base_lr,
        wd_rate=args.wd_rate,
        total_steps=args.total_steps,
        use_grad_ckpt=args.use_grad_ckpt,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    config = TrainConfig(
        batch_train=args.batch_train, batch_eval=args.batch_eval,
        total_steps=args.total_steps, eval_steps=args.eval_steps,
        save_steps=args.save_steps, log_steps=args.log_steps, save_model_path=args.save_model_path,
        save_checkpoint_path=args.save_checkpoint_path,
        description='Training',
        log_format='loss: {train_loss:.4f}',
        use_amp=args.use_amp, gpus=args.gpus, device=args.device
    )

    Trainer(spec, config).train(from_checkpoint=args.from_checkpoint,
                                from_pretrained=args.from_pretrained)
    # just in case
    import os; os._exit(0)


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('train', help='train GPT-2 model')

    group = parser.add_argument_group('Dataset and tokenizer')
    group.add_argument(
        '--dataset', required=True,
        help='HuggingFace dataset identifier or path to a local text file'
    )
    group.add_argument(
        '--train_split', default='train',
        help='split name to use for training'
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
    group.add_argument('--dropout', default=0.1, type=float,
                       help='probability that each element is dropped')

    group = parser.add_argument_group('Training and evaluation')
    group.add_argument('--batch_train', default=64, type=int,
                       help='number of training batch size')
    group.add_argument('--batch_eval', default=64, type=int,
                       help='number of evaluation batch size')
    group.add_argument('--base_lr', default=1e-4, type=float,
                       help='default learning rate')
    group.add_argument('--wd_rate', default=1e-2, type=float,
                       help='weight decay rate')

    group.add_argument('--total_steps', default=1000000, type=int,
                       help='number of total training steps')
    group.add_argument('--eval_steps', default=500, type=int,
                       help='period to evaluate model and record metrics')
    group.add_argument('--save_steps', default=1000, type=int,
                       help='period to save training state to checkpoint')
    group.add_argument('--log_steps', default=100, type=int,
                       help='period to log training metrics')

    group = parser.add_argument_group('Saving and restoring')
    group.add_argument('--save_model_path', default='model.pth',
                       help='save trained model weights to the file')
    group.add_argument('--save_checkpoint_path', default='checkpoint.pth',
                       help='save training state to the checkpoint file')
    group.add_argument('--from_checkpoint', default=None,
                       help='load last training state from checkpoint file')
    group.add_argument('--from_pretrained', default=None,
                       help='initialize parameters from pretrained model')

    group = parser.add_argument_group('Extensions')
    group.add_argument('--use_amp', action='store_true',
                       help='use automatic mixed-precision in training')
    group.add_argument('--use_grad_ckpt', action='store_true',
                       help='use gradient checkpointing in transformer layers')
    group.add_argument('--gpus', default=None, type=int,
                       help='number of gpu devices to use in training')
    group.add_argument('--device', default=None,
                       help='torch device to use (e.g. cuda, mps, cpu)')

    parser.set_defaults(func=train_opengpt3_model)
