import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import PreTrainedTokenizerFast
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils import fusing
from modeling import Transformer
from training import TrainConfig, TrainingPipeline, Trainer
from typing import Optional, Dict, Tuple

from opengpt3.config import resolve_training_settings
from opengpt3.data import load_train_eval_datasets


def build_training_pipeline(
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
    text_field: str = 'text',
    streaming: bool = True,
) -> TrainingPipeline:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_path, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
            if getattr(tokenizer, 'eos_token_id', None) is not None
            else 0
        )
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, reduction='mean'
    )

    def initialize():
        return None

    def prepare_datasets():
        return load_train_eval_datasets(
            dataset,
            train_split=train_split,
            eval_split=eval_split,
            tokenizer=tokenizer,
            seq_len=seq_len,
            text_field=text_field,
            streaming=streaming,
        )

    def construct_model() -> nn.Module:
        hparams = {
            'n_vocab': tokenizer.vocab_size,
            'n_ctx': seq_len,
            'n_embd': dims,
            'n_layer': layers,
            'n_head': heads,
            'embd_pdrop': dropout,
            'resid_pdrop': dropout,
            'attn_pdrop': dropout,
            'local_window': seq_len,
            'rate': rate,
            'layer_norm_epsilon': 1e-5,
        }
        return Transformer(hparams)

    def create_optimizer(params) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        optimizer = fusing.Adam(params, lr=base_lr, weight_decay=wd_rate)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1 - step / total_steps
        )
        return optimizer, scheduler

    def train_objective(data: Dict[str, torch.Tensor], model: nn.Module
                        ) -> Dict[str, torch.Tensor]:
        logits = model(data['input'])
        labels = data['output'].reshape(-1)
        loss = criterion(logits, labels)
        return {'loss': loss}

    def eval_objective(data: Dict[str, torch.Tensor], model: nn.Module
                       ) -> Dict[str, torch.Tensor]:
        logits, _ = model(data['input'], past=None, training=False)
        logits = logits.view(data['input'].shape[0], data['input'].shape[1], -1).transpose(1, 2)
        loss = criterion(logits, data['output'])

        mask = (data['output'] != tokenizer.pad_token_id).float()
        loss = (loss * mask).sum() / mask.sum()
        perplexity = (loss.exp() * mask).sum() / mask.sum()

        return {'loss': loss, 'perplexity': perplexity}

    def on_save(output_dir: str):
        tokenizer.save_pretrained(output_dir)

    return TrainingPipeline(
        initialize=initialize,
        build_datasets=prepare_datasets,
        build_model=construct_model,
        create_optimizer=create_optimizer,
        train_objective=train_objective,
        eval_objective=eval_objective,
        on_save=on_save,
    )


def _select_device(requested: Optional[str]) -> torch.device:
    if requested is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(requested)


def train_opengpt3_model(args: argparse.Namespace):
    settings = resolve_training_settings(args)

    pipeline = build_training_pipeline(
        dataset=settings.dataset,
        tokenizer_path=settings.tokenizer_path,
        seq_len=settings.seq_len,
        layers=settings.layers,
        heads=settings.heads,
        dims=settings.dims,
        rate=settings.rate,
        dropout=settings.dropout,
        base_lr=settings.base_lr,
        wd_rate=settings.wd_rate,
        total_steps=settings.total_steps,
        use_grad_ckpt=settings.use_grad_ckpt,
        train_split=settings.train_split,
        eval_split=settings.eval_split,
        text_field=settings.text_field,
        streaming=settings.streaming,
    )
    config = TrainConfig(
        batch_train=settings.batch_train,
        batch_eval=settings.batch_eval,
        total_steps=settings.total_steps,
        eval_steps=settings.eval_steps,
        save_steps=settings.save_steps,
        log_steps=settings.log_steps,
        save_model_path=settings.save_model_path,
        save_checkpoint_path=settings.save_checkpoint_path,
        description='Training',
        log_format='loss: {train_loss:.4f}',
        use_amp=settings.use_amp,
        gpus=settings.gpus,
        device=_select_device(settings.device),
    )

    Trainer(pipeline, config).train(
        from_checkpoint=settings.from_checkpoint,
        from_pretrained=settings.from_pretrained,
    )
    # just in case
    import os; os._exit(0)


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('train', help='train GPT-2 model')

    parser.add_argument('--config', '-c', default=None,
                        help='path to YAML configuration file')

    group = parser.add_argument_group('Dataset and tokenizer')
    group.add_argument(
        '--dataset', default=None,
        help='HuggingFace dataset identifier or path to a local text file'
    )
    group.add_argument(
        '--train_split', default=None,
        help='split name to use for training'
    )
    group.add_argument(
        '--eval_split', default=None,
        help='split name to use for evaluation'
    )
    group.add_argument(
        '--text_field', default=None,
        help='name of the text column when using a tabular dataset'
    )
    group.add_argument(
        '--tokenizer_path', default=None,
        help='pretrained tokenizer name or path for tokenization'
    )
    group.add_argument(
        '--streaming', default=None,
        help='toggle streaming datasets (true/false)'
    )

    group = parser.add_argument_group('Model configurations')
    group.add_argument('--seq_len', default=None, type=int,
                       help='maximum sequence length')
    group.add_argument('--layers', default=None, type=int,
                       help='number of transformer layers')
    group.add_argument('--heads', default=None, type=int,
                       help='number of multi-heads in attention layer')
    group.add_argument('--dims', default=None, type=int,
                       help='dimension of representation in each layer')
    group.add_argument('--rate', default=None, type=int,
                       help='increase rate of dimensionality in bottleneck')
    group.add_argument('--dropout', default=None, type=float,
                       help='probability that each element is dropped')

    group = parser.add_argument_group('Training and evaluation')
    group.add_argument('--batch_train', default=None, type=int,
                       help='number of training batch size')
    group.add_argument('--batch_eval', default=None, type=int,
                       help='number of evaluation batch size')
    group.add_argument('--base_lr', default=None, type=float,
                       help='default learning rate')
    group.add_argument('--wd_rate', default=None, type=float,
                       help='weight decay rate')

    group.add_argument('--total_steps', default=None, type=int,
                       help='number of total training steps')
    group.add_argument('--eval_steps', default=None, type=int,
                       help='period to evaluate model and record metrics')
    group.add_argument('--save_steps', default=None, type=int,
                       help='period to save training state to checkpoint')
    group.add_argument('--log_steps', default=None, type=int,
                       help='period to log training metrics')

    group = parser.add_argument_group('Saving and restoring')
    group.add_argument('--save_model_path', default=None,
                       help='save trained model weights to the file')
    group.add_argument('--save_checkpoint_path', default=None,
                       help='save training state to the checkpoint file')
    group.add_argument('--from_checkpoint', default=None,
                       help='load last training state from checkpoint file')
    group.add_argument('--from_pretrained', default=None,
                       help='initialize parameters from pretrained model')

    group = parser.add_argument_group('Extensions')
    group.add_argument('--use_amp', action='store_true', default=None,
                       help='use automatic mixed-precision in training')
    group.add_argument('--use_grad_ckpt', action='store_true', default=None,
                       help='use gradient checkpointing in transformer layers')
    group.add_argument('--gpus', default=None, type=int,
                       help='number of gpu devices to use in training')
    group.add_argument('--device', default=None,
                       help='torch device to use (e.g. cuda, mps, cpu)')

    parser.set_defaults(func=train_opengpt3_model)
