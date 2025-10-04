import argparse
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from opengpt3.modeling import Transformer
from opengpt3.evaluation import EvaluationPipeline, EvaluateConfig, Evaluator
from typing import Dict

from opengpt3.config import resolve_evaluation_settings
from opengpt3.data import load_autoregressive_dataset


def build_evaluation_pipeline(
    dataset: str,
    tokenizer_path: str,
    seq_len: int,
    layers: int,
    heads: int,
    dims: int,
    rate: int,
    eval_split: str = 'validation',
    text_field: str = 'text',
    streaming: bool = True,
) -> EvaluationPipeline:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_path, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
            if getattr(tokenizer, 'eos_token_id', None) is not None
            else 0
        )
    criterion = nn.CrossEntropyLoss(reduction='none')

    def initialize():
        return None

    def prepare_dataset():
        return load_autoregressive_dataset(
            dataset,
            split=eval_split,
            tokenizer=tokenizer,
            seq_len=seq_len,
            text_field=text_field,
            streaming=streaming,
        )

    def construct_model(from_model: str | None = None):
        if from_model:
            return Transformer.from_pretrained(from_model)
        hparams = {
            'n_vocab': tokenizer.vocab_size,
            'n_ctx': seq_len,
            'n_embd': dims,
            'n_layer': layers,
            'n_head': heads,
            'embd_pdrop': 0.0,
            'resid_pdrop': 0.0,
            'attn_pdrop': 0.0,
            'local_window': seq_len,
            'rate': rate,
            'layer_norm_epsilon': 1e-5,
        }
        return Transformer(hparams)

    def eval_objective(data: Dict[str, torch.Tensor], model) -> Dict[str, torch.Tensor]:
        logits, _ = model(data['input'], past=None, training=False)
        logits = logits.view(data['input'].shape[0], data['input'].shape[1], -1).transpose(1, 2)
        loss = criterion(logits, data['output'])

        mask = (data['output'] != tokenizer.pad_token_id).float()
        loss = (loss * mask).sum() / mask.sum()
        perplexity = (loss.exp() * mask).sum() / mask.sum()

        return {'loss': loss, 'perplexity': perplexity}

    return EvaluationPipeline(
        initialize=initialize,
        build_dataset=prepare_dataset,
        build_model=construct_model,
        eval_objective=eval_objective,
    )


def evaluate_opengpt3_model(args: argparse.Namespace):
    settings = resolve_evaluation_settings(args)

    pipeline = build_evaluation_pipeline(
        dataset=settings.dataset,
        tokenizer_path=settings.tokenizer_path,
        seq_len=settings.seq_len,
        layers=settings.layers,
        heads=settings.heads,
        dims=settings.dims,
        rate=settings.rate,
        eval_split=settings.eval_split,
        text_field=settings.text_field,
        streaming=settings.streaming,
    )
    config = EvaluateConfig(
        batch_eval=settings.batch_eval,
        total_steps=settings.total_steps,
        use_gpu=settings.use_gpu,
    )

    print(Evaluator(pipeline, config).evaluate(from_model=settings.model_path))


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('evaluate', help='evaluate GPT-2 model')

    parser.add_argument('--config', '-c', default=None,
                        help='path to YAML configuration file')

    parser.add_argument('--model_path', default=None,
                        help='trained GPT-2 model file path')

    group = parser.add_argument_group('Dataset and tokenizer')
    group.add_argument(
        '--dataset', default=None,
        help='HuggingFace dataset identifier or path to a local text file'
    )
    group.add_argument(
        '--eval_split', default=None,
        help='split name to use for evaluation'
    )
    group.add_argument(
        '--tokenizer_path', default=None,
        help='pretrained tokenizer name or path for tokenization'
    )
    group.add_argument(
        '--text_field', default=None,
        help='name of the text column when using a tabular dataset'
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

    group = parser.add_argument_group('Evaluation options')
    group.add_argument('--batch_eval', default=None, type=int,
                       help='number of evaluation batch size')
    group.add_argument('--total_steps', default=None, type=int,
                       help='number of total evaluation steps')
    group.add_argument('--use_gpu', action='store_true', default=None,
                       help='use gpu device in inferencing')

    parser.set_defaults(func=evaluate_opengpt3_model)
