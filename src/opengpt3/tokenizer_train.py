import argparse
import os

from tokenizers import ByteLevelBPETokenizer


def train_tokenizer(args: argparse.Namespace):
    """
    Train a Byte-Level BPE tokenizer from scratch and save tokenizer files.
    """
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=args.files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special_tokens.split(',') if args.special_tokens else []
    )
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_model(args.output_dir)
    print(f"Tokenizer files saved to {args.output_dir}")


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        'tokenizer', help='train a ByteLevel BPE tokenizer'
    )
    parser.add_argument(
        '--files', nargs='+', required=True,
        help='paths to text files to train the tokenizer on'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='directory to save tokenizer files (vocab.json, merges.txt)'
    )
    parser.add_argument(
        '--vocab_size', type=int, default=50257,
        help='size of the tokenizer vocabulary'
    )
    parser.add_argument(
        '--min_frequency', type=int, default=2,
        help='minimum frequency a pair must have to be merged'
    )
    parser.add_argument(
        '--special_tokens', type=str, default='',
        help='comma-separated list of special tokens to add'
    )
    parser.set_defaults(func=train_tokenizer)