import os, sys
import argparse
from opengpt3 import (
    train_model,
    evaluate_model,
    generate_sentences,
    visualize_metrics,
    tokenizer_train,
)


if __name__ == '__main__':
    # ensure local 'src' directory (project source) is prioritized
    base = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    srcdir = os.path.join(base, 'src')
    if os.path.isdir(srcdir) and srcdir not in sys.path:
        sys.path.insert(0, srcdir)
    parser = argparse.ArgumentParser(
        prog='opengpt3',
        description='PyTorch implementation of OpenAI GPT-2')
    subparsers = parser.add_subparsers(dest='subcommands')  # , required=True)
    # The above code is modified for compatibility. Argparse in Python 3.6
    # version does not support `required` option in `add_subparsers`.

    train_model.add_subparser(subparsers)
    evaluate_model.add_subparser(subparsers)
    generate_sentences.add_subparser(subparsers)
    visualize_metrics.add_subparser(subparsers)
    tokenizer_train.add_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)
