import pytest
import torch
import argparse

from opengpt3.modeling.transformer import GPTModel
from opengpt3.generate_sentences import add_subparser as generate_add_subparser

@pytest.fixture(autouse=True)
def dummy_tokenizer_and_model(monkeypatch):
    class DummyTok:
        def __init__(self, *args, **kwargs):
            self.bos_token_id = 0
            self.eos_token_id = 1
            self.pad_token_id = 0

        def encode(self, text, return_tensors=None):
            # return a fake tensor of shape [1, N]
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

        def decode(self, tokens, skip_special_tokens=True):
            return "decoded"

    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self):
            return self

        def to(self, device):
            return self

        def parameters(self):
            # Return a dummy parameter for the test to get the device from
            yield torch.nn.Parameter(torch.empty(0))

        @property
        def hparams(self):
            return {'n_ctx': 64}

        def __call__(self, input_ids, past=None, training=False):
            # Simulate model output: logits and a dummy past
            vocab_size = 50257 # Assuming a common vocab size
            logits = torch.randn(input_ids.shape[0] * input_ids.shape[1], vocab_size)
            dummy_past = (torch.randn(1, 1, 1, 1, 1), torch.randn(1, 1, 1, 1, 1)) # Dummy past state
            return logits, dummy_past

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: DummyTok()
    )
    monkeypatch.setattr(
        GPTModel, 'from_pretrained', lambda *args, **kwargs: DummyModel()
    )

def test_generation_via_cli(capsys):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommands')
    generate_add_subparser(subparsers) # This will add the 'generate' subparser

    args = parser.parse_args([
        'generate',
        '--tokenizer_path', 'dummy',
        '--model_path', 'dummy',
        '--prompt', 'hello',
        '--seq_len', '10',
        '--nucleus_prob', '0.8',
        '--temperature', '0.5',
        '--use_gpu' # flag, no value needed
    ])

    args.func(args)
    captured = capsys.readouterr()
    # Should print the decoded string once
    assert captured.out.strip() == 'decoded'