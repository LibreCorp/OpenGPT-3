import torch
import pytest
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader

@pytest.fixture(autouse=True)
def dummy_tokenizer(monkeypatch):
    class DummyTok:
        def __init__(self, *args, **kwargs):
            self.vocab_size = 10
            self.pad_token_id = 0

        def __call__(self, text, truncation, max_length):
            return {'input_ids': [1, 2, 3]}

    monkeypatch.setattr(
        PreTrainedTokenizerFast, 'from_pretrained', lambda *args, **kwargs: DummyTok()
    )

def test_data_pipeline_basic(tmp_path):
    text = "a b c d e f g h i j k l m n o p\n"
    fpath = tmp_path / "data.txt"
    fpath.write_text(text, encoding='utf-8')

    from opengpt3.train_model import opengpt3TrainingSpec

    class Args:
        dataset = str(fpath)
        tokenizer_path = 'gpt2'
        seq_len = 8
        layers = 1
        heads = 1
        dims = 8
        rate = 2
        dropout = 0.0
        base_lr = 1e-4
        wd_rate = 0.0
        total_steps = 1
        use_grad_ckpt = False
        train_split = 'train'
        eval_split = 'train'

    spec = opengpt3TrainingSpec(
        Args.dataset, Args.tokenizer_path,
        Args.seq_len, Args.layers, Args.heads,
        Args.dims, Args.rate, Args.dropout,
        Args.base_lr, Args.wd_rate, Args.total_steps,
        Args.use_grad_ckpt, Args.train_split, Args.eval_split
    )
    spec.initialize()
    train_ds, _ = spec.prepare_datasets()

    loader = DataLoader(
        train_ds,
        batch_size=2,
        collate_fn=lambda batch: {
            'input': torch.tensor([ex['input'] for ex in batch], dtype=torch.long),
            'output': torch.tensor([ex['output'] for ex in batch], dtype=torch.long),
        }
    )
    batch = next(iter(loader))
    # shapes
    assert 'input' in batch and 'output' in batch
    # inputs are IDs[:-1] padded to seq_len-1 length
    assert batch['input'].shape[1] == (Args.seq_len - 1)