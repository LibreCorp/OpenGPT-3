import os

import torch
import pytest

from opengpt3.training.configuration import TrainConfig
from opengpt3.training.specification import TrainingSpec
from opengpt3.training.training import Trainer


class DummySpec(TrainingSpec):
    def initialize(self):
        # no-op
        pass

    def prepare_datasets(self):
        examples = [{'input': [1, 2], 'output': [2, 3]}] * 4
        return examples, examples

    def construct_model(self):
        return torch.nn.Linear(2, 2)

    def create_optimizer(self, params):
        opt = torch.optim.SGD(params, lr=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
        return opt, sched

    def train_objective(self, data, model):
        inp = data['input'].float()
        tgt = data['output'].float()
        logits = model(inp)
        loss = torch.nn.functional.mse_loss(logits, tgt)
        return {'loss': loss}

    def eval_objective(self, data, model):
        return self.train_objective(data, model)


def test_trainer_runs_and_saves_cpu(tmp_path):
    total_steps = 2
    config = TrainConfig(
        batch_train=2,
        batch_eval=2,
        total_steps=total_steps,
        eval_steps=total_steps * 2,
        save_steps=total_steps * 2,
        log_steps=1,
        save_model_path=str(tmp_path / 'model.pth'),
        save_checkpoint_path=str(tmp_path / 'checkpoint.pth'),
        description='dummy training',
        log_format='',
        use_amp=False,
        gpus=None,
        device=torch.device('cpu'),
    )
    spec = DummySpec()
    Trainer(spec, config).train(from_checkpoint=None, from_pretrained=None)
    assert os.path.exists(str(tmp_path / 'model.pth'))

def test_full_training_pipeline_hello_world(tmp_path):
    text = "hello world\n" * 8
    fpath = tmp_path / "hello.txt"
    fpath.write_text(text, encoding='utf-8')

    from opengpt3.train_model import opengpt3TrainingSpec

    spec = opengpt3TrainingSpec(
        dataset=str(fpath), tokenizer_path="gpt2",
        seq_len=4, layers=1, heads=1, dims=8, rate=2,
        dropout=0.0, base_lr=1e-4, wd_rate=0.0,
        total_steps=2, use_grad_ckpt=False,
        train_split="train", eval_split="train",
    )
    spec.initialize()

    # HF-style save_pretrained
    out_dir = tmp_path / "model"
    config = TrainConfig(
        batch_train=2, batch_eval=2,
        total_steps=2, eval_steps=100, save_steps=100,
        log_steps=10,
        save_model_path=str(out_dir) + os.sep,
        save_checkpoint_path=str(tmp_path / "checkpoint.pth"),
        description="hello-world pipeline", log_format="",
        use_amp=False, gpus=None, device=torch.device("cpu"),
    )
    Trainer(spec, config).train(from_checkpoint=None, from_pretrained=None)

    # directory should contain model + tokenizer
    assert out_dir.is_dir()

    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "tokenizer.json").exists()

    # metrics.json should include train/loss
    metrics = __import__('json').load(open(out_dir / "metrics.json"))
    assert "train/loss" in metrics and metrics["train/loss"]