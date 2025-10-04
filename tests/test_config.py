from argparse import Namespace

from opengpt3.config import (
    resolve_training_settings,
    resolve_evaluation_settings,
    resolve_generation_settings,
)


def _training_namespace(config_path):
    return Namespace(
        config=str(config_path),
        dataset=None,
        train_split=None,
        eval_split=None,
        text_field=None,
        streaming=None,
        tokenizer_path=None,
        seq_len=None,
        layers=None,
        heads=None,
        dims=None,
        rate=None,
        dropout=None,
        batch_train=None,
        batch_eval=None,
        base_lr=None,
        wd_rate=None,
        total_steps=None,
        eval_steps=None,
        save_steps=None,
        log_steps=None,
        save_model_path=None,
        save_checkpoint_path=None,
        use_amp=None,
        use_grad_ckpt=None,
        gpus=None,
        device=None,
        from_checkpoint=None,
        from_pretrained=None,
    )


def test_training_settings_from_yaml(tmp_path):
    config_text = """
dataset:
  path: data.txt
  train_split: train
  eval_split: validation
  text_field: content
  streaming: false
tokenizer:
  path: tokenizer
model:
  seq_len: 32
  layers: 4
  heads: 2
  dims: 64
  rate: 2
  dropout: 0.2
optimizer:
  lr: 0.001
  weight_decay: 0.0
training:
  batch_train: 8
  batch_eval: 4
  total_steps: 16
  eval_steps: 2
  save_steps: 4
  log_steps: 1
  save_model_path: outputs/model
  checkpoint_path: checkpoints/state.pt
  use_amp: true
  device: cpu
resume:
  checkpoint: checkpoints/state.pt
"""
    config_path = tmp_path / "train.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    args = _training_namespace(config_path)
    settings = resolve_training_settings(args)

    assert settings.dataset.endswith("data.txt")
    assert settings.text_field == "content"
    assert settings.streaming is False
    assert settings.seq_len == 32
    assert settings.layers == 4
    assert settings.batch_train == 8
    assert settings.base_lr == 0.001
    assert settings.save_model_path.endswith("outputs/model")
    assert settings.save_checkpoint_path.endswith("checkpoints/state.pt")
    assert settings.use_amp is True
    assert settings.from_checkpoint.endswith("checkpoints/state.pt")


def test_evaluation_settings_from_yaml(tmp_path):
    config_text = """
dataset:
  path: eval.txt
evaluation:
  model_path: saved/model
  batch_eval: 2
tokenizer:
  path: tokenizer
model:
  seq_len: 48
"""
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    args = Namespace(
        config=str(config_path),
        model_path=None,
        dataset=None,
        eval_split=None,
        text_field=None,
        streaming=None,
        tokenizer_path=None,
        seq_len=None,
        layers=None,
        heads=None,
        dims=None,
        rate=None,
        batch_eval=None,
        total_steps=None,
        use_gpu=None,
    )

    settings = resolve_evaluation_settings(args)
    assert settings.dataset.endswith("eval.txt")
    assert settings.model_path.endswith("saved/model")
    assert settings.batch_eval == 2
    assert settings.seq_len == 48


def test_generation_settings_from_yaml(tmp_path):
    config_text = """
generation:
  model_path: saved/model
  tokenizer_path: tokenizer
  prompt: hello
  seq_len: 24
  nucleus_prob: 0.9
  temperature: 0.75
  use_gpu: true
"""
    config_path = tmp_path / "gen.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    args = Namespace(
        config=str(config_path),
        tokenizer_path=None,
        model_path=None,
        prompt=None,
        seq_len=None,
        nucleus_prob=None,
        temperature=None,
        use_gpu=None,
    )

    settings = resolve_generation_settings(args)
    assert settings.prompt == "hello"
    assert settings.seq_len == 24
    assert settings.nucleus_prob == 0.9
    assert settings.temperature == 0.75
    assert settings.use_gpu is True
