"""Configuration helpers for OpenGPT-3 CLI entrypoints.

This module centralises logic for reading YAML configuration files and
combining them with command line overrides.  Historically the project relied
on large "spec" classes to glue together training/evaluation/generation
pipelines.  The helpers defined here move that responsibility into a single
place so the high level scripts can stay focused on driving the pipeline
while all configuration defaults live together.

The design goals are simple:

* YAML configuration files are optional.  When they are present they provide
  defaults that the CLI may override.
* CLI flags always win when a value is explicitly provided, but sensible
  defaults are filled in when neither the config file nor the CLI specify a
  value.
* Paths declared inside a YAML file are resolved relative to the YAML file to
  keep project directories self-contained.

The module purposely keeps the output types light-weight (plain dataclasses)
so the rest of the codebase can depend on structured objects rather than
deeply nested dictionaries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _load_yaml(config_path: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load YAML configuration returning data and its directory.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.  ``None`` yields an empty config.

    Returns
    -------
    tuple(dict, Optional[str])
        Parsed YAML data (never ``None``) and the directory that contains the
        configuration file.  The directory is returned so that relative paths
        inside the YAML file can be made absolute later on.
    """

    if config_path is None:
        return {}, None

    path = os.path.expanduser(config_path)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Top-level YAML structure must be a mapping, got {type(data)!r}"
        )

    return data, os.path.dirname(os.path.abspath(path))


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "1", "yes", "on"}:
            return True
        if normalised in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"Cannot interpret {value!r} as a boolean for '{field}'")


def _maybe_resolve_path(value: Any, *, base_dir: Optional[str]) -> Any:
    """Resolve a relative path from the YAML file.

    Values that look like Hugging Face dataset identifiers (no path separator
    and no file extension) are returned untouched.  Everything else is treated
    as a potential path and, if ``base_dir`` is provided, normalised relative
    to that directory.
    """

    if base_dir is None or not isinstance(value, str):
        return value

    if value.startswith(("hf://", "s3://", "gs://")):
        return value
    if os.path.isabs(value):
        return value

    if any(sep in value for sep in ("/", os.sep, "\\")) or \
       os.path.splitext(value)[1]:
        return os.path.normpath(os.path.join(base_dir, value))

    return value


def _resolve(
    field: str,
    *candidates: Any,
    default: Any = None,
    required: bool = False,
    transform=None,
) -> Any:
    """Return the first usable value from ``candidates``.

    Parameters
    ----------
    field:
        Field name used for error messages.
    *candidates:
        Values to attempt in order of precedence.  ``None`` and empty strings
        are skipped.
    default:
        Fallback value when all candidates are missing.
    required:
        When ``True`` and the resolution fails, a ``ValueError`` is raised.
    transform:
        Optional callable applied to the resolved value.  Should raise an
        exception if the value cannot be coerced.
    """

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, str) and candidate.strip() == "":
            continue
        value = candidate
        if transform is not None:
            try:
                value = transform(candidate)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Invalid value {candidate!r} supplied for '{field}'"
                ) from exc
        return value

    if default is not None:
        if transform is not None:
            return transform(default)
        return default

    if required:
        raise ValueError(f"Missing required configuration value '{field}'")

    return None


# ---------------------------------------------------------------------------
# Dataclasses describing fully resolved configurations
# ---------------------------------------------------------------------------


@dataclass
class TrainingSettings:
    dataset: str
    train_split: str
    eval_split: str
    text_field: str
    tokenizer_path: str
    seq_len: int
    layers: int
    heads: int
    dims: int
    rate: int
    dropout: float
    base_lr: float
    wd_rate: float
    total_steps: int
    batch_train: int
    batch_eval: int
    eval_steps: int
    save_steps: int
    log_steps: int
    save_model_path: str
    save_checkpoint_path: str
    use_amp: bool
    use_grad_ckpt: bool
    gpus: Optional[int]
    device: Optional[str]
    streaming: bool
    from_checkpoint: Optional[str]
    from_pretrained: Optional[str]


@dataclass
class EvaluationSettings:
    dataset: str
    eval_split: str
    text_field: str
    tokenizer_path: str
    seq_len: int
    layers: int
    heads: int
    dims: int
    rate: int
    batch_eval: int
    total_steps: int
    use_gpu: bool
    model_path: str
    streaming: bool


@dataclass
class GenerationSettings:
    tokenizer_path: str
    model_path: str
    prompt: str
    seq_len: int
    nucleus_prob: float
    temperature: float
    use_gpu: bool


# ---------------------------------------------------------------------------
# Resolution helpers for the three entrypoints
# ---------------------------------------------------------------------------


def resolve_training_settings(args: Any) -> TrainingSettings:
    config, base_dir = _load_yaml(getattr(args, "config", None))

    dataset_cfg = config.get("dataset", {}) if isinstance(config, dict) else {}
    tokenizer_cfg = config.get("tokenizer", {}) if isinstance(config, dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    optimiser_cfg = config.get("optimizer", {}) if isinstance(config, dict) else {}
    training_cfg = config.get("training", {}) if isinstance(config, dict) else {}
    resume_cfg = config.get("resume", {}) if isinstance(config, dict) else {}

    dataset = _resolve(
        "dataset",
        getattr(args, "dataset", None),
        dataset_cfg.get("name"),
        dataset_cfg.get("path"),
        required=True,
    )
    dataset = _maybe_resolve_path(dataset, base_dir=base_dir)

    train_split = _resolve(
        "train_split",
        getattr(args, "train_split", None),
        dataset_cfg.get("train_split"),
        default="train",
    )
    eval_split = _resolve(
        "eval_split",
        getattr(args, "eval_split", None),
        dataset_cfg.get("eval_split"),
        default="validation",
    )
    text_field = _resolve(
        "text_field",
        getattr(args, "text_field", None),
        dataset_cfg.get("text_field"),
        default="text",
    )
    streaming = _resolve(
        "streaming",
        getattr(args, "streaming", None),
        dataset_cfg.get("streaming"),
        default=True,
        transform=lambda value: _as_bool(value, field="streaming"),
    )

    tokenizer_path = _resolve(
        "tokenizer_path",
        getattr(args, "tokenizer_path", None),
        tokenizer_cfg.get("path"),
        required=True,
    )
    tokenizer_path = _maybe_resolve_path(tokenizer_path, base_dir=base_dir)

    seq_len = _resolve(
        "seq_len",
        getattr(args, "seq_len", None),
        model_cfg.get("seq_len"),
        default=64,
        transform=int,
    )
    layers = _resolve(
        "layers",
        getattr(args, "layers", None),
        model_cfg.get("layers"),
        default=12,
        transform=int,
    )
    heads = _resolve(
        "heads",
        getattr(args, "heads", None),
        model_cfg.get("heads"),
        default=16,
        transform=int,
    )
    dims = _resolve(
        "dims",
        getattr(args, "dims", None),
        model_cfg.get("dims"),
        default=1024,
        transform=int,
    )
    rate = _resolve(
        "rate",
        getattr(args, "rate", None),
        model_cfg.get("rate"),
        default=4,
        transform=int,
    )
    dropout = _resolve(
        "dropout",
        getattr(args, "dropout", None),
        model_cfg.get("dropout"),
        default=0.1,
        transform=float,
    )

    base_lr = _resolve(
        "base_lr",
        getattr(args, "base_lr", None),
        optimiser_cfg.get("lr"),
        optimiser_cfg.get("base_lr"),
        training_cfg.get("learning_rate"),
        default=1e-4,
        transform=float,
    )
    wd_rate = _resolve(
        "wd_rate",
        getattr(args, "wd_rate", None),
        optimiser_cfg.get("weight_decay"),
        training_cfg.get("weight_decay"),
        default=1e-2,
        transform=float,
    )

    total_steps = _resolve(
        "total_steps",
        getattr(args, "total_steps", None),
        training_cfg.get("total_steps"),
        default=1_000_000,
        transform=int,
    )
    batch_train = _resolve(
        "batch_train",
        getattr(args, "batch_train", None),
        training_cfg.get("batch_train"),
        training_cfg.get("train_batch"),
        default=64,
        transform=int,
    )
    batch_eval = _resolve(
        "batch_eval",
        getattr(args, "batch_eval", None),
        training_cfg.get("batch_eval"),
        training_cfg.get("eval_batch"),
        default=64,
        transform=int,
    )
    eval_steps = _resolve(
        "eval_steps",
        getattr(args, "eval_steps", None),
        training_cfg.get("eval_steps"),
        default=500,
        transform=int,
    )
    save_steps = _resolve(
        "save_steps",
        getattr(args, "save_steps", None),
        training_cfg.get("save_steps"),
        default=1000,
        transform=int,
    )
    log_steps = _resolve(
        "log_steps",
        getattr(args, "log_steps", None),
        training_cfg.get("log_steps"),
        training_cfg.get("logging_steps"),
        default=100,
        transform=int,
    )

    save_model_path = _resolve(
        "save_model_path",
        getattr(args, "save_model_path", None),
        training_cfg.get("save_model_path"),
        training_cfg.get("model_dir"),
        default="model.pth",
    )
    save_model_path = _maybe_resolve_path(save_model_path, base_dir=base_dir)

    save_checkpoint_path = _resolve(
        "save_checkpoint_path",
        getattr(args, "save_checkpoint_path", None),
        training_cfg.get("checkpoint_path"),
        default="checkpoint.pth",
    )
    save_checkpoint_path = _maybe_resolve_path(
        save_checkpoint_path, base_dir=base_dir
    )

    use_amp = _resolve(
        "use_amp",
        getattr(args, "use_amp", None),
        training_cfg.get("use_amp"),
        default=False,
        transform=lambda value: _as_bool(value, field="use_amp"),
    )
    use_grad_ckpt = _resolve(
        "use_grad_ckpt",
        getattr(args, "use_grad_ckpt", None),
        model_cfg.get("use_grad_ckpt"),
        training_cfg.get("use_grad_ckpt"),
        default=False,
        transform=lambda value: _as_bool(value, field="use_grad_ckpt"),
    )

    gpus = _resolve(
        "gpus",
        getattr(args, "gpus", None),
        training_cfg.get("gpus"),
        transform=lambda value: None if value is None else int(value),
    )
    device = _resolve(
        "device",
        getattr(args, "device", None),
        training_cfg.get("device"),
    )

    from_checkpoint = _resolve(
        "from_checkpoint",
        getattr(args, "from_checkpoint", None),
        resume_cfg.get("checkpoint"),
    )
    from_checkpoint = _maybe_resolve_path(
        from_checkpoint, base_dir=base_dir
    )

    from_pretrained = _resolve(
        "from_pretrained",
        getattr(args, "from_pretrained", None),
        resume_cfg.get("pretrained"),
    )
    from_pretrained = _maybe_resolve_path(
        from_pretrained, base_dir=base_dir
    )

    return TrainingSettings(
        dataset=dataset,
        train_split=train_split,
        eval_split=eval_split,
        text_field=text_field,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        layers=layers,
        heads=heads,
        dims=dims,
        rate=rate,
        dropout=dropout,
        base_lr=base_lr,
        wd_rate=wd_rate,
        total_steps=total_steps,
        batch_train=batch_train,
        batch_eval=batch_eval,
        eval_steps=eval_steps,
        save_steps=save_steps,
        log_steps=log_steps,
        save_model_path=save_model_path,
        save_checkpoint_path=save_checkpoint_path,
        use_amp=use_amp,
        use_grad_ckpt=use_grad_ckpt,
        gpus=gpus,
        device=device,
        streaming=streaming,
        from_checkpoint=from_checkpoint,
        from_pretrained=from_pretrained,
    )


def resolve_evaluation_settings(args: Any) -> EvaluationSettings:
    config, base_dir = _load_yaml(getattr(args, "config", None))

    dataset_cfg = config.get("dataset", {}) if isinstance(config, dict) else {}
    tokenizer_cfg = config.get("tokenizer", {}) if isinstance(config, dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    evaluation_cfg = config.get("evaluation", {}) if isinstance(config, dict) else {}

    model_path = _resolve(
        "model_path",
        getattr(args, "model_path", None),
        evaluation_cfg.get("model_path"),
        required=True,
    )
    model_path = _maybe_resolve_path(model_path, base_dir=base_dir)

    dataset = _resolve(
        "dataset",
        getattr(args, "dataset", None),
        dataset_cfg.get("name"),
        dataset_cfg.get("path"),
        required=True,
    )
    dataset = _maybe_resolve_path(dataset, base_dir=base_dir)

    eval_split = _resolve(
        "eval_split",
        getattr(args, "eval_split", None),
        dataset_cfg.get("eval_split"),
        default="validation",
    )
    text_field = _resolve(
        "text_field",
        getattr(args, "text_field", None),
        dataset_cfg.get("text_field"),
        default="text",
    )
    streaming = _resolve(
        "streaming",
        getattr(args, "streaming", None),
        dataset_cfg.get("streaming"),
        default=True,
        transform=lambda value: _as_bool(value, field="streaming"),
    )

    tokenizer_path = _resolve(
        "tokenizer_path",
        getattr(args, "tokenizer_path", None),
        tokenizer_cfg.get("path"),
        required=True,
    )
    tokenizer_path = _maybe_resolve_path(tokenizer_path, base_dir=base_dir)

    seq_len = _resolve(
        "seq_len",
        getattr(args, "seq_len", None),
        model_cfg.get("seq_len"),
        evaluation_cfg.get("seq_len"),
        default=64,
        transform=int,
    )
    layers = _resolve(
        "layers",
        getattr(args, "layers", None),
        model_cfg.get("layers"),
        default=12,
        transform=int,
    )
    heads = _resolve(
        "heads",
        getattr(args, "heads", None),
        model_cfg.get("heads"),
        default=16,
        transform=int,
    )
    dims = _resolve(
        "dims",
        getattr(args, "dims", None),
        model_cfg.get("dims"),
        default=1024,
        transform=int,
    )
    rate = _resolve(
        "rate",
        getattr(args, "rate", None),
        model_cfg.get("rate"),
        default=4,
        transform=int,
    )

    batch_eval = _resolve(
        "batch_eval",
        getattr(args, "batch_eval", None),
        evaluation_cfg.get("batch_eval"),
        default=64,
        transform=int,
    )
    total_steps = _resolve(
        "total_steps",
        getattr(args, "total_steps", None),
        evaluation_cfg.get("total_steps"),
        default=-1,
        transform=int,
    )
    use_gpu = _resolve(
        "use_gpu",
        getattr(args, "use_gpu", None),
        evaluation_cfg.get("use_gpu"),
        default=False,
        transform=lambda value: _as_bool(value, field="use_gpu"),
    )

    return EvaluationSettings(
        dataset=dataset,
        eval_split=eval_split,
        text_field=text_field,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        layers=layers,
        heads=heads,
        dims=dims,
        rate=rate,
        batch_eval=batch_eval,
        total_steps=total_steps,
        use_gpu=use_gpu,
        model_path=model_path,
        streaming=streaming,
    )


def resolve_generation_settings(args: Any) -> GenerationSettings:
    config, base_dir = _load_yaml(getattr(args, "config", None))

    generator_cfg = config.get("generation", {}) if isinstance(config, dict) else {}
    tokenizer_cfg = config.get("tokenizer", {}) if isinstance(config, dict) else {}

    tokenizer_path = _resolve(
        "tokenizer_path",
        getattr(args, "tokenizer_path", None),
        tokenizer_cfg.get("path"),
        generator_cfg.get("tokenizer_path"),
        required=True,
    )
    tokenizer_path = _maybe_resolve_path(tokenizer_path, base_dir=base_dir)

    model_path = _resolve(
        "model_path",
        getattr(args, "model_path", None),
        generator_cfg.get("model_path"),
        required=True,
    )
    model_path = _maybe_resolve_path(model_path, base_dir=base_dir)

    prompt = _resolve(
        "prompt",
        getattr(args, "prompt", None),
        generator_cfg.get("prompt"),
        default="",
    )
    seq_len = _resolve(
        "seq_len",
        getattr(args, "seq_len", None),
        generator_cfg.get("seq_len"),
        default=64,
        transform=int,
    )
    nucleus_prob = _resolve(
        "nucleus_prob",
        getattr(args, "nucleus_prob", None),
        generator_cfg.get("nucleus_prob"),
        default=0.85,
        transform=float,
    )
    temperature = _resolve(
        "temperature",
        getattr(args, "temperature", None),
        generator_cfg.get("temperature"),
        default=1.0,
        transform=float,
    )
    use_gpu = _resolve(
        "use_gpu",
        getattr(args, "use_gpu", None),
        generator_cfg.get("use_gpu"),
        default=False,
        transform=lambda value: _as_bool(value, field="use_gpu"),
    )

    return GenerationSettings(
        tokenizer_path=tokenizer_path,
        model_path=model_path,
        prompt=prompt,
        seq_len=seq_len,
        nucleus_prob=nucleus_prob,
        temperature=temperature,
        use_gpu=use_gpu,
    )


__all__ = [
    "TrainingSettings",
    "EvaluationSettings",
    "GenerationSettings",
    "resolve_training_settings",
    "resolve_evaluation_settings",
    "resolve_generation_settings",
]
