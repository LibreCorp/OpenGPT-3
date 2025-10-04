"""Shared data loading utilities."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import torch
from datasets import load_dataset, IterableDataset


def _ensure_pad_token(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return pad
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        tokenizer.pad_token_id = eos
        return eos
    tokenizer.pad_token_id = 0
    return 0


def _tokenise_autoregressive(
    tokenizer,
    text: str,
    *,
    seq_len: int,
    pad_token_id: int,
):
    tokens = tokenizer(text, truncation=True, max_length=seq_len)["input_ids"]
    if not tokens:
        tokens = [pad_token_id, pad_token_id]
    elif len(tokens) == 1:
        tokens.append(pad_token_id)

    input_ids = tokens[:-1]
    labels = tokens[1:]

    pad_len = seq_len - len(tokens)
    if pad_len > 0:
        input_ids = input_ids + [pad_token_id] * pad_len
        labels = labels + [pad_token_id] * pad_len

    return {
        "input": torch.tensor(input_ids, dtype=torch.long),
        "output": torch.tensor(labels, dtype=torch.long),
    }


def load_autoregressive_dataset(
    dataset: str,
    *,
    split: str,
    tokenizer,
    seq_len: int,
    text_field: str = "text",
    streaming: bool = True,
):
    """Load a dataset and prepare (input, output) tensors for causal LM.

    Parameters
    ----------
    dataset:
        Hugging Face dataset identifier or a path to a local text file.
    split:
        Dataset split to load (e.g. ``"train"`` or ``"validation"``).
    tokenizer:
        Tokenizer providing ``__call__`` and ``pad_token_id`` attributes.
    seq_len:
        Maximum length supplied to the tokenizer.  The resulting ``input``
        tensors will have ``seq_len - 1`` tokens because of the autoregressive
        shift.
    text_field:
        Name of the field containing text in the dataset.
    streaming:
        Whether to use streaming mode when loading the dataset.
    """

    data_files = None
    path_candidate = os.path.expanduser(dataset)
    if os.path.isfile(path_candidate):
        data_files = {"train": path_candidate}
        dataset_name = "text"
    else:
        dataset_name = dataset

    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), ".cache", "hf-datasets")
    os.makedirs(cache_dir, exist_ok=True)

    def _load(with_streaming: bool):
        return load_dataset(
            dataset_name,
            data_files=data_files,
            split=split,
            streaming=with_streaming,
            cache_dir=cache_dir,
        )

    try:
        raw = _load(streaming)
    except (RuntimeError, PermissionError) as exc:
        if streaming and _requires_streaming_fallback(exc):
            raw = _load(False)
        else:  # pragma: no cover - unexpected errors are re-raised
            raise

    pad_token_id = _ensure_pad_token(tokenizer)
    column_names: Optional[Iterable[str]] = getattr(raw, "column_names", None)
    if column_names is not None:
        if text_field not in column_names:
            raise ValueError(
                f"Dataset split '{split}' does not provide a '{text_field}' column"
            )
        remove_columns: Optional[Iterable[str]] = column_names
    else:
        # Streaming datasets expose ``column_names`` as ``None``; explicitly
        # drop the source text column so downstream batches stay tensor-only.
        remove_columns = [text_field]

    dataset = raw.map(
        lambda example: _tokenise_autoregressive(
            tokenizer,
            example[text_field],
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        ),
        remove_columns=remove_columns,
    )

    if not isinstance(dataset, IterableDataset):
        dataset.set_format(type="torch", columns=["input", "output"])

    return dataset


def load_train_eval_datasets(
    dataset: str,
    *,
    train_split: str,
    eval_split: str,
    tokenizer,
    seq_len: int,
    text_field: str = "text",
    streaming: bool = True,
):
    """Convenience wrapper returning both train and eval splits."""

    train_ds = load_autoregressive_dataset(
        dataset,
        split=train_split,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_field=text_field,
        streaming=streaming,
    )
    eval_ds = load_autoregressive_dataset(
        dataset,
        split=eval_split,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_field=text_field,
        streaming=streaming,
    )
    return train_ds, eval_ds


__all__ = [
    "load_autoregressive_dataset",
    "load_train_eval_datasets",
]


def _requires_streaming_fallback(exc: Exception) -> bool:
    message = str(exc)
    return "torch_shm_manager" in message or "share_memory" in message or "Operation not permitted" in message
