# OpenGPT-3 Sample Commands

This document provides sample commands for interacting with the OpenGPT-3 model.

## Prerequisites

Ensure you have set your `PYTHONPATH` environment variable:
```bash
export PYTHONPATH=./src
```

## 1. Training a Model

To train a new model or continue training an existing one:

```bash
python3 -m opengpt3 train \
  --dataset HuggingFaceFW/fineweb \
  --train_split train \
  --eval_split train \
  --tokenizer_path gpt2 \
  --batch_train 2 \
  --batch_eval 2 \
  --total_steps 500 \
  --seq_len 64 \
  --layers 12 \
  --heads 16 \
  --dims 1024 \
  --rate 4 \
  --dropout 0.1 \
  --base_lr 1e-4 \
  --wd_rate 1e-2 \
  --save_model_path model/ \
  --save_checkpoint_path checkpoint/checkpoint.pth \
  --device mps # or cuda, cpu | or do not set and devide shall be picked automatically
```

**Note**:
- Adjust `total_steps`, `batch_train`, `batch_eval`, and model configuration parameters (`seq_len`, `layers`, `heads`, `dims`, `rate`, `dropout`) as needed for your training run.
- The `model/` directory will be created if it doesn't exist, and the trained model and tokenizer will be saved there.

## 2. Generating Text

To generate text using a trained model:

```bash
python3 -m opengpt3 generate \
  --model_path model/ \
  --tokenizer_path model/ \
  --prompt "He is a doctor. His main goal is" \
  --seq_len 64 \
  --nucleus_prob 0.85 \
  --temperature 1.0 \
  --use_gpu # Optional: if you want to use GPU
```

**Note**:
- `--model_path` and `--tokenizer_path` should point to your trained model and tokenizer directories (e.g., `model/`).
- Adjust `--prompt`, `--seq_len`, `--nucleus_prob`, and `--temperature` for desired generation behavior.

## 3. Evaluating a Model

To evaluate a trained model on a dataset:

```bash
python3 -m opengpt3 evaluate \
  --model_path model/ \
  --tokenizer_path model/ \
  --dataset HuggingFaceFW/fineweb \
  --eval_split train \
  --total_steps 100 \
  --batch_eval 16 \
  --seq_len 64 \
  --layers 12 \
  --heads 16 \
  --dims 1024 \
  --rate 4 \
  --use_gpu # Optional: if you want to use GPU
  # or --device cpu/cuda/mps or any other torch supprts
```

**Note**:
- `--model_path` and `--tokenizer_path` should point to your trained model and tokenizer directories.
- Adjust `--dataset`, `--eval_split`, `--total_steps`, and `--batch_eval` as needed.
