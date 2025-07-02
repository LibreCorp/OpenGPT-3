> **Language Models are Few-Shot Learners**
> Brown, T. et al. (2020). *Language Models are Few-Shot Learners*. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

## Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Usage Overview](#usage-overview)
* [Mixed-Precision & Optimizations](#mixed-precision--optimizations)
* [License](#license)

---

## Introduction

OpenGPT-3 is a PyTorch-based replication of the GPT-3 architecture, designed to mirror the original model’s mechanisms and scaling laws. As a fork of the official GPT-2 implementation, this project emphasizes fidelity over finetuning pipelines, providing:

* **Training** of transformer models at GPT-3 scale
* **Zero- and few-shot** text generation
* **Evaluation** on held-out corpora (perplexity, etc.)
* **Interactive visualization** of training metrics

More detailed documentation is available on DeepWiki: https://deepwiki.com/LibreCorp/OpenGPT-3

---

## Features

* Configurable model dimensions, depth, and attention heads to match various GPT-3 sizes
* Support for long context windows (up to 2 048 tokens)
* Resume training from checkpoints or leverage pretrained GPT-2 weights
* Gradient checkpointing for memory efficiency
* NVIDIA Apex integration for mixed-precision and fused CUDA kernels
* Modular codebase for easy experimentation and extension

---

## Dependencies

* Python 3.8+
* `torch` 1.12+
* `transformers`
* `numpy`
* `tqdm`
* `regex`
* `matplotlib`

---

## Installation

Clone this repository and install requirements:

```bash
git clone https://github.com/OpenGPT-3/OpenGPT-3.git
cd OpenGPT-3
pip install -r requirements.txt
```

To enable NVIDIA Apex optimizations:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```

---

## Usage Overview

1. **Prepare data**
   Tokenize your text corpus and build a vocabulary (e.g., with [Expanda](https://github.com/affjljoo3581/Expanda)).

2. **Configure model**
   Edit configuration (hidden layer dimension, number of layers, heads, feed-forward size, sequence length) in the training script or a YAML/JSON config file.

3. **Train**
   Launch training to learn transformer weights from scratch or from a GPT-2 checkpoint.

4. **Generate**
   Use the trained model for zero- or few-shot completion by providing prompts and sampling parameters (temperature, top-p, number of shots).

5. **Evaluate**
   Compute language modeling metrics (perplexity, cross-entropy) on held-out text.

6. **Visualize**
   Plot training loss curves and attention statistics using the built-in visualization module.

---

## Mixed-Precision & Optimizations

* **Automatic Mixed Precision** (`--use_amp`) for faster training on Tensor-Core GPUs
* **Fused optimizers** and layer norms via NVIDIA Apex
* **Gradient checkpointing** (`--use_grad_ckpt`) to trade compute for memory

Ensure that your GPU supports mixed-precision acceleration before enabling these options.

---

## License

This project is licensed under the [Apache 2.0 License](./LICENSE).
