from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

Batch = Dict[str, torch.Tensor]
Objective = Callable[[Batch, nn.Module], Dict[str, torch.Tensor]]


@dataclass
class TrainingPipeline:
    initialize: Callable[[], None]
    build_datasets: Callable[[], Tuple[Dataset, Dataset]]
    build_model: Callable[[], nn.Module]
    create_optimizer: Callable[[Iterator[nn.Parameter]], Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]]
    train_objective: Objective
    eval_objective: Objective
    on_save: Callable[[str], None] = field(default=lambda _: None)


__all__ = ["TrainingPipeline", "Batch", "Objective"]
