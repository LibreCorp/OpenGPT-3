from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

Batch = Dict[str, torch.Tensor]
Objective = Callable[[Batch, nn.Module], Dict[str, torch.Tensor]]


@dataclass
class EvaluationPipeline:
    initialize: Callable[[], None]
    build_dataset: Callable[[], Dataset]
    build_model: Callable[[Optional[str]], nn.Module]
    eval_objective: Objective


__all__ = ["EvaluationPipeline", "Batch", "Objective"]
