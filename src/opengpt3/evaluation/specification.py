import torch
import torch.nn as nn
from opengpt3.data import Dataset
from typing import Dict, Optional


class EvaluationSpec(object):
    def initialize(self):
        pass

    def prepare_dataset(self) -> Dataset:
        raise NotImplementedError()

    def construct_model(self, from_model: Optional[str] = None) -> nn.Module:
        raise NotImplementedError()

    def eval_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                       ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
