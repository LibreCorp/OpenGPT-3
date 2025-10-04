import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict

from opengpt3.evaluation.pipeline import EvaluationPipeline
from opengpt3.evaluation.configuration import EvaluateConfig


class Evaluator(object):
    def __init__(self, pipeline: EvaluationPipeline, config: EvaluateConfig):
        self.pipeline = pipeline
        self.config = config

    

    def evaluate(self, from_model: Optional[str] = None) -> Dict[str, float]:
        # Initialize evaluation environment and prepare a dataset.
        self.pipeline.initialize()
        eval_dataset = self.pipeline.build_dataset()
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_eval,
        )

        # Load trained model parameters.
        model = self.pipeline.build_model(from_model).eval()

        # Move the model to GPU device and convert the data type to half
        # precision.
        if self.config.use_gpu:
            model.cuda().half()

        total_metrics = {}
        for _ in self.config.iterate():
            for batch_idx, data in enumerate(eval_loader):
                if self.config.total_steps != -1 and batch_idx >= self.config.total_steps:
                    break
                batch_metrics = self._eval_step(data, model)

                # Record the batched metrics.
                for k, v in batch_metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = []
                    total_metrics[k].append(v)
            break  # Only iterate once through the dataset

        return {k: sum(v) / len(v) for k, v in total_metrics.items()}

    @torch.no_grad()
    def _eval_step(self, data: Dict[str, torch.Tensor], model: nn.Module
                   ) -> Optional[Dict[str, float]]:
        try:
            if self.config.use_gpu:
                data = {k: v.cuda() for k, v in data.items()}

            metrics = self.pipeline.eval_objective(data, model)
            return {k: v.item() for k, v in metrics.items()}
        except StopIteration:
            return None
