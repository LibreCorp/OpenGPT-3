import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from opengpt3.training import TrainingSpec, TrainConfig, Recorder
from typing import Dict, Optional

try:
    from apex import amp
except ModuleNotFoundError:
    pass

import warnings
warnings.filterwarnings(action='ignore')
import os
import json


class Trainer(object):
    def __init__(self, spec: TrainingSpec, config: TrainConfig):
        self.spec = spec
        self.config = config

    def train(self,
              from_checkpoint: Optional[str] = None,
              from_pretrained: Optional[str] = None):
        if self.config.distributed:
            mp.spawn(self._train, args=(from_checkpoint, from_pretrained),
                     nprocs=self.config.gpus)
        else:
            self._train(0, from_checkpoint, from_pretrained)

    def _train(self,
               rank: int,
               from_checkpoint: Optional[str] = None,
               from_pretrained: Optional[str] = None):
        if self.config.distributed:
            torch.cuda.set_device(rank)
            dist.init_process_group(backend='nccl',
                                    init_method='tcp://127.0.0.1:8000',
                                    world_size=self.config.gpus,
                                    rank=rank)

        # training env and prepare datasets.
        self.spec.initialize()
        train_dataset, eval_dataset = self.spec.prepare_datasets()

        # TODO: support distributed sampler for datasets.
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_train,
            collate_fn=self._collate_fn,
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_eval,
            collate_fn=self._collate_fn,
        )
        self.train_iter = iter(self.train_loader)
        self.eval_iter = iter(self.eval_loader)



        model = self.spec.construct_model().to(self.config.device)
        if from_pretrained:
            ckpt = torch.load(from_pretrained, map_location=self.config.device)
            model.load_state_dict(ckpt['model'])

            # just in case
            del ckpt
            torch.cuda.empty_cache()

        optimizer, scheduler = self.spec.create_optimizer(model.parameters())
        recorder = Recorder()

        if self.config.use_amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', verbosity=0)

        if self.config.distributed:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank])

        start_step = 0



        if from_checkpoint:
            ckpt = torch.load(from_checkpoint, map_location=self.config.device)

            start_step = ckpt['step']
            recorder = ckpt['recorder']

            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])

            train_dataset.assign(ckpt['train_dataset'])
            eval_dataset.assign(ckpt['eval_dataset'])

            if self.config.use_amp:
                amp.load_state_dict(ckpt['amp'])

            # again, just in case
            del ckpt
            torch.cuda.empty_cache()

        if rank == 0:
            training_iters = tqdm.tqdm( # not sure this is needed at all, could just leave simple logging on every --n_log
                range(start_step + 1, self.config.total_steps),
                total=self.config.total_steps,
                desc=self.config.description,
                dynamic_ncols=True)
            training_iters.update(start_step + 1)
        else:
            # In other processes, use simple iterator rather than tqdm one.
            training_iters = range(start_step + 1, self.config.total_steps)

        for step in training_iters:
            torch.cuda.empty_cache() # not sure if this is needed, but let it be here

            recorder.record(
                self._train_step(rank, model, optimizer, scheduler),
                scope='train')

            if (step + 1) % self.config.log_steps == 0:
                recorder.stamp(step, scope='train')
                if rank == 0:
                    training_iters.set_postfix_str(
                        recorder.format(self.config.log_format))


            torch.cuda.empty_cache()


            if (step + 1) % self.config.eval_steps == 0:
                recorder.record(
                    self._eval_step(rank, model), scope='eval')
                recorder.stamp(step, scope='eval')

                if rank == 0:
                    training_iters.set_postfix_str(
                        recorder.format(self.config.log_format))



            if rank == 0 and (step + 1) % self.config.save_steps == 0:
                ckpt = {
                    'step': step,
                    'recorder': recorder,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_dataset': train_dataset.where(),
                    'eval_dataset': eval_dataset.where()
                }

                if self.config.use_amp:
                    ckpt['amp'] = amp.state_dict()

                torch.save(ckpt, self.config.save_checkpoint_path)

        # Since the model is wrapped with `DistributedDataParallel` class in
        # distributed training environment, the original model can be accessed
        # by `module` attribute.
        if self.config.distributed:
            model = model.module

        if rank == 0:
            save_path = self.config.save_model_path
            # If no extension or endswith sep, treat as output directory
            if save_path.endswith(os.sep) or os.path.splitext(save_path)[1] == '':
                out_dir = save_path.rstrip(os.sep)
                os.makedirs(out_dir, exist_ok=True)
                # safetensors here
                model.cpu().save_pretrained(out_dir)
                self.spec.tokenizer.save_pretrained(out_dir)
                # save metrics
                with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
                    json.dump(recorder.metrics, f)
            else:
                # Legacy single-file save
                # just in case
                torch.save({'model': model.cpu().state_dict()},
                           save_path)

    def _train_step(
        self,
        rank: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
    ) -> Dict[str, float]:
        model.train()
        optimizer.zero_grad()

        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        metrics = self.spec.train_objective(batch, model)
        loss = metrics['loss']

        if self.config.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()

        return {k: self._to_value(v) for k, v in metrics.items()}

    @torch.no_grad()
    def _eval_step(
        self,
        rank: int,
        model: nn.Module,
    ) -> Dict[str, float]:
        model.eval()

        try:
            batch = next(self.eval_iter)
        except StopIteration:
            self.eval_iter = iter(self.eval_loader)
            batch = next(self.eval_iter)
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        metrics = self.spec.eval_objective(batch, model)
        return {k: self._to_value(v) for k, v in metrics.items()}


    def _to_value(self, tensor: torch.Tensor) -> float:
        if self.config.distributed:
            tensor = tensor.clone()
            dist.all_reduce(tensor, op=dist.reduce_op.SUM)
            return (tensor / self.config.gpus).item()
        else:
            return tensor.item()

    def _collate_fn(self, batch):
        # collate 'input' and 'output' fields into batched tensors
        return {
            field: torch.tensor([ex[field] for ex in batch], dtype=torch.long)
            for field in ('input', 'output')
        }
