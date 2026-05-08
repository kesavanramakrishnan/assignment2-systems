from __future__ import annotations

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Type


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.all_params = []
        self.local_params = []
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        super().__init__(params, kwargs)
        self.optim = optimizer_cls(self.local_params, **kwargs)
        
    def add_param_group(self, param_group):
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        else:
            params = list(params)

        existing_number = len(self.all_params)
        for i in range(len(params)):
            idx = existing_number + i
            self.all_params.append(params[i])
            if self.owner_of_param(idx) == self.rank:
                self.local_params.append(params[i])

    def owner_of_param(self, idx):
        return idx % self.world_size

    

    def zero_grad(self):
        for param in self.all_params:
            param.grad = None

    def step(self, closure=None, **kwargs):
        loss = None
        loss = self.optim.step(closure=closure, **kwargs)
        for i in range(len(self.all_params)):
            dist.broadcast(self.all_params[i].data, src=self.owner_of_param(i))
        return loss
