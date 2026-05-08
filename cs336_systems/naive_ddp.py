from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)


    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_grads(self):
        grads = []
        params = []
        for param in self.module.parameters():
            if param.grad is not None:
                grads.append(param.grad.data)
                params.append(param)

        flattened_grads = _flatten_dense_tensors(grads)
        dist.all_reduce(flattened_grads)
        flattened_grads /= self.world_size
        synced_grads = _unflatten_dense_tensors(flattened_grads, grads)

        for param, grad in zip(params, synced_grads):
            param.grad.data.copy_(grad)

def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ddp_model.sync_grads()
    

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_grads(self):
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data)
                param.grad.data /= self.world_size


class OverlapDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook())

    def _make_hook(self):
        def hook(param):
            if param.grad is not None:
                param.grad.data /= self.world_size
                handle = dist.all_reduce(param.grad.data, async_op=True)
                self.handles.append(handle)
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_grads(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8, 16)
        self.rel = nn.ReLU()
        self.lin2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.lin2(self.rel(self.lin1(x)))


def single_ref(x, y, steps: int, lr: float):
    torch.manual_seed(0)
    model = ToyModel()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for i in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    return model.state_dict()


def run_toy_ddp():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(1)
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    steps = 5

    if rank == 0:
        ref_state = single_ref(x, y, steps, 0.1)

    if rank == 0:
        torch.manual_seed(0)
    else:
        torch.manual_seed(1234 + rank)
    model = DDP(ToyModel())
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    local_bs = x.shape[0] // world_size

    for i in range(steps):
        start = rank * local_bs
        end = start + local_bs
        optimizer.zero_grad()
        loss = loss_fn(model(x[start:end]), y[start:end])
        loss.backward()
        model.sync_grads()
        optimizer.step()

    dist.barrier()

    if rank == 0:
        for name, param in model.module.state_dict().items():
            close = torch.allclose(param, ref_state[name], atol=1e-6, rtol=1e-5)
            max_diff = (param - ref_state[name]).abs().max().item()
            print(f"{name}: close={close}, max_diff={max_diff:.3e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    run_toy_ddp()
