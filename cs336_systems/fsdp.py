from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.nvtx as nvtx
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from cs336_basics.model import Linear, Embedding
import argparse
import os



class FSDPEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_shard, dtype, weight):
        if weight is None:
            weight = torch.empty((weight_shard.shape[0] * dist.get_world_size(), weight_shard.shape[1]), device=weight_shard.device, dtype=weight_shard.dtype)
            with torch.no_grad():
                dist.all_gather_into_tensor(weight, weight_shard)
        if dtype is not None:
            output = F.embedding(input, weight.to(dtype)).to(weight_shard.dtype)
        else:
            output = F.embedding(input, weight)
        ctx.dtype = dtype
        ctx.save_for_backward(input, weight_shard)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight_shard = ctx.saved_tensors
        world_size = dist.get_world_size()
        full_rows = weight_shard.shape[0] * world_size
        grad_full_weight = torch.zeros((full_rows, weight_shard.shape[1]), device=weight_shard.device, dtype=weight_shard.dtype)
        grad_full_weight.index_add_(0, input.reshape(-1), grad_output.reshape(-1, weight_shard.shape[1]).to(weight_shard.dtype))

        grad_weight_shard = torch.empty_like(weight_shard)
        dist.reduce_scatter_tensor(grad_weight_shard, grad_full_weight)
        grad_weight_shard /= world_size
        return None, grad_weight_shard, None, None

class FSDPLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_shard, bias, dtype, weight):
        if weight is None:
            weight = torch.empty((weight_shard.shape[0] * dist.get_world_size(), weight_shard.shape[1]), device=weight_shard.device, dtype=weight_shard.dtype)
            with torch.no_grad():
                dist.all_gather_into_tensor(weight, weight_shard)
        if dtype is not None:
            if bias is not None:
                output = F.linear(input.to(dtype), weight.to(dtype), bias.to(dtype))
            else:
                output = F.linear(input.to(dtype), weight.to(dtype), None)
            output = output.to(input.dtype)
        else:
            output = F.linear(input, weight, bias)
        ctx.has_bias = bias is not None
        ctx.dtype = dtype
        if bias is not None:
            ctx.save_for_backward(input, weight_shard, bias)
        else:
            ctx.save_for_backward(input, weight_shard)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            input, weight_shard, bias = ctx.saved_tensors
        else:
            input, weight_shard = ctx.saved_tensors
            bias = None
        
        world_size = dist.get_world_size()
        full_weight = torch.empty((weight_shard.shape[0] * dist.get_world_size(), weight_shard.shape[1]), device=weight_shard.device, dtype=weight_shard.dtype)
        dist.all_gather_into_tensor(full_weight, weight_shard)

        dtype = ctx.dtype
        if dtype is not None:
            go = grad_output.to(dtype)
            inp = input.to(dtype)
            w = full_weight.to(dtype)
        else:
            go = grad_output
            inp = input
            w = full_weight

        grad_input = go @ w
        grad_input = grad_input.to(input.dtype)
        grad_full_weight = go.reshape(-1, full_weight.shape[0]).t() @ inp.reshape(-1, full_weight.shape[1])
        grad_full_weight = grad_full_weight.to(weight_shard.dtype)
        
        grad_weight_shard = torch.empty_like(weight_shard)
        dist.reduce_scatter_tensor(grad_weight_shard, grad_full_weight)
        grad_weight_shard /= world_size

        if bias is not None:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
        else:
            grad_bias = None
        return grad_input, grad_weight_shard, grad_bias, None, None


 
class FSDPLinear(Linear):
    def __init__(self, linear: nn.Module, compute_dtype: torch.dtype | None = None):
        nn.Module.__init__(self)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self.dtype = compute_dtype
        self.full_shape = linear.weight.shape
        shard_size = linear.weight.shape[0] // world_size
        self.weight_shard = nn.Parameter(linear.weight[rank*shard_size:(rank+1)*shard_size].clone())
        if hasattr(linear, "bias") and linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.clone())
        else:
            self.bias = None


        self.in_features = linear.weight.shape[1]
        self.out_features = linear.weight.shape[0]
        
        
        self.weight = None
        self.handle = None

    def prefetch(self):
        if self.handle is not None:
            return
        if self.weight is not None:
            return
        self.weight = torch.empty(self.full_shape, device=self.weight_shard.device, dtype=self.weight_shard.dtype)
        with torch.no_grad():
            with nvtx.range("fsdp_prefetch_linear"):
                self.handle = dist.all_gather_into_tensor(self.weight, self.weight_shard, async_op=True)

    def forward(self, input):
        if self.handle is not None:
            with nvtx.range("fsdp_wait_linear"):
                self.handle.wait()
            self.handle = None
        weight = self.weight
        self.weight = None
        return FSDPLinearFunction.apply(input, self.weight_shard, self.bias, self.dtype, weight)




class FSDPEmbedding(Embedding):
    def __init__(self, embedding: nn.Module, compute_dtype: torch.dtype | None = None):
        nn.Module.__init__(self)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        self.handle = None
        self.dtype = compute_dtype
        self.full_shape = embedding.weight.shape
        shard_size = embedding.weight.shape[0] // world_size
        self.weight_shard = nn.Parameter(embedding.weight[rank*shard_size:(rank+1)*shard_size].clone())
        self.weight = None

    def prefetch(self):
        if self.handle is not None:
            return
        if self.weight is not None:
            return
        self.weight = torch.empty(self.full_shape, device=self.weight_shard.device, dtype=self.weight_shard.dtype)
        with torch.no_grad():
            with nvtx.range("fsdp_prefetch_embedding"):
                self.handle = dist.all_gather_into_tensor(self.weight, self.weight_shard, async_op=True)

    def forward(self, input):
        if self.handle is not None:
            with nvtx.range("fsdp_wait_embedding"):
                self.handle.wait()
            self.handle = None
        weight = self.weight
        self.weight = None
        return FSDPEmbeddingFunction.apply(input, self.weight_shard, self.dtype, weight)
        




class FSDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        self.layers = []
        self.wrap(self.module)
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self.hook(i))

    def wrap(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.Linear, Linear)):
                module._modules[name] = FSDPLinear(child, compute_dtype=self.compute_dtype)
                self.layers.append(module._modules[name])
            elif isinstance(child, (nn.Embedding, Embedding)):
                module._modules[name] = FSDPEmbedding(child, compute_dtype=self.compute_dtype)
                self.layers.append(module._modules[name])
            else:
                self.wrap(child)

    def hook(self, i):
        def hook(module, input, output):
            next = i + 2
            if next < len(self.layers):
                self.layers[next].prefetch()
        return hook

    def forward(self, *args, **kwargs):
        if len(self.layers) > 0:
            self.layers[0].prefetch()
            
        if len(self.layers) > 1:
            self.layers[1].prefetch()
        return self.module(*args, **kwargs)

    def grad_sync(self):
        for name, param in self.module.named_parameters():
            if param.grad is not None and "weight_shard" not in name:
                dist.all_reduce(param.grad.data)
                param.grad.data /= dist.get_world_size()

    def finish_gradient_synchronization(self):
        self.grad_sync()


def gather_full_params(fsdp_model):
    out = {}
    for name, param in fsdp_model.module.named_parameters():
        if "weight_shard" in name:
            full_name = name.removesuffix("weight_shard") + "weight"
            new_name = name.removesuffix(".weight_shard")
            moded = fsdp_model.module.get_submodule(new_name)
            full = torch.empty(moded.full_shape, device=param.device, dtype=param.dtype)
            
            dist.all_gather_into_tensor(full, param.data)
            out[full_name] = full
        else:
            out[name] = param.data
    return out
