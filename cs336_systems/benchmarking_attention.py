import torch
import timeit
from cs336_basics.model import *
from cs336_basics.nn_utils import *
from cs336_basics.optimizer import *
from cs336_systems.create_model import *
import argparse
import numpy.typing as npt
import numpy as np
import modal
import pandas as pd
from cs336_systems.modal_utils import app, build_image, user_volume, secrets
from contextlib import nullcontext
import os, subprocess
from pathlib import Path

HEAD_DIMS = [16, 32, 64, 128]
CTX_LENS = [256, 1024, 4096, 8192, 16384, 32768]




@app.function(
    image=build_image(),
    gpu="B200",
    timeout=2700,
    volumes={"/root/data": user_volume}
)
def benchmark_attention(
    batch: int = 8,
    compile: bool = True
):
    if compile:
        compiled_attn = torch.compile(scaled_dot_product_attention)
    else:
        compiled_attn = scaled_dot_product_attention
    times = {}
    for d_model in HEAD_DIMS:
        for ctx in CTX_LENS:
            try:
                q = torch.randn(batch, ctx, d_model, requires_grad=True, device="cuda")
                k = torch.randn(batch, ctx, d_model, requires_grad=True, device="cuda")
                v = torch.randn(batch, ctx, d_model, requires_grad=True, device="cuda")
                mask = torch.tril(torch.ones(ctx, ctx, dtype=torch.bool, device="cuda"))
                
                # Warmup
                for i in range(10):
                    out = compiled_attn(q, k, v, mask)
                    out.sum().backward()
                    q.grad.zero_()
                    k.grad.zero_()
                    v.grad.zero_()
                    torch.cuda.synchronize()
                
                # Timing
                times_f = []
                times_b = []
                memory_usage = []
                torch.cuda.reset_peak_memory_stats()
                for i in range(100):
                    start_f = torch.cuda.Event(enable_timing=True)
                    end_f = torch.cuda.Event(enable_timing=True)
                    
                    start_f.record()
                    out = compiled_attn(q, k, v, mask)
                    end_f.record()
                    torch.cuda.synchronize()
                    
                    times_f.append(start_f.elapsed_time(end_f))
                    
                    start_b = torch.cuda.Event(enable_timing=True)
                    end_b = torch.cuda.Event(enable_timing=True)
                    memory_usage.append(torch.cuda.memory_allocated() / (1024**2))
                    start_b.record()
                    out.sum().backward()
                    end_b.record()
                    q.grad.zero_()
                    k.grad.zero_()
                    v.grad.zero_()
                    torch.cuda.synchronize()
                    
                    times_b.append(start_b.elapsed_time(end_b))
                avg_f = np.mean(times_f)
                avg_b = np.mean(times_b)
                times[(d_model, ctx)] = {"forward_ms": avg_f, "backward_ms": avg_b, "memory_usage": np.mean(memory_usage)}
                print(f"d_model: {d_model}, ctx: {ctx}, forward: {avg_f:.2f} ms, backward: {avg_b:.2f} ms, memory: {np.mean(memory_usage):.2f} MiB")
            except torch.cuda.OutOfMemoryError:
                print(f"d_model: {d_model}, ctx: {ctx} - OOM")
                times[(d_model, ctx)] = {"forward_ms": None, "backward_ms": None, "memory_usage": None}
                try:
                    del q
                except NameError:
                    pass
                try:
                    del k
                except NameError:
                    pass
                try:                    
                    del v
                except NameError:
                    pass
                try:
                    del mask
                except NameError:
                    pass 
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    return times
            
            
            
            

@app.local_entrypoint()
def main():
    output = benchmark_attention.remote()
    print(f"Attention benchmarking completed. Output: {output}")
