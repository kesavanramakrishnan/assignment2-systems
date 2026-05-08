import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.nn_utils import *
from cs336_basics.optimizer import *
from cs336_systems.create_model import *
from cs336_systems.modal_utils import app, build_image, user_volume
from cs336_systems.naive_ddp import DDP
from cs336_systems.sharded_optimizer import ShardedOptimizer
import modal
import numpy as np
import pandas as pd
import os
import timeit



XL_MODEL = ModelConfig(model_type="xl", d_model=2560, d_ff=10240, num_layers=32, num_heads=32, vocab_size=10000, context_length=512)



 

def step(model, optim, samples, targets):
    optim.zero_grad()
    logits = model(samples).reshape(-1, 10000)
    loss = cross_entropy(logits, targets=targets)
    loss.backward()
    torch.cuda.synchronize()
    model.sync_grads()
    torch.cuda.synchronize()
    optim.step()
    torch.cuda.synchronize()


def make_optimizer(model, sharded):
    if sharded:
        return ShardedOptimizer(model.parameters(), AdamW)
    return AdamW(model.parameters())




def worker(rank, world_size, warmup, iters, batch, ctx_length, sharded, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29592"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    config = XL_MODEL
    config.context_length = ctx_length
    torch.manual_seed(0)
    model = create_model(config).to(f"cuda:{rank}")
    model = DDP(model)
    torch.cuda.synchronize()
    mem_after_model_init = torch.cuda.memory_allocated() / (1024**2)
    optim = make_optimizer(model, sharded)
    
    torch.manual_seed(1234 + rank)
    samples = torch.randint(0, 10000, (batch, ctx_length), device=f"cuda:{rank}")
    targets = torch.randint(0, 10000, (batch, ctx_length), device=f"cuda:{rank}").reshape(-1)

    print("warmup")
    for i in range(warmup):
        step(model, optim, samples, targets)
        
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    steps = []
    mem_before = None
    mem_after = None
    print(f"timing: {iters} iters")
    per_iter_timers = []
    for i in range(iters):
        start_time = timeit.default_timer()
        optim.zero_grad()
        logits = model(samples).reshape(-1, 10000)
        loss = cross_entropy(logits, targets=targets)
        loss.backward()
        torch.cuda.synchronize()
        model.sync_grads()
        torch.cuda.synchronize()
        if i == 0:
            mem_before = torch.cuda.memory_allocated() / (1024**2)

        t0 = timeit.default_timer()
        optim.step()
        torch.cuda.synchronize()
        step_time = timeit.default_timer() - t0
        elapsed_iter_ms = timeit.default_timer() - start_time
        per_iter_timers.append(elapsed_iter_ms)
        steps.append(step_time)
        if i == 0:
            mem_after = torch.cuda.memory_allocated() / (1024**2)

    row = {
        "size": "xl",
        "optimizer": "sharded" if sharded else "unsharded",
        "world_size": world_size,
        "node_count": 1,
        "gpus_per_node": 2,
        "batch_per_rank": batch,
        "global_batch": batch * world_size,
        "ctx_length": ctx_length,
        "warmup": warmup,
        "iters": iters,
        "per_iter_ms": float(np.mean(per_iter_timers) * 1000),
        "per_iter_std_ms": float(np.std(per_iter_timers) * 1000),
        "optim_step_mean_ms": float(np.mean(steps) * 1000),
        "optim_step_std_ms": float(np.std(steps) * 1000),
        "mem_after_model_init_mib": mem_after_model_init,
        "mem_before_optim_step_mib": mem_before,
        "mem_after_optim_step_mib": mem_after,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
    }

    if rank == 0:
        q.put(row)

    dist.barrier()
    dist.destroy_process_group()
@app.function(
    image=build_image(),
    gpu="B200:2",
    timeout=2700,
    volumes={"/root/data": user_volume}
)
def profile_remote(warmup, iters, batch, ctx_length, sharded):
    world_size = 2
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(
        worker,
        args=(world_size, warmup, iters, batch, ctx_length, sharded, q),
        nprocs=world_size,
        join=True,
    )
    return q.get()

@app.local_entrypoint()
def sweep(warmup: int = 5, iters: int = 10, batch: int = 1, ctx_length: int = 512, output: str = "sharded_optimizer_benchmark"):
    rows = []
    for sharded in [False, True]:
        row = profile_remote.remote(warmup, iters, batch, ctx_length, sharded)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{output}.csv", index=False)
    print(df.to_markdown(index=False))
