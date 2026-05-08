import os, timeit, subprocess
from cs336_systems.modal_utils import app, build_image, user_volume
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import numpy as np


SIZES_MB = [1, 10, 100, 1000]
WORLD_SIZE = [2, 4, 6]





def worker(rank, world_size, size_elems, warmup, iters, q, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    tensor = torch.ones(size_elems, device=f"cuda:{rank}")
    
    
    #warmup
    
    for i in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    
    #timing
    time_iters = []
    dist.barrier()
    
    for i in range (iters):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        time_iters.append(end - start)
        
    final_times = [0] * world_size
    dist.all_gather_object(final_times, time_iters)
        
    if rank == 0:
        total_time = sum([sum(times) for times in final_times])
        avg_time = total_time / (world_size * iters)
        effective_bw_GBps = 2 * size_elems * 4 * (world_size - 1) / (world_size * avg_time * 1e9)
        flat_times = [t for ranks_iters in final_times for t in ranks_iters]
        std = np.std(flat_times)

        row = {
            "world_size": world_size,
            "size_mb": size_elems * 4 / (1024**2),
            "times_sec": final_times,
            "mean_time_sec": avg_time,
            "std_time_sec": std,
            "effective_bw_GBps": effective_bw_GBps,
        }
        q.put(row)
    
    dist.barrier()
    dist.destroy_process_group()
    


@app.function(
    image=build_image(),
    gpu="B200:2",
    timeout=3600,
    volumes={"/root/data": user_volume}
)
def two_benchmark_all_reduce(size_mb: int = 100, world_size: int = 2, warmup: int = 10, iters: int = 100):
    size_elems = size_mb * (1024**2) // 4
    master_port = str(12345 + torch.randint(0, 10000, (1,)).item())
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(
        worker,
        args=(world_size, size_elems, warmup, iters, q, master_port),
        nprocs=world_size,
        join=True,
    )
    return q.get()



@app.function(
    image=build_image(),
    gpu="B200:4",
    timeout=3600,
    volumes={"/root/data": user_volume}
)
def four_benchmark_all_reduce(size_mb: int = 100, world_size: int = 4, warmup: int = 10, iters: int = 100):
    size_elems = size_mb * (1024**2) // 4
    master_port = str(12345 + torch.randint(0, 10000, (1,)).item())
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(
        worker,
        args=(world_size, size_elems, warmup, iters, q, master_port),
        nprocs=world_size,
        join=True,
    )
    return q.get()

@app.function(
    image=build_image(),
    gpu="B200:6",
    timeout=3600,
    volumes={"/root/data": user_volume}
)
def six_benchmark_all_reduce(size_mb: int = 100, world_size: int = 6, warmup: int = 10, iters: int = 100):
    size_elems = size_mb * (1024**2) // 4
    master_port = str(12345 + torch.randint(0, 10000, (1,)).item())
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(
        worker,
        args=(world_size, size_elems, warmup, iters, q, master_port),
        nprocs=world_size,
        join=True,
    )
    return q.get()


@app.local_entrypoint()
def sweep(warmup: int = 10, iters: int = 100, output: str = "all_reduce_benchmark"):
    rows = []
    for world_size in WORLD_SIZE:
        for size_mb in SIZES_MB:
            if world_size == 2:
                row = two_benchmark_all_reduce.remote(size_mb, world_size, warmup, iters)
            elif world_size == 4:
                row = four_benchmark_all_reduce.remote(size_mb, world_size, warmup, iters)
            elif world_size == 6:
                row = six_benchmark_all_reduce.remote(size_mb, world_size, warmup, iters)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{output}.csv", index=False)
    print(df.to_markdown(index=False))