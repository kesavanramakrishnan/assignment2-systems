import os
import timeit
import subprocess
import argparse
from pathlib import Path
import modal
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx
from cs336_basics.nn_utils import *
from cs336_basics.optimizer import *
from cs336_systems.create_model import *
from cs336_systems.fsdp import FSDP
from cs336_systems.modal_utils import app, build_image, user_volume
from cs336_systems.naive_ddp import DDP


XL_MODEL = ModelConfig(model_type="xl", d_model=2560, d_ff=10240, num_layers=32, num_heads=32, vocab_size=10000, context_length=512)
DEFAULT_NSYS_FLAGS = ["--trace=cuda,nvtx,osrt", "--force-overwrite=true"]


def make_model(model, fsdp_type):
    if fsdp_type == "ddp":
        return DDP(model)
    if fsdp_type == "fsdp":
        return FSDP(model)
    raise ValueError(f"bad fsdp_type: {fsdp_type}")

def sync_model(model, fsdp_type):
    if fsdp_type == "ddp":
        model.sync_grads()
    elif fsdp_type == "fsdp":
        model.finish_gradient_synchronization()


def step(model, optim, samples, targets, fsdp_type):
    with nvtx.range("zero_grad"):
        optim.zero_grad()
    with nvtx.range("forward_loss"):
        logits = model(samples).reshape(-1, 10000)
        loss = cross_entropy(logits, targets=targets)
    with nvtx.range("backward"):
        loss.backward()
    torch.cuda.synchronize()
    with nvtx.range("sync_grads"):
        sync_model(model, fsdp_type)
    torch.cuda.synchronize()
    with nvtx.range("optim_step"):
        optim.step()
    torch.cuda.synchronize()


def worker(rank, world_size, warmup, iters, batch, ctx_length, fsdp_type, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29593"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    
    config = XL_MODEL
    config.context_length = ctx_length
    torch.manual_seed(0)
    model = create_model(config).to(f"cuda:{rank}")
    model = make_model(model, fsdp_type)
    torch.cuda.synchronize()
    mem_after_model = torch.cuda.memory_allocated() / (1024**2)

    optim = AdamW(params=model.parameters())
    torch.manual_seed(1234 + rank)
    samples = torch.randint(0, 10000, (batch, ctx_length), device=f"cuda:{rank}")
    targets = torch.randint(0, 10000, (batch, ctx_length), device=f"cuda:{rank}").reshape(-1)

    print("warmup")
    with nvtx.range("warmup"):
        for i in range(warmup):
            step(model, optim, samples, targets, fsdp_type)

    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print(f"timing: {iters} iters")
    times = []
    mem_before = None
    mem_after = None
    with nvtx.range("timing"):
        for i in range(iters):
            t0 = timeit.default_timer()
            optim.zero_grad()
            with nvtx.range("forward_loss"):
                logits = model(samples).reshape(-1, 10000)
                loss = cross_entropy(logits, targets=targets)
            with nvtx.range("backward"):
                loss.backward()
            torch.cuda.synchronize()
            with nvtx.range("grad_sync"):
                sync_model(model, fsdp_type)
            torch.cuda.synchronize()
            if i == 0:
                mem_before = torch.cuda.memory_allocated() / (1024**2)

            with nvtx.range("optim_step"):
                optim.step()
            torch.cuda.synchronize()
            elapsed = timeit.default_timer() - t0
            times.append(elapsed)
            if i == 0:
                mem_after = torch.cuda.memory_allocated() / (1024**2)

    row = {
        "size": "xl",
        "fsdp_type": fsdp_type,
        "world_size": world_size,
        "node_count": 1,
        "gpus_per_node": 2,
        "batch_per_rank": batch,
        "global_batch": batch * world_size,
        "ctx_length": ctx_length,
        "warmup": warmup,
        "iters": iters,
        "per_iter_ms": float(np.mean(times) * 1000),
        "per_iter_std_ms": float(np.std(times) * 1000),
        "mem_after_model_init_mib": mem_after_model,
        "mem_before_optim_step_mib": mem_before,
        "mem_after_optim_step_mib": mem_after,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
    }

    if rank == 0: q.put(row)

    dist.barrier()
    dist.destroy_process_group()


@app.function(
    image=build_image(),
    gpu="B200:2",
    timeout=2700,
    volumes={"/root/data": user_volume}
)
def profile_remote(warmup, iters, batch, ctx_length, fsdp_type):
    world_size = 2
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(
        worker,
        args=(world_size, warmup, iters, batch, ctx_length, fsdp_type, q),
        nprocs=world_size,
        join=True,
    )
    return q.get()






def run_local_worker(warmup, iters, batch, ctx_length, fsdp_type):
    world_size = 2
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(
        worker,
        args=(world_size, warmup, iters, batch, ctx_length, fsdp_type, q),
        nprocs=world_size,
        join=True,
    )
    print(q.get())


@app.local_entrypoint()
def sweep(warmup: int = 5, iters: int = 10, batch: int = 1, ctx_length: int = 512, output: str = "fsdp_benchmark"):
    rows = []
    for fsdp_type in ["ddp", "fsdp"]:
        row = profile_remote.remote(warmup, iters, batch, ctx_length, fsdp_type)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{output}.csv", index=False)
    print(df.to_markdown(index=False))





@app.function(
    image=build_image(),
    gpu="B200:2",
    timeout=2700,
    volumes={"/root/data": user_volume}
)
def nsys_remote(warmup, iters, batch, ctx_length, fsdp_type, output, dir="/root/data/nsys-output"):
    output_path = f"{dir}/{output}"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "nsys", "profile",
        *DEFAULT_NSYS_FLAGS,
        "-o", output_path,
        "--",
        "python", "-m", "cs336_systems.benchmarking_fsdp",
        "--run-local-worker",
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--batch", str(batch),
        "--ctx_length", str(ctx_length),
        "--fsdp_type", fsdp_type,
    ], check=True)
    user_volume.commit()


@app.local_entrypoint()
def nsys(warmup: int = 1, iters: int = 2, batch: int = 1, ctx_length: int = 512, fsdp_type: str = "fsdp", output: str = "fsdp_profile"):
    nsys_remote.remote(warmup, iters, batch, ctx_length, fsdp_type, output)
    os.makedirs("nsys-output", exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "basics-kesavanr", f"nsys-output/{output}.nsys-rep", "nsys-output/"], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-local-worker", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ctx_length", type=int, default=512)
    parser.add_argument("--fsdp_type", type=str, default="fsdp")
    args = parser.parse_args()
    if args.run_local_worker:
        run_local_worker(args.warmup, args.iters, args.batch, args.ctx_length, args.fsdp_type)
