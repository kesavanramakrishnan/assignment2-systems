from cs336_systems.modal_utils import app, build_image, secrets


CTX = 32768
VOCAB = 151936
DMODEL = 4096
DFF = 11008
LAYERS = 34
HEADS = 32
BS = 2
REP_MS = 2000
WARMUP_MS = 500


def fsdp_worker(rank, world_size, q):
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import time
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import triton.testing
    from cs336_basics.model import BasicsTransformerLM
    from torch.optim import AdamW
    import cs336_basics.model as basics_model
    from cs336_basics.nn_utils import cross_entropy
    from cs336_systems.cutile_attention import CuTileFlashAttentionFunction
    from cs336_systems.fsdp import FSDP
    from cs336_systems.fused_ce import FusedCEFunction
    from torch.utils.checkpoint import checkpoint

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    def cutile_sdpa(Q, K, V, mask=None):
        return CuTileFlashAttentionFunction.apply(Q.contiguous(), K.contiguous(), V.contiguous(), True)
    basics_model.scaled_dot_product_attention = cutile_sdpa

    def block_fwd(self, x):
        x_attn = self.attn(self.ln1(x))
        h = x + x_attn
        x_ffn = checkpoint(self.ffn, self.ln2(h), use_reentrant=False)
        return h + x_ffn
    basics_model.TransformerBlock.forward = block_fwd



    torch.manual_seed(0)
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    if rank == 0:
        print(f"world_size={world_size} dtype={dtype}", flush=True)

    build_t0 = time.perf_counter()
    model = BasicsTransformerLM(
        vocab_size=VOCAB,
        context_length=CTX,
        d_model=DMODEL,
        num_layers=LAYERS,
        num_heads=HEADS,
        d_ff=DFF,
    ).to(device=device, dtype=dtype)
    model.checkpoint_block_size = 4

    # lm_head = model.lm_head
    # model.lm_head = nn.Identity()
    # dist.broadcast(lm_head.weight.data, src=0)

    model = FSDP(model, compute_dtype=None)
    optimizer = AdamW(list(model.parameters()), fused=True)
    
    
    elapsed_b = time.perf_counter() - build_t0

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"built FSDP model in {elapsed_b:.1f}s, shard params {n_params / 1e9:.2f}B", flush=True)

    local_bs = BS // world_size
    full_inputs = torch.randint(0, VOCAB, (BS, CTX), device=device)
    full_targets = torch.randint(0, VOCAB, (BS, CTX), device=device)
    inputs = full_inputs[rank * local_bs:(rank + 1) * local_bs]
    targets = full_targets[rank * local_bs:(rank + 1) * local_bs]

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1)).sum()
        loss.backward()
        model.grad_sync()
        optimizer.step()

    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    
    
    one_step_start = time.perf_counter()
    train_step()
    torch.cuda.synchronize()
    dist.barrier()
    
    
    
    elapsed_one = time.perf_counter() - one_step_start
    eager_peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    if rank == 0:
        print(f"one step: {elapsed_one * 1000:.1f} ms, peak mem {eager_peak_gib:.2f} GiB", flush=True)

    num_prof_iters = 5
    fwd_ms = 0.0
    loss_ms = 0.0
    bwd_ms = 0.0
    sync_ms = 0.0
    opt_ms = 0.0
    for i in range(num_prof_iters):
        optimizer.zero_grad(set_to_none=True)
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_fwd = torch.cuda.Event(enable_timing=True)
        ev_loss = torch.cuda.Event(enable_timing=True)
        ev_bwd = torch.cuda.Event(enable_timing=True)
        ev_sync = torch.cuda.Event(enable_timing=True)
        ev_opt = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        # torch.cuda.synchronize()
        # print("before fwd memory", torch.cuda.memory_allocated() / 1024**3, "GiB", flush=True)
        logits = model(inputs)
        ev_fwd.record()
        loss = cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1)).sum()
        # torch.cuda.synchronize()
        # print("after fwd memory", torch.cuda.memory_allocated() / 1024**3, "GiB", flush=True)
        ev_loss.record()
        loss.backward()
        ev_bwd.record()
        model.grad_sync()
        ev_sync.record()
        # torch.cuda.synchronize()
        # print("before optim memory", torch.cuda.memory_allocated() / 1024**3, "GiB", flush=True)
        optimizer.step()
        # torch.cuda.synchronize()
        # print("after optim memory", torch.cuda.memory_allocated() / 1024**3, "GiB", flush=True)
        ev_opt.record()
        torch.cuda.synchronize()
        fwd_ms += ev_start.elapsed_time(ev_fwd)
        loss_ms += ev_fwd.elapsed_time(ev_loss)
        bwd_ms += ev_loss.elapsed_time(ev_bwd)
        sync_ms += ev_bwd.elapsed_time(ev_sync)
        opt_ms += ev_sync.elapsed_time(ev_opt)
        del loss
        del logits
        torch.cuda.empty_cache()
    fwd_ms /= num_prof_iters
    loss_ms /= num_prof_iters
    bwd_ms /= num_prof_iters
    sync_ms /= num_prof_iters
    opt_ms /= num_prof_iters
    total = fwd_ms + loss_ms + bwd_ms + sync_ms + opt_ms
    if rank == 0:
        print("fwd", fwd_ms, "ms", 100*fwd_ms/total, "%", flush=True)
        print("loss", loss_ms, "ms", 100*loss_ms/total, "%", flush=True)
        print("bwd", bwd_ms, "ms", 100*bwd_ms/total, "%", flush=True)
        print("gradsync", sync_ms, "ms", 100*sync_ms/total, "%", flush=True)
        print("opt", opt_ms, "ms", 100*opt_ms/total, "%", flush=True)
        print("total", total, "ms", flush=True)
        print(f"benching rep={REP_MS}ms warmup={WARMUP_MS}ms", flush=True)
    dist.barrier()
    
    
    bench_t0 = time.perf_counter()
    ms = triton.testing.do_bench(train_step, rep=REP_MS, warmup=WARMUP_MS)
    bench_s = time.perf_counter() - bench_t0
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3

    if rank == 0:
        print(f"DONE: {ms:.2f} ms/step, peak mem {peak_gib:.2f} GiB, took {bench_s:.1f}s", flush=True)
        q.put({
            "median_step_ms": float(ms),
            "peak_mem_gib": float(peak_gib),
            "n_params_shard": int(n_params),
            "build_seconds": float(elapsed_b),
            "bench_wall_seconds": float(bench_s),
            "one_step_ms": float(elapsed_one * 1000),
        })

    dist.barrier()
    dist.destroy_process_group()








@app.function(image=build_image(), gpu="B200:2", secrets=secrets(), timeout=1800)
def bench_b200x2():
    import torch.multiprocessing as mp
    world_size = 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    processes = []
    for rank in range(world_size):
        p = ctx.Process(target=fsdp_worker, args=(rank, world_size, q))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    return q.get()


@app.local_entrypoint()
def main():
    result = bench_b200x2.remote()
    print("got back:", result)
