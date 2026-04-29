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

SIZES = {
    "small":  ModelConfig(model_type="small",  d_model=768,  d_ff=3072,  num_layers=12, num_heads=12, vocab_size=10000, context_length=512),
    "medium": ModelConfig(model_type="medium", d_model=1024, d_ff=4096,  num_layers=24, num_heads=16, vocab_size=10000, context_length=512),
    "large":  ModelConfig(model_type="large",  d_model=1280, d_ff=5120,  num_layers=36, num_heads=20, vocab_size=10000, context_length=512),
    "xl":     ModelConfig(model_type="xl",     d_model=2560, d_ff=10240, num_layers=32, num_heads=32, vocab_size=10000, context_length=512),
    # "10B":    ModelConfig(model_type="10B",    d_model=4608, d_ff=12288, num_layers=50, num_heads=36, vocab_size=10000, context_length=512),
}

CTX_LENS = [256, 1024, 2048]
NSYS_SIZES = ["small", "medium", "large", "xl"]



def run_model(
    mode,
    model,
    optim,
    input,
    target
):
    if mode == "f":
        model.forward(input)
    elif mode == "fb":
        logits = model.forward(input).reshape(-1, 10000)
        loss = cross_entropy(logits, targets=target)
        loss.backward()
        optim.zero_grad()
    elif mode == "train":
        logits = model.forward(input).reshape(-1, 10000)
        loss = cross_entropy(logits, targets=target)
        loss.backward()
        optim.step()
        optim.zero_grad()


def time_model(
    mode,
    model,
    optim,
    input,
    target, 
    autocast: bool = False
):
    cm = nullcontext() if not autocast else torch.autocast(device_type=input.device.type, dtype=torch.bfloat16)
    
    if mode == "f":
        t0 = timeit.default_timer()
        with torch.inference_mode():
            model.forward(input)
        torch.cuda.synchronize()
        elapsed = timeit.default_timer() - t0
        return {"f": elapsed}
    elif mode == "fb":
        t0 = timeit.default_timer()
        with cm:
            logits = model.forward(input).reshape(-1, 10000)
            loss = cross_entropy(logits, targets=target)
        torch.cuda.synchronize()
        elapsed_f = timeit.default_timer() - t0
        t1 = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        elapsed_b = timeit.default_timer() - t1
        optim.zero_grad()
        return {"f": elapsed_f, "b": elapsed_b}
    elif mode == "train":
        t0 = timeit.default_timer()
        logits = model.forward(input).reshape(-1, 10000)
        loss = cross_entropy(logits, targets=target)
        torch.cuda.synchronize()
        elapsed_f = timeit.default_timer() - t0
        t1 = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        elapsed_b = timeit.default_timer() - t1
        t2 = timeit.default_timer()
        optim.step()
        torch.cuda.synchronize()
        elapsed_o = timeit.default_timer() - t2
        optim.zero_grad()
        return {"f": elapsed_f, "b": elapsed_b, "o": elapsed_o}




def profile_model(
    config: ModelConfig,
    mode: str,
    device: str,
    warmup: int,
    iters: int,
    batch: int,
    ctx_length: int | None = None,
    autocast: bool = False
):
    if ctx_length is not None:
        config.context_length = ctx_length
    model = create_model(config)
    model.to(device=device)
    optim = AdamW(params=model.parameters())
    samples = torch.randint(0, 10000, (batch, config.context_length), device=device)
    targets = torch.randint(0, 10000, (batch, config.context_length), device=device).reshape(-1)
    
    print("Start warmup")
    for i in range(warmup):
        time_model(mode=mode, model=model, optim=optim, input=samples, target=targets, autocast=autocast)   
         
    print(f"Start timing: {iters} iters")
    times_f = []
    times_b = []
    times_o = []
    for i in range(iters):
        output = time_model(mode=mode, model=model, optim=optim, input=samples, target=targets, autocast=autocast)
        if mode == "f" or mode == "fb" or mode == "train":
            times_f.append(output["f"])
            if mode == "fb" or mode == "train":
                times_b.append(output["b"])
            if mode == "train":
                times_o.append(output["o"])
                
    output = {
        "size": config.model_type,
        "mode": mode,
        "warmup": warmup,
        "iters": iters,
        "autocast": autocast
    }
    if mode == "f" or mode == "fb" or mode == "train":
        output["forward_mean_ms"] = np.mean(times_f) * 1000
        output["forward_std_ms"] = np.std(times_f) * 1000
        print(f"mean_forward={output['forward_mean_ms']:.2f} ms,  {output['forward_std_ms']:.2f} ms, (n={iters})")
        if mode == "fb" or mode == "train":
            output["backward_mean_ms"] = np.mean(times_b) * 1000
            output["backward_std_ms"] = np.std(times_b) * 1000
            print(f"mean_backward={output['backward_mean_ms']:.2f} ms,  {output['backward_std_ms']:.2f} ms, (n={iters})")
        if mode == "train":
            output["optim_mean_ms"] = np.mean(times_o) * 1000
            output["optim_std_ms"] = np.std(times_o) * 1000
            print(f"mean_optim={output['optim_mean_ms']:.2f} ms,  {output['optim_std_ms']:.2f} ms, (n={iters})")
    output["ctx_length"] = config.context_length
    
    del model, optim
    torch.cuda.empty_cache()
    return output
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, type=str)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iters", required=True, type=int)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--config", required=True, choices=sorted(SIZES))
    parser.add_argument("--ctx_length", required=True, type=int)
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--device", required=True, type=str)
    args = parser.parse_args()
    profile_model(
        config=SIZES[args.config],
        mode=args.mode,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        batch=args.batch,
        ctx_length=args.ctx_length,
        autocast=args.autocast
    )

@app.function(
    image=build_image(),
    gpu="B200",
    timeout=2700,
    volumes={"/root/data": user_volume}
)
def profile_remote(mode, warmup, iters, batch, config, device, ctx_length: int | None = None, autocast: bool = False):
    profile_model(
        mode=mode,
        warmup=warmup,
        iters=iters,
        batch=batch,
        config=config,
        device=device,
        ctx_length=ctx_length,
        autocast=autocast
    )

@app.function(
    image=build_image(),
    gpu="B200",
    timeout=2700,
    volumes={"/root/data": user_volume}
)
def run_sweep(spec: list[dict]) -> list[dict]:
    rows = []
    for setup in spec:
        print(setup)
        out = profile_model(
            config=setup["model_config"],
            mode=setup["mode"],
            device=setup["device"],
            warmup=setup["warmup"],
            iters=setup["iters"],
            batch=setup["batch"],
            ctx_length=setup["ctx_length"],
            autocast=setup["autocast"]
        )
        rows.append(out)
        print(out)
    return rows
    
@app.local_entrypoint()
def sweep(warmup: int = 5, iters: int = 10, batch: int = 4, device: str = "cuda", output: str = "sweep_results", sizes: str = ",".join(NSYS_SIZES), ctx_lengths: str = ",".join(str(x) for x in CTX_LENS)):
    spec = []
    selected_sizes = [size.strip() for size in sizes.split(",") if size.strip()]
    selected_ctx_lengths = [int(ctx.strip()) for ctx in ctx_lengths.split(",") if ctx.strip()]
    for auto in [False, True]:
        for size in selected_sizes:
            model_config = SIZES[size]
            for mode in ["fb"]:
                for ctx_length in selected_ctx_lengths:
                    spec.append({"model_config": model_config, "mode": mode, "warmup": warmup, "iters": iters, "batch": batch, "device": device, "autocast": auto, "ctx_length": ctx_length})
    rows = run_sweep.remote(spec)
    df = pd.DataFrame(rows)
    df.to_csv(f"{output}.csv", index=False)


@app.local_entrypoint()
def modal_main(mode: str, warmup: int, iters: int, batch: int, config: str, ctx_length: int, device: str = "cuda", autocast: bool = False):
    cfg = load_config(config)
    profile_remote.remote(mode, warmup, iters, batch, cfg, device, ctx_length, autocast)
