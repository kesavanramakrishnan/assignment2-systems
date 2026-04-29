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
import torch.cuda.nvtx as nvtx
import pandas as pd
from cs336_systems.modal_utils import app, build_image, user_volume, secrets
import subprocess
import os
from pathlib import Path

SIZES = {
    "small":  ModelConfig(model_type="small",  d_model=768,  d_ff=3072,  num_layers=12, num_heads=12, vocab_size=10000, context_length=512),
    # "medium": ModelConfig(model_type="medium", d_model=1024, d_ff=4096,  num_layers=24, num_heads=16, vocab_size=10000, context_length=512),
    "large":  ModelConfig(model_type="large",  d_model=1280, d_ff=5120,  num_layers=36, num_heads=20, vocab_size=10000, context_length=512),
    # "xl":     ModelConfig(model_type="xl",     d_model=2560, d_ff=10240, num_layers=32, num_heads=32, vocab_size=10000, context_length=512),
    # "10B":    ModelConfig(model_type="10B",    d_model=4608, d_ff=12288, num_layers=50, num_heads=36, vocab_size=10000, context_length=512),
}
CTX_LENS = [256, 1024, 2048]

DEFAULT_NSYS_FLAGS = ["--trace=cuda,nvtx", "--pytorch=functions-trace", "--force-overwrite=true"]




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


def run_model_nvtx(
    mode,
    model,
    optim,
    input,
    target
):
    if mode == "f":
        t0 = timeit.default_timer()
        with nvtx.range("forward"):
            with torch.inference_mode():
                model.forward(input)
        elapsed = timeit.default_timer() - t0
        return {"f": elapsed}
    elif mode == "fb":
        t0 = timeit.default_timer()
        with nvtx.range("forward"):
            logits = model.forward(input).reshape(-1, 10000)
        with nvtx.range("loss"):
            loss = cross_entropy(logits, targets=target)
        elapsed_f = timeit.default_timer() - t0
        t1 = timeit.default_timer()
        with nvtx.range("backward"):
            loss.backward()
        elapsed_b = timeit.default_timer() - t1
        optim.zero_grad()
        return {"f": elapsed_f, "b": elapsed_b}
    elif mode == "train":
        t0 = timeit.default_timer()
        with nvtx.range("forward"):
            logits = model.forward(input).reshape(-1, 10000)
        with nvtx.range("loss"):
            loss = cross_entropy(logits, targets=target)
        elapsed_f = timeit.default_timer() - t0
        t1 = timeit.default_timer()
        with nvtx.range("backward"):
            loss.backward()
        elapsed_b = timeit.default_timer() - t1
        t2 = timeit.default_timer()
        with nvtx.range("optim_step"):
            optim.step()
        elapsed_o = timeit.default_timer() - t2
        optim.zero_grad()
        return {"f": elapsed_f, "b": elapsed_b, "o": elapsed_o}
        



def profile_model(
    config: str,
    mode: str,
    device: str,
    warmup: int,
    iters: int,
    batch: int,
    ctx_length: int
):
    config = SIZES[config]
    config.context_length = ctx_length
    model = create_model(config)
    model.to(device=device)
    optim = AdamW(params=model.parameters())
    samples = torch.randint(0, 10000, (batch, config.context_length), device=device)
    targets = torch.randint(0, 10000, (batch, config.context_length), device=device).reshape(-1)
    
    print("Start warmup")
    with nvtx.range("warmup"):
        for i in range(warmup):
            run_model_nvtx(mode=mode, model=model, optim=optim, input=samples, target=targets)   
         
    print(f"Start timing: {iters} iters")
    with nvtx.range("timing"):
        for i in range(iters):
            output = run_model_nvtx(mode=mode, model=model, optim=optim, input=samples, target=targets)
            
                
    output = {
        "size": config.model_type,
        "mode": mode,
        "warmup": warmup,
        "iters": iters,
    }
    
    del model, optim
    torch.cuda.empty_cache()
    return output
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, type=str)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iters", required=True, type=int)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--ctx_length", required=True, type=int)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--device", required=True, type=str)
    args = parser.parse_args()
    profile_model(
        config=args.config,
        mode=args.mode,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        batch=args.batch,
        ctx_length=args.ctx_length
    )

@app.function(
    image=build_image(),
    gpu="B200",
    timeout=2700,
    volumes={"/root/data": user_volume}
)
def profile_remote(mode, warmup, iters, batch, ctx_length, config, device, output, dir = "/root/data/nsys-output"):
    output_path = f"{dir}/{output}"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "nsys", "profile",
        *DEFAULT_NSYS_FLAGS,
        "-o", output_path,
        "--",
        "python", "-m", "cs336_systems.model_profile",
        "--config", config.model_type,
        "--mode", mode,
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--batch", str(batch),
        "--ctx_length", str(ctx_length),
        "--device", device,
    ], check=True)
    
    subprocess.run([
        "nsys", "stats",
        "--report", "nvtx_sum",
        "--report", "cuda_gpu_kern_sum",
        "--report", "nvtx_kern_sum",
        "--format", "csv",
        "--filter-nvtx", "timing", 
        "--force-overwrite=true",
        "-o", output_path,
        output_path + ".nsys-rep",
    ], check=True)
    
    
    
    user_volume.commit()
    

@app.function(
    image=build_image(),
    timeout=2700,
    gpu="B200",
    volumes={"/root/data": user_volume}
)
def check_nsys():
    result = subprocess.run(["nsys", "stats", "--help"], capture_output=True, text=True)
    print(result.stdout)

@app.local_entrypoint()
def run_check():
    check_nsys.remote()

@app.function(
    image=build_image(),
    timeout=2700,
    volumes={"/root/data": user_volume}
)
def run_sweep(spec: list[dict], output: str) -> list[dict]:
    rows = []
    handles = []
    for setup in spec:
        print(setup)
        handles.append(profile_remote.spawn(
            config=SIZES[setup["config"]], 
            mode=setup["mode"], 
            device=setup["device"], 
            warmup=setup["warmup"], 
            iters=setup["iters"], 
            ctx_length=setup["ctx_length"], 
            batch=setup["batch"],
            output=f"{setup['config']}_{setup['mode']}_{setup['ctx_length']}",
            dir=f"/root/data/nsys-output/{output}"
        ))
    for h in handles:
        h.get()
    print("All runs completed!")    
@app.local_entrypoint()
def sweep(warmup: int = 5, iters: int = 1, batch: int = 4, device: str = "cuda", output: str = "profile_results"):
    output_path = f"nsys-output/{output}"
    os.makedirs(output_path, exist_ok=True)
    spec = []
    for size, config in SIZES.items():
        for mode in ["train"]:
            for ctx_length in CTX_LENS:
                spec.append({"config": size, "mode": mode, "warmup": warmup, "iters": iters, "ctx_length": ctx_length,"batch": batch, "device": device})
    rows = run_sweep.remote(spec, output=output)
    print(f"Downloading results to nsys-output/{output_path}/")
    subprocess.run(["modal", "volume", "get", "--force", "basics-kesavanr", f"nsys-output/{output}", output_path + "/"], check=True)
    print("Completed download")
    # df = pd.DataFrame(rows)
    # df.to_csv(f"{output}.csv", index=False)


@app.local_entrypoint()
def modal_main(mode: str, warmup: int, iters: int, ctx_length: int, config: str, output: str, batch: int = 4, device:str="cuda"):
    cfg = load_config(config)
    profile_remote.remote(mode, warmup, iters, batch, ctx_length, cfg, device, output)