import torch
import pandas as pd
from cs336_systems.modal_utils import app, build_image

image = build_image()

with image.imports():
    import triton
    from cs336_systems.attention import *
    from cs336_basics.model import *

HEAD_DIMS = [16, 32, 64, 128]
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]




def benchmark_attention(dtype_name, d_head, seq_len, impl):
    if dtype_name == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    if d_head == 16:
        B_q = B_k = 128
    elif d_head == 128:
        B_q = B_k = 32
    else:
        B_q = B_k = 64

    row = {
        "dtype": dtype_name,
        "d_head": d_head,
        "seq_len": seq_len,
        "impl": impl,
        "fwd_ms": None,
        "bwd_ms": None,
        "e2e_ms": None,
    }

    try:
        q = torch.randn(1, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(1, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)
        v = torch.randn(1, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)
        do = torch.randn_like(q)
        

        if impl == "triton":
            def attn():
                return TritonFlashAttentionFunction.apply(q, k, v, True, B_q, B_k)
        elif impl == "torch":
            mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool))
            def attn():
                return scaled_dot_product_attention(q, k, v, mask)

        def fwd():
            attn()
            
        out = attn()

        def bwd():
            q.grad = None
            k.grad = None
            v.grad = None
            out.backward(do, retain_graph=True)

        def e2e():
            q.grad = None
            k.grad = None
            v.grad = None
            y = attn()
            y.backward(do)

        row["fwd_ms"] = triton.testing.do_bench(fwd, warmup=25, rep=100)
        row["bwd_ms"] = triton.testing.do_bench(bwd, warmup=25, rep=100)
        row["e2e_ms"] = triton.testing.do_bench(e2e, warmup=25, rep=100)
        
        if impl == "torch":
            del mask
        del q, k, v, do, out
    except torch.cuda.OutOfMemoryError:
        print(f"OOM dtype={dtype_name}, d_head={d_head}, seq_len={seq_len}, impl={impl}")
    torch.cuda.empty_cache()
    return row


@app.function(
    image=image,
    gpu="B200",
    timeout=1800,
)
def run_sweep():
    rows = []
    for dtype_name in ["bf16", "fp32"]:
        for d_head in HEAD_DIMS:
            for seq_len in SEQ_LENS:
                for impl in ["triton", "torch"]:
                    print(f"dtype={dtype_name}, d_head={d_head}, seq_len={seq_len}, impl={impl}")
                    row = benchmark_attention(dtype_name, d_head, seq_len, impl)
                    rows.append(row)
                    print(row)
    return rows


@app.local_entrypoint()
def main(output: str = "attention_times.csv"):
    rows = run_sweep.remote()
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(df.to_markdown(index=False))