from cs336_systems.modal_utils import app, build_image, secrets

atol = 1e-2
rtol = 1e-2


def run_test(bt, d, v, dtype_str, time_it, rep_ms, warmup_ms, seed, fwd_only):
    import torch
    import torch.nn.functional as F
    import triton.testing
    from cs336_systems.fused_ce import FusedCEFunction

    def diff(name, a, b):
        a_float, b_float = a.float(), b.float()
        max_abs = (a_float - b_float).abs().max().item()
        max_rel = ((a_float - b_float).abs() / (b_float.abs() + 1e-6)).max().item()
        ok = torch.allclose(a_float, b_float, atol=atol, rtol=rtol)
        print(name, "max_abs", max_abs, "max_rel", max_rel, "ok" if ok else "fail", flush=True)
        return ok

    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

    print(f"BT={bt} D={d} V={v} dtype={dtype}", flush=True)

    needs_grad = not fwd_only
    x = torch.randn(bt, d, device=device, dtype=dtype, requires_grad=needs_grad)
    w = torch.randn(v, d, device=device, dtype=dtype, requires_grad=needs_grad) * 0.02
    targets = torch.randint(0, v, (bt,), device=device)

    def call_custom(xx, ww, tt):
        return FusedCEFunction.apply(xx, ww, tt)

    def reference(xx, ww, tt):
        logits = xx @ ww.t()
        return F.cross_entropy(logits.float(), tt, reduction="sum")

    out_custom = call_custom(x, w, targets)
    if not fwd_only:
        gx_c, gw_c = torch.autograd.grad(out_custom, (x, w))

    xr = x.detach().clone().requires_grad_(needs_grad)
    wr = w.detach().clone().requires_grad_(needs_grad)
    out_ref = reference(xr, wr, targets)
    if not fwd_only:
        gx_r, gw_r = torch.autograd.grad(out_ref, (xr, wr))

    

    print(f"correctness atol={atol} rtol={rtol}", flush=True)
    ok_l = diff("loss", out_custom, out_ref)
    if fwd_only:
        all_ok = ok_l
    else:
        ok_x = diff("dx", gx_c, gx_r)
        ok_w = diff("dw", gw_c, gw_r)
        all_ok = ok_l and ok_x and ok_w
    if all_ok:
        print("correct", flush=True)
    else:
        print("incorrect", flush=True)

    timing = {}
    if time_it:
        def fwd():
            return call_custom(x, w, targets)
        def fwd_bwd():
            o = call_custom(x, w, targets)
            torch.autograd.grad(o, (x, w))
        def fwd_ref():
            return reference(xr, wr, targets)
        def fwd_bwd_ref():
            o = reference(xr, wr, targets)
            torch.autograd.grad(o, (xr, wr))

        print(f"timing rep={rep_ms}ms warmup={warmup_ms}ms", flush=True)
        ms_f = triton.testing.do_bench(fwd, rep=rep_ms, warmup=warmup_ms)
        ms_f_ref = triton.testing.do_bench(fwd_ref, rep=rep_ms, warmup=warmup_ms)
        print(f"custom fwd {ms_f} ms", flush=True)
        print(f"ref fwd {ms_f_ref} ms", flush=True)
        timing = {"custom_fwd_ms": ms_f, "ref_fwd_ms": ms_f_ref}
        if not fwd_only:
            ms_fb = triton.testing.do_bench(fwd_bwd, rep=rep_ms, warmup=warmup_ms)
            ms_fb_ref = triton.testing.do_bench(fwd_bwd_ref, rep=rep_ms, warmup=warmup_ms)
            print(f"custom fwd+bwd {ms_fb} ms", flush=True)
            print(f"ref fwd+bwd {ms_fb_ref} ms", flush=True)
            timing["custom_fwd_bwd_ms"] = ms_fb
            timing["ref_fwd_bwd_ms"] = ms_fb_ref

    from cs336_systems.fused_ce import fused_ce_dx_kernel, fused_ce_dw_kernel
    for name, k in [("dx", fused_ce_dx_kernel), ("dw", fused_ce_dw_kernel)]:
        bc = getattr(k, "best_config", None)
        if bc is None and hasattr(k, "cache"):
            bc = next(iter(k.cache.values()), None)
        print(f"best {name} config: {bc}", flush=True)

    return {"correctness": all_ok, **timing}


@app.function(image=build_image(), gpu="B200:1", secrets=secrets(), timeout=900)
def test_remote(bt, d, v, dtype_str, time_it, rep_ms, warmup_ms, seed, fwd_only):
    return run_test(bt, d, v, dtype_str, time_it, rep_ms, warmup_ms, seed, fwd_only)


@app.local_entrypoint()
def main(
    bt: int = 256,
    d: int = 128,
    v: int = 4096,
    dtype: str = "bf16",
    time_it: bool = True,
    rep_ms: int = 1000,
    warmup_ms: int = 200,
    seed: int = 0,
    big: bool = False,
    fwd_only: bool = False,
):
    if big:
        bt, d, v = 32768, 4096, 151936
    result = test_remote.remote(
        bt=bt, d=d, v=v,
        dtype_str=dtype,
        time_it=time_it, rep_ms=rep_ms, warmup_ms=warmup_ms,
        seed=seed, fwd_only=fwd_only,
    )
    print("got back:", result)
