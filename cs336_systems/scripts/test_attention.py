from cs336_systems.modal_utils import app, build_image, secrets


def run_test(impl, batch, heads, seq, d, is_causal, dtype_str, atol, rtol, time_it, rep_ms, warmup_ms, seed, fwd_only):
    import torch
    import torch.nn.functional as F
    import triton.testing
    from cs336_systems.attention import TritonFlashAttentionFunction, TorchFlashAttentionFunction
    from cs336_systems.cutile_attention import CuTileFlashAttentionFunction

    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

    print(f"impl={impl} B={batch} H={heads} T={seq} D={d} causal={is_causal} dtype={dtype}", flush=True)

    needs_grad = not fwd_only
    q = torch.randn(batch, heads, seq, d, device=device, dtype=dtype, requires_grad=needs_grad)
    k = torch.randn(batch, heads, seq, d, device=device, dtype=dtype, requires_grad=needs_grad)
    v = torch.randn(batch, heads, seq, d, device=device, dtype=dtype, requires_grad=needs_grad)
    dout = torch.randn(batch, heads, seq, d, device=device, dtype=dtype)

    def call_custom(qq, kk, vv):
        if impl == "cutile":
            return CuTileFlashAttentionFunction.apply(qq, kk, vv, is_causal)
        elif impl == "triton":
            qf = qq.reshape(batch * heads, seq, d)
            kf = kk.reshape(batch * heads, seq, d)
            vf = vv.reshape(batch * heads, seq, d)
            return TritonFlashAttentionFunction.apply(qf, kf, vf, is_causal).reshape(batch, heads, seq, d)
        elif impl == "torch":
            qf = qq.reshape(batch * heads, seq, d)
            kf = kk.reshape(batch * heads, seq, d)
            vf = vv.reshape(batch * heads, seq, d)
            return TorchFlashAttentionFunction.apply(qf, kf, vf, is_causal).reshape(batch, heads, seq, d)
        else:
            raise ValueError(impl)

    def call_ref(qq, kk, vv):
        return F.scaled_dot_product_attention(qq, kk, vv, is_causal=is_causal)

    out_custom = call_custom(q, k, v)
    if not fwd_only:
        gq_c, gk_c, gv_c = torch.autograd.grad(out_custom, (q, k, v), dout)

    qr = q.detach().clone().requires_grad_(needs_grad)
    kr = k.detach().clone().requires_grad_(needs_grad)
    vr = v.detach().clone().requires_grad_(needs_grad)
    out_ref = call_ref(qr, kr, vr)
    if not fwd_only:
        gq_r, gk_r, gv_r = torch.autograd.grad(out_ref, (qr, kr, vr), dout)

    def diff(name, a, b):
        a32, b32 = a.float(), b.float()
        max_abs = (a32 - b32).abs().max().item()
        max_rel = ((a32 - b32).abs() / (b32.abs() + 1e-6)).max().item()
        ok = torch.allclose(a32, b32, atol=atol, rtol=rtol)
        print(name, "max_abs", max_abs, "max_rel", max_rel, "ok" if ok else "fail", flush=True)
        return ok

    print(f"correctness atol={atol} rtol={rtol}", flush=True)
    ok_o = diff("out", out_custom, out_ref)
    if fwd_only:
        all_ok = ok_o
    else:
        ok_q = diff("dq", gq_c, gq_r)
        ok_k = diff("dk", gk_c, gk_r)
        ok_v = diff("dv", gv_c, gv_r)
        all_ok = ok_o and ok_q and ok_k and ok_v
    if all_ok:
        print("correct", flush=True)
    else:
        print("wrong", flush=True)

    timing = {}
    if time_it:
        def fwd():
            return call_custom(q, k, v)
        def fwd_bwd():
            o = call_custom(q, k, v)
            torch.autograd.grad(o, (q, k, v), dout)
        def fwd_ref():
            return call_ref(qr, kr, vr)
        def fwd_bwd_ref():
            o = call_ref(qr, kr, vr)
            torch.autograd.grad(o, (qr, kr, vr), dout)

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

    return {"correctness": all_ok, **timing}


@app.function(image=build_image(), gpu="B200:1", secrets=secrets(), timeout=900)
def test_attention_remote(impl, batch, heads, seq, d, is_causal, dtype_str, atol, rtol, time_it, rep_ms, warmup_ms, seed, fwd_only):
    return run_test(impl, batch, heads, seq, d, is_causal, dtype_str, atol, rtol, time_it, rep_ms, warmup_ms, seed, fwd_only)


@app.local_entrypoint()
def main(
    impl: str = "cutile",
    batch: int = 2,
    heads: int = 32,
    seq: int = 32768,
    d: int = 128,
    causal: bool = True,
    dtype: str = "bf16",
    atol: float = 1e-2,
    rtol: float = 1e-2,
    time_it: bool = True,
    rep_ms: int = 1000,
    warmup_ms: int = 200,
    seed: int = 0,
    big: bool = False,
    fwd_only: bool = False,
):
    if big:
        batch, heads, seq, d = 2, 32, 32768, 128
    result = test_attention_remote.remote(
        impl=impl,
        batch=batch,
        heads=heads,
        seq=seq,
        d=d,
        is_causal=causal,
        dtype_str=dtype,
        atol=atol,
        rtol=rtol,
        time_it=time_it,
        rep_ms=rep_ms,
        warmup_ms=warmup_ms,
        seed=seed,
        fwd_only=fwd_only,
    )
    print("got back:", result)