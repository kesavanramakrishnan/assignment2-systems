import torch
import math
import numpy as np
import cuda.tile as ct
from cuda.tile import ByTarget
from cuda.tile.tune import exhaustive_search
from types import SimpleNamespace


cache = {}

FWD_NO_AUTOTUNE = True
FWD_TBQ = 256
FWD_TBK = 128
FWD_NCTAS = 1
FWD_OCC = 2
FWD_OPT = 3

BWD_NO_AUTOTUNE = True
DQ_TBQ = 128
DQ_TBK = 128
DQ_NCTAS = 1
DQ_OCC = 1
DQ_OPT = 3
DKV_TBQ = 64
DKV_TBK = 128
DKV_NCTAS = 1
DKV_OCC = 2
DKV_OPT = 3

D_TILE_S = 64


@ct.kernel(occupancy=2)
def fwd_kernel(q, k, v, out, lse,
               TBQ: ct.Constant[int],
               TBK: ct.Constant[int],
               D: ct.Constant[int],
               ROOT_D: ct.Constant[float],
               is_causal: ct.Constant[bool]):
    qi = ct.bid(0)
    bi = ct.bid(1)
    hi = ct.bid(2)

    q_off = qi * TBQ + ct.arange(TBQ, dtype=np.int32)
    q_off = q_off[:, None]

    q_tile = ct.load(q, index=(bi, hi, qi, 0),
                     shape=(1, 1, TBQ, D)).reshape((TBQ, D))

    m = ct.full((TBQ, 1), -np.inf, dtype=np.float32)
    l = ct.full((TBQ, 1), 0.0, dtype=np.float32)
    o = ct.full((TBQ, D), 0.0, dtype=np.float32)

    n_kv = ct.cdiv(k.shape[2], TBK)
    n_unmasked = n_kv
    if is_causal:
        end = (qi + 1) * TBQ
        n_kv = ct.cdiv(min(end, k.shape[2]), TBK)
        n_unmasked = (qi * TBQ) // TBK

    for kj in range(n_unmasked):
        kt = ct.load(k, index=(bi, hi, 0, kj),
                     shape=(1, 1, D, TBK),
                     order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
        s = ct.full((TBQ, TBK), 0.0, dtype=np.float32)
        s = ct.mma(q_tile, kt, s)

        new_m = max(m, ct.max(s, axis=1, keepdims=True) * ROOT_D)
        s = s * ROOT_D - new_m

        p = ct.exp2(s, flush_to_zero=True)
        li = ct.sum(p, axis=1, keepdims=True)
        a = ct.exp2(m - new_m, flush_to_zero=True)
        l = l * a + li
        o = o * a

        vt = ct.load(v, index=(bi, hi, kj, 0),
                     shape=(1, 1, TBK, D), latency=4).reshape((TBK, D))
        p = p.astype(vt.dtype)
        o = ct.mma(p, vt, o)
        m = new_m

    for kj in range(n_unmasked, n_kv):
        kt = ct.load(k, index=(bi, hi, 0, kj),
                     shape=(1, 1, D, TBK),
                     order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
        s = ct.full((TBQ, TBK), 0.0, dtype=np.float32)
        s = ct.mma(q_tile, kt, s)

        k_off = kj * TBK + ct.arange(TBK, dtype=np.int32)[None, :]
        keep = ct.full((TBQ, TBK), True, dtype=np.bool_)
        keep = keep & (q_off >= k_off)
        bias = ct.where(keep, 0.0, -np.inf)
        s += bias

        new_m = max(m, ct.max(s, axis=1, keepdims=True) * ROOT_D)
        s = s * ROOT_D - new_m

        p = ct.exp2(s, flush_to_zero=True)
        li = ct.sum(p, axis=1, keepdims=True)
        a = ct.exp2(m - new_m, flush_to_zero=True)
        l = l * a + li
        o = o * a

        vt = ct.load(v, index=(bi, hi, kj, 0),
                     shape=(1, 1, TBK, D), latency=4).reshape((TBK, D))
        p = p.astype(vt.dtype)
        o = ct.mma(p, vt, o)
        m = new_m

    o = ct.truediv(o, l, flush_to_zero=True)
    o = o.reshape((1, 1, TBQ, D)).astype(out.dtype)
    ct.store(out, index=(bi, hi, qi, 0), tile=o)
    lse_v = m + ct.log2(l)
    lse_v = lse_v.reshape((1, 1, TBQ)).astype(np.float32)
    ct.store(lse, index=(bi, hi, qi), tile=lse_v)


def tune_fwd(q, k, v, is_causal):
    B, H, S, D = q.shape
    out = torch.empty_like(q)
    lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    rd = (1.0 / math.sqrt(D)) * (1.0 / math.log(2.0))
    space = [
        SimpleNamespace(TBQ=256, TBK=128, nc=1, occ=2, opt=3),
        SimpleNamespace(TBQ=128, TBK=128, nc=2, occ=2, opt=3),
        SimpleNamespace(TBQ=128, TBK=128, nc=1, occ=2, opt=3),
        SimpleNamespace(TBQ=128, TBK=128, nc=1, occ=1, opt=3),
        SimpleNamespace(TBQ=128, TBK=64,  nc=1, occ=2, opt=3),
        SimpleNamespace(TBQ=64,  TBK=128, nc=1, occ=2, opt=3),
        SimpleNamespace(TBQ=64,  TBK=64,  nc=1, occ=4, opt=3),
        SimpleNamespace(TBQ=256, TBK=64,  nc=2, occ=2, opt=3),
    ]
    with ct.compiler_timeout(10):
        result = exhaustive_search(
            space, torch.cuda.current_stream(),
            grid_fn=lambda c: (math.ceil(S / c.TBQ), B, H),
            kernel=fwd_kernel,
            args_fn=lambda c: (q, k, v, out, lse, c.TBQ, c.TBK, D, rd, is_causal),
            hints_fn=lambda c: {"num_ctas": ByTarget(sm_100=c.nc),
                                "occupancy": ByTarget(sm_100=c.occ),
                                "opt_level": c.opt},
        )
    return result.best.config


def cutile_fwd(q, k, v, is_causal):
    B, H, S, D = q.shape
    key = ("fwd", B, H, S, D, is_causal, q.dtype)
    if key not in cache:
        if FWD_NO_AUTOTUNE:
            cfg = SimpleNamespace(TBQ=FWD_TBQ, TBK=FWD_TBK,
                                  nc=FWD_NCTAS, occ=FWD_OCC, opt=FWD_OPT)
        else:
            cfg = tune_fwd(q, k, v, is_causal)
        tuned = fwd_kernel.replace_hints(num_ctas=ByTarget(sm_100=cfg.nc),
                                         occupancy=ByTarget(sm_100=cfg.occ),
                                         opt_level=cfg.opt)
        cache[key] = (cfg, tuned)
        print("fwd cfg", cfg, flush=True)
    cfg, tuned = cache[key]

    out = torch.empty_like(q)
    lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    rd = (1.0 / math.sqrt(D)) * (1.0 / math.log(2.0))
    grid = (math.ceil(S / cfg.TBQ), B, H)
    ct.launch(torch.cuda.current_stream(), grid, tuned,
              (q, k, v, out, lse, cfg.TBQ, cfg.TBK, D, rd, is_causal))
    return out, lse


@ct.kernel(occupancy=2)
def compute_D(o, do, D_out,
              TS: ct.Constant[int],
              D: ct.Constant[int]):
    si = ct.bid(0)
    bi = ct.bid(1)
    hi = ct.bid(2)

    o_t = ct.load(o, index=(bi, hi, si, 0),
                  shape=(1, 1, TS, D)).reshape((TS, D))
    do_t = ct.load(do, index=(bi, hi, si, 0),
                   shape=(1, 1, TS, D)).reshape((TS, D))

    prod = o_t.astype(np.float32) * do_t.astype(np.float32)
    dv = ct.sum(prod, axis=1, keepdims=True)
    dv = dv.reshape((1, 1, TS)).astype(np.float32)
    ct.store(D_out, index=(bi, hi, si), tile=dv)


@ct.kernel(occupancy=2)
def dq_kernel(q, k, v, do, lse, D_vec, dq,
              TBQ: ct.Constant[int],
              TBK: ct.Constant[int],
              D: ct.Constant[int],
              ROOT_D: ct.Constant[float],
              INV_SQRT_D: ct.Constant[float],
              is_causal: ct.Constant[bool]):
    qi = ct.bid(0)
    bi = ct.bid(1)
    hi = ct.bid(2)

    q_off = qi * TBQ + ct.arange(TBQ, dtype=np.int32)[:, None]

    q_tile = ct.load(q, index=(bi, hi, qi, 0),
                     shape=(1, 1, TBQ, D)).reshape((TBQ, D))
    do_tile = ct.load(do, index=(bi, hi, qi, 0),
                      shape=(1, 1, TBQ, D)).reshape((TBQ, D))
    lse_t = ct.load(lse, index=(bi, hi, qi),
                    shape=(1, 1, TBQ)).reshape((TBQ, 1))
    D_t = ct.load(D_vec, index=(bi, hi, qi),
                  shape=(1, 1, TBQ)).reshape((TBQ, 1))

    dq_acc = ct.full((TBQ, D), 0.0, dtype=np.float32)

    n_kv = ct.cdiv(k.shape[2], TBK)
    n_unmasked = n_kv
    if is_causal:
        end = (qi + 1) * TBQ
        n_kv = ct.cdiv(min(end, k.shape[2]), TBK)
        n_unmasked = (qi * TBQ) // TBK

    for kj in range(n_unmasked):
        kt = ct.load(k, index=(bi, hi, 0, kj),
                     shape=(1, 1, D, TBK),
                     order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
        vt = ct.load(v, index=(bi, hi, 0, kj),
                     shape=(1, 1, D, TBK),
                     order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
        s = ct.full((TBQ, TBK), 0.0, dtype=np.float32)
        s = ct.mma(q_tile, kt, s)

        s = s * ROOT_D
        p = ct.exp2(s - lse_t, flush_to_zero=True)

        dp = ct.full((TBQ, TBK), 0.0, dtype=np.float32)
        dp = ct.mma(do_tile, vt, dp)
        ds = p * (dp - D_t)

        k_n = ct.load(k, index=(bi, hi, kj, 0),
                      shape=(1, 1, TBK, D), latency=2).reshape((TBK, D))
        ds_c = ds.astype(k_n.dtype)
        dq_acc = ct.mma(ds_c, k_n, dq_acc)

    for kj in range(n_unmasked, n_kv):
        kt = ct.load(k, index=(bi, hi, 0, kj),
                     shape=(1, 1, D, TBK),
                     order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
        vt = ct.load(v, index=(bi, hi, 0, kj),
                     shape=(1, 1, D, TBK),
                     order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
        s = ct.full((TBQ, TBK), 0.0, dtype=np.float32)
        s = ct.mma(q_tile, kt, s)

        s = s * ROOT_D
        p = ct.exp2(s - lse_t, flush_to_zero=True)

        k_off = kj * TBK + ct.arange(TBK, dtype=np.int32)[None, :]
        keep = q_off >= k_off
        p = ct.where(keep, p, 0.0)

        dp = ct.full((TBQ, TBK), 0.0, dtype=np.float32)
        dp = ct.mma(do_tile, vt, dp)
        ds = p * (dp - D_t)

        k_n = ct.load(k, index=(bi, hi, kj, 0),
                      shape=(1, 1, TBK, D), latency=2).reshape((TBK, D))
        ds_c = ds.astype(k_n.dtype)
        dq_acc = ct.mma(ds_c, k_n, dq_acc)

    dq_acc = dq_acc * INV_SQRT_D
    dq_out = dq_acc.reshape((1, 1, TBQ, D)).astype(dq.dtype)
    ct.store(dq, index=(bi, hi, qi, 0), tile=dq_out)


@ct.kernel(occupancy=2)
def dkv_kernel(q, k, v, do, lse, D_vec, dk, dv,
               TBQ: ct.Constant[int],
               TBK: ct.Constant[int],
               D: ct.Constant[int],
               ROOT_D: ct.Constant[float],
               INV_SQRT_D: ct.Constant[float],
               is_causal: ct.Constant[bool]):
    kj = ct.bid(0)
    bi = ct.bid(1)
    hi = ct.bid(2)

    k_off = kj * TBK + ct.arange(TBK, dtype=np.int32)[None, :]

    k_tile = ct.load(k, index=(bi, hi, kj, 0),
                     shape=(1, 1, TBK, D)).reshape((TBK, D))
    v_tile = ct.load(v, index=(bi, hi, kj, 0),
                     shape=(1, 1, TBK, D)).reshape((TBK, D))

    dk_acc = ct.full((TBK, D), 0.0, dtype=np.float32)
    dv_acc = ct.full((TBK, D), 0.0, dtype=np.float32)

    n_q = ct.cdiv(q.shape[2], TBQ)
    q_start = 0
    q_full_unmasked = 0
    if is_causal:
        q_start = (kj * TBK) // TBQ
        q_full_unmasked = ct.cdiv((kj + 1) * TBK, TBQ)

    for qi in range(q_start, q_full_unmasked):
        q_off = qi * TBQ + ct.arange(TBQ, dtype=np.int32)[:, None]

        q_tile = ct.load(q, index=(bi, hi, qi, 0),
                         shape=(1, 1, TBQ, D), latency=2).reshape((TBQ, D))
        do_tile = ct.load(do, index=(bi, hi, qi, 0),
                          shape=(1, 1, TBQ, D), latency=2).reshape((TBQ, D))
        lse_t = ct.load(lse, index=(bi, hi, qi),
                        shape=(1, 1, TBQ)).reshape((TBQ, 1))
        D_t = ct.load(D_vec, index=(bi, hi, qi),
                      shape=(1, 1, TBQ)).reshape((TBQ, 1))
        q_t = ct.load(q, index=(bi, hi, 0, qi),
                      shape=(1, 1, D, TBQ),
                      order=(0, 1, 3, 2), latency=2).reshape((D, TBQ))

        s = ct.full((TBK, TBQ), 0.0, dtype=np.float32)
        s = ct.mma(k_tile, q_t, s)

        s = s * ROOT_D
        p = ct.exp2(s - lse_t.reshape((1, TBQ)), flush_to_zero=True)

        keep = q_off.reshape((1, TBQ)) >= k_off.reshape((TBK, 1))
        p = ct.where(keep, p, 0.0)

        do_t = ct.load(do, index=(bi, hi, 0, qi),
                       shape=(1, 1, D, TBQ),
                       order=(0, 1, 3, 2), latency=2).reshape((D, TBQ))
        dpT = ct.full((TBK, TBQ), 0.0, dtype=np.float32)
        dpT = ct.mma(v_tile, do_t, dpT)
        dsT = p * (dpT - D_t.reshape((1, TBQ)))

        p_c = p.astype(do_tile.dtype)
        dv_acc = ct.mma(p_c, do_tile, dv_acc)
        dsT_c = dsT.astype(q_tile.dtype)
        dk_acc = ct.mma(dsT_c, q_tile, dk_acc)

    for qi in range(q_full_unmasked, n_q):
        q_tile = ct.load(q, index=(bi, hi, qi, 0),
                         shape=(1, 1, TBQ, D), latency=2).reshape((TBQ, D))
        do_tile = ct.load(do, index=(bi, hi, qi, 0),
                          shape=(1, 1, TBQ, D), latency=2).reshape((TBQ, D))
        lse_t = ct.load(lse, index=(bi, hi, qi),
                        shape=(1, 1, TBQ)).reshape((TBQ, 1))
        D_t = ct.load(D_vec, index=(bi, hi, qi),
                      shape=(1, 1, TBQ)).reshape((TBQ, 1))
        q_t = ct.load(q, index=(bi, hi, 0, qi),
                      shape=(1, 1, D, TBQ),
                      order=(0, 1, 3, 2), latency=2).reshape((D, TBQ))

        s = ct.full((TBK, TBQ), 0.0, dtype=np.float32)
        s = ct.mma(k_tile, q_t, s)

        s = s * ROOT_D
        p = ct.exp2(s - lse_t.reshape((1, TBQ)), flush_to_zero=True)

        do_t = ct.load(do, index=(bi, hi, 0, qi),
                       shape=(1, 1, D, TBQ),
                       order=(0, 1, 3, 2), latency=2).reshape((D, TBQ))
        dpT = ct.full((TBK, TBQ), 0.0, dtype=np.float32)
        dpT = ct.mma(v_tile, do_t, dpT)
        dsT = p * (dpT - D_t.reshape((1, TBQ)))

        p_c = p.astype(do_tile.dtype)
        dv_acc = ct.mma(p_c, do_tile, dv_acc)
        dsT_c = dsT.astype(q_tile.dtype)
        dk_acc = ct.mma(dsT_c, q_tile, dk_acc)

    dk_acc = dk_acc * INV_SQRT_D
    dk_out = dk_acc.reshape((1, 1, TBK, D)).astype(dk.dtype)
    ct.store(dk, index=(bi, hi, kj, 0), tile=dk_out)
    dv_out = dv_acc.reshape((1, 1, TBK, D)).astype(dv.dtype)
    ct.store(dv, index=(bi, hi, kj, 0), tile=dv_out)


def tune_dq(q, k, v, do, lse, D_vec, is_causal):
    B, H, S, D = q.shape
    dq = torch.empty_like(q)
    rd = (1.0 / math.sqrt(D)) * (1.0 / math.log(2.0))
    isd = 1.0 / math.sqrt(D)
    space = [
        SimpleNamespace(TBQ=256, TBK=128, nc=1, occ=2, opt=3),
        SimpleNamespace(TBQ=128, TBK=128, nc=1, occ=2, opt=3),
        SimpleNamespace(TBQ=128, TBK=128, nc=1, occ=1, opt=3),
        SimpleNamespace(TBQ=128, TBK=64,  nc=1, occ=2, opt=3),
        SimpleNamespace(TBQ=64,  TBK=128, nc=1, occ=2, opt=3),
        SimpleNamespace(TBQ=64,  TBK=64,  nc=1, occ=4, opt=3),
    ]
    with ct.compiler_timeout(10):
        result = exhaustive_search(
            space, torch.cuda.current_stream(),
            grid_fn=lambda c: (math.ceil(S / c.TBQ), B, H),
            kernel=dq_kernel,
            args_fn=lambda c: (q, k, v, do, lse, D_vec, dq, c.TBQ, c.TBK, D, rd, isd, is_causal),
            hints_fn=lambda c: {"num_ctas": ByTarget(sm_100=c.nc),
                                "occupancy": ByTarget(sm_100=c.occ),
                                "opt_level": c.opt},
        )
    return result.best.config
def tune_dkv(q, k, v, do, lse, D_vec, is_causal):
    B, H, S, D = q.shape
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    rd = (1.0 / math.sqrt(D)) * (1.0 / math.log(2.0))
    isd = 1.0 / math.sqrt(D)
    space = [
        SimpleNamespace(TBQ=128, TBK=128, nc=1, occ=2, opt=3),
        SimpleNamespace(TBQ=128, TBK=128, nc=1, occ=1, opt=3),
        SimpleNamespace(TBQ=64,  TBK=128, nc=1, occ=2, opt=3),
        SimpleNamespace(TBQ=128, TBK=64,  nc=1, occ=2, opt=3),
        SimpleNamespace(TBQ=64,  TBK=64,  nc=1, occ=4, opt=3),
    ]
    with ct.compiler_timeout(10):
        result = exhaustive_search(
            space, torch.cuda.current_stream(),
            grid_fn=lambda c: (math.ceil(S / c.TBK), B, H),
            kernel=dkv_kernel,
            args_fn=lambda c: (q, k, v, do, lse, D_vec, dk, dv, c.TBQ, c.TBK, D, rd, isd, is_causal),
            hints_fn=lambda c: {"num_ctas": ByTarget(sm_100=c.nc),
                                "occupancy": ByTarget(sm_100=c.occ),
                                "opt_level": c.opt},
        )
    return result.best.config


def cutile_bwd(q, k, v, o, lse, do, is_causal):
    B, H, S, D = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = o.contiguous()
    do = do.contiguous()
    lse = lse.contiguous()

    D_vec = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    ct.launch(torch.cuda.current_stream(),
              (math.ceil(S / D_TILE_S), B, H),
              compute_D,
              (o, do, D_vec, D_TILE_S, D))

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    rd = (1.0 / math.sqrt(D)) * (1.0 / math.log(2.0))
    isd = 1.0 / math.sqrt(D)

    if BWD_NO_AUTOTUNE:
        key_dq = ("dq_static", q.dtype)
        if key_dq not in cache:
            cache[key_dq] = dq_kernel.replace_hints(num_ctas=ByTarget(sm_100=DQ_NCTAS),
                                                    occupancy=ByTarget(sm_100=DQ_OCC),
                                                    opt_level=DQ_OPT)
        key_dkv = ("dkv_static", q.dtype)
        if key_dkv not in cache:
            cache[key_dkv] = dkv_kernel.replace_hints(num_ctas=ByTarget(sm_100=DKV_NCTAS),
                                                      occupancy=ByTarget(sm_100=DKV_OCC),
                                                      opt_level=DKV_OPT)
        ct.launch(torch.cuda.current_stream(),
                  (math.ceil(S / DQ_TBQ), B, H),
                  cache[key_dq],
                  (q, k, v, do, lse, D_vec, dq, DQ_TBQ, DQ_TBK, D, rd, isd, is_causal))
        ct.launch(torch.cuda.current_stream(),
                  (math.ceil(S / DKV_TBK), B, H),
                  cache[key_dkv],
                  (q, k, v, do, lse, D_vec, dk, dv, DKV_TBQ, DKV_TBK, D, rd, isd, is_causal))
        return dq, dk, dv

    key_dq = ("dq", B, H, S, D, is_causal, q.dtype)
    if key_dq not in cache:
        cfg = tune_dq(q, k, v, do, lse, D_vec, is_causal)
        tuned = dq_kernel.replace_hints(num_ctas=ByTarget(sm_100=cfg.nc),
                                        occupancy=ByTarget(sm_100=cfg.occ),
                                        opt_level=cfg.opt)
        cache[key_dq] = (cfg, tuned)
        print("dq cfg", cfg, flush=True)
    cfg_q, tuned_q = cache[key_dq]

    key_dkv = ("dkv", B, H, S, D, is_causal, q.dtype)
    if key_dkv not in cache:
        cfg = tune_dkv(q, k, v, do, lse, D_vec, is_causal)
        tuned = dkv_kernel.replace_hints(num_ctas=ByTarget(sm_100=cfg.nc),
                                         occupancy=ByTarget(sm_100=cfg.occ),
                                         opt_level=cfg.opt)
        cache[key_dkv] = (cfg, tuned)
        print("dkv cfg", cfg, flush=True)
    cfg_kv, tuned_kv = cache[key_dkv]

    ct.launch(torch.cuda.current_stream(),
              (math.ceil(S / cfg_q.TBQ), B, H),
              tuned_q,
              (q, k, v, do, lse, D_vec, dq, cfg_q.TBQ, cfg_q.TBK, D, rd, isd, is_causal))
    ct.launch(torch.cuda.current_stream(),
              (math.ceil(S / cfg_kv.TBK), B, H),
              tuned_kv,
              (q, k, v, do, lse, D_vec, dk, dv, cfg_kv.TBQ, cfg_kv.TBK, D, rd, isd, is_causal))
    return dq, dk, dv


class CuTileFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        o, lse = cutile_fwd(q, k, v, is_causal)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = cutile_bwd(q, k, v, o, lse, do, ctx.is_causal)
        return dq, dk, dv, None
