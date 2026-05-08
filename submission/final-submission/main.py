import modal

app = modal.App("cs336-leaderboard-safe")

image = (
    modal.Image.debian_slim()
    .apt_install("wget", "gzip")
    .run_commands(
        "wget -nv https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
    )
    .apt_install("libcap2-bin", "libdw1", "cuda-nsight-systems-13-2")
    .pip_install("torch~=2.11.0", "triton", "numpy")
    .pip_install("cuda-tile[tileiras]")
    .add_local_dir("./cs336-basics", remote_path="/cs336-basics", copy=True)
    .run_commands("pip install -e /cs336-basics")
)


CTX = 32768
VOCAB = 151936
DMODEL = 4096
DFF = 11008
LAYERS = 34
HEADS = 32
BS = 2
REP_MS = 2000
WARMUP_MS = 500


with image.imports():
    import os
    import math
    import time
    from types import SimpleNamespace

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.distributed as dist
    import triton
    import triton.testing
    import cuda.tile as ct
    from cuda.tile import ByTarget

    from cs336_basics.model import BasicsTransformerLM, Linear, Embedding
    from cs336_basics.nn_utils import cross_entropy
    import cs336_basics.model as basics_model

    cache = {}
    FWD_TBQ, FWD_TBK = 256, 128
    FWD_NCTAS, FWD_OCC, FWD_OPT = 1, 2, 3
    DQ_TBQ, DQ_TBK = 128, 128
    DQ_NCTAS, DQ_OCC, DQ_OPT = 1, 1, 3
    DKV_TBQ, DKV_TBK = 64, 128
    DKV_NCTAS, DKV_OCC, DKV_OPT = 1, 2, 3
    D_TILE_S = 64

    @ct.kernel(occupancy=2)
    def fwd_kernel(q, k, v, out, lse,
                   TBQ: ct.Constant[int],
                   TBK: ct.Constant[int],
                   D: ct.Constant[int],
                   ROOT_D: ct.Constant[float],
                   is_causal: ct.Constant[bool]):
        qi = ct.bid(0); bi = ct.bid(1); hi = ct.bid(2)
        q_off = qi * TBQ + ct.arange(TBQ, dtype=np.int32)
        q_off = q_off[:, None]
        q_tile = ct.load(q, index=(bi, hi, qi, 0), shape=(1, 1, TBQ, D)).reshape((TBQ, D))
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
            kt = ct.load(k, index=(bi, hi, 0, kj), shape=(1, 1, D, TBK), order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
            s = ct.full((TBQ, TBK), 0.0, dtype=np.float32)
            s = ct.mma(q_tile, kt, s)
            new_m = max(m, ct.max(s, axis=1, keepdims=True) * ROOT_D)
            s = s * ROOT_D - new_m
            p = ct.exp2(s, flush_to_zero=True)
            li = ct.sum(p, axis=1, keepdims=True)
            a = ct.exp2(m - new_m, flush_to_zero=True)
            l = l * a + li
            o = o * a
            vt = ct.load(v, index=(bi, hi, kj, 0), shape=(1, 1, TBK, D), latency=4).reshape((TBK, D))
            p = p.astype(vt.dtype)
            o = ct.mma(p, vt, o)
            m = new_m
        for kj in range(n_unmasked, n_kv):
            kt = ct.load(k, index=(bi, hi, 0, kj), shape=(1, 1, D, TBK), order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
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
            vt = ct.load(v, index=(bi, hi, kj, 0), shape=(1, 1, TBK, D), latency=4).reshape((TBK, D))
            p = p.astype(vt.dtype)
            o = ct.mma(p, vt, o)
            m = new_m
        o = ct.truediv(o, l, flush_to_zero=True)
        o = o.reshape((1, 1, TBQ, D)).astype(out.dtype)
        ct.store(out, index=(bi, hi, qi, 0), tile=o)
        lse_v = m + ct.log2(l)
        lse_v = lse_v.reshape((1, 1, TBQ)).astype(np.float32)
        ct.store(lse, index=(bi, hi, qi), tile=lse_v)

    def cutile_fwd(q, k, v, is_causal):
        B, H, S, D = q.shape
        key = ("fwd", B, H, S, D, is_causal, q.dtype)
        if key not in cache:
            cfg = SimpleNamespace(TBQ=FWD_TBQ, TBK=FWD_TBK, nc=FWD_NCTAS, occ=FWD_OCC, opt=FWD_OPT)
            tuned = fwd_kernel.replace_hints(num_ctas=ByTarget(sm_100=cfg.nc), occupancy=ByTarget(sm_100=cfg.occ), opt_level=cfg.opt)
            cache[key] = (cfg, tuned)
        cfg, tuned = cache[key]
        out = torch.empty_like(q)
        lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
        rd = (1.0 / math.sqrt(D)) * (1.0 / math.log(2.0))
        grid = (math.ceil(S / cfg.TBQ), B, H)
        ct.launch(torch.cuda.current_stream(), grid, tuned, (q, k, v, out, lse, cfg.TBQ, cfg.TBK, D, rd, is_causal))
        return out, lse

    @ct.kernel(occupancy=2)
    def compute_D(o, do, D_out, TS: ct.Constant[int], D: ct.Constant[int]):
        si = ct.bid(0); bi = ct.bid(1); hi = ct.bid(2)
        o_t = ct.load(o, index=(bi, hi, si, 0), shape=(1, 1, TS, D)).reshape((TS, D))
        do_t = ct.load(do, index=(bi, hi, si, 0), shape=(1, 1, TS, D)).reshape((TS, D))
        prod = o_t.astype(np.float32) * do_t.astype(np.float32)
        dv = ct.sum(prod, axis=1, keepdims=True)
        dv = dv.reshape((1, 1, TS)).astype(np.float32)
        ct.store(D_out, index=(bi, hi, si), tile=dv)

    @ct.kernel(occupancy=2)
    def dq_kernel(q, k, v, do, lse, D_vec, dq,
                  TBQ: ct.Constant[int], TBK: ct.Constant[int], D: ct.Constant[int],
                  ROOT_D: ct.Constant[float], INV_SQRT_D: ct.Constant[float], is_causal: ct.Constant[bool]):
        qi = ct.bid(0); bi = ct.bid(1); hi = ct.bid(2)
        q_off = qi * TBQ + ct.arange(TBQ, dtype=np.int32)[:, None]
        q_tile = ct.load(q, index=(bi, hi, qi, 0), shape=(1, 1, TBQ, D)).reshape((TBQ, D))
        do_tile = ct.load(do, index=(bi, hi, qi, 0), shape=(1, 1, TBQ, D)).reshape((TBQ, D))
        lse_t = ct.load(lse, index=(bi, hi, qi), shape=(1, 1, TBQ)).reshape((TBQ, 1))
        D_t = ct.load(D_vec, index=(bi, hi, qi), shape=(1, 1, TBQ)).reshape((TBQ, 1))
        dq_acc = ct.full((TBQ, D), 0.0, dtype=np.float32)
        n_kv = ct.cdiv(k.shape[2], TBK)
        n_unmasked = n_kv
        if is_causal:
            end = (qi + 1) * TBQ
            n_kv = ct.cdiv(min(end, k.shape[2]), TBK)
            n_unmasked = (qi * TBQ) // TBK
        for kj in range(n_unmasked):
            kt = ct.load(k, index=(bi, hi, 0, kj), shape=(1, 1, D, TBK), order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
            vt = ct.load(v, index=(bi, hi, 0, kj), shape=(1, 1, D, TBK), order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
            s = ct.full((TBQ, TBK), 0.0, dtype=np.float32)
            s = ct.mma(q_tile, kt, s)
            s = s * ROOT_D
            p = ct.exp2(s - lse_t, flush_to_zero=True)
            dp = ct.full((TBQ, TBK), 0.0, dtype=np.float32)
            dp = ct.mma(do_tile, vt, dp)
            ds = p * (dp - D_t)
            k_n = ct.load(k, index=(bi, hi, kj, 0), shape=(1, 1, TBK, D), latency=2).reshape((TBK, D))
            ds_c = ds.astype(k_n.dtype)
            dq_acc = ct.mma(ds_c, k_n, dq_acc)
        for kj in range(n_unmasked, n_kv):
            kt = ct.load(k, index=(bi, hi, 0, kj), shape=(1, 1, D, TBK), order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
            vt = ct.load(v, index=(bi, hi, 0, kj), shape=(1, 1, D, TBK), order=(0, 1, 3, 2), latency=2).reshape((D, TBK))
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
            k_n = ct.load(k, index=(bi, hi, kj, 0), shape=(1, 1, TBK, D), latency=2).reshape((TBK, D))
            ds_c = ds.astype(k_n.dtype)
            dq_acc = ct.mma(ds_c, k_n, dq_acc)
        dq_acc = dq_acc * INV_SQRT_D
        dq_out = dq_acc.reshape((1, 1, TBQ, D)).astype(dq.dtype)
        ct.store(dq, index=(bi, hi, qi, 0), tile=dq_out)

    @ct.kernel(occupancy=2)
    def dkv_kernel(q, k, v, do, lse, D_vec, dk, dv,
                   TBQ: ct.Constant[int], TBK: ct.Constant[int], D: ct.Constant[int],
                   ROOT_D: ct.Constant[float], INV_SQRT_D: ct.Constant[float], is_causal: ct.Constant[bool]):
        kj = ct.bid(0); bi = ct.bid(1); hi = ct.bid(2)
        k_off = kj * TBK + ct.arange(TBK, dtype=np.int32)[None, :]
        k_tile = ct.load(k, index=(bi, hi, kj, 0), shape=(1, 1, TBK, D)).reshape((TBK, D))
        v_tile = ct.load(v, index=(bi, hi, kj, 0), shape=(1, 1, TBK, D)).reshape((TBK, D))
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
            q_tile = ct.load(q, index=(bi, hi, qi, 0), shape=(1, 1, TBQ, D), latency=2).reshape((TBQ, D))
            do_tile = ct.load(do, index=(bi, hi, qi, 0), shape=(1, 1, TBQ, D), latency=2).reshape((TBQ, D))
            lse_t = ct.load(lse, index=(bi, hi, qi), shape=(1, 1, TBQ)).reshape((TBQ, 1))
            D_t = ct.load(D_vec, index=(bi, hi, qi), shape=(1, 1, TBQ)).reshape((TBQ, 1))
            q_t = ct.load(q, index=(bi, hi, 0, qi), shape=(1, 1, D, TBQ), order=(0, 1, 3, 2), latency=2).reshape((D, TBQ))
            s = ct.full((TBK, TBQ), 0.0, dtype=np.float32)
            s = ct.mma(k_tile, q_t, s)
            s = s * ROOT_D
            p = ct.exp2(s - lse_t.reshape((1, TBQ)), flush_to_zero=True)
            keep = q_off.reshape((1, TBQ)) >= k_off.reshape((TBK, 1))
            p = ct.where(keep, p, 0.0)
            do_t = ct.load(do, index=(bi, hi, 0, qi), shape=(1, 1, D, TBQ), order=(0, 1, 3, 2), latency=2).reshape((D, TBQ))
            dpT = ct.full((TBK, TBQ), 0.0, dtype=np.float32)
            dpT = ct.mma(v_tile, do_t, dpT)
            dsT = p * (dpT - D_t.reshape((1, TBQ)))
            p_c = p.astype(do_tile.dtype)
            dv_acc = ct.mma(p_c, do_tile, dv_acc)
            dsT_c = dsT.astype(q_tile.dtype)
            dk_acc = ct.mma(dsT_c, q_tile, dk_acc)
        for qi in range(q_full_unmasked, n_q):
            q_tile = ct.load(q, index=(bi, hi, qi, 0), shape=(1, 1, TBQ, D), latency=2).reshape((TBQ, D))
            do_tile = ct.load(do, index=(bi, hi, qi, 0), shape=(1, 1, TBQ, D), latency=2).reshape((TBQ, D))
            lse_t = ct.load(lse, index=(bi, hi, qi), shape=(1, 1, TBQ)).reshape((TBQ, 1))
            D_t = ct.load(D_vec, index=(bi, hi, qi), shape=(1, 1, TBQ)).reshape((TBQ, 1))
            q_t = ct.load(q, index=(bi, hi, 0, qi), shape=(1, 1, D, TBQ), order=(0, 1, 3, 2), latency=2).reshape((D, TBQ))
            s = ct.full((TBK, TBQ), 0.0, dtype=np.float32)
            s = ct.mma(k_tile, q_t, s)
            s = s * ROOT_D
            p = ct.exp2(s - lse_t.reshape((1, TBQ)), flush_to_zero=True)
            do_t = ct.load(do, index=(bi, hi, 0, qi), shape=(1, 1, D, TBQ), order=(0, 1, 3, 2), latency=2).reshape((D, TBQ))
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

    def cutile_bwd(q, k, v, o, lse, do, is_causal):
        B, H, S, D = q.shape
        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        o = o.contiguous(); do = do.contiguous(); lse = lse.contiguous()
        D_vec = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
        ct.launch(torch.cuda.current_stream(), (math.ceil(S / D_TILE_S), B, H), compute_D,
                  (o, do, D_vec, D_TILE_S, D))
        dq = torch.empty_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v)
        rd = (1.0 / math.sqrt(D)) * (1.0 / math.log(2.0))
        isd = 1.0 / math.sqrt(D)
        key_dq = ("dq_static", q.dtype)
        if key_dq not in cache:
            cache[key_dq] = dq_kernel.replace_hints(num_ctas=ByTarget(sm_100=DQ_NCTAS), occupancy=ByTarget(sm_100=DQ_OCC), opt_level=DQ_OPT)
        key_dkv = ("dkv_static", q.dtype)
        if key_dkv not in cache:
            cache[key_dkv] = dkv_kernel.replace_hints(num_ctas=ByTarget(sm_100=DKV_NCTAS), occupancy=ByTarget(sm_100=DKV_OCC), opt_level=DKV_OPT)
        ct.launch(torch.cuda.current_stream(), (math.ceil(S / DQ_TBQ), B, H), cache[key_dq],
                  (q, k, v, do, lse, D_vec, dq, DQ_TBQ, DQ_TBK, D, rd, isd, is_causal))
        ct.launch(torch.cuda.current_stream(), (math.ceil(S / DKV_TBK), B, H), cache[key_dkv],
                  (q, k, v, do, lse, D_vec, dk, dv, DKV_TBQ, DKV_TBK, D, rd, isd, is_causal))
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

    class FSDPLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight_shard, bias, dtype, weight):
            if weight is None:
                weight = torch.empty((weight_shard.shape[0] * dist.get_world_size(), weight_shard.shape[1]), device=weight_shard.device, dtype=weight_shard.dtype)
                dist.all_gather_into_tensor(weight, weight_shard)
            if dtype is not None:
                if bias is not None:
                    output = F.linear(input.to(dtype), weight.to(dtype), bias.to(dtype))
                else:
                    output = F.linear(input.to(dtype), weight.to(dtype), None)
                output = output.to(input.dtype)
            else:
                output = F.linear(input, weight, bias)
            ctx.has_bias = bias is not None
            ctx.dtype = dtype
            if bias is not None:
                ctx.save_for_backward(input, weight_shard, bias)
            else:
                ctx.save_for_backward(input, weight_shard)
            return output
        @staticmethod
        def backward(ctx, grad_output):
            if ctx.has_bias:
                input, weight_shard, bias = ctx.saved_tensors
            else:
                input, weight_shard = ctx.saved_tensors
                bias = None
            world_size = dist.get_world_size()
            full_weight = torch.empty((weight_shard.shape[0] * dist.get_world_size(), weight_shard.shape[1]), device=weight_shard.device, dtype=weight_shard.dtype)
            dist.all_gather_into_tensor(full_weight, weight_shard)
            dtype = ctx.dtype
            if dtype is not None:
                go = grad_output.to(dtype); inp = input.to(dtype); w = full_weight.to(dtype)
            else:
                go = grad_output; inp = input; w = full_weight
            grad_input = go.matmul(w).to(input.dtype)
            grad_full_weight = go.reshape(-1, full_weight.shape[0]).t() @ inp.reshape(-1, full_weight.shape[1])
            grad_full_weight = grad_full_weight.to(weight_shard.dtype)
            grad_weight_shard = torch.empty_like(weight_shard)
            dist.reduce_scatter_tensor(grad_weight_shard, grad_full_weight)
            grad_weight_shard /= world_size
            if bias is not None:
                grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            else:
                grad_bias = None
            return grad_input, grad_weight_shard, grad_bias, None, None

    class FSDPLinear(nn.Module):
        def __init__(self, linear, compute_dtype=None):
            super().__init__()
            rank_ = dist.get_rank()
            world_size_ = dist.get_world_size()
            self.dtype = compute_dtype
            self.full_shape = linear.weight.shape
            shard_size = linear.weight.shape[0] // world_size_
            self.weight_shard = nn.Parameter(linear.weight[rank_*shard_size:(rank_+1)*shard_size].clone())
            if hasattr(linear, "bias") and linear.bias is not None:
                self.bias = nn.Parameter(linear.bias.clone())
            else:
                self.bias = None
            self.in_features = linear.weight.shape[1]
            self.out_features = linear.weight.shape[0]
            self.weight = None
            self.handle = None
        def prefetch(self):
            if self.handle is not None: return
            if self.weight is not None: return
            self.weight = torch.empty(self.full_shape, device=self.weight_shard.device, dtype=self.weight_shard.dtype)
            self.handle = dist.all_gather_into_tensor(self.weight, self.weight_shard, async_op=True)
        def forward(self, input):
            if self.handle is not None:
                self.handle.wait()
                self.handle = None
            weight = self.weight
            self.weight = None
            return FSDPLinearFunction.apply(input, self.weight_shard, self.bias, self.dtype, weight)

    class FSDPEmbeddingFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight_shard, dtype, weight):
            if weight is None:
                weight = torch.empty((weight_shard.shape[0] * dist.get_world_size(), weight_shard.shape[1]), device=weight_shard.device, dtype=weight_shard.dtype)
                dist.all_gather_into_tensor(weight, weight_shard)
            if dtype is not None:
                output = F.embedding(input, weight.to(dtype)).to(weight_shard.dtype)
            else:
                output = F.embedding(input, weight)
            ctx.dtype = dtype
            ctx.save_for_backward(input, weight_shard)
            return output
        @staticmethod
        def backward(ctx, grad_output):
            input, weight_shard = ctx.saved_tensors
            world_size = dist.get_world_size()
            full_rows = weight_shard.shape[0] * world_size
            grad_full_weight = torch.zeros((full_rows, weight_shard.shape[1]), device=weight_shard.device, dtype=weight_shard.dtype)
            grad_full_weight.index_add_(0, input.reshape(-1), grad_output.reshape(-1, weight_shard.shape[1]).to(weight_shard.dtype))
            grad_weight_shard = torch.empty_like(weight_shard)
            dist.reduce_scatter_tensor(grad_weight_shard, grad_full_weight)
            grad_weight_shard /= world_size
            return None, grad_weight_shard, None, None

    class FSDPEmbedding(nn.Module):
        def __init__(self, embedding, compute_dtype=None):
            super().__init__()
            rank_ = dist.get_rank()
            world_size_ = dist.get_world_size()
            self.dtype = compute_dtype
            self.full_shape = embedding.weight.shape
            shard_size = embedding.weight.shape[0] // world_size_
            self.weight_shard = nn.Parameter(embedding.weight[rank_*shard_size:(rank_+1)*shard_size].clone())
            self.weight = None
            self.handle = None
        def prefetch(self):
            if self.handle is not None: return
            if self.weight is not None: return
            self.weight = torch.empty(self.full_shape, device=self.weight_shard.device, dtype=self.weight_shard.dtype)
            self.handle = dist.all_gather_into_tensor(self.weight, self.weight_shard, async_op=True)
        def forward(self, input):
            if self.handle is not None:
                self.handle.wait()
                self.handle = None
            weight = self.weight
            self.weight = None
            return FSDPEmbeddingFunction.apply(input, self.weight_shard, self.dtype, weight)

    class FSDP(nn.Module):
        def __init__(self, module, compute_dtype=None):
            super().__init__()
            self.module = module
            self.compute_dtype = compute_dtype
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
            self.layers = []
            self.wrap(self.module)
            for i, layer in enumerate(self.layers):
                layer.register_forward_hook(self.hook(i))
        def wrap(self, module):
            for name, child in list(module.named_children()):
                if isinstance(child, (nn.Linear, Linear)):
                    module._modules[name] = FSDPLinear(child, compute_dtype=self.compute_dtype)
                    self.layers.append(module._modules[name])
                elif isinstance(child, (nn.Embedding, Embedding)):
                    module._modules[name] = FSDPEmbedding(child, compute_dtype=self.compute_dtype)
                    self.layers.append(module._modules[name])
                else:
                    self.wrap(child)
        def hook(self, i):
            def hook(module, input, output):
                j = i + 2
                if j < len(self.layers):
                    self.layers[j].prefetch()
            return hook
        def forward(self, *args, **kwargs):
            if len(self.layers) > 0: self.layers[0].prefetch()
            if len(self.layers) > 1: self.layers[1].prefetch()
            return self.module(*args, **kwargs)
        def grad_sync(self):
            for name, param in self.module.named_parameters():
                if param.grad is None: continue
                if "weight_shard" in name: continue
                dist.all_reduce(param.grad.data)
                param.grad.data /= dist.get_world_size()


def fsdp_worker(rank, world_size, q):
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
    model = FSDP(model, compute_dtype=None)
    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
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


@app.function(image=image, gpu="B200:2", timeout=1800)
def bench_b200x2():
    import torch.multiprocessing as mp
    world_size = 2
    ctx_ = mp.get_context("spawn")
    q = ctx_.Queue()
    procs = []
    for r in range(world_size):
        p = ctx_.Process(target=fsdp_worker, args=(r, world_size, q))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    return q.get()


@app.local_entrypoint()
def main():
    result = bench_b200x2.remote()
    print("got back:", result)
