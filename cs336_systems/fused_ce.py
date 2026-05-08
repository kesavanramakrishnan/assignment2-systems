import torch
import triton # pyright: ignore[reportMissingImports]
import triton.language as tl # pyright: ignore[reportMissingImports]


@triton.jit
def fused_ce_fwd_kernel(
    X_ptr, W_ptr, T_ptr,
    LOSS_ptr, LSE_ptr, TGT_ptr,
    stride_xt, stride_xd,
    stride_wv, stride_wd,
    stride_tt,
    stride_lset,
    stride_tgtt,
    BT, V, D,
    TILE_BX: tl.constexpr,
    TILE_V: tl.constexpr,
    TILE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * TILE_BX
    
    
    
    target_ptr = tl.make_block_ptr(
        T_ptr + row_start * stride_tt,
        shape=(TILE_BX,),
        strides=(stride_tt,),
        offsets=(0,),
        block_shape=(TILE_BX,),
        order=(0,)
    )
    
    target = tl.load(target_ptr).to(tl.int32)

    row_max = tl.full((TILE_BX,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((TILE_BX,), dtype=tl.float32)
    target_logit = tl.zeros((TILE_BX,), dtype=tl.float32)

    for v_start in range(0, V, TILE_V):
        x_block = tl.make_block_ptr(
            X_ptr + row_start * stride_xt,
            shape=(BT, D),
            strides=(stride_xt, stride_xd),
            offsets=(0, 0),
            block_shape=(TILE_BX, TILE_D),
            order=(1, 0)
        )
        w_offsets = tl.make_block_ptr(
            W_ptr + v_start * stride_wv,
            shape=(V, D),
            strides=(stride_wv, stride_wd),
            offsets=(0, 0),
            block_shape=(TILE_V, TILE_D),
            order=(1, 0)
        )
        w_block = w_offsets
        
        accum = tl.zeros((TILE_BX, TILE_V), dtype=tl.float32)

        for d_start in range(0, D, TILE_D):
            x_tile = tl.load(x_block)
            w_tile = tl.load(w_block, boundary_check=(0,), padding_option="zero")
            logits = tl.dot(x_tile, tl.trans(w_tile))
            accum += logits
            x_block = tl.advance(x_block, (0, TILE_D))
            w_block = tl.advance(w_block, (0, TILE_D))
        v_idx = v_start + tl.arange(0, TILE_V)[None, :]
        accum = tl.where(v_idx < V, accum, float("-inf"))
            
        
        new_row_max = tl.maximum(row_max, tl.max(accum, axis=1))
        
        l = l * tl.exp(row_max - new_row_max) + tl.sum(tl.exp(accum - new_row_max[:, None]), axis=1)
        row_max = new_row_max
        
        in_tile = (target >= v_start) & (target < v_start + TILE_V)
        col = target - v_start
        col_idx = tl.arange(0, TILE_V)[None, :]                                                                                                                             
        target_col = (target - v_start)[:, None]
        match = col_idx == target_col                                                                                                                                       
        hit = tl.sum(tl.where(match, accum, 0.0), axis=1)      
        target_logit = tl.where(in_tile, hit, target_logit)  
    lse = row_max + tl.log(l)                                                                                                                          
    loss_per_row = lse - target_logit                      
    row_idx = row_start + tl.arange(0, TILE_BX)            
    tl.store(LSE_ptr + row_idx * stride_lset, lse)                                                                                                                      
    tl.store(TGT_ptr + row_idx * stride_tgtt, target_logit)                                                                                                             
    tl.atomic_add(LOSS_ptr, tl.sum(loss_per_row))                                     
       
        




def fused_ce_fwd(x, w, targets):
    
    BT, D = x.shape
    V = w.shape[0]
    loss = torch.zeros((), device=x.device, dtype=torch.float32)
    lse = torch.empty((BT,), device=x.device, dtype=torch.float32)
    tgt = torch.empty((BT,), device=x.device, dtype=torch.float32)
    TILE_BX = 32
    TILE_V = 256
    TILE_D = 128
    grid = (triton.cdiv(BT, TILE_BX),)
    fused_ce_fwd_kernel[grid](
        x, w, targets,
        loss, lse, tgt,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        targets.stride(0),
        lse.stride(0),
        tgt.stride(0),
        BT, V, D,
        TILE_BX=TILE_BX,
        TILE_V=TILE_V,
        TILE_D=TILE_D,
        num_warps=4,
        num_stages=3,
    )
    return loss, lse, tgt


@triton.jit
def fused_ce_dx_kernel(
    X_ptr, W_ptr, T_ptr,
    LSE_ptr, GLOSS_ptr,
    DX_ptr,
    stride_xt, stride_xd,
    stride_wv, stride_wd,
    stride_tt,
    stride_lset,
    stride_dxt, stride_dxd,
    BT, V, D,
    TILE_BX: tl.constexpr,
    TILE_V: tl.constexpr,
    TILE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * TILE_BX

    target_ptr = tl.make_block_ptr(
        T_ptr + row_start * stride_tt,
        shape=(TILE_BX,),
        strides=(stride_tt,),
        offsets=(0,),
        block_shape=(TILE_BX,),
        order=(0,)
    )
    target = tl.load(target_ptr).to(tl.int32)

    lse_ptr = tl.make_block_ptr(
        LSE_ptr + row_start * stride_lset,
        shape=(TILE_BX,),
        strides=(stride_lset,),
        offsets=(0,),
        block_shape=(TILE_BX,),
        order=(0,)
    )
    lse = tl.load(lse_ptr)

    grad_loss = tl.load(GLOSS_ptr).to(tl.float32)

    for v_start in range(0, V, TILE_V):
        x_block = tl.make_block_ptr(
            X_ptr + row_start * stride_xt,
            shape=(BT, D),
            strides=(stride_xt, stride_xd),
            offsets=(0, 0),
            block_shape=(TILE_BX, TILE_D),
            order=(1, 0)
        )
        w_block = tl.make_block_ptr(
            W_ptr + v_start * stride_wv,
            shape=(V, D),
            strides=(stride_wv, stride_wd),
            offsets=(0, 0),
            block_shape=(TILE_V, TILE_D),
            order=(1, 0)
        )

        accum = tl.zeros((TILE_BX, TILE_V), dtype=tl.float32)
        for d_start in range(0, D, TILE_D):
            x_tile = tl.load(x_block)
            w_tile = tl.load(w_block, boundary_check=(0,), padding_option="zero")
            accum += tl.dot(x_tile, tl.trans(w_tile))
            x_block = tl.advance(x_block, (0, TILE_D))
            w_block = tl.advance(w_block, (0, TILE_D))

        v_idx = v_start + tl.arange(0, TILE_V)[None, :]
        valid = v_idx < V

        p = tl.exp(accum - lse[:, None])
        col_idx = tl.arange(0, TILE_V)[None, :]
        target_col = (target - v_start)[:, None]
        match = col_idx == target_col
        p = p - tl.where(match, 1.0, 0.0)
        p = p * grad_loss
        p = tl.where(valid, p, 0.0)
        p_cast = p.to(X_ptr.dtype.element_ty)

        w_block = tl.make_block_ptr(
            W_ptr + v_start * stride_wv,
            shape=(V, D),
            strides=(stride_wv, stride_wd),
            offsets=(0, 0),
            block_shape=(TILE_V, TILE_D),
            order=(1, 0)
        )
        dx_block = tl.make_block_ptr(
            DX_ptr + row_start * stride_dxt,
            shape=(BT, D),
            strides=(stride_dxt, stride_dxd),
            offsets=(0, 0),
            block_shape=(TILE_BX, TILE_D),
            order=(1, 0)
        )
        for d_start in range(0, D, TILE_D):
            w_tile = tl.load(w_block, boundary_check=(0,), padding_option="zero")
            dx_contrib = tl.dot(p_cast, w_tile)
            cur_dx = tl.load(dx_block)
            tl.store(dx_block, cur_dx + dx_contrib.to(cur_dx.dtype))
            w_block = tl.advance(w_block, (0, TILE_D))
            dx_block = tl.advance(dx_block, (0, TILE_D))


@triton.jit
def fused_ce_dw_kernel(
    X_ptr, W_ptr, T_ptr,
    LSE_ptr, GLOSS_ptr,
    DW_ptr,
    stride_xt, stride_xd,
    stride_wv, stride_wd,
    stride_tt,
    stride_lset,
    stride_dwv, stride_dwd,
    BT, V, D,
    TILE_BX: tl.constexpr,
    TILE_V: tl.constexpr,
    TILE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    v_start = pid * TILE_V

    grad_loss = tl.load(GLOSS_ptr).to(tl.float32)

    for bt_start in range(0, BT, TILE_BX):
        target_ptr = tl.make_block_ptr(
            T_ptr + bt_start * stride_tt,
            shape=(TILE_BX,),
            strides=(stride_tt,),
            offsets=(0,),
            block_shape=(TILE_BX,),
            order=(0,)
        )
        target = tl.load(target_ptr).to(tl.int32)

        lse_ptr = tl.make_block_ptr(
            LSE_ptr + bt_start * stride_lset,
            shape=(TILE_BX,),
            strides=(stride_lset,),
            offsets=(0,),
            block_shape=(TILE_BX,),
            order=(0,)
        )
        lse = tl.load(lse_ptr)

        x_block = tl.make_block_ptr(
            X_ptr + bt_start * stride_xt,
            shape=(BT, D),
            strides=(stride_xt, stride_xd),
            offsets=(0, 0),
            block_shape=(TILE_BX, TILE_D),
            order=(1, 0)
        )
        w_block = tl.make_block_ptr(
            W_ptr + v_start * stride_wv,
            shape=(V, D),
            strides=(stride_wv, stride_wd),
            offsets=(0, 0),
            block_shape=(TILE_V, TILE_D),
            order=(1, 0)
        )

        accum = tl.zeros((TILE_BX, TILE_V), dtype=tl.float32)
        for d_start in range(0, D, TILE_D):
            x_tile = tl.load(x_block)
            w_tile = tl.load(w_block, boundary_check=(0,), padding_option="zero")
            accum += tl.dot(x_tile, tl.trans(w_tile))
            x_block = tl.advance(x_block, (0, TILE_D))
            w_block = tl.advance(w_block, (0, TILE_D))

        v_idx = v_start + tl.arange(0, TILE_V)[None, :]
        valid = v_idx < V

        p = tl.exp(accum - lse[:, None])
        col_idx = tl.arange(0, TILE_V)[None, :]
        target_col = (target - v_start)[:, None]
        match = col_idx == target_col
        p = p - tl.where(match, 1.0, 0.0)
        p = p * grad_loss
        p = tl.where(valid, p, 0.0)
        p_cast = p.to(X_ptr.dtype.element_ty)

        x_block = tl.make_block_ptr(
            X_ptr + bt_start * stride_xt,
            shape=(BT, D),
            strides=(stride_xt, stride_xd),
            offsets=(0, 0),
            block_shape=(TILE_BX, TILE_D),
            order=(1, 0)
        )
        dw_block = tl.make_block_ptr(
            DW_ptr + v_start * stride_dwv,
            shape=(V, D),
            strides=(stride_dwv, stride_dwd),
            offsets=(0, 0),
            block_shape=(TILE_V, TILE_D),
            order=(1, 0)
        )
        for d_start in range(0, D, TILE_D):
            x_tile = tl.load(x_block)
            dw_contrib = tl.dot(tl.trans(p_cast), x_tile)
            cur_dw = tl.load(dw_block, boundary_check=(0,), padding_option="zero")
            tl.store(dw_block, cur_dw + dw_contrib.to(cur_dw.dtype), boundary_check=(0,))
            x_block = tl.advance(x_block, (0, TILE_D))
            dw_block = tl.advance(dw_block, (0, TILE_D))


def fused_ce_bwd(x, w, targets, lse, target_logit, grad_loss):
    BT, D = x.shape
    V = w.shape[0]

    dx = torch.zeros_like(x)

    TILE_BX = 32
    TILE_V = 256
    TILE_D = 128

    pad_v = (-V) % TILE_V
    V_padded = V + pad_v
    dw = torch.zeros((V_padded, D), device=w.device, dtype=w.dtype)

    grad_loss_f32 = grad_loss.to(torch.float32) if grad_loss.dtype != torch.float32 else grad_loss

    fused_ce_dx_kernel[(triton.cdiv(BT, TILE_BX),)](
        x, w, targets,
        lse, grad_loss_f32,
        dx,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        targets.stride(0),
        lse.stride(0),
        dx.stride(0), dx.stride(1),
        BT, V, D,
        TILE_BX=TILE_BX,
        TILE_V=TILE_V,
        TILE_D=TILE_D,
        num_warps=4,
        num_stages=3,
    )

    fused_ce_dw_kernel[(triton.cdiv(V, TILE_V),)](
        x, w, targets,
        lse, grad_loss_f32,
        dw,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        targets.stride(0),
        lse.stride(0),
        dw.stride(0), dw.stride(1),
        BT, V, D,
        TILE_BX=TILE_BX,
        TILE_V=TILE_V,
        TILE_D=TILE_D,
        num_warps=4,
        num_stages=2,
    )

    return dx, dw[:V]


class FusedCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, targets):
        loss, lse, tgt_logit = fused_ce_fwd(x, w, targets)
        ctx.save_for_backward(x, w, targets, lse, tgt_logit)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        x, w, targets, lse, tgt_logit = ctx.saved_tensors
        dx, dw = fused_ce_bwd(x, w, targets, lse, tgt_logit, grad_loss)
        return dx, dw, None
