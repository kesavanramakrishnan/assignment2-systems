import torch
import triton  # pyright: ignore[reportMissingImports]
import triton.language as tl  # pyright: ignore[reportMissingImports]




def flash_attention_fwd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False
) -> torch.Tensor:
    B_q = 16
    B_k = 16
    num_q_tiles = q.shape[1] // B_q
    num_kv_tiles = k.shape[1] // B_k
    root_d = q.shape[-1] ** 0.5
    output = torch.zeros_like(q)
    lse = torch.full((q.shape[0], q.shape[1]), float("-inf"), device=q.device)
    
    for i in range(num_q_tiles):
        q_tile = q[:, i*B_q:(i+1)*B_q, :]
        row_wise_max = torch.full((q_tile.shape[0], q_tile.shape[1]), float("-inf"), device=q.device)
        l = torch.zeros(q_tile.shape[0], q_tile.shape[1], device=q.device)
        output_tile = torch.zeros_like(q_tile)
        for j in range(num_kv_tiles):
            k_tile = k[:, j*B_k:(j+1)*B_k, :]
            v_tile = v[:, j*B_k:(j+1)*B_k, :]
            pre_attn_scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) / (root_d)
            if is_causal:
                q_indices = torch.arange(q_tile.shape[1], device=q.device) + i * B_q
                k_indices = torch.arange(k_tile.shape[1], device=q.device) + j * B_k
                mask = q_indices[:, None] >= k_indices[None, :]
                pre_attn_scores = torch.where(mask, pre_attn_scores, float("-1e6"))
            cur_row_wise_max = torch.maximum(row_wise_max, pre_attn_scores.max(dim=-1).values)
            scores = torch.exp(pre_attn_scores - cur_row_wise_max.unsqueeze(-1))
            l = torch.exp(row_wise_max - cur_row_wise_max) * l + scores.sum(dim=-1)
            output_tile = torch.exp(row_wise_max - cur_row_wise_max).unsqueeze(-1) * output_tile + torch.matmul(scores, v_tile)
            row_wise_max = cur_row_wise_max
        output[:, i*B_q:(i+1)*B_q, :] = output_tile / l.unsqueeze(-1)
        lse[:, i*B_q:(i+1)*B_q] = row_wise_max + torch.log(l)
    return output, lse

@torch.compile
def flash_attention_bwd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    is_causal: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root_d = q.shape[-1] ** 0.5
    s = torch.matmul(q, k.transpose(-2, -1)) / root_d
    if is_causal:
        q_indices = torch.arange(q.shape[1], device=q.device)[:, None]
        k_indices = torch.arange(k.shape[1], device=k.device)[None, :]
        mask = q_indices >= k_indices
        s = torch.where(mask, s, float("-1e6"))
    scores = torch.exp(s.float() - lse.unsqueeze(-1))
    scores = scores.to(q.dtype)

    dv = scores.transpose(-2, -1).matmul(grad_output)
    dscores = grad_output.matmul(v.transpose(-2, -1))
    d = torch.sum(output * grad_output, dim=-1, keepdim=True)
    ds = scores * (dscores - d)
    dq = ds.matmul(k) / root_d
    dk = ds.transpose(-2, -1).matmul(q) / root_d
    return dq, dk, dv
    
    

class TorchFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        ctx.is_causal = is_causal
        output, lse = flash_attention_fwd_torch(q, k, v, is_causal)
        ctx.save_for_backward(q, k, v, lse, output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, lse, output = ctx.saved_tensors
        dq, dk, dv = flash_attention_bwd_torch(q, k, v, output, lse, grad_output, ctx.is_causal)
        return dq, dk, dv, None 



@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
 # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    
    Q_tile = tl.load(Q_block_ptr)
    num_kv_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    row_max = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    output_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(num_kv_tiles):
        K_tile = tl.load(K_block_ptr)
        V_tile = tl.load(V_block_ptr)
        attn_scores = tl.dot(Q_tile, tl.trans(K_tile))
        attn_scores = attn_scores * scale
        if IS_CAUSAL:
            q_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_indices = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            mask = q_indices[:, None] >= k_indices[None, :]
            attn_scores = tl.where(mask, attn_scores, float("-1e6"))
            
        new_max = tl.maximum(row_max, tl.max(attn_scores, axis=-1))
        scores = tl.exp(attn_scores - new_max[:, None])
        l = tl.exp(row_max - new_max) * l + tl.sum(scores, axis=-1)
        output_tile = tl.exp(row_max - new_max)[:, None] * output_tile
        output_tile = tl.dot(scores.to(V_tile.dtype), V_tile, acc=output_tile)
        row_max = new_max
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    output_tile = output_tile / l[:, None]
    lse = row_max + tl.log(l)
    tl.store(O_block_ptr, output_tile.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, lse)     
        



class TritonFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, B_q=128, B_k=128):
        output = torch.empty_like(q)
        lse = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
        scale = 1 / (q.shape[-1] ** 0.5)
        grid = (triton.cdiv(q.shape[1], B_q), q.shape[0])
        flash_fwd_kernel[grid](
            q, k, v,
            output, lse,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            lse.stride(0), lse.stride(1),
            q.shape[1], k.shape[1],
            scale,
            D=q.shape[-1],
            Q_TILE_SIZE=B_q,
            K_TILE_SIZE=B_k,
            IS_CAUSAL=is_causal
        )
        ctx.save_for_backward(q, k, v, lse, output)
        ctx.is_causal = is_causal
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, lse, output = ctx.saved_tensors
        dq, dk, dv = flash_attention_bwd_torch(q, k, v, output, lse, grad_output, ctx.is_causal)
        return dq, dk, dv, None, None, None 
