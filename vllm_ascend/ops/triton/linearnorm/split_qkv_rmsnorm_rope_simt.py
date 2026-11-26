import os
import weakref

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import extract_slice, get_vectorcore_num, insert_slice


@triton.jit
def precompute_rope_cos_sin_kernel(
    positions_gm_ptr,
    cos_sin_cache_gm_ptr,
    out_cos_sin_gm_ptr,
    batch_size,
    ROPE_DIM: tl.constexpr,
    num_vectorcore: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    batch_per_prog = tl.cdiv(batch_size, num_vectorcore)
    start = pid * batch_per_prog
    end = tl.minimum(start + batch_per_prog, batch_size)

    sin_cos_range = tl.arange(0, ROPE_DIM)

    for off in range(start, end, BLOCK_SIZE):
        batch_off = off + tl.arange(0, BLOCK_SIZE)
        mask = batch_off < end

        pos = tl.load(positions_gm_ptr + batch_off, mask=mask)

        offset = pos[:, None] * ROPE_DIM + sin_cos_range[None, :]
        sin_cos_val = tl.load(cos_sin_cache_gm_ptr + offset).to(tl.float32)

        out_offset = batch_off[:, None] * ROPE_DIM + sin_cos_range[None, :]
        out_mask = (batch_off[:, None] < end) & (sin_cos_range[None, :] < ROPE_DIM)
        tl.store(out_cos_sin_gm_ptr + out_offset, sin_cos_val, mask=out_mask)


@triton.jit
def split_qkv_rmsnorm_rope_simt_kernel(
    input_gm_ptr,
    q_gm_ptr,
    k_gm_ptr,
    v_gm_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    cos_sin_precomputed_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    num_vectorcore: tl.constexpr,
    batch_size_per_iter_per_vec: tl.constexpr,
    v_batch_size_per_iter_per_vec: tl.constexpr,
    qk_head_nums_per_iter_per_vec: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    qk_head_num_sum: tl.constexpr,
):
    pid = tl.program_id(0)

    batch_per_prog = tl.cdiv(batch_size, num_vectorcore)
    start = pid * batch_per_prog
    end = tl.minimum(start + batch_per_prog, batch_size)

    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)

    output_q_nblk_idx = tl.arange(0, q_hidden_size)
    output_q_nmask = output_q_nblk_idx < q_hidden_size
    output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
    output_kv_nmask = output_kv_nblk_idx < kv_hidden_size

    for iter in tl.range(tl.cdiv(end - start, batch_size_per_iter_per_vec)):
        base_batch = start + iter * batch_size_per_iter_per_vec
        batch_indices = base_batch + tl.arange(0, batch_size_per_iter_per_vec)
        mmask = batch_indices < end

        qk_cols = tl.arange(0, q_hidden_size + kv_hidden_size)
        mask = mmask[:, None] & (qk_cols[None, :] < total_hidden_size)

        idx = batch_indices[:, None] * total_hidden_size + qk_cols[None, :]
        values_tmp1 = (
            tl.load(input_gm_ptr + idx, mask=mask).reshape(qk_head_nums_per_iter_per_vec, HEAD_DIM).to(tl.float32)
        )

        if BIAS:
            q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
            k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)

        cos_sin_offset = base_batch * ROPE_DIM + tl.arange(0, batch_size_per_iter_per_vec * ROPE_DIM)
        cos_sin_value = tl.load(
            cos_sin_precomputed_ptr + cos_sin_offset, mask=cos_sin_offset < (end * ROPE_DIM)
        ).reshape(batch_size_per_iter_per_vec, 1, ROPE_DIM)

        cos = extract_slice(
            cos_sin_value, offsets=(0, 0, 0), sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM), strides=(1, 1, 1)
        )
        sin = extract_slice(
            cos_sin_value,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        normalized_values = values_tmp1
        normalized_values = normalized_values * normalized_values
        normalized_values = tl.sum(normalized_values, axis=1) / HEAD_DIM
        normalized_values = 1 / tl.sqrt(normalized_values + eps).reshape(qk_head_nums_per_iter_per_vec, 1)
        normalized_values = values_tmp1 * normalized_values

        normalized_values_tmp = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )
        if BIAS:
            normalized_values_tmp = normalized_values_tmp * q_weight_values + q_bias_values
        else:
            normalized_values_tmp = normalized_values_tmp * q_weight_values

        values_tmp = tl.zeros((batch_size_per_iter_per_vec, q_head_num, ROPE_DIM), dtype=tl.float32)
        x1 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        q_output_idx = output_q_nblk_idx[None, :] + batch_indices[:, None] * q_hidden_size
        out_mask = mmask[:, None] & output_q_nmask[None, :]
        if IS_PARTIAL_ROPE:
            normalized_values_tmp = insert_slice(
                normalized_values_tmp,
                values_tmp,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                q_gm_ptr + q_output_idx,
                normalized_values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                mask=out_mask,
            )
        else:
            tl.store(
                q_gm_ptr + q_output_idx, values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size), mask=out_mask
            )

        normalized_values_tmp1 = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, q_head_num, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )
        if BIAS:
            normalized_values_tmp1 = normalized_values_tmp1 * k_weight_values + k_bias_values
        else:
            normalized_values_tmp1 = normalized_values_tmp1 * k_weight_values

        values_tmp2 = tl.zeros((batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM), dtype=tl.float32)
        x1 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        kv_output_idx = output_kv_nblk_idx[None, :] + batch_indices[:, None] * kv_hidden_size
        out_mask = mmask[:, None] & output_kv_nmask[None, :]
        if IS_PARTIAL_ROPE:
            normalized_values_tmp1 = insert_slice(
                normalized_values_tmp1,
                values_tmp2,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                k_gm_ptr + kv_output_idx,
                normalized_values_tmp1.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=out_mask,
            )
        else:
            tl.store(
                k_gm_ptr + kv_output_idx,
                values_tmp2.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=out_mask,
            )

    for iter in tl.range(tl.cdiv(end - start, v_batch_size_per_iter_per_vec)):
        base_batch = start + iter * v_batch_size_per_iter_per_vec
        batch_indices = base_batch + tl.arange(0, v_batch_size_per_iter_per_vec)
        mmask = batch_indices < end

        v_cols = tl.arange(q_hidden_size + kv_hidden_size, total_hidden_size)
        nmask = v_cols < total_hidden_size
        mask = mmask[:, None] & nmask[None, :]

        idx = batch_indices[:, None] * total_hidden_size + v_cols[None, :]
        values = tl.load(input_gm_ptr + idx, mask=mask)

        out_nblk_idx = tl.arange(0, kv_hidden_size)
        out_nmask = out_nblk_idx < kv_hidden_size
        out_idx = batch_indices[:, None] * kv_hidden_size + out_nblk_idx[None, :]
        out_mask = mmask[:, None] & out_nmask[None, :]
        tl.store(v_gm_ptr + out_idx, values, mask=out_mask)


# 每步 cos/sin 缓存默认关闭：forward context 身份和形状无法区分 draft/verify，
# 开启 MTP/eagle 时可能复用其他 positions 的 cos/sin，图模式下也不能保证 replay 时重算。
_ENABLE_ROPE_COS_SIN_STEP_CACHE = os.getenv("IQUEST_ROPE_COS_SIN_CACHE", "0") == "1"

_ROPE_COS_SIN_CACHE: dict = {"ctx_ref": None, "tensor": None}


def _rope_cos_sin_for_step(positions, cos_sin_cache, batch_size, rope_dim, num_vectorcore, device):
    """默认每次调用均按当前 positions 重算，避免跨 forward 复用旧值。

    实验性缓存仅供无投机解码的 eager A/B 使用；其键只有 forward context
    身份和形状，不能证明 positions 相同，禁止用于 MTP/eagle 或图模式。
    """
    if not _ENABLE_ROPE_COS_SIN_STEP_CACHE:
        ctx = None
    else:
        try:
            from vllm.forward_context import get_forward_context

            ctx = get_forward_context()
        except Exception:
            ctx = None

    cache = _ROPE_COS_SIN_CACHE
    if ctx is not None:
        ref, tensor = cache["ctx_ref"], cache["tensor"]
        if (
            ref is not None
            and ref() is ctx
            and tensor is not None
            and tensor.shape[0] == batch_size
            and tensor.shape[1] == rope_dim
        ):
            return tensor

    out = torch.empty(batch_size, rope_dim, dtype=torch.float32, device=device)
    grid = (num_vectorcore, 1, 1)
    precompute_rope_cos_sin_kernel[grid](
        positions,
        cos_sin_cache,
        out,
        batch_size,
        rope_dim,
        num_vectorcore,
        force_simt_only=True,
    )
    if ctx is not None:
        cache["ctx_ref"] = weakref.ref(ctx)
        cache["tensor"] = out
    return out


def split_qkv_rmsnorm_rope_simt_impl(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_vectorcore = get_vectorcore_num()
    rope_dim = cos_sin_cache.shape[-1]
    batch_size = input.shape[0]
    BIAS = q_bias is not None
    IS_PARTIAL_ROPE = rope_dim != head_dim
    total_hidden_size = q_hidden_size + kv_hidden_size * 2

    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    q_head_num = q_hidden_size // head_dim
    kv_head_num = kv_hidden_size // head_dim
    UB_SIZE = 87040

    if IS_PARTIAL_ROPE:
        factor = 5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 4 + q_head_num * rope_dim
        batch_size_per_iter_per_vec = int(UB_SIZE / input.element_size()) // factor
    else:
        factor = 5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 2 + q_head_num * rope_dim // 2
        batch_size_per_iter_per_vec = int(UB_SIZE / input.element_size()) // factor
    batch_size_per_iter_per_vec = max(1, batch_size_per_iter_per_vec)
    qk_head_num_sum = int(q_head_num + kv_head_num)
    qk_head_nums_per_iter_per_vec = batch_size_per_iter_per_vec * qk_head_num_sum

    v_batch_size_per_iter_per_vec = int(UB_SIZE / torch.float32.itemsize // (kv_hidden_size + 1))

    # 每步只算一次 cos/sin（同一步的 88 层共用同一缓冲）
    cos_sin_precomputed = _rope_cos_sin_for_step(
        positions, cos_sin_cache, batch_size, rope_dim, num_vectorcore, input.device
    )

    grid = (num_vectorcore, 1, 1)
    split_qkv_rmsnorm_rope_simt_kernel[grid](
        input,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        cos_sin_precomputed,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        BIAS,
        head_dim,
        rope_dim,
        rope_dim // 2,
        IS_PARTIAL_ROPE,
        num_vectorcore,
        batch_size_per_iter_per_vec,
        v_batch_size_per_iter_per_vec,
        qk_head_nums_per_iter_per_vec,
        q_head_num,
        kv_head_num,
        q_head_num + kv_head_num,
    )
    return q_output, k_output, v_output


def split_qkv_rmsnorm_rope_simt_impl_fake(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Fake implementation for shape inference during Dynamo/AOT tracing.
    # Note: sin and cos are not used in shape computation, but must be present in signature.
    batch_size = input.shape[0]
    q_output = torch.empty(
        batch_size,
        int(q_hidden_size),
        device=input.device,
        dtype=input.dtype,
    )
    k_output = torch.empty(
        batch_size,
        int(kv_hidden_size),
        device=input.device,
        dtype=input.dtype,
    )
    v_output = torch.empty(
        batch_size,
        int(kv_hidden_size),
        device=input.device,
        dtype=input.dtype,
    )
    return q_output, k_output, v_output


direct_register_custom_op(
    op_name="qkv_rmsnorm_rope_simt",
    op_func=split_qkv_rmsnorm_rope_simt_impl,
    fake_impl=split_qkv_rmsnorm_rope_simt_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
