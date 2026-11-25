# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Fused MXFP8 quantize + PA_NZ packet scatter for the C8 KV cache.

Replaces four kernels per layer (``npu_dynamic_mx_quant`` for K,
``npu_quantize`` for V, ``npu_scatter_pa_kv_cache`` and the Triton
``scatter_k_scales``) with a single kernel that

* computes the per-token-group E8M0 scale and FP8 E4M3 payload of K,
* converts V with the checkpoint's static per-channel reciprocal scale,
* writes K/V into the PA_NZ packet views and the K scales into the 6-D
  NZ scale view.

The K quantization follows ``npu_dynamic_mx_quant``: for every 32-element
group ``exp = floor(log2(amax)) - 8`` (all-zero group -> scale 1), payload
``= x / 2**exp`` rounded to E4M3 (RNE). The FP8 cast is the hardware
conversion (verified bit-identical to torch's ``float8_e4m3fn`` on NPU);
values above 448 are clamped first because the Triton cast yields NaN
where the CANN op saturates.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _quant_scatter_kernel(
    key_ptr,  # bf16 [T, N, D]
    value_ptr,  # bf16 [T, N, D]
    vscale_ptr,  # bf16 [N * D] reciprocal static scale
    slot_ptr,  # int32 [T]
    k_ptr,  # fp8 PA_NZ [Bn, N, D // 32, Bs, 32]
    v_ptr,  # fp8 PA_NZ [Bn, N, D // 32, Bs, 32]
    ks_ptr,  # e8m0 [Bn, N, Bs // 16, D // 64, 16, 2]
    num_pairs,
    NUM_HEADS: tl.constexpr,  # KV heads (input rows are [T, KV, D])
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K_PAGE_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr,
    V_PAGE_STRIDE: tl.constexpr,
    V_HEAD_STRIDE: tl.constexpr,
    KS_PAGE_STRIDE: tl.constexpr,
    KS_HEAD_STRIDE: tl.constexpr,
    KS_SEG_STRIDE: tl.constexpr,
    KS_D1_STRIDE: tl.constexpr,
    KS_FRAG_STRIDE: tl.constexpr,
    KS_PAIR_STRIDE: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
    SCALE_ALG: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    for i in tl.static_range(TOKENS_PER_PROGRAM):
        pair_index = pid * TOKENS_PER_PROGRAM + i
        tok = pair_index // NUM_HEADS
        head = pair_index % NUM_HEADS
        if pair_index < num_pairs:
            slot = tl.load(slot_ptr + tok)
            if slot >= 0:
                blk = slot // BLOCK_SIZE
                off = slot % BLOCK_SIZE

                d32 = tl.arange(0, 32)
                row = tok * (NUM_HEADS * HEAD_DIM) + head * HEAD_DIM
                k_row = k_ptr + blk * K_PAGE_STRIDE + head * K_HEAD_STRIDE + off * 32
                v_row = v_ptr + blk * V_PAGE_STRIDE + head * V_HEAD_STRIDE + off * 32
                ks_base = (
                    blk * KS_PAGE_STRIDE
                    + head * KS_HEAD_STRIDE
                    + (off // 16) * KS_SEG_STRIDE
                    + (off % 16) * KS_FRAG_STRIDE
                )

                # K: one 32-element MX group per iteration over the head dimension.
                for g in tl.static_range(HEAD_DIM // 32):
                    x = tl.load(key_ptr + row + g * 32 + d32).to(tl.float32)
                    amax = tl.max(tl.abs(x))
                    # floor(log2(amax)) via the fp32 exponent field, then -8.
                    abits = amax.to(tl.int32, bitcast=True)
                    e = ((abits >> 23) & 0xFF) - 135
                    e = tl.where(amax == 0.0, 0, e)
                    e = tl.minimum(tl.maximum(e, -127), 127)
                    # scale_alg=1（线上默认）：scale 取到让 amax 不饱和，
                    # 即 exp = ceil(log2(amax/448))；alg0 则是 floor(log2(amax))-8。
                    # 两者只差一次进位：floor 口径下 amax/2^e ∈ [256,512)，>448 就 +1。
                    if SCALE_ALG == 1:
                        e0 = ((e + 127) << 23).to(tl.float32, bitcast=True)   # 2^e
                        inv0 = ((127 - e) << 23).to(tl.float32, bitcast=True)  # 2^-e
                        need = amax * inv0 > 448.0
                        e = tl.where(need, e + 1, e)
                    # 2^e and 2^-e from the exponent field (no exp2/div).
                    scale = ((e + 127) << 23).to(tl.float32, bitcast=True)
                    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
                    qf = tl.minimum(tl.maximum(x * inv, -448.0), 448.0)
                    q = qf.to(tl.float8e4nv)
                    tl.store(k_row + g * (BLOCK_SIZE * 32) + d32, q)
                    sbyte = (e + 127).to(tl.uint8)
                    tl.store(
                        ks_ptr
                        + ks_base
                        + (g // 2) * KS_D1_STRIDE
                        + (g % 2) * KS_PAIR_STRIDE,
                        sbyte,
                    )

                # V: static per-channel scale, no per-token reduction.
                dv = tl.arange(0, HEAD_DIM)
                v = tl.load(value_ptr + row + dv).to(tl.float32)
                vs = tl.load(vscale_ptr + head * HEAD_DIM + dv).to(tl.float32)
                vf = tl.minimum(tl.maximum(v * vs, -448.0), 448.0)
                qv = vf.to(tl.float8e4nv)
                tl.store(v_row + (dv // 32) * (BLOCK_SIZE * 32) + (dv % 32), qv)


@triton.jit
def _wide_quant_scatter_kernel(
    key_ptr,
    value_ptr,
    vscale_ptr,
    slot_ptr,
    k_ptr,
    v_ptr,
    ks_ptr,
    num_tokens,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K_PAGE_STRIDE: tl.constexpr,
    V_PAGE_STRIDE: tl.constexpr,
    KS_PAGE_STRIDE: tl.constexpr,
    KS_SEG_STRIDE: tl.constexpr,
    KS_D1_STRIDE: tl.constexpr,
    KS_FRAG_STRIDE: tl.constexpr,
    KS_PAIR_STRIDE: tl.constexpr,
    SCALE_ALG: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    """Wide-tile variant: BLOCK_TOKENS tokens per program, 2-D [BT, 32] tiles.

    Only valid for NUM_HEADS == 1 (adjacent tokens are contiguous in [T, 1, D]).
    At T=65536 this runs ~4.9x faster than the per-token kernel and ~1.18x
    faster than the native 4-op path, so the prefill-side batch gate can go.
    """
    pid = tl.program_id(0)
    tok = pid * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    valid = tok < num_tokens

    slot_raw = tl.load(slot_ptr + tok, mask=valid, other=0)
    slot_ok = slot_raw >= 0          # 与 npu_scatter_pa_kv_cache 一致：PAD 不写 K/V
    slot = tl.where(slot_ok, slot_raw, 0)   # 但 k_scale 与生产链一致：PAD clamp 到 slot 0
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE

    d32 = tl.arange(0, 32)
    row = tok * HEAD_DIM

    for g in tl.static_range(HEAD_DIM // 32):
        x = tl.load(key_ptr + row[:, None] + g * 32 + d32[None, :],
                    mask=valid[:, None], other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        abits = amax.to(tl.int32, bitcast=True)
        e = ((abits >> 23) & 0xFF) - 135
        e = tl.where(amax == 0.0, 0, e)
        e = tl.minimum(tl.maximum(e, -127), 127)
        if SCALE_ALG == 1:
            inv0 = ((127 - e) << 23).to(tl.float32, bitcast=True)
            e = tl.where(amax * inv0 > 448.0, e + 1, e)
        inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
        qf = tl.minimum(tl.maximum(x * inv[:, None], -448.0), 448.0)
        q = qf.to(tl.float8e4nv)

        k_off = (blk[:, None] * K_PAGE_STRIDE + off[:, None] * 32
                 + g * (BLOCK_SIZE * 32) + d32[None, :])
        tl.store(k_ptr + k_off, q, mask=valid[:, None] & slot_ok[:, None])

        ks_off = (blk * KS_PAGE_STRIDE + (off // 16) * KS_SEG_STRIDE
                  + (off % 16) * KS_FRAG_STRIDE
                  + (g // 2) * KS_D1_STRIDE + (g % 2) * KS_PAIR_STRIDE)
        tl.store(ks_ptr + ks_off, (e + 127).to(tl.uint8), mask=valid)

    dv = tl.arange(0, HEAD_DIM)
    vs = tl.load(vscale_ptr + dv).to(tl.float32)
    v = tl.load(value_ptr + row[:, None] + dv[None, :],
                mask=valid[:, None], other=0.0).to(tl.float32)
    vf = tl.minimum(tl.maximum(v * vs[None, :], -448.0), 448.0)
    qv = vf.to(tl.float8e4nv)
    v_off = (blk[:, None] * V_PAGE_STRIDE + off[:, None] * 32
             + (dv[None, :] // 32) * (BLOCK_SIZE * 32) + (dv[None, :] % 32))
    tl.store(v_ptr + v_off, qv, mask=valid[:, None] & slot_ok[:, None])


def quant_scatter_mxfp8(
    key: torch.Tensor,
    value: torch.Tensor,
    v_scale_reciprocal: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    tokens_per_program: int = 1,
    num_warps: int = 4,
    scale_alg: int = 1,
) -> None:
    """Quantize K/V to MXFP8 and scatter into the C8 packet caches.

    ``key``/``value``: ``[T, N, D]`` bf16 (not yet quantized). Non-contiguous
    inputs are materialized to a contiguous layout first, because the kernel
    addresses rows with the flat ``[T, N, D]`` stride.
    ``key_cache``/``value_cache``: 5-D PA_NZ views ``(Bn, N, D // 32, Bs, 32)``
    whose dim-0 stride is the full packet pitch.
    ``k_scale_cache``: 6-D view ``(Bn, N, Bs // 16, D // 64, 16, 2)``.
    ``slot_mapping``: ``[T]`` int32; negative slots (padding) are skipped, matching
    ``npu_scatter_pa_kv_cache`` (the production K/V path).
    """
    # Callers may hand over strided views: e.g. ``v`` stays a slice of the
    # fused QKV output when enable_iquest_split_qkv_rmsnorm_rope is off, and
    # ``k`` is a strided view before the K-centering subtraction. The kernel
    # computes flat [T, N, D] addresses, so make the layout explicit here.
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    num_tokens, num_heads, head_dim = key.shape
    if num_tokens == 0:
        return
    block_size = key_cache.shape[3]

    ks_view = k_scale_cache.view(torch.uint8) if k_scale_cache.dtype != torch.uint8 else k_scale_cache
    # 宽 tile 路径：NUM_HEADS == 1 时相邻 token 连续，按 T 选 BLOCK_TOKENS
    # （实测 T<=8192 时 32 最优、T>8192 时 64 最优；65536 下 1.18x 于原生）
    if num_heads == 1:
        block_tokens = 32 if num_tokens <= 8192 else 64
        grid = (triton.cdiv(num_tokens, block_tokens),)
        _wide_quant_scatter_kernel[grid](
            key,
            value,
            v_scale_reciprocal,
            slot_mapping,
            key_cache,
            value_cache,
            ks_view,
            num_tokens,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            K_PAGE_STRIDE=key_cache.stride(0),
            V_PAGE_STRIDE=value_cache.stride(0),
            KS_PAGE_STRIDE=k_scale_cache.stride(0),
            KS_SEG_STRIDE=k_scale_cache.stride(2),
            KS_D1_STRIDE=k_scale_cache.stride(3),
            KS_FRAG_STRIDE=k_scale_cache.stride(4),
            KS_PAIR_STRIDE=k_scale_cache.stride(5),
            SCALE_ALG=scale_alg,
            BLOCK_TOKENS=block_tokens,
            num_warps=num_warps,
        )
        return

    num_pairs = num_tokens * num_heads
    grid = (triton.cdiv(num_pairs, tokens_per_program),)
    _quant_scatter_kernel[grid](
        key,
        value,
        v_scale_reciprocal,
        slot_mapping,
        key_cache,
        value_cache,
        k_scale_cache.view(torch.uint8) if k_scale_cache.dtype != torch.uint8 else k_scale_cache,
        num_pairs,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        K_PAGE_STRIDE=key_cache.stride(0),
        K_HEAD_STRIDE=key_cache.stride(1),
        V_PAGE_STRIDE=value_cache.stride(0),
        V_HEAD_STRIDE=value_cache.stride(1),
        KS_PAGE_STRIDE=k_scale_cache.stride(0),
        KS_HEAD_STRIDE=k_scale_cache.stride(1),
        KS_SEG_STRIDE=k_scale_cache.stride(2),
        KS_D1_STRIDE=k_scale_cache.stride(3),
        KS_FRAG_STRIDE=k_scale_cache.stride(4),
        KS_PAIR_STRIDE=k_scale_cache.stride(5),
        TOKENS_PER_PROGRAM=tokens_per_program,
        SCALE_ALG=scale_alg,
        num_warps=num_warps,
    )
