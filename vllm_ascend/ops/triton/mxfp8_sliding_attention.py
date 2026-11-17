#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Triton MXFP8 sliding-window attention for C8 KV-cache layers.

Prototype compute path for the IQuest-Q1 sliding-window layers whose KV cache
is stored in the C8-MXFP8 packet layout (E4M3 values + E8M0 scales). The
kernels read the same PA_NZ strided views that QFA consumes:

    K/V        (kernel_blocks, num_kv_heads, head_dim // 32, block_size, 32)
    K scale    (kernel_blocks, num_kv_heads, block_size // 16, head_dim // 64, 16, 2)

Q is the per-step dynamically quantized query (``npu_dynamic_mx_quant``):
``[num_tokens, num_heads, head_dim]`` E4M3 plus E8M0 group scales shaped
``[num_tokens, num_heads, head_dim // 64, 2]``.

Numerics: every E8M0 scale is a power of two, so dequantizing with a BF16
multiply is exact (BF16 keeps the 3-bit E4M3 mantissa and has a wide enough
exponent). Keeping the payload tiles in BF16 also keeps the UB footprint
small, which lets the KV tile be 128 tokens wide. Softmax runs in log2 space
(exp2) and the LSE is converted back to natural log for the sink pass.

Only the single-KV-head-per-rank layout used by IQuest-Q1 under TP8 is
supported (the ModelSlim C8 path already assumes one head slot per rank).
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

_LN2 = 0.6931471805599453

_LSE_BUFFERS: dict[tuple[int, int, int], torch.Tensor] = {}


def _get_lse_buffer(num_tokens: int, num_heads: int, device: torch.device) -> torch.Tensor:
    """Reuse one FP32 LSE buffer per shape (capture/replay safe addresses)."""
    index = device.index if device.index is not None else 0
    key = (num_tokens, num_heads, index)
    buffer = _LSE_BUFFERS.get(key)
    if buffer is None:
        buffer = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=device)
        _LSE_BUFFERS[key] = buffer
    return buffer


@triton.jit
def _mxfp8_sliding_attn_kernel(
    q_ptr,  # [T, H, D] fp8e4m3
    qs_ptr,  # [T, H, D // 64, 2] uint8 (E8M0)
    k_ptr,  # [Bn, 1, D // 32, Bs, 32] fp8e4m3
    v_ptr,  # [Bn, 1, D // 32, Bs, 32] fp8e4m3
    ks_ptr,  # [Bn, 1, Bs // 16, D // 64, 16, 2] uint8 (E8M0)
    vs_ptr,  # [D] uint8 (static per-channel V scale, E8M0)
    bt_ptr,  # [B, MAX_BLOCKS] int32
    qsl_ptr,  # [B + 1] int32
    seq_ptr,  # [B] int32
    out_ptr,  # [T, H, D] bf16
    lse_ptr,  # [T, H] fp32 (natural log)
    scale,  # float32 softmax scale
    stride_k_bn,  # K packet pitch (elements)
    stride_v_bn,  # V packet pitch (elements)
    stride_ks_bn,  # K-scale packet pitch (elements)
    num_tokens,
    MAX_BLOCKS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    req = tl.program_id(0)
    chunk = tl.program_id(1)

    q_start = tl.minimum(tl.maximum(tl.load(qsl_ptr + req), 0), num_tokens)
    q_end = tl.minimum(tl.maximum(tl.load(qsl_ptr + req + 1), 0), num_tokens)
    q_len = q_end - q_start
    chunk_start = chunk * BLOCK_M
    if q_len <= 0 or chunk_start >= q_len:
        return

    seq_len = tl.maximum(tl.load(seq_ptr + req), 1)

    row = tl.arange(0, ROWS_PAD)
    row_valid = row < (BLOCK_M * NUM_HEADS)
    token_i = row // NUM_HEADS
    head_i = row % NUM_HEADS
    valid = row_valid & ((chunk_start + token_i) < q_len)
    tok = q_start + chunk_start + token_i
    pos = seq_len - q_len + chunk_start + token_i

    d = tl.arange(0, BLOCK_D)
    dd = d // 32
    q_off = tok[:, None] * (NUM_HEADS * BLOCK_D) + head_i[:, None] * BLOCK_D + d[None, :]
    q_loaded = tl.load(q_ptr + q_off, mask=valid[:, None] & (tok[:, None] < num_tokens), other=0.0)
    # E8M0 scales: one byte per 32-element group. Load the four group bytes as
    # separate 1-D vectors (no gather/reshape, which trips the Ascend compiler)
    # and expand them across D with tl.where.
    qbase = (
        tok * (NUM_HEADS * (BLOCK_D // 64) * 2)
        + head_i * ((BLOCK_D // 64) * 2)
    )
    qmask = valid & (tok < num_tokens)
    qs0 = tl.exp2(tl.load(qs_ptr + qbase + 0, mask=qmask, other=127).to(tl.float32) - 127.0)
    qs1 = tl.exp2(tl.load(qs_ptr + qbase + 1, mask=qmask, other=127).to(tl.float32) - 127.0)
    qs2 = tl.exp2(tl.load(qs_ptr + qbase + 2, mask=qmask, other=127).to(tl.float32) - 127.0)
    qs3 = tl.exp2(tl.load(qs_ptr + qbase + 3, mask=qmask, other=127).to(tl.float32) - 127.0)
    qs_scale = tl.where(
        dd[None, :] == 0,
        qs0[:, None].to(tl.bfloat16),
        tl.where(
            dd[None, :] == 1,
            qs1[:, None].to(tl.bfloat16),
            tl.where(dd[None, :] == 2, qs2[:, None].to(tl.bfloat16), qs3[:, None].to(tl.bfloat16)),
        ),
    )
    # BF16 multiply is exact for E4M3 payloads scaled by powers of two.
    q = q_loaded.to(tl.bfloat16) * qs_scale

    # Per-token KV bounds for this query chunk (positions are contiguous per
    # request, so the chunk's union window is bounded by BLOCK_M + WINDOW).
    p0 = seq_len - q_len + chunk_start
    p1 = p0 + tl.minimum(BLOCK_M, q_len - chunk_start) - 1
    kv_start = tl.maximum(p0 - (WINDOW - 1), 0)
    kv_end = p1
    b0 = kv_start // BLOCK_SIZE_KV
    b1 = kv_end // BLOCK_SIZE_KV

    # Only one KV head per rank: every query head maps to kv head 0.
    kv_head = 0
    vs = tl.load(vs_ptr + d)
    vs_bf = tl.exp2(vs.to(tl.float32) - 127.0).to(tl.bfloat16)

    sm_scale_log2e = scale * 1.4426950409
    m_i = tl.full((ROWS_PAD,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((ROWS_PAD,), dtype=tl.float32)
    acc = tl.zeros((ROWS_PAD, BLOCK_D), dtype=tl.float32)

    bt_row = bt_ptr + req * MAX_BLOCKS
    for b in tl.range(b0, b1 + 1):
        page = tl.load(bt_row + b, mask=b < MAX_BLOCKS, other=-1)
        if page >= 0:
            # Shrink the sub-tile range to the part of the block that is
            # inside the window (first/last block are usually partial).
            first_t = kv_start - b0 * BLOCK_SIZE_KV
            last_t = kv_end - b1 * BLOCK_SIZE_KV
            t_begin = tl.where(b == b0, (first_t // BLOCK_N) * BLOCK_N, 0)
            t_end = tl.where(b == b1, last_t + 1, BLOCK_SIZE_KV)
            for t0 in tl.range(t_begin, t_end, BLOCK_N):
                t = t0 + tl.arange(0, BLOCK_N)
                kv_pos = b * BLOCK_SIZE_KV + t

                k_off = (
                    page * stride_k_bn
                    + kv_head * (BLOCK_D // 32) * (BLOCK_SIZE_KV * 32)
                    + (d[None, :] // 32) * (BLOCK_SIZE_KV * 32)
                    + t[:, None] * 32
                    + (d[None, :] % 32)
                )
                k_loaded = tl.load(k_ptr + k_off, mask=(t < BLOCK_SIZE_KV)[:, None], other=0.0)

                kbase = (
                    page * stride_ks_bn
                    + kv_head * (BLOCK_SIZE_KV // 16) * (BLOCK_D // 64) * 16 * 2
                    + (t // 16) * ((BLOCK_D // 64) * 16 * 2)
                    + (t % 16) * 2
                )
                tmask = t < BLOCK_SIZE_KV
                ks0 = tl.exp2(tl.load(ks_ptr + kbase + 0, mask=tmask, other=127).to(tl.float32) - 127.0)
                ks1 = tl.exp2(tl.load(ks_ptr + kbase + 2, mask=tmask, other=127).to(tl.float32) - 127.0)
                ks2 = tl.exp2(tl.load(ks_ptr + kbase + 32, mask=tmask, other=127).to(tl.float32) - 127.0)
                ks3 = tl.exp2(tl.load(ks_ptr + kbase + 34, mask=tmask, other=127).to(tl.float32) - 127.0)
                ks_scale = tl.where(
                    dd[None, :] == 0,
                    ks0[:, None].to(tl.bfloat16),
                    tl.where(
                        dd[None, :] == 1,
                        ks1[:, None].to(tl.bfloat16),
                        tl.where(dd[None, :] == 2, ks2[:, None].to(tl.bfloat16), ks3[:, None].to(tl.bfloat16)),
                    ),
                )
                k = k_loaded.to(tl.bfloat16) * ks_scale

                v_off = (
                    page * stride_v_bn
                    + kv_head * (BLOCK_D // 32) * (BLOCK_SIZE_KV * 32)
                    + (d[None, :] // 32) * (BLOCK_SIZE_KV * 32)
                    + t[:, None] * 32
                    + (d[None, :] % 32)
                )
                v_loaded = tl.load(v_ptr + v_off, mask=(t < BLOCK_SIZE_KV)[:, None], other=0.0)
                v = v_loaded.to(tl.bfloat16) * vs_bf[None, :]

                scores = tl.dot(q, tl.trans(k)) * sm_scale_log2e

                kv_ok = (
                    (kv_pos[None, :] <= pos[:, None])
                    & (kv_pos[None, :] >= (pos[:, None] - (WINDOW - 1)))
                    & (kv_pos[None, :] < seq_len)
                )
                scores = tl.where(kv_ok & valid[:, None], scores, float("-inf"))

                m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                active = m_new > float("-inf")
                p = tl.exp2(scores - m_new[:, None])
                p = tl.where(active[:, None], p, 0.0)
                l_new = tl.sum(p, axis=1)
                alpha = tl.where(active, tl.exp2(m_i - m_new), 0.0)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
                l_i = l_i * alpha + l_new
                m_i = m_new

    has_rows = l_i > 0.0
    l_safe = tl.where(has_rows, l_i, 1.0)
    out = (acc.to(tl.float32) / l_safe[:, None]).to(tl.bfloat16)
    out = tl.where(has_rows[:, None], out, 0.0)
    out_off = tok[:, None] * (NUM_HEADS * BLOCK_D) + head_i[:, None] * BLOCK_D + d[None, :]
    tl.store(out_ptr + out_off, out, mask=valid[:, None] & (tok[:, None] < num_tokens))
    # log2-space LSE -> natural log for the sink pass.
    lse = tl.where(has_rows, (m_i + tl.log2(l_safe)) * 0.6931471805599453, float("-inf"))
    tl.store(lse_ptr + tok * NUM_HEADS + head_i, lse, mask=valid & (tok < num_tokens))


def mxfp8_sliding_attention(
    query_fp8: torch.Tensor,
    query_scale: torch.Tensor,
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata,
    output: torch.Tensor,
    scale: float,
    sliding_window: int,
    v_scale_bytes: torch.Tensor,
    block_n: int = 64,
) -> torch.Tensor:
    """Run the Triton sliding C8 attention and return the per-token LSE.

    ``kv_cache`` is the ``(k, v, k_scale, v_scale_cache)`` tuple produced by
    ``mxfp_paged_cache_views``. ``v_scale_bytes`` is the layer's static
    per-channel E8M0 vector (``num_kv_heads * head_dim`` uint8), neutral
    (127 -> 1.0) for layers without a calibrated V scale.
    """
    k_cache, v_cache, k_scale_cache, _v_scale_cache = kv_cache
    num_tokens, num_heads, head_dim = query_fp8.shape
    num_kv_heads = k_cache.shape[1]
    if num_kv_heads != 1:
        raise NotImplementedError("Triton sliding C8 path expects one KV head per rank.")
    block_size_kv = k_cache.shape[3]
    block_table = attn_metadata.block_tables
    qsl = attn_metadata.query_start_loc_gpu
    seq_lens = attn_metadata.seq_lens_gpu
    num_reqs = qsl.numel() - 1
    max_q_len = int(attn_metadata.max_query_len or num_tokens)
    # Small query chunks (decode / MTP verify) shrink the row tile; prefill
    # chunks keep a wider tile to amortize the KV loads.
    block_m = 1 if max_q_len <= 1 else (2 if max_q_len <= 2 else 4)
    num_chunks = max(1, (max_q_len + block_m - 1) // block_m)

    lse = _get_lse_buffer(num_tokens, num_heads, output.device)
    grid = (num_reqs, num_chunks)
    _mxfp8_sliding_attn_kernel[grid](
        query_fp8,
        query_scale,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_bytes,
        block_table,
        qsl,
        seq_lens,
        output,
        lse,
        scale,
        k_cache.stride(0),
        v_cache.stride(0),
        k_scale_cache.stride(0),
        num_tokens,
        block_table.shape[1],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
        NUM_HEADS=num_heads,
        ROWS_PAD=1 << (block_m * num_heads - 1).bit_length(),
        WINDOW=int(sliding_window),
        BLOCK_SIZE_KV=int(block_size_kv),
    )
    return lse
