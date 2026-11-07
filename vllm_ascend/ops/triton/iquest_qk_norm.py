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
"""Strided per-head RMSNorm for the iQuest-Q1 fused QKV projection.

IQuest-Q1 applies a per-head RMSNorm to the q and k parts of the fused QKV
projection output. ``torch_npu.npu_rms_norm`` only accepts contiguous ND
tensors, so the reference path materializes a contiguous copy of the
strided ``q`` / ``k`` head views on every call (aclnnInplaceCopy_Slice /
_Cast / _TensorMove kernels - ~1 ms per decode step across 88 layers on
8x950DT at conc96, ~4.7% of device time in the 2026-11-07 trace). The
official fused replacement (``npu_qkv_rms_norm_rope_cache``) has no kernel
binary for the 950DT in CANN 9.2.1 ("Op QkvRmsNormRopeCache does not have
any binary"), so the copy cannot be avoided through the aclnn path.

This kernel reads q and k directly from the fused QKV buffer through
explicit strides (Triton handles strided access natively) and writes
fresh contiguous q / k outputs in one launch - no materialization, no
extra fp32 copies. Numerics match ``npu_rms_norm``: fp32 accumulation of
the mean square, fp32 gamma multiply, one bf16 rounding of the result.
"""

import torch
from vllm.triton_utils import tl, triton

# Rows (one row = one head slice of HEAD_DIM elements) per program. A
# (BLOCK_ROWS, HEAD_DIM) fp32 tile stays ~48KB like the sink kernel budget.
_ROWS_PER_PROG = 32


@triton.jit
def _strided_qk_norm_kernel(
    qkv_ptr,  # [T, ROW] fused QKV projection output (bf16/fp16)
    q_out_ptr,  # [T*Q_HEADS, D] contiguous output
    k_out_ptr,  # [T*KV_HEADS, D] contiguous output
    q_gamma_ptr,  # [D]
    k_gamma_ptr,  # [D]
    eps,
    N_Q_PROGS,  # programs owning q rows (scalar, uniform branch)
    T,
    D: tl.constexpr,
    ROW: tl.constexpr,  # fused row width (stride between tokens)
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    K_OFF: tl.constexpr,  # k slice offset inside the row (= Q_HEADS * D)
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    d_offs = tl.arange(0, D)
    if pid < N_Q_PROGS:
        rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid = rows < T * Q_HEADS
        rows = tl.where(valid, rows, 0)
        tok = rows // Q_HEADS
        head = rows % Q_HEADS
        x = tl.load(
            qkv_ptr + tok[:, None] * ROW + head[:, None] * D + d_offs[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(q_gamma_ptr + d_offs).to(tl.float32)
        var = tl.sum(x * x, axis=1) / D
        y = x * tl.rsqrt(var + eps)[:, None] * gamma[None, :]
        tl.store(q_out_ptr + rows[:, None] * D + d_offs[None, :],
                 y.to(q_out_ptr.dtype.element_ty),
                 mask=valid[:, None])
    else:
        rows = (pid - N_Q_PROGS) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid = rows < T * KV_HEADS
        rows = tl.where(valid, rows, 0)
        tok = rows // KV_HEADS
        head = rows % KV_HEADS
        x = tl.load(
            qkv_ptr + tok[:, None] * ROW + K_OFF + head[:, None] * D + d_offs[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(k_gamma_ptr + d_offs).to(tl.float32)
        var = tl.sum(x * x, axis=1) / D
        y = x * tl.rsqrt(var + eps)[:, None] * gamma[None, :]
        tl.store(k_out_ptr + rows[:, None] * D + d_offs[None, :],
                 y.to(k_out_ptr.dtype.element_ty),
                 mask=valid[:, None])


def strided_qk_norm(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm to the q and k slices of a fused QKV tensor.

    The row layout is fixed by QKVParallelLinear: ``[q | k | v]`` with q at
    offset 0, k at ``q_heads * head_dim``.

    Args:
        qkv: fused QKV projection output, [T, row_width]. Only the last dim
            must be contiguous; the token stride is taken from the tensor.
        q_gamma / k_gamma: [head_dim] norm weights.
        q_heads / kv_heads: local per-rank head counts.

    Returns:
        (q_out, k_out): contiguous [T, q_heads*head_dim] / [T, kv_heads*head_dim]
        tensors in qkv's dtype.
    """
    assert qkv.dim() == 2, "fused QKV must be 2D [T, row_width]"
    T, row_width = qkv.shape
    D = head_dim
    k_off = q_heads * D
    assert row_width >= k_off + kv_heads * D, "row too narrow for k slice"

    row_stride = qkv.stride(0)
    q_out = torch.empty((T, q_heads * D), dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty((T, kv_heads * D), dtype=qkv.dtype, device=qkv.device)

    # ROW is the *stride* between tokens (constexpr in-kernel); for the
    # standard contiguous projection output stride == width.
    def cdiv(a, b):
        return (a + b - 1) // b

    block = _ROWS_PER_PROG
    n_q_progs = cdiv(T * q_heads, block)
    grid = (n_q_progs + cdiv(T * kv_heads, block),)
    _strided_qk_norm_kernel[grid](
        qkv,
        q_out,
        k_out,
        q_gamma,
        k_gamma,
        eps,
        n_q_progs,
        T,
        D=D,
        ROW=row_stride,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        K_OFF=k_off,
        BLOCK_ROWS=block,
        num_warps=1,
    )
    return q_out, k_out
