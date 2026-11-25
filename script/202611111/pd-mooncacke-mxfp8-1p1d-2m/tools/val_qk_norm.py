import time

import torch
import torch_npu
from vllm.triton_utils import tl, triton  # noqa: F401  (JIT env init)

from vllm_ascend.ops.triton.iquest_qk_norm import strided_qk_norm

dev = "npu:0"
torch.manual_seed(0)
D = 128


def reference(qkv, q_gamma, k_gamma, q_heads, kv_heads, eps):
    T = qkv.shape[0]
    q_size = q_heads * D
    kv_size = kv_heads * D
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)

    def norm(x, g):
        xf = x.float().view(T, -1, D)
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps) * g.float()).to(x.dtype)

    qn = norm(q, q_gamma).reshape(T, q_size)
    kn = norm(k, k_gamma).reshape(T, kv_size)
    return qn, kn


for T, QH, KVH, dtype in [
    (384, 6, 1, torch.bfloat16),
    (96, 6, 1, torch.bfloat16),
    (8, 6, 1, torch.bfloat16),
    (320, 6, 1, torch.bfloat16),
    (384, 6, 1, torch.float16),
    (77, 3, 1, torch.bfloat16),  # TP16-like odd token count
]:
    row = QH * D + 2 * KVH * D
    qkv = torch.randn(T, row, dtype=dtype, device=dev)
    qg = torch.randn(D, dtype=dtype, device=dev)
    kg = torch.randn(D, dtype=dtype, device=dev)
    eps = 1e-6

    q_out, k_out = strided_qk_norm(qkv, qg, kg, QH, KVH, D, eps)
    torch.npu.synchronize()
    q_ref, k_ref = reference(qkv, qg, kg, QH, KVH, eps)
    qe = (q_out.float() - q_ref.float()).abs().max().item()
    ke = (k_out.float() - k_ref.float()).abs().max().item()
    # bf16 quantum at |x|~2 is ~0.0156; fp32-accum rmsnorm diff should be < 1 ulp
    status = "OK " if qe < 0.02 and ke < 0.02 else "FAIL"
    print(f"{status} T={T:4d} QH={QH} KVH={KVH} {str(dtype):15s} q_err={qe:.4e} k_err={ke:.4e}")

# ---- microbench: reference chain (contiguous + npu_rms_norm) vs fused ----
T, QH, KVH = 384, 6, 1
row = QH * D + 2 * KVH * D
qkv = torch.randn(T, row, dtype=torch.bfloat16, device=dev)
qg = torch.randn(D, dtype=torch.bfloat16, device=dev)
kg = torch.randn(D, dtype=torch.bfloat16, device=dev)
q_size, kv_size = QH * D, KVH * D
eps = 1e-6


def ref_chain(qkv):
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = torch_npu.npu_rms_norm(
        q.view(T, QH, D).contiguous().to(qg.dtype), qg, eps)[0]
    k = torch_npu.npu_rms_norm(
        k.view(T, KVH, D).contiguous().to(kg.dtype), kg, eps)[0]
    return q, k, v


def tri_chain(qkv):
    q, k = strided_qk_norm(qkv, qg, kg, QH, KVH, D, eps)
    v = qkv[:, q_size + kv_size:]
    return q, k, v


for name, fn in [("ref", ref_chain), ("triton", tri_chain)]:
    for _ in range(10):
        fn(qkv)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    N = 200
    for _ in range(N):
        fn(qkv)
    torch.npu.synchronize()
    print(f"{name:7s} {(time.perf_counter()-t0)/N*1e6:8.1f} us/call (T={T})")
