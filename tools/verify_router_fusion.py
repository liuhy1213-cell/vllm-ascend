#!/usr/bin/env python3
"""MoE Router 归一化链的单算子验证（IQuest-Q1 整网实况 shape）。

生产链（A5DeviceAdaptor.moe_gating_top_k, device_op.py:1211）：
    w, ids, _ = npu_moe_gating_top_k(x, k=8, k_group=1, group_count=1,
                                     group_select_mode=1, renorm=0, norm_type=0,
                                     routed_scaling_factor=1.0, eps=1e-20)   # 1 kernel
    w = w / w.sum(dim=-1, keepdim=True)                                       # 2 kernels (RS + Div)

融合候选：
    A: renorm=1 直接传给 CANN op（若数值一致 → 0 新 kernel，砍掉 RS+Div）
    B: Triton _topk_renorm_kernel（sum+div 融成 1 个 kernel，2→1）

精度门禁：ids bit-exact；weights 逐元素一致或 rel err < 1e-6（fp32）。
性能门禁：链耗时对比 + kernel 数量变化（profiler 抓取，顺带定位 PadV3 来源）。

用法（单卡）：
    ASCEND_RT_VISIBLE_DEVICES=0 python3 tools/verify_router_fusion.py \
        --tokens 6,192,1536,65536
"""
import argparse
import statistics
import sys
import time

import torch
import torch_npu
from vllm.triton_utils import tl, triton


@triton.jit
def _topk_renorm_kernel(
    w_ptr,
    out_ptr,
    num_tokens,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = t < num_tokens
    k = tl.arange(0, K)
    w = tl.load(w_ptr + t[:, None] * K + k[None, :], mask=valid[:, None], other=0.0)
    s = tl.sum(w, axis=1)
    inv = 1.0 / s
    out = w * inv[:, None]
    tl.store(out_ptr + t[:, None] * K + k[None, :], out, mask=valid[:, None])


def topk_renorm_triton(w: torch.Tensor) -> torch.Tensor:
    T, K = w.shape
    out = torch.empty_like(w)
    BLOCK_T = 64
    _topk_renorm_kernel[(triton.cdiv(T, BLOCK_T),)](
        w, out, T, K=K, BLOCK_T=BLOCK_T, num_warps=4,
    )
    return out


def gating(x, renorm):
    return torch_npu.npu_moe_gating_top_k(
        x, k=8, bias=None, k_group=1, group_count=1,
        group_select_mode=1, renorm=renorm, norm_type=0,
        routed_scaling_factor=1.0, eps=1e-20,
    )


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


def trace_kernels(fn, label):
    """跑一次 fn，打印 device kernel 名与次数（定位 PadV3 来源）。"""
    import torch_npu.profiler as prof
    print(f"\n  --- kernel trace: {label} ---")
    try:
        with prof.profile(
            activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
        ) as p:
            fn()
            torch.npu.synchronize()
        seen = {}
        for evt in p.key_averages():
            if evt.device_type is not None and evt.count > 0 and evt.key and (
                "aclnn" in evt.key or "kernel" in evt.key.lower()
                or "MoeGating" in evt.key or "Reduce" in evt.key or "Div" in evt.key
                or "Pad" in evt.key
            ):
                seen[evt.key[:64]] = evt.count
        for k, c in seen.items():
            print(f"      {k:64s} n={c}")
        if not seen:
            print("      (key_averages 无 device kernel 记录)")
    except Exception as e:
        print(f"      profiler 不可用: {str(e)[:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="6,192,1536,65536")
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()

    if not torch.npu.is_available():
        print("NPU 不可用")
        return 2
    dev = "npu:0"
    E = a.num_experts

    print(f"router_logits [T,{E}] fp32, k=8, softmax(norm_type=0), 无 bias")
    print(f"{'T':>7s} {'golden(链)µs':>12s} {'A:renorm=1 µs':>14s} {'B:Triton µs':>11s} "
          f"{'A加速':>7s} {'B加速':>7s}  ids一致  weights一致")

    for T in [int(x) for x in a.tokens.split(",") if x.strip()]:
        torch.manual_seed(1000 + T)
        x = torch.randn(T, E, dtype=torch.float32, device=dev)

        # ---- golden：renorm=0 + Python 归一化 ----
        w0, ids0, _ = gating(x, renorm=0)
        w_ref = w0 / w0.sum(dim=-1, keepdim=True)

        # ---- A：renorm=1（CANN op 内部归一化）----
        try:
            wA, idsA, _ = gating(x, renorm=1)
            a_ok = True
        except Exception as e:
            print(f"{T:7d}  A 失败: {str(e)[:100]}")
            a_ok = False
            wA, idsA = None, None

        # ---- B：renorm=0 + Triton 融合 sum+div ----
        wB = topk_renorm_triton(w0)

        ids_same_a = bool((idsA == ids0).all().item()) if a_ok else False
        ids_same_b = True  # B 用同一 op 输出，ids 必然一致
        exact_a = bool((wA == w_ref).all().item()) if a_ok else False
        exact_b = bool((wB == w_ref).all().item())
        rel_a = ((wA - w_ref).abs() / w_ref.abs()).max().item() if a_ok else float("nan")
        rel_b = ((wB - w_ref).abs() / w_ref.abs()).max().item()

        def golden():
            w, _, _ = gating(x, renorm=0)
            return w / w.sum(dim=-1, keepdim=True)

        t_golden = bench(golden, a.iters)
        t_A = bench(lambda: gating(x, renorm=1)[0], a.iters) if a_ok else float("nan")

        def pathB():
            w, _, _ = gating(x, renorm=0)
            return topk_renorm_triton(w)

        t_B = bench(pathB, a.iters)

        mark_a = "bit-exact" if exact_a else f"relerr={rel_a:.2e}"
        mark_b = "bit-exact" if exact_b else f"relerr={rel_b:.2e}"
        print(f"{T:7d} {t_golden:12.1f} {t_A:14.1f} {t_B:11.1f} "
              f"{t_golden/t_A:6.2f}x {t_golden/t_B:6.2f}x  "
              f"{'PASS' if ids_same_a else 'DIFF'}   A:{mark_a}  B:{mark_b}")

    # ---- kernel 级追踪（定位 PadV3 与 kernel 数变化）----
    T = 6
    torch.manual_seed(7)
    x = torch.randn(T, E, dtype=torch.float32, device=dev)
    trace_kernels(lambda: gating(x, renorm=0)[0], f"op only, T={T}")
    trace_kernels(lambda: (lambda w: w / w.sum(dim=-1, keepdim=True))(gating(x, renorm=0)[0]),
                  f"golden chain, T={T}")
    if True:
        try:
            trace_kernels(lambda: gating(x, renorm=1)[0], f"renorm=1, T={T}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
