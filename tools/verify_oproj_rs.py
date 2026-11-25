#!/usr/bin/env python3
"""单算子验证：o_proj 的 TP AllReduce vs ReduceScatter(+AllGather) 组合。

整网实况（IQuest-Q1，TP=8，hidden=3072，bf16）：
    o_proj partial 输出 [T, 3072]（每 rank 持有部分和）
    现状: tensor_model_parallel_all_reduce  → 每 rank 全量 [T,3072]
    改法: sp_reduce_scatter → [T/8,3072]（每 rank 1/8 行）
          POC 组合: RS + sp_all_gather → 全量（语义等价，通信量 ~1.25x vs AR 2x）

精度判据：
    A) RS+AG 组合 vs AR：逐元素一致（求和顺序相同则 bit-exact，bf16 容差兜底）
    B) RS vs AR+shard：每 rank 分片与 AR 后切片一致（bf16 容差）
性能：各 T 的耗时 + 有效带宽。

用法（8 卡空闲机器）：
    torchrun --nproc_per_node=8 --master_port=29590 verify_oproj_rs.py
"""
import os
import statistics
import sys
import time

import torch
import torch_npu


def init_env(local_rank: int):
    global _CFG_CTX
    from vllm.config import VllmConfig, set_current_vllm_config
    _CFG_CTX = set_current_vllm_config(VllmConfig())
    _CFG_CTX.__enter__()

    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    torch.npu.set_device(local_rank)
    init_distributed_environment(
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
        local_rank=local_rank,
        distributed_init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        backend="hccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=int(os.environ["WORLD_SIZE"]))


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    init_env(local_rank)

    from vllm.distributed import (
        tensor_model_parallel_all_reduce as ar,
    )
    from vllm_ascend.models.common.ops.sequence_parallel import (
        sp_all_gather,
        sp_reduce_scatter,
    )

    dev = f"npu:{local_rank}"
    H = 3072
    tokens = [int(x) for x in os.environ.get("RS_TEST_TOKENS", "6,192,1536,65536").split(",")]
    iters = int(os.environ.get("RS_TEST_ITERS", "10"))

    def bench(fn, n=iters):
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(ts)

    if rank == 0:
        print(f"\nTP={world}  H={H}  bf16   （字节口径: AR=2×T·H, RS+AG≈1.25×T·H）")
        print(f"{'T':>7s} {'AR µs':>9s} {'RS+AG µs':>10s} {'RS µs':>8s} {'加速(POC)':>10s} {'AR GB/s':>9s} {'组合 GB/s':>10s}")

    for T in tokens:
        # 每 rank 的 partial 输出（模拟 o_proj 分片 GEMM 结果）
        torch.manual_seed(1000 + rank)
        x = (torch.randn(T, H, dtype=torch.bfloat16, device=dev) * 0.05).contiguous()

        # ---- 精度 A: RS+AG vs AR（数值路径：两者都是 8 份 partial 求和） ----
        y_ar = ar(x.clone())
        # 与模型代码一致：gather 后截断 padding 行
        y_rsag = sp_all_gather(sp_reduce_scatter(x.clone()))[:T]
        ok_a = bool(torch.allclose(y_ar.float(), y_rsag.float(), atol=2 ** -6, rtol=2 ** -6))
        exact_a = bool((y_ar == y_rsag).all().item())

        # ---- 精度 B: RS 分片 vs AR 后切片 ----
        pad = (-T) % world
        y_rs = sp_reduce_scatter(x.clone())
        if pad:
            ref = torch.cat([y_ar, y_ar.new_zeros(pad, H)], dim=0)
        else:
            ref = y_ar
        chunk = ref.shape[0] // world
        ref_shard = ref[rank * chunk:(rank + 1) * chunk]
        ok_b = bool(torch.allclose(y_rs.float(), ref_shard.float(), atol=2 ** -6, rtol=2 ** -6))

        # ---- 性能 ----
        t_ar = bench(lambda: ar(x))
        t_rsag = bench(lambda: sp_all_gather(sp_reduce_scatter(x)))
        t_rs = bench(lambda: sp_reduce_scatter(x))
        bytes_moved_ar = 2 * T * H * 2          # 发 full + 收 full
        bytes_moved_rsag = (T * H + T * H / world) * 2  # 发 full 收 1/8 (RS) + 发 1/8 收 full (AG)

        if rank == 0:
            print(f"{T:7d} {t_ar:9.1f} {t_rsag:10.1f} {t_rs:8.1f} {t_ar/t_rsag:9.2f}x "
                  f"{bytes_moved_ar/t_ar/1e3:9.1f} {bytes_moved_rsag/t_rsag/1e3:10.1f}"
                  f"   精度A: {'一致' if exact_a else ('容差内' if ok_a else 'FAIL')}"
                  f"  精度B: {'PASS' if ok_b else 'FAIL'}")

    import torch.distributed as dist
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
