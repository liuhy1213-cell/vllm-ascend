#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""torch_npu profiler kernel_details.csv 流式分析器（宿主机可用，无需 torch_npu）。

用法:
  python3 analyze_trace.py <kernel_details.csv> [--top 20] [--step-gap-ms 3]

说明（CANN 版本差异）: kernel_details.csv 的 Type 列在此版本是算子类名（Fill/Cast/...），
所有行都是设备侧事件，以 Accelerator Core（AI_CORE/AI_VECTOR_CORE/COMMUNICATION/
AI_CPU/MIX_*）区分执行单元。cpu_op 不在此 CSV。

产出:
  1) 采集窗口 / NPU busy-idle（按 device 流合并区间）
  2) 分桶耗时: comm/attention/matmul/moe/cast/norm-rope/elementwise/copy/sample/other
  3) Top-N kernel（按总时长）
  4) 步进切分: 相邻空隙 > step-gap-ms 切段，decode 侧每段≈一步
"""
import argparse
import csv
import re
import sys
from collections import defaultdict

BUCKETS = [
    ("comm", re.compile(r"hcom_|hccl|Hccl|SendRecv|AllReduce|AllGather|ReduceScatter|Broadcast", re.I)),
    ("attention", re.compile(r"FusedInferAttention|fused_infer_attention|FlashAttention|PromptSparse|PagedAttention|RingAttention|SplitFuse", re.I)),
    ("matmul", re.compile(r"MatMul|BatchMatMul|Gemm|GroupedMatmul|Gmm|Linear", re.I)),
    ("moe", re.compile(r"MoeInitRouting|MoeFinalize|MoeGating|MoeCompute|FusedMoE", re.I)),
    ("quant", re.compile(r"Quant(?!um)|Dequant|PerChannel|MxQuant|MxDequant|GroupedDynamic", re.I)),
    ("norm_rope", re.compile(r"RmsNorm|LayerNorm|triton_rope|RotaryPosition|RoPE", re.I)),
    ("sample", re.compile(r"ArgMax|Argmax|Softmax|TopK|TopP|Multinomial|Sampling|rejection_random|Sort|SimThread|RandomUniform", re.I)),
    ("copy_transpose", re.compile(r"Transpose|Contiguous|TensorMove|ViewCopy|Copy|Concat|ConcatD|Reshape|Fill|IndexSelect|StridedSlice|Unstack|Stack|Scatter|Gather(?!V2)|Slice|expand_kernel|Pad|BroadcastTo|Tile|MaskedFill|IndexPut|ScatterPaKvCache|prepare_inputs", re.I)),
    ("kv_cache", re.compile(r"KvCache|ScatterPaKvCache|CacheAppend|ReshapeAndCache", re.I)),
]


def classify(name: str, core: str) -> str:
    if core == "COMMUNICATION":
        return "comm"
    for bucket, pat in BUCKETS:
        if pat.search(name):
            return bucket
    return "elementwise" if re.match(r"^[A-Z]", name) else "other"


def merge_intervals(iv):
    if not iv:
        return []
    iv.sort()
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--step-gap-ms", type=float, default=3.0)
    args = ap.parse_args()

    bucket_time = defaultdict(float)
    bucket_cnt = defaultdict(int)
    core_time = defaultdict(float)
    kernel_time = defaultdict(lambda: [0.0, 0])  # name -> [total_us, count]
    stream_iv = defaultdict(list)  # "dev:stream" -> [(start,end)]
    n_rows = 0
    t_min, t_max = float("inf"), 0.0

    with open(args.csv_file, newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            try:
                start = float(row["Start Time(us)"].strip())
                dur = float(row["Duration(us)"].strip())
            except (ValueError, KeyError, TypeError):
                continue
            name = (row.get("Name") or "").strip()
            core = (row.get("Accelerator Core") or "").strip()
            stream = (row.get("Stream ID") or "?").strip()
            dev = (row.get("Device_id") or "0").strip()
            stream_iv[(dev, stream)].append((start, start + dur))
            b = classify(name, core)
            bucket_time[b] += dur
            bucket_cnt[b] += 1
            core_time[core] += dur
            k = kernel_time[name]
            k[0] += dur
            k[1] += 1
            t_min = min(t_min, start)
            t_max = max(t_max, start + dur)

    if not n_rows or t_min == float("inf"):
        print("NO_DATA")
        return

    span = t_max - t_min
    print(f"file={args.csv_file}")
    print(f"rows={n_rows} window={span/1e6:.3f}s [{t_min:.0f} .. {t_max:.0f}]")
    print()

    print("== NPU busy/idle per stream ==")
    total_busy_iv = []
    for key, iv in sorted(stream_iv.items()):
        merged = merge_intervals(iv)
        busy = sum(e - s for s, e in merged)
        total_busy_iv.extend(merged)
        print(f"  dev{key[0]}:stream{key[1]:>4}: events={len(iv):>7} busy={busy/1e3:.1f}ms "
              f"({busy/span*100:.1f}% of window)")
    all_merged = merge_intervals(total_busy_iv)
    busy_total = sum(e - s for s, e in all_merged)
    idle_total = span - busy_total
    print(f"  UNION: busy={busy_total/1e3:.1f}ms ({busy_total/span*100:.1f}%) "
          f"idle={idle_total/1e3:.1f}ms ({idle_total/span*100:.1f}%)")
    print()

    print("== time by bucket ==")
    for b, t in sorted(bucket_time.items(), key=lambda x: -x[1]):
        print(f"  {b:<15} {t/1e3:10.1f}ms  {t/span*100:5.1f}%  n={bucket_cnt[b]}")
    print("== time by core ==")
    for c, t in sorted(core_time.items(), key=lambda x: -x[1]):
        print(f"  {c:<18} {t/1e3:10.1f}ms  {t/span*100:5.1f}%")
    print()

    print(f"== top {args.top} kernels ==")
    for name, (t, c) in sorted(kernel_time.items(), key=lambda x: -x[1][0])[: args.top]:
        print(f"  {t/1e3:10.1f}ms  n={c:>7}  avg={t/c:8.1f}us  {name[:110]}")
    print()

    gap_us = args.step_gap_ms * 1000
    gaps = []
    for (s1, e1), (s2, e2) in zip(all_merged, all_merged[1:]):
        g = s2 - e1
        if g > gap_us:
            gaps.append((e1, g))
    print(f"== idle gaps > {args.step_gap_ms}ms: {len(gaps)} 个 ==")
    seg_bounds = [t_min] + [g[0] for g in gaps] + [t_max]
    segs = sorted(seg_bounds[i + 1] - seg_bounds[i] for i in range(len(seg_bounds) - 1))
    if segs:
        n = len(segs)
        tot = sum(segs)
        print(f"  段数={n} 总和={tot/1e3:.1f}ms 中位={segs[n//2]/1e3:.1f}ms "
              f"min={segs[0]/1e3:.1f}ms max={segs[-1]/1e3:.1f}ms mean={tot/n/1e3:.1f}ms")
        big_gap_time = sum(g for _, g in gaps)
        print(f"  大空隙总时长={big_gap_time/1e3:.1f}ms ({big_gap_time/span*100:.1f}% of window)")


if __name__ == "__main__":
    sys.exit(main())
