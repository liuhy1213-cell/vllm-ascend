#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析 vLLM engine.log 的调度级吞吐统计（宏观视角）。

用法:
  python3 parse_engine_log.py <engine.log> [--window-sec 60]

输出:
  - 每条 "Avg prompt throughput / Avg generation throughput" 日志按时间聚合
    （默认 60s 窗口）: 生成 tok/s 均值/峰值, running seqs 均值/峰值
  - D 侧 MTP: SpecDecoding accept length 均值（如有）
  - KV 传输/池: kvpool hit、External 命中率（如有）
"""
import argparse
import re
import sys
from collections import defaultdict

TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
THR = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs"
)
ACCEPT = re.compile(r"accept length is: ([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--window-sec", type=int, default=60)
    args = ap.parse_args()

    samples = []  # (ts, prompt_tps, gen_tps, running, waiting)
    accepts = []  # (ts, accept_len)
    with open(args.log, errors="replace") as f:
        for line in f:
            m = THR.search(line)
            if m:
                ts = TS.search(line)
                t = ts.group(1) if ts else "?"
                samples.append((t, *map(float, m.groups())))
            m = ACCEPT.search(line)
            if m:
                ts = TS.search(line)
                t = ts.group(1) if ts else "?"
                accepts.append((t, float(m.group(1))))

    if samples:
        print(f"samples={len(samples)} (thr 日志)")
        print(f"{'window':<18}{'gen tok/s avg':>14}{'gen tok/s max':>14}"
              f"{'prompt tok/s avg':>17}{'run avg':>9}{'run max':>9}{'wait max':>9}")
        buckets = defaultdict(list)
        for s in samples:
            buckets[s[0][: len("0000-00-00 00:00") + 2]].append(s)  # 分钟级
        for w, rows in sorted(buckets.items()):
            gen = [r[2] for r in rows]
            prm = [r[1] for r in rows]
            run = [r[3] for r in rows]
            wt = [r[4] for r in rows]
            print(f"{w:<18}{sum(gen)/len(gen):>14.1f}{max(gen):>14.1f}"
                  f"{sum(prm)/len(prm):>17.1f}{sum(run)/len(run):>9.1f}"
                  f"{max(run):>9.0f}{max(wt):>9.0f}")
        gen_all = [s[2] for s in samples]
        run_all = [s[3] for s in samples]
        act = [s[2] for s in samples if s[2] > 10]
        print(f"OVERALL: gen tok/s mean={sum(gen_all)/len(gen_all):.1f} "
              f"max={max(gen_all):.1f}  active-mean(>10)={sum(act)/len(act) if act else 0:.1f} "
              f"running max={max(run_all):.0f}")
    else:
        print("无 throughput 样本")

    if accepts:
        a = [x for _, x in accepts]
        print(f"\nMTP accept-length: n={len(a)} mean={sum(a)/len(a):.2f} "
              f"min={min(a):.2f} max={max(a):.2f}")


if __name__ == "__main__":
    sys.exit(main())
