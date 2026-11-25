#!/usr/bin/env python3
# 流式分析 torch_npu trace_view.json（chrome trace）: 设备事件分桶 + top kernels + 空隙。
# 用法: python3 analyze_tracejson.py <trace_view.json> [--top 12] [--gap-ms 3]
import json, sys, re, argparse

p = argparse.ArgumentParser()
p.add_argument('trace_file')
p.add_argument('--top', type=int, default=12)
p.add_argument('--gap-ms', type=float, default=3.0)
a = p.parse_args()

# 流式读取: trace_view.json 是单个大数组,逐字符扫描太慢;用增量 json 解码器
dec = json.JSONDecoder()
buf = ''
events = []
with open(a.trace_file, 'r', encoding='utf-8', errors='replace') as f:
    head = f.read(4096)
    i = head.find('[')
    f.seek(i)
    data = json.load(f)  # 437MB -> 内存可承受(约4-5GB),主机内存充足
if isinstance(data, dict):
    data = data.get('traceEvents', [])
events = data

def bucket(name, args):
    n = name or ''
    ac = str(args.get('Accelerator Core', args.get('accelerator core', '')))
    if ac == 'COMMUNICATION' or 'hccl' in n.lower() or 'Hccl' in n:
        return 'comm'
    ln = n.lower()
    if 'flashattention' in ln or 'fia' in ln or 'attention' in ln:
        return 'attention'
    if 'matmul' in ln or 'groupedmatmul' in ln or 'quantbatchmatmul' in ln:
        return 'matmul' if 'quant' not in ln else 'quant'
    if 'moe' in ln or 'permute' in ln or 'routing' in ln or 'swiglu' in ln:
        return 'moe'
    if 'rmsnorm' in ln or 'rope' in ln or 'norm' in ln:
        return 'norm_rope'
    if 'quant' in ln or 'dequant' in ln:
        return 'quant'
    if 'copy' in ln or 'transpose' in ln or 'cast' in ln or 'slice' in ln:
        return 'copy_transpose'
    if 'aclnn' in ln or 'aivkernel' in ln or 'aicorekernel' in ln or ln.startswith('acl'):
        return 'other_acl'
    return 'other'

dur = {}
cnt = {}
core_time = {}
tmin = None
tmax = None
for e in events:
    if e.get('ph') != 'X':
        continue
    args = e.get('args') or {}
    # 设备事件: args 里有 Device Id 或 stream 字段
    if 'Device Id' not in args and 'device_id' not in args and 'Stream' not in args and 'stream' not in args:
        continue
    ts, d = e.get('ts', 0), e.get('dur', 0)
    name = e.get('name', '?')
    b = bucket(name, args)
    dur[b] = dur.get(b, 0) + d
    cnt[b] = cnt.get(b, 0) + 1
    k = (b, name[:90])
    dur[k] = dur.get(k, 0) + d
    cnt[k] = cnt.get(k, 0) + 1
    ac = str(args.get('Accelerator Core', '?'))
    core_time[ac] = core_time.get(ac, 0) + d
    tmin = ts if tmin is None else min(tmin, ts)
    tmax = ts + d if tmax is None else max(tmax, ts + d)

win = (tmax - tmin) if tmin is not None else 0
print(f'file={a.trace_file}')
print(f'device_events={sum(cnt.get(b,0) for b in list(dur) if isinstance(b,str))} window={win/1e6:.3f}s')
total = sum(v for k, v in dur.items() if isinstance(k, str))
print('\n== time by bucket (us / % of device busy / n) ==')
for b, v in sorted(dur.items(), key=lambda x: -x[1]):
    if isinstance(b, str):
        print(f'  {b:16s} {v/1e3:12.1f}ms {100*v/total:5.1f}%  n={cnt[b]}')
print('\n== time by accelerator core ==')
for c, v in sorted(core_time.items(), key=lambda x: -x[1]):
    print(f'  {c:20s} {v/1e3:12.1f}ms {100*v/total:5.1f}%')
print(f'\n== top {a.top} kernels ==')
kern = [(v, cnt[k], k[1]) for k, v in dur.items() if not isinstance(k, str)]
for v, n, name in sorted(kern, key=lambda x: -x[0])[:a.top]:
    print(f'  {v/1e3:10.1f}ms  n={n:7d}  avg={v/max(n,1):8.1f}us  {name}')
