#!/usr/bin/env python3
"""torch_npu analyse 单目录解析器: python3 run_analyse.py <*_ascend_pt 目录>"""
import sys, time
from torch_npu.profiler.profiler import analyse
t0 = time.time()
analyse(sys.argv[1])
print(f"[analyse done] {sys.argv[1]} in {time.time()-t0:.0f}s")
