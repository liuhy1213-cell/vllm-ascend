#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 PD 分离（4 机 DP2×TP8, mxfp8 KV）— prefill rank0 入口
#   节点: 10.222.43.66 (sjsbg-00001)
#   用法: 本机执行  docker exec -d iquest-mxfp8-p bash /pd/prefill/p0.sh
#   作用: 导出本机变量 → 调用 launch_online_dp.py（它再调同目录 run_dp_template.sh
#         起 1 个 TP8 引擎，监听 :8000，DP rank=0）
# =====================================================================
set -Eeuo pipefail

# ---- 清残留：worker 的 proctitle 是 VLLM::Worker_TPn，必须带大写 VLLM；
#      无残留时 grep/pkill 返回 1，会被 set -e 中断，故用 set +e 包住 ----
set +e
ps aux | grep 'VLLM' | awk '{print $2}' | xargs -r kill -9
ps aux | grep 'vllm' | awk '{print $2}' | xargs -r kill -9
pkill -9 vllm
pkill -9 VLLM
pkill -9 python
pkill -9 ray
set -e
sleep 10

export MODEL_PATH=/models/IQuest-Q1-W8A8C8-mxfp8
export LOCAL_IP=10.222.43.66
export NIC_NAME=enp39s0f3

cd "$(dirname "${BASH_SOURCE[0]}")"
exec /usr/local/python3.12.16/bin/python3 \
  /workspace/examples/external_online_dp/launch_online_dp.py \
  --dp-size 2 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 \
  --dp-address 10.222.43.66 --dp-rpc-port 16500 --vllm-start-port 8000
