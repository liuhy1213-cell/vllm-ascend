#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 TP16 混部 — node2（worker, 10.222.34.2）启动【全字面值，无变量】
#
# 容器: q1-tp16-mix-node2（部署目录挂 /pd，引擎代码挂 /workspace）
# 用法: 在 34.2 上执行  docker exec -d q1-tp16-mix-node2 bash /pd/serve/n2.sh
# 日志: 直接打屏不落盘；必须重定向到 SFS（见 README §4）
#
# ⚠️ 必须等 node1（39.253）的 n1.sh 先起来再执行本脚本；本机 --headless，
#    不监听 API 端口（node-rank 1 再起 API server 会和 head 抢 8005）。
# ⚠️ 除 --node-rank / --headless 与本机 IP 外，其余参数必须与 n1.sh 完全一致，
#    否则两机建的 EngineCore 配置对不上，rank1 会在初始化阶段卡住。
# =====================================================================
set -Eeuo pipefail

# ---- 清理上一轮残留（同 node1）----
set +e
ps aux | grep 'VLLM' | awk '{print $2}' | xargs -r kill -9
ps aux | grep 'vllm' | awk '{print $2}' | xargs -r kill -9
pkill -9 vllm
pkill -9 VLLM
pkill -9 python
pkill -9 ray
set -e
sleep 10

# ---- 网络：HCCL/Gloo/TP 全绑业务网卡 ----
nic_name="enp39s0f3"
local_ip="10.222.34.2"
master_ip="10.222.39.253"
master_port=29563
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export HCCL_CONNECT_TIMEOUT=180
export HCCL_EXEC_TIMEOUT=180

# ---- 引擎环境（与 n1.sh 同一套）----
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${PYTHONPATH:-}:/workspace"

# ---- custom AllReduce（两端必须一致开启，否则 collective 走不同实现会挂）----
export IQUEST_CUSTOM_AR=1
export VLLM_ASCEND_ENABLE_CUSTOM_AR=1
export CUSTOM_AR_DISPATCH=torch
export CUSTOM_AR_LAYOUT=compact
export VLLM_ASCEND_ENABLE_IQUEST_Q1_FUSION=1

mkdir -p /pd/logs 2>/dev/null || echo "[WARN] /pd/logs 建不出来（容器未挂 /pd？）" >&2

exec /usr/local/python3.12.16/bin/python -m vllm.entrypoints.cli.main serve /models/IQuest-Q1 \
    --served-model-name IQuest-Q1 \
    --tensor-parallel-size 16 \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank 1 \
    --master-addr "$master_ip" --master-port "$master_port" \
    --host 0.0.0.0 --port 8005 \
    --headless \
    --max-model-len 8192 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.95 \
    --no-enable-prefix-caching \
    --speculative-config '{"method":"eagle","model":"/models/IQuest-Q1/mtp","num_speculative_tokens":5}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16]}' \
    --cudagraph-metrics