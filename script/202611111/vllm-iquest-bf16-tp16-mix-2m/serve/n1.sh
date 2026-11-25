#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 TP16 混部 — node1（head, 10.222.39.253）启动【全字面值，无变量】
#
# 容器: q1-tp16-mix-node1（部署目录挂 /pd，引擎代码挂 /workspace）
# 用法: 在 39.253 上执行  docker exec -d q1-tp16-mix-node1 bash /pd/serve/n1.sh
# 日志: 直接打屏不落盘；用 docker exec -d 起则输出被丢弃，必须重定向到 SFS
#
# ⚠️ 启动顺序: 本脚本（rank0）→ n2.sh（rank1, --headless）。rendezvous 在本机 29563，
#    rank1 反了会一直重连。混部没有 mooncake_master / proxy，也没有 KV 传输。
#
# ⚠️ 取值口径: max-model-len 8192 / batched-tokens 1024 / seqs 8 / util 0.95 /
#    MTP depth 5 / capture [4,8,16]。capture 档位只到 16 时，seqs×(depth+1) 超过
#    16 的 decode 步会回落 eager（图对那部分不生效）；要压满吞吐可加档到 48:
#    '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16,24,32,48]}'
#
# ⚠️ 8005 若已被别的服务占用（2026-11-11 现场: q1-tp16-ziang-ar-mooncake-2tib-node1
#    正在用），先停它或改下面 --port（两台要一致）。
# =====================================================================
set -Eeuo pipefail

# ---- 清理上一轮残留（同 PD 站脚本口径）----
# 1) worker proctitle 是 VLLM::Worker_TPn_EPn，必须带大写 VLLM 才杀得到；
# 2) 本段杀容器内所有 python —— 不要把 proxy/别的 python 服务放同一容器；
# 3) 用 set +e 包住：无残留时 grep/pkill 返回 1，会被 set -e/pipefail 中断。
set +e
ps aux | grep 'VLLM' | awk '{print $2}' | xargs -r kill -9
ps aux | grep 'vllm' | awk '{print $2}' | xargs -r kill -9
pkill -9 vllm
pkill -9 VLLM
pkill -9 python
pkill -9 ray
set -e
sleep 10

# ---- 端口预检：head 要监听 :8005 和 :29563 ----
PORT=8005
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "[ERROR] 本机 :$PORT 已被占用，先停掉占用者或改 --port（两台一致）:" >&2
    ss -ltnp 2>/dev/null | grep ":$PORT " >&2 || true
    exit 1
fi

# ---- 网络：HCCL/Gloo/TP 全绑业务网卡 ----
nic_name="enp39s0f3"
local_ip="10.222.39.253"
master_ip="10.222.39.253"
master_port=29563
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export HCCL_CONNECT_TIMEOUT=180
export HCCL_EXEC_TIMEOUT=180

# ---- 引擎环境（镜像 ENV 已含 SOC_VERSION / ATB_* / TASK_QUEUE_ENABLE /
#      LCCL_* / ASCEND_* 路径，这里只保留需要覆盖或补充的项）----
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# vLLM 默认 fork；本站沿用 spawn（spawn 会重新 import，故 PYTHONPATH 必须追加式）
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 追加式，不能整体覆盖：镜像默认 PYTHONPATH 已含 acl 等 toolkit 路径，
# 覆盖会让 spawn worker 的 `from acl.rt import memcpy` 报 No module named 'acl'
export PYTHONPATH="${PYTHONPATH:-}:/workspace"

# ---- custom AllReduce（TP16 跨机；插件在 /workspace/custom-ar）----
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
    --nnodes 2 --node-rank 0 \
    --master-addr "$master_ip" --master-port "$master_port" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 8192 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.95 \
    --no-enable-prefix-caching \
    --speculative-config '{"method":"eagle","model":"/models/IQuest-Q1/mtp","num_speculative_tokens":5}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16]}' \
    --cudagraph-metrics