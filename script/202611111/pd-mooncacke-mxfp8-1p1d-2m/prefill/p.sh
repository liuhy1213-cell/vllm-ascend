#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 PD 分离 — prefill（本 2m 站 = 46.181 new4）启动【全字面值，无变量】
#
# 容器: pd-mooncacke-p（部署目录挂载 /pd，引擎代码挂 /workspace）
# 用法: 在 46.181 上执行  docker exec -d pd-mooncacke-p bash /pd/prefill/p.sh
# 日志: 直接打屏不落盘；用 docker exec -d 起则输出被丢弃（要看输出就去掉 -d 前台 exec）
#
# ⚠️ 启动顺序: master(46.181) → decode(47.74) → 本脚本 → proxy
#    master 必须先起（引擎初始化即连 50088）:
#    docker exec -d pd-mooncacke-p bash /pd/mooncake/run_master.sh
#
# ⚠️ 当前取值（2026-11-11 三改，对齐 4m 站）：
#    max-num-seqs 32 / max-num-batched-tokens 32768 / max-model-len 524288（模型原生上限）
#    同步开 --async-scheduling；additional_config 增补 enable_flashcomm1 / enable_cpu_binding /
#    multistream_overlap_shared_expert；不再设 VLLM_WORKER_MULTIPROC_METHOD（默认 fork）。
#    改回长上下文以外的旧口径见 archive/docs/DEPLOY.完整归档.md（旧 env.override.sh 冻结块归档在 archive/legacy/）
# =====================================================================
set -Eeuo pipefail

# ---- 清理上一轮残留（与 4m 站脚本同一口径）----
# 说明：
#   1. worker 的 proctitle 是 `VLLM::Worker_TPn_EPn`，必须带大写 VLLM 才杀得到；
#   2. 本段会杀掉容器内**所有** python —— 同机的 proxy 也是 python，所以 proxy
#      必须在本脚本之后起（顺序 master → decode → prefill → proxy）；
#   3. 用 set +e 包住：无残留时 grep/pkill 返回 1，会被 set -e/pipefail 当成失败。
set +e
ps aux | grep 'VLLM' | awk '{print $2}' | xargs -r kill -9
ps aux | grep 'vllm' | awk '{print $2}' | xargs -r kill -9
pkill -9 vllm
pkill -9 VLLM
pkill -9 python
pkill -9 ray
set -e
sleep 10

# ---- 网络：HCCL/Gloo/TP 全绑业务网卡（镜像 ENV 未设这些，缺省由框架自动选择）----
nic_name="enp39s0f3"
local_ip="10.222.46.181"
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"

export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_BUFFSIZE=400
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
# 对齐 4m 站（2026-11-11 三改）：不设 VLLM_WORKER_MULTIPROC_METHOD，走 vLLM 默认 fork。
# 若回退 spawn，需保留追加式 PYTHONPATH（spawn 会重新 import，覆盖式会让 worker 找不到 acl）。

# ---- mooncake 传输引擎 .so（追加，不覆盖镜像/CANN 的库路径）----
export LD_LIBRARY_PATH=/usr/local/python3.12.16/lib/python3.12/site-packages/mooncake:/usr/local/lib/:${LD_LIBRARY_PATH:-}

# 追加式，不能整体覆盖：镜像默认 PYTHONPATH 已含 acl 等 toolkit 路径，覆盖会让
# spawn worker 的 `from acl.rt import memcpy` 报 No module named 'acl'
# （2026-11-10 06:44 事故）。
export PYTHONPATH="${PYTHONPATH:-}:/workspace"

# ---- Mooncake KV 池（池化模式必配；master 固定 46.181）----
export MOONCAKE_CONFIG_PATH=/pd/mooncake/mooncake.json

mkdir -p /pd/logs/prefill 2>/dev/null || echo "[WARN] /pd/logs/prefill 建不出来（可能是悬空的 /pd/logs 软链），日志目录不可用" >&2

exec /usr/local/python3.12.16/bin/vllm serve /models/IQuest-Q1-W8A8C8-mxfp8 \
    --host 10.222.46.181 \
    --port 7100 \
    --served-model-name IQuest-Q1 \
    --tensor-parallel-size 8 \
    --max-model-len 524288 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 32 \
    --block-size 128 \
    --trust-remote-code \
    --reasoning-parser iquest_q1 \
    --tool-call-parser iquest_q1 \
    --enable-auto-tool-choice \
    --gpu-memory-utilization 0.90 \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_port":28010,"engine_id":"0","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":28010,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"mooncake","use_layerwise":false}}]}}' \
    --enable-prefix-caching \
    --enforce-eager \
    --additional-config '{"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1,"enable_flashcomm1":true,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/pd/logs/profile/prefill","torch_profiler_with_stack":false}' \
    --enable-expert-parallel \
    --async-scheduling \
    --quantization ascend
