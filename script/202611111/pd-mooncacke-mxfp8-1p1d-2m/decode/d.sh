#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 PD 分离 — decode（本 2m 站 = 47.74 new3）启动【全字面值，无变量】
#
# 容器: pd-mooncacke-d（部署目录挂载 /pd，引擎代码挂 /workspace）
# 用法: 在 47.74 上执行  docker exec -d pd-mooncacke-d bash /pd/decode/d.sh
# 日志: 直接打屏不落盘；用 docker exec -d 起则输出被丢弃（要看输出就去掉 -d 前台 exec）
#
# ⚠️ 启动顺序: master(46.181) → 本脚本 → prefill(46.181) → proxy
#    本脚本的 MTP 草稿模型走 /models/IQuest-Q1/mtp，容器需挂该项（见 README §3）
#
# ⚠️ 当前取值（2026-11-11 三改，对齐 4m 站）：
#    max-num-batched-tokens 256 / max-num-seqs 64 / max-model-len 524288（模型原生上限）
#    不再 pin --kv-cache-memory，KV 容量由 --gpu-memory-utilization 0.94 自动推导
#    （4m 站 decode 同为 256/64；旧 env.override.sh 冻结块归档在 archive/legacy/）
# =====================================================================
set -Eeuo pipefail

# ---- 清理上一轮残留（与 4m 站脚本同一口径）----
# 说明：
#   1. worker 的 proctitle 是 `VLLM::Worker_TPn_EPn`，必须带大写 VLLM 才杀得到；
#   2. 本段会杀掉容器内**所有** python —— 同机的 proxy 也是 python，所以 proxy
#      必须在 prefill 之后起（顺序 master → decode → prefill → proxy）；
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
local_ip="10.222.47.74"
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"

# ---- 引擎环境（只保留镜像 ENV 没有、或需要覆盖的项）----
# 以下值与镜像 ENV 逐项比对过（2026-11-10），容器里原本就是这些值，故不再重复导出：
#   ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME / SOC_VERSION=ascend950dt_9582 /
#   TASK_QUEUE_ENABLE=1 / ATB_CXX_ABI=1 / ATB_STREAM_SYNC_EVERY_* /
#   ATB_OPSRUNNER_KERNEL_CACHE_* / ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE /
#   ATB_COMPARE_TILING_EVERY_KERNEL / ATB_MATMUL_SHUFFLE_K_ENABLE /
#   LCCL_DETERMINISTIC / LCCL_PARALLEL
# CANN/ATB 的 set_env.sh 也不再 source：镜像默认 PYTHONPATH/LD_LIBRARY_PATH 已含
#   ascend-toolkit python 站点包（acl.so）与 cann-9.2.1 库，import acl 开箱可用。
#   若将来编译 bishengir/静态 kernel 报找不到编译器，再临时 source
#   ascend-toolkit/set_env.sh 与 nnal/atb/set_env.sh --cxx_abi=1。
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
# vLLM 默认 fork；本站沿用 spawn（spawn 会重新 import，故 PYTHONPATH 必须追加式）
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ---- mooncake 传输引擎 .so（追加，不覆盖镜像/CANN 的库路径）----
export LD_LIBRARY_PATH=/usr/local/python3.12.16/lib/python3.12/site-packages/mooncake:/usr/local/lib/:${LD_LIBRARY_PATH:-}

# 追加式，不能整体覆盖：镜像默认 PYTHONPATH 已含 acl 等 toolkit 路径，覆盖会让
# spawn worker 的 `from acl.rt import memcpy` 报 No module named 'acl'
# （2026-11-10 06:44 事故）。
export PYTHONPATH="${PYTHONPATH:-}:/workspace"

# ---- Mooncake KV 池（池化模式必配；master 固定 46.181（decode 只是客户端））----
export MOONCAKE_CONFIG_PATH=/pd/mooncake/mooncake.json

mkdir -p /pd/logs/decode 2>/dev/null || echo "[WARN] /pd/logs/decode 建不出来（可能是悬空的 /pd/logs 软链），日志目录不可用" >&2

exec /usr/local/python3.12.16/bin/vllm serve /models/IQuest-Q1-W8A8C8-mxfp8 \
    --host 10.222.47.74 \
    --port 7100 \
    --served-model-name IQuest-Q1 \
    --tensor-parallel-size 8 \
    --max-model-len 524288 \
    --max-num-batched-tokens 256 \
    --max-num-seqs 64 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --reasoning-parser iquest_q1 \
    --tool-call-parser iquest_q1 \
    --enable-auto-tool-choice \
    --gpu-memory-utilization 0.94 \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_port":29010,"engine_id":"1","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":29010,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"mooncake","use_layerwise":false}}]}}' \
    --speculative-config '{"method":"eagle","model":"/models/IQuest-Q1/mtp","num_speculative_tokens":3}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128]}' \
    --async-scheduling \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1}' \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/pd/logs/profile/decode","torch_profiler_with_stack":false}' \
    --enable-expert-parallel \
    --quantization ascend
