#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 PD 分离（4 机 DP2×TP8, mxfp8 KV）— prefill 单 DP rank 引擎模板
#
# 本文件不由人手工执行，由同目录的 launch_online_dp.py 拉起（它硬编码
# ./run_dp_template.sh，所以必须在 prefill/ 目录下启动 launcher）。
# 位置参数契约（launch_online_dp.py 传入）：
#   $1 ASCEND_RT_VISIBLE_DEVICES   $2 vLLM 端口        $3 全局 dp_size
#   $4 本 rank 的 dp_rank          $5 dp_address(P0_IP)$6 dp_rpc_port  $7 tp_size
#
# 需由 p0.sh / p1.sh 导出：MODEL_PATH、LOCAL_IP（本机业务 IP）、NIC_NAME
# =====================================================================
set -Eeuo pipefail

: "${MODEL_PATH:?请由 p0.sh/p1.sh 导出 MODEL_PATH}"
: "${LOCAL_IP:?请由 p0.sh/p1.sh 导出本机业务 IP}"
: "${NIC_NAME:?请由 p0.sh/p1.sh 导出业务网卡名}"

DEPLOY_DIR="/pd"
LOG_DIR="${LOG_DIR:-${DEPLOY_DIR}/logs/prefill}"

# ---- 干净环境：去掉代理与可能干扰 HCCL 组网方式的变量 ----
unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY || true
unset HCCL_INTRA_ROCE_ENABLE || true

# ---- 网络：HCCL/Gloo/TP 全绑业务网卡 ----
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export HCCL_ALGO=level0:fullmesh

# ---- 超时/缓冲（DP+EP 跨机，给足余量）----
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=1800
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_BUFFSIZE=1200
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

# ---- 运行时 ----
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES="$1"
mkdir -p "$LOG_DIR"
export ASCEND_PROCESS_LOG_PATH="$LOG_DIR"
export VLLM_USE_FASTOKENS=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V2_MODEL_RUNNER=1   # 与 2m 站一致
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

# 参考模板里有、本站尚未验证的两项（需要时取消注释再测）：
# export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
# export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

# 追加式注入引擎代码树（/workspace = pd-mooncacke-20261110）。
# 不能整体覆盖：镜像默认 PYTHONPATH 里有 acl 等 toolkit 路径，覆盖会导致
# spawn worker 的 `from acl.rt import memcpy` 报 No module named acl。
export PYTHONPATH="${PYTHONPATH:-}:/workspace"

# ---- Mooncake 池（连接器从这两个环境变量取 master 地址）----
export MOONCAKE_CONFIG_PATH=/pd/mooncake/mooncake.json
export MOONCAKE_MASTER=10.222.43.66:50090
export MOONCAKE_GLOBAL_SEGMENT_SIZE=24GB

exec /usr/local/python3.12.16/bin/vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$2" \
  --data-parallel-size "$3" \
  --data-parallel-rank "$4" \
  --data-parallel-address "$5" \
  --data-parallel-rpc-port "$6" \
  --served-model-name IQuest-Q1 \
  --tensor-parallel-size "$7" \
  --max-model-len 524288 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 16 \
  --block-size 128 \
  --trust-remote-code \
  --reasoning-parser iquest_q1 \
  --tool-call-parser iquest_q1 \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.90 \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_port":28010,"engine_id":"0","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":28010,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":8},"decode":{"dp_size":2,"tp_size":8}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"mooncake","use_layerwise":false}}]}}' \
  --enable-prefix-caching \
  --enforce-eager \
  --additional-config '{"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/pd/logs/profile/prefill","torch_profiler_with_stack":false}' \
  --enable-expert-parallel \
  --quantization ascend
