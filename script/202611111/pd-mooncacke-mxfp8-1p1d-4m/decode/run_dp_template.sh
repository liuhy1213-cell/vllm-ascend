#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 PD 分离（4 机 DP2×TP8, mxfp8 KV）— decode 单 DP rank 引擎模板
#
# 由同目录 launch_online_dp.py 拉起（硬编码 ./run_dp_template.sh）。
# 位置参数契约：$1 可见卡  $2 端口  $3 dp_size  $4 dp_rank  $5 dp_address  $6 dp_rpc_port  $7 tp_size
#
# 需由 d0.sh / d1.sh 导出：MODEL_PATH、LOCAL_IP、NIC_NAME
# =====================================================================
set -Eeuo pipefail

: "${MODEL_PATH:?请由 d0.sh/d1.sh 导出 MODEL_PATH}"
: "${LOCAL_IP:?请由 d0.sh/d1.sh 导出本机业务 IP}"
: "${NIC_NAME:?请由 d0.sh/d1.sh 导出业务网卡名}"

DEPLOY_DIR="/pd"
LOG_DIR="${LOG_DIR:-${DEPLOY_DIR}/logs/decode}"

unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY || true
unset HCCL_INTRA_ROCE_ENABLE || true

# ---- 网络 ----
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export HCCL_ALGO=level0:fullmesh

# ---- 超时/缓冲（D 侧 buffsize 比 P 小，按参考值）----
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=1800
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_BUFFSIZE=900
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export VLLM_ASCEND_DP_ALLREDUCE_ON_NPU=1

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
export VLLM_USE_V2_MODEL_RUNNER=1   # mxfp8(C8) KV 的 scatter/scale 处理在 V2 runner 里（v2/attn_utils.py）
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

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
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser iquest_q1 \
  --tool-call-parser iquest_q1 \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.94 \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_port":29010,"engine_id":"1","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":29010,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":8},"decode":{"dp_size":2,"tp_size":8}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"mooncake","use_layerwise":false}}]}}' \
  --speculative-config '{"method":"eagle","model":"/models/IQuest-Q1/mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024]}' \
  --async-scheduling \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/pd/logs/profile/decode","torch_profiler_with_stack":false}' \
  --kv-cache-memory 34359738368 \
  --enable-expert-parallel \
  --quantization ascend
