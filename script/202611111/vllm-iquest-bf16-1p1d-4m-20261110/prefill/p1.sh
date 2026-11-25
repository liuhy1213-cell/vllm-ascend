#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 PD 分离 — P0 (new2 10.222.42.240) prefill 启动【全字面值，无变量】
#
# 容器: iquest-pd-p78（部署目录挂载 /pd，仓库挂载 /workspace）
# 用法: 在 P0 上执行  docker exec -d iquest-pd-p78 bash /pd/prefill/launch_static.sh
# 日志: 直接打屏不落盘（2026-11-10 按要求改回；用 docker exec -d 起则输出被丢弃）
#
# ⚠️ 池化模式（ENABLE_KV_POOL=1）：必须先在 P0 起 mooncake_master 再起本引擎
#    （bash /pd/start.sh start master，或 docker exec -d iquest-pd-p78 bash /pd/mooncake/run_master.sh）
#
# ⚠️ 当前冻结的是 4.4/4.5 纯P/纯D 配比测试值（env.override.sh 临时块）：
#    max-num-seqs 8 / max-num-batched-tokens 32768 / max-model-len 112024
#    恢复生产值请改回: --max-num-seqs 32 --max-num-batched-tokens 8192 --max-model-len 131072
# =====================================================================
ps aux | grep 'VLLM' | awk '{print $2}' | xargs -r kill -9
ps aux | grep 'vllm' | awk '{print $2}' | xargs -r kill -9
pkill -9 vllm
pkill -9 VLLM
pkill -9 python
pkill -9 ray

sleep 10
nic_name="enp39s0f3"
local_ip="10.222.44.142"

export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"


# ---- mooncake 传输引擎 .so（须在 CANN 之后追加，不能覆盖其路径）----
export LD_LIBRARY_PATH=/usr/local/python3.12.16/lib/python3.12/site-packages/mooncake:/usr/local/lib/:${LD_LIBRARY_PATH:-}

# ---- 引擎环境（原 prefill/launch_engine.sh + container.sh 容器级 env，字面值）----
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
# PYTHONPATH 追加 /workspace 而不能整体覆盖：set_env.sh 已把

export PYTHONPATH="${PYTHONPATH:-}:/workspace"

# ---- Mooncake KV 池（池化模式必配；master 固定 P0）----
export MOONCAKE_CONFIG_PATH=/pd/mooncake/mooncake.json

mkdir -p /pd/logs/prefill

exec /usr/local/python3.12.16/bin/vllm serve /mnt/sfs_turbo/models/IQuest-Q1 \
    --host 10.222.44.142 \
    --port 7100 \
    --served-model-name IQuest-Q1 \
    --tensor-parallel-size 16 \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank 1 \
    --headless \
    --master-addr 10.222.43.66 --master-port 29563 \
    --max-model-len 524288 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 8 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --reasoning-parser iquest_q1 \
    --tool-call-parser iquest_q1 \
    --enable-auto-tool-choice \
    --gpu-memory-utilization 0.94 \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_port":28010,"engine_id":"0","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":28010,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":16},"decode":{"dp_size":1,"tp_size":16}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"mooncake","use_layerwise":false}}]}}' \
    --enable-prefix-caching \
    --enforce-eager \
    --additional-config '{"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1}' \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/pd/logs/profile/prefill","torch_profiler_with_stack":false}' \
    --enable-expert-parallel \


