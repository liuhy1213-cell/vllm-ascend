#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 PD 分离 — D0 (new8 10.222.41.88) decode 启动【全字面值，无变量】
#
# 容器: iquest-pd-d78（部署目录挂载 /pd，仓库挂载 /workspace）
# 用法: 在 D0 上执行  docker exec -d iquest-pd-d78 bash /pd/decode/launch_static.sh
# 日志: 直接打屏不落盘（2026-11-10 按要求改回；用 docker exec -d 起则输出被丢弃）
#
# ⚠️ 启动顺序: 池化模式下 master(P0) → decode → prefill → proxy
# ⚠️ 当前冻结的是 4.4/4.5 纯P/纯D 配比测试值（env.override.sh 临时块）：
#    max-num-seqs 6 / max-num-batched-tokens 192
#    恢复生产值请改回: --max-num-seqs 256 --max-num-batched-tokens 1024
# =====================================================================
set -Eeuo pipefail

# ---- CANN / ATB 运行时（镜像内实际版本 cann-9.2.1）----
# CANN/ATB 的 set_env.sh 引用了未加保护的 $ZSH_VERSION 等变量，会触发 nounset
# 报错（"ZSH_VERSION: unbound variable"），source 前后临时关开 -u。
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.2.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/cann-9.2.1/tools/hcc/aarch64-target-linux-gnu/include/c++/14.3.0:/usr/local/Ascend/cann-9.2.1/tools/hcc/aarch64-target-linux-gnu/include/c++/14.3.0/aarch64-target-linux-gnu:/usr/local/Ascend/cann-9.2.1/tools/hcc/aarch64-target-linux-gnu/include/c++/14.3.0/backward

# ---- mooncake 传输引擎 .so（须在 CANN 之后追加，不能覆盖其路径）----
export LD_LIBRARY_PATH=/usr/local/python3.12.16/lib/python3.12/site-packages/mooncake:/usr/local/lib/:${LD_LIBRARY_PATH:-}

# ---- 引擎环境（原 decode/launch_engine.sh + container.sh 容器级 env，字面值）----
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
# ascend-toolkit/latest/python/site-packages（acl/acl.so，camem.py 依赖）注入
# PYTHONPATH，覆盖会导致 spawn worker `from acl.rt import memcpy` 报
# No module named 'acl'（同 prefill/launch_static.sh 2026-11-10 修复）。
export PYTHONPATH="${PYTHONPATH:-}:/workspace"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SOC_VERSION=ascend950dt_9582
export TASK_QUEUE_ENABLE=1
export ATB_CXX_ABI=1
export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
export ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0
export ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0
export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1
export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1
export ATB_COMPARE_TILING_EVERY_KERNEL=0
export ATB_MATMUL_SHUFFLE_K_ENABLE=1
export LCCL_DETERMINISTIC=0
export LCCL_PARALLEL=0

# ---- Mooncake KV 池（池化模式必配；master 固定 P0=10.222.42.240）----
export MOONCAKE_CONFIG_PATH=/pd/mooncake/mooncake.json
export MOONCAKE_MASTER=10.222.42.240:50088
export MOONCAKE_GLOBAL_SEGMENT_SIZE=24GB

mkdir -p /pd/logs/decode

exec /usr/local/python3.12.16/bin/vllm serve /models/IQuest-Q1-W8A8C8-mxfp8 \
    --host 10.222.41.88 \
    --port 7100 \
    --served-model-name IQuest-Q1 \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --max-num-batched-tokens 192 \
    --max-num-seqs 6 \
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
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024]}' \
    --async-scheduling \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1}' \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/pd/logs/profile/decode","torch_profiler_with_stack":false}' \
    --kv-cache-memory 34359738368 \
    --enable-expert-parallel \
    --quantization ascend
