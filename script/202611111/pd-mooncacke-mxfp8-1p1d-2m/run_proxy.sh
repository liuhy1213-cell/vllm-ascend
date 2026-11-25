#!/usr/bin/env bash
# =====================================================================
# PD 负载均衡 proxy（仅 46.181 容器内运行）: 0.0.0.0:1999【全字面值，无变量】
#
# 用法: 在 46.181 上执行  docker exec -d pd-mooncacke-p bash /pd/run_proxy.sh
# 路由: prefill 10.222.46.181:7100  →  decode 10.222.47.74:7100
# 日志: /pd/logs/proxy/proxy.log（/pd 为 SFS 挂载时即落在 SFS，可留存）
#
# ⚠️ 必须最后起：prefill/decode 脚本开头会 pkill -9 python，会把 proxy 一起杀掉
# =====================================================================
set -Eeuo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROXY_SCRIPT=/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
[ -f "$PROXY_SCRIPT" ] || PROXY_SCRIPT="${DEPLOY_DIR}/load_balance_proxy_server_example.py"
[ -f "$PROXY_SCRIPT" ] || { echo "[ERROR] 找不到 proxy 脚本" >&2; exit 1; }

PY_BIN=/usr/local/python3.12.16/bin/python3
[ -x "$PY_BIN" ] || PY_BIN=python3

LOG_DIR="${DEPLOY_DIR}/logs/proxy"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/proxy.log"
echo "[INFO] proxy 启动: $(date '+%F %T') 0.0.0.0:1999 日志=${LOG_FILE}"
echo "[INFO]   prefill: 10.222.46.181:7100"
echo "[INFO]   decode : 10.222.47.74:7100"

exec "$PY_BIN" "$PROXY_SCRIPT" \
  --host 0.0.0.0 \
  --port 1999 \
  --prefiller-hosts 10.222.46.181 \
  --prefiller-ports 7100 \
  --decoder-hosts 10.222.47.74 \
  --decoder-ports 7100 \
  > "$LOG_FILE" 2>&1
