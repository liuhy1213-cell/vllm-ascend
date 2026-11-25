#!/usr/bin/env bash
# =====================================================================
# PD proxy（仅 P0 容器内运行，docker exec -d 调用）: 0.0.0.0:$PROXY_PORT
# 镜像自带 vllm-ascend examples 的 load_balance_proxy（disaggregated_prefill_v1）
# =====================================================================
set -Eeuo pipefail
# 本脚本位于部署目录顶层（/pd/run_proxy.sh），不像 prefill/decode 那样在
# 子目录里，所以这里不能加 /..（否则 DEPLOY_DIR=/ 、source /env.sh 失败）。
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"

PROXY_SCRIPT=/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
[ -f "$PROXY_SCRIPT" ] || PROXY_SCRIPT="${DEPLOY_DIR}/load_balance_proxy_server_example.py"
[ -f "$PROXY_SCRIPT" ] || { echo "[ERROR] 找不到 proxy 脚本" >&2; exit 1; }

PY_BIN=/usr/local/python3.12.16/bin/python3
[ -x "$PY_BIN" ] || PY_BIN=python3

LOG_DIR="${LOG_ROOT}/proxy"; mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/proxy.log"
echo "[INFO] proxy 启动: $(date '+%F %T') 0.0.0.0:${PROXY_PORT} 日志=${LOG_FILE}"
echo "[INFO]   prefill: ${P0_IP}:${PREFILL_API_PORT}"
echo "[INFO]   decode : ${D0_IP}:${DECODE_API_PORT}"

exec "$PY_BIN" "$PROXY_SCRIPT" \
  --host 0.0.0.0 \
  --port "$PROXY_PORT" \
  --prefiller-hosts "$P0_IP" \
  --prefiller-ports "$PREFILL_API_PORT" \
  --decoder-hosts "$D0_IP" \
  --decoder-ports "$DECODE_API_PORT" \
  > "$LOG_FILE" 2>&1
