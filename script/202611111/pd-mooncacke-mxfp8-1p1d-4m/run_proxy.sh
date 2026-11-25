#!/usr/bin/env bash
# =====================================================================
# PD 负载均衡 proxy（4 后端）: 0.0.0.0:8999【全字面值，无变量】
#
# 用法: P0 (43.66) 上执行  docker exec -d iquest-mxfp8-p bash /pd/run_proxy.sh
#
#   后端: prefill 10.222.43.66:8000  10.222.44.142:8000
#         decode  10.222.42.240:8000  10.222.41.88:8000
#
# ⚠️ 必须最后起（引擎脚本开头的清理块会 pkill -9 python，proxy 也是 python）。
# ⚠️ 参考文档里 proxy 用 8000 端口且跑在独立主机；本站没有第 5 台机器，
#    所以 proxy 落在 P0 上、端口改成 8999，避免与 P0 的引擎 8000 冲突。
# 日志: /pd/logs/proxy/proxy.log（/pd 为 SFS 挂载时即落盘可留存）
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
echo "[INFO] proxy 启动: $(date '+%F %T') 0.0.0.0:8999 日志=${LOG_FILE}"
echo "[INFO]   prefill: 10.222.43.66:8000 10.222.44.142:8000"
echo "[INFO]   decode : 10.222.42.240:8000 10.222.41.88:8000"

exec "$PY_BIN" "$PROXY_SCRIPT" \
  --host 0.0.0.0 \
  --port 8999 \
  --prefiller-hosts 10.222.43.66 10.222.44.142 \
  --prefiller-ports 8000 8000 \
  --decoder-hosts 10.222.42.240 10.222.41.88 \
  --decoder-ports 8000 8000 \
  > "$LOG_FILE" 2>&1
