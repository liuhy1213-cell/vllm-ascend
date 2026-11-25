#!/usr/bin/env bash
# =====================================================================
# mooncake_master 拉起（仅 46.181 = P 侧容器内；池化模式的前置条件）
# 用法: docker exec -d pd-mooncacke-p bash /pd/mooncake/run_master.sh
# 探活: ss -ltnp | grep 50088        （引擎初始化即连 50088，必须先起）
#
# 2026-11-10 整改：日志改为绝对路径 /pd/logs/mooncake/master.log。
#   旧写法 `>> mooncacke.log` 是相对路径，随 docker exec 的 CWD 漂移
#   （-w /workspace 时会写到引擎代码树里），且文件名少一个 c。
# =====================================================================
mkdir -p /pd/logs/mooncake
nohup mooncake_master \
    --port 50088 \
    --eviction_high_watermark_ratio 0.9 \
    --eviction_ratio 0.1 \
    --default_kv_lease_ttl 11000 \
    >> /pd/logs/mooncake/master.log 2>&1 &
