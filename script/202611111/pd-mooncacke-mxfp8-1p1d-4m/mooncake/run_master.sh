#!/usr/bin/env bash
# =====================================================================
# mooncake_master 拉起（仅 P0 = 10.222.43.66；PD 池化模式的前置条件）
# 用法: docker exec -d iquest-mxfp8-p bash /pd/mooncake/run_master.sh
# 探活: ss -ltnp | grep 50090     （引擎初始化即连 50090，必须先起）
#
# 注意 43.66 上原本还有别的站的 master 占着 50088（容器 iq-own），本包改用 50090 避开，
# 拉起前先确认端口空闲，否则本脚本的实例会 bind 失败：
#   ss -ltnp | grep 50090
# =====================================================================
mkdir -p /pd/logs/mooncake
nohup mooncake_master \
    --rpc_port 50090 \
    --metrics_port 9004 \
    --eviction_high_watermark_ratio 0.9 \
    --eviction_ratio 0.1 \
    --default_kv_lease_ttl 11000 \
    >> /pd/logs/mooncake/master.log 2>&1 &
