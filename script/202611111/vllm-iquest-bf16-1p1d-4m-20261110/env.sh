#!/usr/bin/env bash
# =====================================================================
# 本包最小 env.sh（2026-11-11 补）
#
# 原包未附带 env.sh，README §3 让 cp 的 ../vllm-iquest-bf16/env.sh 已不存在。
# docker_start.sh 实际只消费下面三个变量 + get_node()，故按本包口径写死：
#   引擎代码 = vllm-ascend-prof（README §3 指定，带 iquest_q1 / MultiConnector 改动）
# 建容器时仍建议显式传角色：bash docker_start.sh p|d
# =====================================================================

VLLM_IMAGE="quay.io/ascend/vllm-ascend:v0.30.0-a5-openeuler"

SFS_ROOT="/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16"
SFS_REPO="${SFS_ROOT}/vllm-ascend-prof"

get_node() {
    case "$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -m1 '^10\.222\.')" in
        10.222.43.66|10.222.44.142) printf 'p' ;;
        10.222.42.240|10.222.41.88) printf 'd' ;;
        *)                          printf 'unknown' ;;
    esac
}