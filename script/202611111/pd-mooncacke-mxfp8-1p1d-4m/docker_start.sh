#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 PD 分离（4 机 DP2×TP8, mxfp8 KV）— 角色容器创建
#
# 用法: bash docker_start.sh [p|d]      （不带参数时按本机业务 IP 自动识别）
#   p → iquest-mxfp8-p   prefill: 10.222.43.66 / 10.222.44.142
#   d → iquest-mxfp8-d   decode : 10.222.42.240 / 10.222.41.88
#
# 口径（与 deploy/pd-mooncacke-20261109 一致）：
#   --rm、--net=host、--shm-size=64g、8 张 davinci、PID1 = sleep infinity
#   挂载：/mnt/sfs_turbo 整盘、两套模型、引擎代码 → /workspace、本目录 → /pd
#
# ⚠️ 与已有容器不重名（iquest-pd-mooncacke-p/d、pd-mooncacke-p/d、iquest-pd-p78 等
#    都还在这四台机器上），避免互相干扰。
# =====================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SFS_ROOT="/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16"
SFS_REPO="${SFS_ROOT}/pd-mooncacke-20261110"          # 引擎代码（releases/v0.30.0，含 external_online_dp 与 MooncakeConnectorV2）
MODEL_QUANT="/mnt/sfs_turbo/models/IQuest-Q1-W8A8C8-mxfp8"
MODEL_BF16="/mnt/sfs_turbo/models/IQuest-Q1"          # MTP 草稿头在它下面（mtp/）
VLLM_IMAGE="quay.io/ascend/vllm-ascend:v0.30.0-a5-openeuler"

P0_IP="10.222.43.66";  P1_IP="10.222.44.142"
D0_IP="10.222.42.240"; D1_IP="10.222.41.88"

local_ips="$(ip -4 -o addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)"
ROLE="${1:-}"
if [ -z "$ROLE" ]; then
    case " $local_ips " in
        *" $P0_IP "*|*" $P1_IP "*) ROLE=p ;;
        *" $D0_IP "*|*" $D1_IP "*) ROLE=d ;;
        *) echo "[ERROR] 本机 IP 不在本站 4 台之内，请显式指定: bash docker_start.sh p|d" >&2; exit 1 ;;
    esac
fi

case "$ROLE" in
    p) NAME="iquest-mxfp8-p" ;;
    d) NAME="iquest-mxfp8-d" ;;
    *) echo "[ERROR] 未知角色 '$ROLE'，用法: bash docker_start.sh [p|d]" >&2; exit 1 ;;
esac

[ -d "$SFS_REPO/vllm_ascend" ] || { echo "[ERROR] 引擎代码树不存在: $SFS_REPO" >&2; exit 1; }
[ -d "$MODEL_QUANT" ] || { echo "[ERROR] 量化权重不存在: $MODEL_QUANT" >&2; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    if [ "${FORCE:-0}" = "1" ]; then
        echo "[INFO] FORCE=1 → 删除已存在的容器 $NAME"; docker rm -f "$NAME" >/dev/null
    else
        echo "[INFO] 容器 $NAME 已存在，跳过（重建: FORCE=1 bash docker_start.sh $ROLE）"; exit 0
    fi
fi

docker run --rm \
  --name "$NAME" \
  --net=host \
  --shm-size=64g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  --device /dev/ummu \
  --device /dev/uburma \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
  -v /etc/hixlep/:/etc/hixlep/ \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/bin/urma_admin:/usr/bin/urma_admin \
  -v /lib/route.conf:/lib/route.conf \
  -v /mnt/sfs_turbo:/mnt/sfs_turbo \
  -v "$MODEL_BF16:/models/IQuest-Q1:ro" \
  -v "$MODEL_QUANT:/models/IQuest-Q1-W8A8C8-mxfp8:ro" \
  -v "$SFS_REPO":/workspace \
  -v "$DEPLOY_DIR":/pd \
  -w /workspace \
  -itd "$VLLM_IMAGE" sleep infinity

echo "[INFO] 容器 $NAME 已创建（角色 $ROLE，idle）"
echo "       /pd        ← $DEPLOY_DIR"
echo "       /workspace ← $SFS_REPO"
echo "       /models    ← IQuest-Q1(mtp) + IQuest-Q1-W8A8C8-mxfp8"
