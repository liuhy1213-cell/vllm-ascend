#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 PD 分离（1P1D 双机, mooncake 池化）— 角色容器创建
#
# 用法: bash docker_start.sh [p|d]     （不带参数时按本机业务 IP 自动识别角色）
#   p = 10.222.46.181 (new4) → pd-mooncacke-p    prefill + proxy + mooncake_master
#   d = 10.222.47.74  (new3) → pd-mooncacke-d    decode + MTP
#
# 与旧脚本 archive/legacy/docker_start.bare.sh 的差异（本脚本为现行口径）：
#   1. shm 1g → 64g（TP8 worker IPC 吃紧）
#   2. PID1 用 sleep infinity（旧脚本 `-itd bash`：attach 后 exit 会连容器一起退）
#   3. 追加挂载 /mnt/sfs_turbo 整盘、两套模型、部署目录 → /pd、引擎代码 → /workspace
#      （旧脚本只挂 /mnt，靠容器内手工建符号链接 + /pd 符号链接 farm）
#
# 注意 --rm: 容器 stop 即被删除；误停/重启 docker 后用本脚本重建。
# 若同名容器已存在（旧 bare 容器），脚本会提示；确认可删时用 FORCE=1 覆盖。
# =====================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 本站常量（全字面值，无 env.sh）----
SFS_ROOT="/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16"
SFS_REPO="${SFS_ROOT}/pd-mooncacke-20261110"          # 引擎代码 → /workspace
# ↑ /workspace 挂载的必须是 vllm-ascend 代码仓。SFS_REPO 默认指向主仓
#   ${SFS_ROOT}/vllm-ascend 的 releases/v0.30.0 linked worktree（同一提交 0b2829930 +
#   3 个未提交部署补丁：mooncake_connector.py / utils.py / worker/v2/attn_utils.py）。
#   若改成 ${SFS_ROOT}/vllm-ascend，需先把这 3 个补丁移植过去。
VLLM_IMAGE="quay.io/ascend/vllm-ascend:v0.30.0-a5-openeuler"
P_IP="10.222.46.181"                                   # new4  prefill/proxy/master
D_IP="10.222.47.74"                                    # new3  decode/MTP

local_ips="$(ip -4 -o addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)"
ROLE="${1:-}"
if [ -z "$ROLE" ]; then
    case " $local_ips " in
        *" $P_IP "*) ROLE=p ;;
        *" $D_IP "*) ROLE=d ;;
        *) echo "[ERROR] 本机 IP 不在 {P=$P_IP, D=$D_IP} 内，请显式指定: bash docker_start.sh p|d" >&2
           exit 1 ;;
    esac
fi

case "$ROLE" in
    p) NAME="pd-mooncacke-p" ;;
    d) NAME="pd-mooncacke-d" ;;
    *) echo "[ERROR] 未知角色 '$ROLE'，用法: bash docker_start.sh [p|d]" >&2; exit 1 ;;
esac

[ -d "$SFS_REPO/vllm_ascend" ] || { echo "[ERROR] 引擎代码树不存在: $SFS_REPO" >&2; exit 1; }
[ -d /mnt/sfs_turbo/models/IQuest-Q1-W8A8C8-mxfp8 ] || { echo "[ERROR] 量化权重不存在" >&2; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    if [ "${FORCE:-0}" = "1" ]; then
        echo "[INFO] FORCE=1 → 删除已存在的容器 $NAME"
        docker rm -f "$NAME" >/dev/null
    else
        echo "[INFO] 容器 $NAME 已存在，跳过创建"
        echo "       如需按本脚本口径重建（旧 bare 容器 shm=1g、无 /pd）: FORCE=1 bash docker_start.sh $ROLE"
        exit 0
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
  -v /mnt/sfs_turbo/models/IQuest-Q1:/models/IQuest-Q1:ro \
  -v /mnt/sfs_turbo/models/IQuest-Q1-W8A8C8-mxfp8:/models/IQuest-Q1-W8A8C8-mxfp8:ro \
  -v "$SFS_REPO":/workspace \
  -v "$DEPLOY_DIR":/pd \
  -w /workspace \
  -itd "$VLLM_IMAGE" sleep infinity

echo "[INFO] 容器 $NAME 已创建（角色 $ROLE，idle）"
echo "       /pd        ← $DEPLOY_DIR"
echo "       /workspace ← $SFS_REPO"
echo "       /models    ← IQuest-Q1(mtp) + IQuest-Q1-W8A8C8-mxfp8"
