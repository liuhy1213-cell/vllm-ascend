#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 TP16 混部（2 机，非 PD 分离）— 角色容器创建
#
# 用法: bash docker_start.sh [n1|n2]     （不带参数时按本机业务 IP 自动识别角色）
#   n1 = 10.222.39.253 (sjsbg-00004) → q1-tp16-mix-node1   head: node-rank 0, API :8005
#   n2 = 10.222.34.2   (sjsbg-00002) → q1-tp16-mix-node2   worker: node-rank 1, --headless
#
# 与 PD 栈（pd-mooncacke-*）的区别：不挂 mooncake 配置、不建 /pd/mooncake，
# 因为混部没有 KV 传输；mooncake 只在那套 AscendStoreConnector 池化变体里才需要。
#
# 注意 --rm: 容器 stop 即被删除；误停/重启 docker 后用本脚本重建。
# 同名容器已存在时脚本只提示；确认可删时用 FORCE=1 覆盖。
# =====================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 本站常量（全字面值，无 env.sh）----
N1_IP="10.222.39.253"                                  # head  (hyperinstance-sjsbg-00004)
N2_IP="10.222.34.2"                                     # worker(hyperinstance-sjsbg-00002)
VLLM_IMAGE="quay.io/ascend/vllm-ascend:v0.30.0-a5-openeuler"

# 引擎代码 → /workspace。必须是 vllm-ascend 代码树（含 custom AR 的
# VLLM_ASCEND_ENABLE_CUSTOM_AR / VLLM_ASCEND_ENABLE_IQUEST_Q1_FUSION）。
#   默认: 本工作区 deploy/vllm-iquest-pqing-bf16（分支 iquest-q1-npu，e84bb2676）
#   已在跑的那套 TP16 混部用的是: /mnt/sfs_turbo/pqing/vllm-iquest-npu-kernel-parsers
#                                   （分支 iquest-q1-npu-kernel-parsers，f5cb1c6e9）
#   要一比一复现现场就把下面这行换成后者。
SFS_REPO="/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/vllm-iquest-pqing-bf16"
# custom allreduce 插件 → /workspace/custom-ar（IQUEST_CUSTOM_AR=1 时才加载）
CUSTOM_AR_REPO="/mnt/sfs_turbo/pqing/ascend-custom-allreduce"   # main, a61f7c7
MODEL_DIR="/mnt/sfs_turbo/models/IQuest-Q1"                     # bf16 + mtp/

local_ips="$(ip -4 -o addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)"
ROLE="${1:-}"
if [ -z "$ROLE" ]; then
    case " $local_ips " in
        *" $N1_IP "*) ROLE=n1 ;;
        *" $N2_IP "*) ROLE=n2 ;;
        *) echo "[ERROR] 本机 IP 不在 {n1=$N1_IP, n2=$N2_IP} 内，请显式指定: bash docker_start.sh n1|n2" >&2
           exit 1 ;;
    esac
fi

case "$ROLE" in
    n1) NAME="q1-tp16-mix-node1" ;;
    n2) NAME="q1-tp16-mix-node2" ;;
    *) echo "[ERROR] 未知角色 '$ROLE'，用法: bash docker_start.sh [n1|n2]" >&2; exit 1 ;;
esac

[ -d "$SFS_REPO/vllm_ascend" ]     || { echo "[ERROR] 引擎代码树不存在: $SFS_REPO" >&2; exit 1; }
[ -d "$CUSTOM_AR_REPO" ]           || { echo "[ERROR] custom AR 树不存在: $CUSTOM_AR_REPO" >&2; exit 1; }
[ -d "$MODEL_DIR/mtp" ]            || { echo "[ERROR] MTP 草稿目录不存在: $MODEL_DIR/mtp" >&2; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    if [ "${FORCE:-0}" = "1" ]; then
        echo "[INFO] FORCE=1 → 删除已存在的容器 $NAME"
        docker rm -f "$NAME" >/dev/null
    else
        echo "[INFO] 容器 $NAME 已存在，跳过创建"
        echo "       如需按本脚本口径重建: FORCE=1 bash docker_start.sh $ROLE"
        exit 0
    fi
fi

# 注意：不要加 -e PYTHONPATH=/workspace（容器级覆盖会冲掉镜像注入的 acl 路径，
# spawn worker 会报 No module named 'acl'）；PYTHONPATH 在 serve/*.sh 里追加。
docker run --rm \
  --name "$NAME" \
  --net=host \
  --privileged \
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
  -v /etc/cloud/topo:/etc/cloud/topo:ro \
  -v /mnt/sfs_turbo:/mnt/sfs_turbo \
  -v "$MODEL_DIR":/models/IQuest-Q1:ro \
  -v "$SFS_REPO":/workspace \
  -v "$CUSTOM_AR_REPO":/workspace/custom-ar:ro \
  -v "$DEPLOY_DIR":/pd \
  -w /workspace \
  -itd "$VLLM_IMAGE" sleep infinity

echo "[INFO] 容器 $NAME 已创建（角色 $ROLE，idle）"
echo "       /pd         ← $DEPLOY_DIR"
echo "       /workspace  ← $SFS_REPO"
echo "       /workspace/custom-ar ← $CUSTOM_AR_REPO"
echo "       /models/IQuest-Q1    ← $MODEL_DIR（含 mtp/）"