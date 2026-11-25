#!/usr/bin/env bash
# =====================================================================
# 新建手动启动用的 PD 角色容器（参考官方 vllm-ascend 启动脚本，2026-11-09）
# 与官方参考脚本的差异（其余挂载/设备逐项一致）：
#   1. 容器名: iquest-pd-mooncacke-p / iquest-pd-mooncacke-d
#   2. 追加挂载: /mnt/sfs_turbo 整盘、两套模型目录、仓库 → /workspace、部署目录 → /pd
#   3. 镜像: 节点上只有 v0.30.0-a5-openeuler（官方脚本的 v0.30.0-a5 标签两台都没有），
#      且 launch_static.sh 的字面路径按该镜像编写
#   4. PID1 用 sleep infinity（官方裸 bash：attach 后 exit 会连容器一起退）
#   5. shm-size 64g（官方 1g；沿用本部署原 container.sh 值，防 TP8 worker IPC 吃紧）
#   6. 去掉 /root/.cache 挂载（宿主机无此目录）
#
# 用法: bash container_mooncacke.sh [p|d]   （不带参数时按本机 IP 自动识别）
#   P0 (new2 10.222.42.240) → iquest-pd-mooncacke-p
#   D0 (new8 10.222.41.88)  → iquest-pd-mooncacke-d
# 注意 --rm: 容器 stop 即被删除，重启 docker/误停后用本脚本重建。
# =====================================================================
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"

ROLE="${1:-$(get_node)}"
case "$ROLE" in
    p) NAME="iquest-pd-mooncacke-p" ;;
    d) NAME="iquest-pd-mooncacke-d" ;;
    *) echo "[ERROR] 无法识别角色: 用 bash container_mooncacke.sh [p|d]" >&2; exit 1 ;;
esac

[ -d "$SFS_REPO/vllm_ascend" ] || {
    echo "[ERROR] 仓库树不在 SFS: $SFS_REPO"; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "[INFO] 容器 $NAME 已存在，跳过创建（先 docker rm -f $NAME）"
    exit 0
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
  -v /etc/cloud/topo:/etc/cloud/topo:ro \
  -v /mnt/sfs_turbo:/mnt/sfs_turbo \
  -v /mnt/sfs_turbo/models/IQuest-Q1:/models/IQuest-Q1:ro \
  -v /mnt/sfs_turbo/models/IQuest-Q1-W8A8C8-mxfp8:/models/IQuest-Q1-W8A8C8-mxfp8:ro \
  -v "$SFS_REPO":/workspace \
  -v "$DEPLOY_DIR":/pd \
  -w /workspace \
  -itd "$VLLM_IMAGE" sleep infinity
# 注意：不要在这里加 -e PYTHONPATH=/workspace。
# 整体覆盖会冲掉镜像 set_env.sh 注入的 acl 路径，worker spawn 报 No module named 'acl'；
# 角色脚本内已用追加式 export PYTHONPATH="${PYTHONPATH:-}:/workspace"。

echo "[INFO] 容器 $NAME 已创建（${ROLE} 角色，idle）"
echo "       SFS 整盘: /mnt/sfs_turbo -> /mnt/sfs_turbo（仓库另挂 /workspace，部署目录另挂 /pd）"
