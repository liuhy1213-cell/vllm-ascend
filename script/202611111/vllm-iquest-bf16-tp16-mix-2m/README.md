# IQuest-Q1 TP16 混部 · 双机（2m）启动包

2026-11-11 创建，口径参照同级 `../pd-mooncacke-mxfp8-1p1d-2m/`、`../pd-mooncacke-mxfp8-1p1d-4m/`
与 `../vllm-iquest-bf16-1p1d-4m-20261110/`：顶层只放**可直接执行的全字面值脚本**
（无 `env.sh` 变量层、无 `start.sh` 编排层）。

> **混部 ≠ PD 分离。** 这里一个 TP16 实例跨两台机同时做 prefill + decode：
> 没有 `--kv-transfer-config`、没有 `mooncake_master`、没有 proxy、没有 KV 传输，
> 对外只有 `10.222.39.253:8005` 一个入口。要 P/D 拆分请看 `../pd-mooncacke-mxfp8-1p1d-*`。

---

## 1. 目录结构

```
vllm-iquest-bf16-tp16-mix-2m/
├── README.md            # 本文件（唯一 runbook）
├── docker_start.sh      # bash docker_start.sh [n1|n2]   建角色容器（--rm，shm 64g）
├── serve/n1.sh          # 39.253  head    TP16 rank0，API :8005
├── serve/n2.sh          # 34.2    worker  TP16 rank1 + --headless
└── logs/                # 运行期日志（拉起时重定向到这里）
```

## 2. 拓扑与端口

| 角色 | 机器 | IP | 容器 | 引擎 |
|---|---|---|---|---|
| head | sjsbg-00004 | `10.222.39.253` | `q1-tp16-mix-node1` | node-rank **0**，API `0.0.0.0:8005` |
| worker | sjsbg-00002 | `10.222.34.2` | `q1-tp16-mix-node2` | node-rank **1**，`--headless` |

| 端口 | 位置 | 说明 |
|---|---|---|
| 8005 | 39.253 | 唯一推理入口（`/health`、`/v1/chat/completions`） |
| 29563 | 39.253 | 跨机 torch rendezvous（`--master-port`） |
| 7100 / 1999 / 28010 / 29010 / 50088 | — | **本包不用**（那些是 PD 站的 API / proxy / KV / mooncake） |

```
client ──► 39.253:8005 ──┐
                          ├─ 一个 TP16 实例（16 卡，256 experts，bf16 权重）
               34.2 ──────┘   rank1 headless
```

## 3. 容器准备（容器不存在时）

```bash
# 39.253
bash docker_start.sh n1        # → q1-tp16-mix-node1
# 34.2
bash docker_start.sh n2        # → q1-tp16-mix-node2
```

脚本口径（`--rm`，停即删）：`--net=host`、`--privileged`、`--shm-size=64g`、8 张 davinci、
PID1 `sleep infinity`，挂载：

| 宿主 | 容器内 |
|---|---|
| 本目录 | `/pd` |
| `$SFS_REPO`（默认 `deploy/vllm-iquest-pqing-bf16`，分支 `iquest-q1-npu` @ `e84bb2676`） | `/workspace` |
| `/mnt/sfs_turbo/pqing/ascend-custom-allreduce`（`main` @ `a61f7c7`） | `/workspace/custom-ar`（ro） |
| `/mnt/sfs_turbo/models/IQuest-Q1`（bf16 + `mtp/`） | `/models/IQuest-Q1`（ro） |
| `/mnt/sfs_turbo` | `/mnt/sfs_turbo` |

> 已在跑的现场那套 TP16 混部 `/workspace` 挂的是
> `/mnt/sfs_turbo/pqing/vllm-iquest-npu-kernel-parsers`（`f5cb1c6e9`）。
> 要一比一复现，改 `docker_start.sh` 顶部 `SFS_REPO` 一行后 `FORCE=1` 重建容器。

**不要**在 `docker run` 里加 `-e PYTHONPATH=/workspace`：会覆盖镜像注入的 acl 路径，
spawn worker 报 `No module named 'acl'`；`PYTHONPATH` 在 `serve/*.sh` 里追加。

验证代码树（两台都做）：

```bash
docker exec q1-tp16-mix-node1 bash -c \
  'PYTHONPATH=/workspace python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"'
```

## 4. 拉起顺序（head → worker）

```bash
TS=$(date +%Y%m%d_%H%M%S)
D=/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/202611111/vllm-iquest-bf16-tp16-mix-2m

# ① head（39.253）rank0 —— 先起
ssh root@10.222.39.253 "docker exec -d q1-tp16-mix-node1 bash -c \
  'bash /pd/serve/n1.sh >> $D/logs/n1-${TS}.log 2>&1'"

# ② worker（34.2）rank1 —— 等 head 开始加载权重后再起（间隔 20~60 s 足够）
ssh root@10.222.34.2 "docker exec -d q1-tp16-mix-node2 bash -c \
  'bash /pd/serve/n2.sh >> $D/logs/n2-${TS}.log 2>&1'"
```

- 两台都是 bf16 606 GB 权重 + 16 卡建图，**就绪量级 5–10 分钟**（SFS 页缓存热时更快）。
- 日志**必须重定向到 SFS**：脚本只打屏，`docker exec -d` 会把屏输出丢掉。
- 看实时输出就去掉 `-d` 做前台 exec（占终端，Ctrl+C 会连引擎一起杀）。
- 顺序不能反：rank0 的 `:29563` 是 rendezvous，rank1 先起会一直重连。

```bash
# 就绪探测
curl -s -o /dev/null -w 'health %{http_code}\n' http://10.222.39.253:8005/health
```

### 脚本开头的残留清理块

```bash
ps aux | grep 'VLLM' | awk '{print $2}' | xargs -r kill -9   # worker proctitle = VLLM::Worker_TPn_EPn
ps aux | grep 'vllm' | awk '{print $2}' | xargs -r kill -9
pkill -9 vllm; pkill -9 VLLM; pkill -9 python; pkill -9 ray
sleep 10
```

用 `set +e` 包住（无残留时 `grep`/`pkill` 返回 1，会被 `set -Eeuo pipefail` 中断）。
清理只在**本容器**内生效；必须**用文件方式执行**，不要内联 `bash -c "…pkill python…"`
—— 内联时 shell 自身命令行含 `vllm`，会被 `ps aux` 匹配后一起杀掉。

## 5. 探活与冒烟

```bash
curl -s -o /dev/null -w 'health %{http_code}\n' http://10.222.39.253:8005/health
curl -s http://10.222.39.253:8005/v1/models | python3 -m json.tool | head

curl -s http://10.222.39.253:8005/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"用一句话介绍你自己"}],"max_tokens":512,"temperature":0}' \
| python3 -c "import sys,json;d=json.load(sys.stdin);m=d['choices'][0]['message'];print('reasoning:',(m.get('reasoning') or '')[:100]);print('content  :',(m.get('content') or '')[:300]);print('usage    :',d['usage'])"
```

- 思维链在 `message.reasoning`（若启用了 `--reasoning-parser iquest_q1`），正文在 `message.content`。
- `max_tokens` 太小（<200）时 token 全耗在 reasoning 上、`content` 为空，不是故障。
- 首次请求慢（30 s 级）是图捕获/权重预热，不是异常。

## 6. 停止

```bash
# 两端各停（按 comm 匹配，不会误杀执行壳）
docker exec q1-tp16-mix-node1 bash -c 'pkill -TERM vllm; sleep 6; pkill -KILL vllm; pkill -KILL VLLM'
docker exec q1-tp16-mix-node2 bash -c 'pkill -TERM vllm; sleep 6; pkill -KILL vllm; pkill -KILL VLLM'

# 宿主机确认显存真的释放（每张卡都要 "No process in device."）
npu-smi info -t proc-mem -i 0
```

不要用 `pkill -f 'VLLM::|vllm serve'` 内联写法：模式串会命中执行它自己的 shell。

## 7. 生效配置（`serve/n1.sh` / `serve/n2.sh`）

| 项 | 值 |
|---|---|
| 模型 | `/models/IQuest-Q1`（**bf16**，无 `--quantization`；`mtp/` 在同目录） |
| 并行 | `TP16` + `--nnodes 2`（rank0 head / rank1 `--headless`）+ `--distributed-executor-backend mp` |
| 上下文 | `--max-model-len 8192`（模型原生上限 524288，这里只开 8k） |
| 批次 | `--max-num-batched-tokens 1024`、`--max-num-seqs 8` |
| 显存 | `--gpu-memory-utilization 0.95` |
| prefix caching | **关** |
| 投机 | `eagle`，`/models/IQuest-Q1/mtp`，`num_speculative_tokens 5` |
| 图模式 | `FULL_DECODE_ONLY`，`cudagraph_capture_sizes [4,8,16]`，`--cudagraph-metrics` |
| custom AR | `IQUEST_CUSTOM_AR=1` / `VLLM_ASCEND_ENABLE_CUSTOM_AR=1` / `CUSTOM_AR_DISPATCH=torch` / `CUSTOM_AR_LAYOUT=compact`（两端一致） |
| 融合 | `VLLM_ASCEND_ENABLE_IQUEST_Q1_FUSION=1` |
| 网络 | `enp39s0f3` 绑 `VLLM_HOST_IP`/`HCCL_IF_IP`/`GLOO_*`/`TP_*`/`HCCL_SOCKET_IFNAME`；`HCCL_CONNECT_TIMEOUT=180`、`HCCL_EXEC_TIMEOUT=180` |
| 进程 | `VLLM_WORKER_MULTIPROC_METHOD=spawn`、`PYTHONPATH` 追加 `/workspace`、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |

### 和现场快照（`../../iquest-tp16-mtp/iquest-tp16-mtp-serve-cmdline.txt`）的差异

| 项 | 本包 | 2026-11-06 快照 |
|---|---|---|
| `max-model-len` | 8192 | 32768 |
| MTP depth | 5 | 3 |
| capture sizes | `[4,8,16]` | `[4,8,16]` |
| EP / parser | 未加（按启动命令原样） | 无 EP / 无 parser |
| custom AR | 开 | 开 |

## 8. 已知坑

| 症状 | 根因 | 处置 |
|---|---|---|
| **8005 起不来 / `Address already in use`** | 2026-11-11 现场 39.253 上 `q1-tp16-ziang-ar-mooncake-2tib-node1` 正占着 8005（还有 34.2 的 node2），且 16 张卡都被它占着 | 先停那套，或改 `serve/n1.sh` 的 `PORT`（两台一致）并换机器；`n1.sh` 已加端口预检，会直接报错退出 |
| rank1 一直重连 / 卡在 init | 顺序反了，或两台参数不一致（尤其 `--max-model-len`、capture sizes、custom AR env） | 先起 rank0；两台参数逐项对齐 |
| `No module named 'acl'` | `PYTHONPATH` 被整体覆盖 | 只追加：`${PYTHONPATH:-}:/workspace`；`docker run` 里别加 `-e PYTHONPATH` |
| 图模式没生效（吞吐低于预期） | `seqs 8 × (depth 5 + 1) = 48` token/步 > capture 上限 16，多数步回落 eager | capture 扩到 `[4,8,16,24,32,48]`，或降 `max-num-seqs` / MTP 深度 |
| 有 `vllm serve` 进程但 8005 不监听、NPU 不占卡 | 假绿：上次启动死在 spawn 阶段的残留僵尸 | `pkill -KILL vllm; pkill -KILL VLLM`，确认 `proc-mem` 为 "No process" 再重启 |
| spawn 阶段直接挂、日志无 Python 栈 | 容器没挂 `/workspace` 或没挂 `mtp/` | 用 `docker_start.sh` 重建；`/models/IQuest-Q1/mtp` 必须存在 |
| 日志时间差 8 小时 | 容器内 UTC，宿主 CST+8 | 换算后再对时间线 |

## 9. 变更记录

- **2026-11-11 创建**：按 `deploy/202611111/` 同级包的口径整理出 TP16 混部启动目录。
  - 启动命令原型来自用户提供的 39.253/34.2 混部 serve（8192 / seqs 8 / MTP5 / capture `[4,8,16]`）；
    补入 rank1 必需的 `--headless`，补 CANN/引擎 env、custom AR env、端口预检与残留清理块。
  - 创建时现场状态：39.253 / 34.2 上 `q1-tp16-ziang-ar-mooncake-2tib-node1/2` 正在跑
    （`39.253:8005` health 200，master-port 29570，524k + EP + Mooncake AscendStore），
    本包要起必须先停那套或换端口/机器。