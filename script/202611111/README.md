# IQuest-Q1 部署包总览 · 20261111

2026-11-11 的部署快照集合。本目录下每个子目录是一个**可独立拉起**的部署包，
覆盖两类形态：**PD 分离（Mooncake 池化）** 与 **TP16 混部（单实例跨机）**。

> 本文件只做索引、横向对比和**代码仓口径登记**；每个包的可执行细节、逐条坑与
> 实测数据，以各自 `README.md` 为准。
>
> **最近核对：2026-11-11 17:05 CST**（引擎/proxy 探活 + 各包 `docker_start.sh`、
> git 工作区状态实测）。状态表见 §8，随时会变。

---

## 0. 速查

| 问题 | 答案 |
|---|---|
| 现在主推哪个包 | `pd-mooncacke-mxfp8-1p1d-2m`（PD 1P1D，2 机，W8A8C8 mxfp8） |
| 现在谁在跑 | 2m PD 的 **P/D 引擎在**（`/health` 200），**proxy 已停**；39.253 的 TP16 混部在服务（现场实例） |
| 推理入口 | 2m PD：`http://10.222.46.181:1999`（探活用 `/healthcheck`）；TP16 混部：`http://10.222.39.253:8005` |
| 代码仓怎么对应 | §2：每个包挂的**不是同一个** vllm-ascend 检出，换包必须重建容器 |
| 上线要几步 | `master → decode → prefill → proxy`（proxy **最后**起，§7） |

---

## 1. 包索引

| 目录 | 形态 | 规模 | 推理入口 | 权重 | 状态（11-11 17:05） |
|---|---|---|---|---|---|
| `pd-mooncacke-mxfp8-1p1d-2m/` | PD 分离 1P1D | 2 机（每侧 TP8+EP） | `http://10.222.46.181:1999` | `IQuest-Q1-W8A8C8-mxfp8` | **引擎在、proxy 停**：P/D `7100/health` 200，`1999/healthcheck` 不通 |
| `pd-mooncacke-mxfp8-1p1d-4m/` | PD 分离 1P1D，DP2 | 4 机（每侧 2×TP8，DP=2） | `http://10.222.43.66:8999` | `IQuest-Q1-W8A8C8-mxfp8` | 未运行（机器被外部 4m 部署占用） |
| `vllm-iquest-bf16-1p1d-4m-20261110/` | PD 分离 1P1D，TP16 | 4 机（每侧 2 机 TP16） | `http://10.222.43.66:1999` | bf16 `IQuest-Q1` | 未运行（同上，机器冲突） |
| `vllm-iquest-bf16-tp16-mix-2m/` | 混部（非 PD） | 2 机，一个 TP16 实例 | `http://10.222.39.253:8005` | bf16 `IQuest-Q1` | 机器上有现场实例在服务（**非本包脚本所建**，容器名见 §3.3） |

> `vllm-iquest-bf16-1p1d-4m-20261110` 与 `pd-mooncacke-mxfp8-1p1d-4m` **共用同一批 4 台机**，
> 二者互斥；`pd-mooncacke-mxfp8-1p1d-2m` 已于 11-11 迁到 46.181/47.74，不再占用那 4 台。

---

## 2. 代码仓与镜像（实测口径）

> 每个包都把**一份具体的 vllm-ascend 检出** bind mount 成容器 `/workspace`；镜像统一
> `quay.io/ascend/vllm-ascend:v0.30.0-a5-openeuler`（节点上没有别的可用标签）。
> 不同包代码不同 ⇒ **不能跨包复用容器**，换包必须用该包的 `docker_start.sh` 重建。

### 2.1 包 → 代码仓 / 权重

| 包 | `/workspace` 挂载源 | 分支 / 提交 | 工作区未提交改动 | 权重挂载（`:ro`） |
|---|---|---|---|---|
| `pd-mooncacke-mxfp8-1p1d-2m` | `pd-mooncacke-20261110/` | `releases/v0.30.0` @ `0b2829930` | 3 处：`distributed/kv_transfer/kv_p2p/mooncake_connector.py`、`utils.py`、`worker/v2/attn_utils.py`（C8 mixed-page + hybrid KV 判定） | `IQuest-Q1`、`IQuest-Q1-W8A8C8-mxfp8` |
| `pd-mooncacke-mxfp8-1p1d-4m` | 同上 | 同上 | 同上 | 同上 |
| `vllm-iquest-bf16-1p1d-4m-20261110` | `vllm-ascend-prof/`（由包内 `env.sh` 指定） | `b7abdf22b`（detached） | 3 处：`models/iquest_q1/model.py`、`quantization/methods/kv_cache/mxfp_c8.py`、`worker/v2/attn_utils.py` | `IQuest-Q1`、`IQuest-Q1-W8A8C8-mxfp8` |
| `vllm-iquest-bf16-tp16-mix-2m` | `deploy/vllm-iquest-pqing-bf16/` | 分支 `iquest-q1-npu` @ `e84bb2676`（ahead 8） | 见该仓 `git status`（含 `static_kernel_*` 等） | 仅 `IQuest-Q1`（bf16，含 `mtp/`） |

补充：

- **2m / 4m PD 的那 3 处改动是工作区修改，不是提交**。若把 `SFS_REPO` 改指回主仓
  `vllm-ascend/`（同为 `0b2829930`），这 3 处会丢，C8/hybrid 行为随之变化。
- **tp16-mix 包脚本与现场实例不同源**：现场跑的是
  `/mnt/sfs_turbo/pqing/vllm-iquest-npu-kernel-parsers`（分支 `iquest-q1-npu-kernel-parsers`
  @ `f5cb1c6e9`）。要一比一复现现场，需按该包 `docker_start.sh` 注释替换 `SFS_REPO`。
- `/workspace` 的内容**完全来自上表挂载**；镜像自带的 `/vllm-workspace` 是另一份副本，
  不要拿它判断进到容器里的到底是哪版代码。
- **改了仓里代码，运行中的引擎不会自动生效**：bind mount 只保证文件可见，已 import 的模块
  仍在进程内存里，必须重启引擎（或重建容器）才生效。

### 2.2 vllm-ascend worktree 现状（2026-11-11 17:05）

| worktree | 分支 / 提交 | 用途 |
|---|---|---|
| `vllm-ascend/` | detached `0b2829930` | 主 checkout，= `releases/v0.30.0` |
| `pd-mooncacke-20261110/` | `releases/v0.30.0` @ `0b2829930` + 3 处本地改动 | **本目录 PD 包使用** |
| `vllm-ascend-prof/` | `b7abdf22b` + 3 处本地改动 | **bf16 4m 包使用** |
| `pd-c8-20261111/` | `perf/c8-overlay-backport-20261111` @ `cee8ef3b0` | S2 C8 mixed-page overlay 回移到 v0.30.0；**尚未成包** |
| `vllm-ascend-512k/` `-opt/` `-pd/` `-prodtest/` `-s2a/` `-s2fix/` `-s2m/` `-s3/` | 各实验分支 | 512k / 优化 / hybrid KV / S2 验证用，**不服务本目录的包** |

仓库远端：`fork-ssh` → `liuhy1213-cell/vllm-ascend`（推送用），`origin` 同仓（gh-proxy 拉取），
`upstream` → `vllm-project/vllm-ascend`。

### 2.3 相邻仓库（非部署包，供溯源）

| 目录 | 分支 / 提交 | 备注 |
|---|---|---|
| `vllm-iquest-q1/` | `main` @ `cf13cab` | iquest_q1 插件（0.1.1） |
| `msmodelslim/` | `Feature/IQuest-Q1_W8A8C8-20261107` @ `d4e4f3eb` | 量化配方与适配器（本地分支，未推送） |
| `msmodelslim-26.2.0/` | — | **gitfile 已悬空**：指向 `/mnt/paas/.../msmodelslim/.git/worktrees/...`，该路径不存在；不要据此目录做 git 操作 |
| `cannbot-skills/` | `master` @ `0715cc3b` | 技能库；49 个 dirty 项是 submodule 指针，不是待提交改动 |

---

## 3. 机器与端口矩阵

> ⚠️ **主机名不可靠**：`sjsbg-00004`、`sjsbg-00002` 等 hostname 在多台机器上重复出现，
> 判断机器**一律以 IP 为准**。

### 3.1 2m PD（`pd-mooncacke-mxfp8-1p1d-2m`）

| IP | 角色 | 容器 | 端口 |
|---|---|---|---|
| `10.222.46.181` | prefill + proxy + mooncake_master | `pd-mooncacke-p` | API `7100`、proxy `1999`、KV producer `28010`、master `50088` |
| `10.222.47.74` | decode + MTP | `pd-mooncacke-d` | API `7100`、KV consumer `29010` |

### 3.2 4m PD（两个 4m 包共用机器，互斥）

| IP | `pd-mooncacke-mxfp8-1p1d-4m` | `vllm-iquest-bf16-1p1d-4m-20261110` |
|---|---|---|
| `10.222.43.66` | prefill DP rank0，API `8000`，proxy `8999`，master `50090`，DP `16500` | prefill TP16 rank0，API `7100`，proxy `1999`，rendezvous `29563` |
| `10.222.44.142` | prefill DP rank1，API `8000` | prefill TP16 rank1（headless） |
| `10.222.42.240` | decode DP rank0，API `8000`，DP `16600` | decode TP16 rank0，API `7100`，master `50088`，rendezvous `29563` |
| `10.222.41.88` | decode DP rank1，API `8000` | decode TP16 rank1（headless） |

两边都用 KV 端口 `28010`(P)/`29010`(D)；mxfp8 包容器名 `iquest-mxfp8-p/d`，
bf16 包容器名 `iquest-pd-mooncacke-p/d`。

### 3.3 TP16 混部（`vllm-iquest-bf16-tp16-mix-2m`）

| IP | 角色 | 容器（本包口径） | 端口 |
|---|---|---|---|
| `10.222.39.253` | head，TP16 rank0 | `q1-tp16-mix-node1` | API `8005`、rendezvous `29563` |
| `10.222.34.2` | worker，TP16 rank1 | `q1-tp16-mix-node2` | headless（不监听） |

> 现场在服务的实例容器名是 `q1-tp16-ziang-ar-mooncake-2tib-node1`，**不是**本包脚本创建的；
> 本包容器的 `--rm` 语义也意味着 stop 即删除，误停需重建。

---

## 4. 怎么选

| 目标 | 用哪个包 |
|---|---|
| 要 PD 分离 + 低 TPOT（当前默认） | `pd-mooncacke-mxfp8-1p1d-2m` |
| 要 16 卡 EP 组 / 更大吞吐，愿意占 4 台机 | `pd-mooncacke-mxfp8-1p1d-4m` |
| 要 bf16 基线、复现 512k/TP16 历史数据 | `vllm-iquest-bf16-1p1d-4m-20261110` |
| 不要 PD、单入口、最小部署面 | `vllm-iquest-bf16-tp16-mix-2m` |

选型依据与实测对比见 `../../docs/IQuest-Q1_512k_性能测试报告.md` 与
`../../docs/IQuest-Q1_512k_经验沉淀.md`。

---

## 5. 通用约定（所有包一致）

1. **脚本变量层不统一，按包看**：
   - `pd-mooncacke-mxfp8-1p1d-2m`：全字面值（`docker_start.sh`、`prefill/p.sh`、`decode/d.sh`、`run_proxy.sh`）。
   - `pd-mooncacke-mxfp8-1p1d-4m`：`docker_start.sh` 顶部常量（`SFS_REPO` / `MODEL_*` / `VLLM_IMAGE`）。
   - `vllm-iquest-bf16-1p1d-4m-20261110`：**有 `env.sh`**（`VLLM_IMAGE`、`SFS_ROOT`、`SFS_REPO`、`get_node()`）。
   - `vllm-iquest-bf16-tp16-mix-2m`：`docker_start.sh` 顶部常量 + `serve/n1.sh`、`serve/n2.sh`。
   换机器时改对应文件里的 IP/端口/路径，**不要假设有统一的 env 层**。
2. **挂载口径**：
   - `本目录 → /pd`（容器内跑 `/pd/...` 的脚本）；`/pd` 必须是**真实 bind mount**，
     不要复用旧容器——旧容器的 bind 可能指向已删除/已移动的旧路径（看到的是旧快照）。
   - `vllm-ascend 代码仓 → /workspace`，具体是哪个检出见 §2.1。
   - `/mnt/sfs_turbo` 整盘 + 模型目录（通常 `:ro`）。
3. **不要在 `docker run` 里加 `-e PYTHONPATH=/workspace`**：会覆盖镜像注入的 acl 路径，
   spawn worker 报 `No module named 'acl'`。`PYTHONPATH` 必须在角色脚本里**追加**：
   `export PYTHONPATH="${PYTHONPATH:-}:/workspace"`。
4. **启动顺序**：`master → decode → prefill → proxy`（混部包为 `head → worker`）。
   - master 必须先起（引擎初始化即连池）；
   - **proxy 必须最后起**：P/D 脚本开头的残留清理块会 `pkill -9 python`，proxy 也是 python。
5. **日志落 SFS**：`docker exec -d` 会丢弃屏输出，拉起时一律重定向到
   `<包>/logs/{prefill,decode,proxy,mooncake}/...`。proxy 自身还会把请求日志写到
   `/pd/logs/proxy/proxy.log`（**与重定向的 stdout 文件不是同一个**）。
6. **探活端点**：引擎 `/health`；PD proxy 是 **`/healthcheck`**（`/health` 会 404）。
   `curl` 返回 `000` 表示连接被拒/超时，即**端口没在监听**。
7. **推理只走 proxy**：各引擎的 API 端口是 `kv_producer` / `kv_consumer` 角色，
   直连发请求不走 PD 编排，只适合 `/health`。
8. **NPU 独占**：一张 davinci 同一时刻只能被一个容器真正占用。起新站前先
   `npu-smi info -t proc-mem -i <id>` 确认 "No process in device."。

---

## 6. 共同坑（实踩）

| 症状 | 根因 | 处置 |
|---|---|---|
| 引擎没起，日志只有 `mkdir: ... File exists` | `/pd/logs/<role>` 是悬空软链 | 容器内 `ln -sfn <本目录>/logs/<role> /pd/logs/<role>`；脚本已容错 `mkdir -p … \|\| echo WARN` |
| `ModuleNotFoundError: No module named 'acl'` | `PYTHONPATH` 被整体覆盖（export 或 `docker run -e`） | 改追加式，且 `docker run` 不要带 `-e PYTHONPATH` |
| 引擎 `/health` 200，但业务请求全失败 | **proxy 没起/已停**，只探了引擎 | 另探 proxy `/healthcheck`；重起 `run_proxy.sh`（最后起） |
| 有 `vllm serve` 进程但端口不监听、NPU 不占卡 | 上次启动死在 spawn 阶段的残留僵尸（“假绿”） | `pkill -KILL vllm; pkill -KILL VLLM`，确认 `proc-mem` 干净再重启 |
| 端口莫名被占 | 同机多容器共享 host 网络端口表 | 起之前 `ss -tlnp`；用 `/proc/<pid>/cgroup` 确认 pid 归属 |
| 清理命令把会话杀了 | 内联 `pkill -f 'vllm serve'` 命中执行它的 shell | 用脚本文件执行，或按 comm 匹配（`pkill -9 VLLM`，不带 `-f`） |
| 容器里的 `/pd` 内容很旧 | 容器是旧的，bind 源目录已被移动/删除（docker 按 inode 保持旧视图） | 用 `docker_start.sh` 重建容器，不要复用旧容器 |
| 改了仓里代码，容器行为没变 | `/workspace` 是 bind mount，但旧进程仍持有已 import 的模块；且部分包挂的是**别的 worktree** | 确认 §2.1 的挂载源就是你以为的那个检出，再重启引擎 |
| 日志时间差 8 小时 | 容器内 UTC，宿主 CST+8 | 换算后再对时间线 |
| decode 起不来，报 "no config.json" | MTP 草稿 `/models/IQuest-Q1/mtp` 未挂载 | 按各包 `docker_start.sh` 的挂载清单补齐 |

---

## 7. 常用命令

```bash
ROOT=/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/202611111

# ---- 2m PD（当前主推）----
TS=$(date +%Y%m%d_%H%M%S); D=$ROOT/pd-mooncacke-mxfp8-1p1d-2m
ssh root@10.222.46.181 'docker exec -d pd-mooncacke-p bash /pd/mooncake/run_master.sh'
ssh root@10.222.47.74  "docker exec -d pd-mooncacke-d bash -c 'bash /pd/decode/d.sh  >> $D/logs/decode/engine-$TS.log 2>&1'"
ssh root@10.222.46.181 "docker exec -d pd-mooncacke-p bash -c 'bash /pd/prefill/p.sh >> $D/logs/prefill/engine-$TS.log 2>&1'"
ssh root@10.222.46.181 "docker exec -d pd-mooncacke-p bash -c 'bash /pd/run_proxy.sh   >> $D/logs/proxy/proxy-stdout.log 2>&1'"

# 探活（000 = 没在监听）
for u in 10.222.46.181:7100/health 10.222.47.74:7100/health 10.222.46.181:1999/healthcheck; do
  printf '%-40s ' "$u"; curl -s -m 3 -o /dev/null -w '%{http_code}\n' "http://$u" || true
done

# 冒烟（走 proxy）
curl -s http://10.222.46.181:1999/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"用一句话介绍你自己"}],"max_tokens":512,"temperature":0}'

# 停止（各机容器内，按 comm 匹配）
docker exec pd-mooncacke-p bash -c 'pkill -TERM vllm; sleep 6; pkill -KILL vllm; pkill -KILL VLLM'
docker exec pd-mooncacke-d bash -c 'pkill -TERM vllm; sleep 6; pkill -KILL vllm; pkill -KILL VLLM'
docker exec pd-mooncacke-p pkill -f load_balance_proxy_server_example
docker exec pd-mooncacke-p pkill -f '[m]ooncake_master'
```

其余包的拉起/停止命令见各自 `README.md` 的拓扑与启动章节。

---

## 8. 状态快照

### 8.1 2026-11-11 17:05 CST（本次实测）

| 站点 | 观测结果 | 说明 |
|---|---|---|
| 2m PD P/D | `10.222.46.181:7100/health` **200**、`10.222.47.74:7100/health` **200** | 引擎自 11:40 CST 起在跑 |
| 2m PD proxy | `10.222.46.181:1999/healthcheck` **000**（不通） | `logs/proxy/proxy.log` 显示最后一次会话 14:29 CST 正常 shutdown；**要用服务需重起 proxy** |
| TP16 混部 | `10.222.39.253:8005/health` **200** | 现场既有实例（容器 `q1-tp16-ziang-ar-mooncake-2tib-node1`），非本目录脚本所建 |
| 4m 机器 | `10.222.43.66:8999`、`:1999` 均 **000** | 43.66/44.142/42.240/41.88 被外部 TP16 部署占用（容器 `iquest-pd-mooncacke-p/d`，`/pd` 指向 `/mnt/sfs_turbo/huawei/zuoqian/...`）；使用前先确认归属 |

### 8.2 2026-11-11 12:30 CST（历史）

| 站点 | 观测结果 |
|---|---|
| 2m PD（46.181 / 47.74） | 全绿：P/D `/health` 200、proxy `/healthcheck` 200、两次冒烟通过（冷启 30 s，二次 0.59 s） |

> 状态随时会变。改完部署请顺手更新本节，**保留观测时间**。

---

## 9. 维护约定

- 新增部署包：目录名用 `<形态>-<量化/精度>-<并行>-<机数>`，例如
  `pd-mooncacke-mxfp8-1p1d-2m`；必须包含 `README.md`、`docker_start.sh`、
  `run_proxy.sh`（PD 包）、`logs/`，并在 §1 索引、§2.1 代码仓表和 §3 机器矩阵里登记。
- **归档镜像**：本目录（不含 `logs/`、`archive/`、`perf_breakdown/`、`*.log`、`*.swp`）已同步到
  vllm-ascend 仓库 `releases/v0.30.0` 分支的 `script/202611111/`。改完脚本或文档后要一并推送，
  否则仓里的部署包会和此处口径不一致。
- 换机器：先改**本文件矩阵**，再改包内 `docker_start.sh` 的 IP 常量与角色脚本的
  `local_ip`/`--host`/proxy 后端，最后**重建容器**验证。
- 换代码版本：改 §2.1 的挂载源/分支，重建容器；若沿用某 worktree 的未提交改动，
  在 §2.1 记清是哪几处。
- 不要在已运行的容器上原地改脚本后直接重跑：`p.sh`/`d.sh` 会 `pkill python`，
  且旧容器可能挂着旧路径。
- 历史包（11-10 及更早）在 `../` 下，未删除；判断现行口径以本目录和
  `../../docs/` 的最近报告为准。

---

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-11 17:05 | 重写：新增 §2 代码仓/镜像/worktree 口径；修正“所有包都是全字面值”的说法（bf16-4m 有 `env.sh`）；状态快照按探活刷新（2m proxy 已停）；§7 探活与 proxy 日志路径修正 |
| 2026-11-11 12:30 | 初版：四个部署包索引、机器矩阵、通用约定与坑 |