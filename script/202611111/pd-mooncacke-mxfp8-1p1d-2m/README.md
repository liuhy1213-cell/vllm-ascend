# IQuest-Q1 PD 分离 · 1P1D 双机（Mooncake 池化）— 2m 站

现行 runbook。顶层只留**可直接执行的全字面值脚本**（没有 `env.sh` 变量层），
旧编排层 / 一次性脚本 / 压测脚本 / 历史文档全部在 `archive/`（未删除）。

最近一次实测拉起：**2026-11-11 11:44（CST）在新机器上全绿**（P=46.181 / D=47.74，
P 就绪 ~2.5 min、D ~4 min，proxy `/healthcheck` 200、两次冒烟通过），
变更记录见 §10；2026-11-10 21:07 的旧机器（42.240/41.88）记录保留在 §10。

> ⚠️ **本站已迁到 46.181 / 47.74**，与 4m 站（`../pd-mooncacke-mxfp8-1p1d-4m`，用
> 42.240 / 41.88 / 43.66 / 44.142）**不再共用机器**。复制命令前仍请确认 IP：
>
> | IP | 机器 | **本站（2m）** | 4m 站 |
> |---|---|---|---|
> | `10.222.46.181` | sjsbg-00004 (new4) | **prefill** + proxy + mooncake_master | 不使用 |
> | `10.222.47.74` | sjsbg-00003 (new3) | **decode** + MTP | 不使用 |
> | `10.222.42.240` / `10.222.41.88` | sjsbg-00002 / -00008 | 不使用（旧站机器，现被 4m 站占用） | decode rank0 / rank1 |
> | `10.222.43.66` / `10.222.44.142` | sjsbg-00001 / -00006 | 不使用 | prefill rank0 / rank1 |
>

---

## 1. 目录结构

```
pd-mooncacke-mxfp8-1p1d-2m/
├── README.md            # 本文件（唯一现行 runbook）
├── docker_start.sh      # bash docker_start.sh [p|d]   建角色容器（--rm，shm 64g）
├── mooncake/
│   ├── mooncake.json    # master_server_address = 10.222.46.181:50088（权威）
│   └── run_master.sh    # 起 mooncake_master（仅 P 侧；日志 → /pd/logs/mooncake/master.log）
├── prefill/p.sh         # 46.181  prefill  TP8+EP  eager  kv_producer:28010
├── decode/d.sh          # 47.74   decode   TP8+EP  图模式 + MTP3  kv_consumer:29010
├── run_proxy.sh         # 46.181:1999  负载均衡（P/D 各指向本站一侧）
├── logs/                # 运行期日志（prefill/ decode/ mooncake/ proxy/）
├── tools/               # profiling 解析脚本（analyze_trace / parse_engine_log / val_qk_norm …）
├── perf_breakdown/      # 某次 decode rank0 的性能拆解产物
└── archive/             # 旧编排层、一次性脚本、压测脚本、历史文档（见 archive/README.md）
```

## 2. 拓扑与端口

```
client ──► 46.181:1999 (proxy) ──► prefill 46.181:7100 (TP8, kv_producer 28010)
                                        │ Mooncake 实时 KV 传输
                                        ▼
                                   decode  47.74:7100  (TP8 + MTP depth3, kv_consumer 29010)

池化: mooncake_master 46.181:50088（P 侧容器 pd-mooncacke-p 内）
```

| 端口 | 位置 | 说明 |
|---|---|---|
| 7100 | 两侧各一 | vLLM API（不同机器可同号） |
| 1999 | 46.181 | proxy 入口；**探活是 `/healthcheck`**（不是 `/health`） |
| 28010 / 29010 | P / D | PD KV 传输端口 |
| 50088 | 46.181 | mooncake_master（另会尝试起 9003 metrics，被占用只告警不影响） |

## 3. 容器准备（容器不存在时）

```bash
# 46.181
bash docker_start.sh p        # → pd-mooncacke-p（prefill + proxy + master）
# 47.74
bash docker_start.sh d        # → pd-mooncacke-d（decode + MTP）
```

脚本口径（`--rm`，停即删）：`--net=host`、`--shm-size=64g`、8 张 davinci、PID1 `sleep infinity`，
挂载 `/mnt/sfs_turbo` 整盘 + `/models/{IQuest-Q1, IQuest-Q1-W8A8C8-mxfp8}`(ro) +
**本目录 → `/pd`** + **vllm-ascend 代码仓 → `/workspace`**。
`/workspace` 必须是 vllm-ascend 代码仓，示例：

```text
/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/vllm-ascend
```

> 当前 `docker_start.sh` 的 `SFS_REPO` 指向
> `/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/pd-mooncacke-20261110` —— 它**本身就是
> vllm-ascend 仓库的 `/workspace` 代码树**：同一仓库的 `releases/v0.30.0` linked worktree
> （gitfile 指向 `vllm-ascend/.git/worktrees/pd-mooncacke-20261110`，提交 `0b2829930`，
> 与主仓当前提交一致）。该 worktree 带 3 个未提交的部署补丁
> （`distributed/kv_transfer/kv_p2p/mooncake_connector.py`、`utils.py`、`worker/v2/attn_utils.py`），
> 如需把 `SFS_REPO` 直接改成主仓 `.../vllm-ascend`，必须先把这 3 个补丁移植过去。

同名容器已存在时只提示不动；确认可删时 `FORCE=1 bash docker_start.sh p|d`。

**验证代码树**（两台都做）：

```bash
docker exec pd-mooncacke-p bash -c \
  'PYTHONPATH=/workspace python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"'
# 期望 /workspace/vllm_ascend/__init__.py
# 宿主机上对应 .../vllm-ascend/vllm_ascend/__init__.py
# （当前实例为其 releases/v0.30.0 worktree：.../pd-mooncacke-20261110/vllm_ascend/__init__.py）
```

### 旧容器（bare 容器 + `/pd` 符号链接 farm）

现网两台跑的 `pd-mooncacke-p/d` 是**旧 bare 容器**（shm 1g、无 `/pd` 挂载）：
`/pd` 是逐条指向本目录顶层条目的符号链接 farm，`/workspace`、`/models/*` 也是软链。
farm 里的 `/pd/logs/{prefill,decode,proxy}` 有可能是**悬空软链**（指向已删除的
`iq-vllm-bf16/pd-mooncacke-mxfp8-1p1d-2m/logs/...`），会让 `mkdir -p` 报 EEXIST。
2026-11-10 已把三处改指本目录 `logs/`，命令：

```bash
D=/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/202611111/pd-mooncacke-mxfp8-1p1d-2m
docker exec pd-mooncacke-p bash -c "
  ln -sfn $D/logs/prefill /pd/logs/prefill
  ln -sfn $D/logs/proxy   /pd/logs/proxy"
```

彻底摆脱 farm 的做法是用 `docker_start.sh` 重建容器（`/pd` 变成真实 bind mount）。

## 4. 拉起顺序（master → decode → prefill → proxy）

两条硬规则：

1. **master 必须先起**（引擎初始化即连 50088）；若 50088 已有监听说明 master 在跑，**不要重复起**
   （会 bind 失败），直接跳到 decode。
2. **proxy 必须最后起**：`p.sh`/`d.sh` 开头的残留清理块会 `pkill -9 python`，proxy 也是 python。

```bash
D=/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/202611111/pd-mooncacke-mxfp8-1p1d-2m
TS=$(date +%Y%m%d_%H%M%S)

# ① master（仅在 50088 无监听时）
ssh root@10.222.46.181 'ss -ltn | grep 50088' || \
ssh root@10.222.46.181 'docker exec -d pd-mooncacke-p bash /pd/mooncake/run_master.sh'

# ② decode（47.74；实测约 3.5 min 就绪，含 13 档图捕获 + MTP）
ssh root@10.222.47.74 "docker exec -d pd-mooncacke-d bash -c \
  'bash /pd/decode/d.sh >> $D/logs/decode/engine-${TS}.log 2>&1'"

# ③ prefill（46.181；实测约 2 min 就绪，SFS 页缓存热时）
ssh root@10.222.46.181 "docker exec -d pd-mooncacke-p bash -c \
  'bash /pd/prefill/p.sh >> $D/logs/prefill/engine-${TS}.log 2>&1'"

# ④ proxy（46.181）
ssh root@10.222.46.181 "docker exec -d pd-mooncacke-p bash -c \
  'bash /pd/run_proxy.sh >> $D/logs/proxy/proxy-stdout.log 2>&1'"
```

- ②③ 互相不阻塞（只依赖 master），**可以并行起**，实测这样总共约 4 min 全绿。
- 日志**必须重定向到 SFS**（`$D/logs/...`）：脚本本身只打屏，`docker exec -d` 会把屏输出丢掉。
- 看实时输出就去掉 `-d` 做前台 exec（占终端，Ctrl+C 会连引擎一起杀）。

### 脚本开头的残留清理块

```bash
ps aux | grep 'VLLM' | awk '{print $2}' | xargs -r kill -9   # worker proctitle = VLLM::Worker_TPn_EPn
ps aux | grep 'vllm' | awk '{print $2}' | xargs -r kill -9
pkill -9 vllm; pkill -9 VLLM; pkill -9 python; pkill -9 ray
sleep 10
```

用 `set +e` 包住（无残留时 `grep`/`pkill` 返回 1，会被 `set -Eeuo pipefail` 中断）。
清理只在**本容器**内生效（容器没开 `--pid=host`），不会误杀同机其它容器的引擎；
但必须**用文件方式执行**，不要内联 `bash -c "…grep 'vllm'…"` —— 内联时 shell 自身命令行
含 `vllm`，会被 `ps aux` 匹配后一起杀掉。

## 5. 探活与冒烟

```bash
curl -s -o /dev/null -w 'P %{http_code}\n' http://10.222.46.181:7100/health
curl -s -o /dev/null -w 'D %{http_code}\n' http://10.222.47.74:7100/health
curl -s -o /dev/null -w 'X %{http_code}\n' http://10.222.46.181:1999/healthcheck
curl -s http://10.222.46.181:1999/v1/models | python3 -m json.tool | head

# PD 全链路（proxy → P 预填 → KV 传输 → D 生成 + MTP）
curl -s http://10.222.46.181:1999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"用一句话介绍你自己"}],"max_tokens":512,"temperature":0}' \
| python3 -c "import sys,json;d=json.load(sys.stdin);m=d['choices'][0]['message'];print('reasoning:',(m.get('reasoning') or '')[:100]);print('content  :',(m.get('content') or '')[:300]);print('usage    :',d['usage'])"
```

- **推理一律走 `:1999`**。两个 `7100` 实例是 `kv_producer`/`kv_consumer` 角色，直连发推理请求
  不走 PD 编排，不要当普通服务端用（只用来 `/health`）。
- 思维链在 `message.reasoning`（`iquest_q1` parser），正文在 `message.content`；
  `max_tokens` 太小（<200）时 token 全耗在 reasoning 上、`content` 会是空，不是故障。

## 6. 停止与清理

```bash
# 引擎（各机自己容器内，按 comm 匹配，不会误杀执行壳）
docker exec pd-mooncacke-p bash -c 'pkill -TERM vllm; sleep 6; pkill -KILL vllm; pkill -KILL VLLM'
docker exec pd-mooncacke-d bash -c 'pkill -TERM vllm; sleep 6; pkill -KILL vllm; pkill -KILL VLLM'
# proxy / master（46.181）
docker exec pd-mooncacke-p pkill -f load_balance_proxy_server_example
docker exec pd-mooncacke-p pkill -f '[m]ooncake_master'

# 宿主机确认显存真的释放（每张卡都要 "No process in device."）
npu-smi info -t proc-mem -i 0
```

不要用 `pkill -f 'VLLM::|vllm serve'` 这种**内联**写法：模式串会命中执行它自己的 shell，
把会话连同命令一起杀掉（2026-11-10 实踩，命令静默中断）。

**假绿**：进程列表里有 `vllm serve`、但 7100 无监听且 NPU 不占卡 = 上次启动死在 spawn 阶段的
残留僵尸（Ctrl+C 只杀会话不杀引擎）。必须清干净、确认 `proc-mem` 为 "No process" 再重启。

## 7. 生效配置（2026-11-11 三改后的脚本口径；资源实测数据仍取自 11-11 11:44 那次运行）

| 项 | prefill（`p.sh`，46.181） | decode（`d.sh`，47.74） |
|---|---|---|
| 模型 | `/models/IQuest-Q1-W8A8C8-mxfp8` + `--quantization ascend`（**KV=float，C8 未启用**） | 同左 |
| 并行 | TP8 + `--enable-expert-parallel` | TP8 + EP |
| 模式 | `--enforce-eager`（P 无图模式） | `FULL_DECODE_ONLY` + `npugraph_ex`，13 档 capture sizes（1…128） |
| 上下文 | `--max-model-len 524288` | `--max-model-len 524288` |
| 批次 | `--max-num-batched-tokens 32768`、`--max-num-seqs 32` | `--max-num-batched-tokens 256`、`--max-num-seqs 64` |
| block-size | 128 | 128 |
| 显存 | `--gpu-memory-utilization 0.90` | `0.94`（不 pin `--kv-cache-memory`，容量自动推导） |
| prefix caching | **开**（池查找依赖） | 关 |
| KV 传输 | producer `28010`, engine_id 0 | consumer `29010`, engine_id 1 |
| MTP | — | `eagle`，`/models/IQuest-Q1/mtp`，depth 3 |
| 调度 | `--async-scheduling` | `--async-scheduling` |
| additional_config | `enable_fused_mc2` + `enable_iquest_triton_qk_norm` + `enable_fused_o_proj_ar` + `enable_flashcomm1` + `enable_cpu_binding` + `multistream_overlap_shared_expert` | 上面三项 + `enable_cpu_binding` + `multistream_overlap_shared_expert` + `recompute_scheduler_enable` + `ascend_compilation_config` |
| profiler | `torch` → `/pd/logs/profile/prefill` | `torch` → `/pd/logs/profile/decode` |

两个连接器（两侧一致）：`MultiConnector` = `MooncakeHybridConnector` + `AscendStoreConnector`
（`backend=mooncake`, `use_layerwise=false`, `prefill/decode` 均 `dp_size=1, tp_size=8`）。

实测资源占用：

| 项 | P | D |
|---|---|---|
| 就绪耗时（从 exec 到 `/health` 200） | ~2 min | ~3.5 min |
| KV blocks | 7260（`Available KV cache memory: 39.00 GiB`） | 5890（32 GiB 硬 pin；三改去掉 pin 后会按 0.94 自动推导，需重测） |
| 524288 token 请求的并发上限 | 1.77× | 1.44× |
| HBM/卡 | 85.7–86.4 GB | 80.0–80.7 GB |

### 环境变量（2026-11-11 三改：与 4m 站对齐，去掉 `spawn`）

| 段 | 内容 |
|---|---|
| 网络 | `nic_name=enp39s0f3` / `local_ip` + `VLLM_HOST_IP`、`HCCL_IF_IP`、`GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME` |
| 引擎 | `VLLM_USE_V2_MODEL_RUNNER` / `HCCL_OP_EXPANSION_MODE=AIV` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000` / `HCCL_BUFFSIZE=400` / `OMP_PROC_BIND=false` / `OMP_NUM_THREADS=10` / `PYTORCH_NPU_ALLOC_CONF` / `ASCEND_RT_VISIBLE_DEVICES=0-7` / `PYTHONUNBUFFERED` / `PYTHONHASHSEED` |
| 库路径 | `LD_LIBRARY_PATH` 追加 mooncake 与 `/usr/local/lib`；`PYTHONPATH` **追加** `/workspace`（不可覆盖） |
| Mooncake | `MOONCAKE_CONFIG_PATH` / `MOONCAKE_MASTER=10.222.46.181:50088` / `MOONCAKE_GLOBAL_SEGMENT_SIZE=24GB` |

精简掉的是镜像 ENV 里本来就有的（逐项比对过）：3 处 `set_env.sh` sourcing、`CPLUS_INCLUDE_PATH`、
`SOC_VERSION`、`TASK_QUEUE_ENABLE`、9 个 `ATB_*`、`LCCL_*`。
2026-11-11 三改起，P/D 都与 4m 站一致，**不设** `VLLM_WORKER_MULTIPROC_METHOD`（走 vLLM 默认 `fork`）。
无论 fork 还是 spawn，`PYTHONPATH` 都必须追加式（整体覆盖会冲掉镜像注入的 acl 路径，
即 2026-11-10 `No module named 'acl'` 事故），这条不要改。

若编译 bishengir / 静态 kernel 报找不到编译器，再把
`source .../ascend-toolkit/set_env.sh` 与 `source .../nnal/atb/set_env.sh --cxx_abi=1` 加回。

### 与 4m 站差异速查

| 项 | 本站（2m） | 4m 站 |
|---|---|---|
| 并行 | 每侧 **TP8**（单机） | 每侧 **TP16**（`nnodes 2` + headless rank1） |
| 权重 | W8A8C8-mxfp8 + `--quantization ascend` | bf16 `IQuest-Q1` |
| 权重路径 | 容器内 `/models/*` | SFS 绝对路径 |
| P 侧 prefix caching | 开 | 关（命令里 `--no/--enable` 重复，后者生效） |
| 角色脚本 | `prefill/p.sh` `decode/d.sh` | `prefill/p0.sh,p1.sh` `decode/d0.sh,d1.sh` |
| 环境变量 | 同一套精简口径 + `spawn` | 20 行 export，无 spawn |

## 8. 已知坑

| 症状 | 根因 | 处置 |
|---|---|---|
| 引擎根本没起，日志只有 `mkdir: cannot create directory '/pd/logs/prefill': File exists` | 旧容器 `/pd` 是 farm，`/pd/logs/<role>` 可能是指向**已删除目录**的悬空软链；`mkdir -p` 遇悬空链接返回 EEXIST，被 `set -e` 中断（2026-11-10 21:04 实踩） | 容器内 `ln -sfn <本目录>/logs/<role> /pd/logs/<role>`；脚本已改成 `mkdir -p … \|\| echo WARN`，不再中断 |
| spawn 阶段 `ModuleNotFoundError: No module named 'acl'`，EngineCore 卡死 | `export PYTHONPATH=/workspace` 整体覆盖，冲掉镜像/toolkit 注入的 acl 路径 | `p.sh`/`d.sh` 已改追加式 `${PYTHONPATH:-}:/workspace`；**`docker_start.sh` 里也不能加 `-e PYTHONPATH=/workspace`**（2026-11-11 11:39 实踩：容器级 -e 覆盖镜像 PYTHONPATH，导致 P/D 两个引擎 spawn 全挂），该行已删除 |
| decode 起不来，pydantic 报 "no config.json" | MTP 草稿 `/models/IQuest-Q1/mtp` 不在容器里 | `docker_start.sh` 已直接挂载；旧容器手工 `ln -sfn /mnt/sfs_turbo/models/IQuest-Q1 /models/IQuest-Q1` |
| 有 `vllm serve` 进程但 7100 不监听、NPU 不占卡 | 假绿：上次启动死在 spawn 阶段的残留僵尸 | `pkill -KILL vllm; pkill -KILL VLLM`，确认 `proc-mem` 为 "No process" 再重启 |
| 端口莫名被占 | 同机多容器共享 host 网络端口表（7100/1999/50088/28010/29010） | 起引擎前 `ss -tlnp`；用 `cat /proc/<pid>/cgroup` 确认 pid 归属哪个容器 |
| 清理命令把自己会话杀了 | 内联 `pkill -f 'VLLM::\|vllm serve'` 时模式串命中执行命令的 shell | 用脚本文件执行，或按 comm 匹配（`pkill -9 VLLM` 不带 `-f`） |
| 日志时间差 8 小时 | 容器内 UTC，宿主 CST+8 | 换算后再对时间线 |
| `logs/` 里没内容 | 引擎日志直打屏不落盘（proxy/master 例外） | 拉起时重定向到 `$D/logs/...` |

## 9. 取值口径与旧值

当前（2026-11-11 三改，对齐 4m 站）：两侧 `--max-model-len 524288`（等于模型原生 `max_position_embeddings`）；
P `batched-tokens 32768 / seqs 32`，D `batched-tokens 256 / seqs 64`；P 侧补 `--async-scheduling`。

历史口径（需要时改回，直接编辑对应行）：

| 阶段 | P | D |
|---|---|---|
| 4.4/4.5 纯P/纯D 配比测试冻结值 | `seqs 8 / batched 32768 / model-len 112024` | `seqs 6 / batched 192 / model-len 131072` |
| 生产值（更早） | `seqs 32 / batched 8192 / model-len 131072` | `seqs 256 / batched 1024` |
| 11-10 二改 | `seqs 16 / batched 32768` | `seqs 64 / batched 1024` + `--kv-cache-memory 32GiB` |

⚠️ 三改后 D 侧**不再 pin** `--kv-cache-memory`，KV 容量由 `--gpu-memory-utilization 0.94` 推导
（旧口径 32 GiB pin 时为 5890 块）；调大 `seqs` 或 `model-len` 时先看启动日志里的
`Maximum concurrency for 524,288 tokens`。

## 10. 变更记录

- **2026-11-11 15:03 — decode 图捕获上限 256 → 128（仅改脚本，服务未重启）**
  - `decode/d.sh`：`cudagraph_capture_sizes` 由 17 档
    `[1,2,4,8,16,24,32,48,64,80,96,112,128,160,192,224,256]` 截断为 13 档
    `[1,2,4,8,16,24,32,48,64,80,96,112,128]`，`cudagraph_mode` 仍为 `FULL_DECODE_ONLY`
  - 影响：单步 token 数 >128 的批（如 `max-num-seqs 64` × MTP depth 3 满载 = 256）会回落到
    eager 路径；高并发时段 TPOT 会有抬升，低并发不受影响。若后续要跑满 64 并发，先测这一档
- **2026-11-11 14:55 — 三改：启动命令与 4m 站逐项对齐（仅改脚本，服务未重启）**
  - `prefill/p.sh`：`--max-num-seqs 16 → 32`；`--additional-config` 增补
    `enable_flashcomm1:true`、`enable_cpu_binding:true`、`multistream_overlap_shared_expert:true`；
    新增 `--async-scheduling`；删除 `export VLLM_WORKER_MULTIPROC_METHOD=spawn`（走 vLLM 默认 `fork`，
    与 4m 的 `p0.sh`/`d0.sh` 一致）
  - `decode/d.sh`：`--max-num-batched-tokens 1024 → 256`；删除 `--kv-cache-memory 34359738368`
    （4m 注释为“不需要指定”，改由 `--gpu-memory-utilization 0.94` 自动推导）
  - 未对齐项（有意保留）：`--quantization ascend`（4m 是 bf16 无此参数）、TP8 vs TP16、
    P `gpu-memory-utilization 0.90`、D 13 档 capture sizes（1…128）、`--profiler-config` 输出目录
  - 影响：**运行中的 P/D 引擎仍是旧配置**，需按 §4 重启后生效；重启前先看
    `--kv-cache-memory` 取消 pin 后 D 侧 `Available KV cache memory` 是否满足 524288 并发
- **2026-11-11 11:44 — 迁移到新机器并成功拉起**
  - 站点迁到 **P=`10.222.46.181` (sjsbg-00004) / D=`10.222.47.74` (sjsbg-00003)**；
    `docker_start.sh`（P_IP/D_IP）、`p.sh`/`d.sh`（`local_ip` + `--host`）、
    `run_proxy.sh`（prefiller/decoder）、`mooncake/mooncake.json`（master=46.181:50088）全部改字面值
  - 旧容器（11-10 10:34 创建，`/pd` 仍绑在已删除的 `deploy/pd-mooncacke-mxfp8-1p1d-2m` 旧路径上）
    引擎为僵尸态、无端口监听，已 `docker rm -f` 后用 `docker_start.sh` 重建（`/pd` = 本目录真实 bind）
  - **首次拉起失败**：`docker_start.sh` 的 `-e PYTHONPATH=/workspace` 覆盖镜像 PYTHONPATH →
    spawn worker `No module named 'acl'`；删除该行并重建容器后正常（见 §8）
  - 实测：P ~2.5 min 就绪（KV 39.00 GiB / 929,280 tokens / 1.77×）、D ~4 min 就绪
    （753,920 tokens / 1.44×）；proxy `/healthcheck` 200；冒烟 57 token，冷启动 30 s、二次 0.59 s
  - 日志：`logs/prefill|decode/engine-20261111_114023.log`、`logs/proxy/proxy.log`
  - 旧机器 42.240 / 41.88 现被 4m 站占用，本站不再使用
  - 文档明确：**`/workspace` 挂载的是 vllm-ascend 代码仓**（示例路径
    `/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/vllm-ascend`）；当前 `SFS_REPO` 指向该仓的
    `releases/v0.30.0` worktree `pd-mooncacke-20261110`（同提交 `0b2829930` + 3 个本地部署补丁）
- **2026-11-10 21:07 — 按新配置成功拉起并验证**
  - P `--max-num-seqs 16`；D `--max-num-batched-tokens 1024 --max-num-seqs 64`；两侧 `--max-model-len 524288`
  - 实测：P ~2 min 就绪、D ~3.5 min 就绪；冒烟 1.6 s / 57 token；HBM P 88 GB、D 82 GB
  - 日志：`logs/prefill|decode/engine-20261110_210721.log`、`logs/proxy/proxy.log`
- **2026-11-10 21:04 — 修复 `/pd/logs` 悬空软链导致的启动中断**（`mkdir` 已改容错；farm 三处改指本目录 `logs/`）
- **2026-11-10 20:4x — 目录整改**
  - `launch_static.sh` → `p.sh`/`d.sh`；`run_proxy.sh` 去 `env.sh` 依赖改字面值；
    `docker_start.sh` 重写为角色参数式（旧 bare 版归档）；`run_master.sh` 日志改绝对路径
  - 环境变量按 4m 站口径精简（30 → 21 行 export）
  - 旧编排层 / 一次性脚本 / 压测脚本 / 历史文档移入 `archive/`；嵌套重复目录收进
    `archive/dup-snapshot-20261109/`（确认无用可 `rm -rf`）
  - `p.sh`/`d.sh` 开头补入残留清理块（与 4m 站同款，`set +e` 包裹）
- 11-10 之前的历史见 `archive/docs/DEPLOY.完整归档.md`（部署步骤 + 配置归档）与
  `archive/docs/REPORT.prof沙盒.md`（性能优化验收）。
