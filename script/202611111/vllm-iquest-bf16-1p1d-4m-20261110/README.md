# IQuest-Q1 PD 分离 · 1P1D 四机（Mooncake 池化）部署包

2026-11-10 归档。本包是**手动拉起用的全字面值脚本集**（`bash` 即可，无 `start.sh` 编排层）。
引擎代码走容器内 `/workspace`（宿主 `vllm-ascend-prof`），权重走 SFS 绝对路径，**不依赖
`/models/*` 符号链接**。

> 权威历史 runbook（双机版、含故障排查实踩记录）见
> `../pd-mooncacke-20261109/DEPLOY.md`。本文只描述 4 机站的口径，两者差异处**以本文为准**。

---

## 1. 包内容

| 文件 | 作用 |
|---|---|
| `docker_start.sh` | 建角色容器（`--rm`、`--net=host`、`--shm-size=64g`、8 张 davinci、PID1 = `sleep infinity`） |
| `mooncake/mooncake.json` | Mooncake 池配置，`master_server_address = 10.222.42.240:50088`（**权威**） |
| `mooncake/run_master.sh` | 起 `mooncake_master --port 50088`（`nohup ... >> mooncacke.log`，文件名少一个 c 是原样保留） |
| `prefill/p0.sh` `prefill/p1.sh` | prefill 组（kv_producer，TP16 双机 rank0/rank1） |
| `decode/d0.sh` `decode/d1.sh` | decode 组（kv_consumer，TP16 双机 rank0/rank1，MTP + 图模式） |
| `run_proxy.sh` | 负载均衡 proxy（`:1999`，P/D 各指向 rank0） |
| `prefill/.p0.sh.swp` | vim 残留 swap，非必需，可直接删（其缓冲区比现行 `p0.sh` 多一段 `ATB_*` 环境变量，见 §7） |

## 2. 拓扑与端口

每台 8×Ascend 950DT，一个角色用**两台机 TP16**（`--nnodes 2`，rank0 先起、rank1 带 `--headless`）。

| 角色 | rank0（head node） | rank1（headless） | KV 端口 | engine_id |
|---|---|---|---|---|
| prefill（kv_producer） | `10.222.43.66` sjsbg-00001 · `prefill/p0.sh` | `10.222.44.142` sjsbg-00006 · `prefill/p1.sh` | 28010 | 0 |
| decode（kv_consumer） | `10.222.42.240` sjsbg-00002 · `decode/d0.sh` | `10.222.41.88` sjsbg-00008 · `decode/d1.sh` | 29010 | 1 |

| 端点 | 位置 |
|---|---|
| API `:7100` | 各 rank0（rank1 headless 不监听，属正常） |
| torch rendezvous `:29563` | 每对 rank0（P 组 43.66 / D 组 42.240） |
| `mooncake_master :50088` | **仅 42.240**（与 decode rank0 同机），必须在引擎之前起来 |
| proxy `:1999` | `43.66`，prefiller→`43.66:7100`，decoder→`42.240:7100` |

数据面：
```
client → 43.66:1999 ─┬─► prefill 43.66/44.142 (TP16, :7100)
                      └─► decode  42.240/41.88 (TP16, :7100, MTP depth3)
                              ▲ Mooncake KV 实时传输（master 42.240:50088）
```

> ⚠️ 上级 `../vllm-iquest-bf16/env.sh` 里 `P0_IP=42.240 / D0_IP=43.66` 的角色标注与本包
> **相反**（2 机时代遗留）。判断 P/D 以 `run_proxy.sh` 与各脚本 body 为准。

## 3. 首次准备（容器不存在时）

`docker_start.sh` 第 20 行 `source "$DEPLOY_DIR/env.sh"`，而**本包未附带 `env.sh`**，直接跑会
`set -e` 退出。先补一份：

```bash
cd /mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/vllm-iquest-bf16-1p1d-4m-20261110
cp ../vllm-iquest-bf16/env.sh ./
```

然后**每台显式传角色**（不要依赖自动识别：`env.sh:get_node()` 只认 P0=42.240/D0=43.66，
在这套 IP 下会把容器名起反）：

```bash
# prefill 组
ssh root@10.222.43.66  'cd /mnt/sfs_turbo/.../vllm-iquest-bf16-1p1d-4m-20261110 && bash docker_start.sh p'
ssh root@10.222.44.142 'cd ... && bash docker_start.sh p'
# decode 组
ssh root@10.222.42.240 'cd ... && bash docker_start.sh d'
ssh root@10.222.41.88  'cd ... && bash docker_start.sh d'
```

容器名：`iquest-pd-mooncacke-p`（43.66 / 44.142）、`iquest-pd-mooncacke-d`（42.240 / 41.88）。
挂载约定（脚本内所有路径都是 `/pd/...` 与 `/mnt/sfs_turbo/...`）：

| 宿主 | 容器内 |
|---|---|
| **本包目录** | `/pd` ← 必须挂这个，否则 `/pd/mooncake/mooncake.json`、`/pd/logs/*` 会落到别的部署目录 |
| `.../iq-vllm-bf16/vllm-ascend-prof` | `/workspace`（`PYTHONPATH` 追加） |
| `/mnt/sfs_turbo` | `/mnt/sfs_turbo`（权重与 MTP 草稿都从这里取） |

容器是 `--rm` 的：`docker stop/rm -f` 后需按上面重建。

验证代码树：

```bash
docker exec iquest-pd-mooncacke-p bash -c \
  'PYTHONPATH=/workspace python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"'
# 期望: .../vllm-ascend-prof/vllm_ascend/__init__.py
```

## 4. 拉起顺序（master → decode → prefill → proxy）

每个脚本开头都会 `pkill -9 python/vllm/ray` 并 `sleep 10`，因此：

- **proxy 必须最后起**（它在 43.66 上是 python，会被 `p0.sh` 杀掉）；
- 同一台机上不要并发跑两个脚本；
- rank0 必须先于 rank1（rendezvous 在 rank0 的 `29563`）。

```bash
# ① master（42.240，引擎初始化即连 50088，必须先起）
ssh root@10.222.42.240 'docker exec -d iquest-pd-mooncacke-d bash /pd/mooncake/run_master.sh'
ssh root@10.222.42.240 'ss -ltnp | grep 50088'        # 有监听才继续

# ② decode rank0 → ③ decode rank1（含 25 档图捕获 + MTP，量级 4 分钟）
ssh root@10.222.42.240 'docker exec -d iquest-pd-mooncacke-d bash /pd/decode/d0.sh'
ssh root@10.222.41.88  'docker exec -d iquest-pd-mooncacke-d bash /pd/decode/d1.sh'

# ④ prefill rank0 → ⑤ prefill rank1（量级 2.5 分钟，SFS 页缓存热时）
ssh root@10.222.43.66  'docker exec -d iquest-pd-mooncacke-p bash /pd/prefill/p0.sh'
ssh root@10.222.44.142 'docker exec -d iquest-pd-mooncacke-p bash /pd/prefill/p1.sh'

# ⑥ proxy
ssh root@10.222.43.66  'docker exec -d iquest-pd-mooncacke-p bash /pd/run_proxy.sh'
```

日志**直打屏不落盘**：`docker exec -d` 的输出会被丢弃。要看引擎输出就用不带 `-d` 的前台 exec
（占终端，Ctrl+C 会连引擎一起杀）。

## 5. 探活与冒烟

```bash
curl -sf http://10.222.43.66:7100/health    && echo P_OK      # prefill rank0
curl -sf http://10.222.42.240:7100/health   && echo D_OK      # decode  rank0
curl -sf http://10.222.43.66:1999/healthcheck && echo X_OK    # proxy（是 /healthcheck，不是 /health）
curl -s  http://10.222.43.66:1999/v1/models                   # 后端全挂时会返回
                                                              # {"error":"All backend instances failed..."}

# PD 全链路（proxy → P 预填 → KV 传输 → D 生成 + MTP）
curl -s http://10.222.43.66:1999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"你好"}],"max_tokens":256}'
# 思维链在 message.reasoning（iquest_q1 parser），正文在 message.content
```

## 6. 停止与清理

```bash
# 引擎（各机自己容器内）。必须带大写 VLLM:: 才杀得到 worker 的 proctitle
docker exec <容器> bash -c "pkill -TERM -f 'vllm serve'; sleep 8; pkill -KILL -f 'VLLM::|vllm serve'"
# proxy / master（43.66 上 proxy，42.240 上 master）
docker exec iquest-pd-mooncacke-p pkill -f load_balance_proxy_server_example
docker exec iquest-pd-mooncacke-d pkill -f '[m]ooncake_master'

# 宿主机确认显存已释放，再到 "No process" 之后才重启
npu-smi info -t proc-mem -i 0
```

**假绿**：进程列表里有 `vllm serve`、但 `:7100` 无监听且 NPU 不占卡，是上次启动死在 spawn
阶段的残留僵尸（Ctrl+C 只杀了会话）。必须先清干净再重启。

## 7. 关键配置差异（P vs D）

| 项 | prefill（`p0/p1.sh`） | decode（`d0/d1.sh`） |
|---|---|---|
| KV 角色 | `kv_producer`，port 28010，engine_id 0 | `kv_consumer`，port 29010，engine_id 1 |
| 连接器 | `MultiConnector`[`MooncakeHybridConnector` + `AscendStoreConnector`(backend=mooncake)] | 同左（consumer 侧） |
| `max-model-len` | 524288 | 524288 |
| batching | `--max-num-batched-tokens 32768 --max-num-seqs 8` | `--max-num-batched-tokens 192 --max-num-seqs 6` |
| `block-size` | 128 | 128 |
| 图模式 | `--enforce-eager` | `npugraph_ex` + `FULL_DECODE_ONLY`（25 档 capture sizes）+ `--async-scheduling` |
| 投机解码 | 无 | `eagle`，`/mnt/sfs_turbo/models/IQuest-Q1/mtp/`，depth 3 |
| 其他 | — | `--kv-cache-memory 34359738368`、`recompute_scheduler_enable`、`multistream_overlap_shared_expert`、`enable_cpu_binding` |
| 共同 | TP16 / `nnodes 2` / `--enable-expert-parallel` / `--gpu-memory-utilization 0.94` / `--disable-hybrid-kv-cache-manager` / `enable_fused_mc2`+`enable_iquest_triton_qk_norm`+`enable_fused_o_proj_ar` / profiler 输出到 `/pd/logs/profile/{prefill,decode}` | 同左 |

## 8. 已知问题 / 待修（按影响排序）

1. **包内缺 `env.sh`** → `docker_start.sh` 起不来（见 §3 的 `cp`）。
2. **`env.sh:get_node()` 角色映射是反的** → 建容器必须显式 `bash docker_start.sh p|d`。
3. **prefix caching 自相矛盾**：`prefill/p0.sh` 同时有 `--no-enable-prefix-caching`（L72）和
   `--enable-prefix-caching`（L80），argparse 后者生效 → 实际是**开**；而 decode 侧是关。
   需定一个口径后删掉多余那行。
4. **脚本头部注释是旧拷贝**：文中的 `max-model-len 112024 / 131072`、`launch_static.sh`、
   容器 `iquest-pd-p78/d78`、"P0(new2 42.240)" 均与 body 不符（body 才是真实值）。
5. **`MOONCAKE_MASTER` / `MOONCAKE_GLOBAL_SEGMENT_SIZE` 设置不齐**（只有 `d1.sh` 有；
   `p0.sh` 有后一个）。实际以 `mooncake.json` 为准，暂不影响，但建议统一。
6. `prefill/.p0.sh.swp` 里的 `ATB_*` 环境变量（`ATB_CXX_ABI`、`ATB_MATMUL_SHUFFLE_K_ENABLE` 等）
   现行 `p0.sh` **没有**，确认是否漏配。

## 9. 现场状态快照（2026-11-10 20:10 CST，非干净状态）

| 机器 | 容器 | 引擎 | 备注 |
|---|---|---|---|
| 43.66 | `iquest-pd-mooncacke-p` | 无 | 僵尸 `VLLM::Worker_TP*` 占 `:29563`/`:28010`，HBM 54G/73G 未释放；`/pd` 挂的是 `../vllm-iquest-bf16`（不是本包）；`:1999` proxy 起着但 P/D 后端全 dead；`:50088` 上另有一个 master（属别的部署） |
| 44.142 | `iquest-pd-mooncacke-p` | prefill rank1 存活，等 rank0 | 正常 headless，无 `:7100` |
| 42.240 | `iquest-pd-p78` + `pd-mooncacke-p`(shm 1g) | 无 | 僵尸 worker 占 `:29563`，HBM 78G；`/pd` 挂的是 `../pd-ration/deploy`；master `:50088` 正常 |
| 41.88 | `iquest-pd-d78` | decode rank1 进程存活 | 本机 8 卡 HBM 全 4.7G 空闲 → 未吃到权重，属假绿 |

结论：当前是**两个半截启动**（P 缺 rank0、D 缺 rank0），对外不可用。下次拉起前按 §6
清干净，并按 §3 + §4 重建容器（把本包挂成 `/pd`）后再起。
