# 部署与配置归档 — 1P1D 双机 PD 分离（mooncacke 站，2026-11-10）

> 本文件 = 部署步骤 runbook + 生效配置完整归档（原
> `PD_mooncacke_1p1d_配置归档_20261110.md` 并入，2026-11-10 起 DEPLOY.md 为唯一部署文档）。
> 背景/架构见同目录 `README.md`（早期拓扑时代文档，IP 等以本文为准）。
> 当前实测拓扑：**P0=new2 10.222.42.240 / D0=new8 10.222.41.88**，容器
> `pd-mooncacke-p` / `pd-mooncacke-d`，代码 `releases/v0.30.0`（vllm_ascend 经
> PYTHONPATH 从 /workspace 加载）。

## 0. 拓扑与约定

```
client ──► proxy:1999(new2) ──► prefill:7100(new2, kv_producer 28010)
                                   │ Mooncake 实时 KV 传输
                                   ▼
                              decode:7100(new8, kv_consumer 29010) + MTP depth3
池化: mooncake_master:50088(new2, 容器 pd-mooncacke-p 内)
```

| 项 | 值 |
|---|---|
| 机器 | P0=new2 `10.222.42.240`，D0=new8 `10.222.41.88`（8×Ascend 950DT，host 网络） |
| 容器 | `pd-mooncacke-p`（new2）/ `pd-mooncacke-d`（new8），镜像 `quay.io/ascend/vllm-ascend:v0.30.0-a5-openeuler`，**--rm** |
| 部署目录 | 本目录，SFS 共享；容器内经符号链接 farm 挂为 `/pd` |
| 引擎代码 | `/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/pd-mooncacke-20261110`（git 分支 `releases/v0.30.0` @5a1e74079）→ 容器内 `/workspace` |
| 模型 | 主模型 `/mnt/sfs_turbo/models/IQuest-Q1-W8A8C8-mxfp8`；MTP 草稿 `/mnt/sfs_turbo/models/IQuest-Q1/mtp`（仅 D 侧需要） |
| 日志 | **直打屏不落盘**（launch_static.sh 无重定向；`docker exec -d` 启动 = 输出丢弃） |
| 时区坑 | 容器内日志时间戳为 UTC，宿主机 CST+8（日志 07:xx = 下午 15:xx） |

## 1. 一次性准备（容器不存在时）

容器是 `--rm` 的，被删后按此重建。两种方式任选：

**方式 A（现状，docker_start.sh 裸容器 + 手工符号链接）**

```bash
# new2 上（new8 把 NAME 改成 pd-mooncacke-d）:
NAME=pd-mooncacke-p bash docker_start.sh   # 脚本内 IMAGE/NAME 为环境变量

# 容器内补符号链接（两台都要；D 侧多一条 /models/IQuest-Q1）:
docker exec pd-mooncacke-p bash -c '
  ln -sfn /mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/pd-mooncacke-20261110 /workspace
  ln -sfn /mnt/sfs_turbo/models/IQuest-Q1-W8A8C8-mxfp8 /models/IQuest-Q1-W8A8C8-mxfp8
  # /pd 符号链接 farm: 部署目录每个文件/子目录逐条 ln -s 到 /pd/（/pd/logs 建真实目录)
'
```

**方式 B（`container_mooncacke.sh` 正规容器）**：自带 /mnt、/models、/pd、/workspace
挂载与 64g shm（容器名 `iquest-pd-mooncacke-p/d`，与 start.sh 的 p78/d78 名不一致，
用方式 B 时启动命令里的容器名要对应改）。注意其 /workspace 固定挂 `vllm-ascend-prof`，
需重指到 `pd-mooncacke-20261110`（releases/v0.30.0）。

**验证代码树**（两台都做）：

```bash
docker exec pd-mooncacke-p bash -c \
  'PYTHONPATH=/workspace python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"'
# 必须输出 .../pd-mooncacke-20261110/vllm_ascend/__init__.py
```

## 2. 启动（顺序：master → decode → prefill → proxy）

```bash
# ① master（仅 new2；池化 ENABLE_KV_POOL=1 的前置条件。探活 50088 端口）
ssh root@10.222.42.240 'docker exec -d pd-mooncacke-p bash /pd/mooncake/run_master.sh'

# ② decode（new8；~4 分钟就绪，含 25 档图捕获 + MTP）
ssh root@10.222.41.88 'docker exec -d pd-mooncacke-d bash /pd/decode/launch_static.sh'

# ③ prefill（new2；~2.5 分钟就绪，SFS 页缓存热时）
ssh root@10.222.42.240 'docker exec -d pd-mooncacke-p bash /pd/prefill/launch_static.sh'

# ④ proxy（new2）
ssh root@10.222.42.240 'docker exec -d pd-mooncacke-p bash /pd/run_proxy.sh'
```

要看引擎输出就用**不带 -d 的前台 exec**（占终端，Ctrl+C 会连引擎一起杀）；
`-d` 后台启动时输出被丢弃（脚本不落盘）。

## 3. 探活与验证

```bash
curl -sf http://10.222.42.240:7100/health   && echo P_OK    # prefill
curl -sf http://10.222.41.88:7100/health    && echo D_OK    # decode
curl -sf http://10.222.42.240:1999/healthcheck && echo X_OK # proxy（注意是 /healthcheck）

# PD 全链路冒烟（proxy → P 预填 → KV 传输 → D 生成）
curl -s http://10.222.42.240:1999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"你好"}],"max_tokens":256}'
# 思维链在 message.reasoning（iquest_q1 parser），正文在 message.content
```

## 4. 停止

```bash
# 引擎（两台各自容器内；必须带大写 VLLM:: 才能杀到 worker 的 proctitle）
docker exec pd-mooncacke-p bash -c "pkill -TERM -f 'vllm serve'; sleep 8; \
  pkill -KILL -f 'VLLM::|vllm serve'"
docker exec pd-mooncacke-d bash -c "pkill -TERM -f 'vllm serve'; sleep 8; \
  pkill -KILL -f 'VLLM::|vllm serve'"
# proxy / master（new2）
docker exec pd-mooncacke-p pkill -f load_balance_proxy_server_example
docker exec pd-mooncacke-p pkill -f '[m]ooncake_master'
# 确认 NPU 已释放再重启（宿主机: npu-smi info -t proc-mem -i 0 应显示 No process）
```

## 5. 故障排查（2026-11-10 实踩三坑）

| 症状 | 根因 | 处置 |
|---|---|---|
| worker spawn 阶段 `ModuleNotFoundError: No module named 'acl'`，EngineCore 卡死 | `export PYTHONPATH=/workspace` 整体覆盖冲掉了 CANN set_env 注入的 toolkit python 路径（spawn worker 里 camem.py `from acl.rt import memcpy` 失败，事故现场 P 侧 14:44 CST） | launch_static.sh 已改为追加式 `${PYTHONPATH:-}:/workspace`；若再见到覆盖式写法即回退此坑 |
| decode 启动 pydantic 报 "no config.json" | D 侧 eagle 草稿 `/models/IQuest-Q1/mtp` 不存在（容器 /models 只链了量化主模型） | 补 `ln -sfn /mnt/sfs_turbo/models/IQuest-Q1 /models/IQuest-Q1`（容器内） |
| 进程列表有 `vllm serve` 但 7100 无监听、NPU 无占卡 | **假绿**：上次启动死在 spawn 阶段的残留僵尸（Ctrl+C 只杀了会话没杀引擎） | 先 `pkill -KILL -f 'VLLM::|vllm serve'` 清干净、确认 NPU 释放，再重启 |

其他注意：
- 同机多容器 host 网络共享端口表（7100/1999/50088/28010/29010），起引擎前先 `ss -tlnp` 查占用。
- 本站可能与其它会话并发操作：动引擎前先看进程，杀进程前确认 pid 归属。
- master 必须先于引擎启动（引擎初始化即连 50088）。
- 容器 `/pd/logs` 是容器层真实目录（--rm 即丢），引擎日志又不落盘 → 重要日志在打屏
  终端里重定向留存。

## 6. 生效配置完整归档（2026-11-10 15:00–15:15 实测，以进程实际命令行为准）

### 6.1 mooncake_master（P0，new2）

```text
/usr/local/python3.12.16/.../mooncake/mooncake_master \
  --port 50088 --eviction_high_watermark_ratio 0.9 \
  --eviction_ratio 0.1 --default_kv_lease_ttl 11000
```
- 由 `bash /pd/mooncake/run_master.sh` 拉起；配置 `/pd/mooncake/mooncake.json`：
  master_server_address=10.222.42.240:50088、global_segment_size=24GB、
  metadata_server=P2PHANDSHAKE、protocol=ascend
- ENABLE_KV_POOL=1（env.pool.override.sh）；池化启动顺序 master → decode → prefill → proxy

### 6.2 Prefill（new2:7100）

启动：`docker exec -d pd-mooncacke-p bash /pd/prefill/launch_static.sh`

| 项 | 值 |
|---|---|
| 模型 | `/models/IQuest-Q1-W8A8C8-mxfp8` + `--quantization ascend`（W8A8，**KV=float，C8 未启用**） |
| 并行 | TP8 × DP1 + `--enable-expert-parallel` |
| 模式 | `--enforce-eager`（P 侧无图模式） |
| 上下文 | `--max-model-len 112024`（110k/1k 口径） |
| 批次 | `--max-num-batched-tokens 32768`、`--max-num-seqs 8` |
| block_size | 128 |
| 显存 | `--gpu-memory-utilization 0.90` |
| 缓存 | `--enable-prefix-caching`（池查找依赖）+ `--disable-hybrid-kv-cache-manager` |
| KV 传输 | MultiConnector(producer, kv_port 28010, engine_id 0) = MooncakeHybridConnector + AscendStoreConnector(backend mooncake, use_layerwise=false) |
| additional_config | `{"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1}` |
| profiler | torch → `/pd/logs/profile/prefill` |

### 6.3 Decode（new8:7100）

启动：`docker exec -d pd-mooncacke-d bash /pd/decode/launch_static.sh`

| 项 | 值 |
|---|---|
| 模型 | 同 P（W8A8C8-mxfp8，KV=float） |
| 并行 | TP8 + EP8 |
| 模式 | 图模式 **FULL_DECODE_ONLY**，cudagraph_capture_sizes 25 档 `[1,2,4,8,16,24,32,48,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024]`；`ascend_compilation_config.enable_npugraph_ex=true, enable_static_kernel=false` |
| 上下文 | `--max-model-len 131072` |
| 批次 | `--max-num-batched-tokens 192`、`--max-num-seqs 6`（5890 blocks×128÷111,024≈6.79→6，88% 占用） |
| KV 显存 | 显式 pin `--kv-cache-memory 34359738368`（32GiB） |
| 显存 | `--gpu-memory-utilization 0.94` |
| MTP | `--speculative-config {"method":"eagle","model":"/models/IQuest-Q1/mtp","num_speculative_tokens":3}` |
| 调度 | `--async-scheduling`（on） |
| prefix caching | 关 |
| KV 传输 | MultiConnector(consumer, kv_port 29010, engine_id 1) = 同款两连接器 |
| additional_config | `{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1}` |
| profiler | torch → `/pd/logs/profile/decode` |

### 6.4 Proxy（P0，new2:1999）

启动：`docker exec -d pd-mooncacke-p bash /pd/run_proxy.sh`
- 镜像自带 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`
- 路由：prefill=10.222.42.240:7100，decode=10.222.41.88:7100

## 7. 配置速查与恢复生产值

当前为 **4.4/4.5 纯P/纯D 配比测试冻结值**。恢复生产值（脚本头部注释原话）：
- P：`--max-num-seqs 32 --max-num-batched-tokens 8192 --max-model-len 131072`
- D：`--max-num-seqs 256 --max-num-batched-tokens 1024`
- 同步清理 `env.override.sh` 尾部"4.4/4.5 配比测试"冻结块（PREFILL_MAX_NUM_SEQS=8 等），
  否则 launch_engine.sh 路径启动会与 launch_static.sh 不一致。

切换引擎代码：改容器内 `/workspace` 符号链接指向目标 worktree → 重启引擎（脚本不用动）。

## 8. 另存：当天上午的手动配置变体（11:48 CST 成功跑通）

容器本地 `engine-current.log`（UTC 03:48）记录的一次成功启动，与上文冻结配置
**不是一套**，注意区分：
- 端口 7200；max_model_len 65535；max_num_batched_tokens 16384；max_num_seqs 64；block_size **512**
- **KV C8 启用**：`--kv-cache-dtype mxfp8 --kv-cache-dtype-skip-layers sliding_window`
- 连接器 **MooncakeConnectorV1**（非 MooncakeHybridConnector）+ AscendStoreConnector
- additional_config 含 `enable_flashcomm1`、`enable_cpu_binding`、`recompute_scheduler_enable`
- async_scheduling on；enforce_eager；env 出现过 VLLM_ASCEND_ENABLE_FLASHCOMM1 /
  VLLM_ASCEND_ENABLE_PREFETCH_MLP / VLLM_RPC_TIMEOUT（镜像报 unknown 警告）
- 走镜像自带 vllm_ascend（当时 /workspace 链接尚未建好）
