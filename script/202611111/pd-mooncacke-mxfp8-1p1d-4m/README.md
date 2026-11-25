# IQuest-Q1 PD 分离 · 1P1D 四机（DP2×TP8 + Mooncake MultiConnector）

4 机站：每角色 2 台机、每台 1 个 TP8 引擎，两台组成一个 **DP size=2** 组（16 卡共享 EP 组）。
`vllm serve` 参数**严格沿用** `../pd-mooncacke-mxfp8-1p1d-2m` 的 P/D 脚本（已验证口径），
只额外加了 4 个 DP 参数 + `external_online_dp` 启动器。

> **状态：2026-11-10 14:14 拉起成功并验证** —— 四台引擎全绿、proxy 全链路冒烟通过。
> `--kv-cache-dtype mxfp8`（C8）**当前未启用**，原因见 §8。
> 目录名里的 `mxfp8` 是原始目标，实际生效的是 float KV。

---

## 1. 拓扑

| 角色 | 节点 | DP rank | 容器 | 引擎端口 |
|---|---|---|---|---|
| prefill | `10.222.43.66` sjsbg-00001 | 0（DP master） | `iquest-mxfp8-p` | 8000 |
| prefill | `10.222.44.142` sjsbg-00006 | 1 | `iquest-mxfp8-p` | 8000 |
| decode | `10.222.42.240` sjsbg-00002 | 0（DP master） | `iquest-mxfp8-d` | 8000 |
| decode | `10.222.41.88` sjsbg-00008 | 1 | `iquest-mxfp8-d` | 8000 |

| 端点 | 位置 |
|---|---|
| `mooncake_master :50090` | P0 = 43.66（50088 被别的站容器 `iq-own` 占着，故改用 50090） |
| DP rendezvous | P 组 `43.66:16500`，D 组 `42.240:16600` |
| PD KV 端口 | P `28010` / D `29010`（`MultiConnector` 内层 `MooncakeHybridConnector`） |
| proxy `:8999` | P0 43.66，后端 = 2 个 P + 2 个 D 的 `:8000` |

```
                    ┌─ prefill 43.66 (dp0) :8000 ─┐   EP 组 = 16 卡（256 experts / top-8）
client → 43.66:8999 ┤                             ├─ DP2
        (proxy)     └─ prefill 44.142(dp1) :8000 ─┘
                          │ Mooncake KV 28010 → 29010
                    ┌─ decode 42.240 (dp0) :8000 ──┐
                    └─ decode  41.88 (dp1) :8000 ──┘  + EAGLE depth3 + npugraph
```

## 2. 目录结构

```
pd-mooncacke-mxfp8-1p1d-4m/
├── README.md
├── docker_start.sh              # bash docker_start.sh [p|d]
├── mooncake/{mooncake.json, run_master.sh}    # master = 43.66:50090
├── prefill/{run_dp_template.sh, p0.sh, p1.sh}
├── decode/ {run_dp_template.sh, d0.sh, d1.sh}
├── run_proxy.sh                 # 4 后端 :8999
└── logs/{prefill,decode,proxy,mooncake}/
```

启动链：`p0.sh` 导出 `MODEL_PATH=/models/IQuest-Q1-W8A8C8-mxfp8`、`LOCAL_IP`、`NIC_NAME`
→ `cd` 到角色目录 → `launch_online_dp.py`（`/workspace/examples/external_online_dp/`）
→ 每个本地 rank 调一次 `./run_dp_template.sh $可见卡 $端口 $dp_size $dp_rank $dp_addr $dp_rpc $tp`
→ `vllm serve`。launcher 硬编码 `./run_dp_template.sh`，**必须在角色目录下运行**。

## 3. 前置条件（本次拉起时踩过的）

**① NPU 设备是独占的** —— 一个 davinci 同一时刻只能被一个容器真正占用。新容器若报
`npu-smi: Failed to obtain the console log level`，就是同机其它容器还占着设备：

```bash
# 先看清谁占着，再删废弃容器（--rm 容器删了不影响宿主机数据）
docker ps --format '{{.Names}}' | while read c; do echo "$c $(docker inspect -f '{{len .HostConfig.Devices}}' $c)"; done
docker rm -f <废弃容器>
```

**② 43.66 的 50088 被 `iq-own` 容器里的 master 占着**（也占 9003 metrics）→ 本包 master 用
**50090 + metrics 9004**，不碰别人的实例。

**③ 42.240 / 41.88 原先跑着 2m 站服务**（`../pd-mooncacke-mxfp8-1p1d-2m`），本包要占用它们做 decode，
已停掉；2m 站要恢复需先停本包。

## 4. 拉起顺序（master → decode×2 → prefill×2 → proxy）

```bash
D=/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/pd-mooncacke-mxfp8-1p1d-4m
TS=$(date +%Y%m%d_%H%M%S)

# ① master（仅 P0；50090 空闲时）
ssh root@10.222.43.66 'docker exec -d iquest-mxfp8-p bash /pd/mooncake/run_master.sh'
ssh root@10.222.43.66 'ss -ltn | grep 50090'

# ② decode 两台
ssh root@10.222.42.240 "docker exec -d iquest-mxfp8-d bash -c 'bash /pd/decode/d0.sh >> $D/logs/decode/d0-${TS}.log 2>&1'"
ssh root@10.222.41.88  "docker exec -d iquest-mxfp8-d bash -c 'bash /pd/decode/d1.sh >> $D/logs/decode/d1-${TS}.log 2>&1'"

# ③ prefill 两台
ssh root@10.222.43.66  "docker exec -d iquest-mxfp8-p bash -c 'bash /pd/prefill/p0.sh >> $D/logs/prefill/p0-${TS}.log 2>&1'"
ssh root@10.222.44.142 "docker exec -d iquest-mxfp8-p bash -c 'bash /pd/prefill/p1.sh >> $D/logs/prefill/p1-${TS}.log 2>&1'"

# ④ proxy（最后，引擎清理块会 pkill python）
ssh root@10.222.43.66 "docker exec -d iquest-mxfp8-p bash -c 'bash /pd/run_proxy.sh >> $D/logs/proxy/proxy-${TS}.log 2>&1'"
```

实测：两台 decode 约 140 s 就绪、两台 prefill 约 130 s 就绪；**P 组和 D 组可以并行起**。

## 5. 探活与 curl 速查

**推理一律走 proxy `10.222.43.66:8999`**；`8000` 上四个实例是 DP rank（pd 角色
`kv_producer`/`kv_consumer`），只用来做 `/health`，直连发推理不走 PD 编排。

### 5.1 健康检查

```bash
# 四台引擎
for ip in 10.222.43.66 10.222.44.142 10.222.42.240 10.222.41.88; do
  printf '%-16s %s\n' "$ip" "$(curl -s -m 4 -o /dev/null -w '%{http_code}' http://$ip:8000/health)"
done

# proxy（是 /healthcheck，不是 /health；/health 会 404）
curl -s -o /dev/null -w 'proxy %{http_code}\n' http://10.222.43.66:8999/healthcheck
curl -s http://10.222.43.66:8999/v1/models | python3 -m json.tool | head -20

# mooncake master
ss -ltn | grep 50090
```

### 5.2 推理请求

```bash
# 普通请求（max_tokens 别小于 512，否则 token 全耗在 reasoning 上、content 为空）
curl -s http://10.222.43.66:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"用一句话介绍你自己"}],"max_tokens":512,"temperature":0}'

# 解析正文 / 思维链 / 用量
curl -s http://10.222.43.66:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":512,"temperature":0}' \
| python3 -c "import sys,json;d=json.load(sys.stdin);m=d['choices'][0]['message'];print('reasoning:',(m.get('reasoning') or '')[:100]);print('content  :',(m.get('content') or '')[:300]);print('usage    :',d['usage'])"

# 流式
curl -N -s http://10.222.43.66:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"写一个快排"}],"max_tokens":512,"temperature":0,"stream":true}'

# 计时（稳态约 0.8–1.0 s / 57 token；首请求 30 s 级是图捕获预热）
time curl -s -o /dev/null http://10.222.43.66:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"你好"}],"max_tokens":128,"temperature":0}'
```

思维链在 `message.reasoning`（`iquest_q1` parser），正文在 `message.content`。

## 6. 生效配置

### 6.1 vllm serve 参数（= `../pd-mooncacke-mxfp8-1p1d-2m` 的 p.sh / d.sh）

| 项 | prefill | decode |
|---|---|---|
| 模型 | `$MODEL_PATH` = `/models/IQuest-Q1-W8A8C8-mxfp8` + `--quantization ascend` | 同左 |
| 并行 | `--tensor-parallel-size $7`(=8) + `--enable-expert-parallel` + 4 个 `--data-parallel-*` | 同左 |
| 模式 | `--enforce-eager` | `--compilation-config FULL_DECODE_ONLY`（25 档）+ `--async-scheduling` |
| 上下文 | `--max-model-len 524288` | `--max-model-len 524288` |
| 批次 | `--max-num-batched-tokens 32768 --max-num-seqs 16` | `--max-num-batched-tokens 1024 --max-num-seqs 64` |
| block-size | 128 | 128 |
| 显存 | `--gpu-memory-utilization 0.90` | `0.94` + `--kv-cache-memory 34359738368` |
| prefix caching | 开 | 关 |
| 混合 KV 管理器 | `--disable-hybrid-kv-cache-manager`（两侧都传，同 2m 站） | 同左 |
| KV 传输 | `MultiConnector`：`MooncakeHybridConnector`(producer 28010, engine_id 0) + `AscendStoreConnector`(backend mooncake) | consumer 29010, engine_id 1，其余同左 |
| MTP | — | `eagle` + `/models/IQuest-Q1/mtp`，depth 3 |
| additional_config | `enable_fused_mc2` + `enable_iquest_triton_qk_norm` + `enable_fused_o_proj_ar` | 上面三项 + `ascend_compilation_config`(npugraph_ex) + `enable_cpu_binding` + `multistream_overlap_shared_expert` + `recompute_scheduler_enable` |
| profiler | torch → `/pd/logs/profile/prefill` | torch → `/pd/logs/profile/decode` |

与 2m 站的**唯一差异**：多 4 个 DP 参数（`--data-parallel-size/rank/address/rpc-port`），
`--host` 由具体 IP 改成 `0.0.0.0`（容器 host 网络，四个引擎都要被外部访问），
以及 `MooncakeHybridConnector` 里的 `prefill/decode: {dp_size: 2, tp_size: 8}`（本站实际是 DP2）。

### 6.2 引擎侧环境变量

参考模板的 DP/EP 版：`HCCL_ALGO=level0:fullmesh`、`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000`、
`HCCL_EXEC_TIMEOUT/CONNECT_TIMEOUT=1800`、`HCCL_BUFFSIZE=1200(P)/900(D)`、`HCCL_DFS_CONFIG`、
`PROMETHEUS_MULTIPROC_DIR`、`VLLM_USE_FASTOKENS=1`、`VLLM_DISABLE_COMPILE_CACHE=1`、
`VLLM_USE_V2_MODEL_RUNNER=1`（与 2m 站一致）、`ASCEND_LOCAL_COMM_RES`、
`MOONCAKE_CONFIG_PATH` / `MOONCAKE_MASTER=10.222.43.66:50090`、`PYTHONPATH` 追加 `/workspace`。

参考里有、本站未启用：`VLLM_ASCEND_ENABLE_FLASHCOMM1`、`VLLM_ASCEND_ENABLE_PREFETCH_MLP`、
`VLLM_ASCEND_DP_ALLREDUCE_ON_NPU`（模板注释里留着，需要时打开）。

## 7. 实测结果（2026-11-10 14:14）

| 项 | P0 / P1 | D0 / D1 |
|---|---|---|
| 就绪耗时 | ~130 s | ~140 s |
| DP 组 | world_size=16（DP2×TP8） | world_size=16 |
| Available KV cache | 57.65 GiB / 10731 blocks | 5890 blocks（32 GiB 硬 pin） |
| 524288 token 请求并发上限 | 2.62× | 1.44× |
| HBM/卡 | ~89.0 GB | ~63.6 GB |

冒烟：首请求 30.7 s（含图捕获/编译预热），稳态 **0.77–0.97 s / 57 token**；
返回 `content` 正常、`reasoning` 与正文分流正确。

## 8. 为什么当前没开 mxfp8（C8）KV —— 以及下一步

权重 recipe 确实是 C8：`quant_model_description.json` 里
`kv_cache_type = K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL`。加上
`--kv-cache-dtype mxfp8 --kv-cache-dtype-skip-layers sliding_window` 后，启动阶段（图捕获）报：

```
NPUGraph: ERROR — capture failed: RuntimeError: npu_scatter_pa_kv_cache ...
Invalid_Argument_Tensor_Dtype(EZ0021): Parameters key, keyCacheRef of
aclnnScatterPaKvCache have incorrect dtypes DT_BFLOAT16, DT_FLOAT8_...
```

即 **KV cache 已经是 fp8，但写入走的是通用 bf16 路径**。引擎侧 C8 后端不是注册表选的，而是由
ModelSlim 量化方法 `quantization/methods/kv_cache/mxfp_c8.py` 在 `create_weights` 里逐层替换
（`layer.attn_backend = AscendC8MXFPAttentionBackend`、`layer.impl.__class__` 换成 C8 实现）；
C8 的 scatter/scale 处理在 `worker/v2/attn_utils.py:348`、`model_runner.py:414`。

另外 `enable_kvpp: true` 也被引擎直接拒绝：`KVPP currently supports only non-hybrid MLA models`
（本模型是 hybrid GQA + sliding window）。

下一步排查方向（按成本排序）：
1. 确认 C8 方法是否被挂上：启动日志里该层是否出现 `MXFP8_PER_CHANNEL` 相关加载；
   `create_weights` 里 `if hasattr(layer, "impl")` 不成立就会静默跳过；
2. 按注释 `A5_C8_MXFP_KV_CACHE_BLOCK_SIZE = 512` 把 `--block-size` 从 128 改成 512 再试；
3. 分别在 eager（`--enforce-eager`）与图模式下复现，判断是否只是图捕获路径缺 C8 支持；
4. 以上都要配合 sink attention 精度回归（`enable_sink_attention=True`）。

## 9. 停止与清理

```bash
# 各台自己容器内（按 comm 匹配；注意 API 进程的 comm 是 vllm，pkill python 杀不到）
docker exec iquest-mxfp8-p bash -c 'pkill -TERM vllm; sleep 6; pkill -KILL vllm; pkill -KILL VLLM'
docker exec iquest-mxfp8-d bash -c 'pkill -TERM vllm; sleep 6; pkill -KILL vllm; pkill -KILL VLLM'
docker exec iquest-mxfp8-p pkill -f load_balance_proxy_server_example
docker exec iquest-mxfp8-p pkill -x mooncake_master
npu-smi info -t proc-mem -i 0     # 每张卡都要 "No process in device."
```

## 10. 已知坑

| 症状 | 原因 | 处置 |
|---|---|---|
| `npu-smi: Failed to obtain the console log level` / import 失败 | NPU 设备被同机其它容器占着（设备独占） | 删掉废弃容器；确认 `proc-mem` 无进程 |
| master 起来就死，日志 `Failed to start master admin server ... 9003` | metrics/admin 端口冲突（iq-own 占 9003） | 本包已用 `--metrics_port 9004` |
| `Template file ./run_dp_template.sh does not exist` | launcher 硬编码相对路径 | 入口脚本已 `cd` 到角色目录 |
| DP 组一直不齐 | 另一台 rank 没起，或 `--dp-address/dp-rpc-port` 两台不一致 | P 组都指 `43.66:16500`，D 组都指 `42.240:16600` |
| `pkill -9 python` 杀不掉引擎 | vllm 的 comm 是 `vllm`（不是 python） | 用 `pkill -9 vllm`，再加 `pkill -9 VLLM` 清 worker |
| 容器里堆积 `vllm` 僵尸（Z 状态） | PID1 是 `sleep infinity`，不回收子进程 | 无害（不占显存/端口），属预期 |
| `enable_fused_mc2` 与 `multistream_overlap_shared_expert` 同时开 | 引擎二选一，会静默关掉 multistream | 同 2m 站，保持现状 |
| 首个请求特别慢（30 s 级） | 图捕获/编译预热 | 正常，稳态 0.8–1.0 s / 57 token |
