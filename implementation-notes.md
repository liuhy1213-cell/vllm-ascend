# IQuest-Q1 PD 分离部署性能优化记录（implementation notes）

> **本文件是 IQuest-Q1 PD 分离（1P1D）部署性能优化的活记录**，随代码一起演进。
>
> **维护纪律**
> 1. 每次改动的**代码与本条记录放在同一个 commit** —— 代码与记录永不脱节。
> 2. 记录里的所有数字必须来自实测：
>    `deploy/iquest-1p1d-2m-mooncake-new78/logs/bench/*.json` 与两侧 `engine.log`；
>    禁止写入"预计/应该/约等于"。
> 3. 每条记录固定六段：**假设 → 改动 → 测量 → 结论 → 回滚开关 → 踩到的坑**。
>    证伪的尝试**也要写**（保留负结果，避免后人重做）。
> 4. 分支：`perf/iquest-pd-hybrid-kv-20261108`（基点 `72dedd23c` = `-prof` HEAD）；
>    worktree：`/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/vllm-ascend-pd`。
> 5. 模型的移植说明、sink 数学、ACL graph 细节见
>    `vllm_ascend/models/iquest_q1/README.md`；本文件只记性能。
> 6. 计划全文见工作区 `docs/PD_hybrid_kv_优化计划.md`。

---

## 0. 进度索引

| 阶段 | 内容 | 状态 |
|---|---|---|
| S0 | 分支 + 基线固化 + 生产侧 EE1006 同步 | 🟡 分支/记录已建；基线已复测（rate 0.7 = 2,201）；生产同步待用户授权 |
| S1 | 重新启用 hybrid KV cache manager（核心解锁） | ✅ 2026-11-08：2,201→5,065 tok/s/卡（2.30×），SLA 内；修 3 处代码缺口 |
| S2 | KV-C8（mxfp8 KV cache）落在 25 个 full 层 | 🟡 滑窗算子探针完成（§3 S2-A：D 侧 decode GO）；实现未开始 |
| S3 | D 侧步进优化 | ⬜ |
| S4 | W4A4C8 权重 | ⬜ |
| S5 | 精度闸门 + 长稳 + 回合主干 | ✅ 精度（GSM8K 95/96，0 错误）/ 长稳（5 段 57.2 min，漂移 <0.4%）/ **回合主干已合并 `f020be6d2`（单机冒烟通过）** |

---

## 1. 范围与验收口径

| 项 | 值 |
|---|---|
| 部署 | `deploy/iquest-1p1d-2m-mooncake-new78`（P0=new7 `10.222.36.53`，D0=new8 `10.222.41.88`） |
| 拓扑 | 每侧 1 进程 × TP8；P = prefill + proxy(:1999)，D = decode(+MTP eagle n=5 + ACL graph) |
| 权重 | `/models/IQuest-Q1-W8A8C8-mxfp8`（`--quantization ascend`），D 侧 draft = `/models/IQuest-Q1/mtp` |
| 负载 | 64k in / 1k out，`prefix 0.9`（`--random-prefix-len 57600 --random-input-len 6400 --random-output-len 1000`），`--ignore-eos` |
| SLA | **mean TPOT ≤ 25 ms** 且 **p99 TTFT ≤ 4 s** |
| 指标 | `total_token_throughput / 16`（P8 + D8 共 16 卡）= tok/s/卡 |
| 目标 | **2,201 → 6,000 tok/s/卡**（对齐 colocated 6000@SLA 的水平） |

---

## 1. 范围与验收口径

| 项 | 值 |
|---|---|
| 部署 | `deploy/iquest-1p1d-2m-mooncake-new78`（P0=new7 `10.222.36.53`，D0=new8 `10.222.41.88`） |
| 拓扑 | 每侧 1 进程 × TP8；P = prefill + proxy(:1999)，D = decode(+MTP eagle n=5 + ACL graph) |
| 权重 | `/models/IQuest-Q1-W8A8C8-mxfp8`（`--quantization ascend`），D 侧 draft = `/models/IQuest-Q1/mtp` |
| 负载 | 64k in / 1k out，`prefix 0.9`（`--random-prefix-len 57600 --random-input-len 6400 --random-output-len 1000`），`--ignore-eos` |
| SLA | **mean TPOT ≤ 25 ms** 且 **p99 TTFT ≤ 4 s** |
| 指标 | `total_token_throughput / 16`（P8 + D8 共 16 卡）= tok/s/卡 |
| 目标 | **2,201 → 6,000 tok/s/卡**（对齐 colocated 6000@SLA 的水平） |

分支：`perf/iquest-pd-hybrid-kv-20261108`（基点 `72dedd23c` = `-prof` HEAD）。
worktree：`/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/vllm-ascend-pd`。

---

## 2. 基线诊断（2026-11-08）

### 2.1 结论

**瓶颈是显存里的并发数，不是算力、不是通信带宽、也不是 KV 池。**

此前工作文档把现象记为"P 侧 KV 块池 16 在途上限"，这是错的：真正的机制是
**`--disable-hybrid-kv-cache-manager` 让 63 个 sliding-window 层按全长存 KV**，
于是每请求 KV 从 0.95 GB 涨到 2.88 GB，两侧都只够 16 个 64k 请求。

### 2.2 证据

| 观测 | 数据 | 含义 |
|---|---|---|
| D 侧 KV 容量 | `Available KV cache memory: 45.62 GiB` → `NPU KV cache size: 1,074,944 tokens` | ÷64k = **16.8 请求** |
| P 侧 KV 容量 | `42.81 GiB` → `1,020,160 tokens` | ÷64k = **15.9 请求** |
| D 侧调度 | `Running: 16 reqs, Waiting: 32 reqs, NPU KV cache usage: 96%` | D 侧显存打满 ⇒ **D 是瓶颈** |
| P 侧调度 | 整个压测窗口 `max Running = 1`、`Waiting = 0` | P 供给侧完全空转 |
| 启动告警 | `Hybrid KV cache manager is disabled for this hybrid model, This means we do not enable any optimizations for saving KV cache memory (e.g., dropping the KV cache outside the sliding window).` | 直接点明机制 |
| KV 传输 | `KV cache transfer ... took 26–35 ms (122 groups, 500 blocks)` | 2.88 GB / 30 ms ≈ **96 GB/s/rank** ⇒ 传输不是瓶颈 |
| 前缀缓存 | `Prefix cache hit rate: 94.0%`，`External prefix cache hit rate: 1.7%` | 池只贡献 1.7%，掉池化损失有限 |
| 吞吐 | 开环 rate 0.7 = **2,201 tok/s/卡**；闭环 c24 = 3,312 但 p99 TTFT 24 s | 并发被显存卡死 |

### 2.3 机制（读码确认）

`--disable-hybrid-kv-cache-manager`（`deploy/.../new78/serve_cmd.sh:44,86`）
→ vLLM `vllm/v1/core/kv_cache_utils.py:2266` 调用
`unify_hybrid_kv_cache_specs()`（`:1919`）
→ 把 63 个 `SlidingWindowSpec` **提升为 `FullAttentionSpec`**（窗口值 4096 仍保留供计算）
→ 88 层合成 **1 个 uniform 组**，组内只有 1 种页几何
→ 滑窗层失去"只保留窗口内 KV"的优化 ⇒ 按全长存。

### 2.4 每请求 KV 账（per card；TP8 ⇒ 每卡 1 个 KV head；K+V = 512 B/token/layer）

| 配置 | 25 full 层 | 63 sliding 层 | 合计 | D 侧可并发 |
|---|---|---|---|---|
| 现状（hybrid off） | 819 MB | 2,064 MB（全长） | **2.88 GB** | ~17（实测 16） |
| 开 hybrid manager | 819 MB | 132 MB（4096 窗口） | **0.95 GB** | ~48 |
| + KV-C8（仅 full 层） | 410 MB | 132 MB | **0.54 GB** | ~85 |

### 2.5 吞吐模型（并发数待实测校正）

现状：并发上限 16（P 15.9 / D 16.8 取小），每请求驻留 ≈ 20 s
（TTFT + 1000 token × ~16 ms TPOT）⇒ 到达率上限 16/20 ≈ **0.8 req/s**
⇒ 输出 541.7 tok/s ⇒ **2,201 tok/s/卡**。

机制成立（瓶颈是显存里的并发数，不是算力），**但并发数的外推是错的**（见 §2.6）。

| 并发上限 | 到达率 | 输出 tok/s | tok/s/卡 |
|---|---|---|---|
| 16（现状，实测） | 0.8 req/s | 541.7 | 2,201 |
| **~25**（hybrid，按 block 池算法推算） | 1.25 req/s | ~845 | **~3,300** |
| **~47**（hybrid + full 组 block 512，推算） | 2.35 req/s | ~1,600 | **~6,000** |

### 2.6 【勘误】并发外推错误：vLLM 是共享 block 池，每块成本取最大组

2026-11-08 读 vLLM v0.30 `vllm/v1/core/kv_cache_utils.py` 的分配逻辑：

- `bytes_per_block = max over groups of (Σ 该组各层的 page_size_bytes)` —— `:1596-1605`
- `num_blocks = available_memory // bytes_per_block` —— `:1750`（**全局一个 block 池**）
- 一个请求从**每个组**各占 block（`:1083-1094` 求和）
- 每层 page 会被 `unify_kv_cache_spec_page_size`（`:1365`，`:2312` 调用）补齐到最大页

代入 iquest（每层 128 token × 1 KV head(TP8) × 128 dim × K/V × bf16 = 64 KB；D 侧 45.62 GiB）：

| 配置 | bytes_per_block | num_blocks | 每请求占块 | 并发 | vs 现状 |
|---|---|---|---|---|---|
| 现状（1 组 88 层 bf16） | 5.77 MB | 8,493 | 544 | **15.6** | 1.0×（实测 16 ✓） |
| hybrid（25 full + 63 sw） | 4.13 MB | 11,864 | 500 + 32 = 532 | **21.8** | **1.4×** |
| hybrid + **full 层 C8** | 4.13 MB | 11,864 | 532 | **21.8** | **1.4×（零增益）** |
| hybrid + **sw 层 C8** | 2.06 MB | 23,728 | 532 | **43.6** | **2.8×** |
| hybrid + 两层都 C8 | 2.06 MB | 23,728 | 532 | 43.6 | 2.8× |

**四条结论**：
1. **公式是求和不是取 min**：`num_blocks / (blocks_full + blocks_sliding)`。一个 block 含
   所有层的页，请求为 full 层拿 500 个 block 才能覆盖 64k，其中滑窗层的 468 页是闲置的。
   实测 8,398 blocks / 1,074,944 tokens 反推 bytes/block = 5.83 MB（含对齐填充），与
   算式 5.77 MB 吻合 ⇒ 模型正确。
2. **hybrid 单独开 = 1.4×（≈3,100 tok/s/卡），不是 3×**。滑窗组 94% 的 block 容量闲置。
3. **block_size 512 无用**：`num_blocks` 与"每请求占块数"都按 block_size 线性缩放，
   比值抵消 ⇒ 并发与 block size 无关。
4. **只把 25 个 full 层 C8 化 = 零增益**（bytes_per_block 由 63 层滑窗组决定）。
   真杠杆是**让 63 个滑窗层也 C8**（2.8×），但 `attention_c8_mxfp.py:766-767`
   明确不支持滑窗 attention ⇒ **本计划最大的技术阻塞项**
   （算子探针已给出解法，见 §3 S2-A：D 侧 decode 走"窗口即视图"可绕开 mode4）。

**⇒ S1 第一件事：量 `NPU KV cache size`（期望 1,074,944 → ~1.52M，**不是 3.2M**）
与 `Maximum concurrency`（8.20x → ~11.9x），用实测重定目标。**
**⇒ 并发封顶 22 后，剩余提升只能靠降低 step_time**（吞吐 = 并发 × (1+accept) / step_time），
其中权重流 25.7 ms/步 → W4A4 的 ~13 ms 是最大一块。

---

## 3. 阶段计划

（进度索引见 §0；此处只列每阶段的技术要点，结论一律回填到 §4/§5。）

### 3.1 S1 的代码侧结论（2026-11-08 读码，**未上机验证**）

**结论：S1 不需要改 iquest 模型代码，只需要部署侧去掉 `--disable-hybrid-kv-cache-manager`。**
下面四条是支撑该结论的读码结果，也是 S1 上机时最该先验的几点：

1. **vLLM 侧确实会分成 2 组。**
   `Attention.__init__`（`vllm/model_executor/layers/attention/attention.py:263-265`）
   优先取 `per_layer_sliding_window` ⇒ iquest 的 63 个滑窗层拿到
   `SlidingWindowSpec(4096)`、25 个 full 层拿到 `FullAttentionSpec`。
   只有 `--disable-hybrid-kv-cache-manager` 才会触发
   `kv_cache_utils.py:2266 → unify_hybrid_kv_cache_specs()`（`:1919`）把滑窗层提升成 full。

2. **MTP draft 必须继续清 `sliding_window`（`model.py:414-424` 的现有行为保留）。**
   若改成保留真实的 `SlidingWindowSpec(512)`，D 侧会变成 **3 组**
   （full 25 / sliding 63 / draft 1）而 P 侧只有 2 组（P 无 draft）⇒ 组数不对齐。
   保持清空则两侧都是 2 组，draft 并入 full 组。

3. **P 侧 88 层 vs D 侧 89 层（多 1 层 draft）的差异已被连接器处理。**
   `mooncake_hybrid_connector.py:891-895`：D 侧 `speculative_config is not None` 时
   把 layer 区间 `+1`，而 `zip(local_kv_caches_base_addrs, remote_kv_caches_base_addrs)`
   以较短的一边为准 ⇒ D 多出来的 draft 条目"绑在最后、永远不被拉取"。
   hybrid 分支的地址注册顺序取自 `kv_cache_config.kv_cache_tensors` 描述符顺序
   （`:1855-1873`），draft 的描述符在目标层之后注册 ⇒ 前 88 层两边一一对应。

4. **S2 的硬门槛是 1 行白名单。**
   `attention_c8_mxfp.py:898-908` 要求 `uses_mooncake_connector()` 为真，
   而 `vllm_ascend/utils.py:1629` 的集合
   `{"MooncakeConnector", "MooncakeConnectorV1", "MooncakeConnectorV2"}`
   **不含 `"MooncakeHybridConnector"`** ⇒ C8 + 当前 PD 配置会直接 NotImplementedError。

**上机前仍未验证的 3 个未知（按风险排序）**：
- a) 两组 block size 能否共存（full 组 C8=512 / sliding 组 bf16=128）——
   vLLM `_align_hybrid_block_size` 要求各组兼容，Ascend 在 `platform.py:405-418` 对 C8 有专门分支。
- b) `get_sw_clipped_blocks`（`:1397-1412`）的 `assert len(block_ids) == len(self.num_swa_blocks)`
   在 hybrid 下是否两侧组数一致即通过。
- c) `AscendMetadata` 只有一份 `block_tables`，hybrid 多组不同 block table 时
   运行时是否正确寻址（这是唯一可能真要改模型/attention 代码的地方）。


### 3.2 S1 上机实录（2026-11-08）

**假设**：去掉 `--disable-hybrid-kv-cache-manager` 后，63 个滑窗层只保留窗口内 KV，
D 侧可用 token 数应显著上升（按共享 block 池算式，理想值见 §2.6）。

**改动**（部署侧，`deploy/iquest-1p1d-2m-mooncake-new78`）：

| 文件 | 改动 |
|---|---|
| `serve_cmd.sh` | 两处 `--disable-hybrid-kv-cache-manager` 改为 `HYBRID_KV_MANAGER=1` 时不加（默认 0 保回退） |
| `serve_cmd.sh` | P 侧前缀缓存与池化解耦：新增 `PREFILL_PREFIX_CACHING`（默认 1）。原因：池只贡献 1.7% 命中、HBM 前缀缓存贡献 94%，关池时若连 HBM 缓存一起关掉，每请求 prefill 从 6.4k 涨到 64k，会把结论污染 |
| `env.sh` | `HYBRID_KV_MANAGER=1`、`PREFILL_PREFIX_CACHING=1`、`SFS_REPO→vllm-ascend-pd`、`PREFILL_MAX_NUM_SEQS 16→48`、`PREFILL_MAX_NUM_BATCHED_TOKENS 4096→8192`、`PREFILL_GPU_MEM_UTIL 0.90→0.94` |
| `env.pool.override.sh` | 删除（`ENABLE_KV_POOL=0`，计划 §计划自检 问题 4） |
| `correctness_gate.sh` / `measure_round.sh` | 新增：首 token/logprobs 一致性闸门、单轮测量汇总 |

**代码改动**（分支 `perf/iquest-pd-hybrid-kv-20261108`）：见 §3.3，两处都是
**hybrid 打开后才暴露的既有缺口**，不是优化代码。

**测量**：见 §4。

---

### 3.3 S1 暴露的三个代码缺口（generic hybrid 从未被跑过）

#### 缺口 1：`MooncakeHybridConnector.register_kv_caches` 无 generic hybrid 分支

现象（D 侧首次以 hybrid 启动，`logs/archive/...`）：

```
File ".../mooncake_hybrid_connector.py", line 1907, in register_kv_caches
    raise TypeError("Mooncake connector does not support this type kv_cache now.")
```

原因：该函数的分支链只有 three 条 —— `not use_hybrid`（普通 GQA）、`use_mamba`、
`use_compress`（DeepSeek-V4 共享页）。**普通 GQA 的 full+sliding 多组布局没有任何分支**，
说明这条路径此前只有 Mamba / DSV4 模型走过。

修法：抽出 `_layer_kv_regions()`（把原 non-hybrid 的 K/V 交错 span / 独立张量逻辑共用），
新增 `elif self.use_hybrid:` 分支：按层登记 `kv_caches_base_addr` / `block_len_per_addr`，
并填 `addr_group_idx`（每个地址属于哪个 KV 组 —— hybrid 传输循环靠它筛组，
留空会把每层数据复制到每个组），最后 `block_stride_per_addr = block_len_per_addr`。
MTP draft 只在 D 侧存在且登记在末尾，传输循环的 `zip` 以短边为准自然丢弃。

#### 缺口 2：Ascend 多组 KV 预算被按「每层私有 buffer」缩放（hybrid 零收益的直接原因）

现象（hybrid 打开后 D 侧启动日志）：

```
WARNING [kv_cache_utils.py:1556] Add 15 padding layers, may waste at most 23.81% KV cache memory
INFO    [worker.py:751] Ascend per-layer KV layout scales the multi-group budget by 0.2921
        (1703936 bytes/block over 5832704 total page bytes) so per-layer buffers fit within device memory.
INFO    NPU KV cache size: 940,002 tokens, Maximum concurrency 7.17x
```

即：**hybrid 打开后可用 KV 反而从 1,074,944 掉到 940,002 tokens**（−12.6%），
比 baseline 更差。

机制（读码 + 日志数字互相印证）：

1. vLLM 的 hybrid 分组是「按类型分桶 → 每桶切成 group_size 层一组」，iquest 是
   **4 组**：full 26 层（25 + MTP draft）/ sliding 21+21+21 层。
   `bytes_per_block = max(组内层数) × page = 26 × 64KB = 1,703,936 B`（与日志数字一致）。
2. 每个组的 `KVCacheTensor.size` 都等于 `bytes_per_block × num_blocks`，且**四组都从
   byte 0 别名同一块 backing**（`kv_cache_utils.py:1759` 的注释图）。一个 block id
   同一时刻只属于一个组，所以物理上只需要一份池。
3. Ascend 的 `_allocate_kv_cache` 只在 `use_hybrid_layout = has_mamba and has_attention`
   时启用 `hybrid_backing` 单一 backing；**attention-only 的 hybrid 走不到**，
   于是退化成「每层一个私有 buffer」（`num_blocks × page_size` 各一份），
   总占用 = `num_blocks × Σ所有层 page = 5,832,704 B/block`，
   是 vLLM 记账的 3.42 倍 ⇒ worker 端的预算预缩放（0.2921）只是在补偿这个退化布局，
   hybrid 省下的内存全部被浪费掉。

修法（两处必须同时改，否则不是 OOM 就是白省）：

| 文件 | 改动 |
|---|---|
| `vllm_ascend/worker/v2/attn_utils.py` | 新增 `can_overlay_attention_kv_groups()` 判定；`_allocate_kv_cache` 的 `use_hybrid_layout` 改为 `has_attention and (has_mamba or can_overlay_attention_kv_groups(...))`，让 attention-only hybrid 也走单一 backing + 每层连续视图 |
| `vllm_ascend/worker/worker.py` | `get_available_kv_cache_memory` 的免缩放分支加入 `overlays_attention_groups`（同一判定），不再乘 0.2921 |

判定条件刻意收窄：≥2 组、`LBNHC`（layer-outer + block-compact，日志
`Using LBNHC KV cache layout`）、全是 bf16/fp16 的 `FullAttentionSpec`/`SlidingWindowSpec`、
各组 `page_size_bytes` 相同。分配路径对 `layer_stride/block_stride` 有 assert，
不满足会直接报错而不是静默错位。

#### 缺口 3：overlay 布局下「按 base_addr 去重」会静默丢掉整组滑窗层（我自己引入的 bug）

现象（首次以 overlay 布局跑通后）：容量数字全对（D 3.2M tokens），但
**P 直连与走 proxy 的首 token 分布完全不同**（P 给 `<think>`（lp≈-1.8e-5），
proxy 给 ` j`（lp≈-2.18）），且**非确定性**（同一 prompt 5 次里 1 次对）。

定位过程（三段对照，不需要改代码）：
1. P 直连 vs **D 直连**（D 自己算 prefill）→ 首 token logprob 一致到 1e-4
   ⇒ 模型/分组/attention 本身没问题。
2. P 直连 vs proxy（走传输）→ 首 token 分布差几个 nat ⇒ 传输路径有问题。
3. 引擎日志：两侧 `Registering interleaved KV span` 各 88/89 条（层数对），
   但按 group 过滤后传的是 25/26 层。

根因：overlay 布局下**不同组的层共用同一段物理槽位**（组 0 的 layer j 与组 1 的
layer j 地址完全相同）。我在 hybrid 分支里加了「同一 base_addr 只登记一次」的去重，
于是 63 个滑窗层全部被当成「KV 共享」跳过 ⇒ **滑窗组从来没被传输过**，
D 侧滑窗 KV 是前一个请求的残留 ⇒ 输出随残留内容漂移（解释了非确定性）。
修正：hybrid 分支不做去重 —— 每个层保留自己的地址条目；block id 同一时刻只属于
一个组，各组的块从各自层的槽位寻址，物理重叠是设计本身。
（物理内存注册仍由 `collect_configured_register_regions` 按 storage 去重，不受影响。）

**正确性判据的教训**：不能拿 greedy 文本相等当闸门 —— 两个引擎 batch/内核不同，
`<think>` 之后两个常见开头 logprob 只差 0.02~0.03，会自然翻转。
改用「P 的 top-1 token 落到 D 的 top-5 且 |Δlogprob| ≤ 0.1 nat」
（`correctness_gate.sh`，多档 prompt 长度）。

---

### S1 重新启用 hybrid KV cache manager

- **改哪**：`serve_cmd.sh` 两处 `--disable-hybrid-kv-cache-manager` 改为可关
  （新增 `HYBRID_KV_MANAGER=1` 时不加该 flag，默认 0 保回退）。
- **为什么可能可行**：`MooncakeHybridConnector` 已实现完整 hybrid 路径
  （`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py:1358`
  与 `:1689` 的 `use_hybrid`：per-group `kv_cache_specs` / `group_block_size` /
  `sw_sizes_tokens` / `get_sw_clipped_blocks`）；`MooncakeConnector` 与
  `AscendStoreConnector` 都声明 `SupportsHMA`
  （`mooncake_hybrid_connector.py:1202`、`kv_pool/ascend_store/ascend_store_connector.py:99`），
  `AscendMultiConnector` 的 assert（`ascend_multi_connector.py:38`）应能通过。
- **要保持**：`vllm_ascend/models/iquest_q1/model.py:414-424` 给 MTP draft 清
  `sliding_window` 的行为**保留**（draft 并入 full 组，浪费 1 层 × 64k = 33 MB/请求）。
- **风险**：P 侧 88 层 vs D 侧 89 层（多 1 层 draft）⇒ 两侧每组 block 数按不同层数分配，
  而传输传的是 P 的 block id。必须验 draft 仍绑在 full 组**末尾**、不打乱其余层对齐。
- **回退阶梯**：(a) 保留 hybrid 但去掉 AscendStore 池（单 `MooncakeHybridConnector`）；
  (b) 只 D 侧 hybrid；(c) 回退原状。

### S2 前置复核：当前部署**没有任何 C8 KV**，且 overlay 要求两组页大小相同

2026-11-08 从两次运行的实测反推（`num_blocks` 来自 connector 的注册日志）：

| 运行 | available | num_blocks | bytes_per_block | 理论值 |
|---|---|---|---|---|
| hybrid + W8A8C8 | 45.62 GiB | 28,748 | **1.7039 MB** | 26 层 × 64KB = 1.7039 MB ✓ |
| hybrid + W4A4C8 | 63.98 GiB | 40,315 | **1.7037 MB** | 同上 ✓ |

⇒ `bytes_per_block` 由 **full 组 26 层 × bf16 页（64KB）** 钉住，**没有 C8**。
日志也直接给出证据：`modelslim_config.py:466` 明确警告
"The checkpoint carries an MXFP8 KV recipe, but --kv-cache-dtype is not 'mxfp8';
the recipe is not enabled"，且所有层的注册日志都是 `page=65536 B`（含 full 层，
如 `model.layers.87`）。`kv_cache_dtype=auto`。

**两条结论**：

1. **只把滑窗组 C8 化 = 零增益**（计划 §自检 问题 3 的结论成立）。
2. **只把 full 组 C8 化也不行 —— 不只是零增益，而是会退回到"每层私有 buffer"**：
   我加的自适应 overlay 判定 `can_overlay_attention_kv_groups()` 要求
   **各组 page_size_bytes 相同**（overlay 的物理前提：同一 block id 在不同组里
   必须能互相同一片存储）。若 full 组 32KB / sliding 组 64KB，判定返回 False
   ⇒ 走回每层私有分配 + 预算乘 0.29 ⇒ 容量从 40,315 block 掉到 ~2,900。
   ⇒ **S2 只有一条可行路线：full + sliding 同时 C8（两组都 32KB），
   收益 ≈ 2×（1.70 → 0.85 MB/block）**；若要分批上，必须先扩展 overlay 支持混合页距。

### S2 【2026-11-08 更正】"一行判定"捷径被否决，as_strided 设计仍是唯一正确几何

我一度认为"混合页距只需放宽 `can_overlay_attention_kv_groups()` 的相等判定"，
理由是 LBNHC 下每层仍连续、组间从 byte 0 别名。**该结论错误，已被鸽笼反例否决**：

- full 组（page 33,792）layer0 **block 1** 在 `[33,792, 67,584)`；
  sliding 组（page 65,536）layer0 **block 0** 在 `[0, 65,536)` ⇒ 重叠 `[33,792, 65,536)`；
- 这是**两个不同 block id、两组同时活**，调度器的"组间 id 独占"拦不住；
- 等页距之所以安全，是因为同 id 落同一段字节，混合页距下这个前提消失。
- 上游注释也写明 layer-outer "only supported for uniform page sizes or single-group models"，
  `_get_packed_kv_cache_groups` 的 gate 是 `if not layout.is_block_outermost ...: return None`。

⇒ **必须改成 block-outer 几何（`as_strided`，`stride(0)=bytes_per_block`）+ 连接器拆分
`block_len`/`block_stride` + attention 视图适配**，即设计稿 §3 的原方案。工程量不降。
错误分支 `feat/iquest-pd-c8-mixed-pages`（`2c2ca4520`）已删除；设计稿已重建（§1 记录该反例）。

### （已作废）旧判断：路径 A 只需要一行判定

初版设计稿（`docs/S2_路径A_混合页距分配器设计稿.md`）假设"混合页距必须改 `as_strided`、
块间距统一成 `bytes_per_block`"，**假设错误**。重读 vLLM
`v1/kv_cache_interface.py:compute_layout_strides()`：`fixed_strides` 只在
`layout.is_block_outermost` 时钉 `block_stride = bytes_per_block`，而本模型是 **LBNHC
（layer-outer）**，所以：

- 组内：`block_stride = 该组 page`、`layer_stride = num_blocks × 该组 page`
  —— **每层仍是一段连续区间**；
- 组间：只是从 byte 0 别名（`byte_offset` 每个 group 循环重置）；
- 混合页距下 full 组占 `NB × 878,592`、sliding 组占 `NB × 1,376,256`，
  都在 `bytes_per_block × NB` 的 backing 内。

⇒ 现有 `_allocate_kv_cache()` overlay 分支的三个校验在混合页距下**全部成立**，
连接器的 `block_len == block_stride` 假设也**不变**。
⇒ **只需放宽 `can_overlay_attention_kv_groups()` 的"跨组页大小相等"条件**
（改为"组内一致"，这由 group spec 保证）。

改动已落在**独立分支** `feat/iquest-pd-c8-mixed-pages`（`2c2ca4520`），**未合入部署分支、
未上机验证**：等页距下行为与改动前完全一致。真正的风险在 C8 算子侧
（`attention_c8_mxfp.py` 的 prefill TND + sink 从未在任何部署跑过，
`37a95d2bc` 的 512 kernel block 也未验证）⇒ 验证顺序应为**先单机跑 C8 + GSM8K + sink 复测，
再上 PD**。

### S2 §4.2 单机验证结论（2026-11-08，友军会话）

**算子层清零，问题 100% 在分配/预算层。**

- `--kv-cache-dtype-skip-layers` **整体否决**：该 flag 只 module 级生效 → 上游 platform 把
  block 512→1,024、`shared_page`=270,336、89 层统一 270,336 字节页 ⇒ 私有 per-layer
  （scale 26/89=0.2921）+ dtype 门拒 fp8 ⇒ V2 崩在 `_reshape_kv_cache_v2:1943`；
  V1 则被 `block_size` 写回 min(组块) 后触发的 C8 守卫拦住（`BLOCK_SIZE=auto` 可绕过）。
- **C8 算子精度 PASS**：绕过两道闸后 GSM8K-100 greedy ×2 = {95, 97}、errors=0、sink 在链内。
- **容量 5.3× 损失**：私有 per-layer 下 sliding 128-token 块按 270,336 字节页分配（2,112 B/token，
  4× 浪费）⇒ 448,133 tokens vs 合并 bf16 的 2,387,389。
- ⇒ S2 在 as_strided 之外还需 **预算语义按 token 归一** + **组间混合块距支撑**（V1 调度器 lcm 可行）。

**记账口径修正（2026-11-08，友军算术）**：`1,376,256 B` / `33,792 B 页` 是作废的页模型
（C8 kernel [512] 硬约束）。按实测页几何（V1 run-3）在 lcm-512 窗口上重算：

| 配置 | 每 512-token 窗口 | B/token |
|---|---|---|
| 全 bf16 | 26 × 262,144 = 6,815,744 | 13,312 |
| full C8 + sliding bf16 | max(25×135,168, 21×262,144) = 5,505,024 | 10,752 |

⇒ 13,312/10,752 = **1.238×**，与 §0 结论吻合；full 组槽位浪费 39%，硬上限 ≈×1.7。
分母必须用**窗口 token 数**，不能按组块尺寸除（V2 的 3.4× 超约即死于此）。

### S2 KV-C8（mxfp8 KV cache）

- **关键结论**：上次 C8 崩溃的根因就是"合组"，所以 **S1 是 S2 的前置**。
  合组后全组只有 1 种页几何，C8 的 512+16 页距被套到 bf16 滑窗层头上 ⇒
  `vllm_ascend/worker/v2/attn_utils.py:1897` 的
  `raw_k_tensor.view(k_dtype).view(k_shape)` 形状不匹配
  （目标 `[1943,128,1,128]` vs 实际 `1943×528×128`）。S1 分 2 组后此崩溃结构性消失。
- **PD 硬门槛**：`vllm_ascend/attention/attention_c8_mxfp.py:898-908` 要求
  `uses_mooncake_connector()` 为真；而 `vllm_ascend/utils.py:1628-1630` 的白名单
  只有 `{"MooncakeConnector","MooncakeConnectorV1","MooncakeConnectorV2"}`，
  **不含 `"MooncakeHybridConnector"`** ⇒ 需加白名单。
- **其它**：MTP draft 的 C8 归属、block size 对齐、图模式兼容、sink 精度复测。

#### S2 旁证：主干合并树在 SLA V2 部署冒烟 PASS（2026-11-08，A 机 iquest-sla-a）

- 主干 `iquest-q1-pd-quant` @ `f020be6d2`（与最新 `57035a230` 零代码差）以 SLA
  optimal 参数（TP8、block 128、bf16 KV、MTP n5、graph、async+EP8+MC2）拉起。
  **SLA 部署走 V2 runner**（`deploy/iquest-sla-throughput/launch_engine.sh:33`
  `VLLM_USE_V2_MODEL_RUNNER=1`），故本冒烟执行了 V2 混合组分配路径——与
  单机 V1 冒烟（见 `55d9b7366`）互补。
- **S1 共享池收益外溢到等页距 SLA 配置**：旧 opt 分支走 "scales the multi-group
  budget by 0.2921" 私有 buffer 回退（696,673 tokens / 5.32x），合并后该回退消失，
  KV **2,387,389 tokens（3.43×，恰=1/0.2921）** / 18.21x。
- 闸门：c28 ×2 = 1,041.6/1,032.1 中位 **1,036.9（+3.9% vs 997.5）**、mTPOT 23.59、
  零失败、acceptance 34.6%；GSM8K {94,95}（errors=0；失败集与全部历史运行 diff 后
  钉死 {12,87,93} 慢性失败基线同样中招，零新增推理错，截断抽签 −2 记备注）。
  数据：`logs/gsm8k/merge_f020be6d2_n100_r{1,2}.jsonl`、`logs/bench/a_mg_c28_r{1,2}.*`。
- A 机终态 = 主干树（env.sh 指 `vllm-ascend`，opt 行注释保留可回退）。

### S2-A 算子探针：C8-QFA 滑窗能力 go/no-go（2026-11-08，docs-only）

**假设**：63 滑窗层 C8 是唯一 2.8× 并发杠杆（§2.6），阻塞在
`attention_c8_mxfp.py:766-767` 的滑窗 NotImplementedError。先探 CANN 9.2.1
`quant_flash_attn`（QFA）能否原生支持滑窗（mask_mode=4 / win_left），探针不过则
永久关闭这条线。

**改动**：零代码改动。容器 `iquest-pd-p`（CANN 9.2.1，与引擎同款）直调 QFA op，
6 轮探针（`/tmp/c8_swa_probe*.py`）。缓存按引擎同款 packet 几何构造
（`mxfp_paged_cache_views`：物理顺序 **K|K_scale|V|V_scale**，PITCH=33792，
K_BYTES=16384、S_BYTES=512），layout_q=TND / layout_kv=PA_NZ / layout_out=TND、
q_descale=N2TGD 5-D、scale 字节=127(e8m0 2^0)，与 `attention_c8_mxfp.py` 调用
逐参对齐。V payload 用单调编码（V[block][row]=16·block+1+row/16，一个 token 的
128 个 dim 同值）让输出直接反读被注意的 token 集合——自一致方法，无需参考实现。

**测量**（T=1 除另注；输出按 e4m3 量化格点解码）：

| 探针 | 配置 | 结果 |
|---|---|---|
| mode4 直路 | quant_mode=1 + mask_mode=4（win_left=1e5 / 15） | ❌ tiling 硬拒："Current quant_mode only supports mask_mode 0 (NO_MASK) and 3 (CAUSAL), SLIDING_WINDOW(4) is not supported" |
| R0 多块正确性 | mode0，4-block 表，seqused=500 | ✅ out=28.5 ≈ 全 500 token 理论值 28.31；LSE=6.221≈ln(500)；全 1 控制组 out=1.0000（归一化精确） |
| R1 尾窗字节一致 | mode0，独立 16-token 缓存（=500 缓存尾部逐字节拷贝） | ✅ out=56.0=尾部内容（b3 rows116..127 的编码值） |
| **R2/R4 窗口即视图** | mode0，**block_table 首条目=窗口起始块 + seqused=16**（不复制缓存） | ✅ R2=48.0（b3 rows0..15）、R4=17.875（b1 rows0..15）——语义：从 block_table 首条目的 row 0 起读 seqused 个 token |
| 显式 mask | mode0+mask ❌（"When mask_mode is 0 (NO_MASK), attn_mask should not be provided"）；mode3+mask[6,16]/[6,2048]/bool ❌；**mode3+int8[2048,2048] ✅** | N1 正控：语义=mask[**绝对 q 位置**, 视图相对 kv 列]，全 1 行=不 mask，带状掩码输出逐行吻合理论（2.25..2.875） |
| quant_mode 枚举 | metadata op 实测 | 仅 {0=HIF8, 1=MXFP8, 6=QK/V 分离 FP8}；**mode5(MXFP4) 不存在**（wrapper 枚举有、C++ 拒） |

**结论：GO（限定 D 侧 decode）**。
1. mode4 直路 NO-GO——kernel tiling 拒绝，"枚举声明≠实现"。
2. 显式 mask 路径 NO-GO——mode0 禁 mask；mode3 的 [2048,2048] 定格网格装不下
   iquest W=4096 窗口（视图截半），且行索引=绝对 q 位置，>2048 序列不可用。
3. **GO 路线 = 窗口即视图（window-as-view）**：滑窗层 decode 时传"平移+裁剪的
   block_table（首条目=窗口起始块）+ seqused_kv=窗口长"，mode0 无 mask，算子级
   精确（R1/R2/R4 逐字节验证，R0/R3 多块正确）。**1P1D 的 D 侧是纯 decode，
   恰好覆盖瓶颈侧** ⇒ §2.6 的 2.8× 杠杆在算子层无阻塞。
4. 引擎侧待办（设计问题，非算子问题）：①AscendMetadata 单份 block_tables，需让
   滑窗组拿到平移视图（= §3.1 未知项 c）；②MooncakeHybridConnector 白名单
   （de03e3845 已做）；③MTP verify（T≤6）滑窗层三选一：mode0 视图多看 ≤5 个
   本步 draft token（泄漏有界）/ mode3 mask 视图截半 / 滑窗层退 T=1 逐 token，
   S2 实现时定；④**P 侧 prefill（T>1+滑窗）无精确算子路径**，P 侧滑窗层维持
   现状（P 非瓶颈，§2.2）。
5. S4/W4A4C8 关联：QFA 无 MXFP4 模式 ⇒ attention 恒 W8A8（配方本就如此），
   S4 无算子阻塞也无 attention 侧收益。

**回滚开关**：不适用（纯探针；S2 实现的开关随实现另记）。

**踩到的坑**：
1. packet 物理顺序是 **K|K_scale|V|V_scale**（K_scale@16384、V@16896）——把
   scale 字节（0x7F=e4m3 NaN）填进 V payload 会经 softmax 污染整行输出
   （v2/v3 探针 NaN 根因）。未写 slot 一律 0x00。
2. 探针输出是 **e4m3 量化值的凸组合**：[32,64) ULP=4、[16,32) ULP=2——读数按
   量化格点解码，别当精确均值。
3. `max_seqlen_kv=-1` 不裁剪注意力范围（mskv=128 仍全量注意 500 token）。
4. mode3 的 attn_mask 形状硬性 [2048,2048] int8（与 max_seqlen 无关）；
   行=绝对 q 位置、列=视图相对 kv 索引；全 1 行按不 mask 处理。
5. `quant_mode=5` 在 wrapper 枚举里存在但 C++ 只收 {0,1,6}——与 mask_mode=4
   同一教训：**接口声明≠算子实现，必须运行时探**。

**v10 补充（同日晚，探针 probe7）：win 参数门也关死 + 系统级结论修订。**

| 探针 | 配置 | 结果 |
|---|---|---|
| P1/P2 win 参数 | mode0 + win_left=15/win_right=0（T=1 / T=6） | ❌ metadata op 硬拒："When maskMode is not 4 (SLIDING_WINDOW), winLeft and winRight must be -1, but got winLeft=15" |
| P5 mode3+win | mode3 + zeros[2048,2048] + win15 | ❌ 同上（win 只属于 mode4，mode4 又被 tiling 拒） |

⇒ **窗口机制穷尽：唯一可用 = 窗口即视图（T=1 decode）**。结合 S1 实测（零 C8 KV、
overlay 要求各组每层 page_size 相等），滑窗 C8 的端到端链条断在三处：
①P 侧 prefill（T>1，chunk 8192 > mask 网格 2048，无 win 参数）无精确算子路径读
C8 滑窗缓存 ⇒ P 滑窗必须 bf16；②connector 裸拷贝块字节 ⇒ P bf16 / D C8 的滑窗
传输会把 bf16 字节写进 C8 缓存（除非 quantize-on-send，大改造）；③滑窗组保持
bf16 ⇒ bytes_per_block 被 sliding bf16 组（21 层 × 65,536 B = 1,376,256 B）钉住。

**S2 修订范围（替代上文结论 3 的"2.8× 杠杆无阻塞"表述）**：full 组 C8 两侧 +
混合页距 allocator（full ~32KB / sliding 64KB 下 `can_overlay_attention_kv_groups()`
按层相等判定必 False，是新的关键路径）+ 滑窗保持 bf16（S1 逻辑不动）⇒
bytes_per_block 1,703,936 → 1,376,256 B（日志锚定：每层 page=65,536 B，D 侧
num_blocks=40,314；此前写的 1.342MB/1.27× 是 KB/KiB 滑倒，21×65,536=1,376,256），
**P/D 池 ~1.24×**（P 48→~59、D 67→~83 在途），吞吐封顶 +24%、TTFT 队列 −19%。
full-C8 为**首次生产 C8**（CLAUDE.md 坑 #7：
C8 KV 从未真跑过；full 层 prefill TND 同为首次）⇒ 精度三件套（GSM8K ≥95 raw +
sink 复测 + 长稳）是硬闸门。窗口即视图探针价值重定位：D 侧滑窗 C8 的算子前置
已验证，但端到端被 ①②锁死——留作 CANN 修 mode4 或传输量化落地后的现成方案，
当下不投工程。

**S2 的价值链存档（2026-11-08 定稿，避免下一轮从头论证）**：

S1+S4 落地后（稳态 8,623 tok/s/卡，TTFT p99 4.72 唯一破项，TPOT 23.5 贴 SLA 25，
等效权重流 ≈970 GB/s/卡已贴 950DT HBM 上限），剩余杠杆全部定性完毕：
- **S3 暂缓**：权重流贴带宽 ⇒ 元素级优化（标量折叠/norm 融合/lm_head）最多回收
  2–5%；MTP 深度 A/B（n=3 vs n=5：6,448 vs 6,388 但 TPOT/TTFT 全劣，accept 前可–3
  个 token 饱和）说明"靠 accept 提吞吐"也到头。定稿于 02ceec158。
- **S2 = 唯一剩下的量级可观结构性杠杆**，两个独立解锁条件，收益二选一取大：
  | 解锁条件 | 范围 | 收益 |
  |---|---|---|
  | A：混合页距 allocator（`can_overlay_attention_kv_groups()` 支持 per-group page） | full 组 C8、滑窗 bf16 | P/D 池 ~1.24×（吞吐封顶 +24%，TTFT 队列 −19%；bpb 1,703,936→1,376,256 B，sliding 21×65,536 钉住） |
  | B：CANN 支持 mode4（或 win 参数出 mode4） | 两组同时 C8（每层 page 均 ~33.8KB——C8 packet 含 scale 实测 33,792 B ⇒ overlay 相等判定自然成立，**无需 allocator 改造**）+ D 侧窗口即视图已验证 + P 侧 prefill 滑窗走 mode4 | **P/D 池 ≈1.94×**（bpb=26×33,792=878,592 B），TTFT 破项修复 |
  当前卡点：A 难在分配器（layer-outer 单一 backing 下混合页距会 offset 互压）；
  B 难在 CANN kernel（tiling 硬拒，本节探针已存证）。
- 精度闸门（任一解锁后）：GSM8K ≥95（raw）+ sink attention 复测（quant/README
  坑 #7）+ 长稳；当前 95/96 为 bf16 KV 对照基线。

**路径 A 评估完成（2026-11-08，17 出设计稿、本会话评估）**：
`docs/S2_路径A_混合页距分配器设计稿.md`（工作区 docs）+ 离线单测
`docs/s2_path_a_allocator_unit_test.py`（22 项全绿，无 NPU）。结论：
① 布局成立——槽位边界/组内不相交/跨槽位不外溢/容量核算全过；
② 连接器改造比稿子初版更必须：交错判定失败后掉进 disjoint 分支会把 K/V 重叠视图
  当独立区域注册（span 覆盖整段 backing）→ 注册冲突；"块距=块长"假设落点在
  `block_stride_per_addr.extend(self.block_len_per_addr)`（:1996）；
③ 退化安全：等页距旧布局下旧判定仍为 True、block_len==block_stride（单测 §E）；
④ 数字修正：×1.238、num_blocks 40,314→≈49,900（单测算得 49,912）、P/D 在途
  ~59/~83。验收清单第 1 步已完成，下一步=第 2 步启动容量（需上机窗口）。

**"一行判定"捷径否决（2026-11-08，17 提出复核结论，经 vLLM 源码+单测 §F 否决）**：
17 提出 layer-outer 下 descriptor 自带每层 NB×page 连续段、`allocate_shared_attention_
kv_cache` 三项校验（layer_stride==NB×page、block_stride==page、end≤backing）在混合
页距下全过 ⇒ 只需去掉 can_overlay 相等判定、无需 as_strided/连接器。**否决依据**：
vLLM `kv_cache_utils.py:1756-1768` 构造注释明言 layer-outer 组间别名 "only supported
for uniform page sizes or single-group models"；`:1979-1995` `_get_packed_kv_cache_
groups` 把全部混合页距机制 gate 在 `is_block_outermost` 之后。几何：layer-outer 下
两组从 byte 0 别名，页网格 33,792 与 65,536 从 0 起交错 ⇒ 任意跨组活页相撞
（full 层0 blk1 [33,792,67,584) 与 sliding 层0 blk0 [0,65,536) 实测重叠
[33,792,65,536)，单测 §F）；**组间块 id 独占救不了**（撞的是不同 id，且两组活块
都成千上万、鸽笼必撞）。等页距下不同 id 不撞（§F 对照）=今天部署安全的原因，
也正是 attn_utils.py:1008 相等判定存在的理由。17 对的部分：校验确实全过、代码会
"跑起来"——但跑起来≠布局正确，跨组静默互踩恰好是该判定要拦的事。
**结论：路径 A 仍需原稿 §3 槽位几何（as_strided + 连接器 block_len/block_stride
拆分 + attention 视图适配），工程量不降**；单测 §D/§E 是路径 A 的目标几何验证，
不是"将来切 block-outer"的假设基线。验证顺序建议（先单机无 PD 跑通 C8 算子 +
GSM8K + sink，再上 PD）成立并采纳——分配器风险与算子风险相互独立。

### S3 D 侧步进

**2026-11-08 实测后的定位（重要）**：

- **D 侧 profile 已采到（2026-11-08，64k/1k、rank0、628 个 decode step）**：
  `torch_npu analyse` 的 `op_statistic.csv` 按算子聚合（设备时间）：

  | 桶 | 占比 | ms/step | 主要内容 |
  |---|---|---|---|
  | MoE matmul | 29.7% | 7.38 | `GroupedMatmulSwigluQuantV2` 15.5% + `GroupedMatmul` 14.2%（MC2，含通信） |
  | attention | 25.2% | 6.26 | `FusedInferAttentionScore`（64k 上下文、滑窗层读 4096 窗口） |
  | elementwise | 17.4% | 4.32 | Slice 2.3% / Cumsum 1.5% / rope 1.4% / Cast 1.3% / muls_add 1.2% / Muls 1.2% / RealDiv 1.1% / sink_apply 1.0% … |
  | norm | 8.2% | 2.05 | `RmsNorm` × 5/层（含 2 个 QK-norm） |
  | routing | 7.6% | 1.89 | `MoeInitRoutingV3` 4.4% / `MoeFinalizeRoutingV2` 1.8% / `MoeGatingTopK` 1.4% |
  | 其它 matmul | 7.4% | 1.85 | QKV/o_proj/lm_head |
  | quant | 4.5% | 1.13 | `DynamicMxQuant` |

  **单步设备时间 24.87 ms**（15.61 s / 628 step），与 TPOT 20.8–23.5 ms 同量级。

- **修正**：此前"每步由权重流主导、S3 只剩 2–5%"的说法**过于绝对**。profile 显示
  纯 matmul 只占 37%，attention 25%，**非 matmul 的 elementwise+norm+routing+quant 合计 38%**
  （高于计划引用的短上下文画像 elementwise 13.9%）。所以 S3 仍有面，但**每一项都是 1–4%**：
  单点最大的是 `RmsNorm` 8.2%（两个 QK-norm 是其中 2/5，而 `enable_iquest_triton_qk_norm`
  的 A/B 已实测无收益）与 routing 三件套 7.6%。
  ⇒ 结论不变（性价比低、均在噪声边缘），但**依据从"带宽顶死"改为"单项占比小"**。
  MTP 深度对照的另一条证据仍成立：每步成本对 token 数不敏感（n=5 与 n=3 的 step 只差 2.3 ms，
  且方向由 accept 决定）⇒ verify 侧没有 token 数带来的额外空间。
- **attention 25% 是最值得动的单点，但被 S2 的算子结论锁死**（滑窗层没法读 C8）；
  full 组 C8 只能削掉其中约 25/26 的一部分 ⇒ 与 S2 的 1.24× 是同一件事。
- **两个 -prof 开关**（第 1 条）：已 A/B，rate 1.8 下 4,845/5,095 vs 对照 5,008/5,122 ——
  噪声内无收益，默认关（`S3_EXTRA=0`）。
- **MTP 深度**：n=3 vs n=5 实测 6,448 vs 6,388 但 TPOT 23.13 vs 20.84 ⇒ **n=5 更优**。
  accept 在 n=3 仍有 2.44–2.61（n=5 是 2.31–2.74）⇒ draft 收益前 2–3 个 token 就饱和，
  而 verify 的权重流是常数 ⇒ 深度越浅越亏。

0. **profile 采集通路已打通**（new78）：`serve_cmd.sh` 增加通用 `DECODE_EXTRA_ARGS` 透传，
   `env.override.sh` 写 `--profiler-config {"profiler":"torch","torch_profiler_dir":"/pd/logs/profile",
   "torch_profiler_with_stack":false}`。采集：热身 → `POST :7100/start_profile` →
   `bench_inner_pd.sh 1 60`（rate 2.0）→ `POST :7100/stop_profile` → 8 rank × 1.4 GB 落在
   `logs/profile/`。分析须**先拷到容器本地盘再 `torch_npu.profiler.profiler.analyse`**
   （SFS 上 IO-bound）。P 侧未开 profiler（`/start_profile` 404），step 归因只需 D 侧。

1. `-prof` 已有但 new78 未开的两项：`enable_fused_o_proj_ar`、`enable_iquest_triton_qk_norm`。
2. **标量折叠**（`model.py:536-549`；config 实测 `attn_out_scale=1.0`、
   `ffn_out_scale=0.53881590608`、layer 0 两者皆 1.0）：
   - `attn_out_scale` 恒 1.0 ⇒ 88 次 `Mul` 是纯空操作。
   - `ffn_out_scale` 折进 MoE 专家 w2 的每块量化 scale（`dequant(W)=W_q·scale`
     ⇒ `scale×0.5388` 在 fp32 下精确），加载期一次性、运行期零开销。
3. **残差+norm 融合**：layer 返回未求和残差对，上层用 `npu_add_rms_norm` 一次算
   （GLM `mtp.py:90-103` / `model.py:410-416` 的 unsummed pair 范式）。
4. **attention 注册 graph-visible custom op**（DeepSeek `vllm_ascend/ops/dsa.py:160-229`）：
   `direct_register_custom_op(mutates_args=["output"], dispatch_key="PrivateUse1")`
   + fake impl + `@eager_break_during_capture`，消掉 `output[:num_tokens] = attn_output` 拷贝。
5. **MoE gate 并入 fused runner**（GLM `is_internal_router`，`glm5next/model.py:245-252`）。
6. **P 侧**：`PREFILL_MAX_NUM_BATCHED_TOKENS` 4096 → 8192（现值 < 6400，一个请求吃不下）。

### S5 回合主干（2026-11-08 执行）

`git merge --no-ff perf/iquest-pd-hybrid-kv-20261108` → 主干 `iquest-q1-pd-quant`，
提交 **`f020be6d2`**（本地，未推送）：19 文件 +1,997/−101，**无冲突**（`model.py` 自动合并）。

合并后**单机冒烟通过**（本机 8×950DT、TP8、ACL graph、util 0.90、max_len 1024）：
`/health` 200、同一 prompt 两次 greedy 输出完全一致、KV 37,332 tokens、无 Traceback。
注意：单机走 **V1 model runner**（`start_service.sh` 不导 `VLLM_USE_V2_MODEL_RUNNER`），
日志里仍有 8 条 `scales the multi-group budget` —— 这是该路径的既有行为，不是回归；
overlay 分配只在 V2 runner（PD 部署）生效，那条路径已由 new78 的 PD 实测覆盖。

**PD 侧验证（2026-11-08 13:00）**：把 new78 部署切到主干树（`vllm-ascend` @ `57035a230`，
容器 `/workspace` 已确认指向该树）后重跑：

| 检查 | 结果 |
|---|---|
| 启动 | P/D `/health` 200，无 Traceback |
| KV 容量 | D **4,512,413** tokens（`-pd` 树 4,512,525，差 0.002%）、P 3,673,755 |
| 正确性闸门 | 50/200/600/1200/3000 词五档 **全 PASS**，Δlogprob ≤ **0.0005 nat** |
| 吞吐（rate 1.8 × 1200 prompts） | **7,095 tok/s/卡**（`-pd` 树 5 段中位 7,105，差 0.15%）、TPOT 21.55 ms、p99 TTFT 2.70 s、1200/1200 零失败 |

⇒ **合并后的主干在 PD 形态下与优化分支等价**，可以按主干部署。
旁证（友军会话，A 机 SLA 单机 V2 runner）：KV 容量 696,673 → **2,387,389 tokens（3.43×）**
——S1 的共享池分配修复在等页距（非 PD）配置上也生效；c28 中位 1,036.9（+3.9%）、
GSM8K {94,95} 零新增推理错。

**生产同步演练（2026-11-08，备用机，未动生产）**：在生产基线 `1b88fd5c8` 的独立 worktree 上，
用生产同款配置（W8A8C8 + MTP n=5 + `FULL_DECODE_ONLY` 图 + block 128 + util 0.94）起单机：

| 提交集 | 首个真实请求 |
|---|---|
| + `7f45a2c5d`（KV 拷贝） | ❌ `RuntimeError: Unknown keyword argument 'actual_seq_lengths' for operator npu_fused_infer_attention_score_v2`（MTP + 图模式的 FIA v2 draft replay） |
| + `0f2dfc0c2`（FIA v2 draft replay key names） | ✅ 首请求正常、三次 greedy 输出一致、引擎侧 EE1006/RuntimeError **0** |

⇒ **生产同步的最小提交集不止一个**；推荐直接切主干（`460ae8a7e`，三形态均已验证）。

**生产已上线（2026-11-08 13:55，用户授权走"切主干"）**：两侧引擎按
`master → decode → prefill → proxy` 顺序重启（池化下 master 必须先起，否则 D 侧
`Initialize mooncake failed`），P/D/proxy 均 200；4 次冒烟全正常（3 次同 prompt 输出一致）、
两侧错误计数 0、D 侧 KV 1,074,944 tokens（生产仍 hybrid off）、P→D 传输正常。
详见 `docs/PD_hybrid_kv_回合主干清单.md` §2.2。

### S4 W4A4C8

切 `/mnt/sfs_turbo/models/IQuest-Q1-W4A4C8-mxfp4-20261107`（164 GB，19 分片）：
`env.override.sh` 加一行 `MODEL_PATH=...`，`container.sh` 加一条只读挂载（要重建容器）。

**2026-11-08 实测：全通，且是本次最大的单点收益。**

- 加载 / ACL graph / eagle MTP / hybrid KV / PD 传输：全通，无报错。
- KV 容量再涨：D 3,217,812 → **4,512,525 tokens**（权重变小，KV 预算变多）。
- 正确性闸门（50/1200/3000 词）：Δlogprob ≤ 0.0003 nat，PASS。
- **SLA 内吞吐：rate 2.2 → 6,388 tok/s/卡**（TPOT 20.8 ms、TTFT p99 3.01 s）；
  rate 2.3 = 6,657 / 6,604，但 TTFT p99 3.07 / 4.21 抖动；rate 2.4 = 6,741 时 4.31 ❌。
  同并发对比 W8A8C8：rate 1.8 下 TPOT 21.9 → 19.7 ms（−2.2 ms）；
  真收益来自**能在 SLA 内多跑 ~0.4 req/s**。
- **GSM8K-100 经 PD 全链路：96/100**（4 截断、0 错误），与 W4A4C8 单机基线一致，
  高于 W8A8C8 的 95/100（噪声内）。
- 回退：删掉 `env.override.sh` 里那行 `MODEL_PATH=` 即回到 W8A8C8（无需重建容器，
  两个权重目录都已挂载）。

---

## 4. 测量记录

> 每阶段一行。`total/16` = `total_token_throughput / 16`（tok/s/卡）。
> 噪声：高并发下同配置 ±5%、TPOT ±1 ms ⇒ **关键结论 N≥2 复测取中位**。

| 阶段 | 配置 | 并发/rate | total/16 | mTPOT | p99 TTFT | 备注 |
|---|---|---|---|---|---|---|
| 基线 | new78 原样（-opt@d8a436033，hybrid off） | rate 0.7 | **2,201** | 16.35 ms | 1.55 s | `logs/bench/pd_open07r1.json` |
| 基线 | 同上 | c24 | 3,312 | 17.16 ms | 24.18 s ❌ | `pd_c24r1.json` |
| 基线 | 同上 | c48 | 3,270 | 17.33 ms | 45.81 s ❌ | `pd_c48r1.json` |
| | | | | | | |
| **S1** | hybrid KV + 池化关闭（两侧同切 -pd） | rate 0.9 | **2,964** | 15.83 ms | 2.39 s | `pd_s1_open09r1.json`（max_conc 23） |
| **S1** | 同上 | rate 1.2 | 3,700 | 17.70 ms | 3.00 s | `pd_s1_open12r1.json`（max_conc 34） |
| **S1** | 同上 | rate 1.6 | 4,634 / 4,606 | 21.95 / 22.35 ms | 2.66 / 2.70 s | `pd_s1_open16r{1,2}.json`（max_conc 49） |
| **S1** | 同上 | rate 1.8 | 5,008 / 5,122 | 21.90 / 21.88 ms | 3.64 / 3.22 s | `pd_s1_open18r{1,2}.json`（max_conc 53/54）|
| **S1** | 同上 | rate 2.0 | 5,413 | 23.26 ms | **8.90 s ❌** | `pd_s1_open20r1.json`（max_conc 打满 64） |
| **基线(A/B)** | 同树 `-pd`，hybrid **off** | rate 0.7 | **2,243** | 16.31 ms | 2.40 s | `pd_base_open07r1.json`（max_conc 18） |
| **基线(A/B)** | 同上 | rate 0.9 | 2,668 | 16.39 ms | **7.62 s ❌** | `pd_base_open09r1.json`（超容量） |
| **基线(A/B)** | 同上 | rate 1.8 | 3,289 | 17.15 ms | **49.6 s ❌** | `pd_base_open18r1.json`（排队到 64，KV 打满） |
| **S3** | + 两个 -prof 开关（fused o_proj AR + Triton QK-norm） | rate 1.8 | 4,845 / 5,095 | 23.17 / 21.14 ms | 6.56 ❌ / 3.05 ✅ | 对照 5,008/5,122；**噪声内无收益**，默认关（`S3_EXTRA=0`） |
| **S4** | W4A4C8 mxfp4 权重（hybrid 仍开） | rate 1.8 | 5,204 | 19.73 ms | 2.83 s | `pd_s4_open18r1.json` |
| **S4** | 同上 | rate 2.0 | 5,862 | 21.90 ms | 3.07 s | `pd_s4_open20r1.json` |
| **S4** | 同上 | rate 2.2 | **6,388** | 20.84 ms | 3.01 s | `pd_s4_open22r1.json`（SLA 内最好点） |
| **S4** | 同上 | rate 2.3 | 6,657 / 6,604 | 22.88 / 22.46 ms | 3.07 ✅ / 4.21 ⚠ | `pd_s4_open23r{1,2}.json`（临界） |
| **S4** | 同上 | rate 2.4 | 6,741 | 22.70 ms | **4.31 ❌** | `pd_s4_open24r1.json` |
| 对照 | 同上，**无共享前缀**（prefix0，输入 64k） | rate 1.0 | 1,605 | 11.73 ms | **78.6 s ❌** | `pd_prefix0_r1.json`｜P 侧 prefill 算力打满 |
| **S3** | MTP 深度 n=3（默认 5） | rate 2.2 | 6,448 | **23.13 ms（更差）** | 3.54 s | `pd_s3_n3_open22.json`｜accept 2.44–2.61 与 n=5 相当、step 不省 ⇒ 回退 n=5 |
| **稳态** | 同上，10 分钟长跑（1400 prompts） | rate 2.2 | **8,623** | 23.51 ms | **4.72 s ⚠** | `pd_soak22.json`｜实测 2.12 req/s、accept 2.6、无 preempt |
| **稳态** | 同上，3×5 分钟连续段（600 prompts/段） | rate 2.0 | 7,635 / 7,624 / 7,654 | 21.53 / 21.47 / 21.65 ms | 2.98 / 2.59 / 2.59 s ✅ | `pd_clean_seg{1,2,3}.json`｜实测 1.88 req/s、conc 64–66、**15 分钟漂移 <0.4%**、seg2 有 1/600 失败（引擎侧 0 错误） |

**⚠ 短跑 vs 稳态（测量口径，2026-11-08 复核）**

把每个 bench json 的 `request_throughput`（实际达成速率）与 `total/16` 并列后，
关系是恒等式：`total/16 = req/s × 65,000 / 16`。**吞吐完全由"实际达成的请求速率"决定**，
而短跑（40–130 prompts）因为 ramp-up + drain，达不成名义 rate：

| 运行 | prompts | 时长 | 达成 req/s | total/16 |
|---|---|---|---|---|
| 基线（rate 0.7） | 40 | 74 s | **0.54** | 2,201 |
| 基线同树（rate 0.7） | 50 | 91 s | **0.55** | 2,243 |
| 基线（rate 0.9） | 60 | — | 0.66 | 2,668（TTFT 7.6 ❌） |
| S1（rate 1.8） | 100 | 81 s | 1.23 | 5,008 / 5,122 |
| S4（rate 2.2） | 120 | 76 s | 1.57 | 6,388 |
| **S4 稳态（rate 2.2）** | 1400 | 660 s | **2.12** | **8,623** |

两侧被低估的程度不同：**基线在 rate 0.7 就已经饱和**（达成 0.54 < 0.7，队列立刻积压，
TTFT p99 2.40 s），短跑数 ≈ 它的稳态数；而优化侧在 rate 2.2 的短跑只跑到 1.57，
稳态是 2.12。⇒ **短跑同协议比值（2.85×）低估了真实差距。**

**稳定性（S5 部分）**：3×5 分钟连续段（rate 2.0，600 prompts/段）结果
7,635 / 7,624 / 7,654 tok/s/卡、TPOT 21.53 / 21.47 / 21.65 ms、p99 TTFT 2.98 / 2.59 / 2.59 s，
**漂移 <0.4%，两侧 preempt/OOM/ERROR 全 0**（seg2 有 1/600 客户端失败）。
⇒ rate 2.0 这一档在**600 prompts/段**下是 SLA 内最稳的稳态工作点：1.88 req/s、
≈7,640 tok/s/卡、TTFT 余量 1.0–1.4 s。

**⚠ 但把段长拉到 1200 prompts 就翻车**（`pd_soak2h_01.json`）：同样 rate 2.0、
1.93 req/s、7,854 tok/s/卡、TPOT 23.53 ms，**p99 TTFT 23.0 s** ——
队列在整段里持续累积不回落（D 侧 `Running` 打满 64、`Waiting` 8–24 一直不清零）。
⇒ **持续容量就在 1.9 req/s 附近，2.0 刚好越过拐点**；600-prompt 段看起来达标只是因为
队列还没攒起来。

**S5 长稳终版（rate 1.8，5×1200 prompts，累计 57.2 分钟带压，2026-11-08）**

| 段 | 完成 | 失败 | tok/s/卡 | TPOT | p99 TTFT | 达成 req/s |
|---|---|---|---|---|---|---|
| 01 | 1200/1200 | 0 | 7,101 | 21.24 ms | 2.46 s | 1.75 |
| 02 | 1200/1200 | 0 | 7,107 | 21.21 ms | 2.32 s | 1.75 |
| 03 | 1200/1200 | 0 | 7,091 | 22.09 ms | **22.81 s ⚠** | 1.75 |
| 04 | 1200/1200 | 0 | 7,116 | 21.12 ms | 2.26 s | 1.75 |
| 05 | 1200/1200 | 0 | 7,105 | 21.35 ms | 2.28 s | 1.75 |

**6,000 请求零失败；吞吐漂移 <0.4%（7,091–7,116）；TPOT 全在 SLA 内；两侧 preempt/OOM/ERROR 全 0；
P 侧 `Delaying free` 累计 16,921 次属正常周转（无堆积）。**
唯一异常：第 3 段一次 **p99 TTFT 22.8 s 的瞬时排队尖峰**（当段吞吐/TPOT 完全不受影响，
下一段自愈 2.26 s）⇒ 线上需把"偶发队列尖峰"列为观察项；另观察到同参数段耗时会在 11.4–21 min 间抖动。
用户确认"比较稳定"，按 5 段收尾（计划要求 2 h 墙钟连续，实际 57.2 min 带压 + 约 2 h 墙钟）。

**最终运营工作点 = rate 1.8**（`pd_soak18_{1,2,3}.json`，3×1200 prompts、连续 34.4 分钟）：

| 段 | 达成 req/s | total/16 | TPOT | p99 TTFT | 失败 |
|---|---|---|---|---|---|
| 1 | 1.74 | 7,065 | 21.37 ms | 2.41 s | 0/1200 |
| 2 | 1.75 | 7,091 | 21.64 ms | 2.32 s | 0/1200 |
| 3 | 1.74 | 7,086 | 21.50 ms | 2.55 s | 0/1200 |

**漂移 0.4%、3,600 请求零失败、两侧 preempt/OOM/ERROR 全 0、accept 2.60–2.77。**
⇒ 对基线（0.55 req/s、≈2,250 tok/s/卡）折算 **≈3.15×**。
（rate 2.0 的 7,854 是"短期能顶住、长期队列发散"的过载点，不能当运营值。）
rate 2.2 能到 2.12 req/s / 8,623，但 TTFT p99 4.72 s 已破线 ⇒ **可运营上限取 2.0（约 7.6–8.1k/卡）**。

**按"实际达成速率"重述结论**：基线 SLA 内上限 ≈ 0.55 req/s（0.66 时 TTFT 7.6 s 破线）；
优化后在 2.12 req/s 时 p99 TTFT 4.72 s（超 0.7 s），2.0 req/s 左右是 SLA 内可持续点
⇒ **2.0 / 0.55 ≈ 3.6×；用实测稳态点算是 7,640 / 2,250 ≈ 3.4×**（基线 2,250 由 rate 0.7 的 2,243 与饱和速率 0.55 夹出）。
基线侧的 10 分钟稳态长跑未做（需再切回 hybrid off 重启两轮），0.55 req/s 是由
"rate 0.7 已饱和 + rate 0.9 破 SLA"两侧夹出来的。

**S1 结论（2026-11-08）**

- **容量**：D 侧 `NPU KV cache size` 1,074,944 → **3,217,812 tokens（2.99×）**，
  `Maximum concurrency (131,072)` 8.20x → **24.55x**；P 侧 1,020,160 → **2,630,224（2.58×）**。
  比计划 §2.6 估算的 1.52M 高得多 —— 原因是 vLLM 会把 sliding 桶切成 3 组（26/21/21/21），
  `bytes_per_block` 由 full 组 26 层钉住（1.70 MB），滑窗组只按 21 层计价，
  每请求 500 + 3×32 = 596 block（比计划的 532 略多但每块更小）。
- **SLA 内的最大吞吐：rate 1.8 → median 5,065 tok/s/卡（2.30×）**，
  TPOT 21.9 ms（≤25 ✓）、p99 TTFT 3.2–3.6 s（≤4 ✓）；再往上（rate 2.0）TTFT 破 8.9 s。
  收益低于线性（并发 2.7× → 吞吐 2.3×）：batch 变大后 TPOT 从 16.4 → 21.9 ms，
  MTP accept 从 ~3.7 降到 2.3–2.6 —— 正是计划 §P2 预警的"新工作点必须重测 accept"。
- **吞吐公式核对**：并发 53 × (1+accept 2.4 换算) / step。实测 step ≈ 22 ms × 2.4 ≈ 53 ms
  仍高于 baseline 的 ~60 ms 一点点改善，但 accept 掉 1.3 抵消了。
  ⇒ 下一步的最大杠杆变成 S3（降 step）与 accept 恢复，而不是继续堆并发。
- **防假绿**：`Running` 真实升到 53/54；`NPU KV cache usage` 峰值 92.4%（不再是 96% 打满）；
  preempt/OOM/abort 全 0；prefix 命中口径未变（P 侧仍开 HBM 前缀缓存，池化已关）。
  MTP accept 2.31–2.63（未塌到 1.x）。
- **同树 A/B（hybrid off，其余配置完全一致）**：SLA 内最大吞吐 **2,243**（rate 0.7）；
  rate 0.9 时 p99 TTFT 已破到 7.6 s、rate 1.8 时 49.6 s ⇒ baseline 的可用工作点只有 rate ~0.7。
  ⇒ **SLA 内吞吐 2,243 → 5,065（2.26×）**，不是靠放开并发换来的假收益（两侧同一份代码、
  同一份部署参数、同一把度量）。
- **口径边界（prefix0 对照）**：把 `--random-prefix-len` 置 0、输入 64k（无共享前缀）后，
  rate 1.0 只能到 1,605 tok/s/卡、p99 TTFT 78.6 s —— P 侧 prefill 算力成为瓶颈
  （1.0 req/s × 64k = 64k tok/s 需求，超过 P 实测峰值 ~38.5k tok/s）。
  ⇒ **6,388 这个数成立的前提是目标口径的 90% 共享前缀**；无共享前缀时瓶颈位置不同。
- **精度**：GSM8K-100 经 PD proxy（P prefill → hybrid KV 传输 → D decode）**95/100**，
  5 条截断、0 错误 —— 与 `-opt` 栈基线画像（94–95 + 3–4 截断）一致。
- **正确性闸门**：`correctness_gate.sh` 5 档 prompt 长度（50/200/600/1200/3000 词）
  首 token logprob 与 P 直连偏差 **≤ 0.0004 nat**（判据 0.1）。

### 精度记录

| 配置 | GSM8K-100（raw 作答率） | 备注 |
|---|---|---|
| W8A8 float KV（量化交付基线） | 98/100 | |
| W8A8C8 | 97/100 | |
| W4A4C8 mxfp4 | 96/100 | |
| -opt 栈（cumsum-fix + sink） | 94–95/100 + 3–4 截断 | raw JSONL 口径；历史文档的 "97/100" 是"已答样本"口径 |

---

## 5. 踩坑记录

1. **重启后首个 bench 低 40%+** 是预热（图/内核 + prefill 突发），不是回归。
   协议：热机 ×1 + 记录 ×2。
2. **开环压测超容量时不报 failed、永远不结束**（vllm bench 无客户端超时）→
   ssh timeout 兜底必须有。
3. `REQ_RATE=x docker exec` **透不进容器**，必须 `docker exec -e REQ_RATE=x`。
4. **SFS 上改代码必须清 `__pycache__`**；worktree 是 bind mount，重启引擎即生效，
   不需重建镜像。
5. 重启一次 ≈ 6–8 分钟（309 GB 权重），实验串行规划。
6. `--kv-cache-memory` 会绕过多组缩放并 OOM ⇒ 多组布局下一律用 `--gpu-memory-utilization`。
7. hybrid 打开后图捕获档位要覆盖新并发（`GRAPH_CAPTURE_SIZES_JSON`），
   否则整批 eager 回退、吞吐崩。
8. **decode 侧的 per-step 归因必须走 `analyse` 的 `op_statistic`** ——
   `trace_view.json` 只有 host 事件，ACL 图回放不重发 host 事件。
9. `analyse` 对 16G prof 目录在 NFS 上 IO-bound 不可用 ⇒ **先拷容器本地盘再 analyse**。
10. 探活一律 TCP 探端口；`pgrep -f` 会自匹配，`pkill` 模式用 `[m]` 括号写法；
    `pkill -f 'vllm'` 会误杀 proxy（proxy 路径含 "vllm"）。
11. **open-loop 口径与 closed-loop 的表观收益可差 2.6×**（排队复利）⇒
    选对口径才能看到真实收益。
12. **`mooncake_hybrid_connector.py` 里的 `envs` 是 `vllm.envs`**；
    需要 vllm_ascend 的环境变量时必须 `from vllm_ascend import envs as ascend_envs`。
    写成 `from vllm_ascend import envs` 会静默遮蔽 `vllm.envs`，症状是
    `AttributeError: module 'vllm_ascend.envs' has no attribute 'VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT'`
    在 `get_finished` 里才炸，并伴随 `malloc(): unsorted double linked list corrupted`
    （进程带着异常继续跑了一段才崩，容易误判成显存/驱动问题）。
13. **deploy 脚本必须在对应节点上执行**：`bash start.sh start proxy` 在别的机器上跑会
    `docker exec` 到本地同名容器（不存在）或探到本机 127.0.0.1:1999 的其它服务，
    打印"proxy 就绪"但实际没起。代理/引擎探活一律打**对端 IP**，不要用 127.0.0.1。
14. **杀 bench 必须在容器内杀**：`pkill -f "measure_round.sh ..."` 只杀本地 wrapper，
    `docker exec ... vllm bench` 的子进程会继续跑。两个 bench 重叠时，proxy 的到达速率是两者之和
    （实测 1.87 req/s ≫ 名义 2.0 的一半），TTFT/吞吐全部失真，且结果文件会互相覆盖。
    正确做法：`docker exec <ctn> pkill -9 -f "[v]llm bench"`，并确认 `ps` 里 bench 进程为 0 再开下一轮。
15. **单次超长压测跑不起来是客户端在生成 prompt**：`vllm bench serve --dataset-name random`
    会先把全部 prompt 物化（64k token × N 条）。1200 条 ≈ 7,700 万 token 尚可（每轮 ~1–3 分钟），
    12,600 条 ≈ 8 亿 token 时客户端会卡在生成阶段、请求量为 0（实测 4 分钟无一个请求到达）。
    ⇒ **长时间长稳要分段**（每段 1200 条），代价是段间有几分钟生成空档——所以"2 h 带压"实际
    需要 ~2.5–3 h 墙钟。
16. **`docker exec` 内 `pkill` 可能静默失败**：两个会话各自踩到——(a) 压测客户端在
    "生成 prompt"阶段时 `pkill -9 -f "[v]llm bench"` 不生效（本次实测两条 bench 重叠，
    必须按 PID 杀并复查进程数为 1）；(b) 清引擎时容器内 pkill 后 9 个孤儿进程仍活着占满
    93.7 GB（友军会话实测）。⇒ 清引擎一律**宿主机侧** `pkill -9 -f '[V]LLM::'` +
    `npu-smi` 核对空闲卡数回到基线，再拉新引擎。
17. **正确性闸门不能用 greedy 文本相等**：不同引擎/内核在近并列位置会翻转
    （`<think>` 之后两个开头 logprob 差 0.02–0.03）；用"P 的 top-1 token 落在 D 的 top-5
    且 |Δlogprob| ≤ 0.1 nat"判据（`correctness_gate.sh`）。KV 错位时 Δ 会是 2–3 nat。
16. **pkill（SIGTERM）后 NPU 显存释放要 >60s**：A 机 V2 崩溃后 2 分钟内重启 V1，
    8 个 worker 在 `init_device` 全灭（`Free memory on device (5.33/91.75 GiB) is less
    than desired GPU memory utilization`）——残留进程的设备上下文还没放完。
    重拉前用 `npu-smi info` 确认每卡 HBM 只剩 ~4.7 GB 基线再启动；
    `VLLM::Worker_TP` defunct 僵尸无害（不占 HBM）。

---

## 6. 阶段沉淀索引（2026-11-08）

| 想看什么 | 去哪 |
|---|---|
| 一页纸结论 / 交付物 / 未完成项 | `docs/PD_hybrid_kv_优化报告.md` §0 |
| 逐阶段实测与负结果 | 本文件 §3–§4 |
| 怎么复现（起停、压测、精度、profile） | 报告 §7 + 本文件 §3.3 的 profile 通路 |
| 合并到主干要做什么 | `docs/PD_hybrid_kv_回合主干清单.md`（22 提交 + §2.1 合并记录） |
| S2 还剩什么、怎么验 | `docs/S2_路径A_混合页距分配器设计稿.md`（§4 六步清单） |
| 踩过的坑（12 条） | 本文件 §5 |

**给下一个人的三句话**：
1. hybrid 打开后 **D 侧 KV 从 1.07M 涨到 4.51M tokens** 是这一轮的全部来源，它由三处代码缺口
   和两个配置开关换来的（§3.3 / §3.2）；
2. **rate 1.8 是运营点，rate 2.0 是过载点** —— 短跑看不出差别，长跑队列会发散（§4）；
3. 正确性闸门**不能用文本相等**，要用首 token logprob（≤0.1 nat）；KV 错位时差 2–3 nat。

## 7. 参考

- 计划全文（本分支）：见工作区 `docs/` 与本文件。
- 方法论：`cannbot-skills/model/model-infer-{profiling,perf-breakdown,kvcache,graph-mode,spec-decode-graph,parallel-analysis,quantization,fusion,multi-stream,prefetch}/`
- 可移植范式：`models/deepseek_v41/`（attention custom op 边界、静态 buffer、quantize-on-write）、
  `models/glm5next/`（投影合并 GEMM、unsummed 残差对、is_internal_router、as_strided cache 视图）。
- **不可移植**：DSA/SFA 稀疏注意力算子族、mHC `npu_hc_pre/post`、KDA 短卷积/递归核、
  Engram 表检索、hash router、DSpark draft —— iquest 是 GQA，
  只能走 `vllm_ascend/attention/attention_v1.py` + `npu_add_rms_norm` 路线。

## 8. 附录：§4.2 C8 启动崩溃法证定论（2026-11-08，A 机合并树 f020be6d2 + V2）

`--kv-cache-dtype mxfp8 --kv-cache-dtype-skip-layers sliding_window --block-size 512`
在 **V2 runner 上启动即崩**（`_reshape_kv_cache_v2:1943`，raw 131,315,712 vs
k_shape [1943,128,1,128]）。机制五环全部对上日志数字（详见设计稿 §3.5）：

1. skip-layers 只在 module 级生效（attention.py:297-322），sliding 层 spec 仍是 bf16；
2. 上游 platform `interface.py:720-762` 混合 quant/高精度页对齐把 **block 512→1,024**、
   shared_page=270,336 写入 `skip_page_size_padded`（日志 "Setting attention block
   size to 1024 tokens"）；
3. 89 个 layer-slot 全部统一到页 270,336（sliding bf16 pad 仅 3.1% 浪费；C8 full 层
   264 B/token 真密度不变）；
4. 私有预算 `worker.py:731-768`：scale = 26/89 = 0.2921 → Available 12.72 GiB → NB=1,943；
5. dtype 门（fp8e4m3fn）拒 overlay → 私有 per-layer buffer → generic tuple 分支
   reshape 崩。**QFA 算子从未执行**。

**反转（修正 11-07/本日早前的三个错误结论）**：

- 2026-11-07 `quant/kv_c8/logs/serve_c8_v3.log`（**V1** + C8 + skip-layers）
  **成功启动并完成生成验证**——旧日志中没有本崩溃签名，"与 11-07 同签名崩溃"是错的；
- 今天才是 **V2+C8 的第一次尝试**：崩溃是 V2 私有分配路径的专属缺口，不是"C8 不可能"；
- 设计稿 §5 "C8 prefill/sink 从未在任何部署跑过"也是错的——V1 跑过（但**精度从未测量**）。

**数字锚点**：预算行 `0.2921 (7028736 bytes/block over 24059904 total page bytes)`
= 26/89 精确；Available 12.72 GiB = 43.5×0.2921；NB 1,943 = 13.658e9/7,028,736；
131,315,712 = 1,943×67,584（C8 页 K 半区）。真 C8 页 = 135,168 B（512-token，kernel
[512] 硬约束）；33,792 页模型作废。

**后果与路径**：算子先验（QFA/sink 精度）走 **V1 runner 零代码**（本日已重启 V1+C8
smoke）；V2 生产化需 S2 Patch 1-4 + **预算语义修正**（私有路径 NB×89×270,336 = 46.7 GB
≫ 13.66 GiB 预算，超约 3.4×——预算把 1,024/512-token 两种块当同格相除）。
skip-layers platform 路径 KV 收益仅 ~12%；真混合页 as_strided 的 ×1.238 仍成立
（=26/21，块尺寸不变）但 block 须 128→512。

### 8.1 run-3 终态与 §4.2 收口（2026-11-08 07:12-07:18）

`BLOCK_SIZE=auto`（省略 --block-size）绕过第二道闸后，V1+C8+skip-layers 在合并树
**完整启动**：4 组 [1024,128,128,128]、预算 `0.2841 (6758400/23789568)` = 25/88、
Available 13.15 GiB、`NPU KV cache size: 448,133 tokens`（concurrency 3.42x）、
生成正常。

- **精度**：GSM8K-100 greedy ×2 = {95, 97}，errors=0（交付 97 / 合并 bf16 {94,95}
  带内）→ **C8 算子（QFA prefill/decode + sink）精度无损**。
- **容量**：448k tokens vs 合并 bf16 2,387,389 = **5.3× 损失**。V1 同样走私有
  per-layer 路径，88 层全按 270,336 B 页分配；sliding 层 128-token 块 × 270,336 页
  = 2,112 B/token（4× 浪费），预算行 bpb = 25×270,336 证实无真混合页。
- **结论**：C8 的失败 100% 在分配/预算层，算子层清零。skip-layers platform 形态
  （无论 V1/V2）整体否决；收益唯一路径 = S2 as_strided（Patch 1-4 + 真混合页 +
  预算语义按 token 归一）。A 机已恢复 optimal bf16（V2+MTP+EP+MC2，07:20 起拉）。
