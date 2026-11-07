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
| S0 | 分支 + 基线固化 + 生产侧 EE1006 同步 | 🟡 分支/记录已建；基线复测与生产同步未做 |
| S1 | 重新启用 hybrid KV cache manager（核心解锁） | ⬜ |
| S2 | KV-C8（mxfp8 KV cache）落在 25 个 full 层 | ⬜ |
| S3 | D 侧步进优化 | ⬜ |
| S4 | W4A4C8 权重 | ⬜ |
| S5 | 精度闸门 + 长稳 + 回合主干 | ⬜ |

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
   明确不支持滑窗 attention ⇒ **本计划最大的技术阻塞项**。

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

### S3 D 侧步进

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

### S4 W4A4C8

切 `/mnt/sfs_turbo/models/IQuest-Q1-W4A4C8-mxfp4-20261107`（已跑通、冒烟过、GSM8K-100=96）。
权重流 38.6 GB/卡 → ~20 GB/卡，decode 步长硬下限 ~26 ms → ~13 ms。

---

## 4. 测量记录

> 每阶段一行。`total/16` = `total_token_throughput / 16`（tok/s/卡）。
> 噪声：高并发下同配置 ±5%、TPOT ±1 ms ⇒ **关键结论 N≥2 复测取中位**。

| 阶段 | 配置 | 并发/rate | total/16 | mTPOT | p99 TTFT | 备注 |
|---|---|---|---|---|---|---|
| 基线 | new78 原样（-opt@d8a436033） | rate 0.7 | **2,201** | 16.35 ms | 1.55 s | `logs/bench/pd_open07r1.json` |
| 基线 | 同上 | c24 | 3,312 | 17.16 ms | 24.18 s ❌ | `pd_c24r1.json` |
| 基线 | 同上 | c48 | 3,270 | 17.33 ms | 45.81 s ❌ | `pd_c48r1.json` |
| | | | | | | |
| S1 | hybrid KV + 两侧同时 | | | | | 待测 |
| S2 | + KV-C8 | | | | | 待测 |
| S3 | + 步进优化 | | | | | 待测 |
| S4 | W4A4C8 | | | | | 待测 |

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

---

## 6. 参考

- 计划全文（本分支）：见工作区 `docs/` 与本文件。
- 方法论：`cannbot-skills/model/model-infer-{profiling,perf-breakdown,kvcache,graph-mode,spec-decode-graph,parallel-analysis,quantization,fusion,multi-stream,prefetch}/`
- 可移植范式：`models/deepseek_v41/`（attention custom op 边界、静态 buffer、quantize-on-write）、
  `models/glm5next/`（投影合并 GEMM、unsummed 残差对、is_internal_router、as_strided cache 视图）。
- **不可移植**：DSA/SFA 稀疏注意力算子族、mHC `npu_hc_pre/post`、KDA 短卷积/递归核、
  Engram 表检索、hash router、DSpark draft —— iquest 是 GQA，
  只能走 `vllm_ascend/attention/attention_v1.py` + `npu_add_rms_norm` 路线。
