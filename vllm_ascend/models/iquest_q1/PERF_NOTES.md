# IQuest-Q1 PD 分离部署性能优化记录

> 本文件是 IQuest-Q1 在 **PD 分离（1P1D）部署**下的性能工作日志。
> 每次改动的**代码与本条记录放在同一个 commit**，保证代码与记录永不脱节。
> 记录里的所有数字必须来自
> `deploy/iquest-1p1d-2m-mooncake-new78/logs/bench/*.json` 与两侧 `engine.log`。
>
> 模型的移植说明、sink 数学、ACL graph 细节见同目录的 `README.md`；本文件只记性能。

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

### 2.5 吞吐模型（说明 6000 为什么可达）

现状：并发上限 16（P 15.9 / D 16.8 取小），每请求驻留 ≈ 20 s
（TTFT + 1000 token × ~16 ms TPOT）⇒ 到达率上限 16/20 ≈ **0.8 req/s**
⇒ 输出 541.7 tok/s ⇒ **2,201 tok/s/卡**。

hybrid 打开后并发上限 ≈ 48 ⇒ 到达率 ≈ 2.4 req/s ⇒ 输出 ≈ 1,600 tok/s ⇒ **≈ 6,000 tok/s/卡**，
且 **TPOT 仍是 16 ms**（远低于 25 ms SLA）。
P 侧供给余量充足（实测峰值 38.5k tok/s，2.4 req/s × 6,400 = 15.4k 需求）。

---

## 3. 阶段计划

| 阶段 | 内容 | 状态 |
|---|---|---|
| S0 | 分支 + 基线固化 + 生产侧 EE1006 同步 | ⏳ 进行中 |
| S1 | **重新启用 hybrid KV cache manager**（核心解锁） | ⬜ |
| S2 | KV-C8（mxfp8 KV cache）落在 25 个 full 层 | ⬜ |
| S3 | D 侧步进优化（并发上去后的 TPOT 约束） | ⬜ |
| S4 | W4A4C8 权重 | ⬜ |
| S5 | 精度闸门 + 长稳 + 回合主干 | ⬜ |

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
