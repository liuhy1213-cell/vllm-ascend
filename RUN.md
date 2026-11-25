# IQuest-Q1 Phase 1 完整代码（P 节点 eager norm 融合）

本目录是 `va-siflow-alg1` 的**完整副本**（整棵代码树，83 MB），额外包含 Phase 1 补丁，
并且**默认开启**（`_FUSE_NORM_CHAIN` 读 `IQUEST_FUSE_NORM`，默认值改为 "1"）。

改动只有一个文件：
`vllm_ascend/models/iquest_q1/model.py`
- decoder layer 的 eager 路径：
  - 链 A：`attn_out_norm -> *scale -> +x -> feed_forward_norm` 融合为 `npu_add_rms_norm`
  - 链 B：`ffn_out*scale -> +h -> 下一层 attention_norm` 融合为一个 `npu_add_rms_norm`
- `attn_out_scale == 1.0` 时省掉 Muls
- 开关：`IQUEST_FUSE_NORM`（本目录默认 "1"；设为 "0" 即回到原行为）

## 使用

容器内：

```bash
export PYTHONPATH=/mnt/sfs_turbo/work/liuhaiyang/va-iq-phase1:${PYTHONPATH}
# 确认解析到本目录
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"
```

然后按你们现有 P 节点 serve 命令启动（**必须保留 `--enforce-eager`**，本补丁针对 eager 路径）。
D 节点不需要用这个目录（图模式下 Inductor 已有 AddRMSNorm 融合）。

## 关闭 / 回滚

```bash
export IQUEST_FUSE_NORM=0      # 同一份代码，行为回到原版
```
或直接使用原来的 `va-siflow-alg1`。

## 实测（39.146，Ascend 950DT x8，30k token prefill x5，TP8，eager）

- 单请求 wall：0.673-0.678 s -> 0.649-0.654 s（**-3.3%**）
- Norm 链内核时间：452.1 ms -> 354.0 ms（**-21.7%**）
- 内核总时间：5897.9 ms -> 5724.5 ms（-2.9%）

## 注意

- **精度需要整网验证**：融合改变了 bf16 残差加法的物化/舍入点，逐位结果会变。
- 本目录是副本，不会影响 `va-siflow-alg1` 或其它人的目录。
- Phase 2（sequence parallel + reduce_scatter）**不包含**，仍在调试（性能看起来好但数值未通过）。
- 详细 diff / review patch 在：
  `/mnt/sfs_turbo/work/liuhaiyang/phase1-eager-norm-fusion/`
