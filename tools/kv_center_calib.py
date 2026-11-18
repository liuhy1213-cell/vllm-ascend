#!/usr/bin/env python3
"""生成 K 中心化用的 μ 校准文件。

对 25 个 full-attention 层，统计 post-rope K 在 (batch, token) 维上的逐通道均值
→ shape [num_kv_heads, head_dim]。再按 TP rank 切片，每 rank 存
    {layer}-{rank}.pt = tensor([mu_rank, mean_abs_k_rank])   # [2, 1, head_dim]
（运行时只读 [0]）

TP8 且 num_kv_heads=8 → 每 rank 1 个 kv head，按 head 连续切分。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = "/mnt/sfs_turbo/work/liuhaiyang/modelslim_full"
MODEL = "/mnt/sfs_turbo/models/IQuest-Q1"
sys.path.insert(0, REPO)

from msmodelslim.core.const import DeviceType  # noqa: E402
import msmodelslim.model.iquest_q1.loader  # noqa: E402,F401
from msmodelslim.model.iquest_q1.model_adapter import IQuestQ1ModelAdapter  # noqa: E402
from msmodelslim.model.iquest_q1.iquest_q1_model import layer_types  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--out", default=f"{REPO}/kv_center")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device("npu:0")
    adapter = IQuestQ1ModelAdapter("IQuest-Q1", Path(MODEL), True)
    cfg = adapter.config
    types = layer_types(cfg)
    full = [i for i, t in enumerate(types) if t == "full_attention"]

    tok = adapter._load_tokenizer(trust_remote_code=True)
    prompts = json.load(open(f"{REPO}/lab_calib/qwen3_cot_w4a4.json"))[: args.samples]
    ids = [tok(p, return_tensors="pt", truncation=True, max_length=args.seq_len)["input_ids"].to(dev)
           for p in prompts]
    model = adapter.init_model(DeviceType.NPU)
    model.eval()
    embed = model.model.embed_tokens.to(dev)
    hs = [embed(i) for i in ids]
    pos = [torch.arange(i.shape[1], device=dev) for i in ids]

    os.makedirs(args.out, exist_ok=True)
    nkv_full = cfg.num_key_value_heads
    assert nkv_full % args.tp == 0, f"kv heads {nkv_full} 不能被 tp {args.tp} 整除"
    per_rank = nkv_full // args.tp
    print(f"full layers {len(full)}, kv heads {nkv_full}, tp {args.tp} → 每 rank {per_rank} 个 head")

    for idx in range(cfg.num_hidden_layers):
        layer = adapter.load_decoder_if_not_exist(model, idx).to(dev)
        if idx in full:
            attn = layer.self_attn
            acc = torch.zeros(attn.num_kv_heads, attn.head_dim, device=dev, dtype=torch.float64)
            acc_abs = torch.zeros_like(acc)
            ntok = 0
            for j in range(len(ids)):
                x = layer.attention_norm(hs[j])
                b, t, _ = x.shape
                k = attn.k_proj(x).view(b, t, attn.num_kv_heads, attn.head_dim)
                k = attn.k_norm(k)
                cos, sin = attn._rope_cos_sin(pos[j], k.dtype)
                k = attn._apply_rope(k, cos, sin)              # [B,T,H,D]
                acc += k.to(torch.float64).sum(dim=(0, 1))
                acc_abs += k.abs().to(torch.float64).sum(dim=(0, 1))
                ntok += b * t
            mu = (acc / ntok).float()
            mean_abs = (acc_abs / ntok).float()
            for r in range(args.tp):
                sl = slice(r * per_rank, (r + 1) * per_rank)
                torch.save(torch.stack([mu[sl], mean_abs[sl]]),
                           os.path.join(args.out, f"{idx}-{r}.pt"))
            print(f"layer {idx:>2}: mu 范围 [{float(mu.min()):.3f}, {float(mu.max()):.3f}] "
                  f"|mu| 均值 {float(mu.abs().mean()):.4f}", flush=True)
        hs = [layer(pos[j], hs[j]) for j in range(len(ids))]
        model.model.layers.pop(-1)
        del layer
        torch.npu.empty_cache()

    print(f"完成，写到 {args.out}")


if __name__ == "__main__":
    main()
