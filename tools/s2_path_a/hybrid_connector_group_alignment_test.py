#!/usr/bin/env python3
"""Offline verification of 17's on-machine confirmation item (1):

Patch 7's group-alignment logic in _transfer_kv_cache_all_groups under the
D3 geometry (D side 5 groups [full 25, sl 21, 21, 21, draft 1], P side 4).
A real KVCacheRecvingThread is constructed with a fake transfer engine;
paths exercised: draft-last skip, misordered-draft loud error, extra-local-
group skip, equal-stride run fast path, asymmetric per-block fallback.
"""
import sys
import types

import torch

# Needs a tree carrying Patch 7: PYTHONPATH=<worktree> python3 tools/... 
# or run inside a container with the tree copied to /tmp/s2_ut.
sys.path.insert(0, "/tmp/s2_ut")

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (  # noqa: E402
    KVCacheRecvingThread,
)

fails = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        fails.append(name)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def batch_transfer_sync_read(self, session, src, dst, lens):
        self.calls.append((list(src), list(dst), list(lens)))
        return 0


# D-side geometry: 5 groups [25, 21, 21, 21, 1] -> 89 layers.
GROUP_SIZES = [25, 21, 21, 21, 1]
GROUP_STRIDES = [111, 222, 222, 222, 333]  # per-layer block stride by group
GROUP_LENS = [10, 20, 20, 20, 30]  # per-layer page payload by group


def d_side_layout():
    addr_group_idx, strides, lens = [], [], []
    for gi, size in enumerate(GROUP_SIZES):
        for _ in range(size):
            addr_group_idx.append({gi})
            strides.append(GROUP_STRIDES[gi])
            lens.append(GROUP_LENS[gi])
    return addr_group_idx, strides, lens


def make_thread(engine, draft_group_pos=None):
    """Build the recving thread; draft_group_pos moves the draft's group for
    the misordering scenario (None = draft-last, the required order)."""
    sizes = GROUP_SIZES
    if draft_group_pos is not None:
        # Rotate so the draft group really sits at draft_group_pos: the group
        # COUNTS then misalign with the peer's [25, 21, 21, 21].
        r = GROUP_SIZES.index(1)  # draft's current position
        sizes = GROUP_SIZES[r:] + GROUP_SIZES[:r]
    addr_group_idx, strides, lens = [], [], []
    for gi, size in enumerate(sizes):
        for _ in range(size):
            addr_group_idx.append({gi})
            strides.append(GROUP_STRIDES[gi % len(GROUP_STRIDES)])
            lens.append(GROUP_LENS[gi % len(GROUP_LENS)])
    n_layers = len(addr_group_idx)
    local_addrs = list(range(1000, 1000 + n_layers))
    remote_addrs = list(range(5000, 5000 + 75))  # P side: 4 groups, 75 layers

    t = KVCacheRecvingThread(
        tp_rank=0,
        tp_size=1,
        _prefill_pp_size=1,
        engine=engine,
        local_engine_id="D0",
        local_handshake_port=1111,
        side_channel_port=2222,
        local_kv_caches_base_addr=local_addrs,
        block_len_per_addr=lens,
        block_stride_per_addr=strides,
        addr_group_idx=addr_group_idx,
        mamba_ssm_size=(0, 0),
        use_hybrid=True,
        has_mamba=False,
        hma_group_size=5,
        ready_event=None,
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                is_deepseek_mla=False,
                hf_config=types.SimpleNamespace(),
                hf_text_config=types.SimpleNamespace(
                    head_dim=128, num_key_value_heads=8, num_hidden_layers=88
                ),
            ),
            cache_config=types.SimpleNamespace(block_size=512),
        ),
        kv_cache_config=types.SimpleNamespace(
            kv_cache_tensors=[None] * 5,
            kv_cache_groups=[types.SimpleNamespace(kv_cache_spec=None) for _ in range(5)],
        ),
        kv_caches={"l0": (torch.zeros(1),)},
    )
    t.kv_cache_specs = [None] * 5
    t.kv_caches_base_addr["P0"] = {1111: remote_addrs}
    t.remote_te_port = {"P0": {1111: 7777}}
    t.remote_group_layer_counts = {"P0": {1111: [25, 21, 21, 21]}}
    t.remote_block_stride_per_addr = {
        "P0": {1111: [GROUP_STRIDES[0]] * 25 + [GROUP_STRIDES[1]] * 50}
    }
    return t


def req_meta(remote_ids, local_ids):
    return {
        "remote_request_id": "r1",
        "remote_block_ids": remote_ids,
        "local_block_ids": local_ids,
        "remote_engine_id": "P0",
        "remote_host": "10.0.0.1",
        "remote_handshake_port": 1111,
        "offset": 0,
        "tp_num_need_pulls": 1,
    }


# --- Scenario A: draft-last, D 5 groups vs P 4 groups -> skip extra group ----
eng = FakeEngine()
t = make_thread(eng)
t._transfer_kv_cache_all_groups(
    req_meta(
        remote_ids=[[0, 1, 2], [5, 6, 7], [8], [9]],
        local_ids=[[0, 1, 2], [5, 6, 7], [], [9], [3]],
    )
)
check("A1 one transfer call", len(eng.calls) == 1)
src, dst, lens = eng.calls[0]
# Group 0 run [0,1,2] on layer 0: src=1000+0*111, dst=5000+0*111, len=2*111+10
check(
    "A2 group0 run fast path (src/dst/len)",
    src[0] == 1000 and dst[0] == 5000 and lens[0] == 232,
    f"src={src[0]} dst={dst[0]} len={lens[0]}",
)
# Draft layer is layer 88 with group tag {4} -> must NOT appear: every src
# address for the draft layer would be 1000+88*333=30_304.
# Synthetic layout: layer k base = 1000+k, block offset = block*stride_g.
check("A3 draft layer never transferred", 1000 + 88 + 3 * 333 not in src, f"{len(src)} srcs")
# Empty local group 2 (block 8, layers k=46..66) contributes nothing.
check("A4 empty local group skipped", all(1000 + k + 8 * 222 not in src for k in range(46, 67)))
# Group 3 block 9 on its first layer (k=67, stride 222).
check("A5 group3 single block pulled", 1000 + 67 + 9 * 222 in src, f"{sorted(set(lens))}")

# --- Scenario B: draft misordered into group 0 -> loud RuntimeError ----------
eng2 = FakeEngine()
t2 = make_thread(eng2, draft_group_pos=0)
try:
    t2._transfer_kv_cache_all_groups(
        req_meta(
            remote_ids=[[0, 1, 2], [5, 6, 7], [8], [9]],
            local_ids=[[0, 1, 2], [5, 6, 7], [], [9], [3]],
        )
    )
    check("B1 misordered draft raises RuntimeError", False, "no exception")
except RuntimeError as e:
    check("B1 misordered draft raises RuntimeError", "misalignment" in str(e).lower() or "sort LAST" in str(e), str(e)[:80])
check("B2 no transfer executed on misalignment", len(eng2.calls) == 0)

# --- Scenario C: asymmetric strides -> per-block, each side its own stride ---
eng3 = FakeEngine()
t3 = make_thread(eng3)
t3.remote_block_stride_per_addr = {"P0": {1111: [999] * 75}}  # peer pitches differ
t3._transfer_kv_cache_all_groups(
    req_meta(
        remote_ids=[[0, 1, 2], [5, 6, 7], [8], [9]],
        local_ids=[[0, 1, 2], [5, 6, 7], [], [9], [3]],
    )
)
src3, dst3, lens3 = eng3.calls[0]
# Layer 0 (group 0): per-block fallback, 3 blocks; src step=111 (local), dst step=999 (peer)
check(
    "C1 asymmetric: src uses local stride",
    1000 + 0 * 111 in src3 and 1000 + 1 * 111 in src3 and 1000 + 2 * 111 in src3,
)
check(
    "C2 asymmetric: dst uses PEER stride",
    5000 + 0 * 999 in dst3 and 5000 + 1 * 999 in dst3 and 5000 + 2 * 999 in dst3,
)
check("C3 asymmetric: lengths are single pages", set(lens3) == {10, 20}, f"{sorted(set(lens3))}")

print()
if fails:
    print(f"ALIGN_FAILED: {fails}")
    sys.exit(1)
print("ALIGN_PASSED: draft-last skip / misorder loud error / asymmetric strides all behave as designed")
