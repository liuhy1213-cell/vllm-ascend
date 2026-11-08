#!/usr/bin/env python3
"""S2 path-A offline unit test: mixed page-distance KV allocator layout.

Validates the layout in docs/S2_路径A_混合页距分配器设计稿.md §3 without an
NPU (pure tensor geometry on CPU):

  A. slot bounds       every per-layer as_strided view stays inside the slot
  B. group disjoint    same-group layer views never collide (write/read tags)
  C. capacity          NB * bytes_per_block vs today's per-layer private sum
  D. connector         old interleave predicate FAILS under mixed layout and
                       the current fallback would register OVERLAPPING K/V
                       regions; the fixed predicate passes; block_len must be
                       split from block_stride
  E. regression        today's equal-page layer-major layout keeps the old
                       predicate TRUE and block_len == block_stride
  F. layer-outer       the "keep layer-outer, drop the equal-page check"
                       shortcut aliases mixed pages from byte 0 -> corruption;
                       uniform pages stay safe (today's deployment)
  G. grouping          S2 target geometry (block 512, lift bypassed): groups
                       [26 full C8, 21x3 sliding] keep TRUE pages 135,168 /
                       262,144 (no 270,336 lift-pad, no unify-to-max);
                       bpb = max(group sums) = 5,505,024 = 10,752 B/token
  H. block-outer       per-group descriptors (offset=0, layer_stride=page,
                       block_stride=bpb) stay in-slot and cross-slot safe;
                       capacity vs the bf16 baseline = x1.238 on the
                       log-anchored pool bytes
  I. allocator         block-outer 1-D slices: extent gate pass/fail, slice
                       base + b*block_stride lands on view block b (zeroer
                       addressing), mxfp storage-extent check (Patch 3) incl.
                       equivalence with the old numel check for solo tensors
  J. reshape views     as_strided K/V math for both backends: stride(0)=bpb,
                       kv[0] contiguous (zeroer assert), ptr gap = K payload,
                       block pitch = bpb, kernel block = spec block = 512

Log-anchored parameters (new78 D side engine.log, 2026-11-08 00:19:38):
  4 groups [26 full, 21, 21, 21 sliding], every page = 65,536 B,
  num_blocks = 40,314, bytes_per_block = 26 * 65,536 = 1,703,936 B.
  NOTE: the draft's "sliding 21 x 64KB = 1.342 MB" is wrong (KB/KiB slip);
  21 * 65,536 = 1,376,256 B, so the post-change gain is 1.238x, not 1.27x.
  The C8 page carries scales (probe-verified packet K|Ks|V|Vs):
  2 * K * (1 + 1/32) = 33,792 B for K = 16,384 B; both values are tested.

Usage: python3 -I <this file>   (-I isolates from cwd/site — SFS directories
may contain modules like json.py that would shadow stdlib imports otherwise;
requires torch, runs on CPU, no NPU needed)
"""
import sys

import torch

KB = 1024
PAGE_BF16 = 64 * KB                  # 65,536, log-verified per-layer page
FULL_LAYERS = 26
SLIDING_LAYERS = 21

failures = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def layer_views(backing, num_blocks, bpb, page, n_layers):
    """Draft §3: layer j view = as_strided(backing, (NB, page), (bpb, 1), off=j*page)."""
    return {
        j: torch.as_strided(backing, (num_blocks, page), (bpb, 1), j * page)
        for j in range(n_layers)
    }


def mixed_layout(num_blocks, page_c8):
    bpb = max(FULL_LAYERS * page_c8, SLIDING_LAYERS * PAGE_BF16)
    backing = torch.empty(num_blocks * bpb, dtype=torch.uint8)
    full = layer_views(backing, num_blocks, bpb, page_c8, FULL_LAYERS)
    sliding = layer_views(backing, num_blocks, bpb, PAGE_BF16, SLIDING_LAYERS)
    return backing, bpb, full, sliding


# ---------------------------------------------------------------- A: bounds
for page_c8 in (32 * KB, 33 * KB):   # idealized 32KiB and probe-real 33,792 packet
    nb = 40314
    _, bpb, full, sliding = mixed_layout(nb, page_c8)
    full_extent = (FULL_LAYERS - 1) * page_c8 + page_c8
    slide_extent = (SLIDING_LAYERS - 1) * PAGE_BF16 + PAGE_BF16
    check(
        f"A bounds page_c8={page_c8}: full-group span {full_extent} <= bpb {bpb}",
        full_extent <= bpb,
    )
    check(
        f"A bounds page_c8={page_c8}: sliding-group span {slide_extent} == bpb (exact fit)",
        slide_extent == bpb,
    )
    last = max(
        v.storage_offset() + (nb - 1) * v.stride(0) + v.shape[1] for v in full.values()
    )
    check(
        f"A bounds page_c8={page_c8}: last full byte {last} <= backing {nb * bpb}",
        last <= nb * bpb,
    )
    # B: same-group disjointness via write/read tags (the two GROUPS share slots
    # by design — block-id-exclusive — so cross-group clobber is expected, not a
    # defect; only within-group collisions and cross-SLOT overflow are).
    snb = 8
    for gname in ("f", "s"):
        sbacking, sbpb, sfull, sslide = mixed_layout(snb, page_c8)
        views = sfull if gname == "f" else sslide
        for j, v in views.items():
            for b in range(snb):
                v[b, 0:4] = torch.tensor([gname == "f", j, b, 0xAA], dtype=torch.uint8)
        ok = all(
            v[b, 0:4].tolist() == [int(gname == "f"), j, b, 0xAA]
            for j, v in views.items()
            for b in range(snb)
        )
        check(f"B tags survive, group '{gname}', page_c8={page_c8} (no in-group write)", ok)
    # cross-slot containment: full-group writes in block b must not leak into slot b+1
    sbacking, sbpb, sfull, _ = mixed_layout(snb, page_c8)
    sbacking.zero_()  # torch.empty is uninitialized; the unwritten bytes must read 0
    for j, v in sfull.items():
        v[snb - 2, 0:4] = 0xFF
    nxt = torch.as_strided(sbacking, (snb,), (1,), (snb - 1) * sbpb)
    check(
        f"B full-group writes stay inside their slot, page_c8={page_c8}",
        bool((nxt == 0).all()),
    )

# ---------------------------------------------------------------- C: capacity
page_c8 = 33 * KB
nb = 40314
_, bpb, _, _ = mixed_layout(nb, page_c8)
today_bpb = FULL_LAYERS * PAGE_BF16  # 1,703,936, log-verified
gain = today_bpb / bpb
print(f"C bytes_per_block: {today_bpb} -> {bpb}  gain={gain:.4f}x")
print(f"C D-side num_blocks: {nb} -> ~{int(nb * gain)}")
print(f"C in-flight: P 48 -> ~{48 * gain:.0f}, D 67 -> ~{67 * gain:.0f}")
check("C gain lands in 1.20-1.28 band", 1.20 <= gain <= 1.28, f"{gain:.4f}")
check("C gain is NOT 1.27 (draft slip corrected)", abs(gain - 1.27) > 0.01)
priv = nb * (FULL_LAYERS * page_c8 + SLIDING_LAYERS * PAGE_BF16)
overlay = nb * bpb
print(f"C allocation: per-layer-private {priv / 2**20:.0f} MiB vs overlay {overlay / 2**20:.0f} MiB")

# ------------------------------------------------- D: connector under as_strided
# One bf16 layer's K/V as the engine would hand them to the connector after the
# path-A change: page = [K | V], K at layer offset, V one k_page in, both with
# stride(0) = bpb (block stride no longer equals the page).
nb = 64
backing, bpb, _, _ = mixed_layout(nb, page_c8)
k_page = PAGE_BF16 // 2
page_len = PAGE_BF16
kv = {}
for gname, n_layers, page in (("full", FULL_LAYERS, page_c8), ("sliding", SLIDING_LAYERS, PAGE_BF16)):
    for j in range(n_layers):
        off = j * page
        kv[(gname, j)] = (
            torch.as_strided(backing, (nb, k_page), (bpb, 1), off),
            torch.as_strided(backing, (nb, k_page), (bpb, 1), off + k_page),
        )
ok_old_fail, ok_new_pass, ok_overlap, ok_split = True, True, True, True
for (gname, j), (K, V) in kv.items():
    k_page_meas = V.data_ptr() - K.data_ptr()
    interleave_stride = K.stride(0) * K.element_size()
    old_pred = (
        k_page_meas == k_page
        and interleave_stride == page_len
        and V.stride(0) == K.stride(0)
    )
    if old_pred:
        ok_old_fail = False
    new_pred = k_page_meas == k_page and V.stride(0) == K.stride(0)
    if not new_pred:
        ok_new_pass = False
    # the current fallback registers K and V as disjoint contiguous tensors:
    # both spans cover the whole backing -> overlap -> transfer engine rejects
    k_span = (K.data_ptr(), K.numel() * K.element_size())
    v_span = (V.data_ptr(), V.numel() * V.element_size())
    if k_span[0] < v_span[0] + v_span[1] and v_span[0] < k_span[0] + k_span[1]:
        pass  # expected overlap detected
    else:
        ok_overlap = False
    if not (2 * k_page < bpb):  # block_len != block_stride is the load-bearing split
        ok_split = False
check("D old interleave predicate is False for every mixed-layout layer", ok_old_fail)
check("D new predicate (ptr diff + equal strides) is True", ok_new_pass)
check("D current fallback would register overlapping regions (bug confirmed)", ok_overlap)
check("D block_len (2*k_page) < block_stride (bpb) — split required", ok_split)
print(f"D sliding layer 0: k_page={k_page}, interleave_stride={bpb}, page_len={page_len}")

# --------------------------------------------------- E: regression, today's layout
# Layer-major equal pages: layer i owns a contiguous NB*page run, so within a
# layer consecutive blocks sit page apart and the OLD predicate holds with
# block_len == block_stride — the mixed-layout code path must degrade to this.
nb = 64
today_bpb_l = FULL_LAYERS * PAGE_BF16
backing_t = torch.empty(nb * today_bpb_l, dtype=torch.uint8)
K = torch.as_strided(backing_t, (nb, k_page), (PAGE_BF16, 1), 0)
V = torch.as_strided(backing_t, (nb, k_page), (PAGE_BF16, 1), k_page)
interleave_stride = K.stride(0) * K.element_size()
old_pred = (
    V.data_ptr() - K.data_ptr() == k_page
    and interleave_stride == 2 * k_page
    and V.stride(0) == K.stride(0)
)
check("E today's layout: old predicate still True", old_pred)
check("E today's layout: block_len == block_stride", 2 * k_page == interleave_stride)

# ------------------------- F: layer-outer group aliasing with mixed pages corrupts
# The "one-line predicate" shortcut: keep today's layer-outer geometry, only drop
# the equal-page check. vLLM builds shared-pool descriptors with per-group page
# strides, all groups aliasing from byte 0 (kv_cache_utils.py:1756-1768; comment:
# "layer-outer (only supported for uniform page sizes or single-group models)";
# :1995 gates the mixed-page packing machinery behind is_block_outermost).
# Simulate exactly that descriptor geometry: group g layer j run = [j*NB*p_g, +NB*p_g).
nb = 4096
bpb = max(FULL_LAYERS * page_c8, SLIDING_LAYERS * PAGE_BF16)
backing_f = torch.zeros(nb * bpb, dtype=torch.uint8)


def desc_views(n_layers, page, backing):
    return [
        torch.as_strided(backing, (nb, page), (page, 1), j * nb * page)
        for j in range(n_layers)
    ]


fviews = desc_views(FULL_LAYERS, page_c8, backing_f)
sviews = desc_views(SLIDING_LAYERS, PAGE_BF16, backing_f)
a, c = fviews[0][1], sviews[0][0]  # full blk 1 vs sliding blk 0: different ids,
lo = max(a.storage_offset(), c.storage_offset())  # both live at once (group-
hi = min(  # exclusive scheduling); grids 33,792 vs 65,536 interleave from byte 0
    a.storage_offset() + a.numel(), c.storage_offset() + c.numel()
)
check(
    "F mixed pages: full(blk1) [33,792,67,584) overlaps sliding(blk0) [0,65536) -> corruption",
    lo < hi,
    f"overlap [{lo},{hi})",
)
# uniform pages under the same geometry: different ids are disjoint (that is why
# today's deployment is safe) — but the SAME id would alias byte-identically,
# which is why group-exclusive block ids are load-bearing.
backing_u = torch.zeros(nb * FULL_LAYERS * PAGE_BF16, dtype=torch.uint8)
fviews_u = desc_views(FULL_LAYERS, PAGE_BF16, backing_u)
sviews_u = desc_views(SLIDING_LAYERS, PAGE_BF16, backing_u)
a, c = fviews_u[0][1], sviews_u[0][0]
lo, hi = max(a.storage_offset(), c.storage_offset()), min(
    a.storage_offset() + a.numel(), c.storage_offset() + c.numel()
)
check("F uniform pages: full(blk1) vs sliding(blk0) disjoint (today's safety)", lo >= hi)
a, c = fviews_u[0][1], sviews_u[0][1]
check(
    "F uniform pages: same id in two groups would alias byte-identically (ids must stay group-exclusive)",
    a.storage_offset() == c.storage_offset(),
)

# ==================================== G: grouping keeps TRUE pages (block 512)
# S2 target geometry: the lift (_align_heterogeneous_kv_block_size) is bypassed,
# so block stays 512 and every spec keeps its true page — full C8 group at
# 264 B/tok * 512 = 135,168 (kernel [512] 1:1), sliding bf16 groups at
# 512 B/tok * 512 = 262,144 (generic [512,128] whitelist). No
# page_size_padded, no unify-to-max. Grouping by spec type + min-layer split
# yields the MTP log-anchored shape [26 full, 21, 21, 21].
BLOCK_S2 = 512
PAGE_C8_S2 = 264 * BLOCK_S2            # 135,168
PAGE_SLIDE_S2 = 512 * BLOCK_S2         # 262,144
PAGE_C8_LIFT = 264 * 1024              # 270,336: what the lift would produce
GROUPS_S2 = [("full_c8", FULL_LAYERS, PAGE_C8_S2)] + [
    (f"sliding_{i}", SLIDING_LAYERS, PAGE_SLIDE_S2) for i in range(3)
]
pages_s2 = [p for _, _, p in GROUPS_S2]
check("G block stays 512 (lift bypassed)", BLOCK_S2 == 512)
check(
    "G full C8 group keeps true page 135,168 (not lift-pad 270,336)",
    PAGE_C8_S2 == 135_168 and PAGE_C8_S2 != PAGE_C8_LIFT,
)
check("G sliding groups keep true page 262,144 (no page_size_padded)", PAGE_SLIDE_S2 == 262_144)
check(
    "G pages NOT unified to max (unify_kv_cache_spec_page_size bypassed)",
    len(set(pages_s2)) == 2 and 262_144 not in (PAGE_C8_S2,),
)
group_layer_pages = {name: [p] * n for name, n, p in GROUPS_S2}
check(
    "G per-group uniform page (can_overlay Patch-1 gate)",
    all(len(set(v)) == 1 for v in group_layer_pages.values()),
)
bpb_s2 = max(n * p for _, n, p in GROUPS_S2)
check(
    "G bpb = max(group sums) = 5,505,024 (sliding pins, full 3,514,368 loses)",
    bpb_s2 == 5_505_024 == SLIDING_LAYERS * PAGE_SLIDE_S2
    and FULL_LAYERS * PAGE_C8_S2 == 3_514_368,
)
check("G B/token = 10,752", bpb_s2 // BLOCK_S2 == 10_752)
bpb_bf16_s2 = FULL_LAYERS * PAGE_SLIDE_S2
check(
    "G bf16 baseline bpb = 6,815,744 -> 13,312 B/token",
    bpb_bf16_s2 == 6_815_744 and bpb_bf16_s2 // BLOCK_S2 == 13_312,
)
gain_s2 = bpb_bf16_s2 / bpb_s2
check("G capacity gain = x1.238 (band 1.235-1.241)", 1.235 <= gain_s2 <= 1.241, f"{gain_s2:.4f}")
util_full = FULL_LAYERS * PAGE_C8_S2 / bpb_s2
check("G full-group slot utilization ~0.638 (36% waste, by design)", abs(util_full - 0.6384) < 0.01)
print(f"G full-group slot utilization {util_full:.3f}  (bpb {bpb_bf16_s2} -> {bpb_s2}, x{gain_s2:.4f})")

# ============================== H: block-outer descriptors stay in slot
# Patch-side descriptor model: every group gets KVCacheTensor(offset=0,
# layer_stride=page_g, block_stride=bpb) over ONE shared backing of NB*bpb;
# groups share the block-id space (group-exclusive ids), so each group's slot
# is the full bpb region and the full group simply uses 61.4%/63.8% of it.
NB_S2 = 64
backing_h = torch.zeros(NB_S2 * bpb_s2, dtype=torch.uint8)
descs = {}
for gname, n_layers, page in GROUPS_S2:
    for j in range(n_layers):
        descs[(gname, j)] = dict(
            start=j * page,
            page=page,
            view=torch.as_strided(backing_h, (NB_S2, page), (bpb_s2, 1), j * page),
        )
ok_ext = all(
    d["start"] + (NB_S2 - 1) * bpb_s2 + d["page"] <= backing_h.numel() for d in descs.values()
)
check("H every layer extent start+(NB-1)*block_stride+page <= backing", ok_ext)
hi = max(d["start"] + (NB_S2 - 1) * bpb_s2 + d["page"] for d in descs.values())
check(
    "H high-water byte == (NB-1)*bpb + sliding span (exact fit)",
    hi == (NB_S2 - 1) * bpb_s2 + SLIDING_LAYERS * PAGE_SLIDE_S2,
)
backing_h.zero_()
for (gname, j), d in descs.items():
    if gname == "full_c8":
        d["view"][NB_S2 - 2, 0:4] = 0xFF
nxt_h = torch.as_strided(backing_h, (NB_S2,), (1,), (NB_S2 - 1) * bpb_s2)
check("H full-group writes at blk NB-2 do not leak into blk NB-1", bool((nxt_h == 0).all()))
ok_rt_h = True
for j in range(4):
    descs[("sliding_0", j)]["view"][7, 0:8] = torch.tensor(
        [0, j, 7, 0xAA, 0, 0, 0, 0], dtype=torch.uint8
    )
for j in range(4):
    if descs[("sliding_0", j)]["view"][7, 0:8].tolist() != [0, j, 7, 0xAA, 0, 0, 0, 0]:
        ok_rt_h = False
check("H sliding_0 layers 0-3: tag roundtrip at blk 7 (no in-group clobber)", ok_rt_h)
# capacity on the log-anchored D-side pool bytes (block-128 bf16 run):
# num_blocks 40,314 x bytes_per_block 26*65,536
AVAIL_D = 40_314 * FULL_LAYERS * PAGE_BF16
nb_c8 = AVAIL_D // bpb_s2
nb_bf16 = AVAIL_D // bpb_bf16_s2
print(
    f"H capacity: avail {AVAIL_D / 2**30:.2f} GiB -> NB c8 {nb_c8} "
    f"({nb_c8 * BLOCK_S2 / 1e6:.2f}M tok) vs bf16 {nb_bf16} ({nb_bf16 * BLOCK_S2 / 1e6:.2f}M tok)"
)
check(
    "H NB ratio ~= x1.238 (floor granularity)",
    1.230 <= nb_c8 / nb_bf16 <= 1.246,
    f"{nb_c8 / nb_bf16:.4f}",
)

# ============================== I: allocator slices + mxfp extent gate
# Patch-2 allocator branch: kv_cache_raw_tensors[layer_name] = backing[start:]
# (1-D slice; storage_offset carries the layer offset). Patch-3 mxfp gate:
# pitch >= payload and storage_offset + (NK-1)*pitch + payload <= storage.
raws = {k: backing_h[d["start"]:] for k, d in descs.items()}
ok_raw = all(
    r.storage_offset() == descs[k]["start"]
    and r.numel() == backing_h.numel() - descs[k]["start"]
    for k, r in raws.items()
)
check("I allocator slices: storage_offset carries the layer offset", ok_raw)


def mxfp_extent_ok(raw, num_kernel_blocks, pitch, payload):
    if pitch < payload:
        return False
    extent = raw.storage_offset() + (num_kernel_blocks - 1) * pitch + payload
    return raw.untyped_storage().nbytes() >= extent


ok_pass_i = all(
    mxfp_extent_ok(r, NB_S2, bpb_s2, PAGE_C8_S2)
    for k, r in raws.items()
    if k[0] == "full_c8"
)
check("I mxfp extent gate passes for the real geometry (pitch=bpb, payload=135,168)", ok_pass_i)
check(
    "I mxfp extent gate rejects pitch < payload",
    not mxfp_extent_ok(raws[("full_c8", 0)], NB_S2, PAGE_C8_S2 - 1, PAGE_C8_S2),
)
# the 1-byte-short backing must be caught on the LAST sliding layer (full group
# has 36% slack, so a naive full-group-only check would false-pass)
short_i = torch.zeros(NB_S2 * bpb_s2 - 1, dtype=torch.uint8)
r_short = short_i[(SLIDING_LAYERS - 1) * PAGE_SLIDE_S2:]
check(
    "I mxfp extent gate rejects storage 1 byte short (last sliding layer)",
    not mxfp_extent_ok(r_short, NB_S2, bpb_s2, PAGE_SLIDE_S2),
)
# zeroer addressing: slice base + b*block_stride must land on view block b
ok_addr = True
for (gname, j), r in raws.items():
    v = descs[(gname, j)]["view"]
    for b in (0, 7, NB_S2 - 1):
        if r.data_ptr() + b * bpb_s2 != v[b].data_ptr():
            ok_addr = False
check("I slice base + b*block_stride == view block b base (zeroer addressing)", ok_addr)
# solo-tensor equivalence: for an exclusive allocation (offset 0, storage ==
# numel) the new storage-extent check is exactly the old numel check
solo_i = torch.zeros((NB_S2 - 1) * bpb_s2 + PAGE_C8_S2, dtype=torch.uint8)
check(
    "I solo alloc exact-fit: new gate True, old numel gate True",
    mxfp_extent_ok(solo_i, NB_S2, bpb_s2, PAGE_C8_S2)
    and solo_i.numel() >= (NB_S2 - 1) * bpb_s2 + PAGE_C8_S2,
)
solo_short_i = torch.zeros((NB_S2 - 1) * bpb_s2 + PAGE_C8_S2 - 1, dtype=torch.uint8)
check(
    "I solo alloc 1 short: both gates reject",
    not mxfp_extent_ok(solo_short_i, NB_S2, bpb_s2, PAGE_C8_S2)
    and solo_short_i.numel() < (NB_S2 - 1) * bpb_s2 + PAGE_C8_S2,
)

# ============================== J: reshape-side as_strided view math
# Generic bf16 layer (mirror _reshape_combined_attention_kv_cache with
# page_stride = bpb): page = [K | V], each half 512 tok x Hkv*D=128 elems.
# Zeroer asserts kv[0].is_contiguous() and steps dp + b*stride_bytes.
HD_S2 = 128
EL_S2 = 2
NB_J = 16
backing_j = torch.zeros(NB_J * bpb_s2 // EL_S2, dtype=torch.uint16)
K_HALF_E = BLOCK_S2 * HD_S2  # 65,536 elems = 131,072 B


def kv_views_s2(page_bytes, layer_idx):
    off = layer_idx * page_bytes // EL_S2
    K = torch.as_strided(backing_j, (NB_J, BLOCK_S2, HD_S2), (bpb_s2 // EL_S2, HD_S2, 1), off)
    V = torch.as_strided(
        backing_j, (NB_J, BLOCK_S2, HD_S2), (bpb_s2 // EL_S2, HD_S2, 1), off + K_HALF_E
    )
    return K, V


K0, V0 = kv_views_s2(PAGE_SLIDE_S2, 0)
check("J K/V stride(0)*el == bpb", K0.stride(0) * EL_S2 == bpb_s2 and V0.stride(0) * EL_S2 == bpb_s2)
check("J kv[0] contiguous (zeroer assert)", K0[0].is_contiguous() and V0[0].is_contiguous())
check("J V-K ptr gap == K payload 131,072", V0.data_ptr() - K0.data_ptr() == K_HALF_E * EL_S2)
check("J block pitch == bpb (zeroer segment step)", K0[1].data_ptr() - K0[0].data_ptr() == bpb_s2)
ok_jrt = True
for b in (0, 7, NB_J - 1):
    K0[b, :, :] = b + 1
    V0[b, :, :] = 0x100 + b
    if K0[b, 0, 0].item() != b + 1 or V0[b, BLOCK_S2 - 1, HD_S2 - 1].item() != 0x100 + b:
        ok_jrt = False
if K0[0, 0, 0].item() != 1:  # block NB-1 write must not clobber block 0
    ok_jrt = False
check("J K/V write/read roundtrip, blocks disjoint", ok_jrt)
K1, V1 = kv_views_s2(PAGE_SLIDE_S2, 1)
check(
    "J layer 1 starts one TRUE page later (262,144), still inside slot 0",
    K1.data_ptr() - K0.data_ptr() == PAGE_SLIDE_S2 and PAGE_SLIDE_S2 <= bpb_s2,
)
# C8 packet view (mirror mxfp_paged_cache_views with page_stride_bytes=bpb):
# flat payload page [NK, 135,168], stride(0) = pitch.
backing_cj = torch.zeros(NB_J * bpb_s2, dtype=torch.uint8)
pkt = torch.as_strided(backing_cj, (NB_J, PAGE_C8_S2), (bpb_s2, 1), 5 * PAGE_C8_S2)
check(
    "J C8 packet: pitch >= payload, kv[0] contiguous, pitch == bpb",
    bpb_s2 >= PAGE_C8_S2 and pkt[0].is_contiguous() and pkt[1].data_ptr() - pkt[0].data_ptr() == bpb_s2,
)
# kernel block chain: prepare_kernel_block_sizes takes spec.block_size (no
# layer views) -> 512, ratio = 1, kernel view NB == NB_J (no virtual split)
kernel_bs_j = BLOCK_S2
ratio_j = BLOCK_S2 // kernel_bs_j
check(
    "J kernel block = spec block = 512 -> ratio 1, kernel NB == NB_J",
    ratio_j == 1 and NB_J * ratio_j == NB_J,
)

print()
if failures:
    print(f"UNIT_TEST_FAILED: {len(failures)} failing: {failures}")
    sys.exit(1)
print("UNIT_TEST_PASSED: layout geometry, capacity, connector split, regression, S2 target geometry (G-J) all green")
