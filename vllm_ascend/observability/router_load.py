# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Read-only scheduler snapshots for the LLM Router /v1/loads contract.

Run in the EngineCore utility handler, never in an HTTP/NPU worker thread.
No allocation, cache touching, connector polling or device synchronization.
"""

import math
import time

PREFILL = 0
DECODE = 1
UNIFIED = 2


def _role(config):
    transfer = config.kv_transfer_config
    if transfer is None:
        return UNIFIED
    if transfer.kv_role == "kv_producer":
        return PREFILL
    if transfer.kv_role == "kv_consumer":
        return DECODE
    if transfer.kv_role == "kv_both":
        return UNIFIED
    raise ValueError(f"Unsupported KV transfer role: {transfer.kv_role}")


def _status(request):
    return request.status.name


def _prefix_hit(manager, request):
    if not manager.prefix_cache_lookup_enabled(request):
        return 0
    # get_computed_blocks may emit BlockStored events. Query the coordinator
    # directly: same native hybrid/SWA/MTP lookup, without stats/events/touch.
    result = manager.coordinator.find_longest_cache_hit(request.block_hashes, max(0, request.num_tokens - 1))
    return int(result[1])


def collect_load(engine):
    scheduler = engine.scheduler
    manager = scheduler.kv_cache_manager
    groups = manager.coordinator.single_type_managers
    if not groups:
        raise ValueError("Router loads require a KV cache group")
    role = _role(engine.vllm_config)
    page_size = math.lcm(*(group.block_size for group in groups))

    # Active physical pages per group; deduplicate shared prefixes and ignore
    # null SWA placeholders. Retained transfer pages still have owners here.
    # Use the largest group footprint, not the sum over attention groups:
    # tokens are a logical capacity unit, not a count of layers or bytes.
    used_by_group = []
    reserved = {}
    for group in groups:
        active_blocks = set()
        for request_id, blocks in group.req_to_blocks.items():
            active_blocks.update(block.block_id for block in blocks if not block.is_null)
            # Null SWA holes occupy logical positions but no physical pages.
            reserved[request_id] = max(reserved.get(request_id, 0), len(blocks) * group.block_size)
        used_by_group.append(len(active_blocks) * group.block_size)
    used_tokens = max(used_by_group)

    waiting = {req.request_id: req for req in scheduler.waiting}
    waiting.update((req.request_id, req) for req in scheduler.skipped_waiting)
    running = {req.request_id: req for req in scheduler.running}
    requests = waiting | running
    p_pending = p_bootstrap = d_pending = 0
    transfer_reqs = retracted_reqs = prealloc_reqs = 0
    for request_id, req in requests.items():
        status = _status(req)
        remote = status == "WAITING_FOR_REMOTE_KVS"
        preempted = status == "PREEMPTED"
        transfer_reqs += remote
        retracted_reqs += preempted
        prealloc_reqs += request_id in waiting and not remote and reserved.get(request_id, 0) == 0
        if role != DECODE:
            # A preempted request must also recompute its generated tokens.
            target = req.num_tokens if preempted else req.num_prompt_tokens
            computed = req.num_computed_tokens
            if computed == 0:
                computed = _prefix_hit(manager, req)
            remaining = max(0, target - computed)
            if remote:
                p_bootstrap += remaining
            else:
                p_pending += remaining
        if role != PREFILL:
            # Preallocated/transfer slots are already in used_tokens. A chunked
            # prefill reserves only part of its prompt; count the remainder.
            d_pending += max(0, req.num_tokens - reserved.get(request_id, 0))

    seq = max(time.time_ns() // 1000, getattr(engine, "_router_load_seq", 0) + 1)
    engine._router_load_seq = seq
    load = {
        "role": role,
        "page_size": page_size,
        "seq": seq,
        "p_pending_uncached_tok": p_pending,
        "p_bootstrap_uncached_tok": p_bootstrap,
        "d_used_kv_tok": used_tokens if role != PREFILL else 0,
        "d_pending_raw_tok": d_pending,
        "kv_used_tok": used_tokens,
        "token_usage": manager.block_pool.get_usage(),
        "running_reqs": len(running),
        "waiting_reqs": len(waiting),
        "bootstrap_reqs": transfer_reqs if role != DECODE else 0,
        "inflight_reqs": len(requests),
        "prealloc_reqs": prealloc_reqs if role != PREFILL else 0,
        "transfer_reqs": transfer_reqs,
        "retracted_reqs": retracted_reqs,
    }
    return {"dp_rank": engine.vllm_config.parallel_config.data_parallel_rank, "router_load": load}


def install_engine_method():
    # Lazy import: the endpoint process and CPU tests do not initialize an NPU.
    from vllm.v1.engine.core import EngineCore

    existing = getattr(EngineCore, "get_router_load_snapshot", None)
    if existing is not None and existing is not collect_load:
        raise RuntimeError("EngineCore.get_router_load_snapshot is already registered")
    EngineCore.get_router_load_snapshot = collect_load
