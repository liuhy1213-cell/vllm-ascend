# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Read-only scheduler snapshots for the LLM Router /v1/loads contract.

Run in the EngineCore utility handler, never in an HTTP/NPU worker thread.
No allocation, cache touching, connector polling or device synchronization.
"""

import math
import time

from vllm_ascend.observability.router_load_history import history_for, install_history_hooks

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


def _prefill_request(manager, req, external_enabled, history):
    status = _status(req)
    preempted = status == "PREEMPTED"
    target = req.num_tokens if preempted else req.num_prompt_tokens
    progress = min(target, max(0, req.num_computed_tokens))
    device_hit = _prefix_hit(manager, req) if progress == 0 and status != "RUNNING" else 0
    accounted = min(target, max(progress, device_hit))
    upper_bound = max(0, target - accounted)
    remote = status == "WAITING_FOR_REMOTE_KVS"
    # num_computed_tokens already includes admitted external hits and scheduler
    # progress. Never subtract either the historical rate or L2 a second time.
    unknown = external_enabled and progress == 0 and status != "RUNNING" and not remote and upper_bound > 0
    sample = history["preempted" if getattr(req, "num_preemptions", 0) > 0 else "normal"]
    minimum_compute = 1 if upper_bound else 0
    if upper_bound and getattr(manager, "use_eagle", False):
        alignment = math.lcm(*(group.block_size for group in manager.coordinator.single_type_managers))
        # MTP must recompute the prompt's final granularity block. A high
        # historical hit rate must never discount that compulsory suffix.
        minimum_compute = min(upper_bound, target - (target - 1) // alignment * alignment)
    estimated_hit = 0
    if unknown and sample["query_tokens"]:
        # Token-weighted conditional rate after L1, not hits / full input and
        # not an average of per-request percentages. Integer arithmetic avoids
        # rounding above the expected hit. Retain at least one compute token.
        estimated_hit = min(upper_bound - minimum_compute, upper_bound * sample["hit_tokens"] // sample["query_tokens"])
    admission = getattr(req, "prefill_stats", None)
    admission_hits = None
    if (
        admission is not None
        and progress > 0
        and getattr(req, "num_preemptions", 0) == 0
        and admission.num_prompt_tokens == target
    ):
        admission_hits = {
            "local_tokens": admission.num_local_cached_tokens,
            "external_tokens": admission.num_external_cached_tokens,
        }
    return {
        "request_id": req.request_id,
        "status": status,
        "target_tokens": target,
        # Scheduled/accounted progress can run ahead of NPU completion. It is
        # not a count of device-synchronized, physically completed tokens.
        "accounted_tokens": accounted,
        "device_lookup_hit_tokens": device_hit if progress == 0 else None,
        "external_lookup_unknown": unknown,
        "external_hit_tokens_estimated": estimated_hit,
        "remaining_tokens": upper_bound - estimated_hit,
        "remaining_tokens_upper_bound": upper_bound,
        "minimum_recompute_tokens": minimum_compute,
        "estimate_source": "external_hit_rate_5m" if unknown and sample["query_tokens"] else "none",
        "admission_cache_hits": admission_hits,
        # Metadata availability differs from transfer progress. This is the
        # originally admitted amount, NOT bytes/tokens still in flight.
        "external_restore_tokens_upper_bound": admission_hits["external_tokens"] if remote and admission_hits else None,
    }


def collect_load(engine):
    scheduler = engine.scheduler
    manager = scheduler.kv_cache_manager
    groups = manager.coordinator.single_type_managers
    if not groups:
        raise ValueError("Router loads require a KV cache group")
    role = _role(engine.vllm_config)
    history = history_for(scheduler)
    external_history = history.snapshot(time.monotonic())
    external_enabled = engine.vllm_config.kv_transfer_config is not None
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
    request_loads = []
    pending_upper = bootstrap_upper = unknown_tokens = estimated_hits = 0
    known_remaining = unknown_reqs = restore_unknown_reqs = restore_tokens_upper = 0
    for request_id, req in requests.items():
        status = _status(req)
        remote = status == "WAITING_FOR_REMOTE_KVS"
        preempted = status == "PREEMPTED"
        transfer_reqs += remote
        retracted_reqs += preempted
        prealloc_reqs += request_id in waiting and not remote and reserved.get(request_id, 0) == 0
        if role != DECODE:
            detail = _prefill_request(manager, req, external_enabled, external_history)
            request_loads.append(detail)
            remaining = detail["remaining_tokens"]
            upper = detail["remaining_tokens_upper_bound"]
            if remote:
                p_bootstrap += remaining
                bootstrap_upper += upper
                restore = detail["external_restore_tokens_upper_bound"]
                if restore is None:
                    restore_unknown_reqs += 1
                else:
                    restore_tokens_upper += restore
            else:
                p_pending += remaining
                pending_upper += upper
            if detail["external_lookup_unknown"]:
                unknown_tokens += upper
                unknown_reqs += 1
            else:
                known_remaining += remaining
            estimated_hits += detail["external_hit_tokens_estimated"]
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
        "schema_version": 2,
        "instance_id": history.instance_id,
        "prefill": {
            "pending_tokens_upper_bound": pending_upper,
            "bootstrap_tokens_upper_bound": bootstrap_upper,
            "known_remaining_tokens": known_remaining,
            "unknown_external_lookup_tokens": unknown_tokens,
            "unknown_external_lookup_reqs": unknown_reqs,
            "estimated_external_hit_tokens": estimated_hits,
            "external_restore_tokens_upper_bound": restore_tokens_upper,
            "external_restore_unknown_reqs": restore_unknown_reqs,
            "external_history": external_history,
            "history_collection_enabled": bool(getattr(scheduler, "log_stats", False)),
        },
    }
    return {
        "dp_rank": engine.vllm_config.parallel_config.data_parallel_rank,
        "router_load": load,
        "requests": request_loads,
    }


def install_engine_method():
    # Lazy import: the endpoint process and CPU tests do not initialize an NPU.
    from vllm.v1.engine.core import EngineCore

    existing = getattr(EngineCore, "get_router_load_snapshot", None)
    if existing is not None and existing is not collect_load:
        raise RuntimeError("EngineCore.get_router_load_snapshot is already registered")
    EngineCore.get_router_load_snapshot = collect_load
    install_history_hooks()
