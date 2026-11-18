# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU contracts: native vLLM Request/KVCacheManager, no NPU model required."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import torch
from fastapi import FastAPI
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.observability.router_load import collect_load, install_engine_method
from vllm_ascend.observability.router_load_api import LoadReader, RouterLoadEndpoint, read_engine_loads


def make_engine(role=None, rank=0, groups=1):
    init_none_hash(sha256)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.float16)
    config = KVCacheConfig(1024, [], [KVCacheGroupSpec([f"layer{i}"], spec) for i in range(groups)])
    manager = KVCacheManager(config, 4096, scheduler_block_size=16, hash_block_size=16, enable_kv_cache_events=True)
    return NS(
        scheduler=NS(kv_cache_manager=manager, waiting=[], skipped_waiting=[], running=[]),
        vllm_config=NS(
            parallel_config=NS(data_parallel_rank=rank),
            kv_transfer_config=NS(kv_role=role) if role is not None else None,
        ),
    )


def req(name, tokens=65):
    return Request(
        name,
        list(range(tokens)),
        SamplingParams(max_tokens=32),
        None,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def test_idle_roles_sequence_and_rank():
    for role, expected in [(None, 2), ("kv_both", 2), ("kv_producer", 0), ("kv_consumer", 1)]:
        engine = make_engine(role, rank=7)
        a, b = collect_load(engine), collect_load(engine)
        assert a["dp_rank"] == 7
        assert a["router_load"]["role"] == expected
        assert a["router_load"]["page_size"] == 16
        assert b["router_load"]["seq"] > a["router_load"]["seq"]
        assert a["router_load"]["d_used_kv_tok"] == 0
        assert a["router_load"]["p_pending_uncached_tok"] == 0


def test_native_prefix_hit_free_cache_and_read_only_lookup():
    engine = make_engine()
    scheduler = engine.scheduler
    manager = scheduler.kv_cache_manager
    warm = req("warm")
    assert manager.allocate_slots(warm, 64) is not None
    manager.cache_blocks(warm, 64)
    manager.free(warm)
    query = req("waiting")
    query.kv_cache_report_mode = "full"
    scheduler.waiting.append(query)
    events_before = list(manager.block_pool.kv_event_queue)
    free_before = manager.block_pool.get_num_free_blocks()
    blocks_before = {k: list(v) for k, v in manager.coordinator.single_type_managers[0].req_to_blocks.items()}
    load = collect_load(engine)["router_load"]
    assert load["p_pending_uncached_tok"] == 1  # native prefix hit = 64/65
    assert load["d_used_kv_tok"] == 0  # free-but-cached pages are NOT active KV
    assert load["d_pending_raw_tok"] == 65
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert list(manager.block_pool.kv_event_queue) == events_before
    assert dict(manager.coordinator.single_type_managers[0].req_to_blocks) == blocks_before
    assert query.num_computed_tokens == 0


def test_shared_pages_not_double_counted_across_requests_or_groups():
    engine = make_engine(groups=2)
    manager = engine.scheduler.kv_cache_manager
    warm = req("warm")
    manager.allocate_slots(warm, 64)
    manager.cache_blocks(warm, 64)
    query = req("shared")
    blocks, hit, _ = manager.get_computed_blocks(query)
    assert hit == 64
    manager.allocate_slots(query, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks)
    for request in [warm, query]:
        request.num_computed_tokens = 65
        request.status = RequestStatus.RUNNING
    engine.scheduler.running = [warm, query]
    load = collect_load(engine)["router_load"]
    assert load["d_used_kv_tok"] == 80  # four shared pages + one new, NOT times two groups
    assert load["p_pending_uncached_tok"] == 0


def test_chunked_prefill_retraction_remote_preallocation_and_release():
    engine = make_engine("kv_consumer")
    scheduler = engine.scheduler
    manager = scheduler.kv_cache_manager
    transfer = req("transfer", 64)
    manager.allocate_slots(transfer, 64)
    transfer.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.skipped_waiting = [transfer]
    retracted = req("retracted", 32)
    retracted.status = RequestStatus.PREEMPTED
    scheduler.waiting = [retracted]
    load = collect_load(engine)["router_load"]
    assert load["d_used_kv_tok"] == 64
    assert load["d_pending_raw_tok"] == 32  # transfer is already fully allocated
    assert load["transfer_reqs"] == load["retracted_reqs"] == 1
    assert load["p_pending_uncached_tok"] == load["p_bootstrap_uncached_tok"] == 0
    manager.free(transfer)
    scheduler.skipped_waiting.clear()
    assert collect_load(engine)["router_load"]["d_used_kv_tok"] == 0

    engine = make_engine()
    chunk = req("chunk", 100)
    engine.scheduler.kv_cache_manager.allocate_slots(chunk, 32)
    chunk.num_computed_tokens = 32
    engine.scheduler.running = [chunk]
    load = collect_load(engine)["router_load"]
    assert load["p_pending_uncached_tok"] == load["d_pending_raw_tok"] == 68
    assert load["d_used_kv_tok"] == 32


def test_producer_scope_and_skip_prefix_read():
    engine = make_engine("kv_producer")
    a = req("a", 40)
    engine.scheduler.waiting = [a]
    load = collect_load(engine)["router_load"]
    assert load["p_pending_uncached_tok"] == 40
    assert load["d_used_kv_tok"] == load["d_pending_raw_tok"] == 0
    engine.scheduler.kv_cache_manager.enable_caching = False
    engine.scheduler.kv_cache_manager.coordinator.find_longest_cache_hit = Mock(side_effect=AssertionError("lookup"))
    assert collect_load(engine)["router_load"]["p_pending_uncached_tok"] == 40


def test_null_holes_and_shared_pages_in_hybrid_groups():
    engine = make_engine("kv_consumer")
    block = lambda idx, null=False: NS(block_id=idx, is_null=null)
    engine.scheduler.kv_cache_manager.coordinator.single_type_managers = [
        NS(block_size=16, req_to_blocks={"a": [block(1), block(2)], "b": [block(1), block(3)]}),
        NS(block_size=32, req_to_blocks={"a": [block(0, True), block(4)]}),
    ]
    engine.scheduler.skipped_waiting = [req("a", 64)]
    load = collect_load(engine)["router_load"]
    assert load["page_size"] == 32
    assert load["d_used_kv_tok"] == 48  # max(3*16, 1*32); ignore null and duplicate page
    assert load["d_pending_raw_tok"] == 0  # SWA holes still cover logical positions


def test_engine_method_registration_is_idempotent(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    monkeypatch.delattr(EngineCore, "get_router_load_snapshot", raising=False)
    install_engine_method()
    install_engine_method()
    try:
        engine = make_engine()
        assert EngineCore.get_router_load_snapshot(engine)["dp_rank"] == 0
    finally:
        del EngineCore.get_router_load_snapshot


def test_dp_transport_keeps_every_rank_and_single_rank():
    async def run():
        core = NS(core_engines=[b"one", b"two"])

        async def call(method, *, engine):
            assert method == "get_router_load_snapshot"
            return collect_load(make_engine(rank=3 if engine == b"one" else 1))

        core._call_utility_async = call
        assert [x["dp_rank"] for x in await read_engine_loads(NS(engine_core=core))] == [1, 3]
        single = NS(call_utility_async=AsyncMock(return_value=collect_load(make_engine(rank=9))))
        assert (await read_engine_loads(NS(engine_core=single)))[0]["dp_rank"] == 9

    asyncio.run(run())


def test_http_contract_alias_sections_errors_and_dp_failure():
    async def run():
        app = FastAPI()
        plugin = RouterLoadEndpoint()
        plugin.attach_router(app)
        core = NS(call_utility_async=AsyncMock(side_effect=lambda method: collect_load(make_engine())))
        await plugin.init_state(NS(engine_core=core), app.state, None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for path in ["/v1/loads", "/v1/load"]:
                response = await client.get(path + "?include=core,disagg,router_load")
                assert response.status_code == 200
                load = response.json()["loads"][0]
                assert {"dp_rank", "num_running_reqs", "num_waiting_reqs", "disaggregation", "router_load"} <= set(load)
                assert load["router_load"]["schema_version"] == 2
            response = await client.get("/v1/loads?include=core")
            assert "router_load" not in response.json()["loads"][0]
            assert "num_running_reqs" in response.json()["loads"][0]
            assert (await client.get("/v1/loads?include=unknown")).status_code == 400
            # Simulate failure with no valid cached snapshot available.
            reader = app.state.ascend_router_load_reader
            reader.cached = None
            core.call_utility_async.side_effect = RuntimeError("engine unavailable")
            response = await client.get("/v1/loads")
            assert response.status_code == 503 and "loads" not in response.json()

    asyncio.run(run())


def test_timeout_cancel_and_one_failed_rank_do_not_accumulate_rpcs(monkeypatch):
    monkeypatch.setattr("vllm_ascend.observability.router_load_api.LOAD_TIMEOUT_SECONDS", 0.01)

    async def run():
        release = asyncio.Event()
        calls = []

        async def call(method, *, engine):
            calls.append(engine)
            if engine == b"bad":
                raise RuntimeError("failed rank")
            await release.wait()
            return collect_load(make_engine())

        reader = LoadReader(NS(engine_core=NS(core_engines=[b"bad", b"slow"], _call_utility_async=call)))
        for _ in range(3):
            with pytest.raises(TimeoutError):
                await reader.read()
        assert calls == [b"bad", b"slow"]
        release.set()
        with pytest.raises(RuntimeError, match="DP rank"):
            await reader.pending

    asyncio.run(run())


def test_native_engine_core_utility_dispatch_and_wire_serialization(monkeypatch):
    import queue

    import msgspec
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState

    state = make_engine(rank=4)
    core = object.__new__(EngineCoreProc)
    core.scheduler = state.scheduler
    core.vllm_config = state.vllm_config
    core.output_queue = queue.Queue()
    core.shutdown_state = EngineShutdownState.RUNNING
    monkeypatch.setattr(EngineCore, "get_router_load_snapshot", collect_load, raising=False)
    core._handle_client_request(EngineCoreRequestType.UTILITY, (2, 123, "get_router_load_snapshot", []))
    client, output = core.output_queue.get_nowait()
    assert client == 2 and output.utility_output.call_id == 123
    assert output.utility_output.failure_message is None
    wire = msgspec.msgpack.decode(msgspec.msgpack.encode(output.utility_output.result.result))
    assert wire["dp_rank"] == 4 and wire["router_load"]["role"] == 2


def test_cancelled_http_waiter_does_not_cancel_pending_query(monkeypatch):
    monkeypatch.setattr("vllm_ascend.observability.router_load_api.LOAD_TIMEOUT_SECONDS", 1)

    async def run():
        release = asyncio.Event()

        async def call(method):
            await release.wait()
            return collect_load(make_engine())

        core = NS(call_utility_async=AsyncMock(side_effect=call))
        reader = LoadReader(NS(engine_core=core))
        task = asyncio.create_task(reader.read())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not reader.pending.cancelled()
        waiter = asyncio.create_task(reader.read())
        release.set()
        assert (await waiter)[0]["dp_rank"] == 0
        assert core.call_utility_async.await_count == 1

    asyncio.run(run())


@pytest.mark.parametrize("mtp", [False, True])
def test_native_hybrid_swa_and_mtp_prefix_boundaries(mtp):
    engine = make_engine()
    full = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.float16)
    swa = SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.float16, sliding_window=32)
    config = KVCacheConfig(
        1024, [], [KVCacheGroupSpec(["full"], full), KVCacheGroupSpec(["swa"], swa, is_eagle_group=mtp)]
    )
    manager = KVCacheManager(
        config, 4096, scheduler_block_size=16, hash_block_size=16, use_eagle=mtp, enable_kv_cache_events=True
    )
    engine.scheduler.kv_cache_manager = manager
    warm = req("warm", 129)
    manager.allocate_slots(warm, 128)
    manager.cache_blocks(warm, 128)
    manager.free(warm)
    query = req("query", 129)
    query.kv_cache_report_mode = "full"
    engine.scheduler.waiting = [query]
    before_events = list(manager.block_pool.kv_event_queue)
    load = collect_load(engine)["router_load"]
    assert list(manager.block_pool.kv_event_queue) == before_events
    native_hit = manager.get_computed_blocks(query)[1]
    assert native_hit > 0
    assert load["p_pending_uncached_tok"] == 129 - native_hit


@pytest.mark.parametrize("allowlist", [None, "", "other_plugin", "ascend_router_load,other_plugin"])
def test_default_registration_without_plugin_configuration(monkeypatch, allowlist):
    import importlib.metadata
    import os

    import vllm.plugins as plugins
    from vllm.plugins.endpoint_plugins.interface import attach_endpoint_plugins, init_endpoint_plugins_state
    from vllm.v1.engine.core import EngineCore

    from vllm_ascend import register

    if allowlist is None:
        monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    else:
        monkeypatch.setenv("VLLM_PLUGINS", allowlist)
    # Restore global loader functions and native once flag after each scenario.
    monkeypatch.setattr(plugins, "load_general_plugins", plugins.load_general_plugins)
    monkeypatch.setattr(plugins, "load_endpoint_plugins", plugins.load_endpoint_plugins)
    monkeypatch.setattr(plugins, "plugins_loaded", False)
    monkeypatch.delattr(EngineCore, "get_router_load_snapshot", raising=False)
    other_general = Mock()
    other_endpoint = NS(name="other_plugin", required_tasks=("generate",))

    def entry_points(*, group):
        factory = (lambda: other_endpoint) if group == plugins.ENDPOINT_PLUGINS_GROUP else other_general
        return [NS(name="other_plugin", value="test:other_plugin", load=lambda: factory)]

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    assert register() == "vllm_ascend.platform.NPUPlatform"
    general, endpoint_loader = plugins.load_general_plugins, plugins.load_endpoint_plugins
    register()
    assert plugins.load_general_plugins is general
    assert plugins.load_endpoint_plugins is endpoint_loader
    plugins.load_general_plugins()
    plugins.load_general_plugins()
    assert EngineCore.get_router_load_snapshot is collect_load
    assert other_general.call_count == (1 if allowlist is None or "other_plugin" in allowlist else 0)
    endpoints = plugins.load_endpoint_plugins(("generate",))
    assert sum(p.name == "ascend_router_load" for p in endpoints) == 1
    assert (other_endpoint in endpoints) == (allowlist is not None and "other_plugin" in allowlist)
    assert not any(p.name == "ascend_router_load" for p in plugins.load_endpoint_plugins(("embed",)))
    assert not any(p.name == "ascend_router_load" for p in plugins.load_endpoint_plugins(None))
    assert os.environ.get("VLLM_PLUGINS") == allowlist

    # Exercise the native API attach/state lifecycle, without manual plugin setup.
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kwargs: [])
    app = FastAPI()
    attach_endpoint_plugins(app, ("generate",))
    core = NS(call_utility_async=AsyncMock(return_value=collect_load(make_engine())))

    async def check():
        await init_endpoint_plugins_state(NS(engine_core=core), app.state, None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            for path in ("/v1/loads", "/v1/load"):
                response = await client.get(path)
                assert response.status_code == 200
                assert response.json()["loads"][0]["router_load"]["waiting_reqs"] == 0

    asyncio.run(check())
