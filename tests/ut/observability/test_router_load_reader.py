# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Load-reader lifecycle, delayed RPC, bounded freshness and allocation tests."""

import asyncio
import gc
import tracemalloc
import weakref
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from vllm_ascend.observability import router_load_api as api


def controlled(monkeypatch):
    clock = [100.0]
    pending = []

    async def query(client):
        future = asyncio.get_running_loop().create_future()
        pending.append(future)
        return await future

    monkeypatch.setattr(api, "time", NS(monotonic=lambda: clock[0]))
    monkeypatch.setattr(api, "read_engine_loads", query)
    monkeypatch.setattr(api, "LOAD_TIMEOUT_SECONDS", 0.005)
    return clock, pending, api.LoadReader(None)


def row(seq=1):
    return [{"dp_rank": 0, "router_load": {"seq": seq}}]


def test_late_success_survives_timeout_and_keeps_sequence_and_age(monkeypatch):
    clock, pending, reader = controlled(monkeypatch)

    async def run():
        with pytest.raises(TimeoutError):
            await reader.read()
        assert len(pending) == 1 and not pending[0].cancelled()
        clock[0] += 0.6
        pending[0].set_result(row(123))
        await reader.pending
        assert (await reader.read())[0]["router_load"]["seq"] == 123
        info = reader.snapshot_info()
        assert info["age_upper_bound_ms"] == pytest.approx(600)
        assert info["last_query_duration_ms"] == pytest.approx(600)
        assert info["refresh_inflight"]
        await asyncio.sleep(0)
        assert len(pending) == 2
        # Refresh is still pending, but cached HTTP reads are immediate.
        for _ in range(100):
            assert (await reader.read())[0]["router_load"]["seq"] == 123
        assert len(pending) == 2
        assert reader.snapshot_info()["age_upper_bound_ms"] == pytest.approx(600)
        await reader.close()

    asyncio.run(run())


def test_concurrent_timeouts_and_disconnects_share_single_query(monkeypatch):
    clock, pending, reader = controlled(monkeypatch)

    async def run():
        tasks = [asyncio.create_task(reader.read()) for _ in range(100)]
        await asyncio.sleep(0)
        for t in tasks[:50]:
            t.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(x, asyncio.CancelledError) for x in outcomes[:50])
        assert all(isinstance(x, TimeoutError) for x in outcomes[50:])
        assert len(pending) == 1 and not pending[0].cancelled()
        for _ in range(20):
            with pytest.raises(TimeoutError):
                await reader.read()
        assert len(pending) == 1
        pending[0].set_result(row(77))
        await reader.pending
        assert (await reader.read())[0]["router_load"]["seq"] == 77
        assert len(pending) == 1
        await reader.close()

    asyncio.run(run())


def test_expired_snapshot_cannot_be_kept_alive_by_http_reads(monkeypatch):
    clock, pending, reader = controlled(monkeypatch)

    async def run():
        t = asyncio.create_task(reader.read())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        pending[0].set_result(row(1))
        await t
        clock[0] += 0.5
        assert await reader.read() == row(1)  # Starts one refresh.
        await asyncio.sleep(0)
        clock[0] += 1.6
        for _ in range(3):
            with pytest.raises(TimeoutError):
                await reader.read()
        assert len(pending) == 2
        # Freshness is measured from dispatch, not completion/HTTP receipt.
        clock[0] += 1
        pending[1].set_result(row(2))
        await reader.pending
        with pytest.raises(TimeoutError):
            await reader.read()
        assert len(pending) == 3
        pending[2].set_result(row(3))
        await reader.pending
        assert await reader.read() == row(3)
        await reader.close()

    asyncio.run(run())


def test_late_error_invalidates_cache_and_recovers_without_unhandled_task(monkeypatch):
    clock, pending, reader = controlled(monkeypatch)

    async def run():
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: errors.append(context))
        with pytest.raises(TimeoutError):
            await reader.read()
        pending[0].set_result(row(1))
        await reader.pending
        clock[0] += 0.6
        assert await reader.read() == row(1)
        await asyncio.sleep(0)
        pending[1].set_exception(RuntimeError("rank unavailable"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert reader.cached is None
        with pytest.raises(TimeoutError):
            await reader.read()
        pending[2].set_result(row(3))
        await reader.pending
        assert await reader.read() == row(3)
        await reader.close()
        gc.collect()
        assert errors == []

    asyncio.run(run())


def test_shutdown_clears_owned_tasks_and_cache_and_prevents_restart(monkeypatch):
    clock, pending, reader = controlled(monkeypatch)

    async def run():
        baseline = asyncio.all_tasks()
        with pytest.raises(TimeoutError):
            await reader.read()
        task = reader.pending
        await reader.close()
        await reader.close()
        assert task.cancelled() and reader.pending is None and reader.cached is None
        assert asyncio.all_tasks() == baseline
        with pytest.raises(RuntimeError, match="closed"):
            await reader.read()

    asyncio.run(run())


def test_api_cached_responses_preserve_sequence_and_expose_age(monkeypatch):
    # Real HTTP serialization and native DP utility adapter, controlled RPCs.
    async def run():
        lifecycle = []

        @asynccontextmanager
        async def native_lifespan(app):
            lifecycle.append("enter")
            yield
            assert app.state.ascend_router_load_reader.closed
            lifecycle.append("exit")

        app = FastAPI(lifespan=native_lifespan)
        endpoint = api.RouterLoadEndpoint()
        endpoint.attach_router(app)
        core = NS(call_utility_async=AsyncMock(return_value=row(42)[0]))
        await endpoint.init_state(NS(engine_core=core), app.state, None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(30):
                r = await client.get("/v1/load?include=router_load")
                assert r.status_code == 200
                assert r.json()["loads"][0]["router_load"]["seq"] == 42
                assert r.json()["snapshot"]["age_upper_bound_ms"] <= 2000
            assert core.call_utility_async.await_count == 1
            # Exercise the composed lifespan, including vLLM-style lifecycle.
            async with app.router.lifespan_context(app):
                assert lifecycle == ["enter"]
            assert lifecycle == ["enter", "exit"]
            assert (await client.get("/v1/loads")).status_code == 503

    asyncio.run(run())


def test_partial_dp_failure_discards_entire_snapshot_then_recovers(monkeypatch):
    monkeypatch.setattr(api, "LOAD_REFRESH_SECONDS", 0)

    async def run():
        fail = False

        async def call(method, *, engine):
            if fail and engine == b"b":
                raise RuntimeError("rank failed")
            return {"dp_rank": 0 if engine == b"a" else 1, "router_load": {"seq": 1}}

        core = NS(core_engines=[b"a", b"b"], _call_utility_async=call)
        reader = api.LoadReader(NS(engine_core=core))
        assert len(await reader.read()) == 2
        fail = True
        assert len(await reader.read()) == 2  # Last valid aggregate; starts refresh.
        with pytest.raises(RuntimeError, match="DP rank"):
            await reader.pending
        assert reader.cached is None
        with pytest.raises(RuntimeError, match="DP rank"):
            await reader.read()
        fail = False
        assert len(await reader.read()) == 2
        await reader.close()

    asyncio.run(run())


def test_repeated_refresh_does_not_retain_snapshots_tasks_or_memory(monkeypatch):
    clock = [100.0]
    refs = []

    class Payload:
        def __init__(self):
            self.data = bytearray(65536)

    async def query(client):
        payload = Payload()
        refs.append(weakref.ref(payload))
        return [{"dp_rank": 0, "payload": payload, "router_load": {"seq": len(refs)}}]

    monkeypatch.setattr(api, "time", NS(monotonic=lambda: clock[0]))
    monkeypatch.setattr(api, "read_engine_loads", query)

    async def run():
        baseline = asyncio.all_tasks()
        reader = api.LoadReader(None)
        tracemalloc.start()
        try:

            async def cycles(n):
                for _ in range(n):
                    clock[0] += 0.6
                    await reader.read()
                    if reader.pending is not None:
                        await reader.pending
                await asyncio.sleep(0)
                gc.collect()
                # Only the current snapshot may own a large payload.
                assert sum(ref() is not None for ref in refs) == 1
                refs.clear()  # The test's weakref list is not product memory.
                gc.collect()
                return tracemalloc.get_traced_memory()[0]

            await cycles(200)
            first = await cycles(2000)
            second = await cycles(2000)
            assert second - first < 128 * 1024, (first, second)
            assert asyncio.all_tasks() == baseline
            await reader.close()
            gc.collect()
            assert reader.cached is None and reader.pending is None
            assert asyncio.all_tasks() == baseline
        finally:
            tracemalloc.stop()
            await reader.close()

    asyncio.run(run())


def test_native_utility_futures_bounded_across_timeouts_errors_and_recovery(monkeypatch):
    from vllm.v1.engine.core_client import AsyncMPClient, _process_utility_output

    monkeypatch.setattr(api, "LOAD_TIMEOUT_SECONDS", 0.002)

    async def run():
        messages = []
        core = NS(
            utility_results={},
            core_engines=[b"p0", b"p1"],
            client_index=0,
            encoder=NS(encode=lambda message: [message]),
            _ensure_output_queue_task=lambda: None,
            ensure_alive=lambda: None,
        )

        async def send(message, engine):
            messages.append((message[1][1], engine))

        core._send_input_message = send
        core._call_utility_async = lambda method, **kw: AsyncMPClient._call_utility_async(core, method, **kw)
        reader = api.LoadReader(NS(engine_core=core))
        for _ in range(30):
            with pytest.raises(TimeoutError):
                await reader.read()
        assert len(core.utility_results) == len(messages) == 2
        for call_id, engine in messages:
            value = {"dp_rank": 0 if engine == b"p0" else 1, "router_load": {"seq": 1}}
            _process_utility_output(
                NS(call_id=call_id, failure_message=None, result=NS(result=value)), core.utility_results
            )
        await reader.pending
        assert len(await reader.read()) == 2
        assert core.utility_results == {}
        await reader.close()

        # Native transport fails after registering a future. Do not retry into
        # an unbounded orphan-future dictionary or delete someone else's RPC.
        core.core_engines = [b"p0"]
        messages.clear()

        async def failed_send(message, engine):
            messages.append((message[1][1], engine))
            raise RuntimeError("transport failed")

        core._send_input_message = failed_send
        reader = api.LoadReader(NS(engine_core=core))
        for _ in range(30):
            with pytest.raises(RuntimeError):
                await reader.read()
        assert len(core.utility_results) == len(messages) == 1
        call_id, _ = messages[0]
        _process_utility_output(
            NS(call_id=call_id, failure_message=None, result=NS(result=row(9)[0])), core.utility_results
        )
        assert core.utility_results == {}
        core._send_input_message = send
        messages.clear()
        waiter = asyncio.create_task(reader.read())
        for _ in range(5):
            await asyncio.sleep(0)
        call_id, _ = messages[0]
        _process_utility_output(
            NS(call_id=call_id, failure_message=None, result=NS(result=row(10)[0])), core.utility_results
        )
        assert await waiter == row(10)
        assert core.utility_results == {}
        await reader.close()

    asyncio.run(run())


def test_failed_refresh_tracebacks_do_not_retain_old_payloads(monkeypatch):
    clock = [100.0]
    refs = []
    count = 0

    class Payload:
        def __init__(self):
            self.data = bytearray(65536)

    async def query(client):
        nonlocal count
        count += 1
        payload = Payload()
        refs.append(weakref.ref(payload))
        if count % 2:
            raise RuntimeError(payload)
        return [{"payload": payload}]

    monkeypatch.setattr(api, "time", NS(monotonic=lambda: clock[0]))
    monkeypatch.setattr(api, "read_engine_loads", query)

    async def run():
        reader = api.LoadReader(None)
        baseline = asyncio.all_tasks()
        for _ in range(2000):
            clock[0] += 3
            with suppress(RuntimeError):
                await reader.read()
        await asyncio.sleep(0)
        gc.collect()
        assert sum(ref() is not None for ref in refs) <= 1
        assert asyncio.all_tasks() == baseline
        await reader.close()
        gc.collect()
        assert all(ref() is None for ref in refs)

    asyncio.run(run())
