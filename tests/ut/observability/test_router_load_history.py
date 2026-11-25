# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unknown external hits use a bounded history, never overwrite real progress."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import httpx
from fastapi import FastAPI
from vllm.v1.request import RequestStatus

from tests.ut.observability.test_router_load import make_engine, req
from vllm_ascend.observability.router_load import collect_load
from vllm_ascend.observability.router_load_api import RouterLoadEndpoint
from vllm_ascend.observability.router_load_history import HitWindow, LoadHistory, history_for, install_history_hooks


def sample(requests=1, queries=128, hits=96, **kwargs):
    return NS(requests=requests, queries=queries, hits=hits, reset=False, **kwargs)


def test_weight_by_tokens_count_misses_expire_and_bound_memory():
    window = HitWindow()
    window.add(1, 100, 100, 1.2)
    window.add(1, 900, 0, 2.2)
    assert window.snapshot(2.2) == dict(requests=2, query_tokens=1000, hit_tokens=100, hit_rate=0.1)
    assert window.snapshot(301.1)["hit_rate"] == 0
    assert window.snapshot(302.1)["hit_rate"] is None
    for second in range(1000):
        for _ in range(20):
            window.add(1, 100, 50, second + 1000.2)
    assert len(window.buckets) <= 300
    assert window.snapshot(1999.3)["hit_rate"] == 0.5


def test_history_reset_does_not_reimport_buffered_samples():
    history = LoadHistory()
    pending = sample()
    history.observe(sample(queries=100, hits=50), 0)
    history.clear(pending)
    pending.requests += 1
    pending.queries += 64
    pending.hits += 16
    history.observe(pending, 1)
    assert history.snapshot(1)["normal"] == dict(requests=1, query_tokens=64, hit_tokens=16, hit_rate=0.25)
    reset = sample(queries=64, hits=0)
    reset.reset = True
    history.observe(reset, 2)
    assert history.snapshot(2)["normal"]["hit_tokens"] == 0
    assert history.reset_count == 2


def test_repeated_poll_does_not_resample_and_dp_histories_are_independent(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 100)
    a, b = make_engine("kv_producer", 0), make_engine("kv_producer", 1)
    a.scheduler.waiting = [req("a", 128)]
    b.scheduler.waiting = [req("b", 128)]
    history_for(a.scheduler).observe(sample(), 99)
    a.scheduler.connector = NS(get_num_new_matched_tokens=Mock(side_effect=AssertionError("extra lookup")))
    for _ in range(10):
        row = collect_load(a)
        assert row["router_load"]["p_pending_uncached_tok"] == 32
        assert row["router_load"]["prefill"]["external_history"]["normal"]["requests"] == 1
        assert row["requests"][0]["external_lookup_unknown"]
        assert row["requests"][0]["estimate_source"] == "external_hit_rate_5m"
    other = collect_load(b)
    assert other["router_load"]["p_pending_uncached_tok"] == 128
    assert row["router_load"]["instance_id"] != other["router_load"]["instance_id"]
    a.scheduler.connector.get_num_new_matched_tokens.assert_not_called()


def test_estimate_applies_only_after_l1_and_never_to_known_progress(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 100)
    engine = make_engine("kv_producer")
    manager = engine.scheduler.kv_cache_manager
    manager.coordinator.find_longest_cache_hit = Mock(return_value=(None, 32))
    request = req("test", 128)
    engine.scheduler.waiting = [request]
    history_for(engine.scheduler).observe(sample(queries=100, hits=50), 99)
    row = collect_load(engine)
    assert row["router_load"]["p_pending_uncached_tok"] == 48  # (128 - L1 32) * (1 - .5)
    assert row["router_load"]["prefill"]["pending_tokens_upper_bound"] == 96
    assert row["router_load"]["prefill"]["estimated_external_hit_tokens"] == 48
    assert request.num_computed_tokens == 0
    request.num_computed_tokens = 96
    request.status = RequestStatus.RUNNING
    engine.scheduler.waiting = []
    engine.scheduler.running = [request]
    manager.coordinator.find_longest_cache_hit = Mock(side_effect=AssertionError("known request re-looked up"))
    row = collect_load(engine)
    assert row["router_load"]["p_pending_uncached_tok"] == 32
    assert row["router_load"]["prefill"]["estimated_external_hit_tokens"] == 0
    request.num_computed_tokens = 112
    assert collect_load(engine)["router_load"]["p_pending_uncached_tok"] == 16


def test_restore_load_failure_recompute_and_completion(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 100)
    engine = make_engine("kv_producer")
    request = req("restore", 128)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_computed_tokens = 96
    request.prefill_stats.set(128, 32, 64)
    engine.scheduler.skipped_waiting = [request]
    history_for(engine.scheduler).observe(sample(queries=100, hits=99), 99)
    row = collect_load(engine)
    assert row["router_load"]["p_bootstrap_uncached_tok"] == 32
    assert row["router_load"]["prefill"]["external_restore_tokens_upper_bound"] == 64
    assert row["requests"][0]["admission_cache_hits"]["external_tokens"] == 64
    # The scheduler invalidates unavailable blocks and falls back to recompute.
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 32
    engine.scheduler.skipped_waiting = []
    engine.scheduler.running = [request]
    assert collect_load(engine)["router_load"]["p_pending_uncached_tok"] == 96
    request.num_computed_tokens = 0
    assert collect_load(engine)["router_load"]["p_pending_uncached_tok"] == 128
    request.num_computed_tokens = 128
    assert collect_load(engine)["router_load"]["p_pending_uncached_tok"] == 0
    engine.scheduler.running = []
    assert collect_load(engine)["requests"] == []


def test_history_expiry_reset_and_preempted_population(monkeypatch):
    clock = [100]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    engine = make_engine("kv_producer")
    request = req("test", 128)
    engine.scheduler.waiting = [request]
    history = history_for(engine.scheduler)
    history.observe(sample(preempted_requests=1, preempted_queries=128, preempted_hits=32), 100)
    assert collect_load(engine)["router_load"]["p_pending_uncached_tok"] == 32
    request.status = RequestStatus.PREEMPTED
    request.num_preemptions = 1
    assert collect_load(engine)["router_load"]["p_pending_uncached_tok"] == 96
    clock[0] = 400
    assert collect_load(engine)["router_load"]["p_pending_uncached_tok"] == 128
    history.observe(sample(preempted_requests=1, preempted_queries=128, preempted_hits=64), 400)
    assert collect_load(engine)["router_load"]["p_pending_uncached_tok"] == 64
    history.clear()
    assert collect_load(engine)["router_load"]["p_pending_uncached_tok"] == 128


def test_full_hit_estimate_retains_compute_and_decode_never_uses_estimator(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 100)
    for role in ("kv_producer", "kv_consumer", None):
        engine = make_engine(role)
        engine.scheduler.waiting = [req("test", 128)]
        history_for(engine.scheduler).observe(sample(queries=128, hits=128), 99)
        row = collect_load(engine)
        expected = {"kv_producer": 1, "kv_consumer": 0, None: 128}[role]
        assert row["router_load"]["p_pending_uncached_tok"] == expected
        if role == "kv_consumer":
            assert row["router_load"]["d_pending_raw_tok"] == 128
            assert row["requests"] == []


def test_hooks_observe_each_native_stats_batch_once_preserve_return_and_reset(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    native_batches = [sample(), sample(queries=128, hits=0)]

    def native_stats(self):
        return NS(connector_prefix_cache_stats=native_batches.pop(0))

    monkeypatch.setattr(Scheduler, "make_stats", native_stats)
    monkeypatch.setattr(Scheduler, "reset_prefix_cache", lambda self, result=True: result)
    install_history_hooks()
    wrapped = Scheduler.make_stats
    install_history_hooks()
    assert Scheduler.make_stats is wrapped
    scheduler = NS(connector_prefix_cache_stats=None)
    a = Scheduler.make_stats(scheduler)
    b = Scheduler.make_stats(scheduler)
    assert a.connector_prefix_cache_stats.hits == 96 and b.connector_prefix_cache_stats.hits == 0
    history = history_for(scheduler)
    assert history.normal.requests == 2 and history.normal.hits == 96 and history.normal.queries == 256
    assert Scheduler.reset_prefix_cache(scheduler, result=False) is False
    assert history.normal.queries == 256
    assert Scheduler.reset_prefix_cache(scheduler) is True
    assert history.normal.queries == 0


def test_optional_http_details_and_legacy_aggregate_use_same_snapshot(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 100)
    engine = make_engine("kv_producer")
    engine.scheduler.waiting = [req("detail", 128)]
    history_for(engine.scheduler).observe(sample(), 99)

    async def run():
        app = FastAPI()
        endpoint = RouterLoadEndpoint()
        endpoint.attach_router(app)
        core = NS(call_utility_async=AsyncMock(side_effect=lambda _: collect_load(engine)))
        await endpoint.init_state(NS(engine_core=core), app.state, None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            default = (await client.get("/v1/loads")).json()["loads"][0]
            assert "requests" not in default
            row = (await client.get("/v1/load?include=core,router_load,requests")).json()["loads"][0]
            assert row["num_waiting_uncached_tokens"] == row["requests"][0]["remaining_tokens"] == 32
            assert row["requests"][0]["request_id"] == "detail"
            assert row["router_load"]["prefill"]["unknown_external_lookup_tokens"] == 128

    asyncio.run(run())


def test_mtp_mandatory_suffix_is_not_discounted_by_history(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 100)
    for tokens, expected in [(128, 16), (129, 1), (143, 15)]:
        engine = make_engine("kv_producer")
        engine.scheduler.kv_cache_manager.use_eagle = True
        engine.scheduler.waiting = [req("mtp", tokens)]
        history_for(engine.scheduler).observe(sample(queries=100, hits=100), 99)
        row = collect_load(engine)
        assert row["router_load"]["p_pending_uncached_tok"] == expected
        assert row["requests"][0]["minimum_recompute_tokens"] == expected


def test_invalid_statistics_do_not_poison_history_or_raise():
    history = LoadHistory()
    history.observe(sample(queries=64, hits=65), 100)
    assert history.normal.queries == 0
    assert history.snapshot(100)["invalid_batches"] == 1
    history.observe(sample(queries=64, hits=32), 101)
    assert history.snapshot(101)["normal"]["hit_rate"] == 0.5
