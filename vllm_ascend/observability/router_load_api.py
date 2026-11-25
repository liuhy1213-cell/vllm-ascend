# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Default Ascend load endpoint. Router polls; Engine does not push metrics."""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from vllm.logger import logger

LOAD_TIMEOUT_SECONDS = 0.4
LOAD_REFRESH_SECONDS = 0.5
LOAD_MAX_AGE_SECONDS = 2.0
LOAD_SECTIONS = frozenset({"core", "disagg", "router_load", "requests"})


async def read_engine_loads(client):
    core = client.engine_core
    # Monitoring must not accumulate utility futures behind maintenance or a
    # failed native send. _call_utility_async registers its future before send,
    # and a transport exception can leave that future awaiting a late reply.
    # Defer rather than deleting/replacing futures owned by vLLM or other APIs.
    if getattr(core, "utility_results", None):
        raise RuntimeError("Engine utility channel is busy")
    if hasattr(core, "ensure_alive"):
        core.ensure_alive()
    # DP's public call_utility_async broadcasts but returns only rank 0.
    # Use the existing per-engine utility transport to retain EVERY rank.
    engines = getattr(core, "core_engines", None)
    if engines is not None:
        if not engines:
            raise RuntimeError("No engine cores available")
        rows = await asyncio.gather(
            *(core._call_utility_async("get_router_load_snapshot", engine=engine) for engine in tuple(engines)),
            return_exceptions=True,
        )
        for row in rows:
            if isinstance(row, BaseException):
                raise RuntimeError("A DP rank did not return a load snapshot") from row
    else:
        rows = [await core.call_utility_async("get_router_load_snapshot")]
    ranks = [row["dp_rank"] for row in rows]
    if len(set(ranks)) != len(ranks):
        raise RuntimeError("Duplicate DP ranks in load snapshot")
    return sorted(rows, key=lambda row: row["dp_rank"])


class LoadReader:
    """Serve bounded-age snapshots while one shielded utility query refreshes.

    Like SGLang's published snapshot reader, HTTP need not wait for a model
    step when a valid snapshot exists. Here the native utility transport is
    retained: refresh is demand-driven, with no scheduler thread or timer.
    Age starts at RPC dispatch, conservatively including queue/transport time.
    A late result is kept, but never assigned a new sequence or a younger age.
    """

    def __init__(self, client):
        self.client = client
        self.pending = None
        self.cached = None
        self.cached_started = 0.0
        self.last_query_duration = None
        self.closed = False

    async def _refresh(self, started):
        try:
            rows = await read_engine_loads(self.client)
        except Exception:
            # A failed/partial DP refresh must not masquerade as a healthy
            # aggregate. Next read retries; no old healthy response is retained.
            self.cached = None
            raise
        finally:
            self.last_query_duration = time.monotonic() - started
        self.cached = rows
        self.cached_started = started
        return rows

    @staticmethod
    def _observe_error(task):
        # Even if every HTTP waiter timed out/cancelled, retrieve a late error.
        # Awaiting the task still propagates it to any remaining waiter.
        if not task.cancelled():
            task.exception()

    def snapshot_info(self):
        return {
            "age_upper_bound_ms": max(0, (time.monotonic() - self.cached_started) * 1000),
            "max_age_ms": LOAD_MAX_AGE_SECONDS * 1000,
            "refresh_inflight": self.pending is not None and not self.pending.done(),
            "last_query_duration_ms": (
                self.last_query_duration * 1000 if self.last_query_duration is not None else None
            ),
        }

    async def close(self):
        """Release API-owned state at shutdown, never during a poll timeout."""
        self.closed = True
        task, self.pending = self.pending, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.cached = None

    async def read(self):
        if self.closed:
            raise RuntimeError("Load reader is closed")
        now = time.monotonic()
        if self.pending is not None and self.pending.done():
            self.pending = None  # _refresh already retained the result.
        age = now - self.cached_started
        if self.pending is None and (self.cached is None or age >= LOAD_REFRESH_SECONDS or age > LOAD_MAX_AGE_SECONDS):
            self.pending = asyncio.create_task(self._refresh(now))
            self.pending.add_done_callback(self._observe_error)
        if self.cached is not None and age <= LOAD_MAX_AGE_SECONDS:
            return self.cached
        # Cold start / expired snapshot: wait briefly, keeping a single RPC in
        # flight after HTTP timeout or cancellation. Other readers join it.
        await asyncio.wait_for(asyncio.shield(self.pending), timeout=LOAD_TIMEOUT_SECONDS)
        if self.cached is None or time.monotonic() - self.cached_started > LOAD_MAX_AGE_SECONDS:
            raise TimeoutError("Load snapshot exceeded maximum age")
        return self.cached


class RouterLoadEndpoint:
    name = "ascend_router_load"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI):
        @asynccontextmanager
        async def reader_lifespan(app):
            # Router lifespans compose with vLLM's custom application lifespan;
            # deprecated on_shutdown handlers may not run with that lifespan.
            try:
                yield
            finally:
                reader = getattr(app.state, "ascend_router_load_reader", None)
                if reader is not None:
                    await reader.close()

        router = APIRouter(lifespan=reader_lifespan)
        router.add_api_route("/v1/loads", self.loads, methods=["GET"])
        router.add_api_route("/v1/load", self.loads, methods=["GET"], include_in_schema=False)
        app.include_router(router)

    async def init_state(self, engine_client, state, args):
        state.ascend_router_load_reader = LoadReader(engine_client) if engine_client is not None else None

    async def loads(self, request: Request, include: str = "core,disagg,router_load"):
        sections = {part.strip() for part in include.split(",") if part.strip()}
        if not sections or sections - LOAD_SECTIONS:
            return JSONResponse(
                {"error": "include must contain core, disagg, router_load and/or requests"}, status_code=400
            )
        reader = getattr(request.app.state, "ascend_router_load_reader", None)
        if reader is None:
            return JSONResponse({"error": "Router load snapshot is unavailable"}, status_code=503)
        try:
            rows = await reader.read()
        except Exception as exc:
            logger.debug("Router load snapshot failed: %s", exc)
            return JSONResponse({"error": "Router load snapshot is unavailable"}, status_code=503)
        loads = []
        for row in rows:
            load = row["router_load"]
            entry = {"dp_rank": row["dp_rank"]}
            if "router_load" in sections:
                entry["router_load"] = load
            if "requests" in sections:
                entry["requests"] = row["requests"]
            if "core" in sections:
                entry.update(
                    num_running_reqs=load["running_reqs"],
                    num_waiting_reqs=load["waiting_reqs"],
                    num_waiting_uncached_tokens=load["p_pending_uncached_tok"] + load["p_bootstrap_uncached_tok"],
                    num_used_tokens=load["kv_used_tok"],
                    num_total_tokens=load["kv_used_tok"] + load["d_pending_raw_tok"],
                    token_usage=load["token_usage"],
                )
            if "disagg" in sections:
                entry["disaggregation"] = {
                    "mode": ("prefill", "decode", "null")[load["role"]],
                    "prefill_bootstrap_queue_reqs": load["bootstrap_reqs"],
                    "decode_prealloc_queue_reqs": load["prealloc_reqs"],
                    "decode_transfer_queue_reqs": load["transfer_reqs"] if load["role"] != 0 else 0,
                    "decode_retracted_queue_reqs": load["retracted_reqs"] if load["role"] != 0 else 0,
                }
            loads.append(entry)
        return JSONResponse({"loads": loads, "snapshot": reader.snapshot_info()}, headers={"Cache-Control": "no-store"})
