# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Default Ascend load endpoint. Router polls; Engine does not push metrics."""

import asyncio

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from vllm.logger import logger

LOAD_TIMEOUT_SECONDS = 0.4
LOAD_SECTIONS = frozenset({"core", "disagg", "router_load"})


async def read_engine_loads(client):
    core = client.engine_core
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
    """At most one pending utility batch per API process, including on timeout.

    Shield from HTTP cancellation: vLLM keeps pending utility futures until the
    core replies. Repeated polls must not accumulate RPCs behind a hung core.
    Completed results are never served as a new snapshot to a later poll.
    """

    def __init__(self, client):
        self.client = client
        self.pending = None

    async def read(self):
        if self.pending is None or self.pending.done():
            if self.pending is not None and not self.pending.cancelled():
                self.pending.exception()  # consume a late error after an HTTP timeout
            self.pending = asyncio.create_task(read_engine_loads(self.client))
        return await asyncio.wait_for(asyncio.shield(self.pending), timeout=LOAD_TIMEOUT_SECONDS)


class RouterLoadEndpoint:
    name = "ascend_router_load"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI):
        router = APIRouter()
        router.add_api_route("/v1/loads", self.loads, methods=["GET"])
        router.add_api_route("/v1/load", self.loads, methods=["GET"], include_in_schema=False)
        app.include_router(router)

    async def init_state(self, engine_client, state, args):
        state.ascend_router_load_reader = LoadReader(engine_client) if engine_client is not None else None

    async def loads(self, request: Request, include: str = "core,disagg,router_load"):
        sections = {part.strip() for part in include.split(",") if part.strip()}
        if not sections or sections - LOAD_SECTIONS:
            return JSONResponse({"error": "include must contain core, disagg and/or router_load"}, status_code=400)
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
        return JSONResponse({"loads": loads}, headers={"Cache-Control": "no-store"})
