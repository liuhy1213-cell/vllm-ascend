# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Local CPU integration: actual Router binary + plugin + native KV state.

No model inference, NPU, production endpoint or hand-written load response.
Usage: python .../router_load_smoke.py ROUTER_REPO ROUTER_BINARY OUTPUT_DIR
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import uvicorn
from fastapi import FastAPI
from vllm.plugins.endpoint_plugins.interface import attach_endpoint_plugins, init_endpoint_plugins_state

from tests.ut.observability.test_router_load import make_engine, req
from vllm_ascend import register
from vllm_ascend.observability.router_load import collect_load


def listener():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    return sock


async def main():
    """Serve two native KV states and verify Router polling before/after release."""
    router_repo, router_bin, output = sys.argv[1:]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    engine0, engine1 = make_engine(rank=0), make_engine(rank=1)
    waiting = req("waiting", 65)
    running = req("running", 32)
    engine0.scheduler.waiting.append(waiting)
    engine1.scheduler.kv_cache_manager.allocate_slots(running, 32)
    running.num_computed_tokens = 32
    engine1.scheduler.running.append(running)

    async def query(method, *, engine):
        return collect_load(engine0 if engine == b"zero" else engine1)

    app = FastAPI()
    register()
    attach_endpoint_plugins(app, ("generate",))
    core = NS(core_engines=[b"zero", b"one"], _call_utility_async=AsyncMock(side_effect=query))
    await init_endpoint_plugins_state(NS(engine_core=core), app.state, None)
    app.add_api_route("/health", lambda: {})
    app.add_api_route("/v1/models", lambda: {"data": [{"id": "load-test"}]})
    sock = listener()
    engine_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    serve = asyncio.create_task(server.serve(sockets=[sock]))
    control = out / "tokenizer-mode"
    control.write_text("ok")
    (out / "worker.json").write_text(
        json.dumps(
            {
                "python_executable": sys.executable,
                "worker_path": str(Path(router_repo) / "tests/vllm_adapter/ha_fault_worker.py"),
                "renderer_config_path": "unused",
                "workers": 1,
            }
        )
    )
    (out / "cache.json").write_text(json.dumps({"alignment_tokens": 16, "groups": [{"id": 0, "block_size": 16}]}))
    router_socket, metrics_socket = listener(), listener()
    router_port, metrics_port = router_socket.getsockname()[1], metrics_socket.getsockname()[1]
    router_socket.close()
    metrics_socket.close()
    command = [
        router_bin,
        "--host",
        "127.0.0.1",
        "--port",
        str(router_port),
        "--prometheus-port",
        str(metrics_port),
        "--engine-type",
        "vllm",
        "--vllm-tokenizer-config",
        str(out / "worker.json"),
        "--vllm-cache-config",
        str(out / "cache.json"),
        "--disable-blockhash",
        "--forward-token-ids-to-engine",
        "false",
        "--policy",
        "cache_aware",
        "--worker-urls",
        engine_url,
    ]
    env = dict(os.environ, HA_TOKENIZER_CONTROL=str(control))
    env.pop("VLLM_ROUTER_KV_EVENT_SOURCES", None)
    with (out / "router.log").open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            async with httpx.AsyncClient(timeout=2) as client:

                async def poll(predicate):
                    for _ in range(100):
                        try:
                            response = await client.get(f"http://127.0.0.1:{router_port}/debug/engine_load_snapshot")
                            data = response.json()
                            if predicate(data.get("loads", [])):
                                return data
                        except (httpx.HTTPError, ValueError):
                            pass
                        assert proc.poll() is None, (out / "router.log").read_text()[-3000:]
                        await asyncio.sleep(0.2)
                    raise AssertionError("Router did not consume the expected snapshot")

                before = await poll(
                    lambda rows: len(rows) == 2
                    and any(r["dp_rank"] == 0 and r["p_pending_uncached_tok"] == 65 for r in rows)
                    and any(r["dp_rank"] == 1 and r["d_used_kv_tok"] == 32 for r in rows)
                )
                engine0.scheduler.waiting.clear()
                engine1.scheduler.kv_cache_manager.free(running)
                engine1.scheduler.running.clear()
                after = await poll(
                    lambda rows: len(rows) == 2
                    and all(r["p_pending_uncached_tok"] == r["d_used_kv_tok"] == 0 for r in rows)
                )
                assert all(
                    b["last_seq"] > a["last_seq"]
                    for a, b in zip(
                        sorted(before["loads"], key=lambda r: r["dp_rank"]),
                        sorted(after["loads"], key=lambda r: r["dp_rank"]),
                    )
                )
                (out / "results.json").write_text(json.dumps({"before": before, "after": after}, indent=2))
                print("PASS: Router consumed both DP ranks and the released/idle snapshot")
        finally:
            proc.terminate()
            await asyncio.to_thread(proc.wait, 10)
            server.should_exit = True
            await serve
            sock.close()


if __name__ == "__main__":
    asyncio.run(main())
