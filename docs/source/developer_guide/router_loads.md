# Router load snapshots

The Ascend plugin adds `GET /v1/loads?include=core,disagg,router_load` for
LLM Router's polling contract. `/v1/load` is an alias. The Router normally
polls every second; the Engine does not push or run a new per-token reporting
loop. Native `/load` and `/metrics` are unchanged.

## Enable

Install this version of vllm-ascend in both the API and EngineCore environments.
The implementation requires vLLM's endpoint-plugin API and V1 utility API,
validated with vLLM 0.30.0. Older releases without endpoint plugins need a
separate adapter.

Add `ascend_router_load` to the existing `VLLM_PLUGINS` allowlist. Keep all
plugins required by the model, connector and platform. Example for this repo:

```bash
export VLLM_PLUGINS=ascend_kv_connector,ascend_model_loader,ascend_service_profiling,ascend_model,ascend_router_load
# Start vllm serve with the existing model/TP/DP/connector configuration.
curl -fsS 'http://127.0.0.1:8000/v1/loads?include=core,disagg,router_load'
```

Use the existing Engine API authentication headers if required. No new HTTP
listener is opened. The plugin is registered in two groups: `general_plugins`
installs the EngineCore method, and `endpoint_plugins` attaches HTTP routes.
The endpoint is opt-in through vLLM's existing plugin mechanism. Missing
Engine methods or failed ranks return 503, never a fabricated idle snapshot.
Invalid `include` returns 400. Responses use `Cache-Control: no-store`.

## Accounting

Each `loads[]` row is one **DP rank**, not one TP/PP worker. Collection runs in
the EngineCore utility handler, serialized with scheduler work. It makes no
device calls or HCCL/TP/PP collectives. Internal DP returns all connected cores;
external DP returns its local core, so the Router must poll every API instance.

| `router_load` field | Meaning |
| --- | --- |
| `role` | `kv_producer=0`, `kv_consumer=1`, no connector or `kv_both=2`. `kv_both` is not assumed to be exclusively P or D. |
| `page_size` | LCM of native group manager block sizes, in their logical token units. |
| `seq` | Strictly increasing per EngineCore, seeded from microsecond wall time; increases after successful idle snapshots too. |
| `p_pending_uncached_tok` | Remaining prefill work in waiting/skipped-waiting/running-chunked requests, minus native local prefix hits or already-computed tokens. Preempted requests include generated tokens requiring recomputation. Zero on D. |
| `p_bootstrap_uncached_tok` | Remaining work in scheduler `WAITING_FOR_REMOTE_KVS`, subtracting known remotely computed tokens. Zero on D. No artificial SGLang-style producer bootstrap queue is introduced. |
| `d_used_kv_tok` | Largest group's active footprint in tokens. Shared pages are deduplicated; null SWA holes and free-but-cached pages are excluded. Zero on P. |
| `d_pending_raw_tok` | Logical tokens not covered by allocated slots, including waiting/preempted/chunked requests. Fully preallocated transfer slots are not counted twice. Zero on P. |
| `kv_used_tok` | Active footprint including P, for diagnostics. |
| `token_usage` | Native physical block-pool usage diagnostic. It is not used to derive active decode tokens. |
| `running_reqs`, `waiting_reqs` | Running count, and deduplicated waiting + skipped-waiting count. |
| `transfer_reqs`, `retracted_reqs` | Remote-KV-wait and preempted states. |
| `prealloc_reqs` | Waiting without reservations, excluding remote-KV wait; zero on P. |
| `bootstrap_reqs`, `inflight_reqs` | Remote-KV-wait on P/unified, and all current scheduler requests, respectively. Debug counts, not token load. |

`core` emits top-level `num_running_reqs`, `num_waiting_reqs`,
`num_waiting_uncached_tokens`, `num_used_tokens`, `num_total_tokens` and
`token_usage`. `disagg` emits `disaggregation` with role and queue counts.
All 16 Router-required fields are present in `router_load`. Unavailable
throughput/latency fields are omitted, not invented.

For hybrid attention, the largest group footprint is a **logical routing cost**,
not total bytes across groups/layers. Pending raw tokens represent logical
unreserved demand, not exact future physical allocation for bounded SWA/state
caches. Compare similar deployments; this does not normalize hardware/model
speed or heterogeneous layouts. No connector network-thread queues, offload
stores, CPU KV pools or device tensors are inspected. Full/SWA paths are
covered here; other cache-spec semantics require additional validation.

Prefix matching uses `coordinator.find_longest_cache_hit`, including native
hybrid/SWA/MTP boundaries. It avoids `get_computed_blocks`, which can publish
BlockStored events. Reading loads does not allocate/free/touch blocks, query
remote caches or update prefix-hit statistics.

## Failure handling and cost

vLLM's public DP utility broadcast returns only the first result. This plugin
uses the existing per-core utility primitive and retains every row. Any rank
failure yields 503; partial snapshots are not labelled complete.

At most one utility batch is outstanding per API process. HTTP timeout is
400 ms, below the Router's normal 500 ms timeout. Timeout/client cancellation
does not cancel the native utility futures; subsequent polls wait for the same
batch rather than accumulating queries behind a hung core. Later polls start
a new batch after completion. Completed stale results are not cached/replayed.

Cost scales with queue length, allocated block-table length and native prefix
lookup cost, once per poll. Local CPU microbenchmark: 640 waiting requests,
5000 tokens each, empty prefix cache, 100 samples: mean **1.366 ms**, p95
**1.427 ms**. This is not an NPU throughput result or a bound for very long
cached prefixes. Measure the target workload before raising poll frequency.

## Validation

CPU tests use native vLLM Request, KVCacheManager, prefix hashes, allocations,
cache reuse and EngineCore utility dispatch/serialization. No NPU model loads.

```bash
# This checkout must be importable; use an absolute path for PYTHONPATH.
export PYTHONPATH=/absolute/path/to/vllm-iquest-npu
python -m pytest --noconftest tests/ut/observability -q
python tests/e2e/cpu/router_load_smoke.py \
  /path/to/llm-router /path/to/vllm-router /persistent/output/load-smoke
```

The smoke test starts isolated loopback API/Router processes, uses this plugin
and native KV state, and verifies both DP rows and their later released/idle
state in `/debug/engine_load_snapshot`. It does not run model inference.
The isolated command bypasses the full repository NPU conftest; that conftest
was blocked locally by missing `scipy` in the NPU patch import chain. This is
not a full-suite or NPU end-to-end pass.
