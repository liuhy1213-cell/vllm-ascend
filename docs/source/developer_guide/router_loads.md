# Router load snapshots

The Ascend plugin adds `GET /v1/loads?include=core,disagg,router_load` for
LLM Router's polling contract. `/v1/load` is an alias. The Router normally
polls every 500 ms; the Engine does not push or run a new per-token reporting
loop. Native `/load` and `/metrics` are unchanged.

## Enable

Install this version of vllm-ascend in both the API and EngineCore environments.
The implementation requires vLLM's endpoint-plugin API and V1 utility API,
validated with vLLM 0.30.0. Older releases without endpoint plugins need a
separate adapter.

The endpoint is enabled by default when the Ascend platform is loaded. No
`VLLM_PLUGINS` setting or new startup flag is required; existing allowlists
remain unchanged, including an empty list. This only enables the load API,
not any other optional endpoint plugin.

```bash
# Start vllm serve with the existing model/TP/DP/connector configuration.
curl -fsS 'http://127.0.0.1:8000/v1/loads?include=core,disagg,router_load'
```

Use the existing Engine API authentication headers if required. No new HTTP
listener is opened. Ascend platform registration installs two idempotent hooks:
the native general-plugin initialization also installs the EngineCore method,
and the native endpoint loader includes the load endpoint for generation tasks.
Both hooks retain the original loaders and their filtering of other plugins;
no environment variables are rewritten. EngineCore imports are deferred until
engine initialization to avoid platform-discovery circular imports. The API
uses the native endpoint attach/state lifecycle. Pooling-only servers do not
receive these routes. Missing Engine methods or failed ranks return 503, never
a fabricated idle snapshot.
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
| `p_pending_uncached_tok` | Remaining prefill work in waiting/skipped-waiting/running-chunked requests, minus native local hits or accounted progress. Schema v2 additionally discounts unknown external hits using the five-minute token-weighted estimate described below. Preempted requests include generated tokens requiring recomputation. Zero on D. |
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
All 16 original Router-required fields remain present in `router_load`. Schema v2 adds `schema_version`, `instance_id`, and `prefill` diagnostics. Unavailable
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

## External cache estimates (schema v2)

An unscheduled request can have reusable external KV even when its local
prefix lookup misses. Previously that entire local miss was counted as new
compute. Schema v2 preserves the conservative upper bound and reports an
explicit estimate for the unresolved part.

For each EngineCore / DP rank, a five-minute history records **native connector
admission lookup** results. The conditional rate is:

```text
h = sum(external hit tokens) / sum(tokens queried after local-cache hits)
U = target tokens - native local hit tokens
estimated external hit = min(floor(U * h), U - mandatory recompute suffix)
remaining estimate = U - estimated external hit
```

Misses belong in the denominator. Long requests weigh more than short ones.
Normal and previously preempted admissions have separate histories. Buckets
have one-second resolution and expire after 300 seconds; fewer than one extra
second can be discarded at a boundary. No-history or expired-history snapshots
use zero estimated hits, not a fabricated average. Sample counts, token counts,
and the rate are exposed. History starts empty on process restart and is cleared
on successful prefix-cache flush, including pending native samples from before
the flush. This also prevents benchmark warmup history from leaking into a
subsequent cold round. An independent external-store reset that bypasses the
Engine is not observable here; use the Engine flush too when resetting a test.

The history is observed at `Scheduler.make_stats`, which already drains each
native statistics batch exactly once. Polling more frequently does not add
samples. Native statistics must be enabled (the default); with
`--disable-log-stats`, `history_collection_enabled` is false and there is no
historical discount. Invalid native batches are ignored and counted in
`external_history.invalid_batches`; they never interrupt inference.

This is a lookup-hit estimator, **not a measurement of successful H2D transfer**.
It does not query Mooncake, mutate cache residency, allocate KV, synchronize the
NPU, or change scheduler admission. "External" means the configured connector;
for the current P-side AscendStore it is CPU L2, but PD connector transfers or
other stores must not universally be labelled L2.

The estimate applies only to queued, unconfirmed external lookups. Running
requests and `WAITING_FOR_REMOTE_KVS` use native `num_computed_tokens` directly;
already-accounted L1/L2 hits are never subtracted again. That counter includes
scheduler-accounted progress, which can run ahead of physical NPU completion.
For MTP, the estimated hit cannot consume the prompt's required final block
suffix. A known running recomputation with zero progress receives no historical
discount. Transfer-error fallback follows the scheduler's updated progress.

`router_load.prefill` contains:

| Field | Meaning |
| --- | --- |
| `pending_tokens_upper_bound`, `bootstrap_tokens_upper_bound` | Work before the historical discount, for the corresponding legacy queue field. |
| `unknown_external_lookup_tokens`, `unknown_external_lookup_reqs` | Local misses whose external hits remain unconfirmed in this snapshot. |
| `estimated_external_hit_tokens` | Discount actually applied to those unknown requests. |
| `known_remaining_tokens` | Remaining accounted work outside the unknown-lookup population; this is not device-synchronized progress. |
| `external_restore_tokens_upper_bound` | Originally admitted external token count for current remote waits where metadata is available; NOT remaining transfer progress. |
| `external_restore_unknown_reqs` | Remote waits whose admission breakdown is unavailable; never infer zero bytes from absent metadata. |
| `external_history`, `history_collection_enabled` | Five-minute samples/rates and availability of their native source. |

The following identity holds within one snapshot:

```text
p_pending_uncached_tok + p_bootstrap_uncached_tok
  = pending_tokens_upper_bound + bootstrap_tokens_upper_bound
    - estimated_external_hit_tokens
  = known_remaining_tokens + unknown_external_lookup_tokens
    - estimated_external_hit_tokens
```

Optional `include=requests` adds native Engine request IDs, status, target and
accounted tokens, local lookup hits, estimate source/amount, mandatory recompute
suffix, and admission cache breakdown when still available. It contains no
prompt/token payload. Native IDs are not assumed to equal client HTTP IDs; a
future Router reconciliation must preserve their correlation explicitly.
The default response omits the per-request list. `instance_id` identifies the
scheduler lifetime; `external_history.reset_count` identifies history resets.

```bash
curl -fsS 'http://127.0.0.1:8000/v1/loads?include=core,router_load,requests'
```

Old consumers can continue to use the legacy aggregate fields, which now
include the labelled historical estimate. New consumers can use the upper
bound and split fields to apply their own estimation policy. Do not subtract
`estimated_external_hit_tokens` from the legacy aggregate a second time.
Decode KV allocation accounting and Router scoring are unchanged. Router's
lifetime reservation can still dominate `max(engine, router)`; correcting the
Engine estimate alone does not fix that independent issue.

## Failure handling and cost

vLLM's public DP utility broadcast returns only the first result. This plugin
uses the existing per-core utility primitive and retains every row. Any rank
failure yields 503; partial snapshots are not labelled complete.

The API keeps one latest successful snapshot and at most one utility batch per
API process. HTTP reads with a valid snapshot return immediately while an
on-demand refresh runs through the existing EngineCore utility transport.
Refreshes start no more often than every 500 ms on a healthy channel; no timer,
background thread, NPU synchronization, or scheduler/model execution changes
are introduced. Concurrent polls share the same in-flight query.

This borrows SGLang's separation of snapshot production from HTTP reads:
`Scheduler.publish_load_snapshot()` publishes through its snapshot writer,
while `TokenizerControlMixin.get_loads()` reads that published state. This
adapter retains vLLM's native utility transport rather than introducing SHM or
modifying its scheduler loop.

Cold/expired reads wait at most 400 ms. HTTP timeout or cancellation does not
cancel the native utility query. Late successful results are retained rather
than discarded at the next poll. A snapshot is valid for at most two seconds,
measured monotonically **from RPC dispatch**, conservatively including queue
and transport time. Returning it never changes its Engine-generated sequence
or resets its age. The response's `snapshot` object exposes
`age_upper_bound_ms`, `max_age_ms`, `refresh_inflight`, and
`last_query_duration_ms`. A dispatch delayed over two seconds is rejected even
if its actual Engine collection happened recently; freshness wins over
availability. No finite timeout can promise availability during an arbitrarily
long Engine stall.

A failed or partial DP refresh invalidates the cached aggregate. On the next
read, the API retries if the native utility channel is free. If another native
utility future is outstanding, monitoring defers without deleting it. This
also bounds the send-failure case: vLLM registers a utility future before send,
and a failed send can leave it awaiting a late reply. Polling must not allocate
more orphan futures behind that failure. A channel that never recovers remains
unavailable until the client/Engine is recovered; it is not silently labelled
healthy. Ordinary inference RPCs are unchanged.

The composed FastAPI router lifespan cancels the API-owned task and drops the
snapshot on shutdown, including when vLLM supplies a custom app lifespan.
Regular request cancellation never uses that shutdown cleanup. There is no
history of snapshots, retry-task list, or per-poll background loop. At most the
previous snapshot and a replacement being built can coexist during refresh;
individual snapshots scale with the current scheduler state.

Flush/reset is visible on the next successful refresh, rather than requiring
the HTTP cache to synchronously query the scheduler. Clients validating a reset
must wait for an increased `external_history.reset_count` and the cleared
statistics; any old response retains its old sequence and bounded age.

Cost scales with queue length, allocated block-table length and native prefix
lookup cost, once per poll. The 2026-11-18 CPU microbenchmark compared the old
reporter with this version using native requests/KV managers, 5000 tokens per
request, an empty prefix cache and 100 samples:

| Queued requests | Old median (ms) | New median (ms) | New p95 (ms) |
| --- | --- | --- | --- |
| 1 | 0.0075 | 0.0120 | 0.0129 |
| 32 | 0.1103 | 0.1840 | 0.1856 |
| 128 | 0.4276 | 0.7131 | 0.7217 |
| 640 | 2.1929 | 3.7033 | 3.7258 |

This ran while the experimental P engines were loading. It is not an NPU
throughput result or a bound for very long cached prefixes. Measure the target
workload before raising poll frequency.

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

### 2026-11-18 experimental validation

The updated reporter passed 29 targeted CPU tests and a 34-request test on the
experimental Ascend 2P1D deployment. The latter checked native metric deltas
against both P histories, aggregate identities, actual queued-request
estimation, successful flush resets, and final Engine health. All 34 requests
succeeded, all 20 load polls succeeded, and native/history token totals matched
exactly. This supplements, but does not replace, the full repository test suite.

A subsequent hisys5k round used 640 requests at requested RPS 6, serial turns
within each session, and 128 output tokens. Engine prefix caches, both dedicated
1248 GiB P-side CPU stores, and Router cache state were flushed first. Router
scoring was unchanged. All 640 requests succeeded; the Engines were healthy and
idle afterward, with history totals again exactly matching native metric deltas.

| Metric | Prior reporter round | Updated reporter round |
| --- | --- | --- |
| Actual send RPS | 5.9674 | 5.8994 |
| Mean TTFT (s) | 4.0496 | 3.9037 |
| P95 TTFT (s) | 9.8609 | 11.2885 |
| Mean TPOT (ms) | 9.8513 | 10.0246 |
| Output tokens/s | 737.4220 | 727.6094 |
| Native L1 + external cache hit tokens / input tokens | 67.1177% | 68.3553% |
| Session P switches / 560 transitions | 198 | 179 |

The updated round passed the mean TTFT < 4 s and mean TPOT < 25 ms criteria.
These are single rounds, not proof of stable performance improvement: actual
send rates differ, P95 TTFT worsened, and the updated round additionally polled
Engine load details for diagnosis. Throughput did not improve.

The updated round's diagnostic monitor encountered load-endpoint HTTP 503 in
38 of 76 sampling cycles; these were not inference errors. Router's separately
sampled view marked P0 stale once and P1 zero times, with maximum snapshot ages
of 3866/2849 ms. The existing 400 ms utility timeout/late-result handling remains
an independent issue. Router lifetime reservations can also still dominate
`max(engine, router)`. These were the remaining limitations of the first experimental revision;
the subsequent LoadReader change described above addresses the utility-result
handling. Router lifetime reservations remain unchanged.

### LoadReader regression and resource-lifetime checks

The 39-test targeted suite includes the historical estimator tests and the
following new reader checks:

- A late successful result survives HTTP timeout, preserving sequence and age.
- 100 concurrent reads with half cancelled share one utility query; repeated
  timeouts do not cancel or duplicate it.
- Repeated HTTP reads cannot renew an expired snapshot. A late reply whose
  dispatch age already exceeds the limit is rejected.
- A failed DP rank invalidates the whole aggregate, with later recovery tested.
- FastAPI router lifespan cleanup composes with a custom application lifespan.
- 4200 successful refreshes allocate 64 KiB test payloads. Weak references prove
  only the current payload survives; traced retained allocation growth between
  two 2000-refresh windows is below 128 KiB. No reader task remains running.
- 2000 alternating successful/failed refreshes do not retain old payloads via
  exception tracebacks, and close releases the last snapshot/task.
- The actual `AsyncMPClient._call_utility_async` and `_process_utility_output`
  are exercised with controlled transport. Thirty timed-out polls leave exactly
  one future per DP rank (two total); replies remove both. Thirty polls after
  a native send failure leave one future rather than accumulating more. A late
  reply clears it, after which polling recovers.

A real-time regression using the unchanged old/new LoadReader class source
sets each utility response delay to 600 ms and sleeps 500 ms between HTTP
attempts. The old implementation completes five RPCs but times out on all five
HTTP attempts. The new implementation has one cold-start timeout followed by
four successful reads (roughly 0.004–0.024 ms within the reader), completing
three RPCs. Repeated reads retain their original sequences. These are transport
simulations, not NPU performance results or a proof against every possible
long-running resource leak.

### LoadReader real-engine verification (2026-11-18)

All four authorized experimental Ascend API/Engine processes were restarted
with unchanged arguments and environment. After readiness, the same cold-cache
640-request RPS-6 protocol completed 640/640 successfully. Mean TTFT was
3.6366 s, mean TPOT 9.9767 ms, and output throughput 725.33 tokens/s. The previous
estimator-only round measured 3.9037 s, 10.0246 ms and 727.61 tokens/s. These single
rounds do not establish a stable performance change.

The diagnostic sampler had one load-API error cycle out of 108, versus 38/76
before the reader fix; the Router's snapshots were never classified stale in
this round. Its maximum P snapshot ages were 1826/1012 ms. Successful sampled
RPCs took up to 1412.84 ms; the valid response's dispatch-age upper bound never
exceeded 2000 ms. Around the single early 503, the prior P0 snapshot was already
1801.87 ms old with refresh still pending. The adapter intentionally still
returns unavailable when no valid snapshot arrives within the wait budget.
Both P histories again matched native metric deltas exactly after the workload.

A subsequent idle-engine soak made 14,400 HTTP load queries across four APIs
(20 queries/s/API, three roughly one-minute epochs). All returned 200; mean
HTTP latency by node/epoch was 0.56–0.64 ms. Sequences advanced throughout.
Memory was sampled every two seconds; the following are per-epoch medians in
MiB, after the inference workload had completed:

| API node | RSS epoch 1 / 2 / 3 | Anonymous RSS epoch 1 / 2 / 3 | Threads |
| --- | --- | --- | --- |
| 10.222.44.20 | 2265.71 / 2265.71 / 2265.71 | 1718.75 / 1718.75 / 1718.75 | 245 throughout |
| 10.222.44.62 | 1793.86 / 1793.43 / 1789.13 | 1710.28 / 1710.28 / 1710.28 | 245 throughout |
| 10.222.37.25 | 2322.67 / 2322.67 / 2322.67 | 1707.66 / 1707.66 / 1707.66 | 245 throughout |
| 10.222.41.28 | 2393.37 / 2393.37 / 2393.37 | 1707.13 / 1707.13 / 1707.13 | 245 throughout |

FD counts stayed within 84–86, 90–92, or 88–90 depending on node; they did not
accumulate across epochs and dropped after the soak client disconnected. All
Engines remained healthy. This is a finite-duration resource-lifetime check,
not a guarantee against all long-running leaks. It complements the explicit
snapshot/task/future bounds and injected-failure unit tests.
