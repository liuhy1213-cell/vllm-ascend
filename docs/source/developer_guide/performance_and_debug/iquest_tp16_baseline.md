# IQuest-Q1 TP16 Baseline (BF16, 512k context, 2×Ascend 950DT)

Baseline performance and accuracy data for IQuest-Q1 served with vLLM-Ascend
using 2 nodes × 8 Ascend 950DT (TP16) and MTP speculative decoding.

- **Date**: 2026-11-08
- **Model**: IQuest-Q1, BF16 (no quantization), `max_model_len=524288`
- **Serving**: vLLM-Ascend, 2 nodes × 8 NPU, `--tensor-parallel-size 16`,
  `--distributed-executor-backend mp`, MTP eagle ×3
- **Service flags**: `--max-num-batched-tokens 1024`, `--max-num-seqs 8`,
  `--gpu-memory-utilization 0.95`, `--no-enable-prefix-caching`,
  `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16]}'`
- **Image**: `quay.io/ascend/vllm-ascend:v0.30.0-a5-openeuler`
- **Client**: evalscope 1.14.1, OpenAI-compatible `/v1/chat/completions`

## Accuracy

Single round, mean over all samples, `reasoning_effort=high`, batch size 8.

| Dataset | n | mean |
|---|---:|---:|
| AIME-2025 | 30 | 0.9667 |
| AIME-2026 | 30 | 0.9333 |
| GSM8K | 1319 | 0.9780 |

## Performance

Cold cache (`--no-enable-prefix-caching`), streaming, `ignore_eos=true`,
output fixed at 1024 tokens. Throughput is reported per 16 NPUs.

SLA thresholds below are defined for this baseline run (not vendor SLA):
1k/32k input: TTFT ≤ 10 s / 30 s, TPOT ≤ 50 ms;
128k: TTFT ≤ 60 s, TPOT ≤ 50 ms; 500k: TTFT ≤ 180 s, TPOT ≤ 60 ms.

| Input | Conc. | TTFT avg (ms) | TTFT p99 (ms) | TPOT avg (ms) | TPOT p99 (ms) | Output tok/s | Total tok/s | SLA |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1k | 1 | 512 | 609 | 17.03 | 21.77 | 57.1 | 115.3 | PASS |
| 1k | 2 | 675 | 691 | 12.97 | 14.60 | 102.0 | 205.9 | PASS |
| 1k | 4 | 759 | 1213 | 16.51 | 18.20 | 210.9 | 424.6 | PASS |
| 1k | 5 | 777 | 842 | 49.74 | 66.17 | 72.5 | 146.2 | PASS |
| 1k | 6 | 985 | 1187 | 52.48 | 74.43 | 89.2 | 179.9 | FAIL |
| 1k | 8 | 942 | 1170 | 58.21 | 79.84 | 113.6 | 228.8 | FAIL |
| 32k | 1 | 6099 | 6129 | 13.82 | 14.06 | 50.6 | 1688.7 | PASS |
| 32k | 2 | 6499 | 6560 | 21.73 | 37.71 | 58.7 | 1957.5 | PASS |
| 32k | 4 | 7530 | 13143 | 29.45 | 52.71 | 86.6 | 2889.5 | PASS |
| 32k | 5 | 7656 | 11134 | 53.08 | 83.14 | 58.4 | 1949.2 | FAIL |
| 32k | 6 | 11276 | 18559 | 59.43 | 97.05 | 67.6 | 2255.9 | FAIL |
| 32k | 8 | 12911 | 27402 | 59.70 | 72.93 | 92.2 | 3075.6 | FAIL |
| 128k | 1 | 23398 | 23779 | 18.93 | 21.98 | 23.9 | 3123.3 | PASS |
| 128k | 2 | 27293 | 35885 | 30.33 | 39.19 | 31.3 | 4079.8 | PASS |
| 128k | 3 | 24982 | 25321 | 46.45 | 56.16 | 33.8 | 4413.5 | PASS |
| 128k | 4 | 33224 | 48126 | 52.72 | 68.55 | 38.2 | 4982.4 | FAIL |
| 500k | 1 | 91521 | 91792 | 27.33 | 29.67 | 8.6 | 4341.0 | PASS |
| 500k | 2 | 119362 | 129068 | 57.72 | 71.39 | 9.5 | 4790.6 | PASS |
| 500k | 3 | 211803 | 216631 | 58.43 | 69.45 | 8.7 | 4393.8 | FAIL |
| 500k | 4 | 304240 | 309960 | 60.17 | 72.74 | 8.3 | 4208.9 | FAIL |

All runs completed with 100% request success.

### Summary

| Input | Max passing concurrency | Peak output throughput |
|---|---:|---:|
| 1k | 5 | 210.9 tok/s @ conc 4 |
| 32k | 4 | 86.6 tok/s @ conc 4 |
| 128k | 3 | 33.8 tok/s @ conc 3 |
| 500k | 2 | 9.5 tok/s @ conc 2 |

## Observations

1. **TPOT step at concurrency ≥ 5**: at 1k and 32k input, TPOT jumps from
   ~16–30 ms to ~50–60 ms when concurrency reaches 5, while request success
   stays at 100%. Output throughput peaks at concurrency 4 (1k: 210.9 tok/s;
   32k: 86.6 tok/s) and then drops. This looks like a scheduler / speculative
   decoding path switch near `max-num-seqs=8`, and is worth investigating
   (e.g. MTP batch thresholds).
2. **Long-context throughput is prefill-bound**: at 128k/500k the total
   throughput stays around 4.2k–5.0k tok/s (mostly prompt tokens), while output
   throughput is only 8–38 tok/s. TTFT grows from ~0.5–1 s at 1k to 23–33 s at
   128k and 92–119 s at 500k.
3. **Very limited concurrency at 512k context**: at 500k input, TPOT already
   reaches 57.7 ms at concurrency 2 (threshold 60 ms), and concurrency 3 fails
   the 180 s TTFT target (211.8 s).

## Functional summary (50 API E2E cases)

PASS 24 / WARN 9 / FAIL 11 / SKIP 6. Main findings:

- `</think>` leaks into visible `content`; `thinking` / `enable_thinking` /
  `reasoning_effort` controls have no effect; `reasoning_effort=medium` is
  accepted silently.
- Chat endpoint does not enable a tool parser
  (`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`);
  `/v1/responses` tool calling returns custom text instead of a standard
  `function_call` item.
- Strict JSON output is not reliable with thinking content enabled.

## Reproduction

Service launch (head node; the worker uses `--node-rank 1 --headless`, the
worker's own `VLLM_HOST_IP`/`HCCL_IF_IP`, and the same `--master-addr`):

```bash
docker run -d --name iquest-q1-tp16-mtp-head \
  --network host --privileged --shm-size 64g \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/ummu --device /dev/uburma \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/hixlep:/etc/hixlep:ro \
  -v /lib/route.conf:/lib/route.conf:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json:ro \
  -v /etc/cloud/topo:/etc/cloud/topo:ro \
  -v /usr/bin/urma_admin:/usr/bin/urma_admin:ro \
  -v <vllm-ascend-repo>:/workspace \
  -v <model-path>:/models/IQuest-Q1:ro \
  -w /workspace \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e GLOO_SOCKET_IFNAME=<ifname> -e HCCL_SOCKET_IFNAME=<ifname> -e TP_SOCKET_IFNAME=<ifname> \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_HOST_IP=<head-ip> -e HCCL_IF_IP=<head-ip> \
  -e HCCL_CONNECT_TIMEOUT=1200 -e HCCL_EXEC_TIMEOUT=7200 \
  quay.io/ascend/vllm-ascend:v0.30.0-a5-openeuler \
  python -m vllm.entrypoints.cli.main serve /models/IQuest-Q1 \
    --served-model-name IQuest-Q1 \
    --tensor-parallel-size 16 --distributed-executor-backend mp \
    --nnodes 2 --node-rank 0 --master-addr <head-ip> --master-port 29561 \
    --max-model-len 524288 --max-num-batched-tokens 1024 --max-num-seqs 8 \
    --gpu-memory-utilization 0.95 --no-enable-prefix-caching \
    --speculative-config '{"method":"eagle","model":"/models/IQuest-Q1/mtp","num_speculative_tokens":3}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16]}' \
    --cudagraph-metrics --host 0.0.0.0 --port 8002
```

Tests were driven by `evalscope-sla-kit` (functional, accuracy, and
`evalscope perf --sla-auto-tune`). Datasets: GSM8K / AIME (accuracy) and
fixed-length random prompts (performance). Per-tier request counts are small
(`number = concurrency × 2`, one run per level), so p99 values are indicative.
