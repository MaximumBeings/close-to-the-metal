# Chapter 16: Observability — Metrics, Logging, Tracing

> **"You cannot tune what you cannot see. And you cannot debug what you cannot trace."**

---

## What This Chapter Covers

A vLLM or llama.cpp deployment that is not instrumented is a black box. You know it is running; you do not know whether it is healthy. This chapter builds a complete observability stack from the ground up: the five metrics every practitioner must watch, the full vLLM Prometheus endpoint decoded field by field, OpenTelemetry distributed tracing for end-to-end request visibility, structured log output for scheduler debugging, and llama.cpp's built-in timing infrastructure. Every metric is given its arithmetic definition and its normal operating range.

---

## 16.1 The Five Essential Metrics

Before any dashboards, five numbers tell the story of an LLM serving system. All five must be measured simultaneously — a system that optimizes one in isolation usually degrades another.

### Metric 1: Time To First Token (TTFT)

```
TTFT = t_first_token − t_request_arrival   (milliseconds)
```

TTFT measures how long a user waits before the response begins. It is dominated by prefill latency: every prompt token must be processed before a single output token can be generated. TTFT is the metric users feel most acutely in interactive products.

**Normal ranges:**

| Workload | Acceptable TTFT | Warning |
|----------|----------------|---------|
| Chat (≤ 512 tok prompt) | < 300 ms | > 1 s |
| RAG (2K–16K tok prompt) | < 2 s | > 5 s |
| Batch / offline | No SLA | — |

**What degrades TTFT:**

- Long prompts with chunked prefill disabled (prefill starvation)
- High `max_num_seqs` competing for the token budget
- KV cache full → new requests queued (admission backpressure)

### Metric 2: Inter-Token Latency (ITL)

```
ITL = t_token_N − t_token_{N-1}   (milliseconds, averaged over all output tokens)
```

ITL measures the cadence of streaming output. It is determined almost entirely by the decode step latency — one forward pass per output token. In a well-configured system, ITL is nearly constant throughout a response.

**Normal ranges:**

| GPU | Model | Typical ITL |
|-----|-------|-------------|
| A100-80 | Llama-3-8B | 5–15 ms |
| A100-80 | Llama-3-70B, TP=4 | 15–40 ms |
| H100 | Llama-3-70B, TP=2 | 8–20 ms |
| RTX 4090 | 7B Q4 | 20–50 ms |

**ITL spikes indicate:**

- Chunked prefill chunks being processed (new arrivals interrupting decode)
- KV block eviction (paging to CPU)
- Thermal throttle (sustained decode causes GPU temperature rise)
- NCCL AllReduce contention (TP deployments)

### Metric 3: Request Throughput

```
throughput = completed_requests / time_window   (requests/second)
output_throughput = sum(output_tokens) / time_window   (tokens/second)
```

Two distinct throughput numbers matter. Request throughput is the scheduling rate; output throughput is the generation rate. A system processing many short requests can have high request throughput but low output throughput, and vice versa.

**[FOUNDATIONAL]** Output token throughput is the primary GPU utilization proxy for decode-bound workloads. If your A100-80 single-GPU output throughput is below ~100 tok/s on a 7B model, the GPU is underutilized — either the batch is too small or the pipeline is stalled.

### Metric 4: GPU utilization

```
gpu_utilization = (time_GPU_active) / (total_elapsed_time) × 100%
```

`nvidia-smi` reports this as `GPU-Util`. Importantly, high GPU-Util is not always a sign of health:

- **GPU-Util near 100%, ITL normal** → healthy, batch is full
- **GPU-Util near 100%, ITL spiking** → compute bottleneck (prefill monopolising GPU time)
- **GPU-Util near 50%** → batch too small, decode underutilising GPU
- **GPU-Util near 0%, requests queued** → KV cache full, scheduler blocked

Memory utilization (`Memory-Util` in nvidia-smi) is equally important:

```
memory_utilization = used_vram / total_vram × 100%
```

Under steady-state serving, memory utilization should be close to `gpu_memory_utilization` (the vLLM knob). If it drifts below 80%, the KV pool is larger than needed — you can serve more concurrent users. If it sits at 100%, you are at capacity.

### Metric 5: KV Cache Hit Rate

```
kv_cache_hit_rate = prefix_cache_hits / total_prompt_tokens
```

With `enable_prefix_caching=true`, vLLM tracks how many prompt tokens were served from the radix cache versus recomputed from scratch. A high hit rate (>80%) on workloads with repeated prefixes (system prompts, few-shot examples, RAG templates) means you are saving significant GPU compute.

**[DEEP DIVE]** The KV cache hit rate should be measured at block granularity, not token granularity. A prompt that shares 99 of 100 tokens with a cached sequence but the mismatch falls at the boundary of a 16-token block has a 0% hit rate for that block. The effective hit rate from an economics perspective is:

```
saved_prefill_tokens = prefix_hit_blocks × block_size
compute_saved_fraction = saved_prefill_tokens / total_prompt_tokens
```

For a system prompt of 2048 tokens with block_size=16: 128 blocks. If 127 of 128 match (one new token at the end), 127/128 = 99.2% of compute is saved.

---

## 16.2 The vLLM Prometheus Endpoint

vLLM exposes a `/metrics` endpoint in Prometheus exposition format when you add `--enable-metrics` to the serve command:

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-metrics \
  --port 8000
```

Metrics are then available at `http://localhost:8000/metrics`.

### Complete metric taxonomy

vLLM organizes its metrics into four families:

**Family 1 — Request lifecycle counters**

```
# HELP vllm:num_requests_running Number of requests currently running on GPU
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 45

# HELP vllm:num_requests_waiting Number of requests waiting to be processed
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 12

# HELP vllm:num_requests_swapped Number of requests swapped to CPU
# TYPE vllm:num_requests_swapped gauge
vllm:num_requests_swapped 3

# HELP vllm:num_preemptions_total Total number of preemptions
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total 7
```

`num_requests_swapped > 0` is a yellow flag: vLLM is evicting KV blocks to CPU because GPU KV pool is full. This causes ITL spikes when the sequence is later resumed.

`num_preemptions_total` counts sequences that were interrupted mid-decode (not just swapped). Repeated preemptions signal that `max_num_seqs` is set too high relative to the KV pool.

**Family 2 — KV cache utilization**

```
# HELP vllm:gpu_cache_usage_perc GPU KV cache usage (fraction of total blocks)
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.87

# HELP vllm:cpu_cache_usage_perc CPU KV cache usage (fraction of total blocks)
# TYPE vllm:cpu_cache_usage_perc gauge
vllm:cpu_cache_usage_perc 0.12

# HELP vllm:gpu_prefix_cache_hit_rate Rolling prefix cache hit rate (last 5 min)
# TYPE vllm:gpu_prefix_cache_hit_rate gauge
vllm:gpu_prefix_cache_hit_rate 0.73
```

`gpu_cache_usage_perc` is the KV pool fill fraction. Values consistently above 0.95 mean you are one large request away from forced eviction. Values consistently below 0.50 mean you have unused capacity — lower `gpu_memory_utilization` to give more headroom to model weights, or increase `max_num_seqs`.

**Family 3 — Latency histograms**

```
# HELP vllm:e2e_request_latency_seconds E2E request latency
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{le="0.1"}  4
vllm:e2e_request_latency_seconds_bucket{le="0.5"}  87
vllm:e2e_request_latency_seconds_bucket{le="1.0"}  234
vllm:e2e_request_latency_seconds_bucket{le="5.0"}  1087
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 1092
vllm:e2e_request_latency_seconds_sum  892.4
vllm:e2e_request_latency_seconds_count  1092

# HELP vllm:time_to_first_token_seconds TTFT histogram
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.01"} 0
vllm:time_to_first_token_seconds_bucket{le="0.1"}  312
vllm:time_to_first_token_seconds_bucket{le="0.5"}  987
...

# HELP vllm:time_per_output_token_seconds ITL histogram (inter-token latency)
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.005"} 0
vllm:time_per_output_token_seconds_bucket{le="0.01"}  45237
vllm:time_per_output_token_seconds_bucket{le="0.05"}  89102
...
```

Prometheus histograms allow computing arbitrary percentiles. To compute P95 TTFT from the histogram buckets:

```
P95 = smallest bucket le where cumulative_count >= 0.95 × total_count
```

**Family 4 — Throughput gauges**

```
# HELP vllm:prompt_tokens_total Total prompt tokens processed
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 4892301

# HELP vllm:generation_tokens_total Total generation tokens produced
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 1234891

# HELP vllm:request_success_total Total successful requests
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop"}      9823
vllm:request_success_total{finished_reason="length"}     177
vllm:request_success_total{finished_reason="abort"}       42
```

`finished_reason="abort"` counts client disconnections and server-side cancellations. A spike here often indicates TTFT SLA violations causing client timeouts — the client gave up before the first token arrived.

`finished_reason="length"` counts requests that hit `max_tokens` without a natural stop. If this is high, your users are getting truncated responses — raise `max_tokens` or `max_model_len`.

### Prometheus scrape configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: vllm
    scrape_interval: 5s
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: /metrics
```

### Key Grafana dashboard queries

```promql
# TTFT P95 (approximation from histogram)
histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[5m]))

# ITL P99
histogram_quantile(0.99, rate(vllm:time_per_output_token_seconds_bucket[5m]))

# Output token throughput (tok/s over 1 minute)
rate(vllm:generation_tokens_total[1m])

# KV cache pressure
vllm:gpu_cache_usage_perc

# Queued requests (admission backpressure)
vllm:num_requests_waiting

# Preemption rate (should be near zero)
rate(vllm:num_preemptions_total[5m])

# Prefix cache efficiency
vllm:gpu_prefix_cache_hit_rate
```

**[COMMON TRAP]** The `vllm:e2e_request_latency_seconds` histogram includes queueing time — if `num_requests_waiting > 0`, E2E latency grows even if per-request processing is fast. Always look at TTFT and ITL separately from E2E latency when diagnosing problems. High E2E with normal TTFT/ITL means your admission rate is too low (`max_num_seqs` too small or GPU too slow).

---

## 16.3 OpenTelemetry Tracing

Prometheus metrics show aggregates. OpenTelemetry (OTel) traces show the path of individual requests through the system — essential for diagnosing tail latency and understanding where specific slow requests get stuck.

### Enabling OTel in vLLM

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp

# Export to a local Jaeger collector
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_SERVICE_NAME="vllm-server"

vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --otlp-traces-endpoint "http://localhost:4317/v1/traces" \
  --port 8000
```

### The vLLM span hierarchy

A single LLM request generates a tree of spans:

```
[span] HTTP request                     T=0 ms ──────────────────────── T=1200 ms
  [span] scheduler.add_request          T=0 ms ── T=0.1 ms
  [span] scheduler.step (waiting)       T=0 ms ─────────────────── T=450 ms
    [span] kv_cache.allocate_blocks     T=448 ms ─ T=450 ms
  [span] engine.execute_model           T=450 ms ─────────── T=510 ms
    [span] worker.prefill               T=450 ms ──── T=505 ms
      [span] attention.prefill_forward  T=450 ms ──── T=504 ms
    [span] worker.decode_step_1         T=505 ms ─── T=515 ms
    [span] worker.decode_step_2         T=515 ms ─── T=525 ms
    ...
  [span] detokenize                     T=1195 ms ─ T=1200 ms
  [span] stream_output                  T=505 ms ─────────────────── T=1200 ms
```

Key spans and what they reveal:

| Span | Normal duration | Slow means |
|------|----------------|-----------|
| `scheduler.step (waiting)` | 0–50 ms | KV cache full or `max_num_seqs` saturated |
| `worker.prefill` | 50–500 ms (prompt-dependent) | Long prompt with no chunked prefill |
| `attention.prefill_forward` | ~90% of prefill time | GPU compute / FlashAttention |
| `worker.decode_step_N` | 5–40 ms | Normal; spike = NCCL stall or thermal |
| `kv_cache.allocate_blocks` | < 1 ms | Spike = fragmentation (block compaction) |
| `detokenize` | < 1 ms | Spike = UTF-8 boundary flush delay |

### Custom OTel instrumentation

You can add spans to your own inference client:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("my_app")

async def call_vllm(prompt: str, request_id: str) -> str:
    with tracer.start_as_current_span("vllm_call") as span:
        span.set_attribute("request.id", request_id)
        span.set_attribute("prompt.length", len(prompt.split()))

        import time
        t0 = time.perf_counter()

        # ... HTTP call to vLLM ...
        first_token_time = None
        tokens = []

        async for token in stream_vllm(prompt):
            if first_token_time is None:
                first_token_time = time.perf_counter()
                span.add_event("first_token",
                               {"ttft_ms": (first_token_time - t0) * 1000})
            tokens.append(token)

        t_end = time.perf_counter()
        span.set_attribute("output.tokens", len(tokens))
        span.set_attribute("e2e_latency_ms", (t_end - t0) * 1000)
        return "".join(tokens)
```

---

## 16.4 Log-Structured Output for Scheduler Debugging

When Prometheus metrics show elevated `num_requests_waiting` or unexpected preemptions, the next step is structured log analysis to understand which requests are being admitted, deferred, or evicted.

### Enabling vLLM debug logging

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Meta-Llama-3-8B-Instruct 2>&1 | \
  grep -E "scheduler|preempt|swap|kv_cache"
```

Key log patterns and their meanings:

```
# Scheduler step summary (every N steps with DEBUG logging)
INFO scheduler.py:321 - Running: 45 reqs, Waiting: 12 reqs,
  Swapped: 0 reqs, Budget: 6144/8192 tokens

# Preemption event
WARN scheduler.py:287 - Preempting request abc123 due to KV capacity:
  running=48 seqs, kv_used=98.3%, selected_victim=longest_prefix

# Swap-in event (sequence resumed from CPU)
DEBUG block_manager.py:441 - Swapping in request def456: 
  blocks=34, src=cpu, dst=gpu, latency_ms=23.4

# Prefix cache hit
DEBUG prefix_cache.py:178 - Cache HIT request ghi789:
  matched_blocks=127/128, saved_tokens=2032, prefix_hash=0xdeadbeef
```

### Building a structured log pipeline

For production, pipe vLLM logs to a structured format:

```bash
# vLLM with JSON logging
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-metrics \
  2>&1 | python3 -c "
import sys, json, re, time

for line in sys.stdin:
    ts = time.time()
    level = 'INFO'
    if 'WARN' in line: level = 'WARN'
    elif 'ERROR' in line: level = 'ERROR'
    elif 'DEBUG' in line: level = 'DEBUG'

    # Extract scheduler state if present
    m = re.search(r'Running: (\d+).*Waiting: (\d+).*Budget: (\d+)/(\d+)', line)
    if m:
        record = {
            'ts': ts, 'level': level,
            'running': int(m.group(1)),
            'waiting': int(m.group(2)),
            'budget_used': int(m.group(3)),
            'budget_total': int(m.group(4)),
        }
        print(json.dumps(record))
    else:
        print(json.dumps({'ts': ts, 'level': level, 'msg': line.strip()}))
" | tee vllm_structured.jsonl
```

---

## 16.5 ASCII Diagram: Metrics Flow

```
  ┌──────────────────────────────────────────────────────┐
  │                  vLLM Server                          │
  │                                                       │
  │  ┌─────────┐  ┌──────────┐  ┌──────────────────┐    │
  │  │Scheduler│→ │  Engine  │→ │   Sampler/Output  │    │
  │  └────┬────┘  └────┬─────┘  └────────┬─────────┘    │
  │       │             │                  │               │
  │       ▼             ▼                  ▼               │
  │  ┌──────────────────────────────────────────────┐    │
  │  │         StatsLogger (internal)                │    │
  │  │  - TTFT per request                           │    │
  │  │  - ITL per token                              │    │
  │  │  - KV block counts                            │    │
  │  │  - Prefix hit/miss per block                  │    │
  │  └──────────────┬───────────────────────────────┘    │
  └─────────────────┼────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
  ┌──────────────┐    ┌────────────────────┐
  │  /metrics    │    │  OTLP gRPC exporter│
  │  (Prometheus)│    │  (OpenTelemetry)   │
  └──────┬───────┘    └─────────┬──────────┘
         │                      │
         ▼                      ▼
  ┌──────────────┐    ┌────────────────────┐
  │  Prometheus  │    │  Jaeger / Tempo     │
  │  + Grafana   │    │  (trace backend)    │
  └──────────────┘    └────────────────────┘

  Metric interpretation pipeline:
  
  TTFT > 1s  →  check: num_requests_waiting? prefill_starvation? long prompt?
  ITL spikes →  check: preemptions? NCCL stall? thermal throttle?
  KV > 95%   →  check: max_num_seqs? max_model_len? prefix cache block size?
  GPU-Util<50%→ check: batch too small? max_num_batched_tokens?
```

---

## 16.6 llama.cpp Timing Infrastructure

llama.cpp provides built-in timing through `llama_perf_context_print` and optional JSON log output. Unlike vLLM's Prometheus approach, llama.cpp reports timing at the **end of inference** rather than streaming live metrics.

### llama_perf_context_print

After each inference run, call:

```c
// C API
llama_perf_context_print(ctx);
```

This prints to stderr:

```
llama_perf_context_print: load time =   423.12 ms
llama_perf_context_print: sample time =    12.43 ms /   256 runs (    0.05 ms per token,  20597.39 tokens per second)
llama_perf_context_print: prompt eval time =  1204.12 ms /   512 tokens (    2.35 ms per token,   425.23 tokens per second)
llama_perf_context_print: eval time =  4892.34 ms /   256 runs (   19.11 ms per token,    52.33 tokens per second)
llama_perf_context_print: total time =  6120.01 ms /   768 tokens
```

Field meanings:

| Field | Meaning | Key metric |
|-------|---------|-----------|
| `load time` | Model load + CUDA context | One-time startup cost |
| `sample time` | Sampling pipeline (softmax + filter + pick) | Overhead per output token |
| `prompt eval time` | Prefill latency | TTFT proxy (total) |
| `eval time` | Decode latency | ITL = eval_time / n_runs |
| `total time` | prompt eval + eval | E2E excluding streaming |

From the example above:
```
TTFT  ≈ prompt eval time = 1204 ms  (512-token prompt)
ITL   ≈ eval time / runs  = 4892 / 256 = 19.1 ms/token
Throughput = 768 tokens / 6.12 s = 125.5 tok/s (total)
```

### JSON log mode

```bash
llama-cli \
  --model /models/llama-3-8b-q4_k_m.gguf \
  --prompt "Explain quantum entanglement" \
  --n-predict 256 \
  --log-format json \
  --log-file /tmp/llama_run.json
```

The JSON log captures per-token timing:

```json
{
  "type": "PERF",
  "t_start_ms": 1716400000.0,
  "t_end_ms":   1716400006.12,
  "n_p_eval": 512,
  "n_eval": 256,
  "t_p_eval_ms": 1204.12,
  "t_eval_ms": 4892.34,
  "t_sample_ms": 12.43,
  "n_sample": 256
}
```

For per-token timing you need to instrument the decode loop directly (see companion code).

### Custom timing harness for llama.cpp

The standard `llama_perf_context_print` gives aggregate stats. For per-token ITL distribution (needed to detect stalls and temperature throttle), you must instrument the decode loop:

```c
// Pseudocode: per-token timing in llama.cpp decode loop
std::vector<double> token_latencies;

for (int i = 0; i < n_predict; i++) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // --- decode one token ---
    if (llama_decode(ctx, batch) != 0) break;

    auto t1 = std::chrono::high_resolution_clock::now();
    double itl_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    token_latencies.push_back(itl_ms);

    // sample next token
    auto logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    // ... sample ...

    // clear batch, add new token
    llama_batch_clear(batch);
    llama_batch_add(batch, next_token, i + n_prompt, {0}, true);
}

// Compute ITL statistics
double sum = 0, sq = 0;
double min_itl = token_latencies[0], max_itl = token_latencies[0];
for (double t : token_latencies) {
    sum += t; sq += t*t;
    min_itl = std::min(min_itl, t);
    max_itl = std::max(max_itl, t);
}
double mean = sum / token_latencies.size();
double stddev = std::sqrt(sq / token_latencies.size() - mean * mean);

printf("ITL: mean=%.2f ms  stddev=%.2f ms  min=%.2f ms  max=%.2f ms\n",
       mean, stddev, min_itl, max_itl);
```

**[DEEP DIVE]** llama.cpp does not have built-in Prometheus support. In production deployments, the recommended approach is a thin wrapper server (llama-server already does this via its `/metrics` endpoint in newer versions) that collects `llama_perf_context_print` output and exposes it as Prometheus gauges. The companion code implements this pattern.

### llama-server monitoring endpoints

llama-server (the HTTP wrapper around llama.cpp) exposes:

```bash
# Health check
curl http://localhost:8080/health
# → {"status":"ok"}

# Server-wide stats
curl http://localhost:8080/metrics
# → Prometheus text format (newer versions)

# Slot-level diagnostics (parallel decodes)
curl http://localhost:8080/slots
# → [{"id":0,"state":1,"prompt":"...","tokens_predicted":45,...}, ...]
```

The `/slots` endpoint is particularly useful for multi-stream debugging: each slot corresponds to one parallel decode sequence (`--parallel N`). A slot in state 0 is idle; state 1 is actively decoding.

---

## 16.7 Worked Example: Diagnosing a TTFT Regression

A production deployment shows P95 TTFT rising from 400 ms to 2,800 ms after a traffic spike. Here is the diagnostic walkthrough using the metrics above.

### Step 1: Check admission backpressure

```promql
vllm:num_requests_waiting
```

Result: jumps from 0 to 47 during the incident.

**Interpretation:** 47 requests queued. New arrivals wait behind them before prefill even starts.

### Step 2: Check KV cache saturation

```promql
vllm:gpu_cache_usage_perc
```

Result: 0.98 (98% full) during incident.

**Interpretation:** KV pool exhausted. Scheduler cannot admit new requests without evicting running ones.

### Step 3: Check if chunked prefill is helping

```promql
histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[5m]))
```

During the incident, P95 TTFT = 2.8 s. Before incident, P95 = 0.4 s.

The TTFT increase = 2.4 s = queuing delay + prefill time.

Queuing delay ≈ waiting_requests × avg_step_time ≈ 47 × 15 ms ≈ 705 ms

But the increase is 2.4 s — much larger than queuing alone. This suggests some requests are doing full prefill without chunking.

### Step 4: Check preemptions

```promql
rate(vllm:num_preemptions_total[5m])
```

Result: 3.2 preemptions/second during incident.

**Interpretation:** With 98% KV utilization, the scheduler is preempting running sequences to free KV blocks for new prefills. Each preemption interrupts decode and adds latency.

### Root cause and fix

The workload shifted to longer prompts (RAG queries with larger context windows). The original config:

```yaml
max_num_seqs: 256
max_model_len: 8192
```

At 256 concurrent seqs × 8192 tokens each = 2.1M tokens required, but KV capacity is only 455K tokens (from chapter 14 calculation). This is the OOM Triangle pattern. The scheduler admits too many sequences, saturates the KV pool, and begins preempting to juggle them.

Fix:

```yaml
max_num_seqs: 50           # lower to match actual KV capacity
max_model_len: 16384       # raise to serve RAG prompts
enable_chunked_prefill: true
max_num_batched_tokens: 16384
```

After fix: P95 TTFT returns to 380 ms, preemption rate drops to 0.

**[COMMON TRAP]** Many teams respond to TTFT regressions by increasing `max_num_batched_tokens` (thinking "more compute headroom"). This makes prefill faster per request but does nothing about the queuing delay caused by KV exhaustion. Always check KV cache utilization first.

---

## 16.8 Alerting Rules

### Prometheus alerting (Alertmanager)

```yaml
# alert_rules.yml
groups:
  - name: vllm_critical
    rules:
      - alert: HighTTFTP95
        expr: histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P95 TTFT > 2s"

      - alert: KVCacheSaturated
        expr: vllm:gpu_cache_usage_perc > 0.95
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "KV cache above 95% — preemptions imminent"

      - alert: HighPreemptionRate
        expr: rate(vllm:num_preemptions_total[5m]) > 1
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Preemption rate > 1/s — scheduler under pressure"

      - alert: RequestsAccumulating
        expr: vllm:num_requests_waiting > 20
        for: 30s
        labels:
          severity: warning
        annotations:
          summary: "More than 20 requests waiting in queue"
```

---

## 16.9 Summary

| Signal | Source | Normal range | Action if abnormal |
|--------|--------|-------------|-------------------|
| TTFT P95 | `/metrics` histogram | < SLA threshold | Check KV usage, chunked prefill, queue depth |
| ITL P99 | `/metrics` histogram | GPU decode speed ± 20% | Check preemptions, NCCL stalls, thermal |
| KV usage | `gpu_cache_usage_perc` | 60–90% | > 95% → lower max_num_seqs |
| Preemptions | `num_preemptions_total` | Near 0 | > 0 → OOM Triangle; reduce seqs or ctx |
| Queue depth | `num_requests_waiting` | < 5 | > 20 → undercapacity; scale or reconfigure |
| Prefix hit rate | `gpu_prefix_cache_hit_rate` | > 70% (chat/RAG) | < 30% → check block_size, system prompt consistency |
| GPU-Util | `nvidia-smi` | > 70% under load | < 50% → batch too small; raise max_num_seqs |

---

## Chapter Notes

**[FOUNDATIONAL]** The five metrics — TTFT, ITL, throughput, GPU utilization, KV hit rate — are not independent. They form a system: increasing throughput usually raises TTFT, raising KV hit rate reduces ITL variance, and reducing GPU utilization gives latency headroom. Optimizing one without watching the others is a common source of production regressions.

**[DEEP DIVE]** Prometheus histograms in vLLM use linear buckets by default. For latency histograms, exponential buckets provide better resolution at both the low end (catching sub-10ms events) and the high end (catching multi-second outliers). vLLM 0.4+ allows custom bucket boundaries via `--otlp-traces-endpoint` configuration.

**[COMMON TRAP]** `nvidia-smi`'s `GPU-Util` samples the GPU at 1/32 s intervals (on most drivers) and reports the fraction of time the GPU had at least one active kernel. A decode loop that keeps the GPU 80% busy but with 20% idle gaps (for memory allocation, NCCL syncs, or kernel launch overhead) shows 100% `GPU-Util`. This metric overstates true utilization. For accurate measurement, use `nvml_device_get_utilization_rates` with fine-grained sampling, or correlate with `ncu` (NVIDIA Nsight Compute) profiles.

---

*Companion code: `code/chapter_16/observability_demo.py` and `code/chapter_16/observability_demo.cpp`*


---

## Chapter Summary

- **Four golden signals for LLM serving**: TTFT (time to first token), ITL (inter-token latency), throughput (tokens/s), and GPU utilization.
- **vLLM Prometheus metrics**: exposed at `/metrics`; key metrics include `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds`, `vllm:num_requests_running`, `vllm:gpu_cache_usage_perc`.
- **P99 vs P50**: P99 TTFT matters for user experience; P50 throughput matters for cost efficiency — optimizing one often worsens the other.
- **GPU utilization pitfall**: `nvidia-smi` reports SM utilization averaged over 1-second windows; a GPU can show 95% while being 50% utilization within each forward pass.
- **Structured logging**: use JSON-formatted logs with `request_id`, `sequence_id`, `model_name`, and timing breakdowns for post-hoc analysis.
- **Distributed tracing**: OpenTelemetry spans from HTTP gateway through vLLM engine through GPU kernels give end-to-end latency breakdown.
- **Dashboard recipe**: Grafana panels for queue depth, cache hit rate, token throughput, and P99 TTFT are the minimum viable observability stack.

---

## Self-Check Questions

1. Your Grafana dashboard shows `vllm:gpu_cache_usage_perc` at 0.97 and `vllm:num_requests_running` at 120. What is happening? What metric would you check next? *(Section 16.3)*

2. P99 TTFT is 3.2 s and P50 TTFT is 0.4 s. What does this distribution tell you about the scheduler? Name two scheduling configurations that could cause this disparity. *(Section 16.2)*

3. `nvidia-smi dmon -s u` shows 89% GPU utilization. Simultaneously, `vllm:time_per_output_token_seconds` P50 is 80 ms. For a 70B BF16 model on an A100, is 80 ms ITL consistent with 89% utilization? Show the arithmetic. *(Section 16.1)*

4. Write a PromQL expression that alerts when the 5-minute rolling P99 TTFT exceeds 2 seconds. *(Section 16.4)*

5. An OpenTelemetry trace for a single request shows: HTTP gateway 2 ms, scheduler queue 450 ms, prefill 620 ms, decode (120 tokens) 4 800 ms, detokenise 3 ms. Identify the bottleneck and suggest one configuration change. *(Section 16.5)*


---

## Worked Solutions

### Question 1
**Observations:** `vllm:gpu_cache_usage_perc = 0.97`, `vllm:num_requests_running = 120`.

**What is happening:**
The KV cache is at 97% capacity with 120 concurrent sequences. The system is at the edge of cache exhaustion. Any additional requests requiring new blocks will trigger **preemption** — the scheduler will evict and recompute (or swap) one or more lower-priority sequences to free blocks for the new request.

**Next metric to check:** `vllm:num_requests_waiting`
- If this is non-zero and growing, requests are queuing because the scheduler cannot admit new sequences without first freeing blocks.
- Also check `vllm:gpu_cache_usage_perc` over time (rate of change): a rapidly rising metric indicates a memory leak or pathologically long sequences consuming all blocks.

**Secondary check:** `vllm:num_preemptions_total` — a rising preemption count alongside 97% cache usage confirms the system is thrashing. Consider reducing `max_num_seqs` or `max_model_len`.

---

### Question 2
**Observations:** P99 TTFT = 3.2 s, P50 TTFT = 0.4 s.

**What the distribution tells you:**
The median request receives prefill service within 0.4 s, but 1 in 100 requests waits 3.2 s. This is a **heavy-tailed waiting distribution**, not a uniform latency problem. The GPU is not uniformly slow — specific requests are waiting in the scheduler queue for a long time before their prefill begins.

**Two scheduling configurations that could cause this:**

1. **`max_num_batched_tokens` is too small relative to `max_num_seqs`.**
   With many short decode requests filling the token budget (e.g., 128 sequences × 1 token = 128 tokens), a large prefill request (e.g., 3,000 tokens) must wait for multiple scheduling rounds until the token budget opens up. P99 TTFT spikes for these "whales."

2. **No chunked prefill + large prompt variance.**
   When a single huge prompt (p99 length = 20K tokens) arrives, it blocks the entire token budget for multiple steps. All other requests admitted after it experience head-of-line blocking, inflating their P99 TTFT. Enabling `--enable-chunked-prefill` breaks this monopoly by splitting large prefills into chunks.

---

### Question 3
**Setup:** 70B BF16 on A100, GPU util = 89%, ITL P50 = 80 ms.

**Expected ITL calculation:**
For a 70B BF16 model, each decode step reads all 70B × 2 bytes = 140 GB of weights from HBM (batch=1). A100 HBM bandwidth = 2 TB/s.

**Theoretical minimum ITL at batch=1:**
```
t_min = 140 GB / 2,000 GB/s = 0.07 s = 70 ms
```

**Observed ITL: 80 ms.**
```
80 ms is within 14% of the 70 ms theoretical minimum.
```

At 89% GPU utilization, the extra 10 ms (80−70) is consistent with scheduling overhead, NCCL synchronization, and Python-layer latency. The 89% util metric reflects SM (streaming multiprocessor) activity, which includes memory load instructions — memory-bound workloads show high GPU util even when compute throughput is low.

**Conclusion:** 80 ms ITL at 89% utilization is **consistent** and expected. The model is deeply memory-bandwidth-bound (arithmetic intensity ≈ 1 FLOP/byte), so 89% SM utilization means the HBM bandwidth is nearly saturated — exactly the correct state for efficient bandwidth-bound inference.

---

### Question 4
**PromQL alert for P99 TTFT > 2 seconds (5-minute window):**

```promql
histogram_quantile(
  0.99,
  rate(vllm:e2e_request_latency_seconds_bucket{phase="prefill"}[5m])
) > 2
```

Or using the TTFT metric directly if exposed:
```promql
histogram_quantile(
  0.99,
  rate(vllm:time_to_first_token_seconds_bucket[5m])
) > 2
```

**Alert rule in YAML:**
```yaml
- alert: HighP99TTFT
  expr: |
    histogram_quantile(0.99,
      rate(vllm:time_to_first_token_seconds_bucket[5m])
    ) > 2
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "P99 TTFT exceeds 2 s"
    description: "5-minute P99 TTFT is {{ $value }}s — check scheduler queue depth and chunked-prefill config."
```

The `for: 2m` clause prevents alerting on momentary spikes (one large request). The 5-minute rate window smooths noisy bucket increments.

---

### Question 5
**Trace breakdown:**
- HTTP gateway: 2 ms
- Scheduler queue: **450 ms** ← large
- Prefill: 620 ms
- Decode (120 tokens): **4,800 ms** ← largest
- Detokenize: 3 ms
- **Total: ~5,875 ms**

**Bottleneck identification:**
The decode phase takes 4,800 ms for 120 tokens = 40 ms/token ITL.

For context: on an A100 with a typical 7–13B model, 40 ms ITL at batch=1 is 2–4× slower than expected (expected ~10–20 ms). This suggests either:
- A **large batch** inflating decode time per token (the request waited 450 ms in queue because the batch was full → decode step is now processing many concurrent sequences → ITL for this request is high due to context-switching overhead), or
- A **memory bandwidth bottleneck** from a very large model with insufficient HBM.

The 450 ms scheduler queue time indicates the system was under significant load — the request was blocked waiting for KV blocks to free.

**One configuration change:** Reduce `--max-num-seqs` from its current value (or set it explicitly to, e.g., 64) to reduce the number of concurrent decode sequences, which will lower per-request ITL at the cost of slightly reduced throughput. Alternatively, enable `--enable-chunked-prefill` to allow the 450 ms queued request to interleave with decode steps rather than waiting for the entire batch to finish prefill.

