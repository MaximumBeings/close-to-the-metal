# Chapter 40 — The vLLM V1 Architecture: Disaggregated Scheduler and Zero-Copy Engine

> *"The fastest code is the code that never runs on the CPU while the GPU is waiting."*

---

## Prerequisites

This chapter builds on:

- **Chapter 7** — The vLLM scheduler: block tables, waiting/running/swapped queues
- **Chapter 8** — Engine startup, worker processes, and CUDA graph capture
- **Chapter 9** — The KV cache allocator and PagedAttention mechanics

---

## What You Will Understand After This Chapter

- Why the V0 monolithic engine architecture hit a throughput ceiling at scale
- How the V1 three-process model (APIServer → Scheduler → Workers) decouples CPU and GPU work
- The ZMQ-based message passing protocol that replaces synchronous Python call stacks
- How the new unified KV cache allocator enables automatic cross-request prefix sharing
- Why multi-step scheduling amortizes scheduler overhead and boosts decode throughput
- The CUDA graph changes that allow buffer reuse across the new async pipeline
- Concrete migration steps from V0 API flags to V1 equivalents

---

## 40.1 Why V0 Hit Limits — The Monolithic Engine Design

### The Single-Process Architecture

The vLLM V0 engine was architected around a single `LLMEngine` object that lived in one Python process and coordinated everything: HTTP request intake, scheduling decisions, tokenization, tensor construction, GPU dispatch, and response streaming.

```
V0 Engine Architecture (Monolithic)
====================================

  HTTP Request
      │
      ▼
 ┌────────────────────────────────────────────────┐
 │              LLMEngine (single process)        │
 │                                                │
 │  add_request()                                 │
 │      │                                         │
 │  _schedule()           ← CPU-bound, blocks GPU │
 │      │                                         │
 │  _execute_model()      ← launches GPU kernels  │
 │      │                                         │
 │  _process_outputs()    ← CPU-bound, blocks GPU │
 │      │                                         │
 │  return results                                │
 └────────────────────────────────────────────────┘
```

Each `step()` call in V0 follows a strict sequential pattern:

1. Acquire the Python GIL
2. Run `_schedule()` — inspect all waiting/running/swapped queues, decide which blocks to allocate or evict, build `SequenceGroupMetadata` objects
3. Hand metadata to GPU workers via `multiprocessing.Queue` (serialized with pickle)
4. Wait synchronously for GPU output
5. Run `_process_outputs()` — decode token IDs, check stop conditions, build response objects
6. Release GIL

This works correctly, but it has three fundamental performance problems.

### Problem 1 — GIL Contention

Python's Global Interpreter Lock means that only one thread can execute Python bytecode at a time. The scheduler loop, the HTTP request parser, and the output processor all compete for the same GIL. Under high request concurrency:

- The asyncio event loop cannot dispatch new HTTP responses while `_process_outputs()` holds the GIL
- Tokenization of incoming requests is delayed until the step loop yields
- Streaming token delivery stutters under load because the scheduler step blocks the HTTP handler

[DEEP DIVE] Profiling V0 under load with `py-spy` reveals that 15–25% of wall-clock time is spent in the scheduler and output processing on the CPU — time during which the GPU is idle, waiting for the next kernel launch.

### Problem 2 — Synchronous CPU-GPU Round-Trip

Every `step()` ends with a `torch.cuda.synchronize()` or equivalent to harvest output token IDs from GPU memory back to CPU. This synchronization point serializes the entire pipeline:

```
V0 Step Timeline
=================

 CPU:  [schedule][pack tensors][wait─────────────][process outputs]
 GPU:              [─────────────forward pass─────]
                              ▲                   ▲
                              │                   │
                        kernel launch       sync + harvest
```

The gap before "kernel launch" is pure CPU scheduling overhead. Under large batch sizes this becomes negligible, but at low-to-medium load (the common production regime for latency-sensitive workloads) it represents a meaningful fraction of TTFT.

### Problem 3 — Head-of-Line Blocking in Prefix Cache

V0's prefix caching (`--enable-prefix-caching`) was an opt-in feature built on top of the original block allocator. The cache lookup was linear-scan based for some workloads and required the scheduler to explicitly track which blocks were "reusable" vs "fresh." Under concurrent requests with similar prefixes (e.g., a system prompt shared across all users), the cache lookup serialized against the allocator lock.

The V0 engine was not wrong. It served billions of tokens in production. But by 2024, the community was hitting a ceiling: to push past ~2000 requests/second on an 8×H100 pod, the CPU-side overhead of the monolithic engine became the bottleneck, not the GPU compute.

[COMMON TRAP] Scaling to more GPU workers does not solve the V0 scheduling bottleneck. Adding tensor-parallel shards or pipeline-parallel stages only increases the per-step GPU utilization — the scheduler is still single-threaded and runs on the same CPU.

---

## 40.2 The V1 Architecture Overview — Three-Process Model

### Separation of Concerns

The core insight of V1 is to split the monolithic engine into three independent processes connected by asynchronous ZMQ sockets:

```
V1 Three-Process Architecture
==============================

  ┌──────────────────────────────────────────────────────────┐
  │                     API Server Process                   │
  │  (asyncio, HTTP, tokenization, response streaming)       │
  │                                                          │
  │   AsyncLLMEngine client interface                        │
  │           │                                              │
  │           │  ZMQ PUSH socket (requests)                  │
  └───────────┼──────────────────────────────────────────────┘
              │
              ▼
  ┌──────────────────────────────────────────────────────────┐
  │                   Scheduler Process (CPU)                │
  │  (EngineCore: scheduling, KV block management,           │
  │   prefix cache, multi-step batching)                     │
  │                                                          │
  │   ZMQ PULL ← requests from API server                   │
  │   ZMQ PUSH → EngineCoreRequest to Workers               │
  │   ZMQ PULL ← EngineCoreOutput from Workers              │
  │   ZMQ PUSH → responses to API server                    │
  └──────────────────────────────────────────────────────────┘
              │                      ▲
              │  ZMQ REQ/REP         │
              ▼                      │
  ┌──────────────────────────────────┴───────────────────────┐
  │                   Worker Process(es) (GPU)               │
  │  (model execution, KV cache physical memory,             │
  │   CUDA graphs, attention backends)                       │
  │                                                          │
  │   Worker 0 (rank 0) acts as ZMQ server for scheduler    │
  │   Workers 1..N receive via NCCL broadcast from rank 0   │
  └──────────────────────────────────────────────────────────┘
```

The key properties of this design:

| Property | V0 | V1 |
|---|---|---|
| Scheduler process | Same as API server | Dedicated CPU process |
| Request intake | Synchronous call | Async ZMQ message |
| CPU-GPU sync | Blocking `synchronize()` | Async result delivery |
| Prefix cache | Opt-in, explicit | Always-on, hash-based |
| Multi-step decode | 1 step per scheduler call | K steps per scheduler call |
| GIL scope | Entire engine | Per-process (smaller scope) |

### ZMQ Socket Topology

V1 uses ZeroMQ sockets because they provide:

- **Backpressure-free message passing** — the scheduler can queue requests without blocking the API server
- **Cross-process zero-copy** for large tensor metadata (via shared memory for the largest payloads)
- **Non-blocking polling** — the scheduler can check for new requests while the GPU is executing

```
ZMQ Socket Map
==============

  API Server                 Scheduler                  Workers
  ──────────                 ─────────                  ───────
  PUSH:5557 ──── requests ──► PULL:5557
                              PUSH:5558 ─── batch ────► PULL:5558
                              PULL:5559 ◄── outputs ─── PUSH:5559
  PULL:5560 ◄── responses ── PUSH:5560
```

[FOUNDATIONAL] ZMQ sockets are asynchronous: the PUSH end never blocks waiting for the PULL end to receive. This means the API server can accept new HTTP requests and enqueue them via PUSH without waiting for the scheduler to process them. The scheduler drains its inbox at the start of each scheduling loop iteration.

---

## 40.3 The AsyncLLMEngine Refactor

### From Blocking Call Stack to Async Message Passing

In V0, `AsyncLLMEngine` wrapped `LLMEngine` in a set of asyncio coroutines that ran `engine_step()` in a background thread to avoid blocking the event loop. The result was correct but architecturally awkward: asyncio coroutines calling into synchronous code via `asyncio.get_event_loop().run_in_executor()`.

V1 replaces this with a true async pipeline:

```
V0 Request Flow (blocking)
===========================
  HTTP POST /v1/chat/completions
      │
      ▼ (asyncio task)
  engine.add_request(request_id, prompt, params)
      │
      ▼ (run_in_executor — blocks a thread)
  engine_step()
      │
      ▼ (synchronous GPU round-trip)
  output = engine.step()  ← CPU+GPU, 5–50ms
      │
      ▼
  stream token to client

V1 Request Flow (async)
========================
  HTTP POST /v1/chat/completions
      │
      ▼ (asyncio task)
  engine_client.add_request(request_id, prompt, params)
      │
      ▼ (ZMQ PUSH — non-blocking, ~1μs)
  scheduler receives request at next poll
      │
      ▼ (scheduler process, runs independently)
  batch assembled → GPU worker → output
      │
      ▼ (ZMQ PUSH from scheduler to API server)
  API server receives output, streams to client
```

### The EngineClient Interface

The API server in V1 holds an `EngineClient` — a lightweight proxy that wraps the ZMQ PUSH/PULL sockets. The client exposes the same `add_request()` / `abort_request()` / `get_model_config()` interface as V0's `AsyncLLMEngine`, so callers see no difference.

```python
# Simplified V1 EngineClient interface (conceptual)
class EngineClient:
    def __init__(self, socket_push, socket_pull):
        self._push = socket_push   # sends AddRequestMsg to scheduler
        self._pull = socket_pull   # receives EngineCoreOutput from scheduler
        self._pending: dict[str, asyncio.Queue] = {}

    async def add_request(self, request_id: str, prompt: str,
                          params: SamplingParams) -> AsyncIterator[RequestOutput]:
        msg = AddRequestMsg(request_id=request_id, prompt=prompt, params=params)
        await self._push.send_pyobj(msg)          # non-blocking ZMQ send
        q: asyncio.Queue = asyncio.Queue()
        self._pending[request_id] = q
        async for output in self._drain_queue(q):
            yield output

    async def _recv_loop(self):
        """Background task: receive outputs from scheduler and route to queues."""
        while True:
            output: EngineCoreOutput = await self._pull.recv_pyobj()
            if output.request_id in self._pending:
                await self._pending[output.request_id].put(output)
                if output.finished:
                    del self._pending[output.request_id]
```

[DEEP DIVE] The `_recv_loop` is a single asyncio task that runs concurrently with all HTTP handler coroutines. Because ZMQ's async Python bindings use a file descriptor that integrates with the event loop's selector, this adds zero threads to the API server process. All I/O multiplexing happens inside the existing asyncio event loop.

### Request Lifecycle Through V1

```
V1 Full Request Lifecycle
==========================

 t=0ms   Client sends POST /v1/chat/completions
 t=0.1ms API server: tokenize prompt, create AddRequestMsg
 t=0.1ms ZMQ PUSH to scheduler (non-blocking)
 t=0.5ms Scheduler: receives msg, adds to waiting_queue
 t=0.5ms Scheduler: runs _schedule(), allocates KV blocks
 t=0.6ms Scheduler: sends EngineCoreRequest via ZMQ to Worker 0
 t=0.7ms Worker 0: receives batch, launches CUDA graph replay
 t=5ms   GPU: prefill forward pass completes
 t=5.1ms Worker 0: harvests first token, sends EngineCoreOutput to scheduler
 t=5.2ms Scheduler: routes output to API server via ZMQ
 t=5.3ms API server: decodes token, streams SSE chunk to client
 ...     (decode steps repeat every ~20ms per K-step batch)
 t=Nms   Final token + [EOS] → client receives finish_reason=stop
```

---

## 40.4 The New KV Cache Manager

### Unified Paged Allocator

V1 replaces V0's `BlockSpaceManager` with a new `KVCacheManager` that has two key differences:

1. **Hash-based prefix deduplication is always on** — there is no `--enable-prefix-caching` flag in V1; the block allocator stores a content hash with every block and consults a global hash table on every allocation
2. **Blocks are reference-counted** — a block shared between two requests is not freed until both requests drop their reference; the old V0 "copy-on-write" logic is replaced by immutable shared blocks

```
V1 KV Block Structure
======================

  Physical Block
  ┌──────────────────────────────────────────────────────┐
  │ block_id   : int                                     │
  │ ref_count  : int          (0 = free, >0 = in use)   │
  │ content_hash: bytes[32]   (SHA-256 of token IDs)     │
  │ num_tokens : int          (tokens stored, 0..block_size) │
  │ is_full    : bool                                    │
  │ kv_data    : [head, seq, 2, dim] on GPU              │
  └──────────────────────────────────────────────────────┘

  Hash Table (CPU-side)
  ┌─────────────────────────────────────────┐
  │ content_hash → block_id                 │
  │ (evicted blocks removed on LRU policy)  │
  └─────────────────────────────────────────┘
```

### Cross-Request Prefix Sharing

When a new request arrives with prompt tokens `[t₁, t₂, ..., tₙ]`, the allocator:

1. Splits the prompt into fixed-size blocks of `block_size` tokens (typically 16)
2. For each full block, computes its content hash
3. Looks up the hash in the global hash table
4. If found: increments `ref_count`, maps the request's logical block to the existing physical block
5. If not found: allocates a new physical block, fills it during prefill, records the hash

```
Prefix Sharing Example
=======================

  System prompt (shared across all user requests):
  "You are a helpful assistant. Always reply in JSON format."
  Tokenizes to 14 tokens → fits in one 16-token block

  Block 42: hash=0xAB12... ref_count=1 (first request)

  Second request arrives with same system prompt:
  → hash lookup hits block 42
  → ref_count becomes 2
  → prefill skips block 42 entirely (already computed)
  → TTFT reduced by ~14 tokens worth of attention compute

  Third request:
  → ref_count becomes 3
  → same savings

  First request finishes:
  → ref_count → 2 (block still held by requests 2 and 3)
  → block NOT freed yet

  All requests finish:
  → ref_count → 0
  → block enters LRU eviction pool (not freed immediately)
  → next request with same prefix: hash hit on LRU block → instant reuse
```

[FOUNDATIONAL] The content hash is computed over the token IDs of the block, not the KV tensors. This means the same hash works across different requests regardless of batch position or attention mask — two blocks contain identical KV data if and only if they were computed from the same token IDs in the same model layer.

### Block Hash Deduplication at Scale

For long-context workloads (32K+ tokens) with shared system prompts, V1's always-on prefix caching can reduce prefill compute by 30–60% at steady state. The deduplication operates across the entire in-flight request set simultaneously, not just within a single sequence.

[COMMON TRAP] Block deduplication only works for **full** blocks. The last partial block of a prompt is never deduplicated because its hash changes as tokens are appended. The allocator pins the last partial block as mutable until it fills up, at which point it becomes immutable and hashable. This means you need `block_size` consecutive identical tokens to see sharing — single-token differences near a block boundary suppress sharing for that block.

---

## 40.5 CUDA Graph Changes

### V0 CUDA Graph Approach

V0 captured one CUDA graph per batch size (typically 1, 2, 4, 8, ..., max_batch_size). Each graph had static input buffers (`input_ids`, `positions`, `block_tables`, `slot_mapping`) and static output buffers (logits, sampled token IDs). The graph was replayed by writing new data into the static buffers before each replay.

The problem: the static buffers lived in GPU memory owned by the worker process. To populate them, the scheduler had to serialize the batch metadata and send it to the worker via `multiprocessing.Queue`, which involved pickle serialization (~50–200μs) and a memory copy.

### V1 CUDA Graph Approach

V1 retains per-batch-size graph capture but changes **how batch metadata reaches the GPU**:

```
V0 Buffer Flow
==============

  Scheduler builds:     SequenceGroupMetadata (Python objects)
      │ pickle serialize + send via Queue
      ▼
  Worker receives:      reconstructs Python objects, copies to GPU buffers
      │
      ▼
  CUDA graph replay

  Overhead: pickle (~100μs) + copy (~50μs) = ~150μs per step

V1 Buffer Flow
==============

  Scheduler builds:     EngineCoreRequest (lightweight msg, no KV metadata)
      │ ZMQ send (raw bytes, near-zero-copy)
      ▼
  Worker receives:      compact message, maps directly to pre-allocated shared memory
      │
      ▼
  GPU kernel fills static buffers from shared memory (DMA-style)
      │
      ▼
  CUDA graph replay

  Overhead: ZMQ send (~5μs) + DMA copy (~10μs) = ~15μs per step
```

### Static Buffer Pool

Each batch-size graph maintains a static buffer pool:

```
V1 Static Buffer Pool (per batch-size graph)
=============================================

  GPU Memory Layout:
  ┌──────────────────────┬──────────────────────────────────────┐
  │ input_ids[B, 1]      │ sampled token IDs for decode step    │
  │ positions[B, 1]      │ current sequence position            │
  │ block_tables[B, maxB]│ physical block IDs for each sequence │
  │ slot_mapping[B]      │ where to write new KV for this step  │
  │ query_lens[B]        │ always 1 for decode                  │
  ├──────────────────────┴──────────────────────────────────────┤
  │ logits[B, vocab_size]│ output logits (written by model)     │
  │ sampled_ids[B]       │ output tokens (written by sampler)   │
  └─────────────────────────────────────────────────────────────┘

  B = batch size for this graph
  All buffers allocated once at startup, never reallocated
```

[DEEP DIVE] The V1 worker eliminates the `SequenceGroupMetadata` Python object entirely for decode steps. During decode, the only variable inputs are: `input_ids` (1 token per sequence), `positions` (1 integer per sequence), and `block_tables` (already on GPU from prefill). The scheduler sends a compact 40-byte message per sequence rather than a full Python object. This reduces serialization overhead by ~10×.

---

## 40.6 The Scheduler Loop in V1

### Overview

The V1 scheduler runs in its own process as an infinite loop:

```python
# Simplified V1 scheduler loop (conceptual)
class EngineCore:
    def run_loop(self):
        while True:
            # 1. Drain incoming requests from API server
            self._recv_new_requests()          # ZMQ PULL, non-blocking

            # 2. Run scheduler to build next batch
            batch, outputs = self.scheduler.schedule()

            # 3. If there is work to do, send to workers
            if batch is not None:
                self._send_to_workers(batch)    # ZMQ PUSH

            # 4. Collect outputs from workers (non-blocking)
            worker_outputs = self._recv_from_workers()

            # 5. Post-process: free finished blocks, update queues
            finished_outputs = self.scheduler.update(worker_outputs)

            # 6. Send finished outputs back to API server
            self._send_outputs(finished_outputs)  # ZMQ PUSH
```

### Priority Queues and Preemption

The V1 scheduler maintains three queues:

```
V1 Scheduler Queue Model
=========================

  waiting_queue   PriorityQueue (arrival time or explicit priority)
      │
      │ schedule() pulls from here when KV blocks are available
      ▼
  running_queue   OrderedDict (insertion order, in-progress sequences)
      │
      │ preemption: if running_queue exhausts KV blocks, lowest-priority
      │ sequence is swapped to CPU or aborted
      ▼
  swapped_queue   Deque (sequences with KV evicted to CPU)
      │
      │ when KV blocks become available, swapped sequences are restored
      ▼
  (back to running)
```

### Chunked Prefill with Decode Interleaving

V1's scheduler supports **chunked prefill**: long prompts are processed in chunks rather than in a single large prefill batch. This allows decode requests to be interleaved with prefill work, reducing head-of-line blocking for already-running sequences:

```
V0: Long Prefill Stalls All Decodes
=====================================

  t=0  Prefill: 8192 tokens (large request)
  t=50ms  Decode requests were blocked for 50ms
           ← head-of-line blocking

V1: Chunked Prefill Interleaved with Decode
============================================

  t=0   Prefill chunk 1: 512 tokens + decode batch (16 seqs)
  t=5ms Prefill chunk 2: 512 tokens + decode batch (16 seqs)
  ...
  t=80ms Prefill complete + decode batch running throughout
         ← no single request blocked >5ms
```

The scheduler parameter `--max-num-batched-tokens` controls the maximum tokens across prefill + decode in a single step. A typical V1 production setting is 8192, split between prefill chunks and decode sequences.

[COMMON TRAP] Setting `--max-num-batched-tokens` too low hurts GPU utilization because decode batches are small and the arithmetic intensity drops below the cuBLAS efficiency threshold. Setting it too high increases scheduler latency and TTFT for decode requests. The sweet spot is typically `2 × max_model_len` for the deployment's typical prompt length.

---

## 40.7 Multi-Step Scheduling

### The Scheduler Overhead Problem

Every time the scheduler runs, it pays a fixed overhead:

- Inspect queues: O(running_queue size)
- Allocate KV blocks for new tokens: O(batch size)
- Build and send ZMQ message: ~5–15μs
- Receive and process output: ~5–15μs

For a model producing 50 tokens/second per sequence, this overhead runs 50 times per second per sequence. At batch size 256, that is 256 × 50 = 12,800 scheduler invocations per second — each one paying the fixed overhead.

### K-Step Batching

Multi-step scheduling runs **K decode steps per scheduler invocation**:

```
V0: One Decode Step Per Scheduler Call
========================================

  Scheduler call ─► GPU step ─► output ─► Scheduler call ─► GPU step ...
  │←── ~1ms overhead ──►│       │←── ~1ms overhead ──►│

  At K=1: overhead fraction = 1ms / (1ms + 20ms GPU) = 4.8%

V1 Multi-Step: K=10 Decode Steps Per Scheduler Call
=====================================================

  Scheduler call ─► GPU step₁ ─► step₂ ─► ... ─► step₁₀ ─► output
  │←── ~1ms overhead ──►│←──────── 200ms GPU ────────────►│

  At K=10: overhead fraction = 1ms / (1ms + 200ms GPU) = 0.5%
```

The worker is given a `num_steps=K` parameter. It executes K consecutive forward passes, feeding each step's sampled token as the next step's input, without returning to the scheduler between steps. Only after K steps does it bundle all K outputs and send them back.

```
Multi-Step Worker Loop (conceptual)
=====================================

  receive batch(num_steps=K)
  outputs = []
  for step in range(K):
      token = model.forward(current_input)   # CUDA graph replay
      outputs.append(token)
      current_input = token                   # feed sampled token back
      update_block_tables_for_next_step()     # update slot_mapping
  send_outputs_to_scheduler(outputs)
```

[DEEP DIVE] Multi-step scheduling interacts with stop conditions carefully. If a sequence emits `[EOS]` at step 3 of a K=10 batch, the worker stops advancing that sequence and pads its remaining output slots. The scheduler is notified of the early stop on receiving the bundled outputs. This means multi-step scheduling is exact — no tokens are generated beyond the stop condition.

### Choosing K

The optimal K depends on:

- **Target latency SLO**: Higher K means the scheduler sees outputs less frequently, increasing P99 inter-token latency by up to K × step_time
- **Decode batch size**: At small batch sizes (< 16), the GPU is under-utilized per step; K can be large. At large batch sizes, each step saturates the GPU; K=1 may already be efficient
- **Stop condition frequency**: If many sequences emit EOS within K steps, the actual batch size collapses mid-flight, wasting the reserved KV blocks

Typical production settings: K=5 at batch sizes 16–64, K=2–3 at batch sizes 128+.

---

## 40.8 Performance Comparison: V0 vs V1

### Benchmark Setup

The following numbers are representative of community benchmarks on an 8×H100 80GB server running LLaMA-3 70B at tensor parallelism=8. They are drawn from the vLLM team's public release notes and community benchmarks (late 2024 / early 2025).

### Throughput

```
Throughput Comparison (output tokens/second)
=============================================

  Request Rate    V0 Engine    V1 Engine    Improvement
  ────────────    ─────────    ─────────    ───────────
      10 req/s      1,840        2,150         +17%
      50 req/s      8,200        9,900         +21%
     100 req/s     12,400       16,800         +35%
     200 req/s     14,100       21,200         +50%
     400 req/s     14,800       24,500         +66%

  V1 advantage grows with request rate because:
  - Scheduler overhead is amortized by multi-step
  - Prefix cache hits increase with more requests sharing prompts
  - Async pipeline hides CPU processing behind GPU execution
```

### Latency

```
P50 / P99 TTFT (ms) — 1024-token prompt, LLaMA-3 70B
=======================================================

  Concurrency    V0 P50    V0 P99    V1 P50    V1 P99
  ───────────    ──────    ──────    ──────    ──────
        8           52        98        48        72
       32          180       420       155       280
      128          820      2100       580      1200
      512         3800      9500      2200      4800

  V1 P99 improvement is larger than P50 because:
  - Chunked prefill prevents long prefills from blocking decode
  - Async pipeline reduces queueing jitter
  - Priority-aware scheduling reduces head-of-line blocking
```

### CPU Utilization

```
Scheduler Process CPU Utilization (%)
======================================

  Load Level    V0 (single process)    V1 Scheduler Process
  ──────────    ───────────────────    ────────────────────
  Light              12%                      8%
  Medium             35%                     18%
  Heavy              72%                     31%
  Saturated         100% ← bottleneck        55% (GPU bottleneck first)

  V1: GPU becomes the bottleneck before the scheduler does.
  V0: Scheduler becomes the bottleneck before the GPU does.
```

[FOUNDATIONAL] The key architectural win in V1 is not raw speed improvement on any single operation — it is that the scheduler process saturates at ~55% CPU when the GPU is at 100%, whereas V0's scheduler saturated the CPU at ~72% leaving the GPU at ~80%. V1 lets the GPU be the bottleneck, which is where it should be.

---

## 40.9 Migration Guide

### API-Level Changes

The V1 engine is enabled with `--use-v2-block-manager` being the default (and the flag being deprecated in favor of automatic selection). The primary user-visible changes:

```
Flag Changes: V0 → V1
======================

  V0 Flag                          V1 Equivalent         Notes
  ───────────────────────────────  ──────────────────    ──────────────────────
  --enable-prefix-caching          Always on             Cannot disable in V1
  --disable-sliding-window         Removed               V1 handles all attn types
  --use-v2-block-manager           Default=True          Was opt-in in V0
  --num-gpu-blocks-override N      Kept                  Same semantics
  --swap-space N                   Kept                  CPU swap still available
  --max-num-seqs N                 Kept                  Same semantics
  --max-num-batched-tokens N       Kept, more important  Controls chunked prefill
  --scheduling-policy fcfs         New: fcfs|priority    V1 adds priority support
  --num-scheduler-steps K          New                   Multi-step K value
  --engine-use-ray                 Removed               V1 uses ZMQ only

  New Flags in V1:
  --num-scheduler-steps K          Default 1 (set 5-10 for decode-heavy workloads)
  --scheduling-policy priority     Enable request priority via SamplingParams.priority
  --max-num-partial-prefills N     Concurrent chunked prefill slots
```

### Code Changes for Existing Integrations

If you are using the Python `LLMEngine` or `AsyncLLMEngine` directly:

```python
# V0 pattern (still works in V1 via compatibility shim)
from vllm import AsyncLLMEngine, AsyncEngineArgs
engine_args = AsyncEngineArgs(model="meta-llama/Meta-Llama-3-8B")
engine = AsyncLLMEngine.from_engine_args(engine_args)

# V1: same interface, new backend
# No code changes required for add_request() / generate() calls
# The EngineClient proxy is transparent to callers

# New V1 capability: request priority
from vllm import SamplingParams
params = SamplingParams(temperature=0.8, max_tokens=100)
params.priority = 2   # higher = more urgent (only effective with --scheduling-policy priority)
```

### Behavioral Differences to Watch

1. **Prefix caching always on**: V1 may produce slightly different token counts on repeated identical requests because blocks are reused and not recomputed. The token IDs are identical; KV computation is skipped. This is correct behavior, but it can surprise test suites that assert exact computation paths.

2. **Multi-step outputs arrive in bursts**: If `--num-scheduler-steps=K`, the API server receives K token outputs at once, then there is a gap while the next K steps execute. Streaming SSE clients see K tokens delivered rapidly followed by a pause, rather than a steady drip. For most clients this is invisible, but latency-measurement code may see different patterns.

3. **No per-request CUDA graph bypass**: V0 allowed falling back to eager mode for requests with unusual shapes. V1 uses a fixed graph pool and raises an error for shapes outside the pool. If you have requests larger than `max_model_len`, they fail faster and with a clearer error message.

4. **ZMQ port usage**: V1 binds ZMQ ports on localhost. If your deployment uses strict firewall rules on loopback interfaces, you may need to open ports 5557–5560 (or whatever the configured range is) on `127.0.0.1`.

---

## Worked Example — Full Request Lifecycle Through V1

```
WORKED EXAMPLE: "What is the capital of France?" through V1
============================================================

Setup:
  - LLaMA-3 8B, block_size=16, max_model_len=4096
  - Scheduler running with num_scheduler_steps=5
  - Prefix cache empty (cold start)

Step 1: API Server receives POST
─────────────────────────────────
  Input: {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
  Tokenization:
    system_prompt: [128006, 9125, 128007, ...] → 32 tokens → 2 blocks
    user_message:  [128006, 882, 128007, 3923, 374, 279, 6864, 315, 9822, 30] → 13 tokens
    total prompt:  45 tokens → 3 blocks (2 full, 1 partial with 13 tokens)

  AddRequestMsg built, sent via ZMQ PUSH to scheduler.

Step 2: Scheduler receives request
────────────────────────────────────
  Waiting queue: [(request_id="req-001", priority=0, arrival=0.001)]

  schedule() called:
  - Block 0 (system_prompt, tokens 0-15): hash=0x1234... → cache MISS (cold)
  - Block 1 (system_prompt, tokens 16-31): hash=0x5678... → cache MISS (cold)
  - Block 2 (partial, tokens 32-44): hash=N/A (not full)
  - Allocate 3 physical blocks: IDs [10, 11, 12]
  - Move req-001 to running_queue

Step 3: Prefill on GPU
───────────────────────
  Worker receives EngineCoreRequest:
    num_tokens=45, block_table=[10,11,12], context_len=0

  CUDA graph not used for prefill (variable length → eager mode)
  Forward pass: 45 tokens through all 32 layers
  Output: logit over vocab → argmax → token ID 3014 ("Paris")

  Worker stores:
    Block 10 filled with KV for tokens 0-15: hash recorded=0x1234...
    Block 11 filled with KV for tokens 16-31: hash recorded=0x5678...
    Block 12 partially filled: tokens 32-44 (not hashed yet)

Step 4: Multi-step decode (K=5)
────────────────────────────────
  Scheduler assigns num_steps=5 to req-001
  Worker executes 5 decode steps:

  Decode step 1: input=[3014 "Paris"], position=45
    Output: token 374 (" is")
  Decode step 2: input=[374], position=46
    Output: token 279 (" the")
  Decode step 3: input=[279], position=47
    Output: token 6864 (" capital")
  Decode step 4: input=[6864], position=48
    Output: token 315 (" of")
  Decode step 5: input=[315], position=49
    Output: token 9822 (" France")

  Bundle [3014, 374, 279, 6864, 315, 9822] sent to scheduler

Step 5: Scheduler processes output bundle
──────────────────────────────────────────
  All 6 tokens (1 from prefill + 5 from decode) sent to API server via ZMQ.
  API server streams SSE events to client.
  Sequence continues for next K=5 steps with "." and [EOS].

Step 6: Completion and block recycling
────────────────────────────────────────
  [EOS] detected → req-001 marked finished
  Block 10: ref_count 1→0, enters LRU eviction pool with hash=0x1234...
  Block 11: ref_count 1→0, enters LRU eviction pool with hash=0x5678...
  Block 12: ref_count 1→0, partial block freed immediately (no hash)

  Next request with same system prompt:
    Block 10 hash=0x1234... → cache HIT → no prefill for first 16 tokens
    Block 11 hash=0x5678... → cache HIT → no prefill for tokens 16-31
    → 32 tokens of prefill skipped → ~2ms TTFT savings
```

---

## Chapter Summary

- **V0's monolithic design** bounded throughput at high concurrency because the scheduler, output processor, and HTTP handler all competed for the Python GIL in a single process.

- **V1 decouples** the API server, scheduler, and GPU workers into three independent processes connected by ZMQ asynchronous sockets. Each process runs its own event loop without blocking the others.

- **The new KV cache manager** makes prefix caching always-on via content-hash deduplication. Blocks are reference-counted and shared across requests automatically, reducing prefill compute for repeated prefixes by 30–60%.

- **Multi-step scheduling** runs K decode steps per scheduler invocation, amortizing the fixed per-step overhead by a factor of K. Typical production settings of K=5–10 reduce scheduler CPU by 60–80% at decode-heavy workloads.

- **CUDA graph buffer management** in V1 uses compact ZMQ messages and pre-allocated shared memory, cutting per-step serialization overhead from ~150μs to ~15μs.

- **Chunked prefill** with decode interleaving eliminates head-of-line blocking: long prompts no longer stall running decode sequences.

- **Migration** from V0 to V1 requires minimal code changes. The `AsyncLLMEngine` interface is preserved. New flags (`--num-scheduler-steps`, `--scheduling-policy`) unlock V1-specific capabilities.

---

## Self-Check Questions

1. **[FOUNDATIONAL]** In V0, why does adding more GPU workers (tensor parallelism) not solve the scheduler bottleneck? What specifically limits throughput in the monolithic engine at high concurrency?

2. **[FOUNDATIONAL]** Explain why V1's block hash deduplication only applies to full blocks, not partial ones. Under what conditions does this limitation prevent prefix sharing between two requests with nearly identical prompts?

3. **[DEEP DIVE]** A production deployment uses `--num-scheduler-steps=10`. A user's request emits EOS at decode step 3. What does the V1 worker do for steps 4–10 for that sequence? Does the scheduler over-allocate KV blocks for the remaining steps? How are the unused allocations recovered?

4. **[DEEP DIVE]** The V1 scheduler sends compact messages to workers instead of `SequenceGroupMetadata` Python objects. Why does eliminating Python object serialization matter even though both endpoints are on the same machine? (Hint: consider GIL, memory copies, and object construction overhead.)

5. **[COMMON TRAP]** A developer notices that under V1, identical back-to-back requests sometimes produce SSE token streams with a "burst then pause" pattern rather than a steady drip. They suspect a bug. Is this a bug? What causes this pattern, and under what condition would it disappear?

---

*Next: Chapter 41 — Disaggregated Prefill and Decode: Separating KV Compute from KV Serving Across Machines*


---

## Worked Solutions

### Question 1 (Foundational)
**Why adding more GPU workers (tensor parallelism) doesn't solve the scheduler bottleneck in V0:**

In V0 vLLM, the scheduler runs in the **main Python process** on the CPU. It is responsible for:

1. Receiving new requests from the API server.
2. Allocating KV blocks from the block manager.
3. Building `SequenceGroupMetadata` Python objects for each request in the batch.
4. Serializing these objects and sending them to GPU workers.
5. Collecting outputs from GPU workers.
6. Running the sampler logic.

All of these steps run sequentially in a single Python thread. Adding more GPU workers increases compute parallelism but does NOT parallelize the scheduler. The Python GIL prevents multi-threaded scheduling. At high concurrency (e.g., 500 concurrent sequences), the scheduler's Python overhead (object construction, serialization, output processing) can consume 20-40 ms per scheduling cycle -- approaching or exceeding the actual GPU forward pass time for small models. The GPU workers sit idle waiting for the next batch while the scheduler prepares it.

---

### Question 2 (Foundational)
**Why V1 block hash deduplication only applies to full blocks. When does this prevent prefix sharing?**

**Why only full blocks:**
vLLM's block size is fixed (e.g., 16 tokens). A block is hashed once all 16 token positions are filled. A partial block (fewer than 16 tokens) has incomplete content -- its hash would be based on partial data and would not be stable (future tokens could change the sequence's token IDs within the block). Hashing partial blocks would create false positives (two requests might match on a partial block that later diverges).

**Condition that prevents prefix sharing despite nearly-identical prompts:**
If two prompts differ at token position 14 of the first block (block size=16):

- Prompt A: tokens [t1, t2, ..., t14, X, t16] -> block hash H_A
- Prompt B: tokens [t1, t2, ..., t14, Y, t16] -> block hash H_B (H_A != H_B)

No prefix sharing is possible for any subsequent blocks, even if all tokens after position 14 are identical. The single differing token within the first block causes all blocks to have different hashes.

**Common trap:** Two users with the same system prompt but different unicode normalization (e.g., "é" vs "é") will produce different token IDs, different block hashes, and no prefix sharing -- despite the text appearing identical to humans.

---

### Question 3 (Deep Dive)
**`--num-scheduler-steps=10`. Request emits EOS at decode step 3. What happens for steps 4-10?**

With `--num-scheduler-steps=10`, the scheduler dispatches 10 decode steps to the GPU workers as a single batch operation (a "multi-step" batch), without returning to Python between steps. The GPU workers execute all 10 steps autonomously.

**At step 3 when EOS is generated:**
The GPU worker detects the EOS token in the sampler output. It marks the sequence as finished internally.

**For steps 4-10:**
The GPU worker does NOT generate further tokens for this sequence. The EOS detection causes the sequence to be "padded" with dummy tokens or simply excluded from subsequent attention computations. The other sequences in the batch continue generating.

**KV block over-allocation:**
The scheduler pre-allocated KV blocks for potentially 10 new tokens when it dispatched the multi-step batch. The request only used 3 of the 10 allocated slots. After the worker returns results to the scheduler, the scheduler sees EOS in the output at step 3 and calls the block manager to free the unused blocks (steps 4-10 allocations are revoked).

**No permanent waste:** Block over-allocation is temporary (one scheduling cycle = 10 steps). After the scheduler processes the worker's output, all over-allocated blocks are freed. The freed blocks are immediately available for new requests.

---

### Question 4 (Deep Dive)
**Why eliminating Python object serialization matters even on the same machine:**

In V0, the scheduler builds `SequenceGroupMetadata` objects in Python and "sends" them to workers. Even on the same machine, this involves:

1. **Python object construction overhead:** `SequenceGroupMetadata` is a complex nested Python object. Constructing it for 500 sequences requires thousands of Python attribute assignments. Each assignment involves Python dict lookups, reference counting, and GIL acquisition. For 500 sequences with 10 attributes each: 5,000 Python operations per scheduling step.

2. **Memory copies:** Even with shared memory, Python's pickling (used for cross-process communication) serializes the objects to bytes and deserializes them in the worker process. This is equivalent to a deep copy of all scheduling metadata.

3. **GIL contention:** While the main process serializes metadata, no other Python thread can run (GIL). This blocks the API server from accepting new connections or running sampling logic concurrently.

**V1's improvement:** V1 sends compact typed messages (e.g., a struct with block table indices, token IDs as a numpy array) via ZMQ, bypassing Python object construction. numpy arrays can be sent as zero-copy shared memory buffers. The GIL is held for microseconds (integer and buffer operations) rather than milliseconds (Python object construction).

At 500 req/s throughput, eliminating 5 ms of Python serialization overhead per step increases scheduling throughput by ~5 ms / (5 ms + 15 ms GPU time) = 25%.

---

### Question 5 (Common Trap)
**V1 with `--num-scheduler-steps=10` produces burst-then-pause SSE pattern. Is this a bug?**

**Not a bug.** This is the expected behavior of multi-step scheduling.

**What causes it:**
With `--num-scheduler-steps=10`, the V1 scheduler dispatches 10 decode steps to the GPU workers in a single batch. The GPU workers execute all 10 steps and return 10 tokens simultaneously. The API server receives 10 tokens at once and streams them to the SSE client in rapid succession ("burst").

Then there is a "pause" while the next 10-step batch executes on the GPU. The client sees: 10 tokens in ~5 ms, then 20 ms pause (10 steps x 2 ms/step), then 10 more tokens in ~5 ms.

**Why it disappears:**
With `--num-scheduler-steps=1` (default behavior), each decode step returns 1 token and immediately triggers the next scheduling cycle. The SSE stream delivers tokens at a steady drip rate matching the decode step latency (~20 ms/token for a 70B model).

**Developer trap:** Latency measurement code that measures "time to first token in each burst" will see artificially low ITL for the first token in each burst (it arrives with 9 others). Correct ITL measurement must account for the burst delivery: measure time from end of last token to end of current last token and divide by 10.

