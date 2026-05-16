# Chapter 18: Disaggregated Prefill and Decode

> "Prefill wants the biggest matrix multiplier you can buy. Decode wants the widest memory bus you
> can find. Running both on the same GPU means neither gets what it needs."
>
> — Production systems insight, echoed across every team that hit p99 spikes at scale

---

## The Problem This Chapter Solves

Picture a production cluster serving a RAG application. Seventy percent of requests are short
chat turns — 200 tokens in, 150 tokens out, fast and cheap. The remaining thirty percent are
deep-research queries that feed 32,000-token retrieved documents as context before generating a
2,000-token synthesis. Both request types land on the same pool of GPUs.

On quiet nights everything is fine. At peak load, a handful of 32K prefill operations seize every
GPU simultaneously for 800 ms each. While they run, the hundreds of short decode requests that
arrived during that window pile up. p99 latency spikes to four seconds. Users of the chat
product — who should be getting 150 ms responses — experience the same stall as the research
queries. The two workloads have poisoned each other.

This chapter explains why that poisoning happens at the hardware level, and how disaggregated
serving solves it by routing prefill and decode to separate, hardware-optimized worker pools
connected by an RDMA KV-transfer fabric.

---

## 18.1 Two Bound Regimes, One GPU

`[FOUNDATIONAL]`

To understand why mixed prefill/decode traffic on a single GPU is a problem, you need to remember
the roofline model from Chapter 17. Every GPU kernel is either:

- **Compute-bound**: the bottleneck is FLOP/s — the tensor cores are always busy, memory keeps up.
- **Memory-bandwidth-bound**: the bottleneck is GB/s — the tensor cores are starved waiting for
  weight bytes to arrive from HBM.

Prefill and decode fall squarely into opposite categories.

```
                     Prefill (large batch of new tokens)
                     ───────────────────────────────────
Arithmetic intensity = 2P / (2P + KV_bytes)   (P = param bytes)

  At batch=1, 512 tokens, 8B model:
    KV generated  = 512 × 32 layers × (128+128) × 2 bytes = 16 MB
    Weight bytes  = 16 GB
    AI ≈ 2×16G / (2×16G + 0.016G) ≈ 1.0 flop/byte  → bandwidth bound

  At batch=512, 512 tokens, 8B model (full prefill batch):
    Effective AI  ≈ 512 flop/byte                   → compute bound

                     Decode (one new token, many sequences)
                     ──────────────────────────────────────
  At batch=32, generating 1 token:
    FLOPs per step ≈ 2P = 32 GB = 16 GFLOP (BF16, B=1 per seq)
    Weight traffic ≈ 16 GB   (load weights once per token, every sequence)
    AI ≈ 1 flop/byte          → always memory-bandwidth bound
```

The key difference is that decode **never escapes memory-bandwidth bound** no matter how many
concurrent users you add, because adding more sequences just proportionally adds more KV cache
reads alongside the constant weight reads. Prefill **does escape** it — once the effective batch
times the token count exceeds the ridge point, you're limited by FLOP/s.

### Hardware Optima Diverge

| Regime  | Bottleneck   | Want in hardware           | Best GPU          |
|---------|--------------|----------------------------|-------------------|
| Prefill | FLOP/s       | High tensor core throughput | H100 SXM 80GB     |
| Decode  | Mem BW (GB/s)| Wide HBM bus, large SRAM   | H100 NVL 94GB     |

The H100 SXM delivers 989 TFLOP/s BF16 and 3.35 TB/s. The H100 NVL pairs two chips with 7.8 TB/s
aggregate bandwidth. If you had infinite budget you would route all prefill to SXM cards and all
decode to NVL cards. Disaggregated serving does exactly this, in software, on any mix of hardware.

---

## 18.2 The Mixed-Traffic Failure Mode

`[COMMON TRAP]`

The failure mode is subtle because it only appears under load and only when prefill and decode
requests interleave unfavourably. Here is the sequence of events that produces a 4-second p99
spike from what should be a 150 ms workload:

```
Timeline (each box = 100 ms, one GPU, shared scheduler)

t=0   ┌─────────────────────────────────────────────────────┐
      │  Decode batch — 64 short sequences, 1 new token each│
      └─────────────────────────────────────────────────────┘
t=100 ┌───────────────────────────────────────┐
      │  Prefill — user A, 32K context (800ms)│
      │                                       │
      │  (All 64 decode sequences WAIT here)  │
      │                                       │
      │                                       │
      │                                       │
      │                                       │
t=900 └───────────────────────────────────────┘
      ┌─────────────────────────────────────────────────────┐
      │  Decode batch resumes — 64 sequences, each 800ms late│
      └─────────────────────────────────────────────────────┘
```

Because vLLM's scheduler runs prefill before decode by default (it must — you cannot generate
without first processing the prompt), a long prefill occupies the GPU as an atomic unit.
Chunked prefill (Chapter 11) helps by splitting the 32K prompt into 2,048-token chunks so decode
can interleave, but chunking adds round-trips and hurts the compute-bound prefill efficiency.
There is no knob setting that fully resolves the tension — you are trading prefill throughput for
decode latency on the same hardware.

The case study numbers make the stakes concrete: before disaggregation, p99 TTFT was 4,100 ms
during the RAG peak window. After disaggregation, p99 fell to 325 ms — a 12.6× improvement —
while total request throughput rose 7.4× because prefill workers ran at full computational
efficiency and decode workers were never starved.

---

## 18.3 Disaggregated Architecture

`[DEEP DIVE]`

Disaggregated prefill-decode (also called PD disaggregation or PD separation) splits the serving
cluster into two distinct pools connected by an RDMA-capable fabric.

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                      Disaggregated Serving Cluster               │
  │                                                                  │
  │  ┌──────────────────────┐       KV Transfer Fabric               │
  │  │   Prefill Pool       │       (RDMA / NVLink-C2C)              │
  │  │  (compute-optimized) │                                        │
  │  │                      │  KV blocks                             │
  │  │  GPU0  GPU1  GPU2    ├────────────────────────────────►       │
  │  │  H100  H100  H100    │  shape: [layers, 2, heads, d_head]     │
  │  │  SXM   SXM   SXM    │  dtype: BF16 / FP8                     │
  │  │                      │  ~0.5 MB per 1K context tokens (8B)    │
  │  │  Receives: prompts   │                                        │
  │  │  Outputs: KV blocks  │              │                         │
  │  └──────────────────────┘              │                         │
  │                                        ▼                         │
  │                              ┌──────────────────────────┐        │
  │                              │   Global KV Store        │        │
  │                              │   (optional, cross-node) │        │
  │                              │   Prefix cache shared    │        │
  │                              │   across decode workers  │        │
  │                              └──────────────┬───────────┘        │
  │                                             │                    │
  │                                             ▼                    │
  │  ┌──────────────────────────────────────────────────────────┐    │
  │  │   Decode Pool  (memory-bandwidth-optimized)              │    │
  │  │                                                          │    │
  │  │   GPU0     GPU1     GPU2     GPU3     GPU4     GPU5      │    │
  │  │   H100NVL  H100NVL  H100NVL  H100NVL  H100NVL  H100NVL  │    │
  │  │                                                          │    │
  │  │   Receives: KV blocks + token ids                        │    │
  │  │   Outputs:  streaming tokens to user                     │    │
  │  └──────────────────────────────────────────────────────────┘    │
  │                                                                  │
  │  Load Balancer / Router                                          │
  │    - Routes prompts → prefill worker (least-loaded)             │
  │    - Tracks KV block location → routes decode → same worker     │
  │    - OR transfers KV blocks to target decode worker             │
  └──────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle Under Disaggregation

1. **Arrival**: request arrives at the load balancer with a 32K token prompt.
2. **Prefill dispatch**: load balancer selects the least-loaded prefill worker and sends the full
   prompt.
3. **Prefill execution**: the prefill GPU runs a single large forward pass over all 32K tokens.
   This is compute-bound. The GPU's tensor cores run at near-peak utilization. TTFT on the prefill
   side is the only latency the user experiences before the first token.
4. **KV serialization and transfer**: the prefill worker serialises the KV cache tensor for this
   sequence — 32 layers × 2 (K+V) × 8 heads × 128 d_head × BF16 × 32K tokens ≈ 512 MB — and
   streams it over RDMA to the designated decode worker.
5. **Decode execution**: the decode worker receives the KV blocks, populates its paged block table,
   and begins generating. It never runs a prefill step. Its GPU is permanently in the
   memory-bandwidth-bound decode regime, which is the best possible use of its wide HBM bus.
6. **Token streaming**: as each token is generated it is forwarded from the decode worker back
   through the router to the user.

### KV Transfer Network Requirements

The KV transfer is the critical path that determines whether disaggregation helps or hurts. If
the transfer latency is larger than the decode latency it would have saved, you have made things
worse.

```
  Transfer latency = KV_bytes / network_bandwidth + 2 × RTT

  For 8B model, 4K context, BF16:
    KV_bytes = 32 layers × 2 × 8 heads × 128 × 4096 tokens × 2 bytes
             = 32 × 2 × 8 × 128 × 4096 × 2 = 536,870,912 bytes ≈ 512 MB

    InfiniBand HDR (200 Gb/s ≈ 25 GB/s):
      Transfer time = 512 MB / 25 GB/s = 20 ms

    NVLink C2C (900 GB/s):
      Transfer time = 512 MB / 900 GB/s = 0.57 ms

  Decode step time at batch=1, H100:
    1000 ms / 209 tok/s = 4.8 ms/token

  Break-even:
    Transfer must complete before N decode steps, where N makes
    the wait worth it. At 20ms / 4.8ms = 4.2 tokens, even HDR
    pays off for any generation longer than ~5 tokens.
```

`[COMMON TRAP]` — **FP8 KV quantization matters for transfer, not just memory**: transferring
512 MB of BF16 KV blocks takes 20 ms over HDR InfiniBand. The same KV in FP8 halves the transfer
to 10 ms. Many teams enable FP8 KV quantization specifically because it cuts both HBM and network
bandwidth requirements, not primarily for the compute gain.

---

## 18.4 Global KV Store: Radically More Effective Capacity

`[DEEP DIVE]`

Disaggregation enables a second optimization that is impossible in the co-located design: a
**global KV store** shared across all decode workers.

In the co-located design, prefix caching (Chapter 11) only reuses KV blocks from earlier requests
that happened to land on the same GPU. If the identical 32K RAG document was prefilled on GPU 0
yesterday and today's request lands on GPU 3, it is prefilled from scratch.

With a global KV store:

```
  ┌───────────────────────────────────────────────────────────┐
  │  Global KV Store  (e.g., Mooncake, distributed DRAM/NVMe) │
  │                                                           │
  │  Key: hash(token_ids[0:N])                                │
  │  Value: KV tensor [layers, 2, heads, d_head, N_tokens]    │
  │                                                           │
  │  ┌──────────────────────────────────────────────────────┐ │
  │  │ "RAG doc A" hash → 512 MB KV tensor  (stored)       │ │
  │  │ "System prompt X" hash → 48 MB KV tensor (stored)   │ │
  │  │ "RAG doc B" hash → 128 MB KV tensor  (stored)       │ │
  │  └──────────────────────────────────────────────────────┘ │
  └───────────────────────────────────────────────────────────┘

  Request arrives: "RAG doc A" + new user question (100 tokens)
    → Router checks global KV store → HIT for "RAG doc A"
    → Transfer 512 MB from store to decode worker (not from prefill GPU)
    → Only 100 tokens need prefill
    → Effective prefill saved: 32K tokens = 800 ms prefill time avoided
```

The effective KV cache capacity is no longer limited by the HBM of any single GPU but by the
aggregate storage of the KV store tier. With commodity NVMe SSDs at ~7 GB/s read bandwidth and
terabytes of capacity, popular document prefixes can be stored at dramatically lower cost than
GPU HBM.

---

## 18.5 vLLM Disaggregated Serving Configuration

`[FOUNDATIONAL]`

vLLM implements disaggregated serving through a `kv_transfer_config` mechanism with pluggable
KV connector backends. As of vLLM 0.6+, the primary production connector is **Mooncake** (from
the Moonshot AI paper that demonstrated the 7.4× throughput improvement). The NIXL connector
(NVIDIA Inference Transfer Library) is the GPU-side complement for NVLink and InfiniBand.

### Prefill Worker Configuration

```yaml
# prefill_worker.yaml
model: meta-llama/Llama-3-8B-Instruct
max_model_len: 32768
max_num_seqs: 8                     # small — prefill is compute bound
gpu_memory_utilization: 0.85

kv_transfer_config:
  kv_connector: MooncakeConnector   # or NixlConnector for NVLink
  kv_role: kv_producer              # this worker generates KV
  kv_rank: 0                        # identifies this worker in the group
  kv_parallel_size: 1
  kv_buffer_device: cuda
  kv_buffer_size: 2e9               # 2 GB staging buffer
```

### Decode Worker Configuration

```yaml
# decode_worker.yaml
model: meta-llama/Llama-3-8B-Instruct
max_model_len: 32768
max_num_seqs: 128                   # large — decode is bandwidth bound
gpu_memory_utilization: 0.90

kv_transfer_config:
  kv_connector: MooncakeConnector
  kv_role: kv_consumer              # this worker receives KV
  kv_rank: 1
  kv_parallel_size: 1
  kv_buffer_device: cuda
  kv_buffer_size: 4e9               # 4 GB staging buffer (larger — many concurrent recvs)
```

### Python API Launch Pattern

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs

# On the prefill node
prefill_args = AsyncEngineArgs(
    model="meta-llama/Llama-3-8B-Instruct",
    max_model_len=32768,
    max_num_seqs=8,
    kv_transfer_config={
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_producer",
        "kv_rank": 0,
    },
)

# On the decode node
decode_args = AsyncEngineArgs(
    model="meta-llama/Llama-3-8B-Instruct",
    max_model_len=32768,
    max_num_seqs=128,
    kv_transfer_config={
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_consumer",
        "kv_rank": 1,
    },
)
```

`[COMMON TRAP]` — **The model must be identical on both workers**: the KV blocks transferred from
the prefill worker assume the same attention head layout, dtype, and quantization as the decode
worker. If you run FP8 KV on the prefill side you must also run FP8 KV on the decode side. Any
mismatch causes silent numerical errors, not a crash.

`[COMMON TRAP]` — **`max_num_seqs` asymmetry is intentional**: prefill workers should run small
batch sizes (4–16) to keep individual request latency low. Decode workers should run large batch
sizes (64–256) to saturate memory bandwidth. Setting them equal on both pools wastes the
architecture.

---

## 18.6 Sizing the Disaggregated Cluster

`[DEEP DIVE]`

The ratio of prefill to decode GPUs depends on the workload's compute-to-generation ratio:

```
  Prefill cost  = n_prompt_tokens × time_per_prefill_token
  Decode cost   = n_output_tokens × time_per_decode_token

  For a RAG query:  2,000 prompt tokens, 500 output tokens
    prefill_time = 2000 / 5000 tok/s = 0.40 s  (H100 SXM, batch=1)
    decode_time  = 500  /  209 tok/s = 2.39 s  (H100 SXM, batch=1)
    ratio = decode_time / prefill_time = 5.97 ≈ 6:1

  → For every prefill GPU you need ~6 decode GPUs to keep them balanced.

  For a short chat turn:  200 prompt tokens, 150 output tokens
    prefill_time = 200 / 5000 = 0.04 s
    decode_time  = 150 / 209  = 0.72 s
    ratio = 18:1

  → Short-output workloads are even more decode-heavy.
  → Mixed workloads: compute a weighted average over your request distribution.
```

The correct cluster ratio is the primary sizing decision in a disaggregated deployment. Teams that
deploy 1:1 prefill/decode ratios waste most of their prefill capacity. The Mooncake paper reports
operating at roughly 1:4 to 1:8 ratios for typical LLM chat workloads.

### Auto-scaling Considerations

Because prefill and decode are now separate deployment units, they can autoscale independently:

- **Prefill autoscaling** triggers on: high request queue depth, rising TTFT p99.
- **Decode autoscaling** triggers on: high active sequences count, rising ITL p99.

This means a sudden burst of long-context RAG queries (which stress prefill) no longer forces you
to scale the entire serving cluster — you scale only the prefill pool.

---

## 18.7 llama.cpp: The Single-Context Design and Why It Fits

`[FOUNDATIONAL]`

llama.cpp has no disaggregated serving mode, and this is a deliberate architectural choice, not
a missing feature.

The llama.cpp execution model is:

```
  Single process → loads model weights once → processes one context → generates tokens
  
  llama_context is the unit of execution:
    - owns KV cache: ggml_tensor of shape [n_layers, 2, n_kv_heads, n_ctx, d_head]
    - owns compute graph: ggml_cgraph rebuilt for each batch
    - single-threaded decode loop: one sequence at a time (or one batch)
```

There is no network layer between prefill and decode because the KV cache lives in the same
process memory that the decode loop reads from. Moving KV blocks to another machine would require
serialising the entire context state, transmitting it, and re-deserialising it — and at that point
you have built a distributed system, which is exactly what llama.cpp deliberately avoids.

This is a **feature for edge and embedded deployment**:

- No network dependency — works fully offline.
- Deterministic latency — no KV transfer jitter.
- Single binary, single process — easy to audit, easy to deploy.
- Lower total memory — no staging buffers, no RDMA queues.

The design trade-off is that llama.cpp cannot serve hundreds of concurrent users efficiently. A
single llama.cpp instance is optimized for one user at a time (or a small batch). The moment you
need to serve more than ~8 concurrent long-context users with sub-second latency guarantees,
llama.cpp's architecture forces you toward vLLM.

```
  When to use llama.cpp:
    ✓  Laptop / desktop inference, single user
    ✓  Edge device, offline, network-constrained
    ✓  Automated pipelines: batch processing, not real-time
    ✓  Prototyping: fast local iteration before production
    ✓  < 4 concurrent users, context < 4K tokens

  When to use vLLM (disaggregated):
    ✓  > 16 concurrent users
    ✓  Mixed short/long context traffic
    ✓  p99 latency SLAs < 500 ms
    ✓  RAG workloads with 4K–128K context
    ✓  Revenue-generating API endpoints
```

---

## 18.8 KV Transfer Mechanics: What Actually Crosses the Wire

`[DEEP DIVE]`

Understanding the KV tensor layout helps you reason about transfer costs and potential bugs.

### Tensor Shape

After a prefill over `T` tokens, the KV cache for a single layer is:

```
  K_cache shape: [n_kv_heads, T, d_head]   dtype: BF16 or FP8
  V_cache shape: [n_kv_heads, T, d_head]   dtype: BF16 or FP8

  Combined per layer: 2 × n_kv_heads × T × d_head × bytes_per_element

  For Llama-3-8B (GQA: n_kv_heads=8, d_head=128, BF16):
    Per layer, 1K tokens: 2 × 8 × 1024 × 128 × 2 = 4,194,304 bytes = 4 MB
    Per layer, 32K tokens:                                          = 128 MB
    All 32 layers, 32K tokens:                                      = 4,096 MB ≈ 4 GB
```

This is the raw tensor. In practice, vLLM pages it into `block_size`-token blocks (default 16
tokens per block). The transfer sends only the populated blocks, not the entire pre-allocated
pool.

### serialization Format

The KV serialiser must preserve:

1. **Block indices**: which slot in the decode worker's block table should receive each block.
2. **Layer order**: blocks must be assigned to the correct layer on the receiving side.
3. **Dtype fidelity**: no silent upcasting — if sender is FP8, receiver must accept FP8.

The Mooncake connector uses a zero-copy RDMA path: the KV tensor's GPU memory is registered with
the RDMA NIC (via GPUDirect RDMA), and the NIC DMA-reads the data directly from GPU HBM without
CPU involvement. This is what makes the 20 ms transfer possible for 512 MB — a CPU-involved
cudaMemcpy path would take 2–5× longer.

---

## 18.9 Common Traps in Disaggregated Deployments

`[COMMON TRAP]`

**Trap 1: Forgetting tensor parallel shapes**

If the prefill pool uses `tensor_parallel_size=2`, the KV blocks are sharded across 2 GPUs.
The decode pool must use the same TP degree, or the KV shard received by one GPU will not
correspond to the heads that GPU is responsible for. Mixed TP degrees are not supported.

**Trap 2: Assuming the global KV store is always faster**

A global KV store read requires a network round-trip. For a 4K context hit over InfiniBand HDR:
8 MB transfer at 25 GB/s = 0.32 ms plus ~0.1 ms RTT = ~0.4 ms. This is faster than a fresh
prefill (4K / 5000 tok/s = 0.8 ms). But for a 512-token context, the fresh prefill takes only
0.1 ms — slower than the network RTT. Always check whether your typical context length is long
enough to make global KV lookup worthwhile.

**Trap 3: Buffer exhaustion causing head-of-line blocking**

The KV staging buffer on the prefill worker (`kv_buffer_size`) must hold all in-flight transfers
simultaneously. If you set it too small and 10 concurrent 512 MB transfers are in flight,
transfers queue behind each other, introducing exactly the kind of stall you were trying to
eliminate. Size the buffer to: `max_concurrent_prefills × avg_kv_size × 2`.

**Trap 4: Routing the second and subsequent requests of a multi-turn conversation**

Turn 1: prefilled on prefill-GPU-0, KV transferred to decode-GPU-3.
Turn 2 arrives: the prompt now includes the generated output from turn 1 (already in
decode-GPU-3's KV cache). If the router sends turn 2 to prefill-GPU-1 for a fresh prefill, the
existing decode-GPU-3 KV cache is wasted and the user's history is prefilled from scratch.
The router must track which decode worker holds the KV state for each conversation and either
route the new prefill to the same pair, or transfer the existing KV to the new prefill worker
before processing.

---

## 18.10 The Full Picture: Latency Budget Under Disaggregation

`[DEEP DIVE]`

With disaggregation working correctly, the user-visible latency breaks down as:

```
  TTFT = prefill_time + KV_transfer_time + (one decode step on decode worker)

  For 32K context, H100 SXM prefill, HDR InfiniBand:
    prefill_time      = 32,000 / 12,500 tok/s = 2,560 ms   (large batch, compute bound)
    KV_transfer_time  = 4,096 MB / 25 GB/s    =   164 ms
    first_decode_step = 1 / 209 tok/s          =   4.8 ms
    ──────────────────────────────────────────────────────
    TTFT              ≈                          2,729 ms

  Compare to co-located (32K, same GPU):
    prefill_time (batch=1) = 32,000 / 5,000    = 6,400 ms
    TTFT ≈ 6,400 ms

  Disaggregation gain on TTFT: 6,400 ms → 2,729 ms = 2.3× improvement
  (higher batch on prefill pool is the key — it uses compute-bound efficiency)

  ITL after first token (decode pool only):
    Disaggregated:  4.8 ms  (decode worker never runs prefill)
    Co-located:     4.8 ms (if no new prefill lands) or 800 ms (if 32K prefill preempts)
    p99 improvement: eliminates 800 ms ITL spikes entirely
```

The p99 ITL improvement is the most commercially valuable gain — it makes the tail latency
predictable. Users experience consistent 4.8 ms per token rather than occasional 800 ms pauses.

---

## Summary

Disaggregated prefill and decode addresses the fundamental hardware incompatibility between two
compute regimes that live inside every LLM inference request. Prefill wants high FLOP/s and runs
best with large batch sizes on compute-dense GPUs. Decode wants high memory bandwidth and runs
best on wide-bus GPUs with many concurrent users. Running both on the same GPU forces a trade-off
that hurts both.

By splitting the cluster into prefill workers and decode workers connected by an RDMA KV-transfer
fabric, production deployments eliminate head-of-line blocking, enable independent autoscaling of
each phase, and unlock a global KV store that extends effective prefix cache capacity to the
aggregate storage of an entire cluster tier. The measured results — 7.4× more requests served,
12.6× tighter p99 — are not an anomaly. They are the consequence of letting each piece of hardware
do the job it was built for.

llama.cpp deliberately opts out of this architecture. Its single-process, single-context design
is optimal for the one-user-at-a-time edge case and a liability for multi-tenant production. That
is not a weakness — it is a scope decision that makes llama.cpp the best tool for its intended
environment.

---

## Key Terms

- **Disaggregated serving** — splitting LLM inference into separate prefill and decode worker
  pools with explicit KV transfer between them.

- **KV transfer fabric** — RDMA or NVLink network over which serialised KV cache tensors move
  from prefill to decode workers.

- **Global KV store** — distributed key-value store that caches KV tensors indexed by token
  sequence hash, enabling cross-worker prefix reuse.

- **kv_producer / kv_consumer** — vLLM roles that designate a worker as a KV sender or receiver.
- **GPUDirect RDMA** — NVIDIA technology that enables RDMA NICs to DMA directly to/from GPU HBM,
  bypassing CPU memory.

- **Prefill/decode ratio** — the number of decode GPUs per prefill GPU required to keep both
  pools saturated for a given workload distribution.

---

*Next: Chapter 19 — Kubernetes, KubeRay, and Auto-Scaling*


---

## Self-Check Questions

1. In a standard vLLM deployment, prefill and decode run on the same GPU. Prefill is compute-bound and decode is memory-bandwidth-bound. Why does mixing them on the same GPU lead to inefficiency? *(Section 18.1)*

2. A disaggregated prefill architecture uses 4 prefill pods and 12 decode pods. The prefill rate is 100 req/s with average prompt length 1 024 tokens. Decode pod ITL target is 40 ms. How do you verify the pod ratio is correct? *(Section 18.3)*

3. After prefill on a prefill pod, the KV cache must be transferred to the decode pod. For a 1 024-token prefill at 32 layers, 32 KV heads, d_k = 128, BF16, compute the transfer size and the transfer time over a 400 Gb/s InfiniBand link. *(Section 18.2)*

4. What happens to in-flight requests when a prefill pod crashes mid-transfer? Describe the retry and recovery procedure in a production system. *(Section 18.4)*

5. Disaggregated prefill adds network latency to TTFT. For the numbers in question 3, compute the fraction of TTFT attributable to KV transfer versus prefill compute (assume prefill compute takes 80 ms). *(Section 18.2)*


---

## Worked Solutions

### Question 1
**Why mixing prefill and decode on the same GPU is inefficient:**

**Prefill is compute-bound.** It processes T tokens in a single forward pass, performing large matrix multiplications (GEMMs) where arithmetic intensity is high (T × d_model FLOPs per weight byte). The GPU's SM cores are the bottleneck; HBM bandwidth is secondary.

**Decode is memory-bandwidth-bound.** It generates 1 token per step, requiring one full read of all model weights from HBM (140 GB for 70B BF16) per step. The arithmetic intensity is ~1 FLOP/byte — far below the hardware's ridge point (~591 FLOP/byte for A100). The GPU's HBM bandwidth is the bottleneck; compute cores are underutilized.

**Mixing penalty:**
1. **Compute-memory conflict:** When a large prefill GEMM runs, it saturates SM cores. Decode operations needing HBM bandwidth are starved because the memory bus is also servicing the prefill's input loads.
2. **KV cache pressure:** A long-context prefill (e.g., 32K tokens) allocates thousands of KV blocks at once, potentially evicting KV blocks that decode sequences needed — forcing costly recomputation.
3. **CUDA kernel serialization:** CUDA kernels from prefill and decode steps are serialized on the same stream; there is no natural pipeline overlap between them on a single GPU.

Disaggregated prefill solves this by dedicating each GPU type to the workload it handles best.

---

### Question 2
**Setup:** 4 prefill pods, 12 decode pods, 100 req/s, avg prompt = 1,024 tokens, decode ITL target = 40 ms.

**Verify the pod ratio:**

**Step 1 — Prefill capacity check.**
Each prefill pod processes prompts at some throughput T_prefill (tokens/s). For a 70B BF16 model at compute-bound prefill on an A100:
- ~1,979 TFLOPS × efficiency / FLOPs per token ≈ 10,000–15,000 tokens/s per pod (estimate).
- At 100 req/s × 1,024 tokens/req = 102,400 tokens/s total prefill demand.
- 4 pods × 12,800 tok/s ≈ 51,200 tok/s capacity — check if this matches actual measurement.

**Step 2 — Decode capacity check.**
Each decode pod serves N concurrent sequences. At ITL target = 40 ms:
```
max concurrent sequences per pod = HBM bandwidth × ITL / (model weight bytes)
= 2,000 GB/s × 0.04 s / 140 GB ≈ 0.57 → batch ~1
```
This suggests decode pods are nearly bandwidth-saturated at batch=1. 12 pods are needed to handle 100 req/s if each pod can only handle ~8–10 concurrent sequences at 40 ms ITL.

**Step 3 — Verify with metrics:**
Monitor `vllm:num_requests_waiting` on decode pods. If decode pods have a backlog growing faster than they drain it, add more decode pods. Monitor prefill pod GPU utilization — if < 70%, consider removing a prefill pod.

---

### Question 3
**KV transfer size for 1,024-token prefill, 32 layers, 32 KV heads, d_k=128, BF16:**

```
size = 1,024 tokens × 32 layers × 2 (K and V) × 32 heads × 128 dim × 2 bytes
     = 1,024 × 32 × 2 × 32 × 128 × 2
     = 1,024 × 32 × 4,096 × 2
     = 1,024 × 262,144
     = 268,435,456 bytes = 256 MB
```

**Transfer time over 400 Gb/s InfiniBand:**
Convert bandwidth: 400 Gb/s = 50 GB/s = 50,000 MB/s
```
t_transfer = 256 MB / 50,000 MB/s = 0.00512 s ≈ 5.12 ms
```

---

### Question 4
**Crash recovery for in-flight requests during KV transfer:**

**What happens at crash:**
The prefill pod completes the forward pass and begins serializing KV blocks into the InfiniBand buffer. Mid-transfer, the pod crashes. The decode pod receives partial KV data.

**The decode pod detects the failure via:**
- Broken TCP/RDMA connection → OS-level signal to the receive side.
- gRPC or ZMQ heartbeat timeout.
- Incomplete transfer checksum verification.

**Retry and recovery procedure:**

1. **Detect:** The decode pod's KV receiver raises a `TransferError` exception within the timeout window (typically 5–10 s).

2. **Discard partial KV:** Any partially written KV blocks for this request are freed and marked invalid in the block table.

3. **Retry at the load balancer:** The load balancer (which tracks request state) sees the decode pod report a `PREFILL_TRANSFER_FAILED` status. It selects a new prefill pod (or the same pod if restarted) and re-dispatches the original prompt.

4. **Re-prefill from scratch:** The new prefill pod runs the forward pass again on the full prompt, producing a fresh KV tensor.

5. **Resume decode:** The KV cache is transferred to the decode pod (same or different), and decoding resumes from the first generated token.

**Time cost:** One full re-prefill latency (e.g., 80 ms for a 1,024-token prompt) + one new KV transfer (5.12 ms). Total additional TTFT penalty ≈ 85 ms.

**Production mitigation:** Use redundant prefill pods with active-active assignment. The load balancer sends the prompt to 2 prefill pods simultaneously; the first to complete transfer wins and the other is cancelled. This reduces retry latency to near-zero in most crash scenarios.

---

### Question 5
**TTFT components from Q3:**
- Prefill compute: 80 ms (given)
- KV transfer: 5.12 ms

**Total TTFT:**
```
TTFT = prefill_compute + KV_transfer = 80 + 5.12 = 85.12 ms
```

**Fraction attributable to KV transfer:**
```
fraction = 5.12 / 85.12 ≈ 6.0%
```

**Interpretation:**
The KV transfer adds a ~6% TTFT overhead for a 1,024-token prompt over 400 Gb/s InfiniBand. This is acceptable for typical use. However, for very short prompts (128 tokens → transfer ≈ 0.64 ms, prefill ≈ 10 ms → TTFT = 10.64 ms, transfer fraction = 6%), the overhead is proportionally similar.

For very long prompts (32K tokens → transfer ≈ 160 ms, prefill ≈ 2,500 ms → TTFT = 2,660 ms, fraction ≈ 6%), the absolute delay grows but fraction stays constant. InfiniBand at 400 Gb/s scales linearly with context length, maintaining the ~6% overhead across most practical context lengths.

