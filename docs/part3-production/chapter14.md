# Chapter 14 — The Eight vLLM Knobs + llama.cpp Equivalents

> *"Every production incident with vLLM can be traced to one of eight numbers.
> Either a number was too large and the server OOM'd, too small and throughput
> collapsed, or two numbers were set without considering each other.
> Know the eight. Know their interactions. Everything else is commentary."*

---

## 14.0 Why This Chapter Matters

Parts I and II built the conceptual foundation.  Now begins **Part III —
Production Configuration**: the engineering discipline of turning those
concepts into a running system that serves real traffic reliably.

Chapter 14 is the entry point.  Every other configuration chapter builds on
the eight parameters introduced here.  If you understand exactly what each
number controls, why it exists, and how it interacts with the others, you
can diagnose any vLLM performance problem from first principles.

**What you will understand after this chapter:**

- What each of the eight vLLM parameters controls, with precise definitions.
- The llama.cpp equivalent for each parameter and where they diverge.
- How to compute safe values from hardware specs and workload requirements.
- The three most common misconfiguration patterns and their symptoms.
- A parameter interaction matrix: which pairs amplify each other's effects.
- Production-ready YAML config templates for three workload archetypes.

**What you need first:**

- Chapter 2 (GPU memory layout — the HBM budget formula).
- Chapter 6 (block pool and KV cache — what block_size controls).
- Chapter 7 (the scheduler — how max_num_seqs and max_num_batched_tokens
  gate admission).

- Chapter 11 (chunked prefill and prefix caching).

---

## 14.1 The Parameter Map  `[FOUNDATIONAL]`

```
THE EIGHT vLLM PARAMETERS
────────────────────────────────────────────────────────────────────────────

┌──────────────────────────────┬──────────────────────────┬───────────────────────────────┐
│ vLLM Parameter               │ What it controls         │ llama.cpp Equivalent          │
├──────────────────────────────┼──────────────────────────┼───────────────────────────────┤
│ max_num_seqs                 │ Max concurrent sequences  │ --parallel (-np)              │
│ max_num_batched_tokens       │ Per-step token budget     │ --batch-size (-b)             │
│ max_model_len                │ Max context length        │ --ctx-size (-c)               │
│ block_size                   │ KV block granularity      │ (fixed; no equivalent)        │
│ gpu_memory_utilization       │ Fraction of HBM claimed   │ --n-gpu-layers (proxy only)   │
│ enable_chunked_prefill       │ Split long prompts        │ --ubatch-size (-ub)           │
│ enable_prefix_caching        │ Radix prefix reuse        │ --cache-prompt                │
│ tensor_parallel_size         │ Multi-GPU TP degree       │ (RPC, experimental)           │
└──────────────────────────────┴──────────────────────────┴───────────────────────────────┘
```

These eight parameters interact through **three shared resources**:

```
Three shared resources that the eight parameters govern:
────────────────────────────────────────────────────────

1. HBM (GPU memory)
   Governed by: gpu_memory_utilization, max_model_len, block_size,
                tensor_parallel_size (shards the weight memory)

2. Token budget (compute per step)
   Governed by: max_num_batched_tokens, max_num_seqs,
                enable_chunked_prefill

3. Scheduler admission
   Governed by: max_num_seqs, max_num_batched_tokens,
                enable_prefix_caching (determines effective KV demand)
```

We examine each parameter in depth, then study their interactions.

---

## 14.2 `max_num_seqs` — Concurrency Cap  `[FOUNDATIONAL]`

### 14.2.1 What it controls

`max_num_seqs` is the maximum number of sequences (requests) the scheduler
will hold in the **running** state simultaneously.  It is the primary knob
for controlling concurrency.

```
vLLM scheduler state at any moment:
  waiting:  requests admitted but not yet started (prompt not prefilled)
  running:  sequences actively being decoded (in the KV cache)
  swapped:  sequences evicted to CPU memory (preempted)

max_num_seqs caps the size of the running set.
```

### 14.2.2 How to choose it

```
WORKED EXAMPLE 14.1 — Choosing max_num_seqs
─────────────────────────────────────────────────────────────────────
Given:
  Model:               LLaMA 3 8B
  GPU:                 A100 80GB
  Weight memory:       16 GB  (BF16)
  Peak activations:    2 GB   (measured by dummy forward pass)
  Available for KV:    80 × 0.90 − 16 − 2 = 54 GB
  max_model_len:       4096 tokens
  KV per token:        2 × 32 layers × 8 heads × 128 dim × 2 bytes
                     = 2 × 32 × 8 × 128 × 2 = 131 072 bytes = 128 KB/token
  KV per sequence:     4096 tokens × 128 KB = 512 MB

Step 1 — Max sequences by KV memory alone:
  54 GB / 512 MB = 108 sequences

Step 2 — Adjust for block fragmentation (~15%):
  108 × 0.85 ≈ 91 sequences

Step 3 — Add safety margin (~10%):
  91 × 0.90 ≈ 82 sequences

Recommended starting value: max_num_seqs = 64
  (round down to a power of 2 for cleaner CUDA graph replay)

Final answer:
  max_num_seqs = 64 for LLaMA 3 8B on A100 80GB with 4K context.
  Setting it higher risks KV cache OOM under peak load.
─────────────────────────────────────────────────────────────────────
```

### 14.2.3 Symptoms of misconfiguration

```
max_num_seqs TOO HIGH:
  Symptom: CUDA out-of-memory error during peak load
           "torch.cuda.OutOfMemoryError: CUDA out of memory"
  Cause:   KV pool exhausted; new blocks cannot be allocated
  Fix:     Reduce max_num_seqs or reduce max_model_len

max_num_seqs TOO LOW:
  Symptom: High request queue depth; throughput well below GPU capacity
           Prometheus: vllm:num_requests_waiting consistently > 0
           GPU utilization: < 60% despite heavy traffic
  Cause:   Scheduler under-fills the batch; GPU is idle waiting for seqs
  Fix:     Increase max_num_seqs (if KV memory permits)
```

### 14.2.4 llama.cpp equivalent: `--parallel` (`-np`)

```
llama-server --parallel 8   # serve up to 8 concurrent users
```

Unlike vLLM, llama.cpp allocates a **fixed context buffer** per slot at
startup.  Setting `--parallel 8` with `--ctx-size 4096` allocates
`8 × 4096 × KV_bytes` of context memory immediately, regardless of how many
users are actually connected.  There is no dynamic KV pool.

---

## 14.3 `max_num_batched_tokens` — Per-Step Token Budget  `[FOUNDATIONAL]`

### 14.3.1 What it controls

`max_num_batched_tokens` sets the maximum number of tokens processed in a
single forward pass — the combined count of all decode tokens and all prefill
chunk tokens in one scheduler step.

```
Per-step token composition:
  decode_tokens   = number of running sequences  (each contributes 1)
  prefill_tokens  = chunk allocated to the current prefill request
  Total           ≤ max_num_batched_tokens
```

### 14.3.2 How to choose it

```
WORKED EXAMPLE 14.2 — Choosing max_num_batched_tokens
─────────────────────────────────────────────────────────────────────
Given:
  max_num_seqs = 64  (from Example 14.1)
  Workload:    RAG with average 4 000-token prompts

Step 1 — Minimum budget to keep decode users supplied:
  Decode tokens per step = 64 × 1 = 64
  → max_num_batched_tokens must be ≥ 64 + some prefill budget

Step 2 — Prefill chunk size goal:
  Target: fill 4 000-token prompt in ≤ 5 scheduler steps
  Chunk size needed: 4000 / 5 = 800 tokens per step

Step 3 — Total budget:
  max_num_batched_tokens = decode (64) + prefill chunk (800) = 864
  → Round to 1024 for alignment

Step 4 — Verify GPU efficiency:
  1024 tokens per step → attention matrix [1024 × KV_len]
  Well above the ~256-token efficiency threshold
  → Tensor cores will be well-utilized

Recommended: max_num_batched_tokens = 4096 (generous for mixed workloads)

Safety rule: max_num_batched_tokens ≥ 4 × max_num_seqs
  4 × 64 = 256 minimum; 4096 gives ample headroom
─────────────────────────────────────────────────────────────────────
```

### 14.3.3 Symptoms of misconfiguration

```
max_num_batched_tokens TOO LOW (< max_num_seqs):
  Symptom: Prefill queue never drains; new requests wait indefinitely
           New user TTFT grows monotonically over time
  Cause:   Entire budget consumed by decode tokens; zero left for prefill
  Fix:     Increase max_num_batched_tokens to at least 4 × max_num_seqs

max_num_batched_tokens TOO HIGH (> GPU memory for activations):
  Symptom: OOM on activation memory during large prefill steps
           (separate from KV cache OOM — happens in forward pass itself)
  Cause:   Activation tensors for N tokens require O(N × d_model) memory
  Fix:     Reduce max_num_batched_tokens or enable chunked prefill
```

### 14.3.4 llama.cpp equivalent: `--batch-size` (`-b`)

```
llama-server --batch-size 4096   # max tokens per llama_decode call
```

In llama.cpp, `--batch-size` is the logical batch — the maximum tokens
submitted in one `llama_decode` call.  Combined with `--ubatch-size` (the
physical micro-batch), it controls both the prefill chunk size and the
peak activation memory during prefill.

---

## 14.4 `max_model_len` — Maximum Context Length  `[FOUNDATIONAL]`

### 14.4.1 What it controls

`max_model_len` sets the maximum number of tokens in any single sequence —
prompt + generated output combined.  It directly determines the maximum
KV blocks allocated per sequence and, therefore, the maximum concurrency
for a given HBM budget.

```
KV memory per sequence = max_model_len × KV_bytes_per_token
                       = max_model_len × 2 × n_layers × n_kv_heads
                                       × head_dim × bytes_per_element
```

Reducing `max_model_len` is the single most effective way to increase
`max_num_seqs` when HBM is the bottleneck.

```
WORKED EXAMPLE 14.3 — max_model_len vs. Concurrency Trade-off
─────────────────────────────────────────────────────────────────────
Given:
  Model: LLaMA 3 8B (n_layers=32, n_kv_heads=8, head_dim=128, BF16)
  Available KV memory: 54 GB (from Example 14.1)
  KV bytes per token: 128 KB (= 2 × 32 × 8 × 128 × 2)

  max_model_len │ KV per seq │ Max seqs (54 GB)
  ──────────────┼────────────┼──────────────────
      512       │    64 MB   │   843
     1024       │   128 MB   │   422
     2048       │   256 MB   │   211
     4096       │   512 MB   │   105
     8192       │     1 GB   │    52
    16384       │     2 GB   │    26
    32768       │     4 GB   │    13
   131072       │    16 GB   │     3

Final answer:
  Halving max_model_len doubles concurrency.
  For a chat workload (average 200-token prompts), max_model_len=2048
  gives 211 concurrent users — far more than max_model_len=8192 (52).
  Use the smallest value that safely covers your p99 context length.
─────────────────────────────────────────────────────────────────────
```

### 14.4.2 Symptoms of misconfiguration

```
max_model_len TOO LARGE:
  Symptom: Low concurrency; high KV memory usage per sequence
           Throughput lower than expected for the hardware
  Fix:     Profile your p99 sequence length; set max_model_len to
           p99 × 1.1 as a safety margin

max_model_len TOO SMALL:
  Symptom: "Request exceeds max_model_len" errors in production
           Long-context requests rejected outright
  Fix:     Increase max_model_len; consider a second deployment with
           a larger context window for long-context traffic
```

### 14.4.3 llama.cpp equivalent: `--ctx-size` (`-c`)

```
llama-server --ctx-size 4096   # context window per slot
```

With `--parallel N`, llama.cpp allocates `N × ctx_size × KV_bytes` at
startup — regardless of actual usage.  There is no demand-driven allocation.
Setting `--ctx-size 131072` with `--parallel 8` on a 24 GB GPU will OOM
immediately.

---

## 14.5 `block_size` — KV Block Granularity  `[FOUNDATIONAL]`

### 14.5.1 What it controls

`block_size` is the number of tokens stored in each KV cache block (Chapter 6).
It is a fine-grained memory management parameter.

```
Block size trade-offs:
─────────────────────────────────────────────────────────────────────

Small block_size (e.g., 8):
  + Low internal fragmentation: last block of a sequence wastes ≤ 7 tokens
  + Finer copy-on-write granularity for beam search
  − More blocks per sequence → larger block table → more metadata overhead
  − More frequent block allocation calls

Large block_size (e.g., 32):
  + Fewer block table entries per sequence
  + Better memory alignment for CUDA memory operations
  − Higher internal fragmentation: last block wastes up to 31 token slots
  − Coarser prefix cache granularity (cache misses are larger)

Default: block_size = 16 (vLLM default)
         Good balance for most workloads.

When to change:
  block_size = 8:  beam search with many short sequences
  block_size = 32: very long sequences (128K+) where metadata overhead matters
```

### 14.5.2 Impact on fragmentation

```
WORKED EXAMPLE 14.4 — Internal Fragmentation
─────────────────────────────────────────────────────────────────────
Given:
  Sequence length:  1000 tokens (prompt + generated)
  block_size:       16

Step 1 — Blocks needed:
  ceil(1000 / 16) = 63 blocks

Step 2 — Tokens in last block:
  1000 mod 16 = 8 tokens used, 8 slots wasted

Step 3 — Fragmentation rate:
  8 / (63 × 16) = 8 / 1008 = 0.8%

At block_size = 32:
  ceil(1000 / 32) = 32 blocks
  1000 mod 32 = 8 slots wasted (by coincidence same here)
  Fragmentation: 8 / (32 × 32) = 0.8%

At very short sequences (20 tokens):
  block_size=16: 2 blocks, 12 wasted → 60% fragmentation!
  block_size=8:  3 blocks,  4 wasted → 20% fragmentation

Final answer:
  Short sequences suffer much more from large block sizes.
  For workloads with average output ≤ 64 tokens, prefer block_size=8.
─────────────────────────────────────────────────────────────────────
```

`[COMMON TRAP]` — `block_size` cannot be changed after the vLLM server
starts.  It is fixed at initialization when the block pool is allocated.
If you change `block_size`, you must restart the server.

---

## 14.6 `gpu_memory_utilization` — HBM Fraction Claimed  `[FOUNDATIONAL]`

### 14.6.1 What it controls

`gpu_memory_utilization` (range: 0.0–1.0) controls what fraction of total
GPU HBM vLLM claims at startup.  After subtracting weights and peak
activation memory, the remainder becomes the KV block pool.

```
KV pool size = total_HBM × gpu_memory_utilization
               − weight_memory
               − peak_activation_memory

If KV pool size < 0: server refuses to start with a clear error.
```

### 14.6.2 How to choose it

```
WORKED EXAMPLE 14.5 — gpu_memory_utilization Sizing
─────────────────────────────────────────────────────────────────────
Given:
  GPU: H100 SXM5 80GB
  Model: LLaMA 3 70B, TP=4 (each GPU holds 17.5B params in BF16 = 35 GB)
  Peak activations (measured): 3 GB per GPU

gpu_memory_utilization = 0.90:
  Claimed: 80 × 0.90 = 72 GB
  Available for KV: 72 − 35 − 3 = 34 GB per GPU
  → Sufficient for ~68 sequences at 4K context

gpu_memory_utilization = 0.95:
  Claimed: 80 × 0.95 = 76 GB
  Available for KV: 76 − 35 − 3 = 38 GB per GPU
  → Sufficient for ~76 sequences at 4K context

gpu_memory_utilization = 0.98:
  Claimed: 80 × 0.98 = 78.4 GB
  Available for KV: 78.4 − 35 − 3 = 40.4 GB per GPU
  → Risk: other GPU processes (NCCL, CUDA driver) need ~1-2 GB
           → may OOM under load if other GPU activity exists

Recommended: 0.90 for shared GPUs; 0.95 for dedicated inference GPUs.
─────────────────────────────────────────────────────────────────────
```

### 14.6.3 Symptoms of misconfiguration

```
gpu_memory_utilization TOO HIGH (e.g., 0.99):
  Symptom: CUDA OOM during high-load periods or NCCL operations
           Other GPU processes (monitoring, CUDA driver) compete for memory
  Fix:     Reduce to 0.90–0.95; leave ≥ 2 GB headroom for the GPU OS

gpu_memory_utilization TOO LOW (e.g., 0.50):
  Symptom: Large fraction of HBM unused; fewer concurrent sequences than
           the GPU can support; low throughput
  Fix:     Increase to 0.90 on dedicated inference machines
```

### 14.6.4 llama.cpp "equivalent": `--n-gpu-layers` (`-ngl`)

llama.cpp does not have a fractional memory utilization flag.  The closest
control is `--n-gpu-layers`, which specifies how many transformer layers
to offload to GPU.  Offloading all layers maximizes GPU utilization;
partial offload splits computation between GPU and CPU.

```
llama-server --n-gpu-layers 32   # offload all 32 layers of an 8B model
llama-server --n-gpu-layers 20   # offload 20 layers; 12 run on CPU
```

This is fundamentally different from `gpu_memory_utilization` — it controls
compute placement, not memory fraction.  Memory is allocated for the layers
that are offloaded; there is no equivalent to the fraction-based KV pool
sizing that vLLM does.

---

## 14.7 `enable_chunked_prefill` — Split Long Prompts  `[FOUNDATIONAL]`

This parameter was covered in depth in Chapter 11.  Here we focus on its
interaction with the other seven parameters.

```
enable_chunked_prefill interactions:
────────────────────────────────────────────────────────────────────

WITH max_num_batched_tokens:
  chunk_size = max_num_batched_tokens − decode_tokens
  → If max_num_batched_tokens is too small, chunk_size → 0 (prefill starves)
  → Rule: max_num_batched_tokens ≥ 4 × max_num_seqs

WITH max_num_seqs:
  decode_tokens = max_num_seqs (in the worst case, all running)
  → More sequences = less budget left for prefill chunks
  → At max_num_seqs=256, max_num_batched_tokens=512:
       chunk budget = 512 − 256 = 256 tokens per step
       A 4096-token prompt takes 4096/256 = 16 steps to prefill

WITH enable_prefix_caching:
  Prefix cache hits reduce effective prompt length to prefill.
  A 5000-token system prompt cached → only user message (~50T) needs prefill.
  → Chunked prefill rarely triggered for cached requests.
  → Enable both together for maximum benefit.

WITH gpu_memory_utilization:
  Larger KV pool → more blocks available → more headroom for interleaved
  prefill and decode without eviction.
  → Higher gpu_memory_utilization reduces preemption under chunked prefill.
```

### 14.7.1 llama.cpp equivalent: `--ubatch-size` (`-ub`)

```
llama-server --ubatch-size 512   # micro-batch size for prefill chunking
```

As covered in Chapter 11, llama.cpp's `--ubatch-size` performs pure
micro-batching (no decode interleaving).  It reduces peak activation memory
at the cost of longer total prefill time — useful for limited-VRAM devices
like RTX 4090 serving large prompts.

---

## 14.8 `enable_prefix_caching` — Radix Prefix Reuse  `[FOUNDATIONAL]`

Also covered in depth in Chapter 11.  Key interactions:

```
enable_prefix_caching interactions:
────────────────────────────────────────────────────────────────────

WITH max_model_len:
  Prefix cache stores blocks for up to max_model_len tokens.
  Long system prompts (10K+) with large max_model_len benefit most.

WITH block_size:
  Prefix cache operates at block granularity.
  Smaller block_size → finer cache keys → higher hit rate for
  prefixes that differ in the last few tokens.
  Larger block_size → coarser keys → occasional missed hits near
  prefix boundaries.

WITH gpu_memory_utilization:
  Prefix cache blocks occupy the same KV pool as active sequences.
  Under heavy load, prefix cache blocks are evicted (LRU) to make room.
  Higher gpu_memory_utilization → larger pool → more cached blocks.
  Monitor: vllm:gpu_prefix_cache_hit_rate (Prometheus)

WITH tensor_parallel_size:
  Prefix caching works correctly with TP > 1.
  Each worker holds its shard of the cached KV blocks.
  Cache hit rate is unaffected by TP degree.
```

### 14.8.1 llama.cpp equivalent: `--cache-prompt`

```
llama-server --cache-prompt   # enable per-session prefix match
```

Single-session linear prefix match only (Chapter 11 §11.6.2).

---

## 14.9 `tensor_parallel_size` — Multi-GPU TP Degree  `[DEEP DIVE]`

### 14.9.1 What it controls

`tensor_parallel_size` (TP) shards the model's weight tensors across
`TP` GPUs, with each GPU holding `1/TP` of each weight matrix.  This allows
serving models that exceed a single GPU's HBM capacity, and reduces the
per-GPU weight memory, leaving more room for the KV cache.

```
Tensor parallelism memory effect:
  Weight memory per GPU = total_weight_memory / TP
  KV cache per GPU      = unchanged (each GPU holds full KV for its shards)

  LLaMA 3 70B in BF16:
    Single GPU: 140 GB weights → requires 2× A100-80GB minimum
    TP=2:        70 GB per GPU → fits on A100-80GB (with KV headroom)
    TP=4:        35 GB per GPU → fits on A100-40GB
```

### 14.9.2 TP communication cost

At the end of each attention and FFN sub-layer, an AllReduce collective
operation synchronizes partial sums across all TP workers.  This adds
latency proportional to the number of parameters in each all-reduce and
the inter-GPU bandwidth.

```
WORKED EXAMPLE 14.6 — AllReduce Latency Cost
─────────────────────────────────────────────────────────────────────
Given:
  Model:       LLaMA 3 70B (n_layers=80)
  TP:          4 GPUs
  AllReduce per layer: 2 (after attention, after FFN)
  Total AllReduces per step: 80 × 2 = 160

  NVLink bandwidth (H100 SXM5): 900 GB/s bidirectional
  AllReduce tensor size per layer: [batch × d_model × 2 bytes]
    At batch=64, d_model=8192: 64 × 8192 × 2 = 1 MB

  AllReduce latency per call ≈ latency_alpha + tensor_bytes / bandwidth
    alpha (latency base):  ~5 μs (NVLink ring)
    1 MB / 900 GB/s     ≈  1.1 μs
    Total per AllReduce: ~6 μs

  Total AllReduce overhead per decode step:
    160 × 6 μs = 960 μs ≈ 1 ms

  Decode step total time: ~25 ms (forward pass)
  Overhead fraction: 1 ms / 25 ms ≈ 4%

  On PCIe (no NVLink): bandwidth ≈ 32 GB/s
    Per AllReduce: 5 μs + 31 μs = 36 μs
    Total: 160 × 36 = 5.8 ms  → 23% overhead → significant

Final answer:
  NVLink: TP overhead ≈ 4% — use TP freely on NVLink systems.
  PCIe:   TP overhead ≈ 23% — minimize TP degree; prefer single-GPU
          or pipeline parallelism if cross-node is required.
─────────────────────────────────────────────────────────────────────
```

### 14.9.3 When to use TP

```
Use tensor_parallel_size > 1 when:
  ✓ Model does not fit on a single GPU (70B at BF16 needs ≥ 2× A100-80GB)
  ✓ Using NVLink-connected GPUs (low AllReduce cost)
  ✓ Minimizing latency is the priority (TP reduces per-step time)

Avoid tensor_parallel_size > 1 when:
  ✗ GPUs are connected only via PCIe (high AllReduce overhead)
  ✗ Model fits comfortably on a single GPU (7B–13B on A100-80GB)
  ✗ Maximising throughput takes priority over latency:
       Pipeline parallelism (PP) is better for throughput at cost of
       higher latency per request
```

### 14.9.4 llama.cpp: RPC (experimental)

llama.cpp's `--rpc` flag enables an experimental multi-machine RPC backend,
but it does not implement true tensor parallelism.  For multi-GPU serving
in production, vLLM is the more mature choice.

---

## 14.10 Parameter Interaction Matrix  `[FOUNDATIONAL]`

The eight parameters are not independent.  The table below shows which
pairs have the strongest interactions and what the effect is.

```
PARAMETER INTERACTION MATRIX
(↑ = increases, ↓ = decreases, ↔ = no direct effect)

Changing →         TTFT    ITL    Throughput   HBM Use   Risk
───────────────────────────────────────────────────────────────────────────────

↑ max_num_seqs
  via throughput:   ↔       ↑↑     ↑↑           ↑↑       OOM if KV pool full
  via decode batch: ↑       ↓       ↑            ↔       More tokens/step

↑ max_num_batched_tokens
  prefill speed:    ↓↓      ↔       ↑            ↔       Activation OOM if huge
  decode padding:   ↔       ↔       ↑            ↔       None

↓ max_model_len
  concurrency:      ↔       ↔       ↑↑           ↓↓      Requests rejected if
                                                          prompt > limit

↓ block_size
  fragmentation:    ↔       ↔       ↑ (small)    ↓       Metadata overhead

↑ gpu_memory_utilization
  pool size:        ↔       ↔       ↑↑           ↑↑      OOM from GPU OS

enable_chunked_prefill
  long prompts:     ↓↓      ↔       ↑            ↔       Prefill starvation
                                                          if tokens budget low

enable_prefix_caching
  repeated prefixes:↓↓↓     ↔       ↑↑           ↑ (pool)Hit rate drops if
                                                          pool too small

↑ tensor_parallel_size
  model fit:        ↓       ↓       ↑ (fit)      ↓/GPU   AllReduce overhead
                                                          on PCIe
```

Key insight from the matrix: **max_num_seqs and gpu_memory_utilization are
tightly coupled**.  Every additional sequence consumes KV blocks from the pool
sized by gpu_memory_utilization.  Increase one without the other and you
either waste capacity or OOM.

---

## 14.11 The Three Most Common Misconfigurations  `[FOUNDATIONAL]`

### 14.11.1 Misconfiguration 1: The OOM Triangle

```
SYMPTOM:
  Server starts fine. Under load (> 70% capacity), random OOM errors.
  Error: "CUDA out of memory" during allocation of new KV blocks.

ROOT CAUSE:
  max_num_seqs set too high for the KV pool sized by gpu_memory_utilization.
  At low load: enough free blocks. At peak: pool exhausted.

DIAGNOSIS:
  Prometheus: vllm:num_preempted_requests > 0  (preemptions before OOM)
  Prometheus: vllm:gpu_cache_usage_perc approaching 1.0 continuously

FIX:
  Option A: Reduce max_num_seqs to match KV pool capacity.
  Option B: Increase gpu_memory_utilization (if headroom exists).
  Option C: Reduce max_model_len to shrink per-sequence KV footprint.

PREVENTION:
  Run the memory budget calculation (Example 14.1) before deployment.
  Always verify: max_num_seqs × KV_per_seq < KV_pool_size × 0.85
```

### 14.11.2 Misconfiguration 2: The Prefill Starvation Loop

```
SYMPTOM:
  New requests join the waiting queue. TTFT for new requests grows
  monotonically over time — 1s, 5s, 30s, never clears.
  Existing decode users receive tokens normally.
  vllm:num_requests_waiting grows without bound.

ROOT CAUSE:
  max_num_batched_tokens ≤ max_num_seqs.
  Decode tokens consume the entire budget; zero tokens left for prefill.

DIAGNOSIS:
  Prometheus: vllm:num_requests_waiting growing
  Prometheus: vllm:num_prefill_tokens_total = 0 or very low

FIX:
  Increase max_num_batched_tokens to at least 4 × max_num_seqs.
  OR reduce max_num_seqs to free up token budget for prefill.

EXAMPLE:
  max_num_seqs = 512
  max_num_batched_tokens = 1024
  → decode_tokens = 512
  → prefill_budget = 1024 - 512 = 512 tokens
  → A 4096-token prompt takes 8 steps to prefill ← marginal but ok

  max_num_batched_tokens = 512
  → decode_tokens = 512
  → prefill_budget = 0  ← STARVATION
```

### 14.11.3 Misconfiguration 3: The Context Cliff

```
SYMPTOM:
  p99 latency spikes when traffic includes prompts > X tokens.
  Requests with prompts ≤ X are fine; requests > X are rejected
  with "Input too long" errors. Users report truncated responses.

ROOT CAUSE:
  max_model_len set to accommodate average-length prompts, not p99.
  Long-tail requests (RAG with large context, chat history) exceed the limit.

DIAGNOSIS:
  Application logs: "Request length N exceeds max_model_len M"
  Prometheus: vllm:num_requests_running drops suddenly when long requests arrive

FIX:
  Profile p99 prompt length from production logs.
  Set max_model_len = p99_prompt_length × 1.1 (10% safety margin).
  If p99 prompt + p99 output > max_model_len: consider two deployments.

TWO-TIER APPROACH:
  Tier 1: max_model_len=4096, max_num_seqs=128  ← short prompts (80% traffic)
  Tier 2: max_model_len=32768, max_num_seqs=8   ← long prompts (20% traffic)
  Route based on prompt length at the gateway layer.
```

---

## 14.12 Production Configuration Templates  `[FOUNDATIONAL]`

Three workload archetypes with recommended configurations.

### 14.12.1 Chat — Short prompts, latency-sensitive

```yaml
# vLLM config: chat workload (7B–13B model, A100-80GB, tight TTFT SLA)
model: meta-llama/Meta-Llama-3-8B-Instruct
dtype: bfloat16
max_model_len: 4096
max_num_seqs: 64
max_num_batched_tokens: 4096
block_size: 16
gpu_memory_utilization: 0.90
enable_chunked_prefill: true
enable_prefix_caching: true
tensor_parallel_size: 1

# llama.cpp equivalent (llama-server):
# --model Meta-Llama-3-8B-Instruct.Q8_0.gguf
# --ctx-size 4096 --parallel 8
# --batch-size 2048 --ubatch-size 512
# --cache-prompt --n-gpu-layers 99
```

### 14.12.2 RAG — Medium prompts, throughput-oriented

```yaml
# vLLM config: RAG workload (8K–32K prompts, H100-80GB)
model: meta-llama/Meta-Llama-3-70B-Instruct
dtype: bfloat16
max_model_len: 16384
max_num_seqs: 32
max_num_batched_tokens: 8192
block_size: 16
gpu_memory_utilization: 0.92
enable_chunked_prefill: true
enable_prefix_caching: true
tensor_parallel_size: 4  # 70B requires 4× H100

# Rationale:
#   max_model_len=16384 covers 95th-percentile RAG context
#   max_num_seqs=32: 32 × 16384 × 128KB = 64 GB KV — fits with 92% util
#   max_num_batched_tokens=8192: 32 decode + 8160 prefill per step
#   prefix caching: system prompt hit rate ~90% in typical RAG
```

### 14.12.3 Batch / Offline — No TTFT SLA, maximize throughput

```yaml
# vLLM config: offline batch workload (summarization, classification)
model: meta-llama/Meta-Llama-3-8B-Instruct
dtype: bfloat16
max_model_len: 8192
max_num_seqs: 256
max_num_batched_tokens: 32768
block_size: 32
gpu_memory_utilization: 0.95
enable_chunked_prefill: false   # unchunked: maximize prefill throughput
enable_prefix_caching: false    # no repeated prefixes in offline batch
tensor_parallel_size: 1

# Rationale:
#   max_num_batched_tokens=32768: large prefill batches maximize GPU util
#   enable_chunked_prefill=false: no need to protect decode latency
#   block_size=32: long sequences benefit from fewer block table entries
#   gpu_memory_utilization=0.95: dedicated machine, squeeze every GB
```

---

## 14.13 Code Listing  `[FOUNDATIONAL]`

See `code/chapter_14/knobs_demo.py` for:

- Memory budget calculator: given hardware + model, output max_num_seqs
- Parameter interaction checker: detect common misconfiguration patterns
- YAML config generator for the three workload archetypes
- Throughput and TTFT estimator: model the effect of each parameter change

See `code/chapter_14/knobs_demo.cpp` for:

- llama.cpp CLI flag → `llama_context_params` struct field annotated mapper
- Memory budget calculator for llama.cpp (`--parallel × --ctx-size × KV`)
- KV pool sizing comparison: vLLM dynamic pool vs. llama.cpp static buffers

---

## 14.14 Summary

```
Key takeaways:

1. The eight parameters govern three shared resources:
   HBM (memory), token budget (compute), and scheduler admission.
   Every misconfiguration traces to one of these three.

2. max_num_seqs × KV_per_seq must fit in the KV pool.
   KV pool = HBM × gpu_memory_utilization − weights − activations.
   Always compute this before setting max_num_seqs.

3. Halving max_model_len doubles concurrency for the same HBM.
   Profile your p99 context length; don't over-allocate.

4. max_num_batched_tokens must satisfy:
   max_num_batched_tokens ≥ 4 × max_num_seqs
   Violating this causes prefill starvation.

5. The three common failure modes:
   OOM Triangle: max_num_seqs too high for the KV pool.
   Prefill Starvation: max_num_batched_tokens ≤ max_num_seqs.
   Context Cliff: max_model_len below p99 prompt length.

6. tensor_parallel_size splits weights across GPUs.
   NVLink: ~4% overhead — use freely.
   PCIe: ~23% overhead — minimize TP; consider alternatives.

7. enable_chunked_prefill + enable_prefix_caching together
   deliver the best TTFT and cost for mixed-latency workloads.
   Always enable both on production RAG deployments.

8. llama.cpp differences:
   --parallel allocates fixed memory at startup (no dynamic KV pool).
   --n-gpu-layers controls compute placement, not memory fraction.
   --cache-prompt is per-session only (no cross-request prefix sharing).
```

---

## 14.15 Self-Check Questions

1. A LLaMA 3 70B model in BF16 is deployed on 4× A100-80GB with TP=4.
   `gpu_memory_utilization=0.90`, peak activations = 3 GB per GPU,
   `max_model_len=8192`.  What is the maximum safe `max_num_seqs`?
   (KV bytes per token for 70B: 2 × 80 layers × 8 heads × 128 dim × 2 = 327 KB)

2. You observe `vllm:num_requests_waiting` growing steadily with
   `max_num_seqs=128` and `max_num_batched_tokens=256`.  Without
   changing `max_num_seqs`, what is the minimum `max_num_batched_tokens`
   to unblock prefill, and why?

3. A workload has p50 prompt length = 512 tokens and p99 = 28 000 tokens.
   You set `max_model_len=4096`.  What percentage of requests will be
   rejected, and what is your recommended two-tier configuration?

4. Your server runs on 4× H100-SXM5 connected via NVLink.  The model is
   LLaMA 3 70B.  Should you use `tensor_parallel_size=4` or
   `tensor_parallel_size=2` with two separate deployments?  Justify.

5. A colleague changes `block_size` from 16 to 32 on a running vLLM
   server using an environment variable.  Will the change take effect?
   What must they do instead?

---

## Where We Go Next

Chapter 15 extends the multi-GPU discussion with full coverage of tensor
parallelism, pipeline parallelism, NVLink vs. PCIe trade-offs, and
llama.cpp's partial GPU offload strategy.  We also cover the Ray-based
distributed executor that vLLM uses to coordinate workers, and the
practical mechanics of launching a multi-GPU deployment.

*Next: Chapter 15 — Multi-GPU Serving (vLLM) and GPU Offload (llama.cpp)*


---

## Chapter Summary

- **Eight levers that govern production vLLM behavior**: `--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`, `--gpu-memory-utilization`, `--tensor-parallel-size`, `--quantization`, `--enable-prefix-caching`, and `--speculative-model`.
- **`--max-num-seqs`**: caps the scheduler's running queue; set too high → OOM; set too low → GPU under-utilized.
- **`--max-num-batched-tokens`**: token budget per forward pass; trades TTFT vs throughput.
- **`--gpu-memory-utilization`**: fraction of HBM reserved for blocks; leave 10% headroom for activation peaks.
- **Interaction effects**: raising `--max-num-seqs` without raising `--max-num-batched-tokens` gives more sequences per batch but fewer tokens per sequence per step.
- **Prefix caching + chunked prefill**: enabling both is the default path to minimising TTFT at high request rates without sacrificing throughput.
- **quantization knob**: `--quantization fp8` on H100 is typically a free 1.5–2× throughput gain with negligible quality loss.
- **llama.cpp equivalents**: `-c` (context), `-n` (max new tokens), `-t` (threads), `-ngl` (GPU layers), `-b` (batch size), `--flash-attn`.

---

## Self-Check Questions

1. You set `--max-num-seqs 128` and `--max-num-batched-tokens 2048`. You have 128 running sequences at decode step T. Each contributes 1 token. Does the batch fit? What if 4 of those are prefill sequences with 400 tokens each? *(Section 14.1)*

2. `--gpu-memory-utilization 0.95` on an A100 80 GB with a 16 GB model: compute the KV block budget in GiB. What is the risk of the 0.05 headroom being insufficient? *(Section 14.4)*

3. A user asks for generation up to 8 192 tokens but `--max-model-len 4096` is set. What happens? Where exactly is the check performed, and what error or truncation occurs? *(Section 14.2)*

4. You enable `--enable-prefix-caching` and `--speculative-model ngram`. Name the two configuration interactions that could cause this combination to break or underperform. *(Sections 14.7, 14.8)*

5. llama.cpp's `-ngl 35` parameter offloads 35 transformer layers to the GPU. For a 32-layer model, what does this value mean in practice? What happens with `-ngl 0`? *(Section 14.9)*


---

## Worked Solutions

### Question 1
**Setup:** LLaMA 3 70B BF16, TP=4, 4× A100-80GB, `gpu_memory_utilization=0.90`, peak activations=3 GB/GPU, `max_model_len=8192`. KV bytes per token = 327 KB.

**Step 1 — HBM budget per GPU.**
Each A100 has 80 GB total. With `gpu_memory_utilization=0.90`, the engine claims:
```
HBM per GPU = 80 × 0.90 = 72 GB
```

**Step 2 — Subtract model weights.**
70B parameters × 2 bytes (BF16) = 140 GB total. Across TP=4:
```
weights per GPU = 140 / 4 = 35 GB
```

**Step 3 — Subtract peak activations.**
```
remaining = 72 − 35 − 3 = 34 GB per GPU
```

**Step 4 — KV cache pool.**
With TP=4 the KV heads are also sharded. Given 327 KB/token total, per-GPU KV cost per token:
```
KV per token per GPU = 327 KB / 4 ≈ 81.75 KB  (roughly; assumes GQA heads are evenly split)
```
But the 34 GB is the total KV pool available per GPU. Total KV pool across 4 GPUs:
```
total KV pool = 34 × 4 = 136 GB
```

**Step 5 — Max sequences.**
Total KV bytes needed for one sequence at `max_model_len=8192`:
```
KV per sequence = 327 KB × 8192 ≈ 2.68 GB
```
Maximum safe concurrent sequences:
```
max_num_seqs = floor(136 / 2.68) ≈ 50 sequences
```

**Common mistake:** forgetting to subtract activation headroom leads to OOM during prefill spikes when activations peak above the assumed 3 GB/GPU.

---

### Question 2
**Setup:** `max_num_seqs=128`, `max_num_batched_tokens=256`. 128 decode sequences at step T.

**Part A — Decode-only step:**
Each of the 128 sequences contributes exactly 1 new token → 128 tokens total.
```
128 ≤ 256   →   the batch fits ✓
```

**Part B — Mixed step with 4 prefill sequences (400 tokens each):**
Decode tokens: 124 × 1 = 124
Prefill tokens: 4 × 400 = 1,600
Total: 124 + 1,600 = **1,724 tokens**
```
1,724 > 256   →   the batch does NOT fit ✗
```

**Resolution:** The scheduler respects `max_num_batched_tokens`. It will either:
1. Chunk the prefill requests into 256-token chunks (if `--enable-chunked-prefill`), or
2. Defer the prefill requests to a future step, admitting only decode traffic now.

Takeaway: `max_num_batched_tokens` is the primary lever for bounding TTFT under mixed workloads. Set it to at least the p95 prompt length to avoid prefill starvation.

---

### Question 3
**Setup:** `gpu_memory_utilization=0.95`, A100 80 GB, model weights = 16 GB.

**Step 1 — HBM claimed:**
```
HBM = 80 × 0.95 = 76 GB
```

**Step 2 — Available for KV blocks:**
```
KV pool = 76 − 16 = 60 GB = 60 × 1024 MiB = 61,440 MiB
```

**Step 3 — Risk of 0.05 headroom.**
The 4 GB headroom must absorb:
- Peak activation tensors during prefill (can spike 2–6 GB for large batches)
- CUDA context memory (~1 GB)
- Fragmentation in the allocator

If a large prefill batch hits the GPU while the activation spike is 5 GB, the 4 GB headroom is **insufficient** → CUDA OOM. The server crashes mid-request.

**Recommendation:** Use `gpu_memory_utilization=0.90` (8 GB headroom) as the default. Only push to 0.95 if you have measured activation peaks and confirmed they stay under 4 GB.

---

### Question 4
**Setup:** `max_model_len=4096`, user requests generation up to 8,192 tokens.

**What happens:**
At request admission time, the scheduler checks:
```python
if prompt_len + max_new_tokens > max_model_len:
    raise ValueError(...)
```
The check is performed in `vllm/engine/llm_engine.py` inside `_validate_inputs()` before the request enters the scheduler queue.

The error returned to the API caller is:
```
400 Bad Request
{"error": {"message": "This model's maximum context length is 4096 tokens. However, you requested 8192 tokens (512 in the messages, 7680 in the completion). Please reduce the length of the messages or completion."}}
```

There is **no silent truncation** in vLLM's default configuration — the request is rejected. In contrast, some frameworks truncate the prompt silently, which can cause unexpected behavior for users who don't monitor the `finish_reason`.

**Fix options:** (a) raise `max_model_len`, (b) reduce `max_tokens` in the request, (c) implement client-side truncation before sending.

---

### Question 5
**Config:** `--enable-prefix-caching` + `--speculative-model ngram`.

**Interaction 1 — Speculative decoding invalidates prefix cache blocks.**
Speculative decoding proposes draft tokens that may be rejected. When a draft is rejected and the sequence backtracks, the KV blocks that were provisionally allocated for draft tokens must be discarded. This makes the block addresses unstable, breaking the assumption prefix caching relies on (stable, hash-consistent blocks). Result: cache hit rate drops dramatically on speculative decode steps.

**Interaction 2 — N-gram draft model uses a sliding context window, not a prefix.**
The n-gram model predicts the next token by looking at the last N tokens. Its speculation window moves as tokens are generated. But prefix caching is beneficial only when the *beginning* of a sequence matches a cached prefix. After the shared system prompt, the n-gram model's window has no relationship to cached prefixes, so any cache benefit is confined to the system-prompt portion. For short system prompts or diverse user messages, combined gain is negligible.

**Combined result:** The two features interfere. Disable speculative decoding for workloads where prefix cache hit rate is the primary optimization goal. Use n-gram speculation only for batch=1 latency-critical workloads where prefix sharing is not a priority.

---

### Question 6 (End-of-chapter set)
**Part A — 128 seqs × 1 decode token = 128 tokens ≤ 2048. Fits.✓**

**Part B — 4 prefill (400 tokens each) + 124 decode (1 token each):**
4 × 400 + 124 = 1,724 tokens > 2,048? No: 1,724 < 2,048. **Fits.✓**
*(Common trap: assuming prefill + decode combined exceeds the limit when it doesn't.)*

---

### Question 7
**Step 1:** `gpu_memory_utilization=0.95` → 80 × 0.95 = 76 GiB claimed.
**Step 2:** KV pool = 76 − 16 = 60 GiB.
**Risk:** 4 GiB headroom must cover peak activation tensors. For large prefill batches (e.g., 128 sequences × 512 tokens), activation peaks can exceed 4 GiB → OOM crash. Keep headroom ≥ 8 GiB (use 0.90) unless activations are measured to be small.

---

### Question 8
**What happens with `max_model_len=4096` and 8,192-token request:**
vLLM validates at admission in `_validate_inputs()`. It raises a 400-class error immediately — no truncation, no generation. The check compares `prompt_tokens + requested_max_new_tokens > max_model_len`.

---

### Question 9
**Two interactions that break `--enable-prefix-caching` + `--speculative-model ngram`:**
1. Draft token rejection causes block revocation, destroying prefix-cache consistency.
2. N-gram speculation is incompatible with chunked-prefill scheduling that prefix caching relies on — the draft model needs the *last* N tokens, not a prefix, so its window conflicts with the prefix-cache block alignment.

---

### Question 10
**`-ngl 35` for a 32-layer model:**
llama.cpp treats any `-ngl` value ≥ the model's total layer count as "offload all layers." For a 32-layer model, `-ngl 35` is effectively `-ngl 32` — all transformer layers go to the GPU. The embedding and output layers may stay on CPU depending on the build.

**`-ngl 0`:** Zero layers offloaded. All compute runs on CPU. GPU is unused entirely. Inference is CPU-only and typically 5–20× slower than GPU inference for large models.

