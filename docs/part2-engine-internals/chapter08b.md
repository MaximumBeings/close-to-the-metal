# Chapter 8.5: CUDA Graphs — Capture, Replay, and Production Latency

> *"The GPU is fast. The path from Python to the GPU is not. CUDA Graphs collapse that path to a single launch."*

## What you will understand after this chapter

- Why CPU kernel-launch overhead becomes the bottleneck at small batch sizes
- How CUDA graph capture and replay work at the hardware level
- Exactly what breaks a CUDA graph and how vLLM works around it
- How vLLM's multi-size graph pool achieves both low latency and flexibility
- When to enable or disable CUDA graphs in production

## What you need first

- Chapter 8 (Startup and Initialization) — the engine warm-up sequence
- Chapter 9 (The Forward Pass) — CUDA kernel dispatch basics

---

## 8b.1  The CPU Launch Overhead Problem

Every CUDA kernel launch involves the CPU:

```
Python / C++ host code
        │
        ▼  cudaLaunchKernel() syscall
┌───────────────────────────────────┐
│   CUDA Driver                     │
│   ─ validate args                 │
│   ─ pack launch descriptor        │
│   ─ enqueue in work queue         │
└───────────────────────────────────┘
        │
        ▼  hardware queue
┌───────────────────────────────────┐
│   GPU Hardware Scheduler          │
│   (dequeue and execute)           │
└───────────────────────────────────┘
```

For a single kernel launch, the CPU overhead is roughly **5–20 μs**. That sounds negligible — until you count how many kernels a forward pass actually launches.

**WORKED EXAMPLE 8b.1 — Kernel count for a 70B model forward pass:**

```
WORKED EXAMPLE 8b.1 — Kernel count, Llama-3-70B
────────────────────────────────────────────────
Model:   80 transformer layers, each containing:
  - RMSNorm:                 1 kernel
  - QKV projection:          1 GEMM
  - RoPE embedding:          1 kernel
  - Attention (FlashAttn):   1 kernel
  - Output projection:       1 GEMM
  - RMSNorm (post-attn):     1 kernel
  - Gate + up projection:    2 GEMMs
  - SiLU activation:         1 kernel
  - Down projection:         1 GEMM
  Per layer:                ~10 kernels
  
80 layers × 10 kernels     = 800 kernels
Embedding lookup            =   2 kernels
LM head + softmax           =   3 kernels
Misc (norms, residuals)     =  ~15 kernels
                            ──────────────
Total:                      ≈ 820 kernels

At 10 μs overhead each:
  CPU overhead = 820 × 10 μs = 8.2 ms per forward pass

Typical GPU compute time (decode, batch=1, 70B, 4×H100):
  ≈ 25 ms

CPU overhead fraction: 8.2 / (8.2 + 25) = 25%
────────────────────────────────────────────────
```

A quarter of your decode latency is just the CPU telling the GPU what to do. At batch=1 (single-user, latency-critical serving), this is unacceptable. At batch=32, the GPU compute time dominates and the launch overhead is small in comparison — but the use cases that care most about latency are precisely the ones with small batch sizes.

---

## 8b.2  What a CUDA Graph Is

A **CUDA graph** records a sequence of GPU operations (kernels, memory copies, memory allocations) as a *graph* of nodes and dependencies. Once captured, the entire graph is submitted to the GPU with **a single CPU call**, regardless of how many individual kernels it contains.

```
Without CUDA graph:
  CPU ──► launch kernel 1 ──► launch kernel 2 ──► ... ──► launch kernel 820
  820 CPU-GPU round trips, each 5–20 μs

With CUDA graph:
  [capture phase]
  CPU ──► record kernel 1, kernel 2, ..., kernel 820  (once, at startup)

  [replay phase — every subsequent request]
  CPU ──► cudaGraphLaunch()  (one call, ~5 μs total)
           └── GPU executes all 820 kernels
```

The graph is a DAG (directed acyclic graph) of operations:

```
┌─────────────────────────────────────────────┐
│              CUDA Graph                      │
│                                              │
│  [RMSNorm]──►[QKV GEMM]──►[RoPE]──►[Attn] │
│                                    │         │
│                                    ▼         │
│                              [Out GEMM]      │
│                                    │         │
│                          (repeat × 80 layers)│
│                                    │         │
│                              [LM Head]       │
└─────────────────────────────────────────────┘

Single cudaGraphLaunch() executes the entire graph.
GPU scheduler sees one work item, not 820 separate launches.
```

### The hardware mechanism

Modern NVIDIA GPUs have a **work submission engine** separate from the compute SMs. Without graphs, the work submission engine is fed by the CPU in real time — it can only process the next kernel after the CPU has launched it. With graphs, the entire sequence is pre-loaded into GPU-side memory. The submission engine walks the graph without any CPU involvement.

---

## 8b.3  Capture: How It Works

Capturing a CUDA graph means running a forward pass in "record" mode. PyTorch's API:

```python
# Warm-up run (to allocate CUDA memory, compile kernels, etc.)
model(inputs)
torch.cuda.synchronize()

# Capture
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    outputs = model(inputs)  # Nothing actually executes here
                              # The CUDA driver intercepts every
                              # kernel launch and records it

# Replay (every subsequent call)
graph.replay()  # Single launch, ~5 μs
# outputs tensor now contains the new result
```

`[COMMON TRAP]` — During capture, the **input tensors must be the same objects** used during replay. You don't pass new inputs to `graph.replay()`. Instead, you copy new data *into* the tensors that were live during capture:

```python
# Wrong — creates new tensors, graph doesn't know about them
graph.replay()
output = model(new_input)   # ← broken, graph uses old tensors

# Correct — mutate the captured tensors in place
input_tensor.copy_(new_input)   # update in place
graph.replay()                  # graph reads from the same memory
# output_tensor now has new result
```

This is the fundamental constraint that drives all of vLLM's CUDA graph design.

---

## 8b.4  What Breaks a CUDA Graph

`[COMMON TRAP]` The following patterns cannot be captured and will cause the graph to either fail silently or fall back to eager mode:

### Dynamic shapes
```
Broken: different sequence lengths in different requests
─────────────────────────────────────────────────────────
  Capture batch=1, seq=128
  Replay with seq=256  → WRONG: different tensor shapes,
                                 different GEMM dimensions
```

The GEMM kernel for Q×K^T is parameterized by `(seq_len, head_dim)`. A different seq_len means a different kernel configuration — you can't replay the captured one.

### CPU-GPU conditional branches
```python
# This cannot be captured:
if some_condition_that_depends_on_gpu_output:
    do_path_a()
else:
    do_path_b()
```

The graph is a static DAG — no conditional branches.

### Dynamic memory allocation
Any `torch.empty()` or `torch.zeros()` inside the forward pass during replay will fail. All tensors must be pre-allocated and reused.

### Non-deterministic NCCL collectives (some variants)
All-reduce operations on some NCCL configurations cannot be captured. vLLM works around this by using NCCL's static graph mode.

---

## 8b.5  vLLM's Multi-Size Graph Pool

vLLM solves the dynamic-shape problem with a **graph pool**: pre-captured graphs for every batch size that might occur at runtime.

```
vLLM graph pool (default configuration):
─────────────────────────────────────────
  Captured sizes:  1, 2, 4, 8, 16, 32, 64, 128, 256
  (exact sizes vary by --max-num-seqs setting)

At runtime, for a batch of size N:
  Find the smallest captured size ≥ N
  Pad the batch with dummy tokens to reach that size
  Replay the corresponding graph
  Discard dummy outputs

Example: 5 real requests → use size-8 graph, pad 3 slots
```

```
┌────────────────────────────────────────────────────────┐
│              vLLM Graph Pool                            │
│                                                         │
│  graph_1    ──►  captured for batch_size=1              │
│  graph_2    ──►  captured for batch_size=2              │
│  graph_4    ──►  captured for batch_size=4              │
│  graph_8    ──►  captured for batch_size=8              │
│  graph_16   ──►  captured for batch_size=16             │
│  ...                                                    │
│  graph_256  ──►  captured for batch_size=256            │
│                                                         │
│  Incoming batch of 5:                                   │
│    ┌──────────────────────────────────────────────┐     │
│    │ seq1 │ seq2 │ seq3 │ seq4 │ seq5 │ PAD │PAD│PAD│  │
│    └──────────────────────────────────────────────┘     │
│    Use graph_8, discard last 3 output slots             │
└────────────────────────────────────────────────────────┘
```

**WORKED EXAMPLE 8b.2 — Memory cost of the graph pool:**

```
WORKED EXAMPLE 8b.2 — Graph pool memory overhead
──────────────────────────────────────────────────
Model: Llama-3-70B, 4×H100 (320 GB total HBM)
Weights: 140 GB (BF16)
Available for KV + graphs: 180 GB

Per-graph capture overhead:
  Activation tensors for max captured size (batch=256):
    256 seq × 4096 hidden_dim × 2 bytes × 80 layers
    = 256 × 4096 × 2 × 80 = 167 MB

  9 graph sizes: 9 × 167 MB ≈ 1.5 GB total

  CUDA graph node metadata: ~50 MB

Total graph pool overhead: ~1.6 GB  (< 1% of HBM)
──────────────────────────────────────────────────
```

The memory cost is negligible. The latency gain is not:

```
Measured decode latency, Llama-3-8B, batch=1, A100 80GB:
  Without CUDA graphs:  ~14 ms
  With CUDA graphs:     ~9 ms
  Reduction: 36% (the CPU overhead fraction)

At batch=32:
  Without:  ~42 ms
  With:     ~40 ms
  Reduction: 5% (GPU-bound, overhead is small)
```

CUDA graphs deliver the most value at the smallest batch sizes — exactly where latency-sensitive applications live.

---

## 8b.6  vLLM's Capture Sequence at Startup

Chapter 8 described the warm-up sequence. Here is what happens specifically for CUDA graph capture during `engine.start_profile()`:

```
vLLM startup — graph capture phase
────────────────────────────────────────────────────────
Step 1: Allocate KV cache blocks (Ch 6 block manager)
         └─ all KV memory pre-allocated, no dynamic alloc
            will occur during capture

Step 2: Run one eager (non-graph) forward pass at max batch
         └─ triggers triton kernel compilation and caching
         └─ ensures all kernels are compiled before capture
         └─ "warms up" CUDA memory caching allocator

Step 3: For each batch size B in [256, 128, ..., 2, 1]:
         a. Load dummy batch of size B (all zeros, valid tokens)
         b. torch.cuda.synchronize()
         c. with torch.cuda.graph(graph_B):
                output = model.forward(dummy_batch)
         d. Store graph_B in pool
         e. torch.cuda.synchronize()

Step 4: Pool ready — all subsequent decode steps use replay
────────────────────────────────────────────────────────
```

Note that capture happens **largest-to-smallest**. This is because CUDA's memory allocator caches allocations from larger runs, so smaller graphs can reuse that cached memory rather than triggering new allocations.

---

## 8b.7  Chunked Prefill and Graphs: The Complication

`[DEEP DIVE]`

Prefill (processing the input prompt) is harder to graph than decode because prompt lengths are variable. vLLM handles this with a **two-path design**:

```
Incoming request
       │
       ├─ PREFILL phase ──► Eager mode (no graph)
       │                    Handles variable prompt lengths
       │                    Chunked prefill splits long prompts
       │                    into fixed-size chunks → can be graphed
       │
       └─ DECODE phase  ──► Graph mode
                            Batch of active sequences, fixed shape
                            Padded to captured size, replayed
```

With chunked prefill enabled (`--enable-chunked-prefill`), prefill chunks are also fixed-size and can be captured. vLLM maintains a separate pool of prefill graphs for each chunk size. This is the configuration that achieves the lowest end-to-end latency for mixed prefill+decode workloads.

---

## 8b.8  When to Disable CUDA Graphs

CUDA graphs are enabled by default in vLLM. You should disable them (`--enforce-eager`) in these situations:

| Situation | Why graphs break | Solution |
|-----------|-----------------|----------|
| Custom attention backends that don't support static graphs | Dynamic dispatch during forward pass | Use a graph-compatible backend (FlashInfer) |
| Very high max sequence length with small GPU memory | Graph pool pre-allocates at max size | Reduce `--max-model-len` or disable graphs |
| Debugging / profiling with tools like `nsys` | Graph replay looks like a single kernel to the profiler | Disable for profiling sessions |
| Models with dynamic control flow (some custom architectures) | Conditional branches can't be captured | No fix — disable graphs |
| LoRA serving with many adapters | Each adapter combination is a different graph | vLLM handles this with per-adapter graphs, but may OOM |

In practice, for standard LLaMA/Mistral/Qwen architectures on a normal GPU, you should almost always leave CUDA graphs enabled.

---

## 8b.9  llama.cpp's Equivalent: Static Computation Graph

llama.cpp does not use CUDA graphs by name but achieves a similar effect through its **static ggml computation graph**. The `ggml_cgraph` is built once during initialization and reused for every forward pass:

```c
// llama.cpp graph build (simplified)
struct ggml_cgraph * llama_build_graph(
    struct llama_context & lctx,
    const llama_batch   & batch)
{
    // Returns a pre-built graph; no dynamic allocation
    // during forward pass
    auto * ctx0  = lctx.ctx_compute.get();
    auto * gf    = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);
    // ... build graph nodes ...
    return gf;
}

// At inference time:
struct ggml_cgraph * gf = llama_build_graph(ctx, batch);
ggml_backend_graph_compute(ctx->backend, gf);  // single dispatch
```

The key difference: ggml's static graph is a software-level construct (operations are pre-ordered at graph build time), while CUDA graphs are a hardware-level feature that eliminates CPU involvement entirely during replay. For CPU backends, ggml's approach is equivalent; for GPU backends, CUDA graphs provide an additional layer of CPU overhead elimination.

---

## 8b.10  FlashInfer and Graph Compatibility

FlashInfer (the modern attention backend now default in production vLLM) is written to be CUDA graph compatible from the start. This was not true of the original Triton-based attention kernels used in early vLLM.

The key design choice: FlashInfer pre-computes attention metadata (KV indices, block tables, sequence lengths) into GPU-side tensors before the graph replay. During replay, the attention kernel reads from those pre-computed tensors — no CPU-side dynamic dispatch needed.

```
FlashInfer + CUDA graph workflow:
─────────────────────────────────
Step 1 (CPU, before graph replay):
  Update GPU-side metadata tensors:
    - kv_indptr: block table pointers
    - kv_indices: physical block addresses
    - q_indptr: query sequence boundaries

Step 2 (graph replay, GPU only):
  FlashInfer reads metadata tensors
  Computes attention with zero CPU involvement
  Writes output to pre-allocated output tensor

Step 3 (CPU, after replay):
  Read outputs — no sync needed until sampling
```

---

## Chapter Summary

CUDA graphs collapse ~820 individual kernel launch calls into a single GPU submission, eliminating CPU overhead that accounts for 20–36% of decode latency at batch=1. Capture records a fixed-shape forward pass once at startup; replay re-executes it with new data copied into the same pre-allocated tensors. vLLM maintains a pool of graphs for each power-of-two batch size up to `max_num_seqs`, padding real batches to the nearest captured size at runtime. Dynamic shapes, conditional branches, and dynamic allocations all break capture; vLLM's architecture is carefully designed to avoid all three in the decode path. Disable graphs (`--enforce-eager`) only when debugging, profiling, or using architectures with genuinely dynamic control flow.

---

## Self-Check Questions

1. A 7B model on an RTX 4090 has a decode latency of 22 ms without CUDA graphs. If CPU overhead is 18% of total latency, what is the expected latency with graphs? Show the arithmetic.

2. You have a batch of 13 active decode sequences. vLLM's graph pool contains sizes [1, 2, 4, 8, 16, 32]. Which graph is used? How many dummy slots are added? What fraction of compute is "wasted" on dummy tokens?

3. Why does vLLM capture graphs in descending order (largest first)? What goes wrong if you capture smallest first?

4. A team is deploying a custom model with a dynamic attention mask that depends on the content of the current token (not just position). Can this model use CUDA graphs? Why or why not?

5. Chunked prefill with fixed chunk size C is enabled. How does this enable graphs for the prefill path? What additional pool of graphs must vLLM maintain?


---

## Worked Solutions

---

### Solution 1 — Expected decode latency with CUDA graphs

**Given:** Without graphs: 22 ms; CPU overhead: 18% of total = 18% × 22 ms = 3.96 ms; GPU compute: 22 − 3.96 = 18.04 ms

**Step 1 — Understand what graphs eliminate.**

CUDA graphs replace repeated CPU-driven kernel launches with a single `cudaGraphLaunch` call. CPU overhead ≈ 0 after graphs (the scheduling work is pre-recorded).

**Step 2 — Expected latency with graphs.**

$$\text{latency with graphs} = \text{GPU compute} + \text{reduced CPU overhead}$$
$$\approx 18.04 \text{ ms} + \sim 0.1 \text{ ms (graph launch)} \approx \textbf{18.1 ms}$$

**Step 3 — Speedup.**

$$\text{speedup} = \frac{22}{18.1} \approx \textbf{1.21}\times \quad (21\% \text{ faster})$$

**Practical note:** The 18% CPU overhead figure is conservative for small batch sizes. At batch size 1, CPU overhead can be 40–60% of total decode latency (most time is spent launching kernels, not executing them). At batch size 32+, GPU compute dominates and graph speedup is smaller in percentage terms but still valuable in absolute terms.

---

### Solution 2 — Graph selection for 13 active sequences

**Given:** Graph pool sizes = {1, 2, 4, 8, 16, 32}; active sequences = 13

**Step 1 — Find the smallest graph size ≥ 13.**

From the pool: 1, 2, 4, 8, **16**, 32. The smallest size ≥ 13 is **16**.

**Step 2 — Dummy slots.**

$$\text{dummy slots} = 16 - 13 = \textbf{3 dummy tokens}$$

The 3 dummy slots are filled with padding tokens. Their K/V blocks are allocated (pointing to a shared dummy block) and their output logits are computed but discarded.

**Step 3 — Compute waste fraction.**

$$\text{waste} = \frac{3}{16} = \textbf{18.75\%}$$

This 18.75% waste means 18.75% of the GPU's compute per step is doing "useful" work for dummy tokens that will never produce real output. This is acceptable — the alternative (running 13 sequences without graphs) would have 18% CPU overhead.

**The break-even point:** If dummy waste > CPU overhead saved, graphs are not worth it. For a 4-sequence graph with 3 dummies: 3/4 = 75% waste > 18% saved → may not be beneficial. vLLM tunes graph sizes to keep waste manageable.

---

### Solution 3 — Why graphs are captured in descending order

**Step 1 — The problem with smallest-first.**

When you capture a graph at batch size 1, CUDA allocates all intermediate tensors for that forward pass. These small tensors occupy fragments of VRAM. When you subsequently try to capture batch size 256, CUDA needs large contiguous tensor allocations — but VRAM is fragmented by the earlier small-tensor allocations. The large allocation may fail even if total free VRAM is sufficient.

**Step 2 — Why largest-first works.**

Capturing batch size 256 first allocates all tensors at maximum size. These are freed after capture. When batch size 128 is captured next, CUDA can reuse the same contiguous regions. Smaller batch sizes reuse increasingly small sub-regions of the same memory space.

**Step 3 — CUDA graph memory persistence.**

Captured graphs retain their tensor memory during replay. Capturing largest-first ensures all graph memory is in a contiguous region of VRAM, minimizing fragmentation during serving.

---

### Solution 4 — Content-dependent dynamic attention mask and CUDA graphs

**The problem:**

CUDA graphs record all GPU operations, including memory addresses and tensor shapes. A graph is valid for replay only if the *exact same sequence of operations with the same shapes* is executed each time.

A content-dependent attention mask means:

- At step T₁: mask is [1,0,1,0,1] (based on token content)
- At step T₂: mask is [0,1,1,0,0] (different tokens, different mask)

These are **different operations** — they branch to different memory addresses and may trigger different execution paths.

**Why graphs fail:**

CUDA graph capture records one specific execution path. If the mask changes the control flow (e.g., a `torch.where` with different conditions), the graph cannot adapt. Replaying the graph from step T₁ for step T₂ would apply the wrong mask.

**Solution:** Use eager mode (no graphs) for this model. The CPU overhead (~3–5 ms) is unavoidable. Alternatively, reformulate the dynamic mask as a fixed mathematical function of token *position* (not content) — then it becomes static and capturable.

---

### Solution 5 — Chunked prefill with CUDA graphs

**Step 1 — Why chunked prefill enables graphing for the prefill path.**

Without chunked prefill, prefill length varies request by request (100 tokens, 1,000 tokens, 3,000 tokens). Variable shapes → cannot use CUDA graphs.

With chunk_size=C (fixed), every prefill step processes exactly C tokens. Fixed shape → graphs are possible.

**Step 2 — Additional graph pool required.**

vLLM maintains two separate graph pools:

1. **Decode graph pool:** batch sizes {1, 2, 4, 8, ..., max_num_seqs} — one graph per size
2. **Prefill chunk graph:** one graph for batch size C (the fixed chunk size)

The prefill graph handles the combined step: C prefill tokens + up to `max_num_seqs` decode tokens. In practice, vLLM also needs graphs for partial chunks (final chunk of a prompt may be < C tokens), so the prefill pool may have multiple sizes.

**Step 3 — Memory cost.**

Each additional graph retains its tensor memory. The prefill chunk graph for C=512 tokens with 32 sequences holds 512-token attention intermediates in memory. This adds 10–50 MB to the graph memory pool depending on model size.

