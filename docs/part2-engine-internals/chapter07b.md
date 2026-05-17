# Chapter 7.5: Continuous Batching — The Iteration-Level Scheduling Loop

> *"Naïve batching waits for the slowest request. Continuous batching never waits. It just keeps feeding the GPU, one token-step at a time."*

## What you will understand after this chapter

- Why static batching wastes GPU capacity and how continuous batching eliminates that waste
- The exact mechanics of the iteration-level scheduling loop
- How token budgets prevent a single long request from starving the batch
- The prefill/decode interleave tradeoff and how vLLM navigates it
- Preemption accounting: when requests are evicted and how the cost is measured
- How llama.cpp handles multi-user batching differently

## What you need first

- Chapter 6 (PagedAttention) — block allocation and eviction
- Chapter 7 (The Scheduler) — request lifecycle states

---

## 7b.1  The Static Batching Problem

Before continuous batching existed, serving systems processed requests in **static batches**: collect N requests, run them all to completion, start the next batch. This is how GPU inference worked in 2021.

```
Static batching timeline:
─────────────────────────────────────────────────────────────────
Batch  │ Req A (5 tokens)  │▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
       │ Req B (50 tokens) │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
       │ Req C (30 tokens) │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░│
                            ▲                                   ▲
                            batch start                         batch end
                                                    (must wait for Req B)

░ = GPU idle (req finished, waiting for rest of batch)
▓ = GPU active

GPU utilization: (5+50+30) / (50×3) = 56.7%
```

Requests A and C finish early but the GPU slot stays allocated until B finishes. For LLM inference, where output lengths vary by 10–100× across requests, this is catastrophic. A single verbose response holds up every other slot in the batch.

**WORKED EXAMPLE 7b.1 — Static batching waste:**

```
WORKED EXAMPLE 7b.1 — Static batch GPU utilization
────────────────────────────────────────────────────
Batch of 8 requests, output lengths (tokens):
  [4, 7, 12, 3, 89, 6, 14, 22]

Max in batch: 89 tokens
Total compute if no waste: 4+7+12+3+89+6+14+22 = 157 token-steps
Total slots consumed:       89 × 8              = 712 token-steps

utilization: 157 / 712 = 22.1%

One outlier (89 tokens) wastes 77.9% of GPU capacity.
────────────────────────────────────────────────────
```

---

## 7b.2  Continuous Batching: The Core Idea

**Continuous batching** (also called *iteration-level scheduling* or *in-flight batching*) makes one simple change: **check for finished requests after every single decode step**, and immediately replace them with new requests from the waiting queue.

```
Continuous batching timeline (same 3 requests):
─────────────────────────────────────────────────────────
Step  │ 1  2  3  4  5  6  7  8  9  10 ... 50
──────┼──────────────────────────────────────────────────
Slot 0│ A  A  A  A  A  D  D  D  D  D  ... D    ← new req D admitted at step 6
Slot 1│ B  B  B  B  B  B  B  B  B  B  ... B
Slot 2│ C  C  C  C  C  C  C  C  C  C  ... (fin at 30) → E admitted

GPU utilization: approaching 100%
```

The key insight: a "batch" is not a static collection of requests. It is a **sliding window** of active sequences, always as full as memory allows.

---

## 7b.3  The Iteration-Level Scheduling Loop

vLLM runs this loop once per GPU forward pass:

```
┌────────────────────────────────────────────────────────────┐
│              Iteration-Level Scheduling Loop               │
│                                                            │
│  1. FREE finished sequences                                │
│     └─ release their KV blocks back to the block pool     │
│                                                            │
│  2. CHECK preemption                                       │
│     └─ if free blocks < low-water mark → evict sequences  │
│                                                            │
│  3. SWAP IN sequences                                      │
│     └─ if free blocks available → restore swapped seqs    │
│                                                            │
│  4. ADMIT new requests                                     │
│     └─ from WAITING queue, up to token budget             │
│                                                            │
│  5. BUILD batch                                            │
│     └─ all RUNNING sequences → one forward pass           │
│                                                            │
│  6. EXECUTE forward pass                                   │
│     └─ GPU compute: attention + FFN + sampling            │
│                                                            │
│  7. EMIT tokens                                            │
│     └─ stream finished tokens to callers                  │
│                                                            │
│  GOTO 1                                                    │
└────────────────────────────────────────────────────────────┘
```

Each iteration takes roughly 20–50 ms for a 7B model on an A100. Steps 1–5 (the scheduling logic) take < 1 ms. The GPU compute in step 6 dominates.

### 7b.3.1  Step 1: Free finished sequences

After every forward pass, the scheduler checks each running sequence's output:

```python
# Simplified vLLM logic
for seq in running_seqs:
    if seq.last_token == EOS or seq.output_len >= seq.max_tokens:
        block_manager.free(seq)
        running_seqs.remove(seq)
        finished.append(seq)
```

Freeing blocks is O(number of blocks held), typically < 1 μs per sequence.

### 7b.3.2  Step 2: Preemption check

If the block pool is low, the scheduler must evict a running sequence to prevent OOM:

```
Preemption triggers when:
  free_blocks < watermark_blocks
  (default: watermark = 0.01 × total_blocks)

Two eviction policies:
  SWAP  — copy KV blocks to CPU RAM, free GPU blocks
          cost: PCIe bandwidth (≈ 16–64 GB/s) × KV size
          
  RECOMPUTE — free GPU blocks, discard KV cache entirely
              cost: full prefill re-run when sequence is resumed
              cheaper in GPU memory, expensive in compute
```

vLLM defaults to RECOMPUTE for sequences shorter than a threshold, SWAP for longer ones. The crossover point depends on CPU↔GPU bandwidth vs. prefill FLOP cost.

### 7b.3.3  Step 4: The token budget

Admitting new requests is gated by a **token budget**: the maximum number of tokens the scheduler will process in a single forward pass.

```
Token budget components:
─────────────────────────────────────────────────────
  max_num_batched_tokens   (vLLM flag, e.g. 4096)
  
  Consumed by:
    prefill tokens:  sum of input lengths of newly admitted requests
    decode tokens:   number of currently running sequences × 1
    
  Constraint:
    prefill_tokens + decode_tokens ≤ max_num_batched_tokens
```

**WORKED EXAMPLE 7b.2 — Token budget allocation:**

```
WORKED EXAMPLE 7b.2 — Token budget, max_num_batched_tokens=2048
────────────────────────────────────────────────────────────────
Currently running:   32 decode sequences
  → 32 decode tokens consumed

Waiting queue (by arrival order):
  Req W1: input_len=512
  Req W2: input_len=1024
  Req W3: input_len=256
  Req W4: input_len=800

Remaining budget: 2048 - 32 = 2016 tokens

Admissions:
  W1: 512 tokens → budget remaining: 2016 - 512 = 1504   ✓ admit
  W2: 1024 tokens → budget remaining: 1504 - 1024 = 480  ✓ admit
  W3: 256 tokens → budget remaining: 480 - 256 = 224     ✓ admit
  W4: 800 tokens → 224 < 800                             ✗ defer to next iter

This iteration processes:
  32 decode seqs + W1 (512) + W2 (1024) + W3 (256) = 1824 tokens total
────────────────────────────────────────────────────────────────
```

`[COMMON TRAP]` — Setting `max_num_batched_tokens` too low starves the prefill path. Long prompts (1K+ tokens) never get admitted because they can never fit the budget alongside active decode sequences. Set it to at least `max_model_len` to guarantee all requests eventually get served.

---

## 7b.4  The Prefill/Decode Interleave Tradeoff

`[DEEP DIVE]`

Mixing prefill and decode in the same forward pass sounds efficient but creates a **latency–throughput tradeoff** that is one of the most important tuning dimensions in production LLM serving.

### The conflict

**Decode** operations are *memory-bandwidth bound*: the GPU must load all model weights once per token, but does almost no arithmetic per weight. The bottleneck is HBM bandwidth.

**Prefill** operations are *compute bound*: processing many tokens simultaneously fills the GPU's ALUs. Prefill gets good hardware utilization.

When you mix them in the same batch, decode sequences "steal" memory bandwidth that could be used to serve prefill tokens faster. Conversely, large prefill chunks inflate the total step time, adding latency to *every* decode sequence in the batch.

```
Decode-only step (32 seqs, batch=32):
  Step time ≈ 25 ms
  Each sequence waits 25 ms for its next token

Mixed step (32 decode + 512-token prefill):
  Step time ≈ 35 ms   (prefill adds compute)
  Each decode sequence now waits 35 ms for next token
  TTFT (time-to-first-token) for new request: 35 ms

Pure prefill step (512-token prompt, no decode):
  Step time ≈ 12 ms
  TTFT: 12 ms
  But 32 running sequences stall completely for 12 ms
```

### The three strategies

```
Strategy          │ TTFT      │ Decode latency │ Throughput │ Use case
──────────────────┼───────────┼────────────────┼────────────┼──────────────────
Pure decode-first │ HIGH      │ LOW            │ MEDIUM     │ Chat, streaming
Pure prefill-first│ LOW       │ HIGH           │ MEDIUM     │ Batch processing
Chunked prefill   │ MEDIUM    │ MEDIUM         │ HIGH       │ Production default
```

**Chunked prefill** (enabled with `--enable-chunked-prefill`) splits long prompts into fixed-size chunks (e.g., 512 tokens) and processes one chunk per iteration alongside the active decode batch. This bounds the step-time inflation to a predictable constant.

```
Chunked prefill, chunk_size=512, prompt_len=2048:
  Iteration 1: 32 decode + 512 prefill tokens
  Iteration 2: 32 decode + 512 prefill tokens
  Iteration 3: 32 decode + 512 prefill tokens
  Iteration 4: 32 decode + 512 prefill tokens (last chunk)
  Iteration 5: 33 decode sequences (new request now running)

Each decode sequence experiences +3–8 ms step-time inflation
but TTFT is bounded by (4 iterations × 30 ms) = 120 ms
instead of a single 80-ms prefill step
```

---

## 7b.5  Preemption Accounting

When a sequence is preempted and later resumed, it must pay a cost. Understanding this cost matters for capacity planning.

### SWAP policy cost

```
Swap-out:
  Size = num_blocks × block_size × 2 (K+V) × head_dim × num_layers × dtype_bytes
  
  Example: Llama-3-8B, 16 blocks, block_size=16, 32 layers, BF16:
    16 × 16 × 2 × 128 × 32 × 2 = 67 MB
  
  At PCIe 4.0 (32 GB/s bidirectional):
    Swap-out time: 67 MB / 32 GB/s ≈ 2 ms
    Swap-in time:  67 MB / 32 GB/s ≈ 2 ms
  
  Total preemption cost: ~4 ms + scheduling delay
```

### RECOMPUTE policy cost

```
Recompute (re-prefill from scratch):
  Cost = prefill_flops(seq_len)
  
  At seq_len=512 on A100 (decode throughput ~2000 tok/s):
    Equivalent decode tokens: 512 prefill → ~10–20 effective decode steps
    Recompute latency: 512/2000 × prefill_ratio ≈ 256 ms at batch=1
    
  For short sequences (<64 tokens): RECOMPUTE is cheaper than SWAP
  For long sequences (>256 tokens): SWAP is almost always cheaper
```

**WORKED EXAMPLE 7b.3 — Preemption impact on P99 latency:**

```
WORKED EXAMPLE 7b.3 — Preemption at peak load
──────────────────────────────────────────────
Setup: 4×A100, Llama-3-70B, 100 concurrent users, GPU memory 88% full

Preemption rate at 88% utilization: ~2% of requests per minute
Average preempted seq length: 350 tokens

SWAP policy:
  Swap-out: 350/16 = 22 blocks → 22 × 16 × 2 × 128 × 80 × 2 = 286 MB
  Swap time: 286 / 64 GB/s ≈ 4.5 ms per preemption event
  P99 latency impact: +4.5 ms × 2% preemption rate ≈ negligible

RECOMPUTE policy:
  Reprefill 350 tokens: ≈ 175 ms
  P99 latency impact: +175 ms × 2% = +3.5 ms average, but individual
  requests experience the full +175 ms
  
Recommendation: use SWAP at high utilization to protect P99
──────────────────────────────────────────────────────────
```

---

## 7b.6  The `max_num_seqs` Ceiling

`max_num_seqs` sets the maximum number of sequences that can run simultaneously — the hard ceiling on batch size. It interacts with memory in a non-obvious way.

```
Memory consumed by max_num_seqs:
  KV cache per sequence at max_model_len:
    = max_model_len × num_layers × 2 × num_kv_heads × head_dim × dtype_bytes

  Example: Llama-3-8B, max_model_len=4096, 32 layers, 8 KV heads, BF16:
    = 4096 × 32 × 2 × 8 × 128 × 2 = 536 MB per sequence (worst case)

  For max_num_seqs=256:
    Worst-case KV: 256 × 536 MB = 137 GB  (more than a single A100!)

vLLM avoids this by using block-level allocation:
  Only allocated blocks are used, not max_model_len blocks per sequence
  Actual KV memory ≈ average_actual_length / max_model_len × worst_case
```

Setting `max_num_seqs` too high doesn't directly OOM — it just means the scheduler *could* admit more sequences than fit in memory, relying on the block manager to catch it. In practice, the block manager's watermark check prevents OOM. Setting it too low artificially limits throughput even when memory is available.

---

## 7b.7  Continuous Batching in llama.cpp

llama.cpp gained continuous batching support in mid-2024 via the `llama_batch` API. The mechanism differs from vLLM in one important way: llama.cpp uses **sequence IDs** to multiplex multiple sequences through the same context window.

```c
// llama.cpp multi-sequence batch (simplified)
llama_batch batch = llama_batch_init(max_tokens, 0, max_seqs);

// Add tokens from multiple sequences to one batch
for (int seq_id = 0; seq_id < n_active_seqs; seq_id++) {
    llama_token tok = next_token[seq_id];
    int pos = seq_positions[seq_id];
    
    // seq_id is the KV cache "lane" for this sequence
    llama_batch_add(batch, tok, pos, {seq_id}, /*logits=*/true);
}

// Single forward pass processes all sequences
llama_decode(ctx, batch);
```

The KV cache in llama.cpp is pre-allocated as a flat buffer:

```
llama.cpp KV layout (n_layer × n_ctx × n_kv_heads × head_dim):
  ┌────────────────────────────────────────────────────────┐
  │ Layer 0  │ pos=0 │ pos=1 │ ... │ pos=n_ctx-1          │
  │          │ seq 0  │ seq 0  │     │ seq 2               │
  └────────────────────────────────────────────────────────┘
  
  Positions are interleaved by sequence; each sequence has
  its own contiguous range within [0, n_ctx).
```

This differs fundamentally from vLLM's paged approach: llama.cpp statically divides the context window, while vLLM allocates blocks dynamically. The tradeoff: llama.cpp is simpler and lower-latency for small N; vLLM handles larger N with better memory efficiency.

```
Practical capacity comparison (Llama-3-8B, 8GB VRAM, max_ctx=2048):

llama.cpp, n_parallel=4:
  4 × 2048 context slots pre-allocated
  All 4 slots always consumed (even if sequences are short)
  Max concurrent: 4

vLLM, A100 80GB:
  Blocks allocated on demand
  Short sequences use fewer blocks
  Max concurrent: 128+ (depends on actual lengths)
```

---

## 7b.8  Measuring Scheduler Efficiency

Three metrics tell you whether the scheduler is performing well:

### Batch utilization

```
batch_utilization = mean(actual_batch_size) / max_num_seqs

Target: > 0.85
Below 0.5: scheduler is being too conservative — raise max_num_seqs
           or check if waiting queue is empty (traffic too low)
```

### Token throughput vs. theoretical

```
theoretical_throughput = max_num_batched_tokens / step_time_ms × 1000
actual_throughput      = tokens_generated / elapsed_seconds

ratio = actual / theoretical
Target: > 0.80
Below 0.60: CPU scheduling overhead, or prefill/decode imbalance
```

### Preemption rate

```
preemption_rate = preemptions / total_steps

Target: < 0.01 (1%)
Above 0.05: memory pressure — reduce max_num_seqs or max_model_len
```

vLLM exposes these via Prometheus metrics:
```
vllm:num_preemptions_total
vllm:gpu_cache_usage_perc
vllm:num_running_seqs
vllm:avg_generation_throughput_toks_per_s
```

---

## Chapter Summary

Continuous batching eliminates the GPU idle time inherent in static batching by replacing finished sequences with new ones after every single decode step. The iteration-level scheduling loop runs in under 1 ms and makes seven decisions per iteration: free finished sequences, check preemption, restore swapped sequences, admit new requests within the token budget, build the batch, execute the forward pass, and emit tokens. The token budget (`max_num_batched_tokens`) prevents any single large prefill from starving the decode path. Chunked prefill bounds step-time inflation by splitting long prompts across multiple iterations. Preemption costs 2–5 ms via SWAP policy or up to 175 ms via RECOMPUTE for 350-token sequences; SWAP is preferred at high utilization. llama.cpp implements a simpler version using static sequence slots rather than dynamic blocks, trading memory efficiency for implementation simplicity.

---

## Self-Check Questions

1. A batch has 16 running decode sequences and a waiting request with a 3000-token prompt. `max_num_batched_tokens=2048`. Will this request be admitted? If not, what happens next iteration?

2. You observe that `vllm:num_preemptions_total` is increasing at 50/minute on a system serving 200 concurrent users. The RECOMPUTE policy is active. What is the approximate P99 latency penalty per preempted request, if average preempted length is 200 tokens and prefill throughput is 3000 tok/s?

3. Compare the KV memory layout in llama.cpp vs. vLLM for 8 concurrent sequences of average length 512 tokens, max context 4096. Which engine uses less memory and by how much?

4. You enable chunked prefill with chunk_size=256 on a system receiving 500-token prompts. How many iterations does each new request spend in the prefill phase? What is the per-iteration step-time overhead compared to a decode-only step?

5. A production system has batch utilization of 0.42 with `max_num_seqs=64`. Traffic is high (waiting queue never empty). What is the most likely cause, and what flag should you adjust first?


---

## Worked Solutions

---

### Solution 1 — 3,000-token prompt with max_num_batched_tokens=2,048

**What we need:** Will this request be admitted? What happens next?

**Step 1 — Without chunked prefill.**

The scheduler tries to fit the entire 3,000-token prefill into one step. But:

$$3{,}000 > 2{,}048 \implies \text{Cannot admit — exceeds per-step token budget}$$

The request stays in the waiting queue. Next iteration: still 3,000 tokens — still blocked. The request will **never** be admitted without chunked prefill because its full prefill always exceeds the budget.

**Step 2 — With chunked prefill enabled.**

The scheduler splits the prefill:

- Step 1: 2,048 tokens of the prompt (leaving 952 tokens remaining)
- Step 2: remaining 952 tokens → prefill complete, decode begins

With chunked prefill: request is admitted immediately, first token generated after 2 steps.

**Step 3 — Practical recommendation.**

Always enable chunked prefill (`--enable-chunked-prefill`) when serving users with potentially long prompts. Without it, a single 3,000-token request can block the queue indefinitely if no step can accommodate it.

---

### Solution 2 — P99 latency penalty from preemptions (RECOMPUTE policy)

**Given:** 50 preemptions/min, average preempted length=200 tokens, prefill throughput=3,000 tok/s

**Step 1 — Recompute cost per preempted request.**

Under the RECOMPUTE policy, all KV blocks are freed and the sequence must redo its entire prefill from scratch when resumed:

$$\text{recompute time} = \frac{200 \text{ tokens}}{3{,}000 \text{ tok/s}} = 0.0667 \text{ s} = \textbf{66.7 ms}$$

**Step 2 — What the user experiences.**

The user's request was mid-decode at token 150 (for example). After preemption and recomputation:

- Prompt is re-processed (66.7 ms added)
- Decode resumes from where it left off

The additional 66.7 ms shows up as an elongated inter-token gap — the user sees the stream pause and then resume.

**Step 3 — Aggregate impact.**

50 preemptions/min × 66.7 ms = 3,333 ms of recompute per minute. If 200 concurrent users are active, each preemption affects one user's P99. A P99 spike of ~70 ms would be observable as a "stutter" in the streaming output.

**Mitigation:** Switch to SWAP policy (transfers KV to CPU instead of discarding) for workloads with expensive prefills, accepting the PCIe latency (~0.3 ms per 2.5 MB of KV) instead of the full recompute cost.

---

### Solution 3 — KV memory comparison: llama.cpp vs vLLM for 8 sequences × 512 tokens

**Assumptions:** 32 layers, 8 KV heads, head_dim=128, FP16; max_context=4,096; block_size=16

**vLLM (dynamic allocation):**

$$\text{blocks needed} = 8 \text{ seqs} \times \lceil 512/16 \rceil = 8 \times 32 = 256 \text{ blocks}$$
$$\text{bytes} = 256 \times 256 \text{ KB} = 65{,}536 \text{ KB} = \textbf{64 MB}$$

vLLM only allocates blocks for *actual* token count. 512 tokens × 8 sequences = actual usage.

**llama.cpp (contiguous preallocated):**

llama.cpp uses a single contiguous KV cache buffer sized for `n_ctx` (max context) × all sequences:

$$\text{bytes} = n_{\text{ctx}} \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times 2 \times 2$$
$$= 4{,}096 \times 32 \times 8 \times 128 \times 2 \times 2 = 536{,}870{,}912 \text{ bytes} = \textbf{512 MB}$$

This is for one continuous context. For 8 parallel sequences, llama.cpp needs 8 parallel contexts:

$$8 \times 512 \text{ MB} = \textbf{4,096 MB = 4 GB}$$

**Comparison:**

$$\frac{\text{llama.cpp}}{\text{vLLM}} = \frac{4{,}096 \text{ MB}}{64 \text{ MB}} = \textbf{64}\times \text{ more memory}$$

vLLM's PagedAttention uses 64× less memory for this workload. This is why vLLM can serve hundreds of concurrent users on a single GPU while llama.cpp server mode is better suited for smaller concurrent loads.

---

### Solution 4 — Chunked prefill iterations and step overhead for 500-token prompt

**Given:** chunk_size=256, prompt_length=500

**Step 1 — Number of prefill iterations.**

$$\text{iterations} = \lceil 500/256 \rceil = 2 \text{ iterations}$$

- Iteration 1: 256 tokens processed
- Iteration 2: 244 tokens processed → prefill complete → first decode step

**Step 2 — Per-iteration step-time overhead vs decode-only.**

A decode-only step processes `batch_size` tokens (e.g., 32 decode sequences × 1 token = 32 tokens). A chunked prefill step processes 256 prefill tokens + 32 decode tokens = 288 tokens.

The step-time overhead is roughly proportional to extra tokens:

$$\text{relative overhead} = \frac{288 - 32}{32} = \frac{256}{32} = 8\times \text{ more tokens per step}$$

In wall-clock time: prefill compute is cheaper per-token than decode for large batches (prefill is compute-bound, decode is memory-bound), but the extra 256 tokens still add ~0.5–2 ms to each step depending on the model size and hardware.

---

### Solution 5 — Low batch utilization (0.42) with full waiting queue

**Given:** Utilization=0.42, max_num_seqs=64, queue never empty

**Step 1 — What 0.42 utilization means.**

On average, only 42% of the 64 possible slots are active at any time = ~27 active sequences. With high traffic and a full queue, the system should be serving as many sequences as possible — yet it isn't.

**Step 2 — Most likely causes.**

*Primary cause:* **KV cache block exhaustion.** The GPU block pool is being depleted by the 64 sequences' combined KV usage, forcing preemptions or rejection of new requests. This keeps the *active* count lower than `max_num_seqs`.

*Secondary cause:* High prefill latency bottleneck. Large prefills occupy the token budget, preventing decode sequences from progressing, leading to stale output and users abandoning connections.

**Step 3 — Fix.**

Try these in order:

1. **Reduce `--gpu-memory-utilization`** (e.g., from 0.90 to 0.80): Allocates less memory to the KV pool but reduces OOM-induced preemptions. Counterintuitively, this can increase throughput by reducing preemption overhead.

2. **Reduce `--max-num-seqs`** to 48: Fewer concurrent sequences means each gets more KV blocks. Utilization may drop (fewer slots) but actual throughput (tokens/second) may increase.

3. **Enable chunked prefill**: Prevents large prefills from starving decode sequences, improving steady-state throughput.

4. **Profile with `--disable-log-stats false`**: Check `num_preemptions_total` and `gpu_cache_usage_perc` to confirm block exhaustion is the cause.

