# Chapter 11 — Prefill, Chunked Prefill, and Prompt Caching

> *"A long prompt is a gift to the model and a burden to the scheduler.
> The model gets more context. Every other request gets to wait.
> Chunked prefill is the peace treaty between long-context users
> and everyone else sharing the same GPU."*

---

## 11.0 Why This Chapter Matters

Everything so far has treated the prompt as a pre-existing fact — something
that has already been converted into KV vectors before decode begins.  That
assumption hides a major operational problem.

A 32 000-token RAG prompt takes roughly **800 ms** to prefill on a single
H100.  During that 800 ms, every other user in the batch receives zero new
tokens.  Their time-to-first-token (TTFT) spikes.  Their perceived latency
collapses.  At 50 000 concurrent users, this is not an edge case — it is
the dominant cause of p99 latency spikes in production.

This chapter covers the full prefill story:

**What you will understand after this chapter:**

- How prefill works mechanically and why it is compute-bound while decode is
  memory-bandwidth bound.
- Why a single long-context request can silently stall all other users.
- How chunked prefill splits long prompts across multiple scheduler steps,
  letting decode tokens run in the budget that remains.
- How vLLM's radix prefix cache (prefix caching) eliminates redundant prefill
  work for repeated prompt prefixes.
- What llama.cpp's `--ubatch-size` and `--cache-prompt` flags control, and
  how they map to the same concepts.

**What you need first:**

- Chapter 7 (the scheduler's batching loop and token budget model).
- Chapter 9 (the forward pass and the distinction between compute-bound and
  memory-bandwidth-bound workloads).
- Chapter 6 (block table and KV cache layout).

---

## 11.1 What Prefill Is  `[FOUNDATIONAL]`

Every LLM request begins with a **prompt** — a sequence of tokens the user
provides.  Before the model can generate the first new token, it must compute
the Key and Value vectors for every prompt token and store them in the KV cache.
This initial computation is called **prefill**.

```
PREFILL PHASE
─────────────────────────────────────────────────────────

Prompt tokens:  [T₁  T₂  T₃  T₄  T₅  T₆  T₇  T₈]   ← 8 tokens

In one forward pass (simplified):
  ┌──────────────────────────────────────────────┐
  │  For every token simultaneously:             │
  │   • Compute Q, K, V vectors                  │
  │   • Attend over all previous tokens (causal) │
  │   • Write K, V to KV cache                   │
  │   • Produce hidden state                      │
  └──────────────────────────────────────────────┘
  Output: KV cache filled for positions 0..7
          Hidden state at position 7 → first logit

DECODE PHASE (one token per step)
─────────────────────────────────────────────────────────

Step 1:  New token [T₉]  → read KV[0..7], attend, write KV[8], emit T₉
Step 2:  New token [T₁₀] → read KV[0..8], attend, write KV[9], emit T₁₀
Step N:  ...
```

The critical difference:

| Phase   | Input tokens per step | KV reads per step | Compute pattern      |
|---------|----------------------|-------------------|----------------------|
| Prefill | N (all at once)      | 0 (writes only)   | Compute-bound        |
| Decode  | 1                    | N accumulated     | Memory-bandwidth-bound |

During **prefill**, the GPU processes a large square-ish matrix (N tokens × d
dimensions), fully utilising tensor cores.  During **decode**, it processes
one row at a time, spending most of its time waiting for weight reads from HBM.

---

## 11.2 The Prefill Tax  `[FOUNDATIONAL]`

The prefill tax is the hidden cost that long-context requests impose on every
other request in the batch.

### 11.2.1 Why prefill monopolises the GPU

vLLM's scheduler, as of mid-2024, sends all pending prefill tokens in a single
forward pass when a request is first admitted.  That forward pass can occupy
the GPU for hundreds of milliseconds — during which no decode tokens are
generated for anyone else.

```
WORKED EXAMPLE 11.1 — Prefill Tax on Decode Latency
─────────────────────────────────────────────────────────────────────
Given:
  Model:          LLaMA 3 8B on H100 SXM5
  Prefill speed:  ~40 000 tokens/s  (H100 BF16, large batch)
  Decode speed:   ~3 500 tokens/s at batch=32
  Long request:   32 000-token RAG prompt
  Active batch:   32 users in decode, all waiting

Step 1 — Compute prefill time for the long request:
  32 000 tokens / 40 000 tokens·s⁻¹ = 0.80 s = 800 ms

Step 2 — How many decode tokens would the 32 waiting users have received?
  3 500 tokens·s⁻¹ × 0.80 s = 2 800 tokens
  Per user: 2 800 / 32 = 87.5 tokens

Step 3 — TTFT spike for decode users:
  Without the long request: ITL ≈ 28 ms (normal decode step)
  With the long request:    TTFT spike ≈ 800 ms
  Ratio: 800 / 28 ≈ 29×

Final answer:
  One 32K-token RAG request stalls 32 decode users for 800 ms each —
  roughly equivalent to missing 87 token opportunities per user.
  TTFT spikes 29× above normal.
─────────────────────────────────────────────────────────────────────
```

At 50 000 concurrent users (the LinkedIn scenario from Chapter 1), long-context
requests are not rare.  If 5% of traffic has prompts ≥ 8 000 tokens, the
scheduler sees a new long prefill request every few milliseconds.  The p99 TTFT
degrades continuously.

### 11.2.2 The asymmetry: prefill is compute-bound, decode is memory-bound

This is worth stating precisely because it drives every design decision in this
chapter.

```
PREFILL — compute-bound:

  Input shape: [N_prompt, d_model]   e.g. [32000, 4096]
  Weight shape: [d_model, d_model]         [4096, 4096]
  GEMM shape:  [32000, 4096] × [4096, 4096] → [32000, 4096]

  Arithmetic intensity = FLOPs / bytes moved
    FLOPs:  2 × 32000 × 4096 × 4096 = 1.07 TFLOP (per layer, QKV proj alone)
    Bytes:  weight read = 4096 × 4096 × 2 = 33.6 MB
    Intensity = 1.07 × 10¹² / 33.6 × 10⁶ ≈ 31 800 FLOP/byte

  H100 ridge point: ~148 (FP16 TFLOP/s / HBM bandwidth)
    = 989 × 10¹² / 3.35 × 10¹² = ~295 FLOP/byte

  31 800 >> 295 → deeply compute-bound

DECODE — memory-bandwidth-bound:

  Input shape: [B, 1, d_model]   e.g. [32, 1, 4096]  (batch × 1 token)
  Weight shape: [d_model, d_model]     [4096, 4096]
  GEMM shape:  [32, 1, 4096] × [4096, 4096] → [32, 1, 4096]

  FLOPs:  2 × 32 × 1 × 4096 × 4096 = 1.07 GFLOP (same weights, tiny input)
  Bytes:  weight read = 33.6 MB (same as prefill)
  Intensity = 1.07 × 10⁹ / 33.6 × 10⁶ ≈ 32 FLOP/byte

  32 << 295 → memory-bandwidth-bound
```

This asymmetry has a useful implication: **decode tokens are cheap to
interleave with chunked prefill tokens** because the two phases are bottlenecked
on different hardware resources.  Decode waits for HBM bandwidth; chunked
prefill waits for tensor core availability.  They do not fully compete.

---

## 11.3 Chunked Prefill  `[FOUNDATIONAL]`

Chunked prefill solves the prefill tax by breaking long prompts into
**chunks** — small groups of tokens processed one scheduler step at a time.
Decode tokens fill the remaining token budget in the same step.

### 11.3.1 The token budget model

The scheduler controls how many tokens execute in each forward pass via:

```
max_num_batched_tokens   ← total token budget per step
```

Without chunked prefill, a 32 000-token prompt consumes the entire budget in
one step.  With chunked prefill enabled:

```
vLLM config (enable_chunked_prefill=True):
  max_num_batched_tokens = 2048   ← example

Step 1:
  ┌─────────────────────────────────────────────┐
  │ Decode tokens:   32 × 1 =   32 tokens       │
  │ Prefill chunk:         2016 tokens (of 32K) │
  │ Total:                 2048 tokens           │
  └─────────────────────────────────────────────┘

Step 2:
  ┌─────────────────────────────────────────────┐
  │ Decode tokens:   32 × 1 =   32 tokens       │
  │ Prefill chunk:         2016 tokens           │
  └─────────────────────────────────────────────┘

...

Step 16:
  ┌─────────────────────────────────────────────┐
  │ Decode tokens:   32 × 1 =   32 tokens       │
  │ Prefill chunk (final):   2016 tokens        │
  │  → prefill complete; request enters decode  │
  └─────────────────────────────────────────────┘
```

The 32 existing decode users receive a new token every step throughout the
16-step prefill.  Their ITL is unaffected.  The long-context user's TTFT
increases (they wait 16 steps instead of 1 before generating their first
token), but the rest of the batch is protected.

### 11.3.2 Decode tokens go first

`[COMMON TRAP]` — A frequent mistake: assuming chunked prefill means alternating
decode and prefill fairly.  In practice, **decode tokens always have priority**.

The scheduler allocates the token budget as follows:
1. Count the decode tokens needed: `B_decode = running_seqs × 1`.
2. Remaining budget: `B_prefill = max_num_batched_tokens - B_decode`.
3. Take the next `B_prefill` tokens from the front of the current prefill queue.

If the running batch is large, decode tokens alone may consume most or all of
the budget, leaving very little for prefill.  This is intentional — it keeps
inter-token latency (ITL) predictable for active users.

```
WORKED EXAMPLE 11.2 — Token Budget Allocation
─────────────────────────────────────────────────────────────────────
Given:
  max_num_batched_tokens = 4096
  Running decode sequences = 120  (each produces 1 decode token)
  New request: 12 000-token RAG prompt, just admitted

Step 1 budget allocation:
  Decode tokens:  120 × 1 = 120
  Prefill budget: 4096 − 120 = 3976 tokens of prompt
  Prefill chunk:  tokens 0–3975 of the 12 000-token prompt

Step 2 budget allocation:
  Running sequences now = 121 (original 120 + new request in prefill)
  But new request is still in prefill, not yet in decode
  Decode tokens:  120 × 1 = 120   (new user not in decode yet)
  Prefill budget: 4096 − 120 = 3976 tokens of prompt
  Prefill chunk:  tokens 3976–7951

Step 3 budget allocation:
  Prefill chunk:  tokens 7952–11927

Step 4 budget allocation:
  Remaining prompt: 12000 − 11928 = 72 tokens
  Prefill chunk:  tokens 11928–11999  (final 72 tokens)
  Decode budget:  4096 − 72 − 120 = 3904 tokens spare (unused this step)
  → New request enters decode queue for step 5

Summary:
  Long-context user TTFT:  4 steps × ~15 ms/step = ~60 ms
  (vs. 12000 / 40000 × 1000 = 300 ms without chunked prefill)
  TTFT reduction: 5× improvement
  Existing 120 decode users: unaffected throughout
─────────────────────────────────────────────────────────────────────
```

### 11.3.3 ASCII diagram — token budget bar

```
max_num_batched_tokens = 4096
│◄──────────────────────── 4096 tokens ──────────────────────────►│

Without chunked prefill (one long request):
├──────────────────────────────────────────────────────────────────┤
│  32K-token prefill chunk occupies entire budget (if ≤ 4096)     │
│  OR multiple steps if budget < 32K — but decode blocked per step │
└──────────────────────────────────────────────────────────────────┘

With chunked prefill (enable_chunked_prefill=True):
Step 1:
├────────┬─────────────────────────────────────────────────────────┤
│ decode │                   prefill chunk                         │
│  120 T │                    3976 T                               │
└────────┴─────────────────────────────────────────────────────────┘

Step 2:
├────────┬─────────────────────────────────────────────────────────┤
│ decode │                   prefill chunk                         │
│  120 T │                    3976 T                               │
└────────┴─────────────────────────────────────────────────────────┘

Step 4 (final chunk, only 72 tokens remain):
├────────┬──────┬──────────────────────────────────────────────────┤
│ decode │prefil│           (budget unused / other requests)       │
│  120 T │ 72 T │                   3904 T spare                   │
└────────┴──────┴──────────────────────────────────────────────────┘

decode users: receive token every step ✓
new user TTFT: 4 steps × 15ms = 60ms (vs. 300ms) ✓
```

### 11.3.4 vLLM configuration

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    enable_chunked_prefill=True,
    max_num_batched_tokens=4096,   # token budget per step
    max_num_seqs=256,              # max concurrent sequences
)
```

The two parameters interact:

```
max_num_batched_tokens   ← controls chunk size (prefill portion)
max_num_seqs             ← controls how many sequences compete for the budget

If max_num_seqs = 256 and each is in decode:
  Decode tokens per step = 256
  Prefill budget = max_num_batched_tokens − 256

Setting max_num_batched_tokens too low (< max_num_seqs) causes the prefill
budget to go negative — the scheduler silently clamps it to 0 and stalls
prefill indefinitely.  Always ensure:

  max_num_batched_tokens ≥ max_num_seqs + minimum_useful_chunk_size
```

`[COMMON TRAP]` — Setting `enable_chunked_prefill=True` with a very small
`max_num_batched_tokens` (e.g., 256) and a large `max_num_seqs` (e.g., 512)
will leave **zero budget for prefill**.  New requests queue indefinitely.
A safe rule of thumb: `max_num_batched_tokens ≥ 4 × max_num_seqs`.

---

## 11.4 Prompt Caching — vLLM's Radix Prefix Cache  `[FOUNDATIONAL]`

Chunked prefill reduces the *per-request* prefill cost.  Prefix caching
eliminates it **entirely** for repeated prompt prefixes.

### 11.4.1 The repeated-prefix observation

Most production LLM deployments share large prompt prefixes across requests:

```
System prompt (shared by all users):
  "You are a helpful customer service agent for Acme Corp.
   You have access to the following product catalog:
   [5 000-token catalog]
   Always respond in a professional tone. ..."

User message (unique per request):
  "What is the return policy for order #48291?"
```

If the 5 000-token system prompt + catalog has already been processed for a
previous request, there is no reason to recompute its KV vectors again.  They
are deterministic: the same tokens at the same positions always produce the
same K and V vectors (assuming no randomness in the model weights, which is
true).

### 11.4.2 How the radix prefix cache works

vLLM implements a **radix tree** (trie) over token sequences.  Each node in
the tree represents a KV block (Chapter 6).  When a new request arrives:

1. The scheduler hashes the prompt tokens, block by block.
2. It walks the radix tree, looking for an existing node whose token sequence
   matches the prefix of the new request's prompt.
3. **Cache hit:** The matched KV blocks are **reused** — the request skips
   prefill for those tokens entirely.
4. **Cache miss:** Normal prefill proceeds; the new KV blocks are added to
   the radix tree for future reuse.

```
RADIX TREE EXAMPLE
────────────────────────────────────────────────────────────────────

Request 1:  "System prompt [5K tokens] | User: What is the return policy?"
  → Prefill all 5K+ tokens; store in radix tree:
    Root → [block 0: tokens 0–15] → [block 1: tokens 16–31] → ... → [block 312: tokens 4992–5007]
    Leaf node: "return policy" query tokens (unique, not shared)

Request 2:  "System prompt [5K tokens] | User: How do I track my order?"
  → Radix walk: blocks 0–312 match! (same system prompt)
  → Cache hit for 5 000 tokens → skip prefill for those blocks
  → Only prefill "How do I track my order?" (8 tokens)
  → TTFT: 8/40000 × 1000 = 0.2 ms   (vs. 125 ms without caching)
```

```
RADIX TREE STRUCTURE (ASCII)
────────────────────────────────────────────────────────────────────

Root
 └─ [Block 0: "You are a helpful customer"]   ← shared prefix
      └─ [Block 1: "service agent for Acme"]
           └─ [Block 2: "Corp. You have access"]
                └─ ...
                     └─ [Block 312: "professional tone..."]
                          ├─ [Block 313a: "return policy"]  ← Req 1 leaf
                          ├─ [Block 313b: "track my order"] ← Req 2 leaf
                          └─ [Block 313c: "invoice status"] ← Req 3 leaf
```

The tree grows lazily.  Blocks are evicted under LRU pressure (same eviction
policy as the block pool from Chapter 6).  The radix tree is the index; the
KV blocks themselves live in the GPU block pool.

### 11.4.3 Hit rate arithmetic

```
WORKED EXAMPLE 11.3 — Prefix Cache Hit Rate and Cost Savings
─────────────────────────────────────────────────────────────────────
Given:
  System prompt length:   5 000 tokens (shared by all requests)
  Average user message:     50 tokens  (unique per request)
  Average prompt total:   5 050 tokens
  Hit rate (after warmup): 95%  (5% cold-start misses)
  Prefill cost:           $0.01 per 1K input tokens (compute cost)
  Daily requests:         100 000

Step 1 — Cost without prefix caching:
  Tokens per request: 5 050
  Daily cost: 100 000 × 5050 / 1000 × $0.01 = $5 050/day

Step 2 — Cost with prefix caching (95% hit rate):
  Cold misses (5%): 5 000 × 5050/1000 × $0.01 = $252.50/day
  Hot hits (95%):   only user message prefilled
    95 000 × 50/1000 × $0.01 = $47.50/day
  Total: $252.50 + $47.50 = $300/day

Step 3 — Savings:
  Reduction: ($5050 − $300) / $5050 = 94.1% compute cost reduction
  TTFT for hit requests: 50/40000 × 1000 = 1.25 ms  (vs. 126 ms)

Final answer:
  95% prefix hit rate → 94% cost reduction, 100× TTFT improvement on hits.
─────────────────────────────────────────────────────────────────────
```

### 11.4.4 What makes a good cache key

The radix tree hashes **token IDs**, not text strings.  This means:

- Whitespace differences in the prompt produce different token IDs → cache miss.
- Adding one token to a shared prefix invalidates all descendant blocks.
- Chat templates that prepend role markers (e.g., `<|system|>`) must be
  included in the prefix for the cache to work correctly.

`[COMMON TRAP]` — If your system prompt ends with a random request ID or
timestamp, prefix caching provides zero benefit — every request has a unique
prefix by definition.  Strip session-specific data from the shared prefix.

### 11.4.5 Enabling prefix caching in vLLM

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    enable_prefix_caching=True,
    enable_chunked_prefill=True,   # combine for maximum effect
    max_num_batched_tokens=4096,
)
```

Starting in vLLM 0.4.x, `enable_prefix_caching` is enabled by default when
using the OpenAI-compatible server.  The Prometheus metric
`vllm:gpu_prefix_cache_hit_rate` measures real-time hit rate.

---

## 11.5 RadixAttention and Global Prefix Trees  `[DEEP DIVE]`

SGLang introduced **RadixAttention** (Zheng et al., 2023), which extends
prefix caching beyond a single engine instance.

### 11.5.1 The single-engine limit

vLLM's radix tree lives in one process's memory.  If you run 8 vLLM workers
behind a load balancer, each worker has its own cache.  A request routed to
worker 3 cannot reuse KV blocks computed by worker 7 for the same system
prompt.

The effective hit rate across the fleet degrades with the number of workers:

```
Effective hit rate (sticky routing) ≈ hit_rate_per_worker = H
Effective hit rate (round-robin)    ≈ H / num_workers  (approximation)
```

With 8 workers and 95% per-worker hit rate and round-robin routing,
the fleet-level hit rate falls toward ~12%.

### 11.5.2 Global prefix trees (BatchLLM, SGLang)

Global prefix trees maintain a **cross-instance radix tree** backed by a
shared KV store (Redis, custom RDMA store, or disaggregated KV tier from
Chapter 18).

- All workers share the same prefix tree index.
- A KV cache hit redirects the request to the worker holding those blocks
  (or fetches them over RDMA).
- Hit rates of 78–91% have been reported in production RAG workloads.

vLLM's disaggregated KV transfer (introduced in v0.6.x) provides the
building blocks for this pattern, though a full cross-instance radix tree
is currently a research/early-production feature.

### 11.5.3 Prefix caching vs. KV cache quantization

Prefix caching stores KV blocks in whatever precision the model uses (BF16
by default).  When combined with INT8 KV cache quantization (Chapter 10 §10.7),
the cached blocks are stored in INT8, halving the radix tree's HBM footprint.

```
Block pool size with prefix caching + INT8 KV:
  = total_gpu_memory × gpu_memory_utilization
    − model_weights
    − peak_activations
    (same formula as Chapter 2, but all KV blocks are INT8)

Effect: ~2× more blocks available in the radix tree
→ higher hit rates under memory pressure
```

---

## 11.6 llama.cpp Equivalents  `[FOUNDATIONAL]`

llama.cpp has no scheduler and no radix tree.  Its equivalents are simpler
but still useful for single-user and small-deployment scenarios.

### 11.6.1 `--ubatch-size` — micro-batch prefill chunking

```
llama.cpp flag:   --ubatch-size N   (default: 512)

Context:
  --batch-size B    ← logical batch (max tokens submitted per decode call)
  --ubatch-size N   ← physical micro-batch (max tokens per GPU kernel call)

When N < B, llama.cpp internally splits the logical batch into chunks
of N tokens each during prefill:

  Logical batch: 4096 tokens (one long prompt)
  ubatch-size:   512

  Kernel call 1: tokens 0–511     → KV written to positions 0–511
  Kernel call 2: tokens 512–1023  → KV written to positions 512–1023
  ...
  Kernel call 8: tokens 3584–4095 → KV written to positions 3584–4095
```

This is **micro-batching**, not interleaved chunked prefill.  Unlike vLLM's
chunked prefill, llama.cpp does not interleave decode tokens between prefill
chunks — it completes all prefill chunks before starting decode.  The benefit
is reduced peak GPU memory, not improved latency for concurrent users.

### 11.6.2 `--cache-prompt` — simple prefix match caching

```
llama.cpp flag:   --cache-prompt   (llama-server only, default: off)

Mechanism:
  On each new request, the server compares the incoming prompt token IDs
  against the current session's KV cache contents.

  Matching prefix:  reuse existing KV blocks (no recompute)
  Non-matching:     clear KV cache from first mismatch; recompute from there

This is a LINEAR PREFIX MATCH, not a radix tree:
  ✓ Same prompt + same session → full cache hit
  ✓ Prompt extended by a few tokens → partial hit
  ✗ Different requests with shared prefix → miss (no cross-request sharing)
  ✗ Out-of-order system prompts → miss
```

### 11.6.3 Comparison table

```
┌─────────────────────────┬────────────────────────┬─────────────────────────┐
│ Feature                 │ vLLM                   │ llama.cpp               │
├─────────────────────────┼────────────────────────┼─────────────────────────┤
│ Chunked prefill         │ enable_chunked_prefill │ --ubatch-size N         │
│                         │ (interleaves decode)   │ (pure micro-batching)   │
├─────────────────────────┼────────────────────────┼─────────────────────────┤
│ Prefix caching          │ enable_prefix_caching  │ --cache-prompt          │
│                         │ (radix tree, cross-req)│ (linear match, 1 session│
├─────────────────────────┼────────────────────────┼─────────────────────────┤
│ Multi-user benefit      │ Yes — all users share  │ No — per-session only   │
├─────────────────────────┼────────────────────────┼─────────────────────────┤
│ Hit rate at scale       │ 50–95% (fleet)         │ ~100% within session    │
├─────────────────────────┼────────────────────────┼─────────────────────────┤
│ Eviction policy         │ LRU (block pool shared)│ Clear on mismatch       │
├─────────────────────────┼────────────────────────┼─────────────────────────┤
│ Storage                 │ GPU HBM (block pool)   │ GPU VRAM (context buf.) │
└─────────────────────────┴────────────────────────┴─────────────────────────┘
```

### 11.6.4 The llama.cpp multi-session workaround

For multi-user serving with llama.cpp (e.g., running a custom server with
`llama_decode`), developers can simulate prefix caching by:

1. Maintaining a fixed "base context" pre-loaded with the system prompt.
2. Forking the context state (copying the KV ring buffer) for each user.
3. Restoring the base context after each user's session ends.

This gives ~100% hit rate for the shared prefix at the cost of memory
proportional to `num_users × ctx_size` (each fork holds a full context copy).
It does not scale beyond a few dozen users.

---

## 11.7 TTFT vs. Throughput Trade-off  `[DEEP DIVE]`

Chunked prefill is not free.  Splitting a large prefill into chunks increases
total prefill FLOP count slightly and adds scheduler overhead per chunk.

### 11.7.1 The throughput cost of chunked prefill

When a prefill is chunked across 16 steps, the attention computation for each
chunk must handle a **causal mask** that grows step by step.  More importantly,
smaller attention matrices are less efficient on GPU tensor cores.

```
WORKED EXAMPLE 11.4 — Prefill Throughput Degradation
─────────────────────────────────────────────────────────────────────
Given:
  Prompt:               4096 tokens
  max_num_batched_tokens: 4096 (large budget)

Without chunking (all 4096 tokens in one pass):
  Attention: [4096 × 4096] matrix → tensor cores highly utilized
  FLOP efficiency: ~85% of peak (large square GEMM)
  Prefill time: 4096 / 40000 = 102 ms

With chunking (chunk_size = 512):
  8 chunks of 512 tokens each
  Chunk 1: [512 × 512] attention → smaller GEMM, lower efficiency
  FLOP efficiency: ~55% of peak (small matrix)
  Per-chunk time: 512 / 22000 ≈ 23 ms  (lower throughput at small size)
  Total: 8 × 23 = 184 ms

Throughput cost: 184 / 102 ≈ 1.8× longer total prefill time
Benefit: decode users see continuous token flow throughout
─────────────────────────────────────────────────────────────────────
```

The trade-off is explicit: **chunked prefill improves TTFT for decode users
at the cost of higher total prefill latency for the long-context user**.

### 11.7.2 Choosing `max_num_batched_tokens`

Larger chunks (larger `max_num_batched_tokens`) reduce the throughput penalty
but increase per-step latency (worse decode ITL).  The sweet spot depends on:

```
Workload type                     │ Recommended max_num_batched_tokens
──────────────────────────────────┼───────────────────────────────────
Chat (short prompts, ≤ 512 T)    │ 2048–4096
RAG (medium prompts, 1–8K T)     │ 4096–8192
Long-context (≥ 32K T)           │ 8192–32768
Batch / offline (no TTFT SLA)    │ 32768–unlimited (chunking off)
```

For latency-sensitive workloads, `max_num_batched_tokens = 4096` with
`enable_chunked_prefill=True` is a robust starting point.

---

## 11.8 Observing Prefill vs. Decode in Prometheus  `[DEEP DIVE]`

vLLM exposes prefill/decode metrics via its Prometheus endpoint:

```
# Prefill throughput (tokens/s entering KV cache):
vllm:num_prefill_tokens_total

# Decode throughput (tokens/s generated):
vllm:num_generation_tokens_total

# TTFT histogram (seconds, for each completed request):
vllm:time_to_first_token_seconds_bucket

# Chunked prefill queue depth:
vllm:num_requests_waiting

# Prefix cache hit rate:
vllm:gpu_prefix_cache_hit_rate
```

A healthy production system shows:

```
Ratio: prefill_tokens / generation_tokens ≈ prompt_length / output_length

For chat (average 200 prompt / 300 output):
  ratio ≈ 0.67 (more decode than prefill) → good utilization

For RAG (average 8000 prompt / 200 output):
  ratio ≈ 40 (prefill-dominated) → watch for TTFT spikes
  → enable chunked prefill + prefix caching

TTFT p99 > 10× TTFT p50 → prefill monopolization detected
  → reduce chunk size, enable chunked prefill
```

---

## 11.9 End-to-End Example — Long RAG Request  `[FOUNDATIONAL]`

Putting it all together: a 16 000-token RAG request arriving into a vLLM
deployment with 64 active decode users.

```
Configuration:
  model:                   LLaMA 3 8B
  enable_chunked_prefill:  True
  enable_prefix_caching:   True
  max_num_batched_tokens:  4096
  max_num_seqs:            128

Incoming request:
  System prompt:  10 000 tokens (shared system prompt, in cache)
  Retrieved docs:  5 000 tokens (unique to this request)
  User question:    100 tokens

Step 0 — Prefix cache check:
  System prompt hash matches → 10 000 tokens recovered from radix tree
  Remaining to prefill: 5 100 tokens (docs + question)

Step 1 — Scheduler allocates token budget:
  Decode tokens:   64 × 1 = 64
  Prefill budget:  4096 − 64 = 4032 tokens
  Prefill chunk:   tokens 0–4031 of the 5100-token remainder

Step 2 — Second chunk:
  Decode tokens:   64 (existing users) + 0 (new user still in prefill)
  Prefill budget:  4096 − 64 = 4032
  Prefill chunk:   tokens 4032–5099 (final 1068 tokens)
  → Prefill complete; new request enters decode

Step 3 onwards:
  New user is now in decode along with 64 existing users (total 65)
  All 65 users receive decode tokens every step

Timeline for the new user:
  Cache lookup:    <1 ms   (radix tree walk)
  Step 1 prefill:  ~15 ms  (4032 tokens at chunk rate)
  Step 2 prefill:  ~15 ms  (1068 tokens + 64 decode)
  TTFT:           ~30 ms   (2 steps × 15 ms)

Without prefix caching + chunked prefill:
  Full prefill: 15100 tokens in one pass = 377 ms
  64 decode users stalled for 377 ms each
  TTFT for new user: 377 ms
```

```
BEFORE (no chunked prefill, no prefix caching):
─────────────────────────────────────────────────────────────────────
Step 1: ████████████████████████████████████████████ 15100 T prefill
        (decode users BLOCKED for 377 ms)

AFTER (chunked prefill + prefix caching):
─────────────────────────────────────────────────────────────────────
Prefix:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 10000 T CACHED (0ms)
Step 1: ████ 64D │███████████████████████████████ 4032 T chunk 1
Step 2: ████ 64D │█████████ 1068 T chunk 2 (done)
Step 3+: ████ 65D decode (all users, including new)

TTFT: 30 ms (13× faster)  |  Existing users: uninterrupted ✓
```

---

## 11.10 Code Listing  `[FOUNDATIONAL]`

See `code/chapter_11/chunked_prefill_demo.py` for:
- TTFT benchmarks comparing unchunked vs. chunked prefill across prompt lengths
- Radix prefix cache hit rate simulator with configurable workload distributions
- Token budget allocation visualiser showing decode vs. prefill split per step
- vLLM prefix cache metric scraper using the Prometheus endpoint

See `code/chapter_11/chunked_prefill_demo.cpp` for:
- llama.cpp micro-batch (ubatch) simulation — split large prompt into N chunks
- KV cache reuse tracker comparing `--cache-prompt` hit/miss across sessions
- Prefill vs. decode compute intensity analysis matching §11.2.2

---

## 11.11 Summary

```
Key takeaways:

1. Prefill is compute-bound; decode is memory-bandwidth-bound.
   A single 32K-token prefill can stall all decode users for 800 ms.
   This is the prefill tax.

2. Chunked prefill splits long prompts across multiple scheduler steps.
   Decode tokens always go first, consuming their share of the token
   budget. The prefill chunk fills the remainder.

3. Token budget rule:
     decode_tokens = num_running_seqs × 1
     prefill_chunk = max_num_batched_tokens − decode_tokens
   Ensure max_num_batched_tokens ≥ 4 × max_num_seqs to avoid
   starving the prefill queue.

4. vLLM's radix prefix cache eliminates repeated prefill work.
   KV blocks for matching token prefixes are reused directly.
   At 95% hit rate and a 5000-token system prompt, the compute
   cost per request falls by 94%.

5. Chunked prefill has a throughput cost: smaller chunks have lower
   tensor core utilization. The prefill takes 1.5–2× longer in
   total wall time, but existing users never see a stall.

6. llama.cpp's --cache-prompt does linear prefix matching per session;
   --ubatch-size controls micro-batch chunk size for prefill.
   Neither feature provides cross-request prefix sharing.

7. In production: combine enable_chunked_prefill=True with
   enable_prefix_caching=True. Together they cut both the latency
   impact and the compute cost of long prompts.
```

---

## 11.12 Self-Check Questions

1. A deployment has `max_num_batched_tokens=2048` and `max_num_seqs=512`.
   What happens to new requests? How would you fix it?

2. Your system prompt is 8 000 tokens.  You observe prefix cache hit rate
   of 2% despite repeated requests.  Name two likely causes.

3. A user asks why their 32K RAG request has a TTFT of 15 seconds despite
   chunked prefill being enabled.  What would you check first?

4. Explain why the first chunk of a chunked prefill has lower GPU efficiency
   than a full unchunked prefill of the same total token count.

5. A colleague proposes running llama.cpp with `--cache-prompt` for a
   production chatbot serving 1 000 concurrent users.  What are the
   limitations versus vLLM's prefix caching?

---

## Where We Go Next

Chapter 12 covers **sampling** — what happens after the forward pass produces
the logit vector.  Temperature, top-k, top-p, min-p, repetition penalties,
and structured output (JSON grammars) are all implemented at this stage.
Sampling is fast (microseconds), but its configuration decisions determine
output quality, diversity, and safety.  We also look at beam search and how
it interacts with the copy-on-write KV cache blocks from Chapter 6.

*Next: Chapter 12 — Sampling: From Logits to Tokens*


---

## Chapter Summary

- **Prefill vs decode**: prefill processes all prompt tokens at once in an arithmetic-intensity-high matmul; decode generates one token per step in a bandwidth-bound operation.
- **Prompt caching (prefix caching)**: if the leading tokens of a new request match a previously cached sequence, vLLM skips the prefill FLOPs for those tokens entirely.

> **LinkedIn Scenario Update:** LinkedIn's product uses standardized system prompts — job recommendation context, profile summaries, search filters — that are nearly identical across users. At a conservative 70% system-prompt reuse rate across 50K concurrent users, prefix caching eliminates 70% of prefill FLOPs for the majority of requests. A typical 500-token system prompt prefill that takes 180ms at 28% GPU utilization drops to ~54ms after caching (the remaining 30% unique tokens still run). This reduction in TTFT is directly visible to users as faster "first word" latency, and the freed prefill compute capacity absorbs more requests, pushing effective throughput up by roughly 2.3× without any hardware change.
- **Radix tree for cache lookup**: vLLM stores cached block sequences in a radix tree keyed by token IDs; longest-prefix matching is O(prefix length) lookup time.
- **KV transfer in disaggregated prefill**: the prefill result (KV tensors) must be moved from the prefill pod to the decode pod over InfiniBand or NVLink fabric.
- **Chunked prefill**: splits a long prompt across multiple scheduler steps, interleaving prefill chunks with decode steps to keep TTFT and ITL both bounded.
- **Prefill arithmetic intensity**: at batch size 1, prefill arithmetic intensity is ~T × d_model FLOPs/byte; much higher than decode, so prefill is compute-bound on modern GPUs.

---

## Self-Check Questions

1. A 4 096-token prompt hits the prefix cache for the first 3 072 tokens. How many prefill FLOPs are saved as a fraction of the full prefill? Assume attention FLOPs dominate and scale as O(T²). *(Section 11.2)*

2. Two requests arrive simultaneously. Request A has a 2 048-token system prompt. Request B has the same 2 048-token system prompt plus 512 unique tokens. With prefix caching enabled, how many KV blocks does B need to compute from scratch? (Block size = 16 tokens.) *(Section 11.2)*

3. Chunked prefill breaks a 4 096-token prompt into 8 chunks of 512 tokens. How does this change TTFT compared to a single-pass prefill? How does it change ITL for simultaneously running decode sequences? *(Section 11.3)*

4. vLLM's radix tree evicts the LRU (least recently used) leaf node when GPU blocks are exhausted. Why evict leaves rather than internal nodes? What would break if internal nodes were evicted first? *(Section 11.2)*

5. In disaggregated prefill, the KV tensors for a 2 048-token prefill at 32 layers, 32 KV heads, d_k = 128, in BF16 must be transferred over InfiniBand at 400 Gb/s. Compute the transfer time in milliseconds. *(Section 11.4)*
