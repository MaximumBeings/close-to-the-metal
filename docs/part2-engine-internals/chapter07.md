# Chapter 7 — The Scheduler and Request Lifecycle

> *"The GPU is always busy. The question is whether it is busy doing useful work
> or busy waiting for the scheduler to decide what to feed it."*

---

## 7.0 Why This Chapter Matters

Chapter 6 showed how PagedAttention turns GPU HBM into a pool of reusable
blocks.  This chapter shows who controls that pool and how requests move through
the system from the moment they arrive to the moment their last token is streamed
to the caller.

The scheduler is the brain of vLLM.  It runs once per iteration (~every 20 ms
for a typical decode batch) and makes three interlocking decisions:

1. **Which waiting requests to admit** (memory budget check).
2. **Which running requests to keep** (or preempt to free memory).
3. **How to package the survivors** into a single GPU kernel call.

Getting these decisions wrong costs throughput, latency, or both.  Getting them
right is what lets vLLM serve 8× more requests than a naïve implementation on
the same hardware.

By the end of this chapter you will be able to:

- Trace a request through all five lifecycle states in vLLM.
- Explain the 7-step scheduling loop step by step.
- Calculate whether a new request can be admitted given current memory pressure.
- Distinguish the two preemption policies and choose between them.
- Understand `max_num_seqs`, `max_num_batched_tokens`, and `max_model_len` as
  the three dials that govern scheduler behavior.

- Build a priority-aware multi-request scheduler for llama.cpp from scratch.

---

## 7.1 Request Lifecycle  `[FOUNDATIONAL]`

### 7.1.1 Five states

Every request in vLLM passes through five states:

```
  WAITING ──admit──▶ RUNNING ──finish──▶ FINISHED_STOPPED
                        │
               preempt (no memory)
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
          SWAPPED              (RECOMPUTE)
         (CPU blks)           (blks freed)
              │
          swap_in (memory freed)
              │
              ▼
           RUNNING
```

| State | Meaning | KV blocks held? |
|-------|---------|-----------------|
| `WAITING` | Queued; no blocks allocated | No |
| `RUNNING` | Actively scheduled this iteration | Yes (GPU) |
| `SWAPPED` | Preempted; blocks on CPU | Yes (CPU) |
| `FINISHED_STOPPED` | Generation ended naturally (`<eos>`) | No |
| `FINISHED_ABORTED` | Cancelled by client | No |

### 7.1.2 The SequenceGroup

vLLM's internal unit is not a request but a **SequenceGroup** — a request that
may contain multiple *parallel* outputs (e.g., `n=4` for sampling four
completions, or `best_of=4` for best-of-N sampling).

```
SequenceGroup
  ├── request_id: "req_42"
  ├── sampling_params: {temperature=0.8, max_tokens=256, n=2}
  ├── arrival_time: 1719000000.123
  └── seqs: [Sequence_0, Sequence_1]     ← two parallel outputs

Sequence_0
  ├── seq_id: 0
  ├── status: RUNNING
  ├── block_table: [phys_3, phys_7, phys_11]
  ├── token_ids: [1, 450, 3796, ...]     ← tokens generated so far
  └── logical_token_count: 48
```

Beam search (Chapter 6 §6.5) creates SequenceGroups with `best_of` sequences
that share prefix blocks.

---

## 7.2 The 7-Step Scheduling Loop  `[DEEP DIVE]`

The scheduler's `schedule()` method is called once per iteration.  Here are
the seven logical steps, which we will examine in detail.

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1  Evict finished sequences                               │
│  STEP 2  Try to swap in preempted sequences (SWAPPED → RUNNING) │
│  STEP 3  Continue running sequences (budget check per seq)      │
│  STEP 4  Preempt if over budget (RUNNING → SWAPPED / recompute) │
│  STEP 5  Admit new requests (WAITING → RUNNING)                 │
│  STEP 6  Build SchedulerOutputs (prefill + decode lists)        │
│  STEP 7  Execute GPU kernels, sample, update state, stream      │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2.1 Step 1 — Evict finished sequences

Any sequence in RUNNING whose last generated token was `<eos>` or that has
reached `max_tokens` is moved to FINISHED and its blocks are freed.  This
happens before any admission decision so the freed blocks are immediately
available.

```python
for seq_group in list(self.running):
    for seq in seq_group.seqs:
        if seq.is_finished():
            self.block_manager.free(seq)
            seq_group.remove_seq(seq)
    if seq_group.is_finished():
        self.running.remove(seq_group)
```

### 7.2.2 Step 2 — Try swap-ins

Sequences in SWAPPED are ordered by arrival time.  For each, the scheduler
checks whether enough GPU blocks are available to swap them back in.  Swap-ins
are attempted greedily until the block budget is consumed.

```python
for seq_group in self.swapped[:]:   # copy: we mutate during iteration
    if self.block_manager.can_swap_in(seq_group):
        self.block_manager.swap_in(seq_group)
        self.swapped.remove(seq_group)
        self.running.append(seq_group)
    else:
        break   # no point trying smaller groups — memory is tight
```

**Key insight**: swap-ins are prioritized over admitting new requests because
swapped sequences have already paid their prefill cost.  Re-admitting them is
cheaper than starting fresh.

### 7.2.3 Step 3 — Reserve decode slots for running sequences

For each currently-running sequence, `append_slot` must succeed before the
iteration proceeds.  The scheduler pre-checks whether each sequence will need
a new block this iteration:

```python
blocks_to_copy: Dict[int, int] = {}   # CoW: old_phys → new_phys

for seq_group in self.running:
    for seq in seq_group.seqs:
        ret = self.block_manager.append_slot(seq)
        if ret is not None:
            old_id, new_id = ret
            if new_id is not None:
                blocks_to_copy[old_id] = new_id   # schedule CoW memcpy
```

### 7.2.4 Step 4 — Preempt if over budget

If Step 3 cannot complete (no free blocks for a running sequence) the scheduler
must preempt.  It preempts the lowest-priority running sequence first (last-in,
first-out by arrival time in the default policy):

```python
while not self._can_append_slot_for_all():
    victim = self.running[-1]   # lowest priority (LIFO default)
    if self.block_manager.can_swap_out(victim):
        self.block_manager.swap_out(victim)
        self.running.remove(victim)
        self.swapped.insert(0, victim)     # front of swapped queue
    else:
        # No CPU space; recompute on readmission
        self.block_manager.free_all(victim)
        self.running.remove(victim)
        self.waiting.insert(0, victim)     # re-insert at head of waiting
```

### 7.2.5 Step 5 — Admit new requests

The scheduler admits requests from WAITING as long as two budgets allow:

```
Budget 1: token budget   (max_num_batched_tokens)
Budget 2: sequence budget (max_num_seqs)
```

For each waiting request:

```python
remaining_token_budget  = max_num_batched_tokens - tokens_in_batch
remaining_seq_budget    = max_num_seqs            - seqs_in_batch

for seq_group in self.waiting[:]:
    num_prompt_tokens = len(seq_group.prompt_token_ids)
    num_new_seqs      = seq_group.num_seqs()

    # Token budget
    if num_prompt_tokens > remaining_token_budget:
        break  # this and all subsequent requests are too large

    # Sequence budget
    if num_new_seqs > remaining_seq_budget:
        break

    # Memory budget
    if not self.block_manager.can_allocate(seq_group):
        break

    # Admit
    self.block_manager.allocate(seq_group)
    self.waiting.remove(seq_group)
    self.running.append(seq_group)
    remaining_token_budget -= num_prompt_tokens
    remaining_seq_budget   -= num_new_seqs
```

### 7.2.6 Step 6 — Build SchedulerOutputs

The scheduler partitions the running batch into two groups:

```
Prefill group:  newly admitted sequences (first forward pass)
Decode group:   already-running sequences (one decode step)
```

In vLLM these are always batched together in a single GPU kernel call —
prefill tokens and decode tokens travel through the same forward pass.
The attention kernel distinguishes them by their `is_prompt` flag.

```python
@dataclass
class SchedulerOutputs:
    scheduled_seq_groups:     List[ScheduledSequenceGroup]
    num_prefill_groups:       int
    num_batched_tokens:       int
    blocks_to_swap_in:        Dict[int, int]
    blocks_to_swap_out:       Dict[int, int]
    blocks_to_copy:           Dict[int, int]   # CoW copies
    ignored_seq_groups:       List[SequenceGroup]
```

### 7.2.7 Step 7 — Execute, sample, update, stream

The `LLMEngine` executes the kernel with the SchedulerOutputs, samples the next
tokens, updates each sequence's token list and block table, then streams any
finished tokens to the caller.

```
execute_model(scheduler_outputs)
    → ModelOutput(logits per sequence)
sampler(logits, sampling_params)
    → SamplerOutput(next_token_ids)
for each sequence:
    seq.append_token(next_token_id)
    if seq.is_finished(): stream final text
    else: yield streaming delta
```

---

## 7.3 The Two Admission Gates  `[DEEP DIVE]`

### 7.3.1 `max_num_seqs`

**What it controls**: maximum number of *sequences* (not requests, because one
request can have multiple sequences for `n>1`) that can be in the RUNNING state
simultaneously.

**Why it exists**: each decode step generates one logit row per sequence.
Sampling and bookkeeping overhead scales linearly with the number of sequences.
Beyond ~256 sequences the Python overhead of managing per-sequence state becomes
the bottleneck.

**Default**: 256.  On A100 80 GB with LLaMA 3 8B you can often push to 512.

**`[COMMON TRAP]`**: Setting `max_num_seqs` to 1 effectively makes vLLM a
serial server — but it does *not* make requests run faster individually, because
the batch size 1 decode is memory-bandwidth-bound regardless (Chapter 2 §2.3).

### 7.3.2 `max_num_batched_tokens`

**What it controls**: the total number of tokens across all sequences in one
forward pass.

```
Batch at iteration t:

  Prefill requests:  req_A (512 tokens),  req_B (200 tokens)
  Decode  requests:  req_C, req_D, req_E  (1 token each)

  Total batched tokens = 512 + 200 + 1 + 1 + 1 = 715
```

If `max_num_batched_tokens = 512` then req_A fills the entire budget and req_B
must wait until the next iteration.  If set to 2048, both prefills fit.

**Why it exists**: large prefills spike activation memory.  The activation
tensor for a forward pass of N tokens has shape `[N, d_model]` at each layer,
which for N=4096, d_model=4096 is 4096 × 4096 × 2 bytes (BF16) ≈ 32 MB per
layer.  With 32 layers that is ~1 GB — a significant fraction of available HBM.
Capping `max_num_batched_tokens` bounds this spike.

**Default**: 2048 for most models; 8192 for models optimized for long contexts.

### 7.3.3 `max_model_len`

The maximum sequence length the model can process.  Any request whose prompt
exceeds `max_model_len` is rejected immediately (before entering WAITING) with
an error.

**Interaction with the block pool**: `max_model_len` determines how many blocks
a single sequence can *theoretically* consume:

```
max_blocks_per_seq = ceil(max_model_len / block_size)
               = ceil(4096 / 16) = 256 blocks
```

The block manager uses this to answer `can_allocate`: if fewer than
`max_blocks_per_seq` free blocks exist and the request has not yet been
allocated, `can_allocate` returns `NEVER` — the request cannot possibly fit
even if all other requests finish.

---

## 7.4 Worked Example — Admission Decision  `[DEEP DIVE]`

### 7.4.1 System configuration

```
Model:              LLaMA 3 8B
GPU:                A100 80 GB
Total KV blocks:    14 350  (after weights, block_size=16)
max_num_seqs:       128
max_num_batched_tokens: 4096
```

### 7.4.2 State at iteration 42

```
RUNNING (80 sequences, consuming 1 860 blocks):
  Seq A: 384 tokens  → 24 blocks
  Seq B: 256 tokens  → 16 blocks
  … (78 more)

WAITING (5 requests):
  R1: prompt_len=200, n=1
  R2: prompt_len=1500, n=1
  R3: prompt_len=300, n=2  (→ 2 sequences)
  R4: prompt_len=50,  n=1
  R5: prompt_len=800, n=1

SWAPPED: (empty)
Free blocks: 14 350 - 1 860 = 12 490
```

### 7.4.3 Iteration 42 scheduling decisions

**Step 1** — Seqs B and three others finish this iteration.
```
freed blocks: 16 + ... = 94 blocks
Free blocks after Step 1: 12 490 + 94 = 12 584
```

**Step 2** — No SWAPPED sequences.

**Step 3** — Append slots for all running sequences.
```
68 sequences need a new block this step (last token fills their current block):
  68 new blocks allocated.
Free blocks: 12 584 - 68 = 12 516
```

**Step 4** — No preemption needed.

**Step 5** — Admit from WAITING:

```
Initial batch state: 76 seqs, 76 decode tokens.
token_budget_remaining = 4096 - 76 = 4020
seq_budget_remaining   = 128  - 76 = 52

R1: prompt=200, seqs=1
  token budget: 200 ≤ 4020  ✓
  seq   budget: 1   ≤ 52    ✓
  block budget: ceil(200/16) = 13 blocks ≤ 12 516  ✓
  → ADMIT.  budget: tokens=3820, seqs=51, blocks=12 503

R2: prompt=1500, seqs=1
  token budget: 1500 ≤ 3820  ✓
  seq   budget: 1    ≤ 51    ✓
  block budget: ceil(1500/16) = 94 blocks ≤ 12 503  ✓
  → ADMIT.  budget: tokens=2320, seqs=50, blocks=12 409

R3: prompt=300, seqs=2
  token budget: 300 ≤ 2320  ✓
  seq   budget: 2   ≤ 50    ✓
  block budget: ceil(300/16) × 2 = 40 blocks ≤ 12 409  ✓
  → ADMIT.  budget: tokens=2020, seqs=48, blocks=12 369

R4: prompt=50, seqs=1
  token budget: 50 ≤ 2020  ✓
  seq   budget: 1  ≤ 48    ✓
  block budget: ceil(50/16)  = 4 blocks  ≤ 12 369  ✓
  → ADMIT.  budget: tokens=1970, seqs=47, blocks=12 365

R5: prompt=800, seqs=1
  token budget: 800 ≤ 1970  ✓
  seq   budget: 1   ≤ 47    ✓
  block budget: ceil(800/16) = 50 blocks ≤ 12 365  ✓
  → ADMIT.

All 5 waiting requests admitted this iteration.
```

**Step 6** — SchedulerOutputs:

```
Prefill group (5 requests): R1(200), R2(1500), R3(300), R4(50), R5(800)
Decode  group (76 seqs):    one token each
Total batched tokens: 200+1500+300+50+800+76 = 2 926  ≤ 4096  ✓
Total sequences:      81+76 = nope — 76 existing + 6 new seqs = 82 ≤ 128  ✓
```

### 7.4.4 Memory pressure scenario

Suppose R2 had `prompt_len = 12 000`:

```
Block budget for R2: ceil(12 000 / 16) = 750 blocks.
With only 12 503 free blocks and 13 + 94 = 107 already allocated for R1,
remaining = 12 503 - 13 = 12 490 > 750.  So R2 is technically admissible.
```

But suppose total free blocks after Step 3 were only 600:

```
R1: 13 blocks  → 600 - 13 = 587.  R1 admitted.
R2: 750 blocks → 587 < 750.  can_allocate → MAY_LATER.  Stop admitting.
```

R2, R3, R4, R5 stay in WAITING.  The scheduler prints a warning if a request
has been waiting for more than `scheduler_policy_timeout` iterations.

---

## 7.5 Preemption Policies in Depth  `[DEEP DIVE]`

### 7.5.1 SWAP vs RECOMPUTE

```
┌─────────────────┬──────────────────────────────┬─────────────────────────┐
│ Policy          │ What happens                 │ Best when               │
├─────────────────┼──────────────────────────────┼─────────────────────────┤
│ SWAP            │ KV blocks copied CPU→GPU     │ Long sequences          │
│                 │ on resume; no recompute.     │ (recompute is expensive) │
│                 │ Needs CPU DRAM headroom.     │ CPU memory available     │
├─────────────────┼──────────────────────────────┼─────────────────────────┤
│ RECOMPUTE       │ Blocks freed; sequence       │ Short sequences         │
│                 │ re-enqueued for full         │ Prefix caching active   │
│                 │ re-prefill on resume.        │ (only suffix recomputed)│
└─────────────────┴──────────────────────────────┴─────────────────────────┘
```

### 7.5.2 Preemption cost formula

**SWAP cost**:

```
t_swap = (num_blocks × block_size_bytes) / PCIe_bandwidth
       = (64 × 2 MB) / 32 GB/s
       = 128 MB / 32 GB/s
       = 4 ms   (for a 1024-token sequence, LLaMA 3 8B, PCIe 4.0)
```

**RECOMPUTE cost** (with prefix caching):

```
t_recompute = (non_cached_tokens / max_num_batched_tokens) × t_prefill_step
```

If the system prompt (512 tokens) is cached and the request has generated
128 additional tokens:

```
non_cached_tokens = 128
t_recompute = (128 / 4096) × 20 ms ≈ 0.6 ms
```

In this case RECOMPUTE is 6× cheaper than SWAP.  This is why vLLM's default
changed to RECOMPUTE when prefix caching is enabled.

### 7.5.3 LIFO preemption order

The default preemption policy is LIFO — the *most recently admitted* running
request is preempted first.  This is optimal under the "convoy effect":
long-running requests that were admitted early are most likely to be near
completion; preempting them wastes more work.  New arrivals have generated
fewer tokens and are cheaper to evict.

```
Arrival order:  R_a (t=0), R_b (t=1), R_c (t=2)  ← LIFO preempts R_c first

                                         ↑ cheapest to preempt
```

An alternative is Shortest-Remaining-Time-First (SRTF), which preempts the
request estimated to run longest.  This requires output-length prediction
(Chapter 11).

---

## 7.6 Scheduling Budgets and Throughput  `[DEEP DIVE]`

### 7.6.1 The throughput–latency trade-off

The key tension in scheduling:

```
Large batch    → high throughput (GPU utilization ↑)
               → high tail latency (TTFT ↑ for new requests)

Small batch    → low TTFT
               → low throughput (GPU compute wasted)
```

The scheduler operates the system at the optimal point by filling the decode
batch to `max_num_seqs` and admitting prefills only when token budget remains.

### 7.6.2 Chunked prefill

`[DEEP DIVE]`  Introduced in vLLM 0.4, **chunked prefill** splits a long
prefill across multiple iterations so it does not monopolise the token budget.

```
Without chunking:
  Iter 1: prefill R_new (4096 tokens) — decode batch starved for one full step.

With chunked prefill (chunk_size=512):
  Iter 1: prefill R_new chunk_0 (512 tokens) + decode 80 seqs (80 tokens)
  Iter 2: prefill R_new chunk_1 (512 tokens) + decode 80 seqs
  ...
  Iter 8: prefill R_new chunk_7 (512 tokens) + decode 80 seqs
```

The decode batch stays active throughout.  The cost is that TTFT for R_new
increases from 1 iteration to 8, but P99 latency for the *decode* batch
improves dramatically.

```
Setting: --enable-chunked-prefill --max-num-batched-tokens 2048
```

### 7.6.3 Token budget interaction with GQA

For GQA models (LLaMA 3, Chapter 4 §4.4), the prefill attention cost is
$O(n^2 \cdot d_{head} / G)$ where G is the number of KV groups.  For LLaMA 3
8B (G=8, H=32, d_head=128):

```
Prefill FLOP estimate for n tokens, single layer:
  Attention:    n² × d_head × 2 × (H/G)  ÷ G  × ... (see Chapter 2)
                ≈ n² × 32 FLOPs (simplified)
  FFN:          n × 14336 × 2 × 3       ≈ 86n × 10³ FLOPs

At n=4096:
  Attention: 4096² × 32 = 536 × 10⁶ per layer   → 17 GFLOP total
  FFN:       4096 × 86k = 352 × 10⁶ per layer   → 11 GFLOP total
```

The attention cost grows quadratically with sequence length, which is why
`max_num_batched_tokens` must be tuned downward for long-context workloads.

---

## 7.7 Priority-Aware Scheduling  `[DEEP DIVE]`

### 7.7.1 SLA tiers

Production deployments often serve two classes of traffic simultaneously:

- **Latency-sensitive** (interactive chat): TTFT target < 500 ms, inter-token
  latency < 50 ms.

- **Best-effort** (batch summarization, offline indexing): throughput matters,
  latency is flexible.

vLLM (as of 0.5) supports a `priority` field on each request.  The scheduler
prefers higher-priority requests in both admission (Step 5) and the preemption
victim selection (Step 4).

```python
# Priority-aware admission: sort WAITING by priority descending
sorted_waiting = sorted(self.waiting,
                        key=lambda sg: sg.priority,
                        reverse=True)

# Priority-aware preemption: evict lowest-priority RUNNING sequence
victim = min(self.running, key=lambda sg: sg.priority)
```

### 7.7.2 Predictive slack allocation

For **agentic workflows** (where a request will be followed by a tool call and
then a continuation), it pays to reserve KV blocks for the anticipated
continuation rather than freeing and reallocating them.

The heuristic: if a sequence just generated a tool-call token (`<tool_call>`),
the scheduler marks its blocks as "reserved" for the next `slack_budget`
iterations.  If a continuation arrives within that window, it reuses the blocks
without a swap.

### 7.7.3 Throughput measurement

The vLLM paper reported that priority-aware scheduling (with 50 % latency-
sensitive, 50 % best-effort traffic) achieves:

```
FCFS:             1.0× throughput (baseline)
Priority (vLLM):  1.8–7.5× throughput improvement for best-effort requests
                  at equal or better P50 latency for high-priority requests.
```

The improvement is largest when high-priority requests are short (they are
served immediately, freeing blocks quickly) and best-effort requests are long
(they tolerate being preempted).

---

## 7.8 The Request Lifecycle — ASCII Pipeline  `[FOUNDATIONAL]`

```
Client request arrives
         │
         ▼
┌────────────────────┐
│   LLMEngine        │
│   add_request()    │
│   tokenise prompt  │
│   → SequenceGroup  │
└────────┬───────────┘
         │ enqueue
         ▼
┌────────────────────┐
│   WAITING queue    │  ◄── can_allocate check before admission
│   (FIFO + priority)│
└────────┬───────────┘
         │ admit (blocks allocated)
         ▼
┌────────────────────────────────────────────────────────────────────┐
│   RUNNING set                                                      │
│                                                                    │
│   Iteration loop:                                                  │
│   ┌──────┐  ┌───────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│   │Sched.│→ │Forward│→ │ Sample   │→ │ Update   │→ │ Stream   │ │
│   │Output│  │ Pass  │  │ (logits) │  │seq state │  │ tokens   │ │
│   └──────┘  └───────┘  └──────────┘  └──────────┘  └──────────┘ │
│                                                                    │
│   Each iteration: ~20 ms decode  |  ~2–200 ms prefill             │
└────────┬───────────────────────────────┬───────────────────────────┘
         │ is_finished()                 │ preempted
         ▼                               ▼
┌──────────────────┐           ┌──────────────────┐
│  FINISHED        │           │  SWAPPED (CPU)   │
│  free blocks     │           │  or re-WAITING   │
│  stream final    │           │  (RECOMPUTE)     │
└──────────────────┘           └──────────────────┘
```

---

## 7.9 llama.cpp — You Are the Scheduler  `[FOUNDATIONAL]`

### 7.9.1 The imperative model

llama.cpp has no built-in scheduler.  When you call `llama_decode()` you are
the scheduler.  You decide:

- Which sequences to include in the batch.
- How many tokens to process.
- When to free and reallocate `llama_seq_id` slots.

This is both powerful (full control) and dangerous (easy to starve sequences
or overflow the KV cache).

### 7.9.2 The llama_batch API

```c
// Create a batch that can hold max_batch tokens, up to max_seq sequences
llama_batch batch = llama_batch_init(max_batch, 0 /*embeddings*/, max_seq);

// Add a token from sequence seq_id at position pos
// logits=true → compute logits for this token
void llama_batch_add(llama_batch* batch,
                     llama_token token,
                     llama_pos   pos,
                     const llama_seq_id* seq_ids,
                     int         n_seq_id,
                     bool        logits);
```

For multi-sequence batching:

```c
for (int s = 0; s < n_seqs; s++) {
    for (int t = 0; t < seq_lens[s]; t++) {
        llama_batch_add(&batch,
                        tokens[s][t],
                        t,                // position in sequence
                        &seq_ids[s], 1,   // this token belongs to seq s
                        (t == seq_lens[s] - 1));  // only last token needs logits
    }
}
llama_decode(ctx, batch);
```

### 7.9.3 `[COMMON TRAP]` — KV context overflow

llama.cpp allocates a KV cache of `n_ctx` tokens (split across `n_seq_max`
sequences) at context creation time.  If your batch fills all `n_ctx` positions
the next call to `llama_decode` returns a non-zero error code:

```c
int ret = llama_decode(ctx, batch);
if (ret != 0) {
    // KV context full — you must free some sequences
    llama_kv_cache_seq_rm(ctx, victim_seq_id, 0, -1);  // free entire seq
}
```

There is no automatic eviction.  If you ignore the return code your model will
silently produce garbage output or crash.

Always check: `if (llama_decode(ctx, batch) != 0) { handle_overflow(); }`.

---

## 7.10 Building a Multi-Request Scheduler for llama.cpp  `[DEEP DIVE]`

### 7.10.1 Design

We will build a simple round-robin scheduler with priority support that wraps
`llama_decode`.  It maintains:

```
pending_queue   : priority_queue<Request>  (WAITING)
active_seqs     : map<seq_id, Request>     (RUNNING)
free_seq_ids    : stack<int>               (available llama sequence IDs)
```

At each iteration:

```
1. Admit as many WAITING requests as fit within (n_ctx - used_ctx).
2. For each ACTIVE sequence, append one new token to batch.
3. Call llama_decode(batch).
4. Sample next token for each active sequence.
5. Remove finished sequences; reclaim seq_ids.
6. Repeat.
```

### 7.10.2 Memory accounting

Unlike vLLM, llama.cpp's KV cache is a flat buffer of `n_ctx` token slots
shared across all sequences.  The "block manager" is simply:

```
used_slots = Σ len(seq) for seq in active_seqs
free_slots = n_ctx - used_slots

Admit new request if: free_slots ≥ request.max_tokens
```

This is the "naïve" allocation from Chapter 6 §6.1 — which is why llama.cpp
throughput is lower than vLLM for multi-user workloads.

See `code/chapter_07/scheduler_demo.cpp` for the full implementation.

---

## 7.11 vLLM vs llama.cpp — Lifecycle Comparison  `[FOUNDATIONAL]`

```
vLLM:                               llama.cpp:
─────────────────────────────────   ────────────────────────────────
Scheduler:  automatic               Scheduler:  you write it
Memory:     PagedAttention          Memory:     flat KV buffer
State:      SequenceGroup           State:      llama_batch + seq_ids
Batching:   continuous, iteration   Batching:   manual per call
Preemption: SWAP or RECOMPUTE       Preemption: manual seq_rm + re-add
Priority:   built-in field          Priority:   user queue management
Tokens/s:   800–3000 (A100)         Tokens/s:   80–300 (M3 Pro, 4-bit)
Use case:   production serving      Use case:   local, embedded, research
```

---

## 7.12 Code Listing  `[FOUNDATIONAL]`

The following program simulates vLLM's scheduling loop including the 7 steps,
two admission gates, both preemption policies, and statistics collection.

```python
# scheduler_demo.py
# Chapter 7 — The Scheduler and Request Lifecycle
#
# Simulates:
#   1. SequenceGroup lifecycle (WAITING → RUNNING → FINISHED/SWAPPED)
#   2. The 7-step scheduling loop
#   3. max_num_seqs and max_num_batched_tokens admission gates
#   4. SWAP and RECOMPUTE preemption policies
#   5. Priority-aware scheduling
#   6. Throughput statistics
#
# Run:
#   python scheduler_demo.py

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional

# ── Sequence status ───────────────────────────────────────────────────────────

class SequenceStatus(Enum):
    WAITING   = auto()
    RUNNING   = auto()
    SWAPPED   = auto()
    FINISHED  = auto()

# ── Sequence ──────────────────────────────────────────────────────────────────

@dataclass
class Sequence:
    seq_id:          int
    prompt_len:      int
    max_new_tokens:  int
    priority:        int   = 0   # higher = more important
    tokens_generated: int  = 0
    status:          SequenceStatus = SequenceStatus.WAITING
    blocks_used:     int   = 0

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.tokens_generated

    def is_finished(self) -> bool:
        return self.tokens_generated >= self.max_new_tokens

    def __lt__(self, other):
        return self.priority > other.priority   # higher priority first

# ── Block manager stub ────────────────────────────────────────────────────────

class SimpleBlockManager:
    """
    Simplified block manager (flat accounting only — not PagedAttention).
    Sufficient to drive the scheduler simulation.
    """

    def __init__(self, total_blocks: int, block_size: int):
        self.total_blocks  = total_blocks
        self.block_size    = block_size
        self.used_blocks   = 0
        self._seq_blocks: Dict[int, int] = {}   # seq_id → num blocks

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: Sequence) -> bool:
        needed = self.blocks_needed(seq.prompt_len)
        return needed <= (self.total_blocks - self.used_blocks)

    def allocate(self, seq: Sequence) -> None:
        n = self.blocks_needed(seq.prompt_len)
        self._seq_blocks[seq.seq_id] = n
        self.used_blocks += n
        seq.blocks_used   = n

    def can_append(self, seq: Sequence) -> bool:
        current_slots = self._seq_blocks[seq.seq_id] * self.block_size
        if seq.total_len + 1 > current_slots:
            return (self.total_blocks - self.used_blocks) >= 1
        return True

    def append(self, seq: Sequence) -> bool:
        current_slots = self._seq_blocks[seq.seq_id] * self.block_size
        if seq.total_len + 1 > current_slots:
            if (self.total_blocks - self.used_blocks) < 1:
                return False
            self._seq_blocks[seq.seq_id] += 1
            self.used_blocks += 1
            seq.blocks_used   = self._seq_blocks[seq.seq_id]
        return True

    def free(self, seq: Sequence) -> None:
        n = self._seq_blocks.pop(seq.seq_id, 0)
        self.used_blocks -= n
        seq.blocks_used   = 0

    def can_swap_out(self) -> bool:
        return True   # simplified: assume CPU always has space

    @property
    def free_blocks(self) -> int:
        return self.total_blocks - self.used_blocks

# ── Scheduler ─────────────────────────────────────────────────────────────────

class PreemptionPolicy(Enum):
    SWAP      = "swap"
    RECOMPUTE = "recompute"

@dataclass
class SchedulerConfig:
    max_num_seqs:            int   = 64
    max_num_batched_tokens:  int   = 2048
    preemption_policy:       PreemptionPolicy = PreemptionPolicy.RECOMPUTE
    enable_priority:         bool  = False


@dataclass
class IterationStats:
    iteration:        int
    n_prefill:        int
    n_decode:         int
    total_tokens:     int
    n_preemptions:    int
    n_waiting:        int
    n_running:        int
    n_swapped:        int
    free_blocks:      int


class Scheduler:

    def __init__(self,
                 config: SchedulerConfig,
                 block_manager: SimpleBlockManager):
        self.cfg    = config
        self.bm     = block_manager
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence]  = []
        self.swapped: List[Sequence]  = []
        self._iter  = 0

    def add_request(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def schedule(self) -> IterationStats:
        self._iter += 1
        n_preemptions = 0

        # ── Step 1: Remove finished sequences ────────────────────────────
        still_running = []
        for seq in self.running:
            if seq.is_finished():
                seq.status = SequenceStatus.FINISHED
                self.bm.free(seq)
            else:
                still_running.append(seq)
        self.running = still_running

        # ── Step 2: Try swap-ins ──────────────────────────────────────────
        swap_in_list = []
        for seq in self.swapped[:]:
            if self.bm.can_allocate(seq):
                self.bm.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                swap_in_list.append(seq)
                self.swapped.remove(seq)
        self.running.extend(swap_in_list)

        # ── Step 3+4: Ensure append slots for all running seqs ───────────
        for seq in list(self.running):
            while not self.bm.can_append(seq):
                if not self.running:
                    break
                # Preempt the lowest-priority / most recently added sequence
                victim = (min(self.running, key=lambda s: s.priority)
                          if self.cfg.enable_priority
                          else self.running[-1])
                n_preemptions += 1
                self.running.remove(victim)
                self.bm.free(victim)
                if self.cfg.preemption_policy == PreemptionPolicy.SWAP:
                    victim.status = SequenceStatus.SWAPPED
                    self.swapped.append(victim)
                else:
                    # RECOMPUTE: re-enqueue at front of waiting
                    victim.status = SequenceStatus.WAITING
                    victim.tokens_generated = 0   # must redo prefill
                    self.waiting.appendleft(victim)

        # ── Step 5: Admit new requests ────────────────────────────────────
        token_budget = self.cfg.max_num_batched_tokens - len(self.running)
        seq_budget   = self.cfg.max_num_seqs - len(self.running)

        # Priority sort if enabled
        waiting_sorted = (sorted(self.waiting,
                                 key=lambda s: s.priority,
                                 reverse=True)
                          if self.cfg.enable_priority
                          else list(self.waiting))

        for seq in waiting_sorted:
            if seq not in self.waiting:
                continue
            if seq.prompt_len > token_budget:
                continue
            if seq_budget <= 0:
                break
            if not self.bm.can_allocate(seq):
                break
            self.bm.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            self.waiting.remove(seq)
            token_budget -= seq.prompt_len
            seq_budget   -= 1

        # ── Step 6+7: Simulate one decode / prefill step ─────────────────
        n_prefill = 0
        n_decode  = 0
        total_tokens = 0

        for seq in list(self.running):
            if seq.tokens_generated == 0:
                # Prefill pass
                n_prefill    += 1
                total_tokens += seq.prompt_len
            else:
                # Decode step
                n_decode     += 1
                total_tokens += 1

            ok = self.bm.append(seq)
            if ok:
                seq.tokens_generated += 1
            else:
                # This should not happen if Step 3-4 ran correctly
                pass

        return IterationStats(
            iteration     = self._iter,
            n_prefill     = n_prefill,
            n_decode      = n_decode,
            total_tokens  = total_tokens,
            n_preemptions = n_preemptions,
            n_waiting     = len(self.waiting),
            n_running     = len(self.running),
            n_swapped     = len(self.swapped),
            free_blocks   = self.bm.free_blocks,
        )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Basic lifecycle trace (8 requests)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 1: Basic Lifecycle Trace")
print("=" * 65)

random.seed(42)
cfg1   = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=512)
bm1    = SimpleBlockManager(total_blocks=200, block_size=16)
sched1 = Scheduler(cfg1, bm1)

requests = [
    Sequence(seq_id=i,
             prompt_len=random.randint(20, 150),
             max_new_tokens=random.randint(10, 40))
    for i in range(8)
]
for req in requests:
    print(f"  req_{req.seq_id}: prompt={req.prompt_len}, max_new={req.max_new_tokens}")

# Enqueue all
for req in requests:
    sched1.add_request(req)

print(f"\n{'Iter':<5} {'prefill':<8} {'decode':<8} {'tokens':<8} "
      f"{'wait':<6} {'run':<5} {'swap':<5} {'free_blk':<9} {'preempt'}")
print("-" * 65)

for iteration in range(30):
    stats = sched1.schedule()
    if iteration < 15 or stats.n_running > 0 or stats.n_waiting > 0:
        print(f"  {stats.iteration:<4} {stats.n_prefill:<8} {stats.n_decode:<8} "
              f"{stats.total_tokens:<8} {stats.n_waiting:<6} {stats.n_running:<5} "
              f"{stats.n_swapped:<5} {stats.free_blocks:<9} {stats.n_preemptions}")
    if stats.n_running == 0 and stats.n_waiting == 0 and stats.n_swapped == 0:
        print(f"  All requests finished at iteration {stats.iteration}.")
        break


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Preemption under memory pressure
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SECTION 2: Preemption Under Memory Pressure")
print("=" * 65)

for policy in [PreemptionPolicy.SWAP, PreemptionPolicy.RECOMPUTE]:
    random.seed(7)
    cfg2   = SchedulerConfig(max_num_seqs=8,
                             max_num_batched_tokens=2048,
                             preemption_policy=policy)
    bm2    = SimpleBlockManager(total_blocks=50, block_size=16)  # tight!
    sched2 = Scheduler(cfg2, bm2)

    heavy_reqs = [
        Sequence(seq_id=i, prompt_len=100, max_new_tokens=30)
        for i in range(10)
    ]
    for r in heavy_reqs:
        sched2.add_request(r)

    total_preemptions = 0
    total_iters       = 0
    for _ in range(200):
        s = sched2.schedule()
        total_preemptions += s.n_preemptions
        total_iters        = s.iteration
        if s.n_running == 0 and s.n_waiting == 0 and s.n_swapped == 0:
            break

    print(f"\n  Policy={policy.value}: "
          f"iters={total_iters}, preemptions={total_preemptions}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Priority-aware scheduling
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SECTION 3: Priority-Aware Scheduling")
print("=" * 65)

random.seed(0)

def run_experiment(enable_priority: bool, label: str):
    cfg   = SchedulerConfig(max_num_seqs=4,
                            max_num_batched_tokens=1024,
                            enable_priority=enable_priority)
    bm    = SimpleBlockManager(total_blocks=300, block_size=16)
    sched = Scheduler(cfg, bm)

    # 3 high-priority short requests + 3 low-priority long requests
    hp_reqs = [Sequence(seq_id=i,   prompt_len=50,  max_new_tokens=20, priority=10)
               for i in range(3)]
    lp_reqs = [Sequence(seq_id=i+3, prompt_len=200, max_new_tokens=80, priority=1)
               for i in range(3)]
    for r in hp_reqs + lp_reqs:
        sched.add_request(r)

    finish_times: Dict[int, int] = {}
    for iteration in range(500):
        s = sched.schedule()
        for seq in (hp_reqs + lp_reqs):
            if seq.is_finished() and seq.seq_id not in finish_times:
                finish_times[seq.seq_id] = s.iteration
        if len(finish_times) == 6:
            break

    hp_avg = sum(finish_times[r.seq_id] for r in hp_reqs) / 3
    lp_avg = sum(finish_times[r.seq_id] for r in lp_reqs) / 3
    print(f"  {label}: HP avg_finish={hp_avg:.1f}, LP avg_finish={lp_avg:.1f}")

run_experiment(False, "FCFS (no priority)  ")
run_experiment(True,  "Priority-aware      ")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Throughput measurement
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SECTION 4: Throughput (tokens/iteration) vs max_num_seqs")
print("=" * 65)

print(f"\n  {'max_num_seqs':<15} {'avg_tokens/iter':<18} {'avg_running'}")
print("  " + "-" * 45)

for max_seqs in [4, 8, 16, 32, 64]:
    random.seed(1)
    cfg   = SchedulerConfig(max_num_seqs=max_seqs,
                            max_num_batched_tokens=4096)
    bm    = SimpleBlockManager(total_blocks=2000, block_size=16)
    sched = Scheduler(cfg, bm)

    # Continuous arrival: 3 new requests every 5 iterations
    seqs_made = 0
    total_tok = 0
    count     = 0

    for it in range(100):
        if it % 5 == 0:
            for _ in range(3):
                r = Sequence(seq_id=seqs_made,
                             prompt_len=random.randint(32, 256),
                             max_new_tokens=random.randint(20, 100))
                sched.add_request(r)
                seqs_made += 1
        s = sched.schedule()
        total_tok += s.total_tokens
        count     += 1

    print(f"  {max_seqs:<15} {total_tok/count:<18.1f} "
          f"{sum(1 for r in sched.running):.1f}")

print("\nDone.")
```

---

## 7.13 Chapter Summary

| Concept | Key fact |
|---------|----------|
| Scheduling loop | 7 steps per iteration: evict → swap-in → append-check → preempt → admit → build → execute |
| SequenceGroup | Unit of scheduling; holds N parallel output sequences |
| `max_num_seqs` | Hard cap on concurrent sequences (default 256) |
| `max_num_batched_tokens` | Hard cap on tokens per forward pass (default 2048) |
| Prefill + decode in one pass | One GPU kernel handles both; scheduler tags each token |
| SWAP preemption | Copy blocks to CPU; costs ~4 ms per 1024-token sequence |
| RECOMPUTE preemption | Discard + re-prefill; cheaper with prefix caching |
| LIFO eviction | Protects long-running sequences; evicts recent arrivals first |
| Chunked prefill | Splits long prefills to keep decode latency stable |
| Priority scheduling | 1.8–7.5× throughput improvement for mixed workloads |
| llama.cpp | No scheduler; KV overflow → non-zero return from `llama_decode` |

### Why this matters for what follows

- **Chapter 8** (Startup and Initialization) describes how the block pool size
  used by the scheduler is determined: the dummy forward pass measures peak
  activation memory exactly, and the remainder is carved into KV blocks.

- **Chapter 9** (The Forward Pass) explains what happens inside `execute_model`
  between Steps 6 and 7 of the scheduling loop.

- **Chapter 11** (Speculative Decoding) adds a *draft* sequence to every
  SequenceGroup; the scheduler must budget for draft + verify tokens together.

---

## 7.14 Further Reading

- Yu et al., "Orca: A Distributed Serving System," OSDI 2022.
  *(Continuous batching + iteration-level scheduling.)*

- Kwon et al., "Efficient Memory Management with PagedAttention," SOSP 2023.
  *(Scheduler + block manager co-design.)*

- Agrawal et al., "SARATHI-Serve: Chunked Prefill for LLM Inference," 2024.
  *(Chunked prefill analysis.)*

- vLLM source: `vllm/core/scheduler.py`.

---

*End of Chapter 7.*


---

## Chapter Summary

- **Three queues**: vLLM's scheduler maintains `waiting` (prefill queued), `running` (active decode), and `swapped` (GPU evicted to CPU) queues.
- **Continuous batching**: at every step, the scheduler selects the maximal set of sequences that fit within the token budget, mixes prefill and decode, and emits a single batched forward pass.

> **LinkedIn Scenario Update:** The LinkedIn deployment running at 28% GPU utilization is paying $1.2M/month because static batching leaves the GPU idle between requests. Switching to continuous batching — the default scheduler mode described in this chapter — would fill those idle cycles with queued requests from the pool of 50K concurrent users. Empirically, deployments at similar request rates see utilization climb from the 25–30% range to 65–75%; at the LinkedIn scale that translates to roughly a $600K/month reduction in compute cost for the same throughput, or a doubling of served request capacity at the same spend.
- **Chunked prefill**: long prefills are split across multiple steps so they do not starve decoding sequences, bounding first-token latency.
- **Preemption policy**: when GPU blocks are exhausted, vLLM either swaps the lowest-priority running sequence to CPU or recomputes from scratch (no swap) depending on `--preemption-mode`.
- **Priority and FCFS**: default scheduling is First-Come-First-Served; custom priority functions can be plugged in for SLA-differentiated workloads.
- **Head-of-line blocking elimination**: because sequences enter and leave the running queue continuously, a long sequence does not block shorter ones behind it.
- **TTFT vs ITL trade-off**: larger prefill chunks reduce TTFT but steal compute from decode steps, increasing inter-token latency for running sequences.

---

## Self-Check Questions

1. The scheduler has 6 sequences in `running` and 10 in `waiting`. The token budget for the next step is 2 048. Three running sequences contribute 1 token each (decode), and two waiting sequences have prefill lengths of 400 and 800 tokens. Show how the scheduler fills the batch. *(Section 7.2)*

2. A sequence is mid-decode at token 150 and the GPU block pool is exhausted. Describe the swap procedure: which data moves, to where, and at what cost in latency and memory. *(Section 7.3)*

3. Chunked prefill splits a 4 096-token prompt into chunks of 512. How many scheduler steps does the prefill take? During those steps, can other sequences continue decoding? *(Section 7.4)*

4. You observe that ITL (inter-token latency) suddenly spikes when new users arrive. Diagnose three scheduler-level causes and the configuration parameter that addresses each. *(Section 7.5)*

5. Why does the scheduler need to call `can_allocate` from the block manager before promoting a sequence from `waiting` to `running`? What would break if it skipped this check? *(Section 7.2)*


---

## Worked Solutions

---

### Solution 1 — Batch filling with 6 running + 10 waiting sequences

**Given:** Token budget=2,048; 6 running (1 token each, decode); 2 waiting prefills (400 and 800 tokens)

**Step 1 — Reserve decode tokens.**

$$\text{decode tokens} = 6 \times 1 = 6$$
$$\text{remaining budget} = 2{,}048 - 6 = 2{,}042$$

**Step 2 — Greedily admit waiting sequences (FCFS by default).**

| Action | Cost | Remaining |
|--------|------|-----------|
| Admit seq-W1 (400-token prefill) | 400 | 2,042 − 400 = **1,642** |
| Admit seq-W2 (800-token prefill) | 800 | 1,642 − 800 = **842** |
| Continue checking remaining 8 waiting sequences | Depends on their lengths | up to 842 more tokens |

**Step 3 — Interpret.**

The final batch contains at minimum: 6 decode + W1 + W2 = 8 sequences, 1,206 tokens. Up to 842 more tokens from the remaining 8 waiting sequences can fit. For instance, if the next waiting sequence has a 600-token prefill: 842 − 600 = 242 remaining → could fit one more ~200-token prefill.

**Step 4 — Why this matters.**

The scheduler's goal is to maximize GPU utilization (fill the token budget) while maintaining fairness (process waiting requests in order). The 2,048 token budget prevents any single step from being dominated by one enormous prefill at the expense of currently running decode sequences.

---

### Solution 2 — Swap procedure for a mid-decode sequence at token 150

**Given:** Sequence at token 150, GPU block pool exhausted

**Step 1 — What data is swapped.**

The sequence's KV cache consists of all physical GPU blocks allocated to it. For a typical model (32 layers, 8 KV heads, d_k=128, FP16) at 150 tokens:

$$\text{blocks} = \lceil 150/16 \rceil = 10 \text{ blocks} \times 256 \text{ KB} = 2.5 \text{ MB of KV data}$$

**Step 2 — Where it moves.**

GPU HBM → CPU DRAM via PCIe. The block manager:
1. Identifies all 10 physical blocks belonging to this sequence.
2. Issues CUDA `cudaMemcpyDeviceToHost` for each block.
3. Allocates equivalent CPU memory buffers.
4. Updates the sequence's `block_table` to point to CPU-side blocks.
5. Marks the GPU blocks as free.

**Step 3 — Latency cost.**

PCIe Gen4 bandwidth ≈ 32 GB/s bidirectional (practical ~16–20 GB/s for small transfers):

$$\text{swap-out time} \approx \frac{2.5 \text{ MB}}{16 \text{ GB/s}} \approx 0.16 \text{ ms}$$

Swap-in (when resumed) adds another ~0.16 ms. Total round-trip: ~0.32 ms for this size sequence.

**Step 4 — Memory freed.**

10 blocks × 256 KB = 2.5 MB returned to the GPU block pool for other sequences.

**Step 5 — When the sequence is resumed.**

The scheduler waits until enough GPU blocks are free to accommodate all 10 blocks. Then it issues `cudaMemcpyHostToDevice` and returns the sequence to the `running` queue. The sequence continues from token 150 without any recalculation.

---

### Solution 3 — Chunked prefill: steps and decode compatibility

**Given:** 4,096-token prompt, chunk_size=512

**Step 1 — Number of prefill steps.**

$$\text{steps} = \lceil 4{,}096 / 512 \rceil = \textbf{8 steps}$$

Each step processes 512 tokens of the prompt before the sequence can begin decoding.

**Step 2 — Can decode sequences continue during chunked prefill?**

**Yes.** This is the key advantage of chunked prefill. Without it (monolithic prefill), the entire 4,096-token prefill occupies one step, blocking ALL decode sequences for that step duration. With chunked prefill:

- Step 1: 512 prefill tokens + decode tokens from all running sequences (e.g., 32 sequences × 1 decode token = 32 tokens). Total: 544 tokens.
- Steps 2–8: Same pattern — 512 prefill tokens per step + running decode tokens.

Decode sequences experience slightly higher per-step latency (because they share the batch with the prefill chunk), but they are never completely blocked.

**Step 3 — TTFT trade-off.**

For the new request: TTFT increases (must wait 8 steps × step_duration instead of 1 step). For other requests: ITL (inter-token latency) is more stable because no single step is dominated by a massive prefill.

---

### Solution 4 — Diagnosing ITL spikes when new users arrive

**What we need:** Three scheduler-level causes and their configuration remedies.

**Cause 1: Large monolithic prefills preempting decode sequences.**

When a 4,096-token prefill arrives, it occupies the entire token budget for one step. All running decode sequences must wait. Each missing decode step adds one ITL to every running request.

*Fix:* Enable chunked prefill with `--enable-chunked-prefill --max-num-batched-tokens 2048`. This limits prefill to 2,048 tokens per step, allowing decode sequences to share the batch.

**Cause 2: Block pool exhaustion triggering preemptions.**

New users require new KV cache blocks. If the pool is full, the scheduler must preempt (swap or recompute) some running sequences, adding their swap latency to ITL.

*Fix:* Reduce `--gpu-memory-utilization` from 0.90 to 0.85 to keep a safety margin in the block pool. Alternatively, use `--max-num-seqs` to cap concurrent requests before the pool fills.

**Cause 3: Scheduler priority bias toward new arrivals (prefill preference).**

If the scheduler greedily admits new prefills at the expense of running decode sequences, active users experience ITL spikes while their sequences are de-prioritized.

*Fix:* Set `--scheduling-policy priority` or tune `--max-num-prefill-seqs` to limit how many new prefills can enter the batch simultaneously, protecting decode sequences' share of the token budget.

---

### Solution 5 — Why can_allocate must be called before promoting waiting→running

**What we need:** What breaks if this check is skipped.

**Step 1 — What can_allocate checks.**

Before promoting a waiting sequence to running, the block manager verifies:
- Enough free physical blocks exist to hold the sequence's first prefill chunk
- The system is not in a "block-starved" state where admitting another sequence would trigger immediate preemption

**Step 2 — What breaks without the check.**

**Scenario:** Block pool has 2 free blocks. A new 512-token sequence is promoted without checking. Prefill begins. After 2 blocks (32 tokens), the scheduler runs `can_append_slot` → fails immediately. The sequence must be preempted.

This creates a **preemption loop**:
1. Sequence admitted without block check
2. Partial prefill consumes available blocks
3. Preemption triggered after a few tokens
4. Swap/recompute cost incurred
5. Sequence returns to waiting queue
6. Cycle repeats

**Step 3 — The correct behavior.**

`can_allocate` returns False when the pool cannot support the sequence. The scheduler leaves the sequence in `waiting` and serves existing running sequences until enough blocks are freed. This prevents wasteful partial-prefill-and-abort cycles and keeps the GPU doing useful work.

