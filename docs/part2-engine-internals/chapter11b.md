# Chapter 11.5: KV Cache Eviction and Compression — When the Cache Is Full

> *"You cannot remember everything forever. The question is not whether to forget —
> it is which memories to keep, and which to let go without losing the thread."*

---

## What You Will Understand

- Why a full KV cache is not a theoretical edge case but the default state
  at 128K+ context windows.
- The **attention sink** phenomenon: why the first few tokens receive
  disproportionate attention mass and must never be evicted.
- **H2O (Heavy Hitter Oracle)**: accumulate attention scores, keep the
  tokens that matter, silently drop the rest.
- **SnapKV**: compress a prompt's KV cache by clustering similar keys into
  pooled representatives.
- **Token merging / CaM**: merge near-duplicate KV pairs before they
  enter the cache, reducing the cache footprint at write time.
- The accuracy-memory tradeoff curve: how much budget is actually needed
  to stay within 1 % perplexity degradation.
- Where eviction hooks into vLLM's `BlockSpaceManager` and how llama.cpp's
  `llama_kv_cache_seq_rm` implements sliding-window eviction.

## What You Need First

- **Chapter 6** — PagedAttention and the block manager (the physical KV
  cache layout, block table, eviction vocabulary).
- **Chapter 11** — Prefill and chunked prefill (how the cache gets filled
  in the first place, and why 128 K context requests fill it quickly).

---

## 11b.1 The Full-Cache Problem  `[FOUNDATIONAL]`

### 11b.1.1 What "full" means in practice

Every KV cache has a fixed capacity measured in **token slots**: the product
of the number of physical blocks and the block size.  In vLLM with a 70 B
model on two A100 80 GB GPUs, that capacity is roughly 40 000–60 000 slots
at BF16 precision (exact value depends on `gpu_memory_utilization` and
quantization).

A 128 000-token context window fills that budget more than twice over.
Modern long-context workloads — RAG over large documents, code review of
entire repositories, multi-turn conversations — routinely hit this ceiling.

```
WHAT HAPPENS WHEN THE KV CACHE IS FULL
────────────────────────────────────────────────────────────────────────

 KV cache capacity:      C = 65 536 token slots
 Request context length: L = 128 000 tokens

 Naive approach:         block on admission until another request finishes
                         → TTFT = infinity for the waiting request

 Alternative 1: REJECT — return HTTP 503, context too long
 Alternative 2: TRUNCATE — silently drop the oldest tokens (dangerous)
 Alternative 3: EVICT    — choose which tokens to remove intelligently
 Alternative 4: COMPRESS — merge tokens to reduce their slot footprint

 ┌─────────────────────────────────────────────────────────────────┐
 │  Token budget at position 128 000:                              │
 │                                                                 │
 │  [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] │
 │  ←────────────────  128 000 tokens  ───────────────────────────→│
 │                                                                 │
 │  Cache capacity: ██████████████████████████████░░░░░░░░░░░░░░░  │
 │                  ←──── 65 536 ─────────────────→               │
 │                                                                 │
 │  Overflow: 62 464 tokens must either be evicted or compressed   │
 └─────────────────────────────────────────────────────────────────┘
```

### 11b.1.2 Why naive truncation is dangerous

If you simply drop the oldest tokens (positions 0 to overflow), you lose
the **system prompt**, the **question**, and any critical context established
at the beginning of the conversation.  The model has no knowledge that tokens
were removed; it will continue generating as though those positions still exist
in the attention window, producing hallucinations or contradictions.

`[COMMON TRAP]` The correct framing is not "which tokens to remove" but
"which tokens can be removed with the least degradation to the model's
ability to continue the generation coherently."

### 11b.1.3 The eviction problem statement

```
Given:
  • A sequence of N tokens, each with a stored (K, V) pair per layer.
  • A cache budget B < N (we must reduce N slots to B slots).
  • A forward pass at position N+1 that will attend over the retained slots.

Goal:
  • Choose the B slots to retain such that the attention output at N+1
    is as close as possible to the full-attention output over all N slots.

Constraint:
  • The decision must be made online (we have seen only positions 0..N).
  • The decision must be made fast (it runs on the decode critical path).
```

Three families of solution have emerged:

| Family | Representative | Key idea |
|--------|---------------|----------|
| Score-based eviction | H2O | Keep tokens with highest accumulated attention |
| Compression / clustering | SnapKV | Merge similar keys into representatives |
| Merge at write time | CaM | Merge before writing; never store duplicates |

We cover each in turn.

---

## 11b.2 Attention Sinks  `[DEEP DIVE]`

### 11b.2.1 The StreamingLLM observation

In 2023, Xiao et al. (*Efficient Streaming Language Models with Attention Sinks*,
arXiv 2309.17453) made a striking empirical observation: when processing very
long sequences, attention weights are **not** distributed uniformly over the
context.  Instead:

1. The **first few tokens** (positions 0–3) receive disproportionately large
   attention weights, even when those tokens carry no semantic relevance to
   the current query.
2. The **most recently generated tokens** also receive large attention weights
   (recency bias).
3. Tokens in the **middle of the context** receive relatively little attention.

The first-token phenomenon is called the **attention sink**.  These tokens act
as a "drain" that absorbs attention probability mass that the softmax must
allocate somewhere — even when no token in the context is a natural strong
match for the query.

```
ATTENTION SINK ILLUSTRATION
─────────────────────────────────────────────────────────────────────────────

 Query: token at position 512 (in a 512-token sequence)

 Attention weights (conceptual, averaged across heads and layers):

  weight
  0.15 │  ■■                                              ■
  0.12 │  ■■                                          ■■■ ■
  0.09 │  ■■                                        ■■■■■ ■
  0.06 │  ■■ ■        ■        ■              ■  ■■■■■■■■ ■
  0.03 │  ■■ ■■ ■ ■ ■■ ■ ■ ■ ■ ■■■ ■ ■ ■  ■ ■■■■■■■■■■ ■
  0.00 └──┬──────────────────────────────────────────────┬──→ position
          0   (sink)                               510 511  (recent)

 Observations:
  • Positions 0-3:  ~15-20% of total attention mass  ← SINKS
  • Positions 4-480: scattered ~0.02% per token       ← "MIDDLE"
  • Positions 481-511: rising curve, ~30% total       ← RECENCY
```

### 11b.2.2 Why sinks form

The softmax function requires that all attention weights sum to 1.0.  During
training, the model learns to route "idle" attention probability mass to the
first tokens — which are always present, always available, and do not carry
specific semantic content (they are often `<bos>`, punctuation, or a
system-prompt header like `"You are a helpful assistant."`).

This is a **learned behavior** that emerges from the training distribution,
not an architectural feature.  The model effectively says: "When no other
token is highly relevant, dump the probability mass here."

### 11b.2.3 The eviction implication

`[COMMON TRAP]` If you evict token 0 based on a simple LRU or low-attention
policy, you corrupt the model's internal routing mechanism.  Perplexity spikes
catastrophically — not because position 0 contained important content, but
because the model's softmax no longer has a valid "idle drain."

**The StreamingLLM rule**: Always keep the first `A` tokens (typically A=4)
regardless of their attention scores.  These are *structural* tokens, not
*semantic* ones.

```
SINK-AWARE EVICTION POLICY
────────────────────────────────────────────────────────────────────

 Token budget: B = 20
 Full sequence: N = 60 tokens
 Sink tokens to always keep: A = 4  (positions 0-3)

 Available eviction candidates: positions 4 .. 59  (56 tokens)
 Tokens to evict: 60 - 20 = 40
 Evictable budget: 56 tokens → evict 40, keep 16 from middle+recent

 Retained set:
   [0 1 2 3]               ← sinks (never evict)
   [selected 12 tokens from middle/recent based on scores]
   [last 4 tokens]         ← recency window (always keep most recent)

 ─────────────────────────────────────────────────────────────────
  Total retained: 4 + 12 + 4 = 20 = B  ✓
```

---

## 11b.3 H2O — Heavy Hitter Oracle  `[DEEP DIVE]`

### 11b.3.1 The core idea

H2O (*H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
Language Models*, Zhang et al., arXiv 2306.14048) observes that attention
scores are highly **skewed**: a small fraction of tokens (the "heavy hitters")
accumulate the vast majority of attention mass across all decode steps.

The H2O algorithm:

1. During decode, accumulate the attention weights each token receives
   across all steps into a running score vector `s[i]`.
2. When the cache is full, evict the tokens with the **lowest** accumulated
   score, subject to always preserving sink tokens.
3. Continue decode; the evicted positions are gone, but the heavy hitters
   remain.

```
H2O ALGORITHM (per layer, single head simplified)
──────────────────────────────────────────────────────────────────────

 State:
   • cache_keys[0..n-1], cache_values[0..n-1]   — stored KV pairs
   • score[0..n-1]                               — accumulated attention
   • B                                           — max budget

 On each decode step t:
   1. Compute attention weights w[i] = softmax(Q_t · K[i] / √d)[i]
   2. Update:  score[i] += w[i]   for all i in cache
   3. Write new K_t, V_t with score[t] = w[t]  (its first contribution)
   4. If len(cache) > B:
        candidates = {i : i >= A}   (exclude first A sink tokens)
        victim = argmin score[candidates]
        evict(victim)
```

### 11b.3.2 Worked example with 8 tokens, keeping top-4

```
WORKED EXAMPLE 11b.1 — H2O Eviction
──────────────────────────────────────────────────────────────────────

Setup:
  Sequence length N = 8 tokens (positions 0..7)
  Cache budget    B = 4
  Sink tokens     A = 2 (positions 0 and 1 always kept)

Accumulated attention scores (after processing all 8 tokens):

  Position:  0     1     2     3     4     5     6     7
  Score:    0.42  0.38  0.12  0.28  0.05  0.31  0.09  0.19
             ↑     ↑                             
            sink  sink  (eviction candidates: 2..7)

Step 1 — Identify candidates (non-sink positions):
  Candidates: {2: 0.12, 3: 0.28, 4: 0.05, 5: 0.31, 6: 0.09, 7: 0.19}

Step 2 — Budget accounting:
  Already keeping: 0, 1  (sinks, 2 slots)
  Remaining budget: B - A = 4 - 2 = 2 slots for non-sink tokens

Step 3 — Rank candidates by score (descending):
  5: 0.31  ← keep
  3: 0.28  ← keep
  7: 0.19
  2: 0.12
  6: 0.09
  4: 0.05

Step 4 — Keep top-2 non-sink candidates: {5, 3}
  Evict: {2, 4, 6, 7}

Final cache: [0, 1, 3, 5]  — 4 tokens, within budget ✓

Attention quality check (next decode step):
  Full attention over [0..7]:  Q·K scores span all 8 tokens
  Evicted attention over [0,1,3,5]: misses positions 2,4,6,7

  Expected degradation: positions 2,4,6 had low scores (0.12, 0.05, 0.09)
  → total missed mass ≈ 0.12+0.05+0.09 = 0.26

  Position 7 (score 0.19) was evicted — this is the largest loss.
  A recency window would have kept it.

ENHANCED H2O with recency window R=1:
  Always keep last R tokens regardless of score.
  Revised candidates: {2,3,4,5,6}  (7 excluded as recent)
  Keep sinks {0,1}, recent {7}, top-1 candidate: {5}
  Final cache: [0, 1, 5, 7]
  Missed mass: 0.12+0.28+0.05+0.09 = 0.54 — worse on average but
  recency-correct for next-step coherence.
```

### 11b.3.3 Cumulative score vs. per-step score

A subtlety: should you use the **total** accumulated score or the **recent**
accumulated score (e.g., exponentially decaying sum)?

- **Total** accumulated score: biased toward early tokens that have been
  "in play" for many decode steps.  Often dominated by sink tokens.
- **Decayed** accumulated score: more responsive to recent attention patterns;
  better for tasks where the model's focus shifts across the sequence.

In practice, H2O uses the simple cumulative sum because it is O(1) to update
and already captures sinks implicitly (sinks accumulate the most mass).

---

## 11b.4 SnapKV — Key Clustering and Pooling  `[DEEP DIVE]`

### 11b.4.1 Motivation: the prompt bottleneck

H2O works well for decode-phase tokens, but long prompts present a different
problem: all N prompt tokens are written to the KV cache during prefill in a
single pass.  By the time the first decode token arrives, the cache is already
at capacity — there is no "gradual accumulation" of attention scores.

SnapKV (*SnapKV: LLM Knows What You are Looking for Before Generation*,
Li et al., arXiv 2404.14469) addresses this: **compress the prompt KV cache
before decode begins**.

### 11b.4.2 The pooling window approach

SnapKV's key insight: within a prompt, many consecutive keys are semantically
similar.  Nearby sentences in a document, bullet points in a list, rows in a
data table — all produce keys that cluster together in embedding space.

The algorithm:

```
SNAPKV ALGORITHM
──────────────────────────────────────────────────────────────────────

 Input:
   • Prompt of N tokens: K[0..N-1], V[0..N-1]  (one layer, one head)
   • Window size W (default 16)
   • Compression budget B < N

 Step 1 — Observation window
   Use the last W prompt tokens as "observation queries" to compute
   attention over the full prompt:

       A_obs[w, i] = softmax(K[N-W+w] · K[i]^T / √d)  for w in [0,W)

   Aggregate: importance[i] = mean_{w} A_obs[w, i]

 Step 2 — Select important positions
   Sort by importance[i], keep top-(B - sink_count) positions
   (always prepend the A sink positions)

 Step 3 — Pool within each selected cluster
   For each selected position i, pool its K and V with the nearest
   W_pool neighbors:

       K_compressed[i] = avg(K[i-W_pool//2 : i+W_pool//2])
       V_compressed[i] = avg(V[i-W_pool//2 : i+W_pool//2])

 Step 4 — Replace prompt KV with compressed KV
   The cache now holds B entries instead of N entries.
   Decode proceeds as normal against the compressed cache.
```

### 11b.4.3 Worked example: compressing 32 tokens to 8

```
WORKED EXAMPLE 11b.2 — SnapKV Compression
──────────────────────────────────────────────────────────────────────

Setup:
  Prompt length:   N = 32 tokens
  Budget:          B = 8
  Window size:     W = 4  (last 4 tokens used as observation queries)
  Pool window:    W_pool = 2
  Sink tokens:     A = 2

Observation window: positions 28, 29, 30, 31

Importance scores (aggregated attention from observation window):
  pos: 0    1    2    3    4    5    6    7    8    9   10   11   ...
  imp: 0.09 0.11 0.03 0.02 0.15 0.18 0.04 0.12 0.02 0.01 0.14 0.16 ...

After selecting top-(B-A) = 6 from non-sink positions:
  Selected: {5, 4, 7, 10, 11, 19}  (plus sinks 0, 1)

Pooling at selected position 5 (W_pool=2):
  K_compressed[5] = avg(K[4], K[5])   (left neighbor only if near boundary)
  V_compressed[5] = avg(V[4], V[5])

Final compressed cache: 8 entries (2 sinks + 6 selected+pooled)

Compression ratio: 32 → 8 = 4× reduction
```

### 11b.4.4 Cosine similarity as a quality proxy

After compression, you can estimate quality loss by computing the cosine
similarity between the original and compressed keys:

```
Quality proxy = (1/B) Σ_i cos_sim(K_original[i], K_compressed[i])

Perfect retention:  quality = 1.0
Typical SnapKV:     quality ≈ 0.85 - 0.95 at 4× compression
```

---

## 11b.5 Token Merging and CaM  `[DEEP DIVE]`

### 11b.5.1 Merge at write time

Both H2O and SnapKV operate on an already-full cache (evicting or clustering
after the fact).  A third family of approaches **merges tokens before they are
written**, keeping the cache permanently smaller.

**CaM** (*CaM: Cache Merging for Memory-Efficient LLMs in the Wild*,
arXiv 2402.05262) is the representative algorithm:

```
CAM ALGORITHM (simplified)
──────────────────────────────────────────────────────────────────────

 On each token write (position t):
   1. Compute K[t] for the new token.
   2. Find the most similar existing key in the cache:
        j* = argmax_{j in cache} cos_sim(K[t], K[j])
   3. If cos_sim(K[t], K[j*]) > threshold τ:
        Merge: K[j*] = (K[j*] + K[t]) / 2   (running mean)
               V[j*] = (V[j*] + V[t]) / 2
        Do NOT write a new entry.
      Else:
        Write K[t], V[t] as a new entry.
```

The cache size grows only when genuinely new information arrives.  For
repetitive contexts (repeated phrases, similar sentences in a document),
the cache may grow at a fraction of the naive rate.

### 11b.5.2 The threshold tradeoff

| Threshold τ | Cache growth | Quality risk |
|-------------|-------------|--------------|
| 0.99 (very strict) | Near 100% of naive | Almost none |
| 0.95 | ~80% of naive | Small |
| 0.90 | ~60% of naive | Moderate |
| 0.80 | ~40% of naive | High |

`[COMMON TRAP]` Setting τ too low merges semantically distinct tokens.
The model cannot distinguish them in subsequent attention computations,
causing coherence failures on tasks requiring precise token identity
(e.g., copying verbatim quotes, exact arithmetic).

### 11b.5.3 Comparison: eviction vs. compression vs. merge

```
APPROACH COMPARISON
──────────────────────────────────────────────────────────────────────

                    H2O          SnapKV         CaM
─────────────────────────────────────────────────────────────────────
 When applied?      Decode       Post-prefill   At write time
 Granularity        Token slot   Token cluster  Token slot
 Cache size after   B (hard cap) B (hard cap)   Variable (< N)
 Attention quality  High hitters Important keys Similar merged
 Implementation     Simple       Moderate       Requires lookup
 Overhead           O(B)         O(N·W)         O(B) per token
 Sink handling      Explicit     Explicit       Implicit (sinks
                                                rarely similar
                                                to new tokens)
─────────────────────────────────────────────────────────────────────
```

---

## 11b.6 The Accuracy-Memory Tradeoff Curve  `[DEEP DIVE]`

### 11b.6.1 What we want to know

The key practical question for a serving engineer is: **how much KV budget
do I actually need to maintain acceptable quality?**

"Acceptable quality" is typically defined as perplexity within 1% of the
full-attention baseline (no eviction).  The companion code sweeps the
budget from 10% to 100% and measures a perplexity proxy (the attention
output cosine similarity to ground truth).

### 11b.6.2 Empirical findings from the literature

```
BUDGET VS. PERPLEXITY (typical LLaMA 3 8B, 4K context, H2O eviction)
──────────────────────────────────────────────────────────────────────

 KV budget   Perplexity    Degradation vs. full
 ──────────────────────────────────────────────
  100%         8.12           baseline (no eviction)
   80%         8.15           +0.4%    ← nearly free
   60%         8.22           +1.2%    ← marginal
   40%         8.41           +3.6%    ← noticeable
   20%         9.04          +11.3%    ← significant
   10%        10.87          +33.9%    ← severe

 "1% degradation target" → budget ≈ 55-65% is sufficient

 With SnapKV (clustered, not random eviction):
   60%         8.18           +0.7%    ← better than H2O at 60%
   40%         8.27           +1.8%    ← better than H2O at 40%
```

### 11b.6.3 Task-dependency

The budget required varies substantially by task:

| Task | Required budget (1% PPL target) |
|------|--------------------------------|
| Open-ended generation | ~55% |
| Multi-hop QA | ~70% |
| summarization | ~50% |
| Retrieval / needle-in-haystack | ~80% |
| Code generation | ~65% |

Retrieval tasks are sensitive because any evicted token might be **the**
needle.  SnapKV's observation-window approach helps here: the last few
tokens of the prompt (which often contain the query) guide which earlier
tokens survive.

### 11b.6.4 Memory savings at scale

```
WORKED EXAMPLE 11b.3 — Memory Savings at 60% Budget
──────────────────────────────────────────────────────────────────────

Given:
  Model:          LLaMA 3 70B (GQA, 8 KV heads, d_head=128, 80 layers)
  Precision:      BF16 (2 bytes per element)
  Context length: 32 768 tokens
  GPU:            H100 SXM5 80 GB

Full KV cache size:
  2 × 80 × 32768 × 8 × 128 × 2 bytes
  = 2 × 80 × 32768 × 8 × 128 × 2
  = 10 737 418 240 bytes ≈ 10.0 GB

At 60% budget:
  10.0 GB × 0.60 = 6.0 GB

Saving per sequence: 4.0 GB
On 4 concurrent long-context requests: 16.0 GB saved
→ Room for 2–3 additional concurrent requests on the same GPU
```

---

## 11b.7 vLLM Integration Points  `[DEEP DIVE]`

### 11b.7.1 Where eviction fits in the block manager

vLLM's `BlockSpaceManager` (vllm/core/block_manager.py) manages the physical
block table.  Eviction hooks into this at two points:

```
VLLM EVICTION INTEGRATION POINTS
──────────────────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────────┐
  │                    LLMEngine.step()                             │
  │                          │                                      │
  │              ┌───────────▼───────────┐                         │
  │              │     Scheduler         │                         │
  │              │  .schedule()          │                         │
  │              └───────────┬───────────┘                         │
  │                          │ SequenceGroupMetadata                │
  │              ┌───────────▼───────────┐                         │
  │              │  BlockSpaceManager    │  ← EVICTION HOOK 1:     │
  │              │  .allocate()          │    before allocation,   │
  │              │  .can_allocate()      │    check if eviction    │
  │              │  .free()              │    needed               │
  │              └───────────┬───────────┘                         │
  │                          │ block_tables                         │
  │              ┌───────────▼───────────┐                         │
  │              │   Worker / GPU        │  ← EVICTION HOOK 2:     │
  │              │   execute_model()     │    attention layer can   │
  │              │   paged_attn_kernel   │    skip evicted block    │
  │              └───────────────────────┘    indices              │
  └─────────────────────────────────────────────────────────────────┘
```

**Hook 1 — Block-level eviction (coarse)**:
Before allocating a new block, if no free blocks remain, evict the entire
block with the lowest average accumulated attention score.  This operates
at block granularity (default 16 tokens per block).

**Hook 2 — Token-level eviction (fine, requires kernel support)**:
The PagedAttention CUDA kernel can be modified to skip specific
block-offset positions, effectively nullifying them for the attention
computation without physically freeing the memory.

### 11b.7.2 The block granularity constraint

`[COMMON TRAP]` vLLM's physical blocks are atomic units of allocation.
You cannot evict half a block.  If block size = 16 and only 2 tokens in
a block are "low value," you must either evict all 16 or none.  This means
H2O-style token-level eviction requires either:

1. A modified kernel that masks individual positions within a block
   (high implementation complexity).
2. Operating at block granularity with a block-level importance score
   (average or max of token scores within the block).

vLLM's `prefix_caching` (radix attention) already uses block-level
hashing; block-level eviction is a natural extension.

### 11b.7.3 Attaching an eviction policy

A minimal eviction hook looks like:

```python
class H2OBlockManager(BlockSpaceManager):
    """BlockSpaceManager subclass with H2O eviction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Map from physical block_id → accumulated attention score
        self.block_scores: dict[int, float] = {}

    def update_scores(self, seq_id: int, attn_weights: list[float]) -> None:
        """Called after each decode step with per-token attention weights."""
        block_table = self.get_block_table(seq_id)
        for block_idx, block_id in enumerate(block_table):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(attn_weights))
            block_score = sum(attn_weights[start:end])
            self.block_scores[block_id] = (
                self.block_scores.get(block_id, 0.0) + block_score
            )

    def evict_one_block(self, protect_n_sink_blocks: int = 1) -> int:
        """Evict the lowest-scoring non-sink block. Return its ID."""
        candidates = {
            bid: score
            for bid, score in self.block_scores.items()
            if self._block_rank(bid) >= protect_n_sink_blocks
        }
        victim = min(candidates, key=candidates.get)
        self.free_block(victim)
        del self.block_scores[victim]
        return victim
```

This is illustrative pseudocode; a production implementation must also
handle CoW (copy-on-write) blocks and prefix-cached blocks that should
never be evicted.

---

## 11b.8 llama.cpp's KV Cache Shifting  `[DEEP DIVE]`

### 11b.8.1 The `llama_kv_cache_seq_rm` approach

llama.cpp takes a simpler, CPU-friendly approach to KV cache management.
The core function is `llama_kv_cache_seq_rm` (llama.cpp source:
`src/llama.cpp`, function `llama_kv_cache_seq_rm`):

```c
// Remove tokens in the range [p0, p1) from sequence seq_id's KV cache.
// If p0 == -1, remove from beginning.
// If p1 == -1, remove to end.
bool llama_kv_cache_seq_rm(
        struct llama_context * ctx,
                   llama_seq_id   seq_id,
                   llama_pos      p0,
                   llama_pos      p1);
```

After removal, the positions are physically compacted by `llama_kv_cache_seq_add`:

```c
// Shift all positions in [p0, p1) by delta.
// Used after seq_rm to close the gap left by removed tokens.
bool llama_kv_cache_seq_add(
        struct llama_context * ctx,
                   llama_seq_id   seq_id,
                   llama_pos      p0,
                   llama_pos      p1,
                   llama_pos      delta);
```

### 11b.8.2 The sliding window pattern

The most common pattern in llama.cpp long-context handling is a **sliding
window** that keeps the first A tokens (sinks) and the last W tokens (recent):

```
LLAMA.CPP SLIDING WINDOW KV EVICTION
──────────────────────────────────────────────────────────────────────

 Context length:  n_ctx = 2048  (hard limit)
 Sink window:     n_keep = 4    (always kept)
 Full context:    current_length = 2048  (at capacity)
 New token to add: position 2048

 Problem: cannot write KV at position 2048 (out of bounds)

 Solution (the "context shift"):

 Step 1 — Remove the "middle" tokens:
   llama_kv_cache_seq_rm(ctx, 0, n_keep, n_keep + n_discard)
   where n_discard = n_ctx / 2  (discard oldest half of non-sink tokens)

 Step 2 — Shift remaining tokens left:
   llama_kv_cache_seq_add(ctx, 0, n_keep + n_discard, -1, -n_discard)
   (move positions [n_keep+n_discard .. end] left by n_discard steps)

 Before shift:
   [0 1 2 3][4 5 6 7 ... 1027][1028 1029 ... 2047]
    ↑sinks↑   ← discarded →    ←   kept recent   →

 After shift:
   [0 1 2 3][1028 1029 ... 2047]
    ↑sinks↑   ← compacted ────→

 New position available: 1028 (room for n_discard new tokens)
```

### 11b.8.3 Performance implications

The `seq_rm` + `seq_add` sequence involves:
1. Zeroing out removed positions in the KV cache tensors.
2. Copying remaining KV data forward in memory (or updating position
   embeddings via `seq_add`).

For RoPE-based models, `seq_add` modifies the stored positions to reflect
the shift, and the attention kernel uses these updated positions for RoPE
computation.  This is O(N · d_head) memory bandwidth — cheap relative to
a forward pass, but non-trivial at N = 128K.

`[COMMON TRAP]` The model was never trained on sequences where the "first"
token after a context shift has a different position index than it was
originally assigned.  Position embeddings encode absolute position;
after a shift, position 1028 is now at cache slot 4, which the model
perceives as "close to position 4."  Quality can degrade on
position-sensitive tasks (e.g., "what was the 500th word you mentioned?").

### 11b.8.4 llama.cpp configuration flags

```
LLAMA.CPP FLAGS FOR KV MANAGEMENT
──────────────────────────────────────────────────────────────────────

Flag                    Effect
──────────────────────────────────────────────────────────────────────
--ctx-size N            Sets n_ctx (KV cache capacity in tokens)
-n N / --n-predict N    Stops before overflow without explicit eviction
--keep N                In llama-server/main, keep first N tokens
                        as sinks before context shift
--cache-type-k TYPE     KV key precision: f16, q8_0, q4_0, q4_1
--cache-type-v TYPE     KV value precision (same options)
--defrag-thold F        Trigger KV defragmentation when fragmentation
                        exceeds fraction F (default 0.1)
──────────────────────────────────────────────────────────────────────

Quantizing the KV cache (--cache-type-k q8_0 --cache-type-v q8_0)
reduces memory by ~2× at the cost of a small quality penalty.
Combined with H2O-style eviction, this is the recommended approach
for long-context deployment on memory-constrained hardware.
```

---

## Chapter Summary

| Concept | Key Takeaway |
|---------|-------------|
| Full cache problem | At 128K+ context, cache overflow is the default, not an edge case |
| Attention sinks | First 4 tokens absorb disproportionate attention; evicting them breaks the model |
| H2O eviction | Accumulate attention scores; evict lowest-scoring non-sink tokens |
| SnapKV compression | Use observation-window attention to identify important prompt tokens; pool nearby keys |
| CaM merge-at-write | Merge new tokens with similar cached keys, keeping the cache permanently smaller |
| Accuracy-memory curve | ~60% KV budget retains quality within 1% perplexity on most tasks |
| vLLM integration | Eviction hooks into `BlockSpaceManager` at block granularity; token-level requires kernel support |
| llama.cpp shifting | `seq_rm` + `seq_add` implements sliding-window eviction; cheap but position-embedding aware |

---

## Self-Check Questions

**Q1 [FOUNDATIONAL]** — A model is processing a 100 K-token document and the
KV cache is full at 64 K slots.  Without any eviction, what happens when the
scheduler tries to decode the next token?  Name two safe strategies and one
unsafe strategy the serving system might use.

**Q2 [FOUNDATIONAL]** — Explain in one paragraph why the first 4 tokens of a
sequence should almost always be retained during eviction, even if their
accumulated attention score is low.  What property of the softmax function
is responsible?

**Q3 [DEEP DIVE]** — In H2O, should you use total accumulated attention score
or a recency-weighted (exponentially decayed) score?  Describe a task where
each choice is clearly better.

**Q4 [DEEP DIVE]** — SnapKV uses the last W tokens of the prompt as an
"observation window" to determine which earlier tokens to keep.  Why is this
choice better than using a random window?  What type of task would break this
assumption?

**Q5 [DEEP DIVE]** — A vLLM deployment has block_size=16 and wants to
implement H2O.  A sequence has 8 blocks.  Blocks 3, 5, and 7 have the
three lowest block-level importance scores.  If the scheduler must free
exactly 2 blocks to accommodate a new sequence, which blocks should it
evict, and what must it do about the block table to maintain a valid
attention computation for the surviving sequence?

---

*End of Chapter 11.5*
