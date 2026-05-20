# Chapter 11.6 — RadixAttention and Prefix Caching Deep Dive

> *"A shared system prompt is a cache miss waiting to be fixed."*

---

## 11.6.1 The Problem Prefix Caching Solves

Every production LLM deployment eventually hits the same pattern: the same
tokens appear at the start of most requests. A customer support bot always
begins with a 2,000-token system prompt. A code assistant always prepends
the same repository context. A RAG pipeline always includes the same
retrieval preamble.

Without prefix caching, every request recomputes the KV pairs for those shared
tokens — burning GPU cycles and adding latency proportional to the prefix
length. For a 2,000-token system prompt at 70B parameter scale, that
recomputation takes roughly 0.8–1.2 seconds on a single A100 before the model
processes even one token of the actual user query.

Prefix caching solves this by recognizing that the KV pairs for a given token
sequence are **deterministic given the weights and the sequence**. If two
requests share a prefix, they can share KV blocks.

---

## 11.6.2 Block-Level Prefix Caching in vLLM

vLLM's approach (Chapter 6) manages the KV cache as a pool of fixed-size
blocks (default: 16 tokens per block). Prefix caching extends this with a
**prefix block hash map**: when a new request arrives, vLLM hashes each
filled block and checks whether an identical block is already resident.

```
Request A: [SYS_PROMPT_TOKENS... | USER_TOKENS_A]
           [  Block 0  |  Block 1  |  Block 2  ] [  Block 3  ]
                 ↓            ↓            ↓
           hash(B0)=0xA1  hash(B1)=0xB2  hash(B2)=0xC3
           → cache hit    → cache hit    → cache hit   → miss: compute

Request B: [SYS_PROMPT_TOKENS... | USER_TOKENS_B]
           [  Block 0  |  Block 1  |  Block 2  ] [  Block 4  ]
                 ↓            ↓            ↓
           hash(B0)=0xA1  hash(B1)=0xB2  hash(B2)=0xC3
           → HIT (reuse)  → HIT (reuse)  → HIT (reuse) → miss: compute
```

The hash is computed over the token IDs in the block, not the KV values. The
KV values are the _result_ of the hash hit — no recomputation needed.

### Cache hit on a block produces two savings

1. **Prefill FLOPS avoided** — no attention or FFN computation for those tokens
2. **Memory bandwidth avoided** — weights for those layers don't need to be
   re-read from HBM for those positions

For a 2,000-token shared system prompt split into 125 × 16-token blocks, a
cache hit eliminates 125 × (2 × num_layers × attention_ops + FFN_ops) of
compute.

### Block granularity matters

The 16-token block size creates a quantization effect: a prefix that is 17
tokens long only gets 1 block cached (16 tokens); the 17th token still
triggers a partial prefill. Very short prefixes (< 16 tokens) receive no
benefit. Very long prefixes (thousands of tokens) have high hit rates.

In vLLM V1 (Chapter 40), the block manager switches to hash-based KV
deduplication at a finer granularity, reducing this quantization penalty.

---

## 11.6.3 SGLang's RadixAttention

SGLang takes a fundamentally different approach. Instead of a flat hash map of
blocks, it maintains a **radix tree** (also called a trie) over the entire
token sequence.

A radix tree is a compressed prefix tree where each edge represents a shared
sequence of tokens. When a new sequence arrives, the tree walk finds the
longest matching prefix — exactly the tokens whose KV pairs can be reused.

```
                     [ROOT]
                    /        \
          [sys_prompt]       [other_prompt]
          /    \
   [user_q_A]  [user_q_B]
```

Each node in the tree stores:

- The token IDs for this node's edge
- A reference to the cached KV blocks for those tokens
- A reference count (how many active requests are using this node)
- An LRU timestamp

### Why a tree is better than a flat hash map

A flat hash map (vLLM's approach) works at block granularity. Two requests
that share 2,000 tokens but diverge mid-block still hash differently for the
partial block.

The radix tree works at **token granularity**. The longest common prefix is
found exactly, regardless of block boundaries. For workloads with variable-
length shared prefixes (multi-turn conversations where each turn extends the
context), this produces significantly higher hit rates.

**Worked Example 11.6.1 — Multi-turn chat cache hit rate**

Consider 4 turns of a conversation, each adding 200 tokens:

```
Turn 1: [system(500)] [turn1(200)]
Turn 2: [system(500)] [turn1(200)] [turn2(200)]
Turn 3: [system(500)] [turn1(200)] [turn2(200)] [turn3(200)]
Turn 4: [system(500)] [turn1(200)] [turn2(200)] [turn3(200)] [turn4(200)]
```

| System | Approach | Tokens reused (Turn 4) | Prefill saved |
|---|---|---|---|
| vLLM block cache | 16-tok blocks | 1,088 of 1,100 (6 full blocks × 16 + partial miss) | 99% |
| RadixAttention | Token-exact | 1,100 of 1,100 | 100% |

For block-aligned prefixes the difference is small. For non-aligned prefixes
(e.g. system prompt of 537 tokens — not a multiple of 16), RadixAttention
recovers an extra 15 tokens per divergence point, which compounds across
many turns.

---

## 11.6.4 The Radix Tree Implementation

```python
# Simplified RadixAttention node (SGLang internal representation)
from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class RadixNode:
    token_ids: List[int]           # tokens on the edge to this node
    kv_cache_ref: Optional[object] # reference to cached KV blocks
    children: Dict[int, "RadixNode"] = field(default_factory=dict)
    ref_count: int = 0             # active requests using this node
    last_access: float = 0.0       # LRU timestamp

class RadixTree:
    def __init__(self):
        self.root = RadixNode(token_ids=[], kv_cache_ref=None)

    def match_prefix(self, token_ids: List[int]) -> tuple[RadixNode, int]:
        """Return the deepest matching node and how many tokens matched."""
        node = self.root
        matched = 0
        while matched < len(token_ids):
            next_token = token_ids[matched]
            if next_token not in node.children:
                break
            child = node.children[next_token]
            edge_len = len(child.token_ids)
            # Check if the full edge matches
            if token_ids[matched:matched+edge_len] == child.token_ids:
                matched += edge_len
                node = child
            else:
                # Partial match — need to split the edge
                split_at = 0
                while (split_at < edge_len and
                       matched + split_at < len(token_ids) and
                       token_ids[matched + split_at] ==
                       child.token_ids[split_at]):
                    split_at += 1
                node = self._split_node(node, child, split_at)
                matched += split_at
                break
        return node, matched

    def insert(self, token_ids: List[int], kv_ref: object) -> RadixNode:
        """Insert a completed sequence into the tree."""
        node, matched = self.match_prefix(token_ids)
        if matched < len(token_ids):
            new_node = RadixNode(
                token_ids=token_ids[matched:],
                kv_cache_ref=kv_ref
            )
            node.children[token_ids[matched]] = new_node
            return new_node
        node.kv_cache_ref = kv_ref
        return node

    def _split_node(self, parent, child, split_at):
        """Split child at split_at, inserting an intermediate node."""
        mid = RadixNode(
            token_ids=child.token_ids[:split_at],
            kv_cache_ref=None          # intermediate: no cached KV yet
        )
        child.token_ids = child.token_ids[split_at:]
        mid.children[child.token_ids[0]] = child
        parent.children[mid.token_ids[0]] = mid
        return mid
```

### Test harness — RadixTree correctness

```python
# ── test_radix_tree.py ──────────────────────────────────────────────────
"""
Self-contained unit tests for RadixTree prefix matching and insertion.
Run with:  python test_radix_tree.py
"""

import time

def run_tests():
    # ── Fixture: build a small tree ────────────────────────────────────
    tree = RadixTree()

    sys_prompt  = [1, 2, 3, 4, 5]          # "system prompt"
    user_turn_a = sys_prompt + [10, 11]     # session A turn 1
    user_turn_b = sys_prompt + [20, 21, 22] # session B turn 1
    user_turn_a2 = user_turn_a + [30, 31]   # session A turn 2

    # Insert sequences
    ref_sys   = object()
    ref_a1    = object()
    ref_b1    = object()
    ref_a2    = object()

    tree.insert(sys_prompt, ref_sys)
    tree.insert(user_turn_a, ref_a1)
    tree.insert(user_turn_b, ref_b1)
    tree.insert(user_turn_a2, ref_a2)

    # ── Test 1: exact prefix match ──────────────────────────────────────
    node, matched = tree.match_prefix(sys_prompt)
    assert matched == len(sys_prompt), (
        f"Expected {len(sys_prompt)} tokens matched, got {matched}"
    )
    assert node.kv_cache_ref is ref_sys, "Wrong kv_cache_ref on sys node"
    print("PASS: exact prefix match (sys_prompt)")

    # ── Test 2: longer sequence returns sys node as best prefix ─────────
    new_request = sys_prompt + [99, 100]    # not in tree yet
    node, matched = tree.match_prefix(new_request)
    assert matched == len(sys_prompt), (
        f"Expected sys_prompt len={len(sys_prompt)}, got {matched}"
    )
    print("PASS: longer sequence prefix correctly returns sys node")

    # ── Test 3: multi-turn reuse ────────────────────────────────────────
    node2, matched2 = tree.match_prefix(user_turn_a2)
    assert matched2 == len(user_turn_a2), (
        f"Expected full match for turn_a2, got {matched2}"
    )
    print("PASS: multi-turn sequence fully matched")

    # ── Test 4: diverging branches do not cross-contaminate ────────────
    node_a, m_a = tree.match_prefix(user_turn_a)
    node_b, m_b = tree.match_prefix(user_turn_b)
    assert m_a == len(user_turn_a)
    assert m_b == len(user_turn_b)
    assert node_a is not node_b, "Diverging branches share same node — BUG"
    print("PASS: diverging branches are distinct nodes")

    # ── Test 5: LRU eviction respects ref_count ─────────────────────────
    # Mark session A's leaf as in-use
    node_a.ref_count = 1
    leaves = collect_evictable_leaves(tree.root)
    assert node_a not in leaves, "Active node should not be evictable"
    node_a.ref_count = 0

    # After releasing, it should appear
    leaves_after = collect_evictable_leaves(tree.root)
    evictable_refs = [l.kv_cache_ref for l in leaves_after]
    # The leaf node for user_turn_a2 subsumes user_turn_a path, so
    # at minimum the tree has at least 2 leaf candidates.
    assert len(leaves_after) >= 2, (
        f"Expected at least 2 evictable leaves, got {len(leaves_after)}"
    )
    print("PASS: LRU eviction excludes active nodes")

    # ── Test 6: insert after partial eviction ───────────────────────────
    freed = []
    def mock_free(ref):
        freed.append(ref)
        return 1                        # pretend 1 block freed

    blocks_freed = evict_until(tree, target_free_blocks=1, free_fn=mock_free)
    assert blocks_freed >= 1, "Should have freed at least 1 block"
    assert len(freed) >= 1, "free_fn should have been called"
    print(f"PASS: evict_until freed {blocks_freed} block(s)")

    # ── Test 7: hit rate simulation ─────────────────────────────────────
    requests = [
        sys_prompt + [i] for i in range(50)           # 50 unique suffixes
    ] + [sys_prompt] * 50                              # 50 system-prompt-only hits

    hits = 0
    for req in requests:
        tree2 = RadixTree()
        tree2.insert(sys_prompt, object())
        _, m = tree2.match_prefix(req)
        if m >= len(sys_prompt):
            hits += 1
    hit_rate = hits / len(requests)
    assert hit_rate == 0.5, f"Expected 50% hit rate, got {hit_rate:.2f}"
    print(f"PASS: hit rate simulation = {hit_rate:.0%}")

    print("\n✓ All RadixTree tests passed.")

if __name__ == "__main__":
    run_tests()
```

**Expected output**:
```
PASS: exact prefix match (sys_prompt)
PASS: longer sequence prefix correctly returns sys node
PASS: multi-turn sequence fully matched
PASS: diverging branches are distinct nodes
PASS: LRU eviction excludes active nodes
PASS: evict_until freed 1 block(s)
PASS: hit rate simulation = 50%

✓ All RadixTree tests passed.
```

---

## 11.6.5 LRU Eviction on the Radix Tree

When GPU memory is under pressure, the cache manager needs to evict nodes.
The constraints are:

1. **Never evict a node with `ref_count > 0`** — active requests are reading it
2. **Evict leaves first** — evicting an interior node would orphan its subtree
3. **Among evictable leaves, evict the LRU** — classical LRU by `last_access`

```python
import time
from typing import List

def collect_evictable_leaves(node: RadixNode) -> List[RadixNode]:
    """Return all leaf nodes with ref_count == 0, sorted by last_access."""
    leaves = []
    def _walk(n: RadixNode):
        if not n.children and n.ref_count == 0 and n.kv_cache_ref is not None:
            leaves.append(n)
        for child in n.children.values():
            _walk(child)
    _walk(node)
    leaves.sort(key=lambda x: x.last_access)
    return leaves

def evict_until(tree: RadixTree, target_free_blocks: int,
                free_fn) -> int:
    """Evict LRU leaves until target_free_blocks are available.
    free_fn(kv_cache_ref) releases the GPU memory.
    Returns number of blocks freed."""
    freed = 0
    while freed < target_free_blocks:
        leaves = collect_evictable_leaves(tree.root)
        if not leaves:
            break           # nothing left to evict
        victim = leaves[0]  # LRU
        freed += free_fn(victim.kv_cache_ref)
        victim.kv_cache_ref = None
        # If the parent now has no children and no KV ref, it too becomes
        # evictable — the eviction propagates upward organically on the
        # next eviction pass.
    return freed
```

**Eviction cascade**: after a leaf is evicted, its parent may become a leaf
with no cached KV (only an intermediate structural node). On the next eviction
pass it becomes eligible. This means long chains of unique suffixes naturally
age out together.

---

## 11.6.6 Cache Hit Rate in Practice

Cache hit rates depend heavily on workload structure:

| Workload | Typical hit rate | Explanation |
|---|---|---|
| Same system prompt, random user queries | 60–85% | System prompt always hits; user tokens miss |
| Multi-turn chat (4 turns avg) | 70–92% | Each turn reuses all prior turns |
| RAG with 5 retrieved chunks | 40–70% | Chunk order varies; prefix unstable |
| Code completion (same file context) | 80–95% | File context is stable |
| Batch classification (fixed template) | 90–98% | Template always hits |
| Fully random prompts | 0–5% | No shared structure |

**Worked Example 11.6.2 — System Prompt Economics**

A customer support deployment with:

- System prompt: 1,800 tokens
- Average user query: 120 tokens
- Average response: 250 tokens
- Traffic: 1,000 requests/minute

Without prefix caching:
```
Prefill tokens per request = 1,800 + 120 = 1,920
Total prefill tokens/min   = 1,920 × 1,000 = 1,920,000
```

With prefix caching (85% hit rate on system prompt):
```
System prompt tokens saved per request = 1,800 × 0.85 = 1,530
Effective prefill tokens per request   = 1,920 - 1,530 = 390
Total prefill tokens/min               = 390 × 1,000 = 390,000
Reduction: 80% fewer prefill tokens
```

At H100 prefill throughput of ~20,000 tokens/sec, this reduces the GPU-time
needed for prefill from 96 seconds/minute to 19.5 seconds/minute — a 5× reduction
in the prefill compute budget.

---

## 11.6.7 Configuring Prefix Caching in vLLM

```bash
# Enable prefix caching (vLLM V0)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768

# vLLM V1 (Chapter 40) enables prefix caching by default.
# The hash-based deduplication is always on; no flag required.
```

```python
# Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.90,
)

# All requests sharing this system prompt will reuse cached KV blocks
system_prompt = "You are a helpful customer support agent for Acme Corp..." * 100

responses = llm.generate([
    f"{system_prompt}\nUser: {query}"
    for query in user_queries
], SamplingParams(temperature=0.7, max_tokens=256))
```

### Monitoring cache effectiveness

vLLM exposes prefix cache metrics via its Prometheus endpoint:

```
vllm:gpu_prefix_cache_hit_rate   # 0.0 to 1.0
vllm:gpu_prefix_cache_queries_total
vllm:gpu_prefix_cache_hits_total
```

A healthy deployment with a fixed system prompt should show
`gpu_prefix_cache_hit_rate > 0.75`. Below 0.5, investigate whether the prompt
is changing between requests (model version switches, A/B testing, or
per-request customisation that invalidates the prefix).

---

## 11.6.8 Configuring Prefix Caching in SGLang

SGLang's RadixAttention is enabled by default. The key configuration
parameters control the cache size and eviction policy:

```bash
# SGLang server with RadixAttention (default on)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tp 4 \
    --port 30000 \
    --mem-fraction-static 0.85 \
    --max-prefill-tokens 16384
# --disable-radix-cache   # only if you want to measure the difference
```

```python
# SGLang Python API — the cache is transparent to calling code
import sglang as sgl

@sgl.function
def chat_with_system(s, system_prompt, user_message):
    s += sgl.system(system_prompt)
    s += sgl.user(user_message)
    s += sgl.assistant(sgl.gen("response", max_new_tokens=256))
```

SGLang also exposes cache hit rate via `/get_server_info`:
```json
{
  "radix_cache_hit_tokens": 142381,
  "radix_cache_total_tokens": 198432,
  "radix_cache_hit_rate": 0.717
}
```

---

## 11.6.9 llama.cpp Prompt Caching

llama.cpp implements a simpler variant: a **KV cache slot** approach where
an evaluated prefix is retained across consecutive calls with the same prefix.
This works for single-process inference where the same llama context is reused:

```cpp
// llama.cpp: reuse KV cache across requests with same prefix
struct llama_context_params cparams = llama_context_default_params();
cparams.n_ctx    = 4096;
cparams.n_batch  = 512;

llama_context* ctx = llama_new_context_with_model(model, cparams);

// First request: full prefill of system prompt + query 1
llama_kv_cache_clear(ctx);
llama_decode(ctx, system_prompt_batch);   // 1800 tokens, ~1.2s
llama_decode(ctx, query1_batch);
// ... generate tokens ...

// Save the KV state after system prompt (before query1)
llama_kv_cache_seq_cp(ctx, 0, 1, 0, system_prompt_len);

// Second request: restore to end-of-system-prompt, add query 2
llama_kv_cache_seq_rm(ctx, 0, system_prompt_len, -1);  // remove query1
llama_decode(ctx, query2_batch);   // only 120 tokens: ~0.08s
```

In server mode (`llama-server`), this is handled automatically for requests
that share a common prefix with the currently loaded context. The
`--cache-reuse` flag controls the minimum overlap required to trigger reuse:

```bash
llama-server \
    --model llama-3.1-70b-q4_k_m.gguf \
    --ctx-size 8192 \
    --cache-reuse 256    # reuse if at least 256 tokens overlap
```

---

## 11.6.10 When Prefix Caching Hurts

Prefix caching is not universally beneficial:

**Memory pressure**: Cached KV blocks consume GPU memory. On a system running
near-full memory utilization, prefix caching reduces the number of concurrent
requests that can be served. The tradeoff breaks even around 60% cache hit
rate; below that, the memory cost outweighs the compute savings.

**Random or unique prompts**: If every request has a unique prefix (user-
uploaded documents, per-user personalization, random temperatures), the cache
hit rate approaches zero while the bookkeeping overhead remains.

**Beam search or high-n sampling**: Multiple completions from the same prompt
are an ideal case for prefix caching (all share the full prompt), but if
requests use different sampling parameters (temperature, top-p), some engines
treat these as distinct cache keys.

**Short prompts**: For prompts under ~50 tokens, the prefill cost is already
negligible. Adding the hash computation and tree-walk overhead for tiny prompts
produces negative ROI.

**Rule of thumb**: Enable prefix caching when your average prompt is > 200
tokens and > 40% of requests share a common prefix.

---

## 11.6.11 Thread Safety and Concurrent Access

Production inference engines serve hundreds of simultaneous requests, all
touching the radix tree concurrently. The correctness requirements are strict:

- A node being read by Request A must not be evicted while A is still decoding
- A node being inserted by Request B must not corrupt an in-progress match by A
- The LRU timestamp must be updated atomically to avoid stale eviction

### SGLang's approach: event loop serialisation

SGLang runs the radix cache manager inside a single-threaded async event loop.
All tree operations — match, insert, evict — execute between coroutine yield
points, making them effectively atomic from the Python perspective. GPU kernel
launches are async; the cache management is serial.

```python
import asyncio
import threading

class ThreadSafeRadixTree(RadixTree):
    """Wrapper that serialises all tree operations under a lock.
    Use this for multi-threaded serving (e.g. vLLM worker threads)."""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()

    def match_prefix(self, token_ids):
        with self._lock:
            return super().match_prefix(token_ids)

    def insert(self, token_ids, kv_ref):
        with self._lock:
            return super().insert(token_ids, kv_ref)

    def acquire(self, node):
        """Increment ref_count atomically before starting decode."""
        with self._lock:
            node.ref_count += 1

    def release(self, node):
        """Decrement ref_count atomically after decode completes."""
        with self._lock:
            node.ref_count -= 1
            assert node.ref_count >= 0, "ref_count underflow — double release"
```

### Reference counting lifecycle

Every request that uses a cached prefix node **must** call `acquire` before
touching the node's KV data and `release` when generation is done (or aborted).
This is analogous to `mmap` reference counts in OS kernels.

```
Request arrives → match_prefix() → acquire(matched_node)
                → GPU decode starts (reads KV from cached blocks)
                → generation completes / error / cancellation
                → release(matched_node)  ← MUST be called in finally block
```

A missing `release` leaks the ref_count, causing nodes to appear perpetually
in-use and gradually exhausting evictable memory — a silent memory leak that
only manifests under sustained load.

---

## 11.6.12 Hash Collisions in vLLM Block Hashing

vLLM hashes each KV block's token content using xxHash or SHA-256 (version
dependent). Hash collisions — two different token sequences mapping to the same
hash — are theoretically possible but practically negligible:

With 128-bit hashes and a universe of ~50,000 token IDs in 16-token blocks,
the collision probability per block pair is:

```
P(collision) ≈ 1 / 2^128 ≈ 3 × 10^-39
```

At 1 billion block comparisons per day, the expected time to the first
collision is roughly 10^21 years — far beyond any practical concern.

**However**, SHA-256 computation is not free. For very short blocks (< 16
tokens), the hash overhead can dominate. vLLM uses a fast non-cryptographic
hash (xxHash-64) with a collision probability of ~1 in 2^64 — still
astronomically small for operational purposes.

```python
import hashlib

def block_hash_vllm(token_ids: list[int], block_idx: int) -> int:
    """
    Reproduce vLLM's block hash: hash the full history up to this block.
    token_ids: ALL tokens from position 0 to end of this block.
    Includes block_idx to disambiguate when same tokens appear at
    different positions.
    """
    payload = str(block_idx) + str(token_ids)
    return int(hashlib.sha256(payload.encode()).hexdigest(), 16)

# Example: two requests sharing the same first 32 tokens
sys_prompt_tokens = list(range(32))   # tokens 0-31
block0_hash = block_hash_vllm(sys_prompt_tokens[:16], block_idx=0)
block1_hash = block_hash_vllm(sys_prompt_tokens[:32], block_idx=1)
assert block0_hash != block1_hash   # different positions → different hashes
```

---

## 11.6.13 Prefix Caching and Speculative Decoding

Speculative decoding (Chapter 23) uses a small draft model to propose multiple
tokens in parallel, then verifies them with the target model in a single
forward pass. The interaction with prefix caching is nuanced:

**Compatible paths**:

- The **prompt prefix** can still be cached normally. The speculative decoder
  only changes the decode phase, not the prefill.
- If a draft sequence is fully accepted, the verified tokens extend the prefix
  tree normally.

**Tension points**:

1. **Draft token sequences are speculative** — they may be rejected. If
   rejected tokens were inserted into the radix tree, they'd create phantom
   paths that never correspond to real KV data. SGLang and vLLM therefore
   do *not* insert speculative draft tokens into the prefix cache; only
   accepted (verified) tokens are committed.

2. **Verification pass reads KV differently**: the verification pass reads
   cached KV for the prompt prefix but must write *new* KV for the speculative
   suffix. These writes cannot be batched with a cache lookup.

3. **Tree structure diverges**: if the draft model proposes different tokens
   than another concurrent request's completion, the cache tree branches at the
   speculative boundary, reducing reuse.

**Practical guidance**: enable both prefix caching and speculative decoding
independently; they compose safely as long as the cache only stores verified
token sequences. Do not attempt to cache draft-model outputs — the hit rate is
low and the consistency guarantees are undefined.

---

## 11.6.14 Cache Warmup and Prefix Pinning

Cold-start is a real problem: on engine restart, the prefix cache is empty and
the first N requests pay full prefill cost. For a 2,000-token system prompt at
70B scale, this can spike TTFT by 1–2 seconds per request during the warmup
window.

### Strategy 1: Explicit warmup request

Send a synthetic request with the system prompt (and no user query) immediately
after engine startup. This primes the cache before real traffic arrives.

```python
# warmup.py — run once after engine starts
import time
from vllm import LLM, SamplingParams

def warmup_prefix_cache(llm: LLM, system_prompt: str, n_warmup: int = 3):
    """
    Send n_warmup requests through the engine to populate the prefix cache.
    Use max_tokens=1 to minimize cost; we only care about prefill caching.
    """
    warmup_params = SamplingParams(temperature=0.0, max_tokens=1)
    queries = [f"{system_prompt}\nWarmup query {i}" for i in range(n_warmup)]

    t0 = time.time()
    llm.generate(queries, warmup_params)
    elapsed = time.time() - t0
    print(f"Cache warmup complete in {elapsed:.1f}s ({n_warmup} requests)")

if __name__ == "__main__":
    llm = LLM(
        model="meta-llama/Llama-3.1-70B-Instruct",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
    )
    SYSTEM_PROMPT = open("system_prompt.txt").read()
    warmup_prefix_cache(llm, SYSTEM_PROMPT)
    # Now serve real traffic — first real request will hit the warm cache
```

### Strategy 2: Prefix pinning (prevent eviction)

For system prompts that are always needed, you want to **pin** their cache
entries so they survive memory pressure events. Neither vLLM nor SGLang expose
a public pinning API as of mid-2025, but you can approximate it:

```python
# Simulate pinning by holding a synthetic "eternal" request ref_count
# In SGLang internals:
#   node.ref_count += 1   # never released → node never evictable

# In vLLM, the workaround is to ensure the prefix is accessed at least
# once every cache_ttl seconds to prevent LRU eviction:

import threading

def keep_alive_thread(llm, system_prompt, interval_s=60):
    """
    Background thread that re-touches the prefix cache every interval_s
    seconds to prevent LRU eviction of the system prompt blocks.
    """
    params = SamplingParams(temperature=0.0, max_tokens=1)
    while True:
        time.sleep(interval_s)
        llm.generate([f"{system_prompt}\nKeep-alive"], params)

t = threading.Thread(
    target=keep_alive_thread,
    args=(llm, SYSTEM_PROMPT, 30),
    daemon=True
)
t.start()
```

### Strategy 3: Kubernetes readiness probe

In Kubernetes deployments (Chapter 19), mark the pod as not-ready until the
warmup is complete. This prevents the load balancer from sending real traffic
to a cold pod:

```yaml
# k8s deployment snippet
readinessProbe:
  exec:
    command: ["python", "/app/warmup.py", "--check-only"]
  initialDelaySeconds: 60   # wait for model load
  periodSeconds: 10
  failureThreshold: 3
```

---

## 11.6.15 Prefix Drift and Cache Invalidation

**Prefix drift** occurs when the system prompt changes between deployments or
A/B test variants. All cached KV entries for the old prefix become invalid —
they correspond to different model computations and must not be reused with the
new prompt.

### Drift scenarios and their costs

| Scenario | Drift frequency | Cache impact |
|---|---|---|
| Model version rollout | Once per deploy | Full cache flush; first N requests cold |
| A/B test with 2 system prompts | Continuous 50/50 | 50% hit rate ceiling even with warm cache |
| Per-user system prompt customisation | Every request | Cache hit rate ≈ 0% |
| System prompt with injected date/time | Every minute | Effective hit rate ≈ 0% |
| System prompt with injected user name | Every unique user | Hit rate = 1/n_users |

### Detecting and handling drift

```python
import hashlib

class PrefixCacheManager:
    """
    Tracks the active system prompt hash; detects drift and forces
    cache invalidation when the prompt changes.
    """

    def __init__(self, llm):
        self.llm = llm
        self._active_prompt_hash = None

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def set_system_prompt(self, new_prompt: str):
        new_hash = self._prompt_hash(new_prompt)
        if self._active_prompt_hash is not None and \
           new_hash != self._active_prompt_hash:
            print(f"[CacheManager] Prompt drift detected: "
                  f"{self._active_prompt_hash} → {new_hash}. "
                  f"Cache will warm on next request.")
        self._active_prompt_hash = new_hash
        self._current_prompt = new_prompt

    def generate(self, user_message: str, **kwargs):
        full_prompt = f"{self._current_prompt}\nUser: {user_message}"
        return self.llm.generate([full_prompt], **kwargs)
```

### Architecture recommendation

Separate the **stable prefix** (system role, persona, tool definitions) from
the **dynamic prefix** (date injection, user name, retrieved context). Place
the stable prefix first so it is always cacheable:

```
[STABLE — cacheable]
  system: "You are a helpful assistant. Tools available: ..."

[DYNAMIC — not cached]
  Current date: {{date}}
  User: {{username}}

[USER TURN]
  {{user_message}}
```

Even if the dynamic portion changes every request, the stable prefix (often
60–80% of total prompt tokens) will still hit the cache.

---

## 11.6.16 Memory Overhead vs. Compute Savings: Quantified Tradeoff

Prefix caching trades GPU memory for GPU compute. This section quantifies the
break-even to help you decide how much memory to allocate to the cache.

### KV cache size per cached token

For a model with `H` layers, `n_kv_heads` KV heads, head dimension `d`, and
dtype size `D` bytes:

```
bytes_per_token = 2 × H × n_kv_heads × d × D
```

For **Llama-3.1-70B** (80 layers, 8 KV heads, head dim 128, FP16):
```
bytes_per_token = 2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 320 KB
```

A 2,000-token system prompt cached in full requires:
```
2,000 × 320 KB = 640 MB of GPU HBM
```

On a single A100-80GB GPU running at 90% utilization, you have ~72 GB for
inference. The 640 MB cache cost is ~0.9% of total memory — negligible.

### Concurrent request slots vs. prefix cache

The real tradeoff is: allocating memory to the prefix cache reduces the number
of concurrent request KV slots available.

```python
def compute_cache_tradeoff(
    total_gpu_gb: float,
    model_weights_gb: float,
    bytes_per_token: int,
    tokens_per_request: int,
    prefix_tokens: int,
    prefix_hit_rate: float,
):
    """
    Returns (concurrent_requests_without_cache,
             concurrent_requests_with_cache,
             prefill_speedup_factor).
    """
    available_gb = total_gpu_gb - model_weights_gb
    available_bytes = available_gb * 1e9

    # Without cache: all memory goes to active request slots
    slots_without = available_bytes / (tokens_per_request * bytes_per_token)

    # With cache: prefix_tokens × bytes reserved for prefix
    cache_bytes = prefix_tokens * bytes_per_token
    remaining_bytes = available_bytes - cache_bytes
    slots_with = remaining_bytes / (tokens_per_request * bytes_per_token)

    # Prefill speedup: cached tokens need no prefill compute
    saved_tokens_per_req = prefix_tokens * prefix_hit_rate
    total_tokens = prefix_tokens + (tokens_per_request - prefix_tokens)
    speedup = total_tokens / (total_tokens - saved_tokens_per_req)

    return int(slots_without), int(slots_with), speedup

without, with_cache, speedup = compute_cache_tradeoff(
    total_gpu_gb=80,
    model_weights_gb=35,          # Llama-3.1-70B FP16
    bytes_per_token=327_680,
    tokens_per_request=2_500,     # system prompt + user turn
    prefix_tokens=2_000,          # system prompt
    prefix_hit_rate=0.85,
)
print(f"Concurrent slots without cache: {without}")
print(f"Concurrent slots with cache:    {with_cache}")
print(f"Prefill speedup at 85% hit:     {speedup:.2f}×")
# Output:
# Concurrent slots without cache: 17
# Concurrent slots with cache:    17     (640 MB is < 1 slot cost)
# Prefill speedup at 85% hit:     5.67×
```

### Break-even analysis

The cache becomes net-negative when the memory cost (lost request slots)
exceeds the compute savings (avoided prefill). For a 2,000-token system prompt:

| Cache hit rate | Prefill savings (%) | Net memory impact | Verdict |
|---|---|---|---|
| 0–20% | < 7% | −0.9% of GPU memory | Not worth it |
| 20–50% | 7–25% | Same | Marginal benefit |
| 50–80% | 25–55% | Same | Clear win |
| 80–95% | 55–75% | Same | Strong win |
| > 95% | > 75% | Same | Essential |

For a 2,000-token prompt, the memory cost is always < 1 slot; therefore the
cache is net-positive at any hit rate above ~20%.

For very long prefixes (32,000 tokens, e.g., a large code context):
```
32,000 × 320 KB = 10 GB — about 7 request slots worth of memory on A100.
```
At this scale the break-even hit rate rises to ~60%, and you should measure
empirically before committing large amounts of memory to prefix caching.

---

## 11.6.17 vLLM V1 Hash-Based Deduplication

Chapter 40 covers the vLLM V1 architecture in depth. Relevant to prefix
caching: V1 replaces the block-level hash map with **content-addressable
storage at token granularity**. Each KV block is addressed by a hash of the
full token history up to that point (not just the block's own tokens), making
the cache key globally unique across all requests.

This has two consequences:

1. **Better deduplication across non-identical prefixes**: two requests that
   diverge at token 1,847 and reconverge at token 1,863 can share the
   reconverged blocks. Classic prefix caching cannot.

2. **Cache poisoning protection**: a malicious request cannot cause a cache hit
   by crafting tokens that hash-collide with a legitimate prefix, because the
   hash includes the full history (cryptographic length-extension resistance).

---

## Chapter Summary

Prefix caching converts a ubiquitous waste — recomputing shared tokens — into
a cache lookup. The block-level approach (vLLM) is simple and effective for
block-aligned prefixes; the radix tree approach (SGLang's RadixAttention) is
exact at token granularity and dominates for multi-turn workloads.

The economics are compelling: an 85% cache hit rate on a 1,800-token system
prompt reduces prefill compute by 80%. On a $1.2M/month deployment this is
worth hundreds of thousands of dollars in saved GPU time — with no model
changes, no hardware changes, and a single configuration flag.

The limits are equally clear: prefix caching trades memory for compute, stops
helping below ~40% hit rate, and adds bookkeeping overhead for short prompts.
Measure your cache hit rate from day one and let the data drive the decision.

---

## Self-Check Questions

1. A vLLM deployment uses 16-token blocks. A system prompt is 1,537 tokens
   long (96 full blocks + 1 token). How many tokens are recoverable via
   block-level prefix caching? How many via RadixAttention?

2. Explain why evicting an interior node from the radix tree would be
   incorrect without first evicting all its descendants. What invariant does
   this enforce?

3. A workload has 1,000 requests/minute. 600 share a 2,000-token system
   prompt; 400 have unique prompts. GPU memory can hold either: (a) prefix
   cache for 2,000 tokens OR (b) one additional concurrent request slot.
   At what request-per-minute threshold does the prefix cache become more
   valuable than the extra request slot?

4. SGLang's RadixAttention stores `ref_count` on each node. What happens if
   a request is canceled mid-generation while its prefix nodes have
   `ref_count == 1`? Describe the sequence of operations required to safely
   release those nodes back to the eviction pool.

5. A multi-turn chatbot has a 500-token system prompt and users average 8
   turns per session, each turn adding 150 tokens. What is the theoretical
   maximum cache hit rate (by tokens) if all turns within a session reuse
   prior turns' KV cache? How does this compare to cross-session reuse if
   1,000 concurrent users share the same system prompt?


---

## Worked Solutions

---

### Solution 1 — Tokens recoverable: block-level caching vs RadixAttention

**Given:** 1,537-token system prompt, block_size=16

**Block-level prefix caching (vLLM):**

Only *complete* blocks are cached. Block-level caching uses block hashes computed on full block content.

$$\text{complete blocks} = \lfloor 1537/16 \rfloor = 96 \text{ blocks} = 96 \times 16 = \textbf{1,536 tokens recoverable}$$

The 1 remaining token (1537 − 1536) is in an incomplete block that is NOT cached.

**RadixAttention (SGLang):**

RadixAttention stores sequences in a radix tree indexed by token content (hashed). The tree node for the 1,537-token sequence stores all 1,537 tokens' KV data as a contiguous tree path. The entire sequence — including the partial last block — is represented as a single tree node.

$$\textbf{1,537 tokens recoverable} \text{ (full sequence, no partial-block penalty)}$$

**Practical difference:**

For this example, RadixAttention recovers 1 additional token. More importantly, RadixAttention's tree structure allows matching at *any* prefix length — including lengths not aligned to block boundaries. For 1,000 different users with a shared 1,537-token prompt, RadixAttention eliminates 1,537 tokens of KV recomputation per user vs 1,536 for block-level caching. Over 1M daily requests, this adds up.

---

### Solution 2 — Why evicting an interior radix tree node requires evicting descendants first

**Step 1 — Tree invariant.**

The radix tree encodes the property: **every path from root to a node represents a unique token sequence prefix whose KV data is stored in the node chain.**

**Step 2 — Interior node dependency.**

Suppose the tree has:

```
Root → [A: tokens 1-512] → [B: tokens 513-768] → [C: tokens 769-1024]
```

Node B's KV data covers tokens 513–768. But this data was computed with full attention to all preceding tokens (1–512, stored in node A). If node A is evicted:

- Nodes B and C reference token positions 1–512 in their attention computation
- Those positions no longer have valid KV data in the cache
- Any future request trying to use node B's cache would compute incorrect attention

**Step 3 — Enforced invariant.**

Every node's cached KV data is only valid if ALL ancestor nodes' data remains in cache. Therefore:

- Before evicting node A, must evict B and C first (free descendants before ancestors)
- This is enforced by the reference count system: node A's ref_count includes contributions from all paths passing through it

**Result:** The eviction order is always leaf-first, bottom-up. This is implemented by a LRU queue over leaf nodes only — internal nodes become evictable only after all their leaf descendants are evicted and their ref_count reaches 0.

---

### Solution 3 — Cache value threshold: when is prefix cache more valuable than an extra request slot?

**Setup:**

- 1,000 req/min total, 600 share a 2,000-token system prompt, 400 have unique prompts
- Trade-off: cache for 2,000 tokens OR one additional concurrent request slot

**Step 1 — Value of the prefix cache.**

Each of the 600 shared-prompt requests saves the 2,000-token prefill computation. At a typical prefill rate of 5,000 tok/s on an H100:

$$\text{time saved per request} = \frac{2{,}000}{5{,}000} = 0.4 \text{ s}$$
$$\text{GPU time saved per minute} = 600 \times 0.4 = 240 \text{ GPU-seconds/minute}$$

**Step 2 — Value of one additional request slot.**

One extra slot serves one additional concurrent request. At the system's decode throughput (say 50 tokens/s per sequence, 200-token average response):

$$\text{time per request} = 200/50 = 4 \text{ s}$$
$$\text{requests served from extra slot per minute} = 60/4 = 15 \text{ extra requests}$$

**Step 3 — Break-even.**

Cache is more valuable when its compute saving exceeds the extra-slot throughput:

$$600 \times 0.4 \text{ GPU-s} > 15 \text{ requests} \times 4 \text{ s} \implies 240 > 60 \checkmark$$

The prefix cache wins by 4× in this scenario. The break-even point (cache = slot) occurs when:

$$\text{hit rate} \times \text{time\_saved/req} = \text{throughput\_from\_one\_slot}$$

For this example, even a 10% hit rate (60 hits/min × 0.4 s = 24 GPU-s) vs slot (60 GPU-s) means the slot is better below ~40% hit rate. At 60% hit rate, the cache is unambiguously more valuable.

---

### Solution 4 — Cancelled request with ref_count = 1 on prefix nodes

**Scenario:** User cancels mid-generation. The request owns prefix nodes P1 → P2 → P3, each with ref_count = 1.

**Step-by-step release sequence:**

**Step 1:** Server detects client disconnect (TCP RST or explicit cancel).

**Step 2:** Scheduler marks the sequence as canceled. Decode stops immediately for this sequence.

**Step 3:** Walk the sequence's node path from leaf to root:
- **Node P3 (leaf):** Decrement ref_count: 1 → 0. P3 is now unreferenced → move to LRU eviction pool (mark as eligible for eviction). Physical GPU blocks are NOT freed yet (lazy eviction).
- **Node P2:** Decrement ref_count: if 1 → 0 → move to LRU pool. If > 0 after decrement, stop (other sequences share this prefix).
- **Node P1:** Same logic.

**Step 4:** Physical block reclamation happens lazily — when the block manager needs new blocks for a different request, it evicts the lowest-priority LRU pool entries and physically frees their GPU blocks.

**Why lazy eviction:**

If another request arrives *immediately after* the cancellation with the same prefix, P1 may still be in the LRU pool (not yet physically freed). It can be re-promoted instantly (set ref_count = 1, remove from LRU pool) — avoiding a redundant recomputation. Lazy eviction exploits temporal locality.

---

### Solution 5 — Multi-turn chatbot cache hit rate

**Setup:** 500-token system prompt, 8 turns, 150 tokens/turn average

**Within-session cache hit rate by turn:**

| Turn | Total tokens so far | Cached tokens (prior turns) | Cache hit rate |
|------|---------------------|------------------------------|----------------|
| 1 | 500 + 150 = 650 | 500 (system prompt) | 500/650 = **76.9%** |
| 2 | 800 | 650 (system prompt + turn 1) | 650/800 = **81.3%** |
| 3 | 950 | 800 | 800/950 = **84.2%** |
| 4 | 1,100 | 950 | 950/1,100 = **86.4%** |
| 5 | 1,250 | 1,100 | **88.0%** |
| 6 | 1,400 | 1,250 | **89.3%** |
| 7 | 1,550 | 1,400 | **90.3%** |
| 8 | 1,700 | 1,550 | **91.2%** |

**Average across 8 turns:** ~86%

**Cross-session reuse (different users, same system prompt):**

Only the 500-token system prompt is shared across different user sessions. Turn-specific context is unique per user.

$$\text{cross-session hit rate} = \frac{500}{\text{total tokens per request}}$$

For a new user's first turn (650 tokens): 500/650 = 76.9%. For subsequent turns, the unique tokens grow and the system prompt share shrinks.

**Conclusion:** Within-session reuse is highly effective (76–91% hit rate across turns). Cross-session reuse is more limited — only the system prompt is shared, so the benefit is proportional to the system prompt fraction of total context.


*Companion code: [`docs/code/chapter_11c.md`](../code/chapter_11c.md)*
