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

Prefix caching solves this by recognising that the KV pairs for a given token
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

## 11.6.11 vLLM V1 Hash-Based Deduplication

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
   a request is cancelled mid-generation while its prefix nodes have
   `ref_count == 1`? Describe the sequence of operations required to safely
   release those nodes back to the eviction pool.

5. A multi-turn chatbot has a 500-token system prompt and users average 8
   turns per session, each turn adding 150 tokens. What is the theoretical
   maximum cache hit rate (by tokens) if all turns within a session reuse
   prior turns' KV cache? How does this compare to cross-session reuse if
   1,000 concurrent users share the same system prompt?
