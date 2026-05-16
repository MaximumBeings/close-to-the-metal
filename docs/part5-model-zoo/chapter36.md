# Chapter 36: Kimi — Long-Context Specialization and Moon-Cache

> *"Serving a 1M-token context window is not 1,000× harder than serving 1,000 tokens — it's qualitatively different. You need a different architecture for the KV cache, not just more memory."*

---

**What you will understand after this chapter:**

- Why standard KV cache management breaks down beyond 32K tokens
- How Moon-Cache implements hierarchical HBM → DRAM → NVMe KV storage
- The economics of long-context serving: cost per million tokens at 128K vs. 1M
- How to configure vLLM for 128K+ contexts in production

**What you need first:**

- Chapter 6 (PagedAttention), Chapter 11 (Chunked Prefill), Chapter 18 (Disaggregated Serving)

---

## 36.1 The Long-Context Problem

A standard vLLM deployment runs fine at context lengths up to 32K tokens. Beyond that, three problems emerge simultaneously:

```
  Problems at Long Context
  
  Context:    1K    4K    16K    32K    64K    128K   256K   1M
              │     │     │      │      │       │      │      │
  KV size:   small  ok   ok    tight  hard   very    OOM    OOM
  (70B BF16)        │     │      │      │      hard    ↑      ↑
                    │     │      │      │       │      
                    │     │      │      │       └── 70B × 128K: 40 GB just for KV
                    │     │      │      │
                    │     │      │      └──── Prefill time: O(n²) in standard attention
                    │     │      │
                    │     │      └────────── Fragmentation: long requests waste blocks
                    │     │
                    │     └────────────────  Still manageable with PagedAttention
```

For a 70B BF16 model:

- At 128K context: KV = 40.9 GB per sequence (just KV, before weights)
- At 1M context:   KV = 320 GB per sequence — exceeds most GPU clusters

Kimi (Moonshot AI) was designed from the ground up to handle these contexts economically.

---

## 36.2 Moon-Cache Architecture

Moon-Cache is Kimi's production KV caching infrastructure, described in their technical report (2024). It implements **hierarchical KV block management** across three storage tiers:

```
  Moon-Cache Tier Hierarchy
  
  ┌─────────────────────────────────────────────────────────────────┐
  │  Tier 1: HBM (GPU Memory)                                      │
  │  Capacity: 80 GB (H100)  |  Bandwidth: 3.35 TB/s              │
  │  Contents: Active sequence KV blocks (currently computing)      │
  │  Block eviction: LRU when new sequence needs space             │
  ├─────────────────────────────────────────────────────────────────┤
  │  Tier 2: DRAM (CPU Memory)                                      │
  │  Capacity: 256–512 GB per node  |  Bandwidth: 100 GB/s (PCIe) │
  │  Contents: Recently used blocks (hot cache, evicted from GPU)  │
  │  Block eviction: LRU when DRAM fills, spill to Tier 3          │
  ├─────────────────────────────────────────────────────────────────┤
  │  Tier 3: NVMe SSD                                              │
  │  Capacity: 4–16 TB per node  |  Bandwidth: 7 GB/s             │
  │  Contents: Cold blocks (sessions from hours ago)               │
  │  Eviction: TTL-based (blocks expire after configurable window) │
  └─────────────────────────────────────────────────────────────────┘
  
  KV Block Lifecycle:
  
  Prefill → [HBM] → (not used for 30s) → [DRAM] → (not used for 1hr) → [NVMe]
                                                                              ↑
                                        When request resumes: [NVMe] → [DRAM] → [HBM]
```

### 36.2.1 Block Granularity

Moon-Cache uses 64-token KV blocks (matching vLLM's default block size). For a 70B model:

```
Block size:  64 tokens × 327,680 bytes/token = 20 MB per block (BF16)
             64 tokens × 163,840 bytes/token = 10 MB per block (INT8)
```

A 128K context = 2,000 blocks × 20 MB = 40 GB of 20 MB blocks.

### 36.2.2 Latency Penalties for Tier Misses

```
WORKED EXAMPLE 36.1 — KV Block Retrieval Latency
─────────────────────────────────────────────────────────────────────
One 64-token KV block (10 MB INT8):

HBM hit:   Already in GPU → 0 ms (always ready)
DRAM hit:  PCIe 4.0 (32 GB/s): 10 MB / 32 GB/s = 0.31 ms
           PCIe 5.0 (64 GB/s): 10 MB / 64 GB/s = 0.16 ms
NVMe hit:  PCIe 4.0 NVMe (7 GB/s): 10 MB / 7 GB/s = 1.43 ms
           (plus NVMe latency overhead: ~100 μs/block access)

For a 128K resume (2,000 blocks all on NVMe):
  Cold retrieval: 2,000 × 1.43 ms = 2,860 ms = 2.86 seconds
  (plus read amplification from NVMe block alignment)
  Practical: 3–5 seconds to resume a cold 128K session
─────────────────────────────────────────────────────────────────────
```

This is why tier placement policy matters: Kimi keeps "hot" sessions (active in last 30 minutes) in DRAM rather than NVMe, reducing resume latency from seconds to ~300ms.

---

## 36.3 Chunked Prefill at 128K

Processing a 128K token prompt naively would require computing attention over 128K×128K = 16B score elements. Flash Attention makes this O(n) in memory but it is still extremely slow:

```
  Chunked Prefill for 128K prompt (model: 70B, H100):
  
  Chunk size: 2,048 tokens
  Chunks: 128,000 / 2,048 = 62.5 → 63 chunks
  
  Per chunk:
    Prefill FLOPS:  2 × 70B × 2,048 = 286 × 10⁹ FLOPs
    Time at 1 PFLOPS: 286 ms
  
  Total prefill time: 63 × 286 ms ≈ 18 seconds
  
  Without chunked prefill: entire 128K in one CUDA call
    Memory: 128K² × 2 bytes = 33 GB for attention scores alone → OOM
    Even with Flash Attention (O(n) memory): very long kernel, no preemption
```

Chunked prefill allows the scheduler to interleave decode steps (for other requests) between prefill chunks, improving latency fairness.

---

## 36.4 Context Window Economics

Long context is expensive. Here is the full cost model:

```
WORKED EXAMPLE 36.2 — Cost per Million Tokens at Different Contexts
─────────────────────────────────────────────────────────────────────
Hardware: 2× H100 (cost: $56/hr combined)
Model: 70B BF16 (fits in 2× H100 with limited KV budget)

Scenario A: 8K context, batch=8
  KV per seq:   8K × 320 KB = 2.5 GB
  Total KV:     8 × 2.5 GB = 20 GB
  Decode TPS:   (3.35 TB/s × 0.85) / (70B × 2B) = 20 tok/s × batch=8 = 160 tok/s
  Output tok/hr: 160 × 3600 = 576,000
  Cost/1M tokens: $56 / 0.576 = $97/1M output tokens

Scenario B: 128K context, batch=1 (memory limited)
  KV per seq:   128K × 320 KB = 40 GB
  Total KV:     1 × 40 GB = 40 GB
  Decode TPS:   (3.35 TB/s × 0.85) / (70B × 2B) = 20 tok/s
  Output tok/hr: 20 × 3600 = 72,000
  Cost/1M tokens: $56 / 0.072 = $778/1M output tokens

Long context premium: $778 / $97 = 8× more expensive per output token
─────────────────────────────────────────────────────────────────────
```

This explains why API providers charge significantly more for long-context requests.

---

## 36.5 Configuring vLLM for Long Context

vLLM supports 128K+ contexts directly. Key parameters:

```bash
# 128K context serving
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \    # chunk size
    --kv-cache-dtype fp8 \             # halve KV memory
    --gpu-memory-utilization 0.92

# For Kimi-compatible models (ultra-long context)
vllm serve moonshot-ai/kimi-model \
    --max-model-len 1048576 \          # 1M tokens
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --swap-space 64 \                  # 64 GB CPU DRAM swap (Tier 2 analog)
    --kv-cache-dtype fp8
```

### 36.5.1 KV Cache Offloading to CPU

vLLM's `--swap-space` implements a simplified version of Moon-Cache Tier 2 — swapping KV blocks to CPU DRAM when GPU memory is full:

```
  vLLM KV Swap Flow
  
  GPU VRAM fills → Scheduler selects lowest-priority sequence for swap
  GPU → CPU: swap KV blocks for that sequence
  CPU DRAM holds blocks until the sequence is scheduled again
  CPU → GPU: restore blocks when sequence resumes
  
  Latency: PCIe bandwidth limited (see Worked Example 36.1)
  Practical: useful for "paused" long conversations, not high-throughput
```

---

## 36.6 Kimi's Production Models

### 36.6.1 Kimi 1.5 and Kimi k1.5

Moonshot AI's public model portfolio as of early 2025:

```
  Model              Context    Modality     Key feature
  ──────────────────────────────────────────────────────────────────
  Kimi (chat)        128K       Text+Image   Production chat API
  Kimi 1.5           128K       Text+Image   Improved reasoning
  Kimi k1.5          128K       Text+Image   Long-CoT RL (like R1)
  Kimi VL            128K       Vision       Document understanding
  Kimi Long Context  1M         Text         Research preview
```

Kimi k1.5 (January 2025) is noteworthy: it achieves o1-level performance on MATH
and code benchmarks using a **long-context RL** training strategy.
Rather than using a small draft model for reasoning (like DeepSeek-R1's RL approach),
Kimi k1.5 trains the base model to generate long thinking traces by rewarding
correct answers — the same GRPO approach covered in Chapter 24, but applied at
context lengths up to 128K tokens per reasoning trace.

**Serving implication:** Kimi k1.5 generates 5,000–32,000 token reasoning traces
before emitting the final answer. At 32K output tokens with a 4K prompt:
```
  KV at peak: 36,000 tokens × 320 KB = 11.5 GB per sequence (70B BF16)
  This is manageable on a single H100 but eliminates batching — each
  long-CoT sequence fully occupies the KV budget during reasoning.
```

### 36.6.2 Sparse Attention for Ultra-Long Context

Beyond 256K tokens, even chunked prefill + FP8 KV can be insufficient.
Production long-context systems employ **sparse attention patterns** to reduce the
O(n²) attention computation:

```
  Sparse Attention Variants
  
  ┌──────────────────────────────────────────────────────────────────┐
  │  Sliding Window Attention (Mistral, Phi-3):                      │
  │    Each token attends to only W preceding tokens (W = 4096)     │
  │    Memory: O(n × W) instead of O(n²)                            │
  │    Loss: no token can see beyond W tokens back (approx only)    │
  │                                                                  │
  │  Dilated / Strided Attention (LongNet):                         │
  │    Attention patterns at increasing strides: {1, 2, 4, 8, ...}  │
  │    Each head sees different temporal scales                      │
  │    Memory: O(n log n)                                            │
  │                                                                  │
  │  Hybrid Local/Global (Gemma 2, Qwen2):                          │
  │    Alternate layers: local window (W=4096) + full attention      │
  │    Global layers see everything; local layers cheaply process    │
  │                                                                  │
  │  Ring Attention (distributed):                                   │
  │    Split the sequence across GPUs in a ring topology             │
  │    Each GPU holds a segment; KV blocks are passed around ring    │
  │    Used for 1M+ token contexts across 32+ GPUs                  │
  └──────────────────────────────────────────────────────────────────┘
```

**vLLM and sliding window:** Models with sliding window attention (e.g., Mistral 7B)
work automatically in vLLM — the sliding window is baked into the model architecture
and handled by Flash Attention's sliding window kernel. No special flag needed.

### 36.6.3 Long-Context vs. RAG — The Engineer's Decision

The availability of 128K+ models does not eliminate RAG. The choice is architectural:

```
  Long-Context Model                  RAG Pipeline
  ─────────────────────────           ─────────────────────────
  All documents in prompt             Documents in vector store
  Simple deployment                   Retrieval step adds latency
  No retrieval errors                 Retrieval can miss relevant chunks
  KV cost: 8× premium at 128K         KV cost: standard (8K prompt)
  Quality: perfect recall             Quality: recall@k limited
  
  Break-even: long-context wins when retrieval accuracy < ~80% AND
  the relevant information is scattered across the document corpus.
  RAG wins for large corpora (> 10M tokens) that don't fit in any context.
```

**Rule of thumb for production:**

- < 100 pages of documents: long-context model (simpler, no retrieval errors)
- 100–1,000 pages: hybrid — retrieve top-20 chunks, stuff into 32K context
- > 1,000 pages: RAG required (no context window handles this economically)

---

### 36.6.4 Multi-Session Memory Budget Arithmetic

The single hardest problem in long-context production serving is not serving one 128K session — it is serving N concurrent 128K sessions while maintaining acceptable TTFT for new arrivals.

```
WORKED EXAMPLE 36.3 — Concurrent Long-Context Sessions
─────────────────────────────────────────────────────────────────────
Hardware: 4× H100 SXM (TP=4), total HBM = 320 GB
Model: Llama-3-70B BF16
Weight footprint: 70B × 2 bytes = 140 GB (split across 4 GPUs = 35 GB/GPU)
HBM available for KV: 320 − 140 = 180 GB total

KV per token (TP=4 shard): 80 layers × 2 heads/GPU × 128 dim × 2 bytes
  = 80 × 256 × 2 = 40,960 bytes/token = 40 KB/token per GPU shard
  Total KV/token across 4 GPUs = 160 KB/token

KV per session at various lengths:
  8K   context →   8K × 160 KB = 1.28 GB
  32K  context →  32K × 160 KB = 5.12 GB
  64K  context →  64K × 160 KB = 10.2 GB
  128K context → 128K × 160 KB = 20.5 GB

Concurrent sessions in 180 GB HBM:
  8K   context: 180 / 1.28  = 140 concurrent sessions
  32K  context: 180 / 5.12  = 35  concurrent sessions
  64K  context: 180 / 10.2  = 17  concurrent sessions
  128K context: 180 / 20.5  =  8  concurrent sessions (zero headroom for new arrivals)

Safe operating point at 128K: keep 6 active HBM sessions, swap 2 to DRAM
─────────────────────────────────────────────────────────────────────
```

This arithmetic drives the scheduling policy: a system that naively allows 8 concurrent 128K sessions will be unable to admit new prefill requests until one session completes, because there is no HBM left for the prefill KV. Production systems impose a **maximum concurrent long-context sessions** cap, typically at 60–70% of the theoretical HBM limit, reserving the headroom for new arrivals.

**vLLM admission control for long-context:**

```python
# vLLM does not have a native "max long-context sessions" cap,
# but you can approximate it at the gateway level:

import asyncio
from collections import defaultdict

class LongContextAdmissionController:
    """
    Limit concurrent sessions above a context threshold.
    Sessions below threshold are never blocked.
    """
    def __init__(self, max_long_sessions: int = 6, threshold_tokens: int = 32768):
        self.max_long_sessions = max_long_sessions
        self.threshold = threshold_tokens
        self._active = 0
        self._lock = asyncio.Lock()

    async def acquire(self, prompt_tokens: int) -> bool:
        if prompt_tokens < self.threshold:
            return True  # short sessions always admitted
        async with self._lock:
            if self._active >= self.max_long_sessions:
                return False  # queue or return 429
            self._active += 1
            return True

    async def release(self, prompt_tokens: int):
        if prompt_tokens >= self.threshold:
            async with self._lock:
                self._active = max(0, self._active - 1)
```

This gateway-level gate prevents GPU memory exhaustion while allowing short requests to continue flowing even when the long-context pool is saturated.

---

### 36.6.5 Session Lifecycle Management

Long-context applications typically involve **multi-turn conversations** — a user repeatedly extends the same context over minutes or hours. This creates three lifecycle phases with different KV management needs:

```
  Long-Context Session Lifecycle
  
  Phase 1: Active (< 5 minutes since last token)
  ┌──────────────────────────────────────────────────────────────────┐
  │  KV location: HBM (GPU memory)                                  │
  │  Scheduling priority: high                                      │
  │  Action: serve decode immediately on next user message          │
  └──────────────────────────────────────────────────────────────────┘
  
  Phase 2: Warm (5 min – 2 hours since last token)
  ┌──────────────────────────────────────────────────────────────────┐
  │  KV location: DRAM (CPU memory, via --swap-space)               │
  │  Resume latency: ~300 ms (DRAM → GPU transfer, 128K session)    │
  │  Action: swap blocks back to GPU on next user message           │
  └──────────────────────────────────────────────────────────────────┘
  
  Phase 3: Cold (> 2 hours since last token)
  ┌──────────────────────────────────────────────────────────────────┐
  │  KV location: NVMe (or re-prefill from stored prompt)           │
  │  Resume latency: 3–10 seconds                                   │
  │  Decision: swap from NVMe OR re-prefill from saved message log  │
  │  Note: re-prefill often faster than NVMe retrieval for < 64K    │
  └──────────────────────────────────────────────────────────────────┘
```

**Re-prefill vs. NVMe restore for cold sessions:**

For sessions that have been idle long enough to fall to NVMe (Phase 3), there is a real engineering decision:

```
WORKED EXAMPLE 36.4 — Re-prefill vs. Cold NVMe Restore
─────────────────────────────────────────────────────────────────────
Session: 32K token conversation history, cold (on NVMe)

Option A: NVMe restore
  Blocks to restore: 32K / 64 = 500 blocks × 20 MB = 10 GB
  NVMe read bandwidth: 7 GB/s
  Transfer time: 10 GB / 7 GB/s ≈ 1.43 seconds
  Plus DRAM → HBM transfer: 10 GB / 32 GB/s ≈ 0.31 seconds
  Total: ≈ 1.74 seconds to restore, then decode begins

Option B: Re-prefill (chunked, 2048 tokens/chunk)
  Chunks: 32K / 2048 = 16 chunks
  Time per chunk: 286 ms (70B @ 1 PFLOPS)
  Total prefill: 16 × 286 ms = 4.58 seconds
  (but: chunks can be interleaved with other requests' decode steps)

Break-even: NVMe restore wins for sessions > ~18K tokens cold
            Re-prefill wins for sessions < ~18K tokens cold
            (because re-prefill latency grows with context; NVMe latency is near-constant)
─────────────────────────────────────────────────────────────────────
```

In practice, production systems use re-prefill for cold sessions up to 16K tokens (the restore overhead is manageable), and NVMe-backed hierarchical restore for longer sessions.

**vLLM session management via the OpenAI API:**

vLLM's server maintains KV blocks by `request_id`. You can implement session persistence at the application layer using a combination of prefix caching and explicit conversation history:

```python
import openai
import json
from pathlib import Path

class PersistentLongContextSession:
    """
    Application-layer session management for vLLM.
    Stores message history; lets vLLM's prefix cache handle KV reuse.
    """
    def __init__(self, session_id: str, client: openai.OpenAI, model: str):
        self.session_id = session_id
        self.client = client
        self.model = model
        self.messages: list[dict] = []
        self._history_file = Path(f"/tmp/session_{session_id}.json")
        self._load()

    def _load(self):
        if self._history_file.exists():
            self.messages = json.loads(self._history_file.read_text())

    def _save(self):
        self._history_file.write_text(json.dumps(self.messages))

    def chat(self, user_message: str, max_tokens: int = 1024) -> str:
        self.messages.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,  # full history → vLLM prefix cache reuses KV
            max_tokens=max_tokens,
            temperature=0.7,
        )
        assistant_reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_reply})
        self._save()
        return assistant_reply

    @property
    def context_tokens(self) -> int:
        """Rough estimate: 4 chars per token on average."""
        return sum(len(m["content"]) for m in self.messages) // 4

# Usage:
# session = PersistentLongContextSession("user_123", client, "Qwen/Qwen2.5-72B")
# reply = session.chat("Continue from where we left off...")
# print(f"Context so far: ~{session.context_tokens:,} tokens")
```

**Key insight:** vLLM's prefix caching means the KV blocks for prior turns are cached on disk (if `--enable-prefix-caching` is set), and subsequent turns only pay for the new tokens' prefill. This is the practical mechanism that makes multi-turn long-context serving tractable — you don't re-pay the 128K prefill cost on every user turn.

---

## 36.7 Chapter Summary

Long-context serving (128K+) requires three things standard inference systems don't have: hierarchical KV storage (HBM → DRAM → NVMe), chunked prefill to avoid OOM during the prefill phase, and acceptance of a significantly higher cost per output token (typically 5–10× premium over 8K context).

Moon-Cache's key insight is that KV blocks have temporal locality — recently active sessions should be in fast storage, old sessions can wait for NVMe retrieval. vLLM's `--swap-space` is a first step toward this; production systems need the full hierarchy.

Kimi k1.5's long-CoT training demonstrates that context length is not just a serving concern — it is also a training constraint that directly shapes the model's capability. The models that reason longest (and therefore score highest on benchmarks) are the ones that create the most challenging serving requirements.

### Where We Go Next

Chapter 37 covers Nemotron — NVIDIA's model family optimized for TensorRT-LLM, and explains the TRT-LLM compilation pipeline that achieves maximum throughput on NVIDIA hardware.


---

## Chapter Summary

- **Kimi's focus**: ultra-long context (up to 1M tokens) for document analysis, code understanding, and multi-document retrieval.
- **Moon-Cache**: hierarchical KV storage across GPU DRAM → CPU DRAM → SSD, enabling 1M-token contexts that do not fit in GPU memory alone.
- **Block-level tiering**: KV blocks are promoted to GPU on access and demoted to CPU or SSD by an LRU policy; the promotion path is optimized for sequential access patterns in long documents.
- **Prefetch heuristics**: Kimi's serving system predicts which KV blocks will be needed in the next decode step and prefetches them from CPU/SSD to GPU ahead of time.
- **Sparse attention**: Kimi uses a combination of local window attention and landmark token attention to reduce the O(T²) attention complexity for very long sequences.
- **Embedding-as-memory**: for contexts exceeding 1M tokens, a separate retrieval component selects relevant passages; the LLM then attends only to retrieved passages, not the full context.
- **vLLM analogy**: Moon-Cache is conceptually an extension of vLLM's swap-to-CPU mechanism, but with a full three-tier hierarchy and proactive prefetching rather than reactive eviction.

---

## Self-Check Questions

1. Moon-Cache has GPU capacity 4 GB, CPU capacity 256 GB, SSD capacity 8 TB for KV blocks. A 1M-token sequence at LLaMA-3 8B (32 layers, 8 GQA KV heads, d_k = 128, BF16) needs how many GB? Verify that CPU+SSD is sufficient. *(Section 36.2)*

2. A decode step accesses KV blocks in positions [950K–1M] (GPU-resident) and [100K–150K] (SSD-resident). Prefetch from SSD takes 500 ms. The decode step itself takes 50 ms. Does Moon-Cache deliver the block in time? What latency strategy resolves this? *(Section 36.3)*

3. Sparse attention with window W = 4096 and landmark token every L = 1024 positions is applied to a 1M-token sequence. Compute the number of attention operations per token vs full attention. *(Section 36.4)*

4. At what context length does Moon-Cache's CPU tier become the bottleneck (when GPU is fully saturated)? Given CPU DRAM bandwidth of 200 GB/s, estimate the maximum context length that allows 10 ms decode steps. *(Section 36.2)*

5. Compare the Kimi Moon-Cache approach to RAG (Retrieval-Augmented Generation) for a 1M-token document Q&A task. Under what conditions is each approach preferable? *(Section 36.5)*
