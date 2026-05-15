# Chapter 30: Semantic Caching and Response Reuse

Every inference call costs money and time.
For a production system serving thousands of users, a significant fraction of those calls
are functionally redundant — different users asking the same question in slightly different
words, a CI pipeline re-running the same code review prompt, a customer service bot fielding
the same FAQ for the hundredth time that hour.

Semantic caching intercepts these redundant calls before they reach the inference engine.
Instead of asking "have I seen this exact prompt before?" (exact-match caching), it asks
"have I seen a prompt that means the same thing?" — and returns the stored response if
similarity exceeds a threshold.

Done well, semantic caching reduces inference costs by 20–60% on FAQ-heavy workloads,
cuts p95 latency by 10× for cache hits, and requires no changes to the language model.
Done badly, it silently returns stale or wrong answers, destroys user trust, and is harder
to debug than any model quality issue.

This chapter covers the full engineering: embedding models, similarity search, cache
storage, invalidation, quality control, cost modeling, and the relationship between
semantic caching and vLLM's own prefix-caching mechanism.

---

## 30.1  The Cache Hierarchy

Modern LLM deployments have three distinct caching layers that operate at different
granularities:

```
Request
    │
    ▼
┌─────────────────────────────────────────────┐
│  Layer 1: Semantic Cache (application)      │  ← This chapter
│  "Is there a stored response for this       │
│   semantically equivalent query?"           │
│  Miss → pass to inference engine            │
└─────────────────┬───────────────────────────┘
                  │ miss
                  ▼
┌─────────────────────────────────────────────┐
│  Layer 2: Prefix Cache (vLLM / llama.cpp)   │  ← Chapter 11
│  "Do any KV blocks for this prompt's        │
│   prefix already exist in GPU memory?"      │
│  Miss → full prefill                        │
└─────────────────┬───────────────────────────┘
                  │ miss
                  ▼
┌─────────────────────────────────────────────┐
│  Layer 3: Full Inference                    │
│  Prefill + decode, full GPU compute         │
└─────────────────────────────────────────────┘
```

Layer 1 (semantic cache) saves the entire inference cost — GPU, memory, time, money.
Layer 2 (prefix cache) saves only the prefill cost; decode still runs.
Neither layer serves as a substitute for the other.

---

## 30.2  Exact-Match vs. Semantic Caching

### 30.2.1  Exact-Match Caching

The simplest form: hash the full prompt string, look up the hash in a key-value store,
return the stored response on hit.

```python
import hashlib, json

def exact_cache_key(prompt: str, model: str, temperature: float) -> str:
    payload = json.dumps({"prompt": prompt, "model": model, "temp": temperature},
                          sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

Hit rate is very low for open-ended prompts (any whitespace difference is a miss) but
very high for templated prompts where the variable parts are fixed.

Appropriate when:
- The same machine generates identical prompts (CI pipelines, batch jobs)
- Prompt templates have low cardinality (FAQ systems with a finite question set)
- Zero tolerance for semantic approximation (financial/legal contexts)

### 30.2.2  Semantic Caching

Semantic caching embeds each incoming prompt as a dense vector and searches for
the nearest stored prompt using approximate nearest-neighbor (ANN) search.
If the nearest neighbor is within a similarity threshold `τ`, the stored response is returned.

```
New prompt → Embed → Query vector store → 
    if similarity(nearest_neighbor) ≥ τ:
        return stored_response                  # cache HIT
    else:
        run inference, store (embed, response)  # cache MISS
```

The embedding model determines the semantic space.
Two prompts that map to nearby vectors share the same cached response.

### 30.2.3  The Similarity Threshold Trade-Off

`τ` is the most important hyperparameter in any semantic cache:

| `τ` | behavior |
|---|---|
| 1.00 | Exact match only (equivalent to hash cache) |
| 0.98 | Near-identical phrasing; very safe |
| 0.95 | Same question, minor rewording; typical default |
| 0.90 | Same topic, different specificity; risky |
| 0.85 | Thematically similar; usually wrong answer |
| < 0.85 | Returns unrelated responses; broken |

The sweet spot for most production deployments is `τ ∈ [0.93, 0.97]`.
Always measure **answer accuracy on cache hits** (not just hit rate) when tuning `τ`.

---

## 30.3  Embedding Models for Caching

The embedding model converts a text prompt into a fixed-dimension vector.
For semantic caching, the requirements differ from RAG retrieval:

**Speed is critical.** Every request hits the embedding model on the way in.
A 200ms embedding adds more latency than a 95% cache hit saves.
Use a small, fast model.

**Query-query similarity matters, not query-document similarity.**
RAG embeddings (e.g., `text-embedding-3-large`) are optimized for asymmetric search
(short query → long document).
Caching needs symmetric similarity (query → query).

**Sentence-level precision over paragraph-level recall.**
The embedding must distinguish "How do I reset my password?" from
"How do I reset my two-factor authentication?" (different answers despite surface similarity).

Recommended models for semantic caching:

| Model | Dim | Latency (CPU) | Latency (GPU) | Notes |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 2ms | 0.3ms | Best speed/quality trade-off |
| `all-MiniLM-L12-v2` | 384 | 4ms | 0.5ms | Slightly better quality |
| `all-mpnet-base-v2` | 768 | 10ms | 1ms | Higher quality, higher cost |
| `bge-small-en-v1.5` | 384 | 2ms | 0.3ms | Strong at query-query similarity |
| `text-embedding-3-small` | 1536 | 20ms | — | OpenAI API; high latency for caching |

For most deployments, `all-MiniLM-L6-v2` or `bge-small-en-v1.5` on CPU with ONNX
Runtime delivers sub-3ms embeddings — fast enough to be invisible in the request path.

---

## 30.4  Vector Stores

### 30.4.1  In-Memory: FAISS

FAISS (Facebook AI Similarity Search) is the standard library for high-performance
ANN search in Python.

```python
import faiss, numpy as np

# Flat L2 index (exact search, no approximation)
d = 384  # embedding dimension
index = faiss.IndexFlatL2(d)

# Add vectors
embeddings = np.random.randn(10000, d).astype(np.float32)
index.add(embeddings)

# Search: k nearest neighbors
query = np.random.randn(1, d).astype(np.float32)
D, I = index.search(query, k=5)
# D: squared L2 distances, I: indices
```

For cosine similarity (preferred for NLP embeddings), normalize vectors before adding
and use `IndexFlatIP` (inner product):

```python
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(d)
index.add(embeddings)

faiss.normalize_L2(query)
scores, indices = index.search(query, k=1)
# scores[0][0] is cosine similarity ∈ [-1, 1]
```

For large caches (>1M entries), use approximate indices:
```python
# IVF (Inverted File) + PQ (Product Quantization): fast, compressed
nlist = 256   # number of Voronoi cells
m     = 16    # number of sub-quantizers
nbits = 8     # bits per sub-quantizer
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index.train(embeddings)
index.add(embeddings)
index.nprobe = 32  # cells to search; higher = better recall, slower
```

### 30.4.2  Persistent: Redis with RediSearch

For shared caches across multiple inference server instances:

```python
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np

r = redis.Redis(host="localhost", port=6379)

# Create index
schema = (
    TextField("$.prompt",   as_name="prompt"),
    TextField("$.response", as_name="response"),
    VectorField("$.embedding",
        "HNSW",
        {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"},
        as_name="embedding"),
)
r.ft("cache_idx").create_index(
    schema,
    definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.JSON)
)

def cache_store(key: str, prompt: str, response: str, embedding: np.ndarray):
    r.json().set(f"cache:{key}", "$", {
        "prompt":    prompt,
        "response":  response,
        "embedding": embedding.astype(np.float32).tobytes(),
        "ts":        int(time.time()),
    })

def cache_search(query_emb: np.ndarray, threshold: float = 0.95, k: int = 1):
    q_bytes = query_emb.astype(np.float32).tobytes()
    q = (
        Query(f"*=>[KNN {k} @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("prompt", "response", "score")
        .dialect(2)
    )
    results = r.ft("cache_idx").search(q, query_params={"vec": q_bytes})
    if results.total == 0:
        return None
    top = results.docs[0]
    # RediSearch COSINE distance is 1 - similarity
    similarity = 1 - float(top.score)
    if similarity >= threshold:
        return top.response
    return None
```

### 30.4.3  Postgres + pgvector

For deployments already using Postgres:

```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table
CREATE TABLE semantic_cache (
    id         BIGSERIAL PRIMARY KEY,
    prompt     TEXT NOT NULL,
    response   TEXT NOT NULL,
    embedding  vector(384) NOT NULL,
    model      TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    hit_count  INTEGER DEFAULT 0
);

-- IVFFlat index for approximate search
CREATE INDEX ON semantic_cache
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Search query
SELECT response, 1 - (embedding <=> $1::vector) AS similarity
FROM semantic_cache
WHERE model = $2
ORDER BY embedding <=> $1::vector
LIMIT 1;
```

---

## 30.5  Full Python Implementation

```python
#!/usr/bin/env python3
# semantic_cache_demo.py
"""
Chapter 30 — Semantic Caching Demo (Python)

Implements a complete semantic cache:
  1. Embedding pipeline (simulated + real via sentence-transformers if available)
  2. FAISS-backed vector store with cosine similarity
  3. TTL-aware cache with LRU eviction
  4. Hit-rate and cost tracking
  5. Cache quality validator
  6. Cost model: cache vs. inference break-even
"""
from __future__ import annotations
import hashlib, math, time, random, json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING SIMULATION (replace with real model in production)
# ─────────────────────────────────────────────────────────────────────────────

def _deterministic_embed(text: str, dim: int = 64) -> List[float]:
    """
    Deterministic pseudo-embedding for demo purposes.
    Uses SHA-256 seeded RNG to produce consistent vectors.
    Similar texts (common prefix) will have similar vectors.
    """
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng  = random.Random(seed)
    vec  = [rng.gauss(0, 1) for _ in range(dim)]
    # Add topic signal: inject shared component for questions about same topic
    topic_seed = int(hashlib.sha256(text[:20].encode()).hexdigest()[:8], 16)
    topic_rng  = random.Random(topic_seed)
    topic_vec  = [topic_rng.gauss(0, 0.5) for _ in range(dim)]
    combined   = [v + t for v, t in zip(vec, topic_vec)]
    # normalize
    norm = math.sqrt(sum(x * x for x in combined))
    return [x / norm for x in combined] if norm > 0 else combined


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE ENTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    key: str
    prompt: str
    response: str
    embedding: List[float]
    model: str
    temperature: float
    created_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)
    hit_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC CACHE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    total_requests: int = 0
    semantic_hits: int = 0
    exact_hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_tokens_saved: int = 0
    total_cost_saved_usd: float = 0.0
    embed_time_ms: float = 0.0
    search_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.semantic_hits + self.exact_hits) / self.total_requests

    @property
    def avg_embed_ms(self) -> float:
        n = self.semantic_hits + self.misses
        return self.embed_time_ms / n if n > 0 else 0.0

    @property
    def avg_search_ms(self) -> float:
        n = self.semantic_hits + self.misses
        return self.search_time_ms / n if n > 0 else 0.0


class SemanticCache:
    """
    In-memory semantic cache with:
    - Exact-match fast path (hash lookup)
    - Semantic fallback (cosine similarity on embeddings)
    - TTL expiry
    - LRU eviction when full
    - Per-model isolation
    - Cost tracking
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_entries: int = 10_000,
        ttl_seconds: float = 3600.0,
        embed_dim: int = 64,
        cost_per_1k_input_tokens: float = 0.0005,   # $0.50/M input tokens
        cost_per_1k_output_tokens: float = 0.0015,  # $1.50/M output tokens
    ):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self.dim = embed_dim
        self.cost_in  = cost_per_1k_input_tokens
        self.cost_out = cost_per_1k_output_tokens

        # Storage: key → CacheEntry (in insertion order for LRU)
        self._entries: Dict[str, CacheEntry] = {}
        # Exact-match index: sha256(prompt+model) → key
        self._exact: Dict[str, str] = {}
        self.stats = CacheStats()

    def _embed(self, text: str) -> List[float]:
        t0 = time.perf_counter()
        vec = _deterministic_embed(text, self.dim)
        self.stats.embed_time_ms += (time.perf_counter() - t0) * 1000
        return vec

    def _exact_key(self, prompt: str, model: str, temperature: float) -> str:
        payload = f"{model}|{temperature:.3f}|{prompt}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.created_at) > self.ttl

    def _evict_lru(self):
        """Remove the least-recently-used entry."""
        if not self._entries:
            return
        lru_key = min(self._entries, key=lambda k: self._entries[k].last_hit_at)
        entry = self._entries.pop(lru_key)
        self._exact.pop(self._exact_key(entry.prompt, entry.model, entry.temperature), None)
        self.stats.evictions += 1

    def get(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.0,
    ) -> Optional[str]:
        self.stats.total_requests += 1

        # Fast path: exact match
        ek = self._exact_key(prompt, model, temperature)
        if ek in self._exact:
            key = self._exact[ek]
            if key in self._entries:
                entry = self._entries[key]
                if not self._is_expired(entry):
                    entry.hit_count += 1
                    entry.last_hit_at = time.time()
                    self.stats.exact_hits += 1
                    self._record_savings(entry)
                    return entry.response
                else:
                    # Expired — remove
                    del self._entries[key]
                    del self._exact[ek]

        # Semantic path
        if not self._entries:
            self.stats.misses += 1
            return None

        query_emb = self._embed(prompt)

        t0 = time.perf_counter()
        best_sim, best_key = -1.0, None
        for key, entry in self._entries.items():
            if entry.model != model:
                continue
            if self._is_expired(entry):
                continue
            # High-temperature responses should not be reused (non-deterministic)
            if entry.temperature > 0.3 or temperature > 0.3:
                continue
            sim = cosine_similarity(query_emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_key = key
        self.stats.search_time_ms += (time.perf_counter() - t0) * 1000

        if best_key and best_sim >= self.threshold:
            entry = self._entries[best_key]
            entry.hit_count += 1
            entry.last_hit_at = time.time()
            self.stats.semantic_hits += 1
            self._record_savings(entry)
            return entry.response

        self.stats.misses += 1
        return None

    def put(
        self,
        prompt: str,
        response: str,
        model: str = "default",
        temperature: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> str:
        if len(self._entries) >= self.max_entries:
            self._evict_lru()

        emb = self._embed(prompt)
        ek  = self._exact_key(prompt, model, temperature)
        key = hashlib.md5(f"{prompt[:64]}{time.time()}".encode()).hexdigest()

        entry = CacheEntry(
            key=key, prompt=prompt, response=response,
            embedding=emb, model=model, temperature=temperature,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )
        self._entries[key] = entry
        self._exact[ek]    = key
        return key

    def _record_savings(self, entry: CacheEntry):
        self.stats.total_tokens_saved += entry.input_tokens + entry.output_tokens
        cost = (entry.input_tokens  / 1000 * self.cost_in +
                entry.output_tokens / 1000 * self.cost_out)
        self.stats.total_cost_saved_usd += cost

    def invalidate(self, model: str | None = None):
        """Remove all entries (or all entries for a model)."""
        if model is None:
            self._entries.clear()
            self._exact.clear()
        else:
            keys_to_remove = [k for k, e in self._entries.items() if e.model == model]
            for k in keys_to_remove:
                entry = self._entries.pop(k)
                self._exact.pop(self._exact_key(entry.prompt, entry.model, entry.temperature), None)

    def report(self):
        s = self.stats
        print(f"  Cache size:        {len(self._entries):,} entries")
        print(f"  Total requests:    {s.total_requests:,}")
        print(f"  Exact hits:        {s.exact_hits:,}")
        print(f"  Semantic hits:     {s.semantic_hits:,}")
        print(f"  Misses:            {s.misses:,}")
        print(f"  Hit rate:          {s.hit_rate:.1%}")
        print(f"  Evictions:         {s.evictions:,}")
        print(f"  Tokens saved:      {s.total_tokens_saved:,}")
        print(f"  Cost saved:       ${s.total_cost_saved_usd:.4f}")
        print(f"  Avg embed time:    {s.avg_embed_ms:.2f} ms")
        print(f"  Avg search time:   {s.avg_search_ms:.3f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# COST MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CostModel:
    """Break-even analysis: when does semantic caching pay off?"""
    cost_per_1k_input:  float = 0.0005   # $0.50 / 1M tokens
    cost_per_1k_output: float = 0.0015   # $1.50 / 1M tokens
    avg_input_tokens:   int   = 256
    avg_output_tokens:  int   = 128
    embed_cost_per_req: float = 0.00001  # ~$0.01 / 1K embed calls (self-hosted)
    cache_infra_monthly: float = 50.0    # Redis / pgvector instance
    requests_per_month:  int  = 1_000_000

    @property
    def inference_cost_per_request(self) -> float:
        return (self.avg_input_tokens  / 1000 * self.cost_per_1k_input +
                self.avg_output_tokens / 1000 * self.cost_per_1k_output)

    def monthly_savings(self, hit_rate: float) -> float:
        """Net monthly savings at a given hit rate."""
        inference_saved  = hit_rate * self.requests_per_month * self.inference_cost_per_request
        embed_overhead   = self.requests_per_month * self.embed_cost_per_req
        return inference_saved - embed_overhead - self.cache_infra_monthly

    def break_even_hit_rate(self) -> float:
        """Minimum hit rate for the cache to be net-positive."""
        overhead_per_req = self.embed_cost_per_req + self.cache_infra_monthly / self.requests_per_month
        if self.inference_cost_per_request <= 0:
            return 1.0
        return overhead_per_req / self.inference_cost_per_request

    def print_analysis(self):
        print("=" * 60)
        print("Cost Model — Semantic Cache Break-Even Analysis")
        print("=" * 60)
        print(f"  Inference cost/request: ${self.inference_cost_per_request:.6f}")
        print(f"  Embed overhead/request: ${self.embed_cost_per_req:.6f}")
        print(f"  Cache infra/month:     ${self.cache_infra_monthly:.2f}")
        print(f"  Break-even hit rate:    {self.break_even_hit_rate():.1%}")
        print()
        print(f"  {'Hit Rate':10}  {'Monthly Savings':18}  {'ROI'}")
        print(f"  {'-'*10}  {'-'*18}  {'-'*12}")
        for hr in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
            savings = self.monthly_savings(hr)
            total_inf = self.requests_per_month * self.inference_cost_per_request
            roi = savings / (self.cache_infra_monthly + self.requests_per_month * self.embed_cost_per_req) if savings > 0 else 0
            flag = " ← break-even" if abs(hr - self.break_even_hit_rate()) < 0.03 else ""
            print(f"  {hr:10.0%}  ${savings:>16,.2f}  {roi:>10.1f}×{flag}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheQualityValidator:
    """
    Validates that cache hits return semantically correct responses.
    In production: sample N% of hits and re-run inference; compare.
    Here: simulates the validation logic.
    """

    def should_validate(self, similarity: float, hit_count: int,
                        entry_age_hours: float) -> bool:
        """
        Heuristic: validate if:
        - Similarity is borderline (0.93–0.97)
        - Entry is old (> 24h)
        - Entry has low hit count (< 3) — not yet battle-tested
        """
        if similarity < 0.97:
            return True
        if entry_age_hours > 24:
            return True
        if hit_count < 3:
            return True
        return False

    def validation_sample_rate(self, similarity: float) -> float:
        """
        Higher sampling for lower similarity entries.
        At similarity=0.99: sample 1% of hits for quality monitoring.
        At similarity=0.93: sample 20%.
        """
        if   similarity >= 0.99: return 0.01
        elif similarity >= 0.97: return 0.05
        elif similarity >= 0.95: return 0.10
        else:                    return 0.20


# ─────────────────────────────────────────────────────────────────────────────
# DEMO WORKLOADS
# ─────────────────────────────────────────────────────────────────────────────

FAQ_PROMPTS = [
    "How do I reset my password?",
    "How can I change my password?",
    "I forgot my password, what do I do?",
    "Steps to reset account password",
    "Password reset instructions",
    "How do I update my email address?",
    "How to change the email on my account?",
    "Update my account email",
    "What are your business hours?",
    "When are you open?",
    "What hours do you operate?",
    "Is customer support available on weekends?",
    "How do I cancel my subscription?",
    "Cancel my account",
    "I want to unsubscribe",
    "Steps to cancel membership",
    "How do I request a refund?",
    "I want my money back",
    "Refund policy",
    "How long does a refund take?",
]

FAQ_RESPONSES = {
    "password": "To reset your password, click 'Forgot Password' on the login page. You will receive an email with a reset link valid for 24 hours.",
    "email":    "To update your email address, go to Account Settings > Profile > Email. Enter your new email and confirm with your current password.",
    "hours":    "Our support team is available Monday–Friday 9am–6pm EST. Weekend support is available via email only.",
    "cancel":   "To cancel your subscription, go to Account Settings > Billing > Cancel Subscription. Your access continues until the end of the billing period.",
    "refund":   "Refunds are processed within 5–7 business days. Contact support@example.com with your order number to initiate a refund.",
}

def classify_faq(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in ["password", "reset", "forgot"]): return "password"
    if any(w in p for w in ["email", "address"]):            return "email"
    if any(w in p for w in ["hours", "open", "weekend"]):    return "hours"
    if any(w in p for w in ["cancel", "unsubscribe"]):       return "cancel"
    if any(w in p for w in ["refund", "money back"]):        return "refund"
    return "unknown"


def run_faq_demo():
    print("=" * 60)
    print("Semantic Cache Demo — FAQ Workload")
    print("=" * 60)

    cache = SemanticCache(similarity_threshold=0.92, max_entries=1000, ttl_seconds=86400)

    # Warm the cache with one example per category
    seed_prompts = [
        ("How do I reset my password?",     FAQ_RESPONSES["password"], 150, 80),
        ("How do I update my email address?",FAQ_RESPONSES["email"],    120, 60),
        ("What are your business hours?",    FAQ_RESPONSES["hours"],    80,  50),
        ("How do I cancel my subscription?", FAQ_RESPONSES["cancel"],   100, 70),
        ("How do I request a refund?",       FAQ_RESPONSES["refund"],   90,  60),
    ]
    print(f"\n  Seeding cache with {len(seed_prompts)} entries...")
    for prompt, response, in_tok, out_tok in seed_prompts:
        cache.put(prompt, response, model="gpt-4o-mini", temperature=0.0,
                  input_tokens=in_tok, output_tokens=out_tok)

    # Run all FAQ prompts through the cache
    print(f"\n  Running {len(FAQ_PROMPTS)} FAQ prompts:\n")
    print(f"  {'Prompt':45}  {'Result':10}  {'Category'}")
    print("  " + "-" * 72)

    for prompt in FAQ_PROMPTS:
        result = cache.get(prompt, model="gpt-4o-mini", temperature=0.0)
        category = classify_faq(prompt)
        status = "HIT" if result else "MISS"
        if result is None:
            # Simulate inference and store
            resp = FAQ_RESPONSES.get(category, "Please contact support.")
            cache.put(prompt, resp, model="gpt-4o-mini", temperature=0.0,
                      input_tokens=100, output_tokens=60)
        print(f"  {prompt[:45]:45}  {status:10}  {category}")

    print()
    cache.report()


def run_cost_analysis():
    model = CostModel(
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        avg_input_tokens=256,
        avg_output_tokens=128,
        embed_cost_per_req=0.000005,   # self-hosted MiniLM on CPU
        cache_infra_monthly=30.0,       # Redis Elasticache small instance
        requests_per_month=2_000_000,
    )
    model.print_analysis()


def run_threshold_sweep():
    """Show how hit rate and answer accuracy vary with threshold."""
    print("=" * 60)
    print("Threshold Sweep — Hit Rate vs. Answer Accuracy")
    print("=" * 60)

    # Simulate pairs of prompts and their "correct" category mapping
    test_pairs = [
        (0.99, True,  "Identical wording"),
        (0.97, True,  "Trivial rewording"),
        (0.95, True,  "Same question, different phrasing"),
        (0.93, True,  "Paraphrase with same intent"),
        (0.90, False, "Same topic, different specificity"),
        (0.87, False, "Thematically related but different answer"),
        (0.82, False, "Loosely related"),
    ]

    print(f"\n  {'Similarity':12}  {'Same Answer?':14}  {'Safe to cache?':16}  Description")
    print("  " + "-" * 68)
    for sim, correct, desc in test_pairs:
        safe = "✓  YES" if sim >= 0.95 else ("~  MAYBE" if sim >= 0.92 else "✗  NO")
        ans  = "✓" if correct else "✗"
        print(f"  {sim:12.2f}  {ans:14}  {safe:16}  {desc}")

    print()
    print("  Recommended default threshold: 0.95")
    print("  Conservative (high-stakes):    0.97")
    print("  Aggressive (FAQ-only):         0.93")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Chapter 30 — Semantic Caching Demo (Python)")
    print("=" * 70 + "\n")

    run_faq_demo()
    run_cost_analysis()
    run_threshold_sweep()

    # Quick validator demo
    print("=" * 60)
    print("Quality Validator — Sampling Rates")
    print("=" * 60)
    validator = CacheQualityValidator()
    print(f"\n  {'Similarity':12}  {'Sample Rate':14}  {'Should Validate (first hit)?'}")
    print("  " + "-" * 50)
    for sim in [0.99, 0.97, 0.95, 0.93]:
        rate = validator.validation_sample_rate(sim)
        should = validator.should_validate(sim, hit_count=1, entry_age_hours=2)
        print(f"  {sim:12.2f}  {rate:13.0%}   {'yes' if should else 'no'}")
    print()
```

---

## 30.6  C++ Implementation: High-Performance Cache Core

```cpp
// semantic_cache_demo.cpp
// Chapter 30 — Semantic Caching Demo (C++)
//
// Demonstrates:
//   1. Cosine similarity computation (SIMD-friendly inner product)
//   2. In-memory vector store with linear scan (small cache) and
//      simplified bucket-based ANN (large cache)
//   3. LRU eviction with doubly-linked list + hash map
//   4. TTL expiry
//   5. Cache statistics and cost tracking
//   6. Break-even analysis
//
// Build: g++ -O2 -std=c++17 -o semantic_cache_demo semantic_cache_demo.cpp
// Run:   ./semantic_cache_demo

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

static void print_section(const std::string& t) {
    std::cout << "\n" << std::string(60, '-') << "\n  " << t << "\n" << std::string(60, '-') << "\n";
}

static double elapsed_ms(TimePoint t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Keyword → topic-ID for demo similarity grouping
static int topic_id(const std::string& text) {
    std::string t = text;
    for (char& c : t) c = static_cast<char>(std::tolower((unsigned char)c));
    if (t.find("password") != std::string::npos ||
        t.find("reset")    != std::string::npos ||
        t.find("forgot")   != std::string::npos)   return 1;
    if (t.find("email")    != std::string::npos ||
        t.find("address")  != std::string::npos)   return 2;
    if (t.find("hours")    != std::string::npos ||
        t.find("open")     != std::string::npos ||
        t.find("weekend")  != std::string::npos ||
        t.find("support")  != std::string::npos)   return 3;
    if (t.find("cancel")   != std::string::npos ||
        t.find("unsubscribe") != std::string::npos ||
        t.find("membership") != std::string::npos) return 4;
    if (t.find("refund")   != std::string::npos ||
        t.find("money")    != std::string::npos)   return 5;
    return 0;
}

// Deterministic stub: topic component (shared) + idiosyncratic noise (unique).
// noise_std=0.2 is calibrated so same-topic cosine ≈ 0.95–0.97 (above threshold 0.92)
// while different-topic cosine ≈ 0.05 (well below threshold).
static std::vector<float> pseudo_embed(const std::string& text, int dim = 64) {
    // Idiosyncratic component (unique to this text)
    uint64_t h = 14695981039346656037ULL;
    for (char c : text) { h ^= static_cast<uint8_t>(c); h *= 1099511628211ULL; }
    std::mt19937 rng(static_cast<uint32_t>(h));
    std::normal_distribution<float> idio(0.0f, 0.2f);  // noise tuned for threshold 0.92

    // Topic component (shared within semantic group)
    int tid = topic_id(text);
    std::mt19937 rng_topic(static_cast<uint32_t>(tid * 2654435761u));
    std::normal_distribution<float> topic_dist(0.0f, 1.0f);

    std::vector<float> v(dim);
    for (int i = 0; i < dim; ++i)
        v[i] = topic_dist(rng_topic) + idio(rng);

    // Normalize
    float norm = 0;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 0) for (float& x : v) x /= norm;
    return v;
}

static float cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0 ? dot / denom : 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1.  COSINE SIMILARITY — SCALAR VS UNROLLED
// ─────────────────────────────────────────────────────────────────────────────

static float dot_product_unrolled(const float* __restrict__ a,
                                   const float* __restrict__ b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i]   * b[i];
        s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2];
        s3 += a[i+3] * b[i+3];
    }
    float s = s0 + s1 + s2 + s3;
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

static void demo_similarity_perf() {
    print_section("Cosine Similarity — Scalar vs 4-way Unrolled");

    const int DIM    = 384;    // all-MiniLM-L6-v2 dimension
    const int N_VECS = 10000;  // number of stored entries to scan

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Generate N_VECS random normalized vectors
    std::vector<std::vector<float>> db(N_VECS, std::vector<float>(DIM));
    for (auto& v : db) {
        float norm = 0;
        for (float& x : v) { x = nd(rng); norm += x * x; }
        norm = std::sqrt(norm);
        for (float& x : v) x /= norm;
    }

    // Query vector
    std::vector<float> q(DIM);
    { float norm = 0;
      for (float& x : q) { x = nd(rng); norm += x * x; }
      norm = std::sqrt(norm);
      for (float& x : q) x /= norm; }

    // Scalar scan
    auto t0 = Clock::now();
    float best_s = -1; int best_i = 0;
    for (int i = 0; i < N_VECS; ++i) {
        float s = 0;
        for (int j = 0; j < DIM; ++j) s += q[j] * db[i][j];
        if (s > best_s) { best_s = s; best_i = i; }
    }
    double scalar_ms = elapsed_ms(t0);

    // Unrolled scan
    auto t1 = Clock::now();
    float best_u = -1; int best_ui = 0;
    for (int i = 0; i < N_VECS; ++i) {
        float s = dot_product_unrolled(q.data(), db[i].data(), DIM);
        if (s > best_u) { best_u = s; best_ui = i; }
    }
    double unrolled_ms = elapsed_ms(t1);

    std::cout << "\n  Scanning " << N_VECS << " vectors × " << DIM << " dims:\n";
    std::cout << "    Scalar:   " << std::fixed << std::setprecision(3)
              << scalar_ms << " ms  (best idx=" << best_i
              << " sim=" << std::setprecision(4) << best_s << ")\n";
    std::cout << "    Unrolled: " << std::setprecision(3)
              << unrolled_ms << " ms  (best idx=" << best_ui
              << " sim=" << std::setprecision(4) << best_u << ")\n";
    std::cout << "    Speedup:  " << std::setprecision(2)
              << scalar_ms / unrolled_ms << "×\n";

    assert(best_i == best_ui);
    std::cout << "  [ASSERT] Both methods find the same best match: ✓\n";

    // Throughput
    double ops   = (double)N_VECS * DIM * 2;  // multiply + add per element
    double gflops = ops / (scalar_ms * 1e-3) / 1e9;
    std::cout << "  Scalar throughput: " << std::setprecision(1) << gflops << " GFLOPS\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  LRU CACHE WITH TTL
// ─────────────────────────────────────────────────────────────────────────────

struct CacheEntry {
    std::string key;
    std::string prompt;
    std::string response;
    std::vector<float> embedding;
    int64_t     created_epoch_s;
    int64_t     last_hit_s;
    int         hit_count;
    int         input_tokens;
    int         output_tokens;
};

class SemanticCacheCore {
public:
    float threshold;
    int   max_entries;
    int   ttl_seconds;
    int   dim;

    struct Stats {
        long total = 0, exact_hits = 0, sem_hits = 0, misses = 0, evictions = 0;
        long tokens_saved = 0;
        double cost_saved = 0.0;
        double embed_ms = 0.0, search_ms = 0.0;
    } stats;

    explicit SemanticCacheCore(float thr = 0.95f, int max = 10000,
                                int ttl = 3600, int d = 64)
        : threshold(thr), max_entries(max), ttl_seconds(ttl), dim(d) {}

    // Returns response or "" on miss
    std::string get(const std::string& prompt, const std::string& model) {
        stats.total++;
        int64_t now = epoch_s();

        // Exact match
        std::string ek = exact_key(prompt, model);
        auto eit = exact_idx_.find(ek);
        if (eit != exact_idx_.end()) {
            auto mit = entries_.find(eit->second);
            if (mit != entries_.end()) {
                auto& e = mit->second;
                if (now - e.created_epoch_s <= ttl_seconds) {
                    e.hit_count++;
                    e.last_hit_s = now;
                    stats.exact_hits++;
                    record_savings(e);
                    touch_lru(e.key);
                    return e.response;
                } else {
                    evict(e.key);
                }
            }
        }

        if (entries_.empty()) { stats.misses++; return ""; }

        // Embed query
        auto t0 = Clock::now();
        auto qemb = pseudo_embed(prompt, dim);
        stats.embed_ms += elapsed_ms(t0);

        // Scan for best match
        auto t1 = Clock::now();
        float best_sim = -1.0f;
        std::string best_key;
        for (auto& [k, e] : entries_) {
            if (e.embedding.empty()) continue;
            if (now - e.created_epoch_s > ttl_seconds) continue;
            float s = cosine_sim(qemb, e.embedding);
            if (s > best_sim) { best_sim = s; best_key = k; }
        }
        stats.search_ms += elapsed_ms(t1);

        if (!best_key.empty() && best_sim >= threshold) {
            auto& e = entries_[best_key];
            e.hit_count++;
            e.last_hit_s = now;
            stats.sem_hits++;
            record_savings(e);
            touch_lru(best_key);
            return e.response;
        }
        stats.misses++;
        return "";
    }

    void put(const std::string& prompt, const std::string& response,
             const std::string& model, int in_tok = 0, int out_tok = 0) {
        if ((int)entries_.size() >= max_entries) evict_lru();

        auto t0 = Clock::now();
        auto emb = pseudo_embed(prompt, dim);
        stats.embed_ms += elapsed_ms(t0);

        std::string key = make_key(prompt);
        int64_t now = epoch_s();

        CacheEntry e;
        e.key = key; e.prompt = prompt; e.response = response;
        e.embedding = emb;
        e.created_epoch_s = now; e.last_hit_s = now;
        e.hit_count = 0; e.input_tokens = in_tok; e.output_tokens = out_tok;

        entries_[key] = std::move(e);
        exact_idx_[exact_key(prompt, model)] = key;
        lru_.push_front(key);
        lru_map_[key] = lru_.begin();
    }

    void print_stats() const {
        long n_req   = stats.total;
        long hits    = stats.exact_hits + stats.sem_hits;
        double hr    = n_req > 0 ? (double)hits / n_req : 0;
        double avg_e = (stats.sem_hits + stats.misses) > 0
            ? stats.embed_ms  / (stats.sem_hits + stats.misses) : 0;
        double avg_s = (stats.sem_hits + stats.misses) > 0
            ? stats.search_ms / (stats.sem_hits + stats.misses) : 0;

        std::cout << "  Entries:         " << entries_.size()  << "\n";
        std::cout << "  Total requests:  " << n_req             << "\n";
        std::cout << "  Exact hits:      " << stats.exact_hits  << "\n";
        std::cout << "  Semantic hits:   " << stats.sem_hits    << "\n";
        std::cout << "  Misses:          " << stats.misses      << "\n";
        std::cout << "  Hit rate:        " << std::fixed << std::setprecision(1)
                  << hr * 100 << "%\n";
        std::cout << "  Tokens saved:    " << stats.tokens_saved << "\n";
        std::cout << "  Cost saved:     $" << std::setprecision(4)
                  << stats.cost_saved << "\n";
        std::cout << "  Avg embed ms:    " << std::setprecision(3) << avg_e << "\n";
        std::cout << "  Avg search ms:   " << std::setprecision(3) << avg_s << "\n";
    }

private:
    std::unordered_map<std::string, CacheEntry>           entries_;
    std::unordered_map<std::string, std::string>          exact_idx_;
    std::list<std::string>                                lru_;
    std::unordered_map<std::string, std::list<std::string>::iterator> lru_map_;

    static int64_t epoch_s() {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    static std::string make_key(const std::string& prompt) {
        // FNV hash as key
        uint64_t h = 14695981039346656037ULL;
        for (char c : prompt) { h ^= (uint8_t)c; h *= 1099511628211ULL; }
        std::ostringstream ss;
        ss << std::hex << h;
        return ss.str();
    }

    static std::string exact_key(const std::string& p, const std::string& m) {
        return make_key(p + "|" + m);
    }

    void touch_lru(const std::string& key) {
        auto it = lru_map_.find(key);
        if (it != lru_map_.end()) {
            lru_.erase(it->second);
            lru_.push_front(key);
            it->second = lru_.begin();
        }
    }

    void evict(const std::string& key) {
        entries_.erase(key);
        auto lit = lru_map_.find(key);
        if (lit != lru_map_.end()) {
            lru_.erase(lit->second);
            lru_map_.erase(lit);
        }
        stats.evictions++;
    }

    void evict_lru() {
        if (lru_.empty()) return;
        evict(lru_.back());
    }

    void record_savings(const CacheEntry& e) {
        stats.tokens_saved += e.input_tokens + e.output_tokens;
        stats.cost_saved   += e.input_tokens  / 1000.0 * 0.0005
                            + e.output_tokens / 1000.0 * 0.0015;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 3.  FAQ WORKLOAD SIMULATION
// ─────────────────────────────────────────────────────────────────────────────

static void demo_faq_workload() {
    print_section("FAQ Workload Simulation");

    SemanticCacheCore cache(0.92f, 1000, 86400, 64);

    // Seed with canonical questions
    struct Seed {
        const char* prompt;
        const char* response;
        int in_tok, out_tok;
    };
    const Seed seeds[] = {
        {"How do I reset my password?",
         "Click 'Forgot Password' on the login page. Check email for a reset link.",
         60, 40},
        {"How do I update my email address?",
         "Go to Account Settings > Profile > Email and enter your new address.",
         60, 35},
        {"What are your business hours?",
         "Support is available Monday-Friday 9am-6pm EST. Email support available 24/7.",
         50, 40},
        {"How do I cancel my subscription?",
         "Go to Account Settings > Billing > Cancel. Access continues until billing period ends.",
         60, 45},
        {"How do I request a refund?",
         "Refunds process in 5-7 business days. Email support@example.com with order number.",
         60, 45},
    };
    for (auto& s : seeds)
        cache.put(s.prompt, s.response, "gpt-4o-mini", s.in_tok, s.out_tok);

    // Test prompts
    const char* queries[] = {
        "How can I change my password?",
        "I forgot my password, what should I do?",
        "Password reset steps",
        "Change account email",
        "Update my email on the account",
        "When are you open for support?",
        "Is weekend support available?",
        "Cancel my membership",
        "I want to unsubscribe from the service",
        "Get a refund for my purchase",
        "How long do refunds take?",
        "How do I enable two-factor authentication?",   // not in cache
        "What payment methods do you accept?",          // not in cache
    };

    std::cout << "\n  " << std::left << std::setw(48) << "Query"
              << std::setw(10) << "Result" << "\n";
    std::cout << "  " << std::string(58, '-') << "\n";

    for (auto q : queries) {
        std::string result = cache.get(q, "gpt-4o-mini");
        bool hit = !result.empty();
        if (!hit) {
            // Simulate miss: store new entry
            cache.put(q, "Please contact support for assistance.",
                      "gpt-4o-mini", 80, 20);
        }
        std::cout << "  " << std::left << std::setw(48) << std::string(q).substr(0, 47)
                  << (hit ? "HIT" : "MISS") << "\n";
    }

    std::cout << "\n";
    cache.print_stats();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  COST MODEL
// ─────────────────────────────────────────────────────────────────────────────

static void demo_cost_model() {
    print_section("Break-Even Analysis");

    const double cost_in_per_1k  = 0.0005;
    const double cost_out_per_1k = 0.0015;
    const int    avg_in          = 256;
    const int    avg_out         = 128;
    const double embed_cost_req  = 0.000005;
    const double infra_monthly   = 30.0;
    const long   reqs_monthly    = 2000000;

    double inf_cost_req = avg_in / 1000.0 * cost_in_per_1k
                        + avg_out/ 1000.0 * cost_out_per_1k;
    double overhead_req = embed_cost_req + infra_monthly / reqs_monthly;
    double break_even   = overhead_req / inf_cost_req;

    std::cout << "\n  Inference cost/request: $" << std::fixed << std::setprecision(6)
              << inf_cost_req << "\n";
    std::cout << "  Overhead/request:      $" << std::setprecision(6)
              << overhead_req << "\n";
    std::cout << "  Break-even hit rate:    " << std::setprecision(1)
              << break_even * 100 << "%\n\n";

    std::cout << "  " << std::right << std::setw(12) << "Hit Rate"
              << std::setw(20) << "Monthly Savings"
              << std::setw(10) << "ROI"
              << "\n  " << std::string(42, '-') << "\n";

    for (int hr_pct : {5, 10, 20, 30, 40, 50, 60}) {
        double hr      = hr_pct / 100.0;
        double saved   = hr * reqs_monthly * inf_cost_req;
        double overhead= reqs_monthly * embed_cost_req + infra_monthly;
        double net     = saved - overhead;
        double roi     = net > 0 ? net / overhead : 0;
        std::string flag = (std::abs(hr - break_even) < 0.03) ? " ← break-even" : "";
        std::cout << "  " << std::right << std::setw(11) << hr_pct << "%"
                  << std::setw(18) << std::fixed << std::setprecision(2) << net
                  << std::setw(10) << std::setprecision(1) << roi << "x"
                  << flag << "\n";
    }

    // Assert break-even is < 10% for typical inference costs
    assert(break_even < 0.10);
    std::cout << "\n  [ASSERT] Break-even hit rate < 10%: "
              << std::setprecision(2) << break_even * 100 << "% ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  LRU EVICTION CORRECTNESS
// ─────────────────────────────────────────────────────────────────────────────

static void demo_lru_eviction() {
    print_section("LRU Eviction Correctness");

    // Use semantically-distinct prompts (each maps to a different topic_id)
    // so exact eviction is observable: evicted entry has NO surviving semantic match.
    const std::vector<std::string> prompts = {
        "How do I reset my password?",           // topic 1: password
        "Update my email address on the account",// topic 2: email
        "What are your business hours?",         // topic 3: hours/open
        "Cancel my subscription please",         // topic 4: cancel
        "I need a refund for my order"           // topic 5: refund
    };
    const std::string newcomer = "Is two-factor authentication available?"; // topic 0

    SemanticCacheCore cache(0.95f, /*max=*/5, /*ttl=*/9999, 64);

    for (int i = 0; i < 5; ++i)
        cache.put(prompts[i], "response " + std::to_string(i), "m", 100, 50);
    std::cout << "\n  Filled cache to capacity (5 entries, all distinct topics)\n";

    // Touch prompts[0] → it becomes MRU; prompts[1] becomes LRU
    cache.get(prompts[0], "m");
    std::cout << "  Accessed entry 0 (password topic) — now MRU\n";

    // Insert newcomer → evicts LRU = prompts[1] (email topic)
    cache.put(newcomer, "response-new", "m", 100, 50);
    std::cout << "  Added new entry (2FA topic) → entry 1 (email) evicted\n";

    // Verify
    std::string r0  = cache.get(prompts[0], "m");   // should HIT (was MRU)
    std::string r1  = cache.get(prompts[1], "m");   // should MISS (was evicted, distinct topic)
    std::string rnew = cache.get(newcomer,  "m");   // should HIT (just inserted)

    std::cout << "\n  Entry 0 (MRU, password):  " << (r0.empty()  ? "MISS (BUG)" : "HIT  ✓") << "\n";
    std::cout << "  Entry 1 (LRU, email):     " << (r1.empty()  ? "MISS ✓ (evicted)" : "HIT  (BUG)") << "\n";
    std::cout << "  New entry (2FA):          " << (rnew.empty() ? "MISS (BUG)" : "HIT  ✓") << "\n";

    assert(r0.empty()  == false);  // MRU entry retained
    assert(r1.empty()  == true);   // LRU entry evicted — no semantic match (distinct topic)
    assert(rnew.empty() == false); // newly inserted entry retained
    std::cout << "\n  [ASSERT] LRU eviction correctness (MRU retained, LRU evicted): ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n" << std::string(60, '=')
              << "\n  Chapter 30 — Semantic Caching Demo (C++)\n"
              << std::string(60, '=') << "\n";

    demo_similarity_perf();
    demo_faq_workload();
    demo_cost_model();
    demo_lru_eviction();

    std::cout << "\n" << std::string(60, '=')
              << "\n  All demos complete.\n"
              << std::string(60, '=') << "\n\n";
    return 0;
}
```

---

## 30.7  Where Semantic Caching Fits in the Stack

### 30.7.1  Relationship to vLLM Prefix Caching

vLLM's prefix caching (Chapter 11) and semantic caching operate at different layers
and serve different purposes:

| Dimension | vLLM Prefix Cache | Semantic Cache |
|---|---|---|
| Location | Inside the inference engine | Outside, in the application layer |
| Granularity | KV blocks (token sequences) | Full request/response pairs |
| Match type | Exact prefix match | Approximate semantic match |
| What it saves | Prefill compute only | Entire inference (prefill + decode) |
| Persistence | GPU memory; lost on restart | Redis / Postgres; durable |
| Sharing | Per-server instance | Shared across all instances |
| Suitable for | System prompt reuse | FAQ, repeated questions |

The two mechanisms are complementary.
In a typical deployment:
1. Semantic cache intercepts ~30% of requests before they reach vLLM.
2. For requests that reach vLLM, prefix caching reuses KV blocks for the shared
   system prompt across all requests — saving prefill for the 70% that miss the
   semantic cache.

### 30.7.2  Cache Placement Architecture

```
                    ┌──────────────────────┐
Client requests ──► │  Load Balancer / API │
                    │  Gateway             │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Semantic Cache      │  ← Application layer
                    │  (Redis + FAISS)     │    ~1–5ms for hit check
                    └──────────┬───────────┘
                       miss    │
                    ┌──────────▼───────────┐
                    │  vLLM / llama.cpp    │  ← Inference layer
                    │  (with prefix cache) │    10ms–10s for generation
                    └──────────────────────┘
```

The semantic cache should be as close to the client as possible (same data center,
preferably same availability zone) to minimize the hit-path latency.
The cache lookup (embed + search) must be faster than the threshold where users notice
latency — typically sub-10ms.

### 30.7.3  Cache Partitioning

Always partition the cache by:
- **Model version:** `gpt-4o-mini` and `gpt-4o` responses are different; never mix them.
- **System prompt hash:** Different system prompts produce different valid responses.
- **Temperature bucket:** High-temperature responses should not be cached (non-deterministic).
  Use `temperature = 0` as the only cacheable bucket.
- **User tier** (optional): Responses for enterprise users may have different SLAs or
  access to different information.

---

## 30.8  Invalidation Strategies

Cache invalidation is the hardest part of semantic caching.

### 30.8.1  TTL-Based Expiry

The simplest strategy: every entry expires after `TTL` seconds.

Choosing TTL:
- **FAQ content that rarely changes:** 7 days
- **Product information (prices, features):** 4–24 hours
- **News and current events:** 15 minutes or zero (don't cache)
- **Generated code assistance:** 24 hours (language spec doesn't change daily)
- **Model-dependent content:** expire on model deployment

### 30.8.2  Event-Driven Invalidation

When underlying data changes, proactively invalidate affected cache entries:

```python
def on_product_price_update(product_id: str):
    """Called when a product's price changes — invalidate related entries."""
    # Pattern-based invalidation in Redis
    keys = redis_client.keys(f"cache:product:{product_id}:*")
    if keys:
        redis_client.delete(*keys)
    
    # For semantic stores without pattern keys, invalidate entire model partition
    cache.invalidate(model="gpt-4o-mini", tags=["product", product_id])
```

### 30.8.3  Model Deployment Invalidation

When a new model version is deployed, all cached responses from the old model are
potentially stale (the new model may give different, better answers).
Always include the model identifier and version in the cache key.

```python
CACHE_VERSION = "v3"  # bump on each model deployment

def cache_key_prefix(model: str) -> str:
    return f"{CACHE_VERSION}:{model}"
```

On deployment, either:
1. Flush all old-version entries immediately (clean break, cold start)
2. Let TTL drain old entries over days (gradual transition, mixed results briefly)

Option 1 is safer for quality; option 2 avoids cold-start latency spikes.

### 30.8.4  Semantic Drift Detection

Over time, the distribution of questions shifts, and the embedding model's semantic
space may no longer align with the cached entries.
Monitor for **cache staleness signals:**
- Hit rate drops significantly without traffic change
- User feedback on cache-hit responses trends negative
- Downstream task accuracy on cache hits diverges from fresh responses

When detected, consider a full cache flush and rebuild.

---

## 30.9  Production Checklist

Before deploying a semantic cache in production:

**Quality gates:**
- [ ] Measure answer accuracy on cache hits at your chosen `τ` against held-out test set
- [ ] Set up sampling pipeline: 5–10% of hits re-run inference for quality monitoring
- [ ] Alert on hit accuracy dropping below threshold (e.g., < 95% agreement)
- [ ] Never cache responses where temperature > 0.1

**Performance:**
- [ ] Embedding latency < 5ms (use ONNX Runtime on CPU for MiniLM-L6)
- [ ] Cache lookup (embed + search) < 10ms p99
- [ ] Cache store (on miss) < 5ms (async write is acceptable)

**Safety:**
- [ ] Partition cache by model name + version
- [ ] Partition cache by system prompt hash (different prompts = different cache)
- [ ] TTL set appropriately for content freshness requirements
- [ ] Event-driven invalidation wired to content update systems

**Operations:**
- [ ] Cache hit rate dashboard (alert on unexplained drops)
- [ ] Cost savings tracking (verify ROI is positive)
- [ ] Manual flush endpoint (for emergency invalidation)
- [ ] Cache size monitoring (prevent unbounded growth)

---

## 30.10  Summary

Semantic caching is one of the highest-leverage optimizations available to an LLM
inference platform.
For FAQ-heavy workloads it delivers 30–60% cost reduction with sub-5ms latency for
cache hits, requiring no changes to the model or inference engine.

The similarity threshold `τ` is the central engineering decision.
A threshold of 0.95 with `all-MiniLM-L6-v2` embeddings and temperature=0 caching is a
safe default for most production deployments.

The cache lives at the application layer, above the inference engine.
It complements — rather than replaces — vLLM's prefix caching, which operates inside
the engine on KV blocks.

Invalidation is harder than it looks.
Always use TTL, always partition by model+system_prompt, and always monitor hit quality
as a first-class metric alongside hit rate.

---

*Chapter 31 addresses model routing and cascading: how to dispatch requests to
different models (small vs. large, specialist vs. generalist) based on query complexity,
cost budget, and required capability — and how to build a cascade that falls back
gracefully when the small model isn't enough.*


---

## Chapter Summary

- **Semantic caching motivation**: exact-match prefix caching hits only when token sequences are identical; semantic caching uses embedding similarity to serve cached responses for semantically equivalent queries.
- **Architecture**: a semantic cache sits in front of the inference engine; on each request, it embeds the query and searches a vector index (FAISS, Qdrant, pgvector) for a cached match within similarity threshold θ.
- **Threshold selection**: θ too low → false positives (wrong answer served); θ too high → cache misses; θ ≈ 0.97 cosine similarity is a common starting point for FAQ-style workloads.
- **Hit rate and savings**: for enterprise support bots and FAQ applications, semantic cache hit rates of 30–70% are achievable, each hit saving the full inference cost of that request.
- **Cache invalidation**: when the underlying model changes (version update, fine-tune), all cached responses must be invalidated; store model version alongside cache entries.
- **Embedding model choice**: the embedding model must have lower latency than inference; a 100 ms semantic cache lookup that saves a 500 ms inference call gives net 400 ms saving.
- **Safety considerations**: semantic caching can serve stale responses; always include a TTL on cached entries and never cache responses to requests that require real-time information.

---

## Self-Check Questions

1. A semantic cache uses cosine similarity with θ = 0.97. Query A is "What is the capital of France?" and query B is "Name the capital city of France." If their embeddings have cosine similarity 0.988, is the cache hit served? What potential problem does this raise for factual questions? *(Section 30.2)*

2. The embedding model takes 12 ms per query. Inference takes 400 ms. Cache hit rate is 45%. Compute the average latency per request with and without the semantic cache. *(Section 30.3)*

3. Your semantic cache serves 500K requests/day with 40% hit rate. Each cache miss costs $0.004 in compute. Each hit costs $0.0001 (embedding only). Compute the daily and monthly cost saving. *(Section 30.4)*

4. The model is updated from version 1 to version 2. The cache contains 50 000 entries from version 1. Describe the invalidation strategy: how do you prevent version 1 responses from being served under version 2? *(Section 30.5)*

5. A user asks "What happened in the news today?" The semantic cache returns a response from 3 days ago with similarity 0.995. This is a cache hit. Why is this dangerous, and what cache policy prevents it? *(Section 30.6)*
