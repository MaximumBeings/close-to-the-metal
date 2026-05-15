# Chapter 30: Semantic Caching — Companion Code

## Python — `semantic_cache_demo.py`

```python
"""
semantic_cache_demo.py
Chapter 30 — Semantic Caching and Response Reuse

Demonstrates:
  1. Exact-match caching (hash-based, O(1))
  2. Semantic-match caching (embedding cosine similarity)
  3. LRU eviction with TTL expiry
  4. Cache statistics and cost model
  5. Break-even hit-rate analysis
  6. Threshold sweep (precision vs. recall tradeoff)
  7. Cache invalidation strategies (TTL buckets, event-driven flush)
  8. FAISS-style index simulation for large caches
  9. Partitioned cache (model × system-prompt hash × temperature bucket)
 10. Quality validation via re-inference sampling

Run:
    python semantic_cache_demo.py

Requirements (all stdlib + numpy + optional sentence-transformers):
    pip install numpy
    # sentence-transformers is optional; demo uses a fast deterministic stub
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# §0  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

EMBED_DIM   = 384      # all-MiniLM-L6-v2 output dimension
THRESHOLD   = 0.93     # default cosine-similarity hit threshold
MAX_ENTRIES = 10_000   # LRU capacity
TTL_SECONDS = 3_600    # default entry lifetime (1 hour)

# ──────────────────────────────────────────────────────────────────────────────
# §1  EMBEDDING STUB
#     In production: replace with sentence_transformers.SentenceTransformer
#     ("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True)
# ──────────────────────────────────────────────────────────────────────────────

# Topic keyword → group ID (mimics semantic clustering)
_TOPIC_KEYWORDS: List[Tuple[int, List[str]]] = [
    (1,  ["password", "reset", "forgot", "credentials"]),
    (2,  ["email", "address", "inbox", "mail"]),
    (3,  ["hours", "open", "weekend", "support", "schedule"]),
    (4,  ["cancel", "unsubscribe", "membership", "terminate"]),
    (5,  ["refund", "money", "charge", "billing", "payment"]),
    (6,  ["shipping", "delivery", "tracking", "order"]),
    (7,  ["account", "login", "sign", "profile"]),
    (8,  ["download", "install", "update", "version"]),
]

def _topic_id(text: str) -> int:
    lower = text.lower()
    for tid, keywords in _TOPIC_KEYWORDS:
        if any(k in lower for k in keywords):
            return tid
    return 0

def pseudo_embed(text: str, dim: int = EMBED_DIM, noise_std: float = 0.2) -> np.ndarray:
    """
    Deterministic stub that produces embeddings with controlled intra-topic
    cosine similarity (~0.95-0.97) and inter-topic similarity (~0.05).

    noise_std=0.2 calibrated so same-topic similarity > threshold=0.92
    while keeping different-topic similarity well below threshold.
    """
    # Topic component — shared within semantic group
    tid = _topic_id(text)
    rng_topic = np.random.default_rng(seed=tid * 2_654_435_761 % (2**32))
    topic_vec = rng_topic.standard_normal(dim).astype(np.float32)

    # Idiosyncratic component — unique to this exact text
    h = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng_idio = np.random.default_rng(seed=h)
    idio_vec = rng_idio.normal(0.0, noise_std, dim).astype(np.float32)

    v = topic_vec + idio_vec
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-norm vectors (fast dot product)."""
    return float(np.dot(a, b))

# ──────────────────────────────────────────────────────────────────────────────
# §2  CACHE ENTRY
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    key:            str
    prompt:         str
    response:       str
    embedding:      np.ndarray
    model:          str
    temperature:    float
    created_at:     float = field(default_factory=time.time)
    last_hit_at:    float = field(default_factory=time.time)
    hit_count:      int   = 0
    input_tokens:   int   = 0
    output_tokens:  int   = 0

# ──────────────────────────────────────────────────────────────────────────────
# §3  CACHE STATS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    total_requests:  int   = 0
    exact_hits:      int   = 0
    semantic_hits:   int   = 0
    misses:          int   = 0
    evictions:       int   = 0
    ttl_expirations: int   = 0
    tokens_saved:    int   = 0
    cost_saved_usd:  float = 0.0
    total_embed_ms:  float = 0.0
    total_search_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.exact_hits + self.semantic_hits) / self.total_requests

    @property
    def avg_embed_ms(self) -> float:
        n = self.exact_hits + self.semantic_hits + self.misses
        return self.total_embed_ms / n if n > 0 else 0.0

    @property
    def avg_search_ms(self) -> float:
        n = self.semantic_hits + self.misses
        return self.total_search_ms / n if n > 0 else 0.0

    def report(self) -> str:
        lines = [
            f"  Total requests:   {self.total_requests}",
            f"  Exact hits:       {self.exact_hits}",
            f"  Semantic hits:    {self.semantic_hits}",
            f"  Misses:           {self.misses}",
            f"  Hit rate:         {self.hit_rate * 100:.1f}%",
            f"  Evictions:        {self.evictions}",
            f"  TTL expirations:  {self.ttl_expirations}",
            f"  Tokens saved:     {self.tokens_saved:,}",
            f"  Cost saved:       ${self.cost_saved_usd:.4f}",
            f"  Avg embed ms:     {self.avg_embed_ms:.3f}",
            f"  Avg search ms:    {self.avg_search_ms:.3f}",
        ]
        return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# §4  COST MODEL
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CostModel:
    """
    Prices in USD per 1 000 tokens (GPT-4o-mini scale as reference).
    Embedding: all-MiniLM-L6-v2 on CPU ~2 ms/query → ~$0.000005/req.
    Redis/pgvector ANN: ~1 ms/query → $30/M req ≈ $0.000030/req.
    """
    input_price_per_1k:   float = 0.0005   # USD / 1k input tokens
    output_price_per_1k:  float = 0.0015   # USD / 1k output tokens
    embed_cost_per_req:   float = 0.000005 # embedding inference
    infra_cost_per_req:   float = 0.000015 # vector search + network

    def inference_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens  / 1000) * self.input_price_per_1k + \
               (output_tokens / 1000) * self.output_price_per_1k

    def overhead_cost(self) -> float:
        return self.embed_cost_per_req + self.infra_cost_per_req

    def break_even_hit_rate(self, avg_input: int = 256, avg_output: int = 128) -> float:
        """
        Minimum hit rate h* such that expected savings ≥ 0.
        h* = overhead / (inference_cost - overhead)
        Simplified: h* ≈ overhead / inference_cost (when overhead << inference_cost)
        """
        inf_cost = self.inference_cost(avg_input, avg_output)
        overhead = self.overhead_cost()
        if inf_cost <= overhead:
            return 1.0  # not worth caching
        return overhead / (inf_cost - overhead)

    def monthly_net_savings(self, hit_rate: float,
                            requests_per_day: int = 100_000,
                            avg_input: int = 256,
                            avg_output: int = 128) -> float:
        total_req  = requests_per_day * 30
        hits       = total_req * hit_rate
        inf_saved  = hits * self.inference_cost(avg_input, avg_output)
        overhead   = total_req * self.overhead_cost()
        return inf_saved - overhead

# ──────────────────────────────────────────────────────────────────────────────
# §5  SEMANTIC CACHE
# ──────────────────────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Two-level semantic cache:
      Level 1 — exact match: hash(prompt + model + temp_bucket) → O(1)
      Level 2 — semantic match: embed → cosine scan → best similarity ≥ threshold

    Eviction: LRU via OrderedDict (O(1) move-to-end on hit).
    Expiry:   TTL checked on every access; expired entries purged lazily.
    """

    def __init__(
        self,
        threshold:   float = THRESHOLD,
        max_entries: int   = MAX_ENTRIES,
        ttl_seconds: float = TTL_SECONDS,
        embed_fn             = None,
        cost_model:  CostModel = None,
    ):
        self.threshold   = threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.embed_fn    = embed_fn or pseudo_embed
        self.cost_model  = cost_model or CostModel()
        self.stats       = CacheStats()

        # LRU store: key → CacheEntry (OrderedDict preserves insertion / access order)
        self._lru: "OrderedDict[str, CacheEntry]" = OrderedDict()
        # Exact-match index: exact_key → entry_key
        self._exact: Dict[str, str] = {}

    # ── key helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    @staticmethod
    def _temp_bucket(temperature: float) -> str:
        """Bucket temperature to avoid over-fragmentation (0→'zero', 0-0.5→'low', …)."""
        if temperature == 0.0:
            return "zero"
        elif temperature <= 0.5:
            return "low"
        elif temperature <= 1.0:
            return "mid"
        else:
            return "high"

    def _exact_key(self, prompt: str, model: str, temperature: float) -> str:
        bucket = self._temp_bucket(temperature)
        raw = f"{prompt}|||{model}|||{bucket}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    # ── TTL ──────────────────────────────────────────────────────────────────

    def _is_expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.created_at) > self.ttl_seconds

    def _evict_key(self, key: str) -> None:
        if key in self._lru:
            entry = self._lru.pop(key)
            ek = self._exact_key(entry.prompt, entry.model, entry.temperature)
            self._exact.pop(ek, None)
            self.stats.evictions += 1

    def _evict_lru(self) -> None:
        if self._lru:
            oldest_key = next(iter(self._lru))
            self._evict_key(oldest_key)

    # ── public API ────────────────────────────────────────────────────────────

    def get(
        self,
        prompt:      str,
        model:       str  = "default",
        temperature: float = 0.0,
    ) -> Optional[str]:
        """Return cached response string, or None on miss."""
        self.stats.total_requests += 1

        # ── Level 1: exact match ──────────────────────────────────────────
        ek = self._exact_key(prompt, model, temperature)
        if ek in self._exact:
            key = self._exact[ek]
            if key in self._lru:
                entry = self._lru[key]
                if self._is_expired(entry):
                    self._evict_key(key)
                    self.stats.ttl_expirations += 1
                else:
                    entry.hit_count  += 1
                    entry.last_hit_at = time.time()
                    self._lru.move_to_end(key)
                    self.stats.exact_hits   += 1
                    self.stats.tokens_saved += entry.input_tokens + entry.output_tokens
                    self.stats.cost_saved_usd += self.cost_model.inference_cost(
                        entry.input_tokens, entry.output_tokens)
                    return entry.response

        if not self._lru:
            self.stats.misses += 1
            return None

        # ── Level 2: semantic match ───────────────────────────────────────
        t0 = time.perf_counter()
        q_emb = self.embed_fn(prompt)
        self.stats.total_embed_ms += (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        best_sim  = -1.0
        best_key  = None
        query_bucket = self._temp_bucket(temperature)
        for key, entry in self._lru.items():
            if self._is_expired(entry):
                continue
            # Partition filter: only match same model and temperature bucket
            if entry.model != model:
                continue
            if self._temp_bucket(entry.temperature) != query_bucket:
                continue
            sim = cosine_similarity(q_emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_key = key
        self.stats.total_search_ms += (time.perf_counter() - t1) * 1000

        if best_key is not None and best_sim >= self.threshold:
            entry = self._lru[best_key]
            entry.hit_count  += 1
            entry.last_hit_at = time.time()
            self._lru.move_to_end(best_key)
            self.stats.semantic_hits    += 1
            self.stats.tokens_saved     += entry.input_tokens + entry.output_tokens
            self.stats.cost_saved_usd   += self.cost_model.inference_cost(
                entry.input_tokens, entry.output_tokens)
            return entry.response

        self.stats.misses += 1
        return None

    def put(
        self,
        prompt:       str,
        response:     str,
        model:        str   = "default",
        temperature:  float = 0.0,
        input_tokens: int   = 0,
        output_tokens: int  = 0,
    ) -> None:
        """Store a prompt → response pair with its embedding."""
        if len(self._lru) >= self.max_entries:
            self._evict_lru()

        t0  = time.perf_counter()
        emb = self.embed_fn(prompt)
        self.stats.total_embed_ms += (time.perf_counter() - t0) * 1000

        key   = self._make_key(prompt)
        entry = CacheEntry(
            key=key, prompt=prompt, response=response,
            embedding=emb, model=model, temperature=temperature,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )
        self._lru[key] = entry
        self._lru.move_to_end(key)
        self._exact[self._exact_key(prompt, model, temperature)] = key

    def flush_model(self, model: str) -> int:
        """Invalidate all entries for a specific model version (deployment flush)."""
        to_remove = [k for k, e in self._lru.items() if e.model == model]
        for k in to_remove:
            self._evict_key(k)
        return len(to_remove)

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove entries whose prompt starts with a given prefix (RAG doc update)."""
        to_remove = [k for k, e in self._lru.items() if e.prompt.startswith(prefix)]
        for k in to_remove:
            self._evict_key(k)
        return len(to_remove)

    def size(self) -> int:
        return len(self._lru)

# ──────────────────────────────────────────────────────────────────────────────
# §6  QUALITY VALIDATOR
# ──────────────────────────────────────────────────────────────────────────────

class CacheQualityValidator:
    """
    Samples a fraction of semantic cache hits and re-runs inference to compare
    the served cached response with a fresh response.

    In production: replace _re_infer with an actual LLM API call and use a
    text-similarity metric (ROUGE-L, BERTScore, or a small LLM judge).

    similarity_threshold is set for token-level Jaccard after punctuation removal.
    Jaccard on paraphrased sentences typically lands in [0.05, 0.30]; a threshold
    of 0.08 filters clearly-wrong cache hits while accepting valid paraphrases.
    """

    def __init__(self, sample_rate: float = 0.1, similarity_threshold: float = 0.08):
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self._checks: List[Dict] = []

    def check(self, query: str, cached_response: str, fresh_response: str) -> bool:
        sim = self._response_similarity(cached_response, fresh_response)
        ok  = sim >= self.similarity_threshold
        self._checks.append({
            "query":   query[:60],
            "sim":     sim,
            "ok":      ok,
        })
        return ok

    @staticmethod
    def _normalize(text: str) -> set:
        """Lowercase, strip punctuation, split into tokens."""
        import re
        return set(re.sub(r"[^\w\s]", "", text.lower()).split())

    @classmethod
    def _response_similarity(cls, a: str, b: str) -> float:
        """
        Token-level Jaccard similarity after punctuation normalization.
        Fast proxy for quality checking; in production use ROUGE-L or BERTScore.
        """
        ta, tb = cls._normalize(a), cls._normalize(b)
        if not ta and not tb:
            return 1.0
        return len(ta & tb) / len(ta | tb)

    def accuracy(self) -> float:
        if not self._checks:
            return 1.0
        return sum(1 for c in self._checks if c["ok"]) / len(self._checks)

    def report(self) -> str:
        if not self._checks:
            return "  No quality checks performed."
        lines = [f"  Quality checks: {len(self._checks)}, accuracy: {self.accuracy()*100:.1f}%"]
        for c in self._checks:
            mark = "✓" if c["ok"] else "✗"
            lines.append(f"  [{mark}] sim={c['sim']:.3f}  \"{c['query']}\"")
        return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# §7  FAISS-STYLE FLAT INDEX SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

class FlatIndex:
    """
    Simulates a FAISS IndexFlatIP (inner product = cosine for unit-norm vectors).
    Production use:
        import faiss
        index = faiss.IndexFlatIP(dim)
        index.add(np.stack(embeddings))
        D, I = index.search(query[None], k=1)
    """

    def __init__(self, dim: int = EMBED_DIM):
        self.dim       = dim
        self.vectors:  List[np.ndarray] = []
        self.metadata: List[str]        = []

    def add(self, vec: np.ndarray, meta: str) -> None:
        self.vectors.append(vec.copy())
        self.metadata.append(meta)

    def search(self, query: np.ndarray, k: int = 1) -> List[Tuple[float, str]]:
        if not self.vectors:
            return []
        mat     = np.stack(self.vectors)          # (N, dim)
        scores  = mat @ query                     # cosine (unit-norm)
        top_k   = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.metadata[i]) for i in top_k]

    def __len__(self) -> int:
        return len(self.vectors)

# ──────────────────────────────────────────────────────────────────────────────
# §8  DEMO FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")

# ── 8.1  Embedding similarity verification ──────────────────────────────────

def demo_similarity_verification() -> None:
    section("Embedding Similarity Verification")

    pairs_same = [
        ("How do I reset my password?",        "How can I change my password?"),
        ("What are your business hours?",       "When are you open for support?"),
        ("I want to cancel my subscription",    "How do I unsubscribe from the service?"),
    ]
    pairs_diff = [
        ("How do I reset my password?",         "What are your business hours?"),
        ("What are your business hours?",       "I want a refund"),
    ]

    print(f"\n  Same-topic pairs (expect sim ≥ {THRESHOLD}):")
    all_above = True
    for a, b in pairs_same:
        sim = cosine_similarity(pseudo_embed(a), pseudo_embed(b))
        mark = "✓" if sim >= THRESHOLD else "✗"
        if sim < THRESHOLD:
            all_above = False
        print(f"  [{mark}] {sim:.4f}  \"{a[:40]}\"")

    print(f"\n  Different-topic pairs (expect sim < {THRESHOLD}):")
    all_below = True
    for a, b in pairs_diff:
        sim = cosine_similarity(pseudo_embed(a), pseudo_embed(b))
        mark = "✓" if sim < THRESHOLD else "✗"
        if sim >= THRESHOLD:
            all_below = False
        print(f"  [{mark}] {sim:.4f}  \"{a[:40]}\"")

    assert all_above, "Some same-topic pairs are below threshold"
    assert all_below, "Some different-topic pairs are above threshold"
    print("\n  [ASSERT] Same-topic ≥ threshold, different-topic < threshold: ✓")

# ── 8.2  FAQ workload ────────────────────────────────────────────────────────

FAQ_SEEDS = [
    # (seed_prompt, seed_response, in_tok, out_tok)
    ("How do I reset my password?",
     "Go to Settings → Security → Reset Password and follow the email link.",
     40, 20),
    ("What are your business hours?",
     "We're open Monday–Friday 9am–6pm EST. Weekend support via email.",
     35, 18),
    ("How do I cancel my subscription?",
     "Visit Account → Billing → Cancel. Your access continues until period end.",
     42, 19),
    ("I need a refund for my purchase",
     "Refunds are processed within 5–7 business days to your original payment method.",
     38, 22),
    ("How do I change my email address?",
     "Go to Profile → Contact Info → Edit Email. A verification link will be sent.",
     44, 21),
]

FAQ_QUERIES = [
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
    "How do I enable two-factor authentication?",   # off-topic → MISS
    "What payment methods do you accept?",
]

def demo_faq_workload() -> None:
    section("FAQ Workload Simulation")

    cache = SemanticCache(threshold=THRESHOLD, max_entries=100, ttl_seconds=9999)

    # Seed the cache with canonical FAQ answers
    for prompt, response, in_tok, out_tok in FAQ_SEEDS:
        cache.put(prompt, response, input_tokens=in_tok, output_tokens=out_tok)

    print(f"\n  {'Query':<50} {'Result'}")
    print(f"  {'─'*50} {'─'*10}")

    results = []
    for q in FAQ_QUERIES:
        resp = cache.get(q)
        hit  = resp is not None
        results.append(hit)
        label = "HIT" if hit else "MISS"
        print(f"  {q:<50} {label}")

    print()
    print(cache.stats.report())

    hit_rate = sum(results) / len(results)
    assert hit_rate >= 0.85, f"FAQ hit rate {hit_rate:.1%} too low (expected ≥85%)"
    print(f"\n  [ASSERT] FAQ hit rate ≥ 85%: {hit_rate:.1%} ✓")

# ── 8.3  Break-even analysis ─────────────────────────────────────────────────

def demo_break_even() -> None:
    section("Break-Even Analysis")

    cm = CostModel(
        input_price_per_1k  = 0.0005,
        output_price_per_1k = 0.0015,
        embed_cost_per_req  = 0.000005,
        infra_cost_per_req  = 0.000015,
    )

    be = cm.break_even_hit_rate(avg_input=256, avg_output=128)
    inf_cost = cm.inference_cost(256, 128)
    overhead = cm.overhead_cost()

    print(f"\n  Inference cost/req: ${inf_cost:.6f}")
    print(f"  Cache overhead/req: ${overhead:.6f}")
    print(f"  Break-even hit rate: {be*100:.2f}%")
    print()

    print(f"  {'Hit Rate':>10}  {'Net Savings/mo':>16}  {'ROI':>8}")
    print(f"  {'─'*10}  {'─'*16}  {'─'*8}")
    for hr_pct in [5, 10, 20, 30, 40, 50, 60]:
        hr      = hr_pct / 100
        net     = cm.monthly_net_savings(hr, requests_per_day=100_000)
        roi     = net / (100_000 * 30 * overhead) if overhead > 0 else 0
        marker  = " ← break-even" if abs(hr - be) < 0.03 else ""
        print(f"  {hr_pct:>9}%  ${net:>14.2f}  {roi:>7.1f}x{marker}")

    assert be < 0.10, f"Break-even {be:.2%} should be < 10%"
    print(f"\n  [ASSERT] Break-even hit rate < 10%: {be*100:.2f}% ✓")

# ── 8.4  LRU eviction correctness ────────────────────────────────────────────

def demo_lru_eviction() -> None:
    section("LRU Eviction Correctness")

    # Distinct-topic prompts so eviction is semantically unambiguous
    prompts = [
        "How do I reset my password?",           # topic 1: password
        "Update my email address on the account",# topic 2: email
        "What are your business hours?",         # topic 3: hours
        "Cancel my subscription please",         # topic 4: cancel
        "I need a refund for my order",          # topic 5: refund
    ]
    newcomer = "Is two-factor authentication supported?"  # topic 7: account/login

    cache = SemanticCache(threshold=0.95, max_entries=5, ttl_seconds=9999)

    for i, p in enumerate(prompts):
        cache.put(p, f"response-{i}", input_tokens=50, output_tokens=30)
    print(f"\n  Filled cache to capacity ({len(prompts)} entries, distinct topics)")

    # Touch entry 0 → becomes MRU; entry 1 becomes LRU
    cache.get(prompts[0])
    print(f"  Accessed entry 0 (password topic) — now MRU")

    # Add newcomer → triggers eviction of LRU = entry 1 (email)
    cache.put(newcomer, "2FA is available in Settings → Security.",
              input_tokens=50, output_tokens=30)
    print(f"  Added newcomer (account/login topic) → entry 1 (email) evicted")

    r0 = cache.get(prompts[0])   # MRU — should HIT
    r1 = cache.get(prompts[1])   # LRU (evicted) — should MISS
    rn = cache.get(newcomer)     # just inserted — should HIT

    print(f"\n  Entry 0 (MRU, password):    {'HIT  ✓' if r0 else 'MISS (BUG)'}")
    print(f"  Entry 1 (LRU, email):       {'MISS ✓ (evicted)' if r1 is None else 'HIT  (BUG)'}")
    print(f"  Newcomer (account/login):   {'HIT  ✓' if rn else 'MISS (BUG)'}")

    assert r0 is not None, "MRU entry was incorrectly evicted"
    assert r1 is None,     "LRU entry should have been evicted"
    assert rn is not None, "Newly inserted entry should be present"
    print(f"\n  [ASSERT] LRU eviction correctness: ✓")

# ── 8.5  TTL expiry ───────────────────────────────────────────────────────────

def demo_ttl_expiry() -> None:
    section("TTL Expiry")

    cache = SemanticCache(threshold=THRESHOLD, max_entries=100, ttl_seconds=0.05)

    cache.put("What is the weather today?", "It is sunny and 72°F.",
              input_tokens=30, output_tokens=10)

    r_before = cache.get("What is the weather today?")
    print(f"\n  Immediate lookup:  {'HIT ✓' if r_before else 'MISS'}")
    assert r_before is not None, "Entry should be present immediately after put"

    time.sleep(0.1)   # let TTL expire (50 ms TTL)

    r_after = cache.get("What is the weather today?")
    print(f"  After TTL expiry:  {'MISS ✓ (expired)' if r_after is None else 'HIT (BUG)'}")
    assert r_after is None, "Entry should be expired after TTL"

    print(f"\n  [ASSERT] TTL expiry works correctly: ✓")

# ── 8.6  Threshold sweep ─────────────────────────────────────────────────────

def demo_threshold_sweep() -> None:
    section("Threshold Sweep — Precision vs. Recall Tradeoff")

    # Ground truth: same-topic pairs should hit; different-topic should not
    same_topic = [
        ("How do I reset my password?", "How can I change my password?"),
        ("What are your business hours?", "When are you open for support?"),
        ("Cancel my subscription", "I want to unsubscribe"),
        ("I need a refund", "How do refunds work?"),
    ]
    diff_topic = [
        ("How do I reset my password?", "What are your business hours?"),
        ("Cancel my subscription", "I need a refund"),
    ]

    # Pre-compute embeddings
    same_sims = [cosine_similarity(pseudo_embed(a), pseudo_embed(b)) for a, b in same_topic]
    diff_sims = [cosine_similarity(pseudo_embed(a), pseudo_embed(b)) for a, b in diff_topic]

    print(f"\n  {'Threshold':>10}  {'TP/Same':>10}  {'FP/Diff':>10}  {'Precision':>12}  {'Recall':>10}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}")

    for tau in [0.85, 0.88, 0.90, 0.92, 0.93, 0.95, 0.97]:
        tp = sum(1 for s in same_sims if s >= tau)
        fp = sum(1 for s in diff_sims if s >= tau)
        tn = sum(1 for s in diff_sims if s <  tau)
        fn = sum(1 for s in same_sims if s <  tau)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        marker = " ◄ recommended" if tau == 0.93 else ""
        print(f"  {tau:>10.2f}  {tp:>10}  {fp:>10}  {precision:>11.1%}  {recall:>10.1%}{marker}")

    print(f"\n  Recommended range: τ ∈ [0.93, 0.97] for precision-recall balance")

# ── 8.7  FAISS flat index simulation ─────────────────────────────────────────

def demo_faiss_index() -> None:
    section("Vector Store — FlatIndex Simulation (FAISS-style)")

    N   = 5000
    rng = np.random.default_rng(42)

    index  = FlatIndex(dim=EMBED_DIM)
    topics = ["password", "email", "hours", "cancel", "refund", "shipping",
              "account", "install"]

    for i in range(N):
        topic = random.choice(topics)
        text  = f"{topic} question {i}"
        vec   = pseudo_embed(text)
        index.add(vec, text)

    query_text = "How do I reset my forgotten password?"
    q_emb      = pseudo_embed(query_text)

    t0      = time.perf_counter()
    results = index.search(q_emb, k=3)
    lat_ms  = (time.perf_counter() - t0) * 1000

    print(f"\n  Index size: {len(index):,} entries  |  Query latency: {lat_ms:.2f} ms")
    print(f"\n  Top-3 results for: \"{query_text}\"")
    for rank, (score, meta) in enumerate(results, 1):
        print(f"    [{rank}] sim={score:.4f}  {meta}")

    best_sim = results[0][0] if results else 0.0
    assert best_sim >= 0.85, f"Top result similarity {best_sim:.4f} too low"
    print(f"\n  [ASSERT] Top result sim ≥ 0.85: {best_sim:.4f} ✓")

# ── 8.8  Invalidation strategies ─────────────────────────────────────────────

def demo_invalidation() -> None:
    section("Cache Invalidation Strategies")

    cache = SemanticCache(threshold=THRESHOLD, max_entries=200, ttl_seconds=9999)

    # Populate with two models and two topic areas
    for i in range(5):
        cache.put(f"llama3 question {i}", f"resp {i}",
                  model="llama-3.1-8b", input_tokens=50, output_tokens=30)
    for i in range(5):
        cache.put(f"pricing question {i}", f"price resp {i}",
                  model="gpt-4o", input_tokens=50, output_tokens=30)
    for i in range(3):
        cache.put(f"docs section intro {i}", f"intro resp {i}",
                  model="gpt-4o", input_tokens=50, output_tokens=30)

    print(f"\n  Cache size before invalidation: {cache.size()}")

    # Strategy A: model-version flush (on new model deployment)
    flushed_a = cache.flush_model("llama-3.1-8b")
    print(f"  Model flush (llama-3.1-8b):     {flushed_a} entries removed")

    # Strategy B: prefix invalidation (on documentation update)
    flushed_b = cache.invalidate_prefix("docs section")
    print(f"  Prefix invalidation (docs):     {flushed_b} entries removed")

    print(f"  Cache size after invalidation:  {cache.size()}")

    assert cache.size() == 5, f"Expected 5 entries remaining, got {cache.size()}"
    print(f"\n  [ASSERT] Correct entries remaining after invalidation: ✓")

# ── 8.9  Quality validation ───────────────────────────────────────────────────

def demo_quality_validation() -> None:
    section("Cache Quality Validation")

    # similarity_threshold=0.08 (Jaccard after punctuation removal; paraphrases land ~0.10–0.35)
    validator = CacheQualityValidator(sample_rate=0.1, similarity_threshold=0.08)

    # Simulate serving cached responses and comparing with fresh ones
    pairs = [
        # (query, cached_response, fresh_response)
        # Pair 1: valid paraphrase — high Jaccard (PASS expected)
        ("How do I reset my password?",
         "To reset your password go to Settings then Security and click Reset Password.",
         "You can reset your password by visiting Settings Security and selecting Reset Password."),
        # Pair 2: valid paraphrase — moderate Jaccard (PASS expected)
        ("What are your business hours?",
         "We are open Monday to Friday from 9am to 6pm for support.",
         "Our support team is available Monday to Friday from 9am to 6pm."),
        # Pair 3: simulated quality degradation — unrelated response (FAIL expected)
        ("Cancel my subscription",
         "Visit Account then Billing and click Cancel to end your subscription.",
         "This product has been discontinued and is no longer available for purchase."),
    ]

    print(f"\n  {'Query':<40} {'Cached sim':>12}  {'OK?':>5}")
    print(f"  {'─'*40}  {'─'*12}  {'─'*5}")
    for query, cached, fresh in pairs:
        ok = validator.check(query, cached, fresh)
        sim = validator._response_similarity(cached, fresh)
        mark = "✓" if ok else "✗"
        print(f"  {query:<40} {sim:>12.3f}  [{mark}]")

    print()
    print(validator.report())

    acc = validator.accuracy()
    print(f"\n  Accuracy: {acc:.1%}")
    # 2/3 pairs should pass (the 3rd is a simulated bad cache hit)
    assert acc >= 0.60, f"Quality accuracy {acc:.1%} unexpectedly low"
    print(f"\n  [ASSERT] Quality validation operational: ✓")

# ── 8.10  Partitioned cache ───────────────────────────────────────────────────

def demo_partitioned_cache() -> None:
    section("Partitioned Cache (model × system-prompt × temperature)")

    # Two caches: one for deterministic (temp=0) and one for creative (temp=0.8)
    cache_det  = SemanticCache(threshold=THRESHOLD, max_entries=100, ttl_seconds=9999)
    cache_cre  = SemanticCache(threshold=THRESHOLD, max_entries=100, ttl_seconds=9999)

    prompt = "Write a one-sentence summary of the French Revolution."

    cache_det.put(prompt, "The French Revolution (1789–1799) ended absolute monarchy.",
                  model="gpt-4o", temperature=0.0,
                  input_tokens=40, output_tokens=12)
    cache_cre.put(prompt, "In a thunderclap of history, France shattered its chains.",
                  model="gpt-4o", temperature=0.8,
                  input_tokens=40, output_tokens=12)

    r_det = cache_det.get(prompt, model="gpt-4o", temperature=0.0)
    r_cre = cache_cre.get(prompt, model="gpt-4o", temperature=0.8)
    r_cross = cache_det.get(prompt, model="gpt-4o", temperature=0.8)

    print(f"\n  Deterministic cache hit:  {'✓' if r_det  else '✗'}  \"{r_det[:50] if r_det else 'MISS'}\"")
    print(f"  Creative cache hit:       {'✓' if r_cre  else '✗'}  \"{r_cre[:50] if r_cre else 'MISS'}\"")
    print(f"  Cross-temp lookup:        {'MISS ✓ (isolated)' if r_cross is None else 'HIT (BUG — temperature partitions crossed)'}")

    assert r_det  is not None, "Deterministic cache miss"
    assert r_cre  is not None, "Creative cache miss"
    assert r_cross is None,    "Temperature partitions should not cross"
    print(f"\n  [ASSERT] Temperature partitioning isolates responses: ✓")

# ──────────────────────────────────────────────────────────────────────────────
# §9  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  Chapter 30 — Semantic Caching Demo (Python)\n{bar}")

    demo_similarity_verification()
    demo_faq_workload()
    demo_break_even()
    demo_lru_eviction()
    demo_ttl_expiry()
    demo_threshold_sweep()
    demo_faiss_index()
    demo_invalidation()
    demo_quality_validation()
    demo_partitioned_cache()

    print(f"\n{bar}\n  All demos complete.\n{bar}\n")

if __name__ == "__main__":
    main()

```

## C++ — `semantic_cache_demo.cpp`

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

// Simple deterministic "embedding": hash text into a fixed-dim float vector
// Keyword → topic-ID mapping for demo similarity
static int topic_id(const std::string& text) {
    std::string t = text;
    // lowercase
    for (char& c : t) c = static_cast<char>(std::tolower((unsigned char)c));
    if (t.find("password") != std::string::npos ||
        t.find("reset") != std::string::npos ||
        t.find("forgot") != std::string::npos)   return 1;
    if (t.find("email") != std::string::npos ||
        t.find("address") != std::string::npos)  return 2;
    if (t.find("hours") != std::string::npos ||
        t.find("open") != std::string::npos ||
        t.find("weekend") != std::string::npos ||
        t.find("support") != std::string::npos)  return 3;
    if (t.find("cancel") != std::string::npos ||
        t.find("unsubscribe") != std::string::npos ||
        t.find("membership") != std::string::npos) return 4;
    if (t.find("refund") != std::string::npos ||
        t.find("money") != std::string::npos)    return 5;
    return 0;
}

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
        "How do I reset my password?",          // topic 1: password
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
    assert(r1.empty()  == true);   // LRU entry evicted — no semantic match possible (distinct topic)
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

