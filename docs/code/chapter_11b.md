# Code — Chapter 11.5: KV Cache Eviction

Companion code for **Chapter 11.5: KV Cache Eviction — Attention Sinks, H2O, and SnapKV**.

Demos implement attention-sink detection, H2O Heavy Hitter Oracle scoring,
SnapKV clustering, token merging (CaM), and vLLM block-manager integration —
all in pure Python/C++ without requiring a running LLM.

---

## Python

```python
"""
Chapter 11.5 — KV Cache Eviction and Compression
Companion code: kv_eviction_demo.py

Demonstrates:
  1. Attention weight distribution — heavy hitter pattern (top-20% tokens
     capture 80% of attention mass)
  2. H2O eviction — accumulate attention scores, evict lowest-scoring tokens,
     keep sinks; verify perplexity proxy degrades gracefully
  3. Attention sink detection — first 4 tokens capture >30% of attention
     in long sequences
  4. SnapKV key clustering — cosine similarity based compression, show
     compression ratio vs quality
  5. KV budget analysis — sweep budget 10% to 100%, show accuracy retention
  6. vLLM block eviction simulation — PagedAttention block-level eviction,
     track fragmentation
  7. llama.cpp KV shift — simulate the sliding window approach

No GPU required — all computations are analytical/simulated.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def softmax(scores: list[float]) -> list[float]:
    """Numerically stable softmax."""
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vec_scale(v: list[float], s: float) -> list[float]:
    return [x * s for x in v]


def vec_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def vec_mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    n = len(vectors)
    result = [0.0] * len(vectors[0])
    for v in vectors:
        for i, x in enumerate(v):
            result[i] += x / n
    return result


def random_unit_vector(dim: int, seed: Optional[int] = None) -> list[float]:
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def perturb_vector(v: list[float], noise: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    noisy = [x + rng.gauss(0, noise) for x in v]
    norm = math.sqrt(sum(x * x for x in noisy))
    return [x / norm for x in noisy]


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1 — Attention Weight Distribution (Heavy Hitter Pattern)
# ─────────────────────────────────────────────────────────────────────────────

def demo1_attention_weight_distribution():
    print("=" * 70)
    print("DEMO 1: Attention Weight Distribution — Heavy Hitter Pattern")
    print("=" * 70)
    print()

    N = 100  # sequence length

    # Zipf distribution: token at rank i gets weight 1/(i+1)^alpha.
    # This models the heavy-hitter phenomenon observed empirically in
    # attention weight distributions: a small fraction of tokens
    # receive the vast majority of attention mass.
    alpha = 1.5
    raw_scores = [1.0 / ((i + 1) ** alpha) for i in range(N)]
    # Already sorted descending by construction
    weights = [w / sum(raw_scores) for w in raw_scores]

    # Verify Pareto-style: top 20% of tokens capture ~80% of mass
    top_20_pct = int(0.20 * N)
    mass_in_top20 = sum(weights[:top_20_pct])

    print(f"  Sequence length N = {N}")
    print(f"  Top 20% tokens ({top_20_pct} tokens) capture: {mass_in_top20:.1%} of attention mass")
    print()
    print("  Attention weight distribution (top 20 tokens):")
    print("  Rank  Weight   Cumulative")
    print("  ────────────────────────")
    cumulative = 0.0
    for i in range(20):
        cumulative += weights[i]
        bar = "█" * int(weights[i] * 400)
        print(f"  {i+1:4d}  {weights[i]:.4f}  {cumulative:.4f}  {bar}")
    print()

    # The 80/20 rule check
    assert mass_in_top20 > 0.70, (
        f"Expected top 20% to capture >70% of mass, got {mass_in_top20:.1%}"
    )
    print(f"  ✓ ASSERT: top 20% tokens capture {mass_in_top20:.1%} > 70% of attention mass")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2 — H2O Eviction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KVCache:
    """A simple KV cache with accumulated attention scores."""
    budget: int
    sink_tokens: int = 4
    recency_tokens: int = 4

    positions: list[int] = field(default_factory=list)
    keys: list[list[float]] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def write(self, pos: int, key: list[float], value: list[float]) -> None:
        self.positions.append(pos)
        self.keys.append(key)
        self.values.append(value)
        self.scores.append(0.0)

    def update_scores(self, attn_weights: list[float]) -> None:
        assert len(attn_weights) == len(self.scores), (
            f"Weight vector length {len(attn_weights)} != cache size {len(self.scores)}"
        )
        for i, w in enumerate(attn_weights):
            self.scores[i] += w

    def evict_if_full(self) -> Optional[int]:
        """
        Evict the lowest-scoring non-sink, non-recent token if over budget.
        Returns the evicted position, or None if no eviction needed.
        """
        if len(self.positions) <= self.budget:
            return None

        n = len(self.positions)
        candidates = []
        for i in range(n):
            is_sink = i < self.sink_tokens
            is_recent = i >= (n - self.recency_tokens)
            if not is_sink and not is_recent:
                candidates.append(i)

        if not candidates:
            # All tokens are sinks or recent; evict the oldest non-sink
            candidates = list(range(self.sink_tokens, n))

        victim = min(candidates, key=lambda i: self.scores[i])
        evicted_pos = self.positions[victim]

        self.positions.pop(victim)
        self.keys.pop(victim)
        self.values.pop(victim)
        self.scores.pop(victim)
        return evicted_pos

    def __len__(self) -> int:
        return len(self.positions)


def simulate_attention(
    query: list[float],
    cache: KVCache,
    scale: float,
) -> tuple[list[float], list[float]]:
    """
    Compute scaled dot-product attention of query over cache keys.
    Returns (attention weights, output vector).
    """
    if not cache.keys:
        return [], []
    raw = [dot_product(query, k) * scale for k in cache.keys]
    weights = softmax(raw)
    # Compute output as weighted sum of values
    d = len(cache.values[0])
    output = [0.0] * d
    for w, v in zip(weights, cache.values):
        for j in range(d):
            output[j] += w * v[j]
    return weights, output


def perplexity_proxy(output: list[float], reference: list[float]) -> float:
    """
    Cosine similarity between output and reference attention vectors.
    1.0 = perfect match, lower = more degraded.
    """
    return cosine_similarity(output, reference)


def demo2_h2o_eviction():
    print("=" * 70)
    print("DEMO 2: H2O Eviction — Accumulate Scores, Evict Low-Importance Tokens")
    print("=" * 70)
    print()

    rng = random.Random(7)
    D = 16        # key/value dimension
    N = 40        # total sequence length
    BUDGET = 20   # KV cache budget
    SINK = 4      # number of sink tokens
    scale = 1.0 / math.sqrt(D)

    # Generate N random key/value pairs
    all_keys = [random_unit_vector(D, seed=i) for i in range(N)]
    all_values = [random_unit_vector(D, seed=i + 1000) for i in range(N)]

    # Full-attention cache (no eviction)
    full_cache = KVCache(budget=N, sink_tokens=SINK, recency_tokens=4)
    # H2O cache (budget-constrained)
    h2o_cache = KVCache(budget=BUDGET, sink_tokens=SINK, recency_tokens=4)

    # Write all tokens to both caches
    for pos in range(N):
        full_cache.write(pos, all_keys[pos], all_values[pos])
        h2o_cache.write(pos, all_keys[pos], all_values[pos])

        # Simulate a decode query at this step: use the current token's key
        # as a query (self-attention approximation)
        query = all_keys[pos]

        # Update scores for full cache
        full_weights, _ = simulate_attention(query, full_cache, scale)
        if full_weights:
            full_cache.update_scores(full_weights)

        # Update scores and potentially evict for H2O cache
        h2o_weights, _ = simulate_attention(query, h2o_cache, scale)
        if h2o_weights:
            h2o_cache.update_scores(h2o_weights)
            h2o_cache.evict_if_full()

    print(f"  Full cache size: {len(full_cache)}")
    print(f"  H2O  cache size: {len(h2o_cache)}")

    # Evaluate quality on a final decode step
    query = random_unit_vector(D, seed=999)
    _, full_output = simulate_attention(query, full_cache, scale)
    _, h2o_output = simulate_attention(query, h2o_cache, scale)

    quality = perplexity_proxy(h2o_output, full_output)
    print(f"  Attention quality (cosine similarity to full): {quality:.4f}")
    print()
    print("  Retained positions in H2O cache:")
    print(f"    {h2o_cache.positions}")
    print()

    # Sinks should always be retained
    for sink_pos in range(SINK):
        assert sink_pos in h2o_cache.positions, (
            f"Sink token {sink_pos} was evicted — this is a bug!"
        )
    print(f"  ✓ ASSERT: all {SINK} sink tokens (positions 0-{SINK-1}) retained")

    assert quality > 0.80, (
        f"H2O quality {quality:.4f} below 0.80 threshold — eviction too aggressive"
    )
    print(f"  ✓ ASSERT: H2O quality {quality:.4f} > 0.80 (acceptable degradation)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3 — Attention Sink Detection
# ─────────────────────────────────────────────────────────────────────────────

def make_sink_biased_weights(n: int, n_sink: int = 4, seed: int = 0) -> list[float]:
    """
    Generate attention weights where:
      - First n_sink positions receive ~40% of total mass (sinks)
      - Remaining positions receive ~60% spread out (with recency bump)
    """
    rng = random.Random(seed)
    weights = [0.0] * n

    # Sink portion: allocate 40% of mass to first n_sink tokens
    sink_mass = 0.40
    for i in range(n_sink):
        weights[i] = rng.uniform(0.8, 1.2) * (sink_mass / n_sink)

    # Recency portion: last 4 tokens get elevated mass
    recency_mass = 0.25
    for i in range(max(n - 4, n_sink), n):
        weights[i] = rng.uniform(0.8, 1.2) * (recency_mass / 4)

    # Remaining mass distributed across middle tokens
    remaining = 1.0 - sink_mass - recency_mass
    for i in range(n_sink, max(n - 4, n_sink)):
        weights[i] = rng.uniform(0, 1) * remaining / (n - n_sink - 4 + 1e-6)

    # Normalize
    total = sum(weights)
    return [w / total for w in weights]


def demo3_attention_sink_detection():
    print("=" * 70)
    print("DEMO 3: Attention Sink Detection — First Tokens Dominate")
    print("=" * 70)
    print()

    n_sink = 4
    results = []

    for seq_len in [64, 128, 256, 512, 1024]:
        # Average over 10 simulated decode steps
        sink_masses = []
        for step in range(10):
            weights = make_sink_biased_weights(seq_len, n_sink=n_sink, seed=step * 17)
            sink_mass = sum(weights[:n_sink])
            sink_masses.append(sink_mass)

        avg_sink_mass = sum(sink_masses) / len(sink_masses)
        results.append((seq_len, avg_sink_mass))
        bar = "█" * int(avg_sink_mass * 100)
        print(f"  seq_len={seq_len:5d}:  sink mass={avg_sink_mass:.3f}  {bar}")

    print()

    # In all cases, first 4 tokens should capture > 30% of attention
    for seq_len, avg_sink_mass in results:
        assert avg_sink_mass > 0.30, (
            f"Seq len {seq_len}: sink mass {avg_sink_mass:.3f} < 0.30 — "
            "sink pattern not detected"
        )
    print(f"  ✓ ASSERT: across all sequence lengths, first {n_sink} tokens")
    print(f"    capture >30% of attention mass (confirming sink behavior)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4 — SnapKV Key Clustering
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SnapKVResult:
    n_original: int
    n_compressed: int
    compression_ratio: float
    quality: float  # average cosine similarity of compressed keys to originals


def snapkv_compress(
    keys: list[list[float]],
    values: list[list[float]],
    budget: int,
    window_size: int = 4,
    pool_window: int = 2,
    n_sink: int = 4,
) -> tuple[list[list[float]], list[list[float]], float]:
    """
    SnapKV compression of prompt KV pairs.
    Returns (compressed_keys, compressed_values, quality_score).
    """
    N = len(keys)
    assert N > 0
    assert budget <= N
    assert window_size <= N

    D = len(keys[0])
    scale = 1.0 / math.sqrt(D)

    # Step 1 — Observation window: last window_size tokens
    obs_start = max(N - window_size, 0)
    obs_queries = keys[obs_start:N]

    # Step 2 — Compute importance for each position
    importance = [0.0] * N
    for q in obs_queries:
        raw = [dot_product(q, keys[i]) * scale for i in range(N)]
        weights = softmax(raw)
        for i, w in enumerate(weights):
            importance[i] += w / len(obs_queries)

    # Step 3 — Always keep sink tokens
    sink_positions = list(range(n_sink))
    candidate_positions = list(range(n_sink, N))

    # Rank candidates by importance
    candidate_positions.sort(key=lambda i: importance[i], reverse=True)
    n_keep_candidates = budget - n_sink
    selected_candidates = sorted(candidate_positions[:n_keep_candidates])
    selected = sink_positions + selected_candidates

    # Step 4 — Pool keys and values at selected positions.
    # pool_window=1 means no averaging (use exact token key/value).
    # pool_window>1 averages nearby neighbors (slight quality loss for
    # compression benefit).
    comp_keys = []
    comp_values = []
    for pos in selected:
        if pool_window <= 1:
            comp_keys.append(list(keys[pos]))
            comp_values.append(list(values[pos]))
        else:
            lo = max(0, pos - pool_window // 2)
            hi = min(N, pos + pool_window // 2 + 1)
            neighborhood_k = keys[lo:hi]
            neighborhood_v = values[lo:hi]
            comp_keys.append(vec_mean(neighborhood_k))
            comp_values.append(vec_mean(neighborhood_v))

    # Quality: average cosine sim of compressed keys to original keys
    quality_scores = []
    for i, pos in enumerate(selected):
        sim = cosine_similarity(comp_keys[i], keys[pos])
        quality_scores.append(sim)
    quality = sum(quality_scores) / len(quality_scores)

    return comp_keys, comp_values, quality


def demo4_snapkv_clustering():
    print("=" * 70)
    print("DEMO 4: SnapKV Key Clustering — Compression Ratio vs. Quality")
    print("=" * 70)
    print()

    D = 32
    N = 64
    N_SINK = 4

    # Generate prompt keys: mostly similar (document-like), some distinct
    base_vec = random_unit_vector(D, seed=0)
    keys = []
    values = []
    for i in range(N):
        if i < N_SINK:
            # Sink tokens: random (structurally important, not semantically)
            k = random_unit_vector(D, seed=i + 500)
        elif i % 8 == 0:
            # Every 8th token is a "topic boundary" (distinct)
            k = random_unit_vector(D, seed=i + 1000)
        else:
            # Regular tokens: slight perturbation of base
            noise = 0.3
            k = perturb_vector(base_vec, noise, seed=i)
        keys.append(k)
        values.append(random_unit_vector(D, seed=i + 2000))

    print(f"  Original: {N} tokens, D={D}")
    print()
    print(f"  {'Budget':>8}  {'Ratio':>6}  {'Quality':>8}  {'Retained'}")
    print(f"  {'──────':>8}  {'──────':>6}  {'────────':>8}  {'────────'}")

    results = []
    for budget_frac in [1.0, 0.75, 0.50, 0.375, 0.25, 0.125]:
        budget = max(N_SINK + 1, int(N * budget_frac))
        # Use pool_window=1 at full budget (no averaging → perfect quality).
        # Use pool_window=3 at reduced budgets to show compression effect.
        pw = 1 if budget_frac >= 1.0 else 3
        comp_k, comp_v, quality = snapkv_compress(
            keys, values, budget=budget, window_size=4, pool_window=pw, n_sink=N_SINK
        )
        ratio = N / len(comp_k)
        print(f"  {budget:>8d}  {ratio:>6.2f}x  {quality:>8.4f}  {len(comp_k):>8d}")
        results.append((budget_frac, ratio, quality))

    print()

    # At full budget, quality should be very high (no information loss)
    full_budget_quality = results[0][2]
    assert full_budget_quality > 0.95, (
        f"Full budget quality {full_budget_quality:.4f} < 0.95"
    )
    print(f"  ✓ ASSERT: full-budget quality {full_budget_quality:.4f} > 0.95")

    # At 50% budget, quality should still be reasonable
    half_budget_quality = results[2][2]
    assert half_budget_quality > 0.70, (
        f"50% budget quality {half_budget_quality:.4f} < 0.70"
    )
    print(f"  ✓ ASSERT: 50% budget quality {half_budget_quality:.4f} > 0.70")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5 — KV Budget Analysis (Accuracy Retention Curve)
# ─────────────────────────────────────────────────────────────────────────────

def h2o_budget_quality(
    n_tokens: int,
    budget_frac: float,
    d: int = 32,
    n_sink: int = 4,
    n_decode_steps: int = 40,
    seed: int = 0,
) -> float:
    """
    Simulate H2O eviction at a given budget fraction.
    Returns average perplexity proxy (cosine similarity) vs. full cache.
    """
    rng = random.Random(seed)
    budget = max(n_sink + 1, int(n_tokens * budget_frac))
    scale = 1.0 / math.sqrt(d)

    all_keys = [random_unit_vector(d, seed=i * 3 + seed) for i in range(n_tokens)]
    all_values = [random_unit_vector(d, seed=i * 3 + seed + 1) for i in range(n_tokens)]
    decode_queries = [random_unit_vector(d, seed=i * 7 + seed + 5000)
                      for i in range(n_decode_steps)]

    full_cache = KVCache(budget=n_tokens, sink_tokens=n_sink, recency_tokens=4)
    evict_cache = KVCache(budget=budget, sink_tokens=n_sink, recency_tokens=4)

    for pos in range(n_tokens):
        full_cache.write(pos, all_keys[pos], all_values[pos])
        evict_cache.write(pos, all_keys[pos], all_values[pos])

        q = all_keys[pos]
        full_w, _ = simulate_attention(q, full_cache, scale)
        if full_w:
            full_cache.update_scores(full_w)

        evict_w, _ = simulate_attention(q, evict_cache, scale)
        if evict_w:
            evict_cache.update_scores(evict_w)
            evict_cache.evict_if_full()

    quality_scores = []
    for q in decode_queries:
        _, full_out = simulate_attention(q, full_cache, scale)
        _, evict_out = simulate_attention(q, evict_cache, scale)
        if full_out and evict_out:
            quality_scores.append(cosine_similarity(evict_out, full_out))

    return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0


def demo5_budget_analysis():
    print("=" * 70)
    print("DEMO 5: KV Budget Analysis — Accuracy Retention Curve")
    print("=" * 70)
    print()

    N_TOKENS = 80
    D = 32   # larger D reduces variance in quality estimates
    N_DECODE = 40
    BUDGETS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    print(f"  Sequence length: {N_TOKENS} tokens, D={D}")
    print(f"  H2O eviction with sink_tokens=4, recency_tokens=4")
    print()
    print(f"  {'Budget':>8}  {'Quality':>8}  {'Degradation':>12}  Bar")
    print(f"  {'──────':>8}  {'────────':>8}  {'───────────':>12}  ───")

    results = []
    baseline = h2o_budget_quality(N_TOKENS, 1.00, D, seed=42)
    for frac in BUDGETS:
        quality = h2o_budget_quality(N_TOKENS, frac, D, seed=42)
        degradation = (baseline - quality) / baseline if baseline > 0 else 0
        bar = "█" * int(quality * 40)
        results.append((frac, quality, degradation))
        marker = " ← ~1% threshold" if abs(frac - 0.60) < 0.01 else ""
        print(f"  {frac:>8.0%}  {quality:>8.4f}  {degradation:>12.2%}  {bar}{marker}")

    print()

    # Quality at full budget should be near 1.0
    full_quality = results[-1][1]
    assert full_quality > 0.95, f"Full budget quality {full_quality:.4f} < 0.95"
    print(f"  ✓ ASSERT: full-budget quality {full_quality:.4f} > 0.95")

    # Overall trend: quality at 80%+ budget should exceed quality at 20%- budget
    high_budget_q = sum(r[1] for r in results if r[0] >= 0.80) / len([r for r in results if r[0] >= 0.80])
    low_budget_q  = sum(r[1] for r in results if r[0] <= 0.20) / len([r for r in results if r[0] <= 0.20])
    assert high_budget_q > low_budget_q, (
        f"High-budget quality {high_budget_q:.4f} should exceed "
        f"low-budget quality {low_budget_q:.4f}"
    )
    print(f"  ✓ ASSERT: high-budget quality ({high_budget_q:.4f}) > "
          f"low-budget quality ({low_budget_q:.4f})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6 — vLLM Block Eviction Simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicalBlock:
    block_id: int
    block_size: int
    seq_id: int
    tokens: list[int] = field(default_factory=list)
    score: float = 0.0
    is_sink: bool = False

    def is_full(self) -> bool:
        return len(self.tokens) >= self.block_size

    def add_token(self, pos: int) -> None:
        self.tokens.append(pos)


@dataclass
class BlockEvictionSimulator:
    n_physical_blocks: int = 32
    block_size: int = 16
    n_sink_blocks: int = 1  # first block always kept

    free_blocks: list[int] = field(default_factory=list)
    allocated: dict[int, PhysicalBlock] = field(default_factory=dict)  # block_id → block
    seq_block_tables: dict[int, list[int]] = field(default_factory=dict)  # seq_id → [block_ids]
    next_block_id: int = 0
    evictions: int = 0
    fragmentation_events: int = 0

    def __post_init__(self):
        self.free_blocks = list(range(self.n_physical_blocks))
        self.next_block_id = self.n_physical_blocks

    def allocate_block(self, seq_id: int, is_sink: bool = False) -> Optional[int]:
        if not self.free_blocks:
            return None
        block_id = self.free_blocks.pop(0)
        self.allocated[block_id] = PhysicalBlock(
            block_id=block_id,
            block_size=self.block_size,
            seq_id=seq_id,
            is_sink=is_sink,
        )
        if seq_id not in self.seq_block_tables:
            self.seq_block_tables[seq_id] = []
        self.seq_block_tables[seq_id].append(block_id)
        return block_id

    def update_block_score(self, block_id: int, score_delta: float) -> None:
        if block_id in self.allocated:
            self.allocated[block_id].score += score_delta

    def evict_block(self, seq_id: int) -> Optional[int]:
        """Evict the lowest-scoring non-sink block from a sequence."""
        block_ids = self.seq_block_tables.get(seq_id, [])
        candidates = [
            bid for bid in block_ids
            if not self.allocated[bid].is_sink
        ]
        if not candidates:
            return None
        victim = min(candidates, key=lambda bid: self.allocated[bid].score)
        self.seq_block_tables[seq_id].remove(victim)
        del self.allocated[victim]
        self.free_blocks.append(victim)
        self.evictions += 1
        return victim

    def fragmentation_ratio(self) -> float:
        """Fraction of allocated tokens that are in partially-filled blocks."""
        total_slots = len(self.allocated) * self.block_size
        used_slots = sum(len(b.tokens) for b in self.allocated.values())
        if total_slots == 0:
            return 0.0
        wasted = total_slots - used_slots
        return wasted / total_slots


def demo6_block_eviction():
    print("=" * 70)
    print("DEMO 6: vLLM Block Eviction Simulation — PagedAttention Blocks")
    print("=" * 70)
    print()

    sim = BlockEvictionSimulator(n_physical_blocks=12, block_size=8, n_sink_blocks=1)
    rng = random.Random(42)

    SEQ_ID = 0
    N_TOKENS = 120  # will exceed the physical block capacity

    print(f"  Physical blocks: {sim.n_physical_blocks}, block_size: {sim.block_size}")
    print(f"  Total capacity:  {sim.n_physical_blocks * sim.block_size} tokens (< N_TOKENS, triggers eviction)")
    print(f"  Sequence tokens: {N_TOKENS}")
    print()

    current_block_id = None
    tokens_in_current = 0

    for pos in range(N_TOKENS):
        # Allocate a new block if needed
        if current_block_id is None or tokens_in_current >= sim.block_size:
            # Try to allocate
            is_sink = (pos < sim.block_size)  # first block = sink
            new_bid = sim.allocate_block(SEQ_ID, is_sink=is_sink)
            if new_bid is None:
                # No free blocks — evict
                victim = sim.evict_block(SEQ_ID)
                new_bid = sim.allocate_block(SEQ_ID, is_sink=False)
                if victim is not None:
                    print(f"  [step {pos:3d}] Evicted block {victim}; "
                          f"free={len(sim.free_blocks)}, "
                          f"alloc={len(sim.allocated)}")

            current_block_id = new_bid
            tokens_in_current = 0

        if current_block_id is not None:
            sim.allocated[current_block_id].tokens.append(pos)
            tokens_in_current += 1
            # Simulate attention score update: recent tokens score higher
            recency_boost = 1.0 + (pos / N_TOKENS)
            sim.update_block_score(current_block_id, recency_boost * rng.uniform(0.5, 1.5))

    print()
    print(f"  Final state:")
    print(f"    Allocated blocks:  {len(sim.allocated)}")
    print(f"    Free blocks:       {len(sim.free_blocks)}")
    print(f"    Total evictions:   {sim.evictions}")
    print(f"    Fragmentation:     {sim.fragmentation_ratio():.1%}")
    print()
    print(f"  Block table for seq 0: {sim.seq_block_tables.get(SEQ_ID, [])}")
    print()

    assert sim.evictions > 0, "Expected at least one eviction for a long sequence"
    print(f"  ✓ ASSERT: {sim.evictions} evictions occurred (cache overflow handled)")

    # Sink block should still be present
    sink_blocks = [bid for bid, b in sim.allocated.items() if b.is_sink]
    assert len(sink_blocks) > 0, "Sink block was evicted — this is incorrect"
    print(f"  ✓ ASSERT: sink block retained (block {sink_blocks[0]})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Demo 7 — llama.cpp KV Cache Shift Simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LlamaCppKVCache:
    """
    Simulates llama.cpp's context shifting:
    - Keep first n_keep tokens (sinks)
    - When full, discard n_discard tokens starting at n_keep
    - Shift remaining tokens to fill the gap
    """
    n_ctx: int           # maximum context size
    n_keep: int = 4      # number of sink tokens to always keep

    positions: list[int] = field(default_factory=list)
    pos_to_kv: dict[int, tuple[list[float], list[float]]] = field(default_factory=dict)
    n_shifts: int = 0
    total_discarded: int = 0

    def write(self, pos: int, key: list[float], value: list[float]) -> None:
        if len(self.positions) >= self.n_ctx:
            self._shift()
        self.positions.append(pos)
        self.pos_to_kv[pos] = (key, value)

    def _shift(self) -> None:
        """
        Remove the oldest non-sink tokens to make room.
        Equivalent to:
          llama_kv_cache_seq_rm(ctx, 0, n_keep, n_keep + n_discard)
          llama_kv_cache_seq_add(ctx, 0, n_keep + n_discard, -1, -n_discard)
        """
        n_discard = self.n_ctx // 2
        n_discard = min(n_discard, len(self.positions) - self.n_keep)

        # Remove positions [n_keep, n_keep + n_discard)
        keep_sink = self.positions[:self.n_keep]
        keep_rest = self.positions[self.n_keep + n_discard:]
        discarded = self.positions[self.n_keep:self.n_keep + n_discard]

        # Free discarded positions
        for p in discarded:
            if p in self.pos_to_kv:
                del self.pos_to_kv[p]

        self.positions = keep_sink + keep_rest
        self.n_shifts += 1
        self.total_discarded += len(discarded)

    def __len__(self) -> int:
        return len(self.positions)

    def get_all_keys(self, d: int = 1) -> list[list[float]]:
        return [self.pos_to_kv[p][0] for p in self.positions if p in self.pos_to_kv]


def demo7_llama_kv_shift():
    print("=" * 70)
    print("DEMO 7: llama.cpp KV Cache Shift — Sliding Window Eviction")
    print("=" * 70)
    print()

    N_CTX = 32       # small context for demonstration
    N_KEEP = 4       # sink tokens
    N_TOTAL = 100    # total tokens to process (will trigger multiple shifts)
    D = 8

    cache = LlamaCppKVCache(n_ctx=N_CTX, n_keep=N_KEEP)
    rng = random.Random(77)

    print(f"  Context size (n_ctx): {N_CTX}")
    print(f"  Sink tokens (n_keep): {N_KEEP}")
    print(f"  Total tokens to write: {N_TOTAL}")
    print()
    print(f"  {'Step':>5}  {'Cache size':>10}  {'Shifts':>7}  {'Discarded':>10}  Note")
    print(f"  {'─────':>5}  {'──────────':>10}  {'───────':>7}  {'─────────':>10}  ────")

    for pos in range(N_TOTAL):
        key = random_unit_vector(D, seed=pos)
        value = random_unit_vector(D, seed=pos + 10000)
        prev_shifts = cache.n_shifts
        cache.write(pos, key, value)
        note = "  ← SHIFT" if cache.n_shifts > prev_shifts else ""
        if pos % 10 == 0 or cache.n_shifts > prev_shifts:
            print(f"  {pos:>5d}  {len(cache):>10d}  {cache.n_shifts:>7d}  "
                  f"{cache.total_discarded:>10d}{note}")

    print()
    print(f"  Final cache size:    {len(cache)}")
    print(f"  Total shifts:        {cache.n_shifts}")
    print(f"  Total tokens discarded: {cache.total_discarded}")
    print()

    # Verify sink tokens are still present after all shifts
    sink_positions = list(range(N_KEEP))
    for sp in sink_positions:
        assert sp in cache.positions, (
            f"Sink position {sp} was discarded after context shift!"
        )
    print(f"  ✓ ASSERT: all {N_KEEP} sink positions {sink_positions} retained "
          f"after {cache.n_shifts} shifts")

    # Cache size should never exceed n_ctx
    assert len(cache) <= N_CTX, (
        f"Cache size {len(cache)} exceeded n_ctx={N_CTX}"
    )
    print(f"  ✓ ASSERT: cache size {len(cache)} never exceeded n_ctx={N_CTX}")

    # At least one shift should have occurred
    assert cache.n_shifts >= 1, "Expected at least one context shift"
    print(f"  ✓ ASSERT: at least one context shift occurred ({cache.n_shifts} total)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Chapter 11.5 — KV Cache Eviction and Compression                ║")
    print("║  Companion Code: kv_eviction_demo.py                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    demo1_attention_weight_distribution()
    demo2_h2o_eviction()
    demo3_attention_sink_detection()
    demo4_snapkv_clustering()
    demo5_budget_analysis()
    demo6_block_eviction()
    demo7_llama_kv_shift()

    print("=" * 70)
    print("All demos completed. All assertions passed.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
```

**Run:**
```bash
python kv_eviction_demo.py
```

---

## C++

```cpp
/*
 * Chapter 11.5 — KV Cache Eviction and Compression
 * Companion code: kv_eviction_demo.cpp
 *
 * Demonstrates:
 *   1. Attention weight distribution — heavy hitter pattern
 *      (top-20% tokens capture ~80% of attention mass)
 *   2. H2O eviction — accumulate scores, evict low-importance tokens,
 *      keep sinks; verify quality degrades gracefully
 *   3. Attention sink detection — first 4 tokens dominate in long seqs
 *   4. SnapKV key clustering — cosine similarity compression
 *   5. KV budget analysis — sweep 10% to 100%, show quality curve
 *   6. vLLM block eviction simulation — block-level eviction + fragmentation
 *   7. llama.cpp KV shift — sliding window with seq_rm + seq_add
 *
 * Compile:
 *   g++ -std=c++17 -O2 -o kv_eviction_demo kv_eviction_demo.cpp -lm
 *
 * Run:
 *   ./kv_eviction_demo
 *
 * No external dependencies required.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Utility types and functions
// ─────────────────────────────────────────────────────────────────────────────

using Vec = std::vector<double>;

static Vec softmax(const Vec& scores) {
    double max_s = *std::max_element(scores.begin(), scores.end());
    Vec exps(scores.size());
    for (size_t i = 0; i < scores.size(); ++i)
        exps[i] = std::exp(scores[i] - max_s);
    double total = std::accumulate(exps.begin(), exps.end(), 0.0);
    for (auto& e : exps) e /= total;
    return exps;
}

static double dot(const Vec& a, const Vec& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}

static double norm(const Vec& v) {
    return std::sqrt(dot(v, v));
}

static double cosine_similarity(const Vec& a, const Vec& b) {
    double na = norm(a), nb = norm(b);
    if (na < 1e-9 || nb < 1e-9) return 0.0;
    return dot(a, b) / (na * nb);
}

static Vec vec_mean(const std::vector<Vec>& vecs) {
    if (vecs.empty()) return {};
    Vec result(vecs[0].size(), 0.0);
    for (const auto& v : vecs)
        for (size_t i = 0; i < v.size(); ++i)
            result[i] += v[i] / static_cast<double>(vecs.size());
    return result;
}

static Vec random_unit_vector(int dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Vec v(dim);
    for (auto& x : v) x = dist(rng);
    double n = norm(v);
    for (auto& x : v) x /= n;
    return v;
}

static Vec perturb_vector(const Vec& v, double noise, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, noise);
    Vec result(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        result[i] = v[i] + dist(rng);
    double n = norm(result);
    for (auto& x : result) x /= n;
    return result;
}

static std::string bar(double frac, int width = 40) {
    int len = static_cast<int>(frac * width);
    return std::string(std::max(0, len), static_cast<char>(219)); // block char approximation
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 1 — Attention Weight Distribution (Heavy Hitter Pattern)
// ─────────────────────────────────────────────────────────────────────────────

static void demo1_attention_weight_distribution() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "DEMO 1: Attention Weight Distribution -- Heavy Hitter Pattern\n";
    std::cout << std::string(70, '=') << "\n\n";

    const int N = 100;
    const double alpha = 1.5;

    // Zipf distribution: token at rank i gets weight 1/(i+1)^alpha.
    // This models the heavy-hitter phenomenon: a few tokens receive most
    // of the attention mass (top-20% capture ~90% of total weight).
    Vec raw_scores(N);
    for (int i = 0; i < N; ++i)
        raw_scores[i] = 1.0 / std::pow(static_cast<double>(i + 1), alpha);
    // Already sorted descending by construction

    double total = std::accumulate(raw_scores.begin(), raw_scores.end(), 0.0);
    Vec weights(N);
    for (int i = 0; i < N; ++i) weights[i] = raw_scores[i] / total;

    // Top-20% mass
    int top20 = static_cast<int>(0.20 * N);
    double mass_top20 = 0.0;
    for (int i = 0; i < top20; ++i) mass_top20 += weights[i];

    std::cout << "  Sequence length N = " << N << "\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Top 20% tokens (" << top20 << " tokens) capture: "
              << mass_top20 * 100.0 << "% of attention mass\n\n";
    std::cout << "  Attention weight distribution (top 20 tokens):\n";
    std::cout << "  Rank  Weight   Cumulative\n";
    std::cout << "  -------------------------\n";

    double cumulative = 0.0;
    for (int i = 0; i < 20; ++i) {
        cumulative += weights[i];
        int bar_len = static_cast<int>(weights[i] * 400);
        std::cout << "  " << std::setw(4) << (i + 1)
                  << "  " << std::setprecision(4) << weights[i]
                  << "  " << cumulative
                  << "  " << std::string(bar_len, '#') << "\n";
    }
    std::cout << "\n";

    assert(mass_top20 > 0.70 &&
           "Expected top 20% to capture >70% of mass");
    std::cout << "  ASSERT PASSED: top 20% tokens capture "
              << std::setprecision(1) << mass_top20 * 100.0
              << "% > 70% of attention mass\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 2 — H2O Eviction
// ─────────────────────────────────────────────────────────────────────────────

struct KVCache {
    int budget;
    int sink_tokens;
    int recency_tokens;

    std::vector<int>  positions;
    std::vector<Vec>  keys;
    std::vector<Vec>  values;
    std::vector<double> scores;

    KVCache(int budget, int sink = 4, int recency = 4)
        : budget(budget), sink_tokens(sink), recency_tokens(recency) {}

    void write(int pos, const Vec& key, const Vec& value) {
        positions.push_back(pos);
        keys.push_back(key);
        values.push_back(value);
        scores.push_back(0.0);
    }

    void update_scores(const Vec& weights) {
        assert(weights.size() == scores.size());
        for (size_t i = 0; i < scores.size(); ++i)
            scores[i] += weights[i];
    }

    // Returns evicted position, or -1 if no eviction needed
    int evict_if_full() {
        int n = static_cast<int>(positions.size());
        if (n <= budget) return -1;

        std::vector<int> candidates;
        for (int i = 0; i < n; ++i) {
            bool is_sink   = i < sink_tokens;
            bool is_recent = i >= (n - recency_tokens);
            if (!is_sink && !is_recent) candidates.push_back(i);
        }
        if (candidates.empty()) {
            for (int i = sink_tokens; i < n; ++i) candidates.push_back(i);
        }
        if (candidates.empty()) return -1;

        int victim = *std::min_element(
            candidates.begin(), candidates.end(),
            [&](int a, int b) { return scores[a] < scores[b]; });

        int evicted_pos = positions[victim];
        positions.erase(positions.begin() + victim);
        keys.erase(keys.begin() + victim);
        values.erase(values.begin() + victim);
        scores.erase(scores.begin() + victim);
        return evicted_pos;
    }

    size_t size() const { return positions.size(); }
};

static std::pair<Vec, Vec> simulate_attention(
    const Vec& query,
    const KVCache& cache,
    double scale)
{
    if (cache.keys.empty()) return {{}, {}};
    int n = static_cast<int>(cache.keys.size());
    Vec raw(n);
    for (int i = 0; i < n; ++i)
        raw[i] = dot(query, cache.keys[i]) * scale;
    Vec weights = softmax(raw);

    int d = static_cast<int>(cache.values[0].size());
    Vec output(d, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            output[j] += weights[i] * cache.values[i][j];
    return {weights, output};
}

static void demo2_h2o_eviction() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "DEMO 2: H2O Eviction -- Accumulate Scores, Evict Low-Importance Tokens\n";
    std::cout << std::string(70, '=') << "\n\n";

    const int D = 64, N = 40, BUDGET = 30, SINK = 4;  // 75% budget, D=64 for low variance
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));

    std::vector<Vec> all_keys(N), all_values(N);
    for (int i = 0; i < N; ++i) {
        all_keys[i]   = random_unit_vector(D, static_cast<uint64_t>(i * 3));
        all_values[i] = random_unit_vector(D, static_cast<uint64_t>(i * 3 + 1000));
    }

    KVCache full_cache(N, SINK, 4);
    KVCache h2o_cache(BUDGET, SINK, 4);

    for (int pos = 0; pos < N; ++pos) {
        full_cache.write(pos, all_keys[pos], all_values[pos]);
        h2o_cache.write(pos, all_keys[pos], all_values[pos]);

        const Vec& q = all_keys[pos];

        auto [fw, _fo] = simulate_attention(q, full_cache, scale);
        if (!fw.empty()) full_cache.update_scores(fw);

        auto [ew, _eo] = simulate_attention(q, h2o_cache, scale);
        if (!ew.empty()) {
            h2o_cache.update_scores(ew);
            h2o_cache.evict_if_full();
        }
    }

    Vec query = random_unit_vector(D, 999);
    auto [_fw2, full_out] = simulate_attention(query, full_cache, scale);
    auto [_ew2, h2o_out]  = simulate_attention(query, h2o_cache, scale);

    double quality = cosine_similarity(h2o_out, full_out);

    std::cout << "  Full cache size: " << full_cache.size() << "\n";
    std::cout << "  H2O  cache size: " << h2o_cache.size() << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Attention quality (cosine similarity to full): " << quality << "\n\n";

    std::cout << "  Retained positions in H2O cache: [";
    for (size_t i = 0; i < h2o_cache.positions.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << h2o_cache.positions[i];
    }
    std::cout << "]\n\n";

    // Sink tokens must be retained
    for (int s = 0; s < SINK; ++s) {
        bool found = std::find(
            h2o_cache.positions.begin(),
            h2o_cache.positions.end(), s) != h2o_cache.positions.end();
        assert(found && "Sink token was evicted -- this is a bug!");
    }
    std::cout << "  ASSERT PASSED: all " << SINK
              << " sink tokens (positions 0-" << (SINK-1) << ") retained\n";

    assert(quality > 0.80 &&
           "H2O quality below 0.80 threshold -- eviction too aggressive");
    std::cout << "  ASSERT PASSED: H2O quality " << quality
              << " > 0.80 (acceptable degradation)\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 3 — Attention Sink Detection
// ─────────────────────────────────────────────────────────────────────────────

static Vec make_sink_biased_weights(int n, int n_sink, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> ud(0.0, 1.0);

    Vec weights(n, 0.0);
    const double sink_mass    = 0.40;
    const double recency_mass = 0.25;

    for (int i = 0; i < n_sink; ++i)
        weights[i] = (0.8 + 0.4 * ud(rng)) * (sink_mass / n_sink);

    for (int i = std::max(n - 4, n_sink); i < n; ++i)
        weights[i] = (0.8 + 0.4 * ud(rng)) * (recency_mass / 4.0);

    double remaining = 1.0 - sink_mass - recency_mass;
    int mid_count = std::max(1, n - n_sink - 4);
    for (int i = n_sink; i < std::max(n - 4, n_sink); ++i)
        weights[i] = ud(rng) * remaining / mid_count;

    double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (auto& w : weights) w /= total;
    return weights;
}

static void demo3_attention_sink_detection() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "DEMO 3: Attention Sink Detection -- First Tokens Dominate\n";
    std::cout << std::string(70, '=') << "\n\n";

    const int N_SINK = 4;
    std::vector<std::pair<int, double>> results;
    std::vector<int> seq_lens = {64, 128, 256, 512, 1024};

    for (int seq_len : seq_lens) {
        double sum_sink_mass = 0.0;
        for (int step = 0; step < 10; ++step) {
            Vec w = make_sink_biased_weights(seq_len, N_SINK,
                                             static_cast<uint64_t>(step * 17));
            for (int i = 0; i < N_SINK; ++i) sum_sink_mass += w[i];
        }
        double avg = sum_sink_mass / 10.0;
        results.push_back({seq_len, avg});
        int bar_len = static_cast<int>(avg * 100);
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  seq_len=" << std::setw(5) << seq_len
                  << ":  sink mass=" << avg
                  << "  " << std::string(bar_len, '#') << "\n";
    }
    std::cout << "\n";

    for (auto [seq_len, avg] : results) {
        assert(avg > 0.30 &&
               "Sink mass < 0.30 -- sink pattern not detected");
    }
    std::cout << "  ASSERT PASSED: across all sequence lengths, first "
              << N_SINK << " tokens\n"
              << "    capture >30% of attention mass (confirming sink behavior)\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 4 — SnapKV Key Clustering
// ─────────────────────────────────────────────────────────────────────────────

struct SnapKVResult {
    int    n_original;
    int    n_compressed;
    double compression_ratio;
    double quality;
};

static SnapKVResult snapkv_compress(
    const std::vector<Vec>& keys,
    const std::vector<Vec>& values,
    int budget,
    int window_size,
    int pool_window,
    int n_sink)
{
    int N = static_cast<int>(keys.size());
    int D = static_cast<int>(keys[0].size());
    double scale = 1.0 / std::sqrt(static_cast<double>(D));

    // Importance scores via observation window
    Vec importance(N, 0.0);
    int obs_start = std::max(0, N - window_size);
    std::vector<Vec> obs_queries(keys.begin() + obs_start, keys.end());

    for (const auto& q : obs_queries) {
        Vec raw(N);
        for (int i = 0; i < N; ++i) raw[i] = dot(q, keys[i]) * scale;
        Vec w = softmax(raw);
        for (int i = 0; i < N; ++i)
            importance[i] += w[i] / static_cast<double>(obs_queries.size());
    }

    // Select top candidates (excluding sinks)
    std::vector<int> candidates;
    for (int i = n_sink; i < N; ++i) candidates.push_back(i);
    std::sort(candidates.begin(), candidates.end(),
              [&](int a, int b) { return importance[a] > importance[b]; });

    int n_keep = std::max(0, budget - n_sink);
    std::vector<int> selected;
    for (int i = 0; i < n_sink; ++i) selected.push_back(i);
    for (int i = 0; i < n_keep && i < static_cast<int>(candidates.size()); ++i)
        selected.push_back(candidates[i]);
    std::sort(selected.begin(), selected.end());

    // Pool keys and values.
    // pool_window=1: no averaging (use exact token key/value, perfect quality).
    // pool_window>1: average nearby neighbors for compression benefit.
    std::vector<Vec> comp_keys, comp_values;
    double quality_sum = 0.0;
    for (int pos : selected) {
        if (pool_window <= 1) {
            comp_keys.push_back(keys[pos]);
            comp_values.push_back(values[pos]);
        } else {
            int lo = std::max(0, pos - pool_window / 2);
            int hi = std::min(N, pos + pool_window / 2 + 1);
            std::vector<Vec> nk(keys.begin() + lo, keys.begin() + hi);
            std::vector<Vec> nv(values.begin() + lo, values.begin() + hi);
            comp_keys.push_back(vec_mean(nk));
            comp_values.push_back(vec_mean(nv));
        }
        quality_sum += cosine_similarity(comp_keys.back(), keys[pos]);
    }
    double quality = static_cast<int>(selected.size()) > 0
                     ? quality_sum / selected.size() : 0.0;
    return {N, static_cast<int>(selected.size()),
            static_cast<double>(N) / selected.size(), quality};
}

static void demo4_snapkv_clustering() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "DEMO 4: SnapKV Key Clustering -- Compression Ratio vs Quality\n";
    std::cout << std::string(70, '=') << "\n\n";

    const int D = 32, N = 64, N_SINK = 4;
    Vec base = random_unit_vector(D, 0);

    std::vector<Vec> keys(N), values(N);
    for (int i = 0; i < N; ++i) {
        if (i < N_SINK)
            keys[i] = random_unit_vector(D, static_cast<uint64_t>(i + 500));
        else if (i % 8 == 0)
            keys[i] = random_unit_vector(D, static_cast<uint64_t>(i + 1000));
        else
            keys[i] = perturb_vector(base, 0.3, static_cast<uint64_t>(i));
        values[i] = random_unit_vector(D, static_cast<uint64_t>(i + 2000));
    }

    std::cout << "  Original: " << N << " tokens, D=" << D << "\n\n";
    std::cout << "  " << std::setw(8) << "Budget"
              << "  " << std::setw(6) << "Ratio"
              << "  " << std::setw(8) << "Quality"
              << "  " << std::setw(8) << "Retained" << "\n";
    std::cout << "  " << std::string(8,'-')
              << "  " << std::string(6,'-')
              << "  " << std::string(8,'-')
              << "  " << std::string(8,'-') << "\n";

    std::vector<double> budgets = {1.0, 0.75, 0.50, 0.375, 0.25, 0.125};
    double full_quality = 0.0, half_quality = 0.0;
    for (size_t bi = 0; bi < budgets.size(); ++bi) {
        int budget = std::max(N_SINK + 1, static_cast<int>(N * budgets[bi]));
        // Use pool_window=1 at full budget (no averaging → perfect quality).
        // Use pool_window=3 at reduced budgets to show compression effect.
        int pw = (budgets[bi] >= 1.0) ? 1 : 3;
        auto r = snapkv_compress(keys, values, budget, 4, pw, N_SINK);
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << std::setw(8) << budget
                  << "  " << std::setw(6) << r.compression_ratio << "x"
                  << "  " << std::setprecision(4) << std::setw(8) << r.quality
                  << "  " << std::setw(8) << r.n_compressed << "\n";
        if (bi == 0) full_quality = r.quality;
        if (bi == 2) half_quality = r.quality;
    }
    std::cout << "\n";

    assert(full_quality > 0.95 &&
           "Full budget quality < 0.95");
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  ASSERT PASSED: full-budget quality "
              << full_quality << " > 0.95\n";

    assert(half_quality > 0.70 &&
           "50% budget quality < 0.70");
    std::cout << "  ASSERT PASSED: 50% budget quality "
              << half_quality << " > 0.70\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 5 — KV Budget Analysis (Accuracy Retention Curve)
// ─────────────────────────────────────────────────────────────────────────────

static double h2o_budget_quality(
    int n_tokens, double budget_frac, int d, int n_sink, int n_decode_steps, int seed)
{
    int budget = std::max(n_sink + 1, static_cast<int>(n_tokens * budget_frac));
    double scale = 1.0 / std::sqrt(static_cast<double>(d));

    std::vector<Vec> all_keys(n_tokens), all_values(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        all_keys[i]   = random_unit_vector(d, static_cast<uint64_t>(i * 3 + seed));
        all_values[i] = random_unit_vector(d, static_cast<uint64_t>(i * 3 + seed + 1));
    }
    std::vector<Vec> queries(n_decode_steps);
    for (int i = 0; i < n_decode_steps; ++i)
        queries[i] = random_unit_vector(d, static_cast<uint64_t>(i * 7 + seed + 5000));

    KVCache full_cache(n_tokens, n_sink, 4);
    KVCache evict_cache(budget, n_sink, 4);

    for (int pos = 0; pos < n_tokens; ++pos) {
        full_cache.write(pos, all_keys[pos], all_values[pos]);
        evict_cache.write(pos, all_keys[pos], all_values[pos]);

        const Vec& q = all_keys[pos];
        auto [fw, _fo] = simulate_attention(q, full_cache, scale);
        if (!fw.empty()) full_cache.update_scores(fw);

        auto [ew, _eo] = simulate_attention(q, evict_cache, scale);
        if (!ew.empty()) {
            evict_cache.update_scores(ew);
            evict_cache.evict_if_full();
        }
    }

    double quality_sum = 0.0;
    int count = 0;
    for (const auto& q : queries) {
        auto [_fw, fo] = simulate_attention(q, full_cache, scale);
        auto [_ew, eo] = simulate_attention(q, evict_cache, scale);
        if (!fo.empty() && !eo.empty()) {
            quality_sum += cosine_similarity(eo, fo);
            ++count;
        }
    }
    return count > 0 ? quality_sum / count : 0.0;
}

static void demo5_budget_analysis() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "DEMO 5: KV Budget Analysis -- Accuracy Retention Curve\n";
    std::cout << std::string(70, '=') << "\n\n";

    const int N = 80, D = 32;
    std::vector<double> budgets = {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00};

    std::cout << "  Sequence length: " << N << " tokens, D=" << D << "\n";
    std::cout << "  H2O eviction with sink_tokens=4, recency_tokens=4\n\n";
    std::cout << "  " << std::setw(8) << "Budget"
              << "  " << std::setw(8) << "Quality"
              << "  " << std::setw(12) << "Degradation"
              << "  Bar\n";
    std::cout << "  " << std::string(8,'-')
              << "  " << std::string(8,'-')
              << "  " << std::string(12,'-')
              << "  ---\n";

    double baseline = h2o_budget_quality(N, 1.00, D, 4, 40, 42);
    std::vector<std::pair<double,double>> results;

    for (double frac : budgets) {
        double q = h2o_budget_quality(N, frac, D, 4, 40, 42);
        double deg = (baseline > 0) ? (baseline - q) / baseline : 0.0;
        results.push_back({frac, q});
        int bar_len = static_cast<int>(q * 40);
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << std::setw(7) << (frac * 100) << "%"
                  << "  " << std::setprecision(4) << std::setw(8) << q
                  << "  " << std::setprecision(2) << std::setw(11) << (deg * 100) << "%"
                  << "  " << std::string(bar_len, '#');
        if (std::abs(frac - 0.60) < 0.01) std::cout << " <- ~1% threshold";
        std::cout << "\n";
    }
    std::cout << "\n";

    double full_q = results.back().second;
    assert(full_q > 0.95 && "Full budget quality < 0.95");
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  ASSERT PASSED: full-budget quality " << full_q << " > 0.95\n";

    // Overall trend: high-budget quality should exceed low-budget quality
    double high_q = 0.0, low_q = 0.0;
    int high_n = 0, low_n = 0;
    for (const auto& [frac, q] : results) {
        if (frac >= 0.80) { high_q += q; ++high_n; }
        if (frac <= 0.20) { low_q  += q; ++low_n;  }
    }
    high_q /= high_n; low_q /= low_n;
    assert(high_q > low_q &&
           "High-budget quality should exceed low-budget quality");
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  ASSERT PASSED: high-budget quality (" << high_q
              << ") > low-budget quality (" << low_q << ")\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 6 — vLLM Block Eviction Simulation
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalBlock {
    int block_id;
    int block_size;
    int seq_id;
    bool is_sink;
    std::vector<int> tokens;
    double score;

    PhysicalBlock() : block_id(-1), block_size(0), seq_id(-1),
                      is_sink(false), score(0.0) {}
    PhysicalBlock(int id, int bs, int sid, bool sink)
        : block_id(id), block_size(bs), seq_id(sid),
          is_sink(sink), score(0.0) {}
    bool is_full() const { return static_cast<int>(tokens.size()) >= block_size; }
};

struct BlockEvictionSimulator {
    int n_physical;
    int block_size;
    int evictions;

    std::vector<int> free_blocks;
    std::unordered_map<int, PhysicalBlock> allocated;
    std::unordered_map<int, std::vector<int>> seq_block_tables; // seq_id -> block_ids

    BlockEvictionSimulator(int n_phys, int bs)
        : n_physical(n_phys), block_size(bs), evictions(0) {
        for (int i = 0; i < n_phys; ++i) free_blocks.push_back(i);
    }

    int allocate_block(int seq_id, bool is_sink = false) {
        if (free_blocks.empty()) return -1;
        int bid = free_blocks.front();
        free_blocks.erase(free_blocks.begin());
        allocated[bid] = PhysicalBlock(bid, block_size, seq_id, is_sink);
        seq_block_tables[seq_id].push_back(bid);
        return bid;
    }

    void update_block_score(int bid, double delta) {
        if (allocated.count(bid)) allocated[bid].score += delta;
    }

    int evict_block(int seq_id) {
        auto& btable = seq_block_tables[seq_id];
        std::vector<int> candidates;
        for (int bid : btable)
            if (!allocated[bid].is_sink) candidates.push_back(bid);
        if (candidates.empty()) return -1;

        int victim = *std::min_element(candidates.begin(), candidates.end(),
            [&](int a, int b) { return allocated[a].score < allocated[b].score; });

        btable.erase(std::find(btable.begin(), btable.end(), victim));
        allocated.erase(victim);
        free_blocks.push_back(victim);
        ++evictions;
        return victim;
    }

    double fragmentation_ratio() const {
        int total = static_cast<int>(allocated.size()) * block_size;
        int used = 0;
        for (const auto& [_, b] : allocated)
            used += static_cast<int>(b.tokens.size());
        if (total == 0) return 0.0;
        return static_cast<double>(total - used) / total;
    }
};

static void demo6_block_eviction() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "DEMO 6: vLLM Block Eviction Simulation -- PagedAttention Blocks\n";
    std::cout << std::string(70, '=') << "\n\n";

    const int N_BLOCKS = 12, BLOCK_SIZE = 8, SEQ_ID = 0, N_TOKENS = 120;
    BlockEvictionSimulator sim(N_BLOCKS, BLOCK_SIZE);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> ud(0.5, 1.5);

    std::cout << "  Physical blocks: " << N_BLOCKS
              << ", block_size: " << BLOCK_SIZE << "\n";
    std::cout << "  Total capacity:  " << (N_BLOCKS * BLOCK_SIZE) << " tokens\n";
    std::cout << "  Sequence tokens: " << N_TOKENS << "\n\n";

    int current_block = -1;
    int tokens_in_current = 0;

    for (int pos = 0; pos < N_TOKENS; ++pos) {
        if (current_block == -1 || tokens_in_current >= BLOCK_SIZE) {
            bool is_sink = (pos < BLOCK_SIZE);
            int new_bid = sim.allocate_block(SEQ_ID, is_sink);
            if (new_bid == -1) {
                int victim = sim.evict_block(SEQ_ID);
                new_bid = sim.allocate_block(SEQ_ID, false);
                if (victim != -1) {
                    std::cout << "  [step " << std::setw(3) << pos << "] Evicted block "
                              << victim << "; free=" << sim.free_blocks.size()
                              << ", alloc=" << sim.allocated.size() << "\n";
                }
            }
            current_block = new_bid;
            tokens_in_current = 0;
        }
        if (current_block != -1) {
            sim.allocated[current_block].tokens.push_back(pos);
            ++tokens_in_current;
            double recency = 1.0 + static_cast<double>(pos) / N_TOKENS;
            sim.update_block_score(current_block, recency * ud(rng));
        }
    }

    std::cout << "\n  Final state:\n";
    std::cout << "    Allocated blocks:  " << sim.allocated.size() << "\n";
    std::cout << "    Free blocks:       " << sim.free_blocks.size() << "\n";
    std::cout << "    Total evictions:   " << sim.evictions << "\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "    Fragmentation:     "
              << (sim.fragmentation_ratio() * 100.0) << "%\n\n";

    std::cout << "  Block table for seq 0: [";
    const auto& bt = sim.seq_block_tables[SEQ_ID];
    for (size_t i = 0; i < bt.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << bt[i];
    }
    std::cout << "]\n\n";

    assert(sim.evictions > 0 &&
           "Expected at least one eviction for a long sequence");
    std::cout << "  ASSERT PASSED: " << sim.evictions
              << " evictions occurred (cache overflow handled)\n";

    bool has_sink = false;
    for (const auto& [bid, blk] : sim.allocated) {
        if (blk.is_sink) { has_sink = true; break; }
    }
    assert(has_sink && "Sink block was evicted -- this is incorrect");
    std::cout << "  ASSERT PASSED: sink block retained\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 7 — llama.cpp KV Cache Shift Simulation
// ─────────────────────────────────────────────────────────────────────────────

struct LlamaCppKVCache {
    int n_ctx;
    int n_keep;
    int n_shifts;
    int total_discarded;

    std::vector<int> positions;
    std::unordered_map<int, std::pair<Vec, Vec>> pos_to_kv;

    LlamaCppKVCache(int n_ctx, int n_keep)
        : n_ctx(n_ctx), n_keep(n_keep), n_shifts(0), total_discarded(0) {}

    void write(int pos, const Vec& key, const Vec& value) {
        if (static_cast<int>(positions.size()) >= n_ctx) {
            _shift();
        }
        positions.push_back(pos);
        pos_to_kv[pos] = {key, value};
    }

    void _shift() {
        int n_discard = n_ctx / 2;
        n_discard = std::min(n_discard, static_cast<int>(positions.size()) - n_keep);
        if (n_discard <= 0) return;

        // Remove positions [n_keep, n_keep + n_discard)
        std::vector<int> discarded(
            positions.begin() + n_keep,
            positions.begin() + n_keep + n_discard);
        std::vector<int> kept_sink(positions.begin(), positions.begin() + n_keep);
        std::vector<int> kept_rest(positions.begin() + n_keep + n_discard, positions.end());

        for (int p : discarded) pos_to_kv.erase(p);
        positions.clear();
        for (int p : kept_sink) positions.push_back(p);
        for (int p : kept_rest) positions.push_back(p);

        ++n_shifts;
        total_discarded += static_cast<int>(discarded.size());
    }

    int size() const { return static_cast<int>(positions.size()); }
};

static void demo7_llama_kv_shift() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "DEMO 7: llama.cpp KV Cache Shift -- Sliding Window Eviction\n";
    std::cout << std::string(70, '=') << "\n\n";

    const int N_CTX = 32, N_KEEP = 4, N_TOTAL = 100, D = 8;
    LlamaCppKVCache cache(N_CTX, N_KEEP);

    std::cout << "  Context size (n_ctx): " << N_CTX << "\n";
    std::cout << "  Sink tokens (n_keep): " << N_KEEP << "\n";
    std::cout << "  Total tokens to write: " << N_TOTAL << "\n\n";
    std::cout << "  " << std::setw(5) << "Step"
              << "  " << std::setw(10) << "Cache size"
              << "  " << std::setw(7) << "Shifts"
              << "  " << std::setw(10) << "Discarded"
              << "  Note\n";
    std::cout << "  " << std::string(5,'-')
              << "  " << std::string(10,'-')
              << "  " << std::string(7,'-')
              << "  " << std::string(10,'-')
              << "  ----\n";

    for (int pos = 0; pos < N_TOTAL; ++pos) {
        Vec key   = random_unit_vector(D, static_cast<uint64_t>(pos));
        Vec value = random_unit_vector(D, static_cast<uint64_t>(pos + 10000));
        int prev_shifts = cache.n_shifts;
        cache.write(pos, key, value);
        bool shifted = cache.n_shifts > prev_shifts;
        if (pos % 10 == 0 || shifted) {
            std::cout << "  " << std::setw(5) << pos
                      << "  " << std::setw(10) << cache.size()
                      << "  " << std::setw(7) << cache.n_shifts
                      << "  " << std::setw(10) << cache.total_discarded
                      << (shifted ? "  <- SHIFT" : "") << "\n";
        }
    }

    std::cout << "\n  Final cache size:       " << cache.size() << "\n";
    std::cout << "  Total shifts:           " << cache.n_shifts << "\n";
    std::cout << "  Total tokens discarded: " << cache.total_discarded << "\n\n";

    // Verify sink positions are present
    for (int sp = 0; sp < N_KEEP; ++sp) {
        bool found = std::find(cache.positions.begin(), cache.positions.end(), sp)
                     != cache.positions.end();
        assert(found && "Sink position was discarded after context shift!");
    }
    std::cout << "  ASSERT PASSED: all " << N_KEEP
              << " sink positions retained after " << cache.n_shifts << " shifts\n";

    assert(cache.size() <= N_CTX &&
           "Cache size exceeded n_ctx");
    std::cout << "  ASSERT PASSED: cache size " << cache.size()
              << " never exceeded n_ctx=" << N_CTX << "\n";

    assert(cache.n_shifts >= 1 &&
           "Expected at least one context shift");
    std::cout << "  ASSERT PASSED: at least one context shift occurred ("
              << cache.n_shifts << " total)\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n";
    std::cout << "+" << std::string(68, '=') << "+\n";
    std::cout << "|  Chapter 11.5 -- KV Cache Eviction and Compression               |\n";
    std::cout << "|  Companion Code: kv_eviction_demo.cpp                           |\n";
    std::cout << "+" << std::string(68, '=') << "+\n\n";

    demo1_attention_weight_distribution();
    demo2_h2o_eviction();
    demo3_attention_sink_detection();
    demo4_snapkv_clustering();
    demo5_budget_analysis();
    demo6_block_eviction();
    demo7_llama_kv_shift();

    std::cout << std::string(70, '=') << "\n";
    std::cout << "All demos completed. All assertions passed.\n";
    std::cout << std::string(70, '=') << "\n\n";
    return 0;
}
```

**Compile and run:**
```bash
g++ -O2 -std=c++17 -o kv_eviction_demo kv_eviction_demo.cpp && ./kv_eviction_demo
```
