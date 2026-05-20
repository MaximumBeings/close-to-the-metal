# Chapter 11: Prefill, Chunked Prefill, and Prompt Caching — Companion Code

## Python — `chunked_prefill_demo.py`

```python
"""
Chapter 11 — Prefill, Chunked Prefill, and Prompt Caching
Companion code: chunked_prefill_demo.py

Demonstrates:
  1. Prefill compute intensity vs. decode (memory-bandwidth-bound analysis)
  2. Token budget allocation: decode-first chunked prefill simulation
  3. TTFT impact — unchunked vs. chunked prefill across prompt lengths
  4. Radix prefix cache hit rate simulator (workload-configurable)
  5. Prefix caching cost savings calculator
  6. vLLM configuration guide for production deployments

No GPU required — all computations are analytical/simulated.
"""

import math
import random
import hashlib
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Compute intensity: prefill vs. decode
# ─────────────────────────────────────────────────────────────────────────────

def compute_arithmetic_intensity(
    prompt_tokens: int,
    batch_size: int,
    d_model: int = 4096,
    is_prefill: bool = True,
) -> dict:
    """
    Compute the arithmetic intensity (FLOP/byte) for one linear projection
    (e.g., the Q projection in self-attention) during prefill or decode.

    Arithmetic intensity = FLOPs / bytes_read_from_HBM

    H100 ridge point ≈ 591 FLOP/byte (1,979 TFLOP/s BF16 dense / 3.35 TB/s HBM3)
    Above ridge point: compute-bound.
    Below ridge point: memory-bandwidth-bound.
    """
    if is_prefill:
        # Input: [prompt_tokens, d_model], Weight: [d_model, d_model]
        m, k, n = prompt_tokens, d_model, d_model
    else:
        # Input: [batch_size, 1, d_model], Weight: [d_model, d_model]
        m, k, n = batch_size, d_model, d_model

    flops = 2 * m * k * n
    # Weight bytes (BF16 = 2 bytes/param); weights dominate HBM reads
    weight_bytes = k * n * 2
    # Input activation bytes
    input_bytes = m * k * 2
    total_bytes = weight_bytes + input_bytes

    intensity = flops / total_bytes

    H100_RIDGE_POINT = 591.0  # FLOP/byte (1,979 TFLOPS BF16 dense / 3.35 TB/s)
    bound = "compute-bound" if intensity > H100_RIDGE_POINT else "memory-bandwidth-bound"

    return {
        "phase": "prefill" if is_prefill else "decode",
        "tokens_processed": m,
        "flops": flops,
        "bytes_transferred": total_bytes,
        "arithmetic_intensity": intensity,
        "h100_ridge_point": H100_RIDGE_POINT,
        "bound": bound,
        "intensity_ratio_to_ridge": intensity / H100_RIDGE_POINT,
    }


def demo_compute_intensity():
    print("=" * 70)
    print("SECTION 1: Arithmetic Intensity — Prefill vs. Decode")
    print("=" * 70)
    print()

    configs = [
        ("Prefill (32K tokens)", 32000, 64, True),
        ("Prefill (8K tokens)",   8000, 64, True),
        ("Prefill (512 tokens)",   512, 64, True),
        ("Decode (batch=64)",        1, 64, False),
        ("Decode (batch=32)",        1, 32, False),
        ("Decode (batch=1)",         1,  1, False),
    ]

    print(f"{'Phase':<30} {'Tokens':>7} {'Intensity':>12} {'vs Ridge':>10} {'Bound'}")
    print("-" * 80)

    for label, tokens, batch, is_prefill in configs:
        r = compute_arithmetic_intensity(tokens, batch, is_prefill=is_prefill)
        t = r["tokens_processed"] if is_prefill else batch
        print(
            f"{label:<30} {t:>7} "
            f"{r['arithmetic_intensity']:>12.1f} "
            f"{r['intensity_ratio_to_ridge']:>9.1f}x  "
            f"{r['bound']}"
        )

    print()
    print("Ridge point (H100 BF16 dense): ~591 FLOP/byte")
    print("Above ridge → compute-bound (tensor core utilization matters)")
    print("Below ridge → memory-bandwidth-bound (HBM bandwidth is the limit)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Token budget allocator: decode-first chunked prefill simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Request:
    request_id: int
    prompt_tokens: int
    prefilled_so_far: int = 0
    in_decode: bool = False
    decode_tokens_generated: int = 0
    ttft_step: Optional[int] = None       # scheduler step when decode started

    @property
    def remaining_prefill(self) -> int:
        return self.prompt_tokens - self.prefilled_so_far

    @property
    def prefill_complete(self) -> bool:
        return self.prefilled_so_far >= self.prompt_tokens


@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 256
    enable_chunked_prefill: bool = True


def simulate_scheduler(
    requests: list,
    config: SchedulerConfig,
    num_steps: int = 50,
    step_time_ms: float = 15.0,
) -> dict:
    """
    Simulate vLLM's decode-first chunked prefill scheduler for N steps.

    Returns per-step statistics and per-request TTFT.
    """
    waiting_queue = list(requests)   # requests not yet started
    running_decode = []              # requests in decode phase
    prefill_queue = []               # requests actively being prefilled

    step_stats = []
    request_ttfts = {}

    for step in range(num_steps):
        # Admit new requests from waiting queue (if space in running pool)
        while waiting_queue and len(running_decode) + len(prefill_queue) < config.max_num_seqs:
            req = waiting_queue.pop(0)
            if config.enable_chunked_prefill:
                prefill_queue.append(req)
            else:
                # Without chunked prefill: admit only if budget allows full prefill
                prefill_queue.append(req)

        # Count decode tokens needed
        decode_count = len(running_decode)

        if config.enable_chunked_prefill:
            prefill_budget = config.max_num_batched_tokens - decode_count
            prefill_budget = max(0, prefill_budget)
        else:
            # Without chunked prefill: all tokens in budget go to first prefill request
            prefill_budget = config.max_num_batched_tokens

        # Allocate prefill budget to prefill queue
        prefill_tokens_this_step = 0
        newly_decoded = []

        if prefill_queue and prefill_budget > 0:
            req = prefill_queue[0]
            chunk = min(req.remaining_prefill, prefill_budget)
            req.prefilled_so_far += chunk
            prefill_tokens_this_step = chunk

            if req.prefill_complete:
                prefill_queue.pop(0)
                req.in_decode = True
                req.ttft_step = step
                request_ttfts[req.request_id] = step * step_time_ms
                newly_decoded.append(req)
                running_decode.append(req)

        elif prefill_queue and not config.enable_chunked_prefill:
            # Unchunked: full prefill may exceed budget — stall decode users
            req = prefill_queue[0]
            # In unchunked mode, we devote entire pass to prefill, ignoring decode budget
            chunk = req.remaining_prefill
            req.prefilled_so_far += chunk
            prefill_tokens_this_step = chunk
            decode_count = 0   # decode stalled this step

            if req.prefill_complete:
                prefill_queue.pop(0)
                req.in_decode = True
                req.ttft_step = step
                request_ttfts[req.request_id] = step * step_time_ms
                running_decode.append(req)

        # Generate one decode token per running decode sequence
        for req in running_decode:
            req.decode_tokens_generated += 1

        step_stats.append({
            "step": step,
            "decode_tokens": decode_count,
            "prefill_tokens": prefill_tokens_this_step,
            "total_tokens": decode_count + prefill_tokens_this_step,
            "running_decode": len(running_decode),
            "in_prefill": len(prefill_queue),
            "waiting": len(waiting_queue),
        })

    return {
        "step_stats": step_stats,
        "request_ttfts": request_ttfts,
        "requests": requests,
    }


def demo_token_budget():
    print("=" * 70)
    print("SECTION 2: Token Budget Allocation — Chunked vs. Unchunked Prefill")
    print("=" * 70)
    print()

    # Scenario: 32 active decode users + one 8K-token RAG request arrives
    num_decode_users = 32
    existing = [
        Request(i, prompt_tokens=200, prefilled_so_far=200, in_decode=True,
                decode_tokens_generated=10)
        for i in range(num_decode_users)
    ]

    new_request = Request(request_id=999, prompt_tokens=8000)

    config_chunked = SchedulerConfig(
        max_num_batched_tokens=4096,
        max_num_seqs=128,
        enable_chunked_prefill=True,
    )
    config_unchunked = SchedulerConfig(
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        enable_chunked_prefill=False,
    )

    for label, config in [("CHUNKED PREFILL", config_chunked),
                           ("UNCHUNKED PREFILL", config_unchunked)]:
        # Fresh request list each time
        req = Request(request_id=999, prompt_tokens=8000)
        decode_users = [
            Request(i, prompt_tokens=200, prefilled_so_far=200, in_decode=True)
            for i in range(num_decode_users)
        ]
        decode_users.append(req)

        print(f"--- {label} (max_tokens={config.max_num_batched_tokens}) ---")
        print(f"{'Step':>4}  {'Decode T':>9}  {'Prefill T':>10}  {'Total T':>8}  "
              f"{'Decode Users':>13}  {'Notes'}")
        print("-" * 70)

        waiting = [req]
        running_decode_sim = decode_users[:-1]  # 32 existing users in decode
        prefill_sim = []
        ttft_step = None

        for step in range(10):
            # Admit from waiting
            while waiting and len(running_decode_sim) + len(prefill_sim) < config.max_num_seqs:
                r = waiting.pop(0)
                prefill_sim.append(r)

            decode_count = len(running_decode_sim)

            if config.enable_chunked_prefill:
                pbudget = max(0, config.max_num_batched_tokens - decode_count)
            else:
                pbudget = 0  # unchunked: we handle below

            prefill_done_this = 0
            notes = ""

            if prefill_sim:
                r = prefill_sim[0]
                if config.enable_chunked_prefill:
                    chunk = min(r.remaining_prefill, pbudget)
                else:
                    chunk = r.remaining_prefill
                    decode_count = 0  # decode stalled
                    notes = "DECODE STALLED"

                r.prefilled_so_far += chunk
                prefill_done_this = chunk

                if r.prefill_complete:
                    prefill_sim.pop(0)
                    running_decode_sim.append(r)
                    if ttft_step is None:
                        ttft_step = step
                        notes = f"RAG user TTFT! ({step * 15:.0f} ms)"

            total = decode_count + prefill_done_this
            print(f"{step:>4}  {decode_count:>9}  {prefill_done_this:>10}  "
                  f"{total:>8}  {len(running_decode_sim):>13}  {notes}")

            if not prefill_sim and all(r.request_id != 999 or r.in_decode
                                       for r in running_decode_sim):
                if step >= (ttft_step or 0) + 2:
                    break

        ttft_ms = (ttft_step or 0) * 15.0
        print(f"\n  RAG user TTFT: {ttft_ms:.0f} ms (step {ttft_step})")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — TTFT benchmark: unchunked vs. chunked across prompt lengths
# ─────────────────────────────────────────────────────────────────────────────

def estimate_ttft(
    prompt_tokens: int,
    num_decode_users: int,
    max_num_batched_tokens: int,
    enable_chunked_prefill: bool,
    prefill_tokens_per_sec: float = 40000.0,
    step_time_ms: float = 15.0,
) -> dict:
    """
    Estimate TTFT for a new request in an analytical model.

    Without chunked prefill:
      TTFT ≈ ceil(prompt_tokens / prefill_tokens_per_sec × 1000) ms
      plus stall during large prefill pass.

    With chunked prefill:
      chunk_size = max_num_batched_tokens - num_decode_users
      num_steps  = ceil(prompt_tokens / chunk_size)
      TTFT       ≈ num_steps × step_time_ms
    """
    if enable_chunked_prefill:
        chunk_size = max(1, max_num_batched_tokens - num_decode_users)
        num_steps = math.ceil(prompt_tokens / chunk_size)
        ttft_ms = num_steps * step_time_ms
        # Stall imposed on decode users: 0 (they get tokens every step)
        decode_stall_ms = 0.0
    else:
        # Full prefill in one (possibly multi-kernel) pass
        ttft_ms = prompt_tokens / prefill_tokens_per_sec * 1000.0
        decode_stall_ms = ttft_ms  # decode users wait the full prefill time

    return {
        "prompt_tokens": prompt_tokens,
        "chunked": enable_chunked_prefill,
        "ttft_ms": ttft_ms,
        "decode_stall_ms": decode_stall_ms,
    }


def demo_ttft_comparison():
    print("=" * 70)
    print("SECTION 3: TTFT Comparison — Unchunked vs. Chunked Prefill")
    print("=" * 70)
    print()

    prompt_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_decode_users = 64
    max_tokens = 4096

    print(f"  Config: max_num_batched_tokens={max_tokens}, "
          f"decode_users={num_decode_users}")
    print()
    print(f"{'Prompt Tokens':>14}  {'TTFT Unchunked':>15}  {'TTFT Chunked':>13}  "
          f"{'Speedup':>9}  {'Decode Stall (Unchunked)':>24}")
    print("-" * 85)

    for pt in prompt_lengths:
        r_no  = estimate_ttft(pt, num_decode_users, max_tokens, False)
        r_yes = estimate_ttft(pt, num_decode_users, max_tokens, True)
        speedup = r_no["ttft_ms"] / r_yes["ttft_ms"] if r_yes["ttft_ms"] > 0 else 1.0
        print(
            f"{pt:>14}  "
            f"{r_no['ttft_ms']:>13.0f} ms  "
            f"{r_yes['ttft_ms']:>11.0f} ms  "
            f"{speedup:>8.1f}x  "
            f"{r_no['decode_stall_ms']:>22.0f} ms"
        )

    print()
    print("Decode stall = time existing users receive no new tokens.")
    print("Chunked prefill eliminates this stall entirely.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Radix prefix cache simulator
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_SIZE = 16  # tokens per KV block (matches vLLM default)


def tokenize_prefix(text: str) -> list:
    """Simulate tokenization as word-level for demonstration."""
    return text.lower().split()


def block_hash(tokens: tuple) -> str:
    """Hash a block of tokens to a stable cache key."""
    return hashlib.sha256(" ".join(str(t) for t in tokens).encode()).hexdigest()[:16]


@dataclass
class RadixNode:
    token_block: tuple           # BLOCK_SIZE token IDs
    block_key: str               # hash of token_block
    children: dict = field(default_factory=dict)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)


class RadixPrefixCache:
    """
    Simplified radix prefix cache implementing the core vLLM prefix caching
    logic: block-level hash matching over token sequences.
    """

    def __init__(self, max_blocks: int = 1000, block_size: int = BLOCK_SIZE):
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.root_children: dict = {}       # hash → RadixNode (first-level blocks)
        self.total_blocks: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.total_tokens_saved: int = 0
        self.total_tokens_prefilled: int = 0
        self.total_prefilled: int = 0

    def _split_into_blocks(self, token_ids: list) -> list:
        """Split token list into blocks of self.block_size."""
        return [
            tuple(token_ids[i:i + self.block_size])
            for i in range(0, len(token_ids) - len(token_ids) % self.block_size, self.block_size)
        ]
        # Note: partial last block is not cached (matches vLLM behavior)

    def lookup_and_store(self, token_ids: list) -> dict:
        """
        Walk radix tree for the given token sequence.
        Returns number of tokens found in cache (hit) and stores new blocks.
        """
        blocks = self._split_into_blocks(token_ids)
        if not blocks:
            return {"cached_tokens": 0, "prefill_tokens": len(token_ids), "hit": False}

        # Walk the tree depth-first
        current_children = self.root_children
        cached_blocks = 0

        for block in blocks:
            key = block_hash(block)
            if key in current_children:
                node = current_children[key]
                node.access_count += 1
                node.last_access = time.time()
                cached_blocks += 1
                current_children = node.children
            else:
                # Miss — insert remaining blocks
                for remaining_block in blocks[cached_blocks:]:
                    rkey = block_hash(remaining_block)
                    if self.total_blocks >= self.max_blocks:
                        self._evict_lru()
                    new_node = RadixNode(
                        token_block=remaining_block,
                        block_key=rkey,
                    )
                    current_children[rkey] = new_node
                    current_children = new_node.children
                    self.total_blocks += 1
                break

        cached_tokens = cached_blocks * self.block_size
        prefill_tokens = len(token_ids) - cached_tokens

        if cached_tokens > 0:
            self.hits += 1
            self.total_tokens_saved += cached_tokens
        else:
            self.misses += 1

        self.total_prefilled += prefill_tokens
        self.total_tokens_prefilled += prefill_tokens

        return {
            "cached_tokens": cached_tokens,
            "prefill_tokens": prefill_tokens,
            "hit": cached_tokens > 0,
            "hit_ratio": cached_tokens / len(token_ids) if token_ids else 0.0,
        }

    def _evict_lru(self):
        """Evict the least-recently-used leaf node (simplified LRU)."""
        # Walk tree to find LRU leaf — simplified: evict from root_children
        if self.root_children:
            lru_key = min(self.root_children,
                         key=lambda k: self.root_children[k].last_access)
            del self.root_children[lru_key]
            self.total_blocks -= 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "hit_rate": self.hit_rate,
            "hits": self.hits,
            "misses": self.misses,
            "tokens_saved": self.total_tokens_saved,
            "tokens_prefilled": self.total_tokens_prefilled,
            "total_blocks_cached": self.total_blocks,
        }


def simulate_prefix_cache_workload(
    system_prompt_tokens: int = 5000,
    avg_user_tokens: int = 50,
    num_requests: int = 1000,
    hit_rate_target: float = 0.95,
    unique_prefix_fraction: float = 0.05,
) -> dict:
    """
    Simulate a realistic workload where most requests share a system prompt prefix.

    unique_prefix_fraction: fraction of requests that have a unique prefix
    (simulating requests that bypass the shared prefix — e.g., different apps).
    """
    cache = RadixPrefixCache(max_blocks=50000)

    # Pre-generate shared system prompt token IDs
    shared_prefix = list(range(system_prompt_tokens))

    total_tokens_without_cache = 0
    total_tokens_with_cache = 0
    results_by_request = []

    for req_id in range(num_requests):
        # Unique requests bypass shared prefix
        if random.random() < unique_prefix_fraction:
            tokens = list(range(req_id * 100, req_id * 100 + system_prompt_tokens + avg_user_tokens))
        else:
            # Standard request: shared prefix + unique user tokens
            user_tokens = [system_prompt_tokens + req_id * avg_user_tokens + i
                          for i in range(avg_user_tokens)]
            tokens = shared_prefix + user_tokens

        total_tokens_without_cache += len(tokens)
        result = cache.lookup_and_store(tokens)
        total_tokens_with_cache += result["prefill_tokens"]
        results_by_request.append(result)

    stats = cache.stats()
    stats["total_tokens_without_cache"] = total_tokens_without_cache
    stats["total_tokens_with_cache"] = total_tokens_with_cache
    stats["compute_reduction_pct"] = (
        (1 - total_tokens_with_cache / total_tokens_without_cache) * 100
        if total_tokens_without_cache > 0 else 0
    )
    return stats


def demo_prefix_cache():
    print("=" * 70)
    print("SECTION 4: Radix Prefix Cache Simulator")
    print("=" * 70)
    print()

    scenarios = [
        ("95% shared prefix, 5% unique",  5000,  50, 1000, 0.05),
        ("80% shared prefix, 20% unique", 5000,  50, 1000, 0.20),
        ("Short system prompt (1K)",       1000,  50, 1000, 0.05),
        ("Long system prompt (10K)",      10000,  50, 1000, 0.05),
        ("Large user messages (500T)",     5000, 500, 1000, 0.05),
    ]

    print(f"{'Scenario':<42}  {'Hit Rate':>9}  {'Tokens Saved':>13}  {'Compute Reduction':>18}")
    print("-" * 90)

    for label, sys_tokens, user_tokens, n_reqs, unique_frac in scenarios:
        random.seed(42)
        stats = simulate_prefix_cache_workload(
            system_prompt_tokens=sys_tokens,
            avg_user_tokens=user_tokens,
            num_requests=n_reqs,
            unique_prefix_fraction=unique_frac,
        )
        print(
            f"{label:<42}  "
            f"{stats['hit_rate']:>8.1%}  "
            f"{stats['tokens_saved']:>13,}  "
            f"{stats['compute_reduction_pct']:>17.1f}%"
        )

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Prefix caching cost savings calculator
# ─────────────────────────────────────────────────────────────────────────────

def compute_cost_savings(
    system_prompt_tokens: int,
    avg_user_tokens: int,
    daily_requests: int,
    hit_rate: float,
    cost_per_1k_input_tokens: float = 0.01,
    prefill_tokens_per_sec: float = 40000.0,
) -> dict:
    """
    Compute daily cost with and without prefix caching.

    Models the compute cost as proportional to tokens prefilled.
    """
    avg_total_tokens = system_prompt_tokens + avg_user_tokens

    # Without caching: prefill all tokens for every request
    daily_tokens_no_cache = daily_requests * avg_total_tokens
    daily_cost_no_cache = daily_tokens_no_cache / 1000.0 * cost_per_1k_input_tokens

    # With caching:
    # Cold miss (fraction = 1 - hit_rate): full prefill
    # Cache hit (fraction = hit_rate): only user tokens prefilled
    cold_requests = daily_requests * (1 - hit_rate)
    hot_requests  = daily_requests * hit_rate

    cold_tokens = cold_requests * avg_total_tokens
    hot_tokens  = hot_requests * avg_user_tokens  # only user-unique tokens

    daily_tokens_with_cache = cold_tokens + hot_tokens
    daily_cost_with_cache = daily_tokens_with_cache / 1000.0 * cost_per_1k_input_tokens

    # TTFT on cache hits (only user tokens to prefill)
    ttft_hit_ms  = avg_user_tokens / prefill_tokens_per_sec * 1000.0
    ttft_miss_ms = avg_total_tokens / prefill_tokens_per_sec * 1000.0
    ttft_avg_ms  = hit_rate * ttft_hit_ms + (1 - hit_rate) * ttft_miss_ms

    return {
        "daily_cost_no_cache":   daily_cost_no_cache,
        "daily_cost_with_cache": daily_cost_with_cache,
        "daily_savings":         daily_cost_no_cache - daily_cost_with_cache,
        "cost_reduction_pct":    (1 - daily_cost_with_cache / daily_cost_no_cache) * 100,
        "ttft_hit_ms":   ttft_hit_ms,
        "ttft_miss_ms":  ttft_miss_ms,
        "ttft_avg_ms":   ttft_avg_ms,
        "ttft_speedup_on_hit": ttft_miss_ms / ttft_hit_ms if ttft_hit_ms > 0 else 1.0,
        "tokens_no_cache":   daily_tokens_no_cache,
        "tokens_with_cache": daily_tokens_with_cache,
    }


def demo_cost_savings():
    print("=" * 70)
    print("SECTION 5: Prefix Caching Cost Savings Calculator")
    print("=" * 70)
    print()

    result = compute_cost_savings(
        system_prompt_tokens=5000,
        avg_user_tokens=50,
        daily_requests=100_000,
        hit_rate=0.95,
    )

    print("Scenario: 5K-token system prompt, 50-token user messages, 100K daily requests")
    print()
    print(f"  Daily cost WITHOUT prefix caching: ${result['daily_cost_no_cache']:>10,.2f}")
    print(f"  Daily cost WITH    prefix caching: ${result['daily_cost_with_cache']:>10,.2f}")
    print(f"  Daily savings:                     ${result['daily_savings']:>10,.2f}")
    print(f"  Cost reduction:                    {result['cost_reduction_pct']:>9.1f}%")
    print()
    print(f"  TTFT on cache hit:   {result['ttft_hit_ms']:>6.1f} ms")
    print(f"  TTFT on cache miss:  {result['ttft_miss_ms']:>6.1f} ms")
    print(f"  TTFT average:        {result['ttft_avg_ms']:>6.1f} ms")
    print(f"  TTFT speedup (hit):  {result['ttft_speedup_on_hit']:>6.1f}x")
    print()

    # Hit rate sensitivity table
    print("  Hit rate sensitivity:")
    print(f"  {'Hit Rate':>9}  {'Daily Cost':>12}  {'Savings %':>10}  {'Avg TTFT (ms)':>14}")
    print("  " + "-" * 55)
    for hr in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        r = compute_cost_savings(5000, 50, 100_000, hr)
        print(f"  {hr:>9.0%}  ${r['daily_cost_with_cache']:>10,.2f}  "
              f"{r['cost_reduction_pct']:>9.1f}%  {r['ttft_avg_ms']:>13.1f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — vLLM configuration guide
# ─────────────────────────────────────────────────────────────────────────────

def demo_vllm_config_guide():
    print("=" * 70)
    print("SECTION 6: vLLM Configuration Guide for Chunked Prefill + Prefix Caching")
    print("=" * 70)
    print()

    configs = {
        "Chat (short prompts ≤ 512T, latency-sensitive)": {
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "max_num_batched_tokens": 2048,
            "max_num_seqs": 256,
            "rationale": "Small chunks keep ITL stable; prefix cache helps with repeat system prompts.",
        },
        "RAG (1K–8K prompts, mixed latency)": {
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 128,
            "rationale": "4K budget balances decode continuity and prefill speed. Prefix cache critical.",
        },
        "Long-context (32K+ prompts, throughput focus)": {
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 32,
            "rationale": "Large chunks reduce chunking overhead at scale. Few seqs to keep per-step latency manageable.",
        },
        "Offline batch (no TTFT SLA)": {
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "rationale": "No chunking overhead; maximize throughput by packing large prefill batches.",
        },
    }

    for workload, cfg in configs.items():
        print(f"  [{workload}]")
        print(f"    enable_chunked_prefill: {cfg['enable_chunked_prefill']}")
        print(f"    enable_prefix_caching:  {cfg['enable_prefix_caching']}")
        print(f"    max_num_batched_tokens: {cfg['max_num_batched_tokens']}")
        print(f"    max_num_seqs:           {cfg['max_num_seqs']}")
        print(f"    Rationale: {cfg['rationale']}")
        print()

    print("  SAFETY CHECK: max_num_batched_tokens >= 4 × max_num_seqs")
    print("  If violated, new requests may never get prefill budget.")
    print()
    for workload, cfg in configs.items():
        ok = cfg["max_num_batched_tokens"] >= 4 * cfg["max_num_seqs"]
        print(f"    {workload[:45]:<45} → {'OK' if ok else 'VIOLATION!'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    demo_compute_intensity()
    demo_token_budget()
    demo_ttft_comparison()
    demo_prefix_cache()
    demo_cost_savings()
    demo_vllm_config_guide()

    print("=" * 70)
    print("Chapter 11 demo complete.")
    print()
    print("Key results:")
    print("  • Prefill arithmetic intensity >> decode intensity (compute vs. BW bound)")
    print("  • Chunked prefill: 5–10× TTFT improvement for long-context requests")
    print("  • Existing decode users: unaffected (no stall) with chunked prefill")
    print("  • Prefix cache at 95% hit rate: 94% compute cost reduction")
    print("  • Average TTFT: 100× faster on cache hits (only user tokens to prefill)")
    print("=" * 70)

```

## C++ — `chunked_prefill_demo.cpp`

```cpp
/*
 * Chapter 11 — Prefill, Chunked Prefill, and Prompt Caching
 * Companion code: chunked_prefill_demo.cpp
 *
 * Demonstrates:
 *   1. Prefill vs. decode arithmetic intensity analysis (matching §11.2.2)
 *   2. llama.cpp micro-batch (--ubatch-size) simulation
 *   3. Prompt prefix cache: linear match (--cache-prompt equivalent)
 *   4. Multi-session KV cache reuse tracker
 *   5. Token budget allocation visualiser (decode-first policy)
 *   6. TTFT estimation table: unchunked vs. chunked
 *
 * Compile:
 *   g++ -std=c++17 -O2 -o chunked_prefill_demo chunked_prefill_demo.cpp
 *
 * Run:
 *   ./chunked_prefill_demo
 *
 * No external dependencies required.
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
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
// Section 1 — Arithmetic intensity: prefill vs. decode
// ─────────────────────────────────────────────────────────────────────────────

struct IntensityResult {
    std::string phase;
    int         tokens_processed;
    double      flops;
    double      bytes_transferred;
    double      arithmetic_intensity;
    double      h100_ridge_point;
    std::string bound;
    double      intensity_ratio;
};

/*
 * compute_arithmetic_intensity
 *
 * Computes FLOP/byte for one linear projection (e.g. Q projection in attention).
 *
 * Prefill: input = [N_prompt, d_model], Weight = [d_model, d_model]
 *          FLOPs = 2 * N_prompt * d_model * d_model
 *
 * Decode:  input = [B, 1, d_model], Weight = [d_model, d_model]
 *          FLOPs = 2 * B * d_model * d_model   (B sequences, 1 token each)
 *
 * H100 ridge point: ~591 FLOP/byte
 *   (1,979 TFLOP/s BF16 dense / 3.35 TB/s HBM3)
 */
IntensityResult compute_arithmetic_intensity(
    int  prompt_tokens,
    int  batch_size,
    int  d_model     = 4096,
    bool is_prefill  = true)
{
    static const double H100_RIDGE = 591.0;  // 1,979 TFLOPS BF16 dense / 3.35 TB/s

    long long m = is_prefill ? prompt_tokens : batch_size;
    long long k = d_model;
    long long n = d_model;

    double flops         = 2.0 * m * k * n;
    double weight_bytes  = static_cast<double>(k * n) * 2.0;   // BF16
    double input_bytes   = static_cast<double>(m * k) * 2.0;
    double total_bytes   = weight_bytes + input_bytes;
    double intensity     = flops / total_bytes;

    IntensityResult r;
    r.phase               = is_prefill ? "prefill" : "decode";
    r.tokens_processed    = static_cast<int>(m);
    r.flops               = flops;
    r.bytes_transferred   = total_bytes;
    r.arithmetic_intensity = intensity;
    r.h100_ridge_point    = H100_RIDGE;
    r.bound               = (intensity > H100_RIDGE) ? "compute-bound" : "memory-BW-bound";
    r.intensity_ratio     = intensity / H100_RIDGE;
    return r;
}

void demo_arithmetic_intensity() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "SECTION 1: Arithmetic Intensity — Prefill vs. Decode\n";
    std::cout << std::string(70, '=') << "\n\n";

    struct Case {
        std::string label;
        int tokens;
        int batch;
        bool is_prefill;
    };

    std::vector<Case> cases = {
        {"Prefill (32K tokens)", 32000, 64, true},
        {"Prefill (8K tokens)",   8000, 64, true},
        {"Prefill (512 tokens)",   512, 64, true},
        {"Decode (batch=64)",        1, 64, false},
        {"Decode (batch=32)",        1, 32, false},
        {"Decode (batch=1)",         1,  1, false},
    };

    std::cout << std::left  << std::setw(30) << "Phase"
              << std::right << std::setw(8)  << "Tokens"
              << std::setw(15) << "Intensity"
              << std::setw(11) << "vs Ridge"
              << "  Bound\n";
    std::cout << std::string(80, '-') << "\n";

    for (auto& c : cases) {
        auto r = compute_arithmetic_intensity(c.tokens, c.batch, 4096, c.is_prefill);
        int t = c.is_prefill ? c.tokens : c.batch;
        std::cout << std::left  << std::setw(30) << c.label
                  << std::right << std::setw(8)  << t
                  << std::setw(14) << std::fixed << std::setprecision(1)
                  << r.arithmetic_intensity
                  << std::setw(9)  << std::setprecision(1)
                  << r.intensity_ratio << "x"
                  << "  " << r.bound << "\n";
    }

    std::cout << "\nRidge point (H100 BF16 dense): ~591 FLOP/byte\n";
    std::cout << "Above ridge -> compute-bound (tensor cores saturated)\n";
    std::cout << "Below ridge -> memory-BW-bound (HBM bandwidth is the limit)\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2 — llama.cpp micro-batch prefill simulation (--ubatch-size)
// ─────────────────────────────────────────────────────────────────────────────

/*
 * In llama.cpp, --ubatch-size N sets the physical micro-batch size.
 * When processing a long prompt with n_batch = 4096 tokens but ubatch = 512,
 * llama_decode is called 8 times internally, each processing 512 tokens.
 *
 * Unlike vLLM's chunked prefill, there is NO interleaving with decode tokens.
 * All prefill chunks complete before decode begins.
 *
 * This simulation models the micro-batch processing and tracks:
 * - Per-chunk kernel launch time (latency model)
 * - Peak GPU memory usage (activation memory per chunk)
 * - Total prefill wall time vs. full-batch prefill
 */

struct MicroBatchResult {
    int     total_tokens;
    int     ubatch_size;
    int     num_chunks;
    double  total_prefill_ms;
    double  unchunked_prefill_ms;
    double  peak_activation_mb_per_chunk;
    double  peak_activation_mb_unchunked;
    double  overhead_ratio;   // chunked / unchunked time ratio
};

/*
 * Model GPU FLOP efficiency as a function of matrix size.
 * Small matrices (<256 tokens) have lower efficiency on tensor cores.
 * Large matrices (≥2K tokens) approach theoretical peak.
 */
double flop_efficiency_at_size(int tokens) {
    if (tokens >= 4096) return 0.85;
    if (tokens >= 2048) return 0.78;
    if (tokens >= 1024) return 0.68;
    if (tokens >=  512) return 0.55;
    if (tokens >=  256) return 0.40;
    return 0.28;
}

MicroBatchResult simulate_ubatch_prefill(
    int   total_tokens,
    int   ubatch_size,
    int   d_model        = 4096,
    int   num_layers     = 32,
    float prefill_tflops = 312.0f)   // A100 BF16 peak TFLOP/s (use 989.0f for H100)
{
    int num_chunks = (total_tokens + ubatch_size - 1) / ubatch_size;

    // Estimate time per chunk based on FLOP count and efficiency
    double chunked_total_ms = 0.0;
    for (int c = 0; c < num_chunks; ++c) {
        int chunk_tokens = std::min(ubatch_size, total_tokens - c * ubatch_size);
        // FLOPs per layer per chunk (attention + FFN, simplified)
        double flops_per_layer = 2.0 * chunk_tokens * d_model * d_model * 4.0;
        double total_flops     = flops_per_layer * num_layers;
        double efficiency      = flop_efficiency_at_size(chunk_tokens);
        double effective_tflops = prefill_tflops * 1e12 * efficiency;
        double chunk_ms         = total_flops / effective_tflops * 1000.0;
        // Add kernel launch overhead (~0.5 ms per chunk)
        chunk_ms += 0.5;
        chunked_total_ms += chunk_ms;
    }

    // Unchunked: single large pass
    double unchunked_efficiency = flop_efficiency_at_size(total_tokens);
    double total_flops_unchunked = 2.0 * total_tokens * d_model * d_model * 4.0 * num_layers;
    double unchunked_ms = total_flops_unchunked / (prefill_tflops * 1e12 * unchunked_efficiency) * 1000.0;

    // Peak activation memory: proportional to chunk_size × d_model × num_layers
    // (simplified; real activations include Q, K, V, attention scores per layer)
    double bytes_per_token_per_layer = d_model * 2.0 * 4;   // 4 activation tensors, BF16
    double peak_act_chunk   = ubatch_size * bytes_per_token_per_layer * num_layers / 1e6;
    double peak_act_full    = total_tokens * bytes_per_token_per_layer * num_layers / 1e6;

    MicroBatchResult r;
    r.total_tokens               = total_tokens;
    r.ubatch_size                = ubatch_size;
    r.num_chunks                 = num_chunks;
    r.total_prefill_ms           = chunked_total_ms;
    r.unchunked_prefill_ms       = unchunked_ms;
    r.peak_activation_mb_per_chunk = peak_act_chunk;
    r.peak_activation_mb_unchunked = peak_act_full;
    r.overhead_ratio             = chunked_total_ms / unchunked_ms;
    return r;
}

void demo_ubatch_simulation() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "SECTION 2: llama.cpp Micro-Batch Prefill Simulation (--ubatch-size)\n";
    std::cout << std::string(70, '=') << "\n\n";

    int prompt_tokens = 8192;

    std::cout << "Prompt: " << prompt_tokens << " tokens (LLaMA 3 8B, H100)\n\n";
    std::cout << std::left  << std::setw(13) << "ubatch-size"
              << std::right << std::setw(8)  << "Chunks"
              << std::setw(16) << "Chunked (ms)"
              << std::setw(18) << "Unchunked (ms)"
              << std::setw(12) << "Overhead"
              << std::setw(16) << "Peak Act. (MB)" << "\n";
    std::cout << std::string(83, '-') << "\n";

    for (int ubatch : {128, 256, 512, 1024, 2048, 4096, 8192}) {
        auto r = simulate_ubatch_prefill(prompt_tokens, ubatch);
        std::cout << std::left  << std::setw(13) << ubatch
                  << std::right << std::setw(8)  << r.num_chunks
                  << std::setw(15) << std::fixed << std::setprecision(1) << r.total_prefill_ms
                  << std::setw(17) << std::setprecision(1) << r.unchunked_prefill_ms
                  << std::setw(10) << std::setprecision(2) << r.overhead_ratio << "x"
                  << std::setw(16) << std::setprecision(1) << r.peak_activation_mb_per_chunk
                  << "\n";
    }

    std::cout << "\nNote: ubatch=8192 equals unchunked (no split).\n";
    std::cout << "Overhead comes from smaller GEMM efficiency + kernel launch cost.\n";
    std::cout << "Peak activation memory scales with ubatch_size, not total prompt length.\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3 — Linear prefix cache (--cache-prompt equivalent)
// ─────────────────────────────────────────────────────────────────────────────

/*
 * llama.cpp's --cache-prompt performs a LINEAR prefix match between the
 * incoming request's token IDs and the current KV cache contents.
 *
 * - Match from position 0 forward.
 * - At first mismatch, truncate the KV cache and recompute from that position.
 * - Unlike vLLM's radix tree, this works only within one session at a time.
 *
 * This simulation models a single llama.cpp server session handling
 * multiple sequential requests and measures cache hit/miss behavior.
 */

struct LinearCacheSession {
    std::vector<int> cached_tokens;    // token IDs currently in KV cache
    int kv_cache_size;                 // maximum tokens in context
    int total_requests;
    int full_hits;
    int partial_hits;
    int full_misses;
    long long total_tokens_saved;
    long long total_tokens_prefilled;

    explicit LinearCacheSession(int ctx_size = 4096)
        : kv_cache_size(ctx_size), total_requests(0),
          full_hits(0), partial_hits(0), full_misses(0),
          total_tokens_saved(0), total_tokens_prefilled(0) {}

    struct LookupResult {
        int cached_prefix_len;   // tokens reused from cache
        int prefill_tokens;      // tokens that need to be prefilled
        bool is_full_hit;
        bool is_partial_hit;
    };

    LookupResult lookup(const std::vector<int>& new_tokens) {
        // Find longest common prefix between cached_tokens and new_tokens
        int common = 0;
        int limit  = std::min((int)cached_tokens.size(), (int)new_tokens.size());
        for (int i = 0; i < limit; ++i) {
            if (cached_tokens[i] == new_tokens[i]) ++common;
            else break;
        }

        LookupResult r;
        r.cached_prefix_len = common;
        r.prefill_tokens    = static_cast<int>(new_tokens.size()) - common;
        r.is_full_hit       = (common == static_cast<int>(new_tokens.size()));
        r.is_partial_hit    = (common > 0 && !r.is_full_hit);

        // Update KV cache: truncate at mismatch, then append new suffix
        cached_tokens.resize(common);
        for (int i = common; i < (int)new_tokens.size(); ++i)
            cached_tokens.push_back(new_tokens[i]);
        // Enforce context size limit
        if ((int)cached_tokens.size() > kv_cache_size)
            cached_tokens.resize(kv_cache_size);

        ++total_requests;
        if (r.is_full_hit)        ++full_hits;
        else if (r.is_partial_hit)++partial_hits;
        else                      ++full_misses;

        total_tokens_saved    += r.cached_prefix_len;
        total_tokens_prefilled += r.prefill_tokens;
        return r;
    }

    void print_stats() const {
        int total = full_hits + partial_hits + full_misses;
        double full_hit_rate = total > 0 ? (double)full_hits  / total : 0;
        double any_hit_rate  = total > 0 ? (double)(full_hits + partial_hits) / total : 0;
        std::cout << "  Full hits:    " << full_hits   << " (" << std::setprecision(1) << std::fixed
                  << full_hit_rate * 100 << "%)\n";
        std::cout << "  Partial hits: " << partial_hits << "\n";
        std::cout << "  Full misses:  " << full_misses  << "\n";
        std::cout << "  Any-hit rate: " << any_hit_rate * 100 << "%\n";
        std::cout << "  Tokens saved: " << total_tokens_saved << "\n";
        std::cout << "  Tokens prefilled: " << total_tokens_prefilled << "\n";
    }
};

void demo_linear_prefix_cache() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "SECTION 3: llama.cpp Linear Prefix Cache (--cache-prompt equivalent)\n";
    std::cout << std::string(70, '=') << "\n\n";

    // Scenario 1: Repeated system prompt (high hit rate)
    {
        std::cout << "--- Scenario A: Consistent system prompt (chat session) ---\n";
        LinearCacheSession sess(8192);

        // System prompt: tokens 0–499 (500 tokens)
        std::vector<int> system_prompt(500);
        std::iota(system_prompt.begin(), system_prompt.end(), 0);

        // 10 turns: system prompt + growing conversation history
        std::vector<int> conversation;
        for (int i = 0; i < 10; ++i) {
            std::vector<int> request = system_prompt;
            // Append accumulated conversation
            for (int t : conversation) request.push_back(t);
            // Add new user message (10 tokens)
            for (int j = 0; j < 10; ++j)
                request.push_back(1000 + i * 10 + j);

            auto r = sess.lookup(request);
            std::cout << "  Turn " << std::setw(2) << i+1
                      << ": " << std::setw(4) << (int)request.size() << " tokens in prompt"
                      << "  | cached=" << std::setw(4) << r.cached_prefix_len
                      << " prefill=" << std::setw(4) << r.prefill_tokens
                      << (r.is_full_hit ? " [FULL HIT]" : r.is_partial_hit ? " [PARTIAL HIT]" : " [MISS]")
                      << "\n";

            // Extend conversation with assistant response (15 tokens)
            for (int j = 0; j < 15; ++j)
                conversation.push_back(2000 + i * 15 + j);
        }
        std::cout << "\n";
        sess.print_stats();
        std::cout << "\n";
    }

    // Scenario 2: Different users, same system prompt — no cross-request sharing
    {
        std::cout << "--- Scenario B: Different users on same server (no cross-user sharing) ---\n";
        LinearCacheSession sess(8192);

        std::vector<int> system_prompt(500);
        std::iota(system_prompt.begin(), system_prompt.end(), 0);

        for (int user = 0; user < 5; ++user) {
            std::vector<int> request = system_prompt;
            // Each user has unique message
            for (int j = 0; j < 20; ++j)
                request.push_back(10000 + user * 100 + j);

            auto r = sess.lookup(request);
            std::cout << "  User " << user + 1
                      << ": " << std::setw(4) << (int)request.size() << " tokens"
                      << "  | cached=" << std::setw(4) << r.cached_prefix_len
                      << " prefill=" << std::setw(4) << r.prefill_tokens
                      << (r.cached_prefix_len > 0 ? " [system prompt cached]" : " [FULL MISS — new user]")
                      << "\n";
        }

        std::cout << "\n";
        std::cout << "  Observation: Only the LAST user's system prompt is in cache.\n";
        std::cout << "  New user always triggers full system prompt prefill (500 tokens).\n";
        std::cout << "  vLLM's radix tree would cache this for ALL users simultaneously.\n\n";
        sess.print_stats();
        std::cout << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4 — Token budget visualiser (decode-first allocation)
// ─────────────────────────────────────────────────────────────────────────────

struct BudgetStep {
    int step;
    int decode_tokens;
    int prefill_tokens;
    int total_tokens;
    int running_decode;
    int prefill_remaining;
    std::string note;
};

std::vector<BudgetStep> simulate_token_budget(
    int max_num_batched_tokens,
    int initial_decode_users,
    int new_request_prompt_tokens,
    int max_steps = 20)
{
    int decode_users    = initial_decode_users;
    int prefill_remaining = new_request_prompt_tokens;
    bool in_prefill     = true;
    std::vector<BudgetStep> steps;

    for (int s = 0; s < max_steps && (in_prefill || decode_users > initial_decode_users); ++s) {
        int decode_count  = decode_users;
        int prefill_budget = std::max(0, max_num_batched_tokens - decode_count);

        BudgetStep step;
        step.step = s;
        step.decode_tokens = decode_count;
        step.running_decode = decode_users;

        if (in_prefill && prefill_remaining > 0) {
            int chunk = std::min(prefill_remaining, prefill_budget);
            step.prefill_tokens    = chunk;
            step.total_tokens      = decode_count + chunk;
            prefill_remaining     -= chunk;
            step.prefill_remaining = prefill_remaining;

            if (prefill_remaining <= 0) {
                in_prefill = false;
                ++decode_users;   // new request enters decode
                step.note = "Prefill complete → decode start";
            }
        } else {
            step.prefill_tokens    = 0;
            step.total_tokens      = decode_count;
            step.prefill_remaining = 0;
            if (s > 2) break;    // show a few pure-decode steps
        }

        steps.push_back(step);
    }
    return steps;
}

void demo_token_budget_visual() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "SECTION 4: Token Budget Visualiser (Decode-First Allocation)\n";
    std::cout << std::string(70, '=') << "\n\n";

    struct Config {
        std::string label;
        int max_tokens;
        int decode_users;
        int prompt_tokens;
    };

    std::vector<Config> cfgs = {
        {"RAG (4K budget, 64 users, 8K prompt)",     4096, 64,  8192},
        {"Chat (2K budget, 32 users, 1K prompt)",    2048, 32,  1024},
        {"Long ctx (16K budget, 32 users, 32K prompt)", 16384, 32, 32768},
    };

    for (auto& c : cfgs) {
        std::cout << "--- " << c.label << " ---\n";
        std::cout << "  max_num_batched_tokens=" << c.max_tokens
                  << "  decode_users=" << c.decode_users
                  << "  prompt=" << c.prompt_tokens << " tokens\n\n";

        auto steps = simulate_token_budget(c.max_tokens, c.decode_users, c.prompt_tokens);

        std::cout << "  " << std::left << std::setw(5) << "Step"
                  << std::right << std::setw(10) << "Decode T"
                  << std::setw(11) << "Prefill T"
                  << std::setw(9)  << "Total T"
                  << std::setw(13) << "Prefil Rem."
                  << "  Note\n";
        std::cout << "  " << std::string(65, '-') << "\n";

        for (auto& s : steps) {
            std::cout << "  " << std::left << std::setw(5) << s.step
                      << std::right << std::setw(10) << s.decode_tokens
                      << std::setw(11) << s.prefill_tokens
                      << std::setw(9)  << s.total_tokens
                      << std::setw(13) << s.prefill_remaining
                      << "  " << s.note << "\n";
        }

        // TTFT estimate
        int ttft_steps = 0;
        for (auto& s : steps) {
            if (!s.note.empty()) { ttft_steps = s.step + 1; break; }
        }
        std::cout << "\n  TTFT: ~" << ttft_steps << " steps × 15ms = "
                  << ttft_steps * 15 << " ms\n";

        // Budget bar for first step
        auto& s0 = steps[0];
        int bar_width = 60;
        int decode_bar = (int)std::round((double)s0.decode_tokens / c.max_tokens * bar_width);
        int prefill_bar = (int)std::round((double)s0.prefill_tokens / c.max_tokens * bar_width);
        int spare_bar   = bar_width - decode_bar - prefill_bar;

        std::cout << "\n  Step 0 budget bar (60 chars = " << c.max_tokens << " tokens):\n";
        std::cout << "  |";
        std::cout << std::string(decode_bar, 'D');
        std::cout << std::string(prefill_bar, 'P');
        if (spare_bar > 0) std::cout << std::string(spare_bar, '.');
        std::cout << "|\n";
        std::cout << "   D=decode  P=prefill chunk  .=unused\n\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5 — TTFT estimation table
// ─────────────────────────────────────────────────────────────────────────────

struct TTFTResult {
    int    prompt_tokens;
    bool   chunked;
    double ttft_ms;
    double decode_stall_ms;
};

TTFTResult estimate_ttft(
    int    prompt_tokens,
    int    num_decode_users,
    int    max_num_batched_tokens,
    bool   enable_chunked_prefill,
    double prefill_tokens_per_sec = 40000.0,
    double step_time_ms           = 15.0)
{
    TTFTResult r;
    r.prompt_tokens = prompt_tokens;
    r.chunked       = enable_chunked_prefill;

    if (enable_chunked_prefill) {
        int chunk_size  = std::max(1, max_num_batched_tokens - num_decode_users);
        int num_steps   = (prompt_tokens + chunk_size - 1) / chunk_size;
        r.ttft_ms       = num_steps * step_time_ms;
        r.decode_stall_ms = 0.0;
    } else {
        r.ttft_ms         = prompt_tokens / prefill_tokens_per_sec * 1000.0;
        r.decode_stall_ms = r.ttft_ms;
    }
    return r;
}

void demo_ttft_table() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "SECTION 5: TTFT Comparison Table\n";
    std::cout << std::string(70, '=') << "\n\n";

    int decode_users = 64;
    int max_tokens   = 4096;

    std::cout << "  Config: max_num_batched_tokens=" << max_tokens
              << "  decode_users=" << decode_users << "\n\n";
    std::cout << "  " << std::left << std::setw(15) << "Prompt Tokens"
              << std::right << std::setw(17) << "TTFT Unchunked"
              << std::setw(15) << "TTFT Chunked"
              << std::setw(10) << "Speedup"
              << std::setw(22) << "Decode Stall (Unchunked)" << "\n";
    std::cout << "  " << std::string(79, '-') << "\n";

    for (int pt : {512, 1024, 2048, 4096, 8192, 16384, 32768}) {
        auto r0 = estimate_ttft(pt, decode_users, max_tokens, false);
        auto r1 = estimate_ttft(pt, decode_users, max_tokens, true);
        double speedup = r0.ttft_ms > 0 ? r0.ttft_ms / r1.ttft_ms : 1.0;

        std::cout << "  " << std::left << std::setw(15) << pt
                  << std::right << std::fixed << std::setprecision(0)
                  << std::setw(14) << r0.ttft_ms << " ms"
                  << std::setw(12) << r1.ttft_ms << " ms"
                  << std::setprecision(1)
                  << std::setw(8) << speedup << "x"
                  << std::setprecision(0)
                  << std::setw(20) << r0.decode_stall_ms << " ms\n";
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6 — Configuration safety check
// ─────────────────────────────────────────────────────────────────────────────

void demo_config_safety() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "SECTION 6: Configuration Safety Check\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << "  Rule: max_num_batched_tokens >= 4 × max_num_seqs\n";
    std::cout << "  Violation causes prefill starvation.\n\n";

    struct Cfg {
        std::string workload;
        int max_tokens;
        int max_seqs;
    };

    std::vector<Cfg> configs = {
        {"Chat (short prompts)",           2048, 256},
        {"RAG (medium prompts)",           4096, 128},
        {"Long-context (32K+ prompts)",   16384,  32},
        {"Offline batch",                 32768, 512},
        {"BAD: tokens too low",            1024, 512},   // violation
        {"BAD: seqs too high",             2048, 1024},  // violation
    };

    std::cout << "  " << std::left << std::setw(35) << "Workload"
              << std::right << std::setw(13) << "max_tokens"
              << std::setw(11) << "max_seqs"
              << std::setw(15) << "Min Required"
              << "  Status\n";
    std::cout << "  " << std::string(80, '-') << "\n";

    for (auto& c : configs) {
        int required = 4 * c.max_seqs;
        bool ok = c.max_tokens >= required;
        std::cout << "  " << std::left << std::setw(35) << c.workload
                  << std::right << std::setw(13) << c.max_tokens
                  << std::setw(11) << c.max_seqs
                  << std::setw(15) << required
                  << "  " << (ok ? "OK" : "VIOLATION — prefill starvation risk!") << "\n";
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\nChapter 11 — Prefill, Chunked Prefill, and Prompt Caching\n";
    std::cout << "Companion code: chunked_prefill_demo.cpp\n";
    std::cout << std::string(70, '=') << "\n\n";

    demo_arithmetic_intensity();
    demo_ubatch_simulation();
    demo_linear_prefix_cache();
    demo_token_budget_visual();
    demo_ttft_table();
    demo_config_safety();

    std::cout << std::string(70, '=') << "\n";
    std::cout << "Chapter 11 C++ demo complete.\n\n";
    std::cout << "Key results:\n";
    std::cout << "  * Prefill: deeply compute-bound (intensity >> H100 ridge point)\n";
    std::cout << "  * Decode:  memory-bandwidth-bound (intensity << ridge point)\n";
    std::cout << "  * llama.cpp --ubatch-size: reduces peak activation memory\n";
    std::cout << "    at the cost of 1.5-2x longer total prefill time.\n";
    std::cout << "  * --cache-prompt: single-session linear match only.\n";
    std::cout << "    New users always prefill the full system prompt.\n";
    std::cout << "  * Chunked prefill: decode users receive tokens every step.\n";
    std::cout << "    TTFT for new requests: 5-13x faster vs. unchunked.\n";
    std::cout << "  * Safety: max_num_batched_tokens >= 4 x max_num_seqs.\n";
    std::cout << std::string(70, '=') << "\n";

    return 0;
}

```

