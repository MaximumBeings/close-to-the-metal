# Code — Chapter 40: The vLLM V1 Architecture

Companion code for **Chapter 40: The vLLM V1 Architecture — Three-Process Design and the New Scheduler**.

Demos simulate vLLM V1's three-process ZMQ message-passing architecture,
async scheduler loop, hash-based KV block deduplication, multi-step
scheduling, and V0→V1 migration path — all without requiring a running vLLM instance.

---

## Python

```python
# vllm_v1_demo.py
# Chapter 40 — The vLLM V1 Architecture
#
# Seven self-contained demos covering:
#   Demo 1: V0 vs V1 architecture model
#   Demo 2: ZMQ message passing model
#   Demo 3: Block hash deduplication
#   Demo 4: Multi-step scheduling
#   Demo 5: Scheduler preemption policy
#   Demo 6: V1 throughput model
#   Demo 7: Migration checklist
#
# Requirements: Python 3.10+, standard library only
# Usage: python vllm_v1_demo.py

import time
import math
import hashlib
import random
import statistics
import threading
import queue
import collections
from dataclasses import dataclass, field
from typing import Optional

SEPARATOR = "=" * 70


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: V0 vs V1 Architecture Model
# ─────────────────────────────────────────────────────────────────────────────

def demo1_v0_vs_v1_architecture():
    """
    Simulate the synchronous (V0) vs asynchronous (V1) request lifecycle.
    Measure simulated CPU overhead and request queue depth over time.
    """
    print(SEPARATOR)
    print("Demo 1: V0 vs V1 Architecture — Synchronous vs Async Lifecycle")
    print(SEPARATOR)

    # --- Simulation parameters ---
    # Time is in microseconds (μs) throughout this demo.
    # All values are representative, not real GPU timings.

    GPU_STEP_US = 20_000        # 20ms forward pass
    SCHED_OVERHEAD_V0_US = 1_500  # 1.5ms: scheduling + pickle + output processing in-GIL
    SCHED_OVERHEAD_V1_US = 150    # 0.15ms: ZMQ msg + async handoff
    REQUEST_ARRIVAL_INTERVAL_US = 3_000   # one request every 3ms

    NUM_REQUESTS = 20

    def simulate_v0(num_requests: int) -> dict:
        """V0: process is synchronous — schedule → GPU → output in serial."""
        clock = 0
        completions = []
        queue_depths = []
        waiting = list(range(num_requests))
        arrival_times = [i * REQUEST_ARRIVAL_INTERVAL_US for i in range(num_requests)]

        idx = 0
        while waiting or idx < num_requests:
            # Accept newly arrived requests
            while idx < num_requests and arrival_times[idx] <= clock:
                idx += 1

            if not waiting:
                clock += REQUEST_ARRIVAL_INTERVAL_US
                continue

            queue_depths.append(len(waiting))
            req = waiting.pop(0)

            # V0: schedule (CPU, blocks GIL), then GPU, then output (CPU, blocks GIL)
            clock += SCHED_OVERHEAD_V0_US // 2   # scheduler phase 1
            clock += GPU_STEP_US                  # GPU forward
            clock += SCHED_OVERHEAD_V0_US // 2   # output processing phase

            completions.append(clock)
            # Accept more arrivals while we were busy
            while idx < num_requests and arrival_times[idx] <= clock:
                waiting.append(idx)
                idx += 1

        total_time = completions[-1]
        avg_latency = statistics.mean([completions[i] - arrival_times[i]
                                       for i in range(num_requests)])
        return {
            "total_time_ms": total_time / 1000,
            "avg_latency_ms": avg_latency / 1000,
            "throughput_rps": num_requests / (total_time / 1_000_000),
            "avg_queue_depth": statistics.mean(queue_depths) if queue_depths else 0,
        }

    def simulate_v1(num_requests: int) -> dict:
        """
        V1: scheduler process runs independently from API server.
        GPU can pipeline with scheduler overhead.
        """
        clock = 0
        completions = []
        queue_depths = []
        arrival_times = [i * REQUEST_ARRIVAL_INTERVAL_US for i in range(num_requests)]
        waiting = []
        idx = 0

        while len(completions) < num_requests:
            # Accept arrivals
            while idx < num_requests and arrival_times[idx] <= clock:
                waiting.append(idx)
                idx += 1

            if not waiting:
                clock += 500  # poll interval
                continue

            queue_depths.append(len(waiting))
            req = waiting.pop(0)

            # V1: scheduler overhead is hidden behind GPU execution
            # Scheduler runs on separate CPU → overlaps with previous GPU step
            sched_time = SCHED_OVERHEAD_V1_US
            gpu_time = GPU_STEP_US
            # Effective step time: max(sched, gpu) not sched+gpu
            step_time = max(sched_time, gpu_time)
            clock += step_time

            completions.append(clock)
            while idx < num_requests and arrival_times[idx] <= clock:
                waiting.append(idx)
                idx += 1

        total_time = completions[-1]
        avg_latency = statistics.mean([completions[i] - arrival_times[i]
                                       for i in range(num_requests)])
        return {
            "total_time_ms": total_time / 1000,
            "avg_latency_ms": avg_latency / 1000,
            "throughput_rps": num_requests / (total_time / 1_000_000),
            "avg_queue_depth": statistics.mean(queue_depths) if queue_depths else 0,
        }

    v0 = simulate_v0(NUM_REQUESTS)
    v1 = simulate_v1(NUM_REQUESTS)

    print(f"\n  Simulation: {NUM_REQUESTS} requests, arrival every {REQUEST_ARRIVAL_INTERVAL_US/1000:.1f}ms")
    print(f"  GPU step time: {GPU_STEP_US/1000:.0f}ms")
    print(f"  Scheduler overhead  V0: {SCHED_OVERHEAD_V0_US/1000:.1f}ms  V1: {SCHED_OVERHEAD_V1_US/1000:.2f}ms")
    print()
    print(f"  {'Metric':<28} {'V0':>12} {'V1':>12} {'Improvement':>14}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*14}")
    print(f"  {'Total wall time (ms)':<28} {v0['total_time_ms']:>12.1f} {v1['total_time_ms']:>12.1f} {(v0['total_time_ms']-v1['total_time_ms'])/v0['total_time_ms']*100:>13.1f}%")
    print(f"  {'Avg request latency (ms)':<28} {v0['avg_latency_ms']:>12.1f} {v1['avg_latency_ms']:>12.1f} {(v0['avg_latency_ms']-v1['avg_latency_ms'])/v0['avg_latency_ms']*100:>13.1f}%")
    print(f"  {'Throughput (req/s)':<28} {v0['throughput_rps']:>12.1f} {v1['throughput_rps']:>12.1f} {(v1['throughput_rps']-v0['throughput_rps'])/v0['throughput_rps']*100:>13.1f}%")
    print(f"  {'Avg queue depth':<28} {v0['avg_queue_depth']:>12.1f} {v1['avg_queue_depth']:>12.1f}")

    # Assertions
    assert v1["total_time_ms"] <= v0["total_time_ms"], "V1 should complete faster than V0"
    assert v1["avg_latency_ms"] <= v0["avg_latency_ms"], "V1 latency should be lower"
    assert v1["throughput_rps"] >= v0["throughput_rps"], "V1 throughput should be higher"

    print("\n  [PASS] All V0 vs V1 assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: ZMQ Message Passing Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MockRequest:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_time: float = 0.0

@dataclass
class MockBatch:
    request_ids: list
    total_tokens: int
    created_at: float = 0.0

@dataclass
class MockOutput:
    request_ids: list
    tokens_generated: int
    processed_at: float = 0.0

def demo2_zmq_message_passing():
    """
    Simulate ZMQ-style scheduler↔worker message passing.
    Measure message latency, batching efficiency, and queue behavior.
    """
    print()
    print(SEPARATOR)
    print("Demo 2: ZMQ Message Passing Model — Scheduler↔Worker Exchange")
    print(SEPARATOR)

    BATCH_SIZE = 8
    NUM_REQUESTS = 40
    ARRIVAL_RATE_HZ = 50        # requests per second
    ZMQ_LATENCY_US = 10         # one-way ZMQ send latency
    SCHEDULER_OVERHEAD_US = 200  # scheduling logic per step
    GPU_STEP_MS = 20             # GPU forward pass

    # Queues simulate ZMQ sockets
    api_to_sched = queue.Queue()   # PUSH → PULL (requests)
    sched_to_worker = queue.Queue()  # PUSH → PULL (batches)
    worker_to_sched = queue.Queue()  # PUSH → PULL (outputs)
    sched_to_api = queue.Queue()     # PUSH → PULL (responses)

    results = {
        "batches_sent": 0,
        "total_requests": 0,
        "batch_sizes": [],
        "round_trip_times_ms": [],
    }

    stop_event = threading.Event()

    def api_server():
        """Simulate API server: produce requests at arrival rate."""
        for i in range(NUM_REQUESTS):
            req = MockRequest(
                request_id=f"req-{i:03d}",
                prompt_tokens=random.randint(32, 256),
                max_new_tokens=random.randint(50, 200),
                arrival_time=time.monotonic(),
            )
            api_to_sched.put(req)
            time.sleep(1.0 / ARRIVAL_RATE_HZ)
        stop_event.set()

    def scheduler():
        """Simulate V1 scheduler: batch requests, send to worker."""
        pending = []
        while not stop_event.is_set() or not api_to_sched.empty():
            # Drain inbox (non-blocking)
            try:
                while True:
                    req = api_to_sched.get_nowait()
                    pending.append(req)
            except queue.Empty:
                pass

            if pending:
                # Build batch up to BATCH_SIZE
                batch_reqs = pending[:BATCH_SIZE]
                pending = pending[BATCH_SIZE:]
                batch = MockBatch(
                    request_ids=[r.request_id for r in batch_reqs],
                    total_tokens=sum(r.prompt_tokens for r in batch_reqs),
                    created_at=time.monotonic(),
                )
                # Simulate ZMQ + scheduler overhead
                time.sleep((ZMQ_LATENCY_US + SCHEDULER_OVERHEAD_US) / 1_000_000)
                sched_to_worker.put((batch, batch_reqs))
                results["batches_sent"] += 1
                results["batch_sizes"].append(len(batch_reqs))

            # Drain worker outputs
            try:
                while True:
                    output, orig_reqs = worker_to_sched.get_nowait()
                    now = time.monotonic()
                    for r in orig_reqs:
                        rtt = (now - r.arrival_time) * 1000
                        results["round_trip_times_ms"].append(rtt)
                        results["total_requests"] += 1
                    sched_to_api.put(output)
            except queue.Empty:
                time.sleep(0.001)

    def worker():
        """Simulate GPU worker: process batch, return output."""
        while not stop_event.is_set() or not sched_to_worker.empty():
            try:
                batch, orig_reqs = sched_to_worker.get(timeout=0.05)
                # Simulate GPU forward pass
                time.sleep(GPU_STEP_MS / 1000)
                output = MockOutput(
                    request_ids=batch.request_ids,
                    tokens_generated=len(batch.request_ids),
                    processed_at=time.monotonic(),
                )
                worker_to_sched.put((output, orig_reqs))
            except queue.Empty:
                pass

    t_api = threading.Thread(target=api_server)
    t_sched = threading.Thread(target=scheduler)
    t_worker = threading.Thread(target=worker)

    t0 = time.monotonic()
    t_api.start(); t_sched.start(); t_worker.start()
    t_api.join(); t_sched.join(timeout=5.0); t_worker.join(timeout=5.0)
    elapsed = time.monotonic() - t0

    rtts = results["round_trip_times_ms"]
    batch_sizes = results["batch_sizes"]

    print(f"\n  Requests sent:   {NUM_REQUESTS}")
    print(f"  Requests served: {results['total_requests']}")
    print(f"  Batches formed:  {results['batches_sent']}")
    print(f"  Wall time:       {elapsed*1000:.0f}ms")
    print()
    if rtts:
        print(f"  Round-trip latency (ms):")
        print(f"    P50: {sorted(rtts)[len(rtts)//2]:.1f}ms")
        print(f"    P95: {sorted(rtts)[int(len(rtts)*0.95)]:.1f}ms")
        print(f"    P99: {sorted(rtts)[int(len(rtts)*0.99)]:.1f}ms")
    if batch_sizes:
        print(f"\n  Batch sizes: mean={statistics.mean(batch_sizes):.1f}  "
              f"max={max(batch_sizes)}  min={min(batch_sizes)}")

    assert results["batches_sent"] > 0, "Should have formed at least one batch"
    assert results["total_requests"] > 0, "Should have served at least one request"
    if batch_sizes:
        assert max(batch_sizes) <= BATCH_SIZE, "Batch size should not exceed limit"

    print("\n  [PASS] ZMQ message passing assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: Block Hash Deduplication
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KVBlock:
    block_id: int
    token_ids: tuple
    content_hash: bytes
    ref_count: int = 0
    is_full: bool = False

class HashBasedPrefixCache:
    """
    Simulate V1's hash-based KV block prefix cache.
    Tracks hit/miss rates and ref-counted block lifecycle.
    """
    BLOCK_SIZE = 16

    def __init__(self, max_blocks: int = 256):
        self.max_blocks = max_blocks
        self.blocks: dict[int, KVBlock] = {}  # block_id → KVBlock
        self.hash_table: dict[bytes, int] = {}  # content_hash → block_id
        self.lru_order: list[int] = []  # block_ids in LRU order (tail=most recent)
        self.next_block_id = 0
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "allocations": 0}

    def _hash_tokens(self, token_ids: tuple) -> bytes:
        return hashlib.sha256(str(token_ids).encode()).digest()[:8]

    def _allocate_physical_block(self, token_ids: tuple) -> KVBlock:
        if len(self.blocks) >= self.max_blocks:
            # Evict LRU block with ref_count == 0
            for bid in self.lru_order[:]:
                blk = self.blocks[bid]
                if blk.ref_count == 0:
                    del self.hash_table[blk.content_hash]
                    del self.blocks[bid]
                    self.lru_order.remove(bid)
                    self.stats["evictions"] += 1
                    break

        content_hash = self._hash_tokens(token_ids)
        blk = KVBlock(
            block_id=self.next_block_id,
            token_ids=token_ids,
            content_hash=content_hash,
            ref_count=1,
            is_full=(len(token_ids) == self.BLOCK_SIZE),
        )
        self.next_block_id += 1
        self.blocks[blk.block_id] = blk
        if blk.is_full:
            self.hash_table[content_hash] = blk.block_id
        self.lru_order.append(blk.block_id)
        self.stats["allocations"] += 1
        return blk

    def get_or_allocate_blocks(self, token_ids: list) -> tuple[list[KVBlock], int]:
        """
        For a sequence of token_ids, return (blocks, prefill_tokens_needed).
        prefill_tokens_needed = number of tokens NOT found in cache.
        """
        blocks = []
        prefill_needed = 0
        n_full_blocks = len(token_ids) // self.BLOCK_SIZE

        for i in range(n_full_blocks):
            chunk = tuple(token_ids[i * self.BLOCK_SIZE:(i + 1) * self.BLOCK_SIZE])
            content_hash = self._hash_tokens(chunk)
            if content_hash in self.hash_table:
                # Cache HIT
                bid = self.hash_table[content_hash]
                blk = self.blocks[bid]
                blk.ref_count += 1
                # Move to end of LRU
                if bid in self.lru_order:
                    self.lru_order.remove(bid)
                self.lru_order.append(bid)
                blocks.append(blk)
                self.stats["hits"] += 1
            else:
                # Cache MISS — need to prefill this block
                blk = self._allocate_physical_block(chunk)
                blocks.append(blk)
                prefill_needed += self.BLOCK_SIZE
                self.stats["misses"] += 1

        # Handle partial tail block (never cached)
        tail_tokens = token_ids[n_full_blocks * self.BLOCK_SIZE:]
        if tail_tokens:
            blk = self._allocate_physical_block(tuple(tail_tokens))
            blocks.append(blk)
            prefill_needed += len(tail_tokens)

        return blocks, prefill_needed

    def release_blocks(self, blocks: list[KVBlock]):
        for blk in blocks:
            if blk.block_id in self.blocks:
                self.blocks[blk.block_id].ref_count = max(
                    0, self.blocks[blk.block_id].ref_count - 1)

    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0


def demo3_block_hash_deduplication():
    """
    Simulate hash-based prefix cache with a shared system prompt workload.
    Measure cache hit rates and prefill token savings.
    """
    print()
    print(SEPARATOR)
    print("Demo 3: Block Hash Deduplication — Prefix Cache Hit/Miss Rates")
    print(SEPARATOR)

    BLOCK_SIZE = 16
    SYSTEM_PROMPT_TOKENS = 64   # 4 full blocks — shared across all requests
    USER_PROMPT_MIN = 10
    USER_PROMPT_MAX = 50
    NUM_REQUESTS = 100
    VOCAB_SIZE = 32000

    random.seed(42)

    # Generate a fixed system prompt token sequence
    system_prompt = list(range(100, 100 + SYSTEM_PROMPT_TOKENS))

    cache = HashBasedPrefixCache(max_blocks=512)

    total_tokens_needed = 0
    total_prefill_done = 0
    all_blocks = []

    for i in range(NUM_REQUESTS):
        user_len = random.randint(USER_PROMPT_MIN, USER_PROMPT_MAX)
        # Most requests share system prompt; a few have unique prompts
        if i % 10 == 0:
            # Unique prompt (no shared prefix)
            tokens = [random.randint(500, VOCAB_SIZE) for _ in range(user_len + SYSTEM_PROMPT_TOKENS)]
        else:
            # Shared system prompt + unique user message
            user_tokens = [random.randint(200, 500) for _ in range(user_len)]
            tokens = system_prompt + user_tokens

        total_tokens_needed += len(tokens)
        blocks, prefill_needed = cache.get_or_allocate_blocks(tokens)
        total_prefill_done += prefill_needed
        all_blocks.append(blocks)

        # Simulate request completion: release blocks
        if i >= 5:  # keep first 5 requests alive for overlap
            cache.release_blocks(all_blocks[i - 5])

    hit_rate = cache.hit_rate()
    prefill_savings_pct = (1 - total_prefill_done / total_tokens_needed) * 100

    print(f"\n  Workload: {NUM_REQUESTS} requests, system prompt={SYSTEM_PROMPT_TOKENS} tokens")
    print(f"  90% share system prompt, 10% fully unique")
    print()
    print(f"  Cache statistics:")
    print(f"    Block hits:          {cache.stats['hits']}")
    print(f"    Block misses:        {cache.stats['misses']}")
    print(f"    Cache hit rate:      {hit_rate*100:.1f}%")
    print(f"    Evictions:           {cache.stats['evictions']}")
    print(f"    Total allocations:   {cache.stats['allocations']}")
    print()
    print(f"  Prefill savings:")
    print(f"    Total tokens in prompts:    {total_tokens_needed}")
    print(f"    Tokens actually prefilled:  {total_prefill_done}")
    print(f"    Prefill savings:            {prefill_savings_pct:.1f}%")

    # Assertions
    assert hit_rate > 0.0, "Should have at least some cache hits"
    assert prefill_savings_pct > 10.0, "Should save at least 10% on prefill"
    assert cache.stats["allocations"] > 0, "Should allocate blocks"

    print("\n  [PASS] Block hash deduplication assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4: Multi-Step Scheduling
# ─────────────────────────────────────────────────────────────────────────────

def demo4_multi_step_scheduling():
    """
    Simulate K-step decode scheduling.
    Show overhead amortization and effective tokens/second vs K.
    """
    print()
    print(SEPARATOR)
    print("Demo 4: Multi-Step Scheduling — Overhead Amortization vs K")
    print(SEPARATOR)

    GPU_STEP_US = 5_000         # 5ms per decode step (batch of 32)
    SCHED_OVERHEAD_US = 500     # 0.5ms per scheduler invocation
    TOTAL_TOKENS = 5_000        # generate 5000 tokens total

    def simulate_k_step(k: int) -> dict:
        """Simulate generating TOTAL_TOKENS with K steps per scheduler call."""
        tokens_generated = 0
        total_time_us = 0
        sched_invocations = 0

        while tokens_generated < TOTAL_TOKENS:
            # One scheduler invocation
            total_time_us += SCHED_OVERHEAD_US
            sched_invocations += 1

            # K GPU steps
            steps_this_inv = min(k, TOTAL_TOKENS - tokens_generated)
            total_time_us += steps_this_inv * GPU_STEP_US
            tokens_generated += steps_this_inv

        throughput = tokens_generated / (total_time_us / 1_000_000)
        overhead_pct = (sched_invocations * SCHED_OVERHEAD_US) / total_time_us * 100

        return {
            "k": k,
            "total_time_ms": total_time_us / 1000,
            "sched_invocations": sched_invocations,
            "throughput_tps": throughput,
            "overhead_pct": overhead_pct,
        }

    k_values = [1, 2, 5, 10, 20, 50]
    results = [simulate_k_step(k) for k in k_values]

    print(f"\n  GPU step time: {GPU_STEP_US/1000:.0f}ms, Scheduler overhead: {SCHED_OVERHEAD_US/1000:.1f}ms")
    print(f"  Total tokens to generate: {TOTAL_TOKENS}")
    print()
    print(f"  {'K':>4}  {'Total time':>12}  {'Sched calls':>12}  {'Throughput':>14}  {'Overhead':>10}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*14}  {'-'*10}")
    for r in results:
        print(f"  {r['k']:>4}  {r['total_time_ms']:>10.0f}ms  {r['sched_invocations']:>12}  "
              f"{r['throughput_tps']:>12.0f}/s  {r['overhead_pct']:>9.1f}%")

    # Show amortization ratio
    baseline_tps = results[0]["throughput_tps"]
    print()
    print(f"  Throughput gain vs K=1:")
    for r in results[1:]:
        gain = (r["throughput_tps"] - baseline_tps) / baseline_tps * 100
        bar = "█" * int(gain / 2)
        print(f"    K={r['k']:>2}: +{gain:4.1f}% {bar}")

    # Assertions
    assert all(results[i]["throughput_tps"] >= results[i-1]["throughput_tps"]
               for i in range(1, len(results))), \
        "Throughput should be non-decreasing as K increases"
    assert all(results[i]["overhead_pct"] <= results[i-1]["overhead_pct"]
               for i in range(1, len(results))), \
        "Overhead fraction should decrease as K increases"
    assert results[-1]["overhead_pct"] < results[0]["overhead_pct"] / 5, \
        "Overhead at K=50 should be <1/5 of K=1"

    print("\n  [PASS] Multi-step scheduling assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5: Scheduler Preemption Policy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SchedulerRequest:
    request_id: str
    priority: int           # 0=normal, 1=high, 2=urgent
    arrival_time: float
    prompt_tokens: int
    remaining_tokens: int
    started: bool = False
    start_time: float = 0.0
    finish_time: float = 0.0

class FIFOScheduler:
    """FIFO: process requests in arrival order."""
    def __init__(self, max_batch: int = 8):
        self.max_batch = max_batch
        self.running: list[SchedulerRequest] = []
        self.waiting: collections.deque = collections.deque()
        self.finished: list[SchedulerRequest] = []
        self.clock = 0.0

    def add_request(self, req: SchedulerRequest):
        self.waiting.append(req)

    def step(self, step_time: float = 0.020):
        """One scheduler step: fill batch from FIFO queue."""
        # Fill running queue from waiting (FIFO order)
        while len(self.running) < self.max_batch and self.waiting:
            req = self.waiting.popleft()
            if not req.started:
                req.started = True
                req.start_time = self.clock
            self.running.append(req)

        if not self.running:
            self.clock += step_time
            return

        self.clock += step_time
        still_running = []
        for req in self.running:
            req.remaining_tokens -= 1
            if req.remaining_tokens <= 0:
                req.finish_time = self.clock
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running

    def run_until_done(self, all_requests: list[SchedulerRequest]):
        for req in sorted(all_requests, key=lambda r: r.arrival_time):
            self.add_request(req)
        while self.running or self.waiting:
            self.step()


class PriorityScheduler:
    """Priority: higher-priority requests preempt lower-priority ones."""
    def __init__(self, max_batch: int = 8):
        self.max_batch = max_batch
        self.running: list[SchedulerRequest] = []
        self.waiting: list[SchedulerRequest] = []  # sorted by priority
        self.finished: list[SchedulerRequest] = []
        self.clock = 0.0

    def add_request(self, req: SchedulerRequest):
        self.waiting.append(req)
        self.waiting.sort(key=lambda r: (-r.priority, r.arrival_time))

    def step(self, step_time: float = 0.020):
        # Fill from waiting, highest priority first
        while len(self.running) < self.max_batch and self.waiting:
            req = self.waiting.pop(0)
            if not req.started:
                req.started = True
                req.start_time = self.clock
            self.running.append(req)

        # If a higher-priority request is waiting and batch is full,
        # preempt the lowest-priority running sequence
        if self.waiting and len(self.running) == self.max_batch:
            lowest_running = min(self.running, key=lambda r: r.priority)
            highest_waiting = self.waiting[0]
            if highest_waiting.priority > lowest_running.priority:
                self.running.remove(lowest_running)
                self.waiting.append(lowest_running)
                self.waiting.sort(key=lambda r: (-r.priority, r.arrival_time))
                req = self.waiting.pop(0)
                req.start_time = self.clock
                self.running.append(req)

        if not self.running:
            self.clock += step_time
            return

        self.clock += step_time
        still_running = []
        for req in self.running:
            req.remaining_tokens -= 1
            if req.remaining_tokens <= 0:
                req.finish_time = self.clock
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running

    def run_until_done(self, all_requests: list[SchedulerRequest]):
        for req in sorted(all_requests, key=lambda r: r.arrival_time):
            self.add_request(req)
        while self.running or self.waiting:
            self.step()


def demo5_preemption_policy():
    """
    Compare FIFO vs priority-aware scheduling.
    Measure head-of-line blocking reduction for high-priority requests.
    """
    print()
    print(SEPARATOR)
    print("Demo 5: Scheduler Preemption — FIFO vs Priority-Aware")
    print(SEPARATOR)

    random.seed(7)
    NUM_REQUESTS = 40
    STEP_TIME = 0.020  # 20ms per step

    def make_requests() -> list[SchedulerRequest]:
        requests = []
        for i in range(NUM_REQUESTS):
            # 80% normal priority, 15% high, 5% urgent
            r = random.random()
            if r < 0.05:
                priority = 2
            elif r < 0.20:
                priority = 1
            else:
                priority = 0
            requests.append(SchedulerRequest(
                request_id=f"req-{i:03d}",
                priority=priority,
                arrival_time=i * 0.050,  # arrive every 50ms
                prompt_tokens=random.randint(64, 512),
                remaining_tokens=random.randint(20, 100),
            ))
        return requests

    def analyze(finished: list[SchedulerRequest], label: str):
        by_priority = collections.defaultdict(list)
        for req in finished:
            wait = req.start_time - req.arrival_time
            latency = req.finish_time - req.arrival_time
            by_priority[req.priority].append((wait, latency))

        print(f"\n  {label}:")
        pname = {0: "Normal", 1: "High", 2: "Urgent"}
        for p in sorted(by_priority.keys(), reverse=True):
            waits, latencies = zip(*by_priority[p])
            print(f"    Priority {p} ({pname[p]:>6}): "
                  f"n={len(waits):>3}  "
                  f"wait P50={sorted(waits)[len(waits)//2]*1000:>6.0f}ms  "
                  f"latency P50={sorted(latencies)[len(latencies)//2]*1000:>6.0f}ms")
        return by_priority

    reqs_fifo = make_requests()
    reqs_prio = [SchedulerRequest(
        request_id=r.request_id, priority=r.priority,
        arrival_time=r.arrival_time, prompt_tokens=r.prompt_tokens,
        remaining_tokens=r.remaining_tokens
    ) for r in reqs_fifo]

    fifo = FIFOScheduler(max_batch=8)
    fifo.run_until_done(reqs_fifo)

    prio = PriorityScheduler(max_batch=8)
    prio.run_until_done(reqs_prio)

    print(f"\n  {NUM_REQUESTS} requests: 5% urgent, 15% high, 80% normal")
    print(f"  Max batch size: 8, step time: {STEP_TIME*1000:.0f}ms")

    fifo_stats = analyze(fifo.finished, "FIFO")
    prio_stats = analyze(prio.finished, "Priority")

    # Check that priority scheduling improves high-priority latency
    if 2 in fifo_stats and 2 in prio_stats:
        fifo_urgent_lat = statistics.mean([lat for _, lat in fifo_stats[2]])
        prio_urgent_lat = statistics.mean([lat for _, lat in prio_stats[2]])
        print(f"\n  Urgent request improvement: "
              f"FIFO avg={fifo_urgent_lat*1000:.0f}ms → "
              f"Priority avg={prio_urgent_lat*1000:.0f}ms")

    assert len(fifo.finished) == NUM_REQUESTS, "FIFO should finish all requests"
    assert len(prio.finished) == NUM_REQUESTS, "Priority scheduler should finish all requests"

    print("\n  [PASS] Preemption policy assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6: V1 Throughput Model
# ─────────────────────────────────────────────────────────────────────────────

def demo6_throughput_model():
    """
    Analytical throughput model comparing V0 and V1 at different
    request rates and batch sizes.
    """
    print()
    print(SEPARATOR)
    print("Demo 6: Throughput Model — V0 vs V1 at Different Load Levels")
    print(SEPARATOR)

    # Model parameters (representative H100 numbers for LLaMA-3 70B TP=8)
    GPU_STEP_MS_PER_BATCH = {
        # batch_size: (prefill_per_token_ms, decode_step_ms)
        1:   (0.12, 8.0),
        8:   (0.10, 10.0),
        16:  (0.09, 12.0),
        32:  (0.08, 15.0),
        64:  (0.07, 18.0),
        128: (0.07, 22.0),
        256: (0.07, 28.0),
    }

    V0_SCHED_OVERHEAD_MS = 1.5   # V0: scheduler in-GIL, per step
    V1_SCHED_OVERHEAD_MS = 0.15  # V1: async ZMQ, amortized by K=5
    V1_K_STEPS = 5               # multi-step K
    PROMPT_TOKENS = 512
    OUTPUT_TOKENS = 128

    def model_throughput(batch_size: int, sched_overhead_ms: float,
                         k_steps: int = 1) -> dict:
        """
        Compute throughput given batch size and scheduling parameters.
        Returns tokens/second and step time.
        """
        if batch_size not in GPU_STEP_MS_PER_BATCH:
            # Interpolate
            keys = sorted(GPU_STEP_MS_PER_BATCH.keys())
            for i in range(len(keys) - 1):
                if keys[i] <= batch_size <= keys[i+1]:
                    lo, hi = keys[i], keys[i+1]
                    t = (batch_size - lo) / (hi - lo)
                    ppt_lo, ds_lo = GPU_STEP_MS_PER_BATCH[lo]
                    ppt_hi, ds_hi = GPU_STEP_MS_PER_BATCH[hi]
                    ppt = ppt_lo + t * (ppt_hi - ppt_lo)
                    ds = ds_lo + t * (ds_hi - ds_lo)
                    break
        else:
            ppt, ds = GPU_STEP_MS_PER_BATCH[batch_size]

        # Prefill time for one batch
        prefill_ms = PROMPT_TOKENS * ppt
        # Decode time: OUTPUT_TOKENS steps, K per scheduler call
        decode_total_ms = (OUTPUT_TOKENS * ds)
        # Scheduler overhead: invoked once per K decode steps + once for prefill
        n_sched_calls = OUTPUT_TOKENS / k_steps + 1
        sched_total_ms = n_sched_calls * sched_overhead_ms
        # Total time per batch of sequences
        total_ms = prefill_ms + decode_total_ms + sched_total_ms
        # Tokens generated per batch
        tokens_out = batch_size * OUTPUT_TOKENS
        tps = tokens_out / (total_ms / 1000)

        return {
            "batch_size": batch_size,
            "total_ms": total_ms,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_total_ms,
            "sched_ms": sched_total_ms,
            "tps": tps,
            "sched_overhead_pct": sched_total_ms / total_ms * 100,
        }

    batch_sizes = [1, 8, 32, 128, 256]

    print(f"\n  Prompt: {PROMPT_TOKENS} tokens, Output: {OUTPUT_TOKENS} tokens")
    print(f"  V1 uses K={V1_K_STEPS} multi-step scheduling")
    print()
    print(f"  {'Batch':>6}  {'V0 TPS':>10}  {'V1 TPS':>10}  {'Gain':>8}  {'V0 Sched%':>10}  {'V1 Sched%':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

    v0_results = []
    v1_results = []

    for bs in batch_sizes:
        v0 = model_throughput(bs, V0_SCHED_OVERHEAD_MS, k_steps=1)
        v1 = model_throughput(bs, V1_SCHED_OVERHEAD_MS, k_steps=V1_K_STEPS)
        v0_results.append(v0)
        v1_results.append(v1)
        gain = (v1["tps"] - v0["tps"]) / v0["tps"] * 100
        print(f"  {bs:>6}  {v0['tps']:>10,.0f}  {v1['tps']:>10,.0f}  "
              f"{gain:>7.1f}%  {v0['sched_overhead_pct']:>9.1f}%  {v1['sched_overhead_pct']:>9.1f}%")

    print()
    print("  V1 advantage is largest at small batch sizes where scheduler")
    print("  overhead is a larger fraction of total step time.")

    # Assertions
    for v0, v1 in zip(v0_results, v1_results):
        assert v1["tps"] > v0["tps"], \
            f"V1 should outperform V0 at batch size {v0['batch_size']}"
        assert v1["sched_overhead_pct"] < v0["sched_overhead_pct"], \
            f"V1 scheduler overhead% should be lower at batch {v0['batch_size']}"

    print("\n  [PASS] Throughput model assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 7: Migration Checklist
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlagSpec:
    v0_flag: str
    v1_flag: Optional[str]
    status: str           # "kept", "renamed", "removed", "new", "default_changed"
    notes: str
    default_v0: str = ""
    default_v1: str = ""

def demo7_migration_checklist():
    """
    Simulate V0 → V1 API compatibility check.
    Parse a configuration, identify deprecated/changed flags, emit migration report.
    """
    print()
    print(SEPARATOR)
    print("Demo 7: Migration Checklist — V0 → V1 Flag Compatibility")
    print(SEPARATOR)

    FLAG_DATABASE: list[FlagSpec] = [
        FlagSpec("--enable-prefix-caching", None, "default_changed",
                 "Always on in V1. Flag ignored but not an error.", "False", "True (forced)"),
        FlagSpec("--disable-sliding-window", None, "removed",
                 "V1 handles sliding window attention internally. Remove this flag.", "", ""),
        FlagSpec("--use-v2-block-manager", None, "default_changed",
                 "Default is now True. Explicit --use-v2-block-manager=True is a no-op.", "False", "True"),
        FlagSpec("--engine-use-ray", None, "removed",
                 "V1 uses ZMQ only. Ray worker mode removed. Remove this flag.", "", ""),
        FlagSpec("--swap-space", "--swap-space", "kept",
                 "Same semantics. CPU swap space in GiB.", "4", "4"),
        FlagSpec("--max-num-seqs", "--max-num-seqs", "kept",
                 "Same semantics. Maximum concurrent sequences.", "256", "256"),
        FlagSpec("--max-num-batched-tokens", "--max-num-batched-tokens", "kept",
                 "Same semantics. More important in V1 for chunked prefill.", "None", "8192"),
        FlagSpec("--num-gpu-blocks-override", "--num-gpu-blocks-override", "kept",
                 "Same semantics.", "", ""),
        FlagSpec("--scheduling-policy", "--scheduling-policy", "new",
                 "New in V1: 'fcfs' (default) or 'priority'.", "N/A", "fcfs"),
        FlagSpec("--num-scheduler-steps", "--num-scheduler-steps", "new",
                 "New in V1: K decode steps per scheduler call.", "N/A", "1"),
        FlagSpec("--max-num-partial-prefills", "--max-num-partial-prefills", "new",
                 "New in V1: concurrent chunked prefill slots.", "N/A", "1"),
        FlagSpec("--tensor-parallel-size", "--tensor-parallel-size", "kept",
                 "Same semantics.", "1", "1"),
        FlagSpec("--pipeline-parallel-size", "--pipeline-parallel-size", "kept",
                 "Same semantics.", "1", "1"),
        FlagSpec("--gpu-memory-utilization", "--gpu-memory-utilization", "kept",
                 "Same semantics.", "0.90", "0.90"),
        FlagSpec("--enforce-eager", "--enforce-eager", "kept",
                 "Still works. Disables CUDA graphs.", "False", "False"),
    ]

    # Simulate a V0 config that a user might have
    simulated_v0_config = {
        "--model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "--tensor-parallel-size": "4",
        "--max-num-seqs": "128",
        "--max-num-batched-tokens": "4096",
        "--gpu-memory-utilization": "0.92",
        "--enable-prefix-caching": None,       # deprecated
        "--disable-sliding-window": None,       # removed
        "--engine-use-ray": None,               # removed
        "--swap-space": "8",
        "--scheduling-policy": None,            # new in V1
        "--num-scheduler-steps": None,          # new in V1
    }

    print(f"\n  Checking V0 configuration against V1 compatibility...")
    print(f"  Model: {simulated_v0_config['--model']}")
    print()

    warnings = []
    errors = []
    suggestions = []

    flag_map = {spec.v0_flag: spec for spec in FLAG_DATABASE}

    for flag, value in simulated_v0_config.items():
        if flag == "--model":
            continue
        spec = flag_map.get(flag)
        if spec is None:
            continue

        if spec.status == "removed":
            errors.append(f"REMOVED:  {flag} — {spec.notes}")
        elif spec.status == "default_changed":
            warnings.append(f"CHANGED:  {flag} — {spec.notes}")
        elif spec.status == "new" and value is None:
            suggestions.append(f"CONSIDER: {spec.v1_flag} — {spec.notes}")
        elif spec.status == "kept":
            pass  # No action needed

    if errors:
        print("  ERRORS (flags that must be removed):")
        for e in errors:
            print(f"    ✗ {e}")
    if warnings:
        print("\n  WARNINGS (behavior changes):")
        for w in warnings:
            print(f"    ⚠ {w}")
    if suggestions:
        print("\n  SUGGESTIONS (new V1 capabilities):")
        for s in suggestions:
            print(f"    + {s}")

    # Generate V1 command
    print("\n  Suggested V1 launch command:")
    v1_flags = []
    for flag, value in simulated_v0_config.items():
        if flag == "--model":
            v1_flags.append(f"--model {value}")
            continue
        spec = flag_map.get(flag)
        if spec is None:
            continue
        if spec.status in ("removed", "default_changed"):
            continue  # drop these
        if spec.status == "kept" and value is not None:
            v1_flags.append(f"{spec.v1_flag} {value}")

    # Add recommended V1 options
    v1_flags.append("--num-scheduler-steps 5")
    v1_flags.append("--max-num-batched-tokens 8192")

    cmd = "vllm serve \\\n    " + " \\\n    ".join(v1_flags)
    print(f"\n    {cmd}")

    # Compatibility score
    total_flags = len([k for k in simulated_v0_config if k != "--model"])
    problem_flags = len(errors) + len(warnings)
    compat_score = (total_flags - problem_flags) / total_flags * 100

    print(f"\n  Compatibility score: {compat_score:.0f}% ({total_flags - problem_flags}/{total_flags} flags compatible)")
    print(f"  Action required: {len(errors)} removal(s), {len(warnings)} behavioral change(s)")

    # Summary table
    print()
    print(f"  {'Flag':<35} {'V0→V1 Status':<16} {'Action':<12}")
    print(f"  {'-'*35} {'-'*16} {'-'*12}")
    for spec in FLAG_DATABASE:
        status_label = {
            "kept": "Kept",
            "default_changed": "Behavior changed",
            "removed": "REMOVED",
            "new": "New in V1",
        }.get(spec.status, spec.status)
        action = {
            "kept": "No change",
            "default_changed": "Review",
            "removed": "Delete flag",
            "new": "Add flag",
        }.get(spec.status, "")
        print(f"  {spec.v0_flag:<35} {status_label:<16} {action:<12}")

    # Assertions
    assert len(errors) > 0, "Should detect removed flags"
    assert len(warnings) > 0, "Should detect changed defaults"
    assert compat_score < 100, "Should find at least some incompatibilities"
    assert compat_score > 50, "Most flags should be compatible"

    print("\n  [PASS] Migration checklist assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 70)
    print("  Chapter 40: The vLLM V1 Architecture")
    print("  Close to the Metal: LLM Inference from First Principles")
    print("=" * 70)

    random.seed(42)

    demo1_v0_vs_v1_architecture()
    demo2_zmq_message_passing()
    demo3_block_hash_deduplication()
    demo4_multi_step_scheduling()
    demo5_preemption_policy()
    demo6_throughput_model()
    demo7_migration_checklist()

    print()
    print(SEPARATOR)
    print("All 7 demos completed. All assertions passed.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
```

**Run:**
```bash
python vllm_v1_demo.py
```

---

## C++

```cpp
// vllm_v1_demo.cpp
// Chapter 40 — The vLLM V1 Architecture
//
// Seven self-contained demos covering:
//   Demo 1: V0 vs V1 architecture model
//   Demo 2: ZMQ message passing model
//   Demo 3: Block hash deduplication
//   Demo 4: Multi-step scheduling
//   Demo 5: Scheduler preemption policy
//   Demo 6: V1 throughput model
//   Demo 7: Migration checklist
//
// Compile: g++ -std=c++17 -O2 -o vllm_v1_demo vllm_v1_demo.cpp -lm
// Run:     ./vllm_v1_demo

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static const std::string SEP(70, '=');

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

static std::mt19937 g_rng(42);

static double uniform(double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(g_rng);
}

static int uniform_int(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(g_rng);
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    return v[idx];
}

static double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// Simple non-cryptographic hash (FNV-1a variant) for block content
static uint64_t fnv1a_hash(const std::vector<int>& tokens) {
    uint64_t hash = 14695981039346656037ULL;
    for (int t : tokens) {
        uint8_t bytes[4];
        memcpy(bytes, &t, 4);
        for (int i = 0; i < 4; i++) {
            hash ^= static_cast<uint64_t>(bytes[i]);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}


// ─────────────────────────────────────────────────────────────────────────────
// Demo 1: V0 vs V1 Architecture Model
// ─────────────────────────────────────────────────────────────────────────────

struct EngineSimResult {
    double total_time_ms;
    double avg_latency_ms;
    double throughput_rps;
    double avg_queue_depth;
};

static EngineSimResult simulate_v0(int num_requests,
                                    double gpu_step_us,
                                    double sched_overhead_us,
                                    double arrival_interval_us)
{
    // V0: synchronous serial pipeline — schedule + GPU + output in sequence
    double clock = 0.0;
    std::vector<double> completions;
    std::vector<double> arrival_times;
    arrival_times.reserve(num_requests);
    for (int i = 0; i < num_requests; i++)
        arrival_times.push_back(i * arrival_interval_us);

    std::deque<int> waiting;
    std::vector<double> queue_depths;
    int idx = 0;

    while (static_cast<int>(completions.size()) < num_requests) {
        // Accept arrived requests
        while (idx < num_requests && arrival_times[idx] <= clock)
            waiting.push_back(idx++);

        if (waiting.empty()) {
            clock += arrival_interval_us;
            continue;
        }

        queue_depths.push_back(static_cast<double>(waiting.size()));
        waiting.pop_front();

        // V0 step: schedule (CPU, blocking) + GPU + output (CPU, blocking)
        clock += sched_overhead_us * 0.5;  // scheduler phase
        clock += gpu_step_us;               // GPU forward pass
        clock += sched_overhead_us * 0.5;  // output processing

        completions.push_back(clock);

        // Accept more arrivals
        while (idx < num_requests && arrival_times[idx] <= clock)
            waiting.push_back(idx++);
    }

    double total_time = completions.back();
    double sum_lat = 0.0;
    for (int i = 0; i < num_requests; i++)
        sum_lat += completions[i] - arrival_times[i];

    double avg_qd = queue_depths.empty() ? 0.0 :
        std::accumulate(queue_depths.begin(), queue_depths.end(), 0.0) / queue_depths.size();

    return {
        total_time / 1000.0,
        (sum_lat / num_requests) / 1000.0,
        num_requests / (total_time / 1e6),
        avg_qd
    };
}

static EngineSimResult simulate_v1(int num_requests,
                                    double gpu_step_us,
                                    double sched_overhead_us,
                                    double arrival_interval_us)
{
    // V1: scheduler runs in separate process, hides overhead behind GPU
    double clock = 0.0;
    std::vector<double> completions;
    std::vector<double> arrival_times;
    for (int i = 0; i < num_requests; i++)
        arrival_times.push_back(i * arrival_interval_us);

    std::deque<int> waiting;
    std::vector<double> queue_depths;
    int idx = 0;

    while (static_cast<int>(completions.size()) < num_requests) {
        while (idx < num_requests && arrival_times[idx] <= clock)
            waiting.push_back(idx++);

        if (waiting.empty()) {
            clock += 500.0;
            continue;
        }

        queue_depths.push_back(static_cast<double>(waiting.size()));
        waiting.pop_front();

        // V1: scheduler runs concurrently with previous GPU step
        // Effective step time = max(sched, gpu), not sched+gpu
        double step_time = std::max(sched_overhead_us, gpu_step_us);
        clock += step_time;

        completions.push_back(clock);

        while (idx < num_requests && arrival_times[idx] <= clock)
            waiting.push_back(idx++);
    }

    double total_time = completions.back();
    double sum_lat = 0.0;
    for (int i = 0; i < num_requests; i++)
        sum_lat += completions[i] - arrival_times[i];

    double avg_qd = queue_depths.empty() ? 0.0 :
        std::accumulate(queue_depths.begin(), queue_depths.end(), 0.0) / queue_depths.size();

    return {
        total_time / 1000.0,
        (sum_lat / num_requests) / 1000.0,
        num_requests / (total_time / 1e6),
        avg_qd
    };
}

static void demo1_v0_vs_v1_architecture() {
    std::cout << "\n" << SEP << "\n";
    std::cout << "Demo 1: V0 vs V1 Architecture — Synchronous vs Async Lifecycle\n";
    std::cout << SEP << "\n";

    const int    NUM_REQUESTS         = 20;
    const double GPU_STEP_US          = 20000.0;
    const double SCHED_OVERHEAD_V0_US = 1500.0;
    const double SCHED_OVERHEAD_V1_US = 150.0;
    const double ARRIVAL_INTERVAL_US  = 3000.0;

    auto v0 = simulate_v0(NUM_REQUESTS, GPU_STEP_US, SCHED_OVERHEAD_V0_US, ARRIVAL_INTERVAL_US);
    auto v1 = simulate_v1(NUM_REQUESTS, GPU_STEP_US, SCHED_OVERHEAD_V1_US, ARRIVAL_INTERVAL_US);

    std::cout << "\n  Simulation: " << NUM_REQUESTS << " requests, "
              << "arrival every " << ARRIVAL_INTERVAL_US/1000.0 << "ms\n";
    std::cout << "  GPU step: " << GPU_STEP_US/1000.0 << "ms  "
              << "Scheduler overhead  V0=" << SCHED_OVERHEAD_V0_US/1000.0 << "ms  "
              << "V1=" << SCHED_OVERHEAD_V1_US/1000.0 << "ms\n\n";

    auto pct_improve = [](double old_val, double new_val) {
        return (old_val - new_val) / old_val * 100.0;
    };
    auto pct_improve_inv = [](double old_val, double new_val) {
        return (new_val - old_val) / old_val * 100.0;
    };

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  " << std::left << std::setw(28) << "Metric"
              << std::right << std::setw(12) << "V0"
              << std::setw(12) << "V1"
              << std::setw(14) << "Improvement" << "\n";
    std::cout << "  " << std::string(28, '-') << std::string(12, '-')
              << std::string(12, '-') << std::string(14, '-') << "\n";

    std::cout << "  " << std::left << std::setw(28) << "Total wall time (ms)"
              << std::right << std::setw(12) << v0.total_time_ms
              << std::setw(12) << v1.total_time_ms
              << std::setw(13) << pct_improve(v0.total_time_ms, v1.total_time_ms)
              << "%\n";
    std::cout << "  " << std::left << std::setw(28) << "Avg latency (ms)"
              << std::right << std::setw(12) << v0.avg_latency_ms
              << std::setw(12) << v1.avg_latency_ms
              << std::setw(13) << pct_improve(v0.avg_latency_ms, v1.avg_latency_ms)
              << "%\n";
    std::cout << "  " << std::left << std::setw(28) << "Throughput (req/s)"
              << std::right << std::setw(12) << v0.throughput_rps
              << std::setw(12) << v1.throughput_rps
              << std::setw(13) << pct_improve_inv(v0.throughput_rps, v1.throughput_rps)
              << "%\n";
    std::cout << "  " << std::left << std::setw(28) << "Avg queue depth"
              << std::right << std::setw(12) << v0.avg_queue_depth
              << std::setw(12) << v1.avg_queue_depth << "\n";

    assert(v1.total_time_ms <= v0.total_time_ms && "V1 should complete faster");
    assert(v1.avg_latency_ms <= v0.avg_latency_ms && "V1 latency should be lower");
    assert(v1.throughput_rps >= v0.throughput_rps && "V1 throughput should be higher");

    std::cout << "\n  [PASS] All V0 vs V1 assertions passed\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// Demo 2: ZMQ Message Passing Model
// ─────────────────────────────────────────────────────────────────────────────

struct MsgRequest {
    std::string request_id;
    int prompt_tokens;
    int max_new_tokens;
    double arrival_time;  // microseconds from simulation start
};

static void demo2_zmq_message_passing() {
    std::cout << "\n" << SEP << "\n";
    std::cout << "Demo 2: ZMQ Message Passing Model — Scheduler<->Worker Exchange\n";
    std::cout << SEP << "\n";

    // Simulation: discrete-event, single-threaded
    const int    NUM_REQUESTS        = 100;
    const int    BATCH_SIZE          = 8;
    const double ARRIVAL_INTERVAL_US = 20000.0;  // 50 req/s = 20ms between
    const double ZMQ_LATENCY_US      = 10.0;
    const double SCHED_OVERHEAD_US   = 200.0;
    const double GPU_STEP_US         = 20000.0;

    // Generate requests
    std::vector<MsgRequest> requests;
    for (int i = 0; i < NUM_REQUESTS; i++) {
        requests.push_back({
            "req-" + std::to_string(i),
            uniform_int(32, 256),
            uniform_int(50, 200),
            i * ARRIVAL_INTERVAL_US
        });
    }

    // Simulate the pipeline
    std::deque<MsgRequest> waiting;
    std::vector<double> rtts;
    std::vector<int> batch_sizes;
    int req_idx = 0;
    double clock = 0.0;
    int total_served = 0;
    int batches_sent = 0;

    while (total_served < NUM_REQUESTS) {
        // Accept arrived requests
        while (req_idx < NUM_REQUESTS && requests[req_idx].arrival_time <= clock)
            waiting.push_back(requests[req_idx++]);

        if (waiting.empty()) {
            clock += 1000.0;  // advance 1ms
            continue;
        }

        // Build batch
        int batch_n = std::min(BATCH_SIZE, static_cast<int>(waiting.size()));
        std::vector<MsgRequest> batch;
        for (int i = 0; i < batch_n; i++) {
            batch.push_back(waiting.front());
            waiting.pop_front();
        }
        batch_sizes.push_back(batch_n);
        batches_sent++;

        // ZMQ send to worker + scheduler overhead
        clock += ZMQ_LATENCY_US + SCHED_OVERHEAD_US;

        // GPU forward pass
        clock += GPU_STEP_US;

        // ZMQ send back to scheduler/API server
        clock += ZMQ_LATENCY_US;

        // Record RTTs
        for (const auto& req : batch) {
            rtts.push_back((clock - req.arrival_time) / 1000.0);
            total_served++;
        }

        // Accept more arrivals
        while (req_idx < NUM_REQUESTS && requests[req_idx].arrival_time <= clock)
            waiting.push_back(requests[req_idx++]);
    }

    std::cout << "\n  Requests: " << NUM_REQUESTS << ", batch size: " << BATCH_SIZE << "\n";
    std::cout << "  Arrival interval: " << ARRIVAL_INTERVAL_US/1000.0 << "ms\n";
    std::cout << "  ZMQ latency: " << ZMQ_LATENCY_US << "us each way\n";
    std::cout << "  Batches formed: " << batches_sent << "\n";
    std::cout << "  Requests served: " << total_served << "\n\n";

    if (!rtts.empty()) {
        std::cout << "  Round-trip latency (ms):\n";
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "    P50: " << percentile(rtts, 0.50) << "ms\n";
        std::cout << "    P95: " << percentile(rtts, 0.95) << "ms\n";
        std::cout << "    P99: " << percentile(rtts, 0.99) << "ms\n";
    }
    if (!batch_sizes.empty()) {
        double mean_bs = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0.0)
                         / batch_sizes.size();
        int max_bs = *std::max_element(batch_sizes.begin(), batch_sizes.end());
        int min_bs = *std::min_element(batch_sizes.begin(), batch_sizes.end());
        std::cout << "\n  Batch sizes: mean=" << std::fixed << std::setprecision(1)
                  << mean_bs << "  max=" << max_bs << "  min=" << min_bs << "\n";
    }

    assert(batches_sent > 0 && "Should have formed at least one batch");
    assert(total_served == NUM_REQUESTS && "Should have served all requests");
    assert(*std::max_element(batch_sizes.begin(), batch_sizes.end()) <= BATCH_SIZE
           && "Batch size should not exceed limit");

    std::cout << "\n  [PASS] ZMQ message passing assertions passed\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// Demo 3: Block Hash Deduplication
// ─────────────────────────────────────────────────────────────────────────────

struct KVBlock {
    int block_id;
    std::vector<int> token_ids;
    uint64_t content_hash;
    int ref_count;
    bool is_full;
};

class HashPrefixCache {
public:
    static constexpr int BLOCK_SIZE = 16;

    explicit HashPrefixCache(int max_blocks = 512)
        : max_blocks_(max_blocks), next_id_(0) {}

    // Returns (allocated blocks list, prefill_tokens_needed)
    std::pair<std::vector<int>, int> get_or_allocate(const std::vector<int>& tokens) {
        std::vector<int> block_ids;
        int prefill_needed = 0;
        int n_full = static_cast<int>(tokens.size()) / BLOCK_SIZE;

        for (int i = 0; i < n_full; i++) {
            std::vector<int> chunk(tokens.begin() + i * BLOCK_SIZE,
                                   tokens.begin() + (i + 1) * BLOCK_SIZE);
            uint64_t h = fnv1a_hash(chunk);
            auto it = hash_table_.find(h);
            if (it != hash_table_.end()) {
                // Cache HIT
                blocks_[it->second].ref_count++;
                // Move to back of LRU
                lru_update(it->second);
                block_ids.push_back(it->second);
                stats_hits_++;
            } else {
                // Cache MISS
                int bid = allocate_block(chunk, h, true);
                block_ids.push_back(bid);
                prefill_needed += BLOCK_SIZE;
                stats_misses_++;
            }
        }

        // Partial tail block
        int tail_start = n_full * BLOCK_SIZE;
        if (tail_start < static_cast<int>(tokens.size())) {
            std::vector<int> tail(tokens.begin() + tail_start, tokens.end());
            uint64_t h = fnv1a_hash(tail);
            int bid = allocate_block(tail, h, false);
            block_ids.push_back(bid);
            prefill_needed += static_cast<int>(tail.size());
        }

        return {block_ids, prefill_needed};
    }

    void release(const std::vector<int>& block_ids) {
        for (int bid : block_ids) {
            auto it = blocks_.find(bid);
            if (it != blocks_.end()) {
                it->second.ref_count = std::max(0, it->second.ref_count - 1);
            }
        }
    }

    double hit_rate() const {
        int total = stats_hits_ + stats_misses_;
        return total > 0 ? static_cast<double>(stats_hits_) / total : 0.0;
    }

    int hits() const { return stats_hits_; }
    int misses() const { return stats_misses_; }
    int evictions() const { return stats_evictions_; }
    int allocations() const { return stats_allocs_; }

private:
    int allocate_block(const std::vector<int>& tokens, uint64_t hash, bool is_full) {
        if (static_cast<int>(blocks_.size()) >= max_blocks_) {
            evict_lru();
        }
        KVBlock blk;
        blk.block_id = next_id_++;
        blk.token_ids = tokens;
        blk.content_hash = hash;
        blk.ref_count = 1;
        blk.is_full = is_full;
        blocks_[blk.block_id] = blk;
        if (is_full) hash_table_[hash] = blk.block_id;
        lru_order_.push_back(blk.block_id);
        stats_allocs_++;
        return blk.block_id;
    }

    void evict_lru() {
        for (auto it = lru_order_.begin(); it != lru_order_.end(); ++it) {
            int bid = *it;
            if (blocks_.count(bid) && blocks_[bid].ref_count == 0) {
                hash_table_.erase(blocks_[bid].content_hash);
                blocks_.erase(bid);
                lru_order_.erase(it);
                stats_evictions_++;
                return;
            }
        }
    }

    void lru_update(int bid) {
        lru_order_.erase(std::remove(lru_order_.begin(), lru_order_.end(), bid),
                         lru_order_.end());
        lru_order_.push_back(bid);
    }

    int max_blocks_;
    int next_id_;
    std::unordered_map<int, KVBlock> blocks_;
    std::unordered_map<uint64_t, int> hash_table_;
    std::deque<int> lru_order_;
    int stats_hits_    = 0;
    int stats_misses_  = 0;
    int stats_evictions_ = 0;
    int stats_allocs_  = 0;
};

static void demo3_block_hash_deduplication() {
    std::cout << "\n" << SEP << "\n";
    std::cout << "Demo 3: Block Hash Deduplication — Prefix Cache Hit/Miss Rates\n";
    std::cout << SEP << "\n";

    const int SYSTEM_PROMPT_TOKENS = 64;
    const int NUM_REQUESTS         = 100;
    const int VOCAB_SIZE           = 32000;

    // Fixed system prompt
    std::vector<int> system_prompt;
    for (int i = 0; i < SYSTEM_PROMPT_TOKENS; i++)
        system_prompt.push_back(100 + i);

    HashPrefixCache cache(512);

    int total_tokens_needed = 0;
    int total_prefill_done  = 0;
    std::vector<std::vector<int>> all_block_ids;

    for (int i = 0; i < NUM_REQUESTS; i++) {
        int user_len = uniform_int(10, 50);
        std::vector<int> tokens;

        if (i % 10 == 0) {
            // Fully unique prompt
            for (int j = 0; j < user_len + SYSTEM_PROMPT_TOKENS; j++)
                tokens.push_back(uniform_int(500, VOCAB_SIZE));
        } else {
            // Shared system prompt + unique user message
            tokens = system_prompt;
            for (int j = 0; j < user_len; j++)
                tokens.push_back(uniform_int(200, 500));
        }

        total_tokens_needed += static_cast<int>(tokens.size());
        auto [block_ids, prefill] = cache.get_or_allocate(tokens);
        total_prefill_done += prefill;
        all_block_ids.push_back(block_ids);

        if (i >= 5) {
            cache.release(all_block_ids[i - 5]);
        }
    }

    double hit_rate = cache.hit_rate();
    double savings_pct = (1.0 - static_cast<double>(total_prefill_done) / total_tokens_needed) * 100.0;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n  Workload: " << NUM_REQUESTS << " requests, "
              << "system_prompt=" << SYSTEM_PROMPT_TOKENS << " tokens\n";
    std::cout << "  90% share system prompt, 10% fully unique\n\n";
    std::cout << "  Cache statistics:\n";
    std::cout << "    Block hits:          " << cache.hits() << "\n";
    std::cout << "    Block misses:        " << cache.misses() << "\n";
    std::cout << "    Cache hit rate:      " << hit_rate * 100.0 << "%\n";
    std::cout << "    Evictions:           " << cache.evictions() << "\n";
    std::cout << "    Total allocations:   " << cache.allocations() << "\n\n";
    std::cout << "  Prefill savings:\n";
    std::cout << "    Total tokens in prompts:   " << total_tokens_needed << "\n";
    std::cout << "    Tokens actually prefilled: " << total_prefill_done << "\n";
    std::cout << "    Prefill savings:           " << savings_pct << "%\n";

    assert(hit_rate > 0.0 && "Should have cache hits");
    assert(savings_pct > 10.0 && "Should save at least 10% on prefill");
    assert(cache.allocations() > 0 && "Should allocate blocks");

    std::cout << "\n  [PASS] Block hash deduplication assertions passed\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// Demo 4: Multi-Step Scheduling
// ─────────────────────────────────────────────────────────────────────────────

struct KStepResult {
    int k;
    double total_time_ms;
    int sched_invocations;
    double throughput_tps;
    double overhead_pct;
};

static KStepResult simulate_k_step(int k, double gpu_step_us,
                                    double sched_overhead_us,
                                    int total_tokens)
{
    int tokens_generated  = 0;
    double total_time_us  = 0.0;
    int sched_invocations = 0;

    while (tokens_generated < total_tokens) {
        total_time_us += sched_overhead_us;
        sched_invocations++;

        int steps = std::min(k, total_tokens - tokens_generated);
        total_time_us += steps * gpu_step_us;
        tokens_generated += steps;
    }

    double throughput = tokens_generated / (total_time_us / 1e6);
    double overhead_pct = (sched_invocations * sched_overhead_us) / total_time_us * 100.0;

    return {k, total_time_us / 1000.0, sched_invocations, throughput, overhead_pct};
}

static void demo4_multi_step_scheduling() {
    std::cout << "\n" << SEP << "\n";
    std::cout << "Demo 4: Multi-Step Scheduling — Overhead Amortization vs K\n";
    std::cout << SEP << "\n";

    const double GPU_STEP_US       = 5000.0;
    const double SCHED_OVERHEAD_US = 500.0;
    const int    TOTAL_TOKENS      = 5000;

    std::vector<int> k_values = {1, 2, 5, 10, 20, 50};
    std::vector<KStepResult> results;
    for (int k : k_values)
        results.push_back(simulate_k_step(k, GPU_STEP_US, SCHED_OVERHEAD_US, TOTAL_TOKENS));

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n  GPU step: " << GPU_STEP_US/1000.0 << "ms, "
              << "Scheduler overhead: " << SCHED_OVERHEAD_US/1000.0 << "ms\n";
    std::cout << "  Total tokens: " << TOTAL_TOKENS << "\n\n";

    std::cout << "  " << std::right << std::setw(4) << "K"
              << std::setw(14) << "Total time"
              << std::setw(13) << "Sched calls"
              << std::setw(15) << "Throughput"
              << std::setw(11) << "Overhead" << "\n";
    std::cout << "  " << std::string(4, '-') << std::string(14, '-')
              << std::string(13, '-') << std::string(15, '-')
              << std::string(11, '-') << "\n";

    for (const auto& r : results) {
        std::cout << "  " << std::setw(4) << r.k
                  << std::setw(11) << static_cast<int>(r.total_time_ms) << "ms"
                  << std::setw(13) << r.sched_invocations
                  << std::setw(12) << static_cast<int>(r.throughput_tps) << "/s"
                  << std::setw(10) << r.overhead_pct << "%\n";
    }

    // Show amortization bar chart
    std::cout << "\n  Throughput gain vs K=1:\n";
    double baseline_tps = results[0].throughput_tps;
    for (size_t i = 1; i < results.size(); i++) {
        double gain = (results[i].throughput_tps - baseline_tps) / baseline_tps * 100.0;
        std::string bar(static_cast<int>(gain / 2), '#');
        std::cout << "    K=" << std::setw(2) << results[i].k
                  << ": +" << std::setw(4) << std::fixed << std::setprecision(1)
                  << gain << "% " << bar << "\n";
    }

    // Assertions
    for (size_t i = 1; i < results.size(); i++) {
        assert(results[i].throughput_tps >= results[i-1].throughput_tps
               && "Throughput should be non-decreasing with K");
        assert(results[i].overhead_pct <= results[i-1].overhead_pct
               && "Overhead fraction should decrease with K");
    }
    assert(results.back().overhead_pct < results.front().overhead_pct / 5.0
           && "Overhead at K=50 should be <1/5 of K=1");

    std::cout << "\n  [PASS] Multi-step scheduling assertions passed\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// Demo 5: Scheduler Preemption Policy
// ─────────────────────────────────────────────────────────────────────────────

struct SchedReq {
    std::string request_id;
    int priority;           // 0=normal, 1=high, 2=urgent
    double arrival_time;
    int remaining_tokens;
    bool started = false;
    double start_time  = 0.0;
    double finish_time = 0.0;
};

struct ByPriorityThenArrival {
    bool operator()(const SchedReq* a, const SchedReq* b) const {
        if (a->priority != b->priority) return a->priority < b->priority; // higher = better
        return a->arrival_time > b->arrival_time; // earlier arrival = better
    }
};

static std::pair<std::map<int, std::vector<double>>, std::map<int, std::vector<double>>>
run_fifo(std::vector<SchedReq> requests, int max_batch, double step_time_ms)
{
    // Returns {priority: [wait_times_ms]}, {priority: [latencies_ms]}
    std::deque<SchedReq*> waiting;
    std::vector<SchedReq*> running;
    std::vector<SchedReq*> finished;

    for (auto& r : requests)
        waiting.push_back(&r);

    double clock = 0.0;
    while (!waiting.empty() || !running.empty()) {
        // Fill running from waiting (FIFO)
        while (static_cast<int>(running.size()) < max_batch && !waiting.empty()) {
            SchedReq* req = waiting.front(); waiting.pop_front();
            if (!req->started) {
                req->started = true;
                req->start_time = clock;
            }
            running.push_back(req);
        }
        if (running.empty()) { clock += step_time_ms; continue; }

        clock += step_time_ms;
        std::vector<SchedReq*> still_running;
        for (auto* req : running) {
            req->remaining_tokens--;
            if (req->remaining_tokens <= 0) {
                req->finish_time = clock;
                finished.push_back(req);
            } else {
                still_running.push_back(req);
            }
        }
        running = still_running;
    }

    std::map<int, std::vector<double>> waits, lats;
    for (auto* req : finished) {
        double wait = req->start_time - req->arrival_time;
        double lat  = req->finish_time - req->arrival_time;
        waits[req->priority].push_back(wait);
        lats[req->priority].push_back(lat);
    }
    return {waits, lats};
}

static std::pair<std::map<int, std::vector<double>>, std::map<int, std::vector<double>>>
run_priority(std::vector<SchedReq> requests, int max_batch, double step_time_ms)
{
    // Priority queue: highest priority, then earliest arrival
    auto cmp = [](SchedReq* a, SchedReq* b) {
        if (a->priority != b->priority) return a->priority < b->priority;
        return a->arrival_time > b->arrival_time;
    };
    std::priority_queue<SchedReq*, std::vector<SchedReq*>, decltype(cmp)> waiting(cmp);
    std::vector<SchedReq*> running;
    std::vector<SchedReq*> finished;

    for (auto& r : requests) waiting.push(&r);

    double clock = 0.0;
    while (!waiting.empty() || !running.empty()) {
        // Fill from waiting, highest priority first
        while (static_cast<int>(running.size()) < max_batch && !waiting.empty()) {
            SchedReq* req = waiting.top(); waiting.pop();
            if (!req->started) {
                req->started = true;
                req->start_time = clock;
            }
            running.push_back(req);
        }

        // Preemption: if a higher-priority req is waiting and batch is full
        if (!waiting.empty() && static_cast<int>(running.size()) == max_batch) {
            SchedReq* highest_waiting = waiting.top();
            auto min_it = std::min_element(running.begin(), running.end(),
                [](SchedReq* a, SchedReq* b) { return a->priority < b->priority; });
            SchedReq* lowest_running = *min_it;
            if (highest_waiting->priority > lowest_running->priority) {
                running.erase(min_it);
                waiting.pop();
                waiting.push(lowest_running);
                highest_waiting->start_time = clock;
                if (!highest_waiting->started) highest_waiting->started = true;
                running.push_back(highest_waiting);
            }
        }

        if (running.empty()) { clock += step_time_ms; continue; }

        clock += step_time_ms;
        std::vector<SchedReq*> still_running;
        for (auto* req : running) {
            req->remaining_tokens--;
            if (req->remaining_tokens <= 0) {
                req->finish_time = clock;
                finished.push_back(req);
            } else {
                still_running.push_back(req);
            }
        }
        running = still_running;
    }

    std::map<int, std::vector<double>> waits, lats;
    for (auto* req : finished) {
        double wait = req->start_time - req->arrival_time;
        double lat  = req->finish_time - req->arrival_time;
        waits[req->priority].push_back(wait);
        lats[req->priority].push_back(lat);
    }
    return {waits, lats};
}

static void demo5_preemption_policy() {
    std::cout << "\n" << SEP << "\n";
    std::cout << "Demo 5: Scheduler Preemption — FIFO vs Priority-Aware\n";
    std::cout << SEP << "\n";

    const int    NUM_REQUESTS = 40;
    const int    MAX_BATCH    = 8;
    const double STEP_TIME_MS = 20.0;
    const double ARRIVAL_MS   = 50.0;

    std::vector<SchedReq> requests_orig;
    for (int i = 0; i < NUM_REQUESTS; i++) {
        double r = static_cast<double>(uniform_int(0, 999)) / 1000.0;
        int prio = (r < 0.05) ? 2 : (r < 0.20) ? 1 : 0;
        requests_orig.push_back({
            "req-" + std::to_string(i),
            prio,
            i * ARRIVAL_MS,
            uniform_int(20, 100),
        });
    }

    std::vector<SchedReq> requests_fifo = requests_orig;
    std::vector<SchedReq> requests_prio = requests_orig;

    auto [fifo_waits, fifo_lats] = run_fifo(requests_fifo, MAX_BATCH, STEP_TIME_MS);
    auto [prio_waits, prio_lats] = run_priority(requests_prio, MAX_BATCH, STEP_TIME_MS);

    std::cout << "\n  " << NUM_REQUESTS << " requests: 5% urgent, 15% high, 80% normal\n";
    std::cout << "  Max batch size: " << MAX_BATCH << ", step time: " << STEP_TIME_MS << "ms\n";

    const char* pname[] = {"Normal", "High", "Urgent"};

    auto print_stats = [&](const std::string& label,
                           const std::map<int, std::vector<double>>& lats_map)
    {
        std::cout << "\n  " << label << ":\n";
        for (int p = 2; p >= 0; p--) {
            if (lats_map.find(p) == lats_map.end()) continue;
            auto lats = lats_map.at(p);
            std::cout << "    Priority " << p << " (" << std::setw(6) << pname[p] << "): "
                      << "n=" << std::setw(3) << lats.size()
                      << "  latency P50=" << std::setw(6) << std::fixed << std::setprecision(0)
                      << percentile(lats, 0.50) << "ms\n";
        }
    };

    print_stats("FIFO", fifo_lats);
    print_stats("Priority", prio_lats);

    // Assertions
    int total_fifo = 0, total_prio = 0;
    for (auto& [p, v] : fifo_lats) total_fifo += v.size();
    for (auto& [p, v] : prio_lats) total_prio += v.size();
    assert(total_fifo == NUM_REQUESTS && "FIFO should finish all requests");
    assert(total_prio == NUM_REQUESTS && "Priority should finish all requests");

    std::cout << "\n  [PASS] Preemption policy assertions passed\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// Demo 6: V1 Throughput Model
// ─────────────────────────────────────────────────────────────────────────────

struct ThroughputResult {
    int batch_size;
    double tps;
    double sched_overhead_pct;
    double total_ms;
};

static ThroughputResult model_throughput(int batch_size,
                                          double sched_overhead_ms,
                                          int k_steps,
                                          int prompt_tokens,
                                          int output_tokens)
{
    // Decode step time: approximate linear model
    // Larger batches → higher decode time due to memory bandwidth
    double decode_ms = 8.0 + batch_size * 0.08;
    // Prefill time: scales with prompt tokens (arithmetic-intensive)
    double prefill_ms_per_tok = 0.07 + 1.0 / (batch_size + 10);
    double prefill_ms = prompt_tokens * prefill_ms_per_tok;

    double decode_total_ms  = output_tokens * decode_ms;
    double n_sched_calls    = static_cast<double>(output_tokens) / k_steps + 1.0;
    double sched_total_ms   = n_sched_calls * sched_overhead_ms;
    double total_ms         = prefill_ms + decode_total_ms + sched_total_ms;

    double tokens_out       = static_cast<double>(batch_size * output_tokens);
    double tps              = tokens_out / (total_ms / 1000.0);
    double overhead_pct     = sched_total_ms / total_ms * 100.0;

    return {batch_size, tps, overhead_pct, total_ms};
}

static void demo6_throughput_model() {
    std::cout << "\n" << SEP << "\n";
    std::cout << "Demo 6: Throughput Model — V0 vs V1 at Different Load Levels\n";
    std::cout << SEP << "\n";

    const double V0_SCHED_MS  = 1.5;
    const double V1_SCHED_MS  = 0.15;
    const int    V1_K         = 5;
    const int    PROMPT_TOKS  = 512;
    const int    OUTPUT_TOKS  = 128;

    std::vector<int> batch_sizes = {1, 8, 32, 128, 256};

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n  Prompt: " << PROMPT_TOKS << " tokens, Output: " << OUTPUT_TOKS
              << " tokens, V1 K=" << V1_K << "\n\n";

    std::cout << "  " << std::right << std::setw(7) << "Batch"
              << std::setw(12) << "V0 TPS"
              << std::setw(12) << "V1 TPS"
              << std::setw(10) << "Gain"
              << std::setw(12) << "V0 Sched%"
              << std::setw(12) << "V1 Sched%" << "\n";
    std::cout << "  " << std::string(7, '-') << std::string(12, '-')
              << std::string(12, '-') << std::string(10, '-')
              << std::string(12, '-') << std::string(12, '-') << "\n";

    std::vector<ThroughputResult> v0_res, v1_res;
    for (int bs : batch_sizes) {
        auto v0 = model_throughput(bs, V0_SCHED_MS, 1, PROMPT_TOKS, OUTPUT_TOKS);
        auto v1 = model_throughput(bs, V1_SCHED_MS, V1_K, PROMPT_TOKS, OUTPUT_TOKS);
        v0_res.push_back(v0);
        v1_res.push_back(v1);
        double gain = (v1.tps - v0.tps) / v0.tps * 100.0;
        std::cout << "  " << std::setw(7) << bs
                  << std::setw(10) << static_cast<int>(v0.tps) << "  "
                  << std::setw(10) << static_cast<int>(v1.tps) << "  "
                  << std::setw(7) << gain << "%"
                  << std::setw(11) << v0.sched_overhead_pct << "%"
                  << std::setw(11) << v1.sched_overhead_pct << "%\n";
    }

    std::cout << "\n  V1 advantage is largest at small batch sizes where\n";
    std::cout << "  scheduler overhead dominates total step time.\n";

    for (size_t i = 0; i < v0_res.size(); i++) {
        assert(v1_res[i].tps > v0_res[i].tps
               && "V1 should outperform V0");
        assert(v1_res[i].sched_overhead_pct < v0_res[i].sched_overhead_pct
               && "V1 overhead% should be lower");
    }

    std::cout << "\n  [PASS] Throughput model assertions passed\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// Demo 7: Migration Checklist
// ─────────────────────────────────────────────────────────────────────────────

struct FlagSpec {
    std::string v0_flag;
    std::string v1_flag;
    std::string status;   // kept, default_changed, removed, new
    std::string notes;
};

static void demo7_migration_checklist() {
    std::cout << "\n" << SEP << "\n";
    std::cout << "Demo 7: Migration Checklist — V0 -> V1 Flag Compatibility\n";
    std::cout << SEP << "\n";

    std::vector<FlagSpec> flags = {
        {"--enable-prefix-caching",    "",                          "default_changed",
         "Always on in V1. Cannot be disabled."},
        {"--disable-sliding-window",   "",                          "removed",
         "V1 handles sliding window internally. Remove this flag."},
        {"--use-v2-block-manager",     "",                          "default_changed",
         "Default is now True. Explicit flag is a no-op."},
        {"--engine-use-ray",           "",                          "removed",
         "V1 uses ZMQ only. Remove this flag."},
        {"--swap-space",               "--swap-space",              "kept",
         "Same semantics."},
        {"--max-num-seqs",             "--max-num-seqs",            "kept",
         "Same semantics."},
        {"--max-num-batched-tokens",   "--max-num-batched-tokens",  "kept",
         "More important in V1 for chunked prefill. Recommend 8192."},
        {"--num-gpu-blocks-override",  "--num-gpu-blocks-override", "kept",
         "Same semantics."},
        {"--scheduling-policy",        "--scheduling-policy",       "new",
         "New in V1: 'fcfs' (default) or 'priority'."},
        {"--num-scheduler-steps",      "--num-scheduler-steps",     "new",
         "New in V1: K decode steps per scheduler call. Recommend 5-10."},
        {"--tensor-parallel-size",     "--tensor-parallel-size",    "kept",
         "Same semantics."},
        {"--gpu-memory-utilization",   "--gpu-memory-utilization",  "kept",
         "Same semantics."},
        {"--enforce-eager",            "--enforce-eager",           "kept",
         "Disables CUDA graphs. Same semantics."},
    };

    // Simulate a V0 config
    std::vector<std::pair<std::string,std::string>> v0_config = {
        {"--model",                    "meta-llama/Meta-Llama-3-8B-Instruct"},
        {"--tensor-parallel-size",     "4"},
        {"--max-num-seqs",             "128"},
        {"--max-num-batched-tokens",   "4096"},
        {"--gpu-memory-utilization",   "0.92"},
        {"--enable-prefix-caching",    ""},
        {"--disable-sliding-window",   ""},
        {"--engine-use-ray",           ""},
        {"--swap-space",               "8"},
    };

    std::cout << "\n  Checking V0 configuration...\n";
    std::cout << "  Model: meta-llama/Meta-Llama-3-8B-Instruct\n\n";

    std::map<std::string, FlagSpec> flag_map;
    for (const auto& f : flags)
        flag_map[f.v0_flag] = f;

    std::vector<std::string> errors, warnings, suggestions;

    for (const auto& [flag, val] : v0_config) {
        if (flag == "--model") continue;
        auto it = flag_map.find(flag);
        if (it == flag_map.end()) continue;

        const FlagSpec& spec = it->second;
        if (spec.status == "removed")
            errors.push_back("REMOVED:  " + flag + " — " + spec.notes);
        else if (spec.status == "default_changed")
            warnings.push_back("CHANGED:  " + flag + " — " + spec.notes);
    }

    // Suggest new V1 flags not in V0 config
    for (const auto& f : flags) {
        if (f.status == "new") {
            bool in_v0 = false;
            for (const auto& [k,v] : v0_config) if (k == f.v0_flag) in_v0 = true;
            if (!in_v0) suggestions.push_back("CONSIDER: " + f.v0_flag + " — " + f.notes);
        }
    }

    if (!errors.empty()) {
        std::cout << "  ERRORS (flags that must be removed):\n";
        for (const auto& e : errors) std::cout << "    [X] " << e << "\n";
    }
    if (!warnings.empty()) {
        std::cout << "\n  WARNINGS (behavior changes):\n";
        for (const auto& w : warnings) std::cout << "    [!] " << w << "\n";
    }
    if (!suggestions.empty()) {
        std::cout << "\n  SUGGESTIONS (new V1 capabilities):\n";
        for (const auto& s : suggestions) std::cout << "    [+] " << s << "\n";
    }

    // Generate V1 command
    std::cout << "\n  Suggested V1 launch command:\n";
    std::cout << "    vllm serve \\\n";
    std::cout << "      --model meta-llama/Meta-Llama-3-8B-Instruct \\\n";
    std::cout << "      --tensor-parallel-size 4 \\\n";
    std::cout << "      --max-num-seqs 128 \\\n";
    std::cout << "      --max-num-batched-tokens 8192 \\\n";
    std::cout << "      --gpu-memory-utilization 0.92 \\\n";
    std::cout << "      --swap-space 8 \\\n";
    std::cout << "      --num-scheduler-steps 5 \\\n";
    std::cout << "      --scheduling-policy fcfs\n";

    // Compatibility score
    int total_flags = static_cast<int>(v0_config.size()) - 1;  // exclude --model
    int problem_flags = static_cast<int>(errors.size() + warnings.size());
    double compat_score = (total_flags - problem_flags) * 100.0 / total_flags;

    std::cout << "\n  Compatibility score: " << std::fixed << std::setprecision(0)
              << compat_score << "% ("
              << (total_flags - problem_flags) << "/" << total_flags
              << " flags compatible)\n";
    std::cout << "  Action required: " << errors.size() << " removal(s), "
              << warnings.size() << " behavioral change(s)\n";

    // Full summary table
    std::cout << "\n  " << std::left << std::setw(36) << "Flag"
              << std::setw(18) << "V0->V1 Status"
              << std::setw(14) << "Action" << "\n";
    std::cout << "  " << std::string(36, '-') << std::string(18, '-')
              << std::string(14, '-') << "\n";
    for (const auto& f : flags) {
        std::string status_label = f.status;
        if (f.status == "kept") status_label = "Kept";
        else if (f.status == "default_changed") status_label = "Behavior changed";
        else if (f.status == "removed") status_label = "REMOVED";
        else if (f.status == "new") status_label = "New in V1";

        std::string action = "";
        if (f.status == "kept") action = "No change";
        else if (f.status == "default_changed") action = "Review";
        else if (f.status == "removed") action = "Delete flag";
        else if (f.status == "new") action = "Add flag";

        std::cout << "  " << std::left << std::setw(36) << f.v0_flag
                  << std::setw(18) << status_label
                  << std::setw(14) << action << "\n";
    }

    assert(!errors.empty() && "Should detect removed flags");
    assert(!warnings.empty() && "Should detect changed defaults");
    assert(compat_score < 100.0 && "Should find incompatibilities");
    assert(compat_score > 50.0 && "Most flags should be compatible");

    std::cout << "\n  [PASS] Migration checklist assertions passed\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n" << SEP << "\n";
    std::cout << "  Chapter 40: The vLLM V1 Architecture\n";
    std::cout << "  Close to the Metal: LLM Inference from First Principles\n";
    std::cout << SEP << "\n";

    demo1_v0_vs_v1_architecture();
    demo2_zmq_message_passing();
    demo3_block_hash_deduplication();
    demo4_multi_step_scheduling();
    demo5_preemption_policy();
    demo6_throughput_model();
    demo7_migration_checklist();

    std::cout << "\n" << SEP << "\n";
    std::cout << "All 7 demos completed. All assertions passed.\n";
    std::cout << SEP << "\n";
    return 0;
}
```

**Compile and run:**
```bash
g++ -O2 -std=c++17 -pthread -o vllm_v1_demo vllm_v1_demo.cpp && ./vllm_v1_demo
```
