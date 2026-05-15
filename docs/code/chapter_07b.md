# Code — Chapter 7.5: Continuous Batching

Companion code for **Chapter 7.5: Continuous Batching — The Iteration-Level Scheduling Loop**.

Demos simulate static vs. continuous batching utilization, the full iteration-level
scheduling loop, token budget admission control, prefill/decode interleave step-time,
preemption cost (SWAP vs. RECOMPUTE), max_num_seqs memory tradeoffs, and
scheduler efficiency metrics — all in pure Python/C++ with no GPU required.

---

## Python

```python
"""
continuous_batching_demo.py — Chapter 7.5: Continuous Batching

Demos (all run without GPU — pure-Python simulation):
  Demo 1: Static batching GPU utilization vs. continuous batching
  Demo 2: Iteration-level scheduling loop simulation
  Demo 3: Token budget allocation (prefill admission control)
  Demo 4: Prefill/decode interleave step-time model
  Demo 5: Preemption cost: SWAP vs RECOMPUTE policy
  Demo 6: max_num_seqs vs. memory capacity tradeoff
  Demo 7: Scheduler efficiency metrics

Run: python continuous_batching_demo.py
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

SEP = "─" * 70


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: Static batching waste
# ─────────────────────────────────────────────────────────────────────────────

def demo_static_vs_continuous():
    print(f"\n{'='*70}")
    print("DEMO 1 — Static Batching vs. Continuous Batching GPU utilization")
    print(f"{'='*70}")

    output_lengths = [4, 7, 12, 3, 89, 6, 14, 22]
    n = len(output_lengths)
    max_len = max(output_lengths)
    total_useful = sum(output_lengths)
    static_slots = max_len * n

    print(f"\nBatch of {n} requests, output lengths: {output_lengths}")
    print(f"Max output length:   {max_len} tokens")
    print(f"Total useful work:   {total_useful} token-steps")
    print(f"Static batch slots:  {static_slots} token-steps")
    static_util = total_useful / static_slots
    print(f"Static utilization:  {total_useful}/{static_slots} = {static_util:.1%}")
    print(f"Wasted capacity:     {(1-static_util):.1%}")

    # Continuous batching simulation: as each req finishes, fill the slot
    # Assume unlimited waiting queue; count active slots each step
    active_steps = 0
    n_slots = n
    remaining = sorted(output_lengths, reverse=True)  # longest first
    queue = list(range(100, 200))  # unlimited demand
    random.seed(42)
    slots = list(remaining)

    step = 0
    while any(s > 0 for s in slots):
        active = sum(1 for s in slots if s > 0)
        active_steps += active
        slots = [max(0, s - 1) for s in slots]
        # refill finished slots
        for i, s in enumerate(slots):
            if s == 0 and queue:
                slots[i] = random.randint(3, 40)
                queue.pop(0)
        step += 1
        if step > 200:
            break

    continuous_util = active_steps / (step * n_slots)
    print(f"\nContinuous batching (simulated, {step} steps):")
    print(f"  Active slot-steps: {active_steps}")
    print(f"  Total slots × steps: {step * n_slots}")
    print(f"  Continuous utilization: {continuous_util:.1%}")
    print(f"  Improvement over static: {continuous_util/static_util:.1f}×")

    assert static_util < 0.60, "Static utilization should be low"
    assert continuous_util > 0.85, "Continuous batching should achieve >85% utilization"
    print("\n✓ Assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: Iteration-level scheduling loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Request:
    req_id: str
    input_len: int
    max_output_len: int
    tokens_generated: int = 0
    state: str = "waiting"   # waiting / prefilling / running / finished

    @property
    def is_finished(self):
        return self.tokens_generated >= self.max_output_len


def demo_scheduling_loop():
    print(f"\n{'='*70}")
    print("DEMO 2 — Iteration-Level Scheduling Loop Simulation")
    print(f"{'='*70}")

    random.seed(0)
    waiting_queue: deque = deque([
        Request("R1", input_len=32, max_output_len=10),
        Request("R2", input_len=64, max_output_len=25),
        Request("R3", input_len=16, max_output_len=5),
        Request("R4", input_len=128, max_output_len=40),
        Request("R5", input_len=8, max_output_len=8),
    ])
    running: List[Request] = []
    finished: List[Request] = []

    MAX_NUM_SEQS = 3
    MAX_BATCHED_TOKENS = 256
    step_times_ms = []

    print(f"\nConfig: max_num_seqs={MAX_NUM_SEQS}, max_num_batched_tokens={MAX_BATCHED_TOKENS}")
    print(f"Requests to serve: {len(waiting_queue)}")
    print(SEP)

    for iteration in range(1, 50):
        # Step 1: free finished
        newly_finished = [r for r in running if r.is_finished]
        for r in newly_finished:
            r.state = "finished"
            running.remove(r)
            finished.append(r)

        # Step 4: admit new requests (token budget)
        decode_tokens = len(running)
        budget_remaining = MAX_BATCHED_TOKENS - decode_tokens
        admitted = []

        while waiting_queue and len(running) < MAX_NUM_SEQS:
            candidate = waiting_queue[0]
            if candidate.input_len <= budget_remaining:
                waiting_queue.popleft()
                candidate.state = "running"
                running.append(candidate)
                budget_remaining -= candidate.input_len
                admitted.append(candidate.req_id)
            else:
                break   # can't admit more this iteration

        # Step 6: simulate forward pass
        prefill_tokens = sum(r.input_len for r in admitted_reqs(running, admitted))
        decode_tokens = len(running) - len(admitted)
        total_tokens = prefill_tokens + decode_tokens

        # Step time model: 0.05ms per token + 5ms base
        step_time = 5.0 + 0.05 * total_tokens
        step_times_ms.append(step_time)

        # Advance all running sequences
        for r in running:
            r.tokens_generated += 1

        if iteration <= 8 or len(finished) == 5:
            print(f"Iter {iteration:2d}: running={len(running)}, "
                  f"admitted={admitted}, finished_total={len(finished)}, "
                  f"step={step_time:.1f}ms")

        if len(finished) == 5:
            print(f"\nAll {len(finished)} requests finished at iteration {iteration}")
            break

    avg_step = sum(step_times_ms) / len(step_times_ms)
    print(f"Average step time: {avg_step:.2f} ms over {len(step_times_ms)} iterations")
    assert len(finished) == 5
    print("\n✓ All requests served")


def admitted_reqs(running, admitted_ids):
    return [r for r in running if r.req_id in admitted_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: Token budget admission control
# ─────────────────────────────────────────────────────────────────────────────

def demo_token_budget():
    print(f"\n{'='*70}")
    print("DEMO 3 — Token Budget Admission Control")
    print(f"{'='*70}")

    MAX_BATCHED_TOKENS = 2048
    running_decode_seqs = 32
    waiting = [
        ("W1", 512),
        ("W2", 1024),
        ("W3", 256),
        ("W4", 800),
        ("W5", 128),
    ]

    decode_tokens = running_decode_seqs
    budget = MAX_BATCHED_TOKENS - decode_tokens
    print(f"\nmax_num_batched_tokens = {MAX_BATCHED_TOKENS}")
    print(f"Running decode seqs: {running_decode_seqs} (consumes {decode_tokens} tokens)")
    print(f"Available budget: {budget} tokens")
    print(SEP)

    admitted = []
    deferred = []
    for req_id, input_len in waiting:
        if input_len <= budget:
            admitted.append((req_id, input_len))
            budget -= input_len
            status = f"✓ ADMIT  (budget → {budget})"
        else:
            deferred.append((req_id, input_len))
            status = f"✗ DEFER  (needs {input_len} > {budget} remaining)"
        print(f"  {req_id}: input_len={input_len:4d}  {status}")

    total_tokens = running_decode_seqs + sum(l for _, l in admitted)
    print(f"\nThis iteration: {total_tokens} total tokens")
    print(f"  Admitted: {[r for r,_ in admitted]}")
    print(f"  Deferred to next iteration: {[r for r,_ in deferred]}")

    assert len(admitted) == 4   # W1, W2, W3, W5 (W5=128 fits remaining 224)
    assert len(deferred) == 1   # W4 (800 > 224 remaining after W1-W3)
    print("\n✓ Assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4: Prefill/decode interleave step-time model
# ─────────────────────────────────────────────────────────────────────────────

def demo_prefill_decode_interleave():
    print(f"\n{'='*70}")
    print("DEMO 4 — Prefill/Decode Interleave Step-Time Model")
    print(f"{'='*70}")

    # Step time model for a 7B model on A100:
    # base_decode_ms = 20ms for 32 decode seqs
    # prefill_overhead = prompt_tokens / 1000 * 10ms (prefill is faster per token)

    decode_seqs = 32
    base_decode_ms = 20.0

    scenarios = [
        ("Decode only",            0),
        ("Decode + 256-tok prefill",  256),
        ("Decode + 512-tok prefill",  512),
        ("Decode + 1024-tok prefill", 1024),
        ("Decode + 2048-tok prefill", 2048),
    ]

    print(f"\n{decode_seqs} active decode sequences (base step: {base_decode_ms} ms)")
    print(f"\n{'Scenario':<35} {'Step (ms)':>10} {'Decode latency':>15} {'Overhead':>10}")
    print(SEP)

    for name, prefill_toks in scenarios:
        prefill_overhead = prefill_toks / 1000 * 10
        step_ms = base_decode_ms + prefill_overhead
        decode_latency_per_tok = step_ms  # each decode seq gets 1 token per step
        overhead_pct = (step_ms - base_decode_ms) / base_decode_ms * 100
        print(f"{name:<35} {step_ms:>10.1f} {decode_latency_per_tok:>14.1f}ms {overhead_pct:>9.0f}%")

    # Chunked prefill analysis
    print(f"\nChunked prefill (chunk_size=256, prompt=2048 tokens):")
    chunk_size = 256
    prompt_len = 2048
    n_chunks = math.ceil(prompt_len / chunk_size)
    chunk_overhead = chunk_size / 1000 * 10
    chunk_step = base_decode_ms + chunk_overhead
    total_ttft = n_chunks * chunk_step
    print(f"  Chunks needed: {n_chunks}")
    print(f"  Per-chunk step time: {chunk_step:.1f} ms")
    print(f"  TTFT: {total_ttft:.0f} ms  (vs {base_decode_ms + 2048/1000*10:.0f} ms single-step)")
    print(f"  Decode step-time inflation: +{chunk_overhead:.1f} ms per chunk (bounded)")

    assert chunk_overhead < (2048 / 1000 * 10), "chunked prefill overhead per step < single-step overhead"
    print("\n✓ Assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5: Preemption cost: SWAP vs RECOMPUTE
# ─────────────────────────────────────────────────────────────────────────────

def demo_preemption_cost():
    print(f"\n{'='*70}")
    print("DEMO 5 — Preemption Cost: SWAP vs. RECOMPUTE Policy")
    print(f"{'='*70}")

    # Model parameters: Llama-3-8B
    num_layers = 32
    num_kv_heads = 8
    head_dim = 128
    dtype_bytes = 2     # BF16
    block_size = 16     # tokens per block
    pcie_bw_gbps = 32   # PCIe 4.0 x16 bidirectional

    # Prefill throughput (tokens/sec at batch=1)
    prefill_throughput_toks_per_sec = 3000

    print(f"\nModel: Llama-3-8B, {num_layers} layers, {num_kv_heads} KV heads, BF16")
    print(f"PCIe bandwidth: {pcie_bw_gbps} GB/s")
    print(f"Prefill throughput: {prefill_throughput_toks_per_sec} tok/s")
    print(SEP)

    seq_lengths = [32, 64, 128, 256, 512, 1024]

    print(f"\n{'Seq len':>8} {'SWAP (ms)':>12} {'RECOMPUTE (ms)':>16} {'Best policy':>14}")
    print(SEP)

    crossover = None
    for seq_len in seq_lengths:
        # KV size in bytes
        n_blocks = math.ceil(seq_len / block_size)
        kv_bytes = n_blocks * block_size * 2 * num_kv_heads * head_dim * num_layers * dtype_bytes
        kv_mb = kv_bytes / 1e6

        # SWAP: copy to CPU + copy back = 2× transfer
        swap_ms = (kv_bytes / (pcie_bw_gbps * 1e9)) * 1000 * 2

        # RECOMPUTE: re-prefill from scratch
        recompute_ms = (seq_len / prefill_throughput_toks_per_sec) * 1000

        best = "RECOMPUTE" if recompute_ms < swap_ms else "SWAP"
        if crossover is None and best == "SWAP":
            crossover = seq_len
        print(f"{seq_len:>8} {swap_ms:>12.2f} {recompute_ms:>16.1f} {best:>14}")

    print(f"\nCrossover: RECOMPUTE wins below ~{crossover} tokens, SWAP wins above")

    # Worked example: 350-token sequence
    seq_len = 350
    n_blocks = math.ceil(seq_len / block_size)
    kv_bytes = n_blocks * block_size * 2 * num_kv_heads * head_dim * num_layers * dtype_bytes
    swap_ms = (kv_bytes / (pcie_bw_gbps * 1e9)) * 1000 * 2
    recompute_ms = (seq_len / prefill_throughput_toks_per_sec) * 1000
    print(f"\nWorked example: 350-token sequence")
    print(f"  KV size:       {kv_bytes/1e6:.1f} MB ({n_blocks} blocks)")
    print(f"  SWAP cost:     {swap_ms:.1f} ms")
    print(f"  RECOMPUTE cost:{recompute_ms:.1f} ms")
    print(f"  Winner:        {'SWAP' if swap_ms < recompute_ms else 'RECOMPUTE'}")

    assert swap_ms < recompute_ms, "SWAP should win for 350-token sequences"
    print("\n✓ Assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6: max_num_seqs vs. memory
# ─────────────────────────────────────────────────────────────────────────────

def demo_max_num_seqs():
    print(f"\n{'='*70}")
    print("DEMO 6 — max_num_seqs vs. Memory Capacity")
    print(f"{'='*70}")

    # Llama-3-8B on A100 80GB
    total_hbm_gb = 80
    weights_gb = 16   # 8B params × 2 bytes BF16
    overhead_gb = 2
    available_for_kv_gb = total_hbm_gb - weights_gb - overhead_gb

    num_layers = 32
    num_kv_heads = 8
    head_dim = 128
    dtype_bytes = 2
    block_size = 16
    block_bytes = block_size * 2 * num_kv_heads * head_dim * num_layers * dtype_bytes

    print(f"\nLlama-3-8B on A100 80GB")
    print(f"  Weights:    {weights_gb} GB")
    print(f"  Overhead:   {overhead_gb} GB")
    print(f"  KV budget:  {available_for_kv_gb} GB")
    print(f"  Block size: {block_size} tokens = {block_bytes/1024:.1f} KB")
    total_blocks = int(available_for_kv_gb * 1024**3 / block_bytes)
    print(f"  Total KV blocks available: {total_blocks:,}")
    print(SEP)

    print(f"\n{'max_num_seqs':>14} {'avg_len':>8} {'blocks_needed':>14} {'fits?':>7} {'util_if_full':>14}")
    print(SEP)

    for max_seqs in [32, 64, 128, 256, 512]:
        for avg_len in [256, 1024]:
            blocks_per_seq = math.ceil(avg_len / block_size)
            blocks_needed = max_seqs * blocks_per_seq
            fits = blocks_needed <= total_blocks
            util = min(1.0, total_blocks / blocks_needed)
            marker = "✓" if fits else "✗ OOM"
            print(f"{max_seqs:>14} {avg_len:>8} {blocks_needed:>14,} {marker:>7} {util:>13.0%}")

    assert total_blocks > 1000, "Should have substantial KV block budget"
    print("\n✓ Assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 7: Scheduler efficiency metrics
# ─────────────────────────────────────────────────────────────────────────────

def demo_scheduler_metrics():
    print(f"\n{'='*70}")
    print("DEMO 7 — Scheduler Efficiency Metrics")
    print(f"{'='*70}")

    # Simulate metrics from a production run
    random.seed(42)

    max_num_seqs = 64
    n_iterations = 500
    step_time_ms = 25.0

    batch_sizes = []
    tokens_generated = 0
    preemptions = 0

    # Simulate batch size varying based on traffic
    base_load = 0.7  # 70% of max capacity

    for i in range(n_iterations):
        # Vary load sinusoidally to simulate traffic pattern
        load = base_load + 0.2 * math.sin(i / 50)
        load = max(0.2, min(1.0, load))
        batch_size = int(max_num_seqs * load)
        batch_sizes.append(batch_size)
        tokens_generated += batch_size
        # Random preemptions (2% of steps)
        if random.random() < 0.02:
            preemptions += 1

    elapsed_sec = n_iterations * step_time_ms / 1000
    avg_batch = sum(batch_sizes) / len(batch_sizes)
    batch_util = avg_batch / max_num_seqs
    actual_throughput = tokens_generated / elapsed_sec
    theoretical_throughput = max_num_seqs / (step_time_ms / 1000)
    efficiency = actual_throughput / theoretical_throughput
    preemption_rate = preemptions / n_iterations

    print(f"\nSimulated production run: {n_iterations} iterations × {step_time_ms} ms")
    print(SEP)
    print(f"  avg_batch_size:     {avg_batch:.1f}  (max={max_num_seqs})")
    print(f"  batch_utilization:  {batch_util:.2%}  (target: >85%)")
    print(f"  actual throughput:  {actual_throughput:.0f} tok/s")
    print(f"  theoretical:        {theoretical_throughput:.0f} tok/s")
    print(f"  scheduler_efficiency:{efficiency:.2%}  (target: >80%)")
    print(f"  preemption_rate:    {preemption_rate:.2%}  (target: <1%)")

    print(f"\nDiagnosis:")
    if batch_util > 0.85:
        print(f"  ✓ Batch utilization healthy")
    else:
        print(f"  ✗ Low batch utilization — check max_num_seqs or traffic volume")
    if efficiency > 0.80:
        print(f"  ✓ Scheduler efficiency healthy")
    else:
        print(f"  ✗ Low scheduler efficiency — check CPU scheduling overhead")
    if preemption_rate < 0.01:
        print(f"  ✓ Preemption rate healthy")
    else:
        print(f"  ✗ High preemption rate — reduce max_num_seqs or max_model_len")

    assert 0.65 < batch_util < 0.95
    assert efficiency > 0.60
    assert preemption_rate < 0.05
    print("\n✓ Assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_static_vs_continuous()
    demo_scheduling_loop()
    demo_token_budget()
    demo_prefill_decode_interleave()
    demo_preemption_cost()
    demo_max_num_seqs()
    demo_scheduler_metrics()
    print(f"\n{'='*70}")
    print("All Chapter 7.5 demos passed.")
    print(f"{'='*70}")
```

**Run:**
```bash
python continuous_batching_demo.py
```

---

## C++

```cpp
/*
 * continuous_batching_demo.cpp — Chapter 7.5: Continuous Batching
 *
 * Demos:
 *   Demo 1: Static batching utilization vs. continuous batching
 *   Demo 2: Iteration-level scheduling loop simulation
 *   Demo 3: Token budget admission control
 *   Demo 4: Prefill/decode interleave step-time model
 *   Demo 5: Preemption cost — SWAP vs. RECOMPUTE
 *   Demo 6: max_num_seqs vs. memory capacity
 *   Demo 7: Scheduler efficiency metrics
 *
 * Compile: g++ -O2 -std=c++17 -o continuous_batching_demo continuous_batching_demo.cpp
 * Run:     ./continuous_batching_demo
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static const std::string SEP(70, '-');

// ─────────────────────────────────────────────────────────────────────────────
// Demo 1: Static batching waste
// ─────────────────────────────────────────────────────────────────────────────

static void demo_static_vs_continuous() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "DEMO 1 — Static Batching vs. Continuous Batching GPU utilization\n";
    std::cout << std::string(70, '=') << "\n";

    std::vector<int> output_lengths = {4, 7, 12, 3, 89, 6, 14, 22};
    int n = (int)output_lengths.size();
    int max_len = *std::max_element(output_lengths.begin(), output_lengths.end());
    int total_useful = std::accumulate(output_lengths.begin(), output_lengths.end(), 0);
    int static_slots = max_len * n;
    double static_util = (double)total_useful / static_slots;

    std::cout << "\nBatch of " << n << " requests, max output: " << max_len << " tokens\n";
    std::cout << "Total useful work:   " << total_useful << " token-steps\n";
    std::cout << "Static batch slots:  " << static_slots << " token-steps\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Static utilization:  " << static_util * 100 << "%\n";
    std::cout << "Wasted capacity:     " << (1.0 - static_util) * 100 << "%\n";

    // Simulate continuous batching
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(3, 40);
    int n_slots = n;
    std::vector<int> slots(output_lengths);
    std::deque<int> queue;
    for (int i = 0; i < 100; ++i) queue.push_back(dist(rng));

    int active_steps = 0;
    int step = 0;
    while (std::any_of(slots.begin(), slots.end(), [](int s){ return s > 0; })) {
        int active = std::count_if(slots.begin(), slots.end(), [](int s){ return s > 0; });
        active_steps += active;
        for (auto& s : slots) s = std::max(0, s - 1);
        for (int i = 0; i < n_slots && !queue.empty(); ++i) {
            if (slots[i] == 0) {
                slots[i] = queue.front(); queue.pop_front();
            }
        }
        if (++step > 200) break;
    }

    double cont_util = (double)active_steps / ((double)step * n_slots);
    std::cout << "\nContinuous batching (" << step << " steps):\n";
    std::cout << "  Active slot-steps: " << active_steps << "\n";
    std::cout << "  Continuous utilization: " << cont_util * 100 << "%\n";
    std::cout << "  Improvement over static: " << cont_util / static_util << "x\n";

    assert(static_util < 0.60);
    assert(cont_util > 0.85);
    std::cout << "\n✓ Assertions passed\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 2: Iteration-level scheduling loop
// ─────────────────────────────────────────────────────────────────────────────

struct Request {
    std::string id;
    int input_len;
    int max_output;
    int tokens_generated = 0;
    std::string state = "waiting";

    bool is_finished() const { return tokens_generated >= max_output; }
};

static void demo_scheduling_loop() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "DEMO 2 — Iteration-Level Scheduling Loop Simulation\n";
    std::cout << std::string(70, '=') << "\n";

    std::deque<Request> waiting = {
        {"R1", 32,  10},
        {"R2", 64,  25},
        {"R3", 16,  5},
        {"R4", 128, 40},
        {"R5", 8,   8},
    };
    std::vector<Request> running;
    std::vector<Request> finished;

    const int MAX_NUM_SEQS = 3;
    const int MAX_BATCHED_TOKENS = 256;
    std::vector<double> step_times;

    std::cout << "\nConfig: max_num_seqs=" << MAX_NUM_SEQS
              << ", max_num_batched_tokens=" << MAX_BATCHED_TOKENS << "\n";
    std::cout << "Requests to serve: " << waiting.size() << "\n";
    std::cout << SEP << "\n";

    for (int iteration = 1; iteration <= 100; ++iteration) {
        // Step 1: free finished
        auto it = std::remove_if(running.begin(), running.end(),
            [&](Request& r) {
                if (r.is_finished()) { r.state = "finished"; finished.push_back(r); return true; }
                return false;
            });
        running.erase(it, running.end());

        // Step 4: admit within token budget
        int decode_tok = (int)running.size();
        int budget = MAX_BATCHED_TOKENS - decode_tok;
        std::vector<std::string> admitted;

        while (!waiting.empty() && (int)running.size() < MAX_NUM_SEQS) {
            Request& cand = waiting.front();
            if (cand.input_len <= budget) {
                budget -= cand.input_len;
                admitted.push_back(cand.id);
                cand.state = "running";
                running.push_back(cand);
                waiting.pop_front();
            } else break;
        }

        // Step-time model
        int prefill_tok = 0;
        for (auto& id : admitted)
            for (auto& r : running) if (r.id == id) prefill_tok += r.input_len;
        double step_ms = 5.0 + 0.05 * (prefill_tok + (int)running.size());
        step_times.push_back(step_ms);

        for (auto& r : running) ++r.tokens_generated;

        if (iteration <= 8 || (int)finished.size() == 5) {
            std::cout << "Iter " << std::setw(2) << iteration
                      << ": running=" << running.size()
                      << ", admitted=[";
            for (size_t i = 0; i < admitted.size(); ++i)
                std::cout << admitted[i] << (i+1<admitted.size()?",":"");
            std::cout << "], finished_total=" << finished.size()
                      << ", step=" << std::fixed << std::setprecision(1) << step_ms << "ms\n";
        }

        if ((int)finished.size() == 5) {
            std::cout << "\nAll 5 requests finished at iteration " << iteration << "\n";
            break;
        }
    }

    double avg_step = std::accumulate(step_times.begin(), step_times.end(), 0.0) / step_times.size();
    std::cout << "Average step time: " << avg_step << " ms\n";
    assert(finished.size() == 5);
    std::cout << "\n✓ All requests served\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 3: Token budget admission control
// ─────────────────────────────────────────────────────────────────────────────

static void demo_token_budget() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "DEMO 3 — Token Budget Admission Control\n";
    std::cout << std::string(70, '=') << "\n";

    const int MAX_BATCHED_TOKENS = 2048;
    const int running_decode_seqs = 32;
    std::vector<std::pair<std::string,int>> waiting = {
        {"W1", 512}, {"W2", 1024}, {"W3", 256}, {"W4", 800}, {"W5", 128}
    };

    int budget = MAX_BATCHED_TOKENS - running_decode_seqs;
    std::cout << "\nmax_num_batched_tokens=" << MAX_BATCHED_TOKENS
              << ", running decode=" << running_decode_seqs
              << ", budget=" << budget << "\n";
    std::cout << SEP << "\n";

    int n_admitted = 0, n_deferred = 0;
    for (auto& [id, len] : waiting) {
        if (len <= budget) {
            budget -= len;
            ++n_admitted;
            std::cout << "  " << id << ": len=" << std::setw(4) << len
                      << "  ✓ ADMIT  (budget → " << budget << ")\n";
        } else {
            ++n_deferred;
            std::cout << "  " << id << ": len=" << std::setw(4) << len
                      << "  ✗ DEFER  (needs " << len << " > " << budget << ")\n";
        }
    }

    assert(n_admitted == 4);  // W1, W2, W3, W5
    assert(n_deferred == 1);  // W4
    std::cout << "\n✓ Assertions passed\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 4: Prefill/decode interleave step-time
// ─────────────────────────────────────────────────────────────────────────────

static void demo_prefill_decode_interleave() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "DEMO 4 — Prefill/Decode Interleave Step-Time Model\n";
    std::cout << std::string(70, '=') << "\n";

    const double base_decode_ms = 20.0;
    std::vector<std::pair<std::string,int>> scenarios = {
        {"Decode only",              0},
        {"Decode + 256-tok prefill", 256},
        {"Decode + 512-tok prefill", 512},
        {"Decode + 1024-tok prefill",1024},
        {"Decode + 2048-tok prefill",2048},
    };

    std::cout << "\n" << std::left << std::setw(35) << "Scenario"
              << std::right << std::setw(10) << "Step(ms)"
              << std::setw(12) << "Overhead%" << "\n";
    std::cout << SEP << "\n";

    for (auto& [name, prefill] : scenarios) {
        double overhead = prefill / 1000.0 * 10.0;
        double step_ms  = base_decode_ms + overhead;
        double pct      = overhead / base_decode_ms * 100;
        std::cout << std::left << std::setw(35) << name
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(10) << step_ms
                  << std::setw(11) << pct << "%\n";
    }

    // Chunked prefill
    int chunk_size = 256, prompt_len = 2048;
    int n_chunks = (prompt_len + chunk_size - 1) / chunk_size;
    double chunk_overhead = chunk_size / 1000.0 * 10.0;
    double chunk_step = base_decode_ms + chunk_overhead;
    std::cout << "\nChunked prefill (chunk=" << chunk_size << ", prompt=" << prompt_len << "):\n";
    std::cout << "  Chunks: " << n_chunks << ", per-chunk step: " << chunk_step << " ms\n";
    std::cout << "  TTFT: " << n_chunks * chunk_step << " ms\n";
    std::cout << "  Decode inflation per chunk: +" << chunk_overhead << " ms (bounded)\n";

    assert(chunk_overhead < 2048.0 / 1000.0 * 10.0);
    std::cout << "\n✓ Assertions passed\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 5: Preemption cost
// ─────────────────────────────────────────────────────────────────────────────

static void demo_preemption_cost() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "DEMO 5 — Preemption Cost: SWAP vs. RECOMPUTE\n";
    std::cout << std::string(70, '=') << "\n";

    const int num_layers  = 32;
    const int num_kv_heads = 8;
    const int head_dim    = 128;
    const int dtype_bytes = 2;
    const int block_size  = 16;
    const double pcie_bw  = 32e9;  // bytes/s
    const double prefill_tps = 3000.0;

    std::vector<int> seq_lengths = {32, 64, 128, 256, 512, 1024};
    std::cout << "\n" << std::setw(8) << "Seq len"
              << std::setw(14) << "SWAP (ms)"
              << std::setw(18) << "RECOMPUTE (ms)"
              << std::setw(14) << "Best\n";
    std::cout << SEP << "\n";

    int crossover = -1;
    for (int seq_len : seq_lengths) {
        int n_blocks = (seq_len + block_size - 1) / block_size;
        long long kv_bytes = (long long)n_blocks * block_size * 2 * num_kv_heads
                           * head_dim * num_layers * dtype_bytes;
        double swap_ms = kv_bytes / pcie_bw * 1000.0 * 2;
        double recompute_ms = seq_len / prefill_tps * 1000.0;
        std::string best = (recompute_ms < swap_ms) ? "RECOMPUTE" : "SWAP";
        if (crossover < 0 && best == "SWAP") crossover = seq_len;
        std::cout << std::setw(8) << seq_len
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << swap_ms
                  << std::setprecision(1) << std::setw(18) << recompute_ms
                  << std::setw(14) << best << "\n";
    }

    // Worked example: 350 tokens
    int seq_len = 350;
    int n_blocks = (seq_len + block_size - 1) / block_size;
    long long kv_bytes = (long long)n_blocks * block_size * 2 * num_kv_heads
                       * head_dim * num_layers * dtype_bytes;
    double swap_ms = kv_bytes / pcie_bw * 1000.0 * 2;
    double recompute_ms = seq_len / prefill_tps * 1000.0;
    std::cout << "\n350-token sequence: SWAP=" << std::fixed << std::setprecision(1)
              << swap_ms << "ms  RECOMPUTE=" << recompute_ms << "ms\n";

    assert(swap_ms < recompute_ms);
    std::cout << "\n✓ Assertions passed\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 6: max_num_seqs vs. memory
// ─────────────────────────────────────────────────────────────────────────────

static void demo_max_num_seqs() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "DEMO 6 — max_num_seqs vs. Memory Capacity\n";
    std::cout << std::string(70, '=') << "\n";

    const double total_hbm_gb = 80.0;
    const double weights_gb   = 16.0;
    const double overhead_gb  = 2.0;
    const double kv_budget_gb = total_hbm_gb - weights_gb - overhead_gb;

    const int num_layers   = 32;
    const int num_kv_heads = 8;
    const int head_dim     = 128;
    const int dtype_bytes  = 2;
    const int block_size   = 16;
    long long block_bytes = (long long)block_size * 2 * num_kv_heads
                          * head_dim * num_layers * dtype_bytes;

    long long total_blocks = (long long)(kv_budget_gb * 1e9 / block_bytes);
    std::cout << "\nA100 80GB, Llama-3-8B: KV budget=" << kv_budget_gb
              << " GB, total blocks=" << total_blocks << "\n";
    std::cout << SEP << "\n";
    std::cout << std::setw(14) << "max_num_seqs"
              << std::setw(10) << "avg_len"
              << std::setw(16) << "blocks_needed"
              << std::setw(8)  << "fits?\n";
    std::cout << SEP << "\n";

    for (int max_seqs : {32, 64, 128, 256, 512}) {
        for (int avg_len : {256, 1024}) {
            long long blks = (long long)max_seqs * ((avg_len + block_size - 1) / block_size);
            bool fits = blks <= total_blocks;
            std::cout << std::setw(14) << max_seqs
                      << std::setw(10) << avg_len
                      << std::setw(16) << blks
                      << std::setw(8) << (fits ? "✓" : "✗ OOM") << "\n";
        }
    }

    assert(total_blocks > 1000);
    std::cout << "\n✓ Assertions passed\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 7: Scheduler efficiency metrics
// ─────────────────────────────────────────────────────────────────────────────

static void demo_scheduler_metrics() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "DEMO 7 — Scheduler Efficiency Metrics\n";
    std::cout << std::string(70, '=') << "\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    const int max_num_seqs    = 64;
    const int n_iterations    = 500;
    const double step_time_ms = 25.0;

    long long tokens_generated = 0;
    int preemptions = 0;
    std::vector<int> batch_sizes;

    for (int i = 0; i < n_iterations; ++i) {
        double load = 0.7 + 0.2 * std::sin(i / 50.0);
        load = std::max(0.2, std::min(1.0, load));
        int batch_size = (int)(max_num_seqs * load);
        batch_sizes.push_back(batch_size);
        tokens_generated += batch_size;
        if (udist(rng) < 0.02) ++preemptions;
    }

    double elapsed_sec = n_iterations * step_time_ms / 1000.0;
    double avg_batch = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0.0)
                     / batch_sizes.size();
    double batch_util   = avg_batch / max_num_seqs;
    double actual_tps   = tokens_generated / elapsed_sec;
    double theory_tps   = max_num_seqs / (step_time_ms / 1000.0);
    double efficiency   = actual_tps / theory_tps;
    double preempt_rate = (double)preemptions / n_iterations;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  avg_batch_size:      " << avg_batch << " (max=" << max_num_seqs << ")\n";
    std::cout << "  batch_utilization:   " << batch_util * 100 << "%\n";
    std::cout << "  actual throughput:   " << (int)actual_tps << " tok/s\n";
    std::cout << "  scheduler_efficiency:" << efficiency * 100 << "%\n";
    std::cout << "  preemption_rate:     " << preempt_rate * 100 << "%\n";

    assert(batch_util > 0.50 && batch_util < 0.99);
    assert(efficiency > 0.50);
    assert(preempt_rate < 0.05);
    std::cout << "\n✓ Assertions passed\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    demo_static_vs_continuous();
    demo_scheduling_loop();
    demo_token_budget();
    demo_prefill_decode_interleave();
    demo_preemption_cost();
    demo_max_num_seqs();
    demo_scheduler_metrics();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "All Chapter 7.5 demos passed.\n";
    std::cout << std::string(70, '=') << "\n";
    return 0;
}
```

**Compile and run:**
```bash
g++ -O2 -std=c++17 -o continuous_batching_demo continuous_batching_demo.cpp && ./continuous_batching_demo
```
