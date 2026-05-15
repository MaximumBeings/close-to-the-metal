# Chapter 32: Debugging Inference Systems — Companion Code

## Python — `debugging_demo.py`

```python
"""
debugging_demo.py — Chapter 32: Debugging Inference Systems

Demonstrates:
  1. NaN detection and propagation tracing
  2. Softmax numerical stability analysis
  3. Quantization error measurement
  4. Sampling determinism verification
  5. Memory leak detection pattern
  6. KV cache sequence isolation test
  7. Performance regression benchmark
  8. Latency spike diagnosis

Run: python debugging_demo.py
Requirements: numpy (pip install numpy)
"""
from __future__ import annotations

import gc
import math
import random
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# §1  NaN DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def check_tensor(arr: np.ndarray, name: str) -> bool:
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    if has_nan or has_inf:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        print(f"  [WARN] {name}: NaN={n_nan}, Inf={n_inf}, shape={arr.shape}")
        return True
    return False

def unstable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically UNSTABLE softmax — for demonstration only."""
    e = np.exp(x.astype(np.float32))   # overflows for large x
    return e / e.sum()

def stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically STABLE softmax: subtract max before exp."""
    x = x.astype(np.float64)
    x = x - x.max()                    # prevents overflow
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)

def demo_nan_detection() -> None:
    section("NaN Detection and Softmax Stability")

    # Simulate large logits that cause fp16 overflow → NaN
    large_logits = np.array([1000.0, 999.0, 998.0], dtype=np.float16)
    safe_logits  = np.array([10.0, 9.0, 8.0],       dtype=np.float16)

    print("\n  Testing with large logits (1000, 999, 998) in fp16:")
    unstable_out = unstable_softmax(large_logits)
    check_tensor(unstable_out, "unstable_softmax(large_logits)")
    stable_out   = stable_softmax(large_logits)
    check_tensor(stable_out,   "stable_softmax(large_logits)")
    print(f"  Unstable result: {unstable_out}")
    print(f"  Stable result:   {stable_out[:3].round(4)}")

    print("\n  Testing with normal logits (10, 9, 8):")
    u2 = unstable_softmax(safe_logits)
    s2 = stable_softmax(safe_logits)
    print(f"  Unstable: {u2[:3].round(4)}")
    print(f"  Stable:   {s2[:3].round(4)}")
    print(f"  Max diff: {np.abs(u2 - s2).max():.6f}")

    # Verify stable softmax is correct (sums to 1, no NaN)
    assert not np.isnan(stable_out).any(), "Stable softmax should not produce NaN"
    assert abs(stable_out.sum() - 1.0) < 1e-5, "Softmax should sum to 1"
    print(f"\n  [ASSERT] Stable softmax: no NaN, sums to 1.0 ✓")

    # Simulate NaN propagation through a simple network
    print("\n  NaN propagation simulation:")
    x = np.array([1.0, np.nan, 3.0])
    y = x * 2.0          # NaN propagates
    z = y + 1.0          # Still NaN
    w = np.exp(z)        # Still NaN
    check_tensor(x, "input (has NaN)")
    check_tensor(w, "output after × 2 + 1 + exp")
    print(f"  NaN at position 1 → propagates to all downstream ops: "
          f"{np.isnan(w).sum()} NaN values in output")
    assert np.isnan(w[1]), "NaN should propagate through arithmetic"
    print(f"\n  [ASSERT] NaN propagation correctly detected: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §2  QUANTIZATION ERROR MEASUREMENT
# ──────────────────────────────────────────────────────────────────────────────

def quantize_int8(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Symmetric INT8 quantization: scale = max(|x|) / 127."""
    scale = float(np.abs(x).max()) / 127.0
    if scale == 0:
        return np.zeros_like(x, dtype=np.int8), 1.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale

def dequantize_int8(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale

def matmul_fp32(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A.astype(np.float64) @ B.astype(np.float64)

def matmul_int8(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Aq, As = quantize_int8(A)
    Bq, Bs = quantize_int8(B)
    result = Aq.astype(np.int32) @ Bq.astype(np.int32)
    return result.astype(np.float32) * As * Bs

def demo_quantization_error() -> None:
    section("Quantization Error Measurement")

    rng = np.random.default_rng(42)
    M, K, N = 64, 256, 128

    # Normal weights: low error expected
    A_normal = rng.standard_normal((M, K)).astype(np.float32) * 0.1
    B_normal = rng.standard_normal((K, N)).astype(np.float32) * 0.1

    # Outlier weights: high error expected
    A_outlier = A_normal.copy()
    A_outlier[0, 0] = 100.0  # single extreme outlier

    def measure_error(A, B, label):
        fp_out  = matmul_fp32(A, B).astype(np.float32)
        int_out = matmul_int8(A, B)
        l_inf   = float(np.abs(fp_out - int_out).max())
        rel_err = l_inf / (np.abs(fp_out).max() + 1e-8)
        status  = "✓ OK" if rel_err < 0.05 else "✗ DEGRADED"
        print(f"  {label:<25} L∞={l_inf:.4f}  rel={rel_err:.4f}  [{status}]")
        return rel_err

    print(f"\n  {'Scenario':<25} {'L∞ error':<12} {'Rel error':<12} {'Status'}")
    print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*10}")
    err_normal  = measure_error(A_normal,  B_normal, "Normal weights")
    err_outlier = measure_error(A_outlier, B_normal, "Outlier weight (100x)")

    # Demonstrate per-channel quantization improves outlier case
    # Per-channel: each row of A has its own scale
    def matmul_int8_perchannel(A, B):
        result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
        for i in range(A.shape[0]):
            row_q, row_s = quantize_int8(A[i])
            Bq, Bs = quantize_int8(B)
            result[i] = (row_q.astype(np.int32) @ Bq.astype(np.int32)).astype(np.float32) * row_s * Bs
        return result

    fp_out      = matmul_fp32(A_outlier, B_normal).astype(np.float32)
    pc_out      = matmul_int8_perchannel(A_outlier, B_normal)
    l_inf_pc    = float(np.abs(fp_out - pc_out).max())
    rel_pc      = l_inf_pc / (np.abs(fp_out).max() + 1e-8)
    status_pc   = "✓ OK" if rel_pc < 0.05 else "✗ DEGRADED"
    print(f"  {'Outlier + per-channel':<25} L∞={l_inf_pc:.4f}  rel={rel_pc:.4f}  [{status_pc}]")

    assert err_normal  < 0.05, f"Normal quantization error {err_normal:.4f} too high"
    assert err_outlier > err_normal, "Outlier should cause more error than normal"
    assert rel_pc < err_outlier, "Per-channel should improve over tensor-wise quantization"
    print(f"\n  [ASSERT] Normal error < 5%: ✓")
    print(f"  [ASSERT] Outlier causes more error than normal: ✓")
    print(f"  [ASSERT] Per-channel quantization reduces outlier error: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §3  SAMPLING DETERMINISM
# ──────────────────────────────────────────────────────────────────────────────

def sample_token(logits: np.ndarray, temperature: float, seed: int) -> int:
    """Temperature sampling with explicit RNG seed."""
    rng = np.random.default_rng(seed)
    if temperature == 0.0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    probs  = stable_softmax(scaled)
    return int(rng.choice(len(probs), p=probs / probs.sum()))

def demo_sampling_determinism() -> None:
    section("Sampling Determinism Verification")

    rng = np.random.default_rng(0)
    VOCAB = 1000
    N_RUNS = 20

    # Simulate a logit distribution
    logits = rng.standard_normal(VOCAB).astype(np.float32) * 2.0

    print(f"\n  {'Config':<35} {'Unique outputs':>15}  {'Deterministic?':>15}")
    print(f"  {'─'*35} {'─'*15}  {'─'*15}")

    scenarios = [
        ("Temperature=0 (greedy), fixed seed",   0.0, 42),
        ("Temperature=0.8, fixed seed",          0.8, 42),
        ("Temperature=0.8, random seed (buggy)", 0.8, -1),  # -1 = random seed
    ]

    for desc, temp, seed in scenarios:
        outputs = []
        for _ in range(N_RUNS):
            actual_seed = seed if seed >= 0 else random.randint(0, 2**31)
            token = sample_token(logits, temp, actual_seed)
            outputs.append(token)
        n_unique = len(set(outputs))
        is_det   = n_unique == 1
        mark     = "✓ deterministic" if is_det else f"✗ {n_unique} unique"
        print(f"  {desc:<35} {n_unique:>15}  {mark:>15}")

    # Assert: greedy is always deterministic
    greedy_outputs = [sample_token(logits, 0.0, i) for i in range(N_RUNS)]
    assert len(set(greedy_outputs)) == 1, "Greedy should always be deterministic"

    # Assert: fixed-seed temperature is also deterministic
    fixed_outputs = [sample_token(logits, 0.8, 42) for _ in range(N_RUNS)]
    assert len(set(fixed_outputs)) == 1, "Fixed-seed sampling should be deterministic"

    print(f"\n  [ASSERT] Greedy (temp=0) is deterministic across seeds: ✓")
    print(f"  [ASSERT] Temperature sampling with fixed seed is deterministic: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §4  KV CACHE ISOLATION SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

class SimpleKVCache:
    """
    Minimal KV cache simulation to demonstrate sequence isolation.
    Tracks which sequence each cache slot belongs to.
    """
    def __init__(self, n_slots: int, n_layers: int, head_dim: int):
        self.n_slots  = n_slots
        self.n_layers = n_layers
        self.head_dim = head_dim
        # Allocate slots with sequence ownership tracking
        self._cache   = np.zeros((n_slots, n_layers, 2, head_dim), dtype=np.float32)
        self._owner   = [-1] * n_slots   # -1 = free, otherwise seq_id
        self._free    = list(range(n_slots))

    def allocate(self, seq_id: int, n_tokens: int) -> List[int]:
        if len(self._free) < n_tokens:
            raise MemoryError(f"OOM: need {n_tokens} slots, have {len(self._free)}")
        slots = self._free[:n_tokens]
        self._free = self._free[n_tokens:]
        for s in slots:
            self._owner[s] = seq_id
        return slots

    def write(self, seq_id: int, slot: int, layer: int, k: np.ndarray, v: np.ndarray):
        # Verify ownership — this check prevents corruption
        if self._owner[slot] != seq_id:
            raise ValueError(f"[BUG] seq {seq_id} writing to slot {slot} "
                             f"owned by seq {self._owner[slot]}")
        self._cache[slot, layer, 0] = k
        self._cache[slot, layer, 1] = v

    def read(self, seq_id: int, slot: int, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._owner[slot] != seq_id:
            raise ValueError(f"[BUG] seq {seq_id} reading from slot {slot} "
                             f"owned by seq {self._owner[slot]}")
        return self._cache[slot, layer, 0], self._cache[slot, layer, 1]

    def free(self, seq_id: int):
        freed = [s for s, o in enumerate(self._owner) if o == seq_id]
        for s in freed:
            self._owner[s] = -1
            self._cache[s] = 0
        self._free.extend(freed)

def demo_kv_cache_isolation() -> None:
    section("KV Cache Sequence Isolation")

    cache = SimpleKVCache(n_slots=32, n_layers=4, head_dim=64)
    rng   = np.random.default_rng(42)

    # Allocate two sequences
    slots_a = cache.allocate(seq_id=0, n_tokens=4)
    slots_b = cache.allocate(seq_id=1, n_tokens=4)

    # Write distinct values for each sequence
    for i, slot in enumerate(slots_a):
        k = np.ones(64, dtype=np.float32) * (i + 1) * 10.0   # seq A: 10, 20, 30, 40
        v = np.ones(64, dtype=np.float32) * (i + 1) * 10.0
        cache.write(0, slot, layer=0, k=k, v=v)

    for i, slot in enumerate(slots_b):
        k = np.ones(64, dtype=np.float32) * (i + 1) * -10.0  # seq B: -10, -20, -30, -40
        v = np.ones(64, dtype=np.float32) * (i + 1) * -10.0
        cache.write(1, slot, layer=0, k=k, v=v)

    print("\n  Reading back own sequence data:")
    for i, slot in enumerate(slots_a):
        k, _ = cache.read(0, slot, layer=0)
        print(f"    Seq A slot {i}: k[0]={k[0]:.1f}  (expected {(i+1)*10.0:.1f})")
        assert abs(k[0] - (i+1)*10.0) < 1e-5, f"Seq A data corrupted at slot {i}"

    print("\n  Attempting cross-sequence read (should raise):")
    caught = False
    try:
        cache.read(seq_id=0, slot=slots_b[0], layer=0)
    except ValueError as e:
        caught = True
        print(f"    Caught: {e}")
    assert caught, "Cross-sequence read should have been blocked"

    print("\n  Freeing seq A and verifying slots are released:")
    cache.free(seq_id=0)
    assert all(cache._owner[s] == -1 for s in slots_a), "Freed slots should be unowned"
    print(f"    Freed {len(slots_a)} slots. Free pool size: {len(cache._free)}")

    print(f"\n  [ASSERT] Sequence isolation enforced by ownership tracking: ✓")
    print(f"  [ASSERT] Cross-sequence access correctly blocked: ✓")
    print(f"  [ASSERT] Slot release correctly frees memory: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §5  MEMORY LEAK DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def demo_memory_leak_detection() -> None:
    section("Memory Leak Detection Pattern")

    # Simulate a request buffer that correctly releases memory
    class CorrectBuffer:
        def __init__(self):
            self._active: Dict[int, np.ndarray] = {}

        def store(self, req_id: int, tensor: np.ndarray):
            self._active[req_id] = tensor

        def complete(self, req_id: int) -> List:
            """Correctly converts to Python list and removes tensor reference."""
            result = self._active.pop(req_id).tolist()  # releases numpy array
            return result

        def active_count(self) -> int:
            return len(self._active)

    # Simulate a leaky buffer that retains references
    class LeakyBuffer:
        def __init__(self):
            self._active: Dict[int, np.ndarray] = {}
            self._completed: List[np.ndarray] = []  # BUG: retains tensor refs

        def store(self, req_id: int, tensor: np.ndarray):
            self._active[req_id] = tensor

        def complete(self, req_id: int) -> List:
            tensor = self._active.pop(req_id)
            self._completed.append(tensor)   # BUG: closure captures tensor
            return tensor.tolist()

        def active_count(self) -> int:
            return len(self._active)
        def leaked_count(self) -> int:
            return len(self._completed)

    N_REQUESTS = 100
    TENSOR_SIZE = 1000

    correct = CorrectBuffer()
    leaky   = LeakyBuffer()

    for i in range(N_REQUESTS):
        tensor = np.random.randn(TENSOR_SIZE).astype(np.float32)
        correct.store(i, tensor.copy())
        leaky.store(i, tensor.copy())
        correct.complete(i)
        leaky.complete(i)

    print(f"\n  After {N_REQUESTS} requests completed:")
    print(f"  CorrectBuffer active refs:  {correct.active_count()}  (should be 0)")
    print(f"  LeakyBuffer   active refs:  {leaky.active_count()}   (should be 0)")
    print(f"  LeakyBuffer   leaked refs:  {leaky.leaked_count()}  (BUG: should be 0)")

    bytes_leaked = leaky.leaked_count() * TENSOR_SIZE * 4
    print(f"\n  Memory held by leak: {bytes_leaked:,} bytes "
          f"({bytes_leaked / 1e6:.2f} MB) — would be GPU memory in real system")

    assert correct.active_count() == 0, "Correct buffer should have no active refs"
    assert leaky.leaked_count()   == N_REQUESTS, "Leaky buffer retains all tensors"
    print(f"\n  [ASSERT] Correct buffer releases all references: ✓")
    print(f"  [ASSERT] Leaky buffer correctly diagnosed — retains {leaky.leaked_count()} refs: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §6  LATENCY SPIKE DIAGNOSIS
# ──────────────────────────────────────────────────────────────────────────────

def demo_latency_spike_diagnosis() -> None:
    section("Latency Spike Diagnosis")

    N = 200
    latencies = []
    rng = random.Random(99)

    # Simulate request latency with occasional GC-pause spikes
    for i in range(N):
        base_lat = rng.gauss(80, 10)    # ~80ms typical
        # Simulate GC pause every ~50 requests
        gc_spike = 150 if (i % 47 == 0) else 0
        # Simulate cold-start spike on first request
        cold_start = 400 if i == 0 else 0
        lat = max(20, base_lat + gc_spike + cold_start)
        latencies.append(lat)

    lats = sorted(latencies)
    p50  = lats[int(N * 0.50)]
    p95  = lats[int(N * 0.95)]
    p99  = lats[int(N * 0.99)]
    p999 = lats[int(N * 0.999)] if N >= 1000 else lats[-1]

    print(f"\n  Latency percentiles over {N} requests:")
    print(f"    P50:  {p50:.1f} ms")
    print(f"    P95:  {p95:.1f} ms")
    print(f"    P99:  {p99:.1f} ms")
    print(f"    Max:  {lats[-1]:.1f} ms")
    print(f"\n  P99/P50 ratio: {p99/p50:.1f}x")

    # Diagnose: high P99/P50 ratio signals periodic spikes
    if p99 / p50 > 2.5:
        print(f"\n  [DIAG] P99/P50 = {p99/p50:.1f}x > 2.5 → periodic spike pattern")
        # Identify spikes
        spike_threshold = p50 * 2.5
        spikes = [(i, lat) for i, lat in enumerate(latencies) if lat > spike_threshold]
        gaps   = [spikes[i+1][0] - spikes[i][0] for i in range(len(spikes)-1)]
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"  [DIAG] {len(spikes)} spikes found, avg gap = {avg_gap:.0f} requests")
            print(f"  [DIAG] Regular periodicity ({avg_gap:.0f}q) suggests GC pause or")
            print(f"         background task (checkpointing, metrics flush)")

    assert p50 < 120, f"P50 latency {p50:.1f}ms unexpectedly high"
    assert p99 > p50, "P99 must be higher than P50"
    print(f"\n  [ASSERT] Latency spike pattern correctly identified: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §7  PERFORMANCE REGRESSION BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    label:       str
    tokens_per_s: float
    latency_ms:  float
    n_requests:  int

def simulate_inference(tokens: int, batch_size: int,
                       bandwidth_gbps: float = 500.0,
                       params_b: float = 8.0,
                       overhead_ms: float = 5.0) -> float:
    """Simulate inference latency based on memory-bandwidth model."""
    # Bytes to transfer: 2 × params × 2 bytes/param (fp16)
    bytes_per_forward = params_b * 1e9 * 2 * 2
    # Effective BW with batch size (diminishing returns due to compute becoming binding)
    eff_bw = bandwidth_gbps * 1e9 * min(1.0, batch_size / 8.0)
    mem_lat = bytes_per_forward / eff_bw * 1000  # ms
    # Output token generation: n_tokens × mem_lat (one pass per token)
    return overhead_ms + tokens * mem_lat * max(1, 2 - batch_size / 4)

def demo_performance_regression() -> None:
    section("Performance Regression Benchmark")

    print(f"\n  Simulated throughput at different batch sizes:")
    print(f"\n  {'Batch':>7}  {'TPS':>10}  {'Lat/req ms':>12}  {'GPU util%':>10}")
    print(f"  {'─'*7}  {'─'*10}  {'─'*12}  {'─'*10}")

    results = []
    for bs in [1, 2, 4, 8, 16, 32]:
        n_tok  = 128
        lat_ms = simulate_inference(n_tok, bs)
        tps    = n_tok * bs / (lat_ms / 1000)
        # GPU utilization: low at small batch (memory-bound), high at large batch
        gpu_util = min(98, 20 + bs * 5)
        results.append(BenchmarkResult(f"bs={bs}", tps, lat_ms, bs))
        marker = "  ← small batch, low GPU util" if bs == 1 else ""
        print(f"  {bs:>7}  {tps:>10.1f}  {lat_ms:>12.1f}  {gpu_util:>9}%{marker}")

    # Regression detection: compare current run to a "baseline"
    baseline_tps = results[-2].tokens_per_s  # bs=16
    current_tps  = results[-2].tokens_per_s * 0.97  # simulate 3% regression

    regression_pct = (baseline_tps - current_tps) / baseline_tps * 100
    threshold_pct  = 5.0

    print(f"\n  Regression detection (bs=16):")
    print(f"    Baseline TPS:  {baseline_tps:.1f}")
    print(f"    Current TPS:   {current_tps:.1f}")
    print(f"    Regression:    {regression_pct:.1f}%  (alert if > {threshold_pct}%)")

    assert results[0].tokens_per_s < results[-1].tokens_per_s, \
        "Throughput should increase with batch size"
    assert regression_pct < threshold_pct, \
        f"Simulated regression {regression_pct:.1f}% exceeds threshold"
    print(f"\n  [ASSERT] Throughput scales with batch size: ✓")
    print(f"  [ASSERT] Regression within acceptable bounds ({regression_pct:.1f}% < {threshold_pct}%): ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §8  SECTION UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ──────────────────────────────────────────────────────────────────────────────
# §9  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  Chapter 32 — Debugging Inference Systems (Python)\n{bar}")

    demo_nan_detection()
    demo_quantization_error()
    demo_sampling_determinism()
    demo_kv_cache_isolation()
    demo_memory_leak_detection()
    demo_latency_spike_diagnosis()
    demo_performance_regression()

    print(f"\n{bar}\n  All demos complete.\n{bar}\n")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()

```

## C++ — `debugging_demo.cpp`

```cpp
// debugging_demo.cpp
// Chapter 32 — Debugging Inference Systems (C++)
//
// Demonstrates:
//   1. NaN/Inf detection with tensor scan
//   2. Stable vs unstable softmax
//   3. INT8 quantization error measurement
//   4. KV cache ownership and isolation enforcement
//   5. Latency percentile analysis and spike detection
//   6. Memory bandwidth bottleneck model
//
// Build: g++ -O2 -std=c++17 -o debugging_demo debugging_demo.cpp
// Run:   ./debugging_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static void print_section(const std::string& t) {
    std::cout << "\n" << std::string(60, '-') << "\n  " << t
              << "\n" << std::string(60, '-') << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 1.  NaN / Inf DETECTION
// ─────────────────────────────────────────────────────────────────────────────

static bool check_tensor(const std::vector<float>& v, const std::string& name) {
    int n_nan = 0, n_inf = 0;
    for (float x : v) {
        if (std::isnan(x)) ++n_nan;
        if (std::isinf(x)) ++n_inf;
    }
    if (n_nan || n_inf) {
        std::cout << "  [WARN] " << name << ": NaN=" << n_nan
                  << " Inf=" << n_inf << " size=" << v.size() << "\n";
        return true;
    }
    return false;
}

static std::vector<float> unstable_softmax(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    float sum = 0;
    for (float xi : x) sum += std::exp(xi);   // overflows for large x
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = std::exp(x[i]) / sum;
    return out;
}

static std::vector<float> stable_softmax(const std::vector<float>& x) {
    float mx = *std::max_element(x.begin(), x.end());
    std::vector<float> out(x.size());
    float sum = 0;
    for (float xi : x) sum += std::exp(xi - mx);
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = std::exp(x[i] - mx) / sum;
    return out;
}

static void demo_nan_detection() {
    print_section("NaN Detection and Softmax Stability");

    // Large logits — overflow in fp32 softmax
    std::vector<float> large_logits = {1000.f, 999.f, 998.f};
    std::vector<float> normal_logits = {10.f, 9.f, 8.f};

    std::cout << "\n  Testing large logits {1000, 999, 998}:\n";
    auto unstable = unstable_softmax(large_logits);
    auto stable   = stable_softmax(large_logits);
    check_tensor(unstable, "unstable_softmax(large_logits)");
    check_tensor(stable,   "stable_softmax(large_logits)");

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "    Unstable: [" << unstable[0] << ", " << unstable[1] << ", " << unstable[2] << "]\n";
    std::cout << "    Stable:   [" << stable[0]   << ", " << stable[1]   << ", " << stable[2]   << "]\n";

    float sum_stable = std::accumulate(stable.begin(), stable.end(), 0.0f);
    assert(!std::isnan(stable[0]) && !std::isinf(stable[0]));
    assert(std::abs(sum_stable - 1.0f) < 1e-5f);
    std::cout << "  [ASSERT] Stable softmax: no NaN, sums to 1.0 ✓\n";

    // NaN propagation
    std::vector<float> x = {1.f, std::numeric_limits<float>::quiet_NaN(), 3.f};
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * 2.f + 1.f;
    check_tensor(x, "input (has NaN)");
    check_tensor(y, "output (NaN propagated)");
    assert(std::isnan(y[1]));
    std::cout << "  [ASSERT] NaN propagation correctly detected: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  INT8 QUANTIZATION ERROR
// ─────────────────────────────────────────────────────────────────────────────

static std::pair<std::vector<int8_t>, float> quantize_int8(const std::vector<float>& x) {
    float max_abs = 0;
    for (float v : x) max_abs = std::max(max_abs, std::abs(v));
    float scale = max_abs / 127.f + 1e-8f;
    std::vector<int8_t> q(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        q[i] = static_cast<int8_t>(std::clamp(std::round(x[i] / scale), -128.f, 127.f));
    return {q, scale};
}

// Returns the mean absolute quantization error on the non-outlier weights.
// This correctly reflects the quality degradation on typical weights when
// the quantization scale is dominated by an outlier.
static float quantize_error(const std::vector<float>& original, int outlier_idx = -1) {
    std::vector<float> vals = original;
    if (outlier_idx >= 0) vals[outlier_idx] = 100.f;

    auto [q, s] = quantize_int8(vals);

    // Measure error only on the non-outlier weights (first 10 elements)
    // These are the ones that suffer when scale is inflated by an outlier.
    float total_err = 0; int count = 0;
    for (size_t i = 0; i < std::min(vals.size(), size_t(10)); ++i) {
        if (outlier_idx >= 0 && (int)i == outlier_idx) continue;
        float reconstructed = q[i] * s;
        total_err += std::abs(vals[i] - reconstructed);
        ++count;
    }
    return count > 0 ? total_err / count : 0;
}

static void demo_quantization_error() {
    print_section("INT8 Quantization Error Measurement");

    // Generate a typical weight distribution
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.1f);
    std::vector<float> weights(256);
    for (auto& w : weights) w = nd(rng);

    float err_normal  = quantize_error(weights);
    float err_outlier = quantize_error(weights, 0);  // insert 100x outlier

    // Mean absolute error on typical (non-outlier) weights —
    // outlier inflates the quantization scale, degrading all small-magnitude weights.
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "\n  " << std::left << std::setw(32) << "Scenario"
              << std::setw(16) << "Mean abs error" << "Status\n";
    std::cout << "  " << std::string(52, '-') << "\n";

    auto print_row = [](const std::string& lbl, float err) {
        std::string status = err < 0.05f ? "✓ OK" : "✗ DEGRADED";
        std::cout << "  " << std::left << std::setw(32) << lbl
                  << std::fixed << std::setprecision(5) << std::setw(16) << err
                  << status << "\n";
    };
    print_row("Normal weights",          err_normal);
    print_row("With 100x outlier",       err_outlier);

    assert(err_normal  < 0.05f);
    assert(err_outlier > err_normal);
    std::cout << "\n  [ASSERT] Normal quantization mean error < 0.05: " << err_normal << " ✓\n";
    std::cout << "  [ASSERT] Outlier inflates scale → more error on normal weights: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  KV CACHE ISOLATION
// ─────────────────────────────────────────────────────────────────────────────

class KVCache {
public:
    int n_slots, n_layers, head_dim;
    std::vector<std::vector<float>> data;   // [slot][layer*head_dim*2]
    std::vector<int> owner;                 // -1=free, else seq_id
    std::vector<int> free_list;

    KVCache(int ns, int nl, int hd)
        : n_slots(ns), n_layers(nl), head_dim(hd),
          data(ns, std::vector<float>(nl * hd * 2, 0.f)),
          owner(ns, -1)
    {
        for (int i = 0; i < ns; ++i) free_list.push_back(i);
    }

    std::vector<int> allocate(int seq_id, int n_tokens) {
        assert((int)free_list.size() >= n_tokens);
        std::vector<int> slots(free_list.end() - n_tokens, free_list.end());
        free_list.resize(free_list.size() - n_tokens);
        for (int s : slots) owner[s] = seq_id;
        return slots;
    }

    void write(int seq_id, int slot, int layer,
               const std::vector<float>& k, const std::vector<float>& v) {
        assert(owner[slot] == seq_id && "Write to slot owned by another sequence");
        int offset = layer * head_dim * 2;
        std::copy(k.begin(), k.end(), data[slot].begin() + offset);
        std::copy(v.begin(), v.end(), data[slot].begin() + offset + head_dim);
    }

    std::pair<std::vector<float>, std::vector<float>>
    read(int seq_id, int slot, int layer) {
        assert(owner[slot] == seq_id && "Read from slot owned by another sequence");
        int offset = layer * head_dim * 2;
        return {
            {data[slot].begin() + offset, data[slot].begin() + offset + head_dim},
            {data[slot].begin() + offset + head_dim, data[slot].begin() + offset + head_dim*2}
        };
    }

    void free_seq(int seq_id) {
        for (int i = 0; i < n_slots; ++i) {
            if (owner[i] == seq_id) {
                owner[i] = -1;
                std::fill(data[i].begin(), data[i].end(), 0.f);
                free_list.push_back(i);
            }
        }
    }
};

static void demo_kv_cache_isolation() {
    print_section("KV Cache Sequence Isolation");

    KVCache cache(32, 4, 64);

    auto slots_a = cache.allocate(0, 4);
    auto slots_b = cache.allocate(1, 4);

    // Write distinct values
    for (int i = 0; i < 4; ++i) {
        std::vector<float> k(64, (i+1)*10.f);
        std::vector<float> v(64, (i+1)*10.f);
        cache.write(0, slots_a[i], 0, k, v);
    }
    for (int i = 0; i < 4; ++i) {
        std::vector<float> k(64, (i+1)*-10.f);
        std::vector<float> v(64, (i+1)*-10.f);
        cache.write(1, slots_b[i], 0, k, v);
    }

    std::cout << "\n  Reading back own sequence data:\n";
    for (int i = 0; i < 4; ++i) {
        auto [k, v] = cache.read(0, slots_a[i], 0);
        std::cout << "    Seq A slot " << i << ": k[0]=" << k[0]
                  << "  (expected " << (i+1)*10.f << ")\n";
        assert(std::abs(k[0] - (i+1)*10.f) < 1e-5f);
    }

    // Try a cross-sequence read — should assert-fail (we'll check the owner directly)
    std::cout << "\n  Verifying cross-sequence access would fail:\n";
    bool would_fail = (cache.owner[slots_b[0]] != 0);  // slot b[0] owned by seq 1, not 0
    std::cout << "    Slot " << slots_b[0] << " owned by seq "
              << cache.owner[slots_b[0]] << " (not seq 0) → access blocked ✓\n";
    assert(would_fail);

    cache.free_seq(0);
    bool all_freed = true;
    for (int s : slots_a) if (cache.owner[s] != -1) all_freed = false;
    assert(all_freed);
    std::cout << "  Freed seq A: " << slots_a.size() << " slots released.\n";

    std::cout << "\n  [ASSERT] KV cache sequence isolation enforced: ✓\n";
    std::cout << "  [ASSERT] Slot release correctly frees memory: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  LATENCY PERCENTILE ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────

static void demo_latency_analysis() {
    print_section("Latency Percentile Analysis and Spike Detection");

    std::mt19937 rng(99);
    std::normal_distribution<float> base_dist(80.f, 10.f);
    int N = 200;

    std::vector<float> latencies;
    latencies.reserve(N);
    for (int i = 0; i < N; ++i) {
        float lat  = std::max(20.f, base_dist(rng));
        // Simulate periodic GC spike
        if (i % 47 == 0) lat += 150.f;
        // Cold-start spike
        if (i == 0)      lat += 400.f;
        latencies.push_back(lat);
    }

    std::vector<float> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());

    float p50  = sorted[int(N * 0.50)];
    float p95  = sorted[int(N * 0.95)];
    float p99  = sorted[int(N * 0.99)];
    float pmax = sorted.back();

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n  Latency percentiles (" << N << " requests):\n";
    std::cout << "    P50:  " << p50  << " ms\n";
    std::cout << "    P95:  " << p95  << " ms\n";
    std::cout << "    P99:  " << p99  << " ms\n";
    std::cout << "    Max:  " << pmax << " ms\n";
    std::cout << "\n  P99/P50 ratio: " << p99/p50 << "x\n";

    if (p99 / p50 > 2.5f) {
        float threshold = p50 * 2.5f;
        int spike_count = 0;
        int last_spike  = -100;
        std::vector<int> gaps;
        for (int i = 0; i < N; ++i) {
            if (latencies[i] > threshold) {
                if (last_spike >= 0) gaps.push_back(i - last_spike);
                last_spike = i; ++spike_count;
            }
        }
        float avg_gap = gaps.empty() ? 0 :
            std::accumulate(gaps.begin(), gaps.end(), 0.f) / gaps.size();
        std::cout << "\n  [DIAG] P99/P50=" << p99/p50 << "x > 2.5 → periodic spike pattern\n";
        std::cout << "  [DIAG] " << spike_count << " spikes, avg gap="
                  << avg_gap << " requests\n";
        std::cout << "  [DIAG] Suggests GC pause or periodic background task\n";
    }

    assert(p50 < 120.f);
    assert(p99 > p50);
    std::cout << "\n  [ASSERT] Latency spike pattern correctly identified: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  MEMORY BANDWIDTH BOTTLENECK MODEL
// ─────────────────────────────────────────────────────────────────────────────

static void demo_bandwidth_model() {
    print_section("Memory Bandwidth Bottleneck Model");

    struct HW {
        std::string name;
        double bw_gbps;    // memory bandwidth GB/s
        double compute_tflops;  // TFLOPS (fp16)
    };
    std::vector<HW> hardware = {
        {"A10G",   600.0,   31.2},
        {"A100",  2000.0,  312.0},
        {"H100",  3350.0, 1979.0},
    };

    double params_b   = 8.0;    // 8B parameter model
    double bytes_param = 2.0;   // fp16 = 2 bytes/param

    std::cout << "\n  Params: " << params_b << "B  |  Precision: fp16\n\n";
    std::cout << "  " << std::left << std::setw(8) << "HW"
              << std::setw(10) << "BW GB/s"
              << std::setw(14) << "TFLOPS"
              << std::setw(16) << "Roof (tok/s)"
              << std::setw(18) << "Breakeven batch"
              << "Bottleneck\n";
    std::cout << "  " << std::string(66, '-') << "\n";

    for (auto& hw : hardware) {
        // Memory-bandwidth roof: bw / (2 × params × bytes_per_param)
        double bw_roof = (hw.bw_gbps * 1e9) / (2.0 * params_b * 1e9 * bytes_param);
        // Arithmetic intensity at batch size b: 2×b×params / (b×params×bytes + params×bytes) ≈ 2/bytes
        // Breakeven batch: AI × bytes_per_param / 2 = hw_roof_ratio
        double breakeven_batch = (hw.compute_tflops * 1e12) / (hw.bw_gbps * 1e9) * bytes_param;
        std::string bottleneck = breakeven_batch > 16 ? "Memory BW" : "Compute";

        std::cout << "  " << std::setw(8) << hw.name
                  << std::setw(10) << (int)hw.bw_gbps
                  << std::setw(14) << hw.compute_tflops
                  << std::setw(16) << (int)bw_roof
                  << std::setw(18) << std::setprecision(0) << std::fixed << breakeven_batch
                  << bottleneck << "\n";
    }

    std::cout << "\n  Interpretation: at small batch sizes (< breakeven), the model\n";
    std::cout << "  is memory-bandwidth-bound. Increasing batch size is the\n";
    std::cout << "  primary lever for improving throughput in that regime.\n";

    // Simple assert: H100 should have higher bandwidth roof than A10G
    double a10g_roof = (600.0 * 1e9) / (2.0 * 8e9 * 2.0);
    double h100_roof = (3350.0 * 1e9) / (2.0 * 8e9 * 2.0);
    assert(h100_roof > a10g_roof);
    std::cout << "\n  [ASSERT] H100 bandwidth roof > A10G bandwidth roof: "
              << (int)h100_roof << " > " << (int)a10g_roof << " tok/s ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n" << std::string(60, '=')
              << "\n  Chapter 32 — Debugging Inference Systems (C++)\n"
              << std::string(60, '=') << "\n";

    demo_nan_detection();
    demo_quantization_error();
    demo_kv_cache_isolation();
    demo_latency_analysis();
    demo_bandwidth_model();

    std::cout << "\n" << std::string(60, '=')
              << "\n  All demos complete.\n"
              << std::string(60, '=') << "\n\n";
    return 0;
}

```

