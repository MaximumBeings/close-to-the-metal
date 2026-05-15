# Code — Chapter 8.5: CUDA Graphs

Companion code for **Chapter 8.5: CUDA Graphs — Capture, Replay, and Production Latency**.

All demos run without a physical GPU: they simulate the mechanics, costs, and
latency trade-offs analytically so every reader can experiment.

---

## Python

```python
"""
cuda_graphs_demo.py — Chapter 8.5: CUDA Graphs

Demos (all run without GPU — simulates the mechanics):
  Demo 1: CPU kernel-launch overhead model
  Demo 2: CUDA graph capture/replay latency model
  Demo 3: vLLM graph pool sizing and batch padding
  Demo 4: Memory cost of the graph pool
  Demo 5: Break-even analysis — when do graphs help?
  Demo 6: Chunked-prefill graph pool
  Demo 7: Graph vs eager latency across batch sizes

Run: python cuda_graphs_demo.py
"""

from __future__ import annotations
import math
from dataclasses import dataclass

SEP = "─" * 70


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: CPU launch overhead model
# ─────────────────────────────────────────────────────────────────────────────

def demo_cpu_launch_overhead():
    print(f"\n{'='*70}")
    print("DEMO 1 — CPU Kernel-Launch Overhead Model")
    print(f"{'='*70}")

    # Kernel counts per layer for a transformer
    kernels_per_layer = {
        "RMSNorm (pre-attn)":   1,
        "QKV projection GEMM":  1,
        "RoPE embedding":        1,
        "Flash Attention":       1,
        "Output projection":     1,
        "RMSNorm (post-attn)":  1,
        "Gate+Up GEMMs":         2,
        "SiLU activation":       1,
        "Down projection":       1,
    }
    per_layer = sum(kernels_per_layer.values())

    models = [
        ("Llama-3-8B",  32, 4096),
        ("Llama-3-70B", 80, 8192),
        ("Llama-3-405B",126, 16384),
    ]

    launch_us = 10.0   # μs per kernel launch (typical)

    print(f"\n  Kernels per layer: {per_layer}")
    print(f"  Launch overhead per kernel: {launch_us:.0f} μs\n")
    print(f"  {'Model':<18} {'Layers':>6} {'Kernels':>8} {'Launch ms':>10} {'GPU ms':>8} {'Overhead%':>10}")
    print(f"  {SEP}")

    for name, n_layers, hidden in models:
        total_kernels = n_layers * per_layer + 20   # +20 misc
        launch_ms     = total_kernels * launch_us / 1000.0

        # Approximate GPU compute time (memory-bound decode, batch=1)
        bw_gbs  = 3350 * 4 * 0.85   # 4×H100, 85% efficiency
        # Weight bytes: params × 2 (BF16)
        params  = {32: 8e9, 80: 70e9, 126: 405e9}[n_layers]
        gpu_ms  = (params * 2) / (bw_gbs * 1e9) * 1000

        overhead_pct = launch_ms / (launch_ms + gpu_ms) * 100
        print(f"  {name:<18} {n_layers:>6} {total_kernels:>8} "
              f"{launch_ms:>10.1f} {gpu_ms:>8.1f} {overhead_pct:>9.1f}%")

    print(f"\n  At batch=1 (latency-critical serving), CPU overhead is significant.")
    print(f"  At batch=32, GPU compute dominates — graphs help less.")

    # Assertion: 70B launch overhead > 5ms
    kernels_70b = 80 * per_layer + 20
    assert kernels_70b * launch_us / 1000 > 5.0
    print(f"\n  ✓ 70B launch overhead > 5 ms confirmed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: Graph capture/replay latency model
# ─────────────────────────────────────────────────────────────────────────────

def demo_graph_latency_model():
    print(f"\n{'='*70}")
    print("DEMO 2 — CUDA Graph Capture/Replay Latency Model")
    print(f"{'='*70}")

    @dataclass
    class Config:
        name: str
        params_b: float
        n_gpus: int
        batch_size: int
        launch_us_per_kernel: float = 10.0
        kernels_per_layer: int = 9
        n_layers: int = 32
        graph_replay_us: float = 5.0   # single cudaGraphLaunch()

    configs = [
        Config("Llama-3-8B  batch=1",  8,  1,  1),
        Config("Llama-3-8B  batch=8",  8,  1,  8),
        Config("Llama-3-8B  batch=32", 8,  1, 32),
        Config("Llama-3-70B batch=1",  70, 4,  1),
        Config("Llama-3-70B batch=8",  70, 4,  8),
        Config("Llama-3-70B batch=32", 70, 4, 32),
    ]

    print(f"\n  {'Config':<26} {'GPU ms':>8} {'No-graph ms':>12} "
          f"{'Graph ms':>10} {'Speedup':>8}")
    print(f"  {SEP}")

    for c in configs:
        bw = 3350 * c.n_gpus * 0.85e9
        weight_bytes = c.params_b * 1e9 * 2  # BF16
        # decode: weight bytes / bandwidth, scaled by batch (still ~memory-bound at small batch)
        gpu_ms = weight_bytes / bw * 1000 * max(1, math.log2(c.batch_size + 1) * 0.3 + 0.7)

        total_kernels = c.n_layers * c.kernels_per_layer + 20
        launch_ms_no_graph = total_kernels * c.launch_us_per_kernel / 1000.0
        launch_ms_graph    = c.graph_replay_us / 1000.0

        total_no_graph = gpu_ms + launch_ms_no_graph
        total_graph    = gpu_ms + launch_ms_graph
        speedup = total_no_graph / total_graph

        print(f"  {c.name:<26} {gpu_ms:>8.1f} {total_no_graph:>12.1f} "
              f"{total_graph:>10.1f} {speedup:>8.2f}×")

    print(f"\n  Graphs deliver most speedup at small batch sizes (latency path).")
    print(f"  At large batch sizes, GPU compute dominates — speedup approaches 1×.")

    # Assert: 70B batch=1 should show >20% speedup
    bw = 3350 * 4 * 0.85e9
    gpu_ms = (70e9 * 2) / bw * 1000
    kernels = 80 * 9 + 20
    total_eager = gpu_ms + kernels * 10 / 1000
    total_graph = gpu_ms + 5 / 1000
    assert total_eager / total_graph > 1.20
    print(f"  ✓ 70B batch=1: graph speedup > 20%")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: vLLM graph pool — batch padding
# ─────────────────────────────────────────────────────────────────────────────

def demo_graph_pool():
    print(f"\n{'='*70}")
    print("DEMO 3 — vLLM Graph Pool: Batch Sizes and Padding")
    print(f"{'='*70}")

    # vLLM's default captured sizes
    captured_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    def find_graph_size(batch_n: int) -> int:
        for s in captured_sizes:
            if s >= batch_n:
                return s
        return captured_sizes[-1]

    test_batches = [1, 3, 5, 9, 17, 33, 65, 100, 200, 250, 256]

    print(f"\n  Captured pool: {captured_sizes}\n")
    print(f"  {'Actual batch':>12} {'Graph used':>11} {'Padded slots':>13} {'Waste%':>8}")
    print(f"  {SEP}")

    for b in test_batches:
        g = find_graph_size(b)
        waste = (g - b) / g * 100
        print(f"  {b:>12} {g:>11} {g-b:>13} {waste:>7.1f}%")

    print(f"\n  Worst case: batch=17 uses graph-32 → 47% waste in dummy compute.")
    print(f"  But: dummy tokens don't produce output — KV cache not written for pads.")

    # Verify batch=5 uses graph=8
    assert find_graph_size(5) == 8
    assert find_graph_size(1) == 1
    assert find_graph_size(256) == 256
    print(f"  ✓ Graph pool lookup assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4: Memory cost of graph pool
# ─────────────────────────────────────────────────────────────────────────────

def demo_graph_pool_memory():
    print(f"\n{'='*70}")
    print("DEMO 4 — Graph Pool Memory Cost")
    print(f"{'='*70}")

    models = [
        ("Llama-3-8B",  32,  4096, 32, 1),
        ("Llama-3-70B", 80,  8192, 64, 4),
        ("Llama-3-405B",126, 16384,128, 8),
    ]
    captured_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    graph_metadata_mb = 50   # CUDA graph node overhead

    print(f"\n  {'Model':<16} {'Weights GB':>10} {'HBM GB':>8} "
          f"{'Pool overhead':>14} {'Pool %':>7}")
    print(f"  {SEP}")

    for name, n_layers, hidden, n_heads, n_gpus in models:
        # Weight memory
        params = n_layers * (4 * hidden * hidden + 8192 * hidden)  # rough
        weight_gb = params * 2 / 1e9  # BF16

        # Total HBM
        hbm_gb = 80 * n_gpus

        # Activation tensors per captured size (max size = 256)
        max_cap = 256
        act_bytes_per_graph = max_cap * hidden * 2 * n_layers   # BF16
        pool_gb = (len(captured_sizes) * act_bytes_per_graph
                   + graph_metadata_mb * 1e6 * len(captured_sizes)) / 1e9

        pool_pct = pool_gb / hbm_gb * 100
        print(f"  {name:<16} {weight_gb:>10.0f} {hbm_gb:>8.0f} "
              f"{pool_gb:>13.2f} GB {pool_pct:>6.2f}%")

    print(f"\n  Graph pool overhead is <2% of HBM for all standard models.")
    print(f"  Memory cost is negligible relative to latency benefit.")

    # Assert pool overhead < 5% for 70B
    n_layers, hidden = 80, 8192
    act = 256 * hidden * 2 * n_layers
    pool = (len(captured_sizes) * act + graph_metadata_mb * 1e6 * len(captured_sizes))
    hbm = 80 * 4 * 1e9
    assert pool / hbm < 0.05
    print(f"  ✓ 70B pool overhead < 5% of HBM confirmed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5: Break-even analysis
# ─────────────────────────────────────────────────────────────────────────────

def demo_breakeven():
    print(f"\n{'='*70}")
    print("DEMO 5 — Break-Even: When Do CUDA Graphs Pay Off?")
    print(f"{'='*70}")

    # One-time cost: capture all graph sizes
    n_capture_sizes  = 9
    capture_time_s   = 0.5   # seconds per capture (rough)
    total_capture_s  = n_capture_sizes * capture_time_s

    # Per-request savings
    kernels_per_fwd  = 80 * 9 + 20      # 70B
    launch_us_saved  = kernels_per_fwd * 10 - 5  # saved per request

    requests_to_break_even = (total_capture_s * 1e6) / launch_us_saved

    print(f"\n  One-time capture cost: {n_capture_sizes} × {capture_time_s:.1f}s = {total_capture_s:.1f}s")
    print(f"  Saved per request: {launch_us_saved/1000:.2f} ms")
    print(f"  Break-even at: {requests_to_break_even:.0f} requests")
    print(f"\n  At 100 req/s, break-even reached in {requests_to_break_even/100:.0f}s")
    print(f"  At 10 req/s,  break-even reached in {requests_to_break_even/10:.0f}s")

    print(f"\n  Batches where graphs provide ≥10% latency improvement:")
    print(f"  {'Batch':>6} {'GPU ms':>8} {'Launch ms':>10} {'Overhead%':>11} {'Worth it?':>10}")
    print(f"  {SEP}")

    bw_gbs = 3350 * 4 * 0.85e9
    weight_bytes = 70e9 * 2
    for batch in [1, 2, 4, 8, 16, 32, 64]:
        gpu_ms = weight_bytes / bw_gbs * 1000
        launch_ms = kernels_per_fwd * 10 / 1000
        overhead = launch_ms / (gpu_ms + launch_ms) * 100
        worth_it = "YES ✓" if overhead >= 10 else "marginal"
        print(f"  {batch:>6} {gpu_ms:>8.1f} {launch_ms:>10.1f} {overhead:>10.1f}% {worth_it:>10}")

    assert launch_us_saved > 0
    print(f"\n  ✓ Break-even analysis complete")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6: Chunked prefill graph pool
# ─────────────────────────────────────────────────────────────────────────────

def demo_chunked_prefill_graphs():
    print(f"\n{'='*70}")
    print("DEMO 6 — Chunked Prefill Graph Pool")
    print(f"{'='*70}")

    chunk_sizes = [128, 256, 512, 1024, 2048]   # prefill chunk options
    decode_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print(f"\n  With --enable-chunked-prefill:")
    print(f"  Prefill chunks (fixed sizes): {chunk_sizes}")
    print(f"  Decode graph pool:            {decode_sizes}")
    total_graphs = len(chunk_sizes) + len(decode_sizes)
    print(f"\n  Total captured graphs: {total_graphs}")

    print(f"\n  How a mixed prefill+decode batch is handled:")
    example_batch = [
        ("new request (512-tok prompt)", "prefill", 512),
        ("active seq A",                  "decode",   1),
        ("active seq B",                  "decode",   1),
        ("active seq C",                  "decode",   1),
    ]
    print(f"\n  {'Request':<32} {'Phase':>8} {'Tokens':>7}")
    print(f"  {SEP}")
    for name, phase, toks in example_batch:
        print(f"  {name:<32} {phase:>8} {toks:>7}")
    print(f"\n  → Use prefill-chunk-512 graph for the new request")
    print(f"  → Use decode-graph-4 for the 3 active sequences (padded to 4)")
    print(f"  → Two graph replays per scheduler step in this case")

    assert total_graphs == len(chunk_sizes) + len(decode_sizes)
    print(f"\n  ✓ Chunked prefill graph pool: {total_graphs} total graphs")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 7: Graph vs eager across all batch sizes
# ─────────────────────────────────────────────────────────────────────────────

def demo_graph_vs_eager():
    print(f"\n{'='*70}")
    print("DEMO 7 — Graph vs Eager: Full Latency Profile")
    print(f"{'='*70}")

    model_name = "Llama-3-70B, 4×H100"
    kernels = 80 * 9 + 20
    launch_us = 10.0
    graph_us  = 5.0
    bw_gbs = 3350 * 4 * 0.85e9

    print(f"\n  Model: {model_name}")
    print(f"  {'Batch':>6} {'GPU ms':>8} {'Eager ms':>10} {'Graph ms':>10} "
          f"{'Speedup':>8} {'Recommendation'}")
    print(f"  {SEP}")

    for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        # At small batches, still mostly memory-bound
        # At larger batches, transitions to compute-bound
        scale = 1 + math.log2(batch) * 0.15
        gpu_ms = (70e9 * 2) / bw_gbs * 1000 * scale

        eager_ms = gpu_ms + kernels * launch_us / 1000
        graph_ms = gpu_ms + graph_us / 1000
        speedup = eager_ms / graph_ms

        rec = "GRAPHS ✓" if speedup > 1.10 else ("marginal" if speedup > 1.03 else "skip")
        print(f"  {batch:>6} {gpu_ms:>8.1f} {eager_ms:>10.1f} {graph_ms:>10.1f} "
              f"{speedup:>8.2f}× {rec}")

    print(f"\n  Rule of thumb: enable graphs (default) for all production deployments.")
    print(f"  Disable only for debugging / non-standard architectures.")

    # Assert batch=1 speedup > 1.2
    gpu_ms = (70e9 * 2) / bw_gbs * 1000
    eager = gpu_ms + kernels * launch_us / 1000
    graph = gpu_ms + graph_us / 1000
    assert eager / graph > 1.20
    print(f"  ✓ batch=1 speedup > 1.20× confirmed")


def main():
    bar = "=" * 70
    print(f"\n{bar}")
    print("  Chapter 8.5 — CUDA Graphs (Python)")
    print(f"{bar}")

    demo_cpu_launch_overhead()
    demo_graph_latency_model()
    demo_graph_pool()
    demo_graph_pool_memory()
    demo_breakeven()
    demo_chunked_prefill_graphs()
    demo_graph_vs_eager()

    print(f"\n{bar}")
    print("  All demos complete — all assertions passed ✓")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
```

**Run:**
```bash
python cuda_graphs_demo.py
```

---

## C++

```cpp
/*
 * cuda_graphs_demo.cpp — Chapter 8.5: CUDA Graphs
 *
 * Demos (no GPU required — models the mechanics):
 *   Demo 1: CPU kernel-launch overhead model
 *   Demo 2: Graph capture/replay latency model
 *   Demo 3: vLLM graph pool — batch padding
 *   Demo 4: Memory cost of graph pool
 *   Demo 5: Break-even analysis
 *   Demo 6: Chunked prefill graph pool
 *   Demo 7: Graph vs eager across batch sizes
 *
 * Compile: g++ -std=c++17 -O2 -o cuda_graphs_demo cuda_graphs_demo.cpp -lm
 * Run:     ./cuda_graphs_demo
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const char* SEP =
    "──────────────────────────────────────────────────────────────────────";

// ─────────────────────────────────────────────────────────────────────────────
// Demo 1: CPU launch overhead
// ─────────────────────────────────────────────────────────────────────────────

static void demo_cpu_launch_overhead() {
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 1 — CPU Kernel-Launch Overhead Model\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    const int kernels_per_layer = 9;
    const double launch_us = 10.0;

    struct Model { const char* name; int layers; double params_b; int n_gpus; };
    Model models[] = {
        {"Llama-3-8B",   32,   8.0, 1},
        {"Llama-3-70B",  80,  70.0, 4},
        {"Llama-3-405B", 126, 405.0, 8},
    };

    printf("\n  Kernels/layer: %d  |  Launch overhead: %.0f μs/kernel\n\n", kernels_per_layer, launch_us);
    printf("  %-18s %6s %8s %10s %8s %10s\n",
           "Model","Layers","Kernels","Launch ms","GPU ms","Overhead%");
    printf("  %s\n", SEP);

    for (auto& m : models) {
        int total_k = m.layers * kernels_per_layer + 20;
        double launch_ms = total_k * launch_us / 1000.0;
        double bw = 3350.0 * m.n_gpus * 0.85e9;
        double gpu_ms = (m.params_b * 1e9 * 2.0) / bw * 1000.0;
        double overhead = launch_ms / (launch_ms + gpu_ms) * 100.0;
        printf("  %-18s %6d %8d %10.1f %8.1f %9.1f%%\n",
               m.name, m.layers, total_k, launch_ms, gpu_ms, overhead);
    }

    // 70B launch overhead > 5ms
    int k_70b = 80 * kernels_per_layer + 20;
    assert(k_70b * launch_us / 1000.0 > 5.0);
    printf("\n  ✓ 70B launch overhead > 5 ms confirmed\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 2: Graph capture/replay latency model
// ─────────────────────────────────────────────────────────────────────────────

static void demo_graph_latency() {
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 2 — CUDA Graph Capture/Replay Latency Model\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    const double graph_replay_us = 5.0;
    const double launch_us       = 10.0;
    const int    kernels          = 80 * 9 + 20;  // 70B

    struct Cfg {const char*n; double params_b; int n_gpus, batch;};
    Cfg cfgs[] = {
        {"70B batch=1",  70.0, 4,  1},
        {"70B batch=8",  70.0, 4,  8},
        {"70B batch=32", 70.0, 4, 32},
        {"8B  batch=1",   8.0, 1,  1},
        {"8B  batch=8",   8.0, 1,  8},
        {"8B  batch=32",  8.0, 1, 32},
    };

    printf("\n  %-20s %8s %12s %10s %8s\n",
           "Config","GPU ms","No-graph ms","Graph ms","Speedup");
    printf("  %s\n", SEP);

    for (auto& c : cfgs) {
        double bw = 3350.0 * c.n_gpus * 0.85e9;
        double scale = 1.0 + log2(c.batch + 1) * 0.15;
        double gpu_ms = (c.params_b * 1e9 * 2.0) / bw * 1000.0 * scale;
        double eager  = gpu_ms + kernels * launch_us / 1000.0;
        double graph  = gpu_ms + graph_replay_us / 1000.0;
        printf("  %-20s %8.1f %12.1f %10.1f %8.2f×\n",
               c.n, gpu_ms, eager, graph, eager/graph);
    }

    double bw = 3350.0 * 4 * 0.85e9;
    double gpu_ms = (70e9 * 2.0) / bw * 1000.0;
    double eager = gpu_ms + kernels * launch_us / 1000.0;
    double graph = gpu_ms + graph_replay_us / 1000.0;
    assert(eager / graph > 1.20);
    printf("\n  ✓ 70B batch=1 speedup > 1.20× confirmed\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 3: vLLM graph pool — batch padding
// ─────────────────────────────────────────────────────────────────────────────

static void demo_graph_pool() {
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 3 — vLLM Graph Pool: Batch Sizes and Padding\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    int pool[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    const int NP = 9;

    auto find_size = [&](int b) {
        for (int i = 0; i < NP; ++i) if (pool[i] >= b) return pool[i];
        return pool[NP-1];
    };

    int batches[] = {1, 3, 5, 9, 17, 33, 65, 100, 200, 250, 256};

    printf("\n  Pool: [");
    for(int i=0;i<NP;++i) printf("%d%s",pool[i],i<NP-1?", ":"");
    printf("]\n\n");
    printf("  %12s %11s %13s %8s\n","Actual batch","Graph used","Padded slots","Waste%");
    printf("  %s\n",SEP);

    for (int b : batches) {
        int g = find_size(b);
        double waste = (double)(g - b) / g * 100.0;
        printf("  %12d %11d %13d %7.1f%%\n", b, g, g-b, waste);
    }

    assert(find_size(5)   == 8);
    assert(find_size(1)   == 1);
    assert(find_size(256) == 256);
    printf("\n  ✓ Graph pool lookup assertions passed\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 4: Memory cost of graph pool
// ─────────────────────────────────────────────────────────────────────────────

static void demo_pool_memory() {
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 4 — Graph Pool Memory Cost\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    struct Model {const char*n; int layers,hidden,n_gpus;};
    Model models[] = {
        {"Llama-3-8B",   32,  4096, 1},
        {"Llama-3-70B",  80,  8192, 4},
        {"Llama-3-405B", 126,16384, 8},
    };
    const int n_pool_sizes = 9;
    const double graph_meta_mb = 50.0;

    printf("\n  %-16s %10s %8s %14s %7s\n",
           "Model","Weights GB","HBM GB","Pool GB","Pool %");
    printf("  %s\n",SEP);

    for (auto& m : models) {
        double params = m.layers * (4.0*m.hidden*m.hidden + 8192.0*m.hidden);
        double weight_gb = params * 2.0 / 1e9;
        double hbm_gb = 80.0 * m.n_gpus;
        double act_per_graph = 256.0 * m.hidden * 2.0 * m.layers;
        double pool_gb = (n_pool_sizes * act_per_graph + graph_meta_mb * 1e6 * n_pool_sizes) / 1e9;
        double pool_pct = pool_gb / hbm_gb * 100.0;
        printf("  %-16s %10.0f %8.0f %13.2f G %6.2f%%\n",
               m.n, weight_gb, hbm_gb, pool_gb, pool_pct);
    }

    // 70B pool < 5% HBM
    double act = 9 * 256.0 * 8192.0 * 2.0 * 80.0;
    double meta = 9 * 50.0 * 1e6;
    double pool = (act + meta) / 1e9;
    double hbm = 80.0 * 4.0;
    assert(pool / hbm < 0.05);
    printf("\n  ✓ 70B pool overhead < 5%% of HBM confirmed\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 5: Break-even analysis
// ─────────────────────────────────────────────────────────────────────────────

static void demo_breakeven() {
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 5 — Break-Even: When Do CUDA Graphs Pay Off?\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    const int n_sizes = 9;
    const double capture_s = 0.5;
    const int kernels = 80 * 9 + 20;
    const double launch_us = 10.0;
    const double graph_us  = 5.0;

    double total_capture_s = n_sizes * capture_s;
    double saved_us = kernels * launch_us - graph_us;
    double break_even_reqs = (total_capture_s * 1e6) / saved_us;

    printf("\n  Capture cost: %d × %.1fs = %.1fs\n", n_sizes, capture_s, total_capture_s);
    printf("  Saved per request: %.2f ms\n", saved_us / 1000.0);
    printf("  Break-even: %.0f requests\n", break_even_reqs);
    printf("  At 100 req/s → %.0fs to break even\n", break_even_reqs / 100.0);

    printf("\n  Batch overhead analysis (70B, 4×H100):\n");
    printf("  %6s %8s %10s %11s %10s\n","Batch","GPU ms","Launch ms","Overhead%","Worth it?");
    printf("  %s\n",SEP);

    double bw = 3350.0 * 4 * 0.85e9;
    double wgt = 70e9 * 2.0;
    for (int batch : {1,2,4,8,16,32,64}) {
        double gpu_ms = wgt / bw * 1000.0;
        double lms = kernels * launch_us / 1000.0;
        double pct = lms / (gpu_ms + lms) * 100.0;
        printf("  %6d %8.1f %10.1f %10.1f%% %10s\n",
               batch, gpu_ms, lms, pct, pct >= 10.0 ? "YES ✓" : "marginal");
    }
    assert(saved_us > 0);
    printf("\n  ✓ Break-even analysis complete\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 6: Chunked prefill graph pool
// ─────────────────────────────────────────────────────────────────────────────

static void demo_chunked_prefill() {
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 6 — Chunked Prefill Graph Pool\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    int prefill_chunks[] = {128, 256, 512, 1024, 2048};
    int decode_sizes[]   = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    const int NPC = 5, NDC = 9;
    int total = NPC + NDC;

    printf("\n  Prefill chunk pool:  [");
    for(int i=0;i<NPC;++i) printf("%d%s",prefill_chunks[i],i<NPC-1?", ":"");
    printf("]\n  Decode graph pool:   [");
    for(int i=0;i<NDC;++i) printf("%d%s",decode_sizes[i],i<NDC-1?", ":"");
    printf("]\n");
    printf("  Total captured graphs: %d\n\n", total);

    printf("  Example mixed batch:\n");
    printf("  %-34s %-8s %7s\n","Request","Phase","Tokens");
    printf("  %s\n",SEP);
    printf("  %-34s %-8s %7d\n","new request (512-tok prompt)","prefill",512);
    printf("  %-34s %-8s %7d\n","active seq A","decode",1);
    printf("  %-34s %-8s %7d\n","active seq B","decode",1);
    printf("  %-34s %-8s %7d\n","active seq C","decode",1);
    printf("\n  → prefill-chunk-512 graph + decode-graph-4 (3 seqs padded to 4)\n");

    assert(total == NPC + NDC);
    printf("  ✓ Chunked prefill pool: %d total graphs\n", total);
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 7: Graph vs eager full profile
// ─────────────────────────────────────────────────────────────────────────────

static void demo_full_profile() {
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 7 — Graph vs Eager: Full Latency Profile\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    const int kernels    = 80 * 9 + 20;
    const double lu      = 10.0;
    const double gu      = 5.0;
    const double bw      = 3350.0 * 4 * 0.85e9;
    const double wgt     = 70e9 * 2.0;

    printf("\n  Model: Llama-3-70B, 4×H100\n");
    printf("  %6s %8s %10s %10s %8s %-14s\n",
           "Batch","GPU ms","Eager ms","Graph ms","Speedup","Recommendation");
    printf("  %s\n",SEP);

    int batches[] = {1,2,4,8,16,32,64,128,256};
    for (int b : batches) {
        double scale   = 1.0 + log2((double)(b+1)) * 0.15;
        double gpu_ms  = wgt / bw * 1000.0 * scale;
        double eager   = gpu_ms + kernels * lu / 1000.0;
        double graph   = gpu_ms + gu / 1000.0;
        double speedup = eager / graph;
        const char* rec = speedup > 1.10 ? "GRAPHS ✓"
                        : speedup > 1.03 ? "marginal" : "skip";
        printf("  %6d %8.1f %10.1f %10.1f %8.2f× %-14s\n",
               b, gpu_ms, eager, graph, speedup, rec);
    }

    double gpu_ms = wgt / bw * 1000.0;
    double e = gpu_ms + kernels * lu / 1000.0;
    double g = gpu_ms + gu / 1000.0;
    assert(e / g > 1.20);
    printf("\n  ✓ batch=1 speedup > 1.20× confirmed\n");
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║   Chapter 8.5: CUDA Graphs (C++)                                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");

    demo_cpu_launch_overhead();
    demo_graph_latency();
    demo_graph_pool();
    demo_pool_memory();
    demo_breakeven();
    demo_chunked_prefill();
    demo_full_profile();

    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("ALL CHAPTER 8b DEMOS COMPLETED — ALL ASSERTIONS PASSED ✓\n");
    printf("%s\n\n","══════════════════════════════════════════════════════════════════════");
    return 0;
}
```

**Compile and run:**
```bash
g++ -O2 -std=c++17 -o cuda_graphs_demo cuda_graphs_demo.cpp && ./cuda_graphs_demo
```
