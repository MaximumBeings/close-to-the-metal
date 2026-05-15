# Chapter 18: Disaggregated Prefill and Decode — Companion Code

## Python — `disaggregated_demo.py`

```python
"""
disaggregated_demo.py — Chapter 18: Disaggregated Prefill and Decode

Demonstrates:
  1. KV cache size calculator per context length and model
  2. KV transfer cost model (latency vs. network bandwidth)
  3. Cluster sizing: optimal prefill-to-decode GPU ratio
  4. Latency budget breakdown (prefill + transfer + decode)
  5. Global KV store hit-rate simulator
  6. vLLM disaggregated config generator
  7. Prefill starvation simulator (co-located vs. disaggregated comparison)

No external dependencies beyond the Python standard library.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Model specifications
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    n_layers: int
    n_kv_heads: int
    d_head: int
    n_params_b: float          # billions
    prefill_tps_h100: float    # tokens/s on H100 SXM at large batch
    decode_tps_h100: float     # tokens/s on H100 SXM at batch=1
    bytes_per_element: int = 2  # BF16


MODELS = {
    "llama-3-8b": ModelSpec(
        name="Llama-3-8B",
        n_layers=32, n_kv_heads=8, d_head=128,
        n_params_b=8.0,
        prefill_tps_h100=12_500,
        decode_tps_h100=209,
    ),
    "llama-3-70b": ModelSpec(
        name="Llama-3-70B",
        n_layers=80, n_kv_heads=8, d_head=128,
        n_params_b=70.0,
        prefill_tps_h100=3_200,
        decode_tps_h100=52,
    ),
    "llama-3-8b-fp8": ModelSpec(
        name="Llama-3-8B (FP8 KV)",
        n_layers=32, n_kv_heads=8, d_head=128,
        n_params_b=8.0,
        prefill_tps_h100=14_000,
        decode_tps_h100=230,
        bytes_per_element=1,   # FP8
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. KV cache size calculator
# ──────────────────────────────────────────────────────────────────────────────

def kv_cache_bytes(model: ModelSpec, n_tokens: int) -> int:
    """Total KV cache bytes for a single sequence of n_tokens."""
    return (2  # K and V
            * model.n_layers
            * model.n_kv_heads
            * model.d_head
            * n_tokens
            * model.bytes_per_element)


def print_kv_size_table():
    print("\n" + "=" * 68)
    print("  KV Cache Size by Context Length")
    print("=" * 68)
    header = f"  {'Model':<22} {'1K tokens':>10} {'4K tokens':>10} {'32K tokens':>12} {'128K tokens':>13}"
    print(header)
    print("  " + "-" * 21 + "  " + "-" * 9 + "  " + "-" * 9 +
          "  " + "-" * 11 + "  " + "-" * 12)

    for key, m in MODELS.items():
        def fmt(b):
            if b >= 1024**3:
                return f"{b/1024**3:.1f} GB"
            return f"{b/1024**2:.0f} MB"

        row = (f"  {m.name:<22}"
               f" {fmt(kv_cache_bytes(m, 1024)):>10}"
               f" {fmt(kv_cache_bytes(m, 4096)):>10}"
               f" {fmt(kv_cache_bytes(m, 32768)):>12}"
               f" {fmt(kv_cache_bytes(m, 131072)):>13}")
        print(row)
    print("=" * 68)


# ──────────────────────────────────────────────────────────────────────────────
# 2. KV transfer cost model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NetworkSpec:
    name: str
    bandwidth_gbps: float   # GB/s
    rtt_ms: float           # round-trip time (ms)


NETWORKS = {
    "1GbE":      NetworkSpec("1 GbE",            0.125,  1.0),
    "10GbE":     NetworkSpec("10 GbE",            1.25,  0.5),
    "100GbE":    NetworkSpec("100 GbE",          12.5,   0.1),
    "IB-HDR":    NetworkSpec("InfiniBand HDR",   25.0,   0.05),
    "IB-NDR":    NetworkSpec("InfiniBand NDR",   50.0,   0.04),
    "NVLink-C2C":NetworkSpec("NVLink C2C",      900.0,   0.001),
}


def transfer_latency_ms(kv_bytes: int, net: NetworkSpec) -> float:
    """Transfer latency in ms: bandwidth-delay + RTT."""
    transfer_ms = (kv_bytes / (net.bandwidth_gbps * 1e9)) * 1000.0
    return transfer_ms + net.rtt_ms


def print_transfer_cost_table():
    model = MODELS["llama-3-8b"]
    context_tokens = [1024, 4096, 32768]

    print("\n" + "=" * 72)
    print("  KV Transfer Latency (ms)  |  Llama-3-8B BF16")
    print("=" * 72)
    header = f"  {'Network':<20} {'1K ctx':>10} {'4K ctx':>10} {'32K ctx':>12}  Usable?"
    print(header)
    print("  " + "-" * 19 + "  " + "-" * 9 + "  " + "-" * 9 +
          "  " + "-" * 11 + "  " + "-" * 7)

    # Decode step time on H100 at batch=1
    decode_step_ms = 1000.0 / model.decode_tps_h100

    for key, net in NETWORKS.items():
        vals = []
        for t in context_tokens:
            b = kv_cache_bytes(model, t)
            lat = transfer_latency_ms(b, net)
            vals.append(lat)

        # Usable if transfer for 4K < 10 decode steps
        usable = "YES" if vals[1] < decode_step_ms * 10 else "NO "

        row = (f"  {net.name:<20}"
               f" {vals[0]:>9.1f}ms"
               f" {vals[1]:>9.1f}ms"
               f" {vals[2]:>11.1f}ms"
               f"  {usable}")
        print(row)

    print("=" * 72)
    print(f"  Note: decode step time = {decode_step_ms:.1f} ms/token (H100, batch=1)")
    print(f"  'Usable' = 4K transfer < {decode_step_ms * 10:.0f} ms (10 decode steps)")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Cluster sizing: optimal prefill/decode ratio
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkloadProfile:
    name: str
    avg_prompt_tokens: int
    avg_output_tokens: int
    description: str


WORKLOADS = [
    WorkloadProfile("Short chat",    200,   150, "Conversational assistant"),
    WorkloadProfile("Code complete", 800,   300, "Code completion, context window"),
    WorkloadProfile("RAG medium",   2000,   500, "4K retrieved context + query"),
    WorkloadProfile("RAG long",    32000,  2000, "Full document RAG synthesis"),
    WorkloadProfile("Summarize",   16000,   400, "Long document summarization"),
]


def compute_ratio(profile: WorkloadProfile, model: ModelSpec) -> float:
    """
    Ratio of decode GPUs to prefill GPUs needed to balance throughput.

    prefill_time = prompt_tokens / prefill_tps
    decode_time  = output_tokens / decode_tps
    ratio        = decode_time / prefill_time
    """
    prefill_time = profile.avg_prompt_tokens / model.prefill_tps_h100
    decode_time  = profile.avg_output_tokens / model.decode_tps_h100
    if prefill_time <= 0:
        return 0.0
    return decode_time / prefill_time


def print_cluster_sizing():
    model = MODELS["llama-3-8b"]
    print("\n" + "=" * 70)
    print(f"  Cluster Sizing: Decode/Prefill GPU Ratio  |  {model.name}  |  H100")
    print("=" * 70)
    header = f"  {'Workload':<16} {'Prompt':>7} {'Output':>7} {'Pf time':>9} {'Dec time':>10} {'Ratio':>8}  Recommendation"
    print(header)
    print("  " + "-" * 15 + "  " + "-" * 6 + "  " + "-" * 6 +
          "  " + "-" * 8 + "  " + "-" * 9 + "  " + "-" * 7 + "  " + "-" * 14)

    for w in WORKLOADS:
        pf_ms = w.avg_prompt_tokens / model.prefill_tps_h100 * 1000
        dec_ms = w.avg_output_tokens / model.decode_tps_h100 * 1000
        ratio = compute_ratio(w, model)
        rec = f"1 pf : {ratio:.0f} dec"
        print(f"  {w.name:<16} {w.avg_prompt_tokens:>7,} {w.avg_output_tokens:>7,}"
              f" {pf_ms:>8.0f}ms {dec_ms:>9.0f}ms {ratio:>8.1f}  {rec}")

    print("=" * 70)
    print("  Note: ratio = decode_time / prefill_time.")
    print("  Mixed workload: compute weighted average over request distribution.")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Latency budget breakdown
# ──────────────────────────────────────────────────────────────────────────────

def print_latency_budget():
    model = MODELS["llama-3-8b"]
    net   = NETWORKS["IB-HDR"]

    print("\n" + "=" * 64)
    print("  TTFT Latency Budget: Co-located vs. Disaggregated")
    print("  Llama-3-8B  |  H100 SXM  |  InfiniBand HDR")
    print("=" * 64)

    ctx_sizes = [512, 2048, 8192, 32768]
    print(f"\n  {'Context':>8}  {'Co-located':>13}  {'Disagg (pf)':>13}  "
          f"{'KV xfer':>10}  {'Disagg TTFT':>13}  {'Speedup':>8}")
    print("  " + "-" * 7 + "  " + "-" * 12 + "  " + "-" * 12 +
          "  " + "-" * 9 + "  " + "-" * 12 + "  " + "-" * 7)

    for t in ctx_sizes:
        # Co-located: batch=1 prefill, no batching benefit
        colocated_tps = model.prefill_tps_h100 / 8  # ~1/8 efficiency at batch=1
        ttft_colocated = t / colocated_tps * 1000

        # Disaggregated: prefill at full large-batch efficiency
        ttft_pf = t / model.prefill_tps_h100 * 1000
        kv_b = kv_cache_bytes(model, t)
        xfer_ms = transfer_latency_ms(kv_b, net)
        first_decode_ms = 1000.0 / model.decode_tps_h100
        ttft_disagg = ttft_pf + xfer_ms + first_decode_ms

        speedup = ttft_colocated / ttft_disagg

        print(f"  {t:>7,}  {ttft_colocated:>11.0f}ms  {ttft_pf:>11.0f}ms  "
              f"{xfer_ms:>8.1f}ms  {ttft_disagg:>11.0f}ms  {speedup:>7.1f}x")

    print("=" * 64)
    print("  Co-located uses single-request prefill throughput (~1/8 of batch peak)")
    print("  Disaggregated uses full batch-mode prefill throughput")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Global KV store hit-rate simulator
# ──────────────────────────────────────────────────────────────────────────────

def simulate_global_kv_store(
    n_requests: int = 1000,
    n_unique_documents: int = 50,
    avg_doc_tokens: int = 8192,
    store_capacity_gb: float = 200.0,
    model: Optional[ModelSpec] = None,
    seed: int = 42,
) -> None:
    """
    Simulate a RAG workload where requests reuse a pool of documents.
    Track KV store hit rate and tokens saved.
    """
    if model is None:
        model = MODELS["llama-3-8b"]

    rng = random.Random(seed)
    # Zipf-like distribution: first few documents are much more popular
    weights = [1.0 / (i + 1) for i in range(n_unique_documents)]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    kv_per_doc_bytes = kv_cache_bytes(model, avg_doc_tokens)
    store_capacity_bytes = int(store_capacity_gb * 1e9)
    max_docs_in_store = store_capacity_bytes // kv_per_doc_bytes

    kv_store: dict = {}   # doc_id -> True (cached)
    hits = 0
    tokens_saved = 0
    tokens_total = 0

    for _ in range(n_requests):
        # Pick document using Zipf weights
        r = rng.random()
        cumulative = 0.0
        doc_id = n_unique_documents - 1
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                doc_id = i
                break

        tokens_total += avg_doc_tokens

        if doc_id in kv_store:
            hits += 1
            tokens_saved += avg_doc_tokens
        else:
            # Evict LRU if at capacity (simple: evict highest id)
            if len(kv_store) >= max_docs_in_store:
                evict = max(kv_store.keys())
                del kv_store[evict]
            kv_store[doc_id] = True

    hit_rate = hits / n_requests * 100
    savings_pct = tokens_saved / tokens_total * 100
    prefill_time_saved_s = tokens_saved / model.prefill_tps_h100

    print("\n" + "=" * 62)
    print("  Global KV Store Simulation")
    print("=" * 62)
    print(f"  Requests simulated     : {n_requests:,}")
    print(f"  Unique documents       : {n_unique_documents}")
    print(f"  Avg doc length         : {avg_doc_tokens:,} tokens")
    print(f"  KV per doc (BF16)      : {kv_per_doc_bytes / 1e6:.0f} MB")
    print(f"  Store capacity         : {store_capacity_gb:.0f} GB"
          f"  ({max_docs_in_store} docs max)")
    print(f"  Distribution           : Zipf (popular docs reused more)")
    print()
    print(f"  Cache hits             : {hits:,} / {n_requests:,}  ({hit_rate:.1f}%)")
    print(f"  Prefill tokens saved   : {tokens_saved:,} / {tokens_total:,}"
          f"  ({savings_pct:.1f}%)")
    print(f"  GPU-seconds saved      : {prefill_time_saved_s:.1f} s"
          f"  ({prefill_time_saved_s / 3600:.2f} GPU-hours)")
    print("=" * 62)

    # ASCII hit rate bar
    bar_w = 40
    bar = "█" * int(hit_rate / 100 * bar_w)
    print(f"\n  Hit rate: {hit_rate:5.1f}%  |{bar:<{bar_w}}|")


# ──────────────────────────────────────────────────────────────────────────────
# 6. vLLM disaggregated config generator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DisaggConfig:
    model_id: str
    max_model_len: int
    prefill_max_seqs: int      = 8
    decode_max_seqs: int       = 128
    prefill_gpu_util: float    = 0.85
    decode_gpu_util: float     = 0.90
    kv_connector: str          = "MooncakeConnector"
    kv_buffer_size_gb: float   = 2.0
    tensor_parallel_size: int  = 1
    kv_dtype: str              = "bf16"


def generate_config(cfg: DisaggConfig) -> None:
    print("\n" + "=" * 60)
    print("  vLLM Disaggregated Deployment Config")
    print("=" * 60)

    def yaml_block(role: str, rank: int, max_seqs: int,
                   gpu_util: float, buf_gb: float) -> str:
        buf_bytes = int(buf_gb * 1e9)
        return f"""
# {role.upper()} WORKER  (rank {rank})
model: {cfg.model_id}
max_model_len: {cfg.max_model_len}
max_num_seqs: {max_seqs}
gpu_memory_utilization: {gpu_util}
tensor_parallel_size: {cfg.tensor_parallel_size}
kv_transfer_config:
  kv_connector: {cfg.kv_connector}
  kv_role: kv_{"producer" if role == "prefill" else "consumer"}
  kv_rank: {rank}
  kv_parallel_size: 1
  kv_buffer_device: cuda
  kv_buffer_size: {buf_bytes:.0e}
  kv_quant_policy: {0 if cfg.kv_dtype == "bf16" else 8}
""".strip()

    print("\n--- prefill_worker.yaml ---\n")
    print(yaml_block("prefill", 0, cfg.prefill_max_seqs,
                     cfg.prefill_gpu_util, cfg.kv_buffer_size_gb))
    print("\n--- decode_worker.yaml ---\n")
    print(yaml_block("decode", 1, cfg.decode_max_seqs,
                     cfg.decode_gpu_util, cfg.kv_buffer_size_gb * 2))
    print("\n" + "=" * 60)

    # Validation warnings
    print("\n  Config validation:")
    ok = True
    if cfg.prefill_max_seqs > 16:
        print(f"  WARN: prefill max_seqs={cfg.prefill_max_seqs} is high — "
              "prefer ≤16 for low TTFT")
        ok = False
    if cfg.decode_max_seqs < 32:
        print(f"  WARN: decode max_seqs={cfg.decode_max_seqs} is low — "
              "prefer ≥64 to saturate BW")
        ok = False
    if cfg.tensor_parallel_size > 1:
        print(f"  NOTE: TP={cfg.tensor_parallel_size} — ensure SAME TP degree"
              " on both workers or KV shards will mismatch")
    if ok:
        print("  OK  Config looks reasonable.")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Prefill starvation simulator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RequestEvent:
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int


def simulate_colocated(requests: List[RequestEvent],
                       model: ModelSpec) -> List[float]:
    """
    Simulate co-located scheduler: prefill runs before decode, no interleaving.
    Returns TTFT per request (ms).
    """
    queue = sorted(requests, key=lambda r: r.arrival_ms)
    ttfts: List[float] = []
    clock = 0.0

    for req in queue:
        # Wait until after arrival
        clock = max(clock, req.arrival_ms)
        # Prefill
        prefill_ms = req.prompt_tokens / model.prefill_tps_h100 * 1000
        clock += prefill_ms
        ttfts.append(clock - req.arrival_ms)

    return ttfts


def simulate_disaggregated(requests: List[RequestEvent],
                            model: ModelSpec,
                            n_prefill_workers: int,
                            network: NetworkSpec) -> List[float]:
    """
    Simulate disaggregated scheduler: prefill workers run independently.
    Returns TTFT per request (ms).
    """
    import heapq

    # Worker free-at times (min-heap)
    worker_free = [0.0] * n_prefill_workers
    heapq.heapify(worker_free)

    queue = sorted(requests, key=lambda r: r.arrival_ms)
    ttfts: List[float] = []
    decode_step_ms = 1000.0 / model.decode_tps_h100

    for req in queue:
        # Assign to first free worker
        earliest_free = heapq.heappop(worker_free)
        start = max(earliest_free, req.arrival_ms)
        prefill_ms = req.prompt_tokens / model.prefill_tps_h100 * 1000
        pf_done = start + prefill_ms
        # KV transfer
        kv_b = kv_cache_bytes(model, req.prompt_tokens)
        xfer_ms = transfer_latency_ms(kv_b, network)
        ttft = pf_done + xfer_ms + decode_step_ms - req.arrival_ms
        ttfts.append(ttft)
        heapq.heappush(worker_free, pf_done)

    return ttfts


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = p / 100.0 * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def print_starvation_simulation():
    model = MODELS["llama-3-8b"]
    net   = NETWORKS["IB-HDR"]

    # Mixed workload: 70% short (200 tok), 30% long RAG (32K tok)
    rng = random.Random(99)
    n_reqs = 200
    arrival_rate_per_s = 20.0    # 20 requests per second
    requests = []
    t = 0.0
    for _ in range(n_reqs):
        t += rng.expovariate(arrival_rate_per_s) * 1000  # convert to ms
        if rng.random() < 0.30:
            # Long RAG
            prompt = rng.randint(28000, 34000)
            output = rng.randint(500, 2000)
        else:
            prompt = rng.randint(150, 300)
            output = rng.randint(100, 200)
        requests.append(RequestEvent(t, prompt, output))

    ttft_coloc = simulate_colocated(requests, model)
    ttft_disagg = simulate_disaggregated(requests, model,
                                         n_prefill_workers=2, network=net)

    print("\n" + "=" * 64)
    print("  Prefill Starvation Simulation (mixed workload)")
    print(f"  {n_reqs} requests: 70% short (200 tok), 30% long RAG (32K tok)")
    print("=" * 64)

    stats = [
        ("Co-located (1 GPU)",    ttft_coloc),
        ("Disaggregated (2 pf)",  ttft_disagg),
    ]
    for label, ttfts in stats:
        p50  = percentile(ttfts, 50)
        p95  = percentile(ttfts, 95)
        p99  = percentile(ttfts, 99)
        mean = sum(ttfts) / len(ttfts)
        print(f"\n  {label}:")
        print(f"    Mean  : {mean:8.0f} ms")
        print(f"    p50   : {p50:8.0f} ms")
        print(f"    p95   : {p95:8.0f} ms")
        print(f"    p99   : {p99:8.0f} ms")

    p99_c = percentile(ttft_coloc,  99)
    p99_d = percentile(ttft_disagg, 99)
    if p99_c > 0:
        print(f"\n  p99 improvement: {p99_c:.0f} ms → {p99_d:.0f} ms"
              f"  ({p99_c / p99_d:.1f}× better)")
    print("=" * 64)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("█" * 62)
    print("  Chapter 18 — Disaggregated Prefill and Decode")
    print("█" * 62)

    print_kv_size_table()
    print_transfer_cost_table()
    print_cluster_sizing()
    print_latency_budget()
    simulate_global_kv_store()

    cfg = DisaggConfig(
        model_id="meta-llama/Llama-3-8B-Instruct",
        max_model_len=32768,
        prefill_max_seqs=8,
        decode_max_seqs=128,
        kv_buffer_size_gb=2.0,
    )
    generate_config(cfg)

    # Config with a warning
    cfg_bad = DisaggConfig(
        model_id="meta-llama/Llama-3-8B-Instruct",
        max_model_len=32768,
        prefill_max_seqs=32,   # too high
        decode_max_seqs=16,    # too low
        tensor_parallel_size=4,
    )
    print("\n>>> Config with issues:")
    generate_config(cfg_bad)

    print_starvation_simulation()


if __name__ == "__main__":
    main()

```

## C++ — `disaggregated_demo.cpp`

```cpp
// disaggregated_demo.cpp — Chapter 18: Disaggregated Prefill and Decode
//
// Compile:  g++ -std=c++17 -O2 -Wall -o disaggregated_demo disaggregated_demo.cpp
// Run:      ./disaggregated_demo
//
// Mirrors disaggregated_demo.py: KV size table, transfer cost model,
// cluster ratio calculator, latency budget breakdown, KV serialiser sketch.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

static std::string repeat_str(const std::string& s, int n) {
    std::string out;
    for (int i = 0; i < n; ++i) out += s;
    return out;
}

static std::string fmt(double v, int prec = 1) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

static std::string fmt_int(long long v) {
    std::string s = std::to_string(std::llabs(v));
    int n = static_cast<int>(s.size());
    std::string out;
    for (int i = 0; i < n; ++i) {
        if (i && (n - i) % 3 == 0) out += ',';
        out += s[static_cast<size_t>(i)];
    }
    if (v < 0) out = "-" + out;
    return out;
}

static std::string fmt_bytes(long long b) {
    if (b >= (1LL << 30))
        return fmt(static_cast<double>(b) / (1LL << 30), 1) + " GB";
    return fmt(static_cast<double>(b) / (1LL << 20), 0) + " MB";
}

static std::string box_line(int w, char c = '=') {
    return repeat_str(std::string(1, c), w);
}

static void print_header(const std::string& title, int w = 62) {
    std::cout << box_line(w) << "\n";
    int pad = std::max(0, (w - static_cast<int>(title.size()) - 2) / 2);
    std::cout << repeat_str(" ", pad) << " " << title << "\n";
    std::cout << box_line(w) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Data model
// ─────────────────────────────────────────────────────────────────────────────

struct ModelSpec {
    std::string name;
    int    n_layers;
    int    n_kv_heads;
    int    d_head;
    double n_params_b;
    double prefill_tps;     // tokens/s (H100 SXM large batch)
    double decode_tps;      // tokens/s (H100 SXM batch=1)
    int    bytes_per_elem;  // 2 = BF16, 1 = FP8
};

static const std::vector<ModelSpec> MODELS = {
    {"Llama-3-8B",          32, 8, 128, 8.0,  12500.0, 209.0, 2},
    {"Llama-3-70B",         80, 8, 128, 70.0,  3200.0,  52.0, 2},
    {"Llama-3-8B (FP8 KV)", 32, 8, 128, 8.0,  14000.0, 230.0, 1},
};

struct NetworkSpec {
    std::string name;
    double bw_gbps;    // GB/s
    double rtt_ms;
};

static const std::vector<NetworkSpec> NETWORKS = {
    {"1 GbE",            0.125,   1.0},
    {"10 GbE",           1.25,    0.5},
    {"100 GbE",          12.5,    0.1},
    {"InfiniBand HDR",   25.0,    0.05},
    {"InfiniBand NDR",   50.0,    0.04},
    {"NVLink C2C",      900.0,    0.001},
};

// ─────────────────────────────────────────────────────────────────────────────
// KV cache size
// ─────────────────────────────────────────────────────────────────────────────

static long long kv_cache_bytes(const ModelSpec& m, int n_tokens) {
    return static_cast<long long>(2)   // K and V
           * m.n_layers
           * m.n_kv_heads
           * m.d_head
           * n_tokens
           * m.bytes_per_elem;
}

static void print_kv_size_table() {
    std::cout << "\n" << box_line(70) << "\n";
    std::cout << "  KV Cache Size by Context Length\n";
    std::cout << box_line(70) << "\n";
    std::cout << "  " << std::left  << std::setw(24) << "Model"
              << std::right
              << std::setw(12) << "1K tokens"
              << std::setw(12) << "4K tokens"
              << std::setw(13) << "32K tokens"
              << std::setw(14) << "128K tokens"
              << "\n";
    std::cout << "  " << repeat_str("-", 23) << "  "
              << repeat_str("-", 10) << "  "
              << repeat_str("-", 10) << "  "
              << repeat_str("-", 11) << "  "
              << repeat_str("-", 12) << "\n";

    for (const auto& m : MODELS) {
        std::cout << "  " << std::left << std::setw(24) << m.name
                  << std::right
                  << std::setw(12) << fmt_bytes(kv_cache_bytes(m,    1024))
                  << std::setw(12) << fmt_bytes(kv_cache_bytes(m,    4096))
                  << std::setw(13) << fmt_bytes(kv_cache_bytes(m,   32768))
                  << std::setw(14) << fmt_bytes(kv_cache_bytes(m,  131072))
                  << "\n";
    }
    std::cout << box_line(70) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// KV transfer cost model
// ─────────────────────────────────────────────────────────────────────────────

static double transfer_latency_ms(long long kv_bytes, const NetworkSpec& net) {
    double xfer_ms = (static_cast<double>(kv_bytes) / (net.bw_gbps * 1e9)) * 1000.0;
    return xfer_ms + net.rtt_ms;
}

static void print_transfer_cost_table() {
    const ModelSpec& m = MODELS[0];  // Llama-3-8B BF16

    double decode_step_ms = 1000.0 / m.decode_tps;

    std::cout << "\n" << box_line(74) << "\n";
    std::cout << "  KV Transfer Latency (ms)  |  " << m.name << "\n";
    std::cout << box_line(74) << "\n";
    std::cout << "  " << std::left  << std::setw(20) << "Network"
              << std::right
              << std::setw(12) << "1K ctx"
              << std::setw(12) << "4K ctx"
              << std::setw(14) << "32K ctx"
              << std::setw(10) << "Usable?"
              << "\n";
    std::cout << "  " << repeat_str("-", 19) << "  "
              << repeat_str("-", 10) << "  "
              << repeat_str("-", 10) << "  "
              << repeat_str("-", 12) << "  "
              << repeat_str("-", 7) << "\n";

    for (const auto& net : NETWORKS) {
        double lat1k  = transfer_latency_ms(kv_cache_bytes(m,  1024), net);
        double lat4k  = transfer_latency_ms(kv_cache_bytes(m,  4096), net);
        double lat32k = transfer_latency_ms(kv_cache_bytes(m, 32768), net);

        std::string usable = lat4k < decode_step_ms * 10 ? "YES" : "NO ";

        std::cout << "  " << std::left  << std::setw(20) << net.name
                  << std::right
                  << std::setw(11) << (fmt(lat1k,  1) + "ms")
                  << std::setw(12) << (fmt(lat4k,  1) + "ms")
                  << std::setw(13) << (fmt(lat32k, 1) + "ms")
                  << std::setw(10) << usable
                  << "\n";
    }
    std::cout << box_line(74) << "\n";
    std::cout << "  decode step = " << fmt(decode_step_ms, 1) << " ms  "
              << "(Usable = 4K transfer < "
              << fmt(decode_step_ms * 10, 0) << " ms)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Cluster sizing
// ─────────────────────────────────────────────────────────────────────────────

struct WorkloadProfile {
    std::string name;
    int avg_prompt_tokens;
    int avg_output_tokens;
};

static const std::vector<WorkloadProfile> WORKLOADS = {
    {"Short chat",    200,   150},
    {"Code complete", 800,   300},
    {"RAG medium",   2000,   500},
    {"RAG long",    32000,  2000},
    {"Summarize",   16000,   400},
};

static void print_cluster_sizing() {
    const ModelSpec& m = MODELS[0];

    std::cout << "\n" << box_line(74) << "\n";
    std::cout << "  Cluster Sizing: Decode/Prefill GPU Ratio"
              << "  |  " << m.name << "  |  H100\n";
    std::cout << box_line(74) << "\n";

    std::cout << "  " << std::left  << std::setw(16) << "Workload"
              << std::right
              << std::setw(8)  << "Prompt"
              << std::setw(8)  << "Output"
              << std::setw(10) << "Pf(ms)"
              << std::setw(10) << "Dec(ms)"
              << std::setw(9)  << "Ratio"
              << "  Rec\n";
    std::cout << "  " << repeat_str("-", 15) << "  "
              << repeat_str("-", 7) << "  "
              << repeat_str("-", 7) << "  "
              << repeat_str("-", 8) << "  "
              << repeat_str("-", 8) << "  "
              << repeat_str("-", 7) << "  "
              << repeat_str("-", 14) << "\n";

    for (const auto& w : WORKLOADS) {
        double pf_ms  = w.avg_prompt_tokens / m.prefill_tps * 1000.0;
        double dec_ms = w.avg_output_tokens / m.decode_tps  * 1000.0;
        double ratio  = dec_ms / pf_ms;
        std::string rec = "1:" + fmt(ratio, 0);

        std::cout << "  " << std::left  << std::setw(16) << w.name
                  << std::right
                  << std::setw(8)  << fmt_int(w.avg_prompt_tokens)
                  << std::setw(8)  << fmt_int(w.avg_output_tokens)
                  << std::setw(9)  << (fmt(pf_ms,  0) + "ms")
                  << std::setw(10) << (fmt(dec_ms, 0) + "ms")
                  << std::setw(9)  << fmt(ratio, 1)
                  << "  " << rec
                  << "\n";
    }
    std::cout << box_line(74) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Latency budget
// ─────────────────────────────────────────────────────────────────────────────

static void print_latency_budget() {
    const ModelSpec& m = MODELS[0];
    const NetworkSpec& net = NETWORKS[3];  // InfiniBand HDR

    std::cout << "\n" << box_line(66) << "\n";
    std::cout << "  TTFT Latency Budget: Co-located vs. Disaggregated\n";
    std::cout << "  " << m.name << "  |  H100 SXM  |  " << net.name << "\n";
    std::cout << box_line(66) << "\n";

    std::cout << "  " << std::right
              << std::setw(8)  << "Context"
              << std::setw(13) << "Co-located"
              << std::setw(12) << "Disagg(pf)"
              << std::setw(10) << "KV xfer"
              << std::setw(13) << "Disagg TTFT"
              << std::setw(9)  << "Speedup"
              << "\n";
    std::cout << "  " << repeat_str("-", 7)  << "  "
              << repeat_str("-", 11) << "  "
              << repeat_str("-", 10) << "  "
              << repeat_str("-", 8)  << "  "
              << repeat_str("-", 11) << "  "
              << repeat_str("-", 7)  << "\n";

    for (int t : {512, 2048, 8192, 32768}) {
        // Co-located: batch=1 prefill (1/8 of peak throughput)
        double coloc_tps  = m.prefill_tps / 8.0;
        double ttft_coloc = t / coloc_tps * 1000.0;

        // Disaggregated: full batch prefill efficiency
        double ttft_pf   = t / m.prefill_tps * 1000.0;
        double xfer_ms   = transfer_latency_ms(kv_cache_bytes(m, t), net);
        double dec1_ms   = 1000.0 / m.decode_tps;
        double ttft_dis  = ttft_pf + xfer_ms + dec1_ms;
        double speedup   = ttft_coloc / ttft_dis;

        std::cout << "  " << std::right
                  << std::setw(8)  << fmt_int(t)
                  << std::setw(12) << (fmt(ttft_coloc, 0) + "ms")
                  << std::setw(12) << (fmt(ttft_pf,   0) + "ms")
                  << std::setw(10) << (fmt(xfer_ms,   1) + "ms")
                  << std::setw(12) << (fmt(ttft_dis,  0) + "ms")
                  << std::setw(8)  << (fmt(speedup,   1) + "x")
                  << "\n";
    }
    std::cout << box_line(66) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// KV serialiser sketch
// ─────────────────────────────────────────────────────────────────────────────

// This is a conceptual sketch of what the KV transfer looks like on the
// prefill side. In production vLLM uses GPUDirect RDMA (zero-copy); this
// CPU-side version illustrates the data layout for educational purposes.

struct KVBlock {
    int      layer_idx;
    int      block_idx;          // position in the decode worker's block table
    int      n_tokens;           // tokens stored in this block (up to block_size)
    std::vector<float> k_data;   // [n_kv_heads * n_tokens * d_head]
    std::vector<float> v_data;   // [n_kv_heads * n_tokens * d_head]
};

struct KVTransferPayload {
    std::string  request_id;
    int          n_layers;
    int          n_kv_heads;
    int          d_head;
    int          total_tokens;
    std::vector<KVBlock> blocks;

    size_t byte_size() const {
        size_t s = 0;
        for (const auto& b : blocks)
            s += (b.k_data.size() + b.v_data.size()) * sizeof(float);
        return s;
    }
};

// Build a minimal KV payload for one prefill result
static KVTransferPayload build_kv_payload(
        const std::string& req_id,
        const ModelSpec& m,
        int n_tokens,
        int block_size = 16)
{
    KVTransferPayload p;
    p.request_id  = req_id;
    p.n_layers    = m.n_layers;
    p.n_kv_heads  = m.n_kv_heads;
    p.d_head      = m.d_head;
    p.total_tokens = n_tokens;

    int block_idx = 0;
    for (int layer = 0; layer < m.n_layers; ++layer) {
        int remaining = n_tokens;
        int tok_start = 0;
        while (remaining > 0) {
            int chunk = std::min(remaining, block_size);
            KVBlock blk;
            blk.layer_idx = layer;
            blk.block_idx = block_idx++;
            blk.n_tokens  = chunk;
            // Allocate (filled with zeros here; real code copies from GPU HBM)
            size_t sz = static_cast<size_t>(m.n_kv_heads * chunk * m.d_head);
            blk.k_data.assign(sz, 0.0f);
            blk.v_data.assign(sz, 0.0f);
            p.blocks.push_back(std::move(blk));
            remaining -= chunk;
            tok_start += chunk;
        }
    }
    return p;
}

static void print_kv_serialiser_sketch() {
    // Use a tiny model spec for the demo (2 layers, 2 heads) to keep output brief
    ModelSpec tiny{"Demo-1B", 2, 2, 64, 1.0, 5000.0, 100.0, 2};
    int n_tokens  = 48;   // 3 blocks of 16 tokens
    int block_size = 16;

    KVTransferPayload payload = build_kv_payload("req-0001", tiny, n_tokens, block_size);

    std::cout << "\n" << box_line(62) << "\n";
    std::cout << "  KV Serialiser Sketch  (conceptual — CPU demo)\n";
    std::cout << box_line(62) << "\n";
    std::cout << "  Model: Demo-1B  |  n_tokens=" << n_tokens
              << "  |  block_size=" << block_size << "\n\n";
    std::cout << "  Payload: request_id=" << payload.request_id << "\n";
    std::cout << "    n_layers  = " << payload.n_layers    << "\n";
    std::cout << "    n_kv_heads= " << payload.n_kv_heads  << "\n";
    std::cout << "    d_head    = " << payload.d_head      << "\n";
    std::cout << "    n_blocks  = " << payload.blocks.size() << "\n";
    std::cout << "    total_bytes = "
              << fmt_bytes(static_cast<long long>(payload.byte_size())) << "\n\n";

    // Print first few blocks
    int shown = 0;
    for (const auto& blk : payload.blocks) {
        if (shown >= 4) { std::cout << "    ... (" << (payload.blocks.size() - 4) << " more blocks)\n"; break; }
        std::cout << "    Block layer=" << blk.layer_idx
                  << " idx=" << blk.block_idx
                  << " tokens=" << blk.n_tokens
                  << " K=" << blk.k_data.size() << " floats"
                  << " V=" << blk.v_data.size() << " floats\n";
        ++shown;
    }

    std::cout << "\n  Real vLLM path:\n";
    std::cout << "    1. Prefill GPU registers HBM tensor with RDMA NIC (GPUDirect)\n";
    std::cout << "    2. NIC DMA-reads KV blocks directly from HBM (no CPU copy)\n";
    std::cout << "    3. Remote NIC DMA-writes into decode GPU HBM\n";
    std::cout << "    4. Decode worker populates block table and starts generation\n";
    std::cout << box_line(62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Global KV store simulation
// ─────────────────────────────────────────────────────────────────────────────

static void simulate_global_kv_store() {
    const ModelSpec& m = MODELS[0];
    int    n_requests   = 1000;
    int    n_unique_docs = 50;
    int    avg_doc_tokens = 8192;
    double store_cap_gb  = 200.0;

    long long kv_per_doc = kv_cache_bytes(m, avg_doc_tokens);
    long long cap_bytes  = static_cast<long long>(store_cap_gb * 1e9);
    int       max_docs   = static_cast<int>(cap_bytes / kv_per_doc);

    // Zipf weights
    std::vector<double> weights(static_cast<size_t>(n_unique_docs));
    double total_w = 0;
    for (int i = 0; i < n_unique_docs; ++i) {
        weights[static_cast<size_t>(i)] = 1.0 / (i + 1);
        total_w += weights[static_cast<size_t>(i)];
    }
    for (auto& w : weights) w /= total_w;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::unordered_map<int, int> kv_store;   // doc_id -> age (for eviction)
    int hits = 0, age = 0;
    long long tokens_saved = 0, tokens_total = 0;

    for (int r = 0; r < n_requests; ++r) {
        // Sample doc from Zipf
        double s = uni(rng);
        double cum = 0;
        int doc_id = n_unique_docs - 1;
        for (int i = 0; i < n_unique_docs; ++i) {
            cum += weights[static_cast<size_t>(i)];
            if (s <= cum) { doc_id = i; break; }
        }
        tokens_total += avg_doc_tokens;

        if (kv_store.count(doc_id)) {
            ++hits;
            tokens_saved += avg_doc_tokens;
            kv_store[doc_id] = age++;
        } else {
            // Evict oldest if full
            if (static_cast<int>(kv_store.size()) >= max_docs) {
                auto oldest = std::min_element(kv_store.begin(), kv_store.end(),
                    [](const auto& a, const auto& b){ return a.second < b.second; });
                kv_store.erase(oldest);
            }
            kv_store[doc_id] = age++;
        }
    }

    double hit_rate   = static_cast<double>(hits) / n_requests * 100.0;
    double saved_pct  = static_cast<double>(tokens_saved) / tokens_total * 100.0;
    double gpu_sec    = static_cast<double>(tokens_saved) / m.prefill_tps;

    std::cout << "\n" << box_line(62) << "\n";
    std::cout << "  Global KV Store Simulation\n";
    std::cout << box_line(62) << "\n";
    std::cout << "  Requests         : " << fmt_int(n_requests)    << "\n";
    std::cout << "  Unique docs      : " << n_unique_docs           << "\n";
    std::cout << "  Avg doc length   : " << fmt_int(avg_doc_tokens) << " tokens\n";
    std::cout << "  KV per doc       : " << fmt_bytes(kv_per_doc)   << "\n";
    std::cout << "  Store capacity   : " << fmt(store_cap_gb, 0)
              << " GB  (" << max_docs << " docs max)\n\n";
    std::cout << "  Cache hits       : " << fmt_int(hits) << " / " << fmt_int(n_requests)
              << "  (" << fmt(hit_rate, 1) << "%)\n";
    std::cout << "  Prefill saved    : " << fmt_int(tokens_saved) << " / "
              << fmt_int(tokens_total) << "  (" << fmt(saved_pct, 1) << "%)\n";
    std::cout << "  GPU-seconds saved: " << fmt(gpu_sec, 1) << " s  ("
              << fmt(gpu_sec / 3600.0, 2) << " GPU-hours)\n";
    std::cout << box_line(62) << "\n";

    // Hit-rate bar
    int bar_w = 40;
    int bar_len = static_cast<int>(std::round(hit_rate / 100.0 * bar_w));
    bar_len = std::max(1, bar_len);
    std::cout << "\n  Hit rate: " << std::setw(5) << fmt(hit_rate, 1) << "%  |"
              << std::string(static_cast<size_t>(bar_len), '#')
              << std::string(static_cast<size_t>(bar_w - bar_len), ' ')
              << "|\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Config generator
// ─────────────────────────────────────────────────────────────────────────────

struct DisaggConfig {
    std::string model_id;
    int         max_model_len;
    int         prefill_max_seqs  = 8;
    int         decode_max_seqs   = 128;
    double      prefill_gpu_util  = 0.85;
    double      decode_gpu_util   = 0.90;
    std::string kv_connector      = "MooncakeConnector";
    double      kv_buffer_gb      = 2.0;
    int         tp_size           = 1;
    std::string kv_dtype          = "bf16";
};

static void print_yaml_worker(const std::string& role, int rank,
                               int max_seqs, double gpu_util,
                               double buf_gb, const DisaggConfig& cfg)
{
    long long buf_bytes = static_cast<long long>(buf_gb * 1e9);
    std::string kv_role = (role == "prefill") ? "kv_producer" : "kv_consumer";
    int quant = (cfg.kv_dtype == "bf16") ? 0 : 8;

    std::cout << "# " << role << " worker  (rank " << rank << ")\n";
    std::cout << "model: " << cfg.model_id << "\n";
    std::cout << "max_model_len: " << cfg.max_model_len << "\n";
    std::cout << "max_num_seqs: " << max_seqs << "\n";
    std::cout << "gpu_memory_utilization: " << fmt(gpu_util, 2) << "\n";
    std::cout << "tensor_parallel_size: " << cfg.tp_size << "\n";
    std::cout << "kv_transfer_config:\n";
    std::cout << "  kv_connector: " << cfg.kv_connector << "\n";
    std::cout << "  kv_role: " << kv_role << "\n";
    std::cout << "  kv_rank: " << rank << "\n";
    std::cout << "  kv_parallel_size: 1\n";
    std::cout << "  kv_buffer_device: cuda\n";
    std::cout << "  kv_buffer_size: " << buf_bytes << "\n";
    std::cout << "  kv_quant_policy: " << quant << "\n";
}

static void generate_config(const DisaggConfig& cfg) {
    std::cout << "\n" << box_line(60) << "\n";
    std::cout << "  vLLM Disaggregated Deployment Config\n";
    std::cout << box_line(60) << "\n\n";

    std::cout << "--- prefill_worker.yaml ---\n\n";
    print_yaml_worker("prefill", 0, cfg.prefill_max_seqs, cfg.prefill_gpu_util,
                       cfg.kv_buffer_gb, cfg);
    std::cout << "\n--- decode_worker.yaml ---\n\n";
    print_yaml_worker("decode", 1, cfg.decode_max_seqs, cfg.decode_gpu_util,
                       cfg.kv_buffer_gb * 2, cfg);

    std::cout << "\n" << box_line(60) << "\n";
    std::cout << "  Config validation:\n";
    bool ok = true;
    if (cfg.prefill_max_seqs > 16) {
        std::cout << "  WARN: prefill max_seqs=" << cfg.prefill_max_seqs
                  << " > 16 -- raises TTFT\n";
        ok = false;
    }
    if (cfg.decode_max_seqs < 32) {
        std::cout << "  WARN: decode max_seqs=" << cfg.decode_max_seqs
                  << " < 32 -- under-utilizes BW\n";
        ok = false;
    }
    if (cfg.tp_size > 1) {
        std::cout << "  NOTE: TP=" << cfg.tp_size
                  << " -- both workers MUST use same TP degree\n";
        ok = false;
    }
    if (ok) std::cout << "  OK  Config looks reasonable.\n";
    std::cout << box_line(60) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    print_header("Chapter 18 - Disaggregated Prefill and Decode");

    print_kv_size_table();
    print_transfer_cost_table();
    print_cluster_sizing();
    print_latency_budget();
    simulate_global_kv_store();
    print_kv_serialiser_sketch();

    DisaggConfig good;
    good.model_id       = "meta-llama/Llama-3-8B-Instruct";
    good.max_model_len  = 32768;
    good.prefill_max_seqs = 8;
    good.decode_max_seqs  = 128;
    good.kv_buffer_gb   = 2.0;
    generate_config(good);

    std::cout << "\n>>> Config with issues:\n";
    DisaggConfig bad;
    bad.model_id       = "meta-llama/Llama-3-8B-Instruct";
    bad.max_model_len  = 32768;
    bad.prefill_max_seqs = 32;
    bad.decode_max_seqs  = 16;
    bad.tp_size        = 4;
    generate_config(bad);

    return 0;
}

```

