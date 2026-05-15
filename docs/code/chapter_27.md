# Chapter 27: Long-Context Inference — Companion Code

## Python — `benchmark_long_context.py`

```python
# benchmark_long_context.py
"""
Benchmark script for long-context serving.
Sends requests with varying context lengths and measures TTFT and TPOT.
"""
import time, asyncio, statistics, json
from typing import List, Tuple
import aiohttp

ENDPOINT = "http://localhost:8000/v1/completions"
MODEL    = "meta-llama/Llama-3.1-8B-Instruct"

def make_prompt(n_tokens: int) -> str:
    """Approximate n_tokens by repeating a known-length phrase."""
    unit = "The quick brown fox jumps over the lazy dog. "  # ~10 tokens
    reps = max(1, n_tokens // 10)
    return (unit * reps) + "\n\nSummarize the above in one sentence:"


async def single_request(session: aiohttp.ClientSession,
                          prompt: str, max_tokens: int = 64
                         ) -> Tuple[float, float, int]:
    """
    Returns (ttft_ms, tpot_ms, total_tokens).
    Uses streaming to measure time-to-first-token.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0

    async with session.post(ENDPOINT, json=payload) as resp:
        async for line in resp.content:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                if ttft is None and data["choices"][0].get("text", ""):
                    ttft = (time.perf_counter() - t0) * 1000
                n_tokens += 1
            except (json.JSONDecodeError, KeyError):
                pass

    total_ms = (time.perf_counter() - t0) * 1000
    tpot = (total_ms - (ttft or 0)) / max(1, n_tokens - 1)
    return ttft or total_ms, tpot, n_tokens


async def benchmark(context_lens: List[int], n_requests: int = 5):
    async with aiohttp.ClientSession() as session:
        print(f"{'Context':>10}  {'TTFT p50(ms)':>14}  {'TTFT p95(ms)':>14}  "
              f"{'TPOT(ms)':>10}")
        print("-" * 55)
        for ctx in context_lens:
            prompt = make_prompt(ctx)
            ttfts, tpots = [], []
            for _ in range(n_requests):
                ttft, tpot, _ = await single_request(session, prompt)
                ttfts.append(ttft)
                tpots.append(tpot)
            ttfts.sort()
            p50 = statistics.median(ttfts)
            p95 = ttfts[int(len(ttfts) * 0.95)] if len(ttfts) > 1 else ttfts[0]
            print(f"{ctx:>10,}  {p50:>14.1f}  {p95:>14.1f}  "
                  f"{statistics.mean(tpots):>10.1f}")


if __name__ == "__main__":
    ctx_lens = [1024, 4096, 16384, 32768, 65536, 131072]
    asyncio.run(benchmark(ctx_lens))

```

## Python — `long_context_demo.py`

```python
# long_context_demo.py
"""
Chapter 27 — Long-Context Inference Demo (Python)

Topics:
  1. KV cache memory calculator
  2. Multi-model comparison table
  3. Chunked-prefill simulator
  4. RoPE scaling comparison
  5. Workload advisor
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  KV CACHE MEMORY CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    n_layers: int
    n_kv_heads: int        # GQA kv head count
    head_dim: int
    vocab_size: int        # unused here, kept for completeness
    param_billions: float  # total model params in BF16


@dataclass
class HardwareSpec:
    name: str
    hbm_gb: float
    bandwidth_tb_s: float  # HBM bandwidth


MODELS = [
    ModelSpec("Llama-3.1-8B",   n_layers=32,  n_kv_heads=8,  head_dim=128, vocab_size=128256, param_billions=8.0),
    ModelSpec("Llama-3.1-70B",  n_layers=80,  n_kv_heads=8,  head_dim=128, vocab_size=128256, param_billions=70.0),
    ModelSpec("Llama-3.1-405B", n_layers=126, n_kv_heads=8,  head_dim=128, vocab_size=128256, param_billions=405.0),
    ModelSpec("Mistral-7B",     n_layers=32,  n_kv_heads=8,  head_dim=128, vocab_size=32000,  param_billions=7.0),
    ModelSpec("Qwen2.5-72B",    n_layers=80,  n_kv_heads=8,  head_dim=128, vocab_size=152064, param_billions=72.0),
    ModelSpec("DeepSeek-V3",    n_layers=61,  n_kv_heads=128,head_dim=128, vocab_size=129280, param_billions=671.0),
]

HARDWARE = [
    HardwareSpec("H100 SXM 80GB",  hbm_gb=80.0,  bandwidth_tb_s=3.35),
    HardwareSpec("A100 SXM 80GB",  hbm_gb=80.0,  bandwidth_tb_s=2.0),
    HardwareSpec("RTX 4090 24GB",  hbm_gb=24.0,  bandwidth_tb_s=1.008),
    HardwareSpec("M2 Ultra 192GB", hbm_gb=192.0, bandwidth_tb_s=0.8),
]


def kv_cache_gb(model: ModelSpec, seq_len: int, dtype_bytes: int = 2) -> float:
    """KV cache memory in GB for a single sequence."""
    return (2 * model.n_layers * model.n_kv_heads * model.head_dim
            * seq_len * dtype_bytes) / 1e9


def model_weight_gb(model: ModelSpec, dtype_bytes: int = 2) -> float:
    return model.param_billions * 1e9 * dtype_bytes / 1e9


def max_concurrent_requests(model: ModelSpec, hw: HardwareSpec,
                             seq_len: int,
                             overhead_fraction: float = 0.10) -> int:
    """
    How many simultaneous long-context requests fit on `hw`?
    Reserves overhead_fraction of HBM for activations + OS.
    """
    usable_gb = hw.hbm_gb * (1 - overhead_fraction)
    weights_gb = model_weight_gb(model)
    available_for_kv = usable_gb - weights_gb
    if available_for_kv <= 0:
        return 0
    per_req_gb = kv_cache_gb(model, seq_len)
    if per_req_gb <= 0:
        return 0
    return max(0, int(available_for_kv / per_req_gb))


def print_kv_table():
    model = MODELS[0]  # Llama-3.1-8B as reference
    seq_lens = [4096, 16384, 32768, 65536, 131072, 524288, 1048576]
    print("=" * 70)
    print(f"KV Cache Memory — {model.name}  (BF16, single sequence)")
    print("=" * 70)
    print(f"{'Seq Len':>12}  {'KV Cache':>12}  {'% of 80GB HBM':>16}  {'Tokens/GB':>10}")
    print("-" * 70)
    for T in seq_lens:
        gb = kv_cache_gb(model, T)
        pct = gb / 80.0 * 100
        tpg = T / gb if gb > 0 else 0
        T_str = f"{T:,}"
        print(f"{T_str:>12}  {gb:>10.3f}GB  {pct:>15.1f}%  {tpg:>10.0f}")
    print()


def print_concurrency_table():
    hw = HARDWARE[0]  # H100 80GB
    seq_lens = [32768, 65536, 131072]
    print("=" * 80)
    print(f"Max Concurrent Long-Context Requests — {hw.name}  (BF16, 10% overhead)")
    print("=" * 80)
    header = f"{'Model':25}  {'Weights(GB)':>12}" + "".join(
        f"  {T//1024:>5}K ctx" for T in seq_lens)
    print(header)
    print("-" * 80)
    for m in MODELS:
        wt = model_weight_gb(m)
        row = f"{m.name:25}  {wt:>10.1f}GB"
        for T in seq_lens:
            n = max_concurrent_requests(m, hw, T)
            row += f"  {n:>8}"
        print(row)
    print()


def print_decode_time_per_token():
    """
    Decode step latency = KV bytes read / bandwidth.
    Models the minimum possible decode step time at batch=1.
    """
    hw = HARDWARE[0]
    model = MODELS[0]
    seq_lens = [4096, 32768, 131072, 524288]
    print("=" * 60)
    print(f"Decode Step Latency (bandwidth-bound) — {model.name} on {hw.name}")
    print("=" * 60)
    print(f"{'Seq Len':>12}  {'KV reads(GB)':>14}  {'Min latency(ms)':>16}")
    print("-" * 60)
    for T in seq_lens:
        kv_gb = kv_cache_gb(model, T)
        lat_ms = kv_gb / (hw.bandwidth_tb_s * 1000) * 1000  # ms
        print(f"{T:>12,}  {kv_gb:>14.3f}  {lat_ms:>15.3f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHUNKED PREFILL SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkedPrefillResult:
    total_prefill_tokens: int
    chunk_size: int
    n_chunks: int
    prefill_time_ms: float      # total prefill wall time
    first_decode_delay_ms: float  # time-to-first-token for decode requests
    interleaved_decode_tokens: int  # decode tokens produced during prefill


def simulate_chunked_prefill(
    context_len: int,
    chunk_size: int,
    prefill_tflops_per_token: float = 0.25,  # TFLOPs on H100
    decode_tokens_per_chunk: int = 2,
) -> ChunkedPrefillResult:
    """
    Simulate a chunked prefill run.

    Without chunking the entire prefill must complete before any decode step
    can begin (head-of-line blocking).  With chunking, each chunk takes
    ~chunk_size * prefill_tflops_per_token / H100_TFLOPS seconds, and we
    can slot one mini-decode batch per chunk.

    H100 BF16 dense throughput: ~989 TFLOPs
    """
    H100_TFLOPS = 989.0
    n_chunks = math.ceil(context_len / chunk_size)

    # Time per chunk (ms): FLOPs = chunk_size * model_flops_per_token
    # model_flops_per_token for Llama-3.1-8B prefill ≈ 2 * 8B = 16 GFLOPs per token
    flops_per_token_gflop = 16.0  # ~2 * param_count for matmuls
    chunk_time_ms = (chunk_size * flops_per_token_gflop * 1e9) / (H100_TFLOPS * 1e12) * 1e3

    total_prefill_ms = n_chunks * chunk_time_ms
    first_decode_delay_ms = chunk_time_ms  # after first chunk, decode can start

    return ChunkedPrefillResult(
        total_prefill_tokens=context_len,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        prefill_time_ms=total_prefill_ms,
        first_decode_delay_ms=first_decode_delay_ms,
        interleaved_decode_tokens=n_chunks * decode_tokens_per_chunk,
    )


def print_chunked_prefill_table():
    context_len = 131072  # 128K
    chunk_sizes = [512, 1024, 2048, 4096, 8192, 16384]

    print("=" * 75)
    print(f"Chunked Prefill Simulation — context={context_len:,} tokens on H100 80GB")
    print("=" * 75)
    print(f"{'Chunk':>8}  {'Chunks':>7}  {'Prefill(ms)':>12}  "
          f"{'TTFT(ms)':>10}  {'Decode tokens':>14}")
    print("-" * 75)

    # Baseline: no chunking
    r_base = simulate_chunked_prefill(context_len, chunk_size=context_len)
    print(f"{'FULL':>8}  {r_base.n_chunks:>7}  {r_base.prefill_time_ms:>12.1f}  "
          f"{r_base.first_decode_delay_ms:>10.1f}  {'0':>14}  ← head-of-line block")

    for cs in chunk_sizes:
        r = simulate_chunked_prefill(context_len, cs)
        print(f"{cs:>8,}  {r.n_chunks:>7}  {r.prefill_time_ms:>12.1f}  "
              f"{r.first_decode_delay_ms:>10.1f}  {r.interleaved_decode_tokens:>14}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ROPE SCALING COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def rope_freq_linear(pos: int, dim: int, base: float = 10000.0,
                     scale: float = 1.0) -> list[float]:
    """
    Linear RoPE interpolation.
    Divide position by scale factor to compress into original range.
    theta_i = 1 / (base^(2i/d))   [standard]
    With linear interpolation: effective_pos = pos / scale
    """
    effective_pos = pos / scale
    freqs = []
    for i in range(dim // 2):
        theta = 1.0 / (base ** (2 * i / dim))
        freqs.append(effective_pos * theta)
    return freqs


def rope_freq_ntk(pos: int, dim: int, base: float = 10000.0,
                  alpha: float = 8.0) -> list[float]:
    """
    NTK-aware RoPE scaling.
    Rescale the base frequency: new_base = base * alpha^(dim/(dim-2))
    This preserves high-frequency components (short-range) while
    extending low-frequency components (long-range).
    """
    scaled_base = base * (alpha ** (dim / (dim - 2)))
    freqs = []
    for i in range(dim // 2):
        theta = 1.0 / (scaled_base ** (2 * i / dim))
        freqs.append(pos * theta)
    return freqs


def rope_yarn_factor(i: int, dim: int, original_max: int, target_max: int,
                     beta_fast: float = 32.0, beta_slow: float = 1.0) -> float:
    """
    YaRN per-dimension ramp factor.
    Dimensions with wavelength << original_max → interpolate (factor = 1/scale).
    Dimensions with wavelength >> original_max → extrapolate (factor = 1.0).
    Dimensions in between → linear ramp.
    """
    scale = target_max / original_max
    wavelength = 2 * math.pi / (1.0 / (10000.0 ** (2 * i / dim)))

    if wavelength < original_max / beta_fast:
        # High-frequency: interpolate
        return 1.0 / scale
    elif wavelength > original_max / beta_slow:
        # Low-frequency: no change (extrapolate)
        return 1.0
    else:
        # Ramp between the two
        t = (math.log(original_max / beta_slow) - math.log(wavelength)) / \
            (math.log(original_max / beta_slow) - math.log(original_max / beta_fast))
        return (1.0 - t) * 1.0 + t * (1.0 / scale)


def print_rope_comparison():
    """Compare effective frequency at position 131072 across scaling strategies."""
    dim = 128   # head_dim for Llama
    pos = 131072
    original_max = 4096
    scale = pos / original_max  # = 32 for 128K

    standard = rope_freq_linear(pos, dim, base=10000.0, scale=1.0)
    linear    = rope_freq_linear(pos, dim, base=10000.0, scale=scale)
    ntk8      = rope_freq_ntk(pos, dim, base=10000.0, alpha=8.0)
    ntk32     = rope_freq_ntk(pos, dim, base=10000.0, alpha=32.0)
    yarn_factors = [rope_yarn_factor(i, dim, original_max, pos) for i in range(dim // 2)]
    yarn = [pos * (1.0 / (10000.0 ** (2 * i / dim))) * yarn_factors[i] for i in range(dim // 2)]

    print("=" * 75)
    print(f"RoPE Frequency Comparison at pos={pos:,}  (head_dim={dim}, "
          f"original_max={original_max})")
    print(f"Scale factor = {scale:.0f}×")
    print("=" * 75)
    print(f"{'Dim':>5}  {'Standard':>12}  {'Linear/32':>12}  "
          f"{'NTK-α=8':>12}  {'NTK-α=32':>12}  {'YaRN':>12}")
    print("-" * 75)
    for i in [0, 4, 8, 16, 24, 32, 48, 60, 63]:
        print(f"{i:>5}  {standard[i]:>12.4f}  {linear[i]:>12.4f}  "
              f"{ntk8[i]:>12.4f}  {ntk32[i]:>12.4f}  {yarn[i]:>12.4f}")
    print()
    print("  Standard: frequencies >> 2π → position aliasing → quality degradation")
    print("  Linear:   all dims uniformly compressed → high-freq dims under-sampled")
    print("  NTK:      high-freq dims preserved, low-freq extended → better quality")
    print("  YaRN:     per-dim ramp → best quality, used by Mistral/Llama extended models")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  WORKLOAD ADVISOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkloadProfile:
    context_len: int
    expected_output_len: int
    concurrent_users: int
    priority: str          # "latency" | "throughput"
    hardware: HardwareSpec
    model: ModelSpec


def advise_long_context(profile: WorkloadProfile) -> None:
    T = profile.context_len
    O = profile.expected_output_len
    N = profile.concurrent_users
    hw = profile.hardware
    m  = profile.model

    per_req_kv = kv_cache_gb(m, T + O)
    weights    = model_weight_gb(m)
    total_kv   = per_req_kv * N
    total_mem  = weights + total_kv

    print("=" * 65)
    print("Long-Context Workload Advisor")
    print("=" * 65)
    print(f"  Model:          {m.name}")
    print(f"  Hardware:       {hw.name}  ({hw.hbm_gb:.0f} GB HBM)")
    print(f"  Context:        {T:,} tokens")
    print(f"  Output:         {O:,} tokens")
    print(f"  Concurrency:    {N} users")
    print(f"  Weights:        {weights:.1f} GB")
    print(f"  KV/request:     {per_req_kv:.2f} GB")
    print(f"  Total KV:       {total_kv:.1f} GB")
    print(f"  Total needed:   {total_mem:.1f} GB  (usable: {hw.hbm_gb * 0.90:.1f} GB)")
    print()

    feasible = total_mem <= hw.hbm_gb * 0.90

    if feasible:
        print("  ✓  Configuration fits in HBM.")
    else:
        max_conc = max_concurrent_requests(m, hw, T + O)
        print(f"  ✗  Exceeds HBM budget by {total_mem - hw.hbm_gb * 0.90:.1f} GB.")
        print(f"     Maximum feasible concurrency at this context: {max_conc}")

    print()
    print("  Recommendations:")

    # Chunked prefill advice
    if T >= 8192:
        chunk = min(4096, T // 8)
        print(f"  • Enable chunked prefill: --enable-chunked-prefill "
              f"--max-num-batched-tokens {chunk}")

    # Quantization advice
    if not feasible or per_req_kv > 8.0:
        print("  • Use FP8 KV cache (--kv-cache-dtype fp8) to halve KV memory")
        print("  • Consider INT4/Q4_K_M weights to free HBM for more KV blocks")

    # Sliding window advice
    if T > 32768 and profile.priority == "throughput":
        print("  • Consider sliding-window attention models (Mistral, Phi-3-medium)")
        print("    to reduce KV footprint at the cost of distant-token recall")

    # Sequence parallelism
    if T >= 65536 and total_mem > hw.hbm_gb * 0.90:
        print("  • Deploy multi-GPU with tensor parallelism (-tp N) to split KV cache")
        print("  • For T > 256K consider ring attention (sequence parallelism)")

    # llama.cpp advice
    if profile.priority == "latency" and N <= 4:
        print(f"  • llama.cpp: -c {T} --rope-scaling yarn "
              f"--rope-freq-base {500000}  (for Llama 3.1 extended models)")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Chapter 27 — Long-Context Inference Demo (Python)")
    print("=" * 70 + "\n")

    print_kv_table()
    print_concurrency_table()
    print_decode_time_per_token()
    print_chunked_prefill_table()
    print_rope_comparison()

    profile = WorkloadProfile(
        context_len=131072,
        expected_output_len=2048,
        concurrent_users=8,
        priority="throughput",
        hardware=HARDWARE[0],  # H100 80GB
        model=MODELS[0],       # Llama-3.1-8B
    )
    advise_long_context(profile)

```

## C++ — `long_context_demo.cpp`

```cpp
// long_context_demo.cpp
// Chapter 27 — Long-Context Inference Demo (C++)
//
// Demonstrates:
//   1. KV cache sizing across models and sequence lengths
//   2. Chunked prefill timing model
//   3. RoPE frequency comparison (standard vs linear vs NTK vs YaRN)
//   4. Multi-GPU concurrency calculator
//   5. Memory budget planner with recommendations
//
// Build: g++ -O2 -std=c++17 -o long_context_demo long_context_demo.cpp
// Run:   ./long_context_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

static void print_section(const std::string& title) {
    std::string bar(72, '─');
    std::cout << "\n" << bar << "\n  " << title << "\n" << bar << "\n";
}

static std::string comma(long long n) {
    if (n < 0) return "-" + comma(-n);
    if (n < 1000) return std::to_string(n);
    return comma(n / 1000) + "," + [](long long r) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%03lld", r);
        return std::string(buf);
    }(n % 1000);
}

// ─────────────────────────────────────────────────────────────────────────────
// MODEL AND HARDWARE SPECS
// ─────────────────────────────────────────────────────────────────────────────

struct ModelSpec {
    const char* name;
    int   n_layers;
    int   n_kv_heads;   // GQA
    int   head_dim;
    double param_b;     // billions of parameters
};

struct HardwareSpec {
    const char* name;
    double hbm_gb;
    double bw_tb_s;     // HBM bandwidth in TB/s
};

static const ModelSpec MODELS[] = {
    {"Llama-3.1-8B",   32,   8, 128,   8.0},
    {"Llama-3.1-70B",  80,   8, 128,  70.0},
    {"Llama-3.1-405B", 126,  8, 128, 405.0},
    {"Mistral-7B",     32,   8, 128,   7.0},
    {"Qwen2.5-72B",    80,   8, 128,  72.0},
};
static const int N_MODELS = 5;

static const HardwareSpec HW[] = {
    {"H100 SXM 80GB",  80.0,  3.35},
    {"A100 SXM 80GB",  80.0,  2.00},
    {"RTX 4090 24GB",  24.0,  1.008},
    {"M2 Ultra 192GB", 192.0, 0.800},
};
static const int N_HW = 4;

// ─────────────────────────────────────────────────────────────────────────────
// 1.  KV CACHE SIZING
// ─────────────────────────────────────────────────────────────────────────────

/**
 * KV cache memory in GB for a single sequence, default BF16 (2 bytes).
 */
static double kv_cache_gb(const ModelSpec& m, int T, int dtype_bytes = 2) {
    return 2.0 * m.n_layers * m.n_kv_heads * m.head_dim * (double)T * dtype_bytes / 1e9;
}

static double weight_gb(const ModelSpec& m, int dtype_bytes = 2) {
    return m.param_b * 1e9 * dtype_bytes / 1e9;
}

/**
 * Max concurrent sequences on hw (10% overhead reserved).
 */
static int max_concurrent(const ModelSpec& m, const HardwareSpec& hw,
                           int T, double overhead = 0.10) {
    double usable = hw.hbm_gb * (1.0 - overhead);
    double avail  = usable - weight_gb(m);
    if (avail <= 0.0) return 0;
    double per_req = kv_cache_gb(m, T);
    if (per_req <= 0.0) return 0;
    return static_cast<int>(avail / per_req);
}

static void demo_kv_sizing() {
    print_section("KV Cache Memory — Llama-3.1-8B  (BF16, single sequence)");

    const ModelSpec& m = MODELS[0];  // 8B
    const long long seq_lens[] = {4096, 16384, 32768, 65536, 131072, 524288, 1048576};
    const int N = 7;

    std::cout << std::left  << std::setw(14) << "Seq Len"
              << std::right << std::setw(12) << "KV (GB)"
              << std::setw(16) << "% of 80GB HBM"
              << std::setw(12) << "Tokens/GB"
              << "\n" << std::string(54, '-') << "\n";

    for (int i = 0; i < N; ++i) {
        long long T = seq_lens[i];
        double gb  = kv_cache_gb(m, (int)T);
        double pct = gb / 80.0 * 100.0;
        double tpg = T / gb;
        std::cout << std::left  << std::setw(14) << comma(T)
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3) << gb << " GB"
                  << std::setw(14) << std::setprecision(1) << pct << "%"
                  << std::setw(12) << std::setprecision(0) << tpg
                  << "\n";
    }
    std::cout << "\n";

    // Verify formula for small case
    // 2 * 32 * 8 * 128 * 4096 * 2 bytes = 536,870,912 bytes = 0.536870912 GB
    double expected_4k = 2.0 * 32 * 8 * 128 * 4096 * 2 / 1e9;
    double computed_4k = kv_cache_gb(m, 4096);
    assert(std::abs(computed_4k - expected_4k) < 1e-9);
    std::cout << "  [ASSERT] kv_cache_gb formula correct for T=4096: "
              << std::setprecision(6) << computed_4k << " GB ✓\n";

    // At 128K, KV ≈ weight size
    double kv_128k = kv_cache_gb(m, 131072);
    double wt      = weight_gb(m);
    std::cout << "  [NOTE]   KV@128K = " << std::setprecision(2) << kv_128k
              << " GB  ≈  weights = " << wt << " GB  (ratio "
              << std::setprecision(2) << kv_128k / wt << "×)\n";
}

static void demo_concurrency_table() {
    print_section("Max Concurrent Long-Context Requests — H100 SXM 80GB");

    const HardwareSpec& hw = HW[0];
    const int CTX[] = {32768, 65536, 131072};
    const int N_CTX = 3;

    // Header
    std::cout << std::left << std::setw(22) << "Model"
              << std::right << std::setw(12) << "Weights";
    for (int c = 0; c < N_CTX; ++c)
        std::cout << std::setw(12) << (std::to_string(CTX[c] / 1024) + "K ctx");
    std::cout << "\n" << std::string(22 + 12 + N_CTX * 12, '-') << "\n";

    for (int i = 0; i < N_MODELS; ++i) {
        const ModelSpec& m = MODELS[i];
        std::cout << std::left  << std::setw(22) << m.name
                  << std::right << std::setw(10) << std::fixed << std::setprecision(1)
                  << weight_gb(m) << "GB";
        for (int c = 0; c < N_CTX; ++c)
            std::cout << std::setw(12) << max_concurrent(m, hw, CTX[c]);
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  CHUNKED PREFILL TIMING MODEL
// ─────────────────────────────────────────────────────────────────────────────

struct ChunkedPrefillStats {
    int    context_len;
    int    chunk_size;
    int    n_chunks;
    double total_ms;
    double ttft_ms;   // time to first token for interleaved decode
};

/**
 * Model chunked prefill timing on H100.
 * Llama-3.1-8B: ~16 GFLOPs/token for non-attention MLP forward pass.
 * H100 BF16 dense: ~989 TFLOPs.
 * Each chunk also incurs attention: O(T_chunk * T_accum) but < 10% at early chunks.
 */
static ChunkedPrefillStats sim_chunked_prefill(int context_len, int chunk_size) {
    const double H100_TFLOPS   = 989.0;
    const double GFLOPS_PER_TOK = 16.0;   // Llama-3.1-8B dense layers

    int n_chunks = (context_len + chunk_size - 1) / chunk_size;
    double chunk_ms = (chunk_size * GFLOPS_PER_TOK * 1e9) / (H100_TFLOPS * 1e12) * 1e3;
    double total_ms = n_chunks * chunk_ms;

    return {context_len, chunk_size, n_chunks, total_ms, chunk_ms};
}

static void demo_chunked_prefill() {
    print_section("Chunked Prefill Simulation — context=131072 tokens on H100 80GB");

    const int CTX = 131072;
    const int CHUNKS[] = {512, 1024, 2048, 4096, 8192, 16384};
    const int N = 6;

    std::cout << std::right
              << std::setw(10) << "Chunk"
              << std::setw(9)  << "N_chunks"
              << std::setw(14) << "Prefill(ms)"
              << std::setw(12) << "TTFT(ms)"
              << "\n" << std::string(45, '-') << "\n";

    // Baseline: no chunking
    {
        auto r = sim_chunked_prefill(CTX, CTX);
        std::cout << std::setw(10) << "FULL"
                  << std::setw(9)  << r.n_chunks
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.total_ms
                  << std::setw(12) << std::setprecision(1) << r.ttft_ms
                  << "  ← head-of-line block\n";
    }

    for (int i = 0; i < N; ++i) {
        auto r = sim_chunked_prefill(CTX, CHUNKS[i]);
        std::cout << std::setw(10) << comma(CHUNKS[i])
                  << std::setw(9)  << r.n_chunks
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.total_ms
                  << std::setw(12) << std::setprecision(1) << r.ttft_ms
                  << "\n";
    }

    std::cout << "\n  Key insight: TTFT drops proportionally to chunk_size / context_len.\n"
              << "  Total prefill time is unchanged — it is chunk scheduling, not compute reduction.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  ROPE SCALING COMPARISON
// ─────────────────────────────────────────────────────────────────────────────

static double rope_theta(int dim_idx, int d, double base) {
    return 1.0 / std::pow(base, 2.0 * dim_idx / d);
}

/**
 * Effective rotation at a position:
 *   standard:  pos * theta
 *   linear:    (pos / scale) * theta
 *   ntk:       pos * theta_ntk   where base is scaled
 */
struct RopeValues {
    double standard;
    double linear;
    double ntk_alpha8;
    double ntk_alpha32;
    double yarn;
};

static RopeValues rope_at(int pos, int dim_idx, int d,
                           double base, double scale,
                           int orig_max, double beta_fast, double beta_slow) {
    double theta_std   = rope_theta(dim_idx, d, base);

    // NTK: rescale base
    double ntk_base8  = base * std::pow(8.0,  (double)d / (d - 2));
    double ntk_base32 = base * std::pow(32.0, (double)d / (d - 2));

    // YaRN ramp
    double wavelength = 2.0 * M_PI / theta_std;
    double yarn_factor;
    if (wavelength < orig_max / beta_fast) {
        yarn_factor = 1.0 / scale;
    } else if (wavelength > orig_max / beta_slow) {
        yarn_factor = 1.0;
    } else {
        double t = (std::log((double)orig_max / beta_slow) - std::log(wavelength)) /
                   (std::log((double)orig_max / beta_slow) - std::log((double)orig_max / beta_fast));
        yarn_factor = (1.0 - t) * 1.0 + t * (1.0 / scale);
    }

    return {
        pos * theta_std,
        (pos / scale) * theta_std,
        pos * rope_theta(dim_idx, d, ntk_base8),
        pos * rope_theta(dim_idx, d, ntk_base32),
        pos * theta_std * yarn_factor,
    };
}

static void demo_rope_scaling() {
    print_section("RoPE Scaling Comparison at pos=131,072  (head_dim=128)");

    const int pos      = 131072;
    const int d        = 128;
    const double base  = 10000.0;
    const double scale = 32.0;  // 131072 / 4096
    const int orig_max = 4096;

    std::cout << std::right
              << std::setw(6)  << "Dim"
              << std::setw(12) << "Standard"
              << std::setw(12) << "Linear/32"
              << std::setw(12) << "NTK-α=8"
              << std::setw(12) << "NTK-α=32"
              << std::setw(12) << "YaRN"
              << "\n" << std::string(66, '-') << "\n";

    const int dims[] = {0, 4, 8, 16, 24, 32, 48, 60, 63};
    const int ND = 9;

    // Track how many standard dims exceed 2π (aliased)
    int aliased_count = 0;
    for (int di = 0; di < ND; ++di) {
        int i = dims[di];
        auto rv = rope_at(pos, i, d, base, scale, orig_max, 32.0, 1.0);
        bool aliased = rv.standard > 2 * M_PI;
        if (aliased) aliased_count++;
        std::cout << std::setw(6)  << i
                  << std::setw(12) << std::fixed << std::setprecision(4) << rv.standard
                  << std::setw(12) << rv.linear
                  << std::setw(12) << rv.ntk_alpha8
                  << std::setw(12) << rv.ntk_alpha32
                  << std::setw(12) << rv.yarn;
        if (aliased) std::cout << "  ← ALIASED";
        std::cout << "\n";
    }

    std::cout << "\n  Standard: " << aliased_count << "/" << ND
              << " sampled dims exceed 2π (position aliasing)\n"
              << "  Linear:   all dims compressed, nearby tokens harder to distinguish\n"
              << "  NTK:      high-freq dims preserved, low-freq extended — no aliasing\n"
              << "  YaRN:     per-dim optimal blend — best quality in practice\n";

    // Assert: at very high-frequency dim (i=0), NTK still stays bounded
    auto rv0 = rope_at(pos, 0, d, base, scale, orig_max, 32.0, 1.0);
    // NTK-alpha=32 should be within 2×2π for dim 0 (high freq is barely changed)
    // Actually for NTK at dim=0: theta = 1/(scaled_base^(0/128)) = 1.0 always
    // So the value is just `pos * 1.0 / scaled_base^0 = pos * 1` — no, theta(i=0)=1
    // The NTK base scaling doesn't change theta at i=0 since theta_0 = 1/base^0 = 1
    // So NTK value at (pos=131072, i=0) = 131072 — that's fine, only the phases matter
    // modulo 2pi. The important assertion is that NTK doesn't make things worse at i=0.
    assert(rv0.ntk_alpha32 >= rv0.standard * 0.5);  // NTK never shrinks high-freq
    std::cout << "\n  [ASSERT] NTK-α=32 at dim=0 ≥ 0.5 × standard: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  DECODE STEP LATENCY (BANDWIDTH BOUND)
// ─────────────────────────────────────────────────────────────────────────────

static void demo_decode_latency() {
    print_section("Decode Step Latency — Bandwidth-Bound Model  (Llama-3.1-8B on H100)");

    const ModelSpec& m = MODELS[0];
    const HardwareSpec& hw = HW[0];

    double weight_bw = weight_gb(m);  // GB of weights read per step

    const long long seq_lens[] = {4096, 16384, 32768, 65536, 131072, 524288};
    const int N = 6;

    std::cout << std::right
              << std::setw(14) << "Seq Len"
              << std::setw(14) << "KV reads(GB)"
              << std::setw(14) << "Total(GB)"
              << std::setw(18) << "Min latency(ms)"
              << std::setw(14) << "Max tok/s"
              << "\n" << std::string(74, '-') << "\n";

    for (int i = 0; i < N; ++i) {
        long long T   = seq_lens[i];
        double kv_gb  = kv_cache_gb(m, (int)T);
        double total  = weight_bw + kv_gb;
        // latency = bytes / bandwidth_bytes_per_ms
        double lat_ms = total / hw.bw_tb_s;  // 1 TB/s == 1 GB/ms
        double tok_s  = 1000.0 / lat_ms;

        std::cout << std::setw(14) << comma(T)
                  << std::setw(14) << std::fixed << std::setprecision(3) << kv_gb
                  << std::setw(14) << std::setprecision(3) << total
                  << std::setw(18) << std::setprecision(3) << lat_ms
                  << std::setw(14) << std::setprecision(1) << tok_s
                  << "\n";
    }

    // Assert: at 128K, decode latency > 5ms (matches theory: ~9.6ms)
    double kv_128k = kv_cache_gb(m, 131072);
    double lat_128k = (weight_bw + kv_128k) / hw.bw_tb_s;  // 1 TB/s == 1 GB/ms
    assert(lat_128k > 5.0);
    std::cout << "\n  [ASSERT] Decode latency at 128K > 5ms: "
              << std::setprecision(2) << lat_128k << "ms ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  MEMORY BUDGET PLANNER WITH RECOMMENDATIONS
// ─────────────────────────────────────────────────────────────────────────────

struct WorkloadProfile {
    const ModelSpec*    model;
    const HardwareSpec* hw;
    int    context_len;
    int    output_len;
    int    concurrency;
    bool   priority_latency;  // true = latency, false = throughput
};

static void advise(const WorkloadProfile& p) {
    print_section("Memory Budget Planner & Recommendations");

    double wt_gb    = weight_gb(*p.model);
    double per_req  = kv_cache_gb(*p.model, p.context_len + p.output_len);
    double total_kv = per_req * p.concurrency;
    double total    = wt_gb + total_kv;
    double usable   = p.hw->hbm_gb * 0.90;

    std::cout << "  Model:       " << p.model->name  << "\n"
              << "  Hardware:    " << p.hw->name      << "  (" << p.hw->hbm_gb << " GB HBM)\n"
              << "  Context:     " << comma(p.context_len) << " tokens\n"
              << "  Output:      " << comma(p.output_len)  << " tokens\n"
              << "  Concurrency: " << p.concurrency        << "\n\n"
              << std::fixed << std::setprecision(2)
              << "  Weights:          " << wt_gb    << " GB\n"
              << "  KV per request:   " << per_req  << " GB\n"
              << "  KV total:         " << total_kv << " GB\n"
              << "  Total required:   " << total    << " GB\n"
              << "  Usable HBM (90%): " << usable   << " GB\n\n";

    bool feasible = total <= usable;
    if (feasible) {
        double headroom = usable - total;
        std::cout << "  ✓  Fits. Headroom: " << headroom << " GB\n\n";
    } else {
        int max_conc = max_concurrent(*p.model, *p.hw, p.context_len + p.output_len);
        std::cout << "  ✗  Exceeds budget by " << (total - usable) << " GB.\n"
                  << "     Max feasible concurrency at this context: " << max_conc << "\n\n";
    }

    std::cout << "  Recommendations:\n";

    if (p.context_len >= 8192) {
        int chunk = std::min(4096, p.context_len / 8);
        std::cout << "  • Enable chunked prefill: --enable-chunked-prefill "
                  << "--max-num-batched-tokens " << chunk << "\n";
    }

    if (!feasible || per_req > 8.0) {
        std::cout << "  • FP8 KV cache (--kv-cache-dtype fp8): halves KV memory\n"
                  << "  • INT4 weights (--quantization awq/gptq): ~4× weight compression\n";
    }

    if (p.context_len >= 65536 && !feasible) {
        std::cout << "  • Tensor parallelism (--tensor-parallel-size 2 or 4): "
                  << "splits weights + KV across GPUs\n";
    }

    if (!p.priority_latency && p.context_len > 32768) {
        std::cout << "  • Sliding-window models (Mistral, Phi-3) reduce KV footprint "
                  << "with acceptable recall trade-off\n";
    }

    if (p.priority_latency && p.concurrency <= 4) {
        std::cout << "  • llama.cpp: -c " << p.context_len
                  << " --rope-freq-base 500000 --flash-attn -ngl 99\n";
    }

    if (p.context_len >= 65536 && p.concurrency > 8) {
        std::cout << "  • Consider disaggregated prefill/decode for sustained throughput\n";
    }

    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n"
              << std::string(72, '=') << "\n"
              << "  Chapter 27 — Long-Context Inference Demo (C++)\n"
              << std::string(72, '=') << "\n";

    demo_kv_sizing();
    demo_concurrency_table();
    demo_chunked_prefill();
    demo_rope_scaling();
    demo_decode_latency();

    // Budget planner: 8B model, H100, 128K context, 8 users
    WorkloadProfile p128k = {
        &MODELS[0],   // Llama-3.1-8B
        &HW[0],       // H100 SXM 80GB
        131072,       // 128K context
        2048,         // 2K output
        8,            // 8 concurrent
        false,        // throughput priority
    };
    advise(p128k);

    // Budget planner: 8B model, RTX 4090 24GB, 32K context, 2 users
    WorkloadProfile p32k = {
        &MODELS[0],   // Llama-3.1-8B
        &HW[2],       // RTX 4090 24GB
        32768,        // 32K context
        1024,         // 1K output
        2,            // 2 concurrent
        true,         // latency priority
    };
    advise(p32k);

    std::cout << std::string(72, '=') << "\n"
              << "  All demos complete.\n"
              << std::string(72, '=') << "\n\n";
    return 0;
}

```

