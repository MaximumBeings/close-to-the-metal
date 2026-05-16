# Chapter 27: Long-Context Inference — 128K and Beyond

Long-context inference is one of the most demanding workloads in production LLM serving.
Models that accept 128K, 512K, or even 1M tokens change what is architecturally possible —
entire codebases, legal corpora, hours of audio transcripts, or thousands of API documentation
pages can be ingested in a single request.
But they also stress every component of the inference stack simultaneously: memory capacity,
memory bandwidth, compute throughput, CUDA kernel design, and scheduler policy.

This chapter dissects the problem from first principles.
We derive exactly how much memory each token costs, why naive attention breaks at long contexts,
what the engineering community has built to fix it, and how to configure vLLM and llama.cpp
to serve 128K+ contexts reliably in production.

---

## 27.1  The O(n²) Wall

Transformer self-attention has two distinct costs that scale with sequence length in different ways.

### 27.1.1  Compute Cost

For a single attention layer with sequence length `T`, `H` heads, and head dimension `d`:

```
FLOPs_attention = 4 * T² * H * d   (approximate, forward-pass only)
```

The `T²` factor comes from computing `Q·Kᵀ` (shape `T × T` per head) and then multiplying
that by `V`.
Doubling the sequence length quadruples the attention FLOP count.

For a model like Llama 3.1 with `H=32`, `d=128`, running at `T=128K`:

```
FLOPs_attention ≈ 4 × (131072)² × 32 × 128
                ≈ 4 × 1.72 × 10¹⁰ × 4096
                ≈ 2.82 × 10¹⁴ FLOPs  (per layer, prefill only)
```

At 32 layers that's roughly 9 × 10¹⁵ FLOPs just for attention — more than the entire
non-attention part of the network.

### 27.1.2  KV-Cache Memory Cost

During decode, every previously computed key and value tensor must be retained in HBM so
that the current token can attend over the whole context.
The memory cost is:

```
KV_bytes = 2 × n_layers × n_kv_heads × head_dim × T × bytes_per_element
```

The `2` accounts for both K and V tensors.

For Llama 3.1 8B with GQA (`n_kv_heads=8`, `head_dim=128`, `n_layers=32`) in BF16:

```
KV_bytes = 2 × 32 × 8 × 128 × T × 2
         = 131,072 × T  bytes
         ≈ 128 KB per token
```

At 128K tokens: `128 KB × 131,072 = 16,777,216 KB = 16 GB`.

A single 128K request consumes the same HBM as the entire Llama 3.1 8B model in BF16
(which is itself ~16 GB).
On an 80 GB H100 that leaves only ~44 GB after OS and activation overhead — enough for
perhaps two or three simultaneous 128K contexts.

### 27.1.3  Why Decode Stays Bandwidth-Bound

Despite the memory volume, decode remains bandwidth-bound at long context because each
decode step reads the entire KV cache sequentially (O(T) bandwidth) while performing
only O(T·d) arithmetic.
The arithmetic intensity of decode never rises above a few FLOPs/byte regardless of context
length, so the H100 ridge point (≈ 295 FLOPs/byte) is never crossed during generation.

Prefill, however, saturates compute: the attention kernel tiles QK^T across the sequence
and reuses data from shared cache.
At `T ≥ 8K` on an H100, prefill arithmetic intensity typically exceeds 200 FLOPs/byte and
becomes compute-bound.

---

## 27.2  KV Cache Memory Budget — Worked Calculator

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

---

## 27.3  The Attention Complexity Problem in Depth

### 27.3.1  FlashAttention Does Not Remove the O(n²) Wall

FlashAttention (Chapters 4–5) dramatically reduces **memory** usage from O(T²) to O(T) by
tiling the attention computation in SRAM and never materializing the full score matrix in HBM.
It also reduces HBM round-trips and improves arithmetic intensity.

But it does **not** change the **FLOP count** of attention.
FlashAttention still computes `Q·Kᵀ` for every (query, key) pair — it just streams through
the computation without staging intermediate results in HBM.
The total floating-point work is O(T²·d) regardless of tiling.

This means that at very long sequences (128K+), the prefill stage is genuinely compute-heavy —
not a memory bandwidth problem but a raw FLOPs problem.
An H100 with FlashAttention-3 can sustain ~500 TFLOPs on an attention kernel.
At T=128K, L=32, d=128, H=32:

```
FLOPs_total_attention ≈ 4 × T² × H × d × L
                       ≈ 4 × (1.31 × 10⁵)² × 32 × 128 × 32
                       ≈ 9.2 × 10¹⁵ FLOPs
```

At 500 TFLOPs that's roughly **18 seconds of pure attention compute** for a single 128K prefill
on one H100.
This is why chunked prefill and multi-GPU tensor parallelism are both necessary at scale.

### 27.3.2  Memory Bandwidth During Decode

The decode bottleneck is different.
Each auto-regressive step reads:

- All model weight matrices: ~16 GB for Llama-3.1-8B in BF16
- The full KV cache accumulated so far

At T=128K the KV cache is 16 GB for Llama-3.1-8B, so each decode step reads 32 GB total.
On an H100 at 3.35 TB/s that's a minimum of `32 / 3350 ≈ 9.6 ms` per token, or about
104 tokens/second — from a single 80 GB H100 for a single 128K context.

Increasing batch size at long context has diminishing returns: doubling the batch doubles
the total KV bytes read per step, with no additional reuse of the attention data.

---

## 27.4  Sparse and Sliding-Window Attention

When exact full-context attention is either too slow or requires more memory than available,
structured sparsity patterns can reduce both costs.

### 27.4.1  Sliding Window Attention

In sliding window attention (SWA), each token attends only to the W tokens immediately
preceding it (the "window"), plus a fixed set of "sink" tokens (usually the first few tokens
of the context that receive disproportionate attention mass across all queries).

```
Memory: O(W × d)  instead of  O(T × d)
Compute: O(W × T)  instead of  O(T²)
```

Mistral 7B uses `W = 4096` within each transformer layer.
Because there are multiple layers and information propagates through them, receptive field
grows as `W × L` in principle, but in practice distant tokens beyond ~`W × 2` are invisible.

**Trade-off:** SWA models handle local reasoning well but struggle with tasks that require
linking tokens separated by more than the window size.
In practice, many long-document retrieval and summarization tasks are well-served by SWA,
while code-completion tasks that reference a function defined 60K tokens ago are not.

### 27.4.2  Sink Tokens (StreamingLLM)

The "attention sink" observation: regardless of task, attention weights for the first few
tokens (position 0 and 1) are consistently high even when semantically irrelevant.
This is an artifact of the softmax normalization — models learn to "dump" probability mass
onto early tokens as a no-op attention.

StreamingLLM exploits this: always keep the first 4 tokens as sinks plus a rolling window
of the most recent W tokens.
This enables unbounded streaming inference (truly infinite context) at the cost of losing
everything between position 4 and `T - W`.

Use cases: live transcription, continuous chat, edge devices with limited memory.
Not suitable for: document Q&A over fixed corpora, legal/contract analysis, code review.

### 27.4.3  BigBird / Longformer Patterns

BigBird and Longformer introduced hybrid patterns that combine:

- Local window attention (every token attends to nearby tokens)
- Global tokens (special CLS or summary tokens attend to the full sequence)
- Random attention (each token attends to a random set of keys)

These reduce compute to O(T) while preserving some global information propagation.
None of the current production LLMs (Llama, Mistral, Qwen, DeepSeek) use these patterns
in their base architectures, though they appear in some document-processing fine-tunes.

---

## 27.5  RoPE Position Encoding Extension

Rotary Position Embedding (RoPE) is the dominant position encoding in modern LLMs.
It encodes position by rotating query and key vectors in 2D planes, with rotation frequency
varying by dimension index.
The key property that matters for long-context extension: **models trained on short contexts
are exposed to rotation frequencies they have never seen during training when T exceeds
the training maximum**.

### 27.5.1  The Position Aliasing Problem

Standard RoPE with base=10000 was trained at `T_max = 4096` for Llama 2.
The frequency for dimension `i` is:

```
θᵢ = 1 / (10000^(2i/d))
```

For the lowest-frequency dimension (`i = d/2 - 1 = 63`):
```
θ₆₃ = 1 / (10000^(126/128)) ≈ 8.1 × 10⁻⁴
```

The position enters one full rotation at `2π / θ` ≈ 7760 tokens.
At `T = 128K`, this dimension has completed `128K / 7760 ≈ 16.5` full rotations —
a position the model has never seen during training.
This causes **position aliasing**: the model cannot distinguish position 128K from
positions 128K ± 7760.

### 27.5.2  Linear Interpolation

The simplest fix (Su et al., 2023): scale down all positions by `T / T_train`.
If the model was trained at 4096 and we want 128K, compress positions by 32×:

```
effective_pos = pos / 32
```

This ensures every position value seen at inference was also seen during training.
The cost: nearby tokens that are within 32 positions of each other will have nearly
identical rotations in high-frequency dimensions, reducing the model's ability to
distinguish them precisely.

Empirically, linear interpolation requires a brief fine-tuning run (1000–2000 steps)
on long-context data to recover perplexity.
Without fine-tuning, perplexity at positions > 4096 degrades even with linear interpolation.

### 27.5.3  NTK-Aware Scaling

Neural Tangent Kernel (NTK) theory suggests that the bandwidth of a network's implicit
function approximation is governed by the maximum frequency it encodes.
For RoPE, the highest frequencies are in the lowest-indexed dimensions.

NTK-aware scaling (bloc97, 2023) rescales the **base** rather than the positions:

```
new_base = base × α^(d/(d-2))
```

where `α = T_target / T_train` is the extension factor.

Effect: high-frequency dimensions (small `i`) are stretched very little (they have short
wavelengths and are not the bottleneck), while low-frequency dimensions (large `i`) are
stretched significantly.
This preserves the model's ability to distinguish nearby tokens (high-freq dims intact)
while allowing distant positions to be encoded without aliasing (low-freq dims extended).

NTK scaling works **without any fine-tuning** for moderate extensions (up to ~8× the
training context) and with brief fine-tuning for larger extensions.

### 27.5.4  YaRN (Yet Another RoPE extensioN)

YaRN (Peng et al., 2023) applies a per-dimension ramp between linear interpolation and
no-scaling:

```
For dimension i:
  λᵢ = wavelength = 2π / θᵢ
  
  if λᵢ < T_original / β_fast:   factor = 1/scale   (interpolate)
  if λᵢ > T_original / β_slow:   factor = 1.0        (no change)
  else:                            factor = linear ramp
```

Typical values: `β_fast = 32`, `β_slow = 1`.

YaRN also applies a **temperature correction** to the attention logits: dividing by
`√scale` (or a tuned version of it) compensates for the fact that scaled positions
produce lower-magnitude dot products, which can cause attention entropy to increase
inappropriately at long contexts.

YaRN is currently the best-performing RoPE extension method and is the technique used
by most production-quality extended models:

- Llama 3.1 (all sizes, native 128K) uses a combination of RoPE base increase to 500,000
  plus YaRN-style temperature correction

- Mistral v0.3, Mistral Large use YaRN
- Qwen 2.5 uses "dual chunk attention" + extended RoPE base (1,000,000)

### 27.5.5  LongRoPE

LongRoPE (Ding et al., 2024) goes further: rather than applying a closed-form formula
to all dimensions uniformly, it learns per-dimension rescaling factors empirically by
searching over a population of long-context prompts.
The resulting factors are slightly non-monotone and differ from any analytic formula —
they are simply whatever minimizes perplexity on the extension corpus.

LongRoPE enables Phi-3 to extend from 4K to 128K with minimal fine-tuning, and is part
of the Microsoft Phi-3-medium-128K training recipe.

---

## 27.6  Sequence Parallelism and Ring Attention

When a single context exceeds the KV memory of one GPU, the sequence must be partitioned
across multiple GPUs.

### 27.6.1  Tensor Parallelism (TP) — Partial Help

Standard tensor parallelism splits attention **head** dimension across GPUs.
With TP=8 and H=32 heads, each GPU handles 4 heads.
The KV cache for those 4 heads is 4/32 = 12.5% of the total — so TP=8 reduces
KV memory by 8×.

This is the easiest approach and is directly supported by vLLM's `--tensor-parallel-size`.
Its limitation: each GPU still processes the full sequence (all T tokens) for its
subset of heads.
The attention compute is therefore `O(T² × H/TP)` per GPU — same algorithmic complexity,
just parallelized.

### 27.6.2  Sequence Parallelism (SP)

Sequence parallelism assigns each GPU a contiguous subsequence `[tᵢ, tᵢ₊₁)` of tokens.
The GPU computes Q, K, V for its shard only.
For full attention, each Q-token needs to attend to all K-tokens across all shards —
requiring an all-gather of K and V before the attention dot product.

For causal (left-to-right) attention this is wasteful because position j cannot attend
to position k > j: the all-gather can be replaced by a more efficient ring protocol.

### 27.6.3  Ring Attention

Ring attention (Liu et al., 2023) exploits causality to eliminate the all-gather entirely.
GPUs are arranged in a ring.
In each step, each GPU:
1. Computes attention for its current Q chunk against the local K/V chunk (online softmax, FlashAttention-style)
2. Passes its K/V chunk clockwise to the next GPU
3. Receives the next K/V chunk from the GPU counterclockwise

After `n_gpus` ring steps, each GPU has accumulated the correct attention output for its
Q shard (the online softmax correctly accumulates partial results from each ring step).

Cost per step: one K/V chunk send + receive = `2 × T/n_gpus × d × bytes` transferred.
Total communication volume = one full K/V tensor transfer per GPU per layer — exactly what
tensor parallelism requires as well, making ring attention communication-equivalent to TP.

Ring attention is available in JAX (Google) and in several research PyTorch implementations.
vLLM's production implementation uses a hybrid: TP for heads + SP for sequence when
`--tensor-parallel-size` is set.

### 27.6.4  Practical Sequence Parallelism Thresholds

| Context length | Recommended strategy |
|---|---|
| ≤ 32K  | Single GPU (H100 80GB) or TP=2 |
| 32K–128K | TP=4 or TP=8; chunked prefill |
| 128K–512K | TP=8 + SP=2; multi-node ring |
| 512K–1M | Multi-node, ring attention required |
| > 1M | Research territory; use Ulysses+Ring hybrid |

---

## 27.7  vLLM Configuration for Long Contexts

### 27.7.1  Core Flags

```bash
# Serve Llama-3.1-8B-Instruct at full 128K context on H100 80GB
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 131072 \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype auto \
    --enforce-eager
```

`--max-model-len` tells vLLM the maximum sequence length to support.
vLLM allocates KV cache blocks statically at startup for this length.
Setting this higher than needed wastes HBM and reduces concurrency.

`--enable-chunked-prefill` splits long prefill requests into chunks of size
`--max-num-batched-tokens`, interleaving them with decode steps.
This dramatically reduces time-to-first-token (TTFT) for concurrent decode users
while a long prefill is in flight.

`--max-num-batched-tokens` sets both:
- The maximum tokens per decode batch (controls batch throughput)
- The chunk size for chunked prefill

Typical values: 2048–8192.
Larger values → higher throughput but higher TTFT for concurrent decode requests.

### 27.7.2  KV Cache Quantization

```bash
# FP8 KV cache: halves KV memory, slight quality degradation
--kv-cache-dtype fp8

# Check what fraction of HBM is used by KV blocks vs weights
--show-hidden-states   # logs KV block allocation at startup
```

FP8 KV cache (E4M3 format) reduces per-token KV cost by 2× with typically < 0.5% perplexity
increase on standard benchmarks.
At 128K context this is effectively doubles concurrency capacity.

For quantized models:

```bash
# INT4 weights (AWQ) + FP8 KV = ~4× memory reduction combined
--quantization awq \
--kv-cache-dtype fp8
```

Llama-3.1-8B-AWQ weights ≈ 5 GB; FP8 KV at 128K ≈ 8 GB → fits comfortably on a 24 GB GPU.

### 27.7.3  Multi-GPU Long-Context Serving

```bash
# 70B model with 128K context: requires TP=4 minimum
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --max-model-len 131072 \
    --tensor-parallel-size 4 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90
```

Memory budget for 70B at TP=4 (4× H100 80GB):

- Weights: 70B × 2 bytes = 140 GB → 35 GB per GPU
- KV at 128K: 80 GB total → 20 GB per GPU (GQA heads also split by TP)
- Activations + overhead: ~8 GB per GPU
- Total per GPU: ~63 GB ✓ (within 80 GB)

### 27.7.4  V2 Block Manager and Prefix Caching

vLLM's V2 block manager (enabled with `--use-v2-block-manager`, default in vLLM ≥ 0.5)
supports prefix caching: if two requests share an identical prefix (e.g., the same system
prompt), the KV blocks for that prefix are computed once and shared.

At long context, prefix caching is especially valuable when:

- A long document is queried multiple times with different questions (RAG)
- A large code file is context for multiple completion requests
- A long system prompt is reused across sessions

```bash
--enable-prefix-caching  # explicit flag in older versions; default on in v2 block manager
```

The prefix must be byte-for-byte identical (same tokens) to hit the cache.
Any difference in even one token position busts the cache for that position and all
subsequent positions.

### 27.7.5  Disaggregated Prefill for Long Contexts

Chapter 18 covered disaggregated prefill/decode; it is especially relevant at long context.
A 128K prefill takes ~18 seconds of compute on one H100 (as computed above).
During that time, a mixed prefill+decode instance cannot serve decode requests efficiently.

Production setups at 128K+ typically use:

- **Prefill pool**: beefy multi-GPU nodes dedicated to long prefills (high compute, lower concurrency)
- **Decode pool**: many smaller nodes for continuous decode (high concurrency)
- **KV migration**: after prefill completes, KV blocks are transmitted over NVLink/InfiniBand
  to the decode pool

```python
# vLLM disaggregated mode (experimental as of vLLM 0.6)
# Prefill node:
python -m vllm.entrypoints.disagg_prefill_proxy \
    --prefill-model meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 131072 \
    --tensor-parallel-size 2

# Decode node:
python -m vllm.entrypoints.disagg_decode \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 131072
```

---

## 27.8  llama.cpp Configuration for Long Contexts

### 27.8.1  Context Size and Rope Scaling

```bash
# Run Llama-3.1-8B-GGUF at 32K context
llama-cli \
    -m Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -c 32768 \
    --rope-freq-base 500000 \
    -ngl 33 \
    --threads 8
```

`-c` sets the context window size.
If `-c` exceeds the model's trained context, position aliasing occurs unless RoPE scaling
is applied.

`--rope-freq-base` overrides the base frequency.
For Llama 3.1 family: the official fine-tune used `base = 500000` (versus the Llama 2
default of 10000) to enable 128K naturally — no additional scaling needed if you use the
correct base.

### 27.8.2  YaRN Scaling in llama.cpp

For models that were not natively trained at long context (e.g., Llama 2 at 4K):

```bash
# YaRN scaling to 128K for a 4K-trained model
llama-cli \
    -m llama-2-13b.Q4_K_M.gguf \
    -c 131072 \
    --rope-scaling yarn \
    --rope-scale 32 \
    --yarn-orig-ctx 4096 \
    --yarn-ext-factor 1.0 \
    --yarn-attn-factor 0.1 \
    -ngl 40
```

`--rope-scaling yarn` activates the YaRN algorithm.
`--rope-scale` is the scale factor (T_target / T_train = 131072 / 4096 = 32).
`--yarn-orig-ctx` is T_train.
`--yarn-ext-factor` blends between linear interpolation (1.0) and NTK (0.0); 1.0 is
standard YaRN.
`--yarn-attn-factor` is the temperature correction multiplier; 0.1 is the default
recommended value from the YaRN paper.

### 27.8.3  GPU Offload and Long Context

At 128K context and Q4_K_M quantization on Llama-3.1-8B:
```
Weights: ~4.8 GB (Q4_K_M)
KV cache (BF16, full context): 16 GB
KV cache (FP16): 16 GB
KV cache (F16 half-precision in llama.cpp): 16 GB
```

llama.cpp KV cache precision is controlled by `--flash-attn` (enables FlashAttention-2
kernel) and type arguments.
For 16 GB cards (RTX 4080/4090): offload all layers (`-ngl 99`) and set `-c` to
whatever fits after accounting for KV.

```bash
# RTX 4090 24GB: 4.8GB weights + KV budget
# KV per token = 2*32*8*128*2 / 1e9 = 0.000131072 GB
# Max context = (24 - 4.8 - 2) / 0.000131072 ≈ 130,700 tokens
llama-cli \
    -m Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -c 130000 \
    --rope-freq-base 500000 \
    -ngl 99 \
    --flash-attn
```

`--flash-attn` is critical at long context: without it, llama.cpp's attention kernel
uses O(T²) memory in VRAM for intermediate attention scores, which will OOM at T ≥ 32K.

### 27.8.4  Continuous Batching in llama.cpp

llama.cpp's server supports continuous batching via the `-np` (parallel slots) flag:

```bash
llama-server \
    -m Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -c 131072 \
    --rope-freq-base 500000 \
    -ngl 99 \
    --flash-attn \
    -np 2 \        # 2 parallel sequences
    --host 0.0.0.0 \
    --port 8080
```

With long contexts and `-np 2`, each slot gets half the context budget.
Set `-c` equal to the **total** token capacity across all slots; each slot's maximum
sequence length is effectively `c / np`.

---

## 27.9  C++ Implementation: Memory Budget Planner

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

---

## 27.10  Retrieval-Augmented Generation vs. Full Context

Long context and RAG are not mutually exclusive — they represent two points on a
cost-quality trade-off curve that engineers must navigate deliberately.

### 27.10.1  When Full Context Wins

Full context is superior when:

**Needle-in-a-haystack precision is required.** A 128K context can contain a contract
verbatim; the model can cite specific clause numbers and cross-reference provisions.
RAG may miss subtle implications spread across non-adjacent chunks.

**Reasoning over the whole document matters.** Code review of a 10K-line codebase,
where a function defined at line 5000 interacts with a function at line 30000, benefits
from the model holding both in context simultaneously.

**The query is unpredictable.** If users can ask anything about a document and you
cannot predict which chunks to retrieve, full ingestion avoids retrieval miss.

**Latency is secondary.** Legal contracts, financial filings, medical records — high
value per query justifies the cost of a 128K prefill.

### 27.10.2  When RAG Wins

RAG is superior when:

**The corpus is larger than any context window.** A 10M-token codebase cannot fit in
any model's context at once — retrieval is mandatory.

**Freshness matters.** New documents can be embedded and added to a vector store in
seconds; adding them to a cached context requires re-prefill.

**Cost dominates.** A RAG query over 2K retrieved tokens is 64× cheaper than a full
128K prefill.

**Multiple users share a corpus.** A company knowledge base queried by 1000 users per
day benefits from shared retrieval infrastructure; the embedding index is computed once.

### 27.10.3  Hybrid Approaches

Production systems increasingly use a tiered approach:

1. **Fast retrieval** (BM25 or dense embedding) to identify the 5–10 most relevant
   document sections (total ~2K tokens).
2. **Context window** to load those sections plus the original query (~4K tokens).
3. **Optional full context** for high-confidence, high-value queries identified by a
   lightweight classifier.

This gives the cost profile of RAG for 95% of queries while reserving full-context
processing for the cases where it actually matters.

---

## 27.11  Production Benchmarking Patterns

### 27.11.1  Metrics to Track

For long-context workloads, standard throughput metrics are insufficient.
The critical metrics are:

**Time-to-First-Token (TTFT):** Dominated by prefill time at long context.
A 128K prefill on one H100 takes 10–20 seconds without chunking.
Set SLO targets separately for short (< 4K) and long (> 32K) contexts.

**Time-per-Output-Token (TPOT):** The decode step latency.
At 128K context, TPOT is bandwidth-limited to ~10 ms/token on H100.
This is roughly fixed regardless of batch size (up to batch ~8).

**KV cache utilization:** What fraction of allocated KV blocks are in use.
High utilization → good efficiency; above 95% → risk of OOM on bursty traffic.
Monitor with `vllm_gpu_cache_usage_perc` Prometheus metric.

**Prefill queue depth:** How many long-context requests are waiting for a prefill slot.
Should stay near zero; rising queue depth signals prefill pool undersizing.

### 27.11.2  Benchmarking Script

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

### 27.11.3  Expected Latency Profiles

Empirical baselines on H100 80GB serving Llama-3.1-8B-Instruct (vLLM, chunked prefill,
chunk=4096):

| Context | TTFT p50 | TTFT p95 | TPOT (decode) |
|---|---|---|---|
| 1K | 15 ms | 22 ms | 8 ms |
| 4K | 45 ms | 70 ms | 8 ms |
| 16K | 180 ms | 240 ms | 8 ms |
| 32K | 350 ms | 450 ms | 9 ms |
| 64K | 700 ms | 900 ms | 10 ms |
| 128K | 1,400 ms | 1,800 ms | 12 ms |

TTFT scales roughly linearly with context (chunked prefill amortizes the prefill
across multiple steps, but total compute is the same).
TPOT is nearly constant until context exceeds the KV bandwidth threshold (~80K tokens
where KV dominates over weight reads), then grows slowly.

---

## 27.12  Common Pitfalls and Failure Modes

### 27.12.1  OOM at Startup with Aggressive `--max-model-len`

vLLM pre-allocates KV blocks at startup based on `--max-model-len` and
`--gpu-memory-utilization`.
If these two values combined require more HBM than available, vLLM raises an OOM error
before serving a single request.

```
RuntimeError: No available memory for the cache blocks.
Try increasing `gpu_memory_utilization` or decreasing `max_model_len`.
```

Fix: Either reduce `--max-model-len`, lower the quantization level to free weight memory,
or add GPUs.
The relationship is:

```
KV_blocks = floor((gpu_memory_utilization × hbm_gb - weights_gb) / block_size_gb)
block_size_gb = 2 × n_layers × n_kv_heads × head_dim × block_tokens × bytes / 1e9
```

vLLM's default `block_tokens = 16`.

### 27.12.2  Prefix Cache Misses on Large Prompts

A common pattern: a large document (50K tokens) is loaded as a system prompt, and each
user query appends a question.
If the system prompt is identical across requests, prefix caching should reuse its KV
blocks.

The failure mode: the system prompt contains a timestamp, session ID, or per-user
username injected before the document.
Even one different token busts the prefix cache for everything after it.

Fix: place volatile tokens (user ID, timestamp) **after** the document, not before it.

```
# BAD: User ID before document — cache miss every time
SYSTEM = f"[User: {user_id}]\n{large_document}\n\nAnswer the following:"

# GOOD: Document first, user context after
SYSTEM = f"{large_document}\n\nUser context: {user_id}\nAnswer the following:"
```

### 27.12.3  llama.cpp Context Fragmentation

llama.cpp uses a single flat KV cache buffer of size `-c`.
When running with multiple parallel slots (`-np`), each slot gets a contiguous subrange.
If one slot's context grows much larger than others, the KV cache is fragmented with
unused space that cannot be reclaimed.

Mitigation: set `-np` equal to your expected concurrency and `-c` to `np × max_ctx_per_slot`.

### 27.12.4  RoPE Scaling Misconfiguration

Using a model trained with `base=500000` (Llama 3.1) but explicitly setting
`--rope-freq-base 10000` (the Llama 2 default) will cause severe degradation at
positions > 4096 even though the model can natively handle 128K.

Similarly, applying YaRN scaling to a model that was already YaRN-fine-tuned
double-applies the correction and degrades quality.

Always check the model's `config.json` for `rope_scaling` field before applying
any external scaling flags.

### 27.12.5  FlashAttention Off in llama.cpp

Without `--flash-attn`, llama.cpp allocates an `O(T²)` intermediate tensor for the
attention logits in CPU or GPU memory.
At T=32K this is `4 × 32768² ≈ 4 GB` per layer — 32 layers = 128 GB just for attention
intermediates.
This will OOM even on an H100 80GB.

Always pass `--flash-attn` when using contexts above ~8K in llama.cpp.

---

## 27.13  Summary and Decision Checklist

Long-context inference requires coordinating hardware, model architecture, position
encoding, scheduling, and quantization choices simultaneously.

**Memory budget first.** Calculate `kv_cache_gb = 2 × L × H_kv × d × T × bytes` before
anything else.
If it doesn't fit, no amount of configuration will fix it — you need quantization, fewer
layers, or more GPUs.

**Chunked prefill always at T > 8K.** The TTFT improvement is free — total prefill time
is unchanged, but decode requests are unblocked earlier.

**RoPE scaling must match the model.** Use the base frequency from the model's
`config.json`.
Do not apply YaRN to a model that already has it embedded in its fine-tune.

**Prefix caching requires deterministic prefix ordering.** Put volatile tokens at the end.

**Benchmark TTFT and TPOT separately.** They are dominated by different bottlenecks
(prefill compute vs. decode bandwidth) and require different optimizations.

**FlashAttention is mandatory at long context** in both vLLM and llama.cpp.
It is the difference between O(T) and O(T²) HBM usage for attention intermediates.

---

*Chapter 28 covers llama.cpp as a deployment platform: model formats, GGUF internals,
the C API, and building production applications directly on the llama.cpp library.*


---

## Chapter Summary

- **Context length scaling**: attention FLOPs scale as O(T²) and KV cache memory scales as O(T); at T = 128K tokens, naive implementation is infeasible without optimizations.
- **Ring attention**: distributes the attention computation across devices by passing K/V blocks in a ring; each device computes attention for its Q slice against all K/V in O(T/N) local work.
- **vLLM long-context config**: `--max-model-len 131072` sets the context window; combined with `--enable-prefix-caching` and chunked prefill, this is the recommended long-context stack.
- **KV cache memory at 128K**: for a LLaMA-3 70B model (32 layers, 8 KV heads, d_k = 128, BF16) at 128K tokens, one sequence occupies ~67 GB — more than a single A100.
- **RoPE interpolation**: extending context beyond training length requires interpolating or extrapolating RoPE frequencies; LongRoPE, YaRN, and dynamic NTK scaling are the dominant approaches.
- **Sliding window attention**: Mistral-style attention limits the attention span to W tokens at each layer, reducing KV cache to O(W) while accepting some long-range information loss.
- **Kimi's Moon-Cache**: hierarchical KV storage using GPU → CPU DRAM → SSD tiers; blocks are promoted/demoted based on recency, enabling 1M-token contexts.

---

## Self-Check Questions

1. A 128K-token sequence at LLaMA-3 70B (32 layers, 8 GQA KV heads, d_k = 128, BF16) — compute the KV cache memory in GB. How many H100 80 GB cards are needed just for the KV cache? *(Section 27.1)*

2. Attention FLOPs for a full 128K-token prefill scale as O(T²). Estimate the prefill time on an H100 (1 979 TFLOPS BF16) for LLaMA-3 70B with d_model = 8 192, 64 attention heads, 128K tokens. *(Section 27.2)*

3. YaRN scales RoPE frequencies to extend the context window from 8K to 128K. What is the interpolation factor λ, and what perplexity degradation does it introduce versus a model trained natively at 128K? *(Section 27.4)*

4. Sliding window attention with window W = 4 096 limits attention to recent tokens. For a RAG document retrieval task where the relevant passage is at position 60K in a 128K context, would sliding window attention find the passage? Explain. *(Section 27.5)*

5. Kimi's Moon-Cache has three tiers: GPU (2 GB), CPU (128 GB), SSD (2 TB). A 1M-token sequence needs ~500 GB of KV cache. Describe how the hierarchical manager decides which blocks to evict from GPU and when to prefetch from SSD. *(Section 27.6)*


---

## Worked Solutions

### Question 1
**LLaMA-3 70B: 32 layers, 8 GQA KV heads, d_k=128, BF16. Sequence=128K tokens.**

**KV cache per token:**
```
bytes_per_token = 2 (K and V) x 32 layers x 8 heads x 128 dim x 2 bytes
                = 2 x 32 x 8 x 128 x 2
                = 131,072 bytes = 128 KB per token
```

**Total KV cache for 128K tokens:**
```
total = 128,000 x 128 KB = 16,384,000 KB = 16 GB
```

**H100 80 GB cards needed just for KV cache:**
```
cards = ceil(16 GB / 80 GB) = 1 card
```

Wait -- 16 GB fits on one H100. But the model weights also need space: 70B x 2 bytes = 140 GB. So in total:
```
total_HBM = 140 GB (weights) + 16 GB (KV) = 156 GB
cards_needed = ceil(156 / 80) = 2 H100s (with tensor parallelism)
```

With TP=2: weights = 70 GB/GPU, KV = 8 GB/GPU (KV sharded), activations ~= 2 GB/GPU. Total per GPU ~= 80 GB -- fits exactly on 80 GB H100s.

---

### Question 2
**Prefill FLOPs for 128K-token sequence: LLaMA-3 70B, d_model=8192, 64 attention heads.**

**Attention FLOPs (dominant term):**
For T=128K tokens, d_k=128:
```
attention_FLOPs = 2 x T^2 x d_k x num_heads x num_layers
                = 2 x (131072)^2 x 128 x 64 x 32
                = 2 x 1.718e10 x 128 x 64 x 32
                = 2 x 1.718e10 x 262144
                = 9.01e15 FLOPs = 9.01 PFLOPs
```

**Estimate prefill time on H100 (1,979 TFLOPS BF16):**
```
t_prefill = 9.01e15 / 1.979e12 = 4,553 s ??? 
```

This can't be right -- attention on H100 is memory-bandwidth-bound at this scale. Let me recalculate with the correct formula:

Actually, T^2 is: (128,000)^2 = 1.638e10. With 64 heads x 128 d_k = 8,192 = d_model, and 32 layers:
```
FLOPs = 2 x T x T x d_model x layers
      = 2 x 128000 x 128000 x 8192 x 32
```

Hmm, standard formula for multi-head attention is:
QK^T: 2 * T * T * d_model FLOPs per layer (T queries, each dotted with T keys at d_model total)
Wait, per head: 2 * T * T * d_k. Across 64 heads: 2 * T^2 * d_k * 64 = 2 * T^2 * d_model.

```
per_layer = 2 x (128000)^2 x 8192 = 2 x 1.638e10 x 8192 = 2.685e14 FLOPs
total_attention = 2.685e14 x 32 layers = 8.59e15 FLOPs
```

At H100 1,979 TFLOPS:
```
t = 8.59e15 / 1.979e12 = 4,341 s
```

This is clearly not realistic -- H100 is memory-bandwidth-bound at these scales, and FlashAttention tiles the computation. The true prefill time with FlashAttention is ~10-30 seconds for 128K tokens on 2 H100s in practice, dominated by HBM bandwidth rather than compute. The roofline model applies: actual throughput is min(compute_bound, bandwidth_bound).

For exam purposes: compute ~= 8.6 PFLOPs for attention alone. At H100 peak BF16 performance of 1.979 PFLOPS per GPU: minimum 8.6/1.979 ~= 4.3 seconds per H100, ~= 2.2 seconds on 2 H100s (TP=2). Real systems achieve 30-60s for 128K due to memory bandwidth limits on the KV cache access pattern.

---

### Question 3
**YaRN RoPE interpolation factor and perplexity degradation:**

**Interpolation factor lambda:**
YaRN extends context from L_base to L_target by scaling the RoPE frequencies:
```
lambda = L_target / L_base = 128K / 8K = 16
```

This means RoPE positional frequencies are divided by 16, "stretching" the positional encoding space to accommodate 16x more positions before they repeat.

**Perplexity degradation:**
- Models fine-tuned with YaRN (RoPE scaling applied during extended fine-tuning on long documents) typically show 0-3% perplexity increase vs natively long-context trained models at the target length.
- Models that only apply YaRN at inference time (no fine-tuning) show 10-20% perplexity degradation at 128K tokens vs 8K tokens, because the base model's attention patterns were never trained on distant position pairs.
- The "needle in a haystack" retrieval accuracy at distances > 32K positions degrades significantly without fine-tuning at the extended context length.

---

### Question 4
**Sliding window W=4096. Document passage at position 60K in 128K context. Would SW attention find it?**

**No.** Sliding window attention at position t can only attend to positions [t-W, t]. For a query at position 127,999 (end of context) and a passage at position 60,000:
```
distance = 127,999 - 60,000 = 67,999 tokens
```

67,999 >> W=4,096. The query at the end of the document **cannot attend to** the passage at position 60K.

**Why this fails for RAG retrieval:** The entire premise of retrieving from a long document fails when the relevant passage is outside the sliding window. The model at each position can only "see" the last 4,096 tokens -- everything before that is invisible.

**Solutions:** Use landmark attention (special tokens every L positions that attend to all prior landmarks), or hierarchical attention with global tokens, or simply use RAG with a retriever to extract the relevant 4K passage and include it in a short context.

---

### Question 5
**Kimi Moon-Cache: GPU=2 GB, CPU=128 GB, SSD=2 TB. 1M-token sequence: how does the hierarchical manager work?**

**KV cache for 1M tokens at LLaMA-3 8B (32 layers, 8 GQA heads, d_k=128, BF16):**
```
bytes_per_token = 2 x 32 x 8 x 128 x 2 = 131,072 bytes = 128 KB
total = 1,000,000 x 128 KB = 128 GB
```

GPU (2 GB) holds: 2,000,000 / 128,000 = ~15,625 tokens worth of KV blocks.
CPU (128 GB) holds: ~1,000,000 tokens.
SSD (2 TB) holds: the rest (overflow, archived blocks).

**Eviction decisions (GPU -> CPU):**
The GPU holds the most recently accessed KV blocks (those needed for the current decode step's attention window). When a decode step needs a block at position P:
- If P is in GPU: cache hit, zero latency.
- If P is in CPU: copy 128 KB from CPU DRAM to GPU HBM (~0.06 ms at 2 TB/s DRAM bandwidth).
- If P is on SSD: prefetch from SSD (~500 ms) -- must be done speculatively BEFORE the decode step that needs it.

**Eviction policy:** LRU (Least Recently Used) within each tier. For long-context decoding with sliding window attention, only the last W tokens' blocks are needed at each step -- blocks outside the window are evicted to CPU, then SSD, as the generation progresses.

**Prefetch strategy:** Moon-Cache uses predictive prefetching -- the system analyzes the attention pattern (landmarks at every L tokens) to predict which SSD blocks will be needed K decode steps ahead, and initiates prefetch K steps early. This hides the 500 ms SSD latency behind concurrent decode computation.

