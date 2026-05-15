# Chapter 2: The GPU and CPU Memory Landscapes — Companion Code

## Python — `memory_demo.py`

```python
#!/usr/bin/env python3
"""
Chapter 2 — Companion Code: The GPU and CPU Memory Landscapes
=============================================================
Demonstrates:
  - Hardware bandwidth reference table
  - Weight memory at every precision (FP32 → INT4)
  - KV cache sizing for any model / batch / sequence
  - vLLM three-region HBM budget split
  - llama.cpp partial GPU offload calculation
  - Arithmetic intensity and bandwidth-bound analysis
  - Memory hierarchy latency comparison

No GPU required.  Run: python memory_demo.py
"""

from dataclasses import dataclass
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Hardware reference data
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUSpec:
    name:          str
    vram_gb:       float       # total HBM / VRAM
    bandwidth_tbs: float       # TB/s
    memory_type:   str

@dataclass
class CPUMemSpec:
    name:          str
    bandwidth_gbs: float       # GB/s
    memory_type:   str

GPU_SPECS = [
    GPUSpec("NVIDIA H100 SXM  (80 GB)",  80,  3.35, "HBM3"),
    GPUSpec("NVIDIA H100 PCIe (80 GB)",  80,  2.00, "HBM3"),
    GPUSpec("NVIDIA A100 SXM  (80 GB)",  80,  2.00, "HBM2e"),
    GPUSpec("NVIDIA A100 PCIe (80 GB)",  80,  1.94, "HBM2e"),
    GPUSpec("NVIDIA RTX 4090  (24 GB)",  24,  1.01, "GDDR6X"),
    GPUSpec("NVIDIA A10G      (24 GB)",  24,  0.60, "GDDR6"),
    GPUSpec("Apple M2 Ultra  (192 GB)", 192,  0.80, "Unified LPDDR5"),
    GPUSpec("Apple M3 Max    (128 GB)", 128,  0.40, "Unified LPDDR5"),
    GPUSpec("Apple M2 Max    ( 96 GB)",  96,  0.40, "Unified LPDDR5"),
    GPUSpec("Apple M2 Pro    ( 32 GB)",  32,  0.20, "Unified LPDDR5"),
]

CPU_MEM_SPECS = [
    CPUMemSpec("Server DDR5",       75,   "DDR5 DRAM"),
    CPUMemSpec("Laptop DDR5",       50,   "DDR5 DRAM"),
    CPUMemSpec("PCIe 4.0 ×16",      32,   "PCIe"),
    CPUMemSpec("PCIe 5.0 ×16",      64,   "PCIe"),
    CPUMemSpec("NVMe SSD",          10,   "NVMe"),
]

# Bytes per parameter for each precision
PRECISION_BYTES: dict[str, float] = {
    "FP32":  4.0,
    "FP16":  2.0,
    "BF16":  2.0,
    "INT8":  1.0,
    "FP8":   1.0,
    "INT4":  0.5,    # approximate, includes block metadata
}


# ──────────────────────────────────────────────────────────────────────────────
# §2.1  Bandwidth reference table
# ──────────────────────────────────────────────────────────────────────────────

def print_bandwidth_table():
    print("\n" + "="*70)
    print("  §2.1  Hardware Bandwidth Reference")
    print("="*70)
    print(f"  {'Hardware':<32} {'Type':<18} {'BW (TB/s)':>10}  {'VRAM':>6}")
    print(f"  {'-'*32} {'-'*18} {'-'*10}  {'-'*6}")
    for g in GPU_SPECS:
        bw_str = f"{g.bandwidth_tbs:.2f} TB/s"
        vram_str = f"{g.vram_gb:.0f} GB"
        print(f"  {g.name:<32} {g.memory_type:<18} {bw_str:>10}  {vram_str:>6}")
    print()
    print(f"  {'Memory / Bus':<32} {'Type':<18} {'BW (GB/s)':>10}")
    print(f"  {'-'*32} {'-'*18} {'-'*10}")
    for c in CPU_MEM_SPECS:
        bw_str = f"{c.bandwidth_gbs:.0f} GB/s"
        print(f"  {c.name:<32} {c.memory_type:<18} {bw_str:>10}")
    print()
    # Highlight ratio
    h100_bw  = 3.35 * 1000   # GB/s
    dram_bw  = 75.0
    ratio    = h100_bw / dram_bw
    print(f"  H100 HBM vs DDR5 DRAM bandwidth ratio: {ratio:.0f}×")
    print(f"  Rule of thumb: HBM is ~30–60× faster than CPU DRAM")


# ──────────────────────────────────────────────────────────────────────────────
# §2.2  Weight memory at every precision
# ──────────────────────────────────────────────────────────────────────────────

def weight_memory_gb(params_billions: float, precision: str) -> float:
    """Return weight memory in GB for a model with params_billions parameters."""
    bpw = PRECISION_BYTES[precision]
    return params_billions * 1e9 * bpw / 1e9   # bytes → GB


def print_weight_memory_table():
    print("\n" + "="*70)
    print("  §2.2  Weight Memory at Every Precision")
    print("="*70)

    models = [
        ("Qwen 2.5 Math 1.5B",  1.5),
        ("Llama 3.2 3B",         3.0),
        ("Llama 3.1 8B",         8.0),
        ("Llama 3.3 70B",       70.0),
        ("Llama 3.1 405B",     405.0),
    ]
    precisions = ["FP32", "BF16", "INT8", "INT4"]

    header = f"  {'Model':<24}" + "".join(f"{p:>10}" for p in precisions)
    print(header)
    print(f"  {'-'*24}" + "-"*40)

    for model_name, params_b in models:
        row = f"  {model_name:<24}"
        for prec in precisions:
            gb = weight_memory_gb(params_b, prec)
            row += f"{gb:>9.1f}G"
        print(row)

    print()
    print("  Formula: memory_GB = params_B × 1e9 × bytes_per_param / 1e9")
    print("  Example (Llama 3.1 8B @ BF16): 8 × 2 = 16 GB")

    # Which GPU fits which model at BF16?
    print()
    print("  BF16 fit analysis:")
    for model_name, params_b in models:
        needed = weight_memory_gb(params_b, "BF16")
        fits = [g.name.split("(")[0].strip()
                for g in GPU_SPECS if g.vram_gb >= needed * 1.15]  # 15% headroom
        fits_str = fits[0] if fits else "No single GPU"
        print(f"    {model_name:<24} needs {needed:>6.1f} GB  →  fits on: {fits_str}")


# ──────────────────────────────────────────────────────────────────────────────
# §2.3  KV cache sizing
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name:        str
    n_layers:    int
    n_kv_heads:  int
    head_dim:    int    # d_head = d_model / n_heads (or explicit)

# Representative models
MODEL_CONFIGS = {
    "Llama-3.1-8B":  ModelConfig("Llama 3.1 8B",  32, 8,  128),
    "Llama-3.3-70B": ModelConfig("Llama 3.3 70B", 80, 8,  128),
    "Qwen2.5-1.5B":  ModelConfig("Qwen 2.5 1.5B", 28, 2,  128),
    "Qwen2.5-7B":    ModelConfig("Qwen 2.5 7B",   28, 4,  128),
}

def kv_cache_bytes(
    model: ModelConfig,
    batch_size: int,
    seq_len: int,
    precision: str = "BF16",
) -> int:
    """
    KV cache memory in bytes.

    Formula:
        bytes = 2 (K+V) × n_layers × n_kv_heads × head_dim
                × batch_size × seq_len × bytes_per_element
    """
    bpe = PRECISION_BYTES[precision]
    return int(2 * model.n_layers * model.n_kv_heads * model.head_dim
               * batch_size * seq_len * bpe)


def print_kv_cache_table():
    print("\n" + "="*70)
    print("  §2.3  KV Cache Sizing")
    print("="*70)

    print("  Formula: 2 × n_layers × n_kv_heads × head_dim × B × T × bytes")
    print()

    # Worked example for Llama 3.1 8B
    m = MODEL_CONFIGS["Llama-3.1-8B"]
    B, T = 1, 4096
    bpe = PRECISION_BYTES["BF16"]
    kv_bytes = kv_cache_bytes(m, B, T, "BF16")
    print(f"  Worked example — {m.name}, batch=1, seq=4096, BF16:")
    print(f"    2  (K and V)")
    print(f"    × {m.n_layers}  (layers)")
    print(f"    × {m.n_kv_heads}  (KV heads, GQA)")
    print(f"    × {m.head_dim}  (head_dim)")
    print(f"    × {B}  (batch_size)")
    print(f"    × {T}  (seq_len)")
    print(f"    × {bpe}  (bytes per element, BF16)")
    print(f"    = {kv_bytes:,} bytes = {kv_bytes/1e6:.1f} MB")
    print()

    # Scaling table
    configs = [
        ("Llama-3.1-8B",  "BF16", [(1,    4096), (1,   32768), (32,  4096), (32, 32768)]),
        ("Llama-3.3-70B", "BF16", [(1,    4096), (1,   32768), (32,  4096), (32, 32768)]),
        ("Qwen2.5-1.5B",  "BF16", [(1,    4096), (1,   32768), (32,  4096), (32, 32768)]),
    ]

    print(f"  {'Model':<20} {'Prec':<6} {'B':>4} {'T':>7} {'KV Cache':>12}")
    print(f"  {'-'*20} {'-'*6} {'-'*4} {'-'*7} {'-'*12}")
    for model_key, prec, bs_ts in configs:
        m = MODEL_CONFIGS[model_key]
        for B, T in bs_ts:
            kv = kv_cache_bytes(m, B, T, prec)
            kv_gb = kv / 1e9
            unit = f"{kv_gb:.2f} GB" if kv_gb >= 1 else f"{kv/1e6:.1f} MB"
            print(f"  {m.name:<20} {prec:<6} {B:>4} {T:>7,} {unit:>12}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# §2.4  vLLM three-region HBM budget
# ──────────────────────────────────────────────────────────────────────────────

def vllm_memory_budget(
    gpu: GPUSpec,
    model: ModelConfig,
    model_precision: str = "BF16",
    gpu_memory_utilization: float = 0.90,
    activation_headroom_gb: float = 1.0,
) -> dict:
    """
    Estimate vLLM's three-region HBM allocation:
      Region 1: model weights
      Region 2: activation headroom (forward pass tensors)
      Region 3: KV cache blocks (the remainder)
    """
    total_gb      = gpu.vram_gb * gpu_memory_utilization
    weights_gb    = weight_memory_gb(
        # estimate params from layer/head/dim heuristic
        # actual params = n_layers × (4 × d_model² + 2 × d_model × d_ff)
        # use weight_memory input as rough rule: passed externally
        # For demo we accept model weight GB directly
        0, model_precision   # placeholder — overridden below
    )
    return {}   # see full version below


def vllm_hbm_split(
    total_vram_gb:      float,
    weights_gb:         float,
    gpu_memory_util:    float = 0.90,
    activation_gb:      float = 1.5,
    kv_block_bytes:     int   = 2 * 32 * 8 * 128 * 16 * 2,  # Llama 3.1 8B, 16-token block, BF16
) -> dict:
    """
    Show how vLLM divides HBM.
    kv_block_bytes: bytes per KV block (2 × n_layers × n_kv_heads × head_dim × block_tokens × bpe)
    """
    budget_gb    = total_vram_gb * gpu_memory_util
    remaining_gb = budget_gb - weights_gb - activation_gb
    remaining_gb = max(remaining_gb, 0.0)

    n_blocks     = int(remaining_gb * 1e9 / kv_block_bytes)
    block_tokens = 16    # vLLM default block size
    total_tokens = n_blocks * block_tokens

    return {
        "total_vram_gb":    total_vram_gb,
        "budget_gb":        round(budget_gb, 2),
        "weights_gb":       round(weights_gb, 2),
        "activation_gb":    round(activation_gb, 2),
        "kv_cache_gb":      round(remaining_gb, 2),
        "n_kv_blocks":      n_blocks,
        "max_kv_tokens":    total_tokens,
    }


def print_vllm_budget():
    print("\n" + "="*70)
    print("  §2.4  vLLM Three-Region HBM Budget  (Llama 3.1 8B, BF16)")
    print("="*70)

    # Llama 3.1 8B weights ~16 GB in BF16
    # KV block for Llama 3.1 8B (32 layers, 8 KV heads, head_dim 128, 16 tokens, BF16):
    kv_block = 2 * 32 * 8 * 128 * 16 * 2   # = 2,097,152 bytes = 2 MB

    gpus_to_show = [g for g in GPU_SPECS if "H100" in g.name or "A100" in g.name or "4090" in g.name]

    print(f"  KV block size (16-token block, BF16): {kv_block / 1e6:.2f} MB/block")
    print()
    print(f"  {'GPU':<32} {'Budget':>8} {'Weights':>8} {'KV GB':>8} "
          f"{'Blocks':>8} {'Max Tok':>10}")
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for g in gpus_to_show:
        budget = vllm_hbm_split(g.vram_gb, 16.0, 0.90, 1.5, kv_block)
        print(f"  {g.name:<32} "
              f"{budget['budget_gb']:>7.1f}G "
              f"{budget['weights_gb']:>7.1f}G "
              f"{budget['kv_cache_gb']:>7.1f}G "
              f"{budget['n_kv_blocks']:>8,} "
              f"{budget['max_kv_tokens']:>10,}")

    print()
    print("  Region layout on H100 SXM (80 GB, 90% util = 72 GB budget):")
    b = vllm_hbm_split(80.0, 16.0, 0.90, 1.5, kv_block)
    bar_total = 60
    w_bar = round(b['weights_gb']   / b['budget_gb'] * bar_total)
    a_bar = round(b['activation_gb']/ b['budget_gb'] * bar_total)
    k_bar = bar_total - w_bar - a_bar
    print(f"  [{'W'*w_bar}{'A'*a_bar}{'K'*k_bar}]")
    print(f"   {'Weights':<{w_bar}} {'Act':<{a_bar}} KV Cache")
    print(f"   {b['weights_gb']:.0f} GB        {b['activation_gb']:.1f} GB   {b['kv_cache_gb']:.0f} GB")


# ──────────────────────────────────────────────────────────────────────────────
# §2.5  llama.cpp partial GPU offload
# ──────────────────────────────────────────────────────────────────────────────

def llamacpp_offload(
    model_name:        str,
    total_layers:      int,
    total_weights_gb:  float,
    gpu_vram_gb:       float,
    kv_cache_gb:       float = 0.5,    # approximate KV cache for typical usage
    overhead_gb:       float = 0.3,
) -> dict:
    """
    Compute how many layers llama.cpp can offload to a GPU.
    Layers are offloaded greedily until VRAM is full.
    """
    available_gb   = gpu_vram_gb - kv_cache_gb - overhead_gb
    available_gb   = max(available_gb, 0.0)
    bytes_per_layer = total_weights_gb / total_layers   # GB per layer
    n_offload      = min(int(available_gb / bytes_per_layer), total_layers)
    pct_offload    = n_offload / total_layers * 100

    return {
        "model":            model_name,
        "total_layers":     total_layers,
        "gpu_vram_gb":      gpu_vram_gb,
        "bytes_per_layer":  round(bytes_per_layer, 3),
        "n_offload":        n_offload,
        "pct_offload":      round(pct_offload, 1),
        "remaining_cpu":    total_layers - n_offload,
    }


def print_offload_table():
    print("\n" + "="*70)
    print("  §2.5  llama.cpp Partial GPU Offload  (-ngl flag)")
    print("="*70)

    scenarios = [
        # (model_name,      layers, weight_gb_q4, gpu_name,         vram_gb)
        ("Llama 3.1 8B  Q4_K_M",   32, 4.7,  "RTX 4090 (24 GB)",  24),
        ("Llama 3.1 8B  Q4_K_M",   32, 4.7,  "M2 Pro   (32 GB)",  32),
        ("Llama 3.3 70B Q4_K_M",   80, 40.0, "RTX 4090 (24 GB)",  24),
        ("Llama 3.3 70B Q4_K_M",   80, 40.0, "M2 Ultra (192 GB)", 192),
        ("Qwen 2.5 1.5B Q4_K_M",   28, 1.0,  "M2 Pro   (32 GB)",  32),
    ]

    print(f"  {'Model + Quant':<28} {'GPU':<20} {'-ngl':>5} {'% GPU':>7} {'CPU layers':>11}")
    print(f"  {'-'*28} {'-'*20} {'-'*5} {'-'*7} {'-'*11}")
    for model_name, layers, w_gb, gpu_name, vram_gb in scenarios:
        r = llamacpp_offload(model_name, layers, w_gb, vram_gb)
        flag = r['n_offload'] if r['n_offload'] < layers else layers
        print(f"  {model_name:<28} {gpu_name:<20} {flag:>5} "
              f"{r['pct_offload']:>6.0f}% {r['remaining_cpu']:>11}")

    print()
    print("  llama.cpp flag: -ngl <n>  (number of layers to offload to GPU)")
    print("  -ngl 999 or -ngl 9999 offloads all layers (common shorthand).")
    print("  Partial offload: CPU handles remaining layers — slower, but model fits.")


# ──────────────────────────────────────────────────────────────────────────────
# §2.6  Arithmetic intensity and bandwidth-bound analysis
# ──────────────────────────────────────────────────────────────────────────────

def arithmetic_intensity(
    flops:       float,   # total floating-point operations
    bytes_moved: float,   # bytes read from / written to HBM
) -> float:
    """FLOPs per byte — the arithmetic intensity."""
    return flops / bytes_moved


def is_bandwidth_bound(
    intensity: float,
    hardware_flops_per_s: float,   # peak FLOPS (e.g. H100 = 989e12)
    hardware_bw_bytes_per_s: float # peak bandwidth (e.g. H100 = 3.35e12)
) -> tuple[bool, float]:
    """
    A kernel is bandwidth-bound if its arithmetic intensity is below
    the hardware's ridge point: peak_flops / peak_bandwidth.
    Returns (is_bw_bound, ridge_point).
    """
    ridge = hardware_flops_per_s / hardware_bw_bytes_per_s
    return intensity < ridge, ridge


def print_arithmetic_intensity():
    print("\n" + "="*70)
    print("  §2.6  Arithmetic Intensity — Bandwidth-Bound vs Compute-Bound")
    print("="*70)

    # H100 SXM specs
    h100_bf16_flops = 989e12    # ~989 TFLOP/s BF16 tensor core
    h100_bw         = 3.35e12   # 3.35 TB/s

    ridge = h100_bf16_flops / h100_bw
    print(f"  H100 SXM ridge point = {h100_bf16_flops/1e12:.0f} TFLOP/s ÷ {h100_bw/1e12:.2f} TB/s")
    print(f"                       = {ridge:.0f} FLOPs/byte\n")

    # Decode step: batch=1, d_model=4096, matrix-vector multiply
    d_model = 4096
    # One weight matrix (4096 × 4096) × 1 token vector
    decode_flops  = 2 * d_model * d_model        # 2 × 4096² ≈ 33M
    decode_bytes  = d_model * d_model * 2        # BF16 weight read ≈ 33 MB
    decode_ai     = arithmetic_intensity(decode_flops, decode_bytes)
    is_bw, _      = is_bandwidth_bound(decode_ai, h100_bf16_flops, h100_bw)

    # Prefill step: batch=1, seq=1024
    seq = 1024
    prefill_flops  = 2 * d_model * d_model * seq   # 2 × 4096² × 1024
    prefill_bytes  = d_model * d_model * 2          # weight read (same matrix)
    prefill_ai     = arithmetic_intensity(prefill_flops, prefill_bytes)
    is_bw_pf, _    = is_bandwidth_bound(prefill_ai, h100_bf16_flops, h100_bw)

    ops = [
        ("Decode  (B=1,  T=1,    single weight mat)", decode_ai,  is_bw),
        ("Prefill (B=1,  T=1024, single weight mat)", prefill_ai, is_bw_pf),
    ]

    print(f"  {'Operation':<48} {'AI (F/B)':>10}  {'Bound':>14}")
    print(f"  {'-'*48} {'-'*10}  {'-'*14}")
    for name, ai, bw_bound in ops:
        bound_str = "BANDWIDTH ←" if bw_bound else "compute"
        print(f"  {name:<48} {ai:>10.1f}  {bound_str:>14}")

    print()
    print(f"  Ridge point = {ridge:.0f} FLOPs/byte")
    print("  Decode AI << ridge → bandwidth-bound: throughput limited by HBM speed")
    print("  Prefill AI >> ridge → compute-bound: throughput limited by CUDA cores")
    print()
    print("  This is why batch size matters for decode:")
    print("  Batching B requests reuses weights once but runs B token vectors,")
    print("  multiplying FLOPs by B while bytes_moved stays constant → AI × B.")

    batch_analysis = []
    for B in [1, 8, 32, 128]:
        ai_b = decode_ai * B
        is_bw_b, _ = is_bandwidth_bound(ai_b, h100_bf16_flops, h100_bw)
        batch_analysis.append((B, ai_b, is_bw_b))

    print(f"\n  {'Batch size':>12} {'AI (F/B)':>12} {'Bound':>14}")
    print(f"  {'-'*12} {'-'*12} {'-'*14}")
    for B, ai_b, bw_b in batch_analysis:
        bound_str = "bandwidth" if bw_b else "compute ←"
        print(f"  {B:>12} {ai_b:>12.1f} {bound_str:>14}")


# ──────────────────────────────────────────────────────────────────────────────
# §2.7  Memory hierarchy latency comparison (intuitive)
# ──────────────────────────────────────────────────────────────────────────────

def print_latency_comparison():
    print("\n" + "="*70)
    print("  §2.7  Memory Latency — Human-Scale Analogy")
    print("="*70)

    # Latencies in nanoseconds; scale to human time (1 ns → 1 second)
    levels = [
        ("GPU Registers / L1",  0.5,    "~0.5 ns"),
        ("GPU L2 cache",        5,      "~5 ns"),
        ("GPU HBM (VRAM)",      100,    "~100 ns"),
        ("CPU DRAM",            100,    "~100 ns"),
        ("PCIe GPU↔CPU",        1000,   "~1 µs"),
        ("NVMe SSD",            100_000,"~100 µs"),
        ("Network (LAN)",       1_000_000, "~1 ms"),
    ]

    print("  If 1 nanosecond = 1 second in human time:\n")
    print(f"  {'Level':<22} {'Actual':>10}   {'Human scale':>14}")
    print(f"  {'-'*22} {'-'*10}   {'-'*14}")
    for name, ns, actual in levels:
        seconds_human = ns     # 1 ns → 1 s
        if seconds_human < 60:
            human = f"{seconds_human:.0f} sec"
        elif seconds_human < 3600:
            human = f"{seconds_human/60:.0f} min"
        elif seconds_human < 86400:
            human = f"{seconds_human/3600:.0f} hrs"
        else:
            human = f"{seconds_human/86400:.0f} days"
        print(f"  {name:<22} {actual:>10}   {human:>14}")

    print()
    print("  Accessing CPU DRAM from the GPU (via PCIe) feels like 20 minutes")
    print("  when GPU HBM access feels like 1.7 minutes. This is why KV cache")
    print("  must stay in GPU HBM — not offloaded to CPU RAM — for fast decode.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nChapter 2 — GPU and CPU Memory Landscapes Demo")
    print("=" * 70)

    print_bandwidth_table()
    print_weight_memory_table()
    print_kv_cache_table()
    print_vllm_budget()
    print_offload_table()
    print_arithmetic_intensity()
    print_latency_comparison()

    print("\n" + "=" * 70)
    print("  All demos complete.")
    print("=" * 70)

```

## C++ — `memory_demo.cpp`

```cpp
/**
 * Chapter 2 — Companion Code (C++): The GPU and CPU Memory Landscapes
 * ====================================================================
 * Mirrors memory_demo.py in idiomatic C++17.
 *
 * Demonstrates:
 *   §2.1  Hardware bandwidth reference table (GPU & CPU memory)
 *   §2.2  Weight memory at every precision (FP32 → INT4)
 *   §2.3  KV cache sizing formula and scaling table
 *   §2.4  vLLM three-region HBM budget (weights / activations / KV blocks)
 *   §2.5  llama.cpp partial GPU offload  (-ngl calculation)
 *   §2.6  Arithmetic intensity — bandwidth-bound vs compute-bound
 *   §2.7  Memory latency human-scale analogy
 *
 * Build:  g++ -std=c++17 -O2 memory_demo.cpp -o memory_demo
 * Run:    ./memory_demo
 *
 * No GPU required.  All calculations are analytical.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct GPUSpec {
    std::string name;
    double      vram_gb;        // total HBM / VRAM
    double      bandwidth_tbs;  // TB/s
    std::string memory_type;
};

struct CPUMemSpec {
    std::string name;
    double      bandwidth_gbs;  // GB/s
    std::string memory_type;
};

struct ModelConfig {
    std::string name;
    int         n_layers;
    int         n_kv_heads;
    int         head_dim;       // d_head = d_model / n_heads
};

// ─────────────────────────────────────────────────────────────────────────────
// Hardware reference data
// ─────────────────────────────────────────────────────────────────────────────

static const std::vector<GPUSpec> GPU_SPECS = {
    {"NVIDIA H100 SXM  (80 GB)",  80,  3.35, "HBM3"},
    {"NVIDIA H100 PCIe (80 GB)",  80,  2.00, "HBM3"},
    {"NVIDIA A100 SXM  (80 GB)",  80,  2.00, "HBM2e"},
    {"NVIDIA A100 PCIe (80 GB)",  80,  1.94, "HBM2e"},
    {"NVIDIA RTX 4090  (24 GB)",  24,  1.01, "GDDR6X"},
    {"NVIDIA A10G      (24 GB)",  24,  0.60, "GDDR6"},
    {"Apple M2 Ultra  (192 GB)", 192,  0.80, "Unified LPDDR5"},
    {"Apple M3 Max    (128 GB)", 128,  0.40, "Unified LPDDR5"},
    {"Apple M2 Max    ( 96 GB)",  96,  0.40, "Unified LPDDR5"},
    {"Apple M2 Pro    ( 32 GB)",  32,  0.20, "Unified LPDDR5"},
};

static const std::vector<CPUMemSpec> CPU_MEM_SPECS = {
    {"Server DDR5",  75.0, "DDR5 DRAM"},
    {"Laptop DDR5",  50.0, "DDR5 DRAM"},
    {"PCIe 4.0 x16", 32.0, "PCIe"},
    {"PCIe 5.0 x16", 64.0, "PCIe"},
    {"NVMe SSD",     10.0, "NVMe"},
};

// Bytes per parameter for each precision
static const std::map<std::string, double> PRECISION_BYTES = {
    {"FP32", 4.0},
    {"FP16", 2.0},
    {"BF16", 2.0},
    {"INT8", 1.0},
    {"FP8",  1.0},
    {"INT4", 0.5},   // approximate, includes block metadata
};

static const std::map<std::string, ModelConfig> MODEL_CONFIGS = {
    {"Llama-3.1-8B",  {"Llama 3.1 8B",  32, 8, 128}},
    {"Llama-3.3-70B", {"Llama 3.3 70B", 80, 8, 128}},
    {"Qwen2.5-1.5B",  {"Qwen 2.5 1.5B", 28, 2, 128}},
    {"Qwen2.5-7B",    {"Qwen 2.5 7B",   28, 4, 128}},
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "  " << title << "\n"
              << std::string(70, '=') << "\n";
}

static double bpe(const std::string& precision) {
    return PRECISION_BYTES.at(precision);
}

// ─────────────────────────────────────────────────────────────────────────────
// §2.1  Bandwidth reference table
// ─────────────────────────────────────────────────────────────────────────────

static void print_bandwidth_table() {
    print_section("§2.1  Hardware Bandwidth Reference");

    std::cout << "  " << std::left  << std::setw(32) << "Hardware"
              << std::left  << std::setw(18) << "Type"
              << std::right << std::setw(12) << "BW (TB/s)"
              << std::right << std::setw(8)  << "VRAM" << "\n";
    std::cout << "  " << std::string(32, '-') << " " << std::string(18, '-')
              << " " << std::string(12, '-') << " " << std::string(8, '-') << "\n";

    for (const auto& g : GPU_SPECS) {
        std::cout << "  " << std::left  << std::setw(32) << g.name
                  << std::left  << std::setw(18) << g.memory_type
                  << std::right << std::setw(9)  << std::fixed << std::setprecision(2)
                  << g.bandwidth_tbs << " TB/s"
                  << std::right << std::setw(6)  << static_cast<int>(g.vram_gb) << " GB\n";
    }

    std::cout << "\n";
    std::cout << "  " << std::left  << std::setw(32) << "Memory / Bus"
              << std::left  << std::setw(18) << "Type"
              << std::right << std::setw(12) << "BW (GB/s)" << "\n";
    std::cout << "  " << std::string(32, '-') << " " << std::string(18, '-')
              << " " << std::string(12, '-') << "\n";

    for (const auto& c : CPU_MEM_SPECS) {
        std::cout << "  " << std::left  << std::setw(32) << c.name
                  << std::left  << std::setw(18) << c.memory_type
                  << std::right << std::setw(9)  << std::fixed << std::setprecision(0)
                  << c.bandwidth_gbs << " GB/s\n";
    }

    double h100_bw_gbs = 3.35 * 1000.0;   // TB/s → GB/s
    double dram_bw     = 75.0;
    double ratio       = h100_bw_gbs / dram_bw;
    std::cout << "\n  H100 HBM vs DDR5 DRAM bandwidth ratio: "
              << std::fixed << std::setprecision(0) << ratio << "×\n";
    std::cout << "  Rule of thumb: HBM is ~30–60× faster than CPU DRAM\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// §2.2  Weight memory at every precision
// ─────────────────────────────────────────────────────────────────────────────

static double weight_memory_gb(double params_billions, const std::string& precision) {
    return params_billions * 1e9 * bpe(precision) / 1e9;   // bytes → GB
}

static void print_weight_memory_table() {
    print_section("§2.2  Weight Memory at Every Precision");

    const std::vector<std::pair<std::string, double>> models = {
        {"Qwen 2.5 Math 1.5B",  1.5},
        {"Llama 3.2 3B",         3.0},
        {"Llama 3.1 8B",         8.0},
        {"Llama 3.3 70B",       70.0},
        {"Llama 3.1 405B",     405.0},
    };
    const std::vector<std::string> precisions = {"FP32", "BF16", "INT8", "INT4"};

    std::cout << "  " << std::left << std::setw(24) << "Model";
    for (const auto& p : precisions)
        std::cout << std::right << std::setw(10) << p;
    std::cout << "\n  " << std::string(24, '-') << std::string(40, '-') << "\n";

    for (const auto& [mname, params_b] : models) {
        std::cout << "  " << std::left << std::setw(24) << mname;
        for (const auto& prec : precisions) {
            double gb = weight_memory_gb(params_b, prec);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << gb << "G";
            std::cout << std::right << std::setw(10) << oss.str();
        }
        std::cout << "\n";
    }

    std::cout << "\n  Formula: memory_GB = params_B × 1e9 × bytes_per_param / 1e9\n";
    std::cout << "  Example (Llama 3.1 8B @ BF16): 8 × 2 = 16 GB\n\n";
    std::cout << "  BF16 fit analysis:\n";
    for (const auto& [mname, params_b] : models) {
        double needed = weight_memory_gb(params_b, "BF16");
        std::string fits = "No single GPU";
        for (const auto& g : GPU_SPECS) {
            if (g.vram_gb >= needed * 1.15) {
                // take the first match (GPU_SPECS is ordered by bandwidth desc)
                size_t pos = g.name.find('(');
                fits = (pos != std::string::npos)
                     ? g.name.substr(0, pos - 1)
                     : g.name;
                break;
            }
        }
        std::cout << "    " << std::left << std::setw(24) << mname
                  << " needs " << std::right << std::setw(6)
                  << std::fixed << std::setprecision(1) << needed
                  << " GB  →  fits on: " << fits << "\n";
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// §2.3  KV cache sizing
// ─────────────────────────────────────────────────────────────────────────────

static long long kv_cache_bytes(
    const ModelConfig& model,
    int    batch_size,
    int    seq_len,
    const std::string& precision = "BF16"
) {
    /*
     * bytes = 2 (K+V) × n_layers × n_kv_heads × head_dim
     *         × batch_size × seq_len × bytes_per_element
     */
    double b = bpe(precision);
    return static_cast<long long>(
        2LL * model.n_layers * model.n_kv_heads * model.head_dim
        * batch_size * seq_len * b
    );
}

static void print_kv_cache_table() {
    print_section("§2.3  KV Cache Sizing");

    std::cout << "  Formula: 2 × n_layers × n_kv_heads × head_dim × B × T × bytes\n\n";

    // Worked example for Llama 3.1 8B
    const auto& m   = MODEL_CONFIGS.at("Llama-3.1-8B");
    int B = 1, T = 4096;
    double bpe_val  = bpe("BF16");
    long long kv_b  = kv_cache_bytes(m, B, T, "BF16");

    std::cout << "  Worked example — " << m.name << ", batch=1, seq=4096, BF16:\n";
    std::cout << "    2  (K and V)\n";
    std::cout << "    × " << m.n_layers   << "  (layers)\n";
    std::cout << "    × " << m.n_kv_heads << "  (KV heads, GQA)\n";
    std::cout << "    × " << m.head_dim   << "  (head_dim)\n";
    std::cout << "    × " << B            << "  (batch_size)\n";
    std::cout << "    × " << T            << "  (seq_len)\n";
    std::cout << "    × " << std::fixed << std::setprecision(1) << bpe_val
              << "  (bytes per element, BF16)\n";
    std::cout << "    = " << kv_b << " bytes = "
              << std::fixed << std::setprecision(1) << kv_b / 1e6 << " MB\n\n";

    // Scaling table
    struct Row { std::string key; std::string prec; int b; int t; };
    std::vector<Row> rows = {
        {"Llama-3.1-8B",  "BF16", 1,   4096},
        {"Llama-3.1-8B",  "BF16", 1,  32768},
        {"Llama-3.1-8B",  "BF16", 32,  4096},
        {"Llama-3.1-8B",  "BF16", 32, 32768},
        {"Llama-3.3-70B", "BF16", 1,   4096},
        {"Llama-3.3-70B", "BF16", 1,  32768},
        {"Llama-3.3-70B", "BF16", 32,  4096},
        {"Llama-3.3-70B", "BF16", 32, 32768},
        {"Qwen2.5-1.5B",  "BF16", 1,   4096},
        {"Qwen2.5-1.5B",  "BF16", 1,  32768},
        {"Qwen2.5-1.5B",  "BF16", 32,  4096},
        {"Qwen2.5-1.5B",  "BF16", 32, 32768},
    };

    std::cout << "  " << std::left  << std::setw(20) << "Model"
              << std::left  << std::setw(6)  << "Prec"
              << std::right << std::setw(4)  << "B"
              << std::right << std::setw(7)  << "T"
              << std::right << std::setw(12) << "KV Cache" << "\n";
    std::cout << "  " << std::string(20,'-') << " " << std::string(6,'-')
              << " " << std::string(4,'-') << " " << std::string(7,'-')
              << " " << std::string(12,'-') << "\n";

    std::string last_key = "";
    for (const auto& r : rows) {
        if (last_key != "" && last_key != r.key) std::cout << "\n";
        last_key = r.key;
        const auto& mc = MODEL_CONFIGS.at(r.key);
        long long kv = kv_cache_bytes(mc, r.b, r.t, r.prec);
        double kv_gb = kv / 1e9;
        std::ostringstream unit;
        if (kv_gb >= 1.0)
            unit << std::fixed << std::setprecision(2) << kv_gb << " GB";
        else
            unit << std::fixed << std::setprecision(1) << kv / 1e6 << " MB";

        // format T with comma (e.g. 4096 → "4,096", 32768 → "32,768")
        std::string t_str;
        if (r.t >= 1000) {
            std::string raw = std::to_string(r.t);
            int thousands = r.t / 1000;
            int remainder = r.t % 1000;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%d,%03d", thousands, remainder);
            t_str = buf;
        } else {
            t_str = std::to_string(r.t);
        }

        std::cout << "  " << std::left  << std::setw(20) << mc.name
                  << std::left  << std::setw(6)  << r.prec
                  << std::right << std::setw(4)  << r.b
                  << std::right << std::setw(7)  << t_str
                  << std::right << std::setw(12) << unit.str() << "\n";
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// §2.4  vLLM three-region HBM budget
// ─────────────────────────────────────────────────────────────────────────────

struct VLLMBudget {
    double total_vram_gb;
    double budget_gb;
    double weights_gb;
    double activation_gb;
    double kv_cache_gb;
    long long n_kv_blocks;
    long long max_kv_tokens;
};

static VLLMBudget vllm_hbm_split(
    double total_vram_gb,
    double weights_gb,
    double gpu_memory_util   = 0.90,
    double activation_gb     = 1.5,
    long long kv_block_bytes = 2LL * 32 * 8 * 128 * 16 * 2   // Llama 3.1 8B, 16-tok, BF16
) {
    double budget_gb    = total_vram_gb * gpu_memory_util;
    double remaining_gb = budget_gb - weights_gb - activation_gb;
    if (remaining_gb < 0.0) remaining_gb = 0.0;

    int block_tokens    = 16;
    long long n_blocks  = static_cast<long long>(remaining_gb * 1e9) / kv_block_bytes;
    long long max_toks  = n_blocks * block_tokens;

    VLLMBudget b;
    b.total_vram_gb  = total_vram_gb;
    b.budget_gb      = std::round(budget_gb  * 100) / 100;
    b.weights_gb     = std::round(weights_gb * 100) / 100;
    b.activation_gb  = std::round(activation_gb * 100) / 100;
    b.kv_cache_gb    = std::round(remaining_gb * 100) / 100;
    b.n_kv_blocks    = n_blocks;
    b.max_kv_tokens  = max_toks;
    return b;
}

static void print_vllm_budget() {
    print_section("§2.4  vLLM Three-Region HBM Budget  (Llama 3.1 8B, BF16)");

    // Llama 3.1 8B: 32 layers, 8 KV heads, head_dim 128, 16-token block, BF16 (2 bytes)
    long long kv_block = 2LL * 32 * 8 * 128 * 16 * 2;   // = 2,097,152 bytes ≈ 2 MB

    std::cout << "  KV block size (16-token block, BF16): "
              << std::fixed << std::setprecision(2) << kv_block / 1e6 << " MB/block\n\n";

    std::cout << "  " << std::left  << std::setw(32) << "GPU"
              << std::right << std::setw(8)  << "Budget"
              << std::right << std::setw(8)  << "Weights"
              << std::right << std::setw(8)  << "KV GB"
              << std::right << std::setw(8)  << "Blocks"
              << std::right << std::setw(10) << "Max Tok" << "\n";
    std::cout << "  " << std::string(32,'-')
              << " " << std::string(8,'-')
              << " " << std::string(8,'-')
              << " " << std::string(8,'-')
              << " " << std::string(8,'-')
              << " " << std::string(10,'-') << "\n";

    for (const auto& g : GPU_SPECS) {
        if (g.name.find("H100") == std::string::npos &&
            g.name.find("A100") == std::string::npos &&
            g.name.find("4090") == std::string::npos) continue;

        auto b = vllm_hbm_split(g.vram_gb, 16.0, 0.90, 1.5, kv_block);
        std::cout << "  " << std::left  << std::setw(32) << g.name
                  << std::right << std::setw(7)  << std::fixed << std::setprecision(1)
                  << b.budget_gb << "G"
                  << std::right << std::setw(7)  << b.weights_gb  << "G"
                  << std::right << std::setw(7)  << b.kv_cache_gb << "G"
                  << std::right << std::setw(9)  << b.n_kv_blocks
                  << std::right << std::setw(11) << b.max_kv_tokens << "\n";
    }

    // ASCII bar chart for H100 SXM
    auto bh = vllm_hbm_split(80.0, 16.0, 0.90, 1.5, kv_block);
    int bar_total = 60;
    int w_bar = static_cast<int>(std::round(bh.weights_gb    / bh.budget_gb * bar_total));
    int a_bar = static_cast<int>(std::round(bh.activation_gb / bh.budget_gb * bar_total));
    int k_bar = bar_total - w_bar - a_bar;

    auto pad = [](int n) { return std::string(std::max(0, n), ' '); };

    std::cout << "\n  Region layout on H100 SXM (80 GB, 90% util = 72 GB budget):\n";
    std::cout << "  [" << std::string(w_bar, 'W')
                       << std::string(a_bar, 'A')
                       << std::string(k_bar, 'K') << "]\n";
    std::cout << "   Weights" << pad(w_bar - 7)
              << "Act" << pad(a_bar - 2) << "KV Cache\n";
    std::cout << "   " << std::fixed << std::setprecision(0) << bh.weights_gb
              << " GB" << pad(w_bar - 5)
              << bh.activation_gb << " GB"
              << pad(a_bar - 4)
              << bh.kv_cache_gb << " GB\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// §2.5  llama.cpp partial GPU offload
// ─────────────────────────────────────────────────────────────────────────────

struct OffloadResult {
    std::string model;
    int    total_layers;
    double gpu_vram_gb;
    double bytes_per_layer;
    int    n_offload;
    double pct_offload;
    int    remaining_cpu;
};

static OffloadResult llamacpp_offload(
    const std::string& model_name,
    int    total_layers,
    double total_weights_gb,
    double gpu_vram_gb,
    double kv_cache_gb = 0.5,
    double overhead_gb = 0.3
) {
    double available_gb    = std::max(0.0, gpu_vram_gb - kv_cache_gb - overhead_gb);
    double bytes_per_layer = total_weights_gb / total_layers;
    int    n_offload       = std::min(
        static_cast<int>(available_gb / bytes_per_layer), total_layers
    );

    OffloadResult r;
    r.model           = model_name;
    r.total_layers    = total_layers;
    r.gpu_vram_gb     = gpu_vram_gb;
    r.bytes_per_layer = std::round(bytes_per_layer * 1000.0) / 1000.0;
    r.n_offload       = n_offload;
    r.pct_offload     = std::round(n_offload * 1000.0 / total_layers) / 10.0;
    r.remaining_cpu   = total_layers - n_offload;
    return r;
}

static void print_offload_table() {
    print_section("§2.5  llama.cpp Partial GPU Offload  (-ngl flag)");

    struct Scenario {
        std::string model_name;
        int    layers;
        double weight_gb;
        std::string gpu_name;
        double vram_gb;
    };
    std::vector<Scenario> scenarios = {
        {"Llama 3.1 8B  Q4_K_M",   32,  4.7, "RTX 4090 (24 GB)",   24},
        {"Llama 3.1 8B  Q4_K_M",   32,  4.7, "M2 Pro   (32 GB)",   32},
        {"Llama 3.3 70B Q4_K_M",   80, 40.0, "RTX 4090 (24 GB)",   24},
        {"Llama 3.3 70B Q4_K_M",   80, 40.0, "M2 Ultra (192 GB)", 192},
        {"Qwen 2.5 1.5B Q4_K_M",   28,  1.0, "M2 Pro   (32 GB)",   32},
    };

    std::cout << "  " << std::left  << std::setw(28) << "Model + Quant"
              << std::left  << std::setw(20) << "GPU"
              << std::right << std::setw(5)  << "-ngl"
              << std::right << std::setw(7)  << "% GPU"
              << std::right << std::setw(11) << "CPU layers" << "\n";
    std::cout << "  " << std::string(28,'-') << " " << std::string(20,'-')
              << " " << std::string(5,'-') << " " << std::string(7,'-')
              << " " << std::string(11,'-') << "\n";

    for (const auto& s : scenarios) {
        auto r   = llamacpp_offload(s.model_name, s.layers, s.weight_gb, s.vram_gb);
        int flag = r.n_offload;  // (already clamped to total_layers)
        std::cout << "  " << std::left  << std::setw(28) << s.model_name
                  << std::left  << std::setw(20) << s.gpu_name
                  << std::right << std::setw(5)  << flag
                  << std::right << std::setw(6)  << std::fixed << std::setprecision(0)
                  << r.pct_offload << "%"
                  << std::right << std::setw(11) << r.remaining_cpu << "\n";
    }

    std::cout << "\n  llama.cpp flag: -ngl <n>  (number of layers to offload to GPU)\n";
    std::cout << "  -ngl 999 or -ngl 9999 offloads all layers (common shorthand).\n";
    std::cout << "  Partial offload: CPU handles remaining layers — slower, but model fits.\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// §2.6  Arithmetic intensity — bandwidth-bound vs compute-bound
// ─────────────────────────────────────────────────────────────────────────────

static double arithmetic_intensity(double flops, double bytes_moved) {
    return flops / bytes_moved;
}

static std::pair<bool, double> is_bandwidth_bound(
    double intensity,
    double hardware_flops_per_s,
    double hardware_bw_bytes_per_s
) {
    double ridge = hardware_flops_per_s / hardware_bw_bytes_per_s;
    return {intensity < ridge, ridge};
}

static void print_arithmetic_intensity() {
    print_section("§2.6  Arithmetic Intensity — Bandwidth-Bound vs Compute-Bound");

    // H100 SXM
    const double h100_bf16_flops = 989e12;   // ~989 TFLOP/s BF16
    const double h100_bw         = 3.35e12;  // 3.35 TB/s

    auto [_, ridge] = is_bandwidth_bound(0, h100_bf16_flops, h100_bw);

    std::cout << "  H100 SXM ridge point = "
              << std::fixed << std::setprecision(0)
              << h100_bf16_flops / 1e12 << " TFLOP/s  ÷  "
              << h100_bw / 1e12 << " TB/s\n";
    std::cout << "                       = "
              << std::fixed << std::setprecision(0) << ridge << " FLOPs/byte\n\n";

    // Decode step: batch=1, d_model=4096, matrix-vector multiply
    int d_model = 4096;
    double decode_flops  = 2.0 * d_model * d_model;       // ~33M FLOPs
    double decode_bytes  = static_cast<double>(d_model) * d_model * 2.0;  // BF16 weights
    double decode_ai     = arithmetic_intensity(decode_flops, decode_bytes);
    auto   [is_bw_dec, __] = is_bandwidth_bound(decode_ai, h100_bf16_flops, h100_bw);

    // Prefill step: batch=1, seq=1024
    int    seq         = 1024;
    double prefill_flops  = 2.0 * d_model * d_model * seq;   // reuse same weights
    double prefill_bytes  = static_cast<double>(d_model) * d_model * 2.0;
    double prefill_ai     = arithmetic_intensity(prefill_flops, prefill_bytes);
    auto   [is_bw_pf, ___] = is_bandwidth_bound(prefill_ai, h100_bf16_flops, h100_bw);

    struct OpRow { std::string name; double ai; bool bw_bound; };
    std::vector<OpRow> ops = {
        {"Decode  (B=1,  T=1,    single weight mat)", decode_ai,  is_bw_dec},
        {"Prefill (B=1,  T=1024, single weight mat)", prefill_ai, is_bw_pf},
    };

    std::cout << "  " << std::left  << std::setw(48) << "Operation"
              << std::right << std::setw(10) << "AI (F/B)"
              << std::right << std::setw(16) << "Bound" << "\n";
    std::cout << "  " << std::string(48,'-') << " " << std::string(10,'-')
              << " " << std::string(16,'-') << "\n";

    for (const auto& op : ops) {
        std::string bound = op.bw_bound ? "BANDWIDTH ←" : "compute";
        std::cout << "  " << std::left  << std::setw(48) << op.name
                  << std::right << std::setw(10) << std::fixed << std::setprecision(1)
                  << op.ai
                  << std::right << std::setw(16) << bound << "\n";
    }

    std::cout << "\n  Ridge point = " << std::fixed << std::setprecision(0)
              << ridge << " FLOPs/byte\n";
    std::cout << "  Decode AI << ridge → bandwidth-bound: throughput limited by HBM speed\n";
    std::cout << "  Prefill AI >> ridge → compute-bound:  throughput limited by CUDA cores\n\n";
    std::cout << "  This is why batch size matters for decode:\n";
    std::cout << "  Batching B requests reuses weights once but runs B token vectors,\n";
    std::cout << "  multiplying FLOPs by B while bytes_moved stays constant → AI × B.\n";

    std::cout << "\n  " << std::right << std::setw(12) << "Batch size"
              << std::right << std::setw(12) << "AI (F/B)"
              << std::right << std::setw(14) << "Bound" << "\n";
    std::cout << "  " << std::string(12,'-') << " " << std::string(12,'-')
              << " " << std::string(14,'-') << "\n";

    for (int B : {1, 8, 32, 128}) {
        double ai_b = decode_ai * B;
        auto [bw_b, ____] = is_bandwidth_bound(ai_b, h100_bf16_flops, h100_bw);
        std::string bound = bw_b ? "bandwidth" : "compute ←";
        std::cout << "  " << std::right << std::setw(12) << B
                  << std::right << std::setw(12) << std::fixed << std::setprecision(1)
                  << ai_b
                  << std::right << std::setw(14) << bound << "\n";
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// §2.7  Memory hierarchy latency comparison (human-scale)
// ─────────────────────────────────────────────────────────────────────────────

static void print_latency_comparison() {
    print_section("§2.7  Memory Latency — Human-Scale Analogy");

    // Latencies in nanoseconds; scale: 1 ns → 1 second (human time)
    struct LatRow { std::string name; double ns; std::string actual; };
    std::vector<LatRow> levels = {
        {"GPU Registers / L1",  0.5,        "~0.5 ns"},
        {"GPU L2 cache",        5.0,        "~5 ns"},
        {"GPU HBM (VRAM)",      100.0,      "~100 ns"},
        {"CPU DRAM",            100.0,      "~100 ns"},
        {"PCIe GPU↔CPU",        1000.0,     "~1 µs"},
        {"NVMe SSD",            100000.0,   "~100 µs"},
        {"Network (LAN)",       1000000.0,  "~1 ms"},
    };

    std::cout << "  If 1 nanosecond = 1 second in human time:\n\n";
    std::cout << "  " << std::left  << std::setw(22) << "Level"
              << std::right << std::setw(10) << "Actual"
              << std::right << std::setw(16) << "Human scale" << "\n";
    std::cout << "  " << std::string(22,'-') << " " << std::string(10,'-')
              << " " << std::string(16,'-') << "\n";

    for (const auto& lv : levels) {
        double s = lv.ns;   // 1 ns → 1 second
        std::string human;
        if      (s < 60.0)    human = std::to_string(static_cast<int>(s)) + " sec";
        else if (s < 3600.0)  human = std::to_string(static_cast<int>(s / 60))   + " min";
        else if (s < 86400.0) human = std::to_string(static_cast<int>(s / 3600)) + " hrs";
        else                  human = std::to_string(static_cast<int>(s / 86400))+ " days";

        std::cout << "  " << std::left  << std::setw(22) << lv.name
                  << std::right << std::setw(10) << lv.actual
                  << std::right << std::setw(16) << human << "\n";
    }

    std::cout << "\n  Accessing CPU DRAM from the GPU (via PCIe) feels like 20 minutes\n";
    std::cout << "  when GPU HBM access feels like 1.7 minutes. This is why KV cache\n";
    std::cout << "  must stay in GPU HBM — not offloaded to CPU RAM — for fast decode.\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// Sanity assertions
// ─────────────────────────────────────────────────────────────────────────────

static void run_assertions() {
    // Weight memory: Llama 3.1 8B at BF16 = 8B × 2 bytes = 16 GB
    assert(std::abs(weight_memory_gb(8.0, "BF16") - 16.0) < 0.01);

    // KV cache: Llama 3.1 8B, batch=1, seq=4096, BF16
    // 2 × 32 × 8 × 128 × 1 × 4096 × 2 = 536,870,912 bytes ≈ 537 MB
    auto m = MODEL_CONFIGS.at("Llama-3.1-8B");
    long long kv = kv_cache_bytes(m, 1, 4096, "BF16");
    assert(kv == 536870912LL);

    // Ridge point check: H100 989 TFLOP/s ÷ 3.35 TB/s = ~295 FLOPs/byte
    auto [_, ridge] = is_bandwidth_bound(0, 989e12, 3.35e12);
    assert(ridge > 290.0 && ridge < 300.0);

    // Offload: 8B INT4 ~4.7 GB on 24 GB GPU, 32 layers
    auto r = llamacpp_offload("test", 32, 4.7, 24.0);
    assert(r.n_offload == 32);   // fully fits
}


// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    run_assertions();

    std::cout << "\nChapter 2 — GPU and CPU Memory Landscapes Demo (C++)\n";
    std::cout << std::string(70, '=') << "\n";

    print_bandwidth_table();
    print_weight_memory_table();
    print_kv_cache_table();
    print_vllm_budget();
    print_offload_table();
    print_arithmetic_intensity();
    print_latency_comparison();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  All assertions passed.  Demo complete.\n";
    std::cout << std::string(70, '=') << "\n";

    return 0;
}

```

