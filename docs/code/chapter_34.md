# Chapter 34: DeepSeek — MLA, MoE, FP8 — Companion Code

## Python — `deepseek_demo.py`

```python
"""
deepseek_demo.py — Chapter 34: DeepSeek — MLA, MoE, and FP8 at Scale

Demonstrates:
  1. MLA KV cache size comparison (standard GQA vs. MLA)
  2. MoE routing simulation with load-balance analysis
  3. FP8 quantization error model
  4. Hardware budget calculator for DeepSeek-V3
  5. Throughput model: MoE vs dense at same active params

Run: python deepseek_demo.py
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
def section(t: str) -> None:
    print(f"\n{'─'*65}\n  {t}\n{'─'*65}")

# ──────────────────────────────────────────────────────────────────────────────
# §1  Model Specs
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class AttentionSpec:
    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    # MLA-specific
    is_mla: bool = False
    d_latent: int = 0       # c dimension
    d_rope: int = 0         # decoupled RoPE dim

MODELS = {
    "Llama-3.1-8B":  AttentionSpec("Llama-3.1-8B",  32, 8,   128),
    "Llama-3.1-70B": AttentionSpec("Llama-3.1-70B", 80, 8,   128),
    "DeepSeek-V2":   AttentionSpec("DeepSeek-V2",   60, 128, 128,
                                    is_mla=True, d_latent=512, d_rope=64),
    "DeepSeek-V3":   AttentionSpec("DeepSeek-V3",   61, 128, 128,
                                    is_mla=True, d_latent=512, d_rope=64),
}

def kv_bytes_per_token(spec: AttentionSpec, dtype_bytes: int = 2) -> int:
    if spec.is_mla:
        # Cache: latent c + decoupled RoPE keys
        return spec.n_layers * (spec.d_latent + spec.d_rope) * dtype_bytes
    else:
        return 2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * dtype_bytes

def demo_mla_comparison():
    section("MLA KV Cache Size Comparison")
    print(f"\n  {'Model':<20} {'KV/token':>12} {'32K ctx (GB)':>14} {'128K ctx (GB)':>15}")
    print(f"  {'─'*20} {'─'*12} {'─'*14} {'─'*15}")

    for name, spec in MODELS.items():
        kv = kv_bytes_per_token(spec)
        ctx32  = kv * 32_000  / 1e9
        ctx128 = kv * 128_000 / 1e9
        print(f"  {name:<20} {kv:>10,}B {ctx32:>13.1f} {ctx128:>14.1f}")

    # Assert MLA is dramatically smaller
    kv_llama70 = kv_bytes_per_token(MODELS["Llama-3.1-70B"])
    kv_dsv3    = kv_bytes_per_token(MODELS["DeepSeek-V3"])
    ratio = kv_llama70 / kv_dsv3
    print(f"\n  DeepSeek-V3 KV cache is {ratio:.1f}× smaller than Llama-3.1-70B")
    assert ratio > 3.0, f"Expected >3× compression, got {ratio:.2f}×"
    print(f"  [ASSERT] MLA compression ratio > 3×: {ratio:.1f}× ✓")

# ──────────────────────────────────────────────────────────────────────────────
# §2  MoE Routing Simulation
# ──────────────────────────────────────────────────────────────────────────────
import random

@dataclass
class MoEConfig:
    n_experts: int    # total routed experts
    top_k: int        # experts selected per token
    n_layers: int

def simulate_moe_routing(cfg: MoEConfig, n_tokens: int, seed: int = 42) -> Dict:
    """Simulate MoE routing and compute load statistics."""
    random.seed(seed)
    expert_counts = [0] * cfg.n_experts
    total_activations = n_tokens * cfg.n_layers * cfg.top_k

    # Simulate: each token/layer picks top_k experts (uniform random for simulation)
    for _ in range(n_tokens * cfg.n_layers):
        chosen = random.sample(range(cfg.n_experts), cfg.top_k)
        for e in chosen:
            expert_counts[e] += 1

    ideal = total_activations / cfg.n_experts
    max_load  = max(expert_counts)
    min_load  = min(expert_counts)
    load_imbalance = max_load / ideal  # >1 means overloaded

    return {
        "expert_counts": expert_counts,
        "ideal_per_expert": ideal,
        "max_load": max_load,
        "min_load": min_load,
        "load_imbalance": load_imbalance,
        "active_fraction": sum(1 for c in expert_counts if c > 0) / cfg.n_experts,
    }

def demo_moe_routing():
    section("MoE Routing — Load Balance Analysis")

    configs = {
        "DeepSeek-V3 (256 exp, top-8)": MoEConfig(256, 8, 61),
        "Mixtral (8 exp, top-2)":        MoEConfig(8,   2, 32),
        "Small MoE (16 exp, top-2)":     MoEConfig(16,  2, 8),
    }

    N_TOKENS = 1000
    print(f"\n  Simulating {N_TOKENS} tokens through each MoE config:\n")
    print(f"  {'Config':<35} {'Experts':>8} {'Top-k':>7} {'Imbalance':>11} {'Active%':>9}")
    print(f"  {'─'*35} {'─'*8} {'─'*7} {'─'*11} {'─'*9}")

    for name, cfg in configs.items():
        stats = simulate_moe_routing(cfg, N_TOKENS)
        print(f"  {name:<35} {cfg.n_experts:>8} {cfg.top_k:>7} "
              f"{stats['load_imbalance']:>10.2f}× {stats['active_fraction']*100:>8.1f}%")

    # With more experts, load should be more balanced (closer to 1.0)
    stats256 = simulate_moe_routing(MoEConfig(256, 2, 61), N_TOKENS)
    stats8   = simulate_moe_routing(MoEConfig(8,   2, 32), N_TOKENS)
    # Both should be close to 1.0 with uniform routing; just check they're valid
    assert 0.5 < stats256["load_imbalance"] < 3.0
    print(f"\n  [ASSERT] Load imbalance within expected range: ✓")

    # Expert utilization at token=32 (batch effect)
    print(f"\n  Expert utilization vs batch size (DeepSeek-V3 config):")
    cfg = MoEConfig(256, 2, 1)  # single layer for clarity
    print(f"  {'Tokens':>8} {'Experts used':>14} {'Utilization':>13}")
    print(f"  {'─'*8} {'─'*14} {'─'*13}")
    for n_tok in [1, 4, 16, 64, 256, 1024]:
        stats = simulate_moe_routing(cfg, n_tok)
        n_used = sum(1 for c in stats["expert_counts"] if c > 0)
        print(f"  {n_tok:>8} {n_used:>14} {n_used/256*100:>12.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# §3  FP8 Quantization Error Model
# ──────────────────────────────────────────────────────────────────────────────
def fp8_e4m3_quantize(x: float, scale: float = 1.0) -> float:
    """Simulate FP8 E4M3 quantization (3 mantissa bits → precision ~12.5%)."""
    if x == 0.0: return 0.0
    x_scaled = x / scale
    # FP8 E4M3: 3 mantissa bits → round to nearest 1/8 in normalized range
    max_fp8 = 448.0
    x_clipped = max(-max_fp8, min(max_fp8, x_scaled))
    if abs(x_clipped) < 2**-9: return 0.0  # subnormal threshold
    # Quantize mantissa to 3 bits
    exp = math.floor(math.log2(abs(x_clipped)))
    mantissa = abs(x_clipped) / (2**exp)
    mantissa_q = round(mantissa * 8) / 8  # 3-bit mantissa
    result = math.copysign(mantissa_q * (2**exp), x_clipped)
    return result * scale

def demo_fp8_model():
    section("FP8 Quantization Error Model")

    import random
    random.seed(42)
    test_values = [random.gauss(0, 1) for _ in range(1000)]

    errors_fp8 = []
    for v in test_values:
        q = fp8_e4m3_quantize(v, scale=1.0)
        if v != 0:
            errors_fp8.append(abs(q - v) / abs(v))

    mean_rel_err = sum(errors_fp8) / len(errors_fp8)
    max_rel_err  = max(errors_fp8)

    print(f"\n  FP8 E4M3 quantization of 1000 Gaussian values:")
    print(f"  Mean relative error: {mean_rel_err*100:.2f}%")
    print(f"  Max relative error:  {max_rel_err*100:.2f}%")
    print(f"  (BF16 mean rel err:  ~0.4%, FP32 ~0.0001%)")

    # BF16: 7 mantissa bits → ~0.4% relative error
    # FP8 E4M3: 3 mantissa bits → ~1.5% expected
    assert mean_rel_err < 0.05, f"FP8 mean error {mean_rel_err:.4f} seems too high"
    print(f"\n  [ASSERT] FP8 quantization error within expected range: ✓")

    # Throughput model: FP8 on H100
    print(f"\n  H100 theoretical throughput by precision:")
    print(f"  {'Precision':<12} {'TFLOPS':>10} {'Speedup vs FP32':>18}")
    print(f"  {'─'*12} {'─'*10} {'─'*18}")
    for prec, tflops in [("FP32", 67), ("BF16", 989), ("FP16", 989), ("FP8", 1979)]:
        print(f"  {prec:<12} {tflops:>10,} {tflops/67:>17.1f}×")

# ──────────────────────────────────────────────────────────────────────────────
# §4  Hardware Budget Calculator
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class HardwareConfig:
    name: str
    vram_gb: float
    n_gpus: int

def hardware_budget(model_params_b: float, dtype_bytes: float,
                    context_k: int, kv_bytes_tok: int,
                    batch_size: int, hw: HardwareConfig) -> dict:
    total_vram = hw.vram_gb * hw.n_gpus
    model_gb   = model_params_b * 1e9 * dtype_bytes / 1e9
    kv_gb      = kv_bytes_tok * context_k * 1000 * batch_size / 1e9
    available  = total_vram - model_gb
    fits       = available > kv_gb
    return {
        "total_vram_gb": total_vram,
        "model_gb": model_gb,
        "kv_gb": kv_gb,
        "headroom_gb": available - kv_gb,
        "fits": fits,
    }

def demo_hardware_budget():
    section("Hardware Budget for DeepSeek-V3 Serving")

    configs = [
        HardwareConfig("8× H100 (80GB)",  80,  8),
        HardwareConfig("8× H200 (141GB)", 141, 8),
        HardwareConfig("16× H100 (80GB)",  80, 16),
    ]

    # DeepSeek-V3: 671B params
    # FP8 = 1 byte/param, INT4 = 0.5 bytes/param
    kv_per_tok = kv_bytes_per_token(MODELS["DeepSeek-V3"])

    print(f"\n  DeepSeek-V3 (671B params, MLA KV={kv_per_tok} bytes/token)\n")
    print(f"  {'Config':<22} {'Weights':>8} {'Precision':>10} {'KV 32K×8':>10} {'Fits?':>7} {'Headroom':>10}")
    print(f"  {'─'*22} {'─'*8} {'─'*10} {'─'*10} {'─'*7} {'─'*10}")

    for hw in configs:
        for dtype, dtype_str in [(1.0, "FP8"), (0.5, "INT4")]:
            b = hardware_budget(671, dtype, 32, kv_per_tok, 8, hw)
            fits_str = "✓" if b["fits"] else "✗"
            print(f"  {hw.name:<22} {b['model_gb']:>7.0f}G {dtype_str:>10} "
                  f"{b['kv_gb']:>9.1f}G {fits_str:>7} {b['headroom_gb']:>8.1f}G")
        print()

    # Verify 8× H200 with FP8 fits
    b_check = hardware_budget(671, 1.0, 32, kv_per_tok, 8,
                               HardwareConfig("8× H200", 141, 8))
    assert b_check["fits"], "8× H200 FP8 should fit DeepSeek-V3"
    print(f"  [ASSERT] 8× H200 (FP8) fits DeepSeek-V3 with {b_check['headroom_gb']:.0f} GB headroom: ✓")

# ──────────────────────────────────────────────────────────────────────────────
# §5  MoE vs Dense Throughput
# ──────────────────────────────────────────────────────────────────────────────
def demo_moe_vs_dense_throughput():
    section("MoE vs Dense Throughput (bandwidth-bound decode)")

    # BW roof: tps = bw_tb_s × 1e12 / (active_params × dtype_bytes)
    h100_bw = 3.35e12  # bytes/s

    models = [
        ("Llama-3.1-70B (dense)", 70e9,  70e9,  2),   # 70B active
        ("DeepSeek-V3 (MoE)",     671e9, 37e9,  1),   # 671B total, 37B active, FP8
        ("Mixtral 8×7B (MoE)",    47e9,  13e9,  2),   # 47B total, 13B active
        ("Llama-3.1-8B (dense)",  8e9,   8e9,   2),   # 8B active
    ]

    print(f"\n  {'Model':<30} {'Total B':>9} {'Active B':>10} {'BW roof':>10} {'vs 70B':>8}")
    print(f"  {'─'*30} {'─'*9} {'─'*10} {'─'*10} {'─'*8}")

    tps_70b = None
    for name, total, active, dtype in models:
        bw_eff = h100_bw * 0.85  # 85% efficiency
        tps = bw_eff / (active * dtype)
        if "70B" in name and "dense" in name: tps_70b = tps
        ratio = tps / tps_70b if tps_70b else 1.0
        print(f"  {name:<30} {total/1e9:>8.0f}B {active/1e9:>9.0f}B {tps:>9,.0f} {ratio:>7.1f}×")

    # Assert DeepSeek MoE faster per decode than 70B dense
    tps_dsv3 = h100_bw * 0.85 / (37e9 * 1)   # FP8
    tps_llama = h100_bw * 0.85 / (70e9 * 2)  # BF16
    assert tps_dsv3 > tps_llama, "MoE 37B active should be faster than 70B dense"
    print(f"\n  [ASSERT] DeepSeek-V3 decode throughput > Llama 70B: "
          f"{tps_dsv3:,.0f} > {tps_llama:,.0f} tok/s ✓")

# ──────────────────────────────────────────────────────────────────────────────
def main():
    bar = "=" * 65
    print(f"\n{bar}\n  Chapter 34 — DeepSeek: MLA, MoE, and FP8 (Python)\n{bar}")
    demo_mla_comparison()
    demo_moe_routing()
    demo_fp8_model()
    demo_hardware_budget()
    demo_moe_vs_dense_throughput()
    print(f"\n{bar}\n  All demos complete.\n{bar}\n")

if __name__ == "__main__":
    main()

```



## C++ — `deepseek_demo.cpp`

```bash
# Compile
g++ -std=c++17 -O2 -o deepseek_demo deepseek_demo.cpp -lm
# Run
./deepseek_demo
```

```cpp
/*
 * deepseek_demo.cpp — Chapter 34: DeepSeek — MLA, MoE, and FP8 at Scale
 *
 * Demonstrates (mirrors deepseek_demo.py):
 *   1. MLA KV cache size comparison (standard GQA vs. MLA)
 *   2. MoE routing simulation with load-balance analysis
 *   3. FP8 quantization error model
 *   4. Hardware budget calculator for DeepSeek-V3
 *   5. MoE vs Dense throughput (bandwidth-bound decode)
 *
 * Compile: g++ -std=c++17 -O2 -o deepseek_demo deepseek_demo.cpp -lm
 * Run:     ./deepseek_demo
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

static void section(const char* title) {
    printf("\n%s\n  %s\n%s\n",
           "─────────────────────────────────────────────────────────────────",
           title,
           "─────────────────────────────────────────────────────────────────");
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  Model Specs
// ─────────────────────────────────────────────────────────────────────────────

struct AttentionSpec {
    const char* name;
    int   n_layers;
    int   n_kv_heads;
    int   head_dim;
    bool  is_mla;
    int   d_latent;  // MLA latent dimension c
    int   d_rope;    // MLA decoupled RoPE dimension
};

static AttentionSpec MODELS[] = {
    {"Llama-3.1-8B",  32, 8,   128, false, 0,   0},
    {"Llama-3.1-70B", 80, 8,   128, false, 0,   0},
    {"DeepSeek-V2",   60, 128, 128, true,  512, 64},
    {"DeepSeek-V3",   61, 128, 128, true,  512, 64},
};
static const int N_MODELS = 4;

static long kv_bytes_per_token(const AttentionSpec& s, int dtype_bytes = 2) {
    if (s.is_mla)
        return (long)s.n_layers * (s.d_latent + s.d_rope) * dtype_bytes;
    return 2L * s.n_layers * s.n_kv_heads * s.head_dim * dtype_bytes;
}

static void demo_mla_comparison() {
    section("MLA KV Cache Size Comparison");
    printf("\n  %-20s %12s %14s %15s\n",
           "Model", "KV/token", "32K ctx (GB)", "128K ctx (GB)");
    printf("  %-20s %12s %14s %15s\n",
           "────────────────────", "────────────", "──────────────", "───────────────");

    for (int i = 0; i < N_MODELS; ++i) {
        const auto& s = MODELS[i];
        long kv = kv_bytes_per_token(s);
        double ctx32  = kv * 32000.0  / 1e9;
        double ctx128 = kv * 128000.0 / 1e9;
        printf("  %-20s %10ldB %13.1f %14.1f\n",
               s.name, kv, ctx32, ctx128);
    }

    long kv_llama70 = kv_bytes_per_token(MODELS[1]);  // Llama-3.1-70B
    long kv_dsv3    = kv_bytes_per_token(MODELS[3]);  // DeepSeek-V3
    double ratio = (double)kv_llama70 / kv_dsv3;
    printf("\n  DeepSeek-V3 KV cache is %.1fx smaller than Llama-3.1-70B\n", ratio);
    assert(ratio > 3.0 && "Expected >3x MLA compression");
    printf("  [ASSERT] MLA compression ratio > 3x: %.1fx ✓\n", ratio);
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  MoE Routing Simulation
// ─────────────────────────────────────────────────────────────────────────────

struct MoEConfig {
    int n_experts;
    int top_k;
    int n_layers;
};

struct MoEStats {
    double load_imbalance;
    double active_fraction;
    int    max_load;
    int    min_load;
    double ideal_per_expert;
};

static MoEStats simulate_moe_routing(const MoEConfig& cfg, int n_tokens, int seed = 42) {
    std::mt19937 rng(seed);
    std::vector<int> counts(cfg.n_experts, 0);
    long total = (long)n_tokens * cfg.n_layers * cfg.top_k;

    for (int i = 0; i < n_tokens * cfg.n_layers; ++i) {
        // Sample top_k unique experts
        std::vector<int> experts(cfg.n_experts);
        std::iota(experts.begin(), experts.end(), 0);
        std::shuffle(experts.begin(), experts.end(), rng);
        for (int k = 0; k < cfg.top_k; ++k)
            counts[experts[k]]++;
    }

    double ideal = (double)total / cfg.n_experts;
    int max_load = *std::max_element(counts.begin(), counts.end());
    int min_load = *std::min_element(counts.begin(), counts.end());
    int n_active = (int)std::count_if(counts.begin(), counts.end(), [](int c){ return c > 0; });

    return {max_load / ideal, (double)n_active / cfg.n_experts,
            max_load, min_load, ideal};
}

static void demo_moe_routing() {
    section("MoE Routing — Load Balance Analysis");

    struct Cfg { const char* name; MoEConfig cfg; };
    Cfg configs[] = {
        {"DeepSeek-V3 (256 exp, top-8)", {256, 8, 61}},
        {"Mixtral (8 exp, top-2)",        {8,   2, 32}},
        {"Small MoE (16 exp, top-2)",     {16,  2, 8}},
    };
    const int N = 3;
    const int N_TOKENS = 1000;

    printf("\n  Simulating %d tokens through each MoE config:\n\n", N_TOKENS);
    printf("  %-35s %8s %7s %11s %9s\n",
           "Config", "Experts", "Top-k", "Imbalance", "Active%");
    printf("  %-35s %8s %7s %11s %9s\n",
           "───────────────────────────────────", "────────", "───────", "───────────", "─────────");

    for (int i = 0; i < N; ++i) {
        auto stats = simulate_moe_routing(configs[i].cfg, N_TOKENS);
        printf("  %-35s %8d %7d %10.2fx %8.1f%%\n",
               configs[i].name, configs[i].cfg.n_experts, configs[i].cfg.top_k,
               stats.load_imbalance, stats.active_fraction * 100.0);
    }

    auto s = simulate_moe_routing({256, 2, 61}, N_TOKENS);
    assert(s.load_imbalance > 0.5 && s.load_imbalance < 3.0);
    printf("\n  [ASSERT] Load imbalance within expected range: ✓\n");

    printf("\n  Expert utilization vs batch size (DeepSeek-V3 config):\n");
    printf("  %8s %14s %13s\n", "Tokens", "Experts used", "Utilization");
    printf("  %8s %14s %13s\n", "────────", "──────────────", "─────────────");
    for (int n : {1, 4, 16, 64, 256, 1024}) {
        auto st = simulate_moe_routing({256, 2, 1}, n);
        int n_used = (int)std::round(st.active_fraction * 256);
        printf("  %8d %14d %12.1f%%\n", n, n_used, n_used / 256.0 * 100.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  FP8 Quantization Error Model
// ─────────────────────────────────────────────────────────────────────────────

static double fp8_e4m3_quantize(double x, double scale = 1.0) {
    if (x == 0.0) return 0.0;
    double xs = x / scale;
    const double max_fp8 = 448.0;
    double xc = std::max(-max_fp8, std::min(max_fp8, xs));
    if (std::abs(xc) < std::pow(2.0, -9)) return 0.0;
    int exp = (int)std::floor(std::log2(std::abs(xc)));
    double mantissa = std::abs(xc) / std::pow(2.0, exp);
    double mq = std::round(mantissa * 8.0) / 8.0;
    double result = std::copysign(mq * std::pow(2.0, exp), xc);
    return result * scale;
}

static void demo_fp8_model() {
    section("FP8 Quantization Error Model");

    std::mt19937_64 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> test_vals(1000);
    for (auto& v : test_vals) v = normal(rng);

    double mean_err = 0.0, max_err = 0.0;
    int cnt = 0;
    for (double v : test_vals) {
        double q = fp8_e4m3_quantize(v, 1.0);
        if (v != 0.0) {
            double rel = std::abs(q - v) / std::abs(v);
            mean_err += rel;
            max_err   = std::max(max_err, rel);
            ++cnt;
        }
    }
    mean_err /= cnt;

    printf("\n  FP8 E4M3 quantization of 1000 Gaussian values:\n");
    printf("  Mean relative error: %.2f%%\n", mean_err * 100);
    printf("  Max relative error:  %.2f%%\n", max_err * 100);
    printf("  (BF16 mean rel err:  ~0.4%%, FP32 ~0.0001%%)\n");

    assert(mean_err < 0.05);
    printf("\n  [ASSERT] FP8 quantization error within expected range: ✓\n");

    printf("\n  H100 theoretical throughput by precision:\n");
    printf("  %-12s %10s %18s\n", "Precision", "TFLOPS", "Speedup vs FP32");
    printf("  %-12s %10s %18s\n", "────────────", "──────────", "──────────────────");
    struct Prec { const char* n; int tf; };
    Prec precs[] = {{"FP32", 67}, {"BF16", 989}, {"FP16", 989}, {"FP8", 1979}};
    for (auto& p : precs)
        printf("  %-12s %10d %17.1fx\n", p.n, p.tf, (double)p.tf / 67.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Hardware Budget Calculator
// ─────────────────────────────────────────────────────────────────────────────

struct HardwareConfig {
    const char* name;
    double vram_gb;
    int    n_gpus;
};

struct BudgetResult {
    double total_vram_gb;
    double model_gb;
    double kv_gb;
    double headroom_gb;
    bool   fits;
};

static BudgetResult hardware_budget(double model_params_b, double dtype_bytes,
                                    int context_k, long kv_bytes_tok, int batch,
                                    const HardwareConfig& hw) {
    double total_vram = hw.vram_gb * hw.n_gpus;
    double model_gb   = model_params_b * 1e9 * dtype_bytes / 1e9;
    double kv_gb      = kv_bytes_tok * (double)context_k * 1000.0 * batch / 1e9;
    double headroom   = total_vram - model_gb - kv_gb;
    return {total_vram, model_gb, kv_gb, headroom, headroom >= 0};
}

static void demo_hardware_budget() {
    section("Hardware Budget for DeepSeek-V3 Serving");

    HardwareConfig configs[] = {
        {"8x H100 (80GB)",  80,  8},
        {"8x H200 (141GB)", 141, 8},
        {"16x H100 (80GB)", 80,  16},
    };

    long kv_per_tok = kv_bytes_per_token(MODELS[3]);  // DeepSeek-V3
    printf("\n  DeepSeek-V3 (671B params, MLA KV=%ld bytes/token)\n\n", kv_per_tok);
    printf("  %-22s %8s %10s %10s %7s %10s\n",
           "Config", "Weights", "Precision", "KV 32K×8", "Fits?", "Headroom");
    printf("  %-22s %8s %10s %10s %7s %10s\n",
           "──────────────────────", "────────", "──────────", "──────────", "───────", "──────────");

    for (auto& hw : configs) {
        struct DType { double b; const char* n; };
        DType dtypes[] = {{1.0, "FP8"}, {0.5, "INT4"}};
        for (auto& dt : dtypes) {
            auto b = hardware_budget(671.0, dt.b, 32, kv_per_tok, 8, hw);
            printf("  %-22s %7.0fG %10s %9.1fG %7s %8.1fG\n",
                   hw.name, b.model_gb, dt.n, b.kv_gb,
                   b.fits ? "✓" : "✗", b.headroom_gb);
        }
        printf("\n");
    }

    auto b_check = hardware_budget(671.0, 1.0, 32, kv_per_tok, 8,
                                   {"8x H200", 141, 8});
    assert(b_check.fits && "8x H200 FP8 should fit DeepSeek-V3");
    printf("  [ASSERT] 8x H200 (FP8) fits DeepSeek-V3 with %.0f GB headroom: ✓\n",
           b_check.headroom_gb);
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  MoE vs Dense Throughput
// ─────────────────────────────────────────────────────────────────────────────

static void demo_moe_vs_dense_throughput() {
    section("MoE vs Dense Throughput (bandwidth-bound decode)");

    const double h100_bw = 3.35e12;  // bytes/s

    struct ModelEntry {
        const char* name;
        double total_b;
        double active_b;
        int    dtype;
    };
    ModelEntry models[] = {
        {"Llama-3.1-70B (dense)", 70e9,  70e9,  2},
        {"DeepSeek-V3 (MoE)",     671e9, 37e9,  1},
        {"Mixtral 8x7B (MoE)",    47e9,  13e9,  2},
        {"Llama-3.1-8B (dense)",  8e9,   8e9,   2},
    };

    printf("\n  %-30s %9s %10s %10s %8s\n",
           "Model", "Total B", "Active B", "BW roof", "vs 70B");
    printf("  %-30s %9s %10s %10s %8s\n",
           "──────────────────────────────", "─────────", "──────────", "──────────", "────────");

    double tps_70b = -1.0;
    for (auto& m : models) {
        double bw_eff = h100_bw * 0.85;
        double tps = bw_eff / (m.active_b * m.dtype);
        if (tps_70b < 0) tps_70b = tps;
        double ratio = tps / tps_70b;
        printf("  %-30s %8.0fB %9.0fB %10.0f %7.1fx\n",
               m.name, m.total_b / 1e9, m.active_b / 1e9, tps, ratio);
    }

    double tps_dsv3  = h100_bw * 0.85 / (37e9 * 1);
    double tps_llama = h100_bw * 0.85 / (70e9 * 2);
    assert(tps_dsv3 > tps_llama);
    printf("\n  [ASSERT] DeepSeek-V3 decode throughput > Llama 70B: "
           "%.0f > %.0f tok/s ✓\n", tps_dsv3, tps_llama);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    const char* bar =
        "═════════════════════════════════════════════════════════════════";
    printf("\n%s\n  Chapter 34 — DeepSeek: MLA, MoE, and FP8 (C++)\n%s\n",
           bar, bar);

    demo_mla_comparison();
    demo_moe_routing();
    demo_fp8_model();
    demo_hardware_budget();
    demo_moe_vs_dense_throughput();

    printf("\n%s\n  All demos complete.\n%s\n\n", bar, bar);
    return 0;
}
```
