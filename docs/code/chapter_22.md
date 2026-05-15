# Chapter 22: LoRA Serving — Companion Code

## Python — `lora_demo.py`

```python
#!/usr/bin/env python3
"""
Chapter 22 — Python: Multi-LoRA Serving Demo
=============================================
Demonstrates:
  - LoRA memory budget calculation
  - Adapter registry with per-request routing
  - vLLM LoRARequest integration (mock, no GPU required)
  - A/B test routing logic
"""

from dataclasses import dataclass
from typing import Optional
import random, hashlib, json

# ─── Memory budget calculator ─────────────────────────────────────────────────

PRECISION_BYTES = {"FP32": 4.0, "BF16": 2.0, "FP16": 2.0, "INT8": 1.0}

@dataclass
class LoRAConfig:
    rank:       int           # r
    alpha:      int           # alpha (scaling = alpha / rank)
    target_modules: list[str] # e.g., ["q_proj", "v_proj", "k_proj", "o_proj"]
    dtype:      str = "BF16"

@dataclass
class ModelArchitecture:
    name:       str
    n_layers:   int
    d_model:    int           # hidden dim
    d_ff:       int           # MLP intermediate dim

MODELS = {
    "Llama-3.1-8B":  ModelArchitecture("Llama 3.1 8B",  32, 4096, 14336),
    "Llama-3.3-70B": ModelArchitecture("Llama 3.3 70B", 80, 8192, 28672),
    "Qwen2.5-7B":    ModelArchitecture("Qwen 2.5 7B",   28, 3584, 18944),
}

MODULE_DIMS = {
    # Each maps to (d_out, d_in) for W ∈ ℝ^{d_out × d_in}
    "q_proj": lambda d, _: (d, d),
    "k_proj": lambda d, _: (d, d),   # simplified; GQA would be smaller
    "v_proj": lambda d, _: (d, d),
    "o_proj": lambda d, _: (d, d),
    "gate_proj": lambda d, f: (f, d),
    "up_proj":   lambda d, f: (f, d),
    "down_proj": lambda d, f: (d, f),
}

def adapter_memory_bytes(
    arch: ModelArchitecture,
    config: LoRAConfig,
) -> dict:
    """
    Calculate total LoRA adapter memory.
    Returns breakdown dict with bytes per module and total.
    """
    bpe = PRECISION_BYTES[config.dtype]
    r   = config.rank
    breakdown = {}

    per_layer_bytes = 0
    for mod in config.target_modules:
        if mod not in MODULE_DIMS:
            continue
        d_out, d_in = MODULE_DIMS[mod](arch.d_model, arch.d_ff)
        A_params = r * d_in
        B_params = d_out * r
        mod_bytes = (A_params + B_params) * bpe
        breakdown[mod] = mod_bytes
        per_layer_bytes += mod_bytes

    total_bytes = per_layer_bytes * arch.n_layers
    breakdown["_per_layer"] = per_layer_bytes
    breakdown["_total"]     = total_bytes
    return breakdown


def print_adapter_budgets():
    print("\n" + "=" * 70)
    print("  LoRA Adapter Memory Budget")
    print("=" * 70)

    configs = [
        LoRAConfig(rank=8,  alpha=8,  target_modules=["q_proj","v_proj"]),
        LoRAConfig(rank=16, alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"]),
        LoRAConfig(rank=64, alpha=64, target_modules=["q_proj","k_proj","v_proj","o_proj",
                                                       "gate_proj","up_proj","down_proj"]),
    ]
    mod_labels = ["q+v", "attn×4", "all×7"]

    print(f"  {'Model':<20} {'Modules':<10} {'Rank':>6} {'Adapter MB':>12} "
          f"{'Adapters/H100':>15}")
    print(f"  {'-'*20} {'-'*10} {'-'*6} {'-'*12} {'-'*15}")

    for arch in MODELS.values():
        for cfg, label in zip(configs, mod_labels):
            mem = adapter_memory_bytes(arch, cfg)
            total_mb = mem["_total"] / 1e6
            # H100 80GB: 80×0.90 − weight_gb − 1.5 activations
            base_gb  = {"Llama 3.1 8B": 16, "Llama 3.3 70B": 140,
                        "Qwen 2.5 7B": 14}[arch.name]
            avail_mb = (80*0.90 - base_gb - 1.5) * 1024
            n_adapters = int(avail_mb / total_mb) if total_mb > 0 else 0
            print(f"  {arch.name:<20} {label:<10} {cfg.rank:>6} "
                  f"{total_mb:>11.1f}M {n_adapters:>15,}")
        print()


# ─── Adapter registry ─────────────────────────────────────────────────────────

@dataclass
class AdapterSpec:
    name:        str
    lora_int_id: int
    lora_path:   str
    rank:        int
    description: str
    traffic_pct: float = 100.0   # percentage of traffic routed here

class AdapterRegistry:
    def __init__(self):
        self._adapters: dict[str, AdapterSpec] = {}
        self._routing_groups: dict[str, list[AdapterSpec]] = {}

    def register(self, spec: AdapterSpec) -> None:
        self._adapters[spec.name] = spec

    def get(self, name: str) -> AdapterSpec:
        return self._adapters[name]

    def register_ab_group(self, group_name: str, specs: list[AdapterSpec]) -> None:
        """Register an A/B group. Traffic split by spec.traffic_pct (must sum to 100)."""
        assert abs(sum(s.traffic_pct for s in specs) - 100.0) < 0.01, \
            "Traffic percentages must sum to 100"
        self._routing_groups[group_name] = specs
        for spec in specs:
            self.register(spec)

    def route_ab(self, group_name: str, user_id: str) -> AdapterSpec:
        """Deterministically route user to an adapter in the A/B group."""
        specs = self._routing_groups[group_name]
        bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        cumulative = 0.0
        for spec in specs:
            cumulative += spec.traffic_pct
            if bucket < cumulative:
                return spec
        return specs[-1]   # fallback


def demo_registry():
    print("\n" + "=" * 70)
    print("  Adapter Registry and A/B Routing Demo")
    print("=" * 70)

    registry = AdapterRegistry()
    registry.register_ab_group("legal-routing", [
        AdapterSpec("legal-v2-stable",    lora_int_id=1,
                    lora_path="/adapters/legal-v2",    rank=16,
                    description="Legal EN stable",     traffic_pct=90.0),
        AdapterSpec("legal-v3-candidate", lora_int_id=10,
                    lora_path="/adapters/legal-v3",    rank=16,
                    description="Legal EN candidate",  traffic_pct=10.0),
    ])

    # Simulate 20 users
    counts = {"legal-v2-stable": 0, "legal-v3-candidate": 0}
    for i in range(200):
        user_id = f"user_{i:04d}"
        spec    = registry.route_ab("legal-routing", user_id)
        counts[spec.name] += 1

    print(f"  Simulated routing for 200 users:")
    for name, count in counts.items():
        pct = count / 200 * 100
        bar = "█" * (count // 5)
        print(f"    {name:<30} {count:>4}  ({pct:4.1f}%)  {bar}")


if __name__ == "__main__":
    print("\nChapter 22 — LoRA Serving Demo")
    print("=" * 70)
    print_adapter_budgets()
    demo_registry()

```

## C++ — `lora_demo.cpp`

```cpp
/**
 * Chapter 22 — C++ Companion: LoRA Memory and Merge Analysis
 * ===========================================================
 * Demonstrates:
 *   - LoRA A and B matrix size calculation for any (d_out, d_in, r)
 *   - Full adapter memory budget for Llama 3.1 8B
 *   - Conceptual merge walkthrough (B·A computed for a tiny example)
 *   - Compression ratio analysis
 *
 * Build:  g++ -std=c++17 -O2 lora_demo.cpp -o lora_demo
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct LoRAConfig {
    int rank;
    int alpha;
};

struct ModuleSpec {
    std::string name;
    int d_out;
    int d_in;
};

struct ModelArch {
    std::string name;
    int n_layers;
    int d_model;
    int d_ff;
};

// ─────────────────────────────────────────────────────────────────────────────
// Memory budget calculations
// ─────────────────────────────────────────────────────────────────────────────

struct AdapterBreakdown {
    std::string module_name;
    long long A_params;
    long long B_params;
    double bytes_mb;
};

static double BPE_BF16 = 2.0;

AdapterBreakdown lora_module_memory(const ModuleSpec& mod, const LoRAConfig& cfg) {
    long long A = static_cast<long long>(cfg.rank) * mod.d_in;
    long long B = static_cast<long long>(mod.d_out) * cfg.rank;
    double bytes_mb = (A + B) * BPE_BF16 / 1e6;
    return {mod.name, A, B, bytes_mb};
}

double full_adapter_mb(const ModelArch& arch, const LoRAConfig& cfg,
                        const std::vector<ModuleSpec>& per_layer_mods) {
    double per_layer = 0.0;
    for (const auto& mod : per_layer_mods) {
        auto bk = lora_module_memory(mod, cfg);
        per_layer += bk.bytes_mb;
    }
    return per_layer * arch.n_layers;
}

static void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "  " << title << "\n"
              << std::string(70, '=') << "\n";
}

void demo_memory_budget() {
    print_section("LoRA Adapter Memory Budget");

    // Llama 3.1 8B
    ModelArch llama8b{"Llama 3.1 8B", 32, 4096, 14336};

    // Q, K, V, O projections (all 4096×4096)
    std::vector<ModuleSpec> attn_mods = {
        {"q_proj", 4096, 4096},
        {"k_proj", 4096, 4096},
        {"v_proj", 4096, 4096},
        {"o_proj", 4096, 4096},
    };
    // + MLP
    std::vector<ModuleSpec> all_mods = attn_mods;
    all_mods.push_back({"gate_proj", 14336, 4096});
    all_mods.push_back({"up_proj",   14336, 4096});
    all_mods.push_back({"down_proj",  4096, 14336});

    struct Scenario {
        std::string label;
        LoRAConfig  cfg;
        std::vector<ModuleSpec> mods;
    };

    std::vector<Scenario> scenarios;
    {
        Scenario s0; s0.label = "r=8,  q+v only"; s0.cfg = {8, 8};
        s0.mods = {{"q_proj",4096,4096},{"v_proj",4096,4096}};
        scenarios.push_back(std::move(s0));
    }
    {
        Scenario s1; s1.label = "r=16, attn x4"; s1.cfg = {16, 16};
        s1.mods = attn_mods;
        scenarios.push_back(std::move(s1));
    }
    {
        Scenario s2; s2.label = "r=64, all  x7"; s2.cfg = {64, 64};
        s2.mods = all_mods;
        scenarios.push_back(std::move(s2));
    }

    double base_gb = 16.0;   // Llama 3.1 8B BF16
    double avail_mb = (80.0 * 0.90 - base_gb - 1.5) * 1024.0;

    std::cout << "  " << std::left  << std::setw(22) << "Scenario"
              << std::right << std::setw(14) << "Adapter (MB)"
              << std::right << std::setw(18) << "Adapters/H100" << "\n";
    std::cout << "  " << std::string(22,'-') << " " << std::string(14,'-')
              << " " << std::string(18,'-') << "\n";

    for (const auto& s : scenarios) {
        double mb = full_adapter_mb(llama8b, s.cfg, s.mods);
        int n_adapters = static_cast<int>(avail_mb / mb);
        std::cout << "  " << std::left  << std::setw(22) << s.label
                  << std::right << std::setw(14) << std::fixed << std::setprecision(1) << mb
                  << std::right << std::setw(18) << n_adapters << "\n";
    }

    std::cout << "\n  Base model: " << base_gb << " GB  |  H100: 80 GB  "
              << "|  Available for adapters: "
              << std::fixed << std::setprecision(1) << avail_mb / 1024.0 << " GB\n";
}

void demo_lora_merge_math() {
    print_section("LoRA Merge: B·A Computation (Toy Example)");

    // Tiny example: d_out=4, d_in=4, r=2
    // A: 2×4, B: 4×2
    const int D = 4, R = 2;

    // A (r × d_in)
    double A[R][D] = {
        {0.1,  0.2, -0.1,  0.3},
        {0.0, -0.1,  0.4, -0.2},
    };
    // B (d_out × r)
    double B[D][R] = {
        { 0.5, -0.1},
        {-0.3,  0.4},
        { 0.2,  0.0},
        { 0.1, -0.2},
    };

    // Compute ΔW = B·A  (D × D result)
    double delta_W[D][D] = {};
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            for (int k = 0; k < R; ++k)
                delta_W[i][j] += B[i][k] * A[k][j];

    const double alpha = 16.0, r = R;
    double scale = alpha / r;   // 16/2 = 8

    std::cout << "  d_out=" << D << "  d_in=" << D << "  r=" << R
              << "  alpha=" << (int)alpha << "  scale=alpha/r=" << scale << "\n\n";

    std::cout << "  B · A  (ΔW before scaling):\n  ";
    for (int i = 0; i < D; ++i) {
        std::cout << (i == 0 ? "[ " : "  ");
        for (int j = 0; j < D; ++j)
            std::cout << std::setw(7) << std::fixed << std::setprecision(3) << delta_W[i][j];
        std::cout << (i == D-1 ? " ]" : "") << "\n  ";
    }

    std::cout << "\n  ΔW × scale (" << scale << "):\n  ";
    for (int i = 0; i < D; ++i) {
        std::cout << (i == 0 ? "[ " : "  ");
        for (int j = 0; j < D; ++j)
            std::cout << std::setw(7) << std::fixed << std::setprecision(3)
                      << delta_W[i][j] * scale;
        std::cout << (i == D-1 ? " ]" : "") << "\n  ";
    }

    std::cout << "\n  W_merged = W_base + ΔW × scale\n"
              << "  During inference: x @ W_merged.T  (zero overhead)\n"
              << "  vLLM injection:   x @ W_base.T + x @ A.T @ B.T × scale\n";

    // Frobenius norm of delta
    double frob = 0.0;
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            frob += (delta_W[i][j] * scale) * (delta_W[i][j] * scale);
    frob = std::sqrt(frob);
    std::cout << "\n  ||ΔW × scale||_F = " << std::fixed << std::setprecision(4)
              << frob << "  (Frobenius norm — size of the weight update)\n";
}

void demo_compression_ratio() {
    print_section("Compression Ratio: LoRA vs Full Fine-Tune");

    struct Scenario {
        std::string model;
        int n_layers, d_model, r;
    };
    std::vector<Scenario> scenarios = {
        {"Llama 3.1 8B",  32, 4096,  16},
        {"Llama 3.3 70B", 80, 8192,  16},
        {"Llama 3.3 70B", 80, 8192,  64},
    };

    std::cout << "  " << std::left  << std::setw(20) << "Model"
              << std::right << std::setw(6)  << "rank"
              << std::right << std::setw(14) << "Full FT (GB)"
              << std::right << std::setw(14) << "LoRA (MB)"
              << std::right << std::setw(12) << "Ratio" << "\n";
    std::cout << "  " << std::string(20,'-') << " " << std::string(6,'-')
              << " " << std::string(14,'-') << " " << std::string(14,'-')
              << " " << std::string(12,'-') << "\n";

    for (const auto& s : scenarios) {
        // Attention matrices only (4 per layer)
        long long full_ft_bytes = 4LL * s.n_layers * s.d_model * s.d_model * 2; // BF16
        long long lora_bytes    = 4LL * s.n_layers * 2 * s.r * s.d_model * 2;
        double full_gb = full_ft_bytes / 1e9;
        double lora_mb = lora_bytes    / 1e6;
        double ratio   = full_ft_bytes / (double)lora_bytes;
        std::cout << "  " << std::left  << std::setw(20) << s.model
                  << std::right << std::setw(6)  << s.r
                  << std::right << std::setw(14) << std::fixed << std::setprecision(1) << full_gb
                  << std::right << std::setw(14) << std::setprecision(1) << lora_mb
                  << std::right << std::setw(11) << std::setprecision(0) << ratio << "×\n";
    }
}

int main() {
    std::cout << "\nChapter 22 — LoRA Serving Demo (C++)\n";
    std::cout << std::string(70, '=') << "\n";

    demo_memory_budget();
    demo_lora_merge_math();
    demo_compression_ratio();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  Demo complete.\n";
    std::cout << std::string(70, '=') << "\n";
    return 0;
}

```

