# Chapter 20: Cost Engineering — Companion Code

## Python — `cost_demo.py`

```python
"""
cost_demo.py — Chapter 20: Cost Engineering — $/Million Tokens

Demonstrates:
  1. $/1M token calculator across hardware × engine × model × precision
  2. GPU pricing model (on-demand / spot / reserved / owned)
  3. utilization sensitivity analysis
  4. Apple M2 Ultra TCO breakeven analysis
  5. $1.2M → $108K waterfall cost reduction
  6. Spot interruption drain simulator
  7. API provider comparison

No external dependencies beyond the Python standard library.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUSpec:
    name: str
    on_demand_hr: float
    spot_discount: float        # fraction off on-demand (0.70 = 70% cheaper)
    reserved_discount: float    # fraction off on-demand (0.38 = 38% cheaper)
    tdp_w: float                # thermal design power (watts)


@dataclass
class ModelBenchmark:
    model: str
    quant: str
    gpu: str
    batch: int
    decode_tps: float
    prefill_tps: float


GPUS = {
    "h100-sxm":   GPUSpec("H100 SXM 80GB",   32.77, 0.70, 0.38, 700),
    "h100-nvl":   GPUSpec("H100 NVL 94GB",   24.00, 0.65, 0.35, 600),
    "a100-80":    GPUSpec("A100 80GB (Lambda)", 2.49,  0.00, 0.00, 400),
    "rtx4090-v":  GPUSpec("RTX 4090 (Vast)",   0.50,  0.40, 0.00, 450),
    "rtx4090-own":GPUSpec("RTX 4090 (Owned)",  0.08,  0.00, 0.00, 450),
    "m2-ultra":   GPUSpec("M2 Ultra (Owned)",  0.046, 0.00, 0.00,  60),
}

# Key benchmarks (from Chapter 17 + roofline estimates)
BENCHMARKS: List[ModelBenchmark] = [
    ModelBenchmark("Llama-3-8B",  "BF16",   "a100-80",    32, 1635, 19870),
    ModelBenchmark("Llama-3-8B",  "BF16",   "h100-sxm",   32, 3200, 40000),
    ModelBenchmark("Llama-3-8B",  "Q4_K_M", "rtx4090-v",   4,  230,  4210),
    ModelBenchmark("Llama-3-8B",  "Q4_K_M", "rtx4090-own", 4,  230,  4210),
    ModelBenchmark("Llama-3-70B", "BF16",   "a100-80",     4,  200,  3200),
    ModelBenchmark("Llama-3-70B", "Q4_K_M", "rtx4090-v",   1,   15,   900),
    ModelBenchmark("Llama-3-70B", "Q4_K_M", "m2-ultra",    2,   45,  1800),
    ModelBenchmark("DeepSeek-R1-8B","BF16", "a100-80",    10,  400,  8000),
]


def cost_per_1m(gpu: GPUSpec, decode_tps: float,
                mode: str = "on_demand") -> float:
    if mode == "spot":
        price = gpu.on_demand_hr * (1 - gpu.spot_discount)
    elif mode == "reserved":
        price = gpu.on_demand_hr * (1 - gpu.reserved_discount)
    else:
        price = gpu.on_demand_hr
    if decode_tps <= 0:
        return float("inf")
    return price / (decode_tps * 3600) * 1_000_000


# ──────────────────────────────────────────────────────────────────────────────
# 1. Full cost matrix
# ──────────────────────────────────────────────────────────────────────────────

def print_cost_matrix():
    print("\n" + "=" * 80)
    print("  $/1M Output Tokens  |  decode-only, sustained utilization")
    print("=" * 80)
    print(f"  {'Model':<20} {'Quant':<8} {'GPU':<22} {'Batch':>5}"
          f" {'dec tps':>8} {'$/1M OD':>9} {'$/1M Spot':>10}")
    print("  " + "-" * 19 + "  " + "-" * 7 + "  " + "-" * 21 + "  " +
          "-" * 4 + "  " + "-" * 7 + "  " + "-" * 8 + "  " + "-" * 9)

    for b in BENCHMARKS:
        gpu = GPUS[b.gpu]
        c_od   = cost_per_1m(gpu, b.decode_tps, "on_demand")
        c_spot = cost_per_1m(gpu, b.decode_tps, "spot")
        spot_str = f"${c_spot:.3f}" if gpu.spot_discount > 0 else "  n/a "
        print(f"  {b.model:<20} {b.quant:<8} {gpu.name:<22} {b.batch:>5}"
              f" {b.decode_tps:>8,.0f} ${c_od:>8.3f} {spot_str:>10}")
    print("=" * 80)


# ──────────────────────────────────────────────────────────────────────────────
# 2. utilization sensitivity
# ──────────────────────────────────────────────────────────────────────────────

def print_utilization_sensitivity():
    gpu   = GPUS["a100-80"]
    # Llama-3-8B BF16 at batch=32 peak
    peak_tps = 1635.0

    print("\n" + "=" * 64)
    print(f"  utilization Sensitivity  |  A100 80GB  |  Llama-3-8B BF16 B=32")
    print("=" * 64)
    print(f"  {'utilization':>13}  {'Eff. tps':>10}  {'$/1M OD':>9}  {'$/1M Spot':>10}  Bar")
    print("  " + "-" * 12 + "  " + "-" * 9 + "  " + "-" * 8 + "  " + "-" * 9 + "  " + "-" * 10)

    for pct in [10, 25, 50, 70, 85, 95, 100]:
        eff = peak_tps * pct / 100
        c_od   = cost_per_1m(gpu, eff, "on_demand")
        c_spot = cost_per_1m(gpu, eff, "spot")
        bar = "█" * (pct // 10)
        flag = " ← typical idle" if pct == 25 else (" ← target" if pct == 85 else "")
        print(f"  {pct:>12}%  {eff:>10,.0f}  ${c_od:>8.3f}  ${c_spot:>9.3f}  {bar}{flag}")

    print("=" * 64)
    c_28 = cost_per_1m(gpu, peak_tps * 0.28)
    c_72 = cost_per_1m(gpu, peak_tps * 0.72)
    print(f"\n  Case study: 28% util → ${c_28:.3f}/1M, after vLLM 72% → ${c_72:.3f}/1M")
    print(f"  utilization improvement alone: {c_28/c_72:.1f}× cost reduction")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Apple M2 Ultra TCO
# ──────────────────────────────────────────────────────────────────────────────

def print_m2_ultra_tco():
    hardware_cost = 5999.0
    years         = 3
    utilization   = 0.50
    electricity_w = 60.0
    kwh_cost      = 0.10

    hours_total = years * 365 * 24 * utilization
    hw_per_hr   = hardware_cost / hours_total
    elec_per_hr = electricity_w / 1000 * kwh_cost
    total_per_hr = hw_per_hr + elec_per_hr

    print("\n" + "=" * 62)
    print("  Apple M2 Ultra TCO  (192 GB unified memory)")
    print("=" * 62)
    print(f"  Hardware cost:       ${hardware_cost:,.0f}")
    print(f"  Amortization:        {years} years at {utilization*100:.0f}% utilization")
    print(f"  Useful hours:        {hours_total:,.0f} h")
    print(f"  Hardware cost/hr:    ${hw_per_hr:.3f}/hr")
    print(f"  Electricity:         {electricity_w:.0f}W at ${kwh_cost}/kWh = ${elec_per_hr:.3f}/hr")
    print(f"  Total cost/hr:       ${total_per_hr:.3f}/hr")
    print()

    for batch, tps, label in [(1, 17, "single user"), (2, 45, "2 users"), (4, 72, "4 users")]:
        c = cost_per_1m(
            GPUSpec("M2 Ultra", total_per_hr, 0, 0, electricity_w),
            tps
        )
        print(f"  Batch={batch} ({label:<12}): {tps:>4} tok/s  → ${c:.2f}/1M tokens")

    # Breakeven vs. Together.ai ($0.88/1M for 70B)
    api_cost_per_1m = 0.88
    api_cost_per_tok = api_cost_per_1m / 1e6
    monthly_cost_hw  = total_per_hr * 24 * 30
    breakeven_tokens = monthly_cost_hw / api_cost_per_tok

    print(f"\n  Monthly hardware cost:    ${monthly_cost_hw:.0f}/month")
    print(f"  API alternative (70B):    ${api_cost_per_1m}/1M tokens (Together.ai)")
    print(f"  Breakeven volume:         {breakeven_tokens/1e6:.0f}M tokens/month")
    print(f"  → Below {breakeven_tokens/1e6:.0f}M tok/mo: cloud API is cheaper")
    print(f"  → Above {breakeven_tokens/1e6:.0f}M tok/mo: M2 Ultra pays for itself")
    print("=" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# 4. $1.2M → $108K cost reduction waterfall
# ──────────────────────────────────────────────────────────────────────────────

def print_cost_waterfall():
    steps = [
        ("Starting point",                        1_200_000,
         "80× A100 on-demand AWS, 28% utilization, custom serving"),
        ("After vLLM + continuous batching",         460_000,
         "utilization 28% → 72%, handles 2.6× traffic, same fleet"),
        ("After prefix caching (73% hit)",           333_000,
         "Prefill load -73%, fleet 80→58 GPUs"),
        ("After chunked prefill + knob tuning",      276_000,
         "p99 TTFT 4.1s→0.9s, tighter SLA, fleet 58→48 GPUs"),
        ("After disaggregation + Lambda Labs",        58_000,
         "8× H100 SXM + 24× H100 NVL @ $2.49/hr"),
        ("After spot decode pool (70%)",              45_000,
         "17 decode workers on spot, checkpoint+drain"),
        ("Final (incl. ops + semantic cache)",       108_000,
         "6 pf + 18 dec GPUs, 73% semantic cache hit, full ops"),
    ]

    print("\n" + "=" * 72)
    print("  Cost Reduction Waterfall: $1.2M → $108K")
    print("=" * 72)
    bar_max = 1_200_000
    bar_w   = 40

    prev = 1_200_000
    for label, cost, note in steps:
        bar_len = int(cost / bar_max * bar_w)
        bar_len = max(1, bar_len)
        reduction = (1 - cost / prev) * 100 if prev > 0 else 0
        arrow = f" ↓{reduction:.0f}%" if reduction > 0 else ""
        bar = "█" * bar_len
        print(f"\n  {label}{arrow}")
        print(f"    ${cost:>10,}/mo  {bar}")
        print(f"    {note}")
        prev = cost

    start = steps[0][1]
    end   = steps[-1][1]
    print(f"\n  Total reduction: ${start:,} → ${end:,} = {start/end:.1f}× cheaper")
    print(f"  Per-token cost:  ${start/(1_635*3600*30*80)*1e6:.2f}/1M → "
          f"${end/(3200*3600*30*24)*1e6:.3f}/1M")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Spot interruption drain simulator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InFlightRequest:
    req_id: str
    output_tokens_remaining: int
    is_reasoning: bool = False


def simulate_spot_drain(requests: List[InFlightRequest],
                         decode_tps: float = 209.0,
                         grace_period_s: int = 120,
                         warn_ahead_s: int   = 120) -> None:
    print("\n" + "=" * 64)
    print(f"  Spot Interruption Drain  |  grace={grace_period_s}s  "
          f"warn_ahead={warn_ahead_s}s")
    print("=" * 64)
    print(f"  In-flight at SIGTERM: {len(requests)}\n")

    drainable = []
    killed    = []
    for r in requests:
        t = r.output_tokens_remaining / decode_tps
        if t <= grace_period_s:
            drainable.append((r, t))
        else:
            killed.append((r, t))

    for r, t in drainable:
        print(f"  [DRAIN] {r.req_id:<18} {r.output_tokens_remaining:>6} tok"
              f"  {t:>6.1f}s  {'(reasoning)' if r.is_reasoning else ''}")
    for r, t in killed:
        print(f"  [KILL ] {r.req_id:<18} {r.output_tokens_remaining:>6} tok"
              f"  {t:>6.1f}s  → client retry")

    drain_time = max((t for _, t in drainable), default=0)
    print(f"\n  Drain completes in: {drain_time:.1f}s")
    print(f"  Requests saved:     {len(drainable)} / {len(requests)}")
    if killed:
        print(f"  Requests retried:   {len(killed)}")
        print(f"  Recommendation: clients must implement retry with backoff")
    else:
        print(f"  All requests complete cleanly within grace period.")
    print("=" * 64)


# ──────────────────────────────────────────────────────────────────────────────
# 6. API provider comparison
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class APIProvider:
    name: str
    model: str
    input_per_1m: float
    output_per_1m: float


PROVIDERS = [
    APIProvider("OpenAI",       "GPT-4o",              2.50,  10.00),
    APIProvider("OpenAI",       "GPT-4o-mini",          0.15,   0.60),
    APIProvider("Anthropic",    "Claude Sonnet 4",      3.00,  15.00),
    APIProvider("Together.ai",  "Llama-3.1-70B-T",     0.88,   0.88),
    APIProvider("Fireworks.ai", "Llama-3.1-8B-I",      0.20,   0.20),
    APIProvider("Groq",         "Llama-3.3-70B-V",     0.59,   0.79),
    APIProvider("Self-hosted",  "Llama-3-8B BF16 H100",0.05,   0.42),
    APIProvider("Self-hosted",  "Llama-3-8B Q4 RTX4090",0.03,  0.35),
]


def print_api_comparison(monthly_input_m: float = 100.0,
                          monthly_output_m: float = 50.0):
    print("\n" + "=" * 74)
    print(f"  API Provider Comparison  |"
          f"  {monthly_input_m:.0f}M input + {monthly_output_m:.0f}M output tokens/month")
    print("=" * 74)
    print(f"  {'Provider':<16} {'Model':<28} {'$/1M out':>9}"
          f" {'Monthly cost':>14}")
    print("  " + "-" * 15 + "  " + "-" * 27 + "  " + "-" * 8 + "  " + "-" * 13)

    for p in PROVIDERS:
        monthly = (p.input_per_1m * monthly_input_m
                   + p.output_per_1m * monthly_output_m)
        print(f"  {p.name:<16} {p.model:<28} ${p.output_per_1m:>8.2f}"
              f" ${monthly:>13,.0f}")

    print("=" * 74)
    print(f"  Input/output ratio: {monthly_input_m:.0f}:{monthly_output_m:.0f}M tokens")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("█" * 64)
    print("  Chapter 20 — Cost Engineering: $/Million Tokens")
    print("█" * 64)

    print_cost_matrix()
    print_utilization_sensitivity()
    print_m2_ultra_tco()
    print_cost_waterfall()
    print_api_comparison()

    # Spot drain simulation
    requests = [
        InFlightRequest("req-chat-001",  120),
        InFlightRequest("req-chat-002",  340),
        InFlightRequest("req-rag-001",  2100),
        InFlightRequest("req-reason-001", 18000, is_reasoning=True),
    ]
    simulate_spot_drain(requests, grace_period_s=120)
    simulate_spot_drain(requests, grace_period_s=600)


if __name__ == "__main__":
    main()

```

## C++ — `cost_demo.cpp`

```cpp
// cost_demo.cpp — Chapter 20: Cost Engineering — $/Million Tokens
//
// Compile:  g++ -std=c++17 -O2 -Wall -o cost_demo cost_demo.cpp
// Run:      ./cost_demo
//
// Covers: cost matrix, utilization sensitivity, M2 Ultra TCO,
//         waterfall reduction, spot drain planner, API comparison.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
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

static std::string fmt(double v, int prec = 2) {
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
    return out;
}

static void print_sep(int w = 62, char c = '=') {
    std::cout << repeat_str(std::string(1, c), w) << "\n";
}

static void print_header(const std::string& t, int w = 62) {
    print_sep(w);
    int pad = std::max(0, (w - static_cast<int>(t.size()) - 2) / 2);
    std::cout << repeat_str(" ", pad) << " " << t << "\n";
    print_sep(w);
}

// ─────────────────────────────────────────────────────────────────────────────
// Data
// ─────────────────────────────────────────────────────────────────────────────

struct GPUSpec {
    std::string name;
    double on_demand_hr;
    double spot_discount;
    double reserved_discount;
    double tdp_w;
};

struct Benchmark {
    std::string model;
    std::string quant;
    std::string gpu_key;
    int    batch;
    double decode_tps;
};

static const std::vector<GPUSpec> GPUS = {
    {"H100 SXM 80GB",      32.77, 0.70, 0.38, 700},
    {"H100 NVL 94GB",      24.00, 0.65, 0.35, 600},
    {"A100 80GB (Lambda)",  2.49, 0.00, 0.00, 400},
    {"RTX 4090 (Vast)",     0.50, 0.40, 0.00, 450},
    {"RTX 4090 (Owned)",    0.08, 0.00, 0.00, 450},
    {"M2 Ultra (Owned)",    0.046,0.00, 0.00,  60},
};

// Index helpers
static const GPUSpec& gpu(int i) { return GPUS[static_cast<size_t>(i)]; }

static const std::vector<Benchmark> BENCHMARKS = {
    {"Llama-3-8B",     "BF16",   "a100", 32, 1635},
    {"Llama-3-8B",     "BF16",   "h100", 32, 3200},
    {"Llama-3-8B",     "Q4_K_M", "rtxv",  4,  230},
    {"Llama-3-8B",     "Q4_K_M", "rtxo",  4,  230},
    {"Llama-3-70B",    "BF16",   "a100",  4,  200},
    {"Llama-3-70B",    "Q4_K_M", "rtxv",  1,   15},
    {"Llama-3-70B",    "Q4_K_M", "m2ul",  2,   45},
    {"DeepSeek-R1-8B", "BF16",   "a100", 10,  400},
};

// Map key → GPUS index
static int gpu_idx(const std::string& k) {
    if (k == "h100")  return 0;
    if (k == "h100n") return 1;
    if (k == "a100")  return 2;
    if (k == "rtxv")  return 3;
    if (k == "rtxo")  return 4;
    return 5;  // m2ul
}

static double cost_per_1m(double price_hr, double decode_tps) {
    if (decode_tps <= 0) return 1e9;
    return price_hr / (decode_tps * 3600.0) * 1e6;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Cost matrix
// ─────────────────────────────────────────────────────────────────────────────

static void print_cost_matrix() {
    print_sep(80);
    std::cout << "  $/1M Output Tokens  |  decode-only, sustained utilization\n";
    print_sep(80);

    std::cout << "  " << std::left  << std::setw(20) << "Model"
              << std::setw(8)  << "Quant"
              << std::setw(22) << "GPU"
              << std::right
              << std::setw(6)  << "Batch"
              << std::setw(9)  << "dec tps"
              << std::setw(10) << "$/1M OD"
              << std::setw(11) << "$/1M Spot"
              << "\n";
    print_sep(80, '-');

    for (const auto& b : BENCHMARKS) {
        int gi = gpu_idx(b.gpu_key);
        const GPUSpec& g = GPUS[static_cast<size_t>(gi)];
        double c_od   = cost_per_1m(g.on_demand_hr, b.decode_tps);
        double c_spot = (g.spot_discount > 0)
            ? cost_per_1m(g.on_demand_hr * (1 - g.spot_discount), b.decode_tps)
            : -1;

        std::string spot_str = (c_spot >= 0) ? ("$" + fmt(c_spot, 3)) : "n/a";

        std::cout << "  " << std::left  << std::setw(20) << b.model
                  << std::setw(8)  << b.quant
                  << std::setw(22) << g.name
                  << std::right
                  << std::setw(6)  << b.batch
                  << std::setw(9)  << fmt_int(static_cast<long long>(b.decode_tps))
                  << std::setw(9)  << ("$" + fmt(c_od, 3))
                  << std::setw(12) << spot_str
                  << "\n";
    }
    print_sep(80);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. utilization sensitivity
// ─────────────────────────────────────────────────────────────────────────────

static void print_utilization_sensitivity() {
    const GPUSpec& g = gpu(2);   // A100 Lambda
    double peak_tps  = 1635.0;

    print_sep(66);
    std::cout << "  utilization Sensitivity  |  A100 80GB  |  Llama-3-8B BF16\n";
    print_sep(66);
    std::cout << "  " << std::right
              << std::setw(13) << "utilization"
              << std::setw(11) << "Eff tps"
              << std::setw(10) << "$/1M OD"
              << "  Bar\n";
    print_sep(66, '-');

    for (int pct : {10, 25, 50, 70, 85, 95, 100}) {
        double eff = peak_tps * pct / 100.0;
        double c   = cost_per_1m(g.on_demand_hr, eff);
        int bar_len = pct / 10;
        std::string bar(static_cast<size_t>(bar_len), '#');
        std::string flag = (pct == 25) ? " <- typical idle" : (pct == 85 ? " <- target" : "");

        std::cout << "  " << std::right
                  << std::setw(12) << (std::to_string(pct) + "%")
                  << std::setw(11) << fmt_int(static_cast<long long>(eff))
                  << std::setw(9)  << ("$" + fmt(c, 3))
                  << "  " << bar << flag << "\n";
    }
    print_sep(66);

    double c28 = cost_per_1m(g.on_demand_hr, peak_tps * 0.28);
    double c72 = cost_per_1m(g.on_demand_hr, peak_tps * 0.72);
    std::cout << "  28% util: $" << fmt(c28, 3) << "/1M  "
              << "72% util: $" << fmt(c72, 3) << "/1M  "
              << "  improvement: " << fmt(c28 / c72, 1) << "x\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. M2 Ultra TCO
// ─────────────────────────────────────────────────────────────────────────────

static void print_m2_ultra_tco() {
    double hw_cost   = 5999.0;
    double years     = 3.0;
    double util      = 0.50;
    double elec_w    = 60.0;
    double kwh       = 0.10;

    double hours     = years * 365 * 24 * util;
    double hw_hr     = hw_cost / hours;
    double elec_hr   = elec_w / 1000.0 * kwh;
    double total_hr  = hw_hr + elec_hr;

    print_sep(62);
    std::cout << "  Apple M2 Ultra TCO  (192 GB unified memory)\n";
    print_sep(62);
    std::cout << "  Hardware cost:      $" << fmt_int(static_cast<long long>(hw_cost)) << "\n";
    std::cout << "  Amortization:       " << static_cast<int>(years) << " years at "
              << static_cast<int>(util * 100) << "% utilization\n";
    std::cout << "  Useful hours:       " << fmt_int(static_cast<long long>(hours)) << " h\n";
    std::cout << "  Hardware cost/hr:   $" << fmt(hw_hr, 3) << "/hr\n";
    std::cout << "  Electricity:        " << static_cast<int>(elec_w) << "W  $"
              << fmt(elec_hr, 4) << "/hr\n";
    std::cout << "  Total cost/hr:      $" << fmt(total_hr, 3) << "/hr\n\n";

    struct BatchPoint { int b; double tps; std::string label; };
    std::vector<BatchPoint> pts = {{1, 17, "single user"}, {2, 45, "2 users"}, {4, 72, "4 users"}};
    for (auto& p : pts) {
        double c = cost_per_1m(total_hr, p.tps);
        std::cout << "  Batch=" << p.b << " (" << std::left << std::setw(11) << (p.label + ")")
                  << std::right << std::setw(4) << p.tps << " tok/s"
                  << "  $" << fmt(c, 2) << "/1M tokens\n";
    }

    double api_per_1m   = 0.88;
    double monthly_hw   = total_hr * 24 * 30;
    double breakeven    = monthly_hw / (api_per_1m / 1e6);

    std::cout << "\n  Monthly hardware cost:  $" << fmt(monthly_hw, 0) << "/month\n";
    std::cout << "  API (Together 70B):     $" << fmt(api_per_1m, 2) << "/1M tokens\n";
    std::cout << "  Breakeven volume:       " << fmt(breakeven / 1e6, 0) << "M tokens/month\n";
    print_sep(62);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Cost waterfall
// ─────────────────────────────────────────────────────────────────────────────

static void print_cost_waterfall() {
    struct Step {
        std::string label;
        long long   cost;
        std::string note;
    };

    std::vector<Step> steps = {
        {"Starting point",                         1'200'000, "80x A100 on-demand, 28% util"},
        {"After vLLM + continuous batching",          460'000, "util 28%->72%, 2.6x traffic"},
        {"After prefix caching (73% hit)",            333'000, "fleet 80->58 GPUs"},
        {"After chunked prefill + knob tuning",       276'000, "p99 TTFT 4.1s->0.9s"},
        {"After disaggregation + Lambda Labs",         58'000, "8x H100 SXM + 24x H100 NVL"},
        {"After spot decode pool (70%)",               45'000, "17 decode workers on spot"},
        {"Final (ops + semantic cache)",              108'000, "6 pf + 18 dec GPUs"},
    };

    print_sep(72);
    std::cout << "  Cost Reduction Waterfall: $1.2M -> $108K\n";
    print_sep(72);

    long long bar_max = 1'200'000;
    int bar_w = 40;
    long long prev = bar_max;

    for (auto& s : steps) {
        int bar_len = static_cast<int>(
            static_cast<double>(s.cost) / bar_max * bar_w);
        bar_len = std::max(1, bar_len);
        double reduction = (1.0 - static_cast<double>(s.cost) / prev) * 100.0;
        std::string arrow = (reduction > 0)
            ? (" -" + fmt(reduction, 0) + "%") : "";

        std::cout << "\n  " << s.label << arrow << "\n";
        std::cout << "    $" << std::setw(11) << fmt_int(s.cost) << "/mo  "
                  << std::string(static_cast<size_t>(bar_len), '#') << "\n";
        std::cout << "    " << s.note << "\n";
        prev = s.cost;
    }

    long long start = steps.front().cost;
    long long end   = steps.back().cost;
    std::cout << "\n  Total: $" << fmt_int(start) << " -> $" << fmt_int(end)
              << "  = " << fmt(static_cast<double>(start) / end, 1) << "x cheaper\n";
    print_sep(72);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Spot drain planner
// ─────────────────────────────────────────────────────────────────────────────

struct DrainReq {
    std::string id;
    int remaining_tokens;
    bool is_reasoning;
};

static void simulate_spot_drain(const std::vector<DrainReq>& reqs,
                                 double decode_tps,
                                 int grace_s) {
    print_sep(64);
    std::cout << "  Spot Interruption Drain  |  grace=" << grace_s << "s\n";
    print_sep(64);
    std::cout << "  In-flight: " << reqs.size() << "\n\n";

    int saved = 0, killed = 0;
    double max_drain = 0;

    for (const auto& r : reqs) {
        double t = r.remaining_tokens / decode_tps;
        max_drain = std::max(max_drain, t);
        std::string tag  = (t <= grace_s) ? "[DRAIN]" : "[KILL ]";
        std::string note = r.is_reasoning ? "  (reasoning)" : "";
        if (t <= grace_s) ++saved; else ++killed;

        std::cout << "  " << tag << " " << std::left << std::setw(18) << r.id
                  << std::right << std::setw(7) << r.remaining_tokens << " tok"
                  << "  " << std::setw(6) << fmt(t, 1) << "s" << note << "\n";
    }

    std::cout << "\n  Drain completes in: " << fmt(max_drain, 1) << "s\n";
    std::cout << "  Saved: " << saved << " / " << reqs.size() << "\n";
    if (killed > 0)
        std::cout << "  Killed: " << killed << " -> client retry\n";
    else
        std::cout << "  All requests drain cleanly.\n";
    print_sep(64);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. API provider comparison
// ─────────────────────────────────────────────────────────────────────────────

static void print_api_comparison() {
    struct Provider {
        std::string vendor;
        std::string model;
        double input_per_1m;
        double output_per_1m;
    };

    std::vector<Provider> providers = {
        {"OpenAI",       "GPT-4o",              2.50, 10.00},
        {"OpenAI",       "GPT-4o-mini",          0.15,  0.60},
        {"Anthropic",    "Claude Sonnet 4",      3.00, 15.00},
        {"Together.ai",  "Llama-3.1-70B-T",     0.88,  0.88},
        {"Fireworks.ai", "Llama-3.1-8B-I",      0.20,  0.20},
        {"Self-hosted",  "Llama-3-8B H100",     0.05,  0.42},
        {"Self-hosted",  "Llama-3-8B Q4 4090",  0.03,  0.35},
    };

    double input_m  = 100.0;
    double output_m =  50.0;

    print_sep(72);
    std::cout << "  API Provider Comparison  |  "
              << input_m << "M in + " << output_m << "M out tok/month\n";
    print_sep(72);
    std::cout << "  " << std::left  << std::setw(14) << "Provider"
              << std::setw(28) << "Model"
              << std::right
              << std::setw(10) << "$/1M out"
              << std::setw(16) << "Monthly cost"
              << "\n";
    print_sep(72, '-');

    for (const auto& p : providers) {
        double monthly = p.input_per_1m * input_m + p.output_per_1m * output_m;
        std::cout << "  " << std::left  << std::setw(14) << p.vendor
                  << std::setw(28) << p.model
                  << std::right
                  << std::setw(9)  << ("$" + fmt(p.output_per_1m, 2))
                  << std::setw(15) << ("$" + fmt_int(static_cast<long long>(monthly)))
                  << "\n";
    }
    print_sep(72);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    print_header("Chapter 20 - Cost Engineering: $/Million Tokens");

    print_cost_matrix();
    print_utilization_sensitivity();
    print_m2_ultra_tco();
    print_cost_waterfall();
    print_api_comparison();

    std::vector<DrainReq> reqs = {
        {"req-chat-001",   120, false},
        {"req-chat-002",   340, false},
        {"req-rag-001",   2100, false},
        {"req-reason-001",18000, true},
    };
    simulate_spot_drain(reqs, 209.0, 120);
    simulate_spot_drain(reqs, 209.0, 600);

    return 0;
}

```

