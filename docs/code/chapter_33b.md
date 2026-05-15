# Chapter 33b — Code: Choosing Your Engine

Companion code for **Chapter 33b: Choosing Your Engine — SGLang, TRT-LLM, MLC-LLM, and Ollama**.

Both listings implement the same eight demos:

1. 5-axis scoring model for engine selection
2. RadixAttention prefix-reuse savings calculator
3. Roofline throughput ceilings per engine
4. TRT-LLM AOT compilation break-even analysis
5. Decision algorithm — flowchart as executable code
6. Structured output throughput model (SGLang vs vLLM)
7. MLC-LLM cross-device comparison
8. Full benchmark matrix — all engines, all configs

---

## Python

```python
"""
engine_comparison_demo.py — Chapter 33b: Choosing Your Engine

Implements:
  Demo 1: 5-axis scoring model for engine selection
  Demo 2: RadixAttention prefix-reuse savings calculator
  Demo 3: Roofline throughput ceiling per engine
  Demo 4: TRT-LLM AOT compilation break-even analysis
  Demo 5: Decision algorithm — flowchart as executable code
  Demo 6: Structured output throughput model (SGLang vs vLLM)
  Demo 7: MLC-LLM cross-device comparison
  Demo 8: Full benchmark matrix — all engines, all configs

Run: python engine_comparison_demo.py
No GPU required — all calculations from first principles.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

SEPARATOR = "─" * 70


# ─────────────────────────────────────────────────────────────────────────────
# Engine and hardware specs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EngineSpec:
    name: str
    # 5 axes, each scored 1–5 (5 = best)
    hardware_flexibility: int   # works on non-NVIDIA hardware
    structured_output:    int   # JSON/grammar constrained gen
    deploy_simplicity:    int   # ease of deployment
    throughput_ceiling:   int   # peak tokens/s on NVIDIA
    quant_support:        int   # breadth of quantization formats
    # Throughput multiplier vs vLLM BF16 baseline (batch=1, 70B)
    tput_multiplier_bf16: float
    tput_multiplier_fp8:  float
    notes: str = ""

    def score(self, weights: Optional[dict] = None) -> float:
        """Weighted average of 5 axes."""
        w = weights or {
            "hardware_flexibility": 1,
            "structured_output":    1,
            "deploy_simplicity":    1,
            "throughput_ceiling":   1,
            "quant_support":        1,
        }
        axes = {
            "hardware_flexibility": self.hardware_flexibility,
            "structured_output":    self.structured_output,
            "deploy_simplicity":    self.deploy_simplicity,
            "throughput_ceiling":   self.throughput_ceiling,
            "quant_support":        self.quant_support,
        }
        total = sum(axes[k] * w[k] for k in axes)
        return total / sum(w.values())


ENGINES = {
    "vLLM BF16":     EngineSpec("vLLM BF16",     5, 3, 4, 3, 4, 1.00, 1.00,
                                "Broad support, PagedAttention, default choice"),
    "vLLM FP8":      EngineSpec("vLLM FP8",       5, 3, 4, 4, 5, 2.00, 2.00,
                                "2× via FP8 weight compression"),
    "SGLang BF16":   EngineSpec("SGLang BF16",    4, 5, 4, 3, 4, 1.20, 1.20,
                                "RadixAttention + structured gen"),
    "SGLang FP8":    EngineSpec("SGLang FP8",     4, 5, 4, 4, 5, 2.30, 2.30,
                                "RadixAttention + FP8"),
    "TRT-LLM FP8":   EngineSpec("TRT-LLM FP8",   2, 2, 2, 5, 5, 2.80, 2.80,
                                "Highest throughput, NVIDIA-only, compile needed"),
    "TRT-LLM FP8+sp":EngineSpec("TRT-LLM FP8+2:4",2,2,2,5,5, 4.00, 4.00,
                                "FP8 + 2:4 sparsity, 4× baseline"),
    "MLC-LLM INT4":  EngineSpec("MLC-LLM INT4",  5, 2, 3, 3, 4, 0.80, 0.80,
                                "Cross-device, browser/mobile capable"),
    "llama.cpp Q4":  EngineSpec("llama.cpp Q4",  5, 2, 4, 2, 3, 0.35, 0.35,
                                "CPU/edge, GGUF ecosystem"),
    "Ollama":        EngineSpec("Ollama",         5, 1, 5, 1, 3, 0.30, 0.30,
                                "Developer UX, wraps llama.cpp"),
}

H100_HBM_BW_GBS  = 3350.0   # GB/s per H100
H100_COST_PER_HR = 28.0      # $/hr
VLLM_BF16_TPS_1xH100_70B = 18.0  # baseline: vLLM BF16, batch=1, 70B, 1×H100


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: 5-Axis Scoring
# ─────────────────────────────────────────────────────────────────────────────

def demo_5axis_scoring():
    print(f"\n{'='*70}")
    print("DEMO 1 — 5-Axis Scoring Model")
    print(f"{'='*70}")

    # Three workload profiles with different weights
    profiles = {
        "Structured output (JSON extraction)": {
            "hardware_flexibility": 1,
            "structured_output":    3,   # highest priority
            "deploy_simplicity":    1,
            "throughput_ceiling":   2,
            "quant_support":        1,
        },
        "Max throughput, stable NVIDIA prod": {
            "hardware_flexibility": 1,
            "structured_output":    1,
            "deploy_simplicity":    1,
            "throughput_ceiling":   6,   # dominant priority
            "quant_support":        2,
        },
        "Developer laptop, easy setup": {
            "hardware_flexibility": 2,
            "structured_output":    1,
            "deploy_simplicity":    4,   # highest priority
            "throughput_ceiling":   1,
            "quant_support":        1,
        },
    }

    for profile_name, weights in profiles.items():
        print(f"\n  Profile: {profile_name}")
        print(f"  {'Engine':<22} {'HW':>4} {'Struct':>6} {'Deploy':>7} "
              f"{'Tput':>6} {'Quant':>6} {'Score':>7}")
        print(f"  {SEPARATOR}")

        scores = []
        for eng in ENGINES.values():
            s = eng.score(weights)
            scores.append((eng, s))
        scores.sort(key=lambda x: x[1], reverse=True)

        for eng, s in scores:
            print(f"  {eng.name:<22} {eng.hardware_flexibility:>4} "
                  f"{eng.structured_output:>6} {eng.deploy_simplicity:>7} "
                  f"{eng.throughput_ceiling:>6} {eng.quant_support:>6} {s:>7.2f}")

        winner = scores[0][0]
        print(f"  → Winner: {winner.name}  ({winner.notes})")

    print()
    struct_scores = sorted(ENGINES.values(),
                            key=lambda e: e.score(profiles["Structured output (JSON extraction)"]),
                            reverse=True)
    tput_scores   = sorted(ENGINES.values(),
                            key=lambda e: e.score(profiles["Max throughput, stable NVIDIA prod"]),
                            reverse=True)
    assert "SGLang" in struct_scores[0].name
    tput_winner = tput_scores[0]
    assert tput_winner.throughput_ceiling >= 4
    ollama_rank = next(i for i, e in enumerate(tput_scores) if e.name == "Ollama")
    assert ollama_rank >= 6
    print(f"  ✓ Structured output winner:  {struct_scores[0].name}")
    print(f"  ✓ Throughput winner:          {tput_winner.name} (ceiling={tput_winner.throughput_ceiling})")
    print(f"  ✓ Ollama throughput rank:     {ollama_rank+1}/{len(tput_scores)} (expected low)")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: RadixAttention Savings
# ─────────────────────────────────────────────────────────────────────────────

def radix_attention_savings(prefix_tokens: int, user_tokens: int,
                             output_tokens: int, n_users: int,
                             params_b: float) -> dict:
    """
    Calculate prefill FLOP savings from RadixAttention.
    First user always cold; subsequent users hit the prefix cache.
    """
    flops_per_token = 2.0 * params_b * 1e9

    total_tokens_no_cache = n_users * (prefix_tokens + user_tokens)
    flops_no_cache = total_tokens_no_cache * flops_per_token

    flops_cold = (prefix_tokens + user_tokens) * flops_per_token
    flops_hits  = (n_users - 1) * user_tokens * flops_per_token
    flops_cache = flops_cold + flops_hits

    saving = (flops_no_cache - flops_cache) / flops_no_cache
    return {
        "flops_no_cache": flops_no_cache,
        "flops_with_cache": flops_cache,
        "saving_pct": saving * 100,
        "prefix_fraction": prefix_tokens / (prefix_tokens + user_tokens),
    }

def demo_radix_attention():
    print(f"\n{'='*70}")
    print("DEMO 2 — RadixAttention Prefix-Reuse Savings")
    print(f"{'='*70}")

    print(f"\n  Model: 70B  |  Output: 256 tokens\n")
    print(f"  {'Prefix':>8} {'User':>6} {'N users':>8} {'No cache':>14} "
          f"{'With cache':>14} {'Saving':>8}")
    print(f"  {SEPARATOR}")

    configs = [
        (512,  128, 10),
        (512,  128, 100),
        (512,  128, 1000),
        (256,  512, 100),
        (1024, 128, 100),
        (128,  512, 100),
    ]
    for prefix, user, n in configs:
        r = radix_attention_savings(prefix, user, 256, n, 70.0)
        no_c = r["flops_no_cache"]  / 1e15
        wi_c = r["flops_with_cache"] / 1e15
        print(f"  {prefix:>8} {user:>6} {n:>8} {no_c:>13.2f}P {wi_c:>13.2f}P "
              f"{r['saving_pct']:>7.1f}%")

    r = radix_attention_savings(512, 128, 256, 100, 70.0)
    assert r["saving_pct"] > 70.0
    print(f"\n  ✓ Worked Example 33b.1: 512-prefix, 100 users → {r['saving_pct']:.1f}% saving")

    print(f"\n  Formula: saving = (N-1) × prefix / (N × (prefix + user))")
    print(f"  At N→∞:  saving → prefix / (prefix + user)")
    for prefix, user in [(512,128), (256,512), (128,512)]:
        asymptote = prefix / (prefix + user) * 100
        print(f"    prefix={prefix}, user={user}: asymptotic saving = {asymptote:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: Roofline Throughput Ceilings
# ─────────────────────────────────────────────────────────────────────────────

def decode_tps_roofline(params_b: float, n_gpus: int,
                         dtype_bytes: float, sparse_factor: float = 1.0) -> float:
    total_bw = H100_HBM_BW_GBS * n_gpus * 0.85 * 1e9
    weight_bytes = params_b * 1e9 * dtype_bytes / sparse_factor
    return total_bw / weight_bytes

def demo_roofline():
    print(f"\n{'='*70}")
    print("DEMO 3 — Roofline Throughput Ceilings (batch=1, 70B model)")
    print(f"{'='*70}")

    n_gpus_options = [1, 2, 4, 8]
    params_b = 70.0

    configs = [
        ("vLLM BF16",          2.0, 1.0),
        ("vLLM FP8",           1.0, 1.0),
        ("SGLang BF16",        2.0, 1.0),
        ("SGLang FP8",         1.0, 1.0),
        ("TRT-LLM FP8",        1.0, 1.0),
        ("TRT-LLM FP8+2:4sp",  1.0, 2.0),
        ("MLC-LLM INT4",       0.5, 1.0),
        ("Ollama Q4_K_M",      0.5, 1.0),
    ]

    print(f"\n  {'Engine':<22}", end="")
    for n in n_gpus_options:
        print(f"  {n}×H100", end="")
    print()
    print(f"  {SEPARATOR}")

    for name, db, sf in configs:
        print(f"  {name:<22}", end="")
        for n in n_gpus_options:
            tps = decode_tps_roofline(params_b, n, db, sf)
            print(f"  {tps:>6.1f}", end="")
        print()

    tps_trt_sp = decode_tps_roofline(params_b, 4, 1.0, 2.0)
    tps_vllm   = decode_tps_roofline(params_b, 4, 2.0, 1.0)
    assert tps_trt_sp > tps_vllm * 3.5
    print(f"\n  ✓ TRT-LLM FP8+2:4 (4×H100): {tps_trt_sp:.0f} tok/s vs vLLM BF16: "
          f"{tps_vllm:.0f} tok/s ({tps_trt_sp/tps_vllm:.1f}×)")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4: TRT-LLM Break-Even Analysis
# ─────────────────────────────────────────────────────────────────────────────

def trtllm_breakeven(compile_hrs: float, n_gpus: int,
                      speedup: float, req_per_hr: float,
                      rev_per_req: float) -> dict:
    compile_cost = compile_hrs * n_gpus * H100_COST_PER_HR
    rev_per_hr   = req_per_hr * rev_per_req
    extra_rev_hr = rev_per_hr * (speedup - 1.0) / speedup
    breakeven_hr = compile_cost / extra_rev_hr if extra_rev_hr > 0 else float("inf")
    return {
        "compile_cost_usd": compile_cost,
        "rev_per_hr": rev_per_hr,
        "extra_rev_per_hr": extra_rev_hr,
        "breakeven_hrs": breakeven_hr,
    }

def demo_breakeven():
    print(f"\n{'='*70}")
    print("DEMO 4 — TRT-LLM Compilation Break-Even Analysis")
    print(f"{'='*70}")

    cases = [
        ("Llama-3.1-8B,  light opt", 0.5, 1, 1.30, 50000, 0.001),
        ("Llama-3.1-8B,  full opt",  0.8, 1, 1.45, 50000, 0.001),
        ("Llama-3.1-70B, FP8",       1.5, 4, 2.40, 10000, 0.005),
        ("Llama-3.1-70B, FP8+sparse",2.0, 4, 4.00, 10000, 0.005),
        ("Nemotron-340B, FP8",        5.0, 8, 2.80, 2000,  0.020),
        ("Low-traffic svc",           1.5, 4, 2.40,   100, 0.005),
    ]

    print(f"\n  {'Config':<32} {'Compile $':>10} {'Extra $/hr':>12} {'Break-even':>12}")
    print(f"  {SEPARATOR}")

    for name, hrs, gpus, speedup, rph, rpr in cases:
        r = trtllm_breakeven(hrs, gpus, speedup, rph, rpr)
        be_str = (f"{r['breakeven_hrs']:.1f} hrs"
                  if r['breakeven_hrs'] < 1000 else "never")
        print(f"  {name:<32} ${r['compile_cost_usd']:>8.0f} ${r['extra_rev_per_hr']:>10.2f} "
              f"{be_str:>12}")

    r = trtllm_breakeven(1.5, 4, 2.40, 10000, 0.005)
    assert r["breakeven_hrs"] < 24.0
    print(f"\n  ✓ Worked Example: 70B FP8 break-even = {r['breakeven_hrs']:.1f} hrs of prod traffic ✓")
    print(f"\n  Rule: if daily production > break-even hours, compile is worthwhile.")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5: Decision Algorithm
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkloadProfile:
    hardware:          str
    stable_model:      bool
    structured_output: bool
    multi_user:        bool
    lora_adapters:     bool
    budget_constrained:bool

def decide_engine(p: WorkloadProfile) -> tuple[str, str]:
    if p.hardware in ("cpu", "apple"):
        if p.multi_user:
            return "llama.cpp", "CPU/Apple Silicon, multi-user → llama.cpp server"
        return "Ollama", "CPU/Apple Silicon, developer mode → Ollama"
    if p.hardware == "browser":
        return "MLC-LLM", "Browser/WebGPU target — only viable option"
    if p.hardware == "amd":
        return "vLLM (ROCm)", "AMD GPU → vLLM with ROCm backend"
    if not p.multi_user:
        return "Ollama", "Single user, NVIDIA → Ollama is simplest"
    if p.structured_output and not p.budget_constrained:
        return "SGLang", "Structured output + shared prefixes → SGLang RadixAttention"
    if p.stable_model and p.budget_constrained and not p.lora_adapters:
        return "TRT-LLM", "Stable model, NVIDIA, max throughput → TRT-LLM (compile once)"
    if p.lora_adapters:
        return "vLLM", "Multi-LoRA serving → vLLM (best LoRA support)"
    return "vLLM", "Default: dynamic workloads, broad support → vLLM"

def demo_decision_algorithm():
    print(f"\n{'='*70}")
    print("DEMO 5 — Decision Algorithm: Flowchart as Executable Code")
    print(f"{'='*70}")

    scenarios = [
        ("JSON extraction API, 4×H100, shared prompt",
         WorkloadProfile("nvidia", False, True,  True,  False, False)),
        ("Max throughput, 8×H100, stable 70B, 6-month horizon",
         WorkloadProfile("nvidia", True,  False, True,  False, True)),
        ("Developer laptop, M3 Max, local coding assistant",
         WorkloadProfile("apple",  False, False, False, False, False)),
        ("Multi-LoRA serving, 2×A100, 50 adapters",
         WorkloadProfile("nvidia", False, False, True,  True,  False)),
        ("Browser-based demo app",
         WorkloadProfile("browser",False, False, True,  False, False)),
        ("AMD MI300X cluster",
         WorkloadProfile("amd",   False, False, True,  False, False)),
        ("Single dev, RTX 4090, testing a new model",
         WorkloadProfile("nvidia", False, False, False, False, False)),
    ]

    print()
    for desc, profile in scenarios:
        engine, reason = decide_engine(profile)
        print(f"  Scenario: {desc}")
        print(f"  → {engine}")
        print(f"     {reason}\n")

    eng, _ = decide_engine(WorkloadProfile("nvidia", False, True, True, False, False))
    assert eng == "SGLang"
    eng, _ = decide_engine(WorkloadProfile("nvidia", True, False, True, False, True))
    assert eng == "TRT-LLM"
    eng, _ = decide_engine(WorkloadProfile("browser", False, False, True, False, False))
    assert eng == "MLC-LLM"
    print("  ✓ Decision algorithm assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6: Structured Output Throughput Model
# ─────────────────────────────────────────────────────────────────────────────

def demo_structured_output():
    print(f"\n{'='*70}")
    print("DEMO 6 — Structured Output Throughput Model")
    print(f"{'='*70}")

    base_tpot_ms   = 50.0
    vllm_mask_ms   = 1.5
    sglang_mask_ms = 0.3
    n_output_tokens= 128

    vllm_unstr_latency  = base_tpot_ms * n_output_tokens
    vllm_str_latency    = (base_tpot_ms + vllm_mask_ms) * n_output_tokens
    sglang_str_latency  = (base_tpot_ms + sglang_mask_ms) * n_output_tokens

    print(f"\n  Scenario: 70B BF16, batch=1, {n_output_tokens} output tokens, JSON output\n")
    print(f"  {'Config':<35} {'Decode (ms)':>12} {'Overhead':>10} {'Relative':>10}")
    print(f"  {SEPARATOR}")

    rows = [
        ("vLLM BF16, no structure",      vllm_unstr_latency, 0),
        ("vLLM + outlines (CPU mask)",   vllm_str_latency,   vllm_mask_ms * n_output_tokens),
        ("SGLang native (CUDA mask)",     sglang_str_latency, sglang_mask_ms * n_output_tokens),
    ]
    base = rows[0][1]
    for name, lat, oh in rows:
        print(f"  {name:<35} {lat:>11.0f} {oh:>9.0f}  {lat/base:>9.2f}×")

    overhead_ratio = vllm_str_latency / sglang_str_latency
    assert overhead_ratio > 1.01
    print(f"\n  SGLang structured gen is {overhead_ratio:.2f}× faster than vLLM+outlines")

    retry_rates = [0.05, 0.15, 0.30, 0.50]
    print(f"\n  Retry overhead (without constrained decoding):")
    print(f"  {'Retry rate':>12} {'Avg attempts':>14} {'Effective latency':>20}")
    print(f"  {SEPARATOR}")
    for rr in retry_rates:
        avg_attempts = 1.0 / (1.0 - rr)
        eff_lat = vllm_unstr_latency * avg_attempts
        print(f"  {rr:>11.0%} {avg_attempts:>14.2f} {eff_lat:>18.0f} ms")
    print(f"\n  ✓ Constrained decoding eliminates retries (retry rate = 0%)")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 7: MLC-LLM Cross-Device Comparison
# ─────────────────────────────────────────────────────────────────────────────

def demo_mlc_cross_device():
    print(f"\n{'='*70}")
    print("DEMO 7 — MLC-LLM Cross-Device Comparison")
    print(f"{'='*70}")

    @dataclass
    class DeviceBenchmark:
        device: str; model: str
        llamacpp_tps: float; mlc_tps: float; vllm_tps: float

    benchmarks = [
        DeviceBenchmark("Apple M3 Max (128GB)",  "Llama-3.1-8B",  48.0, 41.0,  0.0),
        DeviceBenchmark("Apple M3 Max (128GB)",  "Llama-3.1-70B",  8.2,  7.1,  0.0),
        DeviceBenchmark("Apple M3 Max (128GB)",  "Qwen2.5-32B",  12.1, 10.9,  0.0),
        DeviceBenchmark("RTX 4090 (24GB)",       "Llama-3.1-8B",  58.0, 49.0, 320.0),
        DeviceBenchmark("H100 SXM (80GB)",       "Llama-3.1-70B",  0.0,  0.0, 450.0),
        DeviceBenchmark("Browser (WebGPU)",      "Llama-3.2-1B",   0.0, 12.0,  0.0),
        DeviceBenchmark("Browser (WebGPU)",      "Llama-3.2-3B",   0.0,  5.0,  0.0),
    ]

    print(f"\n  {'Device':<26} {'Model':<20} {'llama.cpp':>10} {'MLC-LLM':>9} {'vLLM':>8}")
    print(f"  {SEPARATOR}")
    for b in benchmarks:
        def fmt(v): return f"{v:.1f}" if v > 0 else "N/A"
        print(f"  {b.device:<26} {b.model:<20} {fmt(b.llamacpp_tps):>10} "
              f"{fmt(b.mlc_tps):>9} {fmt(b.vllm_tps):>8}")

    apple_8b = next(b for b in benchmarks if "M3" in b.device and "8B" in b.model)
    ratio = apple_8b.mlc_tps / apple_8b.llamacpp_tps
    assert 0.80 <= ratio <= 1.0
    print(f"\n  ✓ MLC-LLM Apple M3 Max: {ratio:.0%} of llama.cpp throughput ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 8: Full Benchmark Matrix
# ─────────────────────────────────────────────────────────────────────────────

def demo_benchmark_matrix():
    print(f"\n{'='*70}")
    print("DEMO 8 — Full Benchmark Matrix: All Engines, All Configs")
    print(f"{'='*70}")

    params_b = 70.0; n_gpus = 4
    print(f"\n  70B model, {n_gpus}×H100, decode tok/s (bandwidth roofline, batch=1)\n")
    print(f"  {'Engine':<24} {'tok/s':>8} {'vs vLLM BF16':>14} {'GB weights':>12} Notes")
    print(f"  {SEPARATOR}")

    engine_configs = [
        ("vLLM BF16",         2.0, 1.0, "baseline"),
        ("vLLM FP8",          1.0, 1.0, "2× via weight compression"),
        ("SGLang BF16",       2.0, 1.0, "same roofline; gains via prefix reuse"),
        ("SGLang FP8",        1.0, 1.0, "same as vLLM FP8 roofline"),
        ("TRT-LLM FP8",       1.0, 1.0, "AOT compiled, same roofline as vLLM FP8"),
        ("TRT-LLM FP8+2:4",   1.0, 2.0, "2:4 sparsity halves memory BW"),
        ("MLC-LLM INT4",      0.5, 1.0, "INT4 effectively halves weight BW"),
        ("llama.cpp Q4",      0.5, 1.0, "~80% of theoretical (CPU overhead)"),
        ("Ollama",            0.5, 1.0, "wraps llama.cpp, similar ceiling"),
    ]
    vllm_bf16_tps = decode_tps_roofline(params_b, n_gpus, 2.0, 1.0)
    for name, db, sf, notes in engine_configs:
        tps  = decode_tps_roofline(params_b, n_gpus, db, sf)
        w_gb = params_b * 1e9 * db / (sf * 1e9)
        print(f"  {name:<24} {tps:>8.1f} {tps/vllm_bf16_tps:>13.2f}× {w_gb:>10.0f}G  {notes}")

    print(f"\n  ✓ Benchmark matrix complete")


def main():
    bar = "=" * 70
    print(f"\n{bar}")
    print("  Chapter 33b — Choosing Your Engine (Python)")
    print(f"{bar}")

    demo_5axis_scoring()
    demo_radix_attention()
    demo_roofline()
    demo_breakeven()
    demo_decision_algorithm()
    demo_structured_output()
    demo_mlc_cross_device()
    demo_benchmark_matrix()

    print(f"\n{bar}")
    print("  All demos complete.")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
```

---

## C++

```cpp
/*
 * engine_comparison_demo.cpp — Chapter 33b: Choosing Your Engine
 *
 * Implements (mirrors engine_comparison_demo.py):
 *   Demo 1: 5-axis scoring model for engine selection
 *   Demo 2: RadixAttention prefix-reuse savings calculator
 *   Demo 3: Roofline throughput ceilings per engine
 *   Demo 4: TRT-LLM break-even analysis
 *   Demo 5: Decision algorithm — flowchart as executable code
 *   Demo 6: Structured output throughput model
 *   Demo 7: MLC-LLM cross-device comparison
 *   Demo 8: Full benchmark matrix
 *
 * Compile: g++ -std=c++17 -O2 -o engine_comparison_demo engine_comparison_demo.cpp -lm
 * Run:     ./engine_comparison_demo
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const double H100_HBM_BW_GBS  = 3350.0;
static const double H100_COST_PER_HR = 28.0;

static const char* SEP =
    "──────────────────────────────────────────────────────────────────────";

struct EngineSpec {
    const char* name;
    int    hw_flex;
    int    structured;
    int    deploy;
    int    throughput;
    int    quant;
    double tput_mult;
    const char* notes;

    double score(int w_hw, int w_struct, int w_deploy,
                 int w_tput, int w_quant) const {
        double total = hw_flex    * w_hw
                     + structured * w_struct
                     + deploy     * w_deploy
                     + throughput * w_tput
                     + quant      * w_quant;
        return total / (w_hw + w_struct + w_deploy + w_tput + w_quant);
    }
};

static EngineSpec ENGINES[] = {
    {"vLLM BF16",        5,3,4,3,4, 1.00, "Broad support, PagedAttention"},
    {"vLLM FP8",         5,3,4,4,5, 2.00, "2× via FP8 weight compression"},
    {"SGLang BF16",      4,5,4,3,4, 1.20, "RadixAttention + structured gen"},
    {"SGLang FP8",       4,5,4,4,5, 2.30, "RadixAttention + FP8"},
    {"TRT-LLM FP8",      2,2,2,5,5, 2.80, "Highest throughput, compile needed"},
    {"TRT-LLM FP8+2:4",  2,2,2,5,5, 4.00, "FP8 + 2:4 sparsity, 4× baseline"},
    {"MLC-LLM INT4",     5,2,3,3,4, 0.80, "Cross-device, browser capable"},
    {"llama.cpp Q4",     5,2,4,2,3, 0.35, "CPU/edge, GGUF ecosystem"},
    {"Ollama",           5,1,5,1,3, 0.30, "Developer UX, wraps llama.cpp"},
};
static const int N_ENGINES = 9;

static void demo_5axis_scoring() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 1 — 5-Axis Scoring Model\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    struct Profile { const char* name; int w_hw,w_struct,w_deploy,w_tput,w_quant; };
    Profile profiles[] = {
        {"Structured output (JSON)",     1,3,1,2,1},
        {"Max throughput, stable NVIDIA",1,1,1,6,2},
        {"Developer laptop, easy setup", 2,1,4,1,1},
    };

    for (auto& p : profiles) {
        printf("\n  Profile: %s\n", p.name);
        printf("  %-22s %4s %6s %7s %6s %6s %7s\n",
               "Engine","HW","Struct","Deploy","Tput","Quant","Score");
        printf("  %s\n", SEP);

        int order[N_ENGINES];
        for (int i = 0; i < N_ENGINES; ++i) order[i] = i;
        std::sort(order, order+N_ENGINES, [&](int a, int b){
            return ENGINES[a].score(p.w_hw,p.w_struct,p.w_deploy,p.w_tput,p.w_quant)
                 > ENGINES[b].score(p.w_hw,p.w_struct,p.w_deploy,p.w_tput,p.w_quant);
        });
        for (int i = 0; i < N_ENGINES; ++i) {
            const auto& e = ENGINES[order[i]];
            double s = e.score(p.w_hw,p.w_struct,p.w_deploy,p.w_tput,p.w_quant);
            printf("  %-22s %4d %6d %7d %6d %6d %7.2f\n",
                   e.name,e.hw_flex,e.structured,e.deploy,e.throughput,e.quant,s);
        }
        printf("  → Winner: %s  (%s)\n", ENGINES[order[0]].name, ENGINES[order[0]].notes);
    }

    int str_order[N_ENGINES];
    for (int i=0;i<N_ENGINES;++i) str_order[i]=i;
    std::sort(str_order,str_order+N_ENGINES,[](int a,int b){
        return ENGINES[a].score(1,3,1,2,1)>ENGINES[b].score(1,3,1,2,1);
    });
    assert(ENGINES[str_order[0]].structured >= 4);

    int tput_order[N_ENGINES];
    for (int i=0;i<N_ENGINES;++i) tput_order[i]=i;
    std::sort(tput_order,tput_order+N_ENGINES,[](int a,int b){
        return ENGINES[a].score(1,1,1,6,2)>ENGINES[b].score(1,1,1,6,2);
    });
    assert(ENGINES[tput_order[0]].throughput >= 4);

    int ollama_rank=-1;
    for(int i=0;i<N_ENGINES;++i)
        if(strcmp(ENGINES[tput_order[i]].name,"Ollama")==0){ollama_rank=i;break;}
    assert(ollama_rank >= N_ENGINES-2);
    printf("\n  ✓ Assertions passed\n");
}

struct RadixResult { double flops_no_cache, flops_with_cache, saving_pct; };

static RadixResult radix_savings(int prefix,int user,int n,double params_b) {
    double fpt = 2.0*params_b*1e9;
    double no_cache = n*(prefix+user)*fpt;
    double with_cache = (prefix+user)*fpt + (n-1)*user*fpt;
    return {no_cache, with_cache, (no_cache-with_cache)/no_cache*100.0};
}

static void demo_radix_attention() {
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 2 — RadixAttention Prefix-Reuse Savings\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    printf("\n  %8s %6s %8s %14s %14s %8s\n","Prefix","User","N users","No cache","With cache","Saving");
    printf("  %s\n",SEP);

    struct Cfg{int prefix,user,n;};
    Cfg cfgs[]={{512,128,10},{512,128,100},{512,128,1000},{256,512,100},{1024,128,100},{128,512,100}};
    for(auto&c:cfgs){
        auto r=radix_savings(c.prefix,c.user,c.n,70.0);
        printf("  %8d %6d %8d %13.2fP %13.2fP %7.1f%%\n",
               c.prefix,c.user,c.n,
               r.flops_no_cache/1e15,r.flops_with_cache/1e15,r.saving_pct);
    }
    auto r=radix_savings(512,128,100,70.0);
    assert(r.saving_pct>70.0);
    printf("\n  ✓ Worked Example 33b.1: 512-prefix, 100 users → %.1f%% saving\n",r.saving_pct);

    printf("\n  Formula: saving = (N-1) × prefix / (N × (prefix + user))\n");
    for(auto[prefix,user]:std::vector<std::pair<int,int>>{{512,128},{256,512},{128,512}})
        printf("    prefix=%d, user=%d: asymptote = %.1f%%\n",
               prefix,user,(double)prefix/(prefix+user)*100.0);
}

static double decode_tps_roofline(double params_b,int n_gpus,double dtype_bytes,double sparse=1.0){
    double bw=H100_HBM_BW_GBS*n_gpus*0.85*1e9;
    return bw/(params_b*1e9*dtype_bytes/sparse);
}

static void demo_roofline(){
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 3 — Roofline Throughput Ceilings (batch=1, 70B model)\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    int gpus[]={1,2,4,8};
    struct Cfg{const char*n;double db,sf;};
    Cfg cfgs[]={
        {"vLLM BF16",2.0,1.0},{"vLLM FP8",1.0,1.0},
        {"SGLang BF16",2.0,1.0},{"SGLang FP8",1.0,1.0},
        {"TRT-LLM FP8",1.0,1.0},{"TRT-LLM FP8+2:4sp",1.0,2.0},
        {"MLC-LLM INT4",0.5,1.0},{"Ollama Q4_K_M",0.5,1.0},
    };
    printf("\n  %-22s","Engine");
    for(int n:gpus) printf("  %dx H100",n);
    printf("\n  %s\n",SEP);
    for(auto&c:cfgs){
        printf("  %-22s",c.n);
        for(int n:gpus) printf("  %7.1f",decode_tps_roofline(70.0,n,c.db,c.sf));
        printf("\n");
    }
    double trt=decode_tps_roofline(70.0,4,1.0,2.0);
    double vll=decode_tps_roofline(70.0,4,2.0,1.0);
    assert(trt>vll*3.5);
    printf("\n  ✓ TRT-LLM FP8+2:4 (4×H100): %.0f tok/s vs vLLM BF16: %.0f tok/s (%.1fx)\n",
           trt,vll,trt/vll);
}

struct BreakevenResult{double compile_cost,rev_per_hr,extra_rev_hr,breakeven_hrs;};

static BreakevenResult breakeven(double hrs,int n,double speedup,double rph,double rpr){
    double cc=hrs*n*H100_COST_PER_HR;
    double rev=rph*rpr;
    double extra=rev*(speedup-1.0)/speedup;
    return{cc,rev,extra,extra>0?cc/extra:1e9};
}

static void demo_breakeven(){
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 4 — TRT-LLM Compilation Break-Even Analysis\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    struct Case{const char*n;double hrs;int gpus;double speedup,rph,rpr;};
    Case cases[]={
        {"Llama-3.1-8B, light opt",  0.5,1,1.30,50000,0.001},
        {"Llama-3.1-8B, full opt",   0.8,1,1.45,50000,0.001},
        {"Llama-3.1-70B, FP8",       1.5,4,2.40,10000,0.005},
        {"Llama-3.1-70B, FP8+sparse",2.0,4,4.00,10000,0.005},
        {"Nemotron-340B, FP8",        5.0,8,2.80, 2000,0.020},
        {"Low-traffic svc",           1.5,4,2.40,  100,0.005},
    };
    printf("\n  %-32s %10s %12s %12s\n","Config","Compile $","Extra $/hr","Break-even");
    printf("  %s\n",SEP);
    char be[32];
    for(auto&c:cases){
        auto r=breakeven(c.hrs,c.gpus,c.speedup,c.rph,c.rpr);
        if(r.breakeven_hrs<1000) snprintf(be,32,"%.1f hrs",r.breakeven_hrs);
        else snprintf(be,32,"never");
        printf("  %-32s %10.0f %12.2f %12s\n",c.n,r.compile_cost,r.extra_rev_hr,be);
    }
    auto r=breakeven(1.5,4,2.40,10000,0.005);
    assert(r.breakeven_hrs<24.0);
    printf("\n  ✓ 70B FP8 break-even = %.1f hrs ✓\n",r.breakeven_hrs);
}

struct WorkloadProfile{
    const char*hardware;
    bool stable_model,structured_output,multi_user,lora_adapters,budget_constrained;
};

static const char* decide_engine(const WorkloadProfile&p,const char**reason){
    if(strcmp(p.hardware,"cpu")==0||strcmp(p.hardware,"apple")==0){
        if(p.multi_user){*reason="CPU/Apple Silicon, multi-user → llama.cpp";return"llama.cpp";}
        *reason="CPU/Apple Silicon, dev → Ollama";return"Ollama";
    }
    if(strcmp(p.hardware,"browser")==0){*reason="Browser/WebGPU only viable";return"MLC-LLM";}
    if(strcmp(p.hardware,"amd")==0){*reason="AMD → vLLM ROCm";return"vLLM (ROCm)";}
    if(!p.multi_user){*reason="Single user, NVIDIA → Ollama";return"Ollama";}
    if(p.structured_output&&!p.budget_constrained){
        *reason="Structured+shared prefix → SGLang RadixAttention";return"SGLang";}
    if(p.stable_model&&p.budget_constrained&&!p.lora_adapters){
        *reason="Stable+NVIDIA+max tput → TRT-LLM";return"TRT-LLM";}
    if(p.lora_adapters){*reason="Multi-LoRA → vLLM";return"vLLM";}
    *reason="Default: dynamic → vLLM";return"vLLM";
}

static void demo_decision_algorithm(){
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 5 — Decision Algorithm: Flowchart as Executable Code\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    struct Scenario{const char*desc;WorkloadProfile p;};
    Scenario scenarios[]={
        {"JSON extraction, 4xH100, shared prompt",  {"nvidia",false,true, true, false,false}},
        {"Max throughput, 8xH100, stable 70B",      {"nvidia",true, false,true, false,true }},
        {"Developer laptop, M3 Max",                {"apple", false,false,false,false,false}},
        {"Multi-LoRA, 2xA100, 50 adapters",         {"nvidia",false,false,true, true, false}},
        {"Browser-based demo app",                  {"browser",false,false,true,false,false}},
        {"AMD MI300X cluster",                      {"amd",  false,false,true, false,false}},
        {"Single dev, RTX 4090",                    {"nvidia",false,false,false,false,false}},
    };
    printf("\n");
    const char*reason;
    for(auto&sc:scenarios)
        printf("  Scenario: %s\n  → %s\n     %s\n\n",
               sc.desc,decide_engine(sc.p,&reason),reason);

    const char*r2;
    assert(strcmp(decide_engine({"nvidia",false,true,true,false,false},&r2),"SGLang")==0);
    assert(strcmp(decide_engine({"nvidia",true,false,true,false,true}, &r2),"TRT-LLM")==0);
    assert(strcmp(decide_engine({"browser",false,false,true,false,false},&r2),"MLC-LLM")==0);
    printf("  ✓ Decision algorithm assertions passed\n");
}

static void demo_structured_output(){
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 6 — Structured Output Throughput Model\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    const double base=50.0,vm=1.5,sm=0.3; const int n=128;
    double vu=base*n, vs=(base+vm)*n, ss=(base+sm)*n;
    printf("\n  %-35s %12s %10s %10s\n","Config","Decode (ms)","Overhead","Relative");
    printf("  %s\n",SEP);
    struct Row{const char*n;double lat,oh;};
    Row rows[]={"vLLM BF16, no structure",vu,0,
                "vLLM + outlines (CPU mask)",vs,vm*n,
                "SGLang native (CUDA mask)",ss,sm*n};
    for(auto&r:rows)
        printf("  %-35s %12.0f %10.0f %10.2fx\n",r.n,r.lat,r.oh,r.lat/vu);
    assert(vs/ss>1.01);
    printf("\n  SGLang vs vLLM+outlines: %.2fx faster\n",vs/ss);
    printf("  ✓ Constrained decoding eliminates retries entirely\n");
}

static void demo_mlc_cross_device(){
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 7 — MLC-LLM Cross-Device Comparison\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    struct Bench{const char*device,*model;double ll,ml,vl;};
    Bench bs[]={
        {"Apple M3 Max (128GB)","Llama-3.1-8B", 48.0,41.0,0.0},
        {"Apple M3 Max (128GB)","Llama-3.1-70B", 8.2, 7.1,0.0},
        {"Apple M3 Max (128GB)","Qwen2.5-32B",  12.1,10.9,0.0},
        {"RTX 4090 (24GB)",     "Llama-3.1-8B", 58.0,49.0,320.0},
        {"H100 SXM (80GB)",     "Llama-3.1-70B", 0.0, 0.0,450.0},
        {"Browser (WebGPU)",    "Llama-3.2-1B",  0.0,12.0,0.0},
        {"Browser (WebGPU)",    "Llama-3.2-3B",  0.0, 5.0,0.0},
    };
    printf("\n  %-26s %-20s %10s %9s %8s\n","Device","Model","llama.cpp","MLC-LLM","vLLM");
    printf("  %s\n",SEP);
    auto fmt=[](double v,char*buf){if(v>0)snprintf(buf,16,"%.1f",v);else snprintf(buf,16,"N/A");};
    for(auto&b:bs){char ll[16],ml[16],vl[16];fmt(b.ll,ll);fmt(b.ml,ml);fmt(b.vl,vl);
        printf("  %-26s %-20s %10s %9s %8s\n",b.device,b.model,ll,ml,vl);}
    double ratio=bs[0].ml/bs[0].ll;
    assert(ratio>=0.80&&ratio<=1.0);
    printf("\n  ✓ MLC-LLM Apple M3 Max = %.0f%% of llama.cpp ✓\n",ratio*100.0);
}

static void demo_benchmark_matrix(){
    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("DEMO 8 — Full Benchmark Matrix\n");
    printf("%s\n","══════════════════════════════════════════════════════════════════════");

    const double P=70.0; const int G=4;
    printf("\n  70B model, %dx H100, decode tok/s\n\n",G);
    printf("  %-24s %8s %14s %12s\n","Engine","tok/s","vs vLLM BF16","GB weights");
    printf("  %s\n",SEP);

    struct Cfg{const char*n;double db,sf;const char*notes;};
    Cfg cfgs[]={
        {"vLLM BF16",2.0,1.0,"baseline"},
        {"vLLM FP8",1.0,1.0,"2× via weight compression"},
        {"SGLang BF16",2.0,1.0,"gains via prefix reuse"},
        {"SGLang FP8",1.0,1.0,"same as vLLM FP8"},
        {"TRT-LLM FP8",1.0,1.0,"AOT compiled"},
        {"TRT-LLM FP8+2:4",1.0,2.0,"2:4 sparsity"},
        {"MLC-LLM INT4",0.5,1.0,"INT4 weight BW"},
        {"llama.cpp Q4",0.5,1.0,"~80% theoretical"},
        {"Ollama",0.5,1.0,"wraps llama.cpp"},
    };
    double base=decode_tps_roofline(P,G,2.0,1.0);
    for(auto&c:cfgs){
        double tps=decode_tps_roofline(P,G,c.db,c.sf);
        double wgb=P*1e9*c.db/(c.sf*1e9);
        printf("  %-24s %8.1f %13.2fx %10.0fG  %s\n",c.n,tps,tps/base,wgb,c.notes);
    }
    printf("  ✓ Benchmark matrix complete\n");
}

int main(){
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║   Chapter 33b: Choosing Your Engine (C++)                            ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");

    demo_5axis_scoring();
    demo_radix_attention();
    demo_roofline();
    demo_breakeven();
    demo_decision_algorithm();
    demo_structured_output();
    demo_mlc_cross_device();
    demo_benchmark_matrix();

    printf("\n%s\n","══════════════════════════════════════════════════════════════════════");
    printf("ALL CHAPTER 33b DEMOS COMPLETED — ALL ASSERTIONS PASSED ✓\n");
    printf("%s\n\n","══════════════════════════════════════════════════════════════════════");
    return 0;
}
```
