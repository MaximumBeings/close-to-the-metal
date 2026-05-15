# Chapter 17: Benchmarking — Companion Code

## Python — `benchmark_demo.py`

```python
"""
Chapter 17 Companion Code: Benchmarking — Fair Comparisons Between Engines
===========================================================================
Demonstrates:
  1. Results table renderer — TTFT / ITL / throughput / VRAM side-by-side
  2. Throughput vs batch-size ASCII chart
  3. Crossover detector — finds batch size where vLLM overtakes llama.cpp
  4. Cost-efficiency calculator — tok/s per watt and tok/$
  5. llama-bench CSV parser — extract pp/tg metrics
  6. Benchmark checklist validator — flag incomplete methodologies
  7. Latency percentile analyzer — P50/P95/P99 from raw sample list
  8. Performance model — predict throughput from hardware specs

Run:
    python3 benchmark_demo.py

No external dependencies.
"""

import csv
import io
import math
import statistics
from dataclasses import dataclass
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 1. BENCHMARK RESULT DATA CLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    engine:          str
    batch_size:      int
    prefill_tok_s:   float    # tokens/second during prefill
    ttft_ms:         float    # time to first token (ms)
    decode_tok_s:    float    # tokens/second during decode
    itl_ms:          float    # inter-token latency (ms)
    peak_vram_gb:    float    # peak VRAM usage
    gpu_watts:       float    # GPU TDP during benchmark


# ── Measured results: Llama-3-8B Q4_K_M / RTX 4090 (Section 17.7)
RTX4090_RESULTS: list[BenchResult] = [
    # batch=1
    BenchResult("llama.cpp", 1,  2012, 254, 64.1, 15.6, 5.8, 380),
    BenchResult("vLLM",      1,  1843, 278, 58.3, 17.2, 9.4, 420),
    # batch=4
    BenchResult("llama.cpp", 4,  4210, 121, 230.4, 17.4, 6.2, 390),
    BenchResult("vLLM",      4,  5890,  87, 218.7, 18.3, 10.1, 430),
    # batch=16
    BenchResult("llama.cpp", 16, 6800,  75, 540.1, 29.6, 7.1, 420),
    BenchResult("vLLM",      16, 12340, 41, 892.4, 18.0, 13.7, 450),
    # batch=32
    BenchResult("llama.cpp", 32, 7200,  71, 780.3, 41.1, 8.4, 440),
    BenchResult("vLLM",      32, 19870, 26, 1634.8, 19.6, 18.2, 455),
]


def print_results_table(results: list[BenchResult]):
    print(f"\n{'='*92}")
    print("  Benchmark Results: Llama-3-8B Q4_K_M  |  RTX 4090  |  512 input / 256 output tokens")
    print(f"{'='*92}")
    print(f"  {'Batch':>5}  {'Engine':<10}  {'Prefill':>10}  {'TTFT':>8}  "
          f"{'Decode':>10}  {'ITL':>8}  {'VRAM':>8}  {'Winner'}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}")

    batch_sizes = sorted(set(r.batch_size for r in results))
    for b in batch_sizes:
        batch_res = [r for r in results if r.batch_size == b]
        if len(batch_res) != 2:
            continue
        # Sort: llama.cpp first
        batch_res.sort(key=lambda r: r.engine)
        r_llama, r_vllm = batch_res[0], batch_res[1]
        # Determine winner per metric
        decode_winner = "vLLM" if r_vllm.decode_tok_s > r_llama.decode_tok_s else "llama.cpp"
        ttft_winner   = "vLLM" if r_vllm.ttft_ms < r_llama.ttft_ms else "llama.cpp"
        # Overall: decode throughput determines batch utility
        overall = decode_winner

        for r in [r_llama, r_vllm]:
            win_marker = "←" if r.engine == overall else ""
            print(f"  {r.batch_size:>5}  {r.engine:<10}  "
                  f"{r.prefill_tok_s:>8,.0f}/s  {r.ttft_ms:>6.0f}ms  "
                  f"{r.decode_tok_s:>8,.0f}/s  {r.itl_ms:>6.1f}ms  "
                  f"{r.peak_vram_gb:>6.1f}GB  {win_marker}")
        print()

    print(f"{'='*92}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. THROUGHPUT vs BATCH-SIZE ASCII CHART
# ──────────────────────────────────────────────────────────────────────────────

def print_throughput_chart(results: list[BenchResult], metric: str = "decode"):
    """
    metric: "decode" (tok/s) or "prefill" (tok/s)
    """
    batch_sizes = sorted(set(r.batch_size for r in results))
    engines     = sorted(set(r.engine for r in results))

    def get_val(engine, batch):
        for r in results:
            if r.engine == engine and r.batch_size == batch:
                return r.decode_tok_s if metric == "decode" else r.prefill_tok_s
        return 0.0

    max_val = max(get_val(e, b) for e in engines for b in batch_sizes)
    chart_w = 50

    print(f"\n{'='*72}")
    print(f"  {'Decode' if metric == 'decode' else 'Prefill'} Throughput vs Batch Size  (tok/s)")
    print(f"{'='*72}")

    symbols = {"llama.cpp": "●", "vLLM": "■"}
    for engine in engines:
        print(f"\n  {engine}  {symbols.get(engine,'*')}")
        for b in batch_sizes:
            v    = get_val(engine, b)
            frac = v / max_val if max_val > 0 else 0
            bar  = '█' * int(frac * chart_w)
            print(f"    batch={b:>2}  {bar:<{chart_w}}  {v:>8,.0f} tok/s")

    print(f"\n  Scale: max={max_val:,.0f} tok/s")
    print(f"{'='*72}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. CROSSOVER DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

def find_crossover(results: list[BenchResult]) -> dict:
    """
    Find the batch size where vLLM first overtakes llama.cpp on decode throughput.
    Returns dict with crossover info.
    """
    batch_sizes = sorted(set(r.batch_size for r in results))
    crossover   = None

    llama_prev_lead = True   # llama.cpp leads at batch=1

    for b in batch_sizes:
        llama_r = next((r for r in results if r.engine == "llama.cpp" and r.batch_size == b), None)
        vllm_r  = next((r for r in results if r.engine == "vLLM"      and r.batch_size == b), None)
        if not llama_r or not vllm_r:
            continue

        vllm_leads_now = vllm_r.decode_tok_s > llama_r.decode_tok_s
        if vllm_leads_now and llama_prev_lead:
            crossover = {
                "batch_size":        b,
                "llama_decode_tps":  llama_r.decode_tok_s,
                "vllm_decode_tps":   vllm_r.decode_tok_s,
                "vllm_advantage":    vllm_r.decode_tok_s / llama_r.decode_tok_s,
            }
            break
        llama_prev_lead = not vllm_leads_now

    print(f"\n{'='*62}")
    print("  Performance Crossover Analysis")
    print(f"{'='*62}")
    if crossover:
        print(f"  vLLM first overtakes llama.cpp at batch={crossover['batch_size']}")
        print(f"  At crossover:")
        print(f"    llama.cpp : {crossover['llama_decode_tps']:>8,.1f} tok/s")
        print(f"    vLLM      : {crossover['vllm_decode_tps']:>8,.1f} tok/s")
        print(f"    vLLM advantage: {crossover['vllm_advantage']:.2f}×")
    else:
        print("  No crossover detected in the measured batch range.")

    # Also report the full advantage ratio at each batch size
    print(f"\n  Decode throughput ratio (vLLM / llama.cpp):")
    for b in batch_sizes:
        lr = next((r for r in results if r.engine == "llama.cpp" and r.batch_size == b), None)
        vr = next((r for r in results if r.engine == "vLLM"      and r.batch_size == b), None)
        if lr and vr:
            ratio = vr.decode_tok_s / lr.decode_tok_s
            bar   = '█' * int(ratio * 10)
            marker = " ← vLLM ahead" if ratio > 1 else " ← llama.cpp ahead"
            print(f"    batch={b:>2}  {ratio:5.2f}×  {bar}{marker}")

    print(f"{'='*62}")
    return crossover or {}


# ──────────────────────────────────────────────────────────────────────────────
# 4. COST-EFFICIENCY CALCULATOR
# ──────────────────────────────────────────────────────────────────────────────

def print_cost_efficiency(results: list[BenchResult],
                          electricity_usd_per_kwh: float = 0.10,
                          gpu_cost_usd: float = 1599.0):
    """
    Compute:
      tok/s/W   (energy efficiency)
      hours to payback GPU cost at given throughput
    """
    print(f"\n{'='*70}")
    print(f"  Cost Efficiency  |  ${electricity_usd_per_kwh:.2f}/kWh  "
          f"|  GPU cost ${gpu_cost_usd:,.0f}")
    print(f"{'='*70}")
    print(f"  {'Batch':>5}  {'Engine':<10}  {'tok/s/W':>9}  "
          f"{'tok/$0.01':>10}  {'GPU payback':>13}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*9}  {'─'*10}  {'─'*13}")

    batch_sizes = sorted(set(r.batch_size for r in results))
    for b in batch_sizes:
        for r in sorted([x for x in results if x.batch_size == b],
                        key=lambda x: x.engine):
            # Energy efficiency
            tok_per_watt = r.decode_tok_s / max(1, r.gpu_watts)

            # Cost per 1M output tokens
            # energy cost: (tok/s / r.gpu_watts W) → $ per token
            watts_per_tok = r.gpu_watts / max(1, r.decode_tok_s)          # W·s per token
            kwh_per_tok   = watts_per_tok / 3_600_000                      # kWh per token
            usd_per_tok   = kwh_per_tok * electricity_usd_per_kwh          # $ per token
            tok_per_cent  = 0.01 / max(1e-12, usd_per_tok)

            # GPU payback: hours at continuous full throughput to "justify" GPU cost
            # Assume $0.01 per 1000 output tokens as revenue
            usd_per_second = r.decode_tok_s * 0.01 / 1000
            payback_hrs    = gpu_cost_usd / (usd_per_second * 3600)

            print(f"  {b:>5}  {r.engine:<10}  {tok_per_watt:>8.2f}  "
                  f"{tok_per_cent:>10,.0f}  {payback_hrs:>11.0f} h")
        print()

    print(f"{'='*70}")
    print(f"  Note: payback assumes $0.01/1000 output tokens revenue")


# ──────────────────────────────────────────────────────────────────────────────
# 5. LLAMA-BENCH CSV PARSER
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_LLAMABENCH_CSV = """model,size,params,backend,ngl,test,t/s,t/s_err
llama 8B Q4_K - Medium,4.58 GiB,8.03 B,CUDA,33,pp512,2012.34,18.2
llama 8B Q4_K - Medium,4.58 GiB,8.03 B,CUDA,33,tg1,64.12,0.8
llama 8B Q4_K - Medium,4.58 GiB,8.03 B,CUDA,33,tg4,230.45,2.1
llama 8B Q4_K - Medium,4.58 GiB,8.03 B,CUDA,33,tg16,540.12,4.8
llama 8B Q4_K - Medium,4.58 GiB,8.03 B,CUDA,33,tg32,780.34,7.2
"""


@dataclass
class LlamaBenchRow:
    model:   str
    backend: str
    ngl:     int
    test:    str
    tps:     float
    tps_err: float

    @property
    def is_prefill(self) -> bool:
        return self.test.startswith("pp")

    @property
    def is_decode(self) -> bool:
        return self.test.startswith("tg")

    @property
    def n_tokens(self) -> int:
        prefix = "pp" if self.is_prefill else "tg"
        return int(self.test[len(prefix):])

    @property
    def itl_ms(self) -> float:
        return 1000.0 / self.tps if self.tps > 0 else 0.0


def parse_llamabench_csv(csv_text: str) -> list[LlamaBenchRow]:
    rows = []
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    for row in reader:
        rows.append(LlamaBenchRow(
            model   = row["model"],
            backend = row["backend"],
            ngl     = int(row["ngl"]),
            test    = row["test"],
            tps     = float(row["t/s"]),
            tps_err = float(row["t/s_err"]),
        ))
    return rows


def print_llamabench_analysis(csv_text: str):
    rows = parse_llamabench_csv(csv_text)
    print(f"\n{'='*62}")
    print("  llama-bench CSV Analysis")
    print(f"{'='*62}")

    # Prefill
    prefill = [r for r in rows if r.is_prefill]
    if prefill:
        print(f"\n  Prefill (prompt processing):")
        for r in prefill:
            print(f"    pp{r.n_tokens:>5}  {r.tps:>8,.1f} ± {r.tps_err:>5.1f} tok/s  "
                  f"→ TTFT for {r.n_tokens} tokens ≈ {r.n_tokens/r.tps*1000:.0f} ms")

    # Decode
    decode = [r for r in rows if r.is_decode]
    if decode:
        print(f"\n  Decode (token generation):")
        print(f"  {'Batch':>6}  {'tok/s':>8}  {'±':>6}  {'ITL(ms)':>9}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*9}")
        for r in decode:
            print(f"  {r.n_tokens:>6}  {r.tps:>8,.1f}  {r.tps_err:>6.1f}  {r.itl_ms:>9.2f}")

        # Efficiency ratio vs batch=1
        base = next((r.tps for r in decode if r.n_tokens == 1), decode[0].tps)
        print(f"\n  Batch scaling efficiency (vs tg1):")
        for r in decode:
            efficiency = r.tps / (base * r.n_tokens) * 100
            bar = '█' * int(efficiency / 5)
            print(f"    tg{r.n_tokens:>2}  {efficiency:>5.1f}%  {bar}")

    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. BENCHMARK CHECKLIST VALIDATOR
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkSpec:
    # All fields must be filled for a complete, publishable benchmark
    model_name:       Optional[str]   = None
    quantization:     Optional[str]   = None   # e.g. "Q4_K_M", "BF16", "AWQ"
    gpu_model:        Optional[str]   = None
    cuda_version:     Optional[str]   = None
    vllm_version:     Optional[str]   = None
    llamacpp_commit:  Optional[str]   = None
    warmup_iters:     Optional[int]   = None
    repetitions:      Optional[int]   = None
    batch_sizes:      Optional[list]  = None
    prompt_source:    Optional[str]   = None   # e.g. "ShareGPT", "fixed-512"
    metrics_reported: Optional[list]  = None   # e.g. ["TTFT", "ITL", "throughput", "VRAM"]
    clock_locked:     Optional[bool]  = None


def validate_benchmark_spec(spec: BenchmarkSpec) -> list[str]:
    issues = []

    required = [
        ("model_name",       "Model name and quantization not specified"),
        ("quantization",     "quantization level not specified"),
        ("gpu_model",        "GPU model not specified"),
        ("cuda_version",     "CUDA version not pinned"),
        ("prompt_source",    "Prompt distribution not described"),
        ("batch_sizes",      "Batch sizes not specified"),
        ("metrics_reported", "Metrics to report not listed"),
        ("warmup_iters",     "Warmup iterations not specified"),
        ("repetitions",      "Number of repetitions not specified"),
    ]
    for field, msg in required:
        if getattr(spec, field) is None:
            issues.append(f"MISSING: {msg}")

    if spec.repetitions is not None and spec.repetitions < 5:
        issues.append(f"WARN: repetitions={spec.repetitions} < 5 — insufficient for stable stats")

    if spec.warmup_iters is not None and spec.warmup_iters < 3:
        issues.append(f"WARN: warmup_iters={spec.warmup_iters} < 3 — may include JIT compile overhead")

    expected_metrics = {"TTFT", "ITL", "throughput", "VRAM"}
    if spec.metrics_reported:
        missing_metrics = expected_metrics - set(spec.metrics_reported)
        if missing_metrics:
            issues.append(f"INCOMPLETE METRICS: {missing_metrics} not reported")

    if spec.clock_locked is False:
        issues.append("WARN: GPU clocks not locked — thermal throttle may inflate latency variance")

    if spec.batch_sizes and max(spec.batch_sizes) < 16:
        issues.append("WARN: no batch size ≥ 16 — misses the vLLM throughput advantage region")

    return issues


def print_checklist(spec: BenchmarkSpec):
    issues = validate_benchmark_spec(spec)
    print(f"\n{'='*62}")
    print("  Benchmark Methodology Checklist")
    print(f"{'='*62}")
    if not issues:
        print("  ✓  Benchmark specification is complete and sound.")
    else:
        for issue in issues:
            icon = "!!" if issue.startswith("MISSING") else "! "
            print(f"  {icon}  {issue}")
    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. LATENCY PERCENTILE analyzeR
# ──────────────────────────────────────────────────────────────────────────────

def analyze_latencies(samples: list[float], label: str = "Latency"):
    """
    Compute and display P50/P95/P99 from a raw sample list.
    """
    s = sorted(samples)
    n = len(s)

    mean   = statistics.mean(s)
    median = statistics.median(s)
    stdev  = statistics.stdev(s)
    p95    = s[int(0.95 * n)]
    p99    = s[int(0.99 * n)]
    cv     = stdev / mean * 100   # coefficient of variation

    print(f"\n{'='*62}")
    print(f"  {label} Distribution  (n={n})")
    print(f"{'='*62}")
    print(f"  Mean   : {mean:>8.2f} ms")
    print(f"  Median : {median:>8.2f} ms")
    print(f"  StdDev : {stdev:>8.2f} ms  (CV={cv:.1f}%)")
    print(f"  P95    : {p95:>8.2f} ms")
    print(f"  P99    : {p99:>8.2f} ms")
    print(f"  Min    : {s[0]:>8.2f} ms")
    print(f"  Max    : {s[-1]:>8.2f} ms")
    if cv > 10:
        print(f"\n  ⚠  CV={cv:.1f}% > 10% — high variance. Check for thermal throttle or OS noise.")
    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. PERFORMANCE MODEL — predict throughput from specs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HardwareSpec:
    name: str
    hbm_bw_tb_s: float      # memory bandwidth TB/s
    tflops_bf16: float      # peak BF16 TFLOPS

@dataclass
class WorkloadSpec:
    params_b:     float
    dtype_bytes:  int
    batch_size:   int
    input_tokens: int
    output_tokens: int


def predict_throughput(hw: HardwareSpec, wl: WorkloadSpec) -> dict:
    """
    Roofline model predictions for decode and prefill throughput.
    """
    # Decode: bandwidth-bound
    # Load all weights per step (one token generated per step)
    bytes_per_step = wl.params_b * 1e9 * wl.dtype_bytes
    decode_bw_tok_s = (hw.hbm_bw_tb_s * 1e12) / bytes_per_step

    # Prefill: compute-bound above ridge point
    # FLOPs per token = 2 × params
    flops_per_tok     = 2 * wl.params_b * 1e9
    prefill_comp_tok_s = (hw.tflops_bf16 * 1e12) / flops_per_tok
    ridge_tokens      = hw.tflops_bf16 * 1e12 / (hw.hbm_bw_tb_s * 1e12 / wl.dtype_bytes)

    # At batch=wl.batch_size, decode throughput scales (partially)
    # At small batch, each GPU step still loads all weights
    # Total throughput = batch_size × per-step throughput
    total_decode_tok_s = decode_bw_tok_s * wl.batch_size

    return {
        "decode_per_gpu_tok_s":  decode_bw_tok_s,
        "decode_total_tok_s":    total_decode_tok_s,
        "prefill_tok_s":         prefill_comp_tok_s,
        "ridge_tokens":          ridge_tokens,
        "itl_ms":                1000.0 / decode_bw_tok_s,
    }


def print_performance_model():
    gpus = [
        HardwareSpec("RTX 4090", 1.008, 82.6),
        HardwareSpec("A100-80",  2.000, 312.0),
        HardwareSpec("H100-80",  3.350, 989.0),
    ]
    wl = WorkloadSpec(params_b=8.0, dtype_bytes=2,
                      batch_size=1, input_tokens=512, output_tokens=256)

    print(f"\n{'='*72}")
    print(f"  Roofline Performance Model  |  Llama-3-8B BF16  |  batch=1")
    print(f"{'='*72}")
    print(f"  {'GPU':<12}  {'Decode/GPU':>12}  {'ITL(ms)':>9}  "
          f"{'Prefill':>12}  {'Ridge(tok)':>12}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*9}  {'─'*12}  {'─'*12}")
    for gpu in gpus:
        p = predict_throughput(gpu, wl)
        print(f"  {gpu.name:<12}  {p['decode_per_gpu_tok_s']:>10,.0f}/s  "
              f"{p['itl_ms']:>9.1f}  {p['prefill_tok_s']:>10,.0f}/s  "
              f"{p['ridge_tokens']:>12,.0f}")
    print(f"{'='*72}")
    print(f"  Note: actual measured values are ~70-85% of roofline (kernel overhead,")
    print(f"        memory latency, non-weight compute). Use as upper-bound reference.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

import random

def main():
    print("\n" + "█" * 62)
    print("  Chapter 17 — Benchmarking: Companion Demo")
    print("█" * 62)

    # Section 1: Results table
    print_results_table(RTX4090_RESULTS)

    # Section 2: Throughput charts
    print_throughput_chart(RTX4090_RESULTS, metric="decode")
    print_throughput_chart(RTX4090_RESULTS, metric="prefill")

    # Section 3: Crossover detection
    find_crossover(RTX4090_RESULTS)

    # Section 4: Cost efficiency
    print_cost_efficiency(RTX4090_RESULTS)

    # Section 5: llama-bench CSV parsing
    print_llamabench_analysis(SAMPLE_LLAMABENCH_CSV)

    # Section 6: Checklist validator — incomplete spec
    bad_spec = BenchmarkSpec(
        model_name="Llama-3-8B",
        quantization="Q4_K_M",
        gpu_model="RTX 4090",
        # missing: cuda_version, vllm_version, warmup_iters, repetitions, etc.
        batch_sizes=[1, 4],
        metrics_reported=["throughput"],
        clock_locked=False,
    )
    print(f"\n>>> Incomplete benchmark spec:")
    print_checklist(bad_spec)

    # Complete spec
    good_spec = BenchmarkSpec(
        model_name="Llama-3-8B",
        quantization="Q4_K_M",
        gpu_model="RTX 4090",
        cuda_version="12.4",
        vllm_version="0.4.3",
        llamacpp_commit="abc1234",
        warmup_iters=10,
        repetitions=5,
        batch_sizes=[1, 4, 16, 32],
        prompt_source="ShareGPT",
        metrics_reported=["TTFT", "ITL", "throughput", "VRAM"],
        clock_locked=True,
    )
    print(f"\n>>> Complete benchmark spec:")
    print_checklist(good_spec)

    # Section 7: Latency percentile analysis — simulated TTFT samples
    random.seed(17)
    ttft_samples = [random.gauss(280, 45) for _ in range(200)]
    ttft_samples += [random.gauss(800, 120) for _ in range(10)]   # 5% spikes
    analyze_latencies(ttft_samples, "TTFT (simulated online benchmark)")

    # Section 8: Performance model
    print_performance_model()


if __name__ == "__main__":
    main()

```

## C++ — `benchmark_demo.cpp`

```cpp
// benchmark_demo.cpp — Chapter 17: Benchmarking vLLM vs llama.cpp
//
// Compile:  g++ -std=c++17 -O2 -Wall -o benchmark_demo benchmark_demo.cpp
// Run:      ./benchmark_demo
//
// Mirrors benchmark_demo.py: throughput/latency stats, ASCII charts,
// crossover detection, cost-efficiency table, roofline model.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

static std::string repeat_str(const std::string& s, int n) {
    std::string out;
    out.reserve(s.size() * static_cast<size_t>(n > 0 ? n : 0));
    for (int i = 0; i < n; ++i) out += s;
    return out;
}

// Format double with fixed precision into a fresh stream (avoids sticky flags)
static std::string fmt(double v, int prec = 1) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

// Format integer with thousands separator
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

static std::string box_line(int w, char c = '=') { return repeat_str(std::string(1, c), w); }

static void print_header(const std::string& title, int w = 62) {
    std::cout << box_line(w) << "\n";
    int pad = std::max(0, (w - static_cast<int>(title.size()) - 2) / 2);
    std::cout << repeat_str(" ", pad) << " " << title << "\n";
    std::cout << box_line(w) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Data model
// ─────────────────────────────────────────────────────────────────────────────

struct BenchResult {
    std::string engine;      // "vLLM" or "llama.cpp"
    int         batch;
    double      prefill_tps; // tokens/s (prompt processing)
    double      ttft_ms;     // time-to-first-token (ms)
    double      decode_tps;  // tokens/s (generation)
    double      itl_ms;      // inter-token latency (ms)
    double      vram_gb;
};

static std::vector<BenchResult> RTX4090_RESULTS = {
    // engine,      batch, prefill, ttft,   decode, itl,    vram
    {"llama.cpp",   1,   2012.3,  254.0,    64.1, 15.60,  5.8},
    {"vLLM",        1,   1843.1,  278.0,    58.2, 17.20,  9.4},
    {"llama.cpp",   4,   4210.5,  121.0,   230.4, 17.40,  6.2},
    {"vLLM",        4,   5890.2,   87.0,   219.3, 18.30, 10.1},
    {"llama.cpp",  16,   6800.0,   75.0,   540.1, 29.60,  7.1},
    {"vLLM",       16,  12340.0,   41.0,   892.4, 18.00, 13.7},
    {"llama.cpp",  32,   7200.0,   71.0,   780.3, 41.10,  8.4},
    {"vLLM",       32,  19870.0,   26.0,  1635.0, 19.60, 18.2},
};

// ─────────────────────────────────────────────────────────────────────────────
// 1. Results table
// ─────────────────────────────────────────────────────────────────────────────

static void print_results_table(const std::vector<BenchResult>& results) {
    int W = 94;
    std::cout << "\n" << box_line(W) << "\n";
    std::cout << "  Benchmark Results: Llama-3-8B Q4_K_M  |  RTX 4090"
                 "  |  512 input / 256 output tokens\n";
    std::cout << box_line(W) << "\n";

    // Header
    std::cout
        << std::left
        << "  " << std::setw(6)  << "Batch"
        << std::setw(12) << "Engine"
        << std::right
        << std::setw(12) << "Prefill"
        << std::setw(10) << "TTFT"
        << std::setw(12) << "Decode"
        << std::setw(10) << "ITL"
        << std::setw(10) << "VRAM"
        << std::setw(12) << "Winner"
        << "\n";
    std::cout << "  " << repeat_str("-", 5) << "  " << repeat_str("-", 10)
              << "  " << repeat_str("-", 10) << "  " << repeat_str("-", 8)
              << "  " << repeat_str("-", 10) << "  " << repeat_str("-", 8)
              << "  " << repeat_str("-", 8) << "  " << repeat_str("-", 10)
              << "\n";

    // Rows (pairs by batch)
    std::vector<int> batches = {1, 4, 16, 32};
    for (int b : batches) {
        // Collect pair
        BenchResult llama_r{}, vllm_r{};
        for (auto& r : results) {
            if (r.batch == b) {
                if (r.engine == "llama.cpp") llama_r = r;
                else                         vllm_r  = r;
            }
        }
        // Winner by decode throughput
        bool llama_wins = llama_r.decode_tps > vllm_r.decode_tps;
        for (int pass = 0; pass < 2; ++pass) {
            const BenchResult& r = (pass == 0) ? llama_r : vllm_r;
            bool winner = (pass == 0) ? llama_wins : !llama_wins;
            std::cout
                << "  " << std::left << std::setw(6)  << (pass == 0 ? std::to_string(b) : "")
                << std::setw(12) << r.engine
                << std::right
                << std::setw(11) << (fmt_int(static_cast<long long>(r.prefill_tps)) + "/s")
                << std::setw(10) << (fmt(r.ttft_ms, 0) + "ms")
                << std::setw(11) << (fmt_int(static_cast<long long>(r.decode_tps)) + "/s")
                << std::setw(10) << (fmt(r.itl_ms, 1) + "ms")
                << std::setw(9)  << (fmt(r.vram_gb, 1) + "GB")
                << std::setw(12) << (winner ? "<-" : "")
                << "\n";
        }
        std::cout << "\n";
    }
    std::cout << box_line(W) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. ASCII bar chart
// ─────────────────────────────────────────────────────────────────────────────

enum class Metric { DECODE, PREFILL };

static void print_throughput_chart(const std::vector<BenchResult>& results,
                                   Metric metric = Metric::DECODE) {
    int W = 72;
    std::string title = (metric == Metric::DECODE)
        ? "Decode Throughput vs Batch Size  (tok/s)"
        : "Prefill Throughput vs Batch Size  (tok/s)";

    std::cout << "\n" << box_line(W) << "\n";
    std::cout << "  " << title << "\n";
    std::cout << box_line(W) << "\n\n";

    // Find max
    double max_val = 0;
    for (auto& r : results)
        max_val = std::max(max_val, (metric == Metric::DECODE) ? r.decode_tps : r.prefill_tps);

    int bar_w = 50;

    for (const std::string eng : {"llama.cpp", "vLLM"}) {
        char sym = (eng == "llama.cpp") ? '*' : '#';
        std::cout << "  " << eng << "  [" << sym << "]\n";
        for (auto& r : results) {
            if (r.engine != eng) continue;
            double val = (metric == Metric::DECODE) ? r.decode_tps : r.prefill_tps;
            int bar = static_cast<int>(std::round(val / max_val * bar_w));
            bar = std::max(1, bar);
            std::string bar_str(static_cast<size_t>(bar), sym);
            std::cout << "    batch=" << std::setw(2) << r.batch << "  "
                      << std::left << std::setw(bar_w + 2) << bar_str
                      << std::right << fmt_int(static_cast<long long>(val)) << " tok/s\n";
        }
        std::cout << "\n";
    }
    std::cout << "  Scale: max=" << fmt_int(static_cast<long long>(max_val)) << " tok/s\n";
    std::cout << box_line(W) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Crossover detection
// ─────────────────────────────────────────────────────────────────────────────

static void find_crossover(const std::vector<BenchResult>& results) {
    std::cout << "\n" << box_line(62) << "\n";
    std::cout << "  Performance Crossover Analysis\n";
    std::cout << box_line(62) << "\n";

    // Gather decode_tps per engine per batch (sorted)
    struct Pair { int batch; double llama; double vllm; };
    std::vector<Pair> pairs;
    for (int b : {1, 4, 16, 32}) {
        Pair p{b, 0, 0};
        for (auto& r : results) {
            if (r.batch != b) continue;
            if (r.engine == "llama.cpp") p.llama = r.decode_tps;
            else                         p.vllm  = r.decode_tps;
        }
        pairs.push_back(p);
    }

    // Find first batch where vLLM overtakes
    int cross_batch = -1;
    for (auto& p : pairs) {
        if (p.vllm > p.llama) { cross_batch = p.batch; break; }
    }

    if (cross_batch < 0) {
        std::cout << "  vLLM never overtakes llama.cpp in this dataset.\n";
    } else {
        for (auto& p : pairs) {
            if (p.batch == cross_batch) {
                double adv = p.vllm / p.llama;
                std::cout << "  vLLM first overtakes llama.cpp at batch=" << cross_batch << "\n";
                std::cout << "  At crossover:\n";
                std::cout << "    llama.cpp : " << std::setw(8) << fmt_int(static_cast<long long>(p.llama)) << " tok/s\n";
                std::cout << "    vLLM      : " << std::setw(8) << fmt_int(static_cast<long long>(p.vllm))  << " tok/s\n";
                std::cout << "    vLLM advantage: " << fmt(adv, 2) << "x\n";
            }
        }
    }

    // Ratio bar chart
    std::cout << "\n  Decode throughput ratio (vLLM / llama.cpp):\n";
    for (auto& p : pairs) {
        double ratio = p.vllm / p.llama;
        int bar = static_cast<int>(std::round(ratio * 10));
        bar = std::max(1, bar);
        std::string bar_str(static_cast<size_t>(bar), '#');
        std::string label = (ratio >= 1.0) ? "<- vLLM ahead" : "<- llama.cpp ahead";
        std::cout << "    batch=" << std::setw(2) << p.batch << "  "
                  << fmt(ratio, 2) << "x  "
                  << std::left << std::setw(22) << bar_str
                  << std::right << label << "\n";
    }
    std::cout << box_line(62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Cost efficiency
// ─────────────────────────────────────────────────────────────────────────────

static void print_cost_efficiency(const std::vector<BenchResult>& results,
                                  double kwh_cost = 0.10,
                                  double gpu_cost = 1599.0) {
    // TDP assumptions: llama.cpp ~370W, vLLM ~450W (RTX 4090)
    auto tdp = [](const std::string& eng) {
        return eng == "llama.cpp" ? 370.0 : 450.0;
    };

    std::cout << "\n" << box_line(72) << "\n";
    std::cout << "  Cost Efficiency  |  $" << fmt(kwh_cost, 2) << "/kWh"
              << "  |  GPU cost $" << fmt_int(static_cast<long long>(gpu_cost)) << "\n";
    std::cout << box_line(72) << "\n";
    std::cout
        << "  " << std::left  << std::setw(6)  << "Batch"
        << std::setw(12) << "Engine"
        << std::right
        << std::setw(11) << "tok/s/W"
        << std::setw(12) << "tok/$0.01"
        << std::setw(16) << "GPU payback"
        << "\n";
    std::cout << "  " << repeat_str("-", 5) << "  " << repeat_str("-", 10)
              << "  " << repeat_str("-", 9)  << "  " << repeat_str("-", 10)
              << "  " << repeat_str("-", 13) << "\n";

    for (int b : {1, 4, 16, 32}) {
        bool first = true;
        for (const std::string eng : {"llama.cpp", "vLLM"}) {
            for (auto& r : results) {
                if (r.engine != eng || r.batch != b) continue;
                double w         = tdp(eng);
                double tps_per_w = r.decode_tps / w;
                // tok per $0.01 of electricity: $0.01 / (kwh_cost/3600 * w / 1000)
                double sec_per_cent = 0.01 / (kwh_cost / 3600.0 * w / 1000.0);
                double tok_per_cent = r.decode_tps * sec_per_cent;
                // payback hours: gpu_cost / ($0.01 per tok / 1000 * 1000 revenue per hour)
                // assume $0.01 per 1000 output tokens revenue
                double tok_per_hour = r.decode_tps * 3600.0;
                double rev_per_hour = tok_per_hour / 1000.0 * 0.01;
                double payback_h    = gpu_cost / rev_per_hour;

                std::cout
                    << "  " << std::left  << std::setw(6)  << (first ? std::to_string(b) : "")
                    << std::setw(12) << eng
                    << std::right
                    << std::setw(11) << fmt(tps_per_w, 2)
                    << std::setw(12) << fmt_int(static_cast<long long>(tok_per_cent))
                    << std::setw(13) << (fmt_int(static_cast<long long>(payback_h)) + " h")
                    << "\n";
                first = false;
            }
        }
        std::cout << "\n";
    }
    std::cout << box_line(72) << "\n";
    std::cout << "  Note: payback assumes $0.01/1000 output tokens revenue\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. llama-bench output parser simulation
// ─────────────────────────────────────────────────────────────────────────────

struct LlamaBenchRow {
    std::string test;   // e.g. "pp512", "tg1", "tg4"
    double      mean_tps;
    double      stddev_tps;
};

// Simulate parsing llama-bench CSV output
static std::vector<LlamaBenchRow> make_llamabench_rows() {
    // In practice these come from:
    //   llama-bench -m model.gguf -p 512 -n 256 -b 1,4,16,32 --output csv
    return {
        {"pp512",  2012.3, 18.2},
        {"tg1",      64.1,  0.8},
        {"tg4",     230.4,  2.1},
        {"tg16",    540.1,  4.8},
        {"tg32",    780.3,  7.2},
    };
}

static void print_llamabench_analysis(const std::vector<LlamaBenchRow>& rows) {
    std::cout << "\n" << box_line(62) << "\n";
    std::cout << "  llama-bench Output Analysis\n";
    std::cout << box_line(62) << "\n\n";

    // Prefill rows (pp*)
    std::cout << "  Prefill (prompt processing):\n";
    for (auto& r : rows) {
        if (r.test.substr(0, 2) != "pp") continue;
        int tokens = std::stoi(r.test.substr(2));
        double ttft = tokens / r.mean_tps * 1000.0;
        std::cout << "    " << std::setw(8) << r.test << "  "
                  << fmt_int(static_cast<long long>(r.mean_tps)) << " +/- "
                  << fmt(r.stddev_tps, 1) << " tok/s"
                  << "  ->  TTFT ~" << fmt(ttft, 0) << " ms\n";
    }

    // Decode rows (tg*)
    std::cout << "\n  Decode (token generation):\n";
    std::cout << "    " << std::left << std::setw(7) << "Batch"
              << std::right << std::setw(9) << "tok/s"
              << std::setw(8) << "+/-"
              << std::setw(10) << "ITL(ms)" << "\n";
    std::cout << "    " << repeat_str("-", 6) << "  "
              << repeat_str("-", 7) << "  "
              << repeat_str("-", 6) << "  "
              << repeat_str("-", 8) << "\n";

    double base_tps = 0;
    for (auto& r : rows) {
        if (r.test.substr(0, 2) != "tg") continue;
        int batch = std::stoi(r.test.substr(2));
        double itl_ms = 1000.0 / r.mean_tps;
        if (batch == 1) base_tps = r.mean_tps;
        std::cout << "    " << std::left  << std::setw(7) << batch
                  << std::right
                  << std::setw(9) << fmt(r.mean_tps, 1)
                  << std::setw(8) << fmt(r.stddev_tps, 1)
                  << std::setw(10) << fmt(itl_ms, 2)
                  << "\n";
    }

    // Scaling efficiency
    std::cout << "\n  Batch scaling efficiency (vs tg1):\n";
    for (auto& r : rows) {
        if (r.test.substr(0, 2) != "tg") continue;
        int batch = std::stoi(r.test.substr(2));
        if (base_tps <= 0) continue;
        // Perfect scaling would be batch * base_tps
        double perfect = base_tps * batch;
        double eff = r.mean_tps / perfect * 100.0;
        int bar = static_cast<int>(std::round(eff / 5.0));
        bar = std::max(1, bar);
        std::cout << "    tg" << std::setw(2) << batch << "  "
                  << fmt(eff, 1) << "%  "
                  << std::string(static_cast<size_t>(bar), '#') << "\n";
    }
    std::cout << box_line(62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Benchmark methodology checklist
// ─────────────────────────────────────────────────────────────────────────────

struct BenchmarkSpec {
    std::string model_path;
    std::string quantization;
    std::string hardware;
    std::string cuda_version;
    std::string driver_version;
    std::string prompt_distribution;
    int         warmup_iters;
    int         repetitions;
    std::vector<std::string> reported_metrics;
    std::vector<int>         batch_sizes;
    bool                     clocks_locked;
};

static void validate_benchmark_spec(const BenchmarkSpec& spec) {
    std::cout << "\n" << box_line(62) << "\n";
    std::cout << "  Benchmark Methodology Checklist\n";
    std::cout << box_line(62) << "\n";

    bool ok = true;

    auto check = [&](bool cond, const std::string& msg) {
        if (!cond) {
            std::cout << "  !!  MISSING: " << msg << "\n";
            ok = false;
        }
    };

    check(!spec.cuda_version.empty(),      "CUDA version not pinned");
    check(!spec.prompt_distribution.empty(), "Prompt distribution not described");
    check(spec.warmup_iters > 0,           "Warmup iterations not specified");
    check(spec.repetitions  > 0,           "Number of repetitions not specified");

    // Check required metrics present
    std::vector<std::string> required_metrics = {"TTFT", "ITL", "VRAM"};
    for (auto& m : required_metrics) {
        bool found = std::find(spec.reported_metrics.begin(),
                               spec.reported_metrics.end(), m)
                     != spec.reported_metrics.end();
        if (!found) {
            std::cout << "  !   INCOMPLETE METRICS: " << m << " not reported\n";
            ok = false;
        }
    }

    if (!spec.clocks_locked) {
        std::cout << "  !   WARN: GPU clocks not locked -- thermal throttle risk\n";
        ok = false;
    }

    // Check for batch >= 16
    bool has_large_batch = false;
    for (int b : spec.batch_sizes) if (b >= 16) has_large_batch = true;
    if (!has_large_batch) {
        std::cout << "  !   WARN: no batch >= 16 -- misses vLLM throughput advantage\n";
        ok = false;
    }

    if (ok) std::cout << "  OK  Benchmark specification is complete and sound.\n";

    std::cout << box_line(62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Latency distribution analysis
// ─────────────────────────────────────────────────────────────────────────────

static double percentile(std::vector<double>& sorted_vals, double p) {
    if (sorted_vals.empty()) return 0;
    double idx = p / 100.0 * (sorted_vals.size() - 1);
    int lo = static_cast<int>(std::floor(idx));
    int hi = static_cast<int>(std::ceil(idx));
    double frac = idx - lo;
    return sorted_vals[static_cast<size_t>(lo)] * (1 - frac)
         + sorted_vals[static_cast<size_t>(hi)] * frac;
}

static void analyze_latencies() {
    // Simulate online benchmark TTFT samples (ms)
    std::mt19937 rng(42);
    // Most requests are fast; heavy tail from KV cache pressure
    std::lognormal_distribution<double> dist(5.6, 0.4);  // log(270ms) ~ 5.6

    std::vector<double> samples;
    samples.reserve(210);
    for (int i = 0; i < 210; ++i)
        samples.push_back(dist(rng));

    // Clip extremes
    for (auto& s : samples) s = std::max(80.0, std::min(2000.0, s));

    std::sort(samples.begin(), samples.end());

    double mean   = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double sq_sum = 0;
    for (auto& s : samples) sq_sum += (s - mean) * (s - mean);
    double stddev = std::sqrt(sq_sum / samples.size());
    double cv     = stddev / mean * 100.0;
    double p50    = percentile(samples, 50);
    double p95    = percentile(samples, 95);
    double p99    = percentile(samples, 99);

    std::cout << "\n" << box_line(62) << "\n";
    std::cout << "  TTFT Distribution  (simulated online benchmark, n=" << samples.size() << ")\n";
    std::cout << box_line(62) << "\n";
    std::cout << "  Mean   : " << fmt(mean,   2) << " ms\n";
    std::cout << "  Median : " << fmt(p50,    2) << " ms\n";
    std::cout << "  StdDev : " << fmt(stddev, 2) << " ms  (CV=" << fmt(cv, 1) << "%)\n";
    std::cout << "  P95    : " << fmt(p95,    2) << " ms\n";
    std::cout << "  P99    : " << fmt(p99,    2) << " ms\n";
    std::cout << "  Min    : " << fmt(samples.front(), 2) << " ms\n";
    std::cout << "  Max    : " << fmt(samples.back(),  2) << " ms\n";

    if (cv > 10.0)
        std::cout << "\n  ** CV=" << fmt(cv, 1) << "% > 10% -- high variance."
                     " Check for thermal throttle or OS noise.\n";

    std::cout << box_line(62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Roofline model
// ─────────────────────────────────────────────────────────────────────────────

struct GpuSpec {
    std::string name;
    double mem_bw_tbps;     // memory bandwidth (TB/s)
    double flops_tflops;    // BF16 tensor core TFLOP/s
};

static void predict_throughput() {
    // Llama-3-8B: ~16B parameters, BF16 => 32 GB weights => ~16e9 * 2 bytes
    // Decode arithmetic intensity = 2*params / (2*params + kv_cache_bytes) ~ 1 flop/byte
    double params_bytes = 8e9 * 2.0;  // 8B params, BF16

    std::vector<GpuSpec> gpus = {
        {"RTX 4090",   1.008,  82.6},
        {"A100-80",    2.000, 312.0},
        {"H100-80",    3.350, 989.0},
    };

    std::cout << "\n" << box_line(72) << "\n";
    std::cout << "  Roofline Performance Model  |  Llama-3-8B BF16  |  batch=1\n";
    std::cout << box_line(72) << "\n";
    std::cout
        << "  " << std::left  << std::setw(14) << "GPU"
        << std::right
        << std::setw(14) << "Decode/GPU"
        << std::setw(10) << "ITL(ms)"
        << std::setw(14) << "Prefill"
        << std::setw(14) << "Ridge(tok)"
        << "\n";
    std::cout << "  " << repeat_str("-", 12) << "  "
              << repeat_str("-", 12) << "  "
              << repeat_str("-", 7) << "  "
              << repeat_str("-", 12) << "  "
              << repeat_str("-", 12) << "\n";

    for (auto& gpu : gpus) {
        // Decode (batch=1): bandwidth-bound
        // Each forward pass loads all weights once => params_bytes
        double decode_tps = gpu.mem_bw_tbps * 1e12 / params_bytes;
        double itl_ms     = 1000.0 / decode_tps;

        // Prefill: compute-bound above ridge
        // ops per token ~ 2 * params_bytes / 2 = params_bytes (flops, not bytes)
        double flops_per_tok = params_bytes;           // 2 * param_count * bytes_per_param / 2
        double prefill_tps   = gpu.flops_tflops * 1e12 / flops_per_tok;

        // Ridge point (tokens): where arithmetic intensity = mem_bw/flops
        double ridge_tokens = static_cast<long long>(gpu.flops_tflops * 1e12 /
                              (gpu.mem_bw_tbps * 1e12));

        std::cout
            << "  " << std::left  << std::setw(14) << gpu.name
            << std::right
            << std::setw(12) << (fmt_int(static_cast<long long>(decode_tps)) + "/s")
            << std::setw(10) << fmt(itl_ms, 1)
            << std::setw(12) << (fmt_int(static_cast<long long>(prefill_tps)) + "/s")
            << std::setw(14) << fmt_int(static_cast<long long>(ridge_tokens))
            << "\n";
    }
    std::cout << box_line(72) << "\n";
    std::cout << "  Note: actual performance ~70-85% of roofline"
                 " (kernel overhead, non-weight ops)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    print_header("Chapter 17 - Benchmarking: Companion Demo");

    print_results_table(RTX4090_RESULTS);

    print_throughput_chart(RTX4090_RESULTS, Metric::DECODE);
    print_throughput_chart(RTX4090_RESULTS, Metric::PREFILL);

    find_crossover(RTX4090_RESULTS);

    print_cost_efficiency(RTX4090_RESULTS);

    auto bench_rows = make_llamabench_rows();
    print_llamabench_analysis(bench_rows);

    // Incomplete spec — should flag issues
    BenchmarkSpec incomplete{
        "llama-3-8b-q4.gguf", "Q4_K_M", "RTX 4090",
        "", "", "", 0, 0,
        {"TTFT"},
        {1, 4},
        false
    };
    std::cout << "\n>>> Incomplete spec:";
    validate_benchmark_spec(incomplete);

    // Complete spec — should pass
    BenchmarkSpec complete{
        "llama-3-8b-q4.gguf", "Q4_K_M", "RTX 4090",
        "12.4", "550.90", "ShareGPT, 512 in / 256 out", 3, 10,
        {"TTFT", "ITL", "VRAM", "throughput"},
        {1, 4, 16, 32},
        true
    };
    std::cout << "\n>>> Complete spec:";
    validate_benchmark_spec(complete);

    analyze_latencies();

    predict_throughput();

    return 0;
}

```

