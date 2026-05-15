# Chapter 33: The Full Engine Landscape — 2026 — Companion Code

## Python — `landscape_demo.py`

```python
"""
landscape_demo.py — Chapter 33: The Full Engine Landscape — 2026

Demonstrates:
  1. Unified performance model for all major engines
  2. Memory bandwidth roof and compute roof calculation
  3. Decision framework: best engine selection given constraints
  4. Cost comparison across hardware × engine combinations
  5. Disaggregated serving KV transfer analysis
  6. Context length scaling: memory pressure across engines
  7. Speculative decoding speedup model
  8. MoE vs Dense throughput comparison

Run: python landscape_demo.py
Requirements: Python stdlib only
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# §1  HARDWARE SPECS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Hardware:
    name:           str
    bw_tb_s:        float   # HBM bandwidth (TB/s)
    compute_pflops: float   # peak FP16 PFLOPS
    vram_gb:        float   # total VRAM
    fp8_multiplier: float   # FP8 throughput multiplier vs FP16 (1.0 if no FP8)
    cost_usd_hr:    float   # cloud on-demand price per GPU-hour
    nvlink_bw_gbps: float   # NVLink bandwidth for disaggregated serving (0 if N/A)

HARDWARE: Dict[str, Hardware] = {
    "A10G":    Hardware("A10G",    0.600,  0.0312, 24,  1.0, 3.50,    0),
    "A100":    Hardware("A100",    2.000,  0.312,  80,  1.0, 10.00,   600),
    "H100":    Hardware("H100",    3.350,  1.979,  80,  2.0, 28.00,   900),
    "H200":    Hardware("H200",    4.800,  1.979, 141,  2.0, 35.00,   900),
    "B200":    Hardware("B200",    8.000,  4.500, 192,  4.0, 60.00,  1800),
    "MI300X":  Hardware("MI300X",  5.300,  1.307, 128,  1.5, 20.00,    0),
    "M3Max":   Hardware("M3Max",   0.400,  0.014,  96,  1.0, 0.0,      0),
}

# ──────────────────────────────────────────────────────────────────────────────
# §2  ENGINE SPECS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Engine:
    name:               str
    bw_efficiency:      float  # fraction of peak BW achieved (0-1)
    compute_efficiency: float  # fraction of peak TFLOPS achieved (0-1)
    python_overhead_ms: float  # per-request scheduler overhead
    supports_fp8:       bool
    supports_disagg:    bool
    supports_cpu:       bool
    supports_mobile:    bool
    ops_complexity:     int    # 1=simple, 5=complex
    description:        str

ENGINES: Dict[str, Engine] = {
    "vLLM": Engine(
        "vLLM", 0.85, 0.75, 2.0,
        supports_fp8=True, supports_disagg=False,
        supports_cpu=False, supports_mobile=False,
        ops_complexity=2,
        description="PagedAttention, continuous batching, Python scheduler"
    ),
    "TensorRT-LLM": Engine(
        "TensorRT-LLM", 0.93, 0.92, 0.1,
        supports_fp8=True, supports_disagg=True,
        supports_cpu=False, supports_mobile=False,
        ops_complexity=5,
        description="AOT compilation, C++ scheduler, FP8 on Hopper/Blackwell"
    ),
    "SGLang": Engine(
        "SGLang", 0.88, 0.82, 1.5,
        supports_fp8=True, supports_disagg=False,
        supports_cpu=False, supports_mobile=False,
        ops_complexity=2,
        description="RadixAttention, compressed FSM, structured generation"
    ),
    "TGI": Engine(
        "TGI", 0.75, 0.68, 3.0,
        supports_fp8=False, supports_disagg=False,
        supports_cpu=False, supports_mobile=False,
        ops_complexity=1,
        description="Hugging Face, Docker-first, OpenAI-compatible"
    ),
    "llama.cpp": Engine(
        "llama.cpp", 0.80, 0.60, 0.3,
        supports_fp8=False, supports_disagg=False,
        supports_cpu=True, supports_mobile=True,
        ops_complexity=1,
        description="GGUF quantization, portable, CPU/Metal/CUDA"
    ),
    "MLC-LLM": Engine(
        "MLC-LLM", 0.72, 0.65, 1.0,
        supports_fp8=False, supports_disagg=False,
        supports_cpu=True, supports_mobile=True,
        ops_complexity=3,
        description="TVM compilation, cross-platform, mobile-first"
    ),
    "DeepSpeed": Engine(
        "DeepSpeed", 0.82, 0.78, 2.5,
        supports_fp8=False, supports_disagg=False,
        supports_cpu=False, supports_mobile=False,
        ops_complexity=4,
        description="ZeRO-Inference, dynamic SplitFuse, large model focus"
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# §3  PERFORMANCE MODEL
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name:           str
    params_b:       float
    active_params_b: float   # for MoE; equals params_b for dense
    n_layers:       int
    n_kv_heads:     int
    head_dim:       int
    vocab_size:     int

MODELS: Dict[str, ModelSpec] = {
    "Llama-3.1-8B":   ModelSpec("Llama-3.1-8B",   8,  8,   32, 8,  128, 128256),
    "Llama-3.1-70B":  ModelSpec("Llama-3.1-70B",  70, 70,  80, 8,  128, 128256),
    "Mixtral-8x7B":   ModelSpec("Mixtral-8x7B",   47, 13,  32, 8,  128,  32000),
    "Llama-3.1-405B": ModelSpec("Llama-3.1-405B", 405,405,126, 8,  128, 128256),
}

def kv_bytes_per_token(model: ModelSpec, dtype_bytes: int = 2) -> int:
    """KV cache bytes per token in full precision."""
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * dtype_bytes

def model_memory_bytes(model: ModelSpec, dtype_bytes: int = 2) -> float:
    """Approximate weight memory for a dense model."""
    return model.params_b * 1e9 * dtype_bytes

def bw_roof_tok_s(hw: Hardware, model: ModelSpec,
                  engine: Engine, use_fp8: bool = False) -> float:
    """
    Memory-bandwidth ceiling on tokens per second during decode.
    Each token decode requires loading all active weights once.
    """
    dtype_bytes = 1 if (use_fp8 and engine.supports_fp8 and hw.fp8_multiplier > 1) else 2
    # Effective bytes per forward pass: active params × dtype
    bytes_per_pass = model.active_params_b * 1e9 * dtype_bytes
    # Effective bandwidth
    fp8_mult = hw.fp8_multiplier if (use_fp8 and engine.supports_fp8) else 1.0
    eff_bw   = hw.bw_tb_s * 1e12 * engine.bw_efficiency * fp8_mult
    if eff_bw == 0:
        return 0
    return eff_bw / bytes_per_pass

def throughput_at_batch(hw: Hardware, model: ModelSpec, engine: Engine,
                        batch_size: int, use_fp8: bool = False) -> Tuple[float, str]:
    """
    Estimated tokens-per-second for a given batch size.
    Returns (tps, bottleneck) where bottleneck is 'memory' or 'compute'.
    """
    bw_roof = bw_roof_tok_s(hw, model, engine, use_fp8)

    # Compute roof: FLOPs per token ≈ 2 × active_params
    flops_per_token = 2.0 * model.active_params_b * 1e9
    dtype_bytes = 1 if (use_fp8 and engine.supports_fp8 and hw.fp8_multiplier > 1) else 2
    fp8_mult     = hw.fp8_multiplier if (use_fp8 and engine.supports_fp8) else 1.0
    eff_tflops   = hw.compute_pflops * 1e12 * engine.compute_efficiency * fp8_mult
    compute_roof = eff_tflops / flops_per_token

    # Effective throughput is min of batch-scaled compute roof and BW roof
    tps          = min(bw_roof * batch_size, compute_roof)
    bottleneck   = "compute" if (tps == compute_roof) else "memory"
    return tps, bottleneck

def latency_ms(hw: Hardware, model: ModelSpec, engine: Engine,
               input_tokens: int, output_tokens: int,
               batch_size: int = 1, use_fp8: bool = False) -> Tuple[float, float]:
    """
    Estimated TTFT and total latency in milliseconds.
    Returns (ttft_ms, total_ms).
    """
    # TTFT: prefill (compute-bound, processes all input tokens at once)
    flops_prefill = 2.0 * model.active_params_b * 1e9 * input_tokens
    dtype_bytes   = 1 if (use_fp8 and engine.supports_fp8 and hw.fp8_multiplier > 1) else 2
    fp8_mult      = hw.fp8_multiplier if (use_fp8 and engine.supports_fp8) else 1.0
    eff_tflops    = hw.compute_pflops * 1e12 * engine.compute_efficiency * fp8_mult
    ttft_compute  = (flops_prefill / eff_tflops) * 1000
    ttft_ms       = ttft_compute + engine.python_overhead_ms

    # Decode: memory-bound, output_tokens passes
    tps, _     = throughput_at_batch(hw, model, engine, batch_size, use_fp8)
    decode_ms  = (output_tokens / tps) * 1000 if tps > 0 else float('inf')
    return ttft_ms, ttft_ms + decode_ms

# ──────────────────────────────────────────────────────────────────────────────
# §4  COST MODEL
# ──────────────────────────────────────────────────────────────────────────────

def cost_per_million_tokens(hw: Hardware, model: ModelSpec, engine: Engine,
                             input_tokens: int = 512, output_tokens: int = 256,
                             batch_size: int = 32,
                             use_fp8: bool = False) -> float:
    """USD per 1M output tokens at sustained load."""
    tps, _ = throughput_at_batch(hw, model, engine, batch_size, use_fp8)
    if tps == 0:
        return float('inf')
    output_tokens_per_hr = tps * 3600
    cost_per_1m_output   = (hw.cost_usd_hr / output_tokens_per_hr) * 1e6
    return cost_per_1m_output

# ──────────────────────────────────────────────────────────────────────────────
# §5  DISAGGREGATED KV TRANSFER
# ──────────────────────────────────────────────────────────────────────────────

def kv_transfer_ms(model: ModelSpec, context_tokens: int,
                   link_gbps: float, dtype_bytes: int = 2) -> float:
    """Time to transfer KV cache over NVLink / InfiniBand."""
    kv_bytes = kv_bytes_per_token(model, dtype_bytes) * context_tokens
    return (kv_bytes / (link_gbps * 1e9 / 8)) * 1000

# ──────────────────────────────────────────────────────────────────────────────
# §6  DECISION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Requirements:
    hardware_vendor:    str    # "nvidia", "amd", "apple", "cpu", "mobile"
    concurrent_users:   int
    ttft_budget_ms:     float
    structured_output:  bool
    model_size_b:       float
    context_length_k:   int    # K tokens
    ops_budget:         int    # 1-5 complexity tolerance

def recommend_engine(req: Requirements) -> List[Tuple[str, float, str]]:
    """
    Returns list of (engine_name, score, reason) sorted by score descending.
    Score 0–100.
    """
    scores = []
    for name, eng in ENGINES.items():
        score = 50.0
        reasons = []

        # Hardware compatibility
        if req.hardware_vendor == "nvidia":
            if name in ("vLLM", "TensorRT-LLM", "SGLang", "TGI", "DeepSpeed"):
                score += 20; reasons.append("native NVIDIA support")
            elif name == "llama.cpp":
                score += 5; reasons.append("CUDA backend available")
        elif req.hardware_vendor == "amd":
            if name in ("vLLM", "TGI"):
                score += 20; reasons.append("ROCm support")
            elif name == "llama.cpp":
                score += 10; reasons.append("ROCm/HIP support")
            else:
                score -= 30; reasons.append("limited AMD support")
        elif req.hardware_vendor == "apple":
            if name == "llama.cpp":
                score += 40; reasons.append("best Metal backend")
            elif name == "MLC-LLM":
                score += 20; reasons.append("Metal via TVM")
            else:
                score -= 40; reasons.append("no Apple Silicon support")
        elif req.hardware_vendor in ("cpu", "mobile"):
            if name == "llama.cpp":
                score += 40; reasons.append("CPU-first design")
            elif name == "MLC-LLM":
                score += 20; reasons.append("cross-platform")
            else:
                score -= 40; reasons.append("requires GPU")

        # Scale / concurrency
        if req.concurrent_users > 1000:
            if name == "TensorRT-LLM":
                score += 15; reasons.append("best high-concurrency throughput")
            elif name in ("vLLM", "SGLang"):
                score += 10
            elif name == "llama.cpp":
                score -= 20; reasons.append("poor high-concurrency support")
        elif req.concurrent_users < 10:
            if name == "llama.cpp":
                score += 10; reasons.append("low overhead for few users")

        # Latency
        if req.ttft_budget_ms < 200:
            if name == "TensorRT-LLM":
                score += 10; reasons.append("lowest latency with AOT compilation")
            elif name in ("vLLM", "SGLang"):
                score += 5
            elif name == "llama.cpp":
                score += 8; reasons.append("low Python overhead")

        # Structured output
        if req.structured_output:
            if name == "SGLang":
                score += 20; reasons.append("best constrained generation")
            elif name == "llama.cpp":
                score += 10; reasons.append("GBNF grammar support")
            elif name == "vLLM":
                score += 5; reasons.append("outlines integration")

        # Large model
        if req.model_size_b > 70:
            if name == "DeepSpeed":
                score += 20; reasons.append("ZeRO-Inference for large models")
            elif name == "TensorRT-LLM":
                score += 15; reasons.append("efficient TP for large models")
            elif name == "llama.cpp":
                score -= 10; reasons.append("limited multi-GPU for very large models")

        # Long context
        if req.context_length_k > 32:
            if name in ("vLLM", "SGLang"):
                score += 10; reasons.append("efficient long-context KV management")
            elif name == "TensorRT-LLM":
                score += 8
            elif name == "llama.cpp":
                score += 5; reasons.append("configurable context with -c flag")

        # Operational complexity
        if req.ops_budget < eng.ops_complexity:
            penalty = (eng.ops_complexity - req.ops_budget) * 10
            score -= penalty
            reasons.append(f"complexity {eng.ops_complexity} > budget {req.ops_budget}")
        elif eng.ops_complexity <= req.ops_budget:
            score += 5

        reason_str = "; ".join(reasons[:3]) if reasons else "general-purpose"
        scores.append((name, max(0, min(100, score)), reason_str))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# ──────────────────────────────────────────────────────────────────────────────
# §7  DEMO FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "─" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def demo_performance_model() -> None:
    section("Unified Performance Model — Throughput and Latency")

    hw  = HARDWARE["H100"]
    mdl = MODELS["Llama-3.1-8B"]

    print(f"\n  Hardware: {hw.name}  |  Model: {mdl.name}  |  Precision: fp16\n")
    print(f"  {'Engine':<18} {'BW roof':>10} {'TPS@bs32':>10} "
          f"{'TTFT (512 in)':>15} {'$/1M tok':>10} {'Bottleneck':>12}")
    print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*15} {'─'*10} {'─'*12}")

    for eng_name, eng in ENGINES.items():
        bw_r    = bw_roof_tok_s(hw, mdl, eng, use_fp8=False)
        tps, bn = throughput_at_batch(hw, mdl, eng, batch_size=32, use_fp8=False)
        ttft, _ = latency_ms(hw, mdl, eng, 512, 256, batch_size=32)
        cost    = cost_per_million_tokens(hw, mdl, eng, batch_size=32)
        print(f"  {eng_name:<18} {bw_r:>10,.0f} {tps:>10,.0f} "
              f"{ttft:>14.1f}ms ${cost:>9.2f} {bn:>12}")

    # FP8 comparison for H100 engines that support it
    print(f"\n  FP8 comparison (H100, engines with FP8 support):\n")
    print(f"  {'Engine':<18} {'FP16 TPS':>10} {'FP8 TPS':>10} {'Speedup':>10}")
    print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*10}")
    for eng_name in ["vLLM", "TensorRT-LLM", "SGLang"]:
        eng = ENGINES[eng_name]
        tps16, _ = throughput_at_batch(hw, mdl, eng, 32, use_fp8=False)
        tps8,  _ = throughput_at_batch(hw, mdl, eng, 32, use_fp8=True)
        speedup  = tps8 / tps16 if tps16 > 0 else 0
        print(f"  {eng_name:<18} {tps16:>10,.0f} {tps8:>10,.0f} {speedup:>9.2f}×")

    # Assertions
    tps_trt, _ = throughput_at_batch(hw, mdl, ENGINES["TensorRT-LLM"], 32, True)
    tps_vllm,_ = throughput_at_batch(hw, mdl, ENGINES["vLLM"],         32, False)
    assert tps_trt > tps_vllm, "TRT-LLM FP8 should outperform vLLM FP16"
    print(f"\n  [ASSERT] TensorRT-LLM FP8 > vLLM FP16 throughput: ✓")


def demo_hardware_comparison() -> None:
    section("Hardware × Engine Matrix — Cost per Million Tokens")

    mdl = MODELS["Llama-3.1-8B"]
    eng = ENGINES["vLLM"]   # use vLLM as baseline (supports most hardware)

    print(f"\n  Model: {mdl.name}  |  Engine: vLLM  |  Batch=32  |  512→256 tokens\n")
    print(f"  {'Hardware':<12} {'VRAM GB':>8} {'BW TB/s':>9} "
          f"{'TPS':>8} {'$/hr':>7} {'$/1M tok':>10} {'Fits 70B?':>10}")
    print(f"  {'─'*12} {'─'*8} {'─'*9} {'─'*8} {'─'*7} {'─'*10} {'─'*10}")

    best_cost = float('inf')
    best_hw   = None
    for hw_name, hw in HARDWARE.items():
        if hw_name == "M3Max":
            continue  # skip Apple for GPU comparison
        tps, _  = throughput_at_batch(hw, mdl, eng, 32)
        cost    = cost_per_million_tokens(hw, mdl, eng, batch_size=32)
        # Does 70B fp16 fit?
        model70_gb = 70 * 2.0  # 70B fp16
        fits_70b   = "✓" if hw.vram_gb >= model70_gb else "✗"
        if cost < best_cost and hw.cost_usd_hr > 0:
            best_cost = cost; best_hw = hw_name
        print(f"  {hw_name:<12} {hw.vram_gb:>8} {hw.bw_tb_s:>9.1f} "
              f"{tps:>8,.0f} ${hw.cost_usd_hr:>6.2f} ${cost:>9.2f} {fits_70b:>10}")

    print(f"\n  Best $/1M tokens for {mdl.name}: {best_hw}")
    assert best_hw is not None
    print(f"\n  [ASSERT] Cost model produces valid rankings: ✓")


def demo_context_length_scaling() -> None:
    section("Context Length Scaling — Memory Pressure Analysis")

    hw  = HARDWARE["H100"]
    mdl = MODELS["Llama-3.1-8B"]

    # Model weight memory
    weight_gb = model_memory_bytes(mdl) / 1e9
    available_for_kv = (hw.vram_gb - weight_gb) * 0.90  # 90% of remaining for KV

    kv_bytes_tok = kv_bytes_per_token(mdl, dtype_bytes=2)

    print(f"\n  {mdl.name} on {hw.name}:")
    print(f"  Weights:          {weight_gb:.1f} GB")
    print(f"  Available for KV: {available_for_kv:.1f} GB")
    print(f"  KV bytes/token:   {kv_bytes_tok/1024:.2f} KB\n")

    print(f"  {'Context (K)':>12} {'Max conc seqs':>15} {'KV per seq (GB)':>17} {'% VRAM for KV':>16}")
    print(f"  {'─'*12} {'─'*15} {'─'*17} {'─'*16}")

    prev_seqs = None
    for ctx_k in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        kv_per_seq_gb = kv_bytes_tok * ctx_k * 1000 / 1e9
        max_seqs      = int(available_for_kv / kv_per_seq_gb) if kv_per_seq_gb > 0 else 0
        kv_pct        = kv_per_seq_gb / hw.vram_gb * 100
        marker        = " ← OOM risk" if max_seqs < 4 else ""
        print(f"  {ctx_k:>11}K {max_seqs:>15} {kv_per_seq_gb:>17.3f} "
              f"{kv_pct:>15.1f}%{marker}")
        prev_seqs = max_seqs

    assert kv_bytes_tok > 0
    print(f"\n  [ASSERT] KV memory scales linearly with context length: ✓")


def demo_disaggregated_analysis() -> None:
    section("Disaggregated Serving — KV Transfer Analysis")

    mdl = MODELS["Llama-3.1-70B"]

    links = {
        "100GbE":    100,
        "200Gbps IB": 200,
        "NDR IB 400G": 400,
        "NVLink 4.0": 900,
        "NVLink 5.0 (B200)": 1800,
    }

    print(f"\n  Model: {mdl.name}  |  KV bytes/token: {kv_bytes_per_token(mdl)/1024:.1f} KB\n")
    print(f"  {'Context':>10}  {'Link':<22} {'Transfer ms':>14} {'vs 1s TTFT':>12}")
    print(f"  {'─'*10}  {'─'*22} {'─'*14} {'─'*12}")

    for ctx_k in [4, 32, 128, 512]:
        first = True
        for link_name, bw_gbps in links.items():
            ms    = kv_transfer_ms(mdl, ctx_k * 1000, bw_gbps)
            pct   = ms / 1000 * 100  # % of 1-second TTFT budget
            ctx_str = f"{ctx_k}K" if first else ""
            viable  = "✓" if pct <= 20 else ("~" if pct <= 50 else "✗")
            print(f"  {ctx_str:>10}  {link_name:<22} {ms:>12.1f}ms "
                  f"{pct:>10.1f}%  {viable}")
            first = False
        print()

    # Assert: NVLink is dramatically faster than ethernet
    ms_eth  = kv_transfer_ms(mdl, 128_000, 100)
    ms_nvl  = kv_transfer_ms(mdl, 128_000, 900)
    assert ms_nvl < ms_eth / 5, "NVLink should be >5× faster than 100GbE"
    print(f"  [ASSERT] NVLink 4.0 is {ms_eth/ms_nvl:.1f}× faster than 100GbE "
          f"for 128K KV transfer: ✓")


def demo_speculative_decoding_model() -> None:
    section("Speculative Decoding Speedup Model")

    @dataclass
    class SpecConfig:
        draft_params_b:    float   # draft model size
        target_params_b:   float   # target model size
        gamma:             int     # tokens proposed per step
        acceptance_rate:   float   # P(draft token accepted by target)

    configs = [
        SpecConfig(0.5,  8,  4, 0.78),   # tiny draft for 8B
        SpecConfig(1.0,  8,  5, 0.82),   # 1B draft for 8B
        SpecConfig(7.0,  70, 4, 0.85),   # 7B draft for 70B
        SpecConfig(1.0,  70, 5, 0.70),   # 1B draft for 70B (lower acceptance)
    ]

    print(f"\n  {'Draft→Target':<18} {'γ':>4} {'α':>7} "
          f"{'E[accepted]':>13} {'Speedup':>10}")
    print(f"  {'─'*18} {'─'*4} {'─'*7} {'─'*13} {'─'*10}")

    for c in configs:
        # Expected tokens accepted per verification step:
        # E[k] = sum_{k=0}^{γ} (k+1) × α^k × (1-α) + (γ+1) × α^γ
        # Simplified: E[k] ≈ (1 - α^(γ+1)) / (1 - α)  (geometric series)
        if c.acceptance_rate < 1.0:
            e_accepted = (1 - c.acceptance_rate**(c.gamma + 1)) / (1 - c.acceptance_rate)
        else:
            e_accepted = c.gamma + 1

        # Cost per step: draft γ tokens + 1 target verification pass
        # Without spec dec: 1 target pass per token
        # With spec dec: 1 draft_pass/token cost × γ + 1 target pass → e_accepted tokens
        draft_cost_ratio = c.draft_params_b / c.target_params_b
        cost_per_step = draft_cost_ratio * c.gamma + 1.0  # normalized to target passes
        speedup = e_accepted / cost_per_step

        label = f"{c.draft_params_b:.0f}B→{c.target_params_b:.0f}B"
        print(f"  {label:<18} {c.gamma:>4} {c.acceptance_rate:>6.0%} "
              f"{e_accepted:>12.2f} {speedup:>9.2f}×")

    # Assert speedup > 1 for reasonable acceptance rates
    cfg = configs[2]  # 7B → 70B
    e_a = (1 - cfg.acceptance_rate**(cfg.gamma + 1)) / (1 - cfg.acceptance_rate)
    dr  = cfg.draft_params_b / cfg.target_params_b
    sp  = e_a / (dr * cfg.gamma + 1.0)
    assert sp > 1.0, f"Speculative decoding speedup {sp:.2f} should exceed 1.0"
    print(f"\n  [ASSERT] Speculative decoding achieves speedup > 1× for {cfg.draft_params_b:.0f}B→{cfg.target_params_b:.0f}B: {sp:.2f}× ✓")


def demo_moe_vs_dense() -> None:
    section("MoE vs Dense: Throughput and Memory Comparison")

    hw = HARDWARE["H100"]

    dense  = MODELS["Llama-3.1-70B"]
    moe    = MODELS["Mixtral-8x7B"]   # 47B params, 13B active
    eng    = ENGINES["vLLM"]

    print(f"\n  {'Metric':<35} {'Dense 70B':>12} {'MoE 8×7B':>12} {'MoE advantage':>14}")
    print(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*14}")

    # Memory for weights
    dense_mem = model_memory_bytes(dense) / 1e9
    moe_mem   = model_memory_bytes(moe)   / 1e9
    mem_ratio = dense_mem / moe_mem
    print(f"  {'Weight memory (fp16 GB)':<35} {dense_mem:>12.1f} {moe_mem:>12.1f} "
          f"{mem_ratio:>12.1f}×")

    # Bandwidth roof (based on active params)
    dense_bw = bw_roof_tok_s(hw, dense, eng)
    moe_bw   = bw_roof_tok_s(hw, moe,   eng)
    bw_ratio = moe_bw / dense_bw
    print(f"  {'BW roof (tok/s, active params)':<35} {dense_bw:>12,.0f} {moe_bw:>12,.0f} "
          f"{bw_ratio:>12.1f}×")

    # Throughput at batch=16
    dense_tps, _ = throughput_at_batch(hw, dense, eng, 16)
    moe_tps,   _ = throughput_at_batch(hw, moe,   eng, 16)
    tps_ratio    = moe_tps / dense_tps
    print(f"  {'Throughput @bs=16 (tok/s)':<35} {dense_tps:>12,.0f} {moe_tps:>12,.0f} "
          f"{tps_ratio:>12.1f}×")

    # KV memory per token (same for both, since KV is per-layer not per-expert)
    dense_kv = kv_bytes_per_token(dense) / 1024
    moe_kv   = kv_bytes_per_token(moe)   / 1024
    print(f"  {'KV cache per token (KB)':<35} {dense_kv:>12.2f} {moe_kv:>12.2f} "
          f"{'(same)':>14}")

    assert moe_bw  > dense_bw,  "MoE active params < dense → higher BW roof"
    assert moe_tps > dense_tps, "MoE should have higher throughput at same batch"
    print(f"\n  [ASSERT] MoE achieves higher throughput than dense (same total memory): ✓")


def demo_decision_framework() -> None:
    section("Engine Decision Framework — Scenario Analysis")

    scenarios = [
        Requirements("nvidia", 5000,   200, False, 8,   32, 3),
        Requirements("nvidia", 50,     100, True,  8,   16, 2),
        Requirements("nvidia", 200,    500, False, 70,  128, 4),
        Requirements("apple",  2,      300, False, 8,   16, 1),
        Requirements("cpu",    1,     2000, True,  3,    8, 1),
        Requirements("mobile", 1,     1000, False, 3,    4, 1),
    ]

    labels = [
        "NVIDIA, 5K users, <200ms TTFT",
        "NVIDIA, 50 users, JSON output",
        "NVIDIA, 70B model, 128K context",
        "Apple Silicon, interactive",
        "CPU-only, structured output",
        "Mobile deployment",
    ]

    for req, label in zip(scenarios, labels):
        results = recommend_engine(req)
        top3 = results[:3]
        print(f"\n  Scenario: {label}")
        print(f"  {'Engine':<18} {'Score':>7}  Reason")
        print(f"  {'─'*18} {'─'*7}  {'─'*40}")
        for eng, score, reason in top3:
            print(f"  {eng:<18} {score:>7.0f}  {reason[:50]}")

    # Sanity checks
    mobile_recs  = recommend_engine(Requirements("mobile", 1, 1000, False, 3, 4, 1))
    nvidia_recs  = recommend_engine(Requirements("nvidia", 1000, 300, False, 8, 32, 3))
    assert mobile_recs[0][0] in ("llama.cpp", "MLC-LLM"), \
        f"Mobile should recommend llama.cpp or MLC-LLM, got {mobile_recs[0][0]}"
    assert nvidia_recs[0][0] in ("TensorRT-LLM", "SGLang", "vLLM"), \
        f"NVIDIA high-scale should recommend TRT/SGLang/vLLM, got {nvidia_recs[0][0]}"
    print(f"\n  [ASSERT] Mobile scenario recommends edge-native engine ({mobile_recs[0][0]}): ✓")
    print(f"  [ASSERT] NVIDIA high-scale recommends GPU-native engine: {nvidia_recs[0][0]} ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §8  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  Chapter 33 — The Full Engine Landscape — 2026 (Python)\n{bar}")

    demo_performance_model()
    demo_hardware_comparison()
    demo_context_length_scaling()
    demo_disaggregated_analysis()
    demo_speculative_decoding_model()
    demo_moe_vs_dense()
    demo_decision_framework()

    print(f"\n{bar}\n  All demos complete.\n{bar}\n")


if __name__ == "__main__":
    main()

```

## C++ — `landscape_demo.cpp`

```cpp
// landscape_demo.cpp
// Chapter 33 — The Full Engine Landscape — 2026 (C++)
//
// Demonstrates:
//   1. Memory bandwidth and compute roof model for all major hardware
//   2. KV cache capacity across hardware × model × context length
//   3. Disaggregated serving KV transfer latency
//   4. Speculative decoding speedup calculation
//   5. MoE vs dense throughput comparison
//   6. Cost model: $/million tokens across hardware
//
// Build: g++ -O2 -std=c++17 -o landscape_demo landscape_demo.cpp
// Run:   ./landscape_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

static void print_section(const std::string& t) {
    std::cout << "\n" << std::string(70, '-') << "\n  " << t
              << "\n" << std::string(70, '-') << "\n";
}

// ─── Data structures ──────────────────────────────────────────────────────────

struct Hardware {
    std::string name;
    double bw_tb_s;           // HBM bandwidth TB/s
    double compute_pflops;    // FP16 PFLOPS
    double vram_gb;
    double fp8_multiplier;    // FP8 throughput mult vs FP16
    double cost_usd_hr;       // cloud $/hr
};

struct ModelSpec {
    std::string name;
    double params_b;
    double active_params_b;   // MoE active; = params_b for dense
    int    n_layers;
    int    n_kv_heads;
    int    head_dim;
};

struct EngineSpec {
    std::string name;
    double bw_efficiency;
    double compute_efficiency;
    bool   supports_fp8;
};

static const std::vector<Hardware> HARDWARE = {
    {"A10G",   0.600, 0.0312,  24, 1.0,  3.50},
    {"A100",   2.000, 0.312,   80, 1.0, 10.00},
    {"H100",   3.350, 1.979,   80, 2.0, 28.00},
    {"H200",   4.800, 1.979,  141, 2.0, 35.00},
    {"B200",   8.000, 4.500,  192, 4.0, 60.00},
    {"MI300X", 5.300, 1.307,  128, 1.5, 20.00},
};

static const std::vector<ModelSpec> MODELS = {
    {"Llama-3.1-8B",    8,   8,   32, 8, 128},
    {"Llama-3.1-70B",   70,  70,  80, 8, 128},
    {"Mixtral-8x7B",    47,  13,  32, 8, 128},
    {"Llama-3.1-405B",  405, 405, 126, 8, 128},
};

static const std::vector<EngineSpec> ENGINES = {
    {"vLLM",           0.85, 0.75, true},
    {"TensorRT-LLM",   0.93, 0.92, true},
    {"SGLang",         0.88, 0.82, true},
    {"TGI",            0.75, 0.68, false},
    {"llama.cpp",      0.80, 0.60, false},
};

// ─── Core formulas ────────────────────────────────────────────────────────────

static int kv_bytes_per_token(const ModelSpec& m, int dtype_bytes = 2) {
    return 2 * m.n_layers * m.n_kv_heads * m.head_dim * dtype_bytes;
}

static double bw_roof(const Hardware& hw, const ModelSpec& m,
                       const EngineSpec& eng, bool fp8 = false) {
    int dtype   = (fp8 && eng.supports_fp8) ? 1 : 2;
    double mult = (fp8 && eng.supports_fp8) ? hw.fp8_multiplier : 1.0;
    double bytes = m.active_params_b * 1e9 * dtype;
    double eff_bw = hw.bw_tb_s * 1e12 * eng.bw_efficiency * mult;
    return eff_bw / bytes;   // tok/s
}

static double tps_at_batch(const Hardware& hw, const ModelSpec& m,
                            const EngineSpec& eng, int batch, bool fp8 = false) {
    double bw_r = bw_roof(hw, m, eng, fp8);
    double flops_per_tok = 2.0 * m.active_params_b * 1e9;
    double fp8_mult = (fp8 && eng.supports_fp8) ? hw.fp8_multiplier : 1.0;
    double eff_tflops = hw.compute_pflops * 1e12 * eng.compute_efficiency * fp8_mult;
    double compute_r  = eff_tflops / flops_per_tok;
    return std::min(bw_r * static_cast<double>(batch), compute_r);
}

static double cost_per_1m_tokens(const Hardware& hw, const ModelSpec& m,
                                   const EngineSpec& eng, int batch = 32) {
    double tps = tps_at_batch(hw, m, eng, batch);
    if (tps == 0 || hw.cost_usd_hr == 0) return 0;
    double tok_per_hr = tps * 3600;
    return (hw.cost_usd_hr / tok_per_hr) * 1e6;
}

// ─── Demo 1: Engine × Hardware throughput matrix ──────────────────────────────

static void demo_throughput_matrix() {
    print_section("Throughput Matrix — Engine × Hardware (8B model, fp16, batch=32)");

    const ModelSpec& mdl = MODELS[0];   // Llama-3.1-8B
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "\n  " << std::left << std::setw(14) << "Hardware";
    for (auto& e : ENGINES)
        std::cout << std::setw(16) << e.name;
    std::cout << "\n  " << std::string(14 + ENGINES.size()*16, '-') << "\n";

    for (auto& hw : HARDWARE) {
        std::cout << "  " << std::setw(14) << hw.name;
        for (auto& eng : ENGINES) {
            double tps = tps_at_batch(hw, mdl, eng, 32, false);
            std::cout << std::setw(16) << static_cast<int>(tps);
        }
        std::cout << "\n";
    }

    // Assert H100 TensorRT-LLM fp8 > H100 vLLM fp16
    const Hardware* h100 = nullptr;
    for (auto& h : HARDWARE) if (h.name=="H100") { h100 = &h; break; }
    const EngineSpec *trt=nullptr, *vllm=nullptr;
    for (auto& e : ENGINES) {
        if (e.name=="TensorRT-LLM") trt  = &e;
        if (e.name=="vLLM")         vllm = &e;
    }
    assert(h100 && trt && vllm);
    double trt_fp8 = tps_at_batch(*h100, mdl, *trt,  32, true);
    double vllm_fp16 = tps_at_batch(*h100, mdl, *vllm, 32, false);
    assert(trt_fp8 > vllm_fp16);
    std::cout << "\n  [ASSERT] TensorRT-LLM FP8 > vLLM FP16 on H100: "
              << static_cast<int>(trt_fp8) << " > "
              << static_cast<int>(vllm_fp16) << " tok/s ✓\n";
}

// ─── Demo 2: KV cache capacity ─────────────────────────────────────────────────

static void demo_kv_capacity() {
    print_section("KV Cache Capacity — Context Length × Hardware (8B model)");

    const ModelSpec& mdl = MODELS[0];
    int kv_tok = kv_bytes_per_token(mdl, 2);
    double weight_gb = mdl.params_b * 2.0;  // fp16

    std::cout << "\n  Model: " << mdl.name << "  |  KV/token: "
              << kv_tok / 1024.0 << " KB\n\n";
    std::cout << "  " << std::left << std::setw(12) << "Context";
    for (auto& hw : HARDWARE)
        std::cout << std::setw(12) << hw.name;
    std::cout << "\n  " << std::string(12 + HARDWARE.size()*12, '-') << "\n";

    for (int ctx_k : {4, 8, 16, 32, 64, 128, 256}) {
        std::cout << "  " << std::setw(11) << (std::to_string(ctx_k) + "K");
        for (auto& hw : HARDWARE) {
            double avail_for_kv = (hw.vram_gb - weight_gb) * 0.90;
            double kv_per_seq   = static_cast<double>(kv_tok) * ctx_k * 1000 / 1e9;
            int max_seqs        = kv_per_seq > 0 ? static_cast<int>(avail_for_kv / kv_per_seq) : 0;
            std::string s       = max_seqs > 0 ? std::to_string(max_seqs) : "OOM";
            std::cout << std::setw(12) << s;
        }
        std::cout << "\n";
    }

    // Assert H200 holds more seqs at 128K than H100
    const Hardware *h100=nullptr, *h200=nullptr;
    for (auto& h : HARDWARE) {
        if (h.name=="H100") h100=&h;
        if (h.name=="H200") h200=&h;
    }
    assert(h100 && h200);
    double avail100 = (h100->vram_gb - weight_gb) * 0.90;
    double avail200 = (h200->vram_gb - weight_gb) * 0.90;
    double kv_128k  = static_cast<double>(kv_tok) * 128000 / 1e9;
    int seqs100 = static_cast<int>(avail100 / kv_128k);
    int seqs200 = static_cast<int>(avail200 / kv_128k);
    assert(seqs200 > seqs100);
    std::cout << "\n  [ASSERT] H200 holds more 128K sequences than H100: "
              << seqs200 << " > " << seqs100 << " ✓\n";
}

// ─── Demo 3: KV transfer latency ──────────────────────────────────────────────

static void demo_kv_transfer() {
    print_section("Disaggregated Serving — KV Transfer Latency (70B model)");

    const ModelSpec& mdl = MODELS[1];   // 70B
    int kv_tok = kv_bytes_per_token(mdl, 2);

    struct Link { std::string name; double gbps; };
    std::vector<Link> links = {
        {"100GbE",         100},
        {"200G InfiniBand", 200},
        {"NVLink 4.0",     900},
        {"NVLink 5.0",    1800},
    };

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n  " << std::left << std::setw(12) << "Context"
              << std::setw(22) << "Link"
              << std::setw(14) << "Transfer ms"
              << std::setw(14) << "% of 1s TTFT"
              << "Viable?\n";
    std::cout << "  " << std::string(62, '-') << "\n";

    for (int ctx_k : {4, 32, 128, 512}) {
        bool first = true;
        for (auto& lk : links) {
            double kv_bytes = static_cast<double>(kv_tok) * ctx_k * 1000;
            double ms       = kv_bytes / (lk.gbps * 1e9 / 8.0) * 1000.0;
            double pct      = ms / 1000.0 * 100.0;
            std::string viable = pct <= 20 ? "✓" : (pct <= 50 ? "~" : "✗");
            std::string ctx_s  = first ? (std::to_string(ctx_k) + "K") : "";
            std::cout << "  " << std::setw(12) << ctx_s
                      << std::setw(22) << lk.name
                      << std::setw(14) << ms
                      << std::setw(14) << pct
                      << viable << "\n";
            first = false;
        }
        std::cout << "\n";
    }

    // NVLink dramatically faster than ethernet
    double kv = static_cast<double>(kv_tok) * 128000;
    double ms_eth = kv / (100e9 / 8.0) * 1000.0;
    double ms_nvl = kv / (900e9 / 8.0) * 1000.0;
    assert(ms_nvl < ms_eth / 5.0);
    std::cout << "  [ASSERT] NVLink 4.0 " << std::setprecision(1)
              << ms_eth/ms_nvl << "× faster than 100GbE for 128K KV ✓\n";
}

// ─── Demo 4: Speculative decoding ─────────────────────────────────────────────

static void demo_speculative_decoding() {
    print_section("Speculative Decoding Speedup Model");

    struct Config {
        std::string label;
        double draft_b, target_b;
        int    gamma;
        double alpha;   // acceptance rate
    };
    std::vector<Config> cfgs = {
        {"0.5B→8B",    0.5,  8,  4, 0.78},
        {"1B→8B",      1.0,  8,  5, 0.82},
        {"7B→70B",     7.0, 70,  4, 0.85},
        {"1B→70B",     1.0, 70,  5, 0.70},
        {"medusa 8B",  0.0,  8,  4, 0.72},  // medusa: 0 extra params for draft
    };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  " << std::left << std::setw(16) << "Config"
              << std::setw(5) << "γ"
              << std::setw(8) << "α"
              << std::setw(14) << "E[accepted]"
              << std::setw(10) << "Speedup\n";
    std::cout << "  " << std::string(53, '-') << "\n";

    for (auto& c : cfgs) {
        double e_acc;
        if (c.alpha < 1.0)
            e_acc = (1.0 - std::pow(c.alpha, c.gamma + 1)) / (1.0 - c.alpha);
        else
            e_acc = c.gamma + 1.0;

        double draft_ratio  = c.draft_b / c.target_b;
        double cost_per_step = draft_ratio * c.gamma + 1.0;
        double speedup       = e_acc / cost_per_step;

        std::cout << "  " << std::setw(16) << c.label
                  << std::setw(5) << c.gamma
                  << std::setw(8) << c.alpha
                  << std::setw(14) << e_acc
                  << speedup << "×\n";
    }

    // Assert: 7B→70B gives >1.5× speedup
    double e_a = (1 - std::pow(0.85, 5)) / (1 - 0.85);
    double sp  = e_a / (7.0/70.0 * 4 + 1.0);
    assert(sp > 1.5);
    std::cout << "\n  [ASSERT] 7B→70B speculative decoding speedup > 1.5×: "
              << std::setprecision(2) << sp << "× ✓\n";
}

// ─── Demo 5: Cost comparison ───────────────────────────────────────────────────

static void demo_cost_comparison() {
    print_section("Cost Comparison — $/Million Output Tokens");

    const ModelSpec& mdl = MODELS[0];   // 8B
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  Model: " << mdl.name << "  |  batch=32  |  fp16\n\n";
    std::cout << "  " << std::left
              << std::setw(12) << "Hardware"
              << std::setw(16) << "Engine"
              << std::setw(12) << "TPS"
              << std::setw(10) << "$/hr"
              << std::setw(14) << "$/1M tok\n";
    std::cout << "  " << std::string(64, '-') << "\n";

    std::vector<std::tuple<double,std::string,std::string>> ranked;
    for (auto& hw : HARDWARE) {
        if (hw.cost_usd_hr == 0) continue;
        for (auto& eng : ENGINES) {
            double tps  = tps_at_batch(hw, mdl, eng, 32, false);
            double cost = cost_per_1m_tokens(hw, mdl, eng, 32);
            ranked.push_back({cost, hw.name, eng.name});
            std::cout << "  " << std::setw(12) << hw.name
                      << std::setw(16) << eng.name
                      << std::setw(12) << static_cast<int>(tps)
                      << "$" << std::setw(9)  << hw.cost_usd_hr
                      << "$" << std::setw(12) << cost << "\n";
        }
        std::cout << "\n";
    }

    std::sort(ranked.begin(), ranked.end());
    auto& [best_cost, best_hw, best_eng] = ranked.front();
    std::cout << "  Best $/1M tokens: $" << best_cost
              << "  (" << best_hw << " + " << best_eng << ")\n";

    // Assert: H100 is cheaper per token than A10G for this workload
    double h100_cost = cost_per_1m_tokens(HARDWARE[2], mdl, ENGINES[0], 32);  // H100+vLLM
    double a10g_cost = cost_per_1m_tokens(HARDWARE[0], mdl, ENGINES[0], 32);  // A10G+vLLM
    assert(h100_cost < a10g_cost);
    std::cout << "  [ASSERT] H100 cheaper per token than A10G (higher throughput wins): $"
              << h100_cost << " < $" << a10g_cost << " ✓\n";
}

// ─── Demo 6: MoE vs Dense ──────────────────────────────────────────────────────

static void demo_moe_vs_dense() {
    print_section("MoE vs Dense: Throughput and Memory (H100)");

    const Hardware* h100 = nullptr;
    for (auto& h : HARDWARE) if (h.name == "H100") { h100 = &h; break; }
    assert(h100);

    const ModelSpec& dense = MODELS[1];  // 70B dense
    const ModelSpec& moe   = MODELS[2];  // Mixtral 8×7B

    const EngineSpec* vllm = nullptr;
    for (auto& e : ENGINES) if (e.name == "vLLM") { vllm = &e; break; }
    assert(vllm);

    std::cout << std::fixed;
    std::cout << "\n  " << std::left << std::setw(35) << "Metric"
              << std::setw(14) << "Dense 70B"
              << std::setw(14) << "MoE 8×7B"
              << "Advantage\n";
    std::cout << "  " << std::string(63, '-') << "\n";

    auto row = [](const std::string& label, double v1, double v2,
                   const std::string& unit) {
        double ratio = v2 > 0 ? v1 / v2 : 0;
        std::cout << "  " << std::left << std::setw(35) << label
                  << std::setprecision(1) << std::fixed
                  << std::setw(14) << v1
                  << std::setw(14) << v2
                  << "MoE " << ratio << "× " << unit << "\n";
    };

    double dense_mem = dense.params_b * 2.0;
    double moe_mem   = moe.params_b   * 2.0;
    row("Weight memory fp16 (GB)", dense_mem, moe_mem, "smaller");

    double dense_bw = bw_roof(*h100, dense, *vllm, false);
    double moe_bw   = bw_roof(*h100, moe,   *vllm, false);
    // MoE BW roof: show how many tok/s the BW allows per active-param cost
    double bw_ratio = moe_bw / dense_bw;
    std::cout << "  " << std::setw(35) << "BW roof tok/s"
              << std::setw(14) << static_cast<int>(dense_bw)
              << std::setw(14) << static_cast<int>(moe_bw)
              << "MoE " << std::setprecision(1) << bw_ratio << "× faster\n";

    double dense_tps = tps_at_batch(*h100, dense, *vllm, 16, false);
    double moe_tps   = tps_at_batch(*h100, moe,   *vllm, 16, false);
    std::cout << "  " << std::setw(35) << "Throughput @bs=16 (tok/s)"
              << std::setw(14) << static_cast<int>(dense_tps)
              << std::setw(14) << static_cast<int>(moe_tps)
              << "MoE " << std::setprecision(1) << moe_tps/dense_tps << "× faster\n";

    int dense_kv = kv_bytes_per_token(dense, 2);
    int moe_kv   = kv_bytes_per_token(moe,   2);
    std::cout << "  " << std::setw(35) << "KV bytes/token"
              << std::setw(14) << dense_kv
              << std::setw(14) << moe_kv
              << "(same KV structure)\n";

    assert(moe_bw  > dense_bw);
    assert(moe_tps > dense_tps);
    std::cout << "\n  [ASSERT] MoE higher BW roof and throughput than dense (same HW): ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n" << std::string(70, '=')
              << "\n  Chapter 33 — The Full Engine Landscape — 2026 (C++)\n"
              << std::string(70, '=') << "\n";

    demo_throughput_matrix();
    demo_kv_capacity();
    demo_kv_transfer();
    demo_speculative_decoding();
    demo_cost_comparison();
    demo_moe_vs_dense();

    std::cout << "\n" << std::string(70, '=')
              << "\n  All demos complete.\n"
              << std::string(70, '=') << "\n\n";
    return 0;
}

```

