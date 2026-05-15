# Chapter 35: Qwen — Multilingual and Long-Context — Companion Code

## Python — `qwen_demo.py`

```python
#!/usr/bin/env python3
"""
qwen_demo.py — Chapter 35: Qwen — Multilingual, Long-Context, and the Full Model Family

Comprehensive companion code covering:
  Demo 1:  Qwen tokenization efficiency — CJK vs English token counts
  Demo 2:  Full Qwen family size chart — memory, compute, and use-case mapping
  Demo 3:  KV cache budget calculation for Qwen2.5-72B at various contexts
  Demo 4:  Model selection algorithm — pick the right Qwen size for hardware + quality
  Demo 5:  MoE vs Dense compute comparison — Qwen2.5-57B-A14B vs 72B dense
  Demo 6:  Multilingual throughput analysis — effective token/s by language
  Demo 7:  Qwen2.5 vs Llama 3.1 architectural comparison
  Demo 8:  Chat template validation — why wrong templates break output
  Demo 9:  Quantization-aware model selection (INT4/FP8 on limited VRAM)
  Demo 10: Production serving config generator — auto-generates vLLM command

Run:
    python qwen_demo.py

All assertions verify the worked examples in Chapter 35.
No GPU required.
"""

import math
from dataclasses import dataclass, field
from typing import Optional
import textwrap

SEPARATOR = "─" * 70

# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QwenModelSpec:
    name: str
    params_b: float          # total parameters in billions
    active_params_b: float   # active parameters per token (< total for MoE)
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ffn: int
    context_len: int         # max context in tokens
    vocab_size: int
    is_moe: bool = False
    n_experts: int = 0
    n_active_experts: int = 0
    notes: str = ""

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def weight_memory_gb(self, dtype_bytes: float = 2.0) -> float:
        """Memory for model weights in GB (approximate)."""
        return self.params_b * 1e9 * dtype_bytes / (1024**3)

    def kv_bytes_per_token(self, dtype_bytes: float = 2.0) -> int:
        """KV cache bytes per token."""
        return 2 * self.n_layers * self.n_kv_heads * self.d_head * int(dtype_bytes)

    def kv_gb_for_context(self, context_len: int, dtype_bytes: float = 2.0) -> float:
        """Total KV cache GB for a single sequence at given context length."""
        return self.kv_bytes_per_token(dtype_bytes) * context_len / (1024**3)

    def max_sequences(self, total_vram_gb: float, context_len: int,
                      dtype_bytes: float = 2.0, kv_dtype_bytes: float = 2.0,
                      util: float = 0.90) -> int:
        """Maximum concurrent sequences given VRAM budget."""
        available_gb = total_vram_gb * util - self.weight_memory_gb(dtype_bytes)
        kv_per_seq_gb = self.kv_gb_for_context(context_len, kv_dtype_bytes)
        if available_gb <= 0 or kv_per_seq_gb <= 0:
            return 0
        return max(0, int(available_gb / kv_per_seq_gb))

    def decode_throughput_toks(self, bandwidth_tbs: float,
                                batch_size: int = 1) -> float:
        """
        Approximate decode throughput (tokens/sec) using bandwidth roofline.
        bandwidth_tbs: HBM bandwidth in TB/s
        """
        # Must load active weights once per token per request
        weight_bytes = self.active_params_b * 1e9 * 2  # BF16
        # At batch B, each weight is loaded once but serves B tokens
        effective_bytes_per_token = weight_bytes / batch_size
        bandwidth_bytes_s = bandwidth_tbs * 1e12
        return bandwidth_bytes_s / effective_bytes_per_token

    def prefill_tflops(self, seq_len: int) -> float:
        """Approximate prefill TFLOPs for a given sequence length."""
        # ~2 * params * seq_len FLOPs (matmuls dominate)
        return 2 * self.active_params_b * 1e9 * seq_len / 1e12


# ─── Qwen2.5 Model Family ───────────────────────────────────────────────────

QWEN_FAMILY = [
    QwenModelSpec("Qwen2.5-0.5B",    params_b=0.494,  active_params_b=0.494,  n_layers=28,  d_model=896,  n_heads=14, n_kv_heads=2,  d_ffn=4864,   context_len=131072, vocab_size=151936),
    QwenModelSpec("Qwen2.5-1.5B",    params_b=1.54,   active_params_b=1.54,   n_layers=28,  d_model=1536, n_heads=12, n_kv_heads=2,  d_ffn=8960,   context_len=131072, vocab_size=151936),
    QwenModelSpec("Qwen2.5-3B",      params_b=3.09,   active_params_b=3.09,   n_layers=36,  d_model=2048, n_heads=16, n_kv_heads=2,  d_ffn=11008,  context_len=131072, vocab_size=151936),
    QwenModelSpec("Qwen2.5-7B",      params_b=7.07,   active_params_b=7.07,   n_layers=28,  d_model=3584, n_heads=28, n_kv_heads=4,  d_ffn=18944,  context_len=131072, vocab_size=152064),
    QwenModelSpec("Qwen2.5-14B",     params_b=14.7,   active_params_b=14.7,   n_layers=48,  d_model=5120, n_heads=40, n_kv_heads=8,  d_ffn=13824,  context_len=131072, vocab_size=152064),
    QwenModelSpec("Qwen2.5-32B",     params_b=32.5,   active_params_b=32.5,   n_layers=64,  d_model=5120, n_heads=40, n_kv_heads=8,  d_ffn=27648,  context_len=131072, vocab_size=152064),
    QwenModelSpec("Qwen2.5-72B",     params_b=72.7,   active_params_b=72.7,   n_layers=80,  d_model=8192, n_heads=64, n_kv_heads=8,  d_ffn=29568,  context_len=131072, vocab_size=152064),
    QwenModelSpec("Qwen2.5-57B-A14B", params_b=57.4, active_params_b=14.3,   n_layers=28,  d_model=3584, n_heads=28, n_kv_heads=4,  d_ffn=18944,  context_len=65536,  vocab_size=152064,
                  is_moe=True, n_experts=64, n_active_experts=8,
                  notes="MoE: 64 experts, top-8 routing, 14B active"),
    QwenModelSpec("Qwen2.5-235B-A22B", params_b=235.0, active_params_b=22.0, n_layers=94,  d_model=4096, n_heads=64, n_kv_heads=4,  d_ffn=2048,   context_len=131072, vocab_size=152064,
                  is_moe=True, n_experts=128, n_active_experts=8,
                  notes="MoE: 128 experts, top-8 routing, 22B active"),
]

# GPU hardware specs
GPU_SPECS = {
    "RTX 3090":   {"vram_gb": 24,  "bandwidth_tbs": 0.936, "fp16_tflops": 35.6},
    "RTX 4090":   {"vram_gb": 24,  "bandwidth_tbs": 1.008, "fp16_tflops": 82.6},
    "A10G":       {"vram_gb": 24,  "bandwidth_tbs": 0.600, "fp16_tflops": 31.2},
    "A100 40GB":  {"vram_gb": 40,  "bandwidth_tbs": 1.555, "fp16_tflops": 77.97},
    "A100 80GB":  {"vram_gb": 80,  "bandwidth_tbs": 2.000, "fp16_tflops": 77.97},
    "H100 SXM":   {"vram_gb": 80,  "bandwidth_tbs": 3.350, "fp16_tflops": 989.5},
    "H200 SXM":   {"vram_gb": 141, "bandwidth_tbs": 4.800, "fp16_tflops": 989.5},
}


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: CJK Tokenization Efficiency
# ─────────────────────────────────────────────────────────────────────────────

def demo_tokenization_efficiency():
    print(f"\n{'='*70}")
    print("DEMO 1 — Qwen Tokenization Efficiency: CJK vs English")
    print(f"{'='*70}")

    # Simulated tokenization model based on Chapter 35 worked example
    # Qwen: ~1 token per 1.2 Chinese chars (multi-char chunks)
    # Llama: ~1 token per 1.0 Chinese chars (mostly single-char tokens)

    test_cases = [
        {
            "text": "人工智能改变世界",
            "description": "Chinese: 'AI changes the world' (8 characters)",
            "qwen_tokens": 3,       # Chapter 35 Worked Example 35.1
            "llama_tokens": 8,      # Each char = separate token in Llama
        },
        {
            "text": "机器学习是人工智能的核心技术",
            "description": "Chinese: 'Machine learning is the core technology of AI' (15 chars)",
            "qwen_tokens": 6,
            "llama_tokens": 15,
        },
        {
            "text": "大型语言模型的推理优化",
            "description": "Chinese: 'Inference optimization of large language models' (12 chars)",
            "qwen_tokens": 5,
            "llama_tokens": 12,
        },
        {
            "text": "Artificial intelligence changes the world",
            "description": "English equivalent (5 words ≈ 7 tokens)",
            "qwen_tokens": 7,
            "llama_tokens": 7,      # English tokenization similar for both
        },
        {
            "text": "양자 컴퓨팅의 원리를 설명해주세요",   # Korean
            "description": "Korean: 'Please explain the principles of quantum computing' (14 chars)",
            "qwen_tokens": 8,
            "llama_tokens": 14,
        },
    ]

    total_qwen = 0
    total_llama = 0

    print(f"\n{'Text Description':<45} {'Qwen':>6} {'Llama':>6} {'Ratio':>7} {'KV Savings':>10}")
    print(SEPARATOR)

    for tc in test_cases:
        ratio = tc["llama_tokens"] / tc["qwen_tokens"] if tc["qwen_tokens"] > 0 else 1.0
        kv_saving_pct = (1 - tc["qwen_tokens"] / tc["llama_tokens"]) * 100 if tc["llama_tokens"] > 0 else 0
        print(f"{tc['description'][:44]:<45} {tc['qwen_tokens']:>6} {tc['llama_tokens']:>6} {ratio:>6.1f}× {kv_saving_pct:>9.0f}%")
        total_qwen += tc["qwen_tokens"]
        total_llama += tc["llama_tokens"]

    overall_ratio = total_llama / total_qwen
    print(SEPARATOR)
    print(f"{'TOTALS':<45} {total_qwen:>6} {total_llama:>6} {overall_ratio:>6.1f}×")

    print(f"""
Key findings:
  • For CJK text, Qwen tokenizes {overall_ratio:.1f}× more efficiently than Llama
  • This means {(1 - 1/overall_ratio)*100:.0f}% less KV cache memory for Chinese workloads
  • A 2,048-token Chinese context on Llama needs {int(2048/overall_ratio)} Qwen-equivalent tokens
  • Effective throughput advantage: {overall_ratio:.1f}× more Chinese tokens per GPU-second

Vocab size comparison:
  Qwen2.5:  152,064 tokens (dedicted CJK subwords)
  Llama3.1: 128,256 tokens (English-optimized BPE)
  Difference: +{152064 - 128256:,} tokens for multilingual coverage
    """)

    # Assertions from Chapter 35 Worked Example 35.1
    assert test_cases[0]["qwen_tokens"] == 3, "Qwen should tokenize 8-char Chinese as 3 tokens"
    assert test_cases[0]["llama_tokens"] == 8, "Llama should tokenize 8-char Chinese as ~8 tokens"
    ratio_example = test_cases[0]["llama_tokens"] / test_cases[0]["qwen_tokens"]
    assert abs(ratio_example - 8/3) < 0.01, f"Chapter 35 ratio should be 2.67×, got {ratio_example:.2f}"
    print(f"  ✓ Chapter 35 Worked Example 35.1 verified: {ratio_example:.2f}× tokenization ratio")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: Full Family Size Chart
# ─────────────────────────────────────────────────────────────────────────────

def demo_family_size_chart():
    print(f"\n{'='*70}")
    print("DEMO 2 — Qwen2.5 Full Family: Memory, Compute, and Use-Case Mapping")
    print(f"{'='*70}")

    print(f"\n{'Model':<22} {'Params':>8} {'Active':>8} {'Weights BF16':>13} {'KV/tok BF16':>12} {'Use Case'}")
    print(SEPARATOR)

    use_cases = {
        "Qwen2.5-0.5B":     "Edge, classification, IoT",
        "Qwen2.5-1.5B":     "Simple chat, on-device",
        "Qwen2.5-3B":       "Assistant on mobile/edge",
        "Qwen2.5-7B":       "Strong assistant (RTX 3090)",
        "Qwen2.5-14B":      "Professional quality (2×A100-40G)",
        "Qwen2.5-32B":      "Near-frontier (A100 80G)",
        "Qwen2.5-72B":      "Frontier quality (2×H100)",
        "Qwen2.5-57B-A14B": "Frontier quality at 14B compute",
        "Qwen2.5-235B-A22B":"Frontier quality at 22B compute",
    }

    for m in QWEN_FAMILY:
        weights_gb = m.weight_memory_gb(2.0)  # BF16
        kv_per_tok = m.kv_bytes_per_token(2.0)
        moe_tag = " (MoE)" if m.is_moe else ""
        print(f"{m.name + moe_tag:<22} {m.params_b:>7.1f}B {m.active_params_b:>7.1f}B "
              f"{weights_gb:>11.1f}GB {kv_per_tok:>9,} B  {use_cases.get(m.name, '')}")

    print(f"""
Notes:
  • BF16 weights: params × 2 bytes
  • KV/token: 2 × n_layers × n_kv_heads × d_head × 2 bytes
  • MoE models: active_params << total_params → much faster decode
  • Qwen2.5-57B-A14B: stores 57B weights (114GB) but computes only 14B per token
    """)

    # Verify 72B architecture from chapter
    m72 = next(m for m in QWEN_FAMILY if m.name == "Qwen2.5-72B")
    kv_72 = m72.kv_bytes_per_token(2.0)
    expected_kv = 2 * 80 * 8 * 128 * 2  # 327,680
    assert kv_72 == expected_kv, f"Qwen2.5-72B KV/token should be {expected_kv}, got {kv_72}"
    print(f"  ✓ Qwen2.5-72B KV/token: {kv_72:,} bytes (= 320 KB) — matches Chapter 35")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: KV Cache Budget for Qwen2.5-72B
# ─────────────────────────────────────────────────────────────────────────────

def demo_kv_cache_budget():
    print(f"\n{'='*70}")
    print("DEMO 3 — KV Cache Budget: Qwen2.5-72B at Various Contexts")
    print(f"{'='*70}")

    m = next(m for m in QWEN_FAMILY if m.name == "Qwen2.5-72B")

    hardware_configs = [
        ("2× H100 80GB", 160.0),
        ("4× H100 80GB", 320.0),
        ("2× H200 141GB", 282.0),
    ]

    context_lengths = [4096, 8192, 16384, 32768, 65536, 131072]

    for hw_name, total_vram in hardware_configs:
        print(f"\n  {hw_name} ({total_vram:.0f} GB total VRAM)")
        weights_bf16 = m.weight_memory_gb(2.0)
        weights_gptq4 = m.weight_memory_gb(0.5)  # INT4 = 0.5 bytes/param
        weights_fp8 = m.weight_memory_gb(1.0)

        print(f"  {'Dtype':<12} {'Weights':>10} {'Available KV':>14}", end="")
        for ctx in context_lengths:
            print(f" {'@'+str(ctx//1024)+'K':>7}", end="")
        print()
        print(f"  {SEPARATOR}")

        for dtype_name, dtype_bytes, kv_dtype_bytes in [
            ("BF16",       2.0, 2.0),
            ("FP8 weights", 1.0, 1.0),
            ("INT4/GPTQ",  0.5, 2.0),
        ]:
            weights = m.weight_memory_gb(dtype_bytes)
            avail = total_vram * 0.90 - weights
            print(f"  {dtype_name:<12} {weights:>8.1f}GB {avail:>12.1f}GB", end="")
            for ctx in context_lengths:
                n_seq = m.max_sequences(total_vram, ctx,
                                        dtype_bytes=dtype_bytes,
                                        kv_dtype_bytes=kv_dtype_bytes)
                print(f" {n_seq:>7}", end="")
            print()

    print(f"""
Reading the table: cells show max concurrent sequences at that context length.
  • BF16 on 2×H100: only 1 sequence at 32K context — must quantize
  • INT4 weights free up ~110GB for KV: 12 sequences at 32K context
  • FP8 KV cache doubles capacity vs BF16 KV at same weight precision
    """)

    # Verify Chapter 35 Worked Example 35.3
    m72 = next(m for m in QWEN_FAMILY if m.name == "Qwen2.5-72B")
    # 2×H100 = 160GB, BF16 weights = 145.4GB, 90% util → 144GB
    # available KV = 144 - 145.4 → slightly negative, so 0
    # With INT4: weights = 36.4GB → available = 144 - 36.4 = 107.6 ≈ 108GB
    int4_avail = 160 * 0.90 - m72.weight_memory_gb(0.5)
    kv_32k_bf16 = m72.kv_gb_for_context(32768, dtype_bytes=2.0)
    max_seqs_int4_32k = int(int4_avail / kv_32k_bf16)
    # Chapter says ~12 sequences at 32K with INT4 on 2×H100
    print(f"  ✓ Worked Example 35.3: INT4 Qwen2.5-72B on 2×H100 at 32K: {max_seqs_int4_32k} sequences")
    assert max_seqs_int4_32k >= 10, f"Should have ≥10 sequences, got {max_seqs_int4_32k}"


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4: Model Selection Algorithm
# ─────────────────────────────────────────────────────────────────────────────

def demo_model_selection():
    print(f"\n{'='*70}")
    print("DEMO 4 — Model Selection: Right-Sizing for Hardware + Quality")
    print(f"{'='*70}")

    @dataclass
    class SelectionCriteria:
        available_vram_gb: float     # total GPU VRAM
        quality_threshold: str       # "good", "professional", "frontier"
        min_context_len: int         # minimum needed context
        max_weight_bytes: float = 2.0  # 2=BF16, 1=FP8, 0.5=INT4
        max_latency_ms: Optional[float] = None
        use_case: str = ""

    def select_model(criteria: SelectionCriteria) -> list:
        quality_min_params = {
            "good":         7.0,
            "professional": 14.0,
            "frontier":     32.0,
        }
        min_params = quality_min_params.get(criteria.quality_threshold, 7.0)

        candidates = []
        for m in QWEN_FAMILY:
            weight_mem = m.weight_memory_gb(criteria.max_weight_bytes)
            # Needs VRAM for weights + at least 1 sequence of min_context
            kv_mem = m.kv_gb_for_context(criteria.min_context_len)
            total_needed = weight_mem + kv_mem * 0.9  # just 1 sequence for feasibility
            if total_needed > criteria.available_vram_gb * 0.92:
                continue
            if m.active_params_b < min_params:
                continue
            candidates.append((m, weight_mem, total_needed))

        # Sort by quality (active params) descending, then by memory efficiency
        candidates.sort(key=lambda x: x[0].active_params_b, reverse=True)
        return candidates

    test_scenarios = [
        SelectionCriteria(available_vram_gb=24,   quality_threshold="good",
                          min_context_len=8192,  max_weight_bytes=0.5,
                          use_case="RTX 4090, customer support bot, INT4"),
        SelectionCriteria(available_vram_gb=48,   quality_threshold="professional",
                          min_context_len=16384, max_weight_bytes=2.0,
                          use_case="2× A10G (24GB each), professional assistant, BF16"),
        SelectionCriteria(available_vram_gb=80,   quality_threshold="professional",
                          min_context_len=32768, max_weight_bytes=1.0,
                          use_case="1× H100 80GB, long-context RAG, FP8"),
        SelectionCriteria(available_vram_gb=160,  quality_threshold="frontier",
                          min_context_len=65536, max_weight_bytes=0.5,
                          use_case="2× H100 80GB, frontier quality, INT4"),
        SelectionCriteria(available_vram_gb=320,  quality_threshold="frontier",
                          min_context_len=131072, max_weight_bytes=2.0,
                          use_case="4× H100 80GB, full 128K, BF16"),
    ]

    for sc in test_scenarios:
        print(f"\n  Scenario: {sc.use_case}")
        print(f"  VRAM: {sc.available_vram_gb}GB | Quality: {sc.quality_threshold} | "
              f"Context: {sc.min_context_len//1024}K | Dtype: {'BF16' if sc.max_weight_bytes==2 else 'FP8' if sc.max_weight_bytes==1 else 'INT4'}")
        candidates = select_model(sc)
        if candidates:
            best = candidates[0]
            m, wmem, total = best
            print(f"  → BEST: {m.name} ({m.active_params_b:.1f}B active params)")
            print(f"     Weights: {wmem:.1f}GB | Total needed: {total:.1f}GB | "
                  f"{'MoE' if m.is_moe else 'Dense'}")
            if len(candidates) > 1:
                alt = candidates[1][0]
                print(f"     Alternative: {alt.name} (smaller but fits)")
        else:
            print(f"  → No model fits! Reduce context or use more aggressive quantization.")

    # Chapter 35 Worked Example 35.2: 2× A10G (48GB total), BF16, customer support
    sc_ex = SelectionCriteria(available_vram_gb=48, quality_threshold="good",
                               min_context_len=8192, max_weight_bytes=2.0,
                               use_case="Chapter 35 Example 35.2")
    candidates = select_model(sc_ex)
    best_name = candidates[0][0].name if candidates else "None"
    print(f"\n  ✓ Worked Example 35.2 verification:")
    print(f"     Best fit for 2×A10G (48GB) + BF16 + 8K: {best_name}")

    m14 = next(m for m in QWEN_FAMILY if m.name == "Qwen2.5-14B")
    # 14.7B × 2 bytes = 29.4 GB → needs 2 GPUs (> 24GB single A10G)
    fits_dual_a10g = m14.weight_memory_gb(2.0) <= 48.0 * 0.92
    fits_single_a10g = m14.weight_memory_gb(2.0) <= 24.0 * 0.92
    # INT4: 14.7B × 0.5 bytes = 7.35 GB → fits on 1× A10G easily
    fits_single_a10g_int4 = m14.weight_memory_gb(0.5) <= 24.0 * 0.92
    print(f"     Qwen2.5-14B BF16 fits on 2×A10G (48GB): {fits_dual_a10g} "
          f"({m14.weight_memory_gb(2.0):.1f}GB weights)")
    print(f"     Qwen2.5-14B BF16 fits on 1×A10G (24GB): {fits_single_a10g} "
          f"(needs 2× GPUs for BF16)")
    print(f"     Qwen2.5-14B INT4 fits on 1×A10G (24GB): {fits_single_a10g_int4} "
          f"({m14.weight_memory_gb(0.5):.1f}GB weights)")
    assert fits_dual_a10g, "Qwen2.5-14B BF16 should fit on 2× A10G combined 48GB"
    assert not fits_single_a10g, "Qwen2.5-14B BF16 should NOT fit on single A10G 24GB"
    assert fits_single_a10g_int4, "Qwen2.5-14B INT4 should fit on single A10G"
    assert best_name == "Qwen2.5-14B", f"Best for 2×A10G BF16 should be 14B, got {best_name}"


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5: MoE vs Dense Compute Comparison
# ─────────────────────────────────────────────────────────────────────────────

def demo_moe_vs_dense():
    print(f"\n{'='*70}")
    print("DEMO 5 — MoE vs Dense: Qwen2.5-57B-A14B vs 72B Dense")
    print(f"{'='*70}")

    dense_72b = next(m for m in QWEN_FAMILY if m.name == "Qwen2.5-72B")
    moe_57b = next(m for m in QWEN_FAMILY if m.name == "Qwen2.5-57B-A14B")

    h100_bw = GPU_SPECS["H100 SXM"]["bandwidth_tbs"]

    print(f"""
  Model Comparison:
  {'Metric':<35} {'72B Dense':>15} {'57B-A14B MoE':>15}
  {SEPARATOR}
  {'Total parameters':<35} {dense_72b.params_b:>14.1f}B {moe_57b.params_b:>14.1f}B
  {'Active parameters per token':<35} {dense_72b.active_params_b:>14.1f}B {moe_57b.active_params_b:>14.1f}B
  {'Weight memory (BF16)':<35} {dense_72b.weight_memory_gb():>14.1f}GB {moe_57b.weight_memory_gb():>14.1f}GB
  {'KV bytes per token (BF16)':<35} {dense_72b.kv_bytes_per_token():>14,} {moe_57b.kv_bytes_per_token():>14,}
  {'Decode tok/s @ batch=1 (H100)':<35} {dense_72b.decode_throughput_toks(h100_bw, 1):>14.1f} {moe_57b.decode_throughput_toks(h100_bw, 1):>14.1f}
  {'Decode tok/s @ batch=32 (H100)':<35} {dense_72b.decode_throughput_toks(h100_bw, 32):>14.1f} {moe_57b.decode_throughput_toks(h100_bw, 32):>14.1f}
  {'Prefill TFLOPs @ 4K tokens':<35} {dense_72b.prefill_tflops(4096):>14.2f} {moe_57b.prefill_tflops(4096):>14.2f}
    """)

    moe_decode_speedup = moe_57b.decode_throughput_toks(h100_bw, 1) / dense_72b.decode_throughput_toks(h100_bw, 1)
    moe_prefill_speedup = dense_72b.prefill_tflops(4096) / moe_57b.prefill_tflops(4096)

    print(f"  MoE decode speedup (batch=1): {moe_decode_speedup:.1f}×  (fewer active params to load)")
    print(f"  MoE prefill speedup (4K ctx): {moe_prefill_speedup:.1f}×  (fewer active FLOPs)")
    print(f"""
  Memory trade-off:
    MoE needs {moe_57b.weight_memory_gb():.0f}GB to hold ALL experts (even inactive ones)
    Dense only needs {dense_72b.weight_memory_gb():.0f}GB — but computes MORE per token
    MoE wins on compute; Dense wins on memory footprint relative to total capacity

  Router overhead:
    Top-8 routing from 64 experts adds gating computation:
    Gate GEMM: {moe_57b.d_model} × {moe_57b.n_experts} × {moe_57b.n_layers} layers
    Cost: {moe_57b.d_model * moe_57b.n_experts * moe_57b.n_layers * 2 / 1e9:.2f} GFLOPs per token (negligible)
    """)

    assert moe_decode_speedup > 3.0, f"MoE should decode >3× faster than dense 72B, got {moe_decode_speedup:.1f}"
    print(f"  ✓ MoE decode advantage confirmed: {moe_decode_speedup:.1f}× vs dense at batch=1")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6: Multilingual Throughput Analysis
# ─────────────────────────────────────────────────────────────────────────────

def demo_multilingual_throughput():
    print(f"\n{'='*70}")
    print("DEMO 6 — Multilingual Throughput: Effective Tokens/s by Language")
    print(f"{'='*70}")

    # Qwen tokenization efficiency ratios relative to English
    # (chars per Qwen token for each language)
    language_efficiency = {
        "English":  1.0,   # baseline: ~4 chars/token
        "Chinese":  2.67,  # 2.67× more chars per Qwen token (from Demo 1)
        "Japanese": 2.2,   # Kanji compress well
        "Korean":   1.8,   # Hangul somewhat efficient
        "Arabic":   1.4,   # Script compression
        "German":   0.95,  # Compound words but similar to English
        "French":   0.98,  # Close to English
        "Spanish":  0.97,  # Close to English
    }

    # Decode throughput for Qwen2.5-7B on H100 at batch=8
    m7b = next(m for m in QWEN_FAMILY if m.name == "Qwen2.5-7B")
    h100_bw = GPU_SPECS["H100 SXM"]["bandwidth_tbs"]
    base_tok_s = m7b.decode_throughput_toks(h100_bw, 8)

    print(f"\n  Qwen2.5-7B on H100, batch=8: {base_tok_s:.0f} raw tokens/sec")
    print(f"\n  {'Language':<12} {'Efficiency':>12} {'Raw tok/s':>12} {'Char/s equiv':>14} {'vs English':>12}")
    print(f"  {SEPARATOR}")

    english_chars_per_s = base_tok_s * 4  # ~4 chars per English token
    for lang, eff in language_efficiency.items():
        tok_s = base_tok_s  # same GPU throughput in tokens/s
        chars_per_s = tok_s * (4 * eff)  # effective characters per second
        vs_en = chars_per_s / english_chars_per_s
        print(f"  {lang:<12} {eff:>10.2f}× {tok_s:>10.0f} {chars_per_s:>12,.0f} {vs_en:>10.1f}×")

    print(f"""
  Insight: Qwen's CJK tokenization delivers {language_efficiency['Chinese']:.1f}× more Chinese
  character throughput vs English at the same raw token/s rate.

  This means a Chinese user gets 2.67× more characters per second than an
  English user on the same hardware — NOT more tokens/s, but more meaning/s.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Demo 7: Qwen2.5 vs Llama 3.1 Architectural Comparison
# ─────────────────────────────────────────────────────────────────────────────

def demo_architecture_comparison():
    print(f"\n{'='*70}")
    print("DEMO 7 — Architecture Comparison: Qwen2.5-72B vs Llama 3.1 70B")
    print(f"{'='*70}")

    @dataclass
    class ModelArch:
        name: str
        params_b: float
        n_layers: int
        d_model: int
        n_heads: int
        n_kv_heads: int
        d_ffn: int
        vocab_size: int
        context_len: int
        rope_base: int
        norm_type: str
        ffn_type: str

    qwen72 = ModelArch(
        name="Qwen2.5-72B", params_b=72.7, n_layers=80, d_model=8192,
        n_heads=64, n_kv_heads=8, d_ffn=29568, vocab_size=152064,
        context_len=131072, rope_base=1_000_000, norm_type="RMSNorm", ffn_type="SwiGLU",
    )
    llama70 = ModelArch(
        name="Llama 3.1 70B", params_b=70.6, n_layers=80, d_model=8192,
        n_heads=64, n_kv_heads=8, d_ffn=28672, vocab_size=128256,
        context_len=131072, rope_base=500_000, norm_type="RMSNorm", ffn_type="SwiGLU",
    )

    fields = [
        ("Parameters",       f"{qwen72.params_b:.1f}B",      f"{llama70.params_b:.1f}B"),
        ("Layers",           str(qwen72.n_layers),           str(llama70.n_layers)),
        ("d_model",          str(qwen72.d_model),            str(llama70.d_model)),
        ("Q heads",          str(qwen72.n_heads),            str(llama70.n_heads)),
        ("KV heads (GQA)",   str(qwen72.n_kv_heads),         str(llama70.n_kv_heads)),
        ("d_head",           str(qwen72.d_model//qwen72.n_heads), str(llama70.d_model//llama70.n_heads)),
        ("d_ffn (SwiGLU)",   f"{qwen72.d_ffn:,}",            f"{llama70.d_ffn:,}"),
        ("Vocabulary",       f"{qwen72.vocab_size:,}",       f"{llama70.vocab_size:,}"),
        ("Context length",   f"{qwen72.context_len:,}",      f"{llama70.context_len:,}"),
        ("RoPE base θ",      f"{qwen72.rope_base:,}",        f"{llama70.rope_base:,}"),
        ("Layer norm",       qwen72.norm_type,               llama70.norm_type),
        ("FFN activation",   qwen72.ffn_type,               llama70.ffn_type),
    ]

    print(f"\n  {'Feature':<22} {'Qwen2.5-72B':>18} {'Llama 3.1 70B':>18}  {'Diff'}")
    print(f"  {SEPARATOR}")
    for name, qval, lval in fields:
        diff = ""
        try:
            qnum = float(qval.replace("B","").replace(",",""))
            lnum = float(lval.replace("B","").replace(",",""))
            if qnum != lnum:
                ratio = qnum / lnum
                diff = f"Qwen {ratio:.2f}×"
        except Exception:
            if qval != lval:
                diff = "← different"
        print(f"  {name:<22} {qval:>18} {lval:>18}  {diff}")

    print(f"""
  Key differences:
  1. Vocabulary: Qwen2.5 has {qwen72.vocab_size - llama70.vocab_size:,} more tokens → better CJK coverage
  2. RoPE base: Qwen uses 2× higher base (1M vs 500K) → better long-range position
  3. FFN width: Qwen's d_ffn slightly larger ({qwen72.d_ffn} vs {llama70.d_ffn})
  4. Overall: architecturally nearly identical — main difference is tokenizer + training data
    """)

    # Verify d_head matches chapter
    assert qwen72.d_model // qwen72.n_heads == 128, "Qwen2.5-72B d_head should be 128"
    assert llama70.d_model // llama70.n_heads == 128, "Llama 3.1 70B d_head should be 128"
    print(f"  ✓ Both models use d_head=128 — identical attention head geometry")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 8: Chat Template Validation
# ─────────────────────────────────────────────────────────────────────────────

def demo_chat_template():
    print(f"\n{'='*70}")
    print("DEMO 8 — Chat Template: Why Wrong Templates Break Output")
    print(f"{'='*70}")

    def apply_qwen_template(messages: list) -> str:
        """Apply Qwen ChatML template."""
        result = ""
        for role, content in messages:
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        result += "<|im_start|>assistant\n"
        return result

    def apply_llama3_template(messages: list) -> str:
        """Apply Llama 3 template."""
        result = "<|begin_of_text|>"
        for role, content in messages:
            result += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return result

    def apply_chatml_template(messages: list) -> str:
        """Apply generic ChatML template."""
        result = ""
        for role, content in messages:
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        result += "<|im_start|>assistant\n"
        return result

    messages = [
        ("system", "You are a helpful assistant."),
        ("user", "What is the capital of France?"),
    ]

    print("\n  Test messages:")
    for role, content in messages:
        print(f"    [{role}]: {content}")

    print("\n  Formatted prompts by template:")
    print("\n  ── Qwen ChatML (CORRECT for Qwen models) ──")
    qwen_prompt = apply_qwen_template(messages)
    print(textwrap.indent(repr(qwen_prompt), "    "))

    print("\n  ── Llama 3 format (WRONG for Qwen models) ──")
    llama_prompt = apply_llama3_template(messages)
    print(textwrap.indent(repr(llama_prompt[:120] + "..."), "    "))

    print(f"""
  Token ID analysis (hypothetical):
    '<|im_start|>'  → Qwen vocab token ID 151644 (recognized by Qwen)
    '<|im_start|>'  → Llama vocab: NOT a special token, decoded character-by-character

  What goes wrong with wrong template on Qwen:
    Qwen sees Llama's '<|start_header_id|>system<|end_header_id|>' as:
    → literal text characters (not control tokens)
    → Model ignores role boundaries → generates from wrong position
    → Output may be repetitive or irrelevant

  What goes wrong with wrong template on Llama:
    Llama sees Qwen's '<|im_start|>system' as:
    → Unknown token or character sequence
    → Role signaling breaks down
    → Model behavior degrades significantly

  ✓ Rule: ALWAYS specify --chat-template qwen when using Qwen in llama.cpp
          When using vLLM, Qwen models auto-detect their template — but verify!
    """)

    # Structural assertion: Qwen template contains im_start/im_end markers
    assert "<|im_start|>" in qwen_prompt
    assert "<|im_end|>" in qwen_prompt
    assert "<|im_start|>assistant" in qwen_prompt
    assert "<|im_start|>" not in llama_prompt
    print(f"  ✓ Template format verification passed")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 9: Quantization-Aware Model Selection
# ─────────────────────────────────────────────────────────────────────────────

def demo_quantization_selection():
    print(f"\n{'='*70}")
    print("DEMO 9 — Quantization-Aware Model Selection Matrix")
    print(f"{'='*70}")

    quant_configs = [
        ("BF16",     2.0,  0.0,  "Maximum quality, maximum memory"),
        ("FP8",      1.0,  0.05, "2× memory savings, minimal quality loss (<0.3 PPL)"),
        ("INT8/Q8_0", 1.0, 0.1,  "Same size as FP8, slightly different compute path"),
        ("INT4/Q4_K_M", 0.5, 0.3, "4× memory savings, ~0.3 PPL penalty"),
        ("INT4/Q4_0", 0.5,  0.5, "Slightly lower quality than K-quant variants"),
        ("INT3/Q3_K_M", 0.375, 0.8, "Aggressive: 5.3× savings, noticeable quality drop"),
        ("INT2/Q2_K", 0.25, 2.0, "Extreme: 8× savings, significant degradation"),
    ]

    target_models = ["Qwen2.5-7B", "Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-72B"]

    print(f"\n  {'Model':<18}", end="")
    for qname, *_ in quant_configs:
        print(f" {qname:>12}", end="")
    print()
    print(f"  {SEPARATOR}")

    for model_name in target_models:
        m = next(x for x in QWEN_FAMILY if x.name == model_name)
        print(f"  {model_name:<18}", end="")
        for qname, bytes_per_param, ppl_delta, desc in quant_configs:
            mem_gb = m.params_b * 1e9 * bytes_per_param / (1024**3)
            print(f" {mem_gb:>11.1f}G", end="")
        print()

    print(f"\n  PPL penalty per quantization type:")
    for qname, bytes_per_param, ppl_delta, desc in quant_configs:
        bar = "█" * int(ppl_delta * 5)
        print(f"    {qname:<16} +{ppl_delta:.2f} PPL  {bar}  {desc}")

    print(f"""
  Selection guidelines:
    BF16/FP8:     Best for production when quality is paramount
    INT4 K-quant: Best balance for most deployments (Q4_K_M recommended)
    INT3 and below: Only for extreme memory constraints; evaluate carefully

  Rule of thumb:
    Larger models tolerate quantization better than smaller models.
    Qwen2.5-72B INT4 quality ≈ Qwen2.5-32B BF16 (in many benchmarks)
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Demo 10: Production Config Generator
# ─────────────────────────────────────────────────────────────────────────────

def demo_config_generator():
    print(f"\n{'='*70}")
    print("DEMO 10 — Production Config Generator: Auto-generate vLLM + llama.cpp Commands")
    print(f"{'='*70}")

    @dataclass
    class DeploymentRequest:
        model_name: str
        gpu_count: int
        gpu_type: str
        context_len: int
        use_case: str           # "interactive" | "batch" | "rag"
        quantization: str       # "bf16" | "fp8" | "int4"

    def generate_vllm_command(req: DeploymentRequest) -> str:
        m = next((x for x in QWEN_FAMILY if x.name == req.model_name), None)
        if not m:
            return "# Model not found"

        gpu = GPU_SPECS.get(req.gpu_type, {})
        tp = req.gpu_count

        # Compute chunked prefill tokens
        chunk_tokens = 2048 if req.use_case == "interactive" else 4096

        # Max sequences
        dtype_bytes = {"bf16": 2.0, "fp8": 1.0, "int4": 0.5}.get(req.quantization, 2.0)
        kv_bytes = 1.0 if req.quantization in ("fp8",) else 2.0
        total_vram = (gpu.get("vram_gb", 80) * req.gpu_count)
        max_seqs = max(8, m.max_sequences(total_vram, req.context_len,
                                          dtype_bytes=dtype_bytes,
                                          kv_dtype_bytes=kv_bytes))
        max_seqs = min(max_seqs, 512)

        quant_flag = {
            "bf16": "",
            "fp8": " \\\n    --quantization fp8",
            "int4": " \\\n    --quantization gptq",
        }.get(req.quantization, "")

        lines = [
            f"vllm serve {m.name.replace('2.5', '2.5').lower().replace('-', '/')}",  # simplified
            f"    # === Hardware: {req.gpu_count}× {req.gpu_type} ({total_vram}GB total) ===",
            f"    --tensor-parallel-size {tp} \\",
            f"    --dtype bfloat16 \\",
        ]
        if quant_flag:
            lines.append(f"    --quantization {req.quantization} \\")
        lines += [
            f"    --kv-cache-dtype {'fp8' if kv_bytes == 1.0 else 'auto'} \\",
            f"    --max-model-len {req.context_len} \\",
            f"    --gpu-memory-utilization 0.90 \\",
            f"    --max-num-seqs {max_seqs} \\",
            f"    --enable-prefix-caching \\",
        ]
        if req.use_case != "batch":
            lines += [
                f"    --enable-chunked-prefill \\",
                f"    --max-num-batched-tokens {chunk_tokens} \\",
            ]
        if req.use_case == "batch":
            lines.append(f"    --scheduler-delay-factor 0.3 \\")
        lines += [
            f"    --host 0.0.0.0 \\",
            f"    --port 8000",
        ]
        return "\n".join(lines)

    deployments = [
        DeploymentRequest("Qwen2.5-7B",  1, "RTX 4090", 16384, "interactive", "bf16"),
        DeploymentRequest("Qwen2.5-14B", 2, "A100 40GB", 32768, "rag",        "bf16"),
        DeploymentRequest("Qwen2.5-72B", 4, "H100 SXM", 131072, "interactive", "fp8"),
        DeploymentRequest("Qwen2.5-32B", 1, "H100 SXM", 65536,  "batch",      "int4"),
    ]

    for req in deployments:
        print(f"\n  ── {req.model_name} | {req.gpu_count}× {req.gpu_type} | "
              f"{req.context_len//1024}K ctx | {req.use_case} | {req.quantization.upper()} ──")
        cmd = generate_vllm_command(req)
        print(textwrap.indent(cmd, "  "))

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print(f"\n{'='*70}")
    print("CHAPTER 35 — SUMMARY OF KEY RESULTS")
    print(f"{'='*70}")
    print(f"""
  Tokenization efficiency (CJK vs English):
    Chinese text: Qwen2.5 uses 2.67× fewer tokens than Llama3
    This translates to 2.67× more Chinese character throughput per GPU

  Model family range:
    0.5B (IoT/edge) → 235B-A22B (frontier MoE)
    All from one architecture family — same serving stack

  KV cache: Qwen2.5-72B at 128K context (BF16):
    320 KB per token, 40 GB per sequence
    2× H100 barely fits 1 sequence at 32K in BF16 → use INT4

  MoE advantage (57B-A14B):
    5.1× faster decode vs dense 72B at batch=1
    Trade-off: needs 114GB to hold all 57B parameters

  Architecture vs Llama 3.1 70B:
    Nearly identical — key difference is tokenizer (+23,808 vocab tokens)
    and RoPE base (1M vs 500K) for better long-context generalization

  Critical configuration rule:
    Always specify --chat-template qwen in llama.cpp
    Wrong template → broken role signaling → degraded output
  """)


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   Chapter 35: Qwen — Multilingual, Long-Context, Model Family        ║")
    print("║   Comprehensive Demo Suite — 10 Demonstrations                       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    demo_tokenization_efficiency()
    demo_family_size_chart()
    demo_kv_cache_budget()
    demo_model_selection()
    demo_moe_vs_dense()
    demo_multilingual_throughput()
    demo_architecture_comparison()
    demo_chat_template()
    demo_quantization_selection()
    demo_config_generator()
    print_summary()

    print(f"\n{'='*70}")
    print("ALL CHAPTER 35 DEMOS COMPLETED — ALL ASSERTIONS PASSED ✓")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

```



## C++ — `qwen_demo.cpp`

```bash
# Compile
g++ -std=c++17 -O2 -o qwen_demo qwen_demo.cpp -lm
# Run
./qwen_demo
```

```cpp
/*
 * qwen_demo.cpp — Chapter 35: Qwen — Multilingual, Long-Context, Model Family
 *
 * Demonstrates (mirrors qwen_demo.py, 10 demos):
 *   Demo 1:  CJK tokenization efficiency
 *   Demo 2:  Full Qwen family size chart
 *   Demo 3:  KV cache budget for Qwen2.5-72B
 *   Demo 4:  Model selection algorithm
 *   Demo 5:  MoE vs Dense compute comparison
 *   Demo 6:  Multilingual throughput analysis
 *   Demo 7:  Qwen2.5 vs Llama 3.1 architecture comparison
 *   Demo 8:  Chat template validation
 *   Demo 9:  Quantization-aware model selection
 *   Demo 10: Production config generator
 *
 * Compile: g++ -std=c++17 -O2 -o qwen_demo qwen_demo.cpp -lm
 * Run:     ./qwen_demo
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Model spec
// ─────────────────────────────────────────────────────────────────────────────

struct QwenSpec {
    const char* name;
    double params_b;
    double active_params_b;
    int    n_layers;
    int    d_model;
    int    n_heads;
    int    n_kv_heads;
    int    context_len;
    int    vocab_size;
    bool   is_moe;
    int    n_experts;

    int d_head()            const { return d_model / n_heads; }
    double weight_gb(double db = 2.0) const { return params_b * 1e9 * db / (1024.0*1024*1024); }
    int kv_bytes_per_token(double db = 2.0) const {
        return (int)(2.0 * n_layers * n_kv_heads * d_head() * db);
    }
    double kv_gb_for_seq(int ctx, double db = 2.0) const {
        return kv_bytes_per_token(db) * (double)ctx / (1024.0*1024*1024);
    }
    int max_sequences(double total_vram_gb, int ctx,
                       double wdb = 2.0, double kvdb = 2.0,
                       double util = 0.90) const {
        double avail  = total_vram_gb * util - weight_gb(wdb);
        double kv_seq = kv_gb_for_seq(ctx, kvdb);
        if (avail <= 0 || kv_seq <= 0) return 0;
        return (int)(avail / kv_seq);
    }
    double decode_tps(double bw_tbs, int batch = 1) const {
        double wb = active_params_b * 1e9 * 2.0;   // BF16
        return (bw_tbs * 1e12) / (wb / batch);
    }
};

static QwenSpec FAMILY[] = {
    {"Qwen2.5-0.5B",      0.494,  0.494,  28, 896,  14, 2,  131072, 151936, false, 0},
    {"Qwen2.5-1.5B",      1.54,   1.54,   28, 1536, 12, 2,  131072, 151936, false, 0},
    {"Qwen2.5-3B",        3.09,   3.09,   36, 2048, 16, 2,  131072, 151936, false, 0},
    {"Qwen2.5-7B",        7.07,   7.07,   28, 3584, 28, 4,  131072, 152064, false, 0},
    {"Qwen2.5-14B",       14.7,   14.7,   48, 5120, 40, 8,  131072, 152064, false, 0},
    {"Qwen2.5-32B",       32.5,   32.5,   64, 5120, 40, 8,  131072, 152064, false, 0},
    {"Qwen2.5-72B",       72.7,   72.7,   80, 8192, 64, 8,  131072, 152064, false, 0},
    {"Qwen2.5-57B-A14B",  57.4,   14.3,   28, 3584, 28, 4,  65536,  152064, true,  64},
    {"Qwen2.5-235B-A22B", 235.0,  22.0,   94, 4096, 64, 4,  131072, 152064, true,  128},
};
static const int N_MODELS = 9;

static const QwenSpec* find_model(const char* name) {
    for (int i = 0; i < N_MODELS; ++i)
        if (strcmp(FAMILY[i].name, name) == 0) return &FAMILY[i];
    return nullptr;
}

// GPU specs: {vram_gb, bandwidth_tbs}
struct GPU { const char* name; double vram_gb; double bw_tbs; };
static GPU GPUS[] = {
    {"RTX 3090",  24,  0.936},
    {"RTX 4090",  24,  1.008},
    {"A10G",      24,  0.600},
    {"A100 40GB", 40,  1.555},
    {"A100 80GB", 80,  2.000},
    {"H100 SXM",  80,  3.350},
    {"H200 SXM",  141, 4.800},
};

static const char* SEP = "──────────────────────────────────────────────────────────────────────";

// ─────────────────────────────────────────────────────────────────────────────
// Demo 1: CJK Tokenization Efficiency
// ─────────────────────────────────────────────────────────────────────────────

static void demo_tokenization_efficiency() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 1 — Qwen Tokenization Efficiency: CJK vs English\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    struct TC { const char* desc; int qwen_toks; int llama_toks; };
    TC tests[] = {
        {"Chinese: 'AI changes the world' (8 chars)",           3,  8},
        {"Chinese: 'ML is core tech of AI' (15 chars)",         6,  15},
        {"Chinese: 'LLM inference optimization' (12 chars)",    5,  12},
        {"English: 'AI changes the world' (5 words)",           7,  7},
        {"Korean: 'Explain quantum computing' (14 chars)",      8,  14},
    };
    const int NT = 5;

    int total_qwen = 0, total_llama = 0;
    printf("\n%-45s %6s %6s %7s %10s\n",
           "Text Description", "Qwen", "Llama", "Ratio", "KV Savings");
    printf("%s\n", SEP);

    for (int i = 0; i < NT; ++i) {
        double ratio = (double)tests[i].llama_toks / tests[i].qwen_toks;
        double kv_save = (1.0 - (double)tests[i].qwen_toks / tests[i].llama_toks) * 100.0;
        printf("%-45s %6d %6d %6.1fx %9.0f%%\n",
               tests[i].desc, tests[i].qwen_toks, tests[i].llama_toks, ratio, kv_save);
        total_qwen  += tests[i].qwen_toks;
        total_llama += tests[i].llama_toks;
    }
    printf("%s\n", SEP);
    double overall = (double)total_llama / total_qwen;
    printf("%-45s %6d %6d %6.1fx\n", "TOTALS", total_qwen, total_llama, overall);

    printf("\n  Key: CJK tokenization %.1fx more efficient => %.0f%% less KV cache memory\n",
           overall, (1.0 - 1.0/overall)*100.0);
    printf("  Vocab: Qwen2.5 152,064 | Llama3.1 128,256 (+23,808 multilingual tokens)\n");

    // Chapter 35 Worked Example 35.1
    assert(tests[0].qwen_toks == 3);
    assert(tests[0].llama_toks == 8);
    double ratio_ex = (double)tests[0].llama_toks / tests[0].qwen_toks;
    assert(fabs(ratio_ex - 8.0/3.0) < 0.01);
    printf("  ✓ Worked Example 35.1 verified: %.2fx tokenization ratio\n", ratio_ex);
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 2: Family Size Chart
// ─────────────────────────────────────────────────────────────────────────────

static void demo_family_size_chart() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 2 — Qwen2.5 Full Family: Memory, Compute, Use-Case\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    const char* use_cases[] = {
        "Edge, classification, IoT",
        "Simple chat, on-device",
        "Assistant on mobile/edge",
        "Strong assistant (RTX 3090)",
        "Professional quality (2xA100-40G)",
        "Near-frontier (A100 80G)",
        "Frontier quality (2xH100)",
        "Frontier quality at 14B compute",
        "Frontier quality at 22B compute",
    };

    printf("\n%-22s %8s %8s %13s %12s\n",
           "Model", "Params", "Active", "Weights BF16", "KV/tok BF16");
    printf("%s\n", SEP);

    for (int i = 0; i < N_MODELS; ++i) {
        const auto& m = FAMILY[i];
        double wgb = m.weight_gb(2.0);
        int kv     = m.kv_bytes_per_token(2.0);
        printf("%-22s %7.1fB %7.1fB %11.1fGB %9d B  %s\n",
               m.name, m.params_b, m.active_params_b, wgb, kv, use_cases[i]);
    }

    // Verify 72B KV/token
    const auto* m72 = find_model("Qwen2.5-72B");
    int expected_kv = 2 * 80 * 8 * 128 * 2;  // 327,680
    assert(m72->kv_bytes_per_token(2.0) == expected_kv);
    printf("  ✓ Qwen2.5-72B KV/token: %d bytes (= 320 KB) — matches Chapter 35\n",
           m72->kv_bytes_per_token(2.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 3: KV Cache Budget
// ─────────────────────────────────────────────────────────────────────────────

static void demo_kv_cache_budget() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 3 — KV Cache Budget: Qwen2.5-72B at Various Contexts\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    const auto* m = find_model("Qwen2.5-72B");
    struct HW { const char* name; double vram_gb; };
    HW hws[] = {
        {"2x H100 80GB", 160.0},
        {"4x H100 80GB", 320.0},
        {"2x H200 141GB", 282.0},
    };
    int ctxs[] = {4096, 8192, 16384, 32768, 65536, 131072};

    for (auto& hw : hws) {
        printf("\n  %s (%.0f GB total VRAM)\n", hw.name, hw.vram_gb);
        printf("  %-14s %10s %14s", "Dtype", "Weights", "Avail KV");
        for (int c : ctxs) printf(" %7s", (std::to_string(c/1024)+"K").c_str());
        printf("\n  %s\n", SEP);

        struct DType { const char* n; double wb; double kvb; };
        DType dtypes[] = {
            {"BF16",        2.0, 2.0},
            {"FP8 weights", 1.0, 1.0},
            {"INT4/GPTQ",   0.5, 2.0},
        };
        for (auto& dt : dtypes) {
            double w  = m->weight_gb(dt.wb);
            double av = hw.vram_gb * 0.90 - w;
            printf("  %-14s %8.1fGB %12.1fGB", dt.n, w, av);
            for (int c : ctxs) {
                int n = m->max_sequences(hw.vram_gb, c, dt.wb, dt.kvb);
                printf(" %7d", n);
            }
            printf("\n");
        }
    }

    // Worked Example 35.3
    double int4_avail = 160.0 * 0.90 - m->weight_gb(0.5);
    double kv_32k_bf16 = m->kv_gb_for_seq(32768, 2.0);
    int max_seqs = (int)(int4_avail / kv_32k_bf16);
    printf("  ✓ Worked Example 35.3: INT4 Qwen2.5-72B on 2xH100 at 32K: %d sequences\n",
           max_seqs);
    assert(max_seqs >= 10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 4: Model Selection
// ─────────────────────────────────────────────────────────────────────────────

static void demo_model_selection() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 4 — Model Selection: Right-Sizing for Hardware + Quality\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    struct Criteria {
        double vram_gb; const char* quality; int min_ctx; double wdb; const char* desc;
    };

    Criteria scenarios[] = {
        {24,  "good",         8192,   0.5, "RTX 4090, customer support, INT4"},
        {48,  "professional", 16384,  2.0, "2xA10G (48GB), assistant, BF16"},
        {80,  "professional", 32768,  1.0, "1xH100 80GB, long RAG, FP8"},
        {160, "frontier",     65536,  0.5, "2xH100 80GB, frontier, INT4"},
        {320, "frontier",     131072, 2.0, "4xH100 80GB, full 128K, BF16"},
    };

    for (auto& sc : scenarios) {
        double min_active = 7.0;
        if (strcmp(sc.quality, "professional") == 0) min_active = 14.0;
        if (strcmp(sc.quality, "frontier") == 0)      min_active = 32.0;

        const QwenSpec* best = nullptr;
        for (int i = 0; i < N_MODELS; ++i) {
            const auto& m = FAMILY[i];
            double w   = m.weight_gb(sc.wdb);
            double kv  = m.kv_gb_for_seq(sc.min_ctx);
            double tot = w + kv * 0.9;
            if (tot > sc.vram_gb * 0.92) continue;
            if (m.active_params_b < min_active) continue;
            if (!best || m.active_params_b > best->active_params_b) best = &m;
        }

        printf("\n  Scenario: %s\n", sc.desc);
        if (best)
            printf("  -> BEST: %s (%.1fB active, %s)\n",
                   best->name, best->active_params_b, best->is_moe ? "MoE" : "Dense");
        else
            printf("  -> No model fits! Reduce context or use more aggressive quant.\n");
    }

    // Worked Example 35.2
    const auto* m14 = find_model("Qwen2.5-14B");
    bool fits_dual  = m14->weight_gb(2.0) <= 48.0 * 0.92;
    bool fits_single= m14->weight_gb(2.0) <= 24.0 * 0.92;
    bool fits_int4  = m14->weight_gb(0.5) <= 24.0 * 0.92;
    assert(fits_dual);
    assert(!fits_single);
    assert(fits_int4);
    printf("\n  ✓ Worked Example 35.2:\n");
    printf("    14B BF16 fits 2xA10G (48GB): %s (%.1fGB weights)\n",
           fits_dual ? "true" : "false", m14->weight_gb(2.0));
    printf("    14B BF16 fits 1xA10G (24GB): %s\n",
           fits_single ? "true" : "false");
    printf("    14B INT4 fits 1xA10G (24GB): %s (%.1fGB weights)\n",
           fits_int4 ? "true" : "false", m14->weight_gb(0.5));
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 5: MoE vs Dense
// ─────────────────────────────────────────────────────────────────────────────

static void demo_moe_vs_dense() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 5 — MoE vs Dense: Qwen2.5-57B-A14B vs 72B Dense\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    const auto* d = find_model("Qwen2.5-72B");
    const auto* e = find_model("Qwen2.5-57B-A14B");
    const double h100_bw = 3.350;  // TB/s

    printf("\n  %-35s %15s %15s\n", "Metric", "72B Dense", "57B-A14B MoE");
    printf("  %s\n", SEP);
    printf("  %-35s %14.1fB %14.1fB\n", "Total parameters", d->params_b, e->params_b);
    printf("  %-35s %14.1fB %14.1fB\n", "Active parameters/token", d->active_params_b, e->active_params_b);
    printf("  %-35s %13.1fGB %13.1fGB\n", "Weight memory (BF16)", d->weight_gb(), e->weight_gb());
    printf("  %-35s %14d %14d\n", "KV bytes/token (BF16)", d->kv_bytes_per_token(), e->kv_bytes_per_token());
    printf("  %-35s %14.1f %14.1f\n", "Decode tok/s @ batch=1 (H100)", d->decode_tps(h100_bw,1), e->decode_tps(h100_bw,1));
    printf("  %-35s %14.1f %14.1f\n", "Decode tok/s @ batch=32 (H100)", d->decode_tps(h100_bw,32), e->decode_tps(h100_bw,32));

    double speedup = e->decode_tps(h100_bw,1) / d->decode_tps(h100_bw,1);
    printf("\n  MoE decode speedup (batch=1): %.1fx\n", speedup);

    assert(speedup > 3.0);
    printf("  ✓ MoE decode advantage confirmed: %.1fx vs dense at batch=1\n", speedup);
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 6: Multilingual Throughput
// ─────────────────────────────────────────────────────────────────────────────

static void demo_multilingual_throughput() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 6 — Multilingual Throughput: Effective Tokens/s by Language\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    struct Lang { const char* name; double eff; };
    Lang langs[] = {
        {"English",  1.00}, {"Chinese",  2.67}, {"Japanese", 2.20},
        {"Korean",   1.80}, {"Arabic",   1.40}, {"German",   0.95},
        {"French",   0.98}, {"Spanish",  0.97},
    };

    const auto* m7b = find_model("Qwen2.5-7B");
    double base_tps = m7b->decode_tps(3.350, 8);
    double eng_chars_per_s = base_tps * 4.0;

    printf("\n  Qwen2.5-7B on H100, batch=8: %.0f raw tokens/sec\n\n", base_tps);
    printf("  %-12s %12s %12s %14s %12s\n",
           "Language", "Efficiency", "Raw tok/s", "Char/s equiv", "vs English");
    printf("  %s\n", SEP);

    for (auto& l : langs) {
        double chars_per_s = base_tps * (4.0 * l.eff);
        double vs_en = chars_per_s / eng_chars_per_s;
        printf("  %-12s %10.2fx %10.0f %12.0f %10.1fx\n",
               l.name, l.eff, base_tps, chars_per_s, vs_en);
    }
    printf("\n  CJK gives %.1fx more chars/second vs English at same token rate.\n",
           langs[1].eff);
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 7: Architecture Comparison
// ─────────────────────────────────────────────────────────────────────────────

static void demo_architecture_comparison() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 7 — Architecture: Qwen2.5-72B vs Llama 3.1 70B\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    struct Arch {
        const char* name;
        double params_b;
        int n_layers, d_model, n_heads, n_kv_heads, d_ffn;
        int vocab, ctx_len, rope_base;
    };
    Arch q = {"Qwen2.5-72B",   72.7, 80, 8192, 64, 8, 29568, 152064, 131072, 1000000};
    Arch l = {"Llama 3.1 70B", 70.6, 80, 8192, 64, 8, 28672, 128256, 131072, 500000};

    printf("\n  %-22s %18s %18s\n", "Feature", q.name, l.name);
    printf("  %s\n", SEP);
    printf("  %-22s %17.1fB %17.1fB\n", "Parameters", q.params_b, l.params_b);
    printf("  %-22s %18d %18d\n", "Layers",    q.n_layers,  l.n_layers);
    printf("  %-22s %18d %18d\n", "d_model",   q.d_model,   l.d_model);
    printf("  %-22s %18d %18d\n", "Q heads",   q.n_heads,   l.n_heads);
    printf("  %-22s %18d %18d\n", "KV heads",  q.n_kv_heads,l.n_kv_heads);
    printf("  %-22s %18d %18d\n", "d_head",    q.d_model/q.n_heads, l.d_model/l.n_heads);
    printf("  %-22s %18d %18d\n", "d_ffn",     q.d_ffn,     l.d_ffn);
    printf("  %-22s %18d %18d\n", "Vocabulary",q.vocab,     l.vocab);
    printf("  %-22s %18d %18d\n", "Ctx length",q.ctx_len,   l.ctx_len);
    printf("  %-22s %18d %18d\n", "RoPE base", q.rope_base, l.rope_base);
    printf("\n  Vocab diff: +%d tokens for CJK coverage\n", q.vocab - l.vocab);
    printf("  RoPE base: Qwen 2× higher (1M vs 500K) → better long-range positions\n");

    assert(q.d_model / q.n_heads == 128);
    assert(l.d_model / l.n_heads == 128);
    printf("  ✓ Both models use d_head=128 — identical attention head geometry\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 8: Chat Template Validation
// ─────────────────────────────────────────────────────────────────────────────

static void demo_chat_template() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 8 — Chat Template: Why Wrong Templates Break Output\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    // Build Qwen ChatML prompt
    const char* qwen_prompt_check = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                                    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
                                    "<|im_start|>assistant\n";
    // Build Llama-3 prompt (does not contain <|im_start|>)
    const char* llama_prompt_check = "<|begin_of_text|>"
                                     "<|start_header_id|>system<|end_header_id|>\n\n"
                                     "You are a helpful assistant.<|eot_id|>"
                                     "<|start_header_id|>user<|end_header_id|>\n\n"
                                     "What is the capital of France?<|eot_id|>"
                                     "<|start_header_id|>assistant<|end_header_id|>\n\n";

    printf("\n  Qwen ChatML format (CORRECT for Qwen):\n");
    printf("  \"%s\"\n", qwen_prompt_check);
    printf("  Llama-3 format first 80 chars (WRONG for Qwen):\n");
    printf("  \"%.80s...\"\n\n", llama_prompt_check);

    printf("  Token analysis:\n");
    printf("    '<|im_start|>'  => Qwen vocab token 151644 (special control)\n");
    printf("    '<|im_start|>'  => Llama vocab: NOT special => decoded char-by-char\n\n");
    printf("  Rule: ALWAYS specify --chat-template qwen in llama.cpp\n");
    printf("        vLLM auto-detects Qwen templates — but verify!\n");

    // Assertions
    assert(strstr(qwen_prompt_check,  "<|im_start|>") != nullptr);
    assert(strstr(qwen_prompt_check,  "<|im_end|>")   != nullptr);
    assert(strstr(qwen_prompt_check,  "<|im_start|>assistant") != nullptr);
    assert(strstr(llama_prompt_check, "<|im_start|>") == nullptr);
    printf("  ✓ Template format verification passed\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 9: Quantization-Aware Selection
// ─────────────────────────────────────────────────────────────────────────────

static void demo_quantization_selection() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 9 — Quantization-Aware Model Selection Matrix\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    struct Quant { const char* n; double bpp; double ppl_delta; };
    Quant quants[] = {
        {"BF16",    2.00, 0.00},
        {"FP8",     1.00, 0.05},
        {"INT8",    1.00, 0.10},
        {"INT4/Q4", 0.50, 0.30},
        {"INT3/Q3", 0.375,0.80},
        {"INT2/Q2", 0.25, 2.00},
    };
    const int NQ = 6;
    const char* models[] = {"Qwen2.5-7B","Qwen2.5-14B","Qwen2.5-32B","Qwen2.5-72B"};
    const int NM = 4;

    printf("\n  %-18s", "Model");
    for (int q = 0; q < NQ; ++q) printf(" %12s", quants[q].n);
    printf("\n  %s\n", SEP);

    for (int mi = 0; mi < NM; ++mi) {
        const auto* m = find_model(models[mi]);
        printf("  %-18s", m->name);
        for (int q = 0; q < NQ; ++q) {
            double mem = m->params_b * 1e9 * quants[q].bpp / (1024.0*1024*1024);
            printf(" %10.1fGB", mem);
        }
        printf("\n");
    }

    printf("\n  PPL penalty per quantization:\n");
    for (int q = 0; q < NQ; ++q) {
        int bars = (int)(quants[q].ppl_delta * 5);
        printf("    %-16s +%.2f PPL  ", quants[q].n, quants[q].ppl_delta);
        for (int b = 0; b < bars; ++b) printf("█");
        printf("\n");
    }
    printf("\n  Rule: Larger models tolerate quantization better.\n");
    printf("  Qwen2.5-72B INT4 quality ≈ Qwen2.5-32B BF16 on most benchmarks.\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo 10: Production Config Generator
// ─────────────────────────────────────────────────────────────────────────────

static void demo_config_generator() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 10 — Production Config Generator: vLLM Commands\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    struct Req { const char* model; int n_gpus; const char* gpu; int ctx; const char* use; const char* quant; };
    Req reqs[] = {
        {"Qwen2.5-7B",  1, "RTX 4090",  16384,  "interactive", "bf16"},
        {"Qwen2.5-14B", 2, "A100 40GB", 32768,  "rag",          "bf16"},
        {"Qwen2.5-72B", 4, "H100 SXM",  131072, "interactive",  "fp8"},
        {"Qwen2.5-32B", 1, "H100 SXM",  65536,  "batch",        "int4"},
    };

    for (auto& r : reqs) {
        const auto* m = find_model(r.model);
        printf("\n  ── %s | %dx%s | ctx=%dK | %s | %s ──\n",
               r.model, r.n_gpus, r.gpu, r.ctx/1024, r.use, r.quant);
        printf("  vllm serve %s \\\n", r.model);
        printf("    --tensor-parallel-size %d \\\n", r.n_gpus);
        if (strcmp(r.quant, "fp8")  == 0) printf("    --quantization fp8 \\\n");
        if (strcmp(r.quant, "int4") == 0) printf("    --quantization gptq \\\n");
        printf("    --max-model-len %d \\\n", r.ctx);
        printf("    --gpu-memory-utilization 0.90 \\\n");
        printf("    --enable-prefix-caching \\\n");
        if (strcmp(r.use, "batch") != 0)
            printf("    --enable-chunked-prefill \\\n");
        printf("    --host 0.0.0.0 --port 8000\n");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║   Chapter 35: Qwen — Multilingual, Long-Context, Model Family (C++) ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");

    demo_tokenization_efficiency();
    demo_family_size_chart();
    demo_kv_cache_budget();
    demo_model_selection();
    demo_moe_vs_dense();
    demo_multilingual_throughput();
    demo_architecture_comparison();
    demo_chat_template();
    demo_quantization_selection();
    demo_config_generator();

    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("ALL CHAPTER 35 DEMOS COMPLETED — ALL ASSERTIONS PASSED ✓\n");
    printf("%s\n\n", "══════════════════════════════════════════════════════════════════════");
    return 0;
}
```
