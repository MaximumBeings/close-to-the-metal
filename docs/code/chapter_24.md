# Chapter 24: Reasoning Model Inference — Companion Code

## Python — `reasoning_demo.py`

```python
"""
reasoning_demo.py — Chapter 24: Reasoning Model Inference

Demonstrates:
  1. KV cache growth calculator during reasoning trace
  2. Token budget cost model (standard vs. reasoning)
  3. Max sequence capacity under reasoning workload
  4. Batch strategy comparator (mixed reasoning + standard)
  5. Speculative decoding speedup for reasoning (vs. standard)
  6. Deployment config generator (vLLM reasoning-optimized)
  7. Graceful drain planner for long reasoning requests
  8. Thinking budget trade-off analyzer

No external dependencies beyond the Python standard library.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Model specs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    n_layers: int
    n_kv_heads: int
    d_head: int
    n_params_b: float
    decode_tps_h100: float      # batch=1, BF16
    prefill_tps_h100: float     # large batch
    vram_weights_gb: float      # BF16 weights
    vram_weights_q4_gb: float   # Q4_K_M weights (llama.cpp)


MODELS = {
    "llama-3-8b": ModelSpec(
        "Llama-3-8B", 32, 8, 128, 8.0,
        decode_tps_h100=209.0, prefill_tps_h100=12_500.0,
        vram_weights_gb=16.0, vram_weights_q4_gb=5.0,
    ),
    "llama-3-70b": ModelSpec(
        "Llama-3-70B", 80, 8, 128, 70.0,
        decode_tps_h100=52.0, prefill_tps_h100=3_200.0,
        vram_weights_gb=140.0, vram_weights_q4_gb=42.0,
    ),
    "deepseek-r1-8b": ModelSpec(
        "DeepSeek-R1-Distill-8B", 32, 8, 128, 8.0,
        decode_tps_h100=195.0, prefill_tps_h100=11_000.0,
        vram_weights_gb=16.0, vram_weights_q4_gb=5.0,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. KV cache growth during reasoning trace
# ──────────────────────────────────────────────────────────────────────────────

def kv_bytes_per_token(m: ModelSpec, dtype_bytes: int = 2) -> int:
    return 2 * m.n_layers * m.n_kv_heads * m.d_head * dtype_bytes


def print_kv_growth_table():
    print("\n" + "=" * 72)
    print("  KV Cache Growth During Reasoning Trace")
    print("=" * 72)

    checkpoints = [512, 2_048, 8_192, 16_384, 32_768, 50_000, 65_536]

    for key in ["llama-3-8b", "llama-3-70b"]:
        m = MODELS[key]
        kv_per_tok = kv_bytes_per_token(m)
        print(f"\n  {m.name}  ({kv_per_tok // 1024} KB/token BF16):")
        print(f"    {'Tokens':>8}  {'KV Size':>10}  {'% of 80GB HBM':>15}  Bar")
        print(f"    {'-------':>8}  {'-------':>10}  {'-------------':>15}  ---")

        for t in checkpoints:
            kv_gb = t * kv_per_tok / 1e9
            pct   = kv_gb / 80.0 * 100
            bar   = "█" * min(30, int(pct / 2))
            flag  = " ← WARNING" if pct > 80 else (" ← danger" if pct > 60 else "")
            print(f"    {t:>8,}  {kv_gb:>9.2f}GB  {pct:>14.1f}%  {bar}{flag}")

    print("\n  Note: weights consume additional HBM")
    print(f"    Llama-3-8B BF16 weights:  16 GB → HBM left for KV: ~60 GB")
    print(f"    Llama-3-70B BF16 weights: need 2×H100 (160GB) → KV: ~20 GB/GPU")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cost model: standard vs. reasoning
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RequestProfile:
    name: str
    prompt_tokens: int
    output_tokens: int         # includes thinking tokens
    is_reasoning: bool


PROFILES = [
    RequestProfile("Short chat",        200,    200, False),
    RequestProfile("Code gen (std)",    800,    500, False),
    RequestProfile("RAG synthesis",    4096,    800, False),
    RequestProfile("Reasoning (easy)", 512,   4096, True),
    RequestProfile("Reasoning (med)",  512,  16384, True),
    RequestProfile("Reasoning (hard)", 512,  50000, True),
]

GPU_COST_PER_HR = 3.00   # H100 on-demand approximate


def print_cost_model():
    m = MODELS["llama-3-8b"]
    print("\n" + "=" * 76)
    print(f"  Cost Model: Standard vs. Reasoning  |  {m.name}  |  H100 ${GPU_COST_PER_HR}/hr")
    print("=" * 76)
    print(f"  {'Request':<22} {'Prompt':>7} {'Output':>7} {'Time(s)':>9}"
          f" {'$/req':>9} {'$/1M out':>10} {'Type':<10}")
    print("  " + "-" * 21 + "  " + "-" * 6 + "  " + "-" * 6 + "  " + "-" * 8 +
          "  " + "-" * 8 + "  " + "-" * 9 + "  " + "-" * 8)

    for p in PROFILES:
        pf_s   = p.prompt_tokens / m.prefill_tps_h100
        dec_s  = p.output_tokens / m.decode_tps_h100
        total_s = pf_s + dec_s
        cost_req = total_s / 3600 * GPU_COST_PER_HR
        cost_per_1m_out = cost_req / p.output_tokens * 1_000_000
        rtype = "reasoning" if p.is_reasoning else "standard"

        print(f"  {p.name:<22} {p.prompt_tokens:>7,} {p.output_tokens:>7,}"
              f" {total_s:>9.1f} ${cost_req:>8.4f} ${cost_per_1m_out:>9.2f}"
              f"  {rtype}")

    print("=" * 76)
    print("  Reasoning (hard) vs. Short chat: cost ratio =",
          end=" ")
    r_hard = PROFILES[5]
    r_chat = PROFILES[0]
    pf_r = r_hard.prompt_tokens / m.prefill_tps_h100
    dec_r = r_hard.output_tokens / m.decode_tps_h100
    pf_c = r_chat.prompt_tokens / m.prefill_tps_h100
    dec_c = r_chat.output_tokens / m.decode_tps_h100
    ratio = (pf_r + dec_r) / (pf_c + dec_c)
    print(f"{ratio:.0f}×")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Max sequence capacity under reasoning workload
# ──────────────────────────────────────────────────────────────────────────────

def compute_max_seqs(m: ModelSpec,
                      max_model_len: int,
                      gpu_hbm_gb: float = 80.0,
                      dtype_bytes: int = 2) -> int:
    """Max concurrent sequences given KV memory budget."""
    weight_gb = m.vram_weights_gb
    available_gb = gpu_hbm_gb - weight_gb
    kv_per_seq_gb = max_model_len * kv_bytes_per_token(m, dtype_bytes) / 1e9
    if kv_per_seq_gb <= 0:
        return 0
    return max(1, int(available_gb / kv_per_seq_gb))


def print_capacity_table():
    print("\n" + "=" * 72)
    print("  Max Concurrent Sequences  |  H100 80GB  |  KV budget only")
    print("=" * 72)
    print(f"  {'Model':<26} {'max_model_len':>14} {'KV/seq':>10} {'Max seqs':>10}")
    print("  " + "-" * 25 + "  " + "-" * 13 + "  " + "-" * 9 + "  " + "-" * 9)

    for key, m in MODELS.items():
        for mml in [4096, 8192, 32768, 65536]:
            kv_gb = mml * kv_bytes_per_token(m) / 1e9
            max_s = compute_max_seqs(m, mml)
            flag = " ← reasoning" if mml >= 32768 else ""
            print(f"  {m.name:<26} {mml:>14,} {kv_gb:>9.2f}GB {max_s:>10}{flag}")
        print()
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Batch strategy comparator
# ──────────────────────────────────────────────────────────────────────────────

def simulate_mixed_batch(n_standard: int,
                          n_reasoning: int,
                          std_output_tokens: int,
                          reason_output_tokens: int,
                          decode_tps: float) -> dict:
    """
    Simulate a mixed batch of standard + reasoning requests.
    Returns timing stats.
    """
    std_finish  = std_output_tokens    / decode_tps * (n_standard + n_reasoning)
    # reasoning finishes much later
    reason_finish = reason_output_tokens / decode_tps * (n_standard + n_reasoning)

    # After std requests finish, GPU only serves reasoning sequences
    # Average effective batch during decode
    total_time = reason_finish
    std_wait_after_finish = max(0.0, reason_finish - std_finish)

    return {
        "total_time_s": total_time,
        "std_finish_s": std_finish,
        "reason_finish_s": reason_finish,
        "gpu_util_pct": (n_reasoning / (n_standard + n_reasoning)) * 100
                         if total_time > 0 else 0,
    }


def print_batch_strategy():
    m = MODELS["llama-3-8b"]
    print("\n" + "=" * 70)
    print("  Batch Strategy Comparison  |  Llama-3-8B  |  H100")
    print("=" * 70)

    strategies = [
        ("Co-mixed (8 std + 2 reason)",  8, 2),
        ("Co-mixed (4 std + 4 reason)",  4, 4),
        ("Separate pools (std only)",   10, 0),
        ("Separate pools (reason only)", 0, 2),
    ]

    std_tokens    = 200
    reason_tokens = 32_000

    for label, n_std, n_reas in strategies:
        if n_std + n_reas == 0:
            continue

        # Simulate decode phases
        # Phase 1: all sequences decode together until std finishes
        batch_total = n_std + n_reas
        std_time  = std_tokens    / (m.decode_tps_h100 * batch_total) * batch_total
        # After std finishes: only reasoning left
        reas_time = (reason_tokens - std_tokens) / (m.decode_tps_h100 * n_reas) if n_reas > 0 else 0
        total     = std_time + reas_time

        std_ttft  = std_tokens    / (m.decode_tps_h100 * batch_total)
        reas_ttft = reason_tokens / (m.decode_tps_h100 * n_reas) if n_reas > 0 else 0

        print(f"\n  {label}")
        print(f"    Total wall time:  {total:8.1f}s")
        if n_std > 0:
            print(f"    Std TTFT (p50):  {std_ttft:8.1f}s")
        if n_reas > 0:
            print(f"    Reason time:     {reas_ttft:8.1f}s")

    print("\n  Recommendation: separate pools prevent standard requests from")
    print("  waiting behind 153-second reasoning traces.")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Speculative decoding speedup for reasoning
# ──────────────────────────────────────────────────────────────────────────────

def spec_speedup(alpha: float, K: int) -> float:
    """
    Expected speedup from speculative decoding.
    alpha = draft acceptance rate, K = draft tokens per step.
    speedup = (1 - alpha^(K+1)) / ((1 - alpha) * 1)
    But we also pay verification cost — approximate as 1 verification step per K draft.
    """
    expected_accepted = (1 - alpha ** (K + 1)) / (1 - alpha)
    # Each speculative step costs 1 verification pass (parallel) + draft overhead (~0.1)
    steps_per_speculative = 1.1
    return expected_accepted / steps_per_speculative


def print_speculative_analysis():
    m = MODELS["llama-3-8b"]
    print("\n" + "=" * 62)
    print("  Speculative Decoding for Reasoning vs. Standard")
    print("=" * 62)

    K_values   = [2, 4, 8]
    scenarios  = [
        ("Standard chat text",   0.80),
        ("Reasoning chain",      0.55),
        ("Adversarial reasoning",0.35),
    ]

    print(f"  {'Scenario':<26} {'Alpha':>6}", end="")
    for K in K_values:
        print(f"  K={K} speedup", end="")
    print()
    print("  " + "-" * 25 + "  " + "-" * 5, end="")
    for _ in K_values:
        print("  " + "-" * 10, end="")
    print()

    reason_tokens = 32_000
    for label, alpha in scenarios:
        print(f"  {label:<26} {alpha:>6.2f}", end="")
        for K in K_values:
            sp = spec_speedup(alpha, K)
            print(f"  {sp:>9.2f}×", end="")
        print()

    # Concrete time savings
    print(f"\n  Concrete savings for 32K reasoning trace ({m.name}, H100):")
    base_time = reason_tokens / m.decode_tps_h100
    for K in [4]:
        for label, alpha in scenarios:
            sp = spec_speedup(alpha, K)
            new_time = base_time / sp
            saved    = base_time - new_time
            print(f"    {label:<26} K={K}: {base_time:.0f}s → {new_time:.0f}s"
                  f"  (saved {saved:.0f}s)")
    print("=" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# 6. vLLM deployment config generator for reasoning
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningDeploymentConfig:
    model_id: str
    max_reasoning_tokens: int = 32768
    max_prompt_tokens: int    = 4096
    max_answer_tokens: int    = 1024
    gpu_hbm_gb: float         = 80.0
    dtype: str                = "bfloat16"


def generate_reasoning_vllm_config(cfg: ReasoningDeploymentConfig) -> None:
    m_key = "deepseek-r1-8b" if "r1" in cfg.model_id.lower() else "llama-3-8b"
    m = MODELS[m_key]

    max_prompt_tokens = cfg.max_prompt_tokens
    max_answer_tokens = cfg.max_answer_tokens
    max_model_len = max_prompt_tokens + cfg.max_reasoning_tokens + max_answer_tokens
    max_seqs      = compute_max_seqs(m, max_model_len, cfg.gpu_hbm_gb)
    # Be conservative: reserve 15% for overhead
    max_seqs_safe = max(1, int(max_seqs * 0.85))

    print(f"\n  {'=' * 58}")
    print(f"  vLLM Config Generator  |  Reasoning Workload")
    print(f"  {'=' * 58}")
    print(f"  Model: {cfg.model_id}")
    print(f"  max_model_len = {max_prompt_tokens} + {cfg.max_reasoning_tokens}"
          f" + {max_answer_tokens} = {max_model_len}")
    print(f"  max_model_len  = {max_model_len:,} tokens")
    print(f"  KV per seq     = {max_model_len * kv_bytes_per_token(m) / 1e9:.2f} GB")
    print(f"  max_num_seqs   = {max_seqs_safe}  (85% of theoretical {max_seqs})")
    print()

    print(f"""  # vLLM engine args (Python)
  engine_args = AsyncEngineArgs(
      model="{cfg.model_id}",
      max_model_len={max_model_len},
      max_num_seqs={max_seqs_safe},
      gpu_memory_utilization=0.92,
      dtype="{cfg.dtype}",
      enable_chunked_prefill=True,
      max_num_batched_tokens=4096,
      # Disable KV swap — reasoning traces are never cold
      swap_space=0,
  )""")

    print(f"""
  # SamplingParams for reasoning requests
  sampling = SamplingParams(
      max_tokens={cfg.max_reasoning_tokens + cfg.max_answer_tokens},
      temperature=0.6,
      priority=10,       # lower priority than standard requests
  )""")

    if max_seqs_safe <= 4:
        print(f"\n  WARNING: only {max_seqs_safe} concurrent reasoning sequences possible.")
        print(f"  Consider: Q4_K_M weights ({m.vram_weights_q4_gb} GB) to free more KV budget.")
        q4_seqs = compute_max_seqs(
            ModelSpec(m.name, m.n_layers, m.n_kv_heads, m.d_head,
                      m.n_params_b, m.decode_tps_h100, m.prefill_tps_h100,
                      m.vram_weights_q4_gb, m.vram_weights_q4_gb),
            max_model_len, cfg.gpu_hbm_gb
        )
        print(f"  With Q4_K_M: {int(q4_seqs * 0.85)} concurrent reasoning sequences")

    print(f"  {'=' * 58}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Thinking budget trade-off analyzer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThinkingBudgetResult:
    budget_tokens: int
    quality_score: float    # 0–1, simulated
    cost_usd: float
    latency_s: float


def simulate_thinking_budgets(model: ModelSpec,
                               prompt_tokens: int = 512) -> None:
    """
    Simulate how thinking budget affects quality vs. cost trade-off.
    Quality curve is log-saturating (diminishing returns).
    """
    budgets = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    print(f"\n  {'=' * 66}")
    print(f"  Thinking Budget Trade-off  |  {model.name}  |  H100")
    print(f"  {'=' * 66}")
    print(f"  {'Budget':>8}  {'Quality':>9}  {'Latency':>9}  {'$/req':>9}  Quality bar")
    print("  " + "-" * 7 + "  " + "-" * 8 + "  " + "-" * 8 + "  " +
          "-" * 8 + "  " + "-" * 20)

    # Simulate quality as log curve normalized at 32K
    q_max = math.log(32768)
    for b in budgets:
        quality = math.log(b) / q_max
        decode_s = (prompt_tokens + b) / model.decode_tps_h100
        cost     = decode_s / 3600 * GPU_COST_PER_HR
        bar_len  = int(quality * 20)
        bar      = "█" * bar_len

        flag = ""
        if b <= 512:
            flag = " ← fast & cheap"
        elif b >= 16384:
            flag = " ← high quality"

        print(f"  {b:>8,}  {quality:>8.2f}  {decode_s:>7.1f}s  ${cost:>8.4f}  {bar}{flag}")

    print(f"  {'=' * 66}")
    print(f"  Recommendation: match budget to task difficulty.")
    print(f"    Simple math/code:  budget=1024  (fast, cheap)")
    print(f"    Research/proofs:   budget=16384 (high quality)")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("█" * 64)
    print("  Chapter 24 — Reasoning Model Inference")
    print("█" * 64)

    print_kv_growth_table()
    print_cost_model()
    print_capacity_table()
    print_batch_strategy()
    print_speculative_analysis()

    cfg = ReasoningDeploymentConfig(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        max_reasoning_tokens=32768,
        max_prompt_tokens=4096,
        max_answer_tokens=1024,
    )
    generate_reasoning_vllm_config(cfg)

    simulate_thinking_budgets(MODELS["deepseek-r1-8b"])


if __name__ == "__main__":
    main()

```

## C++ — `reasoning_demo.cpp`

```cpp
// reasoning_demo.cpp — Chapter 24 — Reasoning Model Inference
//
// Compile:  g++ -std=c++17 -O2 -Wall -o reasoning_demo reasoning_demo.cpp
// Run:      ./reasoning_demo
//
// Covers: KV cache growth, cost model, capacity limits, batch strategy,
//         speculative decoding speedup, deployment sizing, thinking budgets.

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

static std::string fmt(double v, int prec = 1) {
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
// Model specs
// ─────────────────────────────────────────────────────────────────────────────

struct ModelSpec {
    std::string name;
    int    n_layers;
    int    n_kv_heads;
    int    d_head;
    double n_params_b;
    double decode_tps;
    double prefill_tps;
    double vram_bf16_gb;
    double vram_q4_gb;
};

static const std::vector<ModelSpec> MODELS = {
    {"Llama-3-8B",           32, 8, 128, 8.0,   209.0, 12500.0, 16.0,  5.0},
    {"Llama-3-70B",          80, 8, 128, 70.0,   52.0,  3200.0, 140.0, 42.0},
    {"DeepSeek-R1-Distill-8B",32, 8, 128, 8.0,  195.0, 11000.0, 16.0,  5.0},
};

static long long kv_bytes_per_token(const ModelSpec& m, int dtype_bytes = 2) {
    return static_cast<long long>(2) * m.n_layers * m.n_kv_heads
           * m.d_head * dtype_bytes;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. KV cache growth table
// ─────────────────────────────────────────────────────────────────────────────

static void print_kv_growth() {
    print_sep(72);
    std::cout << "  KV Cache Growth During Reasoning Trace\n";
    print_sep(72);

    std::vector<int> checkpoints = {512, 2048, 8192, 16384, 32768, 50000, 65536};

    for (int mi = 0; mi < 2; ++mi) {
        const auto& m = MODELS[static_cast<size_t>(mi)];
        long long kv_tok = kv_bytes_per_token(m);
        std::cout << "\n  " << m.name
                  << "  (" << kv_tok / 1024 << " KB/token BF16):\n";
        std::cout << "    " << std::right
                  << std::setw(8)  << "Tokens"
                  << std::setw(11) << "KV Size"
                  << std::setw(16) << "% 80GB HBM"
                  << "  Bar\n";
        std::cout << "    " << repeat_str("-", 7) << "  "
                  << repeat_str("-", 9) << "  "
                  << repeat_str("-", 13) << "  "
                  << repeat_str("-", 10) << "\n";

        for (int t : checkpoints) {
            double kv_gb  = static_cast<double>(t) * kv_tok / 1e9;
            double pct    = kv_gb / 80.0 * 100.0;
            int    bar_len = std::max(1, static_cast<int>(pct / 2.0));
            bar_len = std::min(bar_len, 30);
            std::string bar(static_cast<size_t>(bar_len), '#');
            std::string flag = (pct > 80) ? " <- WARNING" : (pct > 60) ? " <- danger" : "";

            std::cout << "    " << std::right
                      << std::setw(8)  << fmt_int(t)
                      << std::setw(10) << (fmt(kv_gb, 2) + "GB")
                      << std::setw(15) << (fmt(pct, 1) + "%")
                      << "  " << bar << flag << "\n";
        }
    }
    print_sep(72);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Cost model
// ─────────────────────────────────────────────────────────────────────────────

struct RequestProfile {
    std::string name;
    int         prompt_tokens;
    int         output_tokens;
    bool        is_reasoning;
};

static const double GPU_COST_HR = 3.00;

static void print_cost_model() {
    const auto& m = MODELS[0];  // Llama-3-8B

    std::vector<RequestProfile> profiles = {
        {"Short chat",        200,     200, false},
        {"Code gen (std)",    800,     500, false},
        {"RAG synthesis",    4096,     800, false},
        {"Reasoning (easy)", 512,    4096, true},
        {"Reasoning (med)",  512,   16384, true},
        {"Reasoning (hard)", 512,   50000, true},
    };

    print_sep(76);
    std::cout << "  Cost Model: Standard vs. Reasoning  |  "
              << m.name << "  |  H100 $" << fmt(GPU_COST_HR, 2) << "/hr\n";
    print_sep(76);

    std::cout << "  " << std::left  << std::setw(22) << "Request"
              << std::right
              << std::setw(8)  << "Prompt"
              << std::setw(8)  << "Output"
              << std::setw(10) << "Time(s)"
              << std::setw(10) << "$/req"
              << std::setw(12) << "$/1M out"
              << "  Type\n";
    std::cout << "  " << repeat_str("-", 21) << "  "
              << repeat_str("-", 7) << "  "
              << repeat_str("-", 7) << "  "
              << repeat_str("-", 8) << "  "
              << repeat_str("-", 8) << "  "
              << repeat_str("-", 10) << "  "
              << repeat_str("-", 9) << "\n";

    for (auto& p : profiles) {
        double pf_s    = p.prompt_tokens  / m.prefill_tps;
        double dec_s   = p.output_tokens  / m.decode_tps;
        double total_s = pf_s + dec_s;
        double cost    = total_s / 3600.0 * GPU_COST_HR;
        double per_1m  = cost / p.output_tokens * 1e6;

        std::cout << "  " << std::left  << std::setw(22) << p.name
                  << std::right
                  << std::setw(8)  << fmt_int(p.prompt_tokens)
                  << std::setw(8)  << fmt_int(p.output_tokens)
                  << std::setw(10) << fmt(total_s, 1)
                  << std::setw(9)  << ("$" + fmt(cost, 4))
                  << std::setw(11) << ("$" + fmt(per_1m, 2))
                  << "  " << (p.is_reasoning ? "reasoning" : "standard")
                  << "\n";
    }
    print_sep(76);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Max sequence capacity
// ─────────────────────────────────────────────────────────────────────────────

static int compute_max_seqs(const ModelSpec& m, int max_model_len,
                              double hbm_gb = 80.0) {
    double available_gb = hbm_gb - m.vram_bf16_gb;
    double kv_per_seq_gb = static_cast<double>(max_model_len)
                           * kv_bytes_per_token(m) / 1e9;
    if (kv_per_seq_gb <= 0) return 1;
    return std::max(1, static_cast<int>(available_gb / kv_per_seq_gb));
}

static void print_capacity_table() {
    print_sep(72);
    std::cout << "  Max Concurrent Sequences  |  H100 80GB\n";
    print_sep(72);
    std::cout << "  " << std::left  << std::setw(26) << "Model"
              << std::right
              << std::setw(15) << "max_model_len"
              << std::setw(11) << "KV/seq"
              << std::setw(11) << "Max seqs"
              << "\n";
    std::cout << "  " << repeat_str("-", 25) << "  "
              << repeat_str("-", 14) << "  "
              << repeat_str("-", 9) << "  "
              << repeat_str("-", 9) << "\n";

    for (const auto& m : MODELS) {
        for (int mml : {4096, 8192, 32768, 65536}) {
            double kv_gb = static_cast<double>(mml) * kv_bytes_per_token(m) / 1e9;
            int max_s    = compute_max_seqs(m, mml);
            std::string flag = (mml >= 32768) ? " <- reasoning" : "";
            std::cout << "  " << std::left  << std::setw(26) << m.name
                      << std::right
                      << std::setw(15) << fmt_int(mml)
                      << std::setw(10) << (fmt(kv_gb, 2) + "GB")
                      << std::setw(11) << max_s
                      << flag << "\n";
        }
        std::cout << "\n";
    }
    print_sep(72);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Speculative decoding speedup
// ─────────────────────────────────────────────────────────────────────────────

static double spec_speedup(double alpha, int K) {
    // Expected accepted tokens per step
    double expected = (1.0 - std::pow(alpha, K + 1)) / (1.0 - alpha);
    // Divide by step cost (verification + small draft overhead)
    return expected / 1.1;
}

static void print_speculative_analysis() {
    const auto& m = MODELS[0];  // Llama-3-8B
    print_sep(64);
    std::cout << "  Speculative Decoding for Reasoning vs. Standard\n";
    print_sep(64);

    struct Scenario { std::string name; double alpha; };
    std::vector<Scenario> scenarios = {
        {"Standard chat text",    0.80},
        {"Reasoning chain",       0.55},
        {"Adversarial reasoning", 0.35},
    };
    std::vector<int> K_vals = {2, 4, 8};

    std::cout << "  " << std::left << std::setw(26) << "Scenario"
              << std::right << std::setw(7) << "Alpha";
    for (int K : K_vals)
        std::cout << std::setw(11) << ("K=" + std::to_string(K));
    std::cout << "\n";
    std::cout << "  " << repeat_str("-", 25) << "  " << repeat_str("-", 6);
    for (size_t ki = 0; ki < K_vals.size(); ++ki) std::cout << "  " << repeat_str("-", 8);
    std::cout << "\n";

    for (auto& sc : scenarios) {
        std::cout << "  " << std::left << std::setw(26) << sc.name
                  << std::right << std::setw(7) << fmt(sc.alpha, 2);
        for (int K : K_vals)
            std::cout << std::setw(10) << (fmt(spec_speedup(sc.alpha, K), 2) + "x");
        std::cout << "\n";
    }

    // Concrete savings at K=4 for 32K reasoning trace
    int reason_tokens = 32000;
    double base_s = reason_tokens / m.decode_tps;
    std::cout << "\n  Concrete savings (32K trace, K=4):\n";
    for (auto& sc : scenarios) {
        double sp    = spec_speedup(sc.alpha, 4);
        double new_s = base_s / sp;
        double saved = base_s - new_s;
        std::cout << "    " << std::left << std::setw(26) << sc.name
                  << fmt(base_s, 0) << "s -> " << fmt(new_s, 0) << "s"
                  << "  (saved " << fmt(saved, 0) << "s)\n";
    }
    print_sep(64);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Thinking budget trade-off
// ─────────────────────────────────────────────────────────────────────────────

static void print_thinking_budgets() {
    const auto& m = MODELS[2];  // DeepSeek-R1-8B
    print_sep(66);
    std::cout << "  Thinking Budget Trade-off  |  " << m.name << "  |  H100\n";
    print_sep(66);

    std::cout << "  " << std::right
              << std::setw(8)  << "Budget"
              << std::setw(10) << "Quality"
              << std::setw(10) << "Latency"
              << std::setw(10) << "$/req"
              << "  Bar\n";
    std::cout << "  " << repeat_str("-", 7) << "  "
              << repeat_str("-", 8) << "  "
              << repeat_str("-", 8) << "  "
              << repeat_str("-", 8) << "  "
              << repeat_str("-", 20) << "\n";

    double q_max = std::log(32768.0);
    for (int b : {256, 512, 1024, 2048, 4096, 8192, 16384, 32768}) {
        double quality  = std::log(static_cast<double>(b)) / q_max;
        double decode_s = (512 + b) / m.decode_tps;
        double cost     = decode_s / 3600.0 * GPU_COST_HR;
        int    bar_len  = static_cast<int>(quality * 20);
        bar_len = std::max(1, bar_len);
        std::string bar(static_cast<size_t>(bar_len), '#');
        std::string flag = (b <= 512) ? " <- fast" : (b >= 16384) ? " <- high Q" : "";

        std::cout << "  " << std::right
                  << std::setw(8)  << fmt_int(b)
                  << std::setw(10) << fmt(quality, 2)
                  << std::setw(9)  << (fmt(decode_s, 1) + "s")
                  << std::setw(9)  << ("$" + fmt(cost, 4))
                  << "  " << bar << flag << "\n";
    }
    print_sep(66);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. vLLM config recommendation
// ─────────────────────────────────────────────────────────────────────────────

static void print_vllm_config(const ModelSpec& m,
                               int prompt_max, int reason_max, int answer_max,
                               double hbm_gb = 80.0) {
    int max_model_len = prompt_max + reason_max + answer_max;
    int max_seqs_raw  = compute_max_seqs(m, max_model_len, hbm_gb);
    int max_seqs_safe = std::max(1, static_cast<int>(max_seqs_raw * 0.85));
    double kv_per_seq_gb = static_cast<double>(max_model_len) * kv_bytes_per_token(m) / 1e9;

    print_sep(60);
    std::cout << "  vLLM Config  |  Reasoning  |  " << m.name << "\n";
    print_sep(60);
    std::cout << "  max_model_len = " << prompt_max << " + " << reason_max
              << " + " << answer_max << " = " << max_model_len << "\n";
    std::cout << "  KV per seq    = " << fmt(kv_per_seq_gb, 2) << " GB\n";
    std::cout << "  max_num_seqs  = " << max_seqs_safe
              << "  (85% of " << max_seqs_raw << ")\n\n";

    std::cout << "  engine_args = AsyncEngineArgs(\n"
              << "      model=\"" << m.name << "\",\n"
              << "      max_model_len=" << max_model_len << ",\n"
              << "      max_num_seqs=" << max_seqs_safe << ",\n"
              << "      gpu_memory_utilization=0.92,\n"
              << "      enable_chunked_prefill=True,\n"
              << "      swap_space=0,  # disable — traces never cold\n"
              << "  )\n";

    if (max_seqs_safe <= 4) {
        double kv_q4_gb = static_cast<double>(max_model_len) * kv_bytes_per_token(m) / 1e9;
        double avail_q4 = hbm_gb - m.vram_q4_gb;
        int q4_seqs     = std::max(1, static_cast<int>(avail_q4 / kv_q4_gb * 0.85));
        std::cout << "\n  WARNING: only " << max_seqs_safe << " concurrent seqs.\n";
        std::cout << "  With Q4_K_M weights (" << fmt(m.vram_q4_gb, 0) << "GB):"
                  << " ~" << q4_seqs << " concurrent seqs.\n";
    }
    print_sep(60);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    print_header("Chapter 24 — Reasoning Model Inference");

    print_kv_growth();
    print_cost_model();
    print_capacity_table();
    print_speculative_analysis();
    print_thinking_budgets();
    print_vllm_config(MODELS[2], 4096, 32768, 1024);

    return 0;
}

```

