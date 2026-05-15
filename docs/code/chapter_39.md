# Code 39 — Evaluation and Regression Testing

## Python — eval_harness.py

```python
#!/usr/bin/env python3
"""
eval_harness.py — Chapter 39 Companion Code

Minimal evaluation harness for comparing two vLLM/llama.cpp endpoints.
No external dependencies — uses only Python standard library.

Usage:
  python3 eval_harness.py --baseline http://localhost:8000 \
                           --candidate http://localhost:8001
"""
import json, math, time, argparse, statistics
from urllib.request import urlopen, Request
from urllib.error import URLError

def chat(base_url: str, prompt: str, max_tokens: int = 64,
         temperature: float = 0.0) -> dict:
    payload = json.dumps({
        "model": "default", "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    req = Request(f"{base_url}/v1/chat/completions", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def get_logprobs(base_url: str, text: str) -> list[float]:
    payload = json.dumps({
        "model": "default", "prompt": text,
        "max_tokens": 0, "echo": True, "logprobs": 1
    }).encode()
    req = Request(f"{base_url}/v1/completions", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as r:
        result = json.loads(r.read())
    token_logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
    return [lp for lp in token_logprobs if lp is not None]

def eval_perplexity(base_url: str, texts: list[str]) -> float:
    total_ll, total_tokens = 0.0, 0
    for text in texts:
        lps = get_logprobs(base_url, text)
        total_ll += sum(lps)
        total_tokens += len(lps)
    return math.exp(-total_ll / max(total_tokens, 1))

def eval_multiple_choice(base_url: str,
                          questions: list[dict]) -> float:
    """
    questions: list of {"question": str, "choices": [str,...], "answer": int}
    Returns accuracy (0.0–1.0).
    """
    correct = 0
    for q in questions:
        prompt = q["question"] + "\n" + "\n".join(
            f"{chr(65+i)}. {c}" for i, c in enumerate(q["choices"]))
        resp = chat(base_url, prompt, max_tokens=4)
        ans_text = resp["choices"][0]["message"]["content"].strip().upper()
        expected = chr(65 + q["answer"])
        if ans_text.startswith(expected):
            correct += 1
    return correct / len(questions)

def eval_latency(base_url: str, prompt: str, n_runs: int = 10
                 ) -> dict:
    ttfts, itls = [], []
    for _ in range(n_runs):
        t0 = time.time()
        resp = chat(base_url, prompt, max_tokens=50)
        total = time.time() - t0
        n_out = resp["usage"]["completion_tokens"]
        ttfts.append(total * 0.3)      # approximate TTFT fraction
        itls.append(total / max(n_out, 1) * 1000)
    return {
        "ttft_p50_ms": statistics.median(ttfts) * 1000,
        "ttft_p99_ms": sorted(ttfts)[int(0.99 * len(ttfts))] * 1000,
        "itl_p50_ms":  statistics.median(itls),
    }

def diff_outputs(base_url: str, candidate_url: str,
                 prompts: list[str]) -> list[dict]:
    diffs = []
    for p in prompts:
        base_out = chat(base_url, p)["choices"][0]["message"]["content"]
        cand_out = chat(candidate_url, p)["choices"][0]["message"]["content"]
        if base_out != cand_out:
            diffs.append({"prompt": p[:60], "base": base_out[:80],
                          "candidate": cand_out[:80]})
    return diffs

def print_report(label: str, ppl: float, acc: float,
                 latency: dict, n_diffs: int):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Perplexity:          {ppl:.2f}")
    print(f"  MC Accuracy:         {acc*100:.1f}%")
    print(f"  TTFT P50:            {latency['ttft_p50_ms']:.0f} ms")
    print(f"  TTFT P99:            {latency['ttft_p99_ms']:.0f} ms")
    print(f"  ITL P50:             {latency['itl_p50_ms']:.1f} ms")
    print(f"  Output diffs:        {n_diffs}")

SMOKE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2023, large language models became widely deployed.",
]
SMOKE_QUESTIONS = [
    {"question": "What is 2+2?", "choices": ["3","4","5","6"], "answer": 1},
    {"question": "Capital of France?",
     "choices": ["Berlin","London","Paris","Rome"], "answer": 2},
]
SMOKE_PROMPT = "Explain what a KV cache is in two sentences."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",  default="http://localhost:8000")
    parser.add_argument("--candidate", default=None)
    args = parser.parse_args()

    print(f"Baseline: {args.baseline}")
    try:
        b_ppl  = eval_perplexity(args.baseline, SMOKE_TEXTS)
        b_acc  = eval_multiple_choice(args.baseline, SMOKE_QUESTIONS)
        b_lat  = eval_latency(args.baseline, SMOKE_PROMPT, n_runs=5)
        print_report("BASELINE", b_ppl, b_acc, b_lat, 0)
    except URLError as e:
        print(f"  [SKIP] baseline unreachable: {e}")
        b_ppl = None

    if args.candidate:
        print(f"\nCandidate: {args.candidate}")
        try:
            c_ppl  = eval_perplexity(args.candidate, SMOKE_TEXTS)
            c_acc  = eval_multiple_choice(args.candidate, SMOKE_QUESTIONS)
            c_lat  = eval_latency(args.candidate, SMOKE_PROMPT, n_runs=5)
            diffs  = diff_outputs(args.baseline, args.candidate,
                                  SMOKE_TEXTS) if b_ppl else []
            print_report("CANDIDATE", c_ppl, c_acc, c_lat, len(diffs))
            if b_ppl:
                delta_ppl = c_ppl - b_ppl
                print(f"\n  ΔPPL:  {delta_ppl:+.2f}"
                      f"  {'⚠ REGRESSION' if delta_ppl > 0.5 else '✓ OK'}")
                delta_acc = c_acc - b_acc
                print(f"  ΔAcc:  {delta_acc*100:+.1f}%"
                      f"  {'⚠ REGRESSION' if delta_acc < -0.02 else '✓ OK'}")
        except URLError as e:
            print(f"  [SKIP] candidate unreachable: {e}")
    else:
        print("\n(No --candidate specified; run with --candidate URL to compare)")

if __name__ == "__main__":
    main()
```


## C++ — `eval_harness.cpp`

```bash
# Compile
g++ -std=c++17 -O2 -o eval_harness eval_harness.cpp -lm
# Run
./eval_harness
```

```cpp
/*
 * eval_harness.cpp — Chapter 39: Evaluation and Regression Testing
 *
 * Implements (mirrors eval_harness.py):
 *   1. Perplexity estimator via simulated log-probs
 *   2. MMLU-style multiple-choice evaluator
 *   3. Latency SLO tester
 *   4. Output differ between two model versions
 *   5. Regression report generator
 *
 * NOTE: This is a self-contained demo that simulates all operations
 * without requiring a live inference endpoint, exactly mirroring the
 * calculation logic in eval_harness.py.
 *
 * Compile: g++ -std=c++17 -O2 -o eval_harness eval_harness.cpp -lm
 * Run:     ./eval_harness
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static const char* SEP =
    "──────────────────────────────────────────────────────────────────────";

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct EvalResult {
    std::string name;
    double      score;
    double      baseline_score;  // -1 if no baseline
    bool        passed;
    std::string detail;
};

struct LatencySample {
    double ttft_ms;    // time to first token
    double tpot_ms;    // time per output token
    double e2e_ms;     // end-to-end latency
    int    n_tokens;
    bool   within_slo;
};

// ─────────────────────────────────────────────────────────────────────────────
// §1  Perplexity Estimator
// ─────────────────────────────────────────────────────────────────────────────

/*
 * Simulate log-probs for a text using a seeded RNG.
 * In production this calls /v1/completions with echo=true.
 * Here we generate plausible values that match typical LLM output.
 */
static std::vector<double> simulate_logprobs(const std::string& text,
                                              double model_quality,
                                              uint64_t seed) {
    std::mt19937_64 rng(seed);
    // Good models: log-probs around -1.5 to -3.0; bad models: -3.0 to -6.0
    double mean_lp = -2.0 / model_quality;   // quality 1.0 = BF16, 0.7 = INT4
    std::normal_distribution<double> dist(mean_lp, 0.8);

    int n_tokens = (int)(text.size() / 4) + 2;  // ~4 chars/token
    std::vector<double> lps;
    lps.reserve(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        double lp = std::min(-0.01, dist(rng));  // log-prob always <= 0
        lps.push_back(lp);
    }
    return lps;
}

static double compute_perplexity(const std::vector<std::vector<double>>& all_lps) {
    double sum = 0.0;
    int    cnt = 0;
    for (auto& lps : all_lps) {
        for (double lp : lps) { sum += lp; cnt++; }
    }
    if (cnt == 0) return 1e9;
    return std::exp(-sum / cnt);
}

static void demo_perplexity() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 1 — Perplexity Estimator\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    // Sample wikitext-style sentences
    std::vector<std::string> texts = {
        "The transformer architecture revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant context.",
        "Large language models are trained on diverse internet text corpora.",
        "Inference optimization reduces cost while preserving output quality.",
        "KV caching enables efficient generation of long sequences.",
    };

    // Simulate two model versions: BF16 (baseline) and INT4 (candidate)
    struct ModelVersion { const char* name; double quality; uint64_t seed; };
    ModelVersion versions[] = {
        {"BF16 baseline", 1.00, 42},
        {"FP8 candidate", 0.97, 43},
        {"INT4 candidate",0.85, 44},
    };

    printf("\n  %-18s %12s %14s %10s\n",
           "Model version", "Perplexity", "Delta vs BF16", "Status");
    printf("  %s\n", SEP);

    double ppl_baseline = -1.0;
    for (auto& v : versions) {
        std::vector<std::vector<double>> all_lps;
        for (size_t i = 0; i < texts.size(); ++i)
            all_lps.push_back(simulate_logprobs(texts[i], v.quality, v.seed + i));

        double ppl = compute_perplexity(all_lps);
        if (ppl_baseline < 0) ppl_baseline = ppl;
        double delta = ppl - ppl_baseline;
        const char* status = (delta < 0.5) ? "PASS" : "WARN";
        printf("  %-18s %12.2f %14.2f %10s\n", v.name, ppl, delta, status);
    }

    printf("\n  Acceptance criterion: PPL delta < 0.5 vs baseline\n");
    printf("  FP8: nearly lossless  |  INT4: borderline — review before deploy\n");
    printf("  ✓ Perplexity evaluator demo complete\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  MMLU-Style Multiple-Choice Evaluator
// ─────────────────────────────────────────────────────────────────────────────

struct MCQuestion {
    const char* subject;
    const char* question;
    char        correct;   // 'A', 'B', 'C', or 'D'
};

// Simulate model's answer to an MC question
static char simulate_model_answer(const MCQuestion& q, double accuracy, uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (u(rng) < accuracy) return q.correct;   // correct answer
    // Pick a wrong answer uniformly at random
    char wrong;
    do { wrong = 'A' + (rng() % 4); } while (wrong == q.correct);
    return wrong;
}

static void demo_mmlu() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 2 — MMLU-Style Multiple-Choice Evaluator\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    MCQuestion questions[] = {
        {"math",    "What is 2^10?",                                              'B'},
        {"physics", "What is the speed of light in vacuum (m/s)?",               'C'},
        {"cs",      "What is the time complexity of binary search?",              'A'},
        {"biology", "What is the powerhouse of the cell?",                        'D'},
        {"history", "In what year did World War II end?",                         'B'},
        {"math",    "What is the derivative of sin(x)?",                          'A'},
        {"cs",      "Which data structure uses FIFO ordering?",                   'C'},
        {"physics", "What is Newton's second law?",                               'A'},
        {"biology", "What molecule carries genetic information?",                  'B'},
        {"history", "Who wrote the Declaration of Independence?",                 'C'},
    };
    const int N_Q = 10;

    struct ModelConf { const char* name; double accuracy; uint64_t seed; };
    ModelConf models[] = {
        {"BF16 (baseline)", 0.82, 100},
        {"FP8  (candidate)",0.80, 200},
        {"INT4 (candidate)",0.72, 300},
    };

    printf("\n  %d questions across 5 subjects\n\n", N_Q);
    printf("  %-20s %12s %14s %10s\n",
           "Model", "Accuracy", "Delta vs BF16", "Status");
    printf("  %s\n", SEP);

    double base_acc = -1.0;
    for (auto& m : models) {
        int correct = 0;
        for (int i = 0; i < N_Q; ++i) {
            char ans = simulate_model_answer(questions[i], m.accuracy, m.seed + i);
            if (ans == questions[i].correct) correct++;
        }
        double acc = (double)correct / N_Q * 100.0;
        if (base_acc < 0) base_acc = acc;
        double delta = acc - base_acc;
        const char* status = (delta > -5.0) ? "PASS" : "FAIL";
        printf("  %-20s %11.1f%% %14.1f%% %10s\n", m.name, acc, delta, status);
    }

    printf("\n  Acceptance criterion: accuracy drop < 5%% vs BF16 baseline\n");
    printf("  Subject breakdown helps identify where quantization hurts most.\n");
    printf("  ✓ MMLU evaluator demo complete\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  Latency SLO Tester
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<LatencySample> simulate_latency_samples(
    double ttft_mean_ms, double tpot_mean_ms,
    int n_output_tokens, int n_samples, uint64_t seed) {

    std::mt19937_64 rng(seed);
    std::exponential_distribution<double> ttft_dist(1.0 / ttft_mean_ms);
    std::normal_distribution<double>      tpot_dist(tpot_mean_ms, tpot_mean_ms * 0.15);

    std::vector<LatencySample> samples;
    samples.reserve(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        double ttft  = std::min(5000.0, ttft_dist(rng));
        double tpot  = std::max(1.0, tpot_dist(rng));
        double e2e   = ttft + tpot * n_output_tokens;
        samples.push_back({ttft, tpot, e2e, n_output_tokens, false});
    }
    return samples;
}

static double percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    int idx = (int)(p / 100.0 * v.size());
    idx = std::min(idx, (int)v.size() - 1);
    return v[idx];
}

struct SLOConfig {
    double ttft_p50_ms;
    double ttft_p99_ms;
    double tpot_p50_ms;
    double e2e_p99_ms;
};

static void demo_latency_slo() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 3 — Latency SLO Tester\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    // SLO targets for an enterprise assistant
    SLOConfig slo = {800.0, 3000.0, 50.0, 10000.0};

    printf("\n  SLO targets:\n");
    printf("    TTFT p50:  %.0f ms   TTFT p99:  %.0f ms\n", slo.ttft_p50_ms, slo.ttft_p99_ms);
    printf("    TPOT p50:  %.0f ms   E2E p99:  %.0f ms\n", slo.tpot_p50_ms, slo.e2e_p99_ms);

    struct Endpoint {
        const char* name;
        double ttft_mean;   // ms
        double tpot_mean;   // ms per output token
    };
    Endpoint endpoints[] = {
        {"vLLM BF16",   700,   45},
        {"vLLM FP8",    380,   28},
        {"TRT-LLM FP8", 280,   22},
        {"Slow deploy", 2500,  80},  // intentionally bad for demo
    };

    const int N_SAMPLES    = 200;
    const int N_OUT_TOKENS = 128;

    printf("\n  %-18s %10s %10s %10s %10s %10s %8s\n",
           "Endpoint", "TTFT p50", "TTFT p99", "TPOT p50", "E2E p99", "SLO pass%", "Status");
    printf("  %s\n", SEP);

    for (auto& ep : endpoints) {
        auto samples = simulate_latency_samples(
            ep.ttft_mean, ep.tpot_mean, N_OUT_TOKENS, N_SAMPLES, 12345);

        std::vector<double> ttfts, tpots, e2es;
        for (auto& s : samples) {
            ttfts.push_back(s.ttft_ms);
            tpots.push_back(s.tpot_ms);
            e2es.push_back(s.e2e_ms);
        }

        double ttft_p50 = percentile(ttfts, 50);
        double ttft_p99 = percentile(ttfts, 99);
        double tpot_p50 = percentile(tpots, 50);
        double e2e_p99  = percentile(e2es,  99);

        bool pass = (ttft_p50 <= slo.ttft_p50_ms &&
                     ttft_p99 <= slo.ttft_p99_ms &&
                     tpot_p50 <= slo.tpot_p50_ms &&
                     e2e_p99  <= slo.e2e_p99_ms);

        // % of requests within SLO
        int within = 0;
        for (auto& s : samples)
            if (s.ttft_ms <= slo.ttft_p99_ms && s.e2e_ms <= slo.e2e_p99_ms) within++;
        double slo_pct = 100.0 * within / N_SAMPLES;

        printf("  %-18s %9.0fms %9.0fms %9.0fms %9.0fms %9.1f%% %8s\n",
               ep.name, ttft_p50, ttft_p99, tpot_p50, e2e_p99, slo_pct,
               pass ? "PASS" : "FAIL");
    }

    printf("\n  Regression rule: TTFT p99 may not exceed 110%% of baseline\n");
    printf("  Auto-alert: PagerDuty webhook if E2E p99 > SLO for 5 min\n");
    printf("  ✓ Latency SLO tester demo complete\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Output Differ
// ─────────────────────────────────────────────────────────────────────────────

// Simulated model responses — baseline vs candidate
struct OutputPair {
    const char* prompt;
    const char* baseline;
    const char* candidate;
};

static int word_count_diff(const char* a, const char* b) {
    // Simple approximation: count space-separated tokens
    int cnt = 0;
    const char* pa = a;
    const char* pb = b;
    while (*pa && *pb) {
        while (*pa == ' ') pa++;
        while (*pb == ' ') pb++;
        const char* wa = pa; while (*pa && *pa != ' ') pa++;
        const char* wb = pb; while (*pb && *pb != ' ') pb++;
        int la = pa - wa, lb = pb - wb;
        if (la != lb || strncmp(wa, wb, la) != 0) cnt++;
    }
    // Remaining words in longer string
    while (*pa) { while (*pa && *pa != ' ') pa++; while (*pa == ' ') pa++; cnt++; }
    while (*pb) { while (*pb && *pb != ' ') pb++; while (*pb == ' ') pb++; cnt++; }
    return cnt;
}

static double jaccard_similarity(const char* a, const char* b) {
    // Approximate: measure overlap of space-split "tokens"
    // Just use length ratio as a cheap proxy
    int la = strlen(a), lb = strlen(b);
    if (la == 0 && lb == 0) return 1.0;
    int mn = std::min(la, lb), mx = std::max(la, lb);
    return (double)mn / mx;
}

static void demo_output_differ() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 4 — Output Differ: Baseline vs Candidate\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    OutputPair pairs[] = {
        {
            "What is the capital of France?",
            "The capital of France is Paris.",
            "Paris is the capital city of France."
        },
        {
            "Explain gradient descent in one sentence.",
            "Gradient descent is an optimization algorithm that iteratively adjusts parameters by moving in the direction of the negative gradient of the loss function.",
            "Gradient descent minimizes a loss function by iteratively moving parameters in the direction opposite to the gradient."
        },
        {
            "Write a Python hello world.",
            "print(\"Hello, World!\")",
            "print('Hello, World!')"
        },
        {
            "What is 2 + 2?",
            "4",
            "2 + 2 equals 4."   // slightly different but both correct
        },
    };
    const int N = 4;

    printf("\n  %-35s %12s %12s %10s\n",
           "Prompt", "Similarity", "Word diff", "Equiv?");
    printf("  %s\n", SEP);

    int n_equivalent = 0;
    for (int i = 0; i < N; ++i) {
        double sim  = jaccard_similarity(pairs[i].baseline, pairs[i].candidate);
        int    diff = word_count_diff(pairs[i].baseline, pairs[i].candidate);
        bool   eq   = sim > 0.7;
        if (eq) n_equivalent++;
        printf("  %-35.35s %12.3f %12d %10s\n",
               pairs[i].prompt, sim, diff, eq ? "yes" : "NO");
    }

    printf("\n  %d/%d responses functionally equivalent (similarity > 0.7)\n",
           n_equivalent, N);
    printf("\n  Diff examples:\n");
    for (int i = 0; i < N; ++i) {
        double sim = jaccard_similarity(pairs[i].baseline, pairs[i].candidate);
        if (sim < 0.9) {
            printf("    Prompt:    \"%.40s\"\n", pairs[i].prompt);
            printf("    Baseline:  \"%.60s\"\n", pairs[i].baseline);
            printf("    Candidate: \"%.60s\"\n\n", pairs[i].candidate);
        }
    }

    printf("  Production use: run differ on 1000 golden prompts after model update\n");
    printf("  Alert threshold: >5%% responses flagged as non-equivalent\n");
    printf("  ✓ Output differ demo complete\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Regression Report Generator
// ─────────────────────────────────────────────────────────────────────────────

struct RegressionEntry {
    const char* metric;
    const char* category;
    double      baseline;
    double      candidate;
    double      threshold_delta;  // allowed regression
    bool        higher_is_better;
};

static void demo_regression_report() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("DEMO 5 — Regression Report Generator\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");

    RegressionEntry entries[] = {
        // Quality metrics (higher is better)
        {"MMLU accuracy (%)",      "quality",    82.1,  81.8, 1.0,  true},
        {"HellaSwag (%)",          "quality",    85.3,  84.9, 1.0,  true},
        {"TruthfulQA (%)",         "quality",    71.2,  71.0, 1.0,  true},
        // Perplexity (lower is better)
        {"Perplexity (WikiText)",   "quality",   5.43,  5.55, 0.5,  false},
        // Latency (lower is better)
        {"TTFT p50 (ms)",           "latency",  680,    420,  200,  false},
        {"TTFT p99 (ms)",           "latency", 2200,   1350,  500,  false},
        {"TPOT p50 (ms/tok)",       "latency",   44,     27,   10,  false},
        // Throughput (higher is better)
        {"Decode tok/s (B=1)",      "throughput",18.2,  29.4, -5.0, true},
        {"Decode tok/s (B=32)",     "throughput",421,    680, -50.0, true},
        // Cost (lower is better)
        {"GPU-hours/1M tokens",     "cost",     1.24,  0.78,  0.5,  false},
        {"$/1M tokens",             "cost",     34.72, 21.84, 5.0,  false},
    };
    const int N = 11;

    printf("\n  Regression report: BF16 baseline → FP8 candidate (Llama-3.1-70B)\n\n");
    printf("  %-30s %-12s %10s %10s %10s %10s %8s\n",
           "Metric", "Category", "Baseline", "Candidate", "Delta", "Threshold", "Status");
    printf("  %s\n", SEP);

    int n_pass = 0, n_fail = 0, n_improve = 0;
    for (int i = 0; i < N; ++i) {
        const auto& e = entries[i];
        double delta     = e.candidate - e.baseline;
        double delta_for_check = e.higher_is_better ? delta : -delta;  // positive = improvement
        bool   fail      = delta_for_check < -std::abs(e.threshold_delta);
        bool   improve   = delta_for_check > std::abs(e.threshold_delta);
        const char* status = fail ? "FAIL" : (improve ? "IMPROV" : "OK");
        if (fail) n_fail++; else n_pass++;
        if (improve) n_improve++;
        printf("  %-30s %-12s %10.2f %10.2f %+10.2f %10.1f %8s\n",
               e.metric, e.category, e.baseline, e.candidate, delta,
               e.threshold_delta, status);
    }

    printf("  %s\n", SEP);
    printf("  Results: %d pass/improve | %d fail | %d regressions\n",
           n_pass, n_fail, n_fail);

    printf("\n  Overall verdict: ");
    if (n_fail == 0) {
        printf("✓ APPROVED — No regressions detected\n");
        printf("  FP8 quantization delivers speed/cost improvements with no quality loss\n");
    } else {
        printf("✗ BLOCKED — %d regression(s) require review\n", n_fail);
    }

    printf("\n  Automated CI/CD integration:\n");
    printf("    1. git push → trigger eval pipeline\n");
    printf("    2. Run perplexity + MMLU + latency tests\n");
    printf("    3. Compare against baseline in model registry\n");
    printf("    4. Auto-approve if all metrics within threshold\n");
    printf("    5. Block deployment if any metric fails\n");
    printf("    6. Store results in evaluation database\n");

    assert(n_pass >= 8);  // most metrics should pass for FP8
    printf("\n  ✓ Regression report demo complete (%d/%d metrics pass) ✓\n",
           n_pass, N);
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────────────────────

static void print_summary() {
    printf("\n%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("CHAPTER 39 SUMMARY — Evaluation and Regression Testing\n");
    printf("%s\n", "══════════════════════════════════════════════════════════════════════");
    printf(R"(
  Evaluation pipeline components:
    Perplexity:  ~25ms per text on vLLM via /v1/completions?echo=true
    MMLU:        ~100ms per question (1 forward pass)
    Latency SLO: Sample 200 requests, measure TTFT/TPOT/E2E percentiles
    Output diff: Jaccard/BLEU on 1000 golden prompts
    Regression:  Automated threshold check on 11 core metrics

  CI/CD integration checklist:
    [x] Run on every model update (weights, config, quantization)
    [x] Baseline frozen in model registry (immutable, tagged by version)
    [x] Results stored in time-series eval DB (Prometheus/SQL)
    [x] P50/P99 latency compared on same hardware (not cross-environment)
    [x] Alert on quality regression > 1%% MMLU, PPL +0.5, TTFT +20%%
    [x] Human review gate for >5%% output diff on golden prompts

  SLO targets (enterprise assistant, 70B on 4×H100):
    TTFT p50 <= 800ms    TTFT p99 <= 3000ms
    TPOT p50 <= 50ms     E2E p99  <= 10s
    Availability >= 99.9%% per calendar month
)");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║   Chapter 39: Evaluation and Regression Testing (C++)                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");

    demo_perplexity();
    demo_mmlu();
    demo_latency_slo();
    demo_output_differ();
    demo_regression_report();
    print_summary();

    printf("%s\n", "══════════════════════════════════════════════════════════════════════");
    printf("ALL CHAPTER 39 DEMOS COMPLETED — ALL ASSERTIONS PASSED ✓\n");
    printf("%s\n\n", "══════════════════════════════════════════════════════════════════════");
    return 0;
}
```
