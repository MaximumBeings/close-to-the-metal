# Chapter 39 — Evaluation and Regression Testing for Inference Systems

> *"Shipping a new model version without an eval suite is not a deployment — it is a prayer."*

---

## Why This Chapter Exists

Part III covered how to deploy, scale, observe, and cost-engineer your inference stack. One gap remains: **how do you know it is correct?** Not fast — correct. The same tooling question applies when you upgrade vLLM, swap to a new quantised weight file, change tensor parallelism, or flip from BF16 to FP8. Each change can silently degrade output quality in ways that latency dashboards and error rates will never catch.

This chapter covers:

- Quality evaluation: perplexity, MMLU, and task-specific benchmarks
- Latency SLOs: TTFT and ITL targets, how to set them, how to monitor them
- A/B testing inference engines and model versions
- Regression suites: what to test, how to automate it, and how to wire it into CI
- Correctness verification: numerical checks, determinism testing, output diffing
- The human-in-the-loop tier: when automated evals are not enough

---

## 39.1 The Two Failure Modes

Inference systems fail in two distinct ways:

**Operational failure** — the service is slow, returns 5xx errors, OOMs, or drops requests. These are visible immediately in metrics and alerts. Chapters 16, 17, and 32 address them.

**Quality failure** — the service responds at normal latency but the responses are wrong, degraded, or subtly different from what the model should produce. These are invisible to infrastructure monitoring. Quality failures are introduced by:

- Weight file changes (new quantization, new checkpoint)
- Engine version upgrades (vLLM 0.4 → 0.5 changed sampling numerics)
- Configuration changes (tensor parallelism affects numerical precision)
- Prompt template changes (missing `add_generation_prompt=True` is silent)
- Hardware changes (A100 → H100 can change BF16 rounding)

**The evaluation suite is your regression test for quality failures.** It should run on every deployment, every model update, and every configuration change, in the same way unit tests run on every code commit.

---

## 39.2 Perplexity — The Baseline Quality Signal

Perplexity is the geometric mean of per-token inverse probability under the model:

```
PPL = exp( -1/N × Σᵢ log p(tᵢ | t₁..tᵢ₋₁) )
```

Lower is better. A well-behaved LLaMA 3 70B should achieve PPL ≈ 3.5–4.5 on WikiText-2 (BF16). A bad quantization might push it above 6.0.

**Practical perplexity eval:**

```python
# ppl_eval.py — quick perplexity check via vLLM log-probs
import json, math, statistics
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Load a fixed evaluation corpus (e.g., 500 WikiText-2 sentences)
with open("eval_corpus.jsonl") as f:
    samples = [json.loads(l)["text"] for l in f][:100]

log_probs_all = []
for text in samples:
    resp = client.completions.create(
        model="default",
        prompt=text,
        max_tokens=0,       # score only, no generation
        echo=True,
        logprobs=1,
    )
    lps = resp.choices[0].logprobs.token_logprobs
    log_probs_all.extend([lp for lp in lps if lp is not None])

ppl = math.exp(-statistics.mean(log_probs_all))
print(f"Perplexity: {ppl:.2f}")
```

**Regression threshold:** Flag a deployment if PPL increases by more than 0.5 points vs the baseline. For production systems handling high-stakes tasks, tighten to 0.2.

---

## 39.3 Task-Specific Benchmarks

Perplexity measures general language model quality. For production systems, you also need task-specific benchmarks that reflect your actual use case.

### 39.3.1 Standard Benchmarks

| Benchmark | What it measures | Good for |
|---|---|---|
| MMLU | 57-subject academic knowledge | General capability regression |
| HumanEval / MBPP | Code generation correctness | Coding assistants |
| GSM8K | Grade-school math | Reasoning quality |
| MT-Bench | Multi-turn instruction following | Chat assistants |
| HELM | Holistic language understanding | Broad regression suite |
| HellaSwag | Commonsense reasoning | Sanity check, fast |

### 39.3.2 Domain-Specific Eval Sets

For production deployments, the most valuable benchmark is one built from **your own traffic**:

1. Collect 500–1000 production prompts (anonymised, with user consent)
2. Generate "golden" responses with the highest-quality reference model (GPT-4o, Claude 3.5 Sonnet)
3. Score new deployments against goldens using an LLM judge:

```python
# llm_judge_eval.py
import json
from openai import OpenAI

judge = OpenAI()  # external judge model (GPT-4o etc.)
candidate = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

def judge_response(prompt: str, golden: str, candidate_resp: str) -> float:
    """Returns a score 1-5 (5 = matches golden quality)."""
    result = judge.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""You are evaluating the quality of an AI response.

Prompt: {prompt}

Golden response: {golden}

Candidate response: {candidate_resp}

Rate the candidate on a scale of 1-5 where:
5 = Equivalent quality to the golden response
4 = Slightly worse but acceptable  
3 = Noticeably worse
2 = Significantly degraded
1 = Wrong or harmful

Respond with just the number."""
        }]
    )
    return float(result.choices[0].message.content.strip())

# Run eval
with open("golden_eval_set.jsonl") as f:
    eval_set = [json.loads(l) for l in f]

scores = []
for item in eval_set[:100]:
    cand_resp = candidate.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": item["prompt"]}],
        max_tokens=512,
        temperature=0.0,   # deterministic for regression
    ).choices[0].message.content
    
    score = judge_response(item["prompt"], item["golden"], cand_resp)
    scores.append(score)
    print(f"Score: {score:.1f}  (prompt: {item['prompt'][:60]}...)")

mean_score = sum(scores) / len(scores)
print(f"\nMean judge score: {mean_score:.2f} / 5.00")
```

**Regression threshold:** Flag if mean judge score drops below baseline by 0.15 or more.

---

## 39.4 Latency SLOs: Setting and Monitoring Them

The right SLOs come from your use case, not from defaults:

### 39.4.1 Deriving SLOs from User Experience

| Use case | TTFT SLO | ITL SLO | Rationale |
|---|---|---|---|
| Chat assistant | p99 ≤ 500ms | p99 ≤ 30ms | Human perceives streaming as real-time below 30ms ITL |
| Code completion (inline) | p99 ≤ 150ms | p99 ≤ 20ms | Must complete before user finishes typing |
| RAG Q&A | p99 ≤ 1000ms | p99 ≤ 50ms | Users accept slight wait for factual answers |
| Batch processing (async) | p99 ≤ 30s | p50 ≤ 100ms | Throughput matters more than latency |
| Document summarization | p99 ≤ 5s | p50 ≤ 50ms | One-shot, user waits knowingly |

### 39.4.2 SLO Alerting in Prometheus

```yaml
# prometheus-slo-rules.yaml
groups:
- name: inference_slo
  rules:
  # TTFT p99 budget burn
  - alert: TTFTSLOBudgetBurn
    expr: |
      histogram_quantile(0.99,
        rate(vllm_time_to_first_token_seconds_bucket[5m])
      ) > 0.5
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "TTFT p99 > 500ms SLO"

  # ITL p99 budget burn
  - alert: ITLSLOBudgetBurn
    expr: |
      histogram_quantile(0.99,
        rate(vllm_inter_token_latency_seconds_bucket[5m])
      ) > 0.03
    for: 2m
    labels:
      severity: warning

  # Error rate budget burn
  - alert: InferenceErrorRateHigh
    expr: |
      rate(vllm_request_failure_total[5m]) /
      rate(vllm_request_total[5m]) > 0.001
    for: 1m
    labels:
      severity: critical
```

### 39.4.3 SLO Regression Testing

Before promoting a new deployment to production, run a synthetic load test and compare latency CDFs:

```bash
# slo_regression_test.sh
# Run against both baseline and candidate, compare p50/p95/p99

BASELINE_URL="http://baseline:8000"
CANDIDATE_URL="http://candidate:8000"

for URL in $BASELINE_URL $CANDIDATE_URL; do
  python -m vllm.entrypoints.openai.benchmark \
    --base-url $URL \
    --model default \
    --dataset-name sharegpt \
    --num-prompts 500 \
    --request-rate 10 \
    --output-json results_$(echo $URL | md5sum | cut -c1-8).json
done

# Compare with Python
python compare_results.py results_*.json
```

---

## 39.5 A/B Testing Inference Engines

When comparing two versions (engine upgrade, model version, quantization change), run proper A/B tests rather than sequential comparisons:

**Shadow mode testing** — production traffic goes to the baseline; a copy goes to the candidate. Candidate responses are not served to users but are evaluated offline:

```
                    ┌─────────────────┐
User Request ──────►│  Load Balancer  │──────► Baseline (serves user)
                    └────────┬────────┘
                             │ mirror 100% of traffic
                             ▼
                    ┌─────────────────┐
                    │    Candidate    │──────► Offline eval pipeline
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   LLM Judge /   │
                    │   PPL eval /    │──────► Comparison report
                    │   Latency diff  │
                    └─────────────────┘
```

**Canary testing** — send 5% of traffic to the candidate and monitor SLO metrics, error rates, and user-reported quality signals (thumbs down, regeneration rates) before full rollout.

**Gradual rollout thresholds:**

```
5% → 24h hold → check: PPL delta < 0.3, SLOs met, error rate < baseline + 0.05%
25% → 24h hold → check: same
50% → 24h hold → check: same  
100% → promote baseline to candidate
```

---

## 39.6 Regression Suite: What to Test Automatically

A complete regression suite for an inference deployment runs three tiers:

### Tier 1 — Fast smoke tests (< 2 minutes, run on every PR)

```python
# smoke_tests.py
import pytest
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

def test_basic_completion():
    """Model responds at all."""
    resp = client.completions.create(model="default",
        prompt="The capital of France is", max_tokens=5)
    assert "Paris" in resp.choices[0].text

def test_chat_format():
    """Chat template works correctly."""
    resp = client.chat.completions.create(model="default",
        messages=[{"role": "user", "content": "Say exactly: HELLO"}],
        max_tokens=10, temperature=0.0)
    assert "HELLO" in resp.choices[0].message.content

def test_streaming():
    """Streaming returns tokens."""
    chunks = list(client.completions.create(model="default",
        prompt="Count: 1 2 3", max_tokens=20, stream=True))
    assert len(chunks) > 3

def test_max_tokens_respected():
    """Max tokens limit is honoured."""
    resp = client.completions.create(model="default",
        prompt="Write a very long essay about", max_tokens=10)
    # Rough token count: allow 10 tokens ≈ 40 chars + BOS overhead
    assert len(resp.choices[0].text) < 100

def test_determinism():
    """temperature=0 is deterministic."""
    kwargs = dict(model="default", prompt="The answer is", max_tokens=20,
                  temperature=0.0)
    r1 = client.completions.create(**kwargs).choices[0].text
    r2 = client.completions.create(**kwargs).choices[0].text
    assert r1 == r2, f"Non-deterministic: '{r1}' vs '{r2}'"
```

### Tier 2 — Quality regression (< 10 minutes, run before any prod deploy)

```python
# quality_regression.py
import math, statistics, json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

BASELINES = {
    "perplexity_wikitext": 4.2,      # acceptable if within +0.5
    "mmlu_5shot_accuracy": 0.78,     # acceptable if within -0.02
    "hellaswag_accuracy": 0.81,      # acceptable if within -0.02
    "gsm8k_pass_at_1": 0.72,        # acceptable if within -0.03
}

def run_quality_checks() -> dict:
    results = {}
    # ... (call each benchmark function) ...
    return results

def check_regressions(results: dict) -> list[str]:
    failures = []
    thresholds = {
        "perplexity_wikitext": ("max_increase", 0.5),
        "mmlu_5shot_accuracy":  ("min_decrease", 0.02),
        "hellaswag_accuracy":   ("min_decrease", 0.02),
        "gsm8k_pass_at_1":      ("min_decrease", 0.03),
    }
    for metric, (direction, tolerance) in thresholds.items():
        baseline = BASELINES[metric]
        actual = results.get(metric)
        if actual is None:
            failures.append(f"MISSING: {metric}")
            continue
        if direction == "max_increase" and actual > baseline + tolerance:
            failures.append(f"REGRESSION {metric}: {actual:.3f} > {baseline:.3f} + {tolerance}")
        elif direction == "min_decrease" and actual < baseline - tolerance:
            failures.append(f"REGRESSION {metric}: {actual:.3f} < {baseline:.3f} - {tolerance}")
    return failures
```

### Tier 3 — Full eval suite (hours, run weekly and before major releases)

Full MMLU, HumanEval, MT-Bench, domain-specific golden set with LLM judge, multi-turn coherence tests, long-context (128K) tests.

---

## 39.7 Numerical Correctness Verification

Beyond output quality, verify that the numerical pipeline is correct after configuration changes:

### 39.7.1 Attention output verification

```python
# numerical_check.py
import numpy as np
import torch
from openai import OpenAI

def check_logprob_consistency(client: OpenAI, prompt: str,
                               expected_top_token: str,
                               min_logprob: float = -2.0) -> bool:
    """
    Verify that the model's top token for a known prompt matches expectation.
    Catches: wrong model loaded, broken tokenizer, sampling misconfiguration.
    """
    resp = client.completions.create(
        model="default",
        prompt=prompt,
        max_tokens=1,
        logprobs=5,
        temperature=0.0,
    )
    top_tokens = resp.choices[0].logprobs.top_logprobs[0]
    best_token = max(top_tokens, key=top_tokens.get)
    best_lp = top_tokens[best_token]
    
    if expected_top_token not in best_token:
        print(f"FAIL: expected '{expected_top_token}', got '{best_token}'")
        return False
    if best_lp < min_logprob:
        print(f"WARN: low confidence {best_lp:.3f} < {min_logprob}")
    return True

# Known prompt → token pairs (model-specific)
KNOWN_PAIRS = [
    ("The capital of France is", "Paris", -0.5),
    ("2 + 2 =", " 4", -0.5),
    ("def fibonacci(n):\n    if n <= 1:\n        return", " n", -1.0),
]

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
all_pass = all(check_logprob_consistency(client, p, t, lp) 
               for p, t, lp in KNOWN_PAIRS)
print("Numerical check:", "PASS" if all_pass else "FAIL")
```

### 39.7.2 KV cache consistency

Prompt caching and chunked prefill should produce identical outputs to non-cached execution:

```python
def test_kv_cache_consistency(client: OpenAI, prompt: str) -> bool:
    """
    Same prompt, different context lengths — outputs should be identical
    after the shared prefix (prefix caching / chunked prefill test).
    """
    # First call: fills KV cache
    r1 = client.completions.create(model="default", prompt=prompt,
        max_tokens=50, temperature=0.0)
    # Second call: should hit prefix cache
    r2 = client.completions.create(model="default", prompt=prompt,
        max_tokens=50, temperature=0.0)
    
    if r1.choices[0].text != r2.choices[0].text:
        print(f"KV CACHE INCONSISTENCY:\n  r1={r1.choices[0].text!r}\n  r2={r2.choices[0].text!r}")
        return False
    return True
```

---

## 39.8 Output Diffing Between Versions

When upgrading engines or weights, systematically diff outputs across the same prompts:

```python
# output_diff.py
import json, difflib
from pathlib import Path
from openai import OpenAI

def collect_outputs(client: OpenAI, prompts: list[str],
                    label: str, out_path: Path):
    outputs = {}
    for i, prompt in enumerate(prompts):
        resp = client.completions.create(
            model="default", prompt=prompt,
            max_tokens=200, temperature=0.0)
        outputs[prompt] = resp.choices[0].text
        if i % 10 == 0: print(f"  {label}: {i}/{len(prompts)}")
    out_path.write_text(json.dumps(outputs, indent=2))

def diff_outputs(path_a: Path, path_b: Path, label_a: str, label_b: str):
    a = json.loads(path_a.read_text())
    b = json.loads(path_b.read_text())
    
    identical = changed = 0
    for prompt in a:
        if a[prompt] == b.get(prompt):
            identical += 1
        else:
            changed += 1
            diff = difflib.unified_diff(
                a[prompt].splitlines(), b.get(prompt, "").splitlines(),
                fromfile=label_a, tofile=label_b, lineterm="")
            print(f"\nPROMPT: {prompt[:80]}")
            print("\n".join(list(diff)[:20]))
    
    print(f"\nSummary: {identical} identical, {changed} changed "
          f"({changed/(identical+changed)*100:.1f}% change rate)")
```

A healthy minor version upgrade should have < 1% output change rate. A quantization change (BF16 → INT4) might have 5–15%. An architecture change (new attention implementation) could have 20–40% but with equal or better quality.

---

## 39.9 Human-in-the-Loop Evaluation

Automated evals catch regressions but miss subtle degradations in style, safety, and task-specific nuance. The right human eval cadence:

| Event | Human eval scope | Turnaround |
|---|---|---|
| Minor version bump | 50 prompts, spot check | Same day |
| New model checkpoint | 200 prompts, full domain coverage | 2–3 days |
| New quantization | 100 prompts, quality-sensitive tasks | 1 day |
| Architecture change | 500 prompts, full rubric | 1 week |
| New use case / market | 1000 prompts, red-team included | 2 weeks |

**Human eval tooling.** Use a simple side-by-side comparison tool (e.g., Argilla, Label Studio, or a custom Flask app) where annotators rate A vs B without knowing which is the new version. Blind evaluation prevents anchoring bias.

**Minimum annotator agreement.** Require κ ≥ 0.6 (Cohen's kappa) between annotators before trusting results. If two annotators disagree on > 40% of comparisons, your rubric is underspecified.

---

## 39.10 The Eval Pipeline in CI/CD

Wire everything together:

```yaml
# .github/workflows/inference-eval.yml
name: Inference Regression Eval

on:
  push:
    paths:
      - 'serving/**'
      - 'configs/**'
  pull_request:
    paths:
      - 'serving/**'

jobs:
  smoke-tests:
    runs-on: [self-hosted, gpu]
    steps:
    - uses: actions/checkout@v4
    - name: Start vLLM candidate
      run: docker compose up -d vllm-candidate
    - name: Wait for readiness
      run: python scripts/wait_for_ready.py --url http://localhost:8000
    - name: Run smoke tests
      run: pytest tests/smoke_tests.py -v --timeout=120
    - name: Run numerical checks
      run: python scripts/numerical_check.py

  quality-regression:
    needs: smoke-tests
    runs-on: [self-hosted, gpu]
    steps:
    - name: Run quality regression
      run: python scripts/quality_regression.py --baseline-file baselines.json
    - name: Compare latency SLOs
      run: python scripts/latency_slo_check.py --duration 300 --rate 5
    - name: Post results to PR
      uses: actions/github-script@v7
      with:
        script: |
          const results = require('./eval_results.json');
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: `## Eval Results\n${results.summary}`
          });
```

---

## 39.11 Code: Evaluation Harness

See `code/chapter_39/eval_harness.py`.

```python
# eval_harness.py
# Chapter 39 — Evaluation and Regression Testing
#
# Implements:
#   1. Perplexity estimator via log-probs
#   2. MMLU-style multiple-choice evaluator
#   3. Latency SLO tester
#   4. Output differ between two endpoints
#   5. Regression report generator
#
# Requirements: pip install openai tqdm
# Usage: python eval_harness.py --baseline http://localhost:8000
#                               --candidate http://localhost:8001

import argparse, json, math, time, statistics, sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import urllib.request

# ── Minimal HTTP client (no external deps beyond openai) ─────────────────────

def chat(base_url: str, prompt: str, max_tokens: int = 100,
          temperature: float = 0.0) -> str:
    """Call /v1/completions and return generated text."""
    payload = json.dumps({
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["choices"][0]["text"]

def get_logprobs(base_url: str, text: str) -> list[float]:
    """Get per-token log-probs for perplexity calculation."""
    payload = json.dumps({
        "model": "default",
        "prompt": text,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        lps = data["choices"][0]["logprobs"]["token_logprobs"]
        return [lp for lp in lps if lp is not None]

# ── Eval components ───────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    name: str
    score: float
    baseline_score: Optional[float] = None
    delta: Optional[float] = None
    passed: bool = True
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.baseline_score is not None:
            self.delta = self.score - self.baseline_score

def eval_perplexity(base_url: str, texts: list[str]) -> float:
    """Compute perplexity over a list of texts."""
    all_lps = []
    for text in texts:
        try:
            lps = get_logprobs(base_url, text)
            all_lps.extend(lps)
        except Exception as e:
            print(f"  Warning: log-prob failed: {e}", file=sys.stderr)
    if not all_lps:
        return float("inf")
    return math.exp(-statistics.mean(all_lps))

def eval_multiple_choice(base_url: str,
                          questions: list[dict]) -> float:
    """
    Evaluate multiple-choice accuracy.
    Each question: {"prompt": str, "choices": ["A)..","B)..","C)..","D).."],
                    "answer": "A"}
    """
    correct = 0
    for q in questions:
        choices_text = "\n".join(q["choices"])
        full_prompt = f"{q['prompt']}\n\n{choices_text}\n\nAnswer:"
        try:
            answer = chat(base_url, full_prompt, max_tokens=5).strip()
            if answer.startswith(q["answer"]):
                correct += 1
        except Exception:
            pass
    return correct / len(questions) if questions else 0.0

def eval_latency(base_url: str, prompts: list[str],
                  n_requests: int = 50) -> dict:
    """Measure TTFT and throughput."""
    ttfts = []
    for prompt in (prompts * ((n_requests // len(prompts)) + 1))[:n_requests]:
        t0 = time.perf_counter()
        try:
            chat(base_url, prompt, max_tokens=1)
            ttfts.append(time.perf_counter() - t0)
        except Exception:
            pass
    if not ttfts:
        return {"p50": float("inf"), "p95": float("inf"), "p99": float("inf")}
    ttfts.sort()
    return {
        "p50": ttfts[int(len(ttfts)*0.50)],
        "p95": ttfts[int(len(ttfts)*0.95)],
        "p99": ttfts[int(len(ttfts)*0.99)],
        "mean": statistics.mean(ttfts),
    }

def diff_outputs(base_url_a: str, base_url_b: str,
                  prompts: list[str]) -> dict:
    """Compare outputs from two endpoints."""
    identical = changed = errors = 0
    diffs = []
    for prompt in prompts:
        try:
            a = chat(base_url_a, prompt, max_tokens=100)
            b = chat(base_url_b, prompt, max_tokens=100)
            if a == b:
                identical += 1
            else:
                changed += 1
                if len(diffs) < 5:
                    diffs.append({"prompt": prompt[:80], "a": a[:100], "b": b[:100]})
        except Exception:
            errors += 1
    total = identical + changed + errors
    return {
        "identical": identical,
        "changed": changed,
        "errors": errors,
        "change_rate": changed / total if total else 0,
        "sample_diffs": diffs,
    }

# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list[EvalResult], diff_result: Optional[dict] = None):
    print("\n" + "=" * 64)
    print("EVAL REGRESSION REPORT")
    print("=" * 64)

    all_passed = True
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        delta_str = f"  Δ={r.delta:+.4f}" if r.delta is not None else ""
        print(f"  {status}  {r.name:<30} score={r.score:.4f}{delta_str}")
        if not r.passed:
            all_passed = False

    if diff_result:
        rate = diff_result["change_rate"] * 100
        print(f"\n  Output change rate: {rate:.1f}%  "
              f"({diff_result['identical']} identical, "
              f"{diff_result['changed']} changed)")
        if diff_result["sample_diffs"]:
            print("\n  Sample diffs:")
            for d in diff_result["sample_diffs"][:2]:
                print(f"    Prompt: {d['prompt']}")
                print(f"    A: {d['a']!r}")
                print(f"    B: {d['b']!r}")

    print("\n" + "=" * 64)
    print(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 64 + "\n")
    return all_passed

# ── CLI ───────────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "The transformer architecture was introduced in 2017 by Vaswani et al.",
    "Python is a high-level programming language known for its readability.",
    "The KV cache stores key and value tensors from previous attention layers.",
    "Flash Attention reduces memory usage by computing attention in tiles.",
    "Speculative decoding uses a draft model to propose tokens for verification.",
]

SAMPLE_QUESTIONS = [
    {"prompt": "What is 2 + 2?", "choices": ["A) 3", "B) 4", "C) 5", "D) 6"], "answer": "B"},
    {"prompt": "What is the capital of France?", "choices": ["A) Berlin", "B) Madrid", "C) Paris", "D) Rome"], "answer": "C"},
    {"prompt": "Which is the largest planet?", "choices": ["A) Earth", "B) Mars", "C) Saturn", "D) Jupiter"], "answer": "D"},
]

LATENCY_PROMPTS = [
    "Summarize the transformer architecture in one sentence.",
    "What is a KV cache?",
    "Explain attention in simple terms.",
]

def main():
    parser = argparse.ArgumentParser(description="Inference eval harness")
    parser.add_argument("--baseline", default="http://localhost:8000")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--baseline-ppl", type=float, default=None)
    parser.add_argument("--baseline-acc", type=float, default=None)
    args = parser.parse_args()

    url = args.candidate or args.baseline
    print(f"\nEvaluating: {url}")

    results = []

    # Perplexity
    print("Running perplexity eval...")
    ppl = eval_perplexity(url, SAMPLE_TEXTS)
    r = EvalResult(name="perplexity", score=ppl,
                   baseline_score=args.baseline_ppl,
                   passed=(args.baseline_ppl is None or
                           ppl <= args.baseline_ppl + 0.5))
    results.append(r)
    print(f"  PPL = {ppl:.3f}")

    # Multiple choice
    print("Running multiple choice eval...")
    acc = eval_multiple_choice(url, SAMPLE_QUESTIONS)
    r = EvalResult(name="mc_accuracy", score=acc,
                   baseline_score=args.baseline_acc,
                   passed=(args.baseline_acc is None or
                           acc >= args.baseline_acc - 0.05))
    results.append(r)
    print(f"  Accuracy = {acc:.3f}")

    # Latency
    print("Running latency eval...")
    lat = eval_latency(url, LATENCY_PROMPTS, n_requests=20)
    r = EvalResult(name="latency_p99_s", score=lat["p99"],
                   passed=lat["p99"] < 2.0,
                   details=lat)
    results.append(r)
    print(f"  p50={lat['p50']:.3f}s  p99={lat['p99']:.3f}s")

    # Output diff (only if both endpoints provided)
    diff_result = None
    if args.candidate and args.baseline != args.candidate:
        print("Running output diff...")
        diff_result = diff_outputs(args.baseline, args.candidate,
                                    SAMPLE_TEXTS + LATENCY_PROMPTS)

    passed = print_report(results, diff_result)
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
```

---

## 39.12 Self-Check Questions

1. **[FOUNDATIONAL]** You upgrade vLLM from v0.4 to v0.5. Your smoke tests pass. Your perplexity eval shows PPL = 4.1 vs baseline 4.0. Should you promote the upgrade? What additional checks would you run before deciding?

2. **[SYSTEMS]** Your LLM judge eval shows a mean score drop from 4.2 to 3.9 after switching from BF16 to INT4 quantization. Your latency improved by 40%. How would you decide whether this trade-off is acceptable? What additional data would you collect?

3. **[DEEP DIVE]** Explain why `temperature=0.0` is required for regression testing but not for production deployment. What breaks if you use `temperature=0.7` in your regression suite?

4. **[APPLIED]** You have a 200-prompt golden eval set with LLM judge scores. You run it against a candidate model and get a mean score of 3.85 vs baseline 4.10. The standard deviation across prompts is 0.8. Is this drop statistically significant? What test would you use?

5. **[SYSTEMS]** Design a canary deployment strategy for a model upgrade where the new model performs better on coding tasks (+0.05 HumanEval) but slightly worse on general knowledge tasks (−0.02 MMLU). Your production traffic is 60% coding, 40% general. How would you measure and decide?

---

## Chapter Summary

Quality and correctness failures in inference systems are invisible to infrastructure monitoring — they require explicit evaluation pipelines. The core stack is: **perplexity** as a fast numerical health check, **task benchmarks** (MMLU, HumanEval, domain-specific golden sets) for capability regression, **latency SLO tests** for operational correctness, and **output diffing** to quantify change between versions.

Tier the eval suite by cost and frequency: smoke tests on every commit, quality regression before every deployment, full eval weekly. Wire them into CI/CD as first-class gates — a deployment that cannot pass its eval suite does not go to production.

Human-in-the-loop evaluation remains essential for major model changes, safety-sensitive tasks, and any case where automated metrics diverge from user perception. The combination of fast automated regression and periodic human review is the minimum viable quality process for a production inference system.

---

*Next: Appendix A — Mathematical Foundations*


---

## Self-Check Questions

1. Your eval suite shows perplexity increased from 8.2 to 9.7 after enabling INT8 KV cache quantization. Is this acceptable? What downstream task benchmark would you run to verify the quality impact? *(Section 39.2)*

2. A shadow mode A/B test runs the candidate model on 10% of traffic. After 48 hours, you have 12 000 candidate responses and 108 000 baseline responses. Is the sample size sufficient to detect a 5% difference in LLM judge score at 95% confidence? Show the power calculation. *(Section 39.5)*

3. Your regression smoke suite has 50 prompts and takes 4 minutes to run. The full quality suite has 5 000 prompts and takes 6 hours. Design a CI/CD pipeline that gates deployments using both suites, specifying at which pipeline stage each runs. *(Section 39.6)*

4. Numerical correctness testing compares logits between baseline and candidate with `allclose(atol=1e-3)`. 0.2% of logit positions exceed the tolerance. At what threshold of failing positions would you block a deployment, and why? *(Section 39.7)*

5. Output diffing on a canary deployment shows 3% of responses changed. How do you decide whether these changes are regressions or improvements, given you cannot manually review all 3% of changes? *(Section 39.8)*


---

## Worked Solutions

### Question 1 (Section 39.12 Foundational)
**vLLM upgrade v0.4 -> v0.5. Smoke tests pass. Perplexity: 4.1 vs baseline 4.0. Should you promote?**

**Do not promote yet.** A 2.5% perplexity increase (4.1 vs 4.0) is statistically meaningful -- perplexity is exponential in cross-entropy, so even small changes compound across long generations.

**Additional checks before deciding:**

1. **Downstream task benchmarks:** Run the eval suite on representative tasks (MMLU, HumanEval for code, summarization quality). Perplexity is a proxy; downstream task accuracy is what actually matters to users.

2. **Statistical significance:** Is the 4.1 vs 4.0 difference within measurement variance? Run the perplexity eval 3 times with different random seeds for decoding and check the standard deviation. If SD > 0.1, the 4.1 may not be significantly different from 4.0.

3. **Regression on known-sensitive prompts:** Run the full golden eval set. Check if specific categories (math, code, instruction following) regressed more than others.

4. **Output diffing:** Run 1,000 identical prompts through both versions with `temperature=0`. If >1% of outputs differ, investigate which changed and whether the changes are improvements or regressions.

5. **A/B shadow test:** Route 1% of traffic to the v0.5 deployment and compare LLM judge scores against v0.4 baseline for 48 hours before making a promotion decision.

---

### Question 2 (Section 39.12 Systems)
**LLM judge score drops 4.2 -> 3.9 after BF16 -> INT4. Latency improved 40%. Is the trade-off acceptable?**

**Framework for deciding:**

**Step 1 -- Quantify the quality drop:**
3.9 / 4.2 = 92.9% of baseline quality. A 7.1% quality reduction is meaningful for user-facing products.

**Step 2 -- Segment the quality drop:**
Run the eval separately on: (a) factual Q&A, (b) reasoning/math, (c) creative writing, (d) instruction following. INT4 quantization tends to hurt long-chain reasoning more than factual recall. If the 7.1% drop is concentrated in reasoning tasks but those are only 10% of your traffic, the weighted quality impact is only 0.7%.

**Step 3 -- Collect user signals:**
Run a 5% traffic A/B test: INT4 vs BF16. Measure: (a) thumbs up/down rate, (b) conversation continuation rate, (c) session abandonment rate. A 7.1% LLM judge score drop often does not translate to the same magnitude of user-perceived quality drop.

**Step 4 -- Cost-quality frontier:**
If the 40% latency improvement enables serving 40% more requests on the same hardware, this directly reduces $/request by 28% (1 - 1/1.4). For a $100K/month deployment, that's $28K/month saved. If user satisfaction drops by 2% (acceptable), this trade-off is favorable.

**Decision:** Acceptable if the quality drop is <5% on user-facing metrics AND the traffic segment affected by quality degradation (typically complex reasoning) is <15% of total traffic. Otherwise, consider INT8 quantization (typically <2% quality loss, ~20% latency improvement) as a middle ground.

---

### Question 3 (Section 39.12 Deep Dive)
**Why temperature=0.0 is required for regression testing but not production.**

**In regression testing:**
The goal is to detect changes in model behavior between two versions. With `temperature > 0`, outputs are stochastic -- the same model on the same prompt will produce different outputs on different runs. This makes it impossible to attribute output differences to the model version change vs. random sampling noise.

`temperature=0.0` uses greedy decoding (always select the highest-probability next token), making the output **deterministic** for a given input and model. You can compare two runs byte-for-byte and any difference is attributable to model or engine changes, not sampling randomness.

**What breaks with temperature=0.7 in regression suite:**
- Two runs of the SAME model version produce different outputs.
- Output diff reports show "regressions" that are just sampling variance.
- True regressions (actual model behavior changes) are masked by noise.
- Statistical tests become necessary and require large sample sizes (thousands of prompts per category) to detect real changes above the noise floor.

**In production:**
`temperature > 0` is appropriate because users benefit from diverse, creative responses and exact determinism is not a goal. The stochasticity makes the model feel more "human" and prevents identical responses to repeated queries.

---

### Question 4 (Section 39.12 Applied)
**200 prompts, candidate mean=3.85, baseline mean=4.10, SD=0.8. Is the drop statistically significant?**

**Two-sample t-test (assuming 200 prompts per model):**

Standard error of the difference in means:
```
SE = sqrt((SD_baseline^2/n) + (SD_candidate^2/n))
   = sqrt((0.8^2/200) + (0.8^2/200))
   = sqrt(0.0032 + 0.0032)
   = sqrt(0.0064)
   = 0.08
```

t-statistic:
```
t = (mean_baseline - mean_candidate) / SE
  = (4.10 - 3.85) / 0.08
  = 0.25 / 0.08
  = 3.125
```

For n=200 (large sample), the t-distribution approaches normal. t=3.125 corresponds to p < 0.002 (two-tailed). The drop is **statistically significant** at p < 0.01 level.

**Effect size (Cohen's d):**
```
d = (4.10 - 3.85) / 0.80 = 0.25 / 0.80 = 0.3125
```

A Cohen's d of 0.31 is a small-to-medium effect. Statistically significant but practically, the magnitude of degradation is modest.

**Recommendation:** Block the deployment if the score drop exceeds a pre-defined threshold (e.g., 0.20 points). At 0.25 drop with statistical significance, this warrants investigation of which prompt categories drove the regression before promoting.

---

### Question 5 (Section 39.12 Systems)
**Canary strategy: new model +0.05 HumanEval, -0.02 MMLU. Traffic: 60% coding, 40% general.**

**Step 1 -- Compute expected weighted quality change:**
```
weighted_change = 0.60 x (+0.05) + 0.40 x (-0.02)
               = 0.030 - 0.008
               = +0.022 net improvement
```

The new model is expected to be net-positive (+2.2%) on the production traffic mix. This suggests promoting, but with caution about the MMLU regression.

**Step 2 -- Canary deployment strategy:**
- Start at 5% traffic to the new model, 95% to baseline.
- Segment the canary: route 5% of *coding requests* and 5% of *general requests* separately.
- Run for 7 days minimum to collect statistically sufficient samples from both segments.

**Step 3 -- Measurement:**
Track separately:
- Coding requests (60% of traffic): HumanEval-style task completion rate, code execution success rate, user thumbs up/down on code responses.
- General knowledge requests (40% of traffic): factual accuracy (can spot-check vs ground truth), user satisfaction signals.

**Step 4 -- Decision criteria:**
- Promote if: coding segment shows improvement AND general segment degradation is within pre-defined tolerance (e.g., < 3% user satisfaction drop).
- Rollback if: general segment shows >5% thumbs-down increase.
- Continue canary at 10%, 25%, 50% before full rollout if both segments are within tolerance.

**Step 5 -- Safeguard:**
Keep the baseline model running on 50% of traffic for 14 days after full rollout to enable instant rollback if late-breaking issues emerge (e.g., hallucinations in specific domains not covered by MMLU).

---

### Question 6 (End-of-chapter)
**Perplexity 8.2 -> 9.7 after INT8 KV cache. Acceptable? What benchmark to run?**

Perplexity increase of 18.3% (9.7/8.2 = 1.183) is substantial -- this is a large quality degradation from KV quantization alone. INT8 KV cache typically causes <2% perplexity increase on well-calibrated implementations; an 18% increase suggests calibration issues or the model is particularly sensitive to KV precision.

**Not acceptable without further investigation.** Run: **MMLU** (factual accuracy), **HumanEval** (code correctness), and a domain-specific task benchmark matching your production use case. If these show <2% accuracy drop despite the perplexity increase, the perplexity metric may be misleading. If accuracy drops >5%, disable INT8 KV cache or switch to FP8 KV (better precision).

---

### Question 7 (End-of-chapter)
**Shadow A/B: 12,000 candidate + 108,000 baseline samples. Sufficient to detect 5% difference at 95% confidence?**

**Power calculation for two-proportion test:**
Assume LLM judge score is binary (good/bad) with baseline rate p_baseline = 0.75 (75% "good" responses). Detect 5% difference: p_candidate = 0.80 or 0.70.

Effect size h = 2 * arcsin(sqrt(p2)) - 2 * arcsin(sqrt(p1))
= 2 * arcsin(sqrt(0.80)) - 2 * arcsin(sqrt(0.75))
= 2 * 1.107 - 2 * 1.047 = 2.214 - 2.094 = 0.12

For alpha=0.05 (95% confidence), beta=0.20 (80% power):
n per group = (z_alpha/2 + z_beta)^2 / h^2 = (1.96 + 0.84)^2 / 0.12^2 = 7.84 / 0.0144 = 544 per group.

With 12,000 candidate samples and 108,000 baseline samples, **both groups far exceed the required 544**. The sample size is more than sufficient to detect a 5% difference at 95% confidence. The unequal sample sizes (10% vs 90% traffic split) reduce statistical efficiency slightly but with 12,000 candidate samples, power remains >99%.

---

### Question 8 (End-of-chapter)
**Smoke suite: 50 prompts, 4 minutes. Full quality suite: 5,000 prompts, 6 hours. CI/CD pipeline design:**

**Pipeline stages:**

```
Stage 1 (PR gate, ~5 min): Smoke suite (50 prompts)
  - Runs on every PR before merge
  - Gate: block merge if any smoke test fails
  - Threshold: 0/50 failures allowed

Stage 2 (Pre-deploy, ~4 min): Smoke suite again on staging
  - Verifies no environment-specific regression
  - Gate: block deploy if any failure

Stage 3 (Canary deploy, runs overnight): Full quality suite (5,000 prompts)
  - Runs after 5% canary deployment is live
  - Compares against baseline golden responses
  - Gate: rollback canary if mean LLM judge score drops >0.10 points
  - Timeline: starts running when canary traffic begins, completes in 6 hours

Stage 4 (Post-promotion, weekly): Full quality suite on 100% traffic
  - Scheduled weekly regression check
  - Catches drift that accumulates post-deployment
  - Alert (non-blocking) if score drops >0.05 from deployment baseline
```

**Key design decisions:**
- Smoke suite on every PR: fast feedback, catches obvious breakage.
- Full suite only post-canary: too slow for PR gates. Run it where it matters most -- after deployment, with real traffic patterns.
- Thresholds differ by stage: strict (0/50) for smoke, permissive (0.10 delta) for full suite.

---

### Question 9 (End-of-chapter)
**Numerical logit check: 0.2% of positions fail allclose(atol=1e-3). Threshold for blocking deployment?**

**Block if >0.1% of logit positions fail.** The 0.2% failure rate exceeds this threshold, so **block the deployment**.

**Reasoning:**
- Logit differences of >1e-3 at 0.2% of positions mean roughly 1 in 500 token predictions is meaningfully different between the baseline and candidate models.
- Over a 1,000-token generation, this affects ~2 token predictions on average.
- The practical impact depends on WHICH positions fail: if failures cluster on high-probability tokens (where the model is confident), the output is likely identical. If failures occur on low-probability "pivot" tokens (where the model choice is marginal), outputs could diverge significantly.

**Threshold rationale:**
- 0.01%: nearly perfect numerical match, safe to proceed.
- 0.05%: acceptable for most deployments; monitor output diff as secondary check.
- 0.1%: review failing positions; proceed if they are low-probability tokens.
- 0.2% (this case): **block** -- too many deviations to be confident outputs are equivalent. Investigate the source of numerical drift (mixed precision in a new layer, different CUDA kernel selection).

---

### Question 10 (End-of-chapter)
**Output diffing: 3% of canary responses changed. How to decide regressions vs improvements?**

**Strategy: Automated triage + sample review.**

**Step 1 -- LLM judge scoring at scale.**
For all changed responses, run an LLM judge (GPT-4 or Claude) that receives: (original prompt, baseline response, candidate response) and scores which is better. At 3% change rate with 100K daily requests = 3,000 changed responses/day -- too many for manual review, but LLM judge can score all at ~$0.01/comparison = $30/day.

```python
for prompt, base_resp, cand_resp in changed_responses:
    verdict = judge.compare(prompt, base_resp, cand_resp)
    # Returns: "baseline_better", "candidate_better", "equivalent"
```

**Step 2 -- Aggregate verdicts.**
If >60% of changed responses are judged "candidate_better" or "equivalent" -> likely improvements/neutral, proceed.
If >30% are judged "baseline_better" -> regressions present, block rollout.

**Step 3 -- Categorical breakdown.**
Cluster the changed responses by topic (using the routing classifier or simple keyword matching) to find regression-prone categories. A 3% change rate might be: 5% changed for coding prompts (improvement: more correct code) + 1% for general (neutral). Category-level analysis reveals whether the changes are beneficial.

**Step 4 -- Human sample review.**
Sample 100 changed responses across categories for manual human review. Calibrate the LLM judge against human judgments to ensure judge accuracy.

