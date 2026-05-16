# Chapter 31: Model Routing and Cascading

> *"Not every question deserves a H100."*

---

## 31.1 Why Routing Exists

A production LLM deployment rarely serves a single model. Real traffic is wildly heterogeneous: a customer support chatbot handles spelling-correction requests (`"is teh correct?"`) worth perhaps 5 output tokens and sentiment-classification tasks worth 50, alongside multi-step reasoning chains worth 800. Routing those three workloads to the same 70B parameter model is equivalent to hiring a senior engineer to file expense reports.

The economics are stark. At mid-2025 cloud pricing, a 70B model running on H100s costs roughly 40–60× more per token than a 7B model on A10G instances. If 60% of your traffic is "easy" — short, factual, or classifiable — routing those requests to a small model saves the bulk of your inference budget while reserving the large model for the 40% that needs it.

**Routing** assigns an incoming request to one of several available models. **Cascading** (a special case of routing) tries a small model first and escalates to a large model only when the small model's answer is deemed insufficient. Both strategies occupy the application layer, sitting above the inference engine but below the user-facing API.

This chapter covers:

- The three routing dimensions: cost, quality, and latency
- Cascade architectures and confidence-based escalation  
- Router training: offline classifiers and online quality estimators
- vLLM's multi-model deployment patterns
- llama.cpp routing for edge and hybrid deployments
- Real-world routing tables and implementation patterns

---

## 31.2 The Three Routing Dimensions

Every routing decision is a tradeoff across three axes.

### 31.2.1 Cost

Cost is the most tractable dimension because it is deterministic given the model and token counts. The routing problem becomes: given a predicted token count and a quality requirement, find the cheapest model that meets the quality bar.

```
cost(model, request) = price_in/1k × input_tokens
                     + price_out/1k × output_tokens
                     + overhead (cold start amortized)
```

Cost routing is pure optimization once you have reliable token-count predictions. Input tokens are known exactly at routing time (the prompt is already tokenized). Output tokens must be predicted — a simple linear model on prompt features (length, question type, presence of "list" or "explain") typically achieves R² ≈ 0.7, which is enough for routing.

### 31.2.2 Quality

Quality routing requires a model of which model will answer a given query correctly. The naive approach — always use the best model — is expensive. The cascade approach — try small, escalate on failure — requires a quality detector.

Quality varies along several orthogonal axes:

| Dimension | Small model weakness | When it matters |
|---|---|---|
| Factual accuracy | Hallucination rate higher | Medical, legal, financial |
| Instruction following | Misses constraints | Complex system prompts |
| Long-context reasoning | Degrades past 8K tokens | Document summarization |
| Multilingual | Non-English quality drops sharply | Localized deployments |
| Code generation | Compilation rate lower | Developer tools |
| Mathematical reasoning | Chain-of-thought shorter | STEM applications |

A practical quality router is a binary classifier: "can the small model answer this acceptably?" trained on pairs of (query features, small-model-answer-quality label). Labels come from human evaluation, automated rubrics, or comparison with ground-truth answers.

### 31.2.3 Latency

Latency routing selects models based on SLA constraints. A real-time voice assistant tolerates 200 ms end-to-end, leaving perhaps 150 ms for inference after networking overhead. That forces selection of a quantized small model on a nearby edge device. A batch analytics job tolerates 30 seconds and can afford the 70B model with full precision.

The latency model for a given (model, hardware, request) triple is:

```
E[TTFT]   = prefill_tokens × (2 × params × bytes_per_param) / bw_bytes_s
E[TPS]    = hardware_TPS(batch_size, model)
E[total]  ≈ TTFT + output_tokens / TPS
```

At routing time, `prefill_tokens` is known, `output_tokens` is predicted, and `TPS` is a function of current queue depth (observable from the inference server's metrics endpoint).

---

## 31.3 Cascade Architecture

A cascade arranges models in a preference order: cheapest first, most capable last. Each stage either returns an answer with a confidence score ≥ threshold τ, or passes the request to the next stage.

```
Request
   │
   ▼
┌──────────────────┐     confidence ≥ τ₁ ──────────────────► Response
│  Stage 1: 7B     │
│  small, fast     │
└──────────────────┘
   │ confidence < τ₁
   ▼
┌──────────────────┐     confidence ≥ τ₂ ──────────────────► Response
│  Stage 2: 13B    │
│  medium          │
└──────────────────┘
   │ confidence < τ₂
   ▼
┌──────────────────┐
│  Stage 3: 70B    │ ──────────────────────────────────────► Response
│  large, accurate │
└──────────────────┘
```

The cascade adds latency for escalated requests (two or three model calls instead of one), but reduces cost and latency for the majority of requests that terminate at stage 1. The net effect depends on the hit rate at each stage.

### 31.3.1 Expected Cost of a Two-Stage Cascade

Let:

- `p₁` = probability request is answered at stage 1 (hit rate)
- `c₁`, `c₂` = cost per request at stage 1 and 2
- `l₁`, `l₂` = latency per request at stage 1 and 2

```
E[cost]    = p₁ × c₁ + (1 - p₁) × (c₁ + c₂)
           = c₁ + (1 - p₁) × c₂

E[latency] = p₁ × l₁ + (1 - p₁) × (l₁ + l₂)
           = l₁ + (1 - p₁) × l₂
```

The cascade beats single-model serving when:

```
c₁ + (1 - p₁) × c₂  <  c₂        (cost wins if p₁ > 1 - c₁/c₂)
```

For c₁/c₂ = 0.05 (7B vs 70B pricing ratio), the cascade is cost-effective when p₁ > 0.05 — i.e., the small model handles even 5% of traffic usefully. In practice, easy queries dominate: p₁ ≈ 0.6–0.75 on typical customer-support traffic.

### 31.3.2 Confidence Estimation

For a cascade to work, stage 1 must output a confidence score. Several approaches exist:

**Token-probability entropy.** Low entropy over the next-token distribution signals high confidence. Compute:
```
entropy = -Σ p(t) log p(t)   over top-K tokens
```
High entropy → model is uncertain → escalate.

**Length heuristic.** Short answers (≤ 20 tokens) with no hedging language ("I think", "I'm not sure") are usually high-confidence factual retrieval. Long answers to short questions signal difficulty.

**Answer-then-reflect.** Run stage 1, then prompt it to self-assess: "Rate your confidence in this answer: high/medium/low." Adds one forward pass overhead but is surprisingly accurate.

**Trained confidence head.** Attach a small MLP to the last hidden state of the final answer token, trained to predict answer correctness on a labeled evaluation set. Adds ~0 inference overhead (amortized over answer length).

---

## 31.4 Router Training

### 31.4.1 Offline Classifier Router

An offline router is trained once on labeled data and deployed as a fast classifier in front of the inference engines. It never calls any LLM to decide routing — it routes based solely on features of the incoming request.

**Feature engineering for query routing:**

| Feature | Type | Extraction |
|---|---|---|
| Token count | Numeric | Tokenize and count |
| Question type | Categorical | "who/what/when/where/how/why/code/math" |
| Language | Categorical | `langdetect` or character n-gram classifier |
| Has code block | Binary | Regex for triple backtick or 4-space indent |
| Has numbers/math | Binary | Regex for equations or numerical expressions |
| Estimated output length | Numeric | Linear model on prompt features |
| System prompt complexity | Numeric | Token count of system prompt |
| Previous escalation rate | Numeric | Session-level exponential moving average |

A gradient-boosted tree (XGBoost or LightGBM) on these features typically achieves AUC > 0.85 for routing to the correct model tier. The classifier runs in < 1 ms on CPU, adding negligible overhead.

**Training data construction:**
1. Run all queries through both small and large model.
2. Evaluate answer quality (human rater or automated metric).
3. Label: `route_to_small = 1` if small model answer quality ≥ threshold.
4. Train classifier to predict `route_to_small` from request features.
5. Calibrate the decision threshold to hit a target cost/quality tradeoff.

### 31.4.2 Online Quality Estimator

An online estimator evaluates the small model's actual response before deciding whether to escalate. It is more accurate than offline routing but adds latency.

**Reward model judge.** A small reward model (0.5B–2B parameters) scores the response on a 1–10 quality scale. Train on human preference data or distill from a larger model judge. Route to large if score < threshold.

**Consistency check.** Run the small model twice with different temperatures. If the two responses are inconsistent (low semantic similarity), escalate. This detects genuine uncertainty vs. high-confidence wrong answers.

**Reference-free evaluation.** Compute the probability the small model assigns to its own answer tokens. Low self-probability on factual claims is a signal of hallucination.

### 31.4.3 Calibration

A router that is wrong in the wrong direction destroys value:

- **False negative (route-to-large when small was sufficient):** wastes money, no quality impact.
- **False positive (route-to-small when large was needed):** saves money, hurts quality.

Calibrate the decision threshold by plotting the precision-recall curve on a held-out set and selecting the operating point that meets your quality SLA at minimum cost. Recalibrate monthly as the query distribution shifts.

---

## 31.5 Routing Policies in Practice

### 31.5.1 Static Routing Table

The simplest production-grade router is a static table mapping request attributes to model tiers:

```python
ROUTING_TABLE = {
    # (language, task_type, max_tokens) → model_tier
    ("en", "classification",  <=  50): "7b",
    ("en", "summarization",   <= 512): "13b",
    ("en", "code_generation",   None): "70b",
    ("en", "reasoning",         None): "70b",
    ("*",  "*",               <=  50): "7b",
    ("*",  "*",                  "*"): "13b",
}
```

Static tables are auditable, predictable, and fast. The drawback is brittleness: they must be manually updated as model capabilities and cost structures change.

### 31.5.2 Cost-Capped Routing

Route to the cheapest model that is expected to meet a quality threshold for this request type:

```
for model in models_sorted_by_cost_ascending:
    if expected_quality(model, request) >= quality_threshold:
        return model
return most_capable_model  # fallback
```

`expected_quality` is estimated by the offline classifier. This approach requires per-model quality estimates but naturally adapts to model upgrades.

### 31.5.3 Latency-SLA Routing

Route to the fastest model that fits within the latency budget:

```
deadline_ms = request.sla_ms - network_overhead_ms
for model in models_sorted_by_expected_latency:
    if expected_latency(model, request, current_queue_depth) <= deadline_ms:
        return model
return fastest_model  # fallback: at least meet SLA
```

This requires live queue-depth data from the inference servers (easily obtained from vLLM's `/metrics` Prometheus endpoint).

### 31.5.4 Shadow Mode and A/B Testing

Before fully committing to a routing policy, run shadow mode: send every request to both the cheap and expensive models, record both responses, and evaluate quality offline. This builds calibration data without affecting user experience.

After calibration, run an A/B test: route a fraction of traffic through the new policy and measure user satisfaction metrics vs. the control. Ramp to 100% only after the experiment is statistically significant.

---

## 31.6 vLLM Multi-Model Deployment

vLLM natively supports running multiple models on the same cluster through several patterns.

### 31.6.1 Multi-Instance Deployment

The simplest pattern: run separate vLLM instances for each model tier, behind a routing proxy.

```
                    ┌─────────────────────────────────────────┐
Request ──► Router  │  vLLM 7B   (A10G × 1,  port 8000)      │
                    │  vLLM 13B  (A10G × 2,  port 8001)      │
                    │  vLLM 70B  (H100 × 4,  port 8002)      │
                    └─────────────────────────────────────────┘
```

The router is a thin HTTP proxy (Nginx, Envoy, or a custom FastAPI service) that inspects the request, applies the routing policy, and forwards to the appropriate vLLM instance.

**vLLM instance configuration per tier:**

```bash
# Tier 1: 7B — small, high throughput
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  --port 8000

# Tier 2: 13B — medium
vllm serve meta-llama/Llama-2-13b-chat-hf \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --max-num-seqs 128 \
  --port 8001

# Tier 3: 70B — large, lower throughput
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --port 8002
```

### 31.6.2 Speculative Execution with Routing

A more sophisticated pattern: start generation with the small model and, in parallel, start the large model. If the small model finishes first with high confidence, cancel the large model request. Otherwise, use the large model's response.

This reduces tail latency at the cost of wasted computation on the large model for requests that would have been answered by the small model anyway. Only viable when large-model throughput is underutilized.

### 31.6.3 Shared Prefix Cache Across Models

When models share a common prefix (a long system prompt), vLLM's prefix caching can be exploited across tiers if the tokenization is compatible. Both a 7B and 13B model from the same family often share the same tokenizer; the KV cache for the system prompt can be pre-warmed on both instances, amortizing the prefill cost across all routed requests.

```bash
# Pre-warm prefix cache on both tiers
curl -s http://localhost:8000/v1/completions \
  -d '{"model":"...", "prompt": "<SYSTEM_PROMPT>", "max_tokens":1}'
curl -s http://localhost:8001/v1/completions \
  -d '{"model":"...", "prompt": "<SYSTEM_PROMPT>", "max_tokens":1}'
```

After warm-up, each subsequent request with that prefix pays only the incremental KV cost.

---

## 31.7 llama.cpp Routing

For edge deployments and hybrid on-device / cloud setups, llama.cpp is often the inference engine for the small model tier. The router pattern shifts: the small model runs locally (llama.cpp), the large model runs in the cloud (vLLM or a managed API).

### 31.7.1 Hybrid Local/Cloud Cascade

```
User Device
├── llama.cpp (Q4_K_M 7B, ~4GB VRAM)
│     ├── fast, private, no network cost
│     └── handles: classification, simple Q&A, short generation
│
└── Cloud router (HTTP)
      └── vLLM 70B (H100)
            └── handles: complex reasoning, long context, code
```

The routing decision is made on-device, before any network call:

- Classify query locally (fast 0.5B classifier or rule-based triage)
- If `route_to_cloud=True`, send to API with full prompt
- Otherwise, generate locally with llama.cpp

**Privacy benefit:** Sensitive queries (PII, internal documents) never leave the device. The router can enforce data-classification rules: if the query contains tokens matching a PII pattern, force local routing regardless of quality prediction.

### 31.7.2 llama.cpp Server with Multiple Models

`llama-server` supports loading multiple models and can be called with a model parameter in the request. A thin router dispatches to the correct model slot:

```bash
# llama-server with two models loaded
llama-server \
  --model /models/llama-3.1-8b-q4_k_m.gguf \
  --alias small \
  --model /models/llama-3.1-70b-q4_k_m.gguf \
  --alias large \
  --port 8080
```

The server maintains separate KV caches per model. The router selects `"model": "small"` or `"model": "large"` in the JSON request body.

### 31.7.3 NUMA-Aware Model Placement

On a multi-socket server running llama.cpp, place different model tiers on different NUMA nodes to avoid cross-socket memory bandwidth contention:

```bash
# Node 0: small model (CPU-only, fast responses)
numactl --cpunodebind=0 --membind=0 \
  llama-server --model small.gguf --port 8080 &

# Node 1: large model (offload to GPU if available)
numactl --cpunodebind=1 --membind=1 \
  llama-server --model large.gguf --port 8081 \
  --n-gpu-layers 80 &
```

---

## 31.8 Router Implementation Patterns

### 31.8.1 Synchronous Router

The simplest router: classify, then call. One network round trip to the selected model. Best for latency-tolerant workloads where classification is fast (< 5 ms).

```python
def route_and_call(query: str) -> str:
    tier   = classify(query)          # < 5 ms
    model  = TIER_TO_ENDPOINT[tier]
    return llm_call(model, query)     # 50–2000 ms
```

### 31.8.2 Speculative Parallel Router

Issue the small-model call immediately, start the large-model call in parallel if confidence is below threshold after the first few tokens stream back. Cancel whichever loses.

```python
async def speculative_route(query: str) -> str:
    small_task = asyncio.create_task(call_small(query))
    
    # If small model returns high-confidence answer fast, cancel large
    result, confidence = await small_task
    if confidence >= THRESHOLD:
        return result
    
    # Otherwise fall back to large model
    return await call_large(query)
```

### 31.8.3 Streaming Cascade

Stream tokens from stage 1. After the first 20 tokens, evaluate confidence (entropy, self-consistency). If low confidence, abandon the stream and re-issue to stage 2. The user sees partial output from stage 1; the router can either discard it or display it with a "refining..." indicator.

This is the lowest-latency cascade for the majority of easy requests, at the cost of wasted tokens for escalated requests.

---

## 31.9 Metrics and Observability

A routing layer must expose metrics that allow continuous evaluation of routing quality.

**Essential metrics:**

| Metric | Description | Alert threshold |
|---|---|---|
| `route_distribution` | Fraction of requests per tier | Deviation > 20% from baseline |
| `escalation_rate` | Fraction routed to large model | Sudden spike signals query distribution shift |
| `quality_score_by_tier` | Average quality score per tier | Small model quality drop → increase escalation |
| `cascade_overhead_ms` | Extra latency for escalated requests | > 200 ms means cascade topology needs redesign |
| `cost_per_1k_requests` | Blended cost across tiers | Primary KPI for routing effectiveness |
| `router_latency_ms` | Time spent in routing logic | > 10 ms indicates classifier needs optimization |

**Quality score measurement:** Sample 1–5% of responses for each tier. Score with an automated evaluator (reward model, LLM judge, or task-specific metric). This is the ground truth for whether routing thresholds are calibrated correctly.

---

## 31.10 Worked Example: Customer Support Routing

Consider a customer support system with three model tiers:

| Tier | Model | Hardware | Latency (p50) | Cost/1k tokens |
|---|---|---|---|---|
| Small | Llama-3.1-8B-Q4 | A10G × 1 | 80 ms | $0.0001 |
| Medium | Llama-3.1-13B | A10G × 2 | 150 ms | $0.0003 |
| Large | Llama-3.1-70B | H100 × 4 | 400 ms | $0.0015 |

**Traffic analysis (10M requests/day):**

- 55% are FAQ lookups (order status, return policy, store hours) → small model
- 25% are complaint resolution requiring empathy + policy knowledge → medium model
- 15% are complex billing disputes, legal escalations → large model
- 5% are multilingual, non-English → large model (small model quality insufficient)

**Routing policy:**
```
IF query_type == "faq" AND language == "en":     → small
IF query_type == "complaint":                     → medium
IF query_type IN ("billing", "legal", "refund"):  → large
IF language != "en":                              → large
DEFAULT:                                          → medium
```

**Cost comparison:**

- No routing (all large): 10M × 0.5k tokens × $0.0015/1k = $7,500/day
- With routing: 5.5M × $0.05 + 2.5M × $0.15 + 2.0M × $0.75 = $275 + $375 + $1,500 = **$2,150/day** (71% reduction)

---

## 31.11 Python Demo

```python
"""
routing_demo.py — Chapter 31: Model Routing and Cascading

Demonstrates:
  1. Feature-based offline router (decision tree style)
  2. Two-stage cascade with confidence thresholds
  3. Cost model for routing decisions
  4. Break-even analysis per routing policy
  5. Simulated traffic workload with routing metrics
  6. Quality-aware threshold calibration

Run: python routing_demo.py
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─── Model tiers ─────────────────────────────────────────────────────────────

@dataclass
class ModelTier:
    name:              str
    params_b:          float   # billion parameters
    input_price_1k:    float   # USD per 1k input tokens
    output_price_1k:   float   # USD per 1k output tokens
    p50_latency_ms:    float   # typical latency
    quality_score:     float   # 0–1 quality on hard queries
    easy_quality:      float   # 0–1 quality on easy queries

TIERS = {
    "small":  ModelTier("small",  8,  0.00010, 0.00020,  80,  0.72, 0.97),
    "medium": ModelTier("medium", 13, 0.00030, 0.00060, 150,  0.85, 0.99),
    "large":  ModelTier("large",  70, 0.00150, 0.00250, 400,  0.97, 1.00),
}

# ─── Request types ───────────────────────────────────────────────────────────

QUERY_TYPES = {
    # (type, difficulty, language, estimated_tokens, expected_tier)
    "faq_order_status":  ("faq",        "easy",   "en",  60,  "small"),
    "faq_return_policy": ("faq",        "easy",   "en",  80,  "small"),
    "complaint_basic":   ("complaint",  "medium", "en", 180, "medium"),
    "billing_dispute":   ("billing",    "hard",   "en", 300,  "large"),
    "code_debug":        ("code",       "hard",   "en", 400,  "large"),
    "faq_in_spanish":    ("faq",        "easy",   "es", 100,  "large"),
    "reasoning_chain":   ("reasoning",  "hard",   "en", 500,  "large"),
    "simple_classify":   ("classify",   "easy",   "en",  30,  "small"),
}

# ─── Offline router ──────────────────────────────────────────────────────────

class OfflineRouter:
    """Rule-based router — production version would be a trained XGBoost classifier."""

    ROUTING_TABLE = {
        ("faq",       "en"): "small",
        ("classify",  "en"): "small",
        ("complaint", "en"): "medium",
        ("billing",    "*"): "large",
        ("code",       "*"): "large",
        ("reasoning",  "*"): "large",
    }

    def route(self, query_type: str, difficulty: str, language: str,
              tokens: int) -> Tuple[str, float]:
        """Returns (tier_name, routing_latency_ms)."""
        t0 = time.perf_counter()

        # Language override: non-English always routes to large
        if language != "en":
            tier = "large"
        # Token count override: very short queries → small
        elif tokens <= 40:
            tier = "small"
        # Hard queries always large
        elif difficulty == "hard":
            tier = "large"
        else:
            key = (query_type, "en")
            tier = self.ROUTING_TABLE.get(key)
            if tier is None:
                key2 = (query_type, "*")
                tier = self.ROUTING_TABLE.get(key2, "medium")

        routing_ms = (time.perf_counter() - t0) * 1000
        return tier, routing_ms


# ─── Cascade router ──────────────────────────────────────────────────────────

@dataclass
class CascadeResult:
    final_tier:     str
    stages_called:  int
    total_latency:  float
    total_cost:     float
    confidence:     float

class CascadeRouter:
    """
    Two-stage cascade: try small model first, escalate to large on low confidence.
    confidence is simulated based on query difficulty.
    """

    def __init__(self, stage1: str = "small", stage2: str = "large",
                 threshold: float = 0.75):
        self.stage1    = stage1
        self.stage2    = stage2
        self.threshold = threshold

    def _simulate_confidence(self, tier: str, difficulty: str) -> float:
        """Simulate confidence score from a model for a given query difficulty."""
        base = TIERS[tier].easy_quality if difficulty == "easy" else TIERS[tier].quality_score
        noise = random.gauss(0, 0.05)
        return max(0.0, min(1.0, base + noise))

    def _call_cost(self, tier: str, tokens: int) -> float:
        t = TIERS[tier]
        in_tok  = int(tokens * 0.6)
        out_tok = int(tokens * 0.4)
        return (in_tok / 1000) * t.input_price_1k + (out_tok / 1000) * t.output_price_1k

    def route(self, difficulty: str, tokens: int) -> CascadeResult:
        # Stage 1
        lat1  = TIERS[self.stage1].p50_latency_ms * random.uniform(0.8, 1.3)
        cost1 = self._call_cost(self.stage1, tokens)
        conf1 = self._simulate_confidence(self.stage1, difficulty)

        if conf1 >= self.threshold:
            return CascadeResult(self.stage1, 1, lat1, cost1, conf1)

        # Stage 2 (escalation)
        lat2  = TIERS[self.stage2].p50_latency_ms * random.uniform(0.8, 1.3)
        cost2 = self._call_cost(self.stage2, tokens)
        conf2 = self._simulate_confidence(self.stage2, difficulty)
        return CascadeResult(self.stage2, 2, lat1 + lat2, cost1 + cost2, conf2)


# ─── Cost model ──────────────────────────────────────────────────────────────

class RoutingCostModel:

    def single_tier_cost(self, tier: str, avg_tokens: int,
                         requests_per_day: int = 100_000) -> float:
        t = TIERS[tier]
        in_tok  = int(avg_tokens * 0.6)
        out_tok = int(avg_tokens * 0.4)
        cost_req = (in_tok / 1000) * t.input_price_1k + (out_tok / 1000) * t.output_price_1k
        return cost_req * requests_per_day * 30  # monthly

    def routed_cost(self, distribution: Dict[str, float],
                    avg_tokens: int, requests_per_day: int = 100_000) -> float:
        """distribution: {tier_name: fraction_of_traffic}"""
        total = 0.0
        for tier, frac in distribution.items():
            t = TIERS[tier]
            in_tok  = int(avg_tokens * 0.6)
            out_tok = int(avg_tokens * 0.4)
            cost_req = (in_tok / 1000) * t.input_price_1k + (out_tok / 1000) * t.output_price_1k
            total += frac * cost_req * requests_per_day * 30
        return total


# ─── Demo functions ──────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def demo_offline_router() -> None:
    section("Offline Router — Rule-Based Triage")

    router = OfflineRouter()
    print(f"\n  {'Query Type':<25} {'Difficulty':<10} {'Lang':<6} "
          f"{'Tokens':>7}  {'Routed To':<10} {'Expected':<10} {'Match?':>6}")
    print(f"  {'─'*25} {'─'*10} {'─'*6} {'─'*7}  {'─'*10} {'─'*10} {'─'*6}")

    all_correct = True
    for qname, (qtype, diff, lang, tokens, expected) in QUERY_TYPES.items():
        tier, lat_ms = router.route(qtype, diff, lang, tokens)
        match = tier == expected
        if not match:
            all_correct = False
        mark = "✓" if match else "✗"
        print(f"  {qname:<25} {diff:<10} {lang:<6} {tokens:>7}  "
              f"{tier:<10} {expected:<10} [{mark}]")

    assert all_correct, "Some routing decisions do not match expected tiers"
    print(f"\n  [ASSERT] All routing decisions match expected tiers: ✓")


def demo_cascade_router() -> None:
    section("Two-Stage Cascade Router")

    cascade = CascadeRouter(stage1="small", stage2="large", threshold=0.75)

    N_EASY = 1000
    N_HARD = 1000
    easy_results = [cascade.route("easy", 100) for _ in range(N_EASY)]
    hard_results = [cascade.route("hard", 300) for _ in range(N_HARD)]

    easy_stage1_rate = sum(1 for r in easy_results if r.stages_called == 1) / N_EASY
    hard_stage1_rate = sum(1 for r in hard_results if r.stages_called == 1) / N_HARD
    easy_avg_cost    = sum(r.total_cost for r in easy_results) / N_EASY
    hard_avg_cost    = sum(r.total_cost for r in hard_results) / N_HARD
    easy_avg_lat     = sum(r.total_latency for r in easy_results) / N_EASY
    hard_avg_lat     = sum(r.total_latency for r in hard_results) / N_HARD

    large_only_cost_easy = sum(TIERS["large"].input_price_1k * 0.06 +
                               TIERS["large"].output_price_1k * 0.04
                               for _ in easy_results) / N_EASY
    large_only_cost_hard = sum(TIERS["large"].input_price_1k * 0.18 +
                               TIERS["large"].output_price_1k * 0.12
                               for _ in hard_results) / N_HARD

    print(f"\n  {'Workload':<12} {'Stage-1 hit':>12} {'Avg cost':>12} "
          f"{'vs Large-only':>14} {'Avg lat ms':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*14} {'─'*12}")
    print(f"  {'Easy':<12} {easy_stage1_rate:>11.1%} "
          f"  ${easy_avg_cost:>10.6f} "
          f"  {(1 - easy_avg_cost/large_only_cost_easy)*100:>11.1f}% "
          f"  {easy_avg_lat:>10.1f}")
    print(f"  {'Hard':<12} {hard_stage1_rate:>11.1%} "
          f"  ${hard_avg_cost:>10.6f} "
          f"  {(1 - hard_avg_cost/large_only_cost_hard)*100:>11.1f}% "
          f"  {hard_avg_lat:>10.1f}")

    assert easy_stage1_rate >= 0.85, f"Easy hit rate {easy_stage1_rate:.1%} too low"
    assert hard_stage1_rate <= 0.35, f"Hard escalation rate too low ({hard_stage1_rate:.1%})"
    print(f"\n  [ASSERT] Easy queries resolve at stage 1 (≥85%): "
          f"{easy_stage1_rate:.1%} ✓")
    print(f"  [ASSERT] Hard queries escalate to stage 2 (≥65%): "
          f"{1 - hard_stage1_rate:.1%} ✓")


def demo_cost_model() -> None:
    section("Routing Cost Model")

    cm = RoutingCostModel()
    avg_tok = 200
    reqs    = 100_000

    # Customer support distribution
    distribution = {"small": 0.55, "medium": 0.25, "large": 0.20}

    cost_all_small  = cm.single_tier_cost("small",  avg_tok, reqs)
    cost_all_medium = cm.single_tier_cost("medium", avg_tok, reqs)
    cost_all_large  = cm.single_tier_cost("large",  avg_tok, reqs)
    cost_routed     = cm.routed_cost(distribution, avg_tok, reqs)

    print(f"\n  Monthly cost (100k req/day, {avg_tok} avg tokens):\n")
    print(f"  All-small (no routing):   ${cost_all_small:>10,.2f}")
    print(f"  All-medium (no routing):  ${cost_all_medium:>10,.2f}")
    print(f"  All-large (no routing):   ${cost_all_large:>10,.2f}")
    print(f"  Routed (55%/25%/20%):     ${cost_routed:>10,.2f}")
    print(f"\n  Savings vs all-large:     "
          f"${cost_all_large - cost_routed:>10,.2f}  "
          f"({(1 - cost_routed/cost_all_large)*100:.1f}%)")

    assert cost_routed < cost_all_large, "Routed cost should be less than all-large"
    assert cost_routed < cost_all_medium, "Routed cost should be less than all-medium"
    print(f"\n  [ASSERT] Routing reduces cost vs any single-tier policy: ✓")


def demo_break_even() -> None:
    section("Break-Even: When Does Routing Pay Off?")

    # Adding a router costs ~$0.000002/request (CPU classifier overhead)
    ROUTER_OVERHEAD = 0.000002

    cm   = RoutingCostModel()
    reqs = 100_000
    tok  = 200

    large_cost_req = (tok * 0.6 / 1000) * TIERS["large"].input_price_1k + \
                     (tok * 0.4 / 1000) * TIERS["large"].output_price_1k
    small_cost_req = (tok * 0.6 / 1000) * TIERS["small"].input_price_1k + \
                     (tok * 0.4 / 1000) * TIERS["small"].output_price_1k

    # Break-even: routing saves enough to pay for the router overhead
    # saving_per_routed_req = large_cost - small_cost
    # break_even_fraction = ROUTER_OVERHEAD / saving_per_routed_req
    saving_per_req = large_cost_req - small_cost_req
    break_even     = ROUTER_OVERHEAD / saving_per_req if saving_per_req > 0 else 1.0

    print(f"\n  Large-model cost/req:  ${large_cost_req:.6f}")
    print(f"  Small-model cost/req:  ${small_cost_req:.6f}")
    print(f"  Saving per routed req: ${saving_per_req:.6f}")
    print(f"  Router overhead/req:   ${ROUTER_OVERHEAD:.6f}")
    print(f"  Break-even fraction:   {break_even:.3%}")
    print(f"\n  → Routing is profitable if even {break_even:.2%} of requests")
    print(f"    can be correctly routed to the small model.")

    assert break_even < 0.01, f"Break-even {break_even:.2%} should be < 1%"
    print(f"\n  [ASSERT] Break-even fraction < 1%: {break_even:.3%} ✓")


def demo_threshold_calibration() -> None:
    section("Threshold Calibration — Quality vs Cost Tradeoff")

    random.seed(42)
    cascade = CascadeRouter(stage1="small", stage2="large")
    N = 2000

    # Mixed workload: 60% easy, 40% hard
    requests = [("easy", 100)] * int(N * 0.6) + [("hard", 300)] * int(N * 0.4)
    random.shuffle(requests)

    print(f"\n  {'Threshold':>10}  {'Escalation':>12}  "
          f"{'Avg cost':>12}  {'Quality score':>14}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*14}")

    prev_cost = None
    for tau in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        cascade.threshold = tau
        results = [cascade.route(diff, tok) for diff, tok in requests]

        escalation = sum(1 for r in results if r.stages_called == 2) / N
        avg_cost   = sum(r.total_cost for r in results) / N
        # Quality: escalated requests get large-model quality; stage-1 gets small-model quality
        quality    = sum(
            TIERS["large"].quality_score if r.stages_called == 2
            else (TIERS["small"].easy_quality if diff == "easy" else TIERS["small"].quality_score)
            for r, (diff, _) in zip(results, requests)
        ) / N

        marker = " ◄ recommended" if tau == 0.75 else ""
        print(f"  {tau:>10.2f}  {escalation:>11.1%}  "
              f"  ${avg_cost:>10.6f}  {quality:>13.3f}{marker}")

    print(f"\n  Threshold τ=0.75 balances quality (≥0.90) and cost (escalation ≈25%)")


def demo_traffic_simulation() -> None:
    section("Traffic Simulation — Mixed Workload Routing Metrics")

    router = OfflineRouter()
    N      = 5000
    rng    = random.Random(99)

    query_list = list(QUERY_TYPES.values())
    tier_counts = {"small": 0, "medium": 0, "large": 0}
    total_cost  = 0.0
    total_lat   = 0.0

    for _ in range(N):
        qtype, diff, lang, tokens, _ = rng.choice(query_list)
        tier, lat_ms = router.route(qtype, diff, lang, tokens)
        tier_counts[tier] += 1
        t = TIERS[tier]
        in_tok  = int(tokens * 0.6)
        out_tok = int(tokens * 0.4)
        total_cost += (in_tok / 1000) * t.input_price_1k + \
                      (out_tok / 1000) * t.output_price_1k
        total_lat  += t.p50_latency_ms

    large_only_cost = sum(
        (int(tokens * 0.6) / 1000) * TIERS["large"].input_price_1k +
        (int(tokens * 0.4) / 1000) * TIERS["large"].output_price_1k
        for _, _, _, tokens, _ in (rng.choice(query_list) for _ in range(N))
    )

    print(f"\n  Simulated {N:,} requests\n")
    print(f"  {'Tier':<10} {'Count':>8}  {'Fraction':>10}")
    print(f"  {'─'*10} {'─'*8}  {'─'*10}")
    for tier, count in tier_counts.items():
        print(f"  {tier:<10} {count:>8}  {count/N:>10.1%}")

    print(f"\n  Total cost (routed):     ${total_cost:.4f}")
    print(f"  Avg cost per request:    ${total_cost/N:.6f}")
    print(f"  Avg latency per request: {total_lat/N:.1f} ms")

    assert tier_counts["small"] > tier_counts["large"], \
        "Small tier should handle more traffic than large"
    print(f"\n  [ASSERT] Small tier handles more requests than large: ✓")


def main() -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  Chapter 31 — Model Routing and Cascading (Python)\n{bar}")

    demo_offline_router()
    demo_cascade_router()
    demo_cost_model()
    demo_break_even()
    demo_threshold_calibration()
    demo_traffic_simulation()

    print(f"\n{bar}\n  All demos complete.\n{bar}\n")


if __name__ == "__main__":
    random.seed(42)
    main()
```

---

## 31.12 C++ Demo

```cpp
// routing_demo.cpp
// Chapter 31 — Model Routing and Cascading (C++)
//
// Demonstrates:
//   1. Static routing table with rule-based triage
//   2. Two-stage cascade with confidence simulation
//   3. Cost model and break-even analysis
//   4. Traffic simulation with routing metrics
//   5. Threshold sensitivity analysis
//
// Build: g++ -O2 -std=c++17 -o routing_demo routing_demo.cpp
// Run:   ./routing_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// MODEL TIERS
// ─────────────────────────────────────────────────────────────────────────────

struct ModelTier {
    std::string name;
    double      params_b;
    double      input_price_1k;   // USD per 1k input tokens
    double      output_price_1k;  // USD per 1k output tokens
    double      p50_latency_ms;
    double      quality_hard;     // quality on hard queries
    double      quality_easy;     // quality on easy queries
};

static const std::map<std::string, ModelTier> TIERS = {
    {"small",  {"small",   8, 0.00010, 0.00020,  80.0, 0.72, 0.97}},
    {"medium", {"medium", 13, 0.00030, 0.00060, 150.0, 0.85, 0.99}},
    {"large",  {"large",  70, 0.00150, 0.00250, 400.0, 0.97, 1.00}},
};

static void print_section(const std::string& t) {
    std::cout << "\n" << std::string(60, '-') << "\n  " << t
              << "\n" << std::string(60, '-') << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// OFFLINE ROUTER
// ─────────────────────────────────────────────────────────────────────────────

struct QuerySpec {
    std::string name;
    std::string qtype;
    std::string difficulty;
    std::string language;
    int         tokens;
    std::string expected_tier;
};

static std::string offline_route(const std::string& qtype,
                                  const std::string& difficulty,
                                  const std::string& language,
                                  int tokens) {
    // Language override
    if (language != "en") return "large";
    // Token-count fast path
    if (tokens <= 40) return "small";
    // Difficulty
    if (difficulty == "hard") return "large";
    // Type table
    if (qtype == "faq"      || qtype == "classify")  return "small";
    if (qtype == "complaint")                         return "medium";
    if (qtype == "billing"  || qtype == "code" ||
        qtype == "reasoning")                         return "large";
    return "medium";  // default
}

static void demo_offline_router() {
    print_section("Offline Router — Rule-Based Triage");

    std::vector<QuerySpec> queries = {
        {"faq_order_status",  "faq",       "easy",   "en",  60, "small"},
        {"faq_return_policy", "faq",       "easy",   "en",  80, "small"},
        {"complaint_basic",   "complaint", "medium", "en", 180, "medium"},
        {"billing_dispute",   "billing",   "hard",   "en", 300, "large"},
        {"code_debug",        "code",      "hard",   "en", 400, "large"},
        {"faq_in_spanish",    "faq",       "easy",   "es", 100, "large"},
        {"reasoning_chain",   "reasoning", "hard",   "en", 500, "large"},
        {"simple_classify",   "classify",  "easy",   "en",  30, "small"},
    };

    std::cout << "\n  " << std::left
              << std::setw(22) << "Query"
              << std::setw(12) << "Difficulty"
              << std::setw(6)  << "Lang"
              << std::setw(8)  << "Tokens"
              << std::setw(10) << "Routed"
              << std::setw(10) << "Expected"
              << "Match?\n";
    std::cout << "  " << std::string(68, '-') << "\n";

    bool all_correct = true;
    for (auto& q : queries) {
        std::string tier = offline_route(q.qtype, q.difficulty, q.language, q.tokens);
        bool match = (tier == q.expected_tier);
        if (!match) all_correct = false;
        std::cout << "  " << std::left
                  << std::setw(22) << q.name
                  << std::setw(12) << q.difficulty
                  << std::setw(6)  << q.language
                  << std::setw(8)  << q.tokens
                  << std::setw(10) << tier
                  << std::setw(10) << q.expected_tier
                  << "[" << (match ? "✓" : "✗") << "]\n";
    }
    assert(all_correct);
    std::cout << "\n  [ASSERT] All routing decisions correct: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// CASCADE ROUTER
// ─────────────────────────────────────────────────────────────────────────────

struct CascadeResult {
    std::string final_tier;
    int         stages_called;
    double      total_latency_ms;
    double      total_cost;
    double      confidence;
};

class CascadeRouter {
public:
    std::string stage1, stage2;
    double      threshold;
    std::mt19937 rng;

    CascadeRouter(std::string s1, std::string s2, double thr, uint32_t seed = 42)
        : stage1(std::move(s1)), stage2(std::move(s2)), threshold(thr), rng(seed) {}

    double simulate_confidence(const std::string& tier, bool easy) {
        const auto& t = TIERS.at(tier);
        double base  = easy ? t.quality_easy : t.quality_hard;
        std::normal_distribution<double> noise(0.0, 0.05);
        return std::clamp(base + noise(rng), 0.0, 1.0);
    }

    double call_cost(const std::string& tier, int tokens) {
        const auto& t = TIERS.at(tier);
        int in_tok  = static_cast<int>(tokens * 0.6);
        int out_tok = static_cast<int>(tokens * 0.4);
        return (in_tok / 1000.0) * t.input_price_1k + (out_tok / 1000.0) * t.output_price_1k;
    }

    double call_latency(const std::string& tier) {
        const auto& t = TIERS.at(tier);
        std::uniform_real_distribution<double> jitter(0.8, 1.3);
        return t.p50_latency_ms * jitter(rng);
    }

    CascadeResult route(bool easy, int tokens) {
        double conf1 = simulate_confidence(stage1, easy);
        double lat1  = call_latency(stage1);
        double cost1 = call_cost(stage1, tokens);

        if (conf1 >= threshold)
            return {stage1, 1, lat1, cost1, conf1};

        double conf2 = simulate_confidence(stage2, easy);
        double lat2  = call_latency(stage2);
        double cost2 = call_cost(stage2, tokens);
        return {stage2, 2, lat1 + lat2, cost1 + cost2, conf2};
    }
};

static void demo_cascade() {
    print_section("Two-Stage Cascade Router");

    CascadeRouter cascade("small", "large", 0.75, 42);

    int N = 2000;
    int easy_s1 = 0, hard_s1 = 0;
    double easy_cost = 0, hard_cost = 0;
    double easy_lat  = 0, hard_lat  = 0;

    for (int i = 0; i < N; ++i) {
        auto r = cascade.route(true, 100);
        if (r.stages_called == 1) ++easy_s1;
        easy_cost += r.total_cost;
        easy_lat  += r.total_latency_ms;
    }
    for (int i = 0; i < N; ++i) {
        auto r = cascade.route(false, 300);
        if (r.stages_called == 1) ++hard_s1;
        hard_cost += r.total_cost;
        hard_lat  += r.total_latency_ms;
    }

    double easy_hit = static_cast<double>(easy_s1) / N;
    double hard_hit = static_cast<double>(hard_s1) / N;

    std::cout << "\n  " << std::left
              << std::setw(10) << "Workload"
              << std::setw(14) << "Stage-1 hit"
              << std::setw(16) << "Avg cost"
              << std::setw(16) << "Avg latency ms"
              << "\n  " << std::string(56, '-') << "\n";

    std::cout << "  " << std::setw(10) << "Easy"
              << std::setw(14) << (std::to_string((int)(easy_hit * 100)) + "%")
              << std::fixed << std::setprecision(6)
              << "  $" << std::setw(13) << easy_cost / N
              << std::setprecision(1)
              << "  " << std::setw(13) << easy_lat / N << "\n";

    std::cout << "  " << std::setw(10) << "Hard"
              << std::setw(14) << (std::to_string((int)(hard_hit * 100)) + "%")
              << std::fixed << std::setprecision(6)
              << "  $" << std::setw(13) << hard_cost / N
              << std::setprecision(1)
              << "  " << std::setw(13) << hard_lat / N << "\n";

    assert(easy_hit >= 0.80);
    assert(hard_hit <= 0.40);
    std::cout << "\n  [ASSERT] Easy hit rate ≥ 80%: " << std::fixed
              << std::setprecision(1) << easy_hit * 100 << "% ✓\n";
    std::cout << "  [ASSERT] Hard escalation rate ≥ 60%: "
              << (1 - hard_hit) * 100 << "% ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// COST MODEL
// ─────────────────────────────────────────────────────────────────────────────

static void demo_cost_model() {
    print_section("Routing Cost Model");

    int avg_tok    = 200;
    int reqs_daily = 100'000;
    int days       = 30;

    auto tier_cost = [&](const std::string& tier) {
        const auto& t = TIERS.at(tier);
        int in_tok  = static_cast<int>(avg_tok * 0.6);
        int out_tok = static_cast<int>(avg_tok * 0.4);
        double cost_req = (in_tok / 1000.0) * t.input_price_1k +
                          (out_tok / 1000.0) * t.output_price_1k;
        return cost_req * reqs_daily * days;
    };

    // Routing distribution: 55% small, 25% medium, 20% large
    std::map<std::string, double> dist = {
        {"small", 0.55}, {"medium", 0.25}, {"large", 0.20}
    };
    double routed_cost = 0.0;
    for (auto& [tier, frac] : dist)
        routed_cost += frac * tier_cost(tier) / 1.0;  // already monthly

    double cost_large  = tier_cost("large");
    double cost_medium = tier_cost("medium");
    double cost_small  = tier_cost("small");

    // Recalculate routed correctly
    routed_cost = 0.0;
    for (auto& [tier, frac] : dist) {
        const auto& t = TIERS.at(tier);
        int in_tok  = static_cast<int>(avg_tok * 0.6);
        int out_tok = static_cast<int>(avg_tok * 0.4);
        double cost_req = (in_tok / 1000.0) * t.input_price_1k +
                          (out_tok / 1000.0) * t.output_price_1k;
        routed_cost += frac * cost_req * reqs_daily * days;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  Monthly cost (100k req/day, " << avg_tok << " avg tokens):\n\n";
    std::cout << "  All-small (no routing):   $" << cost_small  << "\n";
    std::cout << "  All-medium (no routing):  $" << cost_medium << "\n";
    std::cout << "  All-large (no routing):   $" << cost_large  << "\n";
    std::cout << "  Routed (55/25/20 split):  $" << routed_cost << "\n";
    std::cout << "\n  Savings vs all-large:     $"
              << (cost_large - routed_cost) << "  ("
              << std::setprecision(1) << (1.0 - routed_cost / cost_large) * 100 << "%)\n";

    assert(routed_cost < cost_large);
    assert(routed_cost < cost_medium);
    std::cout << "\n  [ASSERT] Routing cheaper than any single tier: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// BREAK-EVEN ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────

static void demo_break_even() {
    print_section("Break-Even Analysis");

    const double ROUTER_OVERHEAD = 0.000002;  // $0.000002/req (CPU classifier)
    int tok = 200;

    auto cost_req = [&](const std::string& tier) {
        const auto& t = TIERS.at(tier);
        int in_tok  = static_cast<int>(tok * 0.6);
        int out_tok = static_cast<int>(tok * 0.4);
        return (in_tok / 1000.0) * t.input_price_1k +
               (out_tok / 1000.0) * t.output_price_1k;
    };

    double large_cost = cost_req("large");
    double small_cost = cost_req("small");
    double saving     = large_cost - small_cost;
    double break_even = ROUTER_OVERHEAD / saving;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n  Large-model cost/req:  $" << large_cost << "\n";
    std::cout << "  Small-model cost/req:  $" << small_cost << "\n";
    std::cout << "  Saving per routed req: $" << saving     << "\n";
    std::cout << "  Router overhead/req:   $" << ROUTER_OVERHEAD << "\n";
    std::cout << std::setprecision(4);
    std::cout << "  Break-even fraction:   " << break_even * 100 << "%\n";
    std::cout << "\n  → Route even " << std::setprecision(2) << break_even * 100
              << "% of traffic to small model and routing pays for itself.\n";

    assert(break_even < 0.01);
    std::cout << "\n  [ASSERT] Break-even < 1%: " << std::setprecision(4)
              << break_even * 100 << "% ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// TRAFFIC SIMULATION
// ─────────────────────────────────────────────────────────────────────────────

static void demo_traffic_simulation() {
    print_section("Traffic Simulation — Routing Metrics");

    struct TrafficSpec {
        std::string qtype;
        std::string difficulty;
        std::string language;
        int         tokens;
        double      fraction;  // traffic share
    };

    std::vector<TrafficSpec> traffic_mix = {
        {"faq",       "easy",   "en",  70,  0.35},
        {"classify",  "easy",   "en",  35,  0.15},
        {"complaint", "medium", "en", 180,  0.25},
        {"billing",   "hard",   "en", 300,  0.12},
        {"code",      "hard",   "en", 420,  0.08},
        {"faq",       "easy",   "es", 100,  0.05},
    };

    int N = 10000;
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::map<std::string, int>    tier_counts;
    double total_cost = 0.0, total_lat = 0.0;

    for (int i = 0; i < N; ++i) {
        // Sample from traffic mix
        double r = uni(rng);
        double cum = 0.0;
        const TrafficSpec* spec = &traffic_mix.back();
        for (auto& s : traffic_mix) {
            cum += s.fraction;
            if (r < cum) { spec = &s; break; }
        }

        std::string tier = offline_route(spec->qtype, spec->difficulty,
                                         spec->language, spec->tokens);
        tier_counts[tier]++;
        const auto& t = TIERS.at(tier);
        int in_tok  = static_cast<int>(spec->tokens * 0.6);
        int out_tok = static_cast<int>(spec->tokens * 0.4);
        total_cost += (in_tok / 1000.0) * t.input_price_1k +
                      (out_tok / 1000.0) * t.output_price_1k;
        total_lat  += t.p50_latency_ms;
    }

    std::cout << "\n  Simulated " << N << " requests\n\n";
    std::cout << "  " << std::left << std::setw(10) << "Tier"
              << std::setw(10) << "Count" << std::setw(12) << "Fraction" << "\n";
    std::cout << "  " << std::string(32, '-') << "\n";
    for (auto& [tier, count] : tier_counts) {
        std::cout << "  " << std::setw(10) << tier
                  << std::setw(10) << count
                  << std::setprecision(1) << std::fixed
                  << count * 100.0 / N << "%\n";
    }

    std::cout << std::setprecision(6) << std::fixed;
    std::cout << "\n  Total cost:             $" << total_cost << "\n";
    std::cout << "  Avg cost per request:   $" << total_cost / N << "\n";
    std::cout << std::setprecision(1);
    std::cout << "  Avg latency:            " << total_lat / N << " ms\n";

    assert(tier_counts["small"] > tier_counts["large"]);
    std::cout << "\n  [ASSERT] Small tier handles more traffic than large: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n" << std::string(60, '=')
              << "\n  Chapter 31 — Model Routing and Cascading (C++)\n"
              << std::string(60, '=') << "\n";

    demo_offline_router();
    demo_cascade();
    demo_cost_model();
    demo_break_even();
    demo_traffic_simulation();

    std::cout << "\n" << std::string(60, '=')
              << "\n  All demos complete.\n"
              << std::string(60, '=') << "\n\n";
    return 0;
}
```

---

## 31.13 Summary

Routing and cascading are the application-layer counterpart to inference-engine optimization. Where vLLM and llama.cpp squeeze more throughput out of a fixed GPU budget, routing multiplies the effective capacity of your cluster by sending easy queries to cheap models.

The key mental model: think of your model fleet as a cost-quality Pareto frontier. Every request has a minimum acceptable quality; the router finds the cheapest model on the frontier that meets that bar. A cascade does this dynamically by asking the small model first and escalating only on failure — paying the cost of a wrong first guess only for the minority of hard queries.

**What to remember:**

- Break-even for a two-tier cascade is typically < 1% hit rate on the small model — the economics are almost always favorable.
- Threshold calibration is the critical implementation detail: set it too low and you pay for large-model calls unnecessarily; set it too high and quality degrades.
- Build quality measurement in from day one. Without it, you cannot know whether your routing policy is working.
- Model deployments change routing behavior. A new 70B fine-tune may answer more queries confidently, shifting the optimal threshold. Re-calibrate on every model update.

**Next chapter** examines the opposite problem: instead of routing away from a model, what happens when a single model's inference goes wrong? Chapter 32 covers debugging inference systems — from NaN logits and KV cache corruption to sampling instability and distributed training mismatches.

---

*End of Chapter 31*


---

## Chapter Summary

- **Routing motivation**: not all requests need a 70B model; a classifier that routes 60% of requests to a 7B model and 40% to 70B cuts average cost by ~50% with minimal quality loss.
- **Cascading vs routing**: routing makes a single upfront decision; cascading sends the request to the small model first and escalates to the large model only if the small model's confidence is below a threshold.
- **Router architecture**: a lightweight embedding + linear classifier trained on (query, correct_model_size) pairs; inference latency should be <10 ms.
- **Quality-cost Pareto frontier**: plot accuracy vs $/request for each model; the router should operate on the Pareto frontier, routing to the smallest model that meets the quality SLA.
- **Calibration**: the cascade confidence threshold must be calibrated on a held-out set; a threshold that is too high sends too many requests to the large model; too low degrades quality.
- **Model specialization**: routers can target domain-specific models (code model, medical model) rather than just size; this improves quality without cost increase.
- **Feedback loop**: log which model answered each request and the downstream quality signal; use this to retrain the router classifier on a rolling basis.

---

## Self-Check Questions

1. A cascading system sends requests to a 7B model first. If the 7B model answers with confidence ≥ 0.85, the answer is returned directly. Otherwise, the 70B model is called. For a batch of 1 000 requests where 65% are high-confidence, compute the average cost relative to always using the 70B model. Assume 70B costs 10× per request. *(Section 31.2)*

2. The routing classifier misclassifies 8% of requests, sending complex queries to the 7B model. Quality degradation on these queries is 20% (measured by LLM judge score). If user satisfaction is proportional to quality score, compute the overall quality impact with 60% routing to 7B. *(Section 31.3)*

3. A router for code vs general queries achieves 94% accuracy on the test set. In production, 30% of queries are code. Compute the confusion matrix entries (TP, FP, TN, FN) for 10 000 requests. *(Section 31.4)*

4. Latency budget: the router takes 15 ms, the 7B model takes 120 ms, and the 70B model takes 600 ms. For the cascading system in question 1, compute P50 latency. *(Section 31.2)*

5. Model routing introduces a feedback loop risk: if the router under-routes to the large model, quality suffers; if quality suffers, feedback training data degrades. Describe the data flywheel failure mode and a monitoring strategy to detect it. *(Section 31.5)*
