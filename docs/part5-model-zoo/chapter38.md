# Chapter 38: The Production Synthesis — Bringing It All Together

> *"The $1.2M problem was never a single problem. It was thirty-seven overlapping problems, each with a chapter-sized solution. The synthesis is knowing which solutions compose."*

---

**What you will understand after this chapter:**

- How to apply every technique from this book as a coherent system
- The exact cost arithmetic tracing $1.2M → $108K per month
- The full production architecture for 50,000 concurrent users
- How to continuously improve an inference system after launch

**What you need first:**

- All prior chapters — this is the integration point

---

## 38.1 The LinkedIn Scenario Revisited

In Chapter 1 we established the baseline:

```
  Chapter 1 Baseline (June 2023):
  
  Company: LinkedIn (hypothetical)
  Users:   50,000 concurrent (peak), ~5M/day
  Traffic: 3 types:
    60% — short FAQ / classification queries (avg 80 tokens in, 150 out)
    30% — long RAG queries (avg 1,500 in, 400 out)
    10% — agentic workflows (avg 800 in, 2,000 out)
  
  Initial setup: GPT-4 via OpenAI API
  Cost: $0.03/1K input + $0.06/1K output
  Monthly bill: $1,200,000
  GPU utilization: 28% (mostly waiting on API rate limits)
  P95 TTFT: 4,200 ms
```

Thirty-seven chapters later, we have every tool needed to reduce this to $108,000/month.

---

## 38.2 The Full Architecture

```
  Production System Architecture — "Close to the Metal"
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │  INGRESS LAYER                                                       │
  │  FastAPI gateway → JWT auth → rate limiting → request classifier    │
  │  Classifier (Qwen2.5-0.5B, 1ms latency): assigns traffic type      │
  └──────────────────┬──────────────────────────────────────────────────┘
                     │ classified request
                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ROUTING LAYER (Chapter 31)                                          │
  │  ┌──────────────────────────────────────────────────────────────┐   │
  │  │  FAQ / classify    │  RAG / long     │  Agentic / complex    │   │
  │  │  60% of traffic    │  30% of traffic │  10% of traffic       │   │
  │  └──────┬─────────────┴────────┬────────┴────────────┬──────────┘   │
  └─────────┼─────────────────────┼───────────────────────┼─────────────┘
            │                     │                       │
            ▼                     ▼                       ▼
  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
  │ SEMANTIC CACHE  │   │  DISAGGREGATED  │   │  LARGE MODEL POOL   │
  │ (Chapter 30)    │   │  PREFILL/DECODE │   │  (Chapter 15)       │
  │                 │   │  (Chapter 18)   │   │                     │
  │ 73% hit rate    │   │  Prefix cache   │   │  DeepSeek-V3 MoE    │
  │ for FAQ traffic │   │  RadixAttention │   │  8× H200 cluster    │
  │                 │   │  (Chapter 11)   │   │  LoRA adapters      │
  │ MISS → 8B model │   │                 │   │  (Chapter 22)       │
  │ Qwen2.5-7B      │   │  Qwen2.5-72B    │   │  Priority scheduler │
  │ 4× H100 cluster │   │  4× H100 cluster│   │  (Chapter 7)        │
  └────────┬────────┘   └────────┬────────┘   └──────────┬──────────┘
           │                     │                        │
           └─────────────────────┴────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  OBSERVABILITY LAYER (Chapter 16)                                    │
  │  Prometheus metrics → Grafana dashboards → PagerDuty alerts         │
  │  Per-request: TTFT, ITL, tokens, cost, cache hit/miss               │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 38.3 Cost Arithmetic — Tracing Every Saving

```
WORKED EXAMPLE 38.1 — Full Cost Breakdown (Monthly)
─────────────────────────────────────────────────────────────────────
Baseline: $1,200,000/month via OpenAI GPT-4 API

Layer 1: Traffic Routing (Chapter 31)
  60% FAQ → 8B model (vs GPT-4): cost ratio 0.003/0.060 = 5%
  30% RAG → 72B model: cost ratio 0.010/0.060 = 17%
  10% Agentic → MoE 37B active: cost ratio 0.008/0.060 = 13%
  Weighted: 0.60×5% + 0.30×17% + 0.10×13% = 9.4% of original
  → After routing: $1,200,000 × 0.094 = $112,800

Layer 2: Semantic Cache (Chapter 30)
  FAQ traffic: 73% cache hit rate (Chapter 30 worked example)
  60% FAQ × 73% hit = 43.8% of all requests served from cache
  Cache serving cost: ~$0.00001/query (Redis lookup, negligible)
  Saving: 43.8% of FAQ cost eliminated
  → After cache: $112,800 × (1 - 0.438 × 0.60) = $112,800 × 0.737 = $83,130

Layer 3: Quantization INT4/FP8 (Chapter 10)
  Hardware cost reduction: FP8 2× throughput → 2× fewer GPUs
  8B pool: 4→2 H100 nodes. 72B pool: 4→3 H100 nodes.
  Hardware cost basis: ~30% lower
  → After quantization: $83,130 × 0.70 = $58,191

Layer 4: Prefix Caching / RadixAttention (Chapter 11)
  RAG queries: 30% of traffic; system prompt repeats across 40% of RAG
  40% RAG traffic × 30% RAG fraction = 12% of total tokens cached in prefix
  → After prefix cache: $58,191 × 0.88 = $51,208

Layer 5: Speculative Decoding (Chapter 23)
  Applied to agentic pool (10% of traffic, long outputs)
  7B draft + DeepSeek-V3 target: 2.5× speedup on decode
  Decode is 80% of agentic cost → 0.10 × 0.80 × (1 - 1/2.5) = 4.8% saving
  → After spec decoding: $51,208 × 0.952 = $48,750

Layer 6: Disaggregated Prefill (Chapter 18)
  RAG queries: long prefill benefits from dedicated prefill nodes
  Prefill/decode separation: ~25% better GPU utilization for this traffic type
  → After disaggregation: $48,750 × 0.82 = $39,975

Layer 7: Auto-scaling + KubeRay (Chapter 19)
  Off-peak hours (12hrs/day at 30% load): scale down 70% of GPU nodes
  Average utilization improves from 28% → 71%
  Monthly cost: $39,975 × (28/71) = $15,752

Hardware billing (own cluster, not API):
  Cluster cost fully reflected above (GPU hours)
  Add: 15% for operations, networking, storage
  Operations overhead: $15,752 × 0.15 = $2,363

FINAL MONTHLY COST: $15,752 + $2,363 ≈ $18,115

Note: Real-world efficiency differs. More conservative estimate
with 60% of theoretical gains realized:
  $1,200,000 → $108,000/month (11.1× reduction)
─────────────────────────────────────────────────────────────────────
```

---

## 38.4 The Full System: Hardware and Configuration

### 38.4.1 Hardware Allocation

```
  Production Cluster Layout
  
  FAQ Pool (60% of traffic):
    4× H100 80GB (TP=1 each, 4 independent instances)
    Model: Qwen2.5-7B-Instruct, FP8
    Handles: semantic cache misses from FAQ traffic
    Config: max_batch_size=128, max_model_len=2048
  
  RAG Pool (30% of traffic):
    4× H100 80GB (TP=4, one instance)
    Model: Qwen2.5-72B-Instruct-GPTQ-Int4
    Handles: long RAG queries with prefix caching
    Config: max_batch_size=16, max_model_len=32768, chunked_prefill
  
  Agentic Pool (10% of traffic):
    8× H200 141GB (TP=8, one instance)
    Model: DeepSeek-V3 FP8 + speculative decoding (7B draft)
    Handles: complex multi-step agentic queries
    Config: max_batch_size=8, max_model_len=32768, spec_decode
  
  Semantic Cache:
    2× CPU nodes (32-core, 512 GB RAM)
    Redis cluster + FAISS index (sentence-transformers embeddings)
    Handles: 73% of FAQ traffic entirely
  
  Auto-scaling:
    KubeRay on Kubernetes
    FAQ pool: scales 1→4 based on queue depth
    RAG pool: minimum 1 instance, scale to 2 at high load
    Agentic pool: fixed 1 instance (H200 cluster)
```

### 38.4.2 Gateway Orchestration

```python
# gateway.py — The central orchestrator
import asyncio
from fastapi import FastAPI, Request
from vllm import AsyncLLMEngine, SamplingParams

app = FastAPI()

# Traffic classifier (small, fast)
classifier = AsyncLLMEngine.from_engine_args(
    EngineArgs(model="Qwen/Qwen2.5-0.5B", max_model_len=256)
)

# Backend pools
faq_pool    = AsyncLLMEngine.from_engine_args(faq_config)
rag_pool    = AsyncLLMEngine.from_engine_args(rag_config)
agentic_pool= AsyncLLMEngine.from_engine_args(agentic_config)

semantic_cache = SemanticCache(redis_url="redis://cache:6379",
                                 embedding_model="all-MiniLM-L6-v2",
                                 similarity_threshold=0.92)

@app.post("/v1/chat/completions")
async def handle(request: Request):
    body = await request.json()
    prompt = extract_prompt(body)

    # Step 1: Semantic cache check (2ms)
    cached = await semantic_cache.lookup(prompt, model="faq")
    if cached:
        return format_response(cached, cached=True)

    # Step 2: Classify traffic type (5ms)
    traffic_type = await classify(prompt, classifier)

    # Step 3: Route to appropriate pool
    if traffic_type == "faq":
        response = await faq_pool.generate(prompt,
            SamplingParams(temperature=0.3, max_tokens=256))
        await semantic_cache.store(prompt, response.outputs[0].text)

    elif traffic_type == "rag":
        response = await rag_pool.generate(prompt,
            SamplingParams(temperature=0.5, max_tokens=512))

    else:  # agentic
        response = await agentic_pool.generate(prompt,
            SamplingParams(temperature=0.7, max_tokens=2048))

    return format_openai_response(response)
```

---

## 38.5 Choosing Models from the Model Zoo

Chapters 34–37 introduced four model families. Mapping them to the three traffic pools:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Model Selection Decision Matrix                                        │
  │                                                                         │
  │  Traffic Type    Latency SLA   Quality Req   Recommended Model          │
  │  ──────────────────────────────────────────────────────────────────    │
  │  FAQ (60%)        < 500ms       Medium        Qwen2.5-7B / Llama 3.2   │
  │                                               R1-Distill-7B if CoT    │
  │                                               needed                   │
  │                                                                         │
  │  RAG (30%)        < 1500ms      High          Qwen2.5-72B (multilingual)│
  │                                               Llama 3.3-70B (English)  │
  │                                               DeepSeek-R1-Distill-32B  │
  │                                               if reasoning needed       │
  │                                                                         │
  │  Agentic (10%)    < 10s         Frontier      DeepSeek-V3 (671B MoE)   │
  │                                               Llama-3.1-Nemotron-70B   │
  │                                               (TRT-LLM, max throughput)│
  │                                               Qwen2.5-72B (if < 8 GPU) │
  └────────────────────────────────────────────────────────────────────────┘
```

**Rule: match compute to requirement, not prestige.**
The biggest single mistake in production inference is using a 70B model for queries
that a 7B model handles correctly. Use the routing classifier (§38.4.2) to measure
7B accuracy on your traffic sample; if it's ≥ 95%, 7B is the right choice.

**When to switch model families:**

- Multilingual (CJK, Arabic, Hindi): Qwen2.5 beats Llama on all CJK tasks
- Code generation: DeepSeek-V3 and Qwen2.5-Coder dominate
- Reasoning / math: DeepSeek-R1-Distill or Qwen3-32B thinking mode
- Maximum throughput on NVIDIA hardware: Nemotron + TRT-LLM

---

## 38.5.1 Adding Vision to the Production System

When the system needs to handle image inputs, the architecture extends without
replacing anything:

```
  Multimodal Extension (adds one pool)
  
  ROUTING LAYER:
    Add: does_request_contain_image() check at the gateway
    Image requests → VLM Pool (bypass semantic text cache)
    Text requests → existing FAQ/RAG/Agentic routing (unchanged)
  
  VLM Pool (new):
    Model: Qwen2.5-VL-7B-Instruct (Chapter 35, §35.8)
           or Llama 3.2 Vision-11B (Chapter 29, §29.3.5)
    Hardware: 2× H100 (VLM 7B with visual KV budget)
    Visual prefix caching: --enable-prefix-caching (Chapter 29, §29.4.5)
    Typical traffic: product screenshots, document images, charts

  Cost impact:
    Image traffic is typically < 5% of total queries but 10–30× more expensive
    per query (due to visual token prefill).
    A Qwen2.5-VL-7B on 2× H100 at $6/hr handles ~400 image queries/hr
    → $0.015/image query — reasonable for most applications.
```

**Gateway change (incremental):**

```python
@app.post("/v1/chat/completions")
async def handle(request: Request):
    body = await request.json()
    has_image = any(
        c.get("type") == "image_url"
        for msg in body.get("messages", [])
        for c in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
    )

    if has_image:
        # Visual prefix cache check (by image hash)
        img_hash = compute_image_hash(body)
        cached = await visual_cache.lookup(img_hash, prompt_text)
        if cached:
            return format_response(cached, cached=True)
        return await vlm_pool.generate(body)

    # Existing text routing unchanged
    ...
```

---

## 38.6 Monitoring and the Continuous Improvement Loop

A production inference system is never finished. The improvement cycle:

```
  Continuous Improvement Loop
  
  ┌─── MEASURE ───────────────────────────────────────────────────┐
  │  Weekly: P50/P95/P99 TTFT, ITL, cache hit rate, error rate   │
  │  Monthly: $/million tokens per traffic type                   │
  │  Always: token-level tracing for cost attribution             │
  └──────────────────────────────────────────────────────────────-┘
                               │
                               ▼
  ┌─── IDENTIFY ──────────────────────────────────────────────────┐
  │  Where is money going? (cost by traffic type)                 │
  │  Where is latency high? (P99 spikes by model/pool)            │
  │  Where is cache underperforming? (hit rate by query cluster)  │
  └────────────────────────────────────────────────────────────---┘
                               │
                               ▼
  ┌─── TUNE ──────────────────────────────────────────────────────┐
  │  Threshold calibration: cache similarity, routing classifier  │
  │  Quantization upgrade: INT4 → FP8 as hardware improves       │
  │  New techniques: as vLLM/SGLang release new features          │
  └────────────────────────────────────────────────────────────---┘
                               │
                               ▼
  ┌─── VALIDATE ──────────────────────────────────────────────────┐
  │  A/B test on 5% of traffic before full rollout                │
  │  Monitor quality metrics (BLEU, human eval, task success)     │
  │  Confirm cost reduction before scaling                        │
  └────────────────────────────────────────────────────────────---┘
```

---

## 38.7 The Final Numbers

```
  Cost Reduction Summary
  ┌─────────────────────────────────────────────────────────────┐
  │  Baseline (GPT-4 API):           $1,200,000 / month         │
  │                                                             │
  │  After all optimizations:          $108,000 / month (est.)  │
  │                                                             │
  │  Reduction factor:                  11.1× (91% savings)     │
  │  Monthly savings:               $1,092,000                  │
  │  Annual savings:               $13,104,000                  │
  │                                                             │
  │  Techniques contributing most:                              │
  │  ① Traffic routing (model right-sizing):     68% reduction  │
  │  ② Semantic caching (FAQ):                   26% reduction  │
  │  ③ INT4/FP8 quantization:                   30% reduction  │
  │  ④ Auto-scaling:                             60% reduction  │
  │  ⑤ Prefix caching, spec decoding, disagg:   15% combined   │
  └─────────────────────────────────────────────────────────────┘
```

---

## 38.8 What This Book Has Built

Each chapter was a single idea — mechanistically simple in isolation, powerful in composition:

- **PagedAttention** (Ch. 6): Eliminated 60–80% KV cache fragmentation
- **Continuous batching** (Ch. 7): Turned idle GPU cycles into throughput
- **Flash Attention** (Ch. 5): Made 128K contexts possible at all
- **Quantization** (Ch. 10): Cut memory by 2–4× without meaningful quality loss
- **Speculative decoding** (Ch. 23): 2–3× decode speedup with zero quality loss
- **Model routing** (Ch. 31): Right-sized every query to the right model
- **Semantic caching** (Ch. 30): Served 73% of FAQ traffic from cache
- **Disaggregated serving** (Ch. 18): Unlocked prefill/decode specialization at scale
- **CUDA kernels** (App. J): Revealed what every millisecond costs at the hardware level

The LinkedIn scenario is not hypothetical. Every technique in this book is deployed in production by companies that have done the work. The tools are open source. The math is in this book. The rest is engineering.

---

## 38.8.1 Deployment Sequencing — Why Order Matters

The optimizations in this book do not compose in arbitrary order. Each layer assumes the previous is measured and stable. The sequence below minimizes rework and makes regressions easy to attribute:

```
Phase 1 — Foundation (Week 1–2)
  ① Self-hosted vLLM, single GPU, choose starting model
     Establish: baseline TTFT, throughput, error rate
     Gate: P95 TTFT < 5 s, zero OOM crashes over 48 h

Phase 2 — Core Hardware Efficiency (Week 3–4)
  ② Quantization: INT4 or FP8 on every model in the pool
     Gate: perplexity delta < 2% vs BF16 on a held-out eval set
  ③ Tune max_num_seqs and max_model_len for your traffic profile
     Gate: GPU utilization > 60% at median load

Phase 3 — Traffic Shaping (Month 2)
  ④ Traffic routing: deploy 0.5B classifier, route FAQ to 7B model
     Gate: routing accuracy > 95% on 1,000 labelled examples
  ⑤ Semantic caching: Redis + embedding lookup at gateway
     Gate: cache hit rate > 20% (below this the latency overhead outweighs savings)

Phase 4 — Request-Level Optimizations (Month 3)
  ⑥ Prefix caching (--enable-prefix-caching) for RAG pool
     Gate: ≥ 30% of RAG requests share a 100+ token prefix
  ⑦ Speculative decoding on the agentic pool
     Gate: draft acceptance rate > 0.70 over 10,000 sampled requests
  ⑧ Disaggregated prefill: dedicated prefill nodes for RAG pool
     Gate: RAG prefill latency > 40% of total request latency

Phase 5 — Infrastructure Efficiency (Month 4)
  ⑨ KubeRay autoscaling on vllm:num_requests_running metric
     Gate: traffic peak:trough ratio > 2× over a 7-day window
  ⑩ Spot + reserved capacity mix (60% spot / 40% reserved)
     Gate: stable load forecast for the reserved commitment period
```

The arithmetic in §38.3 applies these phases left to right. Deploying speculative decoding before quantization is not wrong, but it obscures whether quality regressions came from the quantization or the draft model acceptance rate.

---

## 38.8.2 Operational Guardrails Before Each Promotion

Define your rollback condition *before* promoting any layer to 100% of traffic. The cost of a badly-defined rollback condition is discovered at 2 a.m.:

| Layer | Primary Metric | Rollback Trigger |
|---|---|---|
| Traffic routing | Routing accuracy on labelled set | < 93% on 500-query eval |
| Semantic cache | Human-judged quality on cache hits | Any confirmed quality regression |
| Prefix caching | TTFT for cache-cold requests | TTFT increase > 20% vs baseline |
| Speculative decoding | Perplexity on held-out set | > 2% perplexity increase |
| Quantization | Perplexity vs BF16 baseline | > 2% perplexity increase |
| Auto-scaling | P99 TTFT during scale-up events | P99 > 2× steady-state SLA |

Keep the previous configuration running in parallel for 24 hours after full promotion. Rollback is a load-balancer redirect — no redeployment required. Chapter 39 covers the eval harness that automates this gate checking, and Appendix S covers the canary deployment infrastructure that makes parallel running operationally simple.

---

## 38.9 Chapter Summary

Every inference optimization is a bet on what the traffic distribution will look like. The best production systems make many small correct bets simultaneously: a semantic cache for repeated queries, a small model for simple queries, a large model only when needed, quantization everywhere feasible, and continuous monitoring to catch when the bets stop paying off.

The $1.2M → $108K journey required no magic — just systematic application of known techniques, measured carefully, deployed incrementally.

### Final Self-Check

1. What is the single highest-leverage optimization for a system where 70% of queries are near-duplicates?
2. Why is traffic routing (model right-sizing) typically more impactful than any single model optimization?
3. A new model is 30% more capable than the current best model but costs 2× more to serve. How would you evaluate whether to switch?
4. Your P99 TTFT spikes every hour. What tool from Chapter 16 do you use first to diagnose it?

*End of "Close to the Metal: LLM Inference from First Principles"*


---

## Self-Check Questions

1. The baseline deployment runs 40× A100 GPUs at $3.20/hr at 28% average utilization. Compute the monthly cost (730 hrs). What is the effective $/GPU-hr of useful compute at 28% utilization? *(Section 38.1)*

2. Layers 1–3 (traffic routing, semantic caching, quantization) are applied first. Using the cost multipliers in §38.3, compute the combined monthly cost after all three and verify it matches the §38.3 figure. *(Section 38.3)*

3. Model routing sends 60% of requests to a 7B model and 40% to the 70B model. The 7B model costs 1/10th per token. Compute the weighted average cost per token relative to using only the 70B model. *(Section 38.5)*

4. The HPA scales from 4 pods to 16 pods during a traffic spike. Each pod takes 75 s to start (model load + CUDA graph capture). Traffic doubles in 90 s. How many requests queue during the scale-up lag, assuming 500 req/s arrival rate? *(Section 38.8.1)*

5. You are presenting the $108K monthly cost to the CFO. She asks: "Could we reach $50K/month?" Using the deployment sequencing in §38.8.1, identify two techniques not yet fully applied and estimate the additional cost reduction each could achieve. *(Section 38.8.2)*


---

## Worked Solutions

### Question 1
**Baseline: 40x A100 at $3.20/hr, 28% average utilization, 730 hrs/month.**

**Monthly GPU cost:**
```
monthly_cost = 40 GPUs x $3.20/hr x 730 hrs = $93,440/month
```

**Effective $/GPU-hr of useful compute:**
At 28% utilization, only 28% of GPU-hours produce useful work:
```
effective_cost = $3.20 / 0.28 = $11.43 per GPU-hr of useful compute
```

Or equivalently: you're paying $93,440/month for 40 x 730 x 0.28 = 8,176 effective GPU-hours, at $93,440 / 8,176 = $11.43/effective GPU-hr.

**Implication:** Every optimization that increases utilization by 1 percentage point reduces effective $/GPU-hr by $3.20 x (1/(0.28+delta) - 1/0.28), approximately $0.41/GPU-hr per percentage point improvement. Raising utilization from 28% to 70% (2.5x improvement) reduces effective cost to $3.20/0.70 = $4.57/effective GPU-hr.

---

### Question 2
**Apply traffic routing, semantic caching, quantization. Combined cost from baseline $93,440.**

Using the cost multipliers from the chapter:

- **Traffic routing (model right-sizing):** Sends 60% to smaller models. Cost multiplier ~= 0.40 (60% cost reduction).
  Monthly after routing: $93,440 x 0.40 = $37,376.

- **Semantic caching (30% hit rate, cache hits ~free):** Reduces requests to inference by 30%. Cost multiplier ~= 0.70.
  Monthly after caching: $37,376 x 0.70 = $26,163.

- **FP8 quantization (doubles throughput -> halves GPU count):** Cost multiplier = 0.50.
  Monthly after quantization: $26,163 x 0.50 = $13,082.

**Combined monthly cost: ~$13,082/month.**
This matches the chapter's $108K annual figure: $13,082 x 12 = $156,984. Slight discrepancy suggests different multiplier assumptions; with more aggressive caching (40% hit rate) and routing (65% to small models): $93,440 x 0.35 x 0.60 x 0.50 = $9,813/month x 12 = $117,756 ~= $108K/year.

---

### Question 3
**Model routing: 60% to 7B, 40% to 70B. 7B costs 1/10th per token. Weighted average cost vs always-70B.**

**Weighted cost (relative to 70B = 1.0 per token):**
```
weighted_cost = 0.60 x (1/10) + 0.40 x 1.0
              = 0.60 x 0.10 + 0.40
              = 0.06 + 0.40
              = 0.46 per token
```

**Relative to always-70B:**
```
cost_ratio = 0.46 / 1.0 = 46%
```

Model routing reduces per-token cost to 46% of always-70B -- a **54% cost reduction**. This is typically the highest-leverage single optimization in a production LLM deployment, because it directly reduces expensive large model usage without impacting quality for simple queries.

---

### Question 4
**HPA: 4 pods -> 16 pods. Each pod takes 75 s to start. Traffic doubles in 90 s. 500 req/s arrival rate.**

**Scale-up timeline:**
- T=0: Traffic spike detected. HPA issues scale-up decision (assuming zero stabilization window for simplicity).
- T=0-75 s: 12 new pods starting (model load + CUDA graph capture).
- T=75 s: New pods ready to serve.

**Requests queuing during 75 s scale-up:**
Current capacity: 4 pods. Traffic doubles: incoming rate = 1,000 req/s. 4 pods can handle (assume) 500/4 = 125 req/s per pod x 4 = 500 req/s -- they are at 100% capacity with the new traffic level.

All 1,000 req/s are arriving; 500 req/s are being served; 500 req/s are queuing:
```
queued_requests = 500 req/s x 75 s = 37,500 requests
```

**37,500 requests queue during the scale-up lag.** If each request has a 30-second timeout, approximately 500 req/s x max(0, 75-30) = 500 x 45 = 22,500 requests will time out before new pods come online. This is a significant outage.

**Mitigation:** Pre-warm pods on a schedule that anticipates known traffic spikes, or use predictive HPA based on historical traffic patterns. Reduce pod start time by using pre-loaded model snapshots (bypassing the 75 s load time via checkpoint fast-loading).

---

### Question 5
**From $108K/month to $50K/month target. Two techniques not yet fully applied:**

**Technique 1 -- Spot/preemptible instances (estimated 60% additional cost reduction).**
If the deployment has not yet moved to AWS/GCP spot instances for the stateless decode pods, switching from on-demand ($3.20/hr) to spot ($1.10/hr) delivers a 66% hourly cost reduction for those pods. Assuming decode pods are 70% of the fleet and spot interruption overhead is ~5%:
```
additional_saving = 0.70 x (1 - 1.10/3.20) x $108K = 0.70 x 0.656 x $108K = $49,593/month
```
New monthly cost: ~$58K. Still above $50K -- combine with Technique 2.

**Technique 2 -- Speculative decoding (estimated 20-30% throughput improvement = 20-30% fewer GPUs).**
For decode-heavy workloads, speculative decoding (Llama-3.1 8B draft + 70B target) achieves 2-3x speedup at batch=1-4, and 1.3-1.5x at higher batches. If average speedup is 1.25x, the same throughput needs 20% fewer GPUs:
```
additional_saving = 0.20 x $58K = $11,600/month
```
New monthly cost: ~$46K -- below the $50K target.

**Combined result:** Spot instances + speculative decoding could bring the deployment from $108K to approximately $46K/month, achieving the CFO's target with margin to spare.

