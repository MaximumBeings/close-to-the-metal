# Chapter 20: Cost Engineering — $/Million Tokens

> "Throughput is vanity. Latency is sanity. Cost per token is reality."
>
> — Production ML engineering proverb

---

## 20.1 The Universal Comparator

`[FOUNDATIONAL]`

Every decision in this book — which engine, which GPU, which quantization, whether to disaggregate — ultimately resolves to a single number: **dollars per million output tokens**.

This metric normalizes across hardware generations, cloud providers, deployment modes, and model families. A system that delivers 1,635 tokens per second on an RTX 4090 means nothing until you divide the hardware cost by that throughput. A system that processes 19,870 prefill tokens per second is useless context unless you know what it costs per useful answer.

The formula is deceptively simple:

```
  $/1M output tokens = (GPU cost per hour) / (output tokens per hour) × 1,000,000

  Output tokens per hour = decode_throughput_tok_s × 3600
                         = (decode_tps × batch_utilization_fraction) × 3600

  Example — RTX 4090, vLLM, batch=32:
    decode_tps = 1,635 tok/s
    GPU cost   = $0.50/hr (Vast.ai spot)
    $/1M = $0.50 / (1,635 × 3,600) × 1,000,000 = $0.085/1M tokens
```

This is 12× cheaper than OpenAI's cheapest API tier at publication time, but only valid at sustained batch=32 utilization. The gap narrows rapidly as utilization drops, and disappears entirely at batch=1 idle overnight.

---

## 20.2 GPU Cost Model

`[FOUNDATIONAL]`

GPU compute is sold in three modes, each with a different cost/risk trade-off:

```
  ┌───────────────┬──────────────────┬───────────────────────────────────┐
  │  Mode         │  Price           │  Risk / constraint                │
  ├───────────────┼──────────────────┼───────────────────────────────────┤
  │  On-demand    │  Full price      │  None — terminate any time        │
  │  Spot/Preempt │  30–75% cheaper  │  Can be reclaimed with 2 min warn │
  │  Reserved     │  30–45% cheaper  │  1- or 3-year commitment          │
  └───────────────┴──────────────────┴───────────────────────────────────┘
```

### 2025 Reference Prices (approximate)

```
  ┌────────────────────────────────┬───────────┬───────────┬────────────┐
  │  GPU                           │  On-demand│  Spot     │  Reserved  │
  ├────────────────────────────────┼───────────┼───────────┼────────────┤
  │  H100 SXM 80GB  (AWS p5)       │  $32.77/h │  ~$9.80/h │  ~$20.30/h │
  │  H100 NVL 94GB  (AWS p5e)      │  $24.00/h │  ~$8.40/h │  ~$15.60/h │
  │  A100 80GB      (AWS p4de)     │  $12.00/h │  ~$4.80/h │  ~$7.80/h  │
  │  A100 80GB      (Lambda Labs)  │  $2.49/h  │  n/a      │  n/a       │
  │  RTX 4090       (Vast.ai)      │  $0.50/h  │  $0.30/h  │  n/a       │
  │  RTX 4090       (own hardware) │  $0.08/h* │  n/a      │  n/a       │
  │  Apple M2 Ultra (own hardware) │  $0.06/h* │  n/a      │  n/a       │
  └────────────────────────────────┴───────────┴───────────┴────────────┘
  * Amortised over 3 years + electricity at $0.10/kWh
```

`[COMMON TRAP]` — **Comparing cloud on-demand to own-hardware amortised**: the RTX 4090 at
$0.08/hr looks 6× cheaper than Vast.ai spot at $0.50/hr, but the $0.08 assumes 24/7 utilization
over 3 years with no downtime, no maintenance, no cooling overhead, and no opportunity cost of
the $1,599 capital outlay. At 50% utilization the true cost doubles to $0.16/hr, cutting the
advantage to 3×. At 25% utilization it is $0.32/hr — barely cheaper than spot, with all the
maintenance burden.

---

## 20.3 Tokens/sec/GPU: The Efficiency Driver

`[DEEP DIVE]`

The relationship between throughput and cost per token is linear: double the throughput, halve
the cost. This makes throughput optimization the highest-leverage cost reduction activity.

```
  Cost levers ranked by typical impact:

  1. Batch size                  2–10× throughput gain at scale
     (idle → saturated GPU)

  2. quantization                1.5–3× throughput gain
     (BF16 → Q4_K_M weights)

  3. Hardware generation         1.5–4× throughput gain
     (A100 → H100 SXM)

  4. Prefix caching              up to 10× cost reduction on repeated prefixes
     (cache hit avoids prefill entirely)

  5. Engine choice               1.1–2× gain (context-dependent)
     (llama.cpp → vLLM at batch > 8)

  6. Disaggregation              1.5–3× throughput gain on mixed traffic
     (Chapter 18)
```

### Worked Cost Derivations

**Scenario A: Interactive chat, low concurrency**

```
  Hardware:  RTX 4090 self-hosted
  Engine:    llama.cpp
  Model:     Llama-3-8B Q4_K_M
  Batch:     1 (single user at a time)
  Decode:    64 tok/s

  $/1M = $0.08/hr / (64 × 3600) × 1e6 = $0.347/1M tokens
```

**Scenario B: API serving, high concurrency**

```
  Hardware:  H100 SXM (Lambda Labs, $2.49/hr)
  Engine:    vLLM
  Model:     Llama-3-8B BF16
  Batch:     32 (sustained)
  Decode:    1,635 tok/s

  $/1M = $2.49/hr / (1,635 × 3,600) × 1e6 = $0.423/1M tokens
```

**Scenario C: Same as B but GPU better utilized (prefix cache 60% hit)**

```
  Effective output tokens/hr = 1,635 × 3,600 × (1 + 0.60 prefill savings)
  Note: prefix cache saves *prefill* cost, not decode.
  Effective $/1M output = $2.49 / (1,635 × 3,600) × 1e6 = $0.423/1M output
  But prefill GPU time freed allows +40% more concurrent users → amortised cost drops.
  Effective $/1M with routing headroom ≈ $0.30/1M
```

**Scenario D: Reasoning model — the expensive case**

```
  Hardware:  H100 SXM (Lambda, $2.49/hr)
  Engine:    vLLM, max_num_seqs=10
  Model:     DeepSeek-R1-8B BF16
  Avg output: 16,384 tokens (reasoning trace + answer)
  Decode:    ~195 tok/s × 10 seqs / (effective utilization) ≈ 400 tok/s blended

  $/1M = $2.49 / (400 × 3,600) × 1e6 = $1.73/1M tokens
  ≈ 4× more expensive than standard serving on same hardware
```

---

## 20.4 Self-Hosted vs. Cloud API vs. Consumer Hardware

`[DEEP DIVE]`

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Decision framework: where to serve                                 │
  │                                                                     │
  │  Volume < 10M tokens/month?                                         │
  │    → API provider (OpenAI, Anthropic, Together, Fireworks)          │
  │       No infra, no ops, pay-per-token pricing.                      │
  │       Typical: $0.30–$3.00/1M depending on model.                  │
  │                                                                     │
  │  Volume 10M–1B tokens/month, latency-sensitive?                     │
  │    → Cloud self-hosted (Lambda, CoreWeave, Vast.ai)                 │
  │       vLLM + KubeRay, GPU rental, no hardware ownership.            │
  │       Typical: $0.08–$0.50/1M at good utilization.                 │
  │                                                                     │
  │  Volume > 1B tokens/month, predictable workload?                    │
  │    → Owned hardware or reserved cloud                               │
  │       H100 cluster amortised + electricity.                         │
  │       Typical: $0.02–$0.15/1M at 80%+ utilization.                 │
  │                                                                     │
  │  Edge / privacy / offline?                                          │
  │    → llama.cpp on consumer hardware (RTX 4090, M2 Ultra, M4 Max)   │
  │       One-time hardware cost, zero per-token fee.                   │
  └─────────────────────────────────────────────────────────────────────┘
```

### API Provider Comparison (mid-2025)

```
  Provider         Model             $/1M input   $/1M output
  ─────────────────────────────────────────────────────────
  OpenAI           GPT-4o              $2.50         $10.00
  OpenAI           GPT-4o-mini         $0.15          $0.60
  Anthropic        Claude Sonnet 4     $3.00         $15.00
  Together.ai      Llama-3.1-70B-T     $0.88          $0.88
  Fireworks.ai     Llama-3.1-8B-I      $0.20          $0.20
  Groq             Llama-3.3-70B-V     $0.59          $0.79
  ─────────────────────────────────────────────────────────
  Self-hosted H100 Llama-3-8B BF16     ~$0.05        ~$0.42
  Self-hosted RTX4090 Llama-3-8B Q4   ~$0.03        ~$0.35
```

For any workload where the model quality of Llama-3-8B is sufficient, self-hosted on consumer
hardware is 1.5–4× cheaper than the cheapest API tier at scale, and 10–50× cheaper than
frontier API models.

---

## 20.5 Apple M2 Ultra: TCO Analysis for 70B Serving

`[DEEP DIVE]`

The Apple M2 Ultra is an outlier: 192 GB of unified memory accessible at 800 GB/s, available
in a consumer device (Mac Studio or Mac Pro) for $3,999–$6,999. It can run a full 70B model
in Q4_K_M quantization without GPU offload.

```
  Llama-3-70B Q4_K_M on Apple M2 Ultra:
    Weights:  ~42 GB (fits in 192 GB unified memory)
    Decode:   ~15–20 tok/s (Metal backend, single sequence)
    Prefill:  ~1,500–2,000 tok/s (compute-bound path)

  Hardware cost: $5,999 (M2 Ultra Mac Studio, 192 GB)
  Amortised over 3 years at 50% utilization:
    Hours of use: 3 × 365 × 24 × 0.5 = 13,140 hours
    Hardware cost/hr: $5,999 / 13,140 = $0.457/hr
  Electricity: 60W × $0.10/kWh = $0.006/hr
  Total cost/hr: $0.463/hr

  Decode throughput at batch=1: 17 tok/s
  $/1M output = $0.463 / (17 × 3,600) × 1e6 = $7.56/1M tokens

  Decode throughput at batch=4 (parallel sequences): ~45 tok/s
  $/1M output = $0.463 / (45 × 3,600) × 1e6 = $2.86/1M tokens
```

**The M2 Ultra is not cheap for API serving.** At $2.86–$7.56/1M it is more expensive than
self-hosted H100 serving the same model. Its value proposition is:

```
  ✓  No cloud dependency — fully offline, air-gapped deployments
  ✓  No per-token cost — fixed monthly amortization regardless of volume
  ✓  70B quality at home — competitive with expensive frontier API models
  ✓  Silence and power efficiency — 60W vs 700W for H100
  ✓  Single binary deployment — llama.cpp, no CUDA, no driver management
```

For a team generating 50M tokens/month from a 70B model in an air-gapped environment, the M2
Ultra at $0.463/hr × 24hr/day × 30 days = $333/month is dramatically cheaper than cloud at
$2.86/1M × 50M = $143/month... wait, cloud wins at that volume. The M2 Ultra breakeven is:

```
  Breakeven = hardware_cost_per_month / api_cost_per_token
            = $333/month / ($2.86/1M × 1M) = 116M tokens/month

  Below 116M tokens/month: cloud API is cheaper (just use Together.ai)
  Above 116M tokens/month: M2 Ultra pays for itself (for 70B quality)
```

---

## 20.6 The Running Case Study: $1.2M → $108K

`[DEEP DIVE]`

Chapter 1 introduced a production scenario: 50,000 concurrent users, $1.2M/month, 28% GPU
utilization. Here is how each technique in this book contributed to the 11× cost reduction:

```
  Starting point:
    Fleet: 80× A100 80GB (on-demand, AWS)
    Cost:  $12.00/hr × 80 = $960/hr = $691,200/month
    Additional: networking, storage, ops → $1,200,000/month
    GPU utilization: 28%

  ──────────────────────────────────────────────────────────────────
  Step 1: Switch to vLLM (from custom serving)            Ch 6–8
    Continuous batching raises utilization: 28% → 72%
    Same fleet now handles 2.6× more traffic.
    Cost per token: ÷ 2.6
    Running cost: ~$460,000/month

  Step 2: Prefix caching for RAG workload (73% hit rate)  Ch 11
    73% of prefill GPU time eliminated.
    Effective throughput: +40% headroom.
    Reduce fleet by 28%: 80 → 58 GPUs.
    Running cost: ~$333,000/month

  Step 3: Chunked prefill + knob tuning                   Ch 11, 14
    p99 TTFT drops from 4.1s → 0.9s.
    Enables tighter SLA → can run smaller fleet.
    Fleet: 58 → 48 GPUs.
    Running cost: ~$276,000/month

  Step 4: Disaggregated prefill/decode                    Ch 18
    Replace 48× A100 with: 8× H100 SXM prefill + 24× H100 NVL decode
    on Lambda Labs ($2.49/hr).
    Cost: 32 × $2.49 × 24 × 30 = $57,830/month
    Running cost: ~$58,000/month (includes networking, ops overhead)

  Step 5: Spot instances for decode pool                  Ch 19
    24× H100 NVL decode workers → 70% on spot ($0.87/hr).
    Checkpoint/drain on preemption (see Section 20.7).
    Cost: 8 × $2.49 + 17 × $0.87 + 7 × $2.49 = $53.54/hr
    Monthly: $53.54 × 24 × 30 = $38,549/month + ops ≈ $45,000/month

  Step 6: Semantic cache (Chapter 30 preview)
    73% FAQ hit rate eliminates those requests entirely.
    Effective GPU load: -40%.
    Final fleet: 6 prefill + 18 decode GPUs.
    Final cost: ~$108,000/month (including all overheads)
  ──────────────────────────────────────────────────────────────────

  Total reduction: $1,200,000 → $108,000 = 11.1× cheaper
  Per-token cost: $0.024/1M (down from $0.24/1M)
```

```
  Cost Reduction Waterfall:

  $1,200K  ████████████████████████████████████████████████  Starting point
   $460K  ████████████████████  After vLLM + continuous batching
   $333K  ██████████████        After prefix caching
   $276K  ████████████          After chunked prefill + knobs
    $58K  ██                    After disaggregation + Lambda Labs
    $45K  █▉                    After spot instances
   $108K  ████                  Final (incl. ops, networking, storage)
          0        $400K      $800K     $1,200K
```

---

## 20.7 Spot Instance Interruption Handling

`[DEEP DIVE]`

Spot instances can be reclaimed by the cloud provider at any time with a 2-minute warning.
For LLM serving, the right response is: **drain, not checkpoint**.

Checkpointing a running LLM inference worker (saving the in-progress KV state) is complex,
unreliable, and slow. Draining is simple: stop accepting new requests, let in-flight requests
complete, then terminate cleanly.

```
  Spot interruption handler (pseudo-code):

  on SIGTERM (spot reclaim warning):
    1. Set readiness probe → NotReady
       (load balancer stops routing new requests immediately)
    2. Wait for in-flight requests to complete
       (up to terminationGracePeriodSeconds = 120s for standard,
        600s for reasoning)
    3. Log: "spot instance reclaimed, N requests drained"
    4. Exit cleanly

  On new pod spin-up (replacement spot):
    1. Pull model weights from S3/NFS (pre-cached on node)
    2. Cold start (120s for 8B, 240s for 70B)
    3. Readiness probe passes → traffic resumes

  Client-side resilience:
    - Retry with exponential backoff (3 attempts, 1s/2s/4s)
    - Request IDs enable exactly-once delivery tracking
    - Streaming: on connection drop, resume from last received token
```

**Which pools to put on spot:**

```
  ✓  Decode workers — stateless between requests, easy to drain
  ✓  Batch/offline processing — no latency SLA, interruption is fine
  ✗  Prefill workers — 2-minute reclaim warning may not be enough for
     long prefills; safer on on-demand or reserved
  ✗  Head node (Ray GCS) — must be stable; use on-demand
```

---

## 20.8 Cost Matrix: The Full Picture

`[FOUNDATIONAL]`

```
  $/1M output tokens  |  decode-only, sustained utilization
  ──────────────────────────────────────────────────────────────────────────
                        RTX 4090        A100 80GB       H100 SXM
                       (Vast spot)    (Lambda $2.49)  (Lambda $2.49*)
  ──────────────────────────────────────────────────────────────────────────
  Llama-3-8B BF16    —              $0.42 (B=32)      $0.13 (B=32)
  Llama-3-8B Q4_K_M  $0.09 (B=4)   $0.32 (B=4)       —
  Llama-3-70B BF16   —              $1.68 (B=4)       $0.52 (B=4)
  Llama-3-70B Q4_K_M $0.82 (B=1)   $0.63 (B=2)       —
  DeepSeek-R1-8B     $0.41 (B=1)   $1.73 (B=10)      $0.53 (B=10)
  ──────────────────────────────────────────────────────────────────────────
  * H100 not available on Lambda at publication; prices from CoreWeave/Vast
  Batch = max concurrent sequences at full utilization
  All figures ± 20% depending on actual prompt/output ratio
```

---

## 20.9 Reserved Instances and Breakeven Analysis

`[DEEP DIVE]`

When volume is predictable and high, reserved instances (1- or 3-year commitments) offer
30–45% savings over on-demand, with no interruption risk. The decision is a classic ROI
calculation.

### Reserved vs. On-Demand Breakeven

```
WORKED EXAMPLE 20.3 — Reserved Instance ROI
─────────────────────────────────────────────────────────────────────
GPU: H100 SXM 80GB (AWS p5.48xlarge, 8× H100)

On-demand:  $98.32/hr
Reserved 1-year: $67.40/hr (31% discount)
Reserved 3-year: $52.80/hr (46% discount)

Annual cost:
  On-demand (if running 24/7): $98.32 × 8760 = $861,283/yr
  Reserved 1-year:             $67.40 × 8760 = $590,424/yr  → saves $270,859/yr
  Reserved 3-year:             $52.80 × 8760 = $462,528/yr  → saves $398,755/yr

Break-even utilization (Reserved vs. On-demand):
  If you can't keep the GPU > X% busy, on-demand is cheaper.
  Because reserved is a committed rate (you pay whether running or not):

  Break-even = reserved_rate / on_demand_rate
             = $67.40 / $98.32 = 68.5% utilization

  If utilization > 68.5%: 1-year reserved saves money
  If utilization < 68.5%: on-demand is cheaper (pay only when running)
─────────────────────────────────────────────────────────────────────
```

This breakeven calculation is why reserved instances are appropriate only for
**predictable, high-utilization** workloads. An LLM serving cluster with:
- Stable 24/7 traffic (SaaS product, B2B API) → reserve
- Batch jobs (nightly data processing) → spot
- Experimental / development workloads → on-demand or spot

### Multi-Cloud Arbitrage

No single cloud provider has the cheapest GPUs at every point in time. GPU spot markets
fluctuate based on aggregate demand. Sophisticated operators run **multi-cloud burst capacity**:

```
  Multi-Cloud GPU Strategy
  
  Primary cluster: CoreWeave or Lambda (reserved, stable pricing)
  ┌────────────────────────────────────────────────────────────────┐
  │  H100 × 8 (reserved, CoreWeave ~$20/hr for 1-year reserved)  │
  │  Always on, handles baseline traffic                           │
  │  vLLM + KubeRay, Kubernetes HPA                               │
  └────────────────────────────────────────────────────────────────┘
         ↓ overflow traffic (burst above baseline)
  Burst tier: Vast.ai or Runpod (spot, volatile pricing)
  ┌────────────────────────────────────────────────────────────────┐
  │  H100 or A100 spot (~$2–5/hr, bid-based)                     │
  │  Spun up on demand via Kubernetes Cluster Autoscaler          │
  │  vLLM started with same image as primary cluster              │
  │  Handles traffic spikes; terminated when below threshold       │
  └────────────────────────────────────────────────────────────────┘
```

**KubeRay multi-cloud burst configuration:**

```yaml
# RayCluster resource — heterogeneous worker groups
apiVersion: ray.io/v1alpha1
kind: RayCluster
spec:
  headGroupSpec:
    replicas: 1
    template:
      spec:
        nodeSelector:
          cloud-provider: coreweave   # head always on stable provider
        containers:
        - name: ray-head
          image: vllm/vllm-openai:latest

  workerGroupSpecs:
  - groupName: primary-workers
    replicas: 2
    minReplicas: 2
    maxReplicas: 4
    template:
      spec:
        nodeSelector:
          cloud-provider: coreweave   # reserved H100s
          node.kubernetes.io/instance-type: H100
        tolerations: []

  - groupName: burst-workers
    replicas: 0
    minReplicas: 0
    maxReplicas: 8
    template:
      spec:
        nodeSelector:
          cloud-provider: vastai       # spot A100s
          node.kubernetes.io/instance-type: A100
        tolerations:
        - key: "spot"
          operator: "Exists"
          effect: "NoSchedule"
```

The burst workers scale from 0 to 8 on demand. Because each vLLM worker is stateless
(KV state lives per-request, not per-worker), there is no state migration when workers
appear or disappear.

### Reserved vs. Self-Hosted Decision Matrix

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Monthly tokens    Strategy              Why                         │
  ├──────────────────────────────────────────────────────────────────────┤
  │  < 10M             API provider          No infra overhead           │
  │  10M – 100M        Cloud on-demand/spot  Flexible, pay-as-you-go     │
  │  100M – 2B         Cloud reserved        Predictable cost, no capex  │
  │  > 2B              Own hardware          CapEx wins vs. reserved OpEx│
  └──────────────────────────────────────────────────────────────────────┘

  Own hardware crossover (H100 example):
    H100 purchase price: ~$30,000 (DGX H100 per GPU)
    Cloud reserved 3-year: $52.80/hr × 26,280 hours = $1,388,160 per GPU
    Electricity + cooling: ~$4,000/yr × 3 = $12,000

    Own hardware 3-year total: $30,000 + $12,000 = $42,000 per GPU
    vs. reserved cloud:        $1,388,160 per GPU
    → Own hardware: 33× cheaper over 3 years at 100% utilization

  [COMMON TRAP] The 33× figure assumes 100% utilization, no downtime, no
  cooling infrastructure build-out cost, no ops headcount. At the realistic
  level of 70% utilization with 1 FTE for infrastructure, the advantage
  drops to roughly 8–12× — still compelling at >2B tokens/month, but
  not at lower volumes.
```

---

## Summary

Cost engineering is not a separate concern from the rest of this book — it is the purpose of
every technique in it. Each chapter's optimization translates directly into a cost reduction:
continuous batching eliminates idle GPU time, prefix caching eliminates redundant prefill,
disaggregation routes each workload to the right hardware, speculative decoding reduces decode
time, and quantization shrinks both the memory footprint and the weight-loading bottleneck.

The $1.2M → $108K case study shows that none of these improvements required a different model
or a different business. They required understanding what the hardware was actually doing and
routing work accordingly. The techniques compound: each one frees headroom that the next one
can exploit.

The cost metric — $/1M output tokens — is the right compass. Hardware specs, FLOP/s numbers,
and benchmark rankings are inputs to that calculation, not ends in themselves.

---

## Key Terms

- **$/1M output tokens** — the canonical LLM serving cost metric; normalizes across hardware,
  engines, and model sizes.
- **Spot instance** — cloud VM that can be reclaimed with 2-minute warning; 30–75% cheaper
  than on-demand; suitable for stateless decode workers with graceful drain.
- **GPU utilization** — fraction of time tensor cores are active; low utilization means high
  cost per token; continuous batching is the primary lever for raising it.
- **Amortised hardware cost** — capital expenditure divided by expected lifetime usage hours;
  used to compare owned hardware against cloud rental.
- **Breakeven volume** — monthly token volume at which owned hardware becomes cheaper than
  cloud API; depends on model quality tier, hardware cost, and utilization assumptions.
- **Drain** — the preferred spot-interruption response: stop accepting requests, complete
  in-flight work, then terminate; simpler and more reliable than checkpointing KV state.

---

*Next: Chapter 21 — Security: API Hardening, Injection, Isolation*


---

## Self-Check Questions

1. A vLLM deployment runs 4× A100 80 GB at $3.20/GPU-hr. At 70% average utilization over a month (730 hrs), compute the monthly GPU cost. If average throughput is 1 800 tokens/s, compute the $/million-token cost. *(Section 20.1)*

2. Prefix caching has a 65% cache hit rate on a workload where cache hits reduce prefill compute by 80%. If prefill accounts for 40% of total GPU time, compute the effective GPU utilization reduction. *(Section 20.3)*

3. Spot instances for A100 on AWS cost $1.10/hr vs $3.20/hr on-demand. Spot interruption rate is 15%/hr. For a request rate of 200 req/min with a 2 s retry deadline, compute the expected additional latency from interruptions. *(Section 20.4)*

4. Quantising a 70B model from BF16 to FP8 doubles throughput (same GPU count). Monthly on-demand cost is $45 000. Ignoring quality effects, compute the new cost and the annual saving. *(Section 20.2)*

5. You have 50 000 daily active users with an average of 3 requests/user/day, each generating 800 tokens. Compute the daily token volume and, using the $/million-token figure from question 1, the daily and monthly cost. *(Section 20.1)*
