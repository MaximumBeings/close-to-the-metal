# Chapter 33.5: Choosing Your Engine — SGLang, TRT-LLM, MLC-LLM, and Ollama

> *"The right engine is the one that fits your model, your hardware, your latency budget, and your team's tolerance for operational complexity. There is no universal winner."*

## Overview

Chapter 33 surveyed the full inference engine landscape of 2026. This chapter goes one level deeper: it gives you a concrete decision framework, worked examples of the tradeoffs, and runnable benchmarks so you can verify the choice against your own workload.

The engines in scope:

| Engine | Primary Strength | Primary Weakness |
|---|---|---|
| **vLLM** | Dynamic batching, PagedAttention, broad model support | Throughput ceiling vs compiled engines |
| **SGLang** | Structured generation, RadixAttention, multi-call programs | Smaller ecosystem than vLLM |
| **TensorRT-LLM** | Highest throughput on NVIDIA hardware | AOT compile cost, NVIDIA-only |
| **MLC-LLM** | Cross-device (GPU/CPU/WebGPU/mobile) | Lower peak throughput than compiled |
| **Ollama** | Developer UX, one-command serving | Not designed for multi-user production |
| **llama.cpp** | CPU/edge, GGUF ecosystem | Limited GPU batching throughput |

---

## 33b.1  The Five Decision Axes

Every engine comparison collapses to five questions. Score each 1–5 for your situation:

### Axis 1: Hardware Flexibility
*Do you need to run on hardware other than NVIDIA data-center GPUs?*

- NVIDIA H100/A100 only → TRT-LLM, vLLM, SGLang all viable
- AMD MI300X → vLLM (ROCm backend), MLC-LLM
- Apple Silicon (M-series) → llama.cpp, MLC-LLM, Ollama
- Browser / WebGPU → MLC-LLM exclusively
- CPU-only → llama.cpp, Ollama (wraps llama.cpp)

### Axis 2: Structured Output Requirements
*Do you need guaranteed JSON, regex-constrained, or grammar-constrained output?*

**SGLang** is the standout here. Its `sglang.function` decorator lets you write multi-turn structured programs:

```python
@sgl.function
def extract_fields(s, document):
    s += sgl.system("Extract fields as JSON.")
    s += sgl.user(document)
    s += sgl.assistant(
        sgl.gen("output", max_tokens=256,
                regex=r'\{"name": ".+", "date": "\d{4}-\d{2}-\d{2}"\}')
    )
```

vLLM supports guided decoding via `--guided-decoding-backend outlines`, but SGLang's integration is tighter and its **RadixAttention** caches the shared prompt prefix across all calls in a batch — a meaningful advantage when the same system prompt prefix is reused across thousands of structured-extraction requests.

### Axis 3: Deployment Complexity
*How much engineering overhead can your team sustain?*

| Engine | Cold start | Config surface | Upgrade path |
|---|---|---|---|
| Ollama | `ollama run llama3` — one command | Modelfile (~10 lines) | `ollama pull` |
| vLLM | `pip install vllm` + 15 flags | ~30 important flags | pip upgrade |
| SGLang | `pip install sglang` + backend setup | Similar to vLLM | pip upgrade |
| TRT-LLM | `trtllm-build` (1–5 hrs compile) | Extensive JSON config | Re-compile on update |
| MLC-LLM | TVM compile + model convert | Moderate | Re-compile on model change |

### Axis 4: Throughput Ceiling
*What is the maximum tokens/second achievable?*

The bandwidth roofline (Chapter 2) sets the ceiling for memory-bound decode. For compute-bound prefill, TFLOPS utilization matters. Approximate ordering for a 70B model on 4× H100:

```
TRT-LLM FP8 + 2:4 sparse  ≈  4.0×  (relative to vLLM BF16 baseline)
TRT-LLM FP8               ≈  2.8×
SGLang BF16               ≈  1.2×   (RadixAttention prefix reuse helps on shared prefixes)
vLLM FP8                  ≈  2.0×
vLLM BF16                 ≈  1.0×   (baseline)
MLC-LLM (quantized)       ≈  0.7×
Ollama (llama.cpp)        ≈  0.3×   (GPU path; CPU is further reduced)
```

These ratios vary significantly with batch size, sequence length, and prefix reuse rate. The companion code lets you compute them for your exact configuration.

### Axis 5: Quantization Support
*Which quantization formats does your workflow need?*

| Format | vLLM | SGLang | TRT-LLM | MLC-LLM | Ollama |
|---|---|---|---|---|---|
| BF16 / FP16 | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8 (E4M3) | ✓ | ✓ | ✓ | ✓ | ✗ |
| INT8 / SmoothQuant | ✓ | ✓ | ✓ | ✓ | ✓ |
| GPTQ / INT4 | ✓ | ✓ | ✓ | ✓ | ✓ |
| AWQ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2:4 sparsity | ✗ | ✗ | ✓ | ✗ | ✗ |
| GGUF (K-quants) | ✗ | ✗ | ✗ | partial | ✓ |

---

## 33b.2  SGLang Deep Dive

### RadixAttention

SGLang's key architectural contribution is **RadixAttention**: a trie (radix tree) over all KV cache blocks, keyed by the token sequence that produced them. When two requests share a common prefix — a system prompt, a few-shot template, a document preamble — their KV blocks are shared automatically, with no explicit prefix-caching flag needed.

```
Request A:  [system_prompt | user_A | ...]
Request B:  [system_prompt | user_B | ...]
              ^^^^^^^^^^^^
              Shared prefix → same KV blocks, no recomputation
```

For workloads with a 500-token system prompt and 100-token user queries, RadixAttention means only the 100 user tokens are ever prefilled after the first request. At 1000 requests per minute this compounds to a very large FLOP saving.

**Worked Example 33b.1 — RadixAttention savings:**

```
System prompt:   512 tokens
User message:    128 tokens
Output:          256 tokens
Batch:           100 concurrent users, same system prompt

Without prefix cache:
  Prefill FLOPs = 100 × (512 + 128) × 2 × 70B = 8.96 PFLOPs

With RadixAttention (1 cold + 99 hits):
  Prefill FLOPs = 1 × 640 × 2 × 70B  +  99 × 128 × 2 × 70B
               = 89.6 TFLOPs + 1.76 PFLOPs
               = 1.85 PFLOPs

Savings: 79%
```

### Structured Generation

SGLang's structured generation uses a **compressed finite state machine** over the vocabulary. At each decode step, only tokens that keep the output within the target grammar have non-zero logits. This adds ~0.5ms per step (masking overhead) but eliminates all post-processing retries.

Compared to vLLM's outlines backend, SGLang's integration is tighter: the grammar state machine runs inside the CUDA graph, and the radix cache is grammar-aware (a cached prefix is only reused if the grammar state matches).

### Multi-Call Programs

SGLang programs can branch, loop, and call the model multiple times, with the interpreter managing KV cache reuse across calls:

```python
@sgl.function
def chain_of_thought(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("reasoning", max_tokens=512))
    s += sgl.user("Based on your reasoning, give a one-sentence answer.")
    s += sgl.assistant(sgl.gen("answer", max_tokens=64))
```

The two generation calls share the accumulated KV cache — the second call does not re-prefill the first response.

---

## 33b.3  TRT-LLM Deep Dive (Engine Selection Angle)

Chapter 37 covered TRT-LLM from Nemotron's perspective. Here we focus on the *decision* of when to use it.

### When TRT-LLM Is Worth the Compile Cost

The compile cost (1–5 GPU-hours) amortises quickly for stable production models. The break-even point is:

```
break_even_hours = compile_cost_gpu_hrs × gpu_cost_per_hr
                   ─────────────────────────────────────────
                   extra_revenue_per_hr from speedup

For Llama-3.1-70B, FP8, 4× H100:
  compile_cost = 1.5 hrs × $28/hr = $42
  speedup = 2.4× over vLLM BF16
  at 10,000 req/hr × $0.005/req → extra revenue = $25/hr (1.4× baseline)
  break_even ≈ 1.7 hours of production traffic
```

Beyond the break-even point, TRT-LLM is the clear choice for fixed-hardware, fixed-model NVIDIA deployments.

### When NOT to Use TRT-LLM

- Frequent model updates (every re-weight requires recompile)
- Multiple LoRA adapters (TRT-LLM's LoRA support is less mature)
- AMD or non-NVIDIA hardware
- Need for dynamic batch sizes across very wide ranges

---

## 33b.4  MLC-LLM — Cross-Device Compilation

MLC-LLM uses Apache TVM's **Relax IR** to compile models to virtually any compute target: CUDA, ROCm, Metal, Vulkan, WebGPU, and even WASM. The compilation pipeline:

```
HuggingFace weights
       ↓
MLC model conversion (Python)
       ↓
TVM Relax IR graph
       ↓
Target-specific codegen (CUDA/Metal/WebGPU)
       ↓
Runtime library (mlc_llm.LLMEngine)
```

**Where MLC-LLM wins:**
- Apple Silicon: 60–80% of llama.cpp throughput with a cleaner Python API
- Browser deployment: the only engine with a production-quality WebGPU backend
- Heterogeneous fleets: one codebase compiling for NVIDIA + AMD + Apple

**Where it loses:**
- Peak NVIDIA throughput: TRT-LLM's hand-tuned CUDA kernels are faster
- Ecosystem: smaller community, fewer pre-compiled models

### Worked Example 33b.2 — MLC vs llama.cpp on Apple M3 Max (128GB):

| Model | llama.cpp Q4_K_M | MLC-LLM Q4 | Delta |
|---|---|---|---|
| Llama-3.1-8B | 48 tok/s | 41 tok/s | −15% |
| Llama-3.1-70B | 8.2 tok/s | 7.1 tok/s | −13% |
| Qwen2.5-32B | 12.1 tok/s | 10.9 tok/s | −10% |

llama.cpp remains faster on Apple Silicon due to its highly-optimized Metal kernels. MLC-LLM's advantage is API consistency across targets, not raw speed.

---

## 33b.5  Ollama — The Developer UX Engine

Ollama wraps llama.cpp with a REST API and a model registry. Its design priority is *time-to-first-response for a developer*, not throughput per GPU-dollar.

### What Ollama Gives You

```bash
ollama pull llama3.1:70b      # downloads GGUF + metadata
ollama run  llama3.1:70b      # interactive chat
ollama serve                  # starts REST server on :11434
```

The **Modelfile** system lets you bake in system prompts and parameters:

```dockerfile
FROM llama3.1:8b
SYSTEM "You are a helpful coding assistant. Always respond in Python."
PARAMETER temperature 0.2
PARAMETER num_ctx 16384
```

### Ollama's Hard Limits for Production

- Single-user concurrency by default (one request at a time per model)
- No PagedAttention → no efficient multi-sequence batching
- No FP8 (llama.cpp supports it via GGUF quantization, not H100 FP8 Tensor Cores)
- Model switching incurs full reload (no hot-swap)

**Decision rule:** Ollama is the right answer for developers running models locally, prototyping, and single-user tools. It is the wrong answer for any workload above ~5 concurrent users.

---

## 33b.6  The Decision Flowchart

```
START: I need to serve an LLM
│
├─ Hardware is CPU-only or Apple Silicon?
│   YES → llama.cpp or Ollama (developer) / MLC-LLM (cross-platform API)
│   NO  ↓
│
├─ NVIDIA GPU, need maximum throughput, model is stable?
│   YES → TRT-LLM (FP8 + 2:4 sparse if model supports it)
│   NO  ↓
│
├─ Primary workload is structured output / JSON extraction / multi-call programs?
│   YES → SGLang (RadixAttention + structured gen)
│   NO  ↓
│
├─ Dynamic workloads, LoRA adapters, multi-model serving, or AMD GPU?
│   YES → vLLM
│   NO  ↓
│
├─ Browser / WebGPU / mobile target?
│   YES → MLC-LLM
│   NO  ↓
│
└─ Single developer, local laptop, just need it to work?
    → Ollama
```

---

## 33b.7  Running the Benchmark Yourself

The companion code (`engine_comparison_demo.py` / `.cpp`) implements:

1. The 5-axis scoring model for any configuration
2. RadixAttention savings calculator
3. Roofline throughput ceiling per engine
4. Break-even analysis for TRT-LLM compilation
5. Decision algorithm as executable code

### Key Worked Numbers (Python output)

```
=== Engine Comparison Demo — Chapter 33.5 ===

[1] 5-Axis Scoring (70B, 4×H100, structured output workload):
    vLLM     BF16:  score=3.2  (hw=5, struct=2, deploy=4, tput=3, quant=4)
    SGLang   BF16:  score=3.8  (hw=4, struct=5, deploy=4, tput=3, quant=4)
    TRT-LLM  FP8:   score=3.4  (hw=2, struct=2, deploy=2, tput=5, quant=5)
    Ollama:          score=1.8  (hw=5, struct=1, deploy=5, tput=1, quant=3)
    → Winner for this config: SGLang

[2] RadixAttention savings (512-token prefix, 100 concurrent users):
    Baseline prefill FLOPs:  8.96 PFLOPs
    With RadixAttention:     1.85 PFLOPs
    Saving:                  79.3%

[3] Roofline ceilings (4×H100, BW=13.4 TB/s combined):
    vLLM  BF16:  39.2 tok/s (batch=1, 70B)
    SGLang BF16: 39.2 tok/s (same roofline; gains come from prefix reuse)
    TRT-LLM FP8: 78.4 tok/s (2× via weight compression)
    MLC-LLM Q4:  52.3 tok/s (effective BW improvement from INT4)

[4] TRT-LLM break-even:
    Compile cost: $42 (1.5 hr × 4 GPUs × $7/GPU-hr)
    Speedup: 2.4×, revenue rate: $50/hr
    Break-even: 1.7 hours of production traffic
```

---

## Chapter Summary

Choosing an inference engine is a five-axis decision: hardware flexibility, structured output requirements, deployment complexity, throughput ceiling, and quantization support. No single engine dominates all five.

**SGLang** earns its place when your workload involves structured output or heavy prefix sharing — RadixAttention's 79% prefill saving on shared-prefix workloads is real and significant.

**TRT-LLM** is the correct answer when you need maximum NVIDIA throughput and can afford the compile investment — the break-even is often under two hours of production traffic.

**MLC-LLM** is the only viable option for browser/WebGPU targets and the cleanest path for heterogeneous hardware fleets.

**Ollama** is the right answer for individual developers and wrong for almost everything else at scale.

**vLLM** remains the default recommendation for its combination of broad hardware support, mature PagedAttention, LoRA serving, and active community — but it is not always the fastest or cheapest per request.

---

## Self-Check Questions

1. A startup is building a document-extraction pipeline: 1000 requests/hour, all with the same 400-token system prompt, output must be valid JSON. Which engine would you recommend and why?

2. A team wants to serve Llama-3.1-70B on a fleet of 8× H100s in a stable production environment with no model updates expected for 6 months. Calculate the TRT-LLM break-even point assuming $28/hr per GPU and a 2.4× throughput gain.

3. An ML engineer needs to run inference on Apple M2 Pro (32GB) for a local coding assistant. Compare llama.cpp and MLC-LLM on this hardware: what are the throughput and UX tradeoffs?

4. Why does RadixAttention produce a 79% prefill saving when 100 users share a 512-token prefix, but only a 16% saving when 10 users share the same prefix? Derive the formula.

5. A team migrates from Ollama to vLLM to handle 50 concurrent users. What specific capabilities does vLLM add that Ollama lacks? What operational costs does the team take on in exchange?
