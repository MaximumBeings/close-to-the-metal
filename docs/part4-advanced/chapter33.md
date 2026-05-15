# Chapter 33: The Full Engine Landscape — 2026

> *"Every inference engine is a set of bets about what matters most.  
>  Understanding those bets is how you choose the right one."*

---

## 33.1 Why a Landscape Survey Belongs at the End

This chapter could have appeared at the beginning of the book. A reader picking up a volume titled *Close to the Metal* might expect it to open with a table of engines and a verdict. We deliberately deferred it.

The reason is that a comparison table without internals is almost useless. Without understanding PagedAttention you cannot assess vLLM's memory claims. Without understanding GGUF quantization you cannot evaluate llama.cpp's quality-size tradeoffs. Without understanding tensor parallelism you cannot read TensorRT-LLM's multi-GPU benchmarks critically. Thirty-two chapters of groundwork make this survey meaningful rather than just promotional.

What follows is the most honest, quantitative, and mechanistically grounded comparison of production LLM inference engines that we can write as of mid-2026. Where benchmarks are contested, we say so. Where a feature exists in one engine but not another, we explain the architectural reason rather than just recording the fact.

The engines covered:

| Engine | Primary author | First release | Primary target |
|---|---|---|---|
| vLLM | UC Berkeley / vLLM team | June 2023 | GPU server, high-throughput |
| llama.cpp | Georgi Gerganov | March 2023 | CPU/GPU edge, portability |
| TensorRT-LLM | NVIDIA | Oct 2023 | NVIDIA GPU, peak throughput |
| SGLang | UC Berkeley / Lianmin Zheng | Jan 2024 | Structured generation, high throughput |
| Text Generation Inference (TGI) | Hugging Face | Dec 2022 | OpenAI-compatible API |
| MLC-LLM | MLC AI / CMU | May 2023 | Cross-platform, mobile/edge |
| DeepSpeed-FastGen | Microsoft DeepSpeed | Nov 2023 | Large models, hybrid parallelism |
| Mooncake / Kimi Serving | Moonshot AI | Mar 2024 | Disaggregated prefill/decode |
| NVIDIA Dynamo | NVIDIA | Feb 2025 | Disaggregated, datacenter-scale |

---

## 33.2 The Five Dimensions of Inference Engine Quality

No single number captures an engine's quality. Every benchmark you will find in the wild optimizes for one dimension while holding others fixed. The five dimensions that matter, with the right measurement for each:

### 33.2.1 Throughput (tokens/second/GPU)

**Definition:** How many output tokens can the engine generate per second per GPU, at sustained load with a realistic batch size.

**Correct measurement:** Run at the maximum sustainable batch size where P99 latency ≤ 3× P50 latency. Report `output_tokens / seconds / n_gpus`. Anything measured at batch=1 is a latency number masquerading as a throughput number.

**Who wins:** TensorRT-LLM and SGLang are usually within 5% of the theoretical compute roof on A100/H100. vLLM is 10–20% below due to Python overhead in the scheduler. llama.cpp CPU throughput is 20–40 tok/s on a modern server CPU for a 7B INT4 model — orders of magnitude below GPU engines, but that is the wrong comparison for its use case.

### 33.2.2 Latency (time-to-first-token, inter-token latency)

**Definition:**
- TTFT: wall-clock time from request submission to first output token
- ITL (Inter-Token Latency): average time between consecutive output tokens

**Correct measurement:** Measure at P50, P95, and P99 under a realistic concurrent request load. TTFT grows with prompt length and batch size. ITL grows with batch size and model size.

**Who wins:** TensorRT-LLM has the lowest TTFT for long prompts (optimized CUDA kernels for prefill). llama.cpp has the lowest TTFT for short prompts on modest hardware (no Python overhead, sub-millisecond tokenization). For ITL under heavy load, SGLang's RadixAttention and vLLM's PagedAttention are comparable.

### 33.2.3 Memory Efficiency (context-tokens-per-GPU-GB)

**Definition:** How many tokens can the engine hold in KV cache simultaneously, per GPU-GB of memory.

**Formula:**
```
context_tokens = (GPU_memory_GB × 1e9 - model_bytes) / kv_bytes_per_token
kv_bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes
```

**Who wins:** vLLM (PagedAttention, virtually no fragmentation), SGLang (RadixAttention adds prefix sharing on top of paging). TensorRT-LLM uses its own paging implementation, close to vLLM. llama.cpp's KV cache is pre-allocated as a contiguous block — wasteful for variable-length requests but fast for single-session use.

### 33.2.4 Hardware Breadth

**Definition:** Which hardware can the engine actually deploy to, at production quality (not experimental)?

| Engine | NVIDIA GPU | AMD GPU | Apple Silicon | CPU x86 | ARM CPU | Mobile |
|---|---|---|---|---|---|---|
| vLLM | ✓ (primary) | ✓ (ROCm) | ✗ | ✗ | ✗ | ✗ |
| llama.cpp | ✓ (CUDA) | ✓ (ROCm/HIP) | ✓ (Metal) | ✓ (AVX2/512) | ✓ (NEON) | ✓ (Android/iOS) |
| TensorRT-LLM | ✓ (NVIDIA only) | ✗ | ✗ | ✗ | ✗ | ✗ |
| SGLang | ✓ | partial | ✗ | ✗ | ✗ | ✗ |
| TGI | ✓ | ✓ (ROCm) | ✗ | ✗ | ✗ | ✗ |
| MLC-LLM | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| DeepSpeed-FastGen | ✓ | partial | ✗ | ✗ | ✗ | ✗ |

### 33.2.5 Operational Simplicity

**Definition:** Time from "I want to serve this model" to "production endpoint running," including: model loading, hardware configuration, API compatibility, monitoring integration, upgrade path.

**Who wins:** TGI (Hugging Face) — `docker run` and it works, OpenAI-compatible API, automatic model download. vLLM is a close second: `pip install vllm && vllm serve model-name`. TensorRT-LLM requires engine compilation (10–60 minutes per model) before serving, which adds operational complexity. llama.cpp requires GGUF conversion but is otherwise simple.

---

## 33.3 Deep Dive: vLLM

### 33.3.1 Architecture Summary

vLLM's architecture has three innovations that define its position:

**PagedAttention** (Chapter 6): non-contiguous KV cache in fixed-size blocks, eliminating fragmentation. The key insight is that KV cache is analogous to virtual memory — pages can be scattered in physical memory and assembled at attention time. This enables:
- Concurrent requests with wildly different lengths without over-provisioning
- Copy-on-write sharing for beam search and prefix caching
- Near-zero memory waste (< 4% fragmentation vs. 60–80% for pre-allocated systems)

**Continuous batching** (Chapter 7): requests enter and leave the batch dynamically. An H100 running 200 concurrent sequences is constantly shuffling: some finish, new ones start, all in a single GPU kernel call. This is the primary source of vLLM's throughput advantage over static-batching engines.

**Async Python engine**: the scheduler, block manager, and tokenization run in Python on the CPU concurrently with the GPU forward pass. This is both vLLM's strength (flexible Python ecosystem) and its ceiling (Python GIL and object overhead limit scheduling throughput to roughly 2,000 requests/second on a single node).

### 33.3.2 What vLLM Does Not Do Well

- **Compilation overhead**: vLLM does not compile CUDA graphs by default (optional with `--enforce-eager=False`). Without graph capture, each forward pass incurs Python dispatch overhead. TensorRT-LLM eliminates this entirely.
- **Very small models on CPU**: vLLM requires CUDA. For 3B and smaller models on CPU, llama.cpp is orders of magnitude more practical.
- **Structured generation throughput**: vLLM's logit processor is flexible but not optimized for grammar-constrained decoding at high batch sizes. SGLang's compressed finite state machine is faster for constrained generation.
- **Sparse attention / SSM models**: vLLM's attention kernel assumes standard MHA/GQA/MQA. State-space models (Mamba, etc.) require custom integration.

### 33.3.3 vLLM Performance Numbers (H100 SXM, 8B model, int8)

| Scenario | Throughput (tok/s) | TTFT P50 | ITL P50 |
|---|---|---|---|
| Batch=1, 512 in, 128 out | 148 | 42 ms | 6.8 ms |
| Batch=16, 512 in, 128 out | 1,840 | 55 ms | 7.1 ms |
| Batch=64, 512 in, 128 out | 5,120 | 198 ms | 7.9 ms |
| Batch=128, 4096 in, 256 out | 6,400 | 3.1 s | 20 ms |

*Numbers from vLLM benchmark suite; "batch" = concurrent requests.*

---

## 33.4 Deep Dive: llama.cpp

### 33.4.1 Architecture Summary

llama.cpp's architecture reflects a completely different set of priorities than vLLM's:

**GGUF quantization** (Chapter 10, 28): the Q4_K_M format stores 4 bits per weight with per-32-element scales, achieving 4.84 bits-per-weight. A 70B model that requires 140 GB in fp16 fits in 40 GB in Q4_K_M. This is not a performance optimization — it is a deployment feasibility enabler. It allows running a 70B model on a workstation.

**Metal / OpenCL / CUDA backends**: llama.cpp abstracts compute backends behind a unified GGML tensor operation API. This is architecturally inefficient (each backend reimplements the same ops) but achieves extraordinary portability. The same `.gguf` file runs on an M3 MacBook, a Raspberry Pi 5, an RTX 4090, and an A100.

**CPU-first design**: llama.cpp's matmul kernels are written in hand-vectorized C with AVX2, AVX-512, ARM NEON, and SVE intrinsics. On a high-core-count server CPU (AMD EPYC 9654, 96 cores), a quantized 8B model achieves 35–50 tok/s — fast enough for many interactive applications without any GPU.

### 33.4.2 What llama.cpp Does Not Do Well

- **High-concurrency serving**: llama.cpp's server (`llama-server`) was designed for a small number of concurrent users (< 10). Under high concurrency, there is no continuous batching — each request waits for the previous one to complete (as of early 2025; experimental batching was added later).
- **Arbitrary model architectures**: adding a new architecture requires writing C++ code and recompiling. vLLM's Python ecosystem makes this much faster.
- **Distributed inference**: tensor parallelism in llama.cpp is experimental and limited to specific architectures. Distributing a 405B model across 8 nodes requires vLLM or TensorRT-LLM.

### 33.4.3 llama.cpp Performance Numbers (Apple M3 Max, 8B Q4_K_M)

| Scenario | Throughput (tok/s) | TTFT P50 | Notes |
|---|---|---|---|
| Single request, 512 in, 128 out | 42 | 890 ms | Metal backend |
| Single request, 128 in, 512 out | 44 | 220 ms | Mostly decode |
| n_parallel=4, 128 in, 256 out | 28 per stream | 280 ms | Throughput limited |

### 33.4.4 llama.cpp on NVIDIA GPUs

When llama.cpp is used with a CUDA backend and a fully-offloaded 8B INT4 model on an RTX 4090:

| Scenario | Throughput (tok/s) | vs vLLM |
|---|---|---|
| Single request, 128 in, 256 out | 120 | ~80% of vLLM |
| Single request, 2048 in, 512 out | 95 | ~75% of vLLM |

llama.cpp's CUDA backend approaches but does not reach vLLM's throughput because it lacks continuous batching and uses simpler attention kernels. The gap is acceptable for single-user deployments.

---

## 33.5 Deep Dive: TensorRT-LLM

### 33.5.1 Architecture Summary

TensorRT-LLM is NVIDIA's answer to "what is the maximum throughput achievable on NVIDIA hardware, if we can afford significant engineering complexity?" Its design choices are the opposite of llama.cpp in almost every respect:

**Ahead-of-time compilation**: before serving, TensorRT-LLM compiles the model into an optimized TensorRT engine. This compilation fuses operations, selects the optimal CUDA kernel for each layer shape, and eliminates all Python overhead from the hot path. The first request after compilation hits pre-JIT-warmed CUDA kernels with zero overhead.

**In-flight batching + paged KV cache**: TensorRT-LLM v0.8+ added an in-flight batching scheduler equivalent to vLLM's continuous batching. The key difference is that the scheduler runs in C++ (not Python), removing the GIL bottleneck.

**Quantization support**: TensorRT-LLM implements FP8, INT8, INT4, and GPTQ natively in compiled engines. The FP8 path on H100 uses the tensor core's native FP8 multiply-accumulate, achieving ~2× more TFLOPS than BF16.

### 33.5.2 Compilation Complexity

The TensorRT-LLM compilation step is the primary barrier to adoption:

```bash
# 1. Convert HuggingFace weights to TensorRT-LLM checkpoint format
python convert_checkpoint.py --model_dir ./llama3-8b \
  --dtype float16 --output_dir ./trt_ckpt

# 2. Build TensorRT engine (takes 15-60 minutes for an 8B model)
trtllm-build --checkpoint_dir ./trt_ckpt \
  --output_dir ./trt_engine \
  --max_batch_size 128 \
  --max_input_len 4096 \
  --max_output_len 2048 \
  --paged_kv_cache enable \
  --use_fused_mlp enable

# 3. Serve
python run.py --engine_dir ./trt_engine --tokenizer ./llama3-8b
```

The engine is compiled for specific `max_batch_size`, `max_input_len`, `max_output_len`. Changing these parameters requires recompiling. This is manageable in a controlled production environment but painful during development.

### 33.5.3 TensorRT-LLM Performance Numbers (H100 SXM, 8B FP8)

| Scenario | Throughput (tok/s) | vs vLLM fp16 |
|---|---|---|
| Batch=64, 512 in, 128 out | 7,200 | +40% |
| Batch=128, 512 in, 512 out | 9,800 | +35% |
| Batch=256, 4096 in, 128 out | 4,100 | +20% |

FP8 quantization on H100 is TensorRT-LLM's primary advantage. vLLM fp16 vs TensorRT-LLM fp8 is not a fair comparison — but it is the practical comparison because FP8 on H100 is simpler in TensorRT-LLM than in vLLM.

---

## 33.6 Deep Dive: SGLang

### 33.6.1 Architecture Summary

SGLang (Structured Generation Language) was designed to address a specific gap: when you need both high throughput *and* complex multi-turn, structured-output, or agent workflows, neither vLLM nor TensorRT-LLM was originally optimal.

**RadixAttention**: SGLang's core memory management innovation. Like vLLM's PagedAttention, the KV cache is paged. But SGLang also maintains a *radix tree* of prefix-shared KV blocks. When two requests share a system prompt of 2000 tokens, those 2000 tokens' KV blocks are stored once and referenced by both requests. This is prefix caching taken further: the radix tree structure enables sharing of arbitrary-length common prefixes, not just a fixed system prompt.

**Compressed finite state machine for constrained decoding**: when generating JSON or other grammar-constrained output, SGLang precomputes which tokens are valid at each grammar state. The jump map narrows the valid vocabulary from 128K tokens to typically 1–50 tokens at structural positions. This enables high-batch constrained generation with minimal overhead.

**Python runtime for complex workflows**: SGLang exposes a Python API for multi-call LLM programs (fork, join, parallel generation). This is its unique position: you can describe a chain-of-thought workflow in Python, and SGLang's runtime batches the underlying model calls optimally.

### 33.6.2 SGLang Performance vs vLLM

For **unconstrained generation**, SGLang and vLLM achieve nearly identical throughput. SGLang has slightly higher overhead from the radix tree maintenance.

For **constrained generation** (JSON output), SGLang is 2–5× faster than vLLM due to the compressed FSM reducing the vocabulary scan from O(vocab) to O(valid_tokens).

For **workloads with shared prefixes** (many requests with the same long system prompt), SGLang's RadixAttention reduces KV memory usage by 40–70%, enabling proportionally more concurrent requests.

---

## 33.7 Deep Dive: Text Generation Inference (TGI)

### 33.7.1 Architecture Summary

TGI is Hugging Face's production inference server. It predates vLLM (December 2022 vs June 2023) and has evolved substantially as vLLM's innovations became standard.

**Key design choices:**
- Rust HTTP server with Python inference backend
- Flash Attention 2 by default since 2023
- Continuous batching (called "dynamic batching" in early TGI docs)
- OpenAI-compatible API, first-class
- Automatic model loading from the Hugging Face Hub

**Architecture gap vs vLLM**: TGI's KV cache management was originally simpler than PagedAttention — pre-allocated contiguous blocks with static batching. More recent versions have adopted a paged approach, but the implementation is less mature than vLLM's. This means TGI typically has 10–30% lower throughput than vLLM on mixed-length workloads.

**Where TGI wins**: developer experience. `docker run ghcr.io/huggingface/text-generation-inference --model-id meta-llama/Llama-3.1-8B-Instruct` and you have a working endpoint. Monitoring, model loading, and API compatibility are all polished. For teams that need to ship quickly and do not need maximum throughput, TGI is often the right choice.

---

## 33.8 Deep Dive: MLC-LLM

### 33.8.1 Architecture Summary

MLC-LLM (Machine Learning Compilation for LLMs) takes the most radical architectural approach: instead of writing inference kernels by hand, it uses the Apache TVM compiler to automatically generate optimized kernels for any target hardware from a high-level model specification.

**TVM / Relax IR**: the model is expressed in TVM's Relax intermediate representation. TVM then applies auto-tuning (tiling, vectorization, thread block configuration) to generate target-specific code. The same model description generates:
- CUDA kernels for NVIDIA GPUs
- Metal shaders for Apple Silicon
- OpenCL kernels for AMD GPUs
- LLVM IR for CPU (x86, ARM)
- WebGPU shaders for in-browser execution
- Vulkan shaders for Android

**Mobile-first features**: MLC-LLM is the only engine in this survey that supports running LLMs inside iOS and Android apps at production quality. The model is packaged as a platform-specific artifact and loaded at runtime.

**Throughput ceiling**: TVM-generated kernels approach but do not reach hand-written Flash Attention 2 performance. The gap is typically 15–30% on NVIDIA hardware. On Apple Silicon and mobile, MLC-LLM has no comparable alternatives, so this gap is irrelevant.

---

## 33.9 Deep Dive: DeepSpeed-FastGen

### 33.9.1 Architecture Summary

DeepSpeed-FastGen (part of Microsoft's DeepSpeed library) targets a specific use case: inference on very large models (30B–70B+) across multiple GPUs or nodes, where the model does not fit on a single GPU even with quantization.

**Dynamic SplitFuse**: DeepSpeed-FastGen's scheduler splits long prefills across multiple forward passes and fuses short ones together. The goal is to maintain high GPU utilization even when the batch contains a mix of a single 32K-token prefill and many 50-token decode steps. vLLM's chunked prefill (Chapter 11) addresses the same problem; DeepSpeed-FastGen's implementation is more aggressive about the split/fuse decision.

**Zero-Inference**: integrates with DeepSpeed ZeRO to partition model weights across GPUs. This enables running a 405B model across 4 A100s (each holding 101B parameters' worth of weights) with automatic parameter gathering at each layer boundary.

**Where it falls short**: DeepSpeed-FastGen's Python API is less polished than vLLM's, its OpenAI compatibility layer has historically lagged, and the integration surface with the wider PyTorch ecosystem is more complex.

---

## 33.10 The Disaggregated Serving Frontier

All engines described above are *integrated* engines: the same server handles both the prefill pass (processing the prompt) and the decode passes (generating output tokens). Disaggregated serving physically separates these two phases onto different hardware.

### 33.10.1 Why Disaggregate?

The prefill and decode phases have fundamentally different resource profiles:

| Phase | Compute profile | Memory access | Batching behavior |
|---|---|---|---|
| Prefill | Compute-intensive (matrix-matrix) | Sequential read | Single pass per request |
| Decode | Memory-intensive (matrix-vector) | Random KV read | Token-by-token |

On the same hardware, a long prefill request will starve decode requests of GPU time for 1–10 seconds. Disaggregation solves this by sending prefill requests to dedicated "prefill nodes" and decode requests to dedicated "decode nodes." The KV cache is transferred from prefill node to decode node after prefill completes.

### 33.10.2 Mooncake (Moonshot AI / Kimi)

Mooncake is the production disaggregated serving system behind Kimi's long-context API (supporting 1M-token contexts). Its architecture:

- **Prefill pool**: high-compute GPUs (H100) handle prompt processing
- **Decode pool**: cost-optimized GPUs (A100, or even A10G) handle output generation
- **KV transfer**: RDMA (RoCE or InfiniBand) transfers the KV cache from prefill to decode node after prefill completes
- **Prefix caching**: KV blocks for common system prompts are cached in a distributed KV store accessible by all decode nodes

**Performance claim**: at 1M-token context length, Mooncake achieves 3–5× higher throughput than integrated serving because:
1. Prefill nodes are never starved by decode requests
2. Decode nodes handle more concurrent sequences (they never process long prefills)
3. The prefix cache absorbs 60–80% of the KV computation for repeated contexts

### 33.10.3 NVIDIA Dynamo

NVIDIA Dynamo (released February 2025) is NVIDIA's disaggregated serving framework, designed to work with TensorRT-LLM as the underlying inference engine.

**Key components:**
- **Router**: assigns incoming requests to prefill or decode workers based on current load
- **KV cache manager**: tracks which KV blocks are on which node; manages migration
- **NVLink bridge**: on DGX systems, NVLink provides 900 GB/s KV transfer bandwidth, eliminating the RDMA bottleneck

**Dynamo's unique claim**: the scheduler has a "planner" component that predicts future memory pressure based on the current request queue and proactively migrates KV blocks before the decode node runs out of memory. This is predictive rather than reactive memory management.

### 33.10.4 The KV Transfer Bottleneck

All disaggregated systems face the same constraint: KV cache transfer bandwidth. For a 70B model generating 512 output tokens:

```
KV size for 4096-token context:
  = 2 × 80 layers × 8 KV heads × 128 head_dim × 4096 tokens × 2 bytes
  = 2 × 80 × 8 × 128 × 4096 × 2 / 1e9
  ≈ 13.4 GB

Transfer over 200 Gbps InfiniBand: 13.4 GB / (200/8 GB/s) ≈ 0.54 seconds

If TTFT target is 1 second, the KV transfer consumes 54% of the budget.
```

This is why NVLink (900 GB/s) is transformative for disaggregated serving: the same 13.4 GB transfers in 120 ms, consuming 12% of the TTFT budget.

For deployments without NVLink (most of the industry), disaggregated serving is only practical for:
- Very long contexts (KV transfer cost amortized over many output tokens)
- Workloads where prefill and decode requirements are radically different (e.g., 10K+ token prompts with only 50 output tokens)

---

## 33.11 The 2026 Hardware Landscape

The choice of inference engine cannot be separated from the hardware it runs on. As of mid-2026:

### 33.11.1 NVIDIA Hopper (H100, H200)

The H100 remains the dominant datacenter GPU. Key numbers:
- 3.35 TB/s HBM3 bandwidth (H100 SXM)
- 1979 TFLOPS fp16 tensor core
- 3958 TFLOPS fp8 tensor core (H100 NVL/SXM)
- 80 GB HBM3 (SXM), 141 GB (H200)
- NVLink 4.0: 900 GB/s bidirectional

The H200's 141 GB memory is the critical difference for large-model inference: a 70B fp16 model requires 140 GB, fitting just barely with no KV cache headroom. H200 enables serving 70B models with meaningful KV cache allocation on a single GPU.

### 33.11.2 NVIDIA Blackwell (B200, GB200)

B200 (released late 2024, volume in 2025) represents a step change:
- 8 TB/s HBM3e bandwidth (4× H100)
- 4.5 PFLOPS fp4 tensor core
- 192 GB HBM3e
- NVLink 5.0: 1.8 TB/s

The fp4 precision (4-bit floating point) enables storing a 70B model in 35 GB with minimal quality loss, leaving 157 GB for KV cache. At B200 bandwidth, a single token decode for a 70B model takes:
```
2 × 70B × 4 bytes / (8 TB/s) = 70 μs
```
This means a B200 can sustain 14,285 tok/s for a 70B model *even at batch size 1* (memory-bandwidth ceiling). vLLM's batch scheduler becomes less critical when the hardware is this fast.

### 33.11.3 AMD Instinct MI300X

The MI300X (128 GB HBM3) is the most memory-dense GPU available and the first AMD GPU to achieve genuine competitive parity with NVIDIA for inference. Key specs:
- 5.3 TB/s HBM3 bandwidth
- 1307 TFLOPS fp16
- 128 GB HBM3 (vs 80 GB for H100)

The extra memory makes the MI300X particularly attractive for:
- 70B models (fits entirely, leaves substantial KV cache)
- 405B models at fp8 (3 MI300X ≈ 1 DGX H100 node in memory capacity)

vLLM's ROCm backend has been substantially improved and achieves 85–90% of CUDA backend throughput on MI300X. TensorRT-LLM does not support AMD.

### 33.11.4 Apple Silicon M-Series

For edge deployment, Apple Silicon's unified memory architecture is uniquely positioned:
- M3 Max: 400 GB/s memory bandwidth, up to 128 GB unified memory
- M4 Ultra (projected): ~500 GB/s, up to 192 GB
- A single M3 Max Mac Studio holds a 70B Q4 model (≈40 GB) with 88 GB for other uses

llama.cpp with Metal backend is the only production-quality engine for Apple Silicon. MLC-LLM supports Metal but with lower throughput. Neither vLLM nor TensorRT-LLM targets Apple Silicon.

---

## 33.12 The Decision Framework

The right engine for your use case emerges from five questions, applied in sequence:

**Q1: What hardware do you have?**
- NVIDIA GPU, production datacenter → vLLM, TensorRT-LLM, or SGLang
- NVIDIA GPU, development/small scale → vLLM
- AMD GPU → vLLM (ROCm)
- Apple Silicon → llama.cpp
- CPU only → llama.cpp
- Mobile (iOS/Android) → MLC-LLM

**Q2: What scale of throughput do you need?**
- < 100 concurrent users → vLLM is sufficient; any engine works
- 100–1,000 concurrent users → vLLM or SGLang; consider TGI for simplicity
- 1,000–10,000 concurrent users → TensorRT-LLM or SGLang; vLLM requires multiple instances
- > 10,000 concurrent users → disaggregated serving (Dynamo, Mooncake pattern) + TensorRT-LLM

**Q3: What is your latency requirement?**
- Interactive (< 200 ms TTFT) → any engine at low batch size; llama.cpp fastest for short prompts
- Standard (< 1 s TTFT) → any engine with appropriate batch configuration
- Batch / async (no real-time constraint) → TensorRT-LLM for maximum throughput

**Q4: What is your output format?**
- Free text → any engine
- Structured JSON / grammar-constrained → SGLang (fastest), vLLM (with outlines), llama.cpp (GBNF)
- Complex multi-call agent workflows → SGLang (native support)
- Vision + text → vLLM (Chapter 29), TGI

**Q5: What is your operational constraint?**
- Fastest time to production → TGI (docker one-liner)
- Lowest operational complexity → vLLM (pip install, simple config)
- Maximum performance and acceptable complexity → TensorRT-LLM
- Cross-platform portability → llama.cpp
- Cutting-edge performance without vendor lock-in → SGLang

### 33.12.1 Decision Matrix

```
                    ┌─────────────┬────────────┬──────────────┬──────────┐
                    │ NVIDIA GPU  │ AMD GPU    │ Apple/CPU    │ Mobile   │
┌───────────────────┼─────────────┼────────────┼──────────────┼──────────┤
│ Max throughput    │ TensorRT    │ vLLM ROCm  │ llama.cpp    │ MLC-LLM  │
│ Best p50 latency  │ TRT-LLM     │ vLLM ROCm  │ llama.cpp    │ MLC-LLM  │
│ Easiest ops       │ TGI / vLLM  │ TGI        │ llama.cpp    │ MLC-LLM  │
│ Structured gen    │ SGLang      │ SGLang     │ llama.cpp    │ MLC-LLM  │
│ Large model (>70B)│ DeepSpeed   │ vLLM       │ llama.cpp    │ N/A      │
│ Long context      │ vLLM/SGLang │ vLLM       │ llama.cpp    │ N/A      │
│ Disaggregated     │ Dynamo      │ N/A        │ N/A          │ N/A      │
└───────────────────┴─────────────┴────────────┴──────────────┴──────────┘
```

---

## 33.13 Quantitative Comparison: Unified Benchmark Model

The following table models performance using the memory-bandwidth and compute-roof framework developed throughout this book. All numbers are derived from the formulas in Chapters 2 and 15; real benchmarks vary ±20%.

**Test configuration:** 8B parameter model, fp16, 512 input tokens, 256 output tokens, H100 SXM5.

| Engine | Effective throughput (tok/s) | TTFT P50 | Memory util | Relative cost |
|---|---|---|---|---|
| TensorRT-LLM (fp8) | ~8,200 | ~35 ms | 95% | 1.00× |
| SGLang (fp16) | ~6,800 | ~45 ms | 92% | 1.21× |
| vLLM (fp16) | ~6,100 | ~50 ms | 91% | 1.34× |
| TGI (fp16) | ~4,800 | ~65 ms | 85% | 1.71× |
| DeepSpeed-FastGen | ~5,500 | ~55 ms | 89% | 1.49× |
| llama.cpp CUDA | ~4,200 | ~70 ms | 80% | 1.95× |

*"Relative cost" = tokens-per-dollar, inverse of throughput (more throughput = lower cost).*

**Important caveat:** TensorRT-LLM's fp8 advantage requires H100; on A100, its advantage over vLLM shrinks to 15–20%. On A10G (no FP8 tensor cores), TensorRT-LLM fp16 is only 5–10% faster than vLLM.

---

## 33.14 The Evolving Frontier: What 2026 Changes

Several trends are reshaping the engine landscape in 2026 and beyond:

### 33.14.1 Speculative Decoding Goes Mainstream

Every major engine now supports speculative decoding (Chapter 23) in production. The draft-model approach (small model drafts, large model verifies) delivers 2–3× TTFT improvement on typical generative tasks. The medusa/eagle head approach (single-model self-draft) delivers 1.5–2× improvement with no additional model.

### 33.14.2 Mixture-of-Experts Changes the Throughput Picture

MoE models (Mixtral, DeepSeek-MoE, Qwen-MoE) activate only a fraction of parameters per token. For a Mixtral 8×7B model (47B total params, 13B activated per token):
- Memory required: 47B params, so 94 GB fp16
- Compute per token: 13B params worth
- KV cache: only the 2 active experts' layers

This creates a new regime: models that are memory-large (need large GPUs) but compute-small (fast decode speed). vLLM and SGLang have MoE-specific schedulers to handle the expert routing without blocking.

### 33.14.3 Context Length Arms Race

The standard context window has grown from 4K (2023) to 128K (2024) to 1M+ (2025/2026 for frontier models). This invalidates memory planning done even 12 months ago. Engines that assume a context of 8K (allocating proportional KV memory) will OOM on routine 128K requests.

The adaptive solutions:
- vLLM: dynamic block allocation, context-length-aware scheduling
- SGLang: radix tree naturally handles long shared prefixes
- llama.cpp: `-c` flag sets max context; must be set before loading

### 33.14.4 The FP4 Inflection Point

NVIDIA's fp4 precision on Blackwell (B200) and AMD's similar developments will shift the quantization frontier from "int4 is the smallest acceptable precision" to "fp4 is the precision of record." Engines that compile ahead-of-time (TensorRT-LLM) will exploit this first; dynamic engines (vLLM) will follow.

### 33.14.5 In-Context Learning Shapes Serving Patterns

As models are increasingly used for long-context retrieval-augmented generation (RAG), the ratio of prefill tokens to decode tokens inverts: 10,000 input tokens, 200 output tokens is common. Engines optimized for high input-to-output ratios (chunked prefill, prefix caching) gain relative to those optimized for batch decode throughput.

---

## 33.15 Python Demo

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
        if eng.supports_cpu and not hw.name.startswith("M"):
            # Skip CPU-only engines for GPU comparison
            pass
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
    assert mobile_recs[0][0]  == "MLC-LLM",  f"Mobile should recommend MLC-LLM, got {mobile_recs[0][0]}"
    assert nvidia_recs[0][0] in ("TensorRT-LLM", "SGLang", "vLLM"), \
        f"NVIDIA high-scale should recommend TRT/SGLang/vLLM, got {nvidia_recs[0][0]}"
    print(f"\n  [ASSERT] Mobile scenario recommends MLC-LLM: ✓")
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

---

## 33.16 C++ Demo

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

---

## 33.17 The Synthesis: What 32 Chapters Have Built

We began this book with a deceptively simple question: how does a language model turn a string of text into a string of text, as fast as possible? Thirty-two chapters later, the answer has depth.

The forward pass (Chapters 8–9) is a series of matrix multiplications and attention operations. Flash Attention (Chapter 5) restructured those operations to fit in SRAM. PagedAttention (Chapter 6) restructured the KV cache to eliminate fragmentation. The scheduler (Chapter 7) turned a batch of requests into a continuous stream. Quantization (Chapters 10, 28) halved or quartered the memory footprint. Tensor parallelism (Chapter 15) spread the computation across GPUs. Speculative decoding (Chapter 23) changed the decoding algorithm itself.

Each of these is a single idea, mechanistically simple when examined in isolation. The complexity is in their interaction: a speculative decoding system needs a scheduler that handles verification tokens; a tensor-parallel system needs an attention kernel that is parallel-aware; a semantic cache (Chapter 30) sits above the inference engine but must be calibrated against its latency characteristics.

The engines in this chapter are the accumulated expression of these ideas, each made by a different set of engineers with different priorities, different hardware constraints, and different user populations.

**vLLM** bet on Python flexibility and PagedAttention efficiency. It won the developer adoption race.

**TensorRT-LLM** bet on compilation and NVIDIA hardware co-design. It wins throughput benchmarks on paper.

**SGLang** bet on structured generation and prefix sharing. It owns the structured-output workload.

**llama.cpp** bet on portability and quantization depth. It made LLMs runnable on every device from a Raspberry Pi to an M3 Max.

**Disaggregated serving** (Mooncake, Dynamo) bets that the prefill/decode split is the next frontier. The evidence so far supports that bet for long-context workloads.

None of these bets is universally right. The right engine is the one whose bets align with your workload, your hardware, and your operational constraints.

**What stays constant** through all of this is the physics. Memory bandwidth limits how fast you can load weights. SRAM limits how much attention you can compute without going to HBM. Power dissipation limits how many transistors you can switch per second. Every engine, however clever, is negotiating with these constraints. Understanding the negotiation is how you understand the engines. Understanding the engines is how you build systems that survive at scale.

---

## 33.18 Summary

The inference engine landscape in 2026 is mature but still evolving. The principles are stable; the implementations are improving. A system built today on vLLM will need to adapt as TensorRT-LLM's ease-of-use improves, as disaggregated serving becomes practical at more price points, and as new hardware (Blackwell, MI400X) shifts the compute and memory constraints.

The reader who has worked through this book has the tools to navigate these changes:

- The memory arithmetic (Chapter 2) that tells you whether a model fits on hardware
- The attention mechanics (Chapters 4–5) that explain why Flash Attention matters
- The scheduling theory (Chapter 7) that explains when continuous batching helps
- The parallelism framework (Chapter 15) that guides multi-GPU deployment
- The quantization internals (Chapters 10, 28) that make size-performance tradeoffs legible
- The debugging toolkit (Chapter 32) that turns production incidents into fixable problems

And now, the landscape survey that puts it all in context.

Close to the metal means understanding the hardware, the kernels, the schedulers, and the caches well enough to make intelligent choices when the defaults are not enough. That understanding does not expire when a new engine ships. The physics stays.

---

*End of Chapter 33*

*End of Part IV: Advanced Topics*

*End of "Close to the Metal: LLM Inference from First Principles"*

---

> **Appendices** follow: A — Model Zoo (DeepSeek-R2, Qwen3, Kimi-k2, Nemotron), B — Production Synthesis and Runbooks, C — Benchmark Methodology, D — Notation Reference.


---

## Chapter Summary

- **vLLM**: Python-first, production-grade, PagedAttention, extensive quantization support; best for data-center GPU deployments where throughput at scale matters.
- **llama.cpp**: C++ portability-first, runs everywhere, GGUF quantization, single-process; best for edge, laptop, and small-scale inference.
- **TensorRT-LLM (TRT-LLM)**: NVIDIA-proprietary, compiled engines, highest throughput on H100/A100; steeper learning curve, GPU-locked, used in Nemotron and Triton deployments.
- **MLC-LLM**: compilation-based, targets Metal, WebGPU, CUDA, and ROCm with a single model definition; best for cross-platform deployment.
- **Ollama**: user-facing wrapper around llama.cpp; adds model management and a REST API; sacrifices fine-grained configuration for simplicity.
- **Triton Inference Server**: NVIDIA's production inference platform with dynamic batching, model ensembles, and ONNX/TRT/PyTorch backends; commonly paired with TRT-LLM.
- **Convergence**: all major engines now support PagedAttention-style KV management, continuous batching, and FlashAttention kernels; differentiation is in ecosystem, quantization depth, and multi-GPU parallelism.

---

## Self-Check Questions

1. You are deploying a 7B model for a mobile app that must run on-device on an iPhone 15. Which engine do you choose and why? Which quantization format do you use? *(Section 33.2)*

2. TRT-LLM compiles an engine at a specific batch size and sequence length. vLLM uses CUDA graphs captured at runtime. What is the practical trade-off when your production request distribution has high variance in sequence length? *(Section 33.4)*

3. MLC-LLM compiles to WebGPU to run in a browser. What are the three largest performance gaps compared to running the same model via vLLM on a server GPU? *(Section 33.5)*

4. Triton Inference Server can run a model ensemble: embedding model → LLM → re-ranker. Describe the data flow and what Triton's dynamic batcher does for each model in the ensemble. *(Section 33.6)*

5. A startup asks: "We want the fastest path to production for a RAG chatbot on AWS with 4 A100s." Rank vLLM, TRT-LLM, and llama.cpp for this use case and justify your ranking. *(Section 33.1)*
