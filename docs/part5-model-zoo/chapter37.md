# Chapter 37: Nemotron — NVIDIA's Model and TensorRT-LLM optimization

> *"TensorRT-LLM is not a framework — it is a compiler. The gap between vLLM and TRT-LLM is the gap between interpretive and compiled execution."*

---

**What you will understand after this chapter:**
- What makes Nemotron architecturally suited for NVIDIA hardware
- How TensorRT-LLM compiles a model into an execution engine
- FP8 and 2:4 structured sparsity — what the hardware does and what you gain
- Throughput comparison: TRT-LLM vs. vLLM vs. llama.cpp for the same model

**What you need first:**
- Appendix J (CUDA C++), Chapter 10 (Quantization), Chapter 15 (Multi-GPU)

---

## 37.1 What Is Nemotron?

NVIDIA Nemotron is a family of large language models trained by NVIDIA specifically to demonstrate TensorRT-LLM's optimization capabilities. The key variants:

| Model | Parameters | Architecture | Primary use |
|---|---|---|---|
| Nemotron-4 15B | 15B | Dense, GQA | General assistant |
| Nemotron-4 340B | 340B | Dense, GQA | Frontier-quality assistant |
| Llama-3.1-Nemotron-70B | 70B | Llama arch, RLHF-tuned | Instruction following |
| Llama-3.1-Nemotron-Ultra-253B | 253B | Dense, thinking mode | Complex reasoning |

Nemotron models are trained on NVIDIA hardware with NVIDIA NeMo framework, then optimized for deployment with TensorRT-LLM. They are publicly available on Hugging Face and NVIDIA NGC.

---

## 37.2 TensorRT-LLM — The Compiled Inference Engine

`[FOUNDATIONAL]` TensorRT-LLM works fundamentally differently from vLLM. Understanding this difference is essential for choosing the right engine.

### 37.2.1 Interpreted vs. Compiled Execution

```
  vLLM: Interpreted Execution
  
  Python scheduler ──→ PyTorch ops ──→ CUDA kernels (pre-compiled)
       ↑                    ↑               ↑
  Dynamic routing     Dynamic graph    Generic kernels
  2-3ms overhead      per-request      (not tuned to exact shape)
  
  TensorRT-LLM: Compiled Execution
  
  Model weights + config ──→ [COMPILE ONCE, 10-60 min] ──→ Engine file
                                                                  │
  Request ──→ C++ scheduler ──→ Engine ──→ CUDA kernels          │
                  ↑                  ↑          ↑                 │
              0.1ms overhead    Shape-specific  Auto-tuned for   │
                                  graph        exact GPU/dtype ◀─┘
```

Compilation benefits:
1. **Kernel auto-tuning**: TRT tests thousands of GEMM algorithms and picks the fastest for your exact matrix shape and GPU
2. **Layer fusion**: Multiple operations fused into one kernel (e.g., LayerNorm + matmul + activation)
3. **In-flight batching optimized C++ scheduler**: no Python overhead per request
4. **FP8 and structured sparsity**: native support that PyTorch cannot match

### 37.2.2 The TRT-LLM Compilation Pipeline

```
  TensorRT-LLM Build Pipeline
  
  Source model (HF format)
         │
         ▼
  [1] convert_checkpoint.py
      → Converts weights to TRT-LLM format
      → Applies quantization (FP8, INT8, INT4)
      → ~5-30 minutes
         │
         ▼
  [2] trtllm-build
      → Traces the model graph
      → Runs kernel auto-tuning (tries 1000s of CUDA kernels)
      → Fuses eligible layers
      → Emits compiled .engine file
      → ~10-60 minutes per GPU configuration
         │
         ▼
  [3] .engine file (NVIDIA-specific binary)
      → Can ONLY run on the exact GPU + dtype it was compiled for
      → No Python needed at runtime
         │
         ▼
  [4] tensorrt_llm/runtime or triton_inference_server
      → Load .engine, serve requests
      → C++ scheduler with in-flight batching
      → Sub-millisecond per-request overhead
```

### 37.2.3 Building a TRT-LLM Engine (Nemotron-70B)

```bash
# Step 1: Convert checkpoint
python convert_checkpoint.py \
    --model_dir /path/to/nemotron-70b \
    --output_dir ./trt_ckpt/nemotron-70b-fp8 \
    --dtype float16 \
    --use_fp8_rowwise \      # FP8 for linear layers
    --tp_size 4 \            # tensor parallelism
    --pp_size 1

# Step 2: Build engine
trtllm-build \
    --checkpoint_dir ./trt_ckpt/nemotron-70b-fp8 \
    --output_dir ./trt_engines/nemotron-70b \
    --gemm_plugin fp8 \
    --strongly_typed \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_output_len 2048 \
    --max_beam_width 1 \
    --tp_size 4

# Step 3: Serve
python run.py \
    --engine_dir ./trt_engines/nemotron-70b \
    --max_output_len 2048 \
    --input_text "Explain FP8 quantization"
```

`[COMMON TRAP]` A compiled TRT-LLM engine is **not portable**. An engine built for H100 on SM90 will not run on A100 (SM80). An engine built for TP=4 cannot be used with TP=8. You must recompile for each hardware/parallelism configuration.

---

## 37.3 FP8 on Hopper — What the Hardware Actually Does

`[DEEP DIVE]` H100's 4th-generation Tensor Cores natively support FP8 matrix multiplication. Understanding the hardware path reveals why FP8 is 2× faster than FP16.

```
  H100 Tensor Core Operation (one WGMMA instruction):
  
  FP16:  16×16×16 matrix multiply per Tensor Core per cycle
         Each operation: 16×16×16 × 2 MACs × 1 cycle = 8,192 FLOPs
  
  FP8:   16×16×16 matrix multiply per Tensor Core per cycle
         Each operation: 16×16×16 × 2 MACs × 1 cycle = 8,192 FLOPs
         BUT: can process 2× the elements (FP8 = 1 byte vs FP16 = 2 bytes)
         → Same time, 2× data → 2× effective TFLOPS
  
  Result: 989 TFLOPS (FP16/BF16) → 1,979 TFLOPS (FP8)
```

For inference, TRT-LLM implements FP8 as:
1. Weights stored in FP8 E4M3 (1 byte/param)
2. Activations quantized to FP8 before each linear layer
3. Accumulation in FP32 (within Tensor Core)
4. Output in BF16 for residual additions

### 37.3.1 FP8 Quantization Calibration

FP8's limited dynamic range (±448) requires calibration — computing per-tensor or per-channel scale factors that map BF16 values into the FP8 range:

```python
# TRT-LLM calibration (simplified)
# Run a calibration dataset through the model, record activation ranges
# Then set scale = max(abs(activations)) / 448.0

# In convert_checkpoint.py:
# --calib_dataset "cnn_dailymail"  # calibration dataset
# --calib_batches 512              # number of calibration samples
# → produces per-layer quantization scales
```

Without calibration, FP8 quantization causes significant accuracy degradation. TRT-LLM's calibration dataset for Nemotron is the dataset composition that matches the model's training distribution.

---

## 37.4 Structured Sparsity — 2:4 Sparse Tensor Cores

`[DEEP DIVE]` NVIDIA A100/H100 have hardware support for **2:4 structured sparsity**: in every group of 4 weight values, exactly 2 must be zero. The hardware skips zero multiplications, effectively doubling throughput.

```
  2:4 Sparsity Pattern
  
  Dense weight matrix:
  [0.3, 0.8, 0.1, 0.7, 0.2, 0.9, ...]
  
  2:4 sparse (2 zeros per 4 elements):
  [0.3,   0, 0.1,   0, 0.2,   0, ...]
       ↑       ↑       ↑
   zeros enforce 2:4 pattern
  
  Compressed storage: store only non-zero values + 2-bit index for each
  [0.3, 0.1, 0.2, ...]  with indices [0,2, 0,2, ...]
  
  Memory savings:   50% (half the weights are zero, stored compressed)
  Compute savings:  50% (Sparse Tensor Cores skip zero multiplications)
  
  IMPORTANT: This is structured, not random sparsity.
  Random sparsity cannot use the hardware sparse path.
```

Nemotron models are trained or fine-tuned with sparsity-aware methods that induce the 2:4 pattern during training, so the sparse version maintains >99% of dense quality.

### 37.4.1 Combined FP8 + Sparsity

```
  Combined FP8 + 2:4 sparsity on H100:
  
  FP16 dense:   989 TFLOPS  (baseline)
  FP8 dense:  1,979 TFLOPS  (2× from FP8)
  FP8 sparse: 3,958 TFLOPS  (2× from FP8 + 2× from sparsity = 4×)
  
  Memory for 70B model:
  BF16 dense:   140 GB
  FP8 dense:     70 GB
  FP8 sparse:    35 GB  (50% fewer non-zero weights stored)
```

---

## 37.5 Throughput Comparison: TRT-LLM vs. vLLM vs. llama.cpp

```
WORKED EXAMPLE 37.1 — Throughput Benchmark (Nemotron-70B, H100)
─────────────────────────────────────────────────────────────────────
Hardware: 4× H100 SXM (tensor parallel), batch=32, 512 in → 256 out

Framework        Precision    TPS/GPU    Notes
─────────────────────────────────────────────────────────────────────
TRT-LLM          FP8+sparse   ~2,800     Compiled, Tensor Core optimized
TRT-LLM          FP8 dense    ~2,200     Compiled, no sparsity
TRT-LLM          BF16         ~1,100     Compiled, BF16
vLLM             FP8          ~1,800     PagedAttention, Python overhead
vLLM             BF16         ~900       Standard vLLM
llama.cpp        Q4_K_M       ~350       Single GPU (insufficient for 70B)
                              (requires model split, loses efficiency)

Observations:
- TRT-LLM FP8+sparse: ~3.1× faster than vLLM BF16
- TRT-LLM FP8 dense: ~2.4× faster than vLLM BF16
- vLLM FP8: ~2× faster than vLLM BF16 (as expected)
- The 10-60 min compile time breaks even at ~10 hours of serving
─────────────────────────────────────────────────────────────────────
```

### 37.5.1 When to Use TRT-LLM vs. vLLM

| Factor | Choose TRT-LLM | Choose vLLM |
|---|---|---|
| Throughput requirement | >80% of hardware roof | >50% of hardware roof |
| Model update frequency | Stable (monthly) | Frequent (daily) |
| NVIDIA hardware | Required | Required |
| Compile time tolerance | 30-60 min/config | None |
| Ops team experience | Advanced | Beginner-friendly |
| Custom sampling | Supported (complex) | Easy |
| Multi-modal | Supported | Supported |

---

## 37.6 Serving Nemotron with vLLM

vLLM works with Nemotron models because they use standard Llama architecture:

```bash
# Llama-3.1-Nemotron-70B-Instruct
vllm serve nvidia/Llama-3.1-Nemotron-70B-Instruct-HF \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --quantization fp8 \        # use if H100
    --kv-cache-dtype fp8

# Nemotron-4-340B (needs 8× H100)
vllm serve nvidia/Nemotron-4-340B-Instruct \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --quantization awq          # AWQ INT4 to fit on 8× H100
```

---

## 37.7 TRT-LLM In-Flight Batching

`[PRODUCTION]`

TRT-LLM's C++ scheduler implements **in-flight batching** (also called continuous batching),
the same idea as vLLM's scheduler but without Python overhead.

```
  TRT-LLM In-Flight Batching vs. vLLM
  
  vLLM scheduler:
    Python asyncio event loop → step() → 2-3ms Python overhead per iteration
    GPU utilization: 85-90% at high load
  
  TRT-LLM C++ scheduler:
    C++ event loop → ~0.1ms overhead per iteration
    GPU utilization: 92-96% at high load
    
  Impact at batch=64, 256-token decode:
    vLLM:     256 steps × 2ms Python = 512ms overhead per batch
    TRT-LLM:  256 steps × 0.1ms C++ = 25.6ms overhead per batch
    Saving:   486ms per 64-request batch ≈ 7.5ms per request
```

At short output lengths (64–128 tokens), Python scheduler overhead is a significant
fraction of total latency. TRT-LLM's sub-millisecond scheduler makes it particularly
effective for chatbot use cases with short responses.

### 37.7.1 Triton Inference Server Integration

The standard production deployment for TRT-LLM is via NVIDIA Triton Inference Server:

```bash
# Launch Triton with TRT-LLM backend
docker run --gpus all \
    -v /path/to/engines:/opt/tritonserver/model_repository \
    nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3 \
    tritonserver \
    --model-repository=/opt/tritonserver/model_repository

# model_repository structure:
# nemotron_70b/
#   config.pbtxt         ← model configuration
#   1/                   ← version directory
#     model.py           ← TRT-LLM backend script
#     engines/           ← compiled .engine files
```

**config.pbtxt highlights:**

```protobuf
name: "nemotron_70b"
backend: "tensorrtllm"
max_batch_size: 64

parameters {
  key: "engine_dir"
  value: { string_value: "/opt/tritonserver/model_repository/nemotron_70b/1/engines" }
}
parameters {
  key: "max_tokens_in_paged_kv_cache"
  value: { string_value: "20000" }
}
parameters {
  key: "batch_scheduler_policy"
  value: { string_value: "guaranteed_no_evict" }
}
```

### 37.7.2 Engine Versioning and Rollback

`[OPERATIONS]`

TRT-LLM engines must be recompiled when:
- The model weights change (new fine-tune or post-training update)
- You change precision (BF16 → FP8)
- You change tensor parallelism degree
- You upgrade TRT-LLM version (engine format can change between major versions)

**Production versioning strategy:**

```
  Engine naming convention:
  nemotron-70b_fp8_tp4_maxbatch64_maxin4096_maxout2048_h100_v2.3.0.engine
                ↑    ↑  ↑        ↑         ↑          ↑    ↑
               prec  TP  batch   inlen      outlen     GPU   TRT version

  Deployment workflow:
  1. Build new engine in staging environment (30-60 min)
  2. Run benchmark validation (Chapter 17 scripts)
  3. A/B test on 5% of traffic via load balancer
  4. Promote to 100% if metrics hold for 30 min
  5. Keep old engine for 24h (rollback via load balancer redirect)
```

---

## 37.8 Nemotron-Ultra and Thinking Mode

NVIDIA's Llama-3.1-Nemotron-Ultra-253B (released March 2025) adds a thinking mode
similar to DeepSeek-R1 and Qwen3 — the model can be prompted to reason before answering.

```python
# Thinking mode: include system prompt
messages_think = [
    {
        "role": "system",
        "content": "detailed thinking on"  # enables thinking mode
    },
    {
        "role": "user",
        "content": "Prove that √2 is irrational."
    }
]

# Non-thinking mode: standard instruction following
messages_fast = [
    {
        "role": "system",
        "content": "detailed thinking off"  # disables thinking
    },
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]
```

**Serving implications for thinking mode:**
The 253B model in thinking mode generates 3,000–15,000 token reasoning traces.
At 4× H100 (TP=4) with FP8:
- Non-thinking output: ~800 tok/s aggregate throughput
- Thinking output: ~800 tok/s aggregate throughput
  (throughput is the same per token — but 15K output tokens cost 15× more)

**Practical advice:** Use thinking mode only for tasks where quality matters more than
latency and cost. Route short factual queries to the non-thinking mode via the
`detailed thinking off` system prompt. This is the serving policy optimization from
Chapter 25 applied directly to a model's built-in capability.

---

## 37.9 Chapter Summary

TensorRT-LLM's compilation approach yields 2–4× higher throughput than vLLM at the cost of 30–60 minute compile times and hardware/configuration lock-in. FP8 doubles compute over BF16 on H100; 2:4 structured sparsity doubles it again. Nemotron models are optimized for this full stack. Use TRT-LLM when throughput is the primary metric and the model/hardware configuration is stable.

### Where We Go Next

Chapter 38 is the final chapter of the book — the Production Synthesis. We revisit the LinkedIn scenario from Chapter 1 and trace the path from $1.2M/month to $108K/month by applying every technique across all 37 chapters.


---

## Chapter Summary

- **Nemotron and NVIDIA's ecosystem**: Nemotron models are designed to run on TensorRT-LLM (TRT-LLM) with Triton Inference Server, leveraging NVIDIA-proprietary optimizations for H100/H200.
- **TRT-LLM engine compilation**: TRT-LLM compiles a model into a static CUDA engine at specific batch sizes and sequence lengths; serving requires building an engine file before deployment.
- **FP8 quantization with TRT-LLM**: `trtllm-build` with `--gemm-plugin fp8` generates FP8 weight + FP8 activation kernels for H100 Tensor Cores; typical speedup is 1.8–2.3× over BF16.
- **2:4 structured sparsity**: H100 supports 2:4 weight sparsity natively (2 non-zero values per 4-element group); combined with FP8, this can achieve 3–4× throughput vs FP16 dense.
- **Triton Inference Server integration**: TRT-LLM engines are served via `tritonserver` with `tensorrtllm_backend`; Triton handles batching, multi-model ensembles, and gRPC/REST APIs.
- **Nemotron reward models**: Nemotron-4-340B-Reward is used for RLHF reward scoring; it runs as a prefill-only endpoint with a scalar linear head.
- **Engine shape constraints**: TRT-LLM engines compiled for `maxBatchSize=64, maxInputLen=2048` fail for batch size 65 or input length 2049; padding or runtime switching is required.

---

## Self-Check Questions

1. You build a TRT-LLM engine with `--max-batch-size 32 --max-input-len 2048 --max-output-len 512`. A request arrives with input length 2 200. What happens? *(Section 37.2)*

2. FP8 on H100 delivers 3 958 TOPS vs 1 979 TFLOPS for BF16. A Nemotron-8B model does 8 × 10¹⁰ FLOPs per forward pass at batch 1. Compute the theoretical minimum decode latency in FP8 vs BF16. *(Section 37.3)*

3. 2:4 structured sparsity prunes 50% of weights. After sparsification, FP8 quantization is applied. Compute the memory footprint of a 70B model after both transformations, starting from FP16. *(Section 37.4)*

4. Triton Inference Server routes requests to a TRT-LLM backend ensemble: tokeniser → LLM → de-tokeniser. The LLM backend has `dynamic_batching` enabled with `preferred_batch_size [8, 16, 32]`. A burst of 20 requests arrives. Describe the batching behavior. *(Section 37.5)*

5. Nemotron-4-340B-Reward runs as a prefill-only endpoint. At batch size 32 with average sequence length 1 024, and assuming H100 FP8 at 3 958 TOPS, estimate the reward scoring throughput in requests/second. *(Section 37.6)*
