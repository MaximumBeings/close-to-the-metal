# Chapter 1: Two Engines, One Problem

> *"The most common mistake in inference engineering is treating it like training with the batch size turned down."*

---

**What you will understand by the end of this chapter:**

- Why generating text one token at a time creates a fundamentally different systems problem than training
- Why a GPU is almost always idle during inference, and what that means for performance
- What vLLM is, what llama.cpp is, and the specific trade-off each one makes
- How to run a complete inference request using both engines
- How to decide which engine to use for a given situation

**What you need to know first:**

- You should be comfortable reading Python code
- You should know that a neural network takes an input and produces an output (no deeper ML knowledge needed)
- You should know roughly what a GPU is — a chip designed for parallel computation (no GPU programming experience needed)

---

## 1.1 The Problem: One Token at a Time

`[FOUNDATIONAL]`

### Intuition

Imagine you are dictating a letter to a secretary. You can only give them one word at a time, and they cannot write word number 5 until they have already written words 1 through 4. There is no way to speed this up by hiring more secretaries — the words must come out in order, because each word depends on all the words before it. That dependency chain is the core constraint of language model inference.

Now imagine a different task: grading 100 student essays. You can hand one essay to each of 100 graders and have them all work at the same time. There is no dependency between essays. This is closer to how neural network training works — you process many examples simultaneously.

These two tasks — sequential dictation and parallel grading — capture the essential difference between inference and training.

### Precise Definition

A **language model** generates text by predicting one token at a time. A **token** is a chunk of text — roughly 3–4 characters on average in English. The word "inference" is one token. The phrase "LLM inference" is two tokens.

> **Background: What is a token?**
> Tokenization is the process of splitting text into pieces a model can process. Modern models use a learned vocabulary of ~32,000 to ~128,000 tokens. Common words are single tokens. Rare words split into multiple tokens. Numbers often tokenize one digit at a time. You do not need to understand the tokenization algorithm for this chapter — just know that every piece of text becomes a sequence of integer IDs, and the model works with those integers.

To generate the sentence "The cat sat", the model runs three times:

- **Run 1:** Given "The", predict the next token → outputs "cat"
- **Run 2:** Given "The cat", predict the next token → outputs "sat"
- **Run 3:** Given "The cat sat", predict the next token → outputs a stop signal

This is called **autoregressive generation**. Each run is a complete forward pass through the entire neural network. Run 2 cannot start until Run 1 finishes, because Run 2 needs "cat" as part of its input.

### Worked Example: Counting Forward Passes

```
WORKED EXAMPLE 1.1 — Forward passes required to generate a response
─────────────────────────────────────────────────────────────────────
Given:
  User prompt:         "What is the capital of France?"  →  8 tokens
  Model response:      "The capital of France is Paris." →  7 tokens
  Total forward passes needed: ?

Step 1: Count the prompt processing pass.
  The model reads all 8 prompt tokens in ONE forward pass.
  This is called the "prefill" pass.
  Prefill passes = 1

Step 2: Count the generation passes.
  The model generates 7 tokens, one per pass.
  Generation passes = 7

Step 3: Total.
  Total forward passes = 1 (prefill) + 7 (generation) = 8

Now compare to training:
  During training, a sequence of 15 tokens (8 prompt + 7 response)
  would be processed in exactly 1 forward pass + 1 backward pass.
  All 15 predictions happen in parallel using "teacher forcing."

Final answer:
  Inference requires 8 forward passes.
  Training requires 1 forward pass for the same sequence.
  Inference is 8× more forward passes for this example.
  For a 200-token response: 1 + 200 = 201 forward passes vs. 1 in training.
─────────────────────────────────────────────────────────────────────
```

### ASCII Diagram

```
INFERENCE — Sequential, one token per pass
──────────────────────────────────────────────────────────────────

  Input grows with each step:

  Pass 1 (Prefill):
  ┌─────────────────────────┐     ┌──────────┐
  │ "What is the capital of │ ──▶ │  MODEL   │ ──▶  "The"
  │  France?"  [8 tokens]   │     └──────────┘
  └─────────────────────────┘

  Pass 2 (Decode step 1):
  ┌─────────────────────────────┐  ┌──────────┐
  │ "What is...France?" + "The" │─▶│  MODEL   │──▶  "capital"
  │  [8 + 1 = 9 tokens]         │  └──────────┘
  └─────────────────────────────┘

  Pass 3 (Decode step 2):
  ┌──────────────────────────────────┐  ┌──────────┐
  │ "What is...France?" + "The      │─▶│  MODEL   │──▶  "of"
  │  capital"  [10 tokens]           │  └──────────┘
  └──────────────────────────────────┘

  ...and so on until EOS (end-of-sequence) token.

  Each pass MUST wait for the previous pass to finish.
  ↑ This sequential dependency is the root of all inference complexity.

──────────────────────────────────────────────────────────────────

TRAINING — Parallel, all tokens in one pass
──────────────────────────────────────────────────────────────────

  ┌────────────────────────────────────────────────────────────┐
  │ "What is the capital of France? The capital of France is"  │
  │                  [all 15 tokens at once]                    │
  └────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        ┌──────────┐
                        │  MODEL   │
                        └──────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           "is"            "capital"        "Paris"
        (pos 1→2)          (pos 8→9)       (pos 14→15)
         [all 15 predictions computed simultaneously]

──────────────────────────────────────────────────────────────────
```

`[COMMON TRAP]` A very common mistake is to think: "I have a fast GPU, so inference will be fast." GPU speed is not the primary bottleneck during inference. The sequential dependency above means that raw compute power is mostly wasted. We will see exactly why in Section 1.2.

---

## 1.2 Why the GPU Sits Idle: Memory Bandwidth vs. Compute

`[FOUNDATIONAL]`

### Intuition

Think of a GPU as a factory with thousands of workers (compute units) and a warehouse (memory). The workers can assemble things incredibly fast, but they can only work on materials that have been brought out of the warehouse. The warehouse has a loading dock (memory bandwidth) that can only move so much material per second.

If the workers are so fast that they finish their current materials and have to stand around waiting for the next truck from the warehouse — the factory is **memory-bandwidth-bound**. The bottleneck is the loading dock, not the workers.

If the workers are slow enough that the loading dock keeps up — the factory is **compute-bound**. The bottleneck is the workers.

During LLM inference with a single user, the GPU's "workers" finish their job so fast that they spend most of their time waiting for new model weights to be loaded from memory. The GPU is memory-bandwidth-bound, often severely so.

### Background: Two Key GPU Specifications

> Every GPU has two performance numbers that matter for inference:
>
> **Peak memory bandwidth** — how many bytes per second can be moved between the GPU's memory (called HBM, for High Bandwidth Memory, on data-center GPUs) and the compute units. Measured in terabytes per second (TB/s).
>
> **Peak compute throughput** — how many floating-point operations (FLOPs) per second the GPU can perform. Measured in teraFLOPs per second (TFLOPS).
>
> These two numbers are for completely different things. Bandwidth is about *moving data*. Throughput is about *doing math*. Both limits exist simultaneously, and whichever one you hit first is your bottleneck.

### The Roofline Model

`[FOUNDATIONAL]`

The **roofline model** is a simple framework for predicting which of the two limits — memory bandwidth or compute — constrains a given operation. It uses one key quantity called **arithmetic intensity**.

**Arithmetic intensity** = FLOPs performed ÷ bytes read from memory

If an operation does a lot of math per byte it reads, it is compute-bound. If it does very little math per byte, it is memory-bound. The crossover point is called the **ridge point**.

### Worked Example: Computing the Ridge Point for an H100 GPU

```
WORKED EXAMPLE 1.2 — Ridge point of an NVIDIA H100 SXM GPU
─────────────────────────────────────────────────────────────────────
Given (from NVIDIA's official H100 SXM5 spec sheet):
  Peak memory bandwidth:        3.35 TB/s  =  3,350 GB/s
  Peak compute (BF16 Tensor, dense):  989 TFLOPS  =  989 × 10¹² FLOPs/s
  (Note: 1,979 TFLOPS is the structured-sparsity figure; dense BF16 = 989 TFLOPS)

Step 1: Convert units so they match.
  Bandwidth:  3,350 GB/s  =  3,350 × 10⁹ bytes/s
  Compute:    989 TFLOPS  =  989 × 10¹² FLOPs/s

Step 2: Compute the ridge point.
  Ridge = peak_compute / peak_bandwidth
        = (989 × 10¹² FLOPs/s) / (3.35 × 10¹² bytes/s)
        = 989 / 3.35  FLOPs/byte
        ≈ 295  FLOPs/byte

Step 3: Interpret.
  Any operation with arithmetic intensity > 295 FLOPs/byte
  → compute-bound (GPU math units are the bottleneck)

  Any operation with arithmetic intensity < 295 FLOPs/byte
  → memory-bandwidth-bound (HBM loading dock is the bottleneck)

Final answer: The ridge point for an H100 SXM (dense BF16) is ~295 FLOPs/byte.
─────────────────────────────────────────────────────────────────────
```

### Worked Example: Arithmetic Intensity of LLM Decode

Now let us compute the arithmetic intensity of one decode step for a 7 billion parameter model with a single user (batch size = 1).

```
WORKED EXAMPLE 1.3 — Arithmetic intensity of a single-user decode step
─────────────────────────────────────────────────────────────────────
Given:
  Model:            Llama 3 8B (approximately 7 billion active parameters)
  Precision:        FP16 (each parameter stored as a 16-bit float = 2 bytes)
  Batch size:       1 user generating 1 token

Step 1: How many bytes must be read from memory per decode step?
  Each decode step reads essentially all model weights once.
  (The math: each weight participates in one multiply-add per token.)

  Model parameters:    7 × 10⁹
  Bytes per parameter: 2 (FP16)
  Total bytes read:    7 × 10⁹ × 2 = 14 × 10⁹ bytes = 14 GB

Step 2: How many FLOPs are performed per decode step?
  For a transformer, each parameter participates in roughly 2 FLOPs
  (one multiply, one add) per token per forward pass.

  FLOPs = 2 × parameters × batch_size × tokens_per_step
        = 2 × 7 × 10⁹ × 1 × 1
        = 14 × 10⁹ FLOPs = 14 GFLOPs

Step 3: Compute arithmetic intensity.
  Arithmetic intensity = FLOPs / bytes
                       = 14 × 10⁹ FLOPs / 14 × 10⁹ bytes
                       = 1.0 FLOPs/byte

Step 4: Compare to the ridge point.
  Ridge point (H100): ~295 FLOPs/byte
  Our operation:        ~1  FLOPs/byte

  Our arithmetic intensity is 295× BELOW the ridge point.

Final answer:
  Decode at batch=1 is deeply memory-bandwidth-bound.
  The H100's 989 TFLOPS of compute are almost entirely wasted.
  The bottleneck is loading 14 GB of weights from HBM per token.
─────────────────────────────────────────────────────────────────────
```

This is the most important number in this book. Write it down: **1 FLOP/byte**. That is what single-user decode looks like. The H100 needs ~295 to be compute-bound. We are 295× away.

### ASCII Diagram: The Roofline

```
  ROOFLINE MODEL — H100 SXM

  Achieved
  Throughput
  (TFLOPS)
     │
 989 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╔══════════════════════
     │                              ╔╝  COMPUTE CEILING
     │                           ╔══╝     (989 TFLOPS)
     │                        ╔══╝
     │   MEMORY-BOUND       ╔══╝
     │   REGION           ╔══╝         COMPUTE-BOUND
     │                 ╔══╝            REGION
     │              ╔══╝
     │           ╔══╝
     │        ╔══╝     ← memory bandwidth slope
     │     ╔══╝          (3.35 TB/s)
     │  ╔══╝
  0  └──┴──────────────────┴──────────────────────────────▶
     0                    295                             Arithmetic
                       (ridge)                           Intensity
                                                        (FLOPs/byte)

  Operating points:
  ──────────────────────────────────────────────────────────
  [D] Decode, batch=1    ≈ 1  FLOPs/byte  ← far left, memory-bound
  [P] Prefill, 2048 tok  ≈ 150 FLOPs/byte ← closer to ridge
  [T] Training, large    ≈ 600 FLOPs/byte ← at or above ridge

  [D] is here:
    ↓
  ──[D]───────────────────[P]─────────────────[T]────────── FLOPs/byte
    1                     150                 600

  Every technique in this book moves [D] rightward.
```

### What Does "Memory-Bound at 1 FLOP/byte" Mean in Practice?

Let us compute what the H100 can actually achieve given this arithmetic intensity.

```
WORKED EXAMPLE 1.4 — Achievable throughput at 1 FLOPs/byte
─────────────────────────────────────────────────────────────────────
Given:
  Arithmetic intensity: 1 FLOPs/byte
  H100 memory bandwidth: 3.35 TB/s = 3.35 × 10¹² bytes/s

Step 1: Compute achievable FLOPs/s.
  Achievable = intensity × bandwidth
             = 1 FLOPs/byte × 3.35 × 10¹² bytes/s
             = 3.35 × 10¹² FLOPs/s
             = 3.35 TFLOPS

Step 2: Compare to peak.
  We achieve:  3.35 TFLOPS
  Peak is:     989 TFLOPS
  Utilization: 3.35 / 989 ≈ 0.34%

Step 3: What does this mean for token speed?
  Each decode step does ≈ 14 GFLOPs (from Worked Example 1.3).
  Time per step = FLOPs / achievable_throughput
               = 14 × 10⁹ FLOPs / 3.35 × 10¹² FLOPs/s
               = 0.00418 seconds
               ≈ 4.2 ms per token

  Measured reality on H100 for Llama 3 8B at batch=1: ≈ 9 ms/token.
  (The gap comes from overhead: kernel launch, sampling, memory latency.)

Final answer:
  At batch=1, the H100 uses less than 0.2% of its compute capacity.
  The GPU is essentially idle, waiting for memory to deliver weights.
─────────────────────────────────────────────────────────────────────
```

`[COMMON TRAP]` Students often read "0.17% GPU utilization" and think the system is broken or misconfigured. It is not. This is the correct behavior for single-user inference. `nvidia-smi` will show low utilization even on a perfectly tuned deployment with one user. You need many concurrent users — that is what batching is for — to drive utilization up. This is exactly what vLLM's continuous batching solves, as we will see in Chapter 3.

---

## 1.3 How Batching Helps: A Preview

`[FOUNDATIONAL]`

We just showed that batch size 1 gives ~1 FLOP/byte. What happens if we run 32 users simultaneously?

```
WORKED EXAMPLE 1.5 — Arithmetic intensity with batch size 32
─────────────────────────────────────────────────────────────────────
Given:
  Model:       Llama 3 8B, FP16
  Batch size:  32 users each generating 1 token simultaneously

Step 1: How many bytes are read from memory?
  The model weights are loaded ONCE per forward pass, regardless of
  how many users are in the batch. The weights are shared.

  Bytes read = 7 × 10⁹ parameters × 2 bytes = 14 GB (same as batch=1)

Step 2: How many FLOPs are performed?
  Each of the 32 users generates one token, so:

  FLOPs = 2 × 7 × 10⁹ × 32 × 1
        = 448 × 10⁹ FLOPs = 448 GFLOPs

Step 3: Arithmetic intensity.
  Intensity = 448 GFLOPs / 14 GB = 32 FLOPs/byte

Step 4: Compare to ridge point.
  At batch=1:   ~1  FLOPs/byte  (deeply memory-bound)
  At batch=32:  ~32 FLOPs/byte  (still memory-bound, but 32× better)
  At batch=295: ~295 FLOPs/byte (at the ridge — fully utilizing GPU)

Final answer:
  Arithmetic intensity scales linearly with batch size.
  Batching more users together makes the GPU dramatically more efficient.
  This is the fundamental reason vLLM focuses on concurrent user serving.
─────────────────────────────────────────────────────────────────────
```

```
  ARITHMETIC INTENSITY vs. BATCH SIZE (Llama 3 8B, FP16, H100)

  FLOPs/byte
     │
 295 ┤- - - - - - - - - - - - - - - - - - - -[ridge]- - - - - -
     │                                           ↑
     │                                     fully utilizing GPU
  32 ┤ · · · · · · · · · · · · · [batch=32] ·
     │
   4 ┤ · · · [batch=4] ·
     │
   1 ┤ [batch=1]
     │
   0 └───────────────────────────────────────────────────────▶
     0       4        32                       295       Batch size
                                            (≈ ridge)

  Key insight: the weights cost the same bytes whether you serve
  1 user or 32 users. More users = same memory cost, more compute.
  This is why batching recovers GPU utilization.
```

This insight — that weights are loaded once but compute scales with batch size — is the single most important idea for understanding why high-throughput inference servers exist. Everything vLLM does is in service of keeping the batch size as large as possible.

---

## 1.4 Introducing vLLM

`[FOUNDATIONAL]`

### What vLLM Is

vLLM (pronounced "vee-L-L-M") is an open-source inference server released by researchers at UC Berkeley in June 2023. The key innovation it introduced is called **PagedAttention** — a way of managing the GPU's memory far more efficiently than previous systems. We will spend all of Chapter 5 on PagedAttention. For now, just know that it eliminated a major source of wasted GPU memory, allowing far more concurrent users on the same hardware.

> **What does "inference server" mean?**
> An inference server is a program that listens for requests (via HTTP, like a web server), receives a prompt from a user, runs the model, and sends back a response. It is designed to handle many users at once, keep the GPU busy, and manage all the complexity of batching requests together. vLLM is one such server.

### vLLM's Stack — From API Request to GPU Kernel

```
  vLLM SYSTEM STACK
  ══════════════════════════════════════════════════════
  
  USER / CLIENT
  ┌──────────────────────────────────────────────────┐
  │  POST /v1/completions                            │
  │  {"model": "llama3", "prompt": "Hello..."}       │
  └──────────────────────────────┬───────────────────┘
                                 │  HTTP (OpenAI-compatible API)
                                 ▼
  ┌──────────────────────────────────────────────────┐
  │           FastAPI HTTP Server                    │
  │   receives requests, validates, enqueues         │
  └──────────────────────────────┬───────────────────┘
                                 │  Python async queue
                                 ▼
  ┌──────────────────────────────────────────────────┐
  │           AsyncLLMEngine  (Python)               │
  │                                                  │
  │   ┌────────────────┐   ┌──────────────────────┐  │
  │   │   Scheduler    │   │   BlockManager       │  │
  │   │ decides which  │   │  manages KV cache    │  │
  │   │ requests run   │   │  memory pages (HBM)  │  │
  │   └────────┬───────┘   └──────────────────────┘  │
  └────────────┼─────────────────────────────────────┘
               │  batched token IDs + KV block addresses
               ▼
  ┌──────────────────────────────────────────────────┐
  │           Worker Process  (Python + PyTorch)     │
  │   holds model weights, runs forward passes       │
  └──────────────────────────┬───────────────────────┘
                             │  CUDA kernel calls
                             ▼
  ┌──────────────────────────────────────────────────┐
  │  CUDA Kernels (C++ / CUDA)                       │
  │  FlashAttention-2,  cuBLAS GEMM,  custom ops     │
  └──────────────────────────┬───────────────────────┘
                             │  raw compute
                             ▼
  ┌──────────────────────────────────────────────────┐
  │  NVIDIA GPU  (H100 / A100 / RTX 4090 / ...)      │
  │  HBM holds: weights + KV cache blocks            │
  └──────────────────────────────────────────────────┘

  ══════════════════════════════════════════════════════
```

### What vLLM Optimizes For

vLLM makes one central bet: **many users sharing one GPU is the normal case.** Every design decision follows from that bet.

- It uses continuous batching so new users can be added to an in-progress batch without waiting for the current batch to finish.
- It uses PagedAttention so GPU memory is never wasted on unused sequence slots.
- It uses CUDA graph capture to eliminate Python overhead on the hot path.
- It exposes an OpenAI-compatible API so it can be a drop-in replacement for any service already using the OpenAI SDK.

**What vLLM requires:** A CUDA-capable NVIDIA GPU (or AMD ROCm GPU). It will not run on a CPU alone in production, and it will not run on Apple Silicon natively.

**Typical cold-start time:** 30 seconds for a 7B model, up to 140 seconds for a 70B model (mostly weight loading from disk).

---

## 1.5 Introducing llama.cpp

`[FOUNDATIONAL]`

### What llama.cpp Is

llama.cpp is an open-source C++ library created by Georgi Gerganov, first released in March 2023 — three months before vLLM. It began as a personal experiment to run Meta's LLaMA model on a MacBook. It has since become one of the most popular AI repositories on GitHub and the foundation of almost every local LLM tool (Ollama, LM Studio, Jan, and more).

llama.cpp is not a server by default — it is a C++ library with a public C API. You can build a server on top of it (llama-server is included), but at its core it is a library you link against and call.

> **What is a C API?**
> C is a programming language. A "C API" (Application Programming Interface) is a set of functions you can call from C or C++ code — and from many other languages via bindings. llama.cpp exposes functions like `llama_model_load_from_file()`, `llama_decode()`, `llama_sampler_sample()` that any program can call. Python wrappers (llama-cpp-python) expose these same functions to Python.

### The GGUF Format

Where vLLM uses HuggingFace's standard weight format (safetensors), llama.cpp uses **GGUF** — its own model format. GGUF stands for "GGML Unified Format." It is a single binary file containing:

- All model weights (optionally quantized to INT4, INT5, INT8, etc.)
- All metadata: vocabulary, tokenizer rules, model architecture hyperparameters
- Everything needed to load and run the model without any other files

This self-contained design means you download one file and you are done.

### llama.cpp's Stack — From GGUF File to CPU/GPU Output

```
  llama.cpp SYSTEM STACK
  ══════════════════════════════════════════════════════

  USER / CLIENT
  ┌──────────────────────────────────────────────────┐
  │  ./llama-cli -m model.gguf -p "Hello..."         │
  │  — or —                                          │
  │  POST http://localhost:8080/completion            │
  └──────────────────────────┬───────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────┐
  │  llama-cli / llama-server  (C++ executables)     │
  │  thin wrappers around the core C API             │
  └──────────────────────────┬───────────────────────┘
                             │  calls C API functions
                             ▼
  ┌──────────────────────────────────────────────────┐
  │  llama.cpp  C API  (libllama)                    │
  │                                                  │
  │  llama_model_load_from_file()  ← loads GGUF      │
  │  llama_new_context_with_model() ← allocs KV buf  │
  │  llama_decode()                ← runs fwd pass   │
  │  llama_sampler_sample()        ← picks next tok  │
  └──────────────────────────┬───────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────┐
  │  GGML  (tensor library, C)                       │
  │  builds a compute graph (ggml_cgraph),           │
  │  dispatches to the right backend                 │
  └──────────────────────────┬───────────────────────┘
                             │  backend dispatch
              ┌──────────────┼───────────────┐
              ▼              ▼               ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  CPU backend │  │ CUDA backend │  │Metal backend │
  │  AVX2 / NEON │  │ NVIDIA GPU   │  │ Apple Silicon│
  └──────────────┘  └──────────────┘  └──────────────┘

  ══════════════════════════════════════════════════════
```

### What llama.cpp Optimizes For

llama.cpp makes the opposite bet from vLLM: **running on any hardware with minimal setup is the normal case.** Every design decision follows from that bet.

- It uses memory-mapped GGUF files so models load in seconds (the OS pages in weights lazily).
- It uses INT4/INT5/INT8 quantization so a 70B model fits in 40 GB of RAM.
- It has no external dependencies for CPU inference — not even a GPU driver.
- It runs on Android phones, Raspberry Pi 5, macOS, Windows, and data-center servers equally.

**What llama.cpp does NOT require:** A GPU. A specific operating system. Python. Any cloud service.

**Typical cold-start time:** 1–5 seconds for any model size, because weights are memory-mapped rather than copied into RAM.

---

## 1.6 Side-by-Side: The Same Pipeline, Different Designs

Both engines implement the same logical pipeline for every inference request. The steps are identical. The implementations are completely different.

```
  THE SHARED INFERENCE PIPELINE

  Step 1: TOKENIZE
  ┌─────────────────────────────────────────────────────────┐
  │  "What is the capital of France?"                       │
  │        │                                               │
  │        ▼                                               │
  │  [1867, 374, 279, 6864, 315, 9822, 30]  ← token IDs   │
  └─────────────────────────────────────────────────────────┘
  vLLM:     uses HuggingFace tokenizer (Python, runs on CPU)
  llama.cpp: uses built-in tokenizer from GGUF metadata (C, CPU)

  Step 2: PREFILL (process the prompt)
  ┌─────────────────────────────────────────────────────────┐
  │  All prompt token IDs → ONE forward pass through model  │
  │  Result: K and V tensors saved to KV cache              │
  │          logits for the last token position             │
  └─────────────────────────────────────────────────────────┘
  vLLM:     PyTorch + FlashAttention-2 on CUDA
  llama.cpp: GGML compute graph on CPU/CUDA/Metal

  Step 3: DECODE LOOP (generate response tokens)
  ┌─────────────────────────────────────────────────────────┐
  │  Repeat until EOS or max_tokens:                        │
  │    a) Sample next token from logits                     │
  │    b) Append token to sequence                          │
  │    c) Run forward pass with new token (reads KV cache)  │
  │    d) Get new logits                                    │
  └─────────────────────────────────────────────────────────┘
  vLLM:     managed by Scheduler + Worker, async Python
  llama.cpp: explicit loop in user code, synchronous C++

  Step 4: DETOKENIZE
  ┌─────────────────────────────────────────────────────────┐
  │  [791, 6864, 315, 9822, 374, 12366, 13]                │
  │        │                                               │
  │        ▼                                               │
  │  "The capital of France is Paris."                      │
  └─────────────────────────────────────────────────────────┘
  vLLM:     HuggingFace tokenizer (Python)
  llama.cpp: built-in vocab from GGUF (C)
```

The design difference shows up most clearly in Step 3. vLLM's Scheduler decides which users run each step, manages memory allocation, and handles preemption automatically. In llama.cpp, your C++ code is the scheduler — you call `llama_decode()` explicitly for each user in whatever order you choose.

This is not a weakness of llama.cpp — it is a deliberate choice. For a single user on a laptop, there is nothing to schedule. For a production server with hundreds of users, vLLM's scheduler is essential.

---

## 1.7 The Full Comparison Table

```
  vLLM vs. llama.cpp — Design Decisions Side by Side
  ════════════════════════════════════════════════════════════════════

  DIMENSION            │ vLLM                    │ llama.cpp
  ─────────────────────┼─────────────────────────┼────────────────────
  Primary language     │ Python (+ CUDA C++)      │ C++ (no Python req.)
  Model format         │ Safetensors / HF Hub     │ GGUF (self-contained)
  KV cache design      │ PagedAttention (virtual  │ Fixed contiguous
                       │  pages, dynamic alloc)   │  ring buffer
  Batching             │ Continuous (many users   │ Sequential / small
                       │  in parallel, dynamic)   │  batch
  GPU requirement      │ Yes (CUDA or ROCm)       │ Optional
  Quantization formats │ GPTQ, AWQ, FP8           │ Q2_K through Q8_0
  Primary deployment   │ Data center / cloud      │ Laptop / edge / local
  Cold start time      │ 30–140 seconds           │ 1–5 seconds (mmap)
  Scheduling           │ Automatic (Scheduler)    │ Manual (your code)
  API style            │ OpenAI-compatible REST   │ C API / REST / Python
  LoRA hot-swap        │ Yes (per-request)        │ Limited (load-time)
  Multi-GPU            │ Yes (tensor parallel)    │ Experimental (RPC)
  Apple Silicon        │ No native support        │ Yes (Metal backend)
  CPU-only             │ No (for production)      │ Yes (full feature set)
  ─────────────────────┴─────────────────────────┴────────────────────

  Neither is "better." They solve different halves of the deployment
  spectrum. This book teaches both because real engineers encounter both.

  ════════════════════════════════════════════════════════════════════
```

---

## 1.8 Hello World — Python with vLLM

The following Python code runs a complete inference request using vLLM. Every line has a comment explaining what is happening and why.

```python
# hello_vllm.py
# Requirements: pip install vllm
# Hardware: any CUDA GPU with >= 8 GB VRAM

from vllm import LLM, SamplingParams
import time

# ── Step 1: Load the model ───────────────────────────────────────────
# LLM() triggers the full startup sequence:
#   a) Download weights from HuggingFace Hub (first run only)
#   b) Load weights into GPU HBM (this is the slow part)
#   c) Run a dummy forward pass to measure peak activation memory
#   d) Calculate how much HBM is left for the KV cache
#   e) Allocate the KV cache block pool
#   f) Pre-capture CUDA graphs for batch sizes [1,2,4,8,16,32]
#
# gpu_memory_utilization=0.85 means: use at most 85% of total HBM
# for model weights + KV cache combined. 15% is kept as headroom
# to avoid out-of-memory crashes from unexpected spikes.

t0 = time.perf_counter()
print("Loading model... (this takes 30-60 seconds for a 1B model)")

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

print(f"Model loaded in {time.perf_counter() - t0:.1f}s\n")

# ── Step 2: Define sampling parameters ──────────────────────────────
# These control HOW the model picks the next token from the probability
# distribution it produces. We cover sampling in detail in Chapter 11.
#
# temperature=0.7  : higher = more random, lower = more deterministic
# top_p=0.9        : only consider tokens whose cumulative probability
#                    reaches 90% of the total distribution
# max_tokens=128   : stop after generating at most 128 tokens

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=128,
)

# ── Step 3: Generate ─────────────────────────────────────────────────
# generate() runs the full pipeline for all prompts:
#   tokenize → prefill → decode loop → detokenize
# It returns a list of RequestOutput objects.

prompt = "Explain the difference between prefill and decode in LLM inference."

print("Running inference...")
t1 = time.perf_counter()
outputs = llm.generate([prompt], params)
gen_time = time.perf_counter() - t1

# ── Step 4: Read the results ─────────────────────────────────────────
output = outputs[0]
n_prompt_tokens  = len(output.prompt_token_ids)
n_output_tokens  = len(output.outputs[0].token_ids)

print(f"Prompt ({n_prompt_tokens} tokens):")
print(f"  {output.prompt}")
print(f"\nResponse ({n_output_tokens} tokens):")
print(f"  {output.outputs[0].text}")
print(f"\nGeneration time: {gen_time:.2f}s")
print(f"Throughput:      {n_output_tokens / gen_time:.1f} tokens/sec")
```

**Expected output (approximate, 1B model on RTX 4090):**
```
Model loaded in 8.3s

Running inference...
Prompt (17 tokens):
  Explain the difference between prefill and decode in LLM inference.

Response (112 tokens):
  In large language model inference, there are two distinct phases: ...

Generation time: 1.21s
Throughput:      92.6 tokens/sec
```

---

## 1.9 Hello World — C++ with llama.cpp

The C++ version makes every step of the pipeline explicit. Read the comments carefully — they explain things that vLLM does silently behind the scenes.

```cpp
// hello_llamacpp.cpp
// Build (CPU-only, no GPU needed):
//   git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
//   cmake -B build && cmake --build build -j4
//   # compile our file:
//   g++ -std=c++17 -O2 hello_llamacpp.cpp \
//       -I./include -L./build/src -lllama \
//       -L./build/ggml/src -lggml -o hello_llamacpp
//
// Run:
//   ./hello_llamacpp ./model-q4_k_m.gguf
//
// A GGUF model can be downloaded with:
//   pip install huggingface_hub
//   huggingface-cli download \
//     bartowski/Llama-3.2-1B-Instruct-GGUF \
//     --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
//     --local-dir .

#include "llama.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    // ── Step 1: Initialize the backend ──────────────────────────────
    // llama_backend_init() must be called once at the start.
    // It detects available hardware (CPU, CUDA, Metal), initializes
    // thread pools, and sets up memory allocators.
    llama_backend_init();

    // ── Step 2: Load the model ───────────────────────────────────────
    // n_gpu_layers controls how many transformer layers are offloaded
    // to the GPU. 0 = all on CPU. 99 = all on GPU (if it fits in VRAM).
    // For a Q4_K_M 1B model (~0.7 GB), set to 99 to use GPU fully.
    //
    // KEY DIFFERENCE from vLLM: llama.cpp uses mmap (memory-mapped files).
    // The GGUF file is NOT fully read into RAM. The OS maps the file into
    // virtual memory and loads pages on-demand. This is why startup is
    // fast (seconds) even for large models.
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;  // change to 99 to offload all layers to GPU

    printf("Loading model: %s\n", argv[1]);
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "Load failed\n"); return 1; }
    printf("Model loaded.\n\n");

    // ── Step 3: Create a context ─────────────────────────────────────
    // The context holds:
    //   - The KV cache (pre-allocated as one contiguous buffer)
    //   - The computation state (activations, scratch buffers)
    //
    // n_ctx is the maximum total sequence length (prompt + response).
    // The KV cache size is fixed at allocation time:
    //   size = 2 × n_layers × n_ctx × n_kv_heads × head_dim × 2 bytes
    // For a 1B model at n_ctx=2048: roughly 256 MB.
    //
    // KEY DIFFERENCE from vLLM: this allocation is FIXED and CONTIGUOUS.
    // vLLM's PagedAttention allocates KV memory in dynamic pages.
    // llama.cpp allocates everything upfront in one block.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 2048;  // max context (prompt + generated tokens)
    cp.n_batch = 512;   // max tokens processed per llama_decode() call

    llama_context* ctx = llama_new_context_with_model(model, cp);
    if (!ctx) { fprintf(stderr, "Context failed\n"); return 1; }

    // ── Step 4: Tokenize the prompt ──────────────────────────────────
    // llama_tokenize() converts a text string into integer token IDs.
    // add_special=true prepends the BOS (beginning-of-sequence) token,
    // which the model expects at the start of every prompt.
    const std::string prompt =
        "Explain the difference between prefill and decode in LLM inference.";

    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(
        llama_model_get_vocab(model),
        prompt.c_str(), (int)prompt.size(),
        tokens.data(), (int)tokens.size(),
        /*add_special=*/true, /*parse_special=*/false
    );
    tokens.resize(n_tokens);
    printf("Prompt: %d tokens\n\n", n_tokens);

    // ── Step 5: Prefill ──────────────────────────────────────────────
    // llama_batch_get_one() wraps our token array in a batch struct.
    // llama_decode() runs the full forward pass for all prompt tokens:
    //   - For each token position, compute Q, K, V
    //   - Store K and V into the KV cache at positions [0..n_tokens-1]
    //   - Compute the final logits (probability distribution over vocab)
    //     for the LAST token position
    // After this call, the KV cache contains the "memory" of the prompt.
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill failed\n"); return 1;
    }

    // ── Step 6: Decode loop ──────────────────────────────────────────
    // Set up a sampler. llama_sampler_init_greedy() always picks the
    // highest-probability token. For temperature sampling, use
    // llama_sampler_init_temp(0.7) instead.
    llama_sampler* sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    printf("Response: ");
    int n_generated = 0;

    while (n_generated < 128) {
        // a) Sample: pick the next token from the logit distribution
        //    llama_sampler_sample() reads the logits from ctx and
        //    applies the sampler's logic to return one token ID.
        llama_token next = llama_sampler_sample(sampler, ctx, -1);

        // b) Check for end-of-generation signal
        //    llama_vocab_is_eog() returns true for EOS, EOT, or any
        //    other special token that signals "stop generating."
        if (llama_vocab_is_eog(llama_model_get_vocab(model), next))
            break;

        // c) Detokenize: convert the integer token ID back to text
        char buf[32];
        int len = llama_token_to_piece(
            llama_model_get_vocab(model),
            next, buf, sizeof(buf), 0, false
        );
        if (len > 0) { fwrite(buf, 1, len, stdout); fflush(stdout); }

        // d) Submit the new token for the next decode step.
        //    We create a batch containing only this one token.
        //    llama_decode() runs a forward pass for this single token,
        //    attending over the full KV cache (all previous tokens).
        //    The KV cache grows by 1 position with each call.
        batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) break;

        n_generated++;
    }
    printf("\n\n");

    // ── Step 7: Print timing summary ─────────────────────────────────
    // llama_perf_context_print() shows:
    //   - prompt processing speed (tokens/sec during prefill)
    //   - generation speed (tokens/sec during decode)
    llama_perf_context_print(ctx);

    // ── Step 8: Free resources ───────────────────────────────────────
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
```

**Expected output (approximate, 1B Q4_K_M on Apple M2):**
```
Loading model: Llama-3.2-1B-Instruct-Q4_K_M.gguf
Model loaded.

Prompt: 17 tokens

Response: In large language model inference, prefill and decode
represent two distinct phases of the generation process...

llama_perf_context_print:        load time =   1842.33 ms
llama_perf_context_print: prompt eval time =    312.44 ms /    17 tokens
llama_perf_context_print:        eval time =   3201.18 ms /   108 runs
llama_perf_context_print:       total time =   3517.11 ms /   125 tokens
```

Notice `prompt eval time` (prefill: 17 tokens in 312 ms = ~54 tok/s) is much faster per token than `eval time` (decode: 108 tokens in 3201 ms = ~34 tok/s). Prefill processes many tokens in parallel; decode processes one at a time. This asymmetry appears in both engines and we will explore why in Chapter 4.

---

## 1.10 First Performance Numbers

Before we go any further, here are baseline numbers to anchor your intuition. We will derive every one of these from first principles in Chapter 2.

```
  DECODE SPEED (tokens/sec, batch=1, 7B model)
  ════════════════════════════════════════════════════════════════

  Hardware              Engine          Precision   tok/s (decode)
  ──────────────────────────────────────────────────────────────
  H100 SXM (80 GB)      vLLM            FP16            ~105
  A100 SXM (80 GB)      vLLM            FP16            ~ 75
  RTX 4090 (24 GB)      vLLM            FP16            ~ 55
  RTX 4090 (24 GB)      llama.cpp       Q4_K_M          ~ 90
  M3 Max (128 GB)       llama.cpp       Q4_K_M          ~ 38
  M2 Ultra (192 GB)     llama.cpp       Q4_K_M          ~ 52
  CPU (32-core Xeon)    llama.cpp       Q4_K_M          ~ 10
  ──────────────────────────────────────────────────────────────

  Reference: a human reads at ~5 tokens/sec. Every row above is
  faster than human reading speed — but all rows serve only 1 user.

  RTX 4090 COMPARISON:
  ─────────────────────────────────────────────────────────────
  vLLM FP16:     ~55 tok/s    ← loads 14 GB of weights per token
  llama.cpp Q4:  ~90 tok/s    ← loads  3.5 GB of weights per token
                               (4× smaller = 4× less memory traffic)

  llama.cpp wins at batch=1 on RTX 4090 because quantization
  reduces memory bandwidth usage more than it hurts accuracy.
  ════════════════════════════════════════════════════════════════
```

```
  THROUGHPUT (total tokens/sec, ALL users, 7B model, H100)
  ════════════════════════════════════════════════════════════════

  Batch size    vLLM (FP16)     llama.cpp (Q4)
  ──────────────────────────────────────────
       1            105              N/A*
       4            380              ~180
      16          1,200              ~220
      32          2,100              ~250
     128          3,800              ~270
  ──────────────────────────────────────────
  *llama.cpp does not natively batch concurrent users

  vLLM's throughput scales nearly linearly with batch size
  up to the roofline ridge (~batch=295 for this model).
  llama.cpp's throughput barely increases because it is not
  designed for concurrent multi-user workloads.
  ════════════════════════════════════════════════════════════════
```

The crossover is clear: for a single user, llama.cpp is competitive or faster. For many users, vLLM scales and llama.cpp does not.

---

## 1.11 The Decision Matrix

```
  WHICH ENGINE SHOULD I USE?
  ════════════════════════════════════════════════════════════════

  START HERE
       │
       ▼
  Do you have a CUDA GPU (NVIDIA)?
  ├─ NO ──▶ Do you have Apple Silicon (M1/M2/M3)?
  │          ├─ YES ──▶ llama.cpp  (Metal backend, unified memory)
  │          └─ NO  ──▶ llama.cpp  (CPU AVX2/NEON)
  │
  └─ YES ──▶ How many concurrent users do you expect?
              │
              ├─ 1–4 users (dev machine, personal use)
              │   └──▶ llama.cpp  (simpler, faster cold start)
              │
              ├─ 5–50 users (small team, internal API)
              │   └──▶ Either works; vLLM preferred if GPU > 24 GB
              │
              └─ 50+ users (production API, SaaS)
                  └──▶ vLLM  (continuous batching essential)

  Additional factors:
  ──────────────────────────────────────────────────────────────
  Need OpenAI-compatible API?        → vLLM (built-in)
  Running on a laptop?               → llama.cpp
  No Python allowed (embedded sys)?  → llama.cpp
  Need LoRA hot-swap per request?    → vLLM
  Need structured output (grammars)? → Either (both support it)
  Budget: < $1/hour compute?         → llama.cpp on consumer GPU
  Model > 70B parameters?            → vLLM (multi-GPU TP)
  Experimenting with model internals?→ llama.cpp (C++ is inspectable)
  ════════════════════════════════════════════════════════════════
```

---

## Chapter Summary

- **Autoregressive generation** requires one forward pass per output token. These passes cannot run in parallel because each token depends on all previous tokens. This is the root cause of all inference complexity.

- **Inference is memory-bandwidth-bound**, not compute-bound, for small batches. At batch size 1, a 7B model has an arithmetic intensity of ~1 FLOPs/byte on an H100 — 295× below the ridge point. The GPU's compute units sit nearly idle while the memory bus works.

- **Batching recovers GPU utilization.** Arithmetic intensity scales linearly with batch size because model weights are loaded once per forward pass regardless of how many users are in the batch. This is why serving many users simultaneously is efficient.

- **vLLM** is a Python/CUDA inference server optimized for high-concurrency GPU deployment. It uses PagedAttention to manage KV cache memory efficiently and continuous batching to keep throughput high. It requires a CUDA GPU.

- **llama.cpp** is a C++ inference library optimized for portability and single-user latency. It uses GGUF quantized models loaded via mmap for fast cold start. It runs on CPU, CUDA, Metal, and Vulkan with no mandatory dependencies.

- **Both engines implement the same four-step pipeline:** tokenize → prefill → decode loop → detokenize. The difference is how they manage memory, scheduling, and hardware abstraction.

---

## Self-Check Questions

Try to answer these without looking back. If you cannot, re-read the indicated section.

1. A user asks for a 50-token response to a 20-token prompt. How many forward passes does the model run? What are they called? *(Section 1.1)*

2. The H100 has a peak bandwidth of 3.35 TB/s and peak compute of 989 TFLOPS (dense BF16). What is its ridge point in FLOPs/byte? Show the arithmetic. *(Section 1.2)*

3. You are running a 13B FP16 model (13 billion parameters × 2 bytes = 26 GB of weights) at batch size 1. What is the approximate arithmetic intensity of one decode step? Is this above or below the H100 ridge point? *(Section 1.2 + 1.3)*

4. Why does increasing the batch size from 1 to 32 increase arithmetic intensity by 32× but not increase the number of bytes read from memory by 32×? *(Section 1.3)*

5. What is the main advantage of llama.cpp's mmap-based model loading compared to vLLM's approach, and what does it trade away? *(Sections 1.4 and 1.5)*

---

## Where We Go Next

Chapter 2 opens the memory subsystem in detail. We will compute the exact HBM budget for a vLLM deployment — how many bytes go to weights, how many to activations, how many to the KV cache — and derive the same calculation for llama.cpp on Apple Silicon and on a CUDA GPU with partial offload. You will build a memory budget calculator step by step and be able to predict, before loading a single weight, whether a given model fits on a given piece of hardware and how many concurrent users it can serve.

---

*Code for this chapter: `vllm_book/code/chapter_01/hello_vllm.py` and `hello_llamacpp.cpp`*
*Next: Chapter 2 — The GPU and CPU Memory Landscapes*


---

## Worked Solutions

> Work through each question yourself first. The solutions below are step-by-step
> for learners who are new to inference engineering — no steps are skipped.

---

### Solution 1 — Forward passes for a 20-token prompt + 50-token response

**What we need:** Count the forward passes and name them.

**Step 1 — Understand the two phases of LLM inference.**

Every LLM inference job has exactly two distinct phases:

- **Prefill (prompt processing):** The entire prompt is fed through the model in a *single* forward pass. All 20 prompt tokens are processed simultaneously, and the key-value (KV) cache is populated for every prompt token.

- **Decode (autoregressive generation):** The model generates *one token at a time*. Each generated token is fed back into the model as input for the next step. This continues until an end-of-sequence token is produced or the requested length is reached.

**Step 2 — Count the passes.**

| Phase | Tokens processed | Forward passes |
|-------|-----------------|----------------|
| Prefill | All 20 prompt tokens at once | **1** |
| Decode | 1 token per step × 50 steps | **50** |
| **Total** | | **51** |

**Step 3 — Why exactly one prefill pass?**

During prefill, the attention mechanism allows every prompt token to attend to every other prompt token simultaneously (using the causal mask). This parallelism is why a 20-token prompt takes the same number of forward passes as a 1-token prompt — both take exactly 1. The work per pass is proportional to sequence length (longer prompt = more FLOPs per pass), but always exactly 1 pass.

**Step 4 — Why 50 separate decode passes?**

Each decode step *produces* one token and *consumes* one token (the previous output). The autoregressive property means token N cannot be generated until token N−1 exists. There is no way to parallelize this within a single request. Each of the 50 steps is a separate forward pass.

**Common mistake:** Saying "71 forward passes" (20 + 50 + 1). The 20 prompt tokens are NOT processed one at a time — they are processed together in the single prefill pass.

---

### Solution 2 — H100 ridge point calculation

**What we need:** Ridge point in FLOPs/byte = Peak FLOPS ÷ Peak Bandwidth.

**Step 1 — Write down the values.**

- Peak compute: 989 TFLOPS = 989 × 10¹² FLOPS/s (dense BF16)
- Peak bandwidth: 3.35 TB/s = 3.35 × 10¹² bytes/s

**Step 2 — Compute the ridge point.**

$$\text{Ridge point} = \frac{\text{Peak FLOPS/s}}{\text{Peak Bandwidth (bytes/s)}} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295 \text{ FLOPs/byte}$$

**Step 3 — Interpret the result.**

The ridge point is the arithmetic intensity at which a workload *just barely* saturates both compute and memory simultaneously. It is the crossover point on the roofline model:

- **Below 295 FLOPs/byte:** the workload is **memory-bound** (memory delivers data faster than compute can consume it).
- **Above 295 FLOPs/byte:** the workload is **compute-bound** (compute is the bottleneck).

**Step 4 — Where does decode sit?**

A single-batch decode step has an arithmetic intensity of roughly 1 FLOPs/byte (see Solution 3 below). Since 1 ≪ 295, decode is *deeply* memory-bound. This is the central fact of LLM inference: the GPU's 989 TFLOPS of compute sits almost entirely idle during single-request decode.

---

### Solution 3 — Arithmetic intensity for 13B FP16 model at batch size 1

**What we need:** FLOPs per decode step ÷ bytes read from HBM.

**Step 1 — Compute the weight size.**

$$13 \times 10^9 \text{ parameters} \times 2 \text{ bytes/parameter (FP16)} = 26 \times 10^9 \text{ bytes} = 26 \text{ GB}$$

**Step 2 — Compute the FLOPs per decode step (batch size 1).**

During a single decode step, the model multiplies one input activation vector (the single new token) by each weight matrix. Each multiplication is a dot product. For each parameter:

- 1 multiply + 1 add = **2 FLOPs per weight**

$$\text{FLOPs} = 2 \times 13 \times 10^9 = 26 \times 10^9 = 26 \text{ GFLOPs}$$

**Step 3 — Compute bytes read.**

At batch size 1, each weight is needed exactly once per decode step (the single token's activation is multiplied by every weight). All 26 GB of weights must be streamed from HBM:

$$\text{Bytes read} = 26 \text{ GB}$$

**Step 4 — Compute arithmetic intensity.**

$$\text{Arithmetic intensity} = \frac{26 \text{ GFLOPs}}{26 \text{ GB}} = 1 \text{ FLOP/byte}$$

**Step 5 — Compare to the H100 ridge point.**

H100 ridge point ≈ 295 FLOPs/byte. Our decode intensity = 1 FLOPs/byte.

$$1 \ll 295 \implies \text{Deeply memory-bound}$$

The model is using less than 0.2% of the H100's available compute. The GPU is 99.8% idle on compute — waiting for memory to deliver the next set of weights.

**Intuition:** A 26 GB weight matrix load takes about 26 GB ÷ 3.35 TB/s ≈ 7.8 ms. During those 7.8 ms, the H100 could perform 989 TFLOPS × 0.0078 s ≈ 7.7 TFLOPS of compute — but we only ask it to do 26 GFLOPs. We are wasting ~7,680 GFLOPs of available compute every decode step at batch size 1.

---

### Solution 4 — Why batch size 32 increases arithmetic intensity by 32× without 32× more memory

**What we need:** Explain the asymmetry between FLOPs (scale with batch) and bytes (don't).

**Step 1 — What changes with batch size?**

At batch size 32, we process 32 token vectors simultaneously instead of 1.

**Step 2 — How FLOPs scale.**

Each of the 32 token vectors must be multiplied by every weight matrix. The computation for each token is independent:

$$\text{FLOPs at batch 32} = 32 \times 26 \text{ GFLOPs} = 832 \text{ GFLOPs}$$

FLOPs scale linearly with batch size. ✓

**Step 3 — How bytes read scale.**

Here is the key insight: **the same weights serve all 32 tokens.**

When we read a weight matrix column from HBM, we multiply it by *all 32 token activation vectors* before we need to read the next column. The weight bytes are read *once* regardless of batch size:

$$\text{Bytes read at batch 32} \approx 26 \text{ GB (same as batch 1)}$$

The activation vectors (the 32 token embeddings) are small — each is typically `d_model` floats, so 32 × 4096 × 2 bytes ≈ 256 KB — negligible compared to 26 GB of weights.

**Step 4 — New arithmetic intensity.**

$$\text{Arithmetic intensity at batch 32} = \frac{832 \text{ GFLOPs}}{26 \text{ GB}} = 32 \text{ FLOPs/byte}$$

This is 32× higher than batch size 1 — exactly equal to the batch size. ✓

**Step 5 — The general rule.**

For a model with W weights in FP16 (W bytes × 2):

| Batch size B | FLOPs | Bytes | Arithmetic intensity |
|---|---|---|---|
| 1 | 2W | 2W | 1 FLOPs/byte |
| 32 | 2W × 32 | 2W | 32 FLOPs/byte |
| 295 | 2W × 295 | 2W | 295 FLOPs/byte (ridge!) |

At batch size ~295, an FP16 13B model would *just* saturate the H100's compute. Below that, you are memory-bound. This is why large-scale serving operations care so much about maintaining high batch sizes.

---

### Solution 5 — llama.cpp mmap vs vLLM eager loading: advantage and trade-off

**What we need:** The main advantage of mmap, and what it costs.

**Step 1 — How vLLM loads models (eager loading).**

vLLM reads the entire model checkpoint into GPU HBM before serving any request. For a 70B FP16 model, this means copying 140 GB from disk → CPU DRAM → GPU HBM. On a typical server with an NVMe SSD (7 GB/s read speed), this takes 140 GB ÷ 7 GB/s ≈ 20 seconds *just for disk reads*, plus PCIe transfer time.

**Step 2 — How llama.cpp loads models (mmap).**

`mmap` (memory-mapped file I/O) maps the GGUF model file into the process's virtual address space *without reading it*. The OS page table is updated to say "if address range X is accessed, load it from file offset Y," but no bytes are copied yet. The function returns almost instantly — in milliseconds.

$$\text{Startup time with mmap} \approx \text{milliseconds} \quad \text{vs} \quad \text{Startup time with eager} \approx 10\text{–}60 \text{ seconds}$$

**Step 3 — The advantage.**

The main advantage is **dramatically faster first-token time from a cold start**. A laptop running llama.cpp can serve its first token within 300 ms of launch. This is critical for:

- Interactive desktop applications
- Low-latency mobile/edge use cases
- Testing and development where you restart the server frequently

**Step 4 — The trade-off.**

mmap causes **page faults during inference**. When a weight page is accessed for the first time, the OS must:

1. Detect the page fault
2. Issue a disk I/O to load the page into DRAM
3. Map it into the process's address space
4. Resume execution

This page fault typically costs 1–50 ms and causes **unpredictable latency spikes**. After the model has run a few inference passes and all pages have been loaded ("warm" state), performance stabilizes — but the first few requests pay the penalty.

**Summary table:**

| Property | vLLM (eager) | llama.cpp (mmap) |
|---|---|---|
| Cold start time | 10–60 s | < 1 s |
| First-request latency (cold) | Low | High (page faults) |
| First-request latency (warm) | Low | Low |
| Supports GPU memory pinning | Yes | Partial |
| RAM required | Full model | On-demand |

