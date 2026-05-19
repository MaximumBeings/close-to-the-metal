# Chapter 2: The GPU and CPU Memory Landscapes

> *"You cannot tune what you cannot measure, and you cannot measure what you do not understand. Memory is where inference lives and dies."*

---

**What you will understand by the end of this chapter:**

- The difference between GPU memory (HBM), system RAM (DRAM), and Apple Silicon's unified memory — and why bandwidth is the number that matters most
- How to calculate exactly how many bytes a model's weights consume at any precision (FP32, FP16, BF16, INT8, INT4)
- The KV cache formula — what it is, where it comes from, and how to compute it for any model
- How vLLM splits a GPU's HBM into three regions and why that split determines how many users you can serve
- How llama.cpp loads models differently from vLLM, and how partial GPU offload works

**What you need to know first:**

- Chapter 1 (the autoregressive constraint and why inference is memory-bandwidth-bound)
- You should know that computers have memory — you do not need to know hardware details yet; this chapter introduces them

---

## 2.1 The Memory Hierarchy: Where Data Lives

`[FOUNDATIONAL]`

### Intuition

Every computer has multiple places to store data, each with a different size and speed. Imagine a chef working in a restaurant kitchen:

- The chef's **hands** hold what they are working on right now — tiny capacity, instant access
- The **countertop** holds ingredients for the current dish — small but immediately reachable
- The **refrigerator** holds ingredients for the whole day — larger, but takes a few seconds to walk over
- The **storage room** holds bulk supplies — large, but takes a minute to retrieve
- The **supplier's warehouse** holds everything the restaurant might ever need — enormous, but requires a delivery order placed days in advance

A computer has a nearly identical hierarchy. For inference, two levels matter far more than the others: the GPU's on-chip memory (fast but small) and its main memory bank (large but slower). Understanding the difference between them is the foundation of everything in this chapter.

### Background: What is HBM?

> **HBM (High Bandwidth Memory)** is the type of memory used in data-center GPUs like the NVIDIA H100 and A100. It is physically mounted on the same package as the GPU chip, connected via thousands of tiny wires (the "bus") running in parallel. This parallelism is what gives HBM its enormous bandwidth.
>
> **DRAM (Dynamic Random-Access Memory)** is the system RAM in a regular server or desktop computer — the sticks you plug into a motherboard. It connects to the CPU via a much narrower bus. Bandwidth is typically 50–100× lower than HBM.
>
> **Unified Memory** (Apple Silicon) is a design where the CPU and GPU share the same physical memory pool. There is no separate VRAM. The M2 Ultra's 192 GB of memory is accessible to both the CPU and the GPU cores with the same bandwidth (~800 GB/s), eliminating the PCIe bottleneck that limits GPU offload on x86 systems.

### The Memory Hierarchy for Inference

```
  MEMORY HIERARCHY — Inference perspective
  ═══════════════════════════════════════════════════════════════════

  Level         │ Technology  │ Capacity    │ Bandwidth    │ Latency
  ──────────────┼─────────────┼─────────────┼──────────────┼─────────
  GPU Registers │ SRAM        │ ~30 MB      │ ~100 TB/s    │ < 1 ns
  (L1/L2 cache) │             │             │              │
  ──────────────┼─────────────┼─────────────┼──────────────┼─────────
  GPU HBM       │ HBM3 / HBM2e│ 40–192 GB   │ 2–4 TB/s     │ ~100 ns
  (VRAM)        │             │             │              │
  ──────────────┼─────────────┼─────────────┼──────────────┼─────────
  System DRAM   │ DDR5        │ 32 GB–2 TB  │ 50–100 GB/s  │ ~100 ns
  (CPU RAM)     │             │             │              │
  ──────────────┼─────────────┼─────────────┼──────────────┼─────────
  PCIe Bus      │ PCIe 4.0    │ —           │ ~32 GB/s     │ ~1 µs
  (GPU↔CPU)     │ PCIe 5.0    │ —           │ ~64 GB/s     │
  ──────────────┼─────────────┼─────────────┼──────────────┼─────────
  NVMe SSD      │ PCIe SSD    │ 1 TB–32 TB  │ 5–15 GB/s    │ ~100 µs
  ──────────────┼─────────────┼─────────────┼──────────────┼─────────
  HDD / Network │ SATA / TCP  │ unlimited   │ < 1 GB/s     │ > 1 ms
  ═══════════════════════════════════════════════════════════════════

  For inference, EVERYTHING fits in:  HBM  (fastest after registers)
  When model is too large for HBM:    overflow to DRAM (via PCIe)
  llama.cpp mmap reads from:          NVMe SSD → DRAM → GPU (lazily)

  The bandwidth drop from HBM to DRAM is ~30-60×.
  This is why "spilling" to DRAM hurts performance severely.
```

### Key Bandwidth Numbers to Memorize

These numbers appear throughout the book. Commit them to memory — you will use them in every roofline calculation.

```
  MEMORY BANDWIDTH REFERENCE
  ═══════════════════════════════════════════════════════════════════

  Hardware                      Memory Type      Bandwidth
  ──────────────────────────────────────────────────────────────────
  NVIDIA H100 SXM (80 GB)       HBM3             3.35 TB/s
  NVIDIA H100 PCIe (80 GB)      HBM3             2.00 TB/s
  NVIDIA A100 SXM (80 GB)       HBM2e            2.00 TB/s
  NVIDIA A100 PCIe (80 GB)      HBM2e            1.94 TB/s
  NVIDIA RTX 4090 (24 GB)       GDDR6X           1.01 TB/s
  NVIDIA RTX 3090 (24 GB)       GDDR6X           0.94 TB/s
  NVIDIA A10G (24 GB)           GDDR6             0.60 TB/s
  ──────────────────────────────────────────────────────────────────
  Apple M2 Ultra (192 GB)       Unified LPDDR5   800 GB/s
  Apple M3 Max  (128 GB)        Unified LPDDR5   400 GB/s
  Apple M2 Max  ( 96 GB)        Unified LPDDR5   400 GB/s
  Apple M2 Pro  ( 32 GB)        Unified LPDDR5   200 GB/s
  ──────────────────────────────────────────────────────────────────
  Typical server DRAM (DDR5)    DRAM             50–100 GB/s
  Typical laptop DRAM (DDR5)    DRAM             30–80 GB/s
  PCIe 4.0 × 16 (GPU↔CPU)      PCIe             ~32 GB/s
  PCIe 5.0 × 16 (GPU↔CPU)      PCIe             ~64 GB/s
  ═══════════════════════════════════════════════════════════════════

  Rule of thumb: HBM bandwidth ≈ 30–60× faster than DRAM.
  Apple unified memory sits between HBM and DRAM.
```

`[COMMON TRAP]` When people say "my GPU has 24 GB of memory," they mean VRAM / HBM — the memory on the GPU itself, NOT the system RAM. A machine might have 64 GB of system RAM and only 24 GB of GPU HBM. For inference, **what matters is HBM** because that is where the model must live during the forward pass. System RAM only matters for model loading or CPU offload.

---

## 2.2 Floating-Point Precision: How Many Bytes per Weight?

`[FOUNDATIONAL]`

### Intuition

A model parameter is just a number — a decimal like `0.0034712` or `-1.2847`. To store a number in a computer, you have to decide how much precision you need. More precision = more bytes = more memory = slower to load from memory. Less precision = fewer bytes = fits in less memory = faster to load.

This is the core trade-off of **quantization**: accept slightly less precise numbers in exchange for dramatically smaller memory footprint. A model that does not fit on a GPU at FP16 might fit comfortably at INT4. Whether the quality loss is acceptable depends on the model and the task.

### The Precision Formats

> **Background: How floating-point numbers work**
> A floating-point number stores three things: a sign bit (positive/negative), an exponent (the scale), and a mantissa (the significant digits). More bits for the mantissa = more decimal precision. More bits for the exponent = ability to represent larger/smaller numbers. The formats below make different trade-offs in how they split available bits.

```
  FLOATING-POINT FORMATS USED IN LLM INFERENCE
  ═══════════════════════════════════════════════════════════════════

  Format   Bits  Bytes  Sign  Exponent  Mantissa  Notes
  ─────────────────────────────────────────────────────────────────
  FP32      32     4      1      8        23      Full precision.
                                                  Training default.
                                                  Rarely used for
                                                  inference today.

  FP16      16     2      1      5        10      Half precision.
                                                  vLLM default.
                                                  Supported by all
                                                  CUDA GPUs.

  BF16      16     2      1      8         7      Brain Float 16.
                                                  Same exponent as
                                                  FP32 (better range),
                                                  less mantissa.
                                                  Preferred on H100,
                                                  A100, Apple Silicon.

  INT8       8     1      1      —         7      Integer quantization.
                                                  8-bit signed integer.
                                                  Used in LLM.int8(),
                                                  TensorRT-LLM.

  FP8        8     1      1      4         3      8-bit float.
                                                  H100 native support.
                                                  vLLM KV cache option.

  INT4       4    0.5     —      —         —      4-bit integer.
                                                  Requires dequant step.
                                                  GPTQ, AWQ, GGUF Q4.
                                                  ~0.45 bytes/weight
                                                  (with metadata).
  ─────────────────────────────────────────────────────────────────
  Note: INT4 numbers are typically stored in groups (blocks) with a
  shared scale factor. Actual bytes/weight ≈ 0.45–0.55 depending on
  the block format. For quick estimates, use 0.5 bytes/weight.
  ═══════════════════════════════════════════════════════════════════
```

### Worked Example: Weight Memory at Each Precision

Let us compute the memory required for a Llama 3 8B model at every precision level.

```
WORKED EXAMPLE 2.1 — Weight memory for Llama 3 8B at each precision
─────────────────────────────────────────────────────────────────────
Given:
  Model:           Llama 3 8B
  Parameters:      8.03 × 10⁹  (we will use 8 × 10⁹ for simplicity)

Step 1: FP32 (4 bytes per parameter)
  Memory = 8 × 10⁹ parameters × 4 bytes
         = 32 × 10⁹ bytes
         = 32 GB

Step 2: FP16 or BF16 (2 bytes per parameter)
  Memory = 8 × 10⁹ × 2 bytes
         = 16 × 10⁹ bytes
         = 16 GB

Step 3: INT8 (1 byte per parameter)
  Memory = 8 × 10⁹ × 1 byte
         = 8 × 10⁹ bytes
         = 8 GB

Step 4: INT4 / Q4 (≈ 0.5 bytes per parameter, approximate)
  Memory = 8 × 10⁹ × 0.5 bytes
         = 4 × 10⁹ bytes
         = 4 GB
  (Actual GGUF Q4_K_M: ~4.58 GB due to block metadata)

Summary table:
  FP32:   32 GB   ← 2× the FP16 size, rarely used for inference
  FP16:   16 GB   ← vLLM default; fits on one A100 80 GB with room
  BF16:   16 GB   ← same size as FP16 (different bit layout)
  INT8:    8 GB   ← fits on RTX 4090 (24 GB) easily
  INT4:   ~4 GB   ← fits on RTX 3080 (10 GB) with room for KV cache

Final answer:
  Each halving of precision roughly halves the memory requirement.
  Going from FP16 to INT4 saves 12 GB for a 8B model — enough to fit
  a second model instance on the same GPU.
─────────────────────────────────────────────────────────────────────
```

### Worked Example: Weight Memory for a 70B Model

```
WORKED EXAMPLE 2.2 — Weight memory for Llama 3 70B at each precision
─────────────────────────────────────────────────────────────────────
Given:
  Model:           Llama 3 70B
  Parameters:      70.6 × 10⁹  (we use 70 × 10⁹ for simplicity)

Step 1: FP16 (2 bytes per parameter)
  Memory = 70 × 10⁹ × 2 bytes
         = 140 × 10⁹ bytes
         = 140 GB

  H100 SXM has 80 GB HBM.
  140 GB > 80 GB → does NOT fit on a single H100.
  Need at least 2× H100 for FP16 inference.

Step 2: INT4 (≈ 0.5 bytes per parameter)
  Memory = 70 × 10⁹ × 0.5 bytes
         = 35 × 10⁹ bytes
         = 35 GB

  35 GB < 80 GB → fits on a SINGLE H100 at INT4.
  (Leaves 45 GB for KV cache and activations — very comfortable.)

Step 3: On Apple M2 Ultra (192 GB unified memory)
  FP16: 140 GB — fits (with 52 GB left for KV cache)
  INT4: ~35 GB  — fits easily (157 GB remaining)

Final answer:
  70B FP16 requires 2× H100 (80 GB each) with tensor parallelism.
  70B INT4 fits on a single H100 with ample KV cache headroom.
  70B FP16 fits on a single M2 Ultra.
  This is why llama.cpp + M2 Ultra is a serious production option
  for teams that need FP16 quality without multi-GPU complexity.
─────────────────────────────────────────────────────────────────────
```

### ASCII Diagram: Weight Memory by Model and Precision

```
  MODEL WEIGHT MEMORY (GB) — Common models and precisions
  ═══════════════════════════════════════════════════════════════════

                    FP32    FP16/BF16    INT8    INT4 (~Q4)
                    ────    ─────────    ────    ──────────
  Llama 3 1B         4 GB     2 GB      1 GB     0.7 GB
  Llama 3 8B        32 GB    16 GB      8 GB     4.5 GB
  Llama 3 70B      140 GB    70 GB     35 GB      18 GB
  Llama 3 405B     810 GB   405 GB    202 GB     101 GB

  GPU VRAM reference:
  ───────────────────────────────────────────────────────
  RTX 4090:    24 GB  ████████░░░░░░░░░░░░░░░░░░░░░░
  A100 40GB:   40 GB  ████████████████░░░░░░░░░░░░░░
  A100 80GB:   80 GB  ████████████████████████████████
  H100 80GB:   80 GB  ████████████████████████████████
  H100 NVL:    94 GB  ════════════════════════════════════  (188 GB per NVL pair)

  Rules of thumb:
  ● If weights > 80% of GPU VRAM → too tight (no room for KV cache)
  ● If weights > VRAM → need multiple GPUs or quantization
  ● INT4 ≈ 1 GB per billion parameters (convenient heuristic)
  ═══════════════════════════════════════════════════════════════════
```

`[COMMON TRAP]` The "1 GB per billion parameters at INT4" rule works well for estimates but is not exact. A 70B model at INT4 is ~18 GB, not 35 GB (the "~0.5 bytes" formula gives 35 GB because it uses 0.5 bytes, but actual GGUF Q4_K_M averages closer to ~4.5 bits including block scale metadata). For planning, use 0.5 bytes/param as a conservative upper bound.

---

## 2.3 The KV Cache: The Other Memory Consumer

`[FOUNDATIONAL]`

### Intuition

When the model generates token number 50, it needs to "remember" what was in tokens 1 through 49. In the transformer architecture, this memory is stored as two matrices — called the **Key (K)** and **Value (V)** matrices — for each transformer layer. Together these form the **KV cache**.

Think of the KV cache as a notebook where the model writes a summary of everything it has seen, layer by layer. Each new token it reads adds one more row to the notebook. When the model generates the next token, it reads the entire notebook.

The KV cache has two critical properties:

1. It grows linearly with sequence length (more tokens = bigger notebook)
2. It must be kept in fast memory (HBM or unified memory) for every active user simultaneously

This makes the KV cache the primary constraint on how many concurrent users you can serve — often more limiting than the model weights themselves.

### Background: The Transformer Architecture

> A transformer model consists of **N layers** stacked on top of each other. Each layer has an attention mechanism with **n_heads** attention heads. Each head operates on vectors of size **head_dim** (also called d_k). For a given input token, the attention mechanism computes three vectors: Query (Q), Key (K), and Value (V).
>
> - The **Q vector** is used to ask "what should I attend to?"
> - The **K vector** is used to answer "what do I contain?"
> - The **V vector** is what gets mixed together to produce the output
>
> During generation, the Q vector for the current token attends over all K vectors from previous tokens, producing attention weights. Those weights are then applied to the V vectors to produce the output. The K and V vectors from all previous tokens must be available — that is the KV cache.
>
> **GQA (Grouped Query Attention):** Many modern models (Llama 3, Mistral) use fewer KV heads than query heads. If there are 32 query heads but only 8 KV heads, the KV cache is 4× smaller. This is called Grouped Query Attention (GQA). The n_kv_heads value in the formula below reflects this.

### The KV Cache Formula

```
  KV Cache memory per token =
    2 × n_layers × n_kv_heads × head_dim × bytes_per_element

  Where:
    2           = one K matrix + one V matrix
    n_layers    = number of transformer layers
    n_kv_heads  = number of KV heads (may be less than query heads in GQA)
    head_dim    = dimension of each attention head = hidden_dim / n_heads
    bytes_per_element = 2 for FP16/BF16, 1 for FP8, 4 for FP32

  Total KV cache for a sequence of length L tokens =
    (KV cache per token) × L
```

### Model Architecture Reference

Before working examples, here are the architecture parameters for common models:

```
  MODEL ARCHITECTURE PARAMETERS
  ═══════════════════════════════════════════════════════════════════

  Model          n_layers  n_heads  n_kv_heads  head_dim  hidden_dim
  ─────────────────────────────────────────────────────────────────
  Llama 3.2 1B      16       32         8         64        2048
  Llama 3.2 3B      28       24         8         128       3072
  Llama 3 8B        32       32         8         128       4096
  Llama 3 70B       80       64         8         128       8192
  Llama 3 405B     126       128        8         128      16384
  Mistral 7B        32       32         8         128       4096
  Qwen2.5 7B        28       28         4         128       3584
  DeepSeek-R1 7B    28       28         4         128       3584
  ═══════════════════════════════════════════════════════════════════

  Key observation: most modern models use n_kv_heads = 4 or 8,
  much less than n_heads (GQA). This is a deliberate design choice
  to reduce KV cache size without sacrificing output quality.
```

### Worked Example: KV Cache for Llama 3 8B

```
WORKED EXAMPLE 2.3 — KV cache memory for Llama 3 8B (FP16)
─────────────────────────────────────────────────────────────────────
Given:
  Model:         Llama 3 8B
  n_layers:      32
  n_kv_heads:    8    (GQA: 8 KV heads, 32 query heads)
  head_dim:      128
  Precision:     FP16 → 2 bytes per element
  Sequence:      We will compute for 1 token, then scale

Step 1: KV cache bytes per token.
  KV_per_token = 2 × n_layers × n_kv_heads × head_dim × bytes
               = 2 × 32       × 8          × 128       × 2
               = 2 × 32 × 8 × 128 × 2

  Let us compute this step by step to avoid errors:
    2 × 32     = 64
    64 × 8     = 512
    512 × 128  = 65,536
    65,536 × 2 = 131,072 bytes per token

  Convert to KB: 131,072 / 1024 = 128 KB per token

Step 2: KV cache for a full 8,192-token context.
  Total = 128 KB/token × 8,192 tokens
        = 1,048,576 KB
        = 1,024 MB
        = 1.0 GB per sequence

Step 3: KV cache for 50 concurrent users (each with 8,192 tokens).
  Total = 1.0 GB × 50 = 50 GB

Step 4: Compare to available HBM.
  H100 SXM HBM:           80 GB
  Model weights (FP16):  −16 GB
  Available for KV cache: 64 GB

  64 GB / 1.0 GB per user = 64 users maximum
  (before accounting for activation memory — see Section 2.4)

Final answer:
  Llama 3 8B KV cache = 128 KB per token = 1 GB per 8K-token sequence.
  A single H100 can hold KV caches for roughly 60 concurrent users
  each having an 8K-token conversation.
─────────────────────────────────────────────────────────────────────
```

### Worked Example: KV Cache for Llama 3 70B

```
WORKED EXAMPLE 2.4 — KV cache memory for Llama 3 70B (FP16)
─────────────────────────────────────────────────────────────────────
Given:
  Model:         Llama 3 70B
  n_layers:      80
  n_kv_heads:    8
  head_dim:      128
  Precision:     FP16 → 2 bytes
  Sequence:      Computing per token, then per 8K context

Step 1: KV cache bytes per token.
  KV_per_token = 2 × 80 × 8 × 128 × 2

  Step by step:
    2 × 80     = 160
    160 × 8    = 1,280
    1,280 × 128 = 163,840
    163,840 × 2 = 327,680 bytes per token

  Convert: 327,680 / 1024 = 320 KB per token

Step 2: KV cache for an 8,192-token context.
  Total = 320 KB × 8,192 = 2,621,440 KB = 2,560 MB ≈ 2.5 GB per sequence

Step 3: How many users on 2× H100 (FP16)?
  2× H100 HBM:              160 GB
  Weights (FP16):          −140 GB
  Available for KV cache:    20 GB

  20 GB / 2.5 GB per user = 8 users maximum

  This is very tight. Only 8 concurrent users with 8K contexts.
  This is why 70B FP16 needs careful memory management.

Step 4: How many users on 2× H100 (INT4 weights)?
  2× H100 HBM:              160 GB
  Weights (INT4):           −18 GB
  Available for KV cache:   142 GB

  142 GB / 2.5 GB per user = 56 users maximum

  Quantizing weights to INT4 increases concurrent user capacity
  from 8 to 56 — a 7× improvement in throughput potential.

Final answer:
  70B KV cache = 320 KB/token = 2.5 GB per 8K-token sequence.
  Weight quantization dramatically increases KV cache headroom.
─────────────────────────────────────────────────────────────────────
```

### ASCII Diagram: KV Cache Growth with Sequence Length

```
  KV CACHE GROWTH — Llama 3 8B, FP16, per-user

  Sequence   KV Cache    Context type
  length     per user
  ─────────────────────────────────────────────────────────────────
     512       64 MB    Short chat turn
   1,024      128 MB    Medium document
   2,048      256 MB    Long document
   4,096      512 MB    Book chapter
   8,192        1 GB    Full context window (Llama 3 8B default)
  16,384        2 GB    Extended context
  32,768        4 GB    Long-context models (Llama 3.1)
 131,072       16 GB    Max context (Llama 3.1 405B)

  Visual: KV cache size vs. sequence length (each █ = 128 MB)

    512  ▏█
   1024  ▏██
   2048  ▏████
   4096  ▏████████
   8192  ▏████████████████
  16384  ▏████████████████████████████████
  32768  ▏████████████████████████████████████████████████████████████
         └────────────────────────────────────────────────────────────
                                                              → GB used

  Key insight: KV cache is LINEAR in sequence length.
  Doubling the context length doubles the KV cache.
  At 128K tokens, the KV cache alone is 16 GB per user.
```

`[COMMON TRAP]` A very common mistake is thinking "the KV cache is for the model's memory across conversations." It is not. The KV cache holds the K and V vectors for the **current conversation context only** — the tokens in the current prompt + response. When the conversation ends, its KV cache is freed. The model has no memory between conversations unless you explicitly include prior conversation text in the new prompt.

---

## 2.4 vLLM's HBM Budget: The Three-Way Split

`[FOUNDATIONAL]`

### Intuition

When vLLM starts, it divides the GPU's HBM into exactly three regions:

1. **Model weights** — fixed, loaded once, never freed
2. **Activation workspace** — temporary space used during the forward pass, reused every step
3. **KV cache block pool** — the remaining HBM, divided into fixed-size blocks and handed out to active users

The size of region 3 determines how many concurrent users you can serve. Getting this split right is the most important tuning decision for a vLLM deployment. The `gpu_memory_utilization` parameter controls how much total HBM is claimed by vLLM (regions 1 + 2 + 3 combined).

### Activation Memory: The Temporary Region

During a forward pass, intermediate results are held in memory. These are called **activations** — the outputs of each layer that feed into the next layer. The activation memory peaks during the forward pass and is released afterward.

> **Background: What activations are**
> When you run a token through layer 1 of the transformer, layer 1 produces output vectors. Those vectors are the "activations" for layer 1. They are fed into layer 2, which produces its own activations, and so on. At any point, the GPU must hold the activations for the current layer (and sometimes several recent layers). The peak memory is typically reached at the widest layer of the model.

vLLM measures activation memory by running a **dummy forward pass** during startup — feeding in a batch of fake tokens and watching how much HBM is consumed. This measurement is exact for the specific batch configuration you have chosen.

### Worked Example: Complete HBM Budget for Llama 3 8B on H100

```
WORKED EXAMPLE 2.5 — Full HBM budget for Llama 3 8B on one H100 SXM
─────────────────────────────────────────────────────────────────────
Given:
  GPU:                   NVIDIA H100 SXM
  Total HBM:             80 GB
  gpu_memory_utilization: 0.90  (vLLM will use at most 90% of HBM)
  Model:                 Llama 3 8B in FP16
  max_model_len:         8,192 tokens
  block_size:            16 tokens per KV block (vLLM default)

Step 1: Total HBM available to vLLM.
  Available = 80 GB × 0.90 = 72 GB

Step 2: Model weight memory.
  Weights = 8 × 10⁹ params × 2 bytes/param = 16 GB

Step 3: Activation workspace (measured by dummy forward pass).
  vLLM runs a forward pass with max_num_batched_tokens=8,192 tokens.
  For Llama 3 8B, peak activation memory ≈ 1.5–2 GB.
  We use 2 GB as a conservative estimate.

Step 4: KV cache block pool (what's left).
  Block pool = Available − Weights − Activations
             = 72 GB − 16 GB − 2 GB
             = 54 GB

Step 5: How many KV blocks does 54 GB give us?
  First, compute bytes per block:
    KV_per_token = 128 KB (from Worked Example 2.3)
    block_size   = 16 tokens per block
    Bytes per block = 128 KB/token × 16 tokens = 2,048 KB = 2 MB

  Number of blocks = 54 GB / 2 MB per block
                   = 54,000 MB / 2 MB
                   = 27,000 blocks

Step 6: How many concurrent users can this support?
  Each user needs enough blocks for their current context.
  A user mid-conversation at 1,000 tokens needs:
    1,000 tokens / 16 tokens per block = 63 blocks

  Users supportable at 1,000 tokens average context:
    = 27,000 blocks / 63 blocks per user
    = ~428 users

  Users supportable at 4,096 tokens average context:
    4,096 / 16 = 256 blocks per user
    27,000 / 256 ≈ 105 users

  Users supportable at 8,192 tokens average context:
    8,192 / 16 = 512 blocks per user
    27,000 / 512 ≈ 52 users

Final answer:
  Llama 3 8B FP16 on H100 with gpu_memory_utilization=0.90:
  ─ 54 GB for KV cache blocks
  ─ ~27,000 KV blocks
  ─ ~50–400 concurrent users depending on average context length
─────────────────────────────────────────────────────────────────────
```

### ASCII Diagram: vLLM HBM Partition Map

```
  vLLM HBM LAYOUT — Llama 3 8B on H100 SXM (80 GB)
  ═══════════════════════════════════════════════════════════════════

  0 GB                                                          80 GB
  │                                                                 │
  ▼                                                                 ▼
  ┌──────────────────┬──────┬──────────────────────────────┬───────┐
  │  Model Weights   │Activ.│   KV Cache Block Pool        │Headrm │
  │  (FP16)          │      │   (27,000 blocks × 2 MB)     │       │
  │  ████████████████│██████│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│       │
  │      16 GB       │ 2 GB │           54 GB              │  8 GB │
  └──────────────────┴──────┴──────────────────────────────┴───────┘
           ↑              ↑                    ↑                  ↑
     loaded once     measured by          dynamic,           safety
     at startup      dummy fwd pass       PagedAttention     margin
                                          manages this       (10%)

  Each KV block (2 MB) holds 16 tokens for all layers.
  Blocks are allocated per-user by the BlockManager.
  Freed blocks return to the pool when a request completes.

  KEY: ████ = permanently allocated   ░░░░ = dynamically managed
  ═══════════════════════════════════════════════════════════════════
```

### The `gpu_memory_utilization` Parameter

This is the most impactful single vLLM parameter. It controls how much of the GPU's HBM is claimed for the entire vLLM process.

```
WORKED EXAMPLE 2.6 — Effect of gpu_memory_utilization on block count
─────────────────────────────────────────────────────────────────────
Given: Llama 3 8B FP16 on H100 (80 GB), weights=16 GB, activations=2 GB

  gpu_memory_utilization │ Available │ KV Pool │ Blocks  │ Max users
  ───────────────────────┼───────────┼─────────┼─────────┼──────────
          0.70           │  56 GB    │  38 GB  │ 19,000  │    37
          0.80           │  64 GB    │  46 GB  │ 23,000  │    45
          0.85           │  68 GB    │  50 GB  │ 25,000  │    49
          0.90           │  72 GB    │  54 GB  │ 27,000  │    52
          0.95           │  76 GB    │  58 GB  │ 29,000  │    56
          1.00           │  80 GB    │  62 GB  │ 31,000  │    60  ← RISKY

  (Max users computed at 8,192 token average context)

  Why not always use 1.00?
  Other GPU processes (monitoring, drivers) use small amounts of HBM.
  Unexpected activation spikes during unusual inputs can exceed estimate.
  OOM (Out of Memory) crash with 1.00 is a real risk.
  0.85–0.90 is the typical safe range.
─────────────────────────────────────────────────────────────────────
```

---

## 2.5 llama.cpp's Memory Budget: mmap and Partial Offload

`[FOUNDATIONAL]`

### How llama.cpp Loads Models: mmap

**Memory-mapped files (mmap)** are an operating system feature that maps a file on disk into the process's virtual memory address space. When you access a memory address in the mapped region, the OS reads the corresponding bytes from disk — but only when you actually access them, and only the pages you need.

For llama.cpp, this means:

```
  VLLM MODEL LOADING vs. llama.cpp MMAP LOADING
  ═══════════════════════════════════════════════════════════════════

  vLLM (eager loading):
  ┌──────────┐   read entire    ┌───────────┐   copy to    ┌──────┐
  │ HF Weight│ ──────────────▶ │ System RAM│ ──────────▶ │ HBM  │
  │  Files   │   (sequential,   │  (staging)│  (PCIe DMA) │ (GPU)│
  └──────────┘    slow)         └───────────┘             └──────┘
  Time: 30–140 seconds for large models (dominates cold start)

  llama.cpp (mmap loading):
  ┌──────────┐   mmap()        ┌───────────┐
  │  GGUF    │ ──────────────▶ │ Virtual   │
  │  File    │   (instant,      │ Memory    │ ← pages loaded
  └──────────┘  just mapping)   │  Space    │   on demand by OS
                                └───────────┘
  Time: 1–5 seconds (just parsing metadata + mapping)

  Trade-off: mmap is fast to start but the first pass over the model
  may be slower as the OS pages in data from disk (page faults).
  Subsequent passes use the OS file cache — much faster.
  ═══════════════════════════════════════════════════════════════════
```

### llama.cpp Memory Layout

llama.cpp's memory is split between CPU RAM and GPU VRAM depending on how many layers are offloaded.

```
  LLAMA.CPP MEMORY LAYOUT — n_gpu_layers controls the split
  ═══════════════════════════════════════════════════════════════════

  n_gpu_layers = 0  (CPU only, no GPU):
  ┌──────────────────────────────────────────────────┐
  │  System DRAM                                     │
  │  ┌────────────────────┐  ┌─────────────────────┐ │
  │  │ Weights (all layers│  │ KV cache ring buffer│ │
  │  │  via mmap)         │  │ (pre-allocated)     │ │
  │  │ ████████████████   │  │ ░░░░░░░░░░░░░░░░░   │ │
  │  └────────────────────┘  └─────────────────────┘ │
  └──────────────────────────────────────────────────┘
  GPU not used at all.

  ─────────────────────────────────────────────────────────────────

  n_gpu_layers = 20  (partial offload, 20 layers on GPU):
  ┌──────────────────────────────────────────────────┐
  │  System DRAM                                     │
  │  ┌─────────────────────┐  ┌────────────────────┐ │
  │  │ Layers 21–32 weights│  │  KV cache (CPU     │ │
  │  │  (remaining layers) │  │  portion, if any)  │ │
  │  └─────────────────────┘  └────────────────────┘ │
  └──────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────┐
  │  GPU VRAM                                        │
  │  ┌─────────────────────┐  ┌────────────────────┐ │
  │  │ Layers 1–20 weights │  │  KV cache (GPU     │ │
  │  │                     │  │  portion)          │ │
  │  └─────────────────────┘  └────────────────────┘ │
  └──────────────────────────────────────────────────┘
  Each forward pass: data moves CPU → GPU for upper layers.
  PCIe bandwidth (~32 GB/s) becomes the bottleneck.

  ─────────────────────────────────────────────────────────────────

  n_gpu_layers = 99  (full GPU offload, all layers on GPU):
  ┌──────────────────────────────────────────────────┐
  │  System DRAM                                     │
  │  ┌─────────────────────────────────────────────┐ │
  │  │ Weights (mmap, paged in lazily)             │ │
  │  │ NOT actively used during inference          │ │
  │  └─────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────┐
  │  GPU VRAM                                        │
  │  ┌─────────────────────┐  ┌────────────────────┐ │
  │  │ All layer weights   │  │ Full KV cache ring │ │
  │  │  (copied from mmap) │  │  buffer            │ │
  │  └─────────────────────┘  └────────────────────┘ │
  └──────────────────────────────────────────────────┘
  GPU runs the full forward pass. Fastest configuration.
  ═══════════════════════════════════════════════════════════════════
```

### Worked Example: 7B Q4_K_M on RTX 4090 with Full Offload

```
WORKED EXAMPLE 2.7 — Llama 3 8B Q4_K_M on RTX 4090 (24 GB VRAM)
─────────────────────────────────────────────────────────────────────
Given:
  GPU:        RTX 4090, 24 GB GDDR6X
  Model:      Llama 3 8B, quantized to Q4_K_M (GGUF)
  n_ctx:      4,096 tokens (desired context window)
  n_gpu_layers: 99 (all layers on GPU)

Step 1: Weight memory at Q4_K_M.
  Approximate: 8 × 10⁹ × 0.5 bytes = 4 GB
  Actual Q4_K_M (from HuggingFace): ~4.58 GB
  We use 4.6 GB.

Step 2: KV cache memory at n_ctx = 4,096.
  From Worked Example 2.3: 128 KB per token for Llama 3 8B (FP16)
  But llama.cpp can use different KV cache types.
  Default is FP16:
    KV cache = 128 KB/token × 4,096 tokens = 512 MB = 0.5 GB

Step 3: Activation and scratch memory.
  GGML allocates computation scratch buffers based on model size.
  For an 8B model: approximately 0.3 GB for scratch.

Step 4: Total VRAM requirement.
  Total = Weights + KV cache + Scratch
        = 4.6 GB + 0.5 GB + 0.3 GB
        = 5.4 GB

Step 5: Compare to RTX 4090 VRAM.
  Available: 24 GB
  Required:   5.4 GB
  Remaining: 18.6 GB  ← comfortable margin

  This means n_ctx can be increased significantly:
  At 128 KB/token, remaining 18.6 GB can hold:
    18.6 GB / 128 KB = 145,312 tokens of additional KV cache
  You could comfortably set n_ctx = 32,768 or higher.

Final answer:
  Llama 3 8B Q4_K_M fits on an RTX 4090 (24 GB) with enormous
  KV cache headroom. The bottleneck is not memory but GDDR6X
  bandwidth (1.01 TB/s) — faster for this use case than loading
  FP16 weights (16 GB vs 4.6 GB = 3.5× more bandwidth).
─────────────────────────────────────────────────────────────────────
```

### Worked Example: 70B Q4_K_M on Apple M2 Ultra

```
WORKED EXAMPLE 2.8 — Llama 3 70B Q4_K_M on Apple M2 Ultra (192 GB)
─────────────────────────────────────────────────────────────────────
Given:
  Hardware:   Apple M2 Ultra, 192 GB unified memory
  Bandwidth:  800 GB/s (shared CPU + GPU)
  Model:      Llama 3 70B, Q4_K_M
  n_ctx:      8,192 tokens

Step 1: Weight memory at Q4_K_M.
  70 × 10⁹ × 0.5 bytes ≈ 35 GB
  Actual Q4_K_M: ~39 GB
  We use 39 GB.

Step 2: KV cache at 8,192 tokens.
  From Worked Example 2.4: 320 KB/token for Llama 3 70B (FP16)
  KV cache = 320 KB × 8,192 = 2,621 MB ≈ 2.56 GB

Step 3: Scratch memory.
  ≈ 1 GB for a 70B model.

Step 4: Total memory.
  Total = 39 GB + 2.56 GB + 1 GB = 42.56 GB

Step 5: Compare to M2 Ultra.
  Available: 192 GB
  Required:  42.56 GB
  Remaining: 149.4 GB

  At 320 KB/token, remaining memory supports:
    149 GB / 320 KB = ~465,000 additional tokens of KV cache
  n_ctx = 131,072 would use 320 KB × 131,072 = 40 GB additional
  Total with 131K context: 39 + 40 + 1 = 80 GB — still fits easily.

Step 6: Decode speed estimate.
  Decode bandwidth need: 39 GB weights loaded per token
  Available bandwidth: 800 GB/s
  Theoretical token speed: 800 / 39 ≈ 20 tokens/sec

  (Reality: ~15–20 tok/s on M2 Ultra, close to theoretical)

Final answer:
  70B Q4_K_M fits on M2 Ultra with 149 GB to spare.
  Can support 131K token contexts.
  Decode speed ~15–20 tok/s — faster than FP16 on 2× A100 80GB
  at batch=1, because unified memory eliminates PCIe overhead.
─────────────────────────────────────────────────────────────────────
```

---

## 2.6 Memory Budget Calculator: Python

The following Python script computes the complete memory budget for any model, for both vLLM and llama.cpp configurations.

```python
# memory_budget.py
# Chapter 2 — Close to the Metal: LLM Inference from First Principles
#
# No external dependencies — runs anywhere with Python 3.8+
#
# Usage:
#   python memory_budget.py
#   (Edit the MODEL_CONFIGS dict to add your own models)

from dataclasses import dataclass
from typing import Optional

# ── Data structures ──────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Architecture parameters for a transformer model."""
    name:        str
    n_params:    float   # total parameters (in billions)
    n_layers:    int
    n_kv_heads:  int     # GQA KV heads (may be less than n_heads)
    head_dim:    int     # dimension of each attention head
    n_heads:     int     # total query heads (for reference)

@dataclass
class HardwareConfig:
    """Memory specs for a piece of hardware."""
    name:      str
    memory_gb: float    # total HBM / VRAM / unified memory in GB
    bandwidth_tb_s: float  # memory bandwidth in TB/s

# ── Model registry ───────────────────────────────────────────────────

MODEL_CONFIGS = {
    "llama3-1b":  ModelConfig("Llama 3.2 1B",   1.24, 16,  8,  64, 32),
    "llama3-8b":  ModelConfig("Llama 3 8B",      8.03, 32,  8, 128, 32),
    "llama3-70b": ModelConfig("Llama 3 70B",    70.6,  80,  8, 128, 64),
    "mistral-7b": ModelConfig("Mistral 7B",      7.24, 32,  8, 128, 32),
}

HARDWARE_CONFIGS = {
    "h100-sxm":     HardwareConfig("H100 SXM 80GB",  80.0, 3.35),
    "a100-80gb":    HardwareConfig("A100 SXM 80GB",  80.0, 2.00),
    "rtx4090":      HardwareConfig("RTX 4090 24GB",  24.0, 1.01),
    "m2-ultra":     HardwareConfig("M2 Ultra 192GB", 192.0, 0.80),
    "m3-max":       HardwareConfig("M3 Max 128GB",   128.0, 0.40),
}

# Bytes per parameter for each precision
BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,   # approximate; actual GGUF Q4_K_M is ~0.57
}

# ── Calculation functions ─────────────────────────────────────────────

def weight_memory_gb(model: ModelConfig, precision: str) -> float:
    """How many GB do the model weights occupy?"""
    bytes_per_param = BYTES_PER_PARAM[precision]
    total_bytes = model.n_params * 1e9 * bytes_per_param
    return total_bytes / (1024 ** 3)  # convert to GB

def kv_cache_per_token_kb(model: ModelConfig,
                           kv_precision: str = "fp16") -> float:
    """
    KV cache bytes per token, in KB.
    Formula: 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
    """
    bytes_per_elem = BYTES_PER_PARAM.get(kv_precision, 2.0)
    total_bytes = (2 * model.n_layers * model.n_kv_heads
                   * model.head_dim * bytes_per_elem)
    return total_bytes / 1024  # bytes → KB

def kv_cache_for_context_gb(model: ModelConfig,
                              context_len: int,
                              kv_precision: str = "fp16") -> float:
    """Total KV cache GB for a single user at a given context length."""
    per_token_kb = kv_cache_per_token_kb(model, kv_precision)
    total_kb = per_token_kb * context_len
    return total_kb / (1024 ** 2)  # KB → GB

def vllm_budget(model: ModelConfig,
                hw: HardwareConfig,
                precision: str = "fp16",
                gpu_memory_utilization: float = 0.90,
                context_len: int = 8192,
                block_size: int = 16,
                activation_gb: float = 2.0) -> dict:
    """
    Compute the full vLLM HBM budget.
    Returns a dict with all budget components.
    """
    available_gb   = hw.memory_gb * gpu_memory_utilization
    weights_gb     = weight_memory_gb(model, precision)
    kv_pool_gb     = available_gb - weights_gb - activation_gb
    kv_block_gb    = kv_cache_per_token_kb(model) * block_size / (1024**2)
    n_blocks       = kv_pool_gb / kv_block_gb
    tokens_per_user = context_len
    blocks_per_user = tokens_per_user / block_size
    max_users       = n_blocks / blocks_per_user

    # Theoretical decode speed
    decode_tok_s = (hw.bandwidth_tb_s * 1024) / weights_gb  # GB/s / GB = tok/s

    return {
        "available_gb":       round(available_gb, 1),
        "weights_gb":         round(weights_gb, 1),
        "activation_gb":      round(activation_gb, 1),
        "kv_pool_gb":         round(kv_pool_gb, 1),
        "kv_per_token_kb":    round(kv_cache_per_token_kb(model), 1),
        "kv_block_gb":        round(kv_block_gb * 1024, 1),  # show in MB
        "n_blocks":           int(n_blocks),
        "max_users":          int(max_users),
        "decode_tok_s_batch1": round(decode_tok_s, 0),
    }

# ── Main: print budgets ───────────────────────────────────────────────

def print_vllm_budget(model_key: str, hw_key: str,
                       precision: str = "fp16",
                       context_len: int = 8192):
    model = MODEL_CONFIGS[model_key]
    hw    = HARDWARE_CONFIGS[hw_key]
    b     = vllm_budget(model, hw, precision, context_len=context_len)

    print(f"\n{'═'*60}")
    print(f"  vLLM Budget: {model.name} [{precision}] on {hw.name}")
    print(f"  Context length: {context_len:,} tokens")
    print(f"{'═'*60}")
    print(f"  Total HBM available (90% util): {b['available_gb']:>6.1f} GB")
    print(f"  ├─ Model weights:               {b['weights_gb']:>6.1f} GB")
    print(f"  ├─ Activation workspace:        {b['activation_gb']:>6.1f} GB")
    print(f"  └─ KV cache block pool:         {b['kv_pool_gb']:>6.1f} GB")
    print(f"")
    print(f"  KV cache per token:   {b['kv_per_token_kb']:>6.1f} KB")
    print(f"  Bytes per block ({16} tok): {b['kv_block_gb']:>6.1f} MB")
    print(f"  Total blocks:         {b['n_blocks']:>6,}")
    print(f"  Max concurrent users: {b['max_users']:>6,}")
    print(f"  Decode speed (batch=1): ~{b['decode_tok_s_batch1']:.0f} tok/s")
    print(f"{'═'*60}")

if __name__ == "__main__":
    # Example 1: Llama 3 8B FP16 on H100
    print_vllm_budget("llama3-8b", "h100-sxm", "fp16", 8192)

    # Example 2: Llama 3 70B FP16 on H100 (will show memory pressure)
    print_vllm_budget("llama3-70b", "h100-sxm", "fp16", 8192)

    # Example 3: Llama 3 70B INT4 on H100 (shows quantization benefit)
    print_vllm_budget("llama3-70b", "h100-sxm", "int4", 8192)

    # Example 4: Llama 3 8B FP16 on RTX 4090 (24 GB)
    print_vllm_budget("llama3-8b", "rtx4090", "fp16", 4096)
```

**Expected output (excerpt):**
```
════════════════════════════════════════════════════════════
  vLLM Budget: Llama 3 8B [fp16] on H100 SXM 80GB
  Context length: 8,192 tokens
════════════════════════════════════════════════════════════
  Total HBM available (90% util):   72.0 GB
  ├─ Model weights:                 16.0 GB
  ├─ Activation workspace:           2.0 GB
  └─ KV cache block pool:           54.0 GB

  KV cache per token:              128.0 KB
  Bytes per block (16 tok):          2.0 MB
  Total blocks:                   27,000
  Max concurrent users:               52
  Decode speed (batch=1):          ~215 tok/s
════════════════════════════════════════════════════════════
```

---

## 2.7 llama.cpp Context Parameters: C++ Walkthrough

The following C++ code shows how llama.cpp's context parameters map to the memory calculations we have done in this chapter. Every field is annotated with what it controls and how it affects the memory budget.

```cpp
// memory_layout_demo.cpp
// Chapter 2 — Close to the Metal: LLM Inference from First Principles
//
// This file demonstrates the memory implications of llama_context_params.
// It does NOT run inference — it prints the computed memory budget
// using llama.cpp's own size query functions.
//
// Build: same as hello_llamacpp.cpp from Chapter 1.
// Run:   ./memory_layout_demo <model.gguf>

#include "llama.h"
#include <cstdio>

// Helper: print a size in human-readable form
static void print_size(const char* label, size_t bytes) {
    if (bytes >= 1024ULL * 1024 * 1024)
        printf("  %-35s %6.2f GB\n", label,
               bytes / (1024.0 * 1024.0 * 1024.0));
    else if (bytes >= 1024 * 1024)
        printf("  %-35s %6.1f MB\n", label,
               bytes / (1024.0 * 1024.0));
    else
        printf("  %-35s %6.1f KB\n", label, bytes / 1024.0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    // ── Load the model ───────────────────────────────────────────────
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;   // full GPU offload (if VRAM available)

    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "Load failed\n"); return 1; }

    // ── Inspect model architecture ───────────────────────────────────
    // These functions read the GGUF metadata — no computation needed.
    // They map directly to the parameters in our KV cache formula.
    const llama_model_meta* meta = llama_model_meta(model);
    int n_layers   = (int)llama_model_n_layer(model);
    int n_heads    = (int)llama_model_n_head(model);
    int n_kv_heads = (int)llama_model_n_head_kv(model);
    int embd_dim   = (int)llama_model_n_embd(model);
    int head_dim   = embd_dim / n_heads;  // derived

    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  Model Architecture                              ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  n_layers:    %d\n", n_layers);
    printf("  n_heads:     %d  (query heads)\n", n_heads);
    printf("  n_kv_heads:  %d  (KV heads — GQA if < n_heads)\n", n_kv_heads);
    printf("  embd_dim:    %d  (hidden dimension)\n", embd_dim);
    printf("  head_dim:    %d  (embd_dim / n_heads)\n", head_dim);

    // ── Compute KV cache formula ─────────────────────────────────────
    // KV bytes per token = 2 × n_layers × n_kv_heads × head_dim × 2
    // The final ×2 is for FP16 (2 bytes per element).
    size_t kv_bytes_per_token = (size_t)2 * n_layers * n_kv_heads
                                 * head_dim * 2;  // FP16

    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  KV Cache Formula                                ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  KV bytes per token = 2 × %d × %d × %d × 2\n",
           n_layers, n_kv_heads, head_dim);
    printf("                     = %zu bytes\n", kv_bytes_per_token);
    printf("                     = %.1f KB/token\n",
           kv_bytes_per_token / 1024.0);

    // ── Create contexts with different n_ctx and compare ────────────
    // llama_context_params controls all the memory knobs.
    // We create two contexts to compare their KV cache sizes.

    struct TestCase { int n_ctx; const char* label; };
    TestCase cases[] = {
        {  2048, "n_ctx =  2048 (conservative)" },
        {  8192, "n_ctx =  8192 (typical)" },
        { 32768, "n_ctx = 32768 (extended)" },
    };

    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  KV Cache Memory at Different Context Lengths    ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    for (auto& tc : cases) {
        llama_context_params cp = llama_context_default_params();

        cp.n_ctx    = tc.n_ctx;
        cp.n_batch  = 512;
        cp.n_ubatch = 512;

        // n_seq_max is the maximum number of sequences (users) the
        // context can track simultaneously. Each sequence has its own
        // slice of the KV ring buffer.
        // Default = 1 (single user). For multi-user: set higher.
        cp.n_seq_max = 1;

        // Compute KV cache size without creating a context:
        size_t kv_total = kv_bytes_per_token * (size_t)tc.n_ctx;

        printf("\n  %s\n", tc.label);
        print_size("KV cache total:", kv_total);
        printf("  %-35s %6.0f\n", "KV blocks (at 16 tok/block):",
               (double)tc.n_ctx / 16.0);
    }

    // ── The n_ctx vs. n_seq_max distinction ─────────────────────────
    // n_ctx:     total token slots in the KV ring buffer
    //            = max_tokens_per_sequence × n_seq_max
    // n_seq_max: maximum number of sequences tracked simultaneously
    //
    // Example: n_ctx=4096, n_seq_max=4
    //   → Each of 4 users can have up to 1024 tokens
    //   → Total KV buffer = 4096 tokens worth
    //
    // vLLM handles this differently: it allocates blocks dynamically
    // per-user. llama.cpp pre-allocates the entire buffer upfront.
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  n_ctx vs n_seq_max: Multi-User in llama.cpp     ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  Setting: n_ctx=4096, n_seq_max=4\n");
    printf("  → KV buffer = 4096 token slots total\n");
    printf("  → Each user gets at most 4096/4 = 1024 tokens\n");
    printf("  → Contrast: vLLM allocates blocks dynamically\n");

    size_t kv_4096_1user  = kv_bytes_per_token * 4096;
    size_t kv_4096_4users = kv_bytes_per_token * 4096; // same total!
    print_size("  n_ctx=4096, n_seq_max=1:", kv_4096_1user);
    print_size("  n_ctx=4096, n_seq_max=4:", kv_4096_4users);
    printf("  (Same memory — n_seq_max just changes the token\n");
    printf("   budget per user, not the total allocation)\n");

    // ── Cleanup ──────────────────────────────────────────────────────
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
```

**Expected output for Llama 3 8B:**
```
╔══════════════════════════════════════════════════╗
║  Model Architecture                              ║
╚══════════════════════════════════════════════════╝
  n_layers:    32
  n_heads:     32  (query heads)
  n_kv_heads:   8  (KV heads — GQA if < n_heads)
  embd_dim:  4096  (hidden dimension)
  head_dim:   128  (embd_dim / n_heads)

╔══════════════════════════════════════════════════╗
║  KV Cache Formula                                ║
╚══════════════════════════════════════════════════╝
  KV bytes per token = 2 × 32 × 8 × 128 × 2
                     = 131072 bytes
                     = 128.0 KB/token

╔══════════════════════════════════════════════════╗
║  KV Cache Memory at Different Context Lengths    ║
╚══════════════════════════════════════════════════╝

  n_ctx =  2048 (conservative)
  KV cache total:                  256.0 MB
  KV blocks (at 16 tok/block):     128.0

  n_ctx =  8192 (typical)
  KV cache total:                 1024.0 MB
  KV blocks (at 16 tok/block):     512.0

  n_ctx = 32768 (extended)
  KV cache total:                 4096.0 MB
  KV blocks (at 16 tok/block):    2048.0
```

---

## 2.8 Putting It Together: A Decision Walkthrough

Before moving on, let us use everything from this chapter to answer a realistic planning question.

```
WORKED EXAMPLE 2.9 — "Will this model fit, and how many users can I serve?"
─────────────────────────────────────────────────────────────────────
Question:
  A team wants to serve Llama 3 70B to internal users.
  They have two options:
  (A) Two H100 80GB GPUs with vLLM FP16
  (B) One Apple M2 Ultra (192 GB) with llama.cpp Q4_K_M
  They expect 20 concurrent users with 4,096-token average contexts.
  Which option works, and what are the trade-offs?

OPTION A: 2× H100 + vLLM FP16
─────────────────────────────────────────────────────────────────────
Step 1: Total HBM = 2 × 80 GB = 160 GB
Step 2: Available at gpu_memory_utilization=0.90: 144 GB
Step 3: Weights (FP16, 70B): 70 × 2 = 140 GB
Step 4: Remaining: 144 − 140 − 2 (activations) = 2 GB for KV pool
Step 5: KV per token (70B): 320 KB/token
Step 6: KV cache per user at 4,096 tokens: 320 KB × 4,096 = 1.25 GB
Step 7: Users supportable: 2 GB / 1.25 GB = 1.6 → only 1 user!

PROBLEM: 2× H100 FP16 can barely serve 1 concurrent user at 4K context.
SOLUTION: Use INT8 or INT4 weights.

Step 3 (INT8): 70 × 1 = 70 GB weights
Step 4 (INT8): 144 − 70 − 2 = 72 GB for KV pool
Step 7 (INT8): 72 GB / 1.25 GB = 57 users  ✓

Step 3 (INT4): 70 × 0.5 = 35 GB weights
Step 4 (INT4): 144 − 35 − 2 = 107 GB for KV pool
Step 7 (INT4): 107 GB / 1.25 GB = 85 users ✓

OPTION B: M2 Ultra + llama.cpp Q4_K_M
─────────────────────────────────────────────────────────────────────
Step 1: Total unified memory = 192 GB
Step 2: Weights (Q4_K_M): ~39 GB
Step 3: KV cache for 20 users × 4,096 tokens:
        20 × 320 KB × 4,096 = 20 × 1.25 GB = 25 GB
Step 4: Scratch: ~1 GB
Step 5: Total: 39 + 25 + 1 = 65 GB
Step 6: Available (192 GB): comfortable margin of 127 GB ✓

COMPARISON:
─────────────────────────────────────────────────────────────────────
Dimension            │ 2× H100 + vLLM (INT8)    │ M2 Ultra + llama.cpp
─────────────────────┼──────────────────────────┼────────────────────
Fits 20 users?       │ Yes (57 max)              │ Yes (comfortably)
Decode speed (batch=20)│ ~1,200 tok/s total      │ ~15 tok/s total
Throughput at scale  │ Far superior (batch >>1)  │ Limited
Cost (cloud)         │ ~$16/hr (2× H100 on-demand│ ~$8,000 hardware
Cold start           │ ~60-140 seconds           │ ~10 seconds
Quality              │ Slight INT8 degradation   │ Slight Q4 degradation
─────────────────────────────────────────────────────────────────────

For 20 concurrent users:
  vLLM + 2× H100 (INT8) = high throughput, pay-as-you-go
  llama.cpp + M2 Ultra  = high ownership cost, low latency, offline
─────────────────────────────────────────────────────────────────────
```

---

## Chapter Summary

- **HBM** (GPU memory) has bandwidth 30–60× faster than system DRAM. For inference, everything must fit in HBM (or unified memory on Apple Silicon). System RAM is a last resort.

- **Weight memory** is `n_params × bytes_per_param`. FP16 = 2 bytes, INT4 ≈ 0.5 bytes. A 7B FP16 model needs 14 GB; the same model at INT4 needs ~3.5 GB.

- **KV cache memory per token** = `2 × n_layers × n_kv_heads × head_dim × bytes_per_element`. For Llama 3 8B (FP16) this is 128 KB per token. GQA reduces this by using fewer KV heads than query heads.

- **vLLM splits HBM** into three regions: weights (fixed), activation workspace (measured by dummy forward pass), and KV cache block pool (the remainder). The block pool size determines max concurrent users.

- **llama.cpp uses mmap** so models load in seconds. Memory is split between system RAM and GPU VRAM based on `n_gpu_layers`. The KV cache is a fixed-size contiguous ring buffer allocated at context creation.

- **The key planning calculation:** `KV pool GB ÷ (KV_per_token_KB × context_len / 1024²) = max concurrent users`. Run this before any deployment.

---

## Self-Check Questions

1. An H100 SXM has 3.35 TB/s of memory bandwidth. A server's DRAM has 80 GB/s. How many times faster is HBM than DRAM? Why does this matter for inference? *(Section 2.1)*

2. A model has 13 billion parameters stored in BF16. How many GB of HBM does it require for weights alone? Show the calculation. *(Section 2.2)*

3. A transformer has 40 layers, 8 KV heads, and head_dim = 128. What is the KV cache size per token in bytes (FP16)? In KB? *(Section 2.3)*

4. Using the vLLM HBM budget formula: if you have an A100 80 GB, your model weights are 26 GB (FP16), activations peak at 2 GB, and you set `gpu_memory_utilization=0.85`, how much HBM is left for the KV cache block pool? *(Section 2.4)*

5. Why does llama.cpp's mmap loading start faster than vLLM's eager loading, and what is the trade-off of mmap? *(Section 2.5)*

---

## Where We Go Next

Chapter 3 opens the tokenization pipeline and introduces the batch — the collection of user requests that travel through the model together. We will trace exactly what happens when multiple users' prompts arrive simultaneously: how they are tokenized, how they are packed into a batch, and why the naive approach (wait for all users to arrive before starting) is so much worse than continuous batching. We will compute, step by step, how much GPU time is wasted by static batching and how continuous batching recovers it. Both vLLM and llama.cpp handle this problem — in very different ways.

---

*Code for this chapter: `vllm_book/code/chapter_02/memory_budget.py` and `memory_layout_demo.cpp`*
*Next: Chapter 3 — Tokens, Sequences, and the Batch*


---

## Worked Solutions

---

### Solution 1 — HBM vs DRAM bandwidth comparison

**What we need:** How many times faster HBM is than DRAM, and why it matters.

**Step 1 — Convert to the same units.**

- HBM bandwidth: 3.35 TB/s = 3,350 GB/s
- DRAM bandwidth: 80 GB/s

**Step 2 — Compute the ratio.**

$$\frac{3{,}350 \text{ GB/s}}{80 \text{ GB/s}} = 41.875 \approx 42\times \text{ faster}$$

**Step 3 — Why this matters for inference.**

During every decode step, the GPU must stream the *entire* model weight tensor from wherever it lives:

- If weights live in **HBM** (on-device): streaming 26 GB (13B FP16) takes 26 GB ÷ 3,350 GB/s ≈ **7.8 ms**.
- If weights live in **CPU DRAM** (offloaded): 26 GB ÷ 80 GB/s ≈ **325 ms** — about **42× slower**.

This 42× difference explains why CPU-offloaded inference (e.g., large models partially loaded to RAM) is so much slower than GPU-resident inference: every decode step pays the DRAM bandwidth tax instead of the HBM bandwidth tax.

For llama.cpp on Apple Silicon, unified memory collapses this distinction — the CPU and GPU share the same DRAM at ~100–800 GB/s depending on the chip, which is why M2 Ultra can serve 70B models at reasonable speed without separate HBM.

---

### Solution 2 — Memory footprint for 13B BF16 model

**What we need:** GB of HBM for weights only.

**Step 1 — BF16 size.**

BF16 (Brain Float 16) uses **2 bytes per parameter**. It has the same exponent range as FP32 (8 bits) but only 7 mantissa bits instead of 23. It is the dominant dtype for serving because it avoids the overflow issues of FP16 while using half the memory of FP32.

**Step 2 — Compute.**

$$13 \times 10^9 \text{ parameters} \times 2 \text{ bytes/parameter} = 26 \times 10^9 \text{ bytes} = 26 \text{ GB}$$

**Step 3 — Practical check.**

An A100 80 GB has 80 GB of HBM. A 13B BF16 model occupies 26 GB — leaving 54 GB for KV cache and activations. This is comfortable. A 70B BF16 model would need 140 GB, requiring at least two A100s.

**General formula:**

$$\text{Weight memory (GB)} = \frac{N_{\text{params}} \times \text{bytes\_per\_param}}{10^9}$$

where `bytes_per_param` is 2 for BF16/FP16, 1 for INT8, 0.5 for INT4/NF4.

---

### Solution 3 — KV cache size per token

**What we need:** Bytes per token for the KV cache.

**Step 1 — Identify the components.**

For each layer of the transformer, the KV cache stores two tensors:

- **K (key):** shape `[n_kv_heads, head_dim]` per token
- **V (value):** shape `[n_kv_heads, head_dim]` per token

Given:

- Layers: 40
- KV heads: 8 (GQA — the number of distinct key/value heads, which may be fewer than query heads)
- `head_dim`: 128
- dtype: FP16 → 2 bytes per element

**Step 2 — Compute bytes per token per layer.**

$$\text{KV bytes per token per layer} = 2 \text{ (K+V)} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{dtype bytes}$$
$$= 2 \times 8 \times 128 \times 2 = 4{,}096 \text{ bytes} = 4 \text{ KB per token per layer}$$

**Step 3 — Multiply by number of layers.**

$$\text{KV bytes per token} = 40 \times 4{,}096 = 163{,}840 \text{ bytes} \approx 160 \text{ KB}$$

**Step 4 — Sanity-check with a real model.**

LLaMA-3 8B has 32 layers, 8 KV heads, head_dim=128, BF16:
$$= 32 \times 2 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} = 128 \text{ KB per token}$$

At 4,096-token context this is 128 KB × 4,096 = 512 MB per sequence. Serving 100 concurrent users at this context length would require 50 GB just for KV cache — which is why memory management is the central challenge of LLM serving.

---

### Solution 4 — HBM budget for KV cache (A100 80 GB)

**What we need:** Remaining HBM for KV cache given the three-way split.

**Step 1 — Apply the `gpu_memory_utilization` cap.**

vLLM reserves a fraction of total HBM to prevent OOM errors from unexpected allocations:

$$\text{Usable HBM} = 80 \text{ GB} \times 0.85 = 68 \text{ GB}$$

**Step 2 — Subtract model weights.**

$$68 - 26 = 42 \text{ GB remaining}$$

**Step 3 — Subtract peak activation buffers.**

Activations (intermediate tensors during forward pass) peak at around 1–3 GB depending on batch size and model architecture:

$$42 - 2 = 40 \text{ GB remaining for KV cache pool}$$

**Step 4 — Interpret.**

40 GB of KV cache pool with 160 KB per token (from Solution 3):

$$\text{Max tokens} = \frac{40 \times 1{,}024^3}{163{,}840} \approx \frac{42{,}949{,}672{,}960}{163{,}840} \approx 262{,}144 \text{ tokens}$$

At 4,096 max context per user: 262,144 ÷ 4,096 ≈ **64 concurrent users**. This is the theoretical maximum, assuming full 4K context utilization. Real workloads with shorter average context would serve more users.

**Step 5 — Why the buffer matters.**

The 15% buffer (0.85 utilization) is crucial. Without it, a sudden batch of longer-than-expected sequences would trigger OOM, killing the server process. vLLM measures the peak activation footprint at startup and adjusts the KV pool accordingly.

---

### Solution 5 — Why mmap starts faster and what it trades away

**What we need:** Two-part answer — mechanism of speed, and the trade-off.

**Step 1 — What mmap does.**

`mmap()` is a POSIX syscall that instructs the OS to:

1. Create a mapping from a range of virtual memory addresses → a file on disk
2. Update the process's page table
3. Return immediately — *no bytes are read from disk*

The key is that the OS is making a **promise**: "when you access address X, I will load the corresponding bytes from disk." But it doesn't make good on that promise until you actually access the address.

**Step 2 — Why this is fast at startup.**

For a 70B model (140 GB file), eager loading must:

- Read 140 GB from NVMe → 140 GB ÷ 7 GB/s ≈ 20 s
- Transfer to HBM → 140 GB ÷ (PCIe 4.0 × 64: ~64 GB/s) ≈ 2 s
- Total: ~22 seconds minimum

With mmap:

- Syscall + page table update: **< 1 millisecond**

**Step 3 — The trade-off: page faults.**

When the first inference request arrives and the forward pass accesses a weight page not yet in RAM, the CPU raises a **hardware page fault**:

1. Execution halts for that core
2. OS issues a read from NVMe for the missing page (4 KB)
3. OS updates the page table and resumes execution

Each page fault adds 0.1–50 ms of latency depending on storage speed. A cold model inference can trigger hundreds of page faults, causing the first few requests to be very slow.

**Step 4 — The warm state.**

After all pages have been accessed at least once (the model is "warm"), pages are cached in DRAM and mmap performs similarly to eager loading. The page fault cost is a one-time initialization cost paid by the first users.

**Trade-off summary:**

| | Eager (vLLM) | mmap (llama.cpp) |
|---|---|---|
| Server startup | Slow (20–60 s) | Fast (< 1 s) |
| First request (cold) | Fast | Slow (page faults) |
| First request (warm) | Fast | Fast |
| Memory usage | All weights in RAM immediately | On-demand |
| Best for | High-throughput servers | Laptops, dev, edge |

