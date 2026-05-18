# Chapter 34: DeepSeek — MLA, Mixture of Experts, and FP8 at Scale

> *"DeepSeek-V3 proved that you can train a frontier model for $5M instead of $100M — not by cutting corners, but by rethinking every assumption about architecture and training."*

---

**What you will understand after this chapter:**

- How Multi-head Latent Attention (MLA) compresses the KV cache by 5–13× without quality loss
- How DeepSeek's 256-expert MoE routes tokens and why 8 routed + 1 shared expert fire per token
- Why FP8 training/inference is practical now and what it gains on H100/H200
- How to serve DeepSeek-V2/V3/R1 with vLLM and llama.cpp

**What you need first:**

- Chapter 4 (Attention Mechanics) — KV cache structure and GQA
- Chapter 10 (Quantization) — FP8 basics
- Chapter 15 (Multi-GPU Serving) — tensor parallelism

---

## 34.1 Why DeepSeek Is Architecturally Different

Every model in this book up to Chapter 33 uses standard Multi-Head Attention (MHA) or Grouped-Query Attention (GQA). Both cache K and V tensors at every layer for every token in the context.

DeepSeek-V2, released in May 2024, introduced **Multi-head Latent Attention (MLA)** — a fundamentally different KV cache structure based on low-rank projection. DeepSeek-V3, released December 2024, scaled MLA to 671B parameters with a 256-expert MoE architecture at a reported training cost of $5.576M on H800 clusters. DeepSeek-R1, released January 2025, applied reinforcement learning to produce long chain-of-thought reasoning.

The three architectural innovations that set DeepSeek apart:

```
  Three DeepSeek Innovations
  
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. MLA — Multi-head Latent Attention                           │
  │    KV cache size: Llama 70B = 1,280 KB/token                  │
  │    DeepSeek-V3:  =  576 KB/token  (55% smaller)               │
  │    At 128K context: saves 90 GB HBM on 8-GPU cluster          │
  ├─────────────────────────────────────────────────────────────────┤
  │ 2. MoE — 256 Experts, Top-8 Routing + 1 Shared Expert         │
  │    671B total params, only 37B active per token                │
  │    Throughput: similar to a 37B dense model on same hardware   │
  │    Quality:    competes with GPT-4o, Claude Sonnet             │
  ├─────────────────────────────────────────────────────────────────┤
  │ 3. FP8 Training + Inference                                    │
  │    Mixed precision: FP8 for matmuls, BF16/FP32 for accum       │
  │    H100 FP8 throughput: 2× FP16 tensor cores                  │
  │    Training cost reduction: ~30% vs BF16                       │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 34.2 Multi-head Latent Attention — Deep Dive

`[FOUNDATIONAL]` MLA is the defining architectural innovation of DeepSeek. Understanding it requires revisiting the KV cache derivation from Chapter 4.

### 34.2.1 Standard KV Cache Review

In standard MHA with $H$ heads, head dimension $d_h$, and $n_{\text{layers}}$ layers, the KV cache for one token is:

```
kv_bytes_per_token = 2 × n_layers × H × d_h × dtype_bytes
                       ↑              ↑     ↑
                      K+V           heads  head dim
```

For Llama 3.1 70B (80 layers, 8 KV heads, head_dim=128, BF16):
```
kv = 2 × 80 × 8 × 128 × 2 = 327,680 bytes = 320 KB/token
```

At 128K context: `128,000 × 320 KB = 40 GB` — nearly the entire H100 VRAM.

### 34.2.2 MLA's Key Insight: Low-Rank KV Compression

Instead of caching K and V separately, MLA projects the hidden state into a **latent vector** of much smaller dimension, then re-expands at attention time.

```
  Standard KV Cache (per layer):
  
  hidden state h [1 × d_model] ──W_K──→ K [1 × (H × d_h)]  cached
                                ──W_V──→ V [1 × (H × d_h)]  cached
  Cache per token: 2 × H × d_h values
  
  MLA KV Cache (per layer):
  
  hidden state h [1 × d_model] ──W_c──→ c [1 × d_c]  ← ONLY THIS IS CACHED
  
  At attention time:
  c ──W_KU──→ K [1 × (H × d_h)]   (recomputed from cached c)
  c ──W_VU──→ V [1 × (H × d_h)]   (recomputed from cached c)
  
  Cache per token: d_c values  (d_c << 2 × H × d_h)
```

For DeepSeek-V3: $d_c = 512$, versus $2 \times H \times d_h = 2 \times 128 \times 128 = 32{,}768$.

```
WORKED EXAMPLE 34.1 — MLA KV Cache Size Comparison
─────────────────────────────────────────────────────────────────────
Standard GQA (Llama 3.1 70B):
  n_layers=80, n_kv_heads=8, head_dim=128, dtype=BF16
  Per token: 2 × 80 × 8 × 128 × 2 = 327,680 bytes = 320 KB
  128K context: 128,000 × 320 KB = 40.0 GB

MLA (DeepSeek-V3):
  n_layers=61, d_c=512 (latent dim), dtype=BF16
  Per token: 61 × 512 × 2 = 62,464 bytes = 61 KB
  128K context: 128,000 × 61 KB = 7.6 GB

MLA also caches decoupled RoPE keys (d_c_rope=64 per layer):
  Per token: 61 × 64 × 2 = 7,808 bytes = 7.6 KB
  Total MLA KV: 61 + 7.6 = 68.6 KB/token

Compression ratio:  320 KB / 68.6 KB = 4.7× smaller KV cache
128K context saving: 40.0 GB − 8.8 GB = 31.2 GB recovered
─────────────────────────────────────────────────────────────────────
```

### 34.2.3 Why It Works — The Math

The attention operation computes:
$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_h}}\right) V$$

In MLA, the Q, K, V for the current token are computed as:
$$Q = h \cdot W_{QU} \cdot W_{QR}$$
$$K = c \cdot W_{KU}$$  
$$V = c \cdot W_{VU}$$

where $c = h \cdot W_c$ is the cached latent vector.

The key observation: you can **absorb** $W_{KU}$ into $W_Q$ at query time:

$$Q K^T = (h W_{QU} W_{QR})(c W_{KU})^T = h (W_{QU} W_{QR} W_{KU}^T) c^T$$

So the inner product is computed without ever materializing full K — only the latent $c$ is needed. This means:

1. **Cache only $c$ per token** (dimension $d_c = 512$)
2. **At attention time**, compute $QK^T$ directly from $Q$ and $c$ (never expand to full K)
3. **V computation** from $c$: $\text{output} = \text{softmax}(QK^T) \cdot c \cdot W_{VU}$, which can also be done without materializing full V

```
  MLA Attention Computation Flow
  
  Prefill (new token):
  
  h_new ──W_c──→ c_new [512]  ← stored in KV cache
         ──W_Q──→ Q_new [H × d_h]
  
  Decode (attend over context):
  
  Q_new [H × d_h]
        × 
  {c_0, c_1, ..., c_{t-1}} [t × 512]  ← KV cache (latents only)
        ↓ (via absorbed W_KU)
  Attention scores [H × t]
        × 
  {c_0, ..., c_{t-1}} [t × 512]  ← reused latents (via W_VU)
        ↓
  Output [H × d_h]
```

`[DEEP DIVE]` MLA also handles positional encoding differently. Standard RoPE is applied to K, but since K isn't stored, DeepSeek uses **decoupled RoPE** — a separate $k_R$ vector of dimension $d_R = 64$ that carries position information and IS stored alongside $c$. This adds the 7.6 KB/token overhead seen in Worked Example 34.1.

---

## 34.3 Mixture of Experts — 256 Experts, Top-8 Routing + 1 Shared Expert

`[FOUNDATIONAL]` DeepSeek-V3 uses MoE for the FFN (Feed-Forward Network) layers. Understanding MoE requires understanding what the FFN does in a standard transformer.

### 34.3.1 Standard FFN Review

In a standard transformer FFN:
```
FFN(x) = W_2 × activation(W_1 × x)
         ↑ [d_model × d_ffn]         ↑ [d_ffn × d_model]
```

For a 70B model: $d_{\text{model}} = 8192$, $d_{\text{ffn}} = 28{,}672$ — the FFN is ~67% of total parameters.

### 34.3.2 MoE: Replace One FFN With 256 Specialized FFNs

```
  Standard FFN                    MoE FFN (DeepSeek-V3)
  
  x [d_model]                     x [d_model]
       │                                │
       ▼                                ▼
  FFN (always active)          Router: compute scores for 256 experts
  W_1: d_model × d_ffn              ↓
  W_2: d_ffn × d_model        Top-8 selection (scores → softmax weights)
       │                        / | | | | | | \
       ▼               Expert 12 47 89 ... 183 (8 active routed experts)
  output [d_model]         ↓    ↓  ↓       ↓
                        weighted sum of 8 expert outputs + shared expert
                                     │
                                     ▼
                               output [d_model]
```

DeepSeek-V3 specifics:

- **256 routed experts** + **1 shared expert** (always active)
- **Top-8 routing**: each token activates exactly 8 of the 256 routed experts (plus the always-active shared expert = 9 total)
- **Expert size**: each expert is a small FFN (~1/4 the size of a standard 7B FFN)
- **Effective active params**: 37B out of 671B total (~5.5% active per forward pass)

```
WORKED EXAMPLE 34.2 — DeepSeek-V3 MoE Parameter Counts
─────────────────────────────────────────────────────────────────────
Architecture:
  n_layers = 61
  d_model  = 7168
  n_experts = 256 (routed) + 1 (shared)
  Expert FFN intermediate: 2048 (one expert's d_ffn)
  Top-k = 8

Per-layer MoE parameter count:
  Router W: d_model × 256 = 7168 × 256 = 1.84M params
  Each expert: 2 × (d_model × d_ffn_expert) = 2 × (7168 × 2048) = 29.4M
  All 256 experts: 256 × 29.4M = 7.52B per layer
  Shared expert: 1 × 29.4M = 29.4M per layer

Total MoE params (61 layers): 61 × 7.55B ≈ 460B
Total attention params (61 layers): ≈ 61 × 3.4B ≈ 210B
Total: ≈ 670B (matches reported 671B)

Active per token per layer (9 active experts: 8 routed + 1 shared):
  Shared expert: 29.4M (always)
  Top-8 routed: 8 × 29.4M = 235.2M
  Active MoE per layer: ≈ 264M params

Note: The DeepSeek-V3 paper reports ~37B active params per forward pass.
The exact per-expert size (intermediate dim, gate projections) yields this
figure when summed across all 61 layers including MLA attention weights;
the above uses simplified expert size estimates for illustration.
─────────────────────────────────────────────────────────────────────
```

### 34.3.3 Expert Parallelism — Why It Matters for Serving

With 256 experts distributed across 8 GPUs (32 experts per GPU), the routing step requires **all-to-all communication**: each GPU must send tokens to whichever GPU holds the selected experts.

```
  Expert Parallelism on 8 GPUs
  
  GPU 0 holds experts 0-31
  GPU 1 holds experts 32-63
  ...
  GPU 7 holds experts 224-255
  
  Token arrives at GPU 2. Router selects expert 17 (GPU 0) and expert 89 (GPU 2).
  
  GPU 2 → GPU 0: send token hidden state (all-to-all)
  GPU 0: run expert 17, return result
  GPU 2: run expert 89 locally, combine results
  
  Communication cost: 2 × d_model × dtype_bytes per token per layer
                    = 2 × 7168 × 2 = 28,672 bytes ≈ 28 KB per token/layer
```

This is why MoE models typically need NVLink between GPUs — PCIe bandwidth makes the all-to-all too slow.

### 34.3.4 Load Balancing

If most tokens route to the same experts, some GPUs are overloaded. DeepSeek-V3 uses an **auxiliary-free load balancing** strategy: instead of adding a loss term that penalizes expert imbalance (which hurts model quality), they use a dynamic biasing mechanism during inference to keep expert utilization balanced without modifying gradients.

---

## 34.4 FP8 Training and Inference

DeepSeek-V3 was trained with FP8 mixed precision — the first publicly documented frontier model to do so successfully.

### 34.4.1 FP8 Number Format

FP8 has two variants used for different purposes:

```
  FP8 E4M3 (4 exponent bits, 3 mantissa bits):
  Range: ±448, precision: ~0.1%
  Used for: weights and activations in forward pass
  
  FP8 E5M2 (5 exponent bits, 2 mantissa bits):
  Range: ±57344, larger range, less precision
  Used for: gradients in backward pass
  
  Comparison:
  ┌──────────┬────────┬──────────┬────────────┬─────────────────┐
  │ Format   │ Bits   │ Max val  │ Precision  │ H100 TFLOPS     │
  ├──────────┼────────┼──────────┼────────────┼─────────────────┤
  │ FP32     │ 32     │ 3.4e38   │ ~0.00001%  │ ~67 TFLOPS      │
  │ BF16     │ 16     │ 3.4e38   │ ~0.8%      │ ~1,979 TFLOPS   │
  │ FP16     │ 16     │ 65504    │ ~0.1%      │ ~1,979 TFLOPS   │
  │ FP8 E4M3 │ 8      │ 448      │ ~1.5%      │ ~3,958 TFLOPS   │
  └──────────┴────────┴──────────┴────────────┴─────────────────┘
```

### 34.4.2 Mixed-Precision Training Strategy

```
  DeepSeek-V3 FP8 Training Flow (one forward pass):
  
  Inputs [BF16] ──scale──→ [FP8 E4M3]
  Weights [BF16] ──scale──→ [FP8 E4M3]
  
  GEMM: FP8 × FP8 ──Tensor Cores──→ FP32 accumulator
                                            │
                                            ▼
                                     BF16 output
                                            │
                                   (next layer input)
  
  Gradient:
  dL/dx [FP8 E5M2] × W [FP8 E4M3] → FP32 accumulator → BF16
  
  Weight update: FP32 master weights (held in memory)
```

The critical insight: **accumulate in FP32**. FP8 × FP8 multiplications go through the H100's tensor cores, which produce FP32 intermediate results before accumulating. This prevents numerical instability while using 2× more compute than BF16.

### 34.4.3 FP8 Inference on H100

For inference, both weights and KV cache are stored in FP8:

```
  FP8 Inference Memory Savings:
  
  BF16 (2 bytes): 671B params × 2 = 1,342 GB
  FP8 (1 byte):   671B params × 1 =   671 GB
  
  8× H100 SXM (80 GB each) = 640 GB total → BF16 671B doesn't fit
  FP8 671B: 671 GB → still doesn't fit on 8 GPUs
  
  Practical serving of DeepSeek-V3:
  - 8× H200 (141 GB): 8 × 141 = 1128 GB → FP8 671B fits with headroom
  - 16× H100 (80 GB): 1280 GB → FP8 671B fits
  - AWQ/GPTQ INT4: 671B × 0.5 = 336 GB → fits on 8× H100 with compression
```

---

## 34.5 DeepSeek-R1 — Reasoning and Chain-of-Thought

DeepSeek-R1 fine-tunes V3 with reinforcement learning (using GRPO from Chapter 24) to produce long reasoning traces. Key serving implications:

**Extended output lengths**: R1 regularly produces 2,000–8,000 output tokens for reasoning problems (vs. ~500 for standard chat). This dominates the decode time.

**KV cache growth**: At 8K output + 2K input, the 10K context KV cache for R1 requires:
```
10,000 tokens × 68.6 KB/token (MLA) = 686 MB per sequence
At batch=16: 10.9 GB just for KV cache
```

**Serving R1 efficiently**: Enable `--max-model-len 32768` (or higher), use chunked prefill, and configure `--max-num-seqs` carefully to avoid OOM on long generations.

---

## 34.6 Serving DeepSeek with vLLM

### 34.6.1 Model Support Status

vLLM added DeepSeek-V2 support in v0.5.0 (July 2024) and V3/R1 support in v0.7.0 (January 2025). The MLA implementation computes attention using the compressed latent representation directly.

```bash
# DeepSeek-V2-Lite (16B, fits on 1× A100)
vllm serve deepseek-ai/DeepSeek-V2-Lite \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --trust-remote-code

# DeepSeek-V3 (671B FP8, needs 8× H200 or 16× H100)
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --max-model-len 131072 \
    --kv-cache-dtype fp8 \
    --trust-remote-code

# DeepSeek-R1 (671B, long context with chunked prefill)
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --trust-remote-code
```

### 34.6.2 MoE-Specific vLLM Configuration

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,
    trust_remote_code=True,
    max_model_len=32768,
    # MoE-specific: expert parallel within tensor parallel
    # vLLM automatically detects MoE and adjusts parallelism
    gpu_memory_utilization=0.90,
    kv_cache_dtype="fp8",          # FP8 KV cache to fit more context
    quantization="fp8",            # FP8 weights
)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=2048,
)

# For R1, allow longer output for reasoning traces
r1_sampling = SamplingParams(
    temperature=0.6,
    max_tokens=8192,    # R1 needs room to reason
)
```

---

## 34.7 Serving DeepSeek with llama.cpp

### 34.7.1 GGUF Conversion for MoE Models

llama.cpp supports DeepSeek-V2/V3 through GGUF. MoE models have a different GGUF structure — each expert's weights are stored as separate tensors:

```bash
# Convert (requires sufficient RAM — V3 FP16 needs ~1.3 TB)
python convert_hf_to_gguf.py deepseek-v3 --outtype f16 --outfile deepseek-v3-f16.gguf

# Quantize to Q4_K_M (practical for most use)
./build/bin/llama-quantize deepseek-v3-f16.gguf deepseek-v3-q4km.gguf Q4_K_M

# For DeepSeek-V2-Lite (practical on consumer hardware)
./build/bin/llama-cli \
    -m deepseek-v2-lite-q4_k_m.gguf \
    -n 512 \
    -c 8192 \
    --n-gpu-layers 27    # offload as many layers as VRAM allows
```

### 34.7.2 MLA in llama.cpp

llama.cpp handles MLA by recomputing K and V from the cached latent at each decode step. This is more compute-intensive than standard GQA decode but uses far less memory — the tradeoff is correct for long-context scenarios.

```
  MLA Decode in llama.cpp:
  
  For each decode step:
  1. Look up cached c_i (512-dim) for all context tokens → fast (small cache)
  2. For current layer:
     K_i = c_i × W_KU  (matrix multiply: [t × 512] × [512 × H×d_h])
     V_i = c_i × W_VU  (same)
  3. Q_new × K^T → attention → output
  
  Extra compute vs GQA: O(t × d_c × H × d_h) per layer
  Memory saved:         5× smaller KV cache
  For t=32K, this is 32K × 512 × 1024 × 2 = 34B FLOPs — non-trivial but worth it
```

---

## 34.8 Full HBM Budget: DeepSeek-V3 on 8× H100

```
WORKED EXAMPLE 34.3 — DeepSeek-V3 Serving on 8× H100 (80 GB each)
─────────────────────────────────────────────────────────────────────
Total VRAM: 8 × 80 GB = 640 GB

Model weights (FP8, 1 byte/param):
  671B × 1 byte = 671 GB
  Per GPU: 671 / 8 = 83.9 GB → EXCEEDS 80 GB per GPU!

Solution 1: AWQ INT4 (0.5 bytes/param):
  671B × 0.5 = 336 GB → 42 GB/GPU → fits with 38 GB headroom

Solution 2: 16× H100 (tensor_parallel=8, pipeline_parallel=2):
  Each GPU holds 671/16 = 41.9 GB weights (FP8)
  Remaining per GPU: 80 - 41.9 = 38.1 GB for KV cache + activations

KV Cache (MLA, FP8, 32K context, batch=8):
  Per sequence: 32,000 × 68.6 KB ≈ 2.2 GB
  8 sequences: 17.6 GB → fits per GPU

Practical minimum for DeepSeek-V3 serving with FP8:
  → 8× H200 (141 GB each) = 1128 GB total → 671 GB weights + 457 GB KV headroom ✓
  → 16× H100 (80 GB each) = 1280 GB → 671 GB + 609 GB headroom ✓
─────────────────────────────────────────────────────────────────────
```

---

## 34.9 DeepSeek-R1 Distilled Models

`[PRODUCTION]`

DeepSeek-R1's reasoning capability can be transferred to smaller dense models through
**distillation** — training a small student model to imitate R1's long chain-of-thought
output. The resulting models are dramatically more deployable than the 671B parent.

### 34.9.1 The Distillation Family

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  DeepSeek-R1 Distill Model Family (all open weights, January 2025)       │
  │                                                                          │
  │  Base model        Parameters  Min GPU           Quantized fits on       │
  │  ─────────────────────────────────────────────────────────────────────  │
  │  Qwen2.5-1.5B      1.5B        CPU only          CPU, RPi 5              │
  │  Qwen2.5-7B        7B          RTX 3070 (8 GB)   RTX 3060 (12 GB)        │
  │  Llama-3.1-8B      8B          RTX 3080 (10 GB)  RTX 3070 (8 GB)         │
  │  Qwen2.5-14B       14B         RTX 3090 (24 GB)  RTX 3080 (10 GB)        │
  │  Qwen2.5-32B       32B         2× RTX 3090       RTX 3090 (24 GB) Q4     │
  │  Llama-3.3-70B     70B         4× A100 (40 GB)   2× RTX 3090 Q4          │
  └──────────────────────────────────────────────────────────────────────────┘
```

These models reason in the same `<think>...</think>` format as R1, with similar
chain-of-thought quality on math and coding tasks — at a fraction of the compute cost.

**Why distillation works so well here:**
The R1 parent model produces long, structured reasoning traces. The student model is
trained via supervised fine-tuning on those traces (not RL). The student learns to emit
the *format* of reasoning, which is sufficient to trigger the LLM's underlying
capabilities — it does not need to rediscover reasoning via RL.

### 34.9.2 Serving Distilled R1 on Consumer Hardware

**Qwen2.5-7B-R1-Distill on RTX 3060 (12 GB) with llama.cpp:**

```bash
# Download quantized model
# HuggingFace: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# GGUF from: bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF

# Key flag choices:
#   --n-gpu-layers 33   all layers on GPU
#   -c 16384            16K context for reasoning
llama-server \
    -m DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
    --n-gpu-layers 33 \
    -c 16384 \
    --temp 0.6 \
    --flash-attn \
    --port 8080

# The model uses Qwen's chat template; reasoning traces appear between
# <think> and </think> tags before the final answer.
```

**Memory breakdown for Q4_K_M quantization:**

```
  Qwen2.5-7B-R1-Distill Q4_K_M:
    Weights:      ~4.5 GB (7B × ~4.8 bits / 8)
    KV cache (16K × 28 layers × 4 kv_heads × 128 head_dim × BF16):
      = 2 × 16,384 × 28 × 4 × 128 × 2 bytes = 1.5 GB
    GPU overhead:    ~0.5 GB
    ──────────────────────────
    Total:          ~6.5 GB  ← fits on RTX 3060 (12 GB) with headroom
```

**vLLM serving for distilled models:**

```bash
# Distilled 7B on a single A10G (24 GB) — good for production
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --trust-remote-code

# Distilled 14B on RTX 4090 (24 GB) — fits in BF16
# Key flag choices:
#   --quantization bitsandbytes   4-bit to fit in 24 GB
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --quantization bitsandbytes \
    --trust-remote-code

# Distilled 32B on A100 (40 GB) — fits in FP8
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --dtype float16 \
    --quantization fp8 \
    --max-model-len 32768 \
    --trust-remote-code
```

### 34.9.3 Reasoning Trace Handling

The `<think>` block is part of the generation, not a metadata field.
For production APIs that should hide the thinking process:

```python
import re

def extract_answer(full_output: str) -> tuple[str, str]:
    """Split R1 output into (thinking, answer)."""
    think_match = re.search(r'<think>(.*?)</think>', full_output, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    # Answer is everything after </think>
    answer = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
    return thinking, answer

# Use with vLLM streaming to detect end of <think> block early:
def stream_with_budget_forcing(
    client, prompt: str, max_think_tokens: int = 2048
) -> None:
    """Stop thinking at max_think_tokens even if model hasn't closed the tag."""
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_think_tokens + 512,
        stream=True,
    )
    think_token_count = 0
    in_think = False
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        if "<think>" in delta:
            in_think = True
        if in_think:
            think_token_count += 1
            if think_token_count > max_think_tokens:
                break  # budget exhausted
        print(delta, end="", flush=True)
```

---

## 34.10 DeepSeek-V3-0324 and Model Updates

DeepSeek released a significant update to V3 on 24 March 2025 (V3-0324). Key changes:

```
  DeepSeek-V3-0324 vs. DeepSeek-V3 (December 2024):
  ─────────────────────────────────────────────────
  Architecture:    Unchanged (MLA + MoE + FP8, 671B)
  Weights:         Updated via continued pre-training + RLHF
  Context:         128K (unchanged)

  Improvements:
  - Instruction following: +15% on MT-Bench
  - Code generation: best open model on LiveCodeBench (Mar 2025)
  - Math reasoning: improved on AIME 2025 without long CoT
  - Function calling: significantly more reliable tool use

  Serving:
  - Same vLLM/llama.cpp configuration as original V3
  - HuggingFace: deepseek-ai/DeepSeek-V3-0324
```

The pattern here — large base model updated post-training without architecture change —
is increasingly common. The serving engineer's job is unchanged: the same hardware, the
same config flags, just a new weight file. Maintaining a staging environment that
validates new weight checkpoints against your production metrics (Chapter 17) before
rollout is essential.

---

## 34.11 Multi-Node Serving for DeepSeek-V3

DeepSeek-V3 (671B FP8) requires at minimum 8× H200s (141 GB each) or 16× H100s.
Multi-node serving introduces latency from cross-node communication.

### 34.11.1 Single-Node vs. Multi-Node Topology

```
  Single-node (8× H200, NVLink):
    All-reduce latency: ~10 µs (NVLink 4.0 = 900 GB/s per GPU)
    KV cache: shared within node
    Pipeline parallel: not needed

  Multi-node (16× H100, InfiniBand):
    All-reduce latency: ~50–200 µs (per tensor-parallel step)
    Recommended: tensor_parallel=8, pipeline_parallel=2
    KV cache: pipeline stage owns its layers; no cross-node KV transfer

  The latency tax:
    At ITL (inter-token latency) of 50ms, a 200µs all-reduce is 0.4% overhead.
    At ITL of 5ms (small model, fast GPU), 200µs = 4% overhead — still acceptable.
    The real bottleneck is memory bandwidth for weight reads, not all-reduce.
```

### 34.11.2 vLLM Multi-Node Launch

```bash
# Node 0 (head node) — starts the ray cluster
ray start --head --port=6379

# Node 1 (worker node) — joins the cluster
ray start --address="node0_ip:6379"

# Both nodes: launch vLLM with tensor + pipeline parallelism
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --max-model-len 131072 \
    --kv-cache-dtype fp8 \
    --quantization fp8 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code
```

### 34.11.3 Pipeline Parallel KV Cache Considerations

In pipeline parallelism, stage 0 (first half of layers) and stage 1 (second half) are
on different nodes. The KV cache is per-stage:

```
  Stage 0 (layers 0–29):  KV cache for layers 0–29 on node 0
  Stage 1 (layers 30–59): KV cache for layers 30–59 on node 1

  Total KV per sequence = sum over all layers
  Per-node KV budget = (layers_in_stage / total_layers) × total_KV

  For MLA (68.6 KB/token) at 32K context:
    Total: 32K × 68.6 KB = 2.2 GB per sequence
    Per stage (30/60 layers): 1.1 GB per node per sequence
    8 sequences: 8.8 GB per node for KV cache ← manageable
```

---

## 34.12 Companion Code

The companion Python demo (`deepseek_demo.py`) simulates MLA KV compression, MoE routing with load-balancing analysis, FP8 scaling, and hardware budget calculations. The C++ demo (`deepseek_demo.cpp`) implements the same models.

---

## 34.13 Chapter Summary

DeepSeek introduced three architectural innovations that matter for every inference engineer:

**MLA** compresses the KV cache by ~5× through low-rank latent projection. At 128K context, this recovers ~31 GB of HBM per model instance — the difference between fitting or not fitting on 8× H100.

**MoE** with 256 routed experts and top-8 routing (plus 1 always-active shared expert, giving 9 active experts per token) achieves frontier quality at ~37B effective parameters. The memory cost is 671B params, but the compute cost (and thus throughput) matches a 37B dense model.

**FP8** halves weight memory versus BF16 on supported hardware (H100+), and nearly doubles throughput on Hopper tensor cores.

### Self-Check Questions

1. For a 128K context sequence, how much HBM does the KV cache consume for Llama 3.1 70B vs. DeepSeek-V3 (MLA)?
2. DeepSeek-V3 has 671B parameters but only 37B "active" per token. What is the implication for: (a) memory requirement, (b) throughput at batch=1?
3. Why does MoE serving require NVLink rather than PCIe between GPUs?
4. What is the minimum GPU configuration required to serve DeepSeek-V3 with FP8 weights?

### Where We Go Next

Chapter 35 covers Qwen — the multilingual model family from Alibaba that spans 0.5B to 72B, including MoE variants and the vision-language Qwen-VL series.


---

## Chapter Summary

- **Multi-head Latent Attention (MLA)**: DeepSeek's KV cache compression technique; compresses the KV cache by projecting keys and values through a low-rank bottleneck, reducing KV memory by 5–13× vs full MHA.
- **Mixture of Experts (MoE)**: DeepSeek-V3 uses 256 experts with top-8 routing; only 8 experts are active per token, keeping FLOPs-per-token comparable to a dense 7B model despite having 671B total parameters.
- **FP8 training and inference**: DeepSeek-V3 was trained and is served in FP8; combined with H800 Tensor Core support, this achieves ~2× throughput versus BF16.
- **Expert parallelism at scale**: DeepSeek distributes 256 experts across 64 GPUs; all-to-all communication for expert routing is the dominant communication bottleneck.
- **KV cache savings from MLA**: at sequence length 4 096, MLA reduces per-token KV memory from 512 KB (full MHA at 128 heads) to ~40 KB — a 13× reduction, enabling much larger batch sizes.
- **Serving DeepSeek with vLLM**: requires `--trust-remote-code` and MLA-specific kernel support; `--quantization fp8` leverages native FP8 inference.
- **DeepSeek-R1**: reasoning variant with chain-of-thought; average 8 000 thinking tokens per query dramatically increases KV cache and compute requirements.


---

## Worked Solutions

### Question 1
**KV cache: Llama 3.1 70B vs DeepSeek-V3 (MLA) at 128K context.**

**Llama 3.1 70B (GQA, 8 KV heads, 128 layers... wait: 80 layers, d_k=128, BF16):**
```
KV_per_token = 2 x 80 layers x 8 KV heads x 128 dim x 2 bytes
             = 2 x 80 x 8 x 128 x 2
             = 327,680 bytes = 320 KB per token

Total at 128K: 128,000 x 320 KB = 40,960 MB = 40 GB
```

**DeepSeek-V3 (MLA, c_KV=512 per token):**
MLA caches only the compressed latent vector c_KV of dimension 512 per token (not the full K/V).
```
MLA_KV_per_token = 61 transformer layers x 512 x 2 bytes = 62,464 bytes ~= 61 KB per token

Total at 128K: 128,000 x 61 KB = 7,808 MB ~= 7.6 GB
```

**Comparison:** DeepSeek-V3 MLA uses ~7.6 GB vs Llama's ~40 GB -- approximately **5.3x less KV cache** at 128K context. This is the core efficiency win of MLA that enables DeepSeek-V3 to handle much longer contexts on the same hardware.

---

### Question 2
**DeepSeek-V3: 671B parameters, 37B active per token.**

**(a) Memory requirement:**
The full 671B parameters must be stored in memory even though only 37B are active per token. Expert weights for all 256 experts must be loaded (MoE routing is dynamic -- any expert could be called for any token). At FP8 (1 byte/param):
```
memory = 671B x 1 byte = 671 GB minimum
```
This requires at least 9 x H100-80GB (720 GB total, giving 49 GB headroom) or 8 x H200-141GB (1,128 GB, with ample headroom for KV cache).

**(b) Throughput at batch=1:**
At batch=1, each decode step activates only 37B parameters (top-8 of 256 experts). The memory bandwidth bottleneck is loading active weights:
```
bytes_loaded = 37B x 1 byte/param (FP8) = 37 GB per decode step
time = 37 GB / 3,350 GB/s (H100 HBM bandwidth) = 11 ms per token
```
This is comparable to a 37B dense model -- the sparsity directly translates to throughput at batch=1. At large batch sizes, MoE throughput advantage grows further as the expert routing distributes across all 256 experts.

---

### Question 3
**Why MoE serving requires NVLink rather than PCIe between GPUs:**

In MoE serving, every forward pass includes an all-to-all communication: each GPU sends token activations to the GPUs holding the selected expert weights, and receives the expert outputs back. For DeepSeek-V3 with top-8 routing and 256 experts across 8+ GPUs:

**Communication volume per token per step:**
Each token is sent to 8 different expert GPUs and receives 8 expert outputs back. For d_model=7,168, BF16:
```
per_token_comm = 8 experts x 7,168 x 2 bytes x 2 (send+receive) = 229,376 bytes = 224 KB
```
For a batch of 1,000 tokens:
```
total = 1,000 x 224 KB = 224 MB per forward pass step
```

**Bandwidth requirement:**
If each decode step must complete in 20 ms (50 Hz), the all-to-all bandwidth requirement is:
```
required_bandwidth = 224 MB / 0.020 s = 11.2 GB/s per GPU pair
```

- NVLink 4.0: 900 GB/s bidirectional between all GPU pairs via NVSwitch -> easily handles 11.2 GB/s.
- PCIe 4.0: 16-32 GB/s *total*, shared across all GPU-to-GPU transfers, with high latency (~5 µs vs ~1 µs for NVLink).

On PCIe, the all-to-all becomes the dominant bottleneck, reducing effective throughput by 5-10x. NVLink is effectively mandatory for production MoE serving.

---

### Question 4
**Minimum GPU configuration for DeepSeek-V3 with FP8 weights:**

DeepSeek-V3 at FP8: 671B x 1 byte = 671 GB model weights.

Add KV cache for reasonable context (e.g., 8K tokens at 61 KB/token = 488 MB, negligible).
Add activations and overhead: ~50 GB.

Total: ~721 GB.

**Minimum configuration:**
- 8x H100-SXM5 (80 GB each = 640 GB): **insufficient** (671 GB weights alone exceed 640 GB).
- 10x H100-SXM5 (800 GB): sufficient for weights, minimal KV budget.
- **8x H200 (141 GB each = 1,128 GB)**: sufficient with ample KV cache headroom. This is DeepSeek's recommended configuration.
- Alternatively: 16x A100-80GB (1,280 GB) with TP=16 and EP=8 -- works but high communication overhead.

**Practical minimum: 8x H200-141GB** connected via NVLink -- matching DeepSeek's reference deployment.

