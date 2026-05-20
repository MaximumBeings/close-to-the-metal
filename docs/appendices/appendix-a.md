# Appendix A: Mathematical Foundations for LLM Inference

> *"You don't need a PhD to understand inference. You need fluency in tensors, matrix multiplication, and a few dozen carefully chosen identities."*

---

## A.1 Notation Conventions

Throughout this book, the following notation is used consistently:

| Symbol | Meaning |
|---|---|
| $d_{model}$ | Model hidden dimension (e.g., 4096 for 7B model) |
| $d_{head}$ | Per-attention-head dimension ($d_{model} / n_{heads}$) |
| $n_{heads}$ | Number of query attention heads |
| $n_{kv}$ | Number of KV heads (= $n_{heads}$ in MHA, < $n_{heads}$ in GQA) |
| $n_{layers}$ | Number of transformer layers |
| $L$ | Sequence length (number of tokens) |
| $V$ | Vocabulary size |
| $B$ | Batch size |
| $\mathbf{W}$ | Weight matrix (uppercase bold) |
| $\mathbf{x}$ | Vector (lowercase bold) |
| $\mathbf{X}$ | Matrix or tensor (uppercase bold) |
| $\odot$ | Element-wise (Hadamard) product |
| $\|\cdot\|$ | L2 norm unless subscripted |

---

## A.2 Tensors and Shapes

All major operations in a transformer are tensor operations. Understanding tensor shapes is prerequisite to understanding memory and compute costs.

### A.2.1 The Fundamental Shape: $[B, L, d_{model}]$

A batch of token sequences is represented as a 3D tensor:

```
X ∈ ℝ^{B × L × d_model}

Where:
  B       = batch size (number of sequences)
  L       = sequence length (number of tokens)
  d_model = hidden dimension

Memory: B × L × d_model × bytes_per_element

Example (Llama 3.1 70B, BF16):
  B=8, L=2048, d_model=8192
  Memory = 8 × 2048 × 8192 × 2 bytes = 268 MB
```

### A.2.2 Matrix Multiplication Shapes

The core operation in transformers is GEMM (General Matrix-Matrix Multiplication):

```
C = A × B

A ∈ ℝ^{m × k}
B ∈ ℝ^{k × n}
C ∈ ℝ^{m × n}

FLOPs: 2 × m × k × n
  (factor of 2 for multiply-accumulate)

For a linear layer projecting [B×L, d_model] → [B×L, d_ffn]:
  A = input ∈ ℝ^{(B×L) × d_model}
  B = weight ∈ ℝ^{d_model × d_ffn}
  C = output ∈ ℝ^{(B×L) × d_ffn}
  FLOPs = 2 × (B×L) × d_model × d_ffn
```

### A.2.3 Tensor Contraction Notation

Einstein summation (einsum) notation concisely describes tensor operations:

```python
# Query-Key dot product for attention scores
# Q: [B, n_heads, L_q, d_head]
# K: [B, n_kv,   L_k, d_head]
# scores: [B, n_heads, L_q, L_k]

scores = einsum("bhid,bhjd->bhij", Q, K)
# b = batch, h = head, i = query pos, j = key pos, d = head dim

# Weight projection
# x: [B, L, d_model]
# W: [d_model, d_out]
# y: [B, L, d_out]

y = einsum("bld,do->blo", x, W)
```

---

## A.3 The Attention Mechanism — Full Derivation

### A.3.1 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_head) · V

Where:
  Q ∈ ℝ^{L_q × d_head}   (queries)
  K ∈ ℝ^{L_k × d_head}   (keys)
  V ∈ ℝ^{L_k × d_head}   (values)

Steps:
  1. Scores = Q × K^T         ∈ ℝ^{L_q × L_k}    (FLOPs: 2×L_q×L_k×d_head)
  2. Scaled  = Scores / √d_head                    (FLOPs: L_q × L_k)
  3. Masked  = Scaled + causal_mask               (apply -∞ to future tokens)
  4. Weights = softmax(Masked, dim=-1)             (FLOPs: ~5×L_q×L_k)
  5. Output  = Weights × V    ∈ ℝ^{L_q × d_head}  (FLOPs: 2×L_q×L_k×d_head)

Total FLOPs ≈ 4 × L_q × L_k × d_head + 5 × L_q × L_k
           ≈ 4 × L² × d_head  (when L_q = L_k = L, as in prefill)
```

### A.3.2 Why the √d_head Scaling?

Without scaling, dot products grow with dimension. If Q and K have unit-variance elements, $QK^T$ has variance $d_{head}$, giving standard deviation $\sqrt{d_{head}}$. Without scaling, the softmax input has large magnitude, pushing it into saturation (near-zero gradients). Dividing by $\sqrt{d_{head}}$ restores unit variance.

```
Proof sketch:
  Q_i, K_j ~ N(0, 1) (assume unit-variance initialization)
  
  Q_i · K_j = Σ_{k=1}^{d_head} q_k × k_k
  
  E[Q_i · K_j] = 0
  Var[Q_i · K_j] = d_head × Var[q_k × k_k] = d_head × 1
  
  Std dev of unnormalized score = √d_head
  
  After dividing by √d_head: Std dev = 1 ✓
```

### A.3.3 Multi-Head Attention (MHA)

```
MHA(Q, K, V) = Concat(head_1, ..., head_h) · W_O

head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)

Where:
  W_Q^i ∈ ℝ^{d_model × d_head}
  W_K^i ∈ ℝ^{d_model × d_head}
  W_V^i ∈ ℝ^{d_model × d_head}
  W_O   ∈ ℝ^{(n_heads × d_head) × d_model}

Memory for Q/K/V projections per layer:
  n_heads × d_head × d_model × 3 projections × 2 bytes (BF16)
  = n_heads × (d_model/n_heads) × d_model × 3 × 2
  = 3 × d_model² × 2 bytes
  = 6 × d_model² bytes
```

### A.3.4 Grouped Query Attention (GQA)

GQA reduces KV projection parameters by using fewer KV heads than Q heads:

```
MHA: n_kv = n_heads    (one K,V per Q head)
GQA: n_kv < n_heads    (shared K,V across multiple Q heads)
MQA: n_kv = 1          (extreme case: one K,V for all Q heads)

Memory savings (KV projection):
  MHA: 2 × n_heads × d_head × d_model parameters
  GQA: 2 × n_kv    × d_head × d_model parameters
  Ratio: n_heads / n_kv

Example (Llama 3.1 70B):
  n_heads = 64, n_kv = 8
  KV parameter reduction: 8× less memory for KV projections
  
KV cache savings (runtime):
  Per token per layer:
    MHA: 2 × n_heads × d_head × dtype_bytes
    GQA: 2 × n_kv    × d_head × dtype_bytes
  Savings: same ratio as parameters
```

---

## A.4 The Feed-Forward Network

### A.4.1 Standard FFN vs. SwiGLU

The original transformer used:

```
FFN(x) = ReLU(x·W_1 + b_1) · W_2 + b_2

W_1 ∈ ℝ^{d_model × d_ffn}
W_2 ∈ ℝ^{d_ffn × d_model}
d_ffn = 4 × d_model (original Transformer)
```

Modern LLMs (Llama, Qwen, Mistral) use SwiGLU:

```
SwiGLU(x) = (x·W_gate ⊙ SiLU(x·W_up)) · W_down

SiLU(z) = z × σ(z) = z / (1 + e^{-z})

W_gate ∈ ℝ^{d_model × d_ffn}
W_up   ∈ ℝ^{d_model × d_ffn}
W_down ∈ ℝ^{d_ffn × d_model}

Note: Three weight matrices (not two), so d_ffn is typically
reduced to keep total parameter count similar:
  d_ffn ≈ 8/3 × d_model (rounded to multiple of 128 for efficiency)
  
Example (Llama 3.1 70B): d_model=8192, d_ffn=28,672
  28,672 ≈ (8/3) × 8192 = 21,845 → rounded up to 28,672 (divisible by 256)
```

### A.4.2 FLOPs for FFN

```
SwiGLU FLOPs per token:
  gate projection:  2 × d_model × d_ffn
  up projection:    2 × d_model × d_ffn
  SiLU + hadamard:  2 × d_ffn  (cheap)
  down projection:  2 × d_ffn  × d_model
  Total ≈ 6 × d_model × d_ffn

Full transformer layer per token:
  Attention QKV:  2 × d_model × 3 × d_model  (if MHA, with projections)
  Attention output: 2 × d_model × d_model
  FFN: 6 × d_model × d_ffn
  Total per layer ≈ 8 × d_model² + 6 × d_model × d_ffn

Full model per token:
  Total ≈ n_layers × (8 × d_model² + 6 × d_model × d_ffn)
        ≈ 2 × N_params  (where N_params = total parameters, a well-known approximation)
```

---

## A.5 Softmax and Numerical Stability

### A.5.1 The Naive Softmax Problem

```
softmax(z)_i = e^{z_i} / Σ_j e^{z_j}

Problem: e^{z_i} overflows for z_i > 709 (float32) or z_i > 88 (float16)

Example: z = [1000, 1000, 1000]
  e^{1000} = overflow → NaN → entire distribution is NaN
```

### A.5.2 Numerically Stable Softmax

```
Stable: softmax(z - max(z))_i = e^{z_i - max(z)} / Σ_j e^{z_j - max(z)}

This is mathematically identical to naive softmax:
  e^{z_i - c} / Σ_j e^{z_j - c}
  = e^{z_i} × e^{-c} / (Σ_j e^{z_j} × e^{-c})
  = e^{z_i} / Σ_j e^{z_j}  ✓ (c cancels)

Algorithm (two-pass):
  pass 1: m = max(z)
  pass 2: sum = Σ_j e^{z_j - m}
  output: e^{z_i - m} / sum
```

### A.5.3 Online Softmax (Flash Attention)

Flash Attention requires computing softmax in a single pass (to avoid materializing the full attention matrix). The online algorithm maintains a running maximum and sum:

```
Online softmax (single pass):
  Initialize: m = -∞, d = 0
  
  For each chunk of attention scores [s_1, ..., s_k]:
    m_new = max(m, max(s))
    d = d × e^{m - m_new} + Σ_i e^{s_i - m_new}
    m = m_new
  
  Final: softmax(z)_i = e^{z_i - m} / d

Proof that update is correct:
  After processing [s_1..s_k], d = Σ_{i=1}^{k} e^{s_i - m}
  
  When new chunk [s_{k+1}..s_{k+j}] arrives with new max m_new:
  d_new = Σ_{i=1}^{k} e^{s_i - m_new} + Σ_{i=k+1}^{k+j} e^{s_i - m_new}
        = Σ_{i=1}^{k} e^{s_i - m} × e^{m - m_new} + Σ_{i=k+1}^{k+j} e^{s_i - m_new}
        = d × e^{m - m_new} + Σ_{i=k+1}^{k+j} e^{s_i - m_new}  ✓
```

---

## A.6 Layer Normalization

### A.6.1 Layer Norm vs. RMS Norm

Standard layer normalization (GPT-2 style):

```
LayerNorm(x) = γ ⊙ (x - μ) / σ + β

μ = (1/d) Σ_i x_i          (mean)
σ = √((1/d) Σ_i (x_i - μ)² + ε)   (std dev, ε=1e-5 for stability)
γ, β ∈ ℝ^d                 (learned scale and bias)
```

RMS Norm (Llama, Qwen, Mistral) — no centering:

```
RMSNorm(x) = γ ⊙ x / RMS(x)

RMS(x) = √((1/d) Σ_i x_i² + ε)

Benefits over LayerNorm:
  - No mean subtraction → simpler
  - ~7% faster in practice (one pass instead of two)
  - Empirically equivalent quality
```

### A.6.2 Pre-Norm vs. Post-Norm

```
Post-Norm (original Transformer 2017):
  x = LayerNorm(x + Attention(x))
  x = LayerNorm(x + FFN(x))
  Problem: gradients vanish in deep networks

Pre-Norm (modern LLMs):
  x = x + Attention(LayerNorm(x))
  x = x + FFN(LayerNorm(x))
  Benefit: residual stream flows unchanged; training is more stable
```

---

## A.7 Positional Encoding

### A.7.1 Rotary Positional Encoding (RoPE)

RoPE encodes position by rotating query and key vectors before the dot product:

```
RoPE rotation for position m:
  
  For each pair of dimensions (2i, 2i+1):
    [q_{2i}, q_{2i+1}] → [q_{2i}·cos(mθ_i) - q_{2i+1}·sin(mθ_i),
                           q_{2i}·sin(mθ_i) + q_{2i+1}·cos(mθ_i)]

Where θ_i = 1 / (base^(2i / d_head))
  base = 10000 (original RoPE), 500000 (Llama 3.1), 1000000 (Qwen2.5)

Key property:
  RoPE(q, m) · RoPE(k, n) = f(q, k, m-n)
  
  The dot product depends only on the RELATIVE position (m-n),
  not absolute positions. This is why models can generalize
  to positions not seen during training (with caveats).
```

### A.7.2 Extending Context with YaRN

YaRN (Yet Another RoPE extensioN) extends a model's effective context by adjusting the RoPE base:

```
YaRN scaling factor s = L_new / L_train

For positions beyond L_train, YaRN uses:
  θ_i_new = θ_i / s  (slower rotation for longer sequences)

This allows a 4K-trained model to handle 32K+ contexts
without full fine-tuning, at a small quality cost.
```

---

## A.8 KV Cache Mathematics

### A.8.1 KV Cache Memory Formula

The KV cache stores past keys and values for all layers and heads:

```
KV cache memory per token:
  = 2 × n_layers × n_kv_heads × d_head × bytes_per_element

Breaking down:
  2        = one K tensor + one V tensor
  n_layers = number of transformer layers
  n_kv     = number of KV heads (GQA/MQA reduces this)
  d_head   = head dimension = d_model / n_heads
  dtype    = 2 bytes (BF16), 1 byte (INT8/FP8), 0.5 bytes (INT4)

Examples:
  Llama 3.1 70B (BF16, GQA with n_kv=8):
    = 2 × 80 × 8 × 128 × 2 = 327,680 bytes = 320 KB/token
  
  Llama 3.1 70B (FP8, n_kv=8):
    = 2 × 80 × 8 × 128 × 1 = 163,840 bytes = 160 KB/token

  DeepSeek-V3 (MLA, d_latent=512, d_rope=64, BF16):
    MLA stores one latent KV vector + one decoupled RoPE key (not separate K/V):
    = 1 × n_layers × (d_latent + d_rope) × bytes_per_element
    = 1 × 61 × (512 + 64) × 2 bytes (BF16)
    = 70,272 bytes = 68.6 KB/token  (4.7× smaller than Llama 70B)
```

### A.8.2 Maximum Batch Size from KV Cache

Given a fixed GPU memory budget:

```
Available KV memory = Total GPU memory - Model weight memory - Activations

Maximum sequences × L_max = Available KV memory / (KV bytes per token)

Example (H100 80GB, Llama 3.1 70B BF16):
  Model weights:  70B × 2 bytes = 140 GB (needs 2× H100 = 160 GB)
  Activations:    ~2 GB (at batch=1)
  Available KV:   160 - 140 - 2 = 18 GB
  
  At 4K context: 18 GB / (4096 × 320 KB) = 18 GB / 1.31 GB = 13 sequences
  At 32K context: 18 GB / (32768 × 320 KB) = 18 GB / 10 GB = 1.8 → 1 sequence
```

---

## A.9 FLOPs and Memory Bandwidth Analysis

### A.9.1 Roofline Model

The roofline model classifies operations as compute-bound or memory-bandwidth-bound:

```
Arithmetic Intensity = FLOPs / Bytes accessed

If Intensity > Ridge Point → compute-bound (FLOPs are bottleneck)
If Intensity < Ridge Point → memory-bandwidth-bound (bandwidth is bottleneck)

Ridge Point = Peak FLOPS / Peak Memory Bandwidth

H100 SXM:
  Peak TFLOPS (BF16 dense): 1,979 × 10¹²
  Peak HBM Bandwidth: 3.35 × 10¹²  bytes/s
  Ridge Point: 1,979 / 3.35 ≈ 591 FLOPs/byte
```

### A.9.2 Arithmetic Intensity of LLM Operations

```
GEMM (matrix multiply): very compute-bound at large sizes
  A ∈ ℝ^{m×k}, B ∈ ℝ^{k×n}
  FLOPs: 2mkn
  Bytes: (mk + kn + mn) × dtype_bytes ≈ (mk + kn) × dtype_bytes (for large n)
  Intensity: 2mkn / ((mk + kn) × 2) ≈ n  (for large n, BF16)
  At n=4096: intensity ≈ 4096 FLOPs/byte → above ridge point ✓

Decode attention (GEMV, batch=1):
  q·K^T: FLOPs = 2 × L × d_head
  Bytes: L × d_head × 2 (read K) + d_head × 2 (read q) ≈ 2Ld_head
  Intensity: 2Ld_head / 2Ld_head = 1 FLOPs/byte → far below ridge point ✗
  → Decode is severely memory-bandwidth-bound
```

### A.9.3 Implications for Batching

```
At batch=1 (decode):
  Intensity ≈ 1 FLOPs/byte → bandwidth-limited
  Increasing batch to B: Intensity ≈ B FLOPs/byte
  At batch=32: Intensity ≈ 32 FLOPs/byte (still below ridge at 591)
  
  This explains why decode throughput scales linearly with batch size
  (until memory is full) — it's still bandwidth-bound at typical batch sizes.
  
At batch=256 (prefill):
  Intensity ≈ 256 FLOPs/byte → approaching compute-bound
  Flash Attention fuses operations, reducing memory traffic → higher effective intensity
```

---

## A.10 Quantization Mathematics

### A.10.1 Uniform Quantization

```
Quantize: q = round(x / scale) + zero_point
          scale = (x_max - x_min) / (2^bits - 1)
          
Dequantize: x̂ = (q - zero_point) × scale

Quantization error: ε = x - x̂
  Maximum error: ε_max = scale / 2

For INT8 (256 levels):
  scale = range / 255
  ε_max = range / 510

For FP8 E4M3 (448 levels, range ±448):
  scale = max_abs / 448
  ε_max ≈ max_abs / 896
```

### A.10.2 Block Quantization (GGUF)

```
Block quantization groups k consecutive values:
  For each block of k elements:
    scale_block = max(abs(block)) / (2^bits - 1)
    q_i = round(x_i / scale_block)

Benefit: scale adapts per block, reducing error for non-uniform distributions
Storage: k × bits/8 bytes + 2 bytes for scale (FP16)
Overhead: 2/(k × bits/8) × 100% = 2/(32 × 0.5) × 100% = 12.5% for Q4 k=32
```

### A.10.3 Expected Quantization Error vs. Perplexity

Empirically observed relationship (approximate):

```
Model size vs. tolerable quantization bits:
  7B parameters:  INT4 adds ~0.3-0.5 PPL penalty
  13B parameters: INT4 adds ~0.2-0.3 PPL penalty  
  70B parameters: INT4 adds ~0.1-0.2 PPL penalty
  
Rule of thumb: larger models are more robust to quantization
  because individual weight precision matters less when you have
  many weights to average over.
```

---

## A.11 Speculative Decoding Mathematics

### A.11.1 Acceptance Rate and Expected Speedup

The theoretical speedup from speculative decoding depends on the acceptance rate $\alpha$ (fraction of draft tokens accepted by the target model):

```
Let:
  γ = number of draft tokens per speculation step
  α = per-token acceptance rate (0 ≤ α ≤ 1)

Expected accepted tokens per step:
  E[accepted] = Σ_{k=1}^{γ} α^k = α(1 - α^γ) / (1 - α)

One bonus token is always generated (the rejection correction),
so expected output per step = E[accepted] + 1

Total time per step:
  T_step = T_draft × γ + T_target  (one draft pass + one target pass)

Effective speedup:
  S = (E[accepted] + 1) / T_step × T_target
  S = (E[accepted] + 1) × T_target / (T_draft × γ + T_target)

When T_draft << T_target (draft is much smaller):
  S ≈ E[accepted] + 1
  
At α=0.8, γ=5:
  E[accepted] = 0.8(1-0.8⁵)/(1-0.8) = 0.8 × 0.67264 / 0.2 = 2.69
  S ≈ 2.69 + 1 = 3.69× (theoretical maximum assuming free draft)
```

---

## A.12 Mixture of Experts (MoE) Mathematics

### A.12.1 Top-K Routing

```
Given hidden state x ∈ ℝ^{d_model}:

  Gate logits: g = x · W_gate,  W_gate ∈ ℝ^{d_model × E}  (E = num experts)
  Top-K selection: indices = argtopk(g, k)
  Routing weights: w_i = softmax(g[indices])_i
  
  Output: y = Σ_{i ∈ top_k} w_i × FFN_i(x)

FLOPs vs. Dense:
  Dense FFN: 6 × d_model × d_ffn
  MoE FFN (top-k): 6 × d_model × d_ffn/E × k + gating overhead
  
  If each expert has d_ffn_expert = d_ffn/E × E = d_ffn (total same),
  then active FLOPs ≈ k/E fraction of dense FLOPs
  
  DeepSeek-V3: k=8 routed + 1 shared expert active out of E=256 routed experts
              → 8/256 = 3.1% of routed weights active per FFN, plus the always-on shared expert
```

### A.12.2 Expert Load Balancing

Without load balancing, a few experts get all traffic (expert collapse):

```
Auxiliary load balancing loss (from Switch Transformer):
  
  loss_lb = α × E × Σ_{i=1}^{E} f_i × P_i
  
  Where:
    f_i = fraction of tokens routed to expert i
    P_i = fraction of router probability assigned to expert i
    α   = small coefficient (e.g., 0.01)
  
  Ideal: f_i = 1/E for all experts (uniform distribution)
  
DeepSeek-V3 uses auxiliary-free load balancing:
  Bias term b_i added to logits for each expert
  b_i is dynamically adjusted based on expert utilization
  No loss term needed → better task performance
```

---

## A.13 Summary of Key Formulas

```
┌──────────────────────────────────────────────────────────────────────┐
│ QUICK REFERENCE — Key Formulas                                        │
├──────────────────────────────────────────────────────────────────────┤
│ Attention FLOPs (prefill, L tokens):                                  │
│   ≈ 4 × L² × d_head × n_heads + 8 × L × d_model²                   │
│                                                                        │
│ Full model FLOPs per token (decode):                                   │
│   ≈ 2 × N_params (where N_params = total parameters)                  │
│                                                                        │
│ KV cache memory per token:                                             │
│   = 2 × n_layers × n_kv × d_head × dtype_bytes                       │
│                                                                        │
│ Max throughput (bandwidth-limited decode):                             │
│   tok/s = bandwidth / (2 × N_params)                                  │
│   (loads all weights once per decode step)                             │
│                                                                        │
│ Arithmetic intensity (GEMM, batch B):                                  │
│   ≈ B FLOPs/byte (for weight-stationary, BF16)                       │
│                                                                        │
│ Speculative decoding speedup:                                          │
│   S ≈ (1 + γα(1-α^γ)/(1-α)) / (1 + c×γ)                            │
│   c = T_draft/T_target (cost ratio)                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## A.14 Automatic Differentiation and Backpropagation

> *"Training a neural network is just computing a scalar loss and then applying the chain rule backwards through the computation graph until every weight has a gradient."*

LLM inference does not perform backpropagation — but understanding autograd explains why weights are shaped the way they are, why gradient checkpointing matters, and how quantization-aware training works. This section develops automatic differentiation from scalar partial derivatives through full tensor Jacobians, following the MicroGPT/micrograd pedagogical path.

### A.14.1 Partial Derivatives — From Scalar to Tensor

A **partial derivative** isolates how one input affects the output while holding all others constant. For a function $f(x_1, x_2)$:

```
  ∂f/∂x₁  means: how much does f change when x₁ changes by ε, x₂ fixed?

  f(x₁, x₂) = x₁ + x₂:
    ∂f/∂x₁ = 1    (output changes by ε when x₁ changes by ε)
    ∂f/∂x₂ = 1    (symmetric)

  f(x₁, x₂) = x₁ × x₂:
    ∂f/∂x₁ = x₂   (changing x₁ scales f by x₂)
    ∂f/∂x₂ = x₁   (changing x₂ scales f by x₁)

  f(x₁, x₂) = x₁ / x₂:
    ∂f/∂x₁ = 1/x₂
    ∂f/∂x₂ = -x₁/x₂²
```

**Extending to tensors:** When inputs are vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^k$, the same scalar rules apply element-wise at each index $i$. The derivative $\partial w_i / \partial u_j = 0$ whenever $i \neq j$ for elementwise operations — this gives rise to diagonal Jacobians.

### A.14.2 The Chain Rule

For a composed function $L = f(g(x))$:

$$\frac{dL}{dx} = \frac{dL}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dx}$$

In a computation graph, this means: to get the gradient of $L$ with respect to any intermediate node, multiply all local gradients along the path from $L$ to that node.

```
WORKED EXAMPLE A.14.1 — Chain Rule: L = relu(a·b + c)
────────────────────────────────────────────────────────
Given: a=2, b=3, c=1

Forward:
  p = a * b = 6          local: dp/da = b = 3, dp/db = a = 2
  s = p + c = 7          local: ds/dp = 1,    ds/dc = 1
  L = relu(s) = 7        local: dL/ds = 1  (s > 0)

Backward (chain rule, reverse order):
  dL/ds = 1              (relu gate open, s > 0)
  dL/dp = dL/ds × 1 = 1
  dL/dc = dL/ds × 1 = 1
  dL/da = dL/dp × dp/da = 1 × 3 = 3
  dL/db = dL/dp × dp/db = 1 × 2 = 2

Result: da=3, db=2, dc=1  ✓
────────────────────────────────────────────────────────
```

```
WORKED EXAMPLE A.14.2 — Node Reuse: L = tanh(a) × a, a=1
────────────────────────────────────────────────────────
When a single variable feeds into an operation twice,
gradients accumulate via += from both paths.

Forward:
  t = tanh(1)  ≈ 0.7616
  L = t × a = 0.7616

Two paths from L back to a:
  Path 1 (via tanh):
    dL/dt = a = 1
    dt/da = 1 − tanh²(a) = 1 − 0.580 ≈ 0.420
    contribution: 1 × 0.420 = 0.420

  Path 2 (direct):
    dL/da_direct = t ≈ 0.7616

Total: da = 0.420 + 0.7616 = 1.1816  ✓

Rule: whenever a node is reused, .grad += (never overwrites)
────────────────────────────────────────────────────────
```

### A.14.3 Jacobian Matrices

When the input and output are both vectors/tensors, the derivative becomes a **Jacobian matrix** $J \in \mathbb{R}^{m \times n}$ where $J_{ij} = \partial f_i / \partial x_j$.

**Key insight for elementwise operations:** because output element $i$ depends only on input element $i$, the Jacobian is **diagonal** — only $n$ entries are non-zero out of $n^2$ total. This makes backpropagation $O(n)$ instead of $O(n^2)$.

```
  Jacobian Types for Common Ops:

  ┌──────────────────────────────────────────────────────────┐
  │  Operation         Jacobian ∂output/∂input              │
  ├──────────────────────────────────────────────────────────┤
  │  f(A,B) = A + B    I  (identity)                        │
  │  f(A,B) = A − B    −I  (negative identity)              │
  │  f(A,B) = A ⊙ B   diag(B)  (B values on diagonal)      │
  │  f(A,B) = A / B    diag(1/B)                            │
  │  f(A)   = −A       −I                                   │
  │  f(A)   = relu(A)  diag(A > 0)  (mask)                 │
  │  f(A)   = log(A)   diag(1/A)                            │
  │  f(A)   = exp(A)   diag(exp(A)) = diag(result)          │
  │  f(A)   = A^e      diag(e × A^(e−1))                   │
  │  f(A)   = tanh(A)  diag(1 − tanh²(A))                  │
  │  f(A)   = σ(A)     diag(σ(A)(1 − σ(A)))                │
  └──────────────────────────────────────────────────────────┘

  Industry rule: NEVER form the full matrix.
  All elementwise ops → O(n) backward via diagonal shortcut.
```

**Full 4×4 example for elementwise multiplication** (A=[[1,2],[3,4]], B=[[2,3],[4,5]]):

```
  ∂(A⊙B)/∂A = diag([2, 3, 4, 5])  =  diag(B_flat)

       A₀₀  A₀₁  A₁₀  A₁₁
  R₀₀: [ 2    0    0    0 ]   ← B₀₀ = 2
  R₀₁: [ 0    3    0    0 ]   ← B₀₁ = 3
  R₁₀: [ 0    0    4    0 ]   ← B₁₀ = 4
  R₁₁: [ 0    0    0    5 ]   ← B₁₁ = 5

  Industry implementation: grad_A = upstream * B   (elementwise)
  No matrix formed — pure elementwise multiply, O(n).
```

### A.14.4 Operation Backward Passes — Complete Reference

Every operation stores its inputs as **children** and the local derivative as **local_grads** during the forward pass. During backward, gradient flows as: `child.grad += local_grad × node.grad`.

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Op          Forward          Backward (industry shortcut)       │
  ├──────────────────────────────────────────────────────────────────┤
  │  Add         z = a + b        grad_a = upstream                  │
  │                               grad_b = upstream                  │
  │  Sub         z = a − b        grad_a = +upstream                 │
  │  (=add+neg)                   grad_b = −upstream                 │
  │  Mul         z = a × b        grad_a = upstream × b              │
  │                               grad_b = upstream × a              │
  │  Div         z = a / b        grad_a = upstream / b              │
  │                               grad_b = −upstream × a / b²        │
  │  Neg         z = −a           grad_a = −upstream                 │
  │  ReLU        z = max(0,a)     grad_a = upstream × (a > 0)        │
  │  Log         z = ln(a)        grad_a = upstream / a              │
  │  Exp         z = exp(a)       grad_a = upstream × result         │
  │  Pow(e)      z = a^e          grad_a = upstream × e × a^(e−1)   │
  │  Tanh        z = tanh(a)      grad_a = upstream × (1 − result²)  │
  │  Sigmoid     z = σ(a)         grad_a = upstream × result×(1−r)   │
  └──────────────────────────────────────────────────────────────────┘

  All: result from forward pass is reused — no recomputation.
```

**Sigmoid derivation** (illustrative):

$$\sigma(a) = \frac{1}{1+e^{-a}} \quad \Rightarrow \quad \frac{d\sigma}{da} = \sigma(a)(1-\sigma(a))$$

At $a=1$: $\sigma(1) \approx 0.731$, gradient $\approx 0.731 \times 0.269 \approx 0.197$. Maximum gradient is $0.25$ at $a=0$; saturates toward $0$ as $|a| \to \infty$ — the **vanishing gradient** problem.

### A.14.5 DAG and Topological Sort

A computation graph is a **Directed Acyclic Graph (DAG)**. Backpropagation requires visiting nodes in **reverse topological order** — every consumer must finish propagating before a node can backpropagate to its inputs.

```
  Graph: L = relu(a·b + c·b)

  a=2 ──→ [p₁=a·b=6]
  b=3 ──↗          ↘
  c=4 ──→ [p₂=c·b=12] → [s=p₁+p₂=18] → [L=relu(s)=18]
  b=3 ──↗

  Topological order (DFS post-order):
    Forward:  [a, b, p₁, c, p₂, s, L]
    Reversed: [L, s, p₂, c, p₁, b, a]

  Rule: visit b ONLY after BOTH p₁ and p₂ have run.
  Without topological order: b.grad would be incomplete → wrong.
```

**Python DFS implementation** (from micrograd):

```python
def backward(self):
    topo, visited = [], set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)       # append AFTER children

    build_topo(self)
    self.grad = 1                # seed: dL/dL = 1

    for v in reversed(topo):    # parents before children
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad   # chain rule + accumulate
```

The critical invariant: `child.grad +=` (never `=`). This ensures that when a node is reused — as in $L = s^2$ where $s$ appears as both left and right operand — both gradient contributions accumulate correctly.

### A.14.6 Cross-Entropy Loss — The Dense Jacobian Exception

Softmax followed by cross-entropy is the one common operation that produces a **dense** Jacobian. The full Jacobian $J_{ij} = p_i(\delta_{ij} - p_j)$ is an $n \times n$ matrix — at vocabulary size $n=128,000$ this is 128,000² = 16.4 billion elements. This is why the softmax Jacobian is **never formed** in practice.

```
WORKED EXAMPLE A.14.3 — Cross-Entropy Backward
────────────────────────────────────────────────────────
z = [1.0, 2.0, 3.0],  target = 2

Forward:
  shifted = z − max(z) = [−2, −1, 0]    (numerical stability)
  exps    = [0.135, 0.368, 1.000]
  total   = 1.503
  probs   = [0.090, 0.245, 0.665]
  loss    = −log(probs[2]) ≈ 0.408

Unified backward shortcut (O(vocab)):
  dL/dz[i] = probs[i] − 𝟙[i == target]

  dL/dz[0] = 0.090 − 0 = +0.090   (push down)
  dL/dz[1] = 0.245 − 0 = +0.245   (push down)
  dL/dz[2] = 0.665 − 1 = −0.335   (push up ← target)

Interpretation: gradient pushes the target logit UP
and all other logits DOWN, proportionally to their
current probability.
────────────────────────────────────────────────────────
```

```python
def cross_entropy_backward(probs, target, batch_size=1):
    grad = probs.copy()
    grad[target] -= 1          # subtract 1 from target class
    return grad / batch_size   # average over batch
    # Dense Jacobian never formed. O(vocab) not O(vocab²).
```

### A.14.7 Shape-Transforming Operations

Operations that change shape without changing values have gradients that reshape or accumulate the upstream gradient:

```
  ┌────────────────────────────────────────────────────────────┐
  │  Op           Forward                  Backward            │
  ├────────────────────────────────────────────────────────────┤
  │  Reshape      out = A.view(new_shape)  grad_A = up.view(A.shape)
  │  Broadcast    out = A.expand(shape)    grad_A = up.sum(broadcast dims)
  │  Transpose    out = A.T               grad_A = up.T               │
  │  Concat       out = cat([A,B], dim=0) grad_A = up[:len(A)]        │
  │                                        grad_B = up[len(A):]        │
  └────────────────────────────────────────────────────────────┘

  Broadcast backward must SUM along the broadcast dimension:
  A = [1,2]  broadcast to [[1,2],[1,2],[1,2]]
  upstream = [[g0,g1],[g2,g3],[g4,g5]]
  grad_A   = [g0+g2+g4, g1+g3+g5]   ← sum, not copy
```

### A.14.8 The Value Class — Memory Layout

Every node in a computation graph is a `Value` instance. In Python, `__slots__` replaces the default `__dict__` hash-map with fixed-offset fields:

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    # ~80 bytes/node with __slots__  vs  ~300 bytes without
    # 50,000 nodes × 220 bytes saved = ~11 MB per forward pass

    def __init__(self, data, children=(), local_grads=()):
        self.data         = data    # scalar value (forward pass)
        self.grad         = 0       # dL/d(self)  (backward pass)
        self._children    = children      # input nodes
        self._local_grads = local_grads   # d(self)/d(child)
```

The C++ equivalent is even more compact:

```cpp
struct Value {
    float   data;            // 4 B
    float   grad = 0.0f;     // 4 B
    Value*  children[2];     // 16 B
    float   local_grads[2];  // 8 B
    uint8_t nchildren;       // 1 B (+3 padding)
    // Total: ~36 B vs ~80 B in Python — 2.2× more compact
};
```

### A.14.9 Connection to LLM Inference

Although inference is forward-pass only, autograd mathematics explains several inference-critical phenomena:

| Training phenomenon | Inference consequence |
|---|---|
| Softmax Jacobian is dense | Sampling from logits requires full vocab pass — can't be sparse |
| ReLU gate blocks gradient | Dead neurons produce zero output — model pruning exploits this |
| Tanh/sigmoid saturate | Activations near ±1 carry minimal information — precision matters |
| exp(a) gradient = result | Flash Attention recomputes exp during backward — avoids storing it |
| Cross-entropy gradient: probs − 1 | Temperature scaling shifts the target logit gradient directly |

---

## A.15 Self-Check

1. A model has $d_{model}=4096$, $n_{layers}=32$, $n_{kv}=8$, $d_{head}=128$. How many bytes of KV cache does a single 2048-token sequence require in BF16?

   *Answer: $2 \times 32 \times 8 \times 128 \times 2048 \times 2 = 268,435,456$ bytes = 256 MB*

2. Why does the $\sqrt{d_{head}}$ scaling in attention prevent training instability?

   *Answer: Without scaling, dot products grow with $\sqrt{d_{head}}$, pushing softmax inputs into saturation (near-zero gradients). Scaling restores unit variance.*

3. A 7B BF16 model on an H100 (3.35 TB/s bandwidth). What is the maximum decode throughput at batch=1?

   *Answer: 3.35 × 10¹² / (7 × 10⁹ × 2) ≈ 239 tokens/second*

4. In GQA with $n_{heads}=32$ and $n_{kv}=4$, how many times is each KV head shared by query heads?

   *Answer: 32/4 = 8 query heads share each KV head*


---

## Worked Solutions

### Question 1
**Model: d_model=4096, n_layers=32, n_kv=8, d_head=128. KV cache for 2048-token sequence in BF16.**

**Step 1 — Bytes per token:**
Each token stores one K vector and one V vector per layer, per KV head:
```
bytes_per_token = 2 (K and V) x n_layers x n_kv x d_head x 2 bytes (BF16)
                = 2 x 32 x 8 x 128 x 2
                = 131,072 bytes = 128 KB per token
```

**Step 2 — Total for 2048 tokens:**
```
total = 2048 x 131,072 = 268,435,456 bytes = 256 MB
```

**Common mistake:** Forgetting to multiply by 2 for both K and V. Also note n_kv=8 (GQA), not n_heads=32 — with full MHA the cache would be 4x larger (1 GB vs 256 MB).

---

### Question 2
**Why does sqrt(d_head) scaling prevent training instability?**

**The problem without scaling:**
The dot product q · k sums d_head independent terms. If q and k are initialised with zero-mean unit-variance components (as is standard), then:
```
Var(q · k) = sum_{i=1}^{d_head} Var(q_i * k_i) = d_head x Var(q_i) x Var(k_i) = d_head
```

So std(q · k) = sqrt(d_head). For d_head=64, std = 8; for d_head=128, std = 11.3.

**Why this causes instability:**
Softmax is applied to the raw dot products. When inputs have large variance (std >> 1), softmax produces very peaked distributions (near one-hot). The gradient of a peaked softmax approaches zero almost everywhere — this is the **vanishing gradient** problem. Training becomes extremely slow or collapses.

**The fix:**
Dividing by sqrt(d_head) normalises the variance back to 1:
```
Var(q · k / sqrt(d_head)) = d_head / d_head = 1
```

Now softmax receives inputs with unit standard deviation regardless of d_head, producing well-calibrated attention distributions and healthy gradients throughout training.

---

### Question 3
**7B BF16 model on H100 (3.35 TB/s bandwidth). Maximum decode throughput at batch=1.**

**Model weight bytes:**
```
7B parameters x 2 bytes (BF16) = 14 GB
```

**Bandwidth-bound decode time:**
At batch=1, each decode step must load all model weights from HBM (the weights are the bottleneck, not compute):
```
t_per_token = 14 GB / 3,350 GB/s = 0.00418 s = 4.18 ms per token
```

**Maximum throughput:**
```
tokens_per_second = 1 / 0.00418 = 239 tokens/s
```

**Reality check:** This is the theoretical upper bound assuming perfect memory access patterns and zero overhead. Real-world throughput is typically 70-85% of this (175-200 tok/s) due to CUDA kernel launch overhead, attention KV cache reads, and activation tensor movements.

**Note:** At batch=B, the model weights are amortised across B tokens. For batch=8:
```
t_per_token = 14 GB / (3,350 GB/s * 8) ≈ 0.52 ms -> 1,920 tok/s effective
```
Batching is the primary lever for improving throughput on bandwidth-bound models.

---

### Question 4
**GQA with n_heads=32, n_kv=4. How many times is each KV head shared?**

**Calculation:**
With 32 query heads and 4 KV heads, query heads are grouped into n_kv=4 groups:
```
sharing_ratio = n_heads / n_kv = 32 / 4 = 8
```

Each KV head is shared by **8 query heads**.

**Group structure:**
- KV head 0: serves query heads 0-7
- KV head 1: serves query heads 8-15
- KV head 2: serves query heads 16-23
- KV head 3: serves query heads 24-31

**Memory saving vs MHA:**
MHA requires n_heads=32 KV heads. GQA with n_kv=4 reduces KV cache by:
```
reduction = 32 / 4 = 8x
```

For the model in Q1 (n_layers=32, d_head=128, BF16), this reduces KV cache from:

- MHA: 2 x 32 x 32 x 128 x 2 = 524,288 bytes = 512 KB/token
- GQA (n_kv=4): 131,072 bytes = 128 KB/token (4x fewer KV heads shown above)

At 128K context, the difference is 128K x (512-128) KB = 48 GB saved — enough to fit an additional small model on the same GPU.

