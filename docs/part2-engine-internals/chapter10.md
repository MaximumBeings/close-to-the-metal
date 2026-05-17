# Chapter 10 — Quantization Internals: GGUF vs. vLLM

> *"Every bit you shave off a weight is a bit you don't have to read from HBM.
> Quantization is not a compression trick — it is a memory-bandwidth trade.
> The question is always: how much accuracy are you willing to pay per nanosecond saved?"*

---

## 10.0 Why This Chapter Matters

Chapter 9 showed that LLM inference is overwhelmingly memory-bandwidth bound.
For LLaMA 3 8B on A100, the dominant cost is reading ~17.7 GB of FP16 weights
from HBM every decode step.  Quantization attacks this directly: store weights
at lower precision, read fewer bytes, go faster.

But quantization is not free.  Every bit you drop is representational capacity
lost.  This chapter gives you the tools to reason precisely about that trade-off:

- Why BF16 and FP16 are not the same, and when the difference matters.
- How GGUF's block-wise quantization schemes (Q4_K_M, Q5_K_S, Q8_0) work at
  the bit level — including the exact layout of scales, super-scales, and
  minimums used by the `_K` variants.

- How vLLM's GPTQ, AWQ, and FP8 paths quantize, where they store the
  calibration data, and what the runtime dequantization path looks like.

- How KV cache quantization (INT8/FP8) differs from weight quantization and
  what it costs in accuracy.

By the end of this chapter you will be able to:

- Compute the exact bits-per-weight and memory footprint of any GGUF quant
  type for a given model.

- Explain the difference between round-to-nearest quantization, GPTQ
  (Hessian-weighted), and AWQ (activation-aware).

- Predict the latency improvement from a given quantization scheme using the
  roofline model from Chapter 2.

- Describe the accuracy degradation profile of each scheme and when to choose
  which.

---

## 10.1 The Precision Hierarchy  `[FOUNDATIONAL]`

### 10.1.1 Floating-point formats

Every modern LLM weight starts life in one of four floating-point formats:

```
Format   Bits   Sign   Exponent   Mantissa   Max value    Min normal
──────────────────────────────────────────────────────────────────────
FP32      32      1       8           23      3.4 × 10³⁸   1.2 × 10⁻³⁸
FP16      16      1       5           10      6.5 × 10⁴    6.1 × 10⁻⁵
BF16      16      1       8            7      3.4 × 10³⁸   1.2 × 10⁻³⁸
FP8 E4M3   8      1       4            3      448.0        ~0.002
FP8 E5M2   8      1       5            2      57344.0      ~6 × 10⁻⁵
```

The critical difference between FP16 and BF16:

```
FP16:  5-bit exponent → dynamic range [6×10⁻⁵, 6.5×10⁴]
       Overflow risk: gradients > 65504 become ±inf (training instability)
       More mantissa bits → higher precision for small values

BF16:  8-bit exponent → same range as FP32 ([10⁻³⁸, 10³⁸])
       No overflow risk: directly truncates FP32 → simple hardware conversion
       7 mantissa bits → lower per-value precision than FP16
       Preferred for LLM weights: activations can be large, range matters more
```

### 10.1.2 Integer formats

```
Format   Bits   Range       Representable values   Use case
───────────────────────────────────────────────────────────────────────
INT8       8    [-128, 127]       256               Activations, weights (GPTQ/AWQ)
UINT8      8    [0, 255]         256                KV cache (shifted)
INT4       4    [-8, 7]           16                Weights (GGUF Q4, GPTQ-4bit)
UINT4      4    [0, 15]           16                Weights (GGUF Q4 shifted)
NF4        4    nonlinear        16                 Weights (QLoRA)
```

### 10.1.3 Bits-per-weight and memory footprint

```
Scheme          Bits/weight   8B model (GB)   70B model (GB)
────────────────────────────────────────────────────────────
FP32                32           29.0            256.0
BF16 / FP16         16           14.5            128.0
FP8                  8            7.3             64.0
Q8_0                ~8.5          7.7             68.0
Q5_K_S              ~5.5          5.0             44.0
Q5_K_M              ~5.7          5.2             45.6
Q4_K_S              ~4.4          4.0             35.2
Q4_K_M              ~4.5          4.1             36.0
Q3_K_M              ~3.4          3.1             27.2
Q2_K                ~2.6          2.4             20.8
```

The `8B model` column uses LLaMA 3 8B's actual parameter count (~8.03B).
Exact formula: `size_GB = n_params × bpw / 8 / 1e9`.

---

## 10.2 Symmetric vs. Asymmetric Quantization  `[FOUNDATIONAL]`

### 10.2.1 Symmetric (zero-point = 0)

```
Quantize:   q = round(x / scale)             scale = max(|x|) / q_max
Dequantize: x̂ = q × scale

Range: q ∈ [-q_max, q_max]   (signed)
     = [0, 2×q_max]          (unsigned, with offset)
```

Symmetric quantization wastes one level if the weight distribution is
asymmetric (e.g., all-positive ReLU activations), but it is hardware-friendly
because the zero-point add is eliminated.

### 10.2.2 Asymmetric (non-zero zero-point)

```
Quantize:   q = round(x / scale) + zero_point
Dequantize: x̂ = (q − zero_point) × scale

scale     = (max(x) − min(x)) / (2^bits − 1)
zero_point = round(−min(x) / scale)
```

Asymmetric quantization uses the full integer range regardless of weight
distribution.  GGUF Q4 variants use a zero-point of 8 (shift unsigned [0,15]
to signed [-8, 7]) — this is symmetric under a fixed offset, not fully
asymmetric.

### 10.2.3 Manual worked example — 4-bit symmetric quantization

```
Weight row (8 values):
  x = [-1.20,  0.85, -0.30,  1.50, -0.70,  0.10,  1.20, -0.95]

Step 1: Find scale
  max_abs = max(|-1.20|, |0.85|, ..., |-0.95|) = 1.50
  q_max   = 7   (signed 4-bit: [-8, 7])
  scale   = 1.50 / 7 = 0.2143

Step 2: Quantize each weight
  q_i = round(x_i / 0.2143)

  x     = [-1.20,  0.85, -0.30,  1.50, -0.70,  0.10,  1.20, -0.95]
  x/s   = [-5.60,  3.97, -1.40,  7.00, -3.27,  0.47,  5.60, -4.43]
  q     = [  -6,    4,    -1,     7,    -3,     0,     6,    -4  ]

  Stored as unsigned (q + 8):
  q_u   = [   2,   12,     7,    15,     5,     8,    14,     4  ]
  (fits in 4 bits: [0..15] ✓)

Step 3: Dequantize
  x̂_i = (q_u_i − 8) × 0.2143

  q_u  = [   2,   12,    7,   15,    5,    8,   14,    4  ]
  q_s  = [  -6,    4,   -1,    7,   -3,    0,    6,   -4  ]
  x̂   = [-1.286, 0.857, -0.214, 1.500, -0.643, 0.000, 1.286, -0.857]

Step 4: Quantization error
  error = x − x̂
        = [ 0.086, -0.007, -0.086, 0.000, -0.057, 0.100, -0.086, -0.093]
  max   |error| = 0.100   (at index 5, where x=0.10 → q=0 → x̂=0)
  RMSE          = 0.073

Memory:
  Original (FP32): 8 × 4 = 32 bytes
  Quantized (Q4):  8 × 0.5 + 4 (scale) = 8 bytes
  Compression:     4× reduction
```

---

### 10.2.4 Manual worked example — 8-bit symmetric quantization (INT8)

Same weight row, same method, 8-bit precision. Compare the error to 4-bit.

```
Weight row (8 values):
  x = [-1.20,  0.85, -0.30,  1.50, -0.70,  0.10,  1.20, -0.95]

Step 1: Find scale
  max_abs = 1.50
  q_max   = 127  (signed INT8: [-128, 127], use 127 for symmetry)
  scale   = 1.50 / 127 = 0.011811

Step 2: Quantize
  q_i = round(x_i / 0.011811)

  x     = [-1.20,  0.85, -0.30,  1.50, -0.70,  0.10,  1.20, -0.95]
  x/s   = [-101.6, 71.9, -25.4, 127.0, -59.3,  8.5, 101.6, -80.4]
  q     = [ -102,   72,   -25,   127,   -59,    9,   102,   -80 ]

Step 3: Dequantize
  x̂_i = q_i × 0.011811

  q     = [ -102,   72,  -25,   127,  -59,    9,  102,  -80 ]
  x̂    = [-1.2047, 0.8504, -0.2953, 1.5000, -0.6968, 0.1063, 1.2047, -0.9449]

Step 4: Quantization error
  error = x − x̂
        = [ 0.005, -0.000, -0.005,  0.000, -0.003, -0.006,  -0.005, -0.005]
  max |error| = 0.006   (vs 0.100 for 4-bit)
  RMSE        = 0.004   (vs 0.073 for 4-bit)

Memory:
  Original (FP32): 8 × 4 = 32 bytes
  Quantized (Q8):  8 × 1 + 2 (FP16 scale) = 10 bytes
  Compression:     3.2× reduction  (vs 4× for Q4, but 18× less error)

Key takeaway: 8-bit cuts the max quantization error by 17× compared to 4-bit.
The compression ratio drops from 4× to 3.2×. This is the core tradeoff.
```

### 10.2.5 Manual worked example — 8-bit asymmetric quantization

Use the same weights but with a non-zero zero-point. This matters for
activation tensors (e.g., post-ReLU) that are entirely non-negative.

```
Activation row (8 values, post-ReLU — all non-negative):
  x = [0.00,  0.85,  0.00,  1.50,  0.00,  0.10,  1.20,  0.00]

Step 1: Find range
  x_min = 0.00,  x_max = 1.50

Step 2: Compute scale and zero-point (UINT8: [0, 255])
  scale      = (x_max − x_min) / (255 − 0) = 1.50 / 255 = 0.005882
  zero_point = round(−x_min / scale) = round(0 / 0.005882) = 0

  (Zero-point is 0 here because x_min = 0. Asymmetric gains most
   when x_min ≠ 0, e.g. activations with a positive floor.)

Step 3: Quantize
  q_i = round(x_i / scale) + zero_point = round(x_i / 0.005882)

  x     = [0.00,  0.85,  0.00,  1.50,  0.00,  0.10,  1.20,  0.00]
  x/s   = [0.0,  144.5,   0.0, 255.0,   0.0,  17.0, 204.0,   0.0]
  q     = [  0,   145,     0,   255,     0,    17,   204,     0  ]

Step 4: Dequantize
  x̂_i = (q_i − zero_point) × scale = q_i × 0.005882

  q     = [  0,   145,    0,  255,    0,   17,  204,    0 ]
  x̂    = [0.000, 0.853, 0.000, 1.500, 0.000, 0.100, 1.200, 0.000]

Step 5: Quantization error
  error = [0.000, -0.003, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
  max |error| = 0.003
  RMSE        = 0.001

Comparison — same data quantized symmetrically (INT8, q_max=127):
  scale_sym = 1.50 / 127 = 0.011811
  The negative half of the range [-128, 0] is entirely wasted on ReLU output.
  Effective resolution: 128 levels instead of 255 levels.
  max |error| for symmetric on this data ≈ 0.006 (2× worse than asymmetric)

Key takeaway: for non-negative activations, asymmetric quantization uses the
full 256-level range, halving the error vs. symmetric. The cost is storing
one extra zero_point integer per tensor.
```

---

## 10.3 GGUF Quantization Types  `[DEEP DIVE]`

### 10.3.1 The Q-type naming convention

GGUF quant type names follow the pattern: `Q{bits}_{variant}_{size_hint}`

```
Q4_0    — 4-bit, version 0 (legacy, 32-weight blocks, one FP16 scale)
Q4_1    — 4-bit, version 1 (adds min value, slightly better accuracy)
Q4_K_S  — 4-bit, K variant, Small  (super-blocks of 256, two scales)
Q4_K_M  — 4-bit, K variant, Medium (mixed precision: some layers at Q6_K)
Q5_K_S  — 5-bit, K variant, Small
Q5_K_M  — 5-bit, K variant, Medium
Q6_K    — 6-bit, K variant (no S/M distinction)
Q8_0    — 8-bit, version 0 (32-weight blocks, one FP32 scale)
```

The `_K` suffix means the scheme uses a **super-block** structure with a
secondary (coarser) scale applied to groups of blocks.

### 10.3.2 Q8_0: simplest GGUF type

```
Block size:  32 weights
Per block:   1 × FP16 scale   (2 bytes)
             32 × INT8 values (32 bytes)
Total:       34 bytes per block

Bits per weight: (34 × 8) / 32 = 8.5 bpw

Layout in memory (one block):
  Bytes 0-1:   d   = FP16 scale     (max_abs / 127)
  Bytes 2-33:  q[] = 32 × INT8      (round(x / d))

Dequantize:  x̂ = q[i] × d
```

The signed range of INT8 is [-128, 127].  GGUF Q8_0 uses [-127, 127]
(symmetric) to avoid the asymmetry at -128.

### 10.3.3 Q4_0: legacy 4-bit

```
Block size:  32 weights
Per block:   1 × FP16 scale  (2 bytes)
             16 bytes of packed nibbles (32 × 4-bit)
Total:       18 bytes per block

Bits per weight: (18 × 8) / 32 = 4.5 bpw

Packing: two 4-bit values per byte
  byte k = (q[2k] & 0xF) | ((q[2k+1] & 0xF) << 4)

Dequantize:
  q_unsigned = byte & 0xF  (lower nibble) or byte >> 4 (upper nibble)
  x̂ = (q_unsigned − 8) × scale
```

### 10.3.4 Q4_K: the K-block super-structure  `[DEEP DIVE]`

Q4_K is the format most users actually run (it's the default for `Q4_K_M`).
Understanding it requires understanding the two-level scale hierarchy.

```
Super-block structure (256 weights):

  ┌─────────────────────────────────────────────────────────┐
  │  Super-block (256 weights)                              │
  │                                                         │
  │  super_scale (FP16):  2 bytes  ← scale for the scales  │
  │  super_min   (FP16):  2 bytes  ← min for the mins      │
  │                                                         │
  │  ┌────────┬────────┬────────┬────────┬────────┬────────┐│
  │  │Block 0 │Block 1 │Block 2 │Block 3 │Block 4 │Block 5 ││
  │  │32 wts  │32 wts  │32 wts  │32 wts  │32 wts  │32 wts  ││
  │  └────────┴────────┴────────┴────────┴────────┴────────┘│
  │  ...8 blocks total                                      │
  │                                                         │
  │  scales[0..7]: 6-bit each (packed into 48 bits = 6 bytes)│
  │  mins[0..7]:   6-bit each (packed into 48 bits = 6 bytes)│
  │  data:         256 × 4-bit  = 128 bytes                  │
  └─────────────────────────────────────────────────────────┘

Total per super-block:
  2 (ss) + 2 (sm) + 6 (scales) + 6 (mins) + 128 (data) = 144 bytes

Bits per weight: (144 × 8) / 256 = 4.5 bpw
```

The two-level structure gives better accuracy than Q4_0 because:
1. The 6-bit per-block scales are quantised more accurately (relative to a
   super-scale) than a raw FP16 scale applied to each block independently.
2. The per-block minimum allows asymmetric representation without storing
   a full FP16 per block.

### 10.3.5 Dequantizing one Q4_K block  `[DEEP DIVE]`

```
Given a Q4_K super-block with:
  super_scale = 0.0120   (FP16)
  super_min   = 0.0015   (FP16)
  scales[b]   = raw 6-bit value for block b
  mins[b]     = raw 6-bit value for block b
  data[b][]   = 32 × 4-bit quantized weights for block b

Recover per-block parameters:
  scale_b = super_scale × scales[b]    ← reconstruct FP32 scale
  min_b   = super_min   × mins[b]      ← reconstruct FP32 minimum

Dequantize weight i in block b:
  q    = (data[b][i] & 0xF)   (4-bit unsigned: [0..15])
  x̂   = q × scale_b + min_b

Note: this is asymmetric quantization — min_b shifts the zero point.
The range covers [min_b, 15 × scale_b + min_b].
```

### 10.3.6 Q4_K_M vs Q4_K_S: mixed precision

`_M` (Medium) and `_S` (Small) differ in which layers get which quant type:

```
Q4_K_S:  All layers quantized at Q4_K
          ~4.37 bpw average

Q4_K_M:  Attention Q/K/V/O projections in certain layers → Q6_K
          FFN layers                                      → Q4_K
          Embedding + lm_head                            → Q8_0
          ~4.83 bpw average
          Slightly higher accuracy at the cost of ~10% more memory
```

The heuristic: Q/K/V projections benefit most from extra bits because small
errors in attention weights compound across all heads and all sequence positions.

### 10.3.7 Accuracy vs. compression: the GGUF trade-off table

```
Type      bpw   LLaMA 3 8B  Perplexity  vs FP16   Memory(GB)
───────────────────────────────────────────────────────────────
FP16     16.0      6.14        —           14.5
Q8_0      8.5      6.15       +0.01         7.7
Q6_K      6.6      6.17       +0.03         5.9
Q5_K_M    5.7      6.21       +0.07         5.2
Q5_K_S    5.5      6.23       +0.09         5.0
Q4_K_M    4.5      6.31       +0.17         4.1
Q4_K_S    4.4      6.34       +0.20         4.0
Q3_K_M    3.4      6.62       +0.48         3.1
Q2_K      2.6      7.28       +1.14         2.4

Source: llama.cpp perplexity benchmark, wikitext-2.
Perplexity: lower is better.  FP16 is the reference.
```

`[COMMON TRAP]` Perplexity is not the whole story.  Q3_K_M may have only
+0.48 perplexity on wikitext-2 but can degrade dramatically on reasoning
tasks or long-context generation where errors compound.  Always benchmark
on your target task, not just perplexity.

---

## 10.4 vLLM's Quantization Paths  `[DEEP DIVE]`

vLLM supports three main post-training quantization (PTQ) schemes: GPTQ, AWQ,
and FP8.  They differ in where and how calibration data is used.

### 10.4.1 GPTQ: Hessian-weighted rounding

GPTQ (Frantar et al., 2022) minimizes the L2 reconstruction error of each
weight row using second-order (Hessian) information from calibration data.

```
Problem: given a weight matrix W ∈ ℝ^(n×m), find quantized Ŵ
         minimizing   ‖ (W − Ŵ) · X ‖²_F
         where X ∈ ℝ^(m×d) is a calibration activation matrix.

Key insight: the Hessian H = 2 X X^T / d gives the importance of each weight.
Weights with high H_ii (columns that are frequently activated) must be
quantized more carefully than weights in near-zero columns.

Algorithm (per row, processing column by column):
  1. Compute H = X X^T  (once per layer, from calibration data)
  2. For each column j (left to right):
     a. Quantize w_j → q_j  using round-to-nearest with scale
     b. Compute quantization error: e_j = w_j − q_j × scale
     c. Propagate error to remaining columns:
        w_{j+1:} -= e_j × H_{j, j+1:} / H_{j,j}   ← Cholesky update
```

The error propagation step is what makes GPTQ better than naive RTN (round-
to-nearest) at the same bit width.

```
GPTQ in vLLM:
  - Quantization happens offline (separate tool: AutoGPTQ or llm-compressor)
  - Quantized weights stored as INT4 with FP16 scales and zeros per group
  - Group size: typically 128 (one scale per 128 weights)
  - Runtime: load quantized weights, dequantize to FP16 before matmul
             OR use fused INT4×FP16 CUDA kernels (ExLlamaV2 kernels)

Memory layout (group_size=128, INT4):
  weight_packed: [n, m // 8]      ← 8 INT4 values packed per INT32
  scales:        [n, m // 128]    ← one FP16 scale per group
  zeros:         [n, m // 128]    ← one INT4 zero-point per group
```

### 10.4.2 AWQ: Activation-Aware Weight Quantization

AWQ (Lin et al., 2023) takes a different approach: instead of adjusting
individual weight values after quantization (like GPTQ), it identifies
**salient weights** — those multiplied by large activations — and protects
them by scaling the weight channel before quantization.

```
Key observation (from AWQ paper):
  Only ~1% of weights are "salient" (corresponding to large input channels).
  Quantization error ∝ activation_scale × weight_error.
  → Protect salient weights by scaling up their channel before quantization,
    then scaling the corresponding activation down at runtime.

Algorithm:
  1. Profile activation magnitudes per input channel using calibration data:
     act_scale[c] = mean(|X[:, c]|)   over calibration samples

  2. Find optimal per-channel scale s[c]:
     s[c] = act_scale[c]^α   (α tuned per model, typically 0.5)

  3. Scale weights before quantization:
     W'[:, c] = W[:, c] × s[c]

  4. Quantize W' with standard RTN (round-to-nearest)

  5. At runtime, compensate by scaling the activation:
     x'[c]    = x[c] / s[c]          (fused into preceding layer norm)
     y = x' @ W'_quantized           (accurate because salient channels scaled up)
```

AWQ has a key advantage over GPTQ: the per-channel scale is computed once
and absorbed into the model, so the runtime dequantization path is identical
to naive INT4 — no extra Hessian-based correction needed.

### 10.4.3 FP8: the hardware-native path

NVIDIA H100 and later GPUs support FP8 (E4M3 and E5M2) natively in Tensor
Cores.  vLLM's FP8 path is the simplest and fastest of the three:

```
FP8 quantization in vLLM:
  - Weights stored as FP8 E4M3 (range: ±448)
  - Per-tensor or per-channel FP32 scale stored alongside
  - Matmul: FP8 × FP8 → FP32 accumulation (H100 Tensor Core)
  - No separate dequantization step — the hardware handles it

Memory layout:
  weight_fp8:  [n, m]         ← FP8 E4M3 values
  weight_scale:[n]  or [1]    ← FP32 scale (per-row or per-tensor)

Speedup vs BF16:
  - 2× memory reduction (8 bit vs 16 bit)
  - H100 FP8 Tensor Core: 3958 TFLOP/s vs 1979 TFLOP/s BF16
    → up to 2× compute throughput
  - In practice: 1.3–1.7× end-to-end speedup (memory-bandwidth bound)

Accuracy:
  - FP8 E4M3 has 3 mantissa bits — less precision than INT8 (7 effective bits)
  - Per-tensor scaling: faster but loses ~0.3–0.5 perplexity points
  - Per-channel scaling: slower to compute but near-lossless
```

### 10.4.4 Comparison: GPTQ vs AWQ vs FP8

```
Property              GPTQ             AWQ              FP8
────────────────────────────────────────────────────────────────────────
Calibration data      Required         Required         Optional
Calibration cost      High (hours)     Moderate (mins)  Low / none
Weight precision      INT4             INT4             FP8
Scale granularity     Group (128)      Channel          Tensor or channel
Runtime dequant       Kernel-fused     Kernel-fused     Hardware-native
Hardware requirement  Any GPU          Any GPU          H100+ required
Accuracy vs BF16      −0.2–0.5 ppl     −0.1–0.3 ppl     −0.1–0.3 ppl
Throughput gain       1.5–2.0×         1.5–2.0×         1.3–1.7×
Best for              Older GPUs       Quality-first    H100 deployments
                      tight memory     production       high throughput
```

---

## 10.5 KV Cache Quantization  `[DEEP DIVE]`

### 10.5.1 Why KV cache quantization is different

Weight quantization is applied once, offline, before deployment.  KV cache
quantization is applied **at runtime, every step**, to tensors that are being
actively written (new tokens) and read (attention over history).

```
KV cache memory per token (LLaMA 3 8B):
  BF16: 2 × n_layers × n_kv_heads × d_head × 2 bytes
      = 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KB per token

  INT8: 2 × 32 × 8 × 128 × 1 = 65,536 bytes = 64 KB per token  (2× savings)
  FP8:  same as INT8 = 64 KB per token

KV block (block_size=16 tokens):
  BF16:  16 × 128 KB = 2.0 MB per block
  INT8:  16 × 64 KB  = 1.0 MB per block
```

With INT8 KV cache, the same GPU memory holds 2× as many KV blocks, which
directly translates to 2× longer context or 2× larger batch size.

### 10.5.2 INT8 KV cache: per-token scaling

Unlike weight quantization (which uses per-block or per-group scales),
KV cache quantization typically uses **per-token scales** because K and V
vectors for different tokens can have very different magnitudes.

```
Write path (new token appended to KV cache):
  1. Compute K, V in BF16 from the attention projection.
  2. For each token t:
       scale_k[t] = max(|K[t, :]|) / 127.0
       K_int8[t]  = round(K[t, :] / scale_k[t])
       (same for V)
  3. Store K_int8, V_int8 in the KV cache blocks.
  4. Store scale_k, scale_v alongside (one FP32 per token per head).

Read path (attention over cached tokens):
  1. Load K_int8[t] for all cached tokens.
  2. Dequantize: K[t] = K_int8[t] × scale_k[t]  (FP32 multiply)
  3. Proceed with standard attention computation.

Scale storage overhead:
  LLaMA 3 8B, seq_len=4096:
    scales: 4096 tokens × 32 layers × 8 kv_heads × 2 (K,V) × 4 bytes = 8.4 MB
    KV data (INT8): 4096 × 32 × 8 × 128 × 2 = 256 MB
    Scale fraction: 8.4 / 256 = 3.3%
    Effective bpw: (256 + 8.4) / (256 / 1) × 8 ≈ 8.26 bpw  (not pure 8 bpw)
```

### 10.5.2b Manual worked example — INT8 KV cache quantization (one token, one head)

Step through the full write→store→read→dequantize cycle for a single K vector.

```
Setup: one attention head, d_head = 8 (simplified from 128)
       one newly computed key vector K[t] in BF16

Input K vector (BF16, one token, one head):
  K[t] = [0.42, -1.31, 0.07, 0.88, -0.55, 1.62, -0.20, 0.94]

──────────────────────────────────────────────────────────────────────
WRITE PATH (quantize before storing)
──────────────────────────────────────────────────────────────────────

Step 1: Compute per-token scale
  max_abs    = max(|0.42|, |−1.31|, ..., |0.94|) = 1.62
  scale_k[t] = 1.62 / 127 = 0.012756

Step 2: Quantize to INT8
  q_i = round(K[t, i] / 0.012756)

  K[t]   = [ 0.42, -1.31,  0.07,  0.88, -0.55,  1.62, -0.20,  0.94]
  K/s    = [32.9, -102.7,   5.5,  69.0, -43.1, 127.0, -15.7,  73.7]
  K_int8 = [  33,  -103,     6,    69,   -43,   127,   -16,    74 ]

  All values ∈ [-127, 127] ✓

Step 3: Store
  K_int8[t] = [33, -103, 6, 69, -43, 127, -16, 74]   (8 × INT8  = 8 bytes)
  scale_k[t] = 0.012756                               (1 × FP32  = 4 bytes)
  Total stored: 12 bytes  (vs 16 bytes BF16 → 1.33× savings for d_head=8)
  At d_head=128: 128 + 4 = 132 bytes vs 256 bytes BF16 → 1.94× savings

──────────────────────────────────────────────────────────────────────
READ PATH (dequantize before attention)
──────────────────────────────────────────────────────────────────────

Step 4: Load and dequantize
  K̂[t, i] = K_int8[t, i] × scale_k[t]

  K_int8 = [  33, -103,    6,   69,  -43,  127,  -16,   74]
  K̂[t]  = [0.421, -1.314, 0.077, 0.880, -0.549, 1.620, -0.204, 0.944]

Step 5: Quantization error
  error  = K[t] − K̂[t]
         = [-0.001, 0.004, -0.007, 0.000, -0.001, 0.000, 0.004, -0.004]
  max |error| = 0.007
  RMSE        = 0.004

  For context: the attention score K·Q / √d uses these values.
  With d_head=128, √d = 11.3. An error of 0.007 in K contributes
  at most 0.007 × max(Q_i) / 11.3 ≈ 0.001 to the attention logit.
  This is below the noise floor for standard attention distributions.

──────────────────────────────────────────────────────────────────────
SUMMARY
──────────────────────────────────────────────────────────────────────
  Memory:   1.94× reduction at d_head=128
  Error:    max 0.007 per dimension (≈0.04% of the 1.62 range)
  Overhead: 1 FP32 scale per token per head (3.3% of KV data)
  Cost:     INT8 multiply per cached token on read path (~same FLOPS)
```

### 10.5.3 Accuracy impact of KV quantization

```
KV quantization error accumulates differently from weight quantization:

  Weight quant:   error is fixed at load time; model adapts via calibration.
  KV quant:       error is introduced dynamically for every new token.
                  At long contexts, errors in early K/V entries corrupt
                  attention scores for all subsequent tokens.

Empirical (LLaMA 3 8B, wikitext-2 perplexity):

  Precision    Perplexity   Δ vs BF16
  ──────────────────────────────────
  BF16 KV        6.14          —
  FP8 KV         6.17         +0.03
  INT8 KV        6.21         +0.07
  INT4 KV        6.89         +0.75   ← significant degradation
  INT4 KV+clip   6.43         +0.29   ← with outlier clipping

The [COMMON TRAP]: INT4 KV cache quantization looks tempting (4× memory
saving) but causes visible quality degradation on tasks involving long
context retrieval. INT8 is generally the safe floor.
```

### 10.5.4 vLLM KV cache quantization configuration

```python
# Enable INT8 KV cache in vLLM
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    kv_cache_dtype="fp8",     # or "int8" (experimental)
    # vLLM uses per-tensor FP8 scaling for KV by default
    # Per-token scaling available via --kv-cache-dtype fp8_e5m2
)

# The number of KV blocks available doubles vs BF16:
# Original KV blocks:  (80GB × 0.9 − 14.5GB weights − 3.2GB activations) / 2MB
#                   = ~26,646 blocks  (Chapter 8 §8.5)
# INT8 KV blocks:    same budget but block is 1MB
#                   = ~53,000 blocks  (≈ 2× more)
```

---

## 10.6 The Quantization-Roofline Connection  `[DEEP DIVE]`

### 10.6.1 Memory bandwidth savings

From Chapter 2 §2.3, decode latency is dominated by reading weights:

```
Decode time ≈ weight_bytes / HBM_bandwidth

LLaMA 3 8B on A100 (HBM = 2000 GB/s):

  BF16:   14.5 GB / 2000 GB/s = 7.25 ms
  Q8_0:    7.7 GB / 2000 GB/s = 3.85 ms  (1.88× faster)
  Q4_K_M:  4.1 GB / 2000 GB/s = 2.05 ms  (3.54× faster)
  Q2_K:    2.4 GB / 2000 GB/s = 1.20 ms  (6.04× faster — but accuracy?)
```

But this is the theoretical maximum — in practice there are overheads:

```
Practical decode speedup (B=1, measured):

  BF16   → Q8_0:   1.5–1.7× (dequant overhead eats ~15% of savings)
  BF16   → Q4_K_M: 2.5–3.0× (dequant overhead ~25% at 4-bit)
  BF16   → FP8:    1.3–1.7× (hardware-native, minimal overhead on H100)
```

### 10.6.2 The dequantization tax

At inference time, GGUF quantized weights must be dequantized before the
dot product.  This adds compute:

```
Q4_K_M dequantization cost per super-block (256 weights):
  Load:     144 bytes (super-block)
  Compute:  256 multiplies + 256 adds + 256/32 scale lookups
            ≈ 550 FLOPs per super-block
            ≈ 2.1 FLOPs per weight

Arithmetic intensity with dequant:
  AI = (FLOPs_matmul + FLOPs_dequant) / bytes_loaded
     = (2 × B × n + 2.1 × n) / (n × 4.5 / 8)
                                          ↑ bytes in Q4_K_M format

For B=1 (decode): ≈ (2 + 2.1) / 0.5625 = 7.3 FLOP/byte
  Still memory-bandwidth bound (ridge point ~156) ✓
  But dequant adds ~50% to the FLOPs-per-byte vs pure 4-bit arithmetic.

For B=64 (prefill): FLOPs dominate, dequant overhead matters less.
```

### 10.6.3 The batch size crossover point

```
At large batch sizes, quantized models can become COMPUTE-bound
while FP16 models remain memory-bound — reducing the relative advantage
of quantization.

Crossover analysis (LLaMA 3 8B, A100):

  Format     Ridge point   Crossover batch size
  ─────────────────────────────────────────────
  BF16 (312T/2000G = 156 FLOP/byte)   B ≈ 156
  Q4_K_M (~1/4 bytes, same FLOPs):    B ≈  39   ← crossover is 4× earlier

Above the crossover, quantization no longer helps decode throughput —
the model is compute-bound and the GEMM efficiency of FP16/BF16 cuBLAS
exceeds INT4 matmul.

Practical implication: for high-throughput serving (large batch),
FP8 is often preferred over INT4 because it stays hardware-native
longer before hitting the compute ceiling.
```

---

## 10.7 ASCII: Quantization Format Comparison  `[FOUNDATIONAL]`

```
Weight storage comparison for one 4096×4096 weight matrix (LLaMA 3 8B):

FP16 (reference):
  ┌──────────────────────────────────────────────────────────────────┐
  │  4096 × 4096 × 2 bytes = 32.0 MB                                │
  │  Layout: [4096 rows × 4096 × 2 bytes]                           │
  │  Scale:  none (FP16 is native)                                   │
  └──────────────────────────────────────────────────────────────────┘

Q8_0:
  ┌──────────────────────────────────────────────────────────────────┐
  │  4096 × 4096 / 32 = 524,288 blocks                              │
  │  Per block: [scale:2B][q0..q31: 32B] = 34 bytes                 │
  │  Total: 524,288 × 34 = 17.8 MB  (0.56× of FP16)                │
  └──────────────────────────────────────────────────────────────────┘

Q4_K_M:
  ┌──────────────────────────────────────────────────────────────────┐
  │  4096 × 4096 / 256 = 65,536 super-blocks                        │
  │  Per super-block:                                                │
  │    [super_scale:2B][super_min:2B]                                │
  │    [scales: 6B][mins: 6B]                                        │
  │    [data: 128B (256×4-bit packed)]                               │
  │    = 144 bytes                                                   │
  │  Total: 65,536 × 144 = 9.4 MB  (0.29× of FP16)                 │
  └──────────────────────────────────────────────────────────────────┘

FP8 (H100+):
  ┌──────────────────────────────────────────────────────────────────┐
  │  4096 × 4096 × 1 byte = 16.0 MB                                 │
  │  Scale: one FP32 per row or per tensor = 4096 × 4 = 16 KB       │
  │  Total: ≈ 16.0 MB  (0.50× of FP16)                              │
  │  No dequant step — H100 Tensor Core natively reads FP8           │
  └──────────────────────────────────────────────────────────────────┘

GPTQ INT4 (group_size=128):
  ┌──────────────────────────────────────────────────────────────────┐
  │  Packed weights: 4096 × 4096 / 2 = 8.0 MB                      │
  │  Scales:  4096 × (4096/128) × 2 bytes = 262 KB                  │
  │  Zeros:   4096 × (4096/128) / 2 bytes = 131 KB                  │
  │  Total: ≈ 8.4 MB  (0.26× of FP16)                               │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 10.8 Choosing a Quantization Strategy  `[FOUNDATIONAL]`

```
Decision tree:

Is the target GPU an H100 or newer?
  YES → Use FP8 (hardware-native, minimal accuracy loss)
  NO  → Continue ↓

Is memory the primary constraint (must fit on GPU)?
  YES → What's the minimum acceptable accuracy?
        Near-lossless: Q8_0 or Q6_K
        Slight loss OK: Q5_K_M or Q4_K_M
        Aggressive:     Q3_K_M (benchmark carefully!)
  NO  → Continue ↓

Is this a production serving deployment (latency SLA)?
  YES → Does calibration data exist?
        YES → AWQ (best accuracy/speed for INT4 on A10G/A100)
        NO  → Q4_K_M via llama.cpp (no calibration needed)
  NO  → Research/dev: use Q4_K_M for fast iteration

Is KV cache size a bottleneck (long context or large batch)?
  YES → Enable FP8 or INT8 KV cache
        (halves KV memory → doubles context length or batch)
```

---

## 10.9 Config Knobs  `[FOUNDATIONAL]`

### vLLM quantization flags

```bash
# Load a pre-quantized AWQ model
vllm serve meta-llama/Meta-Llama-3-8B-AWQ \
  --quantization awq \
  --dtype float16

# Load a GPTQ model
vllm serve TheBloke/Llama-3-8B-GPTQ \
  --quantization gptq \
  --dtype float16

# FP8 weights + FP8 KV cache (H100 recommended)
vllm serve meta-llama/Meta-Llama-3-8B \
  --quantization fp8 \
  --kv-cache-dtype fp8_e5m2

# INT8 KV cache only (weights stay BF16)
vllm serve meta-llama/Meta-Llama-3-8B \
  --kv-cache-dtype int8
```

### llama.cpp quantization flags

```bash
# Convert and quantize a Hugging Face model to Q4_K_M
python convert_hf_to_gguf.py meta-llama/Meta-Llama-3-8B --outtype f16
./llama-quantize llama-3-8b-f16.gguf llama-3-8b-Q4_K_M.gguf Q4_K_M

# Run with Q4_K_M (default: CPU)
./llama-cli -m llama-3-8b-Q4_K_M.gguf -n 256 --prompt "Hello"

# GPU offload (all layers to GPU)
./llama-cli -m llama-3-8b-Q4_K_M.gguf -n 256 --n-gpu-layers 99

# Force all pages resident (no page-fault latency — Chapter 8 §8.8.3)
./llama-cli -m llama-3-8b-Q4_K_M.gguf --mlock --n-gpu-layers 99
```

---

## 10.10 Code Listing  `[FOUNDATIONAL]`

See `code/chapter_10/quantization_demo.py` for:

- FP16/BF16/INT8/INT4 quantization error curves across weight distributions
- GGUF Q8_0, Q4_0, and Q4_K block packing/unpacking simulation
- GPTQ error propagation (simplified, without full Hessian)
- AWQ per-channel scaling simulation
- KV cache INT8 quantization and attention accuracy impact
- Roofline speedup predictions for each scheme

See `code/chapter_10/quantization_demo.cpp` for the C++ parallel implementation
including a bit-accurate GGUF Q4_K super-block encoder/decoder.

---

## 10.11 Summary

```
Key takeaways:

1. BF16 is the baseline for LLM serving — same exponent range as FP32,
   no overflow risk, 2× memory vs FP32.

2. GGUF Q4_K_M is the practical sweet spot for local inference:
   4.5 bpw, +0.17 perplexity vs FP16, 3× memory reduction.
   The K-block super-structure gives it much better accuracy than Q4_0
   at the same bit count.

3. vLLM GPTQ and AWQ both target INT4 weight precision:
   GPTQ is more accurate (Hessian correction) but slower to calibrate.
   AWQ is faster to calibrate and has a cleaner runtime path.

4. FP8 is the best option on H100+: hardware-native, minimal calibration,
   1.3–1.7× speedup, near-lossless accuracy.

5. INT8 KV cache halves KV memory at the cost of +0.07 perplexity.
   INT4 KV cache is dangerous for long-context tasks.

6. The dequantization tax means Q4 quantization gives ~3× real speedup
   not the theoretical 4× you'd expect from the bit count.

7. At large batch sizes (B >> ridge_point / 4), quantization stops helping
   because the model becomes compute-bound. FP8 avoids this problem.
```

---

*Next: Chapter 11 — Continuous Batching and Iteration-Level Scheduling*


---

## Chapter Summary

- **Why quantise**: a 70B FP16 model needs 140 GB — 2× A100 80 GB cards. INT4 brings it to ~35 GB, fitting on one card with headroom for KV cache.
- **Post-training quantization (PTQ)**: weights are quantised after training with no gradient updates; AWQ, GPTQ, and FP8 are the dominant PTQ methods for serving.

> **LinkedIn Scenario Update:** At the LinkedIn scale, serving a 70B BF16 model requires 2 A100-80 GPUs per replica purely for weight storage. Switching to FP8 — the H100-native format described in this chapter — halves the weight memory footprint to ~70 GB, fitting the 70B model on a single H100-80 with ~10 GB to spare for KV cache. For the 50K-user workload running on a multi-replica cluster, FP8 effectively doubles the number of users each GPU can serve: the same $1.2M/month cluster now supports ~100K concurrent users, or the same 50K users at half the cost (~$600K/month).
- **AWQ (Activation-aware Weight quantization)**: protects the 1% of channels with highest activation magnitude from precision loss by rescaling before quantization.
- **GPTQ**: layer-by-layer second-order optimization minimizes reconstruction error; slower to run but achieves lower perplexity than naive round-to-nearest.
- **FP8 (E4M3)**: the native precision of H100/H200 Tensor Cores; supported natively by vLLM for both weights and activations, achieving ~2× throughput over BF16.
- **GGUF formats**: llama.cpp's `Q4_K_M`, `Q5_K_S`, etc., pack 4- or 5-bit integers with per-block scales; the `_K` suffix indicates k-quant with higher-quality scale estimation.
- **KV cache quantization**: vLLM supports INT8/FP8 KV cache independently of weight quantization, trading a small quality loss for a 2× cache capacity increase.
- **Perplexity as the quality proxy**: 1–2 PPL increase from quantization is typically acceptable; >3 PPL increase signals problematic quantization.

---

## Self-Check Questions

1. A 70B model uses 70 × 10⁹ parameters. Compute memory footprint in FP16, INT8, and INT4. For each, state whether it fits on (a) 1× A100 80 GB, (b) 2× A100 80 GB. *(Section 10.1)*

2. AWQ identifies 1% of weight channels as "salient" and scales them by α before quantization. If a channel has max activation 40 and a non-salient channel has max activation 1, what scale α would AWQ assign, and why does this help? *(Section 10.3)*

3. FP8 E4M3 has 4 exponent bits and 3 mantissa bits. What is its representable range, and at what point does an FP16 value overflow when cast to FP8? *(Section 10.5)*

4. llama.cpp's Q4_K_M quantises in blocks of 32 weights with a shared scale and an additional min offset. Why does blocking reduce quantization error compared to a single global scale for the entire weight matrix? *(Section 10.6)*

5. You enable INT8 KV cache quantization. The model has 32 KV heads, d_k = 128, max sequence length 4 096, and 50 concurrent sequences. Compute the KV cache saving in gigabytes versus FP16. *(Section 10.7)*


---

## Worked Solutions

---

### Solution 1 — 70B model memory footprint in FP16, INT8, INT4

**Step 1 — Compute footprints.**

| dtype | bytes/param | Total (70B params) |
|-------|-------------|---------------------|
| FP16 | 2 | 70 × 10⁹ × 2 = **140 GB** |
| INT8 | 1 | 70 × 10⁹ × 1 = **70 GB** |
| INT4 | 0.5 | 70 × 10⁹ × 0.5 = **35 GB** |

**Step 2 — Fit on 1× A100 80 GB?**

(Note: HBM is shared with KV cache and activations — model weights alone must fit comfortably below 80 GB)

| dtype | 1× A100 (80 GB) | 2× A100 (160 GB) |
|-------|-----------------|-------------------|
| FP16 (140 GB) | ❌ Does NOT fit | ✅ Fits |
| INT8 (70 GB) | ⚠️ Barely (70 GB leaves only 10 GB for KV+activations) | ✅ Comfortable |
| INT4 (35 GB) | ✅ Fits with 45 GB for KV cache | ✅ Fits |

**Step 3 — Practical note.**

INT8 "fits" on 1× A100 numerically but leaves only ~10 GB for KV cache. At 128 KB/token (32 layers, 8 KV heads, 128 head_dim, BF16), 10 GB supports only 81,920 tokens — limiting concurrency severely. In practice, 70B INT8 typically requires 2× A100.

---

### Solution 2 — AWQ salient channel scaling

**Given:** Salient channel max activation = 40, non-salient max = 1

**Step 1 — AWQ's scaling formula.**

AWQ computes per-channel scales based on the ratio of salient to non-salient activation magnitude:

$$\alpha = \left(\frac{\max|x_{\text{salient}}|}{\max|x_{\text{non-salient}}|}\right)^\gamma$$

where γ ≈ 0.5 (square root scaling balances protection vs over-scaling).

**Step 2 — Compute α.**

$$\alpha = \left(\frac{40}{1}\right)^{0.5} = \sqrt{40} \approx \textbf{6.32}$$

**Step 3 — How this helps.**

Before quantization, AWQ multiplies the salient weight channel by α = 6.32. This means:
- Original weight value: say w = 0.05 (small value in a salient channel)
- Scaled weight: w' = 0.05 × 6.32 = 0.316

When INT4 quantization maps w' to the nearest representable value, the larger magnitude means fewer rounding errors relative to the true value. After quantization, a compensating scale factor of 1/α = 1/6.32 is absorbed into the preceding layer's activations (which are already large for salient channels).

The insight: quantization error is absolute (±0.5 step_size), but its relative magnitude matters. By making salient weight values larger before quantization, AWQ reduces the *relative* quantization error for the most important channels.

---

### Solution 3 — FP8 E4M3 representable range

**Format:** 1 sign bit + 4 exponent bits + 3 mantissa bits = 8 bits total

**Step 1 — Maximum representable value.**

With 4 exponent bits, bias = 2^(4-1) − 1 = 7. Maximum biased exponent = 14 (the value 15 is reserved for NaN/Inf in E4M3).

Maximum exponent value: 2^(14−7) = 2^7 = 128

Maximum mantissa: 3 bits → 1.111₂ = 1 + 1/2 + 1/4 + 1/8 = 1.875

Maximum E4M3 value: 1.875 × 128 = **240** (some implementations cap at **448** using a different encoding)

**Step 2 — FP16 overflow into FP8 E4M3.**

FP16 values with |x| > 240 (or 448 depending on implementation) **cannot be represented** in FP8 E4M3 and would need to be clipped or saturated.

For LLM activations that occasionally spike to 500–1000 (outlier activations), naive FP8 casting causes catastrophic quantization error. Production solutions:
- **Per-tensor scaling:** divide by a calibration scale factor before casting
- **Static per-channel scaling:** use per-channel scales measured during calibration
- **Dynamic scaling:** compute per-batch scaling at inference time (used in H100 FP8 kernels)

---

### Solution 4 — Block quantization vs global scale error reduction

**Step 1 — The global scale problem.**

With a single scale for the entire weight matrix, the scale is determined by the maximum absolute value across all elements:

$$\text{scale} = \frac{\max|W|}{2^{b-1} - 1}$$

If the matrix has one outlier at 100 and 99% of values near 0.01, the scale is dominated by 100. All values in [−0.01, +0.01] quantize to the same 3–4 nearest codes, losing all precision for the majority of the matrix.

**Step 2 — How blocking helps.**

With blocks of 32, each block gets its own scale:
- Block containing the outlier: scale = 100/7 ≈ 14.3 → 4-bit codes resolve ±7 steps
- Blocks with small values (max = 0.02): scale = 0.02/7 ≈ 0.0029 → same 4 bits now resolve ±0.02 range

Small values are quantized with 1000× better resolution in their local block. The outlier block has coarse resolution, but that only affects a small fraction of weights.

**Step 3 — Quantitative improvement.**

For a weight matrix where 1% of values are outliers at 100× normal magnitude:
- Global scale: mean quantization error for normal weights ≈ 14.3 / 2 ≈ 7.15 (per step)
- Block scale (block size 32): mean error for normal weights ≈ 0.0029 / 2 ≈ 0.0015

That is a ~4,800× reduction in quantization error for the 99% of normal weights. This directly translates to lower perplexity (better model quality) for the same number of bits.

---

### Solution 5 — INT8 KV cache saving

**Given:** 32 KV heads, d_k=128, max_seq_len=4,096, 50 concurrent sequences

**Step 1 — FP16 KV cache size.**

$$\text{per-token} = 32 \times 128 \times 2\text{(K+V)} \times 2\text{ bytes} = 16{,}384 \text{ bytes}$$
$$\text{total} = 16{,}384 \times 4{,}096 \times 50 = 3{,}355{,}443{,}200 \text{ bytes} \approx \textbf{3.125 GB}$$

**Step 2 — INT8 KV cache size.**

$$\text{per-token (INT8)} = 32 \times 128 \times 2 \times 1\text{ byte} = 8{,}192 \text{ bytes}$$
$$\text{total} = 8{,}192 \times 4{,}096 \times 50 = 1{,}677{,}721{,}600 \text{ bytes} \approx \textbf{1.5625 GB}$$

**Step 3 — Saving.**

$$\text{saving} = 3.125 - 1.5625 = \textbf{1.5625 GB}$$

**Step 4 — Opportunity cost.**

The 1.5625 GB saving can support additional KV blocks. At 256 KB/block: 1.5625 GB / 256 KB = 6,400 additional blocks = 6,400 × 16 = 102,400 additional token slots. This allows serving ~25 more concurrent users at 4,096-token context — a 50% capacity increase for 2× precision reduction.

The accuracy trade-off: INT8 KV cache introduces quantization error that may degrade model quality by 0.1–0.5% on benchmarks. For most production use cases, this trade-off is acceptable.

