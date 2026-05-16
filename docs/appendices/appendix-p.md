# Appendix P: Introduction to Triton — Python-Embedded GPU Kernel Programming

> *"Triton sits exactly where you want to be: above the CUDA threading model, below the library abstraction layer. You write in Python. The hardware sees optimised PTX."*

---

**What you will understand after this appendix:**

- What Triton is and why it exists alongside CUDA
- The Triton programming model: programs, blocks, and the absence of explicit threads
- How `@triton.jit` compiles Python to PTX via LLVM
- Writing your first Triton kernel: vector addition
- Tile-based GEMM from scratch with real performance numbers
- Fused softmax and its relationship to Flash Attention
- Autotuning with `triton.autotune`
- How FlashAttention-2 uses Triton and what you can read directly in the source

**What you need first:**

- Appendix J (CUDA C++) — especially sections J.2 (GPU hardware) and J.5 (memory hierarchy)
- Python comfort (decorators, NumPy-style indexing)
- No prior Triton experience required

---

## P.1 Why Triton Exists

CUDA C++ is powerful but unforgiving. Writing a high-performance matrix multiplication in CUDA requires understanding warps, shared memory bank conflicts, register pressure, async copies, and WMMA intrinsics — typically 300–600 lines of careful C++ for a single kernel variant. A small mistake in tiling strategy cuts throughput by 3×.

Triton (open-sourced by OpenAI in 2021) solves this by raising the abstraction level by exactly one step. Instead of programming individual threads, you program **tiles** — contiguous blocks of memory that Triton maps to shared memory and tensor cores automatically. The programmer specifies *what* data to operate on (the tile shape and position); Triton decides *how* to do it (register allocation, shared memory layout, async pipelines).

```
Abstraction levels for GPU programming:

CUDA C++ / PTX
  Programmer controls: threads, warps, shared memory, register allocation
  Typical kernel: 300-600 lines for a production GEMM
  Time to write: days to weeks

Triton
  Programmer controls: tile shapes, memory access patterns, arithmetic
  Typical kernel: 50-100 lines for a production GEMM
  Time to write: hours

cuBLAS / CUTLASS library calls
  Programmer controls: matrix sizes and types
  No control over internals
```

The trade-off: Triton gives up some fine-grained control (you cannot directly allocate shared memory or schedule individual instructions) in exchange for productivity and portability. For most LLM kernels, Triton reaches 80–95% of hand-tuned CUDA throughput with 10× less code.

**Triton in production:** FlashAttention-2, FlashAttention-3, SGLang's radix attention kernel, vLLM's custom attention and quantisation kernels, and the Liger Kernel library (fused LLM training ops) are all written in Triton. Reading these kernels after this appendix will be straightforward.

---

## P.2 The Triton Programming Model

### P.2.1 What Triton Hides

In CUDA, you write one function that runs once per thread. With 128×128 = 16,384 threads in a block, you write the per-thread logic and the hardware schedules everything else.

Triton inverts this. You write one function that runs once per **program instance** (roughly equivalent to a CUDA thread block). Inside that function, you operate on **tensors** — 1D or 2D arrays — using SIMD-like operations that apply to every element of the tile simultaneously.

```
CUDA mental model:
  kernel<<<grid, block>>>(args)
  grid = (M/32, N/32)   ← one block per tile
  block = (32, 32)       ← 1024 threads
  each thread: computes one output element

Triton mental model:
  kernel[(grid_size,)](args)
  grid_size = (M // BLOCK_M) * (N // BLOCK_N)
  each program: computes one BLOCK_M × BLOCK_N output tile
  no threads visible to programmer
```

### P.2.2 Program IDs

Every Triton program instance receives a unique `program_id` — its position in the launch grid:

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(axis=0)         # which block am I?
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # [start, start+1, ..., start+BS-1]
    mask = offsets < n_elements          # guard against out-of-bounds
    x = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x * 2.0, mask=mask)
```

`tl.arange(0, BLOCK_SIZE)` returns a vector of integers — the core of Triton's tile abstraction. Operations on this vector apply to all BLOCK_SIZE elements simultaneously.

### P.2.3 The `tl.constexpr` Annotation

Parameters marked `tl.constexpr` are **compile-time constants**. Triton specialises (recompiles) the kernel for each unique combination of `constexpr` values. This is how tile shapes become tunable without runtime overhead — the compiler unrolls loops and generates shape-specific PTX.

```python
BLOCK_SIZE: tl.constexpr = 128   # kernel compiled specifically for 128-wide tiles
```

---

## P.3 First Kernel: Vector Addition

The canonical "hello world" of GPU programming:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,     # pointers to input/output tensors
    n_elements,               # total number of elements
    BLOCK_SIZE: tl.constexpr, # tile width (compile-time constant)
):
    # Step 1: figure out which elements this program instance owns
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Step 2: guard against the last block exceeding array bounds
    mask = offsets < n_elements

    # Step 3: load from HBM into registers
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Step 4: compute (applies to all BLOCK_SIZE elements simultaneously)
    c = a + b

    # Step 5: store result back to HBM
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    # Grid: one program instance per block of BLOCK_SIZE elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return c

# Test
a = torch.rand(1 << 20, device='cuda')   # 1M elements
b = torch.rand(1 << 20, device='cuda')
c = vector_add(a, b)
assert torch.allclose(c, a + b)
print("vector_add: correct")
```

**What happens when you call `vector_add_kernel[grid](...)`:**

1. Triton compiles the function to LLVM IR (if not already cached)
2. LLVM compiles to PTX
3. The PTX is JIT-compiled to SASS (machine code for your specific GPU)
4. CUDA launches `grid[0]` thread blocks, each running the compiled kernel
5. Results: The programmer saw Python; the GPU ran optimised SASS

---

## P.4 Memory Access Patterns

### P.4.1 Coalesced Loads

Triton's `tl.load` coalesces adjacent memory accesses automatically when the offsets form a contiguous range. This is the most important performance property:

```
Coalesced load (fast — one HBM transaction):
  offsets = block_start + tl.arange(0, 128)
  → elements [0,1,2,...,127] loaded in a single 512-byte HBM burst

Strided load (slow — many HBM transactions):
  offsets = tl.arange(0, 128) * 64   # every 64th element
  → 128 separate 4-byte HBM requests (64× more transactions)
```

For LLM inference, the most common pattern is loading weight rows (contiguous) or loading KV cache entries (often non-contiguous — a challenge Triton handles with gather operations).

### P.4.2 2D Tile Loads with Block Pointers

For 2D operations (matrix multiply, attention), Triton provides **block pointers** — a structured way to describe a 2D tile within a larger 2D tensor:

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)   # row block index
    pid_n = tl.program_id(1)   # column block index

    # Tile offsets for this program instance
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [pid_m*BM, ..., pid_m*BM+BM-1]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the A and B tiles for the first K-block
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator (stays in registers throughout the K-loop)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop: accumulate across the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)      # ← this maps to Tensor Core MMA instructions
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)
```

`tl.dot(a, b)` is the key: Triton compiles this to **Tensor Core MMA instructions** (WGMMA on H100, HMMA on A100) automatically when the tile shapes are compatible.

---

## P.5 Worked Example: GEMM Performance

```
WORKED EXAMPLE P.1 — Triton GEMM vs cuBLAS
──────────────────────────────────────────────────────────────────
Hardware: A100-80GB SXM
Matrix:   M=4096, N=4096, K=4096, dtype=FP16
Tile:     BLOCK_M=128, BLOCK_N=256, BLOCK_K=32

Theoretical peak (A100 FP16 Tensor Cores): 312 TFLOPS
FLOPs for this GEMM: 2 × M × N × K = 2 × 4096³ = 137.4 GFLOPs

cuBLAS result:    ~280 TFLOPS  (90% of peak) — hand-tuned by NVIDIA
Triton (tuned):   ~250 TFLOPS  (80% of peak) — 3 hours of work
Triton (default): ~180 TFLOPS  (58% of peak) — 30 minutes of work
PyTorch (naive):   ~60 TFLOPS  (19% of peak) — one line of Python

Key insight: Triton closes 80-90% of the gap to cuBLAS with 10×
less code than a hand-written CUDA GEMM. For custom operations
(fused attention, custom quantisation), this gap is irrelevant
because cuBLAS does not support them at all.
──────────────────────────────────────────────────────────────────
```

---

## P.6 Fused Softmax — A Core LLM Primitive

Softmax appears in every attention computation. The naive implementation requires three passes over data (find max, compute exp, divide). Triton makes it trivial to fuse all three:

```python
@triton.jit
def fused_softmax_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)         # one program per row
    row_start = row_idx * stride_row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Load entire row into registers (must fit: n_cols <= BLOCK_SIZE)
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=-float('inf'))

    # Pass 1: find row max (for numerical stability)
    x_max = tl.max(x, axis=0)

    # Pass 2: shifted exp
    x = tl.exp(x - x_max)

    # Pass 3: normalise
    x = x / tl.sum(x, axis=0)

    tl.store(output_ptr + row_start + cols, x, mask=mask)
```

This is a single HBM read + single HBM write — 2× memory traffic vs the naive three-pass approach. For a 4096-token sequence with 64 attention heads, fused softmax saves ~20 ms per forward pass.

**Why this matters for FlashAttention:** FlashAttention's key insight is the *online softmax* — computing the normaliser incrementally across K-blocks so the full Q×K matrix never materialises. This is exactly what Triton's tile abstraction enables. The FlashAttention-2 Triton kernel (in `flash_attn/flash_attn_triton.py`) is a direct extension of the fused softmax pattern above, with the accumulator tracking both the running max and the running sum.

---

## P.7 Autotuning

Triton's autotuner exhaustively searches tile shape configurations and picks the fastest for your hardware and matrix size:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,
                       'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,
                       'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
                       'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,
                       'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],   # re-tune when matrix shape changes
)
@triton.jit
def matmul_kernel_autotuned(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ... same body as before ...
    pass
```

**What `num_stages` controls:** Triton's software pipelining. `num_stages=4` means four K-blocks are prefetched while the current block is being computed. On H100, `num_stages=3` or `4` typically maximises SM occupancy by hiding HBM latency behind compute.

**What `num_warps` controls:** The number of warps per thread block. More warps = more occupancy = better latency hiding. Fewer warps = more registers per warp = possible for larger tile sizes. Finding the sweet spot is why autotuning exists.

**Autotuning cache:** Triton caches autotuning results in `~/.triton/autotune/`. On first run for a new (M, N, K) shape, expect 5–30 seconds of profiling. Subsequent runs use the cache.

---

## P.8 Triton vs CUDA vs cuBLAS: When to Use Each

```
Decision framework:

Operation exists in cuBLAS/cuDNN/FlashAttention?
  YES → Use the library. It is hand-tuned and well-tested.
  NO  ↓

Is the operation compute-bound (GEMM-like)?
  YES → Triton (reaches 80-90% of cuBLAS for novel operations)
  NO  ↓

Is the operation memory-bandwidth bound and simple (element-wise)?
  YES → Triton (fused kernels eliminate HBM round-trips)
  NO  ↓

Do you need control over warp-level synchronisation, shared memory
bank layout, or inline PTX?
  YES → CUDA C++ (Appendix J, or CUTLASS — Appendix Q)
  NO  → Triton is probably sufficient
```

### P.8.1 Triton Limitations

- **No explicit shared memory:** Triton manages shared memory internally. You cannot directly allocate `__shared__` arrays or control bank layout. This matters for reductions and irregular access patterns.
- **No inline PTX:** You cannot drop down to raw PTX inside Triton. For WMMA intrinsics or hardware-specific instructions, use CUDA.
- **NVIDIA-only in practice:** Triton has AMD ROCm and Intel backends, but maturity varies. For portability, MLC-LLM's TVM/Relax is more reliable (Appendix R).
- **Debugging is harder:** GPU-side printf does not work. Use `triton.testing.assert_close` and small test cases.

---

## P.9 Reading FlashAttention-2 in Triton

With the primitives above, you can now read the actual FlashAttention-2 source (`flash_attn/flash_attn_triton.py`). The key sections:

```python
# From flash_attn_triton.py (simplified for explanation)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,          # Q/K/V pointers, softmax scale = 1/sqrt(d)
    L, O,                        # running logsumexp (L) and output (O) pointers
    ...
    BLOCK_M: tl.constexpr,       # tile along sequence dim (rows of Q)
    BLOCK_N: tl.constexpr,       # tile along sequence dim (cols of K/V)
    BLOCK_DMODEL: tl.constexpr,  # head dimension
):
    # One program per (batch, head, BLOCK_M query tokens)
    start_m = tl.program_id(0)

    # Running statistics for online softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # running sum of exp
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)   # output accumulator

    # Load Q tile (stays in SRAM throughout)
    q = tl.load(Q_block_ptr)  # [BLOCK_M, BLOCK_DMODEL]

    # Iterate over K/V tiles (streamed from HBM)
    for start_n in range(0, seqlen_k, BLOCK_N):
        k = tl.load(K_block_ptr)   # [BLOCK_DMODEL, BLOCK_N]
        v = tl.load(V_block_ptr)   # [BLOCK_N, BLOCK_DMODEL]

        # Compute Q @ K^T scores for this tile
        qk = tl.dot(q, k)           # [BLOCK_M, BLOCK_N]
        qk *= sm_scale

        # Online softmax update
        m_ij = tl.max(qk, axis=1)           # new max for this K-block
        p = tl.exp(qk - m_ij[:, None])      # exp shifted by new max
        l_ij = tl.sum(p, axis=1)

        # Rescale accumulator with the correction factor
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)       # correction for old blocks
        beta  = tl.exp(m_ij - m_i_new)      # correction for new block
        l_i_new = alpha * l_i + beta * l_ij

        acc *= (alpha / l_i_new)[:, None]
        acc += (beta / l_i_new)[:, None] * tl.dot(p.to(tl.float16), v)

        m_i = m_i_new
        l_i = l_i_new

    tl.store(O_block_ptr, acc.to(tl.float16))
```

This is the exact algorithm from section 5.2 of the FlashAttention-2 paper — the online normaliser update that lets you process the full sequence in O(1) memory. The Triton version adds roughly 50 lines of boilerplate around this core. Reading the real source is now a matter of recognising the same primitives.

---

## P.10 Performance Profiling Triton Kernels

### P.10.1 `triton.testing.do_bench`

```python
import triton.testing

# Benchmark: returns median runtime in milliseconds
ms = triton.testing.do_bench(
    lambda: matmul_kernel[grid](a, b, c, M, N, K, ...),
    warmup=25,      # warmup iterations (not timed)
    rep=100,        # measured iterations
)

flops = 2 * M * N * K
tflops = flops / ms * 1e-9
print(f"Time: {ms:.3f} ms | Throughput: {tflops:.1f} TFLOPS")
```

### P.10.2 Nsight Systems Integration

Triton kernels appear in Nsight Systems with their auto-generated names. To add a human-readable name:

```python
with torch.cuda.nvtx.range("my_matmul"):
    matmul_kernel[grid](a, b, c, ...)
```

In Nsight Systems, this block will be labelled and you can measure:

- Kernel duration
- SM occupancy
- HBM bandwidth (GB/s)
- L2 hit rate

A healthy Triton GEMM shows: >85% SM occupancy, HBM bandwidth near roofline, L2 hit rate >60% (for repeated K-blocks).

---

## P.11 Practical Recipes for LLM Inference

### P.11.1 Fused Dequantise + GEMV (INT4 → FP16)

For decode-phase inference, the bottleneck is memory bandwidth. Fusing weight dequantisation and the matrix-vector product into one kernel eliminates an extra HBM write:

```python
@triton.jit
def dequant_gemv_kernel(
    weight_q4_ptr,    # INT4 weights, packed: 2 per byte
    scale_ptr,        # FP16 per-group scales
    x_ptr,            # FP16 input vector
    out_ptr,          # FP16 output vector
    M, K,
    group_size: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros([1], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        # Load packed INT4 (two values per byte)
        packed = tl.load(weight_q4_ptr + row * (K // 2) + k_offs // 2,
                         mask=k_offs < K)
        # Unpack: low nibble and high nibble
        w_lo = (packed & 0xF).to(tl.float16) - 8.0   # dequantise
        w_hi = (packed >> 4).to(tl.float16) - 8.0
        scale = tl.load(scale_ptr + row * (K // group_size) + k_offs // group_size)
        w_lo *= scale
        w_hi *= scale
        # Load input
        x = tl.load(x_ptr + k_offs, mask=k_offs < K)
        acc += tl.sum(w_lo * x[0::2] + w_hi * x[1::2], axis=0)

    tl.store(out_ptr + row, acc.to(tl.float16))
```

This pattern is used in vLLM's GPTQ and AWQ kernel implementations.

### P.11.2 Rotary Position Embedding (RoPE) Fusion

RoPE is applied to every Q and K tensor before attention. The naive implementation allocates intermediate tensors; the fused version does it in-place in one pass:

```python
@triton.jit
def apply_rope_kernel(
    q_ptr, cos_ptr, sin_ptr,
    seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (token, head)
    token_idx = tl.program_id(0)
    head_idx  = tl.program_id(1)

    half_dim = head_dim // 2
    offs = tl.arange(0, BLOCK_SIZE // 2)  # indices for first half

    base = token_idx * head_dim * num_heads + head_idx * head_dim
    q0 = tl.load(q_ptr + base + offs)           # first half
    q1 = tl.load(q_ptr + base + offs + half_dim) # second half

    cos = tl.load(cos_ptr + token_idx * half_dim + offs)
    sin = tl.load(sin_ptr + token_idx * half_dim + offs)

    # RoPE rotation: [q0, q1] → [q0*cos - q1*sin, q1*cos + q0*sin]
    tl.store(q_ptr + base + offs,           q0 * cos - q1 * sin)
    tl.store(q_ptr + base + offs + half_dim, q1 * cos + q0 * sin)
```

---

## P.12 Installation and Setup

```bash
# Triton requires CUDA 11.6+ and Python 3.8+
pip install triton            # latest release
# or for the development version:
pip install triton-nightly    # newer features, less stable

# Verify:
python -c "import triton; print(triton.__version__)"
# Expected: 2.x or 3.x

# Compile cache location:
echo $HOME/.triton/
```

Triton ships with PyTorch 2.0+ as an optional dependency. If you have `torch>=2.0`, you likely already have Triton installed.

---

## P.13 Appendix Summary

Triton occupies the productive middle ground between cuBLAS (no control) and CUDA C++ (full control, high cost). Its tile abstraction maps directly to how modern GPUs execute work — tiles fit in shared memory, `tl.dot` maps to Tensor Core instructions, and the K-loop maps to the prefetching pipeline.

For LLM inference, Triton is most valuable for:

- **Novel fused kernels** not in any existing library (custom quantisation schemes, new attention variants)
- **Rapid prototyping** of ideas from papers before committing to CUDA C++
- **Reading existing kernels** (FlashAttention-2, vLLM's attention, SGLang's paged attention) — all are in Triton and now readable

The reference progression for GPU kernel expertise in LLM inference: start with Appendix J (CUDA fundamentals) → this appendix (Triton for productive kernel writing) → Appendix Q (CUTLASS for maximum-performance GEMM) → Appendix R (Mojo for the future of portable high-performance AI code).

---

## Self-Check Questions

1. A Triton kernel is launched with `grid = (M // 128, N // 256)` and `BLOCK_M=128, BLOCK_N=256`. How many program instances are launched for M=4096, N=8192? What does each program instance compute? *(Section P.2)*

2. Explain why `tl.dot(a, b)` in a Triton kernel with `BLOCK_M=128, BLOCK_N=256, BLOCK_K=64` produces Tensor Core instructions on an A100, but the same call with `BLOCK_K=3` does not. *(Section P.4)*

3. The fused softmax kernel in §P.6 requires the entire row to fit in `BLOCK_SIZE` registers. For a 128K-context model with sequence length 131,072 and FP32 accumulation, how much register memory does one row require? Why does this constraint force FlashAttention to use the tiling approach instead? *(Section P.6)*

4. An autotuner runs 6 configurations × 50 warmup + 100 timed iterations each. Each kernel call takes 2 ms. Estimate the total autotuning time for one (M, N, K) shape. When does this cost amortise? *(Section P.7)*

5. The fused dequantise + GEMV kernel in §P.11.1 loads INT4 weights packed 2 per byte and applies per-group FP16 scales. For a 7B-parameter model with 4096 rows × 4096 columns in each weight matrix and group size=128, compute the total HBM reads for one forward pass token, and compare to serving the same model in BF16. *(Section P.11)*
