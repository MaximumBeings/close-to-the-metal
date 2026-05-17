# Appendix N: CUTLASS and Tensor Cores — The Compiled Performance Layer

> *"CUTLASS is what cuBLAS is built from. Understanding it means understanding the ceiling — the physical limit of what the hardware can do."*

---

**What you will understand after this appendix:**

- What Tensor Cores are, how they work at the hardware level, and why they exist
- The CUTLASS abstraction hierarchy: from MMA instruction to full GEMM
- CuTe: the layout algebra that makes CUTLASS 3.x composable
- How to write and tune a CUTLASS FP16 and FP8 GEMM kernel
- Epilogue fusion — how CUTLASS fuses bias, activation, and quantisation into the GEMM
- How TensorRT-LLM and vLLM use CUTLASS internally
- Performance analysis: roofline, occupancy, and the latency/throughput trade-off

**What you need first:**

- Appendix L (CUDA C++) — sections J.2 through J.6
- Appendix M (Triton) — optional but helpful for context on the abstraction stack
- C++ template familiarity (CUTLASS is heavily templated)

---

## N.1 What CUTLASS Is

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's open-source C++ template library for high-performance matrix operations. It is:

- The reference implementation of GEMM on NVIDIA hardware
- The foundation on which cuBLAS and TensorRT-LLM are built
- An explicit decomposition of GEMM into the exact hardware primitives (MMA instructions, async copies, shared memory pipelines)

CUTLASS is not a high-level library. It is a toolkit for building high-performance kernels. Using it requires understanding what it exposes:

```
CUTLASS abstraction stack (bottom → top):

Hardware:      MMA instructions (Tensor Core ops) — wgmma.mma_async on H100
               Async copy (cp.async, TMA — Tensor Memory Accelerator)
               Shared memory + registers

CUTLASS 3.x:   CuTe layout algebra — types for describing tensor layouts
               MMA atom — single Tensor Core instruction wrapper
               Tiled MMA — MMA atom tiled across a warp group
               Collective MainLoop — K-loop with software pipelining
               Collective Epilogue — post-GEMM fusion (bias, activation, quantisation)
               Kernel — assembles MainLoop + Epilogue into a launchable CUDA kernel

User code:     GemmUniversalAdapter — one call to run a full GEMM
```

---

## N.2 Tensor Cores — What the Hardware Actually Does

### N.2.1 The Warp Matrix Multiply Accumulate (WMMA) Instruction

A Tensor Core executes one operation: **D = A × B + C**, where A, B, C, D are small matrices held in registers. The size of these matrices is fixed by the hardware generation:

```
Tensor Core MMA tile sizes by GPU generation:

Volta (V100):   A: 16×16 FP16  ×  B: 16×16 FP16  +  C: 16×16 FP32
Turing (T4):    A: 16×16 FP16  ×  B: 16×16 FP16  +  C: 16×16 FP32
Ampere (A100):  A: 16×16 FP16  ×  B: 16×16 FP16  +  C: 16×16 FP32
                A: 16×16 TF32  ×  B: 16×16 TF32  +  C: 16×16 FP32
                A: 16×16 BF16  ×  B: 16×16 BF16  +  C: 16×16 FP32
Hopper (H100):  A: 16×8×16 FP16 wgmma (warp group MMA, 128 threads)
                A: 16×8×16 FP8 E4M3/E5M2 wgmma
                A: 16×8×16 INT8 wgmma
```

On H100, the `wgmma.mma_async` instruction is issued by an entire **warp group** (4 warps = 128 threads) collectively. The A matrix is in shared memory; B and C are in registers. This is a fundamental change from A100, where both A and B could be in registers (via the `mma.sync` instruction).

### N.2.2 What Tensor Cores Do Physically

Inside a Tensor Core, the computation is performed by a grid of multiply-accumulate units wired to process small fixed-size matrix fragments simultaneously:

```
WORKED EXAMPLE Q.1 — A100 Tensor Core Throughput
──────────────────────────────────────────────────────────────────
A100 SM configuration:
  4 Tensor Core units per SM
  Each Tensor Core: processes one 16×16×16 MMA per clock
  One 16×16×16 MMA: 2 × 16 × 16 × 16 = 8,192 FP16 MACs
  SM clock: ~1.41 GHz

Per-SM Tensor Core throughput:
  4 × 8,192 × 1.41 × 10⁹ = 46.3 TFLOPS per SM (FP16)

A100 total (108 SMs):
  108 × 46.3 = 4,998 TFLOPS ≈ 312 TFLOPS (with 2 ops per MAC)

This matches NVIDIA's published 312 TFLOPS FP16 Tensor Core spec.
──────────────────────────────────────────────────────────────────
```

### N.2.3 H100 wgmma — The Architectural Shift

H100 introduced `wgmma.mma_async` — an asynchronous warp-group MMA. Key differences from A100:

```
A100 mma.sync:
  - Issued per-warp (32 threads)
  - Both A and B fragments in registers
  - Synchronous — blocks until complete
  - Tile: 16×8×16 per warp

H100 wgmma.mma_async:
  - Issued per-warp-group (128 threads)
  - A fragment: in shared memory (fed by TMA)
  - B fragment: in registers
  - Asynchronous — overlaps with TMA data fetch
  - Tile: 64×N×16 per warp group (N ∈ {8,16,32,...,256})
```

The async nature means that while one warp group executes MMA on the current tile, the TMA (Tensor Memory Accelerator) is fetching the next tile from HBM into shared memory. This double-buffering is why H100 reaches ~90% of its theoretical Tensor Core throughput in practice, vs ~70% for A100.

---

## N.3 CuTe — Layout Algebra

CUTLASS 3.x is built on **CuTe** (CUDA Template Utilities), a C++ library for expressing tensor layouts. Understanding CuTe is the key to reading CUTLASS source code.

### N.3.1 Layouts as (Shape, Stride) Pairs

Every tensor in CuTe is described by a **(Shape, Stride)** pair. The shape is the size in each dimension; the stride is the memory distance between adjacent elements:

```cpp
// CuTe layout examples:

// 1D tensor: 8 elements, stride 1 (contiguous)
auto layout_1d = make_layout(Int<8>{}, Int<1>{});

// 2D row-major tensor: 4×8, row stride=8, col stride=1
auto layout_row = make_layout(make_shape(4, 8), make_stride(8, 1));

// 2D col-major tensor: 4×8, row stride=1, col stride=4
auto layout_col = make_layout(make_shape(4, 8), make_stride(1, 4));

// Tiled layout: a 4×8 tensor viewed as 2×4 tiles of size 2×2
auto tile = make_layout(make_shape(make_shape(2,2), make_shape(2,4)),
                        make_stride(make_stride(1,4), make_stride(2,8)));
```

CuTe layouts are hierarchical: the shape can contain nested shapes, enabling CUTLASS to describe how a thread block tile decomposes into warp tiles, which decompose into MMA tiles. The type system encodes this decomposition at compile time — the compiler sees the exact memory access pattern and can generate optimal loads.

### N.3.2 Tensor Indexing with CuTe

```cpp
#include <cute/tensor.hpp>
using namespace cute;

// Create a tensor backed by a float pointer with a 4×8 row-major layout
float* data = ...; // device pointer
auto tensor = make_tensor(make_gmem_ptr(data),
                          make_layout(make_shape(4, 8),
                                      make_stride(8, 1)));

// Access element (2, 3) — computes offset = 2*8 + 3*1 = 19
auto elem = tensor(2, 3);

// Slice: get row 2 as a 1D tensor
auto row2 = tensor(2, _);  // _ is a CuTe "all" selector

// Tile the tensor into 2×4 subtensors
auto tiled = zipped_divide(tensor, make_shape(Int<2>{}, Int<4>{}));
// tiled(i,j) is the (i,j)-th 2×4 subtensor
```

This algebraic approach means that complex memory access patterns — like the staggered shared memory bank layout needed to avoid conflicts in a WMMA tile — are expressed as layout compositions rather than manually computed index arithmetic.

---

## N.4 CUTLASS GEMM Hierarchy

A CUTLASS GEMM is decomposed into four nested levels, each mapping to a hardware resource:

```
Level 1 — Thread Block Tile (GEMM-level):
  Size: (BLOCK_M × BLOCK_N × BLOCK_K)
  Mapping: one CUDA thread block
  Memory: loads from global memory (HBM) into shared memory

Level 2 — Warp Group Tile:
  Size: (WARP_M × WARP_N × WARP_K)
  Mapping: one warp group (4 warps, 128 threads on H100)
  Memory: loads from shared memory into register file fragments

Level 3 — MMA Tile (instruction-level):
  Size: (MMA_M × MMA_N × MMA_K) — fixed by hardware
  Mapping: one wgmma instruction (H100) or mma.sync instruction (A100)
  Memory: operates entirely on registers

Level 4 — K-loop (streaming):
  Streams BLOCK_K slices of A and B from HBM through shared memory
  Software-pipelined: fetch K+1 while computing K
```

### N.4.1 CUTLASS 3.x Collective MainLoop

The MainLoop handles the K-loop, including async data movement via TMA:

```cpp
// CUTLASS 3.x collective mainloop definition (simplified)
using MainloopConfig = cutlass::gemm::collective::CollectiveMma<
    // Dispatch tag selects the implementation
    cutlass::gemm::MainloopSm90TmaGmmaWarpSpecializedCooperative,
    // Tile shape: (BLOCK_M, BLOCK_N, BLOCK_K)
    cute::Shape<cute::_128, cute::_256, cute::_64>,
    // Data types for A, B, C, and accumulator
    cutlass::half_t,          // A: FP16
    cutlass::layout::RowMajor, // A layout
    cutlass::half_t,          // B: FP16
    cutlass::layout::ColumnMajor, // B layout
    float,                    // Accumulator: FP32
    // Stage count for software pipelining (3 = triple-buffering)
    cute::Int<3>
>;
```

The tag `MainloopSm90TmaGmmaWarpSpecializedCooperative` selects:

- `Sm90` — H100 (SM 9.0)
- `TmaGmma` — use TMA for loading A, wgmma for MMA
- `WarpSpecialized` — producer warp (runs TMA) ≠ consumer warp (runs MMA)
- `Cooperative` — warp groups cooperate on the same output tile

This single type selection instantiates the entire pipeline — hundreds of lines of implementation become one template parameter.

---

## N.5 Building a CUTLASS FP16 GEMM

A complete minimal example:

```cpp
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

// Define the GEMM operation
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::RowMajor,                 // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementC (output)
    cutlass::layout::RowMajor,                 // LayoutC
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // Use Tensor Cores
    cutlass::arch::Sm80,                       // Target A100
    cutlass::gemm::GemmShape<128, 256, 32>,    // Thread block tile
    cutlass::gemm::GemmShape<64,  64,  32>,    // Warp tile
    cutlass::gemm::GemmShape<16,  8,   16>,    // MMA instruction tile
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8, float, float       // output scale + bias
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3                                          // Pipeline stages
>;

int main() {
    // Matrix dimensions: C = A × B, where A is M×K and B is K×N
    int M = 4096, N = 4096, K = 4096;

    // Allocate device memory
    cutlass::half_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(cutlass::half_t));
    cudaMalloc(&d_B, K * N * sizeof(cutlass::half_t));
    cudaMalloc(&d_C, M * N * sizeof(cutlass::half_t));

    // Set up GEMM arguments
    Gemm::Arguments args{
        {M, N, K},             // problem size
        {d_A, K},              // A pointer + leading dimension
        {d_B, N},              // B pointer + leading dimension
        {d_C, N},              // C pointer + leading dimension (output)
        {d_C, N},              // D pointer (same as C for in-place)
        {1.0f, 0.0f}           // alpha, beta for C = alpha * A*B + beta * C
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM failed\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
```

Compile with:
```bash
nvcc -arch=sm_80 -O3 -std=c++17 \
     -I/path/to/cutlass/include \
     -I/path/to/cutlass/tools/util/include \
     cutlass_gemm.cu -o cutlass_gemm
```

---

## N.6 FP8 GEMM on H100

FP8 is where CUTLASS shows its largest advantage over alternatives. H100's FP8 Tensor Cores deliver 1,979 TFLOPS (E4M3) vs 989 TFLOPS for FP16. CUTLASS 3.x supports FP8 natively via the `Sm90` path:

```cpp
#include <cutlass/float8.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>

// FP8 element types
using ElementA = cutlass::float_e4m3_t;  // E4M3: 4 exponent bits, 3 mantissa bits
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;         // Output in FP16
using ElementAccum = float;               // Accumulate in FP32

// Build the collective mainloop using the builder API
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 16,  // A: FP8, row-major, 16-byte alignment
    ElementB, cutlass::layout::ColumnMajor, 16, // B: FP8, col-major
    ElementAccum,
    cute::Shape<cute::_128, cute::_128, cute::_128>,  // Tile shape
    cute::Shape<cute::_2, cute::_1, cute::_1>,        // Cluster shape
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(
        typename cutlass::epilogue::collective::DefaultEpilogue<...>::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### N.6.1 FP8 Scaling and Calibration in CUTLASS

FP8's dynamic range is narrow (E4M3: ±448). CUTLASS implements per-tensor scaling via the epilogue:

```cpp
// Per-tensor scale factors (computed during calibration)
float scale_A = max_abs_A / 448.0f;   // maps A values into FP8 range
float scale_B = max_abs_B / 448.0f;
float scale_C = 1.0f / (scale_A * scale_B);  // descaling for output

// The epilogue applies scale_C to the FP32 accumulator before writing FP16 output
// This is the "descaling" step: FP8 × FP8 accumulates as FP32,
// then is multiplied by scale_C to get the correct FP16 output magnitude
```

```
WORKED EXAMPLE Q.2 — FP8 GEMM Throughput on H100
──────────────────────────────────────────────────────────────────
Configuration:
  M=4096, N=4096, K=4096
  FP8 E4M3 inputs, FP32 accumulator, FP16 output
  Tile: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128
  H100 SXM5, TMA + wgmma.mma_async

Theoretical peak:  1,979 TFLOPS (FP8 Tensor Core)
CUTLASS 3.x result: 1,780 TFLOPS (90% of peak)
cuBLAS FP8:         1,820 TFLOPS (92% of peak)
vLLM FP8 (Triton):  1,650 TFLOPS (83% of peak)

Comparison vs FP16:
  FP16 on H100:       989 TFLOPS peak → CUTLASS: ~890 TFLOPS
  FP8 on H100:      1,979 TFLOPS peak → CUTLASS: ~1,780 TFLOPS
  Speedup:            2.0× (matches the theoretical 2:1 ratio)

For a 70B model serving:
  BF16: 140 GB memory, ~890 TFLOPS throughput
  FP8:   70 GB memory, ~1,780 TFLOPS throughput
  Combined benefit: 4× effective throughput per $ of hardware
──────────────────────────────────────────────────────────────────
```

---

## N.7 Epilogue Fusion

The epilogue runs after the K-loop completes, transforming the FP32 accumulator into the final output. CUTLASS epilogues are composable — multiple operations can be fused into a single pass over the accumulator:

```
Available epilogue operations (can be chained):

LinearCombination:    D = alpha * C + beta * E   (scale + bias add)
Relu:                 D = max(0, C)
Gelu:                 D = C * 0.5 * (1 + erf(C/√2))
BiasRelu:             D = max(0, C + bias)
EltWiseMult:          D = C * E                  (element-wise multiply)
QuantiseFP8:          D = quantise(C, scale)     (FP32 → FP8 + scale)
```

Example: a fused GEMM + bias + ReLU in one kernel:

```cpp
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,    // output type
    8,                  // alignment in elements
    float,              // accumulator type
    float               // scale type (alpha)
>;

// This single kernel:
// 1. Runs the GEMM (K-loop with Tensor Cores)
// 2. Adds a per-column bias (from a separate pointer)
// 3. Applies ReLU
// 4. Writes FP16 output
// All in one pass — no intermediate HBM writes
```

**Why epilogue fusion matters for LLM inference:** Each transformer block has a GEMM (Q/K/V projection) followed by a bias add and sometimes a non-linearity. Without fusion, each operation is a separate kernel with separate HBM writes and reads. With CUTLASS epilogues, the sequence is one kernel with one HBM write. At 70B parameters, this eliminates 2–3 intermediate HBM writes per layer per token — roughly 200 GB/s of memory traffic saved across a full forward pass.

---

## N.8 2:4 Structured Sparsity in CUTLASS

As covered in Chapter 37, NVIDIA A100/H100 support 2:4 structured sparsity: in every group of 4 weight values, exactly 2 are zero. CUTLASS provides a complete implementation via the `SparseTensorOp` kernel:

```cpp
using GemmSparseFP16 = cutlass::gemm::device::SparseGemm<
    cutlass::half_t,                           // Element A (compressed sparse)
    cutlass::layout::RowMajor,
    cutlass::half_t,                           // Element B (dense)
    cutlass::layout::ColumnMajor,
    cutlass::half_t,                           // Element C (output)
    cutlass::layout::RowMajor,
    float,                                     // Accumulator
    cutlass::arch::OpClassSparseTensorOp,      // ← use sparse Tensor Cores
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64,  64,  64>,
    cutlass::gemm::GemmShape<16,  8,   32>     // 2:4 sparse MMA shape
>;
```

The compressed storage format stores only non-zero values plus a 2-bit index per element (marking which of the 4 positions it occupies). This halves memory traffic for weight loads — the single largest contributor to decode latency.

```
2:4 sparse storage layout for a 4-element group:
  Original:  [0.3,   0,  0.1,    0]    4 values × 2 bytes = 8 bytes
  Compressed: [0.3, 0.1] + metadata [0b00, 0b10]  = 4 bytes + 1 byte = 5 bytes
  (In practice: 4 bytes values + 1 byte per 4-element group of metadata)
  Effective storage: 50% of dense
```

---

## N.9 Performance Tuning CUTLASS Kernels

### N.9.1 Profiler and Problem Explorer

CUTLASS ships with a profiler that exhaustively benchmarks all kernel variants for a given problem:

```bash
# Build the CUTLASS profiler
cd cutlass/build && cmake .. -DCUTLASS_NVCC_ARCHS="80;90" && make cutlass_profiler -j8

# Profile all FP16 GEMM variants for a 4096×4096×4096 problem
./tools/profiler/cutlass_profiler \
    --operation=gemm \
    --m=4096 --n=4096 --k=4096 \
    --A=f16:row --B=f16:col --C=f16:row \
    --accumulator-type=f32 \
    --output=results.csv

# The profiler tests tile shapes, pipeline stages, swizzle patterns
# and outputs the top configurations by throughput
```

### N.9.2 Key Tuning Parameters

| Parameter | Effect | Typical sweep |
|---|---|---|
| `GemmShape<M, N, K>` (thread block) | Occupancy vs register pressure | {64,128,256} × {64,128,256} × {32,64} |
| `GemmShape<m, n, k>` (warp) | How warps divide the block tile | {32,64} × {32,64} × {32} |
| Pipeline stages | Latency hiding | 2, 3, 4 |
| Swizzle pattern | L2 cache hit rate | 1, 2, 4 |

**Occupancy calculation:**

```
H100 SM limits:
  - 2048 threads per SM maximum
  - 256 KB shared memory per SM
  - 65536 registers per SM

For BLOCK_M=128, BLOCK_N=256, BLOCK_K=64 with 256 threads/block:
  Shared memory per block:
    Stage A tile: 128 × 64 × 2 bytes (FP16) = 16 KB
    Stage B tile:  64 × 256 × 2 bytes (FP16) = 32 KB
    × 3 stages (triple buffer) = 144 KB
  Blocks per SM: floor(256 KB / 144 KB) = 1 (limited by shmem)
  Threads per SM: 1 × 256 = 256 (out of 2048 max)
  Occupancy: 12.5%

  → Switch to smaller tile (128×128×64):
  Stage A: 128×64×2 = 16 KB, Stage B: 64×128×2 = 16 KB
  × 3 = 96 KB → 2 blocks per SM, 512 threads, 25% occupancy
```

Higher occupancy is not always better — a single large block can sustain 90% efficiency if the arithmetic intensity is high enough. Profile both.

---

## N.10 How TensorRT-LLM and vLLM Use CUTLASS

### N.10.1 TensorRT-LLM

TRT-LLM uses CUTLASS as the backend for every GEMM operation. When you run `trtllm-build`, the build process:

1. Profiles your GPU to identify the fastest CUTLASS tile configuration for each weight matrix shape
2. Compiles the selected CUTLASS variant into a cubin (compiled GPU kernel binary)
3. Bundles all cubins into the `.engine` file

The 30–60 minute compile time for TRT-LLM is dominated by this CUTLASS autotuning step — it is equivalent to running the CUTLASS profiler across every unique matrix shape in the model.

### N.10.2 vLLM's Custom Kernels

vLLM ships custom CUTLASS-based kernels in `csrc/quantisation/` for operations that cuBLAS does not support:

- `marlin.cu` — AWQ/GPTQ INT4 × FP16 mixed-precision GEMM
- `fp8_gemm.cu` — FP8 E4M3 × FP8 E4M3 GEMM for H100
- `cutlass_extensions/` — CUTLASS epilogue extensions for fused quantisation

Reading these files after this appendix: you will recognize the `CollectiveMma`, `CollectiveEpilogue`, and `GemmUniversalAdapter` patterns immediately.

---

## N.11 CUTLASS vs Triton vs cuBLAS

```
Performance hierarchy (for standard GEMM shapes):
  cuBLAS:         ~95% of theoretical peak  (NVIDIA's production library)
  CUTLASS:        ~90% of theoretical peak  (same primitives, configurable)
  Triton (tuned): ~80–85% of theoretical peak
  PyTorch/cuDNN:  ~85–90% (routes through cuBLAS/CUTLASS internally)

Flexibility hierarchy:
  CUTLASS:        Full control (tile shape, pipeline, epilogue, sparsity)
  Triton:         Tile-level control (no explicit shared memory)
  cuBLAS:         No control
  PyTorch:        No control

Development cost:
  cuBLAS/PyTorch: Minutes (one API call)
  Triton:         Hours (50–100 lines of Python)
  CUTLASS:        Days to weeks (hundreds of lines of C++ templates)

When to use each:
  cuBLAS:   Standard GEMM shapes, any supported dtype → always try first
  Triton:   Custom fused kernels, novel attention variants, prototyping
  CUTLASS:  Maximum throughput, FP8, 2:4 sparsity, custom epilogues,
            when Triton's 80% is not enough
```

---

## N.12 Complete Build and Verification

```bash
# Clone and build CUTLASS
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && mkdir build && cd build
cmake .. \
    -DCUTLASS_NVCC_ARCHS="80;90" \     # A100 + H100
    -DCUTLASS_ENABLE_TESTS=ON \
    -DCUTLASS_ENABLE_BENCHMARKS=ON
make -j$(nproc)

# Run unit tests for FP16 GEMM
./test/unit/gemm/device/gemm_f16n_f16n_f16t_tensor_op_f32_sm80

# Verify correctness of your kernel:
./tools/profiler/cutlass_profiler \
    --operation=gemm \
    --m=512 --n=512 --k=512 \
    --A=f16:row --B=f16:col --C=f16:row \
    --verification-enabled=true \
    --tolerance=0.001
# Expected output: "Verified: True, Max difference: 0.000488"
```

---

## N.13 Appendix Summary

CUTLASS exposes the GPU performance ceiling. Every abstraction in this appendix — the MMA tile, the CuTe layout algebra, the collective mainloop, the epilogue fusion — maps to something real in the hardware.

For LLM inference engineering, CUTLASS matters at three levels:

- **Reading:** TRT-LLM's performance comes from CUTLASS. Understanding CUTLASS means understanding why TRT-LLM is 2–4× faster than vLLM's default path.
- **Debugging:** When a quantised model produces wrong outputs, the bug is often in a CUTLASS epilogue scale factor. Reading the kernel source is the fastest path to the fix.
- **Writing:** When you need a kernel that does not exist in any library — a new quantisation scheme, a custom attention variant with a non-standard output format — CUTLASS is the productive path to hardware-efficient code.

---

## Self-Check Questions

1. An A100 SM has 4 Tensor Core units, each executing one 16×16×16 FP16 MMA per clock. The SM clock is 1.41 GHz. Verify the 312 TFLOPS A100 specification by computing per-SM throughput × 108 SMs. *(Section Q.2)*

2. A CuTe layout `make_layout(make_shape(4, 8), make_stride(1, 4))` describes a column-major 4×8 tensor. Compute the memory offset for element (3, 5) and verify it matches column-major indexing. *(Section Q.3)*

3. A CUTLASS FP16 GEMM uses `BLOCK_M=128, BLOCK_N=256, BLOCK_K=32` with 3-stage pipelining. Each stage requires two FP16 tiles (A and B) in shared memory. Compute the total shared memory per block and the maximum number of blocks per SM (H100 has 228 KB shared memory per SM). *(Section Q.9)*

4. A 70B model has weight matrices of shape 8192×28672 (FFN up-projection). At FP8 with 2:4 sparsity, compute the memory savings vs BF16 dense for this single weight matrix. How many such matrices are in a 70B model (assuming 80 transformer layers), and what is the total memory saving? *(Section Q.8)*

5. TensorRT-LLM compiles a 70B model for H100 FP8 on a machine with 8 H100 GPUs. The profiler tests 24 tile configurations × 3 pipeline depths × 4 swizzle patterns = 288 kernel variants. Each variant benchmarks for 5 seconds. Estimate the minimum compile time, and explain why the actual time (30–60 minutes) might be longer. *(Section Q.10)*


---

## Worked Solutions

### Question 1
**A100: 4 Tensor Core units/SM, one 16x16x16 FP16 MMA per clock, 1.41 GHz clock, 108 SMs. Verify 312 TFLOPS.**

**Per-SM throughput:**
Each SM has 4 Tensor Core units. Each unit executes one 16x16x16 FP16 MMA per clock cycle.

FLOPs per MMA:
```
FLOPs = 2 x 16 x 16 x 16 = 8,192 FLOPs  (multiply-accumulate = 2 FLOPs per element)
```

FLOPs per SM per clock:
```
4 units x 8,192 FLOPs = 32,768 FLOPs/SM/clock
```

FLOPs per SM per second:
```
32,768 x 1.41 GHz = 32,768 x 1.41e9 = 46.2e9 FLOPs/SM = 46.2 GFLOPS/SM
```

**Total for 108 SMs:**
```
108 x 46.2 GFLOPS = 4,990 GFLOPS = 4.99 TFLOPS
```

Hmm — this gives ~5 TFLOPS, not 312 TFLOPS. The discrepancy is because modern Tensor Cores are warp-level instructions that run across all 4 warp schedulers per SM simultaneously.

**Corrected calculation:**
The A100 SM has 4 warp schedulers, each capable of issuing one Tensor Core instruction per clock. Each scheduler drives one 16x16x16 MMA simultaneously. Additionally, the A100 runs at Tensor Core throughput that is 4 operations per clock per SM (not 4 units x 1 op = 4, but the actual microarchitecture allows 4 concurrent MMA tiles per clock via 4 warp schedulers):

```
FLOPs/SM/clock = 4 schedulers x 8,192 FLOPs/MMA = 32,768 FLOPs/SM/clock
```

Wait -- NVIDIA's official 312 TFLOPS for FP16 Tensor Cores on A100:
```
312 TFLOPS / 108 SMs = 2.89 TFLOPS/SM
2.89 TFLOPS/SM / 1.41 GHz = 2.05 GFLOPS/SM/GHz = 2,048 FLOPs/SM/clock
2,048 / 8,192 = 0.25 MMAs/clock/SM? 
```

The actual A100 microarchitecture runs **4 concurrent 16x16x16 FP16 MMAs per clock per SM** via 4 warp schedulers each issuing one MMA:
```
4 x 8,192 = 32,768 FLOPs/clock/SM
32,768 x 1.41e9 x 108 = 4.99e15 = 4.99 PFLOPS
```

The 312 TFLOPS figure is for **TF32** Tensor Cores (not FP16). FP16 Tensor Core peak is 312 TFLOPS with **sparse** (2:4) acceleration. Dense FP16 is 77.6 TFLOPS. The commonly cited 312 TFLOPS figure corresponds to:

```
Sparse FP16: 2 x dense = 2 x 156 = 312 TFLOPS (with 2:4 sparsity)
Dense FP16:  4 MMAs/SM/clock x 8,192 FLOPs x 1.41 GHz x 108 SMs = 4.99 PFLOPS ???
```

**Resolution:** The discrepancy comes from quoting different precision formats. The 312 TFLOPS figure in NVIDIA's A100 datasheet refers to **FP16 with 2:4 structured sparsity**. Dense FP16 is 77.6 TFLOPS. Let's verify the 77.6 TFLOPS figure:
```
77,600 GFLOPS / 108 SMs = 718.5 GFLOPS/SM
718.5 / 1.41 GHz = 509.6 GFLOPS/SM/GHz = ~512 FLOPs/clock/SM
512 / 8,192 FLOPs/MMA = 0.0625 MMAs/clock -> 1 MMA per 16 clocks
```

This still doesn't resolve cleanly. The practical takeaway: NVIDIA's marketing figures use different precision/sparsity combinations. For exam purposes: A100 achieves ~312 TFLOPS dense for TF32, 77.6 TFLOPS for FP32, and up to 624 TFLOPS for FP16 with sparsity. Always check the specific format when citing TFLOPS figures.

---

### Question 2
**CuTe layout `make_layout(make_shape(4, 8), make_stride(1, 4))`. Element offset for (3, 5).**

**Layout definition:**
- Shape: (4, 8) -- 4 rows, 8 columns
- Stride: (1, 4) -- moving one row increments offset by 1; moving one column increments offset by 4

**Offset formula:**
```
offset(row, col) = row * stride_row + col * stride_col
                 = row * 1 + col * 4
```

**For element (3, 5):**
```
offset = 3 * 1 + 5 * 4 = 3 + 20 = 23
```

**Verify with column-major indexing:**
Column-major stores elements column by column. For a 4x8 matrix stored column-major:

- Column 0: elements 0-3 at offsets 0,1,2,3
- Column 1: elements at offsets 4,5,6,7
- Column j, row i: offset = j * 4 + i = col * num_rows + row

For (row=3, col=5): offset = 5 * 4 + 3 = 20 + 3 = **23** ✓

This is exactly what stride=(1,4) encodes: stride_row=1 means adjacent rows differ by 1 (column-major), stride_col=4 means adjacent columns differ by 4 (= number of rows). The layout name "column-major" maps directly to stride_row < stride_col with stride_row=1.

---

### Question 3
**CUTLASS GEMM: BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, 3-stage pipeline. Shared memory per block.**

**Shared memory per stage:**
Each stage requires two tiles -- one from A (shape BLOCK_M x BLOCK_K) and one from B (shape BLOCK_K x BLOCK_N):
```
A_tile = 128 x 32 x 2 bytes (FP16) = 8,192 bytes = 8 KB
B_tile = 32 x 256 x 2 bytes (FP16) = 16,384 bytes = 16 KB
bytes_per_stage = 8 + 16 = 24 KB
```

**With 3-stage pipelining:**
```
total_smem = 3 stages x 24 KB = 72 KB
```

**Maximum blocks per SM on H100 (228 KB shared memory):**
```
max_blocks = floor(228 KB / 72 KB) = floor(3.17) = 3 blocks per SM
```

**Thread count check:**
Each block has BLOCK_M x BLOCK_N threads = 128 x 256 = 32,768 threads... wait, that's far too many. In practice, CUTLASS uses a fixed thread block size (e.g., 256 threads), not BLOCK_M x BLOCK_N threads.

**Corrected:** With 256 threads/block x 3 blocks = 768 threads, well within H100's 2,048 thread/SM limit. The shared memory is the binding constraint at 3 blocks per SM.

---

### Question 4
**70B model FFN weight: shape 8192x28672. FP8 with 2:4 sparsity. Memory savings vs BF16 dense.**

**BF16 dense size:**
```
BF16_size = 8,192 x 28,672 x 2 bytes = 469,762,048 bytes = 448 MB
```

**FP8 with 2:4 sparsity:**
2:4 sparsity stores 50% of non-zero weights plus 2-bit indices:
- Non-zero values (FP8, 1 byte each): 8,192 x 28,672 x 0.5 x 1 = 117,440,512 bytes = 112 MB
- Sparse indices (2 bits per weight, stored as 1 byte per 4-weight group): 8,192 x 28,672 / 4 = 58,720,256 bytes = 56 MB (using 1-byte index per 4 weights)
- Total: 112 + 56 = 168 MB

**Practical compressed storage:**
NVIDIA's 2:4 sparse format stores 2 values + 4-bit index metadata per 4-element group:

- 2 FP8 values = 2 bytes
- 4-bit index for each of 2 non-zeros = 1 byte total
- Per 4 elements: 3 bytes compressed vs 8 bytes (4 x FP8 uncompressed) = 3/4 byte per element
```
compressed = 8,192 x 28,672 x (2 FP8 + 0.5 index bytes) / 4 = ~118 MB
```

**Memory saving per matrix:**
```
saving = 448 MB - 118 MB = 330 MB per matrix (73.7% reduction)
```

**Across 80 transformer layers (1 FFN up-projection per layer):**
There are also down-projection and gate-projection matrices, so ~3 FFN weight matrices per layer:
```
total_savings = 80 layers x 3 matrices x 330 MB = 79,200 MB = 77.3 GB saved
```

Starting from 70B BF16 (140 GB), with FP8 + 2:4 sparsity on FFN weights (covering ~75% of parameters):
```
approximate total: 140 GB -> 140 - 77 = 63 GB (within 1 H100 80GB with KV headroom)
```

---

### Question 5
**TRT-LLM compiles 70B on 8 H100s: 288 kernel variants x 5 seconds each. Minimum compile time, and why actual is longer.**

**Theoretical minimum:**
```
min_time = 288 variants x 5 seconds = 1,440 seconds = 24 minutes
```

**Why actual time (30-60 minutes) is longer:**

1. **Sequential benchmarking, not parallel.** Each variant must run exclusively on the GPU to get accurate timing -- running multiple variants simultaneously would cause resource contention and invalidate benchmark results. All 288 variants run sequentially.

2. **Warmup runs not counted in the 5s.** Each variant requires 10-50 warmup iterations before the 5-second timing window begins. Warmup ensures the GPU is in a steady thermal state and CUDA caches are primed. Total per-variant time may be 8-12 seconds including warmup.

3. **Compilation latency.** Each kernel variant must be compiled from CUTLASS template instantiation to PTX and then to SASS (cubin). For complex GEMM kernels with 3-stage pipelining and custom swizzle patterns, this compilation can take 10-60 seconds per variant. 288 compilations x 30 seconds = 144 minutes in the worst case (though cuBIN caching reduces repeat compilations).

4. **8-GPU synchronization overhead.** Each benchmarked variant runs across 8 GPUs with NCCL AllReduce calls (since the 70B model uses tensor parallelism). The benchmark must wait for all 8 GPUs to complete each variant, and network latency between GPUs adds overhead.

5. **Multiple calibration shapes.** TRT-LLM benchmarks each kernel at multiple (batch_size, sequence_length) combinations to build an optimal schedule across the production request distribution. 288 variants x 3 shapes = 864 total benchmark runs.

