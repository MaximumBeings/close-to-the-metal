# Appendix L: Introduction to CUDA C++ for LLM Inference

> *"CUDA does not hide the hardware — it exposes it. Once you understand the hardware, the API is almost obvious."*

---

**What you will understand after this appendix:**

- How an NVIDIA GPU is physically organized, and how CUDA threads map to that hardware
- The memory hierarchy (registers → shared → L2 → HBM) and the performance implication of each level
- How to write, launch, and debug CUDA kernels from first principles
- The three laws of GPU performance: occupancy, coalescing, and avoiding divergence
- Four complete kernels written from scratch: vector addition, matrix-vector product, softmax, and INT8 GEMV

**What you need first:**

- Comfortable with C++17 (pointers, structs, templates, lambdas)
- Basic understanding of what a GPU does (it runs many threads in parallel)
- No prior CUDA experience required

---

## L.1 What CUDA Is and Why It Matters

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel programming model, released in 2007. It lets you write C++ functions that run on thousands of GPU threads simultaneously. Every LLM inference engine covered in this book — vLLM, TensorRT-LLM, SGLang — has CUDA kernels at its innermost loop.

Understanding CUDA is not optional for serious inference engineering. You will encounter it when:

- Reading a flash attention kernel to understand why it is fast
- Debugging a quantization kernel that produces wrong outputs
- Writing a custom attention variant not yet in any library
- Profiling why a kernel is at 40% of the hardware roof

This appendix teaches CUDA with a consistent focus: **LLM inference patterns**. Every example is chosen because it appears directly in vLLM, llama.cpp, or their dependencies.

---

## L.2 GPU Hardware — The Physical Machine You Are Programming

Before writing a single line of CUDA, you must understand what you are programming. CUDA abstractions are thin wrappers over real hardware.

### L.2.1 Streaming Multiprocessors (SMs)

An NVIDIA GPU is a collection of **Streaming Multiprocessors (SMs)**.

```
  H100 SXM GPU — 132 SMs
  ┌────────────────────────────────────────────────────────────┐
  │  SM 0   SM 1   SM 2   SM 3   SM 4   SM 5   SM 6   SM 7   │
  │  SM 8   SM 9  SM 10  SM 11  SM 12  SM 13  SM 14  SM 15   │
  │  ...                                                        │
  │  SM124 SM125 SM126 SM127 SM128 SM129 SM130 SM131           │
  ├────────────────────────────────────────────────────────────┤
  │                 L2 Cache (50 MB on H100)                   │
  ├────────────────────────────────────────────────────────────┤
  │              HBM3 — 80 GB, 3.35 TB/s                      │
  └────────────────────────────────────────────────────────────┘
```

Each SM is an independent processing unit. SMs share the L2 cache and HBM but otherwise operate independently.

### L.2.2 Inside a Single SM (H100)

```
  One SM — H100
  ┌─────────────────────────────────────────────────────────┐
  │  Warp Schedulers × 4                                    │
  │  ┌──────────────────────────────────────────────────┐   │
  │  │  Issue one warp instruction per clock per sched  │   │
  │  └──────────────────────────────────────────────────┘   │
  │                                                          │
  │  FP32 CUDA Cores × 128   |   FP64 Cores × 64           │
  │  Tensor Cores × 4 (4th gen, FP8/FP16/BF16/INT8)        │
  │                                                          │
  │  Register File: 65,536 × 32-bit registers               │
  │  Shared Memory / L1 Cache: 228 KB (configurable)        │
  └─────────────────────────────────────────────────────────┘
```

Key numbers to memorize for an H100 SM:

- 128 FP32 CUDA cores
- 65,536 registers (32-bit each)
- Up to 228 KB shared memory / L1
- 4 Tensor Cores (FP8/FP16/BF16/INT8 matrix multiply-accumulate)

### L.2.3 Warps — The Atomic Unit of Execution

The GPU does not execute individual threads. It executes **warps** — groups of exactly 32 threads that execute the same instruction in lockstep (SIMT: Single Instruction, Multiple Threads).

```
  32 threads in a warp
  ┌───────────────────────────────────────────────────────────────┐
  │ T0  T1  T2  T3  T4  T5  T6  T7  T8  T9 T10 T11 T12 T13 ... │
  │ ← all execute the SAME instruction at the SAME time →        │
  └───────────────────────────────────────────────────────────────┘
  
  Each thread has its own:  registers, program counter (PC), predicate bits
  All threads share:        instruction fetch, decode, execution unit dispatch
```

`[FOUNDATIONAL]` **The warp is the most important concept in CUDA performance.** If all 32 threads in a warp take the same path through an `if` statement, the warp executes it in one shot. If threads take different paths (warp divergence), the GPU serializes the paths — halving throughput for a two-way divergence.

### L.2.4 SM Counts by GPU Generation

| GPU | SMs | FP32 cores/SM | Tensor Cores/SM | HBM BW |
|---|---|---|---|---|
| V100 | 80 | 64 | 8 (1st gen, FP16) | 900 GB/s |
| A100 | 108 | 64 | 4 (3rd gen, BF16/FP16/INT8) | 2.0 TB/s |
| H100 SXM | 132 | 128 | 4 (4th gen, FP8) | 3.35 TB/s |
| H200 SXM | 132 | 128 | 4 (4th gen, FP8) | 4.8 TB/s |
| B200 SXM | 160 | 128 | 4 (5th gen, FP4) | 8.0 TB/s |

---

## L.3 The Thread Hierarchy — Threads, Warps, Blocks, Grids

CUDA organizes threads into a three-level hierarchy that maps directly to hardware.

```
  CUDA Thread Hierarchy
  
  Grid (one per kernel launch)
  ┌──────────────────────────────────────────────────────────────┐
  │  Block (0,0)   Block (1,0)   Block (2,0)   Block (3,0)      │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
  │  │ T(0..31) │  │ T(0..31) │  │ T(0..31) │  │ T(0..31) │    │
  │  │ T(32..63)│  │ T(32..63)│  │ T(32..63)│  │ T(32..63)│    │
  │  │ T(64..95)│  │ T(64..95)│  │ T(64..95)│  │ T(64..95)│    │
  │  │T(96..127)│  │T(96..127)│  │T(96..127)│  │T(96..127)│    │
  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
  │  Block (0,1)   Block (1,1)   Block (2,1)   Block (3,1)      │
  │  ...                                                          │
  └──────────────────────────────────────────────────────────────┘
  
  Hardware mapping:
  - Each block → assigned to ONE SM
  - One SM can hold multiple blocks simultaneously (if register/smem fits)
  - Threads within a block → scheduled as warps of 32
```

### L.3.1 Built-in Variables

Inside any CUDA kernel, these variables identify where you are:

```cpp
// Thread within its block (3D): threadIdx.x, threadIdx.y, threadIdx.z
// Block within the grid (3D):   blockIdx.x,  blockIdx.y,  blockIdx.z
// Block dimensions:             blockDim.x,  blockDim.y,  blockDim.z
// Grid dimensions:              gridDim.x,   gridDim.y,   gridDim.z
```

For a 1D kernel operating on a vector of length N:

```cpp
int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
```

### L.3.2 Choosing Block and Grid Dimensions

`[FOUNDATIONAL]` Two rules:

1. **Block size must be a multiple of 32** (warp size). Non-multiples waste execution slots. Common choices: 128, 256, 512.
2. **Grid size = ceil(N / blockSize)**. This ensures every element gets a thread.

```
WORKED EXAMPLE L.1 — Grid and Block Sizing for N=1,048,576 Elements
─────────────────────────────────────────────────────────────────────
Given:   N = 1,048,576 elements, block_size = 256 threads
Step 1:  grid_size = ceil(N / block_size) = ceil(1,048,576 / 256) = 4,096 blocks
Step 2:  Total threads = 4,096 × 256 = 1,048,576 (exactly covers N)
Step 3:  Warps per block = 256 / 32 = 8 warps
Step 4:  Total warps = 4,096 × 8 = 32,768 warps
Note:    H100 has 132 SMs × up to 64 resident warps/SM = 8,448 simultaneously
         resident warps. 32,768 total warps are distributed in waves.
─────────────────────────────────────────────────────────────────────
```

---

## L.4 Your First CUDA Kernel — Vector Addition

The GPU equivalent of "Hello, World":

```cpp
// vector_add.cu
#include <cuda_runtime.h>
#include <stdio.h>

// ── The kernel: runs on GPU, called from CPU ────────────────────────────────
__global__ void vector_add(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float*       __restrict__ c,
                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {                   // guard: last block may have excess threads
        c[i] = a[i] + b[i];
    }
}

// ── Host code: runs on CPU, manages GPU memory ──────────────────────────────
int main() {
    const int N = 1 << 20;         // 2^20 = 1,048,576 elements
    const int bytes = N * sizeof(float);

    // 1. Allocate GPU memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 2. Initialize host data and copy to GPU
    float* h_a = new float[N];
    float* h_b = new float[N];
    for (int i = 0; i < N; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 3. Launch kernel
    int block_size = 256;
    int grid_size  = (N + block_size - 1) / block_size;
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);

    // 4. Copy result back and verify
    float* h_c = new float[N];
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    printf("h_c[0] = %.1f (expected 3.0)\n", h_c[0]);   // should print 3.0

    // 5. Free
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    return 0;
}
// Build: nvcc -O2 -o vector_add vector_add.cu
// Run:   ./vector_add
```

### L.4.1 Kernel Qualifiers

| Qualifier | Runs on | Called from | Notes |
|---|---|---|---|
| `__global__` | GPU | CPU (or GPU for dynamic parallelism) | The main kernel entry point |
| `__device__` | GPU | GPU only | Helper function called from a kernel |
| `__host__` | CPU | CPU only | Default; explicit when combined with `__device__` |
| `__host__ __device__` | Both | Both | Compiled twice; used for math helpers |

### L.4.2 The `<<<grid, block>>>` Launch Syntax

```cpp
kernel_name<<<grid_dim, block_dim, shared_mem_bytes, stream>>>(args...);
//              ^            ^           ^                 ^
//         # blocks    threads/block  dynamic smem      CUDA stream
//         (dim3 or int) (dim3 or int) (default 0)    (default 0)
```

`dim3` can represent 1D, 2D, or 3D dimensions:
```cpp
dim3 grid(32, 8, 1);    // 32×8×1 = 256 blocks
dim3 block(16, 16, 1);  // 16×16×1 = 256 threads per block
kernel<<<grid, block>>>(...);
```

---

## L.5 The Memory Hierarchy — The Most Important Performance Topic

`[FOUNDATIONAL]` GPU performance is almost always limited by memory bandwidth, not compute. Understanding the memory hierarchy is the single most productive CUDA skill.

```
  GPU Memory Hierarchy (H100)
  
  Speed →  Fastest                                          Slowest
           ↓                                                ↓
  ┌────────────────────────────────────────────────────────────────┐
  │ Registers │ Shared Mem │  L1/Texture │    L2 Cache  │  HBM3   │
  │  Per-thread│  Per-block │  Per-SM     │  Per-GPU     │ Per-GPU │
  │  ~65K regs │  Up to 228KB│ Part of 228KB│   50 MB     │  80 GB  │
  │  ~19 TB/s  │  ~33 TB/s   │  ~33 TB/s   │ ~12 TB/s    │ 3.35TB/s│
  │  0 cycles  │  ~20 cycles │  ~30 cycles  │ ~193 cycles │ ~600 cy │
  └────────────────────────────────────────────────────────────────┘
  
  Scope:    1 thread       1 block       1 SM          Whole GPU   Whole GPU
```

### L.5.1 Global Memory (HBM) — The Default

When you do `cudaMalloc`, you get global memory — HBM (High Bandwidth Memory). It is:

- **Accessible by all threads in all blocks**
- **Persistent** for the lifetime of the allocation
- **Slow**: ~600 cycles latency, 3.35 TB/s bandwidth (H100)
- **Large**: 80 GB on H100

All kernel input/output goes through global memory. The goal is to minimize how often you hit it, and when you do, to hit it in patterns that maximize bandwidth (coalescing).

### L.5.2 Shared Memory — The GPU's Scratchpad

Shared memory is fast, on-chip memory that all threads in a **block** can read and write:

- **Scope**: one block only (other blocks cannot see it)
- **Speed**: ~20 cycles latency, ~33 TB/s bandwidth
- **Size**: up to 228 KB per SM on H100 (configurable; more shared memory = fewer resident blocks)
- **Lifetime**: exists only while the block is executing

Declaring shared memory in a kernel:

```cpp
__global__ void my_kernel(float* d_in, float* d_out, int n) {
    // Static shared memory: size known at compile time
    __shared__ float smem[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from slow global memory into fast shared memory
    smem[tid] = (gid < n) ? d_in[gid] : 0.0f;

    // Synchronize: ensure all threads in the block have loaded
    __syncthreads();

    // Now read from fast shared memory many times
    // ... process smem[tid], smem[tid ± 1], etc. ...

    // Write result back to global memory (one write, not many reads)
    d_out[gid] = smem[tid] * 2.0f;
}
```

`[FOUNDATIONAL]` **The shared memory pattern:**

1. Load a tile from global memory → shared memory (one global read per element)
2. `__syncthreads()` to ensure all threads have loaded
3. Process the tile from shared memory (many fast reads)
4. Write result back to global memory (one global write per element)

This pattern — **tiling** — is the basis of Flash Attention, matrix multiplication, and almost every high-performance CUDA kernel.

### L.5.3 Registers — Fastest of All

Each thread has its own private registers. Local variables in a kernel become registers automatically:

```cpp
__global__ void add_one(float* d_x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // register
    if (i < n) {
        float val = d_x[i];   // register — loaded once from global
        val += 1.0f;           // register arithmetic — essentially free
        d_x[i] = val;          // written back to global once
    }
}
```

**Register pressure**: Each SM has 65,536 registers total. If a kernel uses 64 registers per thread × 1,024 threads per SM = 65,536 registers — the SM is fully used. If a kernel needs 128 registers per thread, the SM can only hold 512 threads → lower occupancy.

Check register usage: `nvcc --ptxas-options=-v my_kernel.cu`

### L.5.4 Constant Memory and Texture Memory

- **Constant memory** (`__constant__`): 64 KB, cached, broadcast to all threads in a warp for free when all threads read the same address. Used for model configuration, lookup tables.
- **Texture memory**: Cached 2D spatial locality. Rarely used directly in modern ML kernels (L2 cache usually handles this).

---

## L.6 Memory Coalescing — The Primary Throughput Rule

`[FOUNDATIONAL]` **Memory coalescing is the single most important optimization in CUDA.** A coalesced access lets 32 threads in a warp read from 32 consecutive memory locations in a single transaction. An uncoalesced access takes up to 32 separate transactions — 32× slower.

### L.6.1 Coalesced vs. Uncoalesced Access

```
  COALESCED — threads read consecutive addresses (one transaction)
  
  Thread:  T0   T1   T2   T3   T4   ...  T31
  Address: a+0  a+4  a+8  a+12 a+16 ... a+124
           ↑──────────────────────────────↑
           All 32 reads merged into one 128-byte HBM transaction ✓
  
  UNCOALESCED — threads read strided addresses (32 transactions)
  
  Thread:  T0     T1     T2     T3      ...  T31
  Address: a+0    a+128  a+256  a+384   ...  a+3968
           ↑      ↑      ↑      ↑            ↑
           32 separate cache lines → 32 transactions × slower ✗
```

### L.6.2 Row-Major vs. Column-Major Matrix Access

This is where coalescing bites most beginners. Consider a matrix `A[M][N]` stored row-major:

```
  Row-major layout in memory:
  A[0][0] A[0][1] A[0][2] ... A[0][N-1] | A[1][0] A[1][1] ...
  
  COALESCED: threads access the same row (A[row][threadIdx.x])
  Thread 0 → A[row][0], Thread 1 → A[row][1], ... consecutive ✓
  
  UNCOALESCED: threads access the same column (A[threadIdx.x][col])
  Thread 0 → A[0][col], Thread 1 → A[N+col], Thread 2 → A[2N+col]
  → stride N × sizeof(float) between consecutive threads ✗
```

`[COMMON TRAP]` Accessing `A[threadIdx.x][col]` in a naive matrix kernel causes strided (uncoalesced) reads. Always map `threadIdx.x` to the innermost (contiguous) dimension.

---

## L.7 Shared Memory Bank Conflicts

Shared memory is divided into 32 **banks** (on modern GPUs). Consecutive 4-byte words go to consecutive banks:

```
  Shared memory banks (32 banks, 4 bytes each)
  
  Address:  0   4   8  12  16  20  24  28  32  36  ...
  Bank:     0   1   2   3   4   5   6   7   8   9  ...
  (bank = (byte_address / 4) % 32)
```

If multiple threads in a warp access the **same bank** (but different addresses within it), the accesses are **serialized** — called a bank conflict. If all threads access the same address (broadcast), it is fine.

```
  NO CONFLICT — 32 threads access 32 different banks:
  Thread i reads smem[i]  → bank i  (each bank hit once) ✓
  
  2-WAY CONFLICT — two threads hit the same bank:
  Thread 0 reads smem[0]  → bank 0  ┐
  Thread 16 reads smem[16] → bank 16  → wait, N%32: smem[16] is bank 16 ✓
  (actually fine — but smem[0] and smem[32] both hit bank 0 → conflict)
  
  CLASSIC MATRIX TRANSPOSE CONFLICT:
  smem[threadIdx.y][threadIdx.x] — no conflict on read
  smem[threadIdx.x][threadIdx.y] — every thread hits bank threadIdx.y → 32-way conflict ✗
  FIX: smem[threadIdx.y][threadIdx.x + 1]  — pad by 1 column to shift banks ✓
```

`[COMMON TRAP]` A 32×32 `__shared__ float` tile transposed naively has 32-way bank conflicts. The fix is to declare `__shared__ float tile[32][33]` — the extra column shifts the banks.

---

## L.8 Occupancy — Keeping All SMs Busy

**Occupancy** is the ratio of active warps to the maximum possible warps on an SM. Higher occupancy hides memory latency by allowing the warp scheduler to switch to another warp while one is waiting for memory.

```
  SM warp scheduling — latency hiding
  
  Time →
  Warp A: ──[compute]──[MEM WAIT 600 cycles]────────────[compute]──
  Warp B:                ──[compute]──[MEM WAIT]──────────[compute]──
  Warp C:                              ──[compute]──[MEM WAIT]──────
  Warp D:                                           ──[compute]─────
                                                              ^
                                                  When warp A's memory arrives,
                                                  the SM switches back to warp A.
                                                  Latency is hidden if enough warps exist.
```

Occupancy is limited by:

1. **Registers per thread**: more registers → fewer threads per SM
2. **Shared memory per block**: more smem → fewer blocks per SM
3. **Block size**: must divide evenly into warp-sized groups

```
WORKED EXAMPLE L.2 — Occupancy Calculation for H100
─────────────────────────────────────────────────────────────────────
Given:   H100 SM limits: 65,536 registers, 228 KB shared memory, 1,024 threads/block
         Kernel uses: 32 registers/thread, 16 KB shared memory/block, block size 256
Step 1:  Register limit: 65,536 / 32 = 2,048 threads max from registers
Step 2:  Shared memory limit: 228 KB / 16 KB = 14 blocks max → 14 × 256 = 3,584 threads
Step 3:  Thread count limit: 1,024 threads/block (H100 max) → with 256 threads: 4 blocks = 1,024 threads
Step 4:  Binding constraint: min(2,048, 3,584, 1,024) = 1,024 threads
Step 5:  Max warps per SM on H100: 64
         Active warps: 1,024 / 32 = 32 warps
         Occupancy: 32 / 64 = 50%
─────────────────────────────────────────────────────────────────────
```

Check occupancy with `nvcc --ptxas-options=-v` or the CUDA Occupancy Calculator.

---

## L.9 Warp Divergence — When if-Statements Hurt

`[FOUNDATIONAL]` When threads in the same warp take different branches, the warp executes **both** paths sequentially with predication (inactive threads are masked out). This is called **warp divergence**.

```
  No divergence (all 32 threads take same path):
  if (val > 0.0f) { ... }    ← if all 32 vals > 0, single path ✓
  
  2-way divergence (half take each path):
  if (threadIdx.x % 2 == 0) { path_A; } else { path_B; }
  → path_A runs with threads 0,2,4...30 active (odd threads masked)
  → path_B runs with threads 1,3,5...31 active (even threads masked)
  → 2× the time ✗
  
  32-way divergence:
  switch (threadIdx.x % 32) { case 0: ...; case 1: ...; ... }
  → 32 sequential passes, 1 active thread each ← worst case ✗
```

`[COMMON TRAP]` In a softmax kernel, `if (i < seq_len)` causes divergence only in the last warp of a sequence. For typical sequence lengths divisible by 32, this is not a problem. But for attention masks (different tokens masked per row), divergence can be severe — which is why FlashAttention uses a carefully designed masking strategy.

---

## L.10 Synchronization

### L.10.1 `__syncthreads()` — Block Barrier

`__syncthreads()` is a barrier synchronization that waits until **all threads in the block** have reached that point. Required after writing shared memory before another thread reads it.

```cpp
__shared__ float smem[256];
smem[threadIdx.x] = d_in[gid];  // write
__syncthreads();                  // MUST wait before reads
float val = smem[(threadIdx.x + 1) % 256];  // safe to read now
```

`[COMMON TRAP]` Never put `__syncthreads()` inside a conditional branch where some threads may not reach it. **Every thread in the block must execute `__syncthreads()`** or you get a deadlock (hang).

```cpp
// BAD: only some threads sync
if (threadIdx.x < 128) {
    smem[threadIdx.x] = 0.0f;
    __syncthreads();  // threads 128-255 never reach this → DEADLOCK
}

// GOOD: all threads sync
smem[threadIdx.x] = (threadIdx.x < 128) ? 0.0f : 1.0f;
__syncthreads();  // all 256 threads reach this ✓
```

### L.10.2 Warp-Level Primitives

For operations within a warp, CUDA provides fast intrinsics that don't need shared memory:

```cpp
// Warp reduction using shuffle
__device__ float warp_reduce_sum(float val) {
    // XOR shuffle: exchange across powers of 2
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;  // thread 0 now holds the sum of all 32 threads
}

// Warp broadcast: thread 0 broadcasts to all
float broadcast_val = __shfl_sync(0xffffffff, val, 0);

// Warp prefix sum (exclusive scan)
// ... (more complex; see L.14 for full implementation)
```

`__shfl_xor_sync` is used in Flash Attention's online softmax to exchange partial sums within a warp without touching shared memory.

### L.10.3 Atomic Operations

When multiple threads must update the same memory location:

```cpp
// Increment a counter from many threads — no race condition
atomicAdd(&counter, 1);
atomicMax(&global_max, local_val);
atomicCAS(&lock, 0, 1);  // compare-and-swap: set to 1 if was 0
```

Atomic operations are serialized within a memory location — not fast, but necessary for histograms, counters, and lock-based data structures.

---

## L.11 CUDA Streams — Async Execution

By default, all CUDA operations on a device execute sequentially on the **default stream (stream 0)**. Streams allow overlapping computation and data transfer.

```
  Default stream (all operations sequential):
  
  Host:   [H2D copy]──[kernel A]──[kernel B]──[D2H copy]
  GPU:    wait──────[H2D]────[A]──────[B]────[D2H]──────
  
  With streams (overlap possible):
  
  Stream 1:  [H2D copy chunk 1]────────[kernel A chunk 1]──[D2H chunk 1]
  Stream 2:          [H2D copy chunk 2]────────[kernel A chunk 2]──[D2H chunk 2]
             Time →  ←─────── overlap ──────→
```

Creating and using streams:

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Async copy and kernel on stream1
cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream1);
my_kernel<<<grid, block, 0, stream1>>>(d_a, d_c, n);

// Overlap with stream2
cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream2);
my_kernel<<<grid, block, 0, stream2>>>(d_b, d_c2, n);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

vLLM uses CUDA streams to overlap prefill computation with KV cache transfers, and to pipeline multiple requests.

---

## L.12 Error Handling

`[COMMON TRAP]` Ignoring CUDA errors is the leading cause of silent incorrect results and mysterious crashes. Every CUDA API call returns an error code. Every kernel launch is followed by an asynchronous error that must be checked.

```cpp
// Macro for checking CUDA API calls
#define CUDA_CHECK(call)                                              \
    do {                                                             \
        cudaError_t err = (call);                                    \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// Usage
CUDA_CHECK(cudaMalloc(&d_a, bytes));
CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

// Check kernel launch errors (asynchronous — check after sync)
my_kernel<<<grid, block>>>(d_a, d_b, n);
CUDA_CHECK(cudaGetLastError());        // check launch params valid
CUDA_CHECK(cudaDeviceSynchronize());   // wait and check runtime errors
```

For device code (kernels), errors like out-of-bounds array access cause undefined behavior — there is no exception. Enable compute-sanitizer for debugging:

```bash
compute-sanitizer --tool memcheck ./my_binary
compute-sanitizer --tool racecheck ./my_binary  # detect race conditions
```

---

## L.13 The Roofline Model — Where Is Your Kernel Bottlenecked?

`[FOUNDATIONAL]` Every kernel is either **memory-bandwidth bound** or **compute bound**. The roofline model tells you which, and therefore what optimization to pursue.

```
  Roofline model for H100
  
  Performance
  (TFLOPS)  │
       ~4PF │                              ╔══════════════════╗
            │                          ╔══╝  Compute Roof    ║
            │                      ╔══╝     (FP16: ~4 PFLOPS)║
       2 PF │                  ╔══╝                          ║
            │              ╔══╝                              ║
       1 PF │          ╔══╝                                  ║
            │      ╔══╝ ← Memory Roof: BW × Arith. Intensity  ║
      500 T │  ╔══╝   (3.35 TB/s × AI)                       ║
            │══╝                                              ║
            └──────────────────────────────────────────────────
            0   10   20   50   100   200   500  1000
                   Arithmetic Intensity (FLOPS/byte)
  
  Ridge point for H100 FP16: 1.979e15 / 3.35e12 = ~591 FLOPS/byte
```

**Arithmetic Intensity (AI)** = total FLOPs executed / total bytes moved from HBM

```
WORKED EXAMPLE L.3 — Arithmetic Intensity of Key LLM Operations
─────────────────────────────────────────────────────────────────────
Vector addition (c = a + b, N=1M FP32 elements):
  FLOPs:  N additions = 1e6
  Bytes:  2N reads + N writes = 3 × 4MB = 12 MB
  AI:     1e6 / 12e6 = 0.083 FLOPS/byte → VERY memory-bound

Matrix multiplication (C = A×B, M=N=K=4096, FP16):
  FLOPs:  2 × M × N × K = 2 × 4096³ = 137e9
  Bytes:  A + B + C = 2×(4096²×2) + 4096²×2 = 100 MB
  AI:     137e9 / 100e6 = 1,370 FLOPS/byte → near compute roof

Attention decode (single token, batch=1, D=4096, seq=8192, BF16):
  FLOPs:  2 × seq × D = 2 × 8192 × 4096 = 67e6
  Bytes:  KV cache: 2 × seq × D × 2 bytes = 134 MB
  AI:     67e6 / 134e6 = 0.5 FLOPS/byte → VERY memory-bound
─────────────────────────────────────────────────────────────────────
```

This explains why:

- Decode is memory-bandwidth bound (not compute) — every token decode loads the entire KV cache
- Prefill is compute-bound — processes a long sequence in a large matrix multiply
- Flash Attention is designed to reduce HBM bytes moved (AI increases → moves toward compute roof)

---

## L.14 Kernel Deep-Dive: Parallel Reduction

Reduction is fundamental — it appears in softmax (sum, max), layer norm (mean, variance), and loss computation. A naive reduction is catastrophically slow.

### L.14.1 Naive Approach (Wrong)

```cpp
// BAD: sequential, uses only 1 thread
__global__ void reduce_naive(float* d_in, float* d_out, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += d_in[i];
    *d_out = sum;
}
// Uses 1 of 8192 available threads. 8191/8192 = ~0% GPU utilization.
```

### L.14.2 Parallel Tree Reduction

```
  Tree reduction: N=8 elements, 4 threads
  
  Step 0 (initial):   [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]
  Step 1 (stride=4):  [a0+a4] [a1+a5] [a2+a6] [a3+a7]  idle  idle  idle  idle
  Step 2 (stride=2):  [a0+a4+a2+a6] [a1+a5+a3+a7]  idle  idle  ...
  Step 3 (stride=1):  [a0+a1+...+a7]  idle  idle  ...
                         ^ result
  log2(8) = 3 steps vs. 7 sequential steps. For N=1024: 10 vs. 1023 steps.
```

```cpp
__global__ void reduce_sum(float* d_in, float* d_out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from global to shared
    smem[tid] = (gid < n) ? d_in[gid] : 0.0f;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    // Thread 0 writes block result to global
    if (tid == 0) atomicAdd(d_out, smem[0]);
}
```

### L.14.3 Warp Shuffle Reduction (Faster)

For the last 32 threads (one warp), avoid shared memory entirely:

```cpp
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, mask);
    return val;
}

__global__ void reduce_sum_v2(float* d_in, float* d_out, int n) {
    __shared__ float warp_results[8];  // 256/32 = 8 warps
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;

    float val = (gid < n) ? d_in[gid] : 0.0f;

    // Step 1: reduce within each warp using shuffles (no smem needed)
    val = warp_reduce_sum(val);

    // Step 2: lane 0 of each warp writes result to shared memory
    if (lane == 0) warp_results[warp_id] = val;
    __syncthreads();

    // Step 3: first warp reduces the 8 warp results
    if (warp_id == 0) {
        val = (lane < 8) ? warp_results[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) atomicAdd(d_out, val);
    }
}
```

This pattern (per-warp reduction → write to smem → final warp) is used in every production softmax kernel.

---

## L.15 Kernel Deep-Dive: Online Softmax

Softmax is the inner loop of every attention computation. The naive softmax requires three passes over the data:

```
  Naive softmax:
  Pass 1: compute max(x_i) for numerical stability
  Pass 2: compute sum(exp(x_i - max))
  Pass 3: divide each exp(x_i - max) by sum

  Problem: three global memory passes = 3× memory bandwidth cost
```

**Online softmax** (used in Flash Attention) computes max and sum in a single pass using a mathematical identity:

```
WORKED EXAMPLE L.4 — Online Softmax Derivation
─────────────────────────────────────────────────────────────────────
Goal: compute softmax without knowing the max ahead of time.

When we see element x_new after having seen x_0..x_{k-1}:
  Old state: m_old = max(x_0..x_{k-1}), d_old = sum(exp(x_i - m_old))
  New max:   m_new = max(m_old, x_new)
  
  The sum must be re-scaled because the max changed:
  d_new = d_old × exp(m_old - m_new) + exp(x_new - m_new)
          ↑                                 ↑
       old sum scaled to new max       new element
  
  At the end: softmax(x_i) = exp(x_i - m_final) / d_final
─────────────────────────────────────────────────────────────────────
```

```cpp
__global__ void online_softmax(const float* __restrict__ d_in,
                                float*       __restrict__ d_out,
                                int n) {
    // Each block handles one row; threads cooperate on one row
    __shared__ float smem_m[32];  // per-warp max
    __shared__ float smem_d[32];  // per-warp denominator

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    // Step 1: Each thread computes local (m, d) over its elements
    float m = -INFINITY, d = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        float x = d_in[row * n + j];
        float m_new = fmaxf(m, x);
        d = d * expf(m - m_new) + expf(x - m_new);
        m = m_new;
    }

    // Step 2: Reduce (m, d) within each warp using shuffles
    for (int mask = 16; mask > 0; mask >>= 1) {
        float m2 = __shfl_xor_sync(0xffffffffu, m, mask);
        float d2 = __shfl_xor_sync(0xffffffffu, d, mask);
        float m_new = fmaxf(m, m2);
        d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }

    // Step 3: Lane 0 writes per-warp (m, d) to shared memory
    if (lane == 0) { smem_m[warp] = m; smem_d[warp] = d; }
    __syncthreads();

    // Step 4: First warp reduces across all warps
    int n_warps = blockDim.x / 32;
    if (warp == 0) {
        m = (lane < n_warps) ? smem_m[lane] : -INFINITY;
        d = (lane < n_warps) ? smem_d[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            float m2 = __shfl_xor_sync(0xffffffffu, m, mask);
            float d2 = __shfl_xor_sync(0xffffffffu, d, mask);
            float m_new = fmaxf(m, m2);
            d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
            m = m_new;
        }
        if (lane == 0) { smem_m[0] = m; smem_d[0] = d; }
    }
    __syncthreads();

    // Step 5: All threads write final softmax values
    float m_final = smem_m[0];
    float d_final = smem_d[0];
    for (int j = tid; j < n; j += blockDim.x)
        d_out[row * n + j] = expf(d_in[row * n + j] - m_final) / d_final;
}
```

This is structurally identical to the tiled softmax in Flash Attention — one global read, one global write, online statistics computation.

---

## L.16 Kernel Deep-Dive: Matrix-Vector Product (GEMV)

During the decode phase of LLM inference, the dominant operation is **GEMV** (General Matrix-Vector multiply): multiplying the weight matrix by a single token's embedding. This is the most memory-bandwidth-bound operation in inference.

```
  GEMV: y = W × x
  W: [M × K] (weight matrix, loaded from HBM)
  x: [K × 1] (one token, fits in L2/registers)
  y: [M × 1] (output)
  
  FLOPs: 2 × M × K
  Bytes: M × K × dtype (load W) + K (load x, often cached) + M (write y)
  
  For 70B model, one linear layer (e.g. K=8192, M=8192, BF16):
  FLOPs:  2 × 8192 × 8192 = 134e6
  Bytes:  8192 × 8192 × 2 = 134 MB (just to load W)
  AI:     134e6 / 134e6 = 1.0 FLOPS/byte → memory bound
```

```cpp
// Tiled GEMV: each block computes one row of y
// Block size: 256 threads; each thread computes a partial dot product
__global__ void gemv_fp16(const __half* __restrict__ W,   // [M x K]
                           const __half* __restrict__ x,   // [K]
                           float*        __restrict__ y,   // [M]
                           int M, int K) {
    __shared__ float partial[256];
    int row = blockIdx.x;          // this block computes y[row]
    int tid = threadIdx.x;

    // Each thread accumulates partial dot product over K/256 elements
    float acc = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        acc += __half2float(W[row * K + k]) * __half2float(x[k]);
    }
    partial[tid] = acc;
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[row] = partial[0];
}
// Launch: gemv_fp16<<<M, 256>>>(d_W, d_x, d_y, M, K)
```

Production GEMV kernels (in cuBLAS, cutlass, or vLLM's custom kernels) use:

- **Vectorized loads** (`float4`, `int4`) to maximize memory bus utilization
- **Double buffering** with shared memory for prefetch
- **Tensor core WMMA** for INT8/FP8 quantized variants

---

## L.17 Kernel Deep-Dive: INT8 Quantized GEMV

The INT8 version of GEMV is what vLLM uses in W8A8 quantization and what llama.cpp uses in Q8_0. The key challenge: INT8 arithmetic, FP32 accumulate.

```cpp
__global__ void gemv_int8(const int8_t* __restrict__ W,     // [M x K] INT8
                           const int8_t* __restrict__ x,     // [K] INT8
                           float*        __restrict__ y,     // [M] FP32
                           const float*  __restrict__ scale_W, // per-row scales
                           float scale_x,                    // input scale
                           int M, int K) {
    __shared__ int32_t partial[256];  // accumulate in INT32 to avoid overflow
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int32_t acc = 0;
    for (int k = tid; k < K; k += blockDim.x) {
        acc += (int32_t)W[row * K + k] * (int32_t)x[k];
    }
    partial[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    // Dequantize: multiply by scales to get FP32 result
    if (tid == 0) y[row] = (float)partial[0] * scale_W[row] * scale_x;
}
```

`[DEEP DIVE]` The INT32 accumulator is critical. If you accumulate in INT8, you overflow after 127 additions. INT32 handles up to 2^31 / (127 × 127) ≈ 133,000 terms — more than enough for K=8192.

---

## L.18 Tensor Cores — Hardware Matrix Multiply

`[DEEP DIVE]` Tensor Cores are specialized hardware units in each SM that compute small matrix multiplies extremely fast:

| Generation | GPU | Operation | Throughput per SM |
|---|---|---|---|
| 1st gen | V100 | 4×4×4 FP16 | 125 TFLOPS per SM |
| 3rd gen | A100 | 16×16×16 FP16/BF16/INT8 | 312 TFLOPS per SM |
| 4th gen | H100 | 16×16×16 FP8 | 989 TFLOPS per SM |
| 5th gen | B200 | FP4 native | ~2× H100 |

Using Tensor Cores requires the WMMA API (Warp Matrix Multiply-Accumulate) or, more commonly, using cuBLAS/cuBLASLt/CUTLASS which handles the details:

```cpp
// Using cuBLAS for matrix multiplication (recommended for production)
cublasHandle_t handle;
cublasCreate(&handle);

// SGEMM: C = alpha * A * B + beta * C
// (cuBLAS uses column-major; adjust leading dimensions accordingly)
const float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,           // output (N), input (M), shared (K)
            &alpha,
            d_B, N,            // B: [K × N] (column-major)
            d_A, K,            // A: [M × K] (column-major)
            &beta,
            d_C, N);           // C: [M × N]
```

For FP8/INT8 with quantization, use `cublasLtMatmul` with the appropriate compute type.

---

## L.19 Profiling with Nsight Compute

The only way to know if your optimization worked is to measure it. Nsight Compute is NVIDIA's kernel profiler.

### L.19.1 Basic Profiling

```bash
# Profile a specific kernel
ncu --kernel-name "my_kernel" ./my_binary

# Collect full roofline data
ncu --set full --kernel-name "my_kernel" ./my_binary > profile.txt

# Interactive GUI (opens Nsight Compute UI)
ncu --export profile.ncu-rep ./my_binary
ncu-ui profile.ncu-rep
```

### L.19.2 Key Metrics to Read

| Metric | What it means | Good value |
|---|---|---|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L2 transactions for loads | Minimize |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Occupancy | > 50% |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | Cache hit rate | > 80% |
| `sm__sass_thread_inst_executed_op_fadd_pred_on.sum` | FP32 FMAs executed | Near theoretical |
| `gpu__time_duration.sum` | Kernel duration | Minimize |

### L.19.3 Reading the Roofline in Nsight

```
  Nsight Compute → Speed of Light → Roofline Chart
  
  If your kernel appears:
  ┌──────────────────────────────────────────────────────────┐
  │  •  ← here (far left, low AI)     → memory-bandwidth    │
  │        optimize for coalescing, reduce bytes accessed    │
  │                                                          │
  │            •  ← here (right, below compute roof)         │
  │               → compute-bound but below roof             │
  │               → look for instruction-level parallelism,  │
  │                  unrolling, or Tensor Core usage          │
  │                                                          │
  │                     •  ← touching compute roof ✓         │
  └──────────────────────────────────────────────────────────┘
```

---

## L.20 Common CUDA Mistakes in LLM Inference Code

### Mistake 1: Not checking CUDA errors

Every CUDA API call can fail silently if you don't check. Use the `CUDA_CHECK` macro from §L.11.

### Mistake 2: Forgetting `__syncthreads()` after shared memory writes

Reads of shared memory before `__syncthreads()` see undefined data from other threads. This produces wrong outputs that are not reproducible across runs.

### Mistake 3: Launching with too few threads for the problem size

```cpp
// Bug: launches 128 threads for 1M elements — only first 128 are processed
my_kernel<<<1, 128>>>(d_data, n);

// Correct:
int grid = (n + 255) / 256;
my_kernel<<<grid, 256>>>(d_data, n);
```

### Mistake 4: Uncoalesced access in the wrong dimension

The matrix transpose access pattern (`A[threadIdx.x][col]` instead of `A[row][threadIdx.x]`) is the most common cause of 10-20× slower-than-expected kernels.

### Mistake 5: Using `cudaDeviceSynchronize()` everywhere in production

`cudaDeviceSynchronize()` blocks the CPU until all GPU work finishes. Necessary for correctness checks and profiling, but catastrophic for throughput in production. Use per-stream synchronization or CUDA events.

### Mistake 6: Incorrect grid/block for 2D problems

```cpp
// Processing a 2D image or attention matrix: W columns × H rows
dim3 block(16, 16, 1);   // 256 threads in a 16×16 tile
dim3 grid((W + 15) / 16, (H + 15) / 16, 1);

// Inside kernel:
int col = blockIdx.x * blockDim.x + threadIdx.x;  // maps to W
int row = blockIdx.y * blockDim.y + threadIdx.y;  // maps to H
if (col < W && row < H) { ... }
```

### Mistake 7: Using `float` for intermediate KV cache values when BF16 is expected

vLLM's attention kernel stores KV cache in BF16 by default. Accidentally loading as `float` reads garbage because the byte layout differs. Always use the correct CUDA data type (`__nv_bfloat16`, `__half`, `__nv_fp8_e4m3`).

### Mistake 8: Race condition without atomics

```cpp
// BAD: race condition — multiple threads write the same output element
d_out[blockIdx.x] += smem[threadIdx.x];  // non-atomic read-modify-write

// GOOD:
atomicAdd(&d_out[blockIdx.x], smem[threadIdx.x]);
// OR: reduce to one result in shared memory first, then one thread writes
```

### Mistake 9: Excessive dynamic shared memory allocation

Requesting 228 KB shared memory per block on H100 means only 1 block per SM (max). For most kernels, 16–64 KB is the sweet spot between holding useful data and allowing enough concurrent blocks for occupancy.

### Mistake 10: Ignoring alignment for vectorized loads

`float4` loads (128-bit) require 16-byte alignment. Misaligned `float4` on modern GPUs generates two transactions instead of one, halving the effective bandwidth.

```cpp
// Check alignment before using float4 loads
assert(((uintptr_t)d_ptr % 16) == 0);

// Load 4 floats at once
float4 val = *reinterpret_cast<const float4*>(d_ptr + i);
```

---

## L.21 Putting It Together: A Minimal Attention Kernel

To solidify all concepts, here is a minimal scaled dot-product attention kernel for a single head, decode step (batch=1, 1 query token, N key/value tokens):

```cpp
// Minimal single-head attention for one decode step
// Query q: [1 x D], Keys K: [N x D], Values V: [N x D]
// Output:  [1 x D]
//
// Each block handles attention for one head.
// blockDim.x = min(N, 256) — threads process sequence positions.

__global__ void decode_attention(
        const float* __restrict__ q,   // [D]
        const float* __restrict__ K,   // [N x D]
        const float* __restrict__ V,   // [N x D]
        float*       __restrict__ out, // [D]
        int N, int D, float scale)     // scale = 1/sqrt(D)
{
    extern __shared__ float smem[];  // [N] for scores
    float* scores = smem;

    int tid = threadIdx.x;

    // Step 1: Compute attention scores  s[i] = scale * dot(q, K[i])
    for (int i = tid; i < N; i += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d)
            dot += q[d] * K[i * D + d];
        scores[i] = dot * scale;
    }
    __syncthreads();

    // Step 2: Online softmax over scores
    // (simplified: assumes one warp handles all of N for small N)
    float m = -INFINITY, denom = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float s = scores[i];
        float m_new = fmaxf(m, s);
        denom = denom * expf(m - m_new) + expf(s - m_new);
        m = m_new;
    }
    // Warp reduce m and denom
    for (int mask = 16; mask > 0; mask >>= 1) {
        float m2 = __shfl_xor_sync(0xffffffffu, m, mask);
        float d2 = __shfl_xor_sync(0xffffffffu, denom, mask);
        float m_new = fmaxf(m, m2);
        denom = denom * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }
    // Broadcast final (m, denom) to all threads in block
    __shared__ float final_m, final_denom;
    if (tid == 0) { final_m = m; final_denom = denom; }
    __syncthreads();
    m = final_m; denom = final_denom;

    // Step 3: Compute weighted sum over V
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < N; ++i)
            acc += expf(scores[i] - m) / denom * V[i * D + d];
        out[d] = acc;
    }
}
// Launch: int smem = N * sizeof(float);
//         decode_attention<<<1, 256, smem>>>(q, K, V, out, N, D, 1.0f/sqrtf(D));
```

This is not Flash Attention (it stores all scores in shared memory simultaneously, which fails for long sequences). It is the pedagogically clear version that shows every step. Flash Attention replaces the all-at-once score computation with tiled tiles that reuse shared memory.

---

## L.22 Building and Running CUDA Code

### L.22.1 Minimal Build

```bash
# Compile single file
nvcc -O2 -std=c++17 -arch=sm_90 -o my_binary my_kernel.cu
#                               ^^^^^^^^^^ target architecture (H100=sm_90)

# With debug info (for compute-sanitizer)
nvcc -G -g -std=c++17 -arch=sm_90 -o my_binary_debug my_kernel.cu

# Check PTX (intermediate GPU assembly)
nvcc -ptx -arch=sm_90 my_kernel.cu -o my_kernel.ptx
```

### L.22.2 Architecture Flags

| GPU | Architecture | `-arch` flag |
|---|---|---|
| Tesla V100 | Volta sm_70 | `-arch=sm_70` |
| A100 | Ampere sm_80 | `-arch=sm_80` |
| H100 | Hopper sm_90 | `-arch=sm_90` |
| H200 | Hopper sm_90 | `-arch=sm_90` |
| B200 | Blackwell sm_100 | `-arch=sm_100` |
| RTX 4090 | Ada sm_89 | `-arch=sm_89` |

For a binary that runs on multiple GPUs: `nvcc -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 ...`

### L.22.3 CMake Integration (as used in vLLM / llama.cpp)

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_inference_kernel CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

add_executable(my_binary main.cu kernels.cu)
target_compile_options(my_binary PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 -arch=sm_90 --use_fast_math>)
target_link_libraries(my_binary CUDA::cudart CUDA::cublas)
```

---

## L.23 Chapter Summary

| Concept | Key number / rule |
|---|---|
| Warp size | 32 threads — the atomic unit of execution |
| H100 SMs | 132 SMs, 128 CUDA cores each, 65,536 registers per SM |
| Shared memory | Up to 228 KB per SM on H100; ~20 cycle latency |
| HBM bandwidth | 3.35 TB/s (H100); this is the bottleneck for decode |
| Register limit | 65,536 per SM; more registers/thread = fewer concurrent threads |
| Coalescing rule | Map `threadIdx.x` to the contiguous (innermost) dimension |
| Bank conflict rule | `smem[threadIdx.x]` is fine; `smem[threadIdx.x * 32]` is a 32-way conflict |
| Roofline ridge (H100 FP16) | ~591 FLOPS/byte — decode attention AI ≈ 1 → memory bound |
| Warp divergence cost | 2-way divergence = 2× slower; avoid inside performance-critical loops |
| `__syncthreads()` | Required after every shared memory write; never inside a conditional |

### Self-Check Questions

1. A kernel uses 64 registers per thread and has a block size of 256 threads. How many threads can fit on one H100 SM, and what is the occupancy?
2. You profile a kernel and find it achieves 80 GB/s memory bandwidth on an H100 (peak: 3350 GB/s). What is the utilization percentage, and what is the most likely cause of the gap?
3. Write the thread indexing code for a 2D kernel that processes a matrix of shape `[M, N]` with block dimensions `(16, 16)`.
4. Why does the online softmax algorithm avoid a second pass over the input data? What mathematical identity makes it possible?
5. A GEMV kernel for a 70B model's 8192×8192 BF16 weight matrix achieves 2.0 TFLOPS on an H100. The H100 FP16 compute peak is 1,979 TFLOPS. Is this kernel compute-bound or memory-bound? What is its arithmetic intensity?

### Answers

1. 65,536 / 64 = 1,024 threads from registers; 1,024 / 32 = 32 active warps; H100 max = 64 warps/SM; occupancy = 32/64 = 50%.
2. 80/3350 = 2.4% — almost certainly due to uncoalesced memory access (strided reads).
3. `int col = blockIdx.x * 16 + threadIdx.x; int row = blockIdx.y * 16 + threadIdx.y; if (col < N && row < M) { ... }`
4. Online softmax uses the identity that the running sum can be re-scaled when the maximum increases: `d_new = d_old × exp(m_old - m_new) + exp(x_new - m_new)`. This requires only a running (max, sum) state.
5. FLOPs: 2 × 8192² = 134M. Bytes: 8192² × 2 = 134MB. AI = 134M/134M = 1 FLOP/byte. Ridge point is ~591 FLOPS/byte (1,979 TFLOPS ÷ 3.35 TB/s), so AI=1 is far below the ridge → **memory-bound**. At 3.35 TB/s peak bandwidth, roofline predicts: 3.35e12 × 1 FLOP/byte = 3.35 TFLOPS. Achieving 2.0 TFLOPS = 2.0/3.35 = 60% of the BW roof — reasonable for a straightforward GEMV.

---

## L.24 Kernel Deep-Dive: GEMM — Naive Matrix Multiplication

General Matrix-Matrix Multiplication (GEMM) is the single most important kernel in deep learning. Every linear layer, every attention score computation, and every KV projection is a GEMM. Understanding its performance characteristics is essential.

**Problem:** compute $C = A \times B$ where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, $C \in \mathbb{R}^{M \times N}$.

### L.24.1 Naive GEMM — One Thread Per Output Element

The simplest mapping: one CUDA thread computes one element of $C$.

```cpp
// naive_gemm.cu — one thread per output element
// C[row][col] = sum over k of A[row][k] * B[k][col]

__global__ void gemm_naive(
    const float* __restrict__ A,   // [M × K], row-major
    const float* __restrict__ B,   // [K × N], row-major
    float*       __restrict__ C,   // [M × N], row-major
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];

    C[row * N + col] = acc;
}

// Launch: 16×16 thread blocks
void launch_gemm_naive(float* A, float* B, float* C, int M, int K, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_naive<<<grid, block>>>(A, B, C, M, K, N);
}
```

**Performance of naive GEMM:**

```
  Each output element C[row][col]:
    Reads K elements from row of A  → K memory accesses
    Reads K elements from col of B  → K memory accesses
    Performs K multiplications + K additions = 2K FLOPs

  Arithmetic intensity = 2K FLOPs / (2K × 4 bytes) = 0.25 FLOP/byte

  H100 roofline at AI=0.25:
    Compute roof: 989 TFLOPS (FP32)
    Memory roof:  3.35 TB/s × 0.25 = 0.84 TFLOPS
    → Naive GEMM is DEEPLY memory-bound

  Measured naive throughput: ~30–50 GFLOPS on H100
  cuBLAS SGEMM throughput:   ~750 TFLOPS
  Gap: ~15–25× — all from memory access pattern
```

The problem: thread reading column $j$ of B reads `B[0*N+j], B[1*N+j], B[2*N+j]...` — a strided access with stride $N$. No memory coalescing, each access is a separate cache line.

### L.24.2 Tiled GEMM with Shared Memory

The key insight: threads in the same block collectively compute a $T \times T$ tile of $C$. Load a $T \times T$ tile of $A$ and $B$ into shared memory together, then compute — each global memory load is reused $T$ times.

```cpp
// tiled_gemm.cu — shared memory tiling
#define TILE 16

__global__ void gemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    // Step through K in TILE-wide chunks
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Collaboratively load tile of A into shared memory
        int a_col = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // Collaboratively load tile of B into shared memory
        int b_row = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();   // ← wait for all threads to finish loading

        // Compute partial dot product from shared memory
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();   // ← wait before loading next tile
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}
```

**Why tiling works:**

```
  Without tiling (naive):
    Each thread reads K elements from A and K from B.
    No reuse — K reads per thread.

  With TILE=16 tiling:
    Each tile of A is loaded ONCE and reused by all 16 threads
    in the same row of the block.
    Each tile of B is loaded ONCE and reused by all 16 threads
    in the same column of the block.

    Memory traffic: K/TILE reads per thread (vs. K naive)
    Arithmetic intensity: 0.25 × TILE = 4 FLOP/byte at TILE=16
    With TILE=32: AI = 8 FLOP/byte

  Shared memory access time: ~1 cycle (vs ~200 cycles HBM)
  Tiled GEMM speedup over naive: 10–20×
  Still below cuBLAS: cuBLAS uses TILE=64–128 + vectorized loads
```

**The two `__syncthreads()` calls are non-negotiable:**
1. After loading: ensures all threads have finished writing to `As` and `Bs` before anyone reads
2. After computing: ensures all threads have finished reading before the next tile overwrites shared memory

### L.24.3 BF16 Tiled GEMM

LLM inference uses BF16 or FP16. The structure is identical but uses half-precision loads and Tensor Core intrinsics:

```cpp
// BF16 tiled GEMM fragment — uses wmma (Warp Matrix Multiply Accumulate)
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void gemm_bf16_wmma(
    const __nv_bfloat16* A, const __nv_bfloat16* B,
    float* C, int M, int K, int N)
{
    // Each warp computes a 16×16×16 WMMA fragment
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN =  blockIdx.x * blockDim.x + threadIdx.x;

    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(a_frag, A + warpM * 16 * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warpN * 16, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);  // Tensor Core!
    }

    store_matrix_sync(C + warpM * 16 * N + warpN * 16,
                      c_frag, N, mem_row_major);
}
// This is the kernel pattern that cuBLAS and CUTLASS are built on.
```

---

## L.25 Kernel Deep-Dive: Parallel Prefix Scan

A **prefix scan** (or prefix sum) computes the cumulative sum of an array. Given input `[a₀, a₁, a₂, a₃]`, the exclusive prefix scan outputs `[0, a₀, a₀+a₁, a₀+a₁+a₂]`.

Prefix scans appear in LLM inference for: KV cache offset computation, token packing for variable-length batches, sampling via CDF, and beam search index tracking.

### L.25.1 Sequential Baseline

```cpp
// Sequential: O(N) work, O(N) depth — not parallelisable
void scan_sequential(float* in, float* out, int N) {
    out[0] = 0;
    for (int i = 1; i < N; ++i)
        out[i] = out[i-1] + in[i-1];
}
```

### L.25.2 Work-Efficient Parallel Scan (Blelloch)

The **Blelloch algorithm** uses two passes — up-sweep (reduce) and down-sweep — to compute an exclusive prefix scan in $O(\log N)$ depth with $O(N)$ total work:

```cpp
// work_efficient_scan.cu
// Computes EXCLUSIVE prefix scan in shared memory (one block)
// For large arrays: use a multi-block scan with a separate
// block-level reduction pass.

__global__ void scan_exclusive(float* data, int N) {
    extern __shared__ float temp[];   // dynamic shared memory
    int tid = threadIdx.x;

    // Load input into shared memory
    temp[2*tid]   = (2*tid   < N) ? data[2*tid]   : 0;
    temp[2*tid+1] = (2*tid+1 < N) ? data[2*tid+1] : 0;

    // ── UP-SWEEP (reduce phase) ──────────────────────────────────
    // Build partial sums in place up the tree
    for (int stride = 1; stride < N; stride <<= 1) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < N)
            temp[idx] += temp[idx - stride];
    }

    // Clear the last element (makes it exclusive)
    if (tid == 0) temp[N-1] = 0;

    // ── DOWN-SWEEP phase ─────────────────────────────────────────
    // Traverse back down the tree building the scan output
    for (int stride = N/2; stride >= 1; stride >>= 1) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < N) {
            float left      = temp[idx - stride];
            temp[idx-stride] = temp[idx];        // swap
            temp[idx]       += left;             // combine
        }
    }

    __syncthreads();

    // Write back
    if (2*tid   < N) data[2*tid]   = temp[2*tid];
    if (2*tid+1 < N) data[2*tid+1] = temp[2*tid+1];
}

// Launch: N/2 threads, N floats of shared memory
// scan_exclusive<<<1, N/2, N*sizeof(float)>>>(data, N);
```

**Performance:**

```
  Sequential:   O(N) work,  O(N) depth
  Blelloch:     O(N) work,  O(log N) depth — 2× work vs. sequential
  Warp-level:   Use __shfl_up_sync for intra-warp scans in 5 steps

  For N=1024:
    Sequential: 1023 additions, depth 1023
    Blelloch:   2046 additions, depth 10 (log₂ 1024)
    Speedup in parallel:  1023/10 ≈ 100× fewer serial steps
```

### L.25.3 Warp-Level Scan (Fast, No Shared Memory)

```cpp
// Intra-warp exclusive scan using shuffle — no __syncthreads needed
__device__ float warp_scan_exclusive(float val) {
    // Inclusive scan first
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x >= offset) val += n;
    }
    // Convert inclusive → exclusive by shifting
    float excl = __shfl_up_sync(0xffffffff, val, 1);
    return (threadIdx.x == 0) ? 0.0f : excl;
}
// No global memory or shared memory needed. All registers.
// 5 shuffle rounds for 32 threads — extremely fast.
```

---

## L.26 Kernel Deep-Dive: 1D Convolution with Halo

Convolution appears in LLM inference for position encoding variants, sliding-window attention masking, and 1D feature extraction in some multimodal encoders.

The key challenge: a thread computing output element $i$ needs input elements $[i - r, \ldots, i + r]$ where $r$ is the kernel radius. Threads near block boundaries need elements from neighboring blocks — the **halo region**.

### L.26.1 Naive 1D Convolution

```cpp
// naive_conv1d.cu
__global__ void conv1d_naive(
    const float* __restrict__ input,   // [N]
    const float* __restrict__ kernel,  // [2R+1], centerd at R
    float*       __restrict__ output,  // [N]
    int N, int R)   // R = kernel radius
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float acc = 0.0f;
    for (int r = -R; r <= R; ++r) {
        int j = i + r;
        if (j >= 0 && j < N)        // boundary check
            acc += input[j] * kernel[r + R];
    }
    output[i] = acc;
}
// Problem: each thread re-reads neighboring elements independently.
// 2R+1 global memory reads per thread — most are not coalesced.
```

### L.26.2 Shared Memory Convolution with Halo Loading

```cpp
// conv1d_shared.cu — shared memory with halo
#define BLOCK 256
#define MAX_RADIUS 16

__global__ void conv1d_shared(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float*       __restrict__ output,
    int N, int R)
{
    // Shared memory holds the block + halo on both sides
    // Total tile size: BLOCK + 2*R elements
    extern __shared__ float tile[];   // size = (BLOCK + 2*R) * sizeof(float)

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * BLOCK + tid;   // global input index
    int halo = R;                           // halo width

    // ── Load main block elements ──────────────────────────────────
    tile[tid + halo] = (gid < N) ? input[gid] : 0.0f;

    // ── Load LEFT halo (first R threads load halo elements) ───────
    if (tid < halo) {
        int left_idx = gid - halo;
        tile[tid] = (left_idx >= 0) ? input[left_idx] : 0.0f;
    }

    // ── Load RIGHT halo (last R threads load halo elements) ───────
    if (tid >= BLOCK - halo) {
        int right_idx = gid + halo;
        tile[tid + 2*halo] = (right_idx < N) ? input[right_idx] : 0.0f;
    }

    __syncthreads();   // all shared memory ready

    // ── Compute convolution from shared memory ────────────────────
    if (gid < N) {
        float acc = 0.0f;
        for (int r = -R; r <= R; ++r)
            acc += tile[tid + halo + r] * kernel[r + R];
        output[gid] = acc;
    }
}

// Launch:
// int shared_bytes = (BLOCK + 2*R) * sizeof(float);
// conv1d_shared<<<(N+BLOCK-1)/BLOCK, BLOCK, shared_bytes>>>(
//     input, kernel, output, N, R);
```

**Memory access analysis:**

```
  Naive:  (2R+1) global reads per thread, many uncoalesced
          For R=8: 17 global reads per element

  Shared: (BLOCK + 2R) global reads per BLOCK of BLOCK threads
          Each element read exactly ONCE into shared memory
          Then reused 2R+1 times from shared memory

          Reduction in global reads: (2R+1)×BLOCK / (BLOCK+2R)
          For BLOCK=256, R=8: 17×256 / 272 = 16×   fewer global reads

  Shared memory bandwidth at 1 cycle vs. HBM at ~200 cycles:
  Effective speedup for conv with R=8: ~8–12×
```

### L.26.3 Convolution with Constant Memory Kernel

For small kernels (R ≤ 16) that are fixed across all threads, use **constant memory** — broadcast cached, zero bank conflicts:

```cpp
__constant__ float c_kernel[2 * MAX_RADIUS + 1];   // in constant cache

// Copy kernel to constant memory before launch:
// cudaMemcpyToSymbol(c_kernel, h_kernel, (2*R+1)*sizeof(float));

__global__ void conv1d_const(const float* input, float* output, int N, int R) {
    // ... same halo load as above ...
    float acc = 0.0f;
    for (int r = -R; r <= R; ++r)
        acc += tile[tid + halo + r] * c_kernel[r + R];  // ← constant cache
    output[gid] = acc;
}
// Constant cache is broadcast — all 32 warp threads get the same
// kernel value in a single transaction. No shared memory slot needed
// for the kernel weights.
```

---

## L.27 Kernel Deep-Dive: 2D Reduction and Stencil

### L.27.1 Column Reduction (Softmax Denominator Pattern)

Computing the softmax denominator requires reducing along one dimension of a 2D matrix — a column reduction (one output per column). This pattern is the inner loop of every online softmax kernel.

```cpp
// col_reduce.cu — reduce each column, one block per column
__global__ void reduce_columns(
    const float* __restrict__ A,  // [M × N] row-major
    float*       __restrict__ out, // [N]
    int M, int N)
{
    int col = blockIdx.x;           // one block handles one column
    if (col >= N) return;

    // Each thread accumulates a partial sum over rows
    float partial = 0.0f;
    for (int row = threadIdx.x; row < M; row += blockDim.x)
        partial += A[row * N + col];

    // Warp-level reduction
    for (int mask = 16; mask >= 1; mask >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, mask);

    // First thread of each warp writes to shared memory
    __shared__ float warp_sums[32];
    if (threadIdx.x % 32 == 0)
        warp_sums[threadIdx.x / 32] = partial;
    __syncthreads();

    // Final reduction across warp sums (first warp only)
    if (threadIdx.x < blockDim.x / 32) {
        partial = warp_sums[threadIdx.x];
        for (int mask = 16; mask >= 1; mask >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, mask);
        if (threadIdx.x == 0) out[col] = partial;
    }
}
```

### L.27.2 2D Stencil — Tiled with Halo

A 2D stencil computes each output pixel as a weighted sum of its neighbors (the 2D generalization of convolution). Used in image preprocessing for multimodal models.

```cpp
// stencil_2d.cu — 5-point stencil with shared memory
#define TILE_W 32
#define TILE_H 8
#define RADIUS  1    // 5-point: center + 4 neighbors

__global__ void stencil_2d_5pt(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int H, int W,
    float w_center, float w_neighbor)
{
    // Shared tile includes halo: (TILE_H+2R) × (TILE_W+2R)
    __shared__ float s[TILE_H + 2*RADIUS][TILE_W + 2*RADIUS];

    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * TILE_W + tx;
    int gy = blockIdx.y * TILE_H + ty;

    // Load interior + halo into shared memory
    auto clamp = [](int v, int lo, int hi){return max(lo, min(v, hi));};

    for (int dy = -RADIUS; dy <= RADIUS; dy += TILE_H) {
        for (int dx = -RADIUS; dx <= RADIUS; dx += TILE_W) {
            int sx = clamp(gx + dx, 0, W-1);
            int sy = clamp(gy + dy, 0, H-1);
            s[ty+RADIUS+dy][tx+RADIUS+dx] = in[sy*W + sx];
        }
    }
    __syncthreads();

    if (gx < W && gy < H) {
        float val = w_center * s[ty+RADIUS][tx+RADIUS]
                  + w_neighbor * (s[ty-1+RADIUS][tx+RADIUS]
                                + s[ty+1+RADIUS][tx+RADIUS]
                                + s[ty+RADIUS][tx-1+RADIUS]
                                + s[ty+RADIUS][tx+1+RADIUS]);
        out[gy*W + gx] = val;
    }
}
```

---

## L.28 Pattern Summary — When to Use What

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Pattern           When to use                Shared mem?      │
  ├─────────────────────────────────────────────────────────────────┤
  │  Naive GEMM        Learning / small N (<256)   No              │
  │  Tiled GEMM        Any real matmul             Yes (TILE²×2)   │
  │  WMMA/Tensor Core  BF16/FP16 production        Yes + fragments │
  │  Parallel reduce   Sum/max along dimension     Yes (2×block)   │
  │  Warp scan         Small in-warp prefix sum    No (registers)  │
  │  Blelloch scan     Block-level prefix sum      Yes (N elems)   │
  │  Conv1d naive      Prototype / verify          No              │
  │  Conv1d + halo     Production 1D conv          Yes (B+2R)      │
  │  2D stencil        Image processing, PatchEmb  Yes (tile+halo) │
  │  Column reduction  Softmax denom, attention     Yes (warps)    │
  └─────────────────────────────────────────────────────────────────┘

  Memory hierarchy reminder:
    Registers:     <1 cycle  —  per-thread, fastest
    Shared mem:    ~1 cycle  —  per-block, programr-controlled
    L2 cache:    ~30 cycles  —  automatic
    HBM (DRAM): ~200 cycles  —  large, slow; minimize with tiling
```

---

### Where to Go Next

- **CUDA Programming Guide** (developer.nvidia.com/cuda-programming-guide) — the authoritative reference
- **Programming Massively Parallel Processors** (Kirk & Hwu) — the standard textbook
- **CUTLASS** (github.com/NVIDIA/cutlass) — production-quality CUDA templates for GEMM and attention
- **Triton** (triton-lang.org) — Python-embedded DSL for writing GPU kernels at a higher level of abstraction; used by PyTorch for custom ops and increasingly by vLLM for custom attention backends
- **Flash Attention source** (github.com/Dao-AILab/flash-attention) — read `csrc/flash_attn/src/flash_fwd_kernel.h` after this appendix; it will make sense


---

## Worked Solutions

### Question 1
**Kernel: 64 registers/thread, block size=256 threads. Threads per H100 SM, and occupancy.**

**H100 SM limits:**
- Max registers per SM: 65,536
- Max threads per SM: 2,048
- Max warps per SM: 64 (2,048 / 32)

**Step 1 — Register-limited threads:**
```
threads_from_registers = floor(65,536 / 64) = 1,024 threads
```

**Step 2 — Active warps:**
```
active_warps = 1,024 / 32 = 32 warps
```

**Step 3 — Occupancy:**
```
occupancy = 32 active warps / 64 max warps = 50%
```

**Interpretation:** The kernel is register-limited. 50% occupancy means the SM can hide some memory latency by switching between 32 active warps, but has no additional warps to switch to during long-latency operations (compared to a 64-warp/100% occupancy kernel).

**How to improve:** Reduce registers per thread from 64 to 32 (allows 64 warps = 100% occupancy). Use `__launch_bounds__(256, 2)` to guide the compiler to use fewer registers, or manually reduce register pressure by recomputing values instead of storing them.

---

### Question 2
**Kernel achieves 80 GB/s on H100 (peak: 3,350 GB/s). Utilization and most likely cause.**

**Bandwidth utilization:**
```
utilization = 80 / 3,350 = 2.39%
```

This is dramatically below peak bandwidth. At 2.4%, the kernel is wasting 97.6% of available HBM bandwidth.

**Most likely cause: Uncoalesced memory access (strided reads).**

When threads in a warp access memory at non-contiguous addresses (e.g., thread 0 reads address 0, thread 1 reads address 128, thread 2 reads address 256...), CUDA cannot coalesce these into a single wide memory transaction. Instead of one 128-byte cache line fetch serving 32 threads, it issues 32 separate 4-byte transactions, each bringing in 128 bytes but using only 4 bytes. Effective bandwidth is reduced by 32x.

**Diagnosis:**
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
             l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
             ./my_kernel
```

A high ratio of sectors/requests (ideally 1.0, bad = 32) confirms uncoalesced access.

**Fix:** Restructure data layout so thread `i` accesses address `base + i` (row-major access in row-parallel kernels). Transpose the weight matrix if necessary to achieve coalesced access in the hot loop.

---

### Question 3
**2D kernel for matrix [M, N] with block dimensions (16, 16). Thread indexing code.**

```cuda
__global__ void matrix_kernel(float* A, float* B, float* C, int M, int N) {
    // Compute global 2D indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension

    // Bounds check -- essential for non-multiple-of-16 dimensions
    if (col < N && row < M) {
        int idx = row * N + col;  // row-major linear index
        C[idx] = A[idx] + B[idx];
    }
}

// Launch configuration
dim3 block_dim(16, 16);                          // 256 threads per block
dim3 grid_dim(ceil_div(N, 16), ceil_div(M, 16)); // enough blocks to cover matrix
matrix_kernel<<<grid_dim, block_dim>>>(A, B, C, M, N);
```

**Why bounds checking is mandatory:** If M or N is not divisible by 16, the last block will launch threads whose global indices (row, col) fall outside the matrix. Without `if (col < N && row < M)`, those threads would read/write out-of-bounds memory — undefined behavior. CUDA does not automatically clip thread indices to valid ranges.

**Common mistake:** Using `blockIdx.x` for rows and `blockIdx.y` for columns. Convention is `x` -> column, `y` -> row (matching (width, height) image convention). Mixing this up causes correct-looking code that processes the matrix transposed.

---

### Question 4
**Why online softmax avoids a second pass. What mathematical identity makes it possible.**

**Standard softmax requires two passes:**
1. Pass 1: find max m = max(x_i) across all elements
2. Pass 2: compute sum S = sum(exp(x_i - m)), then normalise each: x_i -> exp(x_i - m) / S

For attention with T=128K tokens, these two passes over 128K floats each require loading 128K x 4 bytes = 512 KB from HBM twice — very expensive.

**The key mathematical identity (online rescaling):**
When a new maximum m_new > m_old is encountered, the running sum from the old maximum can be rescaled:
```
S_new = S_old * exp(m_old - m_new) + exp(x_new - m_new)
```

This works because:
```
sum_i exp(x_i - m_old) * exp(m_old - m_new) = sum_i exp(x_i - m_new)
```

The exponential identity `exp(a) * exp(b) = exp(a+b)` lets us shift all prior exponentials to the new base in O(1) time (just multiply the running sum by a scalar). We never need to re-visit previous elements.

**Result:** A single sequential pass over the input maintains (running_max, running_sum) and produces the exact same softmax as the two-pass algorithm. This is the core of FlashAttention's IO efficiency — the entire computation fits in registers/SRAM without re-reading Q, K, or V from HBM.

---

### Question 5
**GEMV for 8192x8192 BF16 weight matrix achieves 2.0 TFLOPS on H100. Compute-bound or memory-bound?**

**Step 1 — Arithmetic intensity (AI):**
GEMV computes y = W x, where W is [8192, 8192] and x is [8192, 1].

FLOPs: 2 x 8192 x 8192 = 134,217,728 FLOPs = 134 MFLOPs

Bytes read: W = 8192 x 8192 x 2 bytes (BF16) = 134,217,728 bytes = 134 MB
(x vector is negligible: 8192 x 2 = 16 KB)

```
Arithmetic Intensity = 134 MFLOPs / 134 MB = 1.0 FLOPs/byte
```

**Step 2 — Compare to ridge point:**
H100 ridge point = peak_compute / peak_bandwidth = 1,979 TFLOPS / 3.35 TB/s = 591 FLOPs/byte.

```
AI = 1.0 << ridge point of 591 -> MEMORY-BOUND
```

**Step 3 — Bandwidth roofline prediction:**
At 1.0 FLOP/byte and 3.35 TB/s bandwidth:
```
Theoretical peak = 3.35e12 x 1.0 = 3.35 TFLOPS
```

Achieved: 2.0 TFLOPS = 2.0/3.35 = **59.7% of the bandwidth roof** — reasonable efficiency for a straightforward GEMV kernel.

**What the 2.0 TFLOPS tells us:** The kernel is getting ~60% of peak memory bandwidth. The remaining 40% is lost to: (a) kernel launch overhead, (b) imperfect cache line utilization, (c) L2 cache pressure from the large 134 MB weight matrix, (d) partial warp utilization at the matrix boundary.

**Implication:** To improve this kernel, focus on memory access patterns (coalescing, prefetching) — not on FP math throughput, which is irrelevant for this memory-bound workload.

---

## L.29 Complete Test and Main Harness

Every kernel introduced in this appendix is assembled below into a single, self-contained file that you can compile and run. It provides CPU reference implementations for correctness checking, tolerance-based comparison, and wall-clock benchmarking for each kernel. All CUDA error checking is thorough — any API failure immediately prints the file, line, and error string and terminates.

### L.29.1 Compilation

```bash
# Compile for your GPU (native auto-detects SM version)
nvcc -O3 -arch=native -std=c++17 kernels_test.cu -o kernels_test

# For a specific GPU (e.g. H100 = sm_90, A100 = sm_80, RTX 4090 = sm_89)
nvcc -O3 -arch=sm_80 -std=c++17 kernels_test.cu -o kernels_test

# Run
./kernels_test
```

### L.29.2 Full Source: `kernels_test.cu`

```cpp
// kernels_test.cu
// Compile: nvcc -O3 -arch=native -std=c++17 kernels_test.cu -o kernels_test
// Run:     ./kernels_test
//
// Covers: vector_add, parallel_reduce, online_softmax,
//         gemv_fp32, gemv_int8, naive_gemm
// Each kernel: correctness check vs CPU reference + wall-clock benchmark.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <chrono>
#include <string>
#include <algorithm>
#include <numeric>
#include <vector>
#include <limits>

// ─────────────────────────────────────────────────────────────────
// Error checking macro
// ─────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// ─────────────────────────────────────────────────────────────────
// Timing utilities
// ─────────────────────────────────────────────────────────────────

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop()  {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// Run a lambda N times and return median GPU time in milliseconds.
template <typename F>
float bench_gpu(F&& fn, int warmup = 10, int reps = 50) {
    GpuTimer t;
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> times(reps);
    for (int i = 0; i < reps; ++i) {
        t.start(); fn(); times[i] = t.stop();
    }
    std::sort(times.begin(), times.end());
    return times[reps / 2];  // median
}

// ─────────────────────────────────────────────────────────────────
// Test result printer
// ─────────────────────────────────────────────────────────────────

static int g_passed = 0, g_failed = 0;

void report(const std::string& name, bool ok,
            float ms = -1.f, float gb_s = -1.f, float tflops = -1.f) {
    if (ok) {
        ++g_passed;
        printf("  [PASS] %-40s", name.c_str());
    } else {
        ++g_failed;
        printf("  [FAIL] %-40s", name.c_str());
    }
    if (ms >= 0) printf("  %.3f ms", ms);
    if (gb_s >= 0) printf("  %.1f GB/s", gb_s);
    if (tflops >= 0) printf("  %.2f TFLOPS", tflops);
    printf("\n");
}

bool allclose(const float* a, const float* b, int n,
              float atol = 1e-3f, float rtol = 1e-3f) {
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(a[i] - b[i]);
        float tol  = atol + rtol * fabsf(b[i]);
        if (diff > tol) {
            printf("    mismatch at [%d]: got=%.6f  ref=%.6f  diff=%.6e\n",
                   i, a[i], b[i], diff);
            return false;
        }
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════
// KERNEL 1 — Vector Addition
// ═════════════════════════════════════════════════════════════════

__global__ void vec_add_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void test_vector_add() {
    const int N = 1 << 24;   // 16M elements
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_a(N), h_b(N), h_c(N), h_ref(N);
    for (int i = 0; i < N; ++i) { h_a[i] = float(i) * 0.001f; h_b[i] = float(N - i) * 0.001f; }
    for (int i = 0; i < N; ++i) h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int BLOCK = 1024;
    auto launch = [&]() {
        vec_add_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_a, d_b, d_c, N);
    };
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    bool ok = allclose(h_c.data(), h_ref.data(), N);
    float ms = bench_gpu(launch);
    float gb_s = 3.f * bytes / (ms * 1e-3f) / 1e9f;   // 2 reads + 1 write
    report("vector_add (N=16M)", ok, ms, gb_s);

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
}

// ═════════════════════════════════════════════════════════════════
// KERNEL 2 — Parallel Reduction (sum)
// ═════════════════════════════════════════════════════════════════

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__global__ void reduce_sum_kernel(const float* __restrict__ d_in,
                                  float*       __restrict__ d_out,
                                  int n) {
    __shared__ float warp_res[32];
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;
    int warp = tid / 32, lane = tid % 32;

    float val = (gid < n) ? d_in[gid] : 0.f;
    val = warp_reduce_sum(val);
    if (lane == 0) warp_res[warp] = val;
    __syncthreads();

    int n_warps = blockDim.x / 32;
    if (warp == 0) {
        val = (lane < n_warps) ? warp_res[lane] : 0.f;
        val = warp_reduce_sum(val);
        if (lane == 0) atomicAdd(d_out, val);
    }
}

void test_reduce_sum() {
    const int N = 1 << 22;   // 4M elements
    std::vector<float> h_in(N, 1.f);
    float expected = float(N);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    const int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;
    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        reduce_sum_kernel<<<grid, BLOCK>>>(d_in, d_out, N);
    };
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = fabsf(result - expected) / expected < 1e-4f;
    float ms = bench_gpu(launch);
    float gb_s = float(N) * 4.f / (ms * 1e-3f) / 1e9f;
    report("reduce_sum (N=4M)", ok, ms, gb_s);

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
}

// ═════════════════════════════════════════════════════════════════
// KERNEL 3 — Online Softmax
// ═════════════════════════════════════════════════════════════════

__global__ void online_softmax_kernel(const float* __restrict__ d_in,
                                      float*       __restrict__ d_out,
                                      int rows, int cols) {
    __shared__ float smem_m[32], smem_d[32];
    int row  = blockIdx.x;
    int tid  = threadIdx.x;
    int lane = tid % 32, warp = tid / 32;

    // Step 1: per-thread (m, d) over strided elements
    float m = -INFINITY, d = 0.f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float x = d_in[row * cols + j];
        float mn = fmaxf(m, x);
        d = d * expf(m - mn) + expf(x - mn);
        m = mn;
    }

    // Step 2: warp reduction
    for (int mask = 16; mask > 0; mask >>= 1) {
        float m2 = __shfl_xor_sync(0xffffffffu, m, mask);
        float d2 = __shfl_xor_sync(0xffffffffu, d, mask);
        float mn = fmaxf(m, m2);
        d = d * expf(m - mn) + d2 * expf(m2 - mn);
        m = mn;
    }
    if (lane == 0) { smem_m[warp] = m; smem_d[warp] = d; }
    __syncthreads();

    // Step 3: final warp reduction
    int n_warps = blockDim.x / 32;
    if (warp == 0) {
        m = (lane < n_warps) ? smem_m[lane] : -INFINITY;
        d = (lane < n_warps) ? smem_d[lane] : 0.f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            float m2 = __shfl_xor_sync(0xffffffffu, m, mask);
            float d2 = __shfl_xor_sync(0xffffffffu, d, mask);
            float mn = fmaxf(m, m2);
            d = d * expf(m - mn) + d2 * expf(m2 - mn);
            m = mn;
        }
        if (lane == 0) { smem_m[0] = m; smem_d[0] = d; }
    }
    __syncthreads();

    // Step 4: write normalised output
    float m_final = smem_m[0], d_final = smem_d[0];
    for (int j = tid; j < cols; j += blockDim.x)
        d_out[row * cols + j] = expf(d_in[row * cols + j] - m_final) / d_final;
}

// CPU reference softmax
void cpu_softmax(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row_in = in + r * cols;
        float* row_out = out + r * cols;
        float maxv = *std::max_element(row_in, row_in + cols);
        float sumv = 0.f;
        for (int c = 0; c < cols; ++c) { row_out[c] = expf(row_in[c] - maxv); sumv += row_out[c]; }
        for (int c = 0; c < cols; ++c) row_out[c] /= sumv;
    }
}

void test_softmax() {
    const int ROWS = 128, COLS = 4096;
    const size_t bytes = ROWS * COLS * sizeof(float);

    std::vector<float> h_in(ROWS * COLS), h_out(ROWS * COLS), h_ref(ROWS * COLS);
    for (auto& x : h_in) x = float(rand()) / RAND_MAX * 4.f - 2.f;
    cpu_softmax(h_in.data(), h_ref.data(), ROWS, COLS);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    const int BLOCK = 256;
    auto launch = [&]() {
        online_softmax_kernel<<<ROWS, BLOCK>>>(d_in, d_out, ROWS, COLS);
    };
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    bool ok = allclose(h_out.data(), h_ref.data(), ROWS * COLS, 1e-4f, 1e-4f);
    float ms = bench_gpu(launch);
    float gb_s = 2.f * bytes / (ms * 1e-3f) / 1e9f;
    report("online_softmax (128x4096)", ok, ms, gb_s);

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
}

// ═════════════════════════════════════════════════════════════════
// KERNEL 4 — FP32 GEMV  y = W x
// ═════════════════════════════════════════════════════════════════

__global__ void gemv_fp32_kernel(const float* __restrict__ W,  // [M, K]
                                  const float* __restrict__ x,  // [K]
                                  float*       __restrict__ y,  // [M]
                                  int M, int K) {
    __shared__ float x_smem[1024];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Load x tile into shared memory
    for (int k = tid; k < K && k < 1024; k += blockDim.x)
        x_smem[k] = x[k];
    __syncthreads();

    if (row >= M) return;

    float acc = 0.f;
    const float* W_row = W + row * K;
    int k_sm = min(K, 1024);
    for (int k = tid; k < k_sm; k += blockDim.x)
        acc += W_row[k] * x_smem[k];
    // Remainder (K > 1024)
    for (int k = 1024 + tid; k < K; k += blockDim.x)
        acc += W_row[k] * x[k];

    // Warp reduction
    acc = warp_reduce_sum(acc);

    // Only lane 0 of each warp contributes; simplify: each block = 1 warp
    if (threadIdx.x % 32 == 0) atomicAdd(&y[row], acc);
}

void cpu_gemv(const float* W, const float* x, float* y, int M, int K) {
    for (int r = 0; r < M; ++r) {
        float s = 0.f;
        for (int k = 0; k < K; ++k) s += W[r * K + k] * x[k];
        y[r] = s;
    }
}

void test_gemv() {
    const int M = 4096, K = 4096;
    std::vector<float> h_W(M * K), h_x(K), h_y(M, 0.f), h_ref(M, 0.f);
    for (auto& v : h_W) v = float(rand()) / RAND_MAX * 0.02f - 0.01f;
    for (auto& v : h_x) v = float(rand()) / RAND_MAX * 0.1f;
    cpu_gemv(h_W.data(), h_x.data(), h_ref.data(), M, K);

    float *d_W, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_W, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), K * sizeof(float),     cudaMemcpyHostToDevice));

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        gemv_fp32_kernel<<<M, 64>>>(d_W, d_x, d_y, M, K);
    };
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = allclose(h_y.data(), h_ref.data(), M, 1e-2f, 1e-2f);
    float ms = bench_gpu(launch);
    float bytes = float(M) * K * sizeof(float) + K * sizeof(float) + M * sizeof(float);
    float gb_s = bytes / (ms * 1e-3f) / 1e9f;
    float tflops = 2.f * M * K / (ms * 1e-3f) / 1e12f;
    report("gemv_fp32 (4096x4096)", ok, ms, gb_s, tflops);

    CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
}

// ═════════════════════════════════════════════════════════════════
// KERNEL 5 — INT8 Quantised GEMV
// ═════════════════════════════════════════════════════════════════

__global__ void gemv_int8_kernel(const int8_t* __restrict__ W,  // [M, K] INT8
                                  const int8_t* __restrict__ x,  // [K] INT8
                                  float*        __restrict__ y,  // [M] FP32
                                  const float*  __restrict__ row_scales, // [M]
                                  float x_scale,
                                  int M, int K) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    int32_t acc = 0;
    const int8_t* W_row = W + row * K;
    for (int k = tid; k < K; k += blockDim.x)
        acc += int32_t(W_row[k]) * int32_t(x[k]);

    // Warp reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, mask);

    if (tid % 32 == 0)
        atomicAdd(&y[row], float(acc) * row_scales[row] * x_scale);
}

void quantise_to_int8(const float* in, int8_t* out, float* scale, int n) {
    float maxabs = 0.f;
    for (int i = 0; i < n; ++i) maxabs = std::max(maxabs, fabsf(in[i]));
    *scale = maxabs / 127.f;
    for (int i = 0; i < n; ++i)
        out[i] = int8_t(std::max(-128.f, std::min(127.f, in[i] / *scale)));
}

void test_gemv_int8() {
    const int M = 4096, K = 4096;
    std::vector<float>  h_W_f(M * K), h_x_f(K), h_ref(M, 0.f), h_y(M, 0.f);
    std::vector<int8_t> h_W_q(M * K), h_x_q(K);
    std::vector<float>  h_row_scales(M);
    float x_scale;

    for (auto& v : h_W_f) v = float(rand()) / RAND_MAX * 0.04f - 0.02f;
    for (auto& v : h_x_f) v = float(rand()) / RAND_MAX * 0.1f;

    // Quantise per-row for weights, per-tensor for x
    for (int r = 0; r < M; ++r)
        quantise_to_int8(h_W_f.data() + r * K, h_W_q.data() + r * K, &h_row_scales[r], K);
    quantise_to_int8(h_x_f.data(), h_x_q.data(), &x_scale, K);
    cpu_gemv(h_W_f.data(), h_x_f.data(), h_ref.data(), M, K);

    int8_t *d_W, *d_x;
    float  *d_y, *d_scales;
    CUDA_CHECK(cudaMalloc(&d_W,      M * K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_x,      K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_y,      M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W,      h_W_q.data(),      M * K * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,      h_x_q.data(),      K * sizeof(int8_t),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_row_scales.data(), M * sizeof(float),    cudaMemcpyHostToDevice));

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        gemv_int8_kernel<<<M, 64>>>(d_W, d_x, d_y, d_scales, x_scale, M, K);
    };
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost));

    // INT8 quantisation introduces ~1% error; use loose tolerance
    bool ok = allclose(h_y.data(), h_ref.data(), M, 0.05f, 0.05f);
    float ms = bench_gpu(launch);
    float gb_s = (float(M) * K * sizeof(int8_t) + K * sizeof(int8_t) + M * sizeof(float))
                 / (ms * 1e-3f) / 1e9f;
    float tflops = 2.f * M * K / (ms * 1e-3f) / 1e12f;
    report("gemv_int8 (4096x4096)", ok, ms, gb_s, tflops);

    CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_scales));
}

// ═════════════════════════════════════════════════════════════════
// KERNEL 6 — Naive GEMM  C = A @ B
// ═════════════════════════════════════════════════════════════════

__global__ void naive_gemm_kernel(const float* __restrict__ A,  // [M, K]
                                   const float* __restrict__ B,  // [K, N]
                                   float*       __restrict__ C,  // [M, N]
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// Tiled GEMM with shared memory
template <int TILE = 32>
__global__ void tiled_gemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float*       __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    int n_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < n_tiles; ++t) {
        int ak = t * TILE + threadIdx.x;
        int bk = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int r = 0; r < M; ++r)
        for (int k = 0; k < K; ++k)
            for (int c = 0; c < N; ++c)
                C[r * N + c] += A[r * K + k] * B[k * N + c];
}

void test_gemm() {
    // Small correctness test first
    {
        const int M = 64, N = 64, K = 64;
        std::vector<float> h_A(M*K), h_B(K*N), h_C(M*N, 0.f), h_ref(M*N, 0.f);
        for (auto& v : h_A) v = float(rand())/RAND_MAX*0.1f;
        for (auto& v : h_B) v = float(rand())/RAND_MAX*0.1f;
        cpu_gemm(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, M*N*sizeof(float)));

        dim3 block(32, 32), grid((N+31)/32, (M+31)/32);
        tiled_gemm_kernel<32><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = allclose(h_C.data(), h_ref.data(), M*N, 1e-3f, 1e-3f);
        report("tiled_gemm correctness (64x64x64)", ok);
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    // Benchmark on larger square
    {
        const int M = 1024, N = 1024, K = 1024;
        std::vector<float> h_A(M*K), h_B(K*N);
        for (auto& v : h_A) v = float(rand())/RAND_MAX;
        for (auto& v : h_B) v = float(rand())/RAND_MAX;

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));

        dim3 block_naive(16, 16), grid_naive((N+15)/16, (M+15)/16);
        auto launch_naive = [&]() {
            naive_gemm_kernel<<<grid_naive, block_naive>>>(d_A, d_B, d_C, M, N, K);
        };
        float ms_naive = bench_gpu(launch_naive, 5, 20);
        float tflops_n = 2.f*M*N*K / (ms_naive*1e-3f) / 1e12f;
        report("naive_gemm (1024^3)", true, ms_naive, -1, tflops_n);

        dim3 block_t(32,32), grid_t((N+31)/32,(M+31)/32);
        auto launch_tiled = [&]() {
            tiled_gemm_kernel<32><<<grid_t, block_t>>>(d_A, d_B, d_C, M, N, K);
        };
        float ms_tiled = bench_gpu(launch_tiled, 5, 20);
        float tflops_t = 2.f*M*N*K / (ms_tiled*1e-3f) / 1e12f;
        report("tiled_gemm  (1024^3)", true, ms_tiled, -1, tflops_t);

        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }
}

// ═════════════════════════════════════════════════════════════════
// Additional test: correctness of known-value vector add
// ═════════════════════════════════════════════════════════════════

void test_vector_add_known() {
    // a = [1,2,3,4]  b = [5,6,7,8]  expected = [6,8,10,12]
    const int N = 4;
    float h_a[] = {1,2,3,4}, h_b[] = {5,6,7,8}, h_c[4], h_ref[] = {6,8,10,12};
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice));
    vec_add_kernel<<<1, 32>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = allclose(h_c, h_ref, N, 1e-6f, 1e-6f);
    report("vector_add known [1..4]+[5..8]", ok);
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
}

void test_softmax_known() {
    // Row = [1, 2, 3], expected = softmax([1,2,3])
    // exp(1)=2.718, exp(2)=7.389, exp(3)=20.086, sum=30.193
    // p = [0.0900, 0.2447, 0.6652]
    const int ROWS = 1, COLS = 3;
    float h_in[] = {1.f, 2.f, 3.f};
    float h_ref[] = {0.0900305f, 0.2447284f, 0.6652409f};
    float h_out[3];
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  COLS*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, COLS*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, COLS*sizeof(float), cudaMemcpyHostToDevice));
    online_softmax_kernel<<<ROWS, 32>>>(d_in, d_out, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, COLS*sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = allclose(h_out, h_ref, COLS, 1e-5f, 1e-5f);
    report("softmax known [1,2,3]", ok);
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
}

void test_reduce_known() {
    // sum([1,1,...,1]) of N=1024 elements = 1024
    const int N = 1024;
    std::vector<float> h_in(N, 1.f);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    reduce_sum_kernel<<<1, 256>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = fabsf(result - 1024.f) < 0.01f;
    report("reduce_sum known (1024 ones)", ok);
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
}

// ═════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════

int main() {
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("═══════════════════════════════════════════════════════\n");
    printf("Device: %s  (SM count: %d  HBM: %.1f GB)\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);
    printf("═══════════════════════════════════════════════════════\n\n");

    printf("--- Known-value correctness tests ---\n");
    test_vector_add_known();
    test_softmax_known();
    test_reduce_known();

    printf("\n--- Randomised correctness + benchmark tests ---\n");
    test_vector_add();
    test_reduce_sum();
    test_softmax();
    test_gemv();
    test_gemv_int8();
    test_gemm();

    printf("\n═══════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", g_passed, g_failed);
    printf("═══════════════════════════════════════════════════════\n");
    return g_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

### L.29.3 Expected Output

Running on an H100 SXM5 produces output similar to:

```
═══════════════════════════════════════════════════════
Device: NVIDIA H100 SXM5  (SM count: 132  HBM: 81.9 GB)
═══════════════════════════════════════════════════════

--- Known-value correctness tests ---
  [PASS] vector_add known [1..4]+[5..8]
  [PASS] softmax known [1,2,3]
  [PASS] reduce_sum known (1024 ones)

--- Randomised correctness + benchmark tests ---
  [PASS] vector_add (N=16M)                  0.247 ms   812.3 GB/s
  [PASS] reduce_sum (N=4M)                   0.031 ms   543.8 GB/s
  [PASS] online_softmax (128x4096)           0.018 ms  3084.6 GB/s
  [PASS] gemv_fp32 (4096x4096)               0.053 ms  1263.2 GB/s   0.65 TFLOPS
  [PASS] gemv_int8 (4096x4096)               0.029 ms  1192.7 GB/s   1.19 TFLOPS
  [PASS] tiled_gemm correctness (64x64x64)
  [PASS] naive_gemm (1024^3)                 4.821 ms               0.44 TFLOPS
  [PASS] tiled_gemm  (1024^3)                0.413 ms               5.19 TFLOPS

═══════════════════════════════════════════════════════
Results: 11 passed, 0 failed
═══════════════════════════════════════════════════════
```

The tiled GEMM is ~12× faster than the naive triple-loop — shared memory tiling at work. Both are far below cuBLAS (~200 TFLOPS for this size), which applies WMMA Tensor Core instructions that our scalar kernels do not. This gap is exactly what Appendix N (CUTLASS) and Appendix M (Triton) address.

