# Chapter 2.5: GPU Memory Architecture — Registers, Shared Memory, Caches, and Global Memory

> *"The difference between a GPU kernel that runs at 80% of peak bandwidth and one that runs at 8% is almost always a memory problem. Fix the memory access pattern and the FLOP count takes care of itself."*

---

**What you will understand by the end of this chapter:**

- Every memory space inside a GPU — registers, shared memory, L1, L2, constant, texture, local, and global — what each costs in latency, bandwidth, and capacity
- Why coalesced vs uncoalesced global memory access produces a 30× throughput difference on the same hardware
- How shared memory bank conflicts work and how the padding trick eliminates them
- How the SMEM/L1 split is configured on Ampere and Hopper GPUs
- How FlashAttention, PagedAttention, and tiled GEMM each exploit this hierarchy
- How to calculate occupancy and why it determines how much parallelism hides memory latency

**What you need to know first:**

- Chapter 1 (autoregressive decode, why throughput matters)
- Chapter 2 (HBM vs DRAM at the system level — this chapter goes inside the GPU)

---

## 2.5.1 The GPU Memory Architecture

`[FOUNDATIONAL]`

### On-Chip vs Near-Chip vs Off-Chip

Before mapping the hierarchy, one critical distinction that is consistently blurred in tutorials: **HBM is not on the GPU chip**. It is physically separate DRAM that sits extremely close to the GPU die — but it is not on the silicon itself. This matters because every optimization in this chapter is ultimately about avoiding round trips to HBM, which is the slowest tier you will regularly touch during a kernel.

The three tiers of GPU memory by physical location:

**On-chip (on the GPU die itself):**
These live on the same silicon as the CUDA cores and SMs. Accessing them costs 1–50 cycles. They use SRAM technology — extremely fast but expensive and power-hungry per bit, which is why capacity is tiny (kilobytes to tens of megabytes).

- Registers — inside each SM's compute units
- Shared Memory (SMEM) — also inside the SM, programmer-controlled SRAM scratchpad
- L1 Cache — shares the same physical SRAM block as SMEM on Ampere
- L2 Cache — a large on-die SRAM cache shared across all SMs

**Near-chip (off-die, but in the same package):**
HBM (High Bandwidth Memory) stacks sit on the same **silicon interposer** as the GPU die — an intermediary substrate that connects everything with extremely short, extremely wide signal traces. This 2.5D packaging technique is what gives HBM its multi-terabyte-per-second bandwidth despite being physically off the GPU die. The stacks are typically 2–6 mm away from the GPU die, connected by thousands of through-silicon vias (TSVs).

This is fundamentally different from traditional GDDR memory, which sits on the PCB board — centimeters away, connected via narrow PCIe-width traces.

```
  Traditional GPU (GDDR):
  ┌────────────────────────────┐  PCB board
  │  [GDDR chip] [GDDR chip]   │
  │         ↑ ↑                │  ← long PCB traces (~cm)
  │       [GPU die]            │  ← narrow bus (~256–384 bit)
  └────────────────────────────┘
  Typical BW: 300–900 GB/s

  HBM GPU (A100 / H100):
  ┌────────────────────────────────────────────────┐
  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   │  Silicon interposer
  │  │  HBM     │   │  GPU die │   │  HBM     │   │
  │  │  Stack   │   │  (A100)  │   │  Stack   │   │  ← µm-scale TSV connections
  │  │  (40 GB) │   │  (54B Tr)│   │  (40 GB) │   │  ← 5120-bit bus per stack
  │  └──────────┘   └──────────┘   └──────────┘   │
  └────────────────────────────────────────────────┘
  Typical BW: 2,000–3,350 GB/s  (A100–H100)
```

Why not put more memory directly on the GPU die as SRAM? Because SRAM is ~100× larger per bit than DRAM. An A100 has 40 GB of HBM. Storing 40 GB as on-chip SRAM would require a die roughly 50× the size of the actual A100 — physically and economically impossible with current process nodes.

**Off-chip (system DRAM / NVMe):**
System RAM and storage are separated from the GPU by the PCIe bus. Bandwidth drops to 32–100 GB/s (PCIe) or 5–15 GB/s (NVMe). These are covered in Chapter 2.

---

### The Complete GPU Memory Map

```
  GPU MEMORY ARCHITECTURE  (NVIDIA Ampere — A100 / RTX 3090)
  ═══════════════════════════════════════════════════════════════════════════

  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  ON-CHIP  (GPU silicon die)                                          ║
  ║                                                                       ║
  ║  ┌─────────────────────────────────────────────────────────────────┐  ║
  ║  │  SM (Streaming Multiprocessor)  ×108 on A100                   │  ║
  ║  │                                                                 │  ║
  ║  │  ┌──────────────────┐  ┌──────────────────┐                    │  ║
  ║  │  │  Warp 0          │  │  Warp 1 ... 63   │  ← 64 warps/SM    │  ║
  ║  │  │  ┌──┐ ┌──┐ ...   │  │  ┌──┐ ┌──┐ ...  │                    │  ║
  ║  │  │  │R0│ │R1│        │  │  │R0│ │R1│       │  REGISTERS        │  ║
  ║  │  │  └──┘ └──┘        │  │  └──┘ └──┘       │  ~1 cycle        │  ║
  ║  │  │  255 regs/thread  │  │  per-thread       │  256 KB/SM total │  ║
  ║  │  └──────────────────┘  └──────────────────┘                    │  ║
  ║  │                                                                 │  ║
  ║  │  ┌─────────────────────────────────────────────────────────┐   │  ║
  ║  │  │  Unified SMEM + L1  (192 KB total per SM, Ampere)       │   │  ║
  ║  │  │  ┌────────────────────┐  ┌──────────────────────────┐   │   │  ║
  ║  │  │  │  Shared Memory     │  │  L1 Data Cache           │   │   │  ║
  ║  │  │  │  (SMEM)  0–160 KB  │  │  remainder of 192 KB     │   │   │  ║
  ║  │  │  │  __shared__        │  │  hardware-managed        │   │   │  ║
  ║  │  │  │  programmer ctrl   │  │  global mem cache        │   │   │  ║
  ║  │  │  └────────────────────┘  └──────────────────────────┘   │   │  ║
  ║  │  │  ~5–10 cycles latency                                    │   │  ║
  ║  │  └─────────────────────────────────────────────────────────┘   │  ║
  ║  │                                                                 │  ║
  ║  │  ┌─────────────────────────────────────────────────────────┐   │  ║
  ║  │  │  Constant Cache (~8 KB)  │  Texture Cache (~256 KB)     │   │  ║
  ║  │  └─────────────────────────────────────────────────────────┘   │  ║
  ║  └─────────────────────────────────────────────────────────────────┘  ║
  ║                                                                       ║
  ║  ┌─────────────────────────────────────────────────────────────────┐  ║
  ║  │  L2 Cache  (40 MB on A100, 50 MB on H100)                      │  ║
  ║  │  Shared by ALL SMs  │  ~30–50 cycle latency                    │  ║
  ║  └─────────────────────────────────────────────────────────────────┘  ║
  ╚═══════════════════════════════════════════════════════════════════════╝
                    │  Through silicon interposer (µm-scale)
  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  NEAR-CHIP  (off-die, same package — silicon interposer)             ║
  ║                                                                       ║
  ║  ┌─────────────────────────────────────────────────────────────────┐  ║
  ║  │  HBM (High Bandwidth Memory) — DRAM technology, NOT SRAM       │  ║
  ║  │  A100: 80 GB  @ 2.0 TB/s  │  H100: 80 GB @ 3.35 TB/s          │  ║
  ║  │  ~400–800 cycle latency                                         │  ║
  ║  │  Holds: model weights · KV cache · activations · intermediates  │  ║
  ║  │                                                                 │  ║
  ║  │  Constant Memory (64 KB logical) ─────────── cached on-chip    │  ║
  ║  │  Texture Memory  (part of HBM)  ─────────── cached on-chip     │  ║
  ║  └─────────────────────────────────────────────────────────────────┘  ║
  ╚═══════════════════════════════════════════════════════════════════════╝
                    │  PCIe bus (32–64 GB/s)
  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  OFF-CHIP  (system board)                                            ║
  ║  System DRAM: 50–100 GB/s   │   NVMe SSD: 5–15 GB/s                 ║
  ╚═══════════════════════════════════════════════════════════════════════╝
```

---

### Why Not More On-Chip Memory?

The constraint is physics and economics. SRAM (the technology behind registers, SMEM, and L1/L2 caches) requires 6 transistors per bit. DRAM (the technology behind HBM and system RAM) requires 1 transistor + 1 capacitor per bit. SRAM is ~6× larger per bit and ~10× more power-hungry per bit than DRAM.

| Technology | Transistors/bit | Area/bit | Speed | Cost/GB |
|---|---|---|---|---|
| Register file (SRAM) | 6T | Largest | ~1 cycle | Highest |
| SMEM / L1 / L2 (SRAM) | 6T | Large | 5–50 cycles | High |
| HBM (DRAM, stacked) | 1T+1C | Small | 400–800 cycles | Medium |
| System DRAM (DDR5) | 1T+1C | Small | ~800 cycles + PCIe | Low |

The A100 has 40 MB of L2 cache (on-chip SRAM). Scaling that to 40 GB — the HBM size — would require a die roughly 1,000× larger. HBM's answer is to use dense DRAM stacked vertically in multiple layers and place it millimeters away on the interposer, not micrometers away on the die.

---

### Capacity and Latency Reference

| Memory Space | Location | Capacity | Latency | Bandwidth | Managed by |
|---|---|---|---|---|---|
| Registers | **On-chip** (per SM) | 255 regs/thread · 256 KB/SM | ~1 cycle | ~100 TB/s | Compiler |
| Shared Memory | **On-chip** (per SM) | 0–160 KB/SM (Ampere) | ~5–10 cycles | ~20 TB/s | Programmer |
| L1 Cache | **On-chip** (per SM) | Remainder of 192 KB | ~5–10 cycles | ~20 TB/s | Hardware |
| L2 Cache | **On-chip** (per GPU) | 40 MB (A100), 50 MB (H100) | ~30–50 cycles | ~4–12 TB/s | Hardware |
| Constant Cache | **On-chip** (per SM) | ~8 KB | ~5 cycles (hit) | Broadcast | Hardware |
| HBM / Global Mem | **Near-chip** (interposer) | 40–192 GB | ~400–800 cycles | 2–3.35 TB/s | Programmer |
| Local Memory | **Near-chip** (spill→HBM) | Part of HBM | ~400–800 cycles | 2–3.35 TB/s | Compiler |
| System DRAM | **Off-chip** (PCIe) | 32 GB–2 TB | ~1000+ cycles | 50–100 GB/s | OS |

The fundamental insight every CUDA optimization flows from: **every time a thread touches HBM, it pays a 400–800 cycle penalty**. Tiling, shared memory staging, register blocking, FlashAttention fusion, tensor core pipelining — these are all strategies for reusing data before going back to HBM.

---

## 2.5.2 Registers — The Fastest Memory

`[CORE]`

### What Registers Are

Every CUDA thread has its own private register file. Registers hold the thread's local variables, loop counters, intermediate computation results, and function call state. They are the fastest storage on the GPU — accessing a register takes a single clock cycle, with no latency and no bandwidth limit in the traditional sense.

The GPU's register file is physically located inside each SM and is partitioned among all the thread blocks running on that SM at the same time.

### Register Limits and Register Pressure

An Ampere SM has a 256 KB register file shared among all active warps. Each thread can use a maximum of 255 registers. The tension is this: the more registers each thread uses, the fewer threads can be active simultaneously — reducing the GPU's ability to hide memory latency by context-switching between warps.

```
Register file pressure illustration on A100 (108 SMs):

  SM register file: 256 KB = 65,536 × 32-bit registers

  Scenario A — 32 registers per thread:
    Max threads per SM = 65,536 / 32 = 2,048 threads = 64 warps
    (This is the hardware maximum — full occupancy)

  Scenario B — 64 registers per thread:
    Max threads per SM = 65,536 / 64 = 1,024 threads = 32 warps
    (50% occupancy from registers alone)

  Scenario C — 128 registers per thread:
    Max threads per SM = 65,536 / 128 = 512 threads = 16 warps
    (25% occupancy — severe register pressure)
```

### Register Spilling

When a kernel requests more registers than the hardware provides per thread, the compiler spills excess register data to **local memory** — which is physically located in global memory (HBM). A spilled register access has the same ~400–800 cycle latency as a global memory load. This can make a register-heavy kernel 10–50× slower than one that stays within the register budget.

Check register usage with: `nvcc --ptxas-options=-v mykernel.cu`

### What Lives in Registers for LLM Kernels

In a tiled GEMM kernel (the backbone of every weight matrix multiply in a transformer):
- The accumulator for the output tile: `float acc[TILE_M][TILE_N]` — this is the most register-pressure-intensive part
- Loop counters, pointer arithmetic, scale factors
- Loaded fragments of A and B tiles (before they go to SMEM or come from SMEM)

FlashAttention goes further: it keeps the softmax running statistics (`m` = current max, `l` = running sum of exp) **in registers across the K-tile loop**, avoiding a round-trip to global memory for the normalization denominator. This is the core of why FlashAttention is faster than naive attention — it is a register-level optimization.

---

## 2.5.3 Shared Memory — The Programmable On-Chip Scratchpad

`[CORE]`

### What Shared Memory Is

Shared memory (SMEM) is a fast, on-chip memory pool that is shared among all threads in the same thread block. Unlike registers (private per thread), SMEM is explicitly readable and writable by any thread in the block, making it the natural place to stage data that multiple threads need to reuse — exactly the pattern in matrix tiling.

SMEM has roughly the same latency as the L1 cache (~5–10 cycles), but unlike the L1, the programmer controls what goes in and out. The hardware never evicts or replaces SMEM contents without programmer action — it holds what you wrote until the kernel resets it or the block exits.

### Configuring the SMEM / L1 Split

On Ampere (A100, RTX 3090), each SM has 192 KB of unified SMEM+L1 memory. The split is configurable per kernel:

```cpp
// Request 160 KB SMEM (leaves 32 KB for L1)
cudaFuncSetAttribute(my_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);

// Or via launch config
cudaFuncSetAttribute(my_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);  // maximize SMEM
```

| Configuration | SMEM | L1 |
|---|---|---|
| Default (Ampere) | 100 KB | 92 KB |
| Max SMEM | 160 KB | 32 KB |
| Balanced | 128 KB | 64 KB |

FlashAttention uses the max SMEM configuration: it tiles Q, K, V into SMEM blocks as large as possible to minimize global memory traffic.

### Declaring Shared Memory

```cuda
__global__ void tiled_gemm(float* A, float* B, float* C, int N) {
    // Static allocation (size known at compile time)
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    // Dynamic allocation (size passed at launch: <<<grid, block, smem_bytes>>>)
    extern __shared__ float smem[];
    float* tile_A = smem;
    float* tile_B = smem + TILE * TILE;
}
```

### Bank Conflicts — The Hidden Performance Killer

SMEM is organized into **32 banks**, each 4 bytes wide. The 128-byte cache line maps as:

```
Bank:     0    1    2    3    4   ...   31
Bytes:  0-3  4-7  8-11 12-15 16  ... 124-127
Next 128:  128-131 ...
```

When multiple threads in a warp access the same bank simultaneously (but different addresses within that bank), the accesses are **serialized** — called a bank conflict. A 16-way bank conflict makes 16 sequential transactions from a single warp.

**Worked Example 2.5-A — Bank Conflict in Matrix Access**

```
Shared memory array: float tile[32][32]  (32 rows × 32 cols × 4 bytes = 4 KB)

Access pattern — COLUMN traversal (each thread reads its column):
  Thread 0 reads tile[0][0]   → bank 0
  Thread 1 reads tile[1][0]   → bank 0  ← CONFLICT with thread 0!
  Thread 2 reads tile[2][0]   → bank 0  ← CONFLICT
  ...
  Thread 31 reads tile[31][0] → bank 0  ← CONFLICT

All 32 threads access bank 0 → 32-way conflict → 32 serialized transactions
Effective bandwidth: 1/32 of peak SMEM bandwidth

Access pattern — ROW traversal (each thread reads its column of the transposed tile):
  Thread 0 reads tile[0][0]   → bank 0
  Thread 1 reads tile[0][1]   → bank 1  ← different bank
  Thread 2 reads tile[0][2]   → bank 2  ← different bank
  ...
  Thread 31 reads tile[0][31] → bank 31 ← different bank

All 32 threads access different banks → 0 conflicts → 1 transaction
```

**The Padding Fix:**

Adding one extra column breaks the column-access conflict pattern:

```cuda
// Before: 32-way conflict on column access
__shared__ float tile[32][32];

// After: 0 conflicts on column access (pad by 1 float = 4 bytes)
__shared__ float tile[32][33];  // +1 column shifts each row's bank mapping

// Now thread i accessing tile[i][0]:
//   tile[0][0] → offset 0   → bank 0
//   tile[1][0] → offset 33  → bank (33 % 32) = bank 1
//   tile[2][0] → offset 66  → bank (66 % 32) = bank 2
//   All 32 threads hit different banks → 0 conflicts
```

This one-line change — adding a padding column — routinely provides 2–3× speedup on GEMM kernels with column access patterns.

### Async SMEM Loads (`cp.async`)

Ampere introduced the `cp.async` instruction, which copies data from global memory to shared memory **without going through registers** — and does so asynchronously, allowing the SM to continue executing other instructions while the copy completes.

```cuda
// Traditional load (synchronous, goes through registers):
float val = A[global_idx];        // register load
tile[local_row][local_col] = val; // register to SMEM store

// Async load (Ampere+, bypasses registers, overlaps with computation):
#include <cuda/pipeline>
__pipeline_memcpy_async(
    &tile[local_row][local_col],  // destination: SMEM
    &A[global_idx],               // source: global memory
    sizeof(float));               // 4 bytes
__pipeline_commit();
// ... other work here ...
__pipeline_wait_prior(0);         // sync when needed
```

This is how double-buffering works in CUTLASS and FlashAttention: while the GPU computes on tile N from SMEM, tile N+1 is being fetched from global memory asynchronously, hiding the global memory latency completely.

---

## 2.5.4 L1 and L2 Cache — Hardware-Managed Intermediate Storage

`[CORE]`

### L1 Cache

The L1 cache shares the 192 KB unified block with SMEM. Whatever portion is not allocated to `__shared__` variables becomes L1. The L1 serves automatic caching of:

- Global memory loads accessed via regular pointer reads (`float* A`)
- Local memory (register spill) accesses
- Texture memory reads

The L1 cache line size is **128 bytes** — the fundamental unit of global memory access. This directly determines coalescing behavior (Section 2.5.5).

### L2 Cache

The L2 is a large, chip-wide cache shared by all SMs. On the A100 it is 40 MB; on the H100 it is 50 MB. All SMs compete for L2 bandwidth, which is why high-concurrency kernels can become L2 bandwidth-bound even when the L1 hit rate is low.

**L2 residence control (Ampere+):** CUDA 11.2 introduced `cudaStreamAttrValue` to mark memory regions as **persisting** in L2, which is useful when the same data is accessed repeatedly across kernel launches (e.g., model weights during inference):

```cuda
// Mark weight buffer as L2-persistent (use up to 30 MB of A100's 40 MB L2)
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = weight_ptr;
attr.accessPolicyWindow.num_bytes = weight_bytes;
attr.accessPolicyWindow.hitRatio  = 1.0f;  // try to cache everything
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

For inference, this means frequently accessed weight matrices — such as the query/key/value projection weights for the current decode layer — can be "pinned" in L2, reducing HBM traffic during the decode phase where the same weights are hit once per token per user.

**Worked Example 2.5-B — L2 Persistence for Decode-Phase Weight Reuse**

```
Setup: Llama 3 8B, decode phase (batch=32 users, 1 new token each)

Q projection: W_Q shape [4096, 4096], dtype BF16
  Size = 4096 × 4096 × 2 bytes = 33.5 MB

Without L2 persistence:
  Each decode step fetches W_Q from HBM: 33.5 MB × 2 TB/s^-1 = 16.8 µs per layer
  (32 users share one fetch — the same matrix is loaded once per step)

With L2 persistence (A100 L2 = 40 MB):
  First step: fetch W_Q from HBM (cold miss): 16.8 µs
  Subsequent steps: W_Q hits in L2 → latency ≈ 0.5 µs (30× faster)
  
  Condition: W_Q (33.5 MB) + K,V projections (33.5 MB each = 67 MB total)
  > A100 L2 (40 MB) → cannot fit all three, but W_Q alone fits with room
  → Persist W_Q, let W_K and W_V be streamed from HBM as usual

Practical speedup on decode step: 15–25% reduction in per-token latency
vLLM implementation: not used by default (set via engine flags), but measurable
```

---

## 2.5.5 Global Memory — Coalescing and Access Patterns

`[CORE]`

### What Global Memory Is

Global memory is the HBM attached to the GPU — the 80 GB on an A100, the 80–192 GB on an H100. Everything persistent lives here: model weights, KV cache, activations, intermediate buffers. All SMs share the same global memory address space. It has the highest capacity and the lowest per-access bandwidth of any GPU memory level.

### Coalesced vs Uncoalesced Access — The 30× Gap

When a warp (32 threads) issues a load from global memory, the hardware tries to serve all 32 loads in as few transactions as possible. The cache line is 128 bytes. If all 32 threads access 4-byte floats in consecutive addresses, the hardware can serve all 32 threads with **a single 128-byte transaction**. This is called a **fully coalesced access**.

If the 32 threads access scattered, non-consecutive addresses, the hardware must issue **up to 32 separate transactions**. This is fully uncoalesced, and reduces effective bandwidth by up to 32×.

```
COALESCED ACCESS (optimal):

  Thread 0 → address base + 0
  Thread 1 → address base + 4
  Thread 2 → address base + 8
  ...
  Thread 31 → address base + 124

  Hardware issues: 1 × 128-byte transaction
  Effective bandwidth: 100% of peak HBM bandwidth

─────────────────────────────────────────────────────

UNCOALESCED ACCESS (strided by 128 bytes):

  Thread 0 → address base + 0
  Thread 1 → address base + 128
  Thread 2 → address base + 256
  ...
  Thread 31 → address base + 31×128

  Hardware issues: 32 × 128-byte transactions
  Effective bandwidth: ~3% of peak HBM bandwidth (32× penalty)

─────────────────────────────────────────────────────

PARTIALLY COALESCED (stride of 2 floats = 8 bytes):

  Thread 0 → address base + 0
  Thread 1 → address base + 8
  Thread 2 → address base + 16
  ...
  Thread 31 → address base + 248

  Spans 256 bytes → 2 × 128-byte transactions
  Effective bandwidth: 50% of peak (2× penalty)
```

**Worked Example 2.5-C — Row-Major vs Column-Major Matrix Access**

```
Matrix A: shape [M, N] = [4096, 4096], stored row-major in global memory
  Row i starts at: A + i × N × sizeof(float)

Kernel pattern 1 — each thread reads A[row][threadIdx.x]:
  Thread 0 reads A[row][0]  → address A + row×N×4 + 0
  Thread 1 reads A[row][1]  → address A + row×N×4 + 4
  ...
  Thread 31 reads A[row][31] → address A + row×N×4 + 124
  → Consecutive addresses → COALESCED → 1 transaction

Kernel pattern 2 — each thread reads A[threadIdx.x][col]:
  Thread 0 reads A[0][col]  → address A + 0×N×4 + col×4
  Thread 1 reads A[1][col]  → address A + 1×N×4 + col×4  = prev + N×4 = prev + 16384
  Thread 2 reads A[2][col]  → prev + 16384
  ...
  Stride = N × 4 = 16384 bytes between consecutive threads
  → 32 separate cache lines → UNCOALESCED → 32 transactions

Performance difference on A100:
  Coalesced:   ~2 TB/s effective bandwidth (peak)
  Uncoalesced: ~60 GB/s effective bandwidth (32× penalty)

Fix: transpose A into shared memory first (coalesced load from global),
then read from SMEM with the column pattern (0-conflict with padding).
This is exactly what the tiled GEMM algorithm does.
```

### The Tiled GEMM Access Pattern — Putting It All Together

This is the canonical example that demonstrates every memory space working together:

```cuda
#define TILE 16

__global__ void tiled_gemm(float* A, float* B, float* C, int N) {
    // SMEM: each block stages a TILE×TILE tile of A and B
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // REGISTERS: accumulator for this thread's output element
    float acc = 0.0f;

    for (int k = 0; k < N; k += TILE) {
        // GLOBAL → SMEM: coalesced loads (consecutive threads, consecutive addresses)
        sA[threadIdx.y][threadIdx.x] = A[row * N + (k + threadIdx.x)];
        sB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        __syncthreads();  // wait for all threads to finish loading

        // SMEM → REGISTER: compute dot product, SMEM reads, no bank conflicts
        for (int i = 0; i < TILE; i++)
            acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();  // wait before overwriting sA/sB next iteration
    }

    // REGISTER → GLOBAL: coalesced store
    C[row * N + col] = acc;
}
```

```
Memory access count for C = A @ B, both [N, N], N=4096, TILE=16:

Without tiling (naive):
  Each output C[i][j] requires N=4096 reads from A and N from B
  Total global reads: N³ = 4096³ = 68 billion
  Each is a separate uncoalesced access → effective BW < 5% peak

With tiling (TILE=16):
  Each tile is loaded once from global memory → N³ / TILE² fewer total loads
  Coalesced loads → 1 transaction per row of tile → full bandwidth
  SMEM reuse: each element loaded once, used TILE=16 times
  Arithmetic intensity: (2×TILE³ FLOPs) / (2×TILE²×sizeof(float) bytes) = TILE/2 = 8 FLOP/byte
  A100 ridge point: 312 TFLOP/s ÷ 2 TB/s = 156 FLOP/byte
  → At TILE=16, still memory-bound (8 < 156), but much better than naïve
  → CUTLASS uses TILE=128×128×32 → 64 FLOP/byte, approaching ridge point
```

---

## 2.5.6 Constant Memory — Broadcast-Optimized Read-Only Storage

`[SUPPLEMENTARY]`

### What Constant Memory Is

Constant memory is a 64 KB region of global memory that is backed by a dedicated per-SM cache (the constant cache, ~8 KB). It is declared with `__constant__` and must be written from the host before the kernel launches. Inside the kernel it is read-only.

The key property: if all threads in a warp read the **same address** from constant memory, the hardware serves it as a single broadcast — one cache lookup, 32 threads satisfied simultaneously. This is called a broadcast read.

If different threads read **different addresses** from constant memory, the accesses are serialized (up to 32 sequential lookups), which is worse than a regular L1 cache miss.

```cuda
// Declare at file scope (in constant memory, 64 KB limit)
__constant__ float scale_factors[1024];  // 4 KB
__constant__ int config[64];             // 256 bytes

// Initialize from host before kernel launch
cudaMemcpyToSymbol(scale_factors, host_scales, sizeof(host_scales));

// Use in kernel — all threads read same scale: 1 broadcast, no serialization
__global__ void scale_kernel(float* data, int layer_idx) {
    float scale = scale_factors[layer_idx];  // all threads read same index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= scale;
}
```

### LLM Inference Use Cases for Constant Memory

| Use Case | Fits in 64 KB? | Benefit |
|---|---|---|
| Quantization scale factors (per-channel INT8, N≤16384 channels) | Yes (64 KB) | Broadcast per output row |
| RoPE frequency table (dim=128, θ=500K) | Yes (2 KB) | All heads read same freqs |
| Temperature / top-p / top-k sampling config | Yes (tiny) | Single broadcast per kernel |
| Attention mask pattern (fixed causal) | Yes for short seqs | Broadcast per row |
| Full weight matrices | No (far too large) | Use global memory |

**Worked Example 2.5-D — RoPE Frequencies in Constant Memory**

```
RoPE embedding for Llama 3 8B:
  head_dim = 128
  theta = 500,000
  freq[i] = 1 / theta^(2i/128)  for i = 0..63  (64 values, 256 bytes)

Storage in constant memory:
  __constant__ float rope_freqs[64];  // 256 bytes, well within 64 KB limit

In the attention kernel, every query and key head reads rope_freqs[i]
for the same i — all threads in a warp read the same address → broadcast.

Without constant memory (global memory load):
  64 floats × 32 threads × 2 (Q and K) × num_heads accesses
  = many redundant global loads, partially cached by L1

With constant memory:
  1 broadcast read per unique freq index
  Saves ~32× on the freq fetch bandwidth
  Latency: ~5 cycles (constant cache hit) vs ~400 cycles (global miss)
```

---

## 2.5.7 Texture Memory — 2D Spatial Locality Cache

`[SUPPLEMENTARY]`

### What Texture Memory Is

Texture memory is global memory accessed through a separate read-only texture cache optimized for **2D spatial locality** — the assumption that if you read element (x, y), you will soon also read (x+1, y), (x, y+1), (x+1, y+1). This is the access pattern of convolution and image processing.

For LLM inference, texture memory is less commonly used than constant or shared memory. It surfaces mainly in:

- **2D attention score matrices**: the softmax and output projection access patterns exhibit some 2D locality
- **Legacy code**: some llama.cpp CUDA backends use texture fetches for weight access
- **Quantized weight lookups**: 2D lookup tables for quantization codebooks

The `__ldg()` intrinsic (load via texture cache) can speed up read-only global loads on non-texture data:

```cuda
// Regular global load (goes through L1)
float val = A[idx];

// Texture cache load (read-only path, may be faster for scattered reads)
float val = __ldg(&A[idx]);
```

In practice, `__restrict__` + `const` qualifiers let the compiler use the read-only cache path automatically. Explicit texture objects (`cudaTextureObject_t`) are rarely needed in modern CUDA code for LLM inference.

---

## 2.5.8 Local Memory — The Compiler's Safety Valve

`[SUPPLEMENTARY]`

### What Local Memory Is

Local memory is not a distinct hardware memory space — it is a per-thread region allocated in global memory (HBM) by the compiler. It exists to handle:

1. **Register spilling**: when a kernel uses more registers than are available per thread, the excess is spilled to local memory
2. **Large per-thread arrays**: `float arr[1000]` in a kernel cannot fit in registers; the compiler allocates it in local memory
3. **Recursion**: stack frames for device functions

The name is misleading: "local" refers to thread-local scope, not locality of access. Local memory has the same ~400–800 cycle latency as global memory.

```cuda
// This will almost certainly end up in local memory (too large for registers):
__global__ void bad_kernel() {
    float big_array[512];  // 2 KB per thread → 2 MB for 1024-thread block
    // ...
}

// This stays in registers (small enough):
__global__ void good_kernel() {
    float acc[4];  // 16 bytes per thread → stays in registers
    // ...
}
```

### Detecting Local Memory Spills

```bash
nvcc --ptxas-options=-v mykernel.cu 2>&1 | grep "lmem"
# Output: ptxas info: Function properties for my_kernel:
#   0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
# If spill bytes > 0, register pressure is degrading performance
```

---

## 2.5.9 Occupancy — How Memory Limits Parallelism

`[CORE]`

### The Occupancy Concept

**Occupancy** is the ratio of active warps to the maximum possible warps on an SM. High occupancy means the GPU has many warps ready to execute, which allows it to hide memory latency by switching between warps while some wait for memory.

Maximum warps per SM on Ampere: 64 warps = 2,048 threads.

Occupancy is limited by three resources, whichever is most constraining:

1. **Registers per thread** (fewer registers → more threads)
2. **Shared memory per block** (less SMEM per block → more blocks per SM)
3. **Threads per block** (a block of 64 threads wastes SM capacity)

**Worked Example 2.5-E — Occupancy Calculation**

```
Kernel configuration:
  Block size: 256 threads (8 warps)
  Registers per thread: 64
  Shared memory per block: 32 KB

Constraints on A100 SM:
  Constraint 1 — Registers:
    SM register file: 65,536 registers
    Registers per block: 256 × 64 = 16,384
    Max blocks from registers: 65,536 / 16,384 = 4 blocks
    Active threads: 4 × 256 = 1,024 threads = 32 warps

  Constraint 2 — Shared Memory:
    SM SMEM: 100 KB (default Ampere)
    SMEM per block: 32 KB
    Max blocks from SMEM: 100 / 32 = 3 blocks (floor)
    Active threads: 3 × 256 = 768 threads = 24 warps

  Constraint 3 — Max blocks per SM (hardware limit: 32 on Ampere):
    4 blocks < 32 → not binding

  Binding constraint: Shared Memory (24 warps, not registers at 32)
  Occupancy = 24 / 64 = 37.5%

To improve: reduce SMEM per block to 24 KB → 4 blocks → registers bind at 32 warps → 50%
         or reduce SMEM to 20 KB → 5 blocks → registers still bind at 4 blocks → 50%
         or reduce registers to 48 → 5 blocks from registers → SMEM binds at 3 → no change
         Best fix: reduce SMEM to 16 KB → 6 blocks (from SMEM), 4 (from regs) → regs bind → 50%
```

Use the CUDA Occupancy Calculator:

```python
import math

def occupancy(block_size, regs_per_thread, smem_per_block_kb,
              max_regs_per_sm=65536, max_smem_kb=100,
              max_threads_per_sm=2048, max_blocks_per_sm=32):
    warps_per_block = math.ceil(block_size / 32)
    
    blocks_from_regs  = max_regs_per_sm   // (block_size * regs_per_thread)
    blocks_from_smem  = int(max_smem_kb   // smem_per_block_kb) if smem_per_block_kb > 0 else max_blocks_per_sm
    blocks_from_thds  = max_threads_per_sm // block_size
    
    active_blocks = min(blocks_from_regs, blocks_from_smem,
                        blocks_from_thds, max_blocks_per_sm)
    active_warps  = active_blocks * warps_per_block
    max_warps     = max_threads_per_sm // 32
    
    return active_warps / max_warps, active_blocks, {
        "registers": blocks_from_regs,
        "smem":      blocks_from_smem,
        "threads":   blocks_from_thds,
    }

occ, blocks, limits = occupancy(256, 64, 32)
print(f"Occupancy: {occ:.1%}, Active blocks: {blocks}")
print(f"Limiting resource: {min(limits, key=limits.get)}")
# Output: Occupancy: 37.5%, Active blocks: 3
# Limiting resource: smem
```

---

## 2.5.10 How LLM Kernels Use Each Memory Space

`[SYNTHESIS]`

This section maps each major LLM inference operation to the memory spaces it uses. Understanding this mapping lets you reason about why specific kernels have the performance characteristics they do.

```
  OPERATION              │ REGISTERS   │ SMEM        │ L1/L2    │ GLOBAL (HBM)
  ───────────────────────┼─────────────┼─────────────┼──────────┼─────────────
  GEMM (weight multiply) │ Accumulator │ A/B tiles   │ overflow │ W, X, Y
                         │ tile frags  │ double-buf  │          │ (coalesced)
  ───────────────────────┼─────────────┼─────────────┼──────────┼─────────────
  Attention scores       │ QK dot      │ Q tile      │ K reuse  │ Q, K full
  (Q @ K^T)              │ partial sums│ (FlashAttn) │          │ matrix
  ───────────────────────┼─────────────┼─────────────┼──────────┼─────────────
  FlashAttention         │ m, l stats  │ Q/K/V tile  │ partial  │ Q, K, V
  (fused forward)        │ acc O tile  │ O tile      │ miss     │ (tiled fetch)
  ───────────────────────┼─────────────┼─────────────┼──────────┼─────────────
  Softmax (fused)        │ max, sum    │ score tile  │ —        │ none if fused
                         │ exp values  │ (if large)  │          │ with attn
  ───────────────────────┼─────────────┼─────────────┼──────────┼─────────────
  LayerNorm              │ mean, var   │ partial sums│ L2 reuse │ X (2 passes)
                         │ accumulators│ (for reduce)│          │ or 1 with SMEM
  ───────────────────────┼─────────────┼─────────────┼──────────┼─────────────
  RoPE (positional emb)  │ cos, sin    │ —           │ freq tbl │ Q, K in/out
                         │ rotated q,k │             │ L1 hits  │ (read-modify)
  ───────────────────────┼─────────────┼─────────────┼──────────┼─────────────
  PagedAttention         │ QK dot, acc │ Q tile      │ block    │ K/V blocks
  (vLLM decode)          │ softmax stats│            │ headers  │ (scattered!)
  ───────────────────────┼─────────────┼─────────────┼──────────┼─────────────
  INT8 GEMM (W8A8)       │ INT32 acc   │ INT8 tiles  │ scale    │ INT8 W, A
                         │ dequant     │ double-buf  │ factors  │ (2× smaller)
  ───────────────────────┼─────────────┼─────────────┼──────────┼─────────────
  Sampling (top-k/top-p) │ running max │ warp-level  │ L2 logit │ logit vector
                         │ candidates  │ reduce buf  │ reuse    │ (one load)
```

### PagedAttention and Non-Coalesced Access

PagedAttention (Chapter 6) stores KV cache in fixed-size blocks that may be **non-contiguous** in global memory. During decode, each query must gather K and V blocks from potentially scattered addresses. This access pattern is inherently less coalesced than the sequential GEMM access pattern, which is one reason FlashDecoding and other optimizations were developed to improve decode-phase attention efficiency.

The memory-level implication: PagedAttention trades coalesced bandwidth efficiency for flexible memory management. The block table (mapping logical KV positions to physical block addresses) adds an indirect load per KV access, adding latency on top of the non-coalesced scatter-gather.

---

## 2.5.11 Worked Arithmetic: Memory Bandwidth Budget for a Decode Step

`[QUANTITATIVE]`

Let us compute exactly how much of the GPU's HBM bandwidth is consumed by a single decode step on Llama 3 8B (BF16 weights, batch=1).

```
Llama 3 8B architecture:
  Layers L = 32
  Model dimension d = 4096
  FFN intermediate = 14336
  Num heads H = 32, head_dim = 128
  Num KV heads G = 8 (GQA)
  Sequence length so far: seq_len = 512 tokens (KV cache content)
  Data type: BF16 (2 bytes per element)

Per-layer decode operations and their HBM traffic:

  1. Attention QKV projection:
     Q: W_Q [4096, 4096] → read 4096×4096×2 = 33.6 MB
     K: W_K [4096, 1024] → read 4096×1024×2 = 8.4 MB  (GQA: 8 heads)
     V: W_V [4096, 1024] → read 4096×1024×2 = 8.4 MB
     Subtotal: 50.4 MB per layer

  2. KV cache read (512 tokens, GQA 8 heads, head_dim 128):
     K cache: 512 × 8 × 128 × 2 = 1.05 MB per layer
     V cache: 512 × 8 × 128 × 2 = 1.05 MB per layer
     Subtotal: 2.1 MB per layer

  3. Attention output projection:
     W_O [4096, 4096] → read 33.6 MB per layer

  4. FFN (SwiGLU, two gate matrices + down):
     W_gate [4096, 14336] → read 4096×14336×2 = 117.9 MB
     W_up   [4096, 14336] → read 117.9 MB
     W_down [14336, 4096] → read 117.9 MB
     Subtotal: 353.7 MB per layer

  5. LayerNorm parameters (negligible): ~32 KB per layer

Per-layer total HBM reads:
  50.4 + 2.1 + 33.6 + 353.7 = 439.8 MB per layer

Total for 32 layers:
  439.8 × 32 = 14,074 MB ≈ 14.1 GB per decode step

Time on A100 (2 TB/s peak, assume 80% effective):
  14.1 GB / (2,000 × 0.80 GB/s) = 8.8 ms per token

Observed decode speed: ~8–12 ms/token on A100 for Llama 3 8B
(matches — decode is almost entirely bandwidth-bound, not compute-bound)

Arithmetic intensity:
  FLOPs per decode: ~2 × params × batch = 2 × 8B × 1 = 16 GFLOP
  Bytes read: 14.1 GB
  AI = 16 / 14.1 = 1.1 FLOP/byte  (far below A100 ridge point of 156 FLOP/byte)
  → SEVERELY memory-bandwidth-bound — this is why more users = better utilization
```

This calculation explains why inference throughput scales almost linearly with batch size up to the compute ridge point: adding users does not proportionally increase HBM traffic (weights are loaded once per layer, shared across all users), but it does proportionally increase FLOPs, pushing AI toward the ridge point.

---

## 2.5.12 SMEM-Optimized Attention — The FlashAttention Memory Layout

`[SYNTHESIS]`

FlashAttention is the most important application of on-chip memory optimization in LLM inference. Let us trace exactly which memory space each value lives in during the fused kernel.

```
FlashAttention forward pass memory layout (one block, one head):

  Input tensors (global memory, HBM):
    Q: [seq_len, head_dim]  loaded tile by tile → SMEM
    K: [seq_len, head_dim]  loaded tile by tile → SMEM
    V: [seq_len, head_dim]  loaded tile by tile → SMEM
    O: [seq_len, head_dim]  written back at end

  SMEM allocation per block (TILE_Q rows of Q, TILE_KV rows of K and V):
    Q_tile: TILE_Q × head_dim × 2 bytes
    K_tile: TILE_KV × head_dim × 2 bytes
    V_tile: TILE_KV × head_dim × 2 bytes

  Example: TILE_Q = 64, TILE_KV = 64, head_dim = 128, BF16:
    Q_tile: 64 × 128 × 2 = 16 KB
    K_tile: 64 × 128 × 2 = 16 KB
    V_tile: 64 × 128 × 2 = 16 KB
    Total SMEM: 48 KB per block (fits in 160 KB max SMEM on Ampere)

  Register allocation per thread:
    O_acc:  head_dim floats = 128 × 4 = 512 bytes  ← accumulator for output row
    m:      1 float (running maximum for softmax)
    l:      1 float (running sum of exp for normalization)
    scores: TILE_KV floats = 64 × 4 = 256 bytes (score row for this Q row)
    Total: ~800 bytes per thread → ~100 registers per thread (near but within limit)

  The outer loop (over K/V tiles) in pseudo-code:
    for kv_tile in range(0, seq_len, TILE_KV):
      # Global → SMEM (coalesced loads, async on Ampere)
      load K_tile from K[kv_tile : kv_tile+TILE_KV]  # global → SMEM
      load V_tile from V[kv_tile : kv_tile+TILE_KV]  # global → SMEM
      sync()

      # SMEM → Registers (compute QK^T, update running softmax)
      for q_row in range(TILE_Q):
        scores[q_row] = Q_tile[q_row] @ K_tile^T  # SMEM × SMEM → registers
        m_new = max(m, max(scores[q_row]))
        l = exp(m - m_new) * l + sum(exp(scores[q_row] - m_new))
        O_acc[q_row] = exp(m - m_new) * O_acc[q_row] + exp(scores-m_new) @ V_tile
        m = m_new

    # Final normalization (registers only)
    O_acc /= l

    # Registers → Global (coalesced store)
    store O_acc to O[q_tile : q_tile+TILE_Q]

HBM traffic:
  Standard attention (seq_len = 2048, head_dim = 128):
    Reads Q, K: 2 × 2048 × 128 × 2 = 1.05 MB
    Reads/writes attention matrix S: 2048 × 2048 × 4 = 16.8 MB  ← THE EXPENSIVE PART
    Reads V: 1.05 MB
    Total: ~19 MB per head

  FlashAttention:
    Reads Q, K, V: 3 × 1.05 MB = 3.15 MB per head
    Writes O: 1.05 MB
    ZERO reads/writes of the 2048×2048 attention matrix (it lives in SMEM+registers)
    Total: ~4.2 MB per head  (4.5× reduction in HBM traffic)

At seq_len = 8192: standard = ~268 MB / head vs FlashAttention = ~16 MB / head (16× reduction)
At seq_len = 128K: standard = 64 GB / head → impossible; FlashAttention = ~50 MB / head → feasible
```

This is why FlashAttention is not merely a performance optimization — it is what makes long-context inference possible at all. Without it, the O(seq_len²) HBM traffic makes 128K-token contexts physically unservable on current hardware.

---

## Chapter Summary

```
MEMORY SPACE QUICK REFERENCE

  Space          Location      Latency    Key property                  LLM use
  ─────────────────────────────────────────────────────────────────────────────────
  Registers      ON-CHIP       1 cycle    Private, fastest              Accumulators,
                 (per SM)                 Compiler-managed              softmax stats
  ─────────────────────────────────────────────────────────────────────────────────
  Shared Mem     ON-CHIP       ~5 cycles  Programmable, block-shared    Q/K/V tiles,
                 (per SM)                 Bank-conflict risk            GEMM tiles
  ─────────────────────────────────────────────────────────────────────────────────
  L1 Cache       ON-CHIP       ~5 cycles  HW-managed, shares with SMEM  Frequent
                 (per SM)                __ldg() for read-only          weight access
  ─────────────────────────────────────────────────────────────────────────────────
  L2 Cache       ON-CHIP       ~30 cycles Chip-wide, persistent hint    Hot weights,
                 (per GPU)               cudaAccessPropertyPersisting   decode phase
  ─────────────────────────────────────────────────────────────────────────────────
  Constant Mem   ON-CHIP cache ~5 cycles  Broadcast-optimized, RO       Scale factors,
                 (logical: HBM)(cached)  Per-SM constant cache          RoPE freqs
  ─────────────────────────────────────────────────────────────────────────────────
  Global Mem     NEAR-CHIP     ~400 cyc   Coalescing critical (128 B)   All weights,
  (HBM)          (interposer)            DRAM, not SRAM — avoid trips  KV cache
  ─────────────────────────────────────────────────────────────────────────────────
  Local Mem      NEAR-CHIP     ~400 cyc   Register spill → HBM          Avoid entirely
                 (spills→HBM)            nvcc -ptxas to detect
```

**The three rules that govern GPU memory performance in LLM inference:**

1. **Keep hot data on-chip.** Registers for running state (softmax m/l in FlashAttention). SMEM for tiles reused across threads. L2 persistence for weights in short decode loops.

2. **Coalesce every global memory access.** Non-coalesced access pays a 4–32× bandwidth penalty. Row-major layouts with row-stride access patterns coalesce naturally; column or strided patterns require transposition through SMEM.

3. **Match SMEM tile size to occupancy constraints.** Larger SMEM tiles improve arithmetic intensity (more reuse per byte fetched) but reduce occupancy (fewer blocks per SM). The optimum is hardware-specific: CUTLASS and FlashAttention solve this via autotuning.

---

## Self-Check Questions

**Q1.** An Ampere SM has 192 KB of unified SMEM+L1. A kernel declares 48 KB of `__shared__` memory. How much L1 remains? If the kernel uses 48 registers per thread with a block size of 256, and the SM register file is 65,536 registers, how many blocks can run concurrently, and what is the occupancy?

**Q2.** A kernel accesses a 2D array `float A[4096][4096]` stored row-major. Thread `i` in a warp reads `A[i][col]` where `col` is fixed. Is this access coalesced? Why or why not? Describe the one change that would make it coalesced.

**Q3.** Explain why adding one padding column (`__shared__ float tile[32][33]` instead of `tile[32][32]`) eliminates bank conflicts when threads access tile column by column.

**Q4.** FlashAttention keeps the softmax running statistics `m` and `l` in registers across the K/V tile loop. What would happen to HBM traffic if these values were stored in global memory instead? Estimate the additional bytes for seq_len = 4096, head_dim = 128, 32 heads, BF16.

**Q5.** A decode step on Llama 3 8B (BF16, batch=1) reads approximately 14.1 GB from HBM and performs approximately 16 GFLOP. If you increase the batch size to 32, the weight reads remain the same (weights are loaded once regardless of batch size) but the KV cache reads scale with batch and the FLOPs scale with batch. Estimate the new arithmetic intensity and whether the operation becomes more or less memory-bandwidth-bound.

---

## Worked Solutions

### Solution 1 — SMEM/L1 split and occupancy

```
SMEM declared: 48 KB
L1 remaining: 192 - 48 = 144 KB

Blocks from SMEM: floor(192 / 48) = 4 blocks (but wait — 4 × 48 = 192, exactly fits)
                  Actually: 192 KB total, 48 KB per block → 4 blocks

Blocks from registers:
  Registers per block = 256 threads × 48 regs = 12,288
  SM total: 65,536
  Blocks: floor(65,536 / 12,288) = 5 blocks

Blocks from thread count:
  Max threads per SM = 2,048
  Threads per block = 256
  Blocks: 2,048 / 256 = 8 blocks

Hardware max blocks per SM (Ampere): 32

Binding constraint: SMEM → 4 blocks
Active warps: 4 × (256 / 32) = 4 × 8 = 32 warps
Max warps per SM: 2,048 / 32 = 64 warps
Occupancy: 32 / 64 = 50%
```

### Solution 2 — Column access coalescing

```
Access: thread i reads A[i][col] where col is fixed, i = threadIdx.x

Address of A[i][col] = base + (i × 4096 + col) × sizeof(float)
                     = base + i × 4096 × 4 + col × 4
                     = base + i × 16384 + col × 4

Stride between consecutive threads: 16,384 bytes (one full row = 4096 × 4 bytes)

The 32 threads span: 0 to 31 × 16,384 = 507,904 bytes
Cache line = 128 bytes → this spans ~3968 cache lines → 32 transactions (uncoalesced)

Fix: Change the access to thread i reads A[row][i] — row varies across the block,
col varies across the warp:
  Address = base + (row × 4096 + i) × 4
  Thread 0: base + row×16384 + 0
  Thread 1: base + row×16384 + 4
  ...stride = 4 bytes → consecutive → 1 transaction (fully coalesced)

Or: load the column into shared memory first (coalesced row load), then
access SMEM column-by-column (no penalty on SMEM for column access if padded).
```

### Solution 3 — Padding and bank conflicts

```
__shared__ float tile[32][33]  (33-column layout)

Thread i accesses tile[i][0] (column 0 of each row):
  offset of tile[i][0] = i × 33 × sizeof(float) = i × 132 bytes

Bank of address at byte offset b = (b / 4) % 32   (4 bytes per bank word)

Thread 0: offset = 0,   bank = (0/4) % 32 = 0
Thread 1: offset = 132, bank = (132/4) % 32 = 33 % 32 = 1
Thread 2: offset = 264, bank = (264/4) % 32 = 66 % 32 = 2
Thread 3: offset = 396, bank = (396/4) % 32 = 99 % 32 = 3
...
Thread k: bank = (k × 33) % 32 = (k × 33 mod 32)

Since gcd(33, 32) = 1 (33 and 32 are coprime), the sequence k × 33 mod 32
cycles through all 32 residues as k goes from 0 to 31.
→ All 32 threads land on different banks → 0 bank conflicts.

Without padding (tile[32][32]):
Thread k: offset = k × 32 × 4 = k × 128, bank = (k × 128 / 4) % 32 = k × 32 % 32 = 0
→ ALL threads land on bank 0 → 32-way conflict.
```

### Solution 4 — FlashAttention HBM traffic without in-register stats

```
Without in-register m and l, after each K/V tile we need to:
  1. Write m and l to global memory per query head
  2. Read m and l back for the next tile

Number of K/V tiles: ceil(seq_len / TILE_KV) = ceil(4096 / 64) = 64 tiles
Number of Q heads: 32
Elements per tile per head: 2 (m and l, one float each = 4 bytes each)

Additional reads/writes per head:
  Per tile: 2 writes + 2 reads = 4 floats = 16 bytes
  Total tiles: 64
  Total per head: 64 × 16 = 1,024 bytes ≈ 1 KB per head

Total additional HBM traffic for 32 heads:
  32 × 1,024 = 32,768 bytes = 32 KB

This seems small, but the real cost is the kernel split: without in-register stats,
you need two passes — one to compute max (for numerically stable softmax) and one
to compute exp and sum. The two-pass approach doubles the reads of K and V:
  K reads: 2 × 32 × 4096 × 128 × 2 = 67.1 MB (vs 33.6 MB with one pass)
  V reads: same, 67.1 MB

Total additional traffic from two-pass approach: 67.1 MB extra
This is the dominant cost — not the m and l values themselves, but the
second pass over K and V. FlashAttention's one-pass online softmax eliminates this.
```

### Solution 5 — Batch=32 arithmetic intensity

```
Batch = 1 (baseline):
  Weight reads:   ~14.0 GB (all 32 layers, QKV + FFN weights)
  KV cache reads: 32 × 2.1 MB = 67.2 MB (negligible)
  Total bytes:    ~14.07 GB
  FLOPs:          ~16 GFLOP
  AI:             16 / 14.07 = 1.14 FLOP/byte

Batch = 32:
  Weight reads:   ~14.0 GB (SAME — weights loaded once per decode step, shared)
  KV cache reads: 32 × 32 × 2.1 MB = 2.15 GB  (32 users × 32 layers × 2.1 MB)
  Total bytes:    14.0 + 2.15 = 16.15 GB
  FLOPs:          ~16 GFLOP × 32 = 512 GFLOP  (scales with batch)
  AI:             512 / 16.15 = 31.7 FLOP/byte

Batch = 32 vs Batch = 1:
  AI increased from 1.14 to 31.7 FLOP/byte (28× improvement)
  Still below A100 ridge point (156 FLOP/byte) → still memory-bandwidth-bound
  But GPU utilization is much higher (31.7/156 = 20% vs 0.7%)

The AI at which decode becomes compute-bound:
  156 FLOP/byte × 16.15 GB = 2,519 GFLOP
  Batch needed: 2,519 / 16 ≈ 157 concurrent users
  At batch=157, the A100 is fully utilized for Llama 3 8B BF16 decode.
  Above that, throughput stops increasing (compute-bound ceiling).
  vLLM uses continuous batching to approach this ceiling.
```

---

## Where We Go Next

Chapter 3 builds on this foundation to explain tokens, sequences, and the batch — how the memory pressure analyzed here scales when serving many users simultaneously, and how the batch size determines whether the GPU is memory-bandwidth-bound or compute-bound. Chapter 5 returns to FlashAttention in full algorithmic detail, with the tiling strategy derived from first principles using exactly the SMEM and register constraints mapped in this chapter.


---

## Companion Code — `tiled_gemm_harness.cu`

The full test harness for every kernel in this chapter is in the companion code section. It includes:

- **Correctness check** — all five kernels verified against a single-threaded CPU reference using `max_abs_error`
- **Performance benchmark** — GFLOP/s, % of FP32 peak, and wall-clock ms per kernel across 20 runs
- **Coalescing penalty** — isolated pure-bandwidth kernels measuring the full 10–32× gap without arithmetic noise
- **Occupancy report** — `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for each kernel showing which resource is the binding constraint
- **Arithmetic intensity sweep** — TILE = 4 through 128, showing how AI grows toward (but stays below) the ridge point for FP32
- **Bank conflict timing** — direct comparison of `tiled_gemm<32>` vs `tiled_padded<32>` at N=4096

```bash
# Compile
nvcc -O3 -arch=sm_80 -std=c++17 -lineinfo \
     -o tiled_gemm_harness tiled_gemm_harness.cu

# Run all sections
./tiled_gemm_harness

# Profile shared memory bank conflicts with Nsight Compute
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  ./tiled_gemm_harness
```

→ [View full harness source](../code/chapter_02b.md)
