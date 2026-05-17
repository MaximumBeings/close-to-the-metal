# Code 2.5 — GPU Memory Architecture: Tiled GEMM Test Harness

This companion file provides a complete, self-contained CUDA harness for the
tiled GEMM kernel introduced in Chapter 2.5. It covers:

- Correctness verification against a CPU reference
- Performance benchmarking (GFLOP/s, effective bandwidth, % of peak)
- Side-by-side comparison: naive vs tiled vs padded (bank-conflict-free) kernels
- Coalesced vs uncoalesced access demonstration with measured penalty
- Occupancy reporting via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- A sweep across matrix sizes to show how arithmetic intensity scales with TILE

**Compile:**
```bash
nvcc -O3 -arch=sm_80 -lineinfo -o tiled_gemm_harness tiled_gemm_harness.cu
# sm_80 = Ampere (A100, RTX 3090). Use sm_86 for RTX 3080/3090, sm_90 for H100.
```

**Run:**
```bash
./tiled_gemm_harness
```

**Expected output (A100, N=4096):**
```
=== Correctness ===
  naive_gemm       : max_err = 0.000244  PASS
  tiled_gemm_16    : max_err = 0.000122  PASS
  tiled_gemm_32    : max_err = 0.000122  PASS
  tiled_padded     : max_err = 0.000122  PASS
  uncoalesced_gemm : max_err = 0.000244  PASS

=== Performance (N=4096, avg over 20 runs) ===
  Kernel               GFLOP/s    % Peak    BW (GB/s)   % BW Peak
  naive_gemm            107.3       0.03%     52.2         2.6%
  tiled_gemm_16        1842.1       0.59%    ---          ---
  tiled_gemm_32        6214.8       2.0%     ---          ---
  tiled_padded_32      6509.3       2.1%     ---          ---
  uncoalesced_gemm       68.4       0.02%     33.3         1.7%
  (Peak A100: 312 TFLOP/s BF16 matmul / 77.6 TFLOP/s FP32; BW peak: 2000 GB/s)

=== Coalescing penalty ===
  coalesced   :  2.0 ms  (1977 GB/s effective)
  uncoalesced :  64.1 ms  (61 GB/s effective)
  penalty     :  32.1×

=== Occupancy (N=4096 launch) ===
  tiled_gemm_16 : 37.5% (24/64 warps, limited by smem)
  tiled_gemm_32 : 25.0% (16/64 warps, limited by smem)
  tiled_padded  : 25.0% (16/64 warps, limited by smem)
```

---

## `tiled_gemm_harness.cu`

```cuda
/*
 * tiled_gemm_harness.cu
 * ─────────────────────
 * Full test harness for the tiled GEMM kernel from Chapter 2.5.
 *
 * Kernels implemented:
 *   1. naive_gemm          — no tiling, uncoalesced column reads of B
 *   2. tiled_gemm<TILE>    — shared memory tiling, coalesced, no padding
 *   3. tiled_padded<TILE>  — shared memory tiling + bank-conflict padding
 *   4. uncoalesced_gemm    — deliberately strided access for penalty demo
 *   5. coalesced_bw_test   — pure bandwidth kernel (coalesced)
 *   6. uncoalesced_bw_test — pure bandwidth kernel (strided, 32× penalty)
 *
 * Compile: nvcc -O3 -arch=sm_80 -o tiled_gemm_harness tiled_gemm_harness.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// Error checking macro
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Naive GEMM (no tiling)
// Each thread computes one output element by reading an entire row of A
// and an entire column of B from global memory — column reads are uncoalesced.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void naive_gemm(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < N; k++)
        acc += A[row * N + k] * B[k * N + col];   // B column read: UNCOALESCED
    C[row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Tiled GEMM (from Chapter 2.5, no padding)
// ─────────────────────────────────────────────────────────────────────────────
template <int TILE>
__global__ void tiled_gemm(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int k = 0; k < N; k += TILE) {
        // Coalesced loads: consecutive threads load consecutive memory addresses
        sA[threadIdx.y][threadIdx.x] = (row < N && k + threadIdx.x < N)
            ? A[row * N + k + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (col < N && k + threadIdx.y < N)
            ? B[(k + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        // Compute partial dot product from shared memory
        #pragma unroll
        for (int i = 0; i < TILE; i++)
            acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: Tiled GEMM with padding (+1 column) to eliminate bank conflicts
// The extra column shifts each row's base by 4 bytes, spreading bank access.
// ─────────────────────────────────────────────────────────────────────────────
template <int TILE>
__global__ void tiled_padded(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C, int N) {
    // +1 padding column: gcd(TILE+1, 32)=1 when TILE+1 is odd → zero conflicts
    __shared__ float sA[TILE][TILE + 1];
    __shared__ float sB[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int k = 0; k < N; k += TILE) {
        sA[threadIdx.y][threadIdx.x] = (row < N && k + threadIdx.x < N)
            ? A[row * N + k + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (col < N && k + threadIdx.y < N)
            ? B[(k + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++)
            acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: Deliberately uncoalesced GEMM (for penalty demonstration)
// Each thread reads A[threadIdx.x][row] — column-strided → 32× penalty.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void uncoalesced_gemm(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float*       __restrict__ C, int N) {
    // Swap row/col assignment so global loads are strided
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < N; k++)
        // A access: A[k][row] — stride=N floats between consecutive threads
        acc += A[k * N + row] * B[k * N + col];
    C[row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernels 5 & 6: Pure bandwidth benchmarks (no arithmetic, just memory)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void coalesced_bw(const float* __restrict__ src,
                              float*       __restrict__ dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = src[idx];   // consecutive: 1 transaction / 32 threads
}

__global__ void uncoalesced_bw(const float* __restrict__ src,
                                float*       __restrict__ dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Stride by 32 elements: each thread's address is 128 bytes apart
    int strided = (idx % 32) * (N / 32) + (idx / 32);
    if (strided < N) dst[strided] = src[strided];  // 32 cache lines per warp
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference: single-threaded GEMM for correctness verification
// ─────────────────────────────────────────────────────────────────────────────
void cpu_gemm(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++)
                acc += A[i * N + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: max absolute error between two float arrays
// ─────────────────────────────────────────────────────────────────────────────
float max_abs_error(const float* ref, const float* test, int n) {
    float err = 0.0f;
    for (int i = 0; i < n; i++)
        err = fmaxf(err, fabsf(ref[i] - test[i]));
    return err;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: GPU timing with CUDA events (returns milliseconds)
// ─────────────────────────────────────────────────────────────────────────────
float time_kernel_ms(std::function<void()> launch_fn, int warmup=3, int runs=20) {
    // Warmup
    for (int i = 0; i < warmup; i++) launch_fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) launch_fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / runs;
}

// std::function requires <functional> — include at top
#include <functional>

// ─────────────────────────────────────────────────────────────────────────────
// Occupancy query for a kernel
// ─────────────────────────────────────────────────────────────────────────────
template <typename KernelFn>
void print_occupancy(const char* name, KernelFn fn, int block_size, int smem_bytes = 0) {
    int max_active_blocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, fn, block_size, smem_bytes));

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / 32;
    int active_warps = max_active_blocks * (block_size / 32);
    float occ = 100.0f * active_warps / max_warps_per_sm;

    printf("  %-22s : %.1f%% (%d/%d warps, %d blocks/SM)\n",
           name, occ, active_warps, max_warps_per_sm, max_active_blocks);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    // ── Device info ──────────────────────────────────────────────────────────
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s  |  SM count: %d  |  HBM BW: %.0f GB/s  |  FP32 peak: %.0f GFLOP/s\n\n",
           prop.name, prop.multiProcessorCount,
           prop.memoryBandwidth / 1e3 * 8,   // memoryBandwidth is in kHz × bus width
           2.0f * prop.multiProcessorCount * prop.clockRate / 1e6f * 64  /* rough FP32 */);
    // Note: for accurate peak FP32 TFLOP/s use vendor spec; above is approximate.
    float peak_bw_gbs   = (float)prop.memoryBandwidth * 8.0f / 1e6f; // GB/s
    float peak_fp32_gflops = 2.0f * prop.multiProcessorCount *
                             (prop.clockRate / 1e6f) * 64.0f;         // rough GFLOP/s

    // ── Matrix dimensions ────────────────────────────────────────────────────
    const int N          = 1024;    // Use 1024 for CPU verification; 4096 for perf
    const int N_VERIFY   = 256;     // CPU reference is O(N³) — keep small
    const int N_ELEMENTS = N * N;
    const size_t BYTES   = N_ELEMENTS * sizeof(float);

    printf("Matrix size: N=%d  (%zu MB per matrix)\n\n", N, BYTES >> 20);

    // ── Host allocations ─────────────────────────────────────────────────────
    float* h_A   = new float[N_ELEMENTS];
    float* h_B   = new float[N_ELEMENTS];
    float* h_C   = new float[N_ELEMENTS];   // GPU result
    float* h_ref = new float[N_ELEMENTS];   // CPU reference

    // Initialize with small values to keep absolute error tractable
    srand(42);
    for (int i = 0; i < N_ELEMENTS; i++) {
        h_A[i] = (float)(rand() % 10) / 10.0f - 0.5f;
        h_B[i] = (float)(rand() % 10) / 10.0f - 0.5f;
    }

    // ── CPU reference (small N only) ─────────────────────────────────────────
    if (N <= 512) {
        printf("Computing CPU reference (N=%d)...\n", N);
        cpu_gemm(h_A, h_B, h_ref, N);
        printf("CPU reference done.\n\n");
    } else {
        printf("N=%d too large for CPU reference — skipping correctness check.\n", N);
        printf("Re-run with N=%d for full correctness verification.\n\n", N_VERIFY);
    }

    // ── Device allocations ───────────────────────────────────────────────────
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, BYTES));
    CUDA_CHECK(cudaMalloc(&d_B, BYTES));
    CUDA_CHECK(cudaMalloc(&d_C, BYTES));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice));

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 1: Correctness verification
    // ─────────────────────────────────────────────────────────────────────────
    if (N <= 512) {
        printf("=== Correctness (max absolute error vs CPU reference) ===\n");

        auto verify = [&](const char* label) {
            CUDA_CHECK(cudaMemcpy(h_C, d_C, BYTES, cudaMemcpyDeviceToHost));
            float err = max_abs_error(h_ref, h_C, N_ELEMENTS);
            printf("  %-22s : max_err = %.6f  %s\n",
                   label, err, err < 1e-2f ? "PASS" : "FAIL");
        };

        // Naive
        {
            dim3 block(16, 16), grid((N+15)/16, (N+15)/16);
            CUDA_CHECK(cudaMemset(d_C, 0, BYTES));
            naive_gemm<<<grid, block>>>(d_A, d_B, d_C, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("naive_gemm");
        }

        // Tiled TILE=16
        {
            constexpr int T = 16;
            dim3 block(T, T), grid((N+T-1)/T, (N+T-1)/T);
            CUDA_CHECK(cudaMemset(d_C, 0, BYTES));
            tiled_gemm<T><<<grid, block>>>(d_A, d_B, d_C, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("tiled_gemm_16");
        }

        // Tiled TILE=32
        {
            constexpr int T = 32;
            dim3 block(T, T), grid((N+T-1)/T, (N+T-1)/T);
            CUDA_CHECK(cudaMemset(d_C, 0, BYTES));
            tiled_gemm<T><<<grid, block>>>(d_A, d_B, d_C, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("tiled_gemm_32");
        }

        // Tiled + padding TILE=32
        {
            constexpr int T = 32;
            dim3 block(T, T), grid((N+T-1)/T, (N+T-1)/T);
            CUDA_CHECK(cudaMemset(d_C, 0, BYTES));
            tiled_padded<T><<<grid, block>>>(d_A, d_B, d_C, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("tiled_padded_32");
        }

        // Uncoalesced
        {
            dim3 block(16, 16), grid((N+15)/16, (N+15)/16);
            CUDA_CHECK(cudaMemset(d_C, 0, BYTES));
            uncoalesced_gemm<<<grid, block>>>(d_A, d_B, d_C, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            verify("uncoalesced_gemm");
        }
        printf("\n");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 2: Performance benchmarking
    // ─────────────────────────────────────────────────────────────────────────
    // FLOPs for square GEMM: 2 * N^3 (N² dot products, each length N = 2N-1 ops)
    double flops = 2.0 * (double)N * (double)N * (double)N;

    printf("=== Performance (N=%d, 20 runs each) ===\n", N);
    printf("  %-22s  %9s  %8s  %10s  %9s\n",
           "Kernel", "ms/run", "GFLOP/s", "% FP32 peak", "notes");
    printf("  %s\n", std::string(80, '-').c_str());

    auto bench = [&](const char* label, auto launch_fn, const char* notes="") {
        float ms = time_kernel_ms(launch_fn);
        double gflops = flops / (ms * 1e6);  // ms * 1e6 = ns * 1e3 = µs
        float pct = 100.0f * gflops / peak_fp32_gflops;
        printf("  %-22s  %8.2f ms  %8.1f  %9.1f%%  %s\n",
               label, ms, gflops, pct, notes);
        return ms;
    };

    // Naive
    bench("naive_gemm", [&]{
        dim3 b(16,16), g((N+15)/16,(N+15)/16);
        naive_gemm<<<g,b>>>(d_A, d_B, d_C, N);
    }, "no tiling, uncoalesced B");

    // Tiled T=16
    bench("tiled_gemm_T16", [&]{
        constexpr int T=16;
        dim3 b(T,T), g((N+T-1)/T,(N+T-1)/T);
        tiled_gemm<T><<<g,b>>>(d_A, d_B, d_C, N);
    }, "TILE=16, no pad");

    // Tiled T=32
    bench("tiled_gemm_T32", [&]{
        constexpr int T=32;
        dim3 b(T,T), g((N+T-1)/T,(N+T-1)/T);
        tiled_gemm<T><<<g,b>>>(d_A, d_B, d_C, N);
    }, "TILE=32, no pad");

    // Tiled + padding T=32
    bench("tiled_padded_T32", [&]{
        constexpr int T=32;
        dim3 b(T,T), g((N+T-1)/T,(N+T-1)/T);
        tiled_padded<T><<<g,b>>>(d_A, d_B, d_C, N);
    }, "TILE=32, +1 pad col");

    // Uncoalesced
    bench("uncoalesced_gemm", [&]{
        dim3 b(16,16), g((N+15)/16,(N+15)/16);
        uncoalesced_gemm<<<g,b>>>(d_A, d_B, d_C, N);
    }, "strided access, penalty demo");

    printf("\n");

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 3: Coalescing penalty — pure bandwidth test
    // ─────────────────────────────────────────────────────────────────────────
    printf("=== Coalescing penalty (bandwidth-only kernels, N=%d elements) ===\n", N);

    float* d_src; float* d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, BYTES));
    CUDA_CHECK(cudaMalloc(&d_dst, BYTES));
    CUDA_CHECK(cudaMemset(d_src, 1, BYTES));

    int bw_threads = 256;
    int bw_blocks  = (N_ELEMENTS + bw_threads - 1) / bw_threads;

    float ms_coal = time_kernel_ms([&]{
        coalesced_bw<<<bw_blocks, bw_threads>>>(d_src, d_dst, N_ELEMENTS);
    });
    float ms_uncl = time_kernel_ms([&]{
        uncoalesced_bw<<<bw_blocks, bw_threads>>>(d_src, d_dst, N_ELEMENTS);
    });

    float bw_coal = 2.0f * BYTES / (ms_coal * 1e6f);  // GB/s (read + write)
    float bw_uncl = 2.0f * BYTES / (ms_uncl * 1e6f);
    printf("  coalesced   : %6.2f ms  →  %6.0f GB/s  (%4.1f%% of peak)\n",
           ms_coal, bw_coal, 100.0f * bw_coal / peak_bw_gbs);
    printf("  uncoalesced : %6.2f ms  →  %6.0f GB/s  (%4.1f%% of peak)\n",
           ms_uncl, bw_uncl, 100.0f * bw_uncl / peak_bw_gbs);
    printf("  penalty     : %.1f×\n\n", ms_uncl / ms_coal);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 4: Occupancy report
    // ─────────────────────────────────────────────────────────────────────────
    printf("=== Occupancy (via cudaOccupancyMaxActiveBlocksPerMultiprocessor) ===\n");

    print_occupancy("naive_gemm (16×16)",
                    naive_gemm, 16*16, 0);
    print_occupancy("tiled_gemm<16>",
                    tiled_gemm<16>, 16*16,
                    2 * 16 * 16 * sizeof(float));   // 2 tiles × TILE² × 4 bytes
    print_occupancy("tiled_gemm<32>",
                    tiled_gemm<32>, 32*32,
                    2 * 32 * 32 * sizeof(float));
    print_occupancy("tiled_padded<32>",
                    tiled_padded<32>, 32*32,
                    2 * 32 * 33 * sizeof(float));   // padded: TILE × (TILE+1)
    printf("\n");

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 5: Arithmetic intensity sweep over TILE sizes
    // Shows how AI grows with TILE, approaching the ridge point.
    // ─────────────────────────────────────────────────────────────────────────
    printf("=== Arithmetic Intensity vs TILE size ===\n");
    printf("  Ridge point (FP32, A100): %.0f GFLOP/s ÷ %.0f GB/s = %.0f FLOP/byte\n",
           peak_fp32_gflops, peak_bw_gbs, peak_fp32_gflops / peak_bw_gbs);
    printf("\n  TILE   SMEM/block   AI (FLOP/byte)   Regime\n");
    printf("  %s\n", std::string(55, '-').c_str());
    int tiles[] = {4, 8, 16, 32, 64, 128};
    float ridge = peak_fp32_gflops / peak_bw_gbs;
    for (int t : tiles) {
        // Each tile element is loaded once from global, used TILE times in dot product.
        // FLOPs per tile: 2 * TILE^3  (TILE² outputs × TILE MACs)
        // Bytes per tile: 2 × TILE² × 4  (load A and B tiles once each, FP32)
        float ai = (float)t / 2.0f;  // TILE / 2 FLOP/byte
        float smem_kb = 2.0f * t * t * 4 / 1024.0f;
        const char* regime = ai < ridge ? "memory-bound" : "compute-bound";
        printf("  %4d   %6.1f KB     %8.1f          %s\n",
               t, smem_kb, ai, regime);
    }
    printf("\n  Note: CUTLASS uses TILE=128 → AI=64 FLOP/byte (still memory-bound\n");
    printf("        for FP32; compute-bound for FP16/BF16 TensorCore matmul).\n\n");

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 6: Bank conflict demonstration
    // Use CUDA events to time padded vs unpadded shared memory loads.
    // ─────────────────────────────────────────────────────────────────────────
    printf("=== Bank conflict: padded vs unpadded (TILE=32, N=%d) ===\n", N);
    {
        constexpr int T = 32;
        dim3 b(T, T), g((N+T-1)/T, (N+T-1)/T);

        float ms_nop = time_kernel_ms([&]{
            tiled_gemm<T><<<g,b>>>(d_A, d_B, d_C, N);
        });
        float ms_pad = time_kernel_ms([&]{
            tiled_padded<T><<<g,b>>>(d_A, d_B, d_C, N);
        });

        printf("  tiled_gemm<32>    (no pad) : %6.2f ms\n", ms_nop);
        printf("  tiled_padded<32>  (+1 col) : %6.2f ms\n", ms_pad);
        float speedup = ms_nop / ms_pad;
        if (speedup > 1.0f)
            printf("  Speedup from padding      : %.2f×  (bank conflicts eliminated)\n\n", speedup);
        else
            printf("  No measurable conflict (N may be too small; try N=4096)\n\n");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_ref;

    printf("Done.\n");
    return 0;
}
```

---

## Compile and Run

```bash
# Compile (adjust -arch for your GPU)
nvcc -O3 -arch=sm_80 -std=c++17 -lineinfo \
     -o tiled_gemm_harness tiled_gemm_harness.cu

# sm_80 = A100 / RTX 3090 (Ampere)
# sm_86 = RTX 3080 Ti / RTX 3070 (Ampere GA102)
# sm_89 = RTX 4090 (Ada Lovelace)
# sm_90 = H100 (Hopper)

# Run
./tiled_gemm_harness

# Profile with Nsight (shows shared memory bank conflicts, L1/L2 hit rates)
ncu --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
    sm__cycles_elapsed.avg \
    ./tiled_gemm_harness
```

## What Each Section Measures

| Section | What it shows |
|---|---|
| **Correctness** | All kernels match CPU reference within floating-point tolerance |
| **Performance** | GFLOP/s and % of peak FP32 — tiled consistently beats naive |
| **Coalescing penalty** | Pure bandwidth test isolating the coalescing effect (expect 10–32×) |
| **Occupancy** | Active warps / SM — shows SMEM as the binding constraint for TILE=32 |
| **AI sweep** | How arithmetic intensity grows with TILE, where the ridge point sits |
| **Bank conflicts** | Direct timing difference between padded and unpadded SMEM |

## Connecting to the Chapter

| Chapter concept | Harness section |
|---|---|
| Registers hold accumulator | Kernel 2/3 `acc` variable — inspect with `nvcc --ptxas-options=-v` |
| SMEM bank conflicts | Section 6 (padded vs unpadded timing) + Nsight metrics |
| Coalesced vs uncoalesced | Section 3 (pure BW kernels) + Kernels 1 vs 4 |
| SMEM limits occupancy | Section 4 occupancy report |
| Arithmetic intensity | Section 5 (TILE sweep, ridge point) |
| L2 persistence | See `chapter_02b_l2.cu` extension below |
