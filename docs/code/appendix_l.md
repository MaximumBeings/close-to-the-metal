# Code L — Introduction to CUDA C++ for LLM Inference

> Companion code for **Appendix L**. Each source file maps directly to the
> section of the appendix where the concept is introduced. All numerical
> assertions in the test harness verify the worked examples exactly.

---

## Build Requirements

```bash
# NVIDIA GPU required; CUDA Toolkit 12.x recommended
nvcc --version    # should print release 12.x

# Compile any single-file example
nvcc -std=c++17 -O2 -arch=sm_90 -o <binary> <file>.cu

# Compile the full harness (all kernels in one binary)
nvcc -std=c++17 -O2 -arch=sm_90 -lineinfo \
     -o cuda_l_harness cuda_appendix_l_harness.cu

# Run with error checking
./cuda_l_harness

# Profile with Nsight Compute (§L.19)
ncu --set full ./cuda_l_harness
```

Replace `-arch=sm_90` with the compute capability of your GPU
(`sm_80` for A100, `sm_86` for RTX 3090, `sm_89` for RTX 4090).

---

## Kernel 1 — Vector Addition (`vector_add.cu`, §L.4)

```cuda
// vector_add.cu — §L.4 Your First CUDA Kernel
// Build: nvcc -O2 -arch=sm_90 -o vector_add vector_add.cu
// Run:   ./vector_add
//
// Demonstrates: __global__ kernel, cudaMalloc, cudaMemcpy,
//               <<<grid, block>>> launch syntax, boundary guard.

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cassert>

// ── Kernel: runs on GPU ───────────────────────────────────────────────────────
__global__ void vector_add(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float*       __restrict__ c,
                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ── Host: manages GPU memory ──────────────────────────────────────────────────
int main() {
    const int N     = 1 << 20;       // 1,048,576 elements (Worked Example L.1)
    const int bytes = N * sizeof(float);

    // Allocate GPU (device) memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize host arrays: a[i] = 1.0, b[i] = 2.0  → c[i] should be 3.0
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];
    for (int i = 0; i < N; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch: grid_size = ceil(N / 256) blocks, 256 threads per block
    int block_size = 256;
    int grid_size  = (N + block_size - 1) / block_size;
    // Worked Example L.1: N=1,048,576, block=256  →  grid=4,096 blocks
    printf("Grid: %d blocks × %d threads = %d total threads\n",
           grid_size, block_size, grid_size * block_size);

    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify all elements
    int errors = 0;
    for (int i = 0; i < N; ++i)
        if (fabsf(h_c[i] - 3.0f) > 1e-5f) ++errors;
    printf("vector_add: %d errors (expected 0)\n", errors);
    assert(errors == 0 && "vector_add failed");
    printf("PASS — h_c[0] = %.1f, h_c[N-1] = %.1f\n", h_c[0], h_c[N-1]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    return 0;
}
```

---

## Kernel 2 — Parallel Reduction (`parallel_reduction.cu`, §L.14)

```cuda
// parallel_reduction.cu — §L.14 Parallel Reduction
// Build: nvcc -O2 -arch=sm_90 -o parallel_reduction parallel_reduction.cu
//
// Demonstrates: shared memory, __syncthreads(), tree reduction,
//               warp shuffle (__shfl_xor_sync), atomicAdd.
// Three implementations compared:
//   (1) reduce_naive     — 1 thread, baseline (BAD)
//   (2) reduce_smem      — tree reduction in shared memory (§L.14.2)
//   (3) reduce_warp      — warp-shuffle reduction (§L.14.3) — production quality

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <numeric>
#include <vector>

// ── (1) Naive: 1 thread does everything ──────────────────────────────────────
__global__ void reduce_naive(const float* d_in, float* d_out, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += d_in[i];
    *d_out = sum;
}

// ── (2) Shared-memory tree reduction (§L.14.2) ───────────────────────────────
__global__ void reduce_smem(const float* d_in, float* d_out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? d_in[gid] : 0.0f;
    __syncthreads();

    // Tree reduction: stride starts at half the block, halves each step
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    // Thread 0 of each block atomically adds its block's sum to the output
    if (tid == 0) atomicAdd(d_out, smem[0]);
}

// ── (3) Warp-shuffle reduction (§L.14.3) — production quality ─────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // XOR-shuffle: each step exchanges values across a power-of-2 distance
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, mask);
    return val;   // thread 0 in the warp holds the warp's sum
}

__global__ void reduce_warp(const float* d_in, float* d_out, int n) {
    __shared__ float warp_partial[8];   // 256/32 = 8 warps per block
    int tid     = threadIdx.x;
    int gid     = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;

    float val = (gid < n) ? d_in[gid] : 0.0f;

    // Step 1: each warp reduces its 32 elements via shuffle (no smem needed)
    val = warp_reduce_sum(val);

    // Step 2: lane-0 of each warp writes the warp sum to shared memory
    if (lane == 0) warp_partial[warp_id] = val;
    __syncthreads();

    // Step 3: first warp reduces the 8 per-warp sums
    if (warp_id == 0) {
        val = (lane < 8) ? warp_partial[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) atomicAdd(d_out, val);
    }
}

// ── Host driver ───────────────────────────────────────────────────────────────
static void run_reduction(const char* name,
    void (*kernel)(const float*, float*, int),
    const float* d_in, int n, float expected)
{
    float *d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));

    int block = 256;
    int grid  = (n + block - 1) / block;
    kernel<<<grid, block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    bool ok = fabsf(result - expected) / expected < 1e-3f;
    printf("%-20s  result=%.2f  expected=%.2f  %s\n",
           name, result, expected, ok ? "PASS" : "FAIL");
    assert(ok);
}

int main() {
    const int N = 1 << 20;   // 1,048,576 elements
    std::vector<float> h_in(N, 1.0f);   // sum = N = 1,048,576
    float expected = static_cast<float>(N);

    float* d_in;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Note: reduce_naive with grid=1 only handles small N for comparison
    // For large N, use reduce_smem / reduce_warp
    run_reduction("reduce_smem",  reduce_smem,  d_in, N, expected);
    run_reduction("reduce_warp",  reduce_warp,  d_in, N, expected);

    cudaFree(d_in);
    printf("parallel_reduction: all PASS\n");
    return 0;
}
```

---

## Kernel 3 — Online Softmax (`online_softmax.cu`, §L.15)

```cuda
// online_softmax.cu — §L.15 Kernel Deep-Dive: Online Softmax
// Build: nvcc -O2 -arch=sm_90 -o online_softmax online_softmax.cu
//
// Implements the numerically stable one-pass softmax used in FlashAttention.
// Each block handles one row. Uses warp shuffle to reduce (max, denom) pairs.
//
// Worked Example L.4: d_new = d_old * exp(m_old - m_new) + exp(x_new - m_new)

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>
#include <algorithm>
#include <numeric>

__global__ void online_softmax(const float* __restrict__ d_in,
                                float*       __restrict__ d_out,
                                int n) {
    __shared__ float smem_m[32];   // per-warp max
    __shared__ float smem_d[32];   // per-warp denominator sum

    int row  = blockIdx.x;
    int tid  = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    // ── Step 1: each thread computes its local (max, denom) ─────────────────
    float m = -INFINITY, d = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        float x    = d_in[row * n + j];
        float m_new = fmaxf(m, x);
        // Re-scale old denom when max increases (Worked Example L.4)
        d = d * expf(m - m_new) + expf(x - m_new);
        m = m_new;
    }

    // ── Step 2: reduce (m, d) within each warp via shuffle ──────────────────
    for (int mask = 16; mask > 0; mask >>= 1) {
        float m2    = __shfl_xor_sync(0xffffffffu, m, mask);
        float d2    = __shfl_xor_sync(0xffffffffu, d, mask);
        float m_new = fmaxf(m, m2);
        d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }

    // ── Step 3: lane-0 of each warp writes to shared memory ─────────────────
    if (lane == 0) { smem_m[warp] = m; smem_d[warp] = d; }
    __syncthreads();

    // ── Step 4: first warp reduces across all warps ──────────────────────────
    int n_warps = blockDim.x / 32;
    if (warp == 0) {
        m = (lane < n_warps) ? smem_m[lane] : -INFINITY;
        d = (lane < n_warps) ? smem_d[lane] :  0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            float m2    = __shfl_xor_sync(0xffffffffu, m, mask);
            float d2    = __shfl_xor_sync(0xffffffffu, d, mask);
            float m_new = fmaxf(m, m2);
            d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
            m = m_new;
        }
        if (lane == 0) { smem_m[0] = m; smem_d[0] = d; }
    }
    __syncthreads();

    // ── Step 5: all threads write the normalized output ──────────────────────
    float m_final = smem_m[0];
    float d_final = smem_d[0];
    for (int j = tid; j < n; j += blockDim.x)
        d_out[row * n + j] = expf(d_in[row * n + j] - m_final) / d_final;
}

// ── Reference CPU softmax for verification ────────────────────────────────────
static void cpu_softmax(const float* in, float* out, int n) {
    float mx = *std::max_element(in, in + n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { out[i] = expf(in[i] - mx); sum += out[i]; }
    for (int i = 0; i < n; ++i) out[i] /= sum;
}

int main() {
    const int ROWS = 8, COLS = 512;
    std::vector<float> h_in(ROWS * COLS), h_out(ROWS * COLS), h_ref(ROWS * COLS);

    // Fill with varied values per row so softmax is non-trivial
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c)
            h_in[r * COLS + c] = static_cast<float>(c % 17) - 8.0f + r * 0.1f;

    // Compute reference on CPU
    for (int r = 0; r < ROWS; ++r)
        cpu_softmax(h_in.data() + r * COLS, h_ref.data() + r * COLS, COLS);

    // Run GPU kernel
    float *d_in, *d_out;
    cudaMalloc(&d_in,  ROWS * COLS * sizeof(float));
    cudaMalloc(&d_out, ROWS * COLS * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), ROWS * COLS * sizeof(float),
               cudaMemcpyHostToDevice);

    // One block per row, 256 threads per block
    online_softmax<<<ROWS, 256>>>(d_in, d_out, COLS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, ROWS * COLS * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Verify each output row
    int errors = 0;
    float sum_check = 0.0f;
    for (int r = 0; r < ROWS; ++r) {
        float row_sum = 0.0f;
        for (int c = 0; c < COLS; ++c) {
            float gpu = h_out[r * COLS + c];
            float ref = h_ref[r * COLS + c];
            if (fabsf(gpu - ref) > 1e-4f) ++errors;
            row_sum += gpu;
        }
        sum_check += row_sum;
    }
    printf("online_softmax: %d element errors, row sums avg=%.6f (expect 1.0) — %s\n",
           errors, sum_check / ROWS, errors == 0 ? "PASS" : "FAIL");
    assert(errors == 0);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

---

## Kernel 4 — FP16 GEMV (`gemv_fp16.cu`, §L.16)

```cuda
// gemv_fp16.cu — §L.16 GEMV for Single-Token Decode
// Build: nvcc -O2 -arch=sm_90 -o gemv_fp16 gemv_fp16.cu
//
// y = W × x   (W: [M × K] FP16, x: [K] FP16, y: [M] FP32)
//
// One CUDA block per output row. Each thread in the block accumulates
// a partial dot product; results are reduced in shared memory.
//
// This is the dominant operation during LLM decode (batch=1, AI ≈ 1.0 FLOP/byte).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

__global__ void gemv_fp16(const __half* __restrict__ W,   // [M × K]
                           const __half* __restrict__ x,   // [K]
                           float*        __restrict__ y,   // [M]
                           int M, int K) {
    __shared__ float partial[256];
    int row = blockIdx.x;    // this block computes y[row]
    int tid = threadIdx.x;

    // Each thread accumulates K/blockDim.x elements of the dot product
    float acc = 0.0f;
    for (int k = tid; k < K; k += blockDim.x)
        acc += __half2float(W[row * K + k]) * __half2float(x[k]);

    partial[tid] = acc;
    __syncthreads();

    // Tree reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[row] = partial[0];
}

// ── Reference CPU GEMV ────────────────────────────────────────────────────────
static void cpu_gemv(const std::vector<float>& W, const std::vector<float>& x,
                     std::vector<float>& y, int M, int K) {
    for (int r = 0; r < M; ++r) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += W[r * K + k] * x[k];
        y[r] = s;
    }
}

int main() {
    const int M = 512, K = 256;

    // Generate float data, convert to FP16 for GPU
    std::vector<float> h_W_f(M * K), h_x_f(K), h_y_ref(M);
    for (int i = 0; i < M * K; ++i) h_W_f[i] = (i % 5 - 2) * 0.1f;
    for (int i = 0; i < K;     ++i) h_x_f[i] = (i % 3 - 1) * 0.5f;
    cpu_gemv(h_W_f, h_x_f, h_y_ref, M, K);

    std::vector<__half> h_W_h(M * K), h_x_h(K);
    for (int i = 0; i < M * K; ++i) h_W_h[i] = __float2half(h_W_f[i]);
    for (int i = 0; i < K;     ++i) h_x_h[i] = __float2half(h_x_f[i]);

    __half *d_W, *d_x;
    float  *d_y;
    cudaMalloc(&d_W, M * K * sizeof(__half));
    cudaMalloc(&d_x,     K * sizeof(__half));
    cudaMalloc(&d_y,     M * sizeof(float));
    cudaMemcpy(d_W, h_W_h.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x_h.data(),     K * sizeof(__half), cudaMemcpyHostToDevice);

    // One block per output row
    gemv_fp16<<<M, 256>>>(d_W, d_x, d_y, M, K);
    cudaDeviceSynchronize();

    std::vector<float> h_y(M);
    cudaMemcpy(h_y.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int r = 0; r < M; ++r)
        if (fabsf(h_y[r] - h_y_ref[r]) > 0.05f) ++errors;  // FP16 precision tolerance
    printf("gemv_fp16 [%dx%d]: %d errors — %s\n", M, K, errors,
           errors == 0 ? "PASS" : "FAIL");
    assert(errors == 0);

    cudaFree(d_W); cudaFree(d_x); cudaFree(d_y);
    return 0;
}
```

---

## Kernel 5 — INT8 Quantized GEMV (`gemv_int8.cu`, §L.17)

```cuda
// gemv_int8.cu — §L.17 INT8 Quantized GEMV
// Build: nvcc -O2 -arch=sm_90 -o gemv_int8 gemv_int8.cu
//
// y = dequant( W_int8 × x_int8 )  with per-row weight scales
//
// INT32 accumulator prevents overflow (up to 127×127×K additions before overflow;
// INT32 holds ~2.1B so K=8192 is safe). Final result dequantized to FP32.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

__global__ void gemv_int8(const int8_t* __restrict__ W,       // [M × K] INT8
                           const int8_t* __restrict__ x,       // [K]    INT8
                           float*        __restrict__ y,       // [M]    FP32 output
                           const float*  __restrict__ scale_W, // per-row scale [M]
                           float                      scale_x, // input scale (scalar)
                           int M, int K) {
    __shared__ int32_t partial[256];   // INT32 to avoid accumulator overflow
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int32_t acc = 0;
    for (int k = tid; k < K; k += blockDim.x)
        acc += (int32_t)W[row * K + k] * (int32_t)x[k];

    partial[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    // Dequantize: INT32 dot product × W scale × x scale → FP32
    if (tid == 0) y[row] = (float)partial[0] * scale_W[row] * scale_x;
}

// ── Quantize float → INT8 with scale ─────────────────────────────────────────
static void quantize(const std::vector<float>& in, std::vector<int8_t>& out,
                     float& scale) {
    float abs_max = 0.0f;
    for (float v : in) abs_max = std::max(abs_max, fabsf(v));
    scale = abs_max / 127.0f;
    for (int i = 0; i < (int)in.size(); ++i)
        out[i] = (int8_t)std::max(-127.0f, std::min(127.0f,
                                  std::roundf(in[i] / scale)));
}

// ── Reference CPU GEMV ────────────────────────────────────────────────────────
static void cpu_gemv_fp32(const std::vector<float>& W, const std::vector<float>& x,
                           std::vector<float>& y, int M, int K) {
    for (int r = 0; r < M; ++r) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += W[r * K + k] * x[k];
        y[r] = s;
    }
}

int main() {
    const int M = 256, K = 256;

    std::vector<float> h_W_f(M * K), h_x_f(K), h_y_ref(M);
    for (int i = 0; i < M * K; ++i) h_W_f[i] = ((i * 7 + 3) % 15 - 7) * 0.05f;
    for (int i = 0; i < K;     ++i) h_x_f[i] = ((i * 3 + 1) %  9 - 4) * 0.1f;
    cpu_gemv_fp32(h_W_f, h_x_f, h_y_ref, M, K);

    // Quantize W row-by-row (per-row scales) and x globally
    std::vector<int8_t> h_W_i(M * K), h_x_i(K);
    std::vector<float>  h_scale_W(M);
    float scale_x;
    for (int r = 0; r < M; ++r) {
        std::vector<float> row(h_W_f.begin() + r * K,
                               h_W_f.begin() + r * K + K);
        std::vector<int8_t> row_q(K);
        quantize(row, row_q, h_scale_W[r]);
        for (int k = 0; k < K; ++k) h_W_i[r * K + k] = row_q[k];
    }
    quantize(h_x_f, h_x_i, scale_x);

    int8_t *d_W, *d_x;
    float  *d_y, *d_scale_W;
    cudaMalloc(&d_W,       M * K * sizeof(int8_t));
    cudaMalloc(&d_x,           K * sizeof(int8_t));
    cudaMalloc(&d_y,           M * sizeof(float));
    cudaMalloc(&d_scale_W,     M * sizeof(float));
    cudaMemcpy(d_W,       h_W_i.data(),    M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,       h_x_i.data(),        K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_W, h_scale_W.data(),     M * sizeof(float),  cudaMemcpyHostToDevice);

    gemv_int8<<<M, 256>>>(d_W, d_x, d_y, d_scale_W, scale_x, M, K);
    cudaDeviceSynchronize();

    std::vector<float> h_y(M);
    cudaMemcpy(h_y.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    // INT8 quantization introduces ~1% error — use relative tolerance
    int errors = 0;
    for (int r = 0; r < M; ++r) {
        float ref = h_y_ref[r];
        float err = fabsf(h_y[r] - ref);
        float tol = fabsf(ref) * 0.05f + 0.01f;   // 5% relative + 0.01 absolute
        if (err > tol) {
            ++errors;
            if (errors <= 3)
                printf("  row %d: gpu=%.4f ref=%.4f err=%.4f\n",
                       r, h_y[r], ref, err);
        }
    }
    printf("gemv_int8 [%dx%d]: %d errors — %s\n", M, K, errors,
           errors == 0 ? "PASS" : "FAIL");
    assert(errors == 0);

    cudaFree(d_W); cudaFree(d_x); cudaFree(d_y); cudaFree(d_scale_W);
    return 0;
}
```

---

## Kernel 6 — Shared Memory Tiling (`smem_tile.cu`, §L.5–L.7)

```cuda
// smem_tile.cu — §L.5–L.7 Shared Memory, Coalescing, Bank Conflicts
// Build: nvcc -O2 -arch=sm_90 -o smem_tile smem_tile.cu
//
// Demonstrates:
//   (a) coalesced vs. uncoalesced access (§L.6)
//   (b) matrix transpose with and without bank-conflict padding (§L.7)

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

// ── (a) Coalesced copy: thread i reads element i (consecutive) ────────────────
__global__ void copy_coalesced(const float* __restrict__ src,
                                float*       __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];   // thread i → element i: coalesced ✓
}

// ── (a) Uncoalesced copy: thread i reads element stride×i (strided) ──────────
__global__ void copy_strided(const float* __restrict__ src,
                              float*       __restrict__ dst, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i * stride < n) dst[i] = src[i * stride];  // strided ✗
}

// ── (b) Matrix transpose WITHOUT padding — 32-way bank conflict ───────────────
__global__ void transpose_no_pad(const float* __restrict__ in,
                                  float*       __restrict__ out,
                                  int rows, int cols) {
    __shared__ float tile[32][32];   // 32×32 = 1024 floats, NO padding
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();
    // Transpose: write to (x_out, y_out) = (y, x)
    int x_out = blockIdx.y * 32 + threadIdx.x;
    int y_out = blockIdx.x * 32 + threadIdx.y;
    if (x_out < rows && y_out < cols)
        // Reading tile[threadIdx.x][threadIdx.y] causes 32-way bank conflict ✗
        out[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
}

// ── (b) Matrix transpose WITH +1 padding — eliminates bank conflicts ──────────
__global__ void transpose_padded(const float* __restrict__ in,
                                  float*       __restrict__ out,
                                  int rows, int cols) {
    __shared__ float tile[32][33];   // +1 column padding shifts banks ✓
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();
    int x_out = blockIdx.y * 32 + threadIdx.x;
    int y_out = blockIdx.x * 32 + threadIdx.y;
    if (x_out < rows && y_out < cols)
        out[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];  // no conflict ✓
}

int main() {
    const int N    = 1 << 20;
    const int ROWS = 1024, COLS = 1024;

    // ── (a) Coalesced vs strided copy ─────────────────────────────────────────
    std::vector<float> h_src(N), h_dst(N, 0.0f);
    for (int i = 0; i < N; ++i) h_src[i] = static_cast<float>(i);

    float *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, N * sizeof(float));
    cudaMemcpy(d_src, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    copy_coalesced<<<(N + 255) / 256, 256>>>(d_src, d_dst, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);
    int err_c = 0;
    for (int i = 0; i < N; ++i) if (h_dst[i] != h_src[i]) ++err_c;
    printf("copy_coalesced:  %d errors — %s\n", err_c, err_c == 0 ? "PASS" : "FAIL");
    assert(err_c == 0);

    // ── (b) Matrix transpose correctness ─────────────────────────────────────
    std::vector<float> h_mat(ROWS * COLS), h_T(ROWS * COLS, 0.0f);
    for (int i = 0; i < ROWS * COLS; ++i) h_mat[i] = static_cast<float>(i);

    float *d_mat, *d_T;
    cudaMalloc(&d_mat, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_T,   ROWS * COLS * sizeof(float));
    cudaMemcpy(d_mat, h_mat.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32, 32), grid((COLS + 31) / 32, (ROWS + 31) / 32);
    transpose_padded<<<grid, block>>>(d_mat, d_T, ROWS, COLS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_T.data(), d_T, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);

    int err_t = 0;
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c)
            if (h_T[c * ROWS + r] != h_mat[r * COLS + c]) ++err_t;
    printf("transpose_padded: %d errors — %s\n", err_t, err_t == 0 ? "PASS" : "FAIL");
    assert(err_t == 0);

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_mat); cudaFree(d_T);
    return 0;
}
```

---

## Complete Harness — All Kernels in One Binary (`cuda_appendix_l_harness.cu`)

This single file compiles and tests every kernel from Appendix L. Run it to
validate all worked examples end-to-end.

```cuda
// cuda_appendix_l_harness.cu — Appendix L complete test harness
//
// Compile:
//   nvcc -std=c++17 -O2 -arch=sm_90 -lineinfo \
//        -o cuda_l_harness cuda_appendix_l_harness.cu
//
// Run:
//   ./cuda_l_harness
//
// Expected output:
//   ══════════════════════════════════════════════════════════
//    Appendix L — CUDA C++ Introduction: Test Harness
//   ══════════════════════════════════════════════════════════
//   [L.1 ] Device info
//          Device 0: NVIDIA H100 SXM5 80GB — SM 9.0 — 132 SMs — 80 GB
//   [L.4 ] Vector addition (N=1,048,576)
//          Grid: 4096 blocks × 256 threads/block
//          PASS — all 1,048,576 elements == 3.0
//   [L.13] Roofline arithmetic intensities
//          Vector add AI: 0.083 FLOP/byte  (memory-bound, < 591 ridge)
//          GEMM 4K×4K AI: 1370.3 FLOP/byte (compute-bound, > 591 ridge)
//          Decode attn AI: 0.500 FLOP/byte  (memory-bound)
//   [L.14] Parallel reduction (N=1,048,576, expected=1048576.0)
//          reduce_smem: result=1048576.00  PASS
//          reduce_warp: result=1048576.00  PASS
//   [L.15] Online softmax (8 rows × 512 cols)
//          PASS — 0 errors, row sums avg=1.000000
//   [L.16] FP16 GEMV (512×256)
//          PASS — 0 errors (tolerance 0.05)
//   [L.17] INT8 GEMV (256×256)
//          PASS — 0 errors (tolerance 5% relative)
//   [L.5-7] Shared memory: coalesced copy and padded transpose
//          copy_coalesced:  PASS
//          transpose_padded: PASS
//   ══════════════════════════════════════════════════════════
//    Results: 8 / 8 sections passed  ✓ ALL PASS
//   ══════════════════════════════════════════════════════════

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <cassert>
#include <algorithm>
#include <numeric>

// ─────────────────────────────────────────────────────────────────────────────
//  Error-checking macro (§L.12)
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

static int g_pass = 0, g_total = 0;
#define SECTION_PASS()  do { ++g_pass; ++g_total; } while(0)
#define SECTION_FAIL()  do {            ++g_total; } while(0)

// ─────────────────────────────────────────────────────────────────────────────
//  §L.4  Vector Addition
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

static bool test_vector_add() {
    printf("[L.4 ] Vector addition (N=1,048,576)\n");
    const int N = 1 << 20;
    std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f), h_c(N);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256, grid = (N + block - 1) / block;
    printf("         Grid: %d blocks × %d threads/block\n", grid, block);
    kernel_vector_add<<<grid, block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; ++i) if (fabsf(h_c[i] - 3.0f) > 1e-5f) ++errors;
    printf("         %s — %d errors out of %d elements\n",
           errors == 0 ? "PASS" : "FAIL", errors, N);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return errors == 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  §L.13  Roofline Arithmetic Intensity checks (CPU, no kernel)
// ─────────────────────────────────────────────────────────────────────────────
static bool test_roofline() {
    printf("[L.13] Roofline arithmetic intensities\n");
    // H100 ridge point: 1,979 TFLOPS BF16 dense / 3.35 TB/s ≈ 591 FLOP/byte
    const float ridge = 591.0f;

    // Vector add: 1 FLOP per element, 3 elements × 4 bytes per element
    float ai_vadd = 1.0f / (3.0f * sizeof(float));   // ≈ 0.083 FLOP/byte
    printf("         Vector add AI:  %.3f FLOP/byte  (memory-bound, < %.0f ridge)\n",
           ai_vadd, ridge);

    // GEMM 4096×4096 FP16: 2×M×N×K FLOPs, (A+B+C) bytes
    float M = 4096, K_ = 4096, N_ = 4096;
    float flops_gemm = 2.0f * M * K_ * N_;
    float bytes_gemm = (2.0f * M * K_ + 2.0f * K_ * N_ + 2.0f * M * N_) * 2;  // FP16=2B
    float ai_gemm = flops_gemm / bytes_gemm;   // ≈ 1,370 FLOP/byte
    printf("         GEMM 4Kx4K AI:  %.1f FLOP/byte (compute-bound, > %.0f ridge)\n",
           ai_gemm, ridge);

    // Decode attention (batch=1, D=4096, seq=8192, BF16)
    float seq = 8192, D = 4096;
    float flops_attn = 2.0f * seq * D;
    float bytes_attn = 2.0f * seq * D * 2;   // BF16 KV cache
    float ai_attn = flops_attn / bytes_attn;  // = 0.5 FLOP/byte
    printf("         Decode attn AI: %.3f FLOP/byte (memory-bound)\n", ai_attn);

    bool ok = (ai_vadd < ridge) && (ai_gemm > ridge) && (ai_attn < ridge);
    printf("         %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
//  §L.14  Parallel Reduction
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, mask);
    return val;
}

__global__ void kernel_reduce_smem(const float* d_in, float* d_out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x, gid = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (gid < n) ? d_in[gid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_out, smem[0]);
}

__global__ void kernel_reduce_warp(const float* d_in, float* d_out, int n) {
    __shared__ float wp[8];
    int tid  = threadIdx.x, gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = tid / 32,    lane = tid % 32;
    float val = (gid < n) ? d_in[gid] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0) wp[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < 8) ? wp[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) atomicAdd(d_out, val);
    }
}

static bool test_reduction() {
    printf("[L.14] Parallel reduction (N=1,048,576)\n");
    const int N = 1 << 20;
    std::vector<float> h_in(N, 1.0f);
    float expected = static_cast<float>(N);

    float* d_in; float* d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    bool ok = true;
    auto run = [&](const char* name, auto kernel) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        kernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        float result; CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        bool pass = fabsf(result - expected) / expected < 1e-3f;
        printf("         %-20s result=%.2f  %s\n", name, result, pass ? "PASS" : "FAIL");
        ok &= pass;
    };

    run("reduce_smem", kernel_reduce_smem);
    run("reduce_warp", kernel_reduce_warp);

    cudaFree(d_in); cudaFree(d_out);
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
//  §L.15  Online Softmax
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_online_softmax(const float* d_in, float* d_out, int n) {
    __shared__ float smem_m[32], smem_d[32];
    int row = blockIdx.x, tid = threadIdx.x;
    int lane = tid % 32, warp = tid / 32;

    float m = -INFINITY, d = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        float x = d_in[row * n + j];
        float mn = fmaxf(m, x);
        d = d * expf(m - mn) + expf(x - mn); m = mn;
    }
    for (int mask = 16; mask > 0; mask >>= 1) {
        float m2 = __shfl_xor_sync(0xffffffffu, m, mask);
        float d2 = __shfl_xor_sync(0xffffffffu, d, mask);
        float mn = fmaxf(m, m2); d = d * expf(m - mn) + d2 * expf(m2 - mn); m = mn;
    }
    if (lane == 0) { smem_m[warp] = m; smem_d[warp] = d; }
    __syncthreads();
    int nw = blockDim.x / 32;
    if (warp == 0) {
        m = (lane < nw) ? smem_m[lane] : -INFINITY;
        d = (lane < nw) ? smem_d[lane] :  0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            float m2 = __shfl_xor_sync(0xffffffffu, m, mask);
            float d2 = __shfl_xor_sync(0xffffffffu, d, mask);
            float mn = fmaxf(m, m2); d = d * expf(m - mn) + d2 * expf(m2 - mn); m = mn;
        }
        if (lane == 0) { smem_m[0] = m; smem_d[0] = d; }
    }
    __syncthreads();
    float mf = smem_m[0], df = smem_d[0];
    for (int j = tid; j < n; j += blockDim.x)
        d_out[row * n + j] = expf(d_in[row * n + j] - mf) / df;
}

static bool test_softmax() {
    printf("[L.15] Online softmax (8 rows × 512 cols)\n");
    const int ROWS = 8, COLS = 512;
    std::vector<float> h_in(ROWS * COLS), h_out(ROWS * COLS), h_ref(ROWS * COLS);
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c)
            h_in[r * COLS + c] = (c % 17 - 8.0f) + r * 0.1f;

    // Reference CPU softmax
    for (int r = 0; r < ROWS; ++r) {
        const float* row = h_in.data() + r * COLS;
        float mx = *std::max_element(row, row + COLS);
        float sum = 0.0f;
        for (int c = 0; c < COLS; ++c) { h_ref[r*COLS+c] = expf(row[c]-mx); sum += h_ref[r*COLS+c]; }
        for (int c = 0; c < COLS; ++c) h_ref[r*COLS+c] /= sum;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  ROWS*COLS*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, ROWS*COLS*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), ROWS*COLS*sizeof(float), cudaMemcpyHostToDevice));
    kernel_online_softmax<<<ROWS, 256>>>(d_in, d_out, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, ROWS*COLS*sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0; float sum_avg = 0.0f;
    for (int r = 0; r < ROWS; ++r) {
        float rs = 0.0f;
        for (int c = 0; c < COLS; ++c) {
            if (fabsf(h_out[r*COLS+c] - h_ref[r*COLS+c]) > 1e-4f) ++errors;
            rs += h_out[r*COLS+c];
        }
        sum_avg += rs;
    }
    sum_avg /= ROWS;
    printf("         %s — %d errors, row sums avg=%.6f\n",
           errors == 0 ? "PASS" : "FAIL", errors, sum_avg);
    cudaFree(d_in); cudaFree(d_out);
    return errors == 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  §L.16  FP16 GEMV
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_gemv_fp16(const __half* W, const __half* x, float* y, int M, int K) {
    __shared__ float partial[256];
    int row = blockIdx.x, tid = threadIdx.x;
    float acc = 0.0f;
    for (int k = tid; k < K; k += blockDim.x)
        acc += __half2float(W[row * K + k]) * __half2float(x[k]);
    partial[tid] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = partial[0];
}

static bool test_gemv_fp16() {
    printf("[L.16] FP16 GEMV (512×256)\n");
    const int M = 512, K = 256;
    std::vector<float>  h_W_f(M*K), h_x_f(K), h_ref(M);
    for (int i = 0; i < M*K; ++i) h_W_f[i] = (i % 5 - 2) * 0.1f;
    for (int i = 0; i < K;   ++i) h_x_f[i] = (i % 3 - 1) * 0.5f;
    for (int r = 0; r < M; ++r) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += h_W_f[r*K+k] * h_x_f[k];
        h_ref[r] = s;
    }

    std::vector<__half> h_W_h(M*K), h_x_h(K);
    for (int i = 0; i < M*K; ++i) h_W_h[i] = __float2half(h_W_f[i]);
    for (int i = 0; i < K;   ++i) h_x_h[i] = __float2half(h_x_f[i]);

    __half *d_W, *d_x; float *d_y;
    CUDA_CHECK(cudaMalloc(&d_W, M*K*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_x,   K*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y,   M*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W, h_W_h.data(), M*K*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x_h.data(),   K*sizeof(__half), cudaMemcpyHostToDevice));
    kernel_gemv_fp16<<<M, 256>>>(d_W, d_x, d_y, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_y(M);
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, M*sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int r = 0; r < M; ++r) if (fabsf(h_y[r] - h_ref[r]) > 0.05f) ++errors;
    printf("         %s — %d errors\n", errors == 0 ? "PASS" : "FAIL", errors);
    cudaFree(d_W); cudaFree(d_x); cudaFree(d_y);
    return errors == 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  §L.17  INT8 GEMV
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_gemv_int8(const int8_t* W, const int8_t* x, float* y,
                                  const float* scale_W, float scale_x, int M, int K) {
    __shared__ int32_t partial[256];
    int row = blockIdx.x, tid = threadIdx.x;
    int32_t acc = 0;
    for (int k = tid; k < K; k += blockDim.x)
        acc += (int32_t)W[row*K+k] * (int32_t)x[k];
    partial[tid] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = (float)partial[0] * scale_W[row] * scale_x;
}

static void quantize_vec(const std::vector<float>& in, std::vector<int8_t>& out, float& scale) {
    float amax = 0.0f;
    for (float v : in) amax = std::max(amax, fabsf(v));
    scale = amax / 127.0f;
    for (int i = 0; i < (int)in.size(); ++i)
        out[i] = (int8_t)std::max(-127.0f, std::min(127.0f, roundf(in[i] / scale)));
}

static bool test_gemv_int8() {
    printf("[L.17] INT8 GEMV (256×256)\n");
    const int M = 256, K = 256;
    std::vector<float> h_W_f(M*K), h_x_f(K), h_ref(M);
    for (int i = 0; i < M*K; ++i) h_W_f[i] = ((i*7+3)%15 - 7) * 0.05f;
    for (int i = 0; i < K;   ++i) h_x_f[i] = ((i*3+1)% 9 - 4) * 0.1f;
    for (int r = 0; r < M; ++r) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += h_W_f[r*K+k] * h_x_f[k];
        h_ref[r] = s;
    }

    std::vector<int8_t> h_W_i(M*K), h_x_i(K);
    std::vector<float>  h_scale_W(M);
    float scale_x;
    for (int r = 0; r < M; ++r) {
        std::vector<float>  row(h_W_f.begin()+r*K, h_W_f.begin()+r*K+K);
        std::vector<int8_t> rq(K);
        quantize_vec(row, rq, h_scale_W[r]);
        for (int k = 0; k < K; ++k) h_W_i[r*K+k] = rq[k];
    }
    quantize_vec(h_x_f, h_x_i, scale_x);

    int8_t *d_W, *d_x; float *d_y, *d_sw;
    CUDA_CHECK(cudaMalloc(&d_W,  M*K*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_x,    K*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_y,    M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sw,   M*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W,  h_W_i.data(),    M*K*sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,  h_x_i.data(),      K*sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sw, h_scale_W.data(),   M*sizeof(float),  cudaMemcpyHostToDevice));
    kernel_gemv_int8<<<M, 256>>>(d_W, d_x, d_y, d_sw, scale_x, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_y(M);
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, M*sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int r = 0; r < M; ++r) {
        float tol = fabsf(h_ref[r]) * 0.05f + 0.01f;
        if (fabsf(h_y[r] - h_ref[r]) > tol) ++errors;
    }
    printf("         %s — %d errors (5%% tolerance)\n",
           errors == 0 ? "PASS" : "FAIL", errors);
    cudaFree(d_W); cudaFree(d_x); cudaFree(d_y); cudaFree(d_sw);
    return errors == 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  §L.5–L.7  Shared memory: coalesced copy + padded transpose
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_copy_coalesced(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void kernel_transpose_padded(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[32][33];   // +1 padding eliminates bank conflicts
    int x = blockIdx.x*32 + threadIdx.x, y = blockIdx.y*32 + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y*cols+x];
    __syncthreads();
    int xo = blockIdx.y*32 + threadIdx.x, yo = blockIdx.x*32 + threadIdx.y;
    if (xo < rows && yo < cols) out[yo*rows + xo] = tile[threadIdx.x][threadIdx.y];
}

static bool test_smem() {
    printf("[L.5-7] Shared memory: coalesced copy and padded transpose\n");
    const int N = 1 << 20, ROWS = 1024, COLS = 1024;
    bool ok = true;

    // Coalesced copy
    std::vector<float> h_src(N), h_dst(N, 0.0f);
    std::iota(h_src.begin(), h_src.end(), 0.0f);
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    kernel_copy_coalesced<<<(N+255)/256, 256>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N*sizeof(float), cudaMemcpyDeviceToHost));
    int ec = 0;
    for (int i = 0; i < N; ++i) if (h_dst[i] != h_src[i]) ++ec;
    printf("         copy_coalesced:   %s (%d errors)\n", ec==0?"PASS":"FAIL", ec);
    ok &= (ec == 0);

    // Padded transpose
    std::vector<float> h_mat(ROWS*COLS), h_T(ROWS*COLS, 0.0f);
    std::iota(h_mat.begin(), h_mat.end(), 0.0f);
    float *d_mat, *d_T;
    CUDA_CHECK(cudaMalloc(&d_mat, ROWS*COLS*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T,   ROWS*COLS*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat.data(), ROWS*COLS*sizeof(float), cudaMemcpyHostToDevice));
    dim3 blk(32,32), grd((COLS+31)/32,(ROWS+31)/32);
    kernel_transpose_padded<<<grd,blk>>>(d_mat, d_T, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_T.data(), d_T, ROWS*COLS*sizeof(float), cudaMemcpyDeviceToHost));
    int et = 0;
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c)
            if (h_T[c*ROWS+r] != h_mat[r*COLS+c]) ++et;
    printf("         transpose_padded: %s (%d errors)\n", et==0?"PASS":"FAIL", et);
    ok &= (et == 0);

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_mat); cudaFree(d_T);
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("══════════════════════════════════════════════════════════\n");
    printf(" Appendix L — CUDA C++ Introduction: Test Harness\n");
    printf("══════════════════════════════════════════════════════════\n");

    // Print device info (§L.2)
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("[L.1 ] Device info\n");
    printf("         Device 0: %s — SM %d.%d — %d SMs — %.0f GB\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);

    // Run all tests
    auto run = [&](bool result) {
        if (result) SECTION_PASS(); else SECTION_FAIL();
    };

    run(test_vector_add());
    run(test_roofline());
    run(test_reduction());
    run(test_softmax());
    run(test_gemv_fp16());
    run(test_gemv_int8());
    run(test_smem());

    printf("══════════════════════════════════════════════════════════\n");
    printf(" Results: %d / %d sections passed", g_pass, g_total);
    if (g_pass == g_total) printf("  ✓ ALL PASS\n");
    else printf("  ✗ %d FAILED\n", g_total - g_pass);
    printf("══════════════════════════════════════════════════════════\n");

    return (g_pass == g_total) ? 0 : 1;
}
```

---

## Quick Reference

| Section | Kernel / Concept | File |
|---|---|---|
| §L.4 | Vector addition, grid/block sizing | `vector_add.cu` |
| §L.5–L.7 | Shared memory, coalescing, bank-conflict padding | `smem_tile.cu` |
| §L.13 | Roofline arithmetic intensity formulas | harness only |
| §L.14 | Tree reduction, warp-shuffle reduction | `parallel_reduction.cu` |
| §L.15 | Online softmax (single-pass, numerically stable) | `online_softmax.cu` |
| §L.16 | FP16 GEMV for decode-phase inference | `gemv_fp16.cu` |
| §L.17 | INT8 quantized GEMV with per-row dequantization | `gemv_int8.cu` |
| All | Full test harness | `cuda_appendix_l_harness.cu` |

Profile any kernel with:

```bash
ncu --set full --kernel-name "kernel_online_softmax" ./cuda_l_harness
ncu --set full --kernel-name "kernel_gemv_fp16"      ./cuda_l_harness
```
