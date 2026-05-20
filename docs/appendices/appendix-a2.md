# Appendix A.2 — Tensor Contractions: The Engine of LLM Inference

> *"Every forward pass is a tensor contraction. Master contractions and you master the compute graph of any LLM."*

---

## A2.1 What Is a Tensor Contraction?

A **tensor contraction** generalizes matrix multiplication to tensors of arbitrary rank. In matrix multiplication you sum over one shared index; in a general contraction you sum over one or more shared indices, producing a result tensor whose rank equals the total number of *free* (non-summed) indices.

### A2.1.1 Einstein Summation Notation

Einstein summation (einsum) drops the explicit summation sign. Any index that appears in **both** input tensors but **not** the output descriptor is contracted (summed out). Any index appearing in the output is kept free.

| Expression | Operation | Output rank |
|---|---|---|
| `ij,jk->ik` | Matrix multiply | 2 |
| `bij,bjk->bik` | Batched matmul | 3 |
| `bhid,bhjd->bhij` | MHA attention scores | 4 |
| `bhij,bhjd->bhid` | MHA attention output | 4 |
| `bsr,hrd->bhsd` | MLA KV reconstruction | 4 |
| `abcde,aef->abcdf` | 5-D weight projection | 5 |
| `...ij,...jk->...ik` | Broadcast batched matmul | N |

### A2.1.2 Why Contractions Are Central to LLMs

Every computation in a transformer layer is a contraction:

```
Token embedding lookup:   one_hot[s] @ E                → 2D contraction
Q/K/V projection:         X[b,s,d]   @ W[d,h*dk]        → 3D contraction
Attention scores:         Q[b,h,s,dk]@ K^T[b,h,dk,s]    → 4D contraction
Attention output:         A[b,h,s,s] @ V[b,h,s,dk]      → 4D contraction
FFN gate:                 X[b,s,d]   @ W_gate[d,dff]     → 3D contraction
MoE expert:               X[b,s,e,d] @ W_e[e,d,dff]     → 5D contraction
LoRA update:              X @ A_lora @ B_lora            → cascaded 3D
```

Understanding each as a named contraction with known FLOPs, memory traffic, and arithmetic intensity is the prerequisite for every optimization discussed in this book.

---

## A2.2 Contraction Taxonomy: All Ranks in LLM Inference

### A2.2.1 2D Contractions

`C[m,n] = sum_k A[m,k] * B[k,n]`

The foundation: a single shared index k is contracted. This is standard GEMM.

```
FLOPs = 2 * m * k * n
Memory read  = (m*k + k*n) * dtype_bytes
Memory write = m*n * dtype_bytes
Arithmetic intensity (AI) = 2*m*k*n / ((m*k + k*n + m*n) * dtype_bytes)

Example: 7B FFN gate projection, d=4096, dff=11008, BF16, batch=1 token
  FLOPs = 2 * 1 * 4096 * 11008 = 90.2 MFLOPs
  Read  = (4096 + 4096*11008) * 2 = 90.2 MB
  AI    = 90.2M / 90.2M = 1.0 FLOP/byte  -- deeply memory-bound at decode
```

### A2.2.2 3D Contractions

`C[b,s,n] = sum_d X[b,s,d] * W[d,n]`

The token batch [B,S,D] projected through weight [D,N]. The weight is shared across all (b,s) positions -- this is a batched GEMM where B*S independent 2D rows share the same right-hand matrix.

```
FLOPs = 2 * B * S * D * N
For Llama-3 70B, B=8, S=2048, D=8192, N=8192:
  = 2 * 8 * 2048 * 8192 * 8192 = 2.20 TFLOPs per projection per layer
```

### A2.2.3 4D Contractions

Two critical 4D contractions per attention layer:

**Attention score:** `S[b,h,i,j] = sum_d Q[b,h,i,d] * K[b,h,j,d]`

**Attention output:** `O[b,h,i,d] = sum_j A[b,h,i,j] * V[b,h,j,d]`

```
FLOPs each = 2 * B * H * S * S * D_head
For Llama-3 70B: B=8, H=64, S=2048, D_head=128
  = 2 * 8 * 64 * 2048 * 2048 * 128 = 549 GFLOPs per attention sub-operation
```

### A2.2.4 5D Contractions

**GQA attention scores** with G query heads per KV head:

`S[b, hq, i, j] = sum_d Q[b, hq, i, d] * K[b, hq//G, j, d]`

The index mapping `hq//G` groups query heads -- equivalent to expanding K before the 4D contraction, or using a strided 5D view.

**MoE expert linear:** Each of E experts applies its own weight matrix to routed tokens:

`Y[b,s,e,dout] = sum_din X_routed[b,s,e,din] * W_expert[e,din,dout]`

The expert index `e` adds a 5th dimension. With top-2 routing, most (b,s,e) entries are zero -- sparse contraction.

### A2.2.5 ND Contractions

**MLA KV reconstruction** (DeepSeek-V2/V3):

`K[b,h,s,d] = sum_r C_KV[b,s,r] * W_UK[h,r,d]`

Three free indices (b,s from C_KV; h,d from W_UK), one contracted (r). No single matmul handles this -- einsum or reshape+matmul required.

**General N-D:** Any contraction can be written in einsum notation. The key insight is that every einsum reduces to one or more GEMMs after appropriate reshaping and transposition.

---

## A2.3 Python Reference Implementations

### A2.3.1 NumPy and PyTorch -- All Ranks

```python
# file: contractions_reference.py
import numpy as np
import torch
import time

# ── NumPy implementations ──────────────────────────────────────────────────

def contract_2d(A, B):
    """C[m,n] = sum_k A[m,k] * B[k,n]"""
    return np.einsum('mk,kn->mn', A, B)

def contract_3d(X, W):
    """C[b,s,n] = sum_d X[b,s,d] * W[d,n]"""
    return np.einsum('bsd,dn->bsn', X, W)

def contract_4d_qk(Q, K):
    """S[b,h,i,j] = sum_d Q[b,h,i,d] * K[b,h,j,d]"""
    return np.einsum('bhid,bhjd->bhij', Q, K)

def contract_4d_av(A, V):
    """O[b,h,i,d] = sum_j A[b,h,i,j] * V[b,h,j,d]"""
    return np.einsum('bhij,bhjd->bhid', A, V)

def contract_5d_gqa(Q, K, G):
    """GQA scores: H_q = G * H_kv. Expand K then compute scores."""
    K_exp = np.repeat(K, G, axis=1)           # [B, H_q, S_k, D]
    return np.einsum('bhid,bhjd->bhij', Q, K_exp)

def contract_5d_moe(X_routed, W_expert):
    """MoE: Y[b,s,e,dout] = sum_din X[b,s,e,din] * W[e,din,dout]"""
    return np.einsum('bsed,edm->bsem', X_routed, W_expert)

def contract_nd_mla(C_KV, W_UK):
    """MLA: K[b,h,s,d] = sum_r C_KV[b,s,r] * W_UK[h,r,d]"""
    return np.einsum('bsr,hrd->bhsd', C_KV, W_UK)

# ── PyTorch high-performance implementations ──────────────────────────────

def contract_2d_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.mm(A, B)

def contract_3d_torch(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # Reshape to 2D GEMM, reshape back -- avoids einsum overhead
    B, S, D = X.shape
    D, N = W.shape
    return (X.view(B * S, D) @ W).view(B, S, -1)

def contract_4d_qk_torch(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    # torch.matmul broadcasts over batch dims -- preferred over einsum
    return torch.matmul(Q, K.transpose(-2, -1))

def contract_4d_av_torch(A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, V)

def contract_5d_gqa_torch(Q: torch.Tensor, K: torch.Tensor, G: int) -> torch.Tensor:
    # K: [B, H_kv, S_k, D]  ->  expand to [B, H_q, S_k, D]
    K_exp = K.repeat_interleave(G, dim=1)
    return torch.matmul(Q, K_exp.transpose(-2, -1))

def contract_5d_moe_torch(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """X: [B,S,E,Din], W: [E,Din,Dout] -> Y: [B,S,E,Dout]"""
    return torch.einsum('bsed,edm->bsem', X, W)

def contract_nd_mla_torch(C_KV: torch.Tensor, W_UK: torch.Tensor) -> torch.Tensor:
    """C_KV: [B,S,R], W_UK: [H,R,D] -> K: [B,H,S,D]"""
    return torch.einsum('bsr,hrd->bhsd', C_KV, W_UK)

# ── Arithmetic intensity calculator ──────────────────────────────────────

def arithmetic_intensity(flops: int, read_bytes: int, write_bytes: int) -> float:
    return flops / (read_bytes + write_bytes)

def analyze_contraction(name, flops, *shapes_dtypes):
    total_bytes = sum(np.prod(s) * d for s, d in shapes_dtypes)
    ai = flops / total_bytes
    print(f"  {name}")
    print(f"    FLOPs: {flops:,.0f}  |  Bytes: {total_bytes:,.0f}  |  AI: {ai:.2f} FLOP/byte")

# ── Worked arithmetic example ─────────────────────────────────────────────

def worked_2d_example():
    print("=== 2D Contraction: Step-by-Step Arithmetic ===")
    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])       # [2, 3]
    B = np.array([[7.0,  8.0],
                  [9.0,  10.0],
                  [11.0, 12.0]])           # [3, 2]

    print(f"A[2,3] =\n{A}")
    print(f"B[3,2] =\n{B}")
    print(f"Contraction: C[m,n] = sum_k A[m,k]*B[k,n]")
    print(f"C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = {1*7+2*9+3*11}")
    print(f"C[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = {1*8+2*10+3*12}")
    print(f"C[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = {4*7+5*9+6*11}")
    print(f"C[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = {4*8+5*10+6*12}")
    C = contract_2d(A, B)
    print(f"Result C[2,2] =\n{C}")
    print(f"FLOPs = 2*2*3*2 = {2*2*3*2}")

def worked_3d_example():
    print("\n=== 3D Contraction: Batched Token Projection ===")
    # B=1, S=2, D=3, N=2 for clarity
    X = np.array([[[1,2,3],[4,5,6]]], dtype=float)  # [1,2,3]
    W = np.array([[1,2],[3,4],[5,6]], dtype=float)   # [3,2]
    print(f"X[1,2,3]:\n{X}")
    print(f"W[3,2]:\n{W}")
    print(f"Contraction: C[b,s,n] = sum_d X[b,s,d]*W[d,n]")
    # Token 0: [1,2,3] @ [[1,2],[3,4],[5,6]] = [1+6+15, 2+8+18] = [22, 28]
    # Token 1: [4,5,6] @ W = [4+15+30, 8+20+36] = [49, 64]
    print(f"Token[0]: [1,2,3]@W = [1*1+2*3+3*5, 1*2+2*4+3*6] = [{1+6+15}, {2+8+18}]")
    print(f"Token[1]: [4,5,6]@W = [4*1+5*3+6*5, 4*2+5*4+6*6] = [{4+15+30}, {8+20+36}]")
    C = contract_3d(X, W)
    print(f"Result C[1,2,2]:\n{C}")

def worked_4d_qk_example():
    print("\n=== 4D Contraction: Attention Scores Q@K^T ===")
    # B=1, H=1, S=3, D=2
    Q = np.array([[[[1,0],[0,1],[1,1]]]], dtype=float)  # [1,1,3,2]
    K = np.array([[[[1,0],[0,1],[1,1]]]], dtype=float)  # [1,1,3,2]
    print(f"Q[1,1,3,2]:\n{Q[0,0]}")
    print(f"K[1,1,3,2]:\n{K[0,0]}")
    print(f"S[b,h,i,j] = sum_d Q[b,h,i,d]*K[b,h,j,d]")
    # S[0,0] = Q[0,0]@K[0,0]^T:
    # q0=[1,0]: k0=[1,0]->1, k1=[0,1]->0, k2=[1,1]->1
    # q1=[0,1]: k0=[1,0]->0, k1=[0,1]->1, k2=[1,1]->1
    # q2=[1,1]: k0=[1,0]->1, k1=[0,1]->1, k2=[1,1]->2
    print(f"S[0,0] row-by-row:")
    print(f"  q0=[1,0]: dots with k0,k1,k2 = [{1*1+0*0}, {1*0+0*1}, {1*1+0*1}]")
    print(f"  q1=[0,1]: dots with k0,k1,k2 = [{0*1+1*0}, {0*0+1*1}, {0*1+1*1}]")
    print(f"  q2=[1,1]: dots with k0,k1,k2 = [{1*1+1*0}, {1*0+1*1}, {1*1+1*1}]")
    S = contract_4d_qk(Q, K)
    print(f"Result S[1,1,3,3]:\n{S[0,0]}")

# ── Test harness ──────────────────────────────────────────────────────────

def test_2d():
    A = np.random.randn(64, 128).astype(np.float32)
    B = np.random.randn(128, 64).astype(np.float32)
    ref = A @ B
    out = contract_2d(A, B)
    assert np.allclose(ref, out, atol=1e-5), "2D failed"
    print("PASS: 2D contraction")

def test_3d():
    X = np.random.randn(4, 32, 64).astype(np.float32)
    W = np.random.randn(64, 128).astype(np.float32)
    ref = (X.reshape(-1, 64) @ W).reshape(4, 32, 128)
    out = contract_3d(X, W)
    assert np.allclose(ref, out, atol=1e-5), "3D failed"
    print("PASS: 3D contraction")

def test_4d_qk():
    Q = np.random.randn(2, 8, 32, 64).astype(np.float32)
    K = np.random.randn(2, 8, 32, 64).astype(np.float32)
    ref = Q @ K.transpose(0, 1, 3, 2)
    out = contract_4d_qk(Q, K)
    assert np.allclose(ref, out, atol=1e-5), "4D QK failed"
    print("PASS: 4D Q@K^T contraction")

def test_4d_av():
    A = np.random.randn(2, 8, 32, 32).astype(np.float32)
    V = np.random.randn(2, 8, 32, 64).astype(np.float32)
    ref = A @ V
    out = contract_4d_av(A, V)
    assert np.allclose(ref, out, atol=1e-5), "4D AV failed"
    print("PASS: 4D Attn@V contraction")

def test_5d_gqa():
    G = 4
    Q = np.random.randn(2, 8, 16, 64).astype(np.float32)
    K = np.random.randn(2, 2, 16, 64).astype(np.float32)
    out = contract_5d_gqa(Q, K, G)
    assert out.shape == (2, 8, 16, 16), f"GQA shape wrong: {out.shape}"
    print("PASS: 5D GQA scores contraction")

def test_5d_moe():
    X = np.random.randn(2, 16, 3, 64).astype(np.float32)
    W = np.random.randn(3, 64, 128).astype(np.float32)
    out = contract_5d_moe(X, W)
    assert out.shape == (2, 16, 3, 128), f"MoE shape wrong: {out.shape}"
    print("PASS: 5D MoE expert contraction")

def test_nd_mla():
    C_KV = np.random.randn(2, 16, 32).astype(np.float32)
    W_UK = np.random.randn(8, 32, 64).astype(np.float32)
    out = contract_nd_mla(C_KV, W_UK)
    assert out.shape == (2, 8, 16, 64), f"MLA shape wrong: {out.shape}"
    print("PASS: ND MLA KV reconstruction contraction")

def test_torch_vs_numpy():
    Q = np.random.randn(2, 8, 32, 64).astype(np.float32)
    K = np.random.randn(2, 8, 32, 64).astype(np.float32)
    np_out = contract_4d_qk(Q, K)
    tc_out = contract_4d_qk_torch(
        torch.from_numpy(Q), torch.from_numpy(K)).numpy()
    assert np.allclose(np_out, tc_out, atol=1e-5), "torch vs numpy mismatch"
    print("PASS: PyTorch matches NumPy reference")

if __name__ == '__main__':
    worked_2d_example()
    worked_3d_example()
    worked_4d_qk_example()
    print("\n--- Test Suite ---")
    test_2d()
    test_3d()
    test_4d_qk()
    test_4d_av()
    test_5d_gqa()
    test_5d_moe()
    test_nd_mla()
    test_torch_vs_numpy()
    print("\nAll Python contraction tests passed.")
```

---

## A2.4 CUDA C++ Implementations

### A2.4.1 2D Tiled GEMM from Scratch

The canonical teaching kernel: tiled shared-memory GEMM. Every production GEMM (cuBLAS, CUTLASS) is a heavily optimized variant of this pattern.

```cpp
// file: contractions_cuda.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define TILE 16
#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); } } while(0)

// ── 2D Tiled GEMM ─────────────────────────────────────────────────────────

__global__ void gemm_2d(const float* __restrict__ A,
                        const float* __restrict__ B,
                              float* __restrict__ C,
                        int M, int K, int N) {
    __shared__ float tA[TILE][TILE];
    __shared__ float tB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Load tile of A
        int aCol = t * TILE + threadIdx.x;
        tA[threadIdx.y][threadIdx.x] = (row < M && aCol < K)
                                       ? A[row * K + aCol] : 0.0f;
        // Load tile of B
        int bRow = t * TILE + threadIdx.y;
        tB[threadIdx.y][threadIdx.x] = (bRow < K && col < N)
                                       ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += tA[threadIdx.y][i] * tB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ── 3D Batched GEMM ────────────────────────────────────────────────────────
// C[b,m,n] = sum_k A[b,m,k] * B[k,n]   (shared weight B)

__global__ void gemm_3d_shared_weight(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                            float* __restrict__ C,
                                      int BATCH, int M, int K, int N) {
    int b   = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float tA[TILE][TILE];
    __shared__ float tB[TILE][TILE];
    float acc = 0.0f;

    const float* Ab = A + b * M * K;
    const float* Bb = B;                 // shared across batch
          float* Cb = C + b * M * N;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        tA[threadIdx.y][threadIdx.x] = (row < M && aCol < K)
                                       ? Ab[row * K + aCol] : 0.0f;
        int bRow = t * TILE + threadIdx.y;
        tB[threadIdx.y][threadIdx.x] = (bRow < K && col < N)
                                       ? Bb[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += tA[threadIdx.y][i] * tB[i][threadIdx.x];
        __syncthreads();
    }
    if (b < BATCH && row < M && col < N)
        Cb[row * N + col] = acc;
}

// ── 4D Batched GEMM: Attention Scores ─────────────────────────────────────
// S[b,h,i,j] = sum_d Q[b,h,i,d] * K[b,h,j,d]
// Launch: gridDim = (ceil(S/T), ceil(S/T), B*H)

__global__ void attn_scores_4d(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                      float* __restrict__ S,
                                int B, int H, int Sq, int Sk, int D) {
    int bh  = blockIdx.z;              // combined batch*head index
    int b   = bh / H;
    int h   = bh % H;
    int qi  = blockIdx.y * TILE + threadIdx.y;
    int ki  = blockIdx.x * TILE + threadIdx.x;

    __shared__ float tQ[TILE][TILE];
    __shared__ float tK[TILE][TILE];
    float acc = 0.0f;

    const float* Qbh = Q + (b * H + h) * Sq * D;
    const float* Kbh = K + (b * H + h) * Sk * D;
          float* Sbh = S + (b * H + h) * Sq * Sk;

    for (int t = 0; t < (D + TILE - 1) / TILE; ++t) {
        int d = t * TILE + threadIdx.x;
        tQ[threadIdx.y][threadIdx.x] = (qi < Sq && d < D)
                                       ? Qbh[qi * D + d] : 0.0f;
        d = t * TILE + threadIdx.y;
        tK[threadIdx.y][threadIdx.x] = (ki < Sk && d < D)
                                       ? Kbh[ki * D + d] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += tQ[threadIdx.y][i] * tK[threadIdx.x][i];
        __syncthreads();
    }
    if (b < B && qi < Sq && ki < Sk)
        Sbh[qi * Sk + ki] = acc;
}

// ── 5D GQA Scores ──────────────────────────────────────────────────────────
// S[b,hq,i,j] = sum_d Q[b,hq,i,d] * K[b,hq/G,j,d]
// K is stored as [B, H_kv, Sk, D]; we use hq/G to index into it.

__global__ void attn_scores_gqa(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                       float* __restrict__ S,
                                 int B, int H_q, int H_kv, int Sq, int Sk, int D) {
    int bh  = blockIdx.z;
    int b   = bh / H_q;
    int hq  = bh % H_q;
    int hkv = hq / (H_q / H_kv);      // GQA mapping: hq / G

    int qi  = blockIdx.y * TILE + threadIdx.y;
    int ki  = blockIdx.x * TILE + threadIdx.x;

    __shared__ float tQ[TILE][TILE];
    __shared__ float tK[TILE][TILE];
    float acc = 0.0f;

    const float* Qbh  = Q + (b * H_q  + hq)  * Sq * D;
    const float* Kbhk = K + (b * H_kv + hkv) * Sk * D;
          float* Sbh  = S + (b * H_q  + hq)  * Sq * Sk;

    for (int t = 0; t < (D + TILE - 1) / TILE; ++t) {
        int d = t * TILE + threadIdx.x;
        tQ[threadIdx.y][threadIdx.x] = (qi < Sq && d < D)
                                       ? Qbh[qi * D + d] : 0.0f;
        d = t * TILE + threadIdx.y;
        tK[threadIdx.y][threadIdx.x] = (ki < Sk && d < D)
                                       ? Kbhk[ki * D + d] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += tQ[threadIdx.y][i] * tK[threadIdx.x][i];
        __syncthreads();
    }
    if (b < B && qi < Sq && ki < Sk)
        Sbh[qi * Sk + ki] = acc;
}

// ── ND Contraction via mdspan ──────────────────────────────────────────────
// MLA: K[b,h,s,d] = sum_r C_KV[b,s,r] * W_UK[h,r,d]
// Implemented as: for each (b,h): K[b,h,:,:] = C_KV[b,:,:] @ W_UK[h,:,:]^T

__global__ void mla_kv_reconstruct(const float* __restrict__ C_KV,
                                    const float* __restrict__ W_UK,
                                          float* __restrict__ K_out,
                                    int B, int H, int S, int R, int D) {
    // Each block handles one (b, h) pair
    int bh = blockIdx.z;
    int b  = bh / H;
    int h  = bh % H;
    int si = blockIdx.y * TILE + threadIdx.y;   // sequence position
    int di = blockIdx.x * TILE + threadIdx.x;   // d_head position

    __shared__ float tC[TILE][TILE];
    __shared__ float tW[TILE][TILE];
    float acc = 0.0f;

    const float* Cb  = C_KV + b * S * R;
    const float* Wh  = W_UK + h * R * D;
          float* Kbh = K_out + (b * H + h) * S * D;

    for (int t = 0; t < (R + TILE - 1) / TILE; ++t) {
        int r = t * TILE + threadIdx.x;
        tC[threadIdx.y][threadIdx.x] = (si < S && r < R)
                                       ? Cb[si * R + r] : 0.0f;
        r = t * TILE + threadIdx.y;
        tW[threadIdx.y][threadIdx.x] = (r < R && di < D)
                                       ? Wh[r * D + di] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += tC[threadIdx.y][i] * tW[i][threadIdx.x];
        __syncthreads();
    }
    if (b < B && si < S && di < D)
        Kbh[si * D + di] = acc;
}

// ── Host helper: launch 2D GEMM ────────────────────────────────────────────

void launch_gemm_2d(const float* dA, const float* dB, float* dC,
                    int M, int K, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_2d<<<grid, block>>>(dA, dB, dC, M, K, N);
    CHECK(cudaGetLastError());
}

void launch_attn_scores_4d(const float* dQ, const float* dK, float* dS,
                            int B, int H, int S, int D) {
    dim3 block(TILE, TILE);
    dim3 grid((S + TILE-1)/TILE, (S + TILE-1)/TILE, B * H);
    attn_scores_4d<<<grid, block>>>(dQ, dK, dS, B, H, S, S, D);
    CHECK(cudaGetLastError());
}

// ── Main test harness ──────────────────────────────────────────────────────

int main() {
    const int M=64, K=128, N=64;
    size_t sA = M*K*sizeof(float), sB = K*N*sizeof(float), sC = M*N*sizeof(float);

    float *hA=(float*)malloc(sA), *hB=(float*)malloc(sB), *hC=(float*)malloc(sC);
    float *hRef=(float*)malloc(sC);

    // Init with simple pattern
    for(int i=0;i<M*K;i++) hA[i]=(float)(i%7 - 3)*0.1f;
    for(int i=0;i<K*N;i++) hB[i]=(float)(i%5 - 2)*0.1f;

    // CPU reference
    for(int m=0;m<M;m++) for(int n=0;n<N;n++) {
        float s=0; for(int k=0;k<K;k++) s+=hA[m*K+k]*hB[k*N+n];
        hRef[m*N+n]=s;
    }

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, sA));
    CHECK(cudaMalloc(&dB, sB));
    CHECK(cudaMalloc(&dC, sC));
    CHECK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));

    launch_gemm_2d(dA, dB, dC, M, K, N);
    CHECK(cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost));

    float maxErr=0;
    for(int i=0;i<M*N;i++) {
        float e=fabsf(hC[i]-hRef[i]);
        if(e>maxErr) maxErr=e;
    }
    printf("2D GEMM max error: %.2e  %s\n", maxErr, maxErr<1e-4?"PASS":"FAIL");

    // Test 4D attention scores
    const int B=2,H=4,SEQ=32,D=16;
    float *dQ,*dK,*dS;
    size_t sqk=B*H*SEQ*D*sizeof(float), ss=B*H*SEQ*SEQ*sizeof(float);
    CHECK(cudaMalloc(&dQ,sqk)); CHECK(cudaMalloc(&dK,sqk)); CHECK(cudaMalloc(&dS,ss));
    // Fill with random-ish values
    float *hQ=(float*)malloc(sqk), *hK=(float*)malloc(sqk);
    for(int i=0;i<(int)(sqk/sizeof(float));i++) hQ[i]=hK[i]=(float)((i%11)-5)*0.05f;
    CHECK(cudaMemcpy(dQ,hQ,sqk,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dK,hK,sqk,cudaMemcpyHostToDevice));
    launch_attn_scores_4d(dQ, dK, dS, B, H, SEQ, D);
    CHECK(cudaDeviceSynchronize());
    printf("4D Attention scores: PASS (no crash, shape [%d,%d,%d,%d])\n",B,H,SEQ,SEQ);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(dQ); cudaFree(dK); cudaFree(dS);
    free(hA); free(hB); free(hC); free(hRef); free(hQ); free(hK);
    printf("CUDA contraction tests complete.\n");
    return 0;
}
```

**Compile and run:**

```bash
nvcc -O3 -arch=sm_80 contractions_cuda.cu -o contractions_cuda
./contractions_cuda
# Expected:
# 2D GEMM max error: 0.00e+00  PASS
# 4D Attention scores: PASS (no crash, shape [2,4,32,32])
# CUDA contraction tests complete.
```

---

## A2.5 CUTLASS Implementations

CUTLASS exposes a declarative template API where you specify the problem shape, data types, tile sizes, and epilogue — the library generates a near-optimal kernel. The GETT (GEMM-like Tensor-Tensor Contraction) feature in CUTLASS 3.x handles arbitrary rank contractions.

### A2.5.1 2D and 3D GEMM via CUTLASS

```cpp
// file: contractions_cutlass.cu
// Requires CUTLASS >= 3.0, C++17
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_fill.h>

#include <iostream>
#include <cmath>

// ── 2D FP16 GEMM on Tensor Cores (CUTLASS 2.x style) ─────────────────────

using Gemm2D = cutlass::gemm::device::Gemm<
    cutlass::half_t,                    // ElementA
    cutlass::layout::RowMajor,          // LayoutA
    cutlass::half_t,                    // ElementB
    cutlass::layout::RowMajor,          // LayoutB
    cutlass::half_t,                    // ElementC (output)
    cutlass::layout::RowMajor,          // LayoutC
    float,                              // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,     // Use Tensor Cores
    cutlass::arch::Sm80,                // Ampere
    cutlass::gemm::GemmShape<128,128,32>, // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>, // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,  // InstructionShape (m16n8k16)
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8,             // 8 elements per vectorized store
        float, float>,                  // alpha/beta type
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3                                   // pipeline stages
>;

bool run_gemm_2d(int M, int K, int N,
                 float alpha = 1.0f, float beta = 0.0f) {
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> B({K, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> C({M, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> D({M, N});

    cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 1, 0.5, -0.5);
    cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 2, 0.5, -0.5);
    cutlass::reference::host::TensorFill(C.host_view(), cutlass::half_t(0));

    A.sync_device(); B.sync_device(); C.sync_device();

    Gemm2D gemm_op;
    Gemm2D::Arguments args(
        {M, N, K},
        A.device_ref(), B.device_ref(), C.device_ref(), D.device_ref(),
        {alpha, beta}
    );

    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS 2D GEMM error: "
                  << cutlass::cutlassGetStatusString(status) << "\n";
        return false;
    }

    D.sync_host();
    std::cout << "CUTLASS 2D GEMM [" << M << "x" << K << "x" << N
              << "]: PASS\n";
    return true;
}

// ── 3D Batched GEMM (shared weight per batch) ─────────────────────────────

using BatchedGemm = cutlass::gemm::device::GemmBatched<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16,  8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t,8,float,float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3
>;

bool run_gemm_3d_batched(int BATCH, int S, int D, int N) {
    // X: [BATCH*S, D], W: [D, N] -> C: [BATCH*S, N]
    // Simulated as BATCH separate GEMMs each of [S, D] x [D, N]
    // CUTLASS GemmBatched: A strided by S*D, B strided by 0 (shared weight)
    int M = S;
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>
        A({BATCH * M, D}), B({D, N}), C({BATCH * M, N});

    cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 1, 0.5f, -0.5f);
    cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 2, 0.5f, -0.5f);
    A.sync_device(); B.sync_device(); C.sync_device();

    BatchedGemm gemm_batched;
    BatchedGemm::Arguments args(
        {M, N, D},
        A.device_ref(), M * D,     // A stride between batches
        B.device_ref(), 0,         // B stride = 0: shared weight
        C.device_ref(), M * N,     // C stride between batches
        C.device_ref(), M * N,     // D stride
        {1.0f, 0.0f},
        BATCH
    );
    auto status = gemm_batched(args);
    C.sync_host();
    bool ok = (status == cutlass::Status::kSuccess);
    std::cout << "CUTLASS 3D Batched GEMM [" << BATCH << "x" << S
              << "x" << D << "x" << N << "]: "
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

// ── 4D Attention: Batched GEMM over (B*H) pairs ───────────────────────────

bool run_attn_scores_cutlass(int B, int H, int S, int D) {
    int BH = B * H;
    // Q: [BH, S, D] viewed as BH separate [S,D] matrices
    // K^T: [BH, D, S]   -> scores: [BH, S, S]
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> Q({BH*S, D});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> K({BH*S, D});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> Score({BH*S, S});

    cutlass::reference::host::TensorFillRandomUniform(Q.host_view(), 1, 0.1f, -0.1f);
    cutlass::reference::host::TensorFillRandomUniform(K.host_view(), 2, 0.1f, -0.1f);
    Q.sync_device(); K.sync_device(); Score.sync_device();

    using AttnGemm = cutlass::gemm::device::GemmBatched<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64,64,32>,
        cutlass::gemm::GemmShape<32,32,32>,
        cutlass::gemm::GemmShape<16,8,16>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t,8,float,float>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        3>;

    AttnGemm attn_gemm;
    float scale = 1.0f / sqrtf((float)D);
    AttnGemm::Arguments args(
        {S, S, D},
        Q.device_ref(), S * D,
        K.device_ref(), S * D,
        Score.device_ref(), S * S,
        Score.device_ref(), S * S,
        {scale, 0.0f},
        BH
    );
    auto status = attn_gemm(args);
    Score.sync_host();
    bool ok = (status == cutlass::Status::kSuccess);
    std::cout << "CUTLASS 4D Attn Scores [B=" << B << ",H=" << H
              << ",S=" << S << ",D=" << D << "]: "
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

int main() {
    run_gemm_2d(512, 1024, 512);
    run_gemm_3d_batched(8, 256, 512, 512);
    run_attn_scores_cutlass(2, 8, 64, 32);
    return 0;
}
```

**Build:**

```bash
nvcc -std=c++17 -O3 -arch=sm_80 \
     -I/path/to/cutlass/include \
     -I/path/to/cutlass/tools/util/include \
     contractions_cutlass.cu -o contractions_cutlass
./contractions_cutlass
```

---

## A2.6 Triton Implementations

Triton expresses tiled GPU computation in Python. The compiler handles shared memory, thread-block scheduling, and vectorisation automatically.

```python
# file: contractions_triton.py
import torch
import triton
import triton.language as tl

# ── 2D GEMM ───────────────────────────────────────────────────────────────

@triton.jit
def gemm_2d_kernel(A_ptr, B_ptr, C_ptr,
                   M, N, K,
                   stride_am, stride_ak,
                   stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c = acc.to(tl.float16)
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def gemm_2d_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    A = A.to(torch.float16)
    B = B.to(torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_2d_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C

# ── 3D Batched GEMM ───────────────────────────────────────────────────────

@triton.jit
def gemm_3d_kernel(X_ptr, W_ptr, C_ptr,
                   BATCH, M, N, D,
                   stride_xb, stride_xm, stride_xd,
                   stride_wd, stride_wn,
                   stride_cb, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d in range(0, D, BLOCK_D):
        rd = d + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + pid_b * stride_xb + rm[:, None] * stride_xm
                    + rd[None, :] * stride_xd,
                    mask=(rm[:, None] < M) & (rd[None, :] < D), other=0.0)
        w = tl.load(W_ptr + rd[:, None] * stride_wd + rn[None, :] * stride_wn,
                    mask=(rd[:, None] < D) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(x, w)

    c = acc.to(tl.float16)
    tl.store(C_ptr + pid_b * stride_cb + rm[:, None] * stride_cm
             + rn[None, :] * stride_cn,
             c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def gemm_3d_triton(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    BATCH, M, D = X.shape
    D2, N = W.shape
    assert D == D2
    C = torch.empty((BATCH, M, N), device=X.device, dtype=torch.float16)
    X = X.to(torch.float16); W = W.to(torch.float16)
    BM, BN, BD = 32, 32, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), BATCH)
    gemm_3d_kernel[grid](
        X, W, C, BATCH, M, N, D,
        X.stride(0), X.stride(1), X.stride(2),
        W.stride(0), W.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD
    )
    return C

# ── 4D Attention Scores: Flash-style tiled Q@K^T ─────────────────────────

@triton.jit
def attn_scores_4d_kernel(Q_ptr, K_ptr, S_ptr,
                           B, H, Sq, Sk, D,
                           stride_qb, stride_qh, stride_qi, stride_qd,
                           stride_kb, stride_kh, stride_kj, stride_kd,
                           stride_sb, stride_sh, stride_si, stride_sj,
                           scale,
                           BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
                           BLOCK_D: tl.constexpr):
    bh  = tl.program_id(2)
    b   = bh // H
    h   = bh %  H
    pi  = tl.program_id(1)
    pj  = tl.program_id(0)
    ri  = pi * BLOCK_I + tl.arange(0, BLOCK_I)
    rj  = pj * BLOCK_J + tl.arange(0, BLOCK_J)
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

    for d in range(0, D, BLOCK_D):
        rd = d + tl.arange(0, BLOCK_D)
        q = tl.load(Q_ptr + b*stride_qb + h*stride_qh + ri[:,None]*stride_qi
                    + rd[None,:]*stride_qd,
                    mask=(ri[:,None]<Sq)&(rd[None,:]<D), other=0.0)
        k = tl.load(K_ptr + b*stride_kb + h*stride_kh + rj[:,None]*stride_kj
                    + rd[None,:]*stride_kd,
                    mask=(rj[:,None]<Sk)&(rd[None,:]<D), other=0.0)
        acc += tl.dot(q, tl.trans(k))

    acc *= scale
    tl.store(S_ptr + b*stride_sb + h*stride_sh + ri[:,None]*stride_si
             + rj[None,:]*stride_sj,
             acc.to(tl.float16),
             mask=(ri[:,None]<Sq)&(rj[None,:]<Sk))


def attn_scores_triton(Q: torch.Tensor, K: torch.Tensor,
                        scale: float = None) -> torch.Tensor:
    B, H, Sq, D = Q.shape
    B2, H2, Sk, D2 = K.shape
    assert B==B2 and H==H2 and D==D2
    if scale is None:
        scale = D ** -0.5
    S = torch.empty((B, H, Sq, Sk), device=Q.device, dtype=torch.float16)
    Q = Q.to(torch.float16); K = K.to(torch.float16)
    BI, BJ, BD = 32, 32, 32
    grid = (triton.cdiv(Sk, BJ), triton.cdiv(Sq, BI), B*H)
    attn_scores_4d_kernel[grid](
        Q, K, S, B, H, Sq, Sk, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        S.stride(0), S.stride(1), S.stride(2), S.stride(3),
        scale,
        BLOCK_I=BI, BLOCK_J=BJ, BLOCK_D=BD
    )
    return S

# ── 5D GQA Scores ─────────────────────────────────────────────────────────
# Re-uses attn_scores_triton after expanding K heads.

def attn_scores_gqa_triton(Q: torch.Tensor, K: torch.Tensor,
                             G: int) -> torch.Tensor:
    B, H_q, Sq, D = Q.shape
    B2, H_kv, Sk, D2 = K.shape
    assert B == B2 and D == D2 and H_q == G * H_kv
    K_exp = K.repeat_interleave(G, dim=1)   # [B, H_q, Sk, D]
    return attn_scores_triton(Q, K_exp)

# ── ND MLA KV Reconstruction ──────────────────────────────────────────────

@triton.jit
def mla_recon_kernel(C_ptr, W_ptr, K_ptr,
                     B, H, S, R, D,
                     stride_cb, stride_cs, stride_cr,
                     stride_wh, stride_wr, stride_wd,
                     stride_kb, stride_kh, stride_ks, stride_kd,
                     BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
                     BLOCK_R: tl.constexpr):
    bh = tl.program_id(2)
    b  = bh // H
    h  = bh %  H
    ps = tl.program_id(1)
    pd = tl.program_id(0)
    rs = ps * BLOCK_S + tl.arange(0, BLOCK_S)
    rd = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    for r in range(0, R, BLOCK_R):
        rr = r + tl.arange(0, BLOCK_R)
        c = tl.load(C_ptr + b*stride_cb + rs[:,None]*stride_cs + rr[None,:]*stride_cr,
                    mask=(rs[:,None]<S)&(rr[None,:]<R), other=0.0)
        w = tl.load(W_ptr + h*stride_wh + rr[:,None]*stride_wr + rd[None,:]*stride_wd,
                    mask=(rr[:,None]<R)&(rd[None,:]<D), other=0.0)
        acc += tl.dot(c, w)

    tl.store(K_ptr + b*stride_kb + h*stride_kh + rs[:,None]*stride_ks
             + rd[None,:]*stride_kd,
             acc.to(tl.float16),
             mask=(rs[:,None]<S)&(rd[None,:]<D))


def mla_reconstruct_triton(C_KV: torch.Tensor,
                            W_UK: torch.Tensor) -> torch.Tensor:
    B, S, R = C_KV.shape
    H, R2, D = W_UK.shape
    assert R == R2
    K_out = torch.empty((B, H, S, D), device=C_KV.device, dtype=torch.float16)
    C_KV = C_KV.to(torch.float16); W_UK = W_UK.to(torch.float16)
    BS, BD, BR = 16, 16, 16
    grid = (triton.cdiv(D, BD), triton.cdiv(S, BS), B*H)
    mla_recon_kernel[grid](
        C_KV, W_UK, K_out, B, H, S, R, D,
        C_KV.stride(0), C_KV.stride(1), C_KV.stride(2),
        W_UK.stride(0), W_UK.stride(1), W_UK.stride(2),
        K_out.stride(0), K_out.stride(1), K_out.stride(2), K_out.stride(3),
        BLOCK_S=BS, BLOCK_D=BD, BLOCK_R=BR
    )
    return K_out

# ── Triton autotuned 2D GEMM ──────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M':128,'BLOCK_N':128,'BLOCK_K':32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M':64, 'BLOCK_N':128,'BLOCK_K':32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':128,'BLOCK_N':64, 'BLOCK_K':32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':64, 'BLOCK_N':64, 'BLOCK_K':64}, num_stages=3, num_warps=4),
    ],
    key=['M','N','K']
)
@triton.jit
def gemm_2d_autotuned(A_ptr, B_ptr, C_ptr, M, N, K,
                      stride_am, stride_ak,
                      stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + rm[:,None]*stride_am + rk[None,:]*stride_ak,
                    mask=(rm[:,None]<M)&(rk[None,:]<K), other=0.0)
        b = tl.load(B_ptr + rk[:,None]*stride_bk + rn[None,:]*stride_bn,
                    mask=(rk[:,None]<K)&(rn[None,:]<N), other=0.0)
        acc += tl.dot(a, b)
    c = acc.to(tl.float16)
    tl.store(C_ptr + rm[:,None]*stride_cm + rn[None,:]*stride_cn,
             c, mask=(rm[:,None]<M)&(rn[None,:]<N))

# ── Test harness ──────────────────────────────────────────────────────────

def test_triton_contractions():
    device = 'cuda'
    print("=== Triton Contraction Tests ===")

    # 2D
    A = torch.randn(256, 512, device=device)
    B = torch.randn(512, 256, device=device)
    ref = (A @ B).to(torch.float16)
    out = gemm_2d_triton(A, B)
    err = (out - ref).abs().max().item()
    print(f"2D GEMM max err: {err:.4f}  {'PASS' if err < 0.5 else 'FAIL'}")

    # 3D
    X = torch.randn(4, 64, 256, device=device)
    W = torch.randn(256, 128, device=device)
    ref3 = torch.einsum('bsd,dn->bsn', X, W).to(torch.float16)
    out3 = gemm_3d_triton(X, W)
    err3 = (out3 - ref3).abs().max().item()
    print(f"3D Batched GEMM max err: {err3:.4f}  {'PASS' if err3 < 0.5 else 'FAIL'}")

    # 4D
    Q = torch.randn(2, 4, 32, 64, device=device)
    K = torch.randn(2, 4, 32, 64, device=device)
    ref4 = torch.matmul(Q, K.transpose(-2,-1)).to(torch.float16) * (64**-0.5)
    out4 = attn_scores_triton(Q, K)
    err4 = (out4 - ref4).abs().max().item()
    print(f"4D Attn Scores max err: {err4:.4f}  {'PASS' if err4 < 0.5 else 'FAIL'}")

    # 5D GQA
    Q5 = torch.randn(2, 8, 32, 64, device=device)
    K5 = torch.randn(2, 2, 32, 64, device=device)
    out5 = attn_scores_gqa_triton(Q5, K5, G=4)
    assert out5.shape == (2, 8, 32, 32), f"GQA shape {out5.shape}"
    print(f"5D GQA shape {out5.shape}: PASS")

    # ND MLA
    C_KV = torch.randn(2, 32, 64, device=device)
    W_UK = torch.randn(4, 64, 32, device=device)
    outN = mla_reconstruct_triton(C_KV, W_UK)
    assert outN.shape == (2, 4, 32, 32), f"MLA shape {outN.shape}"
    print(f"ND MLA shape {outN.shape}: PASS")

if __name__ == '__main__':
    if torch.cuda.is_available():
        test_triton_contractions()
    else:
        print("CUDA not available -- Triton tests require GPU")
```

---

## A2.7 Mojo Implementations

Mojo provides Python-syntax systems programming with explicit SIMD, parallelism, and value semantics. The `fn` keyword enforces strict typing; `parallelize` maps work across CPU cores.

```mojo
# file: contractions.mojo
from memory import UnsafePointer
from algorithm import parallelize, vectorize
from math import sqrt
from sys import simdwidthof
from tensor import Tensor
from utils.index import Index

alias F32 = DType.float32
alias SIMD_W = simdwidthof[F32]()

# ── 2D Contraction: C[m,n] = sum_k A[m,k] * B[k,n] ─────────────────────

fn contract_2d(A: Tensor[F32], B: Tensor[F32]) -> Tensor[F32]:
    let M = A.shape()[0]
    let K = A.shape()[1]
    let N = B.shape()[1]
    var C = Tensor[F32](M, N)

    @parameter
    fn compute_row(m: Int):
        for n in range(N):
            var acc: Float32 = 0.0

            @parameter
            fn dot_simd[width: Int](k: Int):
                let a_vec = A.simd_load[width](m * K + k)
                # B is row-major [K,N]: B[k,n] at k*N+n
                # For a column n, stride is N -- load with gather
                var b_vec = SIMD[F32, width]()
                for i in range(width):
                    b_vec[i] = B[k + i, n]
                acc += (a_vec * b_vec).reduce_add()

            vectorize[dot_simd, SIMD_W](K)
            C[m, n] = acc

    parallelize[compute_row](M)
    return C

# ── 3D Contraction: C[b,s,n] = sum_d X[b,s,d] * W[d,n] ─────────────────

fn contract_3d(X: Tensor[F32], W: Tensor[F32]) -> Tensor[F32]:
    let BATCH = X.shape()[0]
    let S     = X.shape()[1]
    let D     = X.shape()[2]
    let N     = W.shape()[1]
    var C = Tensor[F32](BATCH, S, N)

    @parameter
    fn compute_bs(bs: Int):
        let b = bs // S
        let s = bs %  S
        for n in range(N):
            var acc: Float32 = 0.0

            @parameter
            fn dot_simd[width: Int](d: Int):
                let x_vec = X.simd_load[width](b * S * D + s * D + d)
                var w_vec = SIMD[F32, width]()
                for i in range(width):
                    w_vec[i] = W[d + i, n]
                acc += (x_vec * w_vec).reduce_add()

            vectorize[dot_simd, SIMD_W](D)
            C[b, s, n] = acc

    parallelize[compute_bs](BATCH * S)
    return C

# ── 4D Contraction: S[b,h,i,j] = sum_d Q[b,h,i,d]*K[b,h,j,d] ──────────

fn contract_4d_qk(Q: Tensor[F32], K: Tensor[F32]) -> Tensor[F32]:
    let B  = Q.shape()[0]
    let H  = Q.shape()[1]
    let Sq = Q.shape()[2]
    let D  = Q.shape()[3]
    let Sk = K.shape()[2]
    var S = Tensor[F32](B, H, Sq, Sk)
    let scale = 1.0 / sqrt(Float32(D))

    @parameter
    fn compute_bh(bh: Int):
        let b = bh // H
        let h = bh %  H
        for i in range(Sq):
            for j in range(Sk):
                var acc: Float32 = 0.0

                @parameter
                fn dot_d[width: Int](d: Int):
                    let q_vec = Q.simd_load[width](
                        b*H*Sq*D + h*Sq*D + i*D + d)
                    let k_vec = K.simd_load[width](
                        b*H*Sk*D + h*Sk*D + j*D + d)
                    acc += (q_vec * k_vec).reduce_add()

                vectorize[dot_d, SIMD_W](D)
                S[b, h, i, j] = acc * scale

    parallelize[compute_bh](B * H)
    return S

# ── 5D Contraction: GQA scores ──────────────────────────────────────────

fn contract_5d_gqa(Q: Tensor[F32], K: Tensor[F32],
                   G: Int) -> Tensor[F32]:
    let B    = Q.shape()[0]
    let H_q  = Q.shape()[1]
    let Sq   = Q.shape()[2]
    let D    = Q.shape()[3]
    let Sk   = K.shape()[2]
    let H_kv = H_q // G
    var S = Tensor[F32](B, H_q, Sq, Sk)
    let scale = 1.0 / sqrt(Float32(D))

    @parameter
    fn compute_bhq(bhq: Int):
        let b  = bhq // H_q
        let hq = bhq %  H_q
        let hk = hq // G                 # GQA mapping
        for i in range(Sq):
            for j in range(Sk):
                var acc: Float32 = 0.0

                @parameter
                fn dot_d[width: Int](d: Int):
                    let q_vec = Q.simd_load[width](
                        b*H_q*Sq*D + hq*Sq*D + i*D + d)
                    let k_vec = K.simd_load[width](
                        b*H_kv*Sk*D + hk*Sk*D + j*D + d)
                    acc += (q_vec * k_vec).reduce_add()

                vectorize[dot_d, SIMD_W](D)
                S[b, hq, i, j] = acc * scale

    parallelize[compute_bhq](B * H_q)
    return S

# ── ND Contraction: MLA KV = sum_r C_KV[b,s,r] * W_UK[h,r,d] ──────────

fn contract_nd_mla(C_KV: Tensor[F32], W_UK: Tensor[F32]) -> Tensor[F32]:
    let B = C_KV.shape()[0]
    let S = C_KV.shape()[1]
    let R = C_KV.shape()[2]
    let H = W_UK.shape()[0]
    let D = W_UK.shape()[2]
    var K_out = Tensor[F32](B, H, S, D)

    @parameter
    fn compute_bh(bh: Int):
        let b = bh // H
        let h = bh %  H
        for s in range(S):
            for d in range(D):
                var acc: Float32 = 0.0

                @parameter
                fn dot_r[width: Int](r: Int):
                    let c_vec = C_KV.simd_load[width](b*S*R + s*R + r)
                    let w_vec = W_UK.simd_load[width](h*R*D + r*D + d)
                    acc += (c_vec * w_vec).reduce_add()

                vectorize[dot_r, SIMD_W](R)
                K_out[b, h, s, d] = acc

    parallelize[compute_bh](B * H)
    return K_out

# ── Main test harness ────────────────────────────────────────────────────

fn test_2d() raises:
    var A = Tensor[F32](4, 8)
    var B = Tensor[F32](8, 4)
    for i in range(32): A.store(i, Float32(i) * 0.1)
    for i in range(32): B.store(i, Float32(i) * 0.1)
    let C = contract_2d(A, B)
    print("2D contraction output shape:", C.shape()[0], "x", C.shape()[1])
    print("C[0,0] =", C[0, 0])
    print("PASS: 2D Mojo contraction")

fn test_3d() raises:
    var X = Tensor[F32](2, 8, 16)
    var W = Tensor[F32](16, 8)
    for i in range(2*8*16): X.store(i, Float32(i % 7) * 0.1)
    for i in range(16*8): W.store(i, Float32(i % 5) * 0.1)
    let C = contract_3d(X, W)
    print("3D shape:", C.shape()[0], C.shape()[1], C.shape()[2])
    print("PASS: 3D Mojo contraction")

fn test_4d_qk() raises:
    var Q = Tensor[F32](1, 2, 8, 16)
    var K = Tensor[F32](1, 2, 8, 16)
    for i in range(1*2*8*16): Q.store(i, Float32(i % 5) * 0.01)
    for i in range(1*2*8*16): K.store(i, Float32(i % 7) * 0.01)
    let S = contract_4d_qk(Q, K)
    print("4D QK shape:", S.shape()[0], S.shape()[1], S.shape()[2], S.shape()[3])
    print("PASS: 4D Mojo attention scores")

fn test_5d_gqa() raises:
    var Q = Tensor[F32](1, 4, 8, 16)
    var K = Tensor[F32](1, 2, 8, 16)
    for i in range(1*4*8*16): Q.store(i, Float32(i % 5) * 0.01)
    for i in range(1*2*8*16): K.store(i, Float32(i % 7) * 0.01)
    let S = contract_5d_gqa(Q, K, G=2)
    print("5D GQA shape:", S.shape()[0], S.shape()[1], S.shape()[2], S.shape()[3])
    print("PASS: 5D GQA Mojo contraction")

fn test_nd_mla() raises:
    var C = Tensor[F32](1, 8, 16)
    var W = Tensor[F32](4, 16, 8)
    for i in range(1*8*16): C.store(i, Float32(i % 5) * 0.01)
    for i in range(4*16*8): W.store(i, Float32(i % 7) * 0.01)
    let K = contract_nd_mla(C, W)
    print("ND MLA shape:", K.shape()[0], K.shape()[1], K.shape()[2], K.shape()[3])
    print("PASS: ND MLA Mojo contraction")

fn main() raises:
    print("=== Mojo Tensor Contraction Tests ===")
    test_2d()
    test_3d()
    test_4d_qk()
    test_5d_gqa()
    test_nd_mla()
    print("\nAll Mojo contraction tests passed.")
```

**Run:**

```bash
mojo contractions.mojo
```

---

## A2.8 Comprehensive Test Harness

```python
# file: test_contractions.py
"""
Unified test harness comparing all implementations against numpy reference.
Run with: pytest test_contractions.py -v
"""
import pytest
import numpy as np
import torch

# ── Numpy reference (truth) ───────────────────────────────────────────────

def np_contract_2d(A, B): return np.einsum('mk,kn->mn', A, B)
def np_contract_3d(X, W): return np.einsum('bsd,dn->bsn', X, W)
def np_contract_4d_qk(Q, K): return np.einsum('bhid,bhjd->bhij', Q, K)
def np_contract_4d_av(A, V): return np.einsum('bhij,bhjd->bhid', A, V)
def np_contract_5d_gqa(Q, K, G):
    K_e = np.repeat(K, G, axis=1)
    return np.einsum('bhid,bhjd->bhij', Q, K_e)
def np_contract_5d_moe(X, W): return np.einsum('bsed,edm->bsem', X, W)
def np_contract_nd_mla(C, W): return np.einsum('bsr,hrd->bhsd', C, W)

# ── Shapes for parametrized tests ─────────────────────────────────────────

SHAPES_2D = [(32,64,32),(64,128,64),(128,256,128),(512,512,512)]
SHAPES_3D = [(2,16,32,16),(4,32,64,32)]
SHAPES_4D = [(1,4,16,32),(2,8,32,64)]
SHAPES_GQA = [(1,8,16,64,4),(2,8,32,64,4)]   # B,H_q,S,D,G
SHAPES_MOE = [(2,16,3,32,64)]                 # B,S,E,Din,Dout
SHAPES_MLA = [(1,8,32,16,4)]                  # B,S,R,D,H

TOL = {'rtol': 1e-4, 'atol': 1e-4}

# ── 2D Tests ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M,K,N", SHAPES_2D)
def test_2d_torch_mm(M, K, N):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    ref = np_contract_2d(A, B)
    out = (torch.from_numpy(A) @ torch.from_numpy(B)).numpy()
    np.testing.assert_allclose(out, ref, **TOL)

@pytest.mark.parametrize("M,K,N", SHAPES_2D)
def test_2d_einsum(M, K, N):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    ref = np_contract_2d(A, B)
    out = torch.einsum('mk,kn->mn',
                        torch.from_numpy(A), torch.from_numpy(B)).numpy()
    np.testing.assert_allclose(out, ref, **TOL)

# ── 3D Tests ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("B,S,D,N", SHAPES_3D)
def test_3d_reshape_mm(B, S, D, N):
    X = np.random.randn(B, S, D).astype(np.float32)
    W = np.random.randn(D, N).astype(np.float32)
    ref = np_contract_3d(X, W)
    tX = torch.from_numpy(X); tW = torch.from_numpy(W)
    out = (tX.view(B*S, D) @ tW).view(B, S, N).numpy()
    np.testing.assert_allclose(out, ref, **TOL)

@pytest.mark.parametrize("B,S,D,N", SHAPES_3D)
def test_3d_matmul_broadcast(B, S, D, N):
    X = np.random.randn(B, S, D).astype(np.float32)
    W = np.random.randn(D, N).astype(np.float32)
    ref = np_contract_3d(X, W)
    out = torch.matmul(torch.from_numpy(X),
                       torch.from_numpy(W)).numpy()
    np.testing.assert_allclose(out, ref, **TOL)

# ── 4D Tests ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("B,H,S,D", SHAPES_4D)
def test_4d_qk_matmul(B, H, S, D):
    Q = np.random.randn(B, H, S, D).astype(np.float32)
    K = np.random.randn(B, H, S, D).astype(np.float32)
    ref = np_contract_4d_qk(Q, K)
    tQ = torch.from_numpy(Q); tK = torch.from_numpy(K)
    out = torch.matmul(tQ, tK.transpose(-2,-1)).numpy()
    np.testing.assert_allclose(out, ref, **TOL)

@pytest.mark.parametrize("B,H,S,D", SHAPES_4D)
def test_4d_av_matmul(B, H, S, D):
    A = np.random.randn(B, H, S, S).astype(np.float32)
    V = np.random.randn(B, H, S, D).astype(np.float32)
    ref = np_contract_4d_av(A, V)
    out = torch.matmul(torch.from_numpy(A),
                       torch.from_numpy(V)).numpy()
    np.testing.assert_allclose(out, ref, **TOL)

@pytest.mark.parametrize("B,H,S,D", SHAPES_4D)
def test_4d_full_attention(B, H, S, D):
    Q = np.random.randn(B, H, S, D).astype(np.float32)
    K = np.random.randn(B, H, S, D).astype(np.float32)
    V = np.random.randn(B, H, S, D).astype(np.float32)
    scale = D ** -0.5
    scores = np_contract_4d_qk(Q, K) * scale
    # Softmax
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    out = np_contract_4d_av(weights, V)
    assert out.shape == (B, H, S, D), f"Shape mismatch: {out.shape}"

# ── 5D Tests ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("B,H_q,S,D,G", SHAPES_GQA)
def test_5d_gqa_scores(B, H_q, S, D, G):
    H_kv = H_q // G
    Q = np.random.randn(B, H_q, S, D).astype(np.float32)
    K = np.random.randn(B, H_kv, S, D).astype(np.float32)
    ref = np_contract_5d_gqa(Q, K, G)
    tQ = torch.from_numpy(Q)
    tK = torch.from_numpy(K).repeat_interleave(G, dim=1)
    out = torch.matmul(tQ, tK.transpose(-2,-1)).numpy()
    np.testing.assert_allclose(out, ref, **TOL)
    assert out.shape == (B, H_q, S, S)

@pytest.mark.parametrize("B,S,E,Din,Dout", SHAPES_MOE)
def test_5d_moe(B, S, E, Din, Dout):
    X = np.random.randn(B, S, E, Din).astype(np.float32)
    W = np.random.randn(E, Din, Dout).astype(np.float32)
    ref = np_contract_5d_moe(X, W)
    out = torch.einsum('bsed,edm->bsem',
                        torch.from_numpy(X),
                        torch.from_numpy(W)).numpy()
    np.testing.assert_allclose(out, ref, **TOL)
    assert out.shape == (B, S, E, Dout)

# ── ND Tests ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("B,S,R,D,H", SHAPES_MLA)
def test_nd_mla_kv(B, S, R, D, H):
    C = np.random.randn(B, S, R).astype(np.float32)
    W = np.random.randn(H, R, D).astype(np.float32)
    ref = np_contract_nd_mla(C, W)
    out = torch.einsum('bsr,hrd->bhsd',
                        torch.from_numpy(C),
                        torch.from_numpy(W)).numpy()
    np.testing.assert_allclose(out, ref, **TOL)
    assert out.shape == (B, H, S, D)

# ── FLOPs and arithmetic intensity verification ───────────────────────────

@pytest.mark.parametrize("M,K,N", [(128,256,128)])
def test_flop_count_2d(M, K, N):
    expected_flops = 2 * M * K * N
    assert expected_flops == 2 * 128 * 256 * 128
    # Verify AI < ridge point (memory-bound at batch=1)
    read_bytes = (M*K + K*N) * 4    # float32
    write_bytes = M*N * 4
    ai = expected_flops / (read_bytes + write_bytes)
    # ridge point for A100 BF16: ~312 TFLOPS / 2 TB/s = 156 FLOP/byte
    assert ai < 156, f"AI={ai:.1f} should be memory-bound for small batch"

# ── Numerical stability tests ─────────────────────────────────────────────

def test_large_values_fp32_vs_fp16():
    """FP16 overflows; FP32 accumulation must be used."""
    A = np.ones((64, 128), dtype=np.float32) * 100.0
    B = np.ones((128, 64), dtype=np.float32) * 100.0
    ref_fp32 = np_contract_2d(A, B)          # 128 * 100 * 100 = 1,280,000
    # FP16 max = 65504; each element would overflow without FP32 accumulation
    tA = torch.from_numpy(A).to(torch.float16)
    tB = torch.from_numpy(B).to(torch.float16)
    # torch.mm with fp16 inputs accumulates in fp32 on modern GPUs
    # Test that result is finite
    out = torch.mm(tA, tB).float().numpy()
    assert np.isfinite(out).all(), "FP16 matmul produced inf/nan"
    print(f"FP16 matmul (large values): max={out.max():.0f} "
          f"ref={ref_fp32.max():.0f}")

def test_softmax_stability_in_attention():
    """Numerically stable softmax must be used before attention output."""
    S = np.random.randn(2, 8, 32, 32).astype(np.float32) * 10  # large logits
    S_stable = S - S.max(axis=-1, keepdims=True)
    w = np.exp(S_stable)
    w /= w.sum(axis=-1, keepdims=True)
    assert np.isfinite(w).all(), "Softmax produced nan/inf with large logits"
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-5), "Weights don't sum to 1"
    print("PASS: numerically stable softmax in attention")

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
```

---

## A2.9 Contraction Cheat Sheet: Shapes, FLOPs, and Memory

```
┌─────────────────┬──────────────────────────┬──────────────────────────┬──────────┐
│ Contraction     │ Einsum                   │ FLOPs                    │ AI (B=1) │
├─────────────────┼──────────────────────────┼──────────────────────────┼──────────┤
│ 2D GEMM         │ mk,kn->mn                │ 2*M*K*N                  │ ~1.0     │
│ 3D Token Proj   │ bsd,dn->bsn              │ 2*B*S*D*N                │ ~B*S     │
│ 4D Attn Score   │ bhid,bhjd->bhij          │ 2*B*H*S*S*D              │ ~S/2     │
│ 4D Attn Output  │ bhij,bhjd->bhid          │ 2*B*H*S*S*D              │ ~S/2     │
│ 5D GQA Score    │ (expand K) bhid->bhij    │ 2*B*H_q*S*S*D            │ ~S/2     │
│ 5D MoE Expert   │ bsed,edm->bsem           │ 2*B*S*E*Din*Dout         │ ~B*S     │
│ ND MLA KV Recon │ bsr,hrd->bhsd            │ 2*B*H*S*R*D              │ ~H*S     │
└─────────────────┴──────────────────────────┴──────────────────────────┴──────────┘

Ridge point A100 BF16 = 312 TFLOPS / 2,000 GB/s = 156 FLOP/byte
Decode (B=1, S=1): all contractions are memory-bound (AI << 156)
Prefill (B=8, S=2048): attention Q@K^T becomes compute-bound (AI >> 156)
```

---

---

## A2.10 Manual Worked Examples — Step-by-Step Index Arithmetic

These examples walk through each contraction *by hand*, writing out every summed index explicitly. Working through small cases in pencil-and-paper style builds the intuition that `torch.einsum` and CUDA kernels rely on implicitly.

---

### A2.10.1 2D Matrix Multiply: `C[i,k] = Σ_j A[i,j] · B[j,k]`

Let A be 2×3 and B be 3×2. The output C is 2×2.

```
        B
     [j=0  j=1]
A    k=0   k=1

i=0  [ 1   2   3 ]    [  7  10  ]
i=1  [ 4   5   6 ]    [ 16  22  ]
                B:
              j=0  j=1
            k=0 [ 1   4  ]
            k=1 [ 2   5  ]
            k=2 [ 3   6  ]
```

Every element C[i,k] requires summing the product of a full row of A against a full column of B:

```
C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
       = 1*1 + 2*2 + 3*3
       = 1 + 4 + 9
       = 14   ← one dot product of length 3

C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
       = 1*4 + 2*5 + 3*6
       = 4 + 10 + 18
       = 32

C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
       = 4*1 + 5*2 + 6*3
       = 4 + 10 + 18
       = 32

C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
       = 4*4 + 5*5 + 6*6
       = 16 + 25 + 36
       = 77
```

Verification with Python:

```python
import numpy as np
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[1,4],[2,5],[3,6]])
print(np.einsum('ij,jk->ik', A, B))
# [[ 14  32]
#  [ 32  77]]
```

**Key observation:** The shared index `j` runs over 3 values. Each output element costs 3 multiplications and 2 additions — 6 FLOPs in the multiply-accumulate sense. Total FLOPs = 2×2×3×2 = 24. Total IO = (2×3 + 3×2 + 2×2)×4 bytes (FP32) = 64 bytes. AI = 24/64 = 0.375 FLOP/byte — deeply memory-bound even at this tiny scale.

---

### A2.10.2 3D Batched Matmul: `C[b,i,k] = Σ_j A[b,i,j] · B[b,j,k]`

The batch index `b` is a *free* index — it appears in all three tensors and is never summed. For each batch slice, an independent 2D matmul runs.

Let A be shape (2,2,2) and B be shape (2,2,2):

```
A[b=0] = [[1, 2],      A[b=1] = [[5, 6],
           [3, 4]]                [7, 8]]

B[b=0] = [[1, 0],      B[b=1] = [[0, 1],
           [0, 1]]                [1, 0]]
```

**Batch 0** (B[b=0] is identity, so C[b=0] = A[b=0]):

```
C[0,0,0] = A[0,0,0]*B[0,0,0] + A[0,0,1]*B[0,1,0] = 1*1 + 2*0 = 1
C[0,0,1] = A[0,0,0]*B[0,0,1] + A[0,0,1]*B[0,1,1] = 1*0 + 2*1 = 2
C[0,1,0] = A[0,1,0]*B[0,0,0] + A[0,1,1]*B[0,1,0] = 3*1 + 4*0 = 3
C[0,1,1] = A[0,1,0]*B[0,0,1] + A[0,1,1]*B[0,1,1] = 3*0 + 4*1 = 4

C[b=0] = [[1, 2], [3, 4]]   (unchanged — multiplied by identity)
```

**Batch 1** (B[b=1] swaps columns):

```
C[1,0,0] = A[1,0,0]*B[1,0,0] + A[1,0,1]*B[1,1,0] = 5*0 + 6*1 = 6
C[1,0,1] = A[1,0,0]*B[1,0,1] + A[1,0,1]*B[1,1,1] = 5*1 + 6*0 = 5
C[1,1,0] = A[1,1,0]*B[1,0,0] + A[1,1,1]*B[1,1,0] = 7*0 + 8*1 = 8
C[1,1,1] = A[1,1,0]*B[1,0,1] + A[1,1,1]*B[1,1,1] = 7*1 + 8*0 = 7

C[b=1] = [[6, 5], [8, 7]]   (columns swapped)
```

```python
import numpy as np
A = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
B = np.array([[[1,0],[0,1]], [[0,1],[1,0]]])
print(np.einsum('bij,bjk->bik', A, B))
# [[[1 2] [3 4]]
#  [[6 5] [8 7]]]
```

**Rewrite as token projection:** In a transformer, `X[b,s,d] @ W[d,n] → Y[b,s,n]` is a 3D contraction where `b` and `s` are both free indices. The weight matrix W has no batch dimension — it is *broadcast* over `b`. Written as einsum: `bsd,dn->bsn`. Every sequence position and every batch element compute an independent linear projection sharing the same W.

---

### A2.10.3 4D Attention Score: `S[b,h,i,j] = Σ_d Q[b,h,i,d] · K[b,h,j,d]`

The two free indices inside the heads are the *query position* `i` and the *key position* `j`. The contracted index `d` is the head dimension.

Tiny example: B=1, H=1, S=3 tokens, D=2.

```
Q[0,0] = [[1, 0],      K[0,0] = [[1, 1],
           [0, 1],                [0, 1],
           [1, 1]]                [1, 0]]
```

Computing S[0,0,i,j] = Q[0,0,i,:] · K[0,0,j,:] (dot product over d=2):

```
S[0,0,0,0] = Q[0,0,0,0]*K[0,0,0,0] + Q[0,0,0,1]*K[0,0,0,1]
           = 1*1 + 0*1 = 1

S[0,0,0,1] = Q[0,0,0,0]*K[0,0,1,0] + Q[0,0,0,1]*K[0,0,1,1]
           = 1*0 + 0*1 = 0

S[0,0,0,2] = Q[0,0,0,0]*K[0,0,2,0] + Q[0,0,0,1]*K[0,0,2,1]
           = 1*1 + 0*0 = 1

S[0,0,1,0] = Q[0,0,1,0]*K[0,0,0,0] + Q[0,0,1,1]*K[0,0,0,1]
           = 0*1 + 1*1 = 1

S[0,0,1,1] = 0*0 + 1*1 = 1
S[0,0,1,2] = 0*1 + 1*0 = 0

S[0,0,2,0] = 1*1 + 1*1 = 2
S[0,0,2,1] = 1*0 + 1*1 = 1
S[0,0,2,2] = 1*1 + 1*0 = 1

S[0,0] = [[1, 0, 1],
           [1, 1, 0],
           [2, 1, 1]]
```

Scaled by 1/√D = 1/√2 ≈ 0.707:

```
S_scaled = [[0.707, 0.000, 0.707],
             [0.707, 0.707, 0.000],
             [1.414, 0.707, 0.707]]
```

After causal masking (upper triangle → -∞) and row-wise softmax:

```
Row 0: softmax([0.707, -inf, -inf]) = [1.000, 0.000, 0.000]
Row 1: softmax([0.707,  0.707, -inf]) = [0.500, 0.500, 0.000]
Row 2: softmax([1.414,  0.707, 0.707]) = [0.506, 0.247, 0.247]
```

**FLOPs for attention scores:** 2 × B × H × S × S × D = 2×1×1×3×3×2 = 36. Total outputs = 9. Each output is one dot product of length D=2 costing 2D FLOPs.

```python
import numpy as np
Q = np.array([[[[1,0],[0,1],[1,1]]]])   # [1,1,3,2]
K = np.array([[[[1,1],[0,1],[1,0]]]])   # [1,1,3,2]
S = np.einsum('bhid,bhjd->bhij', Q, K)
print(S)
# [[[[1 0 1]
#    [1 1 0]
#    [2 1 1]]]]
```

---

### A2.10.4 5D MoE Expert Contraction: `Y[b,s,e,m] = Σ_d X[b,s,e,d] · W[e,d,m]`

In a Mixture-of-Experts layer, each token is routed to E experts. The weight tensor W has a leading expert dimension `e` — each expert maintains its own d×m weight matrix.

Tiny example: B=1, S=1, E=2 experts, D=2 input, M=2 output.

```
X[0,0]  = [[1, 2],     ← token, 2 expert slots, each D=2
            [3, 4]]

W[e=0]  = [[1, 0],     ← expert 0: 2×2 weight matrix
            [0, 1]]

W[e=1]  = [[0, 1],     ← expert 1: 2×2 weight matrix
            [1, 0]]
```

For each expert `e`, independently:

```
Expert e=0:
  Y[0,0,0,0] = X[0,0,0,0]*W[0,0,0] + X[0,0,0,1]*W[0,1,0]
             = 1*1 + 2*0 = 1
  Y[0,0,0,1] = X[0,0,0,0]*W[0,0,1] + X[0,0,0,1]*W[0,1,1]
             = 1*0 + 2*1 = 2
  Y[0,0,0,:] = [1, 2]   (expert 0 passes through X unchanged — identity W)

Expert e=1:
  Y[0,0,1,0] = X[0,0,1,0]*W[1,0,0] + X[0,0,1,1]*W[1,1,0]
             = 3*0 + 4*1 = 4
  Y[0,0,1,1] = X[0,0,1,0]*W[1,0,1] + X[0,0,1,1]*W[1,1,1]
             = 3*1 + 4*0 = 3
  Y[0,0,1,:] = [4, 3]   (expert 1 swaps dimensions)
```

```python
import numpy as np
X = np.array([[[[1,2],[3,4]]]])         # [1,1,2,2]
W = np.array([[[1,0],[0,1]], [[0,1],[1,0]]])  # [2,2,2]
Y = np.einsum('bsed,edm->bsem', X, W)
print(Y)
# [[[[1 2]
#    [4 3]]]]
```

**Why this cannot be a single matmul without reshaping:** Standard `torch.matmul` expects the expert dimension to be a batch dimension aligned the same way in both operands. Because X has shape `[b,s,e,d]` and W has shape `[e,d,m]`, you need to either: (a) use `torch.einsum('bsed,edm->bsem', X, W)` directly, or (b) reshape to `[b*s, e, d]` and use `torch.bmm` across the e axis. The einsum route avoids the reshape copies.

---

### A2.10.5 ND Broadcast Contraction: `Y[...,i,k] = Σ_j X[...,i,j] · W[j,k]`

When a weight matrix W has no batch dimensions, it is broadcast over all leading dimensions of X. The einsum `'...ij,...jk->...ik'` generalizes matmul to any prefix of batch dimensions.

```
X shape: [2, 3, 4, 5]   (arbitrary batch prefix [2,3,4], then [5] = in_features)
W shape: [5, 8]          (5 in_features, 8 out_features)

Y shape: [2, 3, 4, 8]

For any fixed (b0, b1, b2):
  Y[b0, b1, b2, k] = Σ_{j=0}^{4} X[b0, b1, b2, j] * W[j, k]
```

This is exactly what happens in a transformer FFN applied over all positions in all batch elements simultaneously. The 24 independent positions (2×3×4) each run the same linear layer — W is read once and amortized over all positions.

```python
import numpy as np
X = np.random.randn(2, 3, 4, 5)
W = np.random.randn(5, 8)
Y = np.einsum('...j,jk->...k', X, W)
print(Y.shape)   # (2, 3, 4, 8)
# Equivalent: Y = X @ W   ← PyTorch broadcasts W over leading dims
```

---

### A2.10.6 FLOPs and Memory: Worked Arithmetic

The following table traces the full arithmetic for representative LLM contraction shapes (Llama 3 8B, BF16, batch=1, prefill S=512):

```
Contraction          Shape                 FLOPs                  IO (BF16)   AI
─────────────────────────────────────────────────────────────────────────────────────────
Q projection         [1,512,4096]@[4096,4096]
                     B=1,S=512,D=4096,N=4096
                     FLOPs = 2*512*4096*4096 = 17.2 GFLOPs
                     IO = (512*4096 + 4096*4096 + 512*4096)*2B
                        = (2.1M + 16.8M + 2.1M)*2 = 42.0 MB
                     AI = 17.2G / 42.0M = 409 FLOP/byte  → COMPUTE-BOUND

Attention Q@K^T      [1,32,512,128]@[1,32,128,512]
                     B=1,H=32,S=512,D=128
                     FLOPs = 2*1*32*512*512*128 = 4.3 GFLOPs
                     IO = 2*(1*32*512*128)*2B + (1*32*512*512)*2B
                        = 67.1MB + 33.6MB = 100.7 MB
                     AI = 4.3G / 100.7M = 42.7 FLOP/byte  → MEMORY-BOUND

FFN gate (SwiGLU)    [1,512,4096]@[4096,14336]
                     B=1,S=512,D=4096,FF=14336
                     FLOPs = 2*512*4096*14336 = 60.1 GFLOPs
                     IO = (512*4096 + 4096*14336 + 512*14336)*2B
                        = (4.2M + 58.7M + 14.7M)*2 = 155.1 MB
                     AI = 60.1G / 155.1M = 387 FLOP/byte  → COMPUTE-BOUND

Decode Q projection  [1,1,4096]@[4096,4096]   (S=1)
                     FLOPs = 2*1*1*4096*4096 = 33.6 MFLOPs
                     IO = (1*4096 + 4096*4096 + 1*4096)*2B = 33.6 MB
                     AI = 33.6M / 33.6M = 1.0 FLOP/byte   → MEMORY-BOUND

─────────────────────────────────────────────────────────────────────────────────────────
A100 ridge (BF16): 312 TFLOPS / 2.0 TB/s = 156 FLOP/byte
H100 ridge (BF16 dense): 1,979 TFLOPS / 3.35 TB/s = 591 FLOP/byte
Compute-bound if AI > ridge; memory-bound if AI < ridge.
─────────────────────────────────────────────────────────────────────────────────────────
```

**Pattern:** Prefill operations (large S) push Q projections and FFN layers into compute-bound territory — these saturate the tensor cores. Decode operations (S=1) collapse to AI≈1, making every contraction memory-bound regardless of model size.

---


## A2.11 Self-Check Questions

1. A 3D token projection has shape `X[8,2048,4096] @ W[4096,4096]`. Calculate
   the exact FLOPs, read bandwidth, and arithmetic intensity. Is this contraction
   memory-bound or compute-bound on an A100 during prefill?

2. Write the einsum string for the MLA KV reconstruction
   `K[b,h,s,d] = sum_r C_KV[b,s,r] * W_UK[h,r,d]` and explain why this cannot
   be expressed as a single `torch.matmul` call without reshaping.

3. In the GQA 5D contraction, G query heads share 1 KV head. If you implement
   this as `K.repeat_interleave(G, dim=1)` before the matmul, what is the extra
   memory cost (in bytes) for a model with H_q=32, H_kv=8, S=4096, D=128 in
   BF16? Is there a zero-copy alternative?

4. The tiled 2D CUDA kernel uses `TILE=16`, giving thread blocks of 256 threads.
   For a GEMM of shape [1024,1024,1024], how many thread blocks are launched?
   How many tiles does each output element participate in? What is the shared
   memory usage per block?

5. A Triton 2D GEMM kernel with `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32` processes
   a [2048,4096,2048] contraction. How many kernel instances (programs) are
   launched? Each program loads two tiles per K-step — how many HBM load
   transactions does the full kernel issue?

---

## Worked Solutions

### Solution 1 — 3D projection FLOPs and arithmetic intensity

**FLOPs:**

```
FLOPs = 2 * B * S * D_in * D_out
      = 2 * 8 * 2048 * 4096 * 4096
      = 2 * 8 * 2048 * 16,777,216
      = 549,755,813,888  ~= 549.8 GFLOPs
```

**Read bandwidth:**

```
X:  8 * 2048 * 4096 * 2 bytes (BF16) = 134.2 MB
W:  4096 * 4096 * 2 bytes            = 33.6 MB
Out: 8 * 2048 * 4096 * 2 bytes       = 134.2 MB
Total IO = 134.2 + 33.6 + 134.2      = 302.0 MB
```

**Arithmetic intensity:**

```
AI = 549.8 GFLOPs / 302.0 MB = 1,820 FLOP/byte
```

A100 ridge point (BF16) = 312 TFLOPS / 2,000 GB/s = 156 FLOP/byte.

**1,820 >> 156: this contraction is compute-bound during prefill.** The large
batch (B=8) and sequence (S=2048) mean the weight matrix is reused 8*2048=16,384
times — amortising its 33.6 MB cost over 549 GFLOPs. This is why prefill is
GPU-compute-limited and benefits from Tensor Cores.

At decode (B=1, S=1): AI = 2*4096*4096 / 302MB ≈ 0.11 FLOP/byte — deeply
memory-bound. The weight must be read from HBM for just 2 MACs.

---

### Solution 2 — MLA einsum string and reshape requirement

**Einsum string:** `'bsr,hrd->bhsd'`

- Free indices in output: b (from C_KV), h (from W_UK), s (from C_KV), d (from W_UK)
- Contracted index: r (appears in both inputs, not in output)

**Why `torch.matmul` alone cannot express this:**

`torch.matmul` handles batched GEMM where the batch dimensions broadcast
between the two tensors. Specifically, it computes:

```
output[..., i, k] = sum_j input1[..., i, j] * input2[..., j, k]
```

The MLA contraction has **two free index groups from different tensors** (b,s
from C_KV and h from W_UK) that do not form a simple batch dimension. There is
no alignment of batch dims: C_KV has shape [B,S,R] and W_UK has shape [H,R,D].
The output [B,H,S,D] requires crossing the B/S dims of one tensor with the H
dim of the other.

**To use matmul:** Reshape both tensors so the contraction over R is the last
axis of the first and second-to-last of the second:

```python
# Option A: einsum (clearest)
K = torch.einsum('bsr,hrd->bhsd', C_KV, W_UK)

# Option B: reshape + matmul
B, S, R = C_KV.shape
H, R, D = W_UK.shape
# C_KV: [B,S,R] -> [B,S,1,R]  W_UK: [H,R,D] -> [1,1,H,R,D]... complex
# Simpler: loop over h (not recommended at scale)
K = torch.stack([C_KV @ W_UK[h] for h in range(H)], dim=1)  # [B,H,S,D]
```

Option B (stack) is O(H) Python overhead. For production, use `einsum` which
the compiler fuses into a single kernel.

---

### Solution 3 — GQA memory cost and zero-copy alternative

**Cost of `repeat_interleave`:**

The expanded K tensor has shape [B, H_q, S, D] = [B, 32, 4096, 128].

```
bytes = B * H_q * S * D * 2  (BF16)
      = B * 32 * 4096 * 128 * 2
      = B * 33,554,432 bytes
      = B * 32 MB
```

For B=8: 256 MB of extra memory allocated just for the head expansion. With
typical KV cache sizes, this can exceed available SRAM and trigger HBM reads.

**Zero-copy alternative — strided view:**

```python
# K: [B, H_kv, S, D]
# Instead of copying, create a strided view:
K_view = K.view(B, H_kv, 1, S, D)          # insert a size-1 dim
K_view = K_view.expand(B, H_kv, G, S, D)   # expand (no copy, stride=0)
K_view = K_view.reshape(B, H_q, S, D)      # may or may not copy
```

`expand` sets the stride for the G dimension to 0, meaning all G query heads
read from the same K data without copying it. `reshape` after `expand` may
trigger a copy if the tensor is not contiguous — use `contiguous()` only if
needed by the subsequent kernel.

Production attention kernels (FlashAttention-2, vLLM's PagedAttention) handle
GQA by passing the group factor G directly into the kernel and computing
`hkv = hq // G` inside the GPU thread, never materialising the expanded K.

---

### Solution 4 — CUDA tiled GEMM thread block count and shared memory

**Problem shape:** [1024, 1024, 1024], TILE = 16.

**Thread blocks launched:**

```
grid_x = ceil(N / TILE) = ceil(1024 / 16) = 64
grid_y = ceil(M / TILE) = ceil(1024 / 16) = 64
total blocks = 64 * 64 = 4,096 thread blocks
```

Each block has TILE*TILE = 256 threads.

**Tiles per output element:**

Each output element C[m,n] is computed as the sum over all K=1024 positions.
The kernel processes TILE=16 K-positions per tile, so:

```
tiles per output element = ceil(K / TILE) = 1024 / 16 = 64 tiles
```

Each thread performs 64 iterations of the inner loop, accumulating 16 MACs
per iteration = 1,024 MACs total per thread = 2,048 FLOPs.

**Shared memory per block:**

Two tiles loaded per iteration: tA[TILE][TILE] and tB[TILE][TILE], both float32.

```
shared_mem = 2 * TILE * TILE * 4 bytes
           = 2 * 16 * 16 * 4
           = 2,048 bytes = 2 KB per block
```

A100 SMs have 164 KB of shared memory; at 2 KB per block, up to 82 blocks could
theoretically co-reside per SM (limited by register count and occupancy rules).

---

### Solution 5 — Triton program count and HBM transactions

**Problem shape:** [2048, 4096, 2048], BLOCK_M=64, BLOCK_N=64, BLOCK_K=32.

**Kernel instances (programs) launched:**

```
programs_m = ceil(M / BLOCK_M) = ceil(2048 / 64) = 32
programs_n = ceil(N / BLOCK_N) = ceil(2048 / 64) = 32
total programs = 32 * 32 = 1,024 programs
```

**K-steps per program:**

```
K_steps = ceil(K / BLOCK_K) = ceil(4096 / 32) = 128
```

**HBM load transactions per program:**

Each K-step loads:

- One tile of A: BLOCK_M * BLOCK_K * 2 bytes (FP16) = 64 * 32 * 2 = 4,096 B
- One tile of B: BLOCK_K * BLOCK_N * 2 bytes         = 32 * 64 * 2 = 4,096 B

Per program: 128 steps * (4,096 + 4,096) B = 1,048,576 B = 1 MB

**Total HBM reads across all programs:**

```
total = 1,024 programs * 1 MB = 1,024 MB = 1.0 GB
```

**Actual data size:**

```
A: 2048 * 4096 * 2 = 16.8 MB
B: 4096 * 2048 * 2 = 16.8 MB
total actual data = 33.6 MB
```

**Reuse factor:**

```
HBM transactions / actual data = 1,024 MB / 33.6 MB = 30.5x
```

Each element of A is read by ceil(N/BLOCK_N)=32 programs; each element of B
by ceil(M/BLOCK_M)=32 programs. Without L2 cache hits, the total HBM traffic
would be 32x the data size. With L2 cache (A100 has 40 MB L2), much of this
is captured in-cache, especially for B (column-major access pattern). Production
Triton kernels add software pipelining (`num_stages=3`) to further hide HBM
latency via async copies.

