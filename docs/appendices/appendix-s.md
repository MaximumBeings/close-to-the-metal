# Appendix S — Introduction to `std::mdspan` for CPU Inference

> *"A matrix is just a 1-D array wearing a map."*

---

## S.1 Why mdspan Exists

Every matrix operation in LLM inference — weight multiplication, attention score
computation, KV cache reads — ultimately touches a contiguous block of memory.
But a flat pointer tells you nothing about shape, stride, or access pattern.
For decades C++ engineers bridged this gap with bespoke wrappers: raw `float*`
paired with separately tracked `rows` and `cols` variables, hand-rolled
`Matrix<T>` classes, or Eigen maps.

C++23 standardises the answer: `std::mdspan`. An mdspan is a **non-owning,
multidimensional view** of an existing memory region. It carries three pieces
of metadata alongside the pointer:

- **Extents** — the size of each dimension (static or dynamic)
- **Layout policy** — how multi-index `(i, j, k, …)` maps to a flat offset
- **Accessor policy** — how the flat offset is dereferenced into a value

Think of it as `std::span` promoted to N dimensions, with pluggable memory
layout and access semantics. The pointer and the data it describes live
separately; mdspan is purely the view.

```
┌──────────────────────────────────────────────────────────────────────┐
│  mdspan<float, extents<size_t,4096,4096>, layout_right, default_accessor>
│                                                                      │
│  .data_handle() ──► float* ──► [row-major 4096×4096 weight matrix]  │
│  .extents()     ──► (4096, 4096)                                     │
│  .stride(0)     ──► 4096   (one row = 4096 floats)                  │
│  .stride(1)     ──► 1      (adjacent columns are adjacent bytes)     │
└──────────────────────────────────────────────────────────────────────┘
```

This appendix covers mdspan from first principles, works through the layout and
accessor policies, shows how `submdspan` enables zero-copy tensor slicing, and
builds up to real CPU inference patterns: tiled GEMM, KV cache views, and the
multi-head attention inner loop.

---

## S.2 Compiler and Standard Library Support

`std::mdspan` ships in:

| Compiler / stdlib | Version | Flag |
|---|---|---|
| GCC | 13+ | `-std=c++23` |
| Clang | 17+ | `-std=c++23` |
| MSVC | 19.36+ | `/std:c++latest` |
| Apple Clang | 15+ | `-std=c++23` |

If your toolchain is older, the reference implementation from Kokkos ships as a
single header:

```bash
# Standalone reference implementation (C++17 compatible)
git clone https://github.com/kokkos/mdspan
# then #include "mdspan/mdspan.hpp"
```

All examples in this appendix compile with:

```bash
g++ -std=c++23 -O2 -march=native -o demo demo.cpp
```

---

## S.3 The Minimal mdspan — A 2-D Matrix View

### S.3.1 Declaring and constructing

```cpp
#include <mdspan>
#include <vector>
#include <print>       // C++23

int main() {
    // Allocate 4×6 data
    std::vector<float> storage(24);
    std::iota(storage.begin(), storage.end(), 0.0f);

    // Wrap as a 4-row, 6-column mdspan (row-major by default)
    using MatF = std::mdspan<float,
                             std::extents<std::size_t, 4, 6>>;
    MatF mat(storage.data());

    // Element access via operator()
    std::println("mat(2,3) = {}", mat(2, 3));   // → 15.0
    std::println("rows={} cols={}", mat.extent(0), mat.extent(1));
}
```

`std::extents<IndexType, E0, E1, …>` encodes the shape. When the value is a
compile-time constant (like `4` and `6` above) the compiler can fold the stride
computation entirely into constants. When a dimension is unknown at compile time,
use the sentinel `std::dynamic_extent`:

```cpp
// Runtime-sized matrix — shape known only at runtime
std::mdspan<float,
            std::extents<std::size_t,
                         std::dynamic_extent,
                         std::dynamic_extent>>
    dyn_mat(ptr, rows, cols);
```

Mixing static and dynamic is idiomatic and efficient: a 4096-wide weight slab
where the batch dimension varies at runtime:

```cpp
// Static hidden dim=4096, dynamic batch
std::mdspan<float,
            std::extents<std::size_t, std::dynamic_extent, 4096>>
    weights(ptr, batch_size);
```

### S.3.2 Type aliases — the inference engineer's toolkit

```cpp
// Convenience aliases used throughout this appendix
template<class T, std::size_t R, std::size_t C>
using StaticMat = std::mdspan<T,
    std::extents<std::size_t, R, C>>;

template<class T>
using DynMat = std::mdspan<T,
    std::extents<std::size_t,
                 std::dynamic_extent,
                 std::dynamic_extent>>;

template<class T>
using DynVec = std::mdspan<T,
    std::extents<std::size_t, std::dynamic_extent>>;

template<class T, std::size_t D0, std::size_t D1, std::size_t D2>
using Tensor3 = std::mdspan<T,
    std::extents<std::size_t, D0, D1, D2>>;
```

---

## S.4 Extents in Depth

`std::extents` is a purely compile-time descriptor. Its template parameters
alternate between `IndexType` and one extent per dimension:

```
std::extents<std::size_t, 128, std::dynamic_extent, 64>
              ^index type  ^D0  ^D1 (runtime)          ^D2
```

At runtime, dynamic extents are passed to the mdspan constructor positionally
after the pointer:

```cpp
float buf[128 * N * 64];   // N varies
auto t = std::mdspan<float,
             std::extents<std::size_t, 128, std::dynamic_extent, 64>>(
             buf, N);       // only dynamic extents are passed
```

### S.4.1 Querying extents

```cpp
auto e = t.extents();
std::size_t d0 = e.extent(0);   // 128  (static)
std::size_t d1 = e.extent(1);   // N    (dynamic, runtime value)
std::size_t d2 = e.extent(2);   // 64   (static)
std::size_t total = t.mapping().required_span_size();
```

### S.4.2 `dextents` — shorthand for all-dynamic

When all dimensions are dynamic, `std::dextents<T, N>` is a concise alias:

```cpp
// 3-D all-dynamic tensor
std::mdspan<float, std::dextents<std::size_t, 3>> tensor(ptr, D, H, W);
```

This is the equivalent of NumPy's unconstrained array. In inference code it
appears wherever you cannot statically know batch size, sequence length, or
head count.

---

## S.5 Layout Policies

The layout policy translates a multi-dimensional index into a flat offset. This
is where the real power lies for inference engineering.

### S.5.1 `layout_right` — row-major (C order)

The default. The last dimension varies fastest:

```
offset(i₀, i₁, …, iₙ) = i₀·s₀ + i₁·s₁ + … + iₙ·sₙ
where s_k = product of extents from k+1 to N
```

For a `[M, N]` matrix: `offset(i, j) = i·N + j`.

PyTorch's CPU tensors, NumPy arrays, and vLLM's weight files are all row-major
by default. This is the layout to reach for first.

```cpp
std::mdspan<float, std::extents<std::size_t, 1024, 4096>,
            std::layout_right> W(ptr);
// W(i, j) ≡ ptr[i * 4096 + j]
```

### S.5.2 `layout_left` — column-major (Fortran order)

The first dimension varies fastest:

```
offset(i, j) = i + j·M    for an [M, N] matrix
```

BLAS libraries (LAPACK, cuBLAS host API) expect column-major. When calling
`cblas_sgemm`, you can wrap your storage in `layout_left` mdspan and pass the
strides directly — no transposition copies:

```cpp
std::mdspan<float, std::dextents<std::size_t, 2>,
            std::layout_left> A_colmaj(A_ptr, M, K);
// stride(0) = 1, stride(1) = M — matches CBLAS_COL_MAJOR
```

### S.5.3 `layout_stride` — arbitrary strides

The most general policy. Each dimension has an independent stride stored at
runtime:

```cpp
// A row-major 512×512 matrix inside a larger 1024×1024 buffer
// (top-left quadrant view, no copy)
std::array<std::ptrdiff_t, 2> strides{1024, 1};
std::layout_stride::mapping<std::dextents<std::size_t, 2>>
    map{{512, 512}, strides};
std::mdspan A_sub(big_ptr, map);
```

`layout_stride` is the workhorse for:

- Non-contiguous subviews (every other row, every 4th column)
- Transposed views without copying (`strides = {1, M}` transposes an M×N matrix)
- Interleaved formats (AoS → SoA reinterpretation)

### S.5.4 Worked Example S.1 — Transpose view at zero cost

A standard matrix multiply kernel requires A and Bᵀ (or transposed B). With
`layout_stride` you can present B transposed without allocating or copying:

```cpp
#include <mdspan>
#include <array>

// Original B: [K, N] row-major
float B_data[K * N];
DynMat<float> B(B_data, K, N);
// stride(0) = N, stride(1) = 1

// Transposed view Bᵀ: [N, K] — swap extents AND strides
std::array<std::ptrdiff_t, 2> T_strides{1, static_cast<std::ptrdiff_t>(N)};
std::layout_stride::mapping<std::dextents<std::size_t, 2>>
    T_map{{N, K}, T_strides};
std::mdspan<float, std::dextents<std::size_t, 2>,
            std::layout_stride> Bt(B_data, T_map);

// Bt(n, k) == B(k, n) — same memory, different map
assert(Bt(3, 7) == B(7, 3));
// No allocation. No copy. Zero overhead.
```

In autoregressive decode, every GEMV (matrix-vector multiply) for a single
token reads weight rows sequentially — layout_right is cache-optimal. In
prefill, batched GEMM benefits from different access patterns. The transpose
trick lets the same weight buffer serve both without duplication.

### S.5.5 Custom layout policies

The layout concept is open for extension. Any type satisfying `LayoutMapping`
works. This enables:

- **Tiled layouts** — tiles in L1/L2-friendly chunks (§S.8)
- **Hilbert-curve layouts** — cache-oblivious matrix traversal
- **Quantised layouts** — INT4 packed storage, FP8 blocks

```cpp
// Sketch: a 16×16 tiled layout for 4-bit packed weights
struct layout_tile_16x16 {
    template<class Extents>
    struct mapping {
        using index_type = typename Extents::index_type;
        Extents extents_;

        auto operator()(index_type row, index_type col) const noexcept {
            auto tile_r = row / 16,  elem_r = row % 16;
            auto tile_c = col / 16,  elem_c = col % 16;
            auto tiles_per_row = extents_.extent(1) / 16;
            return (tile_r * tiles_per_row + tile_c) * 256
                   + elem_r * 16 + elem_c;
        }
        // … required_span_size(), strides(), is_always_unique(), etc.
    };
};
```

---

## S.6 Accessor Policies

The accessor policy controls how the flat offset is converted to a reference.
The default `std::default_accessor<T>` is trivial: `ptr[offset]`. But
custom accessors enable powerful patterns.

### S.6.1 Aligned accessor

SIMD loads require 32- or 64-byte alignment. An aligned accessor adds
`[[assume_aligned]]` hints the compiler can exploit:

```cpp
template<class T, std::size_t Alignment>
struct aligned_accessor {
    using element_type     = T;
    using reference        = T&;
    using data_handle_type = T*;
    using offset_policy    = aligned_accessor;   // same type for offsets

    reference access(data_handle_type p, std::ptrdiff_t i) const noexcept {
        return *std::assume_aligned<Alignment>(p + i);
    }
    data_handle_type offset(data_handle_type p, std::ptrdiff_t i) const noexcept {
        return p + i;
    }
};

// A 64-byte-aligned weight matrix
std::mdspan<float,
            std::extents<std::size_t, 4096, 4096>,
            std::layout_right,
            aligned_accessor<float, 64>> W(aligned_ptr);
```

With this accessor, LLVM/GCC will emit AVX-512 aligned loads (`vmovaps`) instead
of unaligned loads (`vmovups`) — a measurable win on inner-loop kernels.

### S.6.2 Atomic accessor

For parallel writes to a shared accumulation buffer without data races:

```cpp
#include <atomic>

template<class T>
struct atomic_ref_accessor {
    using element_type     = T;
    using reference        = std::atomic_ref<T>;
    using data_handle_type = T*;
    using offset_policy    = atomic_ref_accessor;

    reference access(data_handle_type p, std::ptrdiff_t i) const noexcept {
        return std::atomic_ref<T>(p[i]);
    }
    data_handle_type offset(data_handle_type p, std::ptrdiff_t i) const noexcept {
        return p + i;
    }
};
```

Useful when accumulating attention scores across threads — each thread owns a
slice of the Q dimension but writes to the same output buffer.

### S.6.3 Scaled accessor — FP8 dequantisation

LLM weights are often stored as FP8 but processed as FP32. An accessor can
perform dequantisation on every read:

```cpp
struct fp8_to_float_accessor {
    using element_type     = float;          // logical type
    using reference        = float;          // value, not reference (read-only)
    using data_handle_type = uint8_t*;       // physical storage

    // scale stored alongside the accessor
    float scale;

    float access(data_handle_type p, std::ptrdiff_t i) const noexcept {
        // decode E4M3 FP8 → float, then scale
        return fp8_e4m3_to_float(p[i]) * scale;
    }
    data_handle_type offset(data_handle_type p, std::ptrdiff_t i) const noexcept {
        return p + i;
    }
};

// FP8 weight matrix presenting as float view
std::mdspan<float,
            std::dextents<std::size_t, 2>,
            std::layout_right,
            fp8_to_float_accessor>
    W_fp8(fp8_ptr, fp8_to_float_accessor{scale}, rows, cols);

// Caller sees floats; physical reads are uint8_t with on-the-fly decode
float val = W_fp8(i, j);   // fp8_e4m3_to_float(fp8_ptr[i*cols+j]) * scale
```

This is how llama.cpp's GGUF GGML_TYPE_F8_E4M3 weights are conceptually
accessed: a strided view of packed bytes with per-tensor or per-row scales
applied at read time.

---

## S.7 `submdspan` — Zero-Copy Tensor Slicing

`std::submdspan` (C++26, but available in the Kokkos reference implementation
for C++17/23) extracts a lower-dimensional view from an mdspan without touching
the underlying data.

### S.7.1 Slice specifiers

| Specifier | Meaning |
|---|---|
| `std::full_extent` | Take the entire dimension |
| `i` (integer) | Fix dimension to index `i`, reducing rank by 1 |
| `std::pair{lo, hi}` | Slice `[lo, hi)` |
| `std::strided_slice{off, ext, stride}` | Strided range |

### S.7.2 Extracting a single token's embedding

```cpp
// token_emb: [seq_len, d_model] — all token embeddings for a sequence
DynMat<float> token_emb(ptr, seq_len, d_model);

// Extract embedding for token t → shape [d_model]
auto tok_vec = std::submdspan(token_emb,
                              t,                   // fix row dimension
                              std::full_extent);   // keep column dimension
// tok_vec is mdspan<float, extents<size_t, dynamic_extent>>
// Points into the same memory — zero copy
float e0 = tok_vec(0);   // token_emb(t, 0)
```

### S.7.3 Extracting a KV cache head slice

The KV cache in a multi-head attention layer is typically:
`[num_layers, 2, num_heads, seq_len, head_dim]`
(the `2` indexes K vs V).

```cpp
// Full KV cache tensor
std::mdspan<float,
    std::extents<std::size_t,
        std::dynamic_extent,   // num_layers
        2,                     // K=0, V=1
        std::dynamic_extent,   // num_heads
        std::dynamic_extent,   // seq_len
        64>>                   // head_dim = 64 (static)
    kv_cache(ptr, num_layers, num_heads, max_seq_len);

// Extract K matrix for layer 12, head 7: shape [seq_len, 64]
auto K_12_7 = std::submdspan(kv_cache,
    12,                  // layer 12
    0,                   // K
    7,                   // head 7
    std::full_extent,    // all sequence positions
    std::full_extent);   // all head dims
// K_12_7(pos, d) == kv_cache(12, 0, 7, pos, d)
// No copy. The view is valid for the lifetime of kv_cache's backing storage.
```

### S.7.4 Batch slice and strided access

```cpp
// Weight matrix W: [out_features, in_features]
DynMat<float> W(w_ptr, out_features, in_features);

// Every other output neuron (for debug / ablation):
auto W_even = std::submdspan(W,
    std::strided_slice{0, out_features/2, 2},   // rows 0,2,4,...
    std::full_extent);
// W_even.extent(0) == out_features/2
// W_even(i, j)    == W(2*i, j)
```

---

## S.8 Tiled GEMM with mdspan

Matrix multiplication is memory-bound on CPU for the weight sizes typical in
LLM inference (4096×4096 and larger). The standard fix is register tiling:
load small blocks of A and B into L1-resident registers, accumulate into a C
tile, and write back once. mdspan makes the tiling logic clean and the index
arithmetic explicit.

### S.8.1 Tile layout helper

```cpp
// Extract a [TILE_M, TILE_K] block of A starting at (row_base, col_base)
template<std::size_t TM, std::size_t TK, class MDS>
auto tile_of(MDS A, std::size_t row_base, std::size_t col_base) {
    return std::submdspan(A,
        std::pair{row_base, row_base + TM},
        std::pair{col_base, col_base + TK});
}
```

### S.8.2 Tiled GEMM kernel

```cpp
#include <mdspan>
#include <immintrin.h>   // AVX-256

template<std::size_t TM = 4, std::size_t TN = 4, std::size_t TK = 32>
void gemm_tiled(DynMat<const float> A,    // [M, K]
                DynMat<const float> B,    // [K, N]
                DynMat<float>       C) {  // [M, N]
    const auto M = A.extent(0);
    const auto N = B.extent(1);
    const auto K = A.extent(1);

    for (std::size_t m = 0; m < M; m += TM)
    for (std::size_t n = 0; n < N; n += TN)
    for (std::size_t k = 0; k < K; k += TK) {
        // Register accumulator tile
        float acc[TM][TN] = {};

        // Load A tile [TM, TK] and B tile [TK, TN] into registers
        auto At = tile_of<TM, TK>(A, m, k);
        auto Bt = tile_of<TK, TN>(B, k, n);

        for (std::size_t ti = 0; ti < TM; ++ti)
        for (std::size_t tk = 0; tk < TK; ++tk)
        for (std::size_t tj = 0; tj < TN; ++tj)
            acc[ti][tj] += At(ti, tk) * Bt(tk, tj);

        // Write accumulated tile back to C
        for (std::size_t ti = 0; ti < TM; ++ti)
        for (std::size_t tj = 0; tj < TN; ++tj)
            C(m + ti, n + tj) += acc[ti][tj];
    }
}
```

The mdspan views `At` and `Bt` compile to pointer + constant-stride arithmetic —
no heap allocation, no virtual dispatch. With `-O3 -march=native` the inner
accumulation loop auto-vectorises to SIMD FMA instructions.

### S.8.3 Worked Example S.2 — GEMV for single-token decode

During autoregressive decode, batch size = 1: the computation degenerates from
GEMM to GEMV (matrix-vector multiply). The memory access pattern changes
fundamentally:

```
B=1:   output[out]  = Σ_k  W[out, k] * input[k]
                      ^reads entire row of W for each output neuron
```

```cpp
// W: [out_features, hidden_dim], x: [hidden_dim], y: [out_features]
void gemv(DynMat<const float> W,
          DynVec<const float> x,
          DynVec<float>       y) {
    const auto rows = W.extent(0);
    const auto cols = W.extent(1);

    for (std::size_t i = 0; i < rows; ++i) {
        float acc = 0.0f;
        // submdspan extracts row i as a 1-D span — zero copy
        auto row = std::submdspan(W, i, std::full_extent);
        for (std::size_t j = 0; j < cols; ++j)
            acc += row(j) * x(j);
        y(i) = acc;
    }
}
```

With AVX-512 the inner loop becomes 16-wide FMA: 16 floats × 1 FMA = 32
FLOP/cycle → ~100 GB/s effective bandwidth on a modern Xeon at 3 GHz.

---

## S.9 Multi-Head Attention with mdspan

The entire MHA inner loop can be expressed cleanly using mdspan views and
submdspan slices. This section builds the forward pass from scratch, making the
tensor indexing explicit.

### S.9.1 Tensor shapes

```
Q, K, V:  [batch, num_heads, seq_len, head_dim]
scores:   [batch, num_heads, seq_len, seq_len]
output:   [batch, num_heads, seq_len, head_dim]
```

For single-batch inference (batch=1) the batch dimension is dropped.

### S.9.2 Score computation — Q × Kᵀ

```cpp
// score[h, i, j] = dot(Q[h, i, :], K[h, j, :]) / sqrt(head_dim)
void attention_scores(
    std::mdspan<const float,
        std::extents<std::size_t,
            std::dynamic_extent,   // num_heads
            std::dynamic_extent,   // seq_len
            std::dynamic_extent>>  // head_dim
    Q, K,
    std::mdspan<float,
        std::extents<std::size_t,
            std::dynamic_extent,   // num_heads
            std::dynamic_extent,   // seq_len Q
            std::dynamic_extent>>  // seq_len K
    scores)
{
    const auto H   = Q.extent(0);
    const auto S   = Q.extent(1);
    const auto D   = Q.extent(2);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (std::size_t h = 0; h < H; ++h) {
        // Q_h: [S, D],  K_h: [S, D]
        auto Q_h = std::submdspan(Q, h, std::full_extent, std::full_extent);
        auto K_h = std::submdspan(K, h, std::full_extent, std::full_extent);

        for (std::size_t i = 0; i < S; ++i)
        for (std::size_t j = 0; j < S; ++j) {
            float dot = 0.0f;
            auto qi = std::submdspan(Q_h, i, std::full_extent);   // row i
            auto kj = std::submdspan(K_h, j, std::full_extent);   // row j
            for (std::size_t d = 0; d < D; ++d)
                dot += qi(d) * kj(d);
            scores(h, i, j) = dot * scale;
        }
    }
}
```

All `submdspan` calls produce views into the same buffer — the compiler sees
contiguous accesses and vectorises the inner `d` loop.

### S.9.3 Online softmax with mdspan

```cpp
void softmax_rows(
    std::mdspan<float,
        std::extents<std::size_t,
            std::dynamic_extent,    // num_heads
            std::dynamic_extent,    // seq_len
            std::dynamic_extent>>   // seq_len
    scores)
{
    const auto H = scores.extent(0);
    const auto S = scores.extent(1);

    for (std::size_t h = 0; h < H; ++h)
    for (std::size_t i = 0; i < S; ++i) {
        // Extract row [i, :] for head h — causal mask: only j <= i
        auto row = std::submdspan(scores, h, i, std::pair{0UZ, i + 1});

        float max_val = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j <= i; ++j)
            max_val = std::max(max_val, row(j));

        float sum = 0.0f;
        for (std::size_t j = 0; j <= i; ++j) {
            row(j) = std::exp(row(j) - max_val);
            sum += row(j);
        }
        for (std::size_t j = 0; j <= i; ++j)
            row(j) /= sum;

        // Positions j > i are masked — set to 0
        for (std::size_t j = i + 1; j < S; ++j)
            scores(h, i, j) = 0.0f;
    }
}
```

The causal `std::pair{0, i+1}` slice is zero-cost: it reduces the upper bound
of the inner loop without masking or branching.

### S.9.4 Weighted value aggregation

```cpp
// output[h, i, :] = Σ_j scores[h, i, j] * V[h, j, :]
void weighted_sum(
    std::mdspan<const float,
        std::extents<std::size_t,
            std::dynamic_extent,
            std::dynamic_extent,
            std::dynamic_extent>> scores,
    std::mdspan<const float,
        std::extents<std::size_t,
            std::dynamic_extent,
            std::dynamic_extent,
            std::dynamic_extent>> V,
    std::mdspan<float,
        std::extents<std::size_t,
            std::dynamic_extent,
            std::dynamic_extent,
            std::dynamic_extent>> out)
{
    const auto H = scores.extent(0);
    const auto S = scores.extent(1);
    const auto D = V.extent(2);

    for (std::size_t h = 0; h < H; ++h)
    for (std::size_t i = 0; i < S; ++i) {
        auto out_vec = std::submdspan(out, h, i, std::full_extent);
        for (std::size_t d = 0; d < D; ++d) out_vec(d) = 0.0f;

        for (std::size_t j = 0; j <= i; ++j) {   // causal
            float w = scores(h, i, j);
            auto v_row = std::submdspan(V, h, j, std::full_extent);
            for (std::size_t d = 0; d < D; ++d)
                out_vec(d) += w * v_row(d);
        }
    }
}
```

---

## S.10 KV Cache Management with mdspan

The KV cache is the largest runtime allocation in LLM inference — 2 × num_layers
× num_heads × seq_len × head_dim × sizeof(float) bytes per request. mdspan makes
it easy to manage as a preallocated slab with per-request views.

### S.10.1 Slab allocation

```cpp
struct KVCacheSlab {
    // Preallocate max_seq_len slots for one request
    std::vector<float> storage;

    // Shape: [num_layers, 2, num_heads, max_seq_len, head_dim]
    using KVTensor = std::mdspan<float,
        std::extents<std::size_t,
            std::dynamic_extent,   // num_layers
            2,                     // K=0, V=1
            std::dynamic_extent,   // num_heads
            std::dynamic_extent,   // max_seq_len
            std::dynamic_extent>>; // head_dim

    KVTensor view;

    KVCacheSlab(std::size_t num_layers,
                std::size_t num_heads,
                std::size_t max_seq_len,
                std::size_t head_dim)
        : storage(num_layers * 2 * num_heads * max_seq_len * head_dim, 0.0f)
        , view(storage.data(), num_layers, num_heads, max_seq_len, head_dim)
    {}

    // Get K or V for one layer and head, up to current seq position
    auto get(std::size_t layer, int kv, std::size_t head,
             std::size_t seq_pos) {
        return std::submdspan(view,
            layer, kv, head,
            std::pair{0UZ, seq_pos},
            std::full_extent);
        // Returns [seq_pos, head_dim] — live window, zero copy
    }
};
```

### S.10.2 PagedAttention analogy

vLLM's PagedAttention (Chapter 6) manages the KV cache as a pool of fixed-size
blocks. In a CPU-only setting you can replicate the concept with mdspan: divide
the slab into `PAGE_SIZE`-token pages and maintain a free list.

```cpp
constexpr std::size_t PAGE_SIZE = 16;   // tokens per page

// One page: [2, num_heads, PAGE_SIZE, head_dim]
template<std::size_t NH, std::size_t HD>
using KVPage = std::mdspan<float,
    std::extents<std::size_t, 2, NH, PAGE_SIZE, HD>>;

struct PagePool {
    std::vector<float> arena;       // one contiguous arena
    std::vector<std::size_t> free_pages;
    std::size_t page_floats;

    PagePool(std::size_t num_pages, std::size_t num_heads,
             std::size_t head_dim)
        : page_floats(2 * num_heads * PAGE_SIZE * head_dim)
        , arena(num_pages * page_floats)
    {
        for (std::size_t i = num_pages; i-- > 0;)
            free_pages.push_back(i);
    }

    float* alloc_page() {
        auto idx = free_pages.back();
        free_pages.pop_back();
        return arena.data() + idx * page_floats;
    }
    void free_page(float* p) {
        free_pages.push_back((p - arena.data()) / page_floats);
    }
};
```

Each page is an mdspan created on demand — its lifetime matches the request,
not the pool.

---

## S.11 `std::linalg` — mdspan Meets BLAS (C++26)

C++26 adds `<linalg>`, a standardised interface to BLAS-level operations
parameterised over mdspan. This closes the loop between the view abstraction
and high-performance compute.

### S.11.1 Key operations

```cpp
#include <linalg>

// Dot product: result = x · y
float d = std::linalg::dot(x_span, y_span);

// Scale: x *= alpha
std::linalg::scale(2.0f, x_span);

// Matrix-vector: y = A * x
std::linalg::matrix_vector_product(A_span, x_span, y_span);

// GEMM: C = A * B
std::linalg::matrix_product(A_span, B_span, C_span);

// Symmetric GEMM: C = A * Aᵀ + C
std::linalg::symmetric_matrix_rank_k_update(1.0f, A_span, C_span,
    std::linalg::upper_triangle);
```

All operations dispatch to the platform BLAS if one is linked (MKL, OpenBLAS,
Accelerate), falling back to a reference implementation. The dispatch is
transparent — you write mdspan code, the library decides whether to call
`cblas_sgemm` or a built-in kernel.

### S.11.2 Inference GEMM with linalg

```cpp
#include <mdspan>
#include <linalg>

void linear_layer(
    DynMat<const float> W,    // [out, in]
    DynMat<const float> X,    // [batch, in]
    DynMat<float>       Y)    // [batch, out]
{
    // Y = X * Wᵀ  →  each row of X dotted with each row of W
    // linalg::matrix_product expects [M,K] × [K,N] → [M,N]
    // so pass X [batch,in] and Wᵀ [in,out]
    auto Wt = /* layout_stride transpose view (§S.5.4) */;
    std::linalg::matrix_product(X, Wt, Y);
}
```

With MKL linked, `matrix_product` calls `cblas_sgemm` with the right leading-
dimension values derived automatically from the mdspan strides.

---

## S.12 mdspan in SIMD Code — Combining with `std::experimental::simd`

C++23 also standardises `std::experimental::simd` (merged into C++26 as
`std::simd`). Combining mdspan addressing with SIMD loads gives fully
standards-based vectorised kernels without intrinsics.

```cpp
#include <mdspan>
#include <experimental/simd>
namespace stdx = std::experimental;

// Vectorised dot product: a · b where both are 1-D mdspan
float dot_simd(DynVec<const float> a, DynVec<const float> b) {
    using V = stdx::native_simd<float>;
    constexpr auto W = V::size();                // e.g. 8 on AVX2

    const auto N = a.extent(0);
    V acc{0.0f};

    std::size_t i = 0;
    for (; i + W <= N; i += W) {
        V va, vb;
        va.copy_from(a.data_handle() + i, stdx::element_aligned);
        vb.copy_from(b.data_handle() + i, stdx::element_aligned);
        acc += va * vb;
    }
    float result = stdx::reduce(acc);
    for (; i < N; ++i) result += a(i) * b(i);   // scalar tail
    return result;
}
```

This emits AVX2 `vfmadd231ps` instructions on x86 — the same code compiles to
NEON `fmla` on Apple Silicon with no changes.

---

## S.13 Worked Example S.3 — Feed-Forward Layer (FFN)

The Transformer FFN is two linear projections with a nonlinearity:

```
h  = SiLU(X · W₁ᵀ)    [batch, seq, 4*d_model]
out = h  · W₂ᵀ         [batch, seq, d_model]
```

```cpp
#include <mdspan>
#include <cmath>
#include <algorithm>

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// W1: [4*d_model, d_model], W2: [d_model, 4*d_model]
// X:  [batch, seq, d_model], out: [batch, seq, d_model]
void ffn(
    std::mdspan<const float, std::dextents<std::size_t, 3>> X,
    DynMat<const float> W1,
    DynMat<const float> W2,
    std::mdspan<float,  std::dextents<std::size_t, 3>> out,
    std::vector<float>& scratch)   // [batch*seq * 4*d_model]
{
    const auto B  = X.extent(0);
    const auto S  = X.extent(1);
    const auto D  = X.extent(2);
    const auto D4 = W1.extent(0);   // 4 * D

    // Reshape scratch as [B*S, D4]
    DynMat<float> H(scratch.data(), B * S, D4);
    // Reshape X as [B*S, D] for GEMM
    DynMat<const float> X2d(X.data_handle(), B * S, D);
    // Reshape out as [B*S, D]
    DynMat<float> out2d(out.data_handle(), B * S, D);

    // H = X2d × W1ᵀ
    // (use gemm_tiled from §S.8 or std::linalg::matrix_product)
    gemm_tiled(X2d, /* W1ᵀ */ W1, H);   // simplified: assume W1 already transposed

    // SiLU activation in-place
    for (std::size_t i = 0; i < B * S; ++i)
        for (std::size_t j = 0; j < D4; ++j)
            H(i, j) = silu(H(i, j));

    // out2d = H × W2ᵀ
    gemm_tiled(H, W2, out2d);
}
```

All tensor reshape operations (`X2d`, `out2d`, `H`) are mdspan re-wraps of the
same underlying buffers — zero copies, zero allocations.

---

## S.14 Performance Notes and Guidelines

### S.14.1 Static vs. dynamic extents

| Property | Static | Dynamic |
|---|---|---|
| Size in memory | Sizeof pointer only | Pointer + rank × sizeof(index) |
| Stride computation | Compile-time constant | Runtime multiply |
| Aliasing hints | Compiler can assume | May inhibit auto-vec |
| Flexibility | Fixed at compile time | Changed per call |

For weight matrices with fixed hidden dimensions (common in production models),
prefer static extents on the hidden dimension and dynamic on batch/sequence:

```cpp
// Prefer this for d_model=4096 models:
std::mdspan<float,
    std::extents<std::size_t, std::dynamic_extent, 4096>>
// Over:
std::mdspan<float, std::dextents<std::size_t, 2>>
```

The compiler eliminates one runtime multiply from every access.

### S.14.2 `is_contiguous` and cache prefetch hints

```cpp
if constexpr (decltype(mat)::is_always_contiguous()) {
    // emit __builtin_prefetch hints safely
    __builtin_prefetch(mat.data_handle() + prefetch_offset, 0, 3);
}
```

`layout_right` and `layout_left` are always contiguous; `layout_stride` and
custom tiled layouts are not. Checking `is_always_contiguous()` at compile time
allows safe prefetch without branching.

### S.14.3 Avoid `operator()` in innermost hot loops

When the innermost loop is fully unrolled and the compiler cannot see that
strides are constant (dynamic extent), the `operator()` call may generate a
multiply per access. Fix: cache the stride:

```cpp
const std::size_t stride0 = mat.stride(0);   // cache once
const float* base = mat.data_handle();
for (std::size_t i = 0; i < rows; ++i)
    base[i * stride0 + j] += ...;           // pointer arithmetic
```

For static extents the compiler does this automatically; for dynamic extents,
the manual cache eliminates repeated `extent(0)` loads.

### S.14.4 Alignment guarantees

`std::vector<float>` guarantees `alignof(float)` (4 bytes). For AVX-512 you
need 64-byte alignment. Use `std::aligned_alloc`:

```cpp
float* ptr = static_cast<float*>(
    std::aligned_alloc(64, num_elements * sizeof(float)));
// Wrap with mdspan + aligned_accessor (§S.6.1)
```

Or use `std::pmr::vector` with a custom `std::pmr::pool_resource` that enforces
alignment.

---

## S.15 Decision Framework

| You need… | Use… |
|---|---|
| Row-major weight matrix (default) | `layout_right` |
| Column-major for BLAS call | `layout_left` |
| Non-contiguous view / transpose | `layout_stride` |
| Tile-blocked GEMM | Custom `layout_tile` |
| On-the-fly FP8 dequant | Custom accessor |
| Parallel write accumulation | `atomic_ref_accessor` |
| SIMD alignment hints | `aligned_accessor<T,64>` |
| Tensor slice (no copy) | `submdspan` |
| BLAS call without copy | `std::linalg` + mdspan |
| Runtime shapes (batch, seq_len) | `dynamic_extent` |
| Fixed model dim (d_model, head_dim) | Static extent |

---

## S.16 Relation to Existing Inference Frameworks

| Framework | Memory abstraction | mdspan analogy |
|---|---|---|
| llama.cpp | `ggml_tensor` (shape + type + data ptr) | mdspan + custom accessor per GGML type |
| vLLM (C++ kernels) | Flat pointer + size args passed to CUDA kernels | mdspan would replace manual stride args |
| PyTorch ATen (CPU) | `TensorImpl` → `StorageImpl` → `void*` + stride array | Dynamic mdspan with `layout_stride` |
| Eigen | `Map<MatrixXf>` | `DynMat<float>` with `layout_right` |
| BLAS / LAPACK | `lda`, `ldb` leading-dimension arguments | `layout_left` mdspan strides |

Starting from C++23, writing new inference kernels with mdspan rather than raw
pointers gives you the safety and expressiveness of PyTorch tensors with the
overhead of Eigen maps — and full interoperability with `std::linalg` and
`std::simd`.

---

## S.17 Compiling and Running the Examples

All examples in this appendix compile with:

```bash
# C++23 with GCC 13
g++ -std=c++23 -O3 -march=native -Wall -Wextra -o demo demo.cpp

# C++23 with Clang 17
clang++ -std=c++23 -O3 -march=native -stdlib=libc++ -o demo demo.cpp

# Link OpenBLAS for std::linalg dispatch
g++ -std=c++23 -O3 -march=native -o demo demo.cpp -lopenblas

# Using Kokkos reference mdspan on older compilers
g++ -std=c++17 -O3 -march=native -I/path/to/mdspan/include -o demo demo.cpp
```

To verify the worked examples:

```bash
# Worked Example S.1: transpose view correctness
static_assert(Bt(3, 7) == B(7, 3));

# Worked Example S.2: GEMV bandwidth
# Run: ./gemv_bench 4096 4096  → should report ~80–120 GB/s on modern x86
```

---

## S.18 Chapter Summary

`std::mdspan` is the standard C++ answer to a question inference engineers have
been solving ad-hoc for years: how to describe a multidimensional view of
memory without owning it, without copying it, and without paying runtime
overhead for the description.

The key ideas are:

**Extents** encode shape: static dimensions fold to compile-time constants,
dynamic dimensions add one index-word each. Mix static and dynamic to match
your model's architecture — static hidden dim, dynamic batch.

**Layout policies** encode the index-to-offset map: `layout_right` for row-major
(PyTorch default), `layout_left` for BLAS column-major, `layout_stride` for
transposed or non-contiguous views, and custom layouts for tiled or quantised
storage. The transpose-without-copy pattern (§S.5.4) eliminates a common source
of allocation in GEMM code.

**Accessor policies** encode how an offset becomes a value: the default is a raw
pointer dereference, but custom accessors give you aligned loads, atomic writes,
and on-the-fly FP8 dequantisation — all at zero abstraction cost.

**`submdspan`** is the surgical tool: extract a head's KV slice, a single
token's embedding, or a causal attention row without touching the backing data.
Every slice compiles to pointer arithmetic.

**`std::linalg`** (C++26) closes the loop: mdspan views connect directly to
BLAS dispatch, so weight-matrix multiply becomes a one-liner that routes to MKL
or OpenBLAS transparently.

For CPU inference in 2026, mdspan is the lowest-overhead, highest-expressiveness
way to manage the tensor views that sit between your GGUF weight file and your
SIMD kernels.

---

## Self-Check Questions

1. A weight matrix W is stored as `float[4096 * 4096]` in row-major order. You
   need to call a BLAS function that expects column-major. Write the `layout_stride`
   mdspan construction that presents W as column-major without copying any data.

2. Explain why `std::submdspan(K_cache, layer, 0, head, std::pair{0, seq_pos}, std::full_extent)`
   produces a zero-copy view. What happens to the pointer arithmetic at compile time
   when `layer`, `head` are `size_t` and `seq_pos` is `dynamic_extent`?

3. You have a custom FP4 weight format: two 4-bit values packed per byte,
   scale factor per 32-element group. Sketch an `accessor_policy` that unpacks
   FP4 values on read. What type should `reference` be — a `float&` or a `float`?
   Why?

4. Compare the compile-time behaviour of:
   ```cpp
   std::mdspan<float, std::extents<size_t, 4096, 4096>> A(ptr);  // static
   std::mdspan<float, std::dextents<size_t, 2>>         B(ptr, 4096, 4096);  // dynamic
   ```
   For the expression `A(i, j)` vs `B(i, j)`, what assembly differs between the two?
   Under what conditions would you prefer B over A in production inference code?

5. The KV cache for Llama-3 70B has: 80 layers, 8 KV heads, head_dim=128, and
   you are serving a 4096-token context. Calculate the total bytes for one request's
   KV cache at FP16. Then write the `std::extents` declaration for a single mdspan
   that covers the full cache, using static extents for head_dim and dynamic for
   all other dimensions.

---

*— Appendix S covers `std::mdspan` as standardised in C++23 (core) and C++26
(`submdspan`, `std::linalg`). The Kokkos reference implementation
(github.com/kokkos/mdspan) backports all features to C++17.*
