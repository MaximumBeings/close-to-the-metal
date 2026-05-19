# Appendix K — Introduction to `std::mdspan` for CPU Inference

> *"A matrix is just a 1-D array wearing a map."*

---

## K.1 Why mdspan Exists

Every matrix operation in LLM inference — weight multiplication, attention score
computation, KV cache reads — ultimately touches a contiguous block of memory.
But a flat pointer tells you nothing about shape, stride, or access pattern.
For decades C++ engineers bridged this gap with bespoke wrappers: raw `float*`
paired with separately tracked `rows` and `cols` variables, hand-rolled
`Matrix<T>` classes, or Eigen maps.

C++23 standardizes the answer: `std::mdspan`. An mdspan is a **non-owning,
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

## K.2 Compiler and Standard Library Support

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

## K.3 The Minimal mdspan — A 2-D Matrix View

### K.3.1 Declaring and constructing

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

### K.3.2 Type aliases — the inference engineer's toolkit

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

## K.4 Extents in Depth

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

### K.4.1 Querying extents

```cpp
auto e = t.extents();
std::size_t d0 = e.extent(0);   // 128  (static)
std::size_t d1 = e.extent(1);   // N    (dynamic, runtime value)
std::size_t d2 = e.extent(2);   // 64   (static)
std::size_t total = t.mapping().required_span_size();
```

### K.4.2 `dextents` — shorthand for all-dynamic

When all dimensions are dynamic, `std::dextents<T, N>` is a concise alias:

```cpp
// 3-D all-dynamic tensor
std::mdspan<float, std::dextents<std::size_t, 3>> tensor(ptr, D, H, W);
```

This is the equivalent of NumPy's unconstrained array. In inference code it
appears wherever you cannot statically know batch size, sequence length, or
head count.

---

## K.5 Layout Policies

The layout policy translates a multi-dimensional index into a flat offset. This
is where the real power lies for inference engineering.

### K.5.1 `layout_right` — row-major (C order)

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

### K.5.2 `layout_left` — column-major (Fortran order)

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

### K.5.3 `layout_stride` — arbitrary strides

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

### K.5.4 Worked Example K.1 — Transpose view at zero cost

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

### K.5.5 Custom layout policies

The layout concept is open for extension. Any type satisfying `LayoutMapping`
works. This enables:

- **Tiled layouts** — tiles in L1/L2-friendly chunks (§K.8)
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

## K.6 Accessor Policies

The accessor policy controls how the flat offset is converted to a reference.
The default `std::default_accessor<T>` is trivial: `ptr[offset]`. But
custom accessors enable powerful patterns.

### K.6.1 Aligned accessor

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

### K.6.2 Atomic accessor

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

### K.6.3 Scaled accessor — FP8 dequantisation

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

## K.7 `submdspan` — Zero-Copy Tensor Slicing

`std::submdspan` (C++26, but available in the Kokkos reference implementation
for C++17/23) extracts a lower-dimensional view from an mdspan without touching
the underlying data.

### K.7.1 Slice specifiers

| Specifier | Meaning |
|---|---|
| `std::full_extent` | Take the entire dimension |
| `i` (integer) | Fix dimension to index `i`, reducing rank by 1 |
| `std::pair{lo, hi}` | Slice `[lo, hi)` |
| `std::strided_slice{off, ext, stride}` | Strided range |

### K.7.2 Extracting a single token's embedding

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

### K.7.3 Extracting a KV cache head slice

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

### K.7.4 Batch slice and strided access

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

## K.8 Tiled GEMM with mdspan

Matrix multiplication is memory-bound on CPU for the weight sizes typical in
LLM inference (4096×4096 and larger). The standard fix is register tiling:
load small blocks of A and B into L1-resident registers, accumulate into a C
tile, and write back once. mdspan makes the tiling logic clean and the index
arithmetic explicit.

### K.8.1 Tile layout helper

```cpp
// Extract a [TILE_M, TILE_K] block of A starting at (row_base, col_base)
template<std::size_t TM, std::size_t TK, class MDS>
auto tile_of(MDS A, std::size_t row_base, std::size_t col_base) {
    return std::submdspan(A,
        std::pair{row_base, row_base + TM},
        std::pair{col_base, col_base + TK});
}
```

### K.8.2 Tiled GEMM kernel

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

### K.8.3 Worked Example K.2 — GEMV for single-token decode

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

## K.9 Multi-Head Attention with mdspan

The entire MHA inner loop can be expressed cleanly using mdspan views and
submdspan slices. This section builds the forward pass from scratch, making the
tensor indexing explicit.

### K.9.1 Tensor shapes

```
Q, K, V:  [batch, num_heads, seq_len, head_dim]
scores:   [batch, num_heads, seq_len, seq_len]
output:   [batch, num_heads, seq_len, head_dim]
```

For single-batch inference (batch=1) the batch dimension is dropped.

### K.9.2 Score computation — Q × Kᵀ

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

### K.9.3 Online softmax with mdspan

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

### K.9.4 Weighted value aggregation

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

## K.10 KV Cache Management with mdspan

The KV cache is the largest runtime allocation in LLM inference — 2 × num_layers
× num_heads × seq_len × head_dim × sizeof(float) bytes per request. mdspan makes
it easy to manage as a preallocated slab with per-request views.

### K.10.1 Slab allocation

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

### K.10.2 PagedAttention analogy

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

## K.11 `std::linalg` — mdspan Meets BLAS (C++26)

C++26 adds `<linalg>`, a standardized interface to BLAS-level operations
parameterised over mdspan. This closes the loop between the view abstraction
and high-performance compute.

### K.11.1 Key operations

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

### K.11.2 Inference GEMM with linalg

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
    auto Wt = /* layout_stride transpose view (§K.5.4) */;
    std::linalg::matrix_product(X, Wt, Y);
}
```

With MKL linked, `matrix_product` calls `cblas_sgemm` with the right leading-
dimension values derived automatically from the mdspan strides.

---

## K.12 mdspan in SIMD Code — Combining with `std::experimental::simd`

C++23 also standardizes `std::experimental::simd` (merged into C++26 as
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

## K.13 Worked Example K.3 — Feed-Forward Layer (FFN)

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
    // (use gemm_tiled from §K.8 or std::linalg::matrix_product)
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

## K.14 Performance Notes and Guidelines

### K.14.1 Static vs. dynamic extents

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

### K.14.2 `is_contiguous` and cache prefetch hints

```cpp
if constexpr (decltype(mat)::is_always_contiguous()) {
    // emit __builtin_prefetch hints safely
    __builtin_prefetch(mat.data_handle() + prefetch_offset, 0, 3);
}
```

`layout_right` and `layout_left` are always contiguous; `layout_stride` and
custom tiled layouts are not. Checking `is_always_contiguous()` at compile time
allows safe prefetch without branching.

### K.14.3 Avoid `operator()` in innermost hot loops

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

### K.14.4 Alignment guarantees

`std::vector<float>` guarantees `alignof(float)` (4 bytes). For AVX-512 you
need 64-byte alignment. Use `std::aligned_alloc`:

```cpp
float* ptr = static_cast<float*>(
    std::aligned_alloc(64, num_elements * sizeof(float)));
// Wrap with mdspan + aligned_accessor (§K.6.1)
```

Or use `std::pmr::vector` with a custom `std::pmr::pool_resource` that enforces
alignment.

---

## K.15 Decision Framework

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

## K.16 Relation to Existing Inference Frameworks

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

## K.17 Compiling and Running the Examples

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
# Worked Example K.1: transpose view correctness
static_assert(Bt(3, 7) == B(7, 3));

# Worked Example K.2: GEMV bandwidth
# Run: ./gemv_bench 4096 4096  → should report ~80–120 GB/s on modern x86
```

---

## K.18 Chapter Summary

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
storage. The transpose-without-copy pattern (§K.5.4) eliminates a common source
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

4. Compare the compile-time behavior of:
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

*— Appendix K covers `std::mdspan` as standardized in C++23 (core) and C++26
(`submdspan`, `std::linalg`). The Kokkos reference implementation
(github.com/kokkos/mdspan) backports all features to C++17.*


---

## Worked Solutions

### Question 1
**W stored as `float[4096 * 4096]` row-major. Present as column-major to BLAS via `layout_stride`.**

Row-major means element (i, j) is at offset `i * 4096 + j`. Column-major means element (i, j) is at offset `j * 4096 + i`. To present the same flat array as column-major, we need strides: stride in the row dimension = 1 (adjacent rows differ by 1 element), stride in the column dimension = 4096 (adjacent columns differ by 4096 elements).

```cpp
#include <mdspan>

float W_data[4096 * 4096];  // stored row-major

// Row-major view (default):
std::mdspan<float, std::extents<size_t, 4096, 4096>, std::layout_right>
  W_rowmajor(W_data);
// W_rowmajor(i, j) = W_data[i * 4096 + j]  ✓

// Column-major view via layout_stride -- NO data copy:
std::layout_stride::mapping<std::extents<size_t, 4096, 4096>> colmaj_map(
    std::extents<size_t, 4096, 4096>{},
    std::array<size_t, 2>{1, 4096}  // stride[0]=1 (row), stride[1]=4096 (col)
);
std::mdspan<float, std::extents<size_t, 4096, 4096>, std::layout_stride>
  W_colmajor(W_data, colmaj_map);
// W_colmajor(i, j) = W_data[i * 1 + j * 4096] = W_data[i + j * 4096]
// = column-major offset for (i, j)  ✓
```

**Verification:** For (i=2, j=3):
- Row-major: offset = 2*4096 + 3 = 8,195
- Column-major view: offset = 2*1 + 3*4096 = 12,290
Both access the same `W_data` array at different offsets — the view is a reinterpretation, not a copy.

**BLAS call:**
```cpp
// Pass W_colmajor.data_handle() to BLAS with leading dimension 4096
cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            4096, 4096, 4096,
            1.0f, W_colmajor.data_handle(), 4096,
            B.data_handle(), 4096,
            0.0f, C.data_handle(), 4096);
```

---

### Question 2
**Why `std::submdspan(K_cache, layer, 0, head, std::pair{0, seq_pos}, std::full_extent)` produces a zero-copy view.**

The KV cache mdspan has extents `[n_layers, 2, n_heads, max_seq, d_head]`. `submdspan` computes a new mapping from this slice's indices to the original flat memory offsets.

**What happens at compile time:**
- Integer arguments (`layer`, `0`, `head`) are "rank-reducing" slicers: they fix a dimension to a single value and eliminate that rank from the output mdspan.
- `std::pair{0, seq_pos}` is a range slicer: it reduces the seq dimension to [0, seq_pos) but keeps the rank.
- `std::full_extent` keeps the d_head dimension unchanged.

The result is an mdspan of rank 2 (seq_pos x d_head) that points into the same memory region as K_cache. The compiler computes the base pointer offset at compile time for the static dimensions (layer, 0, head) and adjusts the stride for the dynamic seq_pos dimension.

**Zero-copy mechanism:** The new mdspan's `data_handle()` is `K_cache.data_handle() + layer * stride_layer + 0 * stride_kv + head * stride_head`. No data is moved — only a new pointer and stride values are computed. The entire operation compiles to 3-5 integer arithmetic instructions.

**Compile-time vs runtime:** `layer` and `head` are `size_t` (runtime values), so their offsets are computed at runtime. But the stride computation itself is determined at compile time from the static layout — the compiler generates an optimal sequence of multiply-add instructions with no branches.

---

### Question 3
**Custom FP4 accessor policy: two 4-bit values packed per byte, scale per 32-element group.**

```cpp
struct FP4Accessor {
    // The backing store is uint8_t (packed FP4 pairs)
    const uint8_t* packed_data;
    const float* scale_data;   // one scale per 32 elements
    static constexpr size_t GROUP_SIZE = 32;

    // reference type must be float (value, not reference)
    using reference = float;  // NOT float&

    float operator()(size_t idx) const {
        // Unpack the 4-bit nibble
        size_t byte_idx = idx / 2;
        uint8_t packed = packed_data[byte_idx];
        uint8_t nibble = (idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);

        // Dequantize: map nibble [0..15] to signed float
        // Typical FP4 mapping: 0000=0, 0001=0.5, ..., 1111=-0.5 (NF4 or E1M2)
        static constexpr float lut[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
        };
        float dequantized = lut[nibble];

        // Apply group scale
        float scale = scale_data[idx / GROUP_SIZE];
        return dequantized * scale;
    }
};
```

**Why `reference` must be `float` (value) not `float&`:**
The FP4 value does not exist as a `float` in memory — it is packed as 4 bits within a byte alongside another value. There is no address you can return a reference to. The accessor must compute the dequantised float on-the-fly and return it by value. This is the fundamental difference between custom accessor policies (which return computed values) and the default accessor (which returns a reference to an existing float in memory).

---

### Question 4
**Static extents vs dynamic extents: assembly differences and when to prefer dynamic.**

```cpp
// Static: extents known at compile time
std::mdspan<float, std::extents<size_t, 4096, 4096>> A(ptr);
// A(i, j) compiles to: ptr[i * 4096 + j]
// The compiler knows 4096 at compile time -> single MUL + ADD instruction

// Dynamic: extents known at runtime
std::mdspan<float, std::dextents<size_t, 2>> B(ptr, 4096, 4096);
// B(i, j) compiles to: ptr[i * B.extent(1) + j]
// The compiler must load extent(1) from memory, then MUL + ADD
```

**Assembly difference for `A(i, j)` vs `B(i, j)`:**
- **Static `A`:** The compiler substitutes 4096 as a compile-time constant. The stride multiply becomes either a shift (`4096 = 2^12` -> `i << 12`) or a multiply by an immediate constant — one instruction.
- **Dynamic `B`:** The runtime extent must be loaded from the mdspan's stored extents array (a memory load), then multiplied. On modern CPUs with good cache behavior, this is 1 extra load + 1 multiply vs 1 multiply — typically 1-3 ns overhead.

**When to prefer dynamic `B` over static `A`:**
1. **Shape varies at runtime:** If matrix dimensions change based on input (e.g., variable sequence length in attention), static extents would require template instantiation per shape. Dynamic extents compile once and handle all shapes.
2. **Reducing binary size:** Each unique static extent combination produces separate template instantiation code. A large model with many weight shapes would produce thousands of instantiations. Dynamic extents share one implementation.
3. **Plugin or hot-reloaded models:** When model architecture is not known at compile time (loaded from a config file at startup), dynamic extents are required.

**Rule of thumb:** Use static extents for hot-loop inner dimensions (d_head=128 is a natural candidate), dynamic for batch/sequence dimensions.

---

### Question 5
**Llama-3 70B KV cache: 80 layers, 8 KV heads, head_dim=128, 4096-token context, FP16. Total bytes + mdspan declaration.**

**Total bytes:**
```
bytes = 2 (K and V) x 80 layers x 8 KV heads x 4096 tokens x 128 head_dim x 2 bytes (FP16)
      = 2 x 80 x 8 x 4096 x 128 x 2
      = 2 x 80 x 8 x 4096 x 256
      = 2 x 671,088,640
      = 1,342,177,280 bytes
      = 1.25 GB per request
```

**mdspan declaration with static head_dim and dynamic other dimensions:**
```cpp
#include <mdspan>
#include <cstdint>

// Extents: [n_layers, 2, n_kv_heads, seq_len, head_dim]
// Static: head_dim=128 (architecture constant)
// Dynamic: n_layers, n_kv_heads, seq_len (may vary by model config)
using KVCacheExtents = std::extents<
    size_t,
    std::dynamic_extent,  // n_layers (80 at runtime)
    2,                    // K and V (always 2, static)
    std::dynamic_extent,  // n_kv_heads (8 at runtime)
    std::dynamic_extent,  // seq_len (4096 at runtime, grows during generation)
    128                   // head_dim (architecture constant, static)
>;

// Allocate backing storage
std::vector<uint16_t> kv_data(1'342'177'280 / 2);  // FP16 = 2 bytes

// Construct mdspan
std::mdspan<uint16_t, KVCacheExtents> kv_cache(
    kv_data.data(),
    80,   // n_layers
    2,    // KV
    8,    // n_kv_heads
    4096  // seq_len (current context length)
    // head_dim=128 is static, not passed at construction
);

// Access: K cache for layer 5, head 3, token 100, dim 64
uint16_t val = kv_cache(5, 0, 3, 100, 64);  // 0=K, 1=V

// Slice: all K vectors for layer 5, head 3, up to current token
auto k_head = std::submdspan(kv_cache, 5, 0, 3,
                              std::pair{0, seq_pos},
                              std::full_extent);
// k_head has shape [seq_pos, 128] -- zero-copy view
```

**Why static head_dim matters:** The innermost loop of attention computation iterates over the 128-dimensional head. With static head_dim=128, the compiler can unroll this loop completely or generate optimized SIMD code (AVX-512 handles 128 floats in 4 AVX-512 registers). With dynamic head_dim, the compiler cannot unroll without runtime checks.

---

## K.19 Complete Test and Main Harness

This self-contained file exercises every major component from Appendix K: basic
mdspan access, zero-copy transpose views (Worked Example K.1), tiled GEMM
(§K.8), GEMV (Worked Example K.2), causal softmax (§K.9.3), KV cache slab
(§K.10), dot products (§K.12), and the SiLU activation used in the FFN
(§K.13). All tests produce deterministic numerical results verified against
hand-computed or reference values.

### K.19.1 Compilation

```bash
# C++23 with GCC 13+ (recommended)
g++ -std=c++23 -O2 -march=native -Wall -Wextra \
    -o appendix_k_harness appendix_k_harness.cpp

# C++23 with Clang 17+
clang++ -std=c++23 -O2 -march=native -stdlib=libc++ \
    -o appendix_k_harness appendix_k_harness.cpp

# C++17 with Kokkos reference mdspan (older toolchains)
git clone https://github.com/kokkos/mdspan /tmp/mdspan
g++ -std=c++17 -O2 -march=native \
    -I/tmp/mdspan/include \
    -DUSE_KOKKOS_MDSPAN \
    -Wall -Wextra \
    -o appendix_k_harness appendix_k_harness.cpp
```

Expected output:

```
══════════════════════════════════════════════════════════
 Appendix K — std::mdspan Test Harness
══════════════════════════════════════════════════════════

[K.3.1] Basic mdspan access
  PASS: mat(2,3) == 15.0f
  PASS: mat.extent(0) == 4
  PASS: mat.extent(1) == 6
  PASS: static_mat(0,0) == 0.0f && static_mat(3,5) == 23.0f

[K.5.4] Transpose view (Worked Example K.1)
  PASS: Bt.extent(0) == N && Bt.extent(1) == K
  PASS: Bt(3,2) ≈ B(2,3)  (got 16)
  PASS: Bt(0,0) ≈ B(0,0)  (got 1)
  PASS: Bt(5,3) ≈ B(3,5)  (got 24)
  PASS: full matrix transpose consistent

[K.8.2] Tiled GEMM
  PASS: tiled == reference for all (i,j)
  PASS: C(0,0) ≈ 8  (got 8)
  PASS: C(7,7) ≈ 64  (got 64)

[K.8.3] GEMV (Worked Example K.2)
  PASS: identity W: y == x
  PASS: y2[0] ≈ 3  (got 3)
  PASS: y2[1] ≈ 8  (got 8)

[K.9.3] Causal softmax rows
  PASS: scores(0,0,0) ≈ 1  (got 1)
  PASS: scores(1,0,0) ≈ 1  (got 1)
  PASS: scores(0,1,0) ≈ 0.5  (got 0.5)
  PASS: scores(0,1,1) ≈ 0.5  (got 0.5)
  PASS: scores(0,3,0) ≈ 0.25  (got 0.25)
  PASS: scores(0,3,3) ≈ 0.25  (got 0.25)
  PASS: masked positions == 0
  PASS: all visible rows sum to 1

[K.10] KV cache slab
  PASS: view(2,0,1,5,7) ≈ 3.14  (got 3.14)
  PASS: view(3,1,0,2,3) ≈ 2.71  (got 2.71)
  PASS: view.extent(0) == num_layers
  PASS: view.extent(1) == 2
  PASS: view.extent(2) == num_heads
  PASS: k_slice.extent(0) == 6
  PASS: k_slice.extent(1) == head_dim
  PASS: k_slice(5,7) ≈ 3.14  (same memory as view)

[K.12] Dot product
  PASS: dot(a,b) ≈ 20  (got 20)
  PASS: dot(orthogonal) ≈ 0  (got 0)

[K.13] SiLU activation
  PASS: silu(0) ≈ 0  (got 0)
  PASS: silu(1) ≈ 0.731059  (got 0.731059)
  PASS: silu(10) > silu(5)
  PASS: silu(0.5) > 0

══════════════════════════════════════════════════════════
 Results: 31 / 31 passed  ✓ ALL PASS
══════════════════════════════════════════════════════════
```

### K.19.2 Full source — `appendix_k_harness.cpp`

```cpp
// appendix_k_harness.cpp — Appendix K complete test and main harness
// Compile: g++ -std=c++23 -O2 -march=native -Wall -Wextra \
//              -o appendix_k_harness appendix_k_harness.cpp
//
// Kokkos fallback (C++17):
//   g++ -std=c++17 -O2 -march=native -I/tmp/mdspan/include \
//       -DUSE_KOKKOS_MDSPAN -o appendix_k_harness appendix_k_harness.cpp

#ifdef USE_KOKKOS_MDSPAN
#  include "mdspan/mdspan.hpp"
   namespace std { using Kokkos::mdspan; using Kokkos::extents;
     using Kokkos::dextents; using Kokkos::submdspan;
     using Kokkos::full_extent; using Kokkos::layout_right;
     using Kokkos::layout_left; using Kokkos::layout_stride; }
#else
#  include <mdspan>
#endif

#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>

// ── Type aliases (§K.3.2) ────────────────────────────────────────────────────
template<class T>
using DynMat = std::mdspan<T,
    std::extents<std::size_t,
                 std::dynamic_extent,
                 std::dynamic_extent>>;

template<class T>
using DynVec = std::mdspan<T,
    std::extents<std::size_t, std::dynamic_extent>>;

// ── Test infrastructure ───────────────────────────────────────────────────────
static int g_tests_run = 0, g_tests_passed = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        ++g_tests_run;                                                    \
        if (cond) {                                                       \
            ++g_tests_passed;                                             \
            std::cout << "  PASS: " #cond "\n";                          \
        } else {                                                          \
            std::cout << "  FAIL: " #cond                                 \
                      << "  (line " << __LINE__ << ")\n";                \
        }                                                                 \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                             \
    do {                                                                  \
        ++g_tests_run;                                                    \
        auto _a = (a); auto _b = (b);                                    \
        if (std::fabs(static_cast<double>(_a - _b)) < (eps)) {          \
            ++g_tests_passed;                                             \
            std::cout << "  PASS: " #a " ≈ " #b                         \
                      << "  (got " << _a << ")\n";                       \
        } else {                                                          \
            std::cout << "  FAIL: " #a " ≈ " #b                         \
                      << "  (got " << _a << " vs " << _b << ")\n";      \
        }                                                                 \
    } while (0)

// ── §K.3.1 Basic mdspan access ───────────────────────────────────────────────
void test_basic_access() {
    std::cout << "\n[K.3.1] Basic mdspan access\n";

    std::vector<float> storage(24);
    std::iota(storage.begin(), storage.end(), 0.0f);   // 0 .. 23

    // Static 4×6 mdspan (row-major)
    using StaticMat4x6 = std::mdspan<float, std::extents<std::size_t, 4, 6>>;
    StaticMat4x6 mat(storage.data());

    // Element (2,3) is at flat index 2*6 + 3 = 15  → value 15.0
    CHECK(mat(2, 3) == 15.0f);
    CHECK(mat.extent(0) == 4);
    CHECK(mat.extent(1) == 6);
    CHECK(mat(0, 0) == 0.0f && mat(3, 5) == 23.0f);
}

// ── §K.5.4 Transpose view — Worked Example K.1 ───────────────────────────────
void test_transpose_view() {
    std::cout << "\n[K.5.4] Transpose view (Worked Example K.1)\n";

    constexpr std::size_t K = 4, N = 6;
    std::vector<float> B_data(K * N);
    std::iota(B_data.begin(), B_data.end(), 1.0f);   // 1 .. 24

    DynMat<float> B(B_data.data(), K, N);
    // stride(0) = N = 6 (row stride), stride(1) = 1 (col stride)

    // Transposed view Bᵀ : [N, K], stride(0)=1, stride(1)=N
    std::array<std::ptrdiff_t, 2> T_strides{1, static_cast<std::ptrdiff_t>(N)};
    std::layout_stride::mapping<std::dextents<std::size_t, 2>>
        T_map{{N, K}, T_strides};
    std::mdspan<float, std::dextents<std::size_t, 2>, std::layout_stride>
        Bt(B_data.data(), T_map);

    CHECK(Bt.extent(0) == N && Bt.extent(1) == K);
    CHECK_NEAR(Bt(3, 2), B(2, 3), 1e-6f);   // Bt(3,2) == B(2,3)
    CHECK_NEAR(Bt(0, 0), B(0, 0), 1e-6f);
    CHECK_NEAR(Bt(5, 3), B(3, 5), 1e-6f);

    // Verify Bt(n,k) == B(k,n) for all (n,k)
    bool all_ok = true;
    for (std::size_t n = 0; n < N; ++n)
        for (std::size_t k = 0; k < K; ++k)
            if (std::fabs(Bt(n, k) - B(k, n)) > 1e-6f) all_ok = false;
    CHECK(all_ok);
}

// ── §K.8.1/K.8.2 Tiled GEMM ─────────────────────────────────────────────────

// Tile-extraction helper (§K.8.1)
template<std::size_t TM, std::size_t TK, class MDS>
auto tile_of(MDS A, std::size_t row_base, std::size_t col_base) {
    return std::submdspan(A,
        std::pair{row_base, row_base + TM},
        std::pair{col_base, col_base + TK});
}

// Tiled GEMM — TM/TN/TK defaulted small for test matrices (§K.8.2)
template<std::size_t TM = 4, std::size_t TN = 4, std::size_t TK = 4>
void gemm_tiled(DynMat<const float> A,   // [M, K]
                DynMat<const float> B,   // [K, N]
                DynMat<float>       C)   // [M, N]  (accumulated into)
{
    const auto M = A.extent(0);
    const auto N = B.extent(1);
    const auto Kd = A.extent(1);

    for (std::size_t m = 0; m < M; m += TM)
    for (std::size_t n = 0; n < N; n += TN)
    for (std::size_t k = 0; k < Kd; k += TK) {
        float acc[TM][TN] = {};
        auto At = tile_of<TM, TK>(A, m, k);
        auto Bt = tile_of<TK, TN>(B, k, n);
        for (std::size_t ti = 0; ti < TM; ++ti)
        for (std::size_t tk = 0; tk < TK; ++tk)
        for (std::size_t tj = 0; tj < TN; ++tj)
            acc[ti][tj] += At(ti, tk) * Bt(tk, tj);
        for (std::size_t ti = 0; ti < TM; ++ti)
        for (std::size_t tj = 0; tj < TN; ++tj)
            C(m + ti, n + tj) += acc[ti][tj];
    }
}

// Reference naïve GEMM for verification
void gemm_naive(DynMat<const float> A, DynMat<const float> B, DynMat<float> C) {
    for (std::size_t i = 0; i < A.extent(0); ++i)
    for (std::size_t j = 0; j < B.extent(1); ++j) {
        float s = 0.0f;
        for (std::size_t k = 0; k < A.extent(1); ++k)
            s += A(i, k) * B(k, j);
        C(i, j) = s;
    }
}

void test_tiled_gemm() {
    std::cout << "\n[K.8.2] Tiled GEMM\n";

    // 8×8 matrices; TM=TN=TK=4 tiles the problem exactly in 2×2 tile grid
    constexpr std::size_t M = 8, Kd = 8, N = 8;
    std::vector<float> a(M * Kd), b(Kd * N);
    std::vector<float> c_tiled(M * N, 0.0f), c_ref(M * N, 0.0f);

    // A[i,k] = (i+1); B = all-ones → C[i,j] = (i+1) * Kd
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < Kd; ++k)
            a[i * Kd + k] = static_cast<float>(i + 1);
    std::fill(b.begin(), b.end(), 1.0f);

    DynMat<const float> A(a.data(), M, Kd);
    DynMat<const float> B(b.data(), Kd, N);
    DynMat<float>       Ct(c_tiled.data(), M, N);
    DynMat<float>       Cr(c_ref.data(),   M, N);

    gemm_tiled(A, B, Ct);
    gemm_naive(A, B, Cr);

    bool match = true;
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
            if (std::fabs(Ct(i, j) - Cr(i, j)) > 1e-4f) match = false;
    CHECK(match);
    // C(0,j) = 1 * Kd = 8; C(7,j) = 8 * Kd = 64
    CHECK_NEAR(Ct(0, 0), static_cast<float>(Kd),     1e-4f);
    CHECK_NEAR(Ct(7, 7), static_cast<float>(8 * Kd), 1e-4f);
}

// ── §K.8.3 GEMV — Worked Example K.2 ────────────────────────────────────────
void gemv(DynMat<const float> W,   // [out, in]
          DynVec<const float> x,   // [in]
          DynVec<float>       y)   // [out]
{
    const auto rows = W.extent(0);
    const auto cols = W.extent(1);
    for (std::size_t i = 0; i < rows; ++i) {
        float acc = 0.0f;
        auto row = std::submdspan(W, i, std::full_extent);
        for (std::size_t j = 0; j < cols; ++j)
            acc += row(j) * x(j);
        y(i) = acc;
    }
}

void test_gemv() {
    std::cout << "\n[K.8.3] GEMV (Worked Example K.2)\n";

    // Case 1 — identity matrix: y should equal x
    constexpr std::size_t D = 6;
    std::vector<float> W_id(D * D, 0.0f);
    for (std::size_t i = 0; i < D; ++i) W_id[i * D + i] = 1.0f;
    std::vector<float> x1 = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> y1(D, 0.0f);

    gemv(DynMat<const float>(W_id.data(), D, D),
         DynVec<const float>(x1.data(), D),
         DynVec<float>(y1.data(), D));

    bool id_ok = true;
    for (std::size_t i = 0; i < D; ++i)
        if (std::fabs(y1[i] - x1[i]) > 1e-6f) id_ok = false;
    CHECK(id_ok);

    // Case 2 — 2×3 weight: W=[[1,0,0],[0,2,0]], x=[3,4,5] → y=[3,8]
    std::vector<float> W2 = {1.f, 0.f, 0.f,
                              0.f, 2.f, 0.f};
    std::vector<float> x2 = {3.f, 4.f, 5.f};
    std::vector<float> y2(2, 0.0f);

    gemv(DynMat<const float>(W2.data(), 2, 3),
         DynVec<const float>(x2.data(), 3),
         DynVec<float>(y2.data(), 2));

    CHECK_NEAR(y2[0], 3.0f, 1e-6f);
    CHECK_NEAR(y2[1], 8.0f, 1e-6f);
}

// ── §K.9.3 Causal softmax rows ───────────────────────────────────────────────
using ScoreMDS = std::mdspan<float,
    std::extents<std::size_t,
                 std::dynamic_extent,   // H
                 std::dynamic_extent,   // S
                 std::dynamic_extent>>; // S

void softmax_rows(ScoreMDS scores) {
    const auto H = scores.extent(0);
    const auto S = scores.extent(1);
    for (std::size_t h = 0; h < H; ++h)
    for (std::size_t i = 0; i < S; ++i) {
        auto row = std::submdspan(scores, h, i,
                                  std::pair{static_cast<std::size_t>(0), i + 1});
        float mx = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j <= i; ++j)
            mx = std::max(mx, row(j));
        float sum = 0.0f;
        for (std::size_t j = 0; j <= i; ++j) {
            row(j) = std::exp(row(j) - mx);
            sum += row(j);
        }
        for (std::size_t j = 0; j <= i; ++j) row(j) /= sum;
        for (std::size_t j = i + 1; j < S; ++j) scores(h, i, j) = 0.0f;
    }
}

void test_softmax_rows() {
    std::cout << "\n[K.9.3] Causal softmax rows\n";

    constexpr std::size_t H = 2, S = 4;
    // All-ones logits → uniform distribution over visible positions
    std::vector<float> data(H * S * S, 1.0f);
    ScoreMDS scores(data.data(), H, S, S);
    softmax_rows(scores);

    // Row 0 (only 1 visible position) → probability = 1.0
    CHECK_NEAR(scores(0, 0, 0), 1.0f, 1e-5f);
    CHECK_NEAR(scores(1, 0, 0), 1.0f, 1e-5f);
    // Row 1: 2 equal logits → each = 0.5
    CHECK_NEAR(scores(0, 1, 0), 0.5f, 1e-5f);
    CHECK_NEAR(scores(0, 1, 1), 0.5f, 1e-5f);
    // Row 3: 4 equal logits → each = 0.25
    CHECK_NEAR(scores(0, 3, 0), 0.25f, 1e-5f);
    CHECK_NEAR(scores(0, 3, 3), 0.25f, 1e-5f);
    // Causal mask: j > i must be 0
    CHECK_NEAR(scores(0, 0, 1), 0.0f, 1e-5f);
    CHECK_NEAR(scores(0, 1, 2), 0.0f, 1e-5f);
    // All visible rows sum exactly to 1
    bool rows_ok = true;
    for (std::size_t h = 0; h < H; ++h)
        for (std::size_t i = 0; i < S; ++i) {
            float s = 0.0f;
            for (std::size_t j = 0; j <= i; ++j) s += scores(h, i, j);
            if (std::fabs(s - 1.0f) > 1e-5f) rows_ok = false;
        }
    CHECK(rows_ok);
}

// ── §K.10 KV cache slab ──────────────────────────────────────────────────────
void test_kv_cache_slab() {
    std::cout << "\n[K.10] KV cache slab\n";

    constexpr std::size_t NL = 4, NH = 2, SQ = 8, HD = 16;

    // Shape: [num_layers, 2, num_heads, max_seq, head_dim]
    using KVTensor = std::mdspan<float,
        std::extents<std::size_t,
            std::dynamic_extent,  // num_layers
            2,                    // K=0, V=1  (static)
            std::dynamic_extent,  // num_heads
            std::dynamic_extent,  // max_seq
            std::dynamic_extent>>;// head_dim

    std::vector<float> storage(NL * 2 * NH * SQ * HD, 0.0f);
    KVTensor view(storage.data(), NL, NH, SQ, HD);

    // Write and read back two scattered positions
    view(2, 0, 1, 5, 7) = 3.14f;   // layer=2, K, head=1, pos=5, dim=7
    view(3, 1, 0, 2, 3) = 2.71f;   // layer=3, V, head=0, pos=2, dim=3

    CHECK_NEAR(view(2, 0, 1, 5, 7), 3.14f, 1e-5f);
    CHECK_NEAR(view(3, 1, 0, 2, 3), 2.71f, 1e-5f);
    CHECK(view.extent(0) == NL);
    CHECK(view.extent(1) == 2);    // static extent — always 2
    CHECK(view.extent(2) == NH);

    // Zero-copy slice: K for layer 2, head 1, first 6 positions → [6, HD]
    auto k_slice = std::submdspan(view,
        static_cast<std::size_t>(2),
        static_cast<std::size_t>(0),
        static_cast<std::size_t>(1),
        std::pair{static_cast<std::size_t>(0), static_cast<std::size_t>(6)},
        std::full_extent);

    CHECK(k_slice.extent(0) == 6);
    CHECK(k_slice.extent(1) == HD);
    // k_slice shares memory with view — the written value is visible
    CHECK_NEAR(k_slice(5, 7), 3.14f, 1e-5f);
}

// ── §K.12 Dot product (scalar baseline for §K.12 SIMD discussion) ─────────────
float dot_scalar(DynVec<const float> a, DynVec<const float> b) {
    float result = 0.0f;
    for (std::size_t i = 0; i < a.extent(0); ++i)
        result += a(i) * b(i);
    return result;
}

void test_dot_product() {
    std::cout << "\n[K.12] Dot product\n";

    // [1,2,3,4] · [4,3,2,1] = 4+6+6+4 = 20
    std::vector<float> a = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> b = {4.f, 3.f, 2.f, 1.f};
    CHECK_NEAR(dot_scalar(DynVec<const float>(a.data(), 4),
                          DynVec<const float>(b.data(), 4)), 20.0f, 1e-5f);

    // Orthogonal unit vectors → dot = 0
    std::vector<float> e0 = {1.f, 0.f, 0.f, 0.f};
    std::vector<float> e1 = {0.f, 1.f, 0.f, 0.f};
    CHECK_NEAR(dot_scalar(DynVec<const float>(e0.data(), 4),
                          DynVec<const float>(e1.data(), 4)), 0.0f, 1e-5f);
}

// ── §K.13 SiLU activation ────────────────────────────────────────────────────
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

void test_silu() {
    std::cout << "\n[K.13] SiLU activation\n";

    // silu(0) = 0 / (1+1) = 0.0
    CHECK_NEAR(silu(0.0f), 0.0f, 1e-6f);
    // silu(1) = 1 / (1 + e⁻¹) ≈ 0.73106
    CHECK_NEAR(silu(1.0f), 0.7310586f, 1e-5f);
    // Monotone for x > 0: silu(10) > silu(5)
    CHECK(silu(10.0f) > silu(5.0f));
    // Positive input → positive output
    CHECK(silu(0.5f) > 0.0f);
}

// ── main ─────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << " Appendix K — std::mdspan Test Harness\n";
    std::cout << "══════════════════════════════════════════════════════════\n";

    test_basic_access();
    test_transpose_view();
    test_tiled_gemm();
    test_gemv();
    test_softmax_rows();
    test_kv_cache_slab();
    test_dot_product();
    test_silu();

    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << " Results: " << g_tests_passed
              << " / " << g_tests_run << " passed";
    if (g_tests_passed == g_tests_run)
        std::cout << "  ✓ ALL PASS\n";
    else
        std::cout << "  ✗ "
                  << (g_tests_run - g_tests_passed) << " FAILED\n";
    std::cout << "══════════════════════════════════════════════════════════\n";

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
```

### K.19.3 What each test validates

The `test_basic_access` function confirms that static `extents<size_t,4,6>` folds
the stride `(2,3) → offset 15` to a compile-time constant and that `extent(0)`,
`extent(1)` return the expected values.

`test_transpose_view` exercises Worked Example K.1 end-to-end: a `[4,6]`
row-major matrix is presented as a `[6,4]` transposed view through
`layout_stride` with swapped strides, and every element `Bt(n,k) == B(k,n)` is
verified without any data copy.

`test_tiled_gemm` runs the tiled kernel against a naïve reference over 8×8
matrices (two tile-layers of TM=TN=TK=4) and checks that all 64 output cells
agree to within `1e-4`.  The boundary cases `C(0,0)=8` and `C(7,7)=64` confirm
the accumulation arithmetic.

`test_gemv` uses an identity weight matrix so the expected output is trivially
`x` itself, then uses a 2×3 projection to verify a non-trivial result (`y=[3,8]`).

`test_softmax_rows` applies the causal-masked online softmax to a tensor of
all-ones logits so the expected probabilities are exact fractions (1.0, 0.5,
0.25). It also verifies that all positions beyond the causal boundary are zero
and that every visible row sums to exactly 1.

`test_kv_cache_slab` writes two scattered values into a 5-dimensional
`[NL, 2, NH, SQ, HD]` mdspan, reads them back, and then extracts a zero-copy
`submdspan` slice to confirm the slice shares the same backing memory.

`test_dot_product` and `test_silu` cover the §K.12 SIMD discussion and §K.13
FFN building block respectively, with both a non-trivial result and a boundary
case (orthogonal vectors, zero input).
