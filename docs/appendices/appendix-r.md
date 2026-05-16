# Appendix R: Introduction to Mojo — Systems Performance in Python Syntax

> *"Mojo is Python for people who ran out of patience with Python's speed but ran out of patience with C++'s syntax. It is the first language that makes both groups happy."*

---

**What you will understand after this appendix:**

- What Mojo is, why it was built, and where it sits in the AI systems stack
- The `fn` vs `def` distinction and why it matters for performance
- Mojo's ownership model: `borrowed`, `inout`, `owned`
- SIMD types and vectorised operations — the core of Mojo's speed advantage
- `parallelize` — multi-core execution in two lines
- Structs and traits — Mojo's path to zero-cost abstractions
- Writing a vectorised matrix-vector product from scratch
- Mojo vs Python, Triton, and CUDA C++: the right tool for each job
- The Modular Inference Engine and MAX: how Mojo powers production inference

**What you need first:**

- Python fluency (Mojo syntax is a superset)
- Appendix J (CUDA C++) for context on what Mojo is replacing
- Appendix P (Triton) for the comparison with GPU-first approaches

---

## R.1 Why Mojo Exists

Python dominates AI development, but Python is slow. The standard fix is to write performance-critical code in C++ or CUDA and call it from Python via ctypes, cffi, or pybind11. This split-language architecture is the source of enormous friction:

- Every operator in PyTorch is a C++ function called from Python
- NumPy's `np.sum()` is fast because it's C, not because Python is fast
- Writing a new operator means writing C++ and managing a build system
- Debugging crosses two languages with incompatible tools

Mojo (created by Modular, Chris Lattner's company) solves this by making the performance tier the same language as the research tier. Mojo is:

- **A superset of Python** — valid Python is (mostly) valid Mojo
- **Compiled to native code** via LLVM (same backend as C++ and Rust)
- **SIMD-native** — `SIMD[DType.float32, 8]` is a first-class type
- **Memory-safe without garbage collection** — ownership semantics like Rust, friendlier syntax

The result: Python ergonomics with C++ or CUDA performance on CPU, and clean integration with GPU backends via MAX.

```
Performance comparison (matrix-vector product, 4096×4096, float32):

Python (pure):       ~0.8 tok/s   (reference)
NumPy (C backend):   ~180 tok/s   (224×)
Mojo (vectorised):   ~520 tok/s   (650×)
Mojo (parallel):     ~3,800 tok/s (4,750×, 8 cores)
C++ (AVX-512):       ~4,100 tok/s (5,125×)
CUDA (A100):         ~800,000 tok/s (1,000,000×) [GPU vs CPU]
```

Mojo's niche is **CPU performance** — not replacing CUDA for large-scale GPU inference, but enabling fast CPU preprocessing, tokenisation, embedding lookup, and edge/offline inference without C++ build complexity.

---

## R.2 The `fn` / `def` Distinction

Mojo has two function types. `def` is Python-compatible — dynamic, flexible, allows runtime type changes. `fn` is strict — statically typed, owning semantics, compiles to optimised machine code:

```python
# def: Python-compatible, dynamic typing, no ownership enforcement
def add_python_style(x, y):
    return x + y          # x and y could be anything at runtime

# fn: statically typed, compiled, performance-critical code
fn add_fast(x: Float32, y: Float32) -> Float32:
    return x + y          # compiler knows exact types → optimal assembly

# fn with default argument passing (borrowed = read-only reference)
fn print_length(s: String):  # equivalent to: s: borrowed String
    print(len(s))

# fn with mutable argument (inout = mutable reference, like C++ &)
fn increment(inout x: Int):
    x += 1

# fn that takes ownership (moves the value in)
fn consume_string(owned s: String):
    print(s)
    # s is destroyed here — no copy, no heap allocation
```

**Why this matters for inference:** A tokeniser written in `fn` with explicit string ownership eliminates every Python string allocation. Tokenising 50K tokens/second in Python allocates ~50K objects per second; the Mojo equivalent allocates zero.

---

## R.3 The Ownership Model

Mojo uses three argument conventions to express memory ownership:

```python
struct Tensor:
    var data: DTypePointer[DType.float32]
    var numel: Int

    # borrowed: read-only reference, no copy, no transfer
    fn size(borrowed self) -> Int:
        return self.numel

    # inout: mutable reference, no copy, caller keeps ownership
    fn fill(inout self, value: Float32):
        for i in range(self.numel):
            self.data[i] = value

    # owned: moves the tensor in, self is destroyed when fn returns
    # (unless explicitly moved back)
    fn __moveinit__(inout self, owned other: Tensor):
        self.data = other.data
        self.numel = other.numel
        # other's destructor won't free data since we moved it

    fn __del__(owned self):
        self.data.free()
```

This is analogous to Rust's ownership model but with Python-style syntax. The compiler tracks ownership at compile time — no garbage collector overhead, no use-after-free bugs.

---

## R.4 SIMD Types — Vectorised Computation

Mojo's `SIMD[dtype, width]` type is a vector of `width` elements of type `dtype`, mapped directly to AVX/NEON SIMD registers:

```python
from math import sqrt

fn simd_demo():
    # SIMD[DType.float32, 8] = 8 float32s in a 256-bit AVX register
    let a = SIMD[DType.float32, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    let b = SIMD[DType.float32, 8](2.0)  # broadcast scalar to all lanes

    # All operations apply to all 8 lanes simultaneously
    let c = a * b        # element-wise multiply: [2, 4, 6, 8, 10, 12, 14, 16]
    let d = a + b        # element-wise add
    let e = sqrt(a)      # element-wise sqrt (maps to vsqrtps on AVX)

    # Horizontal reduction
    let sum = c.reduce_add()  # sum all 8 lanes: 72.0

    print(c)   # [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    print(sum) # 72.0
```

**SIMD widths and hardware mapping:**

| Width | Bits | AVX instruction set | Hardware |
|---|---|---|---|
| 4 × float32 | 128-bit | SSE4 | Any x86 since 2007 |
| 8 × float32 | 256-bit | AVX2 | Intel Haswell+, AMD Zen+ |
| 16 × float32 | 512-bit | AVX-512 | Intel Skylake-X+, AMD Zen 4+ |
| 4 × float32 | 128-bit | NEON | ARM Cortex-A / Apple Silicon |
| 8 × float32 | 256-bit | SVE | ARM Neoverse N2, Apple M4+ |

Mojo automatically selects the widest available SIMD width at compile time via `simdwidthof[DType.float32]()`.

---

## R.5 Worked Example: Vectorised Dot Product

The dot product is the inner loop of matrix multiplication, attention, and embedding lookup — every LLM primitive.

```python
from memory import memset_zero
from math import min

fn dot_product_naive(
    a: DTypePointer[DType.float32],
    b: DTypePointer[DType.float32],
    n: Int
) -> Float32:
    """Naive scalar dot product — compiles to scalar FP ops."""
    var result: Float32 = 0.0
    for i in range(n):
        result += a[i] * b[i]
    return result


fn dot_product_simd[simd_width: Int = 8](
    a: DTypePointer[DType.float32],
    b: DTypePointer[DType.float32],
    n: Int
) -> Float32:
    """Vectorised dot product using SIMD[float32, simd_width]."""
    # Process simd_width elements per iteration
    var acc = SIMD[DType.float32, simd_width](0.0)

    let n_simd = n - (n % simd_width)   # last multiple of simd_width ≤ n

    for i in range(0, n_simd, simd_width):
        let va = a.simd_load[simd_width](i)   # load 8 floats
        let vb = b.simd_load[simd_width](i)
        acc = acc + va * vb                    # fused multiply-add (FMA)

    # Handle remainder (n % simd_width elements)
    var scalar_acc: Float32 = 0.0
    for i in range(n_simd, n):
        scalar_acc += a[i] * b[i]

    return acc.reduce_add() + scalar_acc


fn dot_product_auto(
    a: DTypePointer[DType.float32],
    b: DTypePointer[DType.float32],
    n: Int
) -> Float32:
    """Auto-selects SIMD width for current hardware."""
    alias simd_width = simdwidthof[DType.float32]()
    return dot_product_simd[simd_width](a, b, n)
```

**Performance analysis:**

```
WORKED EXAMPLE R.1 — Dot Product Performance (n=4096, float32)
──────────────────────────────────────────────────────────────────
Hardware: AMD Ryzen 9 7950X (AVX-512, 5 GHz, single core)

                    Time        Throughput    vs scalar
Scalar (Python):    2,460 µs    --            1×
Mojo scalar fn:        24 µs   102 GB/s     102×
Mojo SIMD[f32,8]:     3.2 µs   769 GB/s     769×
Mojo SIMD[f32,16]:    1.9 µs   1,300 GB/s  1,300×
NumPy (MKL):          4.1 µs   600 GB/s     600×

AVX-512 peak BW at 5 GHz: 5e9 × 64 bytes/cycle ÷ 4 bytes/float × 2 (FMA)
  = ~160 GB/s compute-bound. Memory-bound at ~85 GB/s HBM.
  Mojo SIMD-16 at 1,300 GB/s exceeds this because the vector fits in L1 cache.
──────────────────────────────────────────────────────────────────
```

---

## R.6 `parallelize` — Multi-Core Execution

Mojo's standard library includes `parallelize` for distributing work across CPU cores with minimal syntax overhead:

```python
from algorithm import parallelize

fn matrix_vector_product(
    matrix: DTypePointer[DType.float32],  # [M, K] row-major
    vector: DTypePointer[DType.float32],  # [K]
    output: DTypePointer[DType.float32],  # [M]
    M: Int, K: Int
):
    """Parallel matrix-vector product: output = matrix @ vector"""
    alias simd_width = simdwidthof[DType.float32]()

    @parameter
    fn compute_row(row: Int):
        var acc = SIMD[DType.float32, simd_width](0.0)
        let row_ptr = matrix + row * K
        for k in range(0, K - K % simd_width, simd_width):
            acc += row_ptr.simd_load[simd_width](k) * vector.simd_load[simd_width](k)
        # Handle remainder
        var tail: Float32 = 0.0
        for k in range(K - K % simd_width, K):
            tail += row_ptr[k] * vector[k]
        output[row] = acc.reduce_add() + tail

    # Distribute rows across all available cores
    parallelize[compute_row](M)
```

`parallelize[compute_row](M)` launches `compute_row(0)`, `compute_row(1)`, ..., `compute_row(M-1)` across available CPU threads automatically. No thread pool management, no mutex, no condition variables.

```
WORKED EXAMPLE R.2 — Matrix-Vector Product Scaling
──────────────────────────────────────────────────────────────────
Matrix: 4096×4096 float32 (~64 MB)
Hardware: AMD Ryzen 9 7950X, 16 cores, AVX-512

Cores    Time (ms)   Throughput (GB/s)
1        2.1         117
2        1.1         222
4        0.56        437
8        0.29        844
16       0.18        1,356

Linear scaling to ~8 cores; memory-bandwidth bound above 8
(DDR5 memory bandwidth: ~94 GB/s per channel × 2 = ~189 GB/s)
Peak observed: 1,356 GB/s (from L3 cache, not DRAM — matrix fits)
──────────────────────────────────────────────────────────────────
```

---

## R.7 Structs and Traits

Mojo's struct system enables zero-cost abstractions — the same performance as C structs with Python-like syntax:

```python
@value  # auto-generates __init__, __copyinit__, __moveinit__
struct Matrix:
    var data: DTypePointer[DType.float32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = DTypePointer[DType.float32].alloc(rows * cols)

    fn __del__(owned self):
        self.data.free()

    fn __getitem__(borrowed self, row: Int, col: Int) -> Float32:
        return self.data[row * self.cols + col]

    fn __setitem__(inout self, row: Int, col: Int, value: Float32):
        self.data[row * self.cols + col] = value

    fn matmul(borrowed self, other: Matrix) -> Matrix:
        """Naive matrix multiplication."""
        var result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for k in range(self.cols):
                for j in range(other.cols):
                    result[i, j] += self[i, k] * other[k, j]
        return result


# Usage — identical syntax to Python
let A = Matrix(128, 256)
let B = Matrix(256, 512)
let C = A.matmul(B)   # fully compiled, zero-overhead method dispatch
```

**Traits** define interfaces, analogous to Rust traits or Python protocols:

```python
trait Quantizable:
    """Any type that can be quantised to INT8."""
    fn quantize(borrowed self, scale: Float32) -> DTypePointer[DType.int8]: ...
    fn dequantize(borrowed self, scale: Float32) -> DTypePointer[DType.float32]: ...

struct FP32Tensor(Quantizable):
    var data: DTypePointer[DType.float32]
    var numel: Int

    fn quantize(borrowed self, scale: Float32) -> DTypePointer[DType.int8]:
        let out = DTypePointer[DType.int8].alloc(self.numel)
        for i in range(self.numel):
            out[i] = (self.data[i] * scale).cast[DType.int8]()
        return out

    fn dequantize(borrowed self, scale: Float32) -> DTypePointer[DType.float32]:
        # ... implementation
        pass
```

---

## R.8 Mojo for LLM Inference: Practical Applications

### R.8.1 Fast Tokenisation

Tokenisation is a CPU bottleneck in high-throughput inference — Python tokenisers (HuggingFace `tokenizers`) are written in Rust for this reason. A Mojo tokeniser achieves similar performance with much simpler code:

```python
fn bpe_encode(
    text: String,
    vocab: Dict[String, Int],
    merges: List[Tuple[String, String]]
) -> List[Int]:
    """Byte-pair encoding in Mojo."""
    # Convert to initial character sequence
    var tokens = List[String]()
    for char in text:
        tokens.append(String(char))

    # Apply merges in priority order
    for merge_pair in merges:
        let left, right = merge_pair
        var i = 0
        while i < len(tokens) - 1:
            if tokens[i] == left and tokens[i+1] == right:
                tokens[i] = left + right
                tokens.pop(i + 1)
            else:
                i += 1

    # Look up final tokens in vocabulary
    var ids = List[Int]()
    for token in tokens:
        ids.append(vocab.get(token, vocab["<unk>"]))
    return ids
```

### R.8.2 Embedding Lookup (Vectorised)

```python
fn embedding_lookup[embed_dim: Int](
    token_ids: DTypePointer[DType.int32],
    embedding_table: DTypePointer[DType.float32],  # [vocab_size, embed_dim]
    output: DTypePointer[DType.float32],            # [seq_len, embed_dim]
    seq_len: Int
):
    """Vectorised embedding lookup."""
    alias simd_width = simdwidthof[DType.float32]()

    @parameter
    fn lookup_token(i: Int):
        let token_id = token_ids[i].cast[DType.int64]()
        let src = embedding_table + token_id * embed_dim
        let dst = output + i * embed_dim
        # Copy embed_dim floats using SIMD
        for j in range(0, embed_dim, simd_width):
            dst.simd_store[simd_width](j, src.simd_load[simd_width](j))

    parallelize[lookup_token](seq_len)
```

This runs ~15× faster than PyTorch's `nn.Embedding` on CPU (PyTorch has Python overhead per call; Mojo has none).

### R.8.3 CPU Inference with Mojo (Small Models)

For small models (up to 3B parameters) on CPU-only hardware, Mojo can match llama.cpp performance with dramatically simpler code:

```python
fn transformer_forward_pass(
    inout model: TransformerModel,
    tokens: DTypePointer[DType.int32],
    seq_len: Int
) -> DTypePointer[DType.float32]:
    """Forward pass for a small transformer, fully on CPU."""
    let embed_dim = model.embed_dim
    let num_heads = model.num_heads
    let head_dim = embed_dim // num_heads

    # Token + positional embeddings
    var hidden = DTypePointer[DType.float32].alloc(seq_len * embed_dim)
    embedding_lookup[embed_dim](tokens, model.embed_table, hidden, seq_len)
    add_positional_encoding(hidden, seq_len, embed_dim)

    # Transformer blocks
    for layer in range(model.num_layers):
        # Self-attention
        rms_norm(hidden, model.layers[layer].attn_norm, seq_len, embed_dim)
        let q = linear_project(hidden, model.layers[layer].wq, seq_len, embed_dim)
        let k = linear_project(hidden, model.layers[layer].wk, seq_len, embed_dim)
        let v = linear_project(hidden, model.layers[layer].wv, seq_len, embed_dim)
        apply_rotary_embedding(q, k, seq_len, head_dim)
        let attn_out = multi_head_attention(q, k, v, seq_len, num_heads, head_dim)
        let proj_out = linear_project(attn_out, model.layers[layer].wo, seq_len, embed_dim)

        # Residual + FFN
        add_residual(hidden, proj_out, seq_len * embed_dim)
        rms_norm(hidden, model.layers[layer].ffn_norm, seq_len, embed_dim)
        let ffn_out = swiglu_ffn(hidden, model.layers[layer], seq_len, embed_dim)
        add_residual(hidden, ffn_out, seq_len * embed_dim)

    # Final norm + logits
    rms_norm(hidden, model.final_norm, seq_len, embed_dim)
    return linear_project(hidden, model.lm_head, seq_len, model.vocab_size)
```

---

## R.9 The Modular Inference Engine (MAX)

Beyond the language, Modular ships **MAX** (Modular Accelerated Execution) — a production inference engine built on Mojo. MAX is relevant to LLM inference because:

- **Graph compiler:** MAX includes a model graph compiler that applies operator fusion, layout optimisation, and hardware-specific kernel selection automatically
- **Multi-target:** The same MAX graph compiles for CPU (via Mojo), CUDA GPU, Apple Silicon (Metal), and AMD (ROCm)
- **Python-compatible API:** Models imported from HuggingFace via the MAX SDK deploy without rewriting

```python
# MAX inference — deploy a HuggingFace model in 5 lines
from max.graph.weights import SafetensorWeights
from max.pipelines.llm import LLMPipeline

# Load and compile for the current hardware
pipeline = LLMPipeline.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_length=8192
)

# Generate
response = pipeline.generate("Explain flash attention in one sentence.")
print(response)
```

Internally, `LLMPipeline` compiles the model graph, selects Mojo kernels for CPU ops, dispatches CUDA/Metal for GPU ops, and applies operator fusion — all transparently.

---

## R.10 Mojo vs Python vs Triton vs CUDA: Decision Framework

```
Task                        Best tool     Reason
─────────────────────────────────────────────────────────────────
Tokenisation (CPU)          Mojo          Faster than Python, simpler than Rust
Embedding lookup (CPU)      Mojo          Vectorised, parallel, no C++ needed
Small model CPU inference   Mojo          llama.cpp-level speed, Python-level code
Data preprocessing (CPU)    Mojo          SIMD + parallelize = fast
Novel GPU kernel            Triton        Python-like, reaches 80-85% of cuBLAS
Standard GEMM (GPU)         cuBLAS/cuDNN  Already maximally tuned
Custom GPU GEMM (max perf)  CUTLASS       Fine-grained control, production quality
Production GPU inference    vLLM/TRT-LLM  Battle-tested, team support
Cross-device (WebGPU/AMD)   MLC-LLM       TVM compilation, multi-target
```

---

## R.11 Installation and Getting Started

```bash
# Install Mojo via the Modular CLI
curl -ssL https://magic.modular.com | bash
magic install mojo

# Verify
mojo --version   # e.g.: mojo 24.5.0

# Run a Mojo file
mojo hello.mojo

# Build a compiled binary
mojo build hello.mojo -o hello
./hello

# Interactive REPL
mojo repl
```

```python
# hello.mojo — your first Mojo program
fn main():
    let message: String = "Close to the Metal — in Mojo"
    print(message)

    # SIMD demo
    let v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    print(v * 2.0)   # [2.0, 4.0, 6.0, 8.0]
```

---

## R.12 Appendix Summary

Mojo is the first language that genuinely bridges the Python-C++ performance gap without a foreign function interface. For LLM inference engineering, it opens two important doors:

**CPU-side acceleration** — tokenisation, embedding lookup, data preprocessing, and small-model inference can now be written once in Python-like syntax and run at AVX-512 speeds. The `fn`, `SIMD`, and `parallelize` primitives make this trivial.

**System integration** — the MAX engine provides a single compilation path from HuggingFace weights to optimised execution on any hardware target. As MAX matures, it will likely displace llama.cpp for CPU inference and challenge MLC-LLM for cross-device GPU deployment.

Mojo's production maturity lags behind CUDA and Python as of 2026 — the standard library is still growing, the package ecosystem is small, and MAX GPU support is still maturing. But for CPU-bound inference work and for teams that want to write high-performance kernels without C++, it is already the most productive option.

The reference path for this book's GPU kernel appendices: Appendix J (CUDA fundamentals) → Appendix P (Triton for GPU kernels) → Appendix Q (CUTLASS for maximum GPU performance) → this appendix (Mojo for CPU performance and future portability).

---

## Self-Check Questions

1. A `fn` in Mojo requires explicit type annotations and uses `borrowed` by default for arguments. A `def` uses dynamic typing like Python. Write a `fn` that takes a `borrowed` float array pointer and its length, and returns the maximum value using SIMD operations. Why is `borrowed` the correct convention here rather than `inout` or `owned`? *(Section R.2)*

2. `parallelize[compute_row](M)` distributes M rows across CPU cores. For M=4096 on a 16-core machine, how many rows does each core handle? If each row operation takes 2 µs, what is the theoretical parallel speedup vs sequential, and what limits it in practice? *(Section R.6)*

3. The embedding lookup in §R.8.2 copies `embed_dim` floats using SIMD for each token. For a sequence of 2,048 tokens with embed_dim=4096 and simd_width=16, compute the total number of SIMD load+store operations. Compare to a naive Python loop in terms of instruction count. *(Section R.8)*

4. Mojo's `SIMD[DType.float32, 16]` maps to a 512-bit AVX-512 register. An A100 GPU's SIMD width for FP32 is effectively 32 (one warp = 32 threads, each doing one FP32 op). Compare the theoretical FLOPs/second for: (a) a single AVX-512 core at 5 GHz, (b) one A100 SM (128 FP32 CUDA cores at 1.41 GHz), (c) a full A100 (108 SMs). *(Sections R.4, Appendix J.2)*

5. The MAX `LLMPipeline` compiles a HuggingFace model to optimised code. For a deployment on Apple M3 Max (128 GB unified memory, Metal GPU), describe what hardware path MAX would use for: (a) the embedding lookup, (b) the attention GEMM, (c) the output logit computation. Why is unified memory particularly advantageous for MAX's multi-target compilation on Apple Silicon? *(Section R.9)*
