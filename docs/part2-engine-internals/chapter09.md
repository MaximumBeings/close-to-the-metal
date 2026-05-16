# Chapter 9 — The Forward Pass: CUDA vs. GGML

> *"PyTorch builds the graph dynamically every step and throws it away.
> GGML builds the graph statically once per step and executes it.
> CUDA graphs record the PyTorch graph once and replay it forever.
> Three philosophies, one goal: get the weights multiplied as fast as possible."*

---

## 9.0 Why This Chapter Matters

The forward pass is where all the memory planning of the previous chapters
pays off — or doesn't.  Every latency number in this book ultimately traces
back to what happens during the ~20 ms of each decode iteration.

This chapter dissects the forward pass in both engines:

- **vLLM**: a PyTorch dispatch pipeline culminating in FlashAttention and cuBLAS
  kernels, optionally wrapped in CUDA graph replay to eliminate Python overhead.

- **llama.cpp**: a GGML compute graph — a statically-scheduled DAG of tensor
  operations dispatched to CPU (AVX2/NEON), CUDA, or Metal backends.

By the end of this chapter you will be able to:

- Trace a single vLLM decode step from Python scheduler output to GPU kernel.
- Explain why CUDA graph replay eliminates ~5 ms of Python overhead per step.
- Describe the AllReduce communication cost in tensor-parallel deployments.
- Build and execute a minimal GGML compute graph from scratch.
- Explain how GGML dispatches each op to the appropriate backend.
- Compare dynamic (PyTorch) vs. static (GGML) compute graphs and their
  respective trade-offs.

---

## 9.1 What Happens During One Decode Step  `[FOUNDATIONAL]`

A single decode iteration processes one new token per active sequence.  For a
batch of B sequences the forward pass input is:

```
input_ids:    [B, 1]          ← one new token per sequence
positions:    [B, 1]          ← current position of each token
block_table:  [B, max_blocks] ← KV block mapping per sequence
kv_cache:     list of per-layer GPU tensors (Chapter 6)
```

The model computes:

```
token embedding lookup
  → transformer layer 0
      → RMSNorm + attention (FlashAttention over KV cache)
      → RMSNorm + FFN (SwiGLU: gate × up, then down projection)
  → transformer layer 1
  → ...
  → transformer layer L-1
  → final RMSNorm
  → lm_head matmul
  → logits [B, vocab_size]

sample next token: argmax / top-p / top-k
```

For LLaMA 3 8B (L=32, d=4096, d_ffn=14336) on A100 80GB:

```
FLOPs per decode step (B=64, 1 token per sequence):

  Per layer:
    Attention Q/K/V proj:  3 × 2 × 64 × 4096 × 4096 = 6.44 GFLOP
    Attention O proj:          2 × 64 × 4096 × 4096  = 2.15 GFLOP
    KV cache write:         negligible (memory write only)
    Attention QK^T + AV:    2 × 2 × 64 × 4096 × seq_len  ← depends on history
    FFN gate+up:            2 × 2 × 64 × 4096 × 14336    = 15.1 GFLOP
    FFN down:                   2 × 64 × 14336 × 4096    = 7.55 GFLOP
    RMSNorms (×2):          negligible
    ─────────────────────────────────────────────────────
    Per layer total:        ~31 GFLOP (ignoring KV attn)

  32 layers:                ~992 GFLOP
  lm_head:                  2 × 64 × 4096 × 128256 = 67 GFLOP

  Total per decode step:    ~1059 GFLOP

  A100 BF16 FLOP/s:         312 TFLOP/s
  Theoretical compute time: 1059 × 10⁹ / 312 × 10¹² ≈ 3.4 ms

  Actual decode time:       ~18–25 ms   ← memory-bandwidth bound, not compute
```

The 5× gap between theoretical compute time and actual time is because decode
is **memory-bandwidth bound** — the bottleneck is reading 16 GB of weights from
HBM (Chapter 2 §2.3), not the multiplications themselves.

---

## 9.2 vLLM's Forward Pass Pipeline  `[DEEP DIVE]`

### 9.2.1 From scheduler output to kernel

```
SchedulerOutputs
      │
      ▼
LLMEngine._process_model_outputs()
      │
      ├── _prepare_inputs()
      │     build:  input_ids, positions, block_tables, attention_metadata
      │
      ├── execute_model()
      │     │
      │     └── model.forward(input_ids, positions, kv_caches, attn_metadata)
      │           │
      │           ├── embed_tokens(input_ids)          → [B, d_model]
      │           ├── for layer in transformer_layers:
      │           │     ├── input_layernorm(hidden)    → rms_norm kernel
      │           │     ├── self_attn.forward()        → FlashAttention kernel
      │           │     ├── residual add                → element-wise add
      │           │     ├── post_attn_layernorm()      → rms_norm kernel
      │           │     └── mlp.forward()              → cuBLAS GEMM × 3
      │           ├── norm(hidden)
      │           └── lm_head(hidden)                 → cuBLAS GEMM
      │
      └── sampler.forward(logits, sampling_params)    → sampling kernels
```

### 9.2.2 The attention forward pass in detail

```python
# Simplified from vllm/model_executor/layers/attention/backends/flash_attn.py

class FlashAttentionImpl:

    def forward(
        self,
        query: torch.Tensor,      # [num_tokens, n_heads, d_head]
        key:   torch.Tensor,      # [num_tokens, n_kv_heads, d_head]
        value: torch.Tensor,      # [num_tokens, n_kv_heads, d_head]
        kv_cache: torch.Tensor,   # [2, num_blocks, block_size, n_kv_heads, d_head]
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:

        # 1. Write new K, V into the KV cache at the appropriate positions
        ops.reshape_and_cache_flash(
            key, value,
            kv_cache[0],            # key cache
            kv_cache[1],            # value cache
            attn_metadata.slot_mapping,   # where to write
        )

        # 2. Dispatch to correct kernel based on whether this is prefill or decode
        if attn_metadata.is_prompt:
            # Prefill: FlashAttention-2/3 forward (Chapter 5)
            output = flash_attn_varlen_func(
                query, key, value,
                cu_seqlens_q   = attn_metadata.seq_start_loc,
                cu_seqlens_k   = attn_metadata.seq_start_loc,
                max_seqlen_q   = attn_metadata.max_prompt_len,
                max_seqlen_k   = attn_metadata.max_prompt_len,
                causal         = True,
            )
        else:
            # Decode: paged attention kernel (FlashInfer or vLLM custom)
            output = flash_attn_with_kvcache(
                query,
                kv_cache[0],            # key cache (paged)
                kv_cache[1],            # value cache (paged)
                block_table    = attn_metadata.block_tables,
                cache_seqlens  = attn_metadata.seq_lens_tensor,
                softmax_scale  = self.scale,
                causal         = True,
            )

        return output.view(-1, self.n_heads * self.d_head)
```

### 9.2.3 The FFN forward pass

LLaMA 3's FFN uses SwiGLU (Swish-Gated Linear Unit):

```
FFN(x) = (gate_proj(x) ⊗ silu(up_proj(x))) @ down_proj

where:
  gate_proj: W_gate × x   → [n_tokens, d_ffn]    (cuBLAS GEMM)
  up_proj:   W_up   × x   → [n_tokens, d_ffn]    (cuBLAS GEMM)
  silu(z):   z × σ(z)      → element-wise
  ⊗:         element-wise product
  down_proj: W_down × (⊗) → [n_tokens, d_model]  (cuBLAS GEMM)
```

vLLM fuses the gate×silu operation into a single kernel to save a HBM round-
trip:

```python
# Fused SwiGLU: avoids writing intermediate gate/up tensors to HBM
output = ops.silu_and_mul(x, gate_up_combined)  # single CUDA kernel
```

---

## 9.3 CUDA Graph Replay on the Decode Path  `[DEEP DIVE]`

### 9.3.1 The capture-then-replay flow

Chapter 8 §8.6 described how graphs are captured at startup.  Here we focus on
the replay path during inference.

```
Decode iteration:

Without CUDA graphs:
  ┌──────────────────────────────────────────────────────────────┐
  │  Python scheduler   (1.5 ms)                                │
  │  Build input tensors (0.5 ms)                               │
  │  torch.ops dispatch × 32 layers (3.0 ms Python overhead)   │
  │  GPU compute        (18 ms)                                 │
  │  Sample + update    (0.7 ms)                                │
  │  ─────────────────────────────────────────────              │
  │  Total:             23.7 ms                                 │
  └──────────────────────────────────────────────────────────────┘

With CUDA graphs:
  ┌──────────────────────────────────────────────────────────────┐
  │  Python scheduler   (1.5 ms)                                │
  │  Fill input buffer  (0.1 ms)     ← in-place tensor writes   │
  │  graph.replay()     (0.01 ms)    ← single CUDA API call     │
  │  GPU compute        (18 ms)      ← unchanged                │
  │  Sample + update    (0.7 ms)                                │
  │  ─────────────────────────────────────────────              │
  │  Total:             20.3 ms    (~14% faster)                │
  └──────────────────────────────────────────────────────────────┘
```

The gain is largest for small batches.  At B=1 (single request, streaming):

```
B=1 GPU compute:      ~1.5 ms  (memory-bandwidth bound at small batch)
Python overhead:      ~4.0 ms  (same regardless of batch size)
Overhead fraction without graphs: 4.0 / (4.0 + 1.5) = 72 %!
Overhead fraction with graphs:    0.1 / (0.1 + 1.5) = 6 %

Speedup at B=1:  (4.0 + 1.5) / (0.1 + 1.5) = 3.4×
```

This is why CUDA graphs matter so much for streaming single-request workloads.

### 9.3.2 The static input buffer trick

CUDA graphs cannot be replayed with different input tensor addresses — the
graph records the *pointer values*, not the shapes alone.

vLLM solves this by maintaining **static input buffers** — pre-allocated tensors
that are mutated in-place before each replay:

```python
class ModelRunner:

    def __init__(self):
        # Allocated once at graph-capture time
        self.input_ids_static   = torch.zeros(max_batch, dtype=torch.long,   device="cuda")
        self.positions_static   = torch.zeros(max_batch, dtype=torch.long,   device="cuda")
        self.block_table_static = torch.zeros((max_batch, max_blocks),
                                              dtype=torch.int32, device="cuda")

    def execute_with_graph(self, scheduled_seqs):
        B = len(scheduled_seqs)
        graph_batch = pad_to_graph_size(B)

        # Fill static buffers in-place (no allocation)
        self.input_ids_static[:B]    = torch.tensor(
            [s.last_token_id for s in scheduled_seqs], device="cuda")
        self.positions_static[:B]    = torch.tensor(
            [s.position for s in scheduled_seqs], device="cuda")
        self.block_table_static[:B]  = build_block_table(scheduled_seqs)
        # Padding positions are already zeros from initialization

        # Replay: uses the pre-filled static buffers
        graph, _, output = self.cuda_graphs[graph_batch]
        graph.replay()

        return output[:B]   # discard padding
```

### 9.3.3 `[COMMON TRAP]` — Graph invalidation

CUDA graphs are invalidated whenever the PyTorch memory allocator moves any
tensor that is referenced in the graph.  This can happen if:

1. A separate GPU operation (e.g. a Triton kernel from a plugin) allocates
   memory that forces PyTorch's CUDA memory pool to defragment.
2. You call `torch.cuda.empty_cache()` during inference — never do this.
3. A tensor's data pointer changes between capture and replay.

Symptom: silent wrong outputs or CUDA illegal memory access errors.
Diagnosis: set `CUDA_LAUNCH_BLOCKING=1` and look for pointer mismatch errors.
Fix: ensure all tensors used by the graph are allocated in the static buffers
before capture and never reallocated.

---

## 9.4 Tensor Parallel Communication  `[DEEP DIVE]`

### 9.4.1 AllReduce after every layer

In a tensor-parallel deployment (e.g., 8 GPUs for LLaMA 3 70B), each GPU holds
a shard of every weight matrix (Chapter 8 §8.3.3).  The attention and FFN
computations produce *partial* results that must be summed across all GPUs before
the next layer can proceed.

This sum is an **AllReduce**: every GPU sends its partial result to all others
and receives the global sum.

```
GPU 0 partial:  [B, d_model]   → partial sum from head group 0-7
GPU 1 partial:  [B, d_model]   → partial sum from head group 8-15
...
GPU 7 partial:  [B, d_model]   → partial sum from head group 56-63

AllReduce  →  every GPU gets the sum of all 8 partial results
```

In PyTorch / vLLM:

```python
# After attention output projection (row parallel):
hidden = self.o_proj(attn_out)          # [B, d_model] — partial result
hidden = tensor_model_parallel_all_reduce(hidden)  # NVLink AllReduce

# After FFN down projection (row parallel):
hidden = self.down_proj(gate_up_out)    # [B, d_model] — partial result
hidden = tensor_model_parallel_all_reduce(hidden)  # NVLink AllReduce
```

### 9.4.2 AllReduce cost

```
AllReduce formula (ring algorithm):
  time = 2 × (N-1)/N × data_size / bandwidth

For 8 GPUs on NVLink (600 GB/s bidirectional per GPU):
  data_size = B × d_model × 2 bytes  (BF16)
            = 64 × 8192 × 2 = 1 MB   (LLaMA 3 70B, B=64)
  time = 2 × 7/8 × 1 MB / 600 GB/s
       = 2 × 0.875 × 10⁶ / 6×10¹¹
       = 2.9 µs   per AllReduce

  Per layer: 2 AllReduces (after attn + after FFN)
  All 80 layers: 80 × 2 × 2.9 = 464 µs ≈ 0.5 ms

Compare with PCIe (32 GB/s):
  time per AllReduce = 2 × 0.875 × 10⁶ / 3.2×10¹⁰ = 55 µs
  80 layers × 2: 8.8 ms  ← significant fraction of 20 ms decode step
```

This is why NVLink is critical for tensor-parallel LLM serving.  PCIe-connected
multi-GPU setups (e.g., consumer-grade multi-GPU rigs) pay a heavy AllReduce
tax that can dominate decode latency.

### 9.4.3 Fused AllReduce + quantization

vLLM 0.5+ optionally uses **INT8 AllReduce** (quantize activations to INT8
before AllReduce, dequantize after) to reduce the communication volume by 2×:

```
Standard:  1 MB BF16 per AllReduce @ 600 GB/s = 2.9 µs
INT8:      0.5 MB INT8 per AllReduce @ 600 GB/s = 1.5 µs   (≈2× faster)

Accuracy loss from INT8 AllReduce: < 0.1% on most benchmarks.
```

---

## 9.5 llama.cpp's GGML Compute Graph  `[DEEP DIVE]`

### 9.5.1 What is a GGML compute graph?

GGML (GPT-Generated Model Library) represents computation as an explicit
**directed acyclic graph (DAG)** of tensor operations, built fresh at the start
of each forward pass.

Unlike PyTorch's autograd graph (which exists for gradient computation) the
GGML graph is:

- **Statically allocated**: built into a pre-allocated `ggml_context` arena.
- **Forward-only** (for inference): no gradient tensors.
- **Eagerly scheduled**: topological sort is computed at build time.
- **Backend-dispatched**: each node carries a `backend` flag
  (CPU / CUDA / Metal) and is executed on the appropriate device.

### 9.5.2 Core GGML ops

| GGML op | C function | What it does |
|---------|-----------|--------------|
| Matrix multiply | `ggml_mul_mat(ctx, A, B)` | B×A (column-major convention) |
| Element add | `ggml_add(ctx, a, b)` | in-place element-wise add |
| RoPE | `ggml_rope(ctx, x, pos, n_dims, mode)` | Apply rotary position embedding |
| RMSNorm | `ggml_rms_norm(ctx, x, eps)` | Root-mean-square layer norm |
| SiLU | `ggml_silu(ctx, x)` | Sigmoid-gated linear unit activation |
| Element mul | `ggml_mul(ctx, a, b)` | Element-wise multiply (for SwiGLU gate) |
| Copy | `ggml_cpy(ctx, src, dst)` | Write into KV cache slice |
| View | `ggml_view_*(ctx, src, ...)` | Zero-copy reshaping / slicing |
| Permute | `ggml_permute(ctx, a, ...)` | Transpose / axis reorder |
| Concat | `ggml_concat(ctx, a, b, dim)` | Concatenate along an axis |
| Soft-max | `ggml_soft_max(ctx, x)` | Row-wise softmax |

### 9.5.3 Building the graph — one transformer layer

```c
// Simplified from llama.cpp/src/llama.cpp  build_llama_layer()

struct ggml_tensor* build_llama_layer(
    struct ggml_context* ctx,
    struct ggml_tensor*  cur,      // input hidden state [n_tokens, d_model]
    struct llama_layer*  layer,    // weight tensors for this layer
    struct ggml_tensor*  positions,// token positions [n_tokens]
    struct ggml_tensor*  kq_mask,  // causal mask [n_tokens, n_kv]
    struct ggml_tensor*  K_cache,  // KV cache key tensor
    struct ggml_tensor*  V_cache   // KV cache value tensor
) {
    struct ggml_tensor* residual = cur;

    // ── Attention ────────────────────────────────────────────────────────

    // RMSNorm
    cur = ggml_rms_norm(ctx, cur, 1e-5f);
    cur = ggml_mul(ctx, cur, layer->attn_norm_weight);   // × learned scale

    // Q, K, V projections
    struct ggml_tensor* Q = ggml_mul_mat(ctx, layer->Wq, cur); // [n_tokens, n_heads×d_head]
    struct ggml_tensor* K = ggml_mul_mat(ctx, layer->Wk, cur); // [n_tokens, n_kv×d_head]
    struct ggml_tensor* V = ggml_mul_mat(ctx, layer->Wv, cur); // [n_tokens, n_kv×d_head]

    // Reshape for multi-head layout
    Q = ggml_reshape_3d(ctx, Q, d_head, n_heads, n_tokens);
    K = ggml_reshape_3d(ctx, K, d_head, n_kv_heads, n_tokens);
    V = ggml_reshape_3d(ctx, V, d_head, n_kv_heads, n_tokens);

    // Apply RoPE to Q and K
    Q = ggml_rope(ctx, Q, positions, d_head, rope_mode);
    K = ggml_rope(ctx, K, positions, d_head, rope_mode);

    // Write K, V into KV cache
    struct ggml_tensor* K_cur = ggml_cpy(ctx, K,
                                  ggml_view_3d(ctx, K_cache, ...));  // slice
    struct ggml_tensor* V_cur = ggml_cpy(ctx, V,
                                  ggml_view_3d(ctx, V_cache, ...));

    // Scaled dot-product attention (or FlashAttention if available)
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);    // [d_head, n_tokens, n_heads, 1]
    struct ggml_tensor* KT = ggml_permute(ctx,
                                ggml_view_3d(ctx, K_cache, ...), ...);
    struct ggml_tensor* S  = ggml_mul_mat(ctx, KT, Q);   // scores
    S = ggml_scale(ctx, S, 1.0f / sqrtf(d_head));
    S = ggml_add(ctx, S, kq_mask);
    S = ggml_soft_max(ctx, S);
    struct ggml_tensor* V_view = ggml_view_3d(ctx, V_cache, ...);
    cur = ggml_mul_mat(ctx, V_view, S);       // context vectors

    // Output projection
    cur = ggml_permute(ctx, cur, 0, 2, 1, 3);
    cur = ggml_cont(ctx, cur);
    cur = ggml_mul_mat(ctx, layer->Wo, cur);

    // Residual add
    cur = ggml_add(ctx, cur, residual);
    residual = cur;

    // ── FFN ──────────────────────────────────────────────────────────────

    cur = ggml_rms_norm(ctx, cur, 1e-5f);
    cur = ggml_mul(ctx, cur, layer->ffn_norm_weight);

    struct ggml_tensor* gate = ggml_mul_mat(ctx, layer->Wgate, cur);
    struct ggml_tensor* up   = ggml_mul_mat(ctx, layer->Wup,   cur);
    gate = ggml_silu(ctx, gate);
    gate = ggml_mul(ctx, gate, up);                // SwiGLU
    cur  = ggml_mul_mat(ctx, layer->Wdown, gate);

    // Residual add
    cur = ggml_add(ctx, cur, residual);

    return cur;
}
```

### 9.5.4 Graph execution and backend dispatch

Once built, the graph is executed via:

```c
struct ggml_cplan plan = ggml_graph_plan(gf, n_threads);
ggml_graph_compute(gf, &plan);
```

`ggml_graph_compute` walks the topologically-sorted node list and, for each
node, dispatches to the appropriate backend:

```c
// Simplified from ggml/src/ggml.c

void ggml_compute_forward(struct ggml_compute_params* params,
                           struct ggml_tensor* tensor) {
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            switch (tensor->backend) {
                case GGML_BACKEND_CPU:
                    ggml_compute_forward_mul_mat_cpu(params, tensor); break;
                case GGML_BACKEND_CUDA:
                    ggml_cuda_op_mul_mat(tensor); break;
                case GGML_BACKEND_METAL:
                    ggml_metal_compute_tensor(tensor); break;
            }
            break;
        case GGML_OP_ROPE:
            ggml_compute_forward_rope(params, tensor); break;
        case GGML_OP_RMS_NORM:
            ggml_compute_forward_rms_norm(params, tensor); break;
        // ...
    }
}
```

For `GGML_OP_MUL_MAT` on CPU, the kernel selection depends on the quantization
type and available SIMD extensions:

```
Quantization  AVX2 (x86)         AVX-512 (x86)     NEON (ARM)
─────────────────────────────────────────────────────────────
Q4_0          ggml_vec_dot_q4_0_q8_0 (AVX2 path)    ARM path
Q4_K          q4_K_8_8 kernel       q4_K_8_8 AVX512  ARM path
Q8_0          q8_0_q8_0 (fast)      q8_0 AVX512      ARM NEON
F16           fp16_vec_dot          fp16 AVX512       ARM FP16
F32           BLAS (OpenBLAS/BLIS)  BLAS              Accelerate
```

### 9.5.5 Block-wise dequantization in `ggml_mul_mat`

For quantized weights (Q4_K, Q8_0, etc.) the matmul kernel does not first
dequantize the entire weight matrix.  It dequantizes and multiplies
**one block at a time** (32 weights per block), keeping only FP32 partial sums
in registers:

```c
// Pseudocode: Q4_K × F32 dot product for one row of the weight matrix

float dot = 0.0f;
for (int b = 0; b < n_blocks; b++) {
    // Load quantized block (4 bits × 32 = 16 bytes)
    const uint8_t* q4_block = &W_q4[b * 16];
    const float scale        = W_scales[b];

    // Load 32 activation values
    const float* act_block  = &x[b * 32];

    // Dequantize on the fly and accumulate
    for (int i = 0; i < 32; i++) {
        uint8_t q = (i < 16) ? (q4_block[i/2] & 0xF) : (q4_block[i/2] >> 4);
        float   w = (float)(q - 8) * scale;   // dequant to float
        dot += w * act_block[i];
    }
}
```

This "never materialise the dequantized matrix" approach is critical for memory
efficiency — the weight stays in its compact quantized form in HBM; only the
16-byte block needed for the current dot product is loaded at a time.

---

## 9.6 Dynamic vs. Static Compute Graphs  `[DEEP DIVE]`

### 9.6.1 Three graph philosophies

```
PyTorch eager (vLLM without graphs):
  Build graph dynamically every step → execute immediately.
  Pro:  maximum flexibility (variable shapes, dynamic control flow).
  Con:  Python dispatch overhead every step (~3 ms for 32 layers).

CUDA graph (vLLM default decode path):
  Build graph once (at capture time) → replay with fixed shapes.
  Pro:  eliminates Python overhead (~14% speedup for large batches,
        3× for small batches).
  Con:  shapes must be fixed; capture takes time at startup.

GGML static graph (llama.cpp):
  Build graph fresh each step in C (< 1 µs) → execute in C.
  No Python interpreter involved at all.
  Pro:  no Python overhead; portable (CPU/CUDA/Metal in one codebase).
  Con:  rebuilding the graph each step has small but non-zero overhead;
        optimization opportunities (kernel fusion) are limited.
```

### 9.6.2 Kernel fusion comparison

```
vLLM (cuBLAS + custom kernels):
  gate_up_proj:  cuBLAS GEMM     (standard cuBLAS)
  silu_and_mul:  fused CUDA kernel (gate × silu(up) in one pass)
  down_proj:     cuBLAS GEMM

  Fusion benefit: avoids writing 2 × [B, d_ffn] BF16 tensors to HBM.
  Memory saved: 2 × 64 × 14336 × 2 bytes = 3.7 MB per layer per step.
  Time saved: 3.7 MB / 2000 GB/s = 1.85 µs per layer → 59 µs total (32 layers).

GGML (custom kernels):
  SwiGLU: ggml_silu + ggml_mul  (two separate graph nodes)
  With CUDA backend: fused via custom CUDA kernel in ggml-cuda.cu.
  With CPU backend: two passes over the FFN intermediate tensor.
```

### 9.6.3 `[COMMON TRAP]` — GGML graph arena exhaustion

GGML pre-allocates a fixed-size arena for graph nodes and tensors:

```c
// Too small: common mistake
size_t ctx_size = 1024 * 1024;  // 1 MB — WAY too small for a 32-layer model

// Correct sizing: ≈ n_nodes × tensor_overhead + scratch
// A 32-layer LLaMA 3 8B graph has ~4000 nodes.
// Each ggml_tensor: ~400 bytes → 4000 × 400 = 1.6 MB
// Plus graph edge storage: ~0.5 MB
// Safe total: 4 MB minimum
size_t ctx_size = 4ull * 1024 * 1024;
```

If the arena is exhausted, `ggml_new_tensor()` returns NULL and the very next
`ggml_mul_mat()` call dereferences it → segfault or silent corruption.  Always
size the context with `llama_model_n_ctx_train()` as a guide and add margin.

---

## 9.7 Per-Layer Timing Profile  `[FOUNDATIONAL]`

On an A100 80 GB with LLaMA 3 8B, batch size B=64:

```
Layer component       Time (µs)   Memory read (MB)   Kernel
──────────────────────────────────────────────────────────────────────
RMSNorm (input)            28         0.5             rms_norm_cuda
QKV projection           1050        96.0             cuBLAS GEMM
RoPE                        45         1.0             rope_cuda
KV cache write              80         4.0             reshape_and_cache
FlashAttention              320        varies           flashattn_v2/v3
O projection               350        32.0             cuBLAS GEMM
AllReduce (8×GPU)           3          1.0             NCCL AllReduce
RMSNorm (post-attn)         28         0.5             rms_norm_cuda
Gate+Up projection        2100        192.0            cuBLAS GEMM
SiLU+Mul (fused)           180         7.5             silu_and_mul
Down projection           700         64.0             cuBLAS GEMM
AllReduce (8×GPU)           3          1.0             NCCL AllReduce
──────────────────────────────────────────────────────────────────────
Per-layer total           4887 µs     400 MB
× 32 layers           156 584 µs    12.8 GB total HBM reads
≈ 157 ms

Wait — that's for 32 layers × 64 seqs.  Per-step total: ~18 ms (measured).
The discrepancy: individual kernel times overlap (pipelined), and B=64 is
batched so per-sequence cost is already amortised.
```

The dominant cost is **QKV and FFN matmuls** (~80% of time), confirming the
memory-bandwidth-bound analysis from §9.1.

---

## 9.8 ASCII: One Forward Step — vLLM vs. llama.cpp  `[FOUNDATIONAL]`

```
vLLM (CUDA graph replay path):                 llama.cpp (GGML):
────────────────────────────────────────────  ──────────────────────────────────────────────────
Python scheduler (1.5 ms)                     C scheduler: check free slots (< 1 µs)
    │                                              │
Fill static input buffers (0.1 ms)            Build ggml_cgraph in arena (< 0.5 ms)
    │                                              │
cudaGraphLaunch(graph_64) (0.01 ms)           ggml_graph_plan() (< 0.1 ms)
    │                                              │
    └── [GPU executes the graph]                   └── ggml_graph_compute()
         embed_tokens kernel                             for each node in topo order:
         │                                                  dispatch to backend:
         ├── Layer 0                                        CPU: AVX2/AVX512/NEON
         │   rms_norm                                       CUDA: cuBLAS / custom
         │   QKV GEMM (cuBLAS)                              Metal: MTL kernels
         │   RoPE kernel
         │   KV cache write
         │   FlashAttn kernel
         │   O GEMM (cuBLAS)
         │   AllReduce (NVLink)
         │   rms_norm
         │   Gate+Up GEMM (cuBLAS)
         │   silu_and_mul kernel
         │   Down GEMM (cuBLAS)
         │   AllReduce (NVLink)
         ├── Layer 1 ...
         ...
         ├── Layer 31
         └── lm_head GEMM
    │
Sample (0.7 ms)                               Sample (< 0.1 ms in C)
    │                                              │
Stream token to caller                        Return next_token_id to caller
```

---

## 9.9 Selecting the Attention Backend  `[FOUNDATIONAL]`

vLLM automatically selects the best attention backend based on hardware and
sequence type:

```
At decode time (single new token per sequence):
  1. FlashInfer (vLLM ≥ 0.4, default): paged-KV-aware, decode-optimized
  2. Flash Attention 2 (fallback): standard tiled attention
  3. vLLM custom paged kernel (legacy)

At prefill time (many new tokens):
  1. Flash Attention 3 (H100): TMA async loads, warp specialization
  2. Flash Attention 2 (A100, fallback)
  3. xFormers (older hardware fallback)

Override:  VLLM_ATTENTION_BACKEND=flashinfer|flash_attn|xformers
```

For llama.cpp, the selection is compile-time:

```
# CUDA backend (default, recommended for NVIDIA)
cmake .. -DLLAMA_CUDA=ON

# Metal backend (Apple Silicon)
cmake .. -DLLAMA_METAL=ON

# CPU-only (portable fallback)
cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS
```

---

## 9.10 Code Listing  `[FOUNDATIONAL]`

```python
# forward_pass_demo.py
# Chapter 9 — The Forward Pass: CUDA vs. GGML
#
# Simulates and measures:
#   1. Forward pass operation sequence and FLOP accounting
#   2. CUDA graph vs. eager overhead model
#   3. AllReduce cost for tensor-parallel setups
#   4. Memory-bandwidth-bound performance model
#   5. Per-layer timing breakdown
#   6. GGML graph node count and arena sizing
#
# No GPU required — arithmetic simulation only.
#
# Run:
#   python forward_pass_demo.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math

# ... (see code/chapter_09/forward_pass_demo.py for full implementation)
```

See `code/chapter_09/forward_pass_demo.py` for the complete Python simulation
and `code/chapter_09/forward_pass_demo.cpp` for the full GGML-style DAG
implementation.

---

## 9.11 Chapter Summary

| Concept | Key fact |
|---------|----------|
| Decode FLOPs (8B, B=64) | ~1059 GFLOP; takes ~18 ms — memory-bandwidth bound |
| Python overhead (no graphs) | ~3–5 ms per step; 72 % overhead at B=1 |
| CUDA graph speedup | ~3.4× at B=1, ~14 % at B=64; single `cudaGraphLaunch` |
| Static input buffers | Tensors pre-allocated; filled in-place before each replay |
| AllReduce cost (NVLink 8×) | ~0.5 ms for 32 layers; ~8.8 ms over PCIe |
| GGML graph build time | < 0.5 ms per step (C arena allocation) |
| GGML block dequant | 32 weights dequantized at a time; never full matrix in DRAM |
| vLLM backend selection | Automatic: FlashInfer (decode) / FA3 (prefill H100) |
| llama.cpp backend | Compile-time: CUDA / Metal / CPU (AVX2/NEON) |
| SwiGLU fusion benefit | ~59 µs saved per step (32 layers × 1.85 µs) |

### Why this matters for what follows

- **Chapter 10** (Quantization) describes how quantized weights (Q4_K, INT8,
  FP8) change the matmul kernel inside `GGML_OP_MUL_MAT` and vLLM's GEMM path.

- **Chapter 11** (Speculative Decoding) adds a *draft model* forward pass before
  the main model; understanding the per-layer timing lets you estimate the
  draft overhead precisely.

- **Chapter 12** (Continuous Batching at Scale) shows how the per-step compute
  budget interacts with the scheduler's token budget.

---

## 9.12 Further Reading

- FlashAttention-2 paper: Dao, 2023.
- FlashAttention-3 paper: Shah et al., 2024.
- GGML source: `ggml/src/ggml.c` (`ggml_graph_compute`),
  `ggml/src/ggml-cuda.cu` (CUDA dispatch for each op).

- vLLM source: `vllm/model_executor/models/llama.py`,
  `vllm/attention/backends/flash_attn.py`.

- PyTorch CUDA graphs documentation:
  `https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs`

- NCCL AllReduce performance: `https://github.com/NVIDIA/nccl-tests`

---

*End of Chapter 9.*


---

## Chapter Summary

- **Two forward-pass modes**: prefill processes all prompt tokens in a single batched matmul; decode runs one new token per sequence per step.
- **CUDA graph replay**: decode steps are captured as CUDA graphs and replayed with updated KV cache pointers, eliminating per-step kernel launch overhead.
- **FlashAttention integration**: vLLM calls `flash_attn_varlen_func` for prefill (variable-length packed sequences) and `flash_attn_with_kvcache` for decode.
- **PagedAttention kernel**: the custom `paged_attention_v2` kernel accesses non-contiguous KV blocks via the block table during decode.
- **AllReduce placement**: in tensor-parallel mode, an AllReduce barrier is inserted after each row-parallel matmul (attention output projection, FFN down-projection).
- **Activation checkpointing**: not used in vLLM's inference path (only in training); all activations are recomputed by the graph replay.
- **llama.cpp compute graph**: `ggml_build_forward` constructs a DAG of tensor ops; `ggml_graph_compute` executes it using thread pools with NUMA affinity.

---

## Self-Check Questions

1. During decode, each sequence contributes exactly one query vector of shape (1, num_heads, d_k). The batch has 32 sequences. What shape is the batched Q tensor passed to the attention kernel? *(Section 9.2)*

2. CUDA graph replay requires all tensor shapes to be static. But the KV cache grows every step. How does vLLM handle this — what changes between replays and what stays fixed? *(Section 9.4)*

3. A tensor-parallel forward pass on 4 GPUs inserts 2 AllReduce calls per transformer block. For a 32-block model with blocks taking 1 ms each and AllReduce taking 0.1 ms, compute the overhead fraction. *(Section 9.3)*

4. The `paged_attention_v2` kernel must look up each KV block's physical address via the block table. Sketch the CUDA kernel structure: what data does each thread block access, and how is work divided across blocks and threads? *(Section 9.2)*

5. llama.cpp's ggml compute graph is rebuilt on every forward pass when sequences have different lengths. What is the cost of this rebuild, and how does llama.cpp minimize it for server workloads? *(Section 9.5)*


---

## Worked Solutions

---

### Solution 1 — Batched Q tensor shape for 32 decode sequences

**What we need:** Shape of the query tensor for 32 simultaneous decode sequences.

**Step 1 — Per-sequence query at decode time.**

At decode, each sequence generates exactly 1 new token. The query for sequence s at head h is a vector of shape `[1, d_k]`. With `num_heads` heads: the full per-sequence query is `[1, num_heads, d_k]`.

**Step 2 — Batched query.**

Stacking 32 sequences:

$$Q \text{ shape} = [32, \text{num\_heads}, 1, d_k] \quad \text{or equivalently} \quad [32, 1, \text{num\_heads}, d_k]$$

For LLaMA-3 8B (32 Q-heads, d_k=128): `[32, 32, 1, 128]` = 32 × 32 × 128 = 131,072 elements = 256 KB in BF16.

**Step 3 — Contrast with prefill.**

At prefill with sequence length T: Q shape = `[1, T, num_heads, d_k]`. The `T` dimension is what enables all-to-all attention during prefill. Decode's `1` in that position is what makes it memory-bound (tiny compute per large weight load).

---

### Solution 2 — How KV cache grows between CUDA graph replays

**What we need:** What changes between replays, what stays fixed.

**What STAYS FIXED (required for graph validity):**

- **Tensor shapes:** All input/output tensor shapes are identical across replays
- **KV cache buffer:** The buffer is preallocated at max size — its shape never changes
- **Graph structure:** Same sequence of CUDA kernels is called in the same order

**What CHANGES between replays:**

1. **Values in the KV buffer:** After each decode step, new K and V vectors for the current token are *written* into the preallocated buffer at the current position index. The buffer's shape doesn't change; only its contents do.

2. **Block table contents:** The mapping from sequence logical positions to physical block addresses is passed as an input tensor. Its *shape* is fixed (`[max_num_seqs, max_blocks_per_seq]`), but its *values* change as sequences advance.

3. **Sequence length counters:** A `seq_lens` tensor tracks how far each sequence has progressed. Its shape is fixed; values increment each step.

**Summary:** CUDA graphs require static shapes but allow dynamic values. vLLM engineers the API so that all variable state (KV positions, sequence lengths, block tables) lives in pre-allocated tensors that are updated in-place before each graph replay.

---

### Solution 3 — AllReduce overhead for 4-GPU tensor parallel

**Given:** 32 transformer blocks, 2 AllReduce per block, block_time=1 ms, AllReduce_time=0.1 ms

**Step 1 — Total AllReduce operations.**

$$2 \text{ AllReduce/block} \times 32 \text{ blocks} = 64 \text{ AllReduce calls}$$

**Step 2 — Total time breakdown.**

$$\text{compute time} = 32 \times 1 \text{ ms} = 32 \text{ ms}$$
$$\text{communication time} = 64 \times 0.1 \text{ ms} = 6.4 \text{ ms}$$
$$\text{total with TP} = 32 + 6.4 = 38.4 \text{ ms}$$

**Step 3 — Overhead fraction.**

$$\text{overhead} = \frac{6.4}{38.4} \approx \textbf{16.7\%}$$

**Step 4 — Practical context.**

NVLink on H100 provides 900 GB/s per GPU — AllReduce for a typical activation tensor (16 KB) takes ~18 μs, not 100 μs. The 0.1 ms figure represents PCIe-limited inter-node communication, which is why vLLM prefers intra-node tensor parallelism (NVLink) over inter-node (PCIe/InfiniBand). With NVLink, the 16.7% overhead drops to ~2%.

---

### Solution 4 — paged_attention_v2 kernel structure

**What we need:** Thread block organization and memory access pattern.

**High-level structure:**

The paged attention kernel computes attention for one query vector against a paged KV cache.

**Thread block assignment:**

- Each **CUDA thread block** handles: one (sequence, head) pair
- Grid: `[num_sequences, num_kv_heads]`

**Within a thread block:**

1. **Load query:** All threads collaboratively load the query vector `q[head_dim]` into shared memory (one load from HBM).

2. **Iterate over blocks:** For block_idx = 0, 1, ..., num_blocks_for_this_sequence:
   a. Look up physical block address: `phys_addr = block_table[seq_id][block_idx]`
   b. Load K block from `kv_cache[phys_addr, 0, :, :]` → shared memory (16 tokens × d_k)
   c. Compute dot products q · k_j for all j in this block → local scores
   d. Track running max for online softmax

3. **Second pass:** Reload V blocks, compute weighted sum using final softmax weights

4. **Write output:** Each thread writes its output element to global memory

**Key design decision:** The block table lookup (`phys_addr = block_table[seq_id][block_idx]`) introduces one extra memory indirection per block compared to contiguous attention. This is the cost of paging — mitigated by keeping the block table in shared memory.

---

### Solution 5 — ggml compute graph rebuild cost and minimization

**What we need:** Cost of rebuild, how server mode minimizes it.

**Step 1 — What triggers a rebuild.**

ggml represents computation as a directed acyclic graph (DAG) of tensor operations. The graph nodes specify which operations to run and on which tensors. If the sequence length changes (e.g., from 512 to 513 tokens), some node shapes change (attention mask, positional encoding tensors), requiring graph reconstruction.

**Step 2 — Rebuild cost.**

Rebuilding the ggml graph for a 32-layer LLaMA model involves:
- Allocating ~200–400 graph nodes (each is a small struct)
- Linking nodes in topological order
- Allocating workspace memory for new tensor shapes

Cost: approximately **0.5–2 ms** per rebuild on a modern CPU. This is small but becomes significant if triggered every single decode step (e.g., 35 ms/token becomes 37 ms/token — 6% overhead).

**Step 3 — Server mode minimization strategies.**

1. **Pre-build graphs for common lengths:** llama.cpp server builds and caches graphs for sequence lengths in discrete steps (e.g., 128, 256, 512, 1024, 2048). Most sequences fit an existing cached graph.

2. **Fixed-size batches:** By fixing batch size and context length at server startup, the graph shape becomes static and never needs rebuilding.

3. **Batched decode:** Process multiple sequences simultaneously with a fixed batch tensor shape — the graph shape depends on batch size (fixed) not per-sequence length (variable).

