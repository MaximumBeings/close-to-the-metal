# Chapter 8 — Startup and Initialization

> *"vLLM takes two minutes to start, then serves 3000 tokens per second.
> llama.cpp starts in five seconds, then serves 80 tokens per second.
> They are solving different problems."*

---

## 8.0 Why This Chapter Matters

Before a single token can be generated, both vLLM and llama.cpp must perform
substantial work: loading model weights, probing GPU memory, capturing compute
graphs, and sizing the KV cache pool.  This startup work pays dividends at
inference time — but it also defines the minimum cold-start latency, constrains
what can change at runtime, and reveals important architectural differences
between the two engines.

Understanding startup is also practically important:

- vLLM's **dummy forward pass** is the probe that makes the rest of the memory
  budget work.  If you misunderstand it you will misconfigure
  `--gpu-memory-utilization` and either OOM or leave GPU memory on the table.
- vLLM's **CUDA graph capture** is responsible for the 10–30× reduction in
  Python overhead on the decode path.  Without it each decode step would be
  bottlenecked by Python, not the GPU.
- llama.cpp's **mmap loading** is why it starts in seconds even for 70B models.
  Knowing when it is and isn't beneficial lets you tune cold-start vs.
  throughput trade-offs.

By the end of this chapter you will be able to:

- Trace vLLM's full startup sequence through five phases.
- Explain exactly why the dummy forward pass is necessary and what it measures.
- Describe CUDA graph capture and why it only covers a fixed set of batch sizes.
- Read a GGUF file header and understand the tensor metadata layout.
- Explain mmap-based loading, its zero-copy property, and when it hurts.
- Configure `--gpu-memory-utilization` (vLLM) and `--n-gpu-layers` (llama.cpp)
  with a firm understanding of the underlying memory arithmetic.

---

## 8.1 vLLM Startup — The Five Phases  `[FOUNDATIONAL]`

```
Phase 1: Configuration and validation
Phase 2: Weight loading
Phase 3: Dummy forward pass (memory probe)
Phase 4: KV block pool sizing
Phase 5: CUDA graph capture
```

For LLaMA 3 70B on 8× A100 80 GB (the typical production configuration) these
phases take approximately:

```
Phase 1: Configuration         ~1   s
Phase 2: Weight loading        ~90  s   (NVMe → GPU, FP16, 140 GB)
Phase 3: Dummy forward pass    ~8   s   (max sequence, max batch)
Phase 4: Block pool sizing     ~1   s   (accounting only)
Phase 5: CUDA graph capture    ~30  s   (capture 12 batch sizes)
─────────────────────────────────────
Total                          ~130 s
```

For LLaMA 3 8B on a single A100 80 GB:

```
Phase 2: Weight loading        ~15  s
Phase 3: Dummy forward pass    ~2   s
Phase 5: CUDA graph capture    ~10  s
Total                          ~28  s
```

Compare this with llama.cpp on the same 8B model: mmap + 35 GPU layers:

```
GGUF header parse:  ~0.1 s
Tensor mmap:        ~0.3 s
GPU layer upload:   ~4   s
Total:              ~4.4 s
```

The 6× difference is fundamental: vLLM optimises throughput at the cost of
startup; llama.cpp optimises startup (and flexibility) at the cost of
per-request throughput.

---

## 8.2 Phase 1 — Configuration and Validation  `[FOUNDATIONAL]`

vLLM parses engine arguments, validates model compatibility, and builds the
model configuration:

```python
# Simplified from vllm/engine/llm_engine.py

engine_config = EngineConfig(
    model=model_path,
    dtype="auto",                    # bfloat16 for Ampere/Hopper
    max_model_len=4096,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    block_size=16,
    ...
)

# Validate: can we fit the model on the specified GPUs?
ModelConfig.check_min_memory(engine_config)
```

The key check: `check_min_memory` estimates weight size from the model's
`config.json` and confirms that enough GPU memory exists before loading
anything.  For LLaMA 3 8B:

```
Weight estimate:
  Params:   8.03 × 10⁹
  dtype:    BF16  (2 bytes)
  Size:     8.03 × 10⁹ × 2 = 16.06 GB

A100 80 GB: 16.06 GB < 80 GB  ✓
```

---

## 8.3 Phase 2 — Weight Loading  `[FOUNDATIONAL]`

### 8.3.1 Sources

vLLM loads weights from:

```
HuggingFace SafeTensors:   model-00001-of-00004.safetensors  (most models)
HuggingFace PyTorch bin:   pytorch_model-00001-of-00004.bin
Meta checkpoints:          consolidated.00.pth
```

SafeTensors is preferred because it:
1. Does not execute arbitrary Python (unlike `.bin` which is `pickle`).
2. Supports memory-mapped loading for zero-copy weight access.
3. Has a header that describes tensor shapes before loading data.

### 8.3.2 The loading loop

```python
# Simplified from vllm/model_executor/models/llama.py

def load_weights(self, weights_iterator):
    for name, tensor in weights_iterator:
        # Map HuggingFace weight names to vLLM's internal naming
        mapped_name = _rename_hf_to_vllm(name)
        if mapped_name not in self.state_dict():
            continue

        param = self.state_dict()[mapped_name]

        # Tensor parallelism: shard Q/K/V and FFN weights across GPUs
        if _is_column_parallel(mapped_name):
            shard = tensor[rank * shard_size : (rank + 1) * shard_size]
            param.copy_(shard)
        elif _is_row_parallel(mapped_name):
            shard = tensor[:, rank * shard_size : (rank + 1) * shard_size]
            param.copy_(shard)
        else:
            param.copy_(tensor)
```

### 8.3.3 Tensor parallel sharding

For an 8-GPU tensor-parallel deployment of LLaMA 3 70B:

```
Full Q projection: [d_model, n_heads × d_head] = [8192, 64 × 128] = [8192, 8192]
Per-GPU shard:     [8192, 1024]   (8192 / 8 = 1024 columns)

Full FFN up-proj:  [d_model, d_ffn] = [8192, 28672]
Per-GPU shard:     [8192, 3584]   (28672 / 8 = 3584 columns)
```

After sharding, each GPU holds:

```
Per-GPU weight size = 140 GB / 8 = 17.5 GB
```

### 8.3.4 `[COMMON TRAP]` — dtype mismatch during load

vLLM loads weights in `bfloat16` by default.  If the checkpoint is in `float32`
(some older models) the memory usage during loading temporarily doubles as the
conversion happens:

```
float32 checkpoint: 32 GB
BF16 target:        16 GB
Peak during load:   32 + 16 = 48 GB   ← needs to fit in GPU memory
```

Set `--dtype bfloat16` explicitly to prevent vLLM from auto-detecting `float32`
and attempting an in-memory conversion that may OOM.

---

## 8.4 Phase 3 — The Dummy Forward Pass  `[DEEP DIVE]`

### 8.4.1 Why a dummy pass is necessary

After weights are loaded, vLLM knows exactly how much memory the weights use.
But it does *not* know how much memory a forward pass at maximum load will need
for **activations** — the intermediate tensors computed during each layer.

Activation memory depends on:
- Batch size (number of sequences)
- Sequence length (number of tokens per sequence)
- Model architecture (number of layers, attention heads, FFN width)

And crucially: **PyTorch's memory allocator does not release memory between
operations** — it caches GPU memory in a pool.  The only way to know the true
peak allocation is to actually run the forward pass.

### 8.4.2 What the dummy pass does

```python
# Simplified from vllm/worker/worker.py

def _profile_num_available_blocks(
    self,
    block_size: int,
    gpu_memory_utilization: float,
) -> Tuple[int, int]:

    # 1. Record memory BEFORE dummy pass
    torch.cuda.synchronize()
    free_before, total = torch.cuda.mem_get_info()

    # 2. Run forward pass at MAXIMUM possible load:
    #    - max_num_seqs sequences
    #    - each at max_model_len tokens
    dummy_input = _create_max_load_dummy_batch(
        max_num_seqs    = self.scheduler_config.max_num_seqs,
        max_model_len   = self.model_config.max_model_len,
        block_size      = block_size,
    )
    with torch.no_grad():
        self.model(**dummy_input)

    # 3. Record memory AFTER dummy pass
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()

    # 4. Peak activation usage = difference
    peak_activation_memory = free_before - free_after

    # 5. Available for KV blocks
    usable = total * gpu_memory_utilization
    kv_memory = usable - (total - free_before) - peak_activation_memory

    # 6. Convert bytes to block count
    kv_block_bytes = _kv_block_size_bytes(block_size, self.model_config)
    num_gpu_blocks = max(0, int(kv_memory // kv_block_bytes))

    return num_gpu_blocks, ...
```

### 8.4.3 Memory budget worked example

Configuration:

```
GPU:                 A100 80 GB
total:               80 GB  = 85 899 345 920 bytes
gpu_memory_utilization: 0.90
usable:              80 × 0.90 = 72 GB

Model:               LLaMA 3 8B
Weights (BF16):      16.1 GB  (measured after load)
max_num_seqs:        256
max_model_len:       4096
block_size:          16
```

**Before dummy pass**:

```
free_before = total - weight_memory
           = 80 - 16.1 = 63.9 GB
```

**Dummy pass at maximum load**:

```
Input shape: [256 × 4096] = [1 048 576 tokens]

Per-layer activation memory (BF16, LLaMA 3 8B):
  d_model = 4096, d_ffn = 14336
  Attention input:  [1M, 4096] × 2 bytes ≈ 8 GB
  Q/K/V projections: 3 × [1M, 4096] × 2 ≈ 24 GB
  FFN intermediate: [1M, 14336] × 2 ≈ 28 GB
  → Peak per layer: ~60 GB

But PyTorch's memory allocator reuses memory across layers.
Actual peak measured by mem_get_info: ~3.2 GB (with reuse)
```

**Available for KV blocks**:

```
usable          = 72.0 GB
weights         = 16.1 GB
peak_activations= 3.2  GB
                ─────────
kv_memory       = 52.7 GB

kv_block_bytes  = 2 × 32 × 16 × 8 × 128 × 2 = 2 097 152 bytes ≈ 2 MB
num_gpu_blocks  = 52.7 × 10⁹ / 2 097 152 ≈ 25 128 blocks
```

### 8.4.4 Sensitivity to `--gpu-memory-utilization`

```
gpu_memory_utilization   kv_memory   num_blocks   Max concurrent seqs
                                                   (at 512 tok avg)
────────────────────────────────────────────────────────────────────
0.80                     44.6 GB     21 251       ~660
0.85                     48.6 GB     23 156       ~724
0.90  (default)          52.7 GB     25 128       ~785
0.95                     56.7 GB     27 024       ~844
0.98                     59.1 GB     28 171       ~880
```

Setting `--gpu-memory-utilization 0.95` is safe on most models and yields ~7 %
more concurrent capacity.  Setting it above 0.95 risks OOM during large prefill
bursts (the dummy pass may underestimate activation usage under adversarial
inputs).

### 8.4.5 `[COMMON TRAP]` — Dummy pass underestimates activations

The dummy pass runs with all tokens at the *current* max batch size, but uses
**dummy (random) weights** for the attention patterns — which means the softmax
distribution may differ from real inputs.  Additionally, the KV cache is empty
during the dummy pass, so prefill attention is computed fresh.

In rare cases — long prompts with very uneven attention patterns that trigger
large temporary buffers inside FlashAttention — the dummy pass can underestimate
peak activation by 10–15 %.  If you see intermittent OOM errors during
production, reduce `--gpu-memory-utilization` by 0.02–0.03.

---

## 8.5 Phase 4 — KV Block Pool Sizing  `[FOUNDATIONAL]`

Using `num_gpu_blocks` from Phase 3, vLLM allocates the KV block storage:

```python
# The KV cache is a list of per-layer tensors
# Each tensor: [2, num_blocks, block_size, num_kv_heads, head_dim]

self.gpu_cache: List[torch.Tensor] = []
for layer_idx in range(num_layers):
    layer_cache = torch.zeros(
        (2, num_gpu_blocks, block_size, num_kv_heads, head_dim),
        dtype=cache_dtype,
        device="cuda",
    )
    self.gpu_cache.append(layer_cache)
```

For LLaMA 3 8B with 25 128 blocks:

```
Per-layer cache tensor:
  shape:  [2, 25128, 16, 8, 128]
  dtype:  bfloat16  (2 bytes)
  bytes:  2 × 25128 × 16 × 8 × 128 × 2 = 2.09 × 10⁹ ≈ 2.07 GB

Total across 32 layers:  32 × 2.07 = 66.2 GB  ← close to kv_memory (52.7 GB)
```

Wait — 66 GB ≠ 52.7 GB?  The discrepancy is because `num_blocks` is computed
from `kv_memory` *divided by block size in bytes*, which already includes the
per-layer factor.  Let me re-check:

```
kv_block_bytes per layer = 2 × 1 × 16 × 8 × 128 × 2 = 65 536 bytes
Total across all layers  = 32 × 65 536 = 2 097 152 bytes ≈ 2 MB  ✓ (matches §8.4.3)

So 25 128 blocks × 2 MB/block = 52.7 GB  ✓
```

The formula `kv_block_bytes = 2 × n_layers × block_size × n_kv_heads × d_head × bytes_per_elem`
from Chapter 6 already accounts for all layers.  The per-layer tensor is
allocated separately for indexing convenience but the total bytes are correct.

---

## 8.6 Phase 5 — CUDA Graph Capture  `[DEEP DIVE]`

### 8.6.1 The Python overhead problem

Without CUDA graphs, every decode step involves:

```
Python overhead per step (approximate):
  Scheduler._schedule()              ~1.5 ms
  Build input tensors                ~0.5 ms
  PyTorch op dispatch (32 layers)    ~3.0 ms
  Sampling + token update            ~0.5 ms
  Streaming                          ~0.2 ms
  ───────────────────────────────────────────
  Total Python overhead              ~5.7 ms

Actual GPU compute (decode, bs=64):  ~18 ms
GPU idle fraction:                   5.7 / (18 + 5.7) ≈ 24 %
```

For small decode batches (bs=1–8) the GPU compute is even shorter, making
Python overhead the dominant cost.

### 8.6.2 What CUDA graphs do

A CUDA graph records a sequence of CUDA operations (kernel launches, memory
copies) into a replayable object.  Replay bypasses the Python dispatch overhead:

```
Standard path:
  Python → torch.ops → CUDA API → GPU kernels
  (repeated for every op in every layer, every step)

CUDA graph replay:
  Python → cudaGraphLaunch(captured_graph)  ← single API call
  GPU executes the entire forward pass internally
```

The speedup is most pronounced for small batches where the Python overhead
dominates.

### 8.6.3 Why only specific batch sizes are captured

CUDA graphs are **static**: once captured, the graph has fixed-size input
tensors.  You cannot replay a graph captured for batch size 32 with a batch of
48 tokens — the shapes don't match.

vLLM handles this by capturing graphs for a **set of batch sizes**:

```python
# vllm/worker/worker.py
GRAPH_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 256]
```

At inference time, the actual batch size is **padded up** to the nearest
captured size:

```
Actual decode batch: 37 tokens
Nearest captured:    40 tokens
Padding tokens:      3 (dummy tokens with seq_id=-1)

Padded input fed to graph_40.replay()
After replay: discard logits for dummy positions.
```

The padding cost is small — the padded tokens still run through the network but
their logits are discarded in sampling.

### 8.6.4 Capture procedure

```python
for batch_size in GRAPH_BATCH_SIZES:
    # 1. Warmup: run 3 eager forward passes at this batch size
    #    (populates PyTorch's memory pool, JIT-compiles fused ops)
    for _ in range(3):
        _forward_eager(dummy_inputs[batch_size])

    # 2. Create CUDA graph
    graph = torch.cuda.CUDAGraph()

    # 3. Record
    with torch.cuda.graph(graph):
        output = self.model(**dummy_inputs[batch_size])

    # 4. Store
    self.cuda_graphs[batch_size] = (graph, dummy_inputs[batch_size], output)
```

The graph captures **all CUDA operations** between `torch.cuda.graph()`'s
context manager, including:
- All 32 (or 80) transformer layer forward passes
- FlashAttention kernels for each layer
- FFN matmuls
- RMS norm kernels
- RoPE application
- Sampling (if captured; optional)

### 8.6.5 Memory overhead of captured graphs

Each captured graph holds a copy of the **input and output tensor buffers**.
For a batch-256 graph:

```
Input  buffer: [256, d_model] in BF16 = 256 × 4096 × 2 = 2 MB
Output buffer: [256, vocab_size] in FP32 = 256 × 128 256 × 4 = 131 MB

Per-graph overhead: ~133 MB
15 captured graphs: ~2 GB
```

This is why Phase 5 takes time and uses memory: each captured graph must
allocate its buffers before recording.

### 8.6.6 CUDA graph replay path

At decode time:

```python
def _run_workers_with_graph(self, seq_len: int):
    # Find the smallest captured batch ≥ seq_len
    batch_size = _next_captured_size(seq_len, GRAPH_BATCH_SIZES)

    graph, inputs, output = self.cuda_graphs[batch_size]

    # Fill input buffer in-place (no new allocation)
    inputs["input_ids"][:seq_len]    = self.pending_token_ids
    inputs["positions"][:seq_len]    = self.pending_positions
    inputs["block_table"][:seq_len]  = self.pending_block_table
    inputs["input_ids"][seq_len:]    = 0   # padding

    # Single CUDA API call: replays entire forward pass
    graph.replay()

    # Read outputs
    return output[:seq_len]
```

The `graph.replay()` call is essentially:

```c
// Under the hood: replays all recorded CUDA operations
cudaGraphLaunch(exec_graph, stream);
cudaStreamSynchronize(stream);
```

### 8.6.7 When CUDA graphs are NOT used

vLLM falls back to eager (non-graph) mode for:

1. **Prefill steps**: variable-length prompt processing cannot be captured.
2. **Batch sizes larger than max captured** (>256 in the default config).
3. **When `--enforce-eager` flag is set** (disables all graphs; useful for debugging).
4. **Chunked prefill iterations** where the prefill chunk changes size.

The decode path uses graphs; the prefill path is always eager.

---

## 8.7 The Full vLLM Startup Timeline — ASCII Diagram  `[FOUNDATIONAL]`

```
t=0                                                                   t≈130s
│                                                                          │
├─── Phase 1: Config (1s) ─────────────────────────────────────────────── │
│    parse args, validate, build ModelConfig                               │
│                                                                          │
├─── Phase 2: Weight Load (90s) ───────────────────────────────────────── │
│    │                                                                     │
│    ├── Open SafeTensors files                                            │
│    ├── For each weight shard (4 files × 35 GB each):                    │
│    │     mmap file → GPU  (PCIe DMA, ~1.5 GB/s NVMe read + 32 GB/s PCIe)│
│    │     shard for tensor parallel                                       │
│    │     copy to correct device                                          │
│    └── All 70B params on GPU                                             │
│                                                                          │
├─── Phase 3: Dummy Forward (8s) ──────────────────────────────────────── │
│    │                                                                     │
│    ├── Create dummy batch: 256 seqs × 4096 tokens                       │
│    ├── torch.cuda.synchronize()                                          │
│    ├── mem_get_info() → free_before                                      │
│    ├── model.forward(dummy_batch)  ← full pass, max load                │
│    ├── torch.cuda.synchronize()                                          │
│    ├── mem_get_info() → free_after                                       │
│    └── peak_activations = free_before - free_after                      │
│                                                                          │
├─── Phase 4: Block Pool (1s) ─────────────────────────────────────────── │
│    │                                                                     │
│    ├── num_blocks = kv_memory / kv_block_bytes                          │
│    └── torch.zeros([2, num_blocks, 16, 8, 128]) × 80 layers             │
│                                                                          │
└─── Phase 5: CUDA Graph Capture (30s) ─────────────────────────────────  │
     │                                                                     │
     ├── For each batch_size in [1, 2, 4, 8, ...  256]:                   │
     │     warmup(3 eager passes)                                          │
     │     graph = CUDAGraph()                                             │
     │     with torch.cuda.graph(graph): model.forward(dummy)             │
     │     store graph + input/output buffers                              │
     └── 15 graphs captured, ~2 GB graph memory                           │
                                                                           │
     READY TO SERVE REQUESTS  ◄────────────────────────────────────────── │
```

---

## 8.8 llama.cpp Startup  `[FOUNDATIONAL]`

### 8.8.1 The GGUF file format

GGUF (GPT-Generated Unified Format) is the binary format used by llama.cpp.  A
GGUF file consists of:

```
GGUF file layout:
  ┌──────────────────────────────────────────┐
  │  Magic bytes: "GGUF"  (4 bytes)          │  ← format identifier
  │  Version:  3          (uint32)           │  ← GGUF version
  │  n_tensors            (uint64)           │  ← number of tensors
  │  n_kv                 (uint64)           │  ← number of key-value metadata pairs
  ├──────────────────────────────────────────┤
  │  KV pairs (metadata):                   │
  │    "llama.context_length"    → 4096      │
  │    "llama.embedding_length"  → 4096      │
  │    "llama.block_count"       → 32        │
  │    "llama.attention.head_count" → 32     │
  │    "llama.attention.head_count_kv" → 8  │  ← GQA
  │    "llama.feed_forward_length" → 14336  │
  │    "tokenizer.ggml.model"    → "llama"  │
  │    "tokenizer.ggml.tokens"   → [...]    │  ← vocabulary
  │    "general.quantization_version" → 2  │
  │    ...                                  │
  ├──────────────────────────────────────────┤
  │  Tensor info (per tensor):              │
  │    name:      "blk.0.attn_q.weight"      │
  │    n_dims:    2                          │
  │    shape:     [4096, 4096]               │
  │    dtype:     Q4_K  (integer code)       │
  │    offset:    892834816   (from data start)│
  │  ...  (n_tensors entries)               │
  ├──────────────────────────────────────────┤
  │  DATA SECTION (aligned to 32 bytes)     │
  │    tensor_0 data...                     │
  │    tensor_1 data...                     │
  │    ...                                  │
  └──────────────────────────────────────────┘
```

The header is read in full at startup; only then does llama.cpp know the offset
of each tensor in the file.

### 8.8.2 mmap-based loading

llama.cpp uses `mmap()` to map the GGUF file directly into the process's virtual
address space:

```c
// Simplified from llama.cpp/src/llama.cpp

void* mmap_ptr = mmap(
    NULL,                    // let OS choose virtual address
    file_size,               // size of mapping
    PROT_READ,               // read-only
    MAP_SHARED,              // shared mapping (backed by file)
    fd,                      // file descriptor
    0                        // offset from start of file
);

// Each tensor now has a pointer:
tensor->data = (char*)mmap_ptr + tensor->offset;
```

**The key property of mmap**: the file data is *not* read from disk until the
memory pages are actually accessed.  The OS only brings pages into RAM (via
page faults) when a computation touches them.

Consequence for startup:

```
llama_model_load():
  1. Open GGUF file.            ← stat + open:  instant
  2. mmap file.                 ← mmap syscall:  ~0.3 s  (maps 4 GB)
  3. Parse tensor headers.      ← read header:   ~0.1 s
  DONE — model "loaded" in ~0.4 s
  (Actual weight data not yet in RAM)

First inference:
  CPU layers: page faults on demand as each weight matrix is read.
  GPU layers: cudaMemcpy from mmap region to GPU VRAM (~4 s for 35 layers).
```

### 8.8.3 `--n-gpu-layers` offload

The `--n-gpu-layers N` flag tells llama.cpp to move the top N transformer
layers from CPU to GPU at startup:

```
llama.cpp startup with --n-gpu-layers 35 (LLaMA 3 8B, 32 layers):

  Layers 0–31:  all transformer layers → GPU
  Embedding + lm_head: optionally on GPU

  Per-layer GPU memory (Q4_K_M quantization, 4-bit):
    Weight memory: ≈ 32 / 8 × layer_params ≈ 200 MB per layer
    All 32 layers: ≈ 6.4 GB

  vs FP16:  32 layers × ~500 MB = 16 GB
```

The upload loop:

```c
// For each layer assigned to GPU:
for (int i = 0; i < n_layers_on_gpu; i++) {
    for each tensor in layer[i]:
        cudaMalloc(&gpu_ptr, tensor->nbytes);
        cudaMemcpy(gpu_ptr, tensor->data,   // mmap source
                   tensor->nbytes,
                   cudaMemcpyHostToDevice);
        tensor->data = gpu_ptr;
        tensor->backend = GGML_BACKEND_GPU;
}
```

### 8.8.4 mmap vs eager load — when mmap hurts

mmap is excellent for:
- **Cold start** (first inference starts immediately without waiting for full load).
- **Memory-mapped file sharing** (multiple processes can share the same pages).
- **Large models on memory-limited systems** (only needed pages are resident).

mmap is worse than eager loading for:
- **Sequential full-model access** (e.g., a single long generation that reads
  every weight): eager loading enables the OS prefetcher to stream data without
  page-fault overhead.
- **NVMe drives with high seek latency** (random page faults are expensive).
- **Production servers** (where you want predictable latency, not page-fault spikes).

For production llama.cpp deployments, consider:

```bash
# Lock all model pages into RAM (prevent page faults during inference)
llama-server --model model.gguf --n-gpu-layers 99 --mlock
```

`--mlock` calls `mlock()` on all mapped pages, forcing them into physical RAM
immediately.  Startup takes longer but inference has no page-fault latency spikes.

### 8.8.5 Startup timeline — llama.cpp

```
t=0                                                        t≈4.5s
│                                                               │
├── Open GGUF file (0.1s)                                      │
│   read magic, version, n_tensors, n_kv                       │
│                                                               │
├── Parse KV metadata (0.1s)                                   │
│   context_length, embedding_dim, vocab, ...                  │
│                                                               │
├── Parse tensor info headers (0.1s)                           │
│   names, shapes, dtypes, file offsets for 291 tensors        │
│                                                               │
├── mmap data section (0.3s)                                   │
│   4 GB file → virtual address space                          │
│   (no physical I/O yet)                                       │
│                                                               │
├── Upload GPU layers (4.0s)                                   │
│   32 layers × ~125 MB = 4 GB                                 │
│   PCIe bandwidth: ~32 GB/s → 4 / 32 ≈ 0.125 s per GB        │
│   overhead + sync: total ~4 s                                 │
│                                                               │
└── READY  ◄────────────────────────────────────────────────── │
```

---

## 8.9 Comparing the Two Startup Philosophies  `[FOUNDATIONAL]`

```
                     vLLM                        llama.cpp
                     ────────────────────────    ────────────────────────
Weight format:       SafeTensors / BF16           GGUF (any quant type)
Loading strategy:    Eager (copy to GPU VRAM)     mmap + selective upload
Memory probe:        Dummy forward pass           None (flat KV buffer)
Graph capture:       Yes (15+ batch sizes)        No
Startup time:        30–130 s                     1–10 s
Weight precision:    BF16 (default)               Q4/Q5/Q8 (common)
GPU memory split:    Automatic (dummy pass)        Manual (--n-gpu-layers)
KV cache strategy:   PagedAttention               Flat pre-allocated buffer
Cold-start latency:  High                         Low
Steady-state tput:   High (800–3000 tok/s)        Moderate (80–300 tok/s)
```

---

## 8.10 Tuning Startup for Production  `[FOUNDATIONAL]`

### 8.10.1 Reducing vLLM startup time

```bash
# 1. Use a local model path (avoid HuggingFace Hub download at startup)
vllm serve /local/models/llama-3-8b-instruct

# 2. Skip CUDA graph capture if P99 decode latency is not critical
vllm serve ... --enforce-eager

# 3. Reduce max captured batch sizes
# (edit vllm/worker/worker.py GRAPH_BATCH_SIZES to your actual traffic range)

# 4. Use tensor parallel to distribute weight load time
vllm serve ... --tensor-parallel-size 4
# Each GPU loads 17.5 GB instead of 70 GB → 4× faster load
```

### 8.10.2 Reducing llama.cpp startup time

```bash
# 1. Pre-warm by running a dummy generation at startup
echo "warmup" | llama-cli -m model.gguf --n-gpu-layers 99 -n 1 --prompt "warmup"

# 2. For persistent servers, keep the model loaded between requests
# (llama-server does this by default)

# 3. Use --mlock to avoid page-fault spikes after startup
llama-server -m model.gguf --n-gpu-layers 99 --mlock
```

---

## 8.11 Code Listing  `[FOUNDATIONAL]`

```python
# startup_demo.py
# Chapter 8 — Startup and Initialization
#
# Simulates and measures:
#   1. GGUF header parsing
#   2. Memory budget calculation (dummy-pass model)
#   3. CUDA graph sizing and capture overhead
#   4. Block pool sizing with varying gpu_memory_utilization
#   5. Startup timeline comparison: vLLM vs llama.cpp model
#
# No GPU required — all computations are arithmetic simulations.
#
# Run:
#   python startup_demo.py

import struct
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ── Model configurations ──────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name:           str
    n_layers:       int
    d_model:        int
    n_heads:        int
    n_kv_heads:     int
    d_head:         int
    d_ffn:          int
    vocab_size:     int
    max_model_len:  int

LLAMA3_8B = ModelConfig(
    name="LLaMA 3 8B",
    n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8,
    d_head=128,  d_ffn=14336,  vocab_size=128256, max_model_len=8192,
)

LLAMA3_70B = ModelConfig(
    name="LLaMA 3 70B",
    n_layers=80, d_model=8192, n_heads=64, n_kv_heads=8,
    d_head=128,  d_ffn=28672,  vocab_size=128256, max_model_len=8192,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Weight size calculation
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 1: Model Weight Size Estimation")
print("=" * 65)

def weight_bytes(cfg: ModelConfig, bytes_per_param: float = 2.0) -> int:
    """
    Estimate total model weight bytes.
    Includes: embedding, all transformer layers, lm_head.
    """
    d, H, H_kv, d_h, d_ff, L, V = (
        cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_head,
        cfg.d_ffn, cfg.n_layers, cfg.vocab_size
    )
    # Per-layer params
    # Attention: Q, K, V, O projections
    attn_q  = d * H    * d_h    # Q: d_model → n_heads * d_head
    attn_k  = d * H_kv * d_h    # K: d_model → n_kv_heads * d_head
    attn_v  = d * H_kv * d_h    # V: same as K
    attn_o  = H * d_h  * d      # O: n_heads * d_head → d_model
    # FFN: gate, up, down
    ffn     = d * d_ff + d * d_ff + d_ff * d   # gate + up + down
    # RMS norms: 2 per layer × d
    norms   = 2 * d
    layer_params = attn_q + attn_k + attn_v + attn_o + ffn + norms

    total_params = (
        V * d         +    # token embedding
        L * layer_params + # transformer layers
        d             +    # final norm
        V * d              # lm_head
    )
    return int(total_params * bytes_per_param)

for cfg in [LLAMA3_8B, LLAMA3_70B]:
    w = weight_bytes(cfg)
    print(f"\n  {cfg.name}:")
    print(f"    BF16 weight size: {w / 1e9:.2f} GB")
    print(f"    FP32 weight size: {weight_bytes(cfg, 4.0) / 1e9:.2f} GB")
    # Q4_K_M ≈ 4.5 bits per weight
    print(f"    Q4_K_M estimate:  {weight_bytes(cfg, 4.5/8) / 1e9:.2f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Dummy-pass memory budget
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SECTION 2: KV Block Pool Sizing (Dummy-Pass Budget)")
print("=" * 65)

def kv_block_bytes_total(cfg: ModelConfig, block_size: int) -> int:
    """Bytes for one KV block across all layers."""
    return (2 * cfg.n_layers * block_size *
            cfg.n_kv_heads * cfg.d_head * 2)  # BF16 = 2 bytes

def activation_memory_estimate(cfg: ModelConfig,
                                max_num_seqs: int,
                                max_model_len: int) -> int:
    """
    Rough estimate of peak activation memory during a forward pass.
    PyTorch allocator reuses memory across layers, so peak ≈ 2-3 layers.
    """
    n_tokens = max_num_seqs * max_model_len   # maximum token count
    d = cfg.d_model
    d_ff = cfg.d_ffn
    # Peak: attention input + Q/K/V + attention output + FFN intermediate
    # (2 layers worth, due to PyTorch memory reuse across layers)
    peak = n_tokens * (d + 3*d + d + d_ff) * 2  # BF16
    # Practical measured value is usually ~10-20x smaller due to reuse
    # Use conservative 10x reduction factor
    return peak // 10

def compute_num_blocks(gpu_gb: float,
                       gpu_util: float,
                       cfg: ModelConfig,
                       block_size: int = 16,
                       max_num_seqs: int = 256) -> int:
    total_bytes = gpu_gb * 1e9
    usable      = total_bytes * gpu_util
    weights     = weight_bytes(cfg)
    activations = activation_memory_estimate(cfg, max_num_seqs, cfg.max_model_len)
    kv_memory   = usable - weights - activations
    if kv_memory <= 0:
        return 0
    block_bytes = kv_block_bytes_total(cfg, block_size)
    return int(kv_memory / block_bytes)

print(f"\n  LLaMA 3 8B on A100 80 GB (block_size=16):")
print(f"\n  {'gpu_util':<12} {'kv_memory_GB':<16} {'num_blocks':<12} {'max_seqs@512tok'}")
print("  " + "-" * 55)
for util in [0.80, 0.85, 0.90, 0.92, 0.95]:
    blocks = compute_num_blocks(80, util, LLAMA3_8B)
    kv_mem = (blocks * kv_block_bytes_total(LLAMA3_8B, 16)) / 1e9
    max_seqs = int(blocks * 16 / 512)
    print(f"  {util:<12.2f} {kv_mem:<16.2f} {blocks:<12} {max_seqs}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CUDA graph capture sizing
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SECTION 3: CUDA Graph Capture Overhead")
print("=" * 65)

# Default vLLM captured batch sizes
GRAPH_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 256]

def graph_input_buffer_bytes(cfg: ModelConfig, batch_size: int) -> int:
    """Memory for input/output buffers of a captured graph."""
    input_bytes  = batch_size * cfg.d_model * 2           # BF16 activations
    logit_bytes  = batch_size * cfg.vocab_size * 4        # FP32 logits
    return input_bytes + logit_bytes

total_graph_bytes = sum(
    graph_input_buffer_bytes(LLAMA3_8B, bs)
    for bs in GRAPH_BATCH_SIZES
)

print(f"\n  Captured batch sizes: {GRAPH_BATCH_SIZES}")
print(f"  Total graph buffer overhead: {total_graph_bytes / 1e9:.3f} GB")
print(f"\n  {'Batch':<8} {'Buffer size':<14} {'Padding waste (worst case)'}")
print("  " + "-" * 50)
for i, bs in enumerate(GRAPH_BATCH_SIZES):
    prev = GRAPH_BATCH_SIZES[i-1] if i > 0 else 0
    worst_pad = bs - prev - 1
    buf_mb = graph_input_buffer_bytes(LLAMA3_8B, bs) / 1e6
    print(f"  {bs:<8} {buf_mb:<14.2f} MB   {worst_pad} tokens padding (worst)")

# Demonstrate the padding logic
print(f"\n  Padding demonstration:")
test_batch_sizes = [3, 10, 25, 37, 65, 100]
for actual in test_batch_sizes:
    padded = next((b for b in GRAPH_BATCH_SIZES if b >= actual), GRAPH_BATCH_SIZES[-1])
    waste  = padded - actual
    print(f"    actual={actual:4d}  → use graph_{padded:3d}  "
          f"(padding={waste:3d} tokens, {waste/padded*100:.1f}% waste)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: GGUF header parsing simulation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SECTION 4: GGUF Header Structure")
print("=" * 65)

def build_fake_gguf_header(cfg: ModelConfig) -> bytes:
    """
    Build a minimal synthetic GGUF header for educational inspection.
    Uses the GGUF v3 binary format.
    """
    import io
    buf = io.BytesIO()

    # Magic + version
    buf.write(b"GGUF")
    buf.write(struct.pack("<I", 3))       # version 3

    # n_tensors and n_kv (placeholders — we'll back-fill)
    n_tensors = 5   # we'll write 5 toy tensors
    n_kv      = 6
    buf.write(struct.pack("<Q", n_tensors))
    buf.write(struct.pack("<Q", n_kv))

    # KV metadata
    def write_str(b: io.BytesIO, s: str):
        encoded = s.encode()
        b.write(struct.pack("<Q", len(encoded)))
        b.write(encoded)

    def write_kv_uint32(b: io.BytesIO, key: str, val: int):
        write_str(b, key)
        b.write(struct.pack("<I", 4))   # type UINT32 = 4
        b.write(struct.pack("<I", val))

    def write_kv_str(b: io.BytesIO, key: str, val: str):
        write_str(b, key)
        b.write(struct.pack("<I", 8))   # type STRING = 8
        write_str(b, val)

    write_kv_uint32(buf, "llama.context_length",         cfg.max_model_len)
    write_kv_uint32(buf, "llama.embedding_length",       cfg.d_model)
    write_kv_uint32(buf, "llama.block_count",            cfg.n_layers)
    write_kv_uint32(buf, "llama.attention.head_count",   cfg.n_heads)
    write_kv_uint32(buf, "llama.attention.head_count_kv",cfg.n_kv_heads)
    write_kv_str   (buf, "general.architecture",         "llama")

    return buf.getvalue()

header = build_fake_gguf_header(LLAMA3_8B)
print(f"\n  Synthetic GGUF header for {LLAMA3_8B.name}:")
print(f"  Magic bytes: {header[:4]}")
print(f"  Version:     {struct.unpack('<I', header[4:8])[0]}")
print(f"  n_tensors:   {struct.unpack('<Q', header[8:16])[0]}")
print(f"  n_kv:        {struct.unpack('<Q', header[16:24])[0]}")
print(f"  Header size: {len(header)} bytes")

# Simulate parsing cost
def simulate_gguf_load(n_tensors: int, file_gb: float) -> Dict[str, float]:
    """Simulate timing of GGUF load phases."""
    return {
        "header_parse_s":  0.001 * n_tensors / 291 * 0.1,   # scales with n_tensors
        "mmap_s":          0.3 + file_gb * 0.05,             # mmap overhead
        "gpu_upload_s":    file_gb * 0.5 / 32,               # GB / PCIe bandwidth
    }

print(f"\n  Simulated GGUF load timings:")
for name, n_t, file_gb in [
    ("LLaMA 3 8B  Q4_K_M", 291,  4.7),
    ("LLaMA 3 70B Q4_K_M", 720, 39.0),
    ("LLaMA 3 8B  BF16",   291, 16.1),
]:
    t = simulate_gguf_load(n_t, file_gb)
    total = sum(t.values())
    print(f"  {name:<25} header={t['header_parse_s']:.2f}s  "
          f"mmap={t['mmap_s']:.2f}s  "
          f"upload={t['gpu_upload_s']:.2f}s  "
          f"TOTAL={total:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Startup timeline comparison
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SECTION 5: Startup Timeline Comparison")
print("=" * 65)

def vllm_startup_time(cfg: ModelConfig, n_gpus: int = 1,
                      n_graph_sizes: int = 15) -> Dict[str, float]:
    """Estimate vLLM startup time in seconds."""
    weight_gb    = weight_bytes(cfg) / 1e9
    load_bw_gbs  = 32.0 * n_gpus   # PCIe 4.0 bandwidth
    return {
        "config":        1.0,
        "weight_load":   weight_gb / n_gpus / load_bw_gbs,
        "dummy_pass":    2.0 + cfg.n_layers * 0.05,
        "block_pool":    0.5,
        "graph_capture": n_graph_sizes * 2.0,
    }

def llama_startup_time(cfg: ModelConfig, quant_bpw: float = 4.5,
                       n_gpu_layers: int = None) -> Dict[str, float]:
    """Estimate llama.cpp startup time in seconds."""
    n_layers_gpu = n_gpu_layers if n_gpu_layers is not None else cfg.n_layers
    gpu_weight_gb = weight_bytes(cfg, quant_bpw / 8) * n_layers_gpu / cfg.n_layers / 1e9
    return {
        "header_parse":  0.1,
        "mmap":          0.3,
        "gpu_upload":    gpu_weight_gb / 32.0,   # PCIe bandwidth
    }

print(f"\n  {'Engine + Model':<32} {'Phases':<50} {'Total':>8}")
print("  " + "-" * 92)

for label, times in [
    ("vLLM 8B  (1×A100)",   vllm_startup_time(LLAMA3_8B,  1, 15)),
    ("vLLM 70B (8×A100)",   vllm_startup_time(LLAMA3_70B, 8, 15)),
    ("llama.cpp 8B  Q4_K_M",llama_startup_time(LLAMA3_8B,  4.5, 32)),
    ("llama.cpp 70B Q4_K_M",llama_startup_time(LLAMA3_70B, 4.5, 80)),
]:
    total   = sum(times.values())
    phases  = "  ".join(f"{k}={v:.1f}s" for k, v in times.items())
    print(f"  {label:<32} {phases:<50} {total:>7.1f}s")

print("\nDone.")
```

---

## 8.12 Chapter Summary

| Concept | Key fact |
|---------|----------|
| vLLM startup phases | Config → Weight load → Dummy pass → Block pool → CUDA graphs |
| Dummy forward pass | Runs at max load; measures peak activation memory via `mem_get_info()` |
| KV block count formula | `(GPU × util − weights − activations) / block_bytes` |
| CUDA graphs | Capture 15 batch sizes; replay bypasses Python dispatch |
| Graph padding | Actual batch padded to nearest captured size; waste < (gap-1)/captured |
| GGUF format | Magic + version + n_kv + n_tensors + KV metadata + tensor headers + data |
| mmap loading | Zero-copy virtual mapping; pages faulted in on demand |
| `--n-gpu-layers` | How many transformer layers to upload from mmap to GPU VRAM |
| vLLM vs llama.cpp startup | ~30–130 s vs ~1–10 s; throughput trade-off |
| `--mlock` | Forces all mmap pages resident; eliminates page-fault latency spikes |

### Why this matters for what follows

- **Chapter 9** (The Forward Pass) describes what happens inside the CUDA graph
  captured in Phase 5 — the actual kernel sequence that the graph replays.
- **Chapter 10** (Quantization) explains the Q4_K_M, Q5_K_S, Q8_0 types that
  appear in GGUF tensor headers and how they affect both weight size (Phase 2)
  and inference accuracy.
- **Chapter 13** (Disaggregated Prefill) requires shipping KV blocks from a
  prefill GPU to a decode GPU; the block pool layout from Phase 4 determines
  the granularity of those transfers.

---

## 8.13 Further Reading

- vLLM source: `vllm/worker/worker.py` (dummy forward, graph capture),
  `vllm/model_executor/model_loader/weight_utils.py` (weight loading).
- GGUF specification: `https://github.com/ggerganov/ggml/blob/master/docs/gguf.md`
- llama.cpp source: `src/llama.cpp`, function `llama_model_load_internal()`.
- CUDA Graphs documentation:
  `https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs`

---

*End of Chapter 8.*


---

## Chapter Summary

- **`vllm serve` startup sequence**: CLI parsing → EngineArgs validation → model download or cache hit → weight sharding → CUDA graph capture → scheduler and block manager initialization → HTTP server binding.
- **Weight loading**: vLLM reads safetensors/pytorch_bin shards and immediately shards them across tensor-parallel ranks; no full model ever resides on a single GPU if TP > 1.
- **CUDA graph capture**: vLLM replays decode steps as CUDA graphs rather than re-issuing kernels each time; this requires capturing at all batch sizes up to `--max-num-seqs`.
- **Block pool sizing**: the number of physical KV blocks is determined by subtracting weight memory and activation buffer from available GPU VRAM, divided by bytes-per-block.
- **Warm-up requests**: vLLM issues synthetic prefill and decode requests of the maximum configured lengths during startup to allocate peak activation memory before traffic arrives.
- **llama.cpp startup**: `llama_load_model_from_file` mmap's the GGUF file; quantised weights are read on demand, not pre-loaded; CUDA context is initialized lazily on first GPU layer allocation.

---

## Self-Check Questions

1. An H100 has 80 GB HBM. A LLaMA-3 8B model in FP16 occupies 16 GB. With `--gpu-memory-utilization 0.90`, activation buffers reserved at 1 GB, and block size 16 at 32 KV heads and d_k = 128 in BF16, compute the number of physical KV blocks available. *(Section 8.3)*

2. CUDA graph capture requires running a forward pass at every batch size from 1 to `max-num-seqs`. For `max-num-seqs = 256`, estimate why startup time is O(N) in the number of distinct graph sizes rather than O(N²). *(Section 8.4)*

3. vLLM uses `--tensor-parallel-size 4` on a single 4-GPU node. Describe which weights are column-split, which are row-split, and how the weight loading phase ensures each rank gets the correct shard. *(Section 8.2)*

4. llama.cpp starts serving a first token within 300 ms on a laptop while vLLM takes 30 s on a server GPU. Name three architectural reasons for this startup time difference. *(Section 8.5)*

5. What is the purpose of the warm-up pass during vLLM startup? What memory allocation failure would occur without it? *(Section 8.3)*
