# Appendix C — PyTorch for LLM Inference: From Tensors to Production

> *"PyTorch is not just a training framework. Every vLLM, SGLang, and llama.cpp
> Python backend ultimately speaks PyTorch at its lowest layer."*

---

## C.1 PyTorch's Role in the Inference Stack

PyTorch is the dominant deep learning framework for LLM inference — not because
it was designed for inference, but because it was designed for flexibility. The
same properties that make PyTorch excellent for research (eager execution,
dynamic shapes, Python-native extensibility) also make it possible to build the
complex, adaptive inference engines this book describes.

In a typical LLM serving stack, PyTorch appears at multiple layers:

```
┌─────────────────────────────────────────────────────┐
│  Serving layer    (vLLM, SGLang, TGI, LiteLLM)      │
├─────────────────────────────────────────────────────┤
│  Scheduler        (Python: batch management)         │
├─────────────────────────────────────────────────────┤
│  PyTorch          (eager/compiled model forward)     │
├─────────────────────────────────────────────────────┤
│  ATen / CUDA kernels  (C++/CUDA, FlashAttention)    │
├─────────────────────────────────────────────────────┤
│  cuBLAS / cuDNN   (vendor libraries)                 │
└─────────────────────────────────────────────────────┘
```

Understanding how PyTorch works at each layer gives you the mental model
needed to diagnose performance bottlenecks, write custom kernels, and
optimise memory usage.

---

## C.2 Tensor Fundamentals for Inference Engineers

### C.2.1 The dtype landscape

Every tensor has a **dtype** that determines its precision and memory cost.
For LLM inference the relevant dtypes are:

| dtype | Bits | Range | Notes |
|---|---|---|---|
| `torch.float32` (FP32) | 32 | ±3.4×10³⁸ | Training default; rarely used for inference |
| `torch.bfloat16` (BF16) | 16 | ±3.4×10³⁸ | 8-bit exp; preferred for inference on Ampere+ |
| `torch.float16` (FP16) | 16 | ±6.5×10⁴ | 5-bit exp; overflow risk for long generation |
| `torch.float8_e4m3fn` (FP8) | 8 | ±448 | H100 native; W8A8 quantization |
| `torch.float8_e5m2` (FP8) | 8 | ±57,344 | Wider range variant; used for gradients |
| `torch.int8` (INT8) | 8 | −128…127 | Weight quantization; needs dequantize |
| `torch.int4` (INT4) | 4 | −8…7 | Via bitsandbytes / torchao packing |
| `torch.int32` (INT32) | 32 | ±2.1×10⁹ | Token IDs, indices |
| `torch.bool` | 8 | 0/1 | Attention masks |

```python
import torch

# Dtype inspection
x = torch.randn(4, 4)
print(x.dtype)            # torch.float32
print(x.element_size())   # 4 bytes

# Conversion
x_bf16 = x.to(torch.bfloat16)
print(x_bf16.dtype)       # torch.bfloat16
print(x_bf16.element_size())  # 2 bytes

# Memory cost
def tensor_memory_mb(shape, dtype):
    t = torch.empty(shape, dtype=dtype)
    return t.element_size() * t.numel() / 1e6

# KV cache block: (2, n_heads, block_size, head_dim) for K and V
print(tensor_memory_mb((2, 8, 16, 128), torch.float16))   # 0.032 MB per block
print(tensor_memory_mb((2, 8, 16, 128), torch.bfloat16))  # 0.032 MB (same size)
```

### C.2.2 BF16 vs FP16 for inference

BF16 is preferred for LLM inference because its 8-bit exponent matches FP32,
preventing the overflow and underflow that FP16's 5-bit exponent causes in
very large or very small activations:

```python
import torch

large_val = torch.tensor(70_000.0)
print(large_val.to(torch.float16))   # tensor(inf) — OVERFLOW!
print(large_val.to(torch.bfloat16))  # tensor(70016.) — OK (slight rounding)

small_val = torch.tensor(1e-7)
print(small_val.to(torch.float16))   # tensor(9.9999e-08) — OK here
print(small_val.to(torch.bfloat16))  # tensor(1.0014e-07) — OK
```

Use FP16 only when hardware lacks BF16 support (V100, older GPUs). On Ampere,
Ada, and Hopper, always prefer BF16.

### C.2.3 Strides, contiguity, and memory layout

A PyTorch tensor is a view over a 1-D storage buffer. **Strides** describe
how many elements to skip along each dimension:

```python
x = torch.randn(4, 8)       # 4 rows, 8 columns — row-major
print(x.stride())            # (8, 1): next row = +8 elements, next col = +1
print(x.is_contiguous())     # True — elements are sequential in memory

# Transposing creates a non-contiguous view — no data copy!
xt = x.t()
print(xt.shape)              # (8, 4)
print(xt.stride())           # (1, 8) — column-major (non-contiguous)
print(xt.is_contiguous())    # False

# Making it contiguous (copies data into new layout)
xt_c = xt.contiguous()
print(xt_c.is_contiguous())  # True
print(xt_c.data_ptr() != xt.data_ptr())  # True — new allocation
```

**Why this matters for inference**: many CUDA kernels require contiguous
tensors. Passing a non-contiguous tensor to a GEMM kernel (e.g., inside
FlashAttention) triggers an implicit `.contiguous()` copy that wastes memory
and time. Always ensure KV cache tensors are stored in the correct layout.

### C.2.4 Views vs copies — the golden rule

```python
# Views: same storage, no copy
a = torch.randn(16)
b = a.view(4, 4)       # reshape — view if contiguous
c = a[::2]             # stride-2 slice — view
d = a.unsqueeze(0)     # add dim — view

# Copies: new storage
e = a.clone()           # explicit copy
f = a.contiguous()      # copy only if non-contiguous (else returns self)
g = torch.cat([a, a])  # always new allocation

# Check if two tensors share storage
print(a.storage().data_ptr() == b.storage().data_ptr())  # True (view)
print(a.storage().data_ptr() == e.storage().data_ptr())  # False (copy)
```

Minimising copies in the decode loop is critical. vLLM's paged attention
avoids copies by using views into pre-allocated KV cache blocks.

---

## C.3 Device Management

### C.3.1 CUDA device selection and properties

```python
import torch

# Device enumeration
n_gpus = torch.cuda.device_count()
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}, "
          f"{props.total_memory // 1024**3} GB, "
          f"SM {props.major}.{props.minor}, "
          f"{props.multi_processor_count} SMs")

# Set the active device
torch.cuda.set_device(0)    # or: device = torch.device("cuda:0")

# Move tensors between devices
x = torch.randn(1024, 1024)         # CPU
x_gpu = x.to("cuda:0")              # GPU 0
x_gpu1 = x_gpu.to("cuda:1")        # GPU 1 — cross-device copy via PCIe/NVLink

# Create directly on device
w = torch.empty(4096, 4096, dtype=torch.bfloat16, device="cuda:0")
torch.nn.init.normal_(w, mean=0.0, std=0.02)
```

### C.3.2 CUDA streams

CUDA operations within a stream execute in order. Operations in different
streams can execute concurrently if hardware resources allow.

```python
# Default stream (stream 0): all operations serialised
x = torch.randn(1024, 1024, device="cuda")
y = x.mm(x.t())   # blocking

# Custom streams: overlap memory transfers with compute
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

with torch.cuda.stream(s1):
    a = torch.randn(1024, 1024, device="cuda")  # compute in stream 1

with torch.cuda.stream(s2):
    b = torch.randn(1024, 1024, device="cuda")  # concurrent in stream 2

# Synchronise before combining results
torch.cuda.synchronize()
c = a + b

# Prefetch next batch while processing current (vLLM pattern)
prefetch_stream = torch.cuda.Stream()
compute_stream  = torch.cuda.current_stream()

with torch.cuda.stream(prefetch_stream):
    next_tokens = next_batch.to("cuda", non_blocking=True)  # async H2D

with torch.cuda.stream(compute_stream):
    output = model(current_tokens)  # compute on current batch

# Wait for prefetch before next step
compute_stream.wait_stream(prefetch_stream)
```

### C.3.3 Pinned (page-locked) memory for fast host→device transfers

```python
# Regular host tensor: pageable — slow H2D copy
cpu_tensor = torch.randn(1024, 1024)

# Pinned host tensor: page-locked — 2–3× faster H2D copy
pinned_tensor = torch.randn(1024, 1024).pin_memory()

# Async (non-blocking) transfer to GPU
gpu_tensor = pinned_tensor.to("cuda", non_blocking=True)
# Returns immediately; use CUDA events to check completion

# Check if pinned
print(pinned_tensor.is_pinned())  # True
```

In production, the tokenised input (token ID tensors) should be transferred
as pinned memory to avoid blocking the decode loop during H2D copies.

### C.3.4 Multi-GPU with device guards

```python
# DeviceGuard: temporarily switch active device
for gpu_id in range(4):
    with torch.cuda.device(gpu_id):
        shard = compute_shard(gpu_id)   # runs on gpu_id

# Peer access: tensor on GPU 0 can be directly read by GPU 1 over NVLink
torch.cuda.can_device_access_peer(0, 1)  # True on NVLink systems
torch.cuda.enable_peer_access(1)         # enable from current device (0) to device 1
```

---

## C.4 Inference Mode and Memory Efficiency

### C.4.1 `torch.no_grad()` vs `torch.inference_mode()`

Both disable gradient tracking. `inference_mode` is stricter and faster:

```python
import torch

x = torch.randn(1024, requires_grad=True)

# no_grad: disables gradient computation but tensors can still be viewed by autograd
with torch.no_grad():
    y = x * 2
    print(y.requires_grad)   # False — but autograd graph not fully disabled

# inference_mode: FASTER — prevents any autograd interaction
# Tensors created inside cannot be used in autograd computations after leaving the block
with torch.inference_mode():
    y = x * 2
    print(y.requires_grad)   # False
    print(y.is_inference())  # True — stronger guarantee

# For production inference, ALWAYS use inference_mode
@torch.inference_mode()
def generate(model, tokens):
    return model(tokens)
```

`inference_mode` avoids recording operations in the autograd graph entirely,
saving ~10–15% memory and 3–5% compute vs `no_grad` for large models.

### C.4.2 Autocast: mixed-precision inference

```python
import torch
from torch.amp import autocast

model = MyLLM().cuda().to(torch.float32)  # weights in FP32

# Run forward pass in BF16 — weights and activations cast automatically
with autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(input_ids)

# The recommended pattern for inference (combines both):
@torch.inference_mode()
def forward_bf16(model, input_ids):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(input_ids)
```

### C.4.3 Memory caching allocator

PyTorch's CUDA memory allocator caches freed GPU memory rather than returning
it to the OS. This prevents costly `cudaMalloc`/`cudaFree` calls:

```python
# Current memory state
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.memory_reserved()  / 1e9, "GB reserved (cached)")

# Free memory: clear cache between unrelated workloads
torch.cuda.empty_cache()   # returns unused cached memory to driver

# Memory snapshot for debugging OOM
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run workload ...
snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._dump_snapshot("memory_snapshot.pkl")
# Analyse with: python -m torch.cuda._memory_viz trace_plot memory_snapshot.pkl -o plot.html

# Find memory leaks: tensors holding GPU memory unexpectedly
def report_gpu_tensors(threshold_mb=100):
    import gc
    gc.collect()
    torch.cuda.synchronize()
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            mb = obj.element_size() * obj.nelement() / 1e6
            if mb >= threshold_mb:
                print(f"{mb:.1f} MB: {obj.dtype} {list(obj.shape)}")
```

### C.4.4 Avoiding common memory mistakes

```python
# MISTAKE 1: accumulating tensors in a list without .detach()
outputs = []
for step in range(1000):
    out = model(tokens)
    outputs.append(out)           # keeps entire autograd graph alive!

# FIX: detach or use inference_mode
outputs = []
with torch.inference_mode():
    for step in range(1000):
        out = model(tokens)
        outputs.append(out.cpu())  # move to CPU immediately if not needed on GPU

# MISTAKE 2: .item() inside a loop — forces GPU sync every step
losses = [loss.item() for loss in loss_list]  # 1000 CPU-GPU syncs

# FIX: batch the item() call
gpu_losses = torch.stack(loss_list)
cpu_losses = gpu_losses.cpu().tolist()         # single sync

# MISTAKE 3: not deleting intermediate tensors
def bad_forward(x, W1, W2):
    h = x @ W1   # 1 GB
    o = h @ W2   # 1 GB  — both alive at peak!
    return o

def good_forward(x, W1, W2):
    h = x @ W1
    o = h @ W2
    del h        # release h immediately
    return o
```

---

## C.5 `torch.compile`: JIT Compilation Layer

`torch.compile` (introduced in PyTorch 2.0) uses **TorchDynamo** to trace
Python bytecode and **TorchInductor** to generate optimised CUDA/Triton kernels.

### C.5.1 Basic usage

```python
import torch

# Compile a model once — applies graph captures and kernel fusion
model = MyLLM().cuda().bfloat16()
compiled_model = torch.compile(model, mode="reduce-overhead")

# First call: compilation (~30–120s for large models)
# Subsequent calls: compiled kernel execution
output = compiled_model(input_ids)
```

### C.5.2 Compilation modes

| Mode | Compilation time | Speedup | Use case |
|---|---|---|---|
| `"default"` | Moderate | Moderate | General-purpose |
| `"reduce-overhead"` | Fast | Good | Low-latency inference, variable shapes |
| `"max-autotune"` | Slow (minutes) | Best | Fixed shapes, throughput-maximised serving |
| `"max-autotune-no-cudagraphs"` | Slow | Near-best | When CUDA graphs conflict with dynamic ops |

```python
# For LLM inference with variable batch sizes and sequence lengths:
compiled = torch.compile(model, mode="reduce-overhead", dynamic=True)
# dynamic=True: uses symbolic shapes to avoid recompilation per new shape

# For fixed-shape offline batch processing:
compiled = torch.compile(model, mode="max-autotune", dynamic=False)
```

### C.5.3 CUDA Graphs integration

CUDA Graphs (Chapter 8.5) capture the entire GPU command stream and replay it
without CPU overhead. `torch.compile` can integrate with CUDA Graphs:

```python
# torch.compile with CUDA graphs (mode="reduce-overhead" enables this)
model = torch.compile(model, mode="reduce-overhead")

# Manual CUDA Graph capture for fixed shapes
g = torch.cuda.CUDAGraph()
static_input = torch.randn(1, 512, device="cuda")   # fixed shape

# Warm-up (build the graph)
for _ in range(3):
    with torch.cuda.graph(g):
        static_output = model(static_input)

# Replay without Python overhead
new_input = static_input   # must be SAME storage (in-place update)
new_input.copy_(actual_input)
g.replay()
result = static_output
```

### C.5.4 Inspecting compilation

```python
# View what TorchDynamo captured
import torch._dynamo
torch._dynamo.explain(model)(input_ids)   # shows captured graph

# Disable compilation for debugging
torch._dynamo.disable()   # falls back to eager

# Log compilation events
import logging
logging.getLogger("torch._inductor").setLevel(logging.DEBUG)

# Count recompilations (should be ~0 after warm-up)
torch._dynamo.reset()   # clear compilation cache for fresh measurement
```

### C.5.5 Common failures and fixes

| Failure | Cause | Fix |
|---|---|---|
| `TorchDynamoError: Dynamic shapes` | Shape changes between calls | Use `dynamic=True` |
| `Recompilation triggered` | Python control flow inside model | Use `torch.cond` or refactor |
| `Graph break at` | Unsupported Python construct | Refactor to avoid (e.g., no `print` inside model) |
| `Segfault in compiled kernel` | CUDA version mismatch | Match PyTorch CUDA version to driver |
| Slow first call | Compilation time | Pre-compile with dummy inputs at startup |

---

## C.6 Quantization APIs in PyTorch

### C.6.1 torchao: the modern quantization toolkit

`torchao` (torch-ao) is the current recommended quantization API:

```python
# pip install torchao
from torchao.quantization import (
    quantize_,
    Int8WeightOnlyConfig,
    Int4WeightOnlyConfig,
    Float8WeightOnlyConfig,
    Float8DynamicActivationFloat8WeightConfig,
)
import torch

model = MyLLM().cuda().bfloat16()

# INT8 weight-only (good for CPU/GPU)
quantize_(model, Int8WeightOnlyConfig())

# INT4 weight-only (maximum compression)
quantize_(model, Int4WeightOnlyConfig(group_size=128))

# FP8 W8A8 dynamic (H100 recommended)
quantize_(model, Float8DynamicActivationFloat8WeightConfig())

# Use the quantized model normally — no API change
with torch.inference_mode():
    out = model(input_ids)
```

### C.6.2 bitsandbytes integration

`bitsandbytes` provides 4-bit (NF4) and 8-bit quantization via a patched
`Linear` layer:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit NF4 with double quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,    # quantize the scale factors too
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# Inspect quantized layers
for name, module in model.named_modules():
    if "bnb" in type(module).__name__.lower():
        print(f"Quantized: {name} — {type(module).__name__}")
```

### C.6.3 Per-layer dtype inspection and manipulation

```python
def model_dtype_report(model: torch.nn.Module) -> dict:
    """Show parameter dtype distribution and total memory by dtype."""
    from collections import defaultdict
    stats = defaultdict(lambda: {"params": 0, "bytes": 0})
    for name, param in model.named_parameters():
        dt = str(param.dtype)
        stats[dt]["params"] += param.numel()
        stats[dt]["bytes"]  += param.numel() * param.element_size()
    for dt, s in stats.items():
        print(f"{dt:30s}: {s['params']:>12,} params  "
              f"{s['bytes']/1e9:>7.2f} GB")
    return dict(stats)

# Example output for a BF16 + INT4 mixed model:
# torch.bfloat16:  4,194,304 params     0.01 GB  (embeddings kept BF16)
# torch.int8:   69,000,000,000 params   34.50 GB  (linear weights quantized)
```

---

## C.7 Distributed Inference with `torch.distributed`

### C.7.1 Process group initialisation

```python
import torch
import torch.distributed as dist
import os

def init_distributed():
    """Initialise the process group for tensor-parallel inference."""
    dist.init_process_group(
        backend="nccl",         # NCCL for GPU-to-GPU communication
        init_method="env://",   # reads MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    )
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

# Launch: torchrun --nproc_per_node=4 inference_server.py
```

### C.7.2 Tensor parallelism: column and row splits

Tensor parallelism (Chapter 15) splits weight matrices across GPUs. The
PyTorch primitives that implement this are `all_reduce` and `all_gather`:

```python
import torch.distributed as dist

class ColumnParallelLinear(torch.nn.Module):
    """Split output columns across world_size GPUs."""
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert out_features % world_size == 0
        local_out = out_features // world_size
        self.weight = torch.nn.Parameter(
            torch.empty(local_out, in_features, device=f"cuda:{rank}",
                        dtype=torch.bfloat16)
        )
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)  # local shard output

class RowParallelLinear(torch.nn.Module):
    """Split input rows across world_size GPUs; all_reduce at the end."""
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert in_features % world_size == 0
        local_in = in_features // world_size
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, local_in, device=f"cuda:{rank}",
                        dtype=torch.bfloat16)
        )
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        local_out = torch.nn.functional.linear(x, self.weight)
        dist.all_reduce(local_out, op=dist.ReduceOp.SUM)  # gather partial sums
        return local_out
```

### C.7.3 Communication primitives for inference

```python
# All-reduce: sum tensor shards from all ranks → all ranks get the sum
# Used in: RowParallelLinear (output aggregation)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# All-gather: collect shards from all ranks → each rank gets the full tensor
# Used in: Sequence parallelism (gathering KV cache)
gathered = [torch.empty_like(shard) for _ in range(world_size)]
dist.all_gather(gathered, shard)
full_tensor = torch.cat(gathered, dim=-1)

# Broadcast: send tensor from rank 0 to all ranks
# Used in: distributing new input tokens
dist.broadcast(tensor, src=0)

# Measuring communication overhead
import time
dist.barrier()
t0 = time.perf_counter()
dist.all_reduce(torch.ones(1024, 1024, device="cuda"))
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - t0) * 1000
print(f"All-reduce 4MB: {elapsed_ms:.2f}ms")
```

---

## C.8 Custom Operations and Extensions

### C.8.1 Registering a custom operator with `torch.library`

```python
import torch
from torch import Tensor

# Define a custom operator (Python side)
# The "mylib::flash_attn" namespace separates from built-in ops
@torch.library.custom_op("mylib::scaled_dot_product", mutates_args=())
def scaled_dot_product(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Pure-Python reference implementation."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs  = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)

# Register a CUDA-optimised implementation
@scaled_dot_product.register_kernel("cuda")
def scaled_dot_product_cuda(q, k, v, scale):
    # In production: call into FlashAttention or a custom CUDA kernel
    # Here: fallback to torch (would be replaced with actual CUDA impl)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

# Use it like any built-in op
q = torch.randn(2, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
out = torch.ops.mylib.scaled_dot_product(q, k, v, scale=128**-0.5)
```

### C.8.2 Calling CUDA extensions from Python

```python
# Build a CUDA extension with torch.utils.cpp_extension
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel (for rapid prototyping)
cuda_src = """
__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int hidden_size, float eps
) {
    int batch = blockIdx.x;
    const float* row = x + batch * hidden_size;
    float* out_row   = out + batch * hidden_size;

    // Compute variance
    float variance = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        variance += row[i] * row[i];
    }
    // Block reduce (simplified — use warp shuffle for production)
    __shared__ float shared_var;
    if (threadIdx.x == 0) shared_var = 0.0f;
    __syncthreads();
    atomicAdd(&shared_var, variance);
    __syncthreads();
    float scale = rsqrtf(shared_var / hidden_size + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = row[i] * scale * weight[i];
    }
}

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, float eps) {
    auto out = torch::empty_like(x);
    int batch = x.size(0);
    int hidden = x.size(1);
    rms_norm_kernel<<<batch, 256>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(),
        out.data_ptr<float>(), hidden, eps
    );
    return out;
}
"""

cpp_src = 'torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, float eps);'

rms_norm_ext = load_inline(
    name="rms_norm",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["rms_norm"],
    verbose=False,
)

x = torch.randn(4, 4096, device="cuda")
w = torch.ones(4096, device="cuda")
out = rms_norm_ext.rms_norm(x, w, 1e-5)
print(out.shape)   # (4, 4096)
```

---

## C.9 Profiling and Debugging

### C.9.1 `torch.profiler` — the primary tool

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = MyLLM().cuda().bfloat16()
tokens = torch.randint(0, 50000, (1, 512), device="cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("model_forward"):
        with torch.inference_mode():
            output = model(tokens)

# Print top kernels by CUDA time
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=20
))

# Export to Chrome trace
prof.export_chrome_trace("llm_trace.json")
# Open in chrome://tracing or Perfetto UI
```

### C.9.2 CUDA events for precise timing

```python
import torch

def cuda_timed(fn, *args, warmup=5, repeats=20, **kwargs):
    """Time a CUDA function with proper warm-up and event synchronisation."""
    # Warm up (avoids JIT and cache effects in the measurement)
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(repeats):
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    import statistics
    return {
        "mean_ms":   statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms":    sorted(times)[int(0.95 * len(times))],
        "min_ms":    min(times),
        "max_ms":    max(times),
    }

# Usage
q = torch.randn(1, 32, 1024, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 32, 1024, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 32, 1024, 128, device="cuda", dtype=torch.bfloat16)

times = cuda_timed(
    torch.nn.functional.scaled_dot_product_attention,
    q, k, v
)
print(f"SDPA: {times['median_ms']:.3f}ms median, {times['p95_ms']:.3f}ms P95")
```

### C.9.3 Memory debugging

```python
# Track memory high-water mark
torch.cuda.reset_peak_memory_stats()
output = model(tokens)
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f"Peak memory: {peak_mb:.1f} MB")

# Detailed memory breakdown
print(torch.cuda.memory_summary(abbreviated=True))

# Find the source of large allocations
torch.cuda.memory._record_memory_history(stacks="all")
# ... run workload ...
snap = torch.cuda.memory._snapshot()
# Visualise: python -m torch.cuda._memory_viz trace_plot snap.pkl -o mem.html
```

---

## C.10 `torch.export` and Deployment Targets

### C.10.1 ExportedProgram

`torch.export` produces a fully-captured, serialisable computation graph —
the modern successor to TorchScript:

```python
import torch
from torch.export import export, Dim

model = MyLLM().cuda().bfloat16()
model.eval()

# Define dynamic dimensions (batch and sequence length can vary)
batch  = Dim("batch",  min=1, max=64)
seq    = Dim("seq",    min=1, max=4096)

# Export with sample inputs
sample_input = torch.randint(0, 50000, (1, 128), device="cuda")
exported = export(
    model,
    (sample_input,),
    dynamic_shapes={"x": {0: batch, 1: seq}},
)

# Inspect the exported graph
print(exported.graph)

# Save and load
torch.export.save(exported, "llm_exported.pt2")
loaded = torch.export.load("llm_exported.pt2")
result = loaded.module()(sample_input)
```

### C.10.2 ONNX export

```python
import torch
import torch.onnx

model = MyLLM().cuda().bfloat16()
sample = torch.randint(0, 50000, (1, 128), device="cuda")

torch.onnx.export(
    model,
    (sample,),
    "llm.onnx",
    opset_version=20,                      # use latest for FP8/BF16 support
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "logits":    {0: "batch", 1: "seq"},
    },
    do_constant_folding=True,
)

# Validate with onnxruntime
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("llm.onnx", providers=["CUDAExecutionProvider"])
inputs = {"input_ids": sample.cpu().numpy()}
out = sess.run(None, inputs)
print(out[0].shape)
```

### C.10.3 TorchScript (legacy but still relevant for libtorch)

```python
# scripting: works for simple models
scripted = torch.jit.script(model)
scripted.save("llm_scripted.pt")

# tracing: captures one execution path — use for fixed-shape models
traced = torch.jit.trace(model, (sample,))
traced.save("llm_traced.pt")

# Reload in Python or C++ (see Appendix J)
loaded = torch.jit.load("llm_scripted.pt")
```

---

## C.11 PyTorch Internals: ATen and the Dispatcher

Understanding the ATen layer helps diagnose performance issues and write
kernel extensions correctly.

### C.11.1 ATen: the C++ tensor library

ATen (A Tensor Library) is the C++ backbone of PyTorch. Every Python
tensor operation (e.g., `torch.mm()`) dispatches to an ATen implementation:

```
Python:   torch.mm(x, y)
          ↓
C++:      at::mm(x, y)         (ATen dispatcher)
          ↓
CUDA:     at::native::mm_cuda() → cublasSgemm / cublasGemmEx
```

You can call ATen directly from Python for profiling:
```python
import torch._C._VariableFunctions as _VF

# Call ATen's mm directly (bypasses Python overhead)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
out = torch.mm(x, y)
```

### C.11.2 Operator dispatch keys

Every tensor has a **dispatch key** (e.g., `CUDA`, `CPU`, `AutogradCUDA`)
that controls which backend handles it. This is how PyTorch routes operations:

```python
# Inspect dispatch key
x = torch.randn(4)
print(torch._C._dispatch_key_set(x))  # DispatchKeySet(CPU, AutogradCPU)

x_cuda = x.cuda()
print(torch._C._dispatch_key_set(x_cuda))  # DispatchKeySet(CUDA, AutogradCUDA)

# Inspect all registered kernels for an op
print(torch._C._dispatch_dump("aten::mm"))
```

---

## C.12 Key PyTorch Patterns in vLLM

The vLLM source code uses several PyTorch patterns repeatedly. Understanding
them helps when extending or debugging vLLM.

```python
# Pattern 1: Fused operations with torch.ops.vllm (registered custom ops)
# vLLM registers its PagedAttention, RoPE, and RMSNorm as custom ops
from vllm._custom_ops import paged_attention_v1

# Pattern 2: In-place KV cache updates (no allocation per step)
# key_cache shape: [num_blocks, num_heads, head_size, block_size]
key_cache = torch.zeros(1024, 32, 128, 16, dtype=torch.float16, device="cuda")
# Update block 5, positions 0-15:
key_cache[5].copy_(new_keys)  # in-place, no allocation

# Pattern 3: Gather for paged attention (reading scattered cache blocks)
block_tables = torch.tensor([[5, 3, 7, 2]], device="cuda")  # request 0: blocks
# Index into cache using block_tables — vLLM's paged_attention kernel does this

# Pattern 4: Sampling with top-p / top-k (post-forward pass)
logits = torch.randn(1, 128_000, device="cuda")  # vocab logits
probs  = torch.softmax(logits / temperature, dim=-1)
top_k_probs, top_k_ids = torch.topk(probs, k=50)
# Sample from top-k
next_token = torch.multinomial(top_k_probs, num_samples=1)
token_id   = top_k_ids.gather(-1, next_token)
```

---

## C.13 Worked Example C.1 — Profiling and Optimising a Transformer Block

**Goal**: profile a single transformer decoder layer and identify bottlenecks.

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Minimal transformer block for profiling
class TransformerBlock(nn.Module):
    def __init__(self, d_model=4096, n_heads=32, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2  = nn.RMSNorm(d_model)
        self.gate   = nn.Linear(d_model, d_model * ffn_mult, bias=False)
        self.up     = nn.Linear(d_model, d_model * ffn_mult, bias=False)
        self.down   = nn.Linear(d_model * ffn_mult, d_model, bias=False)

    def forward(self, x):
        with record_function("attention"):
            h = self.norm1(x)
            q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
            B, T, C = q.shape
            n, d = 32, C // 32
            q = q.view(B, T, n, d).transpose(1, 2)
            k = k.view(B, T, n, d).transpose(1, 2)
            v = v.view(B, T, n, d).transpose(1, 2)
            attn = nn.functional.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.o_proj(attn)

        with record_function("ffn"):
            h = self.norm2(x)
            x = x + self.down(nn.functional.silu(self.gate(h)) * self.up(h))
        return x

model = TransformerBlock(4096, 32).cuda().bfloat16()
model = torch.compile(model, mode="reduce-overhead")

x = torch.randn(1, 2048, 4096, device="cuda", dtype=torch.bfloat16)

# Profile
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with torch.inference_mode():
        for _ in range(5):
            _ = model(x)

table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
print(table)
```

**Expected profiler output** (approximate):
```
Name                     CUDA time   % of total
-----------------------  ----------  ----------
attention                8.2ms       45%
ffn                      9.8ms       54%
  └─ aten::mm (gate)     4.1ms       23%
  └─ aten::mm (up)       4.1ms       23%
  └─ aten::mm (down)     1.6ms        9%
  └─ aten::silu          0.05ms       0%
```

**Observation**: FFN dominates. Fusing gate and up projections into a single
`Linear(d_model, 2*ffn_dim)` reduces GEMM launch overhead by 1 kernel call.

---

## C.14 Test Harness — PyTorch Inference Primitives

```python
# ── test_appendix_w.py ─────────────────────────────────────────────────
"""Offline tests for PyTorch inference primitives.
Run with: python test_appendix_w.py"""

import math
import torch

def test_dtype_element_sizes():
    sizes = {
        torch.float32: 4, torch.bfloat16: 2, torch.float16: 2,
        torch.int8: 1, torch.bool: 1,
    }
    for dt, expected in sizes.items():
        t = torch.empty(1, dtype=dt)
        assert t.element_size() == expected, (
            f"{dt}: expected {expected}B, got {t.element_size()}B"
        )
    print(f"PASS: dtype element sizes verified ({len(sizes)} dtypes)")

def test_bf16_no_overflow():
    """BF16 should not overflow at values that overflow FP16."""
    large = torch.tensor(70_000.0)
    fp16_val = large.to(torch.float16)
    bf16_val = large.to(torch.bfloat16)
    assert not torch.isinf(bf16_val), "BF16 overflowed at 70_000!"
    assert torch.isinf(fp16_val),     "FP16 should overflow at 70_000"
    print("PASS: BF16 handles large values; FP16 overflows as expected")

def test_stride_and_contiguity():
    x = torch.randn(4, 8)
    assert x.is_contiguous()
    assert x.stride() == (8, 1)

    xt = x.t()
    assert not xt.is_contiguous()
    assert xt.stride() == (1, 8)

    xt_c = xt.contiguous()
    assert xt_c.is_contiguous()
    assert xt_c.data_ptr() != xt.data_ptr(), "contiguous() should allocate new storage"
    print("PASS: stride and contiguity behave correctly")

def test_view_shares_storage():
    a = torch.randn(16)
    b = a.view(4, 4)
    e = a.clone()
    assert a.storage().data_ptr() == b.storage().data_ptr(), "view should share storage"
    assert a.storage().data_ptr() != e.storage().data_ptr(), "clone should not share"
    print("PASS: view shares storage, clone does not")

def test_inference_mode_no_grad():
    x = torch.randn(4, requires_grad=True)
    with torch.inference_mode():
        y = x * 2
        assert not y.requires_grad
        assert y.is_inference()
    print("PASS: inference_mode disables gradients and marks tensors")

def test_memory_cost_calculation():
    def mem_bytes(shape, dtype):
        t = torch.empty(shape, dtype=dtype)
        return t.element_size() * t.numel()

    # KV cache block: 2 (K+V) × 8 heads × 16 block_size × 128 head_dim × 2 bytes
    kv_block = mem_bytes((2, 8, 16, 128), torch.float16)
    assert kv_block == 2 * 8 * 16 * 128 * 2, f"KV block size mismatch: {kv_block}"
    print(f"PASS: KV cache block = {kv_block:,} bytes ({kv_block/1024:.1f} KB)")

def test_masked_softmax():
    """Masked logit set to -inf should get exactly 0 probability."""
    logits = torch.tensor([2.0, 1.0, float('-inf'), 0.5])
    probs  = torch.softmax(logits, dim=-1)
    assert probs[2].item() == 0.0, "Masked logit should get 0 probability"
    assert not any(math.isnan(p.item()) for p in probs), "No NaN in masked softmax"
    assert abs(probs.sum().item() - 1.0) < 1e-6, "Probabilities should sum to 1"
    print("PASS: masked softmax is numerically stable")

def test_top_k_sampling():
    torch.manual_seed(42)
    logits = torch.randn(1, 128_000)
    probs  = torch.softmax(logits / 0.7, dim=-1)
    top_k_probs, top_k_ids = torch.topk(probs, k=50)
    sample_idx = torch.multinomial(top_k_probs, num_samples=1)
    token_id   = top_k_ids.gather(-1, sample_idx)
    assert 0 <= token_id.item() < 128_000, "Sampled token out of vocab range"
    print(f"PASS: top-k sampling returned token {token_id.item():,}")

def test_column_row_parallel_linear():
    """Simulate tensor parallelism: column split → row split → all_reduce."""
    d_model = 256
    ffn_dim = 512
    world_size = 4

    x = torch.randn(2, d_model)

    # Simulate column-parallel: each rank handles ffn_dim // 4 columns
    W_col = torch.randn(ffn_dim, d_model)
    shards = W_col.chunk(world_size, dim=0)
    partial_outs = [x @ shard.t() for shard in shards]
    col_out = torch.cat(partial_outs, dim=-1)  # equivalent to all-gather
    ref = x @ W_col.t()
    assert torch.allclose(col_out, ref, atol=1e-4), "Column-parallel mismatch"

    # Simulate row-parallel + all-reduce (sum)
    W_row = torch.randn(d_model, ffn_dim)
    row_shards = W_row.chunk(world_size, dim=-1)
    x_shards   = col_out.chunk(world_size, dim=-1)
    partial_row_outs = [xs @ rs.t() for xs, rs in zip(x_shards, row_shards)]
    row_out = sum(partial_row_outs)   # all_reduce SUM
    ref_row = col_out @ W_row.t()
    assert torch.allclose(row_out, ref_row, atol=1e-3), "Row-parallel mismatch"
    print("PASS: column-parallel and row-parallel linear simulation correct")


if __name__ == "__main__":
    test_dtype_element_sizes()
    test_bf16_no_overflow()
    test_stride_and_contiguity()
    test_view_shares_storage()
    test_inference_mode_no_grad()
    test_memory_cost_calculation()
    test_masked_softmax()
    test_top_k_sampling()
    test_column_row_parallel_linear()
    print("\n✓ All PyTorch inference primitive tests passed.")
```

**Expected output:**
```
PASS: dtype element sizes verified (5 dtypes)
PASS: BF16 handles large values; FP16 overflows as expected
PASS: stride and contiguity behave correctly
PASS: view shares storage, clone does not
PASS: inference_mode disables gradients and marks tensors
PASS: KV cache block = 32,768 bytes (32.0 KB)
PASS: masked softmax is numerically stable
PASS: top-k sampling returned token 74,312
PASS: column-parallel and row-parallel linear simulation correct

✓ All PyTorch inference primitive tests passed.
```

---

## C.15 Quick-Reference: Inference Checklist

| Item | Correct | Common mistake |
|---|---|---|
| Gradient mode | `torch.inference_mode()` | Forgetting `no_grad`, gradient memory leak |
| Mixed precision | `autocast(dtype=torch.bfloat16)` | FP32 activations on BF16 weights |
| Memory dtype | BF16 for Ampere+ | FP16 overflow at large values |
| Compilation | `torch.compile(mode="reduce-overhead")` | Uncompiled eager — 20–40% slower |
| Device transfers | `.to("cuda", non_blocking=True)` | Blocking H2D stalls decode loop |
| Pinned memory | Input tokens from pinned buffer | Pageable CPU memory halves H2D throughput |
| KV layout | Contiguous after transpose | Non-contiguous KV triggers hidden copies |
| Timing | CUDA events + `synchronize()` | `time.time()` without sync gives wrong numbers |
| Memory tracking | `reset_peak_memory_stats()` before forward | Cumulative peak includes prior runs |
| Distributed init | `dist.init_process_group("nccl")` | TCP fallback 10× slower than NCCL |

---

*Cross-references: Chapter 8.5 (CUDA Graphs), Chapter 10 (Quantization), Chapter 15
(Tensor Parallelism), Appendix L (CUDA C++ Introduction), Appendix J (libtorch C++ API),
Appendix M (Triton), Appendix N (CUTLASS).*
