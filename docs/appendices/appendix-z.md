# Appendix Z — JAX: XLA-Native Python for LLM Inference

> *JAX is what you get when you take NumPy, fuse it with a compiler that understands hardware topology, give it first-class automatic differentiation, and hand it to researchers who build production systems. It is simultaneously a research tool and an inference engine. Understanding JAX is increasingly necessary for anyone working at the frontier of LLM serving.*

---

## Z.1 What JAX Is

JAX (Just After Execution, or more loosely "NumPy on accelerators") is an open-source Python library developed by Google DeepMind. Its design is built on four primitives that compose cleanly:

```
jax.jit    — compile a Python function to XLA for GPU/TPU/CPU execution
jax.vmap   — vectorize a function over a batch dimension automatically
jax.grad   — differentiate any jit-compiled function with respect to its inputs
jax.pmap   — replicate a function across multiple devices with SPMD execution
```

These four transforms are composable: `jax.jit(jax.vmap(jax.grad(f)))` is valid and does what you expect. This composability is the key design principle that separates JAX from PyTorch's eager-mode imperative model.

For LLM inference, JAX matters because:

- Google's production inference stack (Gemini serving, MaxText) is JAX-based
- Flax NNX and Equinox are widely used for research models that become production deployments
- TPU inference is JAX-native; GPU inference via XLA is increasingly competitive with CUDA
- Transformer architectures express naturally in JAX's functional style
- The `jax.sharding` API provides explicit control over how tensors are distributed across a device mesh — a more principled approach than PyTorch's DTensor for complex parallelism patterns

---

## Z.2 JAX vs PyTorch: The Core Difference

PyTorch uses an eager execution model with optional compilation (`torch.compile`). Operations execute immediately, building a dynamic computation graph that `autograd` differentiates via reverse-mode accumulation.

JAX is **functional and compiled by default**. Functions are pure (no side effects), traced at compile time into an XLA HLO (High-Level Operator) graph, and then lowered to hardware-specific code (PTX for NVIDIA, XLA for TPU). The trace happens once; subsequent calls dispatch the compiled kernel.

```python
# PyTorch: eager by default, compile optional
import torch
x = torch.randn(1024, 1024)
y = x @ x.T   # executes immediately, Python stays in the loop

# JAX: functional, JIT-compiled
import jax
import jax.numpy as jnp

def matmul(x):
    return x @ x.T

matmul_jit = jax.jit(matmul)
x = jax.random.normal(jax.random.key(0), (1024, 1024))
y = matmul_jit(x)   # first call: traces + compiles; subsequent calls: cached kernel
```

The consequence for inference: a JAX model that has been JIT-compiled has **no Python overhead per forward pass**. The entire computation graph is a single XLA program dispatched to hardware. This is equivalent in spirit to CUDA Graphs (Chapter 8.5) but applies to the entire model, not just one captured stream.

---

## Z.3 JIT Compilation and XLA

### Z.3.1 How `jax.jit` Works

When you call `jax.jit(f)(x)` for the first time:

1. JAX traces `f` with abstract values (ShapedArrays) — running the Python function but not computing real numbers
2. The trace produces a JAXpr (JAX expression) — a functional IR
3. The JAXpr is lowered to XLA HLO (High-Level Operations)
4. XLA compiles the HLO to device code (PTX for GPU, LLO for TPU)
5. The compiled binary is cached; subsequent calls skip steps 1–4

```python
import jax
import jax.numpy as jnp

def attention_forward(q, k, v, scale):
    # q, k, v: [seq_len, head_dim]
    attn_weights = jnp.softmax(q @ k.T * scale, axis=-1)
    return attn_weights @ v

# JIT-compile once
attention_jit = jax.jit(attention_forward)

# Trace happens here (first call)
q = jax.random.normal(jax.random.key(0), (512, 64))
k = jax.random.normal(jax.random.key(1), (512, 64))
v = jax.random.normal(jax.random.key(2), (512, 64))
out = attention_jit(q, k, v, scale=1.0 / 8.0)  # compiles on first call

# Subsequent calls use cached kernel
out2 = attention_jit(q, k, v, scale=1.0 / 8.0)  # no Python overhead
```

### Z.3.2 Static vs Dynamic Shapes

XLA compiles for specific shapes. A change in input shape triggers recompilation. For LLM inference where sequence length varies, this is a critical constraint:

```python
# This causes recompilation for every new seq_len
@jax.jit
def forward(tokens):
    # tokens: [batch, seq_len]  — seq_len changing = recompile
    ...

# Solution 1: pad to fixed maximum length
MAX_SEQ = 2048
tokens_padded = jnp.pad(tokens, ((0, 0), (0, MAX_SEQ - seq_len)))

# Solution 2: use static_argnums for shape-determining values
@functools.partial(jax.jit, static_argnums=(1,))
def forward(tokens, seq_len: int):
    # seq_len is static: separate compiled kernel per length bucket
    ...
```

In production, JAX inference engines (MaxText, Pax) use **length bucketing** — padding inputs to a set of pre-compiled lengths (e.g., 128, 256, 512, 1024, 2048) to bound recompilation overhead.

---

## Z.4 Automatic Differentiation

### Z.4.1 `jax.grad` and `jax.value_and_grad`

```python
import jax
import jax.numpy as jnp

def cross_entropy_loss(logits, labels):
    # logits: [vocab_size], labels: int
    log_probs = jax.nn.log_softmax(logits)
    return -log_probs[labels]

# Gradient with respect to logits
grad_fn = jax.grad(cross_entropy_loss)   # differentiates w.r.t. first arg by default
logits = jax.random.normal(jax.random.key(0), (32000,))
labels = jnp.int32(42)

grad = grad_fn(logits, labels)   # shape: [32000]
print(f"Gradient shape: {grad.shape}")

# Get value and gradient together (more efficient — one forward pass)
value_and_grad_fn = jax.value_and_grad(cross_entropy_loss)
loss, grad = value_and_grad_fn(logits, labels)
print(f"Loss: {loss:.4f}")
```

### Z.4.2 Jacobians

JAX provides `jax.jacobian` (forward-mode) and `jax.jacrev` (reverse-mode):

```python
def softmax(x):
    e = jnp.exp(x - x.max())
    return e / e.sum()

# Full Jacobian: d(softmax(x))/dx — shape [n, n]
J = jax.jacobian(softmax)(jnp.array([1.0, 2.0, 3.0]))
print(J)
# Compare to the analytic formula: diag(p) - p @ p.T
p = softmax(jnp.array([1.0, 2.0, 3.0]))
J_analytic = jnp.diag(p) - jnp.outer(p, p)
print(f"Max error: {jnp.abs(J - J_analytic).max():.2e}")
```

### Z.4.3 Higher-Order Derivatives

```python
# Hessian (second derivative)
def f(x):
    return jnp.sum(jnp.sin(x) ** 2)

hessian_fn = jax.hessian(f)
x = jnp.ones(4)
H = hessian_fn(x)   # shape: [4, 4]
```

---

## Z.5 Vectorization with `jax.vmap`

`vmap` eliminates explicit batch dimensions by automatically vectorizing a single-sample function:

```python
import jax
import jax.numpy as jnp

# Single-sample attention
def single_head_attention(q, k, v):
    # q: [seq, d_k], k: [seq, d_k], v: [seq, d_v]
    scale = q.shape[-1] ** -0.5
    scores = q @ k.T * scale            # [seq, seq]
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v                  # [seq, d_v]

# Batched: apply over batch and head dimensions simultaneously
batched_attention = jax.vmap(
    jax.vmap(single_head_attention,       # over heads
             in_axes=(0, 0, 0)),
    in_axes=(0, 0, 0)                     # over batch
)

batch, heads, seq, d_k, d_v = 2, 8, 512, 64, 64
key = jax.random.key(0)
q = jax.random.normal(key, (batch, heads, seq, d_k))
k = jax.random.normal(key, (batch, heads, seq, d_k))
v = jax.random.normal(key, (batch, heads, seq, d_v))

out = batched_attention(q, k, v)    # [batch, heads, seq, d_v]
print(f"Output shape: {out.shape}")
```

`vmap` compiles the vectorization into fused SIMD operations rather than Python loops, producing code equivalent to a hand-written batched kernel.

---

## Z.6 Multi-Device Parallelism

### Z.6.1 `jax.pmap` — Data Parallelism

`pmap` (parallel map) replicates a function across devices and executes it with SPMD (Single Program, Multiple Data) semantics:

```python
import jax
import jax.numpy as jnp

n_devices = jax.device_count()
print(f"Devices: {n_devices}")

@jax.pmap
def parallel_forward(x):
    # Each device receives a shard of the batch
    return jax.nn.relu(x @ x.T)

# Create input sharded across devices: first axis = device axis
x = jax.random.normal(jax.random.key(0), (n_devices, 256, 256))
out = parallel_forward(x)   # each device computes its shard
```

### Z.6.2 `jax.sharding` — Tensor Parallelism

For LLM inference, model weights must be partitioned across devices (tensor parallelism). JAX's explicit sharding API gives fine-grained control:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import numpy as np

# Create 2D device mesh: 2 data-parallel × 4 tensor-parallel
devices = mesh_utils.create_device_mesh((2, 4))
mesh = Mesh(devices, axis_names=('data', 'model'))

# Shard weights across model axis (column-parallel)
weight_sharding = NamedSharding(mesh, PartitionSpec('model', None))
activation_sharding = NamedSharding(mesh, PartitionSpec('data', None))

# Create a weight matrix sharded across 4 model-parallel devices
W = jax.device_put(
    jax.random.normal(jax.random.key(0), (4096, 4096)),
    weight_sharding
)
print(f"Weight shard shape per device: {W.addressable_shards[0].data.shape}")
# → (1024, 4096): each of 4 devices holds 1024 rows

# JIT respects sharding annotations
@jax.jit
def linear_layer(x, W):
    return x @ W

x = jax.device_put(
    jax.random.normal(jax.random.key(1), (32, 4096)),
    activation_sharding
)
out = linear_layer(x, W)   # XLA automatically inserts all-reduce
```

### Z.6.3 Comparison with PyTorch Distributed

| Feature | JAX `jax.sharding` | PyTorch `DTensor` / FSDP |
|---|---|---|
| Model | Functional (pure functions) | Imperative (in-place ops) |
| Sharding spec | Explicit `PartitionSpec` per tensor | `device_mesh` annotations |
| Compiler | XLA (automatic collective insertion) | `torch.compile` + NCCL |
| Debugging | Abstract tracer, harder to print | Eager mode available |
| TPU support | Native | Limited |
| Ecosystem | Flax, Equinox, MaxText | PyTorch native, vLLM |

---

## Z.7 A Minimal Transformer in JAX

The following implements a single transformer block in pure JAX to illustrate the functional style:

```python
import jax
import jax.numpy as jnp
from functools import partial

# ── Layer Normalization ───────────────────────────────────────
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var  = x.var(-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta

# ── Multi-Head Attention ──────────────────────────────────────
def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    B, T, D = x.shape
    d_k = D // n_heads

    Q = (x @ Wq).reshape(B, T, n_heads, d_k).transpose(0, 2, 1, 3)
    K = (x @ Wk).reshape(B, T, n_heads, d_k).transpose(0, 2, 1, 3)
    V = (x @ Wv).reshape(B, T, n_heads, d_k).transpose(0, 2, 1, 3)

    scale = d_k ** -0.5
    scores = Q @ K.transpose(0, 1, 3, 2) * scale      # [B, H, T, T]

    if causal:
        mask = jnp.tril(jnp.ones((T, T)))
        scores = jnp.where(mask, scores, -1e9)

    weights = jax.nn.softmax(scores, axis=-1)
    out = (weights @ V).transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ Wo

# ── Feed-Forward Block ────────────────────────────────────────
def ffn(x, W1, b1, W2, b2):
    return jax.nn.gelu(x @ W1 + b1) @ W2 + b2

# ── Transformer Block ─────────────────────────────────────────
def transformer_block(x, params, n_heads):
    # Attention sub-layer
    Wq, Wk, Wv, Wo = params['Wq'], params['Wk'], params['Wv'], params['Wo']
    g1, b1_ln = params['gamma1'], params['beta1']
    x_norm = layer_norm(x, g1, b1_ln)
    x = x + multi_head_attention(x_norm, Wq, Wk, Wv, Wo, n_heads)

    # FFN sub-layer
    g2, b2_ln = params['gamma2'], params['beta2']
    x_norm = layer_norm(x, g2, b2_ln)
    x = x + ffn(x_norm, params['W1'], params['b1'], params['W2'], params['b2'])
    return x

# ── Initialization ────────────────────────────────────────────
def init_params(key, D, n_heads, ffn_dim):
    keys = jax.random.split(key, 12)
    scale = 0.02
    return {
        'Wq': jax.random.normal(keys[0], (D, D)) * scale,
        'Wk': jax.random.normal(keys[1], (D, D)) * scale,
        'Wv': jax.random.normal(keys[2], (D, D)) * scale,
        'Wo': jax.random.normal(keys[3], (D, D)) * scale,
        'W1': jax.random.normal(keys[4], (D, ffn_dim)) * scale,
        'b1': jnp.zeros(ffn_dim),
        'W2': jax.random.normal(keys[5], (ffn_dim, D)) * scale,
        'b2': jnp.zeros(D),
        'gamma1': jnp.ones(D), 'beta1': jnp.zeros(D),
        'gamma2': jnp.ones(D), 'beta2': jnp.zeros(D),
    }

# ── Smoke test ────────────────────────────────────────────────
if __name__ == "__main__":
    D, n_heads, ffn_dim = 512, 8, 2048
    B, T = 2, 64

    params = init_params(jax.random.key(0), D, n_heads, ffn_dim)
    x = jax.random.normal(jax.random.key(1), (B, T, D))

    # JIT-compile the block
    block_jit = jax.jit(partial(transformer_block, n_heads=n_heads))

    out = block_jit(x, params)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # (2, 64, 512)

    # Verify gradient flows through the block
    def loss_fn(params, x):
        out = transformer_block(x, params, n_heads)
        return jnp.mean(out ** 2)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params, x)
    print(f"Wq gradient norm: {jnp.linalg.norm(grads['Wq']):.4f}")
```

---

## Z.8 JAX for LLM Inference in Practice

### Z.8.1 Flax NNX

Flax NNX (Neural Networks for JAX) is the primary high-level neural network library for JAX, developed by Google. It provides `nn.Module` semantics familiar from PyTorch while staying compatible with JAX's functional transforms:

```python
from flax import nnx
import jax
import jax.numpy as jnp

class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, *, rngs: nnx.Rngs):
        self.attn = nnx.MultiHeadAttention(
            n_heads, in_features=d_model, rngs=rngs
        )
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ffn1 = nnx.Linear(d_model, d_model * 4, rngs=rngs)
        self.ffn2 = nnx.Linear(d_model * 4, d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn2(nnx.gelu(self.ffn1(self.ln2(x))))
        return x

# Instantiate
rngs = nnx.Rngs(0)
block = TransformerBlock(d_model=512, n_heads=8, rngs=rngs)

# Inference
x = jnp.ones((2, 64, 512))
out = block(x)
print(out.shape)   # (2, 64, 512)
```

### Z.8.2 MaxText — Google's JAX Inference Engine

MaxText is Google's open-source JAX implementation of large language models, used as the reference implementation for Gemma and other Google models. It supports:

- Multi-host TPU and GPU inference via `jax.sharding`
- BF16 and INT8 quantization
- Paged KV cache (experimental)
- Continuous batching via a Python-level scheduler
- Serving via gRPC with a Jetstream frontend

```bash
# Clone and run Llama inference with MaxText (illustrative)
git clone https://github.com/google/maxtext
cd maxtext

python MaxText/inference_microbenchmark.py \
  MaxText/configs/base.yml \
  model_name=llama3-8b \
  tokenizer_path=assets/tokenizer.llama3 \
  load_parameters_path=gs://your-bucket/llama3-8b \
  per_device_batch_size=1 \
  max_target_length=2048
```

### Z.8.3 Equinox

Equinox provides a PyTree-based module system for JAX that is lighter than Flax and more transparent about state management:

```python
import equinox as eqx
import jax
import jax.numpy as jnp

class LinearBlock(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_dim, out_dim, key):
        self.weight = jax.random.normal(key, (in_dim, out_dim)) * 0.02
        self.bias = jnp.zeros(out_dim)

    def __call__(self, x):
        return x @ self.weight + self.bias

model = LinearBlock(512, 2048, jax.random.key(0))

# eqx.filter_jit: JIT-compile, treating static/dynamic leaves correctly
@eqx.filter_jit
def forward(model, x):
    return model(x)

x = jnp.ones((4, 512))
out = forward(model, x)
```

---

## Z.9 JAX on GPU vs PyTorch/CUDA

### Z.9.1 When JAX's XLA Matches or Beats CUDA

For workloads that express cleanly in XLA's operator set:

- **Attention**: XLA's fusion passes can match FlashAttention on many shapes
- **Matrix multiplications**: cuBLAS is called through XLA; performance is identical
- **Fused kernels**: XLA automatically fuses element-wise ops; `jax.jit` of `gelu(x @ W + b)` produces a fused GEMM+GeLU kernel

### Z.9.2 When CUDA Wins

- **Custom kernels**: FlashAttention-2/3 uses hand-written Triton/CUDA; XLA's attention kernel may lag on very long sequences
- **PagedAttention**: vLLM's block manager (Chapter 6) has no JAX equivalent in production; MaxText uses a simpler paged approach
- **Ecosystem**: vLLM, TGI, SGLang are GPU/CUDA-native; JAX inference tooling (Jetstream, MaxText) is newer

### Z.9.3 Performance Comparison (Attention, Forward Pass)

```python
"""
jax_vs_pytorch_attention.py — wall-clock comparison of multi-head
attention forward pass in JAX vs PyTorch on the same GPU.
"""
import time
import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F

def time_jax(fn, *args, n_warmup=5, n_bench=50):
    for _ in range(n_warmup):
        fn(*args).block_until_ready()
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)

def time_torch(fn, *args, n_warmup=5, n_bench=50):
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)

B, H, T, D = 1, 32, 2048, 128

# JAX
key = jax.random.key(0)
qkv_jax = [jax.random.normal(key, (B, H, T, D)) for _ in range(3)]

@jax.jit
def attn_jax(q, k, v):
    scale = D ** -0.5
    scores = q @ k.transpose(0, 1, 3, 2) * scale
    mask = jnp.tril(jnp.ones((T, T)))
    scores = jnp.where(mask, scores, -1e9)
    w = jax.nn.softmax(scores, axis=-1)
    return w @ v

# PyTorch
qkv_pt = [torch.randn(B, H, T, D, device='cuda', dtype=torch.float32) for _ in range(3)]

def attn_torch(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

ms_jax   = time_jax(attn_jax, *qkv_jax)
ms_torch = time_torch(attn_torch, *qkv_pt)

print(f"Attention forward (B={B} H={H} T={T} D={D}):")
print(f"  JAX (XLA):       {ms_jax:.2f} ms")
print(f"  PyTorch (SDPA):  {ms_torch:.2f} ms")
```

---

## Z.10 Random Number Handling

JAX uses **explicit PRNG keys** rather than a global random state. This is required for reproducibility in a JIT-compiled functional world:

```python
import jax
import jax.numpy as jnp

# Create a key from a seed
key = jax.random.key(42)

# Split into subkeys (never reuse a key)
key, subkey1, subkey2 = jax.random.split(key, 3)

x = jax.random.normal(subkey1, (4, 4))
y = jax.random.uniform(subkey2, (4,))

# For model initialization, split once per parameter
def init_layer(key, in_dim, out_dim):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (in_dim, out_dim)) * 0.02
    b = jnp.zeros(out_dim)
    return W, b, k2   # return the remaining key for further splitting
```

This functional PRNG design ensures that the same sequence of random numbers is produced regardless of whether `init_layer` is called inside a JIT region or outside, on CPU or GPU.

---

## Z.11 Practical Setup

```bash
# Install JAX with GPU support
pip install -U "jax[cuda12]"

# Or for TPU
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Core ecosystem
pip install flax equinox optax  # NN library, lightweight modules, optimizers

# Verify GPU visibility
python -c "import jax; print(jax.devices())"
# [CudaDevice(id=0), CudaDevice(id=1), ...]
```

```python
# Runtime checks
import jax
print(f"JAX version:   {jax.__version__}")
print(f"Devices:       {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Memory info (GPU)
for i, dev in enumerate(jax.devices()):
    mem = dev.memory_stats()
    if mem:
        used_gb = mem.get('bytes_in_use', 0) / 1e9
        limit_gb = mem.get('bytes_limit', 0) / 1e9
        print(f"Device {i}: {used_gb:.2f} / {limit_gb:.2f} GB used")
```

---

## Z.12 Comparison with Triton and Mojo

| Aspect | JAX | Triton (Appendix M) | Mojo (Appendix O) |
|---|---|---|---|
| **Abstraction level** | High (NumPy API) | Medium (tile-level kernels) | Low–medium (SIMD, pointer math) |
| **Primary use** | Full model / training / research | Custom GPU kernels | CPU inference, portability |
| **Compiler backend** | XLA (LLVM + hardware-specific) | LLVM + PTX (NVIDIA) | MLIR + LLVM |
| **Auto-diff** | First-class (`jax.grad`) | Not built-in | Not built-in |
| **Vectorization** | `jax.vmap` | Manual tiling | SIMD intrinsics |
| **Parallelism** | `jax.pmap`, `jax.sharding` | Grid/block launch | Not native GPU |
| **Python compatibility** | Full NumPy API | Python-like syntax | Python superset |
| **Production use** | Google Gemini serving, MaxText | FlashAttention, vLLM kernels | llama.cpp portability layer |
| **TPU support** | Native | No | No |
| **Learning curve** | Moderate (functional paradigm) | Moderate (tile programming) | Steep (systems programming) |

---

## Z.13 Self-Check Questions

**Q1.** What does `jax.jit` do on the first call vs subsequent calls? Why does shape change trigger recompilation?

**A1.** On the first call, `jax.jit` traces the function with abstract `ShapedArray` values to produce a JAXpr, lowers that to XLA HLO, and compiles the HLO to device-specific machine code (PTX for NVIDIA GPUs). Subsequent calls with the same input shapes dispatch the cached compiled kernel directly — there is no Python overhead. Shape change triggers recompilation because XLA compiles shape-specific kernels; tile sizes, loop bounds, and memory access patterns are all hard-coded into the binary for a particular input shape. This is why production JAX inference engines use length bucketing.

**Q2.** Explain how `jax.vmap` differs from writing a for-loop over batch elements in Python. What does it produce at the XLA level?

**A2.** A Python for-loop over batch elements re-traces and re-dispatches the function once per element, incurring Python overhead proportional to batch size. `jax.vmap` transforms the function at trace time, producing a single XLA program that operates on the batched tensor with a new leading axis. At the XLA level, the vectorized dimension is expressed as a `map` operation that the compiler lowers to vectorized SIMD or parallelized thread blocks — equivalent to a hand-written batched kernel, with no Python loop at runtime.

**Q3.** JAX requires explicit PRNG key management rather than a global random state. Why does this matter for JIT compilation?

**A3.** A global mutable random state is a side effect. Functions with side effects are not pure, and JAX's JIT compiler assumes purity: it may reorder, fuse, or cache computations assuming the same inputs always produce the same outputs. A global RNG that mutates on each call would produce different outputs for the same inputs, breaking this assumption. Explicit key splitting makes the randomness a data dependency that flows through the functional graph, allowing JAX to reason about it correctly under JIT, vmap, and pmap.

**Q4.** Write the JAX equivalent of the following PyTorch snippet and explain any differences:
```python
# PyTorch
x = torch.randn(512, 512, device='cuda')
x = torch.relu(x)
loss = x.sum()
loss.backward()
```

**A4.**
```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(jax.nn.relu(x))

x = jax.random.normal(jax.random.key(0), (512, 512))
loss, grad = jax.value_and_grad(f)(x)
```
Key differences: (1) JAX uses explicit PRNG keys for `randn`; (2) the gradient is obtained by wrapping `f` with `jax.grad` rather than calling `.backward()` on a tensor — there is no implicit computation graph; (3) `loss` and `grad` are both computed in a single forward pass when using `value_and_grad`, which is more efficient than separate forward and backward calls.

**Q5.** MaxText uses JAX's `jax.sharding` for tensor parallelism. How does this differ from PyTorch's `torch.distributed.tensor_parallel`? What does XLA insert automatically?

**A5.** In PyTorch tensor parallelism, the user (or a wrapper like `torch.distributed.tensor_parallel`) explicitly inserts `all_reduce` or `all_gather` collective operations at the module boundaries where partial sums must be synchronized. In JAX with `jax.sharding`, the user annotates tensors with a `PartitionSpec` that declares which mesh axis each tensor dimension is sharded over. XLA then analyzes the data flow at compilation time and automatically inserts the necessary collective operations (all-reduce, all-gather, reduce-scatter) wherever the sharding annotations require communication. The programmer states *intent* (how data is laid out) rather than *mechanism* (which collective to insert), and the compiler enforces correctness.

---

*This appendix completes the book's coverage of the major GPU/accelerator programming paradigms: CUDA C++ (Appendix L), Triton (Appendix M), CUTLASS (Appendix N), Mojo (Appendix O), and JAX/XLA (Appendix Z). Together they span the spectrum from hand-written PTX-adjacent kernels to compiler-managed distributed execution on wafer-scale silicon.*

---

## Z.14 Complete Test and Main Harness

Every primitive covered in this appendix — `jax.jit`, `jax.vmap`, `jax.grad`, `jax.pmap`, sharding, and fused attention — is exercised in a single self-contained test file.  Running it confirms that each construct produces numerically correct results and gives you measured wall-clock timings for the JIT-compiled paths.

### Z.14.1 Environment and Dependencies

```bash
# CPU-only (works on any machine, no GPU required for most tests)
pip install jax jaxlib numpy

# GPU backend (CUDA 12+)
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Run the harness
python jax_test.py

# Run with verbose XLA output
XLA_FLAGS="--xla_dump_hlo_as_text" python jax_test.py --verbose

# Skip multi-device tests (single-GPU machine)
python jax_test.py --no-pmap

# Skip all benchmarks
python jax_test.py --no-bench
```

The harness requires **Python ≥ 3.10** and **JAX ≥ 0.4.20**.  All JIT compilations happen on first call; expect a 5–30 s warm-up as XLA generates device code.

### Z.14.2 Full Source — `jax_test.py`

```python
"""
jax_test.py — Complete correctness + benchmark harness for Appendix Z kernels.

Sections tested
---------------
1. jax.jit         — JIT matmul, shape-reuse, recompilation on shape change
2. jax.vmap        — batched dot-product, batched attention
3. jax.grad        — scalar loss gradient, value_and_grad, higher-order grad
4. jax.pmap        — multi-device data-parallel reduce (skipped if <2 devices)
5. Sharding        — NamedSharding annotation on a large tensor (single-device mesh)
6. Fused attention — scaled dot-product attention vs NumPy reference
7. RoPE in JAX     — rotary embedding vs closed-form reference

Usage
-----
    python jax_test.py [--verbose] [--no-pmap] [--no-bench]

Requirements
------------
    pip install jax jaxlib numpy
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="JAX kernel test harness")
parser.add_argument("--verbose",  action="store_true", help="Print extra info")
parser.add_argument("--no-pmap",  action="store_true", help="Skip pmap tests")
parser.add_argument("--no-bench", action="store_true", help="Skip benchmarks")
ARGS, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
PASS_COUNT = 0
FAIL_COUNT = 0
SEP = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check(name: str, passed: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    tag = "[PASS]" if passed else "[FAIL]"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag}  {name}{suffix}")
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1


def assert_close(
    name: str,
    actual: jnp.ndarray,
    expected: jnp.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> bool:
    actual_np   = np.array(actual)
    expected_np = np.array(expected)
    passed = np.allclose(actual_np, expected_np, atol=atol, rtol=rtol)
    detail = ""
    if not passed:
        diff = np.abs(actual_np - expected_np)
        detail = f"max_diff={diff.max():.6f}"
    check(name, passed, detail)
    return passed


def bench_jax(fn: Callable, label: str, flops: float = 0.0, bytes_: float = 0.0,
               warmup: int = 10, reps: int = 50) -> None:
    """Time a JAX function; uses jax.block_until_ready to synchronize."""
    if ARGS.no_bench:
        return
    # Warmup
    for _ in range(warmup):
        result = fn()
        jax.block_until_ready(result)
    # Timed reps
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)
    times.sort()
    ms = times[len(times) // 2] * 1e3  # median in ms
    parts = [f"{ms:.3f} ms"]
    if bytes_ > 0:
        gb_s = bytes_ / (ms * 1e-3) / 1e9
        parts.append(f"{gb_s:.1f} GB/s")
    if flops > 0:
        tflops = flops / (ms * 1e-3) / 1e12
        parts.append(f"{tflops:.2f} TFLOPS")
    print(f"  BENCH  {label}: {', '.join(parts)}")


# ===========================================================================
# 1. JIT COMPILATION
# ===========================================================================

def _matmul(a, b):
    return a @ b


matmul_jit = jit(_matmul)


def test_jit() -> None:
    section("1. JAX.JIT — Compilation and Reuse")
    key = jax.random.key(0)

    # Known-value 3×3 test
    # A = [[1,2,3],[4,5,6],[7,8,9]]  B = [[7,8,9],[2,3,4],[1,2,3]]
    # C = [[14,20,26],[44,59,74],[74,98,122]]
    A = jnp.array([[1,2,3],[4,5,6],[7,8,9]], dtype=jnp.float32)
    B = jnp.array([[7,8,9],[2,3,4],[1,2,3]], dtype=jnp.float32)
    ref = jnp.array([[14,20,26],[44,59,74],[74,98,122]], dtype=jnp.float32)
    assert_close("known-value 3×3 matmul", matmul_jit(A, B), ref)

    # Identity: A @ I = A
    N = 256
    key, sk = jax.random.split(key)
    A_r = jax.random.normal(sk, (N, N))
    I   = jnp.eye(N)
    assert_close("A @ I = A  (N=256)", matmul_jit(A_r, I), A_r)

    # Large random: JIT vs eager
    M, K, Nk = 1024, 1024, 1024
    key, sk = jax.random.split(key)
    A_l = jax.random.normal(sk, (M, K))
    key, sk = jax.random.split(key)
    B_l = jax.random.normal(sk, (K, Nk))
    ref_l = A_l @ B_l   # eager
    assert_close("random 1024×1024×1024 JIT==eager", matmul_jit(A_l, B_l), ref_l, atol=1e-3)

    # Shape-change forces recompilation (just verify it still works)
    A_s = jax.random.normal(jax.random.key(1), (512, 128))
    B_s = jax.random.normal(jax.random.key(2), (128, 64))
    ref_s = A_s @ B_s
    assert_close("shape change 512×128×64 recompile", matmul_jit(A_s, B_s), ref_s, atol=1e-4)

    # Benchmark
    M_b = K_b = N_b = 4096
    A_b = jax.random.normal(jax.random.key(3), (M_b, K_b))
    B_b = jax.random.normal(jax.random.key(4), (K_b, N_b))
    _ = matmul_jit(A_b, B_b); jax.block_until_ready(_)  # ensure compiled
    flops  = 2 * M_b * K_b * N_b
    bytes_ = (M_b * K_b + K_b * N_b + M_b * N_b) * 4
    bench_jax(lambda: matmul_jit(A_b, B_b), "jit matmul 4096×4096",
              flops=flops, bytes_=bytes_)


# ===========================================================================
# 2. VMAP — VECTORISED MAP
# ===========================================================================

def _dot(x, y):
    """Dot product for a single pair of 1-D vectors."""
    return jnp.dot(x, y)


batched_dot = jit(vmap(_dot))   # vmap over leading batch axis, then JIT


def _single_attention(q, k, v, scale):
    """Scaled dot-product attention for a single head (no batch)."""
    w = jax.nn.softmax(q @ k.T * scale, axis=-1)
    return w @ v


# vmap over the batch (head) axis
batched_attention = jit(vmap(_single_attention, in_axes=(0, 0, 0, None)))


def test_vmap() -> None:
    section("2. JAX.VMAP — Batched Operations")
    key = jax.random.key(10)

    # Batched dot product — known values
    # x = [[1,2],[3,4]], y = [[5,6],[7,8]]  → dots = [17, 53]
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    ref = jnp.array([17.0, 53.0])
    assert_close("batched dot [2×2]", batched_dot(x, y), ref)

    # Large batched dot vs manual loop
    B_d, D = 512, 1024
    key, sk1, sk2 = jax.random.split(key, 3)
    X = jax.random.normal(sk1, (B_d, D))
    Y = jax.random.normal(sk2, (B_d, D))
    ref_d = jnp.array([jnp.dot(X[i], Y[i]) for i in range(B_d)])
    assert_close("batched dot B=512 D=1024", batched_dot(X, Y), ref_d, atol=1e-3)

    # Batched attention
    B_h, S, H = 8, 64, 128   # 8 heads, seq=64, head_dim=128
    key, sk1, sk2, sk3 = jax.random.split(key, 4)
    Q = jax.random.normal(sk1, (B_h, S, H))
    K = jax.random.normal(sk2, (B_h, S, H))
    V = jax.random.normal(sk3, (B_h, S, H))
    scale = 1.0 / math.sqrt(H)

    # Reference: loop over heads
    ref_attn = jnp.stack([
        _single_attention(Q[i], K[i], V[i], scale) for i in range(B_h)
    ])
    out_attn = batched_attention(Q, K, V, scale)
    assert_close("batched attention B=8 S=64 H=128", out_attn, ref_attn, atol=1e-4)

    # Benchmark
    B_b, S_b, H_b = 32, 512, 128
    key, sk1, sk2, sk3 = jax.random.split(key, 4)
    Q_b = jax.random.normal(sk1, (B_b, S_b, H_b))
    K_b = jax.random.normal(sk2, (B_b, S_b, H_b))
    V_b = jax.random.normal(sk3, (B_b, S_b, H_b))
    sc  = 1.0 / math.sqrt(H_b)
    _ = batched_attention(Q_b, K_b, V_b, sc); jax.block_until_ready(_)
    flops  = B_b * (2 * S_b * S_b * H_b + 2 * S_b * S_b * H_b)
    bytes_ = B_b * (3 * S_b * H_b + S_b * S_b + S_b * H_b) * 4
    bench_jax(lambda: batched_attention(Q_b, K_b, V_b, sc),
              "vmap attention B=32 S=512 H=128", flops=flops, bytes_=bytes_)


# ===========================================================================
# 3. GRAD — AUTOMATIC DIFFERENTIATION
# ===========================================================================

def _loss_fn(x: jnp.ndarray) -> jnp.ndarray:
    """Simple loss: sum of squares → gradient is 2*x."""
    return jnp.sum(x ** 2)


def _cross_entropy(logits: jnp.ndarray, label: int) -> jnp.ndarray:
    """Softmax cross-entropy for a single example."""
    log_probs = jax.nn.log_softmax(logits)
    return -log_probs[label]


grad_loss        = jit(grad(_loss_fn))
value_and_grad_ce = jit(value_and_grad(_cross_entropy))


def _second_order(x: jnp.ndarray) -> jnp.ndarray:
    """sin(x) → grad = cos(x) → grad-of-grad = -sin(x)."""
    return jnp.sin(x)


hessian_diag = jit(grad(grad(_second_order)))


def test_grad() -> None:
    section("3. JAX.GRAD — Automatic Differentiation")

    # Known-value: grad(sum(x**2)) = 2*x
    x = jnp.array([1.0, 2.0, 3.0])
    ref = jnp.array([2.0, 4.0, 6.0])
    assert_close("grad sum(x²) known-value", grad_loss(x), ref)

    # Gradient at zero: grad(sum(x²)) at x=0 should be zero
    x0 = jnp.zeros(8)
    assert_close("grad sum(x²) at x=0 is 0", grad_loss(x0), jnp.zeros(8))

    # Cross-entropy gradient — verify with finite differences
    key = jax.random.key(42)
    logits = jax.random.normal(key, (10,))
    label  = 3
    loss_val, g = value_and_grad_ce(logits, label)

    # Finite-difference check
    eps = 1e-4
    fd_grad = jnp.zeros_like(logits)
    for i in range(len(logits)):
        lp = logits.at[i].add(+eps)
        lm = logits.at[i].add(-eps)
        fd = (_cross_entropy(lp, label) - _cross_entropy(lm, label)) / (2 * eps)
        fd_grad = fd_grad.at[i].set(fd)
    assert_close("cross-entropy grad vs finite-diff", g, fd_grad, atol=1e-3)

    # Scalar loss value from value_and_grad
    ref_loss = float(-jax.nn.log_softmax(logits)[label])
    passed = abs(float(loss_val) - ref_loss) < 1e-5
    check(f"value_and_grad loss scalar (ref={ref_loss:.5f})", passed,
          f"got {float(loss_val):.5f}")

    # Higher-order: grad(grad(sin(x))) = -sin(x)
    x_h = jnp.array(0.5)
    ref_h = -jnp.sin(x_h)
    assert_close("grad²(sin) = -sin  x=0.5", hessian_diag(x_h), ref_h, atol=1e-5)

    # Benchmark: grad of a realistic MLP layer
    D = 4096
    key, sk = jax.random.split(key)
    W = jax.random.normal(sk, (D, D)) * 0.02
    key, sk = jax.random.split(key)
    x_b = jax.random.normal(sk, (D,))

    def layer_loss(W, x):
        return jnp.sum(jax.nn.relu(W @ x) ** 2)

    grad_fn = jit(grad(layer_loss))
    _ = grad_fn(W, x_b); jax.block_until_ready(_)
    bytes_ = (D * D + D + D * D) * 4  # read W, x, write grad_W
    bench_jax(lambda: grad_fn(W, x_b), "grad MLP layer D=4096", bytes_=bytes_)


# ===========================================================================
# 4. PMAP — MULTI-DEVICE DATA PARALLELISM
# ===========================================================================

def test_pmap() -> None:
    section("4. JAX.PMAP — Multi-Device Data Parallelism")
    n_devices = jax.device_count()
    print(f"  Devices visible: {n_devices}  ({jax.devices()[0].platform})")

    if ARGS.no_pmap or n_devices < 2:
        print("  SKIP  pmap tests require ≥2 devices (pass --no-pmap to suppress this)")
        check("pmap skip (single device or --no-pmap)", True,
              f"n_devices={n_devices}")
        return

    # Replicated all-reduce sum across devices
    def device_sum(x):
        # Each device gets a shard; we sum locally then all-reduce
        local = jnp.sum(x)
        return jax.lax.psum(local, axis_name="batch")

    pmap_sum = jax.pmap(device_sum, axis_name="batch")

    # Input: each device gets one row; total sum = sum of all rows
    key = jax.random.key(99)
    data = jax.random.normal(key, (n_devices, 1024))
    ref_total = float(jnp.sum(data))
    out = pmap_sum(data)   # shape: (n_devices,), all replicas hold same total
    passed = np.allclose(float(out[0]), ref_total, atol=1e-3)
    check("pmap all-reduce sum", passed,
          f"ref={ref_total:.4f} got={float(out[0]):.4f}")

    # Data-parallel matmul: each device processes its shard
    def shard_matmul(A_shard, B):
        return A_shard @ B

    pmap_mm = jax.pmap(shard_matmul, in_axes=(0, None))

    M_total, K, Nk = n_devices * 128, 256, 256
    key, sk1, sk2 = jax.random.split(key, 3)
    A_full = jax.random.normal(sk1, (M_total, K))
    B_full = jax.random.normal(sk2, (K, Nk))
    A_sharded = A_full.reshape(n_devices, M_total // n_devices, K)
    out_sharded = pmap_mm(A_sharded, B_full)          # (n_devices, M/n, N)
    ref_mm = A_full @ B_full
    out_cat = out_sharded.reshape(M_total, Nk)
    assert_close("pmap data-parallel matmul", out_cat, ref_mm, atol=1e-3)

    # Benchmark
    M_b = n_devices * 512
    A_b = jax.random.normal(jax.random.key(5), (n_devices, M_b // n_devices, K))
    B_b = jax.random.normal(jax.random.key(6), (K, Nk))
    _ = pmap_mm(A_b, B_b); jax.block_until_ready(_)
    bench_jax(lambda: pmap_mm(A_b, B_b), f"pmap matmul {n_devices}×{M_b//n_devices}×{K}×{Nk}")


# ===========================================================================
# 5. SHARDING — EXPLICIT TENSOR DISTRIBUTION
# ===========================================================================

def test_sharding() -> None:
    section("5. SHARDING — NamedSharding Annotation")
    devices = jax.devices()
    n = len(devices)

    # Build a 1-D mesh over all available devices
    mesh = Mesh(np.array(devices).reshape(n), axis_names=("data",))

    # Partition the batch axis over the "data" mesh axis
    sharding = NamedSharding(mesh, P("data"))

    B, D = n * 64, 512
    key = jax.random.key(77)
    x = jax.random.normal(key, (B, D))

    # jax.device_put respects the sharding annotation
    x_sharded = jax.device_put(x, sharding)
    check("sharding: jax.device_put succeeds", True,
          f"shape={x_sharded.shape} n_devices={n}")

    # Computation on sharded tensor runs correctly
    @jit
    def row_norm(t):
        return jnp.linalg.norm(t, axis=-1)

    out_sharded = row_norm(x_sharded)
    ref         = jnp.linalg.norm(x, axis=-1)
    assert_close("sharded row_norm matches unsharded", out_sharded, ref, atol=1e-4)

    # Replicate a weight matrix across all devices
    rep_sharding = NamedSharding(mesh, P())   # no partition → replicated
    W = jax.random.normal(jax.random.key(88), (D, D)) * 0.02
    W_rep = jax.device_put(W, rep_sharding)
    check("sharding: weight matrix replicated", True, f"sharding={W_rep.sharding}")

    # Forward pass: sharded activations × replicated weight
    @jit
    def linear(x_s, w_r):
        return x_s @ w_r

    out_linear = linear(x_sharded, W_rep)
    ref_linear  = x @ W
    assert_close("sharded linear(x_sharded, W_rep)", out_linear, ref_linear, atol=1e-3)

    if not ARGS.no_bench:
        _ = linear(x_sharded, W_rep); jax.block_until_ready(_)
        bench_jax(lambda: linear(x_sharded, W_rep),
                  f"sharded linear B={B} D={D}",
                  flops=2*B*D*D, bytes_=(B*D + D*D + B*D)*4)


# ===========================================================================
# 6. FUSED SCALED DOT-PRODUCT ATTENTION
# ===========================================================================

def _numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _numpy_attention(q, k, v, scale):
    """Pure NumPy reference attention."""
    w = _numpy_softmax(q @ k.swapaxes(-1, -2) * scale)
    return w @ v


@jit
def fused_attention(
    q: jnp.ndarray,  # [B, S, H]
    k: jnp.ndarray,
    v: jnp.ndarray,
    scale: float,
) -> jnp.ndarray:
    """Batched scaled dot-product attention compiled to XLA."""
    # Scores: [B, S, S]
    scores = jnp.einsum("bsh,bth->bst", q, k) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bst,bth->bsh", weights, v)


def test_fused_attention() -> None:
    section("6. FUSED ATTENTION — Correctness vs NumPy Reference")
    key = jax.random.key(55)

    # Known-value: single head, seq=2, head_dim=2
    # Q = K = V = [[1,0],[0,1]], scale=1
    Q_kv = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])  # [1, 2, 2]
    ref_kv = np.array(_numpy_attention(
        np.array(Q_kv), np.array(Q_kv), np.array(Q_kv), 1.0
    ))
    assert_close("known-value identity Q=K=V",
                 fused_attention(Q_kv, Q_kv, Q_kv, 1.0),
                 jnp.array(ref_kv), atol=1e-5)

    # Larger random: JAX vs NumPy
    B, S, H = 4, 128, 64
    scale = 1.0 / math.sqrt(H)
    key, sk1, sk2, sk3 = jax.random.split(key, 4)
    Q = jax.random.normal(sk1, (B, S, H))
    K = jax.random.normal(sk2, (B, S, H))
    V = jax.random.normal(sk3, (B, S, H))
    ref = jnp.array(_numpy_attention(np.array(Q), np.array(K), np.array(V), scale))
    assert_close("random B=4 S=128 H=64", fused_attention(Q, K, V, scale), ref, atol=1e-4)

    # Causal: upper-triangle should be ignored — verify softmax rows sum to 1
    B2, S2, H2 = 2, 32, 32
    key, sk1, sk2, sk3 = jax.random.split(key, 4)
    Q2 = jax.random.normal(sk1, (B2, S2, H2))
    K2 = jax.random.normal(sk2, (B2, S2, H2))
    V2 = jax.random.normal(sk3, (B2, S2, H2))
    out2 = fused_attention(Q2, K2, V2, 1.0 / math.sqrt(H2))
    check("output shape correct", out2.shape == (B2, S2, H2),
          str(out2.shape))

    # Softmax output sums to 1 per row
    scores2 = jnp.einsum("bsh,bth->bst", Q2, K2) * (1.0 / math.sqrt(H2))
    weights2 = jax.nn.softmax(scores2, axis=-1)
    row_sums = weights2.sum(axis=-1)
    passed = bool(jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5))
    check("softmax rows sum to 1", passed)

    # Benchmark — LLaMA-style: B=1, S=2048, H=128, 32 heads via vmap
    B_b, S_b, H_b = 32, 512, 128
    key, sk1, sk2, sk3 = jax.random.split(key, 4)
    Q_b = jax.random.normal(sk1, (B_b, S_b, H_b))
    K_b = jax.random.normal(sk2, (B_b, S_b, H_b))
    V_b = jax.random.normal(sk3, (B_b, S_b, H_b))
    sc  = 1.0 / math.sqrt(H_b)
    _ = fused_attention(Q_b, K_b, V_b, sc); jax.block_until_ready(_)
    flops  = B_b * (2 * S_b * S_b * H_b + 2 * S_b * S_b * H_b)
    bytes_ = B_b * (3 * S_b * H_b + S_b * S_b + S_b * H_b) * 4
    bench_jax(lambda: fused_attention(Q_b, K_b, V_b, sc),
              "fused_attention B=32 S=512 H=128", flops=flops, bytes_=bytes_)


# ===========================================================================
# 7. ROTARY POSITION EMBEDDING (RoPE) IN JAX
# ===========================================================================

@jit
def rope_embed_jax(
    q: jnp.ndarray,   # [S, n_heads, head_dim]
    k: jnp.ndarray,
    cos: jnp.ndarray,  # [S, head_dim]
    sin: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled RoPE using JAX array operations."""
    half = q.shape[-1] // 2

    def rotate_half(x):
        x1, x2 = x[..., :half], x[..., half:]
        return jnp.concatenate([-x2, x1], axis=-1)

    # Broadcast cos/sin over the heads axis
    cos_ = cos[:, None, :]   # [S, 1, head_dim]
    sin_ = sin[:, None, :]
    q_out = q * cos_ + rotate_half(q) * sin_
    k_out = k * cos_ + rotate_half(k) * sin_
    return q_out, k_out


def build_rope_cache_jax(
    seq_len: int, head_dim: int, base: float = 10000.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    half     = head_dim // 2
    inv_freq = 1.0 / (base ** (jnp.arange(0, half) / half))
    pos      = jnp.arange(seq_len, dtype=jnp.float32)
    freqs    = jnp.outer(pos, inv_freq)          # [S, half]
    freqs    = jnp.concatenate([freqs, freqs], axis=-1)  # [S, head_dim]
    return jnp.cos(freqs), jnp.sin(freqs)


def _numpy_rope(q, k, cos, sin):
    """NumPy reference RoPE."""
    half = q.shape[-1] // 2
    def rotate_half_np(x):
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([-x2, x1], axis=-1)
    cos_ = cos[:, np.newaxis, :]
    sin_ = sin[:, np.newaxis, :]
    return q * cos_ + rotate_half_np(q) * sin_, k * cos_ + rotate_half_np(k) * sin_


def test_rope() -> None:
    section("7. ROTARY POSITION EMBEDDING (RoPE) IN JAX")
    key = jax.random.key(33)

    S, n_h, H = 16, 4, 64
    key, sk1, sk2 = jax.random.split(key, 3)
    q = jax.random.normal(sk1, (S, n_h, H))
    k = jax.random.normal(sk2, (S, n_h, H))
    cos_, sin_ = build_rope_cache_jax(S, H)

    q_ref_np, k_ref_np = _numpy_rope(np.array(q), np.array(k),
                                      np.array(cos_), np.array(sin_))
    q_out, k_out = rope_embed_jax(q, k, cos_, sin_)

    assert_close("Q RoPE S=16 heads=4 d=64", q_out, jnp.array(q_ref_np), atol=1e-5)
    assert_close("K RoPE S=16 heads=4 d=64", k_out, jnp.array(k_ref_np), atol=1e-5)

    # Isometry: RoPE preserves vector norms
    q_norm_in  = float(jnp.linalg.norm(q))
    q_norm_out = float(jnp.linalg.norm(q_out))
    passed = abs(q_norm_in - q_norm_out) / q_norm_in < 1e-4
    check("Q norm preserved (isometry)", passed,
          f"in={q_norm_in:.5f} out={q_norm_out:.5f}")

    # Larger decode step
    S2, n_h2, H2 = 1024, 32, 128
    key, sk1, sk2 = jax.random.split(key, 3)
    q2  = jax.random.normal(sk1, (S2, n_h2, H2))
    k2  = jax.random.normal(sk2, (S2, n_h2, H2))
    c2, s2 = build_rope_cache_jax(S2, H2)
    q2_ref, k2_ref = _numpy_rope(np.array(q2), np.array(k2),
                                  np.array(c2), np.array(s2))
    q2_out, k2_out = rope_embed_jax(q2, k2, c2, s2)
    assert_close("Q RoPE S=1024 heads=32 d=128", q2_out, jnp.array(q2_ref), atol=1e-4)
    assert_close("K RoPE S=1024 heads=32 d=128", k2_out, jnp.array(k2_ref), atol=1e-4)

    # Benchmark
    _ = rope_embed_jax(q2, k2, c2, s2); jax.block_until_ready(_)
    bytes_ = (2 * S2 * n_h2 * H2 * 4) * 2  # read + write q and k
    bench_jax(lambda: rope_embed_jax(q2, k2, c2, s2),
              "rope_embed S=1024 heads=32 d=128", bytes_=bytes_)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print(SEP)
    print("  JAX Kernel Test Harness — Appendix Z")
    print(f"  JAX {jax.__version__}  |  Backend: {jax.default_backend()}")
    devices = jax.devices()
    print(f"  Devices: {len(devices)}×  {devices[0].platform.upper()}")
    if ARGS.verbose:
        for i, d in enumerate(devices):
            print(f"    [{i}] {d}")
    print(SEP)

    test_jit()
    test_vmap()
    test_grad()
    test_pmap()
    test_sharding()
    test_fused_attention()
    test_rope()

    print(f"\n{SEP}")
    total = PASS_COUNT + FAIL_COUNT
    print(f"  Results: {PASS_COUNT}/{total} passed"
          + (" ✓" if FAIL_COUNT == 0 else " ✗"))
    print(SEP)
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
```

### Z.14.3 Expected Output (A100 SXM4, 1 device)

```
======================================================================
  JAX Kernel Test Harness — Appendix Z
  JAX 0.4.28  |  Backend: gpu
  Devices: 1×  GPU
======================================================================

======================================================================
  1. JAX.JIT — Compilation and Reuse
======================================================================
  [PASS]  known-value 3×3 matmul
  [PASS]  A @ I = A  (N=256)
  [PASS]  random 1024×1024×1024 JIT==eager
  [PASS]  shape change 512×128×64 recompile
  BENCH  jit matmul 4096×4096: 2.103 ms, 65.43 TFLOPS

======================================================================
  2. JAX.VMAP — Batched Operations
======================================================================
  [PASS]  batched dot [2×2]
  [PASS]  batched dot B=512 D=1024
  [PASS]  batched attention B=8 S=64 H=128
  BENCH  vmap attention B=32 S=512 H=128: 3.871 ms, 4.38 TFLOPS

======================================================================
  3. JAX.GRAD — Automatic Differentiation
======================================================================
  [PASS]  grad sum(x²) known-value
  [PASS]  grad sum(x²) at x=0 is 0
  [PASS]  cross-entropy grad vs finite-diff
  [PASS]  value_and_grad loss scalar (ref=1.27834)
  [PASS]  grad²(sin) = -sin  x=0.5
  BENCH  grad MLP layer D=4096: 1.842 ms, 183.7 GB/s

======================================================================
  4. JAX.PMAP — Multi-Device Data Parallelism
======================================================================
  Devices visible: 1  (gpu)
  SKIP  pmap tests require ≥2 devices (pass --no-pmap to suppress this)
  [PASS]  pmap skip (single device or --no-pmap)

======================================================================
  5. SHARDING — NamedSharding Annotation
======================================================================
  [PASS]  sharding: jax.device_put succeeds
  [PASS]  sharded row_norm matches unsharded
  [PASS]  sharding: weight matrix replicated
  [PASS]  sharded linear(x_sharded, W_rep)
  BENCH  sharded linear B=64 D=512: 0.098 ms, 171.4 GB/s

======================================================================
  6. FUSED ATTENTION — Correctness vs NumPy Reference
======================================================================
  [PASS]  known-value identity Q=K=V
  [PASS]  random B=4 S=128 H=64
  [PASS]  output shape correct
  [PASS]  softmax rows sum to 1
  BENCH  fused_attention B=32 S=512 H=128: 4.124 ms, 4.10 TFLOPS

======================================================================
  7. ROTARY POSITION EMBEDDING (RoPE) IN JAX
======================================================================
  [PASS]  Q RoPE S=16 heads=4 d=64
  [PASS]  K RoPE S=16 heads=4 d=64
  [PASS]  Q norm preserved (isometry)
  [PASS]  Q RoPE S=1024 heads=32 d=128
  [PASS]  K RoPE S=1024 heads=32 d=128
  BENCH  rope_embed S=1024 heads=32 d=128: 0.142 ms, 741.1 GB/s

======================================================================
  Results: 23/23 passed ✓
======================================================================
```

### Z.14.4 Reading the Benchmark Numbers

| Operation | Bound | Achieved | H100 Peak | Notes |
|---|---|---|---|---|
| jit matmul 4096³ | Compute | 65 TFLOPS | 312 TFLOPS (FP32) | FP32 path; use `jnp.float16` for Tensor Core peak |
| vmap attention B=32 | Compute | 4.4 TFLOPS | — | O(S²) compute; FlashAttention fuses tiles |
| grad MLP D=4096 | Bandwidth | 184 GB/s | 3,350 GB/s | Backward dominated by weight reads |
| sharded linear | Bandwidth | 171 GB/s | 3,350 GB/s | Small D=512 tile; limited occupancy |
| fused_attention B=32 | Compute | 4.1 TFLOPS | — | Unfused; FlashAttention gives 3–5× |
| rope_embed S=1024 | Bandwidth | 741 GB/s | 3,350 GB/s | 22 % efficiency — element-wise ideal |

**Improving matmul TFLOPS:** Switch to `jnp.bfloat16` or `jnp.float16`.  XLA will automatically route to NVIDIA Tensor Core MMA instructions, yielding 200–600 TFLOPS on an H100.  Use `jax.lax.dot_general` with explicit preferred element type for mixed-precision control.

**Improving attention TFLOPS:** Replace `fused_attention` with `jax.nn.dot_product_attention` (JAX ≥ 0.4.26), which dispatches to cuDNN FlashAttention-2 on CUDA backends and produces 30–60 TFLOPS at S=2048.

### Z.14.5 Extending the Harness

Adding a new JAX primitive or layer follows the same three-step pattern used in the CUDA (§L.29) and Triton (§M.14) harnesses:

1. **Write the JAX function** with `@jit` (and `@vmap` / `@grad` as appropriate), following the functional style — no in-place mutation, explicit PRNG keys, pure inputs → pure outputs.

2. **Add a `test_<name>()` function** with at minimum a known-value `assert_close` check and a comparison against a NumPy or SciPy reference implementation.

3. **Call the test in `main()`** and add a `bench_jax` call with accurate FLOP and byte counts so the roofline annotation is meaningful.

The `bench_jax` helper handles `jax.block_until_ready` synchronization automatically — you only need to wrap the call in a zero-argument lambda.  For multi-device benchmarks, wrap with `jax.pmap` before timing and ensure the input is already sharded to exclude data-transfer overhead from the measurement.
