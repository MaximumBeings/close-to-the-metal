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

The consequence for inference: a JAX model that has been JIT-compiled has **no Python overhead per forward pass**. The entire computation graph is a single XLA program dispatched to hardware. This is equivalent in spirit to CUDA Graphs (Appendix 8.5) but applies to the entire model, not just one captured stream.

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
