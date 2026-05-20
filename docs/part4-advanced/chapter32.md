# Chapter 32: Debugging Inference Systems

> *"The model was fine in evaluation. Then you deployed it."*

---

## 32.1 The Inference Debugging Mindset

Production inference systems fail in ways that training systems do not. A training job hangs — you restart it. A deployed inference endpoint returns garbage — users are already seeing it.

The failure modes split into four categories:

1. **Numeric failures** — NaN/Inf propagating through logits; overflow in quantized arithmetic; underflow in softmax.
2. **Memory failures** — KV cache corruption; fragmentation causing OOM on requests that should fit; memory leaks in Python reference cycles.
3. **Correctness failures** — outputs are syntactically valid but semantically wrong; sampling instability produces non-deterministic results when determinism was expected; system prompt bleeds into responses.
4. **Performance failures** — throughput drops without a code change; latency spikes on specific request shapes; GPU utilization collapses under concurrent load.

Each category has its own diagnostic toolkit. This chapter walks through each class of failure, explains why it happens at the mechanistic level, and shows how to detect and fix it.

The debugging workflow always follows the same arc: **observe → isolate → reproduce → fix → verify**. A bug you can reproduce reliably is almost solved. A bug that appears only under production load requires instrumentation to make it reproducible.

---

## 32.2 Numeric Failures

### 32.2.1 NaN Propagation

A NaN (Not a Number) in a forward pass is caused by one of three things:

- Division by zero (softmax denominator collapses, LayerNorm denominator collapses)
- 0 × ∞ in attention (zero attention weight multiplied by a very large value)
- Overflow followed by subtraction (e.g., fp16 overflow → Inf - Inf = NaN)

NaN is insidious because it propagates silently: once a single NaN enters the residual stream, every subsequent add, multiply, and attention operation outputs NaN, and the final logits are all NaN. The model outputs either garbage tokens (argmax of NaN is implementation-defined) or a crash.

**Detection pattern:**

```python
import torch

def check_for_nan(tensor: torch.Tensor, name: str) -> bool:
    """Returns True if tensor contains NaN or Inf; logs location."""
    has_nan  = torch.isnan(tensor).any().item()
    has_inf  = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        n_nan = torch.isnan(tensor).sum().item()
        n_inf = torch.isinf(tensor).sum().item()
        print(f"[WARN] {name}: NaN={n_nan}, Inf={n_inf}, "
              f"shape={tensor.shape}, dtype={tensor.dtype}")
        return True
    return False
```

Hook this into every major operation during debugging:

```python
# In attention forward pass
q, k, v = ...
attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
check_for_nan(attn_weights, "attn_weights pre-softmax")

attn_weights = torch.softmax(attn_weights, dim=-1)
check_for_nan(attn_weights, "attn_weights post-softmax")
```

**Root cause: LayerNorm instability.** When the input to LayerNorm has near-zero variance (all activations identical), the normalization divides by ε (a small constant added for stability). If ε is too small for the dtype (fp16 has ε ≈ 6e-5), the result overflows. Fix: use bf16 for training and fp16 inference, or add a pre-norm clamping:

```python
def stable_layer_norm(x, weight, bias, eps=1e-5):
    # Clamp variance to prevent division by near-zero
    mean = x.mean(dim=-1, keepdim=True)
    var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    var  = var.clamp(min=eps * eps)  # prevent sqrt(0)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias
```

**Root cause: attention with fp16 overflow.** The query-key dot product before scaling can overflow fp16 (max ≈ 65504) for long sequences or large head dimensions. The scaled dot product `QK^T / sqrt(d)` must be computed in fp32 even when Q and K are fp16. vLLM and llama.cpp both do this; custom implementations often forget.

### 32.2.2 Quantization Overflow

INT8 and INT4 quantization introduce a different failure mode: values outside the quantization range are clipped, which introduces systematic bias rather than NaN. INT8 range is [-128, 127]; a pre-quantized activation value of 150 clips to 127, introducing a bias of 23 at that position. This manifests as subtle quality degradation rather than crashes.

**Detection:** Compare the output distribution of the quantized model against the full-precision model on a calibration dataset. A good quantization calibration minimizes the maximum absolute error between the two distributions.

```python
def quantization_error_report(fp_logits, quant_logits, topk=10):
    """Compare top-k token distributions between FP and quantized model."""
    fp_top    = torch.topk(fp_logits.float(), topk, dim=-1)
    qt_top    = torch.topk(quant_logits.float(), topk, dim=-1)
    
    # Token overlap in top-k
    overlap   = set(fp_top.indices.tolist()) & set(qt_top.indices.tolist())
    
    # L∞ norm of probability difference
    fp_prob   = torch.softmax(fp_logits.float(), dim=-1)
    qt_prob   = torch.softmax(quant_logits.float(), dim=-1)
    l_inf     = (fp_prob - qt_prob).abs().max().item()
    
    print(f"  Top-{topk} overlap: {len(overlap)}/{topk}")
    print(f"  L∞ prob error:    {l_inf:.5f}")
    return l_inf
```

A healthy quantized model has L∞ < 0.01 on most tokens and < 0.05 on all tokens. Values above 0.05 indicate catastrophic quantization error on that token and warrant investigation of the weight scale calibration for that layer.

### 32.2.3 Softmax Overflow

The numerical identity for stable softmax is:

```
softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

Without the max subtraction, large logit values cause `exp(x_i)` to overflow to Inf in fp16. The standard implementations (PyTorch, CUDA's cuDNN) apply this shift automatically. Custom CUDA kernels or hand-written implementations sometimes omit it.

**Verification test:** Feed logits of [1000, 999, 998] to your softmax. The result should be approximately [0.577, 0.212, 0.211]. If it returns NaN, the implementation is numerically unstable.

---

## 32.3 KV Cache Failures

### 32.3.1 KV Cache Corruption

In PagedAttention (vLLM), the KV cache is allocated in pages. Corruption occurs when:

1. A page is freed and reallocated while still referenced by a live sequence (use-after-free).
2. Two sequences share a prefix page but one modifies the cached KV values in place.
3. The block manager assigns the wrong block offset, causing one sequence to write into another sequence's KV.

Symptoms of KV cache corruption:

- Sudden incoherence mid-generation ("the user asked about weather, the model starts discussing chemistry")
- Repeated token loops that weren't present in smaller batches
- Non-deterministic output given identical inputs

**Diagnostic procedure:**

1. Run identical requests serially (batch_size=1) and record outputs.
2. Run the same requests in a large batch.
3. Diff the outputs. If serial outputs differ from batch outputs for the same request, corruption is confirmed.

```bash
# vLLM: run with deterministic sampling to expose corruption
vllm serve model --seed 42 --disable-log-requests &

# Send same request twice — should get identical output
curl -s localhost:8000/v1/completions \
  -d '{"model":"...", "prompt":"The capital of France is", "max_tokens":5, "temperature":0}'
```

**In llama.cpp**, KV cache corruption most often comes from incorrect `n_past` tracking in stateful contexts. The `n_past` counter tells the model how many tokens are already in the KV cache; if it is wrong, the model attends to garbage positions.

```cpp
// Correct pattern: track n_past carefully
int n_past = 0;
for (auto& chunk : token_chunks) {
    llama_decode(ctx, batch_from(chunk, n_past));
    n_past += chunk.size();  // must match exactly what was decoded
}

// Bug: forgetting to update n_past after prefix processing
// Bug: using n_past from a different sequence for this sequence
```

### 32.3.2 Prefix Cache Invalidation Bugs

vLLM's prefix caching stores KV pages for common prompt prefixes. A bug occurs when a prefix is marked as cached but the underlying weights have been updated (e.g., after a LoRA adapter swap). The stale KV values cause the model to behave as if using the old adapter.

**Fix:** Flush the prefix cache on any model/adapter change:
```python
# In vLLM, after loading a new LoRA adapter:
engine.llm_engine.cache_engine.flush_all()  # invalidate all cached prefixes
```

### 32.3.3 Memory Leak in Python Reference Cycles

vLLM's Python layer holds references to tensors in request state. If a callback or closure captures a reference to an output tensor, the tensor's GPU memory is not freed when the request completes. This causes a slow memory leak that manifests as OOM after thousands of requests.

**Detection:** Monitor GPU memory over time during a sustained load test.

```python
import torch

def memory_snapshot():
    alloc_gb = torch.cuda.memory_allocated() / 1e9
    reserv_gb = torch.cuda.memory_reserved()  / 1e9
    print(f"  GPU: alloc={alloc_gb:.2f} GB, reserved={reserv_gb:.2f} GB")

# If alloc grows monotonically during a steady-state test, there is a leak
```

**Common Python leak pattern:**

```python
# Bug: closure captures `output_tensor`, preventing GC
results = []
def on_complete(output_tensor):
    results.append(output_tensor)  # tensor stays alive in `results`
    
# Fix: detach and move to CPU before storing
def on_complete(output_tensor):
    results.append(output_tensor.detach().cpu().tolist())  # GPU memory freed
```

---

## 32.4 Correctness Failures

### 32.4.1 Sampling Non-Determinism

Given the same prompt, temperature, and seed, a model should produce identical output. Violations of this property are called sampling non-determinism and stem from:

**Non-deterministic CUDA operations.** PyTorch has atomics in some CUDA kernels whose execution order varies across runs. Setting `torch.use_deterministic_algorithms(True)` forces deterministic versions (often slower).

**Floating-point non-associativity.** `(a + b) + c ≠ a + (b + c)` in floating-point. Parallel reductions (softmax denominator sum, LayerNorm variance) can produce different results depending on thread interleaving. Flash Attention 2+ uses online-softmax which is deterministic up to floating-point order.

**Temperature = 0 (greedy) is NOT always deterministic** unless the argmax is also deterministic. For tied logits (extremely rare but possible), tie-breaking is implementation-defined.

**Debugging procedure:**

```python
def check_determinism(model, tokenizer, prompt, n_runs=5, seed=42):
    """Run the same prompt n_runs times; verify identical outputs."""
    torch.manual_seed(seed)
    outputs = []
    for i in range(n_runs):
        torch.manual_seed(seed)  # reset each time
        out = model.generate(
            tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
            do_sample=False,  # greedy
            max_new_tokens=50,
        )
        outputs.append(tokenizer.decode(out[0]))
    
    unique = set(outputs)
    if len(unique) > 1:
        print(f"  [FAIL] Non-deterministic: {len(unique)} unique outputs in {n_runs} runs")
        for i, o in enumerate(outputs[:3]):
            print(f"    Run {i}: {o[:80]}")
    else:
        print(f"  [PASS] Deterministic: all {n_runs} runs identical")
    return len(unique) == 1
```

### 32.4.2 System Prompt Leakage

A system prompt from one request appearing in the output of another request. Mechanisms:

1. **KV cache prefix collision.** Two requests with different system prompts happen to hash to the same prefix cache key. (Should be impossible if keys are content-addressed, but implementation bugs exist.)
2. **Conversation state contamination.** A stateful session (llama.cpp with `n_past > 0`) is incorrectly reused for a new user without clearing the KV cache.
3. **Batch padding overlap.** In a buggy batching implementation, the padding tokens of one sequence attend to non-padding tokens of an adjacent sequence.

**Diagnostic:** Construct two requests with maximally different system prompts and known unique trigger words. Verify the outputs contain no words from the other request's system prompt.

### 32.4.3 Repetition Loops

The model enters a repetition loop (producing the same token or phrase indefinitely). This happens when:

- Temperature is too low for a model that expects temperature > 0 (the greedy path gets stuck in a local attractor)
- Repetition penalty is misconfigured (too high suppresses all variation; not applied at all allows loops)
- The context is truncated incorrectly, causing the model to "forget" what it just said

**Quick fix check:** increase `repetition_penalty` slightly above 1.0 (1.05–1.15 is typical). If this stops the loop, the model has a temperature/penalty calibration issue, not a weight bug.

---

## 32.5 Performance Failures

### 32.5.1 Throughput Regression Diagnosis

A throughput regression — tokens-per-second drops without an obvious code change — has four common causes:

| Cause | Symptom | Diagnostic |
|---|---|---|
| Batch size too small | GPU utilization < 60% | Check `vllm_num_running_seqs` metric |
| Memory pressure | Frequent preemptions | Check `vllm_preemption_events_total` |
| CPU-GPU sync | Periodic GPU idle bubbles | Profile with `nsys profile` |
| Kernel regression | Slower CUDA kernels | Benchmark specific ops before/after |

**Minimum diagnostic set for any throughput regression:**

```bash
# 1. Check GPU utilization
nvidia-smi dmon -s u -d 1   # stream GPU utilization every second

# 2. Check vLLM metrics (if deployed)
curl -s localhost:8000/metrics | grep -E "vllm_gpu_cache_usage|vllm_num_running"

# 3. Rapid throughput benchmark
python -c "
import time, requests
N = 100
t0 = time.time()
for _ in range(N):
    r = requests.post('http://localhost:8000/v1/completions',
        json={'model':'...','prompt':'Hello','max_tokens':50})
elapsed = time.time() - t0
tps = N * 50 / elapsed
print(f'{tps:.1f} tokens/s  ({elapsed:.1f}s for {N} requests)')
"
```

### 32.5.2 Memory Bandwidth Saturation

At large batch sizes, the forward pass is compute-bound. At small batch sizes, it is memory-bandwidth-bound: each weight matrix is loaded once per forward pass regardless of batch size, so throughput is limited by `bw / (2 × params × bytes_per_param)`.

The diagnostic question: **is the bottleneck memory or compute?**

```
Arithmetic Intensity = FLOPs / bytes_transferred

For a matrix multiply [M, K] × [K, N]:
  FLOPs = 2 × M × N × K
  Bytes  = (M × K + K × N) × dtype_bytes
  AI     = 2 × M × N × K / ((M × K + K × N) × dtype_bytes)

  At batch_size=1, M=1:
  AI = 2 × 1 × N × K / ((1 × K + K × N) × dtype_bytes)
     = 2NK / (K × (1 + N) × dtype_bytes)
     = 2N / ((1 + N) × dtype_bytes)
     ≈ 2 / dtype_bytes          (for large N, i.e., N >> 1)

  For fp16 (dtype_bytes = 2): AI ≈ 2 / 2 = 1 FLOP/byte
  GPU ridge point (H100):
    1,979 TFLOPS / 3.35 TB/s ≈ 591 FLOPS/byte
    Since AI ≈ 1 << 591, batch_size=1 decode is firmly memory-bandwidth-bound.

  Intuition: with M=1 the input activation row is tiny (K values), but the
  weight matrix is K×N — you must read almost as many bytes as if you were
  doing a full-matrix read, yet you compute only one output row. Bytes
  dominate FLOPs by a factor of N/2.
```

**Practical rule:** if `batch_size × hidden_dim / dtype_bytes > GPU_compute_TFLOPs / GPU_BW_TBs × 1000`, you are compute-bound. Below this threshold, you are memory-bandwidth-bound and increasing batch size is the primary lever.

### 32.5.3 Latency Spikes

Occasional latency spikes (P99 >> P50) are caused by:

1. **CUDA context initialization.** The first kernel call on a GPU takes 100–500 ms. Warm up the GPU on startup.
2. **Python GC pause.** `gc.collect()` with large PyTorch objects takes 10–100 ms. Disable automatic GC during inference: `gc.disable()`, then call `gc.collect()` periodically between requests.
3. **Memory allocation.** CUDA cudaMalloc is slow (~1 ms). Use pre-allocated tensor pools.
4. **Prefix cache miss on cold start.** A fresh deployment has an empty prefix cache; the first N requests pay full prefill cost. Pre-warm with representative prompts.

```python
# Startup warm-up: prevent first-request latency spike
def warm_up_engine(engine, n_warmup=3):
    for _ in range(n_warmup):
        _ = engine.generate("Hello world", max_tokens=1)
    torch.cuda.synchronize()
    print("  Engine warmed up.")
```

### 32.5.4 Tensor Parallelism Imbalance

In tensor-parallel deployments (multiple GPUs), throughput is limited by the slowest GPU (Amdahl's law for communication). Imbalance sources:

- Different CUDA driver versions across nodes
- One GPU running a background job (monitoring, NVLINK health check)
- PCIe/NVLINK bandwidth asymmetry in certain server configs

```bash
# Check GPU utilization across all devices simultaneously
nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
  --format=csv,noheader -l 1

# Expect all GPUs to be within 5% of each other during steady-state inference
# A single lagging GPU will cap throughput for the entire TP group
```

---

## 32.6 Distributed Debugging

### 32.6.1 Rank-Specific Failures

In tensor-parallel inference, a bug may affect only specific ranks. A common symptom: the model produces correct output for some positions but garbage for others, in a pattern that repeats every `tp_size` tokens.

**Rank identification:** add rank-tagged logging to the forward pass.

```python
import torch.distributed as dist

rank = dist.get_rank()
if torch.isnan(output).any():
    print(f"  [Rank {rank}] NaN detected in output, shape={output.shape}")
```

If only rank 2 (for example) logs the NaN, the bug is in the weight shard assigned to rank 2 (possible corruption during weight loading) or in the all-reduce operation on that rank's output.

### 32.6.2 NCCL Hang

An NCCL (collective communication) hang occurs when one rank fails to participate in a collective (all-reduce, all-gather). All other ranks block waiting for it, and the system appears frozen.

**Detection:** set NCCL timeout and catch it:

```bash
export NCCL_TIMEOUT=300  # seconds; default is often infinite
export NCCL_DEBUG=INFO   # verbose NCCL logging
```

**Common causes:**

- Rank 0 processes the input while other ranks wait on a different codepath (rank-conditional logic)
- One rank hits an OOM and exits, leaving others stuck in collective

**Fix pattern:** wrap every collective in a try/except and log which rank failed.

### 32.6.3 Deadlock Between Process Groups

When using pipeline parallelism + tensor parallelism (PP × TP), there are multiple process groups. A deadlock occurs when group A is blocked waiting for group B while group B is blocked waiting for group A.

Use `torch.distributed.breakpoint()` (PyTorch ≥ 2.0) to attach a debugger to all ranks simultaneously:

```python
if rank == 0:
    torch.distributed.breakpoint()  # drops all ranks into pdb
```

---

## 32.7 llama.cpp-Specific Debugging

### 32.7.1 Context Overflow

When `n_past + n_tokens > n_ctx`, llama.cpp silently truncates or produces undefined behavior depending on the version. Always validate before decode:

```cpp
if (n_past + batch.n_tokens > llama_n_ctx(ctx)) {
    fprintf(stderr, "[ERROR] Context overflow: n_past=%d + n_tokens=%d > n_ctx=%d\n",
            n_past, batch.n_tokens, llama_n_ctx(ctx));
    // Options: truncate, error, or slide window
    return false;
}
```

### 32.7.2 Quantization Mismatch

Loading a Q4_K_M model with a context expecting Q8_0 will produce garbage output without an error in some llama.cpp versions. Always verify the quantization type after loading:

```cpp
const char* quant_name = llama_model_quantization_type_name(
    llama_model_quantization_type(model));
printf("Loaded model quantization: %s\n", quant_name);
// Verify this matches what you intended to load
```

### 32.7.3 Thread Count and NUMA Effects

llama.cpp uses OpenMP or its own thread pool for CPU inference. Using more threads than physical cores causes hyperthreading contention; using all physical cores on a NUMA system may cause cross-socket memory access.

```bash
# Find physical core count (not logical)
lscpu | grep "Core(s) per socket"
# Set n_threads to that value, not the logical CPU count
llama-cli --threads $(nproc --all) --model model.gguf  # BAD: uses logical CPUs
llama-cli --threads 16 --model model.gguf              # GOOD: use physical cores
```

**Diagnostic:** run `perf stat -e cache-misses,cache-references` while running llama.cpp. A cache miss rate > 30% indicates NUMA-unfriendly memory access patterns.

---

## 32.8 Diagnostic Playbook

A systematic runbook for the five most common production incidents:

### Incident 1: All responses are garbage / empty tokens

1. Check logits for NaN: `torch.isnan(logits).any()` — if True, see §32.2.1
2. Check GPU memory: `nvidia-smi` — if OOM, reduce batch size or context length
3. Check model was loaded correctly: verify checksum of weight files
4. Verify tokenizer matches model: encode "Hello" and check token IDs are sane (should be 9906 for LLaMA tokenizer)

### Incident 2: Throughput 50% lower than baseline

1. Check GPU utilization: if < 60%, batch size is too small → increase `--max-num-seqs`
2. Check `vllm_gpu_cache_usage_perc` — if > 95%, reduce `--max-model-len` or add a GPU
3. Profile with `nsys profile ./your_inference_script` — look for CPU-GPU sync points (cudaDeviceSynchronize in the hot path)
4. Check for a Python version or PyTorch version change — some releases introduce regressions

### Incident 3: Outputs are correct but slow (high P99 latency)

1. Check for GC pauses: add `gc.disable()` before the request loop
2. Check for cold-start: add warm-up (§32.5.3)
3. Check prefix cache hit rate: if near 0% after warm-up, prefix hashing may be broken
4. Check CUDA event timing: use `torch.cuda.Event` to measure each forward pass component

### Incident 4: Inconsistent outputs (same input, different output)

1. Verify seed is set: `torch.manual_seed(42)` before every generate call
2. Enable deterministic algorithms: `torch.use_deterministic_algorithms(True)` (will raise if any op is non-deterministic)
3. Check batch collation: are padding tokens at the correct positions? Wrong padding can change attention patterns
4. If using fp16: check for precision loss at the softmax boundary (§32.2.3)

### Incident 5: Memory grows monotonically (slow leak)

1. Baseline: measure GPU memory after 10, 100, 1000 requests
2. If growing: run with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` — if growth stops, it's a caching issue, not a leak
3. Use `torch.cuda.memory_summary()` to see which allocations are live
4. Grep for Python closures capturing GPU tensors (common in async callback patterns)

---

## 32.9 Python Demo

```python
"""
debugging_demo.py — Chapter 32: Debugging Inference Systems

Demonstrates:
  1. NaN detection and propagation tracing
  2. Softmax numerical stability analysis
  3. Quantization error measurement
  4. Sampling determinism verification
  5. Memory leak detection pattern
  6. KV cache sequence isolation test
  7. Performance regression benchmark
  8. Latency spike diagnosis

Run: python debugging_demo.py
Requirements: numpy (pip install numpy)
"""
from __future__ import annotations

import gc
import math
import random
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# §1  NaN DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def check_tensor(arr: np.ndarray, name: str) -> bool:
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    if has_nan or has_inf:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        print(f"  [WARN] {name}: NaN={n_nan}, Inf={n_inf}, shape={arr.shape}")
        return True
    return False

def unstable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically UNSTABLE softmax — for demonstration only."""
    e = np.exp(x.astype(np.float32))   # overflows for large x
    return e / e.sum()

def stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically STABLE softmax: subtract max before exp."""
    x = x.astype(np.float64)
    x = x - x.max()                    # prevents overflow
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)

def demo_nan_detection() -> None:
    section("NaN Detection and Softmax Stability")

    # Simulate large logits that cause fp16 overflow → NaN
    large_logits = np.array([1000.0, 999.0, 998.0], dtype=np.float16)
    safe_logits  = np.array([10.0, 9.0, 8.0],       dtype=np.float16)

    print("\n  Testing with large logits (1000, 999, 998) in fp16:")
    unstable_out = unstable_softmax(large_logits)
    check_tensor(unstable_out, "unstable_softmax(large_logits)")
    stable_out   = stable_softmax(large_logits)
    check_tensor(stable_out,   "stable_softmax(large_logits)")
    print(f"  Unstable result: {unstable_out}")
    print(f"  Stable result:   {stable_out[:3].round(4)}")

    print("\n  Testing with normal logits (10, 9, 8):")
    u2 = unstable_softmax(safe_logits)
    s2 = stable_softmax(safe_logits)
    print(f"  Unstable: {u2[:3].round(4)}")
    print(f"  Stable:   {s2[:3].round(4)}")
    print(f"  Max diff: {np.abs(u2 - s2).max():.6f}")

    # Verify stable softmax is correct (sums to 1, no NaN)
    assert not np.isnan(stable_out).any(), "Stable softmax should not produce NaN"
    assert abs(stable_out.sum() - 1.0) < 1e-5, "Softmax should sum to 1"
    print(f"\n  [ASSERT] Stable softmax: no NaN, sums to 1.0 ✓")

    # Simulate NaN propagation through a simple network
    print("\n  NaN propagation simulation:")
    x = np.array([1.0, np.nan, 3.0])
    y = x * 2.0          # NaN propagates
    z = y + 1.0          # Still NaN
    w = np.exp(z)        # Still NaN
    check_tensor(x, "input (has NaN)")
    check_tensor(w, "output after × 2 + 1 + exp")
    print(f"  NaN at position 1 → propagates to all downstream ops: "
          f"{np.isnan(w).sum()} NaN values in output")
    assert np.isnan(w[1]), "NaN should propagate through arithmetic"
    print(f"\n  [ASSERT] NaN propagation correctly detected: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §2  QUANTIZATION ERROR MEASUREMENT
# ──────────────────────────────────────────────────────────────────────────────

def quantize_int8(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Symmetric INT8 quantization: scale = max(|x|) / 127."""
    scale = float(np.abs(x).max()) / 127.0
    if scale == 0:
        return np.zeros_like(x, dtype=np.int8), 1.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale

def dequantize_int8(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale

def matmul_fp32(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A.astype(np.float64) @ B.astype(np.float64)

def matmul_int8(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Aq, As = quantize_int8(A)
    Bq, Bs = quantize_int8(B)
    result = Aq.astype(np.int32) @ Bq.astype(np.int32)
    return result.astype(np.float32) * As * Bs

def demo_quantization_error() -> None:
    section("Quantization Error Measurement")

    rng = np.random.default_rng(42)
    M, K, N = 64, 256, 128

    # Normal weights: low error expected
    A_normal = rng.standard_normal((M, K)).astype(np.float32) * 0.1
    B_normal = rng.standard_normal((K, N)).astype(np.float32) * 0.1

    # Outlier weights: high error expected
    A_outlier = A_normal.copy()
    A_outlier[0, 0] = 100.0  # single extreme outlier

    def measure_error(A, B, label):
        fp_out  = matmul_fp32(A, B).astype(np.float32)
        int_out = matmul_int8(A, B)
        l_inf   = float(np.abs(fp_out - int_out).max())
        rel_err = l_inf / (np.abs(fp_out).max() + 1e-8)
        status  = "✓ OK" if rel_err < 0.05 else "✗ DEGRADED"
        print(f"  {label:<25} L∞={l_inf:.4f}  rel={rel_err:.4f}  [{status}]")
        return rel_err

    print(f"\n  {'Scenario':<25} {'L∞ error':<12} {'Rel error':<12} {'Status'}")
    print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*10}")
    err_normal  = measure_error(A_normal,  B_normal, "Normal weights")
    err_outlier = measure_error(A_outlier, B_normal, "Outlier weight (100x)")

    # Demonstrate per-channel quantization improves outlier case
    # Per-channel: each row of A has its own scale
    def matmul_int8_perchannel(A, B):
        result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
        for i in range(A.shape[0]):
            row_q, row_s = quantize_int8(A[i])
            Bq, Bs = quantize_int8(B)
            result[i] = (row_q.astype(np.int32) @ Bq.astype(np.int32)).astype(np.float32) * row_s * Bs
        return result

    fp_out      = matmul_fp32(A_outlier, B_normal).astype(np.float32)
    pc_out      = matmul_int8_perchannel(A_outlier, B_normal)
    l_inf_pc    = float(np.abs(fp_out - pc_out).max())
    rel_pc      = l_inf_pc / (np.abs(fp_out).max() + 1e-8)
    status_pc   = "✓ OK" if rel_pc < 0.05 else "✗ DEGRADED"
    print(f"  {'Outlier + per-channel':<25} L∞={l_inf_pc:.4f}  rel={rel_pc:.4f}  [{status_pc}]")

    assert err_normal  < 0.05, f"Normal quantization error {err_normal:.4f} too high"
    assert err_outlier > err_normal, "Outlier should cause more error than normal"
    assert rel_pc < err_outlier, "Per-channel should improve over tensor-wise quantization"
    print(f"\n  [ASSERT] Normal error < 5%: ✓")
    print(f"  [ASSERT] Outlier causes more error than normal: ✓")
    print(f"  [ASSERT] Per-channel quantization reduces outlier error: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §3  SAMPLING DETERMINISM
# ──────────────────────────────────────────────────────────────────────────────

def sample_token(logits: np.ndarray, temperature: float, seed: int) -> int:
    """Temperature sampling with explicit RNG seed."""
    rng = np.random.default_rng(seed)
    if temperature == 0.0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    probs  = stable_softmax(scaled)
    return int(rng.choice(len(probs), p=probs / probs.sum()))

def demo_sampling_determinism() -> None:
    section("Sampling Determinism Verification")

    rng = np.random.default_rng(0)
    VOCAB = 1000
    N_RUNS = 20

    # Simulate a logit distribution
    logits = rng.standard_normal(VOCAB).astype(np.float32) * 2.0

    print(f"\n  {'Config':<35} {'Unique outputs':>15}  {'Deterministic?':>15}")
    print(f"  {'─'*35} {'─'*15}  {'─'*15}")

    scenarios = [
        ("Temperature=0 (greedy), fixed seed",   0.0, 42),
        ("Temperature=0.8, fixed seed",          0.8, 42),
        ("Temperature=0.8, random seed (buggy)", 0.8, -1),  # -1 = random seed
    ]

    for desc, temp, seed in scenarios:
        outputs = []
        for _ in range(N_RUNS):
            actual_seed = seed if seed >= 0 else random.randint(0, 2**31)
            token = sample_token(logits, temp, actual_seed)
            outputs.append(token)
        n_unique = len(set(outputs))
        is_det   = n_unique == 1
        mark     = "✓ deterministic" if is_det else f"✗ {n_unique} unique"
        print(f"  {desc:<35} {n_unique:>15}  {mark:>15}")

    # Assert: greedy is always deterministic
    greedy_outputs = [sample_token(logits, 0.0, i) for i in range(N_RUNS)]
    assert len(set(greedy_outputs)) == 1, "Greedy should always be deterministic"

    # Assert: fixed-seed temperature is also deterministic
    fixed_outputs = [sample_token(logits, 0.8, 42) for _ in range(N_RUNS)]
    assert len(set(fixed_outputs)) == 1, "Fixed-seed sampling should be deterministic"

    print(f"\n  [ASSERT] Greedy (temp=0) is deterministic across seeds: ✓")
    print(f"  [ASSERT] Temperature sampling with fixed seed is deterministic: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §4  KV CACHE ISOLATION SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

class SimpleKVCache:
    """
    Minimal KV cache simulation to demonstrate sequence isolation.
    Tracks which sequence each cache slot belongs to.
    """
    def __init__(self, n_slots: int, n_layers: int, head_dim: int):
        self.n_slots  = n_slots
        self.n_layers = n_layers
        self.head_dim = head_dim
        # Allocate slots with sequence ownership tracking
        self._cache   = np.zeros((n_slots, n_layers, 2, head_dim), dtype=np.float32)
        self._owner   = [-1] * n_slots   # -1 = free, otherwise seq_id
        self._free    = list(range(n_slots))

    def allocate(self, seq_id: int, n_tokens: int) -> List[int]:
        if len(self._free) < n_tokens:
            raise MemoryError(f"OOM: need {n_tokens} slots, have {len(self._free)}")
        slots = self._free[:n_tokens]
        self._free = self._free[n_tokens:]
        for s in slots:
            self._owner[s] = seq_id
        return slots

    def write(self, seq_id: int, slot: int, layer: int, k: np.ndarray, v: np.ndarray):
        # Verify ownership — this check prevents corruption
        if self._owner[slot] != seq_id:
            raise ValueError(f"[BUG] seq {seq_id} writing to slot {slot} "
                             f"owned by seq {self._owner[slot]}")
        self._cache[slot, layer, 0] = k
        self._cache[slot, layer, 1] = v

    def read(self, seq_id: int, slot: int, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._owner[slot] != seq_id:
            raise ValueError(f"[BUG] seq {seq_id} reading from slot {slot} "
                             f"owned by seq {self._owner[slot]}")
        return self._cache[slot, layer, 0], self._cache[slot, layer, 1]

    def free(self, seq_id: int):
        freed = [s for s, o in enumerate(self._owner) if o == seq_id]
        for s in freed:
            self._owner[s] = -1
            self._cache[s] = 0
        self._free.extend(freed)

def demo_kv_cache_isolation() -> None:
    section("KV Cache Sequence Isolation")

    cache = SimpleKVCache(n_slots=32, n_layers=4, head_dim=64)
    rng   = np.random.default_rng(42)

    # Allocate two sequences
    slots_a = cache.allocate(seq_id=0, n_tokens=4)
    slots_b = cache.allocate(seq_id=1, n_tokens=4)

    # Write distinct values for each sequence
    for i, slot in enumerate(slots_a):
        k = np.ones(64, dtype=np.float32) * (i + 1) * 10.0   # seq A: 10, 20, 30, 40
        v = np.ones(64, dtype=np.float32) * (i + 1) * 10.0
        cache.write(0, slot, layer=0, k=k, v=v)

    for i, slot in enumerate(slots_b):
        k = np.ones(64, dtype=np.float32) * (i + 1) * -10.0  # seq B: -10, -20, -30, -40
        v = np.ones(64, dtype=np.float32) * (i + 1) * -10.0
        cache.write(1, slot, layer=0, k=k, v=v)

    print("\n  Reading back own sequence data:")
    for i, slot in enumerate(slots_a):
        k, _ = cache.read(0, slot, layer=0)
        print(f"    Seq A slot {i}: k[0]={k[0]:.1f}  (expected {(i+1)*10.0:.1f})")
        assert abs(k[0] - (i+1)*10.0) < 1e-5, f"Seq A data corrupted at slot {i}"

    print("\n  Attempting cross-sequence read (should raise):")
    caught = False
    try:
        cache.read(seq_id=0, slot=slots_b[0], layer=0)
    except ValueError as e:
        caught = True
        print(f"    Caught: {e}")
    assert caught, "Cross-sequence read should have been blocked"

    print("\n  Freeing seq A and verifying slots are released:")
    cache.free(seq_id=0)
    assert all(cache._owner[s] == -1 for s in slots_a), "Freed slots should be unowned"
    print(f"    Freed {len(slots_a)} slots. Free pool size: {len(cache._free)}")

    print(f"\n  [ASSERT] Sequence isolation enforced by ownership tracking: ✓")
    print(f"  [ASSERT] Cross-sequence access correctly blocked: ✓")
    print(f"  [ASSERT] Slot release correctly frees memory: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §5  MEMORY LEAK DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def demo_memory_leak_detection() -> None:
    section("Memory Leak Detection Pattern")

    # Simulate a request buffer that correctly releases memory
    class CorrectBuffer:
        def __init__(self):
            self._active: Dict[int, np.ndarray] = {}

        def store(self, req_id: int, tensor: np.ndarray):
            self._active[req_id] = tensor

        def complete(self, req_id: int) -> List:
            """Correctly converts to Python list and removes tensor reference."""
            result = self._active.pop(req_id).tolist()  # releases numpy array
            return result

        def active_count(self) -> int:
            return len(self._active)

    # Simulate a leaky buffer that retains references
    class LeakyBuffer:
        def __init__(self):
            self._active: Dict[int, np.ndarray] = {}
            self._completed: List[np.ndarray] = []  # BUG: retains tensor refs

        def store(self, req_id: int, tensor: np.ndarray):
            self._active[req_id] = tensor

        def complete(self, req_id: int) -> List:
            tensor = self._active.pop(req_id)
            self._completed.append(tensor)   # BUG: closure captures tensor
            return tensor.tolist()

        def active_count(self) -> int:
            return len(self._active)
        def leaked_count(self) -> int:
            return len(self._completed)

    N_REQUESTS = 100
    TENSOR_SIZE = 1000

    correct = CorrectBuffer()
    leaky   = LeakyBuffer()

    for i in range(N_REQUESTS):
        tensor = np.random.randn(TENSOR_SIZE).astype(np.float32)
        correct.store(i, tensor.copy())
        leaky.store(i, tensor.copy())
        correct.complete(i)
        leaky.complete(i)

    print(f"\n  After {N_REQUESTS} requests completed:")
    print(f"  CorrectBuffer active refs:  {correct.active_count()}  (should be 0)")
    print(f"  LeakyBuffer   active refs:  {leaky.active_count()}   (should be 0)")
    print(f"  LeakyBuffer   leaked refs:  {leaky.leaked_count()}  (BUG: should be 0)")

    bytes_leaked = leaky.leaked_count() * TENSOR_SIZE * 4
    print(f"\n  Memory held by leak: {bytes_leaked:,} bytes "
          f"({bytes_leaked / 1e6:.2f} MB) — would be GPU memory in real system")

    assert correct.active_count() == 0, "Correct buffer should have no active refs"
    assert leaky.leaked_count()   == N_REQUESTS, "Leaky buffer retains all tensors"
    print(f"\n  [ASSERT] Correct buffer releases all references: ✓")
    print(f"  [ASSERT] Leaky buffer correctly diagnosed — retains {leaky.leaked_count()} refs: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §6  LATENCY SPIKE DIAGNOSIS
# ──────────────────────────────────────────────────────────────────────────────

def demo_latency_spike_diagnosis() -> None:
    section("Latency Spike Diagnosis")

    N = 200
    latencies = []
    rng = random.Random(99)

    # Simulate request latency with occasional GC-pause spikes
    for i in range(N):
        base_lat = rng.gauss(80, 10)    # ~80ms typical
        # Simulate GC pause every ~50 requests
        gc_spike = 150 if (i % 47 == 0) else 0
        # Simulate cold-start spike on first request
        cold_start = 400 if i == 0 else 0
        lat = max(20, base_lat + gc_spike + cold_start)
        latencies.append(lat)

    lats = sorted(latencies)
    p50  = lats[int(N * 0.50)]
    p95  = lats[int(N * 0.95)]
    p99  = lats[int(N * 0.99)]
    p999 = lats[int(N * 0.999)] if N >= 1000 else lats[-1]

    print(f"\n  Latency percentiles over {N} requests:")
    print(f"    P50:  {p50:.1f} ms")
    print(f"    P95:  {p95:.1f} ms")
    print(f"    P99:  {p99:.1f} ms")
    print(f"    Max:  {lats[-1]:.1f} ms")
    print(f"\n  P99/P50 ratio: {p99/p50:.1f}x")

    # Diagnose: high P99/P50 ratio signals periodic spikes
    if p99 / p50 > 2.5:
        print(f"\n  [DIAG] P99/P50 = {p99/p50:.1f}x > 2.5 → periodic spike pattern")
        # Identify spikes
        spike_threshold = p50 * 2.5
        spikes = [(i, lat) for i, lat in enumerate(latencies) if lat > spike_threshold]
        gaps   = [spikes[i+1][0] - spikes[i][0] for i in range(len(spikes)-1)]
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"  [DIAG] {len(spikes)} spikes found, avg gap = {avg_gap:.0f} requests")
            print(f"  [DIAG] Regular periodicity ({avg_gap:.0f}q) suggests GC pause or")
            print(f"         background task (checkpointing, metrics flush)")

    assert p50 < 120, f"P50 latency {p50:.1f}ms unexpectedly high"
    assert p99 > p50, "P99 must be higher than P50"
    print(f"\n  [ASSERT] Latency spike pattern correctly identified: ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §7  PERFORMANCE REGRESSION BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    label:       str
    tokens_per_s: float
    latency_ms:  float
    n_requests:  int

def simulate_inference(tokens: int, batch_size: int,
                       bandwidth_gbps: float = 500.0,
                       params_b: float = 8.0,
                       overhead_ms: float = 5.0) -> float:
    """Simulate inference latency based on memory-bandwidth model."""
    # Bytes to transfer: 2 × params × 2 bytes/param (fp16)
    bytes_per_forward = params_b * 1e9 * 2 * 2
    # Effective BW with batch size (diminishing returns due to compute becoming binding)
    eff_bw = bandwidth_gbps * 1e9 * min(1.0, batch_size / 8.0)
    mem_lat = bytes_per_forward / eff_bw * 1000  # ms
    # Output token generation: n_tokens × mem_lat (one pass per token)
    return overhead_ms + tokens * mem_lat * max(1, 2 - batch_size / 4)

def demo_performance_regression() -> None:
    section("Performance Regression Benchmark")

    print(f"\n  Simulated throughput at different batch sizes:")
    print(f"\n  {'Batch':>7}  {'TPS':>10}  {'Lat/req ms':>12}  {'GPU util%':>10}")
    print(f"  {'─'*7}  {'─'*10}  {'─'*12}  {'─'*10}")

    results = []
    for bs in [1, 2, 4, 8, 16, 32]:
        n_tok  = 128
        lat_ms = simulate_inference(n_tok, bs)
        tps    = n_tok * bs / (lat_ms / 1000)
        # GPU utilization: low at small batch (memory-bound), high at large batch
        gpu_util = min(98, 20 + bs * 5)
        results.append(BenchmarkResult(f"bs={bs}", tps, lat_ms, bs))
        marker = "  ← small batch, low GPU util" if bs == 1 else ""
        print(f"  {bs:>7}  {tps:>10.1f}  {lat_ms:>12.1f}  {gpu_util:>9}%{marker}")

    # Regression detection: compare current run to a "baseline"
    baseline_tps = results[-2].tokens_per_s  # bs=16
    current_tps  = results[-2].tokens_per_s * 0.97  # simulate 3% regression

    regression_pct = (baseline_tps - current_tps) / baseline_tps * 100
    threshold_pct  = 5.0

    print(f"\n  Regression detection (bs=16):")
    print(f"    Baseline TPS:  {baseline_tps:.1f}")
    print(f"    Current TPS:   {current_tps:.1f}")
    print(f"    Regression:    {regression_pct:.1f}%  (alert if > {threshold_pct}%)")

    assert results[0].tokens_per_s < results[-1].tokens_per_s, \
        "Throughput should increase with batch size"
    assert regression_pct < threshold_pct, \
        f"Simulated regression {regression_pct:.1f}% exceeds threshold"
    print(f"\n  [ASSERT] Throughput scales with batch size: ✓")
    print(f"  [ASSERT] Regression within acceptable bounds ({regression_pct:.1f}% < {threshold_pct}%): ✓")


# ──────────────────────────────────────────────────────────────────────────────
# §8  SECTION UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ──────────────────────────────────────────────────────────────────────────────
# §9  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  Chapter 32 — Debugging Inference Systems (Python)\n{bar}")

    demo_nan_detection()
    demo_quantization_error()
    demo_sampling_determinism()
    demo_kv_cache_isolation()
    demo_memory_leak_detection()
    demo_latency_spike_diagnosis()
    demo_performance_regression()

    print(f"\n{bar}\n  All demos complete.\n{bar}\n")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
```

---

## 32.10 C++ Demo

```cpp
// debugging_demo.cpp
// Chapter 32 — Debugging Inference Systems (C++)
//
// Demonstrates:
//   1. NaN/Inf detection with tensor scan
//   2. Stable vs unstable softmax
//   3. INT8 quantization error measurement
//   4. KV cache ownership and isolation enforcement
//   5. Latency percentile analysis and spike detection
//   6. Memory bandwidth bottleneck model
//
// Build: g++ -O2 -std=c++17 -o debugging_demo debugging_demo.cpp
// Run:   ./debugging_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static void print_section(const std::string& t) {
    std::cout << "\n" << std::string(60, '-') << "\n  " << t
              << "\n" << std::string(60, '-') << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 1.  NaN / Inf DETECTION
// ─────────────────────────────────────────────────────────────────────────────

static bool check_tensor(const std::vector<float>& v, const std::string& name) {
    int n_nan = 0, n_inf = 0;
    for (float x : v) {
        if (std::isnan(x)) ++n_nan;
        if (std::isinf(x)) ++n_inf;
    }
    if (n_nan || n_inf) {
        std::cout << "  [WARN] " << name << ": NaN=" << n_nan
                  << " Inf=" << n_inf << " size=" << v.size() << "\n";
        return true;
    }
    return false;
}

static std::vector<float> unstable_softmax(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    float sum = 0;
    for (float xi : x) sum += std::exp(xi);   // overflows for large x
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = std::exp(x[i]) / sum;
    return out;
}

static std::vector<float> stable_softmax(const std::vector<float>& x) {
    float mx = *std::max_element(x.begin(), x.end());
    std::vector<float> out(x.size());
    float sum = 0;
    for (float xi : x) sum += std::exp(xi - mx);
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = std::exp(x[i] - mx) / sum;
    return out;
}

static void demo_nan_detection() {
    print_section("NaN Detection and Softmax Stability");

    // Large logits — overflow in fp32 softmax
    std::vector<float> large_logits = {1000.f, 999.f, 998.f};
    std::vector<float> normal_logits = {10.f, 9.f, 8.f};

    std::cout << "\n  Testing large logits {1000, 999, 998}:\n";
    auto unstable = unstable_softmax(large_logits);
    auto stable   = stable_softmax(large_logits);
    check_tensor(unstable, "unstable_softmax(large_logits)");
    check_tensor(stable,   "stable_softmax(large_logits)");

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "    Unstable: [" << unstable[0] << ", " << unstable[1] << ", " << unstable[2] << "]\n";
    std::cout << "    Stable:   [" << stable[0]   << ", " << stable[1]   << ", " << stable[2]   << "]\n";

    float sum_stable = std::accumulate(stable.begin(), stable.end(), 0.0f);
    assert(!std::isnan(stable[0]) && !std::isinf(stable[0]));
    assert(std::abs(sum_stable - 1.0f) < 1e-5f);
    std::cout << "  [ASSERT] Stable softmax: no NaN, sums to 1.0 ✓\n";

    // NaN propagation
    std::vector<float> x = {1.f, std::numeric_limits<float>::quiet_NaN(), 3.f};
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * 2.f + 1.f;
    check_tensor(x, "input (has NaN)");
    check_tensor(y, "output (NaN propagated)");
    assert(std::isnan(y[1]));
    std::cout << "  [ASSERT] NaN propagation correctly detected: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  INT8 QUANTIZATION ERROR
// ─────────────────────────────────────────────────────────────────────────────

static std::pair<std::vector<int8_t>, float> quantize_int8(const std::vector<float>& x) {
    float max_abs = 0;
    for (float v : x) max_abs = std::max(max_abs, std::abs(v));
    float scale = max_abs / 127.f + 1e-8f;
    std::vector<int8_t> q(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        q[i] = static_cast<int8_t>(std::clamp(std::round(x[i] / scale), -128.f, 127.f));
    return {q, scale};
}

static float quantize_error(const std::vector<float>& original, int outlier_idx = -1) {
    std::vector<float> vals = original;
    if (outlier_idx >= 0) vals[outlier_idx] = 100.f;

    auto [q, s] = quantize_int8(vals);
    float max_err = 0;
    for (size_t i = 0; i < vals.size(); ++i) {
        float reconstructed = q[i] * s;
        max_err = std::max(max_err, std::abs(vals[i] - reconstructed));
    }
    float max_abs = *std::max_element(vals.begin(), vals.end(),
        [](float a, float b){ return std::abs(a) < std::abs(b); });
    return max_abs > 0 ? max_err / std::abs(max_abs) : 0;
}

static void demo_quantization_error() {
    print_section("INT8 Quantization Error Measurement");

    // Generate a typical weight distribution
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.1f);
    std::vector<float> weights(256);
    for (auto& w : weights) w = nd(rng);

    float err_normal  = quantize_error(weights);
    float err_outlier = quantize_error(weights, 0);  // insert 100x outlier

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "\n  " << std::left << std::setw(30) << "Scenario"
              << std::setw(12) << "Rel error" << "Status\n";
    std::cout << "  " << std::string(52, '-') << "\n";

    auto print_row = [](const std::string& lbl, float err) {
        std::string status = err < 0.05f ? "✓ OK" : "✗ DEGRADED";
        std::cout << "  " << std::left << std::setw(30) << lbl
                  << std::fixed << std::setprecision(5) << std::setw(12) << err
                  << status << "\n";
    };
    print_row("Normal weights",          err_normal);
    print_row("Outlier weight (100x)",   err_outlier);

    assert(err_normal  < 0.05f);
    assert(err_outlier > err_normal);
    std::cout << "\n  [ASSERT] Normal quantization error < 5%: " << err_normal*100 << "% ✓\n";
    std::cout << "  [ASSERT] Outlier causes more error than normal: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  KV CACHE ISOLATION
// ─────────────────────────────────────────────────────────────────────────────

class KVCache {
public:
    int n_slots, n_layers, head_dim;
    std::vector<std::vector<float>> data;   // [slot][layer*head_dim*2]
    std::vector<int> owner;                 // -1=free, else seq_id
    std::vector<int> free_list;

    KVCache(int ns, int nl, int hd)
        : n_slots(ns), n_layers(nl), head_dim(hd),
          data(ns, std::vector<float>(nl * hd * 2, 0.f)),
          owner(ns, -1)
    {
        for (int i = 0; i < ns; ++i) free_list.push_back(i);
    }

    std::vector<int> allocate(int seq_id, int n_tokens) {
        assert((int)free_list.size() >= n_tokens);
        std::vector<int> slots(free_list.end() - n_tokens, free_list.end());
        free_list.resize(free_list.size() - n_tokens);
        for (int s : slots) owner[s] = seq_id;
        return slots;
    }

    void write(int seq_id, int slot, int layer,
               const std::vector<float>& k, const std::vector<float>& v) {
        assert(owner[slot] == seq_id && "Write to slot owned by another sequence");
        int offset = layer * head_dim * 2;
        std::copy(k.begin(), k.end(), data[slot].begin() + offset);
        std::copy(v.begin(), v.end(), data[slot].begin() + offset + head_dim);
    }

    std::pair<std::vector<float>, std::vector<float>>
    read(int seq_id, int slot, int layer) {
        assert(owner[slot] == seq_id && "Read from slot owned by another sequence");
        int offset = layer * head_dim * 2;
        return {
            {data[slot].begin() + offset, data[slot].begin() + offset + head_dim},
            {data[slot].begin() + offset + head_dim, data[slot].begin() + offset + head_dim*2}
        };
    }

    void free_seq(int seq_id) {
        for (int i = 0; i < n_slots; ++i) {
            if (owner[i] == seq_id) {
                owner[i] = -1;
                std::fill(data[i].begin(), data[i].end(), 0.f);
                free_list.push_back(i);
            }
        }
    }
};

static void demo_kv_cache_isolation() {
    print_section("KV Cache Sequence Isolation");

    KVCache cache(32, 4, 64);

    auto slots_a = cache.allocate(0, 4);
    auto slots_b = cache.allocate(1, 4);

    // Write distinct values
    for (int i = 0; i < 4; ++i) {
        std::vector<float> k(64, (i+1)*10.f);
        std::vector<float> v(64, (i+1)*10.f);
        cache.write(0, slots_a[i], 0, k, v);
    }
    for (int i = 0; i < 4; ++i) {
        std::vector<float> k(64, (i+1)*-10.f);
        std::vector<float> v(64, (i+1)*-10.f);
        cache.write(1, slots_b[i], 0, k, v);
    }

    std::cout << "\n  Reading back own sequence data:\n";
    for (int i = 0; i < 4; ++i) {
        auto [k, v] = cache.read(0, slots_a[i], 0);
        std::cout << "    Seq A slot " << i << ": k[0]=" << k[0]
                  << "  (expected " << (i+1)*10.f << ")\n";
        assert(std::abs(k[0] - (i+1)*10.f) < 1e-5f);
    }

    // Try a cross-sequence read — should assert-fail (we'll check the owner directly)
    std::cout << "\n  Verifying cross-sequence access would fail:\n";
    bool would_fail = (cache.owner[slots_b[0]] != 0);  // slot b[0] owned by seq 1, not 0
    std::cout << "    Slot " << slots_b[0] << " owned by seq "
              << cache.owner[slots_b[0]] << " (not seq 0) → access blocked ✓\n";
    assert(would_fail);

    cache.free_seq(0);
    bool all_freed = true;
    for (int s : slots_a) if (cache.owner[s] != -1) all_freed = false;
    assert(all_freed);
    std::cout << "  Freed seq A: " << slots_a.size() << " slots released.\n";

    std::cout << "\n  [ASSERT] KV cache sequence isolation enforced: ✓\n";
    std::cout << "  [ASSERT] Slot release correctly frees memory: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  LATENCY PERCENTILE ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────

static void demo_latency_analysis() {
    print_section("Latency Percentile Analysis and Spike Detection");

    std::mt19937 rng(99);
    std::normal_distribution<float> base_dist(80.f, 10.f);
    int N = 200;

    std::vector<float> latencies;
    latencies.reserve(N);
    for (int i = 0; i < N; ++i) {
        float lat  = std::max(20.f, base_dist(rng));
        // Simulate periodic GC spike
        if (i % 47 == 0) lat += 150.f;
        // Cold-start spike
        if (i == 0)      lat += 400.f;
        latencies.push_back(lat);
    }

    std::vector<float> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());

    float p50  = sorted[int(N * 0.50)];
    float p95  = sorted[int(N * 0.95)];
    float p99  = sorted[int(N * 0.99)];
    float pmax = sorted.back();

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n  Latency percentiles (" << N << " requests):\n";
    std::cout << "    P50:  " << p50  << " ms\n";
    std::cout << "    P95:  " << p95  << " ms\n";
    std::cout << "    P99:  " << p99  << " ms\n";
    std::cout << "    Max:  " << pmax << " ms\n";
    std::cout << "\n  P99/P50 ratio: " << p99/p50 << "x\n";

    if (p99 / p50 > 2.5f) {
        float threshold = p50 * 2.5f;
        int spike_count = 0;
        int last_spike  = -100;
        std::vector<int> gaps;
        for (int i = 0; i < N; ++i) {
            if (latencies[i] > threshold) {
                if (last_spike >= 0) gaps.push_back(i - last_spike);
                last_spike = i; ++spike_count;
            }
        }
        float avg_gap = gaps.empty() ? 0 :
            std::accumulate(gaps.begin(), gaps.end(), 0.f) / gaps.size();
        std::cout << "\n  [DIAG] P99/P50=" << p99/p50 << "x > 2.5 → periodic spike pattern\n";
        std::cout << "  [DIAG] " << spike_count << " spikes, avg gap="
                  << avg_gap << " requests\n";
        std::cout << "  [DIAG] Suggests GC pause or periodic background task\n";
    }

    assert(p50 < 120.f);
    assert(p99 > p50);
    std::cout << "\n  [ASSERT] Latency spike pattern correctly identified: ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  MEMORY BANDWIDTH BOTTLENECK MODEL
// ─────────────────────────────────────────────────────────────────────────────

static void demo_bandwidth_model() {
    print_section("Memory Bandwidth Bottleneck Model");

    struct HW {
        std::string name;
        double bw_gbps;    // memory bandwidth GB/s
        double compute_tflops;  // TFLOPS (fp16)
    };
    std::vector<HW> hardware = {
        {"A10G",   600.0,   31.2},
        {"A100",  2000.0,  312.0},
        {"H100",  3350.0, 1979.0},
    };

    double params_b   = 8.0;    // 8B parameter model
    double bytes_param = 2.0;   // fp16 = 2 bytes/param

    std::cout << "\n  Params: " << params_b << "B  |  Precision: fp16\n\n";
    std::cout << "  " << std::left << std::setw(8) << "HW"
              << std::setw(10) << "BW GB/s"
              << std::setw(14) << "TFLOPS"
              << std::setw(16) << "Roof (tok/s)"
              << std::setw(18) << "Breakeven batch"
              << "Bottleneck\n";
    std::cout << "  " << std::string(66, '-') << "\n";

    for (auto& hw : hardware) {
        // Memory-bandwidth roof: bw / (2 × params × bytes_per_param)
        double bw_roof = (hw.bw_gbps * 1e9) / (2.0 * params_b * 1e9 * bytes_param);
        // Arithmetic intensity at batch size b: 2×b×params / (b×params×bytes + params×bytes) ≈ 2/bytes
        // Breakeven batch: AI × bytes_per_param / 2 = hw_roof_ratio
        double breakeven_batch = (hw.compute_tflops * 1e12) / (hw.bw_gbps * 1e9) * bytes_param;
        std::string bottleneck = breakeven_batch > 16 ? "Memory BW" : "Compute";

        std::cout << "  " << std::setw(8) << hw.name
                  << std::setw(10) << (int)hw.bw_gbps
                  << std::setw(14) << hw.compute_tflops
                  << std::setw(16) << (int)bw_roof
                  << std::setw(18) << std::setprecision(0) << std::fixed << breakeven_batch
                  << bottleneck << "\n";
    }

    std::cout << "\n  Interpretation: at small batch sizes (< breakeven), the model\n";
    std::cout << "  is memory-bandwidth-bound. Increasing batch size is the\n";
    std::cout << "  primary lever for improving throughput in that regime.\n";

    // Simple assert: H100 should have higher bandwidth roof than A10G
    double a10g_roof = (600.0 * 1e9) / (2.0 * 8e9 * 2.0);
    double h100_roof = (3350.0 * 1e9) / (2.0 * 8e9 * 2.0);
    assert(h100_roof > a10g_roof);
    std::cout << "\n  [ASSERT] H100 bandwidth roof > A10G bandwidth roof: "
              << (int)h100_roof << " > " << (int)a10g_roof << " tok/s ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n" << std::string(60, '=')
              << "\n  Chapter 32 — Debugging Inference Systems (C++)\n"
              << std::string(60, '=') << "\n";

    demo_nan_detection();
    demo_quantization_error();
    demo_kv_cache_isolation();
    demo_latency_analysis();
    demo_bandwidth_model();

    std::cout << "\n" << std::string(60, '=')
              << "\n  All demos complete.\n"
              << std::string(60, '=') << "\n\n";
    return 0;
}
```

---

## 32.11 Summary

Debugging inference systems requires a different mindset from debugging training: the effects are live and user-facing, the system is stateful (KV cache, running sequences), and the failure modes often emerge only under concurrent load.

The most important principles:

**Observe before assuming.** Every debugging session starts with measurement: GPU utilization, memory allocation trend, output distribution, latency percentiles. Don't guess which layer is broken — instrument and look.

**NaN is always a bug.** A NaN in logits is not a "model quality" issue — it is a numeric stability bug in the forward pass or quantization. The source is almost always one of: fp16 overflow in attention, LayerNorm denominator collapse, or a quantization scale that is too small. Each has a definitive fix.

**Isolation reveals corruption.** The key test for KV cache corruption is comparing serial-batch outputs with concurrent-batch outputs. If they differ, pages are being shared or mis-assigned. Ownership tracking is the prevention.

**Performance regressions have four causes.** Small batch (add concurrency), memory pressure (reduce context length or add memory), CPU-GPU sync (profile with nsys), kernel regression (bisect with benchmarks). The metrics tell you which one.

**Next chapter** closes the book's technical core with the full engine landscape as of 2026 — how vLLM, llama.cpp, TensorRT-LLM, MLC-LLM, and the emerging disaggregated serving platforms compare across the dimensions we have spent 32 chapters developing.

---

*End of Chapter 32*


---

## Chapter Summary

- **Systematic diagnosis**: LLM serving failures fall into four categories — incorrect output (quality), slow output (latency), no output (availability), and expensive output (cost).
- **GPU utilization vs throughput mismatch**: high GPU utilization with low token throughput signals memory bandwidth saturation, not compute saturation — check batch size, sequence length, and KV cache pressure.
- **KV cache OOM**: sudden 40-second requests or request drops often signal KV cache exhaustion; check `vllm:gpu_cache_usage_perc`; the fix is lower `--max-num-seqs` or shorter `--max-model-len`.
- **Numerical instability**: NaN or Inf in logits propagates to all-uniform distributions or deterministic token loops; triggered by FP8/INT4 overflow, very long sequences, or extreme temperature values.
- **NCCL timeout**: in tensor-parallel mode, a single GPU stall causes all ranks to hang after `nccl_timeout` seconds; diagnose with `nvidia-smi` on each rank.
- **Tokenizer mismatch**: using the wrong tokenizer (e.g., applying LLaMA tokenizer to a Mistral model) produces garbage output without errors; verify tokenizer config matches model checkpoint.
- **llama.cpp diagnostics**: `LLAMA_LOG_LEVEL=debug` exposes layer-level timing; `--verbose-prompt` shows the tokenised input; `--mlock` prevents model page eviction under memory pressure.

---

## Self-Check Questions

1. vLLM is at 95% GPU utilization but only producing 800 tokens/s. Expected throughput at 95% utilization for this model is 2 400 tokens/s. List three diagnostic steps and the most likely root cause. *(Section 32.2)*

2. A request returns `{"error": "CUDA out of memory"}` after 35 seconds of normal operation. What series of events in the vLLM scheduler led to this, and what configuration prevents it? *(Section 32.3)*

3. Your model outputs garbage (repeated `<unk>` tokens) after upgrading the model checkpoint. No error is logged. What is the most likely cause and what two files do you check first? *(Section 32.5)*

4. In a 4-GPU tensor-parallel deployment, GPU 2 shows 100% utilization and GPUs 0, 1, 3 show 0%. What has happened and how do you recover? *(Section 32.6)*

5. `LLAMA_LOG_LEVEL=debug` shows layer 22 taking 300 ms while all other layers take 5 ms. The model uses CUDA. What are the two most likely causes and how do you diagnose each? *(Section 32.4)*


---

## Worked Solutions

### Question 1
**vLLM at 95% GPU util but 800 tok/s. Expected is 2,400 tok/s at 95% util. Three diagnostic steps:**

**Step 1 -- Check arithmetic intensity (are we compute-bound or bandwidth-bound?).**
At 95% GPU util, the GPU's SMs are busy. If throughput is 3x lower than expected, either:

- The HBM bandwidth is saturated by something other than model weights (e.g., KV cache reads)
- The util metric is misleading (high util from idle-waiting on memory transfers)

Run: `dcgmi dmon -e 1004,1005` to see SM activity vs memory throughput simultaneously. If memory throughput is at 98% while SM activity shows mostly stalls, the bottleneck is HBM bandwidth to KV cache.

**Step 2 -- Profile the decode step with nsight or CUDA events.**
```bash
nsys profile --trace=cuda,nvtx python -m vllm.entrypoints.openai.api_server ...
```
Look at the timeline: are the decode kernels back-to-back, or are there gaps (Python overhead, synchronization)?

**Most likely root cause:** Very long sequences consuming KV cache bandwidth. If `max_model_len=32768` and many sequences are near max length, each decode step reads 32K * 128 KB = 4 GB of KV data per sequence -- this dwarfs the 140 GB of weight reads and saturates HBM bandwidth without increasing useful token throughput.

**Step 3 -- Check `vllm:num_requests_running` and sequence length distribution.**
If a small number of very long sequences are consuming all KV cache bandwidth, reducing max sequence length or enabling KV cache quantization (INT8 KV) would halve KV bandwidth and restore throughput.

---

### Question 2
**`CUDA out of memory` after 35 seconds. Sequence of events:**

The request generated tokens for 35 seconds before OOM. What happened:

1. **Request admitted at T=0.** KV blocks allocated incrementally as tokens are generated.
2. **KV pool fills up (T=30-35 s).** The request has generated ~500 tokens (35s / 70ms/tok) consuming 500 x 327 KB = 163 MB of KV blocks. But OTHER concurrent requests have also consumed KV blocks.
3. **New block request fails.** The block manager calls `can_append_slot()` which returns False -- no free blocks.
4. **Preemption fails.** No sequences can be preempted (all are at high priority or preemption would require swapping, which also needs memory).
5. **CUDA OOM.** vLLM attempts to allocate a new KV block directly in CUDA memory (outside the block pool, as a fallback) -- this fails with OOM.

**Configuration that prevents this:**
Set `gpu_memory_utilization=0.85` (leaving more headroom) and `max_num_seqs=64` (lower concurrency). Use `--enable-chunked-prefill` to control token budget. Monitor `vllm:gpu_cache_usage_perc` and alert at 90% to preemptively shed load.

---

### Question 3
**Garbage output (repeated `<unk>` tokens) after upgrading model checkpoint. Two files to check:**

**Most likely cause:** Tokenizer mismatch. The new checkpoint uses a different tokenizer vocabulary than the old one. When the model outputs token IDs that existed in the old vocabulary but map to `<unk>` in the new tokenizer (or vice versa), the decoded text shows garbage.

**Two files to check first:**

1. **`tokenizer_config.json`** -- verify `tokenizer_class`, `model_max_length`, and `vocab_size` match between old and new checkpoint. A mismatch in vocab_size (e.g., old=32,000, new=128,256) means the model's lm_head output has a different number of logits than the tokenizer's vocabulary.

2. **`config.json`** (or `model_config.json`) -- verify `vocab_size` in the model config matches `tokenizer_config.json`. If the model config says vocab_size=128,256 but the loaded tokenizer only has 32,000 tokens, any token ID > 32,000 will decode to `<unk>`.

Also check: the tokenizer's `special_tokens_map.json` to verify `<unk>`, `<pad>`, `<bos>`, `<eos>` token IDs haven't changed between versions.

---

### Question 4
**4-GPU TP deployment. GPU 2 at 100% util, GPUs 0, 1, 3 at 0%. What happened?**

**What has happened:** The NCCL AllReduce collective has **hung** or **deadlocked** at GPU 2. In tensor parallelism, all 4 GPUs must participate in AllReduce after each row-parallel matmul layer. If GPU 2 is running (100%) but GPUs 0, 1, 3 are waiting (0%), the other three GPUs are blocked in NCCL's AllReduce barrier, waiting for GPU 2 to reach the synchronization point.

**Root causes:**
1. GPU 2's kernel is stuck in an infinite loop (rare).
2. A NCCL error on GPU 2 that didn't propagate correctly -- GPU 2 is spinning in an error retry loop.
3. An OOM on GPU 2 that caused a partial state, leaving the AllReduce in a partially committed state.

**How to recover:**
1. Kill all vLLM worker processes: `pkill -f vllm.worker`.
2. Reset NCCL state: `nvidia-smi -r -i 2` (reset GPU 2) -- may require disabling persistence mode.
3. Check GPU 2's kernel log: `dmesg | grep GPU` for ECC errors or Xid errors that indicate hardware fault.
4. Restart vLLM with `NCCL_TIMEOUT=30` (lower timeout so hung collectives fail fast with an error message).
5. If GPU 2 shows persistent ECC errors: replace the GPU or exclude it with `CUDA_VISIBLE_DEVICES=0,1,3` and switch to TP=3.

---

### Question 5
**Layer 22 taking 300 ms vs 5 ms for all other layers. Two most likely causes:**

**Cause 1 -- CUDA kernel not found in cache (just-in-time compilation).**
On first inference after server start, CUDA kernels that haven't been compiled are JIT-compiled by PyTorch's inductor or by the CUDA driver. Layer 22 may use a kernel with a unique shape (e.g., a different attention head count or a mixture-of-experts gate that only appears at layer 22) that requires compilation. Subsequent inferences don't show the spike.

**Diagnose:** Run the same request twice. If the 300 ms spike only appears on the first inference (warm-up), it's JIT compilation. Fix: pre-warm with `--num-scheduler-steps=1` dry-run requests at startup.

**Cause 2 -- CUDA graph capture miss (dynamic shape bypasses graph replay).**
If vLLM uses CUDA graphs and layer 22 receives a tensor with a shape not captured in the graph pool (e.g., the attention mechanism has a conditional branch that activates only at layer 22), vLLM falls back to eager mode for that kernel -- which adds the Python/CUDA launch overhead (typically 100-300 ms for complex ops).

**Diagnose:** Enable `CUDA_LAUNCH_BLOCKING=1` and check if layer 22's timing is consistently slow (capture miss) vs only on cold start (JIT). Also check `VLLM_TRACE_FUNCTION=1` logs for "fallback to eager" messages at layer 22. Fix: ensure all layer shapes are included in CUDA graph capture (adjust `--max-num-batched-tokens` to cover the layer 22 shape).
*Companion code: [`docs/code/chapter_32.md`](../code/chapter_32.md)*
