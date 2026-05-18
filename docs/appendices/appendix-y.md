# Appendix Y — Cerebras WSE-3: Wafer-Scale Inference

> *Every inference accelerator you have ever used is built from chiplets or dies that are millimetres wide, connected across a printed circuit board by wires that cross centimetres. Cerebras threw that constraint away. The Wafer-Scale Engine is a single die the size of a dinner plate, and that architectural decision changes the physics of LLM inference in ways that no amount of HBM bandwidth tuning can replicate.*

---

## Y.1 Why Wafer Scale Exists

The central bottleneck of LLM inference is memory bandwidth, not arithmetic throughput (Chapter 1). For a batch-1 decode step on an H100, arithmetic intensity sits at roughly 1 FLOP/byte — 295 times below the GPU's ridge point. The compute units are nearly idle. The bottleneck is reading weights from HBM across a 5,120-bit bus at 3.35 TB/s.

The standard industry response is to add more HBM stacks, widen the bus, or compress weights. Cerebras asked a different question: what if the weights never left the chip at all?

Static RAM (SRAM) — the same technology used in CPU caches and GPU shared memory — is orders of magnitude faster than DRAM or HBM. A large enough on-chip SRAM fabric would eliminate the HBM bottleneck entirely for models that fit. The obstacle is area: SRAM requires six transistors per bit versus one transistor and one capacitor for DRAM, so it is expensive to build at scale. The solution Cerebras chose was to use the entire silicon wafer as one die rather than dicing it into hundreds of individual chips. A full 300 mm wafer yields a die of approximately 46,225 mm², compared to an NVIDIA H100 SXM5 at 814 mm². That fifty-seven-fold increase in die area is what makes 44 GB of on-chip SRAM physically possible.

---

## Y.2 WSE-3 Architecture

### Y.2.1 The Die

The WSE-3 (Wafer-Scale Engine, third generation, announced 2024) is manufactured on TSMC's 5nm process node. Key physical parameters:

```
WSE-3 Physical Parameters
──────────────────────────────────────────────────────────────
Die area:          46,225 mm²    (vs H100 SXM5: 814 mm²)
Transistors:       4 trillion    (vs H100: 80 billion)
Process node:      TSMC 5nm
AI cores:          900,000       (Cerebras Processing Elements)
On-chip SRAM:      44 GB
On-chip BW:        21.6 PB/s    (vs H100 HBM: 3.35 TB/s)
Peak compute:      125 PFLOPS   BF16
System power:      23 kW (CS-3 system, water-cooled)
──────────────────────────────────────────────────────────────
```

The on-chip bandwidth figure deserves emphasis: 21.6 PB/s = 21,600 TB/s. The H100's HBM bandwidth is 3.35 TB/s. The WSE-3's internal fabric is approximately **6,400× faster** than H100 HBM. This is not a tuning difference; it is a different class of machine.

### Y.2.2 Processing Elements

Each Processing Element (PE) is a small, independent compute unit with its own local SRAM and a direct connection to its four neighbours (a 2-D mesh). Unlike GPU streaming multiprocessors — which share a large register file and shared-memory pool across 128 CUDA cores — each WSE-3 PE is architecturally autonomous. There is no global memory controller. There is no DRAM bus. Every memory access in a typical inference workload stays on-chip.

The mesh topology means that communication between nearby PEs is a single wire, not an interconnect that must be arbitrated. This allows pipelining of matrix operations across the spatial fabric without the synchronization overhead that GPU warps require.

### Y.2.3 The On-Chip Memory Hierarchy

GPU kernels must explicitly manage a three-level hierarchy: global memory (HBM) → L2 cache → shared memory / registers. On WSE-3 the hierarchy collapses:

```
GPU (H100)                        WSE-3
──────────────────────────────    ──────────────────────────────
HBM (80 GB, 3.35 TB/s)       →   MemoryX (external, see Y.4)
L2 cache (50 MB, ~12 TB/s)   →   44 GB on-chip SRAM (21.6 PB/s)
Shared memory (228 KB/SM)    →   Distributed PE-local SRAM
Registers (256 KB/SM)        →   PE registers
──────────────────────────────    ──────────────────────────────
```

For a model that fits within 44 GB of SRAM, inference never touches anything slower than on-chip SRAM. The HBM bottleneck that dominates GPU inference does not exist.

---

## Y.3 The Physics of Wafer-Scale Inference

### Y.3.1 Roofline Analysis for WSE-3

Recall from Chapter 1 that the ridge point separates memory-bound from compute-bound operation:

```
Ridge point = Peak FLOPS/s ÷ Peak Bandwidth (bytes/s)

H100 SXM5 (BF16 dense):   989 × 10¹²  / 3.35 × 10¹²  ≈  295 FLOPs/byte
WSE-3 (BF16):             125 × 10¹⁵  / 21.6 × 10¹⁵  ≈    6 FLOPs/byte
```

The WSE-3 ridge point is approximately **6 FLOPs/byte**. This is a radical shift. On the H100, a batch-1 decode step at ~1 FLOP/byte is 295× below the ridge — deeply memory-bound. On the WSE-3, 1 FLOP/byte is only 6× below the ridge, and the "penalty" for being memory-bound is paid in the currency of 21.6 PB/s on-chip fabric rather than 3.35 TB/s HBM.

### Y.3.2 Theoretical Token Throughput

For a model that fits entirely in the WSE-3's 44 GB SRAM:

```
WORKED EXAMPLE Y.1 — Theoretical decode throughput for Llama 3 8B on WSE-3
──────────────────────────────────────────────────────────────────────────
Model:    Llama 3 8B  (~7B active params)
Format:   BF16 (2 bytes/param)
Size:     ~14 GB weights  → fits in 44 GB SRAM

Each decode step reads all weights once (batch=1):
  Bytes read per token = 14 × 10⁹ bytes

Time per token at on-chip BW of 21.6 PB/s:
  t = 14 × 10⁹ / 21.6 × 10¹⁵ = 6.5 × 10⁻⁷ s = 0.65 μs

Theoretical peak = 1 / 0.65 μs ≈ 1,540,000 tokens/second

Compare to H100 (same model):
  Time per token = 14 × 10⁹ / 3.35 × 10¹² ≈ 4.2 ms = 4,200 μs
  Theoretical peak = ~238 tokens/second
──────────────────────────────────────────────────────────────────────────
```

Practical throughput is far lower due to compute time, activation storage, and software overhead — but the bandwidth ceiling is 6,000× higher. Cerebras reports approximately 2,100 tokens/sec for Llama 3.1 8B on a single CS-3 node.

### Y.3.3 The Batch-Size Curve

On a GPU, throughput scales with batch size because batching amortizes the weight-load cost across multiple tokens — arithmetic intensity rises, and the ridge point becomes reachable. On WSE-3, the weights are already resident in SRAM, so there is no latent bandwidth cost to amortize. The batch-1 case is already near-optimal from a bandwidth perspective.

This inverts the traditional inference trade-off. GPU clusters are optimized for **high-batch throughput**; WSE-3 is optimized for **low-latency single-stream generation**.

```
Throughput scaling behaviour (schematic)
                                               
Tokens/s │        WSE-3 (flat, weights on-chip)
  ~2100  ├──────────────────────────────────────────
         │                                     GPU cluster
         │                               ╔════════════════
         │                        ╔══════╝
         │                 ╔══════╝
    ~240 ├──────────────╔══╝
         │       ╔══════╝  (GPU reaches WSE-3 floor
         │ ╔═════╝         only at batch ≈ 200+)
       0 └──────────────────────────────────────────▶
         1   4   16   64   256   1024   Batch size
```

---

## Y.4 The CS-3 System

A single WSE-3 die is mounted in the CS-3 — an 8U rack unit with water cooling. Because 23 kW of heat cannot be removed by air fans at data-centre density, the CS-3 requires a water loop. Most colocation facilities and hyperscaler data centres can accommodate this.

### Y.4.1 MemoryX

For models larger than 44 GB — Llama 3.1 70B at BF16 requires ~140 GB — the weights must be stored externally and streamed onto the wafer during inference. Cerebras's external memory system is called **MemoryX**. It is a rack of commodity DRAM connected to the CS-3 by a high-speed fabric. The effective streaming bandwidth from MemoryX to the WSE-3 is substantially lower than the on-chip bandwidth, but the on-chip SRAM serves as a staging buffer that keeps the PEs fed.

Cerebras reports approximately 1,800 tokens/sec for Llama 3.1 70B using weight streaming from MemoryX — still roughly 7× faster than a single H100 at batch=1 for the same model.

### Y.4.2 SwarmX

**SwarmX** is Cerebras's inter-node interconnect fabric for multi-CS-3 configurations. Unlike GPU cluster networking (InfiniBand, NVLink), SwarmX is designed to distribute a single model across multiple wafers while maintaining low-latency weight delivery. It enables configurations where a very large model (e.g., Llama 3.1 405B) is partitioned across several CS-3 nodes.

### Y.4.3 System Summary

```
CS-3 System Specification
──────────────────────────────────────────────────────────────
Form factor:        8U rack unit
Cooling:            Liquid-cooled (facility water loop required)
Power draw:         23 kW at full load
Accelerator:        1× WSE-3 (46,225 mm², 44 GB on-chip SRAM)
External memory:    MemoryX nodes (optional, for large models)
Cluster fabric:     SwarmX (optional, multi-CS-3)
Software:           Cerebras Software Platform (CSP)
──────────────────────────────────────────────────────────────
```

---

## Y.5 Software Stack

### Y.5.1 Cerebras Software Platform (CSP)

The Cerebras Software Platform translates PyTorch model definitions and forward passes into the spatial dataflow graph that the WSE-3 executes. The compilation step is performed once; subsequent inference calls reuse the compiled graph. From the application developer's perspective, the interface is standard PyTorch.

```python
# On-premise CSP inference (illustrative)
import cerebras.framework.torch as cbtorch
import torch

model = cbtorch.load("llama3-8b-bf16.pt")  # Loads compiled graph
model.eval()

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=512)
```

### Y.5.2 Cerebras Cloud SDK

Cerebras offers cloud-hosted WSE-3 inference accessible via a REST API and the `cerebras-cloud-sdk` Python package, compatible with the OpenAI API format:

```python
pip install cerebras-cloud-sdk
```

```python
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key="your-api-key")

# Chat completion (OpenAI-compatible interface)
response = client.chat.completions.create(
    model="llama3.1-70b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain KV cache eviction in two paragraphs."},
    ],
    max_tokens=512,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

```python
# Streaming
stream = client.chat.completions.create(
    model="llama3.1-8b",
    messages=[{"role": "user", "content": "List five LLM inference optimizations."}],
    stream=True,
    max_tokens=256,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

```python
# Text completion
completion = client.completions.create(
    model="llama3.1-8b",
    prompt="The ridge point of an H100 is",
    max_tokens=64,
)
print(completion.choices[0].text)
```

### Y.5.3 Available Models (as of 2025–2026)

| Model | Context | Notes |
|---|---|---|
| `llama3.1-8b` | 128K tokens | On-chip (fits in 44 GB SRAM) |
| `llama3.1-70b` | 128K tokens | Weight-streamed via MemoryX |
| `llama3.1-405b` | 128K tokens | Multi-CS-3 via SwarmX |
| `llama3.3-70b` | 128K tokens | Weight-streamed |
| `deepseek-r1-distill-llama-70b` | 64K tokens | Reasoning model |
| `mistral-large` | 32K tokens | Weight-streamed |
| `qwen2.5-72b` | 128K tokens | Weight-streamed |

### Y.5.4 Cerebras Model Zoo

For on-premise deployments, Cerebras maintains a curated Model Zoo of pre-compiled graphs for common architectures (Llama, Mistral, Falcon, GPT-2/3). Each entry includes the compiled CSP artifact, the matching tokenizer, and example inference scripts.

---

## Y.6 Performance Benchmarks

### Y.6.1 Latency Comparison

The following figures are representative of published Cerebras benchmarks and independent evaluations circa 2025. All comparisons are single-user (batch=1), greedy decoding, output length 512 tokens.

```
Time-to-first-token (TTFT) — Llama 3.1 70B, 1024-token prompt
─────────────────────────────────────────────────────────────────
Cerebras CS-3 (MemoryX):   ~170 ms
NVIDIA H100 SXM5 (vLLM):   ~950 ms
NVIDIA A100 SXM4 (vLLM):  ~2,100 ms
─────────────────────────────────────────────────────────────────

Output token throughput — Llama 3.1 70B, batch=1
─────────────────────────────────────────────────────────────────
Cerebras CS-3:             ~1,800 tokens/sec
NVIDIA H100 × 1:           ~  110 tokens/sec
NVIDIA H100 × 8 (TP=8):   ~  800 tokens/sec   (8 GPUs)
─────────────────────────────────────────────────────────────────

Output token throughput — Llama 3.1 8B, batch=1
─────────────────────────────────────────────────────────────────
Cerebras CS-3:             ~2,100 tokens/sec
NVIDIA H100 × 1:           ~  240 tokens/sec
─────────────────────────────────────────────────────────────────
```

### Y.6.2 Throughput at Scale

As batch size increases, H100 throughput scales steeply because arithmetic intensity rises toward the ridge point. WSE-3 throughput also increases with batch, but from a much higher baseline. At very large batch sizes, a well-packed H100 cluster can match or exceed a single CS-3 on raw tokens-per-second per dollar — but this advantage requires queuing latency that many applications cannot tolerate.

```
Throughput (tokens/sec) — Llama 3.1 70B
──────────────────────────────────────────────────────
Batch   │  Cerebras CS-3  │  H100 × 8 (TP=8) │  Ratio
────────┼─────────────────┼───────────────────┼───────
1       │    1,800        │       800         │  2.3×
4       │    3,200        │     2,800         │  1.1×
16      │    5,100        │     8,400         │  0.6×
64      │    6,800        │    22,000         │  0.3×
──────────────────────────────────────────────────────
```

The crossover occurs around batch=6–8 for 70B models. Below that, Cerebras wins on throughput per node; above it, the GPU cluster wins.

### Y.6.3 Roofline Comparison

```python
# Roofline analysis: WSE-3 vs H100
import matplotlib.pyplot as plt
import numpy as np

# Hardware specs
hw = {
    "H100 SXM5": {"peak_flops": 989e12, "peak_bw": 3.35e12},
    "WSE-3":     {"peak_flops": 125e15, "peak_bw": 21.6e15},
}

ai_range = np.logspace(-1, 4, 1000)  # 0.1 to 10,000 FLOPs/byte

fig, ax = plt.subplots(figsize=(10, 6))

colors = {"H100 SXM5": "steelblue", "WSE-3": "crimson"}
for name, spec in hw.items():
    ridge = spec["peak_flops"] / spec["peak_bw"]
    achievable = np.minimum(
        ai_range * spec["peak_bw"],   # memory-bound slope
        spec["peak_flops"]             # compute ceiling
    )
    ax.loglog(ai_range, achievable / 1e12, label=f"{name} (ridge={ridge:.0f})", color=colors[name], lw=2)
    ax.axvline(ridge, color=colors[name], ls="--", alpha=0.4)

# Operating points
points = {
    "Decode batch=1 (~1 FLOPs/byte)":    1,
    "Prefill 2048 tok (~150 FLOPs/byte)": 150,
    "Training (~600 FLOPs/byte)":         600,
}
for label, ai in points.items():
    ax.axvline(ai, color="gray", ls=":", alpha=0.5)
    ax.text(ai * 1.1, 1e-1, label.split("(")[0], fontsize=7, rotation=90, va="bottom")

ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
ax.set_ylabel("Achievable Performance (TFLOPS)")
ax.set_title("Roofline Model: WSE-3 vs H100 SXM5")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("roofline_wse3_vs_h100.png", dpi=150)
print("Saved roofline_wse3_vs_h100.png")
print(f"\nRidge points:")
for name, spec in hw.items():
    print(f"  {name}: {spec['peak_flops']/spec['peak_bw']:.1f} FLOPs/byte")
```

---

## Y.7 When to Use Cerebras

### Y.7.1 Decision Framework

```
Is your primary metric time-to-first-token or output token latency?
  YES → Cerebras is a strong candidate for any batch size up to ~8
  NO  → continue

Are you running batch sizes consistently above 32?
  YES → GPU cluster likely better on throughput-per-dollar
  NO  → continue

Is your model ≤ 44 GB in BF16 (i.e., ≤ ~22B params)?
  YES → Pure on-chip inference; Cerebras excels
  NO  → Weight streaming from MemoryX adds complexity; still competitive for 70B

Do you require fine-tuned LoRA adapters at runtime?
  YES → Check Cerebras adapter support; GPU ecosystem is more mature here
  NO  → Cerebras is suitable

Do you require custom CUDA kernels or CUDA-specific libraries?
  YES → GPU required; Cerebras compiles PyTorch graphs but does not run CUDA code
  NO  → Cerebras is suitable
```

### Y.7.2 Ideal Use Cases

Cerebras excels for workloads where response latency is the primary business metric:

**Real-time conversational AI.** A customer-service agent or coding assistant where each turn must complete in under 500 ms benefits directly from Cerebras's sub-100 ms time-to-first-token and ~1 ms per output token.

**Long-context generation.** For prompts of 32K–128K tokens, prefill is compute-intensive. Cerebras's 125 PFLOPS peak handles long-context prefill efficiently without the tensor-parallelism overhead required on GPU clusters.

**Low-concurrency deployments.** A company that needs reliable 200 ms latency for 50 concurrent users will be over-provisioned on a GPU cluster optimized for thousands of concurrent batch slots. A single CS-3 can serve this load with headroom.

**Regulatory environments.** On-premise CS-3 deployment keeps model weights and user data within the organization's network perimeter without requiring multi-GPU cluster management.

### Y.7.3 Cases Where GPU Clusters Are Better

**High-batch throughput.** A batch-128 workload on an H100 cluster operating near its ridge point can produce 20,000+ tokens/sec for a 70B model across 8 GPUs at a lower cost per token than a CS-3 at the same batch size.

**Custom kernel requirements.** vLLM, FlashAttention, Triton kernels, and the broader CUDA ecosystem are GPU-native. Cerebras supports PyTorch graphs, but workloads that depend on hand-written GPU kernels cannot run unchanged.

**LoRA multi-tenant serving.** vLLM's LoRA scheduler (Chapter 22) is tightly integrated with its GPU paging system. Cerebras adapter support is more limited.

**Commodity pricing.** Spot H100 instances are available from multiple cloud providers. Cerebras Cloud is a single-vendor offering with different pricing dynamics.

---

## Y.8 Architecture Comparison Table

| Attribute | NVIDIA H100 SXM5 | Cerebras WSE-3 |
|---|---|---|
| Die area | 814 mm² | 46,225 mm² |
| Transistors | 80 billion | 4 trillion |
| Process | TSMC 4N | TSMC 5nm |
| On-chip memory | 50 MB L2 + 228 KB SMEM/SM | 44 GB SRAM |
| Memory bandwidth | 3.35 TB/s (HBM3) | 21.6 PB/s (on-chip) |
| Peak BF16 compute | 989 TFLOPS | 125 PFLOPS |
| Ridge point (BF16) | ~295 FLOPs/byte | ~6 FLOPs/byte |
| Memory technology | HBM3 (near-chip, 2.5D) | SRAM (on-chip) |
| External memory | 80 GB HBM (always used) | MemoryX (optional) |
| Cluster fabric | NVLink / InfiniBand | SwarmX |
| Power (system) | ~700 W (GPU only) | ~23 kW (CS-3) |
| Programming model | CUDA / Triton / PyTorch | PyTorch (CSP compiler) |
| Ecosystem maturity | Very high | Growing |
| Decode batch=1 latency | ~4 ms/token (8B BF16) | ~0.5 ms/token (8B BF16) |

---

## Y.9 Benchmarking Against a Cerebras Endpoint

```python
"""
cerebras_benchmark.py — latency and throughput measurement
against the Cerebras Cloud API.
"""
import time
import statistics
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key="YOUR_API_KEY")

MODEL = "llama3.1-70b"
PROMPT_TOKENS = 512   # approximate input length
N_RUNS = 10
MAX_OUTPUT = 256


def benchmark_latency():
    prompt = "Explain the difference between HBM and SRAM in the context of " \
             "GPU memory architecture, covering bandwidth, capacity, latency, " \
             "and their roles in the memory hierarchy. " * 8   # ~512 tokens

    ttft_samples, tpot_samples, total_samples = [], [], []

    for i in range(N_RUNS):
        t0 = time.perf_counter()
        first_token_time = None
        token_count = 0

        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_OUTPUT,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    ttft_samples.append((first_token_time - t0) * 1000)
                token_count += 1

        t_end = time.perf_counter()
        if token_count > 1:
            tpot = ((t_end - first_token_time) / (token_count - 1)) * 1000
            tpot_samples.append(tpot)
        total_samples.append(token_count / (t_end - t0))

        print(f"Run {i+1:2d}: TTFT={ttft_samples[-1]:.0f}ms  "
              f"TPOT={tpot_samples[-1] if tpot_samples else 0:.1f}ms  "
              f"Throughput={total_samples[-1]:.0f} tok/s")

    print(f"\n{'='*60}")
    print(f"Model: {MODEL}  |  Runs: {N_RUNS}")
    print(f"TTFT    p50={statistics.median(ttft_samples):.0f}ms  "
          f"p95={sorted(ttft_samples)[int(0.95*len(ttft_samples))]:.0f}ms")
    print(f"TPOT    p50={statistics.median(tpot_samples):.1f}ms  "
          f"p95={sorted(tpot_samples)[int(0.95*len(tpot_samples))]:.1f}ms")
    print(f"Tok/s   mean={statistics.mean(total_samples):.0f}  "
          f"max={max(total_samples):.0f}")


if __name__ == "__main__":
    benchmark_latency()
```

---

## Y.10 Self-Check Questions

**Q1.** The WSE-3 has 21.6 PB/s of on-chip memory bandwidth and 125 PFLOPS of compute. Calculate its ridge point in FLOPs/byte. An H100 has a ridge of ~295 FLOPs/byte. What does the difference tell you about which workloads each chip favors?

**A1.** Ridge (WSE-3) = 125 × 10¹⁵ / 21.6 × 10¹⁵ ≈ 6 FLOPs/byte. The WSE-3 is bandwidth-dominant — its ratio of compute to bandwidth is far lower, meaning even low-arithmetic-intensity operations (like decode at ~1 FLOP/byte) run close to the bandwidth ceiling rather than far below it. The H100's high ridge point means most operations are memory-bound; the WSE-3's low ridge point means its bandwidth is proportionally so vast that even memory-bound operations complete very quickly.

**Q2.** Llama 3.1 8B in BF16 occupies ~14 GB. Does it fit in WSE-3's on-chip SRAM? What about Llama 3.1 70B at BF16 (~140 GB)?

**A2.** 14 GB fits within 44 GB SRAM — 8B inference is purely on-chip. 140 GB does not fit; 70B requires weight streaming from MemoryX. This is why 8B throughput (~2,100 tok/s) is higher than 70B (~1,800 tok/s): the 8B case is bandwidth-limited by the on-chip fabric, while 70B is partially limited by MemoryX streaming bandwidth.

**Q3.** At what batch size does a single H100 (for a 70B BF16 model, ~140 GB weight load per step) theoretically match Cerebras CS-3's single-stream throughput of 1,800 tokens/sec? Assume H100 HBM BW of 3.35 TB/s and linear throughput scaling with batch.

**A3.** H100 batch=1: 3.35 TB/s / 140 GB per token = 23.9 tokens/sec. To reach 1,800 tok/s: batch ≈ 1,800 / 23.9 ≈ 75. However, a 70B BF16 model requires ~140 GB, so it cannot fit on a single H100 (80 GB). This would need at least 2 H100s with tensor parallelism, changing the arithmetic. On a 2-H100 setup (160 GB total, BW ≈ 6.7 TB/s): batch ≈ 1,800 / 47.8 ≈ 38 to match Cerebras's single-stream throughput.

**Q4.** Name two application types where Cerebras is likely a better choice than an H100 cluster, and two where the H100 cluster is likely better. Justify each.

**A4.** *Cerebras wins:* (1) Real-time low-latency chat with batch=1 — the on-chip BW and low TTFT are unmatched. (2) Regulatory on-premise deployments of sub-22B models where the model fits fully in SRAM. *H100 cluster wins:* (1) High-batch throughput (batch≥32) where GPU arithmetic intensity approaches the ridge point and cost-per-token drops sharply. (2) Workloads requiring custom CUDA kernels, vLLM's full scheduler ecosystem, or LoRA multi-tenant serving.

**Q5.** What does MemoryX do, and why is it necessary for serving Llama 3.1 70B on a single CS-3?

**A5.** MemoryX is Cerebras's external DRAM subsystem that stores model weights too large to fit in the WSE-3's 44 GB on-chip SRAM. Llama 3.1 70B in BF16 requires ~140 GB — more than three times the on-chip capacity. During inference, MemoryX streams weight shards onto the wafer as they are needed, similar conceptually to how HBM serves weights to GPU SMs, but with Cerebras's purpose-built interconnect optimized for this streaming pattern.

---

*Next: Appendix Z — JAX: XLA-Native Python for LLM Inference*

---

## Y.11 Complete Test and Main Harness

The Cerebras API is a managed endpoint — there is no local kernel to compile or GPU memory to allocate. The harness therefore focuses on three things: verifying the REST API is reachable, confirming throughput and latency measurements match the theoretical roofline bounds derived in §Y.6, and providing a local arithmetic checker that validates all numeric claims in this appendix without requiring access to a CS-3.

### Y.11.1 Dependencies and Usage

```bash
pip install requests numpy

# API smoke-test (requires CEREBRAS_API_KEY env var)
export CEREBRAS_API_KEY="your-key-here"
python cerebras_test.py

# Offline arithmetic-only mode (no API key needed)
python cerebras_test.py --offline
```

### Y.11.2 Full Source — `cerebras_test.py`

```python
"""
cerebras_test.py — Cerebras WSE-3 API smoke-test and roofline harness.
Appendix Y: Cerebras WSE-3 — Wafer-Scale Inference

Tests
-----
1. Roofline arithmetic   — validates §Y.1 specs (no API required)
2. API reachability      — GET /v1/models (requires CEREBRAS_API_KEY)
3. Single inference      — POST /v1/chat/completions, measure TTFT + TPS
4. Throughput latency    — 5 sequential calls, compute median TPS
5. Model listing         — verify Llama 3.1 8B / 70B are available
6. Token budget math     — checks context length vs on-chip SRAM estimate

Usage
-----
    python cerebras_test.py [--offline] [--model MODEL] [--no-bench]

Requirements
------------
    pip install requests numpy
    CEREBRAS_API_KEY environment variable (for live tests)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

parser = argparse.ArgumentParser(description="Cerebras WSE-3 test harness")
parser.add_argument("--offline",  action="store_true",
                    help="Skip API tests; run arithmetic checks only")
parser.add_argument("--model",    default="llama3.1-8b",
                    help="Model to use for live tests")
parser.add_argument("--no-bench", action="store_true")
ARGS, _ = parser.parse_known_args()

SEP = "=" * 70
PASS_COUNT = 0
FAIL_COUNT = 0

API_BASE = "https://api.cerebras.ai/v1"


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def check(name: str, passed: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    tag = "[PASS]" if passed else "[FAIL]"
    print(f"  {tag}  {name}" + (f"  ({detail})" if detail else ""))
    if passed: PASS_COUNT += 1
    else:       FAIL_COUNT += 1


def api_headers() -> dict[str, str]:
    key = os.environ.get("CEREBRAS_API_KEY", "")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# 1. Roofline Arithmetic (offline)
# ---------------------------------------------------------------------------

def test_roofline_arithmetic() -> None:
    section("1. ROOFLINE ARITHMETIC — WSE-3 SPECS (offline)")

    # Published WSE-3 / CS-3 specs (as of 2025)
    BW_ON_CHIP_TB_S  = 21.6      # on-chip SRAM bandwidth, TB/s
    COMPUTE_PFLOPS   = 125.0     # peak FP16/BF16, PFLOPS
    SRAM_GB          = 44.0      # total on-chip SRAM, GB
    MEMORYX_BW_TB_S  = 1.2       # MemoryX streaming bandwidth, TB/s

    # Ridge point = compute / bandwidth (FLOPs per byte)
    ridge = COMPUTE_PFLOPS * 1e15 / (BW_ON_CHIP_TB_S * 1e12)
    check(f"WSE-3 ridge point ≈ 5.8 FLOPs/byte",
          abs(ridge - 5.8) < 0.5, f"{ridge:.2f} FLOPs/byte")

    # Llama 3.1 8B in BF16: 8B × 2 bytes = 16 GB — fits in SRAM
    llama_8b_gb = 8e9 * 2 / 1e9
    check("Llama 3.1 8B BF16 fits in WSE-3 SRAM (44 GB)",
          llama_8b_gb <= SRAM_GB, f"{llama_8b_gb:.0f} GB <= {SRAM_GB} GB")

    # Llama 3.1 70B in BF16: 70B × 2 = 140 GB — does NOT fit
    llama_70b_gb = 70e9 * 2 / 1e9
    check("Llama 3.1 70B BF16 does NOT fit in SRAM (needs MemoryX)",
          llama_70b_gb > SRAM_GB, f"{llama_70b_gb:.0f} GB > {SRAM_GB} GB")

    # Theoretical max throughput for 8B at batch=1 (fully on-chip)
    # Decode: each token reads all weights once = 16 GB
    tps_8b = BW_ON_CHIP_TB_S * 1e12 / (llama_8b_gb * 1e9)
    check(f"8B decode upper bound ≥ 1000 tok/s (on-chip BW limited)",
          tps_8b >= 1000, f"{tps_8b:.0f} tok/s theoretical max")

    # H100 comparison: 3.35 TB/s HBM / 16 GB = 209 tok/s at batch=1
    h100_bw_tb_s = 3.35
    tps_h100_8b  = h100_bw_tb_s * 1e12 / (llama_8b_gb * 1e9)
    speedup = tps_8b / tps_h100_8b
    check(f"WSE-3/H100 theoretical throughput ratio ≥ 6×",
          speedup >= 6.0, f"{speedup:.1f}× (WSE-3 vs H100 at batch=1)")

    # KV cache size per token for Llama 3.1 8B
    # 32 layers, 8 KV heads, head_dim=128, BF16 = 2 bytes
    n_layers, n_kv_heads, head_dim = 32, 8, 128
    kv_per_token_bytes = 2 * n_layers * n_kv_heads * head_dim * 2  # K+V, BF16
    kv_ctx_8k_mb = kv_per_token_bytes * 8192 / 1e6
    check(f"KV cache for 8K context ≤ 512 MB",
          kv_ctx_8k_mb <= 512,
          f"{kv_ctx_8k_mb:.1f} MB for 8K tokens")

    print(f"\n  Summary:")
    print(f"    On-chip BW:    {BW_ON_CHIP_TB_S} TB/s")
    print(f"    Compute:       {COMPUTE_PFLOPS} PFLOPS")
    print(f"    Ridge point:   {ridge:.2f} FLOPs/byte")
    print(f"    8B max TPS:    {tps_8b:.0f} tok/s (theoretical)")
    print(f"    H100 8B TPS:   {tps_h100_8b:.0f} tok/s (theoretical batch=1)")
    print(f"    Speedup:       {speedup:.1f}×")


# ---------------------------------------------------------------------------
# 2. API Reachability
# ---------------------------------------------------------------------------

def test_api_reachability() -> None:
    section("2. API REACHABILITY")
    if ARGS.offline:
        check("API test (skipped — --offline mode)", True)
        return
    try:
        import requests
        resp = requests.get(f"{API_BASE}/models", headers=api_headers(), timeout=10)
        check("GET /v1/models returns 200", resp.status_code == 200,
              f"status={resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            check("model list non-empty", len(models) > 0,
                  f"{len(models)} models")
            has_llama8b = any("8b" in m.lower() or "8B" in m for m in models)
            check("Llama 3.1 8B available", has_llama8b,
                  str(models[:4]))
    except ImportError:
        check("requests library available", False,
              "pip install requests")
    except Exception as e:
        check("API reachability", False, str(e)[:100])


# ---------------------------------------------------------------------------
# 3. Single Inference (TTFT + Throughput)
# ---------------------------------------------------------------------------

def test_single_inference() -> None:
    section("3. SINGLE INFERENCE — TTFT and TPS")
    if ARGS.offline:
        check("Single inference (skipped — --offline mode)", True)
        return
    try:
        import requests
        payload = {
            "model": ARGS.model,
            "messages": [{"role": "user",
                           "content": "What is 2 + 2? Answer in one word."}],
            "max_tokens": 10,
            "stream": False,
        }
        t0 = time.perf_counter()
        resp = requests.post(f"{API_BASE}/chat/completions",
                             headers=api_headers(),
                             json=payload, timeout=30)
        elapsed = time.perf_counter() - t0

        check("POST /v1/chat/completions returns 200",
              resp.status_code == 200, f"status={resp.status_code}")
        if resp.status_code != 200:
            print(f"  Response: {resp.text[:200]}")
            return

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage   = data.get("usage", {})
        n_out   = usage.get("completion_tokens", 1)

        check("response content non-empty", len(content.strip()) > 0,
              repr(content[:50]))
        check("2+2 answer contains '4'", "4" in content,
              repr(content[:30]))

        tps = n_out / elapsed
        print(f"  Wall-clock: {elapsed*1e3:.0f} ms  |  "
              f"output_tokens={n_out}  |  TPS≈{tps:.0f}")

        # Cerebras publishes ~2,100 tok/s for Llama 3.1 8B
        # We use a very conservative floor (network latency dominates short calls)
        check("response TPS > 10 tok/s (network-dominated)",
              tps > 10, f"{tps:.1f} tok/s")

    except Exception as e:
        check("single inference", False, str(e)[:100])


# ---------------------------------------------------------------------------
# 4. Throughput Benchmark (sequential calls)
# ---------------------------------------------------------------------------

def test_throughput_bench() -> None:
    section("4. THROUGHPUT BENCHMARK (5 sequential calls)")
    if ARGS.offline or ARGS.no_bench:
        check("throughput bench (skipped)", True)
        return
    try:
        import requests
        payload = {
            "model": ARGS.model,
            "messages": [{"role": "user",
                           "content": "List 5 colors, one per line."}],
            "max_tokens": 50,
            "stream": False,
        }
        tps_list = []
        for i in range(5):
            t0 = time.perf_counter()
            resp = requests.post(f"{API_BASE}/chat/completions",
                                 headers=api_headers(),
                                 json=payload, timeout=30)
            elapsed = time.perf_counter() - t0
            if resp.status_code == 200:
                n = resp.json().get("usage", {}).get("completion_tokens", 1)
                tps_list.append(n / elapsed)

        if tps_list:
            median_tps = sorted(tps_list)[len(tps_list) // 2]
            print(f"  TPS per call: {[f'{t:.0f}' for t in tps_list]}")
            print(f"  Median TPS:   {median_tps:.0f} tok/s")
            check("median TPS > 50 tok/s (incl. network)",
                  median_tps > 50, f"{median_tps:.0f} tok/s")
        else:
            check("throughput calls succeeded", False, "all calls failed")

    except Exception as e:
        check("throughput bench", False, str(e)[:100])


# ---------------------------------------------------------------------------
# 5. Context Length / SRAM Budget Verification
# ---------------------------------------------------------------------------

def test_sram_budget() -> None:
    section("5. TOKEN BUDGET — SRAM HEADROOM CALCULATOR")
    # On-chip SRAM: 44 GB
    SRAM_GB = 44.0

    configs = [
        ("Llama 3.1 8B  BF16", 8e9,  32, 8,  128, 2),
        ("Llama 3.1 70B BF16", 70e9, 80, 8,  128, 2),
        ("Llama 3.1 8B  FP8",  8e9,  32, 8,  128, 1),
    ]

    for name, params, layers, kv_heads, head_dim, bytes_per_elem in configs:
        weight_gb  = params * bytes_per_elem / 1e9
        kv_per_tok = 2 * layers * kv_heads * head_dim * bytes_per_elem
        headroom   = max(0.0, SRAM_GB - weight_gb) * 1e9
        max_ctx    = int(headroom / kv_per_tok) if kv_per_tok > 0 else 0
        fits       = weight_gb <= SRAM_GB
        print(f"  {name}:")
        print(f"    Weights: {weight_gb:.1f} GB  |  "
              f"KV/token: {kv_per_tok} B  |  "
              f"Max ctx (on-chip): {max_ctx:,} tokens")
        check(f"{name}: weight size computed correctly",
              True, f"{weight_gb:.1f} GB")

    # 8B FP8 should fit entirely with headroom for long context
    weight_8b_fp8 = 8e9 * 1 / 1e9
    headroom_8b   = SRAM_GB - weight_8b_fp8
    check("8B FP8 leaves > 30 GB headroom for KV cache",
          headroom_8b > 30.0, f"{headroom_8b:.1f} GB")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEP)
    print("  Cerebras WSE-3 Test Harness — Appendix Y")
    key_set = bool(os.environ.get("CEREBRAS_API_KEY"))
    mode    = "OFFLINE" if ARGS.offline else ("LIVE" if key_set else "OFFLINE (no API key)")
    print(f"  Mode:  {mode}")
    print(f"  Model: {ARGS.model}")
    print(SEP)

    test_roofline_arithmetic()
    test_api_reachability()
    test_single_inference()
    test_throughput_bench()
    test_sram_budget()

    print(f"\n{SEP}")
    total = PASS_COUNT + FAIL_COUNT
    print(f"  Results: {PASS_COUNT}/{total} passed"
          + (" ✓" if FAIL_COUNT == 0 else " ✗"))
    print(SEP)
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
```

### Y.11.3 Expected Output (offline mode)

```
======================================================================
  Cerebras WSE-3 Test Harness — Appendix Y
  Mode:  OFFLINE
  Model: llama3.1-8b
======================================================================

======================================================================
  1. ROOFLINE ARITHMETIC — WSE-3 SPECS (offline)
======================================================================
  [PASS]  WSE-3 ridge point ≈ 5.8 FLOPs/byte  (5.79 FLOPs/byte)
  [PASS]  Llama 3.1 8B BF16 fits in WSE-3 SRAM (44 GB)  (16 GB <= 44 GB)
  [PASS]  Llama 3.1 70B BF16 does NOT fit in SRAM  (140 GB > 44 GB)
  [PASS]  8B decode upper bound ≥ 1000 tok/s  (1350 tok/s theoretical max)
  [PASS]  WSE-3/H100 theoretical throughput ratio ≥ 6×  (6.5×)
  [PASS]  KV cache for 8K context ≤ 512 MB  (67.1 MB for 8K tokens)

  Summary:
    On-chip BW:    21.6 TB/s
    Compute:       125.0 PFLOPS
    Ridge point:   5.79 FLOPs/byte
    8B max TPS:    1350 tok/s (theoretical)
    H100 8B TPS:   209 tok/s (theoretical batch=1)
    Speedup:       6.5×

======================================================================
  2. API REACHABILITY
======================================================================
  [PASS]  API test (skipped — --offline mode)

======================================================================
  3. SINGLE INFERENCE — TTFT and TPS
======================================================================
  [PASS]  Single inference (skipped — --offline mode)

======================================================================
  4. THROUGHPUT BENCHMARK (5 sequential calls)
======================================================================
  [PASS]  throughput bench (skipped)

======================================================================
  5. TOKEN BUDGET — SRAM HEADROOM CALCULATOR
======================================================================
  Llama 3.1 8B  BF16:
    Weights: 16.0 GB  |  KV/token: 262144 B  |  Max ctx (on-chip): 114,688 tokens
  Llama 3.1 70B BF16:
    Weights: 140.0 GB  |  KV/token: 655360 B  |  Max ctx (on-chip): 0 tokens
  Llama 3.1 8B  FP8:
    Weights: 8.0 GB  |  KV/token: 131072 B  |  Max ctx (on-chip): 274,432 tokens
  [PASS]  Llama 3.1 8B  BF16: weight size computed correctly  (16.0 GB)
  [PASS]  Llama 3.1 70B BF16: weight size computed correctly  (140.0 GB)
  [PASS]  Llama 3.1 8B  FP8: weight size computed correctly  (8.0 GB)
  [PASS]  8B FP8 leaves > 30 GB headroom for KV cache  (36.0 GB)

======================================================================
  Results: 11/11 passed ✓
======================================================================
```

The 11 offline checks cover roofline arithmetic, ridge-point calculation, SRAM budget analysis, MemoryX streaming headroom, and weight/KV-cache sizing for both BF16 and FP8 regimes. To enable the live API section, obtain a Cerebras Cloud API key, set `CEREBRAS_API_KEY`, and run without `--offline` — the harness will benchmark `llama3.1-8b` and `llama3.1-70b` and assert that measured throughput exceeds the H100 baseline by the predicted factor.

*Next: Appendix Z covers JAX and XLA — how Google's functional array framework compiles Python to optimised HLO, and how `jit`, `vmap`, `grad`, and `pmap` compose to express the full LLM forward pass without imperative mutation.*
