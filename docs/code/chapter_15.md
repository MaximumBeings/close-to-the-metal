# Chapter 15: Multi-GPU Serving — Companion Code

## Python — `multigpu_demo.py`

```python
"""
Chapter 15 Companion Code: Multi-GPU Serving and Tensor Parallelism
=====================================================================
Demonstrates:
  1. Weight shard calculator — bytes per GPU at each TP level
  2. KV cache sizing under TP — tokens per GPU, total pool
  3. AllReduce latency model — NVLink vs PCIe vs InfiniBand
  4. Throughput vs GPU count — compute gain vs AllReduce tax
  5. Break-even batch size — when TP hurts vs helps
  6. Scale-up vs scale-out decision table
  7. Ring-AllReduce step-by-step trace (small N)
  8. llama.cpp tensor-split memory estimator

Run:
    python3 multigpu_demo.py

No external dependencies.
"""

import math
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────────────
# Data classes (reused from ch14 pattern)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUSpec:
    name: str
    hbm_gb: float
    bw_tb_s: float          # memory bandwidth TB/s
    tflops_bf16: float      # peak BF16 TFLOPS
    nvlink_bw_gbs: float    # NVLink bidirectional GB/s (0 = PCIe only)
    pcie_bw_gbs: float      # PCIe bidirectional GB/s


@dataclass
class ModelSpec:
    name: str
    params_b: float
    layers: int
    hidden_dim: int
    num_heads: int
    kv_heads: int
    head_dim: int
    ffn_dim: int
    dtype_bytes: int = 2    # BF16


# ──────────────────────────────────────────────────────────────────────────────
# GPU / model catalogue
# ──────────────────────────────────────────────────────────────────────────────

GPUS = {
    "A100-40":  GPUSpec("A100-40",   40.0, 1.56, 312,  600, 32),
    "A100-80":  GPUSpec("A100-80",   80.0, 2.00, 312,  600, 32),
    "H100-80":  GPUSpec("H100-80",   80.0, 3.35, 989,  900, 64),
    "H200-141": GPUSpec("H200-141", 141.0, 4.80, 989,  900, 64),
    "RTX4090":  GPUSpec("RTX4090",   24.0, 1.01, 165,    0, 32),  # no NVLink
    "RTX3090":  GPUSpec("RTX3090",   24.0, 0.94,  71,    0, 32),
}

MODELS = {
    "Llama-3-8B":  ModelSpec("Llama-3-8B",   8,  32, 4096,  32,  8, 128, 14336),
    "Llama-3-70B": ModelSpec("Llama-3-70B",  70,  80, 8192,  64,  8, 128, 28672),
    "Llama-3.1-405B": ModelSpec("Llama-3.1-405B", 405, 126, 16384, 128, 8, 128, 53248),
    "Mistral-7B":  ModelSpec("Mistral-7B",  7.3,  32, 4096,  32,  8, 128, 14336),
}

MODEL_OVERHEAD_GB = 1.5   # CUDA context, activations

# ──────────────────────────────────────────────────────────────────────────────
# 1. WEIGHT SHARD CALCULATOR
# ──────────────────────────────────────────────────────────────────────────────

def weights_per_gpu_gb(model: ModelSpec, tp: int) -> float:
    total_bytes = model.params_b * 1e9 * model.dtype_bytes
    return total_bytes / (1024**3) / tp


def kv_bytes_per_token_per_gpu(model: ModelSpec, tp: int) -> int:
    kv_heads_per_gpu = max(1, model.kv_heads // tp)
    return 2 * model.layers * kv_heads_per_gpu * model.head_dim * model.dtype_bytes


def kv_pool_gb(gpu: GPUSpec, model: ModelSpec, tp: int,
               gpu_util: float = 0.90) -> float:
    usable = gpu.hbm_gb * gpu_util
    w_gb   = weights_per_gpu_gb(model, tp)
    pool   = usable - w_gb - MODEL_OVERHEAD_GB
    return max(0.0, pool)


def kv_total_tokens(gpu: GPUSpec, model: ModelSpec, tp: int,
                    gpu_util: float = 0.90) -> int:
    pool_gb    = kv_pool_gb(gpu, model, tp, gpu_util)
    bytes_tok  = kv_bytes_per_token_per_gpu(model, tp)
    # Total tokens = sum across all TP ranks
    return int(pool_gb * 1024**3 / bytes_tok) * tp


def print_weight_shard_table(gpu: GPUSpec, model: ModelSpec,
                              tp_options: list[int] = None):
    if tp_options is None:
        # Valid TP values must divide kv_heads
        tp_options = [tp for tp in [1, 2, 4, 8, 16]
                      if model.kv_heads % tp == 0]
    print(f"\n{'='*80}")
    print(f"  Weight Shard + KV Pool Calculator  |  {model.name}  |  {gpu.name}")
    print(f"{'='*80}")
    print(f"  {'TP':>4}  {'Weight/GPU':>12}  {'Fits?':>6}  "
          f"{'KV B/tok':>10}  {'KV Pool/GPU':>13}  {'Total KV Tok':>14}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*6}  {'─'*10}  {'─'*13}  {'─'*14}")
    for tp in tp_options:
        w_gb   = weights_per_gpu_gb(model, tp)
        fits   = "YES" if w_gb < gpu.hbm_gb * 0.90 - MODEL_OVERHEAD_GB else "NO "
        kv_bt  = kv_bytes_per_token_per_gpu(model, tp)
        kp_gb  = kv_pool_gb(gpu, model, tp)
        tot_t  = kv_total_tokens(gpu, model, tp)
        print(f"  {tp:>4}  {w_gb:>10.1f}GB  {fits:>6}  "
              f"{kv_bt:>10,}  {kp_gb:>11.1f}GB  {tot_t:>14,}")
    print(f"{'='*80}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. ALLREDUCE LATENCY MODEL
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Interconnect:
    name: str
    bw_gb_s: float      # bidirectional GB/s
    latency_us: float   # base latency µs (alpha in alpha-beta model)


INTERCONNECTS = {
    "NVLink3 (A100)":    Interconnect("NVLink3 (A100)",    600, 1.0),
    "NVLink4 (H100)":    Interconnect("NVLink4 (H100)",    900, 1.0),
    "PCIe4 ×16":         Interconnect("PCIe4 ×16",          32, 5.0),
    "PCIe5 ×16":         Interconnect("PCIe5 ×16",          64, 3.0),
    "InfiniBand HDR":    Interconnect("InfiniBand HDR",      25, 2.0),  # single-link
}


def allreduce_latency_ms(message_bytes: int, n_gpus: int,
                          link: Interconnect) -> float:
    """
    Ring-AllReduce latency: alpha-beta model.
    Transfer time = 2 × (n-1)/n × message_size / bandwidth
    Fixed latency = 2 × (n-1) × alpha  (2 phases, n-1 steps each)
    """
    n = n_gpus
    transfer_bytes = 2.0 * (n - 1) / n * message_bytes
    transfer_s     = transfer_bytes / (link.bw_gb_s * 1e9)
    fixed_s        = 2.0 * (n - 1) * (link.latency_us * 1e-6)
    return (transfer_s + fixed_s) * 1000   # ms


def allreduce_message_bytes(model: ModelSpec, batch_tokens: int) -> int:
    """Activation tensor size per AllReduce."""
    return batch_tokens * model.hidden_dim * model.dtype_bytes


def print_allreduce_table(model: ModelSpec, batch_tokens: int = 128):
    msg_bytes = allreduce_message_bytes(model, batch_tokens)
    print(f"\n{'='*72}")
    print(f"  AllReduce Latency  |  {model.name}  |  batch={batch_tokens} tokens")
    print(f"  Message size per AllReduce: {msg_bytes:,} bytes ({msg_bytes/1024:.0f} KB)")
    print(f"{'='*72}")
    print(f"  {'Interconnect':<22}  {'TP=2':>8}  {'TP=4':>8}  {'TP=8':>8}  {'TP=16':>9}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")
    for link_name, link in INTERCONNECTS.items():
        row = f"  {link_name:<22}"
        for tp in [2, 4, 8, 16]:
            lat = allreduce_latency_ms(msg_bytes, tp, link)
            row += f"  {lat:>6.2f}ms"
        print(row)
    print(f"{'='*72}")
    print(f"  Note: single transformer forward pass incurs 2×layers AllReduces")


# ──────────────────────────────────────────────────────────────────────────────
# 3. THROUGHPUT VS GPU COUNT
# ──────────────────────────────────────────────────────────────────────────────

def decode_step_latency_ms(gpu: GPUSpec, model: ModelSpec, tp: int,
                            batch_tokens: int, link: Interconnect) -> float:
    """
    Decode step latency = compute time + AllReduce time × 2L
    Compute: load weights from HBM (bandwidth-bound)
    AllReduce: 2 per layer × L layers
    """
    # Weight load time: all weights / (tp × gpu_bw)
    weight_bytes = model.params_b * 1e9 * model.dtype_bytes / tp
    compute_ms   = weight_bytes / (gpu.bw_tb_s * 1e12) * 1000

    # AllReduce cost
    msg_bytes  = allreduce_message_bytes(model, batch_tokens)
    ar_per_ms  = allreduce_latency_ms(msg_bytes, tp, link)
    allreduce_total_ms = 2 * model.layers * ar_per_ms

    return compute_ms + allreduce_total_ms


def print_throughput_vs_gpus(gpu: GPUSpec, model: ModelSpec,
                              batch_tokens: int = 128):
    print(f"\n{'='*78}")
    print(f"  Throughput vs GPU Count  |  {model.name}  |  {gpu.name}")
    print(f"  Batch = {batch_tokens} tokens/step")
    print(f"{'='*78}")

    # Determine interconnect from GPU spec
    nvlink = Interconnect("NVLink", gpu.nvlink_bw_gbs or 1, 1.0)
    pcie   = Interconnect("PCIe",   gpu.pcie_bw_gbs, 5.0)
    use_nvlink = gpu.nvlink_bw_gbs > 0

    link = nvlink if use_nvlink else pcie
    link_label = f"NVLink {gpu.nvlink_bw_gbs} GB/s" if use_nvlink else f"PCIe {gpu.pcie_bw_gbs} GB/s"
    print(f"  Interconnect: {link_label}")
    print(f"\n  {'TP':>4}  {'Step (ms)':>10}  {'Tok/s':>8}  {'vs TP=1':>9}  "
          f"{'AR overhead':>13}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*13}")

    # Baseline (TP=1, no AllReduce)
    base_weight_bytes = model.params_b * 1e9 * model.dtype_bytes
    base_ms = base_weight_bytes / (gpu.bw_tb_s * 1e12) * 1000
    base_toks = batch_tokens / (base_ms / 1000)

    valid_tps = [tp for tp in [1, 2, 4, 8]
                 if model.kv_heads % tp == 0]

    for tp in valid_tps:
        if tp == 1:
            step_ms = base_ms
            ar_pct  = 0.0
        else:
            step_ms = decode_step_latency_ms(gpu, model, tp, batch_tokens, link)
            compute_ms = (model.params_b * 1e9 * model.dtype_bytes / tp) / (gpu.bw_tb_s * 1e12) * 1000
            ar_total   = step_ms - compute_ms
            ar_pct     = ar_total / step_ms * 100

        toks_per_s = batch_tokens / (step_ms / 1000)
        speedup    = toks_per_s / base_toks
        ar_str     = f"{ar_pct:5.1f}%" if tp > 1 else "  —"
        print(f"  {tp:>4}  {step_ms:>10.2f}  {toks_per_s:>8,.0f}  {speedup:>8.2f}×  {ar_str:>13}")

    print(f"{'='*78}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. BREAK-EVEN BATCH SIZE
# ──────────────────────────────────────────────────────────────────────────────

def break_even_batch(gpu: GPUSpec, model: ModelSpec, tp: int,
                     link: Interconnect) -> int:
    """
    Find the batch size where TP decode latency equals single-GPU latency.
    At this point the AllReduce tax is fully amortised.
    """
    # Single GPU compute per token (ms)
    single_w_bytes   = model.params_b * 1e9 * model.dtype_bytes
    single_ms_per_tok = single_w_bytes / (gpu.bw_tb_s * 1e12) * 1000 / 1  # per batch_tokens

    # TP compute: load (1/tp) of weights per GPU, but still needs all tokens
    # AllReduce: fixed per step independent of batch (for small batch)
    # Solve: batch/single_latency(batch) = batch/tp_latency(batch)
    # We'll scan batch sizes
    for B in range(1, 4096):
        single_ms = single_w_bytes / (gpu.bw_tb_s * 1e12) * 1000
        tp_ms     = decode_step_latency_ms(gpu, model, tp, B, link)
        if tp_ms < single_ms:
            return B
    return 4096   # never breaks even in tested range


def print_break_even_table(gpu: GPUSpec, model: ModelSpec):
    print(f"\n{'='*72}")
    print(f"  Break-Even Batch Size  |  {model.name}  |  {gpu.name}")
    print(f"  (batch size where TP decode latency < single-GPU latency)")
    print(f"{'='*72}")

    nvlink = Interconnect("NVLink", gpu.nvlink_bw_gbs or 1, 1.0)
    pcie   = Interconnect("PCIe",   gpu.pcie_bw_gbs, 5.0)

    print(f"  {'TP':>4}  {'NVLink break-even':>20}  {'PCIe break-even':>18}")
    print(f"  {'─'*4}  {'─'*20}  {'─'*18}")
    for tp in [2, 4, 8]:
        if model.kv_heads % tp != 0:
            continue
        if gpu.nvlink_bw_gbs > 0:
            nv_be = break_even_batch(gpu, model, tp, nvlink)
            nv_str = f"{nv_be} tokens"
        else:
            nv_str = "N/A (no NVLink)"
        pcie_be  = break_even_batch(gpu, model, tp, pcie)
        pcie_str = f"{pcie_be} tokens"
        print(f"  {tp:>4}  {nv_str:>20}  {pcie_str:>18}")
    print(f"{'='*72}")
    print(f"  Below break-even: single GPU is faster.")
    print(f"  Above break-even: TP reduces latency.")


# ──────────────────────────────────────────────────────────────────────────────
# 5. SCALE-UP vs SCALE-OUT DECISION
# ──────────────────────────────────────────────────────────────────────────────

def scale_decision(gpu: GPUSpec, model: ModelSpec, n_gpus: int,
                   target_concurrency: int, max_context_len: int,
                   gpu_util: float = 0.90) -> dict:
    """
    Compare TP=n_gpus (scale-up) vs n_gpus replicas (scale-out).
    Returns a comparison dict.
    """
    # Determine valid TP
    valid_tp = model.kv_heads % n_gpus == 0

    # Scale-up (TP)
    if valid_tp:
        tp_kv_total = kv_total_tokens(gpu, model, n_gpus, gpu_util)
        tp_max_sessions = tp_kv_total // max_context_len
    else:
        tp_kv_total     = 0
        tp_max_sessions = 0

    # Scale-out (replicas)
    single_kv = kv_total_tokens(gpu, model, 1, gpu_util) // 1  # single GPU KV
    replica_kv_per_inst = single_kv
    replica_max_sessions_per = replica_kv_per_inst // max_context_len
    replica_max_sessions = replica_max_sessions_per * n_gpus

    # Throughput: replicas scale linearly; TP gains bandwidth speedup offset by AR
    nvlink_ok = gpu.nvlink_bw_gbs > 0
    tp_label  = "TP={} (NVLink)" if nvlink_ok else "TP={} (PCIe)"

    return {
        "n_gpus":           n_gpus,
        "valid_tp":         valid_tp,
        "tp_kv_total_tok":  tp_kv_total,
        "tp_max_sessions":  tp_max_sessions,
        "replica_kv_each":  replica_kv_per_inst,
        "replica_max_sess": replica_max_sessions,
        "fits_single":      weights_per_gpu_gb(model, 1) < gpu.hbm_gb * gpu_util - MODEL_OVERHEAD_GB,
        "nvlink":           nvlink_ok,
    }


def print_scale_decision(gpu: GPUSpec, model: ModelSpec, n_gpus: int,
                          target_concurrency: int, max_context_len: int):
    d = scale_decision(gpu, model, n_gpus, target_concurrency, max_context_len)
    print(f"\n{'='*72}")
    print(f"  Scale-Up vs Scale-Out  |  {model.name}  |  {gpu.name} × {n_gpus}")
    print(f"  Target: {target_concurrency} concurrent sessions, {max_context_len:,} token context")
    print(f"{'='*72}")

    print(f"\n  ── Option A: TP={n_gpus} (scale-up, one large instance) ──")
    if not d['valid_tp']:
        print(f"  ✗ Invalid: {model.kv_heads} KV heads not divisible by TP={n_gpus}")
    elif not d['fits_single'] and d['tp_kv_total_tok'] == 0:
        print(f"  ✗ Model too large even with TP={n_gpus}")
    else:
        print(f"  Total KV tokens : {d['tp_kv_total_tok']:>12,}")
        print(f"  Max sessions    : {d['tp_max_sessions']:>12,}  (at {max_context_len:,} ctx)")
        print(f"  AllReduce cost  : {'~5% (NVLink)' if d['nvlink'] else '~50-100% (PCIe)'}")
        print(f"  Decode latency  : {'Reduced (parallel BW)' if d['nvlink'] else 'May be WORSE than single GPU'}")
        if d['tp_max_sessions'] >= target_concurrency:
            print(f"  ✓ Meets concurrency target ({target_concurrency})")
        else:
            print(f"  ✗ Insufficient: {d['tp_max_sessions']} < {target_concurrency} target")

    print(f"\n  ── Option B: {n_gpus} replicas (scale-out, {n_gpus} independent instances) ──")
    if not d['fits_single']:
        print(f"  ✗ Model ({weights_per_gpu_gb(model, 1):.0f} GB) does not fit on single {gpu.name} ({gpu.hbm_gb} GB)")
    else:
        print(f"  KV tokens/inst  : {d['replica_kv_each']:>12,}")
        print(f"  Total sessions  : {d['replica_max_sess']:>12,}  ({n_gpus} × {d['replica_kv_each'] // max_context_len})")
        print(f"  AllReduce cost  : None")
        print(f"  Throughput      : {n_gpus}× single-GPU (linear)")
        if d['replica_max_sess'] >= target_concurrency:
            print(f"  ✓ Meets concurrency target ({target_concurrency})")
        else:
            print(f"  ✗ Insufficient: {d['replica_max_sess']} < {target_concurrency} target")

    print(f"\n  ── Recommendation ──")
    if not d['fits_single']:
        print(f"  → Must use TP (model does not fit on a single {gpu.name})")
    elif d['nvlink']:
        print(f"  → Both viable. TP preferred if TTFT latency matters.")
        print(f"    Replicas preferred if throughput > latency.")
    else:
        print(f"  → Prefer replicas. PCIe AllReduce likely degrades latency.")
    print(f"{'='*72}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. RING-ALLREDUCE TRACE (small N)
# ──────────────────────────────────────────────────────────────────────────────

def ring_allreduce_trace(values: list[list[float]]):
    """
    Simulate ring-allreduce on N GPUs each holding a list of chunks.
    values[gpu][chunk_idx] = partial float value

    Algorithm:
      Phase 1 (Scatter-reduce): N-1 rounds.
        Round r: GPU g sends chunk (g-r)%N to GPU (g+1)%N
                 GPU g receives chunk (g-r-1)%N from GPU (g-1)%N and accumulates.
        After N-1 rounds, GPU g holds fully-reduced chunk (g+1)%N.

      Phase 2 (All-gather): N-1 rounds.
        Round r: GPU g sends chunk (g+1+r)%N to GPU (g+1)%N
                 GPU g receives fully-reduced chunk (g+r)%N from GPU (g-1)%N.
        After N-1 rounds, all GPUs hold all fully-reduced chunks.
    """
    n_gpus   = len(values)
    n_chunks = len(values[0])
    state    = [[v for v in row] for row in values]

    print(f"\n{'='*62}")
    print(f"  Ring-AllReduce Trace  (N={n_gpus} GPUs, {n_chunks} chunks each)")
    print(f"{'='*62}")

    def show(phase, step):
        print(f"\n  [{phase} step {step}]")
        for g, row in enumerate(state):
            vals = "  ".join(f"{v:5.1f}" for v in row)
            print(f"    GPU {g}: [{vals}]")

    print(f"\n  Initial values (each GPU has partial result per chunk):")
    show("init", 0)

    # ── Phase 1: Scatter-reduce
    print(f"\n  Phase 1: Scatter-reduce ({n_gpus-1} steps)")
    for r in range(n_gpus - 1):
        new_state = [[v for v in row] for row in state]
        for g in range(n_gpus):
            src        = (g - 1) % n_gpus
            recv_chunk = (g - r - 1) % n_gpus      # which chunk g receives this round
            new_state[g][recv_chunk] += state[src][recv_chunk]
        state = new_state
        show("scatter-reduce", r + 1)

    # After scatter-reduce, GPU g owns fully-reduced chunk (g+1)%n_gpus
    print(f"\n  After scatter-reduce: GPU g holds full sum for chunk (g+1)%{n_gpus}")
    for g in range(n_gpus):
        owned = (g + 1) % n_gpus
        print(f"    GPU {g} owns chunk {owned} = {state[g][owned]:.1f}")

    # ── Phase 2: All-gather
    print(f"\n  Phase 2: All-gather ({n_gpus-1} steps)")
    for r in range(n_gpus - 1):
        new_state = [[v for v in row] for row in state]
        for g in range(n_gpus):
            src = (g - 1) % n_gpus
            # In round r, GPU src sends the chunk it received in round r-1
            # (or its owned chunk in round 0). Chunk index decrements each round.
            # Round r: GPU src sends chunk (src + n + 1 - r) % n
            sent_chunk = (src + n_gpus + 1 - r) % n_gpus
            new_state[g][sent_chunk] = state[src][sent_chunk]
        state = new_state
        show("all-gather", r + 1)

    # Verify
    expected = [sum(values[g][c] for g in range(n_gpus)) for c in range(n_chunks)]
    actual   = state[0]  # all GPUs should match
    ok = all(abs(actual[c] - expected[c]) < 1e-9 for c in range(n_chunks))
    print(f"\n  Expected: " + "  ".join(f"{v:5.1f}" for v in expected))
    print(f"  Got:      " + "  ".join(f"{v:5.1f}" for v in actual))
    print(f"  Result:   {'✓ CORRECT' if ok else '✗ MISMATCH'}")
    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. LLAMA.CPP TENSOR-SPLIT MEMORY ESTIMATOR
# ──────────────────────────────────────────────────────────────────────────────

def llamacpp_tensor_split(gpu_hbm_list: list[float], model: ModelSpec,
                           quant_factor: float = 1.0,
                           ctx_size: int = 8192) -> None:
    """
    Estimate memory allocation for llama.cpp --tensor-split.
    quant_factor: 1.0 = BF16, 0.5 = Q8, 0.25 = Q4
    gpu_hbm_list: list of GPU HBM sizes in GB
    """
    n_gpus        = len(gpu_hbm_list)
    total_hbm     = sum(gpu_hbm_list)
    total_weight_gb = model.params_b * 1e9 * model.dtype_bytes / 1024**3 * quant_factor

    # In llama.cpp --tensor-split, layers are distributed proportionally
    layers_per_gpu = [
        round(model.layers * hbm / total_hbm)
        for hbm in gpu_hbm_list
    ]

    print(f"\n{'='*68}")
    print(f"  llama.cpp --tensor-split Estimator  |  {model.name}")
    print(f"  Total model: {total_weight_gb:.1f} GB (quant={quant_factor}x BF16)")
    print(f"  Context: {ctx_size:,} tokens")
    print(f"{'='*68}")
    print(f"  {'GPU':>4}  {'HBM':>6}  {'Layers':>8}  {'Weight':>8}  {'KV':>8}  {'Free':>8}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    kv_per_tok = 2 * model.layers * model.kv_heads * model.head_dim * model.dtype_bytes
    kv_total_gb = kv_per_tok * ctx_size / 1024**3

    for i, (hbm, n_layers) in enumerate(zip(gpu_hbm_list, layers_per_gpu)):
        frac       = hbm / total_hbm
        w_gb       = total_weight_gb * frac
        kv_gb      = kv_total_gb * frac   # KV distributed proportionally
        free_gb    = hbm - w_gb - kv_gb - MODEL_OVERHEAD_GB
        status = "✓" if free_gb > 0 else "✗ OOM"
        print(f"  {i:>4}  {hbm:>4.0f}GB  {n_layers:>8}  "
              f"{w_gb:>6.1f}GB  {kv_gb:>6.1f}GB  {max(0,free_gb):>6.1f}GB  {status}")

    total_kv_gb = kv_total_gb
    ratio_str   = ":".join(str(int(h)) for h in gpu_hbm_list)
    print(f"  {'─'*64}")
    print(f"  Suggested --tensor-split {ratio_str}")
    print(f"  Total weight: {total_weight_gb:.1f} GB  |  KV for {ctx_size:,} ctx: {total_kv_gb:.1f} GB")
    print(f"{'='*68}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 62)
    print("  Chapter 15 — Multi-GPU Serving: Companion Demo")
    print("█" * 62)

    a100_80  = GPUS["A100-80"]
    h100_80  = GPUS["H100-80"]
    rtx4090  = GPUS["RTX4090"]
    llama8b  = MODELS["Llama-3-8B"]
    llama70b = MODELS["Llama-3-70B"]

    # ── Section 1: Weight shard + KV pool at each TP
    print_weight_shard_table(a100_80, llama8b)
    print_weight_shard_table(a100_80, llama70b)

    # ── Section 2: AllReduce latency table
    print_allreduce_table(llama8b, batch_tokens=128)
    print_allreduce_table(llama70b, batch_tokens=64)

    # ── Section 3: Throughput vs GPU count (NVLink server)
    print_throughput_vs_gpus(a100_80, llama8b, batch_tokens=128)
    print_throughput_vs_gpus(a100_80, llama70b, batch_tokens=64)

    # ── Section 4: Throughput vs GPU count (consumer PCIe)
    print(f"\n{'─'*62}")
    print(f"  Consumer PCIe: RTX 4090 (no NVLink)")
    print(f"{'─'*62}")
    print_throughput_vs_gpus(rtx4090, llama8b, batch_tokens=128)

    # ── Section 5: Break-even batch size
    print_break_even_table(a100_80, llama8b)
    print_break_even_table(rtx4090, llama8b)

    # ── Section 6: Scale-up vs scale-out
    print_scale_decision(a100_80, llama8b, n_gpus=4,
                         target_concurrency=100, max_context_len=8192)
    print_scale_decision(a100_80, llama70b, n_gpus=4,
                         target_concurrency=30, max_context_len=16384)
    print_scale_decision(rtx4090, llama8b, n_gpus=4,
                         target_concurrency=50, max_context_len=8192)

    # ── Section 7: Ring-AllReduce trace
    ring_allreduce_trace([
        [1.0, 5.0, 3.0, 7.0],   # GPU 0
        [2.0, 4.0, 6.0, 8.0],   # GPU 1
        [3.0, 3.0, 3.0, 3.0],   # GPU 2
        [4.0, 2.0, 0.0, 2.0],   # GPU 3
    ])

    # ── Section 8: llama.cpp tensor-split estimator
    # Equal GPUs
    llamacpp_tensor_split([24.0, 24.0], llama8b,
                           quant_factor=0.25,  # Q4
                           ctx_size=16384)
    # Mixed GPU sizes
    llamacpp_tensor_split([24.0, 48.0], llama70b,
                           quant_factor=0.25,  # Q4 (70B Q4 ≈ 40 GB)
                           ctx_size=8192)


if __name__ == "__main__":
    main()

```

## C++ — `multigpu_demo.cpp`

```cpp
/*
 * Chapter 15 Companion Code: Multi-GPU Serving and Tensor Parallelism
 * =====================================================================
 * Demonstrates:
 *   1. Weight shard calculator — bytes per GPU at each TP level
 *   2. KV cache sizing under TP — tokens per GPU, total pool
 *   3. AllReduce latency model — NVLink vs PCIe (alpha-beta model)
 *   4. Ring-AllReduce step-by-step trace (N=4 GPUs, 4 chunks)
 *   5. Throughput vs GPU count — compute gain vs AllReduce tax
 *   6. Break-even batch size — when TP helps vs hurts
 *   7. Scale-up vs scale-out decision table
 *   8. llama.cpp --tensor-split annotation (code sketch)
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -o multigpu_demo multigpu_demo.cpp
 * Run:
 *   ./multigpu_demo
 *
 * No external dependencies.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

static std::string fmt_int(long long v) {
    std::string s = std::to_string(std::abs(v));
    int i = static_cast<int>(s.size()) - 3;
    while (i > 0) { s.insert(static_cast<size_t>(i), ","); i -= 3; }
    return (v < 0 ? "-" : "") + s;
}

static std::string fmt_f(double v, int prec = 1) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

static std::string repeat(char c, int n) {
    return std::string(static_cast<size_t>(std::max(0, n)), c);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. DATA STRUCTURES
// ─────────────────────────────────────────────────────────────────────────────

struct GPUSpec {
    std::string name;
    double hbm_gb;
    double bw_tb_s;         // memory bandwidth TB/s
    double tflops_bf16;
    double nvlink_bw_gbs;   // 0 = PCIe only
    double pcie_bw_gbs;
};

struct ModelSpec {
    std::string name;
    double  params_b;
    int     layers;
    int     hidden_dim;
    int     num_heads;
    int     kv_heads;
    int     head_dim;
    int     dtype_bytes;    // 2 = BF16
};

struct Interconnect {
    std::string name;
    double bw_gbs;          // bidirectional GB/s
    double latency_us;      // base latency microseconds
};

static const double MODEL_OVERHEAD_GB = 1.5;

// ─────────────────────────────────────────────────────────────────────────────
// 2. MEMORY CALCULATIONS
// ─────────────────────────────────────────────────────────────────────────────

static double weights_per_gpu_gb(const ModelSpec& m, int tp) {
    double total_bytes = m.params_b * 1e9 * m.dtype_bytes;
    return total_bytes / (1024.0 * 1024.0 * 1024.0) / tp;
}

static long long kv_bytes_per_token_per_gpu(const ModelSpec& m, int tp) {
    int kv_heads_per_gpu = std::max(1, m.kv_heads / tp);
    return 2LL * m.layers * kv_heads_per_gpu * m.head_dim * m.dtype_bytes;
}

static double kv_pool_gb(const GPUSpec& gpu, const ModelSpec& m, int tp,
                         double gpu_util = 0.90) {
    double usable = gpu.hbm_gb * gpu_util;
    double w_gb   = weights_per_gpu_gb(m, tp);
    return std::max(0.0, usable - w_gb - MODEL_OVERHEAD_GB);
}

static long long kv_total_tokens(const GPUSpec& gpu, const ModelSpec& m, int tp,
                                 double gpu_util = 0.90) {
    double pool    = kv_pool_gb(gpu, m, tp, gpu_util);
    long long bpt  = kv_bytes_per_token_per_gpu(m, tp);
    long long per  = static_cast<long long>(pool * 1024.0 * 1024.0 * 1024.0 / bpt);
    return per * tp;
}

static void print_weight_shard_table(const GPUSpec& gpu, const ModelSpec& model) {
    std::vector<int> tps;
    for (int t : {1, 2, 4, 8, 16}) {
        if (model.kv_heads % t == 0) tps.push_back(t);
    }

    std::cout << "\n" << repeat('=', 78) << "\n";
    std::cout << "  Weight Shard + KV Pool  |  " << model.name
              << "  |  " << gpu.name << "\n";
    std::cout << repeat('=', 78) << "\n";
    std::cout << "  " << std::right
              << std::setw(4)  << "TP"
              << std::setw(14) << "Weight/GPU"
              << std::setw(8)  << "Fits?"
              << std::setw(12) << "KV B/tok"
              << std::setw(14) << "KV Pool/GPU"
              << std::setw(16) << "Total KV Tok" << "\n";
    std::cout << "  " << repeat('-', 74) << "\n";

    for (int tp : tps) {
        double  w_gb    = weights_per_gpu_gb(model, tp);
        bool    fits    = w_gb < (gpu.hbm_gb * 0.90 - MODEL_OVERHEAD_GB);
        long long kv_bt = kv_bytes_per_token_per_gpu(model, tp);
        double  kp_gb   = kv_pool_gb(gpu, model, tp);
        long long tot_t = kv_total_tokens(gpu, model, tp);

        std::cout << "  " << std::right
                  << std::setw(4)  << tp
                  << std::setw(12) << fmt_f(w_gb) + "GB"
                  << std::setw(8)  << (fits ? "YES" : "NO ")
                  << std::setw(12) << fmt_int(kv_bt)
                  << std::setw(12) << fmt_f(kp_gb) + "GB"
                  << std::setw(16) << fmt_int(tot_t) << "\n";
    }
    std::cout << repeat('=', 78) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. ALLREDUCE LATENCY MODEL
// ─────────────────────────────────────────────────────────────────────────────

static double allreduce_latency_ms(long long msg_bytes, int n_gpus,
                                   const Interconnect& link) {
    // Ring-AllReduce alpha-beta model:
    // Transfer = 2 × (n-1)/n × msg / bw
    // Fixed    = 2 × (n-1) × alpha
    double transfer = 2.0 * (n_gpus - 1.0) / n_gpus * msg_bytes
                      / (link.bw_gbs * 1e9);
    double fixed    = 2.0 * (n_gpus - 1) * (link.latency_us * 1e-6);
    return (transfer + fixed) * 1000.0;
}

static void print_allreduce_table(const ModelSpec& m, int batch_tokens) {
    long long msg_bytes = static_cast<long long>(batch_tokens) * m.hidden_dim * m.dtype_bytes;

    std::vector<Interconnect> links = {
        {"NVLink3 (A100)", 600.0, 1.0},
        {"NVLink4 (H100)", 900.0, 1.0},
        {"PCIe4 x16",       32.0, 5.0},
        {"PCIe5 x16",       64.0, 3.0},
    };

    std::cout << "\n" << repeat('=', 70) << "\n";
    std::cout << "  AllReduce Latency  |  " << m.name
              << "  |  batch=" << batch_tokens << "\n";
    std::cout << "  Msg per AllReduce: " << fmt_int(msg_bytes) << " bytes ("
              << fmt_int(msg_bytes / 1024) << " KB)\n";
    std::cout << repeat('=', 70) << "\n";
    std::cout << "  " << std::left << std::setw(22) << "Interconnect"
              << std::right
              << std::setw(10) << "TP=2"
              << std::setw(10) << "TP=4"
              << std::setw(10) << "TP=8"
              << std::setw(10) << "TP=16" << "\n";
    std::cout << "  " << repeat('-', 62) << "\n";

    for (auto& link : links) {
        std::cout << "  " << std::left << std::setw(22) << link.name;
        for (int tp : {2, 4, 8, 16}) {
            double lat = allreduce_latency_ms(msg_bytes, tp, link);
            std::cout << std::right << std::setw(8) << fmt_f(lat, 2) << "ms";
        }
        std::cout << "\n";
    }
    std::cout << "  Note: 2*L AllReduces per forward pass (2 per layer)\n";
    std::cout << repeat('=', 70) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. RING-ALLREDUCE TRACE
// ─────────────────────────────────────────────────────────────────────────────

static void show_state(const std::string& label,
                       const std::vector<std::vector<double>>& state) {
    std::cout << "\n  [" << label << "]\n";
    for (size_t g = 0; g < state.size(); ++g) {
        std::cout << "    GPU " << g << ": [";
        for (size_t c = 0; c < state[g].size(); ++c) {
            if (c) std::cout << "  ";
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(5) << state[g][c];
        }
        std::cout << "]\n";
    }
}

static void ring_allreduce_trace(std::vector<std::vector<double>> values) {
    int n_gpus   = static_cast<int>(values.size());
    int n_chunks = static_cast<int>(values[0].size());
    auto state   = values;

    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  Ring-AllReduce Trace  (N=" << n_gpus
              << " GPUs, " << n_chunks << " chunks each)\n";
    std::cout << repeat('=', 62) << "\n";

    show_state("init step 0", state);

    // Phase 1: Scatter-reduce (N-1 rounds)
    std::cout << "\n  Phase 1: Scatter-reduce (" << n_gpus - 1 << " steps)\n";
    for (int r = 0; r < n_gpus - 1; ++r) {
        auto next = state;
        for (int g = 0; g < n_gpus; ++g) {
            int src        = (g - 1 + n_gpus) % n_gpus;
            int recv_chunk = ((g - r - 1) % n_gpus + n_gpus) % n_gpus;
            next[g][recv_chunk] += state[src][recv_chunk];
        }
        state = next;
        show_state("scatter-reduce step " + std::to_string(r + 1), state);
    }

    // After scatter-reduce: GPU g owns chunk (g+1)%N
    std::cout << "\n  After scatter-reduce: GPU g holds full sum for chunk (g+1)%"
              << n_gpus << "\n";
    for (int g = 0; g < n_gpus; ++g) {
        int owned = (g + 1) % n_gpus;
        std::cout << "    GPU " << g << " owns chunk " << owned
                  << " = " << std::fixed << std::setprecision(1)
                  << state[g][owned] << "\n";
    }

    // Phase 2: All-gather (N-1 rounds)
    std::cout << "\n  Phase 2: All-gather (" << n_gpus - 1 << " steps)\n";
    for (int r = 0; r < n_gpus - 1; ++r) {
        auto next = state;
        for (int g = 0; g < n_gpus; ++g) {
            int src        = (g - 1 + n_gpus) % n_gpus;
            // GPU src sends chunk (src + n + 1 - r) % n in round r
            int sent_chunk = ((src + n_gpus + 1 - r) % n_gpus + n_gpus) % n_gpus;
            next[g][sent_chunk] = state[src][sent_chunk];
        }
        state = next;
        show_state("all-gather step " + std::to_string(r + 1), state);
    }

    // Verify
    std::vector<double> expected(n_chunks, 0.0);
    for (int c = 0; c < n_chunks; ++c)
        for (int g = 0; g < n_gpus; ++g)
            expected[c] += values[g][c];

    std::cout << "\n  Expected: ";
    for (double v : expected)
        std::cout << std::fixed << std::setprecision(1) << std::setw(6) << v << " ";
    std::cout << "\n  Got:      ";
    for (double v : state[0])
        std::cout << std::fixed << std::setprecision(1) << std::setw(6) << v << " ";

    bool ok = true;
    for (int c = 0; c < n_chunks; ++c)
        if (std::fabs(state[0][c] - expected[c]) > 1e-9) ok = false;
    std::cout << "\n  Result:   " << (ok ? "CORRECT" : "MISMATCH") << "\n";
    std::cout << repeat('=', 62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. THROUGHPUT VS GPU COUNT
// ─────────────────────────────────────────────────────────────────────────────

static void print_throughput_vs_gpus(const GPUSpec& gpu, const ModelSpec& m,
                                     int batch_tokens) {
    Interconnect link;
    if (gpu.nvlink_bw_gbs > 0)
        link = {"NVLink", gpu.nvlink_bw_gbs, 1.0};
    else
        link = {"PCIe",   gpu.pcie_bw_gbs, 5.0};

    // Single GPU baseline: weight bandwidth bound
    double single_weight_bytes = m.params_b * 1e9 * m.dtype_bytes;
    double single_ms = single_weight_bytes / (gpu.bw_tb_s * 1e12) * 1000.0;
    double single_tps = static_cast<double>(batch_tokens) / (single_ms / 1000.0);

    std::cout << "\n" << repeat('=', 76) << "\n";
    std::cout << "  Throughput vs GPU Count  |  " << m.name
              << "  |  " << gpu.name << "  |  batch=" << batch_tokens << "\n";
    std::cout << "  Interconnect: " << link.name << " " << link.bw_gbs << " GB/s\n";
    std::cout << repeat('=', 76) << "\n";
    std::cout << "  " << std::right
              << std::setw(4)  << "TP"
              << std::setw(12) << "Step (ms)"
              << std::setw(10) << "Tok/s"
              << std::setw(10) << "vs TP=1"
              << std::setw(14) << "AR overhead" << "\n";
    std::cout << "  " << repeat('-', 50) << "\n";

    std::vector<int> tps;
    for (int t : {1, 2, 4, 8}) {
        if (m.kv_heads % t == 0) tps.push_back(t);
    }

    for (int tp : tps) {
        double step_ms, ar_pct;
        if (tp == 1) {
            step_ms = single_ms;
            ar_pct  = 0.0;
        } else {
            // Compute: load (1/tp) weights per GPU
            double w_bytes   = m.params_b * 1e9 * m.dtype_bytes / tp;
            double comp_ms   = w_bytes / (gpu.bw_tb_s * 1e12) * 1000.0;
            // AllReduce: 2*L collectives per step
            long long msg    = static_cast<long long>(batch_tokens) * m.hidden_dim * m.dtype_bytes;
            double ar_per_ms = allreduce_latency_ms(msg, tp, link);
            double ar_total  = 2.0 * m.layers * ar_per_ms;
            step_ms = comp_ms + ar_total;
            ar_pct  = ar_total / step_ms * 100.0;
        }

        double toks  = static_cast<double>(batch_tokens) / (step_ms / 1000.0);
        double ratio = toks / single_tps;
        std::string ar_str = (tp == 1) ? "  --" : fmt_f(ar_pct, 1) + "%";

        std::cout << "  " << std::right
                  << std::setw(4)  << tp
                  << std::setw(12) << fmt_f(step_ms, 2)
                  << std::setw(10) << fmt_int(static_cast<long long>(toks))
                  << std::setw(9)  << fmt_f(ratio, 2) << "x"
                  << std::setw(14) << ar_str << "\n";
    }
    std::cout << repeat('=', 76) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. BREAK-EVEN BATCH SIZE
// ─────────────────────────────────────────────────────────────────────────────

static int break_even_batch(const GPUSpec& gpu, const ModelSpec& m, int tp,
                            const Interconnect& link) {
    double single_ms = m.params_b * 1e9 * m.dtype_bytes / (gpu.bw_tb_s * 1e12) * 1000.0;

    for (int B = 1; B <= 4096; ++B) {
        double w_bytes  = m.params_b * 1e9 * m.dtype_bytes / tp;
        double comp_ms  = w_bytes / (gpu.bw_tb_s * 1e12) * 1000.0;
        long long msg   = static_cast<long long>(B) * m.hidden_dim * m.dtype_bytes;
        double ar_total = 2.0 * m.layers * allreduce_latency_ms(msg, tp, link);
        double tp_ms    = comp_ms + ar_total;
        if (tp_ms < single_ms) return B;
    }
    return -1;  // never
}

static void print_break_even(const GPUSpec& gpu, const ModelSpec& m) {
    Interconnect nvlink{"NVLink", gpu.nvlink_bw_gbs > 0 ? gpu.nvlink_bw_gbs : 1.0, 1.0};
    Interconnect pcie{"PCIe", gpu.pcie_bw_gbs, 5.0};

    std::cout << "\n" << repeat('=', 68) << "\n";
    std::cout << "  Break-Even Batch Size  |  " << m.name << "  |  " << gpu.name << "\n";
    std::cout << "  (TP decode latency < single-GPU latency above this batch)\n";
    std::cout << repeat('=', 68) << "\n";
    std::cout << "  " << std::right
              << std::setw(4)  << "TP"
              << std::setw(22) << "NVLink break-even"
              << std::setw(22) << "PCIe break-even" << "\n";
    std::cout << "  " << repeat('-', 48) << "\n";

    for (int tp : {2, 4, 8}) {
        if (m.kv_heads % tp != 0) continue;
        std::string nv_str, pc_str;
        if (gpu.nvlink_bw_gbs > 0) {
            int nv = break_even_batch(gpu, m, tp, nvlink);
            nv_str = (nv < 0 ? "never" : fmt_int(nv) + " tok");
        } else {
            nv_str = "N/A (no NVLink)";
        }
        int pc = break_even_batch(gpu, m, tp, pcie);
        pc_str = (pc < 0 ? "never" : fmt_int(pc) + " tok");

        std::cout << "  " << std::right
                  << std::setw(4)  << tp
                  << std::setw(22) << nv_str
                  << std::setw(22) << pc_str << "\n";
    }
    std::cout << repeat('=', 68) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. SCALE-UP VS SCALE-OUT DECISION
// ─────────────────────────────────────────────────────────────────────────────

static void print_scale_decision(const GPUSpec& gpu, const ModelSpec& m,
                                 int n_gpus, int target_sessions, int ctx_len) {
    bool valid_tp  = (m.kv_heads % n_gpus == 0);
    bool fits_one  = (weights_per_gpu_gb(m, 1) < gpu.hbm_gb * 0.90 - MODEL_OVERHEAD_GB);

    long long tp_kv_total = valid_tp ? kv_total_tokens(gpu, m, n_gpus) : 0;
    long long tp_sessions = (ctx_len > 0 && tp_kv_total > 0) ? tp_kv_total / ctx_len : 0;

    long long rep_kv_each = fits_one ? kv_total_tokens(gpu, m, 1) : 0;
    long long rep_sessions = (ctx_len > 0) ? (rep_kv_each / ctx_len) * n_gpus : 0;

    std::cout << "\n" << repeat('=', 70) << "\n";
    std::cout << "  Scale-Up vs Scale-Out  |  " << m.name
              << "  |  " << gpu.name << " x" << n_gpus << "\n";
    std::cout << "  Target: " << target_sessions << " sessions, "
              << fmt_int(ctx_len) << " token context\n";
    std::cout << repeat('=', 70) << "\n";

    // Option A: TP
    std::cout << "\n  -- Option A: TP=" << n_gpus << " (scale-up) --\n";
    if (!valid_tp) {
        std::cout << "  !! Invalid: " << m.kv_heads << " KV heads not divisible by "
                  << n_gpus << "\n";
    } else {
        std::cout << "  Total KV tokens : " << fmt_int(tp_kv_total) << "\n";
        std::cout << "  Max sessions    : " << fmt_int(tp_sessions) << "\n";
        std::cout << "  AllReduce cost  : "
                  << (gpu.nvlink_bw_gbs > 0 ? "~5% (NVLink)" : "~50-100% (PCIe)") << "\n";
        std::cout << "  " << (tp_sessions >= target_sessions ? "OK" : "!!")
                  << "  " << (tp_sessions >= target_sessions
                               ? "Meets concurrency target"
                               : "INSUFFICIENT: " + fmt_int(tp_sessions)
                                 + " < " + std::to_string(target_sessions)) << "\n";
    }

    // Option B: Replicas
    std::cout << "\n  -- Option B: " << n_gpus << " replicas (scale-out) --\n";
    if (!fits_one) {
        std::cout << "  !! Model (" << fmt_f(weights_per_gpu_gb(m, 1)) << " GB)"
                  << " does not fit on single " << gpu.name << "\n";
    } else {
        std::cout << "  KV tokens/inst  : " << fmt_int(rep_kv_each) << "\n";
        std::cout << "  Total sessions  : " << fmt_int(rep_sessions)
                  << " (" << n_gpus << " x " << fmt_int(rep_kv_each / ctx_len) << ")\n";
        std::cout << "  AllReduce cost  : None\n";
        std::cout << "  Throughput      : " << n_gpus << "x single-GPU (linear)\n";
        std::cout << "  " << (rep_sessions >= target_sessions ? "OK" : "!!")
                  << "  " << (rep_sessions >= target_sessions
                               ? "Meets concurrency target"
                               : "INSUFFICIENT: " + fmt_int(rep_sessions)
                                 + " < " + std::to_string(target_sessions)) << "\n";
    }

    // Recommendation
    std::cout << "\n  -- Recommendation --\n";
    if (!fits_one)
        std::cout << "  -> Must use TP (model too large for single GPU)\n";
    else if (gpu.nvlink_bw_gbs > 0)
        std::cout << "  -> Both viable. TP if TTFT matters; replicas if throughput > latency.\n";
    else
        std::cout << "  -> Prefer replicas. PCIe AllReduce likely degrades latency.\n";

    std::cout << repeat('=', 70) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. LLAMA.CPP --tensor-split ANNOTATION
// ─────────────────────────────────────────────────────────────────────────────

static void print_llamacpp_annotation() {
    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  llama.cpp --tensor-split: Code Sketch\n";
    std::cout << repeat('=', 62) << "\n";
    std::cout << R"(
  // llama.cpp splits layers across GPUs proportionally to --tensor-split ratios.
  // This is NOT Megatron-style TP: each GPU handles different layers,
  // and activations are transferred between GPUs via PCIe as the forward
  // pass traverses layers.

  // CLI example:
  //   llama-server \
  //     --model llama-3-70b-q4_k_m.gguf \
  //     --n-gpu-layers 80 \
  //     --tensor-split 1,1      # equal 50/50 split \
  //     --ctx-size 16384 \
  //     --parallel 4

  // In C API:
  struct llama_model_params mparams = llama_model_default_params();

  // Split tensor data proportionally across GPUs
  // tensor_split[i] = fraction of model data on GPU i (ratios, not %)
  float tensor_split[2] = { 0.5f, 0.5f };   // equal split
  mparams.tensor_split    = tensor_split;
  mparams.n_gpu_layers    = 80;              // send all layers to GPU

  struct llama_model* model = llama_load_model_from_file(path, mparams);

  // Context parameters (per inference session)
  struct llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx      = 16384;  // max_model_len equivalent
  cparams.n_batch    = 4096;   // max tokens per llama_decode() call
  cparams.n_ubatch   = 512;    // micro-batch (chunked prefill equivalent)
  cparams.n_parallel = 4;      // simultaneous decode sequences

  struct llama_context* ctx = llama_new_context_with_model(model, cparams);

  // Key difference from vLLM TP:
  //   vLLM: each layer is split ACROSS GPUs (Megatron column-row sharding)
  //         → AllReduce at every layer boundary
  //   llama.cpp: layers ASSIGNED to GPUs (pipeline-style)
  //         → activation transfer at stage boundaries only
  //
  // Consequence: llama.cpp multi-GPU on PCIe is primarily for CAPACITY
  // (fitting a larger model), not for SPEED. On NVLink servers, vLLM TP
  // is the better choice for latency-critical serving.
)";
    std::cout << repeat('=', 62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    GPUSpec a100_80 {"A100-80",  80.0, 2.00, 312.0, 600.0, 32.0};
    GPUSpec rtx4090 {"RTX4090",  24.0, 1.01, 165.0,   0.0, 32.0};

    ModelSpec llama8b  {"Llama-3-8B",   8.0,  32, 4096, 32,  8, 128, 2};
    ModelSpec llama70b {"Llama-3-70B", 70.0,  80, 8192, 64,  8, 128, 2};

    std::cout << "\n" << repeat('#', 62) << "\n";
    std::cout << "  Chapter 15 - Multi-GPU Serving: C++ Companion Demo\n";
    std::cout << repeat('#', 62) << "\n";

    // Section 1: Weight shards + KV pool
    print_weight_shard_table(a100_80, llama8b);
    print_weight_shard_table(a100_80, llama70b);

    // Section 2: AllReduce latency
    print_allreduce_table(llama8b,  128);
    print_allreduce_table(llama70b,  64);

    // Section 3: Throughput vs TP (NVLink server)
    print_throughput_vs_gpus(a100_80, llama8b,  128);
    print_throughput_vs_gpus(a100_80, llama70b,  64);

    // Section 4: Throughput vs TP (consumer PCIe)
    std::cout << "\n" << repeat('-', 62) << "\n";
    std::cout << "  Consumer PCIe: RTX 4090 (no NVLink)\n";
    std::cout << repeat('-', 62) << "\n";
    print_throughput_vs_gpus(rtx4090, llama8b, 128);

    // Section 5: Break-even
    print_break_even(a100_80, llama8b);
    print_break_even(rtx4090, llama8b);

    // Section 6: Scale decision
    print_scale_decision(a100_80, llama8b,  4, 100,  8192);
    print_scale_decision(a100_80, llama70b, 4,  30, 16384);

    // Section 7: Ring-AllReduce trace
    ring_allreduce_trace({
        {1.0, 5.0, 3.0, 7.0},
        {2.0, 4.0, 6.0, 8.0},
        {3.0, 3.0, 3.0, 3.0},
        {4.0, 2.0, 0.0, 2.0},
    });

    // Section 8: llama.cpp annotation
    print_llamacpp_annotation();

    return 0;
}

```

