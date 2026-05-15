# Chapter 9: The Forward Pass — CUDA vs. GGML — Companion Code

## Python — `forward_pass_demo.py`

```python
"""
Chapter 9 — The Forward Pass: CUDA vs. GGML
Companion code: forward_pass_demo.py

Sections:
  1. FLOP accounting and memory-bandwidth-bound model
  2. CUDA graph vs. eager Python overhead model
  3. Tensor-parallel AllReduce cost (NVLink vs. PCIe)
  4. Per-layer timing breakdown
  5. GGML-style DAG: builder + topological executor
  6. Block-wise dequantization simulation (Q4 × F32)
  7. End-to-end decode step timing comparison

Run:
  python3 forward_pass_demo.py

No external dependencies beyond the standard library + dataclasses.
"""

from __future__ import annotations
import math
import struct
import time
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# §1  FLOP Accounting and Memory-Bandwidth-Bound Model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    n_layers: int
    d_model: int
    d_ffn: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    bpw: float = 16.0          # bits per weight (16 = BF16/FP16)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def weight_bytes(self) -> int:
        """Approximate total weight bytes (embedding + transformer + lm_head)."""
        # Embedding + lm_head
        embed = 2 * self.vocab_size * self.d_model * (self.bpw / 8)
        # Per-layer: Q/K/V/O projections + gate/up/down projections + norms
        attn = 4 * self.d_model * self.d_model           # Q,K,V,O
        ffn  = 3 * self.d_model * self.d_ffn             # gate, up, down
        norm = 4 * self.d_model                          # 2 × rms_norm weights
        per_layer = (attn + ffn + norm) * (self.bpw / 8)
        return int(embed + self.n_layers * per_layer)


# LLaMA 3 8B
LLAMA3_8B = ModelConfig(
    name="LLaMA-3-8B",
    n_layers=32,
    d_model=4096,
    d_ffn=14336,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
)

# LLaMA 3 70B
LLAMA3_70B = ModelConfig(
    name="LLaMA-3-70B",
    n_layers=80,
    d_model=8192,
    d_ffn=28672,
    n_heads=64,
    n_kv_heads=8,
    vocab_size=128256,
)


def flops_per_decode_step(cfg: ModelConfig, batch_size: int,
                           avg_seq_len: int = 512) -> dict:
    """
    Compute FLOPs for one decode step (1 new token per sequence).

    Returns a breakdown dict with GFLOP values.
    """
    B  = batch_size
    d  = cfg.d_model
    h  = cfg.n_heads
    hk = cfg.n_kv_heads
    dh = cfg.d_head
    df = cfg.d_ffn
    L  = cfg.n_layers
    V  = cfg.vocab_size

    # Each linear: 2 × B × d_in × d_out  (multiply-accumulate → 2 FLOPs)
    # Attention projections per layer
    q_proj  = 2 * B * d * d            # Q: (d → h×dh = d)
    kv_proj = 2 * 2 * B * d * (hk*dh) # K+V: (d → hk×dh each)
    o_proj  = 2 * B * d * d            # O: (h×dh → d)

    # QK^T + AV per layer (approximate, scales with seq_len)
    attn_qkt = 2 * B * h * avg_seq_len * dh   # QK^T: [B,h,1,dh]×[B,h,dh,S]
    attn_av  = 2 * B * h * avg_seq_len * dh   # AV:   [B,h,1,S]×[B,h,S,dh]

    # FFN per layer
    gate_up = 2 * 2 * B * d * df     # gate proj + up proj
    silu_el = 2 * B * df             # SiLU activation (2 FLOPs per elem)
    swiglu  = B * df                 # element-wise multiply
    down    = 2 * B * df * d         # down proj

    per_layer_attn = q_proj + kv_proj + o_proj + attn_qkt + attn_av
    per_layer_ffn  = gate_up + silu_el + swiglu + down
    per_layer      = per_layer_attn + per_layer_ffn

    lm_head = 2 * B * d * V

    total = L * per_layer + lm_head

    return {
        "per_layer_attn_gflop": per_layer_attn / 1e9,
        "per_layer_ffn_gflop":  per_layer_ffn  / 1e9,
        "per_layer_total_gflop":per_layer       / 1e9,
        "all_layers_gflop":     L * per_layer   / 1e9,
        "lm_head_gflop":        lm_head         / 1e9,
        "total_gflop":          total            / 1e9,
    }


def memory_bandwidth_bound_time_ms(cfg: ModelConfig, batch_size: int,
                                    hbm_bw_gbs: float = 2000.0) -> dict:
    """
    Model decode step time as memory-bandwidth-bound.

    The dominant cost is reading all weight matrices from HBM once.
    Each weight is read once per decode step regardless of batch size
    (until batch is large enough to become compute-bound).

    hbm_bw_gbs: HBM bandwidth in GB/s (A100 = 2000, H100 = 3350)
    """
    weight_bytes = cfg.weight_bytes()
    # Approx KV cache reads: 2 × n_kv_heads × d_head × seq_len × 2 bytes per layer
    # For avg_seq_len = 512, B sequences:
    avg_seq_len = 512
    kv_bytes_per_layer = (2 * cfg.n_kv_heads * cfg.d_head *
                          avg_seq_len * batch_size * 2)
    kv_bytes = cfg.n_layers * kv_bytes_per_layer

    total_bytes = weight_bytes + kv_bytes

    # Arithmetic intensity: FLOPs / bytes
    flops = flops_per_decode_step(cfg, batch_size)["total_gflop"] * 1e9
    arith_intensity = flops / total_bytes  # FLOPs/byte

    # Hardware roofline
    a100_peak_tflops = 312.0   # BF16
    a100_hbm_bw      = hbm_bw_gbs
    ridge_point = a100_peak_tflops * 1e12 / (a100_hbm_bw * 1e9)  # FLOPs/byte

    if arith_intensity < ridge_point:
        bound = "memory-bandwidth"
        time_ms = (total_bytes / (a100_hbm_bw * 1e9)) * 1000
    else:
        bound = "compute"
        time_ms = (flops / (a100_peak_tflops * 1e12)) * 1000

    return {
        "weight_bytes_gb":       weight_bytes / 1e9,
        "kv_bytes_gb":           kv_bytes / 1e9,
        "total_bytes_gb":        total_bytes / 1e9,
        "arithmetic_intensity":  arith_intensity,
        "ridge_point":           ridge_point,
        "bound":                 bound,
        "theoretical_time_ms":   time_ms,
    }


def section1_flop_accounting():
    print("=" * 70)
    print("§1  FLOP Accounting and Memory-Bandwidth-Bound Model")
    print("=" * 70)

    for cfg in [LLAMA3_8B, LLAMA3_70B]:
        print(f"\n{'─'*50}")
        print(f"  Model: {cfg.name}")
        print(f"{'─'*50}")

        for B in [1, 8, 64]:
            flops = flops_per_decode_step(cfg, B)
            bw    = memory_bandwidth_bound_time_ms(cfg, B)
            print(f"\n  Batch size B={B}:")
            print(f"    Per-layer attention:  {flops['per_layer_attn_gflop']:.2f} GFLOP")
            print(f"    Per-layer FFN:        {flops['per_layer_ffn_gflop']:.2f} GFLOP")
            print(f"    All {cfg.n_layers} layers:         {flops['all_layers_gflop']:.1f} GFLOP")
            print(f"    lm_head:              {flops['lm_head_gflop']:.1f} GFLOP")
            print(f"    Total:                {flops['total_gflop']:.1f} GFLOP")
            print(f"    Weights:              {bw['weight_bytes_gb']:.2f} GB")
            print(f"    KV cache reads:       {bw['kv_bytes_gb']:.3f} GB")
            print(f"    Arithmetic intensity: {bw['arithmetic_intensity']:.1f} FLOP/byte "
                  f"(ridge={bw['ridge_point']:.0f})")
            print(f"    Bound:                {bw['bound']}")
            print(f"    Theoretical time:     {bw['theoretical_time_ms']:.2f} ms")


# ──────────────────────────────────────────────────────────────────────────────
# §2  CUDA Graph vs. Eager Python Overhead Model
# ──────────────────────────────────────────────────────────────────────────────

# Captured graph sizes (same as Chapter 8 §8.6)
CUDA_GRAPH_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 256]

def pad_to_graph_size(batch_size: int) -> int:
    """Return the smallest captured graph size >= batch_size."""
    for gs in CUDA_GRAPH_SIZES:
        if gs >= batch_size:
            return gs
    return CUDA_GRAPH_SIZES[-1]


def decode_step_timing_ms(batch_size: int,
                           gpu_compute_ms: float,
                           use_cuda_graph: bool) -> dict:
    """
    Model total decode step latency:
      eager path:       Python dispatch per layer
      cuda-graph path:  single cudaGraphLaunch call

    Times are approximate means from profiling LLaMA 3 8B on A100.
    """
    scheduler_ms   = 1.5    # Python scheduler loop
    sample_ms      = 0.7    # sampler + token update

    if use_cuda_graph:
        graph_size     = pad_to_graph_size(batch_size)
        # Padding overhead: proportional to wasted positions
        pad_overhead   = max(0, (graph_size - batch_size)) * 0.001
        fill_buffer_ms = 0.1 + pad_overhead
        launch_ms      = 0.01                        # single cudaGraphLaunch
        overhead_ms    = fill_buffer_ms + launch_ms
    else:
        # Eager: ~0.1 ms per layer for Python dispatch (32 layers = 3.2 ms)
        n_layers       = 32
        per_layer_py   = 0.094          # ~3 ms total for 32 layers
        overhead_ms    = n_layers * per_layer_py + 0.5  # + tensor setup

    total_ms = scheduler_ms + overhead_ms + gpu_compute_ms + sample_ms
    return {
        "scheduler_ms":  scheduler_ms,
        "overhead_ms":   overhead_ms,
        "gpu_compute_ms":gpu_compute_ms,
        "sample_ms":     sample_ms,
        "total_ms":      total_ms,
    }


def gpu_compute_ms_model(cfg: ModelConfig, batch_size: int,
                          hbm_bw_gbs: float = 2000.0) -> float:
    """Approximate GPU compute time (memory-bandwidth bound)."""
    r = memory_bandwidth_bound_time_ms(cfg, batch_size, hbm_bw_gbs)
    return r["theoretical_time_ms"]


def section2_cuda_graph_vs_eager():
    print("\n" + "=" * 70)
    print("§2  CUDA Graph vs. Eager Python Overhead Model")
    print("=" * 70)

    cfg = LLAMA3_8B
    print(f"\n  Model: {cfg.name}  (A100 80GB, HBM 2000 GB/s)")
    print(f"\n  {'B':>4}  {'Graph size':>10}  {'Eager (ms)':>12}  "
          f"{'CG (ms)':>10}  {'Speedup':>8}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}")

    for B in [1, 4, 8, 16, 32, 64]:
        gpu_ms    = gpu_compute_ms_model(cfg, B)
        eager     = decode_step_timing_ms(B, gpu_ms, use_cuda_graph=False)
        cg        = decode_step_timing_ms(B, gpu_ms, use_cuda_graph=True)
        gs        = pad_to_graph_size(B)
        speedup   = eager["total_ms"] / cg["total_ms"]
        print(f"  {B:>4}  {gs:>10}  {eager['total_ms']:>12.2f}  "
              f"{cg['total_ms']:>10.2f}  {speedup:>8.2f}×")

    # Detailed breakdown for B=1
    print(f"\n  Detailed breakdown for B=1:")
    gpu_ms = gpu_compute_ms_model(cfg, 1)
    for label, use_cg in [("Eager", False), ("CUDA Graph", True)]:
        t = decode_step_timing_ms(1, gpu_ms, use_cuda_graph=use_cg)
        print(f"\n    {label}:")
        print(f"      Python scheduler:  {t['scheduler_ms']:.2f} ms")
        print(f"      Dispatch overhead: {t['overhead_ms']:.2f} ms")
        print(f"      GPU compute:       {t['gpu_compute_ms']:.2f} ms")
        print(f"      Sample + update:   {t['sample_ms']:.2f} ms")
        print(f"      Total:             {t['total_ms']:.2f} ms")


# ──────────────────────────────────────────────────────────────────────────────
# §3  Tensor-Parallel AllReduce Cost
# ──────────────────────────────────────────────────────────────────────────────

def allreduce_time_us(n_gpus: int, data_bytes: int,
                       link_bandwidth_gbs: float) -> float:
    """
    Ring-AllReduce time in microseconds.

    time = 2 × (N-1)/N × data_size / bandwidth

    link_bandwidth_gbs: per-GPU unidirectional bandwidth (GB/s)
      NVLink 4th gen: 300 GB/s unidirectional (600 GB/s bidirectional)
      PCIe 4.0 x16:  ~16 GB/s unidirectional
    """
    factor = 2.0 * (n_gpus - 1) / n_gpus
    bw_bytes_per_sec = link_bandwidth_gbs * 1e9
    time_s = factor * data_bytes / bw_bytes_per_sec
    return time_s * 1e6   # µs


def section3_allreduce_cost():
    print("\n" + "=" * 70)
    print("§3  Tensor-Parallel AllReduce Cost")
    print("=" * 70)

    cfg = LLAMA3_70B   # 70B typically run on 8 GPUs TP=8
    n_gpus = 8
    B = 64

    # BF16 tensor: [B, d_model]
    data_bytes_bf16 = B * cfg.d_model * 2   # 2 bytes per BF16 element
    data_bytes_int8 = B * cfg.d_model * 1   # 1 byte per INT8 element

    nvlink_bw  = 300.0   # GB/s unidirectional (NVLink 4)
    pcie_bw    =  16.0   # GB/s unidirectional (PCIe 4.0 x16)

    n_allreduces_per_layer = 2   # after attn + after FFN

    print(f"\n  Model: {cfg.name}, TP={n_gpus}, B={B}")
    print(f"  AllReduce tensor: [{B}, {cfg.d_model}] BF16 = {data_bytes_bf16/1024:.1f} KB\n")

    for link_name, link_bw in [("NVLink-4 (300 GB/s)", nvlink_bw),
                                 ("PCIe 4.0 (16 GB/s)", pcie_bw)]:
        ar_bf16_us = allreduce_time_us(n_gpus, data_bytes_bf16, link_bw)
        ar_int8_us = allreduce_time_us(n_gpus, data_bytes_int8, link_bw)

        total_bf16_ms = (cfg.n_layers * n_allreduces_per_layer
                         * ar_bf16_us) / 1000
        total_int8_ms = (cfg.n_layers * n_allreduces_per_layer
                         * ar_int8_us) / 1000

        print(f"  {link_name}:")
        print(f"    Per AllReduce (BF16): {ar_bf16_us:.2f} µs")
        print(f"    Per AllReduce (INT8): {ar_int8_us:.2f} µs  (2× compressed)")
        print(f"    {cfg.n_layers} layers × {n_allreduces_per_layer} AllReduces:")
        print(f"      BF16: {total_bf16_ms:.2f} ms total")
        print(f"      INT8: {total_int8_ms:.2f} ms total  "
              f"({total_bf16_ms/total_int8_ms:.1f}× faster)")
        print()

    # Show sensitivity to batch size
    print(f"  AllReduce time vs. batch size (NVLink-4, {cfg.name}):\n")
    print(f"  {'B':>4}  {'Bytes (KB)':>12}  {'Per AR (µs)':>12}  "
          f"{'Total (ms)':>12}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*12}")
    for B in [1, 4, 16, 32, 64, 128]:
        db = B * cfg.d_model * 2
        ar_us = allreduce_time_us(n_gpus, db, nvlink_bw)
        tot_ms = cfg.n_layers * n_allreduces_per_layer * ar_us / 1000
        print(f"  {B:>4}  {db/1024:>12.1f}  {ar_us:>12.3f}  {tot_ms:>12.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# §4  Per-Layer Timing Breakdown
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerTimingEntry:
    component: str
    time_us: float
    memory_read_mb: float
    kernel: str


def per_layer_timing(cfg: ModelConfig, batch_size: int,
                      n_gpus: int = 1,
                      hbm_bw_gbs: float = 2000.0) -> List[LayerTimingEntry]:
    """
    Estimate per-component timing for one transformer layer.

    Based on arithmetic intensity analysis and reported A100 profiling numbers
    from vLLM's CUDA profiler traces.
    """
    B  = batch_size
    d  = cfg.d_model
    df = cfg.d_ffn
    hk = cfg.n_kv_heads
    dh = cfg.d_head
    bw = hbm_bw_gbs * 1e9   # bytes/sec

    def gemm_time_us(m: int, n: int, k: int, dtype_bytes: int = 2) -> Tuple[float, float]:
        """Returns (time_us, bytes_read_mb)."""
        weight_bytes = n * k * dtype_bytes
        act_bytes    = m * k * dtype_bytes
        out_bytes    = m * n * dtype_bytes
        total_bytes  = weight_bytes + act_bytes + out_bytes
        time_s = total_bytes / bw
        return time_s * 1e6, total_bytes / 1e6

    def norm_time_us(tokens: int, d: int) -> Tuple[float, float]:
        data = tokens * d * 2         # read once, write once
        return (data / bw) * 1e6, data / 1e6

    # 1. RMSNorm (input)
    rms_us, rms_mb = norm_time_us(B, d)

    # 2. QKV projection: [B, d] → [B, d] Q, [B, hk*dh] K, [B, hk*dh] V
    qkv_us, qkv_mb = gemm_time_us(B, d + 2*hk*dh, d)

    # 3. RoPE (applied to Q and K)
    rope_bytes = 2 * B * (d + hk*dh) * 2   # read + write
    rope_us = (rope_bytes / bw) * 1e6
    rope_mb = rope_bytes / 1e6

    # 4. KV cache write
    kv_write_bytes = 2 * B * hk * dh * 2
    kv_write_us = (kv_write_bytes / bw) * 1e6
    kv_write_mb = kv_write_bytes / 1e6

    # 5. FlashAttention — approximate; depends on history length
    # Dominant cost: reading K and V from KV cache (avg_seq_len=512)
    avg_seq_len = 512
    fa_bytes = 2 * hk * avg_seq_len * dh * B * 2   # K + V reads
    fa_us = (fa_bytes / bw) * 1e6
    fa_mb = fa_bytes / 1e6

    # 6. O projection: [B, d] → [B, d]
    o_us, o_mb = gemm_time_us(B, d, d)

    # 7. AllReduce (only applies in TP > 1)
    if n_gpus > 1:
        ar_data = B * d * 2
        ar_us = allreduce_time_us(n_gpus, ar_data, 300.0)
        ar_mb = ar_data / 1e6
    else:
        ar_us, ar_mb = 0.0, 0.0

    # 8. RMSNorm (post-attention)
    rms2_us, rms2_mb = norm_time_us(B, d)

    # 9. Gate+Up projection: [B, d] → [B, 2*df]
    gu_us, gu_mb = gemm_time_us(B, 2 * df, d)

    # 10. SiLU + element-wise mul (fused)
    silu_bytes = 2 * B * df * 2   # read gate+up, write gate
    silu_us = (silu_bytes / bw) * 1e6
    silu_mb = silu_bytes / 1e6

    # 11. Down projection: [B, df] → [B, d]
    down_us, down_mb = gemm_time_us(B, d, df)

    entries = [
        LayerTimingEntry("RMSNorm (input)",       rms_us,      rms_mb,      "rms_norm_cuda"),
        LayerTimingEntry("QKV projection",         qkv_us,      qkv_mb,      "cuBLAS GEMM"),
        LayerTimingEntry("RoPE",                   rope_us,     rope_mb,     "rope_cuda"),
        LayerTimingEntry("KV cache write",         kv_write_us, kv_write_mb, "reshape_and_cache"),
        LayerTimingEntry("FlashAttention",         fa_us,       fa_mb,       "flashattn_v2/v3"),
        LayerTimingEntry("O projection",           o_us,        o_mb,        "cuBLAS GEMM"),
        LayerTimingEntry("AllReduce",              ar_us,       ar_mb,       "NCCL AllReduce"),
        LayerTimingEntry("RMSNorm (post-attn)",    rms2_us,     rms2_mb,     "rms_norm_cuda"),
        LayerTimingEntry("Gate+Up projection",     gu_us,       gu_mb,       "cuBLAS GEMM"),
        LayerTimingEntry("SiLU+Mul (fused)",       silu_us,     silu_mb,     "silu_and_mul"),
        LayerTimingEntry("Down projection",        down_us,     down_mb,     "cuBLAS GEMM"),
        LayerTimingEntry("AllReduce (FFN)",        ar_us,       ar_mb,       "NCCL AllReduce"),
    ]
    return entries


def section4_per_layer_timing():
    print("\n" + "=" * 70)
    print("§4  Per-Layer Timing Breakdown")
    print("=" * 70)

    cfg    = LLAMA3_8B
    B      = 64
    n_gpus = 8

    entries = per_layer_timing(cfg, B, n_gpus=n_gpus)
    total_us  = sum(e.time_us    for e in entries)
    total_mb  = sum(e.memory_read_mb for e in entries)

    print(f"\n  Model: {cfg.name}, B={B}, TP={n_gpus}, A100 (HBM 2000 GB/s)\n")
    print(f"  {'Component':<28}  {'Time (µs)':>10}  {'Mem read (MB)':>14}  "
          f"{'Kernel':<25}  {'% time':>7}")
    print(f"  {'─'*28}  {'─'*10}  {'─'*14}  {'─'*25}  {'─'*7}")

    for e in entries:
        pct = 100 * e.time_us / total_us if total_us > 0 else 0
        print(f"  {e.component:<28}  {e.time_us:>10.1f}  {e.memory_read_mb:>14.1f}  "
              f"  {e.kernel:<25}  {pct:>6.1f}%")

    print(f"  {'─'*28}  {'─'*10}  {'─'*14}  {'─'*25}  {'─'*7}")
    print(f"  {'Per-layer total':<28}  {total_us:>10.1f}  {total_mb:>14.1f}")
    print(f"  {'×' + str(cfg.n_layers) + ' layers':<28}  "
          f"{total_us*cfg.n_layers/1000:>10.1f} ms  "
          f"{total_mb*cfg.n_layers:>14.1f} MB total")


# ──────────────────────────────────────────────────────────────────────────────
# §5  GGML-Style DAG: Builder + Topological Executor
# ──────────────────────────────────────────────────────────────────────────────

class OpType(Enum):
    INPUT     = auto()    # leaf input tensor
    MUL_MAT   = auto()    # matrix multiply
    ADD       = auto()    # element-wise add
    RMS_NORM  = auto()    # RMS normalization
    SILU      = auto()    # Sigmoid Linear Unit activation
    MUL       = auto()    # element-wise multiply
    ROPE      = auto()    # rotary position embedding
    PERMUTE   = auto()    # axis reorder
    COPY      = auto()    # copy into KV cache slice
    RESHAPE   = auto()    # reshape (view — zero-copy)
    SCALE     = auto()    # multiply by scalar
    SOFT_MAX  = auto()    # row-wise softmax


class Backend(Enum):
    CPU  = "CPU"
    CUDA = "CUDA"
    METAL = "Metal"


@dataclass
class GGMLTensor:
    """
    Represents a node in the GGML compute DAG.
    In real GGML this is `struct ggml_tensor` allocated inside the arena.
    Here we simulate it with a Python dataclass.
    """
    name: str
    shape: Tuple[int, ...]          # logical shape (rows, cols, ...)
    op: OpType
    src: List["GGMLTensor"] = field(default_factory=list)
    backend: Backend = Backend.CPU
    # For simulation: store a float "value" to verify correctness
    _value: Optional[List[float]] = field(default=None, repr=False)
    visited: bool = False


class GGMLContext:
    """
    Minimal simulation of ggml_context — arena allocator + DAG builder.

    In real GGML this is:
      struct ggml_context { void* mem_buffer; size_t mem_size; ... };
    """

    def __init__(self, name: str = "ctx", arena_mb: float = 4.0):
        self.name      = name
        self._nodes: List[GGMLTensor] = []
        self._arena_mb = arena_mb
        self._used_mb  = 0.0
        self._id       = 0

    def _alloc(self, shape: Tuple[int, ...]) -> float:
        """Simulate arena allocation; raise if exhausted."""
        # Each element ~ 4 bytes (F32)
        elems = 1
        for s in shape:
            elems *= s
        size_mb = elems * 4 / 1e6
        self._used_mb += size_mb
        if self._used_mb > self._arena_mb:
            raise MemoryError(
                f"[COMMON TRAP] GGML arena exhausted: "
                f"used {self._used_mb:.2f} MB > {self._arena_mb:.2f} MB limit. "
                f"Increase arena size!"
            )
        return size_mb

    def _new(self, name: str, shape: Tuple[int, ...], op: OpType,
             src: List[GGMLTensor], value=None) -> GGMLTensor:
        self._alloc(shape)
        self._id += 1
        t = GGMLTensor(
            name=f"{name}_{self._id}",
            shape=shape,
            op=op,
            src=list(src),
        )
        t._value = value
        self._nodes.append(t)
        return t

    # ── Leaf (input) tensors ─────────────────────────────────────────

    def new_input(self, name: str, shape: Tuple[int, ...],
                  value: List[float]) -> GGMLTensor:
        return self._new(name, shape, OpType.INPUT, [], value)

    # ── Graph-building ops (each corresponds to a ggml_* function) ───

    def mul_mat(self, A: GGMLTensor, B: GGMLTensor,
                name: str = "mul_mat") -> GGMLTensor:
        # GGML convention: A=[ne0, ne1], B=[ne0, ne2] → out=[ne1, ne2]
        # ggml_mul_mat(A, B) computes A^T × B.
        # Requirement: A.shape[0] == B.shape[0]  (shared inner dimension ne0)
        assert A.shape[0] == B.shape[0], (
            f"mul_mat inner-dim mismatch: A.shape[0]={A.shape[0]} "
            f"!= B.shape[0]={B.shape[0]}  (A={A.name} {A.shape}, B={B.name} {B.shape})"
        )
        ne1 = A.shape[1] if len(A.shape) > 1 else 1
        ne2 = B.shape[1] if len(B.shape) > 1 else 1
        out_shape = (ne1, ne2)
        return self._new(name, out_shape, OpType.MUL_MAT, [A, B])

    def add(self, a: GGMLTensor, b: GGMLTensor,
            name: str = "add") -> GGMLTensor:
        assert a.shape == b.shape, f"Shape mismatch in add: {a.shape} vs {b.shape}"
        return self._new(name, a.shape, OpType.ADD, [a, b])

    def rms_norm(self, x: GGMLTensor, eps: float = 1e-5,
                 name: str = "rms_norm") -> GGMLTensor:
        return self._new(name, x.shape, OpType.RMS_NORM, [x])

    def silu(self, x: GGMLTensor, name: str = "silu") -> GGMLTensor:
        return self._new(name, x.shape, OpType.SILU, [x])

    def mul(self, a: GGMLTensor, b: GGMLTensor,
            name: str = "mul") -> GGMLTensor:
        # GGML ggml_mul: element-wise multiply with broadcasting.
        # In column-major layout (ne0, ne1):
        #   ne0 is the innermost dimension — must match.
        #   ne1 can be 1 in b (broadcast over a's ne1).
        if a.shape != b.shape:
            # ne0 must match
            assert a.shape[0] == b.shape[0], (
                f"Shape mismatch in mul (ne0): {a.shape} vs {b.shape}"
            )
            # ne1 of b must be 1 (broadcast) or match a's ne1
            b_ne1 = b.shape[1] if len(b.shape) > 1 else 1
            a_ne1 = a.shape[1] if len(a.shape) > 1 else 1
            assert b_ne1 == 1 or b_ne1 == a_ne1, (
                f"Shape mismatch in mul (ne1): {a.shape} vs {b.shape}"
            )
        out_shape = a.shape   # output has a's shape (broadcast result)
        return self._new(name, out_shape, OpType.MUL, [a, b])

    def rope(self, x: GGMLTensor, positions: GGMLTensor,
             name: str = "rope") -> GGMLTensor:
        return self._new(name, x.shape, OpType.ROPE, [x, positions])

    def scale(self, x: GGMLTensor, s: float,
              name: str = "scale") -> GGMLTensor:
        return self._new(name, x.shape, OpType.SCALE, [x])

    def soft_max(self, x: GGMLTensor, name: str = "soft_max") -> GGMLTensor:
        return self._new(name, x.shape, OpType.SOFT_MAX, [x])

    def reshape(self, x: GGMLTensor, new_shape: Tuple[int, ...],
                name: str = "reshape") -> GGMLTensor:
        return self._new(name, new_shape, OpType.RESHAPE, [x])

    def permute(self, x: GGMLTensor, name: str = "permute") -> GGMLTensor:
        # Swap first two dims for simplicity
        if len(x.shape) >= 2:
            new_shape = (x.shape[1], x.shape[0]) + x.shape[2:]
        else:
            new_shape = x.shape
        return self._new(name, new_shape, OpType.PERMUTE, [x])


class GGMLGraph:
    """
    Simulation of `struct ggml_cgraph`.
    Holds the topological execution order.
    """

    def __init__(self, ctx: GGMLContext):
        self.ctx   = ctx
        self._order: List[GGMLTensor] = []

    def build_from(self, output: GGMLTensor):
        """
        Topological sort (DFS post-order) from `output` node back to inputs.
        This mirrors ggml_build_forward_expand() in the real library.
        """
        self._order.clear()
        visited: set = set()

        def dfs(node: GGMLTensor):
            if id(node) in visited:
                return
            visited.add(id(node))
            for src in node.src:
                dfs(src)
            self._order.append(node)

        dfs(output)

    def compute(self, backend: Backend = Backend.CPU,
                n_threads: int = 4) -> None:
        """
        Execute the graph in topological order.
        Dispatches each op to the named backend (simulated).
        Mirrors ggml_graph_compute(gf, &plan).
        """
        for node in self._order:
            self._execute_node(node, backend)

    def _execute_node(self, node: GGMLTensor, backend: Backend) -> None:
        """Simulated op dispatch — prints what the real library would do."""
        if node.op == OpType.INPUT:
            return    # nothing to compute for leaf inputs

        src_names = ", ".join(s.name for s in node.src)
        op_name   = node.op.name
        shape_str = "×".join(str(s) for s in node.shape)
        # In real GGML this dispatches to:
        #   CUDA  → ggml_cuda_op_*()
        #   CPU   → ggml_compute_forward_*()
        #   Metal → ggml_metal_compute_tensor()
        node.backend = backend

    def print_graph(self):
        """Print the execution plan — analogous to ggml_graph_print()."""
        print(f"\n  GGML compute graph ({len(self._order)} nodes):\n")
        print(f"  {'#':>3}  {'Name':<30}  {'Op':<12}  {'Shape':<18}  {'Inputs'}")
        print(f"  {'─'*3}  {'─'*30}  {'─'*12}  {'─'*18}  {'─'*40}")
        for i, node in enumerate(self._order):
            shape_str = "(" + ", ".join(str(s) for s in node.shape) + ")"
            src_str   = ", ".join(s.name for s in node.src[:3])
            if len(node.src) > 3:
                src_str += "..."
            print(f"  {i:>3}  {node.name:<30}  {node.op.name:<12}  "
                  f"{shape_str:<18}  {src_str}")


def build_transformer_layer(ctx: GGMLContext,
                             cur: GGMLTensor,
                             d_model: int,
                             d_ffn: int,
                             n_heads: int,
                             n_kv_heads: int,
                             layer_idx: int) -> GGMLTensor:
    """
    Build a single LLaMA-style transformer layer in the GGML compute graph.

    Shape convention: GGML column-major — tensors are stored as
      (inner_dim, outer_dim) = (ne0, ne1).
    So:
      hidden state:  (d_model, n_tokens)
      weight matrix: (d_in, d_out)    ← ggml_mul_mat(W, x) computes W^T × x
                                         → result = (d_out, n_tokens)

    ggml_mul_mat(A, B) rule: A=[ne0,ne1], B=[ne0,ne2] → out=[ne1,ne2]

    Mirrors build_llama_layer() from Chapter 9 §9.5.3.
    """
    d_head    = d_model // n_heads
    n_kv_dim  = n_kv_heads * d_head   # dimension of K and V projections
    n_tokens  = cur.shape[1]           # cur = (d_model, n_tokens)
    pfx       = f"L{layer_idx}"

    # ── Weights: shape (d_in, d_out) ──────────────────────────────────
    # W_q: projects d_model → d_model  →  shape (d_model, d_model)
    # W_k: projects d_model → n_kv_dim →  shape (d_model, n_kv_dim)
    # W_v: same as W_k
    # W_o: projects d_model → d_model  →  shape (d_model, d_model)
    # W_norm: element-wise scale, shape (d_model, 1)  ← broadcast over n_tokens
    # W_gate, W_up: d_model → d_ffn    →  shape (d_model, d_ffn)
    # W_down: d_ffn  → d_model         →  shape (d_ffn,   d_model)

    def _w(name, shape):
        return ctx.new_input(f"{pfx}_{name}", shape, [0.01] * (shape[0] * shape[1]))

    def _norm_w(name):
        # Norm weight: (d_model, 1) — broadcast across tokens
        return ctx.new_input(f"{pfx}_{name}", (d_model, 1), [1.0] * d_model)

    W_norm1 = _norm_w("attn_norm")
    W_q     = _w("Wq",    (d_model, d_model))
    W_k     = _w("Wk",    (d_model, n_kv_dim))
    W_v     = _w("Wv",    (d_model, n_kv_dim))
    W_o     = _w("Wo",    (d_model, d_model))
    W_norm2 = _norm_w("ffn_norm")
    W_gate  = _w("Wgate", (d_model, d_ffn))
    W_up    = _w("Wup",   (d_model, d_ffn))
    W_down  = _w("Wdown", (d_ffn,   d_model))

    positions = ctx.new_input(f"{pfx}_pos", (n_tokens, 1), list(range(n_tokens)))

    residual = cur   # (d_model, n_tokens)

    # ── Attention block ────────────────────────────────────────────────
    # rms_norm: (d_model, n_tokens) → (d_model, n_tokens)
    cur  = ctx.rms_norm(cur,    name=f"{pfx}_rms_norm1")
    # scale by learned weight: broadcast (d_model,1) across n_tokens
    cur  = ctx.mul(cur, W_norm1, name=f"{pfx}_scale_norm1")
    # cur is still (d_model, n_tokens)

    # Q projection: mul_mat(W_q, cur)
    #   W_q=[d_model, d_model], cur=[d_model, n_tokens]
    #   → A=[ne0=d_model, ne1=d_model], B=[ne0=d_model, ne2=n_tokens]
    #   → out=[ne1=d_model, ne2=n_tokens]  i.e., (d_model, n_tokens) ✓
    Q = ctx.mul_mat(W_q, cur, name=f"{pfx}_Q")   # (d_model, n_tokens)
    K = ctx.mul_mat(W_k, cur, name=f"{pfx}_K")   # (n_kv_dim, n_tokens)
    V = ctx.mul_mat(W_v, cur, name=f"{pfx}_V")   # (n_kv_dim, n_tokens)

    # Apply RoPE to Q and K
    Q = ctx.rope(Q, positions, name=f"{pfx}_rope_Q")   # (d_model, n_tokens)
    K = ctx.rope(K, positions, name=f"{pfx}_rope_K")   # (n_kv_dim, n_tokens)

    # Attention scores: QK^T then softmax then weighted sum with V.
    # In GGML:  ggml_mul_mat(K, Q)  where K=[n_kv_dim, n_tokens], Q=[d_model, n_tokens]
    # This requires K.ne0 == Q.ne0 (both share the head dimension after reshape).
    # For the simulation we fold this into a single flash_attn node to keep shapes tidy.
    # (In real llama.cpp this is ggml_flash_attn_ext or the tiled QK^T chain.)
    flash_attn = ctx._new(
        f"{pfx}_flash_attn",
        (d_model, n_tokens),     # output: context vectors (d_model, n_tokens)
        OpType.MUL_MAT,          # closest op type — represents the attn kernel
        [Q, K, V],
    )

    # Output projection: W_o · flash_attn
    cur  = ctx.mul_mat(W_o, flash_attn, name=f"{pfx}_O_proj")  # (d_model, n_tokens)
    cur  = ctx.add(cur, residual,        name=f"{pfx}_attn_res") # (d_model, n_tokens)
    residual = cur

    # ── FFN block (SwiGLU) ─────────────────────────────────────────────
    cur   = ctx.rms_norm(cur,    name=f"{pfx}_rms_norm2")       # (d_model, n_tokens)
    cur   = ctx.mul(cur, W_norm2, name=f"{pfx}_scale_norm2")    # (d_model, n_tokens)

    # gate and up projections: W_gate=[d_model, d_ffn], cur=[d_model, n_tokens]
    # → A.ne0=d_model == B.ne0=d_model → out=[d_ffn, n_tokens] ✓
    gate  = ctx.mul_mat(W_gate, cur,  name=f"{pfx}_gate")       # (d_ffn, n_tokens)
    up    = ctx.mul_mat(W_up,   cur,  name=f"{pfx}_up")         # (d_ffn, n_tokens)
    gate  = ctx.silu(gate,            name=f"{pfx}_silu")        # (d_ffn, n_tokens)
    gate  = ctx.mul(gate, up,         name=f"{pfx}_swiglu")     # (d_ffn, n_tokens)

    # down projection: W_down=[d_ffn, d_model], gate=[d_ffn, n_tokens]
    # → A.ne0=d_ffn == B.ne0=d_ffn → out=[d_model, n_tokens] ✓
    cur   = ctx.mul_mat(W_down, gate, name=f"{pfx}_down")       # (d_model, n_tokens)
    cur   = ctx.add(cur, residual,    name=f"{pfx}_ffn_res")    # (d_model, n_tokens)

    return cur  # (d_model, n_tokens)


def section5_ggml_dag():
    print("\n" + "=" * 70)
    print("§5  GGML-Style DAG: Builder + Topological Executor")
    print("=" * 70)

    # Build a 1-layer transformer graph
    # Use tiny dimensions so the arena stays small
    d_model   = 16
    d_ffn     = 32
    n_heads   = 4
    n_kv_heads= 2
    n_tokens  = 2

    # Arena: 4 MB (enough for a tiny model)
    ctx = GGMLContext("demo_ctx", arena_mb=4.0)

    # Input: hidden state — GGML column-major: (d_model, n_tokens)
    hidden = ctx.new_input("hidden", (d_model, n_tokens),
                            [0.5] * (d_model * n_tokens))

    t0 = time.perf_counter()
    out = build_transformer_layer(
        ctx, hidden,
        d_model=d_model, d_ffn=d_ffn,
        n_heads=n_heads, n_kv_heads=n_kv_heads,
        layer_idx=0
    )
    build_us = (time.perf_counter() - t0) * 1e6

    print(f"\n  Graph built in {build_us:.1f} µs")
    print(f"  Arena used: {ctx._used_mb:.3f} MB / {ctx._arena_mb:.1f} MB")
    print(f"  Total nodes in ctx: {len(ctx._nodes)}")

    # Topological sort + print
    gf = GGMLGraph(ctx)
    t0 = time.perf_counter()
    gf.build_from(out)
    sort_us = (time.perf_counter() - t0) * 1e6
    print(f"  Topological sort:   {sort_us:.1f} µs")
    print(f"  Execution nodes:    {len(gf._order)}")

    gf.print_graph()

    # Execute (simulated)
    t0 = time.perf_counter()
    gf.compute(backend=Backend.CPU)
    exec_us = (time.perf_counter() - t0) * 1e6
    print(f"\n  Simulated compute:  {exec_us:.1f} µs  "
          f"(real dispatch: AVX2 for CPU, cuBLAS for CUDA)")

    # Demonstrate arena exhaustion trap
    print(f"\n  [COMMON TRAP] Arena exhaustion demo:")
    tiny_ctx = GGMLContext("tiny", arena_mb=0.001)  # 1 KB — way too small
    # hidden shape: (d_model, n_tokens) = (16, 2)
    tiny_in  = tiny_ctx.new_input("x", (16, 2), [0.0] * 32)
    try:
        _ = build_transformer_layer(
            tiny_ctx, tiny_in,
            d_model=16, d_ffn=32,
            n_heads=4, n_kv_heads=2, layer_idx=0
        )
    except MemoryError as e:
        print(f"    Caught: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# §6  Block-Wise Dequantization Simulation (Q4 × F32)
# ──────────────────────────────────────────────────────────────────────────────

def quantize_q4(weights: List[float], block_size: int = 32) -> Tuple[List[int], List[float]]:
    """
    Quantize a flat list of FP32 weights to Q4 (4-bit symmetric).

    Per block of `block_size` weights:
      scale = max(abs(w)) / 7   (7 = 2^3 - 1, signed 4-bit range [-8..7])
      q = round(w / scale) + 8  (shift to unsigned [0..15])

    Returns:
      q4_blocks: list of 4-bit values (stored as ints 0..15)
      scales:    list of per-block FP32 scales
    """
    n = len(weights)
    assert n % block_size == 0, "weight count must be multiple of block_size"
    n_blocks = n // block_size

    q4_blocks: List[int]   = []
    scales:    List[float] = []

    for b in range(n_blocks):
        block = weights[b * block_size : (b+1) * block_size]
        max_abs = max(abs(w) for w in block) or 1.0
        scale   = max_abs / 7.0
        scales.append(scale)
        for w in block:
            q = int(round(w / scale))          # [-7, 7]
            q = max(-8, min(7, q))             # clamp
            q4_blocks.append(q + 8)            # shift → [0, 15]

    return q4_blocks, scales


def dequant_dot_q4_f32(q4: List[int], scales: List[float],
                        activations: List[float],
                        block_size: int = 32) -> float:
    """
    Compute dot product of a Q4-quantised weight row with FP32 activations.

    Dequantises one block at a time — never materialises the full weight row.
    This is the inner loop of ggml_mul_mat for Q4_K weights on CPU.

    Returns the dot product as a FP32 scalar.
    """
    n        = len(activations)
    n_blocks = n // block_size
    dot      = 0.0

    for b in range(n_blocks):
        scale     = scales[b]
        act_block = activations[b * block_size : (b+1) * block_size]
        q4_block  = q4[b * block_size : (b+1) * block_size]

        # Dequantise on the fly: (q - 8) * scale
        for i in range(block_size):
            w_float = (q4_block[i] - 8) * scale
            dot    += w_float * act_block[i]

    return dot


def dequant_full_row(q4: List[int], scales: List[float],
                     block_size: int = 32) -> List[float]:
    """Dequantise an entire Q4 row to FP32 (for verification only)."""
    result = []
    n_blocks = len(q4) // block_size
    for b in range(n_blocks):
        scale     = scales[b]
        q4_block  = q4[b * block_size : (b+1) * block_size]
        for q in q4_block:
            result.append((q - 8) * scale)
    return result


def section6_blockwise_dequant():
    print("\n" + "=" * 70)
    print("§6  Block-Wise Dequantization: Q4 × F32 Dot Product")
    print("=" * 70)

    random.seed(42)
    block_size = 32
    n_blocks   = 8
    n          = block_size * n_blocks   # 256 weights

    # Reference weights in FP32
    weights_f32 = [random.uniform(-1.0, 1.0) for _ in range(n)]
    activations = [random.uniform(-1.0, 1.0) for _ in range(n)]

    # Quantize to Q4
    q4, scales = quantize_q4(weights_f32, block_size)

    # Block-wise dequant dot (the real GGML path)
    dot_q4 = dequant_dot_q4_f32(q4, scales, activations, block_size)

    # Reference: dequantise first, then dot (for verification)
    w_dequant   = dequant_full_row(q4, scales, block_size)
    dot_ref     = sum(w * a for w, a in zip(w_dequant, activations))

    # True FP32 reference (ideal)
    dot_f32     = sum(w * a for w, a in zip(weights_f32, activations))

    print(f"\n  n = {n} weights, block_size = {block_size}")
    print(f"  n_blocks = {n_blocks}\n")
    print(f"  Dot product (FP32 weights, ideal):     {dot_f32:.6f}")
    print(f"  Dot product (via full dequant then ×): {dot_ref:.6f}")
    print(f"  Dot product (block-wise, GGML path):   {dot_q4:.6f}")
    print(f"\n  Error (block-wise vs full-dequant):  "
          f"{abs(dot_q4 - dot_ref):.2e}  [should be 0.0]")
    print(f"  Quantization error (Q4 vs FP32):     "
          f"{abs(dot_q4 - dot_f32):.4f}  "
          f"({100*abs(dot_q4-dot_f32)/max(abs(dot_f32),1e-9):.2f}%)")

    # Show memory savings
    f32_bytes = n * 4
    q4_bytes  = n // 2          # 4 bits per weight = 0.5 bytes
    scale_bytes = n_blocks * 4  # one FP32 scale per block
    total_q4  = q4_bytes + scale_bytes
    print(f"\n  Memory:")
    print(f"    FP32 row ({n} weights):  {f32_bytes} bytes")
    print(f"    Q4 data:                {q4_bytes} bytes")
    print(f"    Scales ({n_blocks} blocks):      {scale_bytes} bytes")
    print(f"    Q4 total:               {total_q4} bytes  "
          f"({100*total_q4/f32_bytes:.1f}% of FP32)")

    # Timing: block-wise vs. materialise-then-dot
    iterations = 10_000

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = dequant_dot_q4_f32(q4, scales, activations, block_size)
    blockwise_us = (time.perf_counter() - t0) / iterations * 1e6

    t0 = time.perf_counter()
    for _ in range(iterations):
        w_tmp = dequant_full_row(q4, scales, block_size)
        _     = sum(w * a for w, a in zip(w_tmp, activations))
    materialise_us = (time.perf_counter() - t0) / iterations * 1e6

    print(f"\n  Timing (Python simulation, {n}-weight row, {iterations} iters):")
    print(f"    Block-wise dequant + dot:  {blockwise_us:.2f} µs/iter")
    print(f"    Materialise then dot:      {materialise_us:.2f} µs/iter")


# ──────────────────────────────────────────────────────────────────────────────
# §7  End-to-End Decode Step Timing Comparison
# ──────────────────────────────────────────────────────────────────────────────

def section7_end_to_end_comparison():
    print("\n" + "=" * 70)
    print("§7  End-to-End Decode Step Timing Comparison")
    print("=" * 70)

    print("""
  This section compares estimated end-to-end decode step latency for:

    vLLM (CUDA graph)   — LLaMA 3 8B on A100 80GB, HBM 2000 GB/s
    llama.cpp (CPU)     — LLaMA 3 8B Q4_K_M on M2 Max (Apple Silicon, 400 GB/s)
    llama.cpp (CUDA)    — LLaMA 3 8B Q4_K_M on RTX 4090, HBM 1008 GB/s

  Numbers are analytical models, not empirical measurements.
    """)

    # ── vLLM on A100 ──────────────────────────────────────────────────
    cfg = LLAMA3_8B
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  vLLM on A100 80GB  (B=64, BF16, CUDA graph)           │")
    print("  └─────────────────────────────────────────────────────────┘")

    for B in [1, 8, 32, 64]:
        gpu_ms = memory_bandwidth_bound_time_ms(cfg, B, 2000)["theoretical_time_ms"]
        cg     = decode_step_timing_ms(B, gpu_ms, use_cuda_graph=True)
        eager  = decode_step_timing_ms(B, gpu_ms, use_cuda_graph=False)
        print(f"    B={B:>3}: CUDA graph {cg['total_ms']:>6.2f} ms  "
              f"eager {eager['total_ms']:>6.2f} ms  "
              f"speedup {eager['total_ms']/cg['total_ms']:.2f}×")

    # ── llama.cpp on CPU (M2 Max) ──────────────────────────────────────
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  llama.cpp Q4_K_M on M2 Max CPU  (B=1, 400 GB/s UB)   │")
    print("  └─────────────────────────────────────────────────────────┘")

    # Q4_K_M is ~4.5 bpw average → weight_bytes = FP16 * 4.5/16
    cfg_q4 = ModelConfig(
        name="LLaMA-3-8B-Q4_K_M",
        n_layers=32, d_model=4096, d_ffn=14336,
        n_heads=32, n_kv_heads=8, vocab_size=128256,
        bpw=4.5
    )

    gpu_ms_cpu = memory_bandwidth_bound_time_ms(cfg_q4, 1, 400)["theoretical_time_ms"]
    # GGML overhead: build graph + topo sort < 0.5 ms, no Python
    ggml_overhead = 0.3    # ms
    total_cpu = ggml_overhead + gpu_ms_cpu
    print(f"    B=1:  GGML graph build: {ggml_overhead:.1f} ms  "
          f"compute: {gpu_ms_cpu:.2f} ms  "
          f"total: {total_cpu:.2f} ms")
    print(f"    Throughput: {1000/total_cpu:.1f} tokens/s")

    # ── llama.cpp on RTX 4090 ──────────────────────────────────────────
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  llama.cpp Q4_K_M on RTX 4090  (B=1, 1008 GB/s HBM)   │")
    print("  └─────────────────────────────────────────────────────────┘")

    gpu_ms_4090 = memory_bandwidth_bound_time_ms(cfg_q4, 1, 1008)["theoretical_time_ms"]
    total_4090  = ggml_overhead + gpu_ms_4090
    print(f"    B=1:  GGML graph build: {ggml_overhead:.1f} ms  "
          f"compute: {gpu_ms_4090:.2f} ms  "
          f"total: {total_4090:.2f} ms")
    print(f"    Throughput: {1000/total_4090:.1f} tokens/s")

    # ── Summary table ──────────────────────────────────────────────────
    print()
    print("  ┌────────────────────────────────────────────────────────────┐")
    print("  │  Summary: LLaMA 3 8B, B=1, theoretical roofline estimate  │")
    print("  └────────────────────────────────────────────────────────────┘")
    print(f"  {'Setup':<35}  {'Latency (ms)':>14}  {'Tok/s':>8}")
    print(f"  {'─'*35}  {'─'*14}  {'─'*8}")

    entries_e2e = [
        ("vLLM A100 80GB (BF16, CUDA graph)",
         decode_step_timing_ms(1, gpu_compute_ms_model(cfg, 1), True)["total_ms"]),
        ("vLLM A100 80GB (BF16, eager)",
         decode_step_timing_ms(1, gpu_compute_ms_model(cfg, 1), False)["total_ms"]),
        ("llama.cpp RTX 4090 (Q4_K_M)",  total_4090),
        ("llama.cpp M2 Max CPU (Q4_K_M)", total_cpu),
    ]

    for label, ms in entries_e2e:
        print(f"  {label:<35}  {ms:>14.2f}  {1000/ms:>8.1f}")

    print()
    print("  Note: theoretical roofline only. Actual results vary due to")
    print("  kernel efficiency, cache effects, and DRAM latency.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    section1_flop_accounting()
    section2_cuda_graph_vs_eager()
    section3_allreduce_cost()
    section4_per_layer_timing()
    section5_ggml_dag()
    section6_blockwise_dequant()
    section7_end_to_end_comparison()
    print("\n" + "=" * 70)
    print("All sections complete.")
    print("=" * 70)

```

## C++ — `forward_pass_demo.cpp`

```cpp
/*
 * Chapter 9 — The Forward Pass: CUDA vs. GGML
 * Companion code: forward_pass_demo.cpp
 *
 * Sections:
 *   1. FLOP accounting and memory-bandwidth-bound model
 *   2. CUDA graph vs. eager Python overhead model
 *   3. Tensor-parallel AllReduce cost (NVLink vs. PCIe)
 *   4. Per-layer timing breakdown
 *   5. GGML-style DAG: builder + topological executor
 *   6. Block-wise dequantization (Q4 × F32 dot product)
 *   7. End-to-end decode step timing comparison
 *
 * Compile:
 *   g++ -O2 -std=c++17 -o forward_pass_demo forward_pass_demo.cpp && ./forward_pass_demo
 *
 * No external dependencies.
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

static std::string repeat(const std::string& s, int n) {
    std::string r;
    for (int i = 0; i < n; ++i) r += s;
    return r;
}

static std::string pad_right(const std::string& s, int w) {
    if ((int)s.size() >= w) return s.substr(0, w);
    return s + std::string(w - s.size(), ' ');
}

static std::string pad_left(const std::string& s, int w) {
    if ((int)s.size() >= w) return s.substr(0, w);
    return std::string(w - s.size(), ' ') + s;
}

static std::string fmt_f(double v, int prec = 2) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  FLOP Accounting and Memory-Bandwidth-Bound Model
// ─────────────────────────────────────────────────────────────────────────────

struct ModelConfig {
    std::string name;
    int  n_layers;
    int  d_model;
    int  d_ffn;
    int  n_heads;
    int  n_kv_heads;
    int  vocab_size;
    double bpw = 16.0;      // bits per weight

    int d_head() const { return d_model / n_heads; }

    // Approximate total weight bytes
    double weight_bytes() const {
        double bpb = bpw / 8.0;
        // Embedding + lm_head
        double embed = 2.0 * vocab_size * d_model * bpb;
        // Per-layer: Q/K/V/O + gate/up/down + norms
        double attn  = 4.0 * d_model * d_model;
        double ffn   = 3.0 * d_model * d_ffn;
        double norm  = 4.0 * d_model;
        double per_layer = (attn + ffn + norm) * bpb;
        return embed + n_layers * per_layer;
    }
};

// LLaMA 3 configurations
static const ModelConfig LLAMA3_8B{
    "LLaMA-3-8B", 32, 4096, 14336, 32, 8, 128256, 16.0
};
static const ModelConfig LLAMA3_70B{
    "LLaMA-3-70B", 80, 8192, 28672, 64, 8, 128256, 16.0
};

struct FlopBreakdown {
    double per_layer_attn_gflop;
    double per_layer_ffn_gflop;
    double per_layer_total_gflop;
    double all_layers_gflop;
    double lm_head_gflop;
    double total_gflop;
};

FlopBreakdown flops_per_decode_step(const ModelConfig& cfg,
                                     int batch_size,
                                     int avg_seq_len = 512) {
    double B  = batch_size;
    double d  = cfg.d_model;
    double h  = cfg.n_heads;
    double hk = cfg.n_kv_heads;
    double dh = cfg.d_head();
    double df = cfg.d_ffn;
    double L  = cfg.n_layers;
    double V  = cfg.vocab_size;

    // Linear layers: 2 × B × d_in × d_out
    double q_proj   = 2 * B * d * d;
    double kv_proj  = 2 * 2 * B * d * (hk * dh);
    double o_proj   = 2 * B * d * d;
    double attn_qkt = 2 * B * h * avg_seq_len * dh;
    double attn_av  = 2 * B * h * avg_seq_len * dh;

    double gate_up  = 2 * 2 * B * d * df;
    double silu_el  = 2 * B * df;
    double swiglu   = B * df;
    double down     = 2 * B * df * d;

    double per_layer_attn = q_proj + kv_proj + o_proj + attn_qkt + attn_av;
    double per_layer_ffn  = gate_up + silu_el + swiglu + down;
    double per_layer      = per_layer_attn + per_layer_ffn;
    double lm_head        = 2 * B * d * V;
    double total          = L * per_layer + lm_head;

    return {
        per_layer_attn / 1e9,
        per_layer_ffn  / 1e9,
        per_layer      / 1e9,
        L * per_layer  / 1e9,
        lm_head        / 1e9,
        total          / 1e9,
    };
}

struct BandwidthBound {
    double weight_bytes_gb;
    double kv_bytes_gb;
    double total_bytes_gb;
    double arithmetic_intensity;
    double ridge_point;
    std::string bound;
    double theoretical_time_ms;
};

BandwidthBound memory_bandwidth_bound(const ModelConfig& cfg,
                                       int batch_size,
                                       double hbm_bw_gbs = 2000.0) {
    double wb = cfg.weight_bytes();
    int avg_seq_len = 512;
    double kv = cfg.n_layers * 2.0 * cfg.n_kv_heads * cfg.d_head()
                * avg_seq_len * batch_size * 2;
    double total   = wb + kv;

    auto flops     = flops_per_decode_step(cfg, batch_size);
    double flop_n  = flops.total_gflop * 1e9;
    double arith   = flop_n / total;

    double a100_peak_tflops = 312.0;
    double ridge   = a100_peak_tflops * 1e12 / (hbm_bw_gbs * 1e9);
    bool   bnd_mem = arith < ridge;

    double time_ms = bnd_mem
        ? (total / (hbm_bw_gbs * 1e9)) * 1000.0
        : (flop_n / (a100_peak_tflops * 1e12)) * 1000.0;

    return {
        wb / 1e9, kv / 1e9, total / 1e9,
        arith, ridge,
        bnd_mem ? "memory-bandwidth" : "compute",
        time_ms
    };
}

static void section1_flop_accounting() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "§1  FLOP Accounting and Memory-Bandwidth-Bound Model\n";
    std::cout << std::string(70, '=') << "\n";

    for (const auto& cfg : {LLAMA3_8B, LLAMA3_70B}) {
        std::cout << "\n" << std::string(50, '-') << "\n";
        std::cout << "  Model: " << cfg.name << "\n";
        std::cout << std::string(50, '-') << "\n";

        for (int B : {1, 8, 64}) {
            auto f = flops_per_decode_step(cfg, B);
            auto bw = memory_bandwidth_bound(cfg, B);
            std::cout << "\n  Batch size B=" << B << ":\n";
            std::cout << "    Per-layer attention:  "
                      << fmt_f(f.per_layer_attn_gflop) << " GFLOP\n";
            std::cout << "    Per-layer FFN:        "
                      << fmt_f(f.per_layer_ffn_gflop) << " GFLOP\n";
            std::cout << "    All " << cfg.n_layers << " layers:       "
                      << fmt_f(f.all_layers_gflop, 1) << " GFLOP\n";
            std::cout << "    Total:                "
                      << fmt_f(f.total_gflop, 1) << " GFLOP\n";
            std::cout << "    Weights:              "
                      << fmt_f(bw.weight_bytes_gb) << " GB\n";
            std::cout << "    Arithmetic intensity: "
                      << fmt_f(bw.arithmetic_intensity, 1) << " FLOP/byte"
                      << " (ridge=" << fmt_f(bw.ridge_point, 0) << ")\n";
            std::cout << "    Bound:                " << bw.bound << "\n";
            std::cout << "    Theoretical time:     "
                      << fmt_f(bw.theoretical_time_ms) << " ms\n";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  CUDA Graph vs. Eager Python Overhead Model
// ─────────────────────────────────────────────────────────────────────────────

static const std::vector<int> CUDA_GRAPH_SIZES =
    {1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 256};

static int pad_to_graph_size(int B) {
    for (int gs : CUDA_GRAPH_SIZES)
        if (gs >= B) return gs;
    return CUDA_GRAPH_SIZES.back();
}

struct StepTiming {
    double scheduler_ms;
    double overhead_ms;
    double gpu_compute_ms;
    double sample_ms;
    double total_ms() const {
        return scheduler_ms + overhead_ms + gpu_compute_ms + sample_ms;
    }
};

StepTiming decode_step_timing(int batch_size,
                               double gpu_compute_ms,
                               bool use_cuda_graph) {
    double sched_ms  = 1.5;
    double sample_ms = 0.7;
    double overhead_ms;

    if (use_cuda_graph) {
        int gs  = pad_to_graph_size(batch_size);
        double pad_overhead = std::max(0, gs - batch_size) * 0.001;
        overhead_ms = 0.1 + pad_overhead + 0.01;  // fill + launch
    } else {
        // ~0.094 ms per layer Python overhead + setup
        overhead_ms = 32 * 0.094 + 0.5;
    }

    return {sched_ms, overhead_ms, gpu_compute_ms, sample_ms};
}

static void section2_cuda_graph_vs_eager() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "§2  CUDA Graph vs. Eager Python Overhead Model\n";
    std::cout << std::string(70, '=') << "\n";

    const auto& cfg = LLAMA3_8B;
    std::cout << "\n  Model: " << cfg.name
              << "  (A100 80GB, HBM 2000 GB/s)\n";

    std::cout << "\n  " << pad_left("B",4) << "  " << pad_left("Graph",10)
              << "  " << pad_left("Eager(ms)",12)
              << "  " << pad_left("CG(ms)",10)
              << "  " << pad_left("Speedup",8) << "\n";
    std::cout << "  " << std::string(4,'-') << "  " << std::string(10,'-')
              << "  " << std::string(12,'-') << "  " << std::string(10,'-')
              << "  " << std::string(8,'-') << "\n";

    for (int B : {1, 4, 8, 16, 32, 64}) {
        auto bw     = memory_bandwidth_bound(cfg, B);
        double gm   = bw.theoretical_time_ms;
        auto eager  = decode_step_timing(B, gm, false);
        auto cg     = decode_step_timing(B, gm, true);
        int gs      = pad_to_graph_size(B);
        double spd  = eager.total_ms() / cg.total_ms();

        std::cout << "  " << pad_left(std::to_string(B), 4)
                  << "  " << pad_left(std::to_string(gs), 10)
                  << "  " << pad_left(fmt_f(eager.total_ms()), 12)
                  << "  " << pad_left(fmt_f(cg.total_ms()), 10)
                  << "  " << pad_left(fmt_f(spd) + "×", 8) << "\n";
    }

    // Detailed breakdown B=1
    std::cout << "\n  Detailed breakdown B=1:\n";
    auto bw = memory_bandwidth_bound(cfg, 1);
    for (auto [label, use_cg] : std::vector<std::pair<std::string,bool>>{
            {"Eager", false}, {"CUDA Graph", true}}) {
        auto t = decode_step_timing(1, bw.theoretical_time_ms, use_cg);
        std::cout << "\n    " << label << ":\n";
        std::cout << "      Python scheduler:  " << fmt_f(t.scheduler_ms) << " ms\n";
        std::cout << "      Dispatch overhead: " << fmt_f(t.overhead_ms) << " ms\n";
        std::cout << "      GPU compute:       " << fmt_f(t.gpu_compute_ms) << " ms\n";
        std::cout << "      Sample + update:   " << fmt_f(t.sample_ms) << " ms\n";
        std::cout << "      Total:             " << fmt_f(t.total_ms()) << " ms\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  Tensor-Parallel AllReduce Cost
// ─────────────────────────────────────────────────────────────────────────────

// Ring-AllReduce time in microseconds
static double allreduce_time_us(int n_gpus, long long data_bytes,
                                 double link_bw_gbs) {
    double factor = 2.0 * (n_gpus - 1) / (double)n_gpus;
    double bw_bytes = link_bw_gbs * 1e9;
    return factor * data_bytes / bw_bytes * 1e6;
}

static void section3_allreduce_cost() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "§3  Tensor-Parallel AllReduce Cost\n";
    std::cout << std::string(70, '=') << "\n";

    const auto& cfg = LLAMA3_70B;
    int n_gpus = 8;
    int B = 64;

    long long data_bf16 = (long long)B * cfg.d_model * 2;
    long long data_int8 = (long long)B * cfg.d_model * 1;
    int n_ar_per_layer = 2;

    std::cout << "\n  Model: " << cfg.name << ", TP=" << n_gpus
              << ", B=" << B << "\n";
    std::cout << "  AllReduce tensor: [" << B << ", " << cfg.d_model
              << "] BF16 = " << data_bf16/1024 << " KB\n\n";

    for (auto [link_name, link_bw] : std::vector<std::pair<std::string,double>>{
            {"NVLink-4 (300 GB/s)", 300.0},
            {"PCIe 4.0 (16 GB/s)", 16.0}}) {

        double ar_bf16 = allreduce_time_us(n_gpus, data_bf16, link_bw);
        double ar_int8 = allreduce_time_us(n_gpus, data_int8, link_bw);
        double tot_bf16 = cfg.n_layers * n_ar_per_layer * ar_bf16 / 1000;
        double tot_int8 = cfg.n_layers * n_ar_per_layer * ar_int8 / 1000;

        std::cout << "  " << link_name << ":\n";
        std::cout << "    Per AllReduce (BF16): " << fmt_f(ar_bf16) << " µs\n";
        std::cout << "    Per AllReduce (INT8): " << fmt_f(ar_int8) << " µs  (2× compressed)\n";
        std::cout << "    " << cfg.n_layers << " layers × "
                  << n_ar_per_layer << " AllReduces:\n";
        std::cout << "      BF16: " << fmt_f(tot_bf16) << " ms total\n";
        std::cout << "      INT8: " << fmt_f(tot_int8) << " ms total  ("
                  << fmt_f(tot_bf16/tot_int8, 1) << "× faster)\n\n";
    }

    // Table: AllReduce time vs batch size
    std::cout << "  AllReduce time vs. batch size (NVLink-4, " << cfg.name << "):\n\n";
    std::cout << "  " << pad_left("B",4)
              << "  " << pad_left("Bytes(KB)",12)
              << "  " << pad_left("Per AR(µs)",12)
              << "  " << pad_left("Total(ms)",12) << "\n";
    std::cout << "  " << std::string(4,'-') << "  " << std::string(12,'-')
              << "  " << std::string(12,'-') << "  " << std::string(12,'-') << "\n";

    for (int b : {1, 4, 16, 32, 64, 128}) {
        long long db = (long long)b * cfg.d_model * 2;
        double ar  = allreduce_time_us(n_gpus, db, 300.0);
        double tot = cfg.n_layers * n_ar_per_layer * ar / 1000;
        std::cout << "  " << pad_left(std::to_string(b), 4)
                  << "  " << pad_left(fmt_f(db/1024.0, 1), 12)
                  << "  " << pad_left(fmt_f(ar, 3), 12)
                  << "  " << pad_left(fmt_f(tot, 4), 12) << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Per-Layer Timing Breakdown
// ─────────────────────────────────────────────────────────────────────────────

struct LayerEntry {
    std::string component;
    double time_us;
    double mem_read_mb;
    std::string kernel;
};

std::vector<LayerEntry> per_layer_timing(const ModelConfig& cfg,
                                          int batch_size,
                                          int n_gpus = 1,
                                          double hbm_bw_gbs = 2000.0) {
    double B  = batch_size;
    double d  = cfg.d_model;
    double df = cfg.d_ffn;
    double hk = cfg.n_kv_heads;
    double dh = cfg.d_head();
    double bw = hbm_bw_gbs * 1e9;

    auto gemm = [&](double m, double n, double k) -> std::pair<double,double> {
        double bytes = (n*k + m*k + m*n) * 2;   // weights + acts + output (BF16)
        return {bytes / bw * 1e6, bytes / 1e6};
    };
    auto norm_op = [&](double tokens, double dim) -> std::pair<double,double> {
        double bytes = tokens * dim * 4;     // read + write (FP32-like)
        return {bytes / bw * 1e6, bytes / 1e6};
    };

    auto [rms_us, rms_mb]   = norm_op(B, d);
    auto [qkv_us, qkv_mb]   = gemm(B, d + 2*hk*dh, d);
    double rope_bytes = 2*B*(d + hk*dh)*2;
    double rope_us    = rope_bytes / bw * 1e6;
    double rope_mb    = rope_bytes / 1e6;
    double kv_w_bytes = 2*B*hk*dh*2;
    double kv_w_us    = kv_w_bytes / bw * 1e6;
    double kv_w_mb    = kv_w_bytes / 1e6;
    int avg_seq = 512;
    double fa_bytes   = 2*hk*avg_seq*dh*B*2;
    double fa_us      = fa_bytes / bw * 1e6;
    double fa_mb      = fa_bytes / 1e6;
    auto [o_us, o_mb] = gemm(B, d, d);
    double ar_us = 0, ar_mb = 0;
    if (n_gpus > 1) {
        long long ar_data = (long long)(B * d * 2);
        ar_us = allreduce_time_us(n_gpus, ar_data, 300.0);
        ar_mb = ar_data / 1e6;
    }
    auto [rms2_us, rms2_mb] = norm_op(B, d);
    auto [gu_us, gu_mb]     = gemm(B, 2*df, d);
    double silu_bytes = 2*B*df*2;
    double silu_us    = silu_bytes / bw * 1e6;
    double silu_mb    = silu_bytes / 1e6;
    auto [dn_us, dn_mb]     = gemm(B, d, df);

    return {
        {"RMSNorm (input)",      rms_us,   rms_mb,  "rms_norm_cuda"},
        {"QKV projection",       qkv_us,   qkv_mb,  "cuBLAS GEMM"},
        {"RoPE",                 rope_us,  rope_mb, "rope_cuda"},
        {"KV cache write",       kv_w_us,  kv_w_mb, "reshape_and_cache"},
        {"FlashAttention",       fa_us,    fa_mb,   "flashattn_v2/v3"},
        {"O projection",         o_us,     o_mb,    "cuBLAS GEMM"},
        {"AllReduce (attn)",     ar_us,    ar_mb,   "NCCL AllReduce"},
        {"RMSNorm (post-attn)",  rms2_us,  rms2_mb, "rms_norm_cuda"},
        {"Gate+Up projection",   gu_us,    gu_mb,   "cuBLAS GEMM"},
        {"SiLU+Mul (fused)",     silu_us,  silu_mb, "silu_and_mul"},
        {"Down projection",      dn_us,    dn_mb,   "cuBLAS GEMM"},
        {"AllReduce (FFN)",      ar_us,    ar_mb,   "NCCL AllReduce"},
    };
}

static void section4_per_layer_timing() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "§4  Per-Layer Timing Breakdown\n";
    std::cout << std::string(70, '=') << "\n";

    const auto& cfg = LLAMA3_8B;
    int B = 64, n_gpus = 8;
    auto entries = per_layer_timing(cfg, B, n_gpus);

    double total_us = 0, total_mb = 0;
    for (auto& e : entries) { total_us += e.time_us; total_mb += e.mem_read_mb; }

    std::cout << "\n  Model: " << cfg.name << ", B=" << B
              << ", TP=" << n_gpus << ", A100 (HBM 2000 GB/s)\n\n";
    std::cout << "  " << pad_right("Component",28)
              << "  " << pad_left("Time(µs)",10)
              << "  " << pad_left("Mem(MB)",14)
              << "  " << pad_right("Kernel",25)
              << "  " << pad_left("%time",7) << "\n";
    std::cout << "  " << std::string(28,'-') << "  " << std::string(10,'-')
              << "  " << std::string(14,'-') << "  " << std::string(25,'-')
              << "  " << std::string(7,'-') << "\n";

    for (auto& e : entries) {
        double pct = total_us > 0 ? 100.0 * e.time_us / total_us : 0;
        std::cout << "  " << pad_right(e.component,28)
                  << "  " << pad_left(fmt_f(e.time_us, 1), 10)
                  << "  " << pad_left(fmt_f(e.mem_read_mb, 1), 14)
                  << "  " << pad_right("  "+e.kernel, 25)
                  << "  " << pad_left(fmt_f(pct, 1)+"%", 7) << "\n";
    }

    std::cout << "  " << std::string(28,'-') << "  " << std::string(10,'-')
              << "  " << std::string(14,'-') << "  " << std::string(25,' ')
              << "  " << std::string(7,'-') << "\n";
    std::cout << "  " << pad_right("Per-layer total",28)
              << "  " << pad_left(fmt_f(total_us,1), 10)
              << "  " << pad_left(fmt_f(total_mb,1), 14) << "\n";
    std::cout << "  × " << cfg.n_layers << " layers: "
              << fmt_f(total_us*cfg.n_layers/1000) << " ms\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  GGML-Style DAG: Builder + Topological Executor
// ─────────────────────────────────────────────────────────────────────────────

enum class GgmlOp {
    INPUT, MUL_MAT, ADD, RMS_NORM, SILU, MUL, ROPE, SCALE, SOFT_MAX, RESHAPE, PERMUTE,
};

static const char* op_name(GgmlOp op) {
    switch (op) {
        case GgmlOp::INPUT:    return "INPUT";
        case GgmlOp::MUL_MAT:  return "MUL_MAT";
        case GgmlOp::ADD:      return "ADD";
        case GgmlOp::RMS_NORM: return "RMS_NORM";
        case GgmlOp::SILU:     return "SILU";
        case GgmlOp::MUL:      return "MUL";
        case GgmlOp::ROPE:     return "ROPE";
        case GgmlOp::SCALE:    return "SCALE";
        case GgmlOp::SOFT_MAX: return "SOFT_MAX";
        case GgmlOp::RESHAPE:  return "RESHAPE";
        case GgmlOp::PERMUTE:  return "PERMUTE";
        default:               return "UNKNOWN";
    }
}

enum class GgmlBackend { CPU, CUDA, METAL };

/* Simulates struct ggml_tensor.
   Shapes follow GGML's column-major convention: (ne0, ne1, ...) where
   ne0 is the innermost ("fastest") dimension. */
struct GgmlTensor {
    std::string name;
    std::vector<int> shape;     // (ne0, ne1, ...)
    GgmlOp op;
    std::vector<GgmlTensor*> src;
    GgmlBackend backend = GgmlBackend::CPU;
    int id = 0;
};

/*
 * Simulates ggml_context — a pre-allocated arena that owns all tensor objects.
 *
 * Actual ggml_context:
 *   struct ggml_context { void* mem_buffer; size_t mem_size; ... };
 * All tensors are bump-allocated into mem_buffer.
 */
class GgmlContext {
public:
    std::string name;
    double arena_mb;

    GgmlContext(std::string n, double arena = 4.0)
        : name(std::move(n)), arena_mb(arena), used_mb_(0.0), next_id_(0) {}

    // Allocate and track a tensor
    GgmlTensor* alloc(const std::string& tname,
                      std::vector<int> shape,
                      GgmlOp op,
                      std::vector<GgmlTensor*> src = {}) {
        double elems = 1.0;
        for (int s : shape) elems *= s;
        double size_mb = elems * 4.0 / 1e6;   // simulate float32
        used_mb_ += size_mb;
        if (used_mb_ > arena_mb) {
            throw std::runtime_error(
                "[COMMON TRAP] GGML arena exhausted: used " +
                fmt_f(used_mb_) + " MB > " + fmt_f(arena_mb) +
                " MB limit. Increase arena size!");
        }
        tensors_.push_back(std::make_unique<GgmlTensor>());
        auto* t = tensors_.back().get();
        t->id    = ++next_id_;
        t->name  = tname + "_" + std::to_string(t->id);
        t->shape = std::move(shape);
        t->op    = op;
        t->src   = std::move(src);
        return t;
    }

    // ── Leaf ────────────────────────────────────────────────────────
    GgmlTensor* new_input(const std::string& nm, std::vector<int> shape) {
        return alloc(nm, std::move(shape), GgmlOp::INPUT, {});
    }

    // ── Ops ─────────────────────────────────────────────────────────

    // ggml_mul_mat(A, B): A=[ne0,ne1], B=[ne0,ne2] → [ne1, ne2]
    GgmlTensor* mul_mat(GgmlTensor* A, GgmlTensor* B,
                        const std::string& nm = "mul_mat") {
        assert(!A->shape.empty() && !B->shape.empty());
        assert(A->shape[0] == B->shape[0] &&
               "mul_mat: A.ne0 must equal B.ne0");
        int ne1 = A->shape.size() > 1 ? A->shape[1] : 1;
        int ne2 = B->shape.size() > 1 ? B->shape[1] : 1;
        return alloc(nm, {ne1, ne2}, GgmlOp::MUL_MAT, {A, B});
    }

    // ggml_add: element-wise, with broadcast (A.ne0 == B.ne0)
    GgmlTensor* add(GgmlTensor* a, GgmlTensor* b,
                    const std::string& nm = "add") {
        assert(a->shape[0] == b->shape[0] && "add: ne0 must match");
        return alloc(nm, a->shape, GgmlOp::ADD, {a, b});
    }

    // ggml_mul: element-wise, with broadcast
    GgmlTensor* mul(GgmlTensor* a, GgmlTensor* b,
                    const std::string& nm = "mul") {
        assert(a->shape[0] == b->shape[0] && "mul: ne0 must match");
        return alloc(nm, a->shape, GgmlOp::MUL, {a, b});
    }

    GgmlTensor* rms_norm(GgmlTensor* x, const std::string& nm = "rms_norm") {
        return alloc(nm, x->shape, GgmlOp::RMS_NORM, {x});
    }

    GgmlTensor* silu(GgmlTensor* x, const std::string& nm = "silu") {
        return alloc(nm, x->shape, GgmlOp::SILU, {x});
    }

    GgmlTensor* rope(GgmlTensor* x, GgmlTensor* pos,
                     const std::string& nm = "rope") {
        return alloc(nm, x->shape, GgmlOp::ROPE, {x, pos});
    }

    GgmlTensor* scale(GgmlTensor* x, const std::string& nm = "scale") {
        return alloc(nm, x->shape, GgmlOp::SCALE, {x});
    }

    GgmlTensor* soft_max(GgmlTensor* x, const std::string& nm = "soft_max") {
        return alloc(nm, x->shape, GgmlOp::SOFT_MAX, {x});
    }

    double used_mb() const { return used_mb_; }
    int    n_tensors() const { return (int)tensors_.size(); }

private:
    double used_mb_;
    int    next_id_;
    std::vector<std::unique_ptr<GgmlTensor>> tensors_;
};

/*
 * Simulates struct ggml_cgraph.
 * build_from() performs a DFS post-order topological sort
 * — identical to ggml_build_forward_expand().
 */
class GgmlGraph {
public:
    explicit GgmlGraph(GgmlContext& ctx) : ctx_(ctx) {}

    void build_from(GgmlTensor* output) {
        order_.clear();
        std::unordered_set<GgmlTensor*> visited;
        std::function<void(GgmlTensor*)> dfs = [&](GgmlTensor* t) {
            if (visited.count(t)) return;
            visited.insert(t);
            for (auto* s : t->src) dfs(s);
            order_.push_back(t);
        };
        dfs(output);
    }

    // Simulate ggml_graph_compute: dispatch each op to named backend
    void compute(GgmlBackend backend = GgmlBackend::CPU) {
        for (auto* node : order_) {
            node->backend = backend;
            // In real GGML: ggml_compute_forward() dispatches here
        }
    }

    void print_graph() const {
        std::cout << "\n  GGML compute graph (" << order_.size() << " nodes):\n\n";
        std::cout << "  " << pad_left("#",3) << "  " << pad_right("Name",30)
                  << "  " << pad_right("Op",12) << "  " << pad_right("Shape",18)
                  << "  " << "Inputs\n";
        std::cout << "  " << std::string(3,'-') << "  " << std::string(30,'-')
                  << "  " << std::string(12,'-') << "  " << std::string(18,'-')
                  << "  " << std::string(40,'-') << "\n";

        int i = 0;
        for (auto* t : order_) {
            std::string shape_str = "(";
            for (int si = 0; si < (int)t->shape.size(); si++) {
                if (si) shape_str += ", ";
                shape_str += std::to_string(t->shape[si]);
            }
            shape_str += ")";

            std::string src_str;
            int show = std::min((int)t->src.size(), 3);
            for (int si = 0; si < show; si++) {
                if (si) src_str += ", ";
                src_str += t->src[si]->name.substr(0, 16);
            }
            if ((int)t->src.size() > 3) src_str += "...";

            std::cout << "  " << pad_left(std::to_string(i++), 3)
                      << "  " << pad_right(t->name, 30)
                      << "  " << pad_right(op_name(t->op), 12)
                      << "  " << pad_right(shape_str, 18)
                      << "  " << src_str << "\n";
        }
    }

    int size() const { return (int)order_.size(); }

private:
    GgmlContext& ctx_;
    std::vector<GgmlTensor*> order_;
};

/*
 * Build one LLaMA-style transformer layer in the GGML compute graph.
 *
 * Shape convention: GGML column-major — (ne0=inner_dim, ne1=outer_dim).
 *   hidden state:  (d_model, n_tokens)
 *   weight matrix: (d_in, d_out)
 *   mul_mat(A, B): A=[ne0,ne1], B=[ne0,ne2] → [ne1, ne2]   (A^T × B)
 */
static GgmlTensor* build_transformer_layer(GgmlContext& ctx,
                                            GgmlTensor* cur,
                                            int d_model,
                                            int d_ffn,
                                            int n_heads,
                                            int n_kv_heads,
                                            int layer_idx) {
    int d_head   = d_model / n_heads;
    int n_kv_dim = n_kv_heads * d_head;
    int n_tokens = cur->shape[1];   // cur = (d_model, n_tokens)
    std::string p = "L" + std::to_string(layer_idx);

    // ── Weights ─────────────────────────────────────────────────────────
    auto* W_norm1 = ctx.new_input(p+"_attn_norm", {d_model, 1});
    auto* W_q     = ctx.new_input(p+"_Wq",        {d_model, d_model});
    auto* W_k     = ctx.new_input(p+"_Wk",        {d_model, n_kv_dim});
    auto* W_v     = ctx.new_input(p+"_Wv",        {d_model, n_kv_dim});
    auto* W_o     = ctx.new_input(p+"_Wo",        {d_model, d_model});
    auto* W_norm2 = ctx.new_input(p+"_ffn_norm",  {d_model, 1});
    auto* W_gate  = ctx.new_input(p+"_Wgate",     {d_model, d_ffn});
    auto* W_up    = ctx.new_input(p+"_Wup",       {d_model, d_ffn});
    auto* W_down  = ctx.new_input(p+"_Wdown",     {d_ffn, d_model});
    auto* pos     = ctx.new_input(p+"_pos",        {n_tokens, 1});

    auto* residual = cur;

    // ── Attention block ──────────────────────────────────────────────────
    // cur = (d_model, n_tokens)
    cur = ctx.rms_norm(cur,  p+"_rms1");          // (d_model, n_tokens)
    cur = ctx.mul(cur, W_norm1, p+"_scale1");      // broadcast (d_model,1)→(d_model,n_tokens)

    // Q,K,V projections: mul_mat(W, cur)
    //   W=[d_model, d_out], cur=[d_model, n_tokens]
    //   → ne1_W=d_out, ne2_cur=n_tokens → (d_out, n_tokens)
    auto* Q = ctx.mul_mat(W_q, cur, p+"_Q");       // (d_model, n_tokens)
    auto* K = ctx.mul_mat(W_k, cur, p+"_K");       // (n_kv_dim, n_tokens)
    auto* V = ctx.mul_mat(W_v, cur, p+"_V");       // (n_kv_dim, n_tokens)

    Q = ctx.rope(Q, pos, p+"_ropeQ");              // (d_model, n_tokens)
    K = ctx.rope(K, pos, p+"_ropeK");              // (n_kv_dim, n_tokens)

    // Attention: FlashAttention node (fused QK^T + softmax + AV)
    auto* flash = ctx.alloc(p+"_flash_attn",
                             {d_model, n_tokens},
                             GgmlOp::MUL_MAT, {Q, K, V});

    cur = ctx.mul_mat(W_o, flash, p+"_O_proj");    // (d_model, n_tokens)
    cur = ctx.add(cur, residual, p+"_attn_res");   // (d_model, n_tokens)
    residual = cur;

    // ── FFN (SwiGLU) ────────────────────────────────────────────────────
    cur  = ctx.rms_norm(cur,   p+"_rms2");         // (d_model, n_tokens)
    cur  = ctx.mul(cur, W_norm2, p+"_scale2");     // broadcast

    // gate, up: W=[d_model, d_ffn], cur=[d_model, n_tokens]
    // → mul_mat output ne1=d_ffn, ne2=n_tokens → (d_ffn, n_tokens)
    auto* gate = ctx.mul_mat(W_gate, cur, p+"_gate");  // (d_ffn, n_tokens)
    auto* up   = ctx.mul_mat(W_up,   cur, p+"_up");    // (d_ffn, n_tokens)
    gate = ctx.silu(gate, p+"_silu");                   // (d_ffn, n_tokens)
    gate = ctx.mul(gate, up, p+"_swiglu");              // (d_ffn, n_tokens)

    // down: W=[d_ffn, d_model], gate=[d_ffn, n_tokens]
    // → mul_mat output ne1=d_model, ne2=n_tokens → (d_model, n_tokens)
    cur  = ctx.mul_mat(W_down, gate, p+"_down");   // (d_model, n_tokens)
    cur  = ctx.add(cur, residual, p+"_ffn_res");   // (d_model, n_tokens)

    return cur;  // (d_model, n_tokens)
}

static void section5_ggml_dag() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "§5  GGML-Style DAG: Builder + Topological Executor\n";
    std::cout << std::string(70, '=') << "\n";

    // Tiny dimensions for readable output
    int d_model = 16, d_ffn = 32, n_heads = 4, n_kv_heads = 2, n_tokens = 2;

    GgmlContext ctx("demo_ctx", /*arena_mb=*/4.0);

    // hidden = (d_model, n_tokens) in GGML col-major convention
    auto* hidden = ctx.new_input("hidden", {d_model, n_tokens});

    auto t0 = std::chrono::high_resolution_clock::now();
    auto* out = build_transformer_layer(ctx, hidden,
                                         d_model, d_ffn,
                                         n_heads, n_kv_heads, 0);
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_us = std::chrono::duration<double, std::micro>(t1-t0).count();

    std::cout << "\n  Graph built in " << fmt_f(build_us, 1) << " µs\n";
    std::cout << "  Arena used:   " << fmt_f(ctx.used_mb(), 3) << " MB"
              << " / " << fmt_f(4.0, 1) << " MB\n";
    std::cout << "  Total nodes:  " << ctx.n_tensors() << "\n";

    GgmlGraph gf(ctx);
    t0 = std::chrono::high_resolution_clock::now();
    gf.build_from(out);
    t1 = std::chrono::high_resolution_clock::now();
    double sort_us = std::chrono::duration<double, std::micro>(t1-t0).count();

    std::cout << "  Topo sort:    " << fmt_f(sort_us, 1) << " µs\n";
    std::cout << "  Exec nodes:   " << gf.size() << "\n";

    gf.print_graph();

    t0 = std::chrono::high_resolution_clock::now();
    gf.compute(GgmlBackend::CPU);
    t1 = std::chrono::high_resolution_clock::now();
    double exec_us = std::chrono::duration<double, std::micro>(t1-t0).count();
    std::cout << "\n  Simulated compute: " << fmt_f(exec_us, 1)
              << " µs  (real dispatch: AVX2 for CPU, cuBLAS for CUDA)\n";

    // Arena exhaustion trap
    std::cout << "\n  [COMMON TRAP] Arena exhaustion demo:\n";
    try {
        GgmlContext tiny("tiny", 0.001);  // 1 KB — way too small
        auto* x = tiny.new_input("x", {d_model, n_tokens});
        (void)build_transformer_layer(tiny, x, d_model, d_ffn,
                                       n_heads, n_kv_heads, 0);
    } catch (const std::runtime_error& e) {
        std::cout << "    Caught: " << e.what() << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Block-Wise Dequantization: Q4 × F32 Dot Product
// ─────────────────────────────────────────────────────────────────────────────

// Simple LCG for deterministic "random" weights (no <random> dependency)
struct LcgRng {
    uint64_t state;
    explicit LcgRng(uint64_t seed = 42) : state(seed) {}
    float next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = (float)(state >> 33) / (float)(1u << 31);
        return u * 2.0f - 1.0f;   // uniform [-1, 1]
    }
};

struct Q4Row {
    std::vector<uint8_t> q4;    // 4-bit values stored as 0..15
    std::vector<float>   scales; // per-block FP32 scale
};

Q4Row quantize_q4(const std::vector<float>& w, int block_size = 32) {
    int n = (int)w.size();
    assert(n % block_size == 0);
    int n_blocks = n / block_size;

    Q4Row row;
    row.q4.resize(n);
    row.scales.resize(n_blocks);

    for (int b = 0; b < n_blocks; b++) {
        const float* blk = &w[b * block_size];
        float max_abs = 0.0f;
        for (int i = 0; i < block_size; i++)
            max_abs = std::max(max_abs, std::abs(blk[i]));
        float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
        row.scales[b] = scale;
        for (int i = 0; i < block_size; i++) {
            int q = (int)std::round(blk[i] / scale);
            q = std::max(-8, std::min(7, q));
            row.q4[b * block_size + i] = (uint8_t)(q + 8);  // unsigned [0..15]
        }
    }
    return row;
}

// Block-wise dequant dot: never materialises the full weight row
float dequant_dot_q4_f32(const Q4Row& row,
                           const std::vector<float>& acts,
                           int block_size = 32) {
    int n_blocks = (int)row.scales.size();
    float dot = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        float scale = row.scales[b];
        for (int i = 0; i < block_size; i++) {
            float w = (float)((int)row.q4[b*block_size+i] - 8) * scale;
            dot += w * acts[b*block_size+i];
        }
    }
    return dot;
}

// Reference: dequantise first, then dot
float dequant_full_then_dot(const Q4Row& row,
                             const std::vector<float>& acts,
                             int block_size = 32) {
    int n = (int)acts.size();
    int n_blocks = (int)row.scales.size();
    std::vector<float> wf(n);
    for (int b = 0; b < n_blocks; b++) {
        float sc = row.scales[b];
        for (int i = 0; i < block_size; i++)
            wf[b*block_size+i] = (float)((int)row.q4[b*block_size+i] - 8) * sc;
    }
    float dot = 0.0f;
    for (int i = 0; i < n; i++) dot += wf[i] * acts[i];
    return dot;
}

static void section6_blockwise_dequant() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "§6  Block-Wise Dequantization: Q4 × F32 Dot Product\n";
    std::cout << std::string(70, '=') << "\n";

    const int block_size = 32;
    const int n_blocks   = 8;
    const int n          = block_size * n_blocks;   // 256

    LcgRng rng(42);
    std::vector<float> weights(n), acts(n);
    for (auto& w : weights) w = rng.next();
    for (auto& a : acts)    a = rng.next();

    // FP32 reference dot
    float dot_f32 = 0.0f;
    for (int i = 0; i < n; i++) dot_f32 += weights[i] * acts[i];

    // Quantise to Q4
    Q4Row row = quantize_q4(weights, block_size);

    // Block-wise path (GGML)
    float dot_q4  = dequant_dot_q4_f32(row, acts, block_size);
    // Full dequant then dot (reference)
    float dot_ref = dequant_full_then_dot(row, acts, block_size);

    std::cout << "\n  n=" << n << " weights, block_size=" << block_size
              << ", n_blocks=" << n_blocks << "\n\n";
    std::cout << "  Dot product (FP32 weights, ideal):     "
              << fmt_f(dot_f32, 6) << "\n";
    std::cout << "  Dot product (via full dequant then ×): "
              << fmt_f(dot_ref, 6) << "\n";
    std::cout << "  Dot product (block-wise, GGML path):   "
              << fmt_f(dot_q4, 6) << "\n";

    float err_bw  = std::abs(dot_q4 - dot_ref);
    float err_q4  = std::abs(dot_q4 - dot_f32);
    float pct     = dot_f32 != 0.0f ? 100.0f * err_q4 / std::abs(dot_f32) : 0.0f;

    std::cout << "\n  Error (block-wise vs full-dequant):  "
              << fmt_f(err_bw, 2) << "  [should be 0.00]\n";
    std::cout << "  Quantization error (Q4 vs FP32):     "
              << fmt_f(err_q4, 4) << "  (" << fmt_f(pct, 2) << "%)\n";

    // Memory savings
    int f32_bytes  = n * 4;
    int q4_bytes   = n / 2;
    int scale_bytes= n_blocks * 4;
    int total_q4   = q4_bytes + scale_bytes;
    std::cout << "\n  Memory:\n";
    std::cout << "    FP32 row (" << n << " weights):  " << f32_bytes << " bytes\n";
    std::cout << "    Q4 data:                " << q4_bytes << " bytes\n";
    std::cout << "    Scales (" << n_blocks << " blocks):      " << scale_bytes << " bytes\n";
    std::cout << "    Q4 total:               " << total_q4 << " bytes  ("
              << fmt_f(100.0*total_q4/f32_bytes,1) << "% of FP32)\n";

    assert(err_bw < 1e-5f && "Block-wise and full-dequant paths must agree");
    std::cout << "\n  Assertion: block-wise == full-dequant  PASSED\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  End-to-End Decode Step Timing Comparison
// ─────────────────────────────────────────────────────────────────────────────

static void section7_end_to_end() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "§7  End-to-End Decode Step Timing Comparison\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << R"(
  Comparing estimated end-to-end decode step latency:

    vLLM (CUDA graph)   — LLaMA 3 8B on A100 80GB, HBM 2000 GB/s
    llama.cpp (CPU)     — LLaMA 3 8B Q4_K_M on M2 Max (400 GB/s)
    llama.cpp (CUDA)    — LLaMA 3 8B Q4_K_M on RTX 4090 (1008 GB/s)

  Numbers are analytical roofline models.
)";

    // vLLM A100
    const auto& cfg = LLAMA3_8B;
    std::cout << "  ┌──────────────────────────────────────────────────┐\n";
    std::cout << "  │  vLLM A100 80GB  (BF16, CUDA graph)             │\n";
    std::cout << "  └──────────────────────────────────────────────────┘\n";
    for (int B : {1, 8, 32, 64}) {
        auto bw  = memory_bandwidth_bound(cfg, B, 2000.0);
        auto cg  = decode_step_timing(B, bw.theoretical_time_ms, true);
        auto eg  = decode_step_timing(B, bw.theoretical_time_ms, false);
        double spd = eg.total_ms() / cg.total_ms();
        std::cout << "    B=" << pad_left(std::to_string(B),3)
                  << ": CUDA graph " << fmt_f(cg.total_ms()) << " ms"
                  << "  eager " << fmt_f(eg.total_ms()) << " ms"
                  << "  speedup " << fmt_f(spd) << "×\n";
    }

    // llama.cpp configurations: Q4_K_M (~4.5 bpw)
    ModelConfig cfg_q4 = LLAMA3_8B;
    cfg_q4.name = "LLaMA-3-8B-Q4_K_M";
    cfg_q4.bpw  = 4.5;

    auto print_llamacpp = [&](const std::string& hw, double hbm_bw_gbs) {
        std::cout << "\n  ┌──────────────────────────────────────────────────┐\n";
        std::cout << "  │  llama.cpp " << pad_right(hw, 38) << "│\n";
        std::cout << "  └──────────────────────────────────────────────────┘\n";
        auto bw = memory_bandwidth_bound(cfg_q4, 1, hbm_bw_gbs);
        double ggml_overhead_ms = 0.3;
        double total_ms = ggml_overhead_ms + bw.theoretical_time_ms;
        std::cout << "    B=1:  graph build: " << fmt_f(ggml_overhead_ms,1) << " ms"
                  << "  compute: " << fmt_f(bw.theoretical_time_ms) << " ms"
                  << "  total: "   << fmt_f(total_ms) << " ms\n";
        std::cout << "    Throughput: " << fmt_f(1000.0/total_ms, 1) << " tok/s\n";
    };

    print_llamacpp("Q4_K_M on RTX 4090 (1008 GB/s)", 1008.0);
    print_llamacpp("Q4_K_M on M2 Max CPU (400 GB/s)", 400.0);

    // Summary table
    std::cout << "\n  ┌────────────────────────────────────────────────────────────┐\n";
    std::cout << "  │  Summary: LLaMA 3 8B, B=1, roofline estimate              │\n";
    std::cout << "  └────────────────────────────────────────────────────────────┘\n";
    std::cout << "  " << pad_right("Setup",35) << "  "
              << pad_left("Latency(ms)",12) << "  "
              << pad_left("Tok/s",8) << "\n";
    std::cout << "  " << std::string(35,'-') << "  " << std::string(12,'-')
              << "  " << std::string(8,'-') << "\n";

    auto vllm_cg = decode_step_timing(1, memory_bandwidth_bound(cfg,1,2000).theoretical_time_ms, true);
    auto vllm_eg = decode_step_timing(1, memory_bandwidth_bound(cfg,1,2000).theoretical_time_ms, false);
    auto r4090   = memory_bandwidth_bound(cfg_q4,1,1008.0);
    auto rm2     = memory_bandwidth_bound(cfg_q4,1,400.0);
    double t4090 = 0.3 + r4090.theoretical_time_ms;
    double tm2   = 0.3 + rm2.theoretical_time_ms;

    std::vector<std::pair<std::string,double>> rows = {
        {"vLLM A100 80GB (BF16, CUDA graph)", vllm_cg.total_ms()},
        {"vLLM A100 80GB (BF16, eager)",      vllm_eg.total_ms()},
        {"llama.cpp RTX 4090 (Q4_K_M)",       t4090},
        {"llama.cpp M2 Max CPU (Q4_K_M)",     tm2},
    };
    for (auto& [lbl, ms] : rows) {
        std::cout << "  " << pad_right(lbl,35)
                  << "  " << pad_left(fmt_f(ms),12)
                  << "  " << pad_left(fmt_f(1000.0/ms,1),8) << "\n";
    }

    std::cout << "\n  Note: theoretical roofline only. Actual results vary due to\n";
    std::cout << "  kernel efficiency, cache effects, and DRAM latency.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    section1_flop_accounting();
    section2_cuda_graph_vs_eager();
    section3_allreduce_cost();
    section4_per_layer_timing();
    section5_ggml_dag();
    section6_blockwise_dequant();
    section7_end_to_end();

    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "All sections complete.\n";
    std::cout << std::string(70,'=') << "\n";
    return 0;
}

```

