# Chapter 14: The Eight vLLM Knobs — Companion Code

## Python — `knobs_demo.py`

```python
"""
Chapter 14 Companion Code: The Eight vLLM Knobs + llama.cpp Equivalents
========================================================================
Demonstrates:
  1. Memory budget calculator — how HBM is carved up
  2. OOM predictor — will your config survive?
  3. KV pool sizing — blocks available for concurrent requests
  4. Throughput estimator — tokens/s under given knob settings
  5. TTFT estimator — chunked prefill first-token latency
  6. Parameter interaction matrix — visualise dependency graph
  7. Misconfiguration detector — OOM Triangle, Prefill Starvation, Context Cliff
  8. YAML deployment config generator — chat / RAG / batch templates

Run:
    python3 knobs_demo.py

No external dependencies beyond the Python standard library.
"""

import math
import textwrap
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUSpec:
    name: str
    hbm_gb: float          # total HBM in gigabytes
    bw_tb_s: float         # memory bandwidth TB/s
    tflops_bf16: float     # peak BF16 TFLOPS


@dataclass
class ModelSpec:
    name: str
    params_b: float        # parameter count in billions
    layers: int
    heads: int
    kv_heads: int          # GQA kv heads (= heads for MHA)
    head_dim: int
    dtype_bytes: int = 2   # 2 for BF16/FP16, 4 for FP32


@dataclass
class VLLMConfig:
    max_num_seqs: int               = 256
    max_num_batched_tokens: int     = 8192
    max_model_len: int              = 8192
    block_size: int                 = 16
    gpu_memory_utilization: float   = 0.90
    enable_chunked_prefill: bool    = True
    enable_prefix_caching: bool     = True
    tensor_parallel_size: int       = 1


# ──────────────────────────────────────────────────────────────────────────────
# 2. GPU & MODEL CATALOGUE
# ──────────────────────────────────────────────────────────────────────────────

GPUS = {
    "A100-40": GPUSpec("A100-40",  40.0, 1.56, 312),
    "A100-80": GPUSpec("A100-80",  80.0, 2.00, 312),
    "H100":    GPUSpec("H100",     80.0, 3.35, 989),
    "H200":    GPUSpec("H200",    141.0, 4.80, 989),
    "4090":    GPUSpec("4090",     24.0, 1.01, 165),
    "3090":    GPUSpec("3090",     24.0, 0.94,  71),
}

MODELS = {
    "Llama-3-8B":  ModelSpec("Llama-3-8B",   8,  32,  32,  8,  128),
    "Llama-3-70B": ModelSpec("Llama-3-70B",  70,  80,  64,  8,  128),
    "Llama-3.1-405B": ModelSpec("Llama-3.1-405B", 405, 126, 128, 8, 128),
    "Mistral-7B":  ModelSpec("Mistral-7B",   7.3, 32, 32,  8,  128),
    "Qwen2-72B":   ModelSpec("Qwen2-72B",   72.0, 80, 64,  8,  128),
}


# ──────────────────────────────────────────────────────────────────────────────
# 3. MEMORY BUDGET CALCULATOR
# ──────────────────────────────────────────────────────────────────────────────

def calc_model_weights_gb(model: ModelSpec, tp: int = 1) -> float:
    """Approximate model weight footprint after tensor parallelism."""
    total_params = model.params_b * 1e9
    bytes_per_param = model.dtype_bytes
    gb = (total_params * bytes_per_param) / (1024**3)
    return gb / tp   # TP shards weights across GPUs


def calc_kv_bytes_per_token(model: ModelSpec, tp: int = 1) -> float:
    """
    KV cache bytes per token (both K and V, all layers).
    KV heads are not sharded further by TP in all implementations,
    but vLLM does shard them by TP.
    """
    kv_heads_per_gpu = max(1, model.kv_heads // tp)
    # 2 tensors (K, V) × layers × kv_heads_per_gpu × head_dim × dtype_bytes
    return 2 * model.layers * kv_heads_per_gpu * model.head_dim * model.dtype_bytes


def memory_budget(gpu: GPUSpec, model: ModelSpec, cfg: VLLMConfig) -> dict:
    """
    Break the GPU HBM into its three components and return a budget dict.
    """
    total_hbm_gb   = gpu.hbm_gb * cfg.tensor_parallel_size
    usable_hbm_gb  = total_hbm_gb * cfg.gpu_memory_utilization

    weights_gb     = calc_model_weights_gb(model, cfg.tensor_parallel_size)
    overhead_gb    = 1.5   # CUDA context, activations, miscellaneous

    kv_pool_gb     = usable_hbm_gb - weights_gb - overhead_gb

    kv_bytes_tok   = calc_kv_bytes_per_token(model, cfg.tensor_parallel_size)
    kv_bytes_blk   = kv_bytes_tok * cfg.block_size
    kv_blocks      = int((kv_pool_gb * 1024**3) / kv_bytes_blk)
    max_tokens_kv  = kv_blocks * cfg.block_size

    return {
        "total_hbm_gb":   total_hbm_gb,
        "usable_hbm_gb":  usable_hbm_gb,
        "weights_gb":     weights_gb,
        "overhead_gb":    overhead_gb,
        "kv_pool_gb":     max(0.0, kv_pool_gb),
        "kv_bytes_tok":   kv_bytes_tok,
        "kv_bytes_blk":   kv_bytes_blk,
        "kv_blocks":      max(0, kv_blocks),
        "max_tokens_kv":  max(0, max_tokens_kv),
    }


def print_memory_budget(gpu: GPUSpec, model: ModelSpec, cfg: VLLMConfig):
    b = memory_budget(gpu, model, cfg)
    print(f"\n{'='*62}")
    print(f"  Memory Budget  |  {gpu.name} × {cfg.tensor_parallel_size}  |  {model.name}")
    print(f"{'='*62}")
    bar_total = 50
    def bar(frac): return '█' * int(frac * bar_total) + '░' * (bar_total - int(frac * bar_total))

    w_frac = b['weights_gb'] / b['total_hbm_gb']
    o_frac = b['overhead_gb'] / b['total_hbm_gb']
    k_frac = b['kv_pool_gb'] / b['total_hbm_gb']

    print(f"  Weights  {b['weights_gb']:6.1f} GB  [{bar(w_frac)}]  {w_frac*100:.1f}%")
    print(f"  Overhead {b['overhead_gb']:6.1f} GB  [{bar(o_frac)}]  {o_frac*100:.1f}%")
    print(f"  KV Pool  {b['kv_pool_gb']:6.1f} GB  [{bar(k_frac)}]  {k_frac*100:.1f}%")
    print(f"  {'─'*58}")
    print(f"  KV block size  : {cfg.block_size} tokens")
    print(f"  Bytes / token  : {b['kv_bytes_tok']:,} B")
    print(f"  Bytes / block  : {b['kv_bytes_blk']:,} B")
    print(f"  KV blocks      : {b['kv_blocks']:,}")
    print(f"  Max KV tokens  : {b['max_tokens_kv']:,}")
    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. OOM PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────

def oom_predict(gpu: GPUSpec, model: ModelSpec, cfg: VLLMConfig) -> list[str]:
    """
    Return a list of OOM / misconfiguration warnings.
    Empty list = config looks safe.
    """
    b   = memory_budget(gpu, model, cfg)
    warnings = []

    # 4a. Model doesn't fit
    if b['weights_gb'] > b['usable_hbm_gb']:
        warnings.append(
            f"[CRITICAL] Model weights ({b['weights_gb']:.1f} GB) exceed usable HBM "
            f"({b['usable_hbm_gb']:.1f} GB). OOM guaranteed."
        )

    # 4b. KV pool is negative
    if b['kv_pool_gb'] < 0:
        warnings.append(
            f"[CRITICAL] KV pool is negative ({b['kv_pool_gb']:.1f} GB). "
            "Increase TP or reduce gpu_memory_utilization threshold."
        )
        return warnings   # nothing more to check

    # 4c. max_model_len × max_num_seqs > max_tokens_kv  (OOM Triangle)
    worst_case_tokens = cfg.max_model_len * cfg.max_num_seqs
    if worst_case_tokens > b['max_tokens_kv']:
        warnings.append(
            f"[OOM TRIANGLE] max_model_len × max_num_seqs = "
            f"{cfg.max_model_len:,} × {cfg.max_num_seqs} = {worst_case_tokens:,} tokens "
            f"> KV capacity {b['max_tokens_kv']:,}. "
            "Reduce max_num_seqs or max_model_len."
        )

    # 4d. max_num_batched_tokens << max_model_len with chunked prefill off
    #     → Prefill Starvation Loop
    if (not cfg.enable_chunked_prefill and
            cfg.max_num_batched_tokens < cfg.max_model_len):
        warnings.append(
            f"[PREFILL STARVATION] max_num_batched_tokens ({cfg.max_num_batched_tokens}) "
            f"< max_model_len ({cfg.max_model_len}) and chunked_prefill is OFF. "
            "Long prompts will monopolise the scheduler. Enable chunked_prefill."
        )

    # 4e. Context Cliff — max_model_len > physical capacity
    tokens_per_seq = b['max_tokens_kv'] // max(1, cfg.max_num_seqs)
    if cfg.max_model_len > tokens_per_seq:
        warnings.append(
            f"[CONTEXT CLIFF] max_model_len={cfg.max_model_len:,} but KV pool only "
            f"supports {tokens_per_seq:,} tokens/seq at max_num_seqs={cfg.max_num_seqs}. "
            "Requests will be rejected at runtime once the pool is full."
        )

    # 4f. Block size not a power of two
    if cfg.block_size & (cfg.block_size - 1) != 0:
        warnings.append(
            f"[CONFIG] block_size={cfg.block_size} is not a power of two. "
            "Use 8, 16, or 32 for optimal CUDA kernel performance."
        )

    # 4g. Prefix caching enabled but block_size too small
    if cfg.enable_prefix_caching and cfg.block_size < 16:
        warnings.append(
            f"[PERF] enable_prefix_caching=True but block_size={cfg.block_size}. "
            "Small blocks increase hash-table overhead. Consider block_size=16."
        )

    return warnings


def print_oom_report(gpu: GPUSpec, model: ModelSpec, cfg: VLLMConfig):
    warnings = oom_predict(gpu, model, cfg)
    print(f"\n{'='*62}")
    print(f"  OOM / Misconfiguration Report  |  {model.name}  |  {gpu.name}")
    print(f"{'='*62}")
    if not warnings:
        print("  ✓  No issues detected. Configuration looks safe.")
    else:
        for w in warnings:
            print(f"  ⚠  {w}")
    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. THROUGHPUT ESTIMATOR
# ──────────────────────────────────────────────────────────────────────────────

def estimate_throughput(gpu: GPUSpec, model: ModelSpec, cfg: VLLMConfig,
                        avg_prompt_tokens: int = 512,
                        avg_completion_tokens: int = 256) -> dict:
    """
    Rough tokens/s estimate based on arithmetic intensity model.

    Prefill is compute-bound → throughput = peak_flops / flops_per_token.
    Decode is bandwidth-bound → throughput = peak_bandwidth / bytes_per_token.
    We report both and note which regime dominates.
    """
    # Effective GPU count
    n_gpu = cfg.tensor_parallel_size

    # Prefill: arithmetic intensity ≈ 2 × params / token (GEMM)
    flops_per_prefill_token = 2 * model.params_b * 1e9 / n_gpu   # distributed weight
    prefill_throughput = (gpu.tflops_bf16 * 1e12) / flops_per_prefill_token  # tok/s per GPU

    # Decode: memory bandwidth bound — must load all weights per step
    bytes_per_weight_per_step = (model.params_b * 1e9 * model.dtype_bytes) / n_gpu
    decode_throughput = (gpu.bw_tb_s * 1e12) / bytes_per_weight_per_step   # tok/s per GPU

    # Effective batch throughput (max_num_seqs requests in flight)
    # Prefill: limited by max_num_batched_tokens
    eff_prefill_seqs = min(cfg.max_num_seqs,
                           cfg.max_num_batched_tokens // max(1, avg_prompt_tokens))
    # Decode: all seqs in flight
    eff_decode_seqs  = cfg.max_num_seqs

    prefill_tok_s = prefill_throughput * min(eff_prefill_seqs, cfg.max_num_seqs)
    decode_tok_s  = decode_throughput  * eff_decode_seqs

    return {
        "prefill_throughput_per_gpu_tok_s": prefill_throughput,
        "decode_throughput_per_gpu_tok_s":  decode_throughput,
        "eff_prefill_seqs": eff_prefill_seqs,
        "eff_decode_seqs":  eff_decode_seqs,
        "prefill_tok_s":    prefill_tok_s,
        "decode_tok_s":     decode_tok_s,
    }


def print_throughput(gpu: GPUSpec, model: ModelSpec, cfg: VLLMConfig):
    t = estimate_throughput(gpu, model, cfg)
    print(f"\n{'='*62}")
    print(f"  Throughput Estimate  |  {model.name}  |  {gpu.name}")
    print(f"{'='*62}")
    print(f"  Prefill (compute-bound)")
    print(f"    per GPU              : {t['prefill_throughput_per_gpu_tok_s']:>10,.0f} tok/s")
    print(f"    effective seqs       : {t['eff_prefill_seqs']}")
    print(f"    total prefill        : {t['prefill_tok_s']:>10,.0f} tok/s")
    print(f"  Decode (bandwidth-bound)")
    print(f"    per GPU              : {t['decode_throughput_per_gpu_tok_s']:>10,.0f} tok/s")
    print(f"    effective seqs       : {t['eff_decode_seqs']}")
    print(f"    total decode         : {t['decode_tok_s']:>10,.0f} tok/s")
    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. TTFT ESTIMATOR (chunked prefill)
# ──────────────────────────────────────────────────────────────────────────────

def estimate_ttft(gpu: GPUSpec, model: ModelSpec, cfg: VLLMConfig,
                  prompt_tokens: int) -> dict:
    """
    How long until the first token if chunked prefill is (a) OFF, (b) ON?

    Without chunked prefill:
      TTFT ≈ full prompt prefill latency = prompt_tokens × (2 × params / peak_flops)

    With chunked prefill (decode-first budget):
      The prompt is sliced into ceil(prompt / chunk) chunks.
      Decoding proceeds between chunks, so TTFT ≈ one chunk latency
      (first chunk is processed, then first decode token is emitted).
    """
    n_gpu       = cfg.tensor_parallel_size
    peak_flops  = gpu.tflops_bf16 * 1e12
    flops_per_t = 2 * model.params_b * 1e9 / n_gpu

    # Full-batch prefill (no chunking)
    ttft_no_chunk_s = (prompt_tokens * flops_per_t) / peak_flops

    # Chunked prefill — first chunk is max_num_batched_tokens minus decode slots
    # vLLM reserves decode slots first; assume 64 decode tokens reserved
    decode_reserved   = min(64, cfg.max_num_batched_tokens // 4)
    chunk_size        = cfg.max_num_batched_tokens - decode_reserved
    chunk_size        = max(1, chunk_size)
    n_chunks          = math.ceil(prompt_tokens / chunk_size)
    # First chunk is min(chunk_size, prompt_tokens) — short prompts fit in one step
    first_chunk       = min(chunk_size, prompt_tokens)
    ttft_chunked_s    = (first_chunk * flops_per_t) / peak_flops

    return {
        "prompt_tokens":     prompt_tokens,
        "chunk_size":        chunk_size,
        "first_chunk":       first_chunk,
        "n_chunks":          n_chunks,
        "ttft_no_chunk_ms":  ttft_no_chunk_s  * 1000,
        "ttft_chunked_ms":   ttft_chunked_s   * 1000,
        "speedup":           ttft_no_chunk_s  / max(1e-12, ttft_chunked_s),
    }


def print_ttft_table(gpu: GPUSpec, model: ModelSpec, cfg: VLLMConfig):
    print(f"\n{'='*70}")
    print(f"  TTFT vs Prompt Length  |  {model.name}  |  {gpu.name}")
    print(f"  Chunked prefill: {'ON ' if cfg.enable_chunked_prefill else 'OFF'}")
    print(f"{'='*70}")
    print(f"  {'Prompt (tok)':>12}  {'No-Chunk (ms)':>14}  {'Chunked (ms)':>13}  {'Speedup':>8}")
    print(f"  {'─'*12}  {'─'*14}  {'─'*13}  {'─'*8}")
    for n in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        if n > cfg.max_model_len:
            break
        r = estimate_ttft(gpu, model, cfg, n)
        print(f"  {n:>12,}  {r['ttft_no_chunk_ms']:>14.1f}  {r['ttft_chunked_ms']:>13.1f}  {r['speedup']:>7.1f}×")
    print(f"{'='*70}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. PARAMETER INTERACTION MATRIX
# ──────────────────────────────────────────────────────────────────────────────

PARAMS = [
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "block_size",
    "gpu_memory_utilization",
    "chunked_prefill",
    "prefix_caching",
    "tensor_parallel_size",
]

# (row_param, col_param) → short interaction note
INTERACTIONS: dict[tuple[str, str], str] = {
    ("max_num_seqs",           "max_num_batched_tokens"): "seqs × avg_len ≤ budget",
    ("max_num_seqs",           "max_model_len"):           "OOM Triangle product",
    ("max_num_seqs",           "gpu_memory_utilization"):  "more seqs → need more KV",
    ("max_num_seqs",           "tensor_parallel_size"):    "more TP → more KV headroom",
    ("max_num_batched_tokens", "chunked_prefill"):         "budget = chunk ceiling",
    ("max_num_batched_tokens", "max_model_len"):           "no-chunk: must ≥ max_model_len",
    ("max_model_len",          "gpu_memory_utilization"):  "longer ctx → more KV bytes",
    ("max_model_len",          "block_size"):              "fragmentation ≤ block_size-1",
    ("max_model_len",          "tensor_parallel_size"):    "TP shrinks KV heads/GPU",
    ("block_size",             "prefix_caching"):          "larger block → better hash hit",
    ("block_size",             "gpu_memory_utilization"):  "fragment waste ≤ block_size",
    ("gpu_memory_utilization", "tensor_parallel_size"):    "each GPU gets same fraction",
    ("chunked_prefill",        "prefix_caching"):          "complement: cache+chunk = best",
    ("prefix_caching",         "tensor_parallel_size"):    "radix tree is shared per GPU",
}


def print_interaction_matrix():
    n = len(PARAMS)
    short = [p[:10] for p in PARAMS]   # truncate for column headers
    col_w = 12

    print(f"\n{'='*75}")
    print("  Parameter Interaction Matrix")
    print(f"{'='*75}")
    header = "  " + "".ljust(22)
    for s in short:
        header += s.ljust(col_w)
    print(header)
    print("  " + "─" * 73)

    for i, rp in enumerate(PARAMS):
        row = f"  {rp[:20].ljust(22)}"
        for j, cp in enumerate(PARAMS):
            if i == j:
                cell = "●".center(col_w)
            elif (rp, cp) in INTERACTIONS:
                cell = "▲".center(col_w)
            elif (cp, rp) in INTERACTIONS:
                cell = "▲".center(col_w)
            else:
                cell = "·".center(col_w)
            row += cell
        print(row)

    print(f"\n  ▲ = direct interaction   ● = self   · = independent")
    print(f"\n  Key interactions:")
    for (r, c), note in INTERACTIONS.items():
        print(f"    {r[:22].ljust(22)} ↔ {c[:22].ljust(22)} : {note}")
    print(f"{'='*75}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. MISCONFIGURATION DETECTOR — detailed diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def detect_oom_triangle(cfg: VLLMConfig, kv_capacity: int) -> Optional[str]:
    product = cfg.max_model_len * cfg.max_num_seqs
    if product > kv_capacity:
        shortage = product - kv_capacity
        fix_seqs = kv_capacity // cfg.max_model_len
        fix_len  = kv_capacity // cfg.max_num_seqs
        return (
            f"OOM TRIANGLE DETECTED\n"
            f"  max_model_len × max_num_seqs = {cfg.max_model_len:,} × {cfg.max_num_seqs} "
            f"= {product:,}  >  KV capacity {kv_capacity:,}  (shortage: {shortage:,})\n"
            f"  Fix A: reduce max_num_seqs   → {fix_seqs}\n"
            f"  Fix B: reduce max_model_len  → {fix_len:,}\n"
            f"  Fix C: increase gpu_memory_utilization or add TP"
        )
    return None


def detect_prefill_starvation(cfg: VLLMConfig) -> Optional[str]:
    if not cfg.enable_chunked_prefill and cfg.max_num_batched_tokens < cfg.max_model_len:
        ratio = cfg.max_model_len / cfg.max_num_batched_tokens
        return (
            f"PREFILL STARVATION LOOP DETECTED\n"
            f"  chunked_prefill=OFF and max_num_batched_tokens ({cfg.max_num_batched_tokens:,}) "
            f"< max_model_len ({cfg.max_model_len:,})  [ratio={ratio:.1f}×]\n"
            f"  Effect: a single long prompt monopolises {ratio:.0f} full scheduler steps.\n"
            f"  Fix A: set enable_chunked_prefill=True\n"
            f"  Fix B: raise max_num_batched_tokens ≥ max_model_len"
        )
    return None


def detect_context_cliff(cfg: VLLMConfig, kv_capacity: int) -> Optional[str]:
    tokens_per_seq = kv_capacity // max(1, cfg.max_num_seqs)
    if cfg.max_model_len > tokens_per_seq:
        return (
            f"CONTEXT CLIFF DETECTED\n"
            f"  max_model_len={cfg.max_model_len:,} but KV pool / max_num_seqs "
            f"= {kv_capacity:,} / {cfg.max_num_seqs} = {tokens_per_seq:,} tokens/seq\n"
            f"  Effect: requests accepted up to {tokens_per_seq:,} tokens, "
            f"then hard-rejected at {cfg.max_model_len:,}.\n"
            f"  Users see abrupt failures at context boundary.\n"
            f"  Fix: lower max_model_len to {tokens_per_seq:,} or add KV capacity"
        )
    return None


def print_misconfiguration_report(gpu: GPUSpec, model: ModelSpec, cfg: VLLMConfig):
    b = memory_budget(gpu, model, cfg)
    kv_cap = b['max_tokens_kv']

    issues = [
        detect_oom_triangle(cfg, kv_cap),
        detect_prefill_starvation(cfg),
        detect_context_cliff(cfg, kv_cap),
    ]
    issues = [i for i in issues if i is not None]

    print(f"\n{'='*62}")
    print(f"  Misconfiguration Detector  |  {model.name}  |  {gpu.name}")
    print(f"{'='*62}")
    if not issues:
        print("  ✓  No classic misconfigurations detected.")
    for issue in issues:
        for line in issue.split("\n"):
            print(f"  {line}")
        print()
    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# 9. YAML CONFIG GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_yaml(workload: str, gpu: GPUSpec, model: ModelSpec,
                  cfg: VLLMConfig) -> str:
    b = memory_budget(gpu, model, cfg)

    base = textwrap.dedent(f"""\
        # vLLM deployment config — workload: {workload}
        # Generated for {model.name} on {gpu.name} × {cfg.tensor_parallel_size}
        # KV capacity: {b['max_tokens_kv']:,} tokens  ({b['kv_pool_gb']:.1f} GB pool)
        model: /path/to/{model.name.lower().replace(' ','-')}
        dtype: bfloat16
        tensor_parallel_size: {cfg.tensor_parallel_size}
        max_num_seqs: {cfg.max_num_seqs}
        max_num_batched_tokens: {cfg.max_num_batched_tokens}
        max_model_len: {cfg.max_model_len}
        block_size: {cfg.block_size}
        gpu_memory_utilization: {cfg.gpu_memory_utilization}
        enable_chunked_prefill: {"true" if cfg.enable_chunked_prefill else "false"}
        enable_prefix_caching: {"true" if cfg.enable_prefix_caching else "false"}
    """)

    notes = {
        "chat": textwrap.dedent("""\
            # CHAT tuning notes:
            #   max_num_seqs       — set to peak expected concurrent users
            #   max_model_len      — typical chat ≤ 8 K; cut to 8192 to maximize concurrency
            #   enable_prefix_caching: true   — system prompt hits ~100% after warmup
            #   enable_chunked_prefill: true  — keeps TTFT low for new arrivals
        """),
        "rag": textwrap.dedent("""\
            # RAG tuning notes:
            #   max_num_batched_tokens — raise to 32768 to handle large retrieved docs
            #   max_model_len          — set to retrieval window (16K–128K)
            #   enable_prefix_caching  — retrieval template hits on repeated queries
            #   max_num_seqs           — lower than chat; RAG prompts are large
        """),
        "batch": textwrap.dedent("""\
            # BATCH / offline tuning notes:
            #   max_num_seqs           — maximize for throughput, ignore TTFT
            #   enable_chunked_prefill — can be false; no interactive latency SLA
            #   enable_prefix_caching  — limited benefit for unique offline inputs
            #   gpu_memory_utilization — push to 0.95 (no headroom needed for bursts)
        """),
    }
    return base + "\n" + notes.get(workload, "")


def print_yaml_configs(gpu: GPUSpec, model: ModelSpec):
    b     = memory_budget(gpu, model, VLLMConfig())
    kv_c  = b['max_tokens_kv']

    # Chat: low latency, many short sessions
    chat_cfg = VLLMConfig(
        max_num_seqs=128,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        block_size=16,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        tensor_parallel_size=1,
    )

    # RAG: large context retrieval
    rag_cfg = VLLMConfig(
        max_num_seqs=32,
        max_num_batched_tokens=32768,
        max_model_len=32768,
        block_size=16,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        tensor_parallel_size=1,
    )

    # Batch: offline high-throughput
    batch_cfg = VLLMConfig(
        max_num_seqs=256,
        max_num_batched_tokens=65536,
        max_model_len=4096,
        block_size=32,
        gpu_memory_utilization=0.95,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        tensor_parallel_size=1,
    )

    for wl, cfg in [("chat", chat_cfg), ("rag", rag_cfg), ("batch", batch_cfg)]:
        print(f"\n{'─'*62}")
        print(generate_yaml(wl, gpu, model, cfg))


# ──────────────────────────────────────────────────────────────────────────────
# 10. LLAMA.CPP PARAMETER TABLE
# ──────────────────────────────────────────────────────────────────────────────

def print_llamacpp_table():
    rows = [
        ("vLLM parameter",           "llama.cpp CLI flag",     "llama_context_params field"),
        ("─"*28,                     "─"*25,                   "─"*30),
        ("max_num_seqs",             "--parallel N",           "n_parallel"),
        ("max_num_batched_tokens",   "--ubatch-size N",        "n_ubatch"),
        ("max_model_len",            "--ctx-size N",           "n_ctx"),
        ("block_size",               "(internal, not exposed)", "—"),
        ("gpu_memory_utilization",   "--n-gpu-layers + manual", "n_gpu_layers"),
        ("enable_chunked_prefill",   "(automatic via ubatch)",  "—"),
        ("enable_prefix_caching",    "--cache-prompt",          "—"),
        ("tensor_parallel_size",     "--tensor-split",          "tensor_split[]"),
    ]
    print(f"\n{'='*90}")
    print("  vLLM ↔ llama.cpp Parameter Mapping")
    print(f"{'='*90}")
    for r in rows:
        print(f"  {r[0]:<30}  {r[1]:<27}  {r[2]}")
    print(f"{'='*90}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    gpu   = GPUS["A100-80"]
    model = MODELS["Llama-3-8B"]
    cfg   = VLLMConfig(
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        block_size=16,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        tensor_parallel_size=1,
    )

    print("\n" + "█"*62)
    print("  Chapter 14 — The Eight vLLM Knobs: Companion Demo")
    print("█"*62)

    # ── Section 1: Memory Budget
    print_memory_budget(gpu, model, cfg)

    # ── Section 2: OOM Report (safe config)
    print_oom_report(gpu, model, cfg)

    # ── Section 3: Deliberate OOM Triangle
    bad_cfg = VLLMConfig(
        max_num_seqs=512,          # too many
        max_num_batched_tokens=8192,
        max_model_len=32768,       # too long → product = 16.7M tokens
        block_size=16,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=False,  # starvation!
        enable_prefix_caching=True,
        tensor_parallel_size=1,
    )
    print(f"\n>>> Deliberate bad config (max_num_seqs=512, max_model_len=32768, chunked=OFF)")
    print_oom_report(gpu, model, bad_cfg)

    # ── Section 4: TTFT table
    # Use smaller token budget (2048) so chunking effect is visible across prompt sizes
    ttft_cfg = VLLMConfig(
        max_num_seqs=256,
        max_num_batched_tokens=2048,   # tight budget → chunk = 1984 tokens
        max_model_len=32768,
        block_size=16,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        tensor_parallel_size=1,
    )
    print_ttft_table(gpu, model, ttft_cfg)

    # ── Section 5: Throughput
    print_throughput(gpu, model, cfg)

    # ── Section 6: Interaction matrix
    print_interaction_matrix()

    # ── Section 7: Misconfiguration detector
    print_misconfiguration_report(gpu, model, bad_cfg)

    # ── Section 8: llama.cpp mapping
    print_llamacpp_table()

    # ── Section 9: YAML generators
    print(f"\n{'='*62}")
    print("  Production YAML Config Templates — Llama-3-8B / A100-80")
    print(f"{'='*62}")
    print_yaml_configs(gpu, model)

    # ── Section 10: 70B multi-GPU sanity check
    print(f"\n{'='*62}")
    print("  70B on 2 × A100-80 — budget sanity check")
    print(f"{'='*62}")
    gpu70  = GPUS["A100-80"]
    m70    = MODELS["Llama-3-70B"]
    cfg70  = VLLMConfig(
        max_num_seqs=64,
        max_num_batched_tokens=16384,
        max_model_len=16384,
        block_size=16,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        tensor_parallel_size=2,
    )
    print_memory_budget(gpu70, m70, cfg70)
    print_oom_report(gpu70, m70, cfg70)


if __name__ == "__main__":
    main()

```

## C++ — `knobs_demo.cpp`

```cpp
/*
 * Chapter 14 Companion Code: The Eight vLLM Knobs + llama.cpp Equivalents
 * =========================================================================
 * Demonstrates:
 *   1. HBM memory budget decomposition (weights / overhead / KV pool)
 *   2. KV block count and per-token byte calculation
 *   3. OOM Triangle / Prefill Starvation / Context Cliff detectors
 *   4. TTFT improvement table — chunked vs unchunked prefill
 *   5. llama.cpp CLI flag → llama_context_params struct field mapping
 *   6. Arithmetic intensity: prefill (compute-bound) vs decode (bw-bound)
 *   7. YAML config template generator (chat / RAG / batch)
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -o knobs_demo knobs_demo.cpp
 * Run:
 *   ./knobs_demo
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
// 1. DATA STRUCTURES
// ─────────────────────────────────────────────────────────────────────────────

struct GPUSpec {
    std::string name;
    double hbm_gb;          // total HBM in gigabytes
    double bw_tb_s;         // memory bandwidth TB/s
    double tflops_bf16;     // peak BF16 TFLOPS
};

struct ModelSpec {
    std::string name;
    double  params_b;       // parameter count (billions)
    int     layers;
    int     heads;
    int     kv_heads;       // GQA KV heads (= heads for MHA)
    int     head_dim;
    int     dtype_bytes;    // 2 for BF16/FP16
};

struct VLLMConfig {
    int    max_num_seqs               = 256;
    int    max_num_batched_tokens     = 8192;
    int    max_model_len              = 8192;
    int    block_size                 = 16;
    double gpu_memory_utilization     = 0.90;
    bool   enable_chunked_prefill     = true;
    bool   enable_prefix_caching      = true;
    int    tensor_parallel_size       = 1;
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Repeat a single-byte ASCII character
static std::string repeat(char c, int n) {
    return std::string(static_cast<size_t>(std::max(0, n)), c);
}

// Repeat a UTF-8 string token (e.g. "█", "░", "─")
static std::string repeat_utf8(const std::string& tok, int n) {
    std::string out;
    out.reserve(tok.size() * static_cast<size_t>(std::max(0, n)));
    for (int i = 0; i < n; ++i) out += tok;
    return out;
}

static std::string fmt_int(long long v) {
    // Add thousands separators
    std::string s = std::to_string(std::abs(v));
    int i = static_cast<int>(s.size()) - 3;
    while (i > 0) { s.insert(static_cast<size_t>(i), ","); i -= 3; }
    return (v < 0 ? "-" : "") + s;
}

static std::string fmt_dbl(double v, int prec = 1) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

static std::string bar_chart(double frac, int width = 40) {
    int filled = static_cast<int>(std::clamp(frac, 0.0, 1.0) * width);
    return repeat_utf8("\xe2\x96\x88", filled)          // █
         + repeat_utf8("\xe2\x96\x91", width - filled); // ░
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. MEMORY BUDGET CALCULATOR
// ─────────────────────────────────────────────────────────────────────────────

struct MemoryBudget {
    double total_hbm_gb;
    double usable_hbm_gb;
    double weights_gb;
    double overhead_gb;
    double kv_pool_gb;
    long long kv_bytes_per_token;
    long long kv_bytes_per_block;
    long long kv_blocks;
    long long max_tokens_kv;
};

static MemoryBudget calc_memory_budget(const GPUSpec& gpu, const ModelSpec& model,
                                       const VLLMConfig& cfg) {
    MemoryBudget b{};
    b.total_hbm_gb  = gpu.hbm_gb * cfg.tensor_parallel_size;
    b.usable_hbm_gb = b.total_hbm_gb * cfg.gpu_memory_utilization;

    // Weights sharded by TP
    double total_params = model.params_b * 1e9;
    b.weights_gb = (total_params * model.dtype_bytes) / (1024.0 * 1024.0 * 1024.0)
                   / cfg.tensor_parallel_size;
    b.overhead_gb = 1.5;   // CUDA context + activations

    b.kv_pool_gb = b.usable_hbm_gb - b.weights_gb - b.overhead_gb;

    // KV bytes per token: 2 × layers × kv_heads_per_gpu × head_dim × dtype_bytes
    int kv_heads_per_gpu = std::max(1, model.kv_heads / cfg.tensor_parallel_size);
    b.kv_bytes_per_token = static_cast<long long>(
        2LL * model.layers * kv_heads_per_gpu * model.head_dim * model.dtype_bytes);
    b.kv_bytes_per_block = b.kv_bytes_per_token * cfg.block_size;

    if (b.kv_pool_gb > 0.0) {
        double pool_bytes = b.kv_pool_gb * 1024.0 * 1024.0 * 1024.0;
        b.kv_blocks       = static_cast<long long>(pool_bytes / b.kv_bytes_per_block);
        b.max_tokens_kv   = b.kv_blocks * cfg.block_size;
    } else {
        b.kv_blocks     = 0;
        b.max_tokens_kv = 0;
    }
    return b;
}

static void print_memory_budget(const GPUSpec& gpu, const ModelSpec& model,
                                const VLLMConfig& cfg) {
    auto b = calc_memory_budget(gpu, model, cfg);
    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  Memory Budget  |  " << gpu.name
              << " × " << cfg.tensor_parallel_size
              << "  |  " << model.name << "\n";
    std::cout << repeat('=', 62) << "\n";

    auto row = [&](const std::string& label, double gb) {
        double frac = gb / b.total_hbm_gb;
        std::cout << "  " << std::left << std::setw(9) << label
                  << std::right << std::setw(5) << fmt_dbl(gb) << " GB"
                  << "  [" << bar_chart(frac, 40) << "]"
                  << "  " << std::setw(5) << fmt_dbl(frac * 100.0) << "%\n";
    };
    row("Weights",  b.weights_gb);
    row("Overhead", b.overhead_gb);
    row("KV Pool",  std::max(0.0, b.kv_pool_gb));

    std::cout << "  " << repeat_utf8("\xe2\x94\x80", 58) << "\n";  // ─
    std::cout << "  KV block size  : " << cfg.block_size << " tokens\n";
    std::cout << "  Bytes / token  : " << fmt_int(b.kv_bytes_per_token) << " B\n";
    std::cout << "  Bytes / block  : " << fmt_int(b.kv_bytes_per_block) << " B\n";
    std::cout << "  KV blocks      : " << fmt_int(b.kv_blocks) << "\n";
    std::cout << "  Max KV tokens  : " << fmt_int(b.max_tokens_kv) << "\n";
    std::cout << repeat('=', 62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. OOM / MISCONFIGURATION DETECTOR
// ─────────────────────────────────────────────────────────────────────────────

struct Warning {
    std::string tag;
    std::string message;
};

static std::vector<Warning> detect_issues(const GPUSpec& gpu, const ModelSpec& model,
                                          const VLLMConfig& cfg) {
    auto b = calc_memory_budget(gpu, model, cfg);
    std::vector<Warning> warns;

    // Model doesn't fit
    if (b.weights_gb > b.usable_hbm_gb) {
        warns.push_back({"CRITICAL",
            "Model weights (" + fmt_dbl(b.weights_gb) + " GB) exceed usable HBM ("
            + fmt_dbl(b.usable_hbm_gb) + " GB). OOM guaranteed."});
        return warns;
    }

    // Negative KV pool
    if (b.kv_pool_gb < 0.0) {
        warns.push_back({"CRITICAL",
            "KV pool is negative (" + fmt_dbl(b.kv_pool_gb)
            + " GB). Increase TP or reduce gpu_memory_utilization."});
        return warns;
    }

    // OOM Triangle
    long long worst = static_cast<long long>(cfg.max_model_len) * cfg.max_num_seqs;
    if (worst > b.max_tokens_kv) {
        long long shortage   = worst - b.max_tokens_kv;
        long long fix_seqs   = b.max_tokens_kv / std::max(1, cfg.max_model_len);
        long long fix_len    = b.max_tokens_kv / std::max(1, cfg.max_num_seqs);
        warns.push_back({"OOM TRIANGLE",
            "max_model_len × max_num_seqs = "
            + fmt_int(cfg.max_model_len) + " × " + std::to_string(cfg.max_num_seqs)
            + " = " + fmt_int(worst) + " > KV capacity " + fmt_int(b.max_tokens_kv)
            + " (shortage " + fmt_int(shortage) + ").\n"
            "    Fix A: reduce max_num_seqs → " + fmt_int(fix_seqs) + "\n"
            "    Fix B: reduce max_model_len → " + fmt_int(fix_len) + "\n"
            "    Fix C: increase gpu_memory_utilization or add TP"});
    }

    // Prefill Starvation Loop
    if (!cfg.enable_chunked_prefill &&
        cfg.max_num_batched_tokens < cfg.max_model_len) {
        double ratio = static_cast<double>(cfg.max_model_len) / cfg.max_num_batched_tokens;
        warns.push_back({"PREFILL STARVATION",
            "chunked_prefill=OFF and max_num_batched_tokens ("
            + fmt_int(cfg.max_num_batched_tokens)
            + ") < max_model_len (" + fmt_int(cfg.max_model_len)
            + ") [ratio=" + fmt_dbl(ratio, 1) + "×].\n"
            "    A single long prompt monopolises " + fmt_dbl(ratio, 0)
            + " full scheduler steps.\n"
            "    Fix A: set enable_chunked_prefill=true\n"
            "    Fix B: raise max_num_batched_tokens >= max_model_len"});
    }

    // Context Cliff
    long long toks_per_seq = (cfg.max_num_seqs > 0)
                             ? b.max_tokens_kv / cfg.max_num_seqs : 0;
    if (cfg.max_model_len > toks_per_seq) {
        warns.push_back({"CONTEXT CLIFF",
            "max_model_len=" + fmt_int(cfg.max_model_len)
            + " but KV pool / max_num_seqs = "
            + fmt_int(b.max_tokens_kv) + " / " + std::to_string(cfg.max_num_seqs)
            + " = " + fmt_int(toks_per_seq) + " tokens/seq.\n"
            "    Requests accepted to " + fmt_int(toks_per_seq)
            + " tokens, then hard-rejected at " + fmt_int(cfg.max_model_len) + ".\n"
            "    Fix: lower max_model_len to " + fmt_int(toks_per_seq)
            + " or add KV capacity"});
    }

    return warns;
}

static void print_oom_report(const GPUSpec& gpu, const ModelSpec& model,
                             const VLLMConfig& cfg) {
    auto warns = detect_issues(gpu, model, cfg);
    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  OOM / Misconfiguration Report  |  "
              << model.name << "  |  " << gpu.name << "\n";
    std::cout << repeat('=', 62) << "\n";
    if (warns.empty()) {
        std::cout << "  OK  No issues detected. Configuration looks safe.\n";
    } else {
        for (auto& w : warns) {
            std::cout << "  !! [" << w.tag << "]\n";
            // Indent continuation lines
            std::istringstream ss(w.message);
            std::string line;
            while (std::getline(ss, line)) {
                std::cout << "     " << line << "\n";
            }
            std::cout << "\n";
        }
    }
    std::cout << repeat('=', 62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. TTFT ESTIMATOR
// ─────────────────────────────────────────────────────────────────────────────

static void print_ttft_table(const GPUSpec& gpu, const ModelSpec& model,
                             const VLLMConfig& cfg) {
    double peak_flops    = gpu.tflops_bf16 * 1e12;
    double flops_per_tok = 2.0 * model.params_b * 1e9 / cfg.tensor_parallel_size;

    // Chunk size: budget minus decode reservation
    int decode_reserved = std::min(64, cfg.max_num_batched_tokens / 4);
    int chunk_size      = std::max(1, cfg.max_num_batched_tokens - decode_reserved);

    std::cout << "\n" << repeat('=', 72) << "\n";
    std::cout << "  TTFT vs Prompt Length  |  "
              << model.name << "  |  " << gpu.name << "\n";
    std::cout << "  Chunked prefill: " << (cfg.enable_chunked_prefill ? "ON" : "OFF")
              << "   chunk_size=" << fmt_int(chunk_size) << " tok\n";
    std::cout << repeat('=', 72) << "\n";
    std::cout << "  " << std::right
              << std::setw(12) << "Prompt (tok)"
              << std::setw(16) << "No-Chunk (ms)"
              << std::setw(14) << "Chunked (ms)"
              << std::setw(10) << "Speedup" << "\n";
    std::cout << "  " << repeat('-', 12) << "  "
              << repeat('-', 14) << "  "
              << repeat('-', 12) << "  "
              << repeat('-', 8)  << "\n";

    for (int prompt : {256, 512, 1024, 2048, 4096, 8192, 16384, 32768}) {
        if (prompt > cfg.max_model_len) break;

        double ttft_no_chunk = (static_cast<double>(prompt) * flops_per_tok) / peak_flops;
        int first_chunk       = std::min(chunk_size, prompt);
        double ttft_chunked   = (static_cast<double>(first_chunk) * flops_per_tok) / peak_flops;
        double speedup        = ttft_no_chunk / std::max(1e-15, ttft_chunked);

        std::ostringstream row;
        row << "  "
            << std::right << std::setw(12) << fmt_int(prompt) << "  "
            << std::setw(14) << std::fixed << std::setprecision(1) << ttft_no_chunk * 1000 << "  "
            << std::setw(12) << std::fixed << std::setprecision(1) << ttft_chunked * 1000 << "  "
            << std::setw(7)  << std::fixed << std::setprecision(1) << speedup << "x";
        std::cout << row.str() << "\n";
    }
    std::cout << repeat('=', 72) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. ARITHMETIC INTENSITY ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────

static void print_arithmetic_intensity(const GPUSpec& gpu, const ModelSpec& model) {
    /*
     * Ridge point = peak_flops / peak_bandwidth (FLOP/byte).
     * Prefill AI ≈ 2 × params / (4 × layers × heads × head_dim)  — compute-bound if > ridge
     * Decode  AI ≈ 1 token per step → always well below ridge    — bandwidth-bound
     */
    double ridge = (gpu.tflops_bf16 * 1e12) / (gpu.bw_tb_s * 1e12);

    // Prefill arithmetic intensity for a batch of B tokens:
    // FLOPs = 2 × params_per_gpu × B  (GEMM dominant)
    // Bytes = params_per_gpu × dtype_bytes  (weight loads once per step if B > threshold)
    // AI_prefill ≈ 2B / dtype_bytes
    // At what batch size does prefill become compute-bound?
    // 2B / 2 > ridge  → B > ridge
    long long prefill_breakeven = static_cast<long long>(std::ceil(ridge));

    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  Arithmetic Intensity  |  " << gpu.name << "  |  " << model.name << "\n";
    std::cout << repeat('=', 62) << "\n";
    std::cout << "  Peak compute   : " << gpu.tflops_bf16 << " TFLOP/s (BF16)\n";
    std::cout << "  Peak bandwidth : " << gpu.bw_tb_s << " TB/s\n";
    std::cout << "  Ridge point    : " << std::fixed << std::setprecision(0)
              << ridge << " FLOP/byte\n\n";

    std::cout << "  PREFILL phase:\n";
    std::cout << "    AI ≈ 2 × batch_tokens / dtype_bytes\n";
    std::cout << "    Becomes compute-bound when batch_tokens > "
              << fmt_int(prefill_breakeven) << "\n";
    std::cout << "    (i.e. when AI exceeds ridge = "
              << std::fixed << std::setprecision(0) << ridge << " FLOP/byte)\n\n";

    std::cout << "  DECODE phase:\n";
    std::cout << "    Each step generates 1 token → AI = 1 / dtype_bytes = "
              << std::fixed << std::setprecision(1) << (1.0 / model.dtype_bytes)
              << " FLOP/byte\n";
    std::cout << "    Always far below ridge → memory-bandwidth-bound\n";
    std::cout << "    Throughput ceiling = BW / (params × dtype) = "
              << std::fixed << std::setprecision(0)
              << (gpu.bw_tb_s * 1e12) / (model.params_b * 1e9 * model.dtype_bytes)
              << " tok/s per GPU\n";
    std::cout << repeat('=', 62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. LLAMA.CPP PARAMETER MAPPING TABLE
// ─────────────────────────────────────────────────────────────────────────────

static void print_llamacpp_table() {
    struct Row { std::string vllm; std::string cli; std::string field; };
    std::vector<Row> rows = {
        {"max_num_seqs",           "--parallel N",            "n_parallel"},
        {"max_num_batched_tokens", "--ubatch-size N",         "n_ubatch"},
        {"max_model_len",          "--ctx-size N",            "n_ctx"},
        {"block_size",             "(internal, not exposed)", "—"},
        {"gpu_memory_utilization", "--n-gpu-layers + manual", "n_gpu_layers"},
        {"enable_chunked_prefill", "(via ubatch automatic)",  "—"},
        {"enable_prefix_caching",  "--cache-prompt",          "—"},
        {"tensor_parallel_size",   "--tensor-split R,R,...",  "tensor_split[]"},
    };

    std::cout << "\n" << repeat('=', 88) << "\n";
    std::cout << "  vLLM Parameter <-> llama.cpp Mapping\n";
    std::cout << repeat('=', 88) << "\n";
    std::cout << "  " << std::left
              << std::setw(30) << "vLLM parameter"
              << std::setw(28) << "llama.cpp CLI flag"
              << std::setw(28) << "llama_context_params field" << "\n";
    std::cout << "  " << repeat('-', 86) << "\n";
    for (auto& r : rows) {
        std::cout << "  " << std::left
                  << std::setw(30) << r.vllm
                  << std::setw(28) << r.cli
                  << std::setw(28) << r.field << "\n";
    }
    std::cout << repeat('=', 88) << "\n";

    // Annotated llama_context_params struct sketch
    std::cout << R"(
  // llama.cpp: how these fields look in the C API (abridged)
  //
  // llama_context_params params = llama_context_default_params();
  //
  // params.n_ctx        = 32768;   // max_model_len equivalent
  // params.n_batch      = 8192;    // max tokens fed to llama_decode per call
  // params.n_ubatch     = 2048;    // micro-batch for KV chunking (ubatch-size)
  // params.n_parallel   = 4;       // simultaneous sequences (max_num_seqs)
  // params.n_gpu_layers = 80;      // how many transformer layers offload to GPU
  //
  // // Tensor split: fraction of model for GPU 0 and GPU 1
  // params.tensor_split[0] = 0.5f;
  // params.tensor_split[1] = 0.5f;
  //
  // llama_context* ctx = llama_new_context_with_model(model, params);
)";
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. YAML CONFIG GENERATOR
// ─────────────────────────────────────────────────────────────────────────────

static void print_yaml_config(const std::string& workload, const GPUSpec& gpu,
                              const ModelSpec& model, const VLLMConfig& cfg) {
    auto b = calc_memory_budget(gpu, model, cfg);
    std::cout << "\n" << repeat('-', 60) << "\n";
    std::cout << "# vLLM deployment config - workload: " << workload << "\n";
    std::cout << "# Generated for " << model.name
              << " on " << gpu.name
              << " × " << cfg.tensor_parallel_size << "\n";
    std::cout << "# KV capacity: " << fmt_int(b.max_tokens_kv)
              << " tokens  (" << fmt_dbl(b.kv_pool_gb) << " GB pool)\n";
    std::cout << "model: /path/to/" << model.name << "\n";
    std::cout << "dtype: bfloat16\n";
    std::cout << "tensor_parallel_size: " << cfg.tensor_parallel_size << "\n";
    std::cout << "max_num_seqs: "           << cfg.max_num_seqs << "\n";
    std::cout << "max_num_batched_tokens: " << cfg.max_num_batched_tokens << "\n";
    std::cout << "max_model_len: "          << cfg.max_model_len << "\n";
    std::cout << "block_size: "             << cfg.block_size << "\n";
    std::cout << "gpu_memory_utilization: " << fmt_dbl(cfg.gpu_memory_utilization, 2) << "\n";
    std::cout << "enable_chunked_prefill: " << (cfg.enable_chunked_prefill ? "true" : "false") << "\n";
    std::cout << "enable_prefix_caching: "  << (cfg.enable_prefix_caching  ? "true" : "false") << "\n";

    if (workload == "chat") {
        std::cout << "# CHAT notes: prefix_caching hits ~100% after system-prompt warmup;\n"
                  << "#   max_model_len=8192 maximizes concurrency on this GPU\n";
    } else if (workload == "rag") {
        std::cout << "# RAG notes: large max_num_batched_tokens handles bulky retrieved docs;\n"
                  << "#   prefix_caching saves re-encoding repeated retrieval templates\n";
    } else if (workload == "batch") {
        std::cout << "# BATCH notes: chunked_prefill=false OK (no TTFT SLA);\n"
                  << "#   gpu_memory_utilization=0.95 safe for offline workloads\n";
    }
}

static void print_all_yaml_configs(const GPUSpec& gpu, const ModelSpec& model) {
    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  Production YAML Config Templates\n";
    std::cout << repeat('=', 62) << "\n";

    VLLMConfig chat{};
    chat.max_num_seqs = 128; chat.max_num_batched_tokens = 8192;
    chat.max_model_len = 8192; chat.block_size = 16;
    chat.gpu_memory_utilization = 0.90;
    chat.enable_chunked_prefill = true; chat.enable_prefix_caching = true;
    chat.tensor_parallel_size = 1;
    print_yaml_config("chat", gpu, model, chat);

    VLLMConfig rag{};
    rag.max_num_seqs = 32; rag.max_num_batched_tokens = 32768;
    rag.max_model_len = 32768; rag.block_size = 16;
    rag.gpu_memory_utilization = 0.90;
    rag.enable_chunked_prefill = true; rag.enable_prefix_caching = true;
    rag.tensor_parallel_size = 1;
    print_yaml_config("rag", gpu, model, rag);

    VLLMConfig batch{};
    batch.max_num_seqs = 256; batch.max_num_batched_tokens = 65536;
    batch.max_model_len = 4096; batch.block_size = 32;
    batch.gpu_memory_utilization = 0.95;
    batch.enable_chunked_prefill = false; batch.enable_prefix_caching = false;
    batch.tensor_parallel_size = 1;
    print_yaml_config("batch", gpu, model, batch);

    std::cout << "\n" << repeat('=', 62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. THROUGHPUT ESTIMATOR
// ─────────────────────────────────────────────────────────────────────────────

static void print_throughput(const GPUSpec& gpu, const ModelSpec& model,
                             const VLLMConfig& cfg) {
    double n_gpu       = cfg.tensor_parallel_size;
    double peak_flops  = gpu.tflops_bf16 * 1e12;
    double peak_bw     = gpu.bw_tb_s * 1e12;

    // Prefill: compute-bound
    double flops_per_tok    = 2.0 * model.params_b * 1e9 / n_gpu;
    double prefill_per_gpu  = peak_flops / flops_per_tok;

    // Decode: bandwidth-bound
    double bytes_per_step   = model.params_b * 1e9 * model.dtype_bytes / n_gpu;
    double decode_per_gpu   = peak_bw / bytes_per_step;

    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  Throughput Estimate  |  "
              << model.name << "  |  " << gpu.name << "\n";
    std::cout << repeat('=', 62) << "\n";
    std::cout << "  Prefill (compute-bound)\n";
    std::cout << "    per GPU    : " << std::right << std::setw(10)
              << fmt_int(static_cast<long long>(prefill_per_gpu)) << " tok/s\n";
    std::cout << "    (AI >> ridge: one big GEMM per token)\n\n";
    std::cout << "  Decode (bandwidth-bound)\n";
    std::cout << "    per GPU    : " << std::right << std::setw(10)
              << fmt_int(static_cast<long long>(decode_per_gpu)) << " tok/s\n";
    std::cout << "    (must load all weights from HBM each step)\n";
    std::cout << "    batched (" << cfg.max_num_seqs << " seqs) : "
              << std::right << std::setw(10)
              << fmt_int(static_cast<long long>(decode_per_gpu * cfg.max_num_seqs))
              << " tok/s total\n";
    std::cout << repeat('=', 62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    GPUSpec a100_80{"A100-80", 80.0, 2.00, 312.0};
    ModelSpec llama8b{"Llama-3-8B", 8.0, 32, 32, 8, 128, 2};
    ModelSpec llama70b{"Llama-3-70B", 70.0, 80, 64, 8, 128, 2};

    VLLMConfig safe_cfg{};
    safe_cfg.max_num_seqs            = 64;
    safe_cfg.max_num_batched_tokens  = 8192;
    safe_cfg.max_model_len           = 8192;
    safe_cfg.block_size              = 16;
    safe_cfg.gpu_memory_utilization  = 0.90;
    safe_cfg.enable_chunked_prefill  = true;
    safe_cfg.enable_prefix_caching   = true;
    safe_cfg.tensor_parallel_size    = 1;

    VLLMConfig bad_cfg{};
    bad_cfg.max_num_seqs            = 512;
    bad_cfg.max_num_batched_tokens  = 8192;
    bad_cfg.max_model_len           = 32768;
    bad_cfg.block_size              = 16;
    bad_cfg.gpu_memory_utilization  = 0.90;
    bad_cfg.enable_chunked_prefill  = false;   // starvation
    bad_cfg.enable_prefix_caching   = true;
    bad_cfg.tensor_parallel_size    = 1;

    // TTFT demo: tight token budget to show chunking benefit
    VLLMConfig ttft_cfg = safe_cfg;
    ttft_cfg.max_num_batched_tokens = 2048;
    ttft_cfg.max_model_len          = 32768;

    std::cout << "\n" << repeat('#', 62) << "\n";
    std::cout << "  Chapter 14 - The Eight vLLM Knobs: C++ Companion Demo\n";
    std::cout << repeat('#', 62) << "\n";

    // Section 1: Memory budget — safe config
    print_memory_budget(a100_80, llama8b, safe_cfg);

    // Section 2: OOM report — safe config
    print_oom_report(a100_80, llama8b, safe_cfg);

    // Section 3: Deliberate bad config — all three misconfigurations
    std::cout << "\n>>> Bad config (max_num_seqs=512, max_model_len=32768, chunked=OFF)\n";
    print_oom_report(a100_80, llama8b, bad_cfg);

    // Section 4: Arithmetic intensity
    print_arithmetic_intensity(a100_80, llama8b);

    // Section 5: TTFT table
    print_ttft_table(a100_80, llama8b, ttft_cfg);

    // Section 6: Throughput
    print_throughput(a100_80, llama8b, safe_cfg);

    // Section 7: llama.cpp mapping
    print_llamacpp_table();

    // Section 8: YAML templates
    print_all_yaml_configs(a100_80, llama8b);

    // Section 9: 70B on 2 × A100-80
    VLLMConfig cfg70{};
    cfg70.max_num_seqs           = 64;
    cfg70.max_num_batched_tokens = 16384;
    cfg70.max_model_len          = 16384;
    cfg70.block_size             = 16;
    cfg70.gpu_memory_utilization = 0.90;
    cfg70.enable_chunked_prefill = true;
    cfg70.enable_prefix_caching  = true;
    cfg70.tensor_parallel_size   = 2;

    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  70B on 2 × A100-80 — sanity check\n";
    std::cout << repeat('=', 62) << "\n";
    print_memory_budget(a100_80, llama70b, cfg70);
    print_oom_report(a100_80, llama70b, cfg70);
    print_throughput(a100_80, llama70b, cfg70);

    return 0;
}

```

