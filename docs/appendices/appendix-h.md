# Appendix H: Operational Decision Tree and Troubleshooting Guide

This appendix is a production reference. It is organized as decision trees, quick-lookup tables, and fill-in worksheets that a serving engineer can consult during an incident or capacity-planning session without re-reading the main chapters.

---

## H.1 Symptom → Diagnosis → Fix: The Main Decision Tree

Each section below starts with an observable symptom and walks through a structured diagnostic sequence. Metrics referenced assume a Prometheus/Grafana stack as described in Chapter 16.

---

### H.1.1 TTFT (Time to First Token) is Too High

**Target**: p95 TTFT < 500 ms for interactive workloads.

**Observable signal**: `vllm:time_to_first_token_seconds` p95 > 0.5 s.

```
Is the median prompt length > 4 K tokens?
├── YES → Enable chunked prefill: --enable-chunked-prefill
│         Set --max-num-batched-tokens to 2048–4096.
│         Monitor: prefill chunks per request should be > 1.
│         Expected improvement: 20–50% TTFT reduction for long prompts.
│
├── NO  → Do many requests share the same system prompt?
│         ├── YES → Enable prefix caching: --enable-prefix-caching
│         │         Warm the cache with one representative request before
│         │         opening traffic. Monitor: prefix_cache_hit_rate > 0.7.
│         │         Expected improvement: up to 90% TTFT reduction for
│         │         identical system prompts.
│         │
│         └── NO  → Is GPU utilization during prefill < 60%?
│                   ├── YES → Increase --max-num-batched-tokens (default 32768).
│                   │         Try doubling until GPU utilization reaches 80–90%.
│                   │         Warning: too large increases memory pressure.
│                   │
│                   └── NO  → Is this a disaggregated prefill/decode setup?
│                             ├── YES → Check prefill worker count vs request
│                             │         arrival rate. If prefill queue depth > 5,
│                             │         add prefill replicas.
│                             │         See: Chapter 18.
│                             │
│                             └── NO  → Profile with torch.profiler:
│                                       python -c "
│                                       import torch.profiler as p
│                                       with p.profile(activities=[
│                                           p.ProfilerActivity.CPU,
│                                           p.ProfilerActivity.CUDA
│                                       ]) as prof:
│                                           # run one representative request
│                                           pass
│                                       prof.export_chrome_trace('ttft.json')
│                                       "
│                                       Look for: CPU-bound tokenization,
│                                       host-to-device copy latency,
│                                       Python scheduler overhead.
```

**Quick TTFT checklist**:

- [ ] Chunked prefill enabled for prompts > 4 K tokens
- [ ] Prefix caching enabled if system prompt is shared
- [ ] `--max-num-batched-tokens` tuned to fill GPU
- [ ] Prefill replicas scaled to match arrival rate
- [ ] No CPU stall visible in profiler trace

---

### H.1.2 Throughput Ceiling Hit

**Target**: tokens/s should scale linearly with GPU count until the roofline limit.

**Observable signal**: adding load increases queue depth but not `vllm:generation_tokens_total` rate.

```
Compute roofline ratio: actual_tps / theoretical_tps
├── Ratio < 0.5 → Hardware bottleneck.
│   ├── Are weights in BF16?
│   │   └── YES → Switch to FP8: --quantization fp8
│   │             Run perplexity regression (Appendix H.5) before promoting.
│   │             Expected: ~2× throughput improvement on H100.
│   │             See: Chapter 10, Chapter 37.
│   │
│   ├── Is batch utilization low (avg batch size < max_num_seqs / 2)?
│   │   └── YES → Increase --max-num-seqs (default 256).
│   │             Monitor memory: if OOM, reduce --gpu-memory-utilization first.
│   │             Target: avg batch size > 0.6 × max_num_seqs.
│   │
│   ├── Is median context length > 8 K tokens?
│   │   └── YES → Flash Decoding is activated automatically when using
│   │             FlashInfer backend. Verify: check vllm startup logs for
│   │             "attention backend: flashinfer".
│   │             If using triton backend, switch: --attention-backend flashinfer
│   │
│   └── Is GPU utilization < 70% during decode?
│       └── YES → Disaggregate prefill and decode.
│                 Use vLLM V1 disaggregated scheduler.
│                 Prefill nodes: 1–2 GPUs, decode nodes: remainder.
│                 See: Chapter 18.
│
└── Ratio 0.5–0.9 → Software overhead.
    ├── Check Python scheduler overhead: should be < 1 ms per step.
    ├── Verify CUDA graphs are enabled (default in vLLM V1).
    └── Profile with nsight systems for kernel launch gaps.
```

**Throughput optimization sequence** (run in order, measure after each step):
1. Enable FP8: `--quantization fp8`
2. Tune `--max-num-seqs` and `--max-num-batched-tokens`
3. Enable chunked prefill to improve batching efficiency
4. Switch to FlashInfer backend for long contexts
5. Disaggregate if single-node throughput is saturated

---

### H.1.3 OOM (Out of Memory) Crashes

**Observable signals**: `CUDA out of memory` in logs, `vllm:gpu_cache_usage_perc` at 100%.

```
When does the OOM occur?
│
├── DURING STARTUP (model loading)
│   ├── Reduce --gpu-memory-utilization from 0.9 → 0.85 → 0.80
│   │   (each step frees ~5–8 GB on an 80 GB GPU)
│   ├── Verify tensor parallel size covers weight memory:
│   │   weight_GB = params_B × dtype_bytes / tp_size
│   │   Must fit in gpu_memory × gpu_memory_utilization - activations
│   ├── Use --load-format safetensors for streaming load (lower peak RAM)
│   └── If quantized: verify quantization checkpoint matches --quantization flag
│
├── DURING A TRAFFIC BURST
│   ├── Enable request preemption (default in vLLM; verify not disabled)
│   │   --preemption-mode recompute   # lower peak memory, higher latency
│   │   --preemption-mode swap        # uses CPU RAM, avoid on bandwidth-sensitive paths
│   ├── Reduce --max-model-len to cap per-sequence KV footprint
│   │   KV per sequence = 2 × layers × kv_heads × head_dim × max_len × dtype_bytes
│   │   Reducing max_len from 32K to 16K frees ~50% of per-sequence KV budget
│   └── Add replica and load-balance: route long requests to dedicated nodes
│
├── WITH LORA ADAPTERS
│   ├── Each LoRA adapter occupies: rank × d_model × 2 × num_layers × dtype_bytes
│   │   Example: rank=16, d_model=4096, 32 layers, BF16 ≈ 268 MB per adapter
│   ├── Reduce --max-loras (default 1; multiple adapters multiply this cost)
│   ├── Use --enable-lora-bias carefully (adds bias vectors per adapter)
│   └── Consider adapter merging for static-traffic adapters
│
└── WEIGHT LOADING OOM (host RAM)
    ├── Use --load-format safetensors (streams, lower peak host RAM)
    ├── Verify tensor_parallel_size is set correctly before loading
    └── Use HF_HOME on a fast NVMe to avoid double-buffering in RAM
```

**Memory sanity check commands**:
```bash
# Before starting vLLM, check available GPU memory
nvidia-smi --query-gpu=memory.free,memory.total --format=csv

# Estimate weight memory
python -c "
params_b = 70   # model size in billions
dtype_b  = 2    # BF16 = 2 bytes, FP8 = 1 byte
tp       = 4    # tensor parallel size
print(f'Weight memory per GPU: {params_b * 1e9 * dtype_b / tp / 1e9:.1f} GB')
"
```

---

### H.1.4 Latency Spikes / p99 Much Higher Than p50

**Observable signal**: p99/p50 TPOT ratio > 3×, or periodic latency spikes visible in time-series.

```
Is vllm:num_preemptions_total rising?
├── YES → KV cache pressure causing preemptions.
│         ├── Increase GPU memory allocation or reduce max_num_seqs
│         ├── Enable chunked prefill to reduce peak KV pressure
│         └── Consider prefix caching to reuse existing KV blocks
│
├── NO  → Are long requests co-scheduled with short requests?
│         ├── YES → Enable chunked prefill + priority scheduling:
│         │         --enable-chunked-prefill
│         │         --scheduler-policy priority   # if available
│         │         Long requests are chunked, allowing short requests to interleave
│         │
│         └── NO  → Are spikes correlated with periodic intervals (e.g., every 60s)?
│                   ├── YES → Likely Python garbage collection pause.
│                   │         Add to serving startup:
│                   │         import gc; gc.disable()
│                   │         # Re-enable only outside request hot path
│                   │         Profile with: py-spy record -o gc_trace.svg --pid <PID>
│                   │
│                   └── NO  → Are spikes at startup or after idle periods?
│                             ├── YES → CUDA graph capture stall.
│                             │         Verify CUDA graphs are pre-captured at startup.
│                             │         vLLM V1 captures graphs during warmup by default.
│                             │         Check: "CUDA graph captured for batch size X" in logs.
│                             │
│                             └── NO  → Check for thermal throttling:
│                                       nvidia-smi dmon -s u -d 5
│                                       If GPU clocks drop, check cooling / power limits.
```

**p99 latency investigation commands**:
```bash
# Watch preemption rate in real time
watch -n 2 'curl -s localhost:8000/metrics | grep preemption'

# Check KV cache utilization
curl -s localhost:8000/metrics | grep gpu_cache_usage_perc

# Profile GC pauses (requires py-spy)
pip install py-spy
py-spy record -o profile.svg --pid $(pgrep -f vllm)
```

---

### H.1.5 Output Quality Degraded After Optimization

**Observable signal**: user complaints, MMLU/perplexity regression, output truncation or repetition.

```
Which optimization was applied?
│
├── FP8 QUANTIZATION
│   ├── Run perplexity regression immediately:
│   │   python -m lm_eval --model vllm --model_args pretrained=<model>,
│   │       quantization=fp8 --tasks wikitext --device cuda
│   │   Acceptable regression: < 0.3 perplexity points vs BF16 baseline.
│   ├── If regression > 0.5: use static FP8 calibration (not dynamic):
│   │   llm-compressor calibrate --model <path> --calibration-dataset <data>
│   └── If still regressed: fall back to INT8 weight-only (--quantization bitsandbytes)
│
├── PREFIX CACHING
│   ├── Verify KV block hash collisions are not occurring:
│   │   Check vllm logs for "hash collision" warnings.
│   │   Hash collision probability ≈ 1 / 2^64 per block pair (negligible).
│   ├── Verify cached blocks are not from a different model revision:
│   │   Restart vLLM when updating model weights (cache is invalidated on restart).
│   └── Verify --enable-prefix-caching is not mixing adapters incorrectly:
│       Each LoRA adapter gets its own prefix cache namespace in vLLM V1.
│
├── WEIGHT QUANTIZATION (INT4/AWQ/GPTQ)
│   ├── Run MMLU benchmark:
│   │   python -m lm_eval --model vllm --tasks mmlu --num_fewshot 5
│   │   Acceptable: < 1% absolute drop vs FP16 baseline.
│   ├── INT4 regression > 2%: switch to INT8 or AWQ (usually better than GPTQ)
│   └── Check quantization group size: group_size=128 better than group_size=64
│       for accuracy; group_size=64 saves ~1% memory.
│
└── CONTEXT LENGTH EXTENSION (YaRN / RoPE scaling)
    ├── Test with needle-in-haystack: inject fact at various context positions
    │   and verify retrieval accuracy across the full context window.
    └── Regression at long distances (> 50K tokens): reduce RoPE scale factor.
```

---

### H.1.6 Cost Per Request Too High

**Observable signal**: cost/1K tokens higher than budget target.

```
Run 7-layer optimization stack (Chapter 38) in this order:
│
Layer 1: QUANTIZATION (largest impact)
├── BF16 → FP8: ~2× throughput, same GPU count → ~50% cost reduction
└── Measure: perplexity regression must be < 0.3 points

Layer 2: BATCHING EFFICIENCY
├── Check avg_batch_size / max_num_seqs. Target > 0.65.
├── If < 0.65: increase --max-num-seqs, tune --max-num-batched-tokens
└── Consider request buffering (small latency increase, large cost savings)

Layer 3: PREFIX CACHING
├── Measure prefix_cache_hit_rate in metrics
├── If system prompt reuse rate > 50%, prefix caching alone cuts prefill cost by 40–80%
└── For RAG: cache document embeddings + KV to avoid re-encoding

Layer 4: SPECULATIVE DECODING (output-heavy workloads)
├── Effective when output >> input (e.g., code generation)
├── 2–3× decode speedup → same GPU serves 2–3× more requests
└── Requires draft model; adds ~10% memory overhead

Layer 5: AUTO-SCALING
├── Scale to zero during off-peak (useful for batch or low-traffic deployments)
├── Use KubeRay + vLLM V1 for sub-60s scale-up
└── Cost target: 80–90% GPU utilization at peak, NOT 100% (leaves 10% headroom
    for bursts, preventing SLO violations that require expensive retries)

Layer 6: HARDWARE SELECTION
├── H100 vs A100: H100 ~2× cost per hour but ~3× throughput → lower cost/token
├── Spot/preemptible instances for batch workloads (save 60–70%)
└── NVLink topology matters for TP > 4: always prefer NVLink over PCIe

Layer 7: MODEL DISTILLATION / SMALLER MODELS
├── Can a smaller model meet quality SLOs?
│   70B → 8B with RAG: often matches quality at 8× lower cost
└── Measure: A/B test on representative traffic sample before routing all traffic
```

---

## H.2 The vLLM Flags Quick Reference

The 20 most important vLLM engine arguments. All flags are passed to `vllm serve <model>` or set in `EngineArgs`.

| Flag | Default | What it does | When to change | Chapter |
|------|---------|-------------|----------------|---------|
| `--gpu-memory-utilization` | `0.9` | Fraction of GPU HBM reserved for vLLM (weights + KV cache) | Reduce to 0.85 if OOM at startup; increase to 0.95 if GPU memory is underused | Ch 6 |
| `--max-num-seqs` | `256` | Maximum concurrent sequences in the engine | Increase for high-concurrency workloads; reduce if OOM during bursts | Ch 7 |
| `--max-num-batched-tokens` | `32768` | Max tokens in a single forward pass (prefill + decode combined) | Increase for GPU-starved prefill; decrease to reduce TTFT variance | Ch 11 |
| `--max-model-len` | model default | Maximum sequence length (input + output tokens) | Reduce to save KV cache memory; must be ≤ model's trained context length | Ch 6 |
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism | Set to number of GPUs on a single node | Ch 15 |
| `--pipeline-parallel-size` | `1` | Number of pipeline stages across nodes | Set to number of nodes for multi-node; adds bubble overhead | Ch 15 |
| `--quantization` | `None` | Weight quantization method: `fp8`, `awq`, `gptq`, `bitsandbytes` | Always enable `fp8` on H100 for 2× throughput gain | Ch 10 |
| `--enable-prefix-caching` | `False` (V0) / `True` (V1) | Hash-based KV block deduplication for common prefixes | Enable whenever system prompts are shared across requests | Ch 11 |
| `--enable-chunked-prefill` | `False` (V0) / `True` (V1) | Split long prefills into chunks, interleaved with decode | Enable for interactive workloads with mixed prompt lengths | Ch 11 |
| `--max-loras` | `1` | Max simultaneously loaded LoRA adapters | Increase for multi-tenant LoRA serving; each adapter ~200–500 MB | Ch 22 |
| `--lora-extra-vocab-size` | `0` | Extra vocabulary tokens added by LoRA adapters | Set to adapter's extra vocab size if fine-tuned on new tokens | Ch 22 |
| `--speculative-model` | `None` | Path to draft model for speculative decoding | Set to a small (1–3B) model of same architecture | Ch 23 |
| `--num-speculative-tokens` | `5` | Draft tokens proposed per step (γ) | Tune 3–8; higher γ = better speedup if acceptance rate is high | Ch 23 |
| `--load-format` | `auto` | Weight loading format: `safetensors`, `pt`, `gguf`, `npcache` | Use `safetensors` for faster streaming load; avoid `pt` for large models | Ch 8 |
| `--dtype` | `auto` | Compute dtype: `bfloat16`, `float16`, `float32` | Use `bfloat16` always unless model requires `float16` | Ch 2 |
| `--kv-cache-dtype` | `auto` | KV cache storage dtype: `fp8`, `fp16`, `bf16` | Use `fp8` on H100 to reduce KV memory by 2× | Ch 6 |
| `--preemption-mode` | `recompute` | How to handle KV cache eviction: `recompute` or `swap` | Use `recompute` (default); `swap` only if recompute latency is too high | Ch 7 |
| `--attention-backend` | `flashinfer` (V1) | Attention kernel backend: `flashinfer`, `flash_attn`, `triton` | Use `flashinfer` for best performance on H100/A100 | Ch 5 |
| `--swap-space` | `4` (GB) | CPU RAM allocated for KV block swapping | Increase on nodes with abundant RAM; set 0 to disable swapping | Ch 7 |
| `--disable-log-stats` | `False` | Suppress per-step statistics logging | Enable in production to reduce logging overhead (saves ~2% throughput) | Ch 16 |

**Notes**:

- vLLM V1 enables chunked prefill and prefix caching by default. V0 does not.
- `--quantization fp8` requires H100 or newer; on A100, use `awq` or `gptq`.
- `--tensor-parallel-size` must evenly divide `num_attention_heads` and `num_key_value_heads`.

---

## H.3 llama.cpp CLI Quick Reference

Key flags for `llama-cli` and `llama-server`. Flag names shown for the unified CLI (llama.cpp post-2024 refactor).

| Flag | Default | What it does | When to change |
|------|---------|-------------|----------------|
| `-n` / `--predict` | `128` | Maximum output tokens to generate | Set to expected output length; `-1` for unlimited (use with caution) |
| `-c` / `--ctx-size` | `512` | Context window in tokens (input + output) | Increase to model's trained max; larger contexts use more RAM |
| `-ngl` / `--n-gpu-layers` | `0` | Number of transformer layers offloaded to GPU | Set to model's total layer count for full GPU offload; partial for CPU+GPU |
| `-t` / `--threads` | CPU count | CPU threads for compute | Set to physical core count (not hyperthreads) for CPU inference |
| `-tb` / `--threads-batch` | same as `-t` | Threads used during batch/prefill | Can be set higher than `-t` for better prefill throughput |
| `-b` / `--batch-size` | `2048` | Prompt batch size (tokens processed per chunk) | Increase to 4096–8192 for long prompts; limited by RAM |
| `--ubatch-size` | `512` | Micro-batch size for physical computation | Tune for memory: smaller uses less VRAM peak |
| `-m` / `--model` | required | Path to GGUF model file | — |
| `--mlock` | `false` | Lock model weights in RAM (prevent swapping) | Enable in production to prevent OS from swapping weights to disk |
| `--no-mmap` | `false` | Disable memory-mapped file loading | Enable if mmap causes latency spikes on NFS or slow storage |
| `--numa` | `none` | NUMA strategy: `distribute`, `isolate`, `numactl` | On multi-socket servers, use `numactl` with taskset for best performance |
| `-fa` / `--flash-attn` | `false` | Enable Flash Attention | Enable always when supported (requires compatible GGUF and CUDA backend) |
| `--cache-type-k` | `f16` | KV cache dtype for keys: `f16`, `q8_0`, `q4_0` | Use `q8_0` to reduce VRAM by ~50% with minimal quality loss |
| `--cache-type-v` | `f16` | KV cache dtype for values | Same as above; can differ from key cache type |
| `-nkvo` / `--no-kv-offload` | `false` | Disable KV cache GPU offload | Use if VRAM is tight and CPU KV cache is acceptable |
| `--grp-attn-n` | `1` | Group attention factor for context extension | Set to 4–8 for 4–8× context extension (Self-Extend technique) |
| `--temp` | `0.8` | Sampling temperature | 0.0 for greedy/deterministic; 0.7–1.0 for creative tasks |
| `--top-p` | `0.95` | Nucleus sampling threshold | Reduce to 0.85 for more focused outputs |
| `--top-k` | `40` | Top-K sampling | Set to 0 to disable; combine with top-p |
| `--repeat-penalty` | `1.1` | Penalty for repeated tokens | Increase to 1.2–1.3 if model loops; decrease if output is too sparse |
| `-sp` / `--system-prompt` | `""` | System prompt string | Use `-spf` to load from file |
| `--port` | `8080` | HTTP server port (llama-server) | Change if 8080 is in use |
| `--parallel` | `1` | Simultaneous parallel slots (llama-server) | Increase for multi-user serving; each slot needs separate KV cache |
| `--cont-batching` | `true` | Continuous batching (llama-server) | Keep enabled; disable only for debugging |

**llama.cpp performance quick-start** (70B model, single H100 80GB):
```bash
llama-cli \
  -m Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -ngl 80 \           # all 80 layers on GPU
  -c 32768 \          # 32K context
  -b 4096 \           # large prefill batch
  -fa \               # Flash Attention
  --mlock \           # lock weights in memory
  --cache-type-k q8_0 \  # compressed KV cache
  -n 512
```

---

## H.4 Memory Budget Worksheet

Use this template during capacity planning. Fill in values from left to right; each row feeds the next.

```
═══════════════════════════════════════════════════════════════════════════
                    KV CACHE MEMORY BUDGET WORKSHEET
═══════════════════════════════════════════════════════════════════════════

MODEL CONFIGURATION
  Model name:          ________________
  Parameter count:     _____ B  (e.g., 70)
  Precision (weights): ______   (BF16=2B, FP8=1B, INT4=0.5B, INT8=1B)
  Architecture:        ______   (e.g., Llama-3, Qwen2.5, Mistral)

GPU CONFIGURATION
  GPU model:           ___×  ______________________  (e.g., 4× H100 SXM)
  HBM per GPU:         _____ GB   (H100=80, A100=80, A10=24, L40S=48)
  Total HBM:           _____  ×  _____  =  _____ GB
  Tensor parallel:     _____ GPUs  (set to number of GPUs on node)

WEIGHT MEMORY (per GPU after tensor parallelism)
  weight_bytes = params_B × 1e9 × dtype_bytes / tp_size
              = _____ × 1e9 × _____ / _____
              = _____ GB

ACTIVATION MEMORY (estimate, typically 1–3 GB per GPU)
  activation_GB = _____ GB  (use 2 GB as conservative estimate)

KV CACHE BUDGET (per GPU)
  kv_budget = total_HBM × gpu_memory_utilization - weight_GB - activation_GB
            = _____ × _____ - _____ - _____
            = _____ GB

KV CACHE PER TOKEN (per GPU, across all sequences)
  Architecture parameters:
    num_layers:     _____   (e.g., 80 for Llama-3 70B)
    num_kv_heads:   _____   (e.g., 8 for GQA in Llama-3 70B)
    head_dim:       _____   (typically 128)
    kv_dtype_bytes: _____   (FP16/BF16=2, FP8=1, INT8=1)
    tp_size:        _____   (same as above)

  kv_per_token = 2 × num_layers × (num_kv_heads / tp_size) × head_dim × kv_dtype_bytes
               = 2 × _____ × _____ × _____ × _____
               = _____ bytes per token

MAX SEQUENCE CAPACITY
  max_tokens_in_cache = kv_budget_GB × 1e9 / kv_per_token
                      = _____ × 1e9 / _____
                      = _____ tokens total

  Given max_model_len = _____ tokens per sequence:
  max_sequences = max_tokens_in_cache / max_model_len
                = _____ / _____
                = _____ concurrent sequences (theoretical max)

  Practical max (apply 0.8 efficiency factor):
  practical_max_seqs ≈ max_sequences × 0.8 = _____

BLOCK TABLE SIZE
  block_size = _____ tokens per block (vLLM default: 16)
  num_blocks = max_tokens_in_cache / block_size
             = _____ / _____ = _____ blocks

WORKED EXAMPLE (Llama-3.1 70B, BF16, 4× H100, max_len=32K)
  weight_GB   = 70 × 2 / 4       = 35 GB per GPU
  activation  = 2 GB
  kv_budget   = 80 × 0.90 - 35 - 2 = 35 GB per GPU
  kv_per_tok  = 2 × 80 × (8/4) × 128 × 2 = 81,920 bytes ≈ 80 KB/token
  max_tokens  = 35e9 / 81920     ≈ 427,246 tokens
  max_seqs    = 427,246 / 32768  ≈ 13 sequences (× 0.8 ≈ 10 practical)

  → With FP8 KV cache (kv_dtype=1B):
  kv_per_tok  = 2 × 80 × 2 × 128 × 1 = 40,960 bytes
  max_tokens  = 35e9 / 40960     ≈ 854,492 tokens
  max_seqs    ≈ 20 practical     (2× improvement from KV quantization alone)
═══════════════════════════════════════════════════════════════════════════
```

**Copy-paste calculation script**:
```python
# kv_budget_calc.py
params_b        = 70        # model size in billions
weight_dtype_b  = 2         # BF16
tp_size         = 4         # tensor parallel
hbm_per_gpu_gb  = 80        # H100
num_gpus        = 4
gpu_mem_util    = 0.90
activation_gb   = 2.0

num_layers      = 80
num_kv_heads    = 8
head_dim        = 128
kv_dtype_b      = 2         # FP16/BF16 KV cache
max_model_len   = 32768
block_size      = 16

weight_gb     = params_b * 1e9 * weight_dtype_b / tp_size / 1e9
kv_budget_gb  = hbm_per_gpu_gb * gpu_mem_util - weight_gb - activation_gb
kv_per_tok    = 2 * num_layers * (num_kv_heads // tp_size) * head_dim * kv_dtype_b
max_tokens    = int(kv_budget_gb * 1e9 / kv_per_tok)
max_seqs      = max_tokens // max_model_len
num_blocks    = max_tokens // block_size

print(f"Weight memory per GPU:  {weight_gb:.1f} GB")
print(f"KV cache budget:        {kv_budget_gb:.1f} GB")
print(f"KV bytes per token:     {kv_per_tok:,} bytes ({kv_per_tok/1024:.1f} KB)")
print(f"Max tokens in cache:    {max_tokens:,}")
print(f"Max sequences:          {max_seqs} (theoretical) / {int(max_seqs*0.8)} (practical)")
print(f"Number of KV blocks:    {num_blocks:,}")
```

---

## H.5 SLO Calibration Guide

Service Level Objectives for LLM inference differ fundamentally by use case. The following guidelines are based on industry benchmarks and user perception research (Chapter 16).

### Metric Definitions

| Metric | Definition | Notes |
|--------|-----------|-------|
| **TTFT** | Time from request send to first output token | Includes network + queue + prefill time |
| **TPOT** | Time per output token (inter-token latency) | = 1 / decode_tokens_per_second |
| **E2E Latency** | Total time from request to final token | = TTFT + (output_len - 1) × TPOT |
| **p50** | Median latency | Half of requests are faster |
| **p95** | 95th percentile | 1 in 20 requests exceeds this |
| **p99** | 99th percentile | 1 in 100 requests exceeds this |

### SLO Targets by Use Case

**Interactive Chat (e.g., customer support bot, general assistant)**

| Metric | Tight (premium) | Moderate | Relaxed |
|--------|----------------|----------|---------|
| TTFT p50 | < 200 ms | < 400 ms | < 800 ms |
| TTFT p95 | < 500 ms | < 800 ms | < 1,500 ms |
| TPOT p50 | < 30 ms | < 50 ms | < 80 ms |
| TPOT p95 | < 60 ms | < 100 ms | < 150 ms |

Human perception notes: TTFT > 1 s feels "slow" to most users. TPOT > 100 ms disrupts the illusion of streaming. TPOT > 50 ms is noticeable for fast readers. 15–30 ms TPOT (≈ 33–67 tokens/s) is comfortable for most.

**Coding Assistant (e.g., inline code completion, docstring generation)**

| Metric | Tight | Moderate | Relaxed |
|--------|-------|----------|---------|
| TTFT p50 | < 100 ms | < 200 ms | < 400 ms |
| TTFT p95 | < 250 ms | < 500 ms | < 800 ms |
| TPOT p50 | < 20 ms | < 35 ms | < 60 ms |
| E2E (512 output) | < 3 s | < 6 s | < 10 s |

Notes: Coding completions require faster TTFT than chat because users are actively waiting. Short outputs (50–200 tokens) mean E2E latency is dominated by TTFT. Prefer smaller, faster models (8B–14B) over larger models unless quality requires it.

**Batch Processing (e.g., document summarization, data extraction, offline eval)**

| Metric | Target |
|--------|--------|
| TTFT | Not a primary SLO (batch jobs are not interactive) |
| Throughput | Maximize tokens/s |
| E2E per item | < N × baseline (define per job type) |
| Cost per 1K tokens | Primary optimization target |

Notes: For batch workloads, sacrifice latency for throughput. Set `--max-num-seqs` as high as memory allows. Use FP8 and larger batch sizes. Consider offline batching with vLLM's offline API.

**RAG (Retrieval-Augmented Generation)**

| Metric | Target |
|--------|--------|
| Retrieval TTFT (pre-LLM) | < 100 ms (vector DB query) |
| LLM TTFT | < 300 ms p95 |
| TPOT | < 40 ms p95 |
| Total E2E (incl. retrieval) | < 2 s p95 |

Notes: RAG latency budget must account for retrieval time. LLM TTFT must therefore be tighter. Prefix caching is highly effective for RAG: cache the static document KV blocks, only recompute the query KV. Cache hit rate can reach 70–95% if documents are repeated across queries.

**Agentic / Multi-step Reasoning (e.g., tool-calling loops, chain-of-thought)**

| Metric | Target |
|--------|--------|
| Per-step TTFT | < 300 ms |
| Per-step E2E | < 5 s |
| Total task completion | Defined per workflow |

Notes: Agentic workloads compound latency across N steps. A 10-step agent with 1 s per step has 10 s total latency — budget each step accordingly. Caching is critical: cache the growing conversation context across turns.

### SLO Implementation Checklist

- [ ] Define SLOs in terms of p50 AND p95 (p50 alone hides tail behavior)
- [ ] Alert on p99 for early warning; page on p95 breach
- [ ] Set separate SLOs for TTFT and TPOT (they have different root causes)
- [ ] Instrument with Prometheus histograms, not averages
- [ ] Run synthetic load tests at 110% expected peak before production launch
- [ ] Review SLOs quarterly as model size and traffic patterns evolve

---

## H.6 Common Failure Modes Reference Card

Quick-lookup table for on-call engineers. Each row is a distinct failure mode with its diagnostic command and fastest fix.

| Symptom | Likely Cause | Diagnostic Command | Fix |
|---------|-------------|-------------------|-----|
| TTFT > 2× baseline suddenly | Prefix cache cold (after restart or model update) | `curl localhost:8000/metrics \| grep prefix_cache_hit_rate` | Warm cache with representative requests; consider cache persistence in future |
| TPOT increases with load | KV cache filling, preemptions occurring | `curl localhost:8000/metrics \| grep num_preemptions` | Add GPU replicas; reduce `--max-num-seqs`; increase `--gpu-memory-utilization` |
| Requests timing out | Queue backed up; decode not keeping up with arrival | `curl localhost:8000/metrics \| grep num_waiting` | Scale out replicas; enable chunked prefill; shed load via admission control |
| `CUDA out of memory` at startup | Weight + KV reservation exceeds available HBM | `nvidia-smi --query-gpu=memory.free --format=csv` | Reduce `--gpu-memory-utilization`; increase `--tensor-parallel-size` |
| `CUDA out of memory` during burst | Peak KV cache usage exceeds reservation | Check `vllm:gpu_cache_usage_perc` == 1.0 | Reduce `--max-model-len`; reduce `--max-num-seqs`; enable preemption |
| Output tokens cut short | `max_model_len` too small, sequence hits limit | Check `finish_reason: length` in API responses | Increase `--max-model-len`; ensure sufficient KV budget |
| Repetitive / looping output | Sampling temperature issue or KV cache corruption | Compare BF16 vs quantized output for same seed | Increase `--repeat-penalty`; if after quantization, run perplexity check |
| High variance in response quality | Multiple LoRA adapters mixing up | Check adapter routing logs | Verify `lora_request.lora_name` is set correctly per request |
| GPU at 100% but throughput flat | CUDA kernel serialization or PCIe bottleneck | `nsys profile --trace=cuda,nvtx <cmd>` | Enable NVLink; reduce tensor parallel communication overhead |
| GPU at < 30% during decode | Batch too small; memory-bandwidth underutilized | `nvidia-smi dmon -s u -d 1` | Increase `--max-num-seqs`; implement request queuing to fill batches |
| Latency spikes every ~60s | Python GC pause or log flush | `py-spy record --pid <PID> -o trace.svg` | `gc.disable()` in critical path; reduce log verbosity |
| `torch.cuda.CUDAError: device-side assert` | Vocabulary index out of range in sampling | Check tokenizer config matches model | Verify `--tokenizer` path matches model; check for vocabulary mismatch |
| vLLM won't start with quantized model | Quantization format mismatch | Check error: `quantization config mismatch` | Ensure `--quantization` flag matches checkpoint format exactly |
| Health check failing intermittently | Worker process crashed and restarted | `journalctl -u vllm -n 100` | Check for OOM kills in kernel log; add memory headroom |
| Throughput regression after upgrade | New vLLM version changed defaults | Diff vLLM changelogs for flag changes | Pin to known-good version; compare `--max-num-batched-tokens` defaults |
| LoRA adapter not loading | Wrong adapter rank or base model mismatch | Check: `ValueError: LoRA rank mismatch` | Ensure adapter was trained on same base model and same `lora_rank` |
| All requests returning 429 | Rate limiting or admission control triggered | Check `vllm:num_requests_rejected` | Tune admission control threshold; scale out replicas |
| Degraded quality after FP8 migration | FP8 calibration data not representative | Run: `lm_eval --tasks wikitext` | Re-calibrate FP8 with production-representative data |
| High memory usage on CPU | KV block swapping to CPU RAM overwhelming it | `free -h` on serving node | Disable swapping (`--swap-space 0`) or increase node RAM |
| NCCL timeout errors | Inter-GPU communication failure | `NCCL_DEBUG=INFO vllm serve ...` | Check NVLink / InfiniBand health; verify all GPUs on same fabric |
| Slow first request after idle | CUDA kernel JIT compilation on cold path | Time first vs second request | Pre-warm with synthetic requests; CUDA graphs handle most hot paths |

---

## H.7 Deployment Topology Quick Reference

Common deployment patterns and their recommended configurations.

### Single-Node, Single-GPU (Development / Small Models)

```
vllm serve <model> \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill
```

- Use case: models up to 13B, local development, single-user demo
- Expected throughput: 500–2,000 tokens/s depending on model size

### Single-Node, Multi-GPU (Production Small Cluster)

```
vllm serve <model> \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --attention-backend flashinfer
```

- Use case: 70B models on 8× H100
- Expected throughput: 5,000–15,000 tokens/s

### Multi-Node, Disaggregated (High-Scale Production)

```
# Prefill nodes (2 nodes, 8× H100 each)
vllm serve <model> --role prefill \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --quantization fp8

# Decode nodes (4 nodes, 8× H100 each)
vllm serve <model> --role decode \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 4 \
  --quantization fp8 \
  --max-num-seqs 512
```

- Use case: 405B models, 10,000+ RPS
- See Chapter 18 for full disaggregated setup

---

*This appendix is a living reference. Flag defaults and recommended values change with vLLM releases. Always verify against the installed version's `vllm serve --help` output and release notes.*


---

## Worked Solutions

### Question 1
**Decision: Single GPU deployment, model=7B, 100 req/min, P99 TTFT < 500ms, mixed prompt lengths (128-4096 tokens).**

**Working through the decision tree:**

1. **Model fits on one GPU?** 7B BF16 = 14 GB. A100/H100 80 GB: yes. ✓
2. **Throughput requirement:** 100 req/min ≈ 1.67 req/s. At 200 tok/s throughput (7B on A100) and average 512 output tokens: 0.4 req/s sustained. Need to check if this saturates.
3. **Mixed prompt lengths:** Chunked prefill needed to prevent large prompts from blocking short ones.
4. **P99 TTFT < 500ms with 4096-token prompts:** At ~10,000 tok/s prefill throughput, 4096 tokens takes 410ms. Tight but feasible with prefix caching for repeated system prompts.

**Recommended configuration:**
```bash
vllm serve meta-llama/Llama-3.1-7B-Instruct \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4096
```

---

### Question 2
**Decision: 70B model, 1000 req/min, cost-sensitive, team has no GPU cluster.**

**Decision path:**
1. **Own GPU cluster?** No -> consider cloud or quantization.
2. **Cost-sensitive?** Yes -> FP8 or GGUF quantization to reduce GPU count.
3. **70B FP8:** ~70 GB -> fits on 1x H100 80GB with 10GB KV headroom.
4. **1000 req/min = 16.7 req/s:** At 500 tok/s (70B FP8, H100) with avg 300 output tokens: 1.67 req/s per GPU. Need 10x H100s.

**More cost-effective path:**
- Use Q4_K_M (GGUF, ~40 GB) on 1x H100, serving at ~120 tok/s decode.
- 1000 req/min at avg 300 tokens out = 5,000 tok/s needed. Requires ~42 H100s with GGUF.

**Recommendation:** Model routing. Route 70% of requests to a 7B model (handles simple queries at $0.10x cost), escalate 30% to 70B. Effective GPU count drops 7x. See Chapter 31.

---

### Question 3
**Decision: Streaming chatbot, 10 concurrent users max, Apple M2 MacBook Pro (16 GB RAM).**

**Decision path:**
1. **GPU available?** Apple Silicon MPS (unified memory) — limited VRAM.
2. **16 GB unified memory:** Model must fit within ~12 GB (leaving 4 GB for OS/app).
3. **Model choice:** Llama-3.2-3B Q4_K_M (~2 GB) or Qwen2.5-7B Q4_K_M (~4.5 GB).
4. **10 concurrent users:** llama.cpp `--parallel 10` allocates 10 KV cache slots.

**Configuration:**
```bash
llama-server \
  --model qwen2.5-7b-instruct-q4_k_m.gguf \
  --parallel 10 \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --flash-attn
```

KV per slot: 4096 x 128 KB = 512 MB. Total KV: 10 x 512 MB = 5 GB. Model: 4.5 GB. Total: 9.5 GB < 12 GB available. ✓

---

### Question 4
**Decision: RAG pipeline, same 2048-token system prompt for all users, 500 concurrent requests.**

**Key insight:** All users share the same 2048-token system prompt — prefix caching eliminates 100% of system prompt prefill cost after the first request.

**Decision:**
```bash
vllm serve <model> \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 500 \
  --gpu-memory-utilization 0.92
```

**Expected benefit:** With 2048-token shared prefix at 500 concurrent users, prefix cache hit rate approaches 99%+ after warm-up. TTFT drops from ~200ms (prefill) to ~5ms (cache hit) for subsequent requests. This is the highest-ROI single optimization for RAG deployments.

---

### Question 5
**Decision: Production service, need 99.9% uptime, traffic spikes 10x on weekday mornings.**

**Decision path:**
1. **High availability (99.9% uptime):** Minimum 2 replicas at all times (one can fail while the other serves).
2. **10x traffic spike:** HPA with scale-up headroom + predictive scaling for known morning spikes.
3. **Kubernetes + KubeRay:** Necessary for automated scaling.

**Architecture:**
```yaml
# HPA configuration
minReplicas: 2   # always-on baseline
maxReplicas: 20  # handles 10x spike
scaleUp:
  stabilizationWindowSeconds: 30  # react in 30s
  policies:
  - type: Pods
    value: 4    # add 4 pods at a time
    periodSeconds: 30
```

**Predictive scaling:** Schedule a CronJob to pre-scale to 8 replicas at 7:45 AM weekdays (before the 8 AM traffic spike), then allow HPA to handle residual variance. This eliminates the cold-start penalty during the predictable morning ramp.

**Minimum viable SLA configuration:** PodDisruptionBudget with `minAvailable: 1`, `terminationGracePeriodSeconds` = max_sequence_length x ITL, and node anti-affinity spreading replicas across availability zones.

