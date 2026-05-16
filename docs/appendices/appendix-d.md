# Appendix D: Full vLLM EngineArgs Reference

> *"Every knob matters at scale. Knowing what each parameter does — and what it trades off — is the difference between a system that works and one that works well."*

---

This appendix documents all significant vLLM EngineArgs parameters, grouped by function. Values are for vLLM 0.6.x. Defaults are shown in brackets.

---

## D.1 Model and Loading Parameters

### `--model` (required)
The Hugging Face model ID or local path.

```bash
--model meta-llama/Llama-3.1-8B-Instruct
--model /local/path/to/model
```

### `--tokenizer` [same as model]
Path to tokenizer. Use when tokenizer and model are in separate directories.

```bash
--tokenizer /path/to/tokenizer
```

### `--dtype` [auto]
Weight data type. `auto` selects BF16 on supported hardware, else FP16.

```
Values: auto | float32 | float16 | bfloat16
Notes:
  float32: 2× memory vs BF16, no quality benefit for inference
  float16: use on hardware without BF16 support (older A100s, T4)
  bfloat16: preferred on A100/H100; wider dynamic range than FP16
  auto: almost always correct choice
```

### `--revision` [None]
Specific model revision (commit hash or branch) on Hugging Face Hub.

### `--trust-remote-code` [False]
Allow executing custom model code from the Hugging Face repo. Required for some models (e.g., Phi-3, custom architectures).

```bash
--trust-remote-code  # required for models with custom modeling code
```

### `--load-format` [auto]
How to load model weights.

```
Values: auto | pt | safetensors | npcache | dummy | gguf | bitsandbytes
  auto:         detect from file format
  safetensors:  fastest loading, memory-safe
  pt:           PyTorch .bin files
  npcache:      convert to numpy cache on first load (faster subsequent loads)
  dummy:        load random weights (for testing/benchmarking)
  gguf:         load GGUF format directly into vLLM
  bitsandbytes: use bitsandbytes quantization on load
```

### `--max-model-len` [model's default]
Maximum total sequence length (input + output tokens combined).

```bash
# Set lower than model maximum to reduce KV cache memory
--max-model-len 8192   # saves KV memory if you don't need full context
--max-model-len 131072  # full context for Qwen2.5-72B

# KV cache memory at L tokens:
# mem = L × KV_bytes_per_token
# Reducing max-model-len reduces the KV cache allocation
```

### `--tokenizer-mode` [auto]
How to load the tokenizer.

```
Values: auto | slow | mistral
  auto: use HF fast tokenizer if available
  slow: use slow tokenizer (for debugging token issues)
  mistral: use Mistral's custom tokenizer (for Mistral/Mixtral models)
```

---

## D.2 Parallelism Parameters

### `--tensor-parallel-size` / `-tp` [1]
Number of GPUs for tensor parallelism. Must divide evenly into number of attention heads.

```bash
--tensor-parallel-size 4  # 4-way tensor parallel across 4 GPUs

# Requirements:
# - n_heads % tp_size == 0
# - n_kv_heads % tp_size == 0  (or n_kv_heads == 1)
# - GPUs must be on same node (NCCL required)
# For multi-node: combine with --pipeline-parallel-size
```

### `--pipeline-parallel-size` / `-pp` [1]
Number of pipeline stages. Each stage runs on a separate group of GPUs.

```bash
--pipeline-parallel-size 2  # 2 pipeline stages
# Total GPUs = tp_size × pp_size
# E.g., tp=4, pp=2 → 8 GPUs total

# Use for models that don't fit in a single TP group
# Adds pipeline bubble overhead (5-10% throughput penalty)
```

### `--max-parallel-loading-workers` [1]
Number of processes for parallel weight loading. Speeds up model load time.

```bash
--max-parallel-loading-workers 4  # 4 parallel loaders
```

---

## D.3 Memory Management

### `--gpu-memory-utilization` [0.90]
Fraction of GPU memory to use for the KV cache (after weights and activations).

```bash
--gpu-memory-utilization 0.90  # default: leave 10% free

# Lower to 0.80 if you see OOM during peak load
# Can increase to 0.95 if stable (monitor with nvidia-smi)

# What it controls:
# After loading weights: available = total_gpu_mem × utilization - weight_mem
# This "available" goes to KV cache blocks
```

### `--swap-space` [4] (GB)
CPU DRAM swap space for KV blocks when GPU memory is full.

```bash
--swap-space 16  # 16 GB CPU DRAM swap

# When a sequence is preempted (swapped out):
#   KV blocks move GPU → CPU DRAM
#   When resumed: CPU → GPU
# Latency penalty: PCIe bandwidth (~32 GB/s)
# Set to 0 to disable swapping (requests fail if KV cache full)
```

### `--cpu-offload-gb` [0] (GB)
Amount of model weights to offload to CPU. Use for large models that don't fit on GPU.

```bash
--cpu-offload-gb 20  # offload 20GB of weights to CPU

# Creates split execution: some layers on GPU, some on CPU
# Significant latency penalty per offloaded layer
# Use only when no other option
```

### `--kv-cache-dtype` [auto]
Quantization format for KV cache values.

```
Values: auto | fp8 | fp8_e5m2 | fp8_e4m3
  auto:     match model dtype (BF16 by default)
  fp8:      FP8 KV cache (2× memory savings vs BF16)
  fp8_e5m2: 5-bit exponent, 2-bit mantissa (more range)
  fp8_e4m3: 4-bit exponent, 3-bit mantissa (more precision, preferred)

Memory impact:
  BF16 → FP8: 2× more KV capacity → 2× more max sequences
  Quality impact: small (< 0.3 PPL on most models)
```

### `--block-size` [16]
Number of tokens per KV cache block. Larger blocks improve throughput but waste memory for short sequences.

```
Values: 8 | 16 | 32 (must be power of 2)
  16: default, good balance
  8: better for short sequences (less internal fragmentation)
  32: marginally better throughput for long sequences
```

---

## D.4 Scheduling and Batching

### `--max-num-seqs` [256]
Maximum number of sequences in flight simultaneously.

```bash
--max-num-seqs 512  # allow up to 512 concurrent sequences

# Increase for high-throughput workloads
# Each sequence occupies KV cache blocks
# Too high: OOM. Too low: underutilized GPU
```

### `--max-num-batched-tokens` [max_model_len × max_num_seqs or 2048 for chunked prefill]
Maximum total tokens across all sequences in a single forward pass.

```bash
--max-num-batched-tokens 8192

# Controls compute per step:
# Higher = more GPU utilization during prefill
# Lower = better latency fairness (chunked prefill behavior)
# With --enable-chunked-prefill: typically set to 512-2048
```

### `--scheduler-delay-factor` [0.0]
Fraction of mean decode time to wait before scheduling new requests. Increases decode batch size at the cost of TTFT.

```bash
--scheduler-delay-factor 0.5  # wait 50% of mean decode time

# Set > 0 when throughput > TTFT (batch processing use case)
# Set 0.0 for interactive use cases (minimize TTFT)
```

### `--enable-chunked-prefill` [False]
Split prefill across multiple steps to interleave with decode.

```bash
--enable-chunked-prefill
--max-num-batched-tokens 2048  # chunk size

# Benefits:
#   - Reduces TTFT variance (long prefills don't block short decode requests)
#   - Required for mixed prefill/decode workloads
# Trade-off: slightly lower prefill throughput (more iterations)
```

### `--preemption-mode` [recompute]
What to do when KV cache is exhausted.

```
Values: recompute | swap
  recompute: evict and recompute KV from scratch (no CPU memory needed)
  swap:      move KV blocks to CPU DRAM (requires --swap-space > 0)

Use swap when:
  Recompute cost is high (long context)
  CPU DRAM available
Use recompute when:
  Short sequences (fast recompute)
  Limited CPU memory
```

---

## D.5 Attention Mechanism

### `--attention-backend` [Flash Attention 2 or FlashInfer]
Attention kernel implementation.

```
Values: FLASH_ATTN | FLASHINFER | XFORMERS | ROCM_FLASH | TORCH_SDPA
  FLASH_ATTN:  Flash Attention 2 (default on NVIDIA, good all-around)
  FLASHINFER:  FlashInfer kernels (faster for decode-heavy workloads)
  XFORMERS:    xFormers attention (legacy)
  ROCM_FLASH:  AMD GPU Flash Attention
  TORCH_SDPA:  PyTorch scaled dot-product attention (fallback)

Set via environment variable:
  export VLLM_ATTENTION_BACKEND=FLASHINFER
```

### `--enable-prefix-caching` [False]
Cache and reuse KV blocks for repeated prompt prefixes.

```bash
--enable-prefix-caching

# Enables RadixAttention (Chapter 11)
# Benefits:
#   - Repeated system prompts: compute prefix KV once
#   - RAG workflows: shared context prefix amortized
# Overhead: ~1% memory for hash table
# Recommended: always enable for RAG/chat workloads
```

### `--num-gpu-blocks-override` [None]
Override the auto-calculated number of KV cache GPU blocks.

```bash
--num-gpu-blocks-override 2000  # exactly 2000 GPU blocks

# Use for reproducible benchmarking across configurations
# Normal usage: leave unset (let vLLM calculate)
```

---

## D.6 Quantization

### `--quantization` / `-q` [None]
Post-training quantization method.

```
Values: awq | gptq | squeezellm | fp8 | bitsandbytes | gguf | None
  awq:           AWQ INT4 (Activation-aware Weight Quantization)
  gptq:          GPTQ INT4 (one-shot weight quantization)
  squeezellm:    SqueezeLLM sparse quantization
  fp8:           FP8 E4M3 weight quantization (H100 only)
  bitsandbytes:  4-bit or 8-bit quantization via bitsandbytes
  None:          no quantization (load in original dtype)

# Must match the model format. AWQ model needs --quantization awq.
# Do not set --quantization if model is already in original BF16/FP16.
```

### `--quantization-param-path` [None]
Path to FP8 quantization parameter file (KV cache scale factors).

```bash
--quantization-param-path /path/to/kv_cache_scales.json

# Required when using FP8 KV cache with dynamically quantized models
# Generated by: python examples/fp8/extract_scales.py
```

### `--enforce-eager` [False]
Disable CUDA graph capture. Use for debugging or models with dynamic control flow.

```bash
--enforce-eager  # disables CUDA graphs (slower, but avoids graph capture issues)

# CUDA graphs accelerate decode by 10-15%
# Disable when:
#   Debugging OOM or incorrect outputs
#   Model has unsupported dynamic shapes
#   Adapter hot-swapping (some LoRA configurations)
```

---

## D.7 Speculative Decoding

### `--speculative-model` [None]
Draft model for speculative decoding.

```bash
--speculative-model meta-llama/Llama-3.2-1B  # use 1B as draft
--model meta-llama/Llama-3.1-70B-Instruct    # 70B as target

# Requirements:
#   Draft and target must share tokenizer vocabulary
#   Draft must be substantially smaller than target
```

### `--num-speculative-tokens` [None]
Number of draft tokens to generate per step.

```bash
--num-speculative-tokens 5  # generate 5 draft tokens, verify with target

# Optimal value depends on acceptance rate:
#   High acceptance (α > 0.8): increase to 7-10
#   Low acceptance (α < 0.5):  decrease to 2-3
#   Rule of thumb: start at 5, tune based on acceptance rate metric
```

### `--speculative-draft-tensor-parallel-size` [same as main model]
Tensor parallel size for the draft model.

```bash
--speculative-draft-tensor-parallel-size 1  # run draft on 1 GPU

# Draft models are small; TP=1 usually optimal
# Target can still use TP=4+
```

### `--use-v2-block-manager` [True in recent versions]
Use V2 block manager with speculative decoding support.

```bash
--use-v2-block-manager  # required for speculative decoding
```

---

## D.8 LoRA and Adapters

### `--enable-lora` [False]
Enable LoRA adapter serving.

```bash
--enable-lora
--max-loras 4          # max simultaneously loaded adapters
--max-lora-rank 64     # max rank across all adapters
--lora-extra-vocab-size 0  # extra vocab tokens for LoRA models
```

### `--max-loras` [1]
Maximum number of LoRA adapters loaded simultaneously.

### `--max-lora-rank` [16]
Maximum rank parameter across all LoRA adapters.

### `--long-lora-scaling-factors` [None]
Scaling factors for long-context LoRA adapters. Comma-separated list of floats.

---

## D.9 Server and API

### `--host` [0.0.0.0]
Server hostname.

### `--port` [8000]
Server port.

```bash
--host 127.0.0.1 --port 8080  # local only, port 8080
```

### `--api-key` [None]
API key for authentication. If set, requests must include `Authorization: Bearer <key>`.

```bash
--api-key my-secret-key-here
```

### `--chat-template` [model default]
Jinja2 chat template file or string. Use to override model's built-in template.

```bash
--chat-template ./my_custom_template.jinja

# Useful when:
#   Model's template is wrong or missing
#   Applying non-default instruct format
```

### `--response-role` [assistant]
Role name in chat response.

### `--max-log-len` [None]
Maximum characters to log per request (for privacy).

```bash
--max-log-len 100  # truncate request logs to 100 chars
```

### `--root-path` [None]
URL root path when running behind a reverse proxy.

```bash
--root-path /llm  # serve at http://host/llm/v1/completions
```

---

## D.10 Multimodal

### `--limit-mm-per-prompt` [None]
Maximum number of multimodal items per prompt.

```bash
--limit-mm-per-prompt image=5  # allow up to 5 images per prompt
--limit-mm-per-prompt image=10,video=2
```

### `--mm-processor-kwargs` [None]
Extra keyword arguments for the multimodal processor.

```bash
--mm-processor-kwargs '{"max_dynamic_patch": 4}'
```

---

## D.11 Logging and Observability

### `--disable-log-requests` [False]
Disable per-request logging. Recommended in high-throughput production.

```bash
--disable-log-requests  # production: reduce log volume
```

### `--disable-log-stats` [False]
Disable periodic statistics logging.

### `--log-level` [INFO]
Logging verbosity.

```
Values: DEBUG | INFO | WARNING | ERROR
```

### `--collect-detailed-traces` [None]
Enable detailed tracing for observability (Prometheus/OpenTelemetry).

```bash
--collect-detailed-traces all  # trace all components
--collect-detailed-traces model,scheduler  # trace specific components
```

---

## D.12 Quick Presets by Use Case

```bash
# === HIGH THROUGHPUT (batch inference) ===
vllm serve MODEL \
    --max-num-seqs 512 \
    --max-num-batched-tokens 32768 \
    --scheduler-delay-factor 0.3 \
    --gpu-memory-utilization 0.92

# === LOW LATENCY (interactive) ===
vllm serve MODEL \
    --max-num-seqs 32 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --scheduler-delay-factor 0.0

# === LONG CONTEXT (RAG) ===
vllm serve MODEL \
    --max-model-len 131072 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 4096

# === MEMORY CONSTRAINED (small GPU) ===
vllm serve MODEL \
    --quantization awq \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16

# === SPECULATIVE DECODING ===
vllm serve LARGE_MODEL \
    --speculative-model SMALL_DRAFT_MODEL \
    --num-speculative-tokens 5 \
    --use-v2-block-manager

# === MULTI-GPU (4×H100) ===
vllm serve LARGE_MODEL \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90
```

---

## D.13 EngineArgs in Python

For programmatic configuration:

```python
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_num_seqs=256,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_num_batched_tokens=2048,
    kv_cache_dtype="fp8",
    disable_log_requests=True,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
```

---

## D.14 SamplingParams Reference

SamplingParams controls how tokens are sampled per request (not engine-level):

```python
from vllm import SamplingParams

params = SamplingParams(
    temperature=0.7,        # sampling temperature (0 = greedy)
    top_p=0.9,              # nucleus sampling threshold
    top_k=50,               # top-k sampling (0 = disabled)
    min_p=0.0,              # min-p sampling (0 = disabled)
    repetition_penalty=1.0, # penalize repeated tokens (1.0 = no penalty)
    frequency_penalty=0.0,  # penalize by frequency (0 = no penalty)
    presence_penalty=0.0,   # penalize presence (0 = no penalty)
    max_tokens=512,         # max output tokens
    min_tokens=0,           # min output tokens (0 = no minimum)
    stop=["<|im_end|>"],    # stop sequences
    stop_token_ids=[],      # stop token IDs
    include_stop_str_in_output=False,
    ignore_eos=False,       # continue past EOS token
    logprobs=None,          # return top-k logprobs (None = disabled)
    prompt_logprobs=None,   # return input token logprobs
    skip_special_tokens=True,
    spaces_between_special_tokens=True,
    seed=None,              # random seed for reproducibility
)
```
