# Appendix E: llama.cpp Context and Model Parameters Reference

> *"llama.cpp has a deceptively simple CLI. Under the surface, every parameter tunes a specific trade-off between speed, memory, and quality."*

---

This appendix documents llama.cpp's most important runtime parameters for both the CLI (`llama-cli`) and server (`llama-server`). Parameters are as of build b4000+.

---

## E.1 Model Loading Parameters

### `-m` / `--model` (required)
Path to the GGUF model file.

```bash
-m ./models/llama-3.1-8b-q4_k_m.gguf
```

### `-md` / `--model-draft`
Path to draft model for speculative decoding.

```bash
-m ./models/llama-3.1-70b-q4_k_m.gguf \
-md ./models/llama-3.2-1b-q8_0.gguf    # draft for speculative decoding
```

### `--mlock`
Lock model in RAM (prevents paging to disk).

```bash
--mlock  # prevents OS from swapping model pages to disk
         # requires sufficient RAM; use for consistent latency
```

### `--no-mmap`
Load model fully into memory instead of using memory-mapped file access.

```bash
--no-mmap  # load fully into RAM

# Default: model is mmap'd (reads pages on demand)
# --no-mmap: reads everything upfront
# Use when:
#   - Model is on network filesystem (mmap has high latency)
#   - Need consistent inference latency (avoid page faults)
#   - Profiling (eliminate mmap effects)
```

### `-ngl` / `--n-gpu-layers`
Number of model layers to offload to GPU.

```bash
-ngl 99    # offload all layers (use large number to offload everything)
-ngl 32    # offload 32 layers, rest on CPU
-ngl 0     # CPU-only (no GPU)

# Memory calculation:
# Total layers = n_layers + 1 (embedding)
# Memory per layer ≈ model_size / n_layers
# Example: Llama-3.1-8B Q4_K_M = 4.7 GB, 32 layers
#   1 layer ≈ 4.7 GB / 32 ≈ 147 MB
#   ngl=16: ~2.35 GB GPU + ~2.35 GB CPU (split)
```

### `-sm` / `--split-mode`
How to distribute model across multiple GPUs.

```
Values: none | layer | row
  none:  all on one GPU (no split)
  layer: split by transformer layer (default multi-GPU mode)
  row:   split by matrix row (for very large models)
```

### `-ts` / `--tensor-split`
Tensor split ratios for multi-GPU distribution.

```bash
-ts 3,1    # 75% on GPU 0, 25% on GPU 1 (proportional split)
-ts 1,1    # equal split between 2 GPUs
-ts 1,1,1,1  # equal split across 4 GPUs
```

### `-mg` / `--main-gpu`
GPU index for operations that can only use one GPU.

```bash
-mg 0  # use GPU 0 for non-parallel operations (default)
```

---

## E.2 Context and Sequence Parameters

### `-c` / `--ctx-size` [512]
Context window size in tokens (KV cache size).

```bash
-c 4096    # 4K context
-c 32768   # 32K context
-c 131072  # 128K context (requires GQA model + sufficient VRAM)

# KV cache memory = c × KV_bytes_per_token
# Llama-3.1-8B BF16: 131,072 × 131,072 bytes = 16 GB just for KV
# Use -c as small as your task requires
```

### `-n` / `--n-predict` [-1]
Maximum number of tokens to generate. -1 means unlimited (until EOS or context full).

```bash
-n 256    # generate at most 256 tokens
-n -1     # generate until EOS
-n -2     # generate until context full
```

### `-e` / `--escape`
Enable escape sequences in prompts (allows `\n`, `\t`, etc.).

```bash
-e  # interpret escape sequences in prompt string
```

### `--keep` [0]
Number of initial prompt tokens to keep when context shifts.

```bash
--keep 256  # keep first 256 tokens when truncating context

# When context fills up, llama.cpp discards old tokens
# --keep N: always keep the first N tokens (e.g., system prompt)
# 0: no special protection for initial tokens
# -1: keep all initial tokens (disables context shifting)
```

---

## E.3 Sampling Parameters

### `-temp` / `--temperature` [0.80]
Sampling temperature. Lower = more deterministic.

```bash
--temp 0.7    # slightly creative
--temp 0.0    # greedy decoding (deterministic)
--temp 1.0    # high diversity
```

### `--top-k` [40]
Top-k sampling. Keep only top k most likely tokens.

```bash
--top-k 50   # consider top 50 tokens
--top-k 0    # disable (consider all tokens)
--top-k 1    # greedy decoding (same as temp=0)
```

### `--top-p` [0.95]
Nucleus sampling. Keep tokens whose cumulative probability exceeds p.

```bash
--top-p 0.9   # use tokens that make up top 90% of probability mass
--top-p 1.0   # disable (use all tokens)
```

### `--min-p` [0.05]
Min-p sampling. Discard tokens with probability below this fraction of the most likely token.

```bash
--min-p 0.1   # discard tokens with prob < 10% of top token's prob
--min-p 0.0   # disable
```

### `--repeat-penalty` [1.0]
Penalize repeated tokens. Values > 1 discourage repetition.

```bash
--repeat-penalty 1.1    # mild repetition penalty
--repeat-penalty 1.3    # strong repetition penalty
--repeat-penalty 1.0    # no penalty (default)
```

### `--repeat-last-n` [64]
Number of recent tokens to consider for repetition penalty.

```bash
--repeat-last-n 128   # penalize tokens repeated in last 128
--repeat-last-n -1    # consider entire context
--repeat-last-n 0     # disable repetition penalty
```

### `--frequency-penalty` [0.0]
Penalize tokens proportional to their frequency in the output so far.

### `--presence-penalty` [0.0]
Penalize tokens that have appeared at all (once is enough to reduce their probability).

### `--seed` [-1]
Random seed. -1 = random seed (non-deterministic). Set for reproducible outputs.

```bash
--seed 42   # reproducible sampling
```

### `-s` / `--samplers`
Ordered list of sampler operations to apply.

```bash
--samplers "top_k;tfs_z;typical_p;top_p;min_p;temperature"
# Apply samplers in this order
```

---

## E.4 Batch and Performance Parameters

### `-b` / `--batch-size` [2048]
Physical batch size — number of tokens processed per forward pass during prefill.

```bash
-b 512    # smaller batches (lower memory, slightly slower prefill)
-b 2048   # default (good balance)
-b 4096   # larger batches (faster prefill if model fits)

# This is the compute batch for prefill, NOT number of parallel sequences
```

### `-ub` / `--ubatch-size` [512]
Micro-batch size. Physical processing unit within a batch.

```bash
-ub 512   # process 512 tokens at a time within each batch
```

### `-np` / `--parallel` [1]
Number of parallel sequences (only in llama-server).

```bash
-np 8    # handle 8 simultaneous conversations
# Each slot gets its own KV cache: total KV = np × ctx_size × KV_bytes
```

### `-t` / `--threads` [auto]
Number of CPU threads for inference (when using CPU layers).

```bash
-t 8    # use 8 CPU threads
-t $(nproc)  # use all CPU cores
```

### `-tb` / `--threads-batch` [same as -t]
CPU threads for batch processing (prefill). Can be different from generation threads.

### `--cont-batching` / `-cb`
Enable continuous batching in llama-server (handles multiple requests efficiently).

```bash
-cb   # enable continuous batching (should always be enabled in server)
```

---

## E.5 Quantization Type Reference

GGUF quantization types ordered by quality/size trade-off:

```
Format      | Bits/Weight | Quality | Notes
────────────────────────────────────────────────────────────────────
F32         | 32          | Lossless| Reference, too large for use
F16         | 16          | ~Lossless| Half precision, large
BF16        | 16          | ~Lossless| BFloat16 (preferred for H100)
Q8_0        |  8          | Excellent| INT8, minimal quality loss
Q6_K        |  6          | Very good| K-quant at 6-bit
Q5_K_M      |  5          | Good+   | K-quant medium 5-bit (recommended)
Q5_K_S      |  5          | Good    | K-quant small 5-bit
Q5_0        |  5          | Good    | Legacy 5-bit
Q4_K_M      |  4.5        | Good    | K-quant medium 4-bit (most popular)
Q4_K_S      |  4.2        | Decent  | K-quant small 4-bit
Q4_0        |  4          | Decent  | Legacy 4-bit
Q3_K_M      |  3.4        | Acceptable| K-quant medium 3-bit
Q3_K_S      |  3.0        | OK      | K-quant small 3-bit
Q2_K        |  2.6        | Degraded| Only for extreme compression
IQ4_XS      |  4.3        | Good    | Importance-weighted 4-bit
IQ3_M       |  3.7        | Good    | Importance-weighted 3-bit
IQ2_M       |  2.7        | Decent  | Importance-weighted 2-bit
────────────────────────────────────────────────────────────────────
K-quants: use larger quantization blocks with per-block scales
IQ-quants: use importance sampling to quantize important weights less
```

**Recommendation by use case:**

```bash
# Best quality: Q8_0 or Q6_K
# Balanced (most users): Q4_K_M
# Memory constrained: Q3_K_M
# Extreme compression (quality matters less): Q2_K
# Edge/mobile: IQ4_XS or Q4_K_S
```

---

## E.6 Chat and Prompt Format Parameters

### `--chat-template` [auto-detect]
Chat template to use.

```bash
--chat-template llama3   # Llama 3.1 format
--chat-template qwen     # Qwen format
--chat-template chatml   # ChatML format
--chat-template mistral  # Mistral format
--chat-template gemma    # Gemma format

# Always specify for instruct models — wrong template = garbled output
```

### `-sys` / `--system-prompt`
System prompt to prepend.

```bash
--system-prompt "You are a helpful coding assistant."
```

### `-p` / `--prompt`
Initial prompt string.

```bash
-p "What is the capital of France?"
```

### `-f` / `--file`
Read prompt from file.

```bash
-f ./prompt.txt
```

### `--in-prefix` / `--in-suffix`
Prefix/suffix wrapping user input in interactive mode.

```bash
# For manual chat templating:
--in-prefix "<|im_start|>user\n"
--in-suffix "<|im_end|>\n<|im_start|>assistant\n"
```

### `-i` / `--interactive`
Enable interactive/chat mode.

```bash
-i   # interactive mode (continue generating from user input)
```

### `-r` / `--reverse-prompt`
Reverse prompt string to pause generation and wait for input.

```bash
-r "User:"  # pause when model generates "User:"
```

---

## E.7 llama-server Specific Parameters

### `--host` [127.0.0.1]
Server host address.

```bash
--host 0.0.0.0  # listen on all interfaces
--host 127.0.0.1  # local only
```

### `--port` [8080]
Server port.

```bash
--port 8080
```

### `--api-key`
Bearer token for API authentication.

```bash
--api-key my-secret-key
```

### `--path`
Root path for static files (web UI).

### `--timeout` [600]
Request timeout in seconds.

### `--n-predict` / `-n` [-1]
Default max tokens per request (can be overridden per request).

### `--slots-endpoint-enabled`
Enable the `/slots` endpoint for inspecting active KV cache slots.

---

## E.8 Speculative Decoding Parameters

```bash
# Enable speculative decoding
llama-server \
    -m ./target-70b-q4_k_m.gguf \
    -md ./draft-1b-q8_0.gguf \
    --draft 10 \              # speculate 10 tokens
    -ngl 99 \                 # all layers to GPU
    --draft-ngl 99 \          # draft model also on GPU
    -c 4096 \
    -np 4                     # 4 parallel slots

# Parameters:
--draft N           # number of draft tokens (default: 5)
--draft-ngl N       # GPU layers for draft model
-pv                 # verbose speculative decoding stats
```

---

## E.9 Embedding Parameters

For running llama.cpp as an embedding server:

```bash
llama-server \
    -m ./nomic-embed-text-v1.5-q8_0.gguf \
    --embedding \         # enable embedding mode
    --no-cont-batching \  # embedding models don't use cont batching
    -ngl 99 \
    -c 2048 \
    --port 8080

# Test
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "text-embedding"}'
```

---

## E.10 Complete Server Command Reference

```bash
# Minimal production server
llama-server \
    -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -ngl 99 \
    -c 16384 \
    -np 8 \
    --chat-template qwen \
    -cb \
    --host 0.0.0.0 \
    --port 8080

# High-throughput server (large GPU)
llama-server \
    -m ./models/llama-3.1-70b-instruct-q4_k_m.gguf \
    -ngl 99 \
    -c 8192 \
    -np 16 \
    -b 4096 \
    -ub 1024 \
    -t 8 \
    --chat-template llama3 \
    -cb \
    --host 0.0.0.0 \
    --port 8080

# Long-context server
llama-server \
    -m ./models/qwen2.5-72b-instruct-q4_k_m.gguf \
    -ngl 99 \
    -c 131072 \
    -np 2 \          # fewer slots due to large KV per slot
    --chat-template qwen \
    -cb \
    --host 0.0.0.0 \
    --port 8080

# CPU-only server (no GPU)
llama-server \
    -m ./models/llama-3.2-1b-instruct-q4_k_m.gguf \
    -ngl 0 \          # all layers on CPU
    -t $(nproc) \     # use all CPU cores
    -c 4096 \
    -np 4 \
    --host 0.0.0.0 \
    --port 8080
```

---

## E.11 Performance Tuning Guide

```
Scenario                   | Key Parameters
─────────────────────────────────────────────────────────────────────
Maximize tokens/second     | -b 4096, -ngl 99, -np 1 (decode single request)
Minimize TTFT              | -b 2048, --cont-batching, small context (-c)
Maximize concurrent users  | -np 16+, smaller context, Q4_K_M quantization
Apple Silicon M-series     | -ngl 99, Metal enabled by default in build
CPU-only on server         | -t $(nproc), Q4_0 or Q4_K_M, -b 512
Memory constrained (<4GB)  | Q2_K or Q3_K_S, -c 2048, -np 1
```

---

## E.12 Diagnostic Flags

```bash
# Verbose timing output
--verbose

# Log all tokens as generated
--log-disable   # suppress logs (default: logs to stderr)

# Benchmark mode (measure tokens/second)
llama-bench \
    -m ./model.gguf \
    -n 128 \          # tokens to generate
    -p 512 \          # prompt tokens
    -b 512 \          # batch size
    -r 5              # repeat 5 times (for statistics)

# Output format:
# model | size | params | backend | ngl | test | t/s
# ...   | 4.7G | 8.0B   | CUDA    | 99  | pp512| 7234.56
```
