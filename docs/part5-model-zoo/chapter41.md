# Chapter 41 — Meta Llama 3: Architecture, Ecosystem, and Inference

> *"Llama is not a model. It is the baseline against which everything else is measured."*

---

## 41.1 Why Llama Deserves Its Own Chapter

The Llama family occupies a unique position in LLM inference engineering. It is
simultaneously the dominant open-weight model (by deployment volume), the
reference point for virtually every benchmark, the training target for the
majority of fine-tuning workflows, and the default model in most inference
engine documentation.

When vLLM, llama.cpp, SGLang, TensorRT-LLM, and MLC-LLM publish benchmark
numbers, they publish them on Llama. When a quantization researcher evaluates
a new method, they evaluate it on Llama. When an enterprise deploys a self-
hosted model, the first model they try is Llama.

Understanding Llama's architecture at the level of inference engineering — not
as an ML practitioner, but as someone who needs to configure, optimize, and
serve it — is foundational knowledge.

---

## 41.2 The Llama 3 Architecture

Llama 3 is a decoder-only transformer. There is nothing exotic about its core
design — it deliberately avoids architectural novelty in favour of scale and
training quality. What makes it interesting for inference engineering is the
specific set of choices Meta made and their practical consequences.

### 41.2.1 Grouped Query Attention

Llama 3 uses Grouped Query Attention (GQA, Chapter 4). The query, key, and
value head counts are asymmetric:

| Model | Query heads | KV heads | Group size |
|---|---|---|---|
| Llama 3 8B | 32 | 8 | 4 |
| Llama 3 70B | 64 | 8 | 8 |
| Llama 3 405B | 128 | 8 | 16 |

All three sizes use 8 KV heads regardless of model size. This is the most
inference-relevant architectural decision in Llama 3.

**KV cache sizing consequence:**

```
KV cache per token per layer = 2 × num_kv_heads × head_dim × bytes_per_element

Llama 3 8B  (FP16): 2 × 8 × 128 × 2 = 4,096 bytes = 4 KB/token/layer
Llama 3 70B (FP16): 2 × 8 × 128 × 2 = 4,096 bytes = 4 KB/token/layer
```

Wait — the same 4 KB/token/layer for both? Yes. Because GQA fixes KV heads at
8 regardless of model size, the KV cache per token is identical for 8B and 70B.
The 70B model has more layers (80 vs 32), so:

```
Full KV cache per token:
  8B  (32 layers, FP16): 4 KB × 32  = 128 KB/token
  70B (80 layers, FP16): 4 KB × 80  = 320 KB/token
  405B (126 layers, FP16): 4 KB × 126 = 504 KB/token
```

For a 4,096-token context on 70B: `320 KB × 4,096 = 1.28 GB` — per request.
An H100 with 80 GB VRAM and a 70B model in FP16 (~140 GB, requiring 2 GPUs)
can hold approximately `(80 GB - 70 GB)/320 KB = ~32,000 tokens` of KV cache
per GPU. This is the arithmetic that governs batch size limits.

### 41.2.2 SwiGLU Feed-Forward Network

Llama 3's FFN uses SwiGLU activation with an intermediate dimension of
approximately `8/3 × d_model`, rounded to a multiple of 256:

```
  FFN input: x ∈ R^d_model
  gate  = SiLU(x × W_gate)    W_gate ∈ R^{d_model × d_ffn}
  up    = x × W_up             W_up   ∈ R^{d_model × d_ffn}
  h     = gate ⊙ up            (element-wise product)
  output = h × W_down          W_down ∈ R^{d_ffn × d_model}
```

| Model | d_model | d_ffn | Parameters in FFN |
|---|---|---|---|
| 8B | 4,096 | 14,336 | 3 × 4,096 × 14,336 = 176M per layer |
| 70B | 8,192 | 28,672 | 3 × 8,192 × 28,672 = 705M per layer |
| 405B | 16,384 | 53,248 | 3 × 16,384 × 53,248 = 2.6B per layer |

The three-matrix structure (W_gate, W_up, W_down) means the FFN has 3×
the parameters of a standard two-matrix FFN at the same intermediate dimension.
For inference, this means 3 GEMMs per FFN block, not 2 — relevant for
arithmetic intensity calculations in Chapter 9.

### 41.2.3 RoPE Positional Encoding

Llama 3 uses Rotary Positional Encoding (RoPE) with a base frequency θ = 500,000.
This is 10× larger than Llama 2's θ = 10,000 and is the key change enabling
longer context lengths without extensive fine-tuning on long sequences.

**RoPE scaling for extended context (Llama 3.1):**

Llama 3.1 extends context to 128K tokens using a combination of:

1. **High base frequency** (θ = 500,000 inherited from 3.0)
2. **RoPE scaling factor** applied to further stretch the frequency range

```python
# Llama 3.1 RoPE configuration (from config.json)
rope_scaling = {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
}
```

The "llama3" RoPE type applies frequency-dependent scaling: low-frequency
components (long-range dependencies) are scaled more aggressively than high-
frequency components (local dependencies). This avoids degrading short-context
performance while extending long-context capability.

For inference configuration:

```python
# vLLM: enable full 128K context
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    max_model_len=131072,        # 128K tokens
    gpu_memory_utilization=0.95  # need more for large KV cache
)
```

At 128K context, the KV cache for a single 70B request is:
`320 KB × 131,072 = 40.96 GB` — larger than the model weights on a single H100.
This is why long-context serving requires careful memory management (Chapter 27).

### 41.2.4 RMSNorm Pre-Normalization

Llama 3 uses RMSNorm (Root Mean Square Normalization) applied before each
attention and FFN block (pre-norm), with no bias terms:

```
RMSNorm(x) = x / RMS(x) × γ
RMS(x) = sqrt(mean(x²))
```

RMSNorm is ~10% faster than LayerNorm (no mean subtraction) and is numerically
stable with no bias term to initialise. For inference, this means the
normalization layers fuse cleanly into the attention and FFN kernels — vLLM's
and TensorRT-LLM's kernels typically fuse RMSNorm with the subsequent QKV
projection in a single kernel call.

### 41.2.5 Tokenizer: tiktoken-based BPE

Llama 3 abandons the SentencePiece tokenizer of Llama 2 in favour of a
tiktoken-compatible BPE tokenizer with a 128,256-token vocabulary (vs 32,000
for Llama 2).

**Inference consequences:**

The larger vocabulary increases:
- The output projection matrix size: `d_model × vocab_size`
  - 8B: 4,096 × 128,256 = 525M parameters in the lm_head (~1 GB in FP16)
  - 70B: 8,192 × 128,256 = 1.05B parameters (~2 GB in FP16)
- Softmax compute at each decode step: over 128K logits vs 32K

The larger vocabulary also improves throughput in practice: longer average
token length means fewer decode steps per unit of generated text, so the
per-token softmax overhead is amortised over more characters.

---

## 41.3 The Llama 3 Family

### Llama 3.0 (April 2024)

The base release: 8B and 70B, 8K context, θ = 500,000. The first Meta model
to use GQA at both sizes and the tiktoken vocabulary.

```bash
# Available HuggingFace IDs
meta-llama/Meta-Llama-3-8B
meta-llama/Meta-Llama-3-8B-Instruct
meta-llama/Meta-Llama-3-70B
meta-llama/Meta-Llama-3-70B-Instruct
```

### Llama 3.1 (July 2024)

Extended context to 128K, added the 405B flagship, and introduced the RoPE
scaling described above. The 405B model uses pipeline parallelism to serve
across multiple nodes in production (Chapter 15).

```bash
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.1-70B-Instruct
meta-llama/Llama-3.1-405B-Instruct
```

### Llama 3.2 (September 2024)

Two additions: multimodal capability and small models.

**Multimodal (11B, 90B)**: Vision encoder attached to the language model via
cross-attention layers. Chapter 29 covers multimodal inference; the Llama 3.2
vision models are a canonical example.

**Small models (1B, 3B)**: Pure language models for edge deployment. The 1B
model runs comfortably on an iPhone. The 3B model fits within the thermal
envelope of a Raspberry Pi 5 at 4-bit quantization.

```bash
meta-llama/Llama-3.2-1B-Instruct     # edge
meta-llama/Llama-3.2-3B-Instruct     # edge
meta-llama/Llama-3.2-11B-Vision-Instruct   # multimodal
meta-llama/Llama-3.2-90B-Vision-Instruct   # multimodal
```

### Llama 3.3 (December 2024)

A 70B model with improved instruction following that matches Llama 3.1 405B on
most benchmarks. For deployment purposes, this is the most important release:
405B quality at 70B cost.

```bash
meta-llama/Llama-3.3-70B-Instruct
```

---

## 41.4 The Llama 3 Prompt Format

Llama 3 uses a specific chat template. Getting this wrong produces degraded
or incoherent responses, so understanding it is essential.

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

{system_message}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_message}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```

The special tokens are:
- `<|begin_of_text|>` — BOS (beginning of sequence)
- `<|start_header_id|>` / `<|end_header_id|>` — wraps role names
- `<|eot_id|>` — end of turn
- `<|end_of_text|>` — EOS

```python
# Using the HuggingFace apply_chat_template
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is 2 + 2?"},
]

formatted = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True   # adds the trailing assistant header
)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful assistant.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# What is 2 + 2?<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
```

vLLM and llama.cpp both apply this template automatically when using the
chat endpoint (`/v1/chat/completions`). If using the completions endpoint
(`/v1/completions`) directly, you must format the prompt yourself.

---

## 41.5 Llama Guard: Safety Classification

Meta releases Llama Guard models alongside the main Llama family. Llama Guard
is a fine-tuned Llama model that acts as a safety classifier for LLM inputs
and outputs, categorised against a configurable policy.

```python
from vllm import LLM, SamplingParams

guard = LLM("meta-llama/Llama-Guard-3-8B")

# Classify a user message
classification_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Task: Check if there is unsafe content in the user message according to the safety policy.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes
S2: Non-Violent Crimes
S3: Sex Crimes
...
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: How do I pick a lock?
<END CONVERSATION>

Provide your safety assessment.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

output = guard.generate([classification_prompt],
                         SamplingParams(max_tokens=10))
# Output: "safe" or "unsafe\nS2" (with category)
```

**Inference considerations for Llama Guard:**

- It is a full generative model — it requires a full forward pass per
  classification, unlike a lightweight embedding-based classifier
- Latency: ~20–50ms on an A100 for single input classification
- For high-traffic deployments, Llama Guard runs as a separate service on
  a dedicated GPU, not on the same GPU serving the main model
- Llama Guard 3 8B is the current recommended size — the 1B version is
  significantly less accurate

---

## 41.6 Inference Configuration

### 41.6.1 vLLM recommended configuration

```bash
# Llama 3.1 70B on 2× H100 (tensor parallel)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --max-num-seqs 256 \
    --served-model-name llama-3.1-70b

# Llama 3.3 70B — same config, just change model
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching

# Llama 3.2 3B for edge (single GPU)
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192
```

### 41.6.2 llama.cpp recommended configuration

```bash
# 70B at Q4_K_M (best quality/size tradeoff for 70B)
llama-server \
    --model Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    --ctx-size 32768 \
    --n-gpu-layers 80 \     # all 80 layers to GPU (requires 2× 80GB GPU)
    --threads 8 \
    --batch-size 512 \
    --ubatch-size 128 \
    --port 8080

# 8B at Q4_K_M — runs on a single consumer GPU (RTX 3090/4090)
llama-server \
    --model Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --ctx-size 32768 \
    --n-gpu-layers 33 \     # all 32 layers + embed
    --port 8080

# 3B for edge on Apple Silicon
llama-server \
    --model Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --ctx-size 8192 \
    --n-gpu-layers 99 \     # all to Metal GPU
    --port 8080
```

### 41.6.3 Quantization sweet spots

Based on perplexity evaluation against FP16 baseline:

| Model | Quantization | Size | PPL delta | Recommended use |
|---|---|---|---|---|
| 8B | Q8_0 | 8.5 GB | +0.03 | When quality matters most |
| 8B | Q4_K_M | 4.9 GB | +0.15 | Best quality/size for consumer GPU |
| 8B | Q2_K | 2.9 GB | +0.65 | Edge, Raspberry Pi |
| 70B | Q4_K_M | 43 GB | +0.08 | Dual-consumer or single H100 |
| 70B | Q5_K_M | 51 GB | +0.04 | Single H100 with headroom |
| 70B | IQ4_XS | 40 GB | +0.10 | Tight memory, better than Q4_0 |
| 405B | Q4_K_M | 231 GB | +0.06 | 4× 80GB node |
| 405B | FP8 | 405 GB | +0.01 | 8× H100 data center |

For vLLM (CUDA), use AWQ or FP8 rather than GGUF formats:

```bash
# 70B AWQ (AutoAWQ quantization)
vllm serve casperhansen/llama-3-70b-instruct-awq \
    --quantization awq \
    --tensor-parallel-size 2

# 70B FP8 (H100 recommended)
vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
    --quantization fp8 \
    --tensor-parallel-size 2
```

---

## 41.7 Worked Example 41.1 — Sizing a Llama 3.1 70B Deployment

**Requirements:**
- 500 concurrent users
- P95 TTFT < 2 seconds
- P95 output latency < 5 seconds for 200-token response
- Average context: 1,500 tokens (500 system + 1,000 user)

**Step 1: Memory budget**

Per-request KV cache at 1,500 tokens: `320 KB × 1,500 = 480 MB`
Model weights in FP16: `~140 GB`

On 2× H100 (160 GB total):
- Model weights: 140 GB
- Available for KV cache: 160 × 0.90 - 140 = 4 GB
- Max concurrent requests at 1,500 tokens: `4 GB / 480 MB = 8` ← very tight

Switch to AWQ (4-bit weights): model shrinks to ~35 GB
- Available for KV cache: 144 GB - 35 GB = 109 GB
- Max concurrent requests: `109,000 MB / 480 MB = 227`

Switch to 4× A100 80GB tensor-parallel with AWQ:
- Available: 320 × 0.90 - 35 = 253 GB
- Max concurrent: `253,000 / 480 = 527` — meets the 500-user target

**Step 2: Throughput check**

At 500 concurrent users, each generating 200 tokens:
- Total decode tokens/s needed: 500 × (200 tokens / 5s) = 20,000 tok/s
- 4× A100 AWQ decode throughput (~8,000 tok/s per A100 at batch 128): 32,000 tok/s ✓

**Final configuration:**

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --quantization awq \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --max-num-seqs 512 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90
```

---

## 41.8 Performance Benchmarks

| Hardware | Model | Quant | TTFT (P50) | Decode tok/s (batch 1) | Decode tok/s (batch 32) |
|---|---|---|---|---|---|
| 1× H100 | 8B | FP16 | 48ms | 85 | 2,800 |
| 1× H100 | 8B | FP8 | 41ms | 110 | 3,600 |
| 2× H100 | 70B | FP16 | 95ms | 52 | 1,650 |
| 2× H100 | 70B | FP8 | 78ms | 68 | 2,100 |
| 4× A100 | 70B | AWQ | 125ms | 35 | 1,200 |
| M3 Max (36GB) | 8B | Q4_K_M | 180ms | 38 | — |
| M3 Max (36GB) | 70B | Q4_K_M | 850ms | 8 | — |

---

## 41.9 The Llama Ecosystem

The Llama family has spawned the broadest fine-tuning and derivative ecosystem
of any open model:

**Fine-tuning frameworks explicitly supporting Llama 3:**
`torchtune`, `unsloth`, `axolotl`, `LLaMA-Factory`, `alignment-handbook`

**Notable derivatives:**
- `NousResearch/Meta-Llama-3.1-8B-Instruct` variants (extended fine-tunes)
- `teknium/OpenHermes` derivatives (synthetic data fine-tuning)
- Domain-specific: medical (BioMistral), code (CodeLlama lineage continues)

**Inference engine support matrix:**

| Engine | 8B | 70B | 405B | Vision | Long ctx |
|---|---|---|---|---|---|
| vLLM | ✓ | ✓ | ✓ | ✓ (3.2) | ✓ (128K) |
| llama.cpp | ✓ | ✓ | ✓ | ✓ (3.2) | ✓ |
| SGLang | ✓ | ✓ | ✓ | ✓ | ✓ |
| TensorRT-LLM | ✓ | ✓ | ✓ | partial | ✓ |
| MLC-LLM | ✓ | ✓ | partial | ✗ | partial |
| Ollama | ✓ | ✓ | ✗ | ✓ | limited |

---

## Chapter Summary

Llama 3 is the benchmark, the baseline, and the default. Its architectural
choices — GQA with 8 KV heads, SwiGLU FFN, RoPE with θ = 500,000, the 128K
vocabulary — are individually well-motivated and together produce a model that
is memory-efficient in its KV cache footprint, fast to serve at the right
quantization level, and supported across every major inference engine.

The key numbers to internalise: 320 KB/token/layer for the 70B model's KV
cache; 43 GB for Q4_K_M 70B weights; 4× hardware sizing for 500 concurrent
users at realistic context lengths. Llama 3.3 70B gives you 405B-quality
output at 70B cost — this is the production default for most organisations
running self-hosted inference.

---

## Self-Check Questions

1. A Llama 3.1 70B deployment (2× H100, FP16) serves requests with 4,096-token
   contexts. How many concurrent requests can the KV cache accommodate? What
   changes if you switch to FP8 weights and FP8 KV cache?

2. Llama 3 uses 8 KV heads at all sizes (8B, 70B, 405B). What are the
   memory bandwidth implications for decode throughput? Does a larger model
   with the same KV head count necessarily have lower decode throughput?

3. Explain why the tiktoken tokenizer's 128K vocabulary increases the lm_head
   parameter count. For a 70B model at FP16, what percentage of total model
   memory does the lm_head consume?

4. A system prompt of 2,000 tokens is prepended to all requests. You enable
   prefix caching. The block size is 16 tokens. How many blocks are fully
   cacheable? How many tokens are wasted in the partial last block?

5. Llama 3.3 70B "matches 405B quality on most benchmarks." From an inference
   engineering perspective (not an ML perspective), why is deploying Llama 3.3
   70B preferable to Llama 3.1 405B even if their quality were identical?
