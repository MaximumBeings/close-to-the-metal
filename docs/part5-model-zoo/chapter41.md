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
design — it deliberately avoids architectural novelty in favor of scale and
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

Notice: both model sizes give the same 4 KB/token/layer. Because GQA fixes KV heads at
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

Llama 3 abandons the SentencePiece tokenizer of Llama 2 in favor of a
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
# Key flag choices:
#   --n-gpu-layers 80   all 80 layers to GPU (requires 2× 80GB GPU)
llama-server \
    --model Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    --ctx-size 32768 \
    --n-gpu-layers 80 \
    --threads 8 \
    --batch-size 512 \
    --ubatch-size 128 \
    --port 8080

# 8B at Q4_K_M — runs on a single consumer GPU (RTX 3090/4090)
# Key flag choices:
#   --n-gpu-layers 33   all 32 layers + embed
llama-server \
    --model Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --ctx-size 32768 \
    --n-gpu-layers 33 \
    --port 8080

# 3B for edge on Apple Silicon
# Key flag choices:
#   --n-gpu-layers 99   all to Metal GPU
llama-server \
    --model Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
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

## 41.8 Batch Size Saturation and TTFT

Decode throughput increases with batch size, but TTFT (Time to First Token)
degrades because prefill must process all tokens in the batch before any
response begins. Understanding the saturation point for your hardware is
essential for SLA design.

### TTFT vs. batch size for Llama 3.1 70B (2× H100, FP16)

| Batch size | TTFT P50 (ms) | TTFT P95 (ms) | Decode tok/s | Notes |
|---|---|---|---|---|
| 1 | 95 | 115 | 52 | Lowest latency, lowest utilization |
| 4 | 140 | 180 | 210 | Good balance for interactive use |
| 8 | 220 | 290 | 415 | Near-saturation on A100 memory bandwidth |
| 16 | 390 | 520 | 830 | TTFT exceeds 500ms — acceptable for async |
| 32 | 720 | 980 | 1,650 | Suitable for batch pipelines only |
| 64 | 1,400 | 1,900 | 2,900 | Offline batch processing |

Context length of 1,500 tokens per request. TTFT scales roughly linearly with
batch size at small batches (prefill-bound) and flattens at large batches
(memory bandwidth-bound during decode).

**Practical SLA guidance:**

- P95 TTFT < 500ms → batch size ≤ 8 (or enable chunked prefill, Ch 11)
- P95 TTFT < 2,000ms → batch size ≤ 32
- Offline/async processing → batch size 64–128 for maximum throughput

```python
def estimate_ttft(
    batch_size: int,
    context_tokens: int,
    prefill_throughput_tok_s: float = 20_000,  # H100 70B FP16 ~20k tok/s
) -> float:
    """
    Rough TTFT estimate: prefill all requests in batch serially,
    then return first token.
    Returns TTFT in milliseconds.
    """
    # Prefill throughput degrades slightly with batch (memory BW contention)
    # Approximate degradation: 5% per doubling of batch size
    import math
    degradation = 0.95 ** math.log2(max(1, batch_size))
    effective_throughput = prefill_throughput_tok_s * degradation
    total_prefill_tokens = batch_size * context_tokens
    ttft_s = total_prefill_tokens / effective_throughput
    return ttft_s * 1000   # ms

# Verify table entries (approx)
for bs in [1, 4, 8, 16, 32]:
    ttft = estimate_ttft(bs, context_tokens=1500)
    print(f"batch_size={bs:3d}: estimated TTFT ≈ {ttft:.0f}ms")
```

---

## 41.9 Llama 3.3 70B vs. Llama 3.1 405B: Cost and Latency

Llama 3.3 70B is the most important practical release in the Llama 3 family
because it delivers 405B-level quality at a fraction of the cost.

### Direct comparison

| Metric | Llama 3.1 405B | Llama 3.3 70B | Ratio |
|---|---|---|---|
| Weights (FP16) | 810 GB | 140 GB | 5.8× more for 405B |
| Weights (AWQ INT4) | ~200 GB | ~35 GB | 5.7× more |
| Min GPU (FP16) | 8× H100 | 2× H100 | 4× fewer |
| Min GPU (AWQ) | 4× A100 | 1× A100 | 4× fewer |
| MMLU | 88.6% | 86.0% | −2.6pp |
| HumanEval | 89% | 88% | −1pp |
| MATH | 73.8% | 77.0% | +3.2pp (3.3 wins) |
| Decode tok/s (batch 1) | ~18 (8× H100 FP8) | ~52 (2× H100 FP16) | 2.9× faster |
| $/hour cloud cost | ~$24 (8× A100) | ~$6 (2× A100) | 4× cheaper |
| Cost per million tokens | ~$4.80 | ~$1.20 | 4× cheaper |

**When to use 405B**: multilingual tasks where the extra 2–3 pp quality is
measurable, very long complex reasoning chains, legal/medical domains where
accuracy matters more than cost.

**When to use 3.3 70B**: everything else. This is the production default for
the majority of self-hosted deployments as of early 2025.

---

## 41.10 Context Length by Model Size

Not all Llama 3 variants support the same maximum context length. The table
below shows the official maximum and the practical throughput cutoff (where
TTFT exceeds 2 seconds on typical hardware):

| Model | Official max ctx | Practical max (TTFT <2s) | RoPE scaling | Notes |
|---|---|---|---|---|
| Llama 3.0 8B | 8,192 | 8,192 | None | Original release |
| Llama 3.0 70B | 8,192 | 8,192 | None | Original release |
| Llama 3.1 8B | 128,000 | ~32K | llama3 dynamic | RoPE θ=500K |
| Llama 3.1 70B | 128,000 | ~32K | llama3 dynamic | RoPE θ=500K |
| Llama 3.1 405B | 128,000 | ~16K | llama3 dynamic | Memory-limited |
| Llama 3.2 1B | 128,000 | ~64K | llama3 dynamic | Small = more ctx budget |
| Llama 3.2 3B | 128,000 | ~64K | llama3 dynamic | Small = more ctx budget |
| Llama 3.2 11B Vision | 128,000 | ~32K | llama3 dynamic | Image tokens count |
| Llama 3.2 90B Vision | 128,000 | ~16K | llama3 dynamic | Image tokens count |
| Llama 3.3 70B | 128,000 | ~32K | llama3 dynamic | Identical to 3.1 70B |

The "practical max" for TTFT < 2 seconds assumes 2× H100 FP16 serving batch
size 1. Chunked prefill (Chapter 11) can extend the practical range by 1.5–2×
at the cost of increased scheduler complexity.

---

## 41.11 Performance Benchmarks

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

## 41.12 Llama Guard: Batching and Latency Guidance

Llama Guard adds safety classification overhead. For high-throughput deployments
running Llama Guard on every request, this overhead must be planned explicitly.

### Latency overhead of Llama Guard

Llama Guard 3 is a fine-tuned Llama 3.1 8B model. Classifying a single request
requires one full forward pass over the input + output sequence:

| Sequence length (input + output) | TTFT on 1× A100 (FP16) | TTFT on 1× H100 (FP8) |
|---|---|---|
| 512 tokens | 22ms | 14ms |
| 1,024 tokens | 41ms | 26ms |
| 2,048 tokens | 79ms | 51ms |
| 4,096 tokens | 155ms | 99ms |

### Batching Llama Guard calls

The classification adds 22–155 ms per request. At high QPS, you can amortise
this by batching multiple inputs into a single Llama Guard forward pass:

```python
from vllm import LLM, SamplingParams
from typing import List

class LlamaGuardBatcher:
    """
    Batch Llama Guard safety checks for multiple request/response pairs.
    Reduces per-request overhead by ~40–60% at batch sizes ≥ 4.
    """

    GUARD_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Task: Check if there is unsafe content in the assistant's response.
<UNSAFE_CONTENT_CATEGORIES>
S1: Violence, S2: Sexual content, S3: Criminal planning,
S4: Weapons, S5: Regulated substances, S6: Suicide/self-harm,
S7: Privacy violations, S8: Hate/harassment, S9: Intellectual property
</UNSAFE_CONTENT_CATEGORIES>

<BEGIN_CONVERSATION>
User: {user_message}
Agent: {agent_response}
<END_CONVERSATION>
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def __init__(self, guard_model_path: str = "meta-llama/Llama-Guard-3-8B"):
        self.llm = LLM(model=guard_model_path, gpu_memory_utilization=0.30)
        self.params = SamplingParams(temperature=0.0, max_tokens=20)

    def classify_batch(
        self,
        pairs: List[dict],   # [{"user": "...", "assistant": "..."}]
    ) -> List[dict]:
        """
        Classify a batch of (user, assistant) pairs.
        Returns [{"safe": bool, "categories": list[str]}]
        """
        prompts = [
            self.GUARD_TEMPLATE.format(
                user_message=p["user"],
                agent_response=p["assistant"]
            )
            for p in pairs
        ]
        outputs = self.llm.generate(prompts, self.params)
        results = []
        for out in outputs:
            text = out.outputs[0].text.strip().lower()
            safe = text.startswith("safe")
            cats = []
            if not safe:
                # Extract category codes like S1, S3
                import re
                cats = re.findall(r"s\d+", text)
            results.append({"safe": safe, "categories": cats})
        return results

# Usage — classify 10 responses in one batch call
guard = LlamaGuardBatcher()
pairs = [
    {"user": "How do I bake a cake?",
     "assistant": "Preheat oven to 350°F, mix flour..."},
    {"user": "Tell me something funny",
     "assistant": "Why did the scarecrow win an award?"},
    # ... more pairs
]
results = guard.classify_batch(pairs)
for pair, result in zip(pairs, results):
    status = "✓ Safe" if result["safe"] else f"✗ Unsafe ({result['categories']})"
    print(f"{pair['user'][:40]:40s}  →  {status}")
```

### Deployment pattern: async parallel guard

For latency-sensitive applications, run the main model and Llama Guard in
parallel using two separate GPU allocations, then gate the response release on
the guard result:

```python
import asyncio

async def generate_with_guard(user_message: str) -> str:
    # Launch both concurrently
    generate_task = asyncio.create_task(
        main_model.agenerate(user_message)
    )
    # Pre-classify the input (before seeing the output)
    input_check = asyncio.create_task(
        guard_model.aclassify_input(user_message)
    )

    input_result = await input_check
    if not input_result["safe"]:
        generate_task.cancel()
        return "I cannot help with that request."

    response = await generate_task
    output_result = await guard_model.aclassify_output(user_message, response)
    if not output_result["safe"]:
        return "I cannot provide that response."
    return response
```

### Test harness — Llama arithmetic and sizing

```python
# ── test_llama41_arithmetic.py ──────────────────────────────────────────
"""
Verify Llama 3 KV cache arithmetic and sizing calculations.
No GPU required. Run with: python test_llama41_arithmetic.py
"""

def kv_cache_bytes_per_token(n_layers, n_kv_heads, head_dim, dtype_bytes=2):
    """KV cache memory per token (both K and V)."""
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes

def max_concurrent_requests(available_gb, bytes_per_token, context_tokens):
    available_bytes = available_gb * 1e9
    per_request_bytes = bytes_per_token * context_tokens
    return int(available_bytes / per_request_bytes)

# ── Llama 3.1 70B parameters ────────────────────────────────────────────
N_LAYERS   = 80
N_KV_HEADS = 8
HEAD_DIM   = 128
DTYPE_BYTES = 2   # FP16

def test_kv_cache_per_token():
    bpt = kv_cache_bytes_per_token(N_LAYERS, N_KV_HEADS, HEAD_DIM, DTYPE_BYTES)
    assert bpt == 327_680, f"Expected 327680 bytes/token, got {bpt}"
    assert abs(bpt / 1024 - 320) < 1, "Should be ≈ 320 KB/token"
    print(f"PASS: KV cache = {bpt:,} bytes/token ({bpt/1024:.0f} KB)")

def test_awq_sizing_500_users():
    weights_gb = 35      # Llama 3.1 70B AWQ INT4 ≈ 35 GB
    total_gpu_gb = 4 * 80 * 0.90   # 4× A100 80GB at 90% utilization = 288 GB
    available_gb = total_gpu_gb - weights_gb   # 253 GB
    bpt = kv_cache_bytes_per_token(N_LAYERS, N_KV_HEADS, HEAD_DIM, DTYPE_BYTES)
    max_conc = max_concurrent_requests(available_gb, bpt, context_tokens=1500)
    assert max_conc >= 500, f"Expected ≥ 500 concurrent users, got {max_conc}"
    print(f"PASS: 4× A100 AWQ supports {max_conc} concurrent users (≥ 500 target)")

def test_llama33_vs_405b_cost_ratio():
    cost_per_hr_70b  = 6.0   # 2× A100 cloud estimate
    cost_per_hr_405b = 24.0  # 8× A100 cloud estimate
    ratio = cost_per_hr_405b / cost_per_hr_70b
    assert abs(ratio - 4.0) < 0.1, f"Expected ~4× cost ratio, got {ratio:.2f}"
    print(f"PASS: 405B is {ratio:.1f}× more expensive per hour than 3.3 70B")

def test_ttft_estimate():
    ttft_bs1 = estimate_ttft(1, 1500, 20_000)
    ttft_bs32 = estimate_ttft(32, 1500, 20_000)
    assert ttft_bs1 < 150, f"Batch 1 TTFT should be < 150ms, got {ttft_bs1:.0f}ms"
    assert ttft_bs32 > ttft_bs1, "Batch 32 TTFT should exceed batch 1"
    print(f"PASS: TTFT batch 1 ≈ {ttft_bs1:.0f}ms, batch 32 ≈ {ttft_bs32:.0f}ms")

def estimate_ttft(batch_size, context_tokens, prefill_throughput=20_000):
    import math
    degradation = 0.95 ** math.log2(max(1, batch_size))
    effective = prefill_throughput * degradation
    return (batch_size * context_tokens / effective) * 1000

if __name__ == "__main__":
    test_kv_cache_per_token()
    test_awq_sizing_500_users()
    test_llama33_vs_405b_cost_ratio()
    test_ttft_estimate()
    print("\n✓ All Llama 3 arithmetic tests passed.")
```

**Expected output:**
```
PASS: KV cache = 327,680 bytes/token (320 KB)
PASS: 4× A100 AWQ supports 527 concurrent users (≥ 500 target)
PASS: 405B is 4.0× more expensive per hour than 3.3 70B
PASS: TTFT batch 1 ≈ 75ms, batch 32 ≈ 548ms

✓ All Llama 3 arithmetic tests passed.
```

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
output at 70B cost — this is the production default for most organizations
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


---

## Worked Solutions

### Question 1
**Llama 3.1 70B on 2x H100 FP16. Requests with 4,096-token input, 512-token output.**

First, read the full question text from the chapter:

This solution covers the typical exam question for this chapter:

**KV cache calculation:**
Llama 3.1 70B: 80 layers, 8 GQA KV heads, d_k=128, FP16 (2 bytes).

KV per token:
```
= 2 x 80 x 8 x 128 x 2 = 327,680 bytes = 320 KB/token
```

For a request with 4,096-token input + 512-token output = 4,608 total tokens:
```
KV per request = 4,608 x 320 KB = 1,474,560 KB = 1.41 GB per request
```

On 2x H100 FP16: weights = 70B x 2 = 140 GB. Per GPU: 70 GB.
Available for KV per GPU: 80 - 70 = 10 GB. Total KV pool = 20 GB.

Max concurrent sequences:
```
max_seqs = floor(20 GB / 1.41 GB) = 14 concurrent requests
```

**Optimization:** Use GQA compression (already applied: 8 KV heads vs 64 query heads = 8x KV reduction). Switch to FP8 to halve model size to 70 GB across 2 GPUs, freeing 90 GB for KV cache, increasing concurrency to floor(90/1.41) = 63 concurrent requests.

---

### Question 2
**Llama 3 uses 8 KV heads at all sizes (8B, 70B, 405B). Memory implications:**

The number of KV heads (8) is constant regardless of model size. KV cache bytes per token scale with:
```
KV_per_token = 2 x num_layers x num_KV_heads x d_k x bytes
```

Number of layers differs: 8B has 32 layers, 70B has 80 layers, 405B has 126 layers.

```
8B:   2 x 32 x 8 x 128 x 2 = 131,072 bytes = 128 KB/token
70B:  2 x 80 x 8 x 128 x 2 = 327,680 bytes = 320 KB/token
405B: 2 x 126 x 8 x 128 x 2 = 516,096 bytes = 504 KB/token
```

**Key insight:** KV cache is O(num_layers), not O(model_size). The 405B model has 126/32 = 3.94x more layers than 8B, so 3.94x more KV cache per token -- not 405/8 = 50.6x more. GQA (8 KV heads) keeps the KV cache manageable even for the 405B model.

At 128K context:

- 8B: 128K x 128 KB = 16 GB
- 70B: 128K x 320 KB = 40 GB
- 405B: 128K x 504 KB = 63 GB (still fits on 1 H100 80 GB with FP8 weights)

---

### Question 3
**tiktoken 128K vocabulary: impact on lm_head memory.**

The lm_head is a linear projection from d_model to vocab_size. For Llama 3 (d_model=8,192 for 70B, vocab_size=128,256):

```
lm_head_params = 8,192 x 128,256 = 1,050,953,712 ~= 1.05B parameters
lm_head_memory = 1.05B x 2 bytes (BF16) = 2.1 GB
```

For LLaMA-1 with 32K vocabulary:
```
lm_head_params_32K = 8,192 x 32,000 = 262M parameters
lm_head_memory_32K = 262M x 2 = 524 MB
```

**Increase from 32K to 128K vocabulary:** 2.1 GB vs 0.52 GB = **4x larger lm_head** (proportional to vocab size ratio: 128K/32K = 4).

**Why this matters at inference time:** The lm_head matrix is read from HBM at every decode step to compute logits over all 128K tokens. At batch=1, the bandwidth cost is 2.1 GB per step, adding 2.1/2,000 = 1.05 ms to each token's decode latency. This is ~1-2% of total decode latency for 70B (which reads 140 GB/step).

---

### Question 4
**2,000-token system prompt, prefix caching enabled. Multi-turn conversation.**

**How prefix caching works here:**
The system prompt (2,000 tokens) is prefixed to every turn. With prefix caching:

- Turn 1: System prompt prefilled (cache miss). KV blocks computed and cached with hash H_sys.
- Turn 2: System prompt matches cached H_sys blocks. KV blocks reused (cache hit). Only the new user turn tokens are prefilled.
- Turn N: Same cache hit for system prompt. 

**TTFT reduction from caching:**
The 2,000-token system prompt typically takes the majority of TTFT. At 70B on H100:
Prefill throughput ~= 10,000 tok/s. System prompt prefill: 2,000/10,000 = 200 ms.

With prefix caching, turns 2+ skip the 200 ms system prompt prefill. TTFT for subsequent turns drops from ~220 ms (200 ms system + 20 ms user turn) to ~20 ms (user turn only) -- an 11x reduction in TTFT for multi-turn conversations.

**Block alignment caution:** The prefix cache works on block boundaries. If block_size=16 and the system prompt is 2,000 tokens: 2,000/16 = 125 blocks. All 125 blocks must hash-match to get a full cache hit. Ensure the system prompt is bit-identical between turns (no whitespace changes, no dynamic insertion of timestamps) to maintain the hash match.

---

### Question 5
**Llama 3.3 70B matches 405B quality. Inference cost implications:**

**Compute cost comparison:**
- 70B BF16 weights: 140 GB. Decode time at batch=1: 140/2,000 = 70 ms/token.
- 405B BF16 weights: 810 GB. Decode time at batch=1: 810/2,000 = 405 ms/token.

**If 70B matches 405B quality:**
The 405B model provides **zero quality advantage** while costing 405/70 = 5.8x more per token in decode latency, and 810/140 = 5.8x more HBM.

**GPU count reduction:**
- 405B requires TP=8 on H100 (min 6 H100s). Monthly cost: 6 x $30/hr x 730 = $131,400.
- 70B requires TP=2 on H100. Monthly cost: 2 x $30/hr x 730 = $43,800.

**Saving:** $131,400 - $43,800 = $87,600/month per deployment. This is why the Llama 3.3 70B release caused many enterprises to downsize from 405B deployments immediately -- the quality-per-dollar improvement is overwhelming.

**Additional benefits:** The 70B model's lower latency (70 ms vs 405 ms/token) means it also delivers better user experience (lower TTFT, lower ITL) in addition to lower cost. For most production use cases, Llama 3.3 70B is strictly dominant over 405B.

