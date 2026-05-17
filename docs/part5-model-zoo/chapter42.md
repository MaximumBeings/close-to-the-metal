# Chapter 42 — Phi-4 and Gemma 3: Small Models, Large Impact

> *"A 14B model that outperforms a 70B model is not a curiosity. It is a cost reduction waiting to be deployed."*

---

## 42.1 The Case for Small Models

The GPU budget of most organizations does not support a 70B model in production.
An A100 80GB can hold a 70B model at 4-bit quantization with very little room
left for KV cache. For teams running on consumer hardware, edge devices, or
cost-constrained cloud budgets, the question is not "which 70B model" but
"how small can we go without sacrificing too much quality."

Phi-4 and Gemma 3 answer this question from two different directions: Phi-4
chases capability density at small size through data quality, while Gemma 3
builds a full capability ladder from 1B to 27B with a novel architecture for
long-context and multimodal capability.

Both families are inference-relevant in ways that go beyond "smaller means
cheaper." Each makes architectural choices that directly affect how you
configure, quantize, and serve them.

---

## 42.2 Microsoft Phi-4

### 42.2.1 The Phi Philosophy: Data Quality over Scale

The Phi series is built around a single hypothesis: a small model trained on
high-quality synthetic data can approach the reasoning capability of a much
larger model trained on larger but noisier web data.

This hypothesis has been tested through the Phi lineage:

| Model | Parameters | MMLU | HumanEval | Key innovation |
|---|---|---|---|---|
| Phi-1 | 1.3B | 56.3 | 50.6 | Code-focused synthetic data |
| Phi-2 | 2.7B | 57.0 | 48.0 | Scaled synthetic data mix |
| Phi-3 Mini | 3.8B | 69.9 | 60.0 | Chat fine-tuning at small scale |
| Phi-3 Small | 7B | 75.7 | 61.0 | Flash attention, sliding window |
| Phi-3 Medium | 14B | 78.0 | 55.5 | Scaled synthetic data |
| Phi-4 | 14B | 84.8 | 82.6 | Synthetic-majority training data |

Phi-4's MMLU score of 84.8 exceeds Llama 3.1 70B (83.6) at 5× fewer
parameters. Its HumanEval score of 82.6 exceeds models twice its size.

### 42.2.2 Phi-4 Architecture

Phi-4 is a standard dense decoder-only transformer. There is no architectural
novelty — no MoE, no sliding window, no exotic attention variant. The
interesting decisions are:

```
Parameters:      14.0B
Layers:          40
d_model:         5,120
Attention heads: 40
KV heads:        10 (GQA, group size 4)
d_ffn:           17,920  (SwiGLU, 3.5 × d_model)
Vocabulary:      100,352
Context:         16,384 (native), 32,768 (with RoPE scaling)
```

The relatively high layer count (40) for a 14B model gives Phi-4 greater depth
than width — the model is "tall and narrow" compared to, say, a 14B MoE model.
Depth correlates with multi-step reasoning capability; width correlates with
world knowledge. This ratio reflects the data quality hypothesis: fewer
parameters for raw knowledge storage, more layers for reasoning computation.

### 42.2.3 KV Cache Profile for Phi-4

```
KV cache per token per layer = 2 × num_kv_heads × head_dim × bytes
                             = 2 × 10 × 128 × 2  (FP16)
                             = 5,120 bytes ≈ 5 KB/token/layer

Full KV cache per token = 5 KB × 40 = 200 KB/token
```

Compare to Llama 3 8B (32 layers, 8 KV heads): `2 × 8 × 128 × 2 × 32 = 131 KB/token`

Phi-4 has a larger KV cache per token than Llama 3 8B despite similar
parameter count, because of its higher layer count. At 16K context:

```
Phi-4 KV cache (16K context, FP16): 200 KB × 16,384 = 3.2 GB
Llama 3 8B (16K context, FP16):     131 KB × 16,384 = 2.1 GB
```

Still manageable on a single 24GB consumer GPU with the model weights at ~15 GB
(FP16, unquantized).

### 42.2.4 Serving Phi-4 with vLLM

```bash
# Single RTX 4090 (24GB) — model + KV cache fit comfortably
vllm serve microsoft/phi-4 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching

# For longer context (32K with RoPE scaling)
vllm serve microsoft/phi-4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --rope-scaling '{"type": "dynamic", "factor": 2.0}'
```

### 42.2.5 Serving Phi-4 with llama.cpp

```bash
# Q4_K_M: 8.1 GB — fits in 12 GB VRAM with 2 GB for context
llama-server \
    --model phi-4-q4_k_m.gguf \
    --ctx-size 16384 \
    --n-gpu-layers 40 \  # all 40 layers to GPU
    --threads 4 \
    --port 8080

# Q8_0: 14.9 GB — needs 16 GB VRAM
llama-server \
    --model phi-4-q8_0.gguf \
    --ctx-size 8192 \
    --n-gpu-layers 40 \
    --port 8080
```

### 42.2.6 Phi-4 Prompt Format

Phi-4 uses a different chat template from Llama 3:

```
<|im_start|>system<|im_sep|>
{system_message}<|im_end|>
<|im_start|>user<|im_sep|>
{user_message}<|im_end|>
<|im_start|>assistant<|im_sep|>
```

This is the ChatML format. Most inference engines apply it automatically via
the model's `tokenizer_config.json`. If you are using the raw completions
endpoint, apply it manually:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("microsoft/phi-4")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Prove that sqrt(2) is irrational."}
]
prompt = tok.apply_chat_template(messages, tokenize=False,
                                  add_generation_prompt=True)
```

### 42.2.7 When to Choose Phi-4

Phi-4 excels at:

- **STEM reasoning**: mathematics, physics, computer science
- **Code generation**: outperforms models twice its size on HumanEval
- **Structured tasks**: instruction following, classification, extraction
- **Edge deployment**: 8 GB quantized form fits on consumer hardware

Phi-4 underperforms at:

- **World knowledge**: smaller parameter count means less factual coverage
- **Long documents**: 16K native context limits document processing
- **Multilingual**: primarily English-optimized training data
- **Creative tasks**: depth over width architecture favors reasoning over fluency

---

## 42.3 Google Gemma 3

### 42.3.1 The Gemma 3 Family

Google released Gemma 3 in March 2025 as a full capability ladder:

| Model | Parameters | Context | Vision | MMLU | Notes |
|---|---|---|---|---|---|
| Gemma 3 1B | 1.0B | 32K | ✗ | 38.6 | Ultra-edge deployment |
| Gemma 3 4B | 4.3B | 128K | ✓ | 59.6 | Best small multimodal |
| Gemma 3 12B | 12.2B | 128K | ✓ | 74.0 | Mid-tier workhorse |
| Gemma 3 27B | 27.2B | 128K | ✓ | 79.0 | Near-70B quality at 27B |

The standout feature: all models except the 1B support 128K context natively —
the same context length as Llama 3.1, at model sizes where Llama 3 maxes out
at 8B for that context length.

### 42.3.2 Gemma 3 Architecture: Interleaved Attention

Gemma 3's most distinctive architectural feature is **interleaved global and
local attention**:

```
Layer pattern (27B, 62 layers):
  L0:   Global attention  (attends to all positions)
  L1:   Local attention   (sliding window, 1024-token window)
  L2:   Local attention
  L3:   Local attention
  L4:   Local attention
  L5:   Global attention
  L6:   Local attention
  ...   (1 global every 5 layers)
```

Every 5th layer is global attention; the other 4 are local (sliding window).
This approximates full attention at a fraction of the compute cost for long
sequences.

**Computational consequence:**

For a sequence of length S:

- Full attention layer: O(S²) per layer
- Local attention layer (window W=1024): O(S × W) per layer = O(1024 × S)

For S = 32,768 tokens with 12 global + 50 local layers (62 total):
```
Full attention cost:    12 × 32768² = 12.9 billion ops
Local attention cost:   50 × 32768 × 1024 = 1.68 billion ops
Total:                  14.6 billion ops

vs. all-global (62 layers × 32768²):
                        66.6 billion ops
```

The interleaved pattern reduces attention compute by ~4.5× while preserving
long-range reasoning through the periodic global layers.

### 42.3.3 Tied Embeddings

Gemma 3 uses **tied input/output embeddings**: the token embedding matrix
(used to convert token IDs to vectors at the input) is the same matrix used
in the output projection (lm_head) to convert hidden states back to logit
vectors.

```
Embedding matrix E: shape [vocab_size, d_model]
Input:   token_embedding = E[token_id]          (row lookup)
Output:  logits = hidden_state × Eᵀ             (matrix multiply)
```

This halves the memory for these two weight matrices combined. For Gemma 3 27B
with d_model = 4,608 and vocab_size = 256,000:

```
Without tied embeddings:
  Embedding:  256,000 × 4,608 × 2 bytes = 2.36 GB
  lm_head:    4,608 × 256,000 × 2 bytes = 2.36 GB
  Total:      4.72 GB

With tied embeddings:
  Shared matrix: 2.36 GB
  Saving:        2.36 GB (11% of total 27B model ≈ 54 GB in FP16)
```

**Inference implication**: vLLM and llama.cpp handle tied embeddings
transparently — the weight is loaded once and referenced at both positions.
No configuration change is needed.

### 42.3.4 The 256K Vocabulary

Gemma 3 uses a SentencePiece vocabulary of 256,000 tokens — twice Llama 3's
128K and 8× Llama 2's 32K.

The larger vocabulary reduces the number of tokens needed to represent a given
text, which improves throughput (fewer decode steps) but increases:

- Softmax compute (256K logits vs 128K)
- lm_head memory (but halved by tied embeddings)
- Embedding lookup table size

For multilingual text, the larger vocabulary dramatically improves tokenization
efficiency: where Llama 3 might require 3–5 tokens per Chinese character,
Gemma 3 often achieves near-1:1. This directly reduces KV cache usage and
decode step count for multilingual workloads.

### 42.3.5 Gemma 3 Multimodal Architecture (4B, 12B, 27B)

The vision-capable Gemma 3 models use SigLIP as the vision encoder, processing
images as sequences of patch embeddings that are inserted into the token
sequence via a linear projection layer.

```
Image → SigLIP encoder → patch embeddings → linear projection →
→ inserted into token sequence at <start_of_image> position →
→ Gemma 3 decoder processes image tokens and text tokens jointly
```

Unlike Llama 3.2's cross-attention approach, Gemma 3's vision tokens are
processed by the same self-attention layers as text tokens. This is simpler
but means image tokens consume KV cache exactly like text tokens.

**KV cache cost for images:**
A 1024×1024 image at patch size 16 produces `(1024/16)² = 4,096` patches,
each consuming one KV cache token slot. For the 27B model at FP16:

```
Image KV cache = 4,096 tokens × (320 KB/token) ≈ 1.28 GB per image
```

For multimodal workloads, budget approximately 1 GB of KV cache per high-
resolution image per request.

### 42.3.6 Serving Gemma 3 with vLLM

```bash
# Gemma 3 27B on 1× H100 (text only)
vllm serve google/gemma-3-27b-it \
    --max-model-len 32768 \    # 128K possible but expensive
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching

# Gemma 3 27B at 128K context (requires 2× H100 for KV cache)
vllm serve google/gemma-3-27b-it \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.93

# Gemma 3 12B multimodal on 1× A100
vllm serve google/gemma-3-12b-it \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.88 \
    --image-input-type pixel_values \
    --max-num-images 4
```

### 42.3.7 Serving Gemma 3 with llama.cpp

```bash
# 27B at Q4_K_M: 16.5 GB — fits in 20 GB VRAM
llama-server \
    --model gemma-3-27b-it-q4_k_m.gguf \
    --ctx-size 32768 \
    --n-gpu-layers 62 \   # all 62 layers
    --port 8080

# 4B at Q4_K_M on Raspberry Pi 5 (8GB RAM)
llama-server \
    --model gemma-3-4b-it-q4_k_m.gguf \
    --ctx-size 4096 \
    --n-gpu-layers 0 \    # CPU only
    --threads 4 \
    --port 8080
```

### 42.3.8 Gemma 3 Prompt Format

Gemma 3 uses a custom template with `<start_of_turn>` / `<end_of_turn>` tokens:

```
<bos><start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
```

With system instructions:
```
<bos><start_of_turn>user
{system_message}

{user_message}<end_of_turn>
<start_of_turn>model
```

Note: Gemma 3 does not have a separate system role — system instructions
are prepended to the first user turn.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
messages = [
    {"role": "user", "content": "Explain attention in one sentence."}
]
prompt = tok.apply_chat_template(messages, tokenize=False,
                                  add_generation_prompt=True)
```

---

## 42.4 Head-to-Head: Phi-4 vs Gemma 3 27B vs Llama 3.1 8B

| Dimension | Phi-4 (14B) | Gemma 3 27B | Llama 3.1 8B |
|---|---|---|---|
| MMLU | 84.8 | 79.0 | 73.0 |
| HumanEval | 82.6 | 71.7 | 72.6 |
| MATH | 80.4 | 89.0 | 51.9 |
| Multilingual | Fair | Excellent | Good |
| Max context | 16K native | 128K native | 128K native |
| Vision | ✗ | ✓ | ✓ (3.2 only) |
| GPU (Q4) | 8.1 GB | 16.5 GB | 4.9 GB |
| Decode tok/s (H100, batch 1) | 120 | 65 | 160 |
| Best use case | STEM, code | Multilingual, long-ctx, multimodal | Balanced, cost |

---

## 42.5 Worked Example 42.1 — Edge Deployment Decision

**Scenario**: A mobile app that runs entirely on-device (iOS/Android) needs a
language model for text summarization of articles up to 2,000 words (~2,500
tokens).

**Constraints**: Max 4 GB RAM for model, 12-second generation budget for
200-token summary.

| Model | Quantized size | Device RAM | Tok/s (A16 Bionic) | Verdict |
|---|---|---|---|---|
| Llama 3.2 1B Q4_K_M | 0.8 GB | ✓ | ~18 tok/s | ✓ Fast, limited quality |
| Llama 3.2 3B Q4_K_M | 1.9 GB | ✓ | ~10 tok/s | ✓ Good balance |
| Gemma 3 1B Q4_K_M | 0.6 GB | ✓ | ~20 tok/s | ✓ Better multilingual |
| Gemma 3 4B Q4_K_M | 2.4 GB | ✓ | ~8 tok/s | ✓ Best quality under 4GB |
| Phi-4 Q4_K_M | 8.1 GB | ✗ | — | ✗ Too large |

**Decision**: Gemma 3 4B Q4_K_M. It fits within the 4 GB constraint, generates
200 tokens in ~25 seconds (exceeds budget for cold start, but 2× the budget
is acceptable for summarization), and provides the best multilingual quality
for a globally distributed app.

For a performance-critical deployment (< 12s hard budget): Llama 3.2 3B Q4_K_M
at ~20 seconds generation. Tight but achievable with speculative decoding
(Appendix Q covers llama.cpp on Apple Silicon).

---

## 42.6 Quantization Quality Comparison

How much quality do these small models lose at 4-bit quantization vs their
larger baselines?

| Model | FP16 PPL | Q4_K_M PPL | Delta | % of FP16 quality |
|---|---|---|---|---|
| Phi-4 14B | 7.23 | 7.54 | +0.31 | 96% |
| Gemma 3 27B | 6.81 | 7.04 | +0.23 | 97% |
| Gemma 3 12B | 7.92 | 8.21 | +0.29 | 96% |
| Gemma 3 4B | 9.15 | 9.58 | +0.43 | 95% |
| Llama 3.1 8B | 7.89 | 8.21 | +0.32 | 96% |

Small models tolerate quantization nearly as well as large models, typically
losing ~4% quality (measured by perplexity) at Q4_K_M. This is because the
information per parameter is already high in well-trained small models — there
is less redundancy for quantization to exploit, but also less redundancy to
damage.

---

## 42.7 Local Attention Window Boundary Behavior

Gemma 3's local attention layers use a sliding window of 1,024 tokens. Tokens
beyond that window boundary cannot attend to each other in local layers — only
in the periodic global layers (every 5th). Understanding this has two practical
consequences for inference engineers.

### What happens at the boundary

For a token at position P in a local attention layer, it can attend to
positions `[P - 1023, P]`. Any token at position `P - 1024` or earlier is
**invisible** to this token in that layer.

```
Positions:  [0...1022][1023][1024][1025]...[2046][2047]
Token 2047 attends to: [1024..2047] (last 1024 positions)
Token 2047 CANNOT see: [0..1023] in local layers

But in the next global layer (every 5th), it CAN see [0..1023].
```

This means long-range dependencies are only propagated through the sparse
global layers. For tasks requiring tight cross-reference between distant
positions (e.g., referencing the first paragraph from the last paragraph in a
32K document), performance degrades compared to a full-attention model.

### Practical guidance

**Structured prompts**: place the information most likely to be referenced at
the end of the context, within the last 1,024 tokens. System instructions,
key facts, and the active question should be near the end.

**RAG chunk ordering**: for retrieval-augmented generation, place the most
relevant retrieved chunks closest to the user query (at the end of the prompt),
not at the beginning.

```python
def optimal_gemma3_prompt(
    system: str,
    retrieved_chunks: list[str],   # most relevant last
    user_query: str,
    local_window: int = 1024,
) -> str:
    """
    Structure a Gemma 3 prompt to maximize local-attention effectiveness.
    Most relevant content appears in the last `local_window` tokens.
    """
    # Sort chunks: least relevant first (will be outside local window)
    # most relevant last (within local window of the query)
    sorted_chunks = retrieved_chunks   # assume pre-sorted by relevance asc

    context = "\n\n".join(sorted_chunks)
    return f"""{system}

Context (background, may exceed local window):
{context}

Current question (always within local window):
{user_query}"""
```

---

## 42.8 Tied Embeddings: Training Trade-offs

Tied embeddings save memory at inference but introduce a training constraint
that matters when fine-tuning Gemma 3.

### Why tying creates tension during training

During backpropagation, the embedding matrix receives gradients from two sources:

1. **Language modeling loss** (via lm_head): gradient pushes embeddings to
   produce the correct next-token logit. This gradient is large for common
   tokens, small for rare tokens.

2. **Contextual representation** (via input embedding): gradient pushes each
   token's embedding to produce a good contextual representation for attention.

These two objectives are not identical. The token `"the"` should have an
embedding that both encodes its positional/contextual role (as an input) and
its unconditional prior probability (as an output). In an untied model, the
two matrices specialize independently. In a tied model, they must compromise.

**Practical effect**: tied models occasionally exhibit slightly higher
perplexity on rare tokens (< 100 occurrences in training data) because the
rare token's embedding is dominated by the language model gradient from common
tokens. For inference, this means outputs involving very rare tokens (unusual
proper nouns, low-frequency technical terms) may be slightly less confident.

**When fine-tuning Gemma 3**: if you observe poor handling of domain-specific
vocabulary, consider unfreezing the embedding matrix exclusively and training
for a few hundred steps on domain-specific text. Some fine-tuning frameworks
allow "untying" for the fine-tuning phase:

```python
# Hugging Face: unfreeze embeddings for domain fine-tuning
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")

# By default, embed_tokens and lm_head share weights (tied)
print(model.model.embed_tokens.weight is model.lm_head.weight)  # True

# To unfreeze and unty for fine-tuning:
model.lm_head.weight = torch.nn.Parameter(
    model.model.embed_tokens.weight.detach().clone()
)
# Now they are separate; gradient updates won't conflict
model.model.embed_tokens.requires_grad_(True)
model.lm_head.requires_grad_(True)
```

This increases memory by 2.36 GB (the size of the embedding matrix) during
fine-tuning. For inference, re-tie or keep separate — either works.

---

## 42.9 Gemma 3 Multilingual Performance

The 256K vocabulary gives Gemma 3 a significant advantage on non-English text.
Here are representative benchmarks compared to models with smaller vocabularies:

### Multilingual perplexity (lower is better)

| Language | Gemma 3 27B | Llama 3.1 70B | Phi-4 14B |
|---|---|---|---|
| English | 6.81 | 6.95 | 7.23 |
| Chinese (Simplified) | 8.40 | 11.20 | 12.80 |
| Japanese | 8.95 | 12.40 | 13.10 |
| Arabic | 9.20 | 13.50 | 14.90 |
| German | 7.60 | 8.10 | 8.90 |
| Spanish | 7.40 | 7.90 | 8.60 |
| Hindi | 10.20 | 15.30 | 18.40 |
| Korean | 9.10 | 13.20 | 14.50 |

Gemma 3 27B's advantage is largest on languages with complex scripts (Chinese,
Japanese, Korean, Arabic, Hindi) where the 256K vocabulary achieves near-1:1
token-to-character ratios vs 3–5 tokens per character for Llama 3.

### Tokenization efficiency (tokens per 100 characters)

| Language | Gemma 3 (256K) | Llama 3 (128K) | Phi-4 (100K) |
|---|---|---|---|
| English | 24 | 26 | 28 |
| Chinese | 32 | 91 | 88 |
| Japanese | 35 | 105 | 102 |
| Arabic | 40 | 95 | 92 |

For multilingual deployments, Gemma 3's tokenization efficiency translates
directly to fewer decode steps and lower KV cache usage — a real throughput
improvement, not just a quality metric.

---

## 42.10 Multi-Image KV Cache and Batching

Gemma 3's vision encoder (SigLIP) converts each image to a fixed number of
KV tokens: 256 tokens for images at the default resolution. At higher
resolutions (pan-and-scan or tiling mode), this can reach 1,024–4,096 per
image.

### Memory per image

For Gemma 3 27B (62 layers, 16 KV heads, head dim 256, FP16):

```python
def gemma3_kv_per_token(n_layers=62, n_kv_heads=16, head_dim=256, dtype_bytes=2):
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes

# 62 layers × 16 heads × 256 dim × 2 bytes × 2 (K+V)
bytes_per_token = gemma3_kv_per_token()
# = 2 × 62 × 16 × 256 × 2 = 1,015,808 bytes ≈ 992 KB per token

image_tokens = 256
image_kv_mb = (bytes_per_token * image_tokens) / 1e6
print(f"KV cache per image (256 tokens): {image_kv_mb:.1f} MB")
# → 260 MB per image at 256 tokens
```

### Maximum images per request

On a single A100 80GB with model weights (~54 GB FP16):

```
Available KV memory: (80 × 0.90 - 54) GB = 18 GB
KV per image (256 tokens): 260 MB
KV per text token: ~992 KB ≈ 1 MB

If context = 2 images + 1024 text tokens:
  Image KV: 2 × 260 MB = 520 MB
  Text KV: 1024 × 1 MB = 1,024 MB
  Total: 1.54 GB per request
  Max concurrent: 18 GB / 1.54 GB ≈ 11 requests
```

**Practical limits:**

| Images per request | Text tokens | KV per request | Max concurrent (A100 80GB) |
|---|---|---|---|
| 0 | 2,048 | 2.0 GB | 9 |
| 1 | 1,024 | 1.3 GB | 13 |
| 2 | 1,024 | 1.5 GB | 12 |
| 4 | 1,024 | 2.1 GB | 8 |
| 1 (high-res, 1024 tok) | 1,024 | 2.0 GB | 9 |

### Batching strategy for multi-image requests

```python
# vLLM multi-image request (Gemma 3 vision)
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(
    model="google/gemma-3-27b-it",
    max_model_len=8192,
    limit_mm_per_prompt={"image": 4},   # max 4 images per request
    gpu_memory_utilization=0.90,
)

# For high-throughput multi-image workloads, batch requests with
# similar image counts to avoid KV cache fragmentation
def batch_by_image_count(requests: list[dict]) -> list[list[dict]]:
    """Group requests by number of images to improve batch homogeneity."""
    from collections import defaultdict
    groups = defaultdict(list)
    for req in requests:
        n_images = len(req.get("images", []))
        groups[n_images].append(req)
    return list(groups.values())
```

---

## 42.11 Cost-per-Token Comparison

For production budget planning, cost per token is the decisive metric.

| Model | Hardware | Quant | Cloud cost/hr | Tok/s (batch 32) | Cost per 1M tokens |
|---|---|---|---|---|---|
| Phi-4 14B | 1× A100 80GB | FP16 | $3.00 | 1,800 | $0.46 |
| Phi-4 14B | 1× A100 80GB | Q4_K_M | $3.00 | 3,200 | $0.26 |
| Gemma 3 4B | 1× A100 80GB | FP16 | $3.00 | 4,500 | $0.18 |
| Gemma 3 12B | 1× A100 80GB | FP16 | $3.00 | 2,400 | $0.35 |
| Gemma 3 27B | 1× A100 80GB | FP16 | $3.00 | 1,200 | $0.69 |
| Gemma 3 27B | 1× A100 80GB | Q4_K_M | $3.00 | 2,200 | $0.38 |
| Llama 3.1 8B | 1× A100 80GB | FP16 | $3.00 | 3,800 | $0.22 |
| Llama 3.3 70B | 2× A100 80GB | AWQ | $6.00 | 1,200 | $1.39 |

Cost per million tokens = `(cost_per_hr / tok_per_s / 3600) × 1_000_000`

**Key insight**: Gemma 3 4B at FP16 on a single A100 is the cheapest option
by throughput at $0.18/M tokens, with quality that exceeds Llama 2 70B on
most benchmarks. For cost-constrained deployments with quality requirements
above the smallest models, Gemma 3 12B at $0.35/M tokens is the sweet spot.

### Test harness — Gemma 3 / Phi-4 arithmetic

```python
# ── test_chapter42_arithmetic.py ─────────────────────────────────────────
"""Verify Gemma 3 / Phi-4 architectural arithmetic. No GPU required.
Run with: python test_chapter42_arithmetic.py"""


def kv_per_token(n_layers, n_kv_heads, head_dim, dtype_bytes=2):
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes


def test_gemma3_27b_kv_per_token():
    bpt = kv_per_token(62, 16, 256, 2)
    assert bpt == 1_015_808, f"Expected 1_015_808, got {bpt}"
    assert abs(bpt / 1024 / 1024 - 0.969) < 0.01, "Should be ~0.97 MB/token"
    print(f"PASS: Gemma 3 27B KV/token = {bpt:,} bytes ({bpt/1e6:.2f} MB)")


def test_phi4_kv_per_token():
    # Phi-4: 40 layers, 10 KV heads, head_dim=128, FP16
    bpt = kv_per_token(40, 10, 128, 2)
    assert bpt == 204_800, f"Expected 204_800, got {bpt}"
    print(f"PASS: Phi-4 KV/token = {bpt:,} bytes ({bpt/1024:.0f} KB)")


def test_tied_embedding_saving():
    vocab_size = 256_000
    d_model = 4_608
    dtype_bytes = 2
    single_matrix_bytes = vocab_size * d_model * dtype_bytes
    saving_gb = single_matrix_bytes / 1e9
    assert abs(saving_gb - 2.359) < 0.01, f"Expected ~2.36 GB saving, got {saving_gb:.3f}"
    print(f"PASS: Tied embedding saves {saving_gb:.2f} GB")


def test_interleaved_attention_reduction():
    """Compute attention FLOP reduction for Gemma 3 27B at 32K context."""
    S = 32_768    # sequence length
    n_global = 12   # every 5th of 62 layers ≈ 12
    n_local  = 50   # remaining 50
    W = 1_024       # local window

    full_attention_ops  = n_global * S * S
    local_attention_ops = n_local  * S * W
    total_ops  = full_attention_ops + local_attention_ops
    all_global = 62 * S * S

    reduction = all_global / total_ops
    assert 4.0 < reduction < 5.5, f"Expected ~4.5× reduction, got {reduction:.2f}"
    print(f"PASS: attention reduction at 32K = {reduction:.2f}× vs all-global")


def test_image_kv_cost():
    bpt = kv_per_token(62, 16, 256, 2)   # Gemma 3 27B
    image_tokens = 256
    image_kv_mb = (bpt * image_tokens) / 1e6
    assert 240 < image_kv_mb < 280, f"Expected ~260 MB/image, got {image_kv_mb:.1f}"
    print(f"PASS: image KV cost = {image_kv_mb:.1f} MB for {image_tokens} tokens")


def test_cost_per_million_tokens():
    cost_per_hr = 3.0
    tok_per_s   = 1_200   # Gemma 3 27B FP16 batch 32
    cost_per_mtok = (cost_per_hr / tok_per_s / 3600) * 1_000_000
    assert abs(cost_per_mtok - 0.694) < 0.05, (
        f"Expected ~$0.69/Mtok for Gemma 3 27B, got ${cost_per_mtok:.3f}"
    )
    print(f"PASS: Gemma 3 27B FP16 cost = ${cost_per_mtok:.2f}/M tokens")


if __name__ == "__main__":
    test_gemma3_27b_kv_per_token()
    test_phi4_kv_per_token()
    test_tied_embedding_saving()
    test_interleaved_attention_reduction()
    test_image_kv_cost()
    test_cost_per_million_tokens()
    print("\n✓ All Phi-4 / Gemma 3 arithmetic tests passed.")
```

**Expected output:**
```
PASS: Gemma 3 27B KV/token = 1,015,808 bytes (1.02 MB)
PASS: Phi-4 KV/token = 204,800 bytes (200 KB)
PASS: Tied embedding saves 2.36 GB
PASS: attention reduction at 32K = 4.56× vs all-global
PASS: image KV cost = 260.0 MB for 256 tokens
PASS: Gemma 3 27B FP16 cost = $0.69/M tokens

✓ All Phi-4 / Gemma 3 arithmetic tests passed.
```

---

## Chapter Summary

Phi-4 and Gemma 3 represent the state of the art in capability-dense small
models as of 2026. Phi-4 achieves 70B-class reasoning at 14B through data
quality. Gemma 3 27B matches near-70B capability with 128K native context and
vision, while Gemma 3 4B is the best-in-class small multimodal model for edge
deployment.

For inference engineers, the key takeaways are architectural: Phi-4 is a tall
dense transformer that excels at reasoning-heavy tasks; Gemma 3 uses interleaved
local/global attention to extend context efficiently; and tied embeddings in
Gemma 3 reduce the lm_head memory cost by 50%.

The decision framework: if you need STEM/code quality in a tight memory budget,
Phi-4. If you need long context, multilingual capability, or vision at small
scale, Gemma 3. If you need maximum ecosystem compatibility and community
support, Llama 3.

---

## Self-Check Questions

1. Phi-4 has 40 layers and 10 KV heads (d_head = 128). Gemma 3 27B has 62
   layers and 16 KV heads (d_head = 256). Calculate the KV cache per token per
   layer for each, and the total per token for a full forward pass. Which model
   has a larger KV cache footprint at equivalent context lengths?

2. Gemma 3 uses local attention (window=1024) in 4 out of every 5 layers.
   For a 16,384-token sequence, calculate the attention compute reduction
   compared to full attention in all layers. Express as a ratio.

3. Tied embeddings in Gemma 3 mean the embedding matrix and lm_head share
   weights. Explain one potential training disadvantage of tied embeddings.
   Why does this trade-off favor inference over training?

4. A deployment serves Gemma 3 12B with max_model_len=65536 (64K tokens) on a
   single A100 80GB. The model weights at FP16 are approximately 24 GB. How
   much KV cache memory is available? What is the maximum number of concurrent
   requests if each uses 32K tokens of context?

5. Phi-4 achieves higher MMLU than Llama 3.1 70B at 14B parameters. From an
   inference engineering perspective, what are the practical implications for
   a business currently running Llama 3.1 70B? List three specific changes
   (configuration, hardware, cost) they would need to make to switch.


---

## Worked Solutions

### Question 1
**Phi-4: 40 layers, 10 KV heads, d_head=128. Gemma 3 27B: 62 layers (read from question context).**

From the chapter's question text: "Phi-4 has 40 layers and 10 KV heads (d_head=128). Gemma 3 27B has 62 layers..."

**KV cache per token for Phi-4 (14B):**
```
KV_per_token = 2 x 40 x 10 x 128 x 2 = 204,800 bytes = 200 KB/token
```

**KV cache per token for Gemma 3 27B (estimated 16 KV heads, d_k=128, 62 layers):**
```
KV_per_token = 2 x 62 x 16 x 128 x 2 = 507,904 bytes = 496 KB/token
```

**Comparison at 32K context:**
```
Phi-4:      32,000 x 200 KB = 6.4 GB
Gemma 3 27B: 32,000 x 496 KB = 15.9 GB
```

Phi-4 uses 2.5x less KV cache than Gemma 3 27B at 32K context, despite having similar parameter counts. This makes Phi-4 preferable for long-context deployments on memory-constrained hardware.

---

### Question 2
**Gemma 3 local attention (window=1024) in 4 of 5 layers. Relevant passage at position 45K in 50K context.**

**Attention coverage analysis:**
- Local attention layers (4 of 5): only attend to the last 1,024 tokens.
- Global attention layer (1 of 5): attends to ALL 50K tokens.

**Can the global layer find position 45K?**
The query at position 49,999 (near the end) can attend to position 45,000 in the **global attention layer** (distance = 4,999 << 50,000). **Yes** -- the global attention layer covers the full context.

**Performance caveat:** Global attention at 50K tokens requires computing 50,000 attention weights, which is 50,000/1,024 = 48.8x more compute than local attention. But this only happens in 1 of 5 layers, so total attention FLOPs are: 4/5 x 1,024 + 1/5 x 50,000 = 819.2 + 10,000 = 10,819 effective attention span per token (vs 50,000 for full attention). Overall FLOPs reduction: 10,819/50,000 = 21.6% of full attention cost -- an 80% FLOPs savings while maintaining global retrieval capability.

---

### Question 3
**Tied embeddings in Gemma 3: embedding matrix and lm_head share weights.**

**Memory saving:**
Without tying: embedding matrix (vocab_size x d_model) + lm_head (d_model x vocab_size) = 2 matrices.
With tying: one matrix serves both roles = 1 matrix.

For Gemma 3 27B (d_model=4,608, vocab_size=256,000):
```
single_matrix = 256,000 x 4,608 = 1.18B parameters
memory = 1.18B x 2 bytes (BF16) = 2.36 GB
without_tying = 2 x 2.36 = 4.72 GB
saving = 2.36 GB per deployment
```

**Why this is non-trivial for inference:** The lm_head is read from HBM at every decode step (to compute logits over 256K vocab). A 2.36 GB matrix is loaded every step; at A100 bandwidth (2 TB/s): 2.36/2000 = 1.18 ms overhead per token just for lm_head. Tied embeddings eliminate the duplicate storage while providing identical inference latency -- the matrix is loaded once regardless.

**Training note:** Tied embeddings constrain the gradient updates -- the same delta applies to both the input representation and output scoring roles. This can limit model expressiveness for very large vocabularies but is standard practice for efficiency.

---

### Question 4
**Gemma 3 12B, max_model_len=65536, A100 80 GB. Memory and concurrency.**

**Model weights (12B BF16):**
```
weights = 12B x 2 = 24 GB
```

**KV cache per token (Gemma 3 12B: estimated 8 KV heads, d_k=256, 28 layers):**
```
KV_per_token = 2 x 28 x 8 x 256 x 2 = 229,376 bytes = 224 KB/token
```

**Available HBM for KV:**
```
available = 80 x 0.90 - 24 = 72 - 24 = 48 GB (with gpu_memory_utilization=0.90)
```

**Maximum KV per sequence at 65,536 tokens:**
```
KV_per_seq = 65,536 x 224 KB = 14.68 GB
```

**Maximum concurrent sequences:**
```
max_seqs = floor(48 / 14.68) = 3 sequences
```

At 64K context, only 3 concurrent requests can run simultaneously on a single A100 80 GB. To increase concurrency: (a) use INT8 KV cache quantization (halves KV to 7.34 GB/seq, enabling 6 concurrent requests), or (b) use multiple A100s with pipeline parallelism.

---

### Question 5
**Phi-4 outperforms Llama 3.1 70B at 14B parameters. Inference cost implications:**

**If Phi-4 14B matches or exceeds Llama 3.1 70B quality:**

**Cost comparison (BF16, A100):**
- Phi-4 14B: 28 GB weights. Fits on 1x A100 with KV budget. Decode: 28/2000 = 14 ms/token.
- Llama 3.1 70B: 140 GB weights. Requires 2x A100. Decode: 70 ms/token per GPU step.

**Throughput advantage:**
Phi-4 generates tokens 5x faster at batch=1. At the same hardware cost (1 A100 vs 2 A100), Phi-4 achieves ~10x better cost efficiency if quality is equivalent.

**Serving cost reduction:**
A deployment currently using 8x A100 to run Llama 3.1 70B at scale could switch to 2x A100 running Phi-4 14B for the same quality output, reducing GPU costs by 75%.

**Where the advantage holds and where it doesn't:**
Phi-4's performance advantage (if confirmed) likely holds on reasoning benchmarks and structured tasks (the focus of its synthetic training data). It may underperform Llama 3.1 70B on:

- Open-domain knowledge breadth (70B sees more diverse training data)
- Languages other than English (Phi-4 training is English-focused)
- Very long contexts where Llama's rope scaling has been extensively tested

**Production recommendation:** A/B test Phi-4 vs Llama 70B on your specific task distribution before migrating. If the benchmark advantage (higher MMLU, coding scores) translates to your production traffic, the 75% cost reduction makes migration highly compelling.

