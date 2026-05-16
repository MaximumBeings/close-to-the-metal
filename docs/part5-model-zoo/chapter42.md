# Chapter 42 — Phi-4 and Gemma 3: Small Models, Large Impact

> *"A 14B model that outperforms a 70B model is not a curiosity. It is a cost reduction waiting to be deployed."*

---

## 42.1 The Case for Small Models

The GPU budget of most organisations does not support a 70B model in production.
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
- **Creative tasks**: depth over width architecture favours reasoning over fluency

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
(Appendix M covers llama.cpp on Apple Silicon).

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
   Why does this trade-off favour inference over training?

4. A deployment serves Gemma 3 12B with max_model_len=65536 (64K tokens) on a
   single A100 80GB. The model weights at FP16 are approximately 24 GB. How
   much KV cache memory is available? What is the maximum number of concurrent
   requests if each uses 32K tokens of context?

5. Phi-4 achieves higher MMLU than Llama 3.1 70B at 14B parameters. From an
   inference engineering perspective, what are the practical implications for
   a business currently running Llama 3.1 70B? List three specific changes
   (configuration, hardware, cost) they would need to make to switch.
