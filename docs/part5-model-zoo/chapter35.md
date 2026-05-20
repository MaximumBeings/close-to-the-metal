# Chapter 35: Qwen — Multilingual, Long-Context, and the Full Model Family

> *"Alibaba's Qwen family is the most deployment-tested multilingual model stack outside of the US hyperscalers — 30+ languages, 0.5B to 72B parameters, with vision and audio variants all from one architecture."*

---

**What you will understand after this chapter:**

- How Qwen's architecture differs from Llama (tokenizer, attention, positional encoding)
- How to choose the right Qwen size for a given workload and budget
- Multilingual serving challenges: tokenizer differences, language-specific batching
- How to serve Qwen2.5/Qwen3 efficiently with vLLM and llama.cpp

**What you need first:**

- Chapter 3 (Tokens and Batching), Chapter 10 (Quantization), Chapter 14 (vLLM Knobs)

---

## 35.1 The Qwen Family

Qwen (通义千问, Tongyi Qianwen) is Alibaba DAMO Academy's large language model family. Since its first public release in September 2023, Qwen has evolved through three major generations:

```
  Qwen Family Timeline
  
  Qwen (2023-09): 7B/14B/72B, bilingual (Chinese+English)
  Qwen1.5 (2024-02): 0.5B–110B, multilingual, GQA
  Qwen2 (2024-06): 0.5B–72B, 30+ languages, improved context
  Qwen2.5 (2024-09): 0.5B–72B + MoE 57B-A14B/235B-A22B, 128K ctx
  Qwen3 (2025-04): 0.6B–235B, thinking/non-thinking modes, MoE
  
  Specialized variants:
  Qwen-VL: vision-language (images + text)
  Qwen-Audio: audio understanding
  Qwen-Coder: code generation
  QwQ: reasoning (RL-trained, like DeepSeek-R1)
```

### 35.1.1 Architecture Overview

Qwen2.5 follows a standard transformer architecture with these specific choices:

```
  Qwen2.5-72B Architecture
  ┌──────────────────────────────────────────────────────────────┐
  │ Embedding layer: vocab=152,064 tokens (BPE, SentencePiece)  │
  ├──────────────────────────────────────────────────────────────┤
  │ 80 × Transformer blocks:                                     │
  │   ┌─────────────────────────────────────────────────────┐   │
  │   │ RMSNorm → GQA (64 Q heads, 8 KV heads, d_h=128)    │   │
  │   │ RoPE positional encoding (θ=1,000,000)              │   │
  │   │ → FFN: SwiGLU (d_model=8192, d_ffn=29,568)         │   │
  │   │ → RMSNorm                                           │   │
  │   └─────────────────────────────────────────────────────┘   │
  ├──────────────────────────────────────────────────────────────┤
  │ Output: RMSNorm → Linear (8192 → 152,064) → logits          │
  └──────────────────────────────────────────────────────────────┘
```

Key architectural choices vs. Llama 3.1 70B:

| Feature | Qwen2.5-72B | Llama 3.1 70B |
|---|---|---|
| Vocab size | 152,064 | 128,256 |
| Q heads | 64 | 64 |
| KV heads (GQA) | 8 | 8 |
| Context length | 131,072 | 131,072 |
| RoPE base θ | 1,000,000 | 500,000 |
| FFN type | SwiGLU | SwiGLU |
| Tie embeddings | No | No |

Both are remarkably similar architecturally — the main difference is the tokenizer and training data composition.

---

## 35.2 The Qwen Tokenizer — Why It Matters

Qwen uses a tiktoken-based BPE tokenizer with a vocabulary of 152,064 tokens. This is 24k tokens larger than Llama 3.1's vocabulary.

### 35.2.1 Multilingual Tokenization Efficiency

The large vocabulary includes dedicated tokens for Chinese characters, Japanese kanji, Korean hangul, and common multilingual subwords:

```
WORKED EXAMPLE 35.1 — Token Count Comparison
─────────────────────────────────────────────────────────────────────
Text: "人工智能改变世界" (Chinese: "AI changes the world", 8 characters)

Llama 3.1 tokenizer (128k vocab, less Chinese coverage):
  Tokens: ["人", "工", "智", "能", "改", "变", "世", "界"] → 8 tokens
  (Each character may be its own token due to limited Chinese vocab)

Qwen2.5 tokenizer (152k vocab, strong Chinese coverage):
  Tokens: ["人工智能", "改变", "世界"] → 3 tokens
  (Multi-character chunks recognized as single tokens)

Implication: Same Chinese text = 2.7× fewer Qwen tokens
  → 2.7× less KV cache for Chinese conversations
  → 2.7× higher effective throughput per user turn
─────────────────────────────────────────────────────────────────────
```

This tokenization efficiency is why Qwen is often faster in practice for multilingual workloads despite similar architectural specifications.

### 35.2.2 Special Tokens for Chat Format

Qwen uses specific chat template tokens:
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is the capital of France?
<|im_end|>
<|im_start|>assistant
```

These differ from Llama's `[INST]`/`[/INST]` format. When using vLLM or llama.cpp, always use the model's built-in chat template:
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

---

## 35.3 The Full Qwen Size Chart

```
  Qwen2.5 Family — Choose by Workload
  
  Size        Params  Active   Layers  d_model  Min VRAM  Use case
  ─────────────────────────────────────────────────────────────────
  0.5B        494M    494M      28     896      1 GB      Edge, classification
  1.5B        1.54B   1.54B     28     1,536    3 GB      Simple chat, IoT
  3B          3.09B   3.09B     36     2,048    6 GB      Assistant on device
  7B          7.07B   7.07B     28     3,584   14 GB      Strong assistant (RTX 3090)
  14B         14.7B   14.7B     48     5,120   28 GB      Professional quality (2×A100)
  32B         32.5B   32.5B     64     5,120   64 GB      Near-frontier (A100 80G)
  72B         72.7B   72.7B     80     8,192  144 GB      Frontier (2× H100)
  57B-A14B MoE 57.4B  14.3B     28     3,584  114 GB      Quality at 14B compute
  235B-A22B MoE 235B  22.0B     94     4,096  470 GB      Frontier at 22B compute
```

```
WORKED EXAMPLE 35.2 — Choosing the Right Qwen Size
─────────────────────────────────────────────────────────────────────
Scenario: Customer support chatbot, English+Chinese, response quality
          needs to match GPT-3.5-turbo, budget: 2× A10G (24GB each)

Available VRAM: 2 × 24 GB = 48 GB
Budget-maximizing choice:

Qwen2.5-32B (INT4/AWQ): 32.5B × 0.5 bytes = 16.25 GB → fits on 1 GPU
Qwen2.5-32B (BF16):     32.5B × 2 bytes   = 65 GB   → needs 2 GPUs but exceeds 48GB
Qwen2.5-14B (BF16):     14.7B × 2 bytes   = 29.4 GB → fits on 1× A10G ✓

Best choice: Qwen2.5-14B-Instruct, BF16, single A10G
  or: Qwen2.5-32B-Instruct-AWQ on 1× A10G (better quality, same cost)
─────────────────────────────────────────────────────────────────────
```

---

## 35.4 Qwen2.5 MoE Variants

Qwen2.5-57B-A14B is a 57B parameter MoE model that activates only 14B parameters per token:

```
  Qwen2.5-57B-A14B MoE Architecture
  
  ┌──────────────────────────────────────────────────────────────┐
  │ Attention layers: standard GQA (dense, not MoE)              │
  ├──────────────────────────────────────────────────────────────┤
  │ FFN layers: MoE                                              │
  │   64 experts, top-8 routing (vs DeepSeek's 256, top-8+1 shared)│
  │   Each expert: 1/8 of dense FFN size                         │
  │   Active per token: 8 experts × (1/8 FFN) = 1 full FFN      │
  └──────────────────────────────────────────────────────────────┘
  
  Memory:    57B × 2 bytes = 114 GB (still needs 2× H100)
  Compute:   14B active → similar throughput to 14B dense
  Quality:   comparable to Qwen2.5-72B on many benchmarks
```

---

## 35.5 Serving Qwen with vLLM

### 35.5.1 Standard Deployment

```bash
# Qwen2.5-7B-Instruct (single GPU)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85

# Qwen2.5-72B-Instruct (multi-GPU)
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --enable-chunked-prefill

# Qwen2.5-72B-Instruct-GPTQ-Int4 (fewer GPUs)
vllm serve Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 \
    --tensor-parallel-size 2 \
    --quantization gptq \
    --max-model-len 131072
```

### 35.5.2 Multilingual Batch Considerations

When mixing Chinese and English requests, token lengths can differ dramatically:

```python
# Chinese text tokenizes 2-3× more efficiently in Qwen
# This creates uneven batch sizes — configure max-padded-len accordingly

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_model_len=8192,
    # For multilingual: allow longer decoded lengths
    # Chinese questions often need longer responses
)

# Mixed-language batch
prompts_zh = ["请解释量子计算的原理"]   # Chinese
prompts_en = ["Explain quantum computing"]  # English

sampling = SamplingParams(
    temperature=0.7,
    max_tokens=512,
)
outputs = llm.generate(prompts_zh + prompts_en, sampling)
```

### 35.5.3 Qwen3 Thinking Mode

Qwen3 introduces a "thinking" toggle (similar to DeepSeek-R1's reasoning mode):

```python
# Enable thinking mode for complex reasoning
messages = [{"role": "user", "content": "Solve step by step: ..."}]

# With thinking (longer, more accurate for complex problems)
sampling_think = SamplingParams(
    temperature=0.6,
    max_tokens=8192,       # allow long reasoning chain
    stop=["<|im_end|>"],
)

# Without thinking (faster, good for simple queries)
sampling_fast = SamplingParams(
    temperature=0.7,
    max_tokens=512,
)
```

---

## 35.6 Serving Qwen with llama.cpp

```bash
# Download Qwen2.5-7B-Instruct Q4_K_M GGUF
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
    --include "qwen2.5-7b-instruct-q4_k_m.gguf" --local-dir ./models

# Run with llama.cpp
# Key flag choices:
#   --chat-template qwen   important: use Qwen chat format
./build/bin/llama-cli \
    -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -n 512 \
    -c 16384 \
    --chat-template qwen \
    -p "You are a helpful assistant." \
    --in-prefix "<|im_start|>user\n" \
    --in-suffix "<|im_end|>\n<|im_start|>assistant\n"

# Server mode for multi-user
./build/bin/llama-server \
    -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -c 16384 \
    --chat-template qwen \
    --port 8080 \
    -np 4   # 4 parallel slots
```

`[COMMON TRAP]` Not specifying `--chat-template qwen` results in garbled output because Qwen's `<|im_start|>` tokens get decoded incorrectly. Always explicitly set the template for Qwen models.

---

## 35.7 KV Cache Budget for Qwen2.5

```
WORKED EXAMPLE 35.3 — KV Cache Budget for Qwen2.5-72B
─────────────────────────────────────────────────────────────────────
Architecture: n_layers=80, n_kv_heads=8, head_dim=128, dtype=BF16
KV per token: 2 × 80 × 8 × 128 × 2 = 327,680 bytes (320 KB)

Hardware: 2× H100 80GB (160 GB total)
  Model weights (BF16): 72.7B × 2 = 145.4 GB
  Available for KV:     160 - 145.4 = 14.6 GB (after 90% factor: 13.1 GB)

At 32K context:
  KV per seq:  32,000 × 320 KB = 10.0 GB
  Max sequences: 13.1 / 10.0 = 1 (single sequence at 32K context!)

Recommendation for 32K+ context: use INT8 KV cache
  INT8 KV per token: 320 KB / 2 = 160 KB
  KV per seq at 32K: 5.0 GB
  Max sequences: 13.1 / 5.0 = 2 sequences

Or use Qwen2.5-72B-Instruct-GPTQ-Int4:
  Model weights: 72.7B × 0.5 = 36.4 GB
  Available for KV: 160 - 36.4 = 123.6 GB
  Max sequences at 32K: 123.6 / 10.0 = 12 sequences ✓
─────────────────────────────────────────────────────────────────────
```

---

## 35.8 Qwen-VL and Qwen2.5-VL — The Vision Stack

Qwen's vision-language models share the same language backbone as the text models,
with a vision encoder and projection layer grafted on. Chapter 29 covers the VLM
mechanics in depth; this section covers Qwen-specific deployment decisions.

### 35.8.1 Qwen2.5-VL Architecture Differences

Qwen2.5-VL introduces two serving-relevant changes vs. Qwen2-VL:

**Window attention in the ViT:**
The vision encoder uses sliding window self-attention within each row of the image.
Full self-attention only runs every 4 encoder blocks. This reduces ViT compute from
O(N²) to approximately O(N × W) where W is the window size, enabling processing of
images up to 4K resolution without quadratic cost.

**Unified image/video representation:**
Images are treated as single-frame videos. A 1920×1080 image and a 2-second 960×540
clip at 1fps are processed identically — through the same temporal token budget.

```
  Qwen2.5-VL token budget (image):
    Resolution    Tokens (approx)   Window attn saving
    336×336       256               60% vs. naive
    672×672       784               65%
    1080×1080     1,568             70%
    1920×1080     2,496             72%
```

### 35.8.2 Serving Qwen2.5-VL Sizes

```
  Model              VRAM    KV/img(1080p)   Use case
  ─────────────────────────────────────────────────────────
  Qwen2.5-VL-3B      6 GB    200 MB          Edge / laptop
  Qwen2.5-VL-7B      16 GB   200 MB          Single-GPU API server
  Qwen2.5-VL-32B     80 GB   400 MB          High-accuracy production
  Qwen2.5-VL-72B     160 GB  400 MB          Best quality, 2× H100
```

**vLLM configuration:**

```bash
# 7B — single H100 or A100 (40 GB)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=10,video=3" \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90

# 72B — 2× H100, tensor parallel
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=5,video=1" \
    --gpu-memory-utilization 0.90
```

### 35.8.3 Multi-Image Document Workflows

Qwen2.5-VL-72B scores state-of-the-art on DocVQA and ChartQA benchmarks (2025).
For document processing (PDFs, spreadsheets, presentations):

```python
import openai, base64
from pathlib import Path

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

def pdf_page_to_b64(page_path: str) -> str:
    return base64.b64encode(Path(page_path).read_bytes()).decode()

# Send a multi-page document
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            # Send up to 10 pages (limit-mm-per-prompt="image=10")
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{pdf_page_to_b64('page_1.png')}"}},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{pdf_page_to_b64('page_2.png')}"}},
            {"type": "text",
             "text": "Summarize the key financial figures from these pages."},
        ],
    }],
    max_tokens=1024,
)
```

---

## 35.9 Qwen3 — What Changed from Qwen2.5

Qwen3 (released April 2025) introduced several changes with direct serving implications.

### 35.9.1 Dual-Mode Generation

Every Qwen3 model — from 0.6B to 235B — supports both fast non-thinking and slow thinking modes within the same weights. The switch is a chat-template parameter, not a separate model checkpoint:

```python
# Non-thinking mode (fast, for FAQ and simple queries)
text = tokenizer.apply_chat_template(
    messages, tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,          # ← controls the mode
)

# Thinking mode (slower, better for reasoning and math)
text = tokenizer.apply_chat_template(
    messages, tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
```

In thinking mode the model emits a `<think>...</think>` block before its final answer. Budget: 8,000–32,000 tokens for the reasoning trace depending on model size. This means a single thinking-mode request can consume 10–30× the KV cache of a standard request. Route thinking-mode requests to a dedicated pool with `--max-model-len 32768` and lower `--max-num-seqs` to avoid OOM.

### 35.9.2 Qwen3 MoE Architecture (235B-A22B)

Qwen3's flagship 235B model activates only 22B parameters per forward pass. The routing granularity changed from Qwen2.5's design:

```
Qwen2.5-57B-A14B:  64 experts per FFN layer, route top-8 (coarse-grained)
  Each expert = 1/8 of a full FFN → 8 activated = one full FFN equivalent

Qwen3-235B-A22B:  128 experts per FFN layer, route top-1 (fine-grained)
  Each expert is smaller; only 1 expert activated per token per layer
  
  Serving implication:
  - Full weights: 235B × 2 bytes = 470 GB → requires FP8 to fit 4× H100
  - Active compute per token: ~22B parameters
  - Throughput comparable to a 22B dense model, not 235B
```

### 35.9.3 Qwen3-30B-A3B — The Edge MoE

Qwen3 added a 30B-parameter MoE with only 3B active parameters per token. At GGUF Q4\_K\_M (7.5 bits/param average):

```
Full weights:    30B × 0.5 bytes ≈ 15 GB  (fits on 16 GB VRAM or Apple M-series)
Active compute:  3B per forward pass
Speed (M3 Max):  ~55 tok/s — faster than Qwen2.5-7B at similar quality
```

This is the recommended replacement for Qwen2.5-7B in the FAQ pool of the Chapter 38 architecture if you are running on Apple Silicon edge nodes.

---

## 35.10 Benchmark Comparisons Across Model Families

At the time of writing (May 2026), key benchmarks place the Qwen family as follows:

```
MMLU-Pro (5-shot, rigorous general knowledge):
  Qwen2.5-72B:         79.2%
  Llama-3.1-70B:       73.3%
  DeepSeek-V3 (671B):  81.2%

C-Eval (Chinese knowledge and reasoning):
  Qwen2.5-72B:         90.1%   ← clear Qwen advantage
  Llama-3.1-70B:       70.5%
  DeepSeek-V3 (671B):  91.8%

HumanEval (Python code):
  Qwen2.5-Coder-32B:   92.1%
  DeepSeek-V3 (671B):  89.9%
  Llama-3.1-70B:       80.5%

MATH-500 (competition mathematics, thinking mode):
  Qwen3-32B (thinking): 97.2%
  DeepSeek-R1 (671B):   97.3%
  Llama-3.1-70B:        76.0%
```

**Practical selection rules:**

- Chinese, Japanese, Korean workloads: Qwen at any size tier beats Llama by a wide margin due to tokeniser and training data composition.
- English general assistant: Qwen2.5-72B and Llama-3.1-70B are within 2–3% on English benchmarks; choose by hardware fit and licensing.
- Code generation: Qwen2.5-Coder-32B is the strongest open-weights code model under 70B at time of writing.
- Complex reasoning and math: Qwen3-thinking or DeepSeek-R1; the quality gap over non-thinking models is large (>20pp on MATH-500).

---

## 35.11 Chapter Summary

Qwen's strengths: the widest model size range (0.5B–235B) from a single architecture family, the most efficient multilingual tokenizer for CJK languages (2–3× fewer tokens vs. Llama for Chinese), production-quality MoE variants including the 30B-A3B edge model, dual-mode reasoning in Qwen3, and the Qwen2.5-VL series for multimodal workloads.

Key serving rules: always specify `--chat-template qwen` in llama.cpp; use GPTQ-Int4 for 72B on 2× H100; route thinking-mode requests to a dedicated pool with a higher `max_model_len`; enable `--enable-prefix-caching` for repeated-image VLM workloads.

### Where We Go Next

Chapter 36 covers Kimi — Moonshot AI's production system designed specifically for ultra-long contexts (up to 1M tokens), and Moon-Cache, their hierarchical KV caching system.

---

## Self-Check Questions

1. Qwen2.5-72B uses 8 GQA KV heads and d_k = 128 with 80 transformer layers. Compare the KV cache memory per token versus a standard MHA model with 64 KV heads at the same d_k. *(Section 35.1)*

2. Qwen2.5 has a vocabulary of 152,064 tokens. LLaMA-3 has 128,256. For a Chinese-language prompt of 500 characters, estimate the token count for each tokeniser and explain the difference. *(Section 35.2)*

3. You are building a cost router over the Qwen family (7B, 14B, 32B, 72B). For a request mix that is 60% simple queries, 30% medium, and 10% complex, design a two-stage cascade using only two models. *(Section 35.3)*

4. Qwen2.5-VL-7B processes a 1024×1024 image. Using the token budget table in §35.8.1, compute how many visual tokens are generated and whether a 10-image document fits in a 32K context window alongside a 512-token user question. *(Section 35.8)*

5. Qwen3-235B-A22B has 128 fine-grained experts and activates 1 per token per FFN layer. Compare its memory footprint and per-token compute cost to Qwen2.5-72B dense (BF16). At what GPU count does each model become feasible to serve without quantisation? *(Section 35.9)*


---

## Worked Solutions

### Question 1
**Qwen2.5-72B: 8 GQA KV heads, d_k=128, 80 layers. KV per token vs MHA with 64 KV heads.**

**Qwen2.5-72B (GQA, 8 KV heads):**
```
KV_per_token = 2 x 80 x 8 x 128 x 2 = 327,680 bytes = 320 KB/token
```

**MHA model with 64 KV heads (same d_k=128, 80 layers):**
```
KV_per_token = 2 x 80 x 64 x 128 x 2 = 2,621,440 bytes = 2,560 KB/token = 2.5 MB/token
```

**Ratio:** MHA requires 2,560 / 320 = **8x more KV cache** than Qwen2.5-72B's GQA configuration. This is exactly the GQA compression ratio (64 KV heads / 8 KV heads = 8x). At 128K context: Qwen2.5 needs 40 GB vs 320 GB for MHA -- making long-context serving feasible on 2x H100 vs requiring 4-8x H100.

---

### Question 2
**Qwen2.5 vocab=152,064. LLaMA-3 vocab=128,256. Chinese prompt of 500 characters.**

**Qwen2.5 tokenizer:**
Qwen's tokenizer is trained on a large Chinese corpus with extensive Chinese character coverage. Chinese characters are common, single tokens (many 3-byte UTF-8 sequences = 1 token). 500 Chinese characters -> approximately **500-600 tokens** (most characters are single tokens; some idioms or compound words may be split).

**LLaMA-3 tokenizer:**
LLaMA-3's tokenizer has 128,256 tokens and reasonable multilingual coverage, but Chinese character merging is less aggressive than Qwen's. A 3-byte UTF-8 Chinese character may be split into 2-3 subword tokens. 500 characters -> approximately **750-1,500 tokens** (1.5-3x more than Qwen).

**Difference:** Qwen2.5's larger vocabulary (152K vs 128K) includes more merged Chinese token pairs, resulting in more efficient encoding of Chinese text. At 500 characters, Qwen might use 550 tokens while LLaMA-3 uses 1,200 tokens -- a 2.2x token efficiency advantage for Chinese content. This directly reduces KV cache usage and prefill FLOPs for Chinese-language workloads.

---

### Question 3
**Cost router over Qwen family (7B, 14B, 32B, 72B). 60% simple, 30% medium, 10% complex. Two-model cascade:**

**Optimal two-model selection:**
The goal is to minimize average cost while maintaining quality.

Cost estimates (relative to 72B = 1.0):

- 7B: ~0.10
- 14B: ~0.19
- 32B: ~0.44
- 72B: 1.00

**Best two-model cascade: Qwen2.5-7B (first) + Qwen2.5-72B (escalation):**

Routing:

- Simple (60%): 7B only -> cost = 0.10 per request
- Medium (30%): 7B initial (low confidence) -> escalate to 72B -> cost = 0.10 + 1.00 = 1.10
- Complex (10%): always escalate to 72B -> cost = 0.10 + 1.00 = 1.10

Average cost:
```
avg = 0.60 x 0.10 + 0.30 x 1.10 + 0.10 x 1.10
    = 0.06 + 0.33 + 0.11 = 0.50
```
vs always 72B: 1.00 -> **50% cost reduction**.

**Alternative: 14B + 72B** (slightly better quality on medium tasks):
```
avg = 0.60 x 0.19 + 0.30 x 1.19 + 0.10 x 1.19
    = 0.114 + 0.357 + 0.119 = 0.59
```
41% cost reduction -- worse than 7B + 72B.

**Winner: Qwen2.5-7B + Qwen2.5-72B** for maximum cost efficiency. Use a lightweight classifier trained on task complexity to route between them.

---

### Question 4
**Qwen2.5-VL-7B processes 1024x1024 image. Visual tokens + 10-image document in 32K context.**

From Qwen2.5-VL documentation: 1024x1024 images are encoded at a resolution that produces approximately 768-1,024 visual tokens (using dynamic resolution encoding with 28x28 pixel patches).

Using the chapter's token budget table: **~756 visual tokens** for a 1024x1024 image (14x14 grid of 4x4 merged patches = 196 tokens * 4 = 784, approximately 756 accounting for special tokens).

**10-image document:**
```
visual_tokens = 10 x 756 = 7,560 visual tokens
text_tokens = 512 (user question, estimated)
total = 7,560 + 512 = 8,072 tokens
```

Does it fit in 32K context? 8,072 << 32,768. **Yes, comfortably.** There is room for 24,696 additional tokens for document text or lengthy responses.

**Practical consideration:** Loading 10 large images requires significant preprocessing time (CLIP encoding). For the 7B model, ensure the vision encoder can handle 10 images in a single forward pass (some implementations require sequential image encoding).

---

### Question 5
**Qwen3-235B-A22B: 128 experts, activates 1 per token per FFN. vs Qwen2.5-72B dense BF16.**

**Qwen3-235B-A22B memory footprint:**
Total parameters: 235B. At BF16 (2 bytes/param):
```
weight_memory = 235B x 2 = 470 GB
```
All 235B parameters must be in HBM (any expert could be called). Minimum GPU configuration: 6x H100-80GB (480 GB).

**Per-token compute cost (active parameters):**
Only A22B = 22B parameters activate per token (1 of 128 experts per FFN layer, not 8 like DeepSeek). At BF16:
```
active_per_token = 22B x 2 bytes = 44 GB read per decode step
time = 44 / 3,350 GB/s = 13.1 ms per token (batch=1)
```

**Qwen2.5-72B dense BF16:**
```
weight_memory = 72B x 2 = 144 GB -> 2x H100-80GB minimum
decode_time = 144 / 3,350 = 43 ms per token (batch=1)
```

Wait, 72B dense reads all 144 GB every step, taking 43 ms. Qwen3-235B-A22B reads only 44 GB per step, taking 13 ms. So Qwen3 is **3.3x faster per token at batch=1** despite being 3.3x larger!

**GPU count for serving without quantization:**
- Qwen2.5-72B: 2x H100-80GB (144 GB weights, comfortable with 16 GB KV headroom per GPU)
- Qwen3-235B-A22B: 6x H100-80GB (470 GB weights) -- but 6 GPUs with NVLink required for all-to-all MoE routing

**Summary:** MoE enables serving a higher-quality 235B model at lower per-token latency than the dense 72B, at the cost of 3x more GPU memory and NVLink requirement.
*Companion code: [`docs/code/chapter_35.md`](../code/chapter_35.md)*
