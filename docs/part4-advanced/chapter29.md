# Chapter 29: Multimodal Inference — Vision and Audio

Language models that can only read text are like consultants who have never seen a
document: they can discuss concepts, but they cannot examine the evidence.
Modern production deployments increasingly require models to process images, screenshots,
PDFs rendered as images, medical scans, charts, diagrams, short video clips, and
transcribed audio — all alongside text.

This chapter covers the engineering of multimodal inference from the ground up: how
images become tokens, how vision encoders are architecturally integrated, the memory and
latency costs of visual inputs, and how to configure vLLM and llama.cpp to serve
vision-language models (VLMs) in production.

---

## 29.1  From Pixels to Tokens: The Vision Pipeline

Every vision-language model converts an image into a sequence of "visual tokens" that the
language model treats as ordinary embeddings in its input sequence.
The pipeline has three stages:

```
Image (H×W×3)
    │
    ▼
[Vision Encoder]        ← CLIP, SigLIP, InternViT, EVA-CLIP
    │  Patch embeddings: (H/14)×(W/14) vectors of dim d_vis
    ▼
[Projection Layer]      ← Linear, MLP, or cross-attention connector
    │  Visual tokens: N_vis × d_llm
    ▼
[Language Model]        ← Llama, Mistral, Qwen, etc.
    │  Attends to visual tokens as part of input sequence
    ▼
Generated text
```

### 29.1.1  Patch Extraction

Vision Transformers (ViTs) divide an image into non-overlapping square patches.
A ViT-L/14 (the most common in VLMs) uses 14×14 pixel patches.

For a 336×336 image:
```
n_patches = (336/14) × (336/14) = 24 × 24 = 576 patches
```

Each patch is flattened and linearly projected to dimension `d_vis = 1024` (ViT-L).
Adding a CLS token: 577 total embeddings enter the ViT.

After the ViT transformer blocks, the 576 patch embeddings (excluding CLS) are passed to
the projection layer.

### 29.1.2  The Vision Encoder Zoo

| Encoder | Architecture | Patch | Output dim | Params |
|---|---|---|---|---|
| CLIP ViT-L/14 | ViT-L | 14px | 1024 | 307M |
| CLIP ViT-L/14@336 | ViT-L | 14px | 1024 | 307M |
| SigLIP ViT-SO400M | ViT-So400M | 14px | 1152 | 400M |
| InternViT-300M | ViT | 14px | 1024 | 300M |
| InternViT-6B | ViT-6B | 14px | 3200 | 6B |
| EVA-CLIP ViT-18B | ViT | 14px | 5120 | 18B |
| DFN CLIP ViT-H | ViT-H | 14px | 1280 | 632M |

SigLIP (Sigmoid-loss Language-Image Pre-Training) uses a sigmoid loss instead of
InfoNCE contrastive loss.
It handles variable-resolution inputs more naturally and produces better embeddings
for dense prediction tasks like chart understanding and OCR.

### 29.1.3  Projection Layer Architectures

The projection layer bridges the vision encoder's embedding space to the language model's
embedding space.

**Linear (LLaVA-1.0):**
```
W_proj ∈ ℝ^(d_llm × d_vis)
visual_tokens = patches @ W_proj.T
```
Simplest approach, 576 tokens per image.

**MLP (LLaVA-1.5):**
```
visual_tokens = GELU(patches @ W1 + b1) @ W2 + b2
```
Two-layer MLP with same output count (576 tokens).
Significantly better than linear on complex visual tasks.

**Pixel Shuffle / Sub-image decomposition (LLaVA-1.6, InternVL):**
Each patch neighborhood is merged: 4 adjacent 14px patches → 1 output token via
spatial merging.
Reduces token count from 576 to 144 for standard 336×336 inputs.

High-resolution handling: the image is split into sub-tiles (e.g., 4 tiles of 336×336
plus a global thumbnail) and each tile is independently encoded.
Total tokens = 5 × 144 = 720 for a 672×672 input.

**Cross-attention connector (Flamingo, Idefics):**
Visual features are not injected into the input sequence at all.
Instead, every transformer layer has a gated cross-attention layer that attends to
the visual features as external memory.
This decouples vision capacity from language model context length but requires
training the language model from scratch or with full fine-tuning.

### 29.1.4  Token Count vs. Image Resolution

The number of visual tokens entering the LLM determines latency and KV cache pressure:

| Model | Input resolution | Visual tokens |
|---|---|---|
| LLaVA-1.5 | 336×336 | 576 |
| LLaVA-1.6 (7B) | 672×672 (4 tiles) | 2880 |
| InternVL2-8B | 448×448 (dynamic) | 256–1024 |
| Qwen2-VL-7B | Dynamic NTK | 64–1280 |
| MiniCPM-V 2.6 | Up to 1.8K×1.8K | up to 1792 |
| GPT-4V (est.) | 512×512 tiles | 85 per tile |

At 2880 visual tokens per image, a Llama-3.1-8B language model incurs the same
prefill cost as a 2880-token text prompt — before adding any text instruction.
Images are expensive.

---

## 29.2  Memory Cost of Visual Tokens

Visual tokens consume KV cache exactly like text tokens.
Using the formula from Chapter 27:

```
KV_bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes
```

For Llama-3.1-8B (n_layers=32, n_kv_heads=8, head_dim=128, BF16):
```
= 2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB per token
```

A 4-tile LLaVA-1.6 image (2880 tokens):
```
KV for one image = 2880 × 128 KB = 360 MB
```

On an 80 GB H100 serving Llama-3.1-8B (16 GB weights):
```
Available KV budget = (80 × 0.90 - 16) GB = 56 GB
Images per H100 = 56,000 MB / 360 MB ≈ 155 simultaneous images (with no text)
```

In practice, text + image together plus output tokens reduces this significantly.
A typical production VLM request at 2880 image tokens + 512 text tokens + 256 output
tokens = ~3648 total tokens per request, so the H100 can handle roughly 120 concurrent
requests before KV exhaustion.

---

## 29.3  Architecture Deep Dive: Key VLM Families

### 29.3.1  LLaVA Family

LLaVA (Large Language and Vision Assistant) is the reference architecture for most
open VLMs.

**LLaVA-1.5:**

- Vision encoder: CLIP ViT-L/14@336
- Projection: 2-layer MLP
- LLM: Vicuna-7B or Vicuna-13B
- Resolution: 336×336, 576 visual tokens
- Training: instruction-tune only (LLM + projection; ViT frozen)

**LLaVA-1.6 (LLaVA-NeXT):**

- Vision encoder: CLIP ViT-L/14@336 (unchanged)
- Projection: MLP with pixel shuffle
- LLM: Mistral-7B, Llama-3-8B, or Llama-3-70B
- Resolution: Any of 672×672, 672×1344, 1344×672, 1344×1344 (dynamic)
- Sub-tile strategy: up to 4 tiles + 1 thumbnail → up to 2880 tokens

**Key insight from LLaVA-NeXT:** dividing high-resolution images into tiles processed
by the same 336×336 encoder avoids retraining the ViT.
The language model sees each tile's tokens concatenated with a `<image_tile_sep>` token,
learning to reason across tiles during instruction tuning.

### 29.3.2  InternVL Family

InternVL (International Vision-Language Model) pushes the vision encoder to 6B parameters
while keeping the language model swappable.

**InternVL2-8B:**

- Vision encoder: InternViT-300M (8B variant uses 6B encoder)
- Projection: MLP
- LLM: InternLM2-8B (based on Llama architecture)
- Dynamic resolution: 1–12 tiles of 448×448 depending on image aspect ratio
- Visual tokens: 256 per tile (pixel-shuffle 2× reduction from 32×32 patch grid)
- Context: 8192 for LLM

Dynamic tile selection: a short image (1920×100) uses 2×1 tiles; a tall document
(100×1920) uses 1×4 tiles. The model learns to handle variable token counts during
training via positional encoding extrapolation.

### 29.3.3  Qwen2-VL

Qwen2-VL introduces **Naive Dynamic Resolution** with 2D-RoPE:

- Visual tokens use 2D position IDs (row, column) rather than a flattened 1D index.
- The language model's RoPE is extended to handle 2D patches natively.
- Any resolution input is processed at its native resolution up to a maximum token budget.
- For a 1024×768 image: `(1024/14) × (768/14) ≈ 73 × 54 ≈ 3942` raw patches,
  then reduced by temporal compression for video.

This eliminates tile boundaries and their associated artifacts.
Qwen2-VL-7B handles images from 28×28 to 2048×2048 natively.

### 29.3.4  MiniCPM-V

MiniCPM-V 2.6 (8B total) achieves near-GPT-4V performance at 8B scale through:

- Dual-vision encoder: SigLIP + compression via a lightweight Q-Former
- Token compression: 64 tokens per slice (vs. 576 for raw CLIP)
- Slice strategy: any aspect ratio → 1–9 slices + thumbnail
- Efficient: an 8B model serving at 3-4× fewer visual tokens than LLaVA-1.6

### 29.3.5  Llama 3.2 Vision (11B / 90B)

Meta's Llama 3.2 Vision models (released September 2024) take a different architectural
approach from LLaVA-style prefix injection: **cross-attention**.

```
  LLaVA-style (prefix injection):
    Visual tokens → prepended to input sequence
    LLM self-attention covers visual + text tokens together
    Context length = visual_tokens + text_tokens

  Llama 3.2 Vision (cross-attention):
    Visual tokens → stored as external memory
    Every transformer layer has a cross-attention layer that attends to visual memory
    LLM context length = text_tokens only (visual tokens are separate)
    Trade-off: more parameters (cross-attention layers) but text context not consumed
```

**Architecture details:**

```
  Vision encoder:  ViT-H/14 (CLIP-style, patch_size=14, input_res=560×560)
                   Patches: (560/14)² = 40×40 = 1600 per image
  Projection:      MLP connector → cross-attention key/value pairs
  Language model:  Llama 3.1 backbone (8B / 70B) with added cross-attention layers
                   every 4 transformer blocks

  Visual memory:   1600 patches × (n_cross_attn_layers) KV pairs
                   Not stored in main KV cache — stored in separate visual KV buffer
  Context window:  128K text tokens (unchanged by image presence)
```

**Serving implications of cross-attention:**

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Visual KV storage comparison                                          │
  │                                                                        │
  │  LLaVA-1.6 (prefix injection):                                        │
  │    Visual tokens enter main KV cache → consume from text budget       │
  │    Max concurrent requests limited by: weights + (vis+text) KV        │
  │                                                                        │
  │  Llama 3.2 Vision (cross-attention):                                  │
  │    Visual tokens in separate cross-KV buffer                          │
  │    Text KV cache budget not reduced by image presence                 │
  │    Cross-attn layers ≈ 10% of total layers → small extra memory cost  │
  │    But: cannot share visual KV across requests (each req owns its     │
  │    cross-attn buffer) — no visual prefix caching benefit              │
  └────────────────────────────────────────────────────────────────────────┘
```

**Memory budget for Llama 3.2 Vision 11B on H100 80GB:**

```
  LLM weights (BF16):          22 GB   (11B × 2 bytes)
  Vision encoder (BF16):        1.8 GB  (ViT-H ≈ 632M params)
  Cross-attn layers (BF16):     0.8 GB  (additional cross-attn weights)
  Text KV cache:                55 GB   (80 × 0.90 - 24.6 = usable for KV)
  Per-request cross-KV buffer:  ~300 MB (1600 patches × 32 cross-attn layers × 2 × 128 × 2B)
  ──────────────────────────────────────────────────────
  Concurrent requests:          55 GB / 300 MB ≈ 183 (visual) × text seq overlap
```

**vLLM configuration:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --limit-mm-per-prompt "image=1" \
    --gpu-memory-utilization 0.90
```

Note: Llama 3.2 Vision is officially licensed for commercial use under the Llama 3
Community License. The 90B model requires at least 2× H100 80GB with tensor parallelism.

### 29.3.6  Pixtral-12B

Mistral's Pixtral-12B (released September 2024) contributes a key innovation:
**a native vision encoder trained from scratch** rather than adapting CLIP.

```
  Pixtral ViT (400M params):
    - Custom 16×16 patch size (vs. 14×14 for CLIP-based encoders)
    - Trained jointly with the language model on interleaved image-text data
    - Supports variable image resolutions natively (no fixed input resolution)
    - Sliding window attention within each image row (efficient for wide images)

  Language backbone:
    - Mistral Nemo 12B (Transformer, GQA, sliding window attention)
    - [IMG] and [IMG_BREAK] / [IMG_END] special tokens mark tile boundaries
```

**Resolution handling:**

```
  Image input:  any resolution
  Tiling:       split into rows of 16×16 tiles
  Token count:  (width / 16) × (height / 16) patches
                plus one [IMG_BREAK] token per row
                plus one [IMG_END] token at end

  Example: 1024×768 image
    Tiles:  (1024/16) × (768/16) = 64 × 48 = 3072 tiles
    Break tokens: 48 row breaks + 1 end = 49
    Total visual tokens: 3072 + 49 = 3121

  This is ~2× LLaVA-1.6 HD token count but at much higher resolution
  with no distortion from tile-boundary artifacts.
```

**Serving Pixtral with vLLM:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Pixtral-12B-2409 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --limit-mm-per-prompt "image=8" \
    --gpu-memory-utilization 0.90
```

**Key engineering difference vs. CLIP-based models:**
CLIP was pre-trained on image-text pairs with a contrastive objective — it encodes images
into a semantic embedding space aligned with text. Pixtral's encoder was trained on the
same data distribution as the language model, including code screenshots, charts, and
multilingual text in images. It substantially outperforms CLIP on document understanding
and OCR tasks as a result.

### 29.3.7  Qwen2.5-VL

Qwen2.5-VL (released February 2025) succeeds Qwen2-VL with improved video and document
understanding. Key architectural changes:

```
  Window attention in ViT:
    - Dynamic resolution ViT (same as Qwen2-VL) but with window-partitioned
      self-attention inside the encoder for efficiency
    - Global attention every 4 blocks; local window attention otherwise
    - Result: ViT can process 4K+ images without O(N²) attention cost

  MRoPE (Multi-dimensional RoPE):
    - Extended from 2D (row, col) to 3D (time, row, col) for video
    - Temporal dimension uses different base frequency than spatial
    - Enables video understanding with dense frame sampling

  Token count reduction:
    - Window attention + 2× spatial compression → 4× fewer tokens vs. raw patches
    - 1080×1080 image: ~1568 tokens (vs. ~5000 for naive patch encoding)
```

**Available sizes:** 3B, 7B, 32B, 72B (dense) + 2B, 7B VL variants

**vLLM configuration:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=10,video=3" \
    --gpu-memory-utilization 0.90
```

---

## 29.4  vLLM Multimodal Configuration

### 29.4.1  Supported Models

vLLM supports these VLMs natively (as of vLLM 0.6+):

| Model | Architecture | Visual tokens | Context |
|---|---|---|---|
| `llava-hf/llava-1.5-7b-hf` | Prefix injection (CLIP) | 576 | 4096 |
| `llava-hf/llava-v1.6-mistral-7b-hf` | Prefix injection (CLIP, tiled) | up to 2880 | 32768 |
| `llava-hf/llava-v1.6-34b-hf` | Prefix injection (CLIP, tiled) | up to 2880 | 4096 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | Cross-attention (ViT-H) | 1600 (separate) | 131072 |
| `meta-llama/Llama-3.2-90B-Vision-Instruct` | Cross-attention (ViT-H) | 1600 (separate) | 131072 |
| `mistralai/Pixtral-12B-2409` | Prefix injection (native ViT) | variable | 131072 |
| `InternVL2-8B` | Prefix injection (InternViT) | 64–3072 | 8192 |
| `Qwen/Qwen2-VL-7B-Instruct` | Prefix injection (SigLIP+MRoPE) | 64–1280 | 32768 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Prefix injection (windowed ViT) | 64–1568 | 32768 |
| `openbmb/MiniCPM-V-2_6` | Q-Former compression | 64–576/slice | 8192 |

### 29.4.2  Serving a VLM

```bash
# LLaVA-1.6 on H100 80GB
python -m vllm.entrypoints.openai.api_server \
    --model llava-hf/llava-v1.6-mistral-7b-hf \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --image-input-type pixel_values \
    --image-token-id 32000 \
    --image-input-shape "1,3,336,336" \
    --image-feature-size 2880 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1

# Qwen2-VL-7B on H100 80GB
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=5,video=2" \
    --gpu-memory-utilization 0.90

# MiniCPM-V-2.6 on RTX 4090 24GB
python -m vllm.entrypoints.openai.api_server \
    --model openbmb/MiniCPM-V-2_6 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85
```

### 29.4.3  OpenAI-Compatible Image Request

vLLM's chat completions endpoint accepts images via the OpenAI vision format:

```python
import openai, base64, pathlib

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Option 1: URL reference
response = client.chat.completions.create(
    model="llava-hf/llava-v1.6-mistral-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/chart.png"},
            },
            {
                "type": "text",
                "text": "Describe the trend shown in this chart.",
            },
        ],
    }],
    max_tokens=512,
)
print(response.choices[0].message.content)

# Option 2: Base64-encoded image
def image_to_b64(path: str) -> str:
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode()

response = client.chat.completions.create(
    model="llava-hf/llava-v1.6-mistral-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_to_b64('screenshot.png')}"
                },
            },
            {"type": "text", "text": "What error is shown in this screenshot?"},
        ],
    }],
    max_tokens=256,
)
```

### 29.4.4  Batching Multi-Image Requests

vLLM can batch requests with different numbers of images per request.
The `--limit-mm-per-prompt` flag caps memory consumption:

```bash
--limit-mm-per-prompt "image=10"   # max 10 images per request
```

Internally, vLLM pre-encodes image pixel values using the vision encoder before the
decode loop starts, then inserts the resulting visual embeddings into the token stream
at positions marked by the image placeholder token.

The vision encoder runs asynchronously on the GPU while the language model processes
earlier requests.
In practice this overlap reduces effective TTFT for image-heavy workloads by 15–30%.

### 29.4.5  Visual Prefix Caching in vLLM

`[DEEP DIVE]`

vLLM 0.6 introduced prefix caching for multimodal inputs. When the same image appears
in multiple requests (e.g., a customer service bot analysing the same product screenshot
for different questions), vLLM can reuse the KV cache from the visual tokens across
requests.

```
  Without visual prefix caching:
    Request 1: [image_tokens(1600)] + [question_1(50)]  → full prefill of 1650 tokens
    Request 2: [image_tokens(1600)] + [question_2(50)]  → full prefill of 1650 tokens
    Total prefill: 3300 tokens

  With visual prefix caching (vLLM 0.6+):
    Request 1: [image_tokens(1600)] + [question_1(50)]  → prefill 1650, cache 1600
    Request 2: [image_tokens(1600)] + [question_2(50)]  → cache hit, prefill 50 only
    Total prefill: 1650 + 50 = 1700 tokens  (48% reduction)
```

**How it works:**
vLLM hashes the raw pixel values (or the pre-encoded visual embedding tensor) of each
image to produce a cache key. Identical images — even across different users — share the
same visual KV cache blocks via PagedAttention's copy-on-write mechanism.

**Enabling visual prefix caching:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --enable-prefix-caching \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=5" \
    --gpu-memory-utilization 0.90
```

**Limitations:**

- Applies to prefix-injection models (LLaVA, Qwen-VL, Pixtral) only — not Llama 3.2
  Vision's cross-attention architecture (separate KV buffer per request)

- The image must appear at the beginning of the prompt (prefix position) to be cacheable
- Video tokens benefit similarly: a 60-frame video reused across 10 questions incurs
  prefill cost once, not ten times

**Measured impact in production:**
For a document QA workload (same PDF page, 20 different questions):

- Without visual prefix cache: 20 × 1024 token prefill = 20,480 tokens/batch
- With visual prefix cache: 1024 + 19 × 50 (question only) = 2,000 tokens/batch
- **10× reduction in prefill compute** for repeated-image workloads

---

## 29.5  llama.cpp LLaVA Integration

### 29.5.1  Architecture

In llama.cpp, LLaVA is handled through the `clip.cpp` module:

```
1. Load GGUF model (LLM weights + projection weights)
2. Load CLIP model (vision encoder weights in separate .gguf or embedded)
3. For each image:
   a. Preprocess: resize to 336×336, normalize, extract patches
   b. clip_image_encode(): run CLIP forward pass → 576 × 1024 embeddings
   c. Project embeddings: 576 × 1024 → 576 × d_llm via MLP
   d. Insert projected embeddings into llama batch at <image> token positions
4. llama_decode() with the combined text + image batch
```

### 29.5.2  Running LLaVA in llama.cpp

```bash
# Download the split model files
# LLaVA-1.6 Mistral 7B requires:
#   llava-v1.6-mistral-7b.Q4_K_M.gguf   (LLM + projection)
#   mmproj-model-f16.gguf                 (CLIP vision encoder)

llama-llava-cli \
    -m llava-v1.6-mistral-7b.Q4_K_M.gguf \
    --mmproj mmproj-model-f16.gguf \
    --image /path/to/image.jpg \
    -p "USER: <image>\nDescribe this image in detail.\nASSISTANT:" \
    -ngl 33 \
    --temp 0.1 \
    -c 4096 \
    --flash-attn
```

The `--mmproj` flag specifies the vision encoder model.
The `<image>` token in the prompt is the placeholder that gets replaced by
576 visual embeddings during inference.

### 29.5.3  LLaVA Server Mode

```bash
llama-server \
    -m llava-v1.6-mistral-7b.Q4_K_M.gguf \
    --mmproj mmproj-model-f16.gguf \
    -ngl 33 \
    -c 8192 \
    --flash-attn \
    -np 2 \
    --port 8080

# Send an image request
curl http://localhost:8080/completion \
  -F 'json={"prompt": "USER: <image>\nWhat is in this image?\nASSISTANT:", "n_predict": 256}' \
  -F "image=@photo.jpg"
```

### 29.5.4  Memory Breakdown for LLaVA on llama.cpp

For LLaVA-1.6 Mistral 7B Q4_K_M on an RTX 4090 (24 GB):

```
LLM weights (Q4_K_M):     ~4.4 GB
CLIP ViT-L/14 (F16):       ~0.6 GB
Projection (F16):           ~0.02 GB
KV cache (4096 tokens):    ~1.0 GB
Image preprocessing:        ~0.2 GB
CUDA overhead:              ~0.5 GB
─────────────────────────────────
Total:                     ~6.7 GB (comfortably within 24 GB)
```

For high-resolution (LLaVA-1.6 style 4-tile) with 2880 visual tokens:
```
KV for visual tokens:       ~0.36 GB (360 MB, from 28.2)
KV for text (512 tokens):   ~0.06 GB
Total KV per request:       ~0.42 GB
Fits 42× concurrently with one RTX 4090 KV budget
```

---

## 29.6  Python Demo: Multimodal Pipeline Simulator

```python
# multimodal_demo.py
"""
Chapter 29 — Multimodal Inference Demo (Python)

Simulates the full VLM pipeline:
  1. Image patch extraction and token count calculation
  2. Memory cost analysis across VLM families
  3. Tile strategy simulation (LLaVA-1.6 style)
  4. Batch throughput modeling for vision workloads
  5. Audio (Whisper) pipeline overview
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PATCH EXTRACTION AND TOKEN COUNT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisionEncoderSpec:
    name: str
    patch_size: int      # pixels per patch (one side)
    input_resolution: int  # native encoder input size (square)
    output_dim: int      # embedding dimension
    n_transformer_layers: int
    params_m: float      # million parameters


@dataclass
class VLMSpec:
    name: str
    encoder: VisionEncoderSpec
    projection: str       # "linear" | "mlp" | "pixel_shuffle" | "cross_attention"
    compression_ratio: float  # patch compression (1.0 = no compression)
    llm_layers: int
    llm_kv_heads: int
    llm_head_dim: int
    llm_dim: int
    max_tiles: int        # max sub-image tiles
    thumbnail: bool       # include global thumbnail
    max_resolution: int   # maximum input resolution (one side)


CLIP_VIT_L_336 = VisionEncoderSpec(
    name="CLIP ViT-L/14@336",
    patch_size=14, input_resolution=336,
    output_dim=1024, n_transformer_layers=24, params_m=307,
)

SIGLIP_SO400M = VisionEncoderSpec(
    name="SigLIP ViT-SO400M",
    patch_size=14, input_resolution=384,
    output_dim=1152, n_transformer_layers=27, params_m=400,
)

INTERN_VIT_300M = VisionEncoderSpec(
    name="InternViT-300M",
    patch_size=14, input_resolution=448,
    output_dim=1024, n_transformer_layers=24, params_m=300,
)

VLM_FAMILY = [
    VLMSpec("LLaVA-1.5-7B",   CLIP_VIT_L_336,  "mlp",           1.0, 32, 8, 128, 4096, 1,  False, 336),
    VLMSpec("LLaVA-1.6-7B",   CLIP_VIT_L_336,  "pixel_shuffle", 1.0, 32, 8, 128, 4096, 4,  True,  672),
    VLMSpec("InternVL2-8B",   INTERN_VIT_300M, "pixel_shuffle", 4.0, 32, 8, 128, 4096, 12, True,  4032),
    VLMSpec("Qwen2-VL-7B",    SIGLIP_SO400M,   "mlp",           1.0, 28, 4, 128, 3584, 1,  False, 2048),
    VLMSpec("MiniCPM-V-2.6",  SIGLIP_SO400M,   "pixel_shuffle", 9.0, 32, 8, 128, 4096, 9,  True,  1800),
]


def patches_per_tile(encoder: VisionEncoderSpec) -> int:
    """Number of patches from one encoder tile."""
    n = encoder.input_resolution // encoder.patch_size
    return n * n


def visual_tokens_for_image(
    vlm: VLMSpec, image_w: int, image_h: int
) -> Tuple[int, int, str]:
    """
    Returns (n_visual_tokens, n_tiles, description).
    """
    enc = vlm.encoder
    base_patches = patches_per_tile(enc)
    tokens_per_tile = int(base_patches / vlm.compression_ratio)

    if vlm.max_tiles == 1:
        # Fixed resolution
        n_tiles = 1
        desc = f"fixed {enc.input_resolution}×{enc.input_resolution}"
    else:
        # Dynamic tiling
        max_dim = max(image_w, image_h)
        effective_res = min(max_dim, vlm.max_resolution)
        # Number of tiles in each dimension
        tiles_w = max(1, round(image_w / enc.input_resolution))
        tiles_h = max(1, round(image_h / enc.input_resolution))
        # Cap to max_tiles
        n_content_tiles = min(tiles_w * tiles_h, vlm.max_tiles)
        n_tiles = n_content_tiles + (1 if vlm.thumbnail else 0)
        desc = f"{tiles_w}×{tiles_h} tiles + {'thumbnail' if vlm.thumbnail else 'no thumbnail'}"

    total_tokens = n_tiles * tokens_per_tile
    return total_tokens, n_tiles, desc


def kv_bytes_for_visual_tokens(vlm: VLMSpec, n_tokens: int, dtype_bytes: int = 2) -> int:
    return 2 * vlm.llm_layers * vlm.llm_kv_heads * vlm.llm_head_dim * n_tokens * dtype_bytes


def print_token_table():
    test_images = [
        (336, 336,   "Square thumbnail"),
        (672, 672,   "Standard HD"),
        (1024, 768,  "Landscape photo"),
        (768, 1024,  "Portrait photo"),
        (1920, 1080, "Full HD screenshot"),
        (2048, 2048, "High-res scan"),
    ]

    print("=" * 90)
    print("Visual Token Count and KV Cost by VLM and Image Resolution")
    print("=" * 90)

    for img_w, img_h, img_desc in test_images:
        print(f"\n  Image: {img_w}×{img_h}  ({img_desc})")
        print(f"  {'Model':22}  {'Tokens':8}  {'KV MB':8}  {'Tiling strategy'}")
        print("  " + "-" * 68)
        for vlm in VLM_FAMILY:
            n_tok, n_tiles, desc = visual_tokens_for_image(vlm, img_w, img_h)
            kv_mb = kv_bytes_for_visual_tokens(vlm, n_tok) / 1e6
            print(f"  {vlm.name:22}  {n_tok:8,}  {kv_mb:7.1f}M  {desc}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREFILL COST MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModalityPrefillCost:
    """Prefill cost breakdown for one multimodal request."""
    vision_encode_ms: float    # ViT forward pass
    projection_ms: float       # projection layer
    llm_prefill_ms: float      # LLM processing visual + text tokens
    total_ttft_ms: float       # sum


def estimate_prefill_cost(
    vlm: VLMSpec,
    n_visual_tokens: int,
    n_text_tokens: int,
    h100_tflops: float = 989.0,
) -> ModalityPrefillCost:
    """
    Rough estimate of prefill wall time on H100.
    Vision encoder: ~2 FLOPs/param per forward pass
    LLM: ~2 × param_count FLOPs per token
    """
    enc = vlm.encoder

    # Vision encoder FLOPs: ViT with n_transformer_layers attention blocks
    # Each block: ~4 * d² * n_patches (self-attention) + ~8 * d² * n_patches (FFN)
    n_patches = patches_per_tile(enc)
    vis_flops  = enc.n_transformer_layers * 12 * (enc.output_dim ** 2) * n_patches
    vis_ms     = vis_flops / (h100_tflops * 1e12) * 1e3

    # Projection: 2-layer MLP of dim d_vis → d_llm
    proj_flops = 2 * n_visual_tokens * enc.output_dim * vlm.llm_dim
    proj_ms    = proj_flops / (h100_tflops * 1e12) * 1e3

    # LLM prefill: 2 × n_params × n_tokens FLOPs (approx)
    # Llama-3.1-8B has ~8B params
    llm_params  = vlm.llm_layers * 4 * (vlm.llm_dim ** 2)  # rough (attention + FFN)
    total_tokens = n_visual_tokens + n_text_tokens
    llm_flops   = 2 * llm_params * total_tokens
    llm_ms      = llm_flops / (h100_tflops * 1e12) * 1e3

    total_ms = vis_ms + proj_ms + llm_ms
    return ModalityPrefillCost(vis_ms, proj_ms, llm_ms, total_ms)


def print_prefill_breakdown():
    vlm = VLM_FAMILY[1]  # LLaVA-1.6-7B
    image_configs = [
        (576,  512, "LLaVA-1.5 style (1 tile, 512 text)"),
        (2880, 512, "LLaVA-1.6 HD (4 tiles + thumb, 512 text)"),
        (2880, 128, "LLaVA-1.6 HD (4 tiles, short prompt)"),
        (576,  0,   "Image-only (no text, 576 visual tokens)"),
    ]

    print("=" * 75)
    print("Prefill Cost Breakdown (H100, LLaVA-1.6-7B BF16)")
    print("=" * 75)
    print(f"  {'Config':42}  {'ViT':8}  {'Proj':6}  {'LLM':8}  {'TTFT':8}")
    print("  " + "-" * 74)
    for n_vis, n_text, desc in image_configs:
        cost = estimate_prefill_cost(vlm, n_vis, n_text)
        print(f"  {desc:42}  {cost.vision_encode_ms:6.1f}ms  "
              f"{cost.projection_ms:4.1f}ms  "
              f"{cost.llm_prefill_ms:6.1f}ms  "
              f"{cost.total_ttft_ms:6.1f}ms")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TILE STRATEGY SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def optimal_tiling(
    img_w: int, img_h: int,
    tile_size: int,
    max_tiles: int,
) -> Tuple[int, int, int]:
    """
    Find the tiling (tiles_w, tiles_h) that best preserves aspect ratio
    without exceeding max_tiles.
    Returns (tiles_w, tiles_h, total_tiles).
    """
    aspect = img_w / img_h
    best = (1, 1)
    best_score = float("inf")

    for tw in range(1, max_tiles + 1):
        for th in range(1, max_tiles + 1):
            if tw * th > max_tiles:
                continue
            ratio = tw / th
            score = abs(ratio - aspect)
            if score < best_score:
                best_score = score
                best = (tw, th)

    tw, th = best
    return tw, th, tw * th


def print_tiling_table():
    print("=" * 65)
    print("Optimal Tiling Strategy (LLaVA-1.6, max_tiles=4)")
    print("=" * 65)

    images = [
        (336, 336),
        (672, 336),
        (336, 672),
        (1024, 576),
        (576, 1024),
        (800, 600),
        (1920, 1080),
        (1280, 720),
    ]

    print(f"  {'Image':14}  {'Aspect':8}  {'Tiling':10}  {'Tiles':6}  "
          f"{'Tokens (w/ thumb)':20}")
    print("  " + "-" * 62)
    for w, h in images:
        tw, th, n_tiles = optimal_tiling(w, h, 336, 4)
        aspect = w / h
        with_thumb = (n_tiles + 1) * 576  # 576 per tile, +1 thumbnail
        print(f"  {w}×{h:4}        {aspect:8.2f}  {tw}×{th:<7}   "
              f"{n_tiles:3}    {with_thumb:,}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  AUDIO PIPELINE OVERVIEW (WHISPER)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WhisperSpec:
    name: str
    n_encoder_layers: int
    n_decoder_layers: int
    d_model: int
    n_heads: int
    params_m: float
    languages: int
    wer_librispeech: float  # word error rate on LibriSpeech clean


WHISPER_MODELS = [
    WhisperSpec("tiny",   4,  4,  384, 6,   39,   99, 5.7),
    WhisperSpec("base",   6,  6,  512, 8,   74,   99, 4.2),
    WhisperSpec("small",  12, 12, 768, 12,  244,  99, 3.0),
    WhisperSpec("medium", 24, 24, 1024, 16, 769,  99, 2.0),
    WhisperSpec("large",  32, 32, 1280, 20, 1550, 99, 2.7),
    WhisperSpec("large-v3", 32, 32, 1280, 20, 1550, 100, 2.0),
    WhisperSpec("turbo",  32, 2,  1280, 20, 809,  99, 2.7),
]


def whisper_audio_to_tokens(audio_seconds: float) -> int:
    """
    Whisper processes audio in 30-second chunks.
    Each 30s chunk → 3000 Mel spectrogram frames (25ms hop, 25ms window).
    These are processed by the audio encoder as a fixed sequence of 1500 tokens
    (after 2× temporal downsampling in the encoder).
    """
    n_chunks = math.ceil(audio_seconds / 30.0)
    return n_chunks * 1500  # encoder output tokens per chunk


def print_whisper_overview():
    print("=" * 70)
    print("Whisper Audio Encoder Overview")
    print("=" * 70)
    print(f"  {'Model':12}  {'Params':8}  {'Enc L':6}  {'Dec L':6}  "
          f"{'d_model':8}  {'WER%':6}")
    print("  " + "-" * 58)
    for m in WHISPER_MODELS:
        print(f"  {m.name:12}  {m.params_m:7.0f}M  {m.n_encoder_layers:6}  "
              f"{m.n_decoder_layers:6}  {m.d_model:8}  {m.wer_librispeech:6.1f}")

    print()
    print("  Audio processing pipeline:")
    print("  1. Resample to 16kHz mono")
    print("  2. Compute 80-band log Mel spectrogram: 30s → 3000 frames × 80 bins")
    print("  3. Conv encoder: stride-2 convolutions → 1500 frames × d_model")
    print("  4. Transformer encoder: 1500 × d_model → 1500 encoder outputs")
    print("  5. Decoder cross-attends to encoder outputs, autoregressively generates text")
    print()

    print("  Tokens produced by Whisper encoder for various audio lengths:")
    for secs in [5, 10, 30, 60, 120, 300]:
        tok = whisper_audio_to_tokens(secs)
        chunks = math.ceil(secs / 30)
        print(f"  {secs:4}s  →  {chunks} chunk(s)  →  {tok:,} encoder tokens")
    print()

    print("  Integration with VLMs:")
    print("  Audio features from Whisper encoder feed into an audio projector")
    print("  (similar to vision projector) that maps 1500 × d_whisper → N × d_llm")
    print("  Examples: Qwen-Audio (1500 tokens/30s), InternOmni, MiniCPM-o 3.0")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MULTI-MODAL THROUGHPUT ADVISOR
# ─────────────────────────────────────────────────────────────────────────────

def advise_vlm_deployment(
    vlm: VLMSpec,
    hbm_gb: float,
    img_w: int, img_h: int,
    n_text_tokens: int,
    n_output_tokens: int,
    target_concurrency: int,
) -> None:
    print("=" * 65)
    print("VLM Deployment Advisor")
    print("=" * 65)

    n_vis, n_tiles, tile_desc = visual_tokens_for_image(vlm, img_w, img_h)
    total_input  = n_vis + n_text_tokens
    total_tokens = total_input + n_output_tokens

    # Estimate model weight size (rough: llm 8B + encoder 300M ≈ 8.3B params)
    llm_params_b = (vlm.llm_layers * 4 * vlm.llm_dim * vlm.llm_dim) / 1e9
    enc_params_b = vlm.encoder.params_m / 1e3
    total_params_b = llm_params_b + enc_params_b
    weights_gb   = total_params_b * 2  # BF16

    kv_per_req   = kv_bytes_for_visual_tokens(vlm, total_tokens) / 1e9
    total_kv_gb  = kv_per_req * target_concurrency
    total_needed = weights_gb + total_kv_gb
    usable       = hbm_gb * 0.90
    fits         = total_needed <= usable
    max_conc     = int((usable - weights_gb) / kv_per_req) if kv_per_req > 0 else 0

    print(f"  VLM:         {vlm.name}")
    print(f"  Hardware:    {hbm_gb:.0f} GB GPU")
    print(f"  Image:       {img_w}×{img_h} → {n_vis:,} visual tokens  ({tile_desc})")
    print(f"  Text tokens: {n_text_tokens}")
    print(f"  Output:      {n_output_tokens}")
    print(f"  Total/req:   {total_tokens:,} tokens")
    print()
    print(f"  Weights:     {weights_gb:.1f} GB")
    print(f"  KV/request:  {kv_per_req*1000:.0f} MB")
    print(f"  KV total:    {total_kv_gb*1000:.0f} MB  ({target_concurrency} users)")
    print(f"  Required:    {total_needed:.1f} GB  (usable: {usable:.1f} GB)")
    print()
    if fits:
        print(f"  ✓  Fits. Max concurrency at this resolution: {max_conc}")
    else:
        print(f"  ✗  Exceeds budget. Max feasible concurrency: {max_conc}")
        print(f"     Consider: fewer tiles, smaller encoder (MiniCPM-V), FP8 KV, INT4 weights")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Chapter 29 — Multimodal Inference Demo (Python)")
    print("=" * 70 + "\n")

    print_token_table()
    print_prefill_breakdown()
    print_tiling_table()
    print_whisper_overview()

    advise_vlm_deployment(
        vlm=VLM_FAMILY[1],   # LLaVA-1.6-7B
        hbm_gb=80.0,
        img_w=1024, img_h=768,
        n_text_tokens=512,
        n_output_tokens=256,
        target_concurrency=32,
    )
```

---

## 29.7  C++ Implementation: Vision Pipeline Demo

```cpp
// multimodal_demo.cpp
// Chapter 29 — Multimodal Inference Demo (C++)
//
// Demonstrates without external dependencies:
//   1. Patch extraction geometry and token count
//   2. Vision encoder FLOP estimation
//   3. Tile strategy optimization (aspect-ratio matching)
//   4. KV cache cost for visual tokens
//   5. Memory budget planning for VLMs
//   6. Whisper audio token calculation
//
// Build: g++ -O2 -std=c++17 -o multimodal_demo multimodal_demo.cpp
// Run:   ./multimodal_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

static void print_section(const std::string& title) {
    std::string bar(72, '-');
    std::cout << "\n" << bar << "\n  " << title << "\n" << bar << "\n";
}

static std::string comma(long long n) {
    if (n < 0) return "-" + comma(-n);
    if (n < 1000) return std::to_string(n);
    return comma(n / 1000) + "," + [](long long r){
        char buf[8]; std::snprintf(buf, sizeof(buf), "%03lld", r);
        return std::string(buf);
    }(n % 1000);
}

// ─────────────────────────────────────────────────────────────────────────────
// SPECS
// ─────────────────────────────────────────────────────────────────────────────

struct EncoderSpec {
    const char* name;
    int   patch_px;          // patch side in pixels
    int   input_res;         // encoder input resolution
    int   output_dim;        // embedding dimension
    int   n_layers;          // transformer blocks
    double params_m;         // millions of params
};

struct VLMSpec {
    const char* name;
    EncoderSpec enc;
    double compression;      // patch reduction factor (>1 means fewer tokens)
    int    llm_layers;
    int    llm_kv_heads;
    int    llm_head_dim;
    int    llm_dim;
    int    max_tiles;        // content tiles
    bool   thumbnail;        // add global thumbnail tile
    int    max_resolution;   // max input side in pixels
};

static const EncoderSpec CLIP_L  = {"CLIP ViT-L/14@336", 14, 336, 1024, 24, 307};
static const EncoderSpec SIGLIP  = {"SigLIP ViT-SO400M", 14, 384, 1152, 27, 400};
static const EncoderSpec INTERN  = {"InternViT-300M",    14, 448, 1024, 24, 300};

static const VLMSpec VLMS[] = {
    {"LLaVA-1.5-7B", CLIP_L,  1.0, 32, 8, 128, 4096, 1, false, 336},
    {"LLaVA-1.6-7B", CLIP_L,  1.0, 32, 8, 128, 4096, 4, true,  672},
    {"InternVL2-8B", INTERN,  4.0, 32, 8, 128, 4096, 12,true,  4032},
    {"Qwen2-VL-7B",  SIGLIP,  1.0, 28, 4, 128, 3584, 1, false, 2048},
    {"MiniCPM-V-2.6",SIGLIP,  9.0, 32, 8, 128, 4096, 9, true,  1800},
};
static const int N_VLMS = 5;

// ─────────────────────────────────────────────────────────────────────────────
// 1.  PATCH GEOMETRY
// ─────────────────────────────────────────────────────────────────────────────

static int patches_per_tile(const EncoderSpec& e) {
    int n = e.input_res / e.patch_px;
    return n * n;
}

static int tokens_per_tile(const VLMSpec& v) {
    return static_cast<int>(patches_per_tile(v.enc) / v.compression);
}

// Returns (n_content_tiles, total_tiles_with_thumb, visual_tokens)
static std::tuple<int,int,int> visual_tokens(const VLMSpec& v, int img_w, int img_h) {
    if (v.max_tiles == 1) {
        int total = 1 + (v.thumbnail ? 1 : 0);
        return {1, total, total * tokens_per_tile(v)};
    }
    int tw = std::max(1, (int)std::round((double)img_w / v.enc.input_res));
    int th = std::max(1, (int)std::round((double)img_h / v.enc.input_res));
    int content = std::min(tw * th, v.max_tiles);
    int total   = content + (v.thumbnail ? 1 : 0);
    return {content, total, total * tokens_per_tile(v)};
}

static long long kv_bytes(const VLMSpec& v, int n_tokens, int dtype_bytes = 2) {
    return 2LL * v.llm_layers * v.llm_kv_heads * v.llm_head_dim * n_tokens * dtype_bytes;
}

static void demo_patch_geometry() {
    print_section("Patch Geometry and Visual Token Count");

    struct Img { int w, h; const char* desc; };
    const Img images[] = {
        {336,  336,  "Square thumbnail"},
        {672,  672,  "Standard HD"},
        {1024, 768,  "Landscape photo"},
        {768,  1024, "Portrait photo"},
        {1920, 1080, "Full HD screenshot"},
    };
    const int N_IMG = 5;

    for (int ii = 0; ii < N_IMG; ++ii) {
        auto& img = images[ii];
        std::cout << "\n  Image " << img.w << "×" << img.h << "  (" << img.desc << ")\n";
        std::cout << "  " << std::left << std::setw(22) << "Model"
                  << std::right << std::setw(9) << "Tokens"
                  << std::setw(12) << "KV (MB)"
                  << std::setw(8) << "Tiles"
                  << "\n  " << std::string(51, '-') << "\n";

        for (int vi = 0; vi < N_VLMS; ++vi) {
            auto& v = VLMS[vi];
            auto [content, total, ntok] = visual_tokens(v, img.w, img.h);
            double kv_mb = kv_bytes(v, ntok) / 1e6;
            std::cout << "  " << std::left << std::setw(22) << v.name
                      << std::right << std::setw(9) << comma(ntok)
                      << std::setw(10) << std::fixed << std::setprecision(1) << kv_mb << " MB"
                      << std::setw(8) << total
                      << "\n";
        }
    }
    std::cout << "\n";

    // Assertions
    // LLaVA-1.5: always 576 tokens (1 tile, no thumb, compression=1)
    auto [c1, t1, n1] = visual_tokens(VLMS[0], 1024, 768);
    assert(n1 == 576);
    std::cout << "  [ASSERT] LLaVA-1.5 always produces 576 tokens: " << n1 << " ✓\n";

    // InternVL2: compression=4, so 448/14=32 → 1024 patches / 4 = 256 per tile
    assert(tokens_per_tile(VLMS[2]) == 256);
    std::cout << "  [ASSERT] InternVL2 tokens/tile = 256 (1024 patches ÷ 4): ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  VISION ENCODER FLOP ESTIMATION
// ─────────────────────────────────────────────────────────────────────────────

static double encoder_flops(const EncoderSpec& e) {
    // Per transformer block: ~12 × d² × N FLOPs (self-attn + FFN)
    int N = patches_per_tile(e);
    return (double)e.n_layers * 12.0 * (double)(e.output_dim * e.output_dim) * N;
}

static void demo_encoder_flops() {
    print_section("Vision Encoder FLOP Budget");

    const EncoderSpec encoders[] = {CLIP_L, SIGLIP, INTERN};
    const int N = 3;
    const double H100_TFLOPS = 989.0;

    std::cout << "\n  " << std::left << std::setw(24) << "Encoder"
              << std::right << std::setw(14) << "FLOPs"
              << std::setw(14) << "Time (H100)"
              << std::setw(12) << "Patches"
              << "\n  " << std::string(64, '-') << "\n";

    for (int i = 0; i < N; ++i) {
        auto& e = encoders[i];
        double flops = encoder_flops(e);
        double ms    = flops / (H100_TFLOPS * 1e12) * 1e3;
        int    pats  = patches_per_tile(e);
        std::cout << "  " << std::left << std::setw(24) << e.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2)
                  << flops / 1e9 << " GF"
                  << std::setw(12) << std::setprecision(3) << ms << " ms"
                  << std::setw(12) << pats
                  << "\n";
    }

    // Multi-tile: 4 tiles + 1 thumbnail = 5 passes
    double total_5tile = 5 * encoder_flops(CLIP_L);
    double ms_5tile    = total_5tile / (H100_TFLOPS * 1e12) * 1e3;
    std::cout << "\n  LLaVA-1.6 HD (5 CLIP passes): "
              << std::setprecision(2) << ms_5tile << " ms on H100\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  OPTIMAL TILING
// ─────────────────────────────────────────────────────────────────────────────

static std::tuple<int,int,int> best_tiling(int img_w, int img_h,
                                            int tile_px, int max_tiles) {
    double aspect = (double)img_w / img_h;
    int best_tw = 1, best_th = 1;
    double best_score = 1e18;

    for (int tw = 1; tw <= max_tiles; ++tw)
        for (int th = 1; th <= max_tiles; ++th) {
            if (tw * th > max_tiles) continue;
            double score = std::abs((double)tw / th - aspect);
            if (score < best_score) {
                best_score = score;
                best_tw = tw; best_th = th;
            }
        }
    return {best_tw, best_th, best_tw * best_th};
}

static void demo_tiling() {
    print_section("Optimal Tile Selection (LLaVA-1.6, max_tiles=4, tile=336px)");

    struct Img { int w, h; };
    const Img imgs[] = {
        {336, 336}, {672, 336}, {336, 672},
        {1024, 576}, {576, 1024}, {800, 600}, {1920, 1080},
    };
    const int N = 7;

    std::cout << "\n  " << std::left << std::setw(14) << "Image"
              << std::right << std::setw(10) << "Aspect"
              << std::setw(10) << "Tiling"
              << std::setw(8) << "Tiles"
              << std::setw(20) << "Tokens (w/ thumb)"
              << "\n  " << std::string(62, '-') << "\n";

    for (int i = 0; i < N; ++i) {
        auto& img = imgs[i];
        auto [tw, th, n] = best_tiling(img.w, img.h, 336, 4);
        double aspect = (double)img.w / img.h;
        int tokens = (n + 1) * 576;  // +1 for thumbnail

        std::ostringstream res, til;
        res << img.w << "×" << img.h;
        til << tw << "×" << th;

        std::cout << "  " << std::left << std::setw(14) << res.str()
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << aspect
                  << std::setw(10) << til.str()
                  << std::setw(8) << n
                  << std::setw(20) << comma(tokens)
                  << "\n";
    }

    // Assert: square image tiles as 1×1 or 2×2
    auto [tw_sq, th_sq, n_sq] = best_tiling(672, 672, 336, 4);
    assert(tw_sq == th_sq);  // aspect 1.0 → square tiling
    std::cout << "\n  [ASSERT] Square image → square tiling (" << tw_sq << "×" << th_sq << "): ✓\n";

    // Assert: landscape 16:9 tiles as 2×1
    auto [tw_ls, th_ls, n_ls] = best_tiling(1920, 1080, 336, 4);
    assert(tw_ls > th_ls);  // wider than tall
    std::cout << "  [ASSERT] Landscape image → wider tiling (" << tw_ls << "×" << th_ls << "): ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  KV CACHE BUDGET FOR VLMs
// ─────────────────────────────────────────────────────────────────────────────

static void demo_kv_budget() {
    print_section("KV Cache Budget — H100 80GB VLM Deployment");

    const double HBM_GB      = 80.0;
    const double OVERHEAD    = 0.10;
    const double USABLE      = HBM_GB * (1.0 - OVERHEAD);

    // Rough weight estimates: 8B model BF16 ≈ 16GB + 300-400M encoder ≈ 0.6GB
    const double WEIGHTS_GB  = 16.6;
    const double AVAIL_KV_GB = USABLE - WEIGHTS_GB;

    // Request profile: 1 image (1024×768) + 512 text + 256 output
    const int IMG_W = 1024, IMG_H = 768;
    const int TEXT_TOK = 512, OUT_TOK = 256;

    std::cout << "\n  H100 usable HBM: " << USABLE << " GB\n"
              << "  Weights (LLM + encoder BF16): " << WEIGHTS_GB << " GB\n"
              << "  Available for KV: " << AVAIL_KV_GB << " GB\n\n";

    std::cout << "  " << std::left << std::setw(22) << "VLM"
              << std::right << std::setw(10) << "Vis tok"
              << std::setw(12) << "KV/req(MB)"
              << std::setw(14) << "Max concurr."
              << "\n  " << std::string(58, '-') << "\n";

    for (int vi = 0; vi < N_VLMS; ++vi) {
        auto& v = VLMS[vi];
        auto [c, t, n_vis] = visual_tokens(v, IMG_W, IMG_H);
        int total_tok = n_vis + TEXT_TOK + OUT_TOK;
        double kv_mb  = kv_bytes(v, total_tok) / 1e6;
        int max_conc  = static_cast<int>(AVAIL_KV_GB * 1e3 / kv_mb);

        std::cout << "  " << std::left << std::setw(22) << v.name
                  << std::right << std::setw(10) << comma(n_vis)
                  << std::setw(12) << std::fixed << std::setprecision(1) << kv_mb
                  << std::setw(14) << max_conc
                  << "\n";
    }
    std::cout << "\n  Request profile: 1024×768 image + 512 text + 256 output tokens\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  WHISPER AUDIO TOKEN CALCULATION
// ─────────────────────────────────────────────────────────────────────────────

struct WhisperSpec {
    const char* name;
    int   enc_layers;
    int   dec_layers;
    int   d_model;
    double params_m;
    double wer_clean;
};

static const WhisperSpec WHISPER[] = {
    {"tiny",     4,  4,  384,  39,  5.7},
    {"base",     6,  6,  512,  74,  4.2},
    {"small",    12, 12, 768,  244, 3.0},
    {"medium",   24, 24, 1024, 769, 2.0},
    {"large-v3", 32, 32, 1280, 1550,2.0},
    {"turbo",    32, 2,  1280, 809, 2.7},
};
static const int N_WHISPER = 6;

static int whisper_encoder_tokens(double audio_s) {
    // 30s chunk → 1500 encoder tokens
    int chunks = static_cast<int>(std::ceil(audio_s / 30.0));
    return chunks * 1500;
}

static double whisper_encode_ms(const WhisperSpec& w, double audio_s,
                                  double H100_TFLOPS = 989.0) {
    int chunks = static_cast<int>(std::ceil(audio_s / 30.0));
    // Each 30s chunk: 3000 Mel frames → 1500 after conv
    // Per block: ~12 × d² × 1500 FLOPs
    double flops_per_chunk = w.enc_layers * 12.0 * (double)(w.d_model * w.d_model) * 1500;
    double total_flops = chunks * flops_per_chunk;
    return total_flops / (H100_TFLOPS * 1e12) * 1e3;
}

static void demo_whisper() {
    print_section("Whisper Audio Encoder — Token Count and Compute");

    std::cout << "\n  Model specs:\n";
    std::cout << "  " << std::left << std::setw(12) << "Model"
              << std::right << std::setw(10) << "Params"
              << std::setw(10) << "Enc L"
              << std::setw(10) << "Dec L"
              << std::setw(10) << "d_model"
              << std::setw(10) << "WER%"
              << "\n  " << std::string(62, '-') << "\n";
    for (int i = 0; i < N_WHISPER; ++i) {
        auto& m = WHISPER[i];
        std::cout << "  " << std::left << std::setw(12) << m.name
                  << std::right << std::setw(8) << std::fixed << std::setprecision(0) << m.params_m << "M"
                  << std::setw(10) << m.enc_layers
                  << std::setw(10) << m.dec_layers
                  << std::setw(10) << m.d_model
                  << std::setw(10) << std::setprecision(1) << m.wer_clean
                  << "\n";
    }

    std::cout << "\n  Audio → encoder tokens and encode time (large-v3, H100):\n";
    std::cout << "  " << std::right << std::setw(10) << "Duration"
              << std::setw(10) << "Chunks"
              << std::setw(14) << "Enc tokens"
              << std::setw(14) << "Encode ms"
              << "\n  " << std::string(48, '-') << "\n";

    const double durations[] = {5, 10, 30, 60, 120, 300};
    const int ND = 6;
    for (int i = 0; i < ND; ++i) {
        double secs = durations[i];
        int chunks  = static_cast<int>(std::ceil(secs / 30.0));
        int enc_tok = whisper_encoder_tokens(secs);
        double ms   = whisper_encode_ms(WHISPER[4], secs);  // large-v3
        std::cout << "  " << std::right << std::setw(8) << std::fixed << std::setprecision(0) << secs << "s"
                  << std::setw(10) << chunks
                  << std::setw(14) << comma(enc_tok)
                  << std::setw(14) << std::setprecision(2) << ms << " ms"
                  << "\n";
    }

    // Assert: 30s → exactly 1500 encoder tokens
    assert(whisper_encoder_tokens(30.0) == 1500);
    std::cout << "\n  [ASSERT] 30s audio → 1500 encoder tokens: ✓\n";

    // Assert: encode time < 100ms for 30s chunk on H100
    double ms_30s = whisper_encode_ms(WHISPER[4], 30.0);
    assert(ms_30s < 100.0);
    std::cout << "  [ASSERT] Encode 30s < 100ms on H100: "
              << std::setprecision(2) << ms_30s << "ms ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n"
              << std::string(72, '=') << "\n"
              << "  Chapter 29 — Multimodal Inference Demo (C++)\n"
              << std::string(72, '=') << "\n";

    demo_patch_geometry();
    demo_encoder_flops();
    demo_tiling();
    demo_kv_budget();
    demo_whisper();

    std::cout << "\n" << std::string(72, '=') << "\n"
              << "  All demos complete.\n"
              << std::string(72, '=') << "\n\n";
    return 0;
}
```

---

## 29.8  Video Inference

Video extends the vision pipeline in one dimension: time.

### 29.8.1  Frame Sampling Strategies

| Strategy | Tokens per second | Use case |
|---|---|---|
| 1 fps, 576 tok/frame | 576 | Short clips (< 30s) |
| 2 fps, 576 tok/frame | 1152 | Action recognition |
| Key-frame extraction | Variable | Long-form analysis |
| Temporal compression | ~100 tok/s | Qwen2-VL video mode |

For a 60-second clip at 1 fps with 576 tokens/frame:
```
60 × 576 = 34,560 visual tokens
KV for Llama-3.1-8B = 34,560 × 128 KB = 4.3 GB
```

### 29.8.2  Temporal Position Encoding

Qwen2-VL extends its 2D-RoPE to 3D: (time, row, column).
Each video frame occupies a position in temporal dimension.
Frames at different timestamps get different temporal RoPE phases.

This allows the model to answer questions like "what happened between the 10th and 20th
second?" by attending to temporally-anchored tokens.

### 29.8.3  vLLM Video Config

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=10,video=2" \
    --gpu-memory-utilization 0.90
```

Video request format:
```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "file:///path/to/clip.mp4",
                    "max_pixels": 360 * 420,  # cap token count
                    "fps": 1.0,
                },
            },
            {"type": "text", "text": "What is happening in this video?"},
        ],
    }],
    max_tokens=256,
)
```

---

## 29.9  Production Patterns and Pitfalls

### 29.9.1  Image Pre-processing Pipeline

Do not let the inference server decode and resize JPEGs.
Pre-processing adds significant latency that sits before the GPU pipeline:

```python
# Bad: pass raw JPEG bytes and let the server decode
response = client.chat.completions.create(
    ...,
    image_url={"url": f"data:image/jpeg;base64,{raw_jpeg_b64}"}
)

# Good: pre-process in your application server before sending
from PIL import Image
import io

def preprocess_for_vllm(image_bytes: bytes) -> str:
    """Resize to model's expected tile dimensions and re-encode."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # For LLaVA-1.6: resize to one of the valid tile configurations
    img = img.resize((672, 672), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode()
```

### 29.9.2  Token Budget Enforcement

Visual tokens consume context budget.
A request with 4 tiles + text that exceeds `max-model-len` will be rejected with an OOM
error rather than gracefully truncated (unlike pure text which can be truncated from the left).

Always enforce an upstream token budget before sending:
```python
MAX_VISUAL_TOKENS = 1500
MAX_TEXT_TOKENS   = 512

def validate_request(n_tiles: int, n_text: int) -> bool:
    n_vis = n_tiles * 576  # LLaVA-1.6
    return (n_vis + n_text) <= MAX_VISUAL_TOKENS + MAX_TEXT_TOKENS
```

### 29.9.3  CLIP Encoder Cache

For workloads where the same image is queried multiple times (e.g., a product catalog
image asked about with different questions), cache the CLIP output embeddings:

```python
import hashlib, functools, numpy as np

@functools.lru_cache(maxsize=1024)
def encode_image(image_hash: str, image_b64: str) -> np.ndarray:
    """Cache CLIP embeddings by image hash."""
    # In production: call vision encoder directly, not full VLM
    ...

def get_image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()
```

vLLM 0.6+ supports visual prefix caching via `--enable-prefix-caching` (see §29.4.5).
For models with cross-attention vision (Llama 3.2 Vision), the visual KV buffer is per-request
and cannot be prefix-shared — application-level caching of the CLIP encoder output is the
correct approach there.

For prefix-injection models (LLaVA, Qwen-VL, Pixtral), rely on vLLM's built-in visual
prefix caching rather than application-level solutions when possible — vLLM's block-sharing
is zero-copy and does not require serialising/deserialising embedding tensors.

### 29.9.4  Aspect Ratio and Distortion

Always preserve aspect ratio when pre-processing images.
Stretching a 1920×1080 screenshot to 336×336 (LLaVA-1.5's required input) distorts text
and charts significantly.
Use a padding strategy (letterbox) or dynamic tiling (LLaVA-1.6, InternVL2) instead.

### 29.9.5  llama.cpp LLaVA Quantization Mismatch

The CLIP vision encoder in llama.cpp LLaVA is almost always F16 (the `mmproj` file).
If you quantize the LLM to Q4_K_M but serve the F16 CLIP, the memory split is:

- LLM: ~4.4 GB (Q4_K_M)
- CLIP: ~0.6 GB (F16)

The CLIP model cannot currently be further quantized without significant quality
degradation (visual features are more sensitive to quantization than LLM weights).
This is an active area of research; F8 CLIP is being explored.

---

## 29.10  Summary

Multimodal inference adds a new cost center — visual tokens — that consumes context
budget and KV cache exactly like text tokens.
The token count per image ranges from 64 (heavily compressed) to 2880 (4-tile HD),
and each token costs as much KV memory as a text token.

The vision pipeline is always three stages: patch extraction → vision encoder → projection.
Understanding the patch geometry and tile count for a given image resolution lets you
predict latency and memory cost exactly.

For production: pre-process images before sending to the inference server, enforce token
budgets upstream, cache CLIP embeddings for repeated images, and choose a model whose
tile strategy matches your typical image resolution.

---

*Chapter 30 covers semantic caching and response reuse: how to detect semantically
equivalent requests, cache responses at the application layer, and dramatically reduce
inference costs for repetitive workloads.*


---

## Chapter Summary

- **Vision encoder integration**: multimodal LLMs prepend image features to the token sequence; the vision encoder (ViT-based) produces a fixed number of visual tokens per image.
- **Visual token count**: CLIP ViT-L/14 at 336×336 produces 576 visual tokens; at decode step, these tokens occupy the KV cache identically to text tokens.
- **vLLM multimodal API**: POST to `/v1/chat/completions` with `content: [{type: image_url, image_url: {url: ...}}, {type: text, text: ...}]`; vLLM handles encoding internally.
- **Image caching**: because images are deterministic (same image → same visual tokens), vLLM's prefix caching can cache the KV blocks for the visual token prefix.
- **Audio and video modalities**: audio uses a Whisper-style encoder; video uses frame sampling + ViT; both produce token sequences of length proportional to input duration/frame count.
- **Memory pressure**: a single high-res image can produce 2 304+ visual tokens; a batch of 32 image requests consumes 74K+ visual tokens in the KV cache.
- **llama.cpp multimodal**: `llava-cli` and `llama-server` with `--mmproj` flag load the multimodal projector; images are encoded on CPU and prepended to the prompt.

---

## Self-Check Questions

1. LLaVA-1.5 uses CLIP ViT-L/14 at 336×336 input, producing 576 visual tokens. A user sends a batch of 16 image+text requests. Assuming each image is unique and text prompts are 128 tokens, compute the total prefill tokens and KV cache bytes at LLaMA-3 8B (32 layers, 32 KV heads, d_k = 128, BF16). *(Section 29.2)*

2. If all 16 images in question 1 are identical (same URL), how does vLLM's prefix caching reduce the prefill FLOPs and KV cache? *(Section 29.4)*

3. A video model samples 1 frame/second for a 60-second clip, encoding each frame as 256 visual tokens. Total visual tokens = 15 360. With `--max-model-len 16384`, can this fit? What configuration change is needed? *(Section 29.3)*

4. The multimodal projector maps ViT embeddings from the CLIP embedding space to the LLM embedding space. Why can't you simply concatenate raw pixel values to the text embedding instead? *(Section 29.1)*

5. `llama-server --mmproj llava-v1.5-7b-mmproj-f16.gguf` runs image encoding on CPU. For a batch of 8 images (576 tokens each), estimate the encoding latency on a modern CPU vs on a GPU, and explain why the projector is CPU-resident by default in llama.cpp. *(Section 29.5)*
