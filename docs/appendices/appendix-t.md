# Appendix T — Embedding and Reranker Model Serving

> *"Retrieval-augmented generation is only as fast as its slowest component — which is rarely the LLM."*

---

## T.1 Why Embedding Serving Is Different

Embedding models and reranker models are not generative. They do not produce
tokens one by one; they do not maintain a KV cache that grows with sequence
length; they do not require sampling. They are **encoder-only** (or
encoder-decoder) models whose output is a fixed-size vector or a single scalar.

This changes everything about how you serve them. The KV cache management
strategies of Chapters 6–11 are irrelevant. The batching principles of Chapter 3
apply in a modified form. The performance bottlenecks are different. The
hardware requirements are different.

Yet embedding and reranker models are integral to virtually every
production LLM application. A RAG pipeline calls an embedding model for every
document chunk at indexing time and for every user query at inference time. A
reranker model scores every retrieved candidate before the LLM generates its
response. Serving these models efficiently is as important as serving the LLM
itself.

---

## T.2 Encoder Architecture Fundamentals

### T.2.1 The encoder-only forward pass

An encoder model like BERT, BGE, or E5 performs a single forward pass over the
entire input sequence. All positions attend to all other positions
(bidirectional attention), unlike the causal attention in decoder models.

```
Input:  [CLS] token₁ token₂ ... tokenₙ [SEP]
Output: hidden states at every position ∈ R^{n × d_model}
```

There is no autoregressive decode loop. The model runs once, producing a hidden
state for every input token simultaneously.

### T.2.2 Pooling strategies

The embedding for the full input sequence is derived from the per-token hidden
states via a **pooling operation**:

**CLS pooling**: use the hidden state of the `[CLS]` token (position 0):
```python
embedding = hidden_states[:, 0, :]   # [batch, d_model]
```
BERT-style models trained with a classification objective encode sequence-level
information in the CLS token. BGE and many retrieval models use this.

**Mean pooling**: average all token hidden states (excluding padding):
```python
attention_mask_expanded = attention_mask.unsqueeze(-1).float()
embedding = (hidden_states * attention_mask_expanded).sum(1) / \
            attention_mask_expanded.sum(1)
```
E5, GTE, and Nomic-Embed use mean pooling. It tends to be more robust for
asymmetric (query vs document) retrieval.

**Last-token pooling**: use the hidden state of the last non-padding token.
Preferred for decoder-only models fine-tuned as embedding models (e.g.
`nvidia/NV-Embed-v2`, LLM2Vec models):
```python
# Find last non-padding token for each sequence in batch
last_positions = attention_mask.sum(dim=1) - 1
embedding = hidden_states[torch.arange(batch_size), last_positions, :]
```

**Weighted mean pooling**: assign higher weights to tokens near the end or
beginning:
```python
# Linearly increasing weights
weights = torch.arange(1, seq_len + 1, device=hidden_states.device).float()
weights = weights / weights.sum()
embedding = (hidden_states * weights.unsqueeze(0).unsqueeze(-1)).sum(1)
```

### T.2.3 L2 normalisation

For cosine similarity retrieval, embeddings must be L2-normalised:
```python
import torch.nn.functional as F
embedding = F.normalize(embedding, p=2, dim=-1)
# Now: cosine_similarity(a, b) == dot(a, b) == (a * b).sum()
```
After normalisation, maximum inner product search and cosine similarity are
equivalent — enabling FAISS's IVF-PQ index (which optimises inner product) for
cosine retrieval.

---

## T.3 Key Embedding Models

| Model | Parameters | Dimension | Max tokens | Pooling | MTEB Score |
|---|---|---|---|---|---|
| `BAAI/bge-large-en-v1.5` | 335M | 1,024 | 512 | CLS | 64.2 |
| `BAAI/bge-m3` | 570M | 1,024 | 8,192 | CLS | 68.1 |
| `intfloat/e5-large-v2` | 335M | 1,024 | 512 | Mean | 62.2 |
| `intfloat/multilingual-e5-large` | 560M | 1,024 | 512 | Mean | 61.5 |
| `thenlper/gte-large` | 335M | 1,024 | 512 | Mean | 63.1 |
| `Alibaba-NLP/gte-Qwen2-7B-instruct` | 7.6B | 3,584 | 32,768 | Last | 72.1 |
| `nvidia/NV-Embed-v2` | 7.8B | 4,096 | 32,768 | Last | 72.3 |
| `nomic-ai/nomic-embed-text-v1.5` | 137M | 768 | 8,192 | Mean | 62.4 |

**BGE-M3** is the most practically important: it supports hybrid retrieval
(dense + sparse + multi-vector), handles 8K tokens (long documents), and covers
100+ languages.

**LLM-based embedders** (GTE-Qwen2-7B, NV-Embed-v2): decoder-only LLMs
fine-tuned as embedding models via last-token pooling and contrastive training.
They achieve state-of-the-art quality but cost 20–25× more to serve than
BERT-scale models.

---

## T.4 Serving Embedding Models with vLLM

vLLM added embedding model support in v0.4.0. The key difference: instead of
`generate()`, use `encode()`:

```python
from vllm import LLM

# BERT-scale embedding model
embedder = LLM(
    model="BAAI/bge-large-en-v1.5",
    task="embed",                   # critical: embedding mode
    gpu_memory_utilization=0.80,
    max_model_len=512,
)

texts = [
    "Query: What is machine learning?",
    "Passage: Machine learning is a subset of artificial intelligence..."
]

outputs = embedder.encode(texts)
embeddings = [o.outputs.embedding for o in outputs]
# embeddings[0].shape = (1024,)  — L2 normalised
```

Via the OpenAI-compatible endpoint:

```bash
# Start server
vllm serve BAAI/bge-large-en-v1.5 \
    --task embed \
    --max-model-len 512

# Query via API
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-large-en-v1.5",
    "input": ["What is machine learning?", "ML is a subset of AI..."]
  }'
```

### T.4.1 LLM-based embedding models in vLLM

For larger decoder-based embedders:

```bash
# GTE-Qwen2-7B — LLM fine-tuned as embedder
vllm serve Alibaba-NLP/gte-Qwen2-7B-instruct \
    --task embed \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85
```

```python
# Instruction-prefixed queries (required for LLM embedders)
outputs = embedder.encode([
    "Instruct: Given a web search query, retrieve relevant passages\nQuery: How does attention work?",
    "Passage: Attention is a mechanism that allows the model to focus..."
])
```

### T.4.2 Batching for embedding workloads

Embedding models are purely compute-bound (no autoregressive bottleneck).
Larger batches dramatically improve throughput:

```python
# Batch size tuning for embedding throughput
# Optimal batch size = largest that fits in GPU memory
embedder = LLM(
    model="BAAI/bge-large-en-v1.5",
    task="embed",
    max_num_seqs=1024,            # allow large batches
    gpu_memory_utilization=0.90,
)

# Process large dataset efficiently
import asyncio
from tqdm import tqdm

async def embed_dataset(texts: list[str], batch_size: int = 512):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        outputs = embedder.encode(batch)
        embeddings.extend([o.outputs.embedding for o in outputs])
    return embeddings
```

**Throughput benchmark (BAAI/bge-large-en-v1.5, A100 80GB, 128-token sequences):**

| Batch size | Tokens/sec | Embeddings/sec |
|---|---|---|
| 1 | 4,200 | 33 |
| 32 | 48,000 | 375 |
| 128 | 156,000 | 1,219 |
| 512 | 384,000 | 3,000 |
| 1,024 | 420,000 | 3,281 |

Above batch size ~512 the memory bandwidth is saturated. The optimal batch size
is typically `GPU_MEM_AVAILABLE / (max_seq_len × d_model × bytes)`.

---

## T.5 Serving Embedding Models with llama.cpp

llama.cpp supports embedding mode for BERT-compatible models via GGUF format:

```bash
# Generate embeddings (CLI)
llama-embedding \
    --model bge-large-en-v1.5.Q8_0.gguf \
    --prompt "What is machine learning?" \
    --embd-normalize 2 \   # L2 normalisation
    --embd-output-format json

# Server mode (HTTP API)
llama-server \
    --model bge-large-en-v1.5.Q8_0.gguf \
    --embedding \           # enable embedding endpoint
    --port 8080 \
    --ctx-size 512 \
    --batch-size 512

# Query embedding server
curl http://localhost:8080/v1/embeddings \
  -d '{"input": "What is machine learning?", "model": "bge-large-en-v1.5"}'
```

```cpp
// C API for embedding inference
llama_context_params cparams = llama_context_default_params();
cparams.embeddings = true;       // enable embedding output
cparams.n_ubatch = 512;         // batch size for parallel processing

llama_context* ctx = llama_new_context_with_model(model, cparams);

// Tokenise and evaluate
std::vector<llama_token> tokens = tokenize(text);
llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));

// Get embedding
float* embd = llama_get_embeddings_seq(ctx, 0);
// Normalize
float norm = 0;
for (int i = 0; i < n_embd; i++) norm += embd[i] * embd[i];
norm = std::sqrt(norm);
for (int i = 0; i < n_embd; i++) embd[i] /= norm;
```

---

## T.6 Reranker Models

### T.6.1 What rerankers do

A reranker (cross-encoder) takes a **(query, document) pair** as input and
outputs a single relevance score. Unlike embedding models, which encode query
and document separately, the cross-encoder sees both together and uses
attention across all tokens from both:

```
Input:  [CLS] query tokens [SEP] document tokens [SEP]
Output: logit at [CLS] position → sigmoid → relevance score ∈ [0, 1]
```

This joint encoding makes rerankers far more accurate than embedding-based
retrieval for subtle relevance distinctions, at the cost of O(n × m) inference
(n queries × m candidates).

### T.6.2 Key reranker models

| Model | Parameters | Max tokens | Speed (pairs/s, A100) | BEIR nDCG@10 |
|---|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | 568M | 8,192 | 380 | 56.1 |
| `BAAI/bge-reranker-large` | 435M | 512 | 520 | 53.8 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | 512 | 4,200 | 47.6 |
| `mixedbread-ai/mxbai-rerank-large-v1` | 435M | 512 | 510 | 56.2 |
| `jinaai/jina-reranker-v2-base-multilingual` | 278M | 1,024 | 680 | 55.3 |

The 22M MiniLM model is the latency optimisation option: 8× faster than the
full-size models at moderate quality loss.

### T.6.3 Serving rerankers with vLLM

```python
from vllm import LLM

reranker = LLM(
    model="BAAI/bge-reranker-v2-m3",
    task="score",          # cross-encoder scoring mode
    max_model_len=8192,
)

query = "What is the capital of France?"
documents = [
    "Paris is the capital and largest city of France.",
    "France is a country in Western Europe.",
    "The Eiffel Tower is located in Paris.",
]

# Score (query, document) pairs
outputs = reranker.score(
    [(query, doc) for doc in documents]
)
scores = [o.outputs.score for o in outputs]
# scores ≈ [0.98, 0.41, 0.76]
# Rerank: sort documents by score descending
ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
```

Via API:

```bash
vllm serve BAAI/bge-reranker-v2-m3 \
    --task score \
    --max-model-len 8192

curl http://localhost:8000/v1/score \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "text_1": "What is the capital of France?",
    "text_2": ["Paris is the capital...", "France is a country..."]
  }'
```

### T.6.4 Serving rerankers with llama.cpp

```bash
# llama-server reranker mode
llama-server \
    --model bge-reranker-v2-m3.Q8_0.gguf \
    --reranking \          # enable reranking endpoint
    --port 8081 \
    --ctx-size 8192

# Query
curl http://localhost:8081/v1/reranking \
  -d '{
    "query": "What is the capital of France?",
    "documents": ["Paris is the capital...", "France is a country..."]
  }'
```

---

## T.7 BGE-M3: Hybrid Retrieval in Detail

BGE-M3 is the most capable open-source embedding model as of 2026 and worth
understanding in depth. It supports three retrieval modes simultaneously:

**Dense retrieval**: standard embedding-based semantic search.

**Sparse retrieval** (BM25-like): produces a sparse weight vector over
vocabulary tokens (lexical matching). Efficient for exact keyword matches.

**ColBERT-style multi-vector retrieval**: stores per-token embeddings rather
than a single pooled embedding. Enables fine-grained token-level matching at
the cost of more storage.

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# Encode with all three modes
output = model.encode(
    ["What is quantum entanglement?", "Quantum entanglement explained..."],
    batch_size=64,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True
)

dense_vecs  = output['dense_vecs']     # [N, 1024]
sparse_vecs = output['lexical_weights'] # list of {token: weight} dicts
colbert_vecs = output['colbert_vecs']  # list of [seq_len, 1024] arrays
```

For production RAG with BGE-M3 via vLLM:

```python
# vLLM serves dense mode; use FlagEmbedding for sparse/colbert
embedder = LLM(
    model="BAAI/bge-m3",
    task="embed",
    max_model_len=8192,
    gpu_memory_utilization=0.70,   # leave headroom for batch processing
)
```

---

## T.8 The Full RAG Serving Stack

A production RAG system has three distinct model serving components:

```
User Query
    │
    ▼
[1] Embedding Service (BGE-M3)
    │  Query vector (1024-dim)
    ▼
[2] Vector DB Retrieval (FAISS/Qdrant/Weaviate)
    │  Top-K candidate documents (K=20–100)
    ▼
[3] Reranker Service (BGE-Reranker-v2-m3)
    │  Top-M re-ranked documents (M=3–5)
    ▼
[4] LLM Generation Service (Llama 3.1 70B)
    │  Final answer
    ▼
Response
```

### T.8.1 Latency budget decomposition

For a typical RAG request at P95:

| Component | P95 Latency | Notes |
|---|---|---|
| Query embedding | 8ms | BERT-scale, batch=1 |
| Vector DB retrieval | 5ms | In-memory FAISS, K=50 |
| Reranker (50 docs) | 130ms | BGE-large, 512 tokens each |
| LLM generation | 1,200ms | 70B, 300 output tokens |
| **Total** | **~1,350ms** | LLM dominates |

The reranker at 130ms is often the second-largest latency contributor. The
embedding at 8ms is negligible. Optimisation priority: LLM first, reranker
second.

### T.8.2 Reranker batching for latency reduction

The 50-document reranker call above is sequential. Parallelise it:

```python
import asyncio
from vllm import AsyncLLMEngine, SamplingParams

async def rerank_parallel(query: str, docs: list[str],
                           reranker: AsyncLLMEngine) -> list[float]:
    """Score all (query, doc) pairs in parallel."""
    tasks = [
        reranker.async_score(query, doc)
        for doc in docs
    ]
    scores = await asyncio.gather(*tasks)
    return scores

# With async batching, 50 docs → ~30ms instead of 130ms
```

### T.8.3 Separate GPU allocation

Do not run embedding and reranker models on the same GPU as your LLM. The
memory contention and scheduling overhead degrade all three.

Recommended allocation:
- 1× H100/A100: LLM (70B or 8B)
- 1× L4 (24GB) or T4 (16GB): embedding + reranker
- The L4/T4 handles both embedding (BERT-scale) and reranker (BERT-scale)
  comfortably since they are not simultaneously active in a pipeline

---

## T.9 Worked Example T.1 — Production Embedding Service

**Goal**: serve BGE-M3 at 10,000 embedding requests/minute with < 50ms P95
latency per request (batch size 1).

**Hardware**: 1× A100 40GB

**Analysis**:

At 10,000 req/min = 167 req/s. Average query length = 64 tokens.
BGE-M3 on A100 at batch=1: ~25ms per embedding = 40 embeddings/s.
Throughput needed: 167/s. Gap: 167/40 = 4.2× shortfall.

Fix: use dynamic batching. With a 20ms batching window:
- Requests arriving in 20ms window: 167 × 0.020 = ~3.3 requests
- Batch those together: 1 forward pass instead of 3.3
- Effective throughput: 1000/ms × 1/20ms × batch × embeddings_per_batch

With batch=8: BGE-M3 processes 8 × 64 = 512 tokens in ~45ms.
Throughput: 8 / 0.045s = 178 embeddings/s > 167 needed ✓
P95 latency: 20ms (queue) + 45ms (inference) = 65ms — slightly over budget.

Optimisation: reduce batching window to 10ms:
- Batch=4: ~28ms inference, 10ms queue = 38ms P95 ✓
- Throughput: 4 / 0.028 = 143/s — still short.

Use async requests with batch=8 and 2× parallel forward passes:
- 2 batches of 8 in parallel: 16/0.045s = 355/s > 167 ✓
- P95: 20ms queue + 45ms = 65ms — 30% over budget.

Final: FP16 → INT8 quantization reduces BGE-M3 inference to ~28ms:
- P95: 20ms + 28ms = 48ms ✓ Throughput: 8/0.028 = 286/s ✓

```bash
# Optimal configuration
vllm serve BAAI/bge-m3 \
    --task embed \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --quantization int8 \
    --gpu-memory-utilization 0.80
```

---

## T.10 Instruction Prefixes for Asymmetric Retrieval

Many modern embedding models require instruction prefixes to distinguish
query embeddings from document embeddings. Without the prefix, query and
document embeddings are in the same space but without the directional cue
the model was trained with.

```python
# E5 models
query_prefix    = "query: "
document_prefix = "passage: "

queries    = [query_prefix + q for q in raw_queries]
documents  = [document_prefix + d for d in raw_docs]

# BGE models
query_prefix    = "Represent this sentence for searching relevant passages: "
document_prefix = ""  # BGE documents need no prefix

# GTE-Qwen2 and NV-Embed
query_prefix = "Instruct: Given a web search query, retrieve relevant passages\nQuery: "
document_prefix = ""
```

Using the wrong prefix (or no prefix) on a model that expects one typically
degrades retrieval nDCG@10 by 3–8 points — significant in production.

---

## T.11 Self-Check Questions

1. An embedding model uses mean pooling over 512 tokens. At inference time,
   one of those tokens is a `[PAD]` token. What goes wrong if you include it
   in the mean without using the attention mask? How does the masked mean
   pooling formula fix this?

2. A RAG pipeline uses BGE-M3 for query embedding and a FAISS IVF-PQ index.
   The embeddings are L2-normalised. Explain why `faiss.IndexFlatIP` (inner
   product) gives the same ranking as cosine similarity for L2-normalised
   vectors. What is the advantage of using IP over L2 distance in FAISS?

3. You are choosing between `bge-reranker-large` (435M, 520 pairs/s) and
   `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M, 4,200 pairs/s). Your P95
   latency budget for reranking 100 candidate documents is 50ms. Which model
   fits? At what candidate set size does each model exceed the 50ms budget?

4. Describe the difference between dense, sparse, and multi-vector retrieval in
   BGE-M3. In what retrieval scenario does each mode have the highest advantage?

5. A decoder-only LLM fine-tuned as an embedding model (e.g., NV-Embed-v2)
   uses last-token pooling rather than CLS pooling. Why can't an autoregressive
   model use CLS pooling? What property of the decoder architecture makes last-
   token pooling the natural choice?
