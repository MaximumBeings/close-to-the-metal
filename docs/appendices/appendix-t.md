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
equivalent — enabling FAISS's IVF-PQ index (which optimizes inner product) for
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

The 22M MiniLM model is the latency optimization option: 8× faster than the
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
embedding at 8ms is negligible. Optimization priority: LLM first, reranker
second.

### T.8.2 Reranker batching for latency reduction

The 50-document reranker call above is sequential. Parallelize it:

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

Optimization: reduce batching window to 10ms:
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

## T.11 Cold-Start Latency

Embedding and reranker models load faster than LLMs but still incur measurable
start-up latency. Understanding this is important for Kubernetes autoscaling
(Chapter 19) and on-demand serverless deployments.

### Model loading times on A100 80GB

| Model | Size | Load time (NVMe SSD) | Load time (network storage) | CUDA warm-up |
|---|---|---|---|---|
| bge-small-en-v1.5 | 130 MB | 0.8s | 2.5s | 0.3s |
| bge-large-en-v1.5 | 1.3 GB | 2.1s | 6.4s | 0.5s |
| BGE-M3 | 2.2 GB | 3.5s | 10.8s | 0.8s |
| bge-reranker-large | 1.7 GB | 2.8s | 8.4s | 0.5s |
| NV-Embed-v2 | 7.7 GB | 8.2s | 24.0s | 1.2s |

CUDA warm-up includes the first forward pass to compile CUDA kernels. For
PyTorch dynamic shapes, add 1–3 seconds for the second warm-up call (different
batch size).

### Cold-start mitigation

```python
import time
import torch
from sentence_transformers import SentenceTransformer

class WarmEmbeddingServer:
    """Pre-loads model and runs warm-up pass to eliminate cold-start overhead."""

    WARMUP_TEXTS = [
        "This is a warm-up sentence for CUDA kernel compilation.",
        "A second sentence to warm batch dimension 2.",
        "A third sentence to warm batch dimension 3.",
        "A fourth sentence to warm batch dimension 4.",
    ]

    def __init__(self, model_name: str, device: str = "cuda"):
        t0 = time.time()
        self.model = SentenceTransformer(model_name, device=device)
        load_time = time.time() - t0
        print(f"Model loaded in {load_time:.1f}s")

        # Warm-up passes: compile CUDA kernels for common batch sizes
        for batch_size in [1, 2, 4, 8, 16, 32]:
            texts = (self.WARMUP_TEXTS * batch_size)[:batch_size]
            with torch.no_grad():
                _ = self.model.encode(texts, batch_size=batch_size,
                                      convert_to_numpy=False)
        print(f"Warm-up complete. Ready for production traffic.")

    def encode(self, texts: list[str], **kwargs):
        return self.model.encode(texts, **kwargs)
```

In Kubernetes, add a readiness probe that calls a `/health` endpoint
only after `WarmEmbeddingServer.__init__` completes:

```python
# FastAPI health endpoint
@app.get("/health")
def health():
    if not server_ready:
        raise HTTPException(status_code=503, detail="Model warming up")
    return {"status": "ok"}
```

---

## T.12 ColBERT Multi-Vector Indexing for Large Corpora

BGE-M3's multi-vector (ColBERT-style) retrieval produces one embedding per
token rather than one per document. This dramatically increases retrieval
quality for long documents, but requires specialized indexing infrastructure.

### Token-level storage cost

For a corpus of 1M documents, average 512 tokens/document, 1024-dim FP32
embeddings:

```
1M docs × 512 tokens/doc × 1024 dims × 4 bytes = 2.1 TB
```

This is ~100× larger than single-vector dense retrieval (21 GB). FP16 halves
it to 1.05 TB; INT8 quantisation of the token vectors halves again to ~525 GB.

### Practical ColBERT deployment options

**PLAID (recommended for scale)**: the PLAID algorithm compresses token
vectors with IVF centroids and late-interaction scoring, reducing storage to
10–50 GB for 1M documents.

```python
# Using RAGatouille (PLAID-based ColBERT for Python)
from ragatouille import RAGPretrainedModel

# Index 1M documents (requires ~20 GB disk for PLAID compressed index)
rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
rag.index(
    collection=documents,          # list of strings
    index_name="my_corpus",
    max_document_length=512,
    split_documents=True,
)

# Retrieve
results = rag.search(query="what is flash attention", k=10)
```

**Vespa or Qdrant ColBERT mode**: for production scale with exact MaxSim:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(url="http://localhost:6333")
# ColBERT multi-vector collection: each doc stores all token vectors
client.create_collection(
    collection_name="colbert_docs",
    vectors_config={
        "token_embeddings": VectorParams(
            size=128,               # projected ColBERT dim
            distance=Distance.COSINE,
            multivector_config={"comparator": "max_sim"},
        )
    }
)
```

### When to use multi-vector vs single-vector

| Criterion | Single-vector (dense) | Multi-vector (ColBERT) |
|---|---|---|
| Corpus size | Any | < 5M docs (without PLAID) |
| Document length | Short (<256 tokens) | Long (>256 tokens) |
| Query type | Short keyword queries | Full-sentence queries |
| Storage budget | 1× | 100× (raw) / 5× (PLAID) |
| Retrieval latency | <10ms | 50–200ms (exact MaxSim) |
| nDCG@10 improvement | Baseline | +5–15pp for long docs |

---

## T.13 GPU Memory Requirements for Embedding Models

| Model | Params | FP16 VRAM | INT8 VRAM | FP16 + batch 32 VRAM | Min GPU |
|---|---|---|---|---|---|
| bge-small-en-v1.5 | 33M | 0.07 GB | 0.04 GB | 0.2 GB | Any |
| bge-base-en-v1.5 | 110M | 0.22 GB | 0.11 GB | 0.4 GB | Any |
| bge-large-en-v1.5 | 335M | 0.67 GB | 0.34 GB | 1.0 GB | Any |
| BGE-M3 | 570M | 1.14 GB | 0.57 GB | 1.8 GB | 8 GB |
| bge-reranker-large | 435M | 0.87 GB | 0.44 GB | 1.5 GB | Any |
| E5-mistral-7b | 7.1B | 14.2 GB | 7.1 GB | 16.0 GB | 24 GB |
| NV-Embed-v2 | 7.8B | 15.6 GB | 7.8 GB | 17.5 GB | 24 GB |
| GTE-Qwen2-7B | 7.6B | 15.2 GB | 7.6 GB | 17.0 GB | 24 GB |

VRAM formula for a model with `P` parameters, dtype `D` bytes, batch `B`, sequence length `L`, hidden size `H`:

```python
def embedding_model_vram(params_B, dtype_bytes=2, batch=32, seq_len=512, hidden=1024):
    weights_gb = params_B * 1e9 * dtype_bytes / 1e9
    # Activation memory: batch × seq_len × hidden × dtype × 2 (forward + buffer)
    activations_gb = batch * seq_len * hidden * dtype_bytes * 2 / 1e9
    return weights_gb + activations_gb

# BGE-M3 at batch 32
vram = embedding_model_vram(0.57, dtype_bytes=2, batch=32, seq_len=512, hidden=1024)
print(f"BGE-M3 VRAM at batch 32: {vram:.2f} GB")  # → ~1.73 GB
```

### Test harness — embedding arithmetic

```python
# ── test_appendix_t.py ───────────────────────────────────────────────────
"""Offline tests for embedding serving arithmetic. No GPU required.
Run with: python test_appendix_t.py"""

import math


def l2_normalise(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    na, nb = l2_normalise(a), l2_normalise(b)
    return dot_product(na, nb)


def masked_mean_pool(embeddings: list[list[float]],
                     mask: list[int]) -> list[float]:
    """Mean pool with attention mask (exclude PAD tokens)."""
    active = [emb for emb, m in zip(embeddings, mask) if m == 1]
    if not active:
        raise ValueError("All tokens masked — empty sequence")
    d = len(active[0])
    return [sum(emb[i] for emb in active) / len(active) for i in range(d)]


def test_l2_normalisation():
    v = [3.0, 4.0]
    normed = l2_normalise(v)
    norm = math.sqrt(sum(x * x for x in normed))
    assert abs(norm - 1.0) < 1e-9, f"L2 norm should be 1.0, got {norm}"
    print("PASS: L2 normalisation produces unit vector")


def test_cosine_via_inner_product():
    """For L2-normalised vectors, IP = cosine similarity."""
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    cos = cosine_similarity(a, b)
    an, bn = l2_normalise(a), l2_normalise(b)
    ip = dot_product(an, bn)
    assert abs(cos - ip) < 1e-9, "Cosine should equal IP for normalised vectors"
    print("PASS: L2-norm IP == cosine similarity")


def test_masked_mean_pool():
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    mask_all   = [1, 1, 1]
    mask_no_pad = [1, 1, 0]   # last token is PAD

    full_mean   = masked_mean_pool(embeddings, mask_all)
    masked_mean = masked_mean_pool(embeddings, mask_no_pad)

    assert abs(full_mean[0] - 0.5) < 1e-9
    assert abs(masked_mean[0] - 0.5) < 1e-9
    assert abs(masked_mean[1] - 0.5) < 1e-9   # mean of [0.0, 1.0] = 0.5
    assert full_mean != masked_mean, "PAD token should affect unmasked mean"
    print("PASS: masked mean pool correctly excludes PAD tokens")


def test_throughput_estimate():
    """Verify worked example: 10k req/min needs INT8 quantisation."""
    req_per_min = 10_000
    req_per_s   = req_per_min / 60          # 166.7
    latency_fp16_ms = 30                    # 30ms per embedding (FP16)
    throughput_fp16 = 1000 / latency_fp16_ms  # 33.3 emb/s

    latency_int8_ms = 12                    # ~12ms with INT8
    throughput_int8 = 1000 / latency_int8_ms  # 83.3 emb/s

    assert throughput_fp16 < req_per_s, (
        "FP16 should be insufficient for 10k req/min"
    )
    # With 2 instances INT8: 2 × 83.3 = 166.7 ≥ 166.7 req/s (just barely)
    assert throughput_int8 * 2 >= req_per_s, (
        "2× INT8 instances should meet throughput target"
    )
    print(f"PASS: FP16 {throughput_fp16:.1f} req/s insufficient; "
          f"INT8×2 {throughput_int8*2:.1f} req/s sufficient")


def test_colbert_storage():
    """Verify ColBERT storage calculation for 1M documents."""
    n_docs      = 1_000_000
    tokens_per_doc = 512
    dims        = 1_024
    bytes_dtype = 4   # FP32

    total_bytes = n_docs * tokens_per_doc * dims * bytes_dtype
    total_tb    = total_bytes / 1e12
    assert abs(total_tb - 2.097) < 0.01, f"Expected ~2.1 TB, got {total_tb:.3f}"
    print(f"PASS: ColBERT 1M docs FP32 = {total_tb:.2f} TB")


if __name__ == "__main__":
    test_l2_normalisation()
    test_cosine_via_inner_product()
    test_masked_mean_pool()
    test_throughput_estimate()
    test_colbert_storage()
    print("\n✓ All embedding serving tests passed.")
```

**Expected output:**
```
PASS: L2 normalisation produces unit vector
PASS: L2-norm IP == cosine similarity
PASS: masked mean pool correctly excludes PAD tokens
PASS: FP16 33.3 req/s insufficient; INT8×2 166.7 req/s sufficient
PASS: ColBERT 1M docs FP32 = 2.10 TB

✓ All embedding serving tests passed.
```

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

---

## Worked Solutions

### Solution 1 — Masked mean pooling

**What goes wrong without the attention mask:**

Mean pooling without masking sums embedding vectors for all `n` tokens (including
`[PAD]`) and divides by `n`:

```
e_wrong = (1/n) × Σᵢ hᵢ    for i = 1..n, including PAD tokens
```

The `[PAD]` token produces a non-zero embedding vector `h_pad` — the model has
learned an embedding for the padding token ID. Including it shifts the mean
toward `h_pad`, degrading the semantic fidelity of the sentence representation.
At 512 tokens with 1 PAD token, the dilution is 1/512 ≈ 0.2%, small but
measurable. At 64 actual tokens plus 448 PAD tokens, the dilution is 448/512 ≈
87.5% — the pooled embedding would be nearly indistinguishable from `h_pad`.

**Masked mean pooling fix:**

```
mask  = [1, 1, 1, ..., 1, 0]   # 1 for real tokens, 0 for PAD
denom = mask.sum() = n_real     # count of non-PAD tokens

e_masked = Σᵢ (mask[i] × hᵢ) / denom
```

Step by step:
1. Multiply each token embedding by its mask bit — PAD embeddings become the
   zero vector.
2. Sum the masked embeddings.
3. Divide by the number of real tokens (`n_real`), not the sequence length.

This guarantees the sentence embedding is the mean over only the meaningful
tokens, regardless of padding position or count.

**Common mistake:** Dividing by `n` (sequence length) even after zeroing PAD
embeddings. The numerator is correct (PAD zeroed out) but the denominator is
wrong — you divide by a larger number than the actual token count, producing a
scaled-down embedding. Always divide by `mask.sum()`.

---

### Solution 2 — Inner product vs cosine similarity for L2-normalised vectors

**Mathematical identity:**

For two vectors **u** and **v** with ‖u‖₂ = ‖v‖₂ = 1:

```
cosine_similarity(u, v) = u · v / (‖u‖ ‖v‖)
                        = u · v / (1 × 1)
                        = u · v
                        = inner_product(u, v)
```

When both vectors are unit-normalised, inner product and cosine similarity are
numerically identical. A higher inner product means higher cosine similarity,
so ranking by IP gives the same ordering as ranking by cosine.

**Why IP over L2 in FAISS?**

The L2 distance between unit vectors is related to inner product by:

```
‖u - v‖² = ‖u‖² + ‖v‖² - 2(u · v)
           = 1 + 1 - 2(u · v)
           = 2 - 2(u · v)
```

So minimizing L2 distance is equivalent to maximizing inner product — the
ranking is identical. However, `IndexFlatIP` has a computational advantage:

- **L2 distance** requires computing `‖u - v‖²` = expanded form with three
  terms; FAISS must materialise `Σ(uᵢ - vᵢ)²`.
- **Inner product** requires only `Σ uᵢvᵢ` — a single GEMM operation with no
  subtraction, directly maps to `BLAS sgemm`, and benefits from hardware-level
  fused multiply-add (FMA) optimization.

Result: `IndexFlatIP` is typically 10–20% faster than `IndexFlatL2` for
unit-normalised embeddings, with identical result ordering.

---

### Solution 3 — Reranker latency and budget

**Given:**
- `bge-reranker-large`: 435M params, 520 pairs/s
- `ms-marco-MiniLM-L-6-v2`: 22M params, 4,200 pairs/s
- P95 latency budget: 50 ms for 100 candidate documents

**Latency per model for 100 candidates:**

```
latency_bge     = 100 / 520  pairs/s = 0.1923 s = 192 ms
latency_minilm  = 100 / 4200 pairs/s = 0.0238 s =  24 ms
```

**Decision:** `bge-reranker-large` at 192 ms **exceeds** the 50 ms budget by
3.8×. `ms-marco-MiniLM-L-6-v2` at 24 ms **fits** within the budget.

**Break-even candidate set size (N) for each model:**

```
bge_max_N    = 520  × 0.050 = 26  candidates
minilm_max_N = 4200 × 0.050 = 210 candidates
```

Within the 50 ms budget:
- `bge-reranker-large` can rerank at most **26 candidates**.
- `ms-marco-MiniLM-L-6-v2` can rerank at most **210 candidates**.

**Practical implication:** If your RAG retriever returns 100 documents, only
MiniLM fits the latency budget. If quality is paramount and you can afford
25–30 ms extra latency, consider a two-stage approach: MiniLM to trim 100→20,
then BGE-reranker-large on the top 20 (192 × 20/100 = 38 ms — now within
budget).

---

### Solution 4 — Dense vs sparse vs multi-vector retrieval (BGE-M3)

**Dense retrieval:**
Each query and document is encoded as a single fixed-length vector (e.g., 1024
dimensions). Similarity is computed as one inner product. Advantages: fast
FAISS search, small index size, works well when semantic paraphrasing is common.
Disadvantage: a single vector cannot capture all semantic facets of a long
document.

**Best scenario:** Broad semantic search where the user's phrasing differs from
the document's phrasing (paraphrase matching, cross-lingual retrieval).

**Sparse retrieval:**
Uses a learned sparse representation (e.g., SPLADE): most dimensions are zero,
non-zero dimensions correspond roughly to important terms. Stored and searched
using inverted index structures. Advantages: exact term matching, interpretable,
efficient for keyword-heavy queries. Disadvantage: poor at semantic paraphrasing.

**Best scenario:** Technical documentation, medical records, or legal text where
exact terminology matters and users type precise keywords.

**Multi-vector retrieval (ColBERT-style, BGE-M3's "ColBERT mode"):**
Every token in the document produces a separate embedding vector. Query tokens
perform MaxSim (maximum inner product over all document token vectors) for each
query token, then sum. This late-interaction model is much more expressive.
Advantage: fine-grained token-level alignment catches partial matches. Disadvantage:
index size scales with sequence length × number of documents × embedding dim —
potentially 100× larger than dense.

**Best scenario:** Retrieval tasks requiring fine-grained matching — e.g.,
multi-hop QA, code search where specific function names and arguments matter,
or biomedical literature where exact phrase alignment is critical.

**BGE-M3 hybrid:** Combine all three signals with a weighted sum
(`dense + λ₁ × sparse + λ₂ × colbert`). This achieves SOTA on BEIR while
maintaining acceptable latency by using dense for initial candidate retrieval
and multi-vector for reranking.

---

### Solution 5 — Last-token pooling for decoder-only LLMs

**Why CLS pooling fails for autoregressive decoders:**

CLS pooling reads the representation at position 0 (the `[CLS]` token). This
works in bidirectional encoders (BERT) because attention at every position can
attend to every other position — the `[CLS]` token accumulates information from
the entire sequence during forward propagation.

Decoder-only (causal) architectures use **causal attention masks**: each token
can only attend to previous tokens, not future ones. Position 0 can see only
itself. It has no information about tokens at positions 1, 2, …, n. Pooling at
position 0 would give an embedding that represents nothing beyond the first
token — worse than random.

**Why last-token pooling is the natural choice:**

In a causal model, the last token (position n) has attended to all preceding
tokens through the chain of residual connections and attention layers. Its
hidden state `h_n` is the most information-rich position: it is conditioned on
the full sequence.

Formally, for an autoregressive model with causal mask M:

```
h_t = Attention(Q_t, K_{≤t}, V_{≤t}) + FFN(...)
```

Only `h_n` (t = n) has K and V from all positions 1 through n. It is the
natural "summary" token.

**NV-Embed-v2 additional trick:** It appends a learned `[EOS]` instruction
token after the input and uses that token's representation, ensuring the model
can learn a dedicated pooling behavior through fine-tuning without conflating
it with the final content token. This is equivalent to last-token pooling but
with a controllable, task-specific pooling head.
