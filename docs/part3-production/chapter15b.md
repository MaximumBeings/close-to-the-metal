# Chapter 15.5: Flash Decoding and Context Parallelism — Parallelising Across Sequence Length

> *"Tensor parallelism splits across heads. Data parallelism splits across batches. Flash Decoding splits across the one dimension everyone forgot: the sequence itself."*

## What you will understand after this chapter

- Why attention becomes a bottleneck at long context even with many GPUs
- How Flash Decoding uses the log-sum-exp trick to parallelize the attention reduction
- How context parallelism splits a single 1M-token sequence across multiple GPUs
- The decision matrix for choosing between tensor, data, Flash Decoding, and context parallelism
- Measured latency projections from 8K to 1M context length

## What you need first

- Chapter 5 (Flash Attention) — the tiling and recomputation approach
- Chapter 15 (Multi-GPU Serving) — tensor and pipeline parallelism basics

---

## 15b.1  The Sequence-Length Bottleneck

Chapter 15 showed how tensor parallelism splits the model's weight matrices across GPUs, distributing heads across devices. But there is a dimension tensor parallelism does not touch: the **sequence length**.

During the decode step, each query vector must attend to **all** past key-value pairs.

```
WORKED EXAMPLE 15b.1 — Attention reduction at long context
───────────────────────────────────────────────────────────
Model:     Llama-3-70B, GQA with 8 KV heads, head_dim=128
Context:   100,000 tokens
Query:     1 token (decode step)

Per KV head, attention computation:
  Q:  (1, 128)         — single query vector
  K:  (100000, 128)    — 100K key vectors
  V:  (100000, 128)    — 100K value vectors

  Step 1: scores = Q @ K^T  → (1, 100000)
  Step 2: weights = softmax(scores / sqrt(128))
  Step 3: output  = weights @ V → (1, 128)

Memory read for 8 KV heads, BF16:
  8 × 100000 × 128 × 2 bytes × 2 (K and V) = 409 MB

At 3350 GB/s (H100): 0.12 ms just to read KV cache
At 1M context:       1.2 ms — dominates decode latency
───────────────────────────────────────────────────────────
```

Tensor parallelism distributes the 8 KV heads across 4 GPUs — 2 heads each. But each GPU must still read its **own** 100K key vectors sequentially. The per-GPU memory read is halved, but the sequential bottleneck remains.

---

## 15b.2  Flash Decoding — Parallelising the Reduction

Flash Decoding (Dao et al., 2023) splits the KV sequence into P partitions, computes partial softmax results in parallel, then merges using the log-sum-exp trick.

```
Standard attention (sequential over N keys):
  scores[i] = Q · K[i] / sqrt(d)   for i = 0..N-1
  weights = softmax(scores)         ← needs all N scores
  output  = sum_i(weights[i] * V[i])

Flash Decoding (parallel over P partitions):
  Partition p handles keys [p*N/P .. (p+1)*N/P - 1]

  For each partition p:
    local_scores  = Q · K[p*N/P:(p+1)*N/P]^T / sqrt(d)
    local_max     = max(local_scores)
    local_exp     = exp(local_scores - local_max)
    local_lse     = log(sum(local_exp))          ← log-sum-exp
    local_output  = (local_exp @ V[p]) / sum(local_exp)

  Merge across P partitions:
    global_max = max(local_max[p])
    w[p]       = exp(local_lse[p] + local_max[p] - global_max)
    output     = sum(w[p] * local_output[p]) / sum(w[p])
```

**WORKED EXAMPLE 15b.2 — Flash Decoding merge, 2 partitions:**

```
WORKED EXAMPLE 15b.2
───────────────────────────────────────────────────────────
Setup: 4 keys, 2 partitions of 2 keys, scalar values

Raw scores:  [3.0, 1.0 | 2.0, 4.0]
Values:      [1.0, 2.0 | 3.0, 4.0]

Partition 0: scores=[3.0, 1.0]
  local_max = 3.0
  local_exp = [exp(0)=1.0, exp(-2)=0.135]
  local_sum = 1.135
  local_lse = log(1.135) = 0.127
  local_out = (1.0×1.0 + 0.135×2.0) / 1.135 = 1.119

Partition 1: scores=[2.0, 4.0]
  local_max = 4.0
  local_exp = [exp(-2)=0.135, exp(0)=1.0]
  local_sum = 1.135
  local_lse = log(1.135) = 0.127
  local_out = (0.135×3.0 + 1.0×4.0) / 1.135 = 3.881

Merge:
  global_max = 4.0
  w[0] = exp(0.127 + 3.0 - 4.0) = exp(-0.873) = 0.418
  w[1] = exp(0.127 + 4.0 - 4.0) = exp(0.127)  = 1.135
  output = (0.418×1.119 + 1.135×3.881) / (0.418+1.135)
         = (0.468 + 4.405) / 1.553 = 3.136

Verify with naive softmax over all 4 scores [3,1,2,4]:
  max=4, exp=[exp(-1)=0.368, exp(-3)=0.050, exp(-2)=0.135, 1.0]
  sum=1.553
  output = (0.368×1+0.050×2+0.135×3+1.0×4)/1.553
         = (0.368+0.100+0.405+4.0)/1.553 = 3.136 ✓
───────────────────────────────────────────────────────────
```

---

## 15b.3  Flash Decoding vs FlashAttention-2

`[COMMON TRAP]` These are related but solve different problems:

| Property | FlashAttention-2 | Flash Decoding |
|---|---|---|
| **Primary use** | Prefill / training (many queries) | Single-query decode |
| **Parallelism** | Across query blocks (rows of Q) | Across KV blocks (columns of K,V) |
| **Why it helps** | Many queries: parallelise Q dimension | One query: parallelise KV dimension |
| **Implemented in** | PyTorch, most backends | FlashInfer, Flash-Decoding |

At decode time there is exactly **one** query token. FlashAttention-2's query-tile parallelism has no work to split. Flash Decoding is the correct tool.

---

## 15b.4  Parallelism Dimension Taxonomy

```
┌─────────────────────────────────────────────────────────────┐
│  Inference Parallelism Dimensions                            │
│                                                               │
│  Tensor (TP)     Head/hidden dim   Large models, memory     │
│  Pipeline (PP)   Layer groups      Very deep, >8 GPUs       │
│  Data (DP)       Batch items       High throughput           │
│  Sequence        KV sequence       Long context, latency     │
│    ├── Flash Decoding  (intra-GPU, across SMs)              │
│    └── Context Parallel (inter-GPU, across devices)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 15b.5  Context Parallelism — Ring Attention

Context parallelism splits the **input sequence** across GPUs, with a ring-based communication pattern so each GPU can attend to all tokens:

```
Context Parallelism, 4 GPUs, 1M-token sequence:
  GPU 0: tokens   0–249K   (owns these Q, K, V)
  GPU 1: tokens 250K–499K
  GPU 2: tokens 500K–749K
  GPU 3: tokens 750K–999K

Ring pass schedule (3 steps):
  Step 0: GPU i computes attention(Q_i, K_i, V_i)  [local]
  Step 1: GPU i sends (K_i, V_i) to GPU (i+1)%4
          GPU i receives (K_{i-1}, V_{i-1})
          Computes attention(Q_i, K_{i-1}, V_{i-1}) [accumulate]
  Step 2: Repeat until all blocks visited
  Step 3: Merge partial results per GPU via log-sum-exp

Communication cost (4 GPUs, 250K local tokens, 8 KV heads, d=128, BF16):
  Per pass: 250000 × 8 × 128 × 2 × 2 = 1.02 GB
  3 passes: 3.07 GB total
  At 600 GB/s NVLink: 5.1 ms
  (can be pipelined with compute on modern hardware)
```

`[DEEP DIVE]` Context parallelism is only efficient when NVLink bandwidth is available and sequence length is large enough that per-GPU compute exceeds communication time. On PCIe (64 GB/s), communication dominates at almost all practical sequence lengths.

---

## 15b.6  Decision Matrix

```
seq_len < 32K:
  → Standard TP + DP. Flash Decoding provides minimal gain.

32K ≤ seq_len < 256K, any GPU count:
  → TP + Flash Decoding (P=16–64 partitions intra-GPU).
    No inter-GPU sequence splitting needed.

256K ≤ seq_len, ≥ 8 GPUs with NVLink:
  → TP + Context Parallel (CP=4–8) + Flash Decoding.
    Pipeline KV communication behind compute.

All long-context deployments:
  → Consider KV eviction (Ch 11b) to cap working set size.
```

---

## 15b.7  vLLM and FlashInfer Integration

FlashInfer implements Flash Decoding natively and is vLLM's default attention backend. To verify:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B",
    tensor_parallel_size=4,
    max_model_len=131072,   # 128K — Flash Decoding active automatically
)
# Verify: check attn backend class name → "FlashInferBackend"
```

Flash Decoding activates automatically when `max_model_len` is large. There is no flag to set; FlashInfer selects the optimal partition count based on sequence length at runtime.

---

## 15b.8  Latency Projections Across Context Lengths

```
Llama-3-70B, 4×H100 (NVLink), decode step, batch=1
─────────────────────────────────────────────────────────────────
Context    KV read    Std attn   Flash Dec.   Context Par.
length     (GB)       (ms)       P=32 (ms)    CP=4 (ms)
─────────────────────────────────────────────────────────────────
  8K         0.03      14.1        14.0          N/A
 32K         0.13      14.5        14.2          N/A
128K         0.52      16.8        14.8          N/A
512K         2.07      25.1        16.2         14.9
  1M         4.13      39.4        19.8         16.1
  2M         8.26      67.8        28.3         18.4
─────────────────────────────────────────────────────────────────
Flash Decoding: ~2× speedup at 1M context vs standard
Context Par.:   ~2.5× speedup at 1M; requires NVLink + ≥4 GPUs
Both combined:  recommended for production 1M+ deployments
```

---

## Chapter Summary

Standard tensor parallelism distributes attention across heads but leaves the sequence-length dimension sequential. At 100K+ token contexts, reading all KV pairs dominates decode latency. Flash Decoding parallelises this by splitting the KV sequence into P partitions, computing partial log-sum-exp normalizers independently, then merging with a numerically stable reduce — producing exactly the same output as naive softmax. Context parallelism extends the approach across GPUs using a ring-based KV exchange pattern, suitable for sequences above 256K on NVLink-connected hardware. Flash Decoding is intra-GPU (across SMs, via FlashInfer), context parallelism is inter-GPU. Both can be combined. The crossover point where each strategy pays off depends on sequence length, GPU count, and interconnect bandwidth.

---

## Self-Check Questions

1. Verify the Flash Decoding merge formula numerically: two partitions with scores [2.0, 0.5] and [3.0, 1.0] and corresponding scalar values [1.0, 2.0] and [3.0, 4.0]. Compute the merged output and verify against naive softmax over all four scores.

2. Why can FlashAttention-2 not be used to speed up the decode step in the same way it speeds up prefill? What property of the decode step makes Flash Decoding necessary?

3. Context parallelism with CP=4, local KV size 128K tokens (8 heads, d=128, BF16), NVLink at 600 GB/s. Compute the total communication time for one ring pass. What compute time is needed for communication not to be the bottleneck?

4. A team wants 1M-context serving on a 4-GPU node with PCIe interconnect (64 GB/s). Is context parallelism viable? What would you recommend instead?

5. Flash Decoding with P=32 partitions on a 128K context: each partition handles 4K keys. What is the memory savings per partition compared to materialising the full attention weight vector? (Assume 4-byte floats, 1 query, head_dim=128.)
