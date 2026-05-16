# Chapter 3.5 — The Transformer Block: FFN, Layer Norm, and the Residual Stream

> *"Attention is the part everyone talks about. The FFN is where the model actually remembers things."*

---

## Why This Chapter Exists

Chapter 3 explained how tokens become integers and integers become batches. Chapter 4 will open the attention computation in detail. But between the two sits the complete transformer block — the unit that repeats 32, 80, or 128 times in a modern LLM. Understanding it in full before studying attention means you will never be confused about what gets cached, what gets recomputed, and why the GPU memory footprint scales the way it does.

This chapter covers:

- The residual stream and why it is the central data structure of a transformer
- Layer normalization — what it does, why it matters for numerics, and how RMSNorm differs from LayerNorm
- The FFN (Feed-Forward Network) sub-layer — shape, role, and where most of the parameters live
- The SwiGLU and GeLU activation variants used in production models
- How a complete forward pass flows through one transformer block
- Why MoE (Mixture of Experts) is just a sparse FFN from the serving perspective
- Memory and FLOPs accounting for a single block

**Prerequisites:** Chapter 3 (tokens, batching). No attention knowledge required — attention is treated as a black box here and opened fully in Chapter 4.

---

## 3.5.1 The Residual Stream

The single most important architectural insight in the transformer is the **residual connection**. Every sub-layer adds its output to its input rather than replacing it:

```
x_out = x_in + SubLayer(LayerNorm(x_in))
```

The sequence of residual sums forms what researchers call the **residual stream** — a vector of shape `[seq_len, d_model]` that flows through the entire network. Each layer reads from the stream, computes a delta, and writes that delta back.

This has several practical consequences:

**Gradient highway.** Gradients flow directly from output to input along the residual path, bypassing the sub-layer. This is why transformers can be trained to depths that would cause vanishing gradients in plain MLPs.

**Interpretability.** Because each layer adds a delta to the stream, you can in principle read off the stream at any intermediate layer and get a meaningful representation. "Logit lens" probing and residual stream decomposition rely on this.

**Layer-skipping.** The residual connection is why early exit and layer pruning work at all. If a layer learns an approximately zero delta (near-identity), it can be skipped without catastrophic degradation. Several production quantization schemes exploit this.

**Serving implication.** The residual stream is the tensor you allocate and manage. At batch size B, sequence length n, model dimension d_model, and precision P bytes:

```
residual_stream_bytes = B × n × d_model × P
```

For LLaMA 3 70B (d_model = 8192, P = 2 for BF16, n = 4096, B = 1):

```
1 × 4096 × 8192 × 2 = 67 MB
```

That tensor travels through all 80 layers, being updated in place. It is the largest single activation tensor in the forward pass.

---

## 3.5.2 Layer normalization

Every sub-layer is preceded by a normalization step. Two variants appear in production:

### LayerNorm (original transformer, GPT-2, BERT)

```
LayerNorm(x) = γ · (x − μ) / √(σ² + ε) + β
```

Where μ and σ² are the mean and variance computed across the d_model dimension (per token), γ and β are learned scale and bias parameters of shape `[d_model]`.

**Properties:** Zero mean, unit variance per token. The learned γ and β allow the network to undo normalization if needed.

### RMSNorm (LLaMA 1/2/3, Mistral, Qwen, DeepSeek)

```
RMSNorm(x) = γ · x / RMS(x)    where RMS(x) = √(mean(x²) + ε)
```

RMSNorm removes the mean-centering step (no μ subtraction, no β bias). This makes it roughly 10–15% faster than LayerNorm at minimal accuracy cost — important at d_model = 8192 computed 80 × 2 = 160 times per forward pass.

### Pre-norm vs Post-norm

**Pre-norm** (used by all modern LLMs): normalization happens before the sub-layer, as shown above. This stabilises training and allows higher learning rates.

**Post-norm** (original "Attention Is All You Need"): normalization after the residual add. Harder to train but sometimes slightly better in final accuracy.

### Numerical notes for inference

Layer norm operates on FP16/BF16 activations but accumulates variance sums in FP32 to avoid catastrophic cancellation (subtracting large near-equal numbers). vLLM's CUDA kernels use `__float2half_rn` with FP32 accumulation. llama.cpp uses a similar pattern in its `ggml_norm` kernel. Getting this wrong produces silent numerical drift that manifests as degraded perplexity.

---

## 3.5.3 The Feed-Forward Network

After the attention sub-layer, every transformer block runs the **FFN** (also called MLP or feed-forward layer). The standard form:

```
FFN(x) = W₂ · activation(W₁ · x)
```

Where:

- `x`: input `[d_model]`
- `W₁`: up-projection `[d_model → d_ff]`  
- `W₂`: down-projection `[d_ff → d_model]`
- `d_ff`: FFN hidden dimension (typically 4 × d_model in older models)

**Parameter count.** For LLaMA 3 70B with d_model = 8192:

| Weight | Shape | Parameters |
|--------|-------|-----------|
| W₁ (gate) | 8192 × 28672 | 234M |
| W₃ (up) | 8192 × 28672 | 234M |
| W₂ (down) | 28672 × 8192 | 234M |
| Per layer | — | 702M |
| × 80 layers | — | **56B** |

The FFN holds roughly **two-thirds of the total parameters** in a transformer. Attention gets the spotlight, but the FFN is where most of the weights live.

### Why d_ff ≠ 4 × d_model in modern models

Older models (GPT-2, BERT) used `d_ff = 4 × d_model`. LLaMA and its descendants use `d_ff ≈ 2.67 × d_model` (8/3 ratio) combined with the **SwiGLU** gated activation, which uses *two* up-projections instead of one:

```
SwiGLU(x) = W₂ · (SiLU(W₁ · x) ⊙ W₃ · x)
```

The gating mechanism (`W₃ · x`) acts as a learned mask — it suppresses FFN neurons that are irrelevant for the current token. This improves model quality at the same parameter count, which is why it replaced vanilla GeLU/ReLU in every major post-2022 LLM.

### Activation functions in production

| Model family | Activation | Notes |
|---|---|---|
| GPT-2, BERT | GeLU | Original; smooth approximation to ReLU |
| GPT-3, PaLM | GeLU / SwiGLU | Mixed generation |
| LLaMA 1/2/3 | SwiGLU | d_ff = 8/3 × d_model |
| Mistral 7B | SwiGLU | Same as LLaMA |
| DeepSeek-V2/V3 | SwiGLU + MoE | Sparse FFN (see §3.5.5) |
| Qwen 2.5 | SwiGLU | Standard |
| Nemotron | GeLU | NVIDIA variant |

**SiLU** (Sigmoid Linear Unit, also called Swish) is `x · σ(x)`. Unlike ReLU it is smooth at zero, unlike GeLU it is exact (no approximation). Modern GPU kernels compute it in a single fused pass.

---

## 3.5.4 Complete Block Forward Pass

**Figure 3.5.1 — One Transformer Block: Residual Stream Data Flow**

<img src="data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgNTQwIDUwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogICAgIHN0eWxlPSJtYXgtd2lkdGg6NTYwcHg7Zm9udC1mYW1pbHk6R2VvcmdpYSxzZXJpZjtmb250LXNpemU6MTJweDtkaXNwbGF5OmJsb2NrIj4KICA8ZGVmcz4KICAgIDxtYXJrZXIgaWQ9ImFycjM1IiBtYXJrZXJXaWR0aD0iOCIgbWFya2VySGVpZ2h0PSI4IiByZWZYPSI0IiByZWZZPSI0IiBvcmllbnQ9ImF1dG8iPgogICAgICA8cGF0aCBkPSJNMSwxIEw3LDQgTDEsNyB6IiBmaWxsPSIjMzc0MTUxIi8+CiAgICA8L21hcmtlcj4KICA8L2RlZnM+CiAgPCEtLSBiYWNrZ3JvdW5kIC0tPgogIDxyZWN0IHdpZHRoPSI1NDAiIGhlaWdodD0iNTAwIiBmaWxsPSIjZjlmYWZiIiByeD0iNiIgc3Ryb2tlPSIjZTVlN2ViIiBzdHJva2Utd2lkdGg9IjEiLz4KCiAgPCEtLSByZXNpZHVhbCBzcGluZSAtLT4KICA8bGluZSB4MT0iOTAiIHkxPSIyNSIgeDI9IjkwIiB5Mj0iNDc4IiBzdHJva2U9IiM5Y2EzYWYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtZGFzaGFycmF5PSI1LDMiLz4KICA8dGV4dCB4PSI5MCIgeT0iMTYiIGZpbGw9IiM2YjcyODAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTAiIGZvbnQtc3R5bGU9Iml0YWxpYyI+cmVzaWR1YWwgc3RyZWFtPC90ZXh0PgoKICA8IS0tIHhfaW4gYm94IC0tPgogIDxyZWN0IHg9IjUwIiB5PSIyNSIgd2lkdGg9IjgwIiBoZWlnaHQ9IjMyIiByeD0iNSIgZmlsbD0iI2RiZWFmZSIgc3Ryb2tlPSIjMjU2M2ViIiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDx0ZXh0IHg9IjkwIiB5PSI0NiIgZmlsbD0iIzFkNGVkOCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9ImJvbGQiPnhfaW4gW0IsbixkXTwvdGV4dD4KCiAgPCEtLSBmb3JrIGxpbmUgdG8gUk1TTm9ybSAxIC0tPgogIDxsaW5lIHgxPSI5MCIgeTE9IjU3IiB4Mj0iOTAiIHkyPSI4MCIgc3Ryb2tlPSIjMzc0MTUxIiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDxsaW5lIHgxPSI5MCIgeTE9IjgwIiB4Mj0iMjEwIiB5Mj0iODAiIHN0cm9rZT0iIzM3NDE1MSIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4KICA8bGluZSB4MT0iMjEwIiB5MT0iODAiIHgyPSIyMTAiIHkyPSIxMDAiIHN0cm9rZT0iIzM3NDE1MSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYXJyMzUpIi8+CgogIDwhLS0gUk1TTm9ybSAxIC0tPgogIDxyZWN0IHg9IjE1MCIgeT0iMTAwIiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjM0IiByeD0iNSIgZmlsbD0iI2ZlZjljMyIgc3Ryb2tlPSIjY2E4YTA0IiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDx0ZXh0IHg9IjIxMCIgeT0iMTIyIiBmaWxsPSIjOTI0MDBlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjExIiBmb250LXdlaWdodD0iYm9sZCI+Uk1TTm9ybTwvdGV4dD4KCiAgPCEtLSBhcnJvdyB0byBBdHRlbnRpb24gLS0+CiAgPGxpbmUgeDE9IjIxMCIgeTE9IjEzNCIgeDI9IjIxMCIgeTI9IjE1NCIgc3Ryb2tlPSIjMzc0MTUxIiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhcnIzNSkiLz4KCiAgPCEtLSBBdHRlbnRpb24gYm94IC0tPgogIDxyZWN0IHg9IjE0MCIgeT0iMTU0IiB3aWR0aD0iMTQwIiBoZWlnaHQ9IjUyIiByeD0iNSIgZmlsbD0iI2RiZWFmZSIgc3Ryb2tlPSIjMjU2M2ViIiBzdHJva2Utd2lkdGg9IjIiLz4KICA8dGV4dCB4PSIyMTAiIHk9IjE3NiIgZmlsbD0iIzFlNDBhZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMiIgZm9udC13ZWlnaHQ9ImJvbGQiPkF0dGVudGlvbjwvdGV4dD4KICA8dGV4dCB4PSIyMTAiIHk9IjE5NCIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMCI+USwgSywgViBwcm9qZWN0aW9ucyArIE1IQTwvdGV4dD4KCiAgPCEtLSBhcnJvdyBvdXQgb2YgYXR0ZW50aW9uIC0tPgogIDxsaW5lIHgxPSIyMTAiIHkxPSIyMDYiIHgyPSIyMTAiIHkyPSIyMzAiIHN0cm9rZT0iIzM3NDE1MSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYXJyMzUpIi8+CiAgPHRleHQgeD0iMjIwIiB5PSIyMjQiIGZpbGw9IiM2YjcyODAiIGZvbnQtc2l6ZT0iMTAiPs6UX2F0dG48L3RleHQ+CgogIDwhLS0gcmVzaWR1YWwgYWRkIGNpcmNsZSAxIC0tPgogIDxjaXJjbGUgY3g9IjkwIiBjeT0iMjQ0IiByPSIxNiIgZmlsbD0iI2ZmZmZmZiIgc3Ryb2tlPSIjMDU5NjY5IiBzdHJva2Utd2lkdGg9IjIiLz4KICA8dGV4dCB4PSI5MCIgeT0iMjUwIiBmaWxsPSIjMDU5NjY5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjE4IiBmb250LXdlaWdodD0iYm9sZCI+KzwvdGV4dD4KICA8IS0tIGhvcml6b250YWwgbWVyZ2UgbGluZSAtLT4KICA8bGluZSB4MT0iOTAiIHkxPSI1NyIgeDI9IjkwIiB5Mj0iMjI4IiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMS41Ii8+CiAgPGxpbmUgeDE9IjkwIiB5MT0iMjQ0IiB4Mj0iMjEwIiB5Mj0iMjQ0IiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMS41Ii8+CiAgPGxpbmUgeDE9IjIxMCIgeTE9IjIzMCIgeDI9IjIxMCIgeTI9IjI0NCIgc3Ryb2tlPSIjMzc0MTUxIiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDx0ZXh0IHg9IjQ0IiB5PSIyNDgiIGZpbGw9IiMwNTk2NjkiIGZvbnQtc2l6ZT0iOSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+eCArPSDOlF9hdHRuPC90ZXh0PgoKICA8IS0tIGZvcmsgdG8gUk1TTm9ybSAyIC0tPgogIDxsaW5lIHgxPSI5MCIgeTE9IjI2MCIgeDI9IjkwIiB5Mj0iMjg2IiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMS41Ii8+CiAgPGxpbmUgeDE9IjkwIiB5MT0iMjg2IiB4Mj0iMjEwIiB5Mj0iMjg2IiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMS41Ii8+CiAgPGxpbmUgeDE9IjIxMCIgeTE9IjI4NiIgeDI9IjIxMCIgeTI9IjMwNiIgc3Ryb2tlPSIjMzc0MTUxIiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhcnIzNSkiLz4KCiAgPCEtLSBSTVNOb3JtIDIgLS0+CiAgPHJlY3QgeD0iMTUwIiB5PSIzMDYiIHdpZHRoPSIxMjAiIGhlaWdodD0iMzQiIHJ4PSI1IiBmaWxsPSIjZmVmOWMzIiBzdHJva2U9IiNjYThhMDQiIHN0cm9rZS13aWR0aD0iMS41Ii8+CiAgPHRleHQgeD0iMjEwIiB5PSIzMjgiIGZpbGw9IiM5MjQwMGUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSJib2xkIj5STVNOb3JtPC90ZXh0PgoKICA8IS0tIGFycm93IHRvIEZGTiAtLT4KICA8bGluZSB4MT0iMjEwIiB5MT0iMzQwIiB4Mj0iMjEwIiB5Mj0iMzYwIiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycjM1KSIvPgoKICA8IS0tIEZGTiBib3ggLS0+CiAgPHJlY3QgeD0iMTQwIiB5PSIzNjAiIHdpZHRoPSIxNDAiIGhlaWdodD0iNTIiIHJ4PSI1IiBmaWxsPSIjZmZlZGQ1IiBzdHJva2U9IiNlYTU4MGMiIHN0cm9rZS13aWR0aD0iMiIvPgogIDx0ZXh0IHg9IjIxMCIgeT0iMzgyIiBmaWxsPSIjYzI0MTBjIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEyIiBmb250LXdlaWdodD0iYm9sZCI+U3dpR0xVIEZGTjwvdGV4dD4KICA8dGV4dCB4PSIyMTAiIHk9IjQwMCIgZmlsbD0iIzZiNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1zaXplPSIxMCI+V+KCgsK3KFNpTFUoV+KCgXgpIOKKmSBX4oKDeCk8L3RleHQ+CgogIDwhLS0gYXJyb3cgb3V0IG9mIEZGTiAtLT4KICA8bGluZSB4MT0iMjEwIiB5MT0iNDEyIiB4Mj0iMjEwIiB5Mj0iNDM0IiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycjM1KSIvPgogIDx0ZXh0IHg9IjIyMCIgeT0iNDI4IiBmaWxsPSIjNmI3MjgwIiBmb250LXNpemU9IjEwIj7OlF9mZm48L3RleHQ+CgogIDwhLS0gcmVzaWR1YWwgYWRkIGNpcmNsZSAyIC0tPgogIDxjaXJjbGUgY3g9IjkwIiBjeT0iNDQ4IiByPSIxNiIgZmlsbD0iI2ZmZmZmZiIgc3Ryb2tlPSIjMDU5NjY5IiBzdHJva2Utd2lkdGg9IjIiLz4KICA8dGV4dCB4PSI5MCIgeT0iNDU0IiBmaWxsPSIjMDU5NjY5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjE4IiBmb250LXdlaWdodD0iYm9sZCI+KzwvdGV4dD4KICA8bGluZSB4MT0iOTAiIHkxPSIyNjAiIHgyPSI5MCIgeTI9IjQzMiIgc3Ryb2tlPSIjMzc0MTUxIiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDxsaW5lIHgxPSI5MCIgeTE9IjQ0OCIgeDI9IjIxMCIgeTI9IjQ0OCIgc3Ryb2tlPSIjMzc0MTUxIiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDxsaW5lIHgxPSIyMDAiIHkxPSI0MzQiIHgyPSIyMTAiIHkyPSI0NDgiIHN0cm9rZT0iIzM3NDE1MSIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4KICA8dGV4dCB4PSI0NCIgeT0iNDUyIiBmaWxsPSIjMDU5NjY5IiBmb250LXNpemU9IjkiIHRleHQtYW5jaG9yPSJtaWRkbGUiPnggKz0gzpRfZmZuPC90ZXh0PgoKICA8IS0tIHhfb3V0IC0tPgogIDxsaW5lIHgxPSI5MCIgeTE9IjQ2NCIgeDI9IjkwIiB5Mj0iNDc4IiBzdHJva2U9IiMzNzQxNTEiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycjM1KSIvPgogIDxyZWN0IHg9IjUwIiB5PSI0NjIiIHdpZHRoPSI4MCIgaGVpZ2h0PSIyNCIgcng9IjUiIGZpbGw9IiNkYmVhZmUiIHN0cm9rZT0iIzI1NjNlYiIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4KICA8dGV4dCB4PSI5MCIgeT0iNDc5IiBmaWxsPSIjMWQ0ZWQ4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEwIiBmb250LXdlaWdodD0iYm9sZCI+eF9vdXQg4oaSIG5leHQgbGF5ZXI8L3RleHQ+CgogIDwhLS0gcmlnaHQtc2lkZSBhbm5vdGF0aW9ucyAtLT4KICA8bGluZSB4MT0iMjcwIiB5MT0iMTE3IiB4Mj0iMzQ1IiB5Mj0iMTE3IiBzdHJva2U9IiNkMWQ1ZGIiIHN0cm9rZS13aWR0aD0iMC44IiBzdHJva2UtZGFzaGFycmF5PSIzLDIiLz4KICA8dGV4dCB4PSIzNDgiIHk9IjExMiIgZmlsbD0iIzkyNDAwZSIgZm9udC1zaXplPSIxMCI+zrMgwrcgeCAvIFJNUyh4KTwvdGV4dD4KICA8dGV4dCB4PSIzNDgiIHk9IjEyNiIgZmlsbD0iIzZiNzI4MCIgZm9udC1zaXplPSI5Ij5zdGFiaWxpc2VzIGFjdGl2YXRpb25zPC90ZXh0PgoKICA8bGluZSB4MT0iMjgwIiB5MT0iMTc4IiB4Mj0iMzQ1IiB5Mj0iMTc4IiBzdHJva2U9IiNkMWQ1ZGIiIHN0cm9rZS13aWR0aD0iMC44IiBzdHJva2UtZGFzaGFycmF5PSIzLDIiLz4KICA8dGV4dCB4PSIzNDgiIHk9IjE3NCIgZmlsbD0iIzFlNDBhZiIgZm9udC1zaXplPSIxMCI+S1YgY2FjaGUgcmVhZCBoZXJlPC90ZXh0PgogIDx0ZXh0IHg9IjM0OCIgeT0iMTg4IiBmaWxsPSIjNmI3MjgwIiBmb250LXNpemU9IjkiPm9ubHkgSyxWIHRlbnNvcnMgY2FjaGVkPC90ZXh0PgoKICA8bGluZSB4MT0iMjgwIiB5MT0iMzgyIiB4Mj0iMzQ1IiB5Mj0iMzgyIiBzdHJva2U9IiNkMWQ1ZGIiIHN0cm9rZS13aWR0aD0iMC44IiBzdHJva2UtZGFzaGFycmF5PSIzLDIiLz4KICA8dGV4dCB4PSIzNDgiIHk9IjM3OCIgZmlsbD0iI2MyNDEwYyIgZm9udC1zaXplPSIxMCI+fjIvMyBvZiBtb2RlbCBwYXJhbXM8L3RleHQ+CiAgPHRleHQgeD0iMzQ4IiB5PSIzOTIiIGZpbGw9IiM2YjcyODAiIGZvbnQtc2l6ZT0iOSI+ZF9mZiDiiYggMi42NyDDlyBkX21vZGVsPC90ZXh0Pgo8L3N2Zz4=" style="max-width:560px;width:100%;display:block;margin:1rem 0" alt="diagram"/>



A single transformer block processes input `x` of shape `[B, n, d_model]` as follows:

```
Step 1 — Pre-attention norm:
    x̂ = RMSNorm(x)

Step 2 — Attention:
    Δ_attn = Attention(x̂)     # shape [B, n, d_model]
    x = x + Δ_attn             # residual add

Step 3 — Pre-FFN norm:
    x̂ = RMSNorm(x)

Step 4 — FFN:
    gate = SiLU(W₁ · x̂)       # [B, n, d_ff]
    up   = W₃ · x̂              # [B, n, d_ff]
    Δ_ffn = W₂ · (gate ⊙ up)  # [B, n, d_model]
    x = x + Δ_ffn              # residual add

Output: x    # [B, n, d_model], updated residual stream
```

The block output feeds directly into the next layer's RMSNorm.

### FLOPs per block (single token, decode phase)

At decode, `n = 1` (one new token). With B = 1 for clarity:

| Operation | FLOPs |
|---|---|
| RMSNorm × 2 | 4 × d_model |
| Q/K/V projections | 6 × d_model² |
| Attention (single token) | ~4 × d_model (dot products vs cache) |
| O projection | 2 × d_model² |
| FFN W₁, W₃ | 4 × d_model × d_ff |
| FFN W₂ | 2 × d_ff × d_model |
| **Total** | **~8 × d_model² + 8 × d_model × d_ff** |

For LLaMA 3 70B (d_model=8192, d_ff=28672):

```
8 × 8192² + 8 × 8192 × 28672
= 537M + 1878M
≈ 2.4 GFLOPs per block per token
× 80 blocks = 194 GFLOPs per generated token
```

An H100 SXM5 does ~2000 TFLOPS (BF16 Tensor Core). At 100% efficiency, it could generate:

```
2000 × 10¹² / (194 × 10⁹) ≈ 10,300 tokens/second
```

Real throughput is 1,000–2,500 tok/s — about 4–10× below theoretical peak — because memory bandwidth, not compute, is the bottleneck during decode (see Chapter 2).

---

## 3.5.5 Mixture of Experts: Sparse FFN

MoE models (DeepSeek-V2/V3, Mixtral, Qwen MoE) replace the dense FFN with a **routing + sparse selection** mechanism:

```
MoE_FFN(x) = Σᵢ₌₁ᵏ router_score(x, i) · FFN_i(x)
```

Where:

- `N` total experts (e.g., 64 for DeepSeek-V3's shared+routed design)
- `k` active experts per token (e.g., top-2 or top-8)
- Each `FFN_i` is a full-sized feed-forward network

**Serving implication:** only k/N of the FFN weights are active per token, which reduces FLOPs dramatically. But **all N expert weights must fit in memory** — they cannot be swapped. DeepSeek-V3 has 671B total parameters but only 37B active per token. This is why MoE models need multi-GPU serving even though their compute is modest.

**Router load balancing** adds a training-time auxiliary loss to prevent all tokens from routing to the same expert ("expert collapse"). During inference, the router runs greedily without the auxiliary term.

Chapter 34 covers DeepSeek's MLA + MoE combination in detail.

---

## 3.5.6 Embedding and Unembedding Layers

The transformer block sits between two layers that connect the integer token world to the continuous vector world:

**Token embedding** (input side):
```
x₀ = E[token_id]    # lookup in E ∈ ℝ^(vocab_size × d_model)
```

For LLaMA 3 70B: vocab=128K, d_model=8192 → E is 128K × 8192 = **2GB in BF16**.

**Unembedding / LM head** (output side):
```
logits = x_final · Eᵀ    # [B, n, vocab_size]
```

Many models (including LLaMA) **tie weights** — the LM head reuses the same matrix E transposed. This saves 2GB at no accuracy cost. vLLM detects and honours weight tying automatically.

**RoPE positional encoding** is applied to Q and K *inside* the attention sub-layer (not to the embeddings), which is why positional information flows only through attention, not through the FFN. This matters for long-context extension strategies (Chapter 27).

---

## 3.5.7 KV Cache: What Is Actually Being Cached

Now that you know the full block structure, the KV cache is easy to define precisely:

At each layer, the attention sub-layer projects the residual stream into Q, K, V:

```
Q = x̂ · W_Q    # [n, d_head × H]
K = x̂ · W_K    # [n, d_head × H_kv]
V = x̂ · W_V    # [n, d_head × H_kv]
```

During **prefill** (processing the prompt), K and V for every position are computed and saved to the KV cache.

During **decode** (generating tokens), only the *new* token's K and V are computed; the cached K and V from previous tokens are appended to the cache and reused.

The cache holds K and V tensors for every layer, every head, and every cached position:

```
KV_cache_bytes = 2 × n_layers × H_kv × d_head × n_cached × P
```

Where the factor of 2 is for K and V. For LLaMA 3 70B at n=4096:

```
2 × 80 × 8 × 128 × 4096 × 2 bytes = 1.34 GB
```

Chapter 6 covers how vLLM manages this cache across many concurrent requests using PagedAttention.

---

## 3.5.8 Parameter Count Summary

| Component | LLaMA 3 8B | LLaMA 3 70B |
|---|---|---|
| Embedding | 0.5B | 2.1B |
| Attention (all layers) | 1.0B | 9.4B |
| FFN (all layers) | 5.5B | 56B |
| Layer norms | negligible | negligible |
| LM head (tied) | shared | shared |
| **Total** | **~8B** | **~70B** |

The FFN dominates. This is why FFN quantization (INT4/INT8 on W₁, W₂, W₃) gives large memory savings with modest quality loss, while attention quantization requires more care (Chapter 10).

---

## 3.5.9 Worked Example: One Block, One Token

Let us trace a single decode step through one transformer block with toy numbers: d_model=4, d_ff=8, 1 attention head, BF16.

**Input** (residual stream at this layer, new token):
```
x = [0.5, -0.3, 0.8, 0.1]
```

**Step 1 — RMSNorm:**
```
RMS = √((0.5² + 0.3² + 0.8² + 0.1²) / 4) = √(0.2475) ≈ 0.497
x̂ = x / 0.497 × γ  (assume γ=1)
  = [1.006, -0.604, 1.610, 0.201]
```

**Step 2 — Attention delta:**
The token attends to the KV cache (all prior positions) and produces Δ_attn. For this example, assume:
```
Δ_attn = [0.1, 0.2, -0.1, 0.05]
x = x + Δ_attn = [0.6, -0.1, 0.7, 0.15]
```

**Step 3 — RMSNorm again:**
```
RMS = √((0.6² + 0.1² + 0.7² + 0.15²) / 4) ≈ 0.447
x̂ = [1.342, -0.224, 1.566, 0.335]
```

**Step 4 — FFN (SwiGLU, simplified to d_ff=2):**
```
gate = SiLU(W₁ · x̂)  # two numbers
up   = W₃ · x̂         # two numbers
Δ_ffn = W₂ · (gate ⊙ up)  # back to 4 numbers
x = x + Δ_ffn
```

The FFN reads the normalized residual, amplifies features relevant to predicting the next token, and adds that delta back. This happens 80 times. The final residual stream is projected to vocab_size logits.

---

## 3.5.10 Code: Transformer Block from Scratch

See `code/chapter_03b/transformer_block_demo.py` and `code/chapter_03b/transformer_block_demo.cpp`.

### 3.5.10.1 Python: Full Block with RMSNorm, SwiGLU FFN, and Residual Stream

```python
# transformer_block_demo.py
# Chapter 3.5 — The Transformer Block
#
# Implements:
#   1. RMSNorm
#   2. SwiGLU FFN
#   3. Full transformer block forward pass (attention stubbed)
#   4. Parameter count and memory budget
#   5. KV cache size calculations
#
# Requirements: pip install numpy
# No GPU needed.

import numpy as np
from dataclasses import dataclass
from typing import Optional

# ── Model config ──────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    d_model: int    # residual stream dimension
    d_ff: int       # FFN hidden dimension
    n_heads: int    # attention heads (for KV cache sizing)
    n_kv_heads: int # KV heads (GQA)
    d_head: int     # head dimension
    n_layers: int   # total transformer layers
    vocab_size: int # vocabulary size

LLAMA3_8B  = ModelConfig(4096,  14336, 32, 8,  128, 32, 128256)
LLAMA3_70B = ModelConfig(8192,  28672, 64, 8,  128, 80, 128256)
LLAMA3_1B  = ModelConfig(2048,   8192, 32, 8,   64, 16, 128256)
TOY        = ModelConfig(8,         16,  2, 2,    4,  2,     16)

# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm:
    def __init__(self, d: int, eps: float = 1e-6):
        self.d = d
        self.eps = eps
        self.gamma = np.ones(d, dtype=np.float32)   # learned scale, init=1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: [..., d] → [..., d]"""
        # Accumulate in float32 for numerical stability
        rms = np.sqrt(np.mean(x.astype(np.float64)**2, axis=-1, keepdims=True) + self.eps)
        return (x / rms * self.gamma).astype(x.dtype)

    def params(self) -> int:
        return self.d

# ── SwiGLU FFN ────────────────────────────────────────────────────────────────

def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish: x * sigmoid(x)"""
    return x / (1 + np.exp(-x.astype(np.float64)))

class SwiGLU_FFN:
    """
    SwiGLU Feed-Forward Network.
    FFN(x) = W2 · (SiLU(W1 · x) ⊙ W3 · x)
    """
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        scale = 0.02
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, scale, (d_model, d_ff)).astype(np.float32)
        self.W3 = rng.normal(0, scale, (d_model, d_ff)).astype(np.float32)
        self.W2 = rng.normal(0, scale, (d_ff, d_model)).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: [..., d_model] → [..., d_model]"""
        gate = silu(x @ self.W1)   # [..., d_ff]
        up   = x @ self.W3         # [..., d_ff]
        return (gate * up) @ self.W2  # [..., d_model]

    def params(self) -> int:
        return self.d_model * self.d_ff * 3  # W1 + W2 + W3

# ── Stub attention (black box for this chapter) ───────────────────────────────

class StubAttention:
    """
    Attention is treated as a black box here — fully implemented in Chapter 4.
    This stub adds a small learned delta to test the residual stream plumbing.
    """
    def __init__(self, d_model: int):
        self.d_model = d_model
        rng = np.random.default_rng(99)
        self.W = rng.normal(0, 0.01, (d_model, d_model)).astype(np.float32)

    def forward(self, x: np.ndarray,
                kv_cache: Optional[np.ndarray] = None) -> np.ndarray:
        return x @ self.W

    def params(self) -> int:
        return self.d_model ** 2

# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock:
    """One full transformer block: pre-norm → attention → residual,
       pre-norm → FFN → residual."""
    def __init__(self, cfg: ModelConfig):
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn  = StubAttention(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = SwiGLU_FFN(cfg.d_model, cfg.d_ff)

    def forward(self, x: np.ndarray,
                kv_cache: Optional[np.ndarray] = None) -> np.ndarray:
        # Pre-attention norm + attention + residual
        x = x + self.attn.forward(self.norm1.forward(x), kv_cache)
        # Pre-FFN norm + FFN + residual
        x = x + self.ffn.forward(self.norm2.forward(x))
        return x

    def params(self) -> int:
        return (self.norm1.params() + self.attn.params() +
                self.norm2.params() + self.ffn.params())

# ── Memory and FLOPs accounting ───────────────────────────────────────────────

def model_memory_gb(cfg: ModelConfig, bytes_per_param: int = 2) -> float:
    """BF16 by default. Use 4 for FP32, 1 for INT8."""
    embedding = cfg.vocab_size * cfg.d_model
    per_layer = (
        4 * cfg.d_model**2 +          # Q K V O projections (approx MHA)
        3 * cfg.d_model * cfg.d_ff +  # W1 W2 W3
        2 * cfg.d_model               # two RMSNorm gamma vectors
    )
    total_params = embedding + per_layer * cfg.n_layers
    return total_params * bytes_per_param / 1e9

def kv_cache_gb(cfg: ModelConfig, seq_len: int,
                batch_size: int = 1, bytes_per_elem: int = 2) -> float:
    """KV cache for one batch at given sequence length."""
    return (2 * cfg.n_layers * cfg.n_kv_heads * cfg.d_head *
            seq_len * batch_size * bytes_per_elem) / 1e9

def flops_per_token(cfg: ModelConfig) -> float:
    """Approximate FLOPs for one decode step (n=1)."""
    attn_flops = 8 * cfg.d_model**2         # Q K V O projections
    ffn_flops  = 8 * cfg.d_model * cfg.d_ff # W1 W3 up + W2 down (SwiGLU)
    return (attn_flops + ffn_flops) * cfg.n_layers

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SECTION 1: Worked Example — Toy Block (d=8, d_ff=16)")
    print("=" * 60)

    cfg = TOY
    block = TransformerBlock(cfg)

    # Simulate a residual stream: 3 tokens, d_model=8
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (3, cfg.d_model)).astype(np.float32)
    print(f"\nInput residual stream shape:  {x.shape}")
    print(f"Input (first token): {x[0].round(4)}")

    x_out = block.forward(x)
    print(f"Output residual stream shape: {x_out.shape}")
    print(f"Output (first token): {x_out[0].round(4)}")
    assert x_out.shape == x.shape, "Shape mismatch!"
    print("✓ Shape preserved through block.\n")

    # Demonstrate residual: output ≠ input (delta was added)
    delta = x_out - x
    print(f"Delta norm (first token): {np.linalg.norm(delta[0]):.4f}")
    print(f"Input norm (first token): {np.linalg.norm(x[0]):.4f}")
    print("(Delta << Input is typical in well-trained models)\n")

    print("=" * 60)
    print("SECTION 2: RMSNorm Properties")
    print("=" * 60)
    norm = RMSNorm(cfg.d_model)
    x_test = rng.normal(0, 5, (4, cfg.d_model)).astype(np.float32)
    x_normed = norm.forward(x_test)
    rms_before = np.sqrt(np.mean(x_test**2, axis=-1))
    rms_after  = np.sqrt(np.mean(x_normed**2, axis=-1))
    print(f"\nRMS before normalization: {rms_before.round(3)}")
    print(f"RMS after  normalization: {rms_after.round(3)}")
    print("✓ RMSNorm brings all tokens to unit RMS.\n")

    print("=" * 60)
    print("SECTION 3: SwiGLU vs Plain FFN Activation")
    print("=" * 60)
    ffn = SwiGLU_FFN(cfg.d_model, cfg.d_ff)
    x_in = rng.normal(0, 1, (1, cfg.d_model)).astype(np.float32)
    x_ffn_out = ffn.forward(norm.forward(x_in))
    # SiLU values
    sample = np.linspace(-4, 4, 9)
    silu_vals = silu(sample)
    print(f"\nSiLU(x) for x in [-4..4]:")
    for xi, si in zip(sample, silu_vals):
        bar = "█" * int(max(0, si) * 10)
        print(f"  x={xi:+.1f}  SiLU={si:+.4f}  {bar}")
    print(f"\nFFN output shape: {x_ffn_out.shape}  (same as input ✓)\n")

    print("=" * 60)
    print("SECTION 4: Parameter Count and Memory Budget")
    print("=" * 60)
    configs = [
        ("LLaMA 3 1B",  LLAMA3_1B),
        ("LLaMA 3 8B",  LLAMA3_8B),
        ("LLaMA 3 70B", LLAMA3_70B),
    ]
    print(f"\n{'Model':<16}  {'Params (B)':<12}  {'BF16 (GB)':<12}  {'INT4 (GB)':<12}")
    print("-" * 56)
    for name, cfg_m in configs:
        mem_bf16 = model_memory_gb(cfg_m, bytes_per_param=2)
        mem_int4 = model_memory_gb(cfg_m, bytes_per_param=0.5)
        # Rough param estimate
        p = (cfg_m.vocab_size * cfg_m.d_model +
             (4*cfg_m.d_model**2 + 3*cfg_m.d_model*cfg_m.d_ff) * cfg_m.n_layers)
        print(f"  {name:<14}  {p/1e9:<12.1f}  {mem_bf16:<12.1f}  {mem_int4:<12.1f}")

    print("\n" + "=" * 60)
    print("SECTION 5: KV Cache Sizing")
    print("=" * 60)
    print(f"\n{'Model':<16}  {'n=2K':<10}  {'n=8K':<10}  {'n=32K':<10}  {'n=128K':<10}")
    print("-" * 56)
    for name, cfg_m in configs:
        row = f"  {name:<14}"
        for seq in [2048, 8192, 32768, 131072]:
            row += f"  {kv_cache_gb(cfg_m, seq):<8.2f}GB"
        print(row)

    print("\n" + "=" * 60)
    print("SECTION 6: FLOPs per Generated Token")
    print("=" * 60)
    print(f"\n{'Model':<16}  {'GFLOPs/tok':<14}  {'H100 tok/s (theoretical)'}")
    print("-" * 60)
    H100_TFLOPS = 1979e12  # BF16 Tensor Core
    for name, cfg_m in configs:
        f = flops_per_token(cfg_m)
        tps = H100_TFLOPS / f
        print(f"  {name:<14}  {f/1e9:<14.1f}  {tps:.0f} (practical: ~10x less)")

    print("\nDone.")

if __name__ == "__main__":
    main()
```

### 3.5.10.2 C++: Transformer Block — RMSNorm, SwiGLU, Residual Stream

```cpp
// transformer_block_demo.cpp
// Chapter 3.5 — The Transformer Block
//
// Implements:
//   1. RMSNorm
//   2. SiLU activation
//   3. SwiGLU FFN
//   4. Full block forward pass (attention stubbed)
//   5. Parameter and memory budgets
//
// Build:
//   g++ -std=c++17 -O2 transformer_block_demo.cpp -o transformer_block_demo
//
// Run:
//   ./transformer_block_demo

#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <cassert>
#include <random>
#include <algorithm>
#include <numeric>

using Vec = std::vector<float>;
using Mat = std::vector<Vec>;  // [rows][cols]

// ── Helpers ───────────────────────────────────────────────────────────────────

Mat make_mat(int rows, int cols, float fill = 0.f) {
    return Mat(rows, Vec(cols, fill));
}

// Matrix-vector multiply: y = A * x,  A [rows×cols], x [cols] → y [rows]
Vec matvec(const Mat& A, const Vec& x) {
    int rows = A.size(), cols = A[0].size();
    assert((int)x.size() == cols);
    Vec y(rows, 0.f);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            y[i] += A[i][j] * x[j];
    return y;
}

Vec vec_add(const Vec& a, const Vec& b) {
    Vec c(a.size());
    for (int i = 0; i < (int)a.size(); i++) c[i] = a[i] + b[i];
    return c;
}

Vec vec_mul(const Vec& a, const Vec& b) {   // elementwise
    Vec c(a.size());
    for (int i = 0; i < (int)a.size(); i++) c[i] = a[i] * b[i];
    return c;
}

float vec_norm(const Vec& v) {
    float s = 0; for (auto x : v) s += x*x; return std::sqrt(s);
}

// Gaussian random weight init
Mat rand_mat(int rows, int cols, float std_dev, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.f, std_dev);
    Mat M = make_mat(rows, cols);
    for (auto& row : M) for (auto& v : row) v = dist(rng);
    return M;
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

struct RMSNorm {
    int d;
    float eps;
    Vec gamma;

    RMSNorm(int d, float eps = 1e-6f) : d(d), eps(eps), gamma(d, 1.f) {}

    Vec forward(const Vec& x) const {
        assert((int)x.size() == d);
        double sum_sq = 0.0;
        for (auto v : x) sum_sq += (double)v * v;
        float rms = std::sqrt((float)(sum_sq / d) + eps);
        Vec out(d);
        for (int i = 0; i < d; i++) out[i] = x[i] / rms * gamma[i];
        return out;
    }
};

// ── SiLU / SwiGLU ─────────────────────────────────────────────────────────────

float silu(float x) { return x / (1.f + std::exp(-x)); }

struct SwiGLU_FFN {
    int d_model, d_ff;
    Mat W1, W2, W3;

    SwiGLU_FFN(int d_model, int d_ff, std::mt19937& rng)
        : d_model(d_model), d_ff(d_ff)
    {
        W1 = rand_mat(d_ff, d_model, 0.02f, rng);
        W3 = rand_mat(d_ff, d_model, 0.02f, rng);
        W2 = rand_mat(d_model, d_ff, 0.02f, rng);
    }

    Vec forward(const Vec& x) const {
        Vec g1 = matvec(W1, x);           // [d_ff]
        Vec g3 = matvec(W3, x);           // [d_ff]
        Vec gate(d_ff);
        for (int i = 0; i < d_ff; i++) gate[i] = silu(g1[i]) * g3[i];
        return matvec(W2, gate);           // [d_model]
    }

    long long params() const { return 3LL * d_model * d_ff; }
};

// ── Stub attention ────────────────────────────────────────────────────────────

struct StubAttention {
    int d_model;
    Mat W;

    StubAttention(int d_model, std::mt19937& rng) : d_model(d_model) {
        W = rand_mat(d_model, d_model, 0.01f, rng);
    }

    Vec forward(const Vec& x) const { return matvec(W, x); }
    long long params() const { return (long long)d_model * d_model; }
};

// ── Transformer Block ─────────────────────────────────────────────────────────

struct TransformerBlock {
    RMSNorm     norm1, norm2;
    StubAttention attn;
    SwiGLU_FFN  ffn;

    TransformerBlock(int d_model, int d_ff, std::mt19937& rng)
        : norm1(d_model), norm2(d_model),
          attn(d_model, rng), ffn(d_model, d_ff, rng)
    {}

    Vec forward(const Vec& x) const {
        // Pre-attn norm + attn + residual
        Vec x1 = vec_add(x, attn.forward(norm1.forward(x)));
        // Pre-FFN norm + FFN + residual
        Vec x2 = vec_add(x1, ffn.forward(norm2.forward(x1)));
        return x2;
    }

    long long params() const {
        return norm1.d + norm2.d + attn.params() + ffn.params();
    }
};

// ── Memory accounting ─────────────────────────────────────────────────────────

struct ModelConfig { const char* name; int d,ff,nl,h_kv,d_head,vocab; };

double model_memory_gb(const ModelConfig& c, int bpp = 2) {
    long long embedding = (long long)c.vocab * c.d;
    long long per_layer = 4LL*c.d*c.d + 3LL*c.d*c.ff + 2*c.d;
    return (embedding + per_layer * c.nl) * bpp / 1e9;
}

double kv_cache_gb(const ModelConfig& c, int seq, int bpp = 2) {
    return 2.0 * c.nl * c.h_kv * c.d_head * seq * bpp / 1e9;
}

double flops_per_token(const ModelConfig& c) {
    return (8.0*c.d*c.d + 8.0*c.d*c.ff) * c.nl;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::mt19937 rng(42);

    // ── Section 1: Toy block ─────────────────────────────────────────────────
    printf("============================================================\n");
    printf("SECTION 1: Toy Block  (d=8, d_ff=16)\n");
    printf("============================================================\n\n");

    const int D = 8, FF = 16;
    TransformerBlock block(D, FF, rng);

    std::normal_distribution<float> dist(0.f, 1.f);
    Vec x(D); for (auto& v : x) v = dist(rng);

    printf("Input:  [");
    for (float v : x) printf(" %6.3f", v); printf(" ]\n");

    Vec y = block.forward(x);
    printf("Output: [");
    for (float v : y) printf(" %6.3f", v); printf(" ]\n");
    assert(y.size() == x.size());

    float delta_norm = 0.f;
    for (int i = 0; i < D; i++) delta_norm += (y[i]-x[i])*(y[i]-x[i]);
    delta_norm = std::sqrt(delta_norm);
    printf("Delta norm: %.4f   Input norm: %.4f\n", delta_norm, vec_norm(x));
    printf("✓ Block preserves shape and adds a delta.\n\n");

    // ── Section 2: RMSNorm ───────────────────────────────────────────────────
    printf("============================================================\n");
    printf("SECTION 2: RMSNorm Properties\n");
    printf("============================================================\n\n");

    RMSNorm norm(D);
    Vec xbig(D); for (auto& v : xbig) v = dist(rng) * 5.f;  // large values
    Vec xn = norm.forward(xbig);

    float rms_before = 0.f, rms_after = 0.f;
    for (float v : xbig) rms_before += v*v; rms_before = std::sqrt(rms_before/D);
    for (float v : xn)   rms_after  += v*v; rms_after  = std::sqrt(rms_after /D);
    printf("  RMS before: %.4f\n  RMS after:  %.4f (≈1.0 ✓)\n\n", rms_before, rms_after);

    // ── Section 3: SiLU curve ────────────────────────────────────────────────
    printf("============================================================\n");
    printf("SECTION 3: SiLU Activation Curve\n");
    printf("============================================================\n\n");
    printf("  %-8s  %-10s\n", "x", "SiLU(x)");
    printf("  %s\n", std::string(22, '-').c_str());
    for (float xv = -3.f; xv <= 3.f; xv += 0.75f) {
        float s = silu(xv);
        int bars = std::max(0, (int)((s + 0.5f) * 10));
        printf("  %-8.2f  %+-8.4f  %s\n", xv, s, std::string(bars, '#').c_str());
    }

    // ── Section 4: Memory budget ──────────────────────────────────────────────
    printf("\n============================================================\n");
    printf("SECTION 4: Parameter Count and Memory Budget\n");
    printf("============================================================\n\n");

    ModelConfig models[] = {
        {"LLaMA 3 1B",  2048,  8192, 16, 8,  64, 128256},
        {"LLaMA 3 8B",  4096, 14336, 32, 8, 128, 128256},
        {"LLaMA 3 70B", 8192, 28672, 80, 8, 128, 128256},
    };
    printf("  %-16s  %10s  %10s  %10s\n", "Model", "BF16 (GB)", "INT8 (GB)", "INT4 (GB)");
    printf("  %s\n", std::string(52, '-').c_str());
    for (auto& m : models) {
        printf("  %-16s  %10.1f  %10.1f  %10.1f\n",
               m.name, model_memory_gb(m,2), model_memory_gb(m,1), model_memory_gb(m,0));
    }
    // Note: 0.5 bytes for INT4 — using int to avoid float issues
    printf("  (INT4 uses 0.5 bytes/param)\n");
    printf("\n");
    // redo with proper 0.5 handling:
    printf("  %-16s  %10s\n", "Model", "INT4 actual");
    for (auto& m : models) {
        long long emb = (long long)m.vocab * m.d;
        long long pl  = 4LL*m.d*m.d + 3LL*m.d*m.ff + 2*m.d;
        double params_b = (emb + pl * m.nl) / 1e9;
        printf("  %-16s  %7.1f GB  (%0.1fB params)\n",
               m.name, params_b * 0.5, params_b);
    }

    // ── Section 5: KV cache ───────────────────────────────────────────────────
    printf("\n============================================================\n");
    printf("SECTION 5: KV Cache Sizing\n");
    printf("============================================================\n\n");
    printf("  %-16s  %10s  %10s  %10s  %10s\n",
           "Model", "n=2K", "n=8K", "n=32K", "n=128K");
    printf("  %s\n", std::string(62, '-').c_str());
    for (auto& m : models) {
        printf("  %-16s", m.name);
        for (int s : {2048, 8192, 32768, 131072})
            printf("  %7.2fGB", kv_cache_gb(m, s));
        printf("\n");
    }

    // ── Section 6: FLOPs ──────────────────────────────────────────────────────
    printf("\n============================================================\n");
    printf("SECTION 6: FLOPs per Generated Token\n");
    printf("============================================================\n\n");
    const double H100_TFLOPS = 1979e12;
    printf("  %-16s  %14s  %s\n", "Model", "GFLOPs/token", "H100 theoretical tok/s");
    printf("  %s\n", std::string(56, '-').c_str());
    for (auto& m : models) {
        double f = flops_per_token(m);
        printf("  %-16s  %14.1f  %.0f\n",
               m.name, f/1e9, H100_TFLOPS/f);
    }
    printf("\nDone.\n");
    return 0;
}
```

---

## 3.5.11 Self-Check Questions

1. **[FOUNDATIONAL]** Why does RMSNorm omit the mean-subtraction step compared to LayerNorm? What property of the residual stream makes this safe?

2. **[FOUNDATIONAL]** LLaMA 3 70B uses `d_ff = 28672` with SwiGLU (two weight matrices W₁ and W₃). An older model uses `d_ff = 4 × d_model = 32768` with a single GeLU projection. Which FFN has more parameters? Show the arithmetic.

3. **[DEEP DIVE]** Trace the residual stream through two blocks. If block 1 outputs a delta of norm 0.3 and block 2 outputs a delta of norm 0.1, and the input norm is 1.0, what is the approximate output norm? Why does this matter for floating-point precision in deep models?

4. **[SYSTEMS]** A model has d_model=4096, d_ff=14336, n_layers=32. You quantise only the FFN weights to INT4 and keep attention in BF16. What fraction of model weight memory does each component use before and after? Is this a good trade-off?

5. **[APPLIED]** A DeepSeek-V3-style MoE has 256 experts, each with d_ff=2048, d_model=2048, and top-2 routing. How does its per-token FLOPs compare to a dense model with the same d_model and d_ff=2048? How does its total parameter count compare?

---

## 3b.X  The Llama-3 Reference Architecture — The Book's Implicit Baseline

Throughout this book — roofline calculations in Chapter 2, KV cache formulas in Chapters 4–6, benchmark numbers in Chapters 15–20 — all worked examples use the **Llama-3 8B and 70B architectures** as the canonical reference. This section makes those architectural choices explicit and explains why each choice was made, so readers can adapt the formulas when working with other model families.

### Why Llama-3?

Llama-3 is the most widely deployed open-weights model family as of 2024–2025. Its architecture represents the current consensus on best practices for dense transformer LLMs. Understanding its choices is prerequisite to understanding every performance formula in this book.

### Architectural Choice Table

| Component | Llama-3 Choice | Alternatives | Why Llama-3's Choice |
|-----------|---------------|--------------|----------------------|
| **Position encoding** | RoPE (Rotary Position Embedding) | ALiBi, learned absolute PE, sinusoidal | RoPE rotates Q and K vectors by position-dependent angles; attends at any context length without extrapolation penalty; ALiBi degrades beyond training context; learned absolute PE cannot extend at all |
| **Attention head structure** | GQA (Grouped Query Attention, n_kv_heads=8) | MHA (n_kv_heads=n_q_heads), MQA (n_kv_heads=1) | GQA uses 8 KV heads vs 64 Q heads = 8× KV cache compression with minimal quality loss; MHA is most expressive but most memory-hungry; MQA is maximally compressed but loses too much quality for 70B+ models |
| **normalization** | RMSNorm (pre-norm) | LayerNorm, post-norm | RMSNorm omits mean subtraction: faster (fewer ops), same empirical quality as LayerNorm; pre-norm stabilises deep networks better than post-norm |
| **FFN activation** | SwiGLU: `gate × silu(gate) × up → down` | GeLU, ReLU, GeGLU | Gated units improve quality at the same parameter count; SwiGLU consistently outperforms GeLU in ablations; ReLU is faster but lags quality by ~0.5 PPL |
| **Linear layer biases** | No bias in any linear layer | Bias in QKV, FFN | Removes bias terms that complicate quantization (scale-bias interaction); no measurable quality loss in large models; simplifies INT4/INT8 quantization by eliminating additive offsets |
| **Vocabulary size** | 128,256 (Llama-3) | 32,000 (Llama-1/2), 200,000+ | Larger vocab improves tokenization efficiency for code and multilingual text; embedding table adds ~1 GB but tokenization gain is worth it |

### How GQA Changes the KV Cache Formula

The KV cache size formula from Chapter 2 is:

```
KV_bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × bytes_per_element
```

The factor of 2 counts both K and V tensors. The critical variable is `n_kv_heads`.

**MHA (Multi-Head Attention) — n_kv_heads = n_q_heads = 64 for 70B:**
```
KV = 2 × 80 layers × 64 KV heads × 128 d_head × seq_len × 2 bytes
   = 2 × 80 × 64 × 128 × seq_len × 2
   = 2,621,440 × seq_len bytes
   ≈ 2.5 MB per token
```

**GQA (Grouped Query Attention) — n_kv_heads = 8 for 70B:**
```
KV = 2 × 80 layers × 8 KV heads × 128 d_head × seq_len × 2 bytes
   = 2 × 80 × 8 × 128 × seq_len × 2
   = 327,680 × seq_len bytes
   ≈ 327 KB per token   ← 8× smaller than MHA
```

This 8× reduction is not a rounding error — it is the fundamental reason Llama-3 70B can serve long contexts on 4× A100-80 GPUs. With MHA, the same model would need 8× the KV memory, requiring 8× more GPUs just for the cache.

**Llama-3 8B with GQA (n_kv_heads=8, n_layers=32, d_head=128):**
```
KV = 2 × 32 × 8 × 128 × seq_len × 2
   = 131,072 × seq_len bytes
   ≈ 128 KB per token
```

At a 4K context window with 256 concurrent sessions:
```
Total KV = 256 sessions × 4,096 tokens × 128 KB/token = 128 GB
```
This requires TP=4 across A100-80s — which is exactly why the Case 1 configuration in Section 15.11 uses `tensor_parallel_size: 4`.

### ASCII Diagram: Llama-3 Block Internals

```
Input hidden state  [B, n, d_model]
        │
        ├─── RMSNorm ──────────────────────────────────────────────────────┐
        │    (no mean subtraction; scale parameter only)                   │
        │                                                                  │
        │    ┌─── Wq [d_model, n_q_heads × d_head] ──► Q [B,n,64,128] ─┐ │
        │    ├─── Wk [d_model, n_kv_heads × d_head] ─► K [B,n, 8,128] ─┤ │
        │    └─── Wv [d_model, n_kv_heads × d_head] ─► V [B,n, 8,128] ─┤ │
        │                                                                │ │
        │    RoPE applied to Q and K (not V)                            │ │
        │    (rotate Q,K by position-dependent complex phase)           │ │
        │                                                                │ │
        │    GQA: each of 8 KV head groups serves 8 Q heads             │ │
        │    Attention: softmax(QKᵀ / √d_head) × V                     │ │
        │                                                                │ │
        │    Wo [n_q_heads × d_head, d_model] ──────────────────────────┘ │
        │                                                                  │
        └─── + residual add ◄──────────────────────────────────────────────┘
        │
        │   [KV cache stores K and V only — not Q, not Wo output]
        │
        ├─── RMSNorm ──────────────────────────────────────────────────────┐
        │                                                                  │
        │    SwiGLU FFN:                                                   │
        │      gate = W_gate [d_model, d_ff] × x   (d_ff = 14336 for 8B) │
        │      up   = W_up   [d_model, d_ff] × x                          │
        │      hidden = SiLU(gate) × up             (element-wise)        │
        │      out = W_down [d_ff, d_model] × hidden                      │
        │                                                                  │
        │    No bias in any of W_gate, W_up, W_down                       │
        │                                                                  │
        └─── + residual add ◄──────────────────────────────────────────────┘
        │
Output hidden state [B, n, d_model]
```

### Key Dimensions at a Glance

| Parameter | Llama-3 8B | Llama-3 70B |
|-----------|-----------|------------|
| d_model | 4,096 | 8,192 |
| n_layers | 32 | 80 |
| n_q_heads | 32 | 64 |
| n_kv_heads | 8 | 8 |
| d_head | 128 | 128 |
| d_ff | 14,336 | 28,672 |
| KV per token (BF16) | 128 KB | 327 KB |
| Weights (BF16) | ~16 GB | ~140 GB |

These numbers appear repeatedly throughout the book. When you see "327 KB/token" or "140 GB for 70B weights," this table is the source.

---

## Chapter Summary

The transformer block consists of two sub-layers — attention and FFN — each wrapped in a pre-norm + residual pattern. The **residual stream** is the central data structure: a `[B, n, d_model]` tensor that every layer reads from and writes a delta to. Layer normalization (RMSNorm in modern models) stabilises activations before each sub-layer.

The **FFN** holds roughly two-thirds of all model parameters. Modern LLMs use **SwiGLU** — a gated variant with two up-projections and a SiLU activation — at about 2.67× the model dimension. **MoE models** replace the dense FFN with sparse expert routing, slashing per-token FLOPs while keeping total parameter count high; all experts must remain in memory.

The **KV cache** stores only the K and V projections computed inside the attention sub-layer. Its size grows linearly with sequence length and is independent of the FFN. Understanding where each tensor lives in the block is the prerequisite for reading Chapters 4 through 8 with full comprehension.

---

*Next: Chapter 4 — Inside the Attention Mechanism*


---

## Self-Check Questions

1. A LLaMA-3 8B model has hidden size 4 096 and intermediate size 14 336. Compute the parameter count for one FFN layer (ignoring biases). How does this compare to the attention weight count at the same hidden size with 32 heads? *(Section 3.5.3)*

2. RMSNorm omits the mean-centering step of LayerNorm. Name one training-stability argument for keeping mean-centering and one efficiency argument for dropping it. *(Section 3.5.2)*

3. You double the batch size from 16 to 32 for a single decode step. By what factor does the KV cache memory grow? By what factor does the weight memory grow? *(Section 3.5.7)*

4. SwiGLU uses three weight matrices (W₁, W₂, W₃) rather than the two in a vanilla FFN. What is the purpose of the gating matrix W₂, and how does it interact with the SiLU activation? *(Section 3.5.3)*

5. In an MoE layer with 8 experts and top-2 routing, a token triggers experts 3 and 7. Describe the exact computation path, including what happens to the two expert outputs before the residual add. *(Section 3.5.5)*


---

## Worked Solutions

---

### Solution 1 — FFN parameter count vs attention parameter count (LLaMA-3 8B)

**What we need:** FFN params for one layer, compare to attention params.

**Given:** hidden_size (d_model) = 4,096; intermediate_size = 14,336 (for SwiGLU)

**Step 1 — FFN parameter count (SwiGLU uses three matrices).**

SwiGLU FFN has three weight matrices (no biases in LLaMA):
- W₁ (gate): `d_model × intermediate_size` = 4,096 × 14,336 = 58,720,256
- W₂ (down-projection): `intermediate_size × d_model` = 14,336 × 4,096 = 58,720,256
- W₃ (up-projection): `d_model × intermediate_size` = 4,096 × 14,336 = 58,720,256

$$\text{FFN params} = 3 \times 4{,}096 \times 14{,}336 = 3 \times 58{,}720{,}256 = \textbf{176,160,768} \approx 176 \text{M}$$

**Step 2 — Attention parameter count.**

For LLaMA-3 8B: 32 Q-heads, 8 KV-heads (GQA), head_dim=128:
- Q projection: d_model × (n_q_heads × head_dim) = 4,096 × (32 × 128) = 4,096 × 4,096 = 16,777,216
- K projection: d_model × (n_kv_heads × head_dim) = 4,096 × (8 × 128) = 4,096 × 1,024 = 4,194,304
- V projection: same as K = 4,194,304
- O projection: (n_q_heads × head_dim) × d_model = 4,096 × 4,096 = 16,777,216

$$\text{Attention params} = 16{,}777{,}216 + 4{,}194{,}304 + 4{,}194{,}304 + 16{,}777{,}216 = \textbf{41,943,040} \approx 42 \text{M}$$

**Step 3 — Comparison.**

$$\frac{\text{FFN params}}{\text{Attention params}} = \frac{176 \text{M}}{42 \text{M}} \approx 4.2\times$$

The FFN is approximately **4× larger** than the attention block in LLaMA-3 8B. This is typical for transformer models. FFN dominates the parameter count and therefore dominates both memory bandwidth consumption during decode and compute during prefill.

---

### Solution 2 — RMSNorm vs LayerNorm: stability vs efficiency argument

**What we need:** One argument for and one against keeping mean-centering.

**Background:** LayerNorm computes both mean and variance, then subtracts the mean before scaling. RMSNorm skips the mean subtraction entirely — it only computes the root-mean-square (RMS) and divides by it.

$$\text{LayerNorm: } \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i$$

$$\text{RMSNorm: } \hat{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i \quad \text{where RMS}(x) = \sqrt{\frac{1}{d}\sum_j x_j^2}$$

**Argument FOR keeping mean-centering (stability):**

Mean-centering ensures activations are zero-mean before scaling. If the activations have a large positive or negative bias, RMSNorm does not remove it — the bias remains and can push the distribution into regions where the nonlinearity saturates. For models trained from scratch on diverse data with fluctuating mean activations, LayerNorm's mean subtraction provides a regularizing effect that prevents activation drift.

**Argument FOR dropping mean-centering (efficiency):**

Computing the mean requires a first pass over all `d_model` elements of the vector, followed by a second pass to compute the variance. This is 2× the memory reads compared to RMSNorm, which only needs one pass to compute the sum-of-squares. For large `d_model` values (8,192 for larger models), this 2× pass reduction matters in practice. Additionally, pre-norm architectures (where normalization happens before each sublayer) empirically show that zero-mean initialization and careful learning rate scheduling can substitute for mean-centering, making it redundant at the cost of a measured training run.

---

### Solution 3 — KV cache vs weight memory scaling with batch size

**What we need:** How each scales when batch size doubles from 16 to 32.

**Step 1 — KV cache scaling.**

The KV cache stores the key and value tensors for every *sequence* being served. Each sequence has its own KV history:

$$\text{KV cache} \propto \text{batch\_size} \times \text{sequence\_length} \times \text{KV\_size\_per\_token}$$

Doubling batch size from 16 to 32 doubles the number of sequences → **KV cache doubles** (×2 growth). ✓

**Step 2 — Weight memory scaling.**

Model weights are **shared** across all sequences in the batch. No matter how many sequences you serve simultaneously, the weights occupy the same memory:

$$\text{Weight memory} = \text{constant (independent of batch size)}$$

Doubling batch size: **weight memory unchanged** (×1 growth). ✓

**Step 3 — Implications.**

| Component | Batch 16 → 32 |
|-----------|---------------|
| KV cache | 2× growth |
| Weights | No change |
| Activation buffers | 2× growth (per-token activations) |

This is why "how many concurrent users can I serve?" is fundamentally a KV cache capacity question, not a weight memory question. The weights are a fixed overhead; the KV cache is the variable cost per user.

---

### Solution 4 — SwiGLU gating: W₂ purpose and SiLU interaction

**What we need:** Mechanistic explanation of the gating in SwiGLU.

**Step 1 — Vanilla FFN computation (for comparison).**

$$\text{FFN}(x) = \text{ReLU}(x W_1) \cdot W_2$$

A single weight matrix W₁ projects up, ReLU removes negatives, W₂ projects back down.

**Step 2 — SwiGLU computation.**

$$\text{SwiGLU}(x) = \Big(x W_1 \otimes \text{SiLU}(x W_3)\Big) \cdot W_2$$

Where ⊗ is element-wise multiplication. There are now three matrices:
- **W₁ (up-projection):** projects x from d_model → intermediate_size, producing the "content" vector.
- **W₃ (gate-projection):** independently projects x → intermediate_size, producing the "gate" vector.
- **W₂ (down-projection):** projects the gated result back to d_model.

**Step 3 — What W₂ (gate/W₃) does.**

The gate vector is passed through SiLU (Sigmoid Linear Unit = x × σ(x)), producing values in approximately [0, 1]:

$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

Element-wise multiplying the content vector (from W₁) by the SiLU gate (from W₃) gives the model a learned mechanism to **suppress or amplify individual dimensions** of the intermediate representation. Dimensions where the gate is near 0 are effectively zeroed out; dimensions where the gate is near 1 pass through unchanged. This is a soft, differentiable version of feature selection — the model learns which intermediate features to "let through" as a function of the input x.

The interaction with SiLU is key: SiLU is smooth (differentiable everywhere, unlike ReLU) and approximately linear for large positive values. This prevents the "dead neuron" problem of ReLU while maintaining the gating behavior.

---

### Solution 5 — MoE token routing: expert computation path

**What we need:** Trace a token through 8-expert top-2 routing with experts 3 and 7.

**Step 1 — Router computation.**

The token's hidden state **x** (shape: `[d_model]`) is passed through the router:

$$\text{router\_logits} = x \cdot W_{\text{router}} \in \mathbb{R}^8$$

Softmax is applied to get routing probabilities:

$$p_i = \frac{e^{\text{router\_logits}_i}}{\sum_j e^{\text{router\_logits}_j}}$$

The top-2 probabilities are selected: say p₃ = 0.45 and p₇ = 0.35. These are *renormalized* to sum to 1: p₃_norm = 0.45/0.80 = 0.5625, p₇_norm = 0.35/0.80 = 0.4375.

**Step 2 — Expert 3 computation.**

Token **x** is fed through Expert 3's FFN (an independent MLP with its own weights):

$$h_3 = \text{Expert}_3(x) = W_{2,3} \cdot \text{SiLU}(W_{1,3} \cdot x \otimes W_{3,3} \cdot x)$$

This produces an output vector h₃ ∈ ℝ^d_model.

**Step 3 — Expert 7 computation.**

Same procedure with Expert 7's weights → h₇ ∈ ℝ^d_model. This computation is *independent* of Expert 3 (hence parallelizable).

**Step 4 — Weighted combination.**

The two expert outputs are combined using the renormalized routing weights:

$$h_\text{combined} = p_{3,\text{norm}} \cdot h_3 + p_{7,\text{norm}} \cdot h_7$$
$$= 0.5625 \cdot h_3 + 0.4375 \cdot h_7$$

**Step 5 — Residual addition.**

The combined expert output is added to the residual stream:

$$x_{\text{out}} = x + h_{\text{combined}}$$

The remaining 6 experts (0, 1, 2, 4, 5, 6) are **not involved** in this token's computation. This is the sparsity that makes MoE efficient: while the model has 8× the parameters of a dense FFN, each token only activates 2/8 = 25% of them. Total FLOPs = 2 expert FLOPs + routing overhead, not 8 expert FLOPs.

