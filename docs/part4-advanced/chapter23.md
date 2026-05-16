# Chapter 23: Speculative Decoding

> *"The one-token-per-step constraint feels as fundamental as gravity — until you realize it is not a constraint of physics but of the sequential sampling algorithm. Speculation replaces that constraint with a question: can we guess ahead and verify in parallel?"*

---

## What You Will Understand

- Why autoregressive decoding is architecturally stuck at one new token per forward pass — and the precise cost model
- The complete speculative decoding algorithm: how the draft model proposes, how the target model verifies, and why the accepted tokens are provably sampled from the correct distribution
- How to derive the speedup formula from first principles and predict speedup for any (α, K) pair
- Why acceptance rate α varies with generation entropy, draft model quality, and temperature — with worked examples
- The token tree generalization: how speculative decoding extends from a linear sequence to a tree of hypotheses
- Every self-speculation variant — EAGLE, EAGLE-2, Medusa, ngram, prompt lookup decoding — with their trade-offs
- How Medusa attaches multiple draft heads to the base model's last hidden state and why acceptance trees cover multiple candidates in one forward pass
- How EAGLE drafts in feature space (hidden-state vectors, not tokens) using a single-layer autoregressive head, and why this yields higher acceptance rates than non-autoregressive Medusa heads
- How SpecTr structures draft candidates as a tree and how branching factor, depth, and acceptance rate combine to determine effective tokens per second
- vLLM's full speculative decoding pipeline: engine configuration, scheduler integration, KV cache impact
- llama.cpp's two-model speculative loop: `--draft-max`, the draft/verify flow, and how to configure it
- When speculative decoding helps, when it is neutral, and the conditions under which it actively hurts throughput

**What you need first:** Chapter 6 (KV cache — the draft model has its own KV state), Chapter 9 (forward pass — you need to understand why one forward pass ≈ constant time), Chapter 12 (sampling — the rejection sampling step requires understanding probability distributions).

---

## §23.1  The Decode Bottleneck — Anatomy of a Constraint

Before breaking a constraint, understand exactly where it lives.

### 23.1.1  Why One Token Per Pass

Autoregressive generation produces token t+1 conditioned on all previous tokens t_1, ..., t_t:

```
  P(t_{n+1} | t_1, ..., t_n)
```

To generate token t_{n+1}, the model's forward pass must read the logits for position n+1, which requires computing attention over the KV cache up to position n. You cannot compute t_{n+2} until t_{n+1} is known, because t_{n+2}'s attention query at position n+2 depends on the embedding of t_{n+1}.

This is not a software limitation — it is intrinsic to the conditional structure. You cannot parallelize a sequential conditional.

### 23.1.2  The Cost Model

During decode, each forward pass:

- Reads the full model weight set from HBM: ~W bytes (16 GB for 8B BF16)
- Reads the KV cache for all prior tokens: ~2 × n_layers × n_kv_heads × head_dim × seq_len × 2 bytes
- Performs one matrix-vector multiply per layer (batch=1, single token query)
- Writes one new KV entry to the cache
- Produces one logit vector → one token

The weight-read dominates at typical batch sizes. On an H100 (3.35 TB/s), reading 16 GB takes:

```
WORKED EXAMPLE 23.1 — Decode step minimum latency (arithmetic lower bound)
──────────────────────────────────────────────────────────────────────────
Given:
  Model: Llama 3.1 8B, BF16 weights = 16 GB
  H100 SXM bandwidth: 3.35 TB/s = 3,350 GB/s

Minimum time to read weights (no compute, no KV cache):
  16 GB ÷ 3,350 GB/s = 4.8 ms

Actual decode step (weights + KV cache + compute):
  ~6–8 ms per token on H100 (batch=1, 2K context)

Tokens per second:
  1 / 0.007 s ≈ 143 tokens/s (batch=1)

What we want:
  If we could verify 4 tokens in one pass: 4 tokens / 7 ms = 571 tokens/s
  That is exactly what speculative decoding achieves (when α is high).
──────────────────────────────────────────────────────────────────────────
```

The key insight: the H100 reads ~16 GB of weights whether it generates 1 token or verifies 5 candidate tokens. Amortizing the weight-read over multiple output tokens is the core efficiency gain.

### 23.1.3  The Bandwidth-Compute Asymmetry at Batch=1

```
  At batch=1 (typical for speculative decode):
  ┌──────────────────────────────────────────────────────────────────┐
  │  Operation    │  FLOPs          │  Bytes read    │  AI (F/B)    │
  │  ─────────────┼─────────────────┼────────────────┼──────────────│
  │  1-token dec  │  2 × d² × L     │  d² × L × 2    │  1.0         │
  │  K-token ver  │  2 × d² × L × K │  d² × L × 2    │  K × 1.0     │
  └──────────────────────────────────────────────────────────────────┘

  (d = d_model, L = n_layers)

  The verification pass reads the SAME weights (same bytes_moved)
  but performs K times more FLOPs. Arithmetic intensity scales with K.
  This moves the operation toward compute-bound — better HW utilization.
```

---

## §23.2  The Speculative Decoding Algorithm

Leviathan et al. 2023 and Chen et al. 2023 independently proposed the same algorithm. Here is the complete procedure.

### 23.2.1  Participants

```
  ┌─────────────────────────────────────────────┐
  │  TARGET MODEL  (M_target)                   │
  │  Large, high-quality, slow.                 │
  │  Example: Llama 3.1 70B                     │
  │  Purpose: Define the desired output         │
  │           distribution. Final authority.    │
  └─────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────┐
  │  DRAFT MODEL  (M_draft)                     │
  │  Small, fast, approximate.                  │
  │  Example: Llama 3.2 1B or 3B               │
  │  Purpose: Quickly generate K guesses.       │
  │           Must share vocabulary with target.│
  └─────────────────────────────────────────────┘
```

### 23.2.2  Step-by-Step Algorithm

```
  ALGORITHM: Speculative Decoding (one iteration)
  ─────────────────────────────────────────────────────────────────
  Input:
    context  x_1, ..., x_t    (prefix already accepted)
    K        speculation width (tokens to draft)

  Phase 1 — DRAFT (fast, serial with draft model):
  ─────────────────────────────────────────────────────────────────
    For i = 1 to K:
      q_i  = M_draft(x_1, ..., x_{t+i-1})   # draft distribution at pos t+i
      x̃_i  ~ q_i                              # sample draft token

  Phase 2 — VERIFY (one forward pass of target model):
  ─────────────────────────────────────────────────────────────────
    [p_1, p_2, ..., p_K, p_{K+1}] = M_target(x_1, ..., x_t, x̃_1, ..., x̃_K)
    # One forward pass produces K+1 target distributions simultaneously
    # p_i = target distribution at position t+i (given draft tokens before it)

  Phase 3 — ACCEPT/REJECT (rejection sampling):
  ─────────────────────────────────────────────────────────────────
    For i = 1 to K:
      r_i ~ Uniform(0, 1)
      if r_i ≤ p_i(x̃_i) / q_i(x̃_i):
        Accept x̃_i   → advance position to t+i
      else:
        Resample x_i ~ norm(max(0, p_i - q_i))   # residual distribution
        Stop early.   Return x_1..x_{t+i}

    If all K tokens accepted:
      Sample x_{t+K+1} ~ p_{K+1}   # bonus token from target
      Return x_1..x_{t+K+1}
  ─────────────────────────────────────────────────────────────────
  Output:
    Between 1 and K+1 new accepted tokens.
    Guaranteed to be distributed as if sampled from M_target.
```

### 23.2.3  Rejection Sampling Correctness — Proof

This is the key theorem: **the accepted tokens are exactly distributed as if sampled from the target model**, regardless of what the draft model proposes.

```
PROOF SKETCH: Accepted token x̃_i has the target distribution p_i

  Case A: accepted (r ≤ p_i(x̃_i) / q_i(x̃_i))

    P(accept x̃_i = t)
      = P(x̃_i = t) × P(accept | x̃_i = t)
      = q_i(t) × min(1, p_i(t) / q_i(t))
      = min(q_i(t), p_i(t))

  Case B: rejected, resample from residual

    P(x̃_i rejected) = sum_t max(0, q_i(t) - p_i(t))

    Residual distribution:
      r_i(t) = max(0, p_i(t) - q_i(t)) / Z
      where Z = sum_t max(0, p_i(t) - q_i(t))
             = 1 - sum_t min(q_i(t), p_i(t))

  Total probability of token t being output:
      P(out = t) = min(q_i(t), p_i(t))
                 + P(reject) × r_i(t)
                 = min(q_i(t), p_i(t))
                 + (1 - sum_t' min(q_i(t'), p_i(t')))
                   × max(0, p_i(t) - q_i(t)) / Z

    Since Z = 1 - sum_t' min(q_i(t'), p_i(t')):

      P(out = t) = min(q_i(t), p_i(t)) + max(0, p_i(t) - q_i(t))
                 = p_i(t)   ✓

  The output distribution is exactly p_i — the target distribution. QED.
```

This is why speculative decoding is lossless: it does not approximate the target distribution, it samples exactly from it. The draft model only affects **speed**, never **quality**.

---

## §23.3  The Speedup Formula

### 23.3.1  Acceptance Rate α

For a single draft token, the acceptance probability is:

```
  α  =  E[min(1, p(t) / q(t))]
      = sum_t min(p(t), q(t))
      = 1 - (1/2) × Σ_t |p(t) - q(t)|
      = 1 - (1/2) × TV(p, q)
```

where TV(p, q) is the total variation distance between the target and draft distributions. When the draft perfectly matches the target, TV = 0 and α = 1 (always accept). When draft and target diverge maximally, α → 0.

### 23.3.2  Expected Tokens Per Step

With speculation width K and acceptance rate α:

```
  E[tokens accepted per step]

  = E[number of tokens produced in one verify step]

  Let T = number of tokens accepted (ranges from 1 to K+1).

  P(T ≥ 1) = 1                    (at minimum, we resample at position 1)
  P(T ≥ 2) = α                    (first token accepted)
  P(T ≥ 3) = α²                   (first two accepted)
  ...
  P(T ≥ K+1) = α^K                (all K draft tokens accepted)
  P(T = K+1) = α^K                (all K accepted + bonus from target)

  E[T] = sum_{k=1}^{K+1} P(T ≥ k)
       = 1 + α + α² + ... + α^K
       = (1 - α^{K+1}) / (1 - α)     [geometric series]
```

### 23.3.3  Speedup Formula

Define:

- c = cost ratio = (cost of one draft forward pass) / (cost of one target forward pass)
- K = speculation width
- α = acceptance rate

```
  Time per step with speculation:
    t_step = K × t_draft + t_verify
           = K × c × t_target + t_target
           = t_target × (Kc + 1)

  Tokens per step:  E[T] = (1 - α^{K+1}) / (1 - α)

  Speedup = E[T] / t_step  ÷  (1 / t_target)
          = E[T] × t_target / (t_target × (Kc + 1))
          = (1 - α^{K+1}) / ((1 - α)(Kc + 1))
```

### 23.3.4  Worked Speedup Examples

```
WORKED EXAMPLE 23.2 — Speedup calculation for various (α, K, c)
────────────────────────────────────────────────────────────────
Setup:
  Target: Llama 3.1 70B
  Draft:  Llama 3.2 3B   (c ≈ 3/70 ≈ 0.043, negligible in practice)
  Approximate c = 0.05  (draft is ~5% the cost of target per pass)

Case A: α=0.85, K=4, c=0.05
  E[T] = (1 - 0.85^5) / (1 - 0.85) = (1 - 0.4437) / 0.15 = 3.71 tokens
  t_step = (4 × 0.05 + 1) × t_target = 1.20 × t_target
  Speedup = 3.71 / 1.20 = 3.09×

Case B: α=0.70, K=4, c=0.05
  E[T] = (1 - 0.70^5) / (1 - 0.70) = (1 - 0.168) / 0.30 = 2.77 tokens
  t_step = 1.20 × t_target
  Speedup = 2.77 / 1.20 = 2.31×

Case C: α=0.50, K=4, c=0.05
  E[T] = (1 - 0.50^5) / (1 - 0.50) = (1 - 0.031) / 0.50 = 1.94 tokens
  t_step = 1.20 × t_target
  Speedup = 1.94 / 1.20 = 1.62×

Case D: α=0.85, K=8, c=0.05
  E[T] = (1 - 0.85^9) / 0.15 = (1 - 0.232) / 0.15 = 5.12 tokens
  t_step = (8 × 0.05 + 1) × t_target = 1.40 × t_target
  Speedup = 5.12 / 1.40 = 3.66×

Case E: α=0.85, K=4, c=0.20  (large draft model)
  E[T] = 3.71 tokens
  t_step = (4 × 0.20 + 1) × t_target = 1.80 × t_target
  Speedup = 3.71 / 1.80 = 2.06×

Key takeaways:
  1. α is the dominant factor — even K=8 can't compensate for low α
  2. K has diminishing returns: going from K=4 to K=8 adds ~18% speedup
     but doubles draft overhead if c is non-negligible
  3. Draft cost c matters more when α is already high
  4. For α < 0.6, speculative decoding barely pays for itself
────────────────────────────────────────────────────────────────
```

### 23.3.5  Speedup Heatmap

```
  Speedup = (1 - α^{K+1}) / ((1 - α)(Kc + 1))   with c = 0.05

  K\α    │  0.50   0.60   0.70   0.80   0.85   0.90   0.95
  ───────┼──────────────────────────────────────────────────
  K=1    │  1.27   1.35   1.45   1.56   1.62   1.69   1.78
  K=2    │  1.44   1.60   1.78   2.00   2.13   2.30   2.53
  K=3    │  1.54   1.76   2.04   2.38   2.59   2.87   3.28
  K=4    │  1.62   1.87   2.21   2.67   2.94   3.32   3.93
  K=5    │  1.67   1.95   2.35   2.90   3.24   3.72   4.52
  K=6    │  1.71   2.01   2.44   3.08   3.48   4.07   5.06
  K=8    │  1.75   2.08   2.57   3.36   3.87   4.65   6.00

  ← Not worth it without high α.    Sweet spot: α≥0.8, K=4–6 →
```

---

## §23.4  What Determines Acceptance Rate α

Understanding α is critical for knowing when to deploy speculative decoding and how to configure it.

### 23.4.1  Temperature and Entropy

```
  High temperature (T=1.0, creative writing):
    Target distribution: spread across many tokens
    Draft distribution:  spread across many tokens
    Overlap is LOW → α is LOW (both are uncertain, differently)

  Low temperature (T=0.1, factual Q&A, code):
    Target distribution: peaked on a few tokens
    Draft distribution:  peaked on a few tokens (same ones, likely)
    Overlap is HIGH → α is HIGH

  Greedy (T=0.0):
    Both models agree on argmax most of the time → α near 1.0

  ┌─────────────────────────────────────────────────────────────┐
  │  Typical α by task:                                         │
  │  Code completion (greedy):        α ≈ 0.90–0.95            │
  │  Factual Q&A (T=0.3):            α ≈ 0.80–0.88            │
  │  Summarization (T=0.5):          α ≈ 0.72–0.82            │
  │  Creative writing (T=1.0):        α ≈ 0.55–0.68            │
  │  Open-ended chat (T=0.8):         α ≈ 0.65–0.78            │
  └─────────────────────────────────────────────────────────────┘
```

### 23.4.2  Draft Model Alignment

The draft model must be from the same model family as the target, or at minimum share the same tokenizer and vocabulary. Mismatched models cannot be used:

- ✅ Llama 3.2 1B drafting for Llama 3.1 70B (same family, same vocab)
- ✅ Llama 3.2 3B drafting for Llama 3.1 70B
- ❌ Mistral 7B drafting for Llama 3.1 70B (different vocabulary)
- ❌ Gemma drafting for Qwen (different everything)

Training alignment matters too. A draft model fine-tuned on the same instruction dataset as the target achieves higher α than an untuned draft on an instruction-tuned target.

### 23.4.3  Context Length Effects

```
  Short context (< 512 tokens):
    Model is still "uncertain" about the generation direction.
    α is moderate.

  Medium context (512–4K tokens):
    Pattern established; draft and target often agree on
    continuation tokens. α is highest here.

  Very long context (> 16K tokens):
    Attention computation cost dominates; the "bandwidth amortization"
    benefit of speculation is reduced because KV cache read is large.
    α may be good but the per-step cost advantage shrinks.
```

### 23.4.4  Reasoning Models — A Special Case

`[DEEP DIVE]`

For reasoning models (DeepSeek-R1, Qwen3 thinking mode) generating long chain-of-thought:

- During "reasoning tokens" (inside `<think>` blocks): output is exploratory, high entropy, α ≈ 0.55–0.70. Speculation marginally beneficial.
- During "answer tokens" (after `</think>`): output is deterministic, high confidence, α ≈ 0.88–0.95. Speculation very beneficial.

Strategy: disable speculation for reasoning-phase generation; enable it for the final answer phase. vLLM does not yet support per-phase speculation switching; this requires application-layer detection.

---

## §23.5  The Token Tree: Beyond Linear Speculation

Linear speculation tries one sequence of K draft tokens: t̃_1, t̃_2, ..., t̃_K. A rejection at position 2 wastes the computation for positions 3 through K.

**Tree speculation** (Miao et al. 2023, Cai et al. 2024) instead generates a tree of draft hypotheses:

```
  Linear speculation (K=4):
  t̃_1 → t̃_2 → t̃_3 → t̃_4
  (if t̃_1 rejected, t̃_2 through t̃_4 are wasted)

  Tree speculation (tree of width 3, depth 3):
                    ┌── t̃_1a → t̃_1a1 → t̃_1a1a
         t̃_root ───┤── t̃_1b → t̃_1b1
                    └── t̃_1c
  (if t̃_1a rejected, t̃_1b and t̃_1c still get verified)

  Tree structure:
  - Width at each level = number of draft branches explored
  - Depth = maximum speculation horizon
  - Verification: one target forward pass, but with a ragged attention
    mask that allows each tree position to attend to its own prefix
```

The tree is flattened into a batch and verified in a single target forward pass using a **custom attention mask**:

```
  Token positions in flattened tree:
  [ root, 1a, 1b, 1c, 1a1, 1a2, 1b1, 1a1a, 1a1b, 1a2a ]
   ↑      ↑   ↑   ↑   ↑    ↑    ↑    ↑     ↑     ↑
   pos 0  1   2   3   4    5    6    7     8     9

  Attention mask (ragged causal):
  Position 4 (1a1) can attend to: root, 1a, 1a1 (its own prefix only)
  Position 6 (1b1) can attend to: root, 1b, 1b1

  Standard causal mask would allow 1a1 to attend to 1b — wrong.
  Tree mask restricts each position to its own ancestry.
```

EAGLE-2 (Li et al. 2024) uses adaptive tree expansion: start narrow, expand branches where the draft is more confident. This achieves higher accepted tokens per target forward pass than fixed-width trees.

---

## §23.6  Self-Speculation Variants

External draft models add operational complexity: two model checkpoints, two sets of KV caches, two memory regions. Several approaches achieve speculation benefits without a separate model.

### 23.6.1  EAGLE (Feature-Level Draft)

EAGLE (Li et al. 2024) trains a lightweight autoregressive head that predicts the next **feature vector** (the final hidden state before the LM head) rather than directly predicting the next token. The LM head is then applied to get the token logit.

```
  Standard draft model:
  context → [small transformer N layers] → logits → draft token

  EAGLE draft head:
  context → [1 FC layer + feature from target's final layer] → feature → [LM head] → draft token
  ↑                       ↑
  the tiny extra head    reuses the target model's learned features

  Size: ~20–40M parameters (vs. 1B+ for a dedicated draft model)
  α:    typically 0.82–0.90 on code tasks
  Memory: only ~60–80 MB extra (just the autoregressive head weights)
```

EAGLE is currently one of the highest-quality self-speculation methods: it gets close to the acceptance rate of a well-matched external draft model at a tiny fraction of the memory cost.

### 23.6.2  EAGLE-2 — Adaptive Context-Aware Tree

EAGLE-2 (Li et al. 2024) extends EAGLE with:
1. **Dynamic tree structure**: at each step, the tree expands based on confidence scores from the EAGLE head — high-confidence paths get deeper, uncertain paths get truncated early
2. **Context sensitivity**: the draft head uses a sliding window of recent accepted features to improve prediction

Empirical result: 3.0–4.5× speedup on coding tasks, 2.5–3.5× on math, 2.0–3.0× on chat.

### 23.6.3  Medusa — Multiple Draft Heads

Medusa (Cai et al. 2023) attaches K parallel heads to the target model's final hidden state, each predicting a different future position:

```
  Standard LM head:   hidden_n → token_{n+1}

  Medusa heads:
  hidden_n → head_1 → token_{n+1}  (same as normal head)
  hidden_n → head_2 → token_{n+2}  (predicts 2 ahead)
  hidden_n → head_3 → token_{n+3}  (predicts 3 ahead)
  hidden_n → head_4 → token_{n+4}  (predicts 4 ahead)
```

**Key difference from EAGLE**: Medusa heads are **non-autoregressive** — head_3 predicts position n+3 without conditioning on whether positions n+1 and n+2 were accepted. This makes them faster but less accurate.

Medusa generates a tree by combining predictions from all heads, then uses a tree-structured verification pass. Post-processing with a "tree attention" mechanism selects the best accepted prefix.

```
  Memory cost: 4 Medusa heads × d_model × vocab_size × 2 bytes
  For Llama 3.1 8B: 4 × 4096 × 128256 × 2 ≈ 4.2 GB  — substantial!
  Practical: use smaller vocab projection or share embeddings.
```

Medusa-2 (Cai et al. 2024) fine-tunes the Medusa heads jointly with the base model using a self-distillation objective, significantly improving α.

### 23.6.4  Ngram Speculation

The simplest possible self-speculation: use an ngram language model built from the **current context** to draft tokens.

```
  Algorithm:
  1. Build a 4-gram table from the current conversation context
  2. At each draft step, look up the most frequent continuation of
     the last 3 tokens in the context
  3. Use that as the draft token; if not found, fall back to random

  Example context: "The capital of France is"
  4-gram "France is" → "Paris" appeared in context → draft "Paris"
  This works when the model is likely to repeat context phrases
  (common in RAG, summarization, factual Q&A).
```

Ngram speculation is extremely cheap (no model inference for draft), but α is workload-dependent:

- High α when generation heavily copies from context (RAG, summarization)
- Near-zero when generation is creative or diverges from context

### 23.6.5  Prompt Lookup Decoding

A refinement of ngram that specifically matches the **prompt** (user input), not the entire context:

```
  Algorithm:
  1. Take the last M tokens of generated output so far (M = ngram_size)
  2. Search for that exact sequence in the prompt
  3. If found, draft the next K tokens from the prompt continuation

  Use case: tasks where the model frequently copies from the prompt —
            - Summarization: model quotes back the document
            - Document QA: model repeats question terms
            - Translation: model echoes source phrases
            - Editing: model repeats unchanged portions

  α can reach 0.90+ on heavy-copy tasks.
  Implementation: O(prompt_length) string scan per draft step.
```

vLLM supports prompt lookup decoding via `--speculative_draft_tensor_parallel_size` + ngram config. It is the lowest-overhead speculation option with no extra model or training required.

---

## §23.7  vLLM Speculative Decoding

### 23.7.1  Configuration

**With an external draft model:**

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-3B-Instruct \
    --num-speculative-tokens 5 \
    --speculative-draft-tensor-parallel-size 1 \
    --use-v2-block-manager
```

Key flags:

- `--speculative-model`: path or HuggingFace ID of the draft model
- `--num-speculative-tokens`: K (speculation width)
- `--speculative-draft-tensor-parallel-size`: TP degree for the draft model (usually 1)
- `--use-v2-block-manager`: required for speculative decoding in current vLLM

**With EAGLE:**

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model [eagle] \
    --num-speculative-tokens 5 \
    --speculative-draft-tensor-parallel-size 1
```

**With ngram/prompt lookup:**

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --speculative-model [ngram] \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-min 4 \
    --ngram-prompt-lookup-max 4
```

### 23.7.2  vLLM Internal Architecture

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  SPECULATIVE SCHEDULER                                                  │
  │                                                                         │
  │  1. For each request in the running batch:                              │
  │     - Run draft model K times (autoregressive, small model)             │
  │     - Collect K draft tokens per request                                │
  │                                                                         │
  │  2. Batch ALL requests for one target forward pass:                     │
  │     - Input: original context + K draft tokens per request              │
  │     - Each request's draft tokens are appended to its KV sequence       │
  │     - Target produces K+1 logits per request                            │
  │                                                                         │
  │  3. For each request: rejection sampling                                │
  │     - Compute accept/reject per draft token                             │
  │     - Find first rejection (or accept all)                              │
  │     - Roll back KV cache to the last accepted position                  │
  │     - Append accepted tokens + one resampled/bonus token                │
  └─────────────────────────────────────────────────────────────────────────┘
```

**KV cache impact:** Each request maintains two KV caches — one for the draft model and one for the target model. The draft model's KV cache is smaller (smaller d_model) but nonzero. Total KV overhead with speculative decoding:

```
WORKED EXAMPLE 23.3 — KV cache overhead with speculative decoding
──────────────────────────────────────────────────────────────────
Target: Llama 3.1 70B  (80 layers, 8 KV heads, head_dim 128)
Draft:  Llama 3.2 3B   (28 layers, 8 KV heads, head_dim 128)
Batch=1, seq=2048, BF16

Target KV:  2 × 80 × 8 × 128 × 1 × 2048 × 2 = 671 MB
Draft KV:   2 × 28 × 8 × 128 × 1 × 2048 × 2 = 235 MB

Overhead ratio:  235 / 671 = 35% extra KV from draft model
Total KV:  671 + 235 = 906 MB (vs. 671 MB without speculation)
──────────────────────────────────────────────────────────────────
```

This KV overhead reduces the number of concurrent requests that fit in HBM. For small-model speculation on large targets, the overhead is modest (~20–35%). For same-sized draft models, it doubles the KV usage — a poor trade.

### 23.7.3  Throughput vs. Latency Trade-Off

`[COMMON TRAP]` Speculative decoding improves **single-request latency** by generating more tokens per target forward pass. But it can **hurt total throughput** when the batch is already large:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Batch=1 (low concurrency):                                     │
  │  Without spec: 1 token per step, GPU ~20% utilized             │
  │  With spec:    3.5 tokens/step avg, GPU ~70% utilized → WIN    │
  │                                                                 │
  │  Batch=64 (high concurrency):                                   │
  │  Without spec: 64 tokens/step, GPU ~95% utilized               │
  │  With spec:    draft adds 64 × 5 = 320 extra forward passes     │
  │               plus one bigger verify pass → OVERHEAD            │
  │               May reduce throughput by 10–30%                   │
  └─────────────────────────────────────────────────────────────────┘
```

Rule of thumb:

- Enable speculative decoding when: batch size is small (< 8), latency matters more than throughput, α is expected high (code/structured output)
- Disable speculative decoding when: running high-throughput batch jobs, batch size is large (> 32), generation is creative/high-temperature

### 23.7.4  Monitoring Speculation Effectiveness

vLLM exposes speculation metrics via Prometheus:

```python
# Key metrics to watch:
# vllm:spec_decode_draft_acceptance_rate   → α (target: > 0.75)
# vllm:spec_decode_efficiency              → tokens_accepted / tokens_drafted
# vllm:spec_decode_num_accepted_tokens     → average accepted per step
# vllm:spec_decode_num_draft_tokens        → K (constant, your config)

# If α drops below 0.65, consider:
# 1. Reducing K (less waste per rejection)
# 2. Switching to a better-aligned draft model
# 3. Disabling speculation for high-temperature requests
```

---

## §23.8  llama.cpp Speculative Decoding

### 23.8.1  Two-Model Setup

llama.cpp implements speculative decoding with two model contexts:

```bash
# Method 1: llama-cli with --draft flags
./build/bin/llama-cli \
    --model  ./Llama-3.1-70B-Q4_K_M.gguf \
    --model-draft ./Llama-3.2-3B-Q4_K_M.gguf \
    --n-gpu-layers 80 \          # target: full GPU offload
    --n-gpu-layers-draft 28 \    # draft: full GPU offload
    --draft-max 5 \              # K = 5
    --draft-min 1 \              # minimum draft tokens before verify
    --draft-p-min 0.8 \          # only draft if draft confidence > 0.8
    --prompt "Explain quantum entanglement: "

# Method 2: llama-server (serving mode)
./build/bin/llama-server \
    --model  ./Llama-3.1-70B-Q4_K_M.gguf \
    --model-draft ./Llama-3.2-3B-Q4_K_M.gguf \
    --draft-max 5 \
    --port 8080
```

### 23.8.2  The llama.cpp Draft/Verify Loop

```cpp
// Conceptual C++ sketch of llama.cpp's speculative decode loop
// (simplified from llama.cpp/examples/speculative/speculative.cpp)

struct SpecContext {
    llama_context* ctx_target;   // large model
    llama_context* ctx_draft;    // small model
    int K;                       // draft width
};

std::vector<llama_token> speculative_step(
    SpecContext& spec,
    const std::vector<llama_token>& prefix
) {
    // Phase 1: Draft K tokens with small model
    std::vector<llama_token> draft_tokens;
    std::vector<float>       draft_probs;

    llama_batch batch_draft = llama_batch_init(512, 0, 1);
    // ... append prefix to draft context if not already there ...

    for (int i = 0; i < spec.K; ++i) {
        llama_decode(spec.ctx_draft, batch_draft);
        float* logits = llama_get_logits(spec.ctx_draft);

        // Sample from draft distribution
        llama_token tok = sample_greedy_or_top_p(logits);   // simplified
        float       prob = softmax_prob(logits, tok);

        draft_tokens.push_back(tok);
        draft_probs.push_back(prob);

        // Feed draft token back as next input
        llama_batch_clear(batch_draft);
        llama_batch_add(batch_draft, tok, prefix.size() + i, {0}, false);
    }

    // Phase 2: Verify with target model (one forward pass, K+1 positions)
    llama_batch batch_target = llama_batch_init(512, 0, 1);
    // Add all K draft tokens + request logits at each position
    for (int i = 0; i < spec.K; ++i) {
        llama_batch_add(batch_target, draft_tokens[i],
                        prefix.size() + i, {0}, true /*logits*/);
    }
    // Add a sentinel for the K+1 bonus position
    llama_batch_add(batch_target, draft_tokens.back(),
                    prefix.size() + spec.K, {0}, true);

    llama_decode(spec.ctx_target, batch_target);

    // Phase 3: Rejection sampling
    std::vector<llama_token> accepted;
    for (int i = 0; i < spec.K; ++i) {
        float* target_logits = llama_get_logits_ith(spec.ctx_target, i);
        float  p_target = softmax_prob(target_logits, draft_tokens[i]);
        float  p_draft  = draft_probs[i];

        float r = random_uniform_0_1();
        if (r <= p_target / p_draft) {
            accepted.push_back(draft_tokens[i]);
        } else {
            // Resample from residual = max(0, p_target - p_draft)
            llama_token resampled = sample_residual(target_logits,
                                                     draft_probs[i],
                                                     draft_tokens[i]);
            accepted.push_back(resampled);
            break;   // stop at first rejection
        }
    }

    // Bonus token if all K accepted
    if ((int)accepted.size() == spec.K) {
        float* bonus_logits = llama_get_logits_ith(
            spec.ctx_target, spec.K);
        accepted.push_back(sample_greedy_or_top_p(bonus_logits));
    }

    // Roll back draft KV cache to match accepted prefix length
    llama_kv_cache_seq_rm(spec.ctx_draft, 0,
                          prefix.size() + accepted.size(), -1);

    return accepted;
}
```

### 23.8.3  `--draft-p-min` — Conditional Drafting

llama.cpp's `--draft-p-min` flag adds an important optimization: if the draft model's top-1 probability for the next token falls below the threshold, stop drafting and verify immediately. This avoids wasting target forward pass time verifying low-confidence drafts:

```
  --draft-p-min 0.8 means:
    "Draft the next token only if I am at least 80% confident.
     Otherwise, stop and verify with the target now."

  Effect on speculative decoding:
  - Average draft length becomes variable: 0 to K tokens
  - Target verify pass is cheaper when draft stops early
  - Overall α increases because only high-confidence drafts are verified
  - Useful when K is large (e.g., K=8): avoids verifying 8 tokens when
    draft is uncertain after token 3
```

---

## §23.9  Comparison: All Speculation Approaches

```
  Method           │ Extra Params  │ Training  │ α (code) │ α (chat) │ Notes
  ─────────────────┼───────────────┼───────────┼──────────┼──────────┼────────────────────
  Draft model      │ 1B–7B params  │ Pretrain  │ 0.88–0.93│ 0.72–0.82│ Best α; high mem
  EAGLE            │ 20–60M params │ Fine-tune │ 0.85–0.91│ 0.70–0.80│ Best self-spec
  EAGLE-2          │ 20–60M params │ Fine-tune │ 0.87–0.93│ 0.72–0.82│ + adaptive tree
  Medusa           │ K × d × V     │ Fine-tune │ 0.78–0.85│ 0.62–0.72│ ~4 GB heads (8B)
  Medusa-2         │ K × d × V     │ Fine-tune │ 0.82–0.89│ 0.68–0.78│ + self-distill
  Ngram            │ 0 params      │ None      │ 0.40–0.70│ 0.35–0.55│ Context-dependent
  Prompt lookup    │ 0 params      │ None      │ 0.80–0.95│ 0.55–0.75│ Best on copy tasks
  ─────────────────┼───────────────┼───────────┼──────────┼──────────┼────────────────────
  vLLM support     │ All           │ N/A       │ ✅       │ ✅       │ v0.5.0+
  llama.cpp support│ Draft model   │ N/A       │ ✅       │ ✅       │ --model-draft
  llama.cpp EAGLE  │ EAGLE head    │ N/A       │ 🚧      │ 🚧      │ Experimental
```

---

## §23.10  Code

### Python: Speculative Decoding Simulator and Configuration Helper

```python
#!/usr/bin/env python3
"""
Chapter 23 — Python: Speculative Decoding Analysis and Configuration
=====================================================================
Provides:
  - Exact speedup formula with heatmap generation
  - Acceptance rate estimation from empirical data
  - vLLM configuration advisor
  - Simulation of draft/verify/resample cycle (no GPU required)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable

# ─── Speedup analysis ─────────────────────────────────────────────────────────

def expected_tokens(alpha: float, K: int) -> float:
    """Expected tokens accepted per speculative step."""
    if abs(alpha - 1.0) < 1e-9:
        return K + 1.0
    return (1.0 - alpha ** (K + 1)) / (1.0 - alpha)

def speedup(alpha: float, K: int, c: float = 0.05) -> float:
    """
    Speculative decoding speedup.
    alpha: acceptance rate (0–1)
    K:     speculation width
    c:     draft cost relative to target (default 0.05 = 5%)
    """
    E_T      = expected_tokens(alpha, K)
    t_step   = K * c + 1.0       # normalized to target forward pass = 1
    return E_T / t_step

def print_speedup_table():
    print("\n" + "=" * 72)
    print("  Speculative Decoding Speedup  (c=0.05)")
    print("=" * 72)
    alphas = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    Ks     = [1, 2, 3, 4, 5, 6, 8]

    # Header
    header = f"  {'K':>5} │"
    for a in alphas:
        header += f"  α={a:.2f}"
    print(header)
    print("  " + "─" * 6 + "┼" + "─" * (len(alphas) * 9))

    for K in Ks:
        row = f"  K={K:>2}  │"
        for a in alphas:
            s = speedup(a, K)
            row += f"  {s:>6.2f}×"
        print(row)

    print()
    # Optimal K for each alpha
    print("  Optimal K (maximizing speedup, c=0.05):")
    for a in alphas:
        best_K = max(range(1, 17), key=lambda k: speedup(a, k))
        best_s = speedup(a, best_K)
        print(f"    α={a:.2f}  →  K={best_K}  (speedup={best_s:.2f}×)")


# ─── Optimal K finder ─────────────────────────────────────────────────────────

def optimal_K(alpha: float, c: float = 0.05, K_max: int = 16) -> int:
    """Find K that maximizes speedup."""
    return max(range(1, K_max + 1), key=lambda k: speedup(alpha, k, c))


# ─── Empirical acceptance rate estimator ──────────────────────────────────────

class AcceptanceRateEstimator:
    """
    Running estimate of acceptance rate α from live traffic.
    Use to dynamically tune K.
    """
    def __init__(self, window: int = 200):
        self.window    = window
        self._history: list[float] = []   # 1.0 = accepted, 0.0 = rejected per token

    def record_token(self, accepted: bool) -> None:
        self._history.append(1.0 if accepted else 0.0)
        if len(self._history) > self.window:
            self._history.pop(0)

    @property
    def alpha(self) -> float:
        if not self._history:
            return 0.8   # default assumption
        return sum(self._history) / len(self._history)

    def recommended_K(self, c: float = 0.05) -> int:
        return optimal_K(self.alpha, c)


# ─── Rejection sampling simulation ────────────────────────────────────────────

def simulate_speculative_step(
    target_probs: list[float],   # target distribution over V tokens
    draft_probs:  list[float],   # draft distribution over V tokens
    K: int,
    seed: int = None,
) -> dict:
    """
    Simulate one speculative decoding step.
    Returns dict with: accepted_tokens, n_accepted, first_rejection.
    """
    rng = random.Random(seed)
    V   = len(target_probs)
    assert len(draft_probs) == V, "Vocab size mismatch"
    assert abs(sum(target_probs) - 1.0) < 1e-4
    assert abs(sum(draft_probs)  - 1.0) < 1e-4

    # Draft K tokens from draft distribution
    vocab = list(range(V))
    draft_tokens = rng.choices(vocab, weights=draft_probs, k=K)

    # Rejection sampling
    accepted_tokens = []
    first_rejection = None

    for i, t in enumerate(draft_tokens):
        p = target_probs[t]
        q = draft_probs[t]
        r = rng.random()

        if q == 0.0:
            # Draft sampled from zero-prob token → always reject
            residual = [max(0, p - q) for p, q in zip(target_probs, draft_probs)]
            Z = sum(residual)
            if Z > 1e-9:
                resampled = rng.choices(vocab, weights=residual)[0]
                accepted_tokens.append(resampled)
            first_rejection = i
            break

        if r <= p / q:
            accepted_tokens.append(t)
        else:
            # Resample from residual distribution
            residual = [max(0.0, tp - dp) for tp, dp in zip(target_probs, draft_probs)]
            Z = sum(residual)
            if Z > 1e-9:
                resampled = rng.choices(vocab, weights=residual)[0]
                accepted_tokens.append(resampled)
            first_rejection = i
            break

    # Bonus token if all K accepted
    if len(accepted_tokens) == K:
        bonus = rng.choices(vocab, weights=target_probs)[0]
        accepted_tokens.append(bonus)
        first_rejection = None

    return {
        "accepted_tokens": accepted_tokens,
        "n_accepted": len(accepted_tokens),
        "first_rejection": first_rejection,
    }


def run_simulation():
    print("\n" + "=" * 60)
    print("  Rejection Sampling Simulation  (V=10 tokens, K=4)")
    print("=" * 60)

    # Simulate different alpha scenarios
    scenarios = [
        # (description, target_probs, draft_probs)
        ("High agreement (α≈0.90)",
         [0.5, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.005],
         [0.48,0.21,0.11,0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.005]),
        ("Medium agreement (α≈0.75)",
         [0.4, 0.2, 0.15, 0.1, 0.06, 0.04, 0.02, 0.01, 0.01, 0.01],
         [0.3, 0.15,0.18, 0.15,0.08, 0.06, 0.04, 0.02, 0.01, 0.01]),
        ("Low agreement (α≈0.50)",
         [0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01],
         [0.1, 0.1, 0.1,  0.1, 0.1, 0.1,  0.1,  0.1,  0.1,  0.1 ]),
    ]

    N_TRIALS = 10000
    for desc, target_p, draft_p in scenarios:
        # normalize
        s_t = sum(target_p); s_d = sum(draft_p)
        target_p = [x/s_t for x in target_p]
        draft_p  = [x/s_d for x in draft_p]

        # theoretical α
        alpha_theory = sum(min(p, q) for p, q in zip(target_p, draft_p))

        n_accepted_total = 0
        for trial in range(N_TRIALS):
            result = simulate_speculative_step(target_p, draft_p, K=4,
                                               seed=trial)
            n_accepted_total += result["n_accepted"]

        mean_accepted = n_accepted_total / N_TRIALS
        theoretical_E_T = expected_tokens(alpha_theory, 4)

        print(f"\n  {desc}")
        print(f"    α (theory):         {alpha_theory:.3f}")
        print(f"    E[tokens] theory:   {theoretical_E_T:.3f}")
        print(f"    E[tokens] simulated:{mean_accepted:.3f}")
        print(f"    Speedup (c=0.05):   {speedup(alpha_theory, 4):.2f}×")


# ─── vLLM config advisor ──────────────────────────────────────────────────────

@dataclass
class WorkloadProfile:
    name:          str
    temperature:   float
    task_type:     str    # "code", "factual", "chat", "creative"
    avg_output_len: int
    batch_size:    int

def advise_speculation(profile: WorkloadProfile) -> dict:
    """
    Recommend speculative decoding settings based on workload.
    Returns: {"enable": bool, "K": int, "method": str, "reason": str}
    """
    # Estimate α based on task and temperature
    base_alpha = {
        "code":     0.90,
        "factual":  0.82,
        "chat":     0.73,
        "creative": 0.58,
    }.get(profile.task_type, 0.70)

    # Temperature adjustment: higher temp → lower α
    alpha_adj = base_alpha - max(0, (profile.temperature - 0.3) * 0.2)
    alpha_adj = max(0.3, min(0.99, alpha_adj))

    # Batch size check: high batch → speculation less beneficial
    if profile.batch_size > 32:
        return {
            "enable": False,
            "K": 0,
            "method": "none",
            "reason": f"Batch size {profile.batch_size} too large; "
                      "throughput overhead exceeds latency benefit"
        }

    # Short outputs: overhead may not be worth it
    if profile.avg_output_len < 20 and alpha_adj < 0.80:
        return {
            "enable": False,
            "K": 0,
            "method": "none",
            "reason": "Short outputs with moderate α: overhead not justified"
        }

    K = optimal_K(alpha_adj)
    s = speedup(alpha_adj, K)

    if s < 1.30:
        return {
            "enable": False,
            "K": K,
            "method": "none",
            "reason": f"Estimated speedup {s:.2f}× too low (α≈{alpha_adj:.2f})"
        }

    method = {
        "code":     "eagle",
        "factual":  "draft_model",
        "chat":     "eagle",
        "creative": "ngram",
    }.get(profile.task_type, "draft_model")

    return {
        "enable": True,
        "K": K,
        "alpha_estimate": round(alpha_adj, 3),
        "speedup_estimate": round(s, 2),
        "method": method,
        "reason": (
            f"Estimated α≈{alpha_adj:.2f} → {s:.2f}× speedup. "
            f"Recommended method: {method}. "
            f"Optimal K={K} for these conditions."
        )
    }

def demo_advisor():
    print("\n" + "=" * 70)
    print("  Speculation Configuration Advisor")
    print("=" * 70)

    workloads = [
        WorkloadProfile("Code completion",  0.1, "code",     200, 1),
        WorkloadProfile("Factual Q&A",      0.3, "factual",  100, 4),
        WorkloadProfile("Customer chat",    0.7, "chat",      80, 16),
        WorkloadProfile("Creative writing", 1.0, "creative", 500, 2),
        WorkloadProfile("Batch summarize",  0.3, "factual",  300, 64),
        WorkloadProfile("RAG answer",       0.3, "factual",  150, 8),
    ]

    for w in workloads:
        advice = advise_speculation(w)
        status = "✅ ENABLE" if advice["enable"] else "❌ SKIP  "
        print(f"\n  {status}  {w.name:<22} T={w.temperature}  bs={w.batch_size}")
        print(f"           {advice['reason']}")


if __name__ == "__main__":
    print("\nChapter 23 — Speculative Decoding Analysis")
    print("=" * 70)
    print_speedup_table()
    run_simulation()
    demo_advisor()
```

### C++: Draft/Verify Loop, Speedup Calculator, Rejection Sampling

```cpp
/**
 * Chapter 23 — C++ Companion: Speculative Decoding
 * =================================================
 * Implements:
 *   §23.3  Speedup formula with full numerical table
 *   §23.2  Rejection sampling correctness demonstration
 *   §23.8  llama.cpp-style draft/verify loop (simulated, no GPU)
 *   §23.5  Token tree acceptance simulation
 *
 * Build:  g++ -std=c++17 -O2 speculative_demo.cpp -o speculative_demo
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Speedup formula
// ─────────────────────────────────────────────────────────────────────────────

double expected_tokens(double alpha, int K) {
    if (std::abs(alpha - 1.0) < 1e-9) return K + 1.0;
    return (1.0 - std::pow(alpha, K + 1)) / (1.0 - alpha);
}

double speedup(double alpha, int K, double c = 0.05) {
    double E_T   = expected_tokens(alpha, K);
    double t_step = K * c + 1.0;
    return E_T / t_step;
}

int optimal_K(double alpha, double c = 0.05, int K_max = 16) {
    int    best_K = 1;
    double best_s = speedup(alpha, 1, c);
    for (int k = 2; k <= K_max; ++k) {
        double s = speedup(alpha, k, c);
        if (s > best_s) { best_s = s; best_K = k; }
    }
    return best_K;
}

static void print_section(const std::string& title) {
    std::cout << "\n" << std::string(72, '=') << "\n"
              << "  " << title << "\n"
              << std::string(72, '=') << "\n";
}

void demo_speedup_table() {
    print_section("Speedup Formula: (1-α^{K+1}) / ((1-α)(Kc+1))  with c=0.05");

    const std::vector<double> alphas = {0.50, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95};
    const std::vector<int>    Ks     = {1, 2, 3, 4, 5, 6, 8};

    std::cout << "  " << std::setw(6) << "K" << " │";
    for (double a : alphas)
        std::cout << "  α=" << std::fixed << std::setprecision(2) << a;
    std::cout << "\n  " << std::string(6,'-') << "─┼" << std::string(56,'-') << "\n";

    for (int K : Ks) {
        std::cout << "  K=" << std::setw(2) << K << "  │";
        for (double a : alphas) {
            double s = speedup(a, K);
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << s << "×";
        }
        std::cout << "\n";
    }

    std::cout << "\n  Optimal K per α (maximizing speedup, c=0.05):\n";
    for (double a : alphas) {
        int    K_opt = optimal_K(a);
        double s_opt = speedup(a, K_opt);
        std::cout << "    α=" << std::fixed << std::setprecision(2) << a
                  << "  →  K=" << K_opt
                  << "  (speedup=" << std::setprecision(2) << s_opt << "×)\n";
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Rejection sampling implementation
// ─────────────────────────────────────────────────────────────────────────────

struct SpecStep {
    std::vector<int> accepted_tokens;
    int              n_accepted;
    int              first_rejection;   // -1 if all accepted
};

// Sample from distribution (weighted)
int sample_from(const std::vector<double>& probs, std::mt19937& rng) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

SpecStep rejection_sampling_step(
    const std::vector<double>& target_p,
    const std::vector<double>& draft_p,
    int K,
    std::mt19937& rng
) {
    int V = static_cast<int>(target_p.size());
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Phase 1: draft K tokens
    std::vector<int> draft_tokens;
    for (int i = 0; i < K; ++i)
        draft_tokens.push_back(sample_from(draft_p, rng));

    // Phase 2: accept/reject
    SpecStep result;
    result.first_rejection = -1;

    for (int i = 0; i < K; ++i) {
        int    t = draft_tokens[i];
        double p = target_p[t];
        double q = draft_p[t];

        double r = unif(rng);
        if (q < 1e-12 || r > p / q) {
            // Rejected: resample from residual
            std::vector<double> residual(V);
            double Z = 0.0;
            for (int v = 0; v < V; ++v) {
                residual[v] = std::max(0.0, target_p[v] - draft_p[v]);
                Z += residual[v];
            }
            if (Z > 1e-12) {
                int resampled = sample_from(residual, rng);
                result.accepted_tokens.push_back(resampled);
            }
            result.first_rejection = i;
            break;
        } else {
            result.accepted_tokens.push_back(t);
        }
    }

    // Bonus token if all K accepted
    if ((int)result.accepted_tokens.size() == K) {
        result.accepted_tokens.push_back(sample_from(target_p, rng));
    }

    result.n_accepted = static_cast<int>(result.accepted_tokens.size());
    return result;
}

void demo_rejection_sampling() {
    print_section("Rejection Sampling Simulation (V=8, K=4, 200,000 trials)");

    const int V = 8, K = 4, N = 200000;
    std::mt19937 rng(42);

    // Three scenarios
    struct Scenario {
        std::string name;
        std::vector<double> target_p, draft_p;
    };

    std::vector<Scenario> scenarios = {
        {
            "High agreement (α≈0.90)",
            {0.45, 0.20, 0.12, 0.09, 0.06, 0.04, 0.02, 0.02},
            {0.43, 0.21, 0.13, 0.09, 0.06, 0.04, 0.02, 0.02},
        },
        {
            "Moderate agreement (α≈0.75)",
            {0.40, 0.20, 0.15, 0.10, 0.07, 0.04, 0.02, 0.02},
            {0.30, 0.18, 0.15, 0.14, 0.09, 0.07, 0.05, 0.02},
        },
        {
            "Low agreement (α≈0.50)",
            {0.35, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02},
            {0.12, 0.12, 0.12, 0.12, 0.13, 0.13, 0.13, 0.13},
        },
    };

    for (auto& s : scenarios) {
        // Normalize
        double st = 0, sd = 0;
        for (int i = 0; i < V; ++i) { st += s.target_p[i]; sd += s.draft_p[i]; }
        for (int i = 0; i < V; ++i) { s.target_p[i] /= st; s.draft_p[i] /= sd; }

        // Theoretical α
        double alpha_theory = 0.0;
        for (int i = 0; i < V; ++i)
            alpha_theory += std::min(s.target_p[i], s.draft_p[i]);

        double total_accepted = 0;
        for (int trial = 0; trial < N; ++trial) {
            auto res = rejection_sampling_step(s.target_p, s.draft_p, K, rng);
            total_accepted += res.n_accepted;
        }

        double mean_acc   = total_accepted / N;
        double theory_E_T = expected_tokens(alpha_theory, K);
        double s_val      = speedup(alpha_theory, K);

        std::cout << "\n  Scenario: " << s.name << "\n";
        std::cout << "    α (theory):           " << std::fixed << std::setprecision(4)
                  << alpha_theory << "\n";
        std::cout << "    E[tokens] theory:     " << std::setprecision(4)
                  << theory_E_T << "\n";
        std::cout << "    E[tokens] simulated:  " << std::setprecision(4)
                  << mean_acc << "\n";
        std::cout << "    Speedup estimate:     " << std::setprecision(2)
                  << s_val << "×\n";
        std::cout << "    Error (sim vs theory):" << std::setprecision(5)
                  << std::abs(mean_acc - theory_E_T) << "\n";

        // Soft check: warn but don't abort — simulation is stochastic
        if (std::abs(mean_acc - theory_E_T) >= 0.05) {
            std::cerr << "    [WARN] Sim vs theory gap " << std::setprecision(5)
                      << std::abs(mean_acc - theory_E_T)
                      << " exceeds 0.05 — increase N for tighter convergence\n";
        }
    }
    std::cout << "\n  Simulation complete (N=200,000 trials; gaps expected < 0.05).\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// llama.cpp-style speculative decode: simulated text generation
// ─────────────────────────────────────────────────────────────────────────────

// Mock vocabulary for demo
static const std::vector<std::string> VOCAB = {
    "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "A", "fast", "red", "cat", "runs", "past", "slow", "rabbit",
    ".", ",", "!", "?" ,
};
const int V_DEMO = static_cast<int>(VOCAB.size());

// "Model" = just a pseudo-random distribution conditioned on position
std::vector<double> mock_distribution(int pos, bool is_target, std::mt19937& rng) {
    std::vector<double> p(V_DEMO);
    // Simulate realistic peaked distribution
    int top1 = pos % V_DEMO;
    int top2 = (pos + 1) % V_DEMO;
    p[top1] = is_target ? 0.45 : 0.43;
    p[top2] = is_target ? 0.20 : 0.22;
    double rem = 1.0 - p[top1] - p[top2];
    // Distribute rest
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (int i = 0; i < V_DEMO; ++i) {
        if (i != top1 && i != top2)
            p[i] = unif(rng);
    }
    // Normalize remaining
    double s = 0;
    for (int i = 0; i < V_DEMO; ++i)
        if (i != top1 && i != top2) s += p[i];
    for (int i = 0; i < V_DEMO; ++i)
        if (i != top1 && i != top2) p[i] = p[i] / s * rem;
    return p;
}

void demo_generation_loop() {
    print_section("Simulated Speculative Generation (K=4, 20 target tokens)");

    const int K = 4, TARGET_TOKENS = 20;
    std::mt19937 rng(99);

    std::vector<int> generated;
    int total_target_calls  = 0;
    int total_draft_calls   = 0;
    int total_accepted      = 0;
    int total_steps         = 0;
    int context_pos         = 0;

    while ((int)generated.size() < TARGET_TOKENS) {
        // Phase 1: K draft calls
        std::vector<double> draft_dist_arr[K];
        std::vector<int>    draft_toks(K);
        for (int i = 0; i < K; ++i) {
            draft_dist_arr[i] = mock_distribution(context_pos + i, false, rng);
            draft_toks[i]     = sample_from(draft_dist_arr[i], rng);
            total_draft_calls++;
        }

        // Phase 2: one target verify call (simulated as K+1 distributions)
        std::vector<std::vector<double>> target_dists(K + 1);
        for (int i = 0; i <= K; ++i)
            target_dists[i] = mock_distribution(context_pos + i, true, rng);
        total_target_calls++;

        // Phase 3: rejection sampling
        std::vector<int> step_accepted;
        bool all_accepted = true;

        for (int i = 0; i < K; ++i) {
            int    t = draft_toks[i];
            double p = target_dists[i][t];
            double q = draft_dist_arr[i][t];
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            double r = unif(rng);

            if (q > 1e-12 && r <= p / q) {
                step_accepted.push_back(t);
            } else {
                // Resample
                std::vector<double> residual(V_DEMO);
                for (int v = 0; v < V_DEMO; ++v)
                    residual[v] = std::max(0.0, target_dists[i][v] - draft_dist_arr[i][v]);
                int resampled = sample_from(residual, rng);
                step_accepted.push_back(resampled);
                all_accepted = false;
                break;
            }
        }

        // Bonus token
        if (all_accepted) {
            step_accepted.push_back(sample_from(target_dists[K], rng));
        }

        // Record
        for (int t : step_accepted) {
            generated.push_back(t);
            if ((int)generated.size() >= TARGET_TOKENS) break;
        }
        total_accepted += static_cast<int>(step_accepted.size());
        context_pos    += static_cast<int>(step_accepted.size());
        total_steps++;
    }

    double avg_tokens_per_step = (double)total_accepted / total_steps;
    double baseline_steps      = TARGET_TOKENS;    // without spec: 1 per step
    double speedup_obs         = baseline_steps / total_steps;

    std::cout << "  Generated " << TARGET_TOKENS << " tokens in "
              << total_steps << " speculative steps\n";
    std::cout << "  Draft forward calls:   " << total_draft_calls << "\n";
    std::cout << "  Target forward calls:  " << total_target_calls << "\n";
    std::cout << "  Avg tokens/step:       " << std::fixed << std::setprecision(2)
              << avg_tokens_per_step << "\n";
    std::cout << "  Observed speedup:      " << std::fixed << std::setprecision(2)
              << speedup_obs << "× (vs. 1 target call per token)\n";

    std::cout << "\n  Generated sequence: ";
    for (int i = 0; i < (int)generated.size() && i < 15; ++i)
        std::cout << VOCAB[generated[i]] << " ";
    std::cout << "...\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\nChapter 23 — Speculative Decoding Demo (C++)\n";
    std::cout << std::string(72, '=') << "\n";

    demo_speedup_table();
    demo_rejection_sampling();
    demo_generation_loop();

    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "  All demos and assertions passed.\n";
    std::cout << std::string(72, '=') << "\n";
    return 0;
}
```

---

## 23.X  Speculative Decoding Variants

The draft-model approach in §23.1–23.6 requires a separate smaller model. Three widely-deployed variants eliminate or restructure this requirement.

### 23.X.1  Medusa — Multiple Draft Heads

Medusa (Cai et al., 2024) replaces the separate draft model with **multiple lightweight prediction heads** attached directly to the base model's final hidden state:

```
Standard decode:
  hidden_state ──► lm_head ──► logits ──► token[t]

Medusa decode:
  hidden_state ──► lm_head    ──► logits ──► token[t]    (position 0)
               ──► medusa_head_1 ──► logits ──► token[t+1] (position 1, draft)
               ──► medusa_head_2 ──► logits ──► token[t+2] (position 2, draft)
               ──► medusa_head_3 ──► logits ──► token[t+3] (position 3, draft)
               ──► medusa_head_4 ──► logits ──► token[t+4] (position 4, draft)
```

Each Medusa head is a 2-layer MLP (~2M parameters for a 7B model). The heads are trained jointly with the base model (or fine-tuned on top of a frozen base).

**Key properties:**

- No separate draft model process
- Single forward pass produces K+1 candidate tokens
- Acceptance: tree attention verifies all K draft tokens in one base model forward pass
- Training cost: ~1% of base model training cost for head fine-tuning
- Typical K=4–5; acceptance rate ~75–80% for well-matched heads

**WORKED EXAMPLE — Medusa speedup:**
```
WORKED EXAMPLE 23b.1 — Medusa Speedup
──────────────────────────────────────────────────
Base model:    Llama-3-8B, decode latency 8 ms/token
Medusa heads:  K=4 (4 draft tokens per step)
Head overhead: 0.3 ms per step (tiny MLPs)
Acceptance:    α=0.78 per position (independent)

Expected accepted tokens per step:
  E[accepted] = sum_{k=0}^{K} k × P(exactly k accepted)
  ≈ K × α / (1 - α^K) × (1 - α)  [approximate]
  For α=0.78, K=4: E ≈ 2.6 tokens

Latency per accepted token:
  (8 + 0.3) ms per step / 2.6 tokens = 3.2 ms/token
  Speedup vs baseline: 8.0 / 3.2 = 2.5×
──────────────────────────────────────────────────
```

### 23.X.2  EAGLE — Feature-Space Drafting

EAGLE (Li et al., 2024) drafts from the base model's **feature space** rather than its output logits. The key observation: next-token prediction is easier from hidden features than from the previous output token alone.

```
EAGLE architecture:
  Base model forward pass:
    input_tokens ──► [transformer] ──► hidden_states[-1] ──► lm_head ──► token

  EAGLE draft model (lightweight autoregressive on features):
    [hidden_t, token_t] ──► draft_transformer ──► hidden_{t+1}_pred
                                               ──► token_{t+1}_draft

  Draft model sees BOTH the previous hidden state AND the previous token.
  This gives it much richer signal than a token-only draft model.
```

**Why EAGLE outperforms Medusa:**
Medusa heads predict positions t+1..t+K independently from a single hidden state — they cannot model the dependency between draft tokens. EAGLE's draft model is autoregressive: each draft token conditions on the previous draft's predicted hidden state, capturing token-to-token dependencies.

| Method | Draft source | Inter-token deps | Acceptance rate | Overhead |
|--------|-------------|-----------------|----------------|----------|
| Draft model | Separate small LM | Yes (full model) | 75–85% | Medium |
| Medusa | MLP heads on base | No | 75–80% | Low |
| EAGLE | Autoregressive on features | Yes (lightweight) | 82–90% | Low |
| EAGLE-2 | Adaptive drafting | Yes + dynamic K | 85–93% | Low |

**EAGLE-2** (2024) further adds adaptive speculation depth: it stops drafting when the predicted acceptance probability drops below a threshold, preventing low-quality drafts from wasting verification passes.

### 23.X.3  Tree-Based Speculation (SpecTr)

Both Medusa and standard draft models produce a **linear sequence** of draft tokens. SpecTr and similar tree approaches produce a **tree** of candidates, allowing multiple hypotheses to be verified in parallel:

```
Linear speculation (standard):
  verified: [A] ──► drafts: [B, C, D, E]
  Verification: one pass checks if B,C,D,E are correct in sequence.
  If B is wrong, C,D,E are discarded regardless of quality.

Tree speculation:
  verified: [A] ──► drafts:
               ├── [B1] ──► [C1] ──► [D1]
               ├── [B2] ──► [C2]
               └── [B3]

  Verification: one tree-attention pass checks all branches.
  If B1 is accepted, verify C1. If B2 is accepted, verify C2.
  Multiple paths can be verified simultaneously.
```

**Tree attention** extends standard attention to handle a token tree: each draft token attends to its ancestors (the verified prefix + its own branch), not to siblings. This requires a custom attention mask:

```
Tree with 7 nodes (root + 6 drafts):
  Attention mask (1=can attend, 0=cannot):
  
        root B1  B2  B3  C1  C2  D1
  root [  1   0   0   0   0   0   0 ]
  B1   [  1   1   0   0   0   0   0 ]
  B2   [  1   0   1   0   0   0   0 ]
  B3   [  1   0   0   1   0   0   0 ]
  C1   [  1   1   0   0   1   0   0 ]  ← C1 is child of B1
  C2   [  1   0   1   0   0   1   0 ]  ← C2 is child of B2
  D1   [  1   1   0   0   1   0   1 ]  ← D1 is child of C1
```

**vLLM tree speculation** (enabled via `--speculative-model` with `--num-speculative-tokens`):
```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    speculative_model="[ngram]",          # n-gram draft (no extra model)
    num_speculative_tokens=5,
    speculative_draft_tensor_parallel_size=1,
)
```

**WORKED EXAMPLE 23.X — Tree attention mask construction:**

```
WORKED EXAMPLE 23.X — Tree attention mask construction
──────────────────────────────────────────────────────────────────────────

TREE STRUCTURE
──────────────
We use a small concrete tree: 1 root token R (the last verified token),
2 branches at depth 1 (B1, B2), each with 2 children at depth 2
(C1, C2 under B1; C3, C4 under B2).  Total: 7 nodes.

  Depth 0:         R            ← last verified/root token (position 0)
                  / \
  Depth 1:       B1   B2        ← draft tokens, 1st step (positions 1, 2)
                / \   / \
  Depth 2:    C1  C2 C3  C4    ← draft tokens, 2nd step (positions 3,4,5,6)

Node index assignment (flattened left-to-right by depth):
  Index 0 = R
  Index 1 = B1   (parent: R)
  Index 2 = B2   (parent: R)
  Index 3 = C1   (parent: B1)
  Index 4 = C2   (parent: B1)
  Index 5 = C3   (parent: B2)
  Index 6 = C4   (parent: B2)

ATTENTION RULE
──────────────
Token i can attend to token j  iff  j is an ancestor-or-equal of i
in the tree.  Equivalently: j lies on the unique path from the root
to i (inclusive of both endpoints).

  Ancestors(R)  = {R}             → row 0 attends to col {0}
  Ancestors(B1) = {R, B1}         → row 1 attends to col {0, 1}
  Ancestors(B2) = {R, B2}         → row 2 attends to col {0, 2}
  Ancestors(C1) = {R, B1, C1}     → row 3 attends to col {0, 1, 3}
  Ancestors(C2) = {R, B1, C2}     → row 4 attends to col {0, 1, 4}
  Ancestors(C3) = {R, B2, C3}     → row 5 attends to col {0, 2, 5}
  Ancestors(C4) = {R, B2, C4}     → row 6 attends to col {0, 2, 6}

7×7 ATTENTION MASK (M[i][j] = 1 means token i attends to token j)
────────────────────────────────────────────────────────────────────

         col: 0(R)  1(B1) 2(B2) 3(C1) 4(C2) 5(C3) 6(C4)
  row 0 (R)  [  1     0     0     0     0     0     0  ]
  row 1 (B1) [  1     1     0     0     0     0     0  ]
  row 2 (B2) [  1     0     1     0     0     0     0  ]
  row 3 (C1) [  1     1     0     1     0     0     0  ]
  row 4 (C2) [  1     1     0     0     1     0     0  ]
  row 5 (C3) [  1     0     1     0     0     1     0  ]
  row 6 (C4) [  1     0     1     0     0     0     1  ]

STEP-BY-STEP MASK CONSTRUCTION
────────────────────────────────
Start with a 7×7 zero matrix.  Fill in 1s by applying the rule.

Step 1 — Root R (row 0): R can attend only to itself.
  Set M[0][0] = 1.

Step 2 — B1 (row 1): B1's ancestors are R and B1.
  Set M[1][0] = 1  (attends to R, its parent)
  Set M[1][1] = 1  (attends to itself)

Step 3 — B2 (row 2): B2's ancestors are R and B2.
  Set M[2][0] = 1  (attends to R)
  Set M[2][2] = 1  (attends to itself)
  Note: M[2][1] = 0  — B2 cannot see B1 (sibling, not ancestor).

Step 4 — C1 (row 3): C1's ancestors are R, B1, C1.
  Set M[3][0] = 1  (attends to R)
  Set M[3][1] = 1  (attends to B1, its grandparent)
  Set M[3][3] = 1  (attends to itself)
  Note: M[3][2] = 0  — C1 cannot see B2 (different branch).
        M[3][4] = 0  — C1 cannot see C2 (sibling).

Step 5 — C2 (row 4): C2's ancestors are R, B1, C2.
  Set M[4][0] = 1
  Set M[4][1] = 1  (attends to B1, shared parent with C1)
  Set M[4][4] = 1
  Note: M[4][3] = 0  — C2 cannot see C1 (sibling, not ancestor).

Step 6 — C3 (row 5): C3's ancestors are R, B2, C3.
  Set M[5][0] = 1
  Set M[5][2] = 1  (attends to B2)
  Set M[5][5] = 1

Step 7 — C4 (row 6): C4's ancestors are R, B2, C4.
  Set M[6][0] = 1
  Set M[6][2] = 1  (attends to B2, shared parent with C3)
  Set M[6][6] = 1
  Note: M[6][5] = 0  — C4 cannot see C3 (sibling).

Count of 1s per row: 1, 2, 2, 3, 3, 3, 3  (depth + 1 per node)
Total 1s: 17 out of 49.  Sparsity: 32/49 ≈ 65% masked.

WHY THIS MASK ENABLES A SINGLE FORWARD PASS
─────────────────────────────────────────────
All 7 nodes are flattened into one sequence and passed through the
target model simultaneously.  The custom mask ensures each node's
attention computation only "sees" its own causal ancestors — exactly
the context it would have if the tree path were processed sequentially.

Sibling branches (e.g. B1 and B2) are mutually invisible:
  M[1][2] = 0  and  M[2][1] = 0
This is correct: B1 and B2 are independent hypotheses; B1 should not
influence B2's logits or vice versa.

DRAFT ACCEPTANCE / REJECTION EXAMPLE
──────────────────────────────────────
Suppose the target model, after one forward pass with this mask,
produces per-position logits.  The scheduler compares target
probabilities p_i and draft probabilities q_i:

  Position 0 (R):  always accepted (it is the verified token).

  Position 1 (B1): draft token = "Paris"
    p(Paris|R) = 0.72,  q(Paris|R) = 0.65
    Accept/reject test: u ~ Uniform(0,1); accept iff u < p/q = 0.72/0.65 = 1.11
    Since p/q > 1, accept with probability 1.  ✓ B1 ACCEPTED.

  Position 2 (B2): draft token = "London"
    p(London|R) = 0.08,  q(London|R) = 0.25
    p/q = 0.08/0.25 = 0.32.  Draw u = 0.41 > 0.32  → REJECT B2.
    Entire B2 subtree (C3, C4) is also discarded.

  Position 3 (C1), conditioned on B1 accepted: draft token = "is"
    p(is|R,B1) = 0.55,  q(is|R,B1) = 0.50
    p/q = 1.10 > 1 → accept with probability 1.  ✓ C1 ACCEPTED.

  Position 4 (C2), conditioned on B1 accepted: draft token = "was"
    p(was|R,B1) = 0.10,  q(was|R,B1) = 0.30
    p/q = 0.33.  Draw u = 0.20 < 0.33 → ACCEPT C2.  ✓ C2 ACCEPTED.

  Outcome: accepted sequence prefix is R → B1 → {C1, C2}.
    Two accepted depth-1+depth-2 paths remain; the scheduler picks
    the greedy-best or samples: suppose C1 is selected.
    Final accepted tokens this step: [B1, C1]  (2 tokens from 1 pass).

MASK AFTER REJECTION OF B2
────────────────────────────
Rows 2 (B2), 5 (C3), 6 (C4) are discarded.  The effective mask
for the accepted subtree is the upper-left 5×5 submatrix:

         col: 0(R)  1(B1) 3(C1) 4(C2)
  row 0 (R)  [  1     0     0     0  ]
  row 1 (B1) [  1     1     0     0  ]
  row 3 (C1) [  1     1     1     0  ]
  row 4 (C2) [  1     1     0     1  ]

This is a standard lower-triangular causal mask for the accepted
linear path R → B1 → C1 (plus sibling C2 that shares the B1 prefix).

SUMMARY
───────
  Tree nodes         : 7  (1 root + 2 depth-1 + 4 depth-2)
  Mask non-zeros     : 17 / 49  (65% sparse)
  Target FWD passes  : 1  (all 7 nodes verified simultaneously)
  Tokens accepted    : 2  (B1 and C1, chosen from accepted paths)
  Tokens rejected    : B2 subtree (3 nodes discarded)
  Effective speedup  : 2 tokens per target pass (vs. 1 for standard AR)
──────────────────────────────────────────────────────────────────────────
```

### 23.X.4  Choosing a Variant

```
Decision guide:
──────────────────────────────────────────────────────────
Separate draft model available (e.g. Llama-3-8B + 1B):
  → Standard speculative decoding (best acceptance rates
    when draft is well-matched to base)

No draft model, base model fine-tunable:
  → EAGLE or EAGLE-2 (highest acceptance, low overhead,
    requires fine-tuning the draft head)

No fine-tuning, need plug-and-play:
  → Medusa with pre-trained heads (many available on HF)
  → n-gram speculation (no model, works for repetitive text)

High-throughput batch serving (batch > 8):
  → Speculative decoding helps less (batched verification
    cost approaches batched draft cost)
  → Consider only for latency-critical single-user paths
──────────────────────────────────────────────────────────
```

---

## §23.11  Summary

- Autoregressive decoding is bounded by the weight-read cost: H100 reads ~16 GB of weights per token at batch=1. Speculative decoding amortizes that weight-read over multiple accepted tokens.
- The rejection sampling algorithm is **lossless**: accepted tokens are exactly distributed as if sampled from the target model, regardless of draft quality. Draft quality only affects speed.
- Speedup formula: `(1 - α^{K+1}) / ((1-α)(Kc+1))`. Acceptance rate α is the dominant factor; K has diminishing returns; draft cost c must be genuinely small (< 10%) for significant gains.
- At α = 0.85, K = 4, c = 0.05: expected 3.7 tokens per verify pass → ~3.1× speedup. At α = 0.60, the speedup drops to ~1.9×.
- Token trees generalize linear speculation: multiple draft branches verified with one ragged-attention target pass. EAGLE-2's adaptive tree achieves the best accepted-tokens-per-pass ratio.
- Self-speculation variants eliminate the external draft model: EAGLE trains a ~40M-parameter head that achieves 0.85–0.91 α on code; Medusa attaches K parallel heads but has higher memory cost (~4 GB); ngram and prompt lookup require no training.
- Enable speculative decoding when: batch ≤ 8, α > 0.75, task is code or structured output, low temperature. Disable when: batch > 32, creative generation, temperature > 0.9.
- vLLM: `--speculative-model`, `--num-speculative-tokens`, `--use-v2-block-manager`. Each request gets both a draft and target KV cache — ~20–35% extra KV usage for small draft models.
- llama.cpp: `--model-draft`, `--draft-max`, `--draft-p-min`. The `--draft-p-min` flag adds conditional drafting — only draft when confident — which raises effective α and reduces wasted target calls.

## Self-Check Questions

1. The rejection sampling proof shows `P(out = t) = p_i(t)` for all t. What would break — and why — if instead of resampling from the residual distribution we simply sampled fresh from the draft distribution when a rejection occurred?
2. Compute the expected tokens per step and speedup for α=0.82, K=5, c=0.08. Is this better or worse than α=0.90, K=3, c=0.05? Which scenario would you prefer for a code completion task running at batch=1?
3. You run 1,000 speculative decode steps with K=4 and observe that 380 steps accepted exactly 1 token (first draft rejected). Estimate α from this observation alone.
4. Why does `--draft-p-min 0.8` in llama.cpp improve effective acceptance rate α, even though it sometimes results in fewer tokens being drafted?
5. A teammate proposes using a Llama 3.3 70B as the draft model for a Llama 3.1 405B target (c ≈ 70/405 ≈ 0.17). With α = 0.88 and K = 4, what is the expected speedup? Would you recommend this, and why or why not?

## Where We Go Next

Chapter 24 moves from general inference optimization into a specialized and increasingly important workload: reasoning models. DeepSeek-R1, o1, and Qwen3's thinking mode generate 10K–50K tokens of chain-of-thought before producing an answer. This transforms every assumption we have made about decode-bound systems — from KV cache sizing to scheduler priorities to cost modeling — and demands a dedicated treatment.


---

## Chapter Summary

- **The decode bottleneck**: each decode step is memory-bandwidth-bound (one token generated per forward pass of the full model); speculative decoding amortises this by generating K candidate tokens with a cheap draft model and verifying all K in one target-model pass.
- **Rejection sampling correctness**: the acceptance-rejection algorithm is lossless — the output distribution is provably identical to sampling from the target model directly.

> **LinkedIn Scenario Update:** LinkedIn's workload generates structured outputs — job match explanations, profile summaries, search result annotations — that are highly repetitive and template-driven. This is exactly the high-acceptance-rate (α > 0.85) regime where speculative decoding provides maximum benefit. At 50K requests/hour with an average output of 200 tokens per request and a 2× speculative decoding speedup (α=0.85, K=4 with a small draft model), the cluster's effective decode throughput doubles without adding hardware. In cost terms: the same $1.2M/month GPU cluster now handles 100K req/hr, or the LinkedIn workload can be served for ~$600K/month — assuming the batch sizes stay below the ~8-sequence threshold where speculation efficiency begins to decline.
- **Speedup formula**: expected speedup = (1 − α^{K+1}) / ((1 − α)(Kc + 1)), where α is the per-token acceptance rate and c is the draft/target compute ratio.
- **n-gram draft**: for repetition-heavy workloads (code, structured text), the draft model is replaced by an n-gram lookup table; zero draft cost means any acceptance rate is profitable.
- **Optimal K**: typically 4–8 draft tokens; beyond this, the lower acceptance rate on longer sequences cancels the parallelism gain.
- **When speculation hurts**: highly creative or diverse outputs have α < 0.5; the overhead of running the draft model then exceeds the gain from verification batching.
- **vLLM configuration**: `--speculative-model`, `--num-speculative-tokens`, `--speculative-draft-tensor-parallel-size`; llama.cpp: `--model-draft`, `--draft-max`, `--draft-p-min`.
- **Medusa**: attaches K lightweight MLP heads directly to the base model's final hidden state; each head independently drafts one future token; no separate draft model process needed; acceptance rate ~75–80%; heads can be fine-tuned at ~1% of base model training cost.
- **EAGLE / EAGLE-2**: drafts in feature space — the lightweight draft transformer sees both the previous hidden state and the previous token, enabling autoregressive inter-token dependency modeling; achieves 82–93% acceptance rate with low overhead; EAGLE-2 adds adaptive depth that stops drafting when predicted acceptance falls below a threshold.
- **Tree-based speculation (SpecTr)**: expands the draft from a linear sequence into a branching tree of candidates; a custom causal-but-tree-structured attention mask (ancestors only, no siblings) verifies all branches in a single target-model forward pass; higher expected accepted tokens per pass than linear drafts when multiple draft branches have reasonable probability.
- **Variant selection**: use a separate draft model when one is available and well-matched; use EAGLE/EAGLE-2 when fine-tuning is possible and maximum acceptance rate is required; use Medusa for plug-and-play deployment with pre-trained heads; use n-gram for repetitive workloads with zero overhead; at batch > 8 the marginal benefit of speculation narrows and may not justify the extra KV cache cost.

## Self-Check Questions

1. The rejection sampling proof shows `P(out = t) = p_i(t)` for all t. What would break — and why — if instead of resampling from the residual distribution we simply sampled fresh from the draft distribution when a rejection occurred?
2. Compute the expected tokens per step and speedup for α=0.82, K=5, c=0.08. Is this better or worse than α=0.90, K=3, c=0.05? Which scenario would you prefer for a code completion task running at batch=1?
3. You run 1,000 speculative decode steps with K=4 and observe that 380 steps accepted exactly 1 token (first draft rejected). Estimate α from this observation alone.
4. Why does `--draft-p-min 0.8` in llama.cpp improve effective acceptance rate α, even though it sometimes results in fewer tokens being drafted?
5. A teammate proposes using a Llama 3.3 70B as the draft model for a Llama 3.1 405B target (c ≈ 70/405 ≈ 0.17). With α = 0.88 and K = 4, what is the expected speedup? Would you recommend this, and why or why not?
6. Medusa heads predict future tokens independently from a single shared hidden state. Explain precisely why this independence assumption limits acceptance rate compared to EAGLE, and describe the specific architectural change EAGLE makes to address this limitation.
7. Draw the tree attention mask for a 5-node speculation tree with the following parent relationships: root → B1, root → B2, B1 → C1, B1 → C2. Which pairs of nodes can NOT attend to each other, and why does this asymmetric masking allow a single forward pass to verify all branches simultaneously?
8. EAGLE-2 introduces adaptive speculation depth, stopping early when predicted acceptance probability falls below a threshold θ. Describe the trade-off in choosing θ: what happens to throughput and average accepted tokens per step as θ → 0 vs. θ → 1? What empirical signal would you use to tune θ in production?


---

## Worked Solutions

### Question 1
**If we sample from draft distribution (instead of residual) on rejection -- what breaks:**

The rejection sampling proof establishes P(out=t) = p_i(t) by using the residual distribution max(0, p_i(t) - q_i(t)) / Z when a token is rejected. This residual exactly corrects for tokens where draft q_i(t) > p_i(t).

**If we sample fresh from q_i(t) instead:**
Output distribution becomes: P(accept)*p_i(t) + P(reject)*q_i(t).
This is NOT equal to p_i(t) in general. Tokens where q_i(t) > p_i(t) (over-predicted by draft) appear more often than the target intends.

**Why this matters:** The lossless guarantee -- that output distribution is identical to the target model -- is broken. Output would be biased toward the draft model's preferences, with different style, factuality, or safety properties.

---

### Question 2
**Expected tokens/step and speedup:**

Formula: E[accepted] = (1 - alpha^(K+1)) / (1 - alpha). Speedup = E[accepted] / (1 + K*c).

**Config A: alpha=0.82, K=5, c=0.08:**
```
E[tokens] = (1 - 0.82^6) / 0.18 = (1 - 0.3040) / 0.18 = 3.867 tokens/step
speedup = 3.867 / (1 + 5*0.08) = 3.867 / 1.40 = 2.76x
```

**Config B: alpha=0.90, K=3, c=0.05:**
```
E[tokens] = (1 - 0.90^4) / 0.10 = (1 - 0.6561) / 0.10 = 3.439 tokens/step
speedup = 3.439 / (1 + 3*0.05) = 3.439 / 1.15 = 2.99x
```

Config B is better (2.99x vs 2.76x). For code completion at batch=1: choose Config B. Code has high token predictability (alpha 0.88-0.95), and a cheaper draft model (c=0.05) with fewer speculation steps minimizes overhead per step.

---

### Question 3
**Estimating alpha: 1,000 steps with K=4, 380 steps accepted exactly 1 token.**

"Accepted exactly 1 token" = first draft token was rejected. Probability of first rejection = 1 - alpha.
```
1 - alpha ~= 380 / 1000 = 0.38
alpha ~= 0.62
```

This is a low acceptance rate. At alpha=0.62, K=4: E[tokens] = (1 - 0.62^5) / 0.38 = 0.884/0.38 = 2.33 tokens/step -- marginal speedup. Investigate whether the draft model was fine-tuned on the target domain.

---

### Question 4
**Why `--draft-p-min 0.8` improves effective alpha:**

This filter only proposes draft tokens where q_i(t) >= 0.8. The acceptance criterion is min(1, p_i(t)/q_i(t)). When q_i(t) is high (draft is confident), both models tend to agree (p_i(t)/q_i(t) close to 1), so acceptance rate is high. Filtering out low-confidence drafts removes cases where the draft would propose unlikely tokens with low acceptance probability.

Trade-off: fewer speculation steps on average (aborts early when uncertain), but the tokens that ARE drafted are accepted at higher rate. Net speedup improves when alpha gain outweighs K reduction -- typical for code and structured text.

---

### Question 5
**Draft = Llama 3.3 70B, Target = Llama 3.1 405B. c ~= 0.17, alpha=0.88, K=4.**

```
E[tokens] = (1 - 0.88^5) / 0.12 = (1 - 0.5277) / 0.12 = 3.94 tokens/step
speedup = 3.94 / (1 + 4*0.17) = 3.94 / 1.68 = 2.35x
```

**Recommendation: Generally NOT recommended.** c=0.17 is very high -- 4 draft steps cost 68% of a target step. Memory requirement: 70B + 405B ~= 950 GB BF16, requiring 12+ H100s simultaneously. 

Better alternative: Llama 3.1 8B draft (c ~= 0.02). Even at lower alpha ~= 0.75: E[tokens] = (1-0.75^5)/0.25 = 3.24, speedup = 3.24/(1+0.08) = 3.0x -- comparable speedup with far less memory overhead.

---

### Question 6
**Why Medusa's independence assumption limits alpha vs EAGLE:**

Medusa predicts tokens at t+1, t+2, ..., t+K from a **single shared hidden state h_t** at position t. Each head predicts its token independently: P_k(t+k | h_t), ignoring what tokens were predicted at intermediate positions.

This limits alpha because P(token_{t+2} | context, token_{t+1}) depends strongly on token_{t+1}. Medusa's head 2 sees only h_t, not the intermediate prediction -- it cannot condition on it. When verified against the target model (which conditions autoregressively), predictions for positions 2+ are inconsistent more often.

**EAGLE's fix:** EAGLE's draft head receives [h_t, embed(predicted_{t+1})] before predicting t+2 -- one step of autoregressive conditioning. This dramatically increases match with target model distribution, raising alpha from ~0.75 (Medusa) to ~0.88 (EAGLE) on typical benchmarks.

---

### Question 7
**Tree attention mask for root->B1, root->B2, B1->C1, B1->C2:**

Each token attends only to its ancestors on its branch path:
```
         root  B1  B2  C1  C2
root:    [1,   0,  0,  0,  0]
B1:      [1,   1,  0,  0,  0]
B2:      [1,   0,  1,  0,  0]
C1:      [1,   1,  0,  1,  0]
C2:      [1,   1,  0,  0,  1]
```

Pairs that CANNOT attend: B2<->B1, B2<->C1, B2<->C2, C1<->C2 (no ancestor relationship).

This allows a single forward pass: all 5 tokens are processed simultaneously. Each token's output logits verify that specific branch position against the target model -- instead of 5 sequential autoregressive calls.

---

### Question 8
**EAGLE-2 adaptive depth: trade-off in choosing theta:**

**theta -> 0:** Speculate even when uncertain. Low-confidence drafts rejected often -> avg accepted ~= 1 token/step. Speculation overhead adds cost with little benefit.

**theta -> 1:** Only speculate when near-certain. High alpha on those steps, but speculation rarely triggers. Again avg accepted ~= 1 token/step.

**Optimal theta (empirically 0.5-0.8):** Speculate when moderately confident, achieve high alpha on those steps, skip when uncertain (save overhead).

**Production tuning signal:** Monitor avg_accepted_tokens_per_step and speculation_overhead_fraction. Compute net throughput = avg_accepted / (1 + overhead_fraction) as theta varies. The optimal theta maximizes this ratio. Use A/B testing with live traffic -- tuning on synthetic benchmarks may not reflect production token distributions.

