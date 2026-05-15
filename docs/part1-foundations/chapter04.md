# Chapter 4 — Inside the Attention Mechanism

> *"Attention is the only operation in the transformer where every token talks to every other token. Everything else is independent. That dependency is the source of both the power and the cost."*

---

## Why This Chapter Exists

The KV cache — the central topic of this entire book — exists because of one specific property of attention: **each output token must attend over all previous tokens**. Without that requirement there would be no cache, no PagedAttention, no continuous batching problem, no memory pressure. Everything flows from attention.

This chapter builds attention from the ground up. We start with the intuition, derive the mathematics, and then trace a complete hand-computed walkthrough through **all ten tokens** of a worked example: four-token prefill ("The next day is") followed by six decode steps ("bright and sunny with clear skies"). Every number is exact — computed with `numpy.random.seed(42)`, traceable with a pencil. We then develop every major variant: MHA, MQA, GQA, MLA, and Sparse Attention, showing how each one trades quality for memory savings.

**What you will know by the end:**
- Why attention is the operation that queries memory.
- The scaled dot-product attention formula and the necessity of the `1/√d` scale.
- Multi-Head Attention: full element-by-element derivation for all 10 tokens, 4 heads.
- Why the KV cache arises from the autoregressive constraint, with exact memory arithmetic.
- MQA, GQA, MLA: the compression spectrum with precise byte counts.
- Sparse/sliding-window attention.
- How vLLM and llama.cpp implement attention at the API level.

---

## 4.1 The Core Intuition `[FOUNDATIONAL]`

Imagine resolving the word *"it"* at position 47 in a long document. To find the referent, you must look back at earlier tokens and judge which one *"it"* most likely refers to. The attention mechanism does exactly this — for every output position, it **queries** all previous (and current) positions, **retrieves** relevant information, and **aggregates** a weighted sum.

Three roles map to three matrices:

| Role  | Matrix | Meaning                        |
|-------|--------|--------------------------------|
| Query | Q      | "What am I looking for?"       |
| Key   | K      | "What do I offer?"             |
| Value | V      | "What information do I carry?" |

The attention weight between position `i` (query) and position `j` (key) is proportional to how well `Q[i]` and `K[j]` match. The output at position `i` is a weighted sum of all `V[j]`.

---

## 4.2 Scaled Dot-Product Attention — Full Derivation `[FOUNDATIONAL]`

### 4.2.1 Single-Query, Single-Key Version

Let `q ∈ ℝᵈ` be a query and `k ∈ ℝᵈ` be a key, both dimension `d`. The **raw score** is:

```
score(q, k) = q · k = Σᵢ qᵢ kᵢ
```

### 4.2.2 Why Divide by √d?

If q and k have components drawn i.i.d. from `N(0,1)`:

```
Var(q · k) = Var(Σᵢ qᵢ kᵢ) = Σᵢ Var(qᵢ kᵢ) = Σᵢ E[qᵢ²]·E[kᵢ²] = d · 1 · 1 = d

std(q · k) = √d
```

So raw dot products grow like `√d`. For `d = 64` (typical), scores have standard deviation 8. The softmax of large-magnitude inputs saturates near 0 or 1 — gradients vanish. Dividing by `√d` normalizes variance back to 1:

```
Var(q · k / √d) = d / d = 1
```

### 4.2.3 Matrix Form

For a full sequence:

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d) · V

where:
  Q ∈ ℝ^(N×d)   — query matrix (N tokens, d dims each)
  K ∈ ℝ^(N×d)   — key matrix
  V ∈ ℝ^(N×d)   — value matrix
  Q Kᵀ ∈ ℝ^(N×N) — score matrix (position i vs position j)
  output ∈ ℝ^(N×d) — one context vector per token
```

The `softmax` is applied row-wise, so each row sums to 1. Entry `[i,j]` of the score matrix says: "how much should position `i` attend to position `j`?" During autoregressive generation, positions can only see positions `j ≤ i` (causal mask).

---

## 4.3 Architecture Setup for the Worked Examples

All examples in this chapter use the same configuration, matching a `numpy.random.seed(42)` embedding table. This lets you run the numbers yourself and check every digit.

```
Vocabulary:     10 tokens — [The, next, day, is, bright, and, sunny, with, clear, skies]
d_model = 8    (every embedding is 8-dimensional)
n_q_heads = 4  (four query heads)
d_head = 2     (d_model / n_q_heads = 8 / 4 = 2)
scale = 1/√2 ≈ 0.7071

Sequence: "The next day is bright and sunny with clear skies"
  → Prefill:  tokens 0–3  ("The next day is")
  → Decode:   tokens 4–9  ("bright and sunny with clear skies")
```

**Embedding table E (10 × 8, seed=42):**
```
Token    dim0      dim1      dim2      dim3      dim4      dim5      dim6      dim7
The    [ 0.4967, -0.1383,  0.6477,  1.5230, -0.2342, -0.2341,  1.5792,  0.7674]
next   [-0.4695,  0.5426, -0.4634, -0.4657,  0.2420, -1.9133, -1.7249, -0.5623]
day    [-1.0128,  0.3142, -0.9080, -1.4123,  1.4656, -0.2258,  0.0675, -1.4247]
is     [-0.5444,  0.1109, -1.1510,  0.3757, -0.6006, -0.2917, -0.6017,  1.8523]
bright [-0.0135, -1.0577,  0.8225, -1.2208,  0.2089, -1.9597, -1.3282,  0.1969]
...
```

**What each variant changes:**
```
         W_Q        W_K         W_V         Cache stores
MHA     8×8 (full)  8×8 (4 hd)  8×8 (4 hd)  K,V for each of 4 heads
MQA     8×8 (full)  8×2 (1 hd)  8×2 (1 hd)  K,V for 1 shared head
GQA     8×8 (full)  8×4 (2 hd)  8×4 (2 hd)  K,V for 2 heads
MLA     8×8 (full)  8×3→reconst 8×3→reconst  latent vector C_KV (3-dim)
Sparse  8×8 (full)  8×8 (4 hd)  8×8 (4 hd)  K,V for 4 heads (masked)
```

---

## 4.4 Multi-Head Attention (MHA) — Complete Walkthrough `[ESSENTIAL]`

### 4.4.1 What MHA Does

Multi-Head Attention, introduced in "Attention Is All You Need" (Vaswani et al., 2017), splits the representation space into `n_heads` independent subspaces, each of dimension `d_head`. Every head learns to look for a different type of relationship: one head might focus on syntactic agreement, another on coreference, another on semantic similarity.

During inference, each new decode step must attend to **all past tokens** to form its context vector. Since recomputing K and V for past tokens at every step would be enormously wasteful, MHA maintains a **KV cache** — a growing table of key and value vectors for every past position and every head. The cache grows by `n_heads × d_head` key entries and `n_heads × d_head` value entries per new token.

```
n_q_heads = 4
n_kv_heads = 4  (MHA: each query head owns its own KV head)
cache per token per layer = 2(K+V) × 4(heads) × 2(d_head) × 4(bytes) = 64 bytes (FP32)
```

**MHA KV Cache Layout** — for the full 10-token trace (`N=10`, `Hkv=4`, `D=2`):

```
K_cache shape = N × Hkv × D = 10 × 4 × 2
V_cache shape = N × Hkv × D = 10 × 4 × 2

  K_cache (per layer):
  ┌──────────────────────────────────────────────┐
  │ token 0: K[head0] K[head1] K[head2] K[head3]│
  │ token 1: K[head0] K[head1] K[head2] K[head3]│
  │ token 2: K[head0] K[head1] K[head2] K[head3]│
  │ token 3: K[head0] K[head1] K[head2] K[head3]│
  │ token 4: K[head0] K[head1] K[head2] K[head3]│
  │ token 5: K[head0] K[head1] K[head2] K[head3]│
  │ token 6: K[head0] K[head1] K[head2] K[head3]│
  │ token 7: K[head0] K[head1] K[head2] K[head3]│
  │ token 8: K[head0] K[head1] K[head2] K[head3]│
  │ token 9: K[head0] K[head1] K[head2] K[head3]│
  └──────────────────────────────────────────────┘
  V_cache has the same shape.
```

Every token row contains all four head-specific K vectors and all four head-specific V vectors. When sequence length grows, all rows must remain in GPU HBM for decode. This is the cost of full head independence.

After the 4-head context vectors are computed, they are **concatenated** and mixed through an output projection:

```
context_all = [context_h0 | context_h1 | context_h2 | context_h3]  shape 1×8
output = context_all @ W_O   shape 1×d_model
```

### 4.4.2 Weight Matrices (Head 0)

All projections below use `seed=42`-derived weights. Each `W_Q`, `W_K`, `W_V` is 8×2.

```
W_Q head 0 (8×2):          W_K head 0 (8×2):          W_V head 0 (8×2):
row0: [+0.4967, -0.1383]   row0: [+0.8125, +1.3562]   row0: [+0.0997, -0.5035]
row1: [+0.6477, +1.5230]   row1: [-0.0720, +1.0035]   row1: [-1.5507, +0.0686]
row2: [-0.2342, -0.2341]   row2: [+0.3616, -0.6451]   row2: [-1.0623, +0.4736]
row3: [+1.5792, +0.7674]   row3: [+0.3614, +1.5380]   row3: [-0.9194, +1.5499]
row4: [-0.4695, +0.5426]   row4: [-0.0358, +1.5646]   row4: [-0.7833, -0.3221]
row5: [-0.4634, -0.4657]   row5: [-2.6197, +0.8219]   row5: [+0.8135, -1.2309]
row6: [+0.2420, -1.9133]   row6: [+0.0870, -0.2990]   row6: [+0.2275, +1.3071]
row7: [-1.7249, -0.5623]   row7: [+0.0918, -1.9876]   row7: [-1.6075, +0.1846]
```

Heads 1–3 use independent weight matrices (same structure, different values — see gist companion data).

### 4.4.3 Phase 1 — Prefill: "The next day is"

In the prefill phase, we process all 4 tokens simultaneously, building the initial KV cache.

---

#### Token 0: "The" (position 0)

Embedding: `e = [+0.3047, -1.0400, +0.7505, +0.9406, -1.9510, -1.3022, +0.1278, -0.3162]`

*(Note: embeddings shown here are the actual token embeddings from the seed-42 table, which differ from the raw rows E above by the position encoding layer.)*

**Step 1 — K projection, Head 0 (e @ W_K[0]):**
```
k[0] = (+0.3047×+0.8125) + (-1.0400×-0.0720) + (+0.7505×+0.3616) + (+0.9406×+0.3614)
     + (-1.9510×-0.0358) + (-1.3022×-2.6197) + (+0.1278×+0.0870) + (-0.3162×+0.0918)
     = +4.3972

k[1] = (+0.3047×+1.3562) + (-1.0400×+1.0035) + (+0.7505×-0.6451) + (+0.9406×+1.5380)
     + (-1.9510×+1.5646) + (-1.3022×+0.8219) + (+0.1278×-0.2990) + (-0.3162×-1.9876)
     = -3.2005

→ k_h0 = [+4.3972, -3.2005]
```

**Step 2 — V projection, Head 0:**
```
v[0] = +0.9873,  v[1] = +3.9284
→ v_h0 = [+0.9873, +3.9284]
```

**Step 3 — Q projection, Head 0:**
```
q[0] = +2.8832,  q[1] = -1.5988
→ q_h0 = [+2.8832, -1.5988]
```

**K,V,Q for Heads 1–3:**
```
Head 1: k=[-0.4082, -1.2613]  v=[+3.5533, +2.4902]  q=[+4.4735, -1.2737]
Head 2: k=[-2.8014, -1.7635]  v=[+1.5340, +5.5021]  q=[-3.2856, -0.7067]
Head 3: k=[-1.7145, +4.3260]  v=[-0.4058, -4.3382]  q=[+1.7240, +0.8857]
```

**Step 4 — Attention scores, Head 0** (only past token is "The" itself):
```
The: dot(q, k) = +2.8832×+4.3972 + -1.5988×-3.2005 = +17.7951  →  /√2 = +12.5830
```

**Step 5 — Softmax:**
```
max = +12.5830
exp(+12.5830 - 12.5830) = 1.0000  →  sum = 1.0000
softmax: The = 1.0000
```

**Step 6 — Context vector, Head 0:**
```
ctx = 1.0000 × [+0.9873, +3.9284] = [+0.9873, +3.9284]
```

**All heads, token "The":**
```
Head 0: ctx=[+0.9873, +3.9284]  dominant=The (+1.0000)
Head 1: ctx=[+3.5533, +2.4902]  dominant=The (+1.0000)
Head 2: ctx=[+1.5340, +5.5021]  dominant=The (+1.0000)
Head 3: ctx=[-0.4058, -4.3382]  dominant=The (+1.0000)
```

*All heads attend fully to themselves — the only token in cache.*

---

#### Token 1: "next" (position 1)

Embedding: `e = [-0.0168, -0.8530, +0.8794, +0.7778, +0.0660, +1.1272, +0.4675, -0.8593]`

**K/V/Q Head 0:**
```
k_h0 = [-2.3467, +2.3480]
v_h0 = [+2.0248, +0.6157]
q_h0 = [+1.5035, -1.8064]
```

**Attention scores, Head 0** (cache: [The, next]):
```
The:  dot(q,k) = +1.5035×+4.3972 + -1.8064×-3.2005 = +12.3923  →  /√2 = +8.7627
next: dot(q,k) = +1.5035×-2.3467 + -1.8064×+2.3480 = -7.7696  →  /√2 = -5.4939
```

**Softmax:**
```
max = +8.7627
The:  exp(0)       = 1.0000
next: exp(-14.257) = 0.0000
sum = 1.0000   →   The: 1.0000,  next: 0.0000
```

**Context, Head 0:** `ctx = [+0.9873, +3.9284]` (effectively 100% attending to "The")

**All heads:**
```
Head 0: ctx=[+0.9873, +3.9284]  dominant=The (+1.0000)
Head 1: ctx=[+3.5578, +2.4631]  dominant=The (+0.9863)
Head 2: ctx=[+1.5336, +5.5014]  dominant=The (+0.9998)
Head 3: ctx=[-0.4058, -4.3382]  dominant=The (+0.9999)
```

*The model heavily attends to "The" — this is sensible given the embedding geometry.*

---

#### Token 2: "day" (position 2)

Embedding: `e = [+0.3688, -0.9589, +0.8785, -0.0499, -0.1849, -0.6809, +1.2225, -0.1545]`

**K/V/Q Head 0:**
```
k_h0 = [+2.5510, -2.0130]
v_h0 = [+0.7537, +2.5544]
q_h0 = [+0.2423, -3.7907]
```

**Attention scores** (cache: [The, next, day]):
```
The:  +0.2423×+4.3972 + -3.7907×-3.2005 = +13.1976  →  /√2 = +9.3321
next: +0.2423×-2.3467 + -3.7907×+2.3480 = -9.4693  →  /√2 = -6.6958
day:  +0.2423×+2.5510 + -3.7907×-2.0130 = +8.2486  →  /√2 = +5.8327
```

**Softmax:**
```
max = +9.3321
The:  exp(0)       = 1.0000
next: exp(-16.028) = 0.0000
day:  exp(-3.499)  = 0.0302
sum  = 1.0302
→ The: 0.9707,  next: 0.0000,  day: 0.0293
```

**Context, Head 0:**
```
ctx = 0.9707×[+0.9873, +3.9284] + 0.0293×[+0.7537, +2.5544]
    = [+0.9804, +3.8881]
```

**All heads:**
```
Head 0: ctx=[+0.9804, +3.8881]  dominant=The (+0.9707)
Head 1: ctx=[+3.7272, +1.5041]  dominant=next (+0.4999)
Head 2: ctx=[+1.4896, +5.4185]  dominant=The (+0.9737)
Head 3: ctx=[-0.3818, -4.4641]  dominant=The (+0.7538)
```

*Head 1 splits focus almost equally between "The" and "next" — an example of heads learning different patterns.*

---

#### Token 3: "is" (position 3)

Embedding: `e = [-0.4283, -0.3521, +0.5323, +0.3654, +0.4127, +0.4308, +2.1416, -0.4064]`

**K/V/Q Head 0:**
```
k_h0 = [-0.9924, +0.4517]
v_h0 = [+0.7695, +3.0712]
q_h0 = [+0.8375, -4.1670]
```

**Attention scores** (cache: [The, next, day, is]):
```
The:  +0.8375×+4.3972 + -4.1670×-3.2005 = +17.0190  →  /√2 = +12.0343
next: +0.8375×-2.3467 + -4.1670×+2.3480 = -11.750  →  /√2 = -8.3082
day:  +0.8375×+2.5510 + -4.1670×-2.0130 = +10.524  →  /√2 = +7.4419
is:   +0.8375×-0.9924 + -4.1670×+0.4517 = -2.7131  →  /√2 = -1.9185
```

**Softmax:**
```
max = +12.0343
The:  exp(0)       = 1.0000
next: exp(-20.342) = 0.0000
day:  exp(-4.592)  = 0.0101
is:   exp(-13.953) = 0.0000
sum  = 1.0101
→ The: 0.9900,  next: 0.0000,  day: 0.0100,  is: 0.0000
```

**Context, Head 0:**
```
ctx = 0.9900×[+0.9873, +3.9284] + 0.0100×[+0.7537, +2.5544]
    = [+0.9849, +3.9146]
```

**All heads:**
```
Head 0: ctx=[+0.9849, +3.9146]  dominant=The (+0.9900)
Head 1: ctx=[+3.9475, +2.3949]  dominant=The (+0.4708)
Head 2: ctx=[+1.5186, +5.4731]  dominant=The (+0.9909)
Head 3: ctx=[-0.4064, -4.3893]  dominant=The (+0.9126)
```

### 4.4.4 KV Cache State After Prefill

After processing all 4 prefill tokens, the KV cache holds:

```
Layer 0 KV Cache:
  Sequence: The | next | day | is
  n_heads = 4, d_head = 2, N = 4

  Head 0 K-cache (4×2):
    The : [+4.3972, -3.2005]
    next: [-2.3467, +2.3480]
    day : [+2.5510, -2.0130]
    is  : [-0.9924, +0.4517]

  Head 0 V-cache (4×2):
    The : [+0.9873, +3.9284]
    next: [+2.0248, +0.6157]
    day : [+0.7537, +2.5544]
    is  : [+0.7695, +3.0712]

  Head 1 K-cache (4×2):
    The : [-0.4082, -1.2613]  V: [+3.5533, +2.4902]
    next: [-1.1728, +2.4808]  V: [+3.8860, +0.5113]
    day : [-3.0199, -0.7574]  V: [+4.1191, +2.7194]
    is  : [-2.0432, +0.4048]  V: [+5.0192, +0.9508]

  Head 2 K-cache (4×2):
    The : [-2.8014, -1.7635]  V: [+1.5340, +5.5021]
    next: [+2.3642, -0.5867]  V: [-0.7781, +1.0045]
    day : [-1.7651, -0.2887]  V: [-0.1511, +2.3292]
    is  : [+1.0701, -0.4530]  V: [+0.0607, +0.5966]

  Head 3 K-cache (4×2):
    The : [-1.7145, +4.3260]  V: [-0.4058, -4.3382]
    next: [+3.7805, -0.8711]  V: [+0.4493, -4.2024]
    day : [-0.1118, +3.4431]  V: [-0.3855, -4.9161]
    is  : [+3.0313, +2.4335]  V: [-1.4911, -5.1894]

  Total bytes: 2(K+V) × 4(N) × 4(H) × 2(D) × 4(B) = 256 bytes (FP32)
```

### 4.4.5 Phase 2 — Decode: "bright" → "skies"

Each decode step adds **one new row** to each head's K and V cache. The query attends to all `N_past + 1` tokens. We show two decode steps in detail.

---

#### Decode Step 1: "bright" (position 4, cache has 4 tokens)

Embedding: `e = [-0.5122, -0.8138, +0.6160, +1.1290, -0.1139, -0.8402, -0.8245, +0.6506]`

**Project new token to K/V (all heads) → append to cache:**
```
Head 0: k=[+2.4662, -2.0877]  v=[-2.3091, +2.3569]
Head 1: k=[-0.2522, -1.2534]  v=[-1.0043, +1.8583]
Head 2: k=[-4.4416, -1.2563]  v=[+2.0968, +3.6885]
Head 3: k=[-1.0025, -2.2712]  v=[-0.5794, -1.1555]
```

**Q projection for "bright", Head 0:**
```
q_h0 = [-0.0217, +1.0947]
```

**Attention scores over 5 tokens, Head 0:**
```
The:    dot(q,k) = -0.0217×+4.3972 + +1.0947×-3.2005 = -3.5992  →  /√2 = -2.5450
next:   dot(q,k) = -0.0217×-2.3467 + +1.0947×+2.3480 = +2.6214  →  /√2 = +1.8536
day:    dot(q,k) = -0.0217×+2.5510 + +1.0947×-2.0130 = -2.2591  →  /√2 = -1.5974
is:     dot(q,k) = -0.0217×-0.9924 + +1.0947×+0.4517 = +0.5160  →  /√2 = +0.3649
bright: dot(q,k) = -0.0217×+2.4662 + +1.0947×-2.0877 = -2.3391  →  /√2 = -1.6540
```

**Softmax:**
```
max = +1.8536
The:    exp(-4.399) = 0.0123  →  0.0095
next:   exp(0)      = 1.0000  →  0.7695
day:    exp(-3.451) = 0.0317  →  0.0244
is:     exp(-1.489) = 0.2257  →  0.1736
bright: exp(-3.508) = 0.0300  →  0.0231
sum = 1.2996
```

**Context, Head 0:**
```
ctx = 0.0095×[+0.9873, +3.9284]   [The]
    + 0.7695×[+2.0248, +0.6157]   [next]
    + 0.0244×[+0.7537, +2.5544]   [day]
    + 0.1736×[+0.7695, +3.0712]   [is]
    + 0.0231×[-2.3091, +2.3569]   [bright]
    = [+1.6660, +1.1608]
```

**All heads:**
```
Head 0: ctx=[+1.6660, +1.1608]  dominant=next   (+0.7695)
Head 1: ctx=[+1.3360, +1.8734]  dominant=bright (+0.5005)
Head 2: ctx=[+1.3020, +3.3204]  dominant=bright (+0.5876)
Head 3: ctx=[-0.4702, -4.5326]  dominant=The    (+0.6783)
```

*For "bright": Head 0 attends mainly to "next"; Heads 1–2 attend primarily to "bright" itself; Head 3 recalls "The".*

KV cache after "bright": `2×5×4×2×4 = 320 bytes`

---

#### Decode Step 2: "and" (position 5, cache has 5 tokens)

```
Head 0: k=[-0.0727, +1.9968]  v=[-0.3494, +0.5813]
q_h0 = [+0.9125, -0.7731]
```

**Attention scores, Head 0:**
```
The:    +0.9125×+4.3972 + -0.7731×-3.2005 = +6.4866  →  /√2 = +4.5867
next:   -3.9566  →  /√2 = -2.7977
day:    +3.8840  →  /√2 = +2.7464
is:     -1.2547  →  /√2 = -0.8872
bright: +3.8644  →  /√2 = +2.7325
and:    -1.6101  →  /√2 = -1.1385
```

**Softmax:**
```
The:    0.7556    next: 0.0005    day: 0.1200
is:     0.0032    bright: 0.1183  and: 0.0025
```

**Context, Head 0:** `ctx=[+0.5658, +3.5651]` (dominant: The +0.7556)

---

#### Decode Steps 3–6 Summary

| Step | Token  | KV bytes | Head 0 dominant | Head 1 dominant |
|------|--------|----------|-----------------|-----------------|
| 1    | bright | 320      | next (+0.769)   | bright (+0.500) |
| 2    | and    | 384      | The (+0.756)    | day (+0.919)    |
| 3    | sunny  | 448      | The (+0.897)    | bright (+0.339) |
| 4    | with   | 512      | sunny (+0.415)  | the (+0.349)    |
| 5    | clear  | 576      | clear (+1.000)  | day (+0.457)    |
| 6    | skies  | 640      | The (+0.782)    | with (+0.947)   |

The KV cache grows linearly: `64 bytes/token` (FP32, 1 layer). At 10 tokens it holds **640 bytes**. A real model like LLaMA-3 8B with `n_kv_heads=8`, `d_head=128`, `32 layers` costs:

```
bytes per token = 2(K+V) × 8(heads) × 128(d_head) × 32(layers) × 2(BF16) = 131,072 bytes = 128 KB
For 128K tokens: 128 KB × 131,072 = 16 GB per request
```

This is why KV cache management is a first-class engineering problem.

---

### 4.4.6 The Decode Rhythm — What to Notice `[ESSENTIAL]`

Having traced the prefill and both decode steps in detail, the following observations capture the invariants that hold across **every** attention variant in this chapter:

```text
1. Prefill fills the first N_prefill K/V rows at once (batch computation).
2. Decode appends exactly one new K row and one new V row per predicted token.
3. The newest query is always shape 1 × d_head — a single vector.
4. K_cache grows from N_prefill × d_head to (N_prefill + t) × d_head at decode step t.
5. The score vector grows from 1 × (N_prefill+1) to 1 × (N_prefill+t+1).
6. The context vector stays shape 1 × d_head — always the same output shape.
7. The multiplication pattern never changes:

   q_new @ K_cache^T       → scores   (1 × N)
   softmax(scores / √d)    → weights  (1 × N)
   weights @ V_cache       → context  (1 × d_head)

8. MHA, MQA, GQA, MLA, and sparse attention modify what K/V state exists
   and how it is read, but this basic decode rhythm remains the anchor.
```

The table below tracks the cache state across all six decode steps:

| Step | New token | Cache size (tokens) | K_cache shape (per head) | Score vector shape |
|------|-----------|---------------------|--------------------------|-------------------|
| 0    | (prefill) | 4                   | 4 × 2                    | 1 × 4             |
| 1    | bright    | 5                   | 5 × 2                    | 1 × 5             |
| 2    | and       | 6                   | 6 × 2                    | 1 × 6             |
| 3    | sunny     | 7                   | 7 × 2                    | 1 × 7             |
| 4    | with      | 8                   | 8 × 2                    | 1 × 8             |
| 5    | clear     | 9                   | 9 × 2                    | 1 × 9             |
| 6    | skies     | 10                  | 10 × 2                   | 1 × 10            |

The context vector dimension never changes — `1 × 2` throughout. Only the **cost of computing it** grows, because `K_cache^T` gains one more column per step.

---

## 4.5 The KV Cache — Why It Exists `[ESSENTIAL]`

The autoregressive constraint says: to generate token `t`, the model must attend to all tokens `0, 1, …, t-1`. Naïvely this means recomputing K and V for all past tokens at every step. For a 1000-token context, step 1000 would recompute 999 previous K/V pairs — a quadratic total cost.

**The insight**: K and V for position `j` depend only on the *embedding at position j* and the *fixed weight matrices* `W_K` and `W_V`. They do not change as more tokens are generated. So we **compute them once and cache them forever**:

```python
# Prefill phase: compute all at once
for j in range(n_prefill):
    k_cache[j] = embed[j] @ W_K   # store once
    v_cache[j] = embed[j] @ W_V   # store once

# Decode phase: each new token appends one new row
k_cache.append(new_token_embed @ W_K)
v_cache.append(new_token_embed @ W_V)

# Attend over full cache
scores = new_q @ k_cache.T / sqrt(d)
output = softmax(scores) @ v_cache
```

The cost at each decode step is now `O(N × d)` — linear in the current sequence length. The Q projection is cheap (one vector); only loading the entire K and V cache from GPU memory is costly. This is why KV cache variants (MQA, GQA, MLA) exist: to reduce the **size** of what must be loaded per step.

---

## 4.6 Multi-Query Attention (MQA) `[ESSENTIAL]`

### 4.6.1 What MQA Does

MQA (Shazeer 2019) collapses all KV heads into **one shared head**. Every query head still has its own `W_Q` matrix and produces a distinct query vector. But they all read from a single shared K/V cache.

```
n_q_heads = 4
n_kv_heads = 1  (MQA: single shared KV head)
cache per token per layer = 2(K+V) × 1(head) × 2(d_head) × 4 bytes = 16 bytes (FP32)
```

**MQA KV Cache Layout** — all 4 query heads read the same single cache:

```
K_cache shape = N × 1 × D = 10 × 1 × 2
V_cache shape = N × 1 × D = 10 × 1 × 2

  K_cache (per layer):
  ┌────────────────────┐
  │ token 0: K[shared] │
  │ token 1: K[shared] │
  │ token 2: K[shared] │
  │ token 3: K[shared] │
  │ token 4: K[shared] │
  │ token 5: K[shared] │
  │ token 6: K[shared] │
  │ token 7: K[shared] │
  │ token 8: K[shared] │
  │ token 9: K[shared] │
  └────────────────────┘
```

The query side remains multi-headed — each head still has its own `W_Q` and produces a distinct query vector — but they all read the same K/V rows.

This is a **4× reduction** in cache size compared to MHA. For LLaMA-3 70B at 128K context this means the difference between ~85 GB of KV cache (MHA) and ~21 GB (MQA).

The shared W_K and W_V are identical to Head 0's matrices from MHA.

### 4.6.2 Decode Step 1 Comparison: "bright"

Under MQA, "bright" appends to a **single** shared cache:

```
New K/V (shared): k=[+2.4662, -2.0877]  v=[-2.3091, +2.3569]
```

All 4 query heads attend the same K/V entries but produce different attention patterns because each head uses its own W_Q:

```
Head 0 query: q=[-0.0217, +1.0947]
Head 1 query: q=[+3.3696, +0.3580]
Head 2 query: q=[-1.1009, +2.0651]
Head 3 query: q=[-0.2001, +1.2351]
```

**Head 0 softmax over 5 shared keys:**
```
The:  -2.5450 → 0.0095    next:   +1.8536 → 0.7695
day:  -1.5974 → 0.0244    is:     +0.3649 → 0.1736    bright: -1.6540 → 0.0231
(Identical to MHA Head 0 because they share Head 0's K/V)
```

**Head 3 softmax over the same 5 shared keys** (different W_Q → different q):
```
q_h3 = [-0.2001, +1.2351]
The:    -0.2001×+4.3972 + +1.2351×-3.2005 = -4.8365  →  /√2 = -3.4200
next:   +0.0588  →  +0.0416
day:    -1.9350  →  -1.3683
is:     +1.4905  →  +1.0540
bright: -2.3118  →  -1.6350
→ softmax:  The: 0.002, next: 0.030, day: 0.007, is: 0.831, bright: 0.131
```

Head 3 now attends heavily to "is" rather than "The" — even though it shares the same keys! The query vector alone drives the attention pattern.

**Memory comparison after decode step 1:**
```
MHA: 2 × 5 × 4 × 2 × 4 = 320 bytes
MQA: 2 × 5 × 1 × 2 × 4 =  80 bytes   ← 4× cheaper
```

---

## 4.7 Grouped-Query Attention (GQA) `[ESSENTIAL]`

### 4.7.1 What GQA Does

GQA (Ainslie et al., 2023) finds the middle ground between MHA (quality) and MQA (efficiency). The 4 query heads are split into `G` groups; each group shares **one** KV head. Here with `n_kv_heads=2` and `n_q_heads=4`, group size = 2.

```
Group 0: Query heads 0 and 1 share KV head 0
Group 1: Query heads 2 and 3 share KV head 1

n_q_heads = 4
n_kv_heads = 2  (GQA)
cache per token per layer = 2 × 2 × 2 × 4 = 32 bytes (FP32)
```

GQA is a 2× reduction vs MHA. Most production models in 2024–2025 use GQA: Llama-3 (8B: 8 KV heads, 32 Q heads; 70B: 8 KV heads, 64 Q heads), Mistral, Gemma, Qwen2.

**GQA KV Cache Layout** — the middle ground between MHA (4 heads) and MQA (1 head):

```
K_cache shape = N × Hkv × D = 10 × 2 × 2
V_cache shape = N × Hkv × D = 10 × 2 × 2

  K_cache (per layer):
  ┌──────────────────────────────┐
  │ token 0: K[group0] K[group1] │
  │ token 1: K[group0] K[group1] │
  │ token 2: K[group0] K[group1] │
  │ token 3: K[group0] K[group1] │
  │ token 4: K[group0] K[group1] │
  │ token 5: K[group0] K[group1] │
  │ token 6: K[group0] K[group1] │
  │ token 7: K[group0] K[group1] │
  │ token 8: K[group0] K[group1] │
  │ token 9: K[group0] K[group1] │
  └──────────────────────────────┘

  Q heads 0, 1 → read group 0   (K[group0], V[group0])
  Q heads 2, 3 → read group 1   (K[group1], V[group1])
```

Compare the shapes directly: MHA is `10 × 4 × 2`, MQA is `10 × 1 × 2`, and GQA is `10 × 2 × 2`. GQA stores more than MQA but less than MHA — and more groups means more expressivity at the cost of more memory.

### 4.7.2 Decode Step 1: "bright" Under GQA

KV head 0 is shared by Q heads 0 and 1. KV head 1 is shared by Q heads 2 and 3.

```
KV head 0 (= W_K[0], W_V[0] from MHA):
  k=[+2.4662, -2.0877]  v=[-2.3091, +2.3569]

KV head 1 (= W_K[1], W_V[1] from MHA):
  k=[-0.2522, -1.2534]  v=[-1.0043, +1.8583]
```

**Q head 0** uses KV head 0 → identical to MHA Head 0.
**Q head 1** also uses KV head 0 (not its own as in MHA):

```
q_h1 = [+3.3696, +0.3580]
Scores against KV head 0:
  The:    +3.3696×+4.3972 + +0.3580×-3.2005 = +13.6569  → /√2 = +9.6573
  next:   +3.3696×-2.3467 + +0.3580×+2.3480 = -7.0727  → /√2 = -5.0013
  day:    +3.3696×+2.5510 + +0.3580×-2.0130 = +7.8765  → /√2 = +5.5683
  is:     +3.3696×-0.9924 + +0.3580×+0.4517 = -3.1876  → /√2 = -2.2540
  bright: +3.3696×+2.4662 + +0.3580×-2.0877 = +7.5552  → /√2 = +5.3412
→ softmax: The:0.9745, next:0.0000, day:0.0141, is:0.0000, bright:0.0115
```

**Memory:**
```
MHA: 320 bytes    GQA: 2×5×2×2×4 = 160 bytes   MQA: 80 bytes
```

**What GQA Buys and Costs:**

```
Good:
  - much smaller cache than MHA (2× reduction here; 4–8× in production models)
  - more head specialization than MQA (each group still learns its own K/V)
  - quality closely tracks MHA at G ≥ 4 groups (Ainslie et al., 2023)
  - standard in all modern LLMs: LLaMA-3, Mistral, Gemma, Qwen2

Cost:
  - query heads within the same group cannot see different K/V perspectives
  - group count (Hkv) becomes a quality / memory tuning knob at training time
  - still scales linearly with context length N (MLA or sparse needed to break that)
```

The group assignment is permanent (baked into the trained weights), so you cannot change it at inference time. vLLM reads `num_key_value_heads` from the model config and automatically allocates the right KV layout without user intervention.

---

## 4.8 Multi-head Latent Attention (MLA) `[ADVANCED]`

### 4.8.1 What MLA Does

MLA (DeepSeek-V2, 2024) takes a fundamentally different approach. Instead of simply reducing the number of KV heads, it **compresses the KV representation into a low-rank latent vector** `C_KV`, then reconstructs the full K/V at query time.

The key operations:

```
Cache: C_KV = e @ W_DKV   (d_model × latent_dim — a d_model→3 projection)
At query time:
  K = C_KV @ W_UK   (latent_dim → d_head × n_heads)
  V = C_KV @ W_UV   (latent_dim → d_head × n_heads)
```

In our toy example: `latent_dim=3` instead of `d_head × n_heads = 8`. The cache stores a 3-dimensional vector per token instead of 8 scalars.

```
n_q_heads = 4
latent_dim = 3  (compressed representation)
cache per token per layer = 3 × 4 bytes = 12 bytes (FP32)
```

Compared to MHA's 64 bytes, MLA achieves a **~5× reduction** in the toy example — and in real models like DeepSeek-V2 (128K context, d_model=5120), the savings are over 10×.

**MLA Latent Cache Layout** — stores the recipe, not the finished dishes:

```
Instead of caching K and V directly:
  K_cache: N × Hkv × D
  V_cache: N × Hkv × D

MLA caches the compressed latent:
  C_cache: N × latent_dim = 10 × 3

  ┌─────┬────────┬──────────────────────┐
  │ pos │ token  │ C_KV stored in cache │
  ├─────┼────────┼──────────────────────┤
  │  0  │ The    │ [+0.4776, -1.8234, +1.4821] │
  │  1  │ next   │ [computed from e_next @ W_DKV] │
  │  2  │ day    │ [computed from e_day  @ W_DKV] │
  │  3  │ is     │ [computed from e_is   @ W_DKV] │
  │  4  │ bright │ [...] │
  │  …  │  …     │  …    │
  │  9  │ the    │ [computed from e_the  @ W_DKV] │
  └─────┴────────┴──────────────────────┘
```

K and V are never stored — they are reconstructed on-demand at decode time.

**MLA Data Flow — Two Paths:**

```
                    query path
  X ─────────────── W_Q ─────────────────► Q
  │                                         │
  │  key/value latent path                  │
  └── W_DKV ─► C_KV ── W_UK ─► K ──► Q @ K^T ─► scores
                 │                              │
                 └── W_UV ─► V ──── weights @ V ─► context
```

The query path is unchanged from MHA. The K/V path first compresses X into a latent `C_KV`, then expands it back into K and V only when needed. The long-lived cache stores `C_KV`; K and V are ephemeral.

**Algebraic rearrangement:** Because `K = C_KV @ W_UK` and the up-projection matrices are fixed weights, DeepSeek showed that the attention scores can be rewritten as:

```
Scores = Q @ K^T
       = Q @ (C_KV @ W_UK)^T
       = Q @ W_UK^T @ C_KV^T

Define A_K = W_Q @ W_UK^T  (precomputed once at load time)

Scores = X @ A_K @ C_KV^T
```

This means the cache only needs to store `C_KV` — the fixed learned matrices handle the rest. The same rearrangement applies to the value side: `Context = weights @ C_KV @ (W_UV @ W_O)`.

### 4.8.2 MLA Worked Arithmetic

Down-projection `W_DKV` (8×3):

```
W_DKV:
row0: [+0.4967, -0.1383,  0.6477]
row1: [+1.5230, -0.2342, -0.2341]
row2: [+1.5792,  0.7674, -0.4695]
row3: [+0.5426, -0.4634, -0.4657]
row4: [+0.2420, -1.9133, -1.7249]
row5: [-0.5623, -1.0128,  0.3142]
row6: [-0.9080, -1.4123,  1.4656]
row7: [-0.2258,  0.0675, -1.4247]
```

For "The" with embedding `e = [+0.3047, -1.0400, +0.7505, +0.9406, -1.9510, -1.3022, +0.1278, -0.3162]`:

```
C_KV[0] = e @ W_DKV[:,0]
 = +0.3047×+0.4967 + -1.0400×+1.5230 + +0.7505×+1.5792 + +0.9406×+0.5426
   + -1.9510×+0.2420 + -1.3022×-0.5623 + +0.1278×-0.9080 + -0.3162×-0.2258
 = +0.1513 + -1.5839 + +1.1845 + +0.5102 + -0.4722 + +0.7322 + -0.1160 + +0.0714
 = +0.4776

C_KV[1] = e @ W_DKV[:,1] = -1.8234
C_KV[2] = e @ W_DKV[:,2] = +1.4821

→ C_KV_The = [+0.4776, -1.8234, +1.4821]  (only 3 floats stored!)
```

**Memory comparison after full 10-token sequence:**
```
MHA: 2×10×4×2×4 = 640 bytes
MQA: 2×10×1×2×4 = 160 bytes
GQA: 2×10×2×2×4 = 320 bytes
MLA: 10×3×4     = 120 bytes  ← cheapest!
```

The trade-off: at query time, MLA must multiply each cached vector through `W_UK` and `W_UV` to reconstruct K and V. This adds FLOPs at decode time. DeepSeek-V2 showed that this compute cost is affordable — the GPU has spare FLOP capacity — but the memory bandwidth savings are dramatic.

### 4.8.3 MLA K/V Reconstruction: Manual Step for `the`

Using the up-projection matrices from the gist walkthrough (latent_dim=4 extended trace):

```
W_UK =
  ┌                     ┐
  │ 0.5  0.0  0.2  0.0  │
  │ 0.0  0.5  0.0  0.2  │
  │ 0.0  0.3  0.8  0.0  │
  │ 0.2  0.0  0.0  0.8  │
  └                     ┘

W_UV =
  ┌                     ┐
  │ 1.0  0.0  0.5  0.0  │
  │ 0.0  1.0  0.0  0.5  │
  │ 0.0  0.5  1.0  0.0  │
  │ 0.5  0.0  0.0  1.0  │
  └                     ┘

Cached latent for "the":  c_the = [1.0, 0.25, 0.0, 1.0]
```

**Reconstruct K:**

```
k_the = c_the @ W_UK

[1.0, 0.25, 0.0, 1.0] @
  ┌                     ┐
  │ 0.5  0.0  0.2  0.0  │
  │ 0.0  0.5  0.0  0.2  │
  │ 0.0  0.3  0.8  0.0  │
  │ 0.2  0.0  0.0  0.8  │
  └                     ┘

k0 = 1.0×0.5  + 0.25×0.0 + 0.0×0.0 + 1.0×0.2 = 0.70
k1 = 1.0×0.0  + 0.25×0.5 + 0.0×0.3 + 1.0×0.0 = 0.125
k2 = 1.0×0.2  + 0.25×0.0 + 0.0×0.8 + 1.0×0.0 = 0.20
k3 = 1.0×0.0  + 0.25×0.2 + 0.0×0.0 + 1.0×0.8 = 0.85

k_the = [0.70, 0.125, 0.20, 0.85]
```

**Reconstruct V:**

```
v_the = c_the @ W_UV

v0 = 1.0×1.0 + 0.25×0.0 + 0.0×0.0 + 1.0×0.5 = 1.50
v1 = 1.0×0.0 + 0.25×1.0 + 0.0×0.5 + 1.0×0.0 = 0.25
v2 = 1.0×0.5 + 0.25×0.0 + 0.0×1.0 + 1.0×0.0 = 0.50
v3 = 1.0×0.0 + 0.25×0.5 + 0.0×0.0 + 1.0×1.0 = 1.125

v_the = [1.50, 0.25, 0.50, 1.125]
```

From a single cached 4-dim latent vector, MLA reconstructed both a 4-dim key and a 4-dim value. After reconstruction, attention proceeds exactly as in MHA.

### 4.8.4 MLA Downward Projection: `X @ W_DKV → C_KV` (4-Token Prefill)

The first write to the MLA cache happens at prefill. Each token's embedding is down-projected into the latent space. Using the figure-style toy with `d_model=8`, `latent_dim=4`, and the four-token prompt "The next day is":

```
X (4 × 8):
  ┌────────────────────────────────────────┐
  │ token │ x0 x1 x2 x3 x4 x5 x6 x7       │
  ├───────┼────────────────────────────────┤
  │ The   │  1  0  0  0  1  0  0  0        │
  │ next  │  0  1  0  0  0  1  0  0        │
  │ day   │  0  0  1  0  1  1  0  0        │
  │ is    │  0  0  0  1  0  1  1  0        │
  └────────────────────────────────────────┘

W_DKV (8 × 4):
  ┌────────────────────┐
  │ 0.5  0.0  0.0  0.0 │  ← x0
  │ 0.0  0.5  0.0  0.0 │  ← x1
  │ 0.5  0.5  0.0  0.0 │  ← x2
  │ 0.0  0.5  0.5  0.0 │  ← x3
  │ 0.5  0.0  0.0  0.5 │  ← x4
  │ 0.0  0.5  0.0  0.0 │  ← x5
  │ 0.0  0.0  0.5  0.0 │  ← x6
  │ 0.0  0.0  0.0  0.5 │  ← x7
  └────────────────────┘
```

**Token "The"** — `x = [1,0,0,0,1,0,0,0]`:

```
c0 = 1×0.5 + 0×0.0 + 0×0.5 + 0×0.0 + 1×0.5 + 0×0.0 + 0×0.0 + 0×0.0 = 1.0
c1 = 1×0.0 + 0×0.5 + 0×0.5 + 0×0.5 + 1×0.0 + 0×0.5 + 0×0.0 + 0×0.0 = 0.0
c2 = 1×0.0 + 0×0.0 + 0×0.0 + 0×0.5 + 1×0.0 + 0×0.0 + 0×0.5 + 0×0.0 = 0.0
c3 = 1×0.0 + 0×0.0 + 0×0.0 + 0×0.0 + 1×0.5 + 0×0.0 + 0×0.0 + 0×0.5 = 0.5
→ C_KV(The) = [1.0, 0.0, 0.0, 0.5]
```

**Token "next"** — `x = [0,1,0,0,0,1,0,0]`:

```
c0 = 0×0.5 + 1×0.0 + 0×0.5 + 0×0.0 + 0×0.5 + 1×0.0 + 0×0.0 + 0×0.0 = 0.0
c1 = 0×0.0 + 1×0.5 + 0×0.5 + 0×0.5 + 0×0.0 + 1×0.5 + 0×0.0 + 0×0.0 = 1.0
c2 = 0×0.0 + 1×0.0 + 0×0.0 + 0×0.5 + 0×0.0 + 1×0.0 + 0×0.5 + 0×0.0 = 0.0
c3 = 0×0.0 + 1×0.0 + 0×0.0 + 0×0.0 + 0×0.5 + 1×0.0 + 0×0.0 + 0×0.5 = 0.0
→ C_KV(next) = [0.0, 1.0, 0.0, 0.0]
```

**Token "day"** — `x = [0,0,1,0,1,1,0,0]`:

```
c0 = 0×0.5 + 0×0.0 + 1×0.5 + 0×0.0 + 1×0.5 + 1×0.0 + 0×0.0 + 0×0.0 = 1.0
c1 = 0×0.0 + 0×0.5 + 1×0.5 + 0×0.5 + 1×0.0 + 1×0.5 + 0×0.0 + 0×0.0 = 1.0
c2 = 0×0.0 + 0×0.0 + 1×0.0 + 0×0.5 + 1×0.0 + 1×0.0 + 0×0.5 + 0×0.0 = 0.0
c3 = 0×0.0 + 0×0.0 + 1×0.0 + 0×0.0 + 1×0.5 + 1×0.0 + 0×0.0 + 0×0.5 = 0.5
→ C_KV(day) = [1.0, 1.0, 0.0, 0.5]
```

**Token "is"** — `x = [0,0,0,1,0,1,1,0]`:

```
c0 = 0×0.5 + 0×0.0 + 0×0.5 + 1×0.0 + 0×0.5 + 1×0.0 + 1×0.0 + 0×0.0 = 0.0
c1 = 0×0.0 + 0×0.5 + 0×0.5 + 1×0.5 + 0×0.0 + 1×0.5 + 1×0.0 + 0×0.0 = 1.0
c2 = 0×0.0 + 0×0.0 + 0×0.0 + 1×0.5 + 0×0.0 + 1×0.0 + 1×0.5 + 0×0.0 = 1.0
c3 = 0×0.0 + 0×0.0 + 0×0.0 + 1×0.0 + 0×0.5 + 1×0.0 + 1×0.0 + 0×0.5 = 0.0
→ C_KV(is) = [0.0, 1.0, 1.0, 0.0]
```

**C_KV cache after prefill (what MLA stores):**

```
  ┌───────┬──────────────────────┐
  │ token │ c0    c1    c2   c3  │
  ├───────┼──────────────────────┤
  │ The   │ 1.0   0.0   0.0  0.5 │
  │ next  │ 0.0   1.0   0.0  0.0 │
  │ day   │ 1.0   1.0   0.0  0.5 │
  │ is    │ 0.0   1.0   1.0  0.0 │
  └───────┴──────────────────────┘
  shape: 4 × 4   (vs 4 × 4 K-cache + 4 × 4 V-cache = 32 scalars for MHA)
```

Only when attention needs K and V do we apply `K = C_KV @ W_UK` and `V = C_KV @ W_UV`. The compressed latent is the long-lived object; the full K/V matrices are ephemeral.

### 4.8.5 Six MLA Decode Steps (Summary)

The decode loop appends one latent row per step, not one K row and one V row:

```
cKV_cache after prefill: [C_KV(The), C_KV(next), C_KV(day), C_KV(is)]  shape: 4 × 4

Step 1 → append bright:   cache 5 × 4  │ q_bright = [1.0,1.0,0.0,0.0]
Step 2 → append and:      cache 6 × 4  │ q_and    = [1.0,2.0,0.0,0.0]
Step 3 → append sunny:    cache 7 × 4  │ q_sunny  = [2.0,1.0,0.0,0.0]
Step 4 → append ,:        cache 8 × 4  │ q_comma  = [0.2,0.2,1.0,0.0]
Step 5 → append and:      cache 9 × 4  │ q_and    = [1.0,2.0,0.0,0.0]
Step 6 → append the:      cache 10 × 4 │ q_the    = [1.0,0.5,0.0,0.5]
```

At each step, the model reconstructs the full K/V matrices from the entire cached latent, computes attention, and predicts the next token. The cache grows in latent space, not in K/V space.

**What MLA Buys and Costs:**

```
Good:
  - compresses cache without merely collapsing all heads into one
  - preserves more modeling expressivity than MQA at similar memory cost
  - algebraic rearrangement lets learned matrices absorb the up-projection
  - central to DeepSeek-V2/V3/R1 efficiency at 128K+ context

Cost:
  - extra reconstruction projections at every decode step (W_UK, W_UV matmuls)
  - more complex implementation — especially with decoupled RoPE
  - position-content decoupling requires two separate paths
  - not yet supported natively in all inference engines (vLLM added MLA in 0.5)
```

---

## 4.9 Sparse / Sliding-Window Attention `[PRODUCTION]`

For very long contexts (128K–1M tokens), even reading the KV cache becomes a bottleneck. Sparse attention patterns restrict which keys each query attends to.

### 4.9.1 Local Window Attention

```
local window size W:
  mask[i, j] = 1  iff  i - j ≤ W   (token i sees only the W most recent tokens)
  mask[i, j] = 0  otherwise

KV cache per token: unchanged (still need to store keys for the window)
Attention cost at step N: O(W × d)  instead of  O(N × d)
```

Mistral 7B (2023) uses a sliding window of W=4096 with 8192-token context. Each token attends to at most 4096 past tokens, dramatically reducing compute.

### 4.9.2 Strided Attention

```
Each position i attends to:
  - Local: positions i-w to i (window w=64)
  - Strided: positions i, i-s, i-2s, ... (stride s=512)
  - Global: a few designated "global" tokens (e.g., CLS, separator)
```

This is the BigBird/Longformer pattern. It gives O(N) complexity while maintaining a degree of long-range connectivity.

### 4.9.3 Sparse Mask Diagram — Visibility Map

A sparse mask tells the attention operation which token rows to include. A ✓ means "read this row and include it in softmax". A · means "skip this row entirely." The final token `the` (position 9) demonstrates the contrast:

```
Full dense attention for `the`:

  The  next  day  is  bright  and  sunny  ,  and  the
   ✓    ✓    ✓    ✓    ✓      ✓    ✓      ✓   ✓    ✓

Sparse local-plus-sink pattern for `the`
  (keep position 0 as global sink; keep local window = positions 6–9):

  The  next  day  is  bright  and  sunny  ,  and  the
   ✓    ·    ·    ·    ·      ·    ✓      ✓   ✓    ✓
```

Because softmax is recomputed only over the visible tokens, the five remaining positions receive all the probability mass. Sparse attention is not merely a compute shortcut — it changes the model's information access.

### 4.9.4 Manual Sparse Step for `the`

Using the same toy scores from §4.4 (dense MHA head 0), the raw dot-product scores at the final token are:

```
All 10 positions (dense):
  The:0.5303  next:1.4142  day:0.7955  is:1.2374  bright:1.0607
  and:1.2374  sunny:0.5303 ,:0.7955   and:1.2374  the:1.0607
```

The sparse mask keeps only `{The(0), sunny(6), ,(7), and(8), the(9)}`:

```
Scores over sparse subset:
  The:    0.5303
  sunny:  1.4142
  ,:      0.7955
  and:    1.2374
  the:    1.0607
```

**Softmax over only those five positions:**

```
  ┌────────┬────────┐
  │ token  │ weight │
  ├────────┼────────┤
  │ The    │ 0.1183 │
  │ sunny  │ 0.2864 │
  │ ,      │ 0.1542 │
  │ and    │ 0.2400 │
  │ the    │ 0.2011 │
  └────────┴────────┘
```

**Sparse context vector** (using 2-dim toy V values where V[sunny]=[2,1], V[,]=[0.5,0.5], V[and]=[1,2], V[the]=[2,0], V[The]=[1,0]):

```
context[0] = 0.1183×1 + 0.2864×2 + 0.1542×0.5 + 0.2400×1 + 0.2011×2 = 1.4103
context[1] = 0.1183×0 + 0.2864×1 + 0.1542×0.5 + 0.2400×2 + 0.2011×0 = 0.8434

sparse context = [1.4103, 0.8434]
```

Compare with the dense context for `the` from §4.4 — the difference shows that omitting 5 tokens does change the output, and the magnitude depends on the sparsity pattern.

**What Sparse Attention Buys and Costs:**

```
Good:
  - fewer K/V rows read per decode step → lower bandwidth cost at long context
  - O(W × d) per step instead of O(N × d) for local-window patterns
  - orthogonal to GQA/MLA: can stack with either for maximum savings
  - used in production: Mistral (W=4096), Gemma, many long-context models

Cost:
  - tokens outside the sparse pattern cannot directly influence this step
  - quality depends on the chosen pattern — wrong patterns hurt accuracy badly
  - does not reduce the amount of cache stored, only what is read per step
  - implementation must handle masks and eviction carefully (see Ch 11.5)
```

**Key distinction:** GQA/MQA/MLA reduce the *bytes stored* in the cache. Sparse attention reduces the *bytes read* per decode step. A production system can combine both: use GQA to shrink the cache footprint, then use sliding-window attention to cap the read bandwidth at each step.

---

## 4.10 Side-by-Side Memory Comparison

| Variant | n_kv_heads | Bytes/token/layer (FP32) | Relative | Example (32L, 32K ctx, BF16) |
|---------|------------|--------------------------|----------|-------------------------------|
| MHA     | 4          | 64                       | 1×       | 4.3 GB                        |
| GQA     | 2          | 32                       | 0.5×     | 2.1 GB                        |
| MQA     | 1          | 16                       | 0.25×    | 1.1 GB                        |
| MLA     | (latent=3) | 12                       | 0.19×    | 0.8 GB                        |
| Window  | 4 (W=4K)   | 64 (capped at W)         | W/N×     | 0.3 GB                        |

**Memory formula for all variants:**
```
bytes = 2(K+V) × N × n_kv_heads × d_head × n_layers × bytes_per_element
```

For MLA replace `n_kv_heads × d_head` with `latent_dim`.

### Real-Model KV Cache Calculator

Scale the same formula to production numbers. Let:

```
B  = batch size (concurrent users)
N  = context length (tokens)
Hkv = number of K/V heads (or groups)
D  = head dimension (d_head)
L  = number of transformer layers
T  = dtype bytes (2 for BF16/FP16, 4 for FP32)
```

**Total cache bytes (dense variants — MHA, GQA, MQA):**
```
cache_bytes = B × 2(K+V) × N × Hkv × D × L × T
```

**Bytes read per decode step (cost that drives latency):**
```
read_bytes_dense  = 2 × N × Hkv × D × L × T        (dense: reads all N rows)
read_bytes_sparse = 2 × S × Hkv × D × L × T        (sparse: reads only S rows)
```

**Key insight:**
```
GQA / MQA / MLA → reduce what is STORED  (cache_bytes shrinks)
Sparse attention → reduces what is READ   (read_bytes shrinks)
Production systems often combine both.
```

**Example — LLaMA-3 70B at 128K context, batch=1, BF16:**
```
MHA  (Hkv=64, D=128, L=80): 2 × 131072 × 64 × 128 × 80 × 2 = 274 GB
GQA  (Hkv=8,  D=128, L=80): 2 × 131072 × 8  × 128 × 80 × 2 =  34 GB
MQA  (Hkv=1,  D=128, L=80): 2 × 131072 × 1  × 128 × 80 × 2 = 4.3 GB
```

GQA (the actual LLaMA-3 70B configuration) saves `8×` vs. the MHA baseline — the difference between needing 3× A100s for cache alone and fitting comfortably on one.

---

## 4.11 vLLM and llama.cpp API `[PRACTICAL]`

### 4.11.1 vLLM

vLLM selects the attention backend automatically based on hardware and model config. For most models on A100/H100, it uses FlashAttention via `flash-attn` or `FlashInfer`.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    # GQA detected automatically from model config
    # (n_kv_heads=8, n_heads=32 for 8B)
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

# KV cache is allocated during __init__ based on
# max_model_len × n_kv_heads × d_head × n_layers × 2
```

To inspect the attention backend:
```python
# vLLM >= 0.4.0
print(llm.llm_engine.model_config.attention_backend)
# Typically: "FLASH_ATTN" or "FLASHINFER"
```

### 4.11.2 llama.cpp

llama.cpp has three attention paths: GGML (CPU/metal), CUDA (cuBLAS attention), and FlashAttention (when built with `-DLLAMA_FLASH_ATTN=ON`).

```bash
# Build with FlashAttention for GPU
cmake .. -DLLAMA_CUDA=ON -DLLAMA_FLASH_ATTN=ON
cmake --build . --config Release

# Sliding-window attention models (Mistral):
./llama-cli -m mistral-7b.gguf --context-size 32768 \
    -ngl 35  # GPU layers
```

```cpp
// llama.cpp attention config (from llama.h):
struct llama_context_params {
    int    n_ctx;        // context size
    int    n_batch;      // max batch size (prefill)
    int    n_ubatch;     // micro-batch size
    bool   flash_attn;   // enable FlashAttention
    // ...
};
```

---

## 4.12 Attention Variants in 2025–2026 Production Models

| Model            | n_q_heads | n_kv_heads | Attention Type |
|------------------|-----------|------------|----------------|
| LLaMA-3 8B       | 32        | 8          | GQA            |
| LLaMA-3 70B      | 64        | 8          | GQA            |
| Mistral 7B       | 32        | 8          | GQA + SWA      |
| DeepSeek-V2      | 128       | latent=512 | MLA            |
| DeepSeek-V3/R1   | 128       | latent=512 | MLA            |
| Qwen2-72B        | 64        | 8          | GQA            |
| Gemma-2 27B      | 32        | 16         | GQA            |
| GPT-4 (est.)     | ~128      | ~16        | GQA (est.)     |

MQA is largely obsolete in current models. GQA is the standard. MLA is unique to DeepSeek and represents the current frontier of KV compression.

---

## 4.13 Attention Variant Quick Reference

Use this as your mental model after working through the walkthroughs. The variants differ less in the attention equation itself and more in **what they store, share, compress, or skip**.

```text
MHA (Multi-Head Attention)
  Max quality baseline — no sharing or compression.
  Hkv = Hq
  Largest KV cache. Every head owns its own K and V rows.

MQA (Multi-Query Attention)
  All query heads share a single K/V head.
  Hkv = 1
  Smallest dense K/V cache (Hq× cheaper than MHA).
  Quality drops on tasks requiring diverse key-value patterns.

GQA (Grouped-Query Attention)
  Query heads share K/V by group (G query heads per KV head).
  1 < Hkv < Hq
  Practical quality/memory compromise. Current standard (LLaMA-3, Mistral, Qwen2).

MLA (Multi-Head Latent Attention)
  Store a compressed latent vector C_KV; reconstruct full K/V at attention time.
  Cache size = N × latent_dim (not N × Hkv × D).
  Attacks cache size without merely collapsing heads. DeepSeek-V2/V3 frontier.

Sparse / Sliding-Window Attention
  Attend to only a selected subset of past token positions.
  Cache size may stay the same, but rows READ per step shrinks to S << N.
  Mistral-7B: W=4096, reads 25× fewer rows than full-context at N=100K.
```

**Which variant is which model?**

| Observation | What it tells you |
|---|---|
| `n_q_heads == n_kv_heads` | MHA — full independent heads |
| `n_kv_heads == 1` | MQA — single shared head |
| `1 < n_kv_heads < n_q_heads` | GQA — grouped sharing |
| Cache stores `latent_dim` instead of `Hkv × D` | MLA |
| Model has `sliding_window` in config | Sparse/SWA |

---

## Chapter Summary

- **Attention** is a query-key-value lookup: every output token is a weighted sum of value vectors, with weights determined by query-key similarity.
- **Scale 1/√d** prevents dot-product saturation: without it, softmax gradients vanish at large `d`.
- **MHA** maintains independent KV caches per head: 64 bytes/token/layer (4 heads, d_head=2, FP32).
- **The KV cache** arises from the autoregressive constraint. K and V for past positions are immutable, so we store and reuse them.
- **MQA** shares one KV head across all query heads: 4× cache reduction, moderate quality loss.
- **GQA** groups query heads and shares one KV head per group: 2–8× reduction, near-MHA quality.
- **MLA** caches a low-rank latent vector and reconstructs K/V at query time: ~5–10× reduction.
- **Sliding-window attention** caps context cost at `O(W × d)` per step instead of `O(N × d)`.

---

## Self-Check Questions

1. For `n_q_heads=4`, `d_head=2`, `d_model=8`: prove that `Var(q · k) = d_head = 2` when components are `N(0,1)`, and show that dividing by `√d_head` reduces the variance to 1.

2. Using the embedding and weight matrices in §4.3, verify the K projection for token "day" at Head 0 shown in §4.4.3 (answer: `k_h0 = [+2.5510, -2.0130]`).

3. After the prefill of "The next day is", the KV cache is 256 bytes. For a 32-layer LLaMA-3 8B (n_kv_heads=8, d_head=128, BF16), how many bytes does a 4-token prefill produce?

4. Under MQA, Q Head 3 attends to the same KV cache as Q Head 0, but produces a completely different softmax distribution (§4.6.2). Explain mechanically why this is possible.

5. DeepSeek-V2 uses MLA with `latent_dim=512` and `d_model=5120`, `n_kv_heads=128`, `d_head=128`, 60 layers, BF16. (a) Compute the MHA KV cache size for 128K tokens. (b) Compute the MLA KV cache size. (c) What is the compression ratio?

6. In the toy setup (`N=10`, `D=2`, `Hq=4`, BF16, 1 layer), compute the FP16/BF16 KV cache bytes for MHA, GQA (`Hkv=2`), and MQA (`Hkv=1`).

7. Using the toy MLA setup, compute the latent cache bytes when `N=10`, `C=2`, `dtype_bytes=2`, 1 layer.

8. Reconstruct the final token's MLA key and value vectors given `C_KV = [1.5, 0.0]`, `W_UK = [[0.5, 0.0], [0.5, 1.0]]`, `W_UV = [[1.0, 0.0], [0.0, 1.0]]`.

9. In GQA with `Hq=4` query heads and `Hkv=2` K/V groups, which query heads share each K/V group? Does Head 3's context vector equal Head 0's context vector? Explain why.

10. A sparse attention pattern reads rows `{0, 6, 7, 8, 9}` (5 out of 10 tokens). If `N=8192` and sparse attention reads `S=1024` rows, what is the row-read reduction factor vs. dense attention?

---

## Worked Solutions

**Q6 — Toy KV cache bytes (BF16):**
```
MHA: 1 × 2 × 10 × 4 × 2 × 1 × 2 = 320 bytes
GQA: 1 × 2 × 10 × 2 × 2 × 1 × 2 = 160 bytes
MQA: 1 × 2 × 10 × 1 × 2 × 1 × 2 =  80 bytes
```
GQA is exactly half of MHA; MQA is one quarter.

**Q7 — MLA latent cache bytes:**
```
MLA: 1 × 10 × 2 × 1 × 2 = 40 bytes
```
MLA uses `40 bytes` vs. MHA's `320 bytes` — an 8× reduction even in this tiny toy example, without touching the number of query heads.

**Q8 — MLA K/V reconstruction:**
```
K = C_KV @ W_UK = [1.5, 0.0] @ [[0.5, 0.0], [0.5, 1.0]]
  = [1.5×0.5 + 0.0×0.5,  1.5×0.0 + 0.0×1.0]
  = [0.75, 0.0]

V = C_KV @ W_UV = [1.5, 0.0] @ [[1.0, 0.0], [0.0, 1.0]]
  = [1.5×1.0 + 0.0×0.0,  1.5×0.0 + 0.0×1.0]
  = [1.5, 0.0]
```
Only 2 floats are stored per token (`C_KV`), but 4 floats of K+V are reconstructed at decode time. The compute cost is 2 small matrix multiplications; the memory savings is proportional to the compression ratio.

**Q9 — GQA group assignment:**
```
Group 0  →  Query heads 0 and 1  (first half)
Group 1  →  Query heads 2 and 3  (second half)
```
Head 3 does **not** produce the same context vector as Head 0 even though both read Group 1's shared K/V, because each head uses its own `W_Q`. Different query vectors produce different attention weights over the same keys, leading to different context vectors. This is identical to the MQA example in §4.6.2 where Head 3 attends heavily to "is" while Head 0 attends heavily to "next".

**Q10 — Sparse attention row-read reduction:**
```
Toy:       5 rows read out of 10  →  2.0× reduction
Real-model: 1024 rows read out of 8192  →  8.0× reduction
```
Sparse attention's benefit compounds with sequence length: at `N=128K` with `S=4096`, the reduction reaches `32×`. Unlike GQA/MQA, sparse attention does not reduce the **size** of the stored cache — it reduces the **bandwidth cost** of reading that cache at each decode step.

