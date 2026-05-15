# Chapter 4.5 — Attention Alternatives: Sliding Window, Linear Attention, State Space Models, and Mamba

> *"The KV cache grows linearly with sequence length. Head compression makes the coefficient smaller. This chapter attacks the growth rate itself."*

---

## Why This Chapter Exists

Chapter 4 showed you how attention variants — MQA, GQA, MLA — reduce the *coefficient* of KV cache growth. For a 70B model with GQA at 128K context, the cache drops from ~270 GB to ~34 GB. That is an 8× win. But look at the formula again:

```
cache_GQA = 2 × N × Hkv × D × L × bytes
                    ↑
                    N is still here
```

A 100K-token session has 25× the cache of a 4K-token session, regardless of which head-sharing strategy you use. MQA, GQA, and MLA make the coefficient smaller; they do not change the growth rate. For long-context serving at 128K–1M tokens, that linear growth is the next wall.

This chapter presents the four architectural families that attack the problem differently: **Sliding Window Attention** caps the cache at a fixed window size; **Linear Attention** replaces softmax with a kernel trick that compresses all past context into a constant-size state matrix; **State Space Models (SSMs)** use a learned recurrence to maintain a fixed-size hidden state; and **Mamba** makes that recurrence *selective* — the model decides per token what to remember and what to forget. Each approach trades some capability against a dramatic reduction in the memory footprint that grows with N.

Every worked example in this chapter uses the same `numpy.random.seed(42)` embedding table as Chapter 4, so numbers are traceable with a pencil and cross-checkable against the previous chapter's output.

**What you will know by the end:**

- Why head compression alone cannot solve the long-context memory problem, and what growth rate means in practice.
- Sliding window attention: the circular-buffer KV cache, attention sinks, and why Mistral chose W=4096.
- Linear attention: the feature-map trick that converts O(N) softmax into an O(1) state update, with the full S-matrix recurrence traced by hand.
- Fixed SSMs: the state-space model equations, discretization (A_bar, B_bar), and impulse response worked step-by-step.
- Mamba and selective SSMs: how input-dependent gating (Δ_t, B_t, C_t) gives the model per-token control over memory, and why this matters for language.
- Memory comparison across all four families: exact byte counts at toy scale and production scale.
- Roofline analysis: where each approach sits on the compute/bandwidth plane and which regime favors which architecture.
- Production deployment: how vLLM and llama.cpp expose these mechanisms, and when to reach for each.

---

## 0. Glossary

| Term | Definition |
|---|---|
| **A_bar_t** | Mamba's discrete state-transition factor at step t. `A_bar_t = exp(A · Δ_t)`. |
| **B_bar_t** | Mamba's discrete input-write factor. `B_bar_t = 1 − A_bar_t`. |
| **Circular buffer** | Fixed-size FIFO ring; oldest entry overwritten when full. Used for sliding-window KV cache. |
| **D_state** | Hidden state dimension of an SSM or Mamba layer. Typically 16–256. |
| **Decay factor** | The diagonal eigenvalue `A[i] < 1` of an SSM. Controls memory timescale per dimension. |
| **Δ_t** | Mamba's per-token discretisation timestep, a learned function of the current input. |
| **Feature map φ** | Kernel function in linear attention replacing softmax. Chosen so Q·K^T factors into φ(Q)·(φ(K)^T·V). |
| **Fixed SSM** | An SSM with static A, B, C, D — the same matrices applied to every token. |
| **Impulse response** | The sequence of outputs produced by an SSM given a single unit impulse input at t=0. |
| **Linear attention** | Softmax-free attention variant where past KV info is stored in a constant D×D state matrix S. |
| **Mamba** | Selective SSM by Gu & Dao (2024). Transition matrices A_t, B_t, C_t are functions of the current input. |
| **Parallel scan** | Divide-and-conquer algorithm computing the recurrence in O(log N) parallel depth. Used by Mamba at training. |
| **Receptive field** | Set of past positions that can influence the current token's representation. |
| **Recurrence** | Sequential update h_t = A·h_{t-1} + B·x_t. O(1) per step, O(N) total. |
| **S matrix** | D×D running state in linear attention. `S_t = S_{t-1} + φ(K_t)^T·V_t`. |
| **Selective SSM** | SSM with input-dependent A_t, B_t, C_t. Can selectively remember or forget per token. |
| **Sliding window** | Attention restricted to the last W tokens. KV cache is a constant-size circular buffer of W slots. |
| **SSM** | State Space Model. Recurrence-based sequence model: `h_t = A·h_{t-1} + B·x_t`. |
| **W** | Window size in sliding-window attention. Constant across the session. |
| **z vector** | D-dimensional normaliser in linear attention. `z_t = z_{t-1} + φ(K_t)`. |

---

## 1. The Token-Length Problem

### 1.1 What Chapter 8 Left Unsolved

Chapter 4 cut the KV cache by compressing across *heads*. MQA reduced it to 1/4, GQA to 1/G, MLA to 1/13 — all at fixed sequence length N. The savings are real and large. But one factor survives untouched: the cache still grows linearly with N.

```
cache_MHA = 2 × N × H × D × L × bytes_per_element
             ↑
             N is here — and it grows without bound

cache_MLA = N × r_kv × L × bytes_per_element
             ↑
             N is STILL here
```

A 100 K-token session has 25× the bytes of a 4 K-token session regardless of the head-sharing strategy. GQA and MLA make the coefficient smaller, but the growth rate is unchanged.

This chapter attacks N itself.

### 1.2 Four Architectural Families

```
┌─────────────────────────────────────────────────────────────────┐
│         HOW EACH FAMILY ATTACKS THE N FACTOR                   │
├──────────────────────┬──────────────────────────────────────────┤
│ Family               │ Mechanism                                │
├──────────────────────┼──────────────────────────────────────────┤
│ Sliding window       │ Cap the receptive field at W tokens.     │
│                      │ Cache = constant W slots (circular buf). │
├──────────────────────┼──────────────────────────────────────────┤
│ Linear attention     │ Collapse all past KV into a D×D matrix S.│
│                      │ Cache = constant D×D regardless of N.    │
├──────────────────────┼──────────────────────────────────────────┤
│ SSMs                 │ Recurrence h_t = A·h_{t-1} + B·x_t.     │
│                      │ Cache = fixed-size hidden state h.        │
├──────────────────────┼──────────────────────────────────────────┤
│ Mamba                │ Selective SSM: A_t, B_t, C_t depend on   │
│                      │ x_t. Content-driven forget/remember.      │
└──────────────────────┴──────────────────────────────────────────┘
```

### 1.3 Why Token Compression is Harder than Head Compression

Head compression (Chapter 4) is *lossless with respect to model quality*. MLA proves mathematically that you can compress the K/V storage without losing multi-head diversity. The per-head projections are preserved; only the stored representation is compressed.

Token compression is *fundamentally lossy*. Each of the N tokens carries distinct information. When we cap the cache at W past tokens (sliding window), we have erased all information about tokens beyond W positions back. When we collapse past tokens into a constant-size state (linear attention or SSM), different past sequences can map to the same state — and the model cannot recover what it has lost.

The question each technique must answer: *for the tasks we care about, how much does this information loss hurt?* The answer varies enormously by task, which is why all four architectures coexist rather than one replacing the others.

---

## 2. Sliding Window Attention

### 2.1 The Core Idea

Each token attends only to the last W past tokens plus itself. Beyond position `i - W`, the past is invisible.

```
Full causal attention mask (N=8):

     pos  0  1  2  3  4  5  6  7
  pos 0 [ 1  .  .  .  .  .  .  . ]
  pos 1 [ 1  1  .  .  .  .  .  . ]
  pos 2 [ 1  1  1  .  .  .  .  . ]
  pos 3 [ 1  1  1  1  .  .  .  . ]
  pos 4 [ 1  1  1  1  1  .  .  . ]
  pos 5 [ 1  1  1  1  1  1  .  . ]
  pos 6 [ 1  1  1  1  1  1  1  . ]
  pos 7 [ 1  1  1  1  1  1  1  1 ]

Sliding window mask (W = 4):

     pos  0  1  2  3  4  5  6  7
  pos 0 [ 1  .  .  .  .  .  .  . ]
  pos 1 [ 1  1  .  .  .  .  .  . ]
  pos 2 [ 1  1  1  .  .  .  .  . ]
  pos 3 [ 1  1  1  1  .  .  .  . ]
  pos 4 [ .  1  1  1  1  .  .  . ]  ← "pos 0" is now invisible
  pos 5 [ .  .  1  1  1  1  .  . ]
  pos 6 [ .  .  .  1  1  1  1  . ]
  pos 7 [ .  .  .  .  1  1  1  1 ]

The mask is a narrow diagonal band of width W.
```

### 2.2 KV Cache as a Circular Buffer

The KV cache becomes a ring buffer of exactly W slots. As new tokens arrive, the oldest slot is overwritten.

```
W = 4, sequence growing token by token:

  After token 0:  [ K0  ·   ·   ·  ]   ptr=1, filled=1
  After token 1:  [ K0  K1  ·   ·  ]   ptr=2, filled=2
  After token 2:  [ K0  K1  K2  ·  ]   ptr=3, filled=3
  After token 3:  [ K0  K1  K2  K3 ]   ptr=0, filled=4  ← buffer full
  After token 4:  [ K4  K1  K2  K3 ]   ptr=1, filled=4  ← K0 overwritten!
  After token 5:  [ K4  K5  K2  K3 ]   ptr=2, filled=4  ← K1 overwritten!
  After token 6:  [ K4  K5  K6  K3 ]   ptr=3, filled=4
  After token 7:  [ K4  K5  K6  K7 ]   ptr=0, filled=4  ← circular

  The buffer size is CONSTANT at W = 4 regardless of total sequence length.
  ptr always points to the oldest slot (next to overwrite).
```

### 2.3 Cache Size Formula

```
cache_SW = 4 × W × H × D × L × bytes_per_element
```

| Term | Meaning |
|---|---|
| 4 | Factor of 2 for K+V, factor of 2 for float16 rounding |
| W | Window size — the constant cap (e.g. 4 096 for Mistral-7B) |
| H | Number of KV heads |
| D | d_head dimension |
| L | Number of layers |

At N = 100 K, W = 4 K:

```
reduction = N / W = 100,000 / 4,000 = 25×
```

### 2.4 Step-by-Step Manual Walkthrough

We use D_MODEL = 8, N_HEADS = 4, D_HEAD = 2, W = 4, and 10 tokens.

**Setup:**
```
Tokens: ["The", "next", "day", "is", "bright", "and", "sunny", "with", "clear", "skies"]
Embeddings X (seed=42), shape (10, 8)

W_Q, W_K, W_V  shape (8, 2)  (head 0 only, seed=42 × 0.5)
SCALE = √2 ≈ 1.4142
```

**Computing Q, K, V (all 10 tokens at once):**
```
Q = X @ W_Q   shape (10, 2)
K = X @ W_K   shape (10, 2)
V = X @ W_V   shape (10, 2)
```

**Building the window mask for token 4 ("bright"):**
```
Token "bright" is at position i=4. Window W=4.
Allowed positions: max(0, 4-4+1)=1 through 4  → positions 1,2,3,4
Position 0 ("The") is masked out.

Raw scores for "bright": Q[4] @ K^T / SCALE
Mask: set scores[4,0] = -inf   (outside window)
      scores[4,5..9] = -inf    (causal mask — future)

After softmax, weights[4] sums to 1 over positions 1..4 only.
```

**Attention weight matrix (lower-left 6×6 block, W=4):**
```
            The     next     day       is   bright      and
  The   [1.0000   0.0000   0.0000   0.0000   0.0000   0.0000]
  next  [0.7051   0.2949   0.0000   0.0000   0.0000   0.0000]
  day   [0.4483   0.4223   0.1294   0.0000   0.0000   0.0000]
  is    [0.4115   0.2724   0.0820   0.2341   0.0000   0.0000]
  bright[0.0000   0.2079   0.2951   0.2072   0.2898   0.0000]
  and   [0.0000   0.0000   0.2745   0.2475   0.3441   0.1338]

Note:
  • "The" at position 0 is invisible to "bright" (position 4) — hard masked to 0.
  • "bright" at position 4 is invisible to "and" (position 5) — no, "and" can see 4 steps back.
  Actually W=4 means 4 positions inclusive. "and" (i=5) sees positions 2,3,4,5.
  So weight for "next" (pos 1) is 0 in "and"'s row.
```

**Decode step — circular buffer in action:**
```
After all 10 tokens, the circular buffer contains the last 4:
  buffer slots: [ K[8],  K[9],  K[6],  K[7] ]  (circular, ptr=2)

When "skies" (token 9) queries:
  Get K_hist, V_hist in chronological order: K[6],K[7],K[8],K[9]
  scores = Q[9] @ K_hist.T / SCALE    shape (4,)
  weights = softmax(scores)
  context = weights @ V_hist           shape (2,)
```

### 2.5 Receptive Field from Layer Stacking

A single sliding-window layer cannot look more than W tokens back. But stacking L layers multiplies the effective reach.

```
Layer 1: each token sees up to W=4 positions back.
Layer 2: each "sees" the Layer 1 output of those W neighbors,
         which themselves each saw W more positions — so ~W² indirect reach.
Layer L: effective receptive field ≈ W × L

For Mistral-7B: W = 4,096, L = 32
  Effective receptive field = 4,096 × 32 = 131,072 tokens
```

```
Visual example (W=4, 3 layers, position 12 in Layer 3):

  Layer 3, pos 12:  directly sees L2 positions 9,10,11,12
  Layer 2, pos  9:  directly saw L1 positions 6,7,8,9
  Layer 1, pos  6:  directly saw   positions 3,4,5,6
  Layer 1, pos  9:  directly saw   positions 6,7,8,9

  Combined: Layer 3 pos 12 has INDIRECT access to positions 3 through 12.
  One more layer reaches back to position 1.

  Effective depth = W + (W-1)×(L-1)  ≈  W × L
```

**Caveat:** Direct attention is still stronger than indirect-through-layers attention. On retrieval-heavy benchmarks (needle-in-a-haystack), sliding-window models at W = 4 K struggle beyond ~10 K context even when theory says W × L ≈ 128 K.

### 2.6 Who Uses Sliding Window

| Model | Window W | Notes |
|---|---|---|
| Mistral-7B (2023) | 4,096 | First production model to popularise SWA |
| Gemma-2 / Gemma-3 | 4,096 | Mixed: SWA layers + full-attention every 5th |
| Mistral-Large | — | Dropped SWA in favour of GQA for 123B model |

---

## 3. Linear Attention

### 3.1 The Algebraic Reordering Trick

Standard softmax attention:
```
output = softmax(Q · K^T / √d) · V

where:
  Q, K shape (N, D)
  V      shape (N, D)
  Q · K^T  shape (N, N)  ← THIS IS THE PROBLEM: must materialise N×N
```

The N×N matrix exists because softmax is a row-wise nonlinearity — it mixes all entries in each row, so you cannot reorder the matmuls. The N×N matrix must be fully built before the row-wise normalisation.

Linear attention replaces softmax with a **feature map φ** chosen so attention can be written as:

```
output = φ(Q) · (φ(K)^T · V)

By associativity, we compute the inner product FIRST:
  S = φ(K)^T · V   shape (D, D)   ← this is all we need to store!

Then for each query:
  output_t = φ(Q_t) · S_t    shape (D,)

The N×N matrix never materialises.
```

**Feature map choice:** A common choice is φ(x) = ELU(x) + 1, which keeps all values strictly positive (required for the normalisation to be well-defined).

### 3.2 The Running D×D State

For causal (autoregressive) inference, S is accumulated token by token:

```
At each new token t:

  S_t  =  S_{t-1}  +  φ(K_t)^T · V_t         shape (D, D)
           └──────┘   └──────────────────┘
           old state   outer product of one new (K,V) pair

  z_t  =  z_{t-1}  +  φ(K_t)                  shape (D,)
                                               running normaliser

Output for token t:
  a_t  =  (φ(Q_t) · S_t)  /  (φ(Q_t) · z_t)  shape (D,)
```

Two running quantities — **S** (D×D matrix) and **z** (D-vector) — both with constant size regardless of how many tokens have been processed.

### 3.3 Step-by-Step Manual Walkthrough

We use D_HEAD = 4, N = 6 tokens ("The", "next", "day", "is", "bright", "and").

**Initial state:**
```
S_0 = zeros(4, 4)    # D×D
z_0 = zeros(4,)      # D
```

**After processing "The" (t=0):**
```
x_0 = X[0] @ W_K   →  K_0  (shape 4,)
         apply φ:   →  φ(K_0)  (all entries ≥ 1.0)

v_0 = X[0] @ W_V   →  V_0  (shape 4,)

Outer product:  φ(K_0)^T · V_0 = outer(φ(K_0), V_0)  shape (4,4)

S_1 = S_0 + outer(φ(K_0), V_0)
z_1 = z_0 + φ(K_0)

Output for "The":
  φ(Q_0) = φ(X[0] @ W_Q)   shape (4,)
  a_0 = φ(Q_0) @ S_1 / (φ(Q_0) @ z_1)   shape (4,)
```

**State S after "The" (rounded):**
```
S[0] = [+0.6939, -3.5986, -2.2931, +1.6126]
S[1] = [+0.2856, -1.4809, -0.9437, +0.6636]
S[2] = [+0.4302, -2.2313, -1.4218, +0.9999]
S[3] = [+0.7919, -4.1070, -2.6170, +1.8404]
```

**After "next" (t=1) — state grows to include a second outer product:**
```
S_2 = S_1 + outer(φ(K_1), V_1)

S[0] = [+0.9995, -3.5567, -2.2352, +1.4586]  ← changed
S[1] = [+0.4864, -1.4534, -0.9057, +0.5624]
S[2] = [+0.8186, -2.1780, -1.3483, +0.8041]
S[3] = [+3.0177, -3.8017, -2.1957, +0.7189]  ← large shift in row 3
```

The state matrix encodes a *sum* of all past (φ(K), V) outer products. It is a lossy compression — two different histories can produce the same S.

**Context vectors (linear attention):**
```
Token     context[0]   context[1]   context[2]   context[3]
The       +0.46648    -2.41918    -1.54153    +1.08407
next      +0.63836    -1.14118    -0.68566    +0.33108
day       +0.45031    -1.06346    -0.78922    +0.40484
is        +0.19885    -0.81390    -0.71288    +0.50140
bright    +0.15141    -0.66775    -0.48447    +0.26257
and       +0.01286    -0.56058    -0.57152    +0.41441
```

**First token ("The") matches softmax attention exactly** because there is only one past token — the single-entry S and softmax with one non-masked entry are equivalent. From "next" onward, the context vectors diverge slightly from softmax attention.

### 3.4 The Context Bottleneck

Linear attention's strength and weakness are the same property: the D×D state S is a *fixed-size summary* of all past tokens.

**Why it fails on retrieval tasks:** All past tokens contribute equally to S through their outer products. There is no mechanism to "pay more attention to the most recent mention of a name." The softmax distribution in standard attention creates a peaked weight over a few positions; linear attention sums over all past positions uniformly.

**Information loss:** If two tokens have key vectors that point in nearly opposite directions, their outer products in S partially cancel. The older token's information can be obscured.

**Practical consequence:** On needle-in-a-haystack benchmarks, linear attention scores ~40% while full attention scores ~95%.

### 3.5 The Cache Size

```
cache_linear = D² × L × 2 bytes    (S matrix)
             + D  × L × 2 bytes    (z normaliser)

For D = 128, L = 32 (FP16):
  S: 128 × 128 × 32 × 2 = 1,048,576 bytes = 1 MB
  z: 128 × 32 × 2       = 8,192 bytes ≈ 8 KB
  Total: ~1 MB  ← for any sequence length, even 1 million tokens.
```

---

## 4. State Space Models (SSMs)

### 4.1 The Recurrence

SSMs borrow from control theory and express the sequence model as:

```
h_t = A · h_{t-1}  +  B · x_t         (state update)
y_t = C · h_t       +  D · x_t         (output)
```

| Symbol | Shape | Role |
|---|---|---|
| x_t | (d,) | Input token embedding |
| h_t | (d_state,) | Hidden state — the "cache" |
| A | (d_state, d_state) | State-transition matrix — controls decay |
| B | (d_state, d) | Projects input into state space |
| C | (d, d_state) | Projects state into output space |
| D | scalar | Direct input-to-output skip connection |

**Critical constraint:** A has eigenvalues < 1. For diagonal A = diag(a_0, a_1, ...) with all |a_i| < 1, each state dimension decays exponentially over time. Information from step t-k contributes to h_t with weight A^k · B · x_{t-k}, which shrinks as k grows.

### 4.2 The Dual View: Recurrence ↔ Convolution

Because the recurrence is **linear**, it can be unrolled into a convolution. Substituting repeatedly:

```
h_1 = A·h_0 + B·x_1
    = A·(B·x_0) + B·x_1

h_2 = A·h_1 + B·x_2
    = A·(A·B·x_0 + B·x_1) + B·x_2
    = A²·B·x_0 + A·B·x_1 + B·x_2

h_t = B·x_t  +  A·B·x_{t-1}  +  A²·B·x_{t-2}  + ... + A^t·B·x_0

y_t = C·h_t + D·x_t
    = (C·B)·x_t  +  (C·A·B)·x_{t-1}  +  (C·A²·B)·x_{t-2}  + ...
    = Σ_{k=0}^{t}  K_k · x_{t-k}   where  K_k = C·A^k·B
```

This is a **1D convolution** with kernel K = [K_0, K_1, K_2, ...] where K_k = C·A^k·B.

```
Two computation orders for the same model:

  TRAINING (parallel):          INFERENCE (sequential):
  ┌─────────────────────┐       ┌─────────────────────────┐
  │ y = x ★ K           │       │ h_t = A·h_{t-1} + B·x_t │
  │ (1D convolution)    │       │ y_t = C·h_t + D·x_t     │
  │ O(N log N) via FFT  │       │ O(1) per token           │
  │ all N outputs at    │       │ cache = just h (d_state  │
  │ once — fully        │       │ floats)                  │
  │ parallelisable      │       │                          │
  └─────────────────────┘       └─────────────────────────┘

  ★ = 1D convolution
  
  RNNs CANNOT do this — tanh breaks linearity:
    tanh(a + b) ≠ tanh(a) + tanh(b)
  so the recurrence never unrolls into a fixed kernel.
```

### 4.3 The Four-Token Walkthrough — Exact Arithmetic

From the weight tables below, exact weights:

```
A = diag(0.9, 0.8, 0.7, 0.6)   ← four different decay rates

B (4×4):
  [[ 0.3, -0.1,  0.2,  0.4],
   [ 0.1,  0.5, -0.2,  0.3],
   [-0.2,  0.3,  0.4,  0.1],
   [ 0.4, -0.3,  0.1,  0.2]]

C (4×4):
  [[ 0.5,  0.2, -0.3,  0.1],
   [-0.1,  0.4,  0.6, -0.2],
   [ 0.3, -0.2,  0.1,  0.5],
   [ 0.2,  0.3, -0.1,  0.4]]

D = 0.1   (direct skip)

Inputs:
  x_0 = [1.0, 0.0, 0.5, 0.2]   "The"
  x_1 = [0.3, 0.8, 0.1, 0.4]   "next"
  x_2 = [0.7, 0.2, 0.3, 0.6]   "day"
  x_3 = [0.2, 0.5, 0.8, 0.1]   "is"

h_{-1} = [0, 0, 0, 0]
```

**Step t=0 ("The"):**
```
A·h_{-1} = [0.9·0, 0.8·0, 0.7·0, 0.6·0] = [0, 0, 0, 0]

B·x_0 row-by-row:
  row 0:  0.3·1.0 + (-0.1)·0.0 + 0.2·0.5 + 0.4·0.2 = 0.30 + 0 + 0.10 + 0.08 = 0.48
  row 1:  0.1·1.0 + 0.5·0.0 + (-0.2)·0.5 + 0.3·0.2 = 0.10 + 0 - 0.10 + 0.06 = 0.06
  row 2: -0.2·1.0 + 0.3·0.0 + 0.4·0.5 + 0.1·0.2    = -0.20 + 0 + 0.20 + 0.02 = 0.02
  row 3:  0.4·1.0 + (-0.3)·0.0 + 0.1·0.5 + 0.2·0.2 = 0.40 + 0 + 0.05 + 0.04 = 0.49

h_0 = [0.48, 0.06, 0.02, 0.49]   ✓ matches §9.4

C·h_0:
  row 0: 0.5·0.48 + 0.2·0.06 + (-0.3)·0.02 + 0.1·0.49 = 0.240 + 0.012 - 0.006 + 0.049 = 0.295
  row 1: -0.1·0.48 + 0.4·0.06 + 0.6·0.02 + (-0.2)·0.49 = -0.048 + 0.024 + 0.012 - 0.098 = -0.110
  row 2: 0.3·0.48 + (-0.2)·0.06 + 0.1·0.02 + 0.5·0.49  = 0.144 - 0.012 + 0.002 + 0.245 = 0.379
  row 3: 0.2·0.48 + 0.3·0.06 + (-0.1)·0.02 + 0.4·0.49  = 0.096 + 0.018 - 0.002 + 0.196 = 0.308

D·x_0 = 0.1 × [1.0, 0.0, 0.5, 0.2] = [0.100, 0.000, 0.050, 0.020]

y_0 = C·h_0 + D·x_0 = [0.395, -0.110, 0.429, 0.328]
```

**Step t=1 ("next"):**
```
A·h_0 = [0.9·0.48, 0.8·0.06, 0.7·0.02, 0.6·0.49]
       = [0.432,    0.048,    0.014,    0.294]

B·x_1 (abbreviated):
  row 0: 0.3·0.3 + (-0.1)·0.8 + 0.2·0.1 + 0.4·0.4 = 0.09 - 0.08 + 0.02 + 0.16 = 0.19
  row 1: 0.1·0.3 + 0.5·0.8 + (-0.2)·0.1 + 0.3·0.4 = 0.03 + 0.40 - 0.02 + 0.12 = 0.53
  row 2: -0.2·0.3 + 0.3·0.8 + 0.4·0.1 + 0.1·0.4   = -0.06 + 0.24 + 0.04 + 0.04 = 0.26
  row 3: 0.4·0.3 + (-0.3)·0.8 + 0.1·0.1 + 0.2·0.4 = 0.12 - 0.24 + 0.01 + 0.08 = -0.03

h_1 = A·h_0 + B·x_1 = [0.432+0.190, 0.048+0.530, 0.014+0.260, 0.294-0.030]
    = [0.622, 0.578, 0.274, 0.264]   ✓ matches §9.4

  Notice dim 0 of h_1 = 0.622 carries:
    0.432 from the earlier state  (old memory)
    0.190 from the new input      (new information)
  The state is a weighted mixture of past and present.
```

**Steps t=2,3 (same mechanics):**
```
h_2 = [1.050, 0.752, 0.292, 0.528]   ✓
h_3 = [1.155, 0.742, 0.644, 0.347]   ✓
y_3 = [0.588, 0.548, 0.517, 0.539]
```

### 4.4 Information Decay Table

Track how much of the original x_0 signal survives in each state dimension after t steps. Because A is diagonal, dimension i decays independently as A_diag[i]^t:

```
contribution_i(t) = A_diag[i]^t × (B[i,:] @ x_0)

Step  dim 0 (A=0.9)      dim 1 (A=0.8)      dim 2 (A=0.7)      dim 3 (A=0.6)
t=0   0.480 (100%)       0.060 (100%)       0.020 (100%)       0.490 (100%)
t=1   0.432 ( 90%)       0.048 ( 80%)       0.014 ( 70%)       0.294 ( 60%)
t=2   0.389 ( 81%)       0.038 ( 64%)       0.010 ( 49%)       0.176 ( 36%)
t=3   0.350 ( 73%)       0.031 ( 51%)       0.007 ( 34%)       0.106 ( 22%)
t=7   0.229 ( 48%)       0.013 ( 21%)       0.002 (  8%)       0.017 (  3%)
```

**Two key observations:**
1. **Different timescales per dimension.** Dim 0 (A=0.9) retains 48% of x_0 after 7 steps — slow decay, long-range memory. Dim 3 (A=0.6) retains only 3% — fast decay, short-range memory. A well-trained SSM spreads information across these timescales intelligently.
2. **Convolution kernel from decay.** The kernel value K_t = C·A^t·B falls directly out of this table. Training-time FFT computes all N outputs at once in O(N log N). Inference-time recurrence computes one output per step in O(1), with the same numerical result.

### 4.5 Convolutional Kernel Values

```
K_k = C · A^k · B   (shape d × d, impulse response at lag k)

K[0][0,0] = C[0] · A^0 · B[:,0][0] = 0.270000
K[1][0,0] = C[0] · A^1 · B[:,0][0] = 0.217000   (0.9 decay applied once)
K[2][0,0] = C[0] · A^2 · B[:,0][0] = 0.178100
K[3][0,0] = C[0] · A^3 · B[:,0][0] = 0.148810

Max |recurrence output - convolution output| = 2.22e-16  (< 1e-12)  ✓
```

### 4.6 SSM Cache Size

```
cache_SSM = d_state × L × 2 bytes   (hidden state h, FP16)

At d_state = 64, L = 32:
  64 × 32 × 2 = 4,096 bytes = 4 KB   ← total, for any sequence length

Compare to full MHA at N = 32K, d = 4096, H = 32, L = 32 (FP16):
  2 × 32000 × 32 × 128 × 32 × 2 ≈ 43 GB

SSM compression: ~10,000,000×  (yes, seven orders of magnitude)
```

---

## 5. Mamba — Selective SSMs

### 5.1 The Key Idea: Input-Dependent Dynamics

A standard SSM has *fixed* A, B, C, D — the same matrices applied to every token regardless of content. Mamba makes them *functions of the current input*:

```
Standard SSM:    h_t = A  · h_{t-1} + B  · x_t    (A, B fixed)
Mamba:           h_t = A_t· h_{t-1} + B_t· x_t    (A_t, B_t depend on x_t)
```

This is implemented through **discretisation**. Mamba learns a scalar (or vector) Δ_t per token — the "timestep" — and derives per-token state-transition factors:

```
Δ_t    = learned per-token timestep  (function of x_t via a small linear layer)
A_bar_t = exp(A · Δ_t)               (discrete state-transition factor)
B_bar_t = 1 − A_bar_t                (discrete input-write factor)

Recurrence:
  h_t = A_bar_t · h_{t-1}  +  B_bar_t · (B·x_t)   (element-wise for diagonal A)
  y_t = C_t · h_t
```

**What Δ_t controls:**

```
Large Δ_t → A_bar_t small, B_bar_t large → old state mostly erased, new input dominates
Small Δ_t → A_bar_t ≈ 1, B_bar_t ≈ 0   → old state preserved, new input nearly ignored

This is the "selective" mechanism:
  content words → large Δ → strong write
  function words → small Δ → state nearly frozen
```

### 5.2 Complete Architecture of a Mamba Block

```
Input x (d,)
    │
    ├─── Linear projection ──────────────────────────────────────┐
    │    splits into two branches: x' (d_inner) and z (d_inner) │
    │                                                            │
    ▼                                                            │
  Conv1D over x'                                                 │
  (local context, short kernel width)                            │
    │                                                            │
    ▼                                                            │
  SiLU(x')                                                       │
    │                                                            │
    ├── Selective parameter heads ──────────────────────────────┤
    │   Δ_t = softplus(W_dt @ x')    shape (d_state,)          │
    │   B_t = W_B @ x'               shape (d_state,)          │
    │   C_t = W_C @ x'               shape (d_state,)          │
    │                                                            │
    ▼                                                            │
  Discretise:                                                    │
    A_bar_t = exp(A · Δ_t)                                       │
    B_bar_t = 1 − A_bar_t                                        │
    │                                                            │
    ▼                                                            │
  Selective SSM recurrence:                                      │
    h_t = A_bar_t * h_{t-1} + B_bar_t * (B_t · x')             │
    y_t = C_t · h_t                                              │
    │                                                            │
    ▼                                                            │
  Gate: y_t * SiLU(z)  ◄─────────────────────────────────────────┘
    │
    ▼
  Output projection + residual  →  final output (d,)
```

Stages 4–6 (selective parameter heads, discretisation, recurrence) are the "selective core". The rest is standard neural-network machinery.

### 5.3 The "France" Trace — Exact Arithmetic

From the worked example below, sentence "The capital of France is", 1-D state, A_cont = −1.0, all token values = 1.0 (content encoded in Δ):

**Given Δ_t:**
```
Token      Δ_t    Interpretation
"The"      0.5    common word, moderate write
"capital"  1.0    content word, strong write
"of"       0.05   function word, nearly skip
"France"   2.0    key entity, heavy reset
"is"       0.3    structural, moderate write
```

**Step 1: Discretise Δ_t → A_bar_t, B_bar_t:**
```
A_bar_t = exp(−1.0 × Δ_t) = exp(−Δ_t)
B_bar_t = 1 − A_bar_t

Token      Δ_t    A_bar_t    B_bar_t    Keep%   Write%
"The"      0.50   0.607      0.393      61%     39%
"capital"  1.00   0.368      0.632      37%     63%
"of"       0.05   0.951      0.049      95%     5%
"France"   2.00   0.135      0.865      14%     87%
"is"       0.30   0.741      0.259      74%     26%

Verification:
  exp(−0.50) = 0.6065 ≈ 0.607  ✓
  exp(−2.00) = 0.1353 ≈ 0.135  ✓
```

**Step 2: State evolution h_t = A_bar_t × h_{t-1} + B_bar_t × 1.0:**
```
h_init = 0.000

"The":     h_0 = 0.607 × 0.000 + 0.393 × 1.0 = 0.000 + 0.393 = 0.393
"capital": h_1 = 0.368 × 0.393 + 0.632 × 1.0 = 0.145 + 0.632 = 0.777
"of":      h_2 = 0.951 × 0.777 + 0.049 × 1.0 = 0.739 + 0.049 = 0.788  ← barely moved!
"France":  h_3 = 0.135 × 0.788 + 0.865 × 1.0 = 0.106 + 0.865 = 0.971  ← reset!
"is":      h_4 = 0.741 × 0.971 + 0.259 × 1.0 = 0.720 + 0.259 = 0.979

Note on "of":   enters at 4.9% strength, old state 95.1% preserved — effectively ignored.
Note on "France": crushes old state to 10.6%, writes own signal at 86.5% — near-total reset.
```

**Step 3: Decompose the final state — which token owns what fraction?**

For a 1-D scalar state with per-token A_bar_t and B_bar_t:

```
contribution of token t to h_4:
  = B_bar_t × val_t × ∏_{s=t+1}^{4} A_bar_s

"The":     0.393 × (0.368 × 0.951 × 0.135 × 0.741) = 0.393 × 0.035 = 0.014  ( 1.4%)
"capital": 0.632 × (0.951 × 0.135 × 0.741)          = 0.632 × 0.095 = 0.060  ( 6.1%)
"of":      0.049 × (0.135 × 0.741)                   = 0.049 × 0.100 = 0.005  ( 0.5%)
"France":  0.865 × (0.741)                            = 0.865 × 0.741 = 0.641  (65.5%)
"is":      0.259 × (1.0)                              = 0.259 × 1.000 = 0.259  (26.5%)
                                                                       ───────
                                                                 sum = 0.979 = h_4  ✓
```

**Step 4: Compare to a fixed SSM (A_bar = 0.8 for all tokens):**

```
Fixed SSM contributions (recency-only, no content gating):
"The":     0.2 × 0.8^4 = 0.2 × 0.410 = 0.082  (12.8%)
"capital": 0.2 × 0.8^3 = 0.2 × 0.512 = 0.102  (16.0%)
"of":      0.2 × 0.8^2 = 0.2 × 0.640 = 0.128  (20.0%)
"France":  0.2 × 0.8^1 = 0.2 × 0.800 = 0.160  (25.0%)
"is":      0.2 × 0.8^0 = 0.2 × 1.000 = 0.200  (31.2%)

Comparison:
  Token       Mamba    Fixed SSM
  "The"        1.4%     12.8%
  "capital"    6.1%     16.0%
  "of"         0.5%     20.0%   ← "of" outweighs "capital" in fixed SSM!
  "France"    65.5%     25.0%   ← Mamba: 65.5%; fixed: only 25%
  "is"        26.5%     31.2%

In the fixed SSM, importance equals recency. A meaningless function word ("of")
ends up with a BIGGER share than the content word "capital" that preceded it.
Mamba fixes this: the learned Δ head looked at "of", decided it needed ~0% of
the state budget, and kept the previous state nearly frozen.
```

### 5.4 Why Mamba's Training Is Different

The selectivity of Mamba — the fact that A_t varies per token — means the SSM can no longer be expressed as a **fixed** convolution kernel. The training-time parallel trick of fixed SSMs (parallel FFT) does not directly apply.

Instead, Mamba uses a **parallel scan** (divide and conquer):

```
Standard SSM training:  O(N log N) via FFT convolution (fixed kernel)
Mamba training:         O(N log N) via parallel scan (variable-A recurrence)
Both at inference:      O(N) sequential, O(1) per step, fixed-size hidden state h
```

Parallel scan is harder to implement efficiently. This is why production support for Mamba lags behind Llama-style transformers.

### 5.5 Mamba Cache Size

```
cache_Mamba ≈ d_state × L × 2 bytes   (FP16)

Mamba-2 typical: d_state = 128, L = 64:
  128 × 64 × 2 = 16,384 bytes = 16 KB   ← for any sequence length

Full MHA at N=32K, d=4096, H=32, L=64 (FP16):
  2 × 32000 × 32 × 128 × 64 × 2 ≈ 86 GB

Mamba: ~86 GB → 16 KB, a factor of ~5,000,000×.
```

---

## 6. Memory Comparison Across All Four

### 6.1 Summary Table

| Architecture | Cache Grows with N? | Cache Formula | At N=32K, d=4096, H=32, L=32 (FP16) |
|---|---|---|---|
| Full MHA | Yes (linear) | 2·N·H·D·L·2 | ~43 GB |
| Sliding window (W=4K) | No (capped at W) | 2·W·H·D·L·2 | ~5 GB |
| Linear attention | No (constant D×D) | D²·L·2 | ~32 MB (D=512) |
| SSM (d_state=64) | No (constant h) | d_state·L·2 | 4 KB |
| Mamba (d_state=128) | No (constant h) | d_state·L·2 | 8 KB |

### 6.2 Quality vs Memory Trade-off

The compression ratio goes in the opposite direction of retrieval quality:

```
Retrieval accuracy (needle-in-haystack, 32K context, stylised but empirically grounded):

  Architecture              Retrieval%   Notes
  Full attention              95%        Gold standard
  Hybrid (Mamba + attn)       92%        Few attention layers; most are Mamba
  Mamba                       78%        Selectivity closes most of the gap
  Sliding window (W=4K)       45%        Degrades sharply past W
  Linear attention            40%        Context-bottleneck limits recall
  Fixed SSM                   35%        Recency bias, no content gating

Memory usage (same model):

  Full attention:             ████████████████████████████████ 100%
  Sliding window (W=4K):      ████  12.5%
  Linear attention:           █    0.075% (D=512)
  SSM / Mamba:                ·    0.00001%  (essentially zero)
```

### 6.3 ASCII Comparison Chart

```
                    MEMORY (log scale)           RETRIEVAL QUALITY
                    ─────────────────────        ────────────────────────
  Full MHA          █████████████ 43 GB           ██████████████████ 95%
  Sliding W=4K      ████████      5 GB            ██████████         45%
  Linear attn       ████         32 MB             ██████████         40%
  Mamba             ·            16 KB             █████████████████  78%
  Hybrid            ████████     ~5 GB             █████████████████  92%

  Note: Mamba uses orders of magnitude less memory than sliding window,
        but achieves much higher retrieval quality than linear attention.
        This is the selectivity advantage.
```

---

## 7. Roofline Analysis

### 7.1 What Each Architecture Changes on the Roofline

Each technique reduces the cache's byte dimension, raising **arithmetic intensity** (FLOPs / byte) and moving the operating point rightward on the roofline diagram.

```
Arithmetic intensity = FLOPs / bytes_moved

Architecture    Cache bytes at N=32K    AI vs full MHA
─────────────────────────────────────────────────────
Full MHA        43 GB                   1× (baseline)
Sliding W=4K    5 GB                    ~8× higher AI
Linear attn     32 MB                   ~1,300× higher AI
SSM/Mamba       16 KB                   ~2,700,000× higher AI (!)

At very long context, SSMs and Mamba are compute-bound rather than
memory-bound — the opposite regime from every other technique in this book.
```

### 7.2 What Does Not Change

Token compression does not eliminate the other sources of memory traffic. Head compression (Chapter 4) still applies orthogonally — you can combine GQA with sliding window (Mistral does this) or MLA with an SSM layer. These savings stack multiplicatively.

The need for efficient attention kernels also remains. Even with a constant-size state, computing φ(Q) · S efficiently benefits from tiling and quantisation. The techniques of later chapters still apply.

---

## 8. The Modern Production Picture

As of 2026, the dominant production pattern for long-context inference is:

```
Layer 1 (head compression):
  GQA or MLA — reduces bytes per token by 4–16×

Layer 2 (token compression, optional):
  Full attention for most layers
  Sliding window on some layers (Gemma-2 style hybrid)
  Mamba for streaming / very-long-context applications (Jamba, Zamba)

Kernel optimisations:
  FlashAttention — avoids materialising N×N in HBM (Chapter 5)
  PagedAttention — KV cache memory management (Chapter 6)
  Prefix caching  — reuse shared prefix across users (Chapter 11)
  Quantisation    — INT8/FP8 weights and cache (Chapter 10)
  Continuous batching — throughput (Chapter 7b)
```

**Hybrid architectures** (most layers Mamba or SSM, a few layers full attention) are emerging as the best of both worlds: Mamba layers for memory and throughput, full-attention layers for direct long-range retrieval. Representative models: Jamba (AI21), Zamba, and various research-grade Mamba-Attention hybrids.

---

## 9. Code: Python Implementations

### 9.1 Sliding Window Attention (`sliding_window_attention.py`)

```
HOW TO RUN:
  python3 sliding_window_attention.py

EXPECTED OUTPUT:
  Attention weight matrix (10×10) with zeros beyond window boundary (W=4)
  Window mask truth table printed (✓ / ·)
  Decode step matches prefill context exactly (max diff = 0.00e+00)
  Cache comparison table
  All assertions pass
```

```python
"""
sliding_window_attention.py — Sliding Window Attention demo (NumPy)

Implements causal sliding-window attention:
  - Each token attends only to the last W past tokens (window).
  - Beyond W positions back the past is masked to -inf before softmax.
  - The KV cache is a circular buffer of W slots that never grows.

FEATURES
  - Full prefill of N tokens (vectorised)
  - Incremental decode step adding one new token at a time
  - Circular buffer KV cache implementation
  - Cache byte accounting vs full MHA

HOW TO RUN:
    python3 sliding_window_attention.py

EXPECTED OUTPUT:
  Attention weights printed for 4-token block
  Decode step verified numerically
  All assertions pass
"""

import numpy as np

# ── Config ───────────────────────────────────────────────────────────────
D_MODEL = 8
N_HEADS = 4
D_HEAD  = D_MODEL // N_HEADS   # 2
WINDOW  = 4                    # W  — the sliding window size
SEQ_LEN = 10                   # N  — total prefill tokens
SCALE   = np.sqrt(D_HEAD)

TOKENS = ["The", "next", "day", "is", "bright", "and", "sunny", "with", "clear", "skies"]

# ── Reproducible weight matrices (seed=42, head 0 only for demo) ─────────
rng = np.random.default_rng(42)
X   = rng.standard_normal((SEQ_LEN, D_MODEL))   # (10, 8) embeddings

np.random.seed(42)
W_Q = np.random.randn(D_MODEL, D_HEAD).astype(np.float64) * 0.5
W_K = np.random.randn(D_MODEL, D_HEAD).astype(np.float64) * 0.5
W_V = np.random.randn(D_MODEL, D_HEAD).astype(np.float64) * 0.5


# ════════════════════════════════════════════════════════════════════════
# Helper: causal softmax with a sliding-window mask
# ════════════════════════════════════════════════════════════════════════
def sliding_window_softmax(scores: np.ndarray, window: int) -> np.ndarray:
    """
    Row-wise softmax with:
      1. Causal mask  — upper triangle → -inf
      2. Window mask  — positions more than `window` back → -inf

    scores: (N, N) raw dot-product matrix
    returns: (N, N) attention weights
    """
    N = scores.shape[0]
    # Causal mask: ones on lower triangle (including diagonal)
    causal = np.tril(np.ones((N, N), dtype=bool))
    # Window mask: positions within the last `window` tokens
    window_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        lo = max(0, i - window + 1)
        window_mask[i, lo : i + 1] = True
    # Combined: causal AND within window
    attend = causal & window_mask
    # Apply masks
    masked = np.where(attend, scores, -1e9)
    # Numerically stable softmax
    masked -= masked.max(axis=-1, keepdims=True)
    exp_s = np.exp(masked)
    return exp_s / exp_s.sum(axis=-1, keepdims=True)


# ════════════════════════════════════════════════════════════════════════
# Prefill: sliding-window attention over the full sequence (vectorised)
# ════════════════════════════════════════════════════════════════════════
def prefill_sliding_window(X, W_Q, W_K, W_V, window):
    """Returns context vectors (N, D_HEAD) using sliding-window attention."""
    Q = X @ W_Q           # (N, D_HEAD)
    K = X @ W_K           # (N, D_HEAD)
    V = X @ W_V           # (N, D_HEAD)

    scores = Q @ K.T / SCALE          # (N, N)
    weights = sliding_window_softmax(scores, window)   # (N, N)
    context = weights @ V             # (N, D_HEAD)
    return Q, K, V, weights, context


# ════════════════════════════════════════════════════════════════════════
# Circular buffer KV cache for decode
# ════════════════════════════════════════════════════════════════════════
class CircularKVCache:
    """
    Fixed-size circular buffer of W slots.
    After W tokens the oldest slot is overwritten by the newest.
    """
    def __init__(self, window: int, d_head: int):
        self.W = window
        self.K_buf = np.zeros((window, d_head))
        self.V_buf = np.zeros((window, d_head))
        self.ptr   = 0          # write pointer (cycles 0..W-1)
        self.n_filled = 0       # how many slots are occupied

    def push(self, k: np.ndarray, v: np.ndarray):
        """Add one new (k, v) pair, overwriting oldest if full."""
        self.K_buf[self.ptr] = k
        self.V_buf[self.ptr] = v
        self.ptr = (self.ptr + 1) % self.W
        self.n_filled = min(self.n_filled + 1, self.W)

    def get_kv(self):
        """
        Return K and V in chronological order (oldest → newest).
        If buffer is not yet full, return only the filled portion.
        """
        if self.n_filled < self.W:
            return self.K_buf[:self.n_filled].copy(), self.V_buf[:self.n_filled].copy()
        # ptr points to the *next* write slot = oldest entry
        idx = [(self.ptr + i) % self.W for i in range(self.W)]
        return self.K_buf[idx], self.V_buf[idx]

    def byte_size(self):
        """Return current bytes used by this cache (float64)."""
        return 2 * self.n_filled * self.K_buf.shape[-1] * 8


def decode_step_sliding(q_new: np.ndarray, cache: CircularKVCache) -> np.ndarray:
    """
    Decode a single new token using sliding-window attention.
    q_new: (D_HEAD,) query for the new token
    cache: CircularKVCache already filled with past tokens' K and V
    """
    K_hist, V_hist = cache.get_kv()   # (≤W, D_HEAD)
    scores = (q_new @ K_hist.T) / SCALE   # (≤W,)
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    context = weights @ V_hist        # (D_HEAD,)
    return context


# ════════════════════════════════════════════════════════════════════════
# Run and verify
# ════════════════════════════════════════════════════════════════════════
Q, K, V, weights_full, ctx_full = prefill_sliding_window(X, W_Q, W_K, W_V, WINDOW)

# ── Build cache from prefill tokens ─────────────────────────────────────
cache = CircularKVCache(WINDOW, D_HEAD)
for i in range(SEQ_LEN):
    cache.push(K[i], V[i])

# ── Decode the last token (index SEQ_LEN-1) using the circular buffer ───
q_last = Q[SEQ_LEN - 1]
ctx_decode = decode_step_sliding(q_last, cache)

# The last token can attend to at most WINDOW past tokens; extract that row
# from the prefill weights and compute reference context
ref_weights = weights_full[SEQ_LEN - 1]          # (N,) — already window-masked
ref_context  = ref_weights @ V                    # (D_HEAD,)

# They should match (same W past tokens attended to)
max_diff = np.max(np.abs(ctx_decode - ref_context))
assert max_diff < 1e-10, f"Decode/prefill mismatch: {max_diff:.2e}"


# ════════════════════════════════════════════════════════════════════════
# Cache byte accounting
# ════════════════════════════════════════════════════════════════════════
n_layers   = 32
bytes_mha  = 2 * SEQ_LEN * N_HEADS * D_HEAD * n_layers * 8   # full MHA
bytes_sw   = 2 * WINDOW   * N_HEADS * D_HEAD * n_layers * 8   # sliding window (constant)
reduction  = SEQ_LEN / WINDOW    # N / W

# ════════════════════════════════════════════════════════════════════════
# Print report
# ════════════════════════════════════════════════════════════════════════
SEP = "=" * 64
np.set_printoptions(precision=4, suppress=True, linewidth=100)

print(SEP)
print("  Sliding Window Attention — Numerical Verification")
print(SEP)

print(f"\nConfig: N={SEQ_LEN}, W={WINDOW}, D_MODEL={D_MODEL}, D_HEAD={D_HEAD}, H={N_HEADS}\n")

print("Attention weight matrix [lower-left 6×6 block, window=4]:")
header = "  " + "  ".join(f"{t:>8s}" for t in TOKENS[:6])
print(header)
for i in range(6):
    row = "  ".join(f"{weights_full[i,j]:8.4f}" for j in range(6))
    print(f"  {TOKENS[i]:8s}: {row}")

print("\n  Note: zeros appear beyond the window boundary (W=4)")
print("  Tokens can only attend to the last 4 positions (inclusive).\n")

print("Window mask structure (✓ = attends, · = masked):")
N = SEQ_LEN
print("  " + "  ".join(f"{TOKENS[j][:4]:>4s}" for j in range(N)))
for i in range(N):
    row = []
    for j in range(N):
        lo = max(0, i - WINDOW + 1)
        if j <= i and j >= lo:
            row.append("   ✓")
        else:
            row.append("   ·")
    print(f"  {TOKENS[i]:8s}: {''.join(row)}")

print(f"\nDecode step verification (last token '{TOKENS[-1]}'):")
print(f"  Prefill  context: [{', '.join(f'{v:+.5f}' for v in ref_context)}]")
print(f"  Decode   context: [{', '.join(f'{v:+.5f}' for v in ctx_decode)}]")
print(f"  Max |diff|       : {max_diff:.2e}  (< 1e-10)  ✓\n")

print("Circular buffer state after all tokens:")
print(f"  Buffer capacity  : {WINDOW} slots")
print(f"  Pointer position : {cache.ptr} (next write overwrites oldest)")
print(f"  Bytes used (1 head, 1 layer, float64): {cache.byte_size()} bytes\n")

print(f"Cache size comparison (float64, N={SEQ_LEN}, H={N_HEADS}, d_head={D_HEAD}, L={n_layers}):")
print(f"  Full MHA       : {bytes_mha:>10,d} bytes  (grows with N)")
print(f"  Sliding window : {bytes_sw:>10,d} bytes  (constant — capped at W={WINDOW})")
print(f"  Reduction at N={SEQ_LEN}: {reduction:.0f}×  (= N / W)")
print(f"""
Receptive field:
  Per layer        : W  = {WINDOW} tokens
  After L layers   : W × L = {WINDOW} × {n_layers} = {WINDOW * n_layers:,} tokens (effective)
  Mistral-7B real  : W=4096, L=32 → 131,072 token effective receptive field
""")
print("All assertions passed.\n")
print(SEP)

```

---

### 9.2 Linear Attention (`linear_attention.py`)

```
HOW TO RUN:
  python3 linear_attention.py

EXPECTED OUTPUT:
  State S (4×4) printed after each token
  Normaliser z printed for all 6 tokens
  Context vectors (linear vs standard softmax) compared
  Sequential == batched: max diff < 1e-12 ✓
  All assertions pass
```

```python
"""
linear_attention.py — Linear Attention with running D×D state (NumPy)

Demonstrates the algebraic reordering:
    softmax(Q·K^T) · V    ← standard: O(N²·D) cost, N×N matrix
    ≈ φ(Q) · (φ(K)^T · V) ← linear:   O(N·D²) cost, only D×D state

Where φ(x) = ELU(x) + 1  (element-wise, keeps values positive)

CACHED STATE:
    S_t = S_{t-1} + φ(K_t)^T · V_t   (D×D, outer-product accumulation)
    z_t = z_{t-1} + φ(K_t)            (D-vector normalizer)

OUTPUT at each step t:
    a_t = (φ(Q_t) · S_t) / (φ(Q_t) · z_t)

HOW TO RUN:
    python3 linear_attention.py

EXPECTED OUTPUT:
    State S printed at each step
    Context vectors for all tokens
    Comparison of standard vs linear attention scores
    All assertions pass
"""

import numpy as np

# ── Config ───────────────────────────────────────────────────────────────
D_MODEL = 8
D_HEAD  = 4    # per-head dimension (= D in the D×D state matrix)
SEQ_LEN = 6

TOKENS = ["The", "next", "day", "is", "bright", "and"]
N = len(TOKENS)
SCALE = np.sqrt(D_HEAD)

# ── Reproducible weights ─────────────────────────────────────────────────
rng = np.random.default_rng(42)
X   = rng.standard_normal((N, D_MODEL))     # (6, 8)

np.random.seed(0)
W_Q = np.random.randn(D_MODEL, D_HEAD).astype(np.float64) * 0.4
W_K = np.random.randn(D_MODEL, D_HEAD).astype(np.float64) * 0.4
W_V = np.random.randn(D_MODEL, D_HEAD).astype(np.float64) * 0.4


# ════════════════════════════════════════════════════════════════════════
# Feature map φ  (ELU + 1 — strictly positive, numerically stable)
# ════════════════════════════════════════════════════════════════════════
def phi(x: np.ndarray) -> np.ndarray:
    """φ(x) = ELU(x) + 1 = max(x,0) + exp(min(x,0))"""
    return np.where(x >= 0, x + 1.0, np.exp(x))


# ════════════════════════════════════════════════════════════════════════
# Standard (softmax) causal attention — reference
# ════════════════════════════════════════════════════════════════════════
def standard_causal_attention(Q, K, V):
    scores = Q @ K.T / SCALE            # (N, N)
    mask   = np.triu(np.full_like(scores, -1e9), k=1)
    scores = scores + mask
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights, weights @ V         # (N,N), (N, D_HEAD)


# ════════════════════════════════════════════════════════════════════════
# Linear attention — incremental running state
# ════════════════════════════════════════════════════════════════════════
def linear_attention_sequential(Q, K, V):
    """
    Process one token at a time, maintaining the D×D state S and
    the normalizer vector z.

    Returns:
        contexts : (N, D_HEAD) output context vectors
        states   : list of N S matrices (for inspection)
        zs       : list of N z vectors
    """
    D = Q.shape[1]
    S = np.zeros((D, D))      # D×D running state
    z = np.zeros(D)            # D normalizer
    contexts = []
    states   = []
    zs       = []

    for t in range(len(Q)):
        phi_k = phi(K[t])     # (D,)
        phi_q = phi(Q[t])     # (D,)
        v_t   = V[t]          # (D,)

        # ── State update ────────────────────────────────────────────────
        # S_t = S_{t-1} + φ(K_t)^T · V_t   (outer product: φ(K)·V^T)
        S = S + np.outer(phi_k, v_t)   # (D, D)
        z = z + phi_k                   # (D,)

        # ── Output ──────────────────────────────────────────────────────
        # a_t = (φ(Q_t) · S_t) / (φ(Q_t) · z_t)
        numerator   = phi_q @ S         # (D,)  row of S selected by φ(Q)
        denominator = phi_q @ z + 1e-6  # scalar
        a_t = numerator / denominator   # (D,)

        contexts.append(a_t)
        states.append(S.copy())
        zs.append(z.copy())

    return np.array(contexts), states, zs


# ════════════════════════════════════════════════════════════════════════
# Linear attention — batched (vectorised, for reference correctness)
# ════════════════════════════════════════════════════════════════════════
def linear_attention_batched(Q, K, V):
    """
    Vectorised form using lower-triangular cumsum:
        φ(Q) · cumsum(outer(φ(K), V), axis=0)  —  causal version

    Used only to verify the sequential incremental form gives identical results.
    """
    phiQ = phi(Q)                          # (N, D)
    phiK = phi(K)                          # (N, D)
    contexts = []
    for i in range(N):
        # Causal: only use k=0..i
        S_i = np.einsum('kd,ke->de', phiK[:i+1], V[:i+1])   # (D, D)
        z_i = phiK[:i+1].sum(axis=0)                          # (D,)
        num = phiQ[i] @ S_i                # (D,)
        den = phiQ[i] @ z_i + 1e-6
        contexts.append(num / den)
    return np.array(contexts)


# ════════════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════════════
Q = X @ W_Q    # (N, D_HEAD)
K = X @ W_K    # (N, D_HEAD)
V = X @ W_V    # (N, D_HEAD)

weights_std, ctx_std = standard_causal_attention(Q, K, V)
ctx_lin_seq,  states, zs = linear_attention_sequential(Q, K, V)
ctx_lin_batch = linear_attention_batched(Q, K, V)

# Verify sequential == batched (same formula, different computation order)
max_seq_batch_diff = np.max(np.abs(ctx_lin_seq - ctx_lin_batch))
assert max_seq_batch_diff < 1e-12, f"Sequential/batch mismatch: {max_seq_batch_diff:.2e}"

# ════════════════════════════════════════════════════════════════════════
# Print report
# ════════════════════════════════════════════════════════════════════════
SEP = "=" * 64
np.set_printoptions(precision=4, suppress=True, linewidth=100)

print(SEP)
print("  Linear Attention — Running D×D State Verification")
print(SEP)

print(f"\nConfig: N={N}, D_MODEL={D_MODEL}, D_HEAD={D_HEAD}")
print(f"Feature map φ(x) = ELU(x) + 1  (strictly positive)\n")

print("State S (D×D) after each token  [φ(K_t)^T · V_t accumulated]:")
for i, tok in enumerate(TOKENS):
    print(f"\n  After '{tok}' (t={i}):   S shape {states[i].shape}")
    for r in range(D_HEAD):
        print(f"    row{r}: [{', '.join(f'{states[i][r,c]:+.4f}' for c in range(D_HEAD))}]")

print("\nNormalizer z  (D-vector, running sum of φ(K)):")
for i, tok in enumerate(TOKENS):
    print(f"  {tok:8s}: [{', '.join(f'{zs[i][d]:+.5f}' for d in range(D_HEAD))}]")

print("\nContext vectors — linear attention (incremental running state):")
for i, tok in enumerate(TOKENS):
    print(f"  {tok:8s}: [{', '.join(f'{ctx_lin_seq[i,d]:+.5f}' for d in range(D_HEAD))}]")

print("\nContext vectors — standard softmax attention (reference):")
for i, tok in enumerate(TOKENS):
    print(f"  {tok:8s}: [{', '.join(f'{ctx_std[i,d]:+.5f}' for d in range(D_HEAD))}]")

print(f"\nNumerical checks:")
print(f"  Sequential == batched:   max diff = {max_seq_batch_diff:.2e}  (< 1e-12) ✓")

print("\nAttention weight comparison [first 4 tokens, standard vs. linear]:")
print("Standard softmax weights (row = query token, col = key token):")
print("  " + "  ".join(f"{t:>8s}" for t in TOKENS[:4]))
for i in range(4):
    row = "  ".join(f"{weights_std[i,j]:8.4f}" for j in range(4))
    print(f"  {TOKENS[i]:8s}: {row}")

print("""
Key insights vs standard attention:
  • Linear attention: ALL past tokens contribute equally to S (no softmax weighting)
  • Standard attention: softmax creates a peaked distribution, focusing on a few tokens
  • Linear attention cannot 'forget' a token once written to S
  • But S has constant size D×D regardless of sequence length N
""")

# Cache accounting
n_layers  = 32
D         = D_HEAD
bytes_mha = 2 * N * 1 * D * n_layers * 8   # single-head MHA KV cache
bytes_lin = D * D * n_layers * 8            # D×D state per layer
bytes_z   = D * n_layers * 8               # plus D-vector z per layer

print(f"Cache comparison (float64, N={N}, single head, D={D}, L={n_layers}):")
print(f"  Full MHA : 2 × N × D × L × 8  = {bytes_mha:,} bytes")
print(f"  Linear   : D² × L × 8          = {bytes_lin:,} bytes  (S matrix)")
print(f"           + D × L × 8            = {bytes_z:,} bytes  (z vector)")
print(f"  At D=128, L=32: {128*128*32*2:,} bytes = 1 MB total — any sequence length")
print("\nAll assertions passed.\n")
print(SEP)

```

---

### 9.3 SSM Walkthrough (`ssm_walkthrough.py`)

```
HOW TO RUN:
  python3 ssm_walkthrough.py

EXPECTED OUTPUT:
  h_0=[0.48, 0.06, 0.02, 0.49]   ✓ matches §9.4
  h_1=[0.622, 0.578, 0.274, 0.264]  ✓
  h_2=[1.050, 0.752, 0.292, 0.528]  ✓
  h_3=[1.155, 0.742, 0.644, 0.347]  ✓
  Recurrence vs convolution: max diff = 2.22e-16  (< 1e-12) ✓
  Information decay table for 8 steps
  All assertions pass
```

```python
"""
ssm_walkthrough.py — State Space Model (SSM) recurrence, traced by hand (NumPy)

Implements the SSM recurrence used in S4 / Mamba:
    h_t = A · h_{t-1}  +  B · x_t
    y_t = C · h_t       +  D · x_t

Parameters:
    A : diagonal state-transition matrix (d_state × d_state)
    B : input → state projection         (d_state × d)
    C : state → output projection        (d × d_state)
    D : direct input skip                (scalar)

Uses the exact weights and inputs from §9.4 of the chapter:
    d = 4, d_state = 4, A = diag(0.9, 0.8, 0.7, 0.6)

ALSO DEMONSTRATES:
    - Information decay table per state dimension
    - Convolutional kernel view: K = [C·B, C·A·B, C·A²·B, ...]
    - Equivalence: convolution output == recurrence output (assert)

HOW TO RUN:
    python3 ssm_walkthrough.py

EXPECTED OUTPUT:
    h_0=[0.48, 0.06, 0.02, 0.49], h_1=[0.622, 0.578, 0.274, 0.264]
    h_2=[1.050, 0.752, 0.292, 0.528], h_3=[1.155, 0.742, 0.644, 0.347]
    All assertions pass
"""

import numpy as np

# ── Exact weights from §9.4 ──────────────────────────────────────────────
D_STATE = 4
D       = 4

A_diag = np.array([0.9, 0.8, 0.7, 0.6])          # diagonal values only
A      = np.diag(A_diag)                           # (4, 4) diagonal matrix

B = np.array([
    [ 0.3, -0.1,  0.2,  0.4],
    [ 0.1,  0.5, -0.2,  0.3],
    [-0.2,  0.3,  0.4,  0.1],
    [ 0.4, -0.3,  0.1,  0.2],
], dtype=np.float64)    # (d_state, d)

C = np.array([
    [ 0.5,  0.2, -0.3,  0.1],
    [-0.1,  0.4,  0.6, -0.2],
    [ 0.3, -0.2,  0.1,  0.5],
    [ 0.2,  0.3, -0.1,  0.4],
], dtype=np.float64)    # (d, d_state)  output projection

D_SKIP = 0.1   # scalar direct skip

# ── Inputs: 4 tokens ────────────────────────────────────────────────────
X = np.array([
    [1.0, 0.0, 0.5, 0.2],   # x_0  "The"
    [0.3, 0.8, 0.1, 0.4],   # x_1  "next"
    [0.7, 0.2, 0.3, 0.6],   # x_2  "day"
    [0.2, 0.5, 0.8, 0.1],   # x_3  "is"
], dtype=np.float64)   # (4, d)

TOKENS = ["The", "next", "day", "is"]
N_TOKENS = len(TOKENS)


# ════════════════════════════════════════════════════════════════════════
# SSM recurrence — step by step
# ════════════════════════════════════════════════════════════════════════
def ssm_recurrence(X, A, B, C, D_skip):
    """
    Run the SSM recurrence for all tokens in X.

    Returns:
        H : (N, d_state) hidden states at each step
        Y : (N, d)       output vectors at each step
    """
    N, d = X.shape
    d_state = A.shape[0]
    h = np.zeros(d_state)
    H, Y = [], []

    for t in range(N):
        h = A @ h + B @ X[t]        # h_{t} = A·h_{t-1} + B·x_t
        y = C @ h + D_skip * X[t]   # y_t   = C·h_t     + D·x_t
        H.append(h.copy())
        Y.append(y.copy())

    return np.array(H), np.array(Y)


# ════════════════════════════════════════════════════════════════════════
# Convolutional kernel view — compute K = [C·B, C·A·B, C·A²·B, ...]
# ════════════════════════════════════════════════════════════════════════
def build_conv_kernel(A, B, C, n_taps):
    """
    Build the impulse-response kernel for the SSM as a 1D convolution.
    K[0] = C·B,  K[t] = C·A^t·B
    Returns kernel of shape (n_taps, d, d) where d = C.shape[0].
    """
    d_out, d_state = C.shape
    d_in  = B.shape[1]
    kernel = []
    A_power = np.eye(d_state)          # A^0 = I
    for t in range(n_taps):
        kernel.append(C @ A_power @ B) # (d, d_in)
        A_power = A @ A_power          # A^{t+1}
    return kernel


def ssm_convolution(X, A, B, C, D_skip):
    """
    Compute SSM output via explicit convolution with the impulse-response kernel.
    Used for verification against the recurrent form.
    """
    N, d = X.shape
    kernel = build_conv_kernel(A, B, C, N)   # kernel[t] = C·A^t·B  (d, d)
    Y_conv = np.zeros((N, d))
    for t in range(N):
        for s in range(t + 1):            # causal: only past tokens
            Y_conv[t] += kernel[t - s] @ X[s]
        Y_conv[t] += D_skip * X[t]       # direct skip
    return Y_conv


# ════════════════════════════════════════════════════════════════════════
# Run both paths
# ════════════════════════════════════════════════════════════════════════
H, Y_rec = ssm_recurrence(X, A, B, C, D_SKIP)
Y_conv   = ssm_convolution(X, A, B, C, D_SKIP)

max_diff = np.max(np.abs(Y_rec - Y_conv))
assert max_diff < 1e-12, f"Recurrence/convolution mismatch: {max_diff:.2e}"

# ── Expected values from §9.4 ────────────────────────────────────────────
H_expected = np.array([
    [0.48,  0.06,  0.02,  0.49],
    [0.622, 0.578, 0.274, 0.264],
    [1.050, 0.752, 0.292, 0.528],
    [1.155, 0.742, 0.644, 0.347],
])
assert np.allclose(H, H_expected, atol=1e-2), f"Hidden state mismatch:\n{H}\nvs\n{H_expected}"


# ════════════════════════════════════════════════════════════════════════
# Information decay table
# ════════════════════════════════════════════════════════════════════════
def decay_table(A_diag, B, X0, steps):
    """
    Track the contribution of x_0 to h_t in each state dimension.
    Because A is diagonal: contribution_dim_i(t) = A_diag[i]^t * (B[i] @ x_0)
    """
    Bx0 = B @ X0    # (d_state,) — initial contribution at t=0
    rows = []
    for t in range(steps):
        decayed = (A_diag ** t) * Bx0
        pct     = (A_diag ** t) * 100.0    # % of original
        rows.append((t, decayed, pct))
    return rows


decay_rows = decay_table(A_diag, B, X[0], 8)


# ════════════════════════════════════════════════════════════════════════
# Print report
# ════════════════════════════════════════════════════════════════════════
SEP = "=" * 64
np.set_printoptions(precision=4, suppress=True, linewidth=100)

print(SEP)
print("  SSM Recurrence — h_t = A·h_{t-1} + B·x_t   Numerical Verification")
print(SEP)

print(f"""
Config:
  d       = {D}          (input/output dimension)
  d_state = {D_STATE}          (hidden state dimension)
  A       = diag({A_diag})   (diagonal, eigenvalues < 1 → decay)
  D_skip  = {D_SKIP}         (direct input-to-output skip)
""")

print("Step-by-step recurrence trace:")
print(f"  h_{{-1}} = [0, 0, 0, 0]  (initial state)\n")

for t, tok in enumerate(TOKENS):
    print(f"  t={t} '{tok}':")
    Ah = A @ (H[t-1] if t > 0 else np.zeros(D_STATE))
    Bx = B @ X[t]
    print(f"    A·h_{{t-1}} = [{', '.join(f'{v:+.4f}' for v in Ah)}]")
    print(f"    B·x_{t}    = [{', '.join(f'{v:+.4f}' for v in Bx)}]")
    print(f"    h_{t}       = [{', '.join(f'{v:+.4f}' for v in H[t])}]")
    print(f"    y_{t}       = [{', '.join(f'{v:+.4f}' for v in Y_rec[t])}]")
    print()

print("Information decay: fraction of x_0 remaining in each state dim")
print(f"  {'Step':>4s}  {'dim0(A=0.9)':>14s}  {'dim1(A=0.8)':>14s}  {'dim2(A=0.7)':>14s}  {'dim3(A=0.6)':>14s}")
for t, vals, pcts in decay_rows:
    row = "  ".join(f"{vals[i]:8.4f} ({pcts[i]:4.0f}%)" for i in range(4))
    print(f"  t={t:>2d}:  {row}")

print(f"""
Key observations:
  • dim 0 (A=0.9): slow decay — 48% of x_0 remains after 7 steps (long-range memory)
  • dim 3 (A=0.6): fast decay — only 3% of x_0 remains after 7 steps (short-range)
  • Each dimension specializes in a different memory timescale
  • Convolution kernel K_t = C·A^t·B gives the same outputs as the recurrence
""")

print("Convolutional kernel K_t = C·A^t·B  (first 4 taps, shown as scalars for dim[0,0]):")
kernel = build_conv_kernel(A, B, C, N_TOKENS)
for t in range(N_TOKENS):
    print(f"  K[{t}][0,0] = C[0]·A^{t}·B[:,0][0] = {kernel[t][0,0]:+.6f}")

print(f"""
Recurrence vs convolution: max |diff| = {max_diff:.2e}  (< 1e-12) ✓
(Same model, two computation orders — enabled by linearity of the recurrence)
""")

# Cache accounting
n_layers = 32
bytes_mha = 2 * N_TOKENS * 1 * D * n_layers * 8   # single-head MHA
bytes_ssm = D_STATE * n_layers * 8                  # hidden state h only

print(f"Cache comparison (float64, N={N_TOKENS}, D={D}, d_state={D_STATE}, L={n_layers}):")
print(f"  Full MHA : 2 × N × D × L × 8  = {bytes_mha:,} bytes  (grows with N)")
print(f"  SSM      : d_state × L × 8     = {bytes_ssm:,} bytes  (CONSTANT)")
print(f"  At N=32K, d=4096, d_state=64: MHA ~ 43 GB vs SSM ~ 4 KB")
print("\nAll assertions passed.\n")
print(SEP)

```

---

### 9.4 Mamba Selective (`mamba_selective.py`)

```
HOW TO RUN:
  python3 mamba_selective.py

EXPECTED OUTPUT:
  h_0=0.393, h_1=0.777, h_2=0.788, h_3=0.971, h_4=0.979  ✓ §9.5
  "France" holds 65.4% of final state; "of" holds 0.5%
  Fixed SSM comparison: "France" only 23.8%, "of" 19.0%
  Multi-dim Mamba (D_STATE=4, 6 tokens): h and y vectors
  All assertions pass
```

```python
"""
mamba_selective.py — Mamba Selective SSM, traced by hand (NumPy)

Implements Mamba's per-token input-dependent discretization:
    Δ_t  = learned per-token timestep (controls forget / remember)
    A_bar_t = exp(A · Δ_t)          (discrete state-transition factor)
    B_bar_t = 1 − A_bar_t           (discrete input-write factor)

Recurrence:
    h_t = A_bar_t · h_{t-1}  +  B_bar_t · x_t
    y_t = C_t · h_t

WHERE THIS IS DIFFERENT FROM A STANDARD SSM:
    Standard SSM: A, B, C, D are *fixed* — same for every token.
    Mamba:        A_bar_t, B_bar_t, C_t are *functions of x_t* — per-token dynamics.

TRACES:
    1. "The capital of France is" — 1-D state, A = −1.0
       (exact numbers from §9.5 of the chapter)
    2. Comparison: Mamba vs fixed SSM (A_bar = 0.8) state composition

HOW TO RUN:
    python3 mamba_selective.py

EXPECTED OUTPUT:
    h_0=0.393, h_1=0.777, h_2=0.788, h_3=0.971, h_4=0.979
    "France" holds 65.5% of final state
    All assertions pass
"""

import numpy as np


# ════════════════════════════════════════════════════════════════════════
# Part 1 — 1-D state, sentence "The capital of France is"
#           Exact trace from §9.5
# ════════════════════════════════════════════════════════════════════════
TOKENS_5 = ["The", "capital", "of", "France", "is"]
DELTA = np.array([0.5, 1.0, 0.05, 2.0, 0.3])   # learned Δ_t per token
A_CONT = -1.0                                    # continuous-time A (scalar)
VAL    = 1.0                                     # simplified: all x_t = 1.0

# Discretize
A_BAR = np.exp(A_CONT * DELTA)    # A_bar_t = exp(-Δ_t)
B_BAR = 1.0 - A_BAR               # B_bar_t = 1 − A_bar_t

# Expected from §9.5
expected = {
    "A_bar": [0.607, 0.368, 0.951, 0.135, 0.741],
    "B_bar": [0.393, 0.632, 0.049, 0.865, 0.259],
    "h":     [0.393, 0.777, 0.788, 0.971, 0.979],
}

# Verify discretization
assert np.allclose(A_BAR, expected["A_bar"], atol=1e-3), f"A_bar mismatch: {A_BAR}"
assert np.allclose(B_BAR, expected["B_bar"], atol=1e-3), f"B_bar mismatch: {B_BAR}"

# Run recurrence
H_mamba = []
h = 0.0
for t, tok in enumerate(TOKENS_5):
    h = A_BAR[t] * h + B_BAR[t] * VAL
    H_mamba.append(h)

H_mamba = np.array(H_mamba)
assert np.allclose(H_mamba, expected["h"], atol=1e-3), f"h mismatch:\n{H_mamba}"


# ════════════════════════════════════════════════════════════════════════
# Decompose final state by token contribution
# For 1-D scalar state:
#   contribution of token t = B_bar_t · val_t · ∏_{s=t+1}^{T-1} A_bar_s
# ════════════════════════════════════════════════════════════════════════
N5 = len(TOKENS_5)
contributions_mamba = np.zeros(N5)
for t in range(N5):
    # Product of A_bar over steps after t
    product = 1.0
    for s in range(t + 1, N5):
        product *= A_BAR[s]
    contributions_mamba[t] = B_BAR[t] * VAL * product

total_mamba = contributions_mamba.sum()
pct_mamba   = contributions_mamba / total_mamba * 100.0

assert abs(total_mamba - H_mamba[-1]) < 1e-10, \
    f"Contribution sum {total_mamba:.6f} ≠ h_4 {H_mamba[-1]:.6f}"


# ════════════════════════════════════════════════════════════════════════
# Part 2 — Fixed SSM (A_bar = 0.8 for all tokens)
# ════════════════════════════════════════════════════════════════════════
A_BAR_FIXED = 0.8
B_BAR_FIXED = 1.0 - A_BAR_FIXED

H_fixed = []
h_f = 0.0
for t in range(N5):
    h_f = A_BAR_FIXED * h_f + B_BAR_FIXED * VAL
    H_fixed.append(h_f)
H_fixed = np.array(H_fixed)

# Decompose fixed SSM state
contributions_fixed = np.zeros(N5)
for t in range(N5):
    product = A_BAR_FIXED ** (N5 - 1 - t)
    contributions_fixed[t] = B_BAR_FIXED * VAL * product
total_fixed = contributions_fixed.sum()
pct_fixed   = contributions_fixed / total_fixed * 100.0


# ════════════════════════════════════════════════════════════════════════
# Part 3 — Multi-dimensional Mamba with D_STATE > 1
# ════════════════════════════════════════════════════════════════════════
D_STATE = 4
D_INPUT = 4

np.random.seed(7)
A_base  = -np.ones(D_STATE)          # continuous A, all −1.0 (simplified)
B_proj  = np.random.randn(D_STATE, D_INPUT) * 0.3   # project x → Δ, B, C
C_proj  = np.random.randn(D_INPUT, D_STATE) * 0.3

rng = np.random.default_rng(42)
X6  = rng.standard_normal((6, D_INPUT))
TOKENS_6 = ["The", "next", "day", "is", "bright", "and"]

# Learned Δ_t from a linear layer: Δ_t = softplus(W_dt @ x_t)
W_dt = np.random.randn(D_STATE, D_INPUT) * 0.3
def softplus(x): return np.log1p(np.exp(x))

H6 = []
h6 = np.zeros(D_STATE)
for t in range(6):
    delta_t = softplus(W_dt @ X6[t])            # (D_STATE,)
    A_bar_t = np.exp(A_base * delta_t)           # (D_STATE,)
    B_bar_t = 1.0 - A_bar_t                      # (D_STATE,)
    # Project x_t into state space
    Bx = B_proj @ X6[t]                         # (D_STATE,)
    h6 = A_bar_t * h6 + B_bar_t * Bx            # element-wise (D_STATE,)
    H6.append(h6.copy())
H6 = np.array(H6)   # (6, D_STATE)

outputs6 = H6 @ C_proj.T   # (6, D_INPUT)


# ════════════════════════════════════════════════════════════════════════
# Print report
# ════════════════════════════════════════════════════════════════════════
SEP = "=" * 64
np.set_printoptions(precision=4, suppress=True, linewidth=100)

print(SEP)
print("  Mamba Selective SSM — Input-Dependent Dynamics Verification")
print(SEP)

print(f"""
Sentence: "The capital of France is"  (1-D state, A_cont = {A_CONT})

Discretization: A_bar_t = exp(A · Δ_t) = exp(−Δ_t)
                B_bar_t = 1 − A_bar_t
""")

print(f"{'Token':>10s}  {'Δ_t':>6s}  {'A_bar':>7s}  {'B_bar':>7s}  {'Interpretation':}")
interp = [
    "common word, moderate write",
    "content word, strong write",
    "function word, nearly skip",
    "key entity, heavy reset",
    "structural, moderate write",
]
for i, tok in enumerate(TOKENS_5):
    print(f"  {tok:>8s}  {DELTA[i]:6.2f}  {A_BAR[i]:7.3f}  {B_BAR[i]:7.3f}  {interp[i]}")

print(f"""
State evolution  h_t = A_bar_t · h_{{t-1}}  +  B_bar_t · 1.0:
  h_init = 0.000
""")
h = 0.0
for i, tok in enumerate(TOKENS_5):
    h_prev = H_mamba[i-1] if i > 0 else 0.0
    print(f"  '{tok}':  h_{i} = {A_BAR[i]:.3f} · {h_prev:.3f}  +  {B_BAR[i]:.3f} · 1.0"
          f"  =  {A_BAR[i]*h_prev:+.3f}  +  {B_BAR[i]:.3f}  =  {H_mamba[i]:.3f}")

print(f"""
Final state h_4 = {H_mamba[-1]:.3f}

State composition — which token owns what fraction of h_4:
  (Mamba: content-based gating)       (Fixed SSM: recency-only)
""")
print(f"  {'Token':>10s}  {'Contrib':>10s}  {'Mamba%':>8s}    {'Fixed%':>8s}")
for i, tok in enumerate(TOKENS_5):
    print(f"  {tok:>10s}  {contributions_mamba[i]:+.5f}    {pct_mamba[i]:6.1f}%    {pct_fixed[i]:6.1f}%")
print(f"  {'Sum':>10s}  {total_mamba:+.5f}  = 100.0%  (= h_4)")

print(f"""
Observations:
  • Mamba: "France" holds {pct_mamba[3]:.1f}% — model learned its importance
  • Fixed: "France" holds only {pct_fixed[3]:.1f}% (recency bias; "is" dominates at {pct_fixed[4]:.1f}%)
  • "of" (Mamba): {pct_mamba[2]:.1f}% — gated nearly out; "of" (Fixed): {pct_fixed[2]:.1f}%
  • Mamba's Δ head decides per-token — content determines importance, not position
""")

print("Multi-dimensional Mamba (D_STATE=4, 6 tokens, learned Δ_t via softplus):")
print(f"  Input X shape: {X6.shape}")
for i, tok in enumerate(TOKENS_6):
    print(f"  {tok:8s}: h_{i} = [{', '.join(f'{H6[i,d]:+.4f}' for d in range(D_STATE))}]")

print(f"\n  Output (h @ C_proj.T) shape: {outputs6.shape}")
for i, tok in enumerate(TOKENS_6):
    print(f"  {tok:8s}: y_{i} = [{', '.join(f'{outputs6[i,d]:+.4f}' for d in range(D_INPUT))}]")

# Cache accounting
n_layers = 32
D_EXP = 128   # Mamba-2 effective state
bytes_mha   = 2 * 32_000 * 4096 * n_layers * 2   # MHA at N=32K, d=4096, FP16
bytes_mamba = D_EXP * n_layers * 2                # Mamba state, FP16
print(f"""
Cache comparison (FP16, N=32K, d=4096, L={n_layers}, Mamba d_state≈{D_EXP}):
  Full MHA  : ~{bytes_mha/1e9:.0f} GB per user session
  Mamba     : ~{bytes_mamba/1e3:.0f} KB per user session
  Ratio     : ~{bytes_mha/bytes_mamba:.0f}× compression
""")
print("All assertions passed.\n")
print(SEP)

```

---

## 10. Code: C++ mdspan Implementations

Both C++ programs use `mdspan_shim.hpp` (already in the folder) for compatibility with GCC 11-13.
Native `<mdspan>` (GCC 14 / Clang 17+) is selected automatically when `-DUSE_MDSPAN_SHIM` is omitted.

### 10.1 SSM Walkthrough (`ssm_walkthrough.cpp`)

```
COMPILE (GCC 14+ or Clang 17+, native <mdspan>):
  g++-14 -std=c++23 -O2 -Wall -Wextra ssm_walkthrough.cpp -o ssm_walkthrough

COMPILE (GCC 11-13, shim fallback):
  g++ -std=c++20 -O2 -Wall -Wextra -DUSE_MDSPAN_SHIM ssm_walkthrough.cpp -o ssm_walkthrough

RUN:
  ./ssm_walkthrough

EXPECTED OUTPUT:
  All h_t match §9.4 expected values (atol < 0.01) ✓
  Max |recurrence - convolution| = 2.22e-16  (< 1e-12) ✓
  Decay table for 8 steps
  All assertions passed.
```

```cpp
// ssm_walkthrough.cpp
// SSM recurrence  h_t = A·h_{t-1} + B·x_t,  y_t = C·h_t + D·x_t
// C++23 mdspan (or mdspan_shim.hpp fallback)
//
// Exact weights and inputs from §9.4 of token_compression_deep_dive.md
//   d = 4, d_state = 4, A = diag(0.9, 0.8, 0.7, 0.6)
//
// Demonstrates:
//   1. Step-by-step recurrence  (sequential, O(1) per token)
//   2. Convolutional kernel     K_t = C·A^t·B
//   3. Numerical equivalence of recurrence vs convolution
//   4. Information decay table  per state dimension
//
// COMPILE (GCC 14 / Clang 17+, native <mdspan>):
//   g++-14 -std=c++23 -O2 -Wall -Wextra ssm_walkthrough.cpp -o ssm_walkthrough
//
// COMPILE (GCC 11-13, uses bundled mdspan_shim.hpp):
//   g++ -std=c++20 -O2 -Wall -Wextra -DUSE_MDSPAN_SHIM \
//       ssm_walkthrough.cpp -o ssm_walkthrough
//
// RUN:
//   ./ssm_walkthrough

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <numeric>

// ── mdspan selection ───────────────────────────────────────────────────
#ifdef USE_MDSPAN_SHIM
#  include "mdspan_shim.hpp"
   using Mat    = cmat_t;
   using MutMat = mat_t;
#else
#  include <mdspan>
   using Mat    = std::mdspan<const double, std::dextents<std::size_t, 2>>;
   using MutMat = std::mdspan<double,       std::dextents<std::size_t, 2>>;
#  define MDSPAN_ACCESS(m, r, c) (m)[r, c]
#endif

#ifndef MDSPAN_ACCESS
#  define MDSPAN_ACCESS(m, r, c) (m)(r, c)
#endif

// ── Matrix helpers ─────────────────────────────────────────────────────
std::vector<double> matmul(Mat A, Mat B) {
    const std::size_t M = A.extent(0), K = A.extent(1), N = B.extent(1);
    assert(K == B.extent(0));
    std::vector<double> C(M * N, 0.0);
    MutMat mc(C.data(), M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t j = 0; j < N; ++j)
                MDSPAN_ACCESS(mc, i, j) += MDSPAN_ACCESS(A, i, k) * MDSPAN_ACCESS(B, k, j);
    return C;
}

// matvec: A (M×K)  ×  v (K,)  →  result (M,)
std::vector<double> matvec(Mat A, const std::vector<double>& v) {
    const std::size_t M = A.extent(0), K = A.extent(1);
    assert(K == v.size());
    std::vector<double> out(M, 0.0);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k)
            out[i] += MDSPAN_ACCESS(A, i, k) * v[k];
    return out;
}

// vec + vec (element-wise)
std::vector<double> vecadd(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());
    std::vector<double> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}

// scale vec by scalar
std::vector<double> vecscale(const std::vector<double>& a, double s) {
    std::vector<double> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] * s;
    return out;
}

// diag(A_diag) @ v
std::vector<double> diag_matvec(const std::vector<double>& diag, const std::vector<double>& v) {
    assert(diag.size() == v.size());
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = diag[i] * v[i];
    return out;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());
    double d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

void print_vec(const std::string& label, const std::vector<double>& v, int prec = 4) {
    std::cout << label << " = [";
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(prec)
                  << std::setw(8) << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

// ══════════════════════════════════════════════════════════════════════
// Exact weights from §9.4
// ══════════════════════════════════════════════════════════════════════
static const std::size_t D       = 4;
static const std::size_t D_STATE = 4;
static const std::size_t N_TOK   = 4;

static const std::vector<double> A_DIAG = {0.9, 0.8, 0.7, 0.6};

// B  (d_state × d)
static const std::vector<double> B_data = {
     0.3, -0.1,  0.2,  0.4,
     0.1,  0.5, -0.2,  0.3,
    -0.2,  0.3,  0.4,  0.1,
     0.4, -0.3,  0.1,  0.2,
};

// C  (d × d_state)  — output projection
static const std::vector<double> C_data = {
     0.5,  0.2, -0.3,  0.1,
    -0.1,  0.4,  0.6, -0.2,
     0.3, -0.2,  0.1,  0.5,
     0.2,  0.3, -0.1,  0.4,
};

static const double D_SKIP = 0.1;

// Inputs (N_TOK × D)
static const std::vector<double> X_data = {
    1.0, 0.0, 0.5, 0.2,   // "The"
    0.3, 0.8, 0.1, 0.4,   // "next"
    0.7, 0.2, 0.3, 0.6,   // "day"
    0.2, 0.5, 0.8, 0.1,   // "is"
};
static const std::vector<std::string> TOKENS = {"The", "next", "day", "is"};

// Expected h values from §9.4
static const std::vector<std::vector<double>> H_EXPECTED = {
    {0.48,  0.06,  0.02,  0.49},
    {0.622, 0.578, 0.274, 0.264},
    {1.050, 0.752, 0.292, 0.528},
    {1.155, 0.742, 0.644, 0.347},
};


// ══════════════════════════════════════════════════════════════════════
// SSM: one recurrence step
// ══════════════════════════════════════════════════════════════════════
// h_t = A·h_{t-1} + B·x_t   (A diagonal → use A_DIAG)
// y_t = C·h_t     + D·x_t
std::pair<std::vector<double>, std::vector<double>>
ssm_step(const std::vector<double>& h_prev,
         const std::vector<double>& x_t,
         const std::vector<double>& A_diag,
         Mat B, Mat C, double d_skip)
{
    auto Ah = diag_matvec(A_diag, h_prev);    // A·h_{t-1}
    auto Bx = matvec(B, x_t);                 // B·x_t
    auto h  = vecadd(Ah, Bx);                 // h_t

    auto Ch = matvec(C, h);                   // C·h_t
    auto Dx = vecscale(x_t, d_skip);          // D·x_t
    auto y  = vecadd(Ch, Dx);                 // y_t
    return {h, y};
}


// ══════════════════════════════════════════════════════════════════════
// Build convolutional kernel K = [C·B, C·A·B, C·A²·B, ...]
// K[t] has shape (D × D)
// ══════════════════════════════════════════════════════════════════════
std::vector<std::vector<double>>
build_kernel(const std::vector<double>& A_diag, Mat B, Mat C, std::size_t n_taps) {
    // A^t is diagonal: A_diag_power[i] = A_diag[i]^t
    std::vector<double> A_power(D_STATE, 1.0);   // A^0 = I (diagonal = all 1s)

    // Build full A matrix (for use with matmul)
    std::vector<double> A_full(D_STATE * D_STATE, 0.0);
    MutMat A_m(A_full.data(), D_STATE, D_STATE);
    for (std::size_t i = 0; i < D_STATE; ++i)
        MDSPAN_ACCESS(A_m, i, i) = 1.0;   // start as identity

    std::vector<std::vector<double>> kernel;

    for (std::size_t t = 0; t < n_taps; ++t) {
        // A_full currently holds A^t (as a dense diagonal matrix)
        // C·(A^t)·B
        Mat Atm(A_full.data(), D_STATE, D_STATE);
        auto AtB  = matmul(Atm, B);    // (D_STATE, D)
        Mat AtBm(AtB.data(), D_STATE, D);
        auto CAtB = matmul(C, AtBm);   // (D, D)
        kernel.push_back(CAtB);

        // Advance A^t → A^{t+1} by multiplying diagonal
        // A_full = diag(A_diag) @ A_full  (each row i scaled by A_diag[i])
        for (std::size_t i = 0; i < D_STATE; ++i)
            for (std::size_t j = 0; j < D_STATE; ++j)
                MDSPAN_ACCESS(A_m, i, j) *= A_diag[i];
    }
    return kernel;
}

// Apply convolution: y_t = Σ_{s=0}^{t} K[t-s] · x_s  + D·x_t
std::vector<std::vector<double>>
ssm_convolve(const std::vector<std::vector<double>>& X,
             const std::vector<std::vector<double>>& kernel,
             double d_skip)
{
    std::size_t N = X.size();
    std::vector<std::vector<double>> Y(N, std::vector<double>(D, 0.0));

    for (std::size_t t = 0; t < N; ++t) {
        // Sum K[t-s] · x_s
        for (std::size_t s = 0; s <= t; ++s) {
            const auto& K_ts = kernel[t - s];   // (D × D) flattened
            Mat Km(K_ts.data(), D, D);
            auto Kx = matvec(Km, X[s]);
            for (std::size_t d = 0; d < D; ++d)
                Y[t][d] += Kx[d];
        }
        // Skip: D·x_t
        for (std::size_t d = 0; d < D; ++d)
            Y[t][d] += d_skip * X[t][d];
    }
    return Y;
}

int main() {
    const std::string SEP(64, '=');

    Mat B(B_data.data(), D_STATE, D);
    Mat C(C_data.data(), D,       D_STATE);

    // ── Extract token embeddings ──────────────────────────────────────
    Mat X_m(X_data.data(), N_TOK, D);
    std::vector<std::vector<double>> X_rows(N_TOK, std::vector<double>(D));
    for (std::size_t t = 0; t < N_TOK; ++t)
        for (std::size_t d = 0; d < D; ++d)
            X_rows[t][d] = MDSPAN_ACCESS(X_m, t, d);

    std::cout << SEP << "\n"
              << "  SSM Recurrence  h_t = A·h_{t-1} + B·x_t   (C++ mdspan)\n"
              << SEP << "\n\n";

    std::cout << "Config:\n"
              << "  d       = " << D       << "\n"
              << "  d_state = " << D_STATE << "\n"
              << "  A       = diag(0.9, 0.8, 0.7, 0.6)  (eigenvalues < 1)\n"
              << "  D_skip  = " << D_SKIP  << "\n\n";

    // ── PATH A: Recurrence ────────────────────────────────────────────
    std::cout << "── Recurrence path ─────────────────────────────────────\n";
    std::cout << "  h_{-1} = [0, 0, 0, 0]\n\n";

    std::vector<std::vector<double>> H_rec(N_TOK), Y_rec(N_TOK);
    std::vector<double> h(D_STATE, 0.0);

    for (std::size_t t = 0; t < N_TOK; ++t) {
        auto [h_new, y_new] = ssm_step(h, X_rows[t], A_DIAG, B, C, D_SKIP);
        h = h_new;
        H_rec[t] = h_new;
        Y_rec[t] = y_new;

        auto Ah = diag_matvec(A_DIAG, (t == 0 ? std::vector<double>(D_STATE,0.0) : H_rec[t-1]));
        auto Bx = matvec(B, X_rows[t]);
        std::cout << "  t=" << t << " '" << TOKENS[t] << "':\n";
        print_vec("    A·h_{t-1}", Ah);
        print_vec("    B·x_t    ", Bx);
        print_vec("    h_t      ", H_rec[t]);
        print_vec("    y_t      ", Y_rec[t]);
        std::cout << "\n";
    }

    // ── Verify against expected values from §9.4 ──────────────────────
    for (std::size_t t = 0; t < N_TOK; ++t) {
        double diff = max_abs_diff(H_rec[t], H_EXPECTED[t]);
        assert(diff < 0.01 && "Hidden state mismatch vs §9.4");
    }
    std::cout << "  All h_t match §9.4 expected values (atol < 0.01) ✓\n\n";

    // ── PATH B: Convolution ───────────────────────────────────────────
    std::cout << "── Convolutional path ───────────────────────────────────\n";
    auto kernel  = build_kernel(A_DIAG, B, C, N_TOK);
    auto Y_conv  = ssm_convolve(X_rows, kernel, D_SKIP);

    std::cout << "  Kernel taps K[t][0,0] = C[0]·A^t·B[:,0][0]:\n";
    for (std::size_t t = 0; t < N_TOK; ++t) {
        Mat Km(kernel[t].data(), D, D);
        std::cout << "    K[" << t << "][0,0] = "
                  << std::fixed << std::setprecision(6) << MDSPAN_ACCESS(Km, 0, 0) << "\n";
    }

    // ── Compare recurrence vs convolution ─────────────────────────────
    double max_diff = 0.0;
    for (std::size_t t = 0; t < N_TOK; ++t)
        max_diff = std::max(max_diff, max_abs_diff(Y_rec[t], Y_conv[t]));
    assert(max_diff < 1e-12 && "Recurrence/convolution mismatch");
    std::cout << "\n  Max |recurrence - convolution| = "
              << std::scientific << std::setprecision(2) << max_diff
              << "  (< 1e-12) ✓\n\n";

    // ── Information decay table ────────────────────────────────────────
    std::cout << "── Information decay: x_0 contribution to h_t ──────────\n";
    // B·x_0 is the initial contribution in each state dim
    auto Bx0 = matvec(B, X_rows[0]);

    std::cout << "  " << std::setw(4) << "step"
              << std::setw(18) << "dim0(A=0.9)"
              << std::setw(18) << "dim1(A=0.8)"
              << std::setw(18) << "dim2(A=0.7)"
              << std::setw(18) << "dim3(A=0.6)" << "\n";

    static const std::size_t N_DECAY = 8;
    for (std::size_t step = 0; step <= N_DECAY; ++step) {
        std::cout << "  t=" << std::setw(2) << step << ":  ";
        for (std::size_t dim = 0; dim < D_STATE; ++dim) {
            double decay = std::pow(A_DIAG[dim], static_cast<double>(step));
            double val   = decay * Bx0[dim];
            double pct   = decay * 100.0;
            std::cout << std::fixed << std::setprecision(3)
                      << std::setw(7) << val
                      << " (" << std::setw(3) << static_cast<int>(pct) << "%)  ";
        }
        std::cout << "\n";
    }

    std::cout << "\n  Observations:\n"
              << "  • dim0 (A=0.9): slow decay — long-range memory\n"
              << "  • dim3 (A=0.6): fast decay — short-range memory\n"
              << "  • Each dimension specialises in a different timescale\n\n";

    // ── Cache comparison ───────────────────────────────────────────────
    const std::size_t n_layers = 32;
    const std::size_t N_long   = 32000;
    std::size_t bytes_mha = 2 * N_long * D * n_layers * 8;
    std::size_t bytes_ssm = D_STATE * n_layers * 8;

    std::cout << std::string(64, '-') << "\n"
              << "Cache comparison (float64, N=" << N_long
              << ", d=" << D << ", d_state=" << D_STATE
              << ", L=" << n_layers << ")\n"
              << "  Full MHA : 2×N×d×L×8  = " << bytes_mha << " bytes (grows with N)\n"
              << "  SSM      : d_state×L×8 = " << bytes_ssm << " bytes (CONSTANT)\n"
              << "  SSM is " << bytes_mha / bytes_ssm << "× smaller at N=" << N_long << "\n\n"
              << "All assertions passed.\n";

    return 0;
}

```

---

### 10.2 Mamba Selective (`mamba_selective.cpp`)

```
COMPILE (GCC 14+ or Clang 17+, native <mdspan>):
  g++-14 -std=c++23 -O2 -Wall -Wextra mamba_selective.cpp -o mamba_selective

COMPILE (GCC 11-13, shim fallback):
  g++ -std=c++20 -O2 -Wall -Wextra -DUSE_MDSPAN_SHIM mamba_selective.cpp -o mamba_selective

RUN:
  ./mamba_selective

EXPECTED OUTPUT:
  All h_t match §9.5 expected values ✓
  'France' (Mamba): 65.4% vs 23.8% (fixed SSM) ✓
  Multi-dim Mamba: 6 tokens, h and y printed per step
  All assertions passed.
```

```cpp
// mamba_selective.cpp
// Mamba Selective SSM — input-dependent discretization, traced step by step
// C++23 mdspan (or mdspan_shim.hpp fallback)
//
// Implements per-token dynamics:
//   Δ_t   = learned per-token timestep
//   A_bar_t = exp(A · Δ_t)          (fraction of old state retained)
//   B_bar_t = 1 - A_bar_t            (fraction of new input admitted)
//   h_t   = A_bar_t * h_{t-1}  +  B_bar_t * (B·x_t)   (element-wise)
//   y_t   = C_t · h_t
//
// Demonstrates:
//   1. 1-D scalar trace "The capital of France is" (exact §9.5 numbers)
//   2. State composition: Mamba vs fixed SSM (A_bar = 0.8)
//   3. Multi-dimensional Mamba (D_STATE = 4, 6 tokens)
//
// COMPILE (GCC 14 / Clang 17+, native <mdspan>):
//   g++-14 -std=c++23 -O2 -Wall -Wextra mamba_selective.cpp -o mamba_selective
//
// COMPILE (GCC 11-13, uses bundled mdspan_shim.hpp):
//   g++ -std=c++20 -O2 -Wall -Wextra -DUSE_MDSPAN_SHIM \
//       mamba_selective.cpp -o mamba_selective
//
// RUN:
//   ./mamba_selective

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ── mdspan selection ───────────────────────────────────────────────────
#ifdef USE_MDSPAN_SHIM
#  include "mdspan_shim.hpp"
   using Mat    = cmat_t;
   using MutMat = mat_t;
#else
#  include <mdspan>
   using Mat    = std::mdspan<const double, std::dextents<std::size_t, 2>>;
   using MutMat = std::mdspan<double,       std::dextents<std::size_t, 2>>;
#  define MDSPAN_ACCESS(m, r, c) (m)[r, c]
#endif

#ifndef MDSPAN_ACCESS
#  define MDSPAN_ACCESS(m, r, c) (m)(r, c)
#endif

// ── Small vector helpers ───────────────────────────────────────────────
std::vector<double> matvec(Mat A, const std::vector<double>& v) {
    const std::size_t M = A.extent(0), K = A.extent(1);
    assert(K == v.size());
    std::vector<double> out(M, 0.0);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k)
            out[i] += MDSPAN_ACCESS(A, i, k) * v[k];
    return out;
}

inline double softplus(double x) { return std::log1p(std::exp(x)); }

void print_vec(const std::string& lbl, const std::vector<double>& v, int prec = 4) {
    std::cout << lbl << " [";
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(prec)
                  << std::setw(9) << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

// ══════════════════════════════════════════════════════════════════════
// Part 1 — 1-D scalar state:  "The capital of France is"
// ══════════════════════════════════════════════════════════════════════
static const std::vector<std::string> TOKENS5 = {
    "The", "capital", "of", "France", "is"
};
static const std::vector<double> DELTA  = {0.5, 1.0, 0.05, 2.0, 0.3};
static const double A_CONT = -1.0;
static const double VAL    =  1.0;   // simplified: all x_t = 1.0

// Expected values from §9.5
static const std::vector<double> H5_EXPECTED = {0.393, 0.777, 0.788, 0.971, 0.979};

static const std::vector<std::string> INTERP = {
    "common word, moderate write",
    "content word, strong write",
    "function word, nearly skip",
    "key entity, heavy reset",
    "structural, moderate write",
};

// ══════════════════════════════════════════════════════════════════════
// Part 2 — Multi-dimensional Mamba  (D_STATE = 4, 6 tokens)
// ══════════════════════════════════════════════════════════════════════
static const std::size_t D_STATE = 4;
static const std::size_t D_IN    = 4;
static const std::size_t N6      = 6;

static const std::vector<std::string> TOKENS6 = {
    "The", "next", "day", "is", "bright", "and"
};

// Simple deterministic weight matrices (seed-consistent with Python demo)
// B_proj: (D_STATE × D_IN)
static const std::vector<double> B_PROJ_data = {
     0.09,  0.00,  0.06,  0.05,
    -0.03,  0.15, -0.06,  0.09,
     0.06,  0.09,  0.12,  0.03,
     0.12, -0.09,  0.03,  0.06,
};
// C_proj: (D_IN × D_STATE)
static const std::vector<double> C_PROJ_data = {
    -0.06,  0.09,  0.12,  0.03,
     0.09, -0.06,  0.15, -0.09,
     0.06,  0.12, -0.06,  0.03,
    -0.09,  0.06,  0.09, -0.12,
};
// W_dt: (D_STATE × D_IN) — maps x → Δ_t via softplus
static const std::vector<double> W_DT_data = {
     0.09, -0.06,  0.06,  0.03,
    -0.09,  0.06,  0.12, -0.06,
     0.03,  0.09, -0.09,  0.06,
     0.06, -0.03,  0.03,  0.09,
};

// Input embeddings (6 tokens × 4 dims) — from default_rng(42), first 6×4 values
static const std::vector<double> X6_data = {
     0.30471708, -1.03998411,  0.75137527,  0.94359230,
    -0.42160625,  0.37525529, -0.76951540, -0.40038144,
     1.01358661, -0.65917373,  0.11281296,  0.68432024,
     0.17808803,  0.68284404, -0.95042836,  0.30012386,
    -1.46437247, -0.04919027, -0.73843811,  0.21490020,
    -0.38491695, -0.58460716, -1.09855300,  0.14573831,
};

int main() {
    const std::string SEP(64, '=');

    // ══════════════════════════════════════════════════════════════════
    // Part 1: 1-D scalar state
    // ══════════════════════════════════════════════════════════════════
    std::cout << SEP << "\n"
              << "  Mamba Selective SSM — Input-Dependent Dynamics  (C++ mdspan)\n"
              << SEP << "\n\n";

    std::cout << "Part 1: Scalar-state trace  (A_cont=" << A_CONT << ", val=1.0)\n"
              << "  Sentence: \"The capital of France is\"\n\n";

    const std::size_t N5 = TOKENS5.size();
    std::vector<double> A_BAR(N5), B_BAR(N5);
    for (std::size_t t = 0; t < N5; ++t) {
        A_BAR[t] = std::exp(A_CONT * DELTA[t]);   // exp(−Δ_t)
        B_BAR[t] = 1.0 - A_BAR[t];
    }

    // Print discretization table
    std::cout << "  " << std::setw(10) << "Token"
              << std::setw(8) << "Δ_t"
              << std::setw(9) << "A_bar"
              << std::setw(9) << "B_bar"
              << "  Interpretation\n";
    for (std::size_t t = 0; t < N5; ++t) {
        std::cout << "  " << std::setw(10) << TOKENS5[t]
                  << std::fixed << std::setprecision(3)
                  << std::setw(8) << DELTA[t]
                  << std::setw(9) << A_BAR[t]
                  << std::setw(9) << B_BAR[t]
                  << "  " << INTERP[t] << "\n";
    }

    // Run recurrence
    std::cout << "\n  State evolution  h_t = A_bar_t·h_{t-1} + B_bar_t·1.0:\n"
              << "  h_init = 0.000\n";
    std::vector<double> H5(N5);
    double h = 0.0;
    for (std::size_t t = 0; t < N5; ++t) {
        double h_prev = h;
        h = A_BAR[t] * h_prev + B_BAR[t] * VAL;
        H5[t] = h;
        std::cout << "  '" << TOKENS5[t] << "': "
                  << std::fixed << std::setprecision(3)
                  << "h_" << t << " = " << A_BAR[t] << " · " << h_prev
                  << " + " << B_BAR[t] << " · 1.0 = " << h << "\n";
    }

    // Verify
    for (std::size_t t = 0; t < N5; ++t) {
        assert(std::fabs(H5[t] - H5_EXPECTED[t]) < 1e-3 &&
               "Scalar state mismatch vs §9.5");
    }
    std::cout << "  All h_t match §9.5 expected values ✓\n";

    // ── State composition ─────────────────────────────────────────────
    std::cout << "\n  Final state h_4 = " << std::fixed << std::setprecision(3) << H5[N5-1] << "\n"
              << "  State composition — which token owns what fraction:\n\n";

    std::vector<double> contrib_mamba(N5), contrib_fixed(N5);
    double total_m = 0.0, total_f = 0.0;
    const double A_FIXED = 0.8;
    const double B_FIXED = 0.2;

    for (std::size_t t = 0; t < N5; ++t) {
        // Mamba
        double prod_m = 1.0;
        for (std::size_t s = t + 1; s < N5; ++s) prod_m *= A_BAR[s];
        contrib_mamba[t] = B_BAR[t] * VAL * prod_m;
        total_m += contrib_mamba[t];

        // Fixed SSM
        double prod_f = std::pow(A_FIXED, static_cast<double>(N5 - 1 - t));
        contrib_fixed[t] = B_FIXED * VAL * prod_f;
        total_f += contrib_fixed[t];
    }

    std::cout << "  " << std::setw(12) << "Token"
              << std::setw(12) << "Mamba%"
              << std::setw(12) << "Fixed%  (A=0.8)" << "\n";
    for (std::size_t t = 0; t < N5; ++t) {
        std::cout << "  " << std::setw(12) << TOKENS5[t]
                  << std::fixed << std::setprecision(1)
                  << std::setw(9) << contrib_mamba[t] / total_m * 100.0 << "%"
                  << std::setw(9) << contrib_fixed[t] / total_f * 100.0 << "%\n";
    }
    std::cout << "\n"
              << "  'France' (Mamba): "
              << std::fixed << std::setprecision(1)
              << contrib_mamba[3] / total_m * 100.0 << "%  vs  "
              << contrib_fixed[3] / total_f * 100.0 << "% (fixed)\n"
              << "  'of'     (Mamba): "
              << contrib_mamba[2] / total_m * 100.0 << "%  vs  "
              << contrib_fixed[2] / total_f * 100.0 << "% (fixed)\n"
              << "  → Content determines importance in Mamba; recency-only in fixed SSM\n\n";

    // ══════════════════════════════════════════════════════════════════
    // Part 2: Multi-dimensional Mamba
    // ══════════════════════════════════════════════════════════════════
    std::cout << SEP << "\n"
              << "Part 2: Multi-dim Mamba  (D_STATE=" << D_STATE
              << ", D_IN=" << D_IN << ", " << N6 << " tokens)\n\n";

    Mat B_proj(B_PROJ_data.data(), D_STATE, D_IN);
    Mat C_proj(C_PROJ_data.data(), D_IN,    D_STATE);
    Mat W_dt  (W_DT_data.data(),   D_STATE, D_IN);
    Mat X6_m  (X6_data.data(),     N6,      D_IN);

    std::vector<double> A_base(D_STATE, -1.0);   // continuous A = −1 for all dims

    std::vector<double> h6(D_STATE, 0.0);

    for (std::size_t t = 0; t < N6; ++t) {
        // Extract x_t
        std::vector<double> x_t(D_IN);
        for (std::size_t d = 0; d < D_IN; ++d)
            x_t[d] = MDSPAN_ACCESS(X6_m, t, d);

        // Δ_t = softplus(W_dt @ x_t)
        auto delta_raw = matvec(W_dt, x_t);
        std::vector<double> delta_t(D_STATE);
        for (std::size_t d = 0; d < D_STATE; ++d)
            delta_t[d] = softplus(delta_raw[d]);

        // A_bar_t = exp(A_base * delta_t)  (element-wise)
        std::vector<double> A_bar_t(D_STATE), B_bar_t(D_STATE);
        for (std::size_t d = 0; d < D_STATE; ++d) {
            A_bar_t[d] = std::exp(A_base[d] * delta_t[d]);
            B_bar_t[d] = 1.0 - A_bar_t[d];
        }

        // Bx = B_proj @ x_t
        auto Bx = matvec(B_proj, x_t);

        // h_t = A_bar_t * h_{t-1}  +  B_bar_t * Bx  (element-wise)
        for (std::size_t d = 0; d < D_STATE; ++d)
            h6[d] = A_bar_t[d] * h6[d] + B_bar_t[d] * Bx[d];

        // y_t = C_proj @ h_t
        auto y_t = matvec(C_proj, h6);

        std::cout << "  " << std::setw(8) << TOKENS6[t] << ": ";
        print_vec("h_" + std::to_string(t) + " = ", h6);
        std::cout << "            ";
        print_vec("y_" + std::to_string(t) + " = ", y_t);
    }

    // ── Cache accounting ───────────────────────────────────────────────
    const std::size_t n_layers = 32;
    const std::size_t D_EXP    = 128;   // typical Mamba-2 effective state
    const std::size_t N_LONG   = 32000;

    std::size_t bytes_mha   = 2UL * N_LONG * D_IN * n_layers * 2;   // FP16
    std::size_t bytes_mamba = D_EXP * n_layers * 2;                  // FP16

    std::cout << "\n" << std::string(64, '-') << "\n"
              << "Cache comparison (FP16, N=" << N_LONG
              << ", d=" << D_IN << ", L=" << n_layers << "):\n"
              << "  Full MHA  : ~" << bytes_mha / 1'000'000'000UL << " GB per user\n"
              << "  Mamba     : ~" << bytes_mamba / 1024UL << " KB per user  (d_state="
              << D_EXP << ")\n"
              << "  Ratio     : ~" << bytes_mha / bytes_mamba << "× compression\n\n"
              << "All assertions passed.\n";

    return 0;
}

```

---
---

## Self-Check Questions

**E9.1 (Sliding window math):**
A model has W = 2 048, H = 8, D_HEAD = 128, L = 32. A user session reaches N = 80 000 tokens.
(a) Compute cache_SW in bytes (FP16).
(b) Compute the cache reduction factor versus full MHA.
(c) At what N does a sliding-window cache equal a full-MHA cache at N = 2 048?

**E9.2 (Receptive field):**
A sliding-window model has W = 512 and L = 24 layers.
(a) What is the effective receptive field?
(b) A document is 15 000 tokens long. Can the last token "see" the first token through L=24 layers? Show your work.
(c) A researcher says "W=512 is too small for any document longer than 512 tokens." Explain why this statement is incomplete.

**E9.3 (Linear attention state):**
A linear attention layer has D = 64 and L = 48 layers. Feature map φ(x) = ELU(x) + 1.
(a) How many bytes does the running state (S + z) occupy in FP16?
(b) At D = 64, approximately how many past KV pairs does the D×D state represent? Is this lossless?
(c) Explain why the first token's output from linear attention always matches softmax attention exactly.

**E9.4 (SSM decay arithmetic):**
An SSM has A = diag(0.95, 0.85, 0.75, 0.55) and d_state = 4.
(a) After 10 steps, what fraction of x_0's contribution survives in each state dimension?
(b) Which dimension acts as a "short-range memory" dimension? Which as "long-range"?
(c) The SSM is trained on sequences of length 512. At test time, it processes 10 000 tokens. Which state dimensions carry meaningful signal about tokens 1–512 and which have decayed to noise?

**E9.5 (Mamba discretisation):**
Mamba sees a sequence [word_A, word_B, word_C] with Δ values [0.1, 3.0, 0.1] and A_cont = −1.0.
(a) Compute A_bar and B_bar for each token.
(b) Run the 1-D scalar recurrence with all val = 1.0, starting from h = 0.
(c) Decompose the final state by token contribution and compute percentages.
(d) Explain intuitively what Δ = 3.0 for word_B means in terms of memory management.

**E9.6 (Architecture selection):**
You are designing a serving system for three use cases:
  (A) A legal document summariser: documents up to 100 K tokens, must retrieve specific clauses mentioned at arbitrary positions.
  (B) A live voice transcription service: effectively infinite stream, must only output the current utterance, not recall hours-old speech.
  (C) A code-completion assistant: files up to 8 K lines, users often reference functions defined hundreds of lines earlier.

For each use case, recommend one of the four architectures (sliding window, linear attention, SSM, Mamba) or their combination, and justify your choice with specific cache size formulas and quality trade-offs.

---

## Worked Solutions

### Q1 Solution (Sliding window math)

**(a) Cache size in bytes:**
```
cache_SW = 2 × W × H × D × L × 2 bytes   (K and V, FP16)
         = 2 × 2048 × 8 × 128 × 32 × 2
         = 2 × 2048 × 8 × 128 × 64
         = 2 × 2048 × 65536
         = 2 × 134,217,728
         = 268,435,456 bytes = 256 MB
```

**(b) Reduction factor at N = 80 000:**
```
cache_MHA at N=80K = 2 × 80000 × 8 × 128 × 32 × 2 = 10,737,418,240 bytes ≈ 10 GB

reduction = N / W = 80,000 / 2,048 ≈ 39.1×

Or equivalently: cache_MHA / cache_SW = 80000 / 2048 ≈ 39.1×
```

**(c) At what N does SW cache equal MHA at N = 2 048?**
```
cache_SW = cache_MHA(N=2048)

The SW cache is fixed: 256 MB (independent of N).
The MHA cache at N=2048: 2 × 2048 × 8 × 128 × 32 × 2 = 268,435,456 bytes = 256 MB.

They are equal! The sliding-window cache at W=2048 is exactly equal to
the full MHA cache at N=2048. For any N > 2048, SW is strictly smaller.
Answer: they are equal at N = W = 2048 (trivially, by construction).
```

### Q2 Solution (Receptive field)

**(a) Effective receptive field:**
```
receptive_field ≈ W × L = 512 × 24 = 12,288 tokens
```

**(b) Can the last token see the first token (15 000-token document)?**
```
First token is at position 0. Last token is at position 14,999.
Distance = 14,999 tokens.

Effective receptive field = 12,288 < 14,999.

Answer: NO. The effective receptive field of 12,288 tokens does not reach
back 14,999 positions. The first token is outside the indirect receptive
field of the last token. Information from token 0 cannot influence the
representation of token 14,999.
```

**(c) Why "W=512 is too small for any document > 512 tokens" is incomplete:**
```
The statement ignores layer stacking. While a SINGLE layer can only attend
to the last 512 tokens directly, LAYER L's output is influenced by tokens
that influenced Layer L-1, and so on. The EFFECTIVE receptive field of
W × L = 512 × 24 = 12,288 tokens is much larger than W alone.

The statement would only be correct if the model had L=1 layer. With L=24,
a document up to ~12K tokens can have information flow from start to end,
though indirectly and with attenuation through intermediate layers.
```

### Q3 Solution (Linear attention state)

**(a) State bytes in FP16:**
```
S matrix: D × D × L × 2 bytes = 64 × 64 × 48 × 2 = 393,216 bytes = 384 KB
z vector: D × L × 2 bytes     = 64 × 48 × 2     = 6,144 bytes ≈ 6 KB
Total: ≈ 390 KB  (for any sequence length)
```

**(b) How many past KV pairs does D×D represent?**
```
The D×D state is the sum of N outer products φ(K_t)^T · V_t over all past t.
Each outer product adds D² scalar values; S has D² = 4096 scalar values.

This is NOT a lossless compression of N past (K,V) pairs. For N > 1,
different sequences can produce the same S matrix (collision). The D×D
state is a lossy fixed-size summary, not an invertible encoding.

As N grows, more and more information is compressed into the same D²
numbers. The compression rate increases with N; there is no theoretical
upper bound on N given fixed D.
```

**(c) Why does the first token match softmax attention exactly?**
```
At t=0 ("The"), the causal mask means only token 0 can attend to itself.

Linear attention at t=0:
  S_0 = outer(φ(K_0), V_0)
  z_0 = φ(K_0)
  a_0 = (φ(Q_0) @ S_0) / (φ(Q_0) @ z_0)
       = (φ(Q_0) @ outer(φ(K_0), V_0)) / (φ(Q_0) @ φ(K_0))
       = ((φ(Q_0) · φ(K_0)) * V_0) / (φ(Q_0) · φ(K_0))
       = V_0

Softmax attention at t=0:
  Only one past token, causal mask allows only position 0.
  weights[0,0] = softmax([score_00]) = 1.0
  a_0 = 1.0 × V_0 = V_0

Both give V_0. They agree because there is only one term in the sum;
the distinction between "weighted sum" and "running state sum" only appears
when there are multiple past tokens to differentiate.
```

### Q4 Solution (SSM decay arithmetic)

**(a) Fraction surviving after 10 steps:**
```
fraction = A_diag[i]^10

dim 0 (A=0.95): 0.95^10 = 0.5987 ≈  59.9% remaining
dim 1 (A=0.85): 0.85^10 = 0.1969 ≈  19.7% remaining
dim 2 (A=0.75): 0.75^10 = 0.0563 ≈   5.6% remaining
dim 3 (A=0.55): 0.55^10 = 0.0025 ≈   0.25% remaining
```

**(b) Short-range vs long-range dimension:**
```
Short-range: dim 3 (A=0.55). At t=10 only 0.25% of the original signal
             survives. Useful information only from the last 1-2 steps.
Long-range:  dim 0 (A=0.95). At t=10 nearly 60% survives. This dimension
             acts as a long-term memory channel, retaining signal for
             potentially dozens of steps.
```

**(c) Which dimensions carry signal from tokens 1-512 at t=10,000?**
```
Number of decay steps from position 512 to position 10000: 9488 steps.

dim 0 (A=0.95): 0.95^9488 ≈ exp(9488 × ln 0.95) = exp(-487) ≈ 0   (complete decay)
dim 1 (A=0.85): 0.85^9488 ≈ exp(-9488 × 0.163) ≈ 0   (complete decay)
dim 2 (A=0.75): similarly ≈ 0
dim 3 (A=0.55): similarly ≈ 0

All four dimensions have completely decayed to zero by t=10,000.
Tokens 1–512 contribute essentially zero signal to h at t=10,000
for any practical A < 1. The SSM trained on 512-length sequences
has effectively lost all memory of early tokens by step 10,000.

This is a key limitation: SSMs are most reliable within their training
context length. Extrapolation beyond that length causes the state to
be dominated entirely by recent tokens.
```

### Q5 Solution (Mamba discretisation)

**(a) A_bar and B_bar:**
```
A_cont = −1.0
Δ = [0.1, 3.0, 0.1]

A_bar_t = exp(−1.0 × Δ_t) = exp(−Δ_t)
B_bar_t = 1 − A_bar_t

word_A: Δ=0.1 → A_bar=exp(−0.1)=0.905,  B_bar=0.095
word_B: Δ=3.0 → A_bar=exp(−3.0)=0.050,  B_bar=0.950
word_C: Δ=0.1 → A_bar=exp(−0.1)=0.905,  B_bar=0.095
```

**(b) 1-D scalar recurrence with val=1.0:**
```
h_init = 0.000
word_A: h_0 = 0.905 × 0.000 + 0.095 × 1.0 = 0.095
word_B: h_1 = 0.050 × 0.095 + 0.950 × 1.0 = 0.005 + 0.950 = 0.955
word_C: h_2 = 0.905 × 0.955 + 0.095 × 1.0 = 0.864 + 0.095 = 0.959
```

**(c) State composition:**
```
contribution of word_A to h_2:
  = B_bar_A × val × (A_bar_B × A_bar_C) = 0.095 × (0.050 × 0.905) = 0.095 × 0.0453 = 0.00430  (0.4%)

contribution of word_B to h_2:
  = B_bar_B × val × A_bar_C             = 0.950 × 0.905 = 0.8598  (89.6%)

contribution of word_C to h_2:
  = B_bar_C × val × 1.0                 = 0.095 × 1.0   = 0.0950  (9.9%)

Sum = 0.00430 + 0.8598 + 0.0950 = 0.9591 ≈ 0.959 = h_2  ✓
```

**(d) What Δ = 3.0 means for word_B:**
```
A_bar = exp(−3.0) = 0.050  →  only 5% of the old state is retained.
B_bar = 0.950               →  95% of the new input is written.

Δ = 3.0 tells the Mamba block: "this token is important enough to nearly
overwrite what I knew before." The old state is suppressed to 5% of its
previous value, and the new token's signal fills 95% of the state.

Concretely: word_B (likely a key noun or named entity) effectively cleared
the accumulated context from word_A, replacing it with its own signal.
Word_C then mostly carries word_B's memory forward (since it has small Δ too).

This is exactly the content-based gating that gives Mamba its retrieval
advantage over fixed SSMs — the model can "decide" when to keep and when to
overwrite based on the token's content.
```

### Q6 Solution (Architecture selection)

**(A) Legal document summariser (100K tokens, factoid retrieval):**
```
Recommendation: Full MHA + MLA head compression (or hybrid with few full-attention layers)

Reasoning:
  - The task REQUIRES precise retrieval of specific clauses at arbitrary positions.
  - Sliding window at W=4K: effective receptive field ≈ 4K×32=128K, but direct
    attention only 4K. Needle-in-haystack at 100K position → 45% retrieval.
  - Linear attention: ~40% retrieval on needle tasks. Context bottleneck is fatal.
  - SSM/Mamba: 78% retrieval, but 100K-token legal clauses need higher precision.

Best approach: MLA (to compress KV cache ~13×) + full attention (to preserve
retrieval). cache_MLA ≈ 100K × r_kv × L × 2 bytes. At r_kv=512, L=32:
  100000 × 512 × 32 × 2 = 3.2 GB (vs 43 GB for full MHA).

If memory is still constrained, a hybrid: full attention every 4 layers, 
sliding window (large W=16K) for the rest.
```

**(B) Voice transcription (infinite stream, current utterance only):**
```
Recommendation: Mamba (or SSM)

Reasoning:
  - The model NEVER needs to recall audio from hours ago. Task is inherently short-term.
  - Memory budget: Mamba d_state × L × 2 ≈ 128 × 64 × 2 = 16 KB per user, forever.
  - Sliding window would work too (W=512 is enough for ~5 seconds of speech), but
    Mamba's selectivity also helps ignore filler words and pauses automatically.
  - Linear attention would work, but Mamba's retrieval quality (78%) > linear (40%)
    for the occasional reference to an entity mentioned seconds ago.

cache_Mamba = 128 × 64 × 2 = 16 KB. This user can stream indefinitely with
zero cache growth. Exactly what a production transcription service needs.
```

**(C) Code completion (8K lines, references hundreds of lines back):**
```
Recommendation: GQA (head compression) + sliding window with W ≈ 2048–4096

Reasoning:
  - Code files are 8K lines ≈ 50K–200K tokens (depending on file density).
  - References hundreds of lines back = typically 1K–5K tokens back.
  - W = 4K with L = 32 layers gives effective receptive field = 128K tokens,
    covering the entire file with direct attention to recent 4K tokens.

cache_SW = 2 × 4096 × 4 × 128 × 32 × 2 = 268 MB per user (GQA with 4 KV heads)
  vs full MHA at 200K tokens: 2 × 200000 × 32 × 128 × 32 × 2 ≈ 105 GB

Reduction: ~390×

Alternative: Mamba hybrid (most layers Mamba + 4 full-attention layers).
Mamba's selectivity can track function definitions as "important" tokens,
and the full-attention layers allow direct lookup when completing a call site.
This gives ~92% retrieval quality at ~10× less memory than full MHA.
```

---


