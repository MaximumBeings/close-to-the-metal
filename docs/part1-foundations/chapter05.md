# Chapter 5 — Flash Attention: Tiling, Recomputation, and the Backward Pass

> *"The insight behind FlashAttention is profound: we do not need to ever fully materialize the N×N attention matrix. By keeping data in SRAM, we can compute exact attention while dramatically reducing HBM reads and writes."*

---

## Why This Chapter Exists

Chapter 4 showed you *what* attention computes. This chapter shows you *how to compute it fast*. Standard attention materializes an N×N score matrix — 64 MB for a single head at N=4096 — and performs multiple round-trips through slow HBM memory. FlashAttention (Dao et al., 2022) eliminates those round-trips entirely using two ideas: **tiled block computation** and **online softmax**. Every number in this chapter is traced by hand through complete worked examples so you understand not just what the algorithm does, but why the rescaling works and why the result is numerically identical to standard attention.

**What you will know by the end:**

- Why standard attention is memory-bandwidth-bound, not compute-bound.
- The GPU memory hierarchy and the cost of HBM vs. SRAM access.
- Online softmax: a single-pass algorithm that computes exact softmax with running rescaling.
- Block matrix multiplication: how tiling converts O(MN) reads to O(M+N).
- FlashAttention-1: the complete tiled fused-kernel algorithm with pseudocode.
- Two fully traced end-to-end worked examples: a minimal 4-token case and a large-score-difference stress test.
- FlashAttention-2 and FlashAttention-3 improvements.
- FlashDecoding for long-context inference.
- IO complexity analysis: exact HBM read/write counts.

---

## 5.1 The Memory Bandwidth Problem `[FOUNDATIONAL]`

Modern GPUs have enormous FLOP throughput but modest memory bandwidth. The bottleneck in transformer inference is not arithmetic — it is data movement.

**The standard attention pipeline:**
```
Step 1: S = Q Kᵀ / √d    (N×d) × (d×N) = (N×N) ← write N² floats to HBM
Step 2: P = softmax(S)                           ← read N², write N² to HBM
Step 3: O = P V          (N×N) × (N×d) = (N×d) ← read N², write N×d to HBM

Total HBM writes: S (N²), P (N²), O (N×d)
Total HBM reads:  Q (N×d), K (N×d), V (N×d), S (N²), P (N²)
```

For N=4096, d=64 (one head):
```
N² = 16,777,216 floats = 64 MB (FP32)

Total memory traffic:
  Reads:  2×Nd + 2×N² = 2×1M + 2×64M ≈ 130 MB
  Writes: 2×N² + Nd   = 2×64M + 1M   ≈ 129 MB
  Total:  ≈ 259 MB for ONE head, ONE layer, ONE step

For a 70B model (80 layers, 64 heads, batch=1):
  259 MB × 64 heads × 80 layers = ~1.3 TB of HBM traffic per forward pass!
```

An A100 GPU has ~2 TB/s HBM bandwidth, but only ~19 TB/s SRAM bandwidth. Every unnecessary HBM read/write is roughly 10× slower than the same computation in SRAM.

**FlashAttention's goal**: eliminate the N×N intermediate matrices. Never write S or P to HBM at all. Keep everything in the ~192 KB SRAM and accumulate the final result O there.

---

## 5.2 GPU Memory Hierarchy `[FOUNDATIONAL]`

```
GPU Memory Hierarchy (NVIDIA A100)
════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────┐
  │  HBM (High Bandwidth Memory)                │  ← 40-80 GB
  │  ~2 TB/s bandwidth                          │  ← Slow (off-chip)
  │  Stores: Q, K, V, O, all model weights      │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  L2 Cache                                   │  ← 40 MB
  │  ~5 TB/s bandwidth                          │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  Shared Memory / L1 Cache (SRAM)            │  ← ~192 KB per SM
  │  ~19 TB/s bandwidth                         │  ← Fast (on-chip)
  │  Stores: current tiles Qi, Kj, Vj           │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  Registers                                  │  ← ~256 KB per SM
  │  Fastest possible access                    │
  └─────────────────────────────────────────────┘

HBM vs SRAM:
  HBM bandwidth:  ~2 TB/s   (off-chip)
  SRAM bandwidth: ~19 TB/s  (on-chip) ← ~9.5× faster
  HBM latency:    ~hundreds of ns
  SRAM latency:   ~tens of ns
```

FlashAttention's design philosophy in one sentence: **keep as much computation as possible in SRAM, and minimize HBM reads and writes**. This is called being "IO-aware."

---

## 5.3 Online Softmax — The Algorithmic Foundation `[ESSENTIAL]`

### 5.3.1 The Problem with Standard Softmax

Numerically stable softmax for `x = [x₁, x₂, ..., xₙ]` requires three sequential passes:

```
Standard Safe Softmax (3 passes — BAD for SRAM efficiency)
══════════════════════════════════════════════════════════
Pass 1: m = max(x₁, ..., xₙ)          ← read all N elements
Pass 2: l = Σᵢ exp(xᵢ - m)           ← read all N elements again
Pass 3: softmax(xᵢ) = exp(xᵢ - m) / l ← read all N elements a third time
```

The shift by `m` prevents `exp(xᵢ)` from overflowing for large `xᵢ`. Without it, `exp(1000)` is `+∞` in FP32.

If we process scores block by block (to stay in SRAM), we cannot finish Pass 1 until we've seen *all* blocks — defeating the purpose of tiling.

**The key question:** Can we compute the correct softmax in a **single pass**, updating running statistics as we go, so the final result after seeing all elements is numerically identical to the 3-pass result?

The answer is **yes**.

### 5.3.2 Online Softmax: The Intuition

We maintain two quantities:
- **m**: the maximum value seen so far
- **l**: the sum `Σᵢ exp(xᵢ - m)` for all elements seen so far

When we encounter a new maximum `m_new > m_old`, we can *retroactively correct* all previous contributions with a single multiplication:

```
Old contributions used m_old as reference: exp(xᵢ - m_old)
New reference: exp(xᵢ - m_new) = exp(xᵢ - m_old) × exp(m_old - m_new)
                                              ↑
                               α = correction factor (same for ALL old terms)

Therefore:
  l_new = l_old × α + exp(x_new - m_new)
  where α = exp(m_old - m_new)
```

One scalar multiplication converts the entire running sum from "relative to m_old" to "relative to m_new."

### 5.3.3 Formal Algorithm

```python
# Online Softmax
# Input: vector x of length N
# Output: softmax(x), computed in ONE PASS

m = -inf    # running maximum
l = 0       # running sum of exp(xᵢ - m)

for i in range(N):
    m_prev = m
    m = max(m, x[i])          # Step 1: update max
    α = exp(m_prev - m)        # Step 2: correction factor
    l = l * α + exp(x[i] - m) # Step 3: update running sum

# After loop: m = global max, l = Σᵢ exp(xᵢ - m)
# Final normalization:
softmax = [exp(x[i] - m) / l for i in range(N)]
```

When `m` does not change (`x[i] ≤ m`), `α = exp(0) = 1` — no rescaling, just accumulate.

### 5.3.4 Worked Example 1 — Simple Case

**x = [2, 4, 1, 3]** — one max update at position 1.

```
Initial state: m = -∞, l = 0

─────────────────────────────────────────────────────
Step 1: x₁ = 2
─────────────────────────────────────────────────────
  m_prev = -∞
  m = max(-∞, 2) = 2
  α = exp(-∞ - 2) = exp(-∞) = 0
  l = 0 × 0 + exp(2 - 2) = 0 + 1 = 1.0
  State: m=2, l=1.0

─────────────────────────────────────────────────────
Step 2: x₂ = 4   ← NEW MAXIMUM
─────────────────────────────────────────────────────
  m_prev = 2
  m = max(2, 4) = 4   ← max changes!
  α = exp(2 - 4) = exp(-2) ≈ 0.1353
  l = 1.0 × 0.1353 + exp(4 - 4) = 0.1353 + 1.0 = 1.1353
  State: m=4, l=1.1353

  VERIFY: true sum with m=4:
    exp(2-4) + exp(4-4) = 0.1353 + 1.0 = 1.1353  ✓

─────────────────────────────────────────────────────
Step 3: x₃ = 1   ← below max, no rescaling
─────────────────────────────────────────────────────
  m = max(4, 1) = 4   (no change)
  α = exp(4 - 4) = 1.0
  l = 1.1353 × 1.0 + exp(1 - 4) = 1.1353 + 0.0498 = 1.1851
  State: m=4, l=1.1851

─────────────────────────────────────────────────────
Step 4: x₄ = 3   ← below max
─────────────────────────────────────────────────────
  l = 1.1851 + exp(3 - 4) = 1.1851 + 0.3679 = 1.5530
  State: m=4, l=1.5530

─────────────────────────────────────────────────────
FINAL VERIFICATION
─────────────────────────────────────────────────────
True sum with m=4:
  exp(2-4)+exp(4-4)+exp(1-4)+exp(3-4) = 0.1353+1.0+0.0498+0.3679 = 1.5530 ✓

softmax values:
  softmax(2) = 0.1353 / 1.5530 = 0.0871
  softmax(4) = 1.0000 / 1.5530 = 0.6439
  softmax(1) = 0.0498 / 1.5530 = 0.0321
  softmax(3) = 0.3679 / 1.5530 = 0.2369
  Sum = 1.0000  ✓
```

### 5.3.5 Worked Example 2 — Multiple Max Updates

**x = [3, 2, 5, 1, 6, 4]** — two maximum updates (at positions 3 and 5).

```
Initial: m = -∞, l = 0

─────────────────────────────────────────────────────
Step 1: x₁ = 3
─────────────────────────────────────────────────────
  m=3, α=0, l = 0×0 + exp(0) = 1.0

─────────────────────────────────────────────────────
Step 2: x₂ = 2  (below max)
─────────────────────────────────────────────────────
  m=3, α=1.0, l = 1.0 + exp(2-3) = 1.0 + 0.3679 = 1.3679

─────────────────────────────────────────────────────
Step 3: x₃ = 5  ← FIRST MAX UPDATE (3 → 5)
─────────────────────────────────────────────────────
  m_prev=3, m=5
  α = exp(3 - 5) = exp(-2) = 0.1353

  Conceptual effect of rescaling:
    Old: exp(3-3)=1.0000  → New: 1.0000 × 0.1353 = 0.1353 = exp(3-5) ✓
    Old: exp(2-3)=0.3679  → New: 0.3679 × 0.1353 = 0.0498 = exp(2-5) ✓

  l = 1.3679 × 0.1353 + exp(5-5)
    = 0.1852 + 1.0 = 1.1852

  VERIFY: exp(3-5)+exp(2-5)+exp(5-5) = 0.1353+0.0498+1.0 = 1.1851 ✓

─────────────────────────────────────────────────────
Step 4: x₄ = 1  (below max)
─────────────────────────────────────────────────────
  l = 1.1852 + exp(1-5) = 1.1852 + 0.0183 = 1.2035

─────────────────────────────────────────────────────
Step 5: x₅ = 6  ← SECOND MAX UPDATE (5 → 6)
─────────────────────────────────────────────────────
  m_prev=5, m=6
  α = exp(5 - 6) = exp(-1) = 0.3679
  l = 1.2035 × 0.3679 + exp(6-6) = 0.4427 + 1.0 = 1.4427

─────────────────────────────────────────────────────
Step 6: x₆ = 4  (below max)
─────────────────────────────────────────────────────
  l = 1.4427 + exp(4-6) = 1.4427 + 0.1353 = 1.5780

─────────────────────────────────────────────────────
FINAL VERIFICATION
─────────────────────────────────────────────────────
True sum with m=6:
  exp(3-6)+exp(2-6)+exp(5-6)+exp(1-6)+exp(6-6)+exp(4-6)
= 0.0498+0.0183+0.3679+0.0067+1.0000+0.1353
= 1.5780 ✓

softmax values (= exp(xᵢ-6) / 1.5780):
  softmax(3) = 0.0498/1.5780 = 0.0316
  softmax(2) = 0.0183/1.5780 = 0.0116
  softmax(5) = 0.3679/1.5780 = 0.2331
  softmax(1) = 0.0067/1.5780 = 0.0042
  softmax(6) = 1.0000/1.5780 = 0.6337
  softmax(4) = 0.1353/1.5780 = 0.0857
  Sum = 1.0000 ✓
```

### 5.3.6 Proof of Correctness by Induction

We prove: after processing `x₁, ..., xₙ`:
- **(a)** `mₙ = max(x₁, ..., xₙ)` — the true global maximum
- **(b)** `lₙ = Σᵢ₌₁ᴺ exp(xᵢ - mₙ)` — the true sum

**Base Case (N=1):**
```
m₁ = max(-∞, x₁) = x₁          ✓  (true max of {x₁})
l₁ = 0 × exp(-∞ - x₁) + exp(x₁ - x₁) = 1  ✓
```

**Inductive Step:** Assume (a) and (b) hold for N. Process `x_{N+1}`:

For (a):
```
m_{N+1} = max(mₙ, x_{N+1}) = max(max(x₁,...,xₙ), x_{N+1}) = max(x₁,...,x_{N+1})  ✓
```

For (b):
```
Target: l_{N+1} = Σᵢ₌₁^{N+1} exp(xᵢ - m_{N+1})

Split:  = [Σᵢ₌₁ᴺ exp(xᵢ - m_{N+1})] + exp(x_{N+1} - m_{N+1})

Rewrite each old term:
  exp(xᵢ - m_{N+1}) = exp(xᵢ - mₙ) × exp(mₙ - m_{N+1})
                     = exp(xᵢ - mₙ) × α

Factor out α (constant w.r.t. i):
  Σᵢ₌₁ᴺ exp(xᵢ - m_{N+1}) = α × Σᵢ₌₁ᴺ exp(xᵢ - mₙ) = α × lₙ

Therefore:
  l_{N+1} = α × lₙ + exp(x_{N+1} - m_{N+1})  ← exactly our recurrence  ✓
```

The online algorithm produces **exact** softmax in a **single pass**. QED.

---

## 5.4 Block Matrix Multiplication `[ESSENTIAL]`

### 5.4.1 The Tiling Principle

Matrix multiplication `C = A × B` where A is (M×K) and B is (K×N) can be decomposed into independent block subproblems:

```
Partition A into (r×k) blocks, B into (k×c) blocks:

  A:                   B:
  ┌──────┬──────┐      ┌──────┬──────┬──────┐
  │ A₁₁  │ A₁₂  │  ×   │ B₁₁  │ B₁₂  │ B₁₃  │
  │(r×k) │(r×k) │      │(k×c) │(k×c) │(k×c) │
  ├──────┼──────┤      ├──────┼──────┼──────┤
  │ A₂₁  │ A₂₂  │      │ B₂₁  │ B₂₂  │ B₂₃  │
  └──────┴──────┘      └──────┴──────┴──────┘

  Cᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ  ← accumulate over K-blocks
```

Each block `Aᵢₖ` and `Bₖⱼ` can be loaded into SRAM, multiplied there, and the result accumulated into `Cᵢⱼ` — which also stays in SRAM across iterations. The full matrices A and B are never simultaneously in SRAM.

### 5.4.2 Why Tiling Reduces HBM Traffic

Without tiling: B must be read once per row of A.
```
Total B reads: (M/r) × K×N  ← catastrophically large
```

With tiling: each B-block is paired with exactly one A-block.
```
Total HBM reads: M×K + K×N  ← each matrix read just once
```

Tiling converts O(MN) redundant reads into O(M+N) amortized reads.

### 5.4.3 Why Naïve Block Attention Is Wrong

Without online softmax, a block-wise attempt would be:

```python
for i in range(Tr):        # each Q block
    for j in range(Tc):    # each K/V block
        S_ij = Q_i @ K_j.T
        P_ij = softmax(S_ij)   # ← WRONG: local softmax!
        O_i += P_ij @ V_j
```

**This is incorrect.** `softmax(S_ij)` normalizes only over the `Bᶜ` keys in block j. But the true softmax must normalize over **all N keys simultaneously**:

```
Local (wrong) for row 0 with 2 K-blocks of 4 keys each:
  Block 1 softmax([s₀₀, s₀₁, s₀₂, s₀₃])  ← sums to 1
  Block 2 softmax([s₀₄, s₀₅, s₀₆, s₀₇])  ← sums to 1
  Total: sums to 2 ← WRONG

Correct: softmax([s₀₀, s₀₁, s₀₂, s₀₃, s₀₄, s₀₅, s₀₆, s₀₇]) ← sums to 1
```

---

## 5.5 FlashAttention-1 — The Complete Algorithm `[ESSENTIAL]`

### 5.5.1 Online Softmax Extended to Blocks

For a query block Qᵢ of `Bᵣ` rows, we maintain **per-row** running statistics:

```
m_i ∈ ℝ^Bᵣ   — one maximum per query row
l_i ∈ ℝ^Bᵣ   — one sum per query row
O_i ∈ ℝ^(Bᵣ×d) — partial output matrix

When processing K/V block j (yielding S_ij ∈ ℝ^(Bᵣ×Bᶜ)):

  1. local_max = rowmax(S_ij)  ∈ ℝ^Bᵣ
  2. m_new = max(m_old, local_max)  ∈ ℝ^Bᵣ  (elementwise)
  3. α = exp(m_old - m_new)  ∈ ℝ^Bᵣ  (per-row rescaling factors)
  4. P̃ = exp(S_ij - m_new[:, None])  ∈ ℝ^(Bᵣ×Bᶜ)  (unnorm. attention)
  5. l_new = α × l_old + rowsum(P̃)  ∈ ℝ^Bᵣ
  6. O_new = diag(α) × O_old + P̃ @ V_j  ∈ ℝ^(Bᵣ×d)
             ↑ each row i of O_old is multiplied by α[i]
```

This is the heart of FlashAttention. Row `i` of the output maintains correct normalization across all key blocks using only `O(Bᵣ)` extra state.

### 5.5.2 Full Algorithm

```
FlashAttention-1 Algorithm
══════════════════════════════════════════════════════════════════
Input:  Q, K, V ∈ ℝ^(N×d) in HBM; block sizes Bᵣ, Bᶜ
Output: O ∈ ℝ^(N×d) in HBM; L ∈ ℝ^N (logsumexp, for backward pass)

Tᵣ = ⌈N/Bᵣ⌉  (Q blocks)
Tᶜ = ⌈N/Bᶜ⌉  (K/V blocks)

──────────────────────────────────────────────────────────────────
OUTER LOOP: for i = 1 to Tᵣ
──────────────────────────────────────────────────────────────────

  Load Qᵢ ∈ ℝ^(Bᵣ×d) from HBM → SRAM

  Initialize in SRAM:
    Oᵢ = 0  ∈ ℝ^(Bᵣ×d)   (output accumulator)
    lᵢ = 0  ∈ ℝ^Bᵣ        (sum accumulator)
    mᵢ = -∞ ∈ ℝ^Bᵣ        (max tracker)

  ────────────────────────────────────────────────────────────────
  INNER LOOP: for j = 1 to Tᶜ
  ────────────────────────────────────────────────────────────────

    Load Kⱼ, Vⱼ ∈ ℝ^(Bᶜ×d) from HBM → SRAM

    1. Sᵢⱼ = Qᵢ @ Kⱼᵀ / √d  ∈ ℝ^(Bᵣ×Bᶜ)

    2. m̃ᵢⱼ = rowmax(Sᵢⱼ)     ∈ ℝ^Bᵣ
       mᵢ_new = max(mᵢ, m̃ᵢⱼ)  ∈ ℝ^Bᵣ  (elementwise)

    3. α = exp(mᵢ - mᵢ_new)   ∈ ℝ^Bᵣ

    4. P̃ᵢⱼ = exp(Sᵢⱼ - mᵢ_new)  ∈ ℝ^(Bᵣ×Bᶜ)
              (broadcast mᵢ_new across columns)

    5. l̃ᵢⱼ = rowsum(P̃ᵢⱼ)     ∈ ℝ^Bᵣ
       lᵢ_new = α ⊙ lᵢ + l̃ᵢⱼ  ∈ ℝ^Bᵣ  (⊙ = elementwise)

    6. Oᵢ_new = diag(α) · Oᵢ + P̃ᵢⱼ @ Vⱼ  ∈ ℝ^(Bᵣ×d)

    7. mᵢ ← mᵢ_new;  lᵢ ← lᵢ_new;  Oᵢ ← Oᵢ_new

  ─── end inner loop ──────────────────────────────────────────────

  Finalize:
    Oᵢ = diag(1/lᵢ) · Oᵢ       (divide each row by its sum)
    Lᵢ = mᵢ + log(lᵢ)           (logsumexp, needed for backward pass)

  Write Oᵢ → HBM
  Write Lᵢ → HBM

─── end outer loop ─────────────────────────────────────────────────

Return O, L
```

The logsumexp `L = m + log(l)` is saved to HBM for the **backward pass** during training. During backpropagation, `softmax(Sᵢⱼ) = exp(Sᵢⱼ - Lᵢ)` can be recomputed on the fly from L — no need to store the N×N attention matrix at all.

---

## 5.6 Worked Example A — Minimal Case: N=4, d=2 `[HANDS-ON]`

We trace every scalar through the algorithm.

```
Setup:
  N=4, d=2, √d = √2 ≈ 1.414
  Bᵣ=2, Bᶜ=2  →  Tᵣ=2, Tᶜ=2

Q = ┌──────┐    K = ┌──────┐    V = ┌──────┐
    │1.0 0.0│        │1.0 0.0│        │0.5 0.3│
    │0.0 1.0│        │0.0 1.0│        │0.2 0.8│
    │1.0 1.0│        │1.0 1.0│        │0.6 0.1│
    │0.5 0.5│        │0.5 0.5│        │0.3 0.7│
    └──────┘        └──────┘        └──────┘
```

### 5.6.1 Outer Loop i=1: Process Q₁ = Q[0:2]

```
Q₁ = ┌──────┐    K₁ = K[0:2] = ┌──────┐    K₂ = K[2:4] = ┌──────┐
     │1.0 0.0│                   │1.0 0.0│                   │1.0 1.0│
     │0.0 1.0│                   │0.0 1.0│                   │0.5 0.5│
     └──────┘                   └──────┘                   └──────┘

V₁ = ┌──────┐    V₂ = ┌──────┐
     │0.5 0.3│          │0.6 0.1│
     │0.2 0.8│          │0.3 0.7│
     └──────┘          └──────┘

Initialize: m = [-∞, -∞],  l = [0, 0],  O = [[0,0],[0,0]]
```

**Inner loop j=1: K₁, V₁**

```
Step 1: S₁₁ = Q₁ @ K₁ᵀ / √2

  Q₁ @ K₁ᵀ:
    Row 0: [1.0,0.0]·[1.0,0.0]=1.0   [1.0,0.0]·[0.0,1.0]=0.0
    Row 1: [0.0,1.0]·[1.0,0.0]=0.0   [0.0,1.0]·[0.0,1.0]=1.0

  S₁₁ = ┌──────┐ / √2 = ┌────────────────┐
         │1.0 0.0│         │ 0.7071 0.0000 │
         │0.0 1.0│         │ 0.0000 0.7071 │
         └──────┘         └────────────────┘

Step 2: local_max = rowmax(S₁₁) = [0.7071, 0.7071]
Step 3: m_new = max([-∞,-∞], [0.7071,0.7071]) = [0.7071, 0.7071]
Step 4: α = exp([-∞,-∞] - [0.7071,0.7071]) = [0, 0]   (exp(-∞) = 0)

Step 5: P̃₁₁ = exp(S₁₁ - m_new)
  Row 0: exp([0.7071-0.7071, 0.0000-0.7071]) = exp([0, -0.7071]) = [1.0000, 0.4934]
  Row 1: exp([0.0000-0.7071, 0.7071-0.7071]) = exp([-0.7071, 0]) = [0.4934, 1.0000]

  P̃₁₁ = ┌──────────────┐
          │ 1.0000 0.4934 │
          │ 0.4934 1.0000 │
          └──────────────┘

Step 6: l̃₁₁ = rowsum(P̃₁₁) = [1.4934, 1.4934]
Step 7: l_new = [0,0]⊙[0,0] + [1.4934,1.4934] = [1.4934, 1.4934]

Step 8: O_new = diag([0,0])·O + P̃₁₁ @ V₁
  First term: all zeros

  P̃₁₁ @ V₁:
    Row 0: 1.0000×[0.5,0.3] + 0.4934×[0.2,0.8]
         = [0.5000,0.3000] + [0.0987,0.3947] = [0.5987, 0.6947]
    Row 1: 0.4934×[0.5,0.3] + 1.0000×[0.2,0.8]
         = [0.2467,0.1480] + [0.2000,0.8000] = [0.4467, 0.9480]

  O = ┌──────────────┐
      │ 0.5987 0.6947 │  (unnormalized)
      │ 0.4467 0.9480 │
      └──────────────┘

State after j=1:  m=[0.7071,0.7071]  l=[1.4934,1.4934]
```

**Inner loop j=2: K₂, V₂**

```
Step 1: S₁₂ = Q₁ @ K₂ᵀ / √2

  Q₁ @ K₂ᵀ:
    Row 0: [1.0,0.0]·[1.0,1.0]=1.0  [1.0,0.0]·[0.5,0.5]=0.5
    Row 1: [0.0,1.0]·[1.0,1.0]=1.0  [0.0,1.0]·[0.5,0.5]=0.5

  S₁₂ = ┌──────┐ / √2 = ┌────────────────┐
         │1.0 0.5│         │ 0.7071 0.3536 │
         │1.0 0.5│         │ 0.7071 0.3536 │
         └──────┘         └────────────────┘

Step 2: local_max = [0.7071, 0.7071]
Step 3: m_new = max([0.7071,0.7071], [0.7071,0.7071]) = [0.7071, 0.7071]  ← NO CHANGE
Step 4: α = exp([0,0]) = [1.0, 1.0]  ← No rescaling (max unchanged)

Step 5: P̃₁₂ = exp(S₁₂ - m_new)
  Row 0: exp([0, -0.3535]) = [1.0000, 0.7024]
  Row 1: exp([0, -0.3535]) = [1.0000, 0.7024]

Step 6: l̃₁₂ = rowsum(P̃₁₂) = [1.7024, 1.7024]

Step 7: l_new = [1.0,1.0]⊙[1.4934,1.4934] + [1.7024,1.7024]
              = [1.4934,1.4934] + [1.7024,1.7024] = [3.1958, 3.1958]

Step 8: O_new = diag([1.0,1.0])·O + P̃₁₂ @ V₂
  First term: O unchanged = [[0.5987,0.6947],[0.4467,0.9480]]

  P̃₁₂ @ V₂:
    Row 0: 1.0000×[0.6,0.1] + 0.7024×[0.3,0.7]
         = [0.6000,0.1000] + [0.2107,0.4917] = [0.8107, 0.5917]
    Row 1: same = [0.8107, 0.5917]

  O = [[0.5987+0.8107, 0.6947+0.5917],
       [0.4467+0.8107, 0.9480+0.5917]]
     = [[1.4094, 1.2864],
        [1.2574, 1.5397]]

State after j=2:  m=[0.7071,0.7071]  l=[3.1958,3.1958]
```

**Finalize Q₁ block:**

```
O_final[0] = [1.4094, 1.2864] / 3.1958 = [0.4411, 0.4025]
O_final[1] = [1.2574, 1.5397] / 3.1958 = [0.3934, 0.4819]

L[0] = 0.7071 + log(3.1958) = 0.7071 + 1.1625 = 1.8696
L[1] = 0.7071 + 1.1625 = 1.8696

Output block O₁ = ┌──────────────┐
                   │ 0.4411 0.4025 │  ← final output row 0
                   │ 0.3934 0.4819 │  ← final output row 1
                   └──────────────┘
```

Rows 2–3 follow the same pattern (outer loop i=2). The N×N score matrix was **never written to HBM** — it lived entirely in SRAM.

---

## 5.7 Worked Example B — Large Score Difference (Rescaling Stress Test) `[ESSENTIAL]`

This example demonstrates the critical rescaling mechanism when the maximum changes dramatically between blocks.

```
Setup: N=2, d=1, Bᵣ=2, Bᶜ=1 (one key per block)

Q = [[2.0],   K = [[1.0],   V = [[0.1],
     [1.0]]        [5.0]]        [0.9]]

S = Q @ Kᵀ  (d=1, so √d=1):
  S[0,0] = 2×1 = 2.0    S[0,1] = 2×5 = 10.0
  S[1,0] = 1×1 = 1.0    S[1,1] = 1×5 = 5.0

True attention output row 0:
  softmax([2.0, 10.0]): max=10.0
    softmax(2)  = exp(-8)  / (exp(-8)+1) ≈ 0.000335
    softmax(10) = 1        / (exp(-8)+1) ≈ 0.999665
  output = 0.000335×0.1 + 0.999665×0.9 ≈ 0.8997
```

**Process with FlashAttention (one key per block):**

```
Initialize: m=[-∞,-∞]  l=[0,0]  O=[[0],[0]]

────────────────────────────────────────────────
j=1: K₁=[[1.0]], V₁=[[0.1]]
────────────────────────────────────────────────

S₁ = Q @ K₁ᵀ = [[2.0],[1.0]]
local_max = [2.0, 1.0]
m_new = max([-∞,-∞], [2.0,1.0]) = [2.0, 1.0]
α = exp([-∞,-∞] - [2.0,1.0]) = [0, 0]

P̃₁ = exp([[2.0],[1.0]] - [[2.0],[1.0]]) = exp([[0],[0]]) = [[1.0],[1.0]]
l̃₁ = [1.0, 1.0]
l = [0,0]⊙[0,0] + [1.0,1.0] = [1.0, 1.0]
O = diag([0,0])·[[0],[0]] + [[1.0],[1.0]]·[[0.1]] = [[0.1],[0.1]]

State: m=[2.0,1.0]  l=[1.0,1.0]  O=[[0.1],[0.1]]

────────────────────────────────────────────────
j=2: K₂=[[5.0]], V₂=[[0.9]]
────────────────────────────────────────────────

S₂ = Q @ K₂ᵀ = [[10.0],[5.0]]
local_max = [10.0, 5.0]

m_old = [2.0, 1.0]
m_new = max([2.0,1.0], [10.0,5.0]) = [10.0, 5.0]

┌──────────────────────────────────────────────────────────┐
│  CRITICAL RESCALING:                                      │
│  Row 0: max changes 2.0 → 10.0  (Δ = 8.0 — massive!)   │
│  Row 1: max changes 1.0 →  5.0  (Δ = 4.0)               │
└──────────────────────────────────────────────────────────┘

α = exp(m_old - m_new)
  = exp([2.0,1.0] - [10.0,5.0])
  = exp([-8.0, -4.0])
  = [0.000335, 0.018316]

Interpretation:
  Row 0: old contribution exp(2-2)=1.0 → 1.0×0.000335 = exp(2-10) ✓
  Row 1: old contribution exp(1-1)=1.0 → 1.0×0.018316 = exp(1-5)  ✓

P̃₂ = exp([[10.0],[5.0]] - [[10.0],[5.0]]) = [[1.0],[1.0]]
l̃₂ = [1.0, 1.0]

l_new:
  Row 0: 0.000335 × 1.0 + 1.0 = 1.000335
  Row 1: 0.018316 × 1.0 + 1.0 = 1.018316

O_new = diag(α) @ O + P̃₂ @ V₂
  Row 0: 0.000335×0.1 + 1.0×0.9 = 0.0000335 + 0.9 = 0.9000335
  Row 1: 0.018316×0.1 + 1.0×0.9 = 0.0018316 + 0.9 = 0.9018316

────────────────────────────────────────────────
Finalize:
────────────────────────────────────────────────

O_final[0] = 0.9000335 / 1.000335 = 0.89973 ≈ 0.8997  ✓
O_final[1] = 0.9018316 / 1.018316 = 0.88565

VERIFY row 0 via standard attention:
  softmax([2,10]) = [0.000335, 0.999665]
  output = 0.000335×0.1 + 0.999665×0.9 = 0.89973  ✓
```

The α factor of 0.000335 — eight orders of magnitude smaller than 1 — correctly down-weights the stale contribution from the first key. Even with a score difference of 8, the result is numerically exact.

---

## 5.8 Python Reference Implementation `[CODE]`

```python
import math
import numpy as np
from typing import List, Optional

def online_softmax_1d(x: List[float]):
    """Single-pass online softmax. Returns (softmax_values, global_max, logsumexp)."""
    m = float('-inf')
    l = 0.0
    for xi in x:
        m_prev = m
        m = max(m, xi)
        alpha = math.exp(m_prev - m)
        l = l * alpha + math.exp(xi - m)
    softmax_vals = [math.exp(xi - m) / l for xi in x]
    return softmax_vals, m, m + math.log(l)

def flash_attention_numpy(Q, K, V, block_r=32, block_c=32, causal=False):
    """
    NumPy FlashAttention implementation.
    Q, K, V: (N, d) float32 arrays.
    Returns: O (N, d), L (N,) logsumexp.
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    Tr = math.ceil(N / block_r)
    Tc = math.ceil(N / block_c)

    O = np.zeros((N, d), dtype=Q.dtype)
    L = np.zeros(N, dtype=Q.dtype)

    for i in range(Tr):
        qi_s, qi_e = i * block_r, min((i+1) * block_r, N)
        Q_i = Q[qi_s:qi_e]           # (Br, d) — loaded to SRAM
        br = qi_e - qi_s

        m_i = np.full(br, -np.inf, dtype=Q.dtype)
        l_i = np.zeros(br, dtype=Q.dtype)
        O_i = np.zeros((br, d), dtype=Q.dtype)

        for j in range(Tc):
            kj_s, kj_e = j * block_c, min((j+1) * block_c, N)
            K_j = K[kj_s:kj_e]       # (Bc, d) — loaded to SRAM
            V_j = V[kj_s:kj_e]       # (Bc, d) — loaded to SRAM

            # 1. Compute block of scores
            S_ij = (Q_i @ K_j.T) * scale  # (Br, Bc)

            # 2. Apply causal mask if needed
            if causal:
                for row_idx in range(br):
                    for col_idx in range(kj_e - kj_s):
                        if qi_s + row_idx < kj_s + col_idx:
                            S_ij[row_idx, col_idx] = -np.inf

            # 3. Update row-wise maximum
            local_max = S_ij.max(axis=1)   # (Br,)
            m_new = np.maximum(m_i, local_max)

            # 4. Compute rescaling factor
            alpha = np.exp(m_i - m_new)    # (Br,)

            # 5. Unnormalized attention weights
            P_tilde = np.exp(S_ij - m_new[:, None])  # (Br, Bc)

            # 6. Update sum accumulator
            l_i_new = alpha * l_i + P_tilde.sum(axis=1)

            # 7. Update output accumulator
            # diag(alpha) @ O_i = each row i multiplied by alpha[i]
            O_i = alpha[:, None] * O_i + P_tilde @ V_j

            m_i = m_new
            l_i = l_i_new

        # Finalize: normalize output
        O[qi_s:qi_e] = O_i / l_i[:, None]
        L[qi_s:qi_e] = m_i + np.log(l_i)

    return O, L

# Verification
if __name__ == "__main__":
    np.random.seed(42)
    N, d = 16, 8
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    # Standard attention
    scale = 1.0 / math.sqrt(d)
    S = (Q @ K.T) * scale
    S -= S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=1, keepdims=True)
    O_ref = P @ V

    # FlashAttention
    O_flash, L = flash_attention_numpy(Q, K, V, block_r=4, block_c=4)

    max_err = np.abs(O_flash - O_ref).max()
    print(f"Max absolute error: {max_err:.2e}")  # Should be < 1e-6
    assert max_err < 1e-5, "FlashAttention result mismatch!"
    print("Verification passed!")
```

---

## 5.9 FlashAttention-2: Key Improvements (2023) `[PRODUCTION]`

FlashAttention-2 (Dao, 2023) achieves 2–4× speedup over FA-1 through four changes:

**1. Deferred normalization (fewer per-iteration multiply ops):**
```
FA-1: O_i = alpha * O_i + P_tilde @ V_j   ← alpha varies every iter

FA-2: O_i += rescale_factor * P_tilde @ V_j
      (defer the final 1/l normalization to the very end)
      ← eliminates a per-row vector multiply in the critical path
```

**2. Sequence-level parallelism:**
```
FA-1 grid: (batch × heads)         ← one block per (batch, head)
FA-2 grid: (batch × heads × Tr)   ← one block per (batch, head, Q-tile)
→ long sequences are fully parallelized across SMs
```

**3. Backward pass reformulation:**
```
FA-1 backward: dS = P × (dP - rowsum(dP × P))  ← two rowsum ops
FA-2: precompute D[i] = rowsum(dO[i] × O[i])   ← one precompute, then reuse
```

**4. Work partitioning within warps:**
FA-2 ensures 16×16×16 tensor core fragments are saturated by careful partitioning of the (Bᵣ, Bᶜ, d) work across warps.

---

## 5.10 FlashAttention-3: Hopper Architecture (2024) `[ADVANCED]`

FlashAttention-3 (Shah et al., 2024) exploits NVIDIA H100-specific features:

**Warpgroup Matrix Multiply (WGMMA):** Operates on 128 threads asynchronously, overlapping memory loads with computation:
```
Iteration j:  [Load K_j, V_j (async TMA)]
               ↕  overlapped!
              [Compute S_{j-1} using WGMMA]
              [Online softmax for S_{j-1}]
              [Accumulate O using P_{j-1} @ V_{j-1}]
```

**FP8 support:** H100 FP8 tensor cores give ~2× throughput vs BF16.

**Performance comparison:**
```
A100 80GB:
  Standard PyTorch attention: 35% MFU
  FlashAttention-1:           60% MFU
  FlashAttention-2:           72% MFU

H100 SXM5:
  FlashAttention-2 (A100 kernel): ~35% MFU (not H100-optimized)
  FlashAttention-3:               ~75% MFU (Hopper WGMMA + pipelining)
```

---

## 5.11 FlashDecoding — Long-Context Inference `[PRODUCTION]`

Standard FlashAttention is designed for training where Q, K, V all have length N. At inference (autoregressive decoding), Q has length 1 but K/V can be hundreds of thousands of tokens.

**The problem:**
```
Inference scenario:
  Q: (1, d)   ← new query (or small batch)
  K: (N, d)   ← full KV cache
  V: (N, d)

With standard FA: Tᵣ = ⌈1/Bᵣ⌉ = 1 CUDA block!
  → All Tᶜ K/V blocks processed SERIALLY in that one block
  → GPU utilization collapses for long contexts
```

**FlashDecoding solution:** Parallelize across the K/V dimension instead.

```
Split K and V into P groups (P = Tᶜ):
  Each of P CUDA blocks computes a partial output (O_p, m_p, l_p)
  independently using local softmax.

Final reduction step (another online softmax over P elements):
  m_global = max(m_1, m_2, ..., m_P)
  For each partial (O_p, m_p, l_p):
    α_p = exp(m_p - m_global)
    O_global += α_p * O_p * l_p
    l_global += α_p * l_p
  O_final = O_global / l_global

Grid: (num_heads × batch × P)  ← P-way parallelism over K/V
```

FlashDecoding achieves near-linear scaling with SM count for long contexts, making 128K+ context inference practical.

---

## 5.12 IO Complexity Analysis `[QUANTITATIVE]`

```
Standard Attention IO:
  Reads:  Q(Nd) + K(Nd) + S(N²) + P(N²) = 2Nd + 2N²
  Writes: S(N²) + P(N²) + O(Nd) = 2N² + Nd
  Total:  4N² + 3Nd  ≈  O(N²) for N >> d

FlashAttention IO:
  Reads:  Q(Nd) + K(Nd) × Tᵣ + V(Nd) × Tᵣ   ← K and V read Tᵣ times
        = Nd + 2Nd × ⌈N/Bᵣ⌉
        ≈ Nd × (1 + 2N/Bᵣ) = O(N²d / M)
          where M = SRAM size in floats (M ≥ Bᵣ × d)
  Writes: O(Nd) + L(N) ≈ O(Nd)
  Total:  O(N²d/M)  vs  O(N²) for standard

  For M = 192KB, d=64, BF16: M ≈ 192K/2 = 96K floats
  IO reduction factor: N² / (N²d/M) = M/d = 96K/64 = 1500×  (theoretical)
  In practice: 2–4× wall-clock speedup at N=2048 (many other bottlenecks)
```

HBM reads/writes comparison for N=4096, d=64 (FP32, one head):
```
Standard attention: ~260 MB total
FlashAttention-1:   ~40 MB total   (6.5× reduction)
FlashAttention-2:   ~36 MB total   (7.2× reduction, fewer overhead ops)
```

---

## 5.13 vLLM and llama.cpp FlashAttention Integration `[PRACTICAL]`

### 5.13.1 vLLM

vLLM selects the attention backend from an environment variable or auto-detects:

```python
# Force FlashAttention backend
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

from vllm import LLM
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)
```

For PagedAttention + FlashAttention together, vLLM uses the FlashInfer library which supports paged/non-contiguous KV caches natively:

```python
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"  # vLLM >= 0.5
```

### 5.13.2 llama.cpp

```bash
# Build with FlashAttention
cmake .. -DLLAMA_CUDA=ON -DLLAMA_FLASH_ATTN=ON
cmake --build . --config Release -j $(nproc)

# Verify FA is active
./llama-cli -m llama3-8b.gguf \
    --flash-attn \          # enable FA kernel
    -ngl 35 \               # GPU layers
    --ctx-size 16384 \
    -p "Explain attention mechanisms:"
```

In the llama.cpp source, the FA kernel is in `ggml-cuda/flash-attn.cu` and called via `ggml_flash_attn_ext()`.

---

## Chapter Summary

- **Standard attention** is memory-bandwidth-bound: it materializes an N×N score matrix to HBM, consuming O(N²) reads/writes.
- **The GPU memory hierarchy** has a 9.5× bandwidth gap between HBM (2 TB/s) and SRAM (19 TB/s). FlashAttention exploits this by keeping tiles in SRAM.
- **Online softmax** computes exact softmax in one pass using running max `m` and sum `l`. When a new maximum is found, old contributions are rescaled by `α = exp(m_old - m_new)`.
- **The correctness proof by induction** (§5.3.6) guarantees that after processing all N elements, `mₙ` is the global max and `lₙ` is the true sum — regardless of how many max updates occurred.
- **FlashAttention-1** tiles Q into Bᵣ-row blocks and K/V into Bᶜ-row blocks, maintaining per-row (m, l, O) state in SRAM. The N×N score matrix is never written to HBM.
- **Worked Example A** (N=4, d=2, Bᵣ=Bᶜ=2) traced every scalar through two inner loop iterations, demonstrating the output finalization.
- **Worked Example B** (large score difference: Δ=8) demonstrated the critical rescaling where α=0.000335 correctly down-weights a stale contribution to produce a numerically exact result.
- **FlashAttention-2** adds sequence-level parallelism and deferred normalization for 2–4× additional speedup.
- **FlashDecoding** parallelizes across K/V blocks for single-token decoding, enabling practical 128K+ context inference.

---

## Self-Check Questions

1. For N=512, d=64, FP32: compute (a) the size of the N×N attention matrix in MB, (b) the total HBM traffic for standard attention, (c) the theoretical IO reduction factor for FlashAttention with SRAM size M=192 KB.

2. In online softmax, if the input is `x = [1, 3, 2, 5, 4]`, trace through each step showing `m` and `l` after every element. Verify the final `l` equals the true sum `Σᵢ exp(xᵢ - 5)`.

3. Explain why naïve block-wise softmax (computing local softmax per block independently) gives wrong results. Give a concrete numerical example where two blocks sum to 2 instead of 1.

4. In Worked Example B, the rescaling factor `α = exp(-8) ≈ 0.000335`. If instead the score for key 1 were 20 (not 10), what would `α` be? Is this still numerically representable in FP32? (Hint: FP32 min positive normal ≈ 1.18×10⁻³⁸.)

5. FlashDecoding splits K/V into P=16 groups and runs 16 CUDA blocks in parallel. Each block computes a local `(O_p, m_p, l_p)`. Describe the reduction step: what is the formula for combining these 16 partial results into a single correct output, and why is it equivalent to a single global online softmax?

