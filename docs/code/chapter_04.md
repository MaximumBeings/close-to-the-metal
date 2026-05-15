# Chapter 4: Attention Mechanics — Companion Code

Reproduces every number from the Chapter 4 worked examples: MHA (all 10 tokens, 4 heads, seed=42), MQA, GQA, MLA, and the memory comparison table.

## Python — `attention_demo.py`

```python
# attention_demo.py
# Chapter 4 — Attention Mechanics: MHA, MQA, GQA, MLA
#
# Reproduces every number in the Chapter 4 worked examples exactly.
# Seed: numpy.random.seed(42).  Sequence: "The next day is bright and sunny with clear skies"
#
# Requirements: pip install numpy
# Run:          python attention_demo.py

import numpy as np
np.set_printoptions(precision=4, suppress=True)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

VOCAB = ["The", "next", "day", "is", "bright", "and", "sunny", "with", "clear", "skies"]
D_MODEL  = 8
N_Q_HEADS = 4
D_HEAD   = D_MODEL // N_Q_HEADS  # = 2
SCALE    = 1.0 / np.sqrt(D_HEAD)
PREFILL  = 4   # "The next day is"
DECODE   = 6   # "bright and sunny with clear skies"

rng = np.random.default_rng(42)
E = rng.standard_normal((len(VOCAB), D_MODEL)).astype(np.float32)

def make_weights(seed_offset=0):
    """Generate W_Q, W_K, W_V for all heads (each D_MODEL × D_HEAD)."""
    rng2 = np.random.default_rng(42 + seed_offset)
    WQ = rng2.standard_normal((N_Q_HEADS, D_MODEL, D_HEAD)).astype(np.float32)
    WK = rng2.standard_normal((N_Q_HEADS, D_MODEL, D_HEAD)).astype(np.float32)
    WV = rng2.standard_normal((N_Q_HEADS, D_MODEL, D_HEAD)).astype(np.float32)
    return WQ, WK, WV

WQ_MHA, WK_MHA, WV_MHA = make_weights(0)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — MHA: COMPLETE 10-TOKEN WALKTHROUGH
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1 — Multi-Head Attention (MHA): Full 10-Token Walkthrough")
print("=" * 70)

def mha_forward(embeddings, WQ, WK, WV, n_kv_heads=None, verbose=True):
    """
    Full MHA forward pass with optional verbosity.
    Returns context vectors and KV cache.
    kv_cache shape: (seq_len, n_kv_heads, 2, D_HEAD)
    """
    N = len(embeddings)
    n_kv = n_kv_heads if n_kv_heads else N_Q_HEADS
    kv_cache_k = np.zeros((N, n_kv, D_HEAD), dtype=np.float32)
    kv_cache_v = np.zeros((N, n_kv, D_HEAD), dtype=np.float32)
    context_all = np.zeros((N, N_Q_HEADS, D_HEAD), dtype=np.float32)

    for t, (word, e) in enumerate(zip(VOCAB[:N], embeddings)):
        phase = "PREFILL" if t < PREFILL else "DECODE"

        # Project K, V for this token (for each KV head)
        for h in range(n_kv):
            kv_cache_k[t, h] = e @ WK[h]
            kv_cache_v[t, h] = e @ WV[h]

        # Compute attention for each query head
        for h in range(N_Q_HEADS):
            kv_h = h % n_kv   # GQA/MQA: map query head to KV head
            q = e @ WQ[h]
            # Attend over all tokens 0..t
            scores = np.array([
                np.dot(q, kv_cache_k[j, kv_h]) * SCALE
                for j in range(t + 1)
            ])
            scores_shifted = scores - scores.max()
            weights = np.exp(scores_shifted)
            weights /= weights.sum()

            ctx = sum(weights[j] * kv_cache_v[j, kv_h] for j in range(t + 1))
            context_all[t, h] = ctx

            if verbose and h == 0 and t < 6:
                dominant = VOCAB[np.argmax(weights)]
                dominant_w = weights.max()
                print(f"\n  [{phase}] Token {t} '{word}' | Head 0:")
                print(f"    q    = {q}")
                print(f"    k    = {kv_cache_k[t, 0]}")
                print(f"    v    = {kv_cache_v[t, 0]}")
                print(f"    scores  (scaled) = {scores}")
                print(f"    softmax weights  = {weights}")
                print(f"    context          = {ctx}")
                print(f"    dominant token: '{dominant}' ({dominant_w:.4f})")

        # KV cache size
        cache_bytes = (t + 1) * n_kv * 2 * D_HEAD * 4  # FP32
        if verbose and t < 6:
            print(f"    → KV cache after '{word}': {cache_bytes} bytes (FP32, 1 layer)")

    return context_all, kv_cache_k, kv_cache_v


print("\n--- Prefill + Decode (MHA, Head 0 detail) ---")
ctx_mha, k_cache, v_cache = mha_forward(E, WQ_MHA, WK_MHA, WV_MHA)

print("\n\nAll-heads context summary (last 4 tokens):")
for t in range(PREFILL, PREFILL + 4):
    word = VOCAB[t]
    for h in range(N_Q_HEADS):
        print(f"  Token '{word}' Head {h}: ctx={ctx_mha[t,h]}")

# Memory analysis
print("\n--- MHA Memory Analysis ---")
for n in [4, 5, 6, 10]:
    b = 2 * n * N_Q_HEADS * D_HEAD * 4
    print(f"  After {n:2d} tokens: {b} bytes (FP32, 1 layer)")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — MQA: SINGLE SHARED KV HEAD
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2 — Multi-Query Attention (MQA): 1 shared KV head")
print("=" * 70)

# MQA: only 1 KV head — use Head 0's W_K and W_V from MHA
WK_MQA = WK_MHA[:1]  # shape (1, D_MODEL, D_HEAD)
WV_MQA = WV_MHA[:1]

ctx_mqa, k_mqa, v_mqa = mha_forward(E, WQ_MHA, WK_MQA, WV_MQA, n_kv_heads=1, verbose=False)

print("\n--- Decode Step 1 'bright' — All heads attend same K/V ---")
t = PREFILL  # 'bright'
for h in range(N_Q_HEADS):
    q = E[t] @ WQ_MHA[h]
    scores = np.array([np.dot(q, k_mqa[j, 0]) * SCALE for j in range(t + 1)])
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    dominant = VOCAB[:t+1][np.argmax(weights)]
    print(f"  Head {h}: q={q}  dominant='{dominant}' ({weights.max():.4f})")

print("\n--- Memory comparison at 5 tokens (FP32, 1 layer) ---")
print(f"  MHA: {2*5*N_Q_HEADS*D_HEAD*4} bytes")
print(f"  MQA: {2*5*1*D_HEAD*4} bytes  (4× reduction)")

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — GQA: 2 KV HEADS, GROUPS OF 2
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3 — Grouped-Query Attention (GQA): 2 KV heads")
print("=" * 70)

N_KV_GQA = 2
WK_GQA = WK_MHA[:N_KV_GQA]
WV_GQA = WV_MHA[:N_KV_GQA]

ctx_gqa, k_gqa, v_gqa = mha_forward(E, WQ_MHA, WK_GQA, WV_GQA, n_kv_heads=N_KV_GQA, verbose=False)

print("\n--- Decode Step 1 'bright' — Group assignments ---")
print("  Q heads 0,1 → KV head 0   |   Q heads 2,3 → KV head 1")
t = PREFILL
for h in range(N_Q_HEADS):
    kv_h = h % N_KV_GQA
    q = E[t] @ WQ_MHA[h]
    scores = np.array([np.dot(q, k_gqa[j, kv_h]) * SCALE for j in range(t + 1)])
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    dominant = VOCAB[:t+1][np.argmax(weights)]
    print(f"  Head {h} → KV head {kv_h}: dominant='{dominant}' ({weights.max():.4f})")

print("\n--- Memory comparison at 5 tokens (FP32, 1 layer) ---")
print(f"  MHA: {2*5*N_Q_HEADS*D_HEAD*4} bytes")
print(f"  GQA: {2*5*N_KV_GQA*D_HEAD*4} bytes  (2× reduction)")
print(f"  MQA: {2*5*1*D_HEAD*4} bytes  (4× reduction)")

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — MLA: LOW-RANK LATENT COMPRESSION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4 — Multi-head Latent Attention (MLA): latent_dim=3")
print("=" * 70)

LATENT_DIM = 3
rng3 = np.random.default_rng(99)
W_DKV = rng3.standard_normal((D_MODEL, LATENT_DIM)).astype(np.float32)
W_UK  = rng3.standard_normal((LATENT_DIM, N_Q_HEADS * D_HEAD)).astype(np.float32)
W_UV  = rng3.standard_normal((LATENT_DIM, N_Q_HEADS * D_HEAD)).astype(np.float32)

# Build MLA KV cache
N = len(VOCAB)
C_KV_cache = np.zeros((N, LATENT_DIM), dtype=np.float32)

for t in range(N):
    C_KV_cache[t] = E[t] @ W_DKV

print("\n--- Latent vectors (cache entries) for first 5 tokens ---")
for t in range(5):
    print(f"  '{VOCAB[t]}': C_KV = {C_KV_cache[t]}")

print("\n--- Reconstructed K/V for 'The' ---")
K_reconstructed = (C_KV_cache[0] @ W_UK).reshape(N_Q_HEADS, D_HEAD)
V_reconstructed = (C_KV_cache[0] @ W_UV).reshape(N_Q_HEADS, D_HEAD)
for h in range(N_Q_HEADS):
    print(f"  Head {h}: K={K_reconstructed[h]}  V={V_reconstructed[h]}")

print("\n--- Memory comparison at 10 tokens (FP32, 1 layer) ---")
print(f"  MHA: {2*10*N_Q_HEADS*D_HEAD*4} bytes")
print(f"  GQA: {2*10*N_KV_GQA*D_HEAD*4} bytes")
print(f"  MQA: {2*10*1*D_HEAD*4} bytes")
print(f"  MLA: {10*LATENT_DIM*4} bytes  ({2*10*N_Q_HEADS*D_HEAD*4 // (10*LATENT_DIM*4):.0f}× reduction vs MHA)")

# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — VARIANT COMPARISON ACROSS ALL DECODE STEPS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5 — Complete Memory Growth Comparison")
print("=" * 70)

print(f"\n{'Tokens':>8} {'MHA bytes':>12} {'GQA bytes':>12} {'MQA bytes':>12} {'MLA bytes':>12}")
print("-" * 56)
for n in range(1, 11):
    mha_b  = 2 * n * N_Q_HEADS * D_HEAD * 4
    gqa_b  = 2 * n * N_KV_GQA * D_HEAD * 4
    mqa_b  = 2 * n * 1 * D_HEAD * 4
    mla_b  = n * LATENT_DIM * 4
    label  = VOCAB[n-1]
    print(f"  {n:2d} {label:<6}  {mha_b:>10}    {gqa_b:>10}    {mqa_b:>10}    {mla_b:>10}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — REAL-MODEL SCALE PROJECTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 6 — Real Model Scale (LLaMA-3 8B, BF16)")
print("=" * 70)

models = [
    ("LLaMA-3 8B  (GQA)",  32, 8,  128, 32, 2),
    ("LLaMA-3 70B (GQA)",  64, 8,  128, 80, 2),
    ("DeepSeek-V2 (MLA)",  128, None, None, 60, 2),  # latent=512
]
ctx_lengths = [4096, 32768, 131072]

for name, n_q, n_kv, d_h, n_layers, bpe in models:
    print(f"\n  {name}:")
    for ctx in ctx_lengths:
        if n_kv is not None:
            total = 2 * ctx * n_kv * d_h * n_layers * bpe
        else:
            # DeepSeek MLA: latent_dim=512
            total = ctx * 512 * n_layers * bpe
        print(f"    {ctx//1024:3d}K ctx: {total/1e9:.2f} GB")

print("\n✓ All Chapter 4 worked examples reproduced.")
print("  Every number traces back to the seed-42 embedding table in §4.3.")
```

## C++ — `attention_demo.cpp`

Compile: `g++ -std=c++17 -O2 -o attention_demo attention_demo.cpp -lm`

```cpp
// attention_demo.cpp
// Chapter 4 — Attention Mechanics: MHA, MQA, GQA, MLA
//
// Reproduces the core arithmetic from the Chapter 4 worked examples.
// Uses the same (D_MODEL=8, N_Q_HEADS=4, D_HEAD=2) toy model.
//
// Compile: g++ -std=c++17 -O2 -o attention_demo attention_demo.cpp
// Run:     ./attention_demo

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <random>

// ── Config ──────────────────────────────────────────────────────────────────
static constexpr int D_MODEL   = 8;
static constexpr int N_Q_HEADS = 4;
static constexpr int D_HEAD    = D_MODEL / N_Q_HEADS;   // = 2
static constexpr int N_TOKENS  = 10;
static constexpr int PREFILL   = 4;
static const float   SCALE     = 1.0f / std::sqrt((float)D_HEAD);

using Vec  = std::vector<float>;
using Mat  = std::vector<Vec>;

// ── Helpers ─────────────────────────────────────────────────────────────────

float dot(const Vec& a, const Vec& b) {
    float s = 0;
    for (int i = 0; i < (int)a.size(); ++i) s += a[i] * b[i];
    return s;
}

Vec mat_vec(const Mat& W, const Vec& x) {
    // W is (out_dim × in_dim), x is (in_dim,)
    Vec out(W.size(), 0.0f);
    for (int i = 0; i < (int)W.size(); ++i)
        for (int j = 0; j < (int)x.size(); ++j)
            out[i] += W[i][j] * x[j];
    return out;
}

// Transpose multiply: W is (in_dim × out_dim), x is (in_dim,) → (out_dim,)
Vec mat_vec_T(const Mat& W, const Vec& x) {
    int out_dim = W[0].size();
    Vec out(out_dim, 0.0f);
    for (int i = 0; i < (int)W.size(); ++i)
        for (int j = 0; j < out_dim; ++j)
            out[j] += W[i][j] * x[i];
    return out;
}

Vec softmax(const Vec& x) {
    float m = *std::max_element(x.begin(), x.end());
    Vec e(x.size());
    float s = 0;
    for (int i = 0; i < (int)x.size(); ++i) { e[i] = std::exp(x[i] - m); s += e[i]; }
    for (auto& v : e) v /= s;
    return e;
}

Vec vec_scale(const Vec& a, float s) {
    Vec b(a.size());
    for (int i = 0; i < (int)a.size(); ++i) b[i] = a[i] * s;
    return b;
}

Vec vec_add(const Vec& a, const Vec& b) {
    Vec c(a.size());
    for (int i = 0; i < (int)a.size(); ++i) c[i] = a[i] + b[i];
    return c;
}

void print_vec(const std::string& label, const Vec& v) {
    std::cout << "    " << label << " = [";
    for (int i = 0; i < (int)v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << v[i];
        if (i + 1 < (int)v.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

// ── Deterministic weight generation (lcg-based, matches Python seed=42 spirit)
// (exact seed match would need numpy; these are pedagogically equivalent)
Mat make_weight_matrix(int rows, int cols, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Mat W(rows, Vec(cols));
    for (auto& row : W)
        for (auto& v : row) v = dist(rng);
    return W;
}

// ── Embedding table (10 × 8, deterministic)
Mat make_embeddings() {
    return make_weight_matrix(N_TOKENS, D_MODEL, 42);
}

// ── Weight matrices for MHA (4 heads, D_MODEL×D_HEAD each)
struct Weights {
    std::vector<Mat> WQ, WK, WV;  // [n_heads][D_MODEL][D_HEAD]
};

Weights make_weights_mha(int seed_offset = 0) {
    Weights w;
    for (int h = 0; h < N_Q_HEADS; ++h) {
        w.WQ.push_back(make_weight_matrix(D_MODEL, D_HEAD, 100 + h + seed_offset));
        w.WK.push_back(make_weight_matrix(D_MODEL, D_HEAD, 200 + h + seed_offset));
        w.WV.push_back(make_weight_matrix(D_MODEL, D_HEAD, 300 + h + seed_offset));
    }
    return w;
}

// ── KV Cache ──────────────────────────────────────────────────────────────

struct KVCache {
    // [token][head][d_head]
    std::vector<std::vector<Vec>> K, V;
    int n_kv_heads;

    explicit KVCache(int n_kv) : n_kv_heads(n_kv) {}

    void append(const std::vector<Vec>& k_vec, const std::vector<Vec>& v_vec) {
        K.push_back(k_vec);
        V.push_back(v_vec);
    }

    int size_bytes() const {
        return (int)K.size() * n_kv_heads * D_HEAD * 2 * 4;  // K+V, FP32
    }
};

// ══════════════════════════════════════════════════════════════════════════════
// MHA Forward Pass
// ══════════════════════════════════════════════════════════════════════════════

void run_mha(const Mat& E, const Weights& W, int n_kv_heads, bool verbose) {
    const std::string vocab[] = {"The","next","day","is","bright","and","sunny","with","clear","skies"};
    KVCache cache(n_kv_heads);

    for (int t = 0; t < N_TOKENS; ++t) {
        const Vec& e = E[t];
        bool is_prefill = t < PREFILL;

        // Project K, V for KV heads
        std::vector<Vec> k_new(n_kv_heads), v_new(n_kv_heads);
        for (int h = 0; h < n_kv_heads; ++h) {
            k_new[h] = mat_vec_T(W.WK[h], e);
            v_new[h] = mat_vec_T(W.WV[h], e);
        }
        cache.append(k_new, v_new);

        // Compute attention for each Q head
        for (int h = 0; h < N_Q_HEADS; ++h) {
            int kv_h = h % n_kv_heads;
            Vec q = mat_vec_T(W.WQ[h], e);

            Vec scores;
            for (int j = 0; j <= t; ++j)
                scores.push_back(dot(q, cache.K[j][kv_h]) * SCALE);

            Vec weights = softmax(scores);

            Vec ctx(D_HEAD, 0.0f);
            for (int j = 0; j <= t; ++j)
                ctx = vec_add(ctx, vec_scale(cache.V[j][kv_h], weights[j]));

            if (verbose && h == 0 && t < 6) {
                int dom = (int)(std::max_element(weights.begin(), weights.end()) - weights.begin());
                std::string phase = is_prefill ? "PREFILL" : "DECODE ";
                std::cout << "\n  [" << phase << "] Token " << t << " '" << vocab[t] << "' | Head 0:\n";
                print_vec("q    ", q);
                print_vec("k    ", k_new[0]);
                print_vec("v    ", v_new[0]);
                print_vec("ctx  ", ctx);
                std::cout << "    dominant: '" << vocab[dom] << "' (" << std::fixed
                          << std::setprecision(4) << weights[dom] << ")\n";
                std::cout << "    KV cache: " << cache.size_bytes() << " bytes (FP32, 1 layer)\n";
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// MLA Forward Pass
// ══════════════════════════════════════════════════════════════════════════════

void run_mla(const Mat& E) {
    constexpr int LATENT_DIM = 3;
    auto W_DKV = make_weight_matrix(D_MODEL, LATENT_DIM, 999);
    auto W_UK  = make_weight_matrix(LATENT_DIM, N_Q_HEADS * D_HEAD, 1000);
    auto W_UV  = make_weight_matrix(LATENT_DIM, N_Q_HEADS * D_HEAD, 1001);

    const std::string vocab[] = {"The","next","day","is","bright","and","sunny","with","clear","skies"};

    std::cout << "\n--- MLA: Latent vectors (3-float cache entries) ---\n";
    for (int t = 0; t < 5; ++t) {
        Vec c(LATENT_DIM, 0);
        for (int j = 0; j < D_MODEL; ++j)
            for (int k = 0; k < LATENT_DIM; ++k)
                c[k] += E[t][j] * W_DKV[j][k];
        print_vec(vocab[t], c);
    }

    std::cout << "\n--- MLA memory comparison at 10 tokens (FP32, 1 layer) ---\n";
    auto b = [](int n, int h, int d) { return 2*n*h*d*4; };
    std::cout << "  MHA : " << b(10, N_Q_HEADS, D_HEAD) << " bytes\n";
    std::cout << "  GQA : " << b(10, 2, D_HEAD) << " bytes\n";
    std::cout << "  MQA : " << b(10, 1, D_HEAD) << " bytes\n";
    std::cout << "  MLA : " << 10*LATENT_DIM*4 << " bytes  ("
              << b(10, N_Q_HEADS, D_HEAD) / (10*LATENT_DIM*4) << "× reduction vs MHA)\n";
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Chapter 4 — Attention Mechanics Demo (C++)\n";
    std::cout << "Config: D_MODEL=" << D_MODEL << " N_Q_HEADS=" << N_Q_HEADS
              << " D_HEAD=" << D_HEAD << " SCALE=" << SCALE << "\n";
    std::cout << std::string(70, '=') << "\n";

    Mat E = make_embeddings();
    Weights W_MHA = make_weights_mha(0);

    // ── Part 1: MHA
    std::cout << "\nPART 1 — MHA (4 KV heads): first 6 tokens, Head 0\n";
    run_mha(E, W_MHA, N_Q_HEADS, /*verbose=*/true);

    // ── Part 2: MQA
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "PART 2 — MQA (1 shared KV head): first 6 tokens, Head 0\n";
    std::cout << std::string(70, '=') << "\n";
    run_mha(E, W_MHA, 1, /*verbose=*/true);

    // ── Part 3: GQA
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "PART 3 — GQA (2 KV heads): first 6 tokens, Head 0\n";
    std::cout << std::string(70, '=') << "\n";
    run_mha(E, W_MHA, 2, /*verbose=*/true);

    // ── Part 4: MLA
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "PART 4 — MLA (latent_dim=3)\n";
    std::cout << std::string(70, '=') << "\n";
    run_mla(E);

    // ── Part 5: Memory growth table
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "PART 5 — Memory Growth Comparison (FP32, 1 layer)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::setw(8) << "Tokens"
              << std::setw(12) << "MHA"
              << std::setw(12) << "GQA"
              << std::setw(12) << "MQA"
              << std::setw(12) << "MLA" << "\n";
    std::cout << std::string(56, '-') << "\n";
    for (int n = 1; n <= N_TOKENS; ++n) {
        int mha = 2*n*N_Q_HEADS*D_HEAD*4;
        int gqa = 2*n*2*D_HEAD*4;
        int mqa = 2*n*1*D_HEAD*4;
        int mla = n*3*4;
        std::cout << std::setw(8) << n
                  << std::setw(12) << mha
                  << std::setw(12) << gqa
                  << std::setw(12) << mqa
                  << std::setw(12) << mla << "\n";
    }

    std::cout << "\n✓ All Chapter 4 attention variants demonstrated.\n";
    return 0;
}
```
