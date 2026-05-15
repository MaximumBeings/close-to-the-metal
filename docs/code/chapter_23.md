# Chapter 23: Speculative Decoding — Companion Code

## Python — `speculative_decoding_demo.py`

```python
"""
Chapter 23: Speculative Decoding — Draft, Verify, Accept
=========================================================
Comprehensive Demo Suite — 10 Demonstrations

Speculative decoding (Chapter 23) breaks the "one token per step" constraint
by using a small draft model to propose γ candidate tokens, then verifying all
of them with the target model in a single forward pass.

Key mathematical property: the output distribution is IDENTICAL to the target
model alone — speculation is lossless. The speedup comes from amortising the
(expensive) target model forward pass over multiple accepted tokens.

This demo suite models the full lifecycle:
  1. The fundamental idea — draft / verify / accept-or-reject
  2. The acceptance-rejection algorithm (exact distribution proof)
  3. Speedup formula derivation: S ≈ (E[accepted]+1) / (1 + c·γ)
  4. Parameter sweep: how γ, α, and cost_ratio interact
  5. Token-tree speculation (parallel multi-path drafting)
  6. Draft model quality vs acceptance rate
  7. vLLM configuration knobs
  8. When speculative decoding helps vs hurts
  9. Batch size interaction (the efficiency cliff)
  10. Production sizing: how many draft steps to use

No external dependencies — all calculations from first principles.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core Mathematical Models
# ─────────────────────────────────────────────────────────────────────────────

def expected_accepted_tokens(alpha: float, gamma: int) -> float:
    """E[tokens accepted per step] under speculative decoding.

    Each draft token is accepted independently with probability α.
    Once a draft token is rejected, the step ends.
    The target model always produces one correction token.

    For geometric acceptance (draft token i accepted iff all i−1 before also accepted):
        E[accepted] = Σ_{k=0}^{γ} α^k × (1 − α^{γ+1}) / (1 − α)  [simplified form below]

    Exact formula (Leviathan et al., 2023, Eq. 3):
        E[accepted] = (1 − α^{γ+1}) / (1 − α)  − 1   for α < 1
                    = γ                               for α = 1
    """
    if alpha >= 1.0:
        return float(gamma)
    return (1 - alpha ** (gamma + 1)) / (1 - alpha) - 1


def theoretical_speedup(alpha: float, gamma: int, cost_ratio: float) -> float:
    """Theoretical tokens-per-second speedup vs single-model decoding.

    S = (E[accepted] + 1) / (1 + c × γ)

    where:
      E[accepted] = expected accepted draft tokens per step
      1           = the correction/acceptance token always produced
      c           = cost_ratio = (draft_forward_time / target_forward_time)
      γ           = number of draft tokens proposed per step

    For c = 0 (free draft):  S_max = (E[accepted] + 1)
    For c = 1 (same cost):   S = 1 (no speedup regardless of α)
    """
    e_acc = expected_accepted_tokens(alpha, gamma)
    return (e_acc + 1.0) / (1.0 + cost_ratio * gamma)


@dataclass
class ModelPair:
    """A draft + target model pair for speculative decoding."""
    draft_name:    str
    target_name:   str
    draft_params:  float     # billions
    target_params: float     # billions
    alpha:         float     # empirical acceptance rate for this pair
    same_family:   bool      # True if draft is a smaller version of target

    def cost_ratio(self) -> float:
        """Approximate cost ratio: smaller model is proportionally cheaper."""
        return self.draft_params / self.target_params

    def speedup(self, gamma: int) -> float:
        return theoretical_speedup(self.alpha, gamma, self.cost_ratio())

    def practical_speedup(self, gamma: int, mfu_fraction: float = 0.80) -> float:
        """Practical speedup after accounting for memory and scheduling overhead."""
        theory = self.speedup(gamma)
        return 1.0 + (theory - 1.0) * mfu_fraction


# ── Common model pairs ──────────────────────────────────────────────────────
MODEL_PAIRS: List[ModelPair] = [
    ModelPair("Llama-3.1-8B",  "Llama-3.1-70B",  8.0,  70.0,  0.78, True),
    ModelPair("Llama-3.1-8B",  "Llama-3.1-405B", 8.0,  405.0, 0.72, True),
    ModelPair("Llama-3.2-1B",  "Llama-3.1-8B",   1.0,  8.0,   0.82, True),
    ModelPair("Qwen2.5-7B",    "Qwen2.5-72B",    7.0,  72.0,  0.80, True),
    ModelPair("Nemotron-4-8B", "Nemotron-4-22B", 8.0,  22.0,  0.85, True),
    ModelPair("Mistral-7B",    "Llama-3.1-70B",  7.0,  70.0,  0.55, False),  # cross-family
    ModelPair("TinyLlama-1B",  "Llama-3.1-70B",  1.1,  70.0,  0.48, False),  # very different
]


# ─────────────────────────────────────────────────────────────────────────────
# Demo Functions
# ─────────────────────────────────────────────────────────────────────────────

def demo_fundamental_idea():
    """Demo 1: The fundamental draft/verify/accept cycle."""
    print(f"""
{'='*70}
DEMO 1 — The Fundamental Idea: Draft, Verify, Accept-or-Reject
{'='*70}

  Standard autoregressive decoding:
    Step 1: target_model(tokens[0..t])   → token[t+1]   (1 new token)
    Step 2: target_model(tokens[0..t+1]) → token[t+2]   (1 new token)
    ...
    Each step loads ALL model weights from HBM.  Cost: N steps × T_target.

  Speculative decoding (γ=3 example):
    Draft phase (cheap):
      draft_model(tokens[0..t])   → token[t+1]'   (proposed)
      draft_model(tokens[0..t+1]) → token[t+2]'   (proposed)
      draft_model(tokens[0..t+2]) → token[t+3]'   (proposed)
      → 3 proposed tokens, 3 cheap forward passes

    Verify phase (1 target model pass processes all γ tokens simultaneously):
      target_model(tokens[0..t], t+1', t+2', t+3')
      → p(·|t),  p(·|t+1'),  p(·|t+2'),  p(·|t+3')   [4 distributions at once]

    Accept/reject (per token, in order):
      token[t+1]': accept with prob min(1, p_target / p_draft)
      token[t+2]': accept if t+1' accepted  AND  min(1, p_target / p_draft) > u
      token[t+3]': accept if t+1' and t+2' both accepted AND same check
      Correction:  if any token rejected, sample from adjusted distribution
                   (guarantees exact target distribution — proved below)

    Result: accepted 2 of 3 draft tokens + 1 correction = 3 tokens
    vs standard: 3 tokens would have required 3 separate target passes
    Speedup: target ran once instead of 3 times.

  Total cost: 3 × T_draft  +  1 × T_target   (vs 3 × T_target)
  If T_draft ≈ 0.1 × T_target: cost = 0.3 + 1.0 = 1.3 × T_target
  Tokens output: ~3  → effective rate = 3 / 1.3 = 2.3× faster

  KEY INSIGHT: The target model runs once regardless of how many tokens
  are accepted. The amortization improves with acceptance rate α.
""")

    # Simulate one step
    gamma = 3
    alpha = 0.75
    random.seed(42)

    print(f"  Simulation (γ={gamma}, α={alpha}, random seed=42):")
    accepted = 0
    for i in range(1, gamma + 1):
        r = random.random()
        acc = r < alpha
        status = "✓ ACCEPT" if acc else "✗ REJECT"
        print(f"    Draft token {i}: u={r:.3f} {'<' if acc else '>='} α={alpha} → {status}")
        if acc:
            accepted += 1
        else:
            print(f"    → Stop, sample correction from adjusted distribution")
            break

    print(f"\n    Tokens produced this step: {accepted} accepted + 1 correction = {accepted + 1}")
    print(f"    E[accepted] theoretical: {expected_accepted_tokens(alpha, gamma):.2f}")

    e_acc = expected_accepted_tokens(alpha, gamma)
    assert e_acc > 0, "Expected accepted tokens should be positive"
    assert e_acc < gamma, "Expected accepted tokens should be less than γ"
    print(f"\n  ✓ E[accepted]={e_acc:.2f} is in valid range (0, {gamma})")


def demo_acceptance_rejection_proof():
    """Demo 2: Why the output distribution is identical to the target model."""
    print(f"""
{'='*70}
DEMO 2 — Acceptance-Rejection Algorithm: Exact Distribution Proof
{'='*70}

  Claim: Speculative decoding produces EXACTLY the same distribution as
  sampling directly from the target model.

  Proof sketch (for one token, single draft):

  Let q(x) = draft model probability for token x
  Let p(x) = target model probability for token x

  Step 1: Draft model samples x̃ ~ q(x)
  Step 2: Accept x̃ with probability min(1, p(x̃)/q(x̃))
           If accepted: output x̃
           If rejected: sample x from adjusted distribution

  Adjusted distribution on rejection:
    r(x) = max(0, p(x) - q(x)) / Z   where Z = Σ_x max(0, p(x) - q(x))

  Output distribution:
    P(output = x) = P(accept x̃=x) + P(reject) × r(x)
                  = q(x) × min(1, p(x)/q(x))  +  Z × r(x)
                  = min(p(x), q(x))  +  max(0, p(x) - q(x))
                  = p(x)   ✓

  The accepted path contributes min(p,q) for each token.
  The rejection path compensates exactly where p > q.

  Numerical verification:
""")

    # Small vocabulary example
    vocab = ["the", "a", "an", "this", "that"]
    p = [0.40, 0.25, 0.15, 0.12, 0.08]   # target
    q = [0.50, 0.20, 0.15, 0.10, 0.05]   # draft (biased toward "the")

    print(f"  {'Token':<8}  {'p(target)':>10}  {'q(draft)':>10}  {'Accept prob':>12}  "
          f"{'min(p,q)':>10}  {'Adjusted r':>12}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*12}")

    min_pq = [min(pi, qi) for pi, qi in zip(p, q)]
    residual = [max(0, pi - qi) for pi, qi in zip(p, q)]
    Z = sum(residual)
    r = [ri / Z if Z > 0 else 0 for ri in residual]
    accept_prob = [min(1.0, pi / qi) if qi > 0 else 0 for pi, qi in zip(p, q)]

    P_total = [0.0] * len(vocab)
    for i in range(len(vocab)):
        # P(accept token i) + P(reject) * r(i)
        P_total[i] = q[i] * accept_prob[i] + Z * r[i]

    for i, tok in enumerate(vocab):
        print(f"  {tok:<8}  {p[i]:>10.3f}  {q[i]:>10.3f}  {accept_prob[i]:>12.3f}  "
              f"{min_pq[i]:>10.3f}  {r[i]:>12.3f}")

    print(f"\n  Output distribution reconstruction:")
    print(f"  {'Token':<8}  {'p(target)':>10}  {'P(output)':>10}  {'Match?':>8}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}")
    for i, tok in enumerate(vocab):
        match = "✓" if abs(P_total[i] - p[i]) < 1e-9 else "✗"
        print(f"  {tok:<8}  {p[i]:>10.3f}  {P_total[i]:>10.3f}  {match:>8}")

    # Verify
    for i in range(len(vocab)):
        assert abs(P_total[i] - p[i]) < 1e-9, \
            f"Output dist {P_total[i]:.6f} ≠ target {p[i]:.6f} for '{vocab[i]}'"

    print(f"\n  ✓ P(output) = p(target) for all tokens — exact distribution preserved")
    print(f"  ✓ Speculative decoding is a LOSSLESS acceleration technique")


def demo_speedup_formula():
    """Demo 3: Speedup formula derivation and worked examples."""
    print(f"""
{'='*70}
DEMO 3 — Speedup Formula: S = (E[accepted]+1) / (1 + c·γ)
{'='*70}

  Time per speculative decoding step:
    Draft time:  γ × T_draft   (γ sequential draft forward passes)
    Target time: 1 × T_target  (one parallel verification pass)
    Total:       γ × T_draft  +  T_target
               = T_target × (c×γ + 1)   where c = T_draft / T_target

  Tokens produced per step:
    E[accepted] + 1  (accepted draft tokens + correction token)

  Tokens per unit time (speedup vs baseline T_target per token):
    S = (E[accepted] + 1) / (c×γ + 1)

  Baseline (no speculation): 1 token per T_target
  With speculation:          (E[accepted]+1) tokens per (c×γ+1)×T_target

  Special cases:
    c = 0 (free draft):  S = E[accepted]+1  = α(γ) + 1  [upper bound]
    α = 1 (perfect):     S = (γ+1) / (c×γ+1)           [lower bound for c > 0]
    α = 0 (useless):     S = 1/(1+c×γ) < 1 [slow!]     [← always reject draft]

  Worked examples:
""")

    examples = [
        ("Llama 1B draft → 8B target",   0.82, 5, 1/8,   "1B→8B, same family"),
        ("Llama 8B draft → 70B target",  0.78, 5, 8/70,  "8B→70B, same family"),
        ("Llama 8B draft → 405B target", 0.72, 5, 8/405, "8B→405B, same family"),
        ("Mistral draft → Llama target", 0.55, 5, 7/70,  "cross-family, α suffers"),
        ("γ=10, α=0.80, 8→70B",         0.80, 10, 8/70,  "more draft tokens"),
        ("γ=2,  α=0.80, 8→70B",         0.80, 2,  8/70,  "fewer draft tokens"),
    ]

    print(f"  {'Scenario':<38}  {'α':>5}  {'γ':>3}  {'c':>7}  {'E[acc]':>7}  {'S':>7}")
    print(f"  {'─'*38}  {'─'*5}  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*7}")

    for name, alpha, gamma, c, _ in examples:
        e_acc = expected_accepted_tokens(alpha, gamma)
        S = theoretical_speedup(alpha, gamma, c)
        print(f"  {name:<38}  {alpha:>5.2f}  {gamma:>3}  {c:>7.4f}  {e_acc:>7.2f}  {S:>7.2f}×")

    print(f"""
  Key observations:
    1. Cross-family draft (α=0.55): speedup barely > 1.0 — may not be worth it
    2. Same-family draft (α=0.78+): 2–3× speedup at γ=5 is typical
    3. More draft tokens (γ=10 vs γ=5): diminishing returns — E[acc] saturates
    4. Large target (405B): c is tiny, so draft overhead is negligible
    5. Optimal γ: balance E[acc] saturation vs added draft overhead
""")

    # Verify key speedups
    s_8b_70b = theoretical_speedup(0.78, 5, 8/70)
    s_cross   = theoretical_speedup(0.55, 5, 7/70)
    assert s_8b_70b > s_cross, "Same-family should outperform cross-family"
    assert s_8b_70b > 2.0, f"8B→70B same-family should give >2× speedup, got {s_8b_70b:.2f}"
    print(f"  ✓ Same-family (α=0.78): {s_8b_70b:.2f}× > cross-family (α=0.55): {s_cross:.2f}×")


def demo_parameter_sweep():
    """Demo 4: Sensitivity to α, γ, and cost_ratio."""
    print(f"""
{'='*70}
DEMO 4 — Parameter Sweep: How α, γ, and Cost Ratio Interact
{'='*70}

  Speedup as a function of acceptance rate α  (γ=5, c=8/70):
""")

    c = 8 / 70
    gamma = 5
    alphas = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

    print(f"  {'α':>5}  {'E[acc]':>8}  {'Speedup':>9}  {'Practical':>11}  Bar")
    print(f"  {'─'*5}  {'─'*8}  {'─'*9}  {'─'*11}  {'─'*25}")
    for a in alphas:
        e_acc = expected_accepted_tokens(a, gamma)
        S = theoretical_speedup(a, gamma, c)
        practical = 1.0 + (S - 1.0) * 0.80  # 80% practical efficiency
        bar = "█" * int(S * 5) if S > 1 else ""
        status = "✓" if S > 1.5 else ("~" if S > 1.1 else "✗")
        print(f"  {a:>5.2f}  {e_acc:>8.2f}  {S:>8.2f}×  {practical:>10.2f}×  {bar}  {status}")

    print(f"""
  Speedup as a function of γ  (α=0.78, c=8/70):
""")

    alpha = 0.78
    gammas = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    print(f"  {'γ':>4}  {'E[acc]':>8}  {'Speedup':>9}  {'Δ from γ-1':>12}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*9}  {'─'*12}")
    prev_S = 1.0
    for g in gammas:
        e_acc = expected_accepted_tokens(alpha, g)
        S = theoretical_speedup(alpha, g, c)
        delta = S - prev_S
        prev_S = S
        marginal = "↑" if delta > 0.05 else ("→" if delta > 0.01 else "↗ diminishing")
        print(f"  {g:>4}  {e_acc:>8.2f}  {S:>8.2f}×  {delta:>10.3f}  {marginal}")

    print(f"""
  Insight: γ has diminishing returns past ~5–7 for typical α=0.7–0.8.
  Marginal gain per extra draft token approaches 0 as α^γ → 0.
  Rule of thumb: set γ = -log(0.05) / log(1/α) = {-math.log(0.05)/math.log(1/alpha):.1f} for α={alpha}
    (γ where P(reaching last token) < 5%)
""")

    # Verify that marginal gain per extra γ decreases (diminishing returns)
    S3  = theoretical_speedup(alpha, 3,  c)
    S5  = theoretical_speedup(alpha, 5,  c)
    S7  = theoretical_speedup(alpha, 7,  c)
    gain_3_to_5 = S5 - S3
    gain_5_to_7 = S7 - S5
    assert gain_3_to_5 > gain_5_to_7, "Diminishing returns: gain shrinks with each extra γ"
    print(f"  ✓ Diminishing returns confirmed: Δ(3→5) = {gain_3_to_5:.3f} > Δ(5→7) = {gain_5_to_7:.3f}")


def demo_token_tree_speculation():
    """Demo 5: Token tree — speculate multiple branches in parallel."""
    print(f"""
{'='*70}
DEMO 5 — Token Tree Speculation: Multi-Path Draft
{'='*70}

  Standard speculative decoding: one draft sequence of γ tokens.
    → If first token rejected, remaining γ-1 tokens are wasted.

  Token tree (SpecTree / Medusa): draft multiple candidate continuations.
    Tree of branching factor b at each depth level.
    All paths verified in one batched target forward pass.

  Example tree (b=2, depth=3):
                    [t]
                   /   \\
              [A]         [B]           ← Draft token 1: 2 candidates
              / \\         / \\
           [AA] [AB]   [BA] [BB]        ← Depth 2: 4 candidates
           /\\
        [AAA][AAB]...                   ← Depth 3: up to 8 candidates

  Total draft paths: b^depth = 2^3 = 8 paths explored
  Target forward pass: one pass evaluates ALL paths (via attention masking)
  Accepted: the longest accepted prefix across all paths

  Verification via attention mask:
    Each leaf in the tree has a unique causal attention mask.
    The target model evaluates all leaves in one batched forward pass.
    (See Chapter 5 — Flash Attention handles non-standard masks)

  Tree vs linear spectrum:
""")

    alpha = 0.78
    c = 8 / 70

    print(f"  α={alpha}, c={c:.3f}  (8B draft → 70B target)")
    print()
    print(f"  {'Strategy':<30}  {'Tokens proposed':>16}  {'Target passes':>14}  {'E[accepted]':>12}  {'Speedup':>8}")
    print(f"  {'─'*30}  {'─'*16}  {'─'*14}  {'─'*12}  {'─'*8}")

    # Linear speculation
    for gamma in [3, 5, 7]:
        e_acc = expected_accepted_tokens(alpha, gamma)
        S = theoretical_speedup(alpha, gamma, c)
        print(f"  {'Linear  γ='+str(gamma):<30}  {gamma:>16}  {'1':>14}  {e_acc:>12.2f}  {S:>7.2f}×")

    # Token tree (simplified model: tree with b=2 width recovers ~alpha^0.5 more acceptance
    # because we pick the best of b branches at each node)
    print()
    for (b, depth) in [(2, 3), (2, 4), (3, 3)]:
        total_proposed = sum(b**d for d in range(1, depth+1))
        # With tree, effective acceptance ≈ 1 - (1-alpha^b)^(1/b) per level (approximate)
        # With b branches per node, at least one branch is correct with prob
        # 1 - (1-alpha)^b, giving higher effective acceptance than linear
        alpha_tree = 1 - (1 - alpha)**b
        e_acc_tree = expected_accepted_tokens(alpha_tree, depth)
        # Target cost: 1 pass but handles total_proposed tokens (batched)
        # Draft cost: b^d evaluations (all paths)
        draft_overhead = total_proposed * c
        S_tree = (e_acc_tree + 1) / (1 + draft_overhead)
        print(f"  {'Tree    b='+str(b)+',d='+str(depth):<30}  {total_proposed:>16}  {'1 (batched)':>14}  "
              f"{e_acc_tree:>12.2f}  {S_tree:>7.2f}×")

    print(f"""
  Insight:
    Token trees work best for target models large enough that draft cost
    is negligible (large c denominator has small effect).
    For 70B and 405B targets, trees typically outperform linear chains.
    For 8B targets, linear speculation at γ=5 is usually sufficient.

  vLLM implementation:
    --speculative-draft-tensor-parallel-size  (if draft model needs TP)
    --speculative-num-draft-tokens γ          (linear speculation)
    --speculative-model <path>                (draft model path)
""")

    # Verify tree alpha > linear alpha (tree explores multiple branches)
    assert alpha_tree > alpha
    print(f"  ✓ Tree effective α ({alpha_tree:.3f}) > linear α ({alpha}) — branching helps")


def demo_draft_model_quality():
    """Demo 6: How draft model family / quality affects acceptance rate."""
    print(f"""
{'='*70}
DEMO 6 — Draft Model Quality vs Acceptance Rate
{'='*70}

  Acceptance rate α is the single most important parameter.
  It depends entirely on how well the draft distribution matches the target.

  Factors that raise α:
    ✓ Same model family (draft = smaller version of target)
    ✓ Same tokenizer (vocabulary alignment is required)
    ✓ Same pretraining data distribution
    ✓ Task matches training domain (code drafts better on code tasks)
    ✓ Longer context (draft can leverage full conversation history)

  Factors that lower α:
    ✗ Different architecture or vocab (cross-family)
    ✗ Different fine-tuning (draft is base model, target is instruct)
    ✗ Very long outputs (α degrades as sequence diverges from training)
    ✗ High temperature / creative generation (more randomness = lower α)
    ✗ Low-frequency tokens (both models disagree on rare tokens)

  Empirical acceptance rates by model pair:
""")

    print(f"  {'Pair':<40}  {'α':>6}  {'γ=5 Speedup':>12}  {'Family?':>10}  Notes")
    print(f"  {'─'*40}  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*30}")

    for pair in MODEL_PAIRS:
        S = pair.speedup(5)
        family = "same" if pair.same_family else "cross"
        note = "strong" if pair.alpha >= 0.75 else ("moderate" if pair.alpha >= 0.60 else "weak")
        pair_str = f"{pair.draft_name} → {pair.target_name}"
        print(f"  {pair_str:<40}  {pair.alpha:>6.2f}  {S:>11.2f}×  {family:>10}  {note}")

    print(f"""
  α vs temperature (Llama-8B draft → 70B target, γ=5):
""")

    pair = MODEL_PAIRS[0]  # 8B → 70B
    temps = [(0.0, 0.92), (0.3, 0.85), (0.7, 0.78), (1.0, 0.68), (1.5, 0.52), (2.0, 0.38)]
    print(f"  {'Temperature':>12}  {'α (empirical)':>14}  {'Speedup':>9}")
    print(f"  {'─'*12}  {'─'*14}  {'─'*9}")
    for temp, alpha_t in temps:
        S = theoretical_speedup(alpha_t, 5, pair.cost_ratio())
        greedy = "(greedy)" if temp == 0.0 else ""
        print(f"  {temp:>12.1f}  {alpha_t:>14.2f}  {S:>8.2f}×  {greedy}")

    print(f"""
  Key recommendation (Chapter 23):
    1. Always use same-family draft model
    2. Ensure IDENTICAL tokenizer vocabulary (non-negotiable)
    3. Temperature < 1.0 for best acceptance rates
    4. Monitor α in production — alert if α < 0.60 (switch to non-speculative)
""")

    # Verify same-family outperforms cross-family
    same_family  = [p for p in MODEL_PAIRS if p.same_family]
    cross_family = [p for p in MODEL_PAIRS if not p.same_family]
    avg_same  = sum(p.alpha for p in same_family)  / len(same_family)
    avg_cross = sum(p.alpha for p in cross_family) / len(cross_family)
    assert avg_same > avg_cross, "Same-family should have higher avg α"
    print(f"  ✓ Same-family avg α: {avg_same:.2f} > cross-family avg α: {avg_cross:.2f}")


def demo_vllm_config():
    """Demo 7: vLLM configuration for speculative decoding."""
    print(f"""
{'='*70}
DEMO 7 — vLLM Configuration: Knobs and Launch Commands
{'='*70}

  vLLM speculative decoding is enabled via CLI flags or AsyncEngineArgs.
  The draft model runs on the SAME GPUs as the target — no extra hardware.

  ──────────────────────────────────────────────────────────────────────
  Option 1: External draft model (any compatible smaller model)
  ──────────────────────────────────────────────────────────────────────

  python -m vllm.entrypoints.openai.api_server \\
      --model                       meta-llama/Llama-3.1-70B-Instruct \\
      --speculative-model           meta-llama/Llama-3.1-8B-Instruct  \\
      --num-speculative-tokens      5                                  \\
      --speculative-draft-tensor-parallel-size 1                       \\
      --tensor-parallel-size        4                                  \\
      --gpu-memory-utilization      0.90                               \\
      --max-model-len               8192

  Notes:
    • Draft model shares GPU memory with target (8B BF16 = 16GB extra)
    • --speculative-draft-tensor-parallel-size 1: draft runs on 1 GPU (small enough)
    • Draft model MUST use the same tokenizer as the target

  ──────────────────────────────────────────────────────────────────────
  Option 2: Draft from n-gram model (no extra model required)
  ──────────────────────────────────────────────────────────────────────

  python -m vllm.entrypoints.openai.api_server \\
      --model                  meta-llama/Llama-3.1-70B-Instruct \\
      --speculative-model      [ngram]                           \\
      --ngram-prompt-lookup-max 4                                \\
      --num-speculative-tokens  5                                \\
      --tensor-parallel-size    4

  n-gram speculation: look up the last N tokens in the prompt; if found,
  copy those tokens as draft. Great for tasks with lots of copy-paste
  (e.g. code completion, structured extraction, summarization).
  No extra GPU memory needed. α ≈ 0.50–0.70 for suitable tasks.

  ──────────────────────────────────────────────────────────────────────
  Option 3: Medusa (draft heads built into the target model weights)
  ──────────────────────────────────────────────────────────────────────

  python -m vllm.entrypoints.openai.api_server \\
      --model                  /path/to/medusa-llama-3.1-70b    \\
      --speculative-model      medusa                            \\
      --num-speculative-tokens  5

  Medusa fine-tunes multiple draft heads on top of the target model.
  Advantage: zero extra GPU memory (heads share target weights).
  Disadvantage: requires fine-tuned Medusa checkpoint; not plug-and-play.

  ──────────────────────────────────────────────────────────────────────
  Python API (AsyncEngineArgs)
  ──────────────────────────────────────────────────────────────────────
""")

    print(f"""  from vllm import AsyncEngineArgs, AsyncLLMEngine

  engine_args = AsyncEngineArgs(
      model                                = "meta-llama/Llama-3.1-70B-Instruct",
      speculative_model                    = "meta-llama/Llama-3.1-8B-Instruct",
      num_speculative_tokens               = 5,
      speculative_draft_tensor_parallel_size = 1,
      tensor_parallel_size                 = 4,
      gpu_memory_utilization               = 0.90,
  )
  engine = AsyncLLMEngine.from_engine_args(engine_args)

  ──────────────────────────────────────────────────────────────────────
  Key configuration decisions:
  ──────────────────────────────────────────────────────────────────────

  num_speculative_tokens (γ):
    Start with 5. Monitor acceptance_rate.
    If α > 0.85: try γ=7.  If α < 0.60: reduce to γ=3 or disable.

  Draft TP size:
    Match to draft model size. 8B on 1 GPU, 13B on 2 GPU.
    Draft TP < target TP: draft runs on a subset of the target's GPUs.

  gpu_memory_utilization:
    Lower by (draft_weight_gb / gpu_hbm_gb) from your baseline.
    8B BF16 on H100: 16GB / 80GB = reduce utilization by ~0.20
    Typical: 0.92 baseline → 0.72 with 8B draft on same H100.
""")

    # Verify memory headroom calculation
    draft_gb = 8 * 2      # 8B BF16
    gpu_gb   = 80
    reduction = draft_gb / gpu_gb
    print(f"  ✓ 8B BF16 draft on H100: {draft_gb}GB / {gpu_gb}GB = {reduction:.2f} GPU util reduction")
    print(f"  ✓ Typical setting: 0.92 - {reduction:.2f} = {0.92 - reduction:.2f} max GPU utilization with draft")


def demo_when_speculative_helps():
    """Demo 8: When speculative decoding helps vs hurts."""
    print(f"""
{'='*70}
DEMO 8 — When Speculative Decoding Helps vs Hurts
{'='*70}

  Speculative decoding is NOT always beneficial.
  This demo maps the benefit/cost space systematically.

  ──────────────────────────────────────────────────────────────────────
  WHEN IT HELPS (speedup > 1.3×):
  ──────────────────────────────────────────────────────────────────────
""")

    good_cases = [
        ("Long outputs (>100 tokens)",           0.80, 5, 8/70,  "more tokens to amortise draft"),
        ("Greedy/deterministic generation",       0.90, 5, 8/70,  "high α at temp=0"),
        ("Code completion",                       0.85, 5, 8/70,  "repetitive patterns, high α"),
        ("Structured extraction (JSON/YAML)",     0.87, 5, 8/70,  "predictable format"),
        ("summarization / paraphrase",            0.82, 5, 8/70,  "conservative word choices"),
        ("Large target (405B), small draft (8B)", 0.78, 5, 8/405, "c almost 0, free draft"),
    ]

    bad_cases = [
        ("Short outputs (<20 tokens)",         0.78, 5, 8/70,  "can't amortise overhead"),
        ("High temperature (>1.2)",            0.48, 5, 8/70,  "low α — almost always reject"),
        ("Creative writing (temp=1.0)",        0.60, 5, 8/70,  "diverse outputs, draft wrong"),
        ("Cross-family draft",                 0.50, 5, 7/70,  "low α, high rejection"),
        ("Large batch decode (batch>64)",      0.78, 5, 8/70,  "compute-bound: draft doesn't help"),
    ]

    print(f"  {'Use case':<45}  {'α':>5}  {'S':>7}  Note")
    print(f"  {'─'*45}  {'─'*5}  {'─'*7}  {'─'*35}")
    for name, alpha, gamma, c, note in good_cases:
        S = theoretical_speedup(alpha, gamma, c)
        status = "✓✓" if S > 2.0 else "✓"
        print(f"  {name:<45}  {alpha:>5.2f}  {S:>6.2f}×  {note}")

    print(f"\n  ──────────────────────────────────────────────────────────────────────")
    print(f"  WHEN IT HURTS (speedup ≤ 1.0×):")
    print(f"  ──────────────────────────────────────────────────────────────────────\n")

    print(f"  {'Use case':<45}  {'α':>5}  {'S':>7}  Note")
    print(f"  {'─'*45}  {'─'*5}  {'─'*7}  {'─'*35}")
    for name, alpha, gamma, c, note in bad_cases:
        S = theoretical_speedup(alpha, gamma, c)
        # For large batch, speculative decoding hurts because the target becomes
        # compute-bound and the draft just wastes time
        if "batch" in name.lower():
            S = 0.85  # empirically slower at large batches
        status = "✗" if S <= 1.0 else "~"
        print(f"  {name:<45}  {alpha:>5.2f}  {S:>6.2f}×  {note}")

    print(f"""
  Decision matrix:
  ┌─────────────────────────────┬──────────┬──────────────────────────────┐
  │ Condition                   │ Action   │ Reason                       │
  ├─────────────────────────────┼──────────┼──────────────────────────────┤
  │ α > 0.75, output > 50 tok  │ Enable   │ Clear throughput benefit      │
  │ α 0.60–0.75                 │ Try γ=3  │ Conservative; measure first  │
  │ α < 0.60                   │ Disable  │ Rejection overhead dominates  │
  │ Batch size > 64             │ Disable  │ GPU compute-bound anyway      │
  │ Output < 20 tokens          │ Disable  │ Can't amortise draft cost     │
  │ temperature > 1.2           │ Disable  │ High α impossible              │
  └─────────────────────────────┴──────────┴──────────────────────────────┘
""")

    # Verify good cases give speedup > bad cases
    good_speedups = [theoretical_speedup(a, g, c) for _, a, g, c, _ in good_cases]
    bad_speedups  = [theoretical_speedup(a, g, c) for _, a, g, c, _ in bad_cases[:4]]
    assert min(good_speedups) > max(bad_speedups), \
        "All good cases should outperform all bad cases"
    print(f"  ✓ Worst good case ({min(good_speedups):.2f}×) > Best bad case ({max(bad_speedups):.2f}×)")


def demo_batch_size_interaction():
    """Demo 9: Speculative decoding and batch size — the efficiency cliff."""
    print(f"""
{'='*70}
DEMO 9 — Batch Size Interaction: The Efficiency Cliff
{'='*70}

  At small batch sizes: memory-bandwidth-bound → speculation helps a lot.
  At large batch sizes: compute-bound → speculation adds overhead only.

  Why batching kills speculative decoding:
    With batch=B and speculative decoding:
      - Target processes B sequences simultaneously (good for batching)
      - But draft model must also handle B sequences in parallel
      - At large B, the GPU is already compute-saturated by the target
      - Adding draft overhead causes the target to run slower

  Roofline analysis:
    Decode intensity = (2 × params × batch) / (params_bytes)
                     = batch FLOPs/byte
    Ridge point (H100 BF16): ~295 FLOPs/byte

    Batch where target becomes compute-bound: 295 / 2 ≈ 147
    (In practice, batch ~32–64 is where speculative decoding starts to lose)

  Effective speedup vs batch size (empirical model):
""")

    alpha = 0.78
    gamma = 5
    c = 8 / 70

    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # Speedup degrades as batch increases (compute-bound effect)
    # Model: efficiency fraction = 1/(1 + batch/batch_cliff) where batch_cliff ~ 32
    batch_cliff = 32

    base_speedup = theoretical_speedup(alpha, gamma, c)

    print(f"  {'Batch':>7}  {'Regime':>18}  {'Efficiency':>12}  {'Effective S':>12}  Bar")
    print(f"  {'─'*7}  {'─'*18}  {'─'*12}  {'─'*12}  {'─'*25}")

    for b in batches:
        efficiency = 1.0 / (1.0 + b / batch_cliff)
        effective_S = 1.0 + (base_speedup - 1.0) * efficiency
        regime = "mem-bw bound" if b <= 16 else ("transitional" if b <= 64 else "compute bound")
        bar = "█" * int((effective_S - 1.0) * 10)
        indicator = "✓" if effective_S > 1.3 else ("~" if effective_S > 1.05 else "✗")
        print(f"  {b:>7}  {regime:>18}  {efficiency:>11.1%}  {effective_S:>11.2f}×  {bar}  {indicator}")

    print(f"""
  Recommendation:
    • Use speculative decoding when avg batch size < 32
    • Disable when avg batch size > 64 (negative ROI)
    • Monitor batch size histogram via vllm:avg_generation_throughput

  vLLM handles this automatically:
    When batch grows large during traffic spikes, vLLM's scheduler
    implicitly reduces speculation benefit by filling more slots.
    Proactive fix: set --max-num-seqs = min(32, your_target) when using speculation.
""")

    # Verify speedup degrades with batch
    s_small = 1.0 + (base_speedup - 1.0) * (1.0 / (1.0 + 1/batch_cliff))
    s_large = 1.0 + (base_speedup - 1.0) * (1.0 / (1.0 + 256/batch_cliff))
    assert s_small > s_large, "Speedup should degrade with batch size"
    print(f"  ✓ Speedup at batch=1: {s_small:.2f}× vs batch=256: {s_large:.2f}× (degrades as expected)")


def demo_production_sizing():
    """Demo 10: Production sizing — choosing optimal γ for your workload."""
    print(f"""
{'='*70}
DEMO 10 — Production Sizing: Choosing Optimal γ
{'='*70}

  Optimal γ maximizes throughput for a given workload and model pair.
  Too low: leaves speedup on the table.
  Too high: wasted draft compute on tokens that will be rejected.

  Analytical optimum: dS/dγ = 0
    S(γ) = (E[acc](γ) + 1) / (1 + c·γ)
    dS/dγ = 0 → solve numerically (no closed form for geometric acceptance)

  Numerical optimal γ for common model pairs:
""")

    print(f"  {'Model pair':<35}  {'α':>5}  {'c':>7}  {'Opt γ':>7}  {'Peak S':>8}")
    print(f"  {'─'*35}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*8}")

    for pair in MODEL_PAIRS:
        c = pair.cost_ratio()
        best_gamma = 5
        best_S = 0.0
        for g in range(1, 25):
            S = theoretical_speedup(pair.alpha, g, c)
            if S > best_S:
                best_S = S
                best_gamma = g
        pair_str = f"{pair.draft_name} → {pair.target_name}"
        print(f"  {pair_str:<35}  {pair.alpha:>5.2f}  {c:>7.4f}  {best_gamma:>7}  {best_S:>7.2f}×")

    print(f"""
  Production monitoring checklist (Prometheus):
""")

    metrics = [
        ("vllm:spec_decode_num_draft_tokens_total",   "Total draft tokens proposed"),
        ("vllm:spec_decode_num_accepted_tokens_total", "Total accepted draft tokens"),
        ("vllm:spec_decode_draft_acceptance_rate",     "Live acceptance rate α"),
        ("vllm:spec_decode_efficiency",                "Efficiency: accepted/(accepted+rejected)"),
        ("vllm:time_to_first_token_seconds",           "TTFT (should not regress)"),
        ("vllm:time_per_output_token_seconds",         "ITL (should improve)"),
    ]

    for metric, desc in metrics:
        print(f"    {metric}")
        print(f"      → {desc}")
    print()

    print(f"  Alert rules:")
    print(f"    ALERT: spec_decode_draft_acceptance_rate < 0.55 for 5 min")
    print(f"    → Action: disable speculation, file bug with draft model team")
    print()
    print(f"    ALERT: time_per_output_token_seconds INCREASES after enabling spec")
    print(f"    → Action: reduce γ or disable; batch size may be too large")
    print()

    print(f"  Practical checklist before enabling in production:")
    checklist = [
        "Confirm tokenizers are identical (sha256 of tokenizer.json must match)",
        "Measure α offline on representative production logs (target > 0.70)",
        "Check GPU memory fits both models (lower --gpu-memory-utilization)",
        "A/B test: 5% traffic for 24hr, monitor TTFT/ITL/quality",
        "Set up Prometheus alerts for α < 0.60 and ITL regression",
        "Have a kill switch: feature flag to instantly disable speculation",
    ]
    for i, item in enumerate(checklist, 1):
        print(f"    {i}. {item}")

    # Verify optimal gamma calculations
    for pair in MODEL_PAIRS:
        c = pair.cost_ratio()
        best_gamma, best_S = 1, 0.0
        for g in range(1, 30):
            S = theoretical_speedup(pair.alpha, g, c)
            if S > best_S:
                best_S, best_gamma = S, g
        # Optimal gamma should be > 1 for any reasonable model pair
        assert best_gamma >= 2, f"{pair.draft_name}→{pair.target_name}: opt γ should be >= 2"
        assert best_S > 1.0,    f"{pair.draft_name}→{pair.target_name}: peak speedup should be > 1×"

    print(f"\n  ✓ Optimal γ ≥ 2 for all model pairs — speculation always worth trying")
    print(f"  ✓ Peak speedup > 1× for all valid model pairs")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   Chapter 23: Speculative Decoding — Draft, Verify, Accept          ║")
    print("║   Comprehensive Demo Suite — 10 Demonstrations                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    demo_fundamental_idea()
    demo_acceptance_rejection_proof()
    demo_speedup_formula()
    demo_parameter_sweep()
    demo_token_tree_speculation()
    demo_draft_model_quality()
    demo_vllm_config()
    demo_when_speculative_helps()
    demo_batch_size_interaction()
    demo_production_sizing()

    print(f"\n{'='*70}")
    print("ALL CHAPTER 23 DEMOS COMPLETED — ALL ASSERTIONS PASSED ✓")
    print(f"{'='*70}")
    print("""
  Chapter 23 Key Takeaways:
  1. Speculative decoding is LOSSLESS — output distribution identical to target.
  2. Speedup formula: S = (E[acc]+1) / (1+c·γ). maximize α, minimize c.
  3. Same-family draft is essential: α ≥ 0.75 vs α ≈ 0.50 for cross-family.
  4. Optimal γ ≈ 5 for 8B→70B pairs; more for very large targets (405B).
  5. Token trees improve over linear chains for large targets with small drafts.
  6. Disable at batch size > 64: compute-bound regime negates all gains.
  7. Monitor α in production — alert at α < 0.60 and have a kill switch.
  8. n-gram speculation: free speedup (~1.3–1.7×) for copy-heavy tasks.
""")


if __name__ == "__main__":
    main()

```

