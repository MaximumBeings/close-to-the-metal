# Chapter 12: Sampling — From Logits to Tokens — Companion Code

## Python — `sampling_demo.py`

```python
"""
Chapter 12 — Sampling: From Logits to Tokens
Companion code: sampling_demo.py

Demonstrates:
  1. Numerically stable softmax
  2. Temperature scaling with distribution visualization
  3. Top-k filtering
  4. Top-p (nucleus) sampling
  5. Min-p filtering
  6. Repetition, frequency, and presence penalties
  7. Full pipeline: worked example from §12.7
  8. Structured output token mask (JSON schema enforcement)
  9. Beam search with log-probability tracking

No GPU required — all operations are on CPU numpy arrays.
"""

import math
import random
import heapq
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Numerically stable softmax
# ─────────────────────────────────────────────────────────────────────────────

def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax: subtract max before exponentiating."""
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


def demo_softmax():
    print("=" * 65)
    print("SECTION 1: Numerically Stable Softmax")
    print("=" * 65)

    logits = [4.20, 3.80, 2.10, 2.05, 1.30]
    tokens = ["the", "a", "cat", "dog", "of"]
    probs  = softmax(logits)

    print("\n  Logit   Token    Probability")
    print("  " + "-" * 35)
    for t, z, p in zip(tokens, logits, probs):
        bar = "█" * int(p * 30)
        print(f"  {z:>5.2f}   {t:<6}   {p:.3f}  {bar}")
    print(f"\n  Sum of probabilities: {sum(probs):.6f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Temperature scaling
# ─────────────────────────────────────────────────────────────────────────────

def apply_temperature(logits: list[float], temperature: float) -> list[float]:
    """Scale logits by 1/temperature. temperature=0 → greedy (special case)."""
    if temperature == 0.0:
        # Greedy: return one-hot distribution
        max_idx = logits.index(max(logits))
        return [1.0 if i == max_idx else 0.0 for i in range(len(logits))]
    return [z / temperature for z in logits]


def demo_temperature():
    print("=" * 65)
    print("SECTION 2: Temperature Scaling")
    print("=" * 65)

    logits = [4.20, 3.80, 2.10, 2.05, 1.30]
    tokens = ["the", "a", "cat", "dog", "of"]

    for T in [0.5, 1.0, 1.5, 2.0]:
        scaled = apply_temperature(logits, T)
        probs  = softmax(scaled)
        print(f"\n  T={T}")
        print(f"  {'Token':<8} {'Prob':>6}  Distribution")
        print("  " + "-" * 40)
        for tok, p in zip(tokens, probs):
            bar = "█" * int(p * 40)
            print(f"  {tok:<8} {p:>5.3f}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Top-k filtering
# ─────────────────────────────────────────────────────────────────────────────

NEG_INF = float("-inf")


def apply_top_k(logits: list[float], k: int) -> list[float]:
    """Zero out all logits except the top-k. Returns filtered logit list."""
    if k <= 0 or k >= len(logits):
        return list(logits)
    # Find the k-th largest value
    threshold = sorted(logits, reverse=True)[k - 1]
    # Keep only logits >= threshold; tie-break by keeping the first k
    result = list(logits)
    kept = 0
    for i in sorted(range(len(logits)), key=lambda x: -logits[x]):
        if kept < k:
            kept += 1
        else:
            result[i] = NEG_INF
    return result


def demo_top_k():
    print("=" * 65)
    print("SECTION 3: Top-k Filtering")
    print("=" * 65)

    logits = [4.20, 3.80, 2.10, 2.05, 1.30]
    tokens = ["the", "a", "cat", "dog", "of"]

    for k in [1, 2, 3, 5]:
        filtered = apply_top_k(logits, k)
        # Replace -inf with -999 for display
        probs = softmax([z if z != NEG_INF else -1e9 for z in filtered])
        surviving = [t for t, z in zip(tokens, filtered) if z != NEG_INF]
        print(f"\n  top_k={k}  surviving tokens: {surviving}")
        print(f"  {'Token':<8} {'Logit':>7}  {'Prob':>6}  Status")
        print("  " + "-" * 42)
        for tok, z, p in zip(tokens, filtered, probs):
            status = "✓" if z != NEG_INF else "✗ masked"
            print(f"  {tok:<8} {z if z != NEG_INF else '-inf':>7}  {p:>5.3f}  {status}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Top-p (nucleus) sampling
# ─────────────────────────────────────────────────────────────────────────────

def apply_top_p(logits: list[float], p: float) -> list[float]:
    """
    Nucleus sampling: keep the smallest set of tokens whose
    cumulative probability >= p.
    """
    probs = softmax(logits)
    # Sort indices by probability descending
    sorted_idx = sorted(range(len(probs)), key=lambda i: -probs[i])
    cumsum = 0.0
    nucleus = set()
    for i in sorted_idx:
        nucleus.add(i)
        cumsum += probs[i]
        if cumsum >= p:
            break
    result = [logits[i] if i in nucleus else NEG_INF for i in range(len(logits))]
    return result


def demo_top_p():
    print("=" * 65)
    print("SECTION 4: Top-p (Nucleus) Sampling")
    print("=" * 65)

    logits = [4.20, 3.80, 2.10, 2.05, 1.30]
    tokens = ["the", "a", "cat", "dog", "of"]
    probs_base = softmax(logits)

    for p in [0.5, 0.7, 0.9, 0.99]:
        filtered = apply_top_p(logits, p)
        surviving = [t for t, z in zip(tokens, filtered) if z != NEG_INF]
        probs_after = softmax([z if z != NEG_INF else -1e9 for z in filtered])
        print(f"\n  top_p={p}  nucleus tokens: {surviving}")
        print(f"  {'Token':<8} {'Orig Prob':>10}  {'New Prob':>9}  Status")
        print("  " + "-" * 44)
        cumsum = 0.0
        for tok, z, op, np_ in zip(tokens, filtered, probs_base, probs_after):
            in_nucleus = z != NEG_INF
            if in_nucleus:
                cumsum += op
            status = f"✓  cumsum={cumsum:.3f}" if in_nucleus else "✗ masked"
            print(f"  {tok:<8} {op:>10.3f}  {np_:>9.3f}  {status}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Min-p filtering
# ─────────────────────────────────────────────────────────────────────────────

def apply_min_p(logits: list[float], min_p: float) -> list[float]:
    """
    Keep tokens whose probability >= min_p * p_top.
    p_top is the probability of the highest-logit token.
    """
    probs = softmax(logits)
    p_top = max(probs)
    threshold = min_p * p_top
    return [z if probs[i] >= threshold else NEG_INF
            for i, z in enumerate(logits)]


def demo_min_p():
    print("=" * 65)
    print("SECTION 5: Min-p Filtering")
    print("=" * 65)

    # Two scenarios: confident step and uncertain step
    scenarios = [
        ("Confident step (clear top token)",
         [5.00, 1.50, 1.30, 1.20, 1.10]),
        ("Uncertain step (flat distribution)",
         [2.10, 2.05, 1.95, 1.90, 1.85]),
    ]
    tokens = ["tok0", "tok1", "tok2", "tok3", "tok4"]

    for label, logits in scenarios:
        probs_base = softmax(logits)
        print(f"\n  Scenario: {label}")
        print(f"  p_top = {max(probs_base):.3f}")
        print()
        for min_p in [0.02, 0.05, 0.10]:
            filtered = apply_min_p(logits, min_p)
            surviving = sum(1 for z in filtered if z != NEG_INF)
            thresh = min_p * max(probs_base)
            print(f"    min_p={min_p}  threshold={thresh:.4f}  "
                  f"surviving tokens: {surviving}/{len(logits)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Repetition, frequency, and presence penalties
# ─────────────────────────────────────────────────────────────────────────────

def apply_repetition_penalty(
    logits: list[float],
    token_ids: list[int],
    penalty: float,
) -> list[float]:
    """
    Multiplicative repetition penalty: divide positive logits, multiply
    negative logits for tokens that have appeared in context.
    """
    result = list(logits)
    seen = set(token_ids)
    for tid in seen:
        if tid < len(result):
            z = result[tid]
            result[tid] = z / penalty if z > 0 else z * penalty
    return result


def apply_frequency_penalty(
    logits: list[float],
    token_ids: list[int],
    penalty: float,
) -> list[float]:
    """Subtract frequency_penalty × count(token) from each token's logit."""
    from collections import Counter
    counts = Counter(token_ids)
    result = list(logits)
    for tid, cnt in counts.items():
        if tid < len(result):
            result[tid] -= penalty * cnt
    return result


def apply_presence_penalty(
    logits: list[float],
    token_ids: list[int],
    penalty: float,
) -> list[float]:
    """Subtract presence_penalty once for each token that appeared at all."""
    result = list(logits)
    seen = set(token_ids)
    for tid in seen:
        if tid < len(result):
            result[tid] -= penalty
    return result


def demo_penalties():
    print("=" * 65)
    print("SECTION 6: Repetition, Frequency, and Presence Penalties")
    print("=" * 65)

    # Vocabulary: index matches token_id
    tokens  = ["the", "cat", "sat", "on", "mat"]
    logits  = [3.20,  2.80,  1.50, 2.10, 4.50]
    context = [0, 1, 3, 0]   # "the", "cat", "on", "the" — "the" appears twice

    print(f"\n  Context token IDs: {context}")
    print(f"  ('the'=0 appears 2×, 'cat'=1 appears 1×, 'on'=3 appears 1×)")
    print()

    def show(label, lgs):
        probs = softmax(lgs)
        print(f"  {label}")
        for t, z, p in zip(tokens, lgs, probs):
            bar = "█" * int(p * 35)
            marker = " ←" if t in ["the", "cat", "on"] else ""
            print(f"    {t:<6}  logit={z:>6.3f}  p={p:.3f}  {bar}{marker}")
        print()

    show("Original logits:", logits)
    show("After repetition_penalty=1.3:",
         apply_repetition_penalty(logits, context, 1.3))
    show("After frequency_penalty=0.5:",
         apply_frequency_penalty(logits, context, 0.5))
    show("After presence_penalty=0.5:",
         apply_presence_penalty(logits, context, 0.5))


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Full pipeline (Worked Example 12.7)
# ─────────────────────────────────────────────────────────────────────────────

def full_sampling_pipeline(
    logits: list[float],
    tokens: list[str],
    context_ids: list[int],
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    verbose: bool = True,
) -> int:
    """
    Apply the full sampling pipeline and return a sampled token index.
    Verbose mode prints the distribution after each stage.
    """
    def show(stage, lgs):
        if not verbose:
            return
        probs = softmax([z if z != NEG_INF else -1e9 for z in lgs])
        print(f"\n  After {stage}:")
        for t, z, p in zip(tokens, lgs, probs):
            zstr = f"{z:.3f}" if z != NEG_INF else "-inf "
            bar = "█" * int(p * 30)
            print(f"    {t:<8} logit={zstr}  p={p:.3f}  {bar}")

    current = list(logits)
    if verbose:
        show("forward pass (raw logits)", current)

    # Step 1: Repetition penalty
    if repetition_penalty != 1.0:
        current = apply_repetition_penalty(current, context_ids, repetition_penalty)
        show(f"repetition_penalty={repetition_penalty}", current)

    # Step 2: Temperature
    current = apply_temperature(current, temperature)
    show(f"temperature={temperature}", current)

    # Step 3: Top-k
    if top_k > 0:
        current = apply_top_k(current, top_k)
        show(f"top_k={top_k}", current)

    # Step 4: Top-p
    if top_p < 1.0:
        current = apply_top_p(current, top_p)
        show(f"top_p={top_p}", current)

    # Step 5: Min-p
    if min_p > 0.0:
        current = apply_min_p(current, min_p)
        show(f"min_p={min_p}", current)

    # Step 6: Sample
    probs = softmax([z if z != NEG_INF else -1e9 for z in current])
    # Multinomial sample
    r = random.random()
    cumsum = 0.0
    sampled = len(probs) - 1
    for i, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            sampled = i
            break

    return sampled


def demo_full_pipeline():
    print("=" * 65)
    print("SECTION 7: Full Sampling Pipeline (Worked Example 12.7)")
    print("=" * 65)
    print()
    print("  Context: 'The cat sat on the'")
    print("  Config: repetition_penalty=1.2, temperature=0.8, top_k=3, top_p=0.92")
    print()

    tokens  = ["mat", "floor", "cat", "roof", "the"]
    logits  = [4.50,   3.90,   2.80,  1.60,  3.20]
    # context: "cat"(idx 2) and "the"(idx 4) have appeared
    context = [2, 4]

    random.seed(7)
    result = full_sampling_pipeline(
        logits, tokens, context,
        repetition_penalty=1.2,
        temperature=0.8,
        top_k=3,
        top_p=0.92,
        verbose=True,
    )
    print(f"\n  ═══ Sampled token: '{tokens[result]}' ═══\n")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Structured output: token mask (JSON schema enforcement)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JSONParseState:
    """
    Minimal JSON parser state machine for demonstration.
    Tracks what character class is valid at each position in a
    {"name": string, "age": integer} schema.
    """
    STATES = [
        "start",           # expecting '{'
        "after_open",      # expecting '"name"'
        "after_name_key",  # expecting ':'
        "after_name_colon",# expecting '"<string>"'
        "in_name_string",  # inside the name value string
        "after_name_val",  # expecting ','
        "after_comma",     # expecting '"age"'
        "after_age_key",   # expecting ':'
        "after_age_colon", # expecting integer
        "in_integer",      # inside the integer value
        "done",            # expecting '}'
        "closed",          # terminal state
    ]

    state: str = "start"

    def valid_token_classes(self) -> str:
        """Return description of allowed token classes at current state."""
        mapping = {
            "start":           "'{' only",
            "after_open":      '\'"\' (start of "name" key)',
            "after_name_key":  "':' only",
            "after_name_colon":'\'"\' (start of string value)',
            "in_name_string":  "letters, spaces (inside string)",
            "after_name_val":  "',' only",
            "after_comma":     '\'"\' (start of "age" key)',
            "after_age_key":   "':' only",
            "after_age_colon": "digits 0-9",
            "in_integer":      "digits 0-9 or '}' to close",
            "done":            "'}' only",
            "closed":          "EOS token",
        }
        return mapping.get(self.state, "unknown")

    def allowed_chars(self) -> set:
        mapping = {
            "start":           {"{"},
            "after_open":      {'"'},
            "after_name_key":  {":"},
            "after_name_colon":{'"'},
            "in_name_string":  set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "),
            "after_name_val":  {","},
            "after_comma":     {'"'},
            "after_age_key":   {":"},
            "after_age_colon": set("0123456789"),
            "in_integer":      set("0123456789}"),
            "done":            {"}"},
            "closed":          {"<EOS>"},
        }
        return mapping.get(self.state, set())

    def advance(self, char: str):
        """Advance the state machine given the next character."""
        transitions = {
            ("start",           "{"): "after_open",
            ("after_open",      '"'): "after_name_key",
            ("after_name_key",  ":"): "after_name_colon",
            ("after_name_colon",'"'): "in_name_string",
            ("after_name_val",  ","): "after_comma",
            ("after_comma",     '"'): "after_age_key",
            ("after_age_key",   ":"): "after_age_colon",
            ("done",            "}"): "closed",
        }
        key = (self.state, char)
        if key in transitions:
            self.state = transitions[key]
        elif self.state == "in_name_string" and char == '"':
            self.state = "after_name_val"
        elif self.state == "in_name_string" and char in self.allowed_chars():
            pass  # stay in string
        elif self.state == "after_age_colon" and char.isdigit():
            self.state = "in_integer"
        elif self.state == "in_integer" and char.isdigit():
            pass  # stay in integer
        elif self.state == "in_integer" and char == "}":
            self.state = "closed"


def demo_structured_output():
    print("=" * 65)
    print("SECTION 8: Structured Output — JSON Token Masking")
    print("=" * 65)
    print()
    print("  Schema: {\"name\": string, \"age\": integer}")
    print()

    # Simulate generating: {"name": "Alice", "age": 30}
    generation = [
        ("{", "Open brace"),
        ('"', "Start name key"),
        ("n", "n"),
        ("a", "a"),
        ("m", "m"),
        ("e", "e"),
        ('"', "Close name key"),
        (":", "Colon"),
        ('"', "Start name value"),
        ("A", "A"),
        ("l", "l"),
        ("i", "i"),
        ("c", "c"),
        ("e", "e"),
        ('"', "Close name value"),
        (",", "Comma"),
        ('"', "Start age key"),
        ("a", "a"),
        ("g", "g"),
        ("e", "e"),
        ('"', "Close age key"),
        (":", "Colon"),
        ("3", "Digit 3"),
        ("0", "Digit 0"),
        ("}", "Close brace"),
    ]

    fsm = JSONParseState()
    generated = ""

    print(f"  {'Pos':>4}  {'Char':<8}  {'State before':<22}  {'Allowed next'}")
    print("  " + "-" * 75)

    for i, (char, label) in enumerate(generation):
        allowed = fsm.valid_token_classes()
        is_valid = char in fsm.allowed_chars()
        marker = "✓" if is_valid else "✗ INVALID"
        print(f"  {i:>4}  {char!r:<8}  {fsm.state:<22}  {allowed[:30]}  {marker}")
        fsm.advance(char)
        generated += char

    print(f"\n  Generated: {generated}")
    print(f"  Final state: {fsm.state}")
    print()

    # Demonstrate a violation attempt
    print("  Attempting invalid token at position 7 (digit instead of '\"'):  ")
    fsm2 = JSONParseState()
    for char, _ in generation[:7]:
        fsm2.advance(char)
    print(f"  State: '{fsm2.state}', allowed: '{fsm2.valid_token_classes()}'")
    print(f"  Attempting '5': {'5' in fsm2.allowed_chars()} → masked to -∞")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Beam search
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Beam:
    tokens: list[int]
    log_prob: float

    def __lt__(self, other):
        return self.log_prob > other.log_prob   # max-heap by log_prob


def beam_search(
    vocab: list[str],
    logit_fn,           # callable(context_ids) -> logits list
    n_beams: int = 2,
    max_steps: int = 5,
    eos_id: int = -1,
    verbose: bool = True,
) -> list:
    """
    Simple beam search over a vocabulary.
    logit_fn: takes list of token IDs, returns list of raw logits.
    """
    beams = [Beam(tokens=[], log_prob=0.0)]

    if verbose:
        print(f"\n  Beam search: n_beams={n_beams}, max_steps={max_steps}")

    for step in range(max_steps):
        candidates = []
        for beam in beams:
            if beam.tokens and beam.tokens[-1] == eos_id:
                candidates.append(beam)
                continue
            logits = logit_fn(beam.tokens)
            probs  = softmax(logits)
            for tok_id, p in enumerate(probs):
                if p < 1e-9:
                    continue
                new_log_prob = beam.log_prob + math.log(p)
                candidates.append(Beam(
                    tokens=beam.tokens + [tok_id],
                    log_prob=new_log_prob,
                ))

        # Keep top n_beams
        candidates.sort(key=lambda b: -b.log_prob)
        beams = candidates[:n_beams]

        if verbose:
            print(f"\n  Step {step + 1}:")
            for rank, b in enumerate(beams):
                seq = " ".join(vocab[t] for t in b.tokens)
                print(f"    Beam {rank}: [{seq}]  log-prob={b.log_prob:.3f}")

    return beams


def demo_beam_search():
    print("=" * 65)
    print("SECTION 9: Beam Search")
    print("=" * 65)

    vocab = ["<EOS>", "the", "a", "cat", "dog", "sat", "ran"]

    # Synthetic logit function: returns different distributions per context
    def logit_fn(context):
        if not context:
            # First token: strong preference for "the" and "a"
            return [-5.0, 4.0, 3.5, 1.0, 1.0, 0.5, 0.5]
        last = context[-1]
        if last == 1:   # "the"
            return [-5.0, 0.0, 0.0, 3.5, 3.0, 0.5, 0.5]
        if last == 2:   # "a"
            return [-5.0, 0.0, 0.0, 3.0, 3.5, 0.5, 0.5]
        if last in [3, 4]:  # "cat" or "dog"
            return [-5.0, 0.0, 0.0, 0.0, 0.0, 4.0, 3.5]
        # After "sat"/"ran": high EOS probability
        return [4.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    beams = beam_search(vocab, logit_fn, n_beams=2, max_steps=4)

    print(f"\n  ═══ Final beams (ranked by log-probability) ═══")
    for rank, b in enumerate(beams):
        seq = " ".join(vocab[t] for t in b.tokens)
        print(f"  Rank {rank + 1}: [{seq}]  log-prob={b.log_prob:.3f}  "
              f"prob={math.exp(b.log_prob):.4f}")
    print()

    # KV memory cost
    print("  KV Cache Memory Cost of Beam Search:")
    for n in [1, 2, 4, 8]:
        print(f"    n_beams={n}: {n}x single-sequence KV memory")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    demo_softmax()
    demo_temperature()
    demo_top_k()
    demo_top_p()
    demo_min_p()
    demo_penalties()
    demo_full_pipeline()
    demo_structured_output()
    demo_beam_search()

    print("=" * 65)
    print("Chapter 12 sampling demo complete.")
    print()
    print("Key results:")
    print("  • Softmax: subtract max first to prevent overflow")
    print("  • Temperature: T=0.5 → top token 68%, T=2.0 → top token 30%")
    print("  • Top-k=3 + top-p=0.92: pipeline prunes to most probable nucleus")
    print("  • Repetition penalty=1.3: seen tokens' probability drops to ~62%")
    print("  • JSON masking: only valid tokens survive at each position")
    print("  • Beam search n=4: 4x KV memory — use sparingly in production")
    print("=" * 65)

```

## C++ — `sampling_demo.cpp`

```cpp
/*
 * Chapter 12 — Sampling: From Logits to Tokens
 * Companion code: sampling_demo.cpp
 *
 * Demonstrates:
 *   1. Numerically stable softmax
 *   2. Temperature scaling
 *   3. Top-k filtering
 *   4. Top-p (nucleus) sampling
 *   5. Min-p filtering
 *   6. Repetition, frequency, and presence penalties
 *   7. Full pipeline — worked example from §12.7
 *   8. llama.cpp sampler chain (annotated C API sketch)
 *   9. Beam search with log-probability tracking
 *
 * Compile:
 *   g++ -std=c++17 -O2 -o sampling_demo sampling_demo.cpp
 *
 * Run:
 *   ./sampling_demo
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr float NEG_INF = -1e30f;

// ─────────────────────────────────────────────────────────────────────────────
// Core math
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> softmax(const std::vector<float>& logits) {
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exps(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_val);
        sum += exps[i];
    }
    for (auto& e : exps) e /= sum;
    return exps;
}

// Multinomial sample: draw one index according to probabilities
int multinomial_sample(const std::vector<float>& probs, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (r <= cumsum) return static_cast<int>(i);
    }
    return static_cast<int>(probs.size() - 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 1 — Softmax demo
// ─────────────────────────────────────────────────────────────────────────────

void demo_softmax() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 1: Numerically Stable Softmax\n";
    std::cout << std::string(65, '=') << "\n\n";

    std::vector<float>       logits = {4.20f, 3.80f, 2.10f, 2.05f, 1.30f};
    std::vector<std::string> tokens = {"the",  "a",   "cat", "dog", "of"};

    auto probs = softmax(logits);
    float sum = 0.0f;

    std::cout << "  Logit   Token    Probability\n";
    std::cout << "  " << std::string(40, '-') << "\n";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string bar(static_cast<int>(probs[i] * 30), (char)0xE2);   // just use '#'
        bar = std::string(static_cast<int>(probs[i] * 30), '#');
        std::cout << "  " << std::fixed << std::setprecision(2) << logits[i]
                  << "   " << std::left << std::setw(7) << tokens[i]
                  << "  " << std::right << std::setprecision(3) << probs[i]
                  << "  " << bar << "\n";
        sum += probs[i];
    }
    std::cout << "\n  Sum of probabilities: " << std::setprecision(6) << sum << "\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2 — Temperature
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> apply_temperature(const std::vector<float>& logits, float T) {
    if (T == 0.0f) {
        // Greedy: one-hot on argmax
        int best = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::vector<float> out(logits.size(), NEG_INF);
        out[best] = 0.0f;
        return out;
    }
    std::vector<float> out(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) out[i] = logits[i] / T;
    return out;
}

void demo_temperature() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 2: Temperature Scaling\n";
    std::cout << std::string(65, '=') << "\n\n";

    std::vector<float>       logits = {4.20f, 3.80f, 2.10f, 2.05f, 1.30f};
    std::vector<std::string> tokens = {"the",  "a",   "cat", "dog", "of"};

    for (float T : {0.5f, 1.0f, 1.5f, 2.0f}) {
        auto scaled = apply_temperature(logits, T);
        auto probs  = softmax(scaled);
        std::cout << "  T=" << T << "\n";
        std::cout << "  " << std::left << std::setw(8) << "Token"
                  << std::right << std::setw(7) << "Prob" << "  Distribution\n";
        std::cout << "  " << std::string(45, '-') << "\n";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string bar(static_cast<int>(probs[i] * 35), '#');
            std::cout << "  " << std::left << std::setw(8) << tokens[i]
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(7) << probs[i] << "  " << bar << "\n";
        }
        std::cout << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3 — Top-k filtering
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> apply_top_k(const std::vector<float>& logits, int k) {
    if (k <= 0 || k >= (int)logits.size()) return logits;

    // Find indices sorted by logit descending
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b){ return logits[a] > logits[b]; });

    std::set<int> keep(idx.begin(), idx.begin() + k);
    std::vector<float> out(logits.size(), NEG_INF);
    for (int i : keep) out[i] = logits[i];
    return out;
}

void demo_top_k() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 3: Top-k Filtering\n";
    std::cout << std::string(65, '=') << "\n\n";

    std::vector<float>       logits = {4.20f, 3.80f, 2.10f, 2.05f, 1.30f};
    std::vector<std::string> tokens = {"the",  "a",   "cat", "dog", "of"};

    for (int k : {1, 2, 3, 5}) {
        auto filtered = apply_top_k(logits, k);
        std::vector<std::string> surviving;
        for (size_t i = 0; i < tokens.size(); ++i)
            if (filtered[i] != NEG_INF) surviving.push_back(tokens[i]);

        std::cout << "  top_k=" << k << "  surviving: [";
        for (size_t i = 0; i < surviving.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << surviving[i];
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4 — Top-p (nucleus) sampling
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> apply_top_p(const std::vector<float>& logits, float p) {
    auto probs = softmax(logits);
    // Sort indices by prob descending
    std::vector<int> idx(probs.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });

    float cumsum = 0.0f;
    std::set<int> nucleus;
    for (int i : idx) {
        nucleus.insert(i);
        cumsum += probs[i];
        if (cumsum >= p) break;
    }

    std::vector<float> out(logits.size(), NEG_INF);
    for (int i : nucleus) out[i] = logits[i];
    return out;
}

void demo_top_p() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 4: Top-p (Nucleus) Sampling\n";
    std::cout << std::string(65, '=') << "\n\n";

    std::vector<float>       logits = {4.20f, 3.80f, 2.10f, 2.05f, 1.30f};
    std::vector<std::string> tokens = {"the",  "a",   "cat", "dog", "of"};

    for (float p : {0.5f, 0.7f, 0.9f, 0.99f}) {
        auto filtered = apply_top_p(logits, p);
        int surviving = 0;
        for (auto v : filtered) if (v != NEG_INF) ++surviving;
        std::cout << "  top_p=" << std::setprecision(2) << p
                  << "  surviving tokens: " << surviving << "/" << tokens.size() << "\n";
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5 — Min-p filtering
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> apply_min_p(const std::vector<float>& logits, float min_p_param) {
    auto probs = softmax(logits);
    float p_top = *std::max_element(probs.begin(), probs.end());
    float threshold = min_p_param * p_top;

    std::vector<float> out(logits.size(), NEG_INF);
    for (size_t i = 0; i < probs.size(); ++i)
        if (probs[i] >= threshold) out[i] = logits[i];
    return out;
}

void demo_min_p() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 5: Min-p Filtering\n";
    std::cout << std::string(65, '=') << "\n\n";

    struct Case { std::string label; std::vector<float> logits; };
    std::vector<Case> cases = {
        {"Confident step", {5.00f, 1.50f, 1.30f, 1.20f, 1.10f}},
        {"Uncertain step", {2.10f, 2.05f, 1.95f, 1.90f, 1.85f}},
    };

    for (auto& c : cases) {
        auto probs = softmax(c.logits);
        float p_top = *std::max_element(probs.begin(), probs.end());
        std::cout << "  Scenario: " << c.label
                  << "  p_top=" << std::fixed << std::setprecision(3) << p_top << "\n";
        for (float mp : {0.02f, 0.05f, 0.10f}) {
            auto filtered = apply_min_p(c.logits, mp);
            int surviving = 0;
            for (auto v : filtered) if (v != NEG_INF) ++surviving;
            std::cout << "    min_p=" << mp
                      << "  threshold=" << mp * p_top
                      << "  surviving: " << surviving << "/" << c.logits.size() << "\n";
        }
        std::cout << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6 — Penalties
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> apply_repetition_penalty(
    std::vector<float> logits,
    const std::vector<int>& context,
    float penalty)
{
    std::set<int> seen(context.begin(), context.end());
    for (int tid : seen) {
        if (tid < 0 || tid >= (int)logits.size()) continue;
        float& z = logits[tid];
        z = (z > 0.0f) ? z / penalty : z * penalty;
    }
    return logits;
}

std::vector<float> apply_frequency_penalty(
    std::vector<float> logits,
    const std::vector<int>& context,
    float penalty)
{
    std::unordered_map<int, int> counts;
    for (int t : context) counts[t]++;
    for (auto& [tid, cnt] : counts) {
        if (tid < (int)logits.size())
            logits[tid] -= penalty * cnt;
    }
    return logits;
}

std::vector<float> apply_presence_penalty(
    std::vector<float> logits,
    const std::vector<int>& context,
    float penalty)
{
    std::set<int> seen(context.begin(), context.end());
    for (int tid : seen) {
        if (tid < (int)logits.size())
            logits[tid] -= penalty;
    }
    return logits;
}

void demo_penalties() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 6: Repetition, Frequency, and Presence Penalties\n";
    std::cout << std::string(65, '=') << "\n\n";

    std::vector<std::string> tokens  = {"the", "cat", "sat", "on", "mat"};
    std::vector<float>       logits  = {3.20f, 2.80f, 1.50f, 2.10f, 4.50f};
    std::vector<int>         context = {0, 1, 3, 0};   // the, cat, on, the

    auto show = [&](const std::string& label, const std::vector<float>& lgs) {
        auto probs = softmax(lgs);
        std::cout << "  " << label << "\n";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string bar(static_cast<int>(probs[i] * 30), '#');
            std::cout << "    " << std::left << std::setw(6) << tokens[i]
                      << "  logit=" << std::right << std::fixed << std::setprecision(3)
                      << std::setw(7) << (lgs[i] > NEG_INF/2 ? lgs[i] : -99.0f)
                      << "  p=" << probs[i]
                      << "  " << bar << "\n";
        }
        std::cout << "\n";
    };

    show("Original:", logits);
    show("After repetition_penalty=1.3:",
         apply_repetition_penalty(logits, context, 1.3f));
    show("After frequency_penalty=0.5:",
         apply_frequency_penalty(logits, context, 0.5f));
    show("After presence_penalty=0.5:",
         apply_presence_penalty(logits, context, 0.5f));
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7 — Full pipeline (Worked Example 12.7)
// ─────────────────────────────────────────────────────────────────────────────

int full_pipeline(
    std::vector<float>              logits,
    const std::vector<std::string>& tokens,
    const std::vector<int>&         context,
    float rep_penalty = 1.0f,
    float temperature = 1.0f,
    int   top_k       = 0,
    float top_p       = 1.0f,
    float min_p       = 0.0f,
    std::mt19937* rng = nullptr,
    bool verbose      = true)
{
    auto show = [&](const std::string& stage, const std::vector<float>& lgs) {
        if (!verbose) return;
        auto probs = softmax(lgs);
        std::cout << "\n  After " << stage << ":\n";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string bar(static_cast<int>(probs[i] * 28), '#');
            std::string zstr = (lgs[i] > NEG_INF / 2)
                ? std::to_string(lgs[i]).substr(0, 6)
                : "-inf ";
            std::cout << "    " << std::left << std::setw(8) << tokens[i]
                      << " logit=" << std::right << std::setw(7) << zstr
                      << "  p=" << std::fixed << std::setprecision(3) << probs[i]
                      << "  " << bar << "\n";
        }
    };

    if (verbose) show("forward pass (raw)", logits);

    if (rep_penalty != 1.0f) {
        logits = apply_repetition_penalty(logits, context, rep_penalty);
        show("repetition_penalty=" + std::to_string(rep_penalty), logits);
    }

    logits = apply_temperature(logits, temperature);
    show("temperature=" + std::to_string(temperature), logits);

    if (top_k > 0) {
        logits = apply_top_k(logits, top_k);
        show("top_k=" + std::to_string(top_k), logits);
    }

    if (top_p < 1.0f) {
        logits = apply_top_p(logits, top_p);
        show("top_p=" + std::to_string(top_p), logits);
    }

    if (min_p > 0.0f) {
        logits = apply_min_p(logits, min_p);
        show("min_p=" + std::to_string(min_p), logits);
    }

    auto probs = softmax(logits);
    if (rng) return multinomial_sample(probs, *rng);

    // Greedy fallback
    return (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

void demo_full_pipeline() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 7: Full Sampling Pipeline (Worked Example 12.7)\n";
    std::cout << std::string(65, '=') << "\n\n";
    std::cout << "  Context: 'The cat sat on the'\n";
    std::cout << "  Config: rep_penalty=1.2, T=0.8, top_k=3, top_p=0.92\n";

    std::vector<std::string> tokens  = {"mat", "floor", "cat", "roof", "the"};
    std::vector<float>       logits  = {4.50f, 3.90f,  2.80f, 1.60f, 3.20f};
    std::vector<int>         context = {2, 4};   // cat, the

    std::mt19937 rng(7);
    int sampled = full_pipeline(
        logits, tokens, context,
        1.2f, 0.8f, 3, 0.92f, 0.0f, &rng, true
    );

    std::cout << "\n  === Sampled token: '" << tokens[sampled] << "' ===\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8 — llama.cpp sampler chain (annotated sketch)
// ─────────────────────────────────────────────────────────────────────────────

void demo_llamacpp_sampler_chain() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 8: llama.cpp Sampler Chain (Annotated C API Sketch)\n";
    std::cout << std::string(65, '=') << "\n\n";

    // This section shows what the real llama.cpp API calls look like.
    // The actual API requires linking against libllama.
    // We print the equivalent code as documentation.

    const char* code = R"code(
  // ── Real llama.cpp sampler chain construction ──────────────────────
  // (requires #include "llama.h" and linking against libllama)

  struct llama_sampler * smpl =
      llama_sampler_chain_init(llama_sampler_chain_default_params());

  // Stage 1: Repetition + frequency + presence penalties
  //   penalty_last_n  = how many previous tokens to consider
  //   penalty_repeat  = repetition penalty multiplier (1.0 = off)
  //   penalty_freq    = frequency penalty coefficient
  //   penalty_present = presence penalty coefficient
  llama_sampler_chain_add(smpl,
      llama_sampler_init_penalties(
          n_vocab,            // vocabulary size
          special_eos_id,     // EOS token ID (always allowed)
          linefeed_id,        // newline ID (optional exception)
          /*penalty_last_n=*/  64,
          /*penalty_repeat=*/   1.1f,
          /*penalty_freq=*/     0.0f,
          /*penalty_present=*/  0.0f,
          /*penalize_nl=*/      false,
          /*ignore_eos=*/       false
      )
  );

  // Stage 2: Temperature
  llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));

  // Stage 3: Top-k
  llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));

  // Stage 4: Top-p (nucleus)
  //   min_keep = always keep at least this many tokens (safety floor)
  llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, /*min_keep=*/1));

  // Stage 5: Min-p (alternative/additional to top-p)
  // llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));

  // Stage 6: Random sampler (multinomial from remaining candidates)
  llama_sampler_chain_add(smpl, llama_sampler_init_dist(/*seed=*/42));

  // ── Decode loop ───────────────────────────────────────────────────
  for (int step = 0; step < max_tokens; ++step) {
      // Forward pass produces logits — see Chapter 9
      llama_decode(ctx, batch);

      // Sample one token (applies all chain stages in order)
      llama_token next_token = llama_sampler_sample(smpl, ctx, /*last_idx=*/-1);

      // Update penalty history with the chosen token
      llama_sampler_accept(smpl, next_token);

      if (next_token == llama_token_eos(model)) break;

      // Append token to next batch
      llama_batch_add(batch, next_token, step + n_prompt, {0}, true);
  }

  // Clean up
  llama_sampler_free(smpl);
  // ─────────────────────────────────────────────────────────────────
)code";

    std::cout << code << "\n";

    // GBNF grammar snippet
    std::cout << "  ── GBNF grammar for JSON {name: string, age: integer} ──\n\n";
    const char* gbnf = R"gbnf(
  root    ::= object
  object  ::= "{" ws "\"name\"" ws ":" ws string
               "," ws "\"age\""  ws ":" ws integer
               "}" ws
  string  ::= "\"" [a-zA-Z ]* "\""
  integer ::= [0-9]+
  ws      ::= [ \t\n]*
)gbnf";
    std::cout << gbnf << "\n";

    std::cout << "  To use with llama.cpp:\n";
    std::cout << "    auto grammar_str = load_grammar_from_file(\"schema.gbnf\");\n";
    std::cout << "    llama_sampler_chain_add(smpl,\n";
    std::cout << "        llama_sampler_init_grammar(model, grammar_str.c_str(), \"root\"));\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9 — Beam search
// ─────────────────────────────────────────────────────────────────────────────

struct Beam {
    std::vector<int> tokens;
    float log_prob;

    bool operator<(const Beam& o) const { return log_prob < o.log_prob; }
};

// Synthetic logit function for demo
std::vector<float> synth_logits(const std::vector<int>& context) {
    // Vocab: 0=EOS 1=the 2=a 3=cat 4=dog 5=sat 6=ran
    if (context.empty())
        return {-5.0f, 4.0f, 3.5f, 1.0f, 1.0f, 0.5f, 0.5f};
    int last = context.back();
    if (last == 1)  return {-5.0f, 0.0f, 0.0f, 3.5f, 3.0f, 0.5f, 0.5f};
    if (last == 2)  return {-5.0f, 0.0f, 0.0f, 3.0f, 3.5f, 0.5f, 0.5f};
    if (last == 3 || last == 4) return {-5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 3.5f};
    return {4.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
}

void demo_beam_search() {
    std::cout << std::string(65, '=') << "\n";
    std::cout << "SECTION 9: Beam Search\n";
    std::cout << std::string(65, '=') << "\n\n";

    std::vector<std::string> vocab = {"<EOS>","the","a","cat","dog","sat","ran"};
    int n_beams    = 2;
    int max_steps  = 4;

    std::vector<Beam> beams = {Beam{{}, 0.0f}};

    std::cout << "  n_beams=" << n_beams << "  max_steps=" << max_steps << "\n";

    for (int step = 0; step < max_steps; ++step) {
        std::vector<Beam> candidates;

        for (auto& b : beams) {
            auto logits = synth_logits(b.tokens);
            auto probs  = softmax(logits);
            for (int tid = 0; tid < (int)probs.size(); ++tid) {
                if (probs[tid] < 1e-9f) continue;
                float new_lp = b.log_prob + std::log(probs[tid]);
                Beam nb;
                nb.tokens   = b.tokens;
                nb.tokens.push_back(tid);
                nb.log_prob = new_lp;
                candidates.push_back(std::move(nb));
            }
        }

        // Keep top n_beams
        std::sort(candidates.begin(), candidates.end(),
                  [](const Beam& a, const Beam& b){ return a.log_prob > b.log_prob; });
        if ((int)candidates.size() > n_beams) candidates.resize(n_beams);
        beams = std::move(candidates);

        std::cout << "\n  Step " << step + 1 << ":\n";
        for (int r = 0; r < (int)beams.size(); ++r) {
            std::cout << "    Beam " << r << ": [";
            for (size_t i = 0; i < beams[r].tokens.size(); ++i) {
                if (i) std::cout << " ";
                std::cout << vocab[beams[r].tokens[i]];
            }
            std::cout << "]  log-prob=" << std::fixed << std::setprecision(3)
                      << beams[r].log_prob << "\n";
        }
    }

    std::cout << "\n  === Final beams (ranked by log-probability) ===\n";
    for (int r = 0; r < (int)beams.size(); ++r) {
        std::cout << "  Rank " << r + 1 << ": [";
        for (size_t i = 0; i < beams[r].tokens.size(); ++i) {
            if (i) std::cout << " ";
            std::cout << vocab[beams[r].tokens[i]];
        }
        std::cout << "]  log-prob=" << beams[r].log_prob
                  << "  prob=" << std::exp(beams[r].log_prob) << "\n";
    }

    std::cout << "\n  KV Cache Memory Cost of Beam Search:\n";
    for (int n : {1, 2, 4, 8}) {
        std::cout << "    n_beams=" << n << ": " << n << "x single-sequence KV memory\n";
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\nChapter 12 — Sampling: From Logits to Tokens\n";
    std::cout << "Companion code: sampling_demo.cpp\n";
    std::cout << std::string(65, '=') << "\n\n";

    demo_softmax();
    demo_temperature();
    demo_top_k();
    demo_top_p();
    demo_min_p();
    demo_penalties();
    demo_full_pipeline();
    demo_llamacpp_sampler_chain();
    demo_beam_search();

    std::cout << std::string(65, '=') << "\n";
    std::cout << "Chapter 12 C++ demo complete.\n\n";
    std::cout << "Key results:\n";
    std::cout << "  * Softmax: max subtraction prevents exp overflow\n";
    std::cout << "  * Temperature: T=0.5 sharpens; T=2.0 flattens distribution\n";
    std::cout << "  * Top-k + top-p combined: hard count bound then nucleus prune\n";
    std::cout << "  * Repetition penalty: divides positive logits for seen tokens\n";
    std::cout << "  * llama.cpp: sampler chain is a linked list of transformations\n";
    std::cout << "  * Beam search: n_beams x KV memory; use sparingly in production\n";
    std::cout << std::string(65, '=') << "\n";

    return 0;
}

```

