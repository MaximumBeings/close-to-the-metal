# Chapter 12 — Sampling: From Logits to Tokens

> *"The forward pass answers: given this context, what is the probability
> of every word in the vocabulary?  Sampling answers the harder question:
> which one do we pick?  Temperature is the difference between a model that
> always says the obvious thing and one that occasionally says something
> interesting."*

---

## 12.0 Why This Chapter Matters

The forward pass (Chapter 9) ends with a vector of 128 256 numbers — one
logit per token in the LLaMA 3 vocabulary.  The next token has not yet been
chosen.  Sampling is the decision procedure that converts that raw score
vector into a single token ID.

Sampling decisions have no effect on throughput or GPU utilization — they
run on CPU in microseconds.  But they have an enormous effect on output
quality, diversity, and safety.  Temperature determines whether the model
sounds creative or robotic.  Top-p determines whether it occasionally uses
rare words.  Repetition penalties stop it from looping.  Structured output
(JSON grammars) constrains it to produce syntactically valid data.

This chapter covers the full sampling pipeline, end to end:

**What you will understand after this chapter:**

- Why raw logits cannot be used directly as probabilities.
- How temperature, top-k, top-p, and min-p transform the logit distribution,
  with each step computed by hand.
- What repetition, frequency, and presence penalties do — and when they hurt.
- How structured output (JSON schema, GBNF grammars) works as a token mask
  applied before sampling.
- How beam search expands multiple hypotheses in parallel and how it interacts
  with the copy-on-write KV blocks from Chapter 6.
- How vLLM and llama.cpp implement each of these features.

**What you need first:**

- Chapter 9 (the forward pass produces logits).
- Chapter 6 (block table and copy-on-write, needed for beam search).

---

## 12.1 The Logit Vector  `[FOUNDATIONAL]`

The final layer of the model produces a vector of **logits** — one real-valued
score per vocabulary token.

```
Vocabulary size (LLaMA 3):   128 256 tokens
Logit vector shape:           [128 256]   ← one float per token
Dtype:                        FP32 (upcast from BF16 for numerical stability)

Example logits for a 5-token vocabulary (simplified):
  Token ID │ Token text │   Logit
  ─────────┼────────────┼────────
        0  │ "the"      │   4.20
        1  │ "a"        │   3.80
        2  │ "cat"      │   2.10
        3  │ "dog"      │   2.05
        4  │ "of"       │   1.30
```

Logits are **not** probabilities.  They can be negative, and they do not sum
to 1.  Converting logits to probabilities requires a **softmax**:

```
softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
```

Naïve softmax suffers from overflow when logits are large (exp(100) = ∞).
The numerically stable form subtracts the maximum first:

```
softmax(zᵢ) = exp(zᵢ − max(z)) / Σⱼ exp(zⱼ − max(z))
```

```
WORKED EXAMPLE 12.1 — Softmax on 5 Logits
─────────────────────────────────────────────────────────────────────
Given:   z = [4.20, 3.80, 2.10, 2.05, 1.30]

Step 1 — Subtract max (4.20):
  z' = [0.00, -0.40, -2.10, -2.15, -2.90]

Step 2 — Exponentiate:
  exp(z') = [1.000, 0.670, 0.122, 0.116, 0.055]

Step 3 — Sum:
  Σ = 1.000 + 0.670 + 0.122 + 0.116 + 0.055 = 1.963

Step 4 — Divide:
  p = [0.510, 0.341, 0.062, 0.059, 0.028]

Verification: 0.510 + 0.341 + 0.062 + 0.059 + 0.028 = 1.000 ✓

Interpretation:
  "the"  → 51.0% probability
  "a"    → 34.1%
  "cat"  → 6.2%
  "dog"  → 5.9%
  "of"   → 2.8%
─────────────────────────────────────────────────────────────────────
```

The sampling pipeline applies a series of transformations to the logits
**before** the softmax, shaping the distribution as desired.

```
Sampling pipeline:
  logits [V]
    → repetition penalty      (modify logits for previously seen tokens)
    → temperature scaling     (sharpen or flatten the distribution)
    → top-k filtering         (zero out all but top K logits)
    → top-p (nucleus) filter  (zero out tokens outside cumulative mass p)
    → min-p filter            (optional; remove tokens below min threshold)
    → softmax                 (convert to probabilities)
    → multinomial sample      (draw one token ID)
```

Each step is optional and configurable.  They are applied in the order shown.

---

## 12.2 Temperature  `[FOUNDATIONAL]`

Temperature controls the **sharpness** of the probability distribution.

```
scaled_logit = logit / temperature
```

- `temperature = 1.0` — unchanged; sample from the model's true distribution.
- `temperature < 1.0` — sharpens the distribution; the top token becomes more
  dominant; output is more focused and predictable.
- `temperature > 1.0` — flattens the distribution; rare tokens become more
  likely; output is more varied and creative.
- `temperature → 0` — equivalent to greedy decoding (always pick the argmax).
- `temperature → ∞` — uniform distribution over the vocabulary.

```
WORKED EXAMPLE 12.2 — Temperature Scaling
─────────────────────────────────────────────────────────────────────
Given:   z = [4.20, 3.80, 2.10, 2.05, 1.30]
         Probabilities at T=1.0: [0.510, 0.341, 0.062, 0.059, 0.028]

At T = 0.5 (sharper):
  z / 0.5 = [8.40, 7.60, 4.20, 4.10, 2.60]
  exp(z') = [exp(0), exp(-0.8), exp(-4.2), exp(-4.3), exp(-5.8)]
           = [1.000, 0.449, 0.015, 0.014, 0.003]
  sum = 1.481
  p  = [0.675, 0.303, 0.010, 0.009, 0.003]
  → "the" probability jumps from 51% to 68%

At T = 2.0 (flatter):
  z / 2.0 = [2.10, 1.90, 1.05, 1.025, 0.65]
  exp(z')  = [1.000, 0.819, 0.551, 0.537, 0.472]
  sum = 3.379
  p   = [0.296, 0.242, 0.163, 0.159, 0.140]
  → "the" probability drops from 51% to 30%; rare tokens "cat"/"dog"/"of"
    rise from ~6% to ~15–16% each

Final answer:
  T=0.5 → deterministic-leaning; "the" 68% of the time
  T=1.0 → model's true distribution
  T=2.0 → creative; every token has a real chance
─────────────────────────────────────────────────────────────────────
```

```
TEMPERATURE EFFECT ON DISTRIBUTION
────────────────────────────────────────────────────────────────────

T=0.5 (sharp):
  "the"  ████████████████████████████████████░░░░░░░░░░░░░░  68%
  "a"    ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  30%
  "cat"  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1%
  "dog"  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1%
  "of"   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%

T=1.0 (neutral):
  "the"  █████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░  51%
  "a"    █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  34%
  "cat"  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   6%
  "dog"  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   6%
  "of"   █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3%

T=2.0 (flat):
  "the"  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  30%
  "a"    ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  24%
  "cat"  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  16%
  "dog"  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  16%
  "of"   ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  14%
```

`[COMMON TRAP]` — Setting `temperature=0` in most frameworks does not divide
by zero — it falls back to argmax (greedy decoding).  But setting
`temperature=0.0001` will cause the softmax to compute `exp(logit / 0.0001)`,
producing extremely large exponents that overflow to `inf`, resulting in NaN
probabilities.  Use `temperature=0` explicitly for greedy decoding; never use
very small positive values.

---

## 12.3 Top-k Filtering  `[FOUNDATIONAL]`

Top-k filtering keeps only the `k` highest-logit tokens and sets all others
to `-∞` (effectively zero probability after softmax).

```
WORKED EXAMPLE 12.3 — Top-k Filtering (k=2)
─────────────────────────────────────────────────────────────────────
Given:   z = [4.20, 3.80, 2.10, 2.05, 1.30]   (after temperature)
         k = 2

Step 1 — Find the 2nd largest value (threshold):
  Sorted descending: [4.20, 3.80, 2.10, 2.05, 1.30]
  Threshold = 3.80  (value of the k-th element)

Step 2 — Set all logits below threshold to -∞:
  z_filtered = [4.20, 3.80, -∞, -∞, -∞]

Step 3 — Softmax on filtered logits:
  exp([4.20, 3.80, -∞, -∞, -∞] - 4.20) = [1.000, 0.670, 0, 0, 0]
  sum = 1.670
  p = [0.599, 0.401, 0, 0, 0]

Final answer:
  Only "the" (60%) and "a" (40%) are candidates.
  "cat", "dog", "of" cannot be sampled.
─────────────────────────────────────────────────────────────────────
```

Top-k is deterministic in which tokens survive — it always picks the `k`
highest regardless of how much probability mass they hold.  With a very flat
distribution (temperature=2.0), the top-k tokens might collectively hold only
60% of the probability mass, so the filtering is aggressive.

`[COMMON TRAP]` — Top-k with k=1 is equivalent to greedy decoding.  Top-k with
k = vocabulary_size (128 256 for LLaMA 3) disables filtering entirely.

---

## 12.4 Top-p (Nucleus) Sampling  `[FOUNDATIONAL]`

Top-p sampling — introduced by Holtzman et al. (2019) as **nucleus sampling**
— keeps the smallest set of tokens whose cumulative probability exceeds `p`.

Unlike top-k, the number of tokens retained is **adaptive**: it depends on
the shape of the distribution.  When the model is confident (one token has
95% probability), top-p=0.9 retains just that token.  When the model is
uncertain (probability spread across 50 tokens), top-p=0.9 retains many.

```
WORKED EXAMPLE 12.4 — Top-p Filtering (p=0.9)
─────────────────────────────────────────────────────────────────────
Given:   probabilities after temperature (T=1.0):
  "the"  → 0.510
  "a"    → 0.341
  "cat"  → 0.062
  "dog"  → 0.059
  "of"   → 0.028

Step 1 — Sort by probability, descending:
  "the"  0.510  | cumsum = 0.510
  "a"    0.341  | cumsum = 0.851
  "cat"  0.062  | cumsum = 0.913  ← first cumsum ≥ 0.9

Step 2 — Nucleus: all tokens up to and including the one that pushes
  cumsum past p=0.9:
  Nucleus = {"the", "a", "cat"}

Step 3 — Set all tokens outside nucleus to -∞, renormalize:
  p_nucleus = [0.510, 0.341, 0.062] / (0.510 + 0.341 + 0.062)
            = [0.510, 0.341, 0.062] / 0.913
            = [0.559, 0.373, 0.068]

Final answer:
  Sampling pool: "the" (55.9%), "a" (37.3%), "cat" (6.8%)
  "dog" and "of" are excluded (they fall outside the top-90% nucleus)
─────────────────────────────────────────────────────────────────────
```

Top-p and top-k are frequently **combined**.  Applied in sequence (top-k first,
then top-p), they first bound the candidate count then further prune by
cumulative mass:

```
Common production defaults:
  temperature = 0.7
  top_k       = 50     ← first bound: keep at most 50 candidates
  top_p       = 0.9    ← then prune to 90% cumulative mass
```

---

## 12.5 Min-p Filtering  `[DEEP DIVE]`

Min-p (Nguyen et al., 2023) is an alternative to top-p that scales the
threshold relative to the **top token's probability** rather than using a
fixed cumulative mass.

```
min_p threshold = min_p_param × p_top

A token survives if:  p_token ≥ min_p_param × p_top
```

```
WORKED EXAMPLE 12.5 — Min-p Filtering (min_p=0.05)
─────────────────────────────────────────────────────────────────────
Given:   p = [0.510, 0.341, 0.062, 0.059, 0.028]
         p_top = 0.510 (highest probability token)
         min_p = 0.05

Threshold = 0.05 × 0.510 = 0.0255

Survive if p_token ≥ 0.0255:
  "the"  0.510 ≥ 0.0255 → ✓
  "a"    0.341 ≥ 0.0255 → ✓
  "cat"  0.062 ≥ 0.0255 → ✓
  "dog"  0.059 ≥ 0.0255 → ✓
  "of"   0.028 ≥ 0.0255 → ✓  (just barely)

All 5 tokens survive.  Compare with top-p=0.9 which excluded "dog" and "of".

At a confident step (p_top = 0.95):
  threshold = 0.05 × 0.95 = 0.0475
  Most rare tokens (p < 0.0475) are pruned.

At an uncertain step (p_top = 0.20):
  threshold = 0.05 × 0.20 = 0.01
  Many tokens survive.
─────────────────────────────────────────────────────────────────────
```

Min-p adapts more gracefully than top-p when the model alternates between
confident steps (factual recall) and uncertain steps (creative generation).
It is available in vLLM via `SamplingParams(min_p=0.05)` and in llama.cpp
via `llama_sampler_init_min_p`.

---

## 12.6 Repetition, Frequency, and Presence Penalties  `[FOUNDATIONAL]`

Without penalties, language models tend to repeat themselves — especially
in open-ended generation.  Three logit-level penalties discourage repetition.

All three operate on the **raw logits** before temperature is applied.

### 12.6.1 Repetition penalty (multiplicative)

```
Penalty formulation (Keskar et al., 2019):

  If token t has appeared in the context:
    logit[t] = logit[t] / repetition_penalty   (if logit[t] > 0)
    logit[t] = logit[t] × repetition_penalty   (if logit[t] < 0)
```

Intuition: penalising a positive logit by dividing makes it smaller (less
likely).  Penalising a negative logit by multiplying makes it more negative
(even less likely).  A value of 1.0 means no penalty; 1.3 is a mild penalty.

```
WORKED EXAMPLE 12.6 — Repetition Penalty
─────────────────────────────────────────────────────────────────────
Given:   logit["cat"] = 2.10
         "cat" appeared at positions 3, 7 in the previous context
         repetition_penalty = 1.3

Step 1 — logit > 0, so divide:
  penalized_logit = 2.10 / 1.3 = 1.615

Step 2 — Effect on probability (all else equal):
  Original: exp(2.10) ≈ 8.17
  penalized: exp(1.615) ≈ 5.03
  Probability reduction: 5.03 / 8.17 ≈ 62% of original

Final answer: "cat" probability drops to 62% of its unpenalized value
  because it appeared earlier in the context.
─────────────────────────────────────────────────────────────────────
```

### 12.6.2 Frequency penalty (additive, count-proportional)

```
logit[t] -= frequency_penalty × count(t, context)

count(t, context) = number of times token t appeared in the context
```

The more often a token appears, the harder it is penalized.  Useful for
long-form generation where moderate repetition is fine but loops are not.

### 12.6.3 Presence penalty (additive, binary)

```
logit[t] -= presence_penalty   if count(t, context) > 0
```

A flat penalty applied once if the token appeared at all — regardless of
how many times.  Useful for topic diversity: once a concept appears, nudge
the model to explore something new.

```
Comparison:
┌──────────────────────┬──────────────────────────┬──────────────────────┐
│ Penalty type         │ How penalty scales        │ Best for             │
├──────────────────────┼──────────────────────────┼──────────────────────┤
│ Repetition (mult.)   │ Binary: seen or not       │ Preventing any loop  │
│ Frequency (add.)     │ Linear in count           │ Reducing overuse     │
│ Presence (add.)      │ Binary: seen or not       │ Topic diversity      │
└──────────────────────┴──────────────────────────┴──────────────────────┘
```

`[COMMON TRAP]` — High repetition penalties (> 1.5) push the model toward
unusual tokens that were not in the context.  This can degrade coherence.
Values in the range 1.0–1.3 are typically safe.  Frequency and presence
penalties above 0.5 can prevent the model from repeating necessary function
words ("the", "a", "of").

---

## 12.7 Structured Output  `[FOUNDATIONAL]`

Structured output forces the model to produce text that conforms to a schema
(JSON, a grammar, a regex pattern).  The mechanism is a **token mask** —
a binary vector of length V applied to the logits before sampling.  Tokens
that would violate the schema at the current position are set to `-∞`.

### 12.7.1 How token masking works

```
At each decode step, the structured output engine:

1. Tracks the current parse state (e.g., position in a JSON grammar).
2. Computes the set of tokens that are valid at this position.
3. Builds a mask: allowed[t] = 1 if token t is valid, 0 otherwise.
4. Applies: logit[t] = logit[t] if allowed[t] else -∞

Only valid tokens can be sampled.  The model's distribution is renormalized
over the allowed set.
```

```
EXAMPLE — JSON schema enforcement
──────────────────────────────────────────────────────────────────────
Schema:  {"name": string, "age": integer}
Current generation:  '{"name": "Alice", "age": '
Current parse state: expecting an integer (digits only)

Allowed tokens at this step:
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
  → 10 tokens allowed out of 128 256

All other tokens (letters, punctuation, special chars) → masked to -∞

The model cannot produce "Alice" or "null" or "}" here — only digits.
```

### 12.7.2 vLLM: Outlines integration

vLLM integrates the **Outlines** library (Willard & Louf, 2023) for structured
generation.  Outlines pre-compiles a finite-state machine (FSM) from a JSON
schema or regex, then steps the FSM forward at each decode step to produce
the token mask.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

sampling_params = SamplingParams(
    temperature=0.7,
    guided_json={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age":  {"type": "integer"},
        },
        "required": ["name", "age"],
    }
)

output = llm.generate(
    ["Extract the person's details: Alice is 30 years old."],
    sampling_params,
)
# output is guaranteed to be valid JSON: {"name": "Alice", "age": 30}
```

The FSM is compiled once per unique schema.  At each decode step, vLLM calls
`outlines.fsm.get_next_instruction(state)` to get the allowed token set
(typically a bitmask over the vocabulary), then applies it to the logits.

### 12.7.3 llama.cpp: GBNF grammars

llama.cpp uses **GBNF** (GGML BNF) — a context-free grammar format that
describes the allowed output structure.

```
# GBNF grammar for a simple JSON object with name (string) and age (integer)

root   ::= object
object ::= "{" ws "\"name\"" ws ":" ws string "," ws "\"age\"" ws ":" ws integer "}" ws
string ::= "\"" [a-zA-Z ]* "\""
integer ::= [0-9]+
ws     ::= [ \t\n]*
```

At each decode step, llama.cpp's grammar sampler:
1. Advances the grammar parser over the token just accepted.
2. Queries the grammar for the set of valid next characters/tokens.
3. Builds a token mask and applies it to the logit vector.

GBNF is more expressive than JSON schema alone — it can enforce arbitrary
context-free languages — but requires writing the grammar manually.

### 12.7.4 Performance cost of structured output

Token masking requires iterating over the vocabulary (up to 128 256 tokens)
each step to apply the mask.  This is a CPU-side operation taking ~0.1–1 ms
per step depending on mask density and vocabulary size.  For high-throughput
deployments:

```
Cost model:
  Masking overhead ≈ vocabulary_size × 1 bit / memory_bandwidth
  For 128K vocab, dense mask: 128K bits = 16 KB  → negligible memory
  CPU iteration:  ~100 μs per step (modern CPU, pre-compiled FSM)
  At 100 tokens/s per user: 100 × 0.1 ms = 10 ms/s → ~1% overhead

Overhead is small for individual users.
At 1000 concurrent structured-output users: 1000 × 0.1 ms = 100 ms/step
  → can become significant at very high concurrency.
vLLM batches the mask computation across all sequences in the batch.
```

---

## 12.8 Beam Search  `[DEEP DIVE]`

Greedy decoding and sampling both make **locally optimal** token choices —
one token at a time.  Beam search maintains `n` parallel hypotheses
(beams) and expands each by one token per step, keeping only the `n` most
probable hypotheses at each step.

### 12.8.1 The beam search algorithm

```
BEAM SEARCH — 2 beams, 3 steps, 5-token vocabulary
──────────────────────────────────────────────────────────────────────
Initial state:
  Beam 0: "" — log-prob = 0.0
  Beam 1: "" — log-prob = 0.0

Step 1 — Expand each beam over vocabulary (pick top 2 total):
  From Beam 0: "the" (log-p = -0.67), "a" (log-p = -1.08), ...
  From Beam 1: same (beams are identical at step 1)
  Top 2 across all candidates:
    Beam 0: "the"     log-prob = -0.67
    Beam 1: "a"       log-prob = -1.08

Step 2 — Expand each beam:
  From Beam 0 ("the"): "the cat" (-0.67 + -2.77 = -3.44)
                        "the dog" (-0.67 + -2.83 = -3.50) ...
  From Beam 1 ("a"):   "a cat"   (-1.08 + -2.08 = -3.16)
                        "a dog"   (-1.08 + -2.19 = -3.27) ...
  Top 2:
    Beam 0: "a cat"   log-prob = -3.16
    Beam 1: "a dog"   log-prob = -3.27

Step 3 — Continue until EOS or max_tokens.
Final answer: return the highest log-prob completed sequence.
```

Beam search finds a sequence with higher overall log-probability than greedy
decoding — but it is **not** equivalent to sampling.  Beam search always
produces the same output for the same input (it is deterministic).

### 12.8.2 Beam search and the KV cache

Each beam is a separate sequence.  `n=4` beams means 4 sequences in the KV
cache simultaneously.  This interacts with the block table from Chapter 6:

```
Beam divergence and copy-on-write:
──────────────────────────────────────────────────────────────────────
At step 0, all beams share the same prompt KV blocks:
  Block table:
    Beam 0 → [Block 7, Block 8] (prompt blocks, ref_count = 4)
    Beam 1 → [Block 7, Block 8]
    Beam 2 → [Block 7, Block 8]
    Beam 3 → [Block 7, Block 8]

After step 1, each beam diverges to a different token:
  Copy-on-write: new decode blocks allocated per beam
  Beam 0 → [Block 7, Block 8, Block 12]  (ref_count 7,8 stays 4)
  Beam 1 → [Block 7, Block 8, Block 13]
  Beam 2 → [Block 7, Block 8, Block 14]
  Beam 3 → [Block 7, Block 8, Block 15]

Memory cost of beam search (n beams):
  = n × (prompt_blocks_per_seq + decode_blocks_per_step × steps)
  ≈ n × memory_per_greedy_decode

At n=4: 4× the KV memory of a single greedy sequence.
At n=8: 8× — frequently causes OOM for long sequences.
```

`[COMMON TRAP]` — Beam search with `n=4` on a 70B model can exhaust HBM
even if a single greedy decode fits comfortably.  vLLM limits concurrent
beam search requests via `max_num_seqs / n`.  For production serving with
low latency requirements, sampling (top-p/temperature) is almost always
preferred over beam search.

### 12.8.3 When to use beam search

```
Beam search is best for:
  ✓ Machine translation (fixed-length source-target alignment)
  ✓ summarization (short, factual output)
  ✓ Code generation (syntactically constrained output)
  ✗ Open-ended chat (beam search produces repetitive, "safe" text)
  ✗ Creative writing (beam search eliminates diversity by design)
  ✗ High-throughput serving (4–8× KV memory cost)
```

---

## 12.9 The Sampling Pipeline in vLLM and llama.cpp  `[FOUNDATIONAL]`

### 12.9.1 vLLM: SamplingParams

```python
from vllm import SamplingParams

params = SamplingParams(
    temperature=0.7,        # logit scaling
    top_k=50,               # keep top-50 logits
    top_p=0.9,              # then prune to 90% cumulative mass
    min_p=0.0,              # min-p disabled (0.0 = off)
    repetition_penalty=1.1, # mild penalty for repeated tokens
    frequency_penalty=0.0,  # additive frequency penalty
    presence_penalty=0.0,   # additive presence penalty
    max_tokens=512,         # hard generation limit
    n=1,                    # number of completions (n>1 = beam/multi-sample)
    use_beam_search=False,  # greedy/sampling by default
    best_of=1,              # sample this many, return top-1
    stop=["</s>", "\n\n"],  # stop sequences
    logprobs=None,          # return log-probs if not None
)
```

All sampling logic runs in vLLM's **sampler** — a CPU-side module called
once per scheduler step after the forward pass completes.

### 12.9.2 llama.cpp: sampler chain

llama.cpp implements sampling as a **linked chain** of sampler objects.
Each sampler in the chain transforms the `llama_token_data_array` (the
logit/probability array) in sequence.

```c
// Build a sampler chain (C API)
struct llama_sampler * smpl = llama_sampler_chain_init(
    llama_sampler_chain_default_params()
);

// Add samplers in application order:
llama_sampler_chain_add(smpl,
    llama_sampler_init_penalties(
        n_vocab,
        special_eos_id,
        linefeed_id,
        /*penalty_last_n=*/64,      // look back this many tokens
        /*penalty_repeat=*/1.1f,    // repetition penalty
        /*penalty_freq=*/0.0f,      // frequency penalty
        /*penalty_present=*/0.0f,   // presence penalty
        /*penalize_nl=*/false,
        /*ignore_eos=*/false
    )
);

llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, /*min_keep=*/1));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(/*seed=*/42));

// Sample one token:
llama_token next_token = llama_sampler_sample(smpl, ctx, -1);
llama_sampler_accept(smpl, next_token);  // update penalty history
```

Each `llama_sampler_chain_add` appends a transformation stage.  The chain
is applied in insertion order — the same pipeline described in §12.1.

### 12.9.3 Side-by-side comparison

```
┌────────────────────────┬──────────────────────────┬──────────────────────────────────┐
│ Feature                │ vLLM                     │ llama.cpp                        │
├────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ Temperature            │ SamplingParams.temperature│ llama_sampler_init_temp()        │
│ Top-k                  │ SamplingParams.top_k      │ llama_sampler_init_top_k()       │
│ Top-p                  │ SamplingParams.top_p      │ llama_sampler_init_top_p()       │
│ Min-p                  │ SamplingParams.min_p      │ llama_sampler_init_min_p()       │
│ Repetition penalty     │ repetition_penalty        │ llama_sampler_init_penalties()   │
│ Frequency penalty      │ frequency_penalty         │ (same penalties struct)          │
│ Presence penalty       │ presence_penalty          │ (same penalties struct)          │
│ Beam search            │ use_beam_search=True      │ not built-in (manual)            │
│ Structured output(JSON)│ guided_json= (Outlines)   │ grammar= (GBNF)                  │
│ Stop sequences         │ stop=["..."]              │ llama_sampler_init_grammar()     │
│ Seed                   │ seed=                     │ llama_sampler_init_dist(seed)    │
│ Multiple completions   │ n=                        │ manual loop                      │
└────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

---

## 12.10 Logit Post-Processing: Order Matters  `[DEEP DIVE]`

The order in which samplers are applied significantly affects output quality.
The pipeline in §12.1 is the standard order, but it is worth understanding
why.

```
CORRECT ORDER:
  1. Repetition penalties  (modify logits for seen tokens)
  2. Temperature           (scale all logits)
  3. Top-k                 (hard count filter)
  4. Top-p                 (cumulative mass filter)
  5. Min-p                 (relative threshold filter)
  6. Softmax + sample

WHY THIS ORDER:

  Penalties before temperature:
    Penalties subtract/divide raw logits.  If temperature is applied first,
    the logit scale changes, making penalty magnitudes inconsistent across
    different temperatures.

  Top-k before top-p:
    Top-k provides a hard upper bound on candidates.  Top-p then further
    prunes.  Reversing them (top-p first) can occasionally allow more than
    k tokens if the nucleus is large, defeating the purpose of top-k as
    a safety bound.

  Min-p after top-p (or instead of top-p):
    Min-p and top-p serve similar purposes.  Typically one is used, not both.
    If both are used, apply min-p last as an additional quality gate.
```

`[COMMON TRAP]` — Some inference frameworks apply temperature **after** top-k,
which changes which tokens survive the cut.  Top-k at T=1.0 (keeping tokens
with the 50 highest raw logits) differs from top-k applied to T=0.5-scaled
logits (the ranking is the same, but edge cases around tie-breaking differ).
Always check your framework's application order before comparing results.

---

## 12.11 Worked End-to-End Example  `[FOUNDATIONAL]`

Full pipeline: one decode step from logits to sampled token.

```
WORKED EXAMPLE 12.7 — Full Sampling Pipeline
─────────────────────────────────────────────────────────────────────
Vocabulary: 5 tokens (simplified)
Context so far: "The cat sat on the"
Target: sample the next token

Raw logits from forward pass:
  "mat"   → 4.50
  "floor" → 3.90
  "cat"   → 2.80   ← "cat" appeared earlier in context
  "roof"  → 1.60
  "the"   → 3.20   ← "the" appeared earlier in context

Config: repetition_penalty=1.2, temperature=0.8, top_k=3, top_p=0.92

STEP 1 — Repetition penalty (penalty=1.2):
  "cat" (logit 2.80 > 0): 2.80 / 1.2 = 2.333
  "the" (logit 3.20 > 0): 3.20 / 1.2 = 2.667
  Updated logits:
    "mat"   4.500
    "floor" 3.900
    "cat"   2.333   ← penalized
    "roof"  1.600
    "the"   2.667   ← penalized

STEP 2 — Temperature (T=0.8):
  Divide all logits by 0.8:
    "mat"   5.625
    "floor" 4.875
    "cat"   2.917
    "roof"  2.000
    "the"   3.333

STEP 3 — Top-k (k=3):
  Keep top 3: "mat" (5.625), "floor" (4.875), "the" (3.333)
  Set others to -∞:
    "mat"   5.625
    "floor" 4.875
    "cat"   -∞
    "roof"  -∞
    "the"   3.333

STEP 4 — Softmax over surviving logits:
  Subtract max (5.625):
    "mat"   0.000  → exp = 1.000
    "floor" -0.750 → exp = 0.472
    "the"   -2.292 → exp = 0.101
  Sum = 1.573
  Probabilities:
    "mat"   0.636
    "floor" 0.300
    "the"   0.064

STEP 5 — Top-p (p=0.92):
  Cumulative:
    "mat"   0.636  cumsum = 0.636
    "floor" 0.300  cumsum = 0.936 ← exceeds 0.92
  Nucleus: {"mat", "floor"} (first 2 push cumsum ≥ 0.92)
  Renormalize:
    "mat"   0.636 / 0.936 = 0.679
    "floor" 0.300 / 0.936 = 0.321

STEP 6 — Sample:
  Draw from {"mat": 0.679, "floor": 0.321}
  → Most likely: "mat"

Final answer: next token = "mat"
  Sequence becomes: "The cat sat on the mat"
─────────────────────────────────────────────────────────────────────
```

---

## 12.12 Code Listing  `[FOUNDATIONAL]`

See `code/chapter_12/sampling_demo.py` for:
- Full logit pipeline visualiser: apply each stage and print the
  distribution after every transformation
- Temperature, top-k, top-p, and min-p implemented from scratch
- Repetition, frequency, and presence penalty simulation
- Structured output token mask demo (JSON schema enforcement)
- Beam search trace with log-probability tracking

See `code/chapter_12/sampling_demo.cpp` for:
- llama.cpp sampler chain construction and inspection
- Manual temperature + top-k + top-p + penalties pipeline in C++
- GBNF grammar snippet for JSON object enforcement
- Beam search state machine with copy-on-write block tracking

---

## 12.13 Summary

```
Key takeaways:

1. Raw logits must be converted to probabilities via softmax.
   Use the numerically stable form: subtract max before exponentiating.

2. Temperature scales logits before softmax.
   T < 1.0 → sharper (more greedy); T > 1.0 → flatter (more random).
   Never use T ≈ 0.0001 — use T=0 for greedy decoding explicitly.

3. Top-k keeps the K highest-logit tokens.
   Top-p keeps the smallest nucleus with cumulative probability ≥ p.
   They are typically combined: top-k first, then top-p.

4. Min-p thresholds relative to the top token's probability.
   More adaptive than top-p for variable-confidence generation.

5. Repetition penalty divides logits for seen tokens (multiplicative).
   Frequency penalty subtracts proportionally to occurrence count.
   Presence penalty subtracts once for any token that appeared at all.
   Keep repetition_penalty ≤ 1.3 to avoid incoherence.

6. Structured output applies a token mask from a compiled FSM or grammar.
   Only syntactically valid tokens can be sampled at each step.
   ~0.1 ms CPU overhead per step — negligible at typical concurrency.

7. Beam search maintains n parallel hypotheses.
   KV memory cost = n × single-sequence memory.
   Prefer sampling (top-p + temperature) for production serving;
   reserve beam search for translation and summarization.

8. Sampler order matters:
   penalties → temperature → top-k → top-p → softmax → sample.
```

---

## 12.14 Self-Check Questions

1. The logits for step N are `[-1.2, 0.5, 3.1, 2.8, 0.0]`.  Compute the
   softmax probabilities by hand (numerically stable form).

2. With `top_p=0.95` and `temperature=0.5`, would you expect the nucleus
   (the set of surviving tokens) to be larger or smaller than at `temperature=1.5`?
   Explain why.

3. A user reports that with `repetition_penalty=2.0`, the model keeps
   producing gibberish after the first 20 tokens.  What is the likely cause
   and what value would you recommend?

4. You need to generate valid JSON from a 70B model in production with
   10 000 req/s.  Estimate the CPU overhead of token masking in milliseconds
   per second across the fleet.

5. A request with `use_beam_search=True, n=8` is submitted to a vLLM instance
   serving a 13B model on a single A100-40GB.  The model alone uses 26 GB.
   Each sequence context window is 2 048 tokens.  Estimate whether the KV
   cache will fit, and what `max_num_seqs` cap would prevent OOM.

---

## Where We Go Next

Chapter 13 closes out Part II with **token streaming** — the final mile
between the sampled token ID and the character that appears in the user's
browser.  Detokenization handles multi-byte UTF-8 at character boundaries.
Server-Sent Events carry each token over HTTP as it is generated.  Backpressure
and client cancellation are handled by vLLM's async generator.  llama.cpp's
built-in server implements the same SSE protocol in fewer lines of C++.

*Next: Chapter 13 — Token Streaming: The Last Mile*


---

## Chapter Summary

- **Token selection is the last GPU op**: after the final logit matmul, sampling selects the next token; this step is fast but has significant quality impact.
- **Greedy decoding**: argmax over logits; fully deterministic but prone to repetition and sub-optimal local choices.
- **Temperature scaling**: divides logits by T before softmax; T < 1 sharpens the distribution (more greedy), T > 1 flattens it (more random).
- **Top-k sampling**: zeroes all logits except the top-k; prevents sampling from the long tail of improbable tokens.
- **Top-p (nucleus) sampling**: keeps the smallest set of tokens whose cumulative probability ≥ p; adapts to the shape of the distribution at each step.
- **Repetition penalty**: divides the logit of tokens already in the context, reducing repetition; can hurt quality if set too high.
- **Min-p sampling**: keeps tokens whose probability is at least `min_p × (probability of top token)`; more robust to distribution width variation than fixed k.
- **Beam search in vLLM**: requires CoW for KV cache forking; typically avoided in production due to 4–8× memory overhead for beam width 4–8.
- **Structured decoding (outlines)**: masks logits at each step to enforce a regex, JSON schema, or grammar, enabling guaranteed-valid structured output.

---

## Self-Check Questions

1. The top-5 logits at a decode step are [3.1, 2.9, 2.1, 0.8, 0.1]. After softmax, compute the top-5 probabilities. With temperature T = 0.5, recompute the probabilities. How does the entropy change? *(Section 12.2)*

2. Top-p = 0.9 sampling is applied to the distribution above. Which tokens are included in the nucleus? Show your cumulative probability calculation. *(Section 12.3)*

3. A user reports that a long-context story generation becomes repetitive after 2 000 tokens. You suspect repetition penalty. (a) What value would you set? (b) What side effects might a high penalty introduce? *(Section 12.4)*

4. Beam search with width 4 runs for 200 decode steps. Each step, CoW may be triggered when two beams diverge. In the worst case, how many CoW copies occur? What is the maximum KV cache overhead relative to greedy decoding? *(Section 12.5)*

5. A JSON schema requires the model to output `{"name": "...", "age": N}`. Describe how structured decoding masks logits at the step where `"age": ` has just been emitted to enforce an integer. *(Section 12.6)*
