# Chapter 12.5 — Structured Generation and Constrained Decoding

> *"JSON mode is not a feature. It is a finite state machine wearing a language model's coat."*

---

## 12.5.1 Why Unconstrained Generation Fails for Structured Outputs

Chapter 12 described how a language model produces tokens: logits are computed
over the vocabulary, a sampling strategy selects a token, and the process
repeats. Nothing in this process guarantees that the output is valid JSON,
valid Python, or a valid phone number.

In production, this matters. A pipeline that extracts structured data from
documents and feeds it to a database cannot tolerate malformed JSON. A function-
calling agent cannot tolerate a tool name that doesn't exist. An API that
promises a numeric rating between 1 and 10 cannot return "seven-ish".

The naive solution is prompt engineering — "respond only with valid JSON" — and
retries. In practice, even instruction-tuned models fail on valid JSON 5–15% of
the time on complex schemas, and retries double or triple latency at the tail.

The correct solution is **constrained decoding**: modify the logits at each
step to set the probability of invalid continuations to zero. The model can
only generate tokens that keep the output on a valid path toward the target
structure.

---

## 12.5.2 The Core Mechanism: Logit Masking

At each decode step, before sampling, a constraint processor computes a
**token mask**: a boolean vector over the vocabulary indicating which tokens
are valid continuations given the current generation state.

```
logits: [-2.1, 0.3, 1.8, -0.5, 0.9, ...]   (shape: vocab_size)
mask:   [  0,   1,   1,    0,   1, ...]     (1 = valid, 0 = forbidden)

masked_logits[i] = logits[i] if mask[i] else -inf

softmax(masked_logits) → probability only over valid tokens
```

The model's distribution is not changed — its logits for valid tokens are
preserved. The constraint only eliminates tokens that would lead to
syntactically invalid output. The model's "opinion" about which valid token
is most probable is respected.

This masking is applied at **every decode step**. The state of the constraint
processor advances with each emitted token, tracking what is currently legal
given what has been emitted so far.

---

## 12.5.3 Finite State Machines for JSON

JSON is a regular-ish language: its structure can be captured by a finite state
machine (FSM). An FSM for JSON tracks states like:

```
STATES:
  START         → expects '{' or '['
  OBJECT_OPEN   → expects '"' (key) or '}'
  IN_KEY        → expects any string characters or '"'
  AFTER_KEY     → expects ':'
  IN_VALUE      → expects any value start token
  AFTER_VALUE   → expects ',' or '}'
  ... (simplified)
```

At each state, only a subset of vocabulary tokens can advance the FSM without
entering a dead (invalid) state. The constraint processor maps the current FSM
state to a token mask.

**Worked Example 12.5.1 — JSON object mask**

```python
# Current generation: '{"name": "'
# FSM state: IN_STRING_VALUE (inside a string value)
# Valid next tokens: any character except unescaped '"' and control chars
# Invalid: '{', '[', '}', ']', ',' (would break the string)

# After emitting 'John':
# Generation: '{"name": "John'
# Valid next tokens: any string char OR '"' (close the string value)
```

The mask changes character-by-character. After the closing `"`, the FSM moves
to AFTER_VALUE, and the only valid tokens become `,` or `}`.

### Tokenizer alignment problem

Here lies the hard part. LLM tokenizers operate over **subword tokens**, not
individual characters. The token `"name"` might be a single token, not 6
characters. The FSM needs to work at the same granularity as the tokenizer.

The solution used by `outlines` and `lm-format-enforcer` is to **pre-compute
the token-to-FSM transition table** offline: for each (state, token) pair,
determine whether emitting that token is valid, and if so, what state it leads
to. This pre-computation is done once per (schema, tokenizer) pair and cached.

```python
# Conceptual pre-computation
transition_table = {}   # (state, token_id) → next_state or INVALID

for state in fsm.states:
    for token_id, token_str in enumerate(tokenizer.vocab):
        next_state = fsm.transition(state, token_str)
        transition_table[(state, token_id)] = next_state

# At inference: O(1) lookup per token
next_state = transition_table.get((current_state, sampled_token_id), INVALID)
```

For a 128,000-token vocabulary and an FSM with 200 states, this table has
25.6 million entries but is computed once per schema.

---

## 12.5.4 The `outlines` Library

`outlines` is the reference open-source implementation of FSM-based constrained
decoding. It supports JSON schema, regular expressions, Pydantic models, and
context-free grammars.

### JSON Schema

```python
from pydantic import BaseModel
from typing import List
import outlines

class UserProfile(BaseModel):
    name: str
    age: int
    tags: List[str]

model = outlines.models.transformers(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device="cuda"
)

generator = outlines.generate.json(model, UserProfile)

result = generator(
    "Extract the user profile from: John is 28 years old and likes Python and hiking."
)
# result is a UserProfile instance — guaranteed valid, no retries needed
print(result.name)    # "John"
print(result.age)     # 28
print(result.tags)    # ["Python", "hiking"]
```

### Regular expressions

```python
# Extract a phone number in E.164 format
generator = outlines.generate.regex(model, r"\+[1-9]\d{7,14}")
phone = generator("Contact number from: Call us at +14155552671 for support.")
# phone == "+14155552671" — guaranteed to match the regex
```

### Choice from a list

```python
# Constrain output to one of a fixed set of values
generator = outlines.generate.choice(model, ["positive", "neutral", "negative"])
sentiment = generator("Sentiment of: 'This product exceeded my expectations!'")
# sentiment == "positive" — exactly one of the three options
```

---

## 12.5.5 `lm-format-enforcer`

`lm-format-enforcer` takes a different implementation approach: instead of a
pre-computed transition table, it uses a **character-level trie** over the
vocabulary to efficiently find valid continuations at each step.

```python
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn
)
from transformers import pipeline

schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["sentiment", "confidence"]
}

parser = JsonSchemaParser(schema)
prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = pipe(
    prompt,
    prefix_allowed_tokens_fn=prefix_fn,
    max_new_tokens=50
)
```

The key difference from `outlines`: `lm-format-enforcer` computes valid tokens
lazily at inference time rather than pre-computing a full table. This has
lower startup cost but higher per-step overhead. For schemas that are generated
dynamically (user-provided schemas), the lazy approach is often preferable.

---

## 12.5.6 Context-Free Grammars: Beyond Regular Languages

JSON can be approximately captured with an FSM, but some structures require
more expressive power. Arbitrarily nested JSON (arrays of objects of arrays)
is context-free, not regular. Programming languages are context-free. SQL is
approximately context-free.

Context-free grammars (CFGs) and their EBNF notation are supported by both
`outlines` and llama.cpp's grammar system.

**EBNF grammar for a simplified arithmetic expression:**

```ebnf
root    ::= expr
expr    ::= term (("+" | "-") term)*
term    ::= factor (("*" | "/") factor)*
factor  ::= number | "(" expr ")"
number  ::= [0-9]+
```

At each decode step, the grammar parser determines which tokens can legally
continue the current partial parse. Only those tokens are unmasked.

### llama.cpp grammar sampling

llama.cpp has native EBNF grammar support via its `llama_grammar` API:

```cpp
// C API
const char* grammar_str = R"(
root   ::= object
value  ::= object | array | string | number | bool | null
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array  ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
bool   ::= "true" | "false"
null   ::= "null"
ws     ::= [ \t\n]*
)";

const char* grammar_rule_names[] = {"root", "value", "object",
                                     "array", "string", "number",
                                     "bool", "null", "ws"};
llama_grammar* grammar = llama_grammar_init_impl(
    grammar_str, grammar_rule_names,
    sizeof(grammar_rule_names) / sizeof(grammar_rule_names[0])
);

// Apply grammar during sampling
llama_sample_grammar(ctx, candidates, grammar);
llama_grammar_accept_token(grammar, sampled_token);
```

```bash
# CLI usage
llama-cli \
    --model llama-3.1-8b-q4_k_m.gguf \
    --grammar-file json.gbnf \
    --prompt "Extract: John is 28 years old."
```

Common grammar files are distributed with llama.cpp (`grammars/json.gbnf`,
`grammars/json_arr.gbnf`, `grammars/list.gbnf`).

---

## 12.5.7 vLLM Guided Decoding

vLLM integrates with `outlines` and `lm-format-enforcer` through its
`GuidedDecodingParams`:

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct")

schema = {
    "type": "object",
    "properties": {
        "name":  {"type": "string"},
        "score": {"type": "integer", "minimum": 1, "maximum": 10}
    },
    "required": ["name", "score"]
}

params = SamplingParams(
    temperature=0.0,
    max_tokens=100,
    guided_decoding=GuidedDecodingParams(json=schema)
)

outputs = llm.generate(
    ["Rate this response: 'The explanation was clear and helpful.'"],
    params
)
print(outputs[0].outputs[0].text)
# {"name": "response", "score": 8}  — always valid JSON
```

Via the OpenAI-compatible server:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Extract: John, 28, Python developer"}],
    "guided_json": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"},
        "role": {"type": "string"}
      }
    }
  }'
```

### Choosing between JSON, regex, and choice

| Use case | Recommended approach |
|---|---|
| Extract structured objects with known schema | `guided_json` with Pydantic/JSON Schema |
| Extract a pattern (phone, email, date) | `guided_regex` |
| Classify into fixed categories | `guided_choice` |
| Generate code in a specific language | EBNF grammar |
| Free-form with soft constraints | Prompt engineering (no constraint) |

---

## 12.5.8 SGLang Structured Generation

SGLang has native structured generation as a first-class feature through its
`constrained_decoding` backend:

```python
import sglang as sgl

@sgl.function
def extract_profile(s, text):
    s += sgl.user(f"Extract information from: {text}")
    s += sgl.assistant(
        sgl.gen(
            "profile",
            max_new_tokens=200,
            regex=r'\{"name": "[A-Za-z ]+", "age": [0-9]+\}'
        )
    )

# Or with JSON schema
@sgl.function
def classify(s, text):
    s += sgl.user(f"Classify the sentiment of: {text}")
    s += sgl.assistant(
        sgl.gen(
            "result",
            choices=["positive", "neutral", "negative"]
        )
    )
```

SGLang also supports **speculative constrained decoding**: the constraint
processor runs ahead on the draft model during speculative decoding (Chapter 23),
pre-computing valid tokens for the likely accepted sequence. This eliminates
the constraint evaluation overhead from the critical path.

---

## 12.5.9 Function Calling as Constrained Generation

OpenAI-style function calling is constrained generation in disguise. When the
model is given a set of tool definitions, the output is constrained to either:

1. A valid function call JSON: `{"name": "<one of the tools>", "arguments": {...}}`
2. A plain text response

The constraint FSM has two branches: tool-call path (JSON with the tool name
restricted to the provided list) and free-text path. The model chooses which
path via the first token it emits (typically `{` for a tool call, text for a
response).

```python
# vLLM function calling via OpenAI-compatible endpoint
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)
# The tool call JSON is guaranteed to have a valid function name and schema
```

---

## 12.5.10 Performance Cost of Constraint Evaluation

Constrained decoding is not free. At each decode step, the constraint processor
must compute the token mask before sampling. The cost depends on the approach:

| Approach | Pre-computation | Per-step cost | Best for |
|---|---|---|---|
| outlines (pre-computed table) | 0.5–5s per schema | ~0.1ms (table lookup) | Fixed schemas, high throughput |
| lm-format-enforcer (lazy trie) | Negligible | 1–10ms | Dynamic schemas |
| llama.cpp EBNF grammar | Negligible | 0.5–5ms | Complex grammars |
| SGLang native | 0.1–0.5s per schema | ~0.2ms | SGLang deployments |

For a typical 200-token JSON output, the constraint overhead at 1ms/step
adds ~200ms to generation time. At 0.1ms/step (pre-computed), it adds ~20ms.
For comparison, the model forward pass at H100 throughput takes ~2–5ms per
step for a 70B model. Constraint evaluation is typically < 10% overhead for
pre-computed approaches.

**Worked Example 12.5.2 — Latency budget**

Service: extract structured data from 200-word documents, batch size 1.

```
Without constraints:
  Prefill (250 tokens at 20k tok/s)     = 12.5ms
  Decode (80 output tokens at 200 tok/s) = 400ms
  Total P50                              = ~415ms

With outlines (pre-computed):
  Pre-computation (done at startup)      = 2.5s (one-time)
  Constraint overhead (0.1ms × 80)       = 8ms
  Total P50                              = ~423ms (+2% overhead)

With lm-format-enforcer (lazy):
  Constraint overhead (3ms × 80)         = 240ms
  Total P50                              = ~655ms (+58% overhead)
```

For high-throughput batched use cases, always pre-compute. For dynamic schemas
where you cannot pre-compute, budget for the latency or accept the lazily-
evaluated overhead.

### Schema caching

Pre-computed transition tables should be cached aggressively:

```python
from functools import lru_cache
import outlines

@lru_cache(maxsize=128)
def get_generator(schema_json_str: str):
    """Cache generator per unique schema."""
    import json
    schema = json.loads(schema_json_str)
    return outlines.generate.json(model, schema)

# First call with a schema: 2–5s (pre-computation)
# Subsequent calls: <1ms (cache hit)
gen = get_generator(json.dumps(my_schema, sort_keys=True))
```

---

## 12.5.11 Failure Modes and Debugging

**Empty output**: The constraint is too restrictive for the model to satisfy.
Example: requiring a 10-digit integer when the model wants to say "not
applicable". Fix: add an optional `null` branch to the schema.

**Slow generation**: The FSM has high branching factor at many steps, making
token mask computation expensive. Happens with permissive regex like `.*`.
Fix: tighten the regex or use a schema instead.

**Off-by-one whitespace**: JSON schemas often require exact whitespace (e.g.
`{"key": "value"}` vs `{"key":"value"}`). The FSM may force awkward token
choices when whitespace handling differs between the schema and the model's
natural output style. Fix: add `ws` (whitespace) non-terminals to the grammar.

**Token boundary issues**: A multi-character token (e.g. `true`) may only be
valid at a point where the FSM expects the string `t-r-u-e` character by
character. The pre-computation phase handles this correctly; lazy approaches
may mishandle multi-byte tokens. Check your library version.

**Schema too complex**: Very deep nested schemas with many required fields
produce very long outputs and high constraint evaluation costs. Break complex
extractions into multiple simpler constrained calls.

---

## 12.5.12 Tokenizer Misalignment: A Concrete Failure

The tokenizer alignment problem is easy to state abstractly but surprisingly
hard to debug in practice. This section shows a concrete failure and its fix.

### The scenario

Two models — `model-A` and `model-B` — both use a 128K vocabulary but were
trained with different tokenizers. A user pre-computes a JSON transition table
against `model-A`'s tokenizer, then swaps in `model-B` without regenerating
the table.

**What breaks**: the token ID for the string `"true"` differs between the two
vocabularies. `model-A` encodes `true` as token 4321; `model-B` encodes it as
token 9087. The transition table has an entry for token 4321 in the
`IN_BOOL_VALUE` state; when `model-B` generates token 9087, the lookup returns
`INVALID` and masks it — even though `true` is semantically valid.

```python
# Reproducing the misalignment
def build_transition_table(tokenizer, fsm):
    table = {}
    for state in fsm.states:
        for token_id in range(tokenizer.vocab_size):
            token_str = tokenizer.decode([token_id])
            table[(state, token_id)] = fsm.transition(state, token_str)
    return table

# Always key the cache on (schema_hash, vocab_hash), not on schema alone
import hashlib

def schema_tokenizer_key(schema: dict, tokenizer) -> str:
    schema_hash = hashlib.sha256(str(schema).encode()).hexdigest()[:8]
    vocab_bytes  = str(sorted(tokenizer.get_vocab().items())).encode()
    vocab_hash   = hashlib.sha256(vocab_bytes).hexdigest()[:8]
    return f"{schema_hash}_{vocab_hash}"

_table_cache: dict = {}

def get_transition_table(schema, tokenizer, fsm):
    key = schema_tokenizer_key(schema, tokenizer)
    if key not in _table_cache:
        _table_cache[key] = build_transition_table(tokenizer, fsm)
    return _table_cache[key]
```

If you change the model, the vocabulary hash changes, the cache key changes,
and the table is rebuilt automatically. This is the fix `outlines` applies
internally; if you build your own constraint machinery, replicate it.

---

## 12.5.13 Masking Order: Before or After Temperature?

The logit mask must be applied **before** temperature scaling and softmax.
This is the standard used by all major libraries but is occasionally
implemented incorrectly in custom code:

```python
# WRONG — causes NaN at very low temperatures
probs = torch.softmax(logits / temperature, dim=-1)   # temperature first
probs[invalid_ids] = 0.0
probs /= probs.sum()   # renormalise — may produce NaN if sum ≈ 0

# CORRECT — set invalid to -inf before any temperature operation
logits[invalid_ids] = float('-inf')      # exp(-inf) = 0 exactly — stable
probs = torch.softmax(logits / temperature, dim=-1)   # then temperature
```

At temperature 0.0 (greedy decoding) with a constraint, use `argmax` over the
masked logits directly rather than `softmax` to avoid 0/0:

```python
def constrained_greedy(logits, valid_ids):
    import torch
    mask = torch.full(logits.shape, float('-inf'))
    mask[valid_ids] = logits[valid_ids]
    return mask.argmax().item()
```

---

## 12.5.14 Multi-Turn Constrained Decoding

Constrained decoding usually applies to the entire model output, but in
multi-turn systems you only want to constrain **assistant turns**, not user
turns or system prompts.

```python
from outlines import models, generate

def multi_turn_constrained(
    model, tokenizer,
    turns: list[dict],   # [{"role": "user"/"assistant", "content": "..."}]
    schema: dict,
):
    """Generate the next assistant turn, constrained to schema."""
    generator = generate.json(model, schema)
    prompt = tokenizer.apply_chat_template(
        turns, tokenize=False, add_generation_prompt=True
    )
    return generator(prompt, max_tokens=512)
```

**Critical**: you must **reset the FSM state** at the start of each new
assistant turn. The FSM is stateful within a turn but stateless between turns:

```
Turn 1: FSM: START → ... → ACCEPT
Turn 2: FSM must reset to START (not continue from ACCEPT!)
```

If the FSM is not reset, it begins turn 2 in `ACCEPT` state and immediately
rejects all tokens — producing an empty or truncated output.

---

## 12.5.15 Regex Complexity and Safe Patterns

| Pattern type | Approx. DFA states | Safe? | Notes |
|---|---|---|---|
| Fixed string `"shipped"` | ~8 | ✓ | One linear path |
| Alternation `"a\|b\|c"` | ~10 | ✓ | Linear in choices |
| Bounded repeat `[A-Z]{2,4}` | ~6 | ✓ | Finite |
| Email `[\w.]+@[\w.]+\.[a-z]{2,4}` | ~50 | ✓ | Common field |
| UUID `[0-9a-f]{8}-[0-9a-f]{4}-...` | ~40 | ✓ | Fixed-length |
| Unbounded `.+` | ~3 | ⚠ | Very low constraint value |
| Catastrophic `(a+)+b` | Exponential | ✗ | NFA→DFA blowup — avoid |
| Lookahead `(?=...)` | Unsupported | ✗ | FSMs cannot express lookahead |

Check regex safety with `interegular` before deploying:

```python
import interegular   # pip install interegular

def check_regex_safety(pattern: str, max_states: int = 10_000) -> dict:
    try:
        fsm = interegular.parse_pattern(pattern).to_fsm()
        n = len(fsm.states)
        return {"safe": n < max_states, "states": n,
                "warning": "" if n < max_states else f"{n} states — may be slow"}
    except Exception as e:
        return {"safe": False, "states": -1, "warning": str(e)}

print(check_regex_safety(r"\d{4}-\d{2}-\d{2}"))  # → {'safe': True, ...}
print(check_regex_safety(r"(a+)+b"))               # → {'safe': False, ...}
```

---

## 12.5.16 Quantization and Constraint Interaction

Quantized models (INT4, INT8, FP8) produce logit distributions that differ
slightly from FP16. For constrained decoding, two effects matter:

**Logit magnitude shifts**: quantization adds ~±0.5 noise to logits. The
ranking of valid tokens is usually preserved, but the probability of
marginally-valid tokens can shift by 1–5%. With only 1–2 valid tokens per
step, this is irrelevant; with 10+ valid tokens, the selected token can change.

**Regression test after quantization**: verify that constrained outputs remain
schema-valid after quantization with a sampling run:

```python
import json, jsonschema

def verify_constrained_outputs(outputs: list[str], schema: dict) -> dict:
    passed = sum(1 for o in outputs if _is_valid(o, schema))
    return {"total": len(outputs), "passed": passed,
            "pass_rate": passed / len(outputs)}

def _is_valid(output: str, schema: dict) -> bool:
    try:
        jsonschema.validate(json.loads(output), schema)
        return True
    except Exception:
        return False

# Acceptance criterion: pass_rate >= 0.99 after INT8 quantization
# (100% for FP16; 1% budget for rare logit-rounding failures)
```

### Test harness — structured generation correctness

```python
# ── test_structured_gen.py ───────────────────────────────────────────────
"""Offline tests for FSM masking and transition table correctness.
No GPU required. Run with: python test_structured_gen.py"""

import math

# Minimal character-level FSM for JSON booleans
STATES = {
    "START":  {"t": "T1",     "f": "F1"},
    "T1":     {"r": "T2"},
    "T2":     {"u": "T3"},
    "T3":     {"e": "ACCEPT"},
    "F1":     {"a": "F2"},
    "F2":     {"l": "F3"},
    "F3":     {"s": "F4"},
    "F4":     {"e": "ACCEPT"},
    "ACCEPT": {},
}

def fsm_step(state, char):
    return STATES.get(state, {}).get(char, "INVALID")

def mask_logits(logits_dict, state):
    valid = set(STATES.get(state, {}).keys())
    return {k: (v if k in valid else float('-inf')) for k, v in logits_dict.items()}

def softmax(logits_dict):
    vals = list(logits_dict.values())
    finite = [v for v in vals if v != float('-inf')]
    if not finite:
        raise ValueError("All logits are -inf — no valid token!")
    m = max(finite)
    exps = {k: (math.exp(v - m) if v != float('-inf') else 0.0)
            for k, v in logits_dict.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def test_masking():
    masked = mask_logits({"t": 2.0, "f": 1.5, "n": 3.0}, "START")
    assert masked["t"] == 2.0
    assert masked["f"] == 1.5
    assert masked["n"] == float('-inf')
    print("PASS: invalid tokens masked to -inf")

def test_fsm_true():
    s = "START"
    for ch in "true":
        s = fsm_step(s, ch)
    assert s == "ACCEPT"
    print("PASS: FSM accepts 'true'")

def test_fsm_rejects_wrong_case():
    s = "START"
    s = fsm_step(s, "t")
    s = fsm_step(s, "R")   # uppercase
    assert s == "INVALID"
    print("PASS: FSM rejects 'tRue'")

def test_softmax_stability():
    logits = {"t": 1.0, "f": float('-inf'), "n": float('-inf')}
    probs = softmax(logits)
    assert abs(probs["t"] - 1.0) < 1e-9
    assert not any(math.isnan(v) for v in probs.values())
    print("PASS: softmax(-inf) is numerically stable")

def test_multi_turn_reset():
    # Turn 1 ends in ACCEPT
    s = "START"
    for ch in "true":
        s = fsm_step(s, ch)
    assert s == "ACCEPT"
    # Turn 2 MUST reset
    s = "START"
    for ch in "false":
        s = fsm_step(s, ch)
    assert s == "ACCEPT"
    print("PASS: multi-turn FSM correctly resets between turns")

def test_masking_order():
    """Verify that applying mask before temperature avoids NaN."""
    logits = {"t": 1.0, "f": float('-inf')}
    temperature = 0.01   # very low
    # Mask already applied above; apply temperature
    scaled = {k: (v / temperature if v != float('-inf') else float('-inf'))
              for k, v in logits.items()}
    probs = softmax(scaled)
    assert not any(math.isnan(v) for v in probs.values()), "NaN at low temperature!"
    print("PASS: masking before temperature avoids NaN")


if __name__ == "__main__":
    test_masking()
    test_fsm_true()
    test_fsm_rejects_wrong_case()
    test_softmax_stability()
    test_multi_turn_reset()
    test_masking_order()
    print("\n✓ All structured generation tests passed.")
```

**Expected output:**
```
PASS: invalid tokens masked to -inf
PASS: FSM accepts 'true'
PASS: FSM rejects 'tRue'
PASS: softmax(-inf) is numerically stable
PASS: multi-turn FSM correctly resets between turns
PASS: masking before temperature avoids NaN

✓ All structured generation tests passed.
```

---

## Chapter Summary

Structured generation resolves the reliability gap between what language models
probabilistically produce and what production systems deterministically require.
The mechanism — logit masking via an FSM or grammar parser — is simple in
concept but demands careful attention to tokenizer alignment, pre-computation
strategy, and schema design.

The hierarchy of approaches is: pre-computed FSM (outlines, SGLang native) for
fixed schemas and high throughput; lazy trie (lm-format-enforcer) for dynamic
schemas; EBNF grammars (llama.cpp native) for complex syntactic structures.
Function calling and tool use are constrained generation specialized for the
agentic use case.

The latency overhead is real but manageable: pre-computed approaches add < 5%
to generation time and should be the default for production deployments with
known schemas.

---

## Self-Check Questions

1. A model generates a 100-token JSON response. The token `"null"` has logit
   1.8 at a step where the FSM is in state IN_NUMBER_VALUE (expecting digits
   or `.`). What happens to `"null"`'s probability after masking? Is this
   semantically correct — i.e., should the constraint prevent `null` here?

2. Explain the tokenizer alignment problem. A grammar requires the sequence
   `true` at a given point. The tokenizer has a single token `"true"` (token
   ID 4321) and also individual character tokens `"t"`, `"r"`, `"u"`, `"e"`.
   How should the FSM's transition table handle this? What goes wrong if it
   handles only the character tokens?

3. Compare the pre-computation approach (outlines) with the lazy approach
   (lm-format-enforcer) on the following workload: 10,000 requests/day, each
   with a different user-provided JSON schema, average schema depth 3. Which
   approach is preferable and why?

4. A function-calling system has 50 available tools. The JSON output must have
   `"name"` set to exactly one of those 50 names. Describe the FSM states and
   transitions required to enforce this constraint. How many states does the
   FSM require at minimum?

5. SGLang's speculative constrained decoding runs the constraint processor on
   the draft model's candidate tokens in parallel with the draft model's
   forward pass. What problem does this solve? What assumption about the
   acceptance rate of constrained draft tokens does this optimization rely on?


---

## Worked Solutions

---

### Solution 1 — "null" token in IN_NUMBER_VALUE state

**Situation:** FSM state = IN_NUMBER_VALUE (expecting digits or `.`). Token "null" has logit 1.8.

**Step 1 — Masking.**

The FSM determines that "null" is not a valid continuation in the current state. The constrained decoding system applies:

$$\text{logit}_{\text{null}} \leftarrow -\infty$$

**Step 2 — Post-masking probability.**

$$p_{\text{null}} = \frac{e^{-\infty}}{Z} = \frac{0}{Z} = 0$$

The token "null" has **zero probability** of being selected regardless of its original logit value.

**Step 3 — Is this semantically correct?**

**Yes.** In JSON, an integer field (`"age"`) cannot have a `null` value unless the schema explicitly allows `null` as an alternative type (e.g., `{"type": ["integer", "null"]}`). If the schema says `"age"` is an integer:

- `{"age": null}` → invalid JSON per the schema
- `{"age": 25}` → valid

The constraint correctly prevents the model from generating schema-violating JSON, even though "null" might be a reasonable completion in unconstrained generation.

**Common confusion:** If the schema were `{"age": {"type": ["integer", "null"]}}`, then "null" SHOULD be allowed. The FSM state machine must reflect the full schema — different schemas produce different FSMs, even for the same JSON field name.

---

### Solution 2 — Tokenizer alignment: handling both "true" and "t","r","u","e"

**The problem:**

Grammar requires the string `true`. The vocabulary has:

- Token ID 4321: `"true"` (single token for the entire word)
- Token IDs 500,510,520,530: `"t"`, `"r"`, `"u"`, `"e"` (individual characters)

**What goes wrong with character-only FSM:**

If the FSM is built on character-level transitions:

- State: `START` → sees `"t"` → transitions to `SAW_T`
- State: `SAW_T` → sees `"r"` → transitions to `SAW_TR`
- etc.

When the tokenizer actually produces the single token `"true"` (ID 4321), the FSM doesn't know how to handle it — it's waiting for `"t"` first. The system either rejects the valid token or generates incorrect text by forcing character-by-character generation even when the tokenizer would naturally emit a multi-character token.

**The correct approach (token-level FSM):**

Build the FSM's transition table indexed by **token IDs**, not characters:

```python
# At state IN_BOOLEAN, valid next tokens:
valid_tokens = set()
for token_id, token_text in vocabulary.items():
    # "true" can be produced by the single token OR by starting "t" (then r, u, e)
    if "true".startswith(token_text.strip()):
        valid_tokens.add(token_id)
    # Also check if this token IS "true" entirely
    if token_text.strip() == "true":
        valid_tokens.add(token_id)
```

The FSM pre-computes which token IDs are valid at each state using the full vocabulary. At runtime, the mask directly maps state → set of allowed token IDs.

---

### Solution 3 — Outlines vs lazy enforcement for 10,000 unique schemas

**Comparison:**

| Property | Outlines (pre-computation) | lm-format-enforcer (lazy) |
|----------|---------------------------|---------------------------|
| Setup per schema | Full FSM build + token mask precompute | None |
| Per-step cost | O(1) lookup in precomputed table | O(|valid_tokens|) re-evaluation |
| Memory per schema | ~10–100 MB (token mask for each state) | ~0 |
| Cache miss cost | Cold schema: expensive first build | Every schema is "cold" |

**For 10,000 unique schemas:**

- **Outlines:** First request with schema X → build FSM → ~500 ms latency spike. Cache the FSM for future requests with the same schema. With 10,000 unique schemas each used once: 10,000 × 500 ms = 5,000 seconds of latency penalty spread across requests. Memory: 10,000 schemas × 50 MB = 500 GB → impractical to cache all.

- **lm-format-enforcer (lazy):** Evaluate constraints on-the-fly at each decode step. No upfront cost. Each decode step: ~1–5 ms for constraint evaluation. For a 100-token response: 100 × 3 ms = 300 ms overhead per request — regardless of whether the schema is new or cached.

**Verdict:** For 10,000 unique schemas, **lazy evaluation (lm-format-enforcer) is clearly preferable** — no cold-start penalty, no unbounded memory usage, predictable per-step cost. Outlines is better when a small number of schemas are used repeatedly (cache hit rate > 95%).

---

### Solution 4 — FSM for 50 tool names in a function-calling system

**Task:** The JSON field `"name"` must equal one of 50 specific tool names.

**Step 1 — Trie structure for tool names.**

Build a prefix trie over all 50 names:

```
"" → "get_" → "get_weather"
                → "get_forecast"
     "send_" → "send_email"
                → "send_message"
     "search_" → ...
```

Each node in the trie is a potential FSM state.

**Step 2 — FSM state space.**

States:

- `AFTER_NAME_KEY` (just emitted `"name": "`)
- One state per trie node (prefix of valid names)
- `COMPLETE_NAME_j` for each of the 50 complete names (accepting states)

Minimum states: 1 + (sum of all characters across all tool name prefixes). For 50 names of average length 12: roughly 1 + 50×12/2 (sharing prefixes) ≈ **300 states minimum**, often many fewer due to shared prefixes.

**Step 3 — Transitions.**

At each state, only tokens that extend the current prefix toward a valid name are allowed. For example, at state `SAW_PREFIX_"get_"`, allowed tokens are those whose text continues with "weather", "forecast", or any other valid suffix.

**Step 4 — Token ID pre-computation.**

For each (state, token_id) pair: precompute whether that token is a valid continuation. Store as a dictionary: `{state: frozenset(valid_token_ids)}`. Lookup at decode time: O(1).

---

### Solution 5 — SGLang speculative constrained decoding

**What SGLang does:**

The draft model proposes K tokens speculatively (e.g., K=4). These K tokens are validated in parallel by both:

1. **The verifier model** (correctness check — are these tokens the verifier would have chosen?)
2. **The FSM** (constraint check — are these tokens grammatically valid per the schema?)

**Parallel FSM validation:**

The FSM processes all K draft tokens simultaneously, computing the constraint validity of each position:

```python
draft_tokens = [42, 95, 7, 221]  # proposed by draft model
current_state = fsm.current_state
for i, token in enumerate(draft_tokens):
    if not fsm.is_valid(current_state, token):
        # reject from position i onwards
        accept_mask[i:] = False
        break
    current_state = fsm.transition(current_state, token)
```

**Compound rejection:**

A draft token is rejected if either:

- The verifier would not have produced it (standard speculative decoding rejection)
- The FSM marks it as invalid (constraint violation)

**Throughput benefit:**

Without speculative decoding: 1 token per forward pass. With K=4 speculation: average 2–3 tokens accepted per verifier pass. Constraint violations slightly reduce acceptance rate (the draft model may not always respect the grammar), but the net speedup is still 1.5–2.5× for typical JSON generation tasks.


*Companion code: [`docs/code/chapter_12b.md`](../code/chapter_12b.md)*
