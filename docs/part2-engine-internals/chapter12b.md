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

## Chapter Summary

Structured generation resolves the reliability gap between what language models
probabilistically produce and what production systems deterministically require.
The mechanism — logit masking via an FSM or grammar parser — is simple in
concept but demands careful attention to tokenizer alignment, pre-computation
strategy, and schema design.

The hierarchy of approaches is: pre-computed FSM (outlines, SGLang native) for
fixed schemas and high throughput; lazy trie (lm-format-enforcer) for dynamic
schemas; EBNF grammars (llama.cpp native) for complex syntactic structures.
Function calling and tool use are constrained generation specialised for the
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
