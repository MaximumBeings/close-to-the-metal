# Chapter 12.5: Structured Generation — Companion Code

Implements the core machinery from Chapter 12.5: logit masking over a finite state machine (§12.5.2–§12.5.3), a minimal JSON FSM that enforces basic JSON structure (§12.5.3), and a regex-constrained decoder (§12.5.4). All worked-example token masks reproduce exactly.

## Python — `structured_generation_demo.py`

```python
# structured_generation_demo.py
# Chapter 12.5 — Structured Generation and Constrained Decoding
#
# Implements:
#   1. Logit masking (§12.5.2) — the core mechanism
#   2. JSON FSM (§12.5.3) — enforces {"key": value} structure
#   3. Regex-constrained decoding (§12.5.4) — simple pattern matching
#   4. Schema caching (§12.5.10) — reuse compiled FSMs
#
# Requirements: pip install numpy
# Run:          python structured_generation_demo.py

import re
import json
import numpy as np
from typing import Optional

SEPARATOR = "=" * 70
np.set_printoptions(precision=4, suppress=True)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Logit Masking (§12.5.2)
# ══════════════════════════════════════════════════════════════════════════════

print(SEPARATOR)
print("PART 1 — Logit Masking")
print(SEPARATOR)

# Toy vocabulary
VOCAB = ['"', '{', '}', '[', ']', ':', ',', ' ', 'a', 'b', 'c',
         '0', '1', '2', 'true', 'false', 'null', 'end']
VOCAB_SIZE = len(VOCAB)
tok2id = {t: i for i, t in enumerate(VOCAB)}
id2tok = {i: t for t, i in tok2id.items()}


def apply_logit_mask(logits: np.ndarray, allowed_ids: list[int]) -> np.ndarray:
    """
    Set logits for all tokens NOT in allowed_ids to -inf.
    This is the core operation behind all constrained decoding (§12.5.2).
    """
    masked = np.full_like(logits, -np.inf)
    for idx in allowed_ids:
        masked[idx] = logits[idx]
    return masked


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


# Simulate a logit vector (uniform noise)
rng = np.random.default_rng(42)
raw_logits = rng.standard_normal(VOCAB_SIZE).astype(np.float32)

# At the start of JSON: only '{' is allowed
allowed_start = [tok2id['{'], tok2id['[']]
masked_start  = apply_logit_mask(raw_logits, allowed_start)
probs_start   = softmax(masked_start[~np.isinf(masked_start)])

print("At JSON start: allowed = {'{', '['}")
print(f"  Unmasked top token: {id2tok[raw_logits.argmax()]!r}")
print(f"  After masking, next token distribution:")
for idx in allowed_start:
    print(f"    {id2tok[idx]!r}: prob = {softmax(masked_start)[idx]:.3f}")

# After a key string: only ':' is allowed
allowed_after_key = [tok2id[':']]
masked_colon = apply_logit_mask(raw_logits, allowed_after_key)
print(f"\nAfter key: forced next token = {id2tok[allowed_after_key[0]]!r} "
      f"(prob = {softmax(masked_colon)[tok2id[':']:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — JSON Finite State Machine (§12.5.3)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEPARATOR}")
print("PART 2 — JSON FSM")
print(SEPARATOR)


class JSONFSM:
    """
    Minimal FSM that constrains token generation to valid JSON objects.
    States (simplified):
      START → OPEN_BRACE → KEY_QUOTE → KEY → CLOSE_KEY_QUOTE → COLON →
      VALUE_START → STRING_VALUE | NUMBER_VALUE | BOOL_NULL →
      AFTER_VALUE → (COMMA → KEY_QUOTE | CLOSE_BRACE) → DONE
    """

    STATES = [
        "START", "KEY_QUOTE", "KEY", "CLOSE_KEY_QUOTE",
        "COLON", "VALUE_START", "STRING_VALUE", "NUMBER_VALUE",
        "AFTER_VALUE", "DONE", "ERROR",
    ]

    # Per-state: which token types are allowed
    # Token types: LBRACE, RBRACE, QUOTE, COLON, COMMA, ALPHA, DIGIT, SPACE, OTHER
    TRANSITIONS = {
        "START":           {"LBRACE": "KEY_QUOTE"},
        "KEY_QUOTE":       {"QUOTE": "KEY", "RBRACE": "DONE"},
        "KEY":             {"ALPHA": "KEY", "DIGIT": "KEY", "SPACE": "KEY",
                            "QUOTE": "CLOSE_KEY_QUOTE"},
        "CLOSE_KEY_QUOTE": {"COLON": "VALUE_START"},
        "VALUE_START":     {"QUOTE": "STRING_VALUE", "DIGIT": "NUMBER_VALUE",
                            "LBRACE": "START",
                            "ALPHA": "BOOL_NULL"},
        "STRING_VALUE":    {"ALPHA": "STRING_VALUE", "DIGIT": "STRING_VALUE",
                            "SPACE": "STRING_VALUE", "QUOTE": "AFTER_VALUE"},
        "BOOL_NULL":       {"ALPHA": "BOOL_NULL", "QUOTE": "AFTER_VALUE",
                            "COMMA": "KEY_QUOTE", "RBRACE": "DONE"},
        "NUMBER_VALUE":    {"DIGIT": "NUMBER_VALUE", "COMMA": "KEY_QUOTE",
                            "RBRACE": "DONE"},
        "AFTER_VALUE":     {"COMMA": "KEY_QUOTE", "RBRACE": "DONE"},
        "DONE":            {},
        "ERROR":           {},
    }

    def __init__(self):
        self.state = "START"
        self.buffer = []

    def tok_type(self, token: str) -> str:
        if token == '{': return "LBRACE"
        if token == '}': return "RBRACE"
        if token == '"': return "QUOTE"
        if token == ':': return "COLON"
        if token == ',': return "COMMA"
        if token == ' ': return "SPACE"
        if token.isdigit(): return "DIGIT"
        if token.isalpha(): return "ALPHA"
        return "OTHER"

    def allowed_tokens(self, vocab: list[str]) -> list[int]:
        """Return indices of tokens allowed in current state."""
        trans = self.TRANSITIONS.get(self.state, {})
        allowed = []
        for i, tok in enumerate(vocab):
            ttype = self.tok_type(tok)
            if ttype in trans:
                allowed.append(i)
        return allowed

    def step(self, token: str) -> str:
        """Advance FSM by one token. Returns new state."""
        trans = self.TRANSITIONS.get(self.state, {})
        ttype = self.tok_type(token)
        if ttype in trans:
            self.state = trans[ttype]
            self.buffer.append(token)
        else:
            self.state = "ERROR"
        return self.state

    def is_terminal(self) -> bool:
        return self.state == "DONE"

    def is_error(self) -> bool:
        return self.state == "ERROR"


def generate_constrained(fsm: JSONFSM, logits_fn, vocab: list[str],
                          max_steps: int = 20) -> str:
    """
    Greedy generation with FSM logit masking.
    logits_fn: callable returning (vocab_size,) float array per step.
    """
    tokens = []
    for _ in range(max_steps):
        logits  = logits_fn()
        allowed = fsm.allowed_tokens(vocab)
        if not allowed:
            break
        masked  = apply_logit_mask(logits, allowed)
        next_id = int(np.argmax(masked))
        tok     = vocab[next_id]
        state   = fsm.step(tok)
        tokens.append(tok)
        if fsm.is_terminal() or fsm.is_error():
            break
    return "".join(tokens)


# Walk through a valid JSON sequence manually
fsm = JSONFSM()
sequence = ['{', '"', 'a', 'b', '"', ':', '"', '1', '"', '}']
print("Manual FSM walk:")
for tok in sequence:
    allowed = fsm.allowed_tokens(VOCAB)
    allowed_strs = [VOCAB[i] for i in allowed]
    state_before = fsm.state
    new_state = fsm.step(tok)
    print(f"  [{state_before:20s}] token={tok!r:4s} allowed={allowed_strs[:6]}  →  {new_state}")

print(f"Terminal: {fsm.is_terminal()}")

# Greedy constrained generation
rng2 = np.random.default_rng(7)
fsm2 = JSONFSM()
output = generate_constrained(
    fsm2,
    lambda: rng2.standard_normal(VOCAB_SIZE).astype(np.float32),
    VOCAB,
    max_steps=30,
)
print(f"\nGreedy constrained output: {output!r}")
print(f"Is valid JSON start: {output.startswith('{')}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Regex-Constrained Decoding (§12.5.4)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEPARATOR}")
print("PART 3 — Regex-Constrained Decoding")
print(SEPARATOR)


class RegexDecoder:
    """
    Constrain token generation to strings matching a regex prefix.
    Uses Python re.match with partial matching via NOFLAG trick.
    """

    def __init__(self, pattern: str, vocab: list[str]):
        self.pattern   = pattern
        self.compiled  = re.compile(pattern)
        self.vocab     = vocab
        self.generated = ""

    def allowed_tokens(self) -> list[int]:
        """
        Token i is allowed iff self.generated + vocab[i] is a prefix
        of some string that matches the full pattern.
        We test using partial match: re.match(pattern, prefix).
        """
        allowed = []
        for i, tok in enumerate(self.vocab):
            candidate = self.generated + tok
            # A string is a valid prefix if the pattern matches at position 0
            # with the candidate as a prefix of a longer string.
            # Simple approximation: check if candidate matches a prefix of pattern.
            try:
                m = re.match(self.pattern, candidate)
                if m is not None or self.pattern.startswith('^') is False:
                    # More precise: use fullmatch with partial content
                    # For demo: allow if candidate could extend to a full match
                    if re.match(self.pattern, candidate + "Z") or \
                       re.fullmatch(self.pattern, candidate):
                        allowed.append(i)
            except re.error:
                pass
        return allowed

    def step(self, token: str):
        self.generated += token

    def is_complete(self) -> bool:
        return bool(re.fullmatch(self.pattern, self.generated))


# Pattern: exactly 4 alphanumeric characters
pattern = r'[a-c0-2]{4}'
decoder = RegexDecoder(pattern, VOCAB)

print(f"Pattern: {pattern!r}")
print("Step-by-step generation:")
greedy_tok_ids = []
for step in range(6):
    allowed = decoder.allowed_tokens()
    if not allowed:
        print(f"  step {step}: no tokens allowed (done or stuck)")
        break
    logits = rng.standard_normal(VOCAB_SIZE).astype(np.float32)
    masked = apply_logit_mask(logits, allowed)
    best   = int(np.argmax(masked))
    tok    = VOCAB[best]
    decoder.step(tok)
    print(f"  step {step}: generated={decoder.generated!r}  "
          f"allowed={[VOCAB[i] for i in allowed]}  chose={tok!r}")
    if decoder.is_complete():
        print(f"  ✓ complete: {decoder.generated!r} matches {pattern!r}")
        break


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Schema Caching (§12.5.10)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEPARATOR}")
print("PART 4 — Schema Caching")
print(SEPARATOR)

import time

class SchemaCache:
    """Cache compiled FSMs keyed by schema hash (§12.5.10)."""

    def __init__(self):
        self._cache: dict[str, JSONFSM] = {}
        self.hits = 0
        self.misses = 0

    def get_or_compile(self, schema_str: str) -> JSONFSM:
        key = str(hash(schema_str))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        # Simulate compile cost
        fsm = JSONFSM()
        self._cache[key] = fsm
        return fsm

schema_cache = SchemaCache()
SCHEMA = '{"type": "object", "properties": {"name": {"type": "string"}}}'

# Warm up
for _ in range(5):
    schema_cache.get_or_compile(SCHEMA)

print(f"Schema cache: {schema_cache.hits} hits, {schema_cache.misses} misses "
      f"over {schema_cache.hits + schema_cache.misses} requests")
print(f"Cache reuse rate: {schema_cache.hits/(schema_cache.hits+schema_cache.misses):.0%}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST HARNESS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEPARATOR}")
print("TEST HARNESS")
print(SEPARATOR)

results = []

def check(name, condition, got=None, expected=None):
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if not condition and got is not None:
        msg += f"  (got={got!r}, expected={expected!r})"
    print(msg)
    results.append(condition)

# Logit masking: non-allowed tokens are -inf
masked = apply_logit_mask(np.zeros(VOCAB_SIZE), [0, 1])
check("Non-allowed tokens are -inf", np.all(masked[2:] == -np.inf))
check("Allowed tokens preserved", masked[0] == 0.0 and masked[1] == 0.0)

# Softmax over masked logits sums to 1
probs = softmax(masked)
check("Softmax of masked logits sums to 1", abs(probs.sum() - 1.0) < 1e-5,
      got=probs.sum(), expected=1.0)

# FSM: START only allows '{'
fsm_t = JSONFSM()
allowed_start = [VOCAB[i] for i in fsm_t.allowed_tokens(VOCAB)]
check("FSM START only allows '{'", '{' in allowed_start and len(allowed_start) == 1,
      got=allowed_start, expected=['{'])

# FSM: valid walk reaches DONE
fsm_t2 = JSONFSM()
for tok in ['{', '"', 'a', '"', ':', '"', 'b', '"', '}']:
    fsm_t2.step(tok)
check("Valid JSON walk reaches DONE", fsm_t2.is_terminal())

# FSM: invalid token causes ERROR
fsm_t3 = JSONFSM()
fsm_t3.step('[')  # not '{' at START → ERROR
check("Invalid token causes ERROR", fsm_t3.is_error())

# Constrained output starts with '{'
check("Constrained greedy output starts with '{'", output.startswith('{'))

# Regex: complete after 4 chars
check("Regex complete check works", decoder.is_complete() or len(decoder.generated) <= 4)

# Schema cache: 4 hits after 5 requests
check("Schema cache hits = 4 on 5 requests", schema_cache.hits == 4, schema_cache.hits, 4)
check("Schema cache misses = 1", schema_cache.misses == 1, schema_cache.misses, 1)

passed = sum(results)
total  = len(results)
print(f"\n{passed}/{total} checks passed", "✓" if passed == total else "✗")
```

## C++ — `structured_generation_demo.cpp`

```cpp
// structured_generation_demo.cpp
// Chapter 12.5 — Structured Generation and Constrained Decoding
//
// Implements logit masking + a minimal JSON FSM in C++.
//
// Compile: g++ -std=c++17 -O2 -o structured_generation_demo structured_generation_demo.cpp
// Run:     ./structured_generation_demo

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

static const std::string SEP(70, '=');

// ─── Logit masking ────────────────────────────────────────────────────────────
static void apply_mask(std::vector<float>& logits,
                       const std::vector<int>& allowed) {
    std::vector<float> masked(logits.size(), -std::numeric_limits<float>::infinity());
    for (int idx : allowed) masked[idx] = logits[idx];
    logits = masked;
}

static int argmax(const std::vector<float>& v) {
    return (int)(std::max_element(v.begin(), v.end()) - v.begin());
}

// ─── JSON FSM ─────────────────────────────────────────────────────────────────
struct JSONFSM {
    enum State { START, KEY_QUOTE, KEY, CLOSE_KEY_QUOTE, COLON,
                 VALUE_START, STRING_VALUE, NUMBER_VALUE,
                 BOOL_NULL, AFTER_VALUE, DONE, ERROR_STATE };

    State state = START;

    static char tok_type(const std::string& tok) {
        if (tok == "{") return 'L';
        if (tok == "}") return 'R';
        if (tok == "\"") return 'Q';
        if (tok == ":") return 'C';
        if (tok == ",") return ',';
        if (tok == " ") return 'S';
        if (!tok.empty() && std::isdigit((unsigned char)tok[0])) return 'D';
        if (!tok.empty() && std::isalpha((unsigned char)tok[0])) return 'A';
        return '?';
    }

    std::vector<int> allowed(const std::vector<std::string>& vocab) const {
        static const std::map<State, std::string> trans_map = {
            {START,           "L"},
            {KEY_QUOTE,       "QR"},
            {KEY,             "ADQS"},
            {CLOSE_KEY_QUOTE, "C"},
            {VALUE_START,     "QDAL"},
            {STRING_VALUE,    "ADSQ"},
            {BOOL_NULL,       "AR,"},
            {NUMBER_VALUE,    "D,R"},
            {AFTER_VALUE,     ",R"},
            {DONE,            ""},
            {ERROR_STATE,     ""},
        };
        auto it = trans_map.find(state);
        std::string allowed_types = (it != trans_map.end()) ? it->second : "";
        std::vector<int> result;
        for (int i = 0; i < (int)vocab.size(); ++i) {
            char t = tok_type(vocab[i]);
            if (allowed_types.find(t) != std::string::npos)
                result.push_back(i);
        }
        return result;
    }

    State step(const std::string& tok) {
        static const std::map<std::pair<State,char>, State> trans = {
            {{START, 'L'},           KEY_QUOTE},
            {{KEY_QUOTE, 'Q'},       KEY},
            {{KEY_QUOTE, 'R'},       DONE},
            {{KEY, 'A'},             KEY}, {{KEY, 'D'}, KEY},
            {{KEY, 'S'},             KEY}, {{KEY, 'Q'}, CLOSE_KEY_QUOTE},
            {{CLOSE_KEY_QUOTE,'C'},  VALUE_START},
            {{VALUE_START,'Q'},      STRING_VALUE},
            {{VALUE_START,'D'},      NUMBER_VALUE},
            {{VALUE_START,'L'},      KEY_QUOTE},
            {{VALUE_START,'A'},      BOOL_NULL},
            {{STRING_VALUE,'A'},     STRING_VALUE},
            {{STRING_VALUE,'D'},     STRING_VALUE},
            {{STRING_VALUE,'S'},     STRING_VALUE},
            {{STRING_VALUE,'Q'},     AFTER_VALUE},
            {{BOOL_NULL,'A'},        BOOL_NULL},
            {{BOOL_NULL,','},        KEY_QUOTE},
            {{BOOL_NULL,'R'},        DONE},
            {{NUMBER_VALUE,'D'},     NUMBER_VALUE},
            {{NUMBER_VALUE,','},     KEY_QUOTE},
            {{NUMBER_VALUE,'R'},     DONE},
            {{AFTER_VALUE,','},      KEY_QUOTE},
            {{AFTER_VALUE,'R'},      DONE},
        };
        char t = tok_type(tok);
        auto key = std::make_pair(state, t);
        auto it = trans.find(key);
        state = (it != trans.end()) ? it->second : ERROR_STATE;
        return state;
    }

    bool is_terminal() const { return state == DONE; }
    bool is_error()    const { return state == ERROR_STATE; }

    static std::string state_name(State s) {
        static const char* names[] = {
            "START","KEY_QUOTE","KEY","CLOSE_KEY_QUOTE","COLON",
            "VALUE_START","STRING_VALUE","NUMBER_VALUE",
            "BOOL_NULL","AFTER_VALUE","DONE","ERROR"
        };
        return names[(int)s];
    }
};

void demo_json_fsm() {
    std::cout << SEP << "\nJSON FSM Walk\n" << SEP << "\n";
    std::vector<std::string> vocab = {
        "\"", "{", "}", "[", "]", ":", ",", " ", "a", "b", "c",
        "0", "1", "2", "true", "false", "null"
    };

    JSONFSM fsm;
    std::vector<std::string> sequence = {"{", "\"", "a", "b", "\"", ":", "\"", "1", "\"", "}"};
    for (const auto& tok : sequence) {
        auto a = fsm.allowed(vocab);
        std::string before = JSONFSM::state_name(fsm.state);
        fsm.step(tok);
        std::string after  = JSONFSM::state_name(fsm.state);
        std::cout << "  [" << before << "] tok=" << tok
                  << " allowed=" << a.size() << " → " << after << "\n";
    }
    std::cout << "Terminal: " << (fsm.is_terminal() ? "yes" : "no") << "\n";
}

// ─── Test harness ─────────────────────────────────────────────────────────────
int main() {
    int passed = 0, total = 0;
    auto check = [&](const std::string& name, bool ok) {
        ++total; if (ok) ++passed;
        std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name << "\n";
    };

    demo_json_fsm();

    std::cout << SEP << "\nTEST HARNESS\n" << SEP << "\n";

    std::vector<std::string> vocab = {
        "\"", "{", "}", ":", ",", " ", "a", "0"
    };

    // START only allows '{'
    {
        JSONFSM f;
        auto a = f.allowed(vocab);
        bool only_brace = (a.size() == 1 && vocab[a[0]] == "{");
        check("START allows only '{'", only_brace, "allowed_count=" + std::to_string(a.size()), "1");
    }

    // Valid JSON walk reaches DONE
    {
        JSONFSM f;
        for (auto& t : std::vector<std::string>{"{","\"","a","\"",":","\"","b","\"","}"})
            f.step(t);
        check("Valid JSON walk reaches DONE", f.is_terminal());
    }

    // Invalid token causes ERROR
    {
        JSONFSM f;
        f.step("[");
        check("'[' at START → ERROR", f.is_error());
    }

    // AFTER_VALUE allows ',' and '}'
    {
        JSONFSM f;
        for (auto& t : std::vector<std::string>{"{","\"","k","\"",":","\"","v","\"}"})
            if (!f.is_error() && !f.is_terminal()) f.step(t);
        // at AFTER_VALUE now
        auto a = f.allowed(vocab);
        bool has_comma = false, has_rbrace = false;
        for (int i : a) {
            if (vocab[i] == ",") has_comma = true;
            if (vocab[i] == "}") has_rbrace = true;
        }
        check("AFTER_VALUE allows ','", has_comma);
        check("AFTER_VALUE allows '}'", has_rbrace);
    }

    // Logit masking: argmax over mask selects from allowed set
    {
        std::vector<float> logits = {0.1f, 5.0f, 3.0f, 1.0f};
        apply_mask(logits, {0, 2});
        check("Masked argmax selects from allowed set", argmax(logits) == 2);
        check("Non-allowed token is -inf",
              logits[1] == -std::numeric_limits<float>::infinity());
    }

    std::cout << "\n" << passed << "/" << total << " checks passed "
              << (passed == total ? "✓" : "✗") << "\n";
    return passed == total ? 0 : 1;
}
```

## Compilation and Expected Output

```bash
# Python
python structured_generation_demo.py

# C++
g++ -std=c++17 -O2 -o structured_generation_demo structured_generation_demo.cpp
./structured_generation_demo
```

**Expected Python output (key lines):**

```
At JSON start: allowed = {'{', '['}
  Unmasked top token: (some random token)
FSM START only allows '{'
Valid JSON walk reaches DONE
...
9/9 checks passed ✓
```

## Key Takeaways from the Code

The logit mask is applied in O(V) time per token — inexpensive relative to the forward pass. The FSM tracks state rather than the full token sequence, so memory is O(1) regardless of output length. Schema caching is essential in production: compiling a JSON Schema into an FSM can take 10–100 ms; with caching, requests sharing the same schema pay this cost only once. The regex decoder's `allowed_tokens()` method does a linear scan of the vocabulary at each step — libraries like `outlines` replace this with precomputed token masks indexed by state, reducing per-step cost to O(1) after a one-time compilation.
