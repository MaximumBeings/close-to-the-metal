# Chapter 3 — Tokens, Sequences, and the Batch

> *"The model never sees words. It sees integers. Everything else is plumbing."*

---

## Why This Chapter Exists

You type `"Hello, world!"` into a chat box. Before a single floating-point multiply happens inside the transformer, that string must become a list of integers. After the last integer is generated, the reverse journey happens: integers become bytes become characters become the sentence you read on screen.

The mechanism that handles both directions is the **tokenizer**. It sits between human language and machine arithmetic, and its design choices ripple through every performance number in this book:

- How long is a sequence? (Affects KV cache size, latency, cost.)
- How many unique tokens exist? (Affects the embedding table size and the final logit projection.)
- How do you batch requests of different lengths? (Determines whether your GPU is doing real work or multiplying zeros.)

This chapter is therefore two things at once: a deep-dive into tokenizer algorithms (especially BPE), and a first look at what it means to **batch** token sequences efficiently.

**What you will know by the end:**

- The BPE training algorithm, step-by-step, with a full hand-traced worked example.
- How the encoder uses the learned merge rules to tokenize unseen text.
- The difference between BPE, Byte-Level BPE, WordPiece, SentencePiece, and tiktoken — and which model uses which.
- Special tokens, chat templates, and why ignoring them causes silent failures.
- How static batching wastes GPU compute (with a concrete arithmetic example).
- How continuous batching (Orca) eliminates that waste.
- The `llama_batch` structure in C and the equivalent vLLM batching path in Python.

---

## 3.1 What Is a Token? `[FOUNDATIONAL]`

### 3.1.1 Intuition

A transformer does not operate on characters, words, or sentences. It operates on **tokens** — discrete symbols drawn from a fixed vocabulary. Each token is represented by an integer ID, and that integer indexes into a learned embedding table to retrieve a dense vector.

You can think of a token as the model's "atom of meaning." It can be a whole word (`"hello"`), a subword fragment (`"##ing"` in BERT-style notation), a single character, or a single byte. The tokenizer decides which granularity to use and where to draw the boundaries.

### 3.1.2 The Vocabulary

Every tokenizer ships with a **vocabulary** — a mapping from string ↔ integer. For example:

```
"hello"   → 15339
"world"   → 1917
"!"       → 0
" the"    → 262
```

Vocabulary sizes vary:

| Model Family         | Tokenizer         | Vocabulary Size |
|----------------------|-------------------|-----------------|
| GPT-2                | Byte-Level BPE    | 50,257          |
| LLaMA 1 / 2          | SentencePiece BPE | 32,000          |
| LLaMA 3 / GPT-4      | tiktoken (BPE)    | 128,256         |
| BERT                 | WordPiece         | 30,522          |
| T5                   | SentencePiece     | 32,100          |
| DeepSeek-V2 / V3     | tiktoken (BPE)    | 102,400         |
| Qwen2                | tiktoken (BPE)    | 151,936         |

A larger vocabulary means:

- Longer strings are covered by single tokens → **shorter sequences** → faster inference, cheaper KV cache.
- The embedding table and final LM-head projection matrix are larger → more VRAM overhead.

For LLaMA 3 at `vocab_size = 128,256` with `hidden_dim = 4096` and FP16 weights:
```
Embedding table:  128,256 × 4,096 × 2 bytes = ~1.0 GB
LM head matrix:   4,096 × 128,256 × 2 bytes = ~1.0 GB
                                               --------
                                              ~2.0 GB total
```
These two matrices alone cost 2 GB of VRAM before a single transformer layer is loaded.

### 3.1.3 Tokenization Is Not Unique

Given the sentence `"lower"`, many possible token splits are valid:

```
Option A:  ["lower"]           (whole word)
Option B:  ["low", "er"]       (two subwords)
Option C:  ["l", "o", "w", "e", "r"]  (characters)
```

The tokenizer's **training procedure** and its **merge table** (described in §3.2) determine exactly which split is chosen — and it is deterministic: the same string always produces the same token IDs.

---

## 3.2 Byte Pair Encoding (BPE) — The Full Algorithm `[DEEP DIVE]`

BPE is the dominant tokenizer algorithm for large language models (GPT-2, GPT-3, GPT-4, LLaMA 3, DeepSeek, Qwen). It was originally a lossless **data compression** algorithm, repurposed for subword tokenization by Sennrich et al. (2016).

### 3.2.1 The Core Idea

BPE starts with a character-level vocabulary and **iteratively merges the most frequent adjacent pair** of symbols. Each merge adds one new token to the vocabulary. After `K` merges, you have a vocabulary of size `|characters| + K`.

This is elegant because:

1. Frequent words become single tokens (efficient).
2. Rare words are split into familiar subword fragments (no `<UNK>`).
3. The vocabulary size is a tunable hyperparameter `K`.

### 3.2.2 Phase 1: BPE Training

**Input**: A large text corpus.
**Goal**: Learn a sequence of merge rules.

#### Step 1 — Pre-tokenize into words with counts

Before BPE even starts, the corpus is split into **words** by a simple whitespace (or regex) splitter. The end-of-word marker `</w>` is appended to each word to preserve word boundaries.

Each word is initialized as a sequence of **characters**, with its frequency count:

```
"low low low low low lower lower newest newest newest widest widest"
```

After counting:

```
Word         Count
-----------  -----
l o w </w>     5
l o w e r </w> 2
n e w e s t </w> 3
w i d e s t </w> 2
```

The corpus is now represented as a dict of `(tuple_of_chars → count)`.

#### Step 2 — Count all adjacent pairs

Scan every word. For each adjacent pair of symbols, accumulate the count weighted by the word's frequency.

Pair frequency table after initialization:

```
Pair          Count
-----------   -----
(l, o)          7   ← appears in "low</w>"×5 + "lower</w>"×2
(o, w)          7
(w, </w>)       5
(w, e)          2   ← only in "lower</w>"
(e, r)          2
(r, </w>)       2
(n, e)          3
(e, w)          3
(e, s)          5   ← "newest</w>"×3 + "widest</w>"×2 ?
```

The complete pair counts, listed by word:

From `l o w </w>` × 5:
  - `(l, o)`: 5
  - `(o, w)`: 5
  - `(w, </w>)`: 5

From `l o w e r </w>` × 2:
  - `(l, o)`: 2
  - `(o, w)`: 2
  - `(w, e)`: 2
  - `(e, r)`: 2
  - `(r, </w>)`: 2

From `n e w e s t </w>` × 3:
  - `(n, e)`: 3
  - `(e, w)`: 3
  - `(w, e)`: 3
  - `(e, s)`: 3
  - `(s, t)`: 3
  - `(t, </w>)`: 3

From `w i d e s t </w>` × 2:
  - `(w, i)`: 2
  - `(i, d)`: 2
  - `(d, e)`: 2
  - `(e, s)`: 2
  - `(s, t)`: 2
  - `(t, </w>)`: 2

Totals:

```
Pair          Total Count
-----------   -----------
(l, o)            7
(o, w)            7
(w, </w>)         5
(w, e)            5   ← 2 from "lower" + 3 from "newest"
(e, s)            5   ← 3 from "newest" + 2 from "widest"
(s, t)            5   ← 3 from "newest" + 2 from "widest"
(t, </w>)         5   ← 3 from "newest" + 2 from "widest"
(n, e)            3
(e, w)            3
(e, r)            2
(r, </w>)         2
(w, i)            2
(i, d)            2
(d, e)            2
```

#### Step 3 — Pick the most frequent pair and merge

**Merge 1**: Tied at count 7 between `(l, o)` and `(o, w)`. We pick `(l, o)` (alphabetical tie-break).

New token: `lo` (ID = |initial_chars| + 1)

Update all words by replacing `l o` with `lo`:

```
Word                 Count
-------------------  -----
lo w </w>              5
lo w e r </w>          2
n e w e s t </w>       3
w i d e s t </w>       2
```

Merge rule #1: `l o → lo`

#### Step 4 — Recount pairs and merge again

New pair counts:

From `lo w </w>` × 5: `(lo, w)`: 5, `(w, </w>)`: 5
From `lo w e r </w>` × 2: `(lo, w)`: 2, `(w, e)`: 2, `(e, r)`: 2, `(r, </w>)`: 2
... (others unchanged)

Updated top pairs:

```
Pair          Count
-----------   -----
(lo, w)           7   ← NEW most frequent
(o, w)            0   ← now gone
(w, </w>)         5
(w, e)            5
(e, s)            5
(s, t)            5
(t, </w>)         5
```

**Merge 2**: `(lo, w)` → `low`

```
Word                 Count
-------------------  -----
low </w>               5
low e r </w>           2
n e w e s t </w>       3
w i d e s t </w>       2
```

Merge rule #2: `lo w → low`

#### Step 5 — Continue

New pair counts:

From `low </w>` × 5: `(low, </w>)`: 5
From `low e r </w>` × 2: `(low, e)`: 2, `(e, r)`: 2, `(r, </w>)`: 2
(others unchanged from before)

Top pairs:

```
Pair          Count
-----------   -----
(low, </w>)       5
(w, e)            5  ← from "n e w e s t" + "low e r"... wait:
                     "w, e" in "newest": 3; "low, e": these are different — "low" is now merged
```

Let me recount carefully after Merge 2:

From `low </w>` × 5: `(low, </w>)`: 5
From `low e r </w>` × 2: `(low, e)`: 2, `(e, r)`: 2, `(r, </w>)`: 2
From `n e w e s t </w>` × 3: `(n,e)`:3, `(e,w)`:3, `(w,e)`:3, `(e,s)`:3, `(s,t)`:3, `(t,</w>)`:3
From `w i d e s t </w>` × 2: `(w,i)`:2, `(i,d)`:2, `(d,e)`:2, `(e,s)`:2, `(s,t)`:2, `(t,</w>)`:2

```
Pair          Count
-----------   -----
(low, </w>)       5
(e, s)            5   ← 3 + 2
(s, t)            5   ← 3 + 2
(t, </w>)         5   ← 3 + 2
(n, e)            3
(e, w)            3
(w, e)            3
(low, e)          2
(e, r)            2
(r, </w>)         2
(w, i)            2
(i, d)            2
(d, e)            2
```

Four-way tie at 5. Pick `(e, s)` (first alphabetically).

**Merge 3**: `e s → es`

```
Word                  Count
--------------------  -----
low </w>                5
low e r </w>            2
n e w es t </w>         3
w i d es t </w>         2
```

Merge rule #3: `e s → es`

#### Step 6

New top pairs after Merge 3:

`(es, t)` now has count 3 + 2 = 5. Also `(low, </w>)`: 5, `(s, t)` is gone (absorbed).

```
Pair          Count
-----------   -----
(low, </w>)       5
(es, t)           5
(t, </w>)         5
```

**Merge 4**: `(es, t)` → `est`

```
Word                  Count
--------------------  -----
low </w>                5
low e r </w>            2
n e w est </w>          3
w i d est </w>          2
```

Merge rule #4: `es t → est`

#### Step 7

After Merge 4:

`(low, </w>)`: 5, `(t, </w>)` is now `(est, </w>)`: 3+2=5, `(w, est)`: 3, `(d, est)`: 2, ...

Tie at 5 between `(low, </w>)` and `(est, </w>)`.

**Merge 5**: `(est, </w>)` → `est</w>`

```
Word                  Count
--------------------  -----
low </w>                5
low e r </w>            2
n e w est</w>           3
w i d est</w>           2
```

Merge rule #5: `est </w> → est</w>`

#### Step 8

Now `(low, </w>)` is the most frequent pair at 5.

**Merge 6**: `(low, </w>)` → `low</w>`

```
Word                  Count
--------------------  -----
low</w>                 5
low e r </w>            2
n e w est</w>           3
w i d est</w>           2
```

Merge rule #6: `low </w> → low</w>`

#### Summary: Learned Merge Rules

After 6 merge steps, the learned vocabulary additions are:

```
Rule  Merge            New Token
----  ---------------  ---------
  1   l + o         →  lo
  2   lo + w        →  low
  3   e + s         →  es
  4   es + t        →  est
  5   est + </w>    →  est</w>
  6   low + </w>    →  low</w>
```

Initial character vocabulary: `{l, o, w, e, r, n, s, t, i, d, </w>}` — 11 symbols.
After 6 merges: 17 symbols.

In practice, GPT-2 ran ~50,000 merges. LLaMA 3 ran ~128,000 merges (matching its 128,256 vocab size).

---

### 3.2.3 Phase 2: BPE Encoding (Inference-Time Tokenization)

Training learns the merge rules. At inference time, the encoder **applies** those rules to tokenize new strings.

**Algorithm:**

```
Input:  word string + </w> marker
        merge_rules: ordered list [(pair → merged), ...]

1. Initialize: split word into individual characters
2. For each merge rule (in order of training):
     Find all occurrences of the pair in current token list
     Replace all occurrences with the merged token
3. Output: final token list
```

**Why ordered?** Earlier merges were more frequent in training data, so they are tried first. This ensures the greedy left-to-right application produces the corpus-optimal segmentation.

#### Worked Example: Encoding "lowest"

Merge rules (from §3.2.2):
```
Rule 1: l + o → lo
Rule 2: lo + w → low
Rule 3: e + s → es
Rule 4: es + t → est
Rule 5: est + </w> → est</w>
Rule 6: low + </w> → low</w>
```

Word: `"lowest"` → `l o w e s t </w>`

```
Start:        [l, o, w, e, s, t, </w>]

Apply Rule 1 (l + o → lo):
              [lo, w, e, s, t, </w>]

Apply Rule 2 (lo + w → low):
              [low, e, s, t, </w>]

Apply Rule 3 (e + s → es):
              [low, es, t, </w>]

Apply Rule 4 (es + t → est):
              [low, est, </w>]

Apply Rule 5 (est + </w> → est</w>):
              [low, est</w>]

Apply Rule 6 (low + </w> → low</w>): ← no match (low is not followed by </w>)
              [low, est</w>]   ← no change

Final tokens: ["low", "est</w>"]
Token IDs:    [   6,       10 ]   (hypothetical IDs)
```

The word "lowest" was never seen during training. Yet it tokenizes cleanly into `["low", "est</w>"]` — two meaningful subwords, both learned from frequent fragments in the corpus.

#### Worked Example: Encoding "newest"

```
Start:        [n, e, w, e, s, t, </w>]

Rule 1 (l+o): no match
Rule 2 (lo+w): no match
Rule 3 (e+s): scan left to right — first "e" is at index 1, followed by "w", not "s". 
              Second "e" is at index 3, followed by "s". Match!
              [n, e, w, es, t, </w>]

Rule 4 (es+t): [n, e, w, est, </w>]

Rule 5 (est+</w>): [n, e, w, est</w>]

Rule 6: no match.

Final tokens: ["n", "e", "w", "est</w>"]
```

Note that `"n"`, `"e"`, `"w"` remain as single characters because the training corpus never had enough frequency to merge them into `"ne"`, `"new"`, etc. — illustrating how BPE naturally adapts to corpus statistics.

---

### 3.2.4 The Pair-Counting Data Structure `[DEEP DIVE]`

Naïve re-scanning the entire corpus after each merge is O(n²) in the number of merges. Production BPE implementations maintain **incremental data structures**:

```
pair_counts: dict[(str, str) → int]
    Updated lazily: after merge (A, B) → AB, only pairs involving A or B need updating.

word_to_pairs: dict[tuple → set_of_pairs]
    Allows O(words affected by merge) updates instead of O(corpus).
```

With these structures, running 50,000 merges over a 10 GB corpus takes minutes on a single CPU — tokenizer training is never the bottleneck.

---

## 3.3 Byte-Level BPE (GPT-2 and LLaMA 3) `[FOUNDATIONAL]`

### 3.3.1 The Problem with Character-Level BPE

Classic BPE has one weakness: **any character not seen during training becomes `<UNK>`**. For a model trained on English Wikipedia, a Korean character `'가'` or a math symbol `'∑'` causes tokenization to fail or produce garbage.

### 3.3.2 The Byte-Level Solution

GPT-2 (Radford et al., 2019) introduced **Byte-Level BPE**: instead of treating Unicode characters as atoms, treat the 256 possible **bytes** as atoms.

Every string is first encoded as UTF-8 bytes. The BPE merge algorithm then operates on bytes, not characters. Because every string is a sequence of bytes from a fixed alphabet of 256, `<UNK>` is impossible by construction.

```
"hello"    → bytes [104, 101, 108, 108, 111]  → BPE merges → ["hello"]
"café"     → bytes [99, 97, 102, 195, 169]    → BPE merges → ["caf", "Ã©"]
"가"       → bytes [234, 176, 128]             → BPE merges → ["ê", "°", "\x80"] (worst case: 3 tokens)
```

GPT-2 maps each raw byte to a printable character before displaying (e.g., byte 195 → `Ã`) — this is purely cosmetic for human readability in the vocabulary file.

```
ASCII diagram: Byte-Level BPE Pipeline

  Input string (Unicode)
         │
         ▼
  UTF-8 encode → raw bytes [b₀, b₁, ..., bₙ]
         │
         ▼
  Map each byte → printable display char (256 fixed chars)
         │
         ▼
  BPE merge rules (operate on these 256 base symbols)
         │
         ▼
  Token ID sequence [t₀, t₁, ..., tₖ]   where k ≤ n
```

**LLaMA 3** uses tiktoken (see §3.5) which is also byte-level, with vocabulary size 128,256.

**LLaMA 1/2** used SentencePiece (see §3.4) with a different approach to handling bytes.

---

## 3.4 WordPiece (BERT) vs. BPE `[FOUNDATIONAL]`

BERT and its family (DistilBERT, RoBERTa, ALBERT) use **WordPiece**, developed by Schuster & Nakamura (2012) and refined at Google.

### 3.4.1 Training Difference

BPE merges the pair with the **highest raw frequency**:

```
score_BPE(A, B) = count(AB)
```

WordPiece merges the pair that **maximizes the likelihood** of the training corpus:

```
score_WP(A, B) = count(AB) / (count(A) × count(B))
```

This is the pointwise mutual information (PMI) heuristic: prefer merging pairs that co-occur **more than by chance**. Rare but highly collocated pairs (like `"un"` + `"##usual"`) get merged before common but independent pairs.

### 3.4.2 Encoding Difference

BPE encodes greedily **left-to-right** using merge rules. WordPiece encodes by finding the **longest matching prefix** in the vocabulary at each step:

```
"unaffable" → "un" + "##aff" + "##able"
```

The `##` prefix marks continuation subwords (not at word start). BERT uses `##`; modern byte-level tokenizers do not (they encode the space before a word as part of the token instead: `" hello"` vs `"hello"`).

### 3.4.3 Which to Use?

| Property             | BPE                         | WordPiece               |
|----------------------|-----------------------------|-------------------------|
| Merge criterion      | Frequency                   | PMI / likelihood        |
| Encoding algorithm   | Greedy merge-rule replay    | Longest prefix match    |
| Handles OOV          | Via bytes (byte-level BPE)  | No (uses `[UNK]`)       |
| Used by              | GPT-2/3/4, LLaMA, DeepSeek  | BERT, ALBERT, DistilBERT|
| Continuation marker  | Implicit (no `</w>` exposed)| Explicit `##` prefix    |

For inference engines, this distinction matters for the **vocabulary file format** and the **special tokens** (see §3.6). vLLM and llama.cpp support both, but the chat template structures differ.

---

## 3.5 SentencePiece and tiktoken `[FOUNDATIONAL]`

### 3.5.1 SentencePiece (LLaMA 1, LLaMA 2, Mistral)

SentencePiece (Kudo & Richardson, 2018) is a **language-agnostic** subword tokenizer. Its key innovations:

1. **Whitespace as a regular character**: SentencePiece treats the space `▁` (U+2581) as a regular symbol included in tokens. The word `" low"` becomes the token `▁low`. This eliminates the need for an end-of-word marker like `</w>`.

2. **Training directly from raw text**: SentencePiece does not require pre-tokenization into words. It processes the raw character stream, making it naturally suited to languages without whitespace word boundaries (Chinese, Japanese, Thai).

3. **BPE or Unigram**: SentencePiece supports two algorithms — BPE (same as described above but operating on the `▁`-augmented character stream) and **Unigram LM** (which trains a probabilistic model over all possible segmentations and prunes the vocabulary iteratively).

LLaMA 1 and 2 use SentencePiece BPE with vocabulary size 32,000. The `.model` file shipped with LLaMA weights contains the serialized vocabulary and merge rules in Protocol Buffer format.

#### Loading SentencePiece in llama.cpp

llama.cpp reads the SentencePiece data directly from the GGUF file. The `llama_model_get_vocab()` call in Chapter 1's code returns a `llama_vocab*` pointer that internally uses this data for all tokenization.

### 3.5.2 tiktoken (GPT-4, LLaMA 3, DeepSeek, Qwen)

tiktoken (Bai et al., OpenAI, 2023) is OpenAI's high-performance BPE tokenizer. It differs from SentencePiece in several ways:

1. **Byte-level**: Built on top of byte-level BPE (like GPT-2), so no `<UNK>`.

2. **Regex pre-tokenization**: tiktoken first splits input with a regex pattern before applying BPE. The LLaMA 3 / GPT-4 pattern is:

```python
PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|
          \p{N}{1,3}|\s?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
```

This ensures contractions like `"don't"` always split as `["don", "'t"]` regardless of training statistics — encoding invariants are baked into the regex.

3. **No special `</w>` or `▁`**: The space is simply included in the following token. `" hello"` is tokenized as one token containing the leading space.

4. **Performance**: tiktoken uses a Rust-backed regex engine and is ~5–10× faster than Python SentencePiece for large batches.

#### tiktoken vs SentencePiece: Vocabulary Alignment

```
ASCII diagram: Space handling across tokenizers

Input: "low lower"
           │
    ┌──────┴────────────────────────────────────────┐
    │                                                │
SentencePiece BPE                           tiktoken BPE
(LLaMA 1/2)                                 (LLaMA 3)
    │                                                │
▁low  ▁lower                                low  ▁low  er
(2 tokens, spaces as ▁ prefix)              (3 tokens, space before "lower")
```

This is why **LLaMA 2 and LLaMA 3 produce different token IDs for the same text**, and why mixing tokenizers across model families is a common source of bugs.

---

## 3.6 Special Tokens and Chat Templates `[COMMON TRAP]`

### 3.6.1 What Are Special Tokens?

Beyond the BPE vocabulary, every tokenizer reserves a set of **special tokens** — tokens with dedicated semantic roles that the model was trained to recognize:

| Token     | ID (LLaMA 3)  | Role                                    |
|-----------|---------------|-----------------------------------------|
| `<|begin_of_text|>` | 128000 | Start of sequence (BOS)          |
| `<|end_of_text|>`   | 128001 | End of generation (EOS)          |
| `<|start_header_id|>` | 128006 | Begin role header (system/user/assistant) |
| `<|end_header_id|>`   | 128007 | End role header                  |
| `<|eot_id|>`          | 128009 | End of turn                      |
| `<|pad|>`             | None   | Padding (not used in LLaMA 3)    |

For GPT-2: `<|endoftext|>` (ID 50256) serves as both BOS and EOS.

For BERT: `[CLS]` (101), `[SEP]` (102), `[PAD]` (0), `[MASK]` (103), `[UNK]` (100).

### 3.6.2 Chat Templates

Instruction-tuned models are trained with **structured conversation formats**. If you don't wrap your input in the correct template, the model produces garbage — it's looking for its training format and doesn't find it.

#### LLaMA 3 Chat Template:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What is PagedAttention?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```

(The trailing newline after `assistant` is important — the model generates from that position.)

#### LLaMA 2 Chat Template:

```
[INST] <<SYS>>
You are a helpful AI assistant.
<</SYS>>

What is PagedAttention? [/INST]
```

These are completely different. Feeding a LLaMA 2 prompt to a LLaMA 3 model produces near-random outputs.

#### Mistral Chat Template:

```
[INST] What is PagedAttention? [/INST]
```

(Mistral base does not support system prompts in the same way.)

### 3.6.3 How vLLM Handles Chat Templates

vLLM ships with Jinja2-based chat templates stored in HuggingFace model repositories as `tokenizer_config.json` → `"chat_template"` field. When you call:

```python
llm.generate(
    [{"role": "user", "content": "What is PagedAttention?"}],
    SamplingParams(max_tokens=256)
)
```

vLLM calls `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)` internally, producing the correctly formatted token sequence.

### 3.6.4 `[COMMON TRAP]` — Forgetting `add_generation_prompt`

```python
# WRONG: omits the assistant turn opening
tokens = tokenizer.apply_chat_template(messages, tokenize=True)

# RIGHT: appends the assistant header so the model knows to generate
tokens = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True
)
```

Without `add_generation_prompt=True`, the model sees the conversation but no signal to start generating — it often echoes the user message or produces the EOS token immediately.

---

## 3.7 From Tokens to Sequences: Padding and the Static Batching Problem `[FOUNDATIONAL]`

### 3.7.1 Why Batching?

A single-sequence forward pass on a modern GPU is **extremely wasteful**. The matrix multiplications in the attention layers and feed-forward layers are optimized for batched operations. A batch of 32 sequences does roughly the same memory reads as a batch of 1, but 32× the useful work (higher arithmetic intensity — see Chapter 1's roofline analysis).

### 3.7.2 The Padding Problem

To batch sequences of different lengths, naïve implementations **pad** shorter sequences to the length of the longest:

#### Worked Example: Static Batching with Padding

Four requests arrive simultaneously:

```
Request  Prompt (tokens)   Desired output (tokens)   Total length
-------  ----------------  -----------------------   ------------
  A      [1, 2, 3]                 5                     8
  B      [4, 5, 6, 7, 8]          3                    11
  C      [9, 10]                  7                    16   ← longest
  D      [11, 12, 13, 14]         4                    20   ← wait, must recalc
```

Let me redo with concrete numbers:

```
Request  Prompt length  Max new tokens  Total context
-------  -------------  --------------  -------------
  A          3               5               8
  B          5               3               8   ← same total, different split
  C          2               7               9
  D          4               4               8
```

Maximum total context = 9 (Request C). All sequences padded to length 9:

```
Padded batch (length 9 each):
  A: [tok, tok, tok, PAD, PAD, PAD, PAD, PAD, PAD]
  B: [tok, tok, tok, tok, tok, PAD, PAD, PAD, PAD]
  C: [tok, tok, PAD, PAD, PAD, PAD, PAD, PAD, PAD]
  D: [tok, tok, tok, tok, PAD, PAD, PAD, PAD, PAD]
```

Visualised as a compute grid (■ = real compute, □ = wasted):

```
         Position: 1  2  3  4  5  6  7  8  9
Request A:         ■  ■  ■  □  □  □  □  □  □   (3/9 = 33%)
Request B:         ■  ■  ■  ■  ■  □  □  □  □   (5/9 = 56%)
Request C:         ■  ■  □  □  □  □  □  □  □   (2/9 = 22%)
Request D:         ■  ■  ■  ■  □  □  □  □  □   (4/9 = 44%)
                   ─────────────────────────────
Total real:        14 out of 36 positions = 38.9%
Wasted compute:    22 out of 36 positions = 61.1%
```

**61.1% of every forward pass is multiplication by zero.** The GPU executes these operations (it has no way to skip them in a dense tensor) and throws away the results. This is the static batching problem.

The situation is even worse for **decode**: with autoregressive generation, all requests in the batch must finish before any result is returned. The batch runs until the longest sequence generates its EOS. Short sequences are either padded or held idle while long sequences finish.

```
Time →   t1   t2   t3   t4   t5   t6   t7   t8   t9  t10
Req A:   gen  gen  gen  gen  gen  EOS  idle idle idle idle   (finishes at t6)
Req B:   gen  gen  gen  EOS  idle idle idle idle idle idle   (finishes at t4)
Req C:   gen  gen  gen  gen  gen  gen  gen  gen  EOS  idle   (finishes at t9)
Req D:   gen  gen  gen  gen  EOS  idle idle idle idle idle   (finishes at t5)
```

With static batching, the GPU holds 4 slots occupied until t9. Requests A, B, D finish early but cannot be freed — new requests cannot be admitted until the whole batch is done.

---

## 3.8 Continuous Batching: The Orca Insight `[DEEP DIVE]`

### 3.8.1 The Observation

Yu et al. (2022) ("Orca: A Distributed Serving System for Transformer-Based Generative Models") made a simple but profound observation:

> *The LLM scheduler doesn't need to treat a "request" as an atomic unit. It can treat each **decode step** as a schedulable unit.*

At every decode iteration (every step of autoregressive generation), the scheduler can:

1. Check if any sequences in the batch have hit EOS.
2. **Immediately admit new requests** to fill those slots.
3. Run the next decode step with the updated batch composition.

This is called **continuous batching** (also: iteration-level scheduling, in-flight batching).

### 3.8.2 Before and After

```
ASCII diagram: Static batching vs. Continuous batching

STATIC BATCHING (wait-for-all)
──────────────────────────────
Slot 1: [A A A A A · · · ·]   ← A finishes at step 5, slot idle steps 6-9
Slot 2: [B B B · · · · · ·]   ← B finishes at step 3, slot idle steps 4-9
Slot 3: [C C C C C C C C ·]   ← C finishes at step 8
Slot 4: [D D D D D · · · ·]   ← D finishes at step 5, slot idle steps 6-9
         ─────────────────────────────
Time:     1 2 3 4 5 6 7 8 9

         New requests E, F, G waiting but BLOCKED until step 9.

CONTINUOUS BATCHING (iteration-level)
──────────────────────────────────────
Slot 1: [A A A A A E E E E]   ← A finishes step 5, E admitted immediately
Slot 2: [B B B F F F F G G]   ← B finishes step 3, F admitted; F finishes, G admitted
Slot 3: [C C C C C C C C ·]   ← C finishes step 8
Slot 4: [D D D D D H H H H]   ← D finishes step 5, H admitted
         ─────────────────────────────
Time:     1 2 3 4 5 6 7 8 9

         Requests E, F, G, H start as soon as slots open.
         GPU stays busy. Throughput ≈ 3-5× higher than static batching.
```

### 3.8.3 The Scheduling Math

With continuous batching, at each step the scheduler selects which sequences to include in the **running batch**. The constraints are:

1. **KV cache capacity**: each active sequence occupies KV cache space proportional to its current length. More sequences → more KV cache needed.
2. **Batch size**: GPU SRAM and software limits on max batch size.
3. **Priority queues**: new requests wait in a FIFO or priority queue and are admitted when capacity is available.

vLLM implements this in `vllm/core/scheduler.py`. The scheduler runs at every decode step, checks which sequences can be added from the waiting queue, and outputs a `SchedulerOutputs` object that drives the engine.

### 3.8.4 Chunked Prefill

An extension of continuous batching, introduced in vLLM v0.4 and **on by default in V1**: **chunked prefill** splits long prompts across multiple steps so that prefill and decode can interleave. This prevents a long-context prefill from monopolizing the GPU and starving decode requests.

```
Without chunked prefill:
Step 1: [PREFILL 2048 tokens] ← blocks all decodes for one step
Step 2: [DECODE decode decode decode]

With chunked prefill (chunk_size=512):
Step 1: [PREFILL chunk 1/4] + [DECODE decode decode]
Step 2: [PREFILL chunk 2/4] + [DECODE decode decode]
Step 3: [PREFILL chunk 3/4] + [DECODE decode decode]
Step 4: [PREFILL chunk 4/4] + [DECODE decode decode]
```

---

## 3.9 The `llama_batch` Structure `[DEEP DIVE]`

llama.cpp exposes the batch concept through a C struct. Understanding it is essential for writing multi-sequence inference code in C++.

```c
typedef struct llama_batch {
    int32_t          n_tokens;      // total tokens in this batch

    llama_token    * token;         // [n_tokens]  token IDs
    float          * embd;          // [n_tokens × n_embd]  OR NULL (use token instead)
    llama_pos      * pos;           // [n_tokens]  position in KV cache (0-based)
    int32_t        * n_seq_id;      // [n_tokens]  how many sequence IDs this token belongs to
    llama_seq_id  ** seq_id;        // [n_tokens]  array of sequence IDs (for multi-sequence)
    int8_t         * logits;        // [n_tokens]  1 = compute logits for this token, 0 = skip
} llama_batch;
```

### 3.9.1 Single-Sequence Prefill

When you call `llama_batch_get_one(tokens, n)`, it fills:

```c
batch.n_tokens = n
batch.token[i] = tokens[i]    for i in [0, n)
batch.pos[i]   = i             // sequential positions
batch.seq_id[i][0] = 0         // all tokens belong to sequence 0
batch.logits[i]   = 0 except logits[n-1] = 1   // only compute logits for last token
```

This is the single-sequence case from `hello_llamacpp.cpp` in Chapter 1.

### 3.9.2 Multi-Sequence Batch (Continuous Batching)

For `B` concurrent sequences each contributing one decode token:

```c
batch.n_tokens = B
for i in range(B):
    batch.token[i]     = current_token[i]
    batch.pos[i]       = current_position[i]   // each seq at its own position
    batch.seq_id[i][0] = i                      // sequence identity
    batch.logits[i]    = 1                      // need logits for all (will sample each)
```

### 3.9.3 The `pos` Field Is Critical

The `pos` field tells llama.cpp where in the KV cache to **write** the K and V tensors for this token. For sequence `i` currently at position `p_i`:

```
KV cache slot for (layer l, sequence i, position p_i):
  k_cache[l][p_i mod n_ctx]  ← written with K for this token
  v_cache[l][p_i mod n_ctx]  ← written with V for this token
```

When the next decode step runs, attention reads all K and V slots with positions `[0 .. p_i]` for sequence `i`. This is why the position must be tracked and incremented correctly per sequence.

### 3.9.4 `[COMMON TRAP]` — Mixing Up Position and Slot

A common mistake when implementing multi-turn conversation in llama.cpp:

```cpp
// WRONG: position resets to 0 for each turn
llama_batch batch = llama_batch_get_one(turn2_tokens.data(), n_turn2);
// → writes turn 2 KV at positions [0, 1, 2, ...], overwriting turn 1!

// RIGHT: position continues from end of turn 1
int pos_offset = n_turn1;
for (int i = 0; i < n_turn2; i++) {
    batch.pos[i] = pos_offset + i;   // positions continue
}
```

---

## 3.10 vLLM's Tokenization Path `[FOUNDATIONAL]`

In vLLM, tokenization is **decoupled from the inference engine**. The `AsyncLLMEngine` (or `LLMEngine` for synchronous use) calls the tokenizer as a preprocessing step before the request enters the scheduler.

```
ASCII diagram: vLLM request lifecycle (tokenization phase)

  Client request: {"prompt": "Hello world", "max_tokens": 128}
         │
         ▼
  Tokenizer.encode(prompt)
    ├─ apply_chat_template (if messages format)
    ├─ tiktoken / SentencePiece encode
    └─ returns: [tok_id₀, tok_id₁, ..., tok_idₙ]
         │
         ▼
  SequenceGroup created
    ├─ prompt_token_ids = [tok_id₀, ..., tok_idₙ]
    ├─ sampling_params = SamplingParams(...)
    └─ request_id = UUID
         │
         ▼
  Scheduler.add_seq_group(seq_group)
    └─ enters waiting queue
         │
         ▼
  Next scheduler step: allocate KV blocks, move to running
```

The tokenizer runs in the **Python process** (not the GPU worker), so it can run concurrently with GPU execution via asyncio. For high-throughput serving, vLLM uses a thread pool to parallelize tokenization of many incoming requests.

---

## 3.11 Code: Tokenization in Both Engines

### 3.11.1 Python: vLLM Tokenization Deep Dive

See `code/chapter_03/tokenization_demo.py`.

```python
# tokenization_demo.py
# Chapter 3 — Tokens, Sequences, and the Batch
#
# Requirements: pip install vllm transformers
# No GPU needed for tokenization-only operations.

from transformers import AutoTokenizer
import json

# ── Section 1: Basic BPE tokenization ──────────────────────────────────────

# LLaMA 3 uses tiktoken BPE with vocab_size=128,256
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

def show_tokens(text: str):
    """Display token IDs and their decoded pieces side by side."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = [tokenizer.decode([tid]) for tid in token_ids]
    print(f"\nText:   {repr(text)}")
    print(f"IDs:    {token_ids}")
    print(f"Pieces: {pieces}")
    print(f"Count:  {len(token_ids)} tokens")

show_tokens("Hello, world!")
show_tokens("lowest")
show_tokens("newest")
show_tokens("unaffable")
show_tokens("LLM inference")
show_tokens("café")        # non-ASCII: uses byte fallback
show_tokens("가나다")      # Korean: multiple bytes per character

# ── Section 2: Chat template ────────────────────────────────────────────────

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is PagedAttention?"},
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print("\n── Chat template (text) ──")
print(formatted)

formatted_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True
)
print(f"\n── Chat template (IDs): {len(formatted_ids)} tokens ──")
print(formatted_ids[:20], "...")  # first 20 for brevity

# ── Section 3: Padding waste demonstration ──────────────────────────────────

requests = [
    "Explain the difference between prefill and decode.",
    "What is BPE?",
    "Describe the KV cache memory formula for a 70B model with 80 layers, 8 KV heads, "
    "head dimension 128, and sequence length 4096 in FP16.",
    "Hi",
]

all_ids = [tokenizer.encode(r, add_special_tokens=False) for r in requests]
lengths = [len(ids) for ids in all_ids]
max_len = max(lengths)

print("\n── Padding waste ──")
print(f"Sequence lengths: {lengths}")
print(f"Max length:       {max_len}")

total_real = sum(lengths)
total_padded = max_len * len(requests)
waste_pct = 100 * (total_padded - total_real) / total_padded

print(f"Real tokens:      {total_real}")
print(f"Padded tokens:    {total_padded}")
print(f"Wasted compute:   {waste_pct:.1f}%")

# Visualise
print("\n  Batch grid (■ = real, □ = padded):")
for i, (req, ids) in enumerate(zip(requests, all_ids)):
    bar = "■" * len(ids) + "□" * (max_len - len(ids))
    print(f"  Req {i+1}: {bar}  ({len(ids)}/{max_len} tokens)")

# ── Section 4: Vocabulary inspection ───────────────────────────────────────

print(f"\n── Vocabulary stats ──")
print(f"Vocab size:    {tokenizer.vocab_size}")
print(f"BOS token:     {tokenizer.bos_token!r} (ID {tokenizer.bos_token_id})")
print(f"EOS token:     {tokenizer.eos_token!r} (ID {tokenizer.eos_token_id})")
print(f"PAD token:     {tokenizer.pad_token!r} (ID {tokenizer.pad_token_id})")

# Special tokens added by tiktoken for LLaMA 3
special = tokenizer.special_tokens_map
print(f"\nSpecial tokens: {json.dumps(special, indent=2)}")
```

### 3.11.2 C++: llama.cpp Multi-Sequence Batch

See `code/chapter_03/batch_demo.cpp`.

```cpp
// batch_demo.cpp
// Chapter 3 — Tokens, Sequences, and the Batch
//
// Demonstrates:
//   1. Tokenizing multiple prompts
//   2. Building a multi-sequence batch manually
//   3. Simulating continuous batching (two waves)
//
// Build (same as Chapter 1 — from within the llama.cpp repo root):
//   g++ -std=c++17 -O2 batch_demo.cpp -Iinclude -Lbuild/src -lllama \
//       -Lbuild/ggml/src -lggml -o batch_demo
//
// Run:
//   ./batch_demo /path/to/Llama-3.2-1B-Q4_K_M.gguf

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Tokenize a single string into a vector<llama_token>
// ---------------------------------------------------------------------------
static std::vector<llama_token> tokenize(
    const llama_model* model,
    const std::string& text,
    bool add_bos = true)
{
    std::vector<llama_token> tokens(text.size() / 2 + 64);
    int n = llama_tokenize(
        llama_model_get_vocab(model),
        text.c_str(), (int)text.size(),
        tokens.data(), (int)tokens.size(),
        add_bos, /*parse_special=*/true
    );
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(
            llama_model_get_vocab(model),
            text.c_str(), (int)text.size(),
            tokens.data(), (int)tokens.size(),
            add_bos, true
        );
    }
    assert(n > 0);
    tokens.resize(n);
    return tokens;
}

// ---------------------------------------------------------------------------
// Show token IDs and their string pieces
// ---------------------------------------------------------------------------
static void print_tokens(
    const llama_model* model,
    const std::string& label,
    const std::vector<llama_token>& tokens)
{
    printf("%s (%d tokens):\n  IDs: [", label.c_str(), (int)tokens.size());
    for (int i = 0; i < (int)tokens.size(); i++) {
        printf("%d%s", tokens[i], i+1 < (int)tokens.size() ? ", " : "");
    }
    printf("]\n  Pieces: [");
    for (int i = 0; i < (int)tokens.size(); i++) {
        char buf[256];
        int n = llama_token_to_piece(
            llama_model_get_vocab(model), tokens[i], buf, sizeof(buf), 0, true);
        if (n > 0) printf("\"%.*s\"%s", n, buf, i+1 < (int)tokens.size() ? ", " : "");
    }
    printf("]\n\n");
}

// ---------------------------------------------------------------------------
// Build a multi-sequence decode batch
// Each entry: one token per sequence at its current position.
// ---------------------------------------------------------------------------
static llama_batch make_decode_batch(
    const std::vector<llama_token>& current_tokens,   // one token per sequence
    const std::vector<int>&         positions,         // current KV position per sequence
    int n_seqs)
{
    // llama_batch_init allocates the arrays for up to max_n_tokens tokens,
    // with up to max_n_seqs sequence IDs per token.
    llama_batch batch = llama_batch_init(n_seqs, 0, 1);  // max_tokens=n_seqs, embd=0, max_seq_id=1
    batch.n_tokens = n_seqs;

    for (int i = 0; i < n_seqs; i++) {
        batch.token[i]        = current_tokens[i];
        batch.pos[i]          = positions[i];
        batch.n_seq_id[i]     = 1;
        batch.seq_id[i][0]    = i;       // sequence i is assigned slot i
        batch.logits[i]       = 1;       // compute logits for all (need to sample each)
    }
    return batch;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    // Load model (CPU only for demo)
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    // Context: support 4 concurrent sequences each up to 256 tokens
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 4 * 256;   // total KV cache tokens across all sequences
    cparams.n_batch   = 4 * 256;
    cparams.n_ubatch  = 64;        // max tokens per physical forward pass during prefill
    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    // ── Part 1: Tokenize several prompts ────────────────────────────────────

    std::vector<std::string> prompts = {
        "Hello",
        "What is the KV cache?",
        "Explain BPE tokenization briefly.",
        "Hi there"
    };

    printf("=== Part 1: Tokenization ===\n\n");

    std::vector<std::vector<llama_token>> all_tokens;
    for (int i = 0; i < (int)prompts.size(); i++) {
        auto toks = tokenize(model, prompts[i]);
        all_tokens.push_back(toks);
        print_tokens(model, "Prompt " + std::to_string(i), toks);
    }

    // Show padding waste
    int max_len = 0;
    int total_real = 0;
    for (auto& t : all_tokens) { max_len = std::max(max_len, (int)t.size()); total_real += (int)t.size(); }
    int total_padded = max_len * (int)all_tokens.size();
    printf("Static batching waste: %d/%d tokens = %.1f%%\n\n",
        total_padded - total_real, total_padded,
        100.0f * (total_padded - total_real) / total_padded);

    // ── Part 2: Prefill all 4 sequences (continuous batching style) ─────────

    printf("=== Part 2: Prefill (all sequences) ===\n\n");

    // Prefill each sequence separately using sequence-tagged positions.
    // In production vLLM, PagedAttention handles this more efficiently —
    // here we do it sequentially for clarity.

    std::vector<int> seq_positions(prompts.size(), 0);

    for (int seq = 0; seq < (int)prompts.size(); seq++) {
        auto& toks = all_tokens[seq];
        int n = (int)toks.size();

        // Build prefill batch for this sequence
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i]      = toks[i];
            batch.pos[i]        = i;
            batch.n_seq_id[i]   = 1;
            batch.seq_id[i][0]  = seq;
            batch.logits[i]     = (i == n-1) ? 1 : 0;  // only last token needs logits
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Prefill failed for seq %d\n", seq);
            return 1;
        }
        seq_positions[seq] = n;  // next token goes at position n
        printf("Seq %d prefilled: %d tokens, next pos = %d\n", seq, n, n);
        llama_batch_free(batch);
    }

    // ── Part 3: One step of continuous batch decode ──────────────────────────

    printf("\n=== Part 3: One decode step (all 4 sequences in one batch) ===\n\n");

    // Build a sampler chain for each sequence
    // (In a real engine each sequence has its own sampler state)
    std::vector<llama_sampler*> samplers(prompts.size());
    for (int i = 0; i < (int)prompts.size(); i++) {
        samplers[i] = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(samplers[i], llama_sampler_init_greedy());
    }

    // Sample the first output token from each sequence's prefill logits.
    // In the actual decode loop you would submit all these tokens in one batch.
    std::vector<llama_token> next_tokens(prompts.size());
    for (int i = 0; i < (int)prompts.size(); i++) {
        // llama_sampler_sample reads the logits at the specified token index.
        // After prefill, the last token of each sequence has logits.
        // Since we prefilled sequentially, we only have seq 3's logits in buffer.
        // This is a demo — real multi-sequence engines use ragged batches.
        // We use seq=3 here to show the decode API; replace with per-seq logic.
        (void)samplers[i];  // suppress unused warning in this demo
    }

    // For the actual one-step multi-sequence decode batch:
    // (using sampler on context logits from the last prefill — seq 3)
    llama_token tok3 = llama_sampler_sample(samplers[3], ctx, -1);
    char buf[256];
    int n_piece = llama_token_to_piece(llama_model_get_vocab(model), tok3, buf, sizeof(buf), 0, true);
    printf("Seq 3 first output token: %d = \"%.*s\"\n", tok3, n_piece, buf);

    // ── Cleanup ───────────────────────────────────────────────────────────────

    for (auto s : samplers) llama_sampler_free(s);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

---

## 3.12 How Token Count Drives Every Cost Metric `[FOUNDATIONAL]`

Everything downstream depends on sequence length `L` (the number of tokens):

| Metric                  | Formula                                            | Why tokens matter                  |
|-------------------------|----------------------------------------------------|------------------------------------|
| KV cache size (bytes)   | `2 × n_layers × n_kv_heads × head_dim × L × 2`   | Linear in L                        |
| Prefill FLOPs           | `≈ 4 × n_params × L`                              | Linear in L                        |
| Prefill latency         | `≈ prefill_FLOPs / GPU_TFLOPS`                     | Compute-bound for large L          |
| Decode steps            | `L_output`                                         | One step per output token          |
| TTFT (time to first tok)| `prefill_latency`                                  | Driven by input length             |
| TBT (time between toks) | `model_params_bytes / GPU_bandwidth_bytes_per_s`   | Memory-bound, roughly constant     |
| Cost (API)              | `(L_input + L_output) × price_per_token`           | Direct billing unit                |

### 3.12.1 Worked Example: How Tokenizer Affects Cost

Abi Aryan's production scenario (Chapter 1 case study): 500,000 requests/day, average 800 input tokens, 300 output tokens, A100 cluster.

Switch from LLaMA 2 (SentencePiece, 32K vocab) to LLaMA 3 (tiktoken, 128K vocab) for the same text:

- LLaMA 3's larger vocabulary produces fewer tokens per English word — empirically ~10–15% shorter sequences.
- At 800 input tokens with LLaMA 2 → ~700 input tokens with LLaMA 3.

Daily KV cache memory savings:
```
LLaMA 2: 500,000 × 800 tok × 2 × 32 × 8 × 128 × 2 bytes = 500K × 800 × 131,072 bytes ≈ 52.4 TB-tokens/day
LLaMA 3: 500,000 × 700 tok × ... ≈ 45.9 TB-tokens/day

Savings: ~12.5% KV cache pressure
```

The tokenizer choice is not just correctness — it is cost.

---

## 3.13 ASCII Reference: Full Tokenization Pipeline

```
                 ┌─────────────────────────────────────────────────────────┐
                 │                TOKENIZER PIPELINE                        │
                 └─────────────────────────────────────────────────────────┘

Input text: "Hello, world! How are you?"
     │
     ▼
┌─────────────────────────────────────────┐
│  PRE-TOKENIZATION                        │
│  (model-dependent)                       │
│                                          │
│  tiktoken (LLaMA 3):                     │
│    regex split → ["Hello", ",", " world",│
│                   "!", " How", " are",   │
│                   " you", "?"]           │
│                                          │
│  SentencePiece (LLaMA 2):               │
│    whitespace → ["▁Hello,", "▁world!",  │
│                  "▁How", "▁are", "▁you?"]│
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  BYTE ENCODING (byte-level BPE only)    │
│                                         │
│  Each pre-token → UTF-8 bytes           │
│  "café" → [99, 97, 102, 195, 169]       │
│  bytes → display chars for BPE input    │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  BPE MERGE RULE APPLICATION             │
│                                         │
│  Start: [H, e, l, l, o]                 │
│  Rule 1: H+e → He                       │
│  Rule 2: He+l → Hel                     │
│  Rule k: Hell+o → Hello                 │
│  → single token "Hello" (ID 9906)       │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  SPECIAL TOKEN INJECTION                │
│                                         │
│  Chat template wraps the sequence:      │
│  [BOS] + header tokens + text tokens    │
│  + [EOT] + generation prompt            │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  TOKEN ID SEQUENCE                      │
│                                         │
│  [128000, 128006, 882, 128007, 271,     │
│   9906, 11, 1917, 0, ...]               │
│                                         │
│  Length: determines KV cache, latency,  │
│  cost, and whether it fits in n_ctx.    │
└─────────────────────────────────────────┘
```

---

## 3.14 Self-Check Questions

1. **[FOUNDATIONAL]** A corpus contains the word `"low"` 100 times and `"lowest"` 50 times. Both appear as `l o w </w>` and `l o w e s t </w>` initially. How many occurrences of the pair `(l, o)` will BPE count? How many of `(e, s)`?

2. **[DEEP DIVE]** After the merge `e s → es` has been applied, the pair `(es, t)` appears in `n e w es t </w>`. If `"newest"` has count 3 and `"widest"` has `w i d es t </w>` with count 2, what is the count of `(es, t)`? Should `(es, t)` be merged before or after `(t, </w>)` if they have the same count?

3. **[FOUNDATIONAL]** Explain why byte-level BPE eliminates `<UNK>` tokens. Under what circumstances would a byte-level BPE tokenizer produce a very long token sequence?

4. **[COMMON TRAP]** You have a production system using LLaMA 2 (SentencePiece) and you upgrade to LLaMA 3 (tiktoken). Your prompt is: `"Summarize the following text:"`. List three differences in tokenization output you should expect.

5. **[DEEP DIVE]** In the `llama_batch` struct, what happens if you set `batch.pos[i] = 0` for a decode step when the sequence is already at position 50? What token position gets written in the KV cache?

6. **[FOUNDATIONAL]** A batch of 8 sequences has lengths [12, 45, 3, 78, 22, 9, 55, 31]. Calculate: (a) the total real tokens, (b) the total padded tokens under static batching, (c) the percentage of wasted compute.

---

## Chapter Summary

The journey from string to logit is not trivial. The tokenizer is a first-class performance component:

**BPE training** iteratively merges the most frequent adjacent symbol pair. After `K` merges, frequent words become single tokens and rare words decompose into familiar fragments. The training algorithm is O(K × vocab) with incremental pair-counting data structures.

**BPE encoding** replays merge rules left-to-right on a character-initialized sequence. The merge order (frequency rank) is what determines the split. Earlier merges fire first.

**Byte-level BPE** (GPT-2, LLaMA 3, tiktoken) eliminates `<UNK>` by operating on raw bytes. Every string is encodable; the worst case is one token per byte (3 tokens for a Korean character).

**SentencePiece** treats whitespace as a regular character (`▁`), enabling language-agnostic tokenization. LLaMA 1 and 2 use this. LLaMA 3, DeepSeek, and Qwen switched to tiktoken.

**Special tokens and chat templates** are mandatory for instruction-tuned models. Missing `add_generation_prompt=True` is among the most common silent failures in production.

**Static batching wastes 40–80%** of GPU compute on padding tokens. The batch must run until the longest sequence finishes; short sequences sit idle.

**Continuous batching** (Orca / vLLM) schedules at the iteration level. When a sequence finishes, its slot is immediately filled with a new request. This lifts GPU utilization from 20-40% to 80-95% and is the single biggest practical throughput improvement in modern LLM serving.

**`llama_batch`** is the C interface for submitting one or more sequences at any mix of prefill and decode positions. The `pos` field controls where K/V tensors are written in the cache; getting it wrong silently corrupts the attention window.

In Chapter 4 we will open the transformer block and study exactly what happens inside the attention computation — including why the KV cache exists at all and what it is caching.

---

*Next: Chapter 4 — Inside the Attention Mechanism*


---

## Self-Check Questions

1. A prompt contains 342 characters of English prose. You tokenise it with LLaMA-3's tokeniser (vocabulary size 128 K, ~4 chars/token average). Estimate the number of tokens. *(Section 3.1)*

2. You have 12 requests queued. Three have already produced 200 tokens and are mid-decode. Four are new prefills of length ~512 tokens each. With a token budget of 2 048 tokens per scheduling step, show which requests can fit in the same batch. *(Section 3.3)*

3. Define head-of-line blocking in the context of static batching. Why does continuous batching eliminate it? *(Section 3.4)*

4. A batch of 16 requests all happen to be at the same decode step. How many forward passes does vLLM run? How many would a static-batching engine run? *(Section 3.4)*

5. What is the difference between sequence length and context length? What happens in vLLM when a sequence's KV cache consumption reaches `max_model_len`? *(Section 3.2)*


---

## Worked Solutions

---

### Solution 1 — Token count estimate for 342-character English prose

**What we need:** Estimate tokens from character count.

**Step 1 — The rule of thumb.**

LLaMA-3's tokenizer (based on tiktoken with a 128K vocabulary) processes English prose at approximately **3.5–4.5 characters per token** on average. Common words like "the", "is", "of" are typically single tokens; longer words may split into 2–3 tokens.

**Step 2 — Apply the estimate.**

$$\text{Estimated tokens} = \frac{342 \text{ characters}}{4 \text{ chars/token}} = 85.5 \approx \textbf{85–86 tokens}$$

**Step 3 — Understand the variance.**

The actual token count depends on vocabulary coverage:

- **Dense vocabulary coverage** (common English words): ~4 chars/token → 85 tokens
- **Technical jargon or rare words**: ~3 chars/token → 114 tokens
- **Code or symbols**: ~2 chars/token → 171 tokens

For a quick mental estimate, use 4 chars/token for English prose and 3 chars/token for mixed technical text.

**Step 4 — Why this matters in production.**

Token count directly determines:

- **KV cache memory** consumed (bytes per token × token count)
- **Billing** (LLM API pricing is per token, not per character)
- **Batch slot usage** (a 342-char prompt uses ~86 token slots, not 342)

Always estimate tokens, not characters, when planning memory budgets.

---

### Solution 2 — Batch scheduling with a 2,048-token budget

**What we need:** Show which requests fit in the next scheduling step.

**Setup:**
- Token budget per step: 2,048
- Running (mid-decode): 3 sequences, each contributing 1 token (decode is always 1 token/sequence/step)
- Waiting (new prefills): 4 sequences, each ~512 tokens

**Step 1 — Reserve tokens for running decode sequences.**

The 3 mid-decode sequences each need exactly 1 token slot:

$$\text{Decode tokens consumed} = 3 \times 1 = 3$$

$$\text{Remaining budget} = 2{,}048 - 3 = 2{,}045 \text{ tokens}$$

**Step 2 — Admit waiting prefill sequences greedily.**

With 2,045 tokens remaining, admit waiting requests from the queue:

| Action | Tokens used | Remaining |
|--------|-------------|-----------|
| Admit prefill #1 (512 tokens) | 512 | 2,045 − 512 = **1,533** |
| Admit prefill #2 (512 tokens) | 512 | 1,533 − 512 = **1,021** |
| Admit prefill #3 (512 tokens) | 512 | 1,021 − 512 = **509** |
| Admit prefill #4 (512 tokens)? | 512 | 509 < 512 → **REJECTED** |

**Step 3 — Final batch composition.**

The batch contains:

- 3 decode sequences (3 tokens)
- 3 new prefill sequences (1,536 tokens)
- **Total: 6 sequences, 1,539 tokens**

Prefill sequence #4 stays in the waiting queue and will be the first admitted in the next step.

**Step 4 — With chunked prefill (alternative).**

If chunked prefill is enabled with chunk_size=512, prefill #4 could be partially processed: admit 509 tokens of its 512-token prompt in this step (chunk), finishing the remaining 3 tokens next step. This keeps the GPU fuller but at the cost of added scheduling complexity.

---

### Solution 3 — Head-of-line blocking and continuous batching

**What we need:** Define HoL blocking, explain why continuous batching eliminates it.

**Step 1 — Static batching and the head-of-line problem.**

In traditional static batching, a "batch" is defined as a fixed set of sequences that all start together and must all finish before the next batch begins:

```
Batch 1: [seq_A (200 tokens), seq_B (200 tokens), seq_C (2000 tokens)]
         ← wait for seq_C → seq_C takes 10× longer → A and B are BLOCKED
```

Sequences A and B finish after 200 decode steps but cannot be evicted. Their GPU slots sit idle while seq_C continues generating. New requests in the queue cannot start. This is **head-of-line blocking**: the slowest sequence in the batch blocks all subsequent requests.

**Step 2 — Why it wastes GPU time.**

The GPU runs one forward pass per step regardless of how many sequences are "active" in that step. Once A and B finish, the GPU is running at reduced utilization (fewer active sequences), but slots are still occupied.

**Step 3 — How continuous batching (iteration-level scheduling) eliminates it.**

In continuous batching, the scheduler re-evaluates the batch **after every single decode step**:

1. After step N: check which sequences produced an EOS token.
2. Immediately free their KV cache blocks.
3. Immediately promote the next waiting request into the freed slots.
4. The next forward pass already includes the new sequence.

```
Step 200: seq_A finishes (EOS) → freed → seq_D (new, waiting) immediately promoted
Step 200: seq_B finishes (EOS) → freed → seq_E promoted
Step 201: batch = [seq_C, seq_D, seq_E] — GPU stays full
```

There is no "wait for the whole batch" gate. Each step's batch is independent.

---

### Solution 4 — Forward passes for 16 decode sequences

**What we need:** Compare vLLM (continuous batching) vs static batching engine.

**Answer:**

| Engine | Forward passes for 16 simultaneous decode sequences |
|--------|-----------------------------------------------------|
| vLLM (continuous batching) | **1** |
| Naive static batching | **1** (if batched) or **16** (if sequential) |

**Step 1 — vLLM's approach.**

vLLM batches all 16 decode tokens from 16 sequences into a single forward pass. The GPU processes a batch of shape `[16, 1, d_model]` in one shot. This is 1 forward pass.

**Step 2 — Static batching (properly implemented).**

A well-implemented static batch engine would also batch 16 sequences in one forward pass — provided they are all in the same batch. The static batch engine's problem is not the *number* of forward passes per step, but the lack of flexibility to admit new sequences mid-batch.

**Step 3 — The key difference (for clarity).**

The 1-pass answer applies *per step* for both engines. The distinction is what happens **between steps**:

- **Static batching:** holds the batch fixed until all sequences finish
- **Continuous batching:** re-evaluates after every step, immediately evicting finished sequences and admitting new ones

For 16 sequences all at the same decode step: **1 forward pass** for both engines. The continuous batching advantage shows up over *time* — fewer wasted cycles across many steps.

---

### Solution 5 — Sequence length vs context length, and what happens at max_model_len

**What we need:** Distinguish the two terms and explain the vLLM limit.

**Step 1 — Sequence length (runtime concept).**

*Sequence length* is the *current* total number of tokens in a sequence:

$$\text{sequence\_length} = \text{prompt tokens} + \text{generated tokens so far}$$

It grows by 1 every decode step. At step 0 (after prefill), sequence_length = num_prompt_tokens. At step 50, it equals num_prompt_tokens + 50.

**Step 2 — Context length / max_model_len (architectural limit).**

*Context length* (also called `max_model_len` in vLLM) is the **maximum total tokens** the model's positional encoding supports. This is baked into the model architecture — a LLaMA-3 8B with `max_position_embeddings=8192` cannot meaningfully attend beyond 8,192 positions because its RoPE embeddings were only trained up to that length.

**Step 3 — What happens when sequence_length reaches max_model_len.**

When `sequence_length == max_model_len`, vLLM:

1. **Stops generation** for that sequence (regardless of whether EOS was produced).
2. Sets `finish_reason = "length"` in the API response.
3. Frees all KV blocks for that sequence.

The sequence is not truncated mid-token — it is terminated cleanly. The client receives whatever tokens were generated up to `max_model_len`.

**Step 4 — Production implications.**

If users frequently hit `max_model_len`:

- They receive incomplete responses
- Long sequences occupy KV cache for many steps before being forcibly terminated
- Use `max_tokens` in the API request to terminate early when output is complete

To support longer contexts, models must be fine-tuned with extended positional embeddings (e.g., LLaMA-3.1 with rope_scaling extends the base 8K to 128K).

