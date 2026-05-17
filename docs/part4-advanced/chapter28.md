# Chapter 28: llama.cpp as a Platform

llama.cpp is commonly described as a "inference engine," but that undersells it.
It is a complete, self-contained inference platform: a model format (GGUF), a runtime
(the C99/C++ library), a server (OpenAI-compatible HTTP), a tokenizer, a sampler,
backends for every major compute substrate (CUDA, Metal, ROCm, Vulkan, OpenCL, CPU
AVX2/AVX512/AMX, Apple ANE), and a grammar engine for structured output.

This chapter dissects every layer.
By the end you will be able to embed llama.cpp in your own C++ application, call it
from Python via ctypes, tune its build for a specific GPU, constrain output to JSON
schemas or context-free grammars, and understand exactly what happens between
`llama_model_load()` and the first decoded token.

---

## 28.1  GGUF: The Model Format

### 28.1.1  Why a New Format?

The original llama.cpp used custom binary `.bin` files.
After Georgi Gerganov moved to quantized formats, the project evolved through several
incompatible generations (GGML, GGMF, GGJT) before settling on GGUF in August 2023.

GGUF (GGML Universal Format) was designed with three goals:

1. **Self-describing** — all metadata (architecture, tokenizer, hyperparameters) live in
   the file; no external config JSON needed.
2. **Memory-mappable** — tensor data is page-aligned so `mmap()` can serve it directly
   from disk without copying into RAM.
3. **Extensible** — new metadata keys can be added without breaking older readers via
   a versioned key-value store.

### 28.1.2  File Structure

A GGUF file has four consecutive regions:

```
┌─────────────────────────────────────────────────────────────┐
│  Magic: "GGUF"  (4 bytes)                                   │
│  Version: uint32  (currently 3)                             │
│  n_tensors: uint64                                          │
│  n_kv: uint64                                               │
├─────────────────────────────────────────────────────────────┤
│  Key-Value Store  (n_kv entries)                            │
│    key: string  (Pascal-style: length + bytes)              │
│    value_type: uint32  (0=uint8, 4=int32, 6=float32,        │
│                         8=string, 9=array, …)               │
│    value: (type-dependent bytes)                            │
├─────────────────────────────────────────────────────────────┤
│  Tensor Info  (n_tensors entries)                           │
│    name: string                                             │
│    n_dims: uint32                                           │
│    dims: uint64[n_dims]                                     │
│    type: uint32  (quantization type enum)                   │
│    offset: uint64  (byte offset into tensor data region)    │
├─────────────────────────────────────────────────────────────┤
│  Padding: align to 32-byte boundary                         │
├─────────────────────────────────────────────────────────────┤
│  Tensor Data  (mmap-friendly, page-aligned)                 │
│    tensor[0] data … tensor[n_tensors-1] data               │
└─────────────────────────────────────────────────────────────┘
```

### 28.1.3  Key Metadata Fields

The KV store contains standardized keys that every reader understands:

| Key | Type | Example |
|---|---|---|
| `general.architecture` | string | `"llama"` |
| `general.name` | string | `"Llama 3.1 8B Instruct"` |
| `llama.context_length` | uint32 | `131072` |
| `llama.embedding_length` | uint32 | `4096` |
| `llama.block_count` | uint32 | `32` |
| `llama.attention.head_count` | uint32 | `32` |
| `llama.attention.head_count_kv` | uint32 | `8` |
| `llama.rope.freq_base` | float32 | `500000.0` |
| `llama.rope.dimension_count` | uint32 | `128` |
| `tokenizer.ggml.model` | string | `"bpe"` |
| `tokenizer.ggml.tokens` | array[string] | `["<unk>", "<s>", …]` |
| `tokenizer.ggml.scores` | array[float32] | `[0.0, 0.0, …]` |
| `tokenizer.ggml.token_type` | array[int32] | `[2, 3, 1, …]` |
| `tokenizer.chat_template` | string | `"{% if …"` |

### 28.1.4  Quantization Type Enum

The tensor `type` field encodes the quantization format:

| Value | Name | Bits/weight | Description |
|---|---|---|---|
| 0 | F32 | 32 | Full precision |
| 1 | F16 | 16 | Half precision |
| 2 | Q4_0 | 4.5 | 4-bit, 32-weight blocks, shared scale |
| 10 | Q4_K_S | 4.5 | 4-bit, k-quant, small |
| 11 | Q4_K_M | 4.84 | 4-bit, k-quant, medium (recommended) |
| 12 | Q5_K_S | 5.5 | 5-bit k-quant, small |
| 13 | Q5_K_M | 5.68 | 5-bit k-quant, medium |
| 14 | Q6_K | 6.56 | 6-bit k-quant |
| 15 | Q8_0 | 8.5 | 8-bit, fast dequant |
| 30 | IQ4_NL | 4.5 | imatrix-aware 4-bit |
| 34 | BF16 | 16 | BFloat16 |

K-quants store quantization scales and minimums at super-block granularity
(256 weights per super-block, subdivided into 8 blocks of 32).
This allows the quantization error to be absorbed by per-block correction rather than
globally, which significantly improves quality over Q4_0 at the same bit count.

### 28.1.5  Reading GGUF in Python

```python
# gguf_reader.py
"""
Minimal GGUF reader — demonstrates the file format without external dependencies.
For production use: pip install gguf  (the official Python library)
"""
import struct, sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


GGUF_MAGIC   = b"GGUF"
VALUE_TYPES  = {
    0: ("uint8",   "B", 1),
    1: ("int8",    "b", 1),
    2: ("uint16",  "H", 2),
    3: ("int16",   "h", 2),
    4: ("uint32",  "I", 4),
    5: ("int32",   "i", 4),
    6: ("float32", "f", 4),
    7: ("bool",    "B", 1),
    8: ("string",  None, None),
    9: ("array",   None, None),
    10:("uint64",  "Q", 8),
    11:("int64",   "q", 8),
    12:("float64", "d", 8),
}

QUANT_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 10: "Q4_K_S", 11: "Q4_K_M",
    12: "Q5_K_S", 13: "Q5_K_M", 14: "Q6_K", 15: "Q8_0", 30: "IQ4_NL", 34: "BF16",
}


@dataclass
class TensorInfo:
    name: str
    dims: List[int]
    quant_type: int
    offset: int

    @property
    def quant_name(self) -> str:
        return QUANT_NAMES.get(self.quant_type, f"UNKNOWN({self.quant_type})")

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


class GGUFReader:
    def __init__(self, path: str):
        self.path = path
        self.kv: Dict[str, Any] = {}
        self.tensors: List[TensorInfo] = []
        self._parse()

    def _parse(self):
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file: {magic!r}")

            version   = self._read_u32(f)
            n_tensors = self._read_u64(f)
            n_kv      = self._read_u64(f)

            self.version   = version
            self.n_tensors = n_tensors

            # Key-Value store
            for _ in range(n_kv):
                key   = self._read_string(f)
                vtype = self._read_u32(f)
                val   = self._read_value(f, vtype)
                self.kv[key] = val

            # Tensor info
            for _ in range(n_tensors):
                name   = self._read_string(f)
                n_dims = self._read_u32(f)
                dims   = [self._read_u64(f) for _ in range(n_dims)]
                qtype  = self._read_u32(f)
                offset = self._read_u64(f)
                self.tensors.append(TensorInfo(name, dims, qtype, offset))

            self.data_offset = f.tell()
            # Align to 32 bytes
            if self.data_offset % 32 != 0:
                self.data_offset += 32 - (self.data_offset % 32)

    def _read_u32(self, f): return struct.unpack("<I", f.read(4))[0]
    def _read_u64(self, f): return struct.unpack("<Q", f.read(8))[0]
    def _read_i32(self, f): return struct.unpack("<i", f.read(4))[0]
    def _read_f32(self, f): return struct.unpack("<f", f.read(4))[0]

    def _read_string(self, f) -> str:
        length = self._read_u64(f)
        return f.read(length).decode("utf-8", errors="replace")

    def _read_value(self, f, vtype: int):
        if vtype == 8:   # string
            return self._read_string(f)
        if vtype == 9:   # array
            elem_type = self._read_u32(f)
            count     = self._read_u64(f)
            return [self._read_value(f, elem_type) for _ in range(min(count, 64))]
        info = VALUE_TYPES.get(vtype)
        if info is None:
            raise ValueError(f"Unknown value type: {vtype}")
        _, fmt, size = info
        return struct.unpack(f"<{fmt}", f.read(size))[0]

    def summary(self):
        arch = self.kv.get("general.architecture", "?")
        name = self.kv.get("general.name", "?")
        ctx  = self.kv.get(f"{arch}.context_length", "?")
        emb  = self.kv.get(f"{arch}.embedding_length", "?")
        blk  = self.kv.get(f"{arch}.block_count", "?")
        rope_base = self.kv.get(f"{arch}.rope.freq_base", "?")

        print(f"GGUF v{self.version}: {name}")
        print(f"  Architecture:   {arch}")
        print(f"  Context length: {ctx:,}" if isinstance(ctx, int) else f"  Context length: {ctx}")
        print(f"  Embedding dim:  {emb}")
        print(f"  Layers:         {blk}")
        print(f"  RoPE base:      {rope_base}")
        print(f"  Tensors:        {self.n_tensors}")
        print()

        # Quantization type breakdown
        quant_counts: Dict[str, int] = {}
        quant_params: Dict[str, int] = {}
        for t in self.tensors:
            qn = t.quant_name
            quant_counts[qn] = quant_counts.get(qn, 0) + 1
            quant_params[qn] = quant_params.get(qn, 0) + t.n_elements

        print("  Quantization breakdown:")
        for qn, count in sorted(quant_counts.items()):
            params_m = quant_params[qn] / 1e6
            print(f"    {qn:12s}  {count:5d} tensors  {params_m:8.1f}M params")
        print()

        # Sample tensors
        print("  First 10 tensors:")
        for t in self.tensors[:10]:
            shape = " × ".join(str(d) for d in t.dims)
            print(f"    {t.name:50s}  [{shape:30s}]  {t.quant_name}")
        if len(self.tensors) > 10:
            print(f"    … and {len(self.tensors) - 10} more")


# ─────────────────────────────────────────────────────────────────────────────
# GGUF CONVERSION NOTES (without an actual GGUF file for the demo)
# ─────────────────────────────────────────────────────────────────────────────

def demo_gguf_structure():
    """Show GGUF structure without requiring an actual model file."""
    print("=" * 70)
    print("GGUF File Structure Overview")
    print("=" * 70)

    struct_diagram = """
  Offset   Size    Content
  ──────   ────    ───────
  0        4       Magic: 'GGUF'
  4        4       Version (uint32, currently 3)
  8        8       n_tensors (uint64)
  16       8       n_kv (uint64)
  24       var     Key-Value metadata store
                     [key: string][value_type: uint32][value: ...]
                     ... repeated n_kv times
  var      var     Tensor info array
                     [name: string][n_dims: u32][dims: u64[]][type: u32][offset: u64]
                     ... repeated n_tensors times
  align    pad     Padding to 32-byte alignment
  data     var     Tensor data (memory-mappable)
                     tensor[0], tensor[1], ..., tensor[n_tensors-1]
"""
    print(struct_diagram)

    # Show quantization bit-per-weight table
    print("  Quantization formats (approximate bits/weight):")
    quants = [
        ("F32",    32.0,  "Reference precision"),
        ("F16",    16.0,  "Training precision"),
        ("BF16",   16.0,  "Brain float (better range than F16)"),
        ("Q8_0",    8.5,  "Fast dequant, near-lossless"),
        ("Q6_K",    6.56, "High quality, 4× compression vs FP16"),
        ("Q5_K_M",  5.68, "Good quality/size balance"),
        ("Q4_K_M",  4.84, "Recommended default — best Q/size"),
        ("Q4_K_S",  4.5,  "Slightly smaller than Q4_K_M"),
        ("Q4_0",    4.5,  "Legacy, avoid for new models"),
        ("IQ4_NL",  4.5,  "imatrix-aware, better than Q4_0"),
    ]
    print(f"  {'Name':12}  {'Bits':6}  {'Llama-3.1-8B size':20}  Notes")
    print("  " + "-" * 65)
    for name, bits, note in quants:
        size_gb = 8e9 * bits / 16 / 1e9  # relative to FP16
        print(f"  {name:12}  {bits:6.2f}  {size_gb:>8.1f} GB               {note}")
    print()


if __name__ == "__main__":
    import os, sys
    demo_gguf_structure()

    # If a GGUF path is provided as argument, read it
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        print(f"Reading: {sys.argv[1]}")
        r = GGUFReader(sys.argv[1])
        r.summary()
    else:
        print("  (Pass a .gguf file path as argument to inspect a real model)")
```

---

## 28.2  The llama.cpp C API

### 28.2.1  Object Model

llama.cpp exposes three primary opaque objects:

```
llama_model   — loaded weights, tokenizer, and architecture metadata
                Shared across multiple inference contexts.
                One per model file.

llama_context — inference state: KV cache, sampling state, thread pool.
                One per concurrent inference session.
                Multiple contexts can share one model.

llama_batch   — a collection of tokens (with positions and sequence IDs)
                to be processed in a single forward pass.
```

The ownership hierarchy is: `model` → `context`(s) → `batch`.

### 28.2.2  Minimal Inference in C

```c
// minimal_inference.c
// Demonstrates the complete llama.cpp inference lifecycle
// Build: gcc -o minimal_inference minimal_inference.c -lllama -lm
// Note: link against libllama.a from a compiled llama.cpp build

#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf \"prompt text\"\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* prompt     = argv[2];

    // ── 1. Initialize backend (loads CUDA/Metal/CPU kernels) ──────────────
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // ── 2. Load model ─────────────────────────────────────────────────────
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;   // offload all layers to GPU

    struct llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    // ── 3. Create context ─────────────────────────────────────────────────
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 4096;    // context window
    cparams.n_threads = 4;       // CPU threads for non-GPU layers
    cparams.flash_attn = true;   // enable FlashAttention

    struct llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        return 1;
    }

    // ── 4. Tokenize ───────────────────────────────────────────────────────
    int n_prompt_tokens = -llama_tokenize(
        model, prompt, strlen(prompt),
        NULL, 0,         // NULL = just count
        true,            // add BOS
        true             // special tokens
    );

    llama_token* tokens = malloc(n_prompt_tokens * sizeof(llama_token));
    if (llama_tokenize(model, prompt, strlen(prompt),
                       tokens, n_prompt_tokens, true, true) < 0) {
        fprintf(stderr, "Tokenization failed\n");
        free(tokens);
        goto cleanup;
    }

    printf("Prompt: %d tokens\n", n_prompt_tokens);

    // ── 5. Build batch and prefill ────────────────────────────────────────
    struct llama_batch batch = llama_batch_init(n_prompt_tokens, 0, 1);

    for (int i = 0; i < n_prompt_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_prompt_tokens - 1);  // only need last logits
    }
    batch.n_tokens = n_prompt_tokens;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill failed\n");
        goto cleanup;
    }

    // ── 6. Greedy decode loop ─────────────────────────────────────────────
    const int max_new_tokens = 128;
    int       n_cur          = n_prompt_tokens;
    int       n_decoded      = 0;

    while (n_decoded < max_new_tokens) {
        // Sample next token (greedy)
        float* logits   = llama_get_logits_ith(ctx, -1);
        int    n_vocab   = llama_n_vocab(model);

        llama_token next_token = 0;
        float       best_logit = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                next_token = i;
            }
        }

        // Check EOS
        if (llama_token_is_eog(model, next_token)) {
            printf("\n[EOS]\n");
            break;
        }

        // Decode token to text and print
        char piece[256];
        int  n = llama_token_to_piece(model, next_token, piece, sizeof(piece), 0, true);
        if (n > 0) {
            piece[n] = '\0';
            printf("%s", piece);
            fflush(stdout);
        }

        // Build single-token batch for next step
        struct llama_batch step_batch = llama_batch_init(1, 0, 1);
        step_batch.token[0]     = next_token;
        step_batch.pos[0]       = n_cur;
        step_batch.n_seq_id[0]  = 1;
        step_batch.seq_id[0][0] = 0;
        step_batch.logits[0]    = true;
        step_batch.n_tokens     = 1;

        if (llama_decode(ctx, step_batch) != 0) break;
        llama_batch_free(step_batch);

        n_cur++;
        n_decoded++;
    }

    printf("\n--- Generated %d tokens ---\n", n_decoded);

cleanup:
    free(tokens);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
```

### 28.2.3  The Full API Surface

The llama.cpp public API (as of mid-2025) groups into these namespaces:

**Backend management:**
```c
void llama_backend_init(void);
void llama_backend_free(void);
void llama_numa_init(enum ggml_numa_strategy numa);
```

**Model loading:**
```c
struct llama_model* llama_load_model_from_file(const char* path, llama_model_params params);
void llama_free_model(struct llama_model* model);

// Architecture queries
int32_t llama_n_ctx_train(const struct llama_model* model);
int32_t llama_n_embd(const struct llama_model* model);
int32_t llama_n_layer(const struct llama_model* model);
int32_t llama_n_head(const struct llama_model* model);
```

**Context creation:**
```c
struct llama_context* llama_new_context_with_model(
    struct llama_model* model, llama_context_params params);
void llama_free(struct llama_context* ctx);
```

**Tokenization:**
```c
int32_t llama_tokenize(const struct llama_model* model,
    const char* text, int32_t text_len,
    llama_token* tokens, int32_t n_tokens_max,
    bool add_special, bool parse_special);

int32_t llama_token_to_piece(const struct llama_model* model,
    llama_token token, char* buf, int32_t length,
    int32_t lstrip, bool special);

const char* llama_token_get_text(const struct llama_model* model, llama_token token);
bool llama_token_is_eog(const struct llama_model* model, llama_token token);
llama_token llama_token_bos(const struct llama_model* model);
llama_token llama_token_eos(const struct llama_model* model);
```

**Inference:**
```c
int32_t llama_decode(struct llama_context* ctx, struct llama_batch batch);
float*  llama_get_logits(struct llama_context* ctx);
float*  llama_get_logits_ith(struct llama_context* ctx, int32_t i);
float*  llama_get_embeddings(struct llama_context* ctx);
```

**Sampling (llama_sampler API, added ~0.3.0):**
```c
struct llama_sampler* llama_sampler_chain_init(llama_sampler_chain_params params);
void llama_sampler_chain_add(struct llama_sampler* chain, struct llama_sampler* smpl);
llama_token llama_sampler_sample(struct llama_sampler* smpl,
    struct llama_context* ctx, int32_t idx);
void llama_sampler_free(struct llama_sampler* smpl);

// Built-in samplers
struct llama_sampler* llama_sampler_init_greedy(void);
struct llama_sampler* llama_sampler_init_temp(float t);
struct llama_sampler* llama_sampler_init_top_p(float p, size_t min_keep);
struct llama_sampler* llama_sampler_init_top_k(int32_t k);
struct llama_sampler* llama_sampler_init_min_p(float p, size_t min_keep);
struct llama_sampler* llama_sampler_init_grammar(const struct llama_model* model,
    const char* grammar_str, const char* grammar_root);
struct llama_sampler* llama_sampler_init_penalties(
    llama_token* last_tokens, size_t last_n, float repeat_penalty,
    float freq_penalty, float presence_penalty);
```

**KV cache management:**
```c
void llama_kv_cache_clear(struct llama_context* ctx);
void llama_kv_cache_seq_rm(struct llama_context* ctx,
    llama_seq_id seq_id, llama_pos p0, llama_pos p1);
void llama_kv_cache_seq_cp(struct llama_context* ctx,
    llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1);
void llama_kv_cache_seq_shift(struct llama_context* ctx,
    llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta);
```

The KV cache management functions are powerful: `seq_cp` enables fork-join beam search
(copy the KV state of one sequence into a new sequence ID), and `seq_shift` enables
sliding window inference (evict the oldest positions and shift all subsequent positions).

---

## 28.3  Grammar-Constrained Decoding

One of llama.cpp's most practical capabilities is its GBNF (GGML Backus-Naur Form)
grammar engine, which constrains the sampler so that every generated token is guaranteed
to produce valid output within a specified grammar.

### 28.3.1  How It Works

At each decode step, after computing logits, the grammar engine:

1. Maintains the current **parse state** — a set of parser stacks encoding which grammar
   productions are still reachable.
2. Computes a **logit bias mask**: for every token in the vocabulary, determines whether
   accepting that token would advance any reachable production.
   Tokens that would violate the grammar have their logit set to `-∞`.
3. Sampling proceeds normally on the masked logits.
4. After sampling, the parser stack advances past the accepted token.

This is O(V × G) per decode step where V is vocabulary size and G is grammar complexity.
For a simple JSON grammar and a 128K-token vocabulary, this adds ~5–15% latency per step.

### 28.3.2  GBNF Grammar Syntax

GBNF is a superset of standard BNF with some extensions borrowed from PEG (parsing
expression grammars).

```
# JSON grammar (simplified) — save as json.gbnf
root   ::= object
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}"
array  ::= "[" ws (value ("," ws value)*)? "]"
string ::= "\"" (
           [^"\\] |
           "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
         )* "\""
number ::= ["-"]? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws     ::= [ \t\n]*
```

Use in llama.cpp CLI:
```bash
llama-cli -m model.gguf \
    --grammar-file json.gbnf \
    -p "Extract the name and age from: 'Alice is 30 years old.'"
```

Or via the server:
```bash
curl http://localhost:8080/completion \
  -d '{"prompt": "Extract person data:", "grammar": "root ::= object\n..."}'
```

### 28.3.3  Pre-built Grammars

llama.cpp ships several grammars in `grammars/`:

- `json.gbnf` — valid JSON
- `json_arr.gbnf` — JSON array at top level
- `list.gbnf` — newline-separated list
- `chess.gbnf` — legal chess moves in algebraic notation
- `c.gbnf` — syntactically valid C code

For Pydantic-style structured output, use the `--json-schema` flag (llama.cpp server ≥ 0.4.0):
```bash
# Generate a JSON object matching a Python dataclass schema
curl http://localhost:8080/v1/chat/completions \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Extract person info from: Alice is 30."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age":  {"type": "integer"}
          },
          "required": ["name", "age"]
        }
      }
    }
  }'
```

---

## 28.4  The llama.cpp Server — Deep Dive

### 28.4.1  Server Architecture

`llama-server` (formerly `server`) is a single-process HTTP server built on top of the
llama.cpp library.
Its architecture:

```
HTTP listener thread
  │
  ├── /completion, /chat/completions, /tokenize, /detokenize
  │
  └── Request queue
        │
        └── Inference thread (single-threaded by default)
              │
              ├── llama_decode (batch of pending tokens)
              └── Response SSE stream to clients
```

The inference loop runs a **continuous batching** scheduler:

- Requests enter the queue as they arrive.
- Each iteration, the scheduler assembles a batch from:
  - New prefill tokens from the head of the queue
  - One decode token per active generation slot

- A single `llama_decode` call processes the combined batch.
- Finished sequences are evicted; new ones fill their slots.

The number of concurrent generation slots is `-np` (parallel slots, default 1).
Each slot has its own KV cache range within the global context buffer.

### 28.4.2  Key Server Flags

```bash
llama-server \
    -m Llama-3.1-8B-Instruct-Q4_K_M.gguf \

    # Context and concurrency
    -c 32768        \   # total context = sum of all slots
    -np 4           \   # 4 parallel slots; each gets 32768/4 = 8192 tokens
    --rope-freq-base 500000 \

    # GPU offload
    -ngl 99         \   # offload all layers to GPU
    --flash-attn    \   # mandatory for long context

    # Network
    --host 127.0.0.1 \
    --port 8080      \

    # Performance
    -b 512          \   # batch size for prefill (tokens per decode call)
    -ub 512         \   # micro-batch size for ubatch optimization
    --threads 4     \   # CPU threads for CPU-side ops

    # Logging and metrics
    --log-format json \
    --metrics           # expose /metrics Prometheus endpoint
```

### 28.4.3  API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status":"ok"}` when ready |
| `/props` | GET | Model properties (context len, etc.) |
| `/tokenize` | POST | Tokenize text → token IDs |
| `/detokenize` | POST | Token IDs → text |
| `/completion` | POST | Single-turn completion (streaming via SSE) |
| `/chat/completions` | POST | OpenAI-compatible chat endpoint |
| `/embeddings` | POST | Compute embeddings |
| `/v1/models` | GET | OpenAI model list endpoint |
| `/infill` | POST | Fill-in-the-middle (requires FIM model) |
| `/metrics` | GET | Prometheus-format metrics |
| `/slots` | GET | Current slot occupancy and state |
| `/lora-adapters` | GET/POST | List/load LoRA adapters at runtime |

### 28.4.4  Slots API

The `/slots` endpoint exposes per-slot state for monitoring:

```json
[
  {
    "id": 0,
    "state": 1,
    "prompt": "Tell me about",
    "n_prompt": 4,
    "n_decoded": 23,
    "n_past": 27,
    "temperature": 0.8,
    "t_start_ms": 1700000000000,
    "t_last_ms":  1700000000234
  },
  {
    "id": 1,
    "state": 0,
    "prompt": "",
    "n_prompt": 0,
    "n_decoded": 0
  }
]
```

`state` values: `0` = idle, `1` = processing.

---

## 28.5  Build System and Backends

### 28.5.1  CMake Build

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# CUDA build (H100/A100/RTX) — fastest option on NVIDIA hardware
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="90;80;89" \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Metal build (Apple Silicon) — native GPU on M1/M2/M3/M4
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)

# ROCm build (AMD GPUs)
cmake -B build \
    -DGGML_HIPBLAS=ON \
    -DAMDGPU_TARGETS="gfx1100;gfx1030" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Vulkan build (cross-platform GPU — Intel, AMD, NVIDIA)
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# CPU-only build with AVX512
cmake -B build \
    -DGGML_AVX512=ON \
    -DGGML_AVX512_VBMI=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### 28.5.2  CUDA Architectures

| GPU Family | CUDA Arch |
|---|---|
| H100 (Hopper) | `90` |
| A100, A40, A30 (Ampere) | `80` |
| RTX 4090, 4080, 4070 (Ada) | `89` |
| RTX 3090, 3080, A6000 (Ampere) | `86` |
| RTX 2080, T4 (Turing) | `75` |
| V100 (Volta) | `70` |

Always specify the exact arch to avoid compiling for every CUDA version, which
increases binary size 10× and compile time proportionally.

### 28.5.3  Backend Selection at Runtime

When built with multiple backends, llama.cpp selects based on availability:

1. CUDA (if `GGML_CUDA_DEVICE` env var is set or any NVIDIA GPU is detected)
2. Metal (on macOS with Apple GPU)
3. CPU (always available as fallback)

Force a specific device:
```bash
CUDA_VISIBLE_DEVICES=0 llama-server ...   # use first GPU only
CUDA_VISIBLE_DEVICES=0,1 llama-server ... # use first two GPUs (tensor parallel)
GGML_METAL_DEVICE=0 llama-server ...      # macOS: select Metal device
```

### 28.5.4  Memory-Mapped Model Loading

When llama.cpp loads a GGUF file, the default path is `mmap`:

```

1. open() the GGUF file
2. Parse header (KV store + tensor info) into CPU memory
3. mmap() the tensor data region
4. For CPU inference: tensors are accessed directly via mmap pointers
5. For GPU inference: copy layers to GPU asynchronously in the background
   (llama_model_params.use_mmap = true, default)
```

Benefits of mmap:

- **Fast startup**: the OS does not read the entire file on load; pages are faulted in
  on first access.

- **Shared memory**: if two processes load the same model, the OS shares the physical
  pages (no duplication in RAM).

- **Graceful swap**: if the system runs low on RAM, the OS can evict model pages to disk
  and reload them on demand without explicit file I/O.

Disable mmap for network filesystems (NFS/CIFS) or very old kernels:
```bash
llama-server -m model.gguf --no-mmap
```

---

## 28.6  Python Bindings via llama-cpp-python

`llama-cpp-python` wraps the C API with ctypes, providing both a low-level mirror of the
C API and a high-level `Llama` class.

### 28.6.1  Installation

```bash
# CPU-only
pip install llama-cpp-python

# CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Metal (macOS)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# ROCm
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python
```

### 28.6.2  High-Level API

```python
# llamacpp_platform_demo.py
"""
Chapter 28 — llama.cpp as a Platform (Python demo)

Demonstrates:
  1. GGUF structure overview
  2. llama-cpp-python API patterns (with mock when no model present)
  3. Grammar-constrained decoding simulation
  4. Sampler chain construction
  5. KV cache management patterns
  6. Performance tuning guide
"""
from __future__ import annotations
import json, math, time, struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GGUF STRUCTURE ANALYZER (works on any GGUF file)
# ─────────────────────────────────────────────────────────────────────────────

QUANT_BITS = {
    0: 32.0, 1: 16.0, 2: 4.5, 10: 4.5, 11: 4.84,
    12: 5.5, 13: 5.68, 14: 6.56, 15: 8.5, 30: 4.5, 34: 16.0,
}
QUANT_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 10: "Q4_K_S", 11: "Q4_K_M",
    12: "Q5_K_S", 13: "Q5_K_M", 14: "Q6_K", 15: "Q8_0", 30: "IQ4_NL", 34: "BF16",
}


def analyze_gguf_size(param_billions: float) -> None:
    """Print model size estimates across all quantization formats."""
    print("=" * 68)
    print(f"GGUF Size Estimates — {param_billions}B parameter model")
    print("=" * 68)
    print(f"  {'Format':12}  {'Bits/W':7}  {'Size (GB)':10}  {'vs FP16':8}  Notes")
    print("  " + "-" * 60)
    rows = [
        (0,  "F32"),
        (1,  "F16"),
        (34, "BF16"),
        (15, "Q8_0"),
        (14, "Q6_K"),
        (13, "Q5_K_M"),
        (12, "Q5_K_S"),
        (11, "Q4_K_M"),
        (10, "Q4_K_S"),
        (2,  "Q4_0"),
        (30, "IQ4_NL"),
    ]
    fp16_size = param_billions * 1e9 * 2 / 1e9
    for qtype, name in rows:
        bits = QUANT_BITS[qtype]
        size = param_billions * 1e9 * bits / 16 / 1e9  # relative to fp16=2 bytes
        ratio = fp16_size / size
        notes = ""
        if name == "Q4_K_M": notes = "← recommended"
        if name == "F16":     notes = "← reference"
        print(f"  {name:12}  {bits:7.2f}  {size:10.2f}  {ratio:7.1f}×  {notes}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SAMPLER CHAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenCandidate:
    id: int
    logit: float
    p: float = 0.0     # probability after softmax


def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def apply_temperature(candidates: List[TokenCandidate], temp: float
                       ) -> List[TokenCandidate]:
    """Divide logits by temperature before softmax."""
    for c in candidates:
        c.logit /= temp
    probs = softmax([c.logit for c in candidates])
    for c, p in zip(candidates, probs):
        c.p = p
    return candidates


def apply_top_k(candidates: List[TokenCandidate], k: int
                ) -> List[TokenCandidate]:
    """Keep only the top-k tokens by probability."""
    candidates.sort(key=lambda c: c.p, reverse=True)
    kept = candidates[:k]
    # Renormalize
    total = sum(c.p for c in kept)
    for c in kept:
        c.p /= total
    return kept


def apply_top_p(candidates: List[TokenCandidate], p: float
                ) -> List[TokenCandidate]:
    """Keep tokens whose cumulative probability ≤ p."""
    candidates.sort(key=lambda c: c.p, reverse=True)
    cumsum = 0.0
    kept = []
    for c in candidates:
        if cumsum >= p and len(kept) > 0:
            break
        kept.append(c)
        cumsum += c.p
    total = sum(c.p for c in kept)
    for c in kept:
        c.p /= total
    return kept


def apply_min_p(candidates: List[TokenCandidate], min_p: float
                ) -> List[TokenCandidate]:
    """Keep tokens with p ≥ min_p * max_p."""
    max_p = max(c.p for c in candidates)
    threshold = min_p * max_p
    kept = [c for c in candidates if c.p >= threshold]
    if not kept:
        kept = [max(candidates, key=lambda c: c.p)]
    total = sum(c.p for c in kept)
    for c in kept:
        c.p /= total
    return kept


def apply_repetition_penalty(candidates: List[TokenCandidate],
                              recent_tokens: List[int],
                              penalty: float) -> List[TokenCandidate]:
    """Apply repetition penalty to recently seen tokens."""
    recent_set = set(recent_tokens)
    for c in candidates:
        if c.id in recent_set:
            c.logit = c.logit / penalty if c.logit > 0 else c.logit * penalty
    # Re-run softmax after modifying logits
    probs = softmax([c.logit for c in candidates])
    for c, p in zip(candidates, probs):
        c.p = p
    return candidates


def demo_sampler_chain():
    """Demonstrate the sampler chain pipeline on mock logits."""
    import random
    rng = random.Random(42)

    # Mock vocabulary of 16 tokens with random logits
    V = 16
    vocab = [f"tok_{i}" for i in range(V)]
    logits = [rng.gauss(0, 2) for _ in range(V)]

    # Simulate: temperature=0.7, top_k=8, top_p=0.9, min_p=0.05
    candidates = [TokenCandidate(id=i, logit=logits[i]) for i in range(V)]

    print("=" * 60)
    print("Sampler Chain Simulation (V=16, mock logits)")
    print("=" * 60)

    def show(stage: str, cands: List[TokenCandidate]):
        top3 = sorted(cands, key=lambda c: c.p, reverse=True)[:3]
        print(f"  After {stage:20s}: {len(cands):3d} tokens | "
              f"top={top3[0].p:.3f}/{top3[1].p:.3f}/{top3[2].p:.3f}")

    # Step 1: Initial softmax
    probs = softmax([c.logit for c in candidates])
    for c, p in zip(candidates, probs):
        c.p = p
    show("initial softmax", candidates)

    # Step 2: Temperature
    candidates = apply_temperature(candidates, temp=0.7)
    show("temperature=0.7", candidates)

    # Step 3: Top-K
    candidates = apply_top_k(candidates, k=8)
    show("top_k=8", candidates)

    # Step 4: Top-P
    candidates = apply_top_p(candidates, p=0.90)
    show("top_p=0.90", candidates)

    # Step 5: Min-P
    candidates = apply_min_p(candidates, min_p=0.05)
    show("min_p=0.05", candidates)

    # Final distribution
    print()
    print("  Final distribution:")
    for c in sorted(candidates, key=lambda c: c.p, reverse=True):
        bar = "█" * int(c.p * 40)
        print(f"    {vocab[c.id]:10s}  {c.p:.4f}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GRAMMAR CONSTRAINT SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_grammar_constrained_decoding():
    """
    Show how grammar masking reduces the valid token set at each step.
    Uses a simplified JSON grammar simulation.
    """
    print("=" * 68)
    print("Grammar-Constrained Decoding Simulation — JSON Grammar")
    print("=" * 68)

    # Simulate JSON grammar states
    states = [
        {
            "description": 'Start of object: only "{" allowed',
            "position":    'Before any output',
            "vocab_size":  128000,
            "valid_tokens": 1,
            "valid_examples": ['"{" only'],
        },
        {
            "description": 'After "{": only string key (") or "}" allowed',
            "position":    'After "{"',
            "vocab_size":  128000,
            "valid_tokens": 2,
            "valid_examples": ['"\\""', '"}"'],
        },
        {
            "description": 'Inside key string: any non-quote char + escape',
            "position":    'After opening "\\"" of key',
            "vocab_size":  128000,
            "valid_tokens": 127800,
            "valid_examples": ['alpha/num tokens'],
        },
        {
            "description": 'After key: only ":" allowed',
            "position":    'After closing "\\"" of key',
            "vocab_size":  128000,
            "valid_tokens": 1,
            "valid_examples": ['":"'],
        },
        {
            "description": 'After ":": any JSON value start',
            "position":    'Before value',
            "vocab_size":  128000,
            "valid_tokens": 5,
            "valid_examples": ['"\\"", "[", "{", "true", "false", "null", digit'],
        },
        {
            "description": 'After string value: "," or "}"',
            "position":    'After value',
            "vocab_size":  128000,
            "valid_tokens": 2,
            "valid_examples": ['","', '"}"'],
        },
    ]

    print(f"  {'Position':35}  {'Valid':8}  {'% vocab':8}  Example tokens")
    print("  " + "-" * 75)
    for s in states:
        pct = s["valid_tokens"] / s["vocab_size"] * 100
        examples = ", ".join(s["valid_examples"])
        print(f"  {s['description'][:35]:35}  {s['valid_tokens']:8,}  {pct:8.4f}%  {examples}")

    print()
    print("  Observation: grammar masking reduces valid tokens by 4-6 orders of")
    print("  magnitude for structural positions, while leaving string content nearly")
    print("  unconstrained. The overhead is ~5-15% latency for mask application,")
    print("  but eliminates all retries from malformed JSON output.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  KV CACHE SEQUENCE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KVCacheSlot:
    seq_id: int
    positions: List[int] = field(default_factory=list)
    state: str = "idle"  # idle | prefill | decode | fork


class MockKVCache:
    """Demonstrates KV cache sequence management patterns."""

    def __init__(self, n_ctx: int, n_slots: int):
        self.n_ctx   = n_ctx
        self.n_slots = n_slots
        self.slots   = [KVCacheSlot(seq_id=i) for i in range(n_slots)]

    def assign_slot(self, seq_id: int, prompt_len: int) -> Optional[int]:
        for i, slot in enumerate(self.slots):
            if slot.state == "idle":
                slot.seq_id    = seq_id
                slot.positions = list(range(prompt_len))
                slot.state     = "prefill"
                return i
        return None

    def decode_step(self, slot_idx: int) -> int:
        slot = self.slots[slot_idx]
        new_pos = len(slot.positions)
        slot.positions.append(new_pos)
        slot.state = "decode"
        return new_pos

    def fork(self, src_idx: int, new_seq_id: int) -> Optional[int]:
        """Copy slot state for beam search branching."""
        dst_idx = self.assign_slot(new_seq_id, 0)
        if dst_idx is None:
            return None
        src = self.slots[src_idx]
        dst = self.slots[dst_idx]
        dst.positions = src.positions.copy()
        dst.state     = "fork"
        return dst_idx

    def evict(self, slot_idx: int, evict_before_pos: int):
        """Sliding window: evict positions 0..evict_before_pos."""
        slot = self.slots[slot_idx]
        slot.positions = [p for p in slot.positions if p >= evict_before_pos]

    def free(self, slot_idx: int):
        slot = self.slots[slot_idx]
        slot.positions = []
        slot.state     = "idle"

    def status(self):
        total = sum(len(s.positions) for s in self.slots)
        print(f"  KV Cache: {total}/{self.n_ctx} positions used  "
              f"({total/self.n_ctx*100:.1f}%)")
        for i, s in enumerate(self.slots):
            bar = "█" * (len(s.positions) * 20 // max(1, self.n_ctx))
            print(f"    Slot {i}  seq={s.seq_id:3d}  state={s.state:8s}  "
                  f"tokens={len(s.positions):5d}  {bar}")


def demo_kv_cache_management():
    print("=" * 60)
    print("KV Cache Sequence Management Patterns")
    print("=" * 60)

    cache = MockKVCache(n_ctx=4096, n_slots=4)

    # Pattern 1: Normal inference
    print("\n  [Pattern 1: Normal multi-user inference]")
    s0 = cache.assign_slot(seq_id=100, prompt_len=512)
    s1 = cache.assign_slot(seq_id=101, prompt_len=256)
    for _ in range(32):
        cache.decode_step(s0)
    for _ in range(64):
        cache.decode_step(s1)
    cache.status()

    # Pattern 2: Fork for beam search
    print("\n  [Pattern 2: Fork for beam search]")
    s2 = cache.fork(src_idx=s0, new_seq_id=102)
    cache.decode_step(s0)   # beam 1
    cache.decode_step(s2)   # beam 2 (forked)
    cache.status()

    # Pattern 3: Sliding window eviction
    print("\n  [Pattern 3: Sliding window eviction (keep last 256)]")
    s3 = cache.assign_slot(seq_id=103, prompt_len=800)
    cache.evict(s3, evict_before_pos=800-256)
    cache.status()

    # Pattern 4: Free completed sequence
    print("\n  [Pattern 4: Free completed sequence]")
    cache.free(s1)
    cache.status()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PERFORMANCE TUNING GUIDE
# ─────────────────────────────────────────────────────────────────────────────

def demo_performance_tuning():
    print("=" * 68)
    print("llama.cpp Performance Tuning Guide")
    print("=" * 68)

    scenarios = [
        {
            "name": "H100 80GB — Llama-3.1-70B Q4_K_M, high throughput",
            "flags": [
                ("-m",    "Llama-3.1-70B-Q4_K_M.gguf",   "4-bit quantized, ~38 GB"),
                ("-ngl",  "99",                            "All layers on GPU"),
                ("-c",    "16384",                         "4 slots × 4096"),
                ("-np",   "4",                             "4 parallel slots"),
                ("-b",    "2048",                          "Large prefill batch"),
                ("-ub",   "512",                           "Micro-batch for decode"),
                ("--flash-attn", "",                       "Required for this ctx"),
                ("--rope-freq-base", "500000",             "Llama 3.1 extended"),
            ],
        },
        {
            "name": "RTX 4090 24GB — Llama-3.1-8B Q4_K_M, low latency",
            "flags": [
                ("-m",    "Llama-3.1-8B-Q4_K_M.gguf",    "4-bit, ~4.8 GB"),
                ("-ngl",  "99",                            "All on GPU"),
                ("-c",    "8192",                          "Single context"),
                ("-np",   "1",                             "Single slot = min latency"),
                ("-b",    "512",                           "Moderate batch"),
                ("--flash-attn", "",                       "Faster attention"),
                ("--rope-freq-base", "500000",             "Llama 3.1"),
            ],
        },
        {
            "name": "M2 Ultra 192GB — Llama-3.1-70B Q6_K, quality focus",
            "flags": [
                ("-m",    "Llama-3.1-70B-Q6_K.gguf",     "6-bit, ~57 GB"),
                ("-ngl",  "99",                            "All on Apple GPU"),
                ("-c",    "32768",                         "Long context fits"),
                ("-np",   "2",                             "2 slots"),
                ("-b",    "512",                           "Metal batch size"),
                ("--flash-attn", "",                       "Metal FlashAttention"),
                ("--rope-freq-base", "500000",             "Llama 3.1"),
            ],
        },
        {
            "name": "CPU-only — Llama-3.1-8B Q4_K_M, 32-core server",
            "flags": [
                ("-m",    "Llama-3.1-8B-Q4_K_M.gguf",    "4-bit, ~4.8 GB"),
                ("-ngl",  "0",                             "No GPU offload"),
                ("-c",    "4096",                          "Modest context"),
                ("-np",   "1",                             "Single slot"),
                ("--threads", "16",                        "Half physical cores"),
                ("--threads-batch", "32",                  "All cores for prefill"),
                ("-b",    "512",                           "Batch for AVX prefill"),
            ],
        },
    ]

    for s in scenarios:
        print(f"\n  {s['name']}")
        print("  " + "-" * 60)
        cmd_parts = ["llama-server"]
        for flag, value, comment in s["flags"]:
            if value:
                cmd_parts.append(f"{flag} {value}")
                print(f"    {flag:22s} {value:20s}  # {comment}")
            else:
                cmd_parts.append(flag)
                print(f"    {flag:22s} {'':20s}  # {comment}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Chapter 28 — llama.cpp as a Platform (Python)")
    print("=" * 70 + "\n")

    analyze_gguf_size(8.0)    # 8B model
    analyze_gguf_size(70.0)   # 70B model
    demo_gguf_structure()
    demo_sampler_chain()
    simulate_grammar_constrained_decoding()
    demo_kv_cache_management()
    demo_performance_tuning()
```

---

## 28.7  C++ Implementation: Platform Internals

```cpp
// llamacpp_platform_demo.cpp
// Chapter 28 — llama.cpp as a Platform (C++ demo)
//
// Demonstrates (without requiring an actual llama.cpp installation):
//   1. GGUF header parsing (binary format recreation)
//   2. Quantization block structures (Q4_K_M layout)
//   3. Sampler chain implementation (temperature + top-k + top-p + min-p)
//   4. KV cache sequence management (seq_cp, seq_rm, seq_shift)
//   5. Token trie for grammar validation (simplified GBNF)
//   6. mmap vs malloc loading comparison
//
// Build: g++ -O2 -std=c++17 -o llamacpp_platform_demo llamacpp_platform_demo.cpp
// Run:   ./llamacpp_platform_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

static void print_section(const std::string& title) {
    std::string bar(72, '-');
    std::cout << "\n" << bar << "\n  " << title << "\n" << bar << "\n";
}

static std::string comma(long long n) {
    if (n < 0) return "-" + comma(-n);
    if (n < 1000) return std::to_string(n);
    return comma(n / 1000) + "," + [](long long r) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%03lld", r);
        return std::string(buf);
    }(n % 1000);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1.  GGUF BINARY FORMAT STRUCTURES
// ─────────────────────────────────────────────────────────────────────────────

// GGUF quantization type → bits per weight
static const std::pair<int, std::pair<const char*, double>> QUANT_TABLE[] = {
    { 0,  {"F32",    32.0}},
    { 1,  {"F16",    16.0}},
    { 2,  {"Q4_0",    4.5}},
    {10,  {"Q4_K_S",  4.5}},
    {11,  {"Q4_K_M",  4.84}},
    {12,  {"Q5_K_S",  5.5}},
    {13,  {"Q5_K_M",  5.68}},
    {14,  {"Q6_K",    6.56}},
    {15,  {"Q8_0",    8.5}},
    {30,  {"IQ4_NL",  4.5}},
    {34,  {"BF16",   16.0}},
};
static const int N_QUANT = sizeof(QUANT_TABLE) / sizeof(QUANT_TABLE[0]);

static const char* quant_name(int qtype) {
    for (int i = 0; i < N_QUANT; ++i)
        if (QUANT_TABLE[i].first == qtype)
            return QUANT_TABLE[i].second.first;
    return "UNKNOWN";
}

static double quant_bpw(int qtype) {
    for (int i = 0; i < N_QUANT; ++i)
        if (QUANT_TABLE[i].first == qtype)
            return QUANT_TABLE[i].second.second;
    return 16.0;
}

// Q4_K block structure (256 weights per super-block)
// Layout: 8 sub-blocks of 32 weights each
// Scales: 8 × int8 (one per sub-block)
// Minimums: 8 × int8
// Data: 256 × 4-bit packed (128 bytes)
struct Q4KBlock {
    uint8_t  scales[8];   // dequant scales, one per 32-weight sub-block
    uint8_t  mins[8];     // dequant minimums
    uint8_t  qs[128];     // 256 weights packed as 4-bit pairs
                           // qs[i] = (w[2i] & 0xF) | (w[2i+1] << 4)
};

static_assert(sizeof(Q4KBlock) == 144, "Q4KBlock must be 144 bytes");

static void demo_gguf_format() {
    print_section("GGUF Format: Size Analysis and Quantization Structures");

    // Model size comparison
    const double PARAMS_8B  = 8.0e9;
    const double PARAMS_70B = 70.0e9;

    std::cout << "\n  Size estimates for Llama-3.1-8B:\n";
    std::cout << "  " << std::string(55, '-') << "\n";
    std::cout << std::left  << std::setw(12) << "  Format"
              << std::right << std::setw(10) << "Bits/W"
              << std::setw(12) << "Size (GB)"
              << std::setw(10) << "vs FP16"
              << "\n";

    double fp16_8b = PARAMS_8B * 2 / 1e9;
    for (int i = 0; i < N_QUANT; ++i) {
        const char* name = QUANT_TABLE[i].second.first;
        double bpw  = QUANT_TABLE[i].second.second;
        double size = PARAMS_8B * bpw / 16.0 / 1e9;  // relative to 2 bytes/param
        double ratio = fp16_8b / size;
        std::string note = (std::string(name) == "Q4_K_M") ? " ← recommended" : "";
        std::cout << "  " << std::left  << std::setw(12) << name
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << bpw
                  << std::setw(12) << std::setprecision(2) << size
                  << std::setw(10) << std::setprecision(1) << ratio << "×"
                  << note << "\n";
    }

    // Q4_K block layout
    std::cout << "\n  Q4_K_M block layout (256 weights = 1 super-block):\n";
    std::cout << "    sizeof(Q4KBlock) = " << sizeof(Q4KBlock) << " bytes\n";
    std::cout << "    Layout: 8 scales (8B) + 8 mins (8B) + 128B data = 144B\n";
    std::cout << "    Effective: 256 weights in 144B = 4.5 bits/weight\n";
    std::cout << "    (Q4_K_M uses mixed precision: some sub-blocks use Q6 scales)\n";

    // Memory layout of Q4K data
    Q4KBlock demo_block;
    std::memset(&demo_block, 0, sizeof(demo_block));
    // Pack two 4-bit values into one byte
    int w0 = 7, w1 = 3;  // example weights
    demo_block.qs[0] = (uint8_t)((w0 & 0xF) | (w1 << 4));
    std::cout << "\n  Example packing: w0=" << w0 << " w1=" << w1
              << " → byte=" << (int)demo_block.qs[0]
              << " (unpack: lo=" << (demo_block.qs[0] & 0xF)
              << " hi=" << (demo_block.qs[0] >> 4) << ")\n";

    assert((demo_block.qs[0] & 0xF) == w0);
    assert((demo_block.qs[0] >> 4)  == w1);
    std::cout << "  [ASSERT] 4-bit packing/unpacking correct ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  SAMPLER CHAIN
// ─────────────────────────────────────────────────────────────────────────────

struct TokenProb {
    int   id;
    float logit;
    float p;
};

static void softmax_inplace(std::vector<TokenProb>& candidates) {
    float max_logit = candidates[0].logit;
    for (auto& c : candidates)
        if (c.logit > max_logit) max_logit = c.logit;

    float sum = 0.0f;
    for (auto& c : candidates) {
        c.p = std::exp(c.logit - max_logit);
        sum += c.p;
    }
    for (auto& c : candidates) c.p /= sum;
}

static void apply_temperature(std::vector<TokenProb>& cands, float temp) {
    if (temp == 1.0f) return;
    for (auto& c : cands) c.logit /= temp;
    softmax_inplace(cands);
}

static void apply_top_k(std::vector<TokenProb>& cands, int k) {
    if (k <= 0 || k >= (int)cands.size()) return;
    std::partial_sort(cands.begin(), cands.begin() + k, cands.end(),
        [](const TokenProb& a, const TokenProb& b) { return a.p > b.p; });
    cands.resize(k);
    float sum = 0;
    for (auto& c : cands) sum += c.p;
    for (auto& c : cands) c.p /= sum;
}

static void apply_top_p(std::vector<TokenProb>& cands, float p) {
    std::sort(cands.begin(), cands.end(),
        [](const TokenProb& a, const TokenProb& b) { return a.p > b.p; });
    float cumsum = 0;
    size_t keep = cands.size();
    for (size_t i = 0; i < cands.size(); ++i) {
        cumsum += cands[i].p;
        if (cumsum >= p && i + 1 >= 1) {
            keep = i + 1;
            break;
        }
    }
    cands.resize(keep);
    float sum = 0;
    for (auto& c : cands) sum += c.p;
    for (auto& c : cands) c.p /= sum;
}

static void apply_min_p(std::vector<TokenProb>& cands, float min_p) {
    float max_p = 0;
    for (auto& c : cands) if (c.p > max_p) max_p = c.p;
    float threshold = min_p * max_p;
    cands.erase(std::remove_if(cands.begin(), cands.end(),
        [threshold](const TokenProb& c) { return c.p < threshold; }),
        cands.end());
    if (cands.empty()) return;
    float sum = 0;
    for (auto& c : cands) sum += c.p;
    for (auto& c : cands) c.p /= sum;
}

static int sample_from(const std::vector<TokenProb>& cands, std::mt19937& rng) {
    std::vector<float> probs;
    for (auto& c : cands) probs.push_back(c.p);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return cands[dist(rng)].id;
}

static void demo_sampler_chain() {
    print_section("Sampler Chain: temperature + top_k + top_p + min_p");

    const int V = 32;
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 2.0f);

    std::vector<TokenProb> base_cands(V);
    for (int i = 0; i < V; ++i) base_cands[i] = {i, gauss(rng), 0.0f};
    softmax_inplace(base_cands);

    auto show = [](const std::string& stage, const std::vector<TokenProb>& cands) {
        auto top = cands;
        std::sort(top.begin(), top.end(),
            [](const TokenProb& a, const TokenProb& b) { return a.p > b.p; });
        float entropy = 0;
        for (auto& c : cands)
            if (c.p > 0) entropy -= c.p * std::log2(c.p);
        std::cout << "  " << std::left << std::setw(26) << stage
                  << std::right << std::setw(6) << cands.size() << " tokens"
                  << "  top=" << std::fixed << std::setprecision(3) << top[0].p
                  << "/" << top[1].p
                  << "  H=" << std::setprecision(2) << entropy << " bits\n";
    };

    // Run through the sampler chain
    auto cands = base_cands;
    show("Initial (softmax)", cands);

    apply_temperature(cands, 0.7f);
    show("temp=0.7", cands);

    apply_top_k(cands, 16);
    show("top_k=16", cands);

    apply_top_p(cands, 0.90f);
    show("top_p=0.90", cands);

    apply_min_p(cands, 0.05f);
    show("min_p=0.05", cands);

    // Show final distribution
    std::cout << "\n  Final candidate distribution:\n";
    auto final_sorted = cands;
    std::sort(final_sorted.begin(), final_sorted.end(),
        [](const TokenProb& a, const TokenProb& b) { return a.p > b.p; });
    for (auto& c : final_sorted) {
        int bar = static_cast<int>(c.p * 50);
        std::cout << "    tok_" << std::left << std::setw(4) << c.id
                  << "  " << std::fixed << std::setprecision(4) << c.p
                  << "  " << std::string(bar, '#') << "\n";
    }

    // Monte Carlo sampling: sample 10000 times, verify top token
    std::map<int, int> counts;
    for (int trial = 0; trial < 10000; ++trial)
        counts[sample_from(cands, rng)]++;

    int top_id    = final_sorted[0].id;
    float emp_p   = counts[top_id] / 10000.0f;
    float theory_p = final_sorted[0].p;
    std::cout << "\n  [ASSERT] top token empirical p ≈ theory p: "
              << std::setprecision(3) << emp_p << " vs " << theory_p << " ";
    bool ok = std::abs(emp_p - theory_p) < 0.03f;
    std::cout << (ok ? "✓" : "WARN") << "\n";
    // Soft check only (10K samples has ~1% noise)
    if (!ok)
        std::cerr << "  [WARN] deviation " << std::abs(emp_p - theory_p)
                  << " exceeds 0.03 — increase trials for tighter check\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  KV CACHE SEQUENCE MANAGEMENT
// ─────────────────────────────────────────────────────────────────────────────

struct KVCell {
    int  seq_id = -1;   // -1 = empty
    int  pos    = -1;
};

class MockKVCache {
public:
    std::vector<KVCell> cells;
    int n_ctx;

    explicit MockKVCache(int n_ctx) : n_ctx(n_ctx), cells(n_ctx) {}

    // Assign positions [0, len) to seq_id
    bool assign(int seq_id, int len) {
        int free_start = -1, free_count = 0;
        for (int i = 0; i < n_ctx; ++i) {
            if (cells[i].seq_id == -1) {
                if (free_count == 0) free_start = i;
                if (++free_count == len) break;
            } else {
                free_count = 0; free_start = -1;
            }
        }
        if (free_count < len) return false;
        for (int i = free_start; i < free_start + len; ++i) {
            cells[i] = {seq_id, i - free_start};
        }
        return true;
    }

    // Append one token at the end of seq_id's range
    bool append(int seq_id) {
        int cur_len = 0;
        for (auto& c : cells)
            if (c.seq_id == seq_id) cur_len++;
        for (int i = 0; i < n_ctx; ++i) {
            if (cells[i].seq_id == -1) {
                cells[i] = {seq_id, cur_len};
                return true;
            }
        }
        return false;
    }

    // seq_rm: remove positions [p0, p1) for seq_id
    void seq_rm(int seq_id, int p0, int p1) {
        for (auto& c : cells) {
            if (c.seq_id == seq_id && c.pos >= p0 && c.pos < p1) {
                c = {-1, -1};
            }
        }
    }

    // seq_cp: copy seq_id src → dst for positions [p0, p1)
    bool seq_cp(int src, int dst, int p0, int p1) {
        // Count cells to copy
        std::vector<int> src_cells;
        for (int i = 0; i < n_ctx; ++i)
            if (cells[i].seq_id == src && cells[i].pos >= p0 && cells[i].pos < p1)
                src_cells.push_back(i);

        // Find free cells for dst
        std::vector<int> free_cells;
        for (int i = 0; i < n_ctx; ++i)
            if (cells[i].seq_id == -1)
                free_cells.push_back(i);

        if (free_cells.size() < src_cells.size()) return false;

        for (size_t i = 0; i < src_cells.size(); ++i) {
            cells[free_cells[i]] = {dst, cells[src_cells[i]].pos};
        }
        return true;
    }

    // seq_shift: shift positions for seq_id by delta
    void seq_shift(int seq_id, int p0, int p1, int delta) {
        for (auto& c : cells) {
            if (c.seq_id == seq_id && c.pos >= p0 && c.pos < p1) {
                c.pos += delta;
                if (c.pos < 0) c = {-1, -1};  // shifted out
            }
        }
    }

    int used() const {
        int n = 0;
        for (auto& c : cells) if (c.seq_id >= 0) n++;
        return n;
    }

    void status(const std::string& label) const {
        std::cout << "\n  " << label << "\n";
        std::map<int, int> counts;
        for (auto& c : cells)
            if (c.seq_id >= 0)
                counts[c.seq_id]++;
        for (auto& [sid, cnt] : counts) {
            int bar = cnt * 40 / n_ctx;
            std::cout << "    seq=" << std::setw(4) << sid
                      << "  tokens=" << std::setw(5) << cnt
                      << "  " << std::string(std::max(0, bar), '|') << "\n";
        }
        std::cout << "    Total: " << used() << "/" << n_ctx << " cells used\n";
    }
};

static void demo_kv_management() {
    print_section("KV Cache Sequence Management (seq_rm / seq_cp / seq_shift)");

    MockKVCache cache(2048);

    // Prefill two sequences
    cache.assign(100, 512);
    cache.assign(101, 256);
    for (int i = 0; i < 64; ++i) cache.append(100);
    for (int i = 0; i < 128; ++i) cache.append(101);
    cache.status("After prefill + decode");

    // Fork seq 100 for beam search
    bool forked = cache.seq_cp(/*src*/100, /*dst*/102, /*p0*/0, /*p1*/576);
    std::cout << "  seq_cp(100 → 102): " << (forked ? "✓" : "FAIL (no space)") << "\n";
    cache.append(100);  // beam 1 diverges
    cache.append(102);  // beam 2 diverges
    cache.status("After beam search fork");

    // Sliding window: keep only last 256 tokens for seq 100
    // Remove positions 0..319 (576 - 256 = 320 → remove [0, 320))
    cache.seq_rm(100, 0, 320);
    cache.seq_shift(100, 320, 577, -320);  // shift remaining positions down
    cache.status("After sliding window eviction (keep last 256)");

    // Free completed sequence
    cache.seq_rm(101, 0, 2048);
    cache.status("After freeing seq 101");

    // Assert: cache has correct number of tokens for seq 100
    int count_100 = 0;
    for (auto& c : cache.cells)
        if (c.seq_id == 100) count_100++;
    // seq 100: 256 (kept) + 1 (new decode after eviction)
    std::cout << "\n  [NOTE] seq 100 has " << count_100 << " tokens after eviction\n";
    assert(count_100 > 0);
    std::cout << "  [ASSERT] seq 100 non-empty after eviction ✓\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  GRAMMAR CONSTRAINT SIMULATION
// ─────────────────────────────────────────────────────────────────────────────

// A minimal finite automaton to validate JSON character-by-character
// States: 0=start, 1=in_object, 2=in_string_key, 3=after_colon,
//         4=in_string_value, 5=after_value
enum class JSONState {
    START, IN_OBJECT, IN_KEY, AFTER_KEY,
    AFTER_COLON, IN_VALUE_STR, AFTER_VALUE, DONE, ERROR
};

static JSONState json_transition(JSONState s, char c) {
    switch (s) {
        case JSONState::START:
            if (c == '{') return JSONState::IN_OBJECT;
            return JSONState::ERROR;
        case JSONState::IN_OBJECT:
            if (c == '"') return JSONState::IN_KEY;
            if (c == '}') return JSONState::DONE;
            if (c == ' ' || c == '\n') return s;
            return JSONState::ERROR;
        case JSONState::IN_KEY:
            if (c == '"') return JSONState::AFTER_KEY;
            return s;  // still in key
        case JSONState::AFTER_KEY:
            if (c == ':') return JSONState::AFTER_COLON;
            if (c == ' ') return s;
            return JSONState::ERROR;
        case JSONState::AFTER_COLON:
            if (c == '"') return JSONState::IN_VALUE_STR;
            if (c == ' ') return s;
            return JSONState::ERROR;
        case JSONState::IN_VALUE_STR:
            if (c == '"') return JSONState::AFTER_VALUE;
            return s;
        case JSONState::AFTER_VALUE:
            if (c == ',') return JSONState::IN_OBJECT;
            if (c == '}') return JSONState::DONE;
            if (c == ' ') return s;
            return JSONState::ERROR;
        default:
            return JSONState::ERROR;
    }
}

static bool validate_json_string_object(const std::string& s) {
    JSONState state = JSONState::START;
    for (char c : s) {
        state = json_transition(state, c);
        if (state == JSONState::ERROR) return false;
    }
    return state == JSONState::DONE;
}

static void demo_grammar() {
    print_section("Grammar-Constrained Decoding: JSON FSA Validation");

    std::vector<std::pair<std::string, bool>> test_cases = {
        {R"({"name": "Alice"})",         true},
        {R"({"name": "Bob", "city": "NYC"})", true},   // valid JSON with multiple keys
        {R"({name: "Alice"})",           false},
        {R"({"name" "Alice"})",          false},
        {R"({"k": "v"})",                true},
        {R"(not json)",                  false},
        {R"({})",                        true},
    };

    std::cout << "\n  JSON string-object FSA validation:\n";
    std::cout << "  " << std::string(60, '-') << "\n";

    int passed = 0;
    for (auto& [s, expected] : test_cases) {
        bool got = validate_json_string_object(s);
        bool ok  = (got == expected);
        if (ok) passed++;
        std::cout << "  " << (ok ? "✓" : "?")
                  << "  " << std::left << std::setw(38) << s.substr(0, 38)
                  << "  expected=" << (expected ? "valid  " : "invalid")
                  << "  got=" << (got ? "valid" : "invalid") << "\n";
    }
    std::cout << "\n  " << passed << "/" << test_cases.size()
              << " cases matched expected outcome\n";

    // Grammar masking table
    std::cout << "\n  Token validity by grammar state (fraction of 128K vocab):\n";
    struct StateRow { const char* state; int valid; };
    StateRow rows[] = {
        {"START (need '{')",         1},
        {"IN_OBJECT (need '\"' or '}')",   2},
        {"IN_KEY (any char)",       127800},
        {"AFTER_KEY (need ':')",     1},
        {"AFTER_COLON (need '\"')",  2},
        {"IN_VALUE (any char)",     127800},
        {"AFTER_VALUE (need ','/'}')", 2},
    };
    for (auto& r : rows) {
        double pct = r.valid / 128000.0 * 100;
        std::cout << "  " << std::left << std::setw(38) << r.state
                  << std::right << std::setw(8) << r.valid
                  << "  (" << std::fixed << std::setprecision(4) << pct << "%)\n";
    }
    std::cout << "\n  Structural tokens: ~0.001% vocab valid\n"
              << "  Content tokens:   ~99.8% vocab valid\n"
              << "  Average overhead per step: ~5-15% latency for mask application\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  mmap VS malloc LOADING MODEL
// ─────────────────────────────────────────────────────────────────────────────

static void demo_mmap_vs_malloc() {
    print_section("mmap vs malloc: Model Loading Trade-offs");

    const double MODEL_GB   = 4.84;  // Llama-3.1-8B Q4_K_M
    const double PAGE_SIZE  = 4096.0;  // bytes
    const double FIRST_ACCESS_PAGES = 50.0;  // pages touched on first forward pass

    struct Strategy {
        const char* name;
        double startup_time_s;  // time until first token
        double ram_initial_gb;  // RAM consumed at startup
        double ram_after_n_tokens;  // after 100 tokens
        const char* notes;
    };

    // Empirical estimates on a typical NVMe SSD system
    Strategy strats[] = {
        {
            "malloc + read",
            MODEL_GB / (3.5),         // 3.5 GB/s NVMe sequential read
            MODEL_GB,                  // entire model in RAM
            MODEL_GB,
            "Slow start, all pages hot immediately"
        },
        {
            "mmap (default)",
            0.05,                      // only header+metadata read at open
            0.05,                      // only header in RAM
            FIRST_ACCESS_PAGES * PAGE_SIZE / 1e9,  // only touched pages
            "Fast start, OS faults pages on demand"
        },
        {
            "mmap + mlock",
            MODEL_GB / 3.5 + 0.1,     // read + lock time
            MODEL_GB,                  // all pages locked in RAM
            MODEL_GB,
            "No swap possible; prevents latency spikes"
        },
    };

    std::cout << "\n  Model: Llama-3.1-8B Q4_K_M (" << MODEL_GB << " GB)\n\n";
    std::cout << "  " << std::left << std::setw(20) << "Strategy"
              << std::right << std::setw(16) << "Startup time"
              << std::setw(14) << "RAM at start"
              << "  Notes\n";
    std::cout << "  " << std::string(72, '-') << "\n";

    for (auto& s : strats) {
        std::cout << "  " << std::left << std::setw(20) << s.name
                  << std::right << std::setw(14) << std::fixed << std::setprecision(2)
                  << s.startup_time_s << "s"
                  << std::setw(12) << std::setprecision(2) << s.ram_initial_gb << " GB"
                  << "  " << s.notes << "\n";
    }

    std::cout << "\n  With mmap, two processes loading the same model share physical pages:\n"
              << "  Process A: maps 4.84 GB → 4.84 GB physical pages\n"
              << "  Process B: maps same file → 0 additional physical pages\n"
              << "  Combined RAM: 4.84 GB (not 9.68 GB)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n"
              << std::string(72, '=') << "\n"
              << "  Chapter 28 — llama.cpp as a Platform (C++)\n"
              << std::string(72, '=') << "\n";

    demo_gguf_format();
    demo_sampler_chain();
    demo_kv_management();
    demo_grammar();
    demo_mmap_vs_malloc();

    std::cout << "\n" << std::string(72, '=') << "\n"
              << "  All demos complete.\n"
              << std::string(72, '=') << "\n\n";
    return 0;
}
```

---

## 28.8  Embedding llama.cpp in Your Application

### 28.8.1  CMakeLists.txt Integration

```cmake
# CMakeLists.txt for an application embedding llama.cpp
cmake_minimum_required(VERSION 3.14)
project(my_llm_app CXX)

set(CMAKE_CXX_STANDARD 17)

# Fetch llama.cpp as a subdirectory
include(FetchContent)
FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp
    GIT_TAG        master
)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama)

add_executable(my_llm_app main.cpp)
target_link_libraries(my_llm_app PRIVATE llama)
target_include_directories(my_llm_app PRIVATE ${llama_SOURCE_DIR}/include)
```

### 28.8.2  Thread Safety

The llama.cpp API has well-defined thread safety rules:

- `llama_model` is **read-only after loading** — multiple threads can call model
  query functions concurrently.

- `llama_context` is **not thread-safe** — one thread at a time per context.
- Multiple `llama_context` objects sharing one `llama_model` are safe to use from
  different threads simultaneously.

The typical pattern for a multi-user server:
```cpp
// One model shared globally
llama_model* g_model = llama_load_model_from_file(path, mparams);

// Per-request context (create fresh, or reuse from a pool)
thread_local llama_context* tl_ctx = llama_new_context_with_model(g_model, cparams);
```

### 28.8.3  Python ctypes Integration

For Python applications that cannot install `llama-cpp-python`:

```python
# llamacpp_ctypes.py
"""
Minimal ctypes wrapper for llama.cpp — demonstrates the ABI surface.
Production code should use llama-cpp-python instead.
"""
import ctypes
import ctypes.util
import os

def load_llama_lib(lib_path: str | None = None) -> ctypes.CDLL:
    """Load libllama.so / llama.dll / libllama.dylib."""
    if lib_path is None:
        # Try common locations
        candidates = [
            "libllama.so", "libllama.dylib",
            os.path.join(os.path.dirname(__file__), "libllama.so"),
        ]
        for p in candidates:
            try:
                return ctypes.CDLL(p)
            except OSError:
                continue
        raise FileNotFoundError("libllama not found; build llama.cpp and set LD_LIBRARY_PATH")
    return ctypes.CDLL(lib_path)


def configure_api(lib: ctypes.CDLL):
    """Set arg/return types for key API functions."""
    lib.llama_backend_init.restype  = None
    lib.llama_backend_free.restype  = None

    lib.llama_load_model_from_file.restype  = ctypes.c_void_p
    lib.llama_load_model_from_file.argtypes = [ctypes.c_char_p, ctypes.c_void_p]

    lib.llama_free_model.restype  = None
    lib.llama_free_model.argtypes = [ctypes.c_void_p]

    lib.llama_new_context_with_model.restype  = ctypes.c_void_p
    lib.llama_new_context_with_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lib.llama_free.restype  = None
    lib.llama_free.argtypes = [ctypes.c_void_p]

    lib.llama_n_vocab.restype  = ctypes.c_int32
    lib.llama_n_vocab.argtypes = [ctypes.c_void_p]

    lib.llama_tokenize.restype  = ctypes.c_int32
    lib.llama_tokenize.argtypes = [
        ctypes.c_void_p,   # model
        ctypes.c_char_p,   # text
        ctypes.c_int32,    # text_len
        ctypes.POINTER(ctypes.c_int32),  # tokens
        ctypes.c_int32,    # n_tokens_max
        ctypes.c_bool,     # add_special
        ctypes.c_bool,     # parse_special
    ]
    return lib
```

---

## 28.9  Common Pitfalls

### 28.9.1  Missing `llama_backend_init()`

Every application embedding llama.cpp must call `llama_backend_init()` before any other
llama.cpp call.
It initializes GGML's thread pool, loads CUDA/Metal kernels, and registers signal handlers.
Omitting it produces subtle crashes or hanging threads, not a clean error.

### 28.9.2  Context Size vs. KV Cache Budget

`n_ctx` in `llama_context_params` specifies the **total token capacity** of the KV cache.
If you use multiple sequence IDs, each sequence shares this budget.
A context of `n_ctx=4096` with `-np 4` gives each slot ~1024 tokens, not 4096.

### 28.9.3  `logits` Array Lifetime

`llama_get_logits_ith(ctx, i)` returns a pointer into an internal buffer that is
**valid only until the next `llama_decode()` call**.
Always copy logit values into your own buffer before calling decode again.

### 28.9.4  `llama_batch_free` and `llama_batch_init` Ownership

`llama_batch_init(n_tokens, embd, n_seq_max)` allocates internal arrays.
You must call `llama_batch_free()` for every `llama_batch_init()` call.
Stack-allocated `llama_batch` structs do not require free, but then you manage the
`token`, `pos`, `n_seq_id`, `seq_id`, and `logits` arrays yourself.

### 28.9.5  Grammar Statefulness

The `llama_sampler` returned by `llama_sampler_init_grammar()` maintains internal parse
state that advances with every sampled token.
It cannot be shared across sequences.
Each independent generation that uses grammar must have its own grammar sampler instance.

---

## 28.10  Summary

llama.cpp is a complete inference platform with a stable, minimal C API that maps
cleanly onto the inference lifecycle: backend init → model load → context create →
tokenize → batch decode → sample → stream → free.

The GGUF format is self-describing, memory-mappable, and extensible — understanding its
structure makes quantization formats, model metadata, and tokenizer data transparent.

Grammar-constrained decoding via GBNF eliminates retry loops for structured output
with only 5–15% latency overhead.
The server exposes the full feature set via an OpenAI-compatible HTTP API with
per-slot monitoring.

For embedding: use `FetchContent` to pull llama.cpp into your CMake project,
create one `llama_model` shared across threads, and allocate per-thread contexts.
For Python: `llama-cpp-python` provides a high-level wrapper that handles the C ABI
complexity without sacrificing access to advanced features like grammar, LoRA, and
KV cache sequence manipulation.

---

*Chapter 29 extends llama.cpp to multimodal inputs: how vision encoders (LLaVA, BakLLaVA,
MiniCPM-V) are integrated into the GGUF format, and how images flow through the
projection layers into the language model's token space.*


---

## Chapter Summary

- **llama.cpp as a server**: `llama-server` exposes `/v1/chat/completions`, `/v1/embeddings`, and `/v1/completions` — the same OpenAI-compatible API as vLLM, enabling drop-in substitution.
- **Parallel slots**: `--parallel N` runs N simultaneous generation sequences using shared weights; the key limitation is that GGUF weights are loaded once and shared across slots.
- **Grammar-constrained generation**: `llama.cpp`'s built-in GBNF grammar engine applies CFG masks at each decode step; more expressive than regex but with similar implementation overhead.
- **Metal backend**: on macOS, `llama.cpp` uses Apple's Metal GPU API; unified memory means CPU and GPU share the same physical DRAM, eliminating the PCIe transfer bottleneck.
- **NUMA-aware threading**: `--main-gpu` and `--tensor-split` pin GPU layers; `--threads` maps to CPU cores with NUMA affinity; correct NUMA binding can improve CPU throughput 30–50%.
- **GGUF model format**: header (metadata), key-value store (hyperparams), tensor data (quantised blocks); `llama_load_model_from_file` mmap's the file for zero-copy loading.
- **Embedding server**: `--embedding` mode exposes `/v1/embeddings`; useful for RAG pipelines where the same model is used for both generation and retrieval.

---

## Self-Check Questions

1. `llama-server --parallel 8 --ctx-size 4096` is configured on a machine with 64 GB RAM and a 4-bit Q4_K_M LLaMA-3 8B model (≈ 4.5 GB). Each slot holds its own KV cache. Compute the total memory usage and verify it fits. *(Section 28.2)*

2. GBNF grammar for JSON requires a CFG that accepts only valid JSON. At a decode step where `"age":` has just been generated, list the valid next tokens according to the grammar and explain how the mask is applied to the logit vector. *(Section 28.4)*

3. On Apple M3 Max with 128 GB unified memory, llama.cpp can run a LLaMA-3 70B Q4_K_M model (≈ 40 GB). Why is this possible on the Mac but not on a discrete GPU with 40 GB VRAM? *(Section 28.5)*

4. `--tensor-split 3,1` on a 2-GPU node allocates 75% of layers to GPU 0 and 25% to GPU 1. If GPU 0 has 24 GB and GPU 1 has 8 GB, is this split optimal for a 32 GB model? Show your working. *(Section 28.3)*

5. llama.cpp's `/v1/embeddings` endpoint returns the mean-pooled hidden state at the last layer. For a RAG retrieval task, what are the trade-offs of using the same model for both generation and embedding versus a dedicated embedding model? *(Section 28.6)*


---

## Worked Solutions

### Question 1
**`llama-server --parallel 8 --ctx-size 4096` on 64 GB RAM, Q4_K_M LLaMA-3 8B (~4.5 GB).**

**Memory breakdown:**
Each parallel slot holds its own KV cache. KV cache per slot per token for LLaMA-3 8B (32 layers, 8 KV heads, d_k=128, BF16):
```
KV_per_token = 2 x 32 x 8 x 128 x 2 = 131,072 bytes = 128 KB
KV_per_slot = 4096 tokens x 128 KB = 512 MB
Total KV for 8 slots = 8 x 512 MB = 4,096 MB = 4 GB
```

**Total memory:**
```
Model weights (Q4_K_M): 4.5 GB
KV cache (8 slots):     4.0 GB
Runtime overhead:       ~0.5 GB (GGML tensors, buffers)
Total:                  ~9.0 GB
```

9.0 GB << 64 GB RAM -- **fits comfortably**. The machine has 55 GB of headroom remaining for the OS and other processes.

**Note on Q4_K_M:** The 4.5 GB figure already accounts for the quantization -- the original FP16 model would be 16 GB. Q4_K_M achieves ~3.5x compression while maintaining quality close to FP16 on most benchmarks.

---

### Question 2
**GBNF grammar for JSON at state "age":`  -- valid next tokens:**

After generating `"age":`, the JSON grammar expects one of:
```

- A number literal: digits [0-9], optionally preceded by - (negative)
- null
- whitespace followed by any of the above
```

So the valid next tokens are those whose text begins with:

- `0`, `1`, `2`, ... `9` (digits)
- `-` (negative number)
- `null` (literal)
- ` ` (whitespace, followed by number/null)

**How the mask is applied:**
1. The model produces a logit vector of shape (vocab_size,) -- one logit per token.
2. The GBNF FSM (in its current state after `"age":`) has a set of valid next bytes.
3. llama.cpp maps each vocabulary token to its byte sequence and checks if that byte sequence is consistent with the current FSM state.
4. Tokens inconsistent with the grammar get their logits set to -infinity before sampling.
5. Softmax is applied to the masked logits, normalizing only over valid tokens.

For a typical 32K vocabulary, roughly 100-500 tokens start with digits or null -- the mask sets ~31,500+ tokens to -infinity.

---

### Question 3
**Apple M3 Max 128 GB unified memory can run LLaMA-3 70B Q4_K_M (~40 GB). Why not on discrete 40 GB VRAM?**

**Discrete GPU (40 GB VRAM):**
GPU VRAM is a **fixed, isolated** memory pool. If the model requires 40 GB and the GPU has exactly 40 GB, there is zero space for:

- The KV cache (even 1 token of context needs space)
- CUDA context (~1 GB)
- Activation tensors during forward pass (~2-4 GB)
- Operating system reserved GPU memory (~0.5 GB)
Total required = 40 + 1 + 3 + 0.5 = 44.5 GB > 40 GB available --> OOM at startup.

**Apple Silicon (unified memory):**
The M3 Max uses a **unified memory architecture** where CPU and GPU share the same 128 GB physical memory pool. There is no separate VRAM limit. The GPU can address any part of the 128 GB:

- Model weights: 40 GB
- KV cache: 4 GB (for 32K context)
- System/OS: 8 GB
- Remaining: 76 GB headroom

llama.cpp exploits Metal (Apple's GPU API) to run the model on the integrated GPU while the CPU and GPU share the same memory pool, eliminating data transfer bottlenecks between CPU and GPU memory.

---

### Question 4
**`--tensor-split 3,1` on 2-GPU node. GPU 0: 24 GB, GPU 1: 8 GB. Model: 32 GB. Is split optimal?**

**What `--tensor-split 3,1` means:**
75% of layers on GPU 0, 25% on GPU 1.
```
GPU 0 load = 0.75 x 32 GB = 24 GB
GPU 1 load = 0.25 x 32 GB = 8 GB
```

**Is it optimal?**
GPU 0 has 24 GB and is assigned exactly 24 GB of model weights -- **zero headroom** for:

- KV cache (critical for inference)
- Activation tensors during forward pass (~2-4 GB peak)
- CUDA/Metal context

GPU 0 will OOM at the first inference step with any non-trivial context length.

**Optimal split:**
Leave headroom on each GPU for KV cache and activations. Rule of thumb: use 85% of VRAM for model weights.
```
GPU 0 capacity for weights: 24 x 0.85 = 20.4 GB -> fraction = 20.4 / (20.4 + 6.8) = 75%
GPU 1 capacity for weights:  8 x 0.85 =  6.8 GB -> fraction = 6.8 / 27.2 = 25%
```
Interestingly, the 3:1 split is proportionally correct for the GPU memory sizes. The problem is not the ratio but the **zero headroom** when the model exactly fills both GPUs.

**Fix:** Use a slightly smaller model (Q3_K_M at ~27 GB) or the adjusted split `--tensor-split 2.7,0.9` (27 GB total, leaving 3.3 GB headroom on GPU 0 and 1.1 GB on GPU 1 for KV cache).

---

### Question 5
**llama.cpp `/v1/embeddings` returns mean-pooled last-layer hidden state. RAG trade-offs vs dedicated embedding model:**

**Using same model for generation and embedding:**

Pros:

- Single model in memory (no additional 500 MB-4 GB for a dedicated embedder)
- Consistent tokenization (same vocabulary for both retrieval and generation)
- Simple deployment (one binary, one process)

Cons:

- Decoder-only LLMs are not optimized for embedding. Their last-layer representations encode "what token comes next" -- a causal prediction objective -- not "how semantically similar are these passages" -- a contrastive objective. Embedding quality for retrieval is significantly worse than dedicated bi-encoder models (BGE, E5, Nomic-embed).
- Mean pooling of causal LLM states gives asymmetric embeddings: the last token has attended to all prior tokens, while the first token has attended only to itself. This asymmetry makes cosine similarity unreliable.
- Embedding via a 70B LLM is 100-1000x slower than a dedicated 137M-560M parameter embedding model.

**When to use the dedicated model:**
Virtually always for production RAG. A dedicated embedding model (BGE-M3, E5-mistral-7b) outperforms LLM mean-pooling on MTEB benchmarks by 5-15 nDCG@10 points. The speed difference (1,000 embeddings/s on a GPU for BGE-base vs ~10 embeddings/s for LLaMA-3 70B) makes dedicated models the correct choice for any non-trivial retrieval workload.

