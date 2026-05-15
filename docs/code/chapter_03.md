# Chapter 3: Tokens, Sequences, and the Batch — Companion Code

## Python — `tokenization_demo.py`

```python
# tokenization_demo.py
# Chapter 3 — Tokens, Sequences, and the Batch
#
# Requirements:
#   pip install transformers tiktoken sentencepiece
#
# No GPU needed — tokenization runs entirely on CPU.
# First run will download tokenizer files from HuggingFace (~1 MB).
#
# Run:
#   python tokenization_demo.py

from transformers import AutoTokenizer
import json

# ── Section 1: Basic BPE tokenization ──────────────────────────────────────

# LLaMA 3 uses tiktoken BPE with vocab_size=128,256
# If you don't have access to the gated repo, substitute:
#   "mistralai/Mistral-7B-v0.1"  (SentencePiece, 32K vocab)
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Loading tokenizer: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print(f"Vocab size: {tokenizer.vocab_size}\n")


def show_tokens(text: str) -> None:
    """Display token IDs and their decoded pieces side by side."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = [tokenizer.decode([tid]) for tid in token_ids]
    print(f"Text:   {repr(text)}")
    print(f"IDs:    {token_ids}")
    print(f"Pieces: {pieces}")
    print(f"Count:  {len(token_ids)} tokens")
    print()


print("=== Section 1: Token Inspection ===\n")
show_tokens("Hello, world!")
show_tokens("lowest")        # Chapter 3 BPE worked example
show_tokens("newest")        # Chapter 3 BPE worked example
show_tokens("unaffable")     # rare word → subword fragments
show_tokens("LLM inference")
show_tokens("PagedAttention")
show_tokens("café")          # non-ASCII → byte fallback
show_tokens("가나다")        # Korean: multiple bytes per char → many tokens
show_tokens("1234567890")    # numbers: each digit often a separate token


# ── Section 2: Chat template ────────────────────────────────────────────────

print("=== Section 2: Chat Template ===\n")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is PagedAttention?"},
]

# Text form — shows the exact string fed to the model
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print("── Chat template (text) ──")
print(formatted_text)
print()

# Token ID form — what the model actually sees
formatted_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True
)
print(f"── Chat template (IDs): {len(formatted_ids)} tokens ──")
print(formatted_ids)
print()

# Decode each ID to inspect the structure
print("── Per-token breakdown (first 40 tokens) ──")
for i, tid in enumerate(formatted_ids[:40]):
    piece = tokenizer.decode([tid])
    print(f"  [{i:3d}] id={tid:7d}  piece={repr(piece)}")
print("  ...")
print()


# ── Section 3: Padding waste demonstration ──────────────────────────────────

print("=== Section 3: Padding Waste (Static Batching) ===\n")

requests = [
    "Explain the difference between prefill and decode in LLM inference.",
    "What is BPE?",
    (
        "Describe the KV cache memory formula for a 70-billion-parameter model "
        "with 80 transformer layers, 8 KV heads per layer, head dimension 128, "
        "and a sequence length of 4096 tokens using FP16 precision."
    ),
    "Hi",
]

all_ids = [tokenizer.encode(r, add_special_tokens=False) for r in requests]
lengths  = [len(ids) for ids in all_ids]
max_len  = max(lengths)

print(f"Sequence lengths: {lengths}")
print(f"Max length:       {max_len}")

total_real   = sum(lengths)
total_padded = max_len * len(requests)
waste_pct    = 100 * (total_padded - total_real) / total_padded

print(f"Real tokens:      {total_real}")
print(f"Padded tokens:    {total_padded}  (all padded to max_len)")
print(f"Wasted compute:   {total_padded - total_real} tokens = {waste_pct:.1f}%\n")

# Visual grid
print("  Batch grid  (■ = real token,  □ = padding)\n")
for i, (req, ids) in enumerate(zip(requests, all_ids)):
    real_bar    = "■" * len(ids)
    padding_bar = "□" * (max_len - len(ids))
    pct = 100 * len(ids) / max_len
    print(f"  Req {i+1}: {real_bar}{padding_bar}  ({len(ids):3d}/{max_len} = {pct:4.0f}% real)")

print()


# ── Section 4: Vocabulary and special tokens ────────────────────────────────

print("=== Section 4: Special Tokens ===\n")

print(f"Vocab size:    {tokenizer.vocab_size}")
print(f"BOS token:     {tokenizer.bos_token!r}  (ID {tokenizer.bos_token_id})")
print(f"EOS token:     {tokenizer.eos_token!r}  (ID {tokenizer.eos_token_id})")
print(f"PAD token:     {tokenizer.pad_token!r}  (ID {tokenizer.pad_token_id})")
print()

# All special tokens
print("All special tokens:")
for name, token in tokenizer.special_tokens_map.items():
    if isinstance(token, str):
        tid = tokenizer.convert_tokens_to_ids(token)
        print(f"  {name:30s}: {token!r:30s}  (ID {tid})")
    elif isinstance(token, list):
        for t in token:
            tid = tokenizer.convert_tokens_to_ids(t)
            print(f"  {name:30s}: {t!r:30s}  (ID {tid})")
print()


# ── Section 5: LLaMA 2 vs LLaMA 3 tokenizer comparison ─────────────────────

print("=== Section 5: Tokenizer Comparison (LLaMA 2 vs LLaMA 3) ===\n")
print("(Requires access to both model repos; skipping if unavailable)\n")

llama2_model = "meta-llama/Llama-2-7b-chat-hf"
test_strings = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "LLM inference optimization with PagedAttention and continuous batching.",
]

try:
    tok2 = AutoTokenizer.from_pretrained(llama2_model)
    tok3 = tokenizer  # already loaded above

    print(f"{'Text':<55}  {'LLaMA-2 toks':>12}  {'LLaMA-3 toks':>12}  {'Delta':>6}")
    print("-" * 95)
    for text in test_strings:
        n2 = len(tok2.encode(text, add_special_tokens=False))
        n3 = len(tok3.encode(text, add_special_tokens=False))
        delta = n3 - n2
        print(f"  {text[:52]:<52}  {n2:>12}  {n3:>12}  {delta:>+6}")
    print()
    print("Negative delta = LLaMA 3 uses fewer tokens for the same text.")
    print("Fewer tokens → shorter sequences → less KV cache, lower latency.")

except Exception as e:
    print(f"  Skipped comparison (model access error): {e}")

```

## C++ — `batch_demo.cpp`

```cpp
// batch_demo.cpp
// Chapter 3 — Tokens, Sequences, and the Batch
//
// Demonstrates:
//   1. Tokenizing multiple prompts and printing their pieces
//   2. Measuring static batching padding waste
//   3. Building a prefill batch per sequence (sequence-tagged KV positions)
//   4. One step of a continuous decode batch (all sequences in one llama_decode call)
//
// Build (from within the llama.cpp repo root after cmake build):
//   g++ -std=c++17 -O2 batch_demo.cpp \
//       -Iinclude -Lbuild/src -lllama -Lbuild/ggml/src -lggml \
//       -o batch_demo
//
// Run:
//   ./batch_demo /path/to/Llama-3.2-1B-Q4_K_M.gguf
//
// Model download:
//   huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
//       --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf" --local-dir .

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Tokenize a string into a vector<llama_token>
// add_bos=true prepends the BOS token (needed for the first turn).
// ---------------------------------------------------------------------------
static std::vector<llama_token> tokenize(
    const llama_model* model,
    const std::string& text,
    bool add_bos = true)
{
    // Upper-bound estimate: one token per ~2 bytes + safety margin
    std::vector<llama_token> tokens(text.size() / 2 + 64);

    int n = llama_tokenize(
        llama_model_get_vocab(model),
        text.c_str(), (int)text.size(),
        tokens.data(), (int)tokens.size(),
        add_bos,
        /*parse_special=*/true   // recognize <|begin_of_text|> etc.
    );

    if (n < 0) {
        // Buffer too small — retry with exact size
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
// Convert a single token ID to its text piece
// ---------------------------------------------------------------------------
static std::string token_to_piece(const llama_model* model, llama_token token) {
    char buf[256] = {};
    int n = llama_token_to_piece(
        llama_model_get_vocab(model),
        token, buf, sizeof(buf),
        /*lstrip=*/0, /*special=*/true
    );
    if (n < 0) return "<?>"; // shouldn't happen
    return std::string(buf, n);
}

// ---------------------------------------------------------------------------
// Print token IDs and their pieces for a sequence
// ---------------------------------------------------------------------------
static void print_token_sequence(
    const llama_model* model,
    const std::string& label,
    const std::vector<llama_token>& tokens)
{
    printf("%s  (%d tokens)\n", label.c_str(), (int)tokens.size());
    printf("  IDs   : [");
    for (int i = 0; i < (int)tokens.size(); i++)
        printf("%d%s", tokens[i], i+1 < (int)tokens.size() ? ", " : "");
    printf("]\n");
    printf("  Pieces: [");
    for (int i = 0; i < (int)tokens.size(); i++) {
        std::string p = token_to_piece(model, tokens[i]);
        printf("\"%s\"%s", p.c_str(), i+1 < (int)tokens.size() ? ", " : "");
    }
    printf("]\n\n");
}

// ---------------------------------------------------------------------------
// Print padding waste statistics for a set of sequences
// ---------------------------------------------------------------------------
static void print_padding_stats(const std::vector<std::vector<llama_token>>& seqs) {
    int max_len = 0;
    int total_real = 0;
    for (auto& s : seqs) {
        max_len = std::max(max_len, (int)s.size());
        total_real += (int)s.size();
    }
    int n_seqs      = (int)seqs.size();
    int total_pad   = max_len * n_seqs;
    int wasted      = total_pad - total_real;
    float waste_pct = 100.0f * wasted / total_pad;

    printf("Static batching analysis:\n");
    printf("  Sequence lengths : [");
    for (int i = 0; i < n_seqs; i++)
        printf("%d%s", (int)seqs[i].size(), i+1 < n_seqs ? ", " : "");
    printf("]\n");
    printf("  Max length       : %d\n", max_len);
    printf("  Total real toks  : %d\n", total_real);
    printf("  Total padded toks: %d (all seqs padded to max_len)\n", total_pad);
    printf("  Wasted compute   : %d tokens = %.1f%%\n\n", wasted, waste_pct);

    // Visual grid
    printf("  Batch grid  (# = real token,  . = padding)\n\n");
    for (int i = 0; i < n_seqs; i++) {
        printf("  Seq %d: ", i);
        for (int j = 0; j < max_len; j++)
            printf("%c", j < (int)seqs[i].size() ? '#' : '.');
        printf("  (%d/%d)\n", (int)seqs[i].size(), max_len);
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Prefill a single sequence into the KV cache under its sequence ID.
// Sets logits only for the last token (only that token needs sampling).
// Returns the number of prompt tokens (= position of next token to generate).
// ---------------------------------------------------------------------------
static int prefill_sequence(
    llama_context*                    ctx,
    const std::vector<llama_token>&   tokens,
    int                               seq_id)
{
    int n = (int)tokens.size();

    // llama_batch_init(max_n_tokens, embd_size, max_seq_ids_per_token)
    // embd_size=0 means we use the token[] array (not raw embeddings).
    // max_seq_ids_per_token=1 because each token belongs to exactly one sequence.
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;

    for (int i = 0; i < n; i++) {
        batch.token[i]      = tokens[i];
        batch.pos[i]        = i;            // sequential positions within this sequence
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = seq_id;       // tag every token with our sequence ID
        batch.logits[i]     = (i == n - 1) ? 1 : 0;  // logits only for last token
    }

    int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        fprintf(stderr, "  ERROR: prefill failed for seq_id=%d (ret=%d)\n", seq_id, ret);
        return -1;
    }
    return n;  // next position for this sequence
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
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // ── Load model ────────────────────────────────────────────────────────

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;   // CPU-only for this demo; set 99 for full GPU

    fprintf(stderr, "Loading model: %s\n", argv[1]);
    llama_model* model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return 1;
    }

    // ── Create context large enough for 4 concurrent sequences ────────────
    // n_ctx = 4 seqs × 256 positions each = 1024 total KV slots.
    // All sequences share this pool; llama.cpp tracks per-sequence positions
    // via the seq_id fields in llama_batch.

    const int N_SEQS    = 4;
    const int MAX_POS   = 256;  // max tokens per sequence

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = N_SEQS * MAX_POS;
    cparams.n_batch  = N_SEQS * MAX_POS;
    cparams.n_ubatch = 64;   // physical forward-pass chunk during batch prefill
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;

    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PART 1 — Tokenization demo
    // ═══════════════════════════════════════════════════════════════════════

    printf("\n═══════════════════════════════════════\n");
    printf("PART 1: Tokenization\n");
    printf("═══════════════════════════════════════\n\n");

    // Prompts of deliberately varying length to illustrate padding waste
    std::vector<std::string> prompts = {
        "Hello",
        "What is the KV cache?",
        "Explain BPE tokenization in one sentence.",
        "Hi"
    };

    std::vector<std::vector<llama_token>> all_tokens;
    for (int i = 0; i < (int)prompts.size(); i++) {
        auto toks = tokenize(model, prompts[i]);
        all_tokens.push_back(toks);
        print_token_sequence(model, "Prompt " + std::to_string(i), toks);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PART 2 — Padding waste
    // ═══════════════════════════════════════════════════════════════════════

    printf("═══════════════════════════════════════\n");
    printf("PART 2: Static Batching Padding Waste\n");
    printf("═══════════════════════════════════════\n\n");

    print_padding_stats(all_tokens);

    // ═══════════════════════════════════════════════════════════════════════
    // PART 3 — Prefill all sequences into the shared KV cache
    // Each sequence gets its own seq_id (0, 1, 2, 3).
    // KV entries are tagged so attention is masked per-sequence.
    // ═══════════════════════════════════════════════════════════════════════

    printf("═══════════════════════════════════════\n");
    printf("PART 3: Per-Sequence Prefill\n");
    printf("═══════════════════════════════════════\n\n");

    std::vector<int> positions(N_SEQS, 0);  // tracks current KV position per seq

    for (int seq = 0; seq < N_SEQS; seq++) {
        int n = prefill_sequence(ctx, all_tokens[seq], seq);
        if (n < 0) return 1;
        positions[seq] = n;
        printf("  Seq %d prefilled: %d tokens  (next pos = %d)\n", seq, n, n);
    }
    printf("\n");

    // ═══════════════════════════════════════════════════════════════════════
    // PART 4 — One step of continuous batching: sample + decode for all seqs
    // In continuous batching, each decode step submits ONE token per active
    // sequence in a single llama_decode() call.  The GPU processes all
    // active sequences together — no padding, no idle slots.
    // ═══════════════════════════════════════════════════════════════════════

    printf("═══════════════════════════════════════\n");
    printf("PART 4: One Continuous Decode Step\n");
    printf("═══════════════════════════════════════\n\n");

    // Build one sampler per sequence (each can have independent temperature/top-p)
    std::vector<llama_sampler*> samplers(N_SEQS);
    for (int i = 0; i < N_SEQS; i++) {
        samplers[i] = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(samplers[i], llama_sampler_init_greedy());
    }

    // After prefill, the context holds logits only for the LAST prefilled
    // sequence (seq 3) because we prefilled sequentially and each llama_decode()
    // call overwrites the logit buffer.
    //
    // In a production engine, all sequences would be prefilled in parallel
    // (or chunked together) so all logits are available simultaneously.
    // Here we demonstrate the API for seq 3 as a concrete example.

    llama_token first_tok = llama_sampler_sample(samplers[N_SEQS-1], ctx, -1);
    std::string first_piece = token_to_piece(model, first_tok);
    printf("  Seq 3 first generated token: id=%d  piece=\"%s\"\n\n",
           first_tok, first_piece.c_str());

    // Now build the continuous batch: one fresh token per sequence.
    // (Using first_tok for seq 3; placeholder token 1 for seqs 0-2 for demo)
    printf("  Building continuous decode batch (%d sequences × 1 token each):\n\n",
           N_SEQS);

    // In real usage, each seq has its own sampled next-token. We use tok=1 as
    // a placeholder for sequences 0-2 where we didn't save the logits.
    std::vector<llama_token> decode_tokens = {1, 1, 1, first_tok};

    llama_batch decode_batch = llama_batch_init(N_SEQS, 0, 1);
    decode_batch.n_tokens = N_SEQS;

    for (int i = 0; i < N_SEQS; i++) {
        decode_batch.token[i]     = decode_tokens[i];
        decode_batch.pos[i]       = positions[i];   // CRITICAL: continues from prefill end
        decode_batch.n_seq_id[i]  = 1;
        decode_batch.seq_id[i][0] = i;
        decode_batch.logits[i]    = 1;              // need logits for all (sampling each)

        printf("    Slot %d: token=%5d  pos=%3d  seq_id=%d\n",
               i, decode_tokens[i], positions[i], i);
    }
    printf("\n");

    // Single llama_decode() call processes all 4 sequences simultaneously.
    // This is the heart of continuous batching: no padding, no idle slots,
    // all sequences make progress in one GPU forward pass.
    int ret = llama_decode(ctx, decode_batch);
    if (ret == 0) {
        printf("  llama_decode() succeeded: 4 sequences processed in ONE forward pass.\n");
        printf("  Compare to static batching: 4 × 1 = 4 separate forward passes.\n");
    } else {
        fprintf(stderr, "  ERROR: decode batch failed (ret=%d)\n", ret);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────

    llama_batch_free(decode_batch);
    for (auto s : samplers) llama_sampler_free(s);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("\nDone.\n");
    return 0;
}

```

