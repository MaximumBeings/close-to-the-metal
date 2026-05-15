# Chapter 1: Two Engines, One Problem — Companion Code

## Python — `hello_vllm.py`

```python
# hello_vllm.py
# Chapter 1 — Close to the Metal: LLM Inference from First Principles
#
# Requirements:
#   pip install vllm
#
# Hardware:
#   Any CUDA GPU with >= 8 GB VRAM
#   For CPU-only testing, set enforce_eager=True and use a tiny model
#
# First run will download weights from HuggingFace (~2.5 GB for 1B model).

from vllm import LLM, SamplingParams
import time

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

def main():
    print(f"Loading model: {MODEL}")
    t0 = time.perf_counter()

    # LLM() drives the full startup sequence:
    #   1. Load weights from HuggingFace cache (or download on first run)
    #   2. Run a dummy forward pass to measure peak activation memory
    #   3. Compute remaining HBM: total - weights - activations
    #   4. Allocate KV cache block pool from remaining HBM
    #   5. Pre-capture CUDA graphs for batch sizes [1, 2, 4, 8, 16, 32]
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.85,  # reserve 85% of HBM for model + KV cache
        max_model_len=4096,           # maximum context window length
        enforce_eager=False,          # True disables CUDA graph capture (slower but easier debug)
    )

    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )

    prompts = [
        "Explain the difference between prefill and decode in LLM inference.",
        "What is PagedAttention and why does it matter?",
    ]

    print("Running inference...")
    t1 = time.perf_counter()

    # generate() is synchronous: it submits all prompts, runs the scheduler loop
    # until all requests complete, and returns a list of RequestOutput objects.
    outputs = llm.generate(prompts, params)

    gen_time = time.perf_counter() - t1

    total_output_tokens = 0
    for output in outputs:
        prompt_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        total_output_tokens += output_tokens

        print(f"--- Prompt ({prompt_tokens} tokens) ---")
        print(output.prompt)
        print(f"--- Response ({output_tokens} tokens) ---")
        print(output.outputs[0].text)
        print()

    print(f"Generation time:   {gen_time:.2f}s")
    print(f"Output tokens:     {total_output_tokens}")
    print(f"Throughput:        {total_output_tokens / gen_time:.1f} tokens/sec")


if __name__ == "__main__":
    main()

```

## C++ — `hello_llamacpp.cpp`

```cpp
// hello_llamacpp.cpp
// Chapter 1 — Close to the Metal: LLM Inference from First Principles
//
// Build (CPU only):
//   git clone https://github.com/ggerganov/llama.cpp
//   cd llama.cpp
//   cmake -B build -DGGML_CUDA=OFF
//   cmake --build build --config Release -j$(nproc)
//   g++ -std=c++17 -O2 hello_llamacpp.cpp -Iinclude -Lbuild/src -lllama -Lbuild/ggml/src -lggml -o hello_llamacpp
//
// Build (CUDA):
//   cmake -B build -DGGML_CUDA=ON
//   cmake --build build --config Release -j$(nproc)
//   g++ -std=c++17 -O2 hello_llamacpp.cpp -Iinclude -Lbuild/src -lllama -Lbuild/ggml/src -lggml -o hello_llamacpp
//
// Run:
//   ./hello_llamacpp /path/to/Llama-3.2-1B-Q4_K_M.gguf
//
// Model download (HuggingFace CLI):
//   huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
//       --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf" --local-dir .

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// ---------------------------------------------------------------------------
// Helper: tokenize a string using the model's vocabulary
// Returns the number of tokens written into out_tokens.
// ---------------------------------------------------------------------------
static int tokenize_prompt(
    const llama_model* model,
    const std::string& prompt,
    std::vector<llama_token>& out_tokens)
{
    // Upper-bound estimate: one token per ~2 chars, plus BOS + safety margin
    out_tokens.resize(prompt.size() / 2 + 64);

    int n = llama_tokenize(
        llama_model_get_vocab(model),
        prompt.c_str(),
        (int)prompt.size(),
        out_tokens.data(),
        (int)out_tokens.size(),
        /*add_special=*/true,    // prepend BOS token
        /*parse_special=*/false
    );

    if (n < 0) {
        // Buffer was too small — resize and retry
        out_tokens.resize(-n);
        n = llama_tokenize(
            llama_model_get_vocab(model),
            prompt.c_str(),
            (int)prompt.size(),
            out_tokens.data(),
            (int)out_tokens.size(),
            true, false
        );
    }

    assert(n > 0);
    out_tokens.resize(n);
    return n;
}

// ---------------------------------------------------------------------------
// Helper: convert a single token ID to its text piece
// ---------------------------------------------------------------------------
static std::string token_to_piece(const llama_model* model, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(
        llama_model_get_vocab(model),
        token, buf, sizeof(buf),
        /*lstrip=*/0,
        /*special=*/false
    );
    if (n < 0) return "";
    return std::string(buf, n);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const std::string prompt = (argc >= 3)
        ? argv[2]
        : "Explain the difference between prefill and decode in LLM inference.";

    // ------------------------------------------------------------------
    // Step 1: Initialize the GGML backend
    // llama_backend_init() sets up thread pools, initializes the selected
    // compute backend (CPU / CUDA / Metal), and seeds the RNG.
    // ------------------------------------------------------------------
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // ------------------------------------------------------------------
    // Step 2: Load the model
    // llama_model_load_from_file() parses the GGUF header, reads tensor
    // metadata, and memory-maps the weight bytes. With use_mmap=true
    // (the default), weights are NOT copied into RAM — the OS pages them
    // in lazily. This is why cold start is ~1-2s vs. vLLM's 30-140s.
    // ------------------------------------------------------------------
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;   // 0 = CPU only
                                     // set to 99 to offload all layers to GPU

    fprintf(stderr, "Loading model: %s\n", model_path);
    auto t_load_start = std::chrono::high_resolution_clock::now();

    llama_model* model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model from %s\n", model_path);
        return 1;
    }

    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(
        t_load_end - t_load_start).count();
    fprintf(stderr, "Model loaded in %.0f ms\n\n", load_ms);

    // ------------------------------------------------------------------
    // Step 3: Create a context
    // llama_new_context_with_model() allocates the KV cache ring buffer.
    // Size = 2 * n_layers * n_ctx * n_kv_heads * head_dim * sizeof(fp16)
    // For a 1B model at n_ctx=2048: roughly 256 MB.
    // Unlike vLLM, this is a fixed allocation — no paging, no eviction.
    // ------------------------------------------------------------------
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = 2048;   // maximum context window (prompt + generation)
    ctx_params.n_batch  = 512;    // maximum prompt tokens per llama_decode() call
    ctx_params.n_ubatch = 512;    // maximum tokens per physical forward pass
    ctx_params.n_threads        = 4;   // CPU threads for inference
    ctx_params.n_threads_batch  = 4;   // CPU threads during batch processing

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // ------------------------------------------------------------------
    // Step 4: Tokenize
    // ------------------------------------------------------------------
    std::vector<llama_token> prompt_tokens;
    int n_prompt = tokenize_prompt(model, prompt, prompt_tokens);

    fprintf(stderr, "Prompt (%d tokens): %s\n\n", n_prompt, prompt.c_str());

    // ------------------------------------------------------------------
    // Step 5: Prefill
    // llama_batch_get_one() builds a batch from a contiguous token array.
    // llama_decode() runs the forward pass, writes K and V tensors into
    // the KV cache at positions [0 .. n_prompt-1], and leaves logits for
    // the last token in the context's logit buffer.
    // ------------------------------------------------------------------
    auto t_prefill_start = std::chrono::high_resolution_clock::now();

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "ERROR: prefill failed\n");
        return 1;
    }

    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(
        t_prefill_end - t_prefill_start).count();

    fprintf(stderr, "Prefill: %.0f ms (%.0f tok/s)\n\n",
        prefill_ms, n_prompt / (prefill_ms / 1000.0));

    // ------------------------------------------------------------------
    // Step 6: Decode loop
    // Each iteration:
    //   a) Sample the next token from the logits at the last position
    //   b) Check for EOS
    //   c) Detokenize and print the token piece
    //   d) Submit the new token as a batch of size 1
    //   e) llama_decode() runs a forward pass, extends the KV cache by 1
    // ------------------------------------------------------------------

    // Build a default sampler chain: greedy (temperature=0 equivalent)
    // For temperature + top-p, use llama_sampler_chain_add() with the
    // llama_sampler_init_temp() and llama_sampler_init_top_p() samplers.
    llama_sampler* sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    printf("Generated: ");
    fflush(stdout);

    int n_generated = 0;
    const int max_new_tokens = 128;

    auto t_decode_start = std::chrono::high_resolution_clock::now();

    while (n_generated < max_new_tokens) {
        // a) Sample
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        // b) EOS check
        if (llama_vocab_is_eog(llama_model_get_vocab(model), new_token)) {
            break;
        }

        // c) Detokenize and stream to stdout
        std::string piece = token_to_piece(model, new_token);
        printf("%s", piece.c_str());
        fflush(stdout);

        // d) Submit for next step
        batch = llama_batch_get_one(&new_token, 1);

        // e) Forward pass — extends KV cache by 1 position
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "\nERROR: decode step %d failed\n", n_generated);
            break;
        }

        n_generated++;
    }

    printf("\n\n");

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(
        t_decode_end - t_decode_start).count();

    fprintf(stderr, "Decode:  %.0f ms | %d tokens | %.1f tok/s\n",
        decode_ms, n_generated, n_generated / (decode_ms / 1000.0));

    // llama_perf_context_print() prints vLLM-equivalent timing breakdown:
    // prompt processing speed and generation speed
    llama_perf_context_print(ctx);

    // ------------------------------------------------------------------
    // Step 7: Clean up
    // ------------------------------------------------------------------
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}

```

