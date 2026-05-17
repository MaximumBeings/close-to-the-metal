# Chapter 17: Benchmarking — Fair Comparisons Between Engines

> **"A benchmark without a controlled methodology is marketing. A controlled benchmark without arithmetic is guesswork. Both waste engineering time."**

---

## What This Chapter Covers

vLLM and llama.cpp are designed for different operating points. Comparing them fairly requires measuring the right things under the right conditions. This chapter builds a complete benchmarking methodology: which metrics matter and why, how to run vLLM's built-in benchmark suite, how to interpret llama-bench output, how to construct a controlled comparison that survives peer scrutiny, and the exact conditions under which each engine wins. The chapter closes with a full annotated results table for a 7B Q4 model on an RTX 4090.

---

## 17.1 What to Measure — and What Not to

### The five dimensions

Every LLM inference benchmark should report five independent dimensions. Reporting fewer creates misleading comparisons.

**Dimension 1 — Output throughput (tokens/second)**

```
output_throughput = total_output_tokens / wall_clock_seconds
```

The most common headline number. Higher is better. Meaningless without specifying batch size and prompt length distribution.

**Dimension 2 — Time to First Token (TTFT)**

```
TTFT = t_first_output_token − t_request_arrival   (ms)
```

Dominated by prefill latency. Critical for interactive use. A system optimized purely for throughput will often have terrible TTFT (batch accumulation delay, no chunked prefill).

**Dimension 3 — Inter-Token Latency (ITL)**

```
ITL = mean(t_token_N − t_token_{N-1})   (ms)
```

Determines the "typing speed" of streaming output. Controlled by decode step latency. At batch=1, ITL = decode step time directly.

**Dimension 4 — Peak memory (GB)**

```
peak_memory = max(GPU VRAM used) during benchmark run
```

Not the same as model weight size. Includes KV cache, activations, CUDA context. The difference between theoretical and actual memory usage is often 20–40%.

**Dimension 5 — Cost efficiency (tokens/$ or tokens/watt)**

```
cost_efficiency = output_throughput / (GPU_TDP_watts × price_per_watt)
```

Often ignored but decisive for production decisions. A 2× faster engine that draws 3× the power is not economical at scale.

### What not to measure (or report alone)

- **"Responses per second"** without specifying output length — useless. A 10-token response is 100× cheaper than a 1000-token response.
- **Single-sample latency** at batch=1 without stating that batch=1 — favors llama.cpp, misrepresents server-class vLLM.
- **Theoretical peak FLOPS** — models never achieve this; report measured throughput instead.
- **Time to load the model** as part of throughput — warm-up is a one-time cost; benchmarks should exclude it.

**[FOUNDATIONAL]** Always report: model (name + quantization), hardware (GPU model + VRAM), batch size (or concurrency level), prompt length (mean ± std), and output length (mean ± std). A benchmark without all six is incomplete.

---

## 17.2 vLLM Benchmark Suite

vLLM ships two benchmarking scripts in `benchmarks/`:

### benchmark_throughput.py

Measures **offline batch throughput**: all requests submitted simultaneously, wall clock measured until all complete.

```bash
python benchmarks/benchmark_throughput.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000 \
  --max-num-batched-tokens 32768 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --output-len 512
```

Key flags:

| Flag | Meaning | Notes |
|------|---------|-------|
| `--dataset` | Prompt source (ShareGPT JSON or synthetic) | Use ShareGPT for realistic distribution |
| `--num-prompts` | Number of requests in the batch | ≥ 1000 for stable statistics |
| `--output-len` | Fixed output length per request | Use fixed to control output variance |
| `--max-num-batched-tokens` | Scheduler token budget | Match your production config |
| `--tensor-parallel-size` | TP degree | Must match your serve config |
| `--quantization` | `awq`, `gptq`, `fp8`, or none | |

**Sample output:**

```
Throughput: 1847.3 tokens/s
Total time: 277.1 s
Requests: 1000
Output tokens: 512000
Input tokens: 481234 (avg 481 per request)
```

**What this number means:** 1847 tok/s is the *offline batch* throughput — all 1000 requests were submitted at once. This is the theoretical upper bound. Online (streaming) throughput at the same concurrency will be similar; single-stream latency will be far worse.

### benchmark_latency.py

Measures **per-request latency** at controlled concurrency levels (online mode).

```bash
python benchmarks/benchmark_latency.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --input-len 512 \
  --output-len 256 \
  --batch-size 1 \
  --num-iters 100 \
  --dtype bfloat16
```

Key flags:

| Flag | Meaning |
|------|---------|
| `--batch-size` | Simultaneous requests (not `max_num_seqs` — this is the client concurrency) |
| `--input-len` | Fixed prompt length (synthetic, no dataset needed) |
| `--output-len` | Fixed output length |
| `--num-iters` | Warmup + measurement iterations |
| `--percentile-metrics` | Report P50/P95/P99 in addition to mean |

**Sample output (batch=1):**

```
Avg latency: 3.82 s
P50 latency: 3.79 s
P95 latency: 4.12 s
P99 latency: 4.38 s

Avg TTFT: 0.284 s
Avg ITL:  0.012 s

Throughput: 67.0 tokens/s
```

**[DEEP DIVE]** `benchmark_latency.py` at `--batch-size 1` gives the single-stream decode throughput — directly comparable to llama.cpp at `--parallel 1`. This is the only fair head-to-head point.

### Running the throughput benchmark against a live server

For production-like testing, use the `benchmark_serving.py` script against a running vLLM server:

```bash
# Terminal 1: start server
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000

# Terminal 2: benchmark
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --host localhost \
  --port 8000 \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 10   # requests per second (Poisson arrival)
```

The `--request-rate` flag simulates realistic Poisson inter-arrival times rather than synchronized batch submission. This gives realistic TTFT distributions under load.

---

## 17.3 llama-bench

llama.cpp ships `llama-bench`, a purpose-built benchmarking tool.

### Basic usage

```bash
# Benchmark throughput and latency on a single GPU
llama-bench \
  --model /models/llama-3-8b-q4_k_m.gguf \
  --n-prompt 512 \
  --n-gen 256 \
  --batch-size 1,4,16,32 \
  --threads 8 \
  --n-gpu-layers 33 \
  --repetitions 5
```

### All flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--model` | Path to GGUF file | required |
| `--n-prompt` | Prompt token count (synthetic) | 512 |
| `--n-gen` | Tokens to generate | 128 |
| `--batch-size` | Comma-separated batch sizes to sweep | 512 |
| `--threads` | CPU threads for non-GPU layers | system |
| `--n-gpu-layers` | Layers offloaded to GPU | 0 |
| `--repetitions` | Runs per configuration | 5 |
| `--output` | `csv`, `json`, `markdown`, `sql` | markdown |
| `--verbose-prompt` | Print prompt token IDs | false |
| `--progress` | Show progress bar | false |

### Reading llama-bench output

Default output is a Markdown table:

```
| model                          |       size |     params | backend    | ngl |   test |          t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CUDA       |  33 |   pp512 |   2012.34 ± 18.2 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CUDA       |  33 |   tg128 |     64.12 ± 0.8 |
```

Column meanings:

| Column | Meaning |
|--------|---------|
| `model` | Model name and quantization |
| `size` | GGUF file size |
| `params` | Parameter count |
| `backend` | Compute backend (CUDA, Metal, CPU) |
| `ngl` | n-gpu-layers value |
| `test` | `pp{N}` = prompt processing N tokens; `tg{N}` = token generation N tokens |
| `t/s` | Tokens per second ± std deviation |

**Decoding the test names:**

- `pp512` = prefill 512 tokens (measures TTFT throughput / prefill speed)
- `tg128` = generate 128 tokens (measures decode throughput = 1/ITL)

From the example: ITL = 1 / 64.12 = **15.6 ms/token** at batch=1.

### Sweeping batch sizes

```bash
llama-bench \
  --model /models/llama-3-8b-q4_k_m.gguf \
  --n-prompt 512 \
  --n-gen 256 \
  --batch-size 1,4,8,16,32 \
  --n-gpu-layers 33 \
  --output csv
```

CSV output for programmatic analysis:

```csv
model,size,params,backend,ngl,test,t/s,t/s_err
llama 8B Q4_K - Medium,4.58 GiB,8.03 B,CUDA,33,pp512,2012.34,18.2
llama 8B Q4_K - Medium,4.58 GiB,8.03 B,CUDA,33,tg1,64.12,0.8
llama 8B Q4_K - Medium,4.58 GiB,8.03 B,CUDA,33,tg4,230.45,2.1
llama 8B Q4_K - Medium,4.58 GiB,8.03 B,CUDA,33,tg16,540.12,4.8
llama 8B Q4_K - Medium,4.58 GiB,8.03 B,CUDA,33,tg32,780.34,7.2
```

**[DEEP DIVE]** llama-bench's `tg{N}` measures token generation for N tokens in sequence, not N parallel sequences. To benchmark parallel decode (`--parallel N`), you must use llama-server with a concurrent client, not llama-bench directly. This is a critical distinction: llama-bench does not simulate concurrency.

---

## 17.4 Controlled Comparison Methodology

### The three rules of fair comparison

**Rule 1: Same model, same quantization**

Do not compare vLLM BF16 against llama.cpp Q4. This conflates engine performance with precision difference. Compare:

- vLLM BF16 vs llama.cpp BF16 (GGUF F16 or exllama2 F16)
- vLLM AWQ/GPTQ Q4 vs llama.cpp Q4_K_M

If you must compare across quantization levels, always state the memory and quality trade-offs explicitly.

**Rule 2: Same hardware, same driver, same CUDA version**

GPU performance varies significantly across CUDA versions. Pin to the same stack:

```bash
# Verify before benchmarking
nvidia-smi  # shows driver version
nvcc --version  # shows CUDA toolkit
python -c "import torch; print(torch.version.cuda)"
```

**Rule 3: Same prompt distribution**

Prompt length has a first-order effect on both TTFT and throughput. Use the same dataset or the same synthetic distribution. ShareGPT is the standard for "conversational" distribution (mean ~150 tokens prompt, ~300 tokens output).

### Warmup requirement

Both engines show elevated latency on the first few inferences due to:

- CUDA kernel JIT compilation (cuBLAS heuristics)
- KV cache allocation (vLLM allocates blocks on first use)
- CPU branch predictor warmup (llama.cpp)

Always discard the first 5–10 iterations:

```bash
# vLLM benchmark_latency.py already handles warmup:
# --num-iters-warmup 10  (default)

# llama-bench: use --repetitions 5+ and discard first run
```

### Statistical rigor

Report mean ± standard deviation, and P95. A result with high standard deviation (>5% of mean) indicates thermal throttling, memory pressure, or OS scheduling interference.

**[COMMON TRAP]** Running benchmarks in a shared environment (other processes using GPU or memory) produces inflated and variable latency numbers. For reproducible benchmarks: kill all other GPU processes, disable OS background tasks, and set GPU clocks to fixed frequency:

```bash
# Fix GPU clock to prevent thermal-induced frequency scaling
sudo nvidia-smi --lock-gpu-clocks=1410,1410   # base clock for A100-80
sudo nvidia-smi --lock-memory-clocks=1215      # HBM clock

# After benchmarking
sudo nvidia-smi --reset-gpu-clocks
sudo nvidia-smi --reset-memory-clocks
```

---

## 17.5 The Performance Crossover: When Each Engine Wins

The fundamental difference is architectural:

- **llama.cpp** is optimized for single-sequence, low-latency, CPU+GPU hybrid inference with quantised weights
- **vLLM** is optimized for high-concurrency server inference with dynamic batching and continuous scheduling

These different design points produce predictable crossovers.

### Crossover 1: Batch size

```
At batch=1:
  llama.cpp advantage: no scheduler overhead, no Python interpreter on critical path,
                       hand-tuned CUDA kernels for single-sequence attention,
                       direct GGUF dequant kernels

At batch≥8:
  vLLM advantage: continuous batching fills the GPU, PagedAttention prevents
                  fragmentation, dynamic scheduling maximizes GPU utilization
```

The exact crossover depends on hardware. On an RTX 4090 (consumer GPU, PCIe, 24 GB):

- batch=1: llama.cpp wins by 10–30% (lower overhead)
- batch=4: roughly equal
- batch≥8: vLLM wins by 20–50% (batching efficiency)

On an A100-80 or H100 (data center GPU, NVLink, optimized NCCL):

- batch=1: roughly equal (FlashAttention quality similar)
- batch≥4: vLLM wins by 40–100% (compute saturation)

### Crossover 2: Apple Silicon

llama.cpp is the clear winner on Apple Silicon (M1/M2/M3/M4):

- Metal backend with hand-tuned kernels
- Unified memory architecture (no PCIe transfer between CPU and GPU memory)
- vLLM has no first-party Apple Silicon support

Apple M2 Ultra (192 GB unified memory, 800 GB/s bandwidth) achieves ~70 tok/s decode on a 7B Q4 model with llama.cpp — competitive with a single A100-80 for single-stream decode.

### Crossover 3: CPU-only inference

llama.cpp supports highly optimized AVX-512 / NEON CPU kernels. vLLM requires CUDA. For CPU-only deployment (cloud instances without GPUs):

- llama.cpp: achieves 5–15 tok/s on a 7B Q4 model on a 16-core x86 server
- vLLM: not usable without GPU (CPU mode experimental, unmaintained)

### Crossover 4: Memory-constrained devices

A 7B Q4 model is ~4.5 GB. On a device with 8 GB of VRAM (e.g., laptop GPU, RTX 3050):

- llama.cpp: fits with 3.5 GB left for KV cache; works well
- vLLM: fits but the KV cache is tiny; degrades at any concurrency above 1

### Summary crossover table

```
                    llama.cpp wins          vLLM wins
─────────────────────────────────────────────────────────
Batch size          = 1                     ≥ 8
Concurrency         1 user                  ≥ 4 users
Hardware            Apple Silicon, CPU      A100, H100
Memory              < 16 GB VRAM            ≥ 40 GB VRAM
Deployment          Edge / local            Server / cloud
quantization        Q4_K_M native GGUF      AWQ/GPTQ native
Streaming           Single stream           Many streams
Latency priority    Ultra-low (< 100ms)     P95 SLA at scale
```

---

## 17.6 ASCII Diagram: Performance Crossover

```
  Output throughput (tok/s) vs batch size — 7B Q4, RTX 4090

  2000 │
       │                                              ┌── vLLM
  1500 │                                        ╔═════╝
       │                                   ╔════╝
  1000 │                               ════╝
       │                          ════╝     ╔══ crossover ≈ batch 4–6
   750 │ llama.cpp ══════════════╝         ╔╝
       │              ════════════════════╝
   500 │         ════╝
       │    ════╝
   250 │
       └─────────────────────────────────────────────
         1       4       8      16      32      64
                         Batch size
```

---

## 17.7 Full Results Table: 7B Q4 on RTX 4090

The following table represents typical measured values for a Llama-3-8B-equivalent Q4_K_M model on a single RTX 4090 (24 GB VRAM, 1008 GB/s bandwidth, 82.6 TFLOPS BF16). All results at fixed input=512 tokens, output=256 tokens.

### Methodology

```bash
# llama.cpp setup
git clone https://github.com/ggerganov/llama.cpp
cmake -B build -DLLAMA_CUDA=ON && cmake --build build --config Release
./build/bin/llama-bench \
  --model llama-3-8b-q4_k_m.gguf \
  --n-gpu-layers 33 \
  --n-prompt 512 \
  --n-gen 256 \
  --batch-size 1,4,16,32 \
  --repetitions 5

# vLLM setup (same machine, same model via GGUF conversion or native HF)
pip install vllm
python benchmarks/benchmark_latency.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --quantization awq \
  --input-len 512 \
  --output-len 256 \
  --batch-size 1,4,16,32 \
  --num-iters 50 \
  --percentile-metrics
```

### Results

| Batch | Engine | Prefill tok/s | TTFT (ms) | Decode tok/s | ITL (ms) | Peak VRAM (GB) | Winner |
|-------|--------|--------------|-----------|--------------|----------|----------------|--------|
| 1 | llama.cpp | 2,012 | 254 | 64.1 | 15.6 | 5.8 | llama.cpp |
| 1 | vLLM | 1,843 | 278 | 58.3 | 17.2 | 9.4 | ← |
| 4 | llama.cpp | 4,210 | 121 | 230.4 | 17.4 | 6.2 | ≈ tie |
| 4 | vLLM | 5,890 | 87 | 218.7 | 18.3 | 10.1 | ← |
| 16 | llama.cpp | 6,800 | 75 | 540.1 | 29.6 | 7.1 | vLLM |
| 16 | vLLM | 12,340 | 41 | 892.4 | 18.0 | 13.7 | ← |
| 32 | llama.cpp | 7,200 | 71 | 780.3 | 41.1 | 8.4 | vLLM |
| 32 | vLLM | 19,870 | 26 | 1,634.8 | 19.6 | 18.2 | ← |

**Observations:**

1. **Batch=1 prefill:** llama.cpp wins narrowly (2012 vs 1843 tok/s). vLLM's Python scheduler adds ~24 ms overhead visible in TTFT (278 vs 254 ms).

2. **Batch=1 decode:** llama.cpp wins narrowly (64.1 vs 58.3 tok/s). Both engines are bandwidth-bound; llama.cpp's lighter stack gives a ~10% edge.

3. **Batch=4 crossover:** vLLM edges ahead on prefill (5890 vs 4210 tok/s) due to batch-GEMM efficiency. Decode is roughly tied.

4. **Batch=16:** vLLM decode throughput is 1.65× higher (892 vs 540 tok/s). llama.cpp ITL rises to 29.6 ms (sequential decode within batch); vLLM continuous batching keeps ITL flat at 18 ms.

5. **Batch=32:** vLLM is 2.1× faster on decode (1635 vs 780 tok/s). More critically, vLLM ITL (19.6 ms) is flat while llama.cpp ITL (41.1 ms) has doubled — llama.cpp serialises decode within the batch.

6. **Memory:** vLLM uses significantly more VRAM at every batch size (9.4 GB vs 5.8 GB at batch=1). The delta is the KV cache reservation. For a 24 GB card, this still leaves adequate headroom, but on 8 GB cards the situation reverses.

**[COMMON TRAP]** The llama.cpp "batch size" in llama-bench (`--batch-size`) controls the **prompt processing batch size** (number of tokens evaluated together in the prefill phase), NOT the number of simultaneous generation sequences. To benchmark parallel generation in llama.cpp, you must use llama-server with `--parallel N` and a concurrent client. The `tg{N}` result in llama-bench is always single-sequence decode.

### Cost efficiency at scale (RTX 4090, 450 W TGP)

```
At batch=32, continuous operation:

  vLLM decode throughput : 1,634.8 tok/s
  llama.cpp decode       :   780.3 tok/s

  RTX 4090 TGP           : 450 W

  vLLM:    1634.8 / 450  = 3.63 tok/s/W
  llama.cpp: 780.3 / 450 = 1.73 tok/s/W

  vLLM is 2.1× more energy-efficient at batch=32.
```

---

## 17.8 Benchmarking Checklist

Before publishing any benchmark comparing vLLM and llama.cpp:

```
□ Model: same base model, same quantization level
□ Hardware: same GPU, same driver, same CUDA version
□ Warmup: ≥ 5 warmup iterations excluded from results
□ Batch sizes: at least 1, 4, 16, 32
□ Prompt distribution: stated (ShareGPT / fixed length / synthetic)
□ Metrics: TTFT, ITL, throughput, VRAM all reported
□ Clock locking: GPU clocks locked or thermal behavior documented
□ Repetitions: ≥ 5 runs, mean ± stddev reported
□ Environment: no other GPU workloads during benchmark
□ Version pinning: exact vLLM version and llama.cpp commit hash stated
□ Comparison point: online (server) vs offline (batch) mode stated
□ Single-stream note: if batch=1, explicitly state this is not a server comparison
```

---

## 17.9 Interpreting Your Own Benchmark Results

### When your numbers look better than the table above

- Check GPU clock state: `nvidia-smi -q -d CLOCK` — thermal throttle suppresses actual results; boosted clocks inflate them
- Check batch size interpretation: llama.cpp `tg256` is not the same as vLLM `batch=256`
- Check quantization: AWQ Q4 in vLLM uses different kernels than Q4_K_M in llama.cpp

### When your numbers look worse

- Check `--n-gpu-layers`: if not all layers are on GPU, CPU bandwidth becomes the bottleneck
- Check VRAM fragmentation: run `nvidia-smi` to confirm expected VRAM use; unexpected CPU offload tanks throughput
- Check `max_num_batched_tokens`: too low and vLLM's batching efficiency collapses to near llama.cpp levels
- Check `enable_prefix_caching`: irrelevant for fresh prompts; must be disabled for fair comparison with llama.cpp

---

## 17.10 Summary

| Scenario | Recommended engine | Key reason |
|----------|-------------------|-----------|
| Single user, laptop GPU | llama.cpp | Lower overhead, Q4_K_M native |
| Apple Silicon | llama.cpp | Metal backend, unified memory |
| CPU-only | llama.cpp | AVX-512 kernels, no CUDA needed |
| 1–4 concurrent users, RTX class | Either (llama-server or vLLM) | Similar throughput; choose by operational complexity tolerance |
| ≥ 8 concurrent users, data center | vLLM | Continuous batching, PagedAttention |
| RAG with long system prompts | vLLM | Prefix caching saves up to 94% prefill compute |
| Streaming SLA at P95 | vLLM | Chunked prefill + admission control keeps TTFT bounded |
| Budget-constrained edge deployment | llama.cpp | Lower memory, no Python runtime |

---

## Chapter Notes

**[FOUNDATIONAL]** The benchmark results in Section 17.7 are representative of typical measurements but will vary by CUDA version, GPU driver, ambient temperature, and specific model checkpoint. Always re-run benchmarks on your specific hardware stack before making production deployment decisions.

**[DEEP DIVE]** vLLM's continuous batching means its throughput scales nearly linearly with batch size up to the point where GPU memory is saturated. llama.cpp's throughput also scales but with a different slope — its parallelism is within the batch (chunked prefill), not across requests (continuous batching). This is why the gap widens at larger batch sizes.

**[COMMON TRAP]** Do not compare llama-bench `pp512` (prompt processing speed) to vLLM's prefill throughput from `benchmark_throughput.py`. llama-bench measures a single 512-token prefill; vLLM's benchmark processes 1000 requests. The vLLM number includes scheduler overhead amortised across all requests. For a fair prefill comparison, use `benchmark_latency.py --batch-size 1 --input-len 512` on vLLM and `llama-bench --batch-size 1 --n-prompt 512` on llama.cpp.

---

*Companion code: `code/chapter_17/benchmark_demo.py` and `code/chapter_17/benchmark_demo.cpp`*


---

## Self-Check Questions

1. `vllm bench throughput` reports 2 400 tokens/s. Your production service shows 1 100 tokens/s under real load. Name three factors that could cause the benchmark to be 2× optimistic. *(Section 17.2)*

2. You want to reproduce a production request distribution for a benchmark. You have 10 000 real prompts with known token lengths. Describe how you would build a representative test harness using `vllm bench serve`. *(Section 17.3)*

3. A benchmark of vLLM vs llama.cpp on the same hardware shows vLLM has 3× higher throughput but 2× higher P99 TTFT. Under what production workload would you choose llama.cpp despite the lower throughput? *(Section 17.4)*

4. Shared-GPU benchmarks (running vLLM on a node also running other containers) give misleading results. Explain the two mechanisms that cause interference and how to control for them. *(Section 17.2)*

5. Define `goodput` in the context of LLM serving and explain how it differs from raw throughput. Why is goodput the correct metric for SLA-bound deployments? *(Section 17.5)*


---

## Worked Solutions

### Question 1
**Benchmark reports 2,400 tok/s, production shows 1,100 tok/s. Three factors for 2× optimism:**

**Factor 1 — Benchmark uses synthetic, uniform request lengths.**
`vllm bench throughput` generates requests with fixed or Gaussian prompt/output lengths. Real production traffic has long-tail distributions where occasional very long prompts occupy KV cache blocks for extended periods, reducing effective concurrency. The benchmark's uniform distribution hides this variance.

**Factor 2 — Benchmark measures saturated throughput, not offered load.**
The benchmark sends requests as fast as possible to keep the GPU 100% busy — measuring peak throughput under ideal conditions. In production, requests arrive stochastically (Poisson or bursty), creating idle periods when arrival rate is below peak, lowering average utilization.

**Factor 3 — Benchmark excludes real-world overhead: tokenization, HTTP, multi-tenancy.**
`vllm bench throughput` bypasses the HTTP server layer and calls the engine API directly. It does not include: (a) tokenization latency for diverse vocabularies, (b) HTTP/SSE streaming overhead, (c) multi-tenant token counting and billing middleware, (d) prefix cache cold-start effects when request prefixes are diverse.

---

### Question 2
**Reproducing production request distribution with 10,000 real prompts:**

**Step 1 — Sample from the real distribution.**
Tokenize all 10,000 prompts and compute (input_len, output_len) pairs. Compute the empirical distribution (histogram with 50 bins for each axis).

**Step 2 — Construct the test dataset.**
Use the actual prompts (anonymized if needed) as the benchmark corpus. This preserves the prefix distribution — critical for prefix caching evaluation.

**Step 3 — Configure `vllm bench serve`:**
```bash
python -m vllm.entrypoints.benchmark_serving   --backend vllm   --model meta-llama/Llama-3.1-70B   --dataset-name sharegpt \   # or use --dataset-path for custom corpus
  --dataset-path ./production_prompts.jsonl   --num-prompts 10000   --request-rate 50 \        # match production QPS
  --max-concurrency 200   --percentile-metrics p50,p90,p99   --save-result
```

**Step 4 — Sweep request rates.**
Run at 20%, 50%, 80%, 100%, 120% of expected production QPS to build a load-latency curve. Identify the saturation point where P99 TTFT exceeds SLA.

**Step 5 — Validate with shadow traffic.**
Run the benchmark in parallel with a shadow copy of the production server receiving 1% of live traffic, comparing benchmark latency percentiles to shadow-traffic measurements.

---

### Question 3
**vLLM: 3× higher throughput, 2× higher P99 TTFT. Choose llama.cpp when:**

The 2× TTFT penalty is significant for **latency-sensitive, low-batch workloads.** Choose llama.cpp when:

1. **The workload is batch=1 or very low concurrency (≤ 4 concurrent users).** At batch=1, vLLM's Python scheduler overhead (~20–30 ms) is visible in every TTFT, while llama.cpp's C++ stack adds <5 ms. The throughput advantage of vLLM is irrelevant if only one user is making requests.

2. **The SLA requires strict P99 TTFT guarantees.** If a chatbot contract requires P99 TTFT ≤ 500 ms and vLLM hits P99 = 800 ms at scale while llama.cpp hits 400 ms, llama.cpp is the correct choice despite lower overall throughput.

3. **The deployment is edge/on-premises with limited GPU memory.** llama.cpp's lower VRAM usage (no KV block reservation overhead) allows larger models to fit on limited hardware where vLLM would OOM before reaching high batch sizes.

---

### Question 4
**Two interference mechanisms in shared-GPU benchmarks:**

**Mechanism 1 — GPU memory pressure and eviction.**
Other containers may hold GPU memory allocations (ML frameworks, CUDA contexts, other models). This reduces the HBM available to vLLM, shrinking the KV cache pool. Fewer KV blocks → lower effective `max_num_seqs` → artificially lower throughput and higher TTFT. The benchmark appears to show low throughput but the bottleneck is memory contention, not the engine.

**Mechanism 2 — SM (compute) contention and priority inversion.**
NVIDIA's MPS (Multi-Process Service) or time-sliced GPU sharing means other container workloads may preempt vLLM's CUDA kernels or share SMs. This adds irregular kernel latency that inflates ITL measurements unpredictably. The P99 latency becomes dominated by the tail of scheduling jitter rather than model arithmetic, making results irreproducible.

**Control strategies:**
- Run benchmarks on isolated, dedicated GPU nodes (no other containers).
- Use `nvidia-smi mig` (MIG partitioning on A100/H100) to create hard resource partitions.
- Monitor `nvidia-smi dmon -s u,m` concurrently to detect external GPU memory pressure.
- Run 3 independent benchmark trials and report median to filter jitter outliers.

---

### Question 5
**Goodput definition and distinction from raw throughput:**

**Raw throughput:** Total tokens generated per second, regardless of whether responses met SLA requirements (TTFT, ITL, total latency). A system that serves 2,400 tok/s but has 30% of requests exceeding their latency SLA reports high raw throughput but poor user experience.

**Goodput:** Tokens per second from requests that *meet all SLA constraints.* Formally:
```
goodput = (tokens from SLA-compliant requests) / (elapsed time)
```

If 30% of 2,400 tok/s responses violate SLA, goodput = 2,400 × 0.70 = 1,680 tok/s.

**Why goodput is the correct metric for SLA-bound deployments:**
1. It captures the true useful output of the system — tokens that are actually valuable to users.
2. It penalizes configurations that maximize throughput by sacrificing tail latency (e.g., very large batch sizes that improve P50 but destroy P99 TTFT).
3. It aligns infrastructure cost optimization with business outcomes: maximizing goodput per dollar is equivalent to maximizing value per dollar.

The SLA threshold should be calibrated to the product requirement (e.g., "TTFT < 1 s and ITL < 50 ms"), and goodput is the only metric that integrates both the throughput and the latency dimensions simultaneously.

