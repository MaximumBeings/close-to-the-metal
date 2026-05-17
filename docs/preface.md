# Preface

> *"The gap between 'I called the API' and 'I understand what just happened' is exactly the size of this book."*

---

## Why This Book Exists

Sometime in late 2023, a shift happened quietly in a lot of engineering teams. Models stopped being something you called from the cloud and started being something you ran yourself. The models became available. The hardware became (barely) affordable. And the gap between what an API call costs and what self-hosted inference costs became large enough to justify a serious engineering investment.

That is when the questions started.

*Why is my GPU at 28% utilization when I'm serving real traffic?*
*Why does adding a second user double my latency instead of halving it?*
*Why does this 7B model on my laptop feel faster than that 70B model in my data center?*
*What is a KV cache, really, and why does running out of it feel like the service just fell off a cliff?*

These questions don't have one-sentence answers. They require understanding the systems beneath the Python. They require knowing what a GPU is actually doing while you wait for text to appear. They require, in short, a book about inference engineering — not a book about large language models in general, and not a tutorial on calling OpenAI's API, but a book about the hard, specific, rewarding problem of taking a trained model and making it serve production traffic efficiently.

This is that book.

---

## What Is Different Here

Most writing about LLM inference falls into one of two categories: API tutorials (high-level, engine-agnostic, stops at the HTTP boundary) or research papers (assumes GPU programming expertise, skips intuition, rewards only those who already know the answer).

This book occupies a different space. It is written for engineers who write code professionally, who have called an LLM API and want to understand what happened on the other side, and who need to make real decisions about latency, cost, and hardware.

The structure that makes this possible is the **two-engine parallel**. Every concept is taught through two production-grade inference engines simultaneously:

- **vLLM** — the dominant Python-first serving framework for data center GPUs, with continuous batching, PagedAttention, and a rich ecosystem of quantization and parallelism options.
- **llama.cpp** — the C++ engine that runs everywhere else: laptops, edge devices, small clouds, anywhere a Python dependency or a data center GPU is not an option.

These two engines carry the majority of the book. But inference engineering in 2026 does not stop at two engines. Later chapters cover the broader landscape: **SGLang** (structured generation and RadixAttention), **TensorRT-LLM** (NVIDIA's compiler-optimized path), **MLC-LLM** (hardware-portable deployment), and **Ollama** (developer-first local serving). The vLLM V1 architecture — a ground-up redesign with a three-process ZMQ model and hash-based KV deduplication — gets its own dedicated chapter. The goal throughout is the same: understand the mechanics well enough that you can evaluate any engine, not just the ones that existed when this book was written.

These two primary engines make fundamentally different trade-offs, and the friction between those trade-offs is enormously educational. When you understand why vLLM uses virtual memory paging for KV cache management (Chapter 6) and why llama.cpp does not need to (Chapter 28), you understand memory management in LLM inference at a level that no single-engine treatment can give you. When you implement the same speculative decoding algorithm in both Python and C++ (Chapter 23), you see exactly which parts of the algorithm are essential and which parts are implementation choices.

Every chapter has two companion code blocks — Python for vLLM, C++ for llama.cpp — and every chapter has a running worked example computed by hand with real numbers before any code is introduced. The self-check questions at the end of each chapter are not review trivia; they are the questions you will actually be asked in a production incident or a system design interview.

---

## The Running Case Study

Chapter 1 introduces a scenario that recurs throughout the book: a service handling 50,000 concurrent users, starting from a naive deployment that costs $1.2M per month at 28% GPU utilization. This case study appears as a callout in every major chapter — the scheduler chapter, the quantization chapter, the multi-GPU chapter, the speculative decoding chapter — showing exactly how much of that bill each technique removes, and why. By Chapter 38, every optimization layer has been applied, and the bill has been reduced to $108K per month — an 11× reduction that required no new hardware, no new model, and no magic. Just systematic application of the techniques in this book, measured carefully and deployed incrementally.

This is not a hypothetical. Every technique described is in production somewhere. The tools are open source. The mathematics is in this book. The rest is engineering.

---

## How the Book Is Organized

**Part I: Foundations (Chapters 1–5)** establishes the hardware and algorithmic context that everything else depends on.

- **Chapter 1** introduces both engines — vLLM and llama.cpp — and the LinkedIn scenario that runs through the entire book.
- **Chapter 2** maps the GPU and CPU memory landscape — every performance bottleneck in LLM inference traces back to this map.
- **Chapter 2.5** goes inside the GPU: registers, shared memory, L1/L2 caches, constant memory, and global memory — the on-chip hierarchy that determines kernel performance. Covers bank conflicts, coalescing, occupancy, and how FlashAttention, PagedAttention, and tiled GEMM exploit each space.
- **Chapter 3** explains tokens, sequences, and batching in the depth that most introductions skip.
- **Chapters 4 and 5** build from raw attention mechanics up through Flash Attention, with every matrix operation written out by hand.

---

**Part II: Engine Internals (Chapters 6–13, plus 7.5, 8.5, 11.5, 11.6, 12.5)** goes inside both engines.

- **Chapter 6** is the centerpiece of Part II: PagedAttention and the KV cache block manager, including a detailed block-eviction worked example under memory pressure.
- **Chapter 7** follows the scheduler and request lifecycle. **Chapter 7.5** deepens this into continuous batching — the iteration-level loop, token budget admission control, and preemption accounting that determines real-world throughput.
- **Chapter 8** covers startup and initialization. **Chapter 8.5** explains CUDA graphs — how collapsing 820 kernel launches into one submission eliminates the 25% CPU overhead that otherwise dominates decode latency at small batch sizes.
- **Chapters 9–11** cover the forward pass, quantization, and prefill. **Chapter 11.5** adds KV cache eviction — attention sinks, the H2O Heavy Hitter Oracle, SnapKV, and token merging.
- **Chapter 11.6** is a deep dive into RadixAttention and prefix caching — the radix tree data structure, LRU eviction on the tree, vLLM block-level vs SGLang token-exact cache comparison, and worked economics showing 80% prefill compute reduction on a shared system prompt.
- **Chapter 12** covers sampling — from logits to tokens. **Chapter 12.5** covers structured generation and constrained decoding — FSM-based token masking, JSON schema enforcement, EBNF grammars, the `outlines` and `lm-format-enforcer` libraries, vLLM guided decoding, llama.cpp grammar sampling, and function calling as constrained generation.
- **Chapter 13** completes the path through token streaming.

---

**Part III: Production Configuration (Chapters 14–21, plus 15.5)** is the practical operations manual.

- **Chapter 14** examines the eight vLLM configuration knobs and their llama.cpp equivalents in depth.
- **Chapter 15** covers multi-GPU tensor parallelism, now expanded with expert parallelism and pipeline bubble analysis. **Chapter 15.5** extends this to Flash Decoding and context parallelism, which together unlock attention at 100K+ token contexts.
- **Chapter 16** covers observability. **Chapter 17** covers benchmarking methodology. **Chapter 18** covers disaggregated prefill/decode.
- **Chapter 19** covers Kubernetes and KubeRay auto-scaling. **Chapter 20** covers cost engineering. **Chapter 21** covers API security.
- **Appendix H** provides an operational decision tree and troubleshooting guide for common production failure modes.

---

**Part IV: Advanced Techniques (Chapters 22–33.5)** covers the techniques that separate baseline deployments from optimized ones.

- **Chapter 22** covers LoRA adapter hot-swapping. **Chapter 23** covers speculative decoding — now including Medusa, EAGLE, EAGLE-2, and tree-based speculation with a numerical tree-attention mask worked example.
- **Chapter 24** covers reasoning models. **Chapter 25** covers RL serving policies. **Chapter 26** is the CS336 alignment field guide.
- **Chapter 27** covers long-context inference at 128K+ tokens. **Chapter 28** covers llama.cpp as a programming platform. **Chapter 29** covers multimodal inference.
- **Chapter 30** covers semantic caching. **Chapter 31** covers model routing and cascading. **Chapter 32** covers debugging inference systems.
- **Chapter 33** surveys the full engine landscape in 2026. **Chapter 33.5** provides a practical engine-selection guide for SGLang, TRT-LLM, MLC-LLM, and Ollama.

---

**Part V: The Model Zoo (Chapters 34–42)** examines production model families and the architectural decisions behind them.

- **Chapter 34** covers DeepSeek — MLA, MoE, and FP8 at scale.
- **Chapter 35** covers Qwen — multilingual, long-context, and model family engineering.
- **Chapter 36** covers Kimi — Moon-Cache hierarchical KV storage.
- **Chapter 37** covers Nemotron — TRT-LLM, FP8, and 2:4 sparsity.
- **Chapter 38** synthesizes the entire book into the complete $1.2M → $108K production architecture.
- **Chapter 39** covers evaluation and regression testing.
- **Chapter 40** documents the vLLM V1 architecture — the three-process ZMQ design, hash-based KV block deduplication, and the multi-step scheduler that makes V1 meaningfully faster than V0 at production scale.
- **Chapter 41** covers Meta Llama 3 — the architecture of the ecosystem's dominant open-weight family: GQA with 8 KV heads at all sizes, SwiGLU FFN, RoPE with θ = 500,000, the 128K-token tiktoken vocabulary, the Llama 3.1/3.2/3.3 release progression, Llama Guard safety classification, and complete vLLM and llama.cpp serving configurations including quantization sweet spots.
- **Chapter 42** covers Phi-4 and Gemma 3 — the small-model frontier: Phi-4's data-quality hypothesis and MMLU scores exceeding Llama 3.1 70B at 14B parameters; Gemma 3's interleaved local/global attention for 128K context, tied input/output embeddings, 256K vocabulary, and multimodal SigLIP integration; edge deployment decision framework and quantization quality comparison.

---

**Appendices A–X** (26 appendices, grouped by theme) provide reference material designed for repeated use.

*Foundations* — **A** (Mathematical Foundations), **A.2** (Tensor Contractions — 2D/3D/5D/ND with CUDA, Triton, CUTLASS, Mojo; arithmetic intensity; Einstein notation), **A.3** (The Chain Rule — scalar through transformer backpropagation; Jacobians; softmax/LayerNorm/attention backward; QAT STE; LoRA gradient flow; 15 manual worked examples), **B** (Installation Guide), **C** (PyTorch for LLM Inference: dtypes, devices, `torch.compile`, quantization APIs, `torch.distributed`, custom ops, profiling, `torch.export`)

*LLM Engine References* — **D** (vLLM EngineArgs), **E** (llama.cpp CLI), **F** (Production Templates), **G** (Benchmarking Reference), **H** (Operational Decision Tree)

*Systems Programming — CPU* — **I** (C++ Build Patterns), **J** (libtorch: The C++ API — CMake build, `torch::Tensor`, `torch::jit::load`, IValue system, custom CUDA ops in C++, inference server example, Python→C++ cheatsheet), **K** (`std::mdspan` for CPU Inference — C++23 multidimensional views, custom layouts, FP8 accessor, tiled GEMM, KV cache management)

*GPU Kernel Programming* — **L** (CUDA C++ Introduction), **M** (Introduction to Triton), **N** (CUTLASS and Tensor Cores), **O** (Introduction to Mojo)

*Hardware Platforms* — **P** (ROCm and AMD GPU — MI300X, HIP porting, Composable Kernel, cost comparison), **Q** (Mobile and Edge: Android, Apple Silicon, MLX), **R** (Edge Inference: Raspberry Pi and NVIDIA Jetson)

*Production and Serving* — **S** (CI/CD Pipelines), **T** (Embedding and Reranker Serving — BGE-M3, ColBERT, vLLM embed/score), **U** (Quantization Calibration — AWQ, GPTQ, FP8, GGUF), **V** (TurboQuant)

*Reference* — **W** (Glossary, 85+ terms), **X** (References, 40+ papers)

---

## What This Book Is Not

This book does not teach you to train language models. It does not cover fine-tuning workflows, dataset curation, or RLHF pipelines (except Chapter 25–26, which cover those topics specifically as they affect the serving stack). It does not assume you will use any particular cloud provider. It does not tell you which model is best — models change faster than books can track, but the hardware constraints and algorithmic trade-offs that govern inference efficiency have been stable for years and will remain so.

This book will not become obsolete when the next model family ships. The chapter on PagedAttention is as relevant for a 2026 model as for a 2024 model, because the physics of GPU memory has not changed. The chapter on speculative decoding will still be correct when the acceptance rates and speedup multipliers look different, because the mathematics of the acceptance-rejection algorithm is independent of any particular model pair.

---

## A Note on the Companion Code

Every chapter that introduces a system concept has a companion code file in the `code/` directory: a Python demo (`*_demo.py`) and, where applicable, a C++ demo (`*_demo.cpp`). Each demo file is self-contained, has no external dependencies beyond the Python standard library or a C++ compiler, and contains assertions that verify the key worked examples from the chapter text.

Running `python3 chapter_06/paged_attention_demo.py` should feel like re-reading the chapter at 10× speed — the same numbers, the same concepts, the same edge cases, but now interactive and verifiable. The C++ files compile with `g++ -O2 -std=c++17` unless otherwise noted; the CUDA files (Appendix L) require `nvcc`.

The code is intentionally not production code. It is educational code: clear rather than clever, readable rather than fast, honest about the simplifications it makes. When you are ready to write production code, the chapter will have pointed you to the right vLLM source files, the right llama.cpp functions, and the right papers.

---

## Acknowledgements

This book was written during a period when LLM inference engineering was transforming from a specialized curiosity into a mainstream engineering discipline. The open-source communities around vLLM and llama.cpp made this book possible — every technique described here has been implemented, debugged, and documented by engineers who published their work. The papers in Appendix X represent years of insight that this book attempts to make accessible without losing the rigor.

The LinkedIn scenario in Chapter 1 is fictional, but the number — $1.2M per month for a real-time text service at scale — is grounded in conversations with engineers who have lived it. The $108K target is grounded in the same conversations. The gap between those two numbers is the reason this book exists.

---

## How to Read This Book

If you are new to inference engineering, read from Chapter 1 through Chapter 13 before skipping around. The mental model built in those thirteen chapters — the hardware landscape, the memory hierarchy, the KV cache, the scheduler — is the foundation for everything else.

If you have operational experience with vLLM or llama.cpp but want to go deeper, Part III and Part IV are designed to be read in any order. Each chapter states its prerequisites explicitly.

If you are preparing for a specific problem — a cost reduction project, a latency SLA you are missing, a hardware upgrade you are evaluating — use the Chapter 38 synthesis and the index to find the relevant chapters, then follow the cross-references backward.

The book is written to be read twice: once for understanding, and once as a reference during a production incident when you have fifteen minutes and a graph that looks wrong.

---

*The first time you see a KV cache eviction cause a 40-second request, you will wish you had read Chapter 6 before it happened. This book is here so you read it first.*

---

## Quick-Reference: Which Chapter Answers Your Question?

If you are searching for a specific answer rather than reading linearly, use this table.

| Question | Chapter |
|---|---|
| **What is a KV cache and why does running out of it feel catastrophic?** | Ch 6 (PagedAttention) |
| **Why is my GPU at 28% utilization under real traffic?** | Ch 2 (Memory Landscapes) + Ch 3 (Batching) |
| **What is continuous batching and why does it change everything?** | Ch 3 (Tokens and the Batch) |
| **What are GPU registers, shared memory, L1/L2, constant memory, and global memory?** | Ch 2.5 (GPU Memory Architecture) |
| **Why does coalesced vs uncoalesced global memory access produce a 30× gap?** | Ch 2.5 (GPU Memory Architecture) |
| **What are bank conflicts and how does the padding trick fix them?** | Ch 2.5 (GPU Memory Architecture) |
| **What exactly is Flash Attention and why is it faster?** | Ch 5 (Flash Attention) |
| **What is inside one transformer block — FFN, norms, residual stream?** | Ch 3.5 (Transformer Block) |
| **How does vLLM schedule requests and prevent head-of-line blocking?** | Ch 7 (Scheduler) |
| **What happens at `vllm serve` startup — model loading, weight sharding?** | Ch 8 (Startup) |
| **What is the difference between prefill and decode, really?** | Ch 11 (Prefill and Prompt Caching) |
| **How does quantization work — GGUF vs. AWQ vs. FP8?** | Ch 10 (Quantization Internals) |
| **Which 8 knobs most affect vLLM performance in production?** | Ch 14 (The Eight vLLM Knobs) |
| **How do I serve a 70B model on 4 GPUs — what is tensor parallelism?** | Ch 15 (Multi-GPU Serving) |
| **What metrics should I export and which alerts matter?** | Ch 16 (Observability) |
| **How do I fairly benchmark vLLM vs llama.cpp?** | Ch 17 (Benchmarking) |
| **What is disaggregated prefill and when should I use it?** | Ch 18 (Disaggregated Prefill/Decode) |
| **How do I deploy vLLM on Kubernetes at scale?** | Ch 19 (Kubernetes and KubeRay) |
| **What is the actual $/million-token cost of different setups?** | Ch 20 (Cost Engineering) |
| **How do I stop prompt injection attacks on my inference API?** | Ch 21 (Security) |
| **How do I serve 50 LoRA adapters without 50× the memory?** | Ch 22 (LoRA Serving) |
| **What is speculative decoding and when does it help?** | Ch 23 (Speculative Decoding) |
| **How do reasoning models (o1-style) change inference requirements?** | Ch 24 (Reasoning Model Inference) |
| **How do I serve 128K-context requests without OOM?** | Ch 27 (Long-Context Inference) |
| **How do I run llama.cpp as a proper production server?** | Ch 28 (llama.cpp as a Platform) |
| **How do I run vision/audio models through vLLM?** | Ch 29 (Multimodal Inference) |
| **How do I cache repetitive prompts to cut inference cost by 50–80%?** | Ch 30 (Semantic Caching) |
| **How do I route requests to the cheapest model that can answer them?** | Ch 31 (Model Routing and Cascading) |
| **How do I debug a vLLM instance that is slow or returning garbage?** | Ch 32 (Debugging Inference Systems) |
| **Why does my prefix cache hit rate stay below 40%?** | Ch 11.6 (RadixAttention and Prefix Caching) |
| **How do I guarantee valid JSON output from my LLM?** | Ch 12.5 (Structured Generation) |
| **How does DeepSeek's MLA + MoE architecture affect serving?** | Ch 34 (DeepSeek) |
| **What are the inference implications of Llama 3's 8 KV heads at all sizes?** | Ch 41 (Meta Llama 3) |
| **How does Phi-4 outperform Llama 3 70B at 14B parameters?** | Ch 42 (Phi-4 and Gemma 3) |
| **How does Gemma 3's interleaved local/global attention enable 128K context cheaply?** | Ch 42 (Phi-4 and Gemma 3) |
| **How do I evaluate whether my serving setup is correct and stays correct?** | Ch 39 (Evaluation and Regression Testing) |
| **What are the mathematical foundations — softmax, attention, backprop?** | Appendix A |
| **How do tensor contractions work — GEMM, batched matmul, GQA, MoE, MLA?** | Appendix A.2 |
| **How does backpropagation work step by step — Jacobians, softmax/LayerNorm/attention gradients, QAT, LoRA?** | Appendix A.3 |
| **How do I install vLLM and llama.cpp from scratch?** | Appendix B |
| **What do all the vLLM EngineArgs flags mean?** | Appendix D |
| **What does every llama.cpp CLI flag do?** | Appendix E |
| **I want copy-paste production configs — Dockerfiles, YAML, nginx** | Appendix F |
| **What are tiled GEMM, prefix scan, convolution in CUDA C++?** | Appendix L |
| **How do I run llama.cpp on Android or Apple Silicon — and what about MLX?** | Appendix Q |
| **How do I run llama.cpp on a Raspberry Pi or NVIDIA Jetson?** | Appendix R |
| **How do I build a CI/CD pipeline for LLM inference — canary deploys, model eval gates, load testing?** | Appendix S |
| **How do I serve BGE or E5 embedding models and rerankers for RAG?** | Appendix T |
| **How do I run vLLM on an AMD MI300X instead of NVIDIA?** | Appendix P |
| **How do I calibrate and quantize my own model to AWQ, GPTQ, FP8, or GGUF?** | Appendix U |
| **How does PyTorch work under the hood — dtypes, strides, inference mode, torch.compile, custom ops?** | Appendix C |
| **How do I build an inference server directly in C++ using libtorch without a Python runtime?** | Appendix J |

---

*— Oluwaseyi Awoga, written with AI assistance, May 2026*
*May 2026*
