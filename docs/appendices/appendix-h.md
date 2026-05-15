# Appendix H: Glossary

Terms are defined in the context of LLM inference engineering. Chapter references indicate where each concept is explained in depth.

---

## A

**Activation Quantization**
Quantizing not just model weights but the intermediate activation tensors during forward passes. Required for FP8 inference since activations also need to fit in the narrower FP8 range. Contrast with weight-only quantization.

**Arithmetic Intensity**
FLOPs divided by bytes accessed from memory for a given operation. The ratio determines whether an operation is compute-bound (high intensity, bottlenecked by FLOPS) or memory-bandwidth-bound (low intensity, bottlenecked by memory reads/writes). Decode attention has intensity ~1 FLOP/byte; large GEMM has intensity ~100+ FLOP/byte. → Chapter 2, Appendix J

**Attention Head**
One of $n_{heads}$ parallel attention mechanisms in a multi-head attention layer. Each head learns to attend to different aspects of the input. Heads are computed in parallel and concatenated. → Chapter 5

**Attention Sink**
The phenomenon (StreamingLLM, 2023) where the first few tokens of a sequence accumulate disproportionately high attention weights regardless of content; these tokens must be kept in the KV cache even during eviction. Exploited by StreamingLLM to enable infinite-length generation by always retaining sink tokens plus a sliding window of recent tokens. → Chapter 11.5

**Automatic Prefix Caching (APC)**
vLLM's default-on feature that deduplicates KV cache blocks with identical token sequences using a hash-based trie, eliminating redundant prefill computation. APC is enabled by default in vLLM V1; requests sharing a common prefix (e.g., a system prompt) reuse the cached KV blocks without re-running the prefill. → Chapter 11

**Auto-scaling**
Dynamically adjusting the number of serving replicas based on load. vLLM integrates with KubeRay for autoscaling. Key metric: queue depth or GPU utilization. → Chapter 19

**AWQ (Activation-aware Weight Quantization)**
A post-training INT4 quantization method that uses per-channel scale factors chosen to minimize quantization error for the most important (highest-activation) weight channels. Typically higher quality than GPTQ at the same bit width. → Chapter 10

---

## B

**Batch Size**
Number of sequences processed simultaneously. Larger batches improve GPU utilization (more FLOPs per memory access) but increase KV cache memory usage. vLLM uses continuous batching where "batch size" varies per step. → Chapter 3

**BF16 (Brain Float 16)**
A 16-bit floating point format with 8 exponent bits and 7 mantissa bits. Preferred over FP16 for training and inference because its larger dynamic range (same as FP32) avoids overflow. H100/A100 natively support BF16 Tensor Cores. → Chapter 2

**Block Manager**
vLLM component that manages allocation and deallocation of KV cache blocks. Implements the virtual-to-physical block table mapping analogous to virtual memory paging. → Chapter 6

**Block Size**
In PagedAttention, the number of tokens stored per KV cache block (vLLM default: 16). Smaller blocks reduce internal fragmentation but increase block table size. → Chapter 6

**Bursty Traffic**
Workload pattern with periods of high request arrival rate followed by low arrival rate. Production systems must handle bursts without queue overflow or timeout.

---

## C

**Causal Mask**
A mask applied during attention computation that prevents tokens from attending to future tokens. Ensures autoregressive generation: token $i$ can only attend to tokens $0, 1, ..., i$. Implemented by setting future positions to $-\infty$ before softmax. → Appendix A

**Chunked Prefill**
Splitting long prompt processing (prefill) into smaller chunks processed over multiple steps. Benefits: prevents long prefills from blocking decode operations for other sequences; reduces TTFT variance. → Chapter 11

**Continuous Batching**
Serving strategy where new requests are added to the batch as soon as slots become available, rather than waiting for the entire batch to finish. Eliminates the "padding waste" of static batching. vLLM's default mode. → Chapter 7

**Context Length**
Maximum number of tokens (input + output) a model can process. Limited by RoPE maximum position, KV cache memory, and Flash Attention implementation. Extended via YaRN or other techniques. → Chapter 27

**Context Parallelism (CP)**
A parallelism strategy that splits a single long sequence across multiple GPUs along the sequence dimension, using ring attention to communicate KV blocks between devices. Each GPU attends over its local query slice against the full KV sequence, which is passed ring-style between peers. Enables processing sequences longer than any single GPU's HBM. → Chapter 15

**Copy-on-Write (CoW)**
KV cache optimization where multiple sequences sharing a common prefix point to the same physical KV blocks. When one sequence diverges (decodes a different token), only then is the block copied. Used in beam search and prefix caching. → Chapter 6

---

## D

**Decode Phase**
The autoregressive generation phase where one token is produced per forward pass. Memory-bandwidth-bound (loads all model weights per step). Also called "generation" phase. Contrast with Prefill Phase. → Chapter 3

**Decode Throughput**
Number of output tokens generated per second across all active sequences. Limited by memory bandwidth (bandwidth / model_bytes_per_token). → Chapter 2

**Disaggregated Serving**
Architecture separating prefill computation and decode computation onto different hardware pools. Prefill nodes are compute-optimized (large batch prefill); decode nodes are bandwidth-optimized (large decode batch). → Chapter 18

**Draft Model**
In speculative decoding, a small fast model used to speculatively generate $\gamma$ candidate tokens which are verified by the larger target model in parallel. Must share the same tokenizer vocabulary as the target. → Chapter 23

---

## E

**EAGLE**
An efficient speculative decoding method (Li et al., 2024) that trains a lightweight autoregressive draft model on the base model's feature space rather than output tokens, achieving higher acceptance rates than Medusa with similar overhead. EAGLE's draft model consumes the target model's hidden states as additional input, making its predictions contextually richer than token-level draft models. → Chapter 23

**Expert Parallelism (EP)**
A parallelism strategy for Mixture-of-Experts models that distributes expert FFN blocks across GPUs, using all-to-all communication to route tokens to the appropriate expert device. Each GPU holds a subset of experts; the router dispatches tokens across the cluster, and results are gathered back after expert computation. Requires all-to-all communication when routing tokens to remote experts. → Chapter 15

**Expert Utilization**
In MoE models, the fraction of tokens routed to each expert. Ideally uniform (balanced); in practice, without load balancing, a small fraction of experts receive most tokens (expert collapse). → Appendix A

---

## F

**Flash Attention**
Attention algorithm that fuses the softmax and weighted average into a single GPU kernel, computing attention in tiles to avoid materializing the $O(L^2)$ attention score matrix in HBM. Reduces memory complexity from $O(L^2)$ to $O(L)$. → Chapter 5

**Flash Decoding**
A technique (Dao et al., 2023) that parallelizes the attention reduction over the key-value sequence dimension by splitting KV into partitions, computing partial log-sum-exp normalizers independently, and merging results. Particularly effective for single-query decode over long contexts because it exposes more parallelism than standard Flash Attention, which is limited by the query dimension. → Chapter 15.5

**FlashInfer**
A high-performance attention kernel library that serves as vLLM's default attention backend (2025+), implementing Flash Decoding, paged attention, and CUDA graph-compatible interfaces. FlashInfer provides optimized kernels for both prefill and decode phases, with automatic dispatch based on batch shape and context length. → Chapter 5

**FLOPs (Floating Point Operations)**
Count of floating point multiply-add operations. Used to measure model complexity and compare hardware performance. Common in roofline analysis. Note: FLOPs ≠ FLOPS (the latter is per second). → Appendix A

**FLOPS (Floating Point Operations Per Second)**
Hardware throughput metric. H100 SXM: 989 TFLOPS BF16, 1,979 TFLOPS FP8. → Appendix J

**FP8**
8-bit floating point format. Two variants: E4M3 (4 exponent bits, 3 mantissa bits, range ±448) and E5M2 (5 exponent bits, 2 mantissa bits, wider range). H100 Tensor Cores natively execute FP8 GEMMs at 2× the FLOPS of BF16. Requires calibration. → Chapter 10, Chapter 37

**Fragmentation (KV Cache)**
Wasted KV cache memory due to misalignment between sequence lengths and block boundaries (internal fragmentation) or blocks reserved for sequences that could be interleaved (external fragmentation). PagedAttention reduces both. → Chapter 6

---

## G

**GEMM (General Matrix-Matrix Multiplication)**
Core operation in transformer layers. Weight projection layers are large GEMMs. Throughput is compute-bound at large batch sizes.

**GEMV (General Matrix-Vector Multiplication)**
A GEMM where one operand is a vector (batch size = 1). Dominant during single-sequence decode. Memory-bandwidth-bound. → Chapter 9, Appendix J

**GGUF (GGML Universal Format)**
Binary file format used by llama.cpp to store quantized model weights, tokenizer, and metadata in a single file. Supports multiple quantization types (Q4_K_M, Q8_0, etc.). → Chapter 10

**GPTQ**
Post-training quantization method using second-order gradient information (Hessian) to minimize quantization error layer by layer. Supports INT4 and INT8. → Chapter 10

**GPU Utilization**
Percentage of time the GPU's compute units are active. Low utilization (< 50%) indicates memory-bandwidth bottleneck or scheduling inefficiency. High utilization (> 80%) indicates compute bottleneck or good batching. → Chapter 16

**GQA (Grouped Query Attention)**
Attention variant where multiple query heads share the same key/value heads. Reduces KV cache memory by a factor of $n_{heads}/n_{kv}$. Used by Llama 3.1, Qwen2.5, Mistral. → Chapter 6, Appendix A

---

## H

**H2O (Heavy Hitter Oracle)**
A KV cache eviction policy that retains tokens with the highest cumulative attention scores ("heavy hitters") plus a fixed set of recent tokens, evicting all others when the cache is full. Heavy hitters are identified by accumulating attention weight statistics during generation; tokens that have been attended to most across all past steps are deemed most important. → Chapter 11.5

**H100**
NVIDIA Hopper architecture GPU with 80GB HBM3 memory, 3.35 TB/s bandwidth, 989 TFLOPS BF16, 1,979 TFLOPS FP8. The primary GPU for LLM serving as of 2024-2025. → Chapter 2

**HBM (High Bandwidth Memory)**
Memory technology used in data center GPUs. Stacked DRAM dies with very wide memory bus. H100 HBM3: 3.35 TB/s. H200 HBM3e: 4.8 TB/s. Model weights and KV cache are stored in HBM during inference. → Chapter 2

**HBM Bandwidth**
Rate at which data can be read from/written to GPU HBM. The primary bottleneck for memory-bound operations (decode attention, GEMV). Determines maximum decode token throughput. → Chapter 2

---

## I

**In-flight Batching**
See Continuous Batching.

**INT4 / INT8**
Integer quantization formats. INT4 stores weights in 4 bits (16 distinct values), INT8 in 8 bits (256 distinct values). Model weights are dequantized to FP16/BF16 before arithmetic (weight-only quantization). → Chapter 10

**ITL (Inter-Token Latency)**
Time between successive output tokens after the first. Determined by decode speed. For streaming applications, target ITL < 20ms for smooth text display. → Chapter 16

---

## K

**KV Cache**
Memory storing key and value tensors from past attention computations. Enables incremental decoding without recomputing all previous tokens. Memory grows linearly with sequence length × batch size × layers × KV heads. → Chapter 6

**KV Cache Eviction**
The process of removing KV blocks from the cache when it is full, to make room for new tokens in long-context generation. Policies include H2O (heavy hitter oracle), SnapKV (cluster-based compression), and StreamingLLM's attention-sink approach (keep sink tokens + sliding window). Eviction introduces approximation error; its impact on output quality depends on which tokens are dropped. → Chapter 11.5

**KV Compression**
Techniques to reduce KV cache memory: GQA (fewer KV heads), quantization (FP8/INT8 KV), MLA (low-rank projection), offloading to DRAM/NVMe. → Chapters 6, 34, 36

---

## L

**LoRA (Low-Rank Adaptation)**
Parameter-efficient fine-tuning technique adding low-rank update matrices $\Delta W = A \cdot B$ to frozen pretrained weights. Small in memory (rank × d_model × 2 parameters). vLLM supports serving multiple LoRA adapters simultaneously. → Chapter 22

**Long-Range Dependency**
Token relationships spanning many positions in the context window. Transformer self-attention theoretically handles unlimited range, but in practice limited by context length and positional encoding. → Chapter 27

---

## M

**Medusa**
A speculative decoding variant (Cai et al., 2024) that adds multiple lightweight MLP heads to the base model's final hidden state, each predicting a future token position, enabling K draft tokens per forward pass without a separate model. Unlike standard speculative decoding, Medusa requires a fine-tuning step to train the additional heads. Acceptance rates are typically lower than EAGLE but the implementation is simpler (no separate draft model). → Chapter 23

**Memory Bandwidth**
See HBM Bandwidth.

**MHA (Multi-Head Attention)**
Original attention mechanism with $n_{heads}$ independent attention heads, each with their own Q, K, V projections. Replaced by GQA in modern models for reduced KV cache. → Appendix A

**MLA (Multi-head Latent Attention)**
DeepSeek's KV cache compression technique that projects keys and values to a low-rank latent space before caching, reducing KV memory by ~4.7× vs standard MHA. During prefill, MLA computes full-rank KV for attention; only the compressed latent vectors are stored. During decode, latent vectors are up-projected on the fly. Used in DeepSeek-V2 and DeepSeek-V3. → Chapter 34

**MoE (Mixture of Experts)**
Architecture where the FFN layer is replaced by N expert FFNs with a router that selects top-K experts per token. Increases model capacity without proportional compute cost. → Chapter 15, Chapter 34

**Moon-Cache**
Kimi's three-tier KV cache hierarchy (HBM → DRAM → NVMe) that stores KV blocks across memory tiers based on recency and access frequency, enabling context windows far larger than HBM capacity. Hot blocks remain in HBM; warm blocks spill to DRAM; cold blocks are persisted to NVMe. A background prefetcher predicts which blocks will be needed and promotes them before the next decode step. Enables serving ultra-long contexts (1M tokens) by tiering rarely-accessed KV blocks to slower storage. → Chapter 36

**MQA (Multi-Query Attention)**
Attention variant with one shared K and V head across all Q heads. Maximum KV memory reduction but quality lower than GQA. Used by some older models. → Appendix A

---

## N

**NCCL (NVIDIA Collective Communications Library)**
GPU communication library for all-reduce, all-to-all, broadcast operations between GPUs. Used by vLLM for tensor parallelism. Requires NVLink or InfiniBand for good performance. → Chapter 15

**Nucleus Sampling**
Sampling strategy selecting tokens whose cumulative probability exceeds threshold $p$ (top-p). Balances diversity and quality. → Chapter 12

**NVLink**
High-bandwidth interconnect between GPUs on the same node. H100 NVLink: 900 GB/s vs PCIe 5.0: 128 GB/s. Essential for tensor parallel performance. → Chapter 15

---

## O

**Occupancy (GPU)**
Fraction of maximum warps that can run simultaneously on a Streaming Multiprocessor. Limited by register usage and shared memory. Higher occupancy → better latency hiding. → Appendix J

**Online Softmax**
Algorithm computing softmax in a single pass by maintaining running maximum and denominator. Used in Flash Attention to avoid materializing the full attention matrix. → Chapter 5, Appendix A

---

## P

**PagedAttention**
vLLM's KV cache management system inspired by OS virtual memory paging. Allocates KV cache in fixed-size blocks and uses a block table to map logical sequence positions to physical blocks. Eliminates KV cache fragmentation. → Chapter 6

**Pipeline Parallelism (PP)**
Model parallelism splitting transformer layers across GPUs sequentially. GPU 0 runs layers 0-N/k, GPU 1 runs layers N/k to 2N/k, etc. Introduces pipeline bubble overhead. → Chapter 15

**Prefix Caching**
Caching KV blocks for repeated prompt prefixes (e.g., system prompts). Subsequent requests with the same prefix reuse cached KV without recomputation. vLLM implements this as RadixAttention. → Chapter 11

**Prefill Phase**
The initial computation that processes the entire input prompt simultaneously and generates the first output token. Compute-bound for long prompts. Creates the initial KV cache. → Chapter 3

**Prompt Caching**
See Prefix Caching.

---

## Q

**Quantization**
Reducing numerical precision of model weights (and/or activations and KV cache) to reduce memory and increase throughput. Trade-off: lower bits → more memory savings → more quality degradation. → Chapter 10

---

## R

**RadixAttention**
SGLang's KV cache sharing mechanism that organizes cached KV blocks in a radix tree keyed by token sequences, automatically sharing prefix blocks across requests with common prefixes. vLLM's prefix caching system is architecturally similar, using a hash-based trie rather than a true radix tree. Both systems enable O(prefix_len) cache lookup and avoid redundant prefill computation for shared prefixes. → Chapter 11

**Request Throughput**
Number of completed requests per second. Distinct from token throughput. Depends on output length distribution and batching efficiency. → Chapter 17

**Ridge Point (Roofline)**
The arithmetic intensity at which performance transitions from memory-bandwidth-bound to compute-bound. = Peak FLOPS / Peak Memory Bandwidth. H100: ~295 FLOPs/byte (BF16). → Appendix A

**RoPE (Rotary Position Encoding)**
Positional encoding method encoding token positions by rotating Q and K vectors. Key property: dot product depends only on relative position (m-n), not absolute positions. Used by Llama, Qwen, Mistral, DeepSeek. → Appendix A

**RMSNorm**
Layer normalization variant using Root Mean Square normalization without mean subtraction. ~7% faster than LayerNorm. Used by Llama, Qwen, Mistral. → Appendix A

---

## S

**Semantic Cache**
Cache that stores LLM responses indexed by semantic similarity (embedding similarity), not exact text match. Enables cache hits for semantically equivalent but differently worded questions. Hit rate: typically 60-80% for FAQ workloads. → Chapter 30

**SnapKV**
A KV cache compression method that clusters similar key vectors using a pooling window and retains only the cluster centroids, reducing KV cache size while preserving attention quality. SnapKV identifies "important" key positions by observing which positions receive high attention from the most recent query tokens, then retains those positions and discards the rest. → Chapter 11.5

**Shared Memory (CUDA)**
On-chip SRAM on each GPU Streaming Multiprocessor. ~228KB per SM on H100. Much faster than HBM (bandwidth > 19 TB/s vs 3.35 TB/s for HBM). Flash Attention uses shared memory for its tiled computation. → Appendix J

**SM (Streaming Multiprocessor)**
Basic compute unit of an NVIDIA GPU. H100 SXM has 132 SMs. Each SM contains 128 CUDA cores, Tensor Cores, shared memory, and register file. → Appendix J

**Speculative Decoding**
Inference acceleration technique using a small draft model to propose $\gamma$ candidate tokens, then verifying all with the target model in a single forward pass. Maintains exact target model output distribution. Speedup: 2-3× for long outputs. → Chapter 23

**Structured Sparsity (2:4)**
NVIDIA's hardware-accelerated sparsity format requiring at least 2 zeros in every group of 4 weights, enabling 2× TFLOPS on H100 Sparse Tensor Cores. A weight matrix is pruned and then stored in a compressed format with a 2-bit metadata mask per group of 4. The Sparse Tensor Core decompresses on the fly during GEMM. Requires a fine-pruning or one-shot pruning step to reach the 2:4 pattern. → Chapter 10, Chapter 37

**SwiGLU**
Activation function used in modern FFN layers: $\text{SwiGLU}(x) = (\text{SiLU}(x \cdot W_{gate})) \odot (x \cdot W_{up})$. Empirically superior to GeLU. Requires 3 weight matrices vs 2 for standard FFN. → Appendix A

---

## T

**Tensor Parallelism (TP)**
Model parallelism splitting individual weight matrices across GPUs. Column-parallel for $W_Q, W_K, W_V, W_{gate}$; row-parallel for $W_O, W_{down}$. Requires all-reduce between GPUs per layer. → Chapter 15

**Tensor Core**
Specialized compute unit in NVIDIA GPUs performing matrix multiply-accumulate (MMA) operations. Support BF16, FP16, INT8, FP8, TF32. Provide ~8× higher throughput than CUDA cores for GEMM. → Appendix J

**Token**
Fundamental unit of text processed by LLMs. Approximately 4 characters for English. Tokenization maps text to integer token IDs via BPE or similar algorithms. → Chapter 3

**Token Throughput**
Output tokens generated per second. Limited by: (1) hardware bandwidth in decode, (2) model FLOPS in prefill. → Chapter 17

**TRT-LLM (TensorRT-LLM)**
NVIDIA's compiled inference engine. Converts HuggingFace models to AOT-compiled `.engine` files with kernel auto-tuning, layer fusion, and FP8/sparsity support. 2-3× higher throughput than vLLM at cost of 30-60 min compile time. → Chapter 37

**TTFT (Time to First Token)**
Latency from sending request to receiving first output token. Determined by prefill speed. Critical for interactive applications. P95 target: < 500ms for most use cases. → Chapter 16

---

## V

**vLLM**
Open-source LLM inference engine developed by UC Berkeley Sky Computing Lab. Key innovations: PagedAttention (KV cache management), continuous batching, prefix caching. Leading throughput among Python-based serving frameworks. → Chapters 6-8

**vLLM V1**
The late-2024 architectural refactor of vLLM that introduced a disaggregated scheduler/worker design connected via ZMQ sockets, replacing the monolithic V0 engine with an async message-passing architecture. V1 enables chunked prefill and prefix caching by default, improves multi-GPU startup time, and provides cleaner separation between the scheduler (CPU) and workers (GPU). → Chapters 7-8

**VRAM**
GPU video RAM. Same as HBM for data center GPUs. Constrains maximum model size (weights) and KV cache capacity. → Chapter 2

---

## W

**Warp**
Group of 32 CUDA threads executing the same instruction simultaneously (SIMT). The basic scheduling unit on a GPU SM. Warp divergence (different execution paths) reduces efficiency. → Appendix J

**Weight-Only Quantization**
Quantizing model weights (not activations) to INT4/INT8. Weights are dequantized to FP16 before matrix multiply. Reduces memory footprint without modifying compute path. Used by GGUF Q4_K_M, AWQ, GPTQ. → Chapter 10

---

## Y

**YaRN (Yet Another RoPE extensioN)**
Context length extension technique for RoPE-based models. Adjusts RoPE base frequency to support longer sequences than the model was trained on. Achieves 4-8× context extension with minimal quality loss. → Chapter 27
