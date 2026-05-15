# Appendix I: References and Papers

This appendix lists the primary research papers, technical reports, and documentation referenced throughout the book. Organized by topic. All papers are publicly available on arXiv or through official channels.

---

## I.1 Foundational Transformer Architecture

**Attention Is All You Need**
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I. (2017).
*Advances in Neural Information Processing Systems (NeurIPS).*
https://arxiv.org/abs/1706.03762

*The paper that introduced the transformer architecture: multi-head attention, positional encoding, encoder-decoder structure. Every model in this book is derived from this paper.*

---

**Improving Language Understanding by Generative Pre-Training (GPT)**
Radford, A., Narasimhan, K., Salimans, T., Sutskever, I. (2018). OpenAI.
https://openai.com/research/language-unsupervised

*Introduced decoder-only transformers for language modeling. The architecture used by GPT-4, Llama, Qwen, and all models in this book.*

---

**RoFormer: Enhanced Transformer with Rotary Position Embedding**
Su, J., Murtadha, A., Lu, Y., Pan, J., Wen, B., Liu, Y., Huang, S., Wen, Y. (2021).
https://arxiv.org/abs/2104.09864

*Introduced Rotary Positional Embeddings (RoPE) — the positional encoding used by Llama 3, Qwen2.5, DeepSeek-V3, and most modern LLMs.*

---

**GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**
Ainslie, J., Lee-Thorp, J., de Jong, M., Zeiler, M.G., Sanghai, S., Tay, Y. (2023).
https://arxiv.org/abs/2305.13245

*Introduced Grouped Query Attention (GQA) — the KV head reduction technique used in Llama 3.1, Qwen2.5, Mistral, and others. 8× KV memory reduction with minimal quality loss.*

---

**Root Mean Square Layer Normalization**
Zhang, B., Sennrich, R. (2019).
https://arxiv.org/abs/1910.07467

*Introduced RMSNorm — the normalization used by Llama, Qwen, and most modern LLMs. Faster than LayerNorm (no mean subtraction) with equivalent quality.*

---

**GLU Variants Improve Transformer**
Noam Shazeer (2020).
https://arxiv.org/abs/2002.05202

*Introduced SwiGLU, the activation function used in modern FFN layers (Llama, Qwen, PaLM, GPT-4).*

---

## I.2 Efficient Attention

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
Dao, T., Fu, D.Y., Ermon, S., Rudra, A., Ré, C. (2022).
*NeurIPS 2022.*
https://arxiv.org/abs/2205.14135

*Introduced Flash Attention: tiled attention computation using on-chip SRAM. Enables 128K+ context windows. Chapter 5.*

---

**FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
Dao, T. (2023).
https://arxiv.org/abs/2307.08691

*Improved Flash Attention with better GPU parallelism (2× speedup). The version implemented in vLLM.*

---

**FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**
Shah, J., Bikshandi, G., Zhang, Y., Thambidurai, D., Ramani, A., Dao, T. (2024).
https://arxiv.org/abs/2407.08608

*H100-specific Flash Attention using warp specialization and TMA. 1.5-2× speedup over Flash Attention-2.*

---

**Online normalizer calculation for softmax**
Milakov, M., Gimelshein, N. (2018).
https://arxiv.org/abs/1805.02867

*Introduced the online softmax algorithm used in Flash Attention. Appendix A.*

---

## I.3 KV Cache and Memory Management

**Efficient Memory Management for Large Language Model Serving with PagedAttention**
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., Stoica, I. (2023).
*SOSP 2023.*
https://arxiv.org/abs/2309.06180

*The vLLM paper. Introduced PagedAttention — the core KV cache management system. Chapter 6.*

---

**SGLang: Efficient Execution of Structured Language Model Programs**
Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C., Jain, S., Liang, Y., Goodman, J., Wang, X., Gonzalez, J.E., Stoica, I. (2023).
https://arxiv.org/abs/2312.07104

*Introduced RadixAttention for prefix caching. Chapter 11.*

---

**InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**
Lee, W., Moon, J., Kim, J. (2024).
https://arxiv.org/abs/2406.19707

*KV cache eviction and dynamic management for long context.*

---

## I.4 Quantization

**GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**
Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. (2022).
*ICLR 2023.*
https://arxiv.org/abs/2210.17323

*Introduced GPTQ — layer-wise quantization using second-order information. Chapter 10.*

---

**AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**
Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., Han, S. (2023).
https://arxiv.org/abs/2306.00978

*Introduced AWQ — channel-wise scaling for robust INT4 quantization. Chapter 10.*

---

**LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**
Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. (2022).
*NeurIPS 2022.*
https://arxiv.org/abs/2208.07339

*Mixed-precision decomposition handling outlier features in INT8 quantization.*

---

**FP8-LM: Training FP8 Large Language Models**
Peng, R., Li, Y., Gu, Q., Chen, W., Zhang, X., Zhou, S., Xiao, G., Hao, J., Lin, J., Yiu, S. (2023).
https://arxiv.org/abs/2310.18313

*Demonstrated FP8 training and inference. Foundation for H100 FP8 inference. Chapter 10, Chapter 37.*

---

**SqueezeLLM: Dense-and-Sparse Quantization**
Kim, S., Hooper, C., Gholami, A., Dong, Z., Li, X., Shen, S., Mahoney, M.W., Keutzer, K. (2023).
https://arxiv.org/abs/2306.07629

*Sparse-quantization combining INT4 with non-uniform precision for outliers.*

---

## I.5 Speculative Decoding

**Fast Inference from Transformers via Speculative Decoding**
Leviathan, Y., Kalman, M., Matias, Y. (2022). Google.
*ICML 2023.*
https://arxiv.org/abs/2211.17192

*Introduced speculative decoding (drafter-verifier). Chapter 23.*

---

**Accelerating Large Language Model Decoding with Speculative Sampling**
Chen, C., Borgeaud, S., Irving, G., Lespiau, J.B., Sifre, L., Jumper, J. (2023). DeepMind.
https://arxiv.org/abs/2302.01318

*Parallel formulation of speculative decoding with rejection sampling proof of exact distribution preservation.*

---

**Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**
Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J.D., Chen, D., Dao, T. (2024).
https://arxiv.org/abs/2401.10774

*Multiple draft heads attached directly to the target model. No separate draft model needed.*

---

## I.6 Distributed and Disaggregated Serving

**Orca: A Distributed Serving System for Transformer-Based Generative Models**
Yu, G., Kim, J., Jeong, H., Cho, G., Lee, S., Ko, J., Shin, J., Won, Y. (2022).
*OSDI 2022.*
https://www.usenix.org/conference/osdi22/presentation/yu

*Introduced continuous batching (iteration-level scheduling). The conceptual predecessor to vLLM's scheduler.*

---

**Splitwise: Efficient Generative LLM Inference Using Phase Splitting**
Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, Í., Maleki, S., Bianchini, R. (2024). Microsoft.
https://arxiv.org/abs/2311.18677

*Formal analysis of disaggregated prefill/decode. Chapter 18.*

---

**DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving**
Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., Guo, Y. (2024).
*OSDI 2024.*
https://arxiv.org/abs/2401.09670

*Production disaggregated serving system. Chapter 18.*

---

**Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**
Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., Catanzaro, B. (2019). NVIDIA.
https://arxiv.org/abs/1909.08053

*Introduced tensor parallelism for transformers (column-parallel + row-parallel linear). Chapter 15.*

---

## I.7 Long Context

**YaRN: Efficient Context Window Extension of Large Language Models**
Peng, B., Quesnelle, J., Fan, H., Shippole, E. (2023).
https://arxiv.org/abs/2309.00071

*Context extension via RoPE frequency adjustment. Chapter 27.*

---

**LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens**
Ding, Y., Zhang, L., Zhang, S., Xu, Y., Shang, L., Zhao, H., Yang, Y., Yang, M. (2024). Microsoft.
https://arxiv.org/abs/2402.13753

*Progressive context extension to 2M tokens.*

---

## I.8 Mixture of Experts

**Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**
Fedus, W., Zoph, B., Shazeer, N. (2021). Google.
*Journal of Machine Learning Research.*
https://arxiv.org/abs/2101.03961

*Introduced top-1 MoE routing with auxiliary load balancing loss. Chapter 15.*

---

**GLaM: Efficient Scaling of Language Models with Mixture-of-Experts**
Du, N., Huang, Y., Dai, A.M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A.W., Firat, O., Zoph, B., Dean, J., Le, Q.V. (2021). Google.
https://arxiv.org/abs/2112.06905

*MoE with 1.2T parameters (97B active). Demonstrated MoE quality at scale.*

---

## I.9 Specific Models

**DeepSeek-V3 Technical Report**
DeepSeek-AI (2024).
https://arxiv.org/abs/2412.19437

*Multi-head Latent Attention (MLA), 256-expert MoE, auxiliary-free load balancing, FP8 training. Chapter 34.*

---

**DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**
DeepSeek-AI (2025).
https://arxiv.org/abs/2501.12948

*Reasoning model via RL. Long chain-of-thought generation.*

---

**Qwen2.5 Technical Report**
Qwen Team (2024). Alibaba.
https://arxiv.org/abs/2412.15115

*0.5B–72B model family with 152K vocabulary, GQA, SwiGLU. Chapter 35.*

---

**Qwen3 Technical Report**
Qwen Team (2025). Alibaba.
https://arxiv.org/abs/2505.09388

*Thinking/non-thinking modes, 0.6B–235B MoE. Chapter 35.*

---

**Kimi k1.5: Scaling Reinforcement Learning with LLMs**
Moonshot AI (2025).
https://arxiv.org/abs/2501.12599

*Long context RL training and Moon-Cache infrastructure. Chapter 36.*

---

**Llama 3 Model Card**
Meta AI (2024).
https://github.com/meta-llama/llama3

*Llama 3.1 70B/405B architecture, 128K context, GQA. Referenced throughout.*

---

**Llama-3.1-Nemotron Technical Overview**
NVIDIA (2024).
https://developer.nvidia.com/blog/llama-3-1-nemotron-70b

*NVIDIA's RLHF-tuned Llama variant. Chapter 37.*

---

## I.10 Systems and Infrastructure

**Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning**
Zheng, L., Li, Z., Zhang, H., Zhuang, Y., Chen, E.P., Huang, Y., Wang, Y., Xu, Y., Zhuo, D., Xing, E.P., Gonzalez, J.E., Stoica, I. (2022).
*OSDI 2022.*
https://arxiv.org/abs/2201.12023

*Automatic parallelism planning for distributed model serving.*

---

**Ray: A Distributed Framework for Emerging AI Applications**
Moritz, P., Nishihara, R., Wang, S., Tumanov, A., Liaw, R., Liang, E., Elibol, M., Yang, L., Paul, W., Jordan, M.I., Stoica, I. (2018).
*OSDI 2018.*
https://arxiv.org/abs/1712.05889

*Ray: the distributed computing framework underlying vLLM's distributed execution. Chapter 19.*

---

**CUDA Programming Guide**
NVIDIA (2024).
https://docs.nvidia.com/cuda/cuda-c-programming-guide/

*The authoritative CUDA reference. Appendix J.*

---

**Roofline: An Insightful Visual Performance Model for Multicore Architectures**
Williams, S., Waterman, A., Patterson, D. (2009).
*Communications of the ACM.*
https://doi.org/10.1145/1498765.1498785

*The Roofline model for hardware performance analysis. Chapter 2, Appendix A.*

---

## I.11 Benchmarking and Evaluation

**Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference**
Chiang, W., Zheng, L., Sheng, Y., Angelopoulos, A.N., Li, T., Li, D., Zhang, H., Zhu, B., Jordan, M., Gonzalez, J.E., Stoica, I. (2024).
https://arxiv.org/abs/2403.04132

*Human preference evaluation of LLMs. Used to validate model quality post-quantization.*

---

**Perplexity (PPL)**
See: Jurafsky, D., Martin, J.H. *Speech and Language Processing* (3rd ed., draft).
https://web.stanford.edu/~jurafsky/slp3/

*Standard reference for language model evaluation metrics including perplexity.*

---

## I.12 Online Resources

**vLLM Documentation**
https://docs.vllm.ai/

**llama.cpp GitHub Repository**
https://github.com/ggml-org/llama.cpp

**Hugging Face Model Hub**
https://huggingface.co/models

**NVIDIA Developer Blog — LLM Inference**
https://developer.nvidia.com/blog/tag/llm-inference/

**Transformer Explainer (visual)**
https://poloclub.github.io/transformer-explainer/

**LLM Visualization**
https://bbycroft.net/llm

**Andrej Karpathy — Let's build GPT from scratch (video)**
https://www.youtube.com/watch?v=kCc8FmEb1nY

**Stanford CS336: Language Modeling from Scratch**
https://stanford-cs336.github.io/spring2024/

**Lilian Weng — Large Transformer Model Inference Optimization**
https://lilianweng.github.io/posts/2023-01-10-inference-optimization/

---

## I.13 Citation Index by Chapter

| Chapter | Key Papers |
|---|---|
| Ch. 5 (Flash Attention) | Dao 2022, Dao 2023, Dao 2024, Milakov 2018 |
| Ch. 6 (PagedAttention) | Kwon 2023 |
| Ch. 7 (Scheduler) | Yu 2022 (Orca) |
| Ch. 10 (Quantization) | Frantar 2022 (GPTQ), Lin 2023 (AWQ), Dettmers 2022 (INT8), Peng 2023 (FP8) |
| Ch. 11 (Prefix Caching) | Zheng 2023 (SGLang/RadixAttention) |
| Ch. 15 (Multi-GPU) | Shoeybi 2019 (Megatron), Fedus 2021 (Switch), Moritz 2018 (Ray) |
| Ch. 18 (Disaggregated) | Patel 2024 (Splitwise), Zhong 2024 (DistServe) |
| Ch. 23 (Speculative) | Leviathan 2022, Chen 2023, Cai 2024 (Medusa) |
| Ch. 27 (Long Context) | Peng 2023 (YaRN), Ding 2024 (LongRoPE) |
| Ch. 34 (DeepSeek) | DeepSeek-AI 2024 (V3), DeepSeek-AI 2025 (R1) |
| Ch. 35 (Qwen) | Qwen Team 2024 (2.5), Qwen Team 2025 (3) |
| Ch. 36 (Kimi) | Moonshot AI 2025 |
| Ch. 37 (Nemotron) | NVIDIA 2024 |
| Appendix A (Math) | Vaswani 2017, Su 2021 (RoPE), Ainslie 2023 (GQA), Zhang 2019 (RMS) |
| Appendix J (CUDA) | NVIDIA CUDA Guide, Williams 2009 (Roofline) |
