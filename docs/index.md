---
hide:
  - navigation
  - toc
---

<div class="book-page">

<!-- ── HERO / COVER ───────────────────────────────────────────── -->
<div class="book-cover-row">

  <div class="book-spine-wrap">
    <div class="book-cover">
      <div class="book-cover-inner">
        <div class="book-label">Production LLM Inference Engineering</div>
        <div class="book-title">Close to<br>the Metal</div>
        <div class="book-rule"></div>
        <div class="book-engines">LLM Inference from First Principles</div>
        <div class="book-cover-footer">
          <div class="book-authors">Oluwaseyi Awoga</div>
          <div class="book-edition">52 Chapters · 26 Appendices · May 2026</div>
        </div>
      </div>
    </div>
  </div>

  <div class="book-blurb">
    <p class="book-quote">"The gap between 'I called the API' and 'I understand what just happened' is exactly the size of this book."</p>
    <p class="book-desc">A systems engineering textbook for engineers who serve LLMs in production. Core concepts are taught through two primary engines — <strong>vLLM</strong> and <strong>llama.cpp</strong> — with later chapters extending to <strong>SGLang</strong>, <strong>TensorRT-LLM</strong>, <strong>MLC-LLM</strong>, and <strong>Ollama</strong>. Every chapter includes worked arithmetic, companion code, and production-grade configurations.</p>
    <div class="book-metrics">
      <div class="book-metric"><span class="bm-n">52</span><span class="bm-l">chapters</span></div>
      <div class="book-metric"><span class="bm-n">28</span><span class="bm-l">appendices</span></div>
      <div class="book-metric"><span class="bm-n">48</span><span class="bm-l">code demos</span></div>
      <div class="book-metric"><span class="bm-n">11×</span><span class="bm-l">cost savings</span></div>
    </div>
    <div class="book-cta-row">
      <a href="part1-foundations/chapter01.html" class="bcta-primary">Start Reading — Chapter 1 →</a>
      <a href="preface.html" class="bcta-secondary">Read the Preface</a>
    </div>
  </div>

</div>

<!-- ── CASE STUDY BANNER ─────────────────────────────────────── -->
<div class="case-banner">
  <div class="case-text">
    <strong>The Running Case Study</strong> — A service at 50,000 concurrent users, costing <span class="case-before">$1.2M/month</span> at 28% GPU utilization. By Chapter 38, systematic application of the techniques in this book reduces the bill to <span class="case-after">$108K/month</span>. No new hardware. No new model.
  </div>
  <div class="case-numbers">
    <div class="cn-before">$1.2M<small>/mo</small></div>
    <div class="cn-arrow">→ 11×</div>
    <div class="cn-after">$108K<small>/mo</small></div>
  </div>
</div>

<!-- ── TABLE OF CONTENTS ─────────────────────────────────────── -->
<div class="toc-section">
  <h2 class="toc-heading">Table of Contents</h2>

  <div class="toc-parts">

    <div class="toc-part">
      <div class="toc-part-header toc-pi">
        <span class="toc-part-label">Part I</span>
        <span class="toc-part-title">Foundations</span>
        <span class="toc-part-range">Ch 1–5</span>
      </div>
      <div class="toc-entries">
        <a href="part1-foundations/chapter01.html" class="toc-entry"><span class="toc-num">1</span><span class="toc-etitle">Two Engines, One Problem</span></a>
        <a href="part1-foundations/chapter02.html" class="toc-entry"><span class="toc-num">2</span><span class="toc-etitle">The GPU and CPU Memory Landscapes</span></a>
        <a href="part1-foundations/chapter02b.html" class="toc-entry"><span class="toc-num">2.5</span><span class="toc-etitle">GPU Memory Architecture — Registers, Shared Memory, Caches, and Global Memory</span></a>
        <a href="part1-foundations/chapter03.html" class="toc-entry"><span class="toc-num">3</span><span class="toc-etitle">Tokens, Sequences, and the Batch</span></a>
        <a href="part1-foundations/chapter03b.html" class="toc-entry"><span class="toc-num">3.5</span><span class="toc-etitle">The Transformer Block</span></a>
        <a href="part1-foundations/chapter04.html" class="toc-entry"><span class="toc-num">4</span><span class="toc-etitle">Attention Mechanics</span></a>
        <a href="part1-foundations/chapter04b.html" class="toc-entry"><span class="toc-num">4.5</span><span class="toc-etitle">Attention Alternatives — Sliding Window, Linear Attention, SSMs, and Mamba</span></a>
        <a href="part1-foundations/chapter05.html" class="toc-entry"><span class="toc-num">5</span><span class="toc-etitle">Flash Attention — Tiling and Recomputation</span></a>
      </div>
    </div>

    <div class="toc-part">
      <div class="toc-part-header toc-pii">
        <span class="toc-part-label">Part II</span>
        <span class="toc-part-title">Engine Internals</span>
        <span class="toc-part-range">Ch 6–13</span>
      </div>
      <div class="toc-entries">
        <a href="part2-engine-internals/chapter06.html" class="toc-entry"><span class="toc-num">6</span><span class="toc-etitle">PagedAttention and the Block Manager</span></a>
        <a href="part2-engine-internals/chapter07.html" class="toc-entry"><span class="toc-num">7</span><span class="toc-etitle">The Scheduler and Request Lifecycle</span></a>
        <a href="part2-engine-internals/chapter07b.html" class="toc-entry"><span class="toc-num">7.5</span><span class="toc-etitle">Continuous Batching — The Iteration-Level Scheduling Loop</span></a>
        <a href="part2-engine-internals/chapter08.html" class="toc-entry"><span class="toc-num">8</span><span class="toc-etitle">Startup and Initialization</span></a>
        <a href="part2-engine-internals/chapter08b.html" class="toc-entry"><span class="toc-num">8.5</span><span class="toc-etitle">CUDA Graphs — Capture, Replay, and Production Latency</span></a>
        <a href="part2-engine-internals/chapter09.html" class="toc-entry"><span class="toc-num">9</span><span class="toc-etitle">The Forward Pass — CUDA vs. GGML</span></a>
        <a href="part2-engine-internals/chapter10.html" class="toc-entry"><span class="toc-num">10</span><span class="toc-etitle">Quantization Internals — GGUF, AWQ, FP8</span></a>
        <a href="part2-engine-internals/chapter11.html" class="toc-entry"><span class="toc-num">11</span><span class="toc-etitle">Prefill, Chunked Prefill, and Prompt Caching</span></a>
        <a href="part2-engine-internals/chapter11b.html" class="toc-entry"><span class="toc-num">11.5</span><span class="toc-etitle">KV Cache Eviction — Attention Sinks, H2O, and SnapKV</span></a>
        <a href="part2-engine-internals/chapter11c.html" class="toc-entry"><span class="toc-num">11.6</span><span class="toc-etitle">RadixAttention and Prefix Caching Deep Dive</span></a>
        <a href="part2-engine-internals/chapter12.html" class="toc-entry"><span class="toc-num">12</span><span class="toc-etitle">Sampling — From Logits to Tokens</span></a>
        <a href="part2-engine-internals/chapter12b.html" class="toc-entry"><span class="toc-num">12.5</span><span class="toc-etitle">Structured Generation and Constrained Decoding</span></a>
        <a href="part2-engine-internals/chapter13.html" class="toc-entry"><span class="toc-num">13</span><span class="toc-etitle">Token Streaming — The Last Mile</span></a>
      </div>
    </div>

    <div class="toc-part">
      <div class="toc-part-header toc-piii">
        <span class="toc-part-label">Part III</span>
        <span class="toc-part-title">Production Configuration</span>
        <span class="toc-part-range">Ch 14–21</span>
      </div>
      <div class="toc-entries">
        <a href="part3-production/chapter14.html" class="toc-entry"><span class="toc-num">14</span><span class="toc-etitle">The Eight vLLM Knobs + llama.cpp Equivalents</span></a>
        <a href="part3-production/chapter15.html" class="toc-entry"><span class="toc-num">15</span><span class="toc-etitle">Multi-GPU Serving and Tensor Parallelism</span></a>
        <a href="part3-production/chapter15b.html" class="toc-entry"><span class="toc-num">15.5</span><span class="toc-etitle">Flash Decoding and Context Parallelism</span></a>
        <a href="part3-production/chapter16.html" class="toc-entry"><span class="toc-num">16</span><span class="toc-etitle">Observability — Metrics, Logging, Tracing</span></a>
        <a href="part3-production/chapter17.html" class="toc-entry"><span class="toc-num">17</span><span class="toc-etitle">Benchmarking — Fair Comparisons Between Engines</span></a>
        <a href="part3-production/chapter18.html" class="toc-entry"><span class="toc-num">18</span><span class="toc-etitle">Disaggregated Prefill and Decode</span></a>
        <a href="part3-production/chapter19.html" class="toc-entry"><span class="toc-num">19</span><span class="toc-etitle">Kubernetes and KubeRay Auto-Scaling</span></a>
        <a href="part3-production/chapter20.html" class="toc-entry"><span class="toc-num">20</span><span class="toc-etitle">Cost Engineering — $/Million Tokens</span></a>
        <a href="part3-production/chapter21.html" class="toc-entry"><span class="toc-num">21</span><span class="toc-etitle">Security — API Hardening and Injection Defense</span></a>
      </div>
    </div>

    <div class="toc-part">
      <div class="toc-part-header toc-piv">
        <span class="toc-part-label">Part IV</span>
        <span class="toc-part-title">Advanced Techniques</span>
        <span class="toc-part-range">Ch 22–33</span>
      </div>
      <div class="toc-entries">
        <a href="part4-advanced/chapter22.html" class="toc-entry"><span class="toc-num">22</span><span class="toc-etitle">LoRA Serving and Adapter Hot-Swapping</span></a>
        <a href="part4-advanced/chapter23.html" class="toc-entry"><span class="toc-num">23</span><span class="toc-etitle">Speculative Decoding</span></a>
        <a href="part4-advanced/chapter24.html" class="toc-entry"><span class="toc-num">24</span><span class="toc-etitle">Reasoning Model Inference</span></a>
        <a href="part4-advanced/chapter25.html" class="toc-entry"><span class="toc-num">25</span><span class="toc-etitle">RL and Serving Policies</span></a>
        <a href="part4-advanced/chapter26.html" class="toc-entry"><span class="toc-num">26</span><span class="toc-etitle">CS336 Alignment Field Guide</span></a>
        <a href="part4-advanced/chapter27.html" class="toc-entry"><span class="toc-num">27</span><span class="toc-etitle">Long-Context Inference — 128K and Beyond</span></a>
        <a href="part4-advanced/chapter28.html" class="toc-entry"><span class="toc-num">28</span><span class="toc-etitle">llama.cpp as a Platform</span></a>
        <a href="part4-advanced/chapter29.html" class="toc-entry"><span class="toc-num">29</span><span class="toc-etitle">Multimodal Inference — Vision and Audio</span></a>
        <a href="part4-advanced/chapter30.html" class="toc-entry"><span class="toc-num">30</span><span class="toc-etitle">Semantic Caching and Response Reuse</span></a>
        <a href="part4-advanced/chapter31.html" class="toc-entry"><span class="toc-num">31</span><span class="toc-etitle">Model Routing and Cascading</span></a>
        <a href="part4-advanced/chapter32.html" class="toc-entry"><span class="toc-num">32</span><span class="toc-etitle">Debugging Inference Systems</span></a>
        <a href="part4-advanced/chapter33.html" class="toc-entry"><span class="toc-num">33</span><span class="toc-etitle">The Full Engine Landscape — 2026</span></a>
        <a href="part4-advanced/chapter33b.html" class="toc-entry"><span class="toc-num">33.5</span><span class="toc-etitle">Choosing Your Engine — SGLang, TRT-LLM, MLC-LLM, Ollama</span></a>
      </div>
    </div>

    <div class="toc-part">
      <div class="toc-part-header toc-pv">
        <span class="toc-part-label">Part V</span>
        <span class="toc-part-title">The Model Zoo</span>
        <span class="toc-part-range">Ch 34–42</span>
      </div>
      <div class="toc-entries">
        <a href="part5-model-zoo/chapter34.html" class="toc-entry"><span class="toc-num">34</span><span class="toc-etitle">DeepSeek — MLA, MoE, and FP8 at Scale</span></a>
        <a href="part5-model-zoo/chapter35.html" class="toc-entry"><span class="toc-num">35</span><span class="toc-etitle">Qwen — Multilingual and Long-Context</span></a>
        <a href="part5-model-zoo/chapter36.html" class="toc-entry"><span class="toc-num">36</span><span class="toc-etitle">Kimi — Long-Context and Moon-Cache</span></a>
        <a href="part5-model-zoo/chapter37.html" class="toc-entry"><span class="toc-num">37</span><span class="toc-etitle">Nemotron and TensorRT-LLM</span></a>
        <a href="part5-model-zoo/chapter38.html" class="toc-entry"><span class="toc-num">38</span><span class="toc-etitle">The Production Synthesis — $1.2M → $108K</span></a>
        <a href="part5-model-zoo/chapter39.html" class="toc-entry"><span class="toc-num">39</span><span class="toc-etitle">Evaluation and Regression Testing</span></a>
        <a href="part5-model-zoo/chapter40.html" class="toc-entry"><span class="toc-num">40</span><span class="toc-etitle">The vLLM V1 Architecture — Three-Process Design</span></a>
        <a href="part5-model-zoo/chapter41.html" class="toc-entry"><span class="toc-num">41</span><span class="toc-etitle">Meta Llama 3 — Architecture, Ecosystem, and Inference</span></a>
        <a href="part5-model-zoo/chapter42.html" class="toc-entry"><span class="toc-num">42</span><span class="toc-etitle">Phi-4 and Gemma 3 — Small Models, Large Impact</span></a>
      </div>
    </div>

    <div class="toc-part">
      <div class="toc-part-header toc-pa">
        <span class="toc-part-label">App.</span>
        <span class="toc-part-title">Appendices A – Z</span>
        <span class="toc-part-range">Reference</span>
      </div>
      <div class="toc-entries">
        <a href="appendices/appendix-a.html" class="toc-entry"><span class="toc-num">A</span><span class="toc-etitle">Mathematical Foundations</span></a>
        <a href="appendices/appendix-a2.html" class="toc-entry"><span class="toc-num">A.2</span><span class="toc-etitle">Tensor Contractions — 2D/3D/5D/ND with CUDA, Triton, CUTLASS, Mojo — Manual Worked Examples</span></a>
        <a href="appendices/appendix-a3.html" class="toc-entry"><span class="toc-num">A.3</span><span class="toc-etitle">The Chain Rule — From Scalars to Transformer Backpropagation</span></a>
        <a href="appendices/appendix-b.html" class="toc-entry"><span class="toc-num">B</span><span class="toc-etitle">Installation Guide — vLLM and llama.cpp</span></a>
        <a href="appendices/appendix-c.html" class="toc-entry"><span class="toc-num">C</span><span class="toc-etitle">PyTorch for LLM Inference — Tensors, Devices, Compile, Quantization</span></a>
        <a href="appendices/appendix-d.html" class="toc-entry"><span class="toc-num">D</span><span class="toc-etitle">vLLM EngineArgs Complete Reference</span></a>
        <a href="appendices/appendix-e.html" class="toc-entry"><span class="toc-num">E</span><span class="toc-etitle">llama.cpp CLI Flag Reference</span></a>
        <a href="appendices/appendix-f.html" class="toc-entry"><span class="toc-num">F</span><span class="toc-etitle">Production Templates — Docker, YAML, nginx</span></a>
        <a href="appendices/appendix-g.html" class="toc-entry"><span class="toc-num">G</span><span class="toc-etitle">Benchmarking Reference — Metrics and Methodology</span></a>
        <a href="appendices/appendix-h.html" class="toc-entry"><span class="toc-num">H</span><span class="toc-etitle">Operational Decision Tree and Troubleshooting Guide</span></a>
        <a href="appendices/appendix-i.html" class="toc-entry"><span class="toc-num">I</span><span class="toc-etitle">C++ Build Patterns for Inference Systems</span></a>
        <a href="appendices/appendix-j.html" class="toc-entry"><span class="toc-num">J</span><span class="toc-etitle">libtorch — The C++ API for Production Inference</span></a>
        <a href="appendices/appendix-k.html" class="toc-entry"><span class="toc-num">K</span><span class="toc-etitle">std::mdspan for CPU Inference — Multidimensional Views in C++23</span></a>
        <a href="appendices/appendix-l.html" class="toc-entry"><span class="toc-num">L</span><span class="toc-etitle">CUDA C++ Introduction for Inference Engineers</span></a>
        <a href="appendices/appendix-m.html" class="toc-entry"><span class="toc-num">M</span><span class="toc-etitle">Introduction to Triton — Python-Embedded GPU Kernel Programming</span></a>
        <a href="appendices/appendix-n.html" class="toc-entry"><span class="toc-num">N</span><span class="toc-etitle">CUTLASS and Tensor Cores — The Compiled Performance Layer</span></a>
        <a href="appendices/appendix-o.html" class="toc-entry"><span class="toc-num">O</span><span class="toc-etitle">Introduction to Mojo — Systems Performance in Python Syntax</span></a>
        <a href="appendices/appendix-p.html" class="toc-entry"><span class="toc-num">P</span><span class="toc-etitle">ROCm and AMD GPU Inference</span></a>
        <a href="appendices/appendix-q.html" class="toc-entry"><span class="toc-num">Q</span><span class="toc-etitle">Mobile and Edge Deployment — Android and Apple Silicon</span></a>
        <a href="appendices/appendix-r.html" class="toc-entry"><span class="toc-num">R</span><span class="toc-etitle">Edge Inference on Linux SBCs — Raspberry Pi and NVIDIA Jetson</span></a>
        <a href="appendices/appendix-s.html" class="toc-entry"><span class="toc-num">S</span><span class="toc-etitle">CI/CD Pipelines for LLM Inference Systems</span></a>
        <a href="appendices/appendix-t.html" class="toc-entry"><span class="toc-num">T</span><span class="toc-etitle">Embedding and Reranker Model Serving</span></a>
        <a href="appendices/appendix-u.html" class="toc-entry"><span class="toc-num">U</span><span class="toc-etitle">Quantization Calibration Workflow — AWQ, GPTQ, FP8, and GGUF</span></a>
        <a href="appendices/appendix-v.html" class="toc-entry"><span class="toc-num">V</span><span class="toc-etitle">TurboQuant — Online Vector Quantization for KV Cache</span></a>
        <a href="appendices/appendix-w.html" class="toc-entry"><span class="toc-num">W</span><span class="toc-etitle">Glossary — 70+ Terms Defined</span></a>
        <a href="appendices/appendix-x.html" class="toc-entry"><span class="toc-num">X</span><span class="toc-etitle">References — 40+ Key Papers</span></a>
        <a href="appendices/appendix-y.html" class="toc-entry"><span class="toc-num">Y</span><span class="toc-etitle">Cerebras WSE-3 — Wafer-Scale Inference</span></a>
        <a href="appendices/appendix-z.html" class="toc-entry"><span class="toc-num">Z</span><span class="toc-etitle">JAX — XLA-Native Python for LLM Inference</span></a>
      </div>
    </div>

  </div>
</div>

<!-- ── FOOTER NOTE ───────────────────────────────────────────── -->
<div class="book-footer">
  <em>"The first time you see a KV cache eviction cause a 40-second request, you will wish you had read Chapter 6 before it happened. This book is here so you read it first."</em>
  <br><strong>— Oluwaseyi Awoga, written with AI assistance, May 2026 (Under development)</strong>
</div>

</div>
