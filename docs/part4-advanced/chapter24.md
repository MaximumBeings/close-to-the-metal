# Chapter 24: Reasoning Models — From RLHF to o1, R1, and the Long-Decode Problem

> "Supervised fine-tuning teaches a model what humans did. Reinforcement learning teaches it
> what humans *should have* done — and occasionally discovers things humans never tried."
>
> — Paraphrase of the core insight behind InstructGPT, 2022

---

## Why This Chapter Exists

Every model in this book so far was trained with **next-token prediction on human text**. That
ceiling is the quality of whatever humans wrote down. Reasoning models break through that ceiling
by using reinforcement learning to let the model discover reasoning strategies that produce
correct answers — strategies that may never appear verbatim in any human-written corpus.

Understanding how reasoning models are *trained* is not separate from understanding how to
*serve* them. The training procedure determines the output distribution, the token budget
behavior, the KV cache growth profile, and the cost model. This chapter covers the full
vertical: from the policy gradient mathematics to the production serving decisions that follow
from them.

---

## 24.1 The Supervised Learning Ceiling

`[FOUNDATIONAL]`

Standard LLM training uses **supervised fine-tuning (SFT)**: given a prompt, predict the next
token from a human-written response. The loss is cross-entropy against the reference distribution.

```
  SFT objective:
    L_SFT = -E[log π_θ(a | s)]

  where:
    π_θ  = model policy parameterised by weights θ
    a    = token from the reference (human-written) response
    s    = preceding context (prompt + previous tokens)
```

SFT is powerful but has a hard ceiling: **the model can only learn to imitate what it has seen**.
For tasks requiring multi-step reasoning, the training data may contain correct final answers but
wrong or missing intermediate steps. The model learns to pattern-match to answers, not to reason.

```
  Example failure — SFT model on math:
    Training data: "Q: 17 × 23? A: 391"
    Model learns: "after arithmetic question, output plausible-looking number"
    Model on "17 × 29": outputs "493" (a plausible but wrong number)

  The model has memorised the *format* of arithmetic answers, not arithmetic.
```

Reinforcement learning addresses this by using *outcome feedback* — whether the final answer is
correct — to reward the model for any reasoning path that produces right answers, even if that
path never appeared in training data.

---

## 24.2 Reinforcement Learning Foundations

`[FOUNDATIONAL]`

### The RL Framing for Language Models

Language model inference maps naturally to a Markov Decision Process (MDP):

```
  State    s_t  = prompt + tokens generated so far
  Action   a_t  = next token chosen from vocabulary
  Policy   π_θ  = the LLM; maps state → probability distribution over tokens
  Reward   R    = scalar signal after the full sequence is generated
  Episode  = one complete generation (prompt → final token)

  Goal: find θ that maximizes expected reward E[R(s_0, a_0, a_1, ..., a_T)]
```

The key difference from supervised learning: the reward arrives **after the full sequence**, not
after each token. The model must figure out which of the thousands of token choices it made
contributed to the good or bad outcome — the **credit assignment problem**.

### Policy Gradient: REINFORCE

The simplest policy gradient algorithm is REINFORCE (Williams, 1992):

```
  ∇_θ J(θ) = E[∑_t ∇_θ log π_θ(a_t | s_t) × R]

  In words: increase the log-probability of each token in the sequence
            by an amount proportional to the total reward received.

  Update rule:
    θ ← θ + α × ∑_t ∇_θ log π_θ(a_t | s_t) × R

  Intuition:
    If the full answer was correct (R > 0), nudge the model to be
    slightly more likely to generate this sequence again.
    If incorrect (R < 0), nudge it away.
```

REINFORCE has high variance — a single good generation could be lucky, not skillful. The fix
is a **baseline** that subtracts the average expected reward:

```
  ∇_θ J(θ) = E[∑_t ∇_θ log π_θ(a_t | s_t) × (R - b)]

  b = baseline (typically the value function V(s_t))
  (R - b) = "advantage": how much better was this action than average?
```

---

## 24.3 RLHF: The First Generation

`[DEEP DIVE]`

**Reinforcement Learning from Human Feedback (RLHF)** — introduced in InstructGPT (Ouyang et al.,
2022) — is the training recipe that turned GPT-3 into ChatGPT. It has three stages:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  RLHF Pipeline                                                  │
  │                                                                 │
  │  Stage 1: Supervised Fine-Tuning (SFT)                         │
  │    Base model + human-written demonstrations → SFT model       │
  │    Standard cross-entropy loss on demonstration data            │
  │                                                                 │
  │  Stage 2: Reward Model Training                                 │
  │    Human labellers compare pairs of SFT outputs:               │
  │    "Which response is better — A or B?"                        │
  │    Train reward model R_φ to predict human preference:         │
  │    L_RM = -E[log σ(R_φ(y_w) - R_φ(y_l))]                     │
  │    where y_w = preferred output, y_l = rejected output         │
  │                                                                 │
  │  Stage 3: RL Fine-Tuning (PPO)                                  │
  │    Use reward model as reward signal.                           │
  │    Policy π_θ generates outputs, R_φ scores them.              │
  │    PPO updates π_θ to maximize R_φ(output) while staying       │
  │    close to the SFT model (KL penalty).                         │
  └─────────────────────────────────────────────────────────────────┘
```

### Proximal Policy optimization (PPO)

PPO (Schulman et al., 2017) is the RL algorithm used in RLHF. It addresses REINFORCE's high
variance and instability through **clipped surrogate objectives**:

```
  PPO objective:
    L_PPO = E[min(r_t(θ) × A_t,  clip(r_t(θ), 1-ε, 1+ε) × A_t)]

  where:
    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
    A_t    = advantage estimate at step t
    ε      = clip threshold (typically 0.2)

  The clip prevents the policy from moving too far from the old policy
  in a single update — the "proximal" in PPO.
```

PPO for LLMs requires four models simultaneously:

```
  ┌────────────────────────────────────────────────────────────────────┐
  │  PPO Training Infrastructure for LLMs                             │
  │                                                                    │
  │  ┌──────────────┐   ┌──────────────┐   ┌──────────┐  ┌─────────┐ │
  │  │ Actor        │   │ Critic       │   │ Reward   │  │ Ref     │ │
  │  │ (π_θ)        │   │ (V_ψ)        │   │ Model    │  │ Model   │ │
  │  │ Generates    │   │ Estimates    │   │ (R_φ)    │  │ (π_SFT) │ │
  │  │ responses    │   │ value V(s_t) │   │ Scores   │  │ KL base │ │
  │  │ Updated      │   │ for advantage│   │ response │  │ Not     │ │
  │  │ every step   │   │ Updated each │   │ Frozen   │  │ updated │ │
  │  └──────────────┘   │ step         │   └──────────┘  └─────────┘ │
  │                     └──────────────┘                              │
  │                                                                    │
  │  Total memory: 4× model size minimum (often 8× with gradients)   │
  │  For a 7B model: 4 × 14 GB BF16 = 56 GB minimum                 │
  └────────────────────────────────────────────────────────────────────┘
```

The KL penalty keeps the RL-trained model from drifting too far from the SFT baseline (which
would cause it to "reward hack" — find degenerate outputs that score well on R_φ but are
useless in practice):

```
  Full RLHF reward:
    R_total = R_φ(response) - β × KL(π_θ || π_SFT)

  β = KL coefficient (typically 0.01–0.1)
  KL term penalises large deviations from the SFT model's distribution
```

### The Cost of RLHF

```
  Human labeller cost (preference pairs):
    GPT-4 level: ~1M preference pairs × $0.50/pair = $500,000
    Per labeller session: 4–8 hours, 200–400 comparisons/hour

  Compute for PPO training:
    7B model: ~64 A100s for 2–4 days ≈ $15,000–$30,000
    70B model: ~512 A100s for 1 week ≈ $500,000+

  Reward model quality is the ceiling:
    If R_φ is miscalibrated, the actor optimizes for proxy rewards,
    not human intent. "Reward hacking" degrades quality despite
    high RL reward scores.
```

---

## 24.4 From RLHF to Reasoning: Process vs. Outcome Rewards

`[DEEP DIVE]`

RLHF rewards *style* — whether a response sounds helpful, harmless, honest to a human labeller.
Reasoning models need to reward *correctness* — whether the final answer is right. This shift
enables **verifiable rewards**, which eliminates the need for a human-trained reward model.

### Outcome Reward Models (ORM)

An ORM scores only the final answer:

```
  R_ORM(response) = 1 if final_answer == ground_truth else 0
                    (or a continuous score for partial credit)

  Advantage: perfectly verifiable for math, code, logic puzzles
  Disadvantage: sparse signal — long chains of reasoning only get
                rewarded at the very end
```

### Process Reward Models (PRM)

A PRM scores each intermediate reasoning step:

```
  R_PRM(step_1, step_2, ..., step_N) = sum of per-step scores

  step score = {1: correct step, 0: neutral, -1: incorrect step}

  Advantage: dense signal — model gets feedback at each reasoning step
  Disadvantage: requires step-level human annotations (expensive)
                or a trained process reward model (can be wrong)

  Used in: Let's Verify Step by Step (Lightman et al., 2023)
           OpenAI's o1 training (partially disclosed)
```

### The Sparse Reward Problem

For a 50-step proof, REINFORCE with ORM gives exactly one bit of signal for thousands of token
choices. Most of the reasoning tokens receive zero gradient signal in any individual episode:

```
  Reasoning trace: 5,000 tokens across 30 steps

  With ORM (outcome reward):
    Tokens 1–4,999: gradient = 0 (no signal yet)
    Token 5,000 (final answer): gradient ∝ (R - baseline)
    All 5,000 tokens get the *same* reward signal — correct or not

  With PRM (process reward):
    After step 1 (~150 tokens): gradient ∝ R_step1
    After step 2 (~150 tokens): gradient ∝ R_step2
    ...
    Much denser signal, better credit assignment
```

---

## 24.5 GRPO: DeepSeek's Efficient Alternative to PPO

`[DEEP DIVE]`

DeepSeek-R1's key training innovation is **Group Relative Policy optimization (GRPO)** (Shao et
al., 2024). GRPO eliminates the critic model from PPO, cutting memory requirements nearly in half.

### The Key Insight

Instead of estimating the advantage A_t using a learned value function V_ψ(s_t), GRPO estimates
it from a *group* of sampled responses to the same prompt:

```
  GRPO procedure for one prompt:
    1. Sample G responses from the current policy:
       {y_1, y_2, ..., y_G} ~ π_θ(· | prompt)

    2. Score each response with the reward model:
       {r_1, r_2, ..., r_G}

    3. normalize within the group (compute advantage):
       A_i = (r_i - mean({r_1..r_G})) / std({r_1..r_G})

    4. Update the policy using the normalized advantages:
       L_GRPO = -E[∑_t log π_θ(a_t|s_t) × A_i]
                + β × KL(π_θ || π_ref)

  No critic model needed — the group mean is the baseline.
```

```
  Memory comparison (7B model, BF16):

  PPO:
    Actor   14 GB (π_θ, gradients: +28 GB)
    Critic  14 GB (V_ψ, gradients: +28 GB)
    Reward  14 GB (frozen)
    Ref     14 GB (frozen)
    Total:  ~112 GB minimum

  GRPO:
    Actor   14 GB (π_θ, gradients: +28 GB)
    Reward  14 GB (frozen)
    Ref     14 GB (frozen)
    Total:  ~70 GB minimum  (−38% vs PPO)
```

### Why Group Sampling Works

The group mean is an unbiased estimate of the baseline as long as G is large enough (typically
G=8 to G=64 in practice). The variance of the advantage estimate decreases as G increases:

```
  Var(baseline estimate) ≈ Var(rewards) / G

  At G=8:    ~12.5% of single-sample variance
  At G=64:   ~1.6% of single-sample variance

  Typical DeepSeek-R1 training: G=8, 64 rollouts per batch
```

---

## 24.6 The DeepSeek-R1 Training Pipeline

`[DEEP DIVE]`

DeepSeek-R1 (DeepSeek-AI, 2025) is the first fully open-weight reasoning model with a disclosed
training procedure. It uses a multi-stage pipeline:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  DeepSeek-R1 Training Pipeline                                      │
  │                                                                     │
  │  Stage 0: Base model (DeepSeek-V3)                                  │
  │    Dense, MoE, 671B total / 37B active parameters                  │
  │    Pre-trained on 14.8T tokens                                      │
  │                                                                     │
  │  Stage 1: Cold Start SFT                                            │
  │    Train on small set (~1K–10K) of long chain-of-thought examples  │
  │    Purpose: teach the model the <think>...</think> format           │
  │    Without this, RL from scratch produces incoherent reasoning      │
  │                                                                     │
  │  Stage 2: Reasoning RL (GRPO)                                       │
  │    Reward: correctness of final answer (ORM on math/code)          │
  │    Format reward: penalise malformed <think> blocks                 │
  │    Run for thousands of steps                                       │
  │    Key emergence: model spontaneously learns to:                    │
  │      • Verify its own work ("wait, let me check...")                │
  │      • Backtrack when stuck ("that approach doesn't work")         │
  │      • Explore alternative methods ("another way to see this...")   │
  │                                                                     │
  │  Stage 3: Rejection Sampling + SFT                                  │
  │    Use Stage 2 model to generate many reasoning traces              │
  │    Keep only the traces that led to correct final answers          │
  │    Fine-tune on these "verified" traces (now acting as SFT data)   │
  │    Adds: helpfulness, safety, non-reasoning tasks                   │
  │                                                                     │
  │  Stage 4: Final RL                                                  │
  │    Short additional RL phase with combined reward:                  │
  │    correctness + helpfulness + safety                               │
  │    Stabilises the model for production deployment                   │
  │                                                                     │
  │  Distillation → R1-Distill-{7B, 8B, 14B, 32B, 70B}               │
  │    Use Stage 3/4 reasoning traces as SFT data for smaller models   │
  │    The small models *learn to mimic reasoning*, not discover it     │
  └─────────────────────────────────────────────────────────────────────┘
```

### The "Aha Moment"

The DeepSeek-R1 paper documents a striking emergence during Stage 2 RL: without any explicit
training signal to do so, the model began producing phrases like *"Wait, let me reconsider"*
and *"That seems wrong, let me try again"* — self-correction behavior that emerged purely from
the reward signal of getting final answers right.

This is the core argument for RL over SFT for reasoning: SFT can only teach the model to
produce text that looks like reasoning. RL causes the model to *discover* reasoning strategies
that actually work.

```
  Self-correction emergence (from DeepSeek-R1 paper):
    Early RL training:  model produces linear reasoning chains
    Mid RL training:    occasional "wait" tokens appear
    Late RL training:   systematic self-verification is common
    No supervision:     model discovered this behavior on its own
```

---

## 24.7 How o1 and o3 Differ (What We Know)

`[DEEP DIVE]`

OpenAI has not fully disclosed o1's training procedure, but the available evidence suggests:

```
  What is disclosed / strongly implied:
    • Multi-stage RL training (not pure SFT)
    • Process Reward Model used for step-level credit assignment
    • "Chain-of-thought as compute" framing: longer thinking = better answers
    • Scaling law: more test-time compute (longer thinking) → better performance
    • The thinking tokens are hidden from users in the o1 interface
      (they are present in the KV cache and consumed at inference time)

  Key difference from DeepSeek-R1:
    • o1's reasoning traces are never shown to users or in weights
    • R1's reasoning traces are visible and the weights are open
    • o1 likely uses significantly more RL compute and larger PRMs
    • OpenAI reports o1 training compute > GPT-4 training

  What we infer from benchmark behavior:
    • o1 exhibits systematic self-consistency checking
    • o1-preview shows ~3× more "tokens of thinking" on hard vs easy problems
      — evidence of adaptive budget allocation
    • o3 extends this: test-time compute is explicitly configurable
```

### The Test-Time Compute Scaling Law

The central empirical finding of the o1 class of models is that **inference compute scales like
training compute** — adding more thinking tokens improves performance on hard tasks following a
predictable power law:

```
  Performance ∝ (thinking_tokens)^α   for some domain-specific α > 0

  Empirical observations:
    Competition math (AIME):   doubling thinking budget → +4–8% accuracy
    PhD science (GPQA):        doubling thinking budget → +2–5% accuracy
    Simple factual QA:         doubling thinking budget → ~0% improvement
                               (model already knows, thinking tokens wasted)

  Implication: the optimal thinking budget is *problem-dependent*.
  A fixed budget wastes compute on easy problems and starves hard ones.
```

---

## 24.8 Qwen3 and Adaptive Thinking Mode

`[FOUNDATIONAL]`

Qwen3 (Alibaba, 2025) takes a different approach: a **single model** that can operate in two
modes, controlled at inference time:

```
  Standard mode:    <|im_start|>user ... <|im_end|>
                    → produces answer directly (no <think> block)

  Thinking mode:    /think
                    → produces <think>...</think> then answer

  Hybrid mode:      /think budget=2048
                    → thinking is capped at 2048 tokens
                    → if model hasn't resolved the problem, it stops thinking
                       and gives its best answer so far
```

This is architecturally simpler than training two separate models — Qwen3 is trained with both
thinking and non-thinking examples, and the mode is a learned prompt-level switch.

The **budget cap** is the key serving engineering lever: it bounds the worst-case KV cache
growth and makes latency predictable without requiring a separate deployment pool.

---

## 24.9 Inference Implications: What RL Training Changes

`[FOUNDATIONAL]`

Everything from Section 24.1 onward in this chapter originally covered inference mechanics. With
the training background established, those mechanics now have deeper explanations:

**Why reasoning tokens are longer than standard output:**
The RL training reward is correctness of the final answer. The policy learned that *longer
reasoning chains tend to produce more correct answers* — this is directly reinforced. The model
is not being verbose for style; it is executing a learned strategy.

**Why acceptance rates are lower for speculative decoding:**
Reasoning text is less predictable than chat text because the RL policy has discovered diverse
problem-solving strategies. The same arithmetic problem might be solved by direct multiplication,
by decomposition, by estimation-then-verification — each approach produces different tokens. A
draft model trained purely on SFT data does not know which strategy the target model will choose.

**Why batching is harder:**
Each reasoning sequence independently explores a solution path. Unlike standard generation where
similar prompts produce similar tokens (high batch coherence), reasoning sequences rapidly
diverge. KV cache fragmentation and unequal sequence lengths both worsen at the batch level.

---

## 24.10 The KV Cache Explosion

`[DEEP DIVE]`

With the training context clear, the KV cache growth is easy to explain: every reasoning token
the model generates becomes part of the context for all subsequent tokens. The model is
*re-reading its own thinking* at each step — that is why the reasoning chain is useful rather
than wasteful.

```
  Llama-3-8B, GQA (8 KV heads, 128 d_head, 32 layers, BF16):
    KV bytes per token = 32 × 2 × 8 × 128 × 2 = 131,072 bytes = 128 KB/token

  KV Cache Growth During Reasoning:

  GB   ┤
  7.0  ┤                                             ████
  6.0  ┤                                       ████████
  5.0  ┤                                 ████████
  4.0  ┤                           ████████
  3.0  ┤                     ████████
  2.0  ┤               ████████
  1.0  ┤         ████████
  0.1  ┤█▌ ← prompt
       └─────────────────────────────────────────── output tokens
        0      10K     20K     30K     40K     50K

  At 512 prompt + 50,000 reasoning tokens:
    Llama-3-8B:   50,512 × 128 KB = 6.47 GB
    Llama-3-70B:  50,512 × 320 KB = 16.16 GB  (320 KB/token for 80 layers)

  An H100 (80 GB) serving Llama-3-70B BF16 (140 GB weights) needs 2× GPUs.
  KV budget per GPU after weights: ~10 GB → max ~32K reasoning tokens.
  For 50K reasoning tokens: 3× H100 required.
```

`[COMMON TRAP]` — **Sizing `max_model_len` based on prompt length only**: the effective context
is `prompt + thinking + answer`. For a reasoning model with 50K max thinking tokens, set:

```python
  max_model_len = max_prompt_tokens + max_thinking_tokens + max_answer_tokens
               = 4096 + 50000 + 1024
               = 55120  →  round up to 65536
```

---

## 24.11 Serving Reasoning Models with vLLM

`[DEEP DIVE]`

### Correct Engine Configuration

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# WRONG: standard chat config
bad_args = AsyncEngineArgs(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_model_len=8192,      # ← too small: reasoning traces overflow
    max_num_seqs=64,         # ← too many: KV pool exhausted immediately
)

# CORRECT: reasoning-optimized config
good_args = AsyncEngineArgs(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_model_len=65536,     # 4K prompt + 60K thinking + 1K answer
    max_num_seqs=8,          # small: each sequence holds ~8 GB KV
    gpu_memory_utilization=0.92,
    enable_chunked_prefill=True,
    max_num_batched_tokens=4096,
    swap_space=0,            # disable KV swap: reasoning traces are never cold
)

# Reasoning request
reasoning_params = SamplingParams(
    max_tokens=32768,
    temperature=0.6,
    priority=10,             # lower than standard requests
)
```

### Priority Queuing: Separate Reasoning from Standard Traffic

The most critical architectural decision for mixed deployments:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Traffic Router                                                  │
  │                                                                  │
  │  Detection criteria:                                             │
  │    • model_id in {"deepseek-r1", "qwen3"} with /think           │
  │    • X-Request-Type: reasoning header                            │
  │    • estimated output > 2000 tokens                              │
  │         ↓                              ↓                         │
  │  Standard pool                 Reasoning pool                    │
  │  max_num_seqs=128              max_num_seqs=8                    │
  │  max_model_len=4096            max_model_len=65536               │
  │  TTFT SLA: < 200ms             TTFT SLA: < 5000ms                │
  │  ITL SLA:  < 30ms              ITL SLA:  < 100ms                 │
  └──────────────────────────────────────────────────────────────────┘
```

### Streaming the `<think>` Block

Users experience the full reasoning latency before the answer unless you stream:

```python
async def stream_reasoning_response(prompt: str, client):
    stream = await client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=32768,
    )
    in_think = False
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if "<think>" in delta:
            in_think = True
            print("\n[Thinking...]", end="", flush=True)
        elif "</think>" in delta:
            in_think = False
            print("\n[Answer:]", end="", flush=True)
        else:
            style = "\033[2m" if in_think else "\033[1m"   # dim vs bold
            print(style + delta + "\033[0m", end="", flush=True)
```

---

## 24.12 Serving Reasoning Models with llama.cpp

`[FOUNDATIONAL]`

llama.cpp's single-sequence architecture is well-suited for reasoning in the single-user case.

```bash
# Critical flags for reasoning models
# Key flag choices:
#   --ctx-size 65536       MUST cover prompt + full reasoning trace
#   --n-predict 32768      hard cap on output tokens
#   --temp 0.6             reasoning models prefer lower temperature
#   --repeat-penalty 1.0   do NOT penalise repetition — reasoning repeats concepts
llama-cli \
  --model deepseek-r1-distill-llama-8b-q4_k_m.gguf \
  --ctx-size 65536 \
  --n-predict 32768 \
  --temp 0.6 \
  --repeat-penalty 1.0 \
  --prompt "[prompt here]"
```

`[COMMON TRAP]` — **Setting `--repeat-penalty > 1.0` for reasoning models**: standard chat
benefits from repetition penalties to avoid looping. Reasoning models *deliberately* revisit
earlier steps ("as I computed above..."). A penalty > 1.0 will cause the model to avoid
referencing its own reasoning, degrading quality dramatically.

```
  Memory planning for Llama-3-8B Q4_K_M at ctx=65536:
    Model weights:   5.0 GB (Q4_K_M)
    KV cache:        65,536 × 128 KB/tok = 8.59 GB
    Total:           13.6 GB

    Fits in:  RTX 4090 (24 GB) ✓
              Apple M2 Pro 16 GB  ← tight, use ctx=32768 instead
              Apple M2 Ultra 192 GB ✓ (comfortable)
              CPU-only 32 GB RAM ✓ (slow: 3-8 tok/s)
```

---

## 24.13 Cost Model for Reasoning

`[DEEP DIVE]`

The RL training objective maximizes correctness, which correlates with longer thinking. This
makes reasoning models intrinsically expensive per *answer*, not per *token*:

```
  Standard request (Llama-3-8B):
    Input:   512 tokens
    Output:  200 tokens
    Time:    (512/12500) + (200/209) = 0.04 + 0.96 = 1.0 s
    Cost:    1.0/3600 × $2.49 = $0.00069/request = $0.69/1M output

  Reasoning request (DeepSeek-R1-8B):
    Input:   512 tokens
    Thinking:32,000 tokens ← the RL training made this necessary
    Answer:  500 tokens
    Time:    (512/11000) + (32500/195) = 0.05 + 166.7 = 166.7 s
    Cost:    166.7/3600 × $2.49 = $0.115/request = $3.55/1M output

  Cost ratio: $3.55 / $0.69 = 5.1× per output token
  But per *correct answer*: reasoning model may be 10–100× more
  accurate on hard tasks, shifting the cost/quality calculus entirely.
```

### Thinking Budget as the Primary Cost Lever

```
  Budget     Quality (simulated)   Latency    $/request
  ─────────────────────────────────────────────────────
     256        53%                  3.9 s     $0.003  ← fast, cheap
     512        60%                  5.3 s     $0.004
   1,024        67%                  7.9 s     $0.007
   4,096        80%                 23.6 s     $0.020
  16,384        93%                 86.6 s     $0.072  ← high quality
  32,768       100%                170.7 s     $0.142

  Recommendation:
    Routine tasks (code formatting, simple QA):  budget=1024
    Technical analysis, proofs:                  budget=16384
    Research-grade / competitive math:           budget=32768+
```

---

## 24.14 Speculative Decoding for Reasoning Models

`[DEEP DIVE]`

Recall from Chapter 23: speculative decoding uses a small draft model to propose K tokens, which
the large model verifies in a single parallel forward pass.

For reasoning models, the draft acceptance rate is lower because the RL policy has learned
diverse, unpredictable solution paths:

```
  Acceptance rate α by text type:
    Standard chat text:      α ≈ 0.80–0.85  → K=4 speedup: 2.6–3.1×
    SFT-trained reasoning:   α ≈ 0.65–0.75  → K=4 speedup: 2.0–2.4×
    RL-trained reasoning:    α ≈ 0.45–0.60  → K=4 speedup: 1.7–1.9×

  Why RL lowers acceptance:
    The RL policy discovers solution strategies that diverge from
    the average human reasoning pattern in the draft model's SFT data.
    The draft model predicts "likely next reasoning token"; the RL model
    may choose an unexpected but effective approach.

  Best practice:
    Use a draft model from the same reasoning model family
    (e.g., DeepSeek-R1-1.5B to draft for DeepSeek-R1-8B).
    Acceptance rates recover to α ≈ 0.65–0.75 when the draft
    was distilled from the same RL teacher.
```

---

## 24.15 Deployment Checklist

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Reasoning Model Deployment Checklist                           │
  ├─────────────────────────────────────────────────────────────────┤
  │  Training-aware sizing                                          │
  │    □  max_model_len = prompt + max_thinking + max_answer        │
  │    □  Understand distilled vs. RL-native: distilled models need │
  │       less thinking budget to reach quality; RL-native more    │
  │    □  /think budget= tested for your task difficulty class      │
  ├─────────────────────────────────────────────────────────────────┤
  │  Infrastructure                                                 │
  │    □  Separate pool from standard chat traffic                  │
  │    □  KV swap disabled (reasoning traces are never cold)        │
  │    □  terminationGracePeriodSeconds ≥ max_reasoning_time        │
  │    □  Streaming enabled (perceived latency improvement)         │
  ├─────────────────────────────────────────────────────────────────┤
  │  Cost controls                                                  │
  │    □  max_tokens hard cap per request                           │
  │    □  Thinking budget prompt engineering where supported        │
  │    □  Cost/request tracked separately from standard traffic     │
  │    □  --repeat-penalty 1.0 for llama.cpp (do NOT penalise)      │
  ├─────────────────────────────────────────────────────────────────┤
  │  Quality                                                        │
  │    □  Draft model from same family for speculative decoding     │
  │    □  Temperature 0.5–0.7 (reasoning benefits from lower temp)  │
  │    □  Verified: model correctly outputs </think> at end of trace│
  └─────────────────────────────────────────────────────────────────┘
```

---

## Summary

Reasoning models are the product of a specific training pipeline — RL with verifiable rewards —
that enables models to discover problem-solving strategies beyond what appears in human text.
Understanding this pipeline explains everything about their inference behavior: why they generate
tens of thousands of thinking tokens (the RL reward directly reinforces longer, more thorough
reasoning), why acceptance rates for speculative decoding are lower (the policy is less
predictable), and why they cost 5–10× more per answer than standard models (the thinking tokens
are not waste — they are the computation).

The training timeline runs: base model → cold start SFT (teaches the `<think>` format) → GRPO
RL (discovers reasoning strategies) → rejection sampling SFT (distils verified traces) → final
RL (adds helpfulness and safety). The distilled variants (R1-Distill-8B, etc.) compress the
reasoning capability into smaller models via SFT on the RL model's verified traces — they mimic
the behavior without rediscovering it.

Serving implications flow directly from this: size `max_model_len` for the full thinking trace,
deploy in isolated pools with lower SLA targets, use same-family draft models for speculation,
and match thinking budget to task difficulty to avoid paying for unnecessary compute.

---

## Key Terms

- **RLHF** — Reinforcement Learning from Human Feedback; three-stage pipeline (SFT → reward
  model → PPO) that produces instruction-following models.

- **PPO** — Proximal Policy optimization; the RL algorithm in RLHF; uses clipped surrogate
  objectives and requires four simultaneous model copies.

- **GRPO** — Group Relative Policy optimization; DeepSeek's PPO alternative that eliminates
  the critic by normalising rewards within a group of sampled responses.

- **ORM / PRM** — Outcome / Process Reward Model; ORM scores final answers only; PRM scores
  each reasoning step; PRM provides denser signal but requires step annotations.

- **Cold start** — the initial SFT phase in DeepSeek-R1's pipeline that teaches the model the
  `<think>...</think>` format before RL begins.

- **Thinking budget** — a configurable cap on reasoning tokens; bounds KV cache growth and
  makes latency predictable; supported natively in Qwen3 and via prompt engineering in R1.

- **Test-time compute scaling** — the empirical finding that more thinking tokens improve
  accuracy on hard tasks following a power law; the central motivation for the o1/o3 class.

- **Distillation (reasoning)** — training a smaller model on verified reasoning traces from a
  larger RL-trained model; the smaller model learns to mimic reasoning without rediscovering it.

---

*Next: Chapter 25 — RL Serving Policies*


---

## Self-Check Questions

1. A reasoning model generates an average of 8 000 thinking tokens before a 200-token answer. A standard model generates only 200 tokens. Compare the per-request KV cache consumption, GPU time, and $/request assuming the same model size and hardware cost. *(Section 24.1)*

2. Chain-of-thought reasoning requires the model to "think out loud" before answering. Name two ways this changes vLLM's scheduling behavior compared to a standard request. *(Section 24.2)*

3. Early stopping in reasoning models truncates the thinking chain when a confidence threshold is met. Describe how you would implement this as a sampling callback in vLLM. *(Section 24.4)*

4. A reasoning model's `<think>` tokens should not be streamed to the user but should be included in the KV cache for subsequent turns. Describe the token masking and KV cache inclusion logic required. *(Section 24.3)*

5. Compare the TTFT and total latency for a 70B reasoning model with 8 000 thinking tokens vs a 7B standard model with 200 output tokens on an A100. Which has lower TTFT? Which costs less per correct answer? *(Section 24.5)*


---

## Worked Solutions

### Question 1
**Reasoning model: 8,000 thinking + 200 answer tokens = 8,200 total. Standard: 200 tokens.**

**KV cache consumption:**
For a 70B model (32 layers, 8 KV heads, d=128, BF16), KV cost = 327 KB/token.
```
Reasoning: 8,200 x 327 KB = 2.68 GB per request
Standard:    200 x 327 KB = 65.4 MB per request
Ratio: 41x more KV cache for reasoning model
```

**GPU time (memory-bandwidth-bound decode):**
Reasoning: 8,200 decode steps x 70 ms/step = 574 s
Standard: 200 decode steps x 10 ms/step (7B model) = 2 s
GPU time ratio for same task: 287x (70B reasoning vs 7B standard)

**Cost per request:**
Reasoning model: ~574 GPU-seconds. Standard 7B: ~2 GPU-seconds.
If the reasoning model is 10x more accurate on hard tasks: cost-per-correct-answer may be lower for hard tasks where the standard model fails frequently.

---

### Question 2
**Two ways CoT reasoning changes vLLM's scheduling behavior:**

**Change 1 -- Much longer KV cache lifetime per request.**
A reasoning request holds KV blocks for 8,000+ decode steps. At any time, reasoning "whale" requests consume a disproportionate share of the KV block pool. The scheduler may preempt them if the pool exhausts, forcing costly recomputation of thinking tokens.

**Change 2 -- Unpredictable generation length.**
Standard requests have bounded max_tokens. Reasoning models generate thinking chains of wildly variable length (200 to 50,000+ tokens). The scheduler cannot accurately predict KV block requirements at admission time. vLLM handles this by incrementally allocating blocks and using preemption as a safety valve, but this increases scheduling overhead compared to predictable workloads.

---

### Question 3
**Early stopping as a sampling callback:**

```python
THINK_END_TOKEN_ID = tokenizer.encode("</think>")[0]

def thinking_stop_callback(request_id, outputs):
    for output in outputs:
        if output.token_ids and output.token_ids[-1] == THINK_END_TOKEN_ID:
            return True  # signal vLLM to stop generation
        # Confidence-based: stop if top-1 prob > threshold AND min length met
        if output.logprobs and len(output.token_ids) > 500:
            top_prob = max(math.exp(lp) for lp in output.logprobs[-1].values())
            if top_prob > 0.95:
                return True
    return False

sampling_params = SamplingParams(
    max_tokens=32768,
    stop_token_ids=[THINK_END_TOKEN_ID],
    # stop=["</think>"] also works via built-in stop string
)
```

The most reliable signal is the </think> delimiter the model generates naturally. Confidence-based stopping works as a fallback for models without explicit thinking delimiters.

---

### Question 4
**Token masking and KV cache inclusion for thinking tokens:**

**Not streaming to user:**
Apply a filter in the SSE streaming path:
```

- Buffer all tokens after <think>
- Do not yield buffered tokens to the SSE output stream  
- After </think> is seen, resume yielding answer tokens
```

**KV cache inclusion (automatic):**
Thinking tokens are part of the sequence from the KV cache perspective. When the model generates </think> and transitions to answer generation, all thinking tokens are already in KV blocks. The model attends to the full thinking chain when generating the answer -- no special handling needed. vLLM automatically includes all generated tokens in KV blocks regardless of whether they are streamed to the user.

**Multi-turn:** Store the full response (including thinking tokens) for multi-turn context. Include them in subsequent turns' KV cache -- the model benefits from its own prior reasoning.

---

### Question 5
**TTFT and total latency: 70B reasoning (8K thinking) vs 7B standard (200 tokens) on A100.**

**TTFT comparison:**
- Reasoning 70B: TTFT for first *answer* token = prefill of input + 8,000 decode steps.
  - 500-token input prefill: ~50 ms. 8,000 decode steps x 70 ms = 560 s.
  - TTFT for answer: ~560 seconds.
- Standard 7B: prefill 500 tokens: ~5 ms. First decode: ~10 ms. TTFT ~= 15 ms.

The 7B standard model has ~37,000x lower TTFT. Reasoning models are unsuitable for real-time interactive applications requiring sub-second response.

**Total latency:**
- Reasoning 70B: 560 s + 200 x 70 ms = ~574 s total
- Standard 7B: 15 ms + 200 x 10 ms = ~2 s total

**Cost per correct answer:**
If reasoning model is 10x more accurate on hard math/coding tasks, and costs 287x more GPU time, cost-per-correct-answer is 287/10 = 28.7x higher. But if standard model achieves <5% accuracy on hard tasks (vs reasoning model's 50%), the reasoning model's cost-per-correct-answer is actually lower for those tasks (cost 28.7x, correctness 10x better: 2.87x cost per correct answer, not 28.7x). The choice depends entirely on task difficulty and accuracy requirements.
*Companion code: [`docs/code/chapter_24.md`](../code/chapter_24.md)*
