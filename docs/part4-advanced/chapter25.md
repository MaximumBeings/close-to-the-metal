# Chapter 25 — Reinforcement Learning, Feedback, and Serving Policies

> *"RLHF changes the model you deploy. Serving-policy optimization changes how you use it. Both matter for production quality, but they operate on completely different timescales and require completely different tooling."*

Chapter 24 covered how reinforcement learning training methods — RLHF, GRPO, PPO — shape the reasoning capabilities of a model before it ever touches a request. This chapter covers what happens *after* the model is deployed: how inference engineers apply RL concepts to the decisions a serving system makes in real time. You will encounter the same mathematical vocabulary — policy, reward, state, action, trajectory — but the agent is no longer a gradient descent loop over GPU hours. The agent is your production inference stack, making microsecond-to-millisecond routing decisions on every user request.

This distinction is important enough to state plainly at the start:

- **RLHF, RLAIF, DPO, PPO, GRPO** are offline or near-offline training processes. They change the weights of the model you deploy. Your vLLM or llama.cpp instance serves the resulting model but has no awareness of how it was trained.
- **Serving-policy optimization** — model routing, prompt selection, cache policy, tool selection, retrieval depth, clarification decisions — is an online process. The "policy" is the logic inside your serving stack, and you can update it continuously from production signals without retraining a single weight.

Both approaches improve user-perceived quality, but conflating them leads to overengineering: teams spend months on RLHF pipelines to solve problems that a two-week bandit experiment on prompt routing would have addressed at a fraction of the cost.

---

## 25.1 The RL Vocabulary for Inference Engineers

### 25.1.1 Core Concepts

**Policy (π):** Any function that maps observations to actions. In serving, a policy is the logic that decides, for a given request, which model to call, with what parameters, using which prompt template. Your current hardcoded routing rule is a deterministic policy. A learned bandit is a stochastic policy.

**State (s):** The observable information available when a decision must be made. In a serving context, state includes request metadata (latency budget, user tier, estimated input complexity, current queue depth, GPU utilization), historical context (is this a returning user? have we seen similar requests before?), and system state (cache hit probability, available models, backend health).

**Action (a):** A discrete or continuous choice made by the policy. Examples: select model A vs. B vs. C; set max_tokens to 256 vs. 1024; use cached response vs. re-generate; invoke tool vs. answer directly; ask clarifying question vs. attempt answer.

**Reward (r):** A scalar signal indicating how good an action was. Reward may arrive immediately (latency of the response was within budget: +1) or with delay (user rated the response 4/5: +0.8). Reward design is where most production RL efforts fail — a poorly specified reward function is optimized perfectly and gives you exactly what you asked for, not what you wanted.

**Trajectory (τ):** The sequence of (state, action, reward) tuples across a single conversation or session: (s₀, a₀, r₀), (s₁, a₁, r₁), ..., (s_T, a_T, r_T). A single API call is a one-step trajectory. A multi-turn conversation is a multi-step trajectory where earlier routing decisions affect the state at later turns.

**Return (G):** The cumulative reward over a trajectory, potentially discounted: G = Σₜ γᵗ rₜ. For serving decisions, γ is often 1 (no discounting) over the span of a single request, and close to 1 across a session — there is rarely a reason to discount rewards heavily in the serving domain.

**Value function (V^π(s)):** The expected return from state s under policy π. If you could compute this perfectly, optimal decision-making is trivial. In practice, value functions are approximated from historical data.

### 25.1.2 How Training-Time RL Feeds Into Serving

The output of RLHF/GRPO/DPO training is a set of model weights. Those weights encode:

- A distribution over responses that reflects the preference signal the model was trained on
- Calibrated confidence (or miscalibration, depending on training quality)
- Behavioral tendencies: verbosity, refusal rates, format adherence, reasoning chain quality

Your serving infrastructure cares about these properties because they affect downstream routing decisions. A model trained with strong RLHF alignment needs less prompt engineering for format compliance. A model trained with GRPO for math reasoning is more likely to use `<think>` tags and show work, which affects latency budgeting. A DPO-trained model may refuse more requests, which affects your fallback routing logic.

The serving engineer does not need to understand the gradient derivations from Chapter 24 to work with these models. But understanding *what was optimized* — the reward function, the preference data source, the KL penalty strength — helps predict behavior at serving time and design better post-deployment policies.

---

## 25.2 Training-Time Alignment Methods: A Serving Engineer's Summary

This section provides the conceptual summary needed for serving decisions. The full training mathematics are in Chapter 24.

### 25.2.1 RLHF (Reinforcement Learning from Human Feedback)

**What it does:** A reward model is trained to predict human preference scores, then PPO or a similar RL algorithm trains the LLM to maximize the predicted reward while staying close to a reference model via a KL penalty.

**Serving implications:**

- The model follows instructions more reliably, reducing the need for defensive prompt engineering
- Refusal rates are higher; budget for fallback routing to a less-restricted model for edge cases
- The model's probability distribution is narrower (higher confidence on preferred formats); speculative decoding acceptance rates improve
- KL-constrained training means the model has not drifted far from its base; calibration is generally preserved

### 25.2.2 RLAIF (Reinforcement Learning from AI Feedback)

**What it does:** The same pipeline as RLHF, but an existing strong LLM (e.g., GPT-4, Claude) generates the preference labels instead of human annotators.

**Serving implications:**

- Faster and cheaper to iterate than RLHF; expect more domain-specific variants
- The quality ceiling is bounded by the annotator model; if your target task outperforms the annotator, RLAIF rewards may be miscalibrated
- Useful for aligning specialized models (code, math, domain-specific) where human annotation is expensive

### 25.2.3 DPO (Direct Preference Optimization)

**What it does:** Bypasses the explicit reward model. Derives a closed-form training objective directly from preference pairs (chosen response, rejected response) using a reparameterization of the reward function in terms of the optimal policy.

**Serving implications:**

- No explicit reward model at serving time; the alignment is baked into the weights
- Training is more stable than PPO (no adversarial reward hacking in the reward model)
- The implicit reward can be extracted at inference time: `r_implicit(x, y) = β log[π_θ(y|x) / π_ref(y|x)]`. This can be used as a re-ranking signal for multiple candidate generations
- DPO-aligned models tend to be longer and more verbose than RLHF models (a known artifact of the length-controlled winrate phenomenon)

### 25.2.4 PPO in the Serving Context

PPO is primarily relevant to serving engineers because it is used in RLHF pipelines to produce aligned models. However, PPO's clipping mechanism has a conceptual analogue in serving policy updates: when you update a routing policy in production, you do not want the new policy to differ drastically from the old policy (this is the serving analogue of the PPO trust region). Gradual rollout, A/B testing, and shadow deployments serve the same stabilizing function that PPO's clip parameter ε serves in training.

### 25.2.5 GRPO in the Serving Context

GRPO-trained models (Chapter 24) have distinctive serving characteristics worth knowing:

- They generate significantly longer responses (chain-of-thought reasoning)
- Their KV cache growth per token is identical to standard models, but the number of tokens is 3–10× higher
- Their speculative decoding acceptance rate is lower (~0.55 vs. ~0.80 for chat models) because reasoning chains are less predictable
- They respond well to token budgets (`/think budget=N` in Qwen3 format)

---

## 25.3 The Bandit Framework for Serving Decisions

Bandits are the right tool when you face a decision with uncertain outcomes, want to learn from production feedback, and cannot afford the delay or cost of full RL with trajectory-level rewards.

### 25.3.1 The Multi-Armed Bandit Problem

You have K actions (model versions, prompt templates, cache policies). Each action i has an unknown expected reward μᵢ. At each timestep t you choose an action aₜ and observe reward rₜ ~ P(r | a = aₜ). The goal is to maximize cumulative reward while learning which action is best.

The fundamental tension is **exploration vs. exploitation**: you must occasionally try suboptimal actions to gather information, while mostly exploiting what you already know to be good.

**ε-greedy policy:** With probability ε, choose a random action; otherwise choose the action with the highest estimated reward. Simple and robust. ε = 0.05–0.10 is typical in production.

```python
import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_actions: int, epsilon: float = 0.1):
        self.n_actions = n_actions
        self.epsilon   = epsilon
        self.counts    = np.zeros(n_actions)        # times each action taken
        self.values    = np.zeros(n_actions)        # estimated mean reward

    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # explore
        return int(np.argmax(self.values))            # exploit

    def update(self, action: int, reward: float):
        self.counts[action] += 1
        n = self.counts[action]
        # Incremental mean update: avoids storing all observations
        self.values[action] += (reward - self.values[action]) / n
```

**UCB (Upper Confidence Bound):** Choose the action with the highest upper confidence bound: a_t = argmax_i [μ̂ᵢ + c√(ln t / nᵢ)]. UCB naturally explores actions that have been tried fewer times. Better than ε-greedy for non-stationary environments where action rewards change over time.

**Thompson Sampling:** Maintain a posterior distribution over each action's reward parameter. At each step, sample one reward estimate from each posterior and choose the action with the highest sample. For Bernoulli rewards (user clicks, thumbs up), use a Beta(α, β) posterior where α = successes + 1 and β = failures + 1.

```python
class ThompsonSamplingBandit:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.alpha = np.ones(n_actions)   # successes + 1 (Beta prior)
        self.beta  = np.ones(n_actions)   # failures  + 1

    def select_action(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, action: int, reward: float):
        """reward should be 0 or 1 for Beta-Bernoulli conjugate update."""
        self.alpha[action] += reward
        self.beta[action]  += (1 - reward)
```

Thompson Sampling is generally preferred in production LLM serving because it explores efficiently even with very sparse rewards (user feedback arrives on only a small fraction of requests).

### 25.3.2 Contextual Bandits

Standard bandits ignore the context of the request. A contextual bandit takes a feature vector x (the "context") representing the current state, and learns a separate reward estimate for each (context, action) pair. This is the right formalism for most LLM serving decisions, where the optimal action depends on the request.

Context features for LLM serving:

| Feature | How to compute | Example actions affected |
|---------|---------------|-------------------------|
| Estimated input complexity | Token count, perplexity, entity density | Model tier selection |
| Predicted output length | Embedding similarity to length-labeled historical requests | max_tokens budget |
| User latency budget | SLA tier, explicit timeout header | Model routing, streaming vs. batch |
| Topic/domain classification | Embedding + KNN to labeled examples | Prompt template selection |
| Cache similarity score | Top-1 cosine distance to cache entries | Semantic cache vs. re-generate |
| Session context length | Number of turns, total tokens so far | Summarization trigger |
| Request hour-of-day | Timestamp | Load-adaptive routing |

A linear contextual bandit (LinUCB) estimates rewards as a linear function of the context:

```
r̂(x, a) = x^T θ_a
```

and maintains separate ridge regression weight vectors θ_a for each action a. The UCB exploration bonus is derived from the uncertainty in the estimate.

For serving decisions, a simpler approach often suffices: segment requests into a small number of discrete context buckets (e.g., "short/simple", "long/complex", "math/code", "conversational") and run a separate standard bandit per bucket.

---

## 25.4 Serving Decisions as a Bandit Problem

### 25.4.1 Model Routing

**Decision:** Given a request, which model variant to use — small/fast, large/slow, or specialized.

**Reward design:** The challenge is defining a single scalar reward that captures the trade-off. A useful decomposition:

```
r = quality_score × (1 - latency_penalty) × (1 - cost_penalty)

latency_penalty = clamp((observed_latency - target_latency) / target_latency, 0, 1)
cost_penalty    = clamp(actual_cost / budget_per_request, 0, 1)
quality_score   = user_rating / 5.0  (if available)
                  OR output_length_normalized_log_prob  (proxy)
                  OR downstream_task_accuracy           (if measurable)
```

**Bandit setup:** Three actions (small/medium/large model). Context: estimated complexity bucket. After each request, compute reward and update the appropriate bandit.

**Practical insight:** In most production deployments, the routing bandit converges quickly (within a few thousand requests per bucket) because the reward distribution is fairly stable. The main value is catching drift — when a model update changes the quality/latency trade-off, the bandit re-learns the optimal routing within hours rather than requiring a manual re-evaluation.

### 25.4.2 Prompt Template Selection

**Decision:** Which of K prompt templates to use for a given task type.

**Why this matters:** Different prompt formulations produce different output distributions. A prompt that says "Answer in exactly three bullet points" will produce shorter, more structured responses than "Please help me with...". For a given user segment, one prompt may produce significantly higher satisfaction scores.

**Reward:** User engagement signals — session continuation rate, thumbs up rate, copy-to-clipboard rate. These signals are delayed (minutes to hours) and sparse, making Thompson Sampling the preferred exploration strategy.

**Implementation note:** Prompt selection is a cheap action — the same model serves all templates, so there is no GPU cost difference. This makes aggressive exploration (ε = 0.2 or higher) acceptable.

### 25.4.3 Retrieval Depth in RAG Pipelines

**Decision:** How many documents to retrieve and include in the prompt context.

**Actions:** Retrieve top-k where k ∈ {1, 3, 5, 10, 20}. Each action has a different latency cost (more retrieval time), context length (more prompt tokens → more prefill compute), and quality ceiling (more context = potentially better answers, but also more noise).

**Reward:** A latency-adjusted quality metric:

```
r = answer_correctness - α × (k / k_max) - β × (latency / latency_budget)
```

where `answer_correctness` is measured by a downstream checker (exact match, model-graded, or A/B comparison) and α, β are tunable coefficients that reflect your product's latency sensitivity.

**Bandit context:** User query type (factual lookup vs. synthesis vs. open-ended) strongly predicts optimal k. A factual lookup ("What is the capital of France?") benefits from k=1; a synthesis task ("Summarize the five most relevant papers on X") benefits from k=10 or more.

### 25.4.4 Semantic Cache Policy

**Decision:** Whether to serve a cached response or re-generate.

**Challenge:** Semantic similarity alone is not sufficient to decide whether cached responses are acceptable. A query about "current stock price" should never be served from cache (time-sensitive). A query about "explain the Pythagorean theorem" is almost always safe to cache.

**Bandit formulation:** The action is whether to cache-serve at a given similarity threshold θ. Actions: always re-generate, cache if similarity > 0.85, cache if similarity > 0.90, cache if similarity > 0.95. The reward includes both cache-hit rate (latency reduction) and user satisfaction (did the cached response actually meet their need?).

**Practical finding:** Most production semantic caches should use two-level thresholding: similarity > 0.95 for aggressive caching, similarity 0.85–0.95 for A/B testing (serve 50% from cache, 50% fresh), similarity < 0.85 always re-generate. A bandit on the middle tier quickly learns which query types are safe to cache at 0.85.

### 25.4.5 Tool Selection

**Decision:** Whether to call an external tool (search, calculator, code executor) or answer directly from the model.

**State:** The query type, the model's self-assessed uncertainty (log-probability of the first generated token), the tool's latency estimate, and the current system load.

**Reward:** Correctness of the final answer, minus latency penalty for tool calls. Tool calls add 200–2000 ms depending on the tool; the expected quality gain must justify this.

**Bandit on tool routing:** Train a classifier on historical requests where tool use was attempted, using the outcome (user satisfaction, correctness) as the label. Use the classifier confidence as a contextual feature in the routing bandit. This produces a principled threshold that adapts to the model's evolving capabilities as you update it.

### 25.4.6 Clarification vs. Answer Decisions

**Decision:** Whether to ask a clarifying question when the request is ambiguous.

**The trade-off:** Clarification improves answer quality but adds latency (the user must respond before the system proceeds) and can frustrate users who expected an immediate answer. Over-clarifying is one of the most common complaints about AI assistants.

**Reward design:**

```
r_clarify   = quality_of_final_answer - clarification_delay_penalty - user_frustration_proxy
r_answer    = quality_of_immediate_answer
```

where `user_frustration_proxy` is estimated from session abandonment rate when clarification is asked.

**Bandit finding from production:** Clarification is worth asking only when ambiguity is high (multiple distinct interpretations with very different answer trajectories) AND the latency budget is relaxed. For real-time chat, the optimal policy is usually "attempt answer, offer to clarify if user is unsatisfied." For asynchronous workflows (report generation, batch processing), clarification is almost always worth the delay.

---

## 25.5 Reward Design: Getting It Right

Reward design is the hardest part of production RL. Every reward function will be exploited by whatever policy you train against it.

### 25.5.1 Goodhart's Law in LLM Serving

Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure." In LLM serving:

- **Optimize for session length** → model learns to be verbose and conversational rather than concise and accurate
- **Optimize for thumbs-up rate** → model learns to be sycophantic; it tells users what they want to hear
- **Optimize for latency** → routing bandit converges to always selecting the smallest, fastest model regardless of quality
- **Optimize for cache hit rate** → cache policy becomes too aggressive, serving stale or irrelevant responses
- **Optimize for refusal rate (safety)** → model over-refuses legitimate requests, damaging utility

The solution is **composite reward functions with explicit floor constraints:**

```python
def serving_reward(
    latency_ms: float,
    user_rating: float | None,
    is_safe: bool,
    is_correct: bool | None,
    latency_budget_ms: float,
    min_quality_threshold: float = 3.0,
) -> float:
    # Hard safety constraint: unsafe responses get worst possible reward
    if not is_safe:
        return -1.0

    # Quality floor: if quality is unacceptable, don't reward speed
    if user_rating is not None and user_rating < min_quality_threshold:
        return 0.0

    # Latency reward (only meaningful if quality is acceptable)
    latency_factor = max(0.0, 1.0 - latency_ms / latency_budget_ms)

    # Quality reward (normalized to [0, 1])
    quality_factor = (user_rating / 5.0) if user_rating is not None else 0.5

    return quality_factor * 0.7 + latency_factor * 0.3
```

Key principles:

1. **Hard constraints before soft rewards.** Safety violations, format failures, and obviously wrong answers should produce negative or zero reward regardless of latency or cost. Never let an optimization pressure override a hard constraint.
2. **Reward the joint outcome, not individual components.** A fast response that is wrong is worse than a slow response that is right; the reward function must reflect this with multiplicative rather than additive terms.
3. **Separate exploration rewards from deployment rewards.** During exploration phases (A/B testing, bandit learning), use a more tolerant reward function. During deployment, tighten the quality floor.

### 25.5.2 Latency-Adjusted Reward

Latency is not uniformly important. A 500ms latency increase for a user with a 30-second deadline is negligible; the same increase for a real-time voice application is catastrophic. The reward function should use *relative* latency rather than absolute:

```
latency_reward = 1.0 if latency < target
               = exp(-k × (latency - target) / target) if latency > target
```

The exponential decay with coefficient k reflects that modest latency overruns are tolerable, but large overruns compound user dissatisfaction nonlinearly.

### 25.5.3 Reward Hacking and Detection

Reward hacking is when the policy achieves high reward by exploiting a loophole in the reward function rather than by improving the actual outcome you care about. Detection strategies:

- **Monitor proxy metrics alongside the reward metric.** If thumbs-up rate increases but session continuation decreases, the model is becoming sycophantic.
- **Periodic random sampling and human evaluation.** Sample 1–5% of high-reward responses for human review. If reviewers consistently disagree with the reward signal, the reward function needs adjustment.
- **Distribution shift monitoring.** Track the distribution of actions taken by the policy. If the routing bandit shifts 95% of traffic to a single model, investigate why before assuming this is optimal.
- **Counterfactual logging.** Log the action not taken and (where possible) its expected reward. This allows offline analysis of whether the policy is genuinely improving or just gaming the metric.

---

## 25.6 Offline Policy Evaluation Before Rollout

Before deploying any updated serving policy to production, you need an offline estimate of how it will perform. This is the serving analogue of model evaluation before deployment.

### 25.6.1 Direct Method (DM)

Re-run the existing log data through the new policy and evaluate the estimated reward. Simple but biased — the new policy may take actions that were never logged, so you have no reward estimate for them.

### 25.6.2 Inverse Propensity Scoring (IPS)

Correct for the action distribution mismatch between the logging policy (the old policy that generated your data) and the evaluation policy (the new policy you want to evaluate):

```
V̂_IPS(π_new) = (1/n) Σᵢ [π_new(aᵢ | xᵢ) / π_log(aᵢ | xᵢ)] × rᵢ
```

This is an unbiased estimator when the logging policy has full support (π_log(a | x) > 0 for all actions a). In practice, clip the importance weights to prevent high-variance estimates from rare actions:

```python
def ips_estimate(
    actions:       list[int],
    rewards:       list[float],
    new_policy_probs:  list[float],   # π_new(a | x) for each logged action
    log_policy_probs:  list[float],   # π_log(a | x) for each logged action
    clip_weight:   float = 10.0,
) -> float:
    weights = [
        min(new_p / log_p, clip_weight)
        for new_p, log_p in zip(new_policy_probs, log_policy_probs)
    ]
    n = len(rewards)
    return sum(w * r for w, r in zip(weights, rewards)) / n
```

### 25.6.3 Doubly Robust Estimation (DR)

Combines DM and IPS: uses IPS to correct the DM estimate, getting the benefits of both (low bias from IPS, lower variance from DM):

```
V̂_DR(π_new) = (1/n) Σᵢ [ r̂(xᵢ, π_new(xᵢ))
                          + (π_new(aᵢ|xᵢ) / π_log(aᵢ|xᵢ))
                            × (rᵢ - r̂(xᵢ, aᵢ)) ]
```

where r̂(x, a) is a learned reward model. DR is the industry standard for offline policy evaluation in recommendation systems and ad ranking; it transfers directly to LLM serving decisions.

### 25.6.4 Shadow Mode Evaluation

Before full rollout, run the new policy in shadow mode: route all traffic through the old policy for actual responses, but also compute what the new policy would have done and log it. Compare the reward distributions offline. This is more conservative than IPS but requires no reward model and has no variance inflation.

---

## 25.7 Safety Constraints in Serving Policies

RL optimization without safety constraints will find ways to increase reward that violate your safety requirements. This is not a theoretical concern — it happens reliably in production.

### 25.7.1 Constrained Optimization

The formal framework is Constrained MDP: maximize expected reward subject to expected constraint violation ≤ δ:

```
max_π  E[R(τ)]
s.t.   E[C(τ)] ≤ δ
```

where C(τ) is a constraint cost (e.g., fraction of responses flagged as unsafe). In practice, this is implemented as a Lagrangian relaxation:

```
L(π, λ) = E[R(τ)] - λ × (E[C(τ)] - δ)
```

The Lagrange multiplier λ is updated to enforce the constraint. If constraint violations increase, λ increases, penalizing the policy more heavily for unsafe actions.

### 25.7.2 Hard Guardrails at the Serving Layer

Rather than encoding safety as a reward, implement safety as a hard filter that intercepts outputs before they reach users:

```
         ┌──────────┐        ┌──────────────┐
request  │  Policy  │ action │  vLLM / cpp  │ output  ┌──────────────┐
────────►│ (bandit/ │───────►│  generation  │────────►│ Safety filter│──► user
         │  router) │        │              │         │  (classifier)│
         └──────────┘        └──────────────┘         └──────┬───────┘
                                                             │ unsafe
                                                             ▼
                                                    ┌────────────────┐
                                                    │ Fallback/refuse│
                                                    └────────────────┘
```

The safety filter should operate independently of the policy and should not be part of the reward function. This creates a defense-in-depth architecture: the serving policy optimizes for quality given that safety is enforced, rather than trading safety for quality.

### 25.7.3 Distributional Safety

Beyond per-request safety, monitor safety at the population level. A serving policy that routes 1% of requests through a less-restricted model for efficiency gains may produce an acceptable per-request safety rate while creating a systemic vulnerability for targeted attacks. Use population-level safety audits (monthly random sampling with human evaluation) alongside per-request filters.

---

## 25.8 vLLM as the Execution Engine for Serving Policies

The serving policy makes decisions; vLLM executes them. This clean separation is the key architectural pattern.

### 25.8.1 Passing Policy Decisions to vLLM

```python
from vllm import LLM, SamplingParams

def execute_policy_decision(
    llm: LLM,
    prompt: str,
    policy_action: dict,   # output from your serving policy
) -> str:
    """
    policy_action contains the decisions the bandit/policy made:
    - max_tokens: budget decision
    - temperature: creativity/determinism trade-off
    - model (if using multi-model routing, requires separate LLM instances)
    - stop_sequences: based on expected output format
    """
    sp = SamplingParams(
        temperature=policy_action.get("temperature", 0.7),
        max_tokens=policy_action.get("max_tokens", 512),
        stop=policy_action.get("stop_sequences", []),
        top_p=policy_action.get("top_p", 1.0),
    )
    outputs = llm.generate([prompt], sp)
    return outputs[0].outputs[0].text
```

### 25.8.2 Multi-Model Routing with vLLM

For routing between model tiers, maintain a pool of vLLM instances and dispatch based on the policy's action:

```python
from vllm import LLM, SamplingParams

class ModelPool:
    def __init__(self, model_configs: dict[str, dict]):
        """
        model_configs: {"small": {"path": "...", "gpu": "cuda:0"},
                        "large": {"path": "...", "gpu": "cuda:1"}}
        """
        self.models = {
            name: LLM(
                model=cfg["path"],
                device=cfg["gpu"],
                dtype="bfloat16",
                gpu_memory_utilization=0.85,
            )
            for name, cfg in model_configs.items()
        }

    def generate(self, model_name: str, prompts: list[str],
                 params: SamplingParams) -> list[str]:
        llm = self.models[model_name]
        outputs = llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]
```

### 25.8.3 vLLM Metrics for Reward Computation

vLLM exposes metrics useful for reward computation:

```python
# After each generation, extract serving metrics for reward calculation
output = llm.generate([prompt], params)[0]

# Time to first token (TTFT) — for latency reward
ttft_ms = output.metrics.first_token_time * 1000  # convert to ms

# Total generation time — for latency budget reward
total_time_ms = output.metrics.finished_time * 1000

# Token counts — for cost reward
input_tokens  = len(output.prompt_token_ids)
output_tokens = len(output.outputs[0].token_ids)
cost_estimate = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
```

These metrics feed directly into the reward function, closing the loop: the bandit takes an action, vLLM executes it and returns metrics, the reward function evaluates the outcome, and the bandit updates.

---

## 25.9 llama.cpp as a Local Experimentation Runtime

Before deploying a new serving policy to production with vLLM, llama.cpp is an effective local testing harness for several reasons:

- **Zero infrastructure:** Runs on a laptop or single workstation. Ideal for reward function prototyping and bandit algorithm testing before production deployment.
- **Predictable latency:** llama.cpp's latency is stable and reproducible on the same hardware, making it useful for calibrating latency reward functions before production deployment where vLLM's PagedAttention introduces more variability.
- **Offline evaluation:** Many serving policy decisions can be evaluated offline using historical request logs and a local llama.cpp instance to re-score responses.
- **Edge deployment:** When the serving policy includes an edge/local model as one of the routing options, llama.cpp is typically the execution engine for that option.

### 25.9.1 llama.cpp as a Bandit Action

In a multi-tier routing setup:

| Tier | Model | Runtime | Latency | Cost/1M tokens |
|------|-------|---------|---------|----------------|
| Edge | Qwen 2.5 1.5B Q4_K_M | llama.cpp on device | 20–80 ms | $0 (local) |
| Mid | Llama 3.1 8B | vLLM on shared GPU | 80–300 ms | $0.10 |
| High | Llama 3.3 70B | vLLM on A100 cluster | 500–2000 ms | $0.90 |
| API | GPT-4o | External API | 1000–5000 ms | $5.00 |

The bandit selects among these tiers based on context (request complexity, latency budget, user SLA). A Thompson Sampling bandit with Beta posteriors per tier per context bucket converges to near-optimal routing within a few thousand requests.

### 25.9.2 Reward Calibration with llama.cpp

When developing the reward function, calibrate it against known-good and known-bad responses using llama.cpp locally:

```python
import subprocess, json

def score_with_llama_cpp(
    prompt: str,
    response: str,
    model_path: str,
    judging_prompt_template: str,
) -> float:
    """Use a local judge model to score a response."""
    judging_prompt = judging_prompt_template.format(
        prompt=prompt,
        response=response
    )
    result = subprocess.run(
        ["llama-cli", "-m", model_path, "-p", judging_prompt,
         "--temp", "0", "-n", "10"],
        capture_output=True, text=True
    )
    # Parse score from output (e.g., "Score: 4")
    output = result.stdout.strip()
    try:
        score = float(output.split("Score:")[-1].strip().split()[0])
        return score / 5.0   # normalize to [0, 1]
    except (ValueError, IndexError):
        return 0.5            # default if parsing fails
```

This local judging pipeline lets you validate your reward function design before deploying the full production scoring infrastructure.

---

## 25.10 Putting It Together: A Production Serving Policy Loop

The complete architecture combines a contextual bandit, a model pool, reward computation, and offline evaluation:

```
                  ┌─────────────────────────────────────┐
                  │         Serving Policy Loop          │
                  │                                      │
  Request ──────► │  1. Extract context features         │
                  │  2. Contextual bandit selects action  │
                  │  3. Execute via vLLM / llama.cpp      │
                  │  4. Safety filter                     │
                  │  5. Return response to user           │
                  │  6. [async] Collect reward signal     │
                  │  7. [async] Update bandit             │
                  └─────────────────────────────────────┘
                              │
                  ┌─────────────────────────────────────┐
                  │       Offline Policy Evaluation      │
                  │  - IPS / DR estimates before rollout │
                  │  - Shadow mode A/B testing           │
                  │  - Reward hacking detection          │
                  │  - Human sampling audit              │
                  └─────────────────────────────────────┘
```

Key operational decisions:

1. **Reward delay tolerance:** If user feedback arrives within seconds, online bandit updates are feasible. If feedback arrives hours later (e.g., task completion), batch updates nightly.
2. **Exploration budget:** In production, keep ε ≤ 0.05–0.10 to limit degraded user experiences from exploration. In staging/canary, explore aggressively (ε = 0.30).
3. **Model update frequency:** When you deploy a new model version, reset the relevant bandit arms to a uniform prior. The bandit will re-learn the new model's quality/latency profile quickly.
4. **Safety immutability:** Safety filters and hard constraints should not be part of the bandit's action space. They operate as post-processing layers that cannot be disabled by policy optimization.

---

## 25.11 Key Takeaways

The RL vocabulary from Chapter 24 (policy, reward, trajectory, advantage) applies directly to serving decisions, but the timescales and stakes are different. Training-time RL operates over days and modifies model weights irreversibly; serving-time policy optimization operates over minutes and is fully reversible. This asymmetry means you can be more aggressive with experimentation in serving policy than in model training.

The central distinction remains the chapter's opening claim: RLHF/RLAIF/DPO changes what the model can do; bandit-based serving policy optimization changes how you use it. A well-aligned model with poor routing is less effective than a moderately-aligned model with intelligent routing. In practice, the highest-value improvements come from doing both: align the model well, then continuously optimize the serving decisions that determine which requests reach which model with which prompt.

vLLM is the execution layer where policy decisions become GPU computation. llama.cpp is the local sandbox where reward functions and bandit algorithms are prototyped before production deployment. The chapters that follow cover the specific LLM inference mechanics those execution layers implement.

---

## Companion Code

`code/chapter_24/serving_policy_demo.py` implements:

- `EpsilonGreedyBandit`, `UCBBandit`, `ThompsonSamplingBandit` — core bandit algorithms
- `ContextualBandit` — linear contextual bandit with feature bucketing
- `ModelRoutingSimulation` — simulates a three-tier routing decision over 10,000 requests
- `OfflinePolicyEvaluator` — IPS and DR offline estimators
- `reward_function` — composite reward with latency and safety constraints
- `RewardHackingDetector` — monitors for divergence between reward metric and proxy metrics

`code/chapter_24/serving_policy_demo.cpp` implements the same algorithms in C++, with numerical examples demonstrating Thompson Sampling convergence and IPS variance reduction.

---

## References

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction.* MIT Press.
- Lattimore & Szepesvári (2020). *Bandit Algorithms.* Cambridge University Press.
- Dudík et al. (2011). *Doubly Robust Policy Evaluation and Learning.* ICML 2011.
- Langford & Zhang (2007). *The Epoch-Greedy Algorithm for Contextual Multi-Armed Bandits.* NeurIPS 2007.
- Li et al. (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation.* WWW 2010.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Rafailov et al. (2023). *Direct Preference Optimization.* arXiv:2305.18290.
- Achiam (2018). *Spinning Up in Deep RL.* OpenAI.


---

## Chapter Summary

- **RLHF in the serving stack**: online RLHF (GRPO, PPO) requires the policy model to serve tokens to a reward model and receive gradient updates in real time — this blurs the boundary between training and inference.
- **GRPO**: Group Relative Policy optimization generates G completions per prompt, scores them with a reward model, and updates the policy using the relative rankings; serving must batch G completions efficiently.
- **Reward model serving**: the reward model is a separate endpoint; latency between the inference pod and reward pod determines the training step time.
- **KV cache invalidation during training**: a policy update changes the model weights, invalidating all cached KV blocks; prefix caching must be flushed or tagged with a model version.
- **Rollout vs update phases**: during rollout, the policy is frozen; during the update, weights change and traffic must be drained first to avoid mixed-version serving.
- **vLLM as a rollout server**: vLLM's `--enforce-eager` flag disables CUDA graph caching, which is required when weights are updated mid-serving.
- **Throughput requirement**: GRPO with G=8 completions per prompt requires 8× the token generation capacity of standard inference; this often drives the need for speculative decoding in the rollout phase.

---

## Self-Check Questions

1. GRPO generates G=8 completions per prompt and ranks them by reward. If a prompt batch has 32 prompts, how many sequences does the vLLM scheduler see simultaneously? How does this affect the KV cache budget? *(Section 25.2)*

2. A policy update invalidates all prefix-cached KV blocks. The cache hit rate before the update was 70%. Estimate the impact on TTFT for the first 60 seconds after a policy update. *(Section 25.4)*

3. You are running online RLHF with a separate reward model at `http://reward:8000`. The reward model has a P99 latency of 250 ms. What is the minimum training step time if each step requires one reward evaluation per completion? *(Section 25.3)*

4. `--enforce-eager` disables CUDA graph replay. For a 70B model, decode throughput drops by approximately 30% (typical CUDA graph speedup). Quantify the impact on GRPO rollout time for a batch of 256 prompts × 8 completions × 200 tokens each. *(Section 25.5)*

5. You need to serve the policy model to users while simultaneously running GRPO updates. Describe a blue-green deployment strategy that prevents users from seeing a mid-update model. *(Section 25.6)*


---

## Worked Solutions

### Question 1
**GRPO: G=8 completions per prompt, batch=32 prompts. Total sequences for scheduler:**

```
total_sequences = 32 prompts x 8 completions = 256 sequences
```

**KV cache budget impact:**
Each of the 256 sequences needs its own KV cache blocks. If each sequence generates 200 tokens average, and KV cost per token = 327 KB (70B model):
```
total_KV = 256 x 200 x 327 KB = 256 x 65.4 MB = 16.75 GB
```
This is a 256x increase over serving a single sequence. The scheduler must either:

1. Set max_num_seqs >= 256 and ensure the KV pool can hold all 256 sequences simultaneously, or
2. Process the 32 prompts in smaller sub-batches (G=2 or G=4) to fit within the KV budget.

GRPO training is therefore far more memory-intensive than inference serving. A typical 4x A100 serving deployment (max_num_seqs=128) must be reconfigured with larger KV reservations for GRPO rollouts.

---

### Question 2
**Policy update invalidates prefix-cached KV blocks. Cache hit rate was 70%. Impact on TTFT for first 60 seconds:**

**Before update:** 70% of requests skip prefill entirely (cache hit). TTFT for hits ~= 0 ms (just decode start). Average TTFT = 0.70 x 0 + 0.30 x T_prefill where T_prefill ~= 500 ms (assumed).
```
avg_TTFT_before = 0.30 x 500 ms = 150 ms
```

**After policy update:** Cache is invalidated. All new blocks must be computed. Hit rate drops to 0% immediately, recovering to ~70% as the new cache warms up. Warmup time depends on request rate -- at 100 req/min, after 60 s (100 requests), the most frequent prefixes are re-cached.

```
avg_TTFT_first_60s = 1.0 x 500 ms = 500 ms  (all cache misses)
```

**TTFT increase:** 500 ms vs 150 ms = 3.3x worse TTFT for the first 60 seconds after update. For production RLHF, schedule policy updates during low-traffic windows or pre-warm the cache with synthetic requests before re-enabling production traffic.

---

### Question 3
**Online RLHF: reward model P99 latency = 250 ms. Minimum training step time:**

Each training step requires at least one reward evaluation per completion. The reward model call is on the critical path:

```
min_step_time >= reward_model_P99_latency = 250 ms
```

But the step also includes:

- Forward pass through policy model (vLLM inference): ~100-500 ms depending on sequence length
- Backward pass (gradient computation): ~2-5x forward pass time
- Optimizer step: ~10-50 ms

**Minimum step time** (reward model is the bottleneck only if it exceeds all other components):
```
min_step_time = max(250 ms, forward_pass + backward_pass + optimizer) 
             >= 250 ms
```

For a 70B model with 200-token completions: forward ~= 200 x 70 ms = 14 s, backward ~= 28 s. The reward model 250 ms P99 is NOT the bottleneck -- the training backward pass is ~112x slower. However, if 256 completions need reward evaluation sequentially, total reward time = 256 x 250 ms = 64 s, which DOES become the bottleneck.

**Solution:** Parallelize reward evaluation across completions using a batched reward model API or multiple reward model replicas.

---

### Question 4
**`--enforce-eager` drops decode throughput 30%. GRPO rollout: 256 prompts x 8 completions x 200 tokens.**

**Total tokens to generate:**
```
total_tokens = 256 x 8 x 200 = 409,600 tokens
```

**Throughput without CUDA graphs (30% penalty):**
If baseline throughput = 1,800 tok/s, without graphs: 1,800 x 0.70 = 1,260 tok/s.

**Rollout time comparison:**
```
with CUDA graphs:    409,600 / 1,800 = 227.6 s
without CUDA graphs: 409,600 / 1,260 = 325.1 s
```

**Additional time from --enforce-eager:**
```
delta = 325.1 - 227.6 = 97.5 s per rollout batch
```

For GRPO training with 100 update steps, the overhead is 97.5 x 100 = 9,750 s = 2.7 hours of extra rollout time. Re-enable CUDA graphs for GRPO unless the reason for --enforce-eager (e.g., dynamic control flow in the model) is mandatory.

---

### Question 5
**Blue-green deployment for serving policy while running GRPO updates:**

**Setup:**
- Blue: current stable policy model serving user traffic
- Green: model being updated via GRPO

**Strategy:**

1. **Deploy Blue:** Start Blue on GPU cluster A, serving 100% of user traffic. Blue is frozen -- no weight updates.

2. **GRPO on separate cluster:** Run GRPO rollouts and updates on GPU cluster B using a copy of the Blue weights as the starting point. Cluster B never receives user traffic.

3. **Checkpoint:** After N GRPO update steps, save the updated Green checkpoint to object storage.

4. **Parallel validation:** Load Green onto a shadow cluster C. Run quality eval (LLM judge, perplexity) against the eval set. If metrics pass (score >= Blue threshold), proceed to swap.

5. **Atomic swap:** Update the load balancer to route 100% of traffic from Blue to Green in a single atomic operation. Green becomes the new Blue. Old Blue cluster is freed for the next GRPO cycle.

**Key invariant:** Users never see a model that is mid-update. The GRPO update modifies only the Green copy; Blue is immutable throughout. The swap is instantaneous from the load balancer's perspective.

**Additional safeguard:** Keep Blue running for 15 minutes after the swap with 0% traffic. If Green shows elevated error rate (P99 TTFT spike, quality regression), immediately revert by updating load balancer back to Blue.

