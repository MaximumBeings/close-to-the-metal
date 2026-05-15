# Chapter 26 — CS336 Assignment 5 Alignment Field Guide

> *"The distance between 'I understand the GRPO paper' and 'my training loop converges' is exactly the distance between reading this chapter and having run the code."*

Chapters 23 and 24 established the theory and serving implications of alignment training. This chapter is the implementation guide — a field manual for building the CS336 Assignment 5 pipeline from scratch. It covers the mandatory reasoning-RL track (zero-shot baseline → SFT → Expert Iteration → GRPO) and the optional instruction-tuning supplement (SFT on UltraChat → DPO on HH preference data). For each deliverable in the assignment, this chapter provides a checklist, at least one worked numerical example, and a self-contained Python harness you can run without a GPU cluster.

The chapter makes one architectural point explicit on every page: **vLLM generates; the training loop updates.** These are separate processes, often running on separate GPUs. vLLM never touches gradients. The training loop never touches the vLLM inference engine except to push updated weights into it after each checkpoint.

---

## 26.1 The Two-GPU Mental Model

Before any code, get this diagram into your head:

```
GPU 0 (Training GPU)                      GPU 1 (Inference GPU)
─────────────────────────────             ─────────────────────────────
HuggingFace model (policy)                vLLM engine
  - .forward() → logits                     - .generate() → text tokens
  - .backward() → gradients                 - PagedAttention KV cache
  - optimizer.step() → new weights          - CUDA kernels (FlashAttn)
  - NO generation, NO sampling              - NO gradients, NO .backward()

Weight sync direction:  GPU 0 ──────────────────────────────► GPU 1
                       (after each checkpoint, push via load_policy_into_vllm)
```

The training loop on GPU 0 uses `torch.autograd` to compute gradients and update weights. vLLM on GPU 1 uses highly optimized CUDA kernels to generate text at maximum throughput. They share no PyTorch computational graph. The only coupling is the periodic weight synchronization after each training step.

When you see an error like `RuntimeError: Expected all tensors to be on the same device`, it almost always means you accidentally passed a training-GPU tensor to the vLLM engine or vice versa.

---

## 26.2 Assignment Structure and Deliverables at a Glance

### Mandatory Track (Reasoning RL)

| Section | Problem Name | Points | What You Build |
|---------|-------------|--------|---------------|
| §3 | `math_baseline` | 4 | Zero-shot evaluation with vLLM |
| §4.2 | `tokenize_prompt_and_output` | 2 | Tokenization with response_mask |
| §4.2 | `compute_entropy` | 1 | Per-token entropy |
| §4.2 | `get_response_log_probs` | 2 | Log-prob extraction |
| §4.2 | `masked_normalize` | 1 | Masked sum with constant |
| §4.2 | `sft_microbatch_train_step` | 3 | Single SFT gradient step |
| §4.2 | `log_generations` | 1 | Training-loop logging helper |
| §4.3 | `sft_experiment` | 2 | Full SFT run on MATH |
| §5 | `expert_iteration_experiment` | 2 | EI loop on MATH |
| §7.2 | `compute_group_normalized_rewards` | 2 | Group-norm advantages |
| §7.2 | `compute_naive_policy_gradient_loss` | 1 | -A * log π |
| §7.2 | `compute_grpo_clip_loss` | 2 | Clipped GRPO objective |
| §7.2 | `compute_policy_gradient_loss` | 1 | Loss type dispatcher |
| §7.2 | `masked_mean` | 1 | Masked mean |
| §7.2 | `grpo_microbatch_train_step` | 3 | Single GRPO gradient step |
| §7.2 | `grpo_train_loop` | 5 | Full GRPO training loop |
| §8 | GRPO experiments | 15 | Ablations + leaderboard |

### Optional Supplement (Instruction Tuning / DPO)

| Section | Problem Name | Points | What You Build |
|---------|-------------|--------|---------------|
| §2.1–2.4 | Baselines | 16 | MMLU, GSM8K, AlpacaEval, SST |
| §3.2 | `data_loading` | 3 | Packed SFT dataset |
| §3.2 | `sft_script` | 4 | Instruction tuning loop |
| §4 | Evaluation | 16 | Benchmarks post-SFT |
| §5.2 | `look_at_hh` | 2 | HH data analysis |
| §5.3 | `dpo_loss` | 2 | Per-instance DPO loss |
| §5.4 | `dpo_training` | 4 | DPO training loop |

---

## 26.3 Problem `math_baseline` — Zero-Shot Evaluation

### 26.3.1 Checklist

- [ ] Load MATH validation split from `validation.jsonl`
- [ ] Format each example with the r1_zero prompt
- [ ] Initialize vLLM on a single GPU with `enable_prefix_caching=True`
- [ ] Set `stop=["</answer>"]` and `include_stop_str_in_output=True`
- [ ] Run `llm.generate(prompts, sampling_params)` in batches
- [ ] Score each output with `r1_zero_reward_fn(response, ground_truth)`
- [ ] Serialize results to JSONL for later analysis
- [ ] Report format_reward rate, answer_reward rate, total accuracy

### 26.3.2 The r1_zero Prompt (Verbatim)

```
A conversation between User and Assistant. The User asks a question, and the
Assistant solves it. The Assistant first thinks about the reasoning process in
the mind and then provides the User with the answer. The reasoning process is
enclosed within <think> </think> and answer is enclosed within <answer> </answer>
tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.

User: {question}
Assistant: <think>
```

**Critical:** The prompt ends with `<think>` already open. The model generates the interior of the thinking block, then closes it with `</think>`, then generates `<answer>...</answer>`. If you omit the trailing `<think>`, you will see format_reward ≈ 0.3 instead of ≈ 0.85.

### 26.3.3 vLLM Setup (Minimal)

```python
from vllm import LLM, SamplingParams
from unittest.mock import patch

def init_vllm(model_path: str, device: str = "cuda:1", seed: int = 42) -> LLM:
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch  = patch(
        "vllm.worker.worker.Worker"
        "._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_path,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=0.85,
            seed=seed,
        )

EVAL_PARAMS = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True,   # ← without this: format_reward = 0
)
```

### 26.3.4 Worked Example: Three Output Categories

**Category A — Both rewards = 1:**
```
Response: <think>
  Natalia sold 48 clips in April. In May she sold 48/2 = 24 clips.
  Total = 48 + 24 = 72.
</think>
<answer> 72 </answer>

reward_fn output: {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0}
```

**Category B — format_reward=1, answer_reward=0:**
```
Response: <think>
  48 + 48/2 = 48 + 24 = 60
</think>
<answer> 60 </answer>

reward_fn output: {"reward": 0.0, "format_reward": 1.0, "answer_reward": 0.0}
```
The model followed the format correctly but made an arithmetic error. This is the majority of failures early in training.

**Category C — format_reward=0, answer_reward=0:**
```
Response: Natalia sold 48 clips in April and 24 in May. So she sold 72 clips
altogether.

reward_fn output: {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
```
The model ignored the `<think>` continuation and answered in prose. Usually caused by the model having a strong pretrained bias toward direct Q&A format. With the r1_zero prompt and Qwen 2.5 Math 1.5B, Category C accounts for roughly 10–15% of zero-shot outputs.

**Is Category C a parser bug or a model issue?** For Qwen 2.5 Math 1.5B, it is a model issue. The reward function correctly identifies that the tags are missing. You can confirm this by checking whether the raw response string contains `<answer>` — if not, the model failed to generate the format.

### 26.3.5 Evaluation Script (Skeleton)

```python
import json
from pathlib import Path

def run_zero_shot_eval(
    model_path: str,
    data_path: str,
    output_path: str,
    reward_fn,
    n_examples: int | None = None,
) -> dict:
    # Load data
    examples = [json.loads(l) for l in Path(data_path).read_text().splitlines()]
    if n_examples:
        examples = examples[:n_examples]

    # Format prompts
    r1_zero_template = Path("prompts/r1_zero.prompt").read_text()
    prompts = [r1_zero_template.format(question=ex["problem"]) for ex in examples]
    gts     = [ex["answer"] for ex in examples]

    # Generate
    llm = init_vllm(model_path)
    outputs = llm.generate(prompts, EVAL_PARAMS)

    # Score and serialize
    records = []
    for ex, gt, out in zip(examples, gts, outputs):
        response = out.outputs[0].text
        info = reward_fn(response, gt)
        records.append({"problem": ex["problem"], "response": response,
                         "ground_truth": gt, **info})

    Path(output_path).write_text("\n".join(json.dumps(r) for r in records))

    n = len(records)
    return {
        "n":             n,
        "answer_acc":    sum(r["answer_reward"] for r in records) / n,
        "format_rate":   sum(r["format_reward"]  for r in records) / n,
        "total_reward":  sum(r["reward"]         for r in records) / n,
    }
```

---

## 26.4 Problem `tokenize_prompt_and_output`

### 26.4.1 Checklist

- [ ] Tokenize prompt and output *separately* (do not tokenize their concatenation)
- [ ] Concatenate token IDs: `full_ids = prompt_ids + output_ids`
- [ ] Pad to max length in the batch with `pad_token_id`
- [ ] `input_ids = full_ids_tensor[:, :-1]` (drop last token)
- [ ] `labels    = full_ids_tensor[:, 1:]`  (drop first token)
- [ ] Set `response_mask[i, prompt_len-1 : prompt_len-1+output_len] = 1`
- [ ] Verify: `response_mask.sum(dim=1)` equals `[len(output_ids[i]) for i in batch]`

### 26.4.2 The Off-By-One (Explained Numerically)

Suppose prompt = `[10, 11, 12]` and output = `[20, 21, 22, 23]`.

Full sequence: `[10, 11, 12, 20, 21, 22, 23]` — length 7.

After shift:
```
input_ids: [10, 11, 12, 20, 21, 22]    (positions 0–5, predicts next token)
labels:    [11, 12, 20, 21, 22, 23]    (positions 1–6, the actual next tokens)
```

At position 2 in `input_ids`, the value is 12 (last prompt token). The label at position 2 is 20 (first output token). **This is the first response prediction.** So `response_mask` should be 1 starting at position 2, not position 3.

```
response_mask: [0, 0, 1, 1, 1, 1]
                          ↑
              position = prompt_len - 1 = 3 - 1 = 2
```

This is the source of the `prompt_len - 1` formula. If you set `response_mask[i, prompt_len:]`, you skip the first response token and the model never learns to generate the opening of the response.

### 26.4.3 Self-Contained Harness

```python
import torch

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    prompt_ids_list = [tokenizer.encode(p)                        for p in prompt_strs]
    output_ids_list = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    full_ids_list   = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    max_len         = max(len(s) for s in full_ids_list)
    padded          = [s + [pad_id] * (max_len - len(s)) for s in full_ids_list]
    ids_t           = torch.tensor(padded, dtype=torch.long)      # (B, max_len)
    input_ids       = ids_t[:, :-1]
    labels          = ids_t[:, 1:]
    response_mask   = torch.zeros_like(labels)
    for i, (p_ids, o_ids) in enumerate(zip(prompt_ids_list, output_ids_list)):
        start = len(p_ids) - 1
        end   = start + len(o_ids)
        response_mask[i, start:end] = 1
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

# ── Verification ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    result = tokenize_prompt_and_output(
        ["Solve: 2+2="],
        ["<think>\n2+2=4\n</think>\n<answer> 4 </answer>"],
        tok,
    )
    n_response_tokens = (result["labels"][0] !=
                         (tok.pad_token_id or tok.eos_token_id)).sum()
    mask_sum = result["response_mask"][0].sum()
    # mask_sum should equal the number of output tokens
    assert mask_sum > 0, "response_mask is all zeros — check the off-by-one"
    print(f"Response tokens masked: {mask_sum.item()}")
    print("tokenize_prompt_and_output: OK")
```

---

## 26.5 Problems `compute_entropy`, `get_response_log_probs`, `masked_normalize`

### 26.5.1 compute_entropy

```python
import torch, torch.nn.functional as F

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """(B, T, V) → (B, T) per-token entropy."""
    log_p = F.log_softmax(logits, dim=-1)    # numerically stable
    return -(log_p.exp() * log_p).sum(dim=-1)

# Quick sanity check: uniform distribution over V=4 → entropy = log(4)
if __name__ == "__main__":
    import math
    logits = torch.zeros(1, 3, 4)           # uniform over 4 tokens
    H = compute_entropy(logits)
    expected = math.log(4)
    assert abs(H[0, 0].item() - expected) < 1e-5, f"Expected {expected}, got {H[0,0].item()}"
    print(f"Uniform entropy over 4 tokens: {H[0,0].item():.4f} (expected {expected:.4f})")
    print("compute_entropy: OK")
```

**Why log_softmax instead of softmax().log()?** `softmax` may produce exact zeros for large negative logits, and `log(0) = -inf`. `log_softmax` uses the logsumexp trick to avoid this numerical issue.

### 26.5.2 get_response_log_probs

```python
def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits    = model(input_ids).logits                     # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)               # (B, T, V)
    # Gather the log-prob of the actual label at each position
    gathered  = log_probs.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)                                           # (B, T)
    result = {"log_probs": gathered}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result
```

**Test:** For a deterministic model that always predicts a single token with probability 1, `log_probs` should be 0.0 for the correct label and -inf elsewhere. For a random model, `log_probs` should be approximately `-log(V)` on average.

### 26.5.3 masked_normalize

```python
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    masked = tensor * mask
    total  = masked.sum()    if dim is None else masked.sum(dim=dim)
    return total / normalize_constant

# ── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = torch.tensor([[1., 2., 3., 4.]])
    m = torch.tensor([[1., 1., 0., 1.]])     # ignore position 2
    result = masked_normalize(t, m, normalize_constant=3.0, dim=1)
    # sum of masked = 1+2+4 = 7; divided by 3 = 2.333...
    assert abs(result.item() - 7/3) < 1e-5
    print(f"masked_normalize: {result.item():.4f} (expected {7/3:.4f})")
    print("masked_normalize: OK")
```

---

## 26.6 Problem `sft_microbatch_train_step`

### 26.6.1 Checklist

- [ ] Compute NLL: `-policy_log_probs` weighted by `response_mask`
- [ ] Normalize: divide by `(normalize_constant × batch_size)` or use `masked_normalize`
- [ ] Scale by `1 / gradient_accumulation_steps`
- [ ] Call `loss.backward()`
- [ ] Return `loss.detach()` and a metadata dict

### 26.6.2 Implementation

```python
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,   # (B, T)
    response_mask: torch.Tensor,      # (B, T)
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    B = policy_log_probs.shape[0]
    nll = masked_normalize(
        -policy_log_probs, response_mask,
        normalize_constant=normalize_constant * B,
    )
    loss = nll / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), {"sft_nll": nll.item()}
```

### 26.6.3 What `loss.backward()` Does Here

After `loss.backward()`, each parameter tensor `θ.grad` contains the gradient of `loss` with respect to `θ`. The gradient accumulation pattern collects these across `gradient_accumulation_steps` microbatches before calling `optimizer.step()`. The division by `gradient_accumulation_steps` ensures that the accumulated gradient is the *average* gradient across the full logical batch, not the sum.

**Common mistake:** Calling `optimizer.zero_grad()` *inside* the microbatch loop instead of *outside*. This resets gradients after every microbatch, so only the last microbatch's gradient is used.

```python
# WRONG: zeros gradients inside the accumulation loop
for microbatch in microbatches:
    log_probs = get_response_log_probs(...)["log_probs"]
    sft_microbatch_train_step(log_probs, mask, grad_accum_steps)
    optimizer.zero_grad()    # ← BUG: erases previous microbatch gradients
optimizer.step()

# CORRECT: zeros gradients outside the accumulation loop
optimizer.zero_grad()        # ← once, before all microbatches
for microbatch in microbatches:
    log_probs = get_response_log_probs(...)["log_probs"]
    sft_microbatch_train_step(log_probs, mask, grad_accum_steps)
optimizer.step()             # ← once, after all microbatches
```

---

## 26.7 Problem `sft_experiment` — Full SFT Run

### 26.7.1 Dataset Sizes and Expected Outcomes

| Dataset size | Expected val accuracy | Notes |
|-------------|----------------------|-------|
| 128 examples | 5–8% | Format reward improves; answer barely moves |
| 256 examples | 8–12% | Slight improvement |
| 512 examples | 12–15% | Model starts generating correct structure |
| 1024 examples | 15–18% | Clear improvement over zero-shot |
| Full (~4K) | 20–28% | Solid SFT performance |
| Full (filtered) | 25–32% | Filtered-correct SFT typically beats unfiltered |

**Filtering for correctness:** Before training, run the vLLM model on the SFT data and keep only examples where the generated answer matches the ground truth. This "correctness filter" reduces dataset size by 30–50% but concentrates the training signal.

### 26.7.2 Gradient Clipping

Always clip gradients for SFT:

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
```

Without clipping, large gradient updates early in training can damage the model's pretrained representations irreversibly. The loss will drop initially (the model is being pushed in the gradient direction aggressively) but then spike when the representations become incoherent.

### 26.7.3 WandB Metrics Setup

```python
import wandb
wandb.define_metric("train_step")
wandb.define_metric("eval_step")
wandb.define_metric("train/*", step_metric="train_step")
wandb.define_metric("eval/*",  step_metric="eval_step")

# In training loop:
wandb.log({"train/loss": loss.item(), "train_step": step})

# In evaluation loop:
wandb.log({"eval/answer_acc": metrics["answer_acc"], "eval_step": eval_step})
```

Using separate step metrics for train and eval allows you to plot them with different x-axes in the WandB dashboard.

---

## 26.8 Problem `expert_iteration_experiment`

### 26.8.1 The Loop in Code

```
for ei_step in range(n_ei_steps):
    # 1. Sample questions
    questions = sample(train_data, n_prompts_per_batch)

    # 2. Generate G rollouts per question (vLLM)
    prompts_repeated = [format_prompt(q) for q in questions for _ in range(G)]
    gts_repeated     = [q["answer"]      for q in questions for _ in range(G)]
    outputs = vllm.generate(prompts_repeated, rollout_params)

    # 3. Filter: keep only correct rollouts
    sft_data = []
    for prompt, gt, output in zip(prompts_repeated, gts_repeated, outputs):
        response = output.outputs[0].text
        if reward_fn(response, gt)["answer_reward"] == 1.0:
            sft_data.append((prompt, response))

    # 4. SFT on correct rollouts
    if len(sft_data) > 0:
        for epoch in range(sft_epochs):
            for batch in dataloader(sft_data, batch_size):
                # ... tokenize, forward, backward, step
    
    # 5. Sync weights to vLLM
    load_policy_into_vllm_instance(policy, vllm_engine)
```

### 26.8.2 The `min_tokens=4` Requirement

```python
rollout_params = SamplingParams(
    temperature=1.0,
    max_tokens=1024,
    min_tokens=4,                       # ← prevents empty responses
    n=G,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)
```

Without `min_tokens=4`, the model can generate an empty response (just `</answer>`) which immediately hits the stop token. An empty response produces an empty `output_ids` list. If you then call `tokenize_prompt_and_output` with `output_str=""`, you get a response_mask of all zeros. `masked_normalize(loss, all_zeros_mask)` computes `0 / constant = 0`, and `0.backward()` produces zero gradients — the update is silently dropped. This is not an error; it is a silent correctness bug.

### 26.8.3 EI vs. GRPO Entropy Comparison

A key deliverable in the assignment is comparing entropy under EI and GRPO. The expected pattern:

- **EI:** Entropy decreases monotonically. The model is supervised on its own correct outputs repeatedly, narrowing its distribution. By iteration 5, entropy on reasoning chains is 20–35% lower than at initialization.
- **GRPO:** Entropy may initially decrease then stabilize, or decrease more slowly. GRPO updates the policy toward higher-reward outputs and away from lower-reward outputs simultaneously, which can prevent the mode collapse that EI induces.

If your EI entropy is *not* decreasing, check that you are computing entropy on response tokens only (using `response_mask`), not on prompt tokens.

---

## 26.9 Policy Gradient Foundations (Quick Reference)

These are the four identities you need to implement the GRPO losses. If any of them is unclear, Chapter 24 §24.2–24.5 provides the full derivations.

**Identity 1 — REINFORCE gradient:**
```
∇_θ J(π_θ) = E_{τ~π} [ Σ_t ∇_θ log π_θ(a_t | s_t) · R(τ) ]
```
*Implementation:* `pg_loss = -(log_prob * R).sum()` then `pg_loss.backward()`

**Identity 2 — Baseline subtraction is unbiased:**
```
E_{a~π}[∇_θ log π_θ(a|s) · b(s)] = 0  for any b independent of a
```
*Implementation:* subtract group mean reward from each reward before computing loss

**Identity 3 — Importance sampling for off-policy correction:**
```
ratio = π_θ(a | s) / π_θ_old(a | s) = exp(log π_θ - log π_θ_old)
```
*Implementation:* `ratio = (policy_log_probs - old_log_probs.detach()).exp()`

**Identity 4 — PPO/GRPO clip function:**
```
clipped_objective = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```
*Implementation:* `torch.min(ratio * adv, ratio.clamp(1-eps, 1+eps) * adv)`

---

## 26.10 Problem `compute_group_normalized_rewards`

### 26.10.1 Checklist

- [ ] Call `reward_fn(response, gt)` for each of the `B = n_questions × G` responses
- [ ] Reshape rewards into `(n_questions, G)` groups
- [ ] Compute per-group mean and (optionally) std
- [ ] Normalize: `A_i = (r_i - mean) / (std + eps)` or `A_i = r_i - mean`
- [ ] Return advantages as flat tensor of shape `(B,)`, raw rewards `(B,)`, metadata dict

### 26.10.2 Worked Numerical Example

Six responses for 2 questions, G=3. Raw rewards:

```
Question 0: [1, 0, 1]   → mean=0.667, std=0.471
Question 1: [0, 0, 0]   → mean=0.000, std=0.000
```

With `normalize_by_std=True`:
```
Question 0 advantages: [(1-0.667)/(0.471+1e-6), (0-0.667)/(0.471+1e-6), (1-0.667)/(0.471+1e-6)]
                     = [0.707, -1.414, 0.707]
Question 1 advantages: [(0-0)/(0+1e-6), (0-0)/(0+1e-6), (0-0)/(0+1e-6)]
                     = [0, 0, 0]   (effectively; std≈0 so normalization is ~1/eps ≈ large,
                                     but numerator is 0)
```

With `normalize_by_std=False` (Dr. GRPO):
```
Question 0 advantages: [1-0.667, 0-0.667, 1-0.667] = [0.333, -0.667, 0.333]
Question 1 advantages: [0, 0, 0]
```

**Observation:** All-zero groups produce zero advantages under both methods. This is correct — when every rollout from a question is wrong, there's no differential signal and the gradient contribution should be zero.

### 26.10.3 Implementation

```python
import torch
from typing import Callable

def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    n = len(rollout_responses)
    assert n % group_size == 0

    raw = torch.tensor(
        [reward_fn(r, g)["reward"]
         for r, g in zip(rollout_responses, repeated_ground_truths)],
        dtype=torch.float32,
    )
    advantages = torch.zeros_like(raw)
    n_q = n // group_size
    for q in range(n_q):
        s, e  = q * group_size, (q + 1) * group_size
        group = raw[s:e]
        mean  = group.mean()
        if normalize_by_std:
            std = group.std() + advantage_eps
            advantages[s:e] = (group - mean) / std
        else:
            advantages[s:e] = group - mean

    return advantages, raw, {
        "mean_reward":  raw.mean().item(),
        "frac_correct": (raw > 0.5).float().mean().item(),
    }
```

---

## 26.11 Problem `compute_naive_policy_gradient_loss`

```python
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,  # (B, 1)
    policy_log_probs: torch.Tensor,           # (B, T)
) -> torch.Tensor:                            # (B, T)
    return -raw_rewards_or_advantages * policy_log_probs  # broadcasts (B,1)→(B,T)
```

**Numerical test:**

```python
if __name__ == "__main__":
    adv      = torch.tensor([[2.0]])          # positive advantage → reduce loss
    log_prob = torch.tensor([[-0.5, -1.0]])   # 2 tokens
    loss     = compute_naive_policy_gradient_loss(adv, log_prob)
    # expected: [[-2*(-0.5), -2*(-1.0)]] = [[1.0, 2.0]]
    assert loss.shape == (1, 2)
    assert abs(loss[0, 0].item() - 1.0) < 1e-6
    assert abs(loss[0, 1].item() - 2.0) < 1e-6
    print("compute_naive_policy_gradient_loss: OK")
```

---

## 26.12 Problem `compute_grpo_clip_loss`

### 26.12.1 Checklist

- [ ] Compute `log_ratio = policy_log_probs - old_log_probs.detach()`
- [ ] Compute `ratio = log_ratio.exp()`
- [ ] Compute `unclipped = ratio * advantages`
- [ ] Compute `clipped = ratio.clamp(1-eps, 1+eps) * advantages`
- [ ] `per_token_loss = -torch.min(unclipped, clipped)`
- [ ] Track and return `clip_fraction` in metadata

### 26.12.2 Numerical Walkthrough

Single token, advantage = +1.0, epsilon = 0.2.

| Scenario | ratio | unclipped | clipped | min | loss |
|----------|-------|-----------|---------|-----|------|
| Policy unchanged | 1.0 | 1.0 | 1.0 | 1.0 | -1.0 |
| Policy slightly better (ratio=1.1) | 1.1 | 1.1 | 1.1 | 1.1 | -1.1 |
| Policy much better (ratio=1.5) | 1.5 | 1.5 | **1.2** | 1.2 | **-1.2** (clipped) |
| Policy worse (ratio=0.8) | 0.8 | 0.8 | 0.8 | 0.8 | -0.8 |
| Policy much worse (ratio=0.5) | 0.5 | 0.5 | **0.8** | 0.5 | -0.5 |

The clip prevents the policy from gaining more than (1+ε)A reward from making a token much more likely, disincentivizing large updates. When advantage is negative, the clip prevents the policy from being penalized beyond (1-ε)A for making a token slightly less likely.

### 26.12.3 Implementation

```python
def compute_grpo_clip_loss(
    advantages: torch.Tensor,       # (B, 1)
    policy_log_probs: torch.Tensor, # (B, T)
    old_log_probs: torch.Tensor,    # (B, T)
    cliprange: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    log_ratio = policy_log_probs - old_log_probs.detach()
    ratio     = log_ratio.exp()
    adv_exp   = advantages.expand_as(ratio)
    unclipped = ratio * adv_exp
    clipped   = ratio.clamp(1 - cliprange, 1 + cliprange) * adv_exp
    loss      = -torch.min(unclipped, clipped)
    is_clipped = (ratio < 1 - cliprange) | (ratio > 1 + cliprange)
    return loss, {"clip_fraction": is_clipped.float().mean().item()}
```

---

## 26.13 Problem `masked_mean`

```python
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum().clamp(min=1)
    return masked.sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)

# Test: two sequences, different lengths
if __name__ == "__main__":
    t    = torch.tensor([[2., 2., 2., 2., 0., 0.],   # 4 active tokens
                         [2., 2., 2., 2., 2., 2.]])   # 6 active tokens
    mask = torch.tensor([[1., 1., 1., 1., 0., 0.],
                         [1., 1., 1., 1., 1., 1.]])
    # Per-sequence means: both should be 2.0
    result = masked_mean(t, mask, dim=1)
    assert result[0].item() == 2.0 and result[1].item() == 2.0
    print("masked_mean per-sequence: OK")
```

---

## 26.14 Problem `grpo_microbatch_train_step`

### 26.14.1 Checklist

- [ ] Call `compute_policy_gradient_loss` with the correct `loss_type`
- [ ] Call `masked_mean(per_token_loss, response_mask, dim=1)` → shape `(B,)`
- [ ] Take `.mean()` over batch dimension → scalar
- [ ] Divide by `gradient_accumulation_steps`
- [ ] Call `.backward()`
- [ ] Return `loss.detach()` and metadata

### 26.14.2 The Full Dispatcher

```python
from typing import Literal

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards:  torch.Tensor | None = None,
    advantages:   torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs,
                                      old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict]:
    per_token_loss, meta = compute_policy_gradient_loss(
        policy_log_probs, loss_type,
        raw_rewards=raw_rewards, advantages=advantages,
        old_log_probs=old_log_probs, cliprange=cliprange,
    )
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), meta
```

---

## 26.15 Problem `grpo_train_loop` — The Full GRPO Pipeline

### 26.15.1 Annotated Skeleton

```python
def grpo_train_loop(policy, vllm_engine, train_data, val_data, cfg):
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.95)
    )
    rollout_params = SamplingParams(
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        min_tokens=4, n=cfg.group_size,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    for step in range(cfg.n_steps):
        # ── Step 1: Sample questions ────────────────────────────────────────
        n_q = cfg.rollout_batch_size // cfg.group_size
        questions = sample_without_replacement(train_data, n_q)

        # ── Step 2: Generate rollouts (vLLM, GPU 1) ─────────────────────────
        prompts = [format_r1zero(q["problem"]) for q in questions]
        gts     = [q["answer"]                 for q in questions]
        # Each prompt generates cfg.group_size responses; outputs is len(prompts)
        # but each output has G completions in output.outputs[0..G-1]
        raw_outputs = vllm_engine.generate(prompts, rollout_params)

        # Flatten to (n_q * G,) lists
        rollout_responses = [
            out.outputs[g].text
            for out in raw_outputs
            for g in range(cfg.group_size)
        ]
        repeated_gts = [gt for gt in gts for _ in range(cfg.group_size)]

        # ── Step 3: Compute advantages ──────────────────────────────────────
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=cfg.group_size,
            advantage_eps=cfg.advantage_eps,
            normalize_by_std=cfg.normalize_by_std,
        )

        # ── Step 4: [GRPO-Clip only] Compute old log-probs ─────────────────
        old_log_probs_list = None
        if cfg.loss_type == "grpo_clip":
            old_log_probs_list = []
            repeated_prompts = [p for p in prompts for _ in range(cfg.group_size)]
            with torch.inference_mode():
                for p, r in zip(repeated_prompts, rollout_responses):
                    tok = tokenize_prompt_and_output([p], [r], tokenizer)
                    lp  = get_response_log_probs(
                        policy,
                        tok["input_ids"].to(policy.device),
                        tok["labels"].to(policy.device),
                    )["log_probs"]
                    old_log_probs_list.append(lp.cpu())

        # ── Step 5: Inner training loop ─────────────────────────────────────
        policy.train()
        optimizer.zero_grad()
        micro_bs = cfg.rollout_batch_size // cfg.gradient_accumulation_steps

        for epoch in range(cfg.epochs_per_rollout_batch):
            for mb_start in range(0, cfg.rollout_batch_size, micro_bs):
                mb_end = mb_start + micro_bs
                mb_prompts   = repeated_prompts[mb_start:mb_end]    # type: ignore
                mb_responses = rollout_responses[mb_start:mb_end]
                mb_adv       = advantages[mb_start:mb_end].unsqueeze(1).to(policy.device)
                mb_raw       = raw_rewards[mb_start:mb_end].unsqueeze(1).to(policy.device)

                tok = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                input_ids = tok["input_ids"].to(policy.device)
                labels    = tok["labels"].to(policy.device)
                mask      = tok["response_mask"].to(policy.device)

                lp_dict = get_response_log_probs(policy, input_ids, labels)
                policy_lp = lp_dict["log_probs"]

                old_lp = None
                if old_log_probs_list is not None:
                    # Pad/slice old_lp to match current tokenization length
                    old_lp = _align_log_probs(
                        old_log_probs_list[mb_start:mb_end], policy_lp.shape
                    ).to(policy.device)

                grpo_microbatch_train_step(
                    policy_log_probs=policy_lp,
                    response_mask=mask,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    loss_type=cfg.loss_type,
                    raw_rewards=mb_raw if cfg.loss_type == "no_baseline" else None,
                    advantages=mb_adv  if cfg.loss_type != "no_baseline" else None,
                    old_log_probs=old_lp,
                    cliprange=cfg.cliprange,
                )

            if (epoch + 1) % cfg.epochs_per_rollout_batch == 0 or \
               epoch == cfg.epochs_per_rollout_batch - 1:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # ── Step 6: Sync weights to vLLM ────────────────────────────────────
        load_policy_into_vllm_instance(policy, vllm_engine)

        # ── Step 7: Periodic evaluation ─────────────────────────────────────
        if step % cfg.eval_every == 0:
            policy.eval()
            metrics = evaluate_vllm(vllm_engine, r1_zero_reward_fn,
                                    val_prompts[:1024], val_gts[:1024],
                                    EVAL_PARAMS)
            wandb.log({"eval/answer_acc": metrics["answer"], "eval_step": step // cfg.eval_every})
            policy.train()
```

### 26.15.2 The `_align_log_probs` Helper

Old log-probs were computed before the training loop tokenized the microbatch. Due to padding, the sequence lengths may not match:

```python
def _align_log_probs(
    old_lp_list: list[torch.Tensor],
    target_shape: torch.Size,
) -> torch.Tensor:
    """
    Pad or trim a list of per-token log-prob tensors to match target_shape (B, T).
    """
    B, T = target_shape
    result = torch.zeros(B, T)
    for i, lp in enumerate(old_lp_list[:B]):
        L = min(lp.shape[-1], T)
        result[i, :L] = lp[0, :L] if lp.dim() == 2 else lp[:L]
    return result
```

---

## 26.16 GRPO Experiments — Deliverable Checklist

### Learning Rate Sweep (Problem `grpo_learning_rate`)

- [ ] Sweep over at least: 1e-6, 5e-6, 1e-5, 5e-5
- [ ] Run each for 200 steps (or stop early if divergence detected)
- [ ] Report val answer reward curves for each LR
- [ ] Report final val accuracy for the best LR
- [ ] Identify divergence threshold (usually 5e-5 or 1e-4)
- [ ] Note: "divergence" means grad_norm consistently >20, NOT just high loss

### Baseline Ablation (Problem `grpo_baselines`)

- [ ] Compare `no_baseline` vs `reinforce_with_baseline`
- [ ] Use best LR from sweep
- [ ] Plot val reward curves and grad_norm curves side by side
- [ ] Comment: does `reinforce_with_baseline` reduce grad_norm spikes?

### Length Normalization (Problem `grpo_length_normalization`)

- [ ] Compare `masked_mean` vs `masked_normalize(constant=max_gen_len=1024)`
- [ ] Key metric to report: gradient norm variance across steps
- [ ] `masked_normalize` should show lower norm variance

### Std Normalization (Problem `grpo_group_standard_deviation`)

- [ ] Compare `normalize_by_std=True` vs `normalize_by_std=False`
- [ ] Watch for: are all-incorrect question groups getting amplified?
- [ ] Dr. GRPO (`normalize_by_std=False`) tends to produce lower advantage magnitudes for hard questions where all rollouts fail

### Off-Policy Sweep (Problem `grpo_off_policy_sweep`)

- [ ] Implement: set `epochs_per_rollout_batch > 1` and `loss_type = "grpo_clip"`
- [ ] Compare vs on-policy baseline with same gradient steps
- [ ] Plot both vs. gradient steps and vs. wall-clock time
- [ ] Key metric: does off-policy improve sample efficiency (performance per vLLM inference second)?

### Clip Ablation (Problem `grpo_off_policy_clip_ablation`)

- [ ] Implement `GRPO-No-Clip`: `-ratio * advantage` (no `torch.min`)
- [ ] Use same off-policy hyperparameters as best off-policy run
- [ ] Expected: entropy collapses faster without clip; grad norms spike

---

## 26.17 Optional Supplement — Instruction Tuning and DPO

### 26.17.1 Packed SFT Dataset

The key design choice: concatenate all documents end-to-end (separated by EOS token) then split into fixed-length sequences. This maximizes GPU utilization by eliminating padding.

```python
import torch
from torch.utils.data import Dataset
import json, gzip
from pathlib import Path

class PackedSFTDataset(Dataset):
    ALPACA_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{prompt}\n\n### Response:\n{response}"
    )

    def __init__(self, tokenizer, dataset_path: str,
                 seq_length: int, shuffle: bool = True):
        self.seq_length = seq_length
        opener = gzip.open if str(dataset_path).endswith(".gz") else open
        with opener(dataset_path, "rt") as f:
            examples = [json.loads(line) for line in f]
        if shuffle:
            import random; random.shuffle(examples)

        eos = tokenizer.eos_token_id
        all_ids = []
        for ex in examples:
            text = self.ALPACA_TEMPLATE.format(
                prompt=ex["prompt"], response=ex["response"]
            )
            ids = tokenizer.encode(text)
            all_ids.extend(ids + [eos])   # EOS between documents

        # Split into seq_length chunks (drop last incomplete chunk)
        n_chunks = len(all_ids) // seq_length
        self.chunks = [
            all_ids[i * seq_length : (i + 1) * seq_length]
            for i in range(n_chunks)
        ]

    def __len__(self): return len(self.chunks)

    def __getitem__(self, i):
        ids = torch.tensor(self.chunks[i], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}
```

### 26.17.2 DPO Loss (Self-Contained)

```python
import torch, torch.nn.functional as F, contextlib

def sequence_log_prob(model, tokens: torch.Tensor) -> torch.Tensor:
    """Sum of log-probs over all tokens in the sequence."""
    with torch.no_grad() if not model.training else contextlib.nullcontext():
        logits = model(tokens).logits           # (1, T, V)
    log_p = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T-1, V)
    labels = tokens[:, 1:]                     # (1, T-1)
    return log_p.gather(2, labels.unsqueeze(-1)).squeeze(-1).sum()

def dpo_loss(
    policy_model,
    ref_model,
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    beta: float = 0.1,
) -> torch.Tensor:
    ALPACA = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{p}\n\n### Response:\n{r}"
        "{eos}"
    )
    eos = tokenizer.eos_token

    def tokenize(response: str) -> torch.Tensor:
        text = ALPACA.format(p=prompt, r=response, eos=eos)
        return tokenizer(text, return_tensors="pt").input_ids.to(policy_model.device)

    tokens_w = tokenize(chosen)
    tokens_l = tokenize(rejected)

    # Policy log-probs (with gradients)
    pi_log_w = sequence_log_prob(policy_model, tokens_w)
    pi_log_l = sequence_log_prob(policy_model, tokens_l)

    # Reference log-probs (no gradients; may be on different device)
    with torch.no_grad():
        ref_log_w = sequence_log_prob(ref_model, tokens_w.to(ref_model.device))
        ref_log_l = sequence_log_prob(ref_model, tokens_l.to(ref_model.device))

    ref_log_w = ref_log_w.to(policy_model.device)
    ref_log_l = ref_log_l.to(policy_model.device)

    log_ratio_diff = (pi_log_w - ref_log_w) - (pi_log_l - ref_log_l)
    return -F.logsigmoid(beta * log_ratio_diff)

# ── Sanity check: random policy + ref → loss ≈ log(2) ≈ 0.693 ──────────────
if __name__ == "__main__":
    import math
    # When π_θ = π_ref exactly, log_ratio_diff = 0, loss = -log(σ(0)) = log(2)
    loss_when_equal = -math.log(1 / (1 + math.exp(0)))
    print(f"DPO loss at initialization (π_θ = π_ref): ≈ {loss_when_equal:.4f}")
    print(f"(expected: log(2) = {math.log(2):.4f})")
```

### 26.17.3 DPO Training Gradient Accumulation

```python
optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=1e-6)
grad_accum_steps = 32

optimizer.zero_grad()
for idx, (prompt, chosen, rejected) in enumerate(train_loader):
    loss = dpo_loss(policy_model, ref_model, tokenizer,
                    prompt, chosen, rejected, beta=0.1)
    (loss / grad_accum_steps).backward()

    if (idx + 1) % grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

**Implicit reward classification accuracy** (validation metric for DPO):

```python
def dpo_val_accuracy(policy_model, ref_model, tokenizer, val_data, beta=0.1):
    correct = 0
    for prompt, chosen, rejected in val_data:
        with torch.no_grad():
            pi_w = sequence_log_prob(policy_model, tokenize(prompt + chosen))
            pi_l = sequence_log_prob(policy_model, tokenize(prompt + rejected))
            rf_w = sequence_log_prob(ref_model,    tokenize(prompt + chosen))
            rf_l = sequence_log_prob(ref_model,    tokenize(prompt + rejected))
        # Model prefers chosen if implicit reward(chosen) > implicit reward(rejected)
        if (pi_w - rf_w) > (pi_l - rf_l):
            correct += 1
    return correct / len(val_data)
```

---

## 26.18 What vLLM Generates vs. What the Training Loop Updates

This table resolves every architectural confusion in the assignment:

| Operation | Where it runs | Uses gradients? | Modifies weights? |
|-----------|--------------|----------------|------------------|
| `llm.generate(prompts, params)` | vLLM, GPU 1 | No | No |
| `model(input_ids).logits` | HuggingFace, GPU 0 | Yes (during training) | No |
| `loss.backward()` | HuggingFace, GPU 0 | Yes (accumulates) | No |
| `optimizer.step()` | HuggingFace, GPU 0 | No (consumes) | **Yes** |
| `load_policy_into_vllm_instance(...)` | CPU (copy) | No | **Yes (vLLM copy)** |

A common confusion: "does vLLM see updated weights during rollout generation?" The answer is: **only after `load_policy_into_vllm_instance` is called**. Within a GRPO step, the rollout is generated with the old weights. The training loop updates the HuggingFace copy. After the training step, `load_policy_into_vllm_instance` copies the new weights into vLLM. At the *next* step, rollouts are generated with the new weights.

---

## 26.19 Common Bugs and Fixes

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `format_reward` always 0 | Missing `include_stop_str_in_output=True` | Set it in SamplingParams |
| Loss is NaN from step 1 | All-zero response_mask (empty rollout) | Set `min_tokens=4` in SamplingParams |
| Accuracy flat after 50 steps | `load_policy_into_vllm_instance` not called | Call it after every optimizer.step() |
| `RuntimeError: device mismatch` | Tensor from GPU 0 passed to vLLM (GPU 1) | Move to correct device before calling |
| Gradient accumulation not working | `optimizer.zero_grad()` inside microbatch loop | Move zero_grad() outside |
| Response_mask off by one | `mask[i, prompt_len:]` instead of `[i, prompt_len-1:]` | Fix start index to `prompt_len - 1` |
| DPO loss stuck at log(2) | Reference model on wrong device, `.to()` not called | Move ref outputs to policy device |
| GRPO-Clip not clipping | `old_log_probs.detach()` missing | Add `.detach()` to old_log_probs |
| Entropy → 0 after 10 steps | Learning rate too high; mode collapse | Reduce LR; check grad_norm |
| vLLM OOM during rollout | `gpu_memory_utilization` too high | Reduce to 0.7 or lower |

---

## 26.20 Companion Code

`code/chapter_25/alignment_demo.py` is a self-contained module implementing all functions from this chapter. It runs without a GPU: a `MockRewardFn` and `MockModel` simulate the reward and model respectively, so every function can be exercised and unit-tested on a laptop.

Running `python alignment_demo.py` executes 50 steps of simulated GRPO training, prints a reward curve, and verifies that advantages, log-probs, and losses are numerically correct.

`code/chapter_25/alignment_demo.cpp` implements the mathematical kernels — group normalization, GRPO-Clip, DPO loss — as standalone functions with worked examples in `main()`. Compile with `g++ -std=c++17 -O2 alignment_demo.cpp -o alignment_demo`.

---

## References

- CS336 Staff (2025). *Assignment 5: Alignment and Reasoning RL.* Stanford CS336 Spring 2025.
- CS336 Staff (2025). *Assignment 5 Supplement: Instruction Tuning and RLHF.* Stanford CS336 Spring 2025.
- DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
- Shao et al. (2024). *DeepSeekMath.* arXiv:2402.03300.
- Liu et al. (2025). *Understanding R1-Zero-Like Training.* arXiv:2503.20783.
- Rafailov et al. (2023). *Direct Preference Optimization.* arXiv:2305.18290.
- Zelikman et al. (2022). *STaR: Bootstrapping Reasoning with Reasoning.* arXiv:2203.14465.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.


---

## Chapter Summary

- **Alignment stack overview**: RLHF comprises three phases — supervised fine-tuning (SFT), reward modeling (RM), and policy optimization (PPO/DPO) — each with distinct serving requirements.
- **SFT serving**: standard fine-tuning inference; the SFT model is the base for reward and policy models; serving it is identical to base model inference.
- **Reward model architecture**: typically a base LLM with a scalar linear head; a reward query requires a full forward pass with no generation, making it prefill-only and compute-bound.
- **DPO eliminates the RM**: Direct Preference optimization folds reward into the policy update using log-ratio of policy vs reference model; this requires serving both policy and reference simultaneously.
- **Inference compute budget for alignment evaluation**: evaluating a policy checkpoint across standard benchmarks (MMLU, MT-Bench, HumanEval) requires O(benchmark_size × model_forward_passes) GPU-hours.
- **Constitutional AI (CAI)**: uses the model itself as a critic in a chain of self-critique + revision; serving must handle long multi-turn prompts and sequential critique/revision generations.
- **Preference data quality**: alignment quality is bounded by human annotation consistency; a Krippendorff's α < 0.7 on preference labels is a warning sign for training instability.

---

## Self-Check Questions

1. A reward model scores a completion by running a forward pass over the concatenated prompt + completion. The prompt is 256 tokens and the completion is 512 tokens. Is this prefill or decode? How many FLOPs does it require relative to a 512-token generation? *(Section 26.2)*

2. DPO requires the reference model (frozen SFT model) to compute log-probabilities on each training batch. You serve both policy and reference on the same GPU. If the policy is updated every 100 steps, what is the memory pressure from keeping both models hot? *(Section 26.3)*

3. Constitutional AI runs N critique-revision cycles. If N=3 and each cycle generates 200 critique tokens then 300 revision tokens, compute the total generation tokens per training example vs standard RLHF with one 500-token completion. *(Section 26.4)*

4. MT-Bench uses an LLM judge (GPT-4 or equivalent) to score model outputs. Name two failure modes of LLM judges that could give misleading alignment evaluation results. *(Section 26.5)*

5. Krippendorff's α = 0.55 on your preference annotation dataset. What does this mean for reward model training stability, and what would you do before continuing to the PPO stage? *(Section 26.6)*
