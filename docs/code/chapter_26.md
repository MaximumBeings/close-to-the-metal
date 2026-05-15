# Chapter 26: CS336 Alignment Field Guide — Companion Code

## Python — `alignment_demo.py`

```python
#!/usr/bin/env python3
"""
Chapter 26 — Companion Code: CS336 A5 Alignment Field Guide
============================================================
Self-contained implementations of every key function from the assignment:
  - tokenize_prompt_and_output
  - compute_entropy
  - get_response_log_probs
  - masked_normalize / masked_mean
  - sft_microbatch_train_step
  - compute_group_normalized_rewards
  - compute_naive_policy_gradient_loss
  - compute_grpo_clip_loss
  - compute_policy_gradient_loss
  - grpo_microbatch_train_step
  - dpo_loss (per-instance)
  - simulate_grpo_training   (50-step toy demo)
  - simulate_dpo_training    (convergence demo)

No GPU, no HuggingFace, no vLLM required.
MockModel and MockRewardFn simulate the real components.

Run: python alignment_demo.py
"""

import math
import random
import contextlib
import torch
import torch.nn.functional as F
from typing import Callable, Literal
from dataclasses import dataclass

random.seed(42)
torch.manual_seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# Mock infrastructure (replaces vLLM / HuggingFace in demos)
# ──────────────────────────────────────────────────────────────────────────────

class MockTokenizer:
    """Minimal tokenizer for unit tests."""
    pad_token_id = 0
    eos_token_id = 1
    eos_token    = "<|eos|>"
    vocab_size   = 32

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Simple deterministic mapping: ord(char) % 30 + 2
        ids = [(ord(c) % 30) + 2 for c in text]
        if add_special_tokens:
            ids = [2] + ids    # BOS
        return ids

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = torch.tensor([self.encode(text)], dtype=torch.long)
        return type("Batch", (), {"input_ids": ids})()


class MockModel(torch.nn.Module):
    """Tiny 2-layer model for unit tests."""

    def __init__(self, vocab_size: int = 32, hidden: int = 16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.proj  = torch.nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        h      = self.embed(input_ids)
        logits = self.proj(h)
        return type("Output", (), {"logits": logits})()


def mock_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    """
    Toy reward function: award 1.0 if response contains the ground truth string,
    0.5 if it has the right format tags, else 0.0.
    """
    has_answer_tags = "<answer>" in response and "</answer>" in response
    has_think_tags  = "<think>"  in response and "</think>"  in response
    format_reward   = 1.0 if (has_answer_tags and has_think_tags) else 0.0
    answer_correct  = ground_truth.strip() in response
    answer_reward   = 1.0 if answer_correct else 0.0
    total           = format_reward * 0.5 + answer_reward * 0.5
    return {"reward": total, "format_reward": format_reward, "answer_reward": answer_reward}


# ──────────────────────────────────────────────────────────────────────────────
# §25.4  tokenize_prompt_and_output
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, torch.Tensor]:
    """
    Tokenize prompts and outputs separately, concatenate, build response_mask.

    Returns:
        input_ids:     (B, L-1)
        labels:        (B, L-1)
        response_mask: (B, L-1)  — 1 on response positions in labels
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    prompt_ids_list = [tokenizer.encode(p)                          for p in prompt_strs]
    output_ids_list = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    full_ids_list   = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    max_len         = max(len(s) for s in full_ids_list)

    padded  = [s + [pad_id] * (max_len - len(s)) for s in full_ids_list]
    ids_t   = torch.tensor(padded, dtype=torch.long)    # (B, max_len)

    input_ids = ids_t[:, :-1]     # drop last
    labels    = ids_t[:, 1:]      # drop first (shift)

    response_mask = torch.zeros_like(labels)
    for i, (p_ids, o_ids) in enumerate(zip(prompt_ids_list, output_ids_list)):
        # In labels, first response prediction is at index prompt_len - 1
        start = len(p_ids) - 1
        end   = start + len(o_ids)
        if end > labels.shape[1]:
            end = labels.shape[1]
        response_mask[i, start:end] = 1

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def test_tokenize_prompt_and_output():
    tok    = MockTokenizer()
    result = tokenize_prompt_and_output(
        ["Hello "],
        ["world!"],
        tok,
    )
    mask_sum = result["response_mask"][0].sum().item()
    assert mask_sum > 0, "response_mask is all zeros"
    assert result["input_ids"].shape[1] == result["labels"].shape[1]
    assert result["response_mask"].shape == result["labels"].shape
    print(f"  tokenize_prompt_and_output: OK  (response tokens masked = {int(mask_sum)})")


# ──────────────────────────────────────────────────────────────────────────────
# §25.5  compute_entropy
# ──────────────────────────────────────────────────────────────────────────────

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """(B, T, V) → (B, T) per-token entropy H(p) = -Σ p log p."""
    log_p = F.log_softmax(logits, dim=-1)
    return -(log_p.exp() * log_p).sum(dim=-1)


def test_compute_entropy():
    # Uniform over 4 tokens → entropy = log(4)
    logits   = torch.zeros(1, 3, 4)
    H        = compute_entropy(logits)
    expected = math.log(4)
    assert H.shape == (1, 3)
    assert abs(H[0, 0].item() - expected) < 1e-5, f"Got {H[0,0].item()}, expected {expected}"
    print(f"  compute_entropy: OK  (uniform-4 entropy = {H[0,0].item():.4f}, expected {expected:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# §25.5  get_response_log_probs
# ──────────────────────────────────────────────────────────────────────────────

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Returns per-token log p(label | context) for all positions."""
    logits    = model(input_ids).logits                          # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)                    # (B, T, V)
    gathered  = log_probs.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)                                                 # (B, T)
    result = {"log_probs": gathered}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def test_get_response_log_probs():
    model = MockModel(vocab_size=32, hidden=8)
    tok   = tokenize_prompt_and_output(["abc"], ["de"], MockTokenizer())
    out   = get_response_log_probs(
        model, tok["input_ids"], tok["labels"], return_token_entropy=True
    )
    B, T  = tok["input_ids"].shape
    assert out["log_probs"].shape == (B, T)
    assert out["token_entropy"].shape == (B, T)
    assert (out["log_probs"] <= 0).all(), "log-probs must be ≤ 0"
    print(f"  get_response_log_probs: OK  (mean log-prob = {out['log_probs'].mean().item():.3f})")


# ──────────────────────────────────────────────────────────────────────────────
# §25.5  masked_normalize / masked_mean
# ──────────────────────────────────────────────────────────────────────────────

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    masked = tensor * mask
    total  = masked.sum() if dim is None else masked.sum(dim=dim)
    return total / normalize_constant


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum().clamp(min=1)
    return masked.sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)


def test_masked_operations():
    t    = torch.tensor([[1., 2., 3., 4.]])
    m    = torch.tensor([[1., 1., 0., 1.]])
    norm = masked_normalize(t, m, normalize_constant=3.0, dim=1)
    mean = masked_mean(t, m, dim=1)
    # normalize: (1+2+4)/3 = 7/3 ≈ 2.333
    # mean:      (1+2+4)/3 = 7/3 ≈ 2.333 (same here)
    assert abs(norm.item() - 7/3) < 1e-5, f"normalize: {norm.item()}"
    assert abs(mean.item() - 7/3) < 1e-5, f"mean: {mean.item()}"

    # Different lengths: masked_mean should give equal weight per sequence
    t2   = torch.tensor([[2., 2., 2., 2., 0., 0.],
                          [2., 2., 2., 2., 2., 2.]])
    m2   = torch.tensor([[1., 1., 1., 1., 0., 0.],
                          [1., 1., 1., 1., 1., 1.]])
    res  = masked_mean(t2, m2, dim=1)
    assert abs(res[0].item() - 2.0) < 1e-5
    assert abs(res[1].item() - 2.0) < 1e-5
    print("  masked_normalize / masked_mean: OK")


# ──────────────────────────────────────────────────────────────────────────────
# §25.6  sft_microbatch_train_step
# ──────────────────────────────────────────────────────────────────────────────

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    B   = policy_log_probs.shape[0]
    nll = masked_normalize(
        -policy_log_probs, response_mask,
        normalize_constant=normalize_constant * B,
    )
    loss = nll / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), {"sft_nll": nll.item()}


def test_sft_microbatch_train_step():
    model    = MockModel(vocab_size=32, hidden=8)
    tok      = tokenize_prompt_and_output(["test "], ["answer"], MockTokenizer())
    lp_dict  = get_response_log_probs(model, tok["input_ids"], tok["labels"])
    lp       = lp_dict["log_probs"].requires_grad_(True)

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loss, meta = sft_microbatch_train_step(lp, tok["response_mask"], gradient_accumulation_steps=4)
    # Gradient should have been computed
    assert loss.item() > 0, "loss should be positive (NLL)"
    print(f"  sft_microbatch_train_step: OK  (loss = {loss.item():.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# §25.10  compute_group_normalized_rewards
# ──────────────────────────────────────────────────────────────────────────────

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


def test_compute_group_normalized_rewards():
    # 2 questions, G=3: question 0 has mixed rewards, question 1 all wrong
    responses = [
        "<think>step</think><answer>4</answer>",   # q0, correct
        "<think>step</think><answer>3</answer>",   # q0, wrong
        "<think>step</think><answer>4</answer>",   # q0, correct
        "no tags",                                  # q1, wrong
        "no tags",                                  # q1, wrong
        "no tags",                                  # q1, wrong
    ]
    gts = ["4", "4", "4", "99", "99", "99"]

    adv, raw, meta = compute_group_normalized_rewards(
        mock_reward_fn, responses, gts,
        group_size=3, normalize_by_std=True,
    )
    assert adv.shape == (6,)
    assert raw.shape == (6,)
    # All-wrong group (q1) should have zero advantages
    assert adv[3:].abs().max().item() < 1e-4, "All-wrong group should have ~0 advantage"
    print(f"  compute_group_normalized_rewards: OK  "
          f"(q0_adv={adv[:3].tolist()}, q1_adv={adv[3:].tolist()})")


# ──────────────────────────────────────────────────────────────────────────────
# §25.11  compute_naive_policy_gradient_loss
# ──────────────────────────────────────────────────────────────────────────────

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,   # (B, 1)
    policy_log_probs: torch.Tensor,            # (B, T)
) -> torch.Tensor:                             # (B, T)
    return -raw_rewards_or_advantages * policy_log_probs   # broadcast (B,1)→(B,T)


def test_naive_pg_loss():
    adv      = torch.tensor([[2.0]])
    log_prob = torch.tensor([[-0.5, -1.0]])
    loss     = compute_naive_policy_gradient_loss(adv, log_prob)
    assert loss.shape == (1, 2)
    assert abs(loss[0, 0].item() - 1.0) < 1e-6
    assert abs(loss[0, 1].item() - 2.0) < 1e-6
    print("  compute_naive_policy_gradient_loss: OK")


# ──────────────────────────────────────────────────────────────────────────────
# §25.12  compute_grpo_clip_loss
# ──────────────────────────────────────────────────────────────────────────────

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
    return loss, {"clip_fraction": is_clipped.float().mean().item(),
                   "mean_ratio":   ratio.mean().item()}


def test_compute_grpo_clip_loss():
    # Single token, advantage=+1, ε=0.2
    # ratio=1.5 → clipped at 1+0.2=1.2 → loss = -min(1.5, 1.2) = -1.2
    adv      = torch.tensor([[1.0]])
    pi_lp    = torch.tensor([[math.log(1.5)]])  # policy is 1.5x more likely
    old_lp   = torch.tensor([[0.0]])             # old log-prob = 0 → prob = 1
    loss, m  = compute_grpo_clip_loss(adv, pi_lp, old_lp, cliprange=0.2)
    assert abs(loss[0, 0].item() - (-1.2)) < 1e-5, f"Expected -1.2, got {loss[0,0].item()}"
    assert m["clip_fraction"] == 1.0, "Should be clipped"
    print(f"  compute_grpo_clip_loss: OK  (clipped loss = {loss[0,0].item():.4f}, "
          f"clip_fraction = {m['clip_fraction']:.1f})")


# ──────────────────────────────────────────────────────────────────────────────
# §25.14  compute_policy_gradient_loss (dispatcher)
# ──────────────────────────────────────────────────────────────────────────────

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards:   torch.Tensor | None = None,
    advantages:    torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange:     float | None = None,
) -> tuple[torch.Tensor, dict]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "no_baseline requires raw_rewards"
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "reinforce_with_baseline requires advantages"
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        assert advantages is not None, "grpo_clip requires advantages"
        assert old_log_probs is not None, "grpo_clip requires old_log_probs"
        assert cliprange is not None, "grpo_clip requires cliprange"
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")


# ──────────────────────────────────────────────────────────────────────────────
# §25.14  grpo_microbatch_train_step
# ──────────────────────────────────────────────────────────────────────────────

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards:   torch.Tensor | None = None,
    advantages:    torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange:     float | None = None,
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


def test_grpo_microbatch_train_step():
    model = MockModel(vocab_size=32, hidden=8)
    tok   = tokenize_prompt_and_output(["q "], ["answer"], MockTokenizer())
    B, T  = tok["input_ids"].shape

    lp_dict = get_response_log_probs(model, tok["input_ids"], tok["labels"])
    lp      = lp_dict["log_probs"]

    adv     = torch.tensor([[0.5]])
    old_lp  = lp.detach().clone()

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # Test reinforce_with_baseline
    loss, _ = grpo_microbatch_train_step(
        lp, tok["response_mask"],
        gradient_accumulation_steps=4,
        loss_type="reinforce_with_baseline",
        advantages=adv,
    )
    assert isinstance(loss.item(), float)
    print(f"  grpo_microbatch_train_step (reinforce_with_baseline): OK  "
          f"(loss = {loss.item():.5f})")


# ──────────────────────────────────────────────────────────────────────────────
# §25.17  DPO loss
# ──────────────────────────────────────────────────────────────────────────────

def sequence_log_prob_mock(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    no_grad: bool = False,
) -> torch.Tensor:
    """Sum of log-probs over all sequence positions."""
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        logits  = model(tokens).logits            # (1, T, V)
        log_p   = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T-1, V)
        labels  = tokens[:, 1:]                   # (1, T-1)
        per_tok = log_p.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return per_tok.sum()


def dpo_loss_mock(
    policy_model: torch.nn.Module,
    ref_model:    torch.nn.Module,
    tokenizer:    MockTokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    beta: float = 0.1,
) -> torch.Tensor:
    """Per-instance DPO loss using mock models."""
    def tokenize(text: str) -> torch.Tensor:
        return torch.tensor([tokenizer.encode(text)], dtype=torch.long)

    tokens_w = tokenize(prompt + chosen)
    tokens_l = tokenize(prompt + rejected)

    pi_log_w = sequence_log_prob_mock(policy_model, tokens_w, no_grad=False)
    pi_log_l = sequence_log_prob_mock(policy_model, tokens_l, no_grad=False)

    rf_log_w = sequence_log_prob_mock(ref_model, tokens_w, no_grad=True)
    rf_log_l = sequence_log_prob_mock(ref_model, tokens_l, no_grad=True)

    log_ratio_diff = (pi_log_w - rf_log_w) - (pi_log_l - rf_log_l)
    return -F.logsigmoid(beta * log_ratio_diff)


def test_dpo_loss():
    # When π_θ = π_ref (same model, no training), loss = -log(σ(0)) = log(2)
    model = MockModel(vocab_size=32, hidden=8)
    ref   = MockModel(vocab_size=32, hidden=8)
    # Give both models the same weights
    ref.load_state_dict(model.state_dict())
    ref.eval()

    tok  = MockTokenizer()
    loss = dpo_loss_mock(model, ref, tok, "question: ", "correct answer", "wrong answer")
    expected = math.log(2)
    # With same weights, loss ≈ log(2) ≈ 0.693 (may not be exact due to different sequence lengths)
    print(f"  dpo_loss (π_θ=π_ref): loss = {loss.item():.4f}, "
          f"expected ≈ {expected:.4f} (log 2)")
    print("  dpo_loss: OK")


# ──────────────────────────────────────────────────────────────────────────────
# §25.15  simulate_grpo_training (toy 50-step run)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    n_steps:                  int   = 50
    group_size:               int   = 4
    rollout_batch_size:       int   = 8        # 2 questions × 4 rollouts
    lr:                       float = 1e-3
    gradient_accumulation:    int   = 2
    advantage_eps:            float = 1e-6
    normalize_by_std:         bool  = True
    loss_type:                str   = "reinforce_with_baseline"
    cliprange:                float = 0.2
    log_every:                int   = 10


def simulate_grpo_training(cfg: GRPOConfig = GRPOConfig()):
    """
    Toy GRPO training loop.
    'Questions' are integers 1–20; correct answer = str(q).
    Model generates a string; reward_fn checks if string contains the number.
    """
    print("\n  Simulating GRPO training (toy math task, 50 steps)...")

    model     = MockModel(vocab_size=32, hidden=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    tokenizer = MockTokenizer()

    train_qs  = list(range(1, 21))   # questions 1–20
    reward_history: list[float] = []

    def generate_rollout(prompt: str) -> str:
        """Fake vLLM: model just deterministically picks top token repeatedly."""
        with torch.no_grad():
            toks = tokenizer.encode(prompt)
            for _ in range(10):
                inp    = torch.tensor([toks], dtype=torch.long)
                logits = model(inp).logits[0, -1, :]
                next_t = int(logits.argmax().item())
                toks.append(next_t)
        chars = [chr(t + 30) for t in toks[len(tokenizer.encode(prompt)):]]
        return "".join(chars)

    for step in range(cfg.n_steps):
        # ── Rollout ──────────────────────────────────────────────────────
        n_q      = cfg.rollout_batch_size // cfg.group_size
        questions = random.choices(train_qs, k=n_q)
        prompts   = [f"Q{q} " for q in questions]
        gts       = [str(q) for q in questions]

        rollout_responses = [generate_rollout(p) for p in prompts for _ in range(cfg.group_size)]
        repeated_gts      = [gt for gt in gts for _ in range(cfg.group_size)]

        advantages, raw_rewards, meta = compute_group_normalized_rewards(
            mock_reward_fn, rollout_responses, repeated_gts,
            group_size=cfg.group_size,
            advantage_eps=cfg.advantage_eps,
            normalize_by_std=cfg.normalize_by_std,
        )

        repeated_prompts_flat = [p for p in prompts for _ in range(cfg.group_size)]

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        micro_bs = cfg.rollout_batch_size // cfg.gradient_accumulation

        for mb_start in range(0, cfg.rollout_batch_size, micro_bs):
            mb_end   = mb_start + micro_bs
            mb_p     = repeated_prompts_flat[mb_start:mb_end]
            mb_r     = rollout_responses[mb_start:mb_end]
            mb_adv   = advantages[mb_start:mb_end].unsqueeze(1)

            tok      = tokenize_prompt_and_output(mb_p, mb_r, tokenizer)
            lp_dict  = get_response_log_probs(model, tok["input_ids"], tok["labels"])

            grpo_microbatch_train_step(
                lp_dict["log_probs"], tok["response_mask"],
                gradient_accumulation_steps=cfg.gradient_accumulation,
                loss_type=cfg.loss_type,
                advantages=mb_adv,
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        reward_history.append(meta["mean_reward"])

        if (step + 1) % cfg.log_every == 0:
            recent = reward_history[-cfg.log_every:]
            print(f"    Step {step+1:3d}: mean_reward = {sum(recent)/len(recent):.4f}  "
                  f"frac_correct = {meta['frac_correct']:.3f}")

    print(f"\n  Training complete.  Final 10-step mean reward: "
          f"{sum(reward_history[-10:])/10:.4f}")
    return reward_history


# ──────────────────────────────────────────────────────────────────────────────
# simulate_dpo_training (convergence demo)
# ──────────────────────────────────────────────────────────────────────────────

def simulate_dpo_training(n_steps: int = 100, beta: float = 0.1, lr: float = 1e-2):
    """
    Toy DPO training showing implicit reward classification accuracy converging.
    Chosen response = 'good'; rejected response = 'bad'.
    """
    print("\n  Simulating DPO training (binary preference demo)...")

    policy_model = MockModel(vocab_size=32, hidden=16)
    ref_model    = MockModel(vocab_size=32, hidden=16)
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    tokenizer = MockTokenizer()
    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=lr)

    prompts   = [f"question {i} " for i in range(20)]
    chosens   = ["<think>correct reasoning</think><answer>yes</answer>"] * 20
    rejecteds = ["wrong answer"] * 20

    for step in range(n_steps):
        prompt   = random.choice(prompts)
        chosen   = random.choice(chosens)
        rejected = random.choice(rejecteds)

        loss = dpo_loss_mock(policy_model, ref_model, tokenizer,
                              prompt, chosen, rejected, beta=beta)
        loss.backward()
        if (step + 1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % 25 == 0:
            # Classification accuracy: does model prefer chosen over rejected?
            correct = 0
            total   = 0
            with torch.no_grad():
                for p, c, r in zip(prompts[:10], chosens[:10], rejecteds[:10]):
                    toks_c = torch.tensor([tokenizer.encode(p + c)], dtype=torch.long)
                    toks_r = torch.tensor([tokenizer.encode(p + r)], dtype=torch.long)
                    lp_c   = sequence_log_prob_mock(policy_model, toks_c, no_grad=True)
                    lp_r   = sequence_log_prob_mock(policy_model, toks_r, no_grad=True)
                    ref_c  = sequence_log_prob_mock(ref_model,    toks_c, no_grad=True)
                    ref_r  = sequence_log_prob_mock(ref_model,    toks_r, no_grad=True)
                    if (lp_c - ref_c) > (lp_r - ref_r):
                        correct += 1
                    total += 1
            print(f"    Step {step+1:3d}: loss = {loss.item():.4f}  "
                  f"classification_acc = {correct}/{total}")


# ──────────────────────────────────────────────────────────────────────────────
# Unit test runner
# ──────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    print("\nRunning unit tests...")
    test_tokenize_prompt_and_output()
    test_compute_entropy()
    test_get_response_log_probs()
    test_masked_operations()
    test_sft_microbatch_train_step()
    test_compute_group_normalized_rewards()
    test_naive_pg_loss()
    test_compute_grpo_clip_loss()
    test_grpo_microbatch_train_step()
    test_dpo_loss()
    print("\nAll unit tests passed.\n")


if __name__ == "__main__":
    print("Chapter 26 — CS336 Alignment Field Guide Demo")
    print("=" * 60)
    run_all_tests()
    reward_history = simulate_grpo_training()
    simulate_dpo_training()
    print("\n" + "=" * 60)
    print("All demos complete.")
    print("=" * 60)

```

## C++ — `alignment_demo.cpp`

```cpp
/**
 * Chapter 26 — CS336 Alignment Field Guide
 * ==============================================================
 * Implements: group-normalized rewards, GRPO-Clip loss, DPO loss,
 * masked mean/normalize, policy gradient math, and a 50-step GRPO simulation.
 *
 * Build:  g++ -std=c++17 -O2 alignment_demo.cpp -o alignment_demo
 * Run:    ./alignment_demo
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <functional>

// ─────────────────────────────────────────────────────────────────────────────
// §25.5  Masked operations
// ─────────────────────────────────────────────────────────────────────────────

// Sum of tensor[i] where mask[i] == 1, divided by normalize_constant
double masked_normalize(
    const std::vector<double>& tensor,
    const std::vector<double>& mask,
    double normalize_constant
) {
    double total = 0.0;
    for (size_t i = 0; i < tensor.size(); ++i)
        total += tensor[i] * mask[i];
    return total / normalize_constant;
}

// Mean over positions where mask == 1
double masked_mean_vec(
    const std::vector<double>& tensor,
    const std::vector<double>& mask
) {
    double sum = 0.0, count = 0.0;
    for (size_t i = 0; i < tensor.size(); ++i) {
        sum   += tensor[i] * mask[i];
        count += mask[i];
    }
    return (count > 0) ? sum / count : 0.0;
}


// ─────────────────────────────────────────────────────────────────────────────
// §25.10  Group-Normalized Rewards (compute_group_normalized_rewards)
// ─────────────────────────────────────────────────────────────────────────────

struct GroupNormResult {
    std::vector<double> advantages;
    std::vector<double> raw_rewards;
    double mean_reward;
    double frac_correct;
};

GroupNormResult compute_group_normalized_rewards(
    const std::vector<double>& raw_rewards,
    int group_size,
    double advantage_eps   = 1e-6,
    bool normalize_by_std  = true
) {
    int n  = static_cast<int>(raw_rewards.size());
    assert(n % group_size == 0);
    int n_q = n / group_size;

    std::vector<double> advantages(n, 0.0);

    for (int q = 0; q < n_q; ++q) {
        int s = q * group_size;
        int e = s + group_size;

        double mean = 0.0;
        for (int i = s; i < e; ++i) mean += raw_rewards[i];
        mean /= group_size;

        double std_dev = 0.0;
        if (normalize_by_std) {
            for (int i = s; i < e; ++i) {
                double diff = raw_rewards[i] - mean;
                std_dev += diff * diff;
            }
            std_dev = std::sqrt(std_dev / group_size) + advantage_eps;
        }

        for (int i = s; i < e; ++i) {
            if (normalize_by_std)
                advantages[i] = (raw_rewards[i] - mean) / std_dev;
            else
                advantages[i] = raw_rewards[i] - mean;
        }
    }

    double sum_r = 0.0, n_correct = 0.0;
    for (double r : raw_rewards) {
        sum_r    += r;
        n_correct += (r > 0.5) ? 1.0 : 0.0;
    }

    return { advantages, raw_rewards, sum_r / n, n_correct / n };
}


// ─────────────────────────────────────────────────────────────────────────────
// §25.11  Naive Policy Gradient Loss
// ─────────────────────────────────────────────────────────────────────────────

// Per-token loss: -advantage * log_prob (broadcasted over sequence)
std::vector<double> naive_pg_loss(
    double advantage,
    const std::vector<double>& log_probs   // per-token log-probs for one sequence
) {
    std::vector<double> loss(log_probs.size());
    for (size_t t = 0; t < log_probs.size(); ++t)
        loss[t] = -advantage * log_probs[t];
    return loss;
}


// ─────────────────────────────────────────────────────────────────────────────
// §25.12  GRPO-Clip Loss
// ─────────────────────────────────────────────────────────────────────────────

struct ClipLossResult {
    std::vector<double> per_token_loss;
    double clip_fraction;
    double mean_ratio;
};

ClipLossResult compute_grpo_clip_loss(
    double advantage,
    const std::vector<double>& policy_log_probs,
    const std::vector<double>& old_log_probs,
    double cliprange = 0.2
) {
    size_t T = policy_log_probs.size();
    assert(old_log_probs.size() == T);

    std::vector<double> loss(T);
    double sum_ratio = 0.0, clipped_count = 0.0;

    for (size_t t = 0; t < T; ++t) {
        double log_ratio = policy_log_probs[t] - old_log_probs[t];
        double ratio     = std::exp(log_ratio);
        sum_ratio       += ratio;

        double unclipped = ratio * advantage;
        double clipped   = std::clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantage;
        loss[t]          = -std::min(unclipped, clipped);

        bool is_clipped = (ratio < 1.0 - cliprange) || (ratio > 1.0 + cliprange);
        if (is_clipped) clipped_count += 1.0;
    }

    return { loss, clipped_count / T, sum_ratio / T };
}


// ─────────────────────────────────────────────────────────────────────────────
// §25.17  DPO Loss (per instance, analytical)
// ─────────────────────────────────────────────────────────────────────────────

// DPO loss given log-ratio difference: -log σ(β × log_ratio_diff)
double dpo_loss_analytical(double log_ratio_diff, double beta = 0.1) {
    double bx  = beta * log_ratio_diff;
    double sig = 1.0 / (1.0 + std::exp(-bx));
    return -std::log(std::max(sig, 1e-12));
}

// Implicit reward: β * log(π_θ(y|x) / π_ref(y|x))
// Classification: prefer chosen if reward(chosen) > reward(rejected)
bool dpo_classifies_correctly(
    double policy_log_chosen,    double policy_log_rejected,
    double ref_log_chosen,       double ref_log_rejected,
    double beta = 0.1
) {
    double reward_chosen   = beta * (policy_log_chosen   - ref_log_chosen);
    double reward_rejected = beta * (policy_log_rejected - ref_log_rejected);
    return reward_chosen > reward_rejected;
}


// ─────────────────────────────────────────────────────────────────────────────
// §25.15  Simulated GRPO Training Loop (Toy Task)
// ─────────────────────────────────────────────────────────────────────────────

struct GRPOState {
    // "Model" is parameterized by a single scalar: its bias toward correct answers.
    // reward_prob = sigmoid(bias)
    double bias = 0.0;
    double lr   = 0.5;
};

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double dsigmoid(double x) { double s = sigmoid(x); return s * (1.0 - s); }

void simulate_grpo_training(int n_steps = 50, int group_size = 4) {
    GRPOState state;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    std::cout << "\n  Simulated GRPO training (toy Bernoulli policy, "
              << n_steps << " steps, G=" << group_size << ")\n\n";
    std::cout << "  " << std::setw(8)  << "Step"
              << std::setw(12) << "Bias"
              << std::setw(14) << "Reward_prob"
              << std::setw(16) << "Mean_reward\n";
    std::cout << "  " << std::string(50, '-') << "\n";

    for (int step = 0; step < n_steps; ++step) {
        // Generate G rollouts from current policy
        double reward_prob = sigmoid(state.bias);
        std::vector<double> rewards(group_size);
        for (int g = 0; g < group_size; ++g)
            rewards[g] = (unif(rng) < reward_prob) ? 1.0 : 0.0;

        // Compute group-normalized advantages
        auto result = compute_group_normalized_rewards(rewards, group_size, 1e-6, true);

        // Policy gradient update: grad ≈ Σ_i advantage_i * d(log π) / d(bias)
        // log π(reward=1 | bias) = -log(1 + exp(-bias)) = log(sigmoid(bias))
        // d/d(bias) = 1 - sigmoid(bias) for the "correct" action
        double grad = 0.0;
        for (int g = 0; g < group_size; ++g) {
            // action = 1 if reward=1 (correct), 0 if reward=0
            // score function: ∇ log π(a|s) = a - sigmoid(bias)
            double action = rewards[g];
            double score  = action - reward_prob;
            grad += result.advantages[g] * score;
        }
        grad /= group_size;

        // Gradient ascent on J (gradient descent on -J)
        state.bias += state.lr * grad;

        if ((step + 1) % 10 == 0) {
            double mean_r = result.mean_reward;
            std::cout << "  " << std::setw(8)  << (step + 1)
                      << std::setw(12) << std::fixed << std::setprecision(4) << state.bias
                      << std::setw(14) << std::setprecision(4) << sigmoid(state.bias)
                      << std::setw(16) << std::setprecision(4) << mean_r << "\n";
        }
    }
    std::cout << "\n  Final reward_prob = " << std::fixed << std::setprecision(4)
              << sigmoid(state.bias) << " (target: 1.0)\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// Demonstrations
// ─────────────────────────────────────────────────────────────────────────────

static void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "  " << title << "\n"
              << std::string(70, '=') << "\n";
}

void demo_group_normalization() {
    print_section("Group-Normalized Rewards (§25.10)");

    // 2 questions, G=3: q0 has mixed rewards, q1 all wrong
    std::vector<double> raw = {1.0, 0.0, 1.0,   // q0: correct, wrong, correct
                                0.0, 0.0, 0.0};  // q1: all wrong

    for (bool use_std : {true, false}) {
        auto r = compute_group_normalized_rewards(raw, 3, 1e-6, use_std);
        std::cout << "  normalize_by_std=" << (use_std ? "true " : "false")
                  << "  advantages: [";
        for (int i = 0; i < 6; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << r.advantages[i];
        }
        std::cout << "]\n";
    }
    std::cout << "\n  All-wrong group (q1) should produce advantages ≈ 0.\n";
    std::cout << "  Dr. GRPO variant (normalize_by_std=false) prevents amplification\n";
    std::cout << "  of questions where all rewards are identical.\n";
}

void demo_grpo_clip_walkthrough() {
    print_section("GRPO-Clip Loss Walkthrough (§25.12)");

    // Single token sequence, advantage = +1.0, ε = 0.2
    const double adv = 1.0, eps = 0.2;

    std::cout << "  advantage = " << adv << ",  ε = " << eps << "\n\n";
    std::cout << "  " << std::setw(8)  << "ratio"
              << std::setw(14) << "unclipped"
              << std::setw(14) << "clipped"
              << std::setw(14) << "per-tok-loss"
              << std::setw(14) << "clipped?\n";
    std::cout << "  " << std::string(64, '-') << "\n";

    for (double ratio : {0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0}) {
        double lp     = std::log(ratio);
        double old_lp = 0.0;
        std::vector<double> plp = {lp}, olp = {old_lp};
        auto result = compute_grpo_clip_loss(adv, plp, olp, eps);

        bool clipped = (ratio < 1 - eps) || (ratio > 1 + eps);
        std::cout << std::fixed << std::setprecision(2)
                  << "  " << std::setw(8)  << ratio
                  << std::setw(14) << ratio * adv
                  << std::setw(14) << std::clamp(ratio, 1-eps, 1+eps) * adv
                  << std::setw(14) << result.per_token_loss[0]
                  << std::setw(14) << (clipped ? "yes ← " : "no") << "\n";
    }
}

void demo_dpo_loss() {
    print_section("DPO Loss — Analytical Behavior (§25.17)");

    std::cout << "  β = 0.1\n\n";
    std::cout << "  " << std::setw(20) << "log_ratio_diff"
              << std::setw(16) << "β × lrd"
              << std::setw(16) << "σ(β × lrd)"
              << std::setw(14) << "loss\n";
    std::cout << "  " << std::string(66, '-') << "\n";

    for (double lrd : {-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0}) {
        double bx   = 0.1 * lrd;
        double sig  = 1.0 / (1.0 + std::exp(-bx));
        double loss = -std::log(std::max(sig, 1e-12));
        std::cout << std::fixed << std::setprecision(2)
                  << "  " << std::setw(20) << lrd
                  << std::setw(16) << bx
                  << std::setw(16) << std::setprecision(4) << sig
                  << std::setw(14) << loss << "\n";
    }

    std::cout << "\n  At initialization (π_θ = π_ref): log_ratio_diff = 0\n";
    std::cout << "  → loss = log(2) = " << std::log(2.0) << "\n";
    std::cout << "\n  Classification: model correctly ranks (chosen, rejected)\n";
    std::cout << "  if log_ratio_diff > 0 (model assigns higher implicit reward to chosen).\n";
}

void demo_masked_operations() {
    print_section("Masked Normalize / Masked Mean (§25.5)");

    // Short sequence (4 tokens) and long sequence (7 tokens), both value=2.0
    std::vector<double> values_short = {2, 2, 2, 2, 0, 0, 0};
    std::vector<double> mask_short   = {1, 1, 1, 1, 0, 0, 0};
    std::vector<double> values_long  = {2, 2, 2, 2, 2, 2, 2};
    std::vector<double> mask_long    = {1, 1, 1, 1, 1, 1, 1};

    double max_len = 7.0;

    double mm_short   = masked_mean_vec(values_short, mask_short);
    double mm_long    = masked_mean_vec(values_long,  mask_long);
    double mn_short   = masked_normalize(values_short, mask_short, max_len);
    double mn_long    = masked_normalize(values_long,  mask_long,  max_len);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Short (4 tokens): masked_mean = " << mm_short
              << "  masked_normalize(7) = " << mn_short << "\n";
    std::cout << "  Long  (7 tokens): masked_mean = " << mm_long
              << "  masked_normalize(7) = " << mn_long  << "\n\n";
    std::cout << "  masked_mean weights short responses more (2.0 vs 2.0 — equal here\n";
    std::cout << "  but per-token gradient = 2/4=0.50 short vs 2/7=0.29 long).\n";
    std::cout << "  masked_normalize gives equal per-token weight regardless of length.\n";
}

void demo_off_by_one_explanation() {
    print_section("Response Mask Off-By-One (§25.4 Visual)");

    // Sequence: prompt=[10,11,12], output=[20,21,22,23]
    std::vector<int> full = {10, 11, 12, 20, 21, 22, 23};
    int prompt_len = 3, output_len = 4;

    std::cout << "  Full sequence:  [";
    for (size_t i = 0; i < full.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << full[i];
    }
    std::cout << "] (prompt_len=" << prompt_len << ", output_len=" << output_len << ")\n\n";

    // After shift
    std::cout << "  After shift:\n";
    std::cout << "  input_ids: [";
    for (int i = 0; i < (int)full.size()-1; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << full[i];
    }
    std::cout << "]\n";

    std::cout << "  labels:    [";
    for (int i = 1; i < (int)full.size(); ++i) {
        if (i > 1) std::cout << ", ";
        std::cout << full[i];
    }
    std::cout << "]\n\n";

    // Response mask: 1 starting at prompt_len - 1
    int resp_start = prompt_len - 1;
    std::cout << "  response_mask: [";
    for (int i = 0; i < (int)full.size()-1; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << ((i >= resp_start && i < resp_start + output_len) ? 1 : 0);
    }
    std::cout << "]\n";
    std::cout << "                  ^--- starts at index " << resp_start
              << " = prompt_len-1 = " << (prompt_len-1) << "\n\n";
    std::cout << "  At index " << resp_start << ", input_ids=" << full[resp_start]
              << " (last prompt token), labels=" << full[resp_start+1]
              << " (first output token).\n";
    std::cout << "  This is the first position where we want to compute response NLL.\n";
    std::cout << "  Using mask[i, prompt_len:] would skip this position — a silent bug.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit Tests
// ─────────────────────────────────────────────────────────────────────────────

void run_tests() {
    print_section("Unit Tests");

    // Test 1: group normalization — all-wrong group → zero advantages
    {
        std::vector<double> raw = {1.0, 0.0, 1.0,   0.0, 0.0, 0.0};
        auto r = compute_group_normalized_rewards(raw, 3, 1e-6, true);
        double q1_adv_max = 0.0;
        for (int i = 3; i < 6; ++i)
            q1_adv_max = std::max(q1_adv_max, std::abs(r.advantages[i]));
        assert(q1_adv_max < 1e-3 && "All-wrong group should have ~0 advantage");
        std::cout << "  [PASS] All-wrong group → advantages ≈ 0\n";
    }

    // Test 2: naive PG loss — sign and magnitude
    {
        double adv = 2.0;
        std::vector<double> lp = {-0.5, -1.0};
        auto loss = naive_pg_loss(adv, lp);
        assert(std::abs(loss[0] - 1.0) < 1e-9 && "Expected -adv*lp[0] = 1.0");
        assert(std::abs(loss[1] - 2.0) < 1e-9 && "Expected -adv*lp[1] = 2.0");
        std::cout << "  [PASS] Naive PG loss: -2*(-0.5)=1.0, -2*(-1.0)=2.0\n";
    }

    // Test 3: GRPO-Clip — ratio=1.5, adv=1.0, eps=0.2 → loss=-1.2
    {
        std::vector<double> pi_lp = {std::log(1.5)};
        std::vector<double> old   = {0.0};
        auto r = compute_grpo_clip_loss(1.0, pi_lp, old, 0.2);
        assert(std::abs(r.per_token_loss[0] - (-1.2)) < 1e-5 && "Expected clipped to -1.2");
        assert(r.clip_fraction == 1.0 && "Should be clipped");
        std::cout << "  [PASS] GRPO-Clip: ratio=1.5 clipped to 1.2, loss=-1.2\n";
    }

    // Test 4: DPO loss at π_θ=π_ref → log(2)
    {
        double loss = dpo_loss_analytical(0.0, 0.1);
        assert(std::abs(loss - std::log(2.0)) < 1e-9);
        std::cout << "  [PASS] DPO loss at π_θ=π_ref = log(2) = "
                  << std::fixed << std::setprecision(6) << loss << "\n";
    }

    // Test 5: masked mean — ignores zero-mask positions
    {
        std::vector<double> t = {5.0, 2.0, 0.0, 0.0};
        std::vector<double> m = {1.0, 1.0, 0.0, 0.0};
        double mm = masked_mean_vec(t, m);
        assert(std::abs(mm - 3.5) < 1e-9 && "Mean of {5,2} = 3.5");
        std::cout << "  [PASS] masked_mean: mean({5,2}) = 3.5\n";
    }

    std::cout << "\n  All tests passed.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\nChapter 26 — CS336 Alignment Field Guide Demo (C++)\n";
    std::cout << std::string(70, '=') << "\n";

    run_tests();
    demo_off_by_one_explanation();
    demo_masked_operations();
    demo_group_normalization();
    demo_grpo_clip_walkthrough();
    demo_dpo_loss();
    simulate_grpo_training(50, 4);

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  All demos complete.\n";
    std::cout << std::string(70, '=') << "\n";
    return 0;
}

```

