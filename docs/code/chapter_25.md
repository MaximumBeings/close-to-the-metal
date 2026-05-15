# Chapter 25: RL and Serving Policies — Companion Code

## Python — `serving_policy_demo.py`

```python
#!/usr/bin/env python3
"""
Chapter 25 — Companion Code: RL, Feedback, and Serving Policies
================================================================
Demonstrates bandit algorithms, contextual bandits, offline policy evaluation,
composite reward functions, and reward hacking detection for LLM serving.
No GPU required.  Run: python serving_policy_demo.py
"""

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from collections import defaultdict

random.seed(42)
np.random.seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# §24.3 Bandit Algorithms
# ──────────────────────────────────────────────────────────────────────────────

class EpsilonGreedyBandit:
    """ε-greedy bandit for model routing / prompt selection."""

    def __init__(self, n_actions: int, epsilon: float = 0.1):
        self.n_actions = n_actions
        self.epsilon   = epsilon
        self.counts    = np.zeros(n_actions)
        self.values    = np.zeros(n_actions)

    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.values))

    def update(self, action: int, reward: float):
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n


class UCBBandit:
    """Upper Confidence Bound bandit (UCB1)."""

    def __init__(self, n_actions: int, c: float = 2.0):
        self.n_actions = n_actions
        self.c         = c
        self.counts    = np.zeros(n_actions)
        self.values    = np.zeros(n_actions)
        self.t         = 0

    def select_action(self) -> int:
        self.t += 1
        # Try each action at least once
        for a in range(self.n_actions):
            if self.counts[a] == 0:
                return a
        ucb = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, action: int, reward: float):
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n


class ThompsonSamplingBandit:
    """
    Thompson Sampling with Beta-Bernoulli conjugate update.
    Best for binary rewards (thumbs-up, correct/incorrect).
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.alpha = np.ones(n_actions)   # successes + 1
        self.beta  = np.ones(n_actions)   # failures  + 1

    def select_action(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, action: int, reward: float):
        """reward should be 0.0 or 1.0."""
        self.alpha[action] += float(reward)
        self.beta[action]  += float(1.0 - reward)

    def mean_estimates(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)


# ──────────────────────────────────────────────────────────────────────────────
# §24.3.2 Contextual Bandit (Feature-Bucket Approach)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RequestContext:
    """Features extracted from a serving request."""
    complexity_bucket: str   # "simple" | "medium" | "complex"
    latency_budget_ms: float
    user_tier: str           # "free" | "pro" | "enterprise"
    estimated_tokens: int


class ContextualBandit:
    """
    Contextual bandit via feature bucketing.
    Maintains a separate Thompson Sampling bandit per context bucket.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.bandits: dict[str, ThompsonSamplingBandit] = {}

    def _bucket_key(self, ctx: RequestContext) -> str:
        return f"{ctx.complexity_bucket}_{ctx.user_tier}"

    def _get_bandit(self, ctx: RequestContext) -> ThompsonSamplingBandit:
        key = self._bucket_key(ctx)
        if key not in self.bandits:
            self.bandits[key] = ThompsonSamplingBandit(self.n_actions)
        return self.bandits[key]

    def select_action(self, ctx: RequestContext) -> int:
        return self._get_bandit(ctx).select_action()

    def update(self, ctx: RequestContext, action: int, reward: float):
        self._get_bandit(ctx).update(action, reward)

    def summary(self) -> str:
        lines = ["Contextual Bandit — per-bucket mean reward estimates:"]
        for key, bandit in sorted(self.bandits.items()):
            estimates = bandit.mean_estimates()
            lines.append(f"  [{key}]: " + " | ".join(
                f"A{i}={v:.3f}" for i, v in enumerate(estimates)
            ))
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# §24.5 Reward Function Design
# ──────────────────────────────────────────────────────────────────────────────

def serving_reward(
    latency_ms: float,
    user_rating: float | None,
    is_safe: bool,
    latency_budget_ms: float,
    min_quality: float = 3.0,
) -> float:
    """
    Composite reward: hard safety constraint → quality floor → joint score.
    Returns value in [-1, 1].
    """
    if not is_safe:
        return -1.0
    if user_rating is not None and user_rating < min_quality:
        return 0.0
    latency_factor  = max(0.0, 1.0 - latency_ms / latency_budget_ms)
    quality_factor  = (user_rating / 5.0) if user_rating is not None else 0.5
    return quality_factor * 0.7 + latency_factor * 0.3


def latency_adjusted_reward(
    observed_ms: float,
    target_ms: float,
    quality: float,
    k: float = 3.0,
) -> float:
    """
    Exponential decay penalty for latency overruns.
    quality ∈ [0, 1]; return value ∈ [0, 1].
    """
    if observed_ms <= target_ms:
        latency_factor = 1.0
    else:
        overrun = (observed_ms - target_ms) / target_ms
        latency_factor = math.exp(-k * overrun)
    return quality * latency_factor


# ──────────────────────────────────────────────────────────────────────────────
# §24.6 Offline Policy Evaluation
# ──────────────────────────────────────────────────────────────────────────────

class OfflinePolicyEvaluator:
    """
    Inverse Propensity Scoring (IPS) and Doubly Robust (DR) estimators
    for evaluating a new policy using logged data from an old policy.
    """

    def __init__(self, clip_weight: float = 10.0):
        self.clip_weight = clip_weight

    def ips_estimate(
        self,
        actions:          list[int],
        rewards:          list[float],
        new_policy_probs: list[float],   # π_new(a | x) for each logged (x,a)
        log_policy_probs: list[float],   # π_log(a | x) for each logged (x,a)
    ) -> float:
        """
        Unbiased estimate of E_{π_new}[r] using logged data from π_log.
        Clipped to reduce variance from rare actions.
        """
        assert len(actions) == len(rewards) == len(new_policy_probs) == len(log_policy_probs)
        weights = [
            min(np / lp, self.clip_weight)
            for np, lp in zip(new_policy_probs, log_policy_probs)
        ]
        return sum(w * r for w, r in zip(weights, rewards)) / len(rewards)

    def dr_estimate(
        self,
        actions:           list[int],
        rewards:           list[float],
        new_policy_probs:  list[float],
        log_policy_probs:  list[float],
        reward_model:      Callable[[int], float],   # r̂(x, a) per logged x
    ) -> float:
        """
        Doubly Robust estimator: uses reward model + IPS correction.
        Lower variance than pure IPS; unbiased if either reward model or
        logging policy is correctly specified.
        """
        n = len(rewards)
        total = 0.0
        for a, r, np_prob, lp_prob in zip(
            actions, rewards, new_policy_probs, log_policy_probs
        ):
            r_hat_new_action = reward_model(a)   # DM component
            weight = min(np_prob / lp_prob, self.clip_weight)
            total += r_hat_new_action + weight * (r - reward_model(a))
        return total / n


# ──────────────────────────────────────────────────────────────────────────────
# §24.5.3 Reward Hacking Detector
# ──────────────────────────────────────────────────────────────────────────────

class RewardHackingDetector:
    """
    Monitors for divergence between the optimized reward signal and
    proxy metrics that should correlate with true quality.
    Flags Goodhart's Law violations.
    """

    def __init__(self, correlation_window: int = 100, alert_threshold: float = -0.3):
        self.correlation_window = correlation_window
        self.alert_threshold    = alert_threshold
        self.reward_history:    list[float] = []
        self.proxy_history:     list[float] = []

    def record(self, reward: float, proxy: float):
        self.reward_history.append(reward)
        self.proxy_history.append(proxy)

    def check_for_hacking(self) -> dict:
        n = min(len(self.reward_history), self.correlation_window)
        if n < 20:
            return {"status": "insufficient_data", "n": n}
        recent_rewards = self.reward_history[-n:]
        recent_proxies = self.proxy_history[-n:]
        corr = float(np.corrcoef(recent_rewards, recent_proxies)[0, 1])
        is_hacking = corr < self.alert_threshold
        return {
            "status":      "hacking_detected" if is_hacking else "ok",
            "correlation": round(corr, 4),
            "window":      n,
            "threshold":   self.alert_threshold,
        }


# ──────────────────────────────────────────────────────────────────────────────
# §24.4 Model Routing Simulation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name:              str
    mean_quality:      float      # E[quality | using this model]
    mean_latency_ms:   float
    cost_per_1k_tok:   float
    quality_std:       float = 0.15
    latency_std_ms:    float = 30.0


MODELS = [
    ModelSpec("small_local",  mean_quality=0.55, mean_latency_ms=80,   cost_per_1k_tok=0.0),
    ModelSpec("medium_gpu",   mean_quality=0.75, mean_latency_ms=200,  cost_per_1k_tok=0.10),
    ModelSpec("large_gpu",    mean_quality=0.90, mean_latency_ms=800,  cost_per_1k_tok=0.90),
]


def simulate_model_response(model: ModelSpec, seed: int) -> tuple[float, float]:
    """Returns (quality, latency_ms) for a simulated response from the model."""
    rng = np.random.default_rng(seed)
    quality  = float(np.clip(rng.normal(model.mean_quality, model.quality_std), 0, 1))
    latency  = float(max(10.0, rng.normal(model.mean_latency_ms, model.latency_std_ms)))
    return quality, latency


def run_routing_simulation(n_requests: int = 2000, latency_budget_ms: float = 400.0):
    bandit = ContextualBandit(n_actions=len(MODELS))
    reward_history: list[float] = []
    action_history: list[int]   = []

    contexts = [
        RequestContext(
            complexity_bucket=random.choice(["simple", "medium", "complex"]),
            latency_budget_ms=latency_budget_ms,
            user_tier=random.choice(["free", "pro"]),
            estimated_tokens=random.randint(50, 800),
        )
        for _ in range(n_requests)
    ]

    for i, ctx in enumerate(contexts):
        action  = bandit.select_action(ctx)
        model   = MODELS[action]
        quality, latency = simulate_model_response(model, seed=i)
        reward  = serving_reward(
            latency_ms=latency,
            user_rating=quality * 5.0,
            is_safe=True,
            latency_budget_ms=ctx.latency_budget_ms,
        )
        bandit.update(ctx, action, reward)
        reward_history.append(reward)
        action_history.append(action)

    return bandit, reward_history, action_history


# ──────────────────────────────────────────────────────────────────────────────
# Demonstration
# ──────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def demo_bandits():
    print_section("Bandit Algorithm Comparison (1000 steps, 3 actions)")
    TRUE_REWARDS = [0.3, 0.6, 0.5]    # action 1 is actually best

    algos = {
        "ε-greedy (ε=0.10)": EpsilonGreedyBandit(3, epsilon=0.10),
        "ε-greedy (ε=0.05)": EpsilonGreedyBandit(3, epsilon=0.05),
        "UCB (c=2.0)":        UCBBandit(3, c=2.0),
        "Thompson Sampling":  ThompsonSamplingBandit(3),
    }
    n_steps = 1000
    for name, bandit in algos.items():
        np.random.seed(99)
        total_reward = 0.0
        for t in range(n_steps):
            a      = bandit.select_action()
            r      = float(np.random.random() < TRUE_REWARDS[a])
            bandit.update(a, r)
            total_reward += r
        if hasattr(bandit, 'values'):
            est = bandit.values
        elif hasattr(bandit, 'mean_estimates'):
            est = bandit.mean_estimates()
        else:
            est = bandit.alpha / (bandit.alpha + bandit.beta)
        print(f"  {name:<30}  cumulative_reward={total_reward:.0f}  "
              f"estimates=[{', '.join(f'{v:.3f}' for v in est)}]")
    print(f"\n  True rewards: {TRUE_REWARDS}")
    print(f"  Optimal action: 1 (reward=0.6); optimal cumulative ≈ {0.6*n_steps:.0f}")


def demo_routing_simulation():
    print_section("Model Routing Simulation (2000 requests, contextual bandit)")
    print(f"  Models: {[m.name for m in MODELS]}")
    print(f"  True mean quality: {[m.mean_quality for m in MODELS]}")
    print(f"  True mean latency (ms): {[m.mean_latency_ms for m in MODELS]}")

    bandit, rewards, actions = run_routing_simulation(n_requests=2000)

    # Show convergence by window
    window = 200
    for start in range(0, 2000, window):
        end  = min(start + window, 2000)
        mean_r = np.mean(rewards[start:end])
        counts = [actions[start:end].count(i) for i in range(len(MODELS))]
        print(f"  Requests {start:4d}–{end:4d}: "
              f"mean_reward={mean_r:.3f}  "
              f"route_pcts=[{', '.join(f'{100*c/(end-start):.0f}%' for c in counts)}]")

    print("\n" + bandit.summary())


def demo_reward_design():
    print_section("Reward Function Design Examples")
    cases = [
        ("Fast, good, safe",     50,  4.5, True),
        ("Fast, poor, safe",     50,  1.5, True),
        ("Slow, good, safe",    600,  4.5, True),
        ("Good but unsafe",      50,  4.5, False),
        ("Very slow, good",    2000,  4.5, True),
    ]
    budget = 400.0
    print(f"  {'Scenario':<30} {'latency':>8} {'rating':>7} {'safe':>5} {'reward':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*5} {'-'*8}")
    for desc, lat, rating, safe in cases:
        r = serving_reward(lat, rating, safe, budget)
        print(f"  {desc:<30} {lat:8.0f} {rating:7.1f} {str(safe):>5}  {r:8.4f}")


def demo_offline_evaluation():
    print_section("Offline Policy Evaluation (IPS and DR Estimators)")
    np.random.seed(7)
    n = 500

    # Simulate logged data from old policy (uniform over 3 actions)
    old_action_probs = [1/3, 1/3, 1/3]
    true_rewards     = [0.3, 0.7, 0.5]   # unknown to us; action 1 is best

    actions = np.random.choice(3, size=n)
    rewards = np.array([true_rewards[a] + np.random.normal(0, 0.1) for a in actions])

    # New policy: always choose action 1 (probability 1.0 for action 1)
    new_action_probs_per_logged = [
        1.0 if a == 1 else 0.0 for a in actions
    ]
    log_action_probs_per_logged = [old_action_probs[a] for a in actions]

    evaluator = OfflinePolicyEvaluator(clip_weight=10.0)
    ips_val = evaluator.ips_estimate(
        list(actions), list(rewards),
        new_action_probs_per_logged,
        log_action_probs_per_logged,
    )

    reward_model  = lambda a: true_rewards[a]    # assume we have a good reward model
    dr_val = evaluator.dr_estimate(
        list(actions), list(rewards),
        new_action_probs_per_logged,
        log_action_probs_per_logged,
        reward_model,
    )

    ground_truth = true_rewards[1]    # always action 1 → reward 0.7
    print(f"  Ground truth E[r | new_policy] = {ground_truth:.3f}")
    print(f"  IPS estimate:                    {ips_val:.4f}")
    print(f"  DR  estimate:                    {dr_val:.4f}")
    print(f"\n  IPS error: {abs(ips_val - ground_truth):.4f}")
    print(f"  DR  error: {abs(dr_val  - ground_truth):.4f}")
    print("  (DR typically has lower absolute error due to reward model correction)")


def demo_reward_hacking():
    print_section("Reward Hacking Detection")
    detector = RewardHackingDetector(correlation_window=50, alert_threshold=-0.3)

    # Phase 1: healthy optimization (reward and proxy are positively correlated)
    print("  Phase 1 (steps 1–60): Healthy optimization")
    for t in range(60):
        quality = 0.5 + t / 200.0
        reward  = quality * 0.9 + np.random.normal(0, 0.05)
        proxy   = quality * 0.8 + np.random.normal(0, 0.08)
        detector.record(reward, proxy)
    result = detector.check_for_hacking()
    print(f"    {result}")

    # Phase 2: reward hacking (reward goes up but proxy goes down)
    print("  Phase 2 (steps 61–120): Reward hacking — reward rises, quality proxy falls")
    for t in range(60):
        reward  = 0.8 + t / 150.0 + np.random.normal(0, 0.03)  # hacked reward ↑
        proxy   = 0.7 - t / 100.0 + np.random.normal(0, 0.06)  # true quality ↓
        detector.record(reward, proxy)
    result = detector.check_for_hacking()
    print(f"    {result}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nChapter 25 — Serving Policy Demo")
    print("=" * 70)
    demo_bandits()
    demo_routing_simulation()
    demo_reward_design()
    demo_offline_evaluation()
    demo_reward_hacking()
    print("\n" + "=" * 70)
    print("  All demos complete.")
    print("=" * 70)

```

## C++ — `serving_policy_demo.cpp`

```cpp
/**
 * Chapter 25 — Serving Policy Algorithms
 * =============================================================
 * Implements: ε-greedy, UCB, Thompson Sampling, composite reward function,
 * IPS offline estimator, and reward hacking detection.
 *
 * Build:  g++ -std=c++17 -O2 serving_policy_demo.cpp -o serving_policy_demo
 * Run:    ./serving_policy_demo
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <map>

// ─────────────────────────────────────────────────────────────────────────────
// §24.3  Bandit Algorithms
// ─────────────────────────────────────────────────────────────────────────────

class EpsilonGreedyBandit {
public:
    EpsilonGreedyBandit(int n_actions, double epsilon = 0.1, unsigned seed = 42)
        : n_(n_actions), epsilon_(epsilon), counts_(n_actions, 0),
          values_(n_actions, 0.0), rng_(seed) {}

    int select_action() {
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        if (unif(rng_) < epsilon_) {
            std::uniform_int_distribution<int> rand_action(0, n_ - 1);
            return rand_action(rng_);
        }
        return static_cast<int>(
            std::max_element(values_.begin(), values_.end()) - values_.begin()
        );
    }

    void update(int action, double reward) {
        counts_[action]++;
        double n = static_cast<double>(counts_[action]);
        values_[action] += (reward - values_[action]) / n;
    }

    const std::vector<double>& values() const { return values_; }

private:
    int n_;
    double epsilon_;
    std::vector<int>    counts_;
    std::vector<double> values_;
    std::mt19937 rng_;
};


class UCBBandit {
public:
    UCBBandit(int n_actions, double c = 2.0, unsigned seed = 42)
        : n_(n_actions), c_(c), counts_(n_actions, 0),
          values_(n_actions, 0.0), t_(0), rng_(seed) {}

    int select_action() {
        ++t_;
        for (int a = 0; a < n_; ++a)
            if (counts_[a] == 0) return a;

        std::vector<double> ucb(n_);
        for (int a = 0; a < n_; ++a)
            ucb[a] = values_[a] + c_ * std::sqrt(std::log(t_) / counts_[a]);
        return static_cast<int>(
            std::max_element(ucb.begin(), ucb.end()) - ucb.begin()
        );
    }

    void update(int action, double reward) {
        counts_[action]++;
        double n = static_cast<double>(counts_[action]);
        values_[action] += (reward - values_[action]) / n;
    }

    const std::vector<double>& values() const { return values_; }

private:
    int n_;
    double c_;
    std::vector<int>    counts_;
    std::vector<double> values_;
    int t_;
    std::mt19937 rng_;
};


class ThompsonSamplingBandit {
public:
    ThompsonSamplingBandit(int n_actions, unsigned seed = 42)
        : n_(n_actions), alpha_(n_actions, 1.0), beta_(n_actions, 1.0),
          rng_(seed) {}

    int select_action() {
        std::vector<double> samples(n_);
        for (int a = 0; a < n_; ++a) {
            std::gamma_distribution<double> ga(alpha_[a], 1.0);
            std::gamma_distribution<double> gb(beta_[a],  1.0);
            double x = ga(rng_), y = gb(rng_);
            samples[a] = x / (x + y);   // Beta sample via Gamma ratio
        }
        return static_cast<int>(
            std::max_element(samples.begin(), samples.end()) - samples.begin()
        );
    }

    void update(int action, double reward) {
        alpha_[action] += reward;
        beta_[action]  += (1.0 - reward);
    }

    std::vector<double> mean_estimates() const {
        std::vector<double> means(n_);
        for (int a = 0; a < n_; ++a)
            means[a] = alpha_[a] / (alpha_[a] + beta_[a]);
        return means;
    }

private:
    int n_;
    std::vector<double> alpha_, beta_;
    std::mt19937 rng_;
};


// ─────────────────────────────────────────────────────────────────────────────
// §24.5  Reward Function
// ─────────────────────────────────────────────────────────────────────────────

double serving_reward(
    double latency_ms,
    double user_rating,        // 0–5, or -1 if unavailable
    bool   is_safe,
    double latency_budget_ms,
    double min_quality = 3.0
) {
    if (!is_safe)                              return -1.0;
    if (user_rating >= 0 && user_rating < min_quality) return 0.0;

    double latency_factor = std::max(0.0, 1.0 - latency_ms / latency_budget_ms);
    double quality_factor = (user_rating >= 0) ? user_rating / 5.0 : 0.5;
    return quality_factor * 0.7 + latency_factor * 0.3;
}

double latency_adjusted_reward(
    double observed_ms, double target_ms, double quality, double k = 3.0
) {
    double latency_factor = (observed_ms <= target_ms)
        ? 1.0
        : std::exp(-k * (observed_ms - target_ms) / target_ms);
    return quality * latency_factor;
}


// ─────────────────────────────────────────────────────────────────────────────
// §24.6  Offline Policy Evaluation (IPS)
// ─────────────────────────────────────────────────────────────────────────────

double ips_estimate(
    const std::vector<double>& rewards,
    const std::vector<double>& new_policy_probs,
    const std::vector<double>& log_policy_probs,
    double clip_weight = 10.0
) {
    int n = static_cast<int>(rewards.size());
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        double w = std::min(new_policy_probs[i] / log_policy_probs[i], clip_weight);
        total += w * rewards[i];
    }
    return total / n;
}


// ─────────────────────────────────────────────────────────────────────────────
// Demonstrations
// ─────────────────────────────────────────────────────────────────────────────

static void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "  " << title << "\n"
              << std::string(70, '=') << "\n";
}

void demo_bandits() {
    print_section("Bandit Algorithm Comparison (1000 steps, 3 actions)");

    const std::vector<double> TRUE_REWARDS = {0.3, 0.6, 0.5};
    const int N_STEPS  = 1000;
    const int N_TRIALS = 5;   // repeated trials for stability

    struct AlgoResult { std::string name; double cum_reward; std::vector<double> estimates; };
    std::vector<AlgoResult> results;

    auto run_bandit = [&](auto& bandit, const std::string& name) {
        std::mt19937 env_rng(99);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        double total = 0.0;
        for (int t = 0; t < N_STEPS; ++t) {
            int    a = bandit.select_action();
            double r = (unif(env_rng) < TRUE_REWARDS[a]) ? 1.0 : 0.0;
            bandit.update(a, r);
            total += r;
        }
        return total;
    };

    {
        EpsilonGreedyBandit b(3, 0.10, 42);
        double cr = run_bandit(b, "ε-greedy");
        results.push_back({"ε-greedy (ε=0.10)", cr, b.values()});
    }
    {
        UCBBandit b(3, 2.0, 42);
        double cr = run_bandit(b, "UCB");
        results.push_back({"UCB (c=2.0)", cr, b.values()});
    }
    {
        ThompsonSamplingBandit b(3, 42);
        double cr = run_bandit(b, "Thompson");
        results.push_back({"Thompson Sampling", cr, b.mean_estimates()});
    }

    std::cout << "  " << std::left << std::setw(26) << "Algorithm"
              << std::right << std::setw(18) << "Cumulative reward"
              << "   Estimates\n";
    std::cout << "  " << std::string(68, '-') << "\n";
    for (auto& r : results) {
        std::cout << "  " << std::left  << std::setw(26) << r.name
                  << std::right << std::setw(18) << std::fixed << std::setprecision(0)
                  << r.cum_reward << "   [";
        for (int i = 0; i < (int)r.estimates.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(3) << r.estimates[i];
        }
        std::cout << "]\n";
    }
    std::cout << "\n  True rewards: [0.300, 0.600, 0.500]\n";
    std::cout << "  Optimal action: 1; optimal cumulative ≈ 600\n";
}

void demo_reward_design() {
    print_section("Reward Function Design Examples");

    struct Case { std::string desc; double lat; double rating; bool safe; };
    std::vector<Case> cases = {
        {"Fast, good, safe",    50,  4.5,  true},
        {"Fast, poor, safe",    50,  1.5,  true},
        {"Slow, good, safe",   600,  4.5,  true},
        {"Good but unsafe",     50,  4.5,  false},
        {"Very slow, good",   2000,  4.5,  true},
    };

    double budget = 400.0;
    std::cout << "  " << std::left  << std::setw(30) << "Scenario"
              << std::right << std::setw(10) << "Latency"
              << std::setw(8)  << "Rating"
              << std::setw(7)  << "Safe"
              << std::setw(10) << "Reward" << "\n";
    std::cout << "  " << std::string(65, '-') << "\n";

    for (auto& c : cases) {
        double r = serving_reward(c.lat, c.rating, c.safe, budget);
        std::cout << "  " << std::left << std::setw(30) << c.desc
                  << std::right
                  << std::setw(10) << std::fixed << std::setprecision(0) << c.lat
                  << std::setw(8)  << std::setprecision(1) << c.rating
                  << std::setw(7)  << (c.safe ? "true" : "false")
                  << std::setw(10) << std::setprecision(4) << r << "\n";
    }
}

void demo_offline_evaluation() {
    print_section("IPS Offline Policy Evaluation");

    // Old policy: uniform over 3 actions
    // New policy: always action 1 (best)
    const int N = 500;
    const std::vector<double> true_rewards = {0.3, 0.7, 0.5};
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> action_dist(0, 2);
    std::normal_distribution<double> noise(0.0, 0.1);

    std::vector<double> rewards(N), new_probs(N), log_probs(N);
    for (int i = 0; i < N; ++i) {
        int a       = action_dist(rng);
        rewards[i]  = std::clamp(true_rewards[a] + noise(rng), 0.0, 1.0);
        log_probs[i] = 1.0 / 3.0;
        new_probs[i] = (a == 1) ? 1.0 : 0.0;
    }

    double ips_val = ips_estimate(rewards, new_probs, log_probs, 10.0);
    double ground  = true_rewards[1];

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Ground truth E[r | new_policy] = " << ground       << "\n";
    std::cout << "  IPS estimate:                    " << ips_val       << "\n";
    std::cout << "  Error:                           " << std::abs(ips_val - ground) << "\n";
}

void demo_grpo_clip_math() {
    print_section("GRPO-Clip Function: Numerical Walkthrough");

    const double eps     = 0.2;
    const double adv_pos = 1.0;
    const double adv_neg = -1.0;

    auto clip_val = [&](double ratio, double advantage) -> double {
        double unclipped = ratio * advantage;
        double lo = 1.0 - eps, hi = 1.0 + eps;
        double clipped = std::clamp(ratio, lo, hi) * advantage;
        return std::min(unclipped, clipped);
    };

    std::cout << "  ε = " << eps << ", advantage = +1.0\n";
    std::cout << "  " << std::setw(10) << "ratio"
              << std::setw(14) << "unclipped"
              << std::setw(12) << "clipped_rv"
              << std::setw(12) << "min (obj)"
              << std::setw(10) << "loss\n";
    std::cout << "  " << std::string(58, '-') << "\n";

    for (double ratio : {0.5, 0.8, 1.0, 1.1, 1.5, 2.0}) {
        double obj  = clip_val(ratio, adv_pos);
        double loss = -obj;
        bool is_clipped = (ratio < 1-eps) || (ratio > 1+eps);
        std::cout << std::fixed << std::setprecision(2)
                  << "  " << std::setw(10) << ratio
                  << std::setw(14) << (ratio * adv_pos)
                  << std::setw(12) << std::clamp(ratio, 1-eps, 1+eps) * adv_pos
                  << std::setw(12) << obj
                  << std::setw(10) << loss
                  << (is_clipped ? "  ← clipped" : "") << "\n";
    }
}

void demo_dpo_loss_math() {
    print_section("DPO Loss: Analytical Behavior");

    const double beta = 0.1;
    auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    auto dpo_loss = [&](double log_ratio_diff) {
        return -std::log(sigmoid(beta * log_ratio_diff));
    };

    std::cout << "  When π_θ = π_ref: log_ratio_diff = 0\n";
    std::cout << "  DPO loss = -log(σ(0)) = -log(0.5) = log(2) ≈ "
              << dpo_loss(0.0) << "\n\n";

    std::cout << "  " << std::setw(24) << "log_ratio_diff"
              << std::setw(14) << "β × lrd"
              << std::setw(12) << "σ(β×lrd)"
              << std::setw(12) << "loss\n";
    std::cout << "  " << std::string(62, '-') << "\n";

    for (double lrd : {-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0}) {
        double bx   = beta * lrd;
        double sig  = sigmoid(bx);
        double loss = -std::log(sig);
        std::cout << std::fixed << std::setprecision(2)
                  << "  " << std::setw(24) << lrd
                  << std::setw(14) << bx
                  << std::setw(12) << sig
                  << std::setw(12) << loss << "\n";
    }
    std::cout << "\n  Correctly classified when log_ratio_diff > 0 (model prefers chosen).\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\nChapter 25 — Serving Policy Demo (C++)\n";
    std::cout << std::string(70, '=') << "\n";

    demo_bandits();
    demo_reward_design();
    demo_offline_evaluation();
    demo_grpo_clip_math();
    demo_dpo_loss_math();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  All demos complete.\n";
    std::cout << std::string(70, '=') << "\n";
    return 0;
}

```

