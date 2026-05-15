# Chapter 31: Model Routing and Cascading — Companion Code

## Python — `routing_demo.py`

```python
"""
routing_demo.py — Chapter 31: Model Routing and Cascading

Demonstrates:
  1. Feature-based offline router (rule-based, mimics a trained classifier)
  2. Two-stage cascade with confidence thresholds
  3. Cost model for routing decisions
  4. Break-even analysis per routing policy
  5. Simulated traffic workload with routing metrics
  6. Quality-aware threshold calibration

Run: python routing_demo.py
Requirements: Python stdlib only
"""
from __future__ import annotations
import math, random, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ── Model tiers ───────────────────────────────────────────────────────────────
@dataclass
class ModelTier:
    name:            str
    params_b:        float
    input_price_1k:  float   # USD per 1k input tokens
    output_price_1k: float   # USD per 1k output tokens
    p50_latency_ms:  float
    quality_hard:    float   # 0-1 quality on hard queries
    quality_easy:    float   # 0-1 quality on easy queries

TIERS: Dict[str, ModelTier] = {
    "small":  ModelTier("small",   8, 0.00010, 0.00020,  80.0, 0.72, 0.97),
    "medium": ModelTier("medium", 13, 0.00030, 0.00060, 150.0, 0.85, 0.99),
    "large":  ModelTier("large",  70, 0.00150, 0.00250, 400.0, 0.97, 1.00),
}

# ── Query specs ───────────────────────────────────────────────────────────────
# (qtype, difficulty, language, estimated_tokens, expected_tier)
QUERY_TYPES: Dict[str, Tuple[str, str, str, int, str]] = {
    "faq_order_status":  ("faq",       "easy",   "en",  60, "small"),
    "faq_return_policy": ("faq",       "easy",   "en",  80, "small"),
    "complaint_basic":   ("complaint", "medium", "en", 180, "medium"),
    "billing_dispute":   ("billing",   "hard",   "en", 300, "large"),
    "code_debug":        ("code",      "hard",   "en", 400, "large"),
    "faq_in_spanish":    ("faq",       "easy",   "es", 100, "large"),
    "reasoning_chain":   ("reasoning", "hard",   "en", 500, "large"),
    "simple_classify":   ("classify",  "easy",   "en",  30, "small"),
}

# ── Offline router ────────────────────────────────────────────────────────────
class OfflineRouter:
    _TABLE: Dict[Tuple[str, str], str] = {
        ("faq",       "en"): "small",
        ("classify",  "en"): "small",
        ("complaint", "en"): "medium",
        ("billing",   "*"):  "large",
        ("code",      "*"):  "large",
        ("reasoning", "*"):  "large",
    }

    def route(self, qtype: str, difficulty: str,
              language: str, tokens: int) -> Tuple[str, float]:
        t0 = time.perf_counter()
        if language != "en":          tier = "large"
        elif tokens <= 40:            tier = "small"
        elif difficulty == "hard":    tier = "large"
        else:
            tier = self._TABLE.get((qtype, "en")) or \
                   self._TABLE.get((qtype, "*"), "medium")
        return tier, (time.perf_counter() - t0) * 1000

# ── Cascade router ────────────────────────────────────────────────────────────
@dataclass
class CascadeResult:
    final_tier:       str
    stages_called:    int
    total_latency_ms: float
    total_cost:       float
    confidence:       float

class CascadeRouter:
    def __init__(self, stage1="small", stage2="large", threshold=0.75, seed=42):
        self.stage1 = stage1; self.stage2 = stage2
        self.threshold = threshold; self._rng = random.Random(seed)

    def _conf(self, tier: str, easy: bool) -> float:
        t = TIERS[tier]
        base = t.quality_easy if easy else t.quality_hard
        return max(0.0, min(1.0, base + self._rng.gauss(0, 0.05)))

    def _cost(self, tier: str, tokens: int) -> float:
        t = TIERS[tier]
        return ((tokens*0.6)/1000)*t.input_price_1k + ((tokens*0.4)/1000)*t.output_price_1k

    def _lat(self, tier: str) -> float:
        return TIERS[tier].p50_latency_ms * self._rng.uniform(0.8, 1.3)

    def route(self, easy: bool, tokens: int) -> CascadeResult:
        lat1 = self._lat(self.stage1); cost1 = self._cost(self.stage1, tokens)
        conf1 = self._conf(self.stage1, easy)
        if conf1 >= self.threshold:
            return CascadeResult(self.stage1, 1, lat1, cost1, conf1)
        lat2 = self._lat(self.stage2); cost2 = self._cost(self.stage2, tokens)
        conf2 = self._conf(self.stage2, easy)
        return CascadeResult(self.stage2, 2, lat1+lat2, cost1+cost2, conf2)

# ── Cost model ────────────────────────────────────────────────────────────────
class RoutingCostModel:
    @staticmethod
    def single_tier_monthly(tier: str, avg_tok=200, reqs_day=100_000) -> float:
        t = TIERS[tier]
        r = ((avg_tok*0.6)/1000)*t.input_price_1k + ((avg_tok*0.4)/1000)*t.output_price_1k
        return r * reqs_day * 30

    @staticmethod
    def routed_monthly(dist: Dict[str,float], avg_tok=200, reqs_day=100_000) -> float:
        total = 0.0
        for tier, frac in dist.items():
            t = TIERS[tier]
            r = ((avg_tok*0.6)/1000)*t.input_price_1k + ((avg_tok*0.4)/1000)*t.output_price_1k
            total += frac * r * reqs_day * 30
        return total

# ── Demo helpers ──────────────────────────────────────────────────────────────
def section(title: str) -> None:
    bar = "─"*60
    print(f"\n{bar}\n  {title}\n{bar}")

# ── Demo 1: offline router ────────────────────────────────────────────────────
def demo_offline_router():
    section("Offline Router — Rule-Based Triage")
    router = OfflineRouter()
    print(f"\n  {'Query':<25} {'Diff':<8} {'Lang':<6} {'Tok':>5}  "
          f"{'Routed':<10} {'Expected':<10} {'OK?':>5}")
    print(f"  {'─'*25} {'─'*8} {'─'*6} {'─'*5}  {'─'*10} {'─'*10} {'─'*5}")
    all_ok = True
    for name, (qtype, diff, lang, tok, expected) in QUERY_TYPES.items():
        tier, _ = router.route(qtype, diff, lang, tok)
        ok = (tier == expected); all_ok = all_ok and ok
        print(f"  {name:<25} {diff:<8} {lang:<6} {tok:>5}  "
              f"{tier:<10} {expected:<10} [{'✓' if ok else '✗'}]")
    assert all_ok, "Some routing decisions mismatched expected tiers"
    print(f"\n  [ASSERT] All routing decisions match expected tiers: ✓")

# ── Demo 2: cascade router ────────────────────────────────────────────────────
def demo_cascade_router():
    section("Two-Stage Cascade Router (small → large, τ=0.75)")
    cascade = CascadeRouter("small", "large", threshold=0.75, seed=42)
    N = 2000
    easy_r = [cascade.route(True,  100) for _ in range(N)]
    hard_r = [cascade.route(False, 300) for _ in range(N)]
    easy_hit = sum(1 for r in easy_r if r.stages_called==1) / N
    hard_hit = sum(1 for r in hard_r if r.stages_called==1) / N
    easy_cost = sum(r.total_cost for r in easy_r) / N
    hard_cost = sum(r.total_cost for r in hard_r) / N
    easy_lat  = sum(r.total_latency_ms for r in easy_r) / N
    hard_lat  = sum(r.total_latency_ms for r in hard_r) / N
    large_ec = ((100*0.6)/1000)*TIERS["large"].input_price_1k + ((100*0.4)/1000)*TIERS["large"].output_price_1k
    large_hc = ((300*0.6)/1000)*TIERS["large"].input_price_1k + ((300*0.4)/1000)*TIERS["large"].output_price_1k
    print(f"\n  {'Workload':<10} {'Stage-1 hit':>12} {'Avg cost':>12} {'vs Large-only':>14} {'Avg lat ms':>12}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*14} {'─'*12}")
    print(f"  {'Easy':<10} {easy_hit:>11.1%}  ${easy_cost:>10.6f}  {(1-easy_cost/large_ec)*100:>11.1f}%  {easy_lat:>10.1f}")
    print(f"  {'Hard':<10} {hard_hit:>11.1%}  ${hard_cost:>10.6f}  {(1-hard_cost/large_hc)*100:>11.1f}%  {hard_lat:>10.1f}")
    assert easy_hit >= 0.80, f"Easy hit rate {easy_hit:.1%} < 80%"
    assert hard_hit <= 0.40, f"Hard hit rate {hard_hit:.1%} > 40%"
    print(f"\n  [ASSERT] Easy stage-1 hit rate ≥ 80%: {easy_hit:.1%} ✓")
    print(f"  [ASSERT] Hard escalation rate ≥ 60%:  {1-hard_hit:.1%} ✓")

# ── Demo 3: cost model ────────────────────────────────────────────────────────
def demo_cost_model():
    section("Routing Cost Model")
    avg_tok = 200; reqs_day = 100_000
    dist = {"small": 0.55, "medium": 0.25, "large": 0.20}
    cm = RoutingCostModel()
    c_small  = cm.single_tier_monthly("small",  avg_tok, reqs_day)
    c_medium = cm.single_tier_monthly("medium", avg_tok, reqs_day)
    c_large  = cm.single_tier_monthly("large",  avg_tok, reqs_day)
    c_routed = cm.routed_monthly(dist, avg_tok, reqs_day)
    savings = c_large - c_routed
    print(f"\n  Monthly cost (100k req/day, {avg_tok} avg tokens):\n")
    print(f"  All-small:   ${c_small:>10,.2f}")
    print(f"  All-medium:  ${c_medium:>10,.2f}")
    print(f"  All-large:   ${c_large:>10,.2f}")
    print(f"  Routed:      ${c_routed:>10,.2f}")
    print(f"\n  Savings vs all-large: ${savings:>10,.2f}  ({savings/c_large*100:.1f}%)")
    assert c_routed < c_large, "Routed cost should be < all-large cost"
    print(f"\n  [ASSERT] Routing cheaper than all-large (70% savings): ✓")

# ── Demo 4: break-even ────────────────────────────────────────────────────────
def demo_break_even():
    section("Break-Even: When Does Routing Pay Off?")
    OVERHEAD = 0.000002   # USD/req — CPU classifier cost
    tok = 200
    def cost_req(tier):
        t = TIERS[tier]
        return ((tok*0.6)/1000)*t.input_price_1k + ((tok*0.4)/1000)*t.output_price_1k
    saving = cost_req("large") - cost_req("small")
    be = OVERHEAD / saving
    print(f"\n  Large cost/req:   ${cost_req('large'):.6f}")
    print(f"  Small cost/req:   ${cost_req('small'):.6f}")
    print(f"  Saving/req:       ${saving:.6f}")
    print(f"  Router overhead:  ${OVERHEAD:.6f}")
    print(f"  Break-even:       {be:.4%}")
    print(f"\n  → Route even {be:.2%} of traffic to small model and it pays off.")
    assert be < 0.01
    print(f"\n  [ASSERT] Break-even < 1%: {be:.4%} ✓")

# ── Demo 5: threshold calibration ────────────────────────────────────────────
def demo_threshold_calibration():
    section("Cascade Threshold Calibration — Quality vs Cost")
    rng = random.Random(42); N = 3000
    workload = [(True,100)]*int(N*0.6) + [(False,300)]*int(N*0.4)
    rng.shuffle(workload)
    print(f"\n  {'τ':>8}  {'Escalation':>12}  {'Avg cost':>12}  {'Avg quality':>13}  Note")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*13}  {'─'*20}")
    for tau in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        cascade = CascadeRouter("small","large",threshold=tau,seed=42)
        results = [cascade.route(easy,tok) for easy,tok in workload]
        esc  = sum(1 for r in results if r.stages_called==2) / N
        cost = sum(r.total_cost for r in results) / N
        qual = sum(
            (TIERS["large"].quality_hard if not e else TIERS["large"].quality_easy)
            if r.stages_called==2 else
            (TIERS["small"].quality_easy if e else TIERS["small"].quality_hard)
            for r,(e,_) in zip(results,workload)
        ) / N
        note = "◄ recommended" if tau==0.75 else ""
        print(f"  {tau:>8.2f}  {esc:>11.1%}  ${cost:>11.6f}  {qual:>12.3f}  {note}")
    print(f"\n  τ=0.75: quality≥0.90, escalation≈25%, balanced operating point")

# ── Demo 6: traffic simulation ────────────────────────────────────────────────
def demo_traffic_simulation():
    section("Traffic Simulation — Mixed Workload Routing Metrics")
    router = OfflineRouter(); rng = random.Random(99); N = 10_000
    traffic_mix = [
        ("faq","easy","en",70,0.35), ("classify","easy","en",35,0.15),
        ("complaint","medium","en",180,0.25), ("billing","hard","en",300,0.12),
        ("code","hard","en",420,0.08), ("faq","easy","es",100,0.05),
    ]
    tier_counts: Dict[str,int] = {"small":0,"medium":0,"large":0}
    total_cost = total_lat = 0.0
    for _ in range(N):
        r = rng.random(); cum = 0.0
        qtype,diff,lang,tok,_ = traffic_mix[-1]
        for e in traffic_mix:
            cum += e[4]
            if r < cum: qtype,diff,lang,tok = e[0],e[1],e[2],e[3]; break
        tier,_ = router.route(qtype,diff,lang,tok)
        tier_counts[tier] += 1
        t = TIERS[tier]
        total_cost += ((tok*0.6)/1000)*t.input_price_1k + ((tok*0.4)/1000)*t.output_price_1k
        total_lat  += t.p50_latency_ms
    print(f"\n  Simulated {N:,} requests\n")
    print(f"  {'Tier':<10} {'Count':>8}  {'Fraction':>10}")
    print(f"  {'─'*10} {'─'*8}  {'─'*10}")
    for tier in ["small","medium","large"]:
        n = tier_counts[tier]
        print(f"  {tier:<10} {n:>8}  {n/N:>10.1%}")
    print(f"\n  Total cost:             ${total_cost:.4f}")
    print(f"  Avg cost per request:   ${total_cost/N:.6f}")
    print(f"  Avg latency:            {total_lat/N:.1f} ms")
    assert tier_counts["small"] > tier_counts["large"]
    print(f"\n  [ASSERT] Small tier handles more traffic than large: ✓")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    bar = "="*60
    print(f"\n{bar}\n  Chapter 31 — Model Routing and Cascading (Python)\n{bar}")
    demo_offline_router()
    demo_cascade_router()
    demo_cost_model()
    demo_break_even()
    demo_threshold_calibration()
    demo_traffic_simulation()
    print(f"\n{bar}\n  All demos complete.\n{bar}\n")

if __name__ == "__main__":
    random.seed(42)
    main()

```

## C++ — `routing_demo.cpp`

```cpp
// routing_demo.cpp  —  Chapter 31: Model Routing and Cascading (C++)
//
// Demonstrates:
//   1. Static rule-based router
//   2. Two-stage cascade with confidence simulation
//   3. Cost model and break-even analysis
//   4. Traffic simulation with routing metrics
//   5. Threshold sensitivity analysis
//
// Build: g++ -O2 -std=c++17 -o routing_demo routing_demo.cpp
// Run:   ./routing_demo

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

// ─── Model tiers ─────────────────────────────────────────────────────────────
struct ModelTier {
    std::string name;
    double params_b, input_price_1k, output_price_1k, p50_latency_ms;
    double quality_hard, quality_easy;
};

static const std::map<std::string, ModelTier> TIERS = {
    {"small",  {"small",   8, 0.00010, 0.00020,  80.0, 0.72, 0.97}},
    {"medium", {"medium", 13, 0.00030, 0.00060, 150.0, 0.85, 0.99}},
    {"large",  {"large",  70, 0.00150, 0.00250, 400.0, 0.97, 1.00}},
};

static void print_section(const std::string& t) {
    std::cout << "\n" << std::string(60,'-') << "\n  " << t
              << "\n" << std::string(60,'-') << "\n";
}

// ─── Offline router ───────────────────────────────────────────────────────────
static std::string offline_route(const std::string& qtype,
                                  const std::string& difficulty,
                                  const std::string& language,
                                  int tokens) {
    if (language != "en") return "large";
    if (tokens <= 40)     return "small";
    if (difficulty == "hard") return "large";
    if (qtype == "faq" || qtype == "classify")              return "small";
    if (qtype == "complaint")                               return "medium";
    if (qtype == "billing" || qtype == "code" ||
        qtype == "reasoning")                               return "large";
    return "medium";
}

static void demo_offline_router() {
    print_section("Offline Router — Rule-Based Triage");

    struct Spec { std::string name,qtype,diff,lang; int tok; std::string expected; };
    std::vector<Spec> queries = {
        {"faq_order_status",  "faq",       "easy",   "en",  60, "small"},
        {"faq_return_policy", "faq",       "easy",   "en",  80, "small"},
        {"complaint_basic",   "complaint", "medium", "en", 180, "medium"},
        {"billing_dispute",   "billing",   "hard",   "en", 300, "large"},
        {"code_debug",        "code",      "hard",   "en", 400, "large"},
        {"faq_in_spanish",    "faq",       "easy",   "es", 100, "large"},
        {"reasoning_chain",   "reasoning", "hard",   "en", 500, "large"},
        {"simple_classify",   "classify",  "easy",   "en",  30, "small"},
    };

    std::cout << "\n  " << std::left
              << std::setw(22) << "Query"
              << std::setw(10) << "Diff"
              << std::setw(6)  << "Lang"
              << std::setw(7)  << "Tok"
              << std::setw(10) << "Routed"
              << std::setw(10) << "Expected"
              << "OK?\n";
    std::cout << "  " << std::string(65,'-') << "\n";

    bool all_ok = true;
    for (auto& q : queries) {
        std::string tier = offline_route(q.qtype, q.diff, q.lang, q.tok);
        bool ok = (tier == q.expected);
        if (!ok) all_ok = false;
        std::cout << "  " << std::setw(22) << q.name
                  << std::setw(10) << q.diff << std::setw(6) << q.lang
                  << std::setw(7)  << q.tok  << std::setw(10) << tier
                  << std::setw(10) << q.expected << "[" << (ok?"✓":"✗") << "]\n";
    }
    assert(all_ok);
    std::cout << "\n  [ASSERT] All routing decisions correct: ✓\n";
}

// ─── Cascade router ───────────────────────────────────────────────────────────
struct CascadeResult {
    std::string final_tier;
    int stages_called;
    double total_latency_ms, total_cost, confidence;
};

class CascadeRouter {
public:
    std::string stage1, stage2;
    double threshold;
    std::mt19937 rng;

    CascadeRouter(std::string s1, std::string s2, double thr, uint32_t seed=42)
        : stage1(std::move(s1)), stage2(std::move(s2)), threshold(thr), rng(seed) {}

    double sim_conf(const std::string& tier, bool easy) {
        double base = easy ? TIERS.at(tier).quality_easy : TIERS.at(tier).quality_hard;
        std::normal_distribution<double> n(0.0, 0.05);
        return std::clamp(base + n(rng), 0.0, 1.0);
    }
    double cost(const std::string& tier, int tok) {
        auto& t = TIERS.at(tier);
        return (tok*0.6/1000)*t.input_price_1k + (tok*0.4/1000)*t.output_price_1k;
    }
    double lat(const std::string& tier) {
        std::uniform_real_distribution<double> u(0.8,1.3);
        return TIERS.at(tier).p50_latency_ms * u(rng);
    }

    CascadeResult route(bool easy, int tokens) {
        double l1=lat(stage1), c1=cost(stage1,tokens), cf1=sim_conf(stage1,easy);
        if (cf1 >= threshold) return {stage1, 1, l1, c1, cf1};
        double l2=lat(stage2), c2=cost(stage2,tokens), cf2=sim_conf(stage2,easy);
        return {stage2, 2, l1+l2, c1+c2, cf2};
    }
};

static void demo_cascade() {
    print_section("Two-Stage Cascade Router (small → large, τ=0.75)");

    CascadeRouter cascade("small","large",0.75,42);
    int N=2000;
    int easy_s1=0, hard_s1=0;
    double easy_c=0, hard_c=0, easy_l=0, hard_l=0;

    for (int i=0;i<N;++i) { auto r=cascade.route(true,100);
        if(r.stages_called==1)++easy_s1; easy_c+=r.total_cost; easy_l+=r.total_latency_ms; }
    for (int i=0;i<N;++i) { auto r=cascade.route(false,300);
        if(r.stages_called==1)++hard_s1; hard_c+=r.total_cost; hard_l+=r.total_latency_ms; }

    double eh=easy_s1*1.0/N, hh=hard_s1*1.0/N;
    auto large_cost=[&](int tok){
        return (tok*0.6/1000)*TIERS.at("large").input_price_1k +
               (tok*0.4/1000)*TIERS.at("large").output_price_1k; };

    std::cout << std::fixed;
    std::cout << "\n  " << std::left << std::setw(10)<<"Workload"
              << std::setw(14)<<"Stage-1 hit" << std::setw(16)<<"Avg cost"
              << std::setw(16)<<"vs Large-only" << "Avg lat ms\n";
    std::cout << "  " << std::string(56,'-') << "\n";
    std::cout << "  " << std::setw(10)<<"Easy"
              << std::setw(13) << (int)(eh*100+0.5) << "%"
              << std::setprecision(6) << "  $" << std::setw(12)<<easy_c/N
              << std::setprecision(1) << "  "
              << std::setw(12)<<(1-easy_c/N/large_cost(100))*100<<"%" << "  "
              << easy_l/N << "\n";
    std::cout << "  " << std::setw(10)<<"Hard"
              << std::setw(13) << (int)(hh*100+0.5) << "%"
              << std::setprecision(6) << "  $" << std::setw(12)<<hard_c/N
              << std::setprecision(1) << "  "
              << std::setw(12)<<(1-hard_c/N/large_cost(300))*100<<"%" << "  "
              << hard_l/N << "\n";

    assert(eh >= 0.80);
    assert(hh <= 0.40);
    std::cout << "\n  [ASSERT] Easy stage-1 hit ≥ 80%: " << std::setprecision(1)
              << eh*100 << "% ✓\n";
    std::cout << "  [ASSERT] Hard escalation ≥ 60%:  "
              << (1-hh)*100 << "% ✓\n";
}

// ─── Cost model ───────────────────────────────────────────────────────────────
static void demo_cost_model() {
    print_section("Routing Cost Model");

    int avg_tok=200, reqs_day=100'000, days=30;
    auto monthly=[&](const std::string& tier){
        auto& t=TIERS.at(tier);
        double r=(avg_tok*0.6/1000)*t.input_price_1k+(avg_tok*0.4/1000)*t.output_price_1k;
        return r*reqs_day*days;
    };

    double c_small=monthly("small"), c_medium=monthly("medium"), c_large=monthly("large");
    // 55/25/20 routed distribution
    double c_routed = 0.55*monthly("small")+0.25*monthly("medium")+0.20*monthly("large");

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  Monthly cost (100k req/day, "<<avg_tok<<" avg tokens):\n\n";
    std::cout << "  All-small:   $" << c_small  << "\n";
    std::cout << "  All-medium:  $" << c_medium << "\n";
    std::cout << "  All-large:   $" << c_large  << "\n";
    std::cout << "  Routed:      $" << c_routed << "\n";
    std::cout << "\n  Savings vs all-large: $" << c_large-c_routed
              << " (" << std::setprecision(1) << (1-c_routed/c_large)*100 << "%)\n";

    assert(c_routed < c_large);
    std::cout << "\n  [ASSERT] Routing cheaper than all-large (~70% savings): ✓\n";
}

// ─── Break-even ───────────────────────────────────────────────────────────────
static void demo_break_even() {
    print_section("Break-Even Analysis");

    const double OVERHEAD = 0.000002;
    int tok=200;
    auto cr=[&](const std::string& tier){
        auto& t=TIERS.at(tier);
        return (tok*0.6/1000)*t.input_price_1k+(tok*0.4/1000)*t.output_price_1k; };

    double saving=cr("large")-cr("small");
    double be=OVERHEAD/saving;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n  Large cost/req:   $" << cr("large") << "\n";
    std::cout << "  Small cost/req:   $" << cr("small")  << "\n";
    std::cout << "  Saving/req:       $" << saving       << "\n";
    std::cout << "  Router overhead:  $" << OVERHEAD     << "\n";
    std::cout << std::setprecision(4);
    std::cout << "  Break-even:       " << be*100 << "%\n";
    std::cout << "\n  → Route even " << std::setprecision(2) << be*100
              << "% of traffic to small model and routing pays for itself.\n";

    assert(be < 0.01);
    std::cout << "\n  [ASSERT] Break-even < 1%: " << std::setprecision(4)
              << be*100 << "% ✓\n";
}

// ─── Traffic simulation ───────────────────────────────────────────────────────
static void demo_traffic_simulation() {
    print_section("Traffic Simulation — Mixed Workload Routing Metrics");

    struct TS { std::string qtype,diff,lang; int tok; double frac; };
    std::vector<TS> mix = {
        {"faq","easy","en",70,0.35}, {"classify","easy","en",35,0.15},
        {"complaint","medium","en",180,0.25}, {"billing","hard","en",300,0.12},
        {"code","hard","en",420,0.08}, {"faq","easy","es",100,0.05},
    };

    int N=10000; std::mt19937 rng(99);
    std::uniform_real_distribution<double> uni(0.0,1.0);
    std::map<std::string,int> tc; tc["small"]=tc["medium"]=tc["large"]=0;
    double total_cost=0, total_lat=0;

    for (int i=0;i<N;++i) {
        double r=uni(rng); double cum=0;
        const TS* s=&mix.back();
        for (auto& e:mix){cum+=e.frac;if(r<cum){s=&e;break;}}
        std::string tier=offline_route(s->qtype,s->diff,s->lang,s->tok);
        tc[tier]++;
        auto& t=TIERS.at(tier);
        total_cost+=(s->tok*0.6/1000)*t.input_price_1k+(s->tok*0.4/1000)*t.output_price_1k;
        total_lat+=t.p50_latency_ms;
    }

    std::cout << "\n  Simulated " << N << " requests\n\n";
    std::cout << "  " << std::left << std::setw(10)<<"Tier"
              << std::setw(10)<<"Count" << "Fraction\n";
    std::cout << "  " << std::string(30,'-') << "\n";
    for (auto& [tier,cnt] : tc)
        std::cout << "  " << std::setw(10)<<tier << std::setw(10)<<cnt
                  << std::setprecision(1) << std::fixed << cnt*100.0/N << "%\n";

    std::cout << std::setprecision(6) << "\n  Total cost:            $" << total_cost << "\n";
    std::cout << "  Avg cost per request:  $" << total_cost/N << "\n";
    std::cout << std::setprecision(1) << "  Avg latency:           " << total_lat/N << " ms\n";

    assert(tc["small"] > tc["large"]);
    std::cout << "\n  [ASSERT] Small tier handles more traffic than large: ✓\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "\n" << std::string(60,'=')
              << "\n  Chapter 31 — Model Routing and Cascading (C++)\n"
              << std::string(60,'=') << "\n";

    demo_offline_router();
    demo_cascade();
    demo_cost_model();
    demo_break_even();
    demo_traffic_simulation();

    std::cout << "\n" << std::string(60,'=')
              << "\n  All demos complete.\n"
              << std::string(60,'=') << "\n\n";
    return 0;
}

```

