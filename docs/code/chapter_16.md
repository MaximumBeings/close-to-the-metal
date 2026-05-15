# Chapter 16: Observability — Companion Code

## Python — `observability_demo.py`

```python
"""
Chapter 16 Companion Code: Observability — Metrics, Logging, Tracing
=====================================================================
Demonstrates:
  1. Prometheus metrics parser — parse /metrics text format
  2. Histogram percentile calculator — P50/P95/P99 from buckets
  3. Latency health checker — classify TTFT / ITL health
  4. KV cache utilization analyzer — fill trend and alert thresholds
  5. Alert rule evaluator — fire/resolve logic
  6. Simulated time-series: TTFT regression diagnosis walkthrough
  7. Structured log event parser — extract scheduler state lines
  8. llama_perf_context_print parser — extract timing fields
  9. Per-token ITL statistics — mean/stddev/P95/P99/histogram

Run:
    python3 observability_demo.py

No external dependencies.
"""

import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 1. PROMETHEUS METRICS PARSER
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_METRICS = """
# HELP vllm:num_requests_running Number of requests currently running on GPU
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 45
# HELP vllm:num_requests_waiting Number of requests waiting to be processed
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 12
# HELP vllm:num_requests_swapped Number of requests swapped to CPU
# TYPE vllm:num_requests_swapped gauge
vllm:num_requests_swapped 3
# HELP vllm:num_preemptions_total Total number of preemptions
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total 7
# HELP vllm:gpu_cache_usage_perc GPU KV cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.87
# HELP vllm:cpu_cache_usage_perc CPU KV cache usage
# TYPE vllm:cpu_cache_usage_perc gauge
vllm:cpu_cache_usage_perc 0.12
# HELP vllm:gpu_prefix_cache_hit_rate Rolling prefix cache hit rate
# TYPE vllm:gpu_prefix_cache_hit_rate gauge
vllm:gpu_prefix_cache_hit_rate 0.73
# HELP vllm:prompt_tokens_total Total prompt tokens processed
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 4892301
# HELP vllm:generation_tokens_total Total generation tokens produced
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 1234891
# HELP vllm:request_success_total Total successful requests
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop"} 9823
vllm:request_success_total{finished_reason="length"} 177
vllm:request_success_total{finished_reason="abort"} 42
# HELP vllm:e2e_request_latency_seconds E2E request latency
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{le="0.1"} 4
vllm:e2e_request_latency_seconds_bucket{le="0.5"} 87
vllm:e2e_request_latency_seconds_bucket{le="1.0"} 234
vllm:e2e_request_latency_seconds_bucket{le="2.0"} 876
vllm:e2e_request_latency_seconds_bucket{le="5.0"} 1087
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 1092
vllm:e2e_request_latency_seconds_sum 892.4
vllm:e2e_request_latency_seconds_count 1092
# HELP vllm:time_to_first_token_seconds TTFT histogram
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.01"} 0
vllm:time_to_first_token_seconds_bucket{le="0.05"} 2
vllm:time_to_first_token_seconds_bucket{le="0.1"} 18
vllm:time_to_first_token_seconds_bucket{le="0.2"} 134
vllm:time_to_first_token_seconds_bucket{le="0.5"} 412
vllm:time_to_first_token_seconds_bucket{le="1.0"} 678
vllm:time_to_first_token_seconds_bucket{le="2.0"} 891
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 903
vllm:time_to_first_token_seconds_sum 312.4
vllm:time_to_first_token_seconds_count 903
# HELP vllm:time_per_output_token_seconds ITL histogram
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.005"} 0
vllm:time_per_output_token_seconds_bucket{le="0.01"} 45237
vllm:time_per_output_token_seconds_bucket{le="0.02"} 87432
vllm:time_per_output_token_seconds_bucket{le="0.05"} 92103
vllm:time_per_output_token_seconds_bucket{le="0.1"} 92388
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 92401
vllm:time_per_output_token_seconds_sum 1102.3
vllm:time_per_output_token_seconds_count 92401
"""


@dataclass
class Gauge:
    name: str
    labels: dict[str, str]
    value: float


@dataclass
class HistogramBucket:
    le: float      # upper bound (inf for +Inf)
    count: int


@dataclass
class Histogram:
    name: str
    buckets: list[HistogramBucket]
    total_sum: float
    total_count: int

    def percentile(self, p: float) -> float:
        """Interpolated percentile from histogram buckets."""
        if self.total_count == 0:
            return 0.0
        target = p * self.total_count
        prev_count = 0
        prev_le    = 0.0
        for b in self.buckets:
            if b.count >= target:
                # Linear interpolation within bucket
                if b.le == float('inf'):
                    return prev_le * 1.2  # crude: 20% above last finite bucket
                width = b.le - prev_le
                frac  = (target - prev_count) / max(1, b.count - prev_count)
                return prev_le + frac * width
            prev_count = b.count
            if b.le != float('inf'):
                prev_le = b.le
        return prev_le

    def mean(self) -> float:
        return self.total_sum / max(1, self.total_count)


class PrometheusParser:
    def __init__(self, text: str):
        self.gauges:     list[Gauge]     = []
        self.histograms: dict[str, Histogram] = {}
        self._parse(text)

    def _parse(self, text: str):
        hist_buckets: dict[str, list[HistogramBucket]] = {}
        hist_sum:     dict[str, float] = {}
        hist_count:   dict[str, int]   = {}

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse labels
            m_label = re.match(r'(\S+)\{([^}]*)\}\s+(\S+)', line)
            m_plain = re.match(r'(\S+)\s+(\S+)', line)

            if m_label:
                name_full = m_label.group(1)
                label_str = m_label.group(2)
                value_str = m_label.group(3)
                labels = dict(
                    kv.split('=') for kv in
                    re.findall(r'\w+="[^"]*"', label_str)
                )
                labels = {k: v.strip('"') for k, v in labels.items()}

                try:
                    value = float(value_str)
                except ValueError:
                    continue

                # Histogram bucket?
                if name_full.endswith('_bucket'):
                    base = name_full[:-7]
                    le_str = labels.get('le', '0')
                    le_val = float('inf') if le_str == '+Inf' else float(le_str)
                    hist_buckets.setdefault(base, []).append(
                        HistogramBucket(le_val, int(value)))
                else:
                    self.gauges.append(Gauge(name_full, labels, value))

            elif m_plain:
                name_full = m_plain.group(1)
                try:
                    value = float(m_plain.group(2))
                except ValueError:
                    continue

                if name_full.endswith('_sum'):
                    hist_sum[name_full[:-4]] = value
                elif name_full.endswith('_count'):
                    hist_count[name_full[:-6]] = int(value)
                else:
                    self.gauges.append(Gauge(name_full, {}, value))

        # Assemble histograms
        for base, buckets in hist_buckets.items():
            buckets.sort(key=lambda b: b.le)
            self.histograms[base] = Histogram(
                name=base,
                buckets=buckets,
                total_sum=hist_sum.get(base, 0.0),
                total_count=hist_count.get(base, 0),
            )

    def get_gauge(self, name: str) -> Optional[float]:
        for g in self.gauges:
            if g.name == name and not g.labels:
                return g.value
        return None

    def get_labelled_gauges(self, name: str) -> list[Gauge]:
        return [g for g in self.gauges if g.name == name]

    def get_histogram(self, name: str) -> Optional[Histogram]:
        return self.histograms.get(name)


def print_metrics_summary(text: str):
    p = PrometheusParser(text)
    print(f"\n{'='*62}")
    print("  vLLM Prometheus Metrics Summary")
    print(f"{'='*62}")

    # Gauges
    gauge_names = [
        ("vllm:num_requests_running",    "Running requests"),
        ("vllm:num_requests_waiting",    "Waiting requests"),
        ("vllm:num_requests_swapped",    "Swapped to CPU"),
        ("vllm:num_preemptions_total",   "Total preemptions"),
        ("vllm:gpu_cache_usage_perc",    "GPU KV cache fill"),
        ("vllm:cpu_cache_usage_perc",    "CPU KV cache fill"),
        ("vllm:gpu_prefix_cache_hit_rate","Prefix cache hit rate"),
    ]
    print("\n  Gauges / Counters:")
    for metric, label in gauge_names:
        v = p.get_gauge(metric)
        if v is not None:
            if 'perc' in metric or 'rate' in metric:
                print(f"    {label:<32} : {v*100:.1f}%")
            else:
                print(f"    {label:<32} : {v:,.0f}")

    # Success breakdown
    succ = p.get_labelled_gauges("vllm:request_success_total")
    if succ:
        print("\n  Request outcomes:")
        for g in succ:
            reason = g.labels.get("finished_reason", "?")
            print(f"    {reason:<12} : {g.value:,.0f}")

    # Histograms
    hist_specs = [
        ("vllm:e2e_request_latency_seconds",   "E2E latency  ", 1000),
        ("vllm:time_to_first_token_seconds",   "TTFT         ", 1000),
        ("vllm:time_per_output_token_seconds", "ITL          ", 1000),
    ]
    print("\n  Latency histograms (ms):")
    print(f"  {'Metric':<18}  {'Mean':>8}  {'P50':>8}  {'P95':>8}  {'P99':>8}")
    print(f"  {'─'*18}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for hist_name, label, scale in hist_specs:
        h = p.get_histogram(hist_name)
        if h:
            mean = h.mean() * scale
            p50  = h.percentile(0.50) * scale
            p95  = h.percentile(0.95) * scale
            p99  = h.percentile(0.99) * scale
            print(f"  {label}  {mean:>8.1f}  {p50:>8.1f}  {p95:>8.1f}  {p99:>8.1f}")

    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. LATENCY HEALTH CHECKER
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HealthThresholds:
    ttft_warn_ms:  float = 500.0
    ttft_crit_ms:  float = 2000.0
    itl_warn_ms:   float = 50.0
    itl_crit_ms:   float = 200.0
    kv_warn:       float = 0.85
    kv_crit:       float = 0.95
    queue_warn:    int   = 10
    queue_crit:    int   = 30
    preempt_warn:  float = 0.5   # per second


def health_check(text: str, thresholds: HealthThresholds = HealthThresholds()):
    p = PrometheusParser(text)
    issues = []

    def status(val, warn, crit, fmt=".1f", scale=1.0):
        v = val * scale
        if v >= crit * scale: return "CRIT", v
        if v >= warn * scale: return "WARN", v
        return "OK  ", v

    # TTFT P95
    h_ttft = p.get_histogram("vllm:time_to_first_token_seconds")
    if h_ttft:
        ttft_p95_ms = h_ttft.percentile(0.95) * 1000
        st, _ = status(ttft_p95_ms, thresholds.ttft_warn_ms, thresholds.ttft_crit_ms)
        issues.append((st, f"TTFT P95 = {ttft_p95_ms:.0f} ms "
                        f"(warn>{thresholds.ttft_warn_ms:.0f}, crit>{thresholds.ttft_crit_ms:.0f})"))

    # ITL P99
    h_itl = p.get_histogram("vllm:time_per_output_token_seconds")
    if h_itl:
        itl_p99_ms = h_itl.percentile(0.99) * 1000
        st, _ = status(itl_p99_ms, thresholds.itl_warn_ms, thresholds.itl_crit_ms)
        issues.append((st, f"ITL P99  = {itl_p99_ms:.0f} ms "
                        f"(warn>{thresholds.itl_warn_ms:.0f}, crit>{thresholds.itl_crit_ms:.0f})"))

    # KV cache
    kv = p.get_gauge("vllm:gpu_cache_usage_perc")
    if kv is not None:
        st, _ = status(kv, thresholds.kv_warn, thresholds.kv_crit)
        issues.append((st, f"KV fill  = {kv*100:.1f}% "
                        f"(warn>{thresholds.kv_warn*100:.0f}%, crit>{thresholds.kv_crit*100:.0f}%)"))

    # Queue depth
    waiting = p.get_gauge("vllm:num_requests_waiting")
    if waiting is not None:
        w = int(waiting)
        st = "CRIT" if w >= thresholds.queue_crit else "WARN" if w >= thresholds.queue_warn else "OK  "
        issues.append((st, f"Queue    = {w} waiting "
                        f"(warn>{thresholds.queue_warn}, crit>{thresholds.queue_crit})"))

    print(f"\n{'='*62}")
    print("  Health Check")
    print(f"{'='*62}")
    any_crit = any(i[0] == "CRIT" for i in issues)
    any_warn = any(i[0] == "WARN" for i in issues)
    overall = "CRITICAL" if any_crit else "WARNING" if any_warn else "HEALTHY"
    print(f"  Overall: {overall}")
    print()
    for status_str, msg in issues:
        icon = "!!" if status_str.strip() == "CRIT" else "! " if status_str.strip() == "WARN" else "✓ "
        print(f"  [{status_str}] {icon} {msg}")
    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. KV CACHE utilization TIME SERIES
# ──────────────────────────────────────────────────────────────────────────────

def simulate_kv_time_series(n_steps: int = 60,
                             seed: int = 42) -> list[tuple[float, float, int]]:
    """
    Simulate a KV cache utilization time series.
    Returns list of (time_s, kv_fill, waiting_queue).
    Simulates a traffic spike at step 30 causing KV saturation.
    """
    random.seed(seed)
    series = []
    kv = 0.55
    queue = 0

    for t in range(n_steps):
        # Traffic spike: steps 30-45 have 3× normal arrival rate
        spike = 1 + 2.0 * (30 <= t <= 45) * max(0, 1 - (t - 30) / 20)
        # KV fill drifts up during spike, down otherwise
        delta = random.gauss(0.008 * spike - 0.003, 0.005)
        kv = max(0.30, min(0.999, kv + delta))
        # Queue grows when KV is near saturation
        if kv > 0.93:
            queue = min(queue + random.randint(3, 8), 80)
        else:
            queue = max(0, queue - random.randint(0, 4))
        series.append((t * 5.0, kv, queue))

    return series


def print_kv_time_series(series: list[tuple[float, float, int]]):
    width = 40
    print(f"\n{'='*70}")
    print("  KV Cache utilization Time Series (simulated 5-min window)")
    print(f"{'='*70}")
    print(f"  {'Time':>6}  {'KV fill':>8}  {'Queue':>6}  "
          f"{'utilization bar':<{width}}  Status")
    print(f"  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*width}  {'─'*8}")
    for ts, kv, q in series[::3]:   # sample every 3 steps for readability
        filled = int(kv * width)
        bar    = '█' * filled + '░' * (width - filled)
        status = "CRIT" if kv > 0.95 else "WARN" if kv > 0.85 else "OK  "
        icon   = "!!" if status == "CRIT" else "! " if status == "WARN" else "  "
        print(f"  {ts:>5.0f}s  {kv*100:>7.1f}%  {q:>6}  {bar}  {status}{icon}")
    print(f"{'='*70}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. ALERT RULE EVALUATOR
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AlertRule:
    name: str
    expr_fn: object        # callable(metrics_text) -> bool
    for_steps: int = 2     # must be true this many consecutive steps
    severity: str = "warning"
    annotation: str = ""
    _consecutive: int = 0
    firing: bool = False

    def evaluate(self, metrics_text: str) -> tuple[bool, str]:
        triggered = self.expr_fn(metrics_text)
        if triggered:
            self._consecutive += 1
        else:
            self._consecutive = 0
            self.firing = False

        if self._consecutive >= self.for_steps and not self.firing:
            self.firing = True
            return True, f"FIRING [{self.severity.upper()}] {self.name}: {self.annotation}"
        if not triggered and self.firing:
            self.firing = False
            return True, f"RESOLVED {self.name}"
        return False, ""


def make_alert_rules() -> list[AlertRule]:
    def kv_saturated(text):
        p = PrometheusParser(text)
        v = p.get_gauge("vllm:gpu_cache_usage_perc")
        return v is not None and v > 0.95

    def high_queue(text):
        p = PrometheusParser(text)
        v = p.get_gauge("vllm:num_requests_waiting")
        return v is not None and v > 20

    def high_preemptions(text):
        p = PrometheusParser(text)
        v = p.get_gauge("vllm:num_preemptions_total")
        return v is not None and v > 10   # simplified: counter threshold

    return [
        AlertRule("KVCacheSaturated", kv_saturated, for_steps=2,
                  severity="critical", annotation="KV cache >95%"),
        AlertRule("RequestsAccumulating", high_queue, for_steps=1,
                  severity="warning",  annotation=">20 requests waiting"),
        AlertRule("HighPreemptionCount", high_preemptions, for_steps=1,
                  severity="warning",  annotation="Preemptions accumulating"),
    ]


def print_alert_evaluation(metrics_series: list[str]):
    rules = make_alert_rules()
    print(f"\n{'='*62}")
    print("  Alert Rule Evaluation (simulated metric snapshots)")
    print(f"{'='*62}")
    for i, snap in enumerate(metrics_series):
        p = PrometheusParser(snap)
        kv    = p.get_gauge("vllm:gpu_cache_usage_perc") or 0
        queue = p.get_gauge("vllm:num_requests_waiting") or 0
        print(f"\n  t={i*30}s  KV={kv*100:.0f}%  queue={int(queue)}")
        for rule in rules:
            fired, msg = rule.evaluate(snap)
            if msg:
                print(f"    → {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. STRUCTURED LOG PARSER
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_LOGS = """
INFO scheduler.py:321 - Running: 45 reqs, Waiting: 3 reqs, Swapped: 0 reqs, Budget: 6144/8192 tokens
INFO scheduler.py:321 - Running: 46 reqs, Waiting: 3 reqs, Swapped: 0 reqs, Budget: 6912/8192 tokens
WARN scheduler.py:287 - Preempting request abc123 due to KV capacity: running=48 seqs kv_used=98.3%
INFO scheduler.py:321 - Running: 44 reqs, Waiting: 12 reqs, Swapped: 3 reqs, Budget: 7168/8192 tokens
DEBUG block_manager.py:441 - Swapping in request def456: blocks=34 src=cpu dst=gpu latency_ms=23.4
DEBUG prefix_cache.py:178 - Cache HIT request ghi789: matched_blocks=127 total_blocks=128 saved_tokens=2032
INFO scheduler.py:321 - Running: 42 reqs, Waiting: 18 reqs, Swapped: 0 reqs, Budget: 7424/8192 tokens
WARN scheduler.py:287 - Preempting request xyz789 due to KV capacity: running=45 seqs kv_used=97.1%
"""


@dataclass
class SchedulerEvent:
    ts: float
    running: int = 0
    waiting: int = 0
    swapped: int = 0
    budget_used: int = 0
    budget_total: int = 0
    event_type: str = "step"
    details: str = ""


def parse_scheduler_logs(log_text: str) -> list[SchedulerEvent]:
    events = []
    ts = time.time()
    for i, line in enumerate(log_text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        ev = SchedulerEvent(ts=ts + i * 0.5)

        m = re.search(r'Running: (\d+).*Waiting: (\d+).*Swapped: (\d+).*Budget: (\d+)/(\d+)', line)
        if m:
            ev.running      = int(m.group(1))
            ev.waiting      = int(m.group(2))
            ev.swapped      = int(m.group(3))
            ev.budget_used  = int(m.group(4))
            ev.budget_total = int(m.group(5))
            ev.event_type   = "step"
            events.append(ev)

        elif 'Preempting' in line:
            m2 = re.search(r'request (\w+) .*kv_used=([\d.]+)%', line)
            ev.event_type = "preempt"
            ev.details    = m2.group(0) if m2 else line
            events.append(ev)

        elif 'Cache HIT' in line:
            m3 = re.search(r'matched_blocks=(\d+).*saved_tokens=(\d+)', line)
            ev.event_type = "cache_hit"
            ev.details    = m3.group(0) if m3 else line
            events.append(ev)

    return events


def print_log_analysis(log_text: str):
    events = parse_scheduler_logs(log_text)
    print(f"\n{'='*70}")
    print("  Scheduler Log Analysis")
    print(f"{'='*70}")
    print(f"  {'Event':<12}  {'Running':>8}  {'Waiting':>8}  "
          f"{'Budget%':>8}  {'Detail'}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*30}")
    for ev in events:
        if ev.event_type == "step":
            bp = ev.budget_used / max(1, ev.budget_total) * 100
            alarm = " !" if ev.waiting > 5 else ""
            print(f"  {'step':<12}  {ev.running:>8}  {ev.waiting:>8}  "
                  f"{bp:>7.0f}%  {alarm}")
        elif ev.event_type == "preempt":
            print(f"  {'PREEMPT':<12}  {'─':>8}  {'─':>8}  {'─':>8}  "
                  f"{ev.details[:40]}")
        elif ev.event_type == "cache_hit":
            print(f"  {'CACHE HIT':<12}  {'─':>8}  {'─':>8}  {'─':>8}  "
                  f"{ev.details[:40]}")

    # Summary
    steps     = [e for e in events if e.event_type == "step"]
    preempts  = [e for e in events if e.event_type == "preempt"]
    hits      = [e for e in events if e.event_type == "cache_hit"]
    print(f"\n  Summary: {len(steps)} scheduler steps, "
          f"{len(preempts)} preemptions, {len(hits)} cache hits")
    if steps:
        avg_wait = sum(e.waiting for e in steps) / len(steps)
        avg_bp   = sum(e.budget_used / max(1, e.budget_total) for e in steps) / len(steps) * 100
        print(f"  Avg waiting: {avg_wait:.1f} reqs  |  Avg budget: {avg_bp:.0f}%")
    print(f"{'='*70}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. LLAMA_PERF_CONTEXT_PRINT PARSER
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_LLAMA_PERF = """
llama_perf_context_print:        load time =     423.12 ms
llama_perf_context_print:      sample time =      12.43 ms /   256 runs (    0.05 ms per token, 20597.39 tokens per second)
llama_perf_context_print: prompt eval time =    1204.12 ms /   512 tokens (    2.35 ms per token,   425.23 tokens per second)
llama_perf_context_print:        eval time =    4892.34 ms /   256 runs (   19.11 ms per token,    52.33 tokens per second)
llama_perf_context_print:       total time =    6120.01 ms /   768 tokens
"""


@dataclass
class LlamaPerfStats:
    load_ms:          float = 0.0
    sample_ms:        float = 0.0
    sample_n:         int   = 0
    sample_tok_s:     float = 0.0
    prompt_eval_ms:   float = 0.0
    prompt_tokens:    int   = 0
    prompt_tok_ms:    float = 0.0   # ms per token
    eval_ms:          float = 0.0
    eval_n:           int   = 0
    eval_tok_ms:      float = 0.0   # ITL
    total_ms:         float = 0.0
    total_tokens:     int   = 0


def parse_llama_perf(text: str) -> LlamaPerfStats:
    s = LlamaPerfStats()
    for line in text.splitlines():
        m_load  = re.search(r'load time\s*=\s*([\d.]+) ms', line)
        m_samp  = re.search(r'sample time\s*=\s*([\d.]+) ms /\s*(\d+) runs.*?([\d.]+) tokens per second', line)
        m_peval = re.search(r'prompt eval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) ms per token', line)
        m_eval  = re.search(r'eval time\s*=\s*([\d.]+) ms /\s*(\d+) runs.*?([\d.]+) ms per token', line)
        m_total = re.search(r'total time\s*=\s*([\d.]+) ms /\s*(\d+) tokens', line)

        if m_load:  s.load_ms        = float(m_load.group(1))
        if m_samp:
            s.sample_ms   = float(m_samp.group(1))
            s.sample_n    = int(m_samp.group(2))
            s.sample_tok_s = float(m_samp.group(3))
        if m_peval:
            s.prompt_eval_ms  = float(m_peval.group(1))
            s.prompt_tokens   = int(m_peval.group(2))
            s.prompt_tok_ms   = float(m_peval.group(3))
        if m_eval:
            s.eval_ms      = float(m_eval.group(1))
            s.eval_n       = int(m_eval.group(2))
            s.eval_tok_ms  = float(m_eval.group(3))
        if m_total:
            s.total_ms     = float(m_total.group(1))
            s.total_tokens = int(m_total.group(2))
    return s


def print_llama_perf(text: str):
    s = parse_llama_perf(text)
    print(f"\n{'='*62}")
    print("  llama_perf_context_print Analysis")
    print(f"{'='*62}")
    print(f"  Load time          : {s.load_ms:>8.1f} ms  (one-time startup)")
    print(f"  Prompt eval (TTFT) : {s.prompt_eval_ms:>8.1f} ms  "
          f"({s.prompt_tokens} tokens @ {s.prompt_tok_ms:.2f} ms/tok)")
    print(f"  Decode (ITL)       : {s.eval_tok_ms:>8.2f} ms/tok  "
          f"({s.eval_n} tokens, {s.eval_ms:.0f} ms total)")
    print(f"  Sample overhead    : {s.sample_ms:>8.1f} ms  "
          f"({s.sample_ms/max(1,s.eval_ms)*100:.1f}% of decode)")
    print(f"  Total              : {s.total_ms:>8.1f} ms  "
          f"({s.total_tokens} tokens)")
    throughput = s.total_tokens / (s.total_ms / 1000)
    print(f"  Overall throughput : {throughput:>8.1f} tok/s")
    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. PER-TOKEN ITL STATISTICS AND HISTOGRAM
# ──────────────────────────────────────────────────────────────────────────────

def simulate_token_latencies(n: int = 256, seed: int = 99) -> list[float]:
    """
    Simulate per-token latency series for a decode run.
    Includes occasional spikes (chunked prefill interrupts, NCCL stall).
    """
    random.seed(seed)
    lats = []
    for i in range(n):
        base = random.gauss(19.0, 2.0)           # nominal 19 ms ITL
        # Occasional prefill chunk interrupts every ~32 tokens
        if i > 0 and i % 32 == 0:
            base += random.uniform(30, 80)        # chunked prefill stall
        lats.append(max(1.0, base))
    return lats


@dataclass
class LatencyStats:
    n: int
    mean: float
    stddev: float
    p50: float
    p95: float
    p99: float
    min_val: float
    max_val: float


def compute_stats(lats: list[float]) -> LatencyStats:
    s = sorted(lats)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean)**2 for x in s) / n
    return LatencyStats(
        n=n, mean=mean, stddev=math.sqrt(variance),
        p50=s[int(0.50 * n)], p95=s[int(0.95 * n)], p99=s[int(0.99 * n)],
        min_val=s[0], max_val=s[-1],
    )


def print_itl_histogram(lats: list[float]):
    stats = compute_stats(lats)
    print(f"\n{'='*62}")
    print(f"  Per-Token ITL Statistics  (n={stats.n} tokens)")
    print(f"{'='*62}")
    print(f"  Mean   : {stats.mean:>8.2f} ms")
    print(f"  StdDev : {stats.stddev:>8.2f} ms")
    print(f"  Min    : {stats.min_val:>8.2f} ms")
    print(f"  P50    : {stats.p50:>8.2f} ms")
    print(f"  P95    : {stats.p95:>8.2f} ms")
    print(f"  P99    : {stats.p99:>8.2f} ms")
    print(f"  Max    : {stats.max_val:>8.2f} ms")

    # ASCII histogram
    print(f"\n  ITL Distribution (ms):")
    bins = [0, 10, 15, 20, 25, 30, 40, 60, 100, float('inf')]
    labels = ["<10", "10-15", "15-20", "20-25", "25-30", "30-40", "40-60", "60-100", ">100"]
    counts = [0] * len(labels)
    for v in lats:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i+1]:
                counts[i] += 1
                break

    max_count = max(counts) if counts else 1
    bar_width  = 30
    for label, count in zip(labels, counts):
        bar = '█' * int(count / max_count * bar_width)
        pct = count / len(lats) * 100
        print(f"  {label:>8}  {bar:<{bar_width}}  {count:>5} ({pct:.1f}%)")

    # Spike detection
    spikes = [l for l in lats if l > stats.mean + 3 * stats.stddev]
    if spikes:
        print(f"\n  Spikes (>mean+3σ): {len(spikes)} events, "
              f"max={max(spikes):.1f}ms")
        print(f"  → Likely cause: chunked prefill interrupts or NCCL stalls")
    print(f"{'='*62}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 62)
    print("  Chapter 16 — Observability: Companion Demo")
    print("█" * 62)

    # Section 1: Parse and summarize Prometheus metrics
    print_metrics_summary(SAMPLE_METRICS)

    # Section 2: Health check
    health_check(SAMPLE_METRICS)

    # Section 3: KV cache time series
    series = simulate_kv_time_series(n_steps=60)
    print_kv_time_series(series)

    # Section 4: Alert rule evaluation across simulated snapshots
    # Create snapshots with rising KV fill
    def make_snap(kv: float, waiting: int, preemptions: int) -> str:
        return (f"vllm:gpu_cache_usage_perc {kv}\n"
                f"vllm:num_requests_waiting {waiting}\n"
                f"vllm:num_preemptions_total {preemptions}\n")

    snapshots = [
        make_snap(0.80,  5,  2),   # t=0s: healthy
        make_snap(0.88, 10,  5),   # t=30s: warning
        make_snap(0.96, 35, 12),   # t=60s: critical
        make_snap(0.97, 47, 18),   # t=90s: critical (persists)
        make_snap(0.90, 22,  9),   # t=120s: improving
        make_snap(0.78,  4,  9),   # t=150s: resolved
    ]
    print_alert_evaluation(snapshots)

    # Section 5: Log analysis
    print_log_analysis(SAMPLE_LOGS)

    # Section 6: llama.cpp perf output
    print_llama_perf(SAMPLE_LLAMA_PERF)

    # Section 7: Per-token ITL statistics
    lats = simulate_token_latencies(n=256)
    print_itl_histogram(lats)


if __name__ == "__main__":
    main()

```

## C++ — `observability_demo.cpp`

```cpp
/*
 * Chapter 16 Companion Code: Observability — Metrics, Logging, Tracing
 * =====================================================================
 * Demonstrates:
 *   1. llama_perf_context_print output parser
 *   2. Per-token ITL timing harness (simulated decode loop)
 *   3. ITL statistics: mean / stddev / min / max / P95 / P99
 *   4. ASCII ITL histogram with spike detection
 *   5. Prometheus text-format metric emitter (for llama-server wrappers)
 *   6. KV cache utilization trend analyzer
 *   7. Alert rule evaluator (threshold-based)
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -o observability_demo observability_demo.cpp
 * Run:
 *   ./observability_demo
 *
 * No external dependencies.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

static std::string repeat(char c, int n) {
    return std::string(static_cast<size_t>(std::max(0, n)), c);
}

static std::string fmt_f(double v, int prec = 2) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. LLAMA_PERF FIELDS
// ─────────────────────────────────────────────────────────────────────────────

struct LlamaPerfStats {
    double load_ms        = 0;
    double sample_ms      = 0;
    int    sample_n       = 0;
    double sample_tok_s   = 0;
    double prompt_eval_ms = 0;
    int    prompt_tokens  = 0;
    double prompt_tok_ms  = 0;
    double eval_ms        = 0;
    int    eval_n         = 0;
    double eval_tok_ms    = 0;   // per-token decode latency (ITL)
    double total_ms       = 0;
    int    total_tokens   = 0;
};

// Simplified parser: matches key fields from known format
static LlamaPerfStats parse_llama_perf(const std::string& text) {
    LlamaPerfStats s;
    std::istringstream ss(text);
    std::string line;

    auto extract_first_float = [](const std::string& l) -> double {
        size_t pos = l.find('=');
        if (pos == std::string::npos) return 0;
        std::istringstream tmp(l.substr(pos + 1));
        double v = 0; tmp >> v; return v;
    };
    auto extract_int_after = [](const std::string& l, const std::string& tok) -> int {
        size_t pos = l.find(tok);
        if (pos == std::string::npos) return 0;
        std::istringstream tmp(l.substr(pos + tok.size()));
        int v = 0; tmp >> v; return v;
    };

    while (std::getline(ss, line)) {
        if (line.find("load time") != std::string::npos)
            s.load_ms = extract_first_float(line);
        else if (line.find("sample time") != std::string::npos) {
            s.sample_ms  = extract_first_float(line);
            s.sample_n   = extract_int_after(line, "/ ");
        }
        else if (line.find("prompt eval time") != std::string::npos) {
            s.prompt_eval_ms = extract_first_float(line);
            s.prompt_tokens  = extract_int_after(line, "/ ");
            // ms per token: look for "    X.XX ms per token"
            size_t p = line.rfind("(");
            if (p != std::string::npos) {
                std::istringstream tmp(line.substr(p + 1));
                tmp >> s.prompt_tok_ms;
            }
        }
        else if (line.find("eval time") != std::string::npos &&
                 line.find("prompt") == std::string::npos) {
            s.eval_ms   = extract_first_float(line);
            s.eval_n    = extract_int_after(line, "/ ");
            size_t p = line.rfind("(");
            if (p != std::string::npos) {
                std::istringstream tmp(line.substr(p + 1));
                tmp >> s.eval_tok_ms;
            }
        }
        else if (line.find("total time") != std::string::npos) {
            s.total_ms     = extract_first_float(line);
            s.total_tokens = extract_int_after(line, "/ ");
        }
    }
    return s;
}

static void print_llama_perf(const LlamaPerfStats& s) {
    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  llama_perf_context_print Analysis\n";
    std::cout << repeat('=', 62) << "\n";

    auto row = [](const std::string& label, const std::string& val,
                  const std::string& note = "") {
        std::cout << "  " << std::left << std::setw(24) << label
                  << std::right << std::setw(10) << val;
        if (!note.empty()) std::cout << "  " << note;
        std::cout << "\n";
    };

    row("Load time",           fmt_f(s.load_ms)   + " ms",  "(one-time startup)");
    row("Prompt eval (TTFT)",  fmt_f(s.prompt_eval_ms) + " ms",
        "(" + std::to_string(s.prompt_tokens) + " tokens @ " + fmt_f(s.prompt_tok_ms) + " ms/tok)");
    row("Decode ITL",          fmt_f(s.eval_tok_ms) + " ms/tok",
        "(" + std::to_string(s.eval_n) + " tokens, " + fmt_f(s.eval_ms, 0) + " ms total)");
    row("Sample overhead",     fmt_f(s.sample_ms)  + " ms",
        fmt_f(s.sample_ms / std::max(1.0, s.eval_ms) * 100, 1) + "% of decode");
    row("Total",               fmt_f(s.total_ms, 0) + " ms",
        std::to_string(s.total_tokens) + " tokens");

    double tps = s.total_tokens / (s.total_ms / 1000.0);
    row("Overall throughput",  fmt_f(tps, 1) + " tok/s", "");
    std::cout << repeat('=', 62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. PER-TOKEN TIMING HARNESS (simulated decode loop)
// ─────────────────────────────────────────────────────────────────────────────

struct PerTokenMetrics {
    std::vector<double> latencies_ms;

    double mean() const {
        double s = 0;
        for (double v : latencies_ms) s += v;
        return s / latencies_ms.size();
    }

    double stddev() const {
        double m = mean(), sq = 0;
        for (double v : latencies_ms) sq += (v - m) * (v - m);
        return std::sqrt(sq / latencies_ms.size());
    }

    double percentile(double p) const {
        auto s = latencies_ms;
        std::sort(s.begin(), s.end());
        size_t idx = static_cast<size_t>(p * s.size());
        idx = std::min(idx, s.size() - 1);
        return s[idx];
    }

    double min() const { return *std::min_element(latencies_ms.begin(), latencies_ms.end()); }
    double max() const { return *std::max_element(latencies_ms.begin(), latencies_ms.end()); }
};

// Simulate the per-token timing loop that a practitioner would instrument.
// In real llama.cpp code this wraps llama_decode() calls.
static PerTokenMetrics simulate_decode_timing(int n_tokens, unsigned seed = 42) {
    /*
     * Real instrumentation pattern (shown as comments):
     *
     * std::vector<double> latencies;
     * for (int i = 0; i < n_predict; i++) {
     *     auto t0 = std::chrono::high_resolution_clock::now();
     *
     *     llama_decode(ctx, batch);                     // ← actual decode
     *
     *     auto t1 = std::chrono::high_resolution_clock::now();
     *     double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
     *     latencies.push_back(ms);
     *
     *     // Sample next token, update batch...
     * }
     */

    // Simulated version using std::this_thread::sleep_for equivalent via chrono
    std::mt19937 rng(seed);
    std::normal_distribution<double> base_dist(19.0, 2.0);  // nominal 19 ms
    std::uniform_real_distribution<double> spike_dist(30.0, 80.0);

    PerTokenMetrics m;
    for (int i = 0; i < n_tokens; i++) {
        double lat = std::max(1.0, base_dist(rng));
        // Chunked prefill interrupt every ~32 tokens
        if (i > 0 && i % 32 == 0)
            lat += spike_dist(rng);
        m.latencies_ms.push_back(lat);
    }
    return m;
}

static void print_itl_stats(const PerTokenMetrics& m) {
    double mean   = m.mean();
    double stddev = m.stddev();
    double p50    = m.percentile(0.50);
    double p95    = m.percentile(0.95);
    double p99    = m.percentile(0.99);

    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  Per-Token ITL Statistics  (n=" << m.latencies_ms.size() << " tokens)\n";
    std::cout << repeat('=', 62) << "\n";

    auto stat_row = [](const std::string& lbl, double val) {
        std::cout << "  " << std::left << std::setw(10) << lbl
                  << std::right << std::setw(8) << std::fixed
                  << std::setprecision(2) << val << " ms\n";
    };
    stat_row("Mean",   mean);
    stat_row("StdDev", stddev);
    stat_row("Min",    m.min());
    stat_row("P50",    p50);
    stat_row("P95",    p95);
    stat_row("P99",    p99);
    stat_row("Max",    m.max());

    // ASCII histogram
    const std::vector<double> edges  = {0,10,15,20,25,30,40,60,100,1e9};
    const std::vector<std::string> lbls = {"<10","10-15","15-20","20-25",
                                           "25-30","30-40","40-60","60-100",">100"};
    std::vector<int> counts(lbls.size(), 0);
    for (double v : m.latencies_ms) {
        for (size_t i = 0; i + 1 < edges.size(); ++i) {
            if (v >= edges[i] && v < edges[i+1]) { counts[i]++; break; }
        }
    }
    int max_count = *std::max_element(counts.begin(), counts.end());
    int bar_w     = 30;

    std::cout << "\n  ITL Distribution (ms):\n";
    for (size_t i = 0; i < lbls.size(); ++i) {
        int bar_len = max_count > 0 ? counts[i] * bar_w / max_count : 0;
        double pct  = counts[i] * 100.0 / m.latencies_ms.size();
        std::cout << "  " << std::right << std::setw(8) << lbls[i]
                  << "  " << std::left  << std::setw(bar_w)
                  << std::string(static_cast<size_t>(bar_len), '#')
                  << "  " << std::right << std::setw(5) << counts[i]
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";
    }

    // Spike detection
    int spikes = 0;
    double max_spike = 0;
    double threshold = mean + 3 * stddev;
    for (double v : m.latencies_ms) {
        if (v > threshold) { spikes++; max_spike = std::max(max_spike, v); }
    }
    if (spikes > 0) {
        std::cout << "\n  Spikes (>mean+3s): " << spikes << " events, "
                  << "max=" << fmt_f(max_spike, 1) << "ms\n";
        std::cout << "  -> Likely: chunked prefill interrupt or NCCL stall\n";
    }
    std::cout << repeat('=', 62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. PROMETHEUS METRIC EMITTER
// ─────────────────────────────────────────────────────────────────────────────

struct VLLMSnapshot {
    int    running          = 0;
    int    waiting          = 0;
    int    swapped          = 0;
    long   preemptions      = 0;
    double gpu_cache_perc   = 0;
    double prefix_hit_rate  = 0;
    long   prompt_tokens    = 0;
    long   gen_tokens       = 0;
    double ttft_mean_s      = 0;
    double itl_mean_s       = 0;
};

static void emit_prometheus(const VLLMSnapshot& snap) {
    /*
     * This shows the Prometheus exposition format that a llama-server wrapper
     * would emit at GET /metrics to make llama.cpp observable via Prometheus.
     *
     * Real implementation would use a proper Prometheus client library
     * (e.g., prometheus-cpp) but the format is just HTTP plain text.
     */
    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  Prometheus /metrics Output (simulated)\n";
    std::cout << repeat('=', 62) << "\n";

    auto gauge = [](const std::string& name, const std::string& help, double v) {
        std::cout << "# HELP " << name << " " << help << "\n";
        std::cout << "# TYPE " << name << " gauge\n";
        std::cout << name << " " << std::fixed << std::setprecision(3) << v << "\n";
    };
    auto counter = [](const std::string& name, const std::string& help, long v) {
        std::cout << "# HELP " << name << " " << help << "\n";
        std::cout << "# TYPE " << name << " counter\n";
        std::cout << name << " " << v << "\n";
    };

    gauge("vllm:num_requests_running",     "Requests running on GPU",    snap.running);
    gauge("vllm:num_requests_waiting",     "Requests waiting",           snap.waiting);
    gauge("vllm:num_requests_swapped",     "Requests swapped to CPU",    snap.swapped);
    counter("vllm:num_preemptions_total",  "Total preemptions",          snap.preemptions);
    gauge("vllm:gpu_cache_usage_perc",     "GPU KV cache fill fraction", snap.gpu_cache_perc);
    gauge("vllm:gpu_prefix_cache_hit_rate","Prefix cache hit rate",      snap.prefix_hit_rate);
    counter("vllm:prompt_tokens_total",    "Total prompt tokens",        snap.prompt_tokens);
    counter("vllm:generation_tokens_total","Total generation tokens",    snap.gen_tokens);
    gauge("vllm:ttft_mean_seconds",        "Mean TTFT (s)",              snap.ttft_mean_s);
    gauge("vllm:itl_mean_seconds",         "Mean ITL (s)",               snap.itl_mean_s);

    std::cout << repeat('=', 62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. KV utilization TREND analyzeR
// ─────────────────────────────────────────────────────────────────────────────

struct KVSample {
    double time_s;
    double fill;
    int    queue;
};

static std::vector<KVSample> simulate_kv_series(int n = 30, unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0, 0.005);

    std::vector<KVSample> out;
    double fill = 0.55;
    int queue   = 0;

    for (int t = 0; t < n; t++) {
        double spike = (t >= 15 && t <= 22) ? 0.02 : 0.0;
        fill = std::clamp(fill + noise(rng) + spike - 0.003, 0.3, 0.999);
        if (fill > 0.93)
            queue = std::min(queue + (int)(rng() % 8 + 3), 60);
        else
            queue = std::max(0, queue - (int)(rng() % 4));
        out.push_back({t * 10.0, fill, queue});
    }
    return out;
}

static void print_kv_series(const std::vector<KVSample>& series) {
    const int W = 36;
    std::cout << "\n" << repeat('=', 66) << "\n";
    std::cout << "  KV Cache utilization Trend\n";
    std::cout << repeat('=', 66) << "\n";
    std::cout << "  " << std::right
              << std::setw(6) << "Time"
              << std::setw(9) << "KV fill"
              << std::setw(7) << "Queue"
              << "  " << std::left << std::setw(W) << "Bar"
              << "  Status\n";
    std::cout << "  " << repeat('-', 64) << "\n";

    for (auto& s : series) {
        int filled = static_cast<int>(s.fill * W);
        std::string bar(static_cast<size_t>(filled), '#');
        bar += std::string(static_cast<size_t>(W - filled), '.');

        const char* status;
        if (s.fill > 0.95)      status = "CRIT";
        else if (s.fill > 0.85) status = "WARN";
        else                    status = "OK  ";

        std::cout << "  " << std::right
                  << std::setw(5) << static_cast<int>(s.time_s) << "s"
                  << std::setw(8) << std::fixed << std::setprecision(1)
                  << s.fill * 100 << "%"
                  << std::setw(7) << s.queue
                  << "  " << std::left << std::setw(W) << bar
                  << "  " << status << "\n";
    }
    std::cout << repeat('=', 66) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. ALERT RULE EVALUATOR
// ─────────────────────────────────────────────────────────────────────────────

struct AlertState {
    std::string name;
    double      threshold;
    bool        is_crit;
    int         for_steps;
    int         _count    = 0;
    bool        firing    = false;
};

static void evaluate_alerts(const std::vector<KVSample>& series) {
    AlertState kv_crit  {"KVCacheSaturated",     0.95, true,  2};
    AlertState kv_warn  {"KVHighutilization",    0.85, false, 3};
    AlertState q_warn   {"RequestsAccumulating", 20.0, false, 1};

    std::cout << "\n" << repeat('=', 62) << "\n";
    std::cout << "  Alert Rule Evaluator\n";
    std::cout << repeat('=', 62) << "\n";

    for (auto& s : series) {
        bool any_event = false;

        auto check = [&](AlertState& rule, double val) {
            bool triggered = val >= rule.threshold;
            if (triggered) rule._count++;
            else           rule._count = 0;

            if (rule._count >= rule.for_steps && !rule.firing) {
                rule.firing = true;
                std::cout << "  t=" << std::setw(5) << static_cast<int>(s.time_s) << "s  "
                          << "[" << (rule.is_crit ? "CRITICAL" : "WARNING ") << "]  "
                          << "FIRING: " << rule.name << "\n";
                any_event = true;
            } else if (!triggered && rule.firing) {
                rule.firing = false;
                std::cout << "  t=" << std::setw(5) << static_cast<int>(s.time_s) << "s  "
                          << "[RESOLVED]  " << rule.name << "\n";
                any_event = true;
            }
        };

        check(kv_crit, s.fill);
        check(kv_warn, s.fill);
        check(q_warn,  static_cast<double>(s.queue));
    }
    std::cout << repeat('=', 62) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n" << repeat('#', 62) << "\n";
    std::cout << "  Chapter 16 - Observability: C++ Companion Demo\n";
    std::cout << repeat('#', 62) << "\n";

    // Section 1: Parse and display llama_perf output
    const std::string PERF_OUTPUT = R"(
llama_perf_context_print:        load time =     423.12 ms
llama_perf_context_print:      sample time =      12.43 ms /   256 runs
llama_perf_context_print: prompt eval time =    1204.12 ms /   512 tokens (    2.35 ms per token,   425.23 tokens per second)
llama_perf_context_print:        eval time =    4892.34 ms /   256 runs (   19.11 ms per token,    52.33 tokens per second)
llama_perf_context_print:       total time =    6120.01 ms /   768 tokens
)";
    auto perf = parse_llama_perf(PERF_OUTPUT);
    print_llama_perf(perf);

    // Section 2: Per-token ITL timing harness
    auto metrics = simulate_decode_timing(256, 42);
    print_itl_stats(metrics);

    // Section 3: Prometheus metric emitter
    VLLMSnapshot snap;
    snap.running         = 45;
    snap.waiting         = 12;
    snap.swapped         = 3;
    snap.preemptions     = 7;
    snap.gpu_cache_perc  = 0.87;
    snap.prefix_hit_rate = 0.73;
    snap.prompt_tokens   = 4892301;
    snap.gen_tokens      = 1234891;
    snap.ttft_mean_s     = 0.346;
    snap.itl_mean_s      = 0.0119;
    emit_prometheus(snap);

    // Section 4: KV utilization trend
    auto kv_series = simulate_kv_series(30, 7);
    print_kv_series(kv_series);

    // Section 5: Alert evaluation
    evaluate_alerts(kv_series);

    return 0;
}

```

