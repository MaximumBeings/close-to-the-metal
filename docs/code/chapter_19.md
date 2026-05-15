# Chapter 19: Kubernetes and KubeRay — Companion Code

## Python — `kubernetes_demo.py`

```python
"""
kubernetes_demo.py — Chapter 19: Kubernetes, KubeRay, and llm.d

Demonstrates:
  1. KubeRay RayService manifest generator (vLLM on Kubernetes)
  2. llm.d LLMDeployment manifest generator
  3. HPA custom metrics configuration generator
  4. Prometheus ServiceMonitor + PrometheusRule generator
  5. Disaggregated cluster manifest (prefill + decode pools)
  6. Cluster cost estimator (GPU hours vs. autoscale config)
  7. Readiness probe health-check client simulation
  8. Graceful drain planner

No external dependencies beyond the Python standard library.
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# 1. KubeRay RayService manifest generator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VLLMWorkerConfig:
    model_id: str
    max_model_len: int          = 32768
    max_num_seqs: int           = 64
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int   = 1
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    port: int                   = 8000


@dataclass
class KubeRayConfig:
    cluster_name: str
    namespace: str              = "inference"
    ray_version: str            = "2.35.0"
    min_replicas: int           = 1
    max_replicas: int           = 8
    gpu_class: str              = "h100-sxm"
    gpu_per_worker: int         = 1
    worker_cpu: str             = "8"
    worker_memory: str          = "80Gi"
    target_requests_per_replica: int = 10
    downscale_delay_s: int      = 300


def generate_rayservice_manifest(worker: VLLMWorkerConfig,
                                  kuberay: KubeRayConfig) -> str:
    cold_start_estimate = 120 if "8b" in worker.model_id.lower() else 240

    return f"""apiVersion: ray.io/v1
kind: RayService
metadata:
  name: {kuberay.cluster_name}
  namespace: {kuberay.namespace}
spec:
  serveConfigV2: |
    applications:
      - name: vllm-app
        route_prefix: /
        import_path: vllm_serve:deployment
        deployments:
          - name: VLLMDeployment
            num_replicas: {kuberay.min_replicas}
            ray_actor_options:
              num_gpus: {kuberay.gpu_per_worker}
            autoscaling_config:
              min_replicas: {kuberay.min_replicas}
              max_replicas: {kuberay.max_replicas}
              target_num_ongoing_requests_per_replica: {kuberay.target_requests_per_replica}
              upscale_delay_s: 30
              downscale_delay_s: {kuberay.downscale_delay_s}
  upgradeStrategy:
    type: NewCluster
    newClusterAdditionalWaitTime: 60s
  rayClusterConfig:
    rayVersion: "{kuberay.ray_version}"
    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray-ml:{kuberay.ray_version}-gpu
              resources:
                requests: {{cpu: "4", memory: "16Gi"}}
                limits:   {{cpu: "4", memory: "16Gi"}}
    workerGroupSpecs:
      - groupName: vllm-workers
        replicas: {kuberay.min_replicas}
        minReplicas: {kuberay.min_replicas}
        maxReplicas: {kuberay.max_replicas}
        rayStartParams: {{}}
        template:
          spec:
            tolerations:
              - key: nvidia.com/gpu
                operator: Exists
            containers:
              - name: vllm-worker
                image: vllm/vllm-openai:latest
                command:
                  - python
                  - -m
                  - vllm.entrypoints.openai.api_server
                  - --model={worker.model_id}
                  - --max-model-len={worker.max_model_len}
                  - --max-num-seqs={worker.max_num_seqs}
                  - --gpu-memory-utilization={worker.gpu_memory_utilization}
                  - --tensor-parallel-size={worker.tensor_parallel_size}
                  {"- --enable-prefix-caching" if worker.enable_prefix_caching else ""}
                  {"- --enable-chunked-prefill" if worker.enable_chunked_prefill else ""}
                  - --port={worker.port}
                resources:
                  requests:
                    cpu: "{kuberay.worker_cpu}"
                    memory: "{kuberay.worker_memory}"
                    nvidia.com/gpu: "{kuberay.gpu_per_worker}"
                  limits:
                    cpu: "{kuberay.worker_cpu}"
                    memory: "{kuberay.worker_memory}"
                    nvidia.com/gpu: "{kuberay.gpu_per_worker}"
                readinessProbe:
                  httpGet:
                    path: /health
                    port: {worker.port}
                  initialDelaySeconds: {cold_start_estimate}
                  periodSeconds: 10
                  failureThreshold: 6
                terminationGracePeriodSeconds: 120
"""


def print_rayservice_manifest():
    worker = VLLMWorkerConfig(
        model_id="meta-llama/Llama-3-8B-Instruct",
        max_model_len=32768,
        max_num_seqs=64,
    )
    kuberay = KubeRayConfig(
        cluster_name="vllm-llama3-8b",
        namespace="inference",
        min_replicas=2,
        max_replicas=8,
    )
    print("\n" + "=" * 64)
    print("  KubeRay RayService Manifest")
    print("=" * 64)
    print(generate_rayservice_manifest(worker, kuberay))


# ──────────────────────────────────────────────────────────────────────────────
# 2. llm.d LLMDeployment manifest generator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMDConfig:
    deployment_name: str
    namespace: str              = "inference"
    model_id: str               = "meta-llama/Llama-3-8B-Instruct"
    model_format: str           = "safetensors"
    hardware_class: str         = "h100-sxm"
    gpu_count: int              = 1
    engine: str                 = "vllm"
    engine_version: str         = "0.6.0"
    max_num_seqs: int           = 64
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    min_replicas: int           = 1
    max_replicas: int           = 8
    scale_metric: str           = "queue_depth"
    scale_target: int           = 5
    ttft_p99_ms: int            = 500
    itl_p99_ms: int             = 50
    disaggregation: bool        = False
    prefill_class: str          = "h100-sxm"
    decode_class: str           = "h100-nvl"
    prefill_replicas: int       = 2
    decode_replicas: int        = 8


def generate_llmd_manifest(cfg: LLMDConfig) -> str:
    disagg_block = ""
    if cfg.disaggregation:
        disagg_block = f"""
  disaggregation:
    enabled: true
    prefillClass: {cfg.prefill_class}
    decodeClass:  {cfg.decode_class}
    prefillReplicas: {cfg.prefill_replicas}
    decodeReplicas: {cfg.decode_replicas}"""

    return f"""apiVersion: inference.llm.d/v1alpha1
kind: LLMDeployment
metadata:
  name: {cfg.deployment_name}
  namespace: {cfg.namespace}
spec:
  model:
    name: {cfg.model_id}
    revision: main
    format: {cfg.model_format}

  hardware:
    class: {cfg.hardware_class}
    gpuCount: {cfg.gpu_count}

  engine:
    type: {cfg.engine}
    version: "{cfg.engine_version}"
    config:
      max_num_seqs: {cfg.max_num_seqs}
      gpu_memory_utilization: {cfg.gpu_memory_utilization}
      enable_prefix_caching: {"true" if cfg.enable_prefix_caching else "false"}
      enable_chunked_prefill: {"true" if cfg.enable_chunked_prefill else "false"}

  scaling:
    minReplicas: {cfg.min_replicas}
    maxReplicas: {cfg.max_replicas}
    metric: {cfg.scale_metric}
    targetValue: "{cfg.scale_target}"{disagg_block}

  sla:
    ttft_p99_ms: {cfg.ttft_p99_ms}
    itl_p99_ms:  {cfg.itl_p99_ms}
"""


def print_llmd_manifests():
    print("\n" + "=" * 64)
    print("  llm.d LLMDeployment Manifest (standard)")
    print("=" * 64)
    cfg = LLMDConfig(
        deployment_name="llama3-8b-standard",
        model_id="meta-llama/Llama-3-8B-Instruct",
    )
    print(generate_llmd_manifest(cfg))

    print("\n" + "=" * 64)
    print("  llm.d LLMDeployment Manifest (disaggregated)")
    print("=" * 64)
    cfg_disagg = LLMDConfig(
        deployment_name="llama3-8b-disagg",
        model_id="meta-llama/Llama-3-8B-Instruct",
        disaggregation=True,
        prefill_replicas=2,
        decode_replicas=8,
        ttft_p99_ms=300,
        itl_p99_ms=20,
    )
    print(generate_llmd_manifest(cfg_disagg))


# ──────────────────────────────────────────────────────────────────────────────
# 3. HPA + Prometheus adapter config generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_hpa_manifest(deployment_name: str,
                           namespace: str,
                           min_replicas: int,
                           max_replicas: int,
                           queue_target: int) -> str:
    return f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {deployment_name}-hpa
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {deployment_name}
  minReplicas: {min_replicas}
  maxReplicas: {max_replicas}
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_requests_waiting
        target:
          type: AverageValue
          averageValue: "{queue_target}"
    - type: Pods
      pods:
        metric:
          name: vllm_gpu_cache_usage_perc
        target:
          type: AverageValue
          averageValue: "0.80"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
"""


def print_hpa_config():
    print("\n" + "=" * 64)
    print("  Kubernetes HPA Manifest (custom vLLM metrics)")
    print("=" * 64)
    print(generate_hpa_manifest(
        deployment_name="vllm-llama3-8b",
        namespace="inference",
        min_replicas=1,
        max_replicas=8,
        queue_target=5,
    ))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Prometheus ServiceMonitor + alerting rules
# ──────────────────────────────────────────────────────────────────────────────

def generate_service_monitor(app_label: str, namespace: str) -> str:
    return f"""apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {app_label}-metrics
  namespace: {namespace}
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: {app_label}
  endpoints:
    - port: metrics
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s
"""


def generate_prometheus_rules(namespace: str) -> str:
    return f"""apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: vllm-alerts
  namespace: {namespace}
spec:
  groups:
    - name: vllm.kv_cache
      interval: 30s
      rules:
        - alert: KVCacheSaturation
          expr: vllm:gpu_cache_usage_perc > 0.95
          for: 2m
          labels:
            severity: critical
          annotations:
            summary: "KV cache > 95% on {{ $labels.pod }}"
            description: "OOM risk — KV at {{ $value | humanizePercentage }}"

        - alert: HighWaitingRequests
          expr: vllm:num_requests_waiting > 20
          for: 1m
          labels:
            severity: warning
          annotations:
            summary: "Queue depth > 20 on {{ $labels.pod }}"
            description: "Scale-out may be needed"

        - alert: TTFTRegression
          expr: |
            histogram_quantile(0.99,
              rate(vllm:time_to_first_token_seconds_bucket[5m])) > 0.5
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "TTFT p99 > 500ms"
            description: "p99 TTFT is {{ $value }}s — SLA breach risk"
"""


def print_monitoring_config():
    print("\n" + "=" * 64)
    print("  Prometheus ServiceMonitor + Alerting Rules")
    print("=" * 64)
    print(generate_service_monitor("vllm-worker", "inference"))
    print(generate_prometheus_rules("inference"))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Cluster cost estimator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUPricing:
    name: str
    on_demand_per_hr: float
    spot_discount: float      # fraction, e.g. 0.70 = 70% discount
    reserved_discount: float  # fraction, e.g. 0.40 = 40% discount


GPU_PRICING = {
    "h100-sxm-aws":   GPUPricing("H100 SXM (AWS p5)",    32.77, 0.70, 0.38),
    "h100-nvl-aws":   GPUPricing("H100 NVL (AWS p5e)",   24.00, 0.65, 0.35),
    "a100-40-aws":    GPUPricing("A100 40GB (AWS p4d)",   12.00, 0.60, 0.32),
    "a100-80-lambda": GPUPricing("A100 80GB (Lambda)",    2.49,  0.00, 0.00),
    "rtx4090-vast":   GPUPricing("RTX 4090 (Vast.ai)",   0.50,  0.00, 0.00),
}


def print_cluster_cost_estimate():
    print("\n" + "=" * 72)
    print("  Cluster Cost Estimator  |  50K user RAG deployment")
    print("=" * 72)

    configs = [
        # (label, prefill gpus, decode gpus, gpu_key, hours/day)
        ("Baseline (all H100 SXM, co-located)", 0, 20, "h100-sxm-aws", 24),
        ("Disaggregated (4 pf SXM + 16 dec NVL)", 4, 16, None, 24),
        ("With autoscale (avg 50% utilization)",  2,  8, None, 24),
    ]

    print(f"  {'Config':<42} {'GPUs':>5} {'$/hr':>8} {'$/day':>10} {'$/mo':>12}")
    print("  " + "-" * 41 + "  " + "-" * 4 + "  " + "-" * 7 +
          "  " + "-" * 9 + "  " + "-" * 11)

    for label, pf, dec, gpu_key, hours in configs:
        if gpu_key:
            g = GPU_PRICING[gpu_key]
            cost_hr = (pf + dec) * g.on_demand_per_hr
        else:
            pf_price = GPU_PRICING["h100-sxm-aws"].on_demand_per_hr
            dec_price = GPU_PRICING["h100-nvl-aws"].on_demand_per_hr
            cost_hr = pf * pf_price + dec * dec_price
        cost_day = cost_hr * hours
        cost_mo  = cost_day * 30
        total_gpus = pf + dec

        print(f"  {label:<42} {total_gpus:>5} ${cost_hr:>7.2f} ${cost_day:>9.2f} ${cost_mo:>10.2f}")

    print("=" * 72)
    print("  Spot pricing (for non-critical workloads):")
    g = GPU_PRICING["h100-sxm-aws"]
    spot_mo = 20 * g.on_demand_per_hr * (1 - g.spot_discount) * 24 * 30
    print(f"    20× H100 SXM spot: ${spot_mo:,.0f}/month"
          f"  (vs ${20 * g.on_demand_per_hr * 24 * 30:,.0f} on-demand)")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Readiness probe health-check simulation
# ──────────────────────────────────────────────────────────────────────────────

def simulate_pod_startup(model_size_b: float,
                          gpu_bandwidth_tbps: float = 3.35,
                          initial_delay_s: int = 120) -> None:
    """
    Simulate vLLM pod startup sequence and probe timing.
    model_size_b: model parameters in billions
    """
    # Weight load time: weights in BF16
    weight_bytes = model_size_b * 1e9 * 2
    weight_load_s = weight_bytes / (gpu_bandwidth_tbps * 1e12)

    # CUDA init, graph capture, warmup
    cuda_init_s  = 8.0
    warmup_s     = 15.0
    total_cold_s = cuda_init_s + weight_load_s + warmup_s

    print(f"\n  {'=' * 58}")
    print(f"  Pod Startup Simulation  |  {model_size_b:.0f}B model  |  H100")
    print(f"  {'=' * 58}")
    print(f"  CUDA init + driver:      {cuda_init_s:6.1f} s")
    print(f"  Weight load ({weight_bytes/1e9:.0f} GB BF16): {weight_load_s:6.1f} s"
          f"  ({gpu_bandwidth_tbps} TB/s)")
    print(f"  Graph capture + warmup:  {warmup_s:6.1f} s")
    print(f"  ─────────────────────────────────────")
    print(f"  Total cold start:        {total_cold_s:6.1f} s")
    print(f"  initialDelaySeconds rec: {int(total_cold_s * 1.25):6d} s  (+25% margin)")
    print()

    # Probe timeline
    probe_interval = 10
    print(f"  Probe timeline (initialDelaySeconds={initial_delay_s}, "
          f"interval={probe_interval}s):")

    probes_before_ready = initial_delay_s // probe_interval
    first_ready_probe   = int(total_cold_s / probe_interval) + 1

    for probe_n in range(1, min(probes_before_ready + 3, 20)):
        probe_t = probe_n * probe_interval
        # Pod not ready during initialDelay
        if probe_t <= initial_delay_s:
            status = "  (waiting — initialDelaySeconds not elapsed)"
        elif probe_t < total_cold_s:
            status = "  NOT READY — model still loading"
        else:
            status = "  READY ✓"
        print(f"    t={probe_t:4d}s: probe #{probe_n}{status}")

    if initial_delay_s < total_cold_s * 0.9:
        print(f"\n  WARNING: initialDelaySeconds={initial_delay_s} < "
              f"cold_start={total_cold_s:.0f}s")
        print(f"           Probes will fire during loading → possible restart loop")
    else:
        print(f"\n  OK: initialDelaySeconds adequately covers cold start.")
    print(f"  {'=' * 58}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Graceful drain planner
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DrainRequest:
    request_id: str
    prompt_tokens: int
    expected_output_tokens: int
    elapsed_output_tokens: int  # how far along at SIGTERM


def simulate_graceful_drain(requests: List[DrainRequest],
                             decode_tps: float = 209.0,
                             grace_period_s: int = 120) -> None:
    print(f"\n  {'=' * 62}")
    print(f"  Graceful Drain Simulation  "
          f"(terminationGracePeriodSeconds={grace_period_s})")
    print(f"  {'=' * 62}")
    print(f"  In-flight requests at SIGTERM: {len(requests)}\n")

    total_time = 0.0
    all_complete = True

    for req in requests:
        remaining_tokens = req.expected_output_tokens - req.elapsed_output_tokens
        time_needed_s = remaining_tokens / decode_tps
        total_time = max(total_time, time_needed_s)
        status = "OK" if time_needed_s <= grace_period_s else "WILL BE KILLED"
        print(f"  {req.request_id:<14} remaining={remaining_tokens:>6} tok"
              f"  needs={time_needed_s:>6.1f}s  [{status}]")
        if time_needed_s > grace_period_s:
            all_complete = False

    print(f"\n  Max drain time needed: {total_time:.1f}s")
    if all_complete:
        print(f"  OK: all requests complete within grace period ({grace_period_s}s)")
    else:
        print(f"  WARNING: some requests will be killed mid-generation")
        print(f"  Recommendation: increase terminationGracePeriodSeconds to "
              f"{int(total_time * 1.1)}")
    print(f"  {'=' * 62}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("█" * 64)
    print("  Chapter 19 — Kubernetes, KubeRay, and llm.d")
    print("█" * 64)

    print_rayservice_manifest()
    print_llmd_manifests()
    print_hpa_config()
    print_monitoring_config()
    print_cluster_cost_estimate()

    # Startup simulations
    simulate_pod_startup(model_size_b=8,  initial_delay_s=120)
    simulate_pod_startup(model_size_b=70, initial_delay_s=180)

    # Graceful drain with mixed request types
    rng = random.Random(42)
    in_flight = [
        DrainRequest("req-short-001",  200,  150, rng.randint(10, 140)),
        DrainRequest("req-short-002",  350,  200, rng.randint(50, 190)),
        DrainRequest("req-rag-001",   4096, 1024, rng.randint(100, 800)),
        DrainRequest("req-reason-001", 512, 8192, rng.randint(500, 2000)),
    ]
    simulate_graceful_drain(in_flight, grace_period_s=120)

    # Show that reasoning requests need longer grace period
    reasoning_requests = [
        DrainRequest("reason-A", 512, 32768, rng.randint(5000, 15000)),
        DrainRequest("reason-B", 512, 50000, rng.randint(1000, 5000)),
    ]
    simulate_graceful_drain(reasoning_requests, grace_period_s=120)
    simulate_graceful_drain(reasoning_requests, grace_period_s=600)


if __name__ == "__main__":
    main()

```

## C++ — `kubernetes_demo.cpp`

```cpp
// kubernetes_demo.cpp — Chapter 19: Kubernetes, KubeRay, and llm.d
//
// Compile:  g++ -std=c++17 -O2 -Wall -o kubernetes_demo kubernetes_demo.cpp
// Run:      ./kubernetes_demo
//
// Covers: manifest generation, readiness probe simulation, cluster cost
//         estimation, graceful drain planner, HPA threshold calculator.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

static std::string repeat_str(const std::string& s, int n) {
    std::string out;
    for (int i = 0; i < n; ++i) out += s;
    return out;
}

static std::string fmt(double v, int prec = 1) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

static std::string fmt_int(long long v) {
    std::string s = std::to_string(std::llabs(v));
    int n = static_cast<int>(s.size());
    std::string out;
    for (int i = 0; i < n; ++i) {
        if (i && (n - i) % 3 == 0) out += ',';
        out += s[static_cast<size_t>(i)];
    }
    return out;
}

static void print_sep(int w = 62, char c = '=') {
    std::cout << repeat_str(std::string(1, c), w) << "\n";
}

static void print_header(const std::string& title, int w = 62) {
    print_sep(w);
    int pad = std::max(0, (w - static_cast<int>(title.size()) - 2) / 2);
    std::cout << repeat_str(" ", pad) << " " << title << "\n";
    print_sep(w);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. RayService manifest generator (simplified YAML output)
// ─────────────────────────────────────────────────────────────────────────────

struct WorkerConfig {
    std::string model_id       = "meta-llama/Llama-3-8B-Instruct";
    int         max_model_len  = 32768;
    int         max_num_seqs   = 64;
    double      gpu_mem_util   = 0.90;
    int         tp_size        = 1;
    bool        prefix_caching = true;
    bool        chunked_prefill= true;
    int         port           = 8000;
};

struct KubeRayConfig {
    std::string cluster_name    = "vllm-cluster";
    std::string ns              = "inference";
    std::string ray_version     = "2.35.0";
    int         min_replicas    = 1;
    int         max_replicas    = 8;
    int         gpu_per_worker  = 1;
    std::string worker_cpu      = "8";
    std::string worker_mem      = "80Gi";
    int         target_reqs_per = 10;
    int         downscale_s     = 300;
};

static void print_rayservice_manifest(const WorkerConfig& w,
                                       const KubeRayConfig& k) {
    int cold_start = (w.model_id.find("70b") != std::string::npos ||
                      w.model_id.find("70B") != std::string::npos) ? 240 : 120;

    std::cout << "\n";
    print_sep(64);
    std::cout << "  KubeRay RayService Manifest\n";
    print_sep(64);
    std::cout << "\napiVersion: ray.io/v1\n"
              << "kind: RayService\n"
              << "metadata:\n"
              << "  name: " << k.cluster_name << "\n"
              << "  namespace: " << k.ns << "\n"
              << "spec:\n"
              << "  serveConfigV2: |\n"
              << "    applications:\n"
              << "      - name: vllm-app\n"
              << "        deployments:\n"
              << "          - name: VLLMDeployment\n"
              << "            num_replicas: " << k.min_replicas << "\n"
              << "            ray_actor_options:\n"
              << "              num_gpus: " << k.gpu_per_worker << "\n"
              << "            autoscaling_config:\n"
              << "              min_replicas: " << k.min_replicas << "\n"
              << "              max_replicas: " << k.max_replicas << "\n"
              << "              target_num_ongoing_requests_per_replica: "
              << k.target_reqs_per << "\n"
              << "              downscale_delay_s: " << k.downscale_s << "\n"
              << "  upgradeStrategy:\n"
              << "    type: NewCluster\n"
              << "    newClusterAdditionalWaitTime: 60s\n"
              << "  rayClusterConfig:\n"
              << "    rayVersion: \"" << k.ray_version << "\"\n"
              << "    workerGroupSpecs:\n"
              << "      - groupName: vllm-workers\n"
              << "        minReplicas: " << k.min_replicas << "\n"
              << "        maxReplicas: " << k.max_replicas << "\n"
              << "        template:\n"
              << "          spec:\n"
              << "            containers:\n"
              << "              - name: vllm-worker\n"
              << "                image: vllm/vllm-openai:latest\n"
              << "                command:\n"
              << "                  - python\n"
              << "                  - -m\n"
              << "                  - vllm.entrypoints.openai.api_server\n"
              << "                  - --model=" << w.model_id << "\n"
              << "                  - --max-model-len=" << w.max_model_len << "\n"
              << "                  - --max-num-seqs=" << w.max_num_seqs << "\n";
    if (w.prefix_caching)  std::cout << "                  - --enable-prefix-caching\n";
    if (w.chunked_prefill) std::cout << "                  - --enable-chunked-prefill\n";
    std::cout << "                resources:\n"
              << "                  limits:\n"
              << "                    nvidia.com/gpu: \""
              << k.gpu_per_worker << "\"\n"
              << "                readinessProbe:\n"
              << "                  httpGet: {path: /health, port: "
              << w.port << "}\n"
              << "                  initialDelaySeconds: " << cold_start << "\n"
              << "                  periodSeconds: 10\n"
              << "                terminationGracePeriodSeconds: 120\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. llm.d LLMDeployment manifest
// ─────────────────────────────────────────────────────────────────────────────

struct LLMDConfig {
    std::string name              = "llama3-8b";
    std::string ns                = "inference";
    std::string model_id          = "meta-llama/Llama-3-8B-Instruct";
    std::string hw_class          = "h100-sxm";
    int         gpu_count         = 1;
    std::string engine            = "vllm";
    int         max_num_seqs      = 64;
    double      gpu_mem_util      = 0.90;
    int         min_replicas      = 1;
    int         max_replicas      = 8;
    int         scale_target      = 5;
    int         ttft_p99_ms       = 500;
    int         itl_p99_ms        = 50;
    bool        disaggregation    = false;
    std::string prefill_class     = "h100-sxm";
    std::string decode_class      = "h100-nvl";
    int         prefill_replicas  = 2;
    int         decode_replicas   = 8;
};

static void print_llmd_manifest(const LLMDConfig& c) {
    std::cout << "\napiVersion: inference.llm.d/v1alpha1\n"
              << "kind: LLMDeployment\n"
              << "metadata:\n"
              << "  name: " << c.name << "\n"
              << "  namespace: " << c.ns << "\n"
              << "spec:\n"
              << "  model:\n"
              << "    name: " << c.model_id << "\n"
              << "  hardware:\n"
              << "    class: " << c.hw_class << "\n"
              << "    gpuCount: " << c.gpu_count << "\n"
              << "  engine:\n"
              << "    type: " << c.engine << "\n"
              << "    config:\n"
              << "      max_num_seqs: " << c.max_num_seqs << "\n"
              << "      gpu_memory_utilization: " << fmt(c.gpu_mem_util, 2) << "\n"
              << "  scaling:\n"
              << "    minReplicas: " << c.min_replicas << "\n"
              << "    maxReplicas: " << c.max_replicas << "\n"
              << "    metric: queue_depth\n"
              << "    targetValue: \"" << c.scale_target << "\"\n";

    if (c.disaggregation) {
        std::cout << "  disaggregation:\n"
                  << "    enabled: true\n"
                  << "    prefillClass: " << c.prefill_class << "\n"
                  << "    decodeClass: "  << c.decode_class  << "\n"
                  << "    prefillReplicas: " << c.prefill_replicas << "\n"
                  << "    decodeReplicas: "  << c.decode_replicas  << "\n";
    }

    std::cout << "  sla:\n"
              << "    ttft_p99_ms: " << c.ttft_p99_ms << "\n"
              << "    itl_p99_ms:  " << c.itl_p99_ms  << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Readiness probe startup simulator
// ─────────────────────────────────────────────────────────────────────────────

static void simulate_pod_startup(double model_params_b,
                                  double gpu_bw_tbps = 3.35,
                                  int    initial_delay_s = 120) {
    double weight_bytes  = model_params_b * 1e9 * 2.0;  // BF16
    double weight_load_s = weight_bytes / (gpu_bw_tbps * 1e12);
    double cuda_init_s   = 8.0;
    double warmup_s      = 15.0;
    double cold_start_s  = cuda_init_s + weight_load_s + warmup_s;
    int    rec_delay     = static_cast<int>(cold_start_s * 1.25);

    std::cout << "\n";
    print_sep(60);
    std::cout << "  Pod Startup Simulation  |  "
              << fmt(model_params_b, 0) << "B model  |  H100\n";
    print_sep(60);
    std::cout << "  CUDA init + driver:       " << fmt(cuda_init_s,  1) << " s\n";
    std::cout << "  Weight load (" << fmt(weight_bytes / 1e9, 0) << " GB BF16): "
              << fmt(weight_load_s, 1) << " s"
              << "  (" << fmt(gpu_bw_tbps, 2) << " TB/s)\n";
    std::cout << "  Graph capture + warmup:   " << fmt(warmup_s, 1) << " s\n";
    std::cout << "  " << repeat_str("-", 38) << "\n";
    std::cout << "  Total cold start:         " << fmt(cold_start_s, 1) << " s\n";
    std::cout << "  initialDelaySeconds rec:  " << rec_delay
              << " s  (+25% margin)\n\n";

    // Show a few probe events
    int probe_interval = 10;
    std::cout << "  Probe timeline (initialDelaySeconds=" << initial_delay_s
              << ", interval=" << probe_interval << "s):\n";

    for (int probe = 1; probe <= 5; ++probe) {
        int t = probe * probe_interval;
        std::string status;
        if (t <= initial_delay_s)
            status = "  (waiting — initialDelaySeconds not elapsed)";
        else if (static_cast<double>(t) < cold_start_s)
            status = "  NOT READY — model loading";
        else
            status = "  READY";

        std::cout << "    t=" << std::setw(4) << t << "s: probe #" << probe
                  << status << "\n";
    }
    std::cout << "    ...\n";

    // First ready probe
    int first_ready = static_cast<int>(cold_start_s / probe_interval + 1) * probe_interval;
    first_ready = std::max(first_ready, initial_delay_s + probe_interval);
    std::cout << "    t=" << std::setw(4) << first_ready << "s: probe -> READY\n\n";

    if (initial_delay_s < static_cast<int>(cold_start_s * 0.9)) {
        std::cout << "  WARNING: initialDelaySeconds too low — restart loop risk\n";
    } else {
        std::cout << "  OK: initialDelaySeconds covers cold start.\n";
    }
    print_sep(60);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Cluster cost estimator
// ─────────────────────────────────────────────────────────────────────────────

struct GPUPricing {
    std::string name;
    double on_demand_hr;
    double spot_discount;
};

static const std::vector<GPUPricing> GPU_PRICES = {
    {"H100 SXM (AWS)",   32.77, 0.70},
    {"H100 NVL (AWS)",   24.00, 0.65},
    {"A100 80GB (Lambda)",2.49, 0.00},
    {"RTX 4090 (Vast)",   0.50, 0.00},
};

static void print_cluster_cost() {
    std::cout << "\n";
    print_sep(74);
    std::cout << "  Cluster Cost Estimator  |  50K user RAG deployment\n";
    print_sep(74);

    struct Config { std::string label; int pf; int dec; int pf_idx; int dec_idx; };
    std::vector<Config> configs = {
        {"All H100 SXM co-located (20 GPUs)",         0, 20, 0, 0},
        {"Disaggregated (4 SXM prefill + 16 NVL dec)",4, 16, 0, 1},
        {"Disaggregated + autoscale avg 50%",          2,  8, 0, 1},
    };

    std::cout << "  " << std::left  << std::setw(44) << "Config"
              << std::right
              << std::setw(6) << "GPUs"
              << std::setw(10) << "$/hr"
              << std::setw(11) << "$/day"
              << std::setw(13) << "$/month"
              << "\n";
    std::cout << "  " << repeat_str("-", 43) << "  "
              << repeat_str("-", 5) << "  "
              << repeat_str("-", 8) << "  "
              << repeat_str("-", 9) << "  "
              << repeat_str("-", 11) << "\n";

    for (auto& c : configs) {
        double cost_hr = c.pf * GPU_PRICES[static_cast<size_t>(c.pf_idx)].on_demand_hr
                       + c.dec* GPU_PRICES[static_cast<size_t>(c.dec_idx)].on_demand_hr;
        int total_gpus = c.pf + c.dec;
        std::cout << "  " << std::left  << std::setw(44) << c.label
                  << std::right
                  << std::setw(6)  << total_gpus
                  << std::setw(9)  << ("$" + fmt(cost_hr, 2))
                  << std::setw(11) << ("$" + fmt(cost_hr * 24, 2))
                  << std::setw(13) << ("$" + fmt_int(static_cast<long long>(cost_hr * 24 * 30)))
                  << "\n";
    }
    print_sep(74);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Graceful drain planner
// ─────────────────────────────────────────────────────────────────────────────

struct DrainRequest {
    std::string id;
    int remaining_tokens;
};

static void simulate_graceful_drain(const std::vector<DrainRequest>& reqs,
                                     double decode_tps,
                                     int grace_period_s) {
    std::cout << "\n";
    print_sep(64);
    std::cout << "  Graceful Drain Simulation"
              << "  (terminationGracePeriodSeconds=" << grace_period_s << ")\n";
    print_sep(64);
    std::cout << "  In-flight at SIGTERM: " << reqs.size() << "\n\n";

    double max_t = 0;
    bool all_ok  = true;

    for (auto& r : reqs) {
        double t = r.remaining_tokens / decode_tps;
        max_t = std::max(max_t, t);
        std::string status = (t <= grace_period_s) ? "OK" : "WILL BE KILLED";
        if (t > grace_period_s) all_ok = false;

        std::cout << "  " << std::left  << std::setw(16) << r.id
                  << std::right
                  << " remaining=" << std::setw(6) << r.remaining_tokens << " tok"
                  << "  needs=" << std::setw(6) << fmt(t, 1) << "s"
                  << "  [" << status << "]\n";
    }

    std::cout << "\n  Max drain time: " << fmt(max_t, 1) << "s\n";
    if (all_ok) {
        std::cout << "  OK: all requests finish within " << grace_period_s << "s\n";
    } else {
        std::cout << "  WARNING: increase terminationGracePeriodSeconds to "
                  << static_cast<int>(max_t * 1.1) << "\n";
    }
    print_sep(64);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    print_header("Chapter 19 - Kubernetes, KubeRay, and llm.d");

    WorkerConfig w;
    w.model_id = "meta-llama/Llama-3-8B-Instruct";

    KubeRayConfig k;
    k.cluster_name = "vllm-llama3-8b";
    k.min_replicas = 2;
    k.max_replicas = 8;
    print_rayservice_manifest(w, k);

    std::cout << "\n";
    print_sep(60);
    std::cout << "  llm.d LLMDeployment Manifest (standard)\n";
    print_sep(60);
    LLMDConfig ld;
    print_llmd_manifest(ld);

    std::cout << "\n";
    print_sep(60);
    std::cout << "  llm.d LLMDeployment Manifest (disaggregated)\n";
    print_sep(60);
    LLMDConfig ld_disagg;
    ld_disagg.name = "llama3-8b-disagg";
    ld_disagg.disaggregation = true;
    ld_disagg.ttft_p99_ms = 300;
    print_llmd_manifest(ld_disagg);

    simulate_pod_startup(8.0,  3.35, 120);
    simulate_pod_startup(70.0, 3.35, 180);

    print_cluster_cost();

    // Graceful drain — mixed standard + reasoning
    std::vector<DrainRequest> reqs = {
        {"req-short-1",    112},
        {"req-rag-1",      643},
        {"req-reason-1",  7191},
    };
    simulate_graceful_drain(reqs, 209.0, 120);

    // Reasoning with long traces — needs longer grace
    std::vector<DrainRequest> long_reqs = {
        {"reason-A", 24111},
        {"reason-B", 48429},
    };
    simulate_graceful_drain(long_reqs, 209.0, 120);
    simulate_graceful_drain(long_reqs, 209.0, 600);

    return 0;
}

```

