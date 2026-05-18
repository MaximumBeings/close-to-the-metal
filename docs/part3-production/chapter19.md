# Chapter 19: Serving in Production — Kubernetes, KubeRay, and llm.d

> "The GPU is five percent of the problem. The other ninety-five percent is keeping it alive,
> scaling it on demand, and routing traffic to the right worker at three in the morning when
> nobody is watching."
>
> — Production ML engineering, paraphrased from every post-mortem ever written

---

## The Gap Between a Working Model and a Production Service

You have a vLLM instance that benchmarks at 1,635 tokens per second. You can restart it with a
shell command. You know it will OOM if more than 128 sequences hit it simultaneously. You have
no idea what happens if the host machine reboots during a request.

This is the gap that Kubernetes, KubeRay, and llm.d are built to close. This chapter covers
the orchestration layer — the part of the stack that sits above the GPU and below the load
balancer, responsible for keeping inference workers alive, scaling them to demand, routing
traffic intelligently, and doing all of it without manual intervention.

---

## 19.1 Why Bare-Metal vLLM Is Not Enough

`[FOUNDATIONAL]`

Running vLLM directly on a server is fine for development. In production, five categories of
failure will hit you:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Production failure modes for bare-metal vLLM                  │
  │                                                                 │
  │  1. Process crash        → no auto-restart, user sees error    │
  │  2. Demand spike         → no scale-out, queue grows, SLA miss │
  │  3. Idle overnight       → no scale-in, you pay for idle GPU   │
  │  4. Model update         → downtime or manual rolling restart  │
  │  5. Multi-tenant traffic → no isolation, noisy neighbors      │
  └─────────────────────────────────────────────────────────────────┘
```

Each of these is solved by the Kubernetes ecosystem:

- **Process crash** → Pod restartPolicy + readiness probe keeps traffic away until ready.
- **Demand spike** → Horizontal Pod Autoscaler (HPA) on custom metrics (queue depth, p99).
- **Idle overnight** → HPA scales to zero (with KEDA) or to minimum replica count.
- **Model update** → RollingUpdate strategy; new pods pass readiness check before old pods die.
- **Multi-tenant** → Namespace isolation, ResourceQuota, PriorityClass.

The question is not whether to use Kubernetes for production LLM serving. It is which
Kubernetes-native abstraction best matches how LLM workloads actually run.

---

## 19.2 Kubernetes Primitives for GPU Serving

`[FOUNDATIONAL]`

Before KubeRay and llm.d, you need a firm grip on the Kubernetes primitives they build on.

### GPU Node Pools

GPU nodes are tainted so that only GPU-requesting pods land on them:

```yaml
# Node taint (set by cloud provider or cluster admin)
taints:
  - key: nvidia.com/gpu
    effect: NoSchedule

# Pod toleration + resource request
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
resources:
  limits:
    nvidia.com/gpu: "1"   # request exactly 1 GPU
```

For multi-GPU pods (tensor parallel vLLM, TP=4):

```yaml
resources:
  limits:
    nvidia.com/gpu: "4"
```

Kubernetes will only schedule this pod to a node with 4 free GPUs.

### The Readiness Probe Problem

vLLM takes 60–120 seconds to load a 70B model. During that window, sending it traffic will
result in errors or OOMs. The readiness probe gates traffic:

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 90    # wait 90s before first check
  periodSeconds: 10
  failureThreshold: 6        # allow 60s of failure before declaring unready
```

`[COMMON TRAP]` — **Setting `initialDelaySeconds` too low**: if the probe fires before the model
finishes loading, the pod is immediately marked unready, Kubernetes may restart it, and you enter
a crash loop. Set `initialDelaySeconds` to your measured cold-start time + 20% margin. For a 70B
FP16 model on 8×H100, that is typically 180–240 seconds.

### Horizontal Pod Autoscaler on Custom Metrics

Standard HPA scales on CPU or memory. For LLM serving you want to scale on:

- **Queue depth** — `vllm:num_requests_waiting`
- **GPU utilization** — scraped from DCGM exporter
- **p99 TTFT** — from vLLM's Prometheus histogram

This requires installing the Prometheus Adapter to bridge Prometheus metrics into the Kubernetes
Custom Metrics API:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-worker
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_requests_waiting    # from Prometheus adapter
        target:
          type: AverageValue
          averageValue: "5"              # scale out when any pod has >5 waiting
```

---

## 19.3 KubeRay: Ray on Kubernetes

`[DEEP DIVE]`

KubeRay is the Kubernetes operator that manages Ray clusters. Ray is the distributed computing
framework that vLLM uses for its tensor-parallel multi-GPU execution and for distributed serving
via RayServe. KubeRay introduces two Custom Resource Definitions (CRDs):

```
  RayCluster     — defines a Ray cluster: head node + worker node groups
  RayService     — wraps a RayCluster with zero-downtime serving semantics
```

### The RayCluster Topology

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Kubernetes Cluster                                              │
  │                                                                  │
  │  ┌─────────────────────────────────────────────────────────┐    │
  │  │  RayCluster  (managed by KubeRay operator)              │    │
  │  │                                                         │    │
  │  │  ┌──────────────────┐                                   │    │
  │  │  │  Head Pod         │  Ray GCS (Global Control Store)  │    │
  │  │  │  CPU: 4           │  Dashboard: :8265                │    │
  │  │  │  RAM: 16 GB       │  Serves as autoscaler monitor    │    │
  │  │  └──────────┬────────┘                                  │    │
  │  │             │  Ray cluster bus (gRPC)                   │    │
  │  │    ┌────────┴───────────────────────────┐               │    │
  │  │    │                                    │               │    │
  │  │  ┌─▼──────────────┐  ┌─────────────────▼──┐            │    │
  │  │  │  Worker Pod 0   │  │  Worker Pod 1       │  ...      │    │
  │  │  │  GPU: 1 (H100)  │  │  GPU: 1 (H100)      │            │    │
  │  │  │  RAM: 80 GB     │  │  RAM: 80 GB         │            │    │
  │  │  │  vLLM replica   │  │  vLLM replica       │            │    │
  │  │  └─────────────────┘  └─────────────────────┘            │    │
  │  └─────────────────────────────────────────────────────────┘    │
  │                                                                  │
  │  KubeRay Operator (runs in kube-system or dedicated namespace)   │
  │    Watches RayCluster/RayService CRDs                           │
  │    Creates/deletes worker Pods as needed                        │
  │    Reconciles cluster state continuously                        │
  └──────────────────────────────────────────────────────────────────┘
```

### RayCluster Manifest

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: vllm-cluster
  namespace: inference
spec:
  rayVersion: "2.35.0"

  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
          - name: ray-head
            image: rayproject/ray-ml:2.35.0-gpu
            resources:
              requests: { cpu: "4", memory: "16Gi" }
              limits:   { cpu: "4", memory: "16Gi" }

  workerGroupSpecs:
    - groupName: vllm-workers
      replicas: 2
      minReplicas: 1
      maxReplicas: 8
      rayStartParams: {}
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
                - --model=meta-llama/Llama-3-8B-Instruct
                - --max-num-seqs=64
                - --gpu-memory-utilization=0.90
              resources:
                requests: { cpu: "8", memory: "80Gi", nvidia.com/gpu: "1" }
                limits:   { cpu: "8", memory: "80Gi", nvidia.com/gpu: "1" }
              readinessProbe:
                httpGet: { path: /health, port: 8000 }
                initialDelaySeconds: 120
                periodSeconds: 10
```

### RayService: Zero-Downtime Updates

`RayService` wraps a `RayCluster` with a canary upgrade strategy. When you push a new model
version or vLLM upgrade:

```
  Old RayCluster (v1): serving 100% traffic
       ↓  new RayService spec applied
  New RayCluster (v2): spinning up
       ↓  readiness check passes
  Traffic shifts: v1=50%, v2=50%
       ↓  health monitored for upgradeWaitTimeSeconds
  Traffic shifts: v1=0%, v2=100%
       ↓
  Old RayCluster (v1): terminated
```

This is the only zero-downtime upgrade path for stateful vLLM workers. Without it, any model
update requires a hard restart and a cold-start gap.

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: vllm-service
spec:
  serveConfigV2: |
    applications:
      - name: vllm-app
        route_prefix: /
        import_path: vllm_serve:deployment
        deployments:
          - name: VLLMDeployment
            num_replicas: 2
            ray_actor_options:
              num_gpus: 1
  upgradeStrategy:
    type: NewCluster
    newClusterAdditionalWaitTime: 60s
  rayClusterConfig: ...  # RayCluster spec as above
```

### Autoscaling on Queue Depth

KubeRay's autoscaler reads Ray's internal queue metrics. For vLLM, the most meaningful trigger
is the number of waiting requests. You configure this via Ray Serve's autoscaling policy:

```yaml
deployments:
  - name: VLLMDeployment
    autoscaling_config:
      min_replicas: 1
      max_replicas: 8
      target_num_ongoing_requests_per_replica: 10   # scale out above 10 in-flight/replica
      upscale_delay_s: 30
      downscale_delay_s: 300    # wait 5 min before scaling in (avoid thrash)
```

`[COMMON TRAP]` — **Setting `downscale_delay_s` too low**: scaling in after 30 seconds means
any quiet period triggers a cold-start. LLM cold starts take 1–3 minutes. If your traffic is
bursty (many short quiet periods), you will pay cold-start latency repeatedly. Set
`downscale_delay_s` to at least 3× your measured cold-start time.

---

## 19.4 llm.d: Kubernetes-Native LLM Serving

`[DEEP DIVE]`

llm.d (pronounced "LLM-dee") is an emerging CNCF-aligned standard for declaring LLM serving
workloads as first-class Kubernetes objects. Where KubeRay wraps existing Ray infrastructure,
llm.d is designed from scratch with LLM-specific semantics at the CRD level.

The core design principle: LLM serving should be *declarative*. You describe the outcome
(model, hardware class, SLA targets, scaling policy) and the controller figures out how to
achieve it using whatever engine is available.

### The LLMDeployment CRD

```yaml
apiVersion: inference.llm.d/v1alpha1
kind: LLMDeployment
metadata:
  name: llama3-8b-prod
  namespace: inference
spec:
  model:
    name: meta-llama/Llama-3-8B-Instruct
    revision: main
    format: safetensors           # or gguf, awq, gptq

  hardware:
    class: h100-sxm               # maps to a Kubernetes node label
    gpuCount: 1
    memoryGb: 80

  engine:
    type: vllm                    # or llama-cpp, sglang, trt-llm
    version: "0.6.0"
    config:
      max_num_seqs: 64
      gpu_memory_utilization: 0.90
      enable_prefix_caching: true
      enable_chunked_prefill: true

  scaling:
    minReplicas: 1
    maxReplicas: 8
    metric: queue_depth           # or ttft_p99, gpu_utilization
    targetValue: "5"              # scale out when queue_depth > 5

  sla:
    ttft_p99_ms: 500
    itl_p99_ms:  50

  disaggregation:
    enabled: true
    prefillClass: h100-sxm        # prefill workers use this hardware class
    decodeClass:  h100-nvl        # decode workers use this hardware class
    prefillReplicas: 2
    decodeReplicas: 8
```

### Disaggregation as a First-Class Concept

The `disaggregation` stanza is the key advance over plain KubeRay. llm.d's scheduler
understands that a single `LLMDeployment` may require two distinct pod types with different
hardware requirements, and it creates and manages both pools with a single manifest. Under the
hood, it generates:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  llm.d controller output for the manifest above                 │
  │                                                                 │
  │  Deployment: llama3-8b-prefill   (2 replicas, h100-sxm nodes)  │
  │    - kv_role: kv_producer                                       │
  │    - max_num_seqs: 8                                            │
  │    - Connected to kv-transfer service                           │
  │                                                                 │
  │  Deployment: llama3-8b-decode    (8 replicas, h100-nvl nodes)  │
  │    - kv_role: kv_consumer                                       │
  │    - max_num_seqs: 128                                          │
  │    - Connected to kv-transfer service                           │
  │                                                                 │
  │  Service: llama3-8b-gateway                                     │
  │    - Routes prefill requests → prefill pool                     │
  │    - Routes decode traffic  → decode pool                       │
  │    - Tracks conversation affinity (same decode worker)          │
  └─────────────────────────────────────────────────────────────────┘
```

### Gateway API Integration

llm.d integrates with the Kubernetes Gateway API to route LLM traffic by model version,
priority class, and user tier:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-routing
spec:
  rules:
    - matches:
        - headers:
            - name: X-Priority
              value: "high"
      backendRefs:
        - name: llama3-8b-high-tier   # premium deployment, more GPUs
          port: 8000
    - matches:
        - path:
            value: /v1/chat/completions
      backendRefs:
        - name: llama3-8b-standard
          port: 8000
```

### llm.d vs. KubeRay: When to Use Which

```
  ┌─────────────────┬────────────────────────────┬──────────────────────────┐
  │                 │  KubeRay                   │  llm.d                   │
  ├─────────────────┼────────────────────────────┼──────────────────────────┤
  │ Maturity        │ Production-ready (2023+)   │ Early adopter (2024+)    │
  │ Engine support  │ Ray-based (vLLM, SGLang)   │ Engine-agnostic          │
  │ Disaggregation  │ Manual (two RayClusters)   │ First-class in CRD       │
  │ Gateway         │ Ray Serve routes           │ k8s Gateway API native   │
  │ Learning curve  │ Moderate (Ray + k8s)       │ Low (pure k8s)           │
  │ Best for        │ Existing Ray ecosystem     │ New greenfield deployments│
  └─────────────────┴────────────────────────────┴──────────────────────────┘
```

---

## 19.5 Heterogeneous GPU Pools

`[DEEP DIVE]`

Production clusters often have mixed GPU generations. A reasonable 2025 configuration:

```
  Node pool A: 4 nodes × 8× H100 SXM   →  prefill workers (compute-dense)
  Node pool B: 8 nodes × 8× H100 NVL   →  decode workers (bandwidth-wide)
  Node pool C: 2 nodes × 8× A100       →  dev/test workloads
  Node pool D: 4 nodes × 8× RTX 4090   →  batch/offline workloads
```

Kubernetes node labels and pod node affinity route each workload to the right pool:

```yaml
# Node pool A label (set by cluster admin)
labels:
  gpu-class: h100-sxm
  inference-role: prefill

# Pod affinity for prefill workers
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: inference-role
              operator: In
              values: ["prefill"]
```

`[COMMON TRAP]` — **Forgetting to isolate decode workers from the compute pool**: if your
scheduler is allowed to place decode workers on H100 SXM nodes (when those nodes have free
GPU slots), you end up with mixed prefill/decode on compute-dense hardware and lose the
bandwidth advantage you built the decode pool for. Always use `requiredDuringScheduling`
affinity, not `preferredDuringScheduling`.

---

## 19.6 Prometheus + Grafana in Kubernetes

`[FOUNDATIONAL]`

Chapter 16 covered what vLLM's `/metrics` endpoint exposes. In Kubernetes, you need three
additional pieces to get those metrics into Grafana:

```
  vLLM Pod  →  ServiceMonitor (Prometheus Operator CRD)
           →  Prometheus (scrapes based on ServiceMonitor labels)
           →  Grafana (visualises via Prometheus data source)
```

### ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm-metrics
  namespace: inference
  labels:
    release: prometheus   # must match Prometheus operator's serviceMonitorSelector
spec:
  selector:
    matchLabels:
      app: vllm-worker    # select pods with this label
  endpoints:
    - port: metrics       # named port in the vLLM Service
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s
```

### Alert: KV Cache Saturation

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: vllm-kv-alerts
  namespace: inference
spec:
  groups:
    - name: vllm.kv_cache
      rules:
        - alert: KVCacheSaturation
          expr: |
            vllm:gpu_cache_usage_perc > 0.95
          for: 2m
          labels:
            severity: critical
          annotations:
            summary: "KV cache > 95% — OOM risk"
            description: "Pod {{ $labels.pod }} KV cache at {{ $value | humanizePercentage }}"

        - alert: HighWaitingRequests
          expr: |
            vllm:num_requests_waiting > 20
          for: 1m
          labels:
            severity: warning
          annotations:
            summary: "Queue depth > 20 — consider scaling out"
```

---

## 19.7 Graceful Drain and Rollout

`[DEEP DIVE]`

When Kubernetes terminates a vLLM pod (scaling in, rolling update, or node drain), in-flight
requests must complete before the pod dies. The correct sequence:

```
  1. Kubernetes sends SIGTERM to the pod
  2. Pod stops accepting new requests (readiness probe → NotReady)
  3. In-flight requests complete (or time out after terminationGracePeriodSeconds)
  4. Kubernetes sends SIGKILL after grace period
```

Configure in your Deployment:

```yaml
spec:
  template:
    spec:
      terminationGracePeriodSeconds: 120   # max 120s for in-flight requests to finish
      containers:
        - name: vllm
          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh", "-c", "sleep 5"]   # brief delay for load balancer to drain
```

vLLM's OpenAI-compatible server handles SIGTERM by draining the request queue. For very long
reasoning traces (Chapter 24), you may need `terminationGracePeriodSeconds: 600` to avoid
cutting off a 500-second generation mid-flight.

---

## 19.8 The Running Case Study: 50K Users on KubeRay

`[DEEP DIVE]`

Applying KubeRay to the production scenario from Chapter 1:

```
  Traffic: 50K concurrent users  →  ~3,000 req/s peak
  Workload: 70% short (200 tok), 30% RAG (32K tok)
  SLA: TTFT p99 < 500ms, ITL p99 < 50ms

  ┌──────────────────────────────────────────────────────────────────┐
  │  KubeRay Cluster Layout                                          │
  │                                                                  │
  │  Ray Head Pod (1× CPU node)                                      │
  │    GCS, autoscaler, dashboard                                    │
  │                                                                  │
  │  Prefill Worker Group (2–8 pods, H100 SXM, autoscale on p99)    │
  │    RayService → vLLM kv_producer                                 │
  │    Scales: queue_depth > 5                                       │
  │                                                                  │
  │  Decode Worker Group (8–32 pods, H100 NVL, autoscale on ITL)    │
  │    RayService → vLLM kv_consumer                                 │
  │    Scales: active_sequences > 100/pod                            │
  │                                                                  │
  │  Semantic Cache (Chapter 30): Redis cluster, 73% hit rate        │
  │    Intercepts before prefill: saves 73% of prefill GPU-time      │
  │                                                                  │
  │  Prometheus + Grafana: scraping all pods via ServiceMonitor      │
  │  PagerDuty alerting: KVCacheSaturation, HighWaitingRequests      │
  └──────────────────────────────────────────────────────────────────┘

  Steady-state cluster size:
    Prefill:  4 pods × 1 H100 SXM  (autoscale 2–8)
    Decode:  16 pods × 1 H100 NVL  (autoscale 8–32)
    Cost:    20 × $3.00/hr = $60/hr = $1,440/day (vs. $40K/day at launch)
```

---

## 19.9 Autoscaling Arithmetic

The right autoscaling threshold is not a guess — it is a calculation. Here is the full derivation for a decode worker HPA:

**Target: keep p99 inter-token latency ≤ 50 ms at peak load.**

Step 1 — measure peak active sequences per decode pod at acceptable latency:

```
Load test result: p99 ITL ≤ 50ms when active_sequences ≤ 80/pod
(each H100 NVL running LLaMA 3 70B at tensor_parallel_size=1)
```

Step 2 — compute required pods at peak:

```
Peak demand: 3,000 req/s × avg decode length 150 tokens × 50ms ITL
= 3,000 × 150 × 0.05 = 22,500 token-slots needed
Per-pod capacity: 80 active_sequences
Required pods: ceil(22,500 / 80) = 282
```

This first estimate is incorrect — not all 3,000 req/s are in the decode phase simultaneously. With TTFT of ~200ms and decode length of 150 tokens at 50ms ITL, each request occupies a decode slot for 150 × 0.05 = 7.5 seconds. So concurrent decode requests:

```
concurrent_decode = arrival_rate × decode_time
= 3,000/s × 7.5s = 22,500

... still 282 pods. This IS correct for sustained peak.
```

With the semantic cache absorbing 73% of requests (Chapter 30), effective arrival rate = 810 req/s:

```
concurrent_decode = 810 × 7.5 = 6,075
Required pods = ceil(6,075 / 80) = 76 pods
```

Step 3 — set HPA thresholds with headroom:

```yaml
# hpa-decode.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-decode-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-decode
  minReplicas: 8
  maxReplicas: 96            # 20% headroom above 76
  metrics:
  - type: External
    external:
      metric:
        name: vllm_active_sequences_per_pod
      target:
        type: AverageValue
        averageValue: "64"   # 80% of capacity — scale before saturation
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30   # react fast
      policies:
      - type: Pods
        value: 4
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300  # drain slowly
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
```

**Cold-start penalty.** Each new vLLM pod takes ~90 seconds to load a 70B model. Setting `scaleUp.stabilizationWindowSeconds: 30` means the HPA decides to add pods quickly, but those pods only become ready 90 seconds later. Design for this by keeping `minReplicas` at a floor that handles expected baseline load without scaling events.

---

## 19.10 Multi-Cluster and Multi-Region Serving

For global deployments, a single Kubernetes cluster is insufficient. The production pattern:

```
                     ┌─────────────────────────────────┐
                     │     Global Load Balancer         │
                     │  (Cloudflare, AWS Global Accel)  │
                     └─────────────┬───────────────────┘
                                   │  geo-route by latency
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
      │  US-East-1   │   │  EU-West-1   │   │  AP-South-1  │
      │  KubeRay     │   │  KubeRay     │   │  KubeRay     │
      │  8 H100 pods │   │  4 H100 pods │   │  4 H100 pods │
      └──────────────┘   └──────────────┘   └──────────────┘
              │                    │                     │
              └────────────────────┼────────────────────┘
                                   │ overflow routing
                              ┌────▼────┐
                              │ Spot    │
                              │ Burst   │
                              │ Cluster │
                              └─────────┘
```

**KV cache is not shared across regions.** Each cluster prefills its own requests. A request that overflows from US-East to EU-West will incur a full re-prefill. This is acceptable for stateless API calls but problematic for long multi-turn sessions.

**Session affinity** routes multi-turn sessions to the same cluster using a consistent hash on `session_id`. This keeps the KV cache warm for long conversations:

```yaml
# Istio VirtualService — session affinity
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: vllm-session-affinity
spec:
  http:
  - match:
    - headers:
        x-session-id:
          regex: ".*"
    route:
    - destination:
        host: vllm-us-east
      weight: 0
    headers:
      request:
        set:
          x-cluster-override: "us-east"
```

---

## 19.11 Cost Visibility in Kubernetes

The GPU cost is the dominant line item. Track it at pod level:

```yaml
# Add cost labels to every vLLM pod
metadata:
  labels:
    cost-center: "inference-prod"
    model: "llama3-70b"
    role: "decode"   # or "prefill"
```

Then query Prometheus for cost attribution:

```promql
# GPU-hours consumed per model per day
sum by (model, role) (
  increase(vllm_gpu_utilization_seconds_total[24h])
) / 3600
```

With GPU cost at $3.00/hr (H100 on-demand), the daily cost per model:

```promql
sum by (model) (
  increase(vllm_gpu_utilization_seconds_total[24h])
) / 3600 * 3.00
```

Alert when daily GPU spend exceeds budget:

```yaml
# Prometheus alerting rule
- alert: DailyGPUSpendExceeded
  expr: |
    sum(increase(vllm_gpu_utilization_seconds_total[24h])) / 3600 * 3.00 > 2000
  for: 5m
  annotations:
    summary: "Daily GPU spend exceeds $2,000"
```

---

## 19.12 Self-Check Questions

1. **[FOUNDATIONAL]** A vLLM pod takes 120 seconds to be ready after scheduling. Your HPA `scaleUp.stabilizationWindowSeconds` is 60 seconds. Draw the timeline from when traffic spikes to when new capacity is actually serving requests. What is the minimum response time to a traffic spike?

2. **[SYSTEMS]** Your decode HPA is set to target `active_sequences = 60` per pod. Current load requires 500 concurrent decode slots. The semantic cache has a 60% hit rate and absorbs requests before they reach the decoder. How many decoder pods are needed? What changes if the cache hit rate drops to 20%?

3. **[DEEP DIVE]** Explain why the `terminationGracePeriodSeconds` for a vLLM decode pod should be set to `max_sequence_length × inter_token_latency` rather than a fixed 30 seconds.

4. **[APPLIED]** You have a multi-region deployment with US, EU, and AP clusters. A user in Singapore (nearest: AP) sends a 10K-token prompt. AP is at 95% capacity. The request overflows to US. What is the additional latency from re-prefilling vs. the inter-region network latency (~180ms RTT)? When does overflow routing make sense?

---

## Summary

Kubernetes turns a working vLLM instance into a production service. The path from "runs on my
workstation" to "handles 50K concurrent users" requires: GPU node pools with proper taints and
tolerations, readiness probes calibrated to model cold-start time, autoscaling on LLM-specific
metrics (queue depth, p99 latency), and zero-downtime rolling updates.

KubeRay provides the distributed Ray execution layer that vLLM's tensor parallelism and multi-GPU
features depend on, plus the RayService abstraction for canary updates. llm.d goes further:
it makes disaggregated prefill/decode, priority routing, and hardware-class affinity
first-class concepts in a single declarative manifest, removing the operational complexity of
managing two separate Ray clusters for a single model deployment.

The LLM inference field is moving toward Kubernetes-native orchestration precisely because every
production concern — uptime, cost, scaling, traffic routing, multi-tenancy — already has a
battle-tested solution in the Kubernetes ecosystem. llm.d is still maturing, but the direction
is clear: serving a model should eventually be as simple as applying a YAML manifest and letting
the controller handle everything below.

---

## Key Terms

- **KubeRay** — Kubernetes operator that manages Ray clusters via `RayCluster` and `RayService`
  CRDs, enabling distributed vLLM workloads on Kubernetes.

- **llm.d** — Kubernetes CRD standard for declarative LLM serving; `LLMDeployment` expresses
  model, hardware, SLA, and disaggregation requirements in a single manifest.

- **RayService** — KubeRay CRD that wraps a RayCluster with zero-downtime upgrade semantics
  and traffic shifting.

- **ServiceMonitor** — Prometheus Operator CRD that tells Prometheus which pods to scrape.
- **terminationGracePeriodSeconds** — Kubernetes setting that controls how long a pod has to
  drain in-flight requests before being forcibly killed.

- **HPA (Horizontal Pod Autoscaler)** — Kubernetes controller that adjusts replica count based
  on observed metrics; for LLM serving, driven by custom metrics via the Prometheus Adapter.

- **Node affinity** — Kubernetes scheduling constraint that pins pods to nodes matching
  specified labels; used to route prefill workers to compute-dense GPUs and decode workers
  to bandwidth-wide GPUs.

---

*Next: Chapter 20 — Cost Engineering: $/Million Tokens*


---

## Self-Check Questions

1. An HPA is configured with `targetAverageValue: 20` for the custom metric `vllm_active_sequences_per_pod`. Current pods are at [18, 22, 19, 25]. Does the HPA scale up? Show the arithmetic using the stabilization formula. *(Section 19.2)*

2. A vLLM pod takes 45 s to load the model and another 30 s to warm up CUDA graphs. Traffic is routed immediately after the container's readiness probe passes at T=20 s. What goes wrong, and how do you fix the probe configuration? *(Section 19.3)*

3. You configure `preemptionPolicy: Never` for your vLLM pods. A spot instance node is reclaimed. Describe what Kubernetes does with the pod and how you minimize request loss. *(Section 19.5)*

4. KubeRay scales the Ray cluster underlying vLLM. What additional metric would you expose to the HPA beyond `vllm_active_sequences` to handle burst traffic with cold-start penalty? *(Section 19.9)*

5. Write the PromQL cost attribution expression that converts GPU-hours to dollars for a vLLM deployment across multiple model namespaces. *(Section 19.11)*


---

## Worked Solutions

### Question 1 (Section 19.12)
**Setup:** Pod ready-time = 120 s, `scaleUp.stabilizationWindowSeconds = 60 s`.

**Timeline from traffic spike:**

```
T=0    Traffic spike detected. HPA sees target metric exceeded.
T=60   Stabilization window expires. HPA issues scale-up decision.
       (HPA waited 60 s to confirm the spike is sustained, not a blip.)
T=60   Kubernetes schedules new pods.
T=60–90  Scheduler finds nodes, pulls container image (if cached: ~5 s; cold: ~30 s).
T=90   Container starts. Model loading begins.
T=90–210 Model loading takes 120 s (120 s ready-time).
T=210  Pod reports Ready. Traffic routed to new pod.
```

**Minimum response time to a traffic spike:**
```
60 s (stabilization) + 120 s (pod ready) = 180 s minimum
```
Plus image pull time (5–30 s) and Kubernetes scheduling (~5 s): **realistic minimum 190–215 seconds** from spike to new capacity.

**Implication:** Any workload that spikes and subsides within 3 minutes will not benefit from HPA scale-up. Pre-warm pods or use predictive scaling for known traffic patterns.

---

### Question 2 (Section 19.12)
**Setup:** Decode HPA target = 60 active sequences/pod. Need 500 slots. Cache hit rate = 60%.

**With 60% cache hit rate:**
Requests that hit the semantic cache never reach the decoder. Effective decoder demand:
```
decoder_requests = 500 × (1 − 0.60) = 200 concurrent slots needed
```

Pods needed:
```
pods = ceil(200 / 60) = ceil(3.33) = 4 decoder pods
```

**With 20% cache hit rate:**
```
decoder_requests = 500 × (1 − 0.20) = 400 slots
pods = ceil(400 / 60) = ceil(6.67) = 7 decoder pods
```

**Change from 60% → 20% hit rate:** 3 additional pods required. At $3.20/GPU-hr, that's an additional $230/day cost just from cache miss rate degradation. This quantifies the financial value of maintaining high cache hit rates.

---

### Question 3 (Section 19.12)
**Why `terminationGracePeriodSeconds` = `max_sequence_length × inter_token_latency`:**

When Kubernetes signals a pod to terminate (SIGTERM), vLLM must finish all in-flight requests before shutting down. An in-flight request may be in mid-generation — it has already received its prompt (prefill done), and its KV cache is allocated.

If the pod is killed before generation completes:

- The user receives a truncated or empty response.
- The KV cache (up to `max_sequence_length` blocks) is lost without producing a useful output.

**The correct grace period:**
```
grace = max_sequence_length × ITL
```
Example: `max_sequence_length=2048`, ITL=20 ms → grace = 2048 × 0.02 = 40.96 s ≈ 41 s.

This ensures the worst-case in-flight request (one that just started generating at the moment of SIGTERM) has time to complete before the pod is force-killed.

**Why 30 s is wrong:** A 30 s fixed grace period accommodates sequences of at most 30/0.02 = 1,500 tokens. Requests with `max_tokens=2048` will be cut off. The correct value must be derived from the actual model performance characteristics, not a fixed constant.

---

### Question 4 (Section 19.12)
**Setup:** US, EU, AP clusters. User in Singapore, AP at 95%, overflow to US. AP→US RTT=180 ms.

**Cost of re-prefilling in US vs. AP network latency:**

Assuming a 10K-token prompt and a prefill throughput of ~15,000 tok/s per pod:
```
prefill_time = 10,000 / 15,000 ≈ 0.667 s = 667 ms (on US prefill GPU)
```

**Additional latency from overflow:**
- Network transmission of 10K tokens (JSON, ~40 KB at ~4 bytes/token): 40 KB over ~100 Mbps connection = 3.2 ms (negligible).
- AP → US RTT: 180 ms (one way; round-trip adds another 180 ms for response).
- Re-prefill in US: 667 ms.

**Total overflow penalty ≈ 667 + 180 ms RTT = 847 ms additional TTFT** vs. serving locally at AP (~667 ms → TTFT = 667 ms if AP not overloaded).

**When does overflow routing make sense?**
When AP is at 95% capacity, the queuing delay at AP (waiting for a slot) exceeds the overflow cost of 847 ms. If AP queue depth means requests wait 2 s before prefill begins, routing to US costs 847 ms — still faster by >1 s. For short prompts where prefill is <100 ms, overflow rarely makes sense (the 180 ms network latency alone exceeds local latency).

---

### Question 5 (End-of-chapter)
**HPA with `targetAverageValue: 20` for `vllm_active_sequences_per_pod`.**
Current pods: [18, 22, 19, 25]. Total = 84, 4 pods.

**HPA desired replicas formula:**
```
desiredReplicas = ceil(currentReplicas × currentMetricValue / desiredMetricValue)
currentMetricValue = average = (18+22+19+25) / 4 = 84/4 = 21
desiredReplicas = ceil(4 × 21 / 20) = ceil(4.2) = 5
```

Yes, the HPA **scales up** to 5 pods (from 4). The average exceeds the target, triggering a scale-out.

---

### Question 6
**Pod readiness probe at T=20 s but model needs 75 s total (45 s load + 30 s CUDA warm-up):**

The readiness probe passes at T=20 s (perhaps an HTTP health check returns 200 OK after the HTTP server starts, before the model is fully loaded). Kubernetes immediately routes traffic to the pod.

**What goes wrong:**
- Requests arrive between T=20 and T=75 s. The model is not yet loaded or CUDA graphs not yet captured.
- vLLM will either queue requests (if the engine is partially initialized) or return 500 errors.
- CUDA graph capture during this period receives live traffic, potentially causing OOM or capture failures if the batch sizes vary during warm-up.

**Fix:** Configure a startup probe that tests the actual inference endpoint:
```yaml
startupProbe:
  httpGet:
    path: /v1/models    # returns models only after engine is ready
    port: 8000
  failureThreshold: 30
  periodSeconds: 5      # 30 × 5 = 150 s startup window
```
Set the readinessProbe to probe `/health` after the startupProbe succeeds, ensuring the model is fully loaded and CUDA graphs captured before traffic is admitted.

---

### Question 7
**`preemptionPolicy: Never` + spot instance reclaimed.**

Kubernetes **cannot preempt** higher-priority pods to reschedule the vLLM pod (because `preemptionPolicy: Never` means the pod will not evict others). When the spot node is reclaimed:

1. Kubernetes taints the node with `node.kubernetes.io/not-ready`.
2. The vLLM pod receives SIGTERM (if `terminationGracePeriodSeconds` allows, it drains current requests; otherwise SIGKILL).
3. Kubernetes adds the pod to the pending queue. It cannot evict other pods to find space.
4. The pod stays pending until a new node is available (either spot price drops, or cluster autoscaler provisions a new node).

**Minimize request loss:**
- Set `terminationGracePeriodSeconds` = max_sequence_length × ITL to drain in-flight requests.
- Use PodDisruptionBudget to ensure at least N−1 replicas remain serving during node eviction.
- Enable vLLM's graceful shutdown: `--graceful-shutdown-timeout 60` so in-flight sequences complete before the pod exits.
- Use multi-AZ placement with `topologySpreadConstraints` so a single spot instance reclaim doesn't take down the only active replica.

---

### Question 8
**Additional metric beyond `vllm_active_sequences` for burst traffic with cold-start penalty:**

`vllm_pending_request_queue_depth` (or `vllm:num_requests_waiting`) — the number of requests waiting in the scheduler queue before prefill begins.

**Why this is better for cold-start-aware scaling:**
Active sequences only reflect currently running requests — they don't capture incoming demand that hasn't been admitted yet. A growing queue signals that demand exceeds current capacity *before* the pod's metric becomes saturated.

Configure HPA to scale when:
```
vllm_pending_request_queue_depth > 10  (for 30 s)
```
This fires the scale-up signal earlier (when the queue starts building) rather than after active sequences are saturated, giving the 75 s cold-start time to complete before the queue grows further.

---

### Question 9
**PromQL cost attribution for GPU-hours to dollars:**

```promql
sum by (namespace, model) (
  rate(vllm:gpu_seconds_total[1h]) * on(namespace) group_left()
  kube_pod_labels{label_cost_per_gpu_hour="3.20"}
) * 3.20
```

Or more practically, using node-level GPU metrics:
```promql
sum by (namespace) (
  rate(DCGM_FI_DEV_GPU_UTIL[1h]) / 100
  * on(node) group_left()
  kube_node_labels{label_gpu_cost_per_hour="3.20"}
) * 3.20
```

This gives dollars per hour per namespace. Multiply by 730 for monthly cost:
```promql
sum by (namespace) (
  rate(vllm:gpu_seconds_total[730h]) * 3.20 / 3600
)
```

